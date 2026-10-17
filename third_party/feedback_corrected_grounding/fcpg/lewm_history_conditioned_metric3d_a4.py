"""Causal action-history conditioning for the projectable A4 LeWM.

The dense action-prefix model predicts from one current observation and a
planned future action prefix. MetaWorld's mocap controller has tracking lag,
so the same current pose and future action can evolve differently depending
on the commands executed immediately before the observation. This module
adds that missing causal input without changing the frozen observation model,
base dynamics, or metric-3D readout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix import ActionPrefixDynamics, dense_prefix_mse
from fcpg.lewm_action_prefix_joint_a4 import (
    ACTION_DIM,
    ANCHOR_COUNT,
    LATENT_DIM,
    STATE_SLOTS,
    gripper_aperture_loss,
)
from fcpg.lewm_camera_action_joint_a4 import (
    transform_world_actions_to_camera,
)
from fcpg.lewm_metric3d_joint_a4 import (
    METRIC3D_CHECKPOINT_TYPE,
    METRIC3D_WORLD_PROTOCOL,
    PROJECTED_APERTURE_LOSS_WEIGHT,
    PROJECTED_METRIC_LOSS_WEIGHT,
    WORLD_LOSS_WEIGHT,
    WORLD_SCALE_M,
    Metric3DCameraActionLeWM,
    load_metric3d_checkpoint,
)


HISTORY_CONDITIONED_PROTOCOL = "a4_metric3d_world_action_history_v1"
HISTORY_CONDITIONED_CHECKPOINT_TYPE = "a4_metric3d_world_action_history"
HISTORY_ACTION_STEPS = 2
HISTORY_ADAPTER_HIDDEN_DIM = 256


@dataclass(frozen=True)
class HistoryConditionedMetric3DOutput:
    current_state: torch.Tensor
    target_state: torch.Tensor
    predicted_state: torch.Tensor
    base_predicted_state: torch.Tensor
    predicted_q_px: torch.Tensor
    base_predicted_q_px: torch.Tensor
    predicted_world_xyz: torch.Tensor
    base_predicted_world_xyz: torch.Tensor
    latent_loss: torch.Tensor
    world_loss: torch.Tensor
    metric_loss: torch.Tensor
    gripper_aperture_loss: torch.Tensor
    loss: torch.Tensor


class CausalActionHistoryConditioner(nn.Module):
    """Inject two previous action chunks into frozen prefix conditions."""

    def __init__(
        self,
        base_dynamics: ActionPrefixDynamics,
        *,
        hidden_dim: int = HISTORY_ADAPTER_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if not isinstance(base_dynamics, ActionPrefixDynamics):
            raise TypeError("history conditioner requires ActionPrefixDynamics")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.base_dynamics = base_dynamics
        for parameter in self.base_dynamics.parameters():
            parameter.requires_grad_(False)
        input_dim = LATENT_DIM + HISTORY_ACTION_STEPS * ACTION_DIM
        self.condition_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), LATENT_DIM),
        )
        # The first forward pass exactly matches the frozen parent dynamics.
        nn.init.zeros_(self.condition_adapter[-1].weight)
        nn.init.zeros_(self.condition_adapter[-1].bias)

    @property
    def action_dim(self) -> int:
        return int(self.base_dynamics.action_dim)

    def _validate_history(
        self,
        current_state: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> None:
        batch_size = int(current_state.shape[0])
        if history_action.shape != (batch_size, HISTORY_ACTION_STEPS, ACTION_DIM):
            raise ValueError("history_action must have shape [B,2,20]")
        if future_action.ndim != 3 or future_action.shape[0] != batch_size:
            raise ValueError("future_action must have shape [B,H,20]")
        if history_action.dtype != current_state.dtype:
            raise TypeError("history_action and current_state must share dtype")
        if history_action.device != current_state.device:
            raise ValueError("history_action and current_state must share device")
        if not bool(torch.isfinite(history_action).all()):
            raise ValueError("history_action contains non-finite values")

    def forward(
        self,
        current_state: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
        *,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_history(current_state, history_action, future_action)
        base_condition = self.base_dynamics.encode_prefix(
            current_state,
            future_action,
            task_index=task_index,
        )
        normalized_history = (
            history_action
            - self.base_dynamics.action_mean.to(dtype=history_action.dtype)
        ) / self.base_dynamics.action_std.to(dtype=history_action.dtype)
        flattened_history = normalized_history.reshape(
            current_state.shape[0], HISTORY_ACTION_STEPS * ACTION_DIM
        )
        expanded_history = flattened_history[:, None].expand(
            -1, future_action.shape[1], -1
        )
        correction = self.condition_adapter(
            torch.cat([base_condition, expanded_history], dim=-1)
        )
        adapted_condition = base_condition + correction
        shared_current = current_state[:, None].expand(
            -1,
            future_action.shape[1],
            STATE_SLOTS,
            LATENT_DIM,
        )
        residual = self.base_dynamics.residual_predictor(
            shared_current, adapted_condition
        )
        return shared_current + residual


class HistoryConditionedMetric3DLeWM(nn.Module):
    """Frozen world-v2 model with a trainable causal history conditioner."""

    def __init__(
        self,
        base_model: Metric3DCameraActionLeWM,
        *,
        hidden_dim: int = HISTORY_ADAPTER_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, Metric3DCameraActionLeWM):
            raise TypeError("base_model must be Metric3DCameraActionLeWM")
        if base_model.residual_frame != "world":
            raise ValueError("history conditioning requires world-v2 metric3d")
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.history_dynamics = CausalActionHistoryConditioner(
            self.base_model.dynamics,
            hidden_dim=int(hidden_dim),
        )

    @property
    def frame_encoder(self) -> nn.Module:
        return self.base_model.frame_encoder

    @property
    def state_projector(self) -> nn.Module:
        return self.base_model.state_projector

    @property
    def dynamics(self) -> nn.Module:
        """Expose frozen parent dynamics for no-history diagnostics."""

        return self.base_model.dynamics

    @property
    def metric_readout(self) -> nn.Module:
        return self.base_model.metric_readout

    def train(self, mode: bool = True) -> "HistoryConditionedMetric3DLeWM":
        super().train(mode)
        self.base_model.eval()
        self.history_dynamics.condition_adapter.train(mode)
        return self

    def encode_observations(self, **kwargs: Any) -> torch.Tensor:
        return self.base_model.encode_observations(**kwargs)

    def transform_future_action(
        self, future_action: torch.Tensor, batch: Mapping[str, Any]
    ) -> torch.Tensor:
        return self.base_model.transform_future_action(future_action, batch)

    def decode_3d(self, **kwargs: Any) -> Any:
        return self.base_model.decode_3d(**kwargs)

    @staticmethod
    def _camera_rotation(batch: Mapping[str, Any]) -> torch.Tensor:
        extrinsic = batch.get("camera_extrinsic_cam_from_world")
        if not torch.is_tensor(extrinsic) or extrinsic.ndim != 3:
            raise KeyError("history model requires camera extrinsics")
        if extrinsic.shape[1:] != (4, 4):
            raise ValueError("camera extrinsic must have shape [B,4,4]")
        return extrinsic[:, :3, :3]

    def predict_with_history(
        self,
        *,
        current_state: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
        task_index: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> torch.Tensor:
        transformed_future = self.transform_future_action(future_action, batch)
        transformed_history = transform_world_actions_to_camera(
            history_action,
            self._camera_rotation(batch),
        )
        return self.history_dynamics(
            current_state,
            transformed_history.to(dtype=current_state.dtype),
            transformed_future.to(dtype=current_state.dtype),
            task_index=task_index,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        history_action_override: torch.Tensor | None = None,
    ) -> HistoryConditionedMetric3DOutput:
        required = (
            "history_action",
            "current_keypoints_xyz",
            "future_keypoints_xyz",
            "future_action",
            "future_q_true",
            "future_valid",
            "task_index",
            "camera_intrinsic",
            "camera_extrinsic_cam_from_world",
        )
        for name in required:
            if name not in batch or not torch.is_tensor(batch[name]):
                raise KeyError(f"history-conditioned batch requires {name!r}")
        with torch.no_grad():
            base = self.base_model.camera_action_model(batch)
            base_decoded = self.base_model.decode_3d(
                predicted_state=base.predicted_state,
                current_state=base.current_state,
                current_keypoints_xyz=batch["current_keypoints_xyz"].float(),
                future_action=batch["future_action"].float(),
                task_index=batch["task_index"].long(),
                camera_intrinsic=batch["camera_intrinsic"].float(),
                extrinsic_cam_from_world=batch[
                    "camera_extrinsic_cam_from_world"
                ].float(),
            )
        history_action = (
            batch["history_action"]
            if history_action_override is None
            else history_action_override
        )
        predicted_state = self.predict_with_history(
            current_state=base.current_state,
            history_action=history_action.float(),
            future_action=batch["future_action"].float(),
            task_index=batch["task_index"].long(),
            batch=batch,
        )
        decoded = self.base_model.decode_3d(
            predicted_state=predicted_state,
            current_state=base.current_state,
            current_keypoints_xyz=batch["current_keypoints_xyz"].float(),
            future_action=batch["future_action"].float(),
            task_index=batch["task_index"].long(),
            camera_intrinsic=batch["camera_intrinsic"].float(),
            extrinsic_cam_from_world=batch[
                "camera_extrinsic_cam_from_world"
            ].float(),
        )

        target_state = base.target_state.detach()
        latent_loss = dense_prefix_mse(predicted_state, target_state)
        target_world = batch["future_keypoints_xyz"].float()
        valid = batch["future_valid"].bool()
        weights = valid.float()
        world_squared = (
            (decoded.predicted_world_xyz - target_world) / WORLD_SCALE_M
        ).square().mean(dim=-1)
        world_loss = (world_squared * weights).sum() / weights.sum().clamp_min(1.0)

        target_q = batch["future_q_true"].float()
        width = int(batch["current_rgb"].shape[-1])
        height = int(batch["current_rgb"].shape[-2])
        scale = target_q.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        metric_squared = ((decoded.predicted_q_px - target_q) / scale).square().mean(
            dim=-1
        )
        metric_loss = (metric_squared * weights).sum() / weights.sum().clamp_min(1.0)
        aperture_loss = gripper_aperture_loss(
            decoded.predicted_q_px,
            target_q,
            valid,
            image_size=(width, height),
        )
        loss = (
            latent_loss
            + WORLD_LOSS_WEIGHT * world_loss
            + PROJECTED_METRIC_LOSS_WEIGHT * metric_loss
            + PROJECTED_APERTURE_LOSS_WEIGHT * aperture_loss
        )
        return HistoryConditionedMetric3DOutput(
            current_state=base.current_state,
            target_state=target_state,
            predicted_state=predicted_state,
            base_predicted_state=base.predicted_state,
            predicted_q_px=decoded.predicted_q_px,
            base_predicted_q_px=base_decoded.predicted_q_px,
            predicted_world_xyz=decoded.predicted_world_xyz,
            base_predicted_world_xyz=base_decoded.predicted_world_xyz,
            latent_loss=latent_loss,
            world_loss=world_loss,
            metric_loss=metric_loss,
            gripper_aperture_loss=aperture_loss,
            loss=loss,
        )


def build_history_conditioned_model(
    parent_checkpoint: Mapping[str, Any],
    *,
    hidden_dim: int = HISTORY_ADAPTER_HIDDEN_DIM,
) -> HistoryConditionedMetric3DLeWM:
    if parent_checkpoint.get("checkpoint_type") != METRIC3D_CHECKPOINT_TYPE:
        raise ValueError("history parent must be metric3d")
    if parent_checkpoint.get("protocol") != METRIC3D_WORLD_PROTOCOL:
        raise ValueError("history parent must use world-v2 metric3d")
    if float(parent_checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("history parent must inherit SIGReg=0.02")
    return HistoryConditionedMetric3DLeWM(
        load_metric3d_checkpoint(parent_checkpoint),
        hidden_dim=int(hidden_dim),
    )


def history_conditioned_checkpoint_payload(
    model: HistoryConditionedMetric3DLeWM,
    *,
    step: int,
    seed: int,
    parent_checkpoint: str,
    raw_dataset: str,
    source_cache: str,
) -> dict[str, Any]:
    if not isinstance(model, HistoryConditionedMetric3DLeWM):
        raise TypeError("model must be HistoryConditionedMetric3DLeWM")
    return {
        "checkpoint_type": HISTORY_CONDITIONED_CHECKPOINT_TYPE,
        "protocol": HISTORY_CONDITIONED_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "tasks": list(TASKS),
        "sigreg_weight": SIGREG_WEIGHT,
        "parent_checkpoint": str(parent_checkpoint),
        "raw_dataset": str(raw_dataset),
        "source_cache": str(source_cache),
        "history_action_steps": HISTORY_ACTION_STEPS,
        "hidden_dim": int(model.history_dynamics.condition_adapter[1].out_features),
        "causal_inputs": ["current_observation", "previous_10_raw_actions", "planned_action_prefix"],
        "adapter_state": model.history_dynamics.condition_adapter.state_dict(),
    }


def load_history_conditioned_checkpoint(
    checkpoint: Mapping[str, Any],
) -> HistoryConditionedMetric3DLeWM:
    if checkpoint.get("checkpoint_type") != HISTORY_CONDITIONED_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not history-conditioned metric3d")
    if checkpoint.get("protocol") != HISTORY_CONDITIONED_PROTOCOL:
        raise ValueError("history-conditioned protocol drifted")
    if tuple(checkpoint.get("tasks", ())) != TASKS:
        raise ValueError("history-conditioned task panel drifted")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("history-conditioned checkpoint must use SIGReg=0.02")
    if int(checkpoint.get("history_action_steps", -1)) != HISTORY_ACTION_STEPS:
        raise ValueError("history action contract drifted")
    parent_path = Path(str(checkpoint.get("parent_checkpoint", ""))).expanduser()
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping):
        raise TypeError("history parent checkpoint must contain a mapping")
    model = build_history_conditioned_model(
        parent,
        hidden_dim=int(checkpoint.get("hidden_dim", HISTORY_ADAPTER_HIDDEN_DIM)),
    )
    state = checkpoint.get("adapter_state")
    if not isinstance(state, Mapping):
        raise ValueError("history checkpoint lacks adapter_state")
    model.history_dynamics.condition_adapter.load_state_dict(state, strict=True)
    return model


__all__ = [
    "CausalActionHistoryConditioner",
    "HISTORY_ACTION_STEPS",
    "HISTORY_CONDITIONED_CHECKPOINT_TYPE",
    "HISTORY_CONDITIONED_PROTOCOL",
    "HistoryConditionedMetric3DLeWM",
    "HistoryConditionedMetric3DOutput",
    "build_history_conditioned_model",
    "history_conditioned_checkpoint_payload",
    "load_history_conditioned_checkpoint",
]
