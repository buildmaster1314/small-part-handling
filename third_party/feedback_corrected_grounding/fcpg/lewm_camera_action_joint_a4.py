"""Camera-frame action conditioning for the seven-task A4 LeWM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix_joint_a4 import (
    ACTION_DIM,
    LOSS_PROFILE_UNIFORM,
    JointA4Output,
    JointSevenTaskActionPrefixLeWM,
    load_joint_checkpoint,
)
from fcpg.lewm_action_prefix import ActionPrefixDynamics


CAMERA_ACTION_PROTOCOL = "a4_joint_visual_camera_frame_action_v1"
CAMERA_ACTION_CHECKPOINT_TYPE = "a4_joint_visual_camera_frame_action"
EXPANDED_CAMERA_ACTION_PROTOCOL = "a4_joint_visual_camera_frame_action_expanded_v2"
EXPANDED_CAMERA_ACTION_CHECKPOINT_TYPE = (
    "a4_joint_visual_camera_frame_action_expanded"
)
EXPANDED_RESIDUAL_SCALE = 0.1
RAW_ACTION_DIM = 4
RAW_ACTIONS_PER_TOKEN = ACTION_DIM // RAW_ACTION_DIM
EXPANDED_DYNAMICS_CONFIG = {
    "action_dim": ACTION_DIM,
    "state_dim": 192,
    "num_slots": 4,
    "num_tasks": len(TASKS),
    "max_horizon": 8,
    "prefix_depth": 6,
    "prefix_heads": 6,
    "prefix_mlp_dim": 1536,
    "predictor_depth": 10,
    "predictor_hidden_dim": 4096,
    "fusion_dim": 1536,
    "state_mlp_dim": 1536,
    "dropout": 0.1,
}


def _camera_rotation_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    extrinsic = batch.get("camera_extrinsic_cam_from_world")
    if not torch.is_tensor(extrinsic):
        raise KeyError(
            "camera-frame action conditioning requires "
            "camera_extrinsic_cam_from_world"
        )
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (4, 4):
        raise ValueError("camera extrinsic must have shape [B,4,4]")
    if not bool(torch.isfinite(extrinsic).all()):
        raise ValueError("camera extrinsic contains non-finite values")
    return extrinsic[:, :3, :3]


def transform_world_actions_to_camera(
    future_action: torch.Tensor,
    camera_rotation_world_to_camera: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-frame XYZ commands while preserving gripper commands."""

    if future_action.ndim != 3 or future_action.shape[-1] != ACTION_DIM:
        raise ValueError("future_action must have shape [B,H,20]")
    batch_size, horizon, _ = future_action.shape
    if camera_rotation_world_to_camera.shape != (batch_size, 3, 3):
        raise ValueError("camera rotation must have shape [B,3,3]")
    if not future_action.is_floating_point() or not bool(
        torch.isfinite(future_action).all()
    ):
        raise ValueError("future_action must be finite floating point")
    if not bool(torch.isfinite(camera_rotation_world_to_camera).all()):
        raise ValueError("camera rotation contains non-finite values")
    if future_action.device != camera_rotation_world_to_camera.device:
        raise ValueError("future_action and camera rotation must share device")

    raw = future_action.reshape(
        batch_size, horizon, RAW_ACTIONS_PER_TOKEN, RAW_ACTION_DIM
    )
    rotation = camera_rotation_world_to_camera.to(dtype=future_action.dtype)
    camera_xyz = torch.einsum("bij,bhtj->bhti", rotation, raw[..., :3])
    transformed = torch.cat([camera_xyz, raw[..., 3:4]], dim=-1)
    return transformed.reshape(batch_size, horizon, ACTION_DIM)


def camera_action_statistics(
    index: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Compute camera-frame action normalization on train rows only."""

    if "camera_extrinsic_cam_from_world" not in arrays:
        raise KeyError("camera action statistics require camera extrinsics")
    train = np.flatnonzero(np.asarray(index["split"]).astype(str) == "train")
    if train.size == 0:
        raise ValueError("camera action statistics require train rows")
    actions = np.asarray(index["future_action"], dtype=np.float32)[train]
    extrinsic = np.asarray(
        arrays["camera_extrinsic_cam_from_world"], dtype=np.float32
    )[train]
    if actions.ndim != 4 or actions.shape[-1] != ACTION_DIM:
        raise ValueError("indexed future actions must have shape [N,8,H,20]")
    if extrinsic.shape != (train.size, 4, 4):
        raise ValueError("indexed camera extrinsics must have shape [N,4,4]")
    raw = actions.reshape(*actions.shape[:-1], RAW_ACTIONS_PER_TOKEN, RAW_ACTION_DIM)
    camera_xyz = np.einsum(
        "nij,nbhtj->nbhti", extrinsic[:, :3, :3], raw[..., :3]
    )
    transformed = np.concatenate([camera_xyz, raw[..., 3:4]], axis=-1).reshape(
        -1, ACTION_DIM
    )
    if not np.isfinite(transformed).all():
        raise ValueError("camera-frame action statistics are non-finite")
    return {
        "action_mean": transformed.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(transformed.std(axis=0), 1.0e-6).astype(
            np.float32
        ),
    }


class CameraActionJointSevenTaskLeWM(nn.Module):
    """Run action dynamics in the current camera coordinate frame."""

    def __init__(self, base_model: JointSevenTaskActionPrefixLeWM) -> None:
        super().__init__()
        if not isinstance(base_model, JointSevenTaskActionPrefixLeWM):
            raise TypeError("base_model must be a joint seven-task A4 LeWM")
        self.base_model = base_model

    @property
    def frame_encoder(self) -> nn.Module:
        return self.base_model.frame_encoder

    @property
    def state_projector(self) -> nn.Module:
        return self.base_model.state_projector

    @property
    def dynamics(self) -> nn.Module:
        return self.base_model.dynamics

    @property
    def metric_readout(self) -> nn.Module:
        return self.base_model.metric_readout

    def encode_observations(self, **kwargs: Any) -> torch.Tensor:
        """Expose the base observation encoder for inference-only tools."""

        return self.base_model.encode_observations(**kwargs)

    def transform_future_action(
        self,
        future_action: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> torch.Tensor:
        return transform_world_actions_to_camera(
            future_action,
            _camera_rotation_from_batch(batch),
        )

    def forward(self, batch: Mapping[str, Any]) -> JointA4Output:
        transformed_batch = dict(batch)
        transformed_batch["future_action"] = self.transform_future_action(
            batch["future_action"], batch
        )
        return self.base_model(transformed_batch)


class ExpandedActionPrefixDynamics(nn.Module):
    """Add a zero-initialized high-capacity residual to proven dynamics."""

    def __init__(
        self,
        base_dynamics: ActionPrefixDynamics,
        expansion: ActionPrefixDynamics,
        *,
        residual_scale: float = EXPANDED_RESIDUAL_SCALE,
    ) -> None:
        super().__init__()
        if not isinstance(base_dynamics, ActionPrefixDynamics) or not isinstance(
            expansion, ActionPrefixDynamics
        ):
            raise TypeError("expanded dynamics requires two ActionPrefixDynamics")
        for name in ("action_dim", "state_dim", "num_slots", "num_tasks"):
            if getattr(base_dynamics, name) != getattr(expansion, name):
                raise ValueError(f"expanded dynamics {name} differs from base")
        if expansion.max_horizon < base_dynamics.max_horizon:
            raise ValueError("expanded dynamics horizon cannot be shorter than base")
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError("expanded dynamics residual scale must be in (0,1]")
        self.base_dynamics = base_dynamics
        self.expansion = expansion
        self.residual_scale = float(residual_scale)

    @property
    def action_dim(self) -> int:
        return int(self.base_dynamics.action_dim)

    def set_action_normalization(
        self,
        mean: torch.Tensor | list[float],
        std: torch.Tensor | list[float],
    ) -> None:
        self.base_dynamics.set_action_normalization(mean, std)
        self.expansion.set_action_normalization(mean, std)

    def forward(
        self,
        current_state: torch.Tensor,
        future_actions: torch.Tensor,
        task_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_prediction = self.base_dynamics(
            current_state, future_actions, task_index=task_index
        )
        expansion_prediction = self.expansion(
            current_state, future_actions, task_index=task_index
        )
        return base_prediction + self.residual_scale * (
            expansion_prediction - current_state[:, None]
        )


def build_camera_action_joint_model(
    parent_checkpoint: Mapping[str, Any],
    *,
    action_statistics: Mapping[str, np.ndarray | torch.Tensor],
    loss_profile: str | None = None,
) -> CameraActionJointSevenTaskLeWM:
    if float(parent_checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("camera-action parent SIGReg must be exactly 0.02")
    base_model = load_joint_checkpoint(parent_checkpoint)
    if loss_profile is not None:
        base_model.loss_profile = loss_profile
    base_model.dynamics.set_action_normalization(
        action_statistics["action_mean"], action_statistics["action_std"]
    )
    return CameraActionJointSevenTaskLeWM(base_model)


def build_expanded_camera_action_joint_model(
    camera_action_checkpoint: Mapping[str, Any],
    *,
    seed: int,
    expansion_config: Mapping[str, Any] = EXPANDED_DYNAMICS_CONFIG,
) -> CameraActionJointSevenTaskLeWM:
    """Expand a proven camera-action model without changing its initial output."""

    base_model = load_camera_action_joint_checkpoint(camera_action_checkpoint)
    old_dynamics = base_model.base_model.dynamics
    if not isinstance(old_dynamics, ActionPrefixDynamics):
        raise TypeError("camera-action parent dynamics type drifted")
    config = dict(expansion_config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        expansion = ActionPrefixDynamics(**config)
    expansion.set_action_normalization(
        old_dynamics.action_mean, old_dynamics.action_std
    )
    base_model.base_model.dynamics = ExpandedActionPrefixDynamics(
        old_dynamics,
        expansion,
        residual_scale=EXPANDED_RESIDUAL_SCALE,
    )
    return base_model


def camera_action_checkpoint_payload(
    model: CameraActionJointSevenTaskLeWM,
    *,
    step: int,
    seed: int,
    parent_checkpoint: str,
    raw_dataset: str,
    action_statistics: Mapping[str, np.ndarray | torch.Tensor],
    loss_profile: str | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_type": CAMERA_ACTION_CHECKPOINT_TYPE,
        "protocol": CAMERA_ACTION_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "tasks": list(TASKS),
        "sigreg_weight": SIGREG_WEIGHT,
        "loss_profile": (
            model.base_model.loss_profile
            if loss_profile is None
            else loss_profile
        ),
        "camera_conditioning": "continuous_world_to_camera_rotation_on_action",
        "parent_checkpoint": str(parent_checkpoint),
        "raw_dataset": str(raw_dataset),
        "action_statistics": {
            name: torch.as_tensor(value).detach().cpu().float().tolist()
            for name, value in action_statistics.items()
        },
        "model_state": model.state_dict(),
    }


def load_camera_action_joint_checkpoint(
    checkpoint: Mapping[str, Any],
) -> CameraActionJointSevenTaskLeWM:
    if checkpoint.get("checkpoint_type") != CAMERA_ACTION_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a camera-action joint A4 model")
    if checkpoint.get("protocol") != CAMERA_ACTION_PROTOCOL:
        raise ValueError("camera-action joint A4 protocol drifted")
    if tuple(checkpoint.get("tasks", ())) != TASKS:
        raise ValueError("camera-action joint A4 task panel drifted")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("camera-action joint A4 SIGReg must be exactly 0.02")
    parent_path = str(checkpoint.get("parent_checkpoint", ""))
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping):
        raise TypeError("camera-action parent checkpoint must contain a mapping")
    statistics = checkpoint.get("action_statistics")
    state = checkpoint.get("model_state")
    if not isinstance(statistics, Mapping) or not isinstance(state, Mapping):
        raise ValueError("camera-action joint A4 checkpoint is incomplete")
    model = build_camera_action_joint_model(
        parent,
        action_statistics=statistics,
        loss_profile=str(checkpoint.get("loss_profile", LOSS_PROFILE_UNIFORM)),
    )
    model.load_state_dict(state, strict=True)
    return model


def expanded_camera_action_checkpoint_payload(
    model: CameraActionJointSevenTaskLeWM,
    *,
    step: int,
    seed: int,
    parent_checkpoint: str,
    raw_dataset: str,
    expansion_config: Mapping[str, Any] = EXPANDED_DYNAMICS_CONFIG,
) -> dict[str, Any]:
    if not isinstance(model.dynamics, ExpandedActionPrefixDynamics):
        raise TypeError("expanded checkpoint requires ExpandedActionPrefixDynamics")
    return {
        "checkpoint_type": EXPANDED_CAMERA_ACTION_CHECKPOINT_TYPE,
        "protocol": EXPANDED_CAMERA_ACTION_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "tasks": list(TASKS),
        "sigreg_weight": SIGREG_WEIGHT,
        "camera_conditioning": "continuous_world_to_camera_rotation_on_action",
        "capacity_strategy": "proven_base_plus_zero_initialized_residual",
        "residual_scale": EXPANDED_RESIDUAL_SCALE,
        "parent_checkpoint": str(parent_checkpoint),
        "raw_dataset": str(raw_dataset),
        "expansion_config": dict(expansion_config),
        "model_state": model.state_dict(),
    }


def load_expanded_camera_action_joint_checkpoint(
    checkpoint: Mapping[str, Any],
) -> CameraActionJointSevenTaskLeWM:
    if checkpoint.get("checkpoint_type") != EXPANDED_CAMERA_ACTION_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not an expanded camera-action model")
    if checkpoint.get("protocol") != EXPANDED_CAMERA_ACTION_PROTOCOL:
        raise ValueError("expanded camera-action protocol drifted")
    if tuple(checkpoint.get("tasks", ())) != TASKS:
        raise ValueError("expanded camera-action task panel drifted")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("expanded camera-action SIGReg must be exactly 0.02")
    if float(checkpoint.get("residual_scale", -1.0)) != EXPANDED_RESIDUAL_SCALE:
        raise ValueError("expanded camera-action residual scale drifted")
    parent_path = str(checkpoint.get("parent_checkpoint", ""))
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("expansion_config")
    state = checkpoint.get("model_state")
    if not isinstance(parent, Mapping) or not isinstance(config, Mapping):
        raise ValueError("expanded camera-action checkpoint parent/config is incomplete")
    if not isinstance(state, Mapping):
        raise ValueError("expanded camera-action checkpoint state is incomplete")
    model = build_expanded_camera_action_joint_model(
        parent,
        seed=int(checkpoint.get("seed", 0)),
        expansion_config=config,
    )
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "CAMERA_ACTION_CHECKPOINT_TYPE",
    "CAMERA_ACTION_PROTOCOL",
    "EXPANDED_CAMERA_ACTION_CHECKPOINT_TYPE",
    "EXPANDED_CAMERA_ACTION_PROTOCOL",
    "EXPANDED_DYNAMICS_CONFIG",
    "EXPANDED_RESIDUAL_SCALE",
    "CameraActionJointSevenTaskLeWM",
    "ExpandedActionPrefixDynamics",
    "build_camera_action_joint_model",
    "build_expanded_camera_action_joint_model",
    "camera_action_checkpoint_payload",
    "camera_action_statistics",
    "expanded_camera_action_checkpoint_payload",
    "load_camera_action_joint_checkpoint",
    "load_expanded_camera_action_joint_checkpoint",
    "transform_world_actions_to_camera",
]
