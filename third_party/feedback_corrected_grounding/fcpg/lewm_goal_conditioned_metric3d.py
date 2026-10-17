"""Direct goal/entity-conditioned LeWM dynamics for active manipulation.

This module keeps the seven-task history-conditioned LeWM as the dynamics
backbone, but puts task-goal context into its action-prefix condition before
the prediction is made.  The controlled entity is a first-class output: for
Reach it is the end effector, while for Push it is the puck/object.  This is
deliberately different from a post-hoc endpoint sidecar.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix_joint_a4 import (
    ACTION_DIM,
    ANCHOR_COUNT,
    LATENT_DIM,
    STATE_SLOTS,
)
from fcpg.lewm_camera_action_joint_a4 import transform_world_actions_to_camera
from fcpg.lewm_history_conditioned_metric3d_a4 import (
    HISTORY_ACTION_STEPS,
    HistoryConditionedMetric3DLeWM,
)
from fcpg.lewm_metric3d_joint_a4 import WORLD_SCALE_M


GOAL_CONDITIONED_PROTOCOL = "a4_metric3d_world_goal_entity_direct_v1"
GOAL_CONDITIONED_CHECKPOINT_TYPE = "a4_metric3d_world_goal_entity_direct"
TASK_COUNT = len(TASKS)
TASK_EMBED_DIM = 16
HORIZON_EMBED_DIM = 16
GOAL_VECTOR_DIM = 36
GOAL_FEATURE_DIM = GOAL_VECTOR_DIM + TASK_EMBED_DIM
DEFAULT_HIDDEN_DIM = 256
DEFAULT_MAX_ENTITY_DELTA_M = 0.30


@dataclass(frozen=True)
class GoalConditionedOutput:
    current_state: torch.Tensor
    predicted_state: torch.Tensor
    predicted_world_xyz: torch.Tensor
    predicted_entity_xyz: torch.Tensor
    predicted_goal_error: torch.Tensor
    prefix_condition: torch.Tensor
    robot_loss: torch.Tensor | None
    entity_loss: torch.Tensor | None
    goal_loss: torch.Tensor | None
    loss: torch.Tensor | None


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite tensor")


class GoalConditionedMetric3DLeWM(nn.Module):
    """Inject current goal context into the action-conditioned LeWM directly.

    The parent visual encoder, metric readout, and seven-task dynamics remain
    frozen in the first development phase.  Only the goal-conditioning path
    and controlled-entity head are trainable.  Both residual paths are
    zero-initialized, so a fresh model is exactly the parent forecast.
    """

    def __init__(
        self,
        parent: HistoryConditionedMetric3DLeWM,
        *,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        max_entity_delta_m: float = DEFAULT_MAX_ENTITY_DELTA_M,
    ) -> None:
        super().__init__()
        if not isinstance(parent, HistoryConditionedMetric3DLeWM):
            raise TypeError("parent must be HistoryConditionedMetric3DLeWM")
        if parent.base_model.residual_frame != "world":
            raise ValueError("goal-conditioned model requires world-frame parent")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if not torch.isfinite(torch.tensor(float(max_entity_delta_m))) or max_entity_delta_m <= 0.0:
            raise ValueError("max_entity_delta_m must be finite and positive")

        self.parent = parent
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.hidden_dim = int(hidden_dim)
        self.max_entity_delta_m = float(max_entity_delta_m)

        self.task_embedding = nn.Embedding(TASK_COUNT, TASK_EMBED_DIM)
        self.horizon_embedding = nn.Embedding(HORIZON, HORIZON_EMBED_DIM)
        self.goal_encoder = nn.Sequential(
            nn.LayerNorm(GOAL_FEATURE_DIM),
            nn.Linear(GOAL_FEATURE_DIM, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), LATENT_DIM),
        )
        self.goal_condition_adapter = nn.Sequential(
            nn.LayerNorm(2 * LATENT_DIM),
            nn.Linear(2 * LATENT_DIM, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), LATENT_DIM),
        )
        # Zero-initialized residuals preserve the parent checkpoint exactly.
        nn.init.zeros_(self.goal_condition_adapter[-1].weight)
        nn.init.zeros_(self.goal_condition_adapter[-1].bias)

        entity_input_dim = (
            STATE_SLOTS * LATENT_DIM
            + LATENT_DIM
            + GOAL_FEATURE_DIM
            + HORIZON_EMBED_DIM
        )
        self.entity_head = nn.Sequential(
            nn.LayerNorm(entity_input_dim),
            nn.Linear(entity_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 3),
        )
        nn.init.zeros_(self.entity_head[-1].weight)
        nn.init.zeros_(self.entity_head[-1].bias)

    @property
    def trainable_parameters(self):
        return (
            list(self.goal_encoder.parameters())
            + list(self.goal_condition_adapter.parameters())
            + list(self.entity_head.parameters())
            + list(self.task_embedding.parameters())
            + list(self.horizon_embedding.parameters())
        )

    def train(self, mode: bool = True) -> "GoalConditionedMetric3DLeWM":
        super().train(mode)
        self.parent.eval()
        self.goal_encoder.train(mode)
        self.goal_condition_adapter.train(mode)
        self.entity_head.train(mode)
        self.task_embedding.train(mode)
        self.horizon_embedding.train(mode)
        return self

    @staticmethod
    def _rotation(batch: Mapping[str, Any]) -> torch.Tensor:
        extrinsic = batch.get("camera_extrinsic_cam_from_world")
        if not torch.is_tensor(extrinsic) or extrinsic.shape[1:] != (4, 4):
            raise ValueError("camera extrinsic must have shape [B,4,4]")
        _require_finite(extrinsic, "camera extrinsic")
        return extrinsic[:, :3, :3]

    def _goal_features(
        self,
        *,
        current_entity_xyz: torch.Tensor,
        current_goal_xyz: torch.Tensor,
        current_goal_velocity: torch.Tensor,
        current_world_xyz: torch.Tensor,
        task_index: torch.Tensor,
        rotation_world_to_camera: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(current_entity_xyz.shape[0])
        if current_entity_xyz.shape != (batch_size, 3):
            raise ValueError("current_entity_xyz must have shape [B,3]")
        for name, value in (
            ("current_goal_xyz", current_goal_xyz),
            ("current_goal_velocity", current_goal_velocity),
        ):
            if value.shape != (batch_size, 3):
                raise ValueError(f"{name} must have shape [B,3]")
        if current_world_xyz.shape != (batch_size, ANCHOR_COUNT, 3):
            raise ValueError("current_world_xyz must have shape [B,3,3]")
        if task_index.shape != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        if bool((task_index < 0).any()) or bool((task_index >= TASK_COUNT).any()):
            raise ValueError("task_index is outside the seven-task panel")
        for name, value in (
            ("current_entity_xyz", current_entity_xyz),
            ("current_goal_xyz", current_goal_xyz),
            ("current_goal_velocity", current_goal_velocity),
            ("current_world_xyz", current_world_xyz),
        ):
            _require_finite(value, name)

        eef = current_world_xyz[:, -1]
        world_vectors = torch.stack(
            (
                current_entity_xyz,
                current_goal_xyz,
                current_goal_velocity,
                eef,
                current_goal_xyz - current_entity_xyz,
                current_goal_xyz - eef,
            ),
            dim=1,
        ) / float(WORLD_SCALE_M)
        camera_vectors = torch.einsum(
            "bij,bvj->bvi", rotation_world_to_camera, world_vectors
        )
        task = self.task_embedding(task_index.long())
        return torch.cat(
            [world_vectors.reshape(batch_size, -1), camera_vectors.reshape(batch_size, -1), task],
            dim=-1,
        )

    def _predict_dynamics(
        self,
        *,
        current_state: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
        task_index: torch.Tensor,
        goal_features: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current_state.shape[1:] != (STATE_SLOTS, LATENT_DIM):
            raise ValueError("current_state must have shape [B,4,192]")
        if history_action.shape[1:] != (HISTORY_ACTION_STEPS, ACTION_DIM):
            raise ValueError("history_action must have shape [B,2,20]")
        if future_action.ndim != 3 or future_action.shape[-1] != ACTION_DIM:
            raise ValueError("future_action must have shape [B,H,20]")
        if future_action.shape[1] > HORIZON:
            raise ValueError("future action horizon exceeds eight")
        if goal_features.shape[0] != current_state.shape[0]:
            raise ValueError("goal feature batch does not match current state")
        _require_finite(current_state, "current_state")
        _require_finite(history_action, "history_action")
        _require_finite(future_action, "future_action")

        transformed_future = self.parent.transform_future_action(future_action, batch)
        transformed_history = transform_world_actions_to_camera(
            history_action,
            self._rotation(batch),
        ).to(dtype=current_state.dtype)
        base_dynamics = self.parent.history_dynamics.base_dynamics
        with torch.no_grad():
            base_condition = base_dynamics.encode_prefix(
                current_state,
                transformed_future.to(dtype=current_state.dtype),
                task_index=task_index,
            )
            normalized_history = (
                transformed_history
                - base_dynamics.action_mean.to(dtype=transformed_history.dtype)
            ) / base_dynamics.action_std.to(dtype=transformed_history.dtype)
            flattened_history = normalized_history.reshape(
                current_state.shape[0], HISTORY_ACTION_STEPS * ACTION_DIM
            )
            history_input = torch.cat(
                [base_condition, flattened_history[:, None].expand(-1, future_action.shape[1], -1)],
                dim=-1,
            )
            history_correction = self.parent.history_dynamics.condition_adapter(history_input)

        goal_token = self.goal_encoder(goal_features)
        goal_token_h = goal_token[:, None].expand(-1, future_action.shape[1], -1)
        goal_correction = self.goal_condition_adapter(
            torch.cat([base_condition, goal_token_h], dim=-1)
        )
        condition = base_condition + history_correction + goal_correction
        shared_current = current_state[:, None].expand(
            -1, future_action.shape[1], STATE_SLOTS, LATENT_DIM
        )
        residual = base_dynamics.residual_predictor(shared_current, condition)
        return shared_current + residual, condition

    def _predict_entity(
        self,
        *,
        predicted_state: torch.Tensor,
        prefix_condition: torch.Tensor,
        goal_features: torch.Tensor,
        current_entity_xyz: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon = predicted_state.shape[:2]
        task_and_horizon = goal_features[:, None].expand(-1, horizon, -1)
        horizon_ids = torch.arange(horizon, device=predicted_state.device)
        horizon_embedding = self.horizon_embedding(horizon_ids)[None].expand(
            batch_size, -1, -1
        )
        features = torch.cat(
            [
                predicted_state.reshape(batch_size, horizon, -1),
                prefix_condition,
                task_and_horizon,
                horizon_embedding,
            ],
            dim=-1,
        )
        delta = float(self.max_entity_delta_m) * torch.tanh(self.entity_head(features))
        return current_entity_xyz[:, None] + delta

    def predict(
        self,
        *,
        current_state: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
        current_entity_xyz: torch.Tensor,
        current_goal_xyz: torch.Tensor,
        current_goal_velocity: torch.Tensor,
        task_index: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._goal_features(
            current_entity_xyz=current_entity_xyz,
            current_goal_xyz=current_goal_xyz,
            current_goal_velocity=current_goal_velocity,
            current_world_xyz=batch["current_keypoints_xyz"],
            task_index=task_index,
            rotation_world_to_camera=self._rotation(batch),
        )
        predicted_state, prefix = self._predict_dynamics(
            current_state=current_state,
            history_action=history_action,
            future_action=future_action,
            task_index=task_index,
            goal_features=features,
            batch=batch,
        )
        predicted_entity = self._predict_entity(
            predicted_state=predicted_state,
            prefix_condition=prefix,
            goal_features=features,
            current_entity_xyz=current_entity_xyz,
        )
        return predicted_state, predicted_entity, prefix

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
        current_entity_xyz: torch.Tensor,
        current_goal_xyz: torch.Tensor,
        current_goal_velocity: torch.Tensor,
        task_index: torch.Tensor,
        batch: Mapping[str, Any],
        target_xyz: torch.Tensor | None = None,
        target_entity_xyz: torch.Tensor | None = None,
        target_goal_xyz: torch.Tensor | None = None,
    ) -> GoalConditionedOutput:
        predicted_state, predicted_entity, prefix = self.predict(
            current_state=current_state,
            history_action=history_action,
            future_action=future_action,
            current_entity_xyz=current_entity_xyz,
            current_goal_xyz=current_goal_xyz,
            current_goal_velocity=current_goal_velocity,
            task_index=task_index,
            batch=batch,
        )
        decoded = self.parent.base_model.decode_3d(
            predicted_state=predicted_state,
            current_state=current_state,
            current_keypoints_xyz=batch["current_keypoints_xyz"].float(),
            future_action=future_action.float(),
            task_index=task_index.long(),
            camera_intrinsic=batch["camera_intrinsic"].float(),
            extrinsic_cam_from_world=batch[
                "camera_extrinsic_cam_from_world"
            ].float(),
        )
        robot_loss = entity_loss = goal_loss = loss = None
        if target_xyz is not None or target_entity_xyz is not None or target_goal_xyz is not None:
            if target_xyz is None or target_entity_xyz is None or target_goal_xyz is None:
                raise ValueError("all three target tensors are required together")
            robot_loss = ((decoded.predicted_world_xyz - target_xyz) / WORLD_SCALE_M).square().mean()
            entity_loss = ((predicted_entity - target_entity_xyz) / WORLD_SCALE_M).square().mean()
            predicted_relative = predicted_entity - target_goal_xyz
            target_relative = target_entity_xyz - target_goal_xyz
            goal_loss = ((predicted_relative - target_relative) / WORLD_SCALE_M).square().mean()
            # The goal-relative target is a coordinate change of the entity
            # target, not permission to pull the prediction directly to the
            # goal. Keep one effective entity term in the optimization.
            loss = robot_loss + 2.0 * entity_loss
        return GoalConditionedOutput(
            current_state=current_state,
            predicted_state=predicted_state,
            predicted_world_xyz=decoded.predicted_world_xyz,
            predicted_entity_xyz=predicted_entity,
            predicted_goal_error=predicted_entity - (
                current_goal_xyz[:, None]
                if target_goal_xyz is None
                else target_goal_xyz
            ),
            prefix_condition=prefix,
            robot_loss=robot_loss,
            entity_loss=entity_loss,
            goal_loss=goal_loss,
            loss=loss,
        )


def direct_checkpoint_payload(
    model: GoalConditionedMetric3DLeWM,
    *,
    step: int,
    seed: int,
    parent_checkpoint: str,
    dataset: str,
    train_grounding: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(model, GoalConditionedMetric3DLeWM):
        raise TypeError("model must be GoalConditionedMetric3DLeWM")
    return {
        "checkpoint_type": GOAL_CONDITIONED_CHECKPOINT_TYPE,
        "protocol": GOAL_CONDITIONED_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "tasks": list(TASKS),
        "sigreg_weight": SIGREG_WEIGHT,
        "parent_checkpoint": str(parent_checkpoint),
        "dataset": str(dataset),
        "train_grounding": str(train_grounding),
        "hidden_dim": int(model.hidden_dim),
        "max_entity_delta_m": float(model.max_entity_delta_m),
        "goal_encoder": model.goal_encoder.state_dict(),
        "goal_condition_adapter": model.goal_condition_adapter.state_dict(),
        "entity_head": model.entity_head.state_dict(),
        "task_embedding": model.task_embedding.state_dict(),
        "horizon_embedding": model.horizon_embedding.state_dict(),
        "metrics": dict(metrics),
    }


def load_goal_conditioned_checkpoint(
    checkpoint: Mapping[str, Any],
    parent: HistoryConditionedMetric3DLeWM,
) -> GoalConditionedMetric3DLeWM:
    """Restore a frozen goal-conditioned model from its checkpoint payload."""

    if checkpoint.get("checkpoint_type") != GOAL_CONDITIONED_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a goal-conditioned metric-3D LeWM")
    if checkpoint.get("protocol") != GOAL_CONDITIONED_PROTOCOL:
        raise ValueError("goal-conditioned checkpoint protocol mismatch")
    if tuple(checkpoint.get("tasks", ())) != tuple(TASKS):
        raise ValueError("goal-conditioned checkpoint task panel mismatch")
    if float(checkpoint.get("sigreg_weight", float("nan"))) != float(SIGREG_WEIGHT):
        raise ValueError("goal-conditioned checkpoint SIGReg mismatch")

    model = GoalConditionedMetric3DLeWM(
        parent,
        hidden_dim=int(checkpoint["hidden_dim"]),
        max_entity_delta_m=float(checkpoint["max_entity_delta_m"]),
    )
    for name in (
        "goal_encoder",
        "goal_condition_adapter",
        "entity_head",
        "task_embedding",
        "horizon_embedding",
    ):
        state = checkpoint.get(name)
        if not isinstance(state, Mapping):
            raise ValueError(f"goal-conditioned checkpoint is missing {name}")
        getattr(model, name).load_state_dict(state, strict=True)
    return model


__all__ = [
    "GOAL_CONDITIONED_CHECKPOINT_TYPE",
    "GOAL_CONDITIONED_PROTOCOL",
    "GoalConditionedMetric3DLeWM",
    "GoalConditionedOutput",
    "direct_checkpoint_payload",
    "load_goal_conditioned_checkpoint",
]
