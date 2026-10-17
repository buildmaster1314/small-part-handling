"""Small goal-relative residual head for frozen LeWM metric forecasts.

The sidecar deliberately does not predict dynamics.  A frozen LeWM supplies
the action-conditioned H1-H8 anchor forecast; this module only uses the
current grounded measurement and a deployable goal projection to make the
forecast endpoint-aware.  It is intended for development diagnostics before
any end-to-end controller integration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS
from fcpg.custom_tasks.towel_wipe import SawyerTowelWipeEnvV0


SIDECAR_PROTOCOL = "goal_relative_pixel_sidecar_v1"
ANCHOR_COUNT = 3
ACTION_DIM = 20


def _as_float_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def project_world_goal(
    goal_world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_cam_from_world: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Project one world point with the same CV camera convention as A4."""

    point = _as_float_array(goal_world, (3,), "goal_world").astype(np.float64)
    camera_matrix = _as_float_array(intrinsic, (3, 3), "intrinsic").astype(np.float64)
    extrinsic = _as_float_array(
        extrinsic_cam_from_world, (4, 4), "extrinsic_cam_from_world"
    ).astype(np.float64)
    camera_point = extrinsic[:3, :3] @ point + extrinsic[:3, 3]
    depth = float(camera_point[2])
    if depth <= 1.0e-8:
        return np.zeros((2,), dtype=np.float32), depth, False
    homogeneous = camera_matrix @ camera_point
    pixel = (homogeneous[:2] / homogeneous[2]).astype(np.float32)
    return pixel, depth, bool(np.isfinite(pixel).all())


def towel_goal_world(task_instance: int, phase: str) -> np.ndarray:
    """Return the task-defined towel target for a recorded expert phase.

    The phase is a collection-time proxy for the target currently exposed in
    the towel observation.  It is not q_true and is only used by this
    development sidecar; a deployment version must replace it with a visual
    goal detector or an explicit task goal input.
    """

    left, right = SawyerTowelWipeEnvV0.task_instance_endpoints(int(task_instance))
    normalized = str(phase).lower()
    if "finish" in normalized or "lift_after_wipe" in normalized:
        goal = np.asarray((0.0, float(left[1]), SawyerTowelWipeEnvV0.FINISH_HEIGHT))
    elif any(token in normalized for token in ("descend", "grasp", "lift_towel")):
        # The live towel controller first raises the grasped towel before
        # returning to the wiping surface.  This is the controller-owned
        # target, not the surface endpoint used during wiping.
        goal = left.copy()
        goal[2] = SawyerTowelWipeEnvV0.INITIAL_GRASP_LIFT_HEIGHT + 0.03
    elif "right" in normalized:
        goal = right
    else:
        goal = left
    return np.asarray(goal, dtype=np.float32)


def towel_goal_projection(
    task_instance: int,
    phase: str,
    intrinsic: np.ndarray,
    extrinsic_cam_from_world: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    return project_world_goal(
        towel_goal_world(task_instance, phase), intrinsic, extrinsic_cam_from_world
    )


class GoalRelativePixelSidecar(nn.Module):
    """Bounded endpoint residual over a frozen pixel forecast.

    Inputs use pixel coordinates normalized by image width/height.  The final
    layer is zero initialized, so a fresh sidecar is exactly the frozen base
    prediction.  One shared head is used for all anchors and horizons, with
    learned identity embeddings preserving those two roles.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        task_dim: int = 16,
        horizon_dim: int = 8,
        anchor_dim: int = 8,
        max_residual_px: float = 32.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or task_dim <= 0 or horizon_dim <= 0 or anchor_dim <= 0:
            raise ValueError("sidecar dimensions must be positive")
        if not np.isfinite(max_residual_px) or max_residual_px <= 0.0:
            raise ValueError("max_residual_px must be positive and finite")
        self.max_residual_px = float(max_residual_px)
        self.task_embedding = nn.Embedding(len(TASKS), int(task_dim))
        self.horizon_embedding = nn.Embedding(HORIZON, int(horizon_dim))
        self.anchor_embedding = nn.Embedding(ANCHOR_COUNT, int(anchor_dim))
        # Five 2-D coordinate pairs (base, current, two relative errors, goal)
        # plus validity and the identity/context embeddings.
        input_dim = 10 + 1 + task_dim + horizon_dim + anchor_dim
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        base_q_px: torch.Tensor,
        current_q_px: torch.Tensor,
        goal_px: torch.Tensor,
        task_index: torch.Tensor,
        *,
        image_size: tuple[int, int],
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_q_px.ndim != 4 or tuple(base_q_px.shape[1:]) != (
            HORIZON,
            ANCHOR_COUNT,
            2,
        ):
            raise ValueError("base_q_px must have shape [B,8,3,2]")
        batch_size = int(base_q_px.shape[0])
        if current_q_px.shape != (batch_size, ANCHOR_COUNT, 2):
            raise ValueError("current_q_px must have shape [B,3,2]")
        if goal_px.shape != (batch_size, 2):
            raise ValueError("goal_px must have shape [B,2]")
        if task_index.shape != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        if valid is not None and valid.shape != (batch_size, ANCHOR_COUNT):
            raise ValueError("valid must have shape [B,3]")
        width, height = int(image_size[0]), int(image_size[1])
        if width <= 1 or height <= 1:
            raise ValueError("image_size must be larger than one pixel")
        scale = base_q_px.new_tensor([float(width - 1), float(height - 1)])
        base = base_q_px.float() / scale
        current = (current_q_px.float() / scale)[:, None].expand(
            -1, HORIZON, -1, -1
        )
        goal = (goal_px.float() / scale)[:, None, None].expand(
            -1, HORIZON, ANCHOR_COUNT, -1
        )
        base_flat = base.reshape(batch_size, HORIZON, ANCHOR_COUNT, 2)
        current_flat = current.reshape(batch_size, HORIZON, ANCHOR_COUNT, 2)
        goal_flat = goal.reshape(batch_size, HORIZON, ANCHOR_COUNT, 2)
        validity = (
            torch.ones(
                (batch_size, HORIZON, ANCHOR_COUNT, 1),
                device=base.device,
                dtype=base.dtype,
            )
            if valid is None
            else valid.float()[:, None, :, None].expand(
                -1, HORIZON, -1, -1
            )
        )
        task = self.task_embedding(task_index.long())[:, None, None].expand(
            -1, HORIZON, ANCHOR_COUNT, -1
        )
        horizon_ids = torch.arange(HORIZON, device=base.device)
        horizon = self.horizon_embedding(horizon_ids)[None, :, None].expand(
            batch_size, -1, ANCHOR_COUNT, -1
        )
        anchor_ids = torch.arange(ANCHOR_COUNT, device=base.device)
        anchor = self.anchor_embedding(anchor_ids)[None, None].expand(
            batch_size, HORIZON, -1, -1
        )
        features = torch.cat(
            [
                base_flat,
                current_flat,
                base_flat - goal_flat,
                current_flat - goal_flat,
                goal_flat,
                validity,
                task,
                horizon,
                anchor,
            ],
            dim=-1,
        )
        residual = self.max_residual_px * torch.tanh(self.head(features))
        return (base_flat + residual / scale) * scale


class GoalRelativeActionSidecar(GoalRelativePixelSidecar):
    """Goal residual head conditioned on the commanded H1-H8 action prefix.

    The frozen LeWM remains the only dynamics model.  This sidecar gets the
    same action prefix that produced the base forecast, so its residual cannot
    be a purely phase- or target-conditioned displacement guess.
    """

    action_context_dim = 16

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        input_dim = (
            10
            + 1
            + self.task_embedding.embedding_dim
            + self.horizon_embedding.embedding_dim
            + self.anchor_embedding.embedding_dim
            + self.action_context_dim
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(ACTION_DIM),
            nn.Linear(ACTION_DIM, 32),
            nn.GELU(),
            nn.Linear(32, self.action_context_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.head[1].out_features),
            nn.GELU(),
            nn.Linear(self.head[1].out_features, self.head[3].out_features),
            nn.GELU(),
            nn.Linear(self.head[3].out_features, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        base_q_px: torch.Tensor,
        current_q_px: torch.Tensor,
        goal_px: torch.Tensor,
        task_index: torch.Tensor,
        *,
        future_action: torch.Tensor,
        image_size: tuple[int, int],
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if future_action.ndim != 3 or tuple(future_action.shape[1:]) != (
            HORIZON,
            ACTION_DIM,
        ):
            raise ValueError("future_action must have shape [B,8,20]")
        if future_action.shape[0] != base_q_px.shape[0]:
            raise ValueError("future_action batch must match base_q_px")
        if base_q_px.ndim != 4 or tuple(base_q_px.shape[1:]) != (
            HORIZON,
            ANCHOR_COUNT,
            2,
        ):
            raise ValueError("base_q_px must have shape [B,8,3,2]")
        batch_size = int(base_q_px.shape[0])
        if current_q_px.shape != (batch_size, ANCHOR_COUNT, 2):
            raise ValueError("current_q_px must have shape [B,3,2]")
        if goal_px.shape != (batch_size, 2):
            raise ValueError("goal_px must have shape [B,2]")
        if task_index.shape != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        if valid is not None and valid.shape != (batch_size, ANCHOR_COUNT):
            raise ValueError("valid must have shape [B,3]")
        width, height = int(image_size[0]), int(image_size[1])
        if width <= 1 or height <= 1:
            raise ValueError("image_size must be larger than one pixel")
        scale = base_q_px.new_tensor([float(width - 1), float(height - 1)])
        base = base_q_px.float() / scale
        current = (current_q_px.float() / scale)[:, None].expand(
            -1, HORIZON, -1, -1
        )
        goal = (goal_px.float() / scale)[:, None, None].expand(
            -1, HORIZON, ANCHOR_COUNT, -1
        )
        validity = (
            torch.ones(
                (batch_size, HORIZON, ANCHOR_COUNT, 1),
                device=base.device,
                dtype=base.dtype,
            )
            if valid is None
            else valid.float()[:, None, :, None].expand(-1, HORIZON, -1, -1)
        )
        task = self.task_embedding(task_index.long())[:, None, None].expand(
            -1, HORIZON, ANCHOR_COUNT, -1
        )
        horizon_ids = torch.arange(HORIZON, device=base.device)
        horizon = self.horizon_embedding(horizon_ids)[None, :, None].expand(
            batch_size, -1, ANCHOR_COUNT, -1
        )
        anchor_ids = torch.arange(ANCHOR_COUNT, device=base.device)
        anchor = self.anchor_embedding(anchor_ids)[None, None].expand(
            batch_size, HORIZON, -1, -1
        )
        action = self.action_projection(future_action.float())[:, :, None].expand(
            -1, -1, ANCHOR_COUNT, -1
        )
        base_flat = base.reshape(batch_size, HORIZON, ANCHOR_COUNT, 2)
        current_flat = current.reshape(batch_size, HORIZON, ANCHOR_COUNT, 2)
        goal_flat = goal.reshape(batch_size, HORIZON, ANCHOR_COUNT, 2)
        features = torch.cat(
            [
                base_flat,
                current_flat,
                base_flat - goal_flat,
                current_flat - goal_flat,
                goal_flat,
                validity,
                task,
                horizon,
                anchor,
                action,
            ],
            dim=-1,
        )
        residual = self.max_residual_px * torch.tanh(self.head(features))
        return (base_flat + residual / scale) * scale


def sidecar_payload(
    model: GoalRelativePixelSidecar,
    *,
    step: int,
    seed: int,
    parent_checkpoint: str,
    feature_cache: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    action_aware = isinstance(model, GoalRelativeActionSidecar)
    return {
        "checkpoint_type": "goal_relative_pixel_sidecar",
        "protocol": (
            "goal_relative_action_sidecar_v1"
            if action_aware
            else SIDECAR_PROTOCOL
        ),
        "step": int(step),
        "seed": int(seed),
        "parent_checkpoint": str(parent_checkpoint),
        "feature_cache": str(feature_cache),
        "max_residual_px": float(model.max_residual_px),
        "model_config": {
            "hidden_dim": int(model.head[1].out_features),
            "task_dim": int(model.task_embedding.embedding_dim),
            "horizon_dim": int(model.horizon_embedding.embedding_dim),
            "anchor_dim": int(model.anchor_embedding.embedding_dim),
        },
        "metrics": dict(metrics),
        "model_state": model.state_dict(),
    }


def load_sidecar_checkpoint(payload: Mapping[str, Any]) -> GoalRelativePixelSidecar:
    protocol = payload.get("protocol")
    action_aware = protocol == "goal_relative_action_sidecar_v1"
    if protocol not in (SIDECAR_PROTOCOL, "goal_relative_action_sidecar_v1"):
        raise ValueError("checkpoint is not a goal-relative sidecar")
    config = payload.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("sidecar checkpoint is missing model_config")
    model_class = GoalRelativeActionSidecar if action_aware else GoalRelativePixelSidecar
    model = model_class(
        hidden_dim=int(config["hidden_dim"]),
        task_dim=int(config["task_dim"]),
        horizon_dim=int(config["horizon_dim"]),
        anchor_dim=int(config["anchor_dim"]),
        max_residual_px=float(payload["max_residual_px"]),
    )
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("sidecar checkpoint is missing model_state")
    model.load_state_dict(state, strict=True)
    return model
