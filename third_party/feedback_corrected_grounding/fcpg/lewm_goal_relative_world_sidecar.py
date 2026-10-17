"""Goal-relative world-space residual for a frozen LeWM forecast.
The parent LeWM remains the only action-conditioned dynamics model.  This
small sidecar only uses its predicted world-space end-effector trajectory and
the currently observed target state to correct endpoint error.  It is meant
for a bounded development experiment, not as a replacement for the parent
model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


HORIZON = 8
ANCHOR_COUNT = 3
EEF_SLOT = 2
WORLD_SCALE_M = 0.1
WORLD_SIDECAR_PROTOCOL = "goal_relative_world_sidecar_v1"


class GoalRelativeWorldSidecar(nn.Module):
    """Predict a bounded EEF residual from a frozen world forecast.

    ``task_index`` is local to the moving-goal experiment: reach is 0 and push
    is 1.  The sidecar does not receive future goal positions; it receives the
    current target and its one-step observed velocity, which is the deployable
    target context available at the current decision.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        task_dim: int = 8,
        horizon_dim: int = 8,
        max_residual_m: float = 0.08,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or task_dim <= 0 or horizon_dim <= 0:
            raise ValueError("sidecar dimensions must be positive")
        if not torch.isfinite(torch.tensor(float(max_residual_m))) or max_residual_m <= 0:
            raise ValueError("max_residual_m must be positive and finite")
        self.max_residual_m = float(max_residual_m)
        self.task_embedding = nn.Embedding(2, int(task_dim))
        self.horizon_embedding = nn.Embedding(HORIZON, int(horizon_dim))

        # Seven 3-D vectors: base EEF, current EEF, current goal, target
        # velocity, and three relative displacements.
        input_dim = 7 * 3 + int(task_dim) + int(horizon_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 3),
        )
        # A fresh sidecar is exactly the frozen forecast and cannot damage the
        # parent model before it has learned a useful endpoint correction.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        base_world_xyz: torch.Tensor,
        current_world_xyz: torch.Tensor,
        current_goal_xyz: torch.Tensor,
        current_goal_velocity: torch.Tensor,
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        if base_world_xyz.ndim != 4 or tuple(base_world_xyz.shape[1:]) != (
            HORIZON,
            ANCHOR_COUNT,
            3,
        ):
            raise ValueError("base_world_xyz must have shape [B,8,3,3]")
        batch_size = int(base_world_xyz.shape[0])
        expected_current = (batch_size, ANCHOR_COUNT, 3)
        if tuple(current_world_xyz.shape) != expected_current:
            raise ValueError("current_world_xyz must have shape [B,3,3]")
        if tuple(current_goal_xyz.shape) != (batch_size, 3):
            raise ValueError("current_goal_xyz must have shape [B,3]")
        if tuple(current_goal_velocity.shape) != (batch_size, 3):
            raise ValueError("current_goal_velocity must have shape [B,3]")
        if tuple(task_index.shape) != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        if task_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("task_index must be an integer tensor")
        if bool((task_index < 0).any()) or bool((task_index >= 2).any()):
            raise ValueError("moving-goal task_index must contain only 0 and 1")
        tensors = (
            base_world_xyz,
            current_world_xyz,
            current_goal_xyz,
            current_goal_velocity,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("sidecar inputs must be finite")

        base_eef = base_world_xyz[:, :, EEF_SLOT]
        current_eef = current_world_xyz[:, EEF_SLOT]
        scale = base_eef.new_tensor(WORLD_SCALE_M)
        base_eef_n = base_eef / scale
        current_eef_n = current_eef[:, None] / scale
        goal_n = current_goal_xyz[:, None] / scale
        velocity_n = current_goal_velocity[:, None] / scale
        features = torch.cat(
            [
                base_eef_n,
                current_eef_n.expand(-1, HORIZON, -1),
                goal_n.expand(-1, HORIZON, -1),
                velocity_n.expand(-1, HORIZON, -1),
                base_eef_n - goal_n,
                current_eef_n.expand(-1, HORIZON, -1) - goal_n,
                base_eef_n - current_eef_n.expand(-1, HORIZON, -1),
            ],
            dim=-1,
        )
        task = self.task_embedding(task_index.long())[:, None].expand(
            -1, HORIZON, -1
        )
        horizon_ids = torch.arange(HORIZON, device=base_world_xyz.device)
        horizon = self.horizon_embedding(horizon_ids)[None].expand(
            batch_size, -1, -1
        )
        features = torch.cat([features, task, horizon], dim=-1)
        residual = self.max_residual_m * torch.tanh(self.head(features))
        result = base_world_xyz.clone()
        result[:, :, EEF_SLOT] = base_eef + residual
        return result


def sidecar_payload(
    model: GoalRelativeWorldSidecar,
    *,
    step: int,
    seed: int,
    train_grounding: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(model, GoalRelativeWorldSidecar):
        raise TypeError("model must be GoalRelativeWorldSidecar")
    return {
        "checkpoint_type": "lewm_goal_relative_world_sidecar",
        "protocol": WORLD_SIDECAR_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "train_grounding": str(train_grounding),
        "max_residual_m": float(model.max_residual_m),
        "state_dict": model.state_dict(),
        "metrics": dict(metrics),
    }
