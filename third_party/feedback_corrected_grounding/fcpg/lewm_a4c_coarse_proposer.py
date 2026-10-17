"""Small seven-task action-prefix proposer for the audited A4c panel.

The proposer supplies a plausible action center.  It is not a replacement for
LeWM: downstream control must still use the frozen world model to score or
refine candidates around this center.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS
from fcpg.lewm_action_prefix_joint_a4 import ACTION_DIM, LATENT_DIM, STATE_SLOTS


PROTOCOL = "lewm_a4c_coarse_proposer_v1"
CHECKPOINT_TYPE = "lewm_a4c_coarse_proposer"
HISTORY_STEPS = 2


class A4CCoarseActionProposer(nn.Module):
    """Predict an H8 action center from frozen LeWM state and causal history."""

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        slot_dim: int = 128,
        history_dim: int = 128,
        task_dim: int = 32,
        camera_feature_dim: int = 0,
        camera_dim: int = 64,
        residual_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if min(hidden_dim, slot_dim, history_dim, task_dim, camera_dim) <= 0:
            raise ValueError("proposer dimensions must be positive")
        if camera_feature_dim < 0:
            raise ValueError("camera_feature_dim cannot be negative")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.hidden_dim = int(hidden_dim)
        self.slot_dim = int(slot_dim)
        self.history_dim = int(history_dim)
        self.task_dim = int(task_dim)
        self.camera_feature_dim = int(camera_feature_dim)
        self.camera_dim = int(camera_dim)
        self.residual_scale = float(residual_scale)

        self.state_norm = nn.LayerNorm(LATENT_DIM)
        self.slot_projection = nn.Sequential(
            nn.Linear(LATENT_DIM, self.slot_dim),
            nn.GELU(),
        )
        self.history_projection = nn.Sequential(
            nn.LayerNorm(HISTORY_STEPS * ACTION_DIM),
            nn.Linear(HISTORY_STEPS * ACTION_DIM, self.history_dim),
            nn.GELU(),
        )
        self.task_embedding = nn.Embedding(len(TASKS), self.task_dim)
        self.camera_projection = (
            nn.Sequential(
                nn.LayerNorm(self.camera_feature_dim),
                nn.Linear(self.camera_feature_dim, self.camera_dim),
                nn.GELU(),
            )
            if self.camera_feature_dim > 0
            else None
        )
        fused_dim = STATE_SLOTS * self.slot_dim + self.history_dim + self.task_dim
        if self.camera_projection is not None:
            fused_dim += self.camera_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(self.hidden_dim, HORIZON * ACTION_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        current_state: torch.Tensor,
        history_action: torch.Tensor,
        task_index: torch.Tensor,
        camera_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_state.ndim != 3 or current_state.shape[1:] != (
            STATE_SLOTS,
            LATENT_DIM,
        ):
            raise ValueError("current_state must have shape [B,4,192]")
        batch_size = int(current_state.shape[0])
        if history_action.shape != (batch_size, HISTORY_STEPS, ACTION_DIM):
            raise ValueError("history_action must have shape [B,2,20]")
        if task_index.shape != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        if task_index.dtype != torch.long:
            raise TypeError("task_index must use torch.long")
        if bool(torch.any((task_index < 0) | (task_index >= len(TASKS)))):
            raise ValueError("task_index is outside the seven-task panel")
        if self.camera_projection is None:
            if camera_condition is not None:
                raise ValueError("camera_condition was supplied to a camera-free model")
        elif camera_condition is None or camera_condition.shape != (
            batch_size,
            self.camera_feature_dim,
        ):
            raise ValueError(
                "camera_condition must have shape "
                f"[B,{self.camera_feature_dim}]"
            )

        state = self.slot_projection(self.state_norm(current_state)).flatten(1)
        history = self.history_projection(history_action.flatten(1))
        task = self.task_embedding(task_index)
        features = [state, history, task]
        if self.camera_projection is not None:
            features.append(self.camera_projection(camera_condition.float()))
        raw_delta = self.head(self.trunk(torch.cat(features, dim=-1)))
        delta = self.residual_scale * torch.tanh(
            raw_delta.reshape(batch_size, HORIZON, ACTION_DIM)
        )
        base = history_action[:, -1:, :].expand(-1, HORIZON, -1)
        return torch.clamp(base + delta, min=-1.0, max=1.0)


def weighted_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Phase-balanced Smooth-L1 objective with explicit XYZ/gripper terms."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B,H,20]")
    if prediction.shape[1:] != (HORIZON, ACTION_DIM):
        raise ValueError("action prefix shape drifted")
    if sample_weight.shape != (prediction.shape[0],):
        raise ValueError("sample_weight must have shape [B]")
    weight = sample_weight.float().clamp_min(0.0)
    if not bool(torch.any(weight > 0.0)):
        raise ValueError("sample_weight must contain a positive value")
    element = F.smooth_l1_loss(prediction, target, reduction="none")
    xyz = element[:, :, [index for index in range(ACTION_DIM) if index % 4 != 3]]
    gripper = element[:, :, 3::4]
    xyz_row = xyz.mean(dim=(1, 2))
    gripper_row = gripper.mean(dim=(1, 2))
    normalizer = weight.sum().clamp_min(1.0e-6)
    xyz_loss = (xyz_row * weight).sum() / normalizer
    gripper_loss = (gripper_row * weight).sum() / normalizer
    loss = xyz_loss + gripper_loss
    return loss, {
        "xyz_loss": xyz_loss,
        "gripper_loss": gripper_loss,
    }


def checkpoint_payload(
    model: A4CCoarseActionProposer,
    *,
    step: int,
    seed: int,
    parent_checkpoint: str,
    control_index: str,
    input_mode: str = "state_history",
) -> dict[str, Any]:
    if not isinstance(model, A4CCoarseActionProposer):
        raise TypeError("model must be A4CCoarseActionProposer")
    return {
        "checkpoint_type": CHECKPOINT_TYPE,
        "protocol": PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "tasks": list(TASKS),
        "parent_checkpoint": str(parent_checkpoint),
        "control_index": str(control_index),
        "input_mode": str(input_mode),
        "config": {
            "hidden_dim": model.hidden_dim,
            "slot_dim": model.slot_dim,
            "history_dim": model.history_dim,
            "task_dim": model.task_dim,
            "camera_feature_dim": model.camera_feature_dim,
            "camera_dim": model.camera_dim,
            "residual_scale": model.residual_scale,
        },
        "model_state": model.state_dict(),
    }


def load_checkpoint(checkpoint: Mapping[str, Any]) -> A4CCoarseActionProposer:
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not an A4c coarse proposer")
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError("A4c coarse proposer protocol drifted")
    if tuple(checkpoint.get("tasks", ())) != TASKS:
        raise ValueError("A4c coarse proposer task panel drifted")
    config = checkpoint.get("config")
    state = checkpoint.get("model_state")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("A4c coarse proposer checkpoint is incomplete")
    model = A4CCoarseActionProposer(
        hidden_dim=int(config["hidden_dim"]),
        slot_dim=int(config["slot_dim"]),
        history_dim=int(config["history_dim"]),
        task_dim=int(config["task_dim"]),
        camera_feature_dim=int(config.get("camera_feature_dim", 0)),
        camera_dim=int(config.get("camera_dim", 64)),
        residual_scale=float(config["residual_scale"]),
    )
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "A4CCoarseActionProposer",
    "CHECKPOINT_TYPE",
    "HISTORY_STEPS",
    "PROTOCOL",
    "checkpoint_payload",
    "load_checkpoint",
    "weighted_action_loss",
]
