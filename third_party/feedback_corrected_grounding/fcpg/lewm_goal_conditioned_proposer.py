"""Compact deployable action-prefix proposals for goal-conditioned LeWM MPC."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS
from fcpg.lewm_action_prefix_joint_a4 import LATENT_DIM, STATE_SLOTS


ACTION_DIM = 20
HISTORY_STEPS = 2
TASK_EMBED_DIM = 16
WORLD_SCALE_M = 0.25
CHECKPOINT_TYPE = "goal_conditioned_action_prefix_proposer"
PROTOCOL = "goal_conditioned_action_prefix_proposer_v1"


def advantage_weighted_prefix_target(
    candidate_action: np.ndarray,
    target_entity_xyz: np.ndarray,
    target_goal_xyz: np.ndarray,
    *,
    temperature_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth outcome-weighted action target and its candidate weights."""

    actions = np.asarray(candidate_action, dtype=np.float32)
    entity = np.asarray(target_entity_xyz, dtype=np.float32)
    goal = np.asarray(target_goal_xyz, dtype=np.float32)
    if actions.ndim != 4 or actions.shape[2:] != (HORIZON, ACTION_DIM):
        raise ValueError("candidate_action must have shape [N,C,8,20]")
    expected = actions.shape[:2] + (HORIZON, 3)
    if entity.shape != expected or goal.shape != expected:
        raise ValueError("entity and goal targets must have shape [N,C,8,3]")
    if not np.isfinite(actions).all() or not np.isfinite(entity).all() or not np.isfinite(goal).all():
        raise ValueError("prefix target inputs must be finite")
    temperature = float(temperature_m)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_m must be finite and positive")
    terminal_cost = np.linalg.norm(entity[:, :, -1] - goal[:, :, -1], axis=-1)
    logits = -(terminal_cost - terminal_cost.min(axis=1, keepdims=True)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    target = np.einsum("nc,ncha->nha", weights, actions, optimize=True)
    return np.ascontiguousarray(target, dtype=np.float32), np.ascontiguousarray(weights)


@dataclass(frozen=True)
class GoalConditionedPrefixOutput:
    action_prefix: torch.Tensor
    latent: torch.Tensor


class GoalConditionedActionPrefixProposer(nn.Module):
    """Predict one smooth H8 proposal from the current grounded LeWM state."""

    def __init__(
        self,
        *,
        hidden_dim: int = 384,
        slot_dim: int = 128,
        use_entity_estimate: bool = False,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0 or int(slot_dim) <= 0:
            raise ValueError("hidden_dim and slot_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.slot_dim = int(slot_dim)
        self.use_entity_estimate = bool(use_entity_estimate)
        self.task_embedding = nn.Embedding(len(TASKS), TASK_EMBED_DIM)
        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, self.slot_dim),
            nn.GELU(),
        )
        self.history_encoder = nn.Sequential(
            nn.LayerNorm(HISTORY_STEPS * ACTION_DIM),
            nn.Linear(HISTORY_STEPS * ACTION_DIM, self.slot_dim),
            nn.GELU(),
        )
        goal_input_dim = 12 + (9 if self.use_entity_estimate else 0) + TASK_EMBED_DIM
        self.goal_encoder = nn.Sequential(
            nn.LayerNorm(goal_input_dim),
            nn.Linear(goal_input_dim, self.slot_dim),
            nn.GELU(),
        )
        fusion_dim = STATE_SLOTS * self.slot_dim + 2 * self.slot_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(self.hidden_dim, HORIZON * ACTION_DIM)

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        history_action: torch.Tensor,
        current_goal_xyz: torch.Tensor,
        current_goal_velocity: torch.Tensor,
        current_world_xyz: torch.Tensor,
        task_index: torch.Tensor,
        current_entity_xyz: torch.Tensor | None = None,
    ) -> GoalConditionedPrefixOutput:
        batch = int(current_state.shape[0])
        if current_state.shape != (batch, STATE_SLOTS, LATENT_DIM):
            raise ValueError("current_state must have shape [B,4,192]")
        if history_action.shape != (batch, HISTORY_STEPS, ACTION_DIM):
            raise ValueError("history_action must have shape [B,2,20]")
        if current_goal_xyz.shape != (batch, 3) or current_goal_velocity.shape != (batch, 3):
            raise ValueError("goal and goal velocity must have shape [B,3]")
        if current_world_xyz.shape != (batch, 3, 3):
            raise ValueError("current_world_xyz must have shape [B,3,3]")
        if self.use_entity_estimate:
            if current_entity_xyz is None or current_entity_xyz.shape != (batch, 3):
                raise ValueError(
                    "entity-conditioned proposer requires current_entity_xyz [B,3]"
                )
        elif current_entity_xyz is not None and current_entity_xyz.shape != (batch, 3):
            raise ValueError("current_entity_xyz must have shape [B,3]")
        if task_index.shape != (batch,):
            raise ValueError("task_index must have shape [B]")
        tensors = (
            current_state,
            history_action,
            current_goal_xyz,
            current_goal_velocity,
            current_world_xyz,
        )
        if current_entity_xyz is not None:
            tensors = (*tensors, current_entity_xyz)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("proposer inputs must be finite")
        if bool((task_index < 0).any()) or bool((task_index >= len(TASKS)).any()):
            raise ValueError("task_index is outside the seven-task panel")

        slots = self.slot_encoder(current_state).reshape(batch, -1)
        history = self.history_encoder(history_action.reshape(batch, -1))
        eef = current_world_xyz[:, -1]
        goal_vector = torch.cat(
            [
                current_goal_xyz,
                current_goal_velocity,
                eef,
                current_goal_xyz - eef,
            ],
            dim=-1,
        ) / float(WORLD_SCALE_M)
        if self.use_entity_estimate:
            assert current_entity_xyz is not None
            entity_vector = torch.cat(
                [
                    current_entity_xyz,
                    current_entity_xyz - eef,
                    current_goal_xyz - current_entity_xyz,
                ],
                dim=-1,
            ) / float(WORLD_SCALE_M)
            goal_vector = torch.cat([goal_vector, entity_vector], dim=-1)
        goal = self.goal_encoder(
            torch.cat([goal_vector, self.task_embedding(task_index.long())], dim=-1)
        )
        latent = self.fusion(torch.cat([slots, history, goal], dim=-1))
        action = torch.tanh(self.action_head(latent)).reshape(batch, HORIZON, ACTION_DIM)
        return GoalConditionedPrefixOutput(action_prefix=action, latent=latent)


def proposer_checkpoint_payload(
    model: GoalConditionedActionPrefixProposer,
    *,
    step: int,
    args: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_type": CHECKPOINT_TYPE,
        "protocol": PROTOCOL,
        "step": int(step),
        "model_config": {
            "hidden_dim": model.hidden_dim,
            "slot_dim": model.slot_dim,
            "use_entity_estimate": model.use_entity_estimate,
        },
        "state_dict": model.state_dict(),
        "args": dict(args),
        "metrics": dict(metrics),
    }


def load_goal_conditioned_prefix_proposer(
    payload: Mapping[str, Any],
) -> GoalConditionedActionPrefixProposer:
    if payload.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError("unexpected action-prefix proposer checkpoint type")
    config = payload.get("model_config")
    state = payload.get("state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("proposer checkpoint is missing model configuration or state")
    model = GoalConditionedActionPrefixProposer(
        hidden_dim=int(config["hidden_dim"]),
        slot_dim=int(config["slot_dim"]),
        use_entity_estimate=bool(config.get("use_entity_estimate", False)),
    )
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "CHECKPOINT_TYPE",
    "GoalConditionedActionPrefixProposer",
    "GoalConditionedPrefixOutput",
    "PROTOCOL",
    "advantage_weighted_prefix_target",
    "load_goal_conditioned_prefix_proposer",
    "proposer_checkpoint_payload",
]
