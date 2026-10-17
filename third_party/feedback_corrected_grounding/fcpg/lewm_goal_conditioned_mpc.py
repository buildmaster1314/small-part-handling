"""Causal candidate construction and scoring for goal-conditioned LeWM MPC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


ACTION_DIM = 4
RAW_STEPS_PER_TOKEN = 5
HORIZON = 8
HISTORY_TOKENS = 2
CANDIDATE_COUNT = 8

CANDIDATE_NAMES = (
    "center",
    "x_positive",
    "x_negative",
    "y_positive",
    "y_negative",
    "z_positive",
    "z_negative",
    "random_smooth",
)


def _finite_action_bounds(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 0:
        result = np.full(ACTION_DIM, float(result), dtype=np.float32)
    if result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be scalar or one finite 4-vector")
    return result


def repeated_action_prefix(
    action: np.ndarray,
    *,
    action_low: Any = -1.0,
    action_high: Any = 1.0,
) -> np.ndarray:
    """Repeat one currently available action over the eight-token horizon."""

    current = np.asarray(action, dtype=np.float32)
    if current.shape != (ACTION_DIM,) or not np.isfinite(current).all():
        raise ValueError("action must be one finite 4-vector")
    low = _finite_action_bounds(action_low, name="action_low")
    high = _finite_action_bounds(action_high, name="action_high")
    if np.any(low >= high):
        raise ValueError("action_low must be strictly below action_high")
    if np.any(current < low) or np.any(current > high):
        raise ValueError("action lies outside the action bounds")
    return np.ascontiguousarray(
        np.broadcast_to(
            current,
            (HORIZON * RAW_STEPS_PER_TOKEN, ACTION_DIM),
        ),
        dtype=np.float32,
    )


def build_smooth_candidate_bank(
    center_prefix: np.ndarray,
    rng: np.random.Generator,
    *,
    amplitude: float = 0.25,
    action_low: Any = -1.0,
    action_high: Any = 1.0,
) -> np.ndarray:
    """Build the same eight-way smooth perturbation family used for training.

    Candidate zero is the causal proposal. The remaining candidates perturb
    only Cartesian commands; the proposal's gripper command is preserved.
    """

    center = np.asarray(center_prefix, dtype=np.float32)
    raw_steps = HORIZON * RAW_STEPS_PER_TOKEN
    if center.shape != (raw_steps, ACTION_DIM) or not np.isfinite(center).all():
        raise ValueError(f"center_prefix must have shape [{raw_steps},4]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    scale = float(amplitude)
    if not np.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("amplitude must be finite and in (0,1]")
    low = _finite_action_bounds(action_low, name="action_low")
    high = _finite_action_bounds(action_high, name="action_high")
    if np.any(low >= high):
        raise ValueError("action_low must be strictly below action_high")
    if np.any(center < low) or np.any(center > high):
        raise ValueError("center_prefix lies outside the action bounds")

    actions = np.repeat(center[None], CANDIDATE_COUNT, axis=0)
    directions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    knot_x = np.linspace(0.0, 1.0, 5)
    query_x = np.linspace(0.0, 1.0, raw_steps)
    for candidate in range(1, CANDIDATE_COUNT):
        if candidate <= len(directions):
            knots = np.repeat(directions[candidate - 1][None], 5, axis=0)
            knots += 0.15 * rng.normal(size=knots.shape).astype(np.float32)
        else:
            knots = rng.normal(size=(5, 3)).astype(np.float32)
        knots /= max(float(np.sqrt(np.mean(knots * knots))), 1.0e-6)
        noise = np.stack(
            [
                np.interp(query_x, knot_x, knots[:, axis])
                for axis in range(3)
            ],
            axis=-1,
        ).astype(np.float32)
        actions[candidate, :, :3] = np.clip(
            actions[candidate, :, :3] + scale * noise,
            low[:3],
            high[:3],
        )
    return np.ascontiguousarray(
        actions.reshape(CANDIDATE_COUNT, HORIZON, -1),
        dtype=np.float32,
    )


def causal_action_history(actions: Sequence[np.ndarray]) -> np.ndarray:
    """Return the last ten executed raw actions as two left-padded tokens."""

    values = np.asarray(list(actions), dtype=np.float32)
    if values.size == 0:
        values = np.empty((0, ACTION_DIM), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError("actions must form a [T,4] array")
    if not np.isfinite(values).all():
        raise ValueError("actions must be finite")
    raw_count = HISTORY_TOKENS * RAW_STEPS_PER_TOKEN
    padded = np.zeros((raw_count, ACTION_DIM), dtype=np.float32)
    available = values[-raw_count:]
    if len(available):
        padded[-len(available) :] = available
    return np.ascontiguousarray(padded.reshape(HISTORY_TOKENS, -1))


def predicted_goal_cost(
    predicted_entity_xyz: np.ndarray,
    current_goal_xyz: np.ndarray,
    current_goal_velocity: np.ndarray,
    *,
    horizon_tokens: int = HORIZON,
    terminal_weight: float = 1.0,
    path_weight: float = 0.15,
) -> np.ndarray:
    """Score candidates against a constant-velocity goal forecast."""

    predicted = np.asarray(predicted_entity_xyz, dtype=np.float32)
    if predicted.ndim != 3 or predicted.shape[1:] != (HORIZON, 3):
        raise ValueError("predicted_entity_xyz must have shape [C,8,3]")
    goal = np.asarray(current_goal_xyz, dtype=np.float32)
    velocity = np.asarray(current_goal_velocity, dtype=np.float32)
    if goal.shape != (3,) or velocity.shape != (3,):
        raise ValueError("goal and velocity must each have shape [3]")
    if not np.isfinite(predicted).all() or not np.isfinite(goal).all() or not np.isfinite(velocity).all():
        raise ValueError("goal-cost inputs must be finite")
    terminal = float(terminal_weight)
    path = float(path_weight)
    horizon = int(horizon_tokens)
    if not 1 <= horizon <= HORIZON:
        raise ValueError(f"horizon_tokens must be in 1..{HORIZON}")
    if not np.isfinite([terminal, path]).all() or terminal <= 0.0 or path < 0.0:
        raise ValueError("terminal_weight must be positive and path_weight non-negative")
    raw_offsets = RAW_STEPS_PER_TOKEN * np.arange(1, horizon + 1, dtype=np.float32)
    future_goal = goal[None] + raw_offsets[:, None] * velocity[None]
    distance = np.linalg.norm(predicted[:, :horizon] - future_goal[None], axis=-1)
    return terminal * distance[:, -1] + path * distance.mean(axis=1)


def selected_raw_chunk(candidate_action: np.ndarray, candidate_index: int) -> np.ndarray:
    """Return the first five raw actions from one selected candidate."""

    candidates = np.asarray(candidate_action, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[1:] != (HORIZON, 20):
        raise ValueError("candidate_action must have shape [C,8,20]")
    index = int(candidate_index)
    if not 0 <= index < candidates.shape[0]:
        raise ValueError("candidate_index is outside the candidate bank")
    return np.ascontiguousarray(
        candidates[index, 0].reshape(RAW_STEPS_PER_TOKEN, ACTION_DIM)
    )


__all__ = [
    "ACTION_DIM",
    "CANDIDATE_COUNT",
    "CANDIDATE_NAMES",
    "HORIZON",
    "RAW_STEPS_PER_TOKEN",
    "build_smooth_candidate_bank",
    "causal_action_history",
    "predicted_goal_cost",
    "repeated_action_prefix",
    "selected_raw_chunk",
]
