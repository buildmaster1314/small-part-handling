"""Causal action proposals and costs for Towel receding-horizon control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fcpg.lewm_goal_action_selector import (
    integrated_cartesian_action_forecast,
    world_goal_distance_cost,
)


ACTION_DIM = 4
HORIZON = 8
RAW_STEPS_PER_MODEL_STEP = 5
HISTORY_CHUNKS = 2
TOOL_CENTER_ANCHOR = 2
PRIMARY_HORIZON_INDEX = 5
ACTION_SCALE_M = 0.01

CANDIDATE_NAMES = (
    "center",
    "reverse",
    "hold",
    "half",
    "transverse_u_positive",
    "transverse_u_negative",
    "transverse_v_positive",
    "transverse_v_negative",
)

POLICY_CANDIDATE_NAMES = (
    "policy",
    "reverse",
    "hold",
    "half",
    "fast",
    "quarter",
    "reverse_half",
    "three_quarter",
)


def _finite_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite {size}-vector")
    return result


def _action_bound(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(ACTION_DIM, float(result), dtype=np.float64)
    if result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be scalar or one finite 4-vector")
    return result


def build_goal_centered_prefix(
    current_tool_world: np.ndarray,
    goal_tool_world: np.ndarray,
    *,
    gripper_command: float,
    action_low: Any = -1.0,
    action_high: Any = 1.0,
    gain: float = 18.0,
    action_scale_m: float = ACTION_SCALE_M,
) -> np.ndarray:
    """Build a causal geometric action prefix from the current tool and goal.

    The virtual rollout uses only the current tool position and current task
    endpoint. It does not query future observations or step a simulator.
    """

    current = _finite_vector(current_tool_world, size=3, name="current_tool_world")
    goal = _finite_vector(goal_tool_world, size=3, name="goal_tool_world")
    low = _action_bound(action_low, name="action_low")
    high = _action_bound(action_high, name="action_high")
    if np.any(low >= high):
        raise ValueError("action_low must be strictly below action_high")
    controller_gain = float(gain)
    scale = float(action_scale_m)
    grip = float(gripper_command)
    if not np.isfinite(controller_gain) or controller_gain <= 0.0:
        raise ValueError("gain must be positive and finite")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("action_scale_m must be positive and finite")
    if not np.isfinite(grip):
        raise ValueError("gripper_command must be finite")

    raw_count = HORIZON * RAW_STEPS_PER_MODEL_STEP
    result = np.empty((raw_count, ACTION_DIM), dtype=np.float64)
    virtual_tool = current.copy()
    for raw_step in range(raw_count):
        xyz = np.clip(
            controller_gain * (goal - virtual_tool),
            low[:3],
            high[:3],
        )
        result[raw_step, :3] = xyz
        result[raw_step, 3] = np.clip(grip, low[3], high[3])
        virtual_tool += scale * xyz
    return np.ascontiguousarray(result, dtype=np.float32)


def build_policy_repeated_prefix(
    current_action: np.ndarray,
    *,
    action_low: Any = -1.0,
    action_high: Any = 1.0,
) -> np.ndarray:
    """Repeat only the currently available policy action over the horizon."""

    action = _finite_vector(current_action, size=ACTION_DIM, name="current_action")
    low = _action_bound(action_low, name="action_low")
    high = _action_bound(action_high, name="action_high")
    if np.any(low >= high):
        raise ValueError("action_low must be strictly below action_high")
    if np.any(action < low) or np.any(action > high):
        raise ValueError("current_action lies outside the action bounds")
    return np.ascontiguousarray(
        np.broadcast_to(
            action,
            (HORIZON * RAW_STEPS_PER_MODEL_STEP, ACTION_DIM),
        ),
        dtype=np.float32,
    )


def build_policy_centered_towel_candidates(
    policy_prefix: np.ndarray,
    *,
    action_low: Any = -1.0,
    action_high: Any = 1.0,
) -> np.ndarray:
    """Build horizontal Towel proposals without reading the task goal.

    Z motion and the gripper command always remain equal to the current policy
    prefix. Only horizontal speed and direction vary, so candidate ranking does
    not get a hand-coded target direction or create an avoidable lane change.
    """

    center = np.asarray(policy_prefix, dtype=np.float64)
    expected = (HORIZON * RAW_STEPS_PER_MODEL_STEP, ACTION_DIM)
    if center.shape != expected or not np.isfinite(center).all():
        raise ValueError(f"policy_prefix must have shape {expected} and be finite")
    low = _action_bound(action_low, name="action_low")
    high = _action_bound(action_high, name="action_high")
    if np.any(low >= high):
        raise ValueError("action_low must be strictly below action_high")
    if np.any(center < low) or np.any(center > high):
        raise ValueError("policy_prefix lies outside the action bounds")

    xy = center[:, :2]
    candidates = np.broadcast_to(
        center[None], (len(POLICY_CANDIDATE_NAMES), *center.shape)
    ).copy()
    candidates[0, :, :2] = xy
    candidates[1, :, :2] = -xy
    candidates[2, :, :2] = 0.0
    candidates[3, :, :2] = 0.50 * xy
    candidates[4, :, :2] = 1.25 * xy
    candidates[5, :, :2] = 0.25 * xy
    candidates[6, :, :2] = -0.50 * xy
    candidates[7, :, :2] = 0.75 * xy
    candidates = np.clip(candidates, low[None, None], high[None, None])
    return np.ascontiguousarray(
        candidates.reshape(len(POLICY_CANDIDATE_NAMES), HORIZON, -1),
        dtype=np.float32,
    )


def build_towel_candidate_prefixes(
    center_action: np.ndarray,
    *,
    action_low: Any = -1.0,
    action_high: Any = 1.0,
) -> np.ndarray:
    """Return eight deterministic candidates around one causal center prefix."""

    center = np.asarray(center_action, dtype=np.float64)
    expected = (HORIZON * RAW_STEPS_PER_MODEL_STEP, ACTION_DIM)
    if center.shape != expected or not np.isfinite(center).all():
        raise ValueError(f"center_action must have shape {expected} and be finite")
    low = _action_bound(action_low, name="action_low")
    high = _action_bound(action_high, name="action_high")
    if np.any(low >= high):
        raise ValueError("action_low must be strictly below action_high")
    if np.any(center < low) or np.any(center > high):
        raise ValueError("center_action lies outside the action bounds")

    xyz = center[:, :3]
    transverse_u = np.empty_like(xyz)
    transverse_v = np.empty_like(xyz)
    for index, direction in enumerate(xyz):
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= np.finfo(np.float64).eps:
            transverse_u[index] = 0.0
            transverse_v[index] = 0.0
            continue
        normal = direction / magnitude
        reference = np.zeros(3, dtype=np.float64)
        reference[int(np.argmin(np.abs(normal)))] = 1.0
        first = np.cross(normal, reference)
        first /= np.linalg.norm(first)
        second = np.cross(normal, first)
        second /= np.linalg.norm(second)
        transverse_u[index] = magnitude * first
        transverse_v[index] = magnitude * second
    candidates = np.broadcast_to(
        center[None], (len(CANDIDATE_NAMES), *center.shape)
    ).copy()
    candidates[0, :, :3] = xyz
    candidates[1, :, :3] = -xyz
    candidates[2, :, :3] = 0.0
    candidates[3, :, :3] = 0.50 * xyz
    candidates[4, :, :3] = transverse_u
    candidates[5, :, :3] = -transverse_u
    candidates[6, :, :3] = transverse_v
    candidates[7, :, :3] = -transverse_v
    candidates = np.clip(candidates, low[None, None], high[None, None])
    return np.ascontiguousarray(
        candidates.reshape(len(CANDIDATE_NAMES), HORIZON, -1),
        dtype=np.float32,
    )


def causal_history_chunks(actions: Sequence[np.ndarray]) -> np.ndarray:
    """Return the last ten executed raw actions as two left-padded chunks."""

    values = np.asarray(list(actions), dtype=np.float32)
    if values.size == 0:
        values = np.empty((0, ACTION_DIM), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError("actions must form a [T,4] array")
    if not np.isfinite(values).all():
        raise ValueError("actions must be finite")
    raw_count = HISTORY_CHUNKS * RAW_STEPS_PER_MODEL_STEP
    result = np.zeros((raw_count, ACTION_DIM), dtype=np.float32)
    available = values[-raw_count:]
    if len(available):
        result[-len(available) :] = available
    return np.ascontiguousarray(result.reshape(HISTORY_CHUNKS, -1))


def lewm_world_goal_cost(
    predicted_world_xyz: np.ndarray,
    goal_tool_world: np.ndarray,
    *,
    horizon_index: int = PRIMARY_HORIZON_INDEX,
) -> np.ndarray:
    """Score one candidate bank from LeWM world-coordinate forecasts."""

    predicted = np.asarray(predicted_world_xyz, dtype=np.float32)
    if predicted.ndim != 4 or predicted.shape[2:] != (3, 3):
        raise ValueError("predicted_world_xyz must have shape [C,H,3,3]")
    goal = _finite_vector(goal_tool_world, size=3, name="goal_tool_world")
    return world_goal_distance_cost(
        predicted[None],
        goal.astype(np.float32)[None],
        horizon_index=int(horizon_index),
        anchor_index=TOOL_CENTER_ANCHOR,
    )[0]


def tool_xy_goal_cost(
    predicted_world_xyz: np.ndarray,
    goal_tool_world: np.ndarray,
    *,
    horizon_index: int = PRIMARY_HORIZON_INDEX,
) -> np.ndarray:
    """Score the tool-center XY endpoint for one candidate bank."""

    predicted = np.asarray(predicted_world_xyz, dtype=np.float32)
    if predicted.ndim != 4 or predicted.shape[2:] != (3, 3):
        raise ValueError("predicted_world_xyz must have shape [C,H,3,3]")
    horizon = int(horizon_index)
    if horizon < 0:
        horizon += int(predicted.shape[1])
    if not 0 <= horizon < predicted.shape[1]:
        raise ValueError("horizon_index is outside the forecast horizon")
    goal = _finite_vector(goal_tool_world, size=3, name="goal_tool_world")
    return np.linalg.norm(
        predicted[:, horizon, TOOL_CENTER_ANCHOR, :2] - goal[None, :2],
        axis=-1,
    )


def tool_xy_goal_reach_cost(
    predicted_world_xyz: np.ndarray,
    goal_tool_world: np.ndarray,
    *,
    horizon_index: int = PRIMARY_HORIZON_INDEX,
    terminal_weight: float = 0.05,
) -> np.ndarray:
    """Score whether the tool reaches the XY goal at any step up to a horizon."""

    predicted = np.asarray(predicted_world_xyz, dtype=np.float32)
    if predicted.ndim != 4 or predicted.shape[2:] != (3, 3):
        raise ValueError("predicted_world_xyz must have shape [C,H,3,3]")
    horizon = int(horizon_index)
    if horizon < 0:
        horizon += int(predicted.shape[1])
    if not 0 <= horizon < predicted.shape[1]:
        raise ValueError("horizon_index is outside the forecast horizon")
    weight = float(terminal_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("terminal_weight must be finite and non-negative")
    goal = _finite_vector(goal_tool_world, size=3, name="goal_tool_world")
    distance = np.linalg.norm(
        predicted[:, : horizon + 1, TOOL_CENTER_ANCHOR, :2]
        - goal[None, None, :2],
        axis=-1,
    )
    return np.min(distance, axis=1) + weight * distance[:, horizon]


def kinematic_world_goal_cost(
    current_anchor_world: np.ndarray,
    candidate_action: np.ndarray,
    goal_tool_world: np.ndarray,
    *,
    horizon_index: int = PRIMARY_HORIZON_INDEX,
) -> np.ndarray:
    """Score the same bank using command integration without learned dynamics."""

    current = np.asarray(current_anchor_world, dtype=np.float32)
    candidates = np.asarray(candidate_action, dtype=np.float32)
    if current.shape != (3, 3):
        raise ValueError("current_anchor_world must have shape [3,3]")
    if candidates.ndim != 3 or candidates.shape[1:] != (HORIZON, 20):
        raise ValueError("candidate_action must have shape [C,8,20]")
    forecast = integrated_cartesian_action_forecast(
        current[None], candidates[None], action_scale_m=ACTION_SCALE_M
    )
    goal = _finite_vector(goal_tool_world, size=3, name="goal_tool_world")
    return world_goal_distance_cost(
        forecast,
        goal.astype(np.float32)[None],
        horizon_index=int(horizon_index),
        anchor_index=TOOL_CENTER_ANCHOR,
    )[0]


def center_calibrated_world_forecast(
    predicted_world_xyz: np.ndarray,
    current_anchor_world: np.ndarray,
    candidate_action: np.ndarray,
    *,
    center_index: int = 0,
) -> np.ndarray:
    """Remove common absolute-readout bias using the analytic center forecast.

    The calibrated forecast keeps LeWM's candidate-relative dynamics while
    forcing the designated center candidate to equal the causal kinematic
    baseline at every horizon. No future observation is used.
    """

    predicted = np.asarray(predicted_world_xyz, dtype=np.float32)
    current = np.asarray(current_anchor_world, dtype=np.float32)
    candidates = np.asarray(candidate_action, dtype=np.float32)
    if predicted.ndim != 4 or predicted.shape[1:] != (HORIZON, 3, 3):
        raise ValueError("predicted_world_xyz must have shape [C,8,3,3]")
    if current.shape != (3, 3):
        raise ValueError("current_anchor_world must have shape [3,3]")
    if candidates.shape != (predicted.shape[0], HORIZON, 20):
        raise ValueError("candidate_action must align with predicted candidates")
    index = int(center_index)
    if not 0 <= index < predicted.shape[0]:
        raise ValueError("center_index is outside the candidate bank")
    kinematic = integrated_cartesian_action_forecast(
        current[None], candidates[None], action_scale_m=ACTION_SCALE_M
    )[0]
    common_bias = predicted[index] - kinematic[index]
    return np.ascontiguousarray(predicted - common_bias[None], dtype=np.float32)


def selected_raw_chunk(candidate_action: np.ndarray, candidate_index: int) -> np.ndarray:
    """Return only the first five raw actions from one selected prefix."""

    candidates = np.asarray(candidate_action, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[1:] != (HORIZON, 20):
        raise ValueError("candidate_action must have shape [C,8,20]")
    index = int(candidate_index)
    if not 0 <= index < candidates.shape[0]:
        raise ValueError("candidate_index is outside the candidate bank")
    return np.ascontiguousarray(
        candidates[index, 0].reshape(RAW_STEPS_PER_MODEL_STEP, ACTION_DIM)
    )


__all__ = [
    "ACTION_SCALE_M",
    "CANDIDATE_NAMES",
    "POLICY_CANDIDATE_NAMES",
    "PRIMARY_HORIZON_INDEX",
    "build_goal_centered_prefix",
    "build_policy_centered_towel_candidates",
    "build_policy_repeated_prefix",
    "build_towel_candidate_prefixes",
    "causal_history_chunks",
    "center_calibrated_world_forecast",
    "kinematic_world_goal_cost",
    "lewm_world_goal_cost",
    "selected_raw_chunk",
    "tool_xy_goal_cost",
    "tool_xy_goal_reach_cost",
]
