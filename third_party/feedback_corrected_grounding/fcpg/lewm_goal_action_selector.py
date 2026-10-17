"""CPU diagnostics for selecting an action from frozen LeWM forecasts.

The selector deliberately has no learned parameters.  It scores each frozen
LeWM candidate forecast by its distance to an explicit goal and reports how
well that ranking agrees with the recorded future.  The goal can be a useful
development oracle, but it must be replaced by a deployable goal input before
this becomes a controller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _finite_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.ndim}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _validate_forecast_inputs(
    forecast: np.ndarray,
    target: np.ndarray,
    goal_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = _finite_array(forecast, ndim=5, name="forecast")
    actual = _finite_array(target, ndim=5, name="target")
    goal = _finite_array(goal_px, ndim=2, name="goal_px")
    if predicted.shape != actual.shape:
        raise ValueError(
            "forecast and target must have shape [N,C,H,A,2] and match, "
            f"got {predicted.shape} and {actual.shape}"
        )
    if predicted.shape[-1] != 2:
        raise ValueError("forecast coordinates must have final dimension 2")
    if goal.shape != (predicted.shape[0], 2):
        raise ValueError(
            f"goal_px must have shape {(predicted.shape[0], 2)}, got {goal.shape}"
        )
    if predicted.shape[1] < 2 or predicted.shape[2] < 1:
        raise ValueError("forecasts need at least two candidates and one horizon")
    return predicted.astype(np.float64), actual.astype(np.float64), goal.astype(
        np.float64
    )


def goal_distance_cost(
    forecast: np.ndarray,
    goal_px: np.ndarray,
    *,
    horizon_index: int = -1,
    anchor_index: int | None = 2,
) -> np.ndarray:
    """Return one lower-is-better goal distance per ``[snapshot,candidate]``.

    ``forecast`` has shape ``[N,C,H,A,2]``.  By default the tool-center anchor
    (index 2) at the final requested horizon is scored.  Passing
    ``anchor_index=None`` averages the distance over all anchors and is useful
    as a sensitivity diagnostic.
    """

    values = _finite_array(forecast, ndim=5, name="forecast")
    goals = _finite_array(goal_px, ndim=2, name="goal_px")
    if values.shape[-1] != 2 or goals.shape != (values.shape[0], 2):
        raise ValueError("forecast and goal_px have incompatible coordinate shapes")
    horizon = int(horizon_index)
    if horizon < 0:
        horizon += int(values.shape[2])
    if not 0 <= horizon < values.shape[2]:
        raise ValueError("horizon_index is outside the forecast horizon")
    selected = values[:, :, horizon]
    if anchor_index is None:
        distances = np.linalg.norm(selected - goals[:, None, None, :], axis=-1)
        return np.mean(distances, axis=-1, dtype=np.float64)
    anchor = int(anchor_index)
    if not 0 <= anchor < values.shape[3]:
        raise ValueError("anchor_index is outside the anchor dimension")
    return np.linalg.norm(selected[:, :, anchor] - goals[:, None, :], axis=-1)


def world_goal_distance_cost(
    forecast: np.ndarray,
    goal_world: np.ndarray,
    *,
    horizon_index: int = -1,
    anchor_index: int = 2,
) -> np.ndarray:
    """Return world-space goal distance for ``[N,C,H,A,3]`` forecasts."""

    values = _finite_array(forecast, ndim=5, name="forecast")
    goals = _finite_array(goal_world, ndim=2, name="goal_world")
    if values.shape[-1] != 3 or goals.shape != (values.shape[0], 3):
        raise ValueError("forecast and goal_world have incompatible coordinate shapes")
    horizon = int(horizon_index)
    if horizon < 0:
        horizon += int(values.shape[2])
    if not 0 <= horizon < values.shape[2]:
        raise ValueError("horizon_index is outside the forecast horizon")
    anchor = int(anchor_index)
    if not 0 <= anchor < values.shape[3]:
        raise ValueError("anchor_index is outside the anchor dimension")
    return np.linalg.norm(
        values[:, :, horizon, anchor] - goals[:, None], axis=-1
    )


def integrated_cartesian_action_forecast(
    current_anchor_world: np.ndarray,
    future_action: np.ndarray,
    *,
    action_scale_m: float = 0.01,
) -> np.ndarray:
    """Integrate commanded Cartesian deltas as a kinematic ranking baseline.

    MetaWorld applies each raw XYZ action as a mocap displacement scaled by
    ``0.01 m``. This baseline deliberately ignores servo lag, contact and
    workspace clipping; those are precisely the effects a learned dynamics
    model must improve upon.
    """

    current = _finite_array(
        current_anchor_world, ndim=3, name="current_anchor_world"
    ).astype(np.float64)
    action = _finite_array(future_action, ndim=4, name="future_action").astype(
        np.float64
    )
    if current.shape[-1] != 3:
        raise ValueError("current anchors must have shape [N,A,3]")
    if action.shape[0] != current.shape[0] or action.shape[-1] % 4 != 0:
        raise ValueError("future actions must align as [N,C,H,4*raw_steps]")
    raw_steps = action.shape[-1] // 4
    if raw_steps <= 0:
        raise ValueError("future actions must contain at least one raw step")
    scale = float(action_scale_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("action_scale_m must be positive and finite")
    raw = action.reshape(*action.shape[:-1], raw_steps, 4)
    chunk_delta = scale * raw[..., :3].sum(axis=-2)
    cumulative = np.cumsum(chunk_delta, axis=2)
    return (
        current[:, None, None, :, :]
        + cumulative[:, :, :, None, :]
    ).astype(np.float32)


def average_costs_by_physical_snapshot(
    predicted_cost: np.ndarray,
    actual_cost: np.ndarray,
    physical_snapshot: np.ndarray,
    *,
    actual_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average view-dependent predictions without duplicating physical labels."""

    predicted = _finite_array(predicted_cost, ndim=2, name="predicted_cost")
    actual = _finite_array(actual_cost, ndim=2, name="actual_cost")
    identity = _finite_array(
        physical_snapshot, ndim=1, name="physical_snapshot"
    )
    if predicted.shape != actual.shape or identity.shape[0] != predicted.shape[0]:
        raise ValueError("cost rows and physical snapshot identities must align")
    tolerance = float(actual_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("actual_tolerance must be finite and non-negative")
    unique = np.unique(identity)
    predicted_rows: list[np.ndarray] = []
    actual_rows: list[np.ndarray] = []
    for value in unique:
        selected = identity == value
        labels = actual[selected]
        reference = labels[0]
        if float(np.max(np.abs(labels - reference))) > tolerance:
            raise ValueError("physical snapshot views have inconsistent actual costs")
        predicted_rows.append(np.mean(predicted[selected], axis=0, dtype=np.float64))
        actual_rows.append(np.mean(labels, axis=0, dtype=np.float64))
    return np.stack(predicted_rows), np.stack(actual_rows), unique


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("ranking requires a one-dimensional array of size >= 2")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a tie-aware Spearman correlation without SciPy."""

    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum())
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def select_lowest_cost(cost: np.ndarray) -> np.ndarray:
    """Select the lowest-cost candidate with deterministic first-slot ties."""

    values = _finite_array(cost, ndim=2, name="cost")
    if values.shape[1] < 2:
        raise ValueError("cost must contain at least two candidates")
    return np.argmin(values, axis=1).astype(np.int64)


def selector_metrics(
    predicted_cost: np.ndarray,
    actual_cost: np.ndarray,
    *,
    expert_index: int | None = None,
) -> dict[str, Any]:
    """Summarize ranking and selected-action regret for one method."""

    predicted = _finite_array(predicted_cost, ndim=2, name="predicted_cost")
    actual = _finite_array(actual_cost, ndim=2, name="actual_cost")
    if predicted.shape != actual.shape:
        raise ValueError("predicted_cost and actual_cost must have the same shape")
    if predicted.shape[1] < 2:
        raise ValueError("selector metrics need at least two candidates")
    expert: int | None = None
    if expert_index is not None:
        expert = int(expert_index)
        if not 0 <= expert < predicted.shape[1]:
            raise ValueError("expert_index is outside the candidate dimension")

    selected = select_lowest_cost(predicted)
    row = np.arange(predicted.shape[0])
    selected_actual = actual[row, selected]
    best_actual = np.min(actual, axis=1)
    rank_values = np.asarray(
        [spearman_rank_correlation(p, a) for p, a in zip(predicted, actual)],
        dtype=np.float64,
    )
    actual_spread = np.ptp(actual, axis=1)
    selected_regret = selected_actual - best_actual
    normalized_selected = selected_regret / np.maximum(actual_spread, 1.0e-8)
    result: dict[str, Any] = {
        "row_count": int(predicted.shape[0]),
        "candidate_count": int(predicted.shape[1]),
        "spearman_mean": float(np.mean(rank_values)),
        "spearman_median": float(np.median(rank_values)),
        "top1_accuracy": float(np.mean(selected == np.argmin(actual, axis=1))),
        "top2_accuracy": float(
            np.mean(
                np.any(
                    np.argsort(predicted, axis=1)[:, :2]
                    == np.argmin(actual, axis=1)[:, None],
                    axis=1,
                )
            )
        ),
        "top3_accuracy": float(
            np.mean(
                np.any(
                    np.argsort(predicted, axis=1)[:, :3]
                    == np.argmin(actual, axis=1)[:, None],
                    axis=1,
                )
            )
        ),
        "selected_cost_mean": float(np.mean(selected_actual)),
        "oracle_best_cost_mean": float(np.mean(best_actual)),
        "top1_regret_mean": float(np.mean(selected_regret)),
        "top1_regret_median": float(np.median(selected_regret)),
        "normalized_top1_regret_mean": float(np.mean(normalized_selected)),
        "within_0_25px_rate": float(np.mean(selected_regret <= 0.25)),
        "within_0_5px_rate": float(np.mean(selected_regret <= 0.5)),
        "within_1px_rate": float(np.mean(selected_regret <= 1.0)),
        "within_2px_rate": float(np.mean(selected_regret <= 2.0)),
        "random_top1_accuracy": 1.0 / float(predicted.shape[1]),
        "actual_candidate_spread_mean": float(np.mean(actual_spread)),
        "selected_candidate_indices": selected.tolist(),
    }
    if expert is not None:
        expert_actual = actual[:, expert]
        expert_regret = expert_actual - best_actual
        normalized_expert = expert_regret / np.maximum(actual_spread, 1.0e-8)
        result.update(
            {
                "expert_index": expert,
                "expert_regret_mean": float(np.mean(expert_regret)),
                "expert_normalized_regret_mean": float(np.mean(normalized_expert)),
            }
        )
    else:
        result["expert_baseline_available"] = False
    return result


def grouped_selector_metrics(
    predicted_cost: np.ndarray,
    actual_cost: np.ndarray,
    groups: Mapping[str, Sequence[Any]],
    *,
    expert_index: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute the same metrics for each non-empty metadata group."""

    predicted = np.asarray(predicted_cost)
    actual = np.asarray(actual_cost)
    if predicted.ndim != 2 or actual.shape != predicted.shape:
        raise ValueError("grouped costs must be matching [N,C] arrays")
    result: dict[str, dict[str, Any]] = {}
    for name, values in groups.items():
        labels = np.asarray(values).astype(str)
        if labels.shape != (predicted.shape[0],):
            raise ValueError(f"group {name} must align with cost rows")
        for label in sorted(set(labels.tolist())):
            selected = labels == label
            result[f"{name}={label}"] = selector_metrics(
                predicted[selected], actual[selected], expert_index=expert_index
            )
    return result


__all__ = [
    "average_costs_by_physical_snapshot",
    "goal_distance_cost",
    "grouped_selector_metrics",
    "integrated_cartesian_action_forecast",
    "select_lowest_cost",
    "selector_metrics",
    "spearman_rank_correlation",
    "world_goal_distance_cost",
]
