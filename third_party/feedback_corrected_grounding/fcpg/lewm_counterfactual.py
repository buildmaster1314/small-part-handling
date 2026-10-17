"""Counterfactual action-candidate utilities for grounded LeWM evaluation.

The helpers in this module are environment independent.  They construct a
fixed candidate set around one expert action chunk and evaluate whether a
world model preserves the ordering of the candidates' observed outcomes.
Lower costs are always better.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np


PILOT_RAW_ACTION_STEPS = 25
PILOT_ACTION_DIM = 4
PILOT_TRANSITIONS = 5
PILOT_TRANSITION_ACTION_DIM = 20
PILOT_HORIZONS = (1, 3, 5)
PILOT_NOISE_SCALES = (0.05, 0.15, 0.30)
PILOT_CANDIDATE_NAMES = (
    "expert",
    "policy",
    "expert_noise_0.05",
    "expert_noise_0.15",
    "expert_noise_0.3",
    "same_task_shuffled",
    "random_0",
)


@dataclass(frozen=True)
class ActionCandidates:
    """Named raw-action chunks with shape ``[C, H, A]``."""

    names: tuple[str, ...]
    actions: np.ndarray
    expert_index: int

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions)
        if actions.ndim != 3:
            raise ValueError("candidate actions must have shape [C, H, A]")
        if actions.shape[0] != len(self.names):
            raise ValueError("candidate names and action rows differ")
        if not 0 <= self.expert_index < len(self.names):
            raise ValueError("expert_index is outside the candidate set")
        if len(set(self.names)) != len(self.names):
            raise ValueError("candidate names must be unique")
        if not np.isfinite(actions).all():
            raise ValueError("candidate actions must be finite")


def _action_bounds(
    value: float | Sequence[float] | np.ndarray,
    action_dim: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 0:
        result = np.full(action_dim, float(result), dtype=np.float32)
    if result.shape != (action_dim,):
        raise ValueError(f"{name} must be scalar or have shape {(action_dim,)}")
    return result


def build_action_candidates(
    expert_action: np.ndarray,
    *,
    policy_action: np.ndarray,
    shuffled_action: np.ndarray,
    rng: np.random.Generator,
    noise_scales: Sequence[float] = (0.05, 0.15, 0.30),
    random_count: int = 1,
    action_low: float | Sequence[float] | np.ndarray = -1.0,
    action_high: float | Sequence[float] | np.ndarray = 1.0,
) -> ActionCandidates:
    """Build the preregistered expert/policy/local/shuffled candidate family.

    ``expert_action``, ``policy_action`` and ``shuffled_action`` are raw action
    chunks, not flattened LeWM transition vectors.  Local perturbations keep
    the expert gripper command (the final action dimension) unchanged so the
    diagnostic primarily tests motion consequences rather than arbitrary
    gripper flips.
    """

    expert = np.asarray(expert_action, dtype=np.float32)
    policy = np.asarray(policy_action, dtype=np.float32)
    shuffled = np.asarray(shuffled_action, dtype=np.float32)
    if expert.ndim != 2:
        raise ValueError("expert_action must have shape [H, A]")
    if policy.shape != expert.shape or shuffled.shape != expert.shape:
        raise ValueError("policy and shuffled chunks must match expert_action")
    if random_count < 0:
        raise ValueError("random_count must be non-negative")
    scales = tuple(float(value) for value in noise_scales)
    if any(not np.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("noise scales must be finite and positive")
    if len(set(scales)) != len(scales):
        raise ValueError("noise scales must be unique")

    action_dim = int(expert.shape[1])
    low = _action_bounds(action_low, action_dim, "action_low")
    high = _action_bounds(action_high, action_dim, "action_high")
    if bool(np.any(low >= high)):
        raise ValueError("every action_low value must be below action_high")

    rows = [np.clip(expert, low, high), np.clip(policy, low, high)]
    names = ["expert", "policy"]
    for scale in scales:
        noise = rng.normal(0.0, scale, size=expert.shape).astype(np.float32)
        if action_dim > 1:
            noise[:, -1] = 0.0
        rows.append(np.clip(expert + noise, low, high))
        names.append(f"expert_noise_{scale:g}")
    rows.append(np.clip(shuffled, low, high))
    names.append("same_task_shuffled")
    for index in range(random_count):
        rows.append(rng.uniform(low, high, size=expert.shape).astype(np.float32))
        names.append(f"random_{index}")
    return ActionCandidates(
        names=tuple(names),
        actions=np.stack(rows).astype(np.float32, copy=False),
        expert_index=0,
    )


def reshape_action_chunks(
    actions: np.ndarray,
    *,
    raw_steps: int = PILOT_RAW_ACTION_STEPS,
    action_dim: int = PILOT_ACTION_DIM,
    transitions: int = PILOT_TRANSITIONS,
) -> np.ndarray:
    """Reshape raw action chunks into fixed-size LeWM transition vectors."""

    value = np.asarray(actions, dtype=np.float32)
    if value.ndim not in {2, 3}:
        raise ValueError("actions must have shape [R, A] or [C, R, A]")
    if value.shape[-2:] != (raw_steps, action_dim):
        raise ValueError(
            f"raw action chunks must end in shape {(raw_steps, action_dim)}, "
            f"got {tuple(value.shape)}"
        )
    if raw_steps % transitions:
        raise ValueError("raw_steps must be divisible by transitions")
    transition_dim = (raw_steps // transitions) * action_dim
    return value.reshape(*value.shape[:-2], transitions, transition_dim)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks with stable handling of exact ties."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("ranking requires a one-dimensional array of size >= 2")
    if not np.isfinite(values).all():
        raise ValueError("ranking values must be finite")
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
    """Compute Spearman correlation without depending on SciPy."""

    left_rank = _average_ranks(np.asarray(left))
    right_rank = _average_ranks(np.asarray(right))
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum())
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _minimum_indices(values: np.ndarray) -> np.ndarray:
    return np.flatnonzero(values == values.min())


def _topk_tie_selection(
    predicted: np.ndarray,
    top_k: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return fixed selections, boundary tie members, and boundary draw count."""

    fixed: list[int] = []
    for value in np.unique(predicted):
        tied = np.flatnonzero(predicted == value).tolist()
        remaining = top_k - len(fixed)
        if remaining <= 0:
            break
        if len(tied) <= remaining:
            fixed.extend(tied)
            continue
        return tuple(fixed), tuple(tied), remaining
    if len(fixed) != top_k:
        raise RuntimeError("failed to resolve the predicted top-k set")
    return tuple(fixed), (), 0


def _expected_topk_hit(
    actual_best: set[int],
    fixed: tuple[int, ...],
    boundary: tuple[int, ...],
    boundary_count: int,
) -> float:
    if actual_best.intersection(fixed):
        return 1.0
    if not boundary:
        return 0.0
    hits = len(actual_best.intersection(boundary))
    if hits == 0:
        return 0.0
    total = 0
    successful = 0
    for selected in combinations(boundary, boundary_count):
        total += 1
        if actual_best.intersection(selected):
            successful += 1
    return float(successful / total)


def _expected_topk_actual_cost(
    actual: np.ndarray,
    fixed: tuple[int, ...],
    boundary: tuple[int, ...],
    boundary_count: int,
) -> float:
    if not boundary:
        return float(actual[list(fixed)].min())
    values = []
    for selected in combinations(boundary, boundary_count):
        indices = (*fixed, *selected)
        values.append(float(actual[list(indices)].min()))
    if not values:
        raise RuntimeError("top-k tie set produced no selections")
    return float(np.mean(values))


def _expert_pairwise_accuracy(
    predicted: np.ndarray,
    actual: np.ndarray,
    expert_index: int,
) -> float:
    scores = []
    for index in range(predicted.size):
        if index == expert_index:
            continue
        predicted_delta = float(predicted[expert_index] - predicted[index])
        actual_delta = float(actual[expert_index] - actual[index])
        if predicted_delta == 0.0 and actual_delta == 0.0:
            scores.append(1.0)
        elif predicted_delta == 0.0 or actual_delta == 0.0:
            scores.append(0.5)
        else:
            scores.append(float(np.sign(predicted_delta) == np.sign(actual_delta)))
    return float(np.mean(scores))


def _validated_outcome(
    value: np.ndarray | None,
    predicted: np.ndarray,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.shape != predicted.shape:
        raise ValueError(f"{name} must share candidate shape [C]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def counterfactual_ranking_metrics(
    predicted_cost: np.ndarray,
    actual_cost: np.ndarray,
    *,
    expert_index: int,
    top_k: int = 3,
    actual_return: np.ndarray | None = None,
    actual_progress: np.ndarray | None = None,
    actual_oracle_progress: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score one same-snapshot candidate family.

    Both inputs use the same frozen goal representation and lower is better.
    ``actual_cost`` is measured from the simulator branch outcome, while
    ``predicted_cost`` is computed from the LeWM-predicted future target.
    """

    predicted = np.asarray(predicted_cost, dtype=np.float64)
    actual = np.asarray(actual_cost, dtype=np.float64)
    if predicted.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("predicted_cost and actual_cost must share shape [C]")
    if predicted.size < 2:
        raise ValueError("at least two candidates are required")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("candidate costs must be finite")
    if not 0 <= expert_index < predicted.size:
        raise ValueError("expert_index is outside the candidate set")
    if not 1 <= top_k <= predicted.size:
        raise ValueError("top_k must be in [1, C]")

    returns = _validated_outcome(actual_return, predicted, "actual_return")
    progress = _validated_outcome(actual_progress, predicted, "actual_progress")
    oracle_progress = _validated_outcome(
        actual_oracle_progress,
        predicted,
        "actual_oracle_progress",
    )

    predicted_best = _minimum_indices(predicted)
    actual_best = _minimum_indices(actual)
    predicted_best_set = set(int(index) for index in predicted_best)
    actual_best_set = set(int(index) for index in actual_best)
    fixed, boundary, boundary_count = _topk_tie_selection(predicted, top_k)
    actual_span = max(float(actual.max() - actual.min()), 1e-12)
    best_actual_cost = float(actual.min())
    expected_top1_actual_cost = float(actual[predicted_best].mean())
    expected_topk_actual_cost = _expected_topk_actual_cost(
        actual,
        fixed,
        boundary,
        boundary_count,
    )
    top1_regret = expected_top1_actual_cost - best_actual_cost
    topk_regret = expected_topk_actual_cost - best_actual_cost
    top1_hit = float(
        len(predicted_best_set.intersection(actual_best_set)) / len(predicted_best_set)
    )
    expert_top1 = (
        float(1.0 / len(predicted_best_set))
        if expert_index in predicted_best_set
        else 0.0
    )
    expert_topk = (
        1.0
        if expert_index in fixed
        else (
            float(boundary_count / len(boundary))
            if expert_index in boundary
            else 0.0
        )
    )
    result: dict[str, Any] = {
        "spearman": spearman_rank_correlation(predicted, actual),
        "predicted_best_index": (
            int(predicted_best[0]) if predicted_best.size == 1 else -1
        ),
        "actual_best_index": int(actual_best[0]) if actual_best.size == 1 else -1,
        "predicted_best_indices": tuple(int(index) for index in predicted_best),
        "actual_best_indices": tuple(int(index) for index in actual_best),
        "predicted_topk_fixed_indices": fixed,
        "predicted_topk_boundary_indices": boundary,
        "predicted_topk_boundary_count": int(boundary_count),
        "top1_hit": top1_hit,
        "topk_hit": _expected_topk_hit(
            actual_best_set,
            fixed,
            boundary,
            boundary_count,
        ),
        "expert_top1": expert_top1,
        "expert_topk": expert_topk,
        "expert_pairwise_accuracy": _expert_pairwise_accuracy(
            predicted,
            actual,
            expert_index,
        ),
        "top1_regret": top1_regret,
        "topk_regret": topk_regret,
        "normalized_top1_regret": top1_regret / actual_span,
        "normalized_topk_regret": topk_regret / actual_span,
        "regret": top1_regret,
        "normalized_regret": top1_regret / actual_span,
        "expected_top1_actual_cost": expected_top1_actual_cost,
        "expected_topk_actual_cost": expected_topk_actual_cost,
    }
    if returns is not None:
        result["return_spearman"] = spearman_rank_correlation(-predicted, returns)
    if progress is not None:
        result["progress_spearman"] = spearman_rank_correlation(-predicted, progress)
    if oracle_progress is not None:
        result["oracle_progress_spearman"] = spearman_rank_correlation(
            -predicted,
            oracle_progress,
        )
    return result


def _episode_values(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    task_key: str,
    episode_key: str,
) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        task = str(row[task_key])
        episode = int(row[episode_key])
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError(f"non-finite {metric} for task={task} episode={episode}")
        grouped.setdefault(task, {}).setdefault(episode, []).append(value)
    if not grouped:
        raise ValueError("task-macro aggregation requires at least one row")
    return {
        task: {
            episode: float(np.mean(values))
            for episode, values in episodes.items()
        }
        for task, episodes in grouped.items()
    }


def task_macro_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
    task_key: str = "task",
    episode_key: str = "episode",
) -> dict[str, Any]:
    """Bootstrap episodes independently within task, then average task means."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    grouped = _episode_values(
        rows,
        metric,
        task_key=task_key,
        episode_key=episode_key,
    )
    task_names = tuple(sorted(grouped))
    arrays = {
        task: np.asarray(list(grouped[task].values()), dtype=np.float64)
        for task in task_names
    }
    estimate = float(np.mean([values.mean() for values in arrays.values()]))
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for sample in range(bootstrap_samples):
        task_means = []
        for task in task_names:
            values = arrays[task]
            indices = rng.integers(0, values.size, size=values.size)
            task_means.append(float(values[indices].mean()))
        samples[sample] = float(np.mean(task_means))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci95": [float(low), float(high)],
        "bootstrap_samples": int(bootstrap_samples),
        "tasks": list(task_names),
        "episodes_per_task": {
            task: int(arrays[task].size) for task in task_names
        },
    }


def paired_task_macro_bootstrap(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
    pair_keys: Sequence[str] = ("task", "episode", "snapshot", "horizon"),
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Return a paired task-macro delta with positive meaning right is better."""

    def index(
        rows: Sequence[Mapping[str, Any]],
        label: str,
    ) -> dict[tuple[Any, ...], float]:
        result: dict[tuple[Any, ...], float] = {}
        for row in rows:
            key = tuple(row[name] for name in pair_keys)
            if key in result:
                raise ValueError(f"duplicate {label} paired row: {key}")
            result[key] = float(row[metric])
        return result

    left = index(left_rows, "left")
    right = index(right_rows, "right")
    if set(left) != set(right):
        missing_left = sorted(set(right).difference(left))
        missing_right = sorted(set(left).difference(right))
        raise ValueError(
            "paired rows differ: "
            f"missing_left={missing_left[:3]} missing_right={missing_right[:3]}"
        )
    sign = 1.0 if higher_is_better else -1.0
    delta_rows = []
    for key in sorted(left):
        values = dict(zip(pair_keys, key))
        values[metric] = sign * (right[key] - left[key])
        delta_rows.append(values)
    result = task_macro_bootstrap(
        delta_rows,
        metric,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    result["orientation"] = "positive_means_right_better"
    result["raw_delta"] = (
        "right_minus_left" if higher_is_better else "left_minus_right"
    )
    return result
