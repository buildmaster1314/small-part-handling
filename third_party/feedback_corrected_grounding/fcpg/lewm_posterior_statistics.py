"""Paired physical-episode statistics for the LeWM posterior gate.

The input is already reduced to one MSE per physical episode, anchor, method,
and evaluation window.  This module keeps method pairing exact, resamples
whole ``(task, episode_id)`` clusters, and gives every observed
``task x anchor`` cell equal macro weight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


PRIMARY_WINDOW = "o1_o2"
SECONDARY_WINDOW = "r0"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 9_531

EpisodeId = int | str
AnchorId = int | str
SupportKey = tuple[str, EpisodeId, AnchorId, str]
PrimaryKey = tuple[str, EpisodeId, AnchorId]
TaskAnchorKey = tuple[str, AnchorId]


def _strict_integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _identity(value: Any, *, label: str) -> int | str:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise TypeError(f"{label} must be an integer or non-empty string")


def _finite_mse(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _stable_mean(values: Sequence[float] | np.ndarray, *, label: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} inputs must be finite")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    result = scale * float(np.mean(array / scale))
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _identity_order(value: int | str) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _primary_key_order(key: PrimaryKey) -> tuple[Any, ...]:
    task, episode_id, anchor = key
    return (task, _identity_order(episode_id), _identity_order(anchor))


def validate_primary_window(primary_window: Any) -> str:
    """Require the preregistered primary window and reject recovery onset."""

    window = _nonempty_string(primary_window, label="primary_window")
    if window == SECONDARY_WINDOW:
        raise ValueError("r0 is a secondary diagnostic and cannot be primary")
    if window != PRIMARY_WINDOW:
        raise ValueError(f"primary_window must be exactly {PRIMARY_WINDOW!r}")
    return window


def _parse_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[SupportKey, float]],
    tuple[str, ...],
    tuple[str, ...],
]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of mappings")
    if not records:
        raise ValueError("records must not be empty")

    by_method: dict[str, dict[SupportKey, float]] = {}
    seen: set[tuple[str, EpisodeId, AnchorId, str, str]] = set()
    windows: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        task = _nonempty_string(record.get("task"), label=f"record {index} task")
        episode_id = _identity(
            record.get("episode_id"), label=f"record {index} episode_id"
        )
        anchor = _identity(record.get("anchor"), label=f"record {index} anchor")
        method = _nonempty_string(
            record.get("method"), label=f"record {index} method"
        )
        window = _nonempty_string(
            record.get("window"), label=f"record {index} window"
        )
        mse = _finite_mse(record.get("mse"), label=f"record {index} mse")

        duplicate_key = (task, episode_id, anchor, method, window)
        if duplicate_key in seen:
            raise ValueError(f"duplicate posterior statistic record: {duplicate_key}")
        seen.add(duplicate_key)
        windows.add(window)
        by_method.setdefault(method, {})[(task, episode_id, anchor, window)] = mse

    methods = tuple(sorted(by_method))
    reference_method = methods[0]
    reference_support = set(by_method[reference_method])
    for method in methods[1:]:
        support = set(by_method[method])
        if support != reference_support:
            missing = sorted(
                reference_support - support,
                key=lambda key: (
                    key[0],
                    _identity_order(key[1]),
                    _identity_order(key[2]),
                    key[3],
                ),
            )
            unexpected = sorted(
                support - reference_support,
                key=lambda key: (
                    key[0],
                    _identity_order(key[1]),
                    _identity_order(key[2]),
                    key[3],
                ),
            )
            raise ValueError(
                "method support differs: "
                f"reference={reference_method!r} method={method!r} "
                f"missing={missing[:3]} unexpected={unexpected[:3]}"
            )
    return by_method, methods, tuple(sorted(windows))


def _primary_values(
    values: Mapping[SupportKey, float], *, primary_window: str
) -> dict[PrimaryKey, float]:
    return {
        (task, episode_id, anchor): mse
        for (task, episode_id, anchor, window), mse in values.items()
        if window == primary_window
    }


def _support_groups(
    support: set[PrimaryKey],
) -> tuple[
    dict[str, tuple[EpisodeId, ...]],
    dict[str, tuple[AnchorId, ...]],
    tuple[TaskAnchorKey, ...],
]:
    episodes: dict[str, set[EpisodeId]] = {}
    anchors: dict[str, set[AnchorId]] = {}
    for task, episode_id, anchor in support:
        episodes.setdefault(task, set()).add(episode_id)
        anchors.setdefault(task, set()).add(anchor)
    if not episodes:
        raise ValueError(f"records contain no {PRIMARY_WINDOW!r} primary support")
    episodes_by_task = {
        task: tuple(sorted(task_episodes, key=_identity_order))
        for task, task_episodes in sorted(episodes.items())
    }
    anchors_by_task = {
        task: tuple(sorted(anchors[task], key=_identity_order))
        for task in episodes_by_task
    }
    cells = tuple(
        (task, anchor)
        for task in episodes_by_task
        for anchor in anchors_by_task[task]
    )
    return episodes_by_task, anchors_by_task, cells


def _cluster_draws(
    support: set[PrimaryKey],
    episodes_by_task: Mapping[str, tuple[EpisodeId, ...]],
    anchors_by_task: Mapping[str, tuple[AnchorId, ...]],
    *,
    samples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Draw task-stratified whole episodes, retaining every reported cell.

    Complete episode-by-anchor panels never need a redraw.  For an incomplete
    but method-matched panel, a cluster draw can omit all supporting episodes
    for a sparse anchor.  Such an undefined macro draw is redrawn as a whole;
    individual anchors are never sampled independently.
    """

    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    rejected = 0
    for task, episodes in episodes_by_task.items():
        episode_position = {
            episode_id: index for index, episode_id in enumerate(episodes)
        }
        masks = np.zeros((len(anchors_by_task[task]), len(episodes)), dtype=bool)
        for anchor_index, anchor in enumerate(anchors_by_task[task]):
            for episode_id in episodes:
                masks[anchor_index, episode_position[episode_id]] = (
                    task,
                    episode_id,
                    anchor,
                ) in support
        if not np.all(masks.any(axis=1)):
            raise RuntimeError(f"task {task!r} contains an anchor with no support")

        indices = rng.integers(0, len(episodes), size=(samples, len(episodes)))
        for _ in range(10_000):
            valid = np.all(np.any(masks[:, indices], axis=2), axis=0)
            if bool(np.all(valid)):
                break
            invalid = np.flatnonzero(~valid)
            rejected += int(invalid.size)
            indices[invalid] = rng.integers(
                0, len(episodes), size=(invalid.size, len(episodes))
            )
        else:
            raise ValueError(
                f"unable to draw complete physical-episode clusters for task {task!r}"
            )
        result[task] = indices
    return result, rejected


def _cell_rows(
    baseline: Mapping[PrimaryKey, float],
    method_values: Mapping[PrimaryKey, float],
    cells: Sequence[TaskAnchorKey],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, anchor in cells:
        keys = sorted(
            (
                key
                for key in baseline
                if key[0] == task and key[2] == anchor
            ),
            key=_primary_key_order,
        )
        baseline_values = np.asarray([baseline[key] for key in keys], dtype=np.float64)
        candidate_values = np.asarray(
            [method_values[key] for key in keys], dtype=np.float64
        )
        differences = baseline_values - candidate_values
        rows.append(
            {
                "task": task,
                "anchor": anchor,
                "physical_episode_count": len(keys),
                "baseline_mse": _stable_mean(
                    baseline_values, label=f"{task}/{anchor} baseline MSE"
                ),
                "method_mse": _stable_mean(
                    candidate_values, label=f"{task}/{anchor} method MSE"
                ),
                "improvement": _stable_mean(
                    differences, label=f"{task}/{anchor} improvement"
                ),
            }
        )
    return rows


def _bootstrap_method(
    baseline: Mapping[PrimaryKey, float],
    method_values: Mapping[PrimaryKey, float],
    episodes_by_task: Mapping[str, tuple[EpisodeId, ...]],
    cells: Sequence[TaskAnchorKey],
    draws: Mapping[str, np.ndarray],
    *,
    samples: int,
) -> np.ndarray:
    macro = np.zeros(samples, dtype=np.float64)
    for task, anchor in cells:
        episodes = episodes_by_task[task]
        present = np.asarray(
            [(task, episode_id, anchor) in baseline for episode_id in episodes],
            dtype=bool,
        )
        differences = np.zeros(len(episodes), dtype=np.float64)
        for episode_index, episode_id in enumerate(episodes):
            key = (task, episode_id, anchor)
            if key in baseline:
                differences[episode_index] = baseline[key] - method_values[key]
        selected = draws[task]
        denominator = present[selected].sum(axis=1)
        if np.any(denominator == 0):
            raise RuntimeError("cluster bootstrap produced an empty task-anchor cell")
        scale = float(np.max(np.abs(differences[present])))
        if scale == 0.0:
            cell_samples = np.zeros(samples, dtype=np.float64)
        else:
            normalized = differences / scale
            cell_samples = (
                normalized[selected].sum(axis=1) / denominator * scale
            )
        macro += cell_samples / len(cells)
    if not np.isfinite(macro).all():
        raise ValueError("paired task-anchor bootstrap produced non-finite samples")
    return macro


def _coverage(
    support: set[PrimaryKey],
    methods: Sequence[str],
    windows: Sequence[str],
    episodes_by_task: Mapping[str, tuple[EpisodeId, ...]],
    anchors_by_task: Mapping[str, tuple[AnchorId, ...]],
    cells: Sequence[TaskAnchorKey],
) -> dict[str, Any]:
    expected_pairs = sum(
        len(episodes_by_task[task]) * len(anchors_by_task[task])
        for task in episodes_by_task
    )
    complete_episodes = sum(
        all((task, episode_id, anchor) in support for anchor in anchors_by_task[task])
        for task, episodes in episodes_by_task.items()
        for episode_id in episodes
    )
    return {
        "methods": list(methods),
        "windows": list(windows),
        "support_identical_across_methods": True,
        "task_count": len(episodes_by_task),
        "unique_anchor_count": len({anchor for _, anchor in cells}),
        "task_anchor_count": len(cells),
        "physical_episode_count": sum(map(len, episodes_by_task.values())),
        "complete_physical_episode_count": complete_episodes,
        "primary_episode_anchor_count_per_method": len(support),
        "primary_episode_anchor_expected_count": expected_pairs,
        "primary_episode_anchor_fraction": len(support) / expected_pairs,
        "episodes_per_task": {
            task: len(episodes) for task, episodes in episodes_by_task.items()
        },
        "episodes_per_task_anchor": [
            {
                "task": task,
                "anchor": anchor,
                "physical_episode_count": sum(
                    (task, episode_id, anchor) in support
                    for episode_id in episodes_by_task[task]
                ),
            }
            for task, anchor in cells
        ],
    }


def summarize_posterior_statistics(
    records: Sequence[Mapping[str, Any]],
    baseline_method: str,
    *,
    primary_window: str = PRIMARY_WINDOW,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate episode records and summarize every method against baseline.

    Positive improvement means that the method has lower MSE than the
    baseline.  The point estimate and every bootstrap replicate first average
    paired MSE differences within each ``task x anchor`` cell and then average
    those cells equally.
    """

    window = validate_primary_window(primary_window)
    baseline_name = _nonempty_string(baseline_method, label="baseline_method")
    samples = _strict_integer(
        bootstrap_samples, label="bootstrap_samples", minimum=1
    )
    seed = _strict_integer(bootstrap_seed, label="bootstrap_seed", minimum=0)
    by_method, methods, windows = _parse_records(records)
    if baseline_name not in by_method:
        raise ValueError(f"baseline method {baseline_name!r} is absent")
    comparison_methods = tuple(method for method in methods if method != baseline_name)
    if not comparison_methods:
        raise ValueError("posterior statistics require a non-baseline method")

    primary = {
        method: _primary_values(values, primary_window=window)
        for method, values in by_method.items()
    }
    baseline = primary[baseline_name]
    support = set(baseline)
    if not support:
        raise ValueError(f"records contain no {window!r} primary support")
    for method in comparison_methods:
        if set(primary[method]) != support:
            raise RuntimeError("validated method support changed during primary filtering")

    episodes_by_task, anchors_by_task, cells = _support_groups(support)
    draws, rejected_draws = _cluster_draws(
        support,
        episodes_by_task,
        anchors_by_task,
        samples=samples,
        seed=seed,
    )
    coverage = _coverage(
        support,
        methods,
        windows,
        episodes_by_task,
        anchors_by_task,
        cells,
    )

    comparisons: dict[str, dict[str, Any]] = {}
    for method in comparison_methods:
        per_cell = _cell_rows(baseline, primary[method], cells)
        bootstrap = _bootstrap_method(
            baseline,
            primary[method],
            episodes_by_task,
            cells,
            draws,
            samples=samples,
        )
        comparisons[method] = {
            "method": method,
            "baseline_method": baseline_name,
            "baseline_macro_mse": _stable_mean(
                [row["baseline_mse"] for row in per_cell],
                label="baseline task-anchor macro MSE",
            ),
            "method_macro_mse": _stable_mean(
                [row["method_mse"] for row in per_cell],
                label=f"{method} task-anchor macro MSE",
            ),
            "improvement": _stable_mean(
                [row["improvement"] for row in per_cell],
                label=f"{method} task-anchor macro improvement",
            ),
            "improvement_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "coverage": {
                "paired_episode_anchor_count": len(support),
                "physical_episode_cluster_count": coverage[
                    "physical_episode_count"
                ],
                "task_anchor_count": len(cells),
                "method_to_baseline_support_fraction": 1.0,
            },
            "per_task_anchor": per_cell,
        }

    return {
        "schema_version": 1,
        "metric": "mse",
        "lower_is_better": True,
        "primary_window": window,
        "baseline_method": baseline_name,
        "aggregation": {
            "pair_key": ["task", "episode_id", "anchor", "window"],
            "physical_episode_cluster_key": ["task", "episode_id"],
            "macro_cell": ["task", "anchor"],
            "cell_weighting": "equal_task_anchor",
            "effect_orientation": "baseline_mse_minus_method_mse",
        },
        "bootstrap": {
            "method": "paired_physical_episode_cluster_within_task_task_anchor_macro",
            "samples": samples,
            "seed": seed,
            "confidence": 0.95,
            "interval": "percentile",
            "incomplete_panel_cluster_redraws": rejected_draws,
        },
        "coverage": coverage,
        "comparisons": comparisons,
    }


def paired_task_anchor_macro_bootstrap(
    records: Sequence[Mapping[str, Any]],
    baseline_method: str,
    *,
    primary_window: str = PRIMARY_WINDOW,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compatibility entry point named for the resampling procedure."""

    return summarize_posterior_statistics(
        records,
        baseline_method,
        primary_window=primary_window,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "PRIMARY_WINDOW",
    "SECONDARY_WINDOW",
    "paired_task_anchor_macro_bootstrap",
    "summarize_posterior_statistics",
    "validate_primary_window",
]
