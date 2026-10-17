"""Strict paired statistics for the LeWM anchor-visual 4x4 matrix.

The evaluator emits one teacher-forced prediction MSE per episode.  This
module compares every train/evaluation-mode cell with the
``triplet_nominal``/``triplet_nominal`` reference while preserving episode
pairing and giving every task equal weight.  It intentionally contains no
model, filesystem, or GPU code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from fcpg.lewm_anchor_ablation import (
    PARAMETER_COMPATIBLE_TRIPLET_MODES,
    TRIPLET_CORRECTED,
    TRIPLET_NOMINAL,
    validate_anchor_visual_override,
)


EpisodeKey = tuple[int, str, int]
CellKey = tuple[str, str]

BASELINE_CELL: CellKey = (TRIPLET_NOMINAL, TRIPLET_NOMINAL)


def _strict_integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _finite_mse(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _finite_stat(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
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
    return _finite_stat(scale * float(np.mean(array / scale)), label=label)


def _stable_sum(values: Sequence[float], *, label: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} inputs must be finite")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    normalized_sum = math.fsum(float(value) for value in array / scale)
    return _finite_stat(scale * normalized_sum, label=label)


def _episode_key(block: Mapping[str, Any]) -> EpisodeKey:
    source_index = _strict_integer(
        block.get("source_index"), label="episode source_index"
    )
    task_name = block.get("task_name")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("episode task_name must be a non-empty string")
    episode_id = _strict_integer(block.get("episode_id"), label="episode_id")
    return (source_index, task_name, episode_id)


def _episode_values(record: Mapping[str, Any]) -> dict[EpisodeKey, float]:
    blocks = record.get("episode_blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise TypeError("matrix record episode_blocks must be a sequence")
    result: dict[EpisodeKey, float] = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            raise TypeError("each episode block must be a mapping")
        key = _episode_key(block)
        if key in result:
            raise ValueError(f"duplicate episode key in matrix cell: {key}")
        teacher_forced = block.get("teacher_forced")
        if not isinstance(teacher_forced, Mapping):
            raise ValueError(f"episode {key} is missing teacher_forced metrics")
        result[key] = _finite_mse(
            teacher_forced.get("predicted_mse"),
            label=f"episode {key} teacher_forced.predicted_mse",
        )
    if not result:
        raise ValueError("matrix record contains no episode blocks")
    return result


def _parse_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[CellKey, dict[EpisodeKey, float]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of mappings")
    cells: dict[CellKey, dict[EpisodeKey, float]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each matrix record must be a mapping")
        train_mode = validate_anchor_visual_override(
            record.get("train_mode"), field_name="train_mode"  # type: ignore[arg-type]
        )
        eval_mode = validate_anchor_visual_override(
            record.get("eval_mode"), field_name="eval_mode"  # type: ignore[arg-type]
        )
        key = (train_mode, eval_mode)
        if key in cells:
            raise ValueError(
                "duplicate matrix cell: "
                f"train_mode={train_mode} eval_mode={eval_mode}"
            )
        cells[key] = _episode_values(record)

    expected = {
        (train_mode, eval_mode)
        for train_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES
        for eval_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES
    }
    actual = set(cells)
    if actual != expected:
        raise ValueError(
            "incomplete LeWM 4x4 matrix: "
            f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
        )

    baseline_keys = set(cells[BASELINE_CELL])
    for (train_mode, eval_mode), episodes in cells.items():
        keys = set(episodes)
        if keys != baseline_keys:
            raise ValueError(
                "episode keys differ across matrix cells: "
                f"train_mode={train_mode} eval_mode={eval_mode} "
                f"missing={sorted(baseline_keys - keys)[:3]} "
                f"unexpected={sorted(keys - baseline_keys)[:3]}"
            )
    return cells


def _task_keys(keys: set[EpisodeKey]) -> dict[str, tuple[EpisodeKey, ...]]:
    grouped: dict[str, list[EpisodeKey]] = {}
    for key in sorted(keys):
        grouped.setdefault(key[1], []).append(key)
    if not grouped:
        raise ValueError("paired matrix contains no tasks")
    return {task: tuple(task_keys) for task, task_keys in sorted(grouped.items())}


def _task_means(
    values: Mapping[EpisodeKey, float],
    grouped_keys: Mapping[str, Sequence[EpisodeKey]],
) -> dict[str, float]:
    return {
        task: _stable_mean(
            [values[key] for key in keys], label=f"task {task!r} MSE mean"
        )
        for task, keys in grouped_keys.items()
    }


def _paired_bootstrap_matrix(
    cells: Mapping[CellKey, Mapping[EpisodeKey, float]],
    grouped_keys: Mapping[str, Sequence[EpisodeKey]],
    *,
    samples: int,
    seed: int,
    chunk_size: int = 64,
) -> dict[CellKey, tuple[np.ndarray, np.ndarray]]:
    """Bootstrap all cells with the same paired draws and bounded memory."""

    cell_order = [
        (train_mode, eval_mode)
        for train_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES
        for eval_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES
    ]
    baseline = cells[BASELINE_CELL]
    task_count = len(grouped_keys)
    absolute = np.zeros((samples, len(cell_order)), dtype=np.float64)
    relative = np.zeros_like(absolute)
    rng = np.random.default_rng(seed)

    for task, keys in grouped_keys.items():
        baseline_values = np.asarray(
            [baseline[key] for key in keys], dtype=np.float64
        )
        if np.any(baseline_values <= 0.0):
            raise ValueError(
                "every baseline episode MSE must be positive for paired relative "
                f"bootstrap; task={task!r}"
            )
        cell_values = np.asarray(
            [[cells[cell][key] for cell in cell_order] for key in keys],
            dtype=np.float64,
        )
        baseline_scale = float(np.max(baseline_values))
        cell_scales = np.max(cell_values, axis=0)
        safe_cell_scales = np.where(cell_scales > 0.0, cell_scales, 1.0)
        normalized_baseline = baseline_values / baseline_scale
        normalized_cells = cell_values / safe_cell_scales

        for start in range(0, samples, chunk_size):
            stop = min(start + chunk_size, samples)
            indices = rng.integers(
                0,
                baseline_values.size,
                size=(stop - start, baseline_values.size),
            )
            baseline_means = (
                np.mean(normalized_baseline[indices], axis=1) * baseline_scale
            )
            cell_means = (
                np.mean(normalized_cells[indices], axis=1) * cell_scales
            )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                differences = baseline_means[:, None] - cell_means
                relative_effects = differences / baseline_means[:, None]
            absolute[start:stop] += differences / task_count
            relative[start:stop] += relative_effects / task_count

    if not np.isfinite(absolute).all() or not np.isfinite(relative).all():
        raise ValueError("paired bootstrap produced non-finite samples")
    return {
        cell: (absolute[:, index].copy(), relative[:, index].copy())
        for index, cell in enumerate(cell_order)
    }


def nominal_corrected_2x2_decomposition(
    train_by_eval_mse: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Return raw point effects for the nominal/corrected 2x2 submatrix.

    Input and training effects are oriented so positive means lower MSE after
    switching from nominal to corrected.  ``raw_interaction`` uses the standard
    outcome-scale contrast ``CC - CN - NC + NN``; positive values therefore
    indicate that the combined corrected cell is worse than the additive raw-
    MSE expectation.
    """

    def value(train_mode: str, eval_mode: str) -> float:
        train_row = train_by_eval_mse.get(train_mode)
        if not isinstance(train_row, Mapping):
            raise ValueError(f"2x2 decomposition is missing train mode {train_mode}")
        if eval_mode not in train_row:
            raise ValueError(
                f"2x2 decomposition is missing cell ({train_mode}, {eval_mode})"
            )
        return _finite_mse(
            train_row[eval_mode],
            label=f"2x2 cell ({train_mode}, {eval_mode})",
        )

    nn = value(TRIPLET_NOMINAL, TRIPLET_NOMINAL)
    nc = value(TRIPLET_NOMINAL, TRIPLET_CORRECTED)
    cn = value(TRIPLET_CORRECTED, TRIPLET_NOMINAL)
    cc = value(TRIPLET_CORRECTED, TRIPLET_CORRECTED)
    nominal_train_input = _finite_stat(
        nn - nc, label="nominal-train corrected-input effect"
    )
    corrected_train_input = _finite_stat(
        cn - cc, label="corrected-train corrected-input effect"
    )
    nominal_eval_training = _finite_stat(
        nn - cn, label="nominal-eval corrected-training effect"
    )
    corrected_eval_training = _finite_stat(
        nc - cc, label="corrected-eval corrected-training effect"
    )
    matched_gain = _finite_stat(nn - cc, label="matched corrected gain")
    raw_interaction = _stable_sum(
        [cc, -cn, -nc, nn], label="raw 2x2 interaction"
    )
    return {
        "orientation": {
            "input_effect": "positive_means_corrected_eval_has_lower_mse",
            "training_effect": "positive_means_corrected_train_has_lower_mse",
            "matched_gain": "positive_means_corrected_corrected_has_lower_mse",
            "raw_interaction": "CC_minus_CN_minus_NC_plus_NN",
        },
        "cells": {
            "nominal_train_nominal_eval": nn,
            "nominal_train_corrected_eval": nc,
            "corrected_train_nominal_eval": cn,
            "corrected_train_corrected_eval": cc,
        },
        "fixed_train_input_effect": {
            TRIPLET_NOMINAL: nominal_train_input,
            TRIPLET_CORRECTED: corrected_train_input,
        },
        "fixed_eval_training_effect": {
            TRIPLET_NOMINAL: nominal_eval_training,
            TRIPLET_CORRECTED: corrected_eval_training,
        },
        "matched_gain": matched_gain,
        "raw_interaction": raw_interaction,
    }


def summarize_paired_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 5_120,
) -> dict[str, Any]:
    """Validate and summarize all 16 paired anchor-visual matrix cells."""

    samples = _strict_integer(
        bootstrap_samples, label="bootstrap_samples", minimum=1
    )
    seed = _strict_integer(bootstrap_seed, label="bootstrap_seed", minimum=0)
    cells = _parse_records(records)
    grouped_keys = _task_keys(set(cells[BASELINE_CELL]))
    baseline = cells[BASELINE_CELL]
    baseline_task_means = _task_means(baseline, grouped_keys)
    for task, mean in baseline_task_means.items():
        if not math.isfinite(mean) or mean <= 0.0:
            raise ValueError(
                f"baseline task mean must be positive for task {task!r}; got {mean}"
            )
    for key, value in baseline.items():
        if value <= 0.0:
            raise ValueError(
                "every baseline episode MSE must be positive for paired relative "
                f"bootstrap; episode={key}"
            )

    train_by_eval: dict[str, dict[str, dict[str, Any]]] = {}
    macro_mse: dict[str, dict[str, float]] = {}
    flat_cells = []
    baseline_macro_mse = _stable_mean(
        list(baseline_task_means.values()), label="baseline task-macro MSE"
    )
    episode_count = len(baseline)
    task_count = len(grouped_keys)
    bootstrap_cells = _paired_bootstrap_matrix(
        cells,
        grouped_keys,
        samples=samples,
        seed=seed,
    )
    for train_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        train_by_eval[train_mode] = {}
        macro_mse[train_mode] = {}
        for eval_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
            cell_values = cells[(train_mode, eval_mode)]
            cell_task_means = _task_means(cell_values, grouped_keys)
            per_task = {}
            for task in grouped_keys:
                absolute_delta = _finite_stat(
                    baseline_task_means[task] - cell_task_means[task],
                    label=f"task {task!r} absolute delta",
                )
                per_task[task] = {
                    "baseline_mse": baseline_task_means[task],
                    "cell_mse": cell_task_means[task],
                    "absolute_delta": absolute_delta,
                    "relative_improvement": _finite_stat(
                        absolute_delta / baseline_task_means[task],
                        label=f"task {task!r} relative improvement",
                    ),
                    "episode_count": len(grouped_keys[task]),
                }
            task_macro_mse = _stable_mean(
                list(cell_task_means.values()), label="cell task-macro MSE"
            )
            absolute_delta = _stable_mean(
                [value["absolute_delta"] for value in per_task.values()],
                label="cell macro absolute delta",
            )
            relative_improvement = _stable_mean(
                [value["relative_improvement"] for value in per_task.values()],
                label="cell macro relative improvement",
            )
            absolute_samples, relative_samples = bootstrap_cells[
                (train_mode, eval_mode)
            ]
            relative_ci = [
                _finite_stat(
                    np.quantile(relative_samples, quantile),
                    label=f"relative bootstrap quantile {quantile}",
                )
                for quantile in (0.025, 0.975)
            ]
            absolute_ci = [
                _finite_stat(
                    np.quantile(absolute_samples, quantile),
                    label=f"absolute bootstrap quantile {quantile}",
                )
                for quantile in (0.025, 0.975)
            ]
            probability = _finite_stat(
                np.mean(relative_samples > 0.0),
                label="bootstrap probability of improvement",
            )
            cell_summary = {
                "train_mode": train_mode,
                "eval_mode": eval_mode,
                "task_macro_mse": task_macro_mse,
                "macro_relative_improvement": relative_improvement,
                "absolute_delta": absolute_delta,
                "relative_improvement_95_ci": relative_ci,
                "absolute_delta_95_ci": absolute_ci,
                "probability_improvement_gt_zero": probability,
                "episode_count": episode_count,
                "task_count": task_count,
                "per_task": per_task,
            }
            train_by_eval[train_mode][eval_mode] = cell_summary
            macro_mse[train_mode][eval_mode] = task_macro_mse
            flat_cells.append(cell_summary)

    return {
        "schema_version": 1,
        "metric": "teacher_forced.predicted_mse",
        "lower_is_better": True,
        "modes": list(PARAMETER_COMPATIBLE_TRIPLET_MODES),
        "baseline": {
            "train_mode": BASELINE_CELL[0],
            "eval_mode": BASELINE_CELL[1],
            "task_macro_mse": baseline_macro_mse,
            "per_task_mse": baseline_task_means,
        },
        "aggregation": {
            "episode_pair_key": ["source_index", "task_name", "episode_id"],
            "episode_to_task": "equal_episode_mean",
            "task_to_macro": "equal_task_mean",
            "relative_scale": "paired_resampled_baseline_task_mean",
            "effect_orientation": "positive_means_lower_mse_than_baseline",
        },
        "bootstrap": {
            "method": "paired_episode_within_task_equal_task_macro",
            "samples": samples,
            "seed": seed,
            "confidence": 0.95,
        },
        "episode_count": episode_count,
        "task_count": task_count,
        "episodes_per_task": {
            task: len(keys) for task, keys in grouped_keys.items()
        },
        "train_by_eval": train_by_eval,
        "cells": flat_cells,
        "nominal_corrected_2x2": nominal_corrected_2x2_decomposition(macro_mse),
    }
