"""Deterministic Markdown rendering for the LeWM anchor-visual matrix.

The renderer is intentionally pure: it performs no I/O and never infers or
fills missing experimental results.  Its output is explicitly labelled as
development-validation evidence rather than a sealed-test result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from fcpg.lewm_anchor_ablation import (
    PARAMETER_COMPATIBLE_TRIPLET_MODES,
    TRIPLET_CORRECTED,
    TRIPLET_NOMINAL,
)


_MODES = PARAMETER_COMPATIBLE_TRIPLET_MODES
_FACTOR_CELLS = ("NN", "NC", "CN", "CC")
_FACTOR_EFFECT_ORDER = (
    "eval_input_at_nominal_train",
    "eval_input_at_corrected_train",
    "training_at_nominal_eval",
    "training_at_corrected_eval",
    "matched_nominal_to_corrected",
    "interaction",
)
_PROVENANCE_PRIORITY = (
    "optimizer_step",
    "parameter_count",
    "seed",
    "target_sha256",
    "data_split_sha256",
    "complete_4x4",
)
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
_FORMAL_SUMMARY_PROTOCOL = "fcpg_lewm_anchor_visual_matrix_v1"
_DEVELOPMENT_GATE_PROTOCOL = "fcpg_lewm_anchor_visual_dev_selection_v1"
_DEVELOPMENT_GATE_SIGN_FLIP_ALTERNATIVE = "greater"
_DEVELOPMENT_GATE_ESTIMAND = (
    "mean_task(1 - mean_episode,H candidate_ratio / "
    "mean_episode,H control_ratio)"
)
_DEVELOPMENT_GATE_PRIMARY_MINIMUM_EFFECT = 0.03
_DEVELOPMENT_GATE_PRIMARY_CI_LOWER_MINIMUM = 0.0
_DEVELOPMENT_GATE_CLEAN_MARGIN = 1.05
_DEVELOPMENT_GATE_MINIMUM_POSITIVE_TASKS = 3
_DEVELOPMENT_GATE_MAXIMUM_TASK_DEGRADATION = 0.05
_DEVELOPMENT_GATE_EXPECTED_TASKS = 4
_DEVELOPMENT_GATE_PRIMARY_EPISODES = 112
_DEVELOPMENT_GATE_CLEAN_EPISODES = 56
_DEVELOPMENT_GATE_UNSEEN_MILD_EPISODES = 29
_DEVELOPMENT_GATE_HOLM_ALPHA = 0.05
_STAT_ABSOLUTE_TOLERANCE = 1e-12
_STAT_RELATIVE_TOLERANCE = 1e-12
_GATE_CONDITION_ORDER = (
    "primary_superiority",
    "clean_noninferiority",
    "task_direction",
    "representation_noncollapse",
    "corrector_train_unseen_directional_agreement",
)
_REPRESENTATION_METRICS = ("effective_rank", "participation_ratio")
_REPRESENTATION_ABSOLUTE_MINIMUM = {
    "effective_rank": 8.0,
    "participation_ratio": 4.0,
}
_REPRESENTATION_OVERALL_RATIO_MINIMUM = 0.9
_REPRESENTATION_PER_TASK_RATIO_MINIMUM = 0.8
_SECONDARY_HOLM_ORDER = (
    "paired_visual_vs_triplet_corrected",
    "paired_innovation_vs_paired_visual",
    "triplet_corrected_vs_triplet_nominal",
)
_PAIR_EFFECT_FIELDS = {
    "estimand",
    "candidate_macro",
    "control_macro",
    "relative_improvement",
    "relative_improvement_95_ci",
    "candidate_to_control_ratio",
    "candidate_to_control_ratio_one_sided_95_upper",
    "per_task",
    "episode_count",
    "task_count",
    "passed",
}
_PAIRED_STATISTICS_FIELDS = {
    "schema_version",
    "metric",
    "lower_is_better",
    "modes",
    "baseline",
    "aggregation",
    "bootstrap",
    "episode_count",
    "task_count",
    "episodes_per_task",
    "train_by_eval",
    "cells",
    "nominal_corrected_2x2",
}
_PAIRED_STATISTICS_CELL_FIELDS = {
    "train_mode",
    "eval_mode",
    "task_macro_mse",
    "macro_relative_improvement",
    "absolute_delta",
    "relative_improvement_95_ci",
    "absolute_delta_95_ci",
    "probability_improvement_gt_zero",
    "episode_count",
    "task_count",
    "per_task",
}
_PAIRED_STATISTICS_TASK_FIELDS = {
    "baseline_mse",
    "cell_mse",
    "absolute_delta",
    "relative_improvement",
    "episode_count",
}
_DEVELOPMENT_GATE_FIELDS = {
    "schema_version",
    "protocol",
    "development_only",
    "primary",
    "clean_noninferiority",
    "task_direction",
    "representation_noncollapse",
    "corrector_train_unseen_sensitivity",
    "secondary_holm_family",
    "conditions",
    "passed",
    "decision",
}


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _ascii_text(value: object, *, label: str, safe_key: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must contain only ASCII characters") from exc
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{label} must not contain ASCII control characters")
    if safe_key and not _SAFE_KEY.fullmatch(value):
        raise ValueError(f"{label} must be a non-empty ASCII key")
    return value


def _probability(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _interval(value: object, *, label: str) -> tuple[float, float]:
    interval = _sequence(value, label=label)
    if len(interval) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    low = _finite(interval[0], label=f"{label}[0]")
    high = _finite(interval[1], label=f"{label}[1]")
    if low > high:
        raise ValueError(f"{label} lower endpoint must not exceed upper")
    return low, high


def _require_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=_STAT_RELATIVE_TOLERANCE,
        abs_tol=_STAT_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(
            f"{label} disagrees with recomputed value: "
            f"expected={expected!r} got={actual!r}"
        )


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{label} has incomplete or unexpected keys: "
            f"missing={missing} unexpected={unexpected}"
        )


def _structure_matches_recomputed(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(
                float(actual),
                expected,
                rel_tol=_STAT_RELATIVE_TOLERANCE,
                abs_tol=_STAT_ABSOLUTE_TOLERANCE,
            )
        )
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                _structure_matches_recomputed(actual[key], expected_value)
                for key, expected_value in expected.items()
            )
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return (
            isinstance(actual, Sequence)
            and not isinstance(actual, (str, bytes))
            and len(actual) == len(expected)
            and all(
                _structure_matches_recomputed(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _validated_modes(summary: Mapping[str, Any]) -> tuple[str, ...]:
    modes = _sequence(summary.get("modes"), label="modes")
    if any(not isinstance(mode, str) for mode in modes):
        raise TypeError("every mode must be a string")
    result = tuple(modes)
    if result != _MODES:
        raise ValueError(
            "modes must contain the complete canonical anchor-visual order: "
            f"expected={list(_MODES)!r} got={list(result)!r}"
        )
    return result


def _validated_matrix(
    value: object,
    modes: tuple[str, ...],
    *,
    label: str,
) -> dict[str, dict[str, float]]:
    matrix = _mapping(value, label=label)
    _exact_keys(matrix, set(modes), label=label)
    result: dict[str, dict[str, float]] = {}
    for train_mode in modes:
        row = _mapping(matrix[train_mode], label=f"{label}[{train_mode!r}]")
        _exact_keys(row, set(modes), label=f"{label}[{train_mode!r}]")
        result[train_mode] = {
            eval_mode: _finite(
                row[eval_mode],
                label=f"{label}[{train_mode!r}][{eval_mode!r}]",
            )
            for eval_mode in modes
        }
    return result


def _validated_element_pooled_matrix(
    summary: Mapping[str, Any],
    modes: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    mse = _validated_matrix(
        summary.get("train_by_eval"),
        modes,
        label="train_by_eval",
    )
    for train_mode in modes:
        for eval_mode in modes:
            if mse[train_mode][eval_mode] < 0.0:
                raise ValueError(
                    "train_by_eval element-pooled MSE values must be "
                    "non-negative"
                )
    baseline = mse[TRIPLET_NOMINAL][TRIPLET_NOMINAL]
    if baseline <= 0.0:
        raise ValueError(
            "train_by_eval triplet_nominal/triplet_nominal element-pooled "
            "MSE must be positive"
        )

    submitted = _validated_matrix(
        summary.get("relative_improvement_vs_triplet_nominal"),
        modes,
        label="relative_improvement_vs_triplet_nominal",
    )
    recomputed: dict[str, dict[str, float]] = {}
    for train_mode in modes:
        recomputed[train_mode] = {}
        for eval_mode in modes:
            expected = math.fsum(
                (baseline, -mse[train_mode][eval_mode])
            ) / baseline
            _require_close(
                submitted[train_mode][eval_mode],
                expected,
                label=(
                    "relative_improvement_vs_triplet_nominal"
                    f"[{train_mode!r}][{eval_mode!r}]"
                ),
            )
            recomputed[train_mode][eval_mode] = expected
    return mse, recomputed


def _validated_mode(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if value not in _MODES:
        raise ValueError(f"{label} must be one of {list(_MODES)!r}")
    return value


def _validated_paired_episode_statistics(
    value: object,
    modes: tuple[str, ...],
) -> tuple[
    tuple[str, str],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, float],
    dict[str, int],
]:
    label = "paired_episode_statistics"
    statistics = _mapping(value, label=label)
    _exact_keys(statistics, _PAIRED_STATISTICS_FIELDS, label=label)

    schema_version = _positive_int(
        statistics["schema_version"], label=f"{label}.schema_version"
    )
    if schema_version != 1:
        raise ValueError(f"{label}.schema_version must be exactly 1")
    metric = _ascii_text(statistics["metric"], label=f"{label}.metric")
    if metric != "teacher_forced.predicted_mse":
        raise ValueError(f"{label}.metric is unexpected")
    if not _boolean(
        statistics["lower_is_better"], label=f"{label}.lower_is_better"
    ):
        raise ValueError(f"{label}.lower_is_better must be true")

    raw_modes = _sequence(statistics["modes"], label=f"{label}.modes")
    paired_modes = tuple(
        _validated_mode(mode, label=f"{label}.modes[{index}]")
        for index, mode in enumerate(raw_modes)
    )
    if paired_modes != modes:
        raise ValueError(f"{label}.modes must match the canonical mode order")

    aggregation = _mapping(
        statistics["aggregation"], label=f"{label}.aggregation"
    )
    _exact_keys(
        aggregation,
        {
            "episode_pair_key",
            "episode_to_task",
            "task_to_macro",
            "relative_scale",
            "effect_orientation",
        },
        label=f"{label}.aggregation",
    )
    pair_key = tuple(
        _ascii_text(item, label=f"{label}.aggregation.episode_pair_key")
        for item in _sequence(
            aggregation["episode_pair_key"],
            label=f"{label}.aggregation.episode_pair_key",
        )
    )
    if pair_key != ("source_index", "task_name", "episode_id"):
        raise ValueError(f"{label}.aggregation.episode_pair_key is unexpected")
    for field, expected in {
        "episode_to_task": "equal_episode_mean",
        "task_to_macro": "equal_task_mean",
        "relative_scale": "paired_resampled_baseline_task_mean",
        "effect_orientation": "positive_means_lower_mse_than_baseline",
    }.items():
        actual = _ascii_text(
            aggregation[field], label=f"{label}.aggregation.{field}"
        )
        if actual != expected:
            raise ValueError(f"{label}.aggregation.{field} is unexpected")

    bootstrap = _mapping(statistics["bootstrap"], label=f"{label}.bootstrap")
    _exact_keys(
        bootstrap,
        {"method", "samples", "seed", "confidence"},
        label=f"{label}.bootstrap",
    )
    method = _ascii_text(
        bootstrap["method"], label=f"{label}.bootstrap.method"
    )
    if method != "paired_episode_within_task_equal_task_macro":
        raise ValueError(f"{label}.bootstrap.method is unexpected")
    _positive_int(bootstrap["samples"], label=f"{label}.bootstrap.samples")
    _nonnegative_int(bootstrap["seed"], label=f"{label}.bootstrap.seed")
    _require_close(
        _probability(
            bootstrap["confidence"], label=f"{label}.bootstrap.confidence"
        ),
        0.95,
        label=f"{label}.bootstrap.confidence",
    )

    episode_count = _positive_int(
        statistics["episode_count"], label=f"{label}.episode_count"
    )
    task_count = _positive_int(
        statistics["task_count"], label=f"{label}.task_count"
    )
    raw_counts = _mapping(
        statistics["episodes_per_task"], label=f"{label}.episodes_per_task"
    )
    episodes_per_task: dict[str, int] = {}
    for raw_task, raw_count in raw_counts.items():
        task = _ascii_text(
            raw_task, label=f"{label}.episodes_per_task task"
        )
        episodes_per_task[task] = _positive_int(
            raw_count, label=f"{label}.episodes_per_task[{task!r}]"
        )
    if len(episodes_per_task) != task_count:
        raise ValueError(f"{label}.task_count must match episodes_per_task")
    if sum(episodes_per_task.values()) != episode_count:
        raise ValueError(
            f"{label}.episode_count must equal the episodes_per_task total"
        )

    baseline = _mapping(statistics["baseline"], label=f"{label}.baseline")
    _exact_keys(
        baseline,
        {"train_mode", "eval_mode", "task_macro_mse", "per_task_mse"},
        label=f"{label}.baseline",
    )
    baseline_key = (
        _validated_mode(
            baseline["train_mode"], label=f"{label}.baseline.train_mode"
        ),
        _validated_mode(
            baseline["eval_mode"], label=f"{label}.baseline.eval_mode"
        ),
    )
    if baseline_key != (TRIPLET_NOMINAL, TRIPLET_NOMINAL):
        raise ValueError(
            f"{label}.baseline must be triplet_nominal/triplet_nominal"
        )
    raw_baseline_tasks = _mapping(
        baseline["per_task_mse"], label=f"{label}.baseline.per_task_mse"
    )
    _exact_keys(
        raw_baseline_tasks,
        set(episodes_per_task),
        label=f"{label}.baseline.per_task_mse",
    )
    baseline_task_mse: dict[str, float] = {}
    for task in sorted(episodes_per_task):
        task_mse = _finite(
            raw_baseline_tasks[task],
            label=f"{label}.baseline.per_task_mse[{task!r}]",
        )
        if task_mse <= 0.0:
            raise ValueError(
                f"{label}.baseline.per_task_mse[{task!r}] must be positive"
            )
        baseline_task_mse[task] = task_mse
    expected_baseline_macro = (
        math.fsum(baseline_task_mse.values()) / task_count
    )
    _require_close(
        _finite(
            baseline["task_macro_mse"],
            label=f"{label}.baseline.task_macro_mse",
        ),
        expected_baseline_macro,
        label=f"{label}.baseline.task_macro_mse",
    )

    expected_keys = {
        (train_mode, eval_mode)
        for train_mode in modes
        for eval_mode in modes
    }
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    canonical_raw_cells: dict[tuple[str, str], Mapping[str, Any]] = {}
    raw_cells = _sequence(statistics["cells"], label=f"{label}.cells")
    for index, raw_cell in enumerate(raw_cells):
        cell_label = f"{label}.cells[{index}]"
        cell = _mapping(raw_cell, label=cell_label)
        _exact_keys(
            cell, _PAIRED_STATISTICS_CELL_FIELDS, label=cell_label
        )
        key = (
            _validated_mode(
                cell["train_mode"], label=f"{cell_label}.train_mode"
            ),
            _validated_mode(
                cell["eval_mode"], label=f"{cell_label}.eval_mode"
            ),
        )
        if key in cells:
            raise ValueError(
                f"duplicate {label} cell: train={key[0]} eval={key[1]}"
            )
        canonical_raw_cells[key] = cell
        raw_per_task = _mapping(
            cell["per_task"], label=f"{cell_label}.per_task"
        )
        _exact_keys(
            raw_per_task,
            set(baseline_task_mse),
            label=f"{cell_label}.per_task",
        )
        task_rows: list[dict[str, float]] = []
        for task, baseline_mse in baseline_task_mse.items():
            task_label = f"{cell_label}.per_task[{task!r}]"
            row = _mapping(raw_per_task[task], label=task_label)
            _exact_keys(
                row, _PAIRED_STATISTICS_TASK_FIELDS, label=task_label
            )
            _require_close(
                _finite(
                    row["baseline_mse"],
                    label=f"{task_label}.baseline_mse",
                ),
                baseline_mse,
                label=f"{task_label}.baseline_mse",
            )
            cell_mse = _finite(
                row["cell_mse"], label=f"{task_label}.cell_mse"
            )
            if cell_mse < 0.0:
                raise ValueError(f"{task_label}.cell_mse must be non-negative")
            absolute_delta = math.fsum((baseline_mse, -cell_mse))
            _require_close(
                _finite(
                    row["absolute_delta"],
                    label=f"{task_label}.absolute_delta",
                ),
                absolute_delta,
                label=f"{task_label}.absolute_delta",
            )
            relative_improvement = absolute_delta / baseline_mse
            _require_close(
                _finite(
                    row["relative_improvement"],
                    label=f"{task_label}.relative_improvement",
                ),
                relative_improvement,
                label=f"{task_label}.relative_improvement",
            )
            task_episodes = _positive_int(
                row["episode_count"],
                label=f"{task_label}.episode_count",
            )
            if task_episodes != episodes_per_task[task]:
                raise ValueError(
                    f"{task_label}.episode_count disagrees with "
                    "episodes_per_task"
                )
            task_rows.append(
                {
                    "cell_mse": cell_mse,
                    "absolute_delta": absolute_delta,
                    "relative_improvement": relative_improvement,
                }
            )

        task_macro_mse = (
            math.fsum(row["cell_mse"] for row in task_rows) / task_count
        )
        macro_absolute_delta = (
            math.fsum(row["absolute_delta"] for row in task_rows)
            / task_count
        )
        macro_relative_improvement = (
            math.fsum(row["relative_improvement"] for row in task_rows)
            / task_count
        )
        for field, expected in (
            ("task_macro_mse", task_macro_mse),
            ("absolute_delta", macro_absolute_delta),
            ("macro_relative_improvement", macro_relative_improvement),
        ):
            _require_close(
                _finite(cell[field], label=f"{cell_label}.{field}"),
                expected,
                label=f"{cell_label}.{field}",
            )
        if (
            _positive_int(
                cell["episode_count"],
                label=f"{cell_label}.episode_count",
            )
            != episode_count
        ):
            raise ValueError(
                f"{cell_label}.episode_count disagrees with {label}"
            )
        if (
            _positive_int(
                cell["task_count"], label=f"{cell_label}.task_count"
            )
            != task_count
        ):
            raise ValueError(f"{cell_label}.task_count disagrees with {label}")

        relative_interval = _interval(
            cell["relative_improvement_95_ci"],
            label=f"{cell_label}.relative_improvement_95_ci",
        )
        if relative_interval[1] > 1.0 + _STAT_ABSOLUTE_TOLERANCE:
            raise ValueError(
                f"{cell_label}.relative_improvement_95_ci endpoints "
                "cannot exceed 1"
            )
        cells[key] = {
            "task_macro_mse": task_macro_mse,
            "macro_relative_improvement": macro_relative_improvement,
            "macro_absolute_delta": macro_absolute_delta,
            "bootstrap_95_ci": relative_interval,
            "probability_improvement": _probability(
                cell["probability_improvement_gt_zero"],
                label=(
                    f"{cell_label}.probability_improvement_gt_zero"
                ),
            ),
            "num_tasks": task_count,
            "num_episodes": episode_count,
        }
        _interval(
            cell["absolute_delta_95_ci"],
            label=f"{cell_label}.absolute_delta_95_ci",
        )

    if set(cells) != expected_keys:
        missing = sorted(expected_keys - set(cells))
        unexpected = sorted(set(cells) - expected_keys)
        raise ValueError(
            f"{label} must contain the complete 4x4 matrix: "
            f"missing={missing} unexpected={unexpected}"
        )
    _require_close(
        cells[baseline_key]["task_macro_mse"],
        expected_baseline_macro,
        label=f"{label}.baseline task_macro_mse",
    )

    train_by_eval = _mapping(
        statistics["train_by_eval"], label=f"{label}.train_by_eval"
    )
    _exact_keys(train_by_eval, set(modes), label=f"{label}.train_by_eval")
    for train_mode in modes:
        row = _mapping(
            train_by_eval[train_mode],
            label=f"{label}.train_by_eval[{train_mode!r}]",
        )
        _exact_keys(
            row, set(modes), label=f"{label}.train_by_eval[{train_mode!r}]"
        )
        for eval_mode in modes:
            submitted = _mapping(
                row[eval_mode],
                label=(
                    f"{label}.train_by_eval[{train_mode!r}]"
                    f"[{eval_mode!r}]"
                ),
            )
            expected = canonical_raw_cells[(train_mode, eval_mode)]
            if not _structure_matches_recomputed(submitted, expected):
                raise ValueError(
                    f"{label}.train_by_eval[{train_mode!r}]"
                    f"[{eval_mode!r}] disagrees with canonical cells"
                )

    factor_cells = {
        "NN": cells[(TRIPLET_NOMINAL, TRIPLET_NOMINAL)]["task_macro_mse"],
        "NC": cells[(TRIPLET_NOMINAL, TRIPLET_CORRECTED)]["task_macro_mse"],
        "CN": cells[(TRIPLET_CORRECTED, TRIPLET_NOMINAL)]["task_macro_mse"],
        "CC": cells[(TRIPLET_CORRECTED, TRIPLET_CORRECTED)]["task_macro_mse"],
    }
    nn, nc, cn, cc = (factor_cells[name] for name in ("NN", "NC", "CN", "CC"))
    expected_2x2 = {
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
            TRIPLET_NOMINAL: math.fsum((nn, -nc)),
            TRIPLET_CORRECTED: math.fsum((cn, -cc)),
        },
        "fixed_eval_training_effect": {
            TRIPLET_NOMINAL: math.fsum((nn, -cn)),
            TRIPLET_CORRECTED: math.fsum((nc, -cc)),
        },
        "matched_gain": math.fsum((nn, -cc)),
        "raw_interaction": math.fsum((cc, -cn, -nc, nn)),
    }
    submitted_2x2 = _mapping(
        statistics["nominal_corrected_2x2"],
        label=f"{label}.nominal_corrected_2x2",
    )
    if not _structure_matches_recomputed(submitted_2x2, expected_2x2):
        raise ValueError(
            f"{label}.nominal_corrected_2x2 disagrees with canonical cells"
        )
    return baseline_key, cells, factor_cells, episodes_per_task



def _validated_bootstrap(
    value: object,
    modes: tuple[str, ...],
    *,
    expected_baseline: tuple[str, str],
    paired_cells: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[tuple[str, str], dict[tuple[str, str], dict[str, Any]]]:
    bootstrap = _mapping(value, label="paired_episode_bootstrap")
    _exact_keys(
        bootstrap,
        {"baseline", "cells"},
        label="paired_episode_bootstrap",
    )
    baseline = _mapping(
        bootstrap["baseline"], label="paired_episode_bootstrap.baseline"
    )
    _exact_keys(
        baseline,
        {"train_mode", "eval_mode"},
        label="paired_episode_bootstrap.baseline",
    )
    baseline_key = (
        _validated_mode(
            baseline["train_mode"],
            label="paired_episode_bootstrap.baseline.train_mode",
        ),
        _validated_mode(
            baseline["eval_mode"],
            label="paired_episode_bootstrap.baseline.eval_mode",
        ),
    )
    if baseline_key != (TRIPLET_NOMINAL, TRIPLET_NOMINAL):
        raise ValueError(
            "paired_episode_bootstrap baseline must be "
            "triplet_nominal/triplet_nominal"
        )
    if baseline_key != expected_baseline:
        raise ValueError(
            "paired_episode_bootstrap baseline disagrees with "
            "paired_episode_statistics"
        )

    cells = _sequence(
        bootstrap["cells"], label="paired_episode_bootstrap.cells"
    )
    expected_cells = {
        (train_mode, eval_mode)
        for train_mode in modes
        for eval_mode in modes
    }
    required = {
        "train_mode",
        "eval_mode",
        "macro_relative_improvement",
        "macro_absolute_delta",
        "bootstrap_95_ci",
        "probability_improvement",
        "num_tasks",
        "num_episodes",
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_cell in enumerate(cells):
        cell = _mapping(raw_cell, label=f"paired_episode_bootstrap.cells[{index}]")
        _exact_keys(
            cell,
            required,
            label=f"paired_episode_bootstrap.cells[{index}]",
        )
        key = (
            _validated_mode(
                cell["train_mode"],
                label=f"paired_episode_bootstrap.cells[{index}].train_mode",
            ),
            _validated_mode(
                cell["eval_mode"],
                label=f"paired_episode_bootstrap.cells[{index}].eval_mode",
            ),
        )
        if key in result:
            raise ValueError(
                "duplicate paired_episode_bootstrap cell: "
                f"train={key[0]} eval={key[1]}"
            )
        interval = _sequence(
            cell["bootstrap_95_ci"],
            label=f"paired_episode_bootstrap.cells[{index}].bootstrap_95_ci",
        )
        if len(interval) != 2:
            raise ValueError("every bootstrap_95_ci must contain exactly two values")
        low = _finite(
            interval[0],
            label=f"paired_episode_bootstrap.cells[{index}].bootstrap_95_ci[0]",
        )
        high = _finite(
            interval[1],
            label=f"paired_episode_bootstrap.cells[{index}].bootstrap_95_ci[1]",
        )
        if low > high:
            raise ValueError("bootstrap_95_ci lower endpoint must not exceed upper")
        probability = _finite(
            cell["probability_improvement"],
            label=(
                f"paired_episode_bootstrap.cells[{index}]."
                "probability_improvement"
            ),
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability_improvement must be in [0, 1]")
        result[key] = {
            "macro_relative_improvement": _finite(
                cell["macro_relative_improvement"],
                label=(
                    f"paired_episode_bootstrap.cells[{index}]."
                    "macro_relative_improvement"
                ),
            ),
            "macro_absolute_delta": _finite(
                cell["macro_absolute_delta"],
                label=(
                    f"paired_episode_bootstrap.cells[{index}]."
                    "macro_absolute_delta"
                ),
            ),
            "bootstrap_95_ci": (low, high),
            "probability_improvement": probability,
            "num_tasks": _positive_int(
                cell["num_tasks"],
                label=f"paired_episode_bootstrap.cells[{index}].num_tasks",
            ),
            "num_episodes": _positive_int(
                cell["num_episodes"],
                label=f"paired_episode_bootstrap.cells[{index}].num_episodes",
            ),
        }
    actual_cells = set(result)
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        unexpected = sorted(actual_cells - expected_cells)
        raise ValueError(
            "paired_episode_bootstrap must contain the complete 4x4 matrix: "
            f"missing={missing} unexpected={unexpected}"
        )
    for key, submitted in result.items():
        expected = paired_cells[key]
        label = (
            "paired_episode_bootstrap cell "
            f"train={key[0]} eval={key[1]}"
        )
        for field in (
            "macro_relative_improvement",
            "macro_absolute_delta",
            "probability_improvement",
        ):
            _require_close(
                submitted[field], expected[field], label=f"{label}.{field}"
            )
        for index, (submitted_value, expected_value) in enumerate(
            zip(
                submitted["bootstrap_95_ci"],
                expected["bootstrap_95_ci"],
                strict=True,
            )
        ):
            _require_close(
                submitted_value,
                expected_value,
                label=f"{label}.bootstrap_95_ci[{index}]",
            )
        for field in ("num_tasks", "num_episodes"):
            if submitted[field] != expected[field]:
                raise ValueError(
                    f"{label}.{field} disagrees with paired_episode_statistics"
                )
        result[key] = {
            field: expected[field]
            for field in (
                "macro_relative_improvement",
                "macro_absolute_delta",
                "bootstrap_95_ci",
                "probability_improvement",
                "num_tasks",
                "num_episodes",
            )
        }
    return expected_baseline, result


def _validated_factor_decomposition(
    value: object,
    *,
    expected_cells: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    factor = _mapping(value, label="factor_decomposition")
    _exact_keys(factor, {"cells", "effects"}, label="factor_decomposition")
    raw_cells = _mapping(factor["cells"], label="factor_decomposition.cells")
    _exact_keys(
        raw_cells,
        set(_FACTOR_CELLS),
        label="factor_decomposition.cells",
    )
    submitted_cells = {
        name: _finite(
            raw_cells[name], label=f"factor_decomposition.cells[{name!r}]"
        )
        for name in _FACTOR_CELLS
    }
    for name, cell_mse in submitted_cells.items():
        if cell_mse < 0.0:
            raise ValueError(
                f"factor_decomposition.cells[{name!r}] task-macro MSE "
                "must be non-negative"
            )
        _require_close(
            cell_mse,
            expected_cells[name],
            label=(
                f"factor_decomposition.cells[{name!r}] relative to "
                "paired_episode_statistics"
            ),
        )
    cells = dict(expected_cells)
    nn = cells["NN"]
    if nn <= 0.0:
        raise ValueError(
            "factor_decomposition.cells['NN'] task-macro MSE must be positive"
        )

    raw_effects = _mapping(factor["effects"], label="factor_decomposition.effects")
    _exact_keys(
        raw_effects,
        set(_FACTOR_EFFECT_ORDER),
        label="factor_decomposition.effects",
    )
    recomputed_raw = {
        "eval_input_at_nominal_train": math.fsum(
            (cells["NN"], -cells["NC"])
        ),
        "eval_input_at_corrected_train": math.fsum(
            (cells["CN"], -cells["CC"])
        ),
        "training_at_nominal_eval": math.fsum(
            (cells["NN"], -cells["CN"])
        ),
        "training_at_corrected_eval": math.fsum(
            (cells["NC"], -cells["CC"])
        ),
        "matched_nominal_to_corrected": math.fsum(
            (cells["NN"], -cells["CC"])
        ),
        "interaction": math.fsum(
            (cells["CC"], -cells["CN"], -cells["NC"], cells["NN"])
        ),
    }
    effects: dict[str, dict[str, float]] = {}
    for name in _FACTOR_EFFECT_ORDER:
        effect = _mapping(
            raw_effects[name], label=f"factor_decomposition.effects[{name!r}]"
        )
        expected = (
            {"raw_mse_delta"}
            if name == "interaction"
            else {"raw_mse_delta", "relative_improvement"}
        )
        _exact_keys(
            effect,
            expected,
            label=f"factor_decomposition.effects[{name!r}]",
        )
        submitted_raw = _finite(
            effect["raw_mse_delta"],
            label=f"factor_decomposition.effects[{name!r}].raw_mse_delta",
        )
        expected_raw = recomputed_raw[name]
        _require_close(
            submitted_raw,
            expected_raw,
            label=f"factor_decomposition.effects[{name!r}].raw_mse_delta",
        )
        effects[name] = {"raw_mse_delta": expected_raw}
        if name == "interaction":
            continue
        submitted_relative = _finite(
            effect["relative_improvement"],
            label=f"factor_decomposition.effects[{name!r}].relative_improvement",
        )
        expected_relative = expected_raw / nn
        _require_close(
            submitted_relative,
            expected_relative,
            label=(
                f"factor_decomposition.effects[{name!r}]"
                ".relative_improvement"
            ),
        )
        effects[name]["nn_scaled_task_macro_mse_reduction"] = (
            expected_relative
        )
    return cells, effects


def _escape_cell(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _provenance_rows(value: object) -> list[tuple[str, str]]:
    provenance = _mapping(value, label="provenance")
    if not provenance:
        raise ValueError("provenance must not be empty")
    preferred = [name for name in _PROVENANCE_PRIORITY if name in provenance]
    remaining = sorted(set(provenance) - set(preferred))
    rows: list[tuple[str, str]] = []
    for key in [*preferred, *remaining]:
        if not isinstance(key, str) or not _SAFE_KEY.fullmatch(key):
            raise ValueError("provenance keys must be non-empty ASCII keys")
        raw = provenance[key]
        if isinstance(raw, bool):
            rendered = "true" if raw else "false"
        elif isinstance(raw, int):
            rendered = str(raw)
        elif isinstance(raw, float):
            rendered = f"{_finite(raw, label=f'provenance[{key!r}]'):.6f}"
        elif isinstance(raw, str):
            if not raw:
                raise ValueError(f"provenance[{key!r}] must not be empty")
            rendered = _escape_cell(raw)
        else:
            raise TypeError(f"provenance[{key!r}] must be a scalar value")
        rows.append((key, rendered))
    return rows


def _matrix_table(
    matrix: Mapping[str, Mapping[str, float]],
    modes: tuple[str, ...],
) -> list[str]:
    lines = [
        "| train / eval | " + " | ".join(modes) + " |",
        "| --- | " + " | ".join("---:" for _ in modes) + " |",
    ]
    for train_mode in modes:
        values = " | ".join(f"{matrix[train_mode][mode]:.6f}" for mode in modes)
        lines.append(f"| {train_mode} | {values} |")
    return lines


def _validated_task_effects(
    value: object,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    raw_tasks = _mapping(value, label=label)
    if not raw_tasks:
        raise ValueError(f"{label} must not be empty")
    tasks: dict[str, dict[str, Any]] = {}
    for raw_task, raw_row in raw_tasks.items():
        task = _ascii_text(raw_task, label=f"{label} task")
        row = _mapping(raw_row, label=f"{label}[{task!r}]")
        _exact_keys(
            row,
            {
                "candidate",
                "control",
                "relative_improvement",
                "episode_count",
            },
            label=f"{label}[{task!r}]",
        )
        candidate = _finite(
            row.get("candidate"), label=f"{label}[{task!r}].candidate"
        )
        if candidate < 0.0:
            raise ValueError(f"{label}[{task!r}].candidate must be non-negative")
        control = _finite(
            row.get("control"), label=f"{label}[{task!r}].control"
        )
        if control <= 0.0:
            raise ValueError(f"{label}[{task!r}].control must be positive")
        relative_improvement = _finite(
            row.get("relative_improvement"),
            label=f"{label}[{task!r}].relative_improvement",
        )
        expected_relative_improvement = 1.0 - candidate / control
        _require_close(
            relative_improvement,
            expected_relative_improvement,
            label=f"{label}[{task!r}].relative_improvement",
        )
        tasks[task] = {
            "candidate": candidate,
            "control": control,
            "relative_improvement": expected_relative_improvement,
            "episode_count": _positive_int(
                row.get("episode_count"),
                label=f"{label}[{task!r}].episode_count",
            ),
        }
    return {task: tasks[task] for task in sorted(tasks)}


def _validated_pair_effect(
    value: object,
    *,
    label: str,
    extra_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    effect = _mapping(value, label=label)
    _exact_keys(
        effect,
        _PAIR_EFFECT_FIELDS | set(extra_fields),
        label=label,
    )
    tasks = _validated_task_effects(effect.get("per_task"), label=f"{label}.per_task")
    task_count = _positive_int(effect.get("task_count"), label=f"{label}.task_count")
    if task_count != len(tasks):
        raise ValueError(f"{label}.task_count must match per_task")
    episode_count = _positive_int(
        effect.get("episode_count"), label=f"{label}.episode_count"
    )
    if episode_count != sum(row["episode_count"] for row in tasks.values()):
        raise ValueError(f"{label}.episode_count must equal the per-task total")

    estimand = _ascii_text(effect.get("estimand"), label=f"{label}.estimand")
    if estimand != _DEVELOPMENT_GATE_ESTIMAND:
        raise ValueError(f"{label}.estimand does not match the frozen protocol")

    expected_candidate_macro = math.fsum(
        row["candidate"] for row in tasks.values()
    ) / task_count
    expected_control_macro = math.fsum(
        row["control"] for row in tasks.values()
    ) / task_count
    expected_ratio = math.fsum(
        row["candidate"] / row["control"] for row in tasks.values()
    ) / task_count
    expected_relative_improvement = 1.0 - expected_ratio

    candidate_macro = _finite(
        effect.get("candidate_macro"), label=f"{label}.candidate_macro"
    )
    control_macro = _finite(
        effect.get("control_macro"), label=f"{label}.control_macro"
    )
    candidate_to_control_ratio = _finite(
        effect.get("candidate_to_control_ratio"),
        label=f"{label}.candidate_to_control_ratio",
    )
    relative_improvement = _finite(
        effect.get("relative_improvement"),
        label=f"{label}.relative_improvement",
    )
    _require_close(
        candidate_macro,
        expected_candidate_macro,
        label=f"{label}.candidate_macro",
    )
    _require_close(
        control_macro,
        expected_control_macro,
        label=f"{label}.control_macro",
    )
    _require_close(
        candidate_to_control_ratio,
        expected_ratio,
        label=f"{label}.candidate_to_control_ratio",
    )
    _require_close(
        relative_improvement,
        expected_relative_improvement,
        label=f"{label}.relative_improvement",
    )

    confidence_interval = _interval(
        effect.get("relative_improvement_95_ci"),
        label=f"{label}.relative_improvement_95_ci",
    )
    if confidence_interval[1] > 1.0 + _STAT_ABSOLUTE_TOLERANCE:
        raise ValueError(
            f"{label}.relative_improvement_95_ci endpoints cannot exceed 1 "
            "for non-negative error ratios"
        )
    ratio_upper = _finite(
        effect.get("candidate_to_control_ratio_one_sided_95_upper"),
        label=f"{label}.candidate_to_control_ratio_one_sided_95_upper",
    )
    if ratio_upper < 0.0:
        raise ValueError(
            f"{label}.candidate_to_control_ratio_one_sided_95_upper "
            "must be non-negative"
        )
    return {
        "estimand": estimand,
        "candidate_macro": expected_candidate_macro,
        "control_macro": expected_control_macro,
        "candidate_to_control_ratio": expected_ratio,
        "candidate_to_control_ratio_one_sided_95_upper": ratio_upper,
        "relative_improvement": expected_relative_improvement,
        "relative_improvement_95_ci": confidence_interval,
        "per_task": tasks,
        "episode_count": episode_count,
        "task_count": task_count,
        "passed": _boolean(effect.get("passed"), label=f"{label}.passed"),
    }


def _clean_ratio_margin(
    gate: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> float:
    explicit = []
    for field in ("ratio_upper_margin", "margin"):
        if field in clean:
            explicit.append(
                _finite(
                    clean[field],
                    label=f"development_gate.clean_noninferiority.{field}",
                )
            )
    if "clean_ratio_upper_margin" in gate:
        explicit.append(
            _finite(
                gate["clean_ratio_upper_margin"],
                label="development_gate.clean_ratio_upper_margin",
            )
        )
    for margin in explicit:
        _require_close(
            margin,
            _DEVELOPMENT_GATE_CLEAN_MARGIN,
            label="development_gate clean ratio upper margin",
        )
    return _DEVELOPMENT_GATE_CLEAN_MARGIN


def _validated_representation_row(
    value: object,
    *,
    label: str,
    ratio_minimum: float,
) -> dict[str, Any]:
    row = _mapping(value, label=label)
    _exact_keys(
        row,
        {"samples", *_REPRESENTATION_METRICS, "passed"},
        label=label,
    )
    result: dict[str, Any] = {
        "samples": _positive_int(row.get("samples"), label=f"{label}.samples")
    }
    for metric in _REPRESENTATION_METRICS:
        raw_metric = _mapping(row.get(metric), label=f"{label}.{metric}")
        _exact_keys(
            raw_metric,
            {"candidate", "control", "ratio", "passed"},
            label=f"{label}.{metric}",
        )
        candidate = _finite(
            raw_metric.get("candidate"), label=f"{label}.{metric}.candidate"
        )
        if candidate < 0.0:
            raise ValueError(f"{label}.{metric}.candidate must be non-negative")
        control = _finite(
            raw_metric.get("control"), label=f"{label}.{metric}.control"
        )
        if control <= 0.0:
            raise ValueError(f"{label}.{metric}.control must be positive")
        ratio = _finite(raw_metric.get("ratio"), label=f"{label}.{metric}.ratio")
        expected_ratio = candidate / control
        _require_close(ratio, expected_ratio, label=f"{label}.{metric}.ratio")
        passed = _boolean(
            raw_metric.get("passed"), label=f"{label}.{metric}.passed"
        )
        expected_passed = bool(
            candidate >= _REPRESENTATION_ABSOLUTE_MINIMUM[metric]
            and expected_ratio >= ratio_minimum
        )
        if passed != expected_passed:
            raise ValueError(
                f"{label}.{metric}.passed disagrees with frozen absolute and "
                "candidate/control ratio thresholds"
            )
        result[metric] = {
            "candidate": candidate,
            "control": control,
            "ratio": expected_ratio,
            "passed": expected_passed,
        }
    passed = _boolean(row.get("passed"), label=f"{label}.passed")
    expected_passed = all(
        result[metric]["passed"] for metric in _REPRESENTATION_METRICS
    )
    if passed != expected_passed:
        raise ValueError(f"{label}.passed disagrees with its metric checks")
    result["passed"] = expected_passed
    return result


def _validated_representation(
    value: object,
    *,
    expected_tasks: set[str],
) -> dict[str, Any]:
    representation = _mapping(
        value, label="development_gate.representation_noncollapse"
    )
    _exact_keys(
        representation,
        {"overall", "per_task", "passed"},
        label="development_gate.representation_noncollapse",
    )
    overall = _validated_representation_row(
        representation.get("overall"),
        label="development_gate.representation_noncollapse.overall",
        ratio_minimum=_REPRESENTATION_OVERALL_RATIO_MINIMUM,
    )
    raw_tasks = _mapping(
        representation.get("per_task"),
        label="development_gate.representation_noncollapse.per_task",
    )
    if any(not isinstance(task, str) for task in raw_tasks):
        raise TypeError("development_gate representation task names must be strings")
    if set(raw_tasks) != expected_tasks:
        raise ValueError(
            "development_gate representation task set must match primary per_task"
        )
    per_task = {
        task: _validated_representation_row(
            raw_tasks[task],
            label=f"development_gate.representation_noncollapse.per_task[{task!r}]",
            ratio_minimum=_REPRESENTATION_PER_TASK_RATIO_MINIMUM,
        )
        for task in sorted(raw_tasks)
    }
    passed = _boolean(
        representation.get("passed"),
        label="development_gate.representation_noncollapse.passed",
    )
    expected_passed = bool(
        overall["passed"] and all(row["passed"] for row in per_task.values())
    )
    if passed != expected_passed:
        raise ValueError(
            "development_gate representation passed disagrees with its checks"
        )
    return {"overall": overall, "per_task": per_task, "passed": expected_passed}


def _validated_secondary_holm(value: object) -> list[dict[str, Any]]:
    raw_rows = _sequence(value, label="development_gate.secondary_holm_family")
    rows: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(raw_rows):
        label = f"development_gate.secondary_holm_family[{index}]"
        row = _mapping(raw_row, label=label)
        _exact_keys(
            row,
            {
                "name",
                "relative_improvement",
                "p_value",
                "alternative",
                "holm_threshold",
                "holm_adjusted_p",
                "rejected",
            },
            label=label,
        )
        name = _ascii_text(row.get("name"), label=f"{label}.name", safe_key=True)
        if name in rows:
            raise ValueError(f"duplicate development_gate secondary contrast: {name}")
        rows[name] = {
            "name": name,
            "relative_improvement": _finite(
                row.get("relative_improvement"),
                label=f"{label}.relative_improvement",
            ),
            "p_value": _probability(row.get("p_value"), label=f"{label}.p_value"),
            "alternative": _ascii_text(
                row.get("alternative"),
                label=f"{label}.alternative",
                safe_key=True,
            ),
            "holm_threshold": _probability(
                row.get("holm_threshold"), label=f"{label}.holm_threshold"
            ),
            "holm_adjusted_p": _probability(
                row.get("holm_adjusted_p"), label=f"{label}.holm_adjusted_p"
            ),
            "rejected": _boolean(
                row.get("rejected"), label=f"{label}.rejected"
            ),
        }
    if set(rows) != set(_SECONDARY_HOLM_ORDER):
        raise ValueError(
            "development_gate.secondary_holm_family must contain the complete "
            "canonical contrast family"
        )
    canonical_rows = [rows[name] for name in _SECONDARY_HOLM_ORDER]
    for row in canonical_rows:
        if row["alternative"] != _DEVELOPMENT_GATE_SIGN_FLIP_ALTERNATIVE:
            raise ValueError(
                "development_gate secondary sign-flip alternative must be "
                f"{_DEVELOPMENT_GATE_SIGN_FLIP_ALTERNATIVE!r}"
            )
    canonical_rank = {
        name: index for index, name in enumerate(_SECONDARY_HOLM_ORDER)
    }
    ordered_rows = sorted(
        canonical_rows,
        key=lambda row: (row["p_value"], canonical_rank[row["name"]]),
    )
    running_adjusted = 0.0
    continue_rejecting = True
    family_size = len(ordered_rows)
    for rank, row in enumerate(ordered_rows):
        multiplier = family_size - rank
        expected_threshold = _DEVELOPMENT_GATE_HOLM_ALPHA / multiplier
        running_adjusted = max(
            running_adjusted,
            min(1.0, multiplier * row["p_value"]),
        )
        expected_rejected = bool(
            continue_rejecting and row["p_value"] <= expected_threshold
        )
        name = row["name"]
        _require_close(
            row["holm_threshold"],
            expected_threshold,
            label=f"development_gate.secondary_holm_family[{name!r}].holm_threshold",
        )
        _require_close(
            row["holm_adjusted_p"],
            running_adjusted,
            label=f"development_gate.secondary_holm_family[{name!r}].holm_adjusted_p",
        )
        if row["rejected"] != expected_rejected:
            raise ValueError(
                "development_gate.secondary_holm_family"
                f"[{name!r}].rejected disagrees with Holm(alpha=0.05)"
            )
        row["holm_threshold"] = expected_threshold
        row["holm_adjusted_p"] = running_adjusted
        row["rejected"] = expected_rejected
        if not expected_rejected:
            continue_rejecting = False
    return canonical_rows


def _validated_development_gate(value: object) -> dict[str, Any]:
    gate = _mapping(value, label="development_gate")
    _exact_keys(gate, _DEVELOPMENT_GATE_FIELDS, label="development_gate")
    schema_version = _positive_int(
        gate.get("schema_version"), label="development_gate.schema_version"
    )
    if schema_version != 1:
        raise ValueError("development_gate.schema_version must be 1")
    if gate.get("development_only") is not True:
        raise ValueError("development_gate must be development_only")
    protocol = _ascii_text(
        gate.get("protocol"), label="development_gate.protocol", safe_key=True
    )
    if protocol != _DEVELOPMENT_GATE_PROTOCOL:
        raise ValueError("unexpected development_gate protocol")

    primary = _validated_pair_effect(
        gate.get("primary"), label="development_gate.primary"
    )
    if primary["task_count"] != _DEVELOPMENT_GATE_EXPECTED_TASKS:
        raise ValueError("development_gate.primary must contain exactly four tasks")
    if primary["episode_count"] != _DEVELOPMENT_GATE_PRIMARY_EPISODES:
        raise ValueError("development_gate.primary must contain 112 mild episodes")
    expected_primary_passed = bool(
        primary["relative_improvement"]
        >= _DEVELOPMENT_GATE_PRIMARY_MINIMUM_EFFECT
        and primary["relative_improvement_95_ci"][0]
        > _DEVELOPMENT_GATE_PRIMARY_CI_LOWER_MINIMUM
    )
    if primary["passed"] != expected_primary_passed:
        raise ValueError(
            "development_gate.primary.passed disagrees with the frozen effect "
            "and CI thresholds"
        )
    primary["passed"] = expected_primary_passed

    raw_clean = _mapping(
        gate.get("clean_noninferiority"),
        label="development_gate.clean_noninferiority",
    )
    clean = _validated_pair_effect(
        raw_clean, label="development_gate.clean_noninferiority"
    )
    if set(clean["per_task"]) != set(primary["per_task"]):
        raise ValueError("development_gate clean task set must match primary per_task")
    if clean["task_count"] != _DEVELOPMENT_GATE_EXPECTED_TASKS:
        raise ValueError(
            "development_gate.clean_noninferiority must contain exactly four tasks"
        )
    if clean["episode_count"] != _DEVELOPMENT_GATE_CLEAN_EPISODES:
        raise ValueError(
            "development_gate.clean_noninferiority must contain 56 clean episodes"
        )
    clean["ratio_upper_margin"] = _clean_ratio_margin(gate, raw_clean)
    expected_clean_passed = bool(
        clean["candidate_to_control_ratio_one_sided_95_upper"]
        < clean["ratio_upper_margin"]
    )
    if clean["passed"] != expected_clean_passed:
        raise ValueError(
            "development_gate clean passed disagrees with upper bound and margin"
        )
    clean["passed"] = expected_clean_passed

    raw_direction = _mapping(
        gate.get("task_direction"), label="development_gate.task_direction"
    )
    _exact_keys(
        raw_direction,
        {"positive_tasks", "task_count", "worst_relative_improvement", "passed"},
        label="development_gate.task_direction",
    )
    submitted_positive_tasks = _nonnegative_int(
        raw_direction.get("positive_tasks"),
        label="development_gate.task_direction.positive_tasks",
    )
    submitted_task_count = _positive_int(
        raw_direction.get("task_count"),
        label="development_gate.task_direction.task_count",
    )
    submitted_worst = _finite(
        raw_direction.get("worst_relative_improvement"),
        label="development_gate.task_direction.worst_relative_improvement",
    )
    submitted_direction_passed = _boolean(
        raw_direction.get("passed"),
        label="development_gate.task_direction.passed",
    )
    primary_task_effects = [
        row["relative_improvement"] for row in primary["per_task"].values()
    ]
    expected_positive_tasks = sum(effect > 0.0 for effect in primary_task_effects)
    expected_worst = min(primary_task_effects)
    expected_direction_passed = bool(
        expected_positive_tasks >= _DEVELOPMENT_GATE_MINIMUM_POSITIVE_TASKS
        and expected_worst >= -_DEVELOPMENT_GATE_MAXIMUM_TASK_DEGRADATION
    )
    if submitted_positive_tasks != expected_positive_tasks:
        raise ValueError(
            "development_gate.task_direction.positive_tasks disagrees with primary"
        )
    if submitted_task_count != primary["task_count"]:
        raise ValueError("development_gate task counts disagree")
    _require_close(
        submitted_worst,
        expected_worst,
        label="development_gate.task_direction.worst_relative_improvement",
    )
    if submitted_direction_passed != expected_direction_passed:
        raise ValueError(
            "development_gate.task_direction.passed disagrees with frozen thresholds"
        )
    task_direction = {
        "positive_tasks": expected_positive_tasks,
        "task_count": primary["task_count"],
        "worst_relative_improvement": expected_worst,
        "passed": expected_direction_passed,
    }

    raw_unseen = _mapping(
        gate.get("corrector_train_unseen_sensitivity"),
        label="development_gate.corrector_train_unseen_sensitivity",
    )
    unseen = _validated_pair_effect(
        raw_unseen,
        label="development_gate.corrector_train_unseen_sensitivity",
        extra_fields=frozenset({"selection_disclosure"}),
    )
    disclosure = _ascii_text(
        raw_unseen.get("selection_disclosure"),
        label=(
            "development_gate.corrector_train_unseen_sensitivity."
            "selection_disclosure"
        ),
    )
    if "not an independent test" not in disclosure.lower():
        raise ValueError(
            "development_gate unseen disclosure must state not an independent test"
        )
    unseen["selection_disclosure"] = disclosure
    primary_tasks = set(primary["per_task"])
    if set(unseen["per_task"]) != primary_tasks:
        raise ValueError("development_gate unseen task set must match primary per_task")
    if unseen["task_count"] != _DEVELOPMENT_GATE_EXPECTED_TASKS:
        raise ValueError(
            "development_gate corrector-train-unseen sensitivity must contain "
            "exactly four tasks"
        )
    if unseen["episode_count"] != _DEVELOPMENT_GATE_UNSEEN_MILD_EPISODES:
        raise ValueError(
            "development_gate corrector-train-unseen sensitivity must contain "
            "29 mild episodes"
        )
    unseen_task_effects = [
        row["relative_improvement"] for row in unseen["per_task"].values()
    ]
    expected_unseen_passed = bool(
        unseen["relative_improvement"] > 0.0
        and sum(effect > 0.0 for effect in unseen_task_effects)
        >= _DEVELOPMENT_GATE_MINIMUM_POSITIVE_TASKS
        and min(unseen_task_effects)
        >= -_DEVELOPMENT_GATE_MAXIMUM_TASK_DEGRADATION
    )
    if unseen["passed"] != expected_unseen_passed:
        raise ValueError(
            "development_gate corrector-train-unseen passed disagrees with "
            "frozen directional thresholds"
        )
    unseen["passed"] = expected_unseen_passed

    representation = _validated_representation(
        gate.get("representation_noncollapse"), expected_tasks=primary_tasks
    )
    secondary = _validated_secondary_holm(gate.get("secondary_holm_family"))

    raw_conditions = _mapping(
        gate.get("conditions"), label="development_gate.conditions"
    )
    _exact_keys(
        raw_conditions,
        set(_GATE_CONDITION_ORDER),
        label="development_gate.conditions",
    )
    conditions = {
        name: _boolean(
            raw_conditions[name], label=f"development_gate.conditions[{name!r}]"
        )
        for name in _GATE_CONDITION_ORDER
    }
    section_results = {
        "primary_superiority": primary["passed"],
        "clean_noninferiority": clean["passed"],
        "task_direction": task_direction["passed"],
        "representation_noncollapse": representation["passed"],
        "corrector_train_unseen_directional_agreement": unseen["passed"],
    }
    if conditions != section_results:
        raise ValueError("development_gate conditions disagree with section results")
    passed = _boolean(gate.get("passed"), label="development_gate.passed")
    if passed != all(conditions.values()):
        raise ValueError("development_gate passed disagrees with conditions")
    decision = _ascii_text(
        gate.get("decision"), label="development_gate.decision", safe_key=True
    )
    expected_decision = (
        "fresh_multiseed_outer_holdout_confirmation_candidate"
        if passed
        else "no_go"
    )
    if decision != expected_decision:
        raise ValueError("development_gate decision disagrees with passed")
    return {
        "protocol": protocol,
        "decision": decision,
        "passed": passed,
        "conditions": conditions,
        "primary": primary,
        "clean_noninferiority": clean,
        "task_direction": task_direction,
        "corrector_train_unseen_sensitivity": unseen,
        "representation_noncollapse": representation,
        "secondary_holm_family": secondary,
    }


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _task_effect_table(tasks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "| task | PI mean normalized error | TC mean normalized error | episode-equal relative effect | episodes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, row in tasks.items():
        lines.append(
            f"| {_escape_cell(task)} | {row['candidate']:.6f} | "
            f"{row['control']:.6f} | {row['relative_improvement']:.6f} | "
            f"{row['episode_count']} |"
        )
    return lines


def _development_gate_lines(value: object) -> list[str]:
    gate = _validated_development_gate(value)
    primary = gate["primary"]
    primary_low, primary_high = primary["relative_improvement_95_ci"]
    clean = gate["clean_noninferiority"]
    direction = gate["task_direction"]
    unseen = gate["corrector_train_unseen_sensitivity"]
    unseen_low, unseen_high = unseen["relative_improvement_95_ci"]
    representation = gate["representation_noncollapse"]

    lines = [
        "## Development Selection Gate",
        "",
        "This is a post-hoc single-seed development pilot, not a sealed test result.",
        (
            "Under deterministic replay of the current code against the frozen "
            "data, the 168 main-set episodes partition 127+41. They are "
            "conservatively treated as corrector-exposed: 127 replay into "
            "gradient training and 41 into validation/best-checkpoint selection."
        ),
        (
            "This replay-based classification must not be presented as a "
            "historical fact about exact legacy-run episode exposure."
        ),
        "The main 168-episode set is therefore not an independent test.",
        (
            "The corrector-train-unseen slice is a deterministic legacy split "
            "replay, not historical checkpoint provenance: the legacy checkpoint "
            "did not persist exact episode membership or training-code provenance."
        ),
        (
            "Inference is conditional on seed 5120 and this fixed four-task set; "
            "it does not support population inference over seeds or tasks."
        ),
        (
            "Generic Go/No-Go fields in the raw 4x4 cell reports are not inputs "
            "to, and are not part of, this development gate."
        ),
        "",
        "| field | value |",
        "| --- | --- |",
        f"| protocol | {gate['protocol']} |",
        f"| decision | {gate['decision']} |",
        f"| passed | {_status(gate['passed'])} |",
        "",
        "### Gate Conditions",
        "",
        "| condition | result |",
        "| --- | --- |",
        *(
            f"| {name} | {_status(result)} |"
            for name, result in gate["conditions"].items()
        ),
        "",
        "### Primary PI vs TC (mild; Episode/Task-Equal)",
        "",
        (
            "The primary episode/task-equal relative effect is 1 minus the mean "
            "of task-specific PI/TC error ratios. Within each task, candidate and "
            "control errors are equal-horizon, equal-episode means; across tasks, "
            "all four tasks have equal weight. It is not 1 minus the ratio of the "
            "PI and TC task macros."
        ),
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| PI macro normalized error | {primary['candidate_macro']:.6f} |",
        f"| TC macro normalized error | {primary['control_macro']:.6f} |",
        (
            "| episode/task-equal relative effect | "
            f"{primary['relative_improvement']:.6f} |"
        ),
        f"| paired bootstrap 95% CI low | {primary_low:.6f} |",
        f"| paired bootstrap 95% CI high | {primary_high:.6f} |",
        f"| passed | {_status(primary['passed'])} |",
        "",
        "#### Primary PI vs TC by Task",
        "",
        *_task_effect_table(primary["per_task"]),
        "",
        (
            f"Task-direction check: {_status(direction['passed'])}; positive tasks: "
            f"{direction['positive_tasks']}/{direction['task_count']}; worst task "
            f"relative effect: {direction['worst_relative_improvement']:.6f}."
        ),
        "",
        "### Clean Noninferiority PI vs TC",
        "",
        (
            "Clean evidence is task-macro noninferiority only; it does not "
            "establish per-task clean noninferiority."
        ),
        "",
        "| metric | value |",
        "| --- | ---: |",
        (
            "| PI/TC one-sided 95% ratio upper bound | "
            f"{clean['candidate_to_control_ratio_one_sided_95_upper']:.6f} |"
        ),
        f"| protocol ratio upper margin | {clean['ratio_upper_margin']:.6f} |",
        f"| passed | {_status(clean['passed'])} |",
        "",
        "### Corrector-Gradient-Unseen Sensitivity",
        "",
        f"Disclosure: {unseen['selection_disclosure']}",
        "",
        "| metric | value |",
        "| --- | ---: |",
        (
            "| PI vs TC episode/task-equal relative effect | "
            f"{unseen['relative_improvement']:.6f} |"
        ),
        f"| paired bootstrap 95% CI low | {unseen_low:.6f} |",
        f"| paired bootstrap 95% CI high | {unseen_high:.6f} |",
        f"| directional agreement passed | {_status(unseen['passed'])} |",
        "",
        "#### Corrector-Gradient-Unseen PI vs TC by Task",
        "",
        *_task_effect_table(unseen["per_task"]),
        "",
        "### Representation Noncollapse Checks",
        "",
        (
            "Frozen checks require PI effective rank >= 8.0, PI participation "
            "ratio >= 4.0, and PI/TC >= 0.9 overall or >= 0.8 per task."
        ),
        "",
        "| scope | metric | samples | PI | TC | PI/TC | result |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    representation_scopes = [
        ("overall", representation["overall"]),
        *representation["per_task"].items(),
    ]
    for scope, row in representation_scopes:
        for metric in _REPRESENTATION_METRICS:
            check = row[metric]
            lines.append(
                f"| {_escape_cell(scope)} | {metric} | {row['samples']} | "
                f"{check['candidate']:.6f} | {check['control']:.6f} | "
                f"{check['ratio']:.6f} | {_status(check['passed'])} |"
            )
    lines.extend(
        [
            "",
            f"Representation noncollapse: {_status(representation['passed'])}.",
            "",
            "### Secondary Holm Family",
            "",
            "Secondary contrasts cannot rescue a primary-gate failure.",
            (
                "PI and PV are pre-LayerNorm raw-source-information-equivalent "
                "only; no equivalence is claimed after LayerNorm. Their matched "
                "comparison cannot prove an independent causal effect of the "
                "innovation channel."
            ),
            (
                "Each raw p-value is one-sided with alternative=greater for a "
                "positive episode/task-equal candidate effect. Holm(alpha=0.05) "
                "covers exactly the three displayed prespecified mild-condition "
                "secondary contrasts."
            ),
            "",
            "| contrast | episode/task-equal relative effect | one-sided raw p (greater) | Holm threshold | Holm-adjusted p | rejected |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in gate["secondary_holm_family"]:
        lines.append(
            f"| {row['name']} | {row['relative_improvement']:.6f} | "
            f"{row['p_value']:.6f} | {row['holm_threshold']:.6f} | "
            f"{row['holm_adjusted_p']:.6f} | {_status(row['rejected'])} |"
        )
    return lines


def render_anchor_visual_markdown(
    summary: Mapping[str, Any],
    development_gate: Mapping[str, Any] | None = None,
) -> str:
    """Validate and render an anchor-visual development summary as Markdown."""

    root = _mapping(summary, label="summary")
    embedded_gate = "development_gate" in root
    if embedded_gate and development_gate is not None:
        raise ValueError(
            "external development_gate must not override summary.development_gate"
        )
    if embedded_gate:
        development_gate = _mapping(
            root["development_gate"], label="summary.development_gate"
        )

    if "protocol" in root:
        summary_protocol = _ascii_text(
            root["protocol"], label="summary.protocol", safe_key=True
        )
        if summary_protocol != _FORMAL_SUMMARY_PROTOCOL:
            raise ValueError("unexpected anchor-visual matrix summary protocol")
    required = {
        "development_only",
        "modes",
        "train_by_eval",
        "relative_improvement_vs_triplet_nominal",
        "paired_episode_statistics",
        "paired_episode_bootstrap",
        "factor_decomposition",
        "provenance",
    }
    missing = sorted(required - set(root))
    if development_gate is None:
        missing.append("development_gate")
    if missing:
        raise ValueError(
            "formal anchor-visual summary is incomplete: "
            f"missing={missing}"
        )
    if root.get("development_only") is not True:
        raise ValueError(
            "formal anchor-visual summary must be development_only=true"
        )
    validated_gate = _validated_development_gate(development_gate)
    modes = _validated_modes(root)
    mse, relative = _validated_element_pooled_matrix(root, modes)
    provenance = _provenance_rows(root.get("provenance"))

    paired_baseline, paired_cells, paired_factor_cells, paired_counts = (
        _validated_paired_episode_statistics(
            root["paired_episode_statistics"], modes
        )
    )
    gate_counts = {
        task: (
            validated_gate["primary"]["per_task"][task]["episode_count"]
            + validated_gate["clean_noninferiority"]["per_task"][task]["episode_count"]
        )
        for task in validated_gate["primary"]["per_task"]
    }
    if paired_counts != gate_counts:
        raise ValueError(
            "paired_episode_statistics episode population disagrees with "
            "development_gate primary+clean slices"
        )
    bootstrap = _validated_bootstrap(
        root["paired_episode_bootstrap"],
        modes,
        expected_baseline=paired_baseline,
        paired_cells=paired_cells,
    )
    decomposition = _validated_factor_decomposition(
        root["factor_decomposition"], expected_cells=paired_factor_cells
    )

    lines = [
        "> **DEVELOPMENT VALIDATION ONLY**",
        ">",
        "> This is not a sealed test result and must not be presented as final test evidence.",
        "",
        "# LeWM Anchor-Visual Input Matrix",
        "",
        (
            "The matrix, factor decomposition, and development gate use distinct "
            "estimands: element-pooled cell MSE, task-macro cell MSE, and an "
            "episode/task-equal primary effect, respectively."
        ),
        "",
        "## Element-Pooled Prediction MSE Matrix (train rows, eval columns)",
        "",
        *_matrix_table(mse, modes),
        "",
        (
            "## Element-Pooled Relative MSE Reduction vs "
            "triplet_nominal/triplet_nominal"
        ),
        "",
        (
            "Each value is recomputed from the displayed element-pooled cell MSE "
            "and NN baseline; positive values indicate lower MSE."
        ),
        "",
        *_matrix_table(relative, modes),
        "",
        "## Paired Episode Bootstrap (Episode-Within-Task/Task-Equal)",
        "",
    ]
    baseline, paired_cells = bootstrap
    lines.extend(
        [
            f"Baseline: train={baseline[0]}, eval={baseline[1]}.",
            (
                "Intervals resample paired episodes within each task, recompute "
                "the task-specific baseline denominator, and give each task "
                "equal weight."
            ),
            "",
            "| train | eval | task-equal paired relative effect | task-macro absolute MSE delta | CI low | CI high | P(effect > 0) | tasks | episodes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for train_mode in modes:
        for eval_mode in modes:
            cell = paired_cells[(train_mode, eval_mode)]
            low, high = cell["bootstrap_95_ci"]
            lines.append(
                f"| {train_mode} | {eval_mode} | "
                f"{cell['macro_relative_improvement']:.6f} | "
                f"{cell['macro_absolute_delta']:.6f} | "
                f"{low:.6f} | {high:.6f} | "
                f"{cell['probability_improvement']:.6f} | "
                f"{cell['num_tasks']} | {cell['num_episodes']} |"
            )
    lines.extend(
        ["", "## Task-Macro Nominal/Corrected 2x2 Factor Decomposition", ""]
    )
    factor_cells, effects = decomposition
    lines.extend(
        [
            "N = triplet_nominal; C = triplet_corrected.",
            (
                "Cells are equal-task macro MSE values. Main and matched effects "
                "are recomputed from these four cells; positive raw deltas for "
                "those effects mean lower MSE."
            ),
            (
                "NN-scaled task-macro MSE reductions use NN as a common "
                "denominator. Interaction is CC-CN-NC+NN on the MSE outcome "
                "scale: a positive value means CC is worse (higher MSE) than "
                "the additive expectation; only a negative value indicates "
                "extra synergistic MSE reduction."
            ),
            "",
            "| train / eval | N | C |",
            "| --- | ---: | ---: |",
            f"| N | {factor_cells['NN']:.6f} | {factor_cells['NC']:.6f} |",
            f"| C | {factor_cells['CN']:.6f} | {factor_cells['CC']:.6f} |",
            "",
            "| effect | raw task-macro MSE delta | NN-scaled task-macro MSE reduction |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in _FACTOR_EFFECT_ORDER:
        effect = effects[name]
        relative_value = effect.get("nn_scaled_task_macro_mse_reduction")
        relative_text = (
            f"{relative_value:.6f}" if relative_value is not None else "N/A"
        )
        lines.append(
            f"| {name} | {effect['raw_mse_delta']:.6f} | "
            f"{relative_text} |"
        )
    lines.extend(["", *_development_gate_lines(development_gate)])
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "| field | value |",
            "| --- | --- |",
            *(f"| {key} | {value} |" for key, value in provenance),
            "",
        ]
    )
    rendered = "\n".join(lines)
    rendered.encode("ascii")
    return rendered
