"""Strict data contracts for the LeWM anchor-visual input ablation.

This module deliberately contains no model or evaluator code.  It defines the
parameter-compatible input modes, enumerates the complete train-by-evaluation
matrix, and validates already-produced cell reports before aggregating them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


TRIPLET_NOMINAL = "triplet_nominal"
TRIPLET_CORRECTED = "triplet_corrected"
PAIRED_VISUAL = "paired_visual"
PAIRED_INNOVATION = "paired_innovation"

PARAMETER_COMPATIBLE_TRIPLET_MODES = (
    TRIPLET_NOMINAL,
    TRIPLET_CORRECTED,
    PAIRED_VISUAL,
    PAIRED_INNOVATION,
)
TRIPLET_ANCHOR_VISUAL_MODES = PARAMETER_COMPATIBLE_TRIPLET_MODES
REFERENCE_TRIPLET_MODE = TRIPLET_NOMINAL
EXPECTED_OPTIMIZER_STEP = 2_000


@dataclass(frozen=True, slots=True)
class AnchorAblationCell:
    """One parameter-matched train-mode by evaluation-mode intervention."""

    train_mode: str
    eval_mode: str

    def __post_init__(self) -> None:
        validate_anchor_visual_override(self.train_mode, field_name="train_mode")
        validate_anchor_visual_override(self.eval_mode, field_name="eval_mode")

    @property
    def key(self) -> tuple[str, str]:
        return (self.train_mode, self.eval_mode)


def validate_anchor_visual_override(
    mode: str,
    *,
    field_name: str = "anchor_visual_override",
) -> str:
    """Return a legal parameter-compatible override or fail closed.

    ``single_corrected`` is intentionally rejected: its visual projection has a
    different input width, so switching to it would not be a weight-compatible
    evaluation-time intervention.
    """

    if not isinstance(mode, str):
        raise TypeError(f"{field_name} must be a string")
    if mode not in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        allowed = ", ".join(PARAMETER_COMPATIBLE_TRIPLET_MODES)
        raise ValueError(
            f"{field_name} must be a parameter-compatible triplet mode "
            f"({allowed}); got {mode!r}"
        )
    return mode


def validate_triplet_override(mode: str) -> str:
    """Compatibility alias with a shorter, experiment-facing name."""

    return validate_anchor_visual_override(mode)


def build_anchor_ablation_cells() -> tuple[AnchorAblationCell, ...]:
    """Construct the complete, deterministic 4x4 train-by-eval cell set."""

    return tuple(
        AnchorAblationCell(train_mode=train_mode, eval_mode=eval_mode)
        for train_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES
        for eval_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES
    )


def _candidate_values(
    report: Mapping[str, Any],
    names: Sequence[str],
) -> list[Any]:
    containers: list[Mapping[str, Any]] = [report]
    for container_name in ("contract", "provenance"):
        container = report.get(container_name)
        if isinstance(container, Mapping):
            containers.append(container)
    return [container[name] for container in containers for name in names if name in container]


def _required_value(
    report: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
) -> Any:
    values = _candidate_values(report, names)
    if not values:
        raise ValueError(f"anchor ablation report is missing {label}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"anchor ablation report contains conflicting {label} values")
    return first


def _optional_value(report: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    values = _candidate_values(report, names)
    if not values:
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError("anchor ablation report contains conflicting mode values")
    return first


def _strict_int(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _strict_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "None":
        raise ValueError(f"{label} must be a non-empty hash string")
    return value


def _metric_value(report: Mapping[str, Any], metric: str) -> float:
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string")
    if "." in metric:
        value: Any = report
        for component in metric.split("."):
            if not isinstance(value, Mapping) or component not in value:
                raise ValueError(f"report is missing metric path {metric!r}")
            value = value[component]
    else:
        metrics = report.get("metrics")
        if isinstance(metrics, Mapping) and metric in metrics:
            value = metrics[metric]
        elif metric in report:
            value = report[metric]
        else:
            raise ValueError(f"report is missing metric {metric!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric {metric!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metric {metric!r} must be finite")
    return result


def _report_cell(report: Mapping[str, Any]) -> AnchorAblationCell:
    train_mode = _required_value(
        report,
        ("train_mode", "train_anchor_visual_mode", "training_anchor_visual_mode"),
        label="train anchor-visual mode",
    )
    eval_mode = _required_value(
        report,
        ("eval_mode", "eval_anchor_visual_mode", "anchor_visual_override"),
        label="evaluation anchor-visual mode",
    )
    return AnchorAblationCell(str(train_mode), str(eval_mode))


def summarize_anchor_ablation_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    metric: str = "prediction_mse",
    lower_is_better: bool = True,
    expected_parameter_count: int | None = None,
    expected_seed: int | None = None,
    expected_target_sha256: str | None = None,
    expected_data_split_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and aggregate a complete 4x4 anchor-input ablation.

    Every report must identify its train/evaluation modes and provide ``step``,
    parameter count, seed, target hash, data-split hash, and the requested
    metric.  Common aliases are accepted for provenance fields, but duplicate
    aliases within one report must agree.  Missing, duplicate, or unexpected
    cells fail before any aggregate is returned.
    """

    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
        raise TypeError("reports must be a sequence of mappings")
    if not isinstance(lower_is_better, bool):
        raise TypeError("lower_is_better must be a bool")

    expected_cells = {cell.key for cell in build_anchor_ablation_cells()}
    cell_values: dict[tuple[str, str], float] = {}
    parameter_counts: set[int] = set()
    seeds: set[int] = set()
    target_hashes: set[str] = set()
    split_hashes: set[str] = set()

    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("each anchor ablation report must be a mapping")
        cell = _report_cell(report)
        if cell.key in cell_values:
            raise ValueError(
                "duplicate anchor ablation matrix cell: "
                f"train={cell.train_mode} eval={cell.eval_mode}"
            )

        step = _strict_int(
            _required_value(
                report,
                ("step", "optimizer_step", "global_step"),
                label="optimizer step",
            ),
            label="optimizer step",
        )
        if step != EXPECTED_OPTIMIZER_STEP:
            raise ValueError(
                f"anchor ablation cell must use step={EXPECTED_OPTIMIZER_STEP}; "
                f"got {step} for train={cell.train_mode} eval={cell.eval_mode}"
            )

        parameter_counts.add(
            _strict_int(
                _required_value(
                    report,
                    (
                        "parameter_count",
                        "model_parameter_count",
                        "expected_parameter_count",
                    ),
                    label="parameter count",
                ),
                label="parameter count",
                positive=True,
            )
        )
        seeds.add(
            _strict_int(
                _required_value(
                    report,
                    ("seed", "training_seed"),
                    label="training seed",
                ),
                label="training seed",
            )
        )
        target_hashes.add(
            _strict_hash(
                _required_value(
                    report,
                    ("target_sha256", "target_encoder_sha256"),
                    label="target hash",
                ),
                label="target hash",
            )
        )
        split_hashes.add(
            _strict_hash(
                _required_value(
                    report,
                    (
                        "data_split_sha256",
                        "split_sha256",
                        "episode_manifest_sha256",
                    ),
                    label="data split hash",
                ),
                label="data split hash",
            )
        )
        cell_values[cell.key] = _metric_value(report, metric)

    actual_cells = set(cell_values)
    if actual_cells != expected_cells:
        missing = sorted(expected_cells.difference(actual_cells))
        unexpected = sorted(actual_cells.difference(expected_cells))
        raise ValueError(
            "incomplete anchor ablation 4x4 matrix: "
            f"missing={missing} unexpected={unexpected}"
        )
    if len(parameter_counts) != 1:
        raise ValueError("anchor ablation parameter counts differ across cells")
    if len(seeds) != 1:
        raise ValueError("anchor ablation training seeds differ across cells")
    if len(target_hashes) != 1:
        raise ValueError("anchor ablation target hashes differ across cells")
    if len(split_hashes) != 1:
        raise ValueError("anchor ablation data split hashes differ across cells")

    parameter_count = next(iter(parameter_counts))
    seed = next(iter(seeds))
    target_sha256 = next(iter(target_hashes))
    data_split_sha256 = next(iter(split_hashes))
    if expected_parameter_count is not None and parameter_count != _strict_int(
        expected_parameter_count, label="expected_parameter_count", positive=True
    ):
        raise ValueError(
            f"parameter count mismatch: expected {expected_parameter_count}, "
            f"got {parameter_count}"
        )
    if expected_seed is not None and seed != _strict_int(
        expected_seed, label="expected_seed"
    ):
        raise ValueError(f"training seed mismatch: expected {expected_seed}, got {seed}")
    if expected_target_sha256 is not None and target_sha256 != _strict_hash(
        expected_target_sha256, label="expected_target_sha256"
    ):
        raise ValueError("target hash does not match the expected target")
    if (
        expected_data_split_sha256 is not None
        and data_split_sha256
        != _strict_hash(
            expected_data_split_sha256,
            label="expected_data_split_sha256",
        )
    ):
        raise ValueError("data split hash does not match the expected split")

    baseline_key = (REFERENCE_TRIPLET_MODE, REFERENCE_TRIPLET_MODE)
    baseline_value = cell_values[baseline_key]
    if abs(baseline_value) <= 1e-12:
        raise ValueError("triplet_nominal baseline metric must be non-zero")

    matrix: dict[str, dict[str, float]] = {}
    relative: dict[str, dict[str, float]] = {}
    cells = []
    for train_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        matrix[train_mode] = {}
        relative[train_mode] = {}
        for eval_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
            value = cell_values[(train_mode, eval_mode)]
            improvement = (
                (baseline_value - value) / abs(baseline_value)
                if lower_is_better
                else (value - baseline_value) / abs(baseline_value)
            )
            matrix[train_mode][eval_mode] = value
            relative[train_mode][eval_mode] = improvement
            cells.append(
                {
                    "train_mode": train_mode,
                    "eval_mode": eval_mode,
                    "value": value,
                    "relative_improvement_vs_triplet_nominal": improvement,
                }
            )

    return {
        "schema_version": 1,
        "metric": metric,
        "lower_is_better": lower_is_better,
        "modes": list(PARAMETER_COMPATIBLE_TRIPLET_MODES),
        "baseline": {
            "train_mode": REFERENCE_TRIPLET_MODE,
            "eval_mode": REFERENCE_TRIPLET_MODE,
            "value": baseline_value,
        },
        "train_by_eval": matrix,
        "relative_improvement_vs_triplet_nominal": relative,
        "cells": cells,
        "provenance": {
            "optimizer_step": EXPECTED_OPTIMIZER_STEP,
            "parameter_count": parameter_count,
            "seed": seed,
            "target_sha256": target_sha256,
            "data_split_sha256": data_split_sha256,
            "complete_4x4": True,
        },
    }


def summarize_anchor_ablation(
    reports: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias for :func:`summarize_anchor_ablation_reports`."""

    return summarize_anchor_ablation_reports(reports, **kwargs)

