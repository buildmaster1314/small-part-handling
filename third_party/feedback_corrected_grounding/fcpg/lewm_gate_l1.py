"""Preregistered Gate L1 analysis for the grounded-LeWM 2x2 matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


TRAIN_GROUNDINGS = ("nominal", "fcpg")
EVAL_GROUNDINGS = ("nominal", "fcpg")
PRIMARY_HORIZONS = (1, 3, 5)

DEFAULT_THRESHOLDS = {
    "relative_improvement_min": 0.10,
    "clean_regression_max": 0.05,
    "task_direction_min": 3,
    "rank_ratio_min": 0.90,
    "per_task_rank_ratio_min": 0.80,
    "effective_rank_absolute_min": 8.0,
    "participation_ratio_absolute_min": 4.0,
}


def _cell(
    matrix: Mapping[str, Any],
    condition: str,
    train_grounding: str,
    eval_grounding: str,
) -> Mapping[str, Any]:
    try:
        report = matrix[condition][train_grounding][eval_grounding]
    except KeyError as error:
        raise ValueError(
            "incomplete Gate L1 matrix cell: "
            f"condition={condition} train={train_grounding} eval={eval_grounding}"
        ) from error
    if not isinstance(report, Mapping):
        raise TypeError("Gate L1 matrix reports must be mappings")
    return report


def _episode_endpoints(
    report: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("evaluator report is missing metrics")
    blocks = metrics.get("episode_blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise ValueError("evaluator report is missing episode_blocks")
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            raise TypeError("episode block must be a mapping")
        key = (str(block["task_name"]), int(block["episode_id"]))
        if key in result:
            raise ValueError(f"duplicate episode block: {key}")
        endpoints = block.get("endpoints")
        if not isinstance(endpoints, Mapping):
            raise ValueError(f"episode block {key} has no endpoints")
        result[key] = endpoints
    if not result:
        raise ValueError("evaluator report contains no episode blocks")
    return result


def _ratio(endpoint: Mapping[str, Any]) -> float:
    predicted = float(endpoint["predicted_mse"])
    identity = float(endpoint["hold_last_identity_mse"])
    if not np.isfinite(predicted) or not np.isfinite(identity) or identity <= 0.0:
        raise ValueError("episode endpoint has invalid prediction/identity MSE")
    return predicted / identity


def _aggregate_episode_ratios(
    matrix: Mapping[str, Any],
    *,
    conditions: Sequence[str],
    train_grounding: str,
    eval_grounding: str,
    horizons: Sequence[int] = PRIMARY_HORIZONS,
) -> dict[str, dict[int, float]]:
    values: dict[tuple[str, int], list[float]] = {}
    expected_keys: set[tuple[str, int]] | None = None
    for condition in conditions:
        blocks = _episode_endpoints(
            _cell(matrix, condition, train_grounding, eval_grounding)
        )
        keys = set(blocks)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(
                f"episode identity differs across camera conditions at {condition}"
            )
        for key, endpoints in blocks.items():
            for horizon in horizons:
                endpoint = endpoints.get(str(horizon))
                if not isinstance(endpoint, Mapping):
                    raise ValueError(f"episode {key} has no H={horizon} endpoint")
                values.setdefault(key, []).append(_ratio(endpoint))
    result: dict[str, dict[int, float]] = {}
    for (task, episode), ratios in values.items():
        result.setdefault(task, {})[episode] = float(np.mean(ratios))
    return result


def _task_means(values: Mapping[str, Mapping[int, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("no task values to aggregate")
    result = {}
    for task, episodes in values.items():
        if not episodes:
            raise ValueError(f"task {task!r} has no episodes")
        result[task] = float(np.mean(list(episodes.values())))
    return result


def _macro(values: Mapping[str, float]) -> float:
    if not values:
        raise ValueError("no task values for macro aggregation")
    return float(np.mean(list(values.values())))


def _paired_bootstrap(
    nominal: Mapping[str, Mapping[int, float]],
    fcpg: Mapping[str, Mapping[int, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if set(nominal) != set(fcpg):
        raise ValueError("nominal/FCPG task sets differ")
    paired: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task in sorted(nominal):
        if set(nominal[task]) != set(fcpg[task]):
            raise ValueError(f"nominal/FCPG episode sets differ for task {task}")
        episode_ids = sorted(nominal[task])
        paired[task] = (
            np.asarray([nominal[task][episode] for episode in episode_ids]),
            np.asarray([fcpg[task][episode] for episode in episode_ids]),
        )
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    relative = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        nominal_tasks = []
        fcpg_tasks = []
        for nominal_values, fcpg_values in paired.values():
            indices = rng.integers(0, len(nominal_values), size=len(nominal_values))
            nominal_tasks.append(float(nominal_values[indices].mean()))
            fcpg_tasks.append(float(fcpg_values[indices].mean()))
        nominal_macro = float(np.mean(nominal_tasks))
        fcpg_macro = float(np.mean(fcpg_tasks))
        differences[sample] = nominal_macro - fcpg_macro
        relative[sample] = differences[sample] / max(nominal_macro, 1e-12)
    return {
        "samples": samples,
        "seed": seed,
        "method": "paired_episode_within_task_equal_task_macro",
        "difference_95_ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "relative_improvement_95_ci": [
            float(np.quantile(relative, 0.025)),
            float(np.quantile(relative, 0.975)),
        ],
    }


def _representation_metric(
    report: Mapping[str, Any], metric: str, task: str | None = None
) -> float:
    metrics = report["metrics"]
    representation = metrics["representation"]
    values = representation["overall"] if task is None else representation["per_task"][task]
    value = float(values[metric])
    if not np.isfinite(value):
        raise ValueError(f"non-finite representation metric {metric}")
    return value


def _validate_common_targets_and_hold_last(
    matrix: Mapping[str, Any], conditions: Sequence[str]
) -> dict[str, Any]:
    encoder_hashes = set()
    condition_target_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, set[str]] = {name: set() for name in TRAIN_GROUNDINGS}
    for condition in conditions:
        reference_hold: dict[tuple[str, int, str], float] | None = None
        target_hashes = set()
        for train in TRAIN_GROUNDINGS:
            for evaluation in EVAL_GROUNDINGS:
                report = _cell(matrix, condition, train, evaluation)
                if report.get("grounding_mode") != evaluation:
                    raise ValueError("report grounding_mode disagrees with matrix cell")
                checkpoint_hashes[train].add(str(report.get("checkpoint_sha256")))
                target_manifest = report.get("oracle_target_cache")
                if (
                    not isinstance(target_manifest, Sequence)
                    or len(target_manifest) != 1
                    or not isinstance(target_manifest[0], Mapping)
                ):
                    raise ValueError("each Gate L1 report requires one oracle target cache")
                target_hashes.add(str(target_manifest[0].get("array_sha256")))
                encoder_hashes.add(
                    str(target_manifest[0].get("encoder_checkpoint_sha256"))
                )
                blocks = _episode_endpoints(report)
                hold = {
                    (task, episode, horizon): float(endpoint["hold_last_identity_mse"])
                    for (task, episode), endpoints in blocks.items()
                    for horizon, endpoint in endpoints.items()
                }
                if reference_hold is None:
                    reference_hold = hold
                elif hold != reference_hold:
                    raise ValueError(
                        f"hold-last targets differ across 2x2 cells for {condition}"
                    )
        if len(target_hashes) != 1:
            raise ValueError(f"oracle target differs across 2x2 cells for {condition}")
        condition_target_hashes[condition] = next(iter(target_hashes))
    if len(encoder_hashes) != 1 or "None" in encoder_hashes:
        raise ValueError("Gate L1 conditions do not share one frozen target encoder")
    if any(len(values) != 1 for values in checkpoint_hashes.values()):
        raise ValueError("a train-grounding row silently switched checkpoints")
    return {
        "encoder_checkpoint_sha256": next(iter(encoder_hashes)),
        "condition_target_sha256": condition_target_hashes,
        "checkpoint_sha256": {
            key: next(iter(value)) for key, value in checkpoint_hashes.items()
        },
        "hold_last_identical_across_cells": True,
    }


def summarize_gate_l1(
    matrix: Mapping[str, Any],
    *,
    primary_conditions: Sequence[str] = ("rotation5", "translation4"),
    clean_condition: str = "clean",
    stress_condition: str = "rotation10",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 3072,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Validate all cells and apply the preregistered L1 decision."""

    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        limits.update({key: float(value) for key, value in thresholds.items()})
    all_conditions = [clean_condition, *primary_conditions, stress_condition]
    provenance = _validate_common_targets_and_hold_last(matrix, all_conditions)

    nominal_primary = _aggregate_episode_ratios(
        matrix,
        conditions=primary_conditions,
        train_grounding="nominal",
        eval_grounding="nominal",
    )
    fcpg_primary = _aggregate_episode_ratios(
        matrix,
        conditions=primary_conditions,
        train_grounding="fcpg",
        eval_grounding="fcpg",
    )
    nominal_task = _task_means(nominal_primary)
    fcpg_task = _task_means(fcpg_primary)
    nominal_macro = _macro(nominal_task)
    fcpg_macro = _macro(fcpg_task)
    relative_improvement = (nominal_macro - fcpg_macro) / max(nominal_macro, 1e-12)
    bootstrap = _paired_bootstrap(
        nominal_primary,
        fcpg_primary,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    task_improvements = {
        task: (nominal_task[task] - fcpg_task[task]) / max(nominal_task[task], 1e-12)
        for task in sorted(nominal_task)
    }
    positive_tasks = sum(value > 0.0 for value in task_improvements.values())
    hold_last_tasks = {task: value < 1.0 for task, value in fcpg_task.items()}

    nominal_clean = _macro(
        _task_means(
            _aggregate_episode_ratios(
                matrix,
                conditions=(clean_condition,),
                train_grounding="nominal",
                eval_grounding="nominal",
            )
        )
    )
    fcpg_clean = _macro(
        _task_means(
            _aggregate_episode_ratios(
                matrix,
                conditions=(clean_condition,),
                train_grounding="fcpg",
                eval_grounding="fcpg",
            )
        )
    )
    clean_regression = (fcpg_clean - nominal_clean) / max(nominal_clean, 1e-12)

    intervention = {}
    for train in TRAIN_GROUNDINGS:
        nominal_eval = _macro(
            _task_means(
                _aggregate_episode_ratios(
                    matrix,
                    conditions=primary_conditions,
                    train_grounding=train,
                    eval_grounding="nominal",
                )
            )
        )
        fcpg_eval = _macro(
            _task_means(
                _aggregate_episode_ratios(
                    matrix,
                    conditions=primary_conditions,
                    train_grounding=train,
                    eval_grounding="fcpg",
                )
            )
        )
        intervention[train] = {
            "nominal_eval": nominal_eval,
            "fcpg_eval": fcpg_eval,
            "fcpg_eval_improves": fcpg_eval < nominal_eval,
        }

    rank = {"overall": {}, "per_task": {}}
    for metric in ("effective_rank", "participation_ratio"):
        nominal_values = [
            _representation_metric(
                _cell(matrix, condition, "nominal", "nominal"), metric
            )
            for condition in primary_conditions
        ]
        fcpg_values = [
            _representation_metric(_cell(matrix, condition, "fcpg", "fcpg"), metric)
            for condition in primary_conditions
        ]
        nominal_value = float(np.mean(nominal_values))
        fcpg_value = float(np.mean(fcpg_values))
        absolute_minimum = limits[
            "effective_rank_absolute_min"
            if metric == "effective_rank"
            else "participation_ratio_absolute_min"
        ]
        rank["overall"][metric] = {
            "nominal": nominal_value,
            "fcpg": fcpg_value,
            "ratio": fcpg_value / max(nominal_value, 1e-12),
            "absolute_minimum": absolute_minimum,
        }
    for task in sorted(nominal_task):
        rank["per_task"][task] = {}
        for metric in ("effective_rank", "participation_ratio"):
            nominal_value = float(
                np.mean(
                    [
                        _representation_metric(
                            _cell(matrix, condition, "nominal", "nominal"),
                            metric,
                            task,
                        )
                        for condition in primary_conditions
                    ]
                )
            )
            fcpg_value = float(
                np.mean(
                    [
                        _representation_metric(
                            _cell(matrix, condition, "fcpg", "fcpg"),
                            metric,
                            task,
                        )
                        for condition in primary_conditions
                    ]
                )
            )
            rank["per_task"][task][metric] = {
                "nominal": nominal_value,
                "fcpg": fcpg_value,
                "ratio": fcpg_value / max(nominal_value, 1e-12),
            }

    rank_overall_pass = all(
        values["ratio"] >= limits["rank_ratio_min"]
        and values["fcpg"] >= values["absolute_minimum"]
        for values in rank["overall"].values()
    )
    rank_per_task_pass = all(
        values["ratio"] >= limits["per_task_rank_ratio_min"]
        for task_values in rank["per_task"].values()
        for values in task_values.values()
    )
    checks = {
        "relative_improvement_at_least_10_percent": relative_improvement
        >= limits["relative_improvement_min"],
        "paired_bootstrap_ci_excludes_zero": bootstrap["difference_95_ci"][0] > 0.0,
        "at_least_three_tasks_improve": positive_tasks
        >= int(limits["task_direction_min"]),
        "every_task_beats_hold_last": all(hold_last_tasks.values()),
        "clean_regression_within_five_percent": clean_regression
        <= limits["clean_regression_max"],
        "overall_representation_not_collapsed": rank_overall_pass,
        "per_task_representation_not_collapsed": rank_per_task_pass,
        "same_checkpoint_interventions_agree": all(
            values["fcpg_eval_improves"] for values in intervention.values()
        ),
        "matched_diagonal_direction_agrees": fcpg_macro < nominal_macro,
    }
    stress = {}
    for train in TRAIN_GROUNDINGS:
        for evaluation in EVAL_GROUNDINGS:
            report = _cell(matrix, stress_condition, train, evaluation)
            stress[f"train_{train}__eval_{evaluation}"] = report["metrics"][
                "h10_stress"
            ]

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "protocol": "fcpg_grounded_lewm_gate_l1",
        "decision": "Go" if passed else "No-Go",
        "passed": passed,
        "thresholds": limits,
        "checks": checks,
        "provenance": provenance,
        "primary": {
            "conditions": list(primary_conditions),
            "horizons": list(PRIMARY_HORIZONS),
            "nominal_task_macro_ratio": nominal_macro,
            "fcpg_task_macro_ratio": fcpg_macro,
            "relative_improvement": relative_improvement,
            "task_relative_improvement": task_improvements,
            "positive_task_count": positive_tasks,
            "fcpg_beats_hold_last_by_task": hold_last_tasks,
            "bootstrap": bootstrap,
        },
        "clean": {
            "condition": clean_condition,
            "nominal_task_macro_ratio": nominal_clean,
            "fcpg_task_macro_ratio": fcpg_clean,
            "relative_regression": clean_regression,
        },
        "same_checkpoint_intervention": intervention,
        "representation": rank,
        "h10_stress_not_in_gate": stress,
    }


__all__ = [
    "DEFAULT_THRESHOLDS",
    "EVAL_GROUNDINGS",
    "PRIMARY_HORIZONS",
    "TRAIN_GROUNDINGS",
    "summarize_gate_l1",
]
