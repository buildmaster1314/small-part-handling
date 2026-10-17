"""Paired statistics for the LeWM Predictor Isolation Gate.

The independent sampling unit is a physical source trajectory.  Rows are
reduced in the following order before any effect is computed or bootstrapped:

``phase -> event -> camera view -> source trajectory -> task macro``.

Camera views derived from the same source trajectory are paired observations,
not additional physical samples.  Every bootstrap replicate therefore
resamples source trajectories within task and uses the same draws for all
methods.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 18_101
FACTUAL_METHOD = "factual"
BASELINE_METHODS = (
    "hold_last",
    "constant_velocity",
    "zero_action",
    "shuffled_action",
)
MINIMUM_IMPROVEMENT = {
    "hold_last": 0.10,
    "constant_velocity": 0.10,
    "zero_action": 0.05,
    "shuffled_action": 0.05,
}
MINIMUM_TASK_DIRECTION_FRACTION = 0.75
MINIMUM_TASK_COUNT = 4

Identity: TypeAlias = int | str
SupportKey: TypeAlias = tuple[
    str, Identity, Identity, str, Identity, Identity
]
SourceValues: TypeAlias = dict[str, dict[str, np.ndarray]]


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _identity(value: Any, *, label: str) -> Identity:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise TypeError(f"{label} must be an integer or non-empty string")


def _strict_integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _squared_error(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _alias_value(
    record: Mapping[str, Any],
    primary: str,
    aliases: Sequence[str],
    *,
    label: str,
) -> Any:
    names = (primary, *aliases)
    present = [(name, record[name]) for name in names if name in record]
    if not present:
        raise ValueError(f"{label} is required (accepted fields: {list(names)!r})")
    first_name, first_value = present[0]
    for name, value in present[1:]:
        if value != first_value:
            raise ValueError(
                f"conflicting {label} aliases: {first_name}={first_value!r}, "
                f"{name}={value!r}"
            )
    return first_value


def _identity_order(value: Identity) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _stable_mean(values: Sequence[float] | np.ndarray, *, label: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} values must be finite")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    result = scale * float(np.mean(array / scale, dtype=np.float64))
    if not math.isfinite(result):
        raise ValueError(f"{label} mean must remain finite")
    return result


def _parse_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[SupportKey, float]],
    tuple[str, ...],
    dict[str, tuple[Identity, ...]],
    dict[tuple[str, Identity], tuple[tuple[str, Identity], ...]],
    dict[tuple[str, Identity], tuple[Identity, ...]],
]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of mappings")
    if not records:
        raise ValueError("records must not be empty")

    by_method: dict[str, dict[SupportKey, float]] = {}
    view_identity: dict[tuple[str, Identity], tuple[Identity, str]] = {}
    views_by_source_camera: dict[
        tuple[str, Identity], dict[str, set[Identity]]
    ] = {}

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        task = _nonempty_string(record.get("task"), label=f"record {index} task")
        view_episode = _identity(
            _alias_value(
                record,
                "view_episode_id",
                ("physical_episode_id",),
                label=f"record {index} view episode",
            ),
            label=f"record {index} view episode",
        )
        source_trajectory = _identity(
            _alias_value(
                record,
                "source_trajectory_id",
                ("resampling_unit_id",),
                label=f"record {index} source trajectory",
            ),
            label=f"record {index} source trajectory",
        )
        camera_view = _nonempty_string(
            record.get("camera_view"), label=f"record {index} camera_view"
        )
        event = _identity(
            _alias_value(
                record,
                "event_id",
                ("event_index",),
                label=f"record {index} event",
            ),
            label=f"record {index} event",
        )
        phase = _identity(
            _alias_value(
                record,
                "phase_index",
                ("phase",),
                label=f"record {index} phase",
            ),
            label=f"record {index} phase",
        )
        method = _nonempty_string(
            record.get("method"), label=f"record {index} method"
        )
        error = _squared_error(
            _alias_value(
                record,
                "squared_error",
                ("mse",),
                label=f"record {index} squared error",
            ),
            label=f"record {index} squared error",
        )

        view_key = (task, view_episode)
        identity = (source_trajectory, camera_view)
        previous = view_identity.setdefault(view_key, identity)
        if previous != identity:
            raise ValueError(
                f"view episode {view_key!r} changes source trajectory or camera "
                f"view: {previous!r} != {identity!r}"
            )
        views_by_source_camera.setdefault(
            (task, source_trajectory), {}
        ).setdefault(camera_view, set()).add(view_episode)

        key = (
            task,
            view_episode,
            source_trajectory,
            camera_view,
            event,
            phase,
        )
        values = by_method.setdefault(method, {})
        if key in values:
            raise ValueError(
                "duplicate predictor statistic record: "
                f"{(*key, method)!r}"
            )
        values[key] = error

    methods = tuple(sorted(by_method))
    reference_support = set(by_method[methods[0]])
    for method in methods[1:]:
        support = set(by_method[method])
        if support != reference_support:
            raise ValueError(
                f"method support differs for {method!r}: "
                f"missing={len(reference_support - support)} "
                f"unexpected={len(support - reference_support)}"
            )

    source_views: dict[tuple[str, Identity], tuple[tuple[str, Identity], ...]] = {}
    sources_by_task: dict[str, list[Identity]] = {}
    camera_sets_by_task: dict[str, set[tuple[str, ...]]] = {}
    for source_key, by_camera in views_by_source_camera.items():
        task, source = source_key
        repeated = {
            camera: tuple(sorted(views, key=_identity_order))
            for camera, views in by_camera.items()
            if len(views) != 1
        }
        if repeated:
            raise ValueError(
                "each source trajectory must have exactly one view episode per "
                f"camera view; source={source_key!r} repeated={repeated!r}"
            )
        camera_names = tuple(sorted(by_camera))
        if len(camera_names) < 2:
            raise ValueError(
                f"source trajectory {source_key!r} needs at least two paired "
                "camera views"
            )
        camera_sets_by_task.setdefault(task, set()).add(camera_names)
        source_views[source_key] = tuple(
            (camera, next(iter(by_camera[camera]))) for camera in camera_names
        )
        sources_by_task.setdefault(task, []).append(source)

    inconsistent = {
        task: sorted(camera_sets)
        for task, camera_sets in camera_sets_by_task.items()
        if len(camera_sets) != 1
    }
    if inconsistent:
        raise ValueError(
            "paired camera-view support must be identical across source "
            f"trajectories within task: {inconsistent!r}"
        )

    immutable_sources = {
        task: tuple(sorted(sources, key=_identity_order))
        for task, sources in sorted(sources_by_task.items())
    }
    event_phases: dict[tuple[str, Identity, Identity], set[Identity]] = {}
    for task, view, _source, _camera, event, phase in reference_support:
        event_phases.setdefault((task, view, event), set()).add(phase)
    immutable_phases = {
        key: tuple(sorted(phases, key=_identity_order))
        for key, phases in event_phases.items()
    }
    return by_method, methods, immutable_sources, source_views, immutable_phases


def _source_values(
    by_method: Mapping[str, Mapping[SupportKey, float]],
    methods: Sequence[str],
    sources_by_task: Mapping[str, Sequence[Identity]],
    source_views: Mapping[tuple[str, Identity], Sequence[tuple[str, Identity]]],
    phases_by_event: Mapping[tuple[str, Identity, Identity], Sequence[Identity]],
) -> SourceValues:
    events_by_view: dict[tuple[str, Identity], set[Identity]] = {}
    for task, view, event in phases_by_event:
        events_by_view.setdefault((task, view), set()).add(event)

    output: SourceValues = {method: {} for method in methods}
    for method in methods:
        values = by_method[method]
        for task, sources in sources_by_task.items():
            reduced_sources: list[float] = []
            for source in sources:
                view_means: list[float] = []
                for camera_view, view in source_views[(task, source)]:
                    event_means: list[float] = []
                    for event in sorted(
                        events_by_view[(task, view)], key=_identity_order
                    ):
                        phase_values = [
                            values[
                                (
                                    task,
                                    view,
                                    source,
                                    camera_view,
                                    event,
                                    phase,
                                )
                            ]
                            for phase in phases_by_event[(task, view, event)]
                        ]
                        event_means.append(
                            _stable_mean(
                                phase_values,
                                label=f"{task}/{view}/{event}/{method} phases",
                            )
                        )
                    view_means.append(
                        _stable_mean(
                            event_means,
                            label=f"{task}/{source}/{camera_view}/{method} events",
                        )
                    )
                reduced_sources.append(
                    _stable_mean(
                        view_means,
                        label=f"{task}/{source}/{method} paired views",
                    )
                )
            output[method][task] = np.asarray(
                reduced_sources, dtype=np.float64
            )
    return output


def _bootstrap_draws(
    sources: Mapping[str, Sequence[Identity]], *, samples: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {}
    for task, task_sources in sources.items():
        count = len(task_sources)
        draws[task] = rng.integers(0, count, size=(samples, count))
    return draws


def _comparison(
    source_values: SourceValues,
    *,
    factual_method: str,
    baseline_method: str,
    draws: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    tasks = tuple(source_values[factual_method])
    sample_count = next(iter(draws.values())).shape[0]
    bootstrap_improvement = np.zeros(sample_count, dtype=np.float64)
    task_improvements: list[float] = []
    factual_task_mse: list[float] = []
    baseline_task_mse: list[float] = []
    per_task: dict[str, dict[str, float | int | bool]] = {}

    for task in tasks:
        factual = source_values[factual_method][task]
        baseline = source_values[baseline_method][task]
        if factual.shape != baseline.shape or factual.ndim != 1:
            raise RuntimeError(f"paired source arrays differ for task {task!r}")
        if np.any(baseline <= 0.0):
            raise ValueError(
                f"{baseline_method} source-trajectory squared error must be "
                f"positive for ratio statistics in task {task!r}"
            )
        factual_mean = _stable_mean(factual, label=f"{task}/{factual_method}")
        baseline_mean = _stable_mean(baseline, label=f"{task}/{baseline_method}")
        improvement = 1.0 - factual_mean / baseline_mean
        indices = draws[task]
        factual_bootstrap = np.mean(factual[indices], axis=1, dtype=np.float64)
        baseline_bootstrap = np.mean(baseline[indices], axis=1, dtype=np.float64)
        if np.any(baseline_bootstrap <= 0.0):
            raise ValueError(
                f"bootstrap produced a non-positive {baseline_method} "
                f"denominator for task {task!r}"
            )
        task_bootstrap = 1.0 - factual_bootstrap / baseline_bootstrap
        bootstrap_improvement += task_bootstrap / len(tasks)
        task_improvements.append(improvement)
        factual_task_mse.append(factual_mean)
        baseline_task_mse.append(baseline_mean)
        per_task[task] = {
            "factual_mse": factual_mean,
            "baseline_mse": baseline_mean,
            "relative_improvement": improvement,
            "same_direction": improvement > 0.0,
            "source_trajectory_count": int(factual.size),
        }

    if not np.isfinite(bootstrap_improvement).all():
        raise ValueError("bootstrap improvements must remain finite")
    positive_count = sum(
        bool(row["same_direction"]) for row in per_task.values()
    )
    return {
        "factual_method": factual_method,
        "baseline_method": baseline_method,
        "orientation": "positive_means_factual_lower_squared_error",
        "factual_task_macro_mse": _stable_mean(
            factual_task_mse, label=f"{factual_method} task-macro MSE"
        ),
        "baseline_task_macro_mse": _stable_mean(
            baseline_task_mse, label=f"{baseline_method} task-macro MSE"
        ),
        "relative_improvement": _stable_mean(
            task_improvements,
            label=f"{factual_method} over {baseline_method} task-macro effect",
        ),
        "relative_improvement_95_ci": [
            float(np.quantile(bootstrap_improvement, 0.025)),
            float(np.quantile(bootstrap_improvement, 0.975)),
        ],
        "positive_task_count": positive_count,
        "task_count": len(tasks),
        "positive_task_fraction": positive_count / len(tasks),
        "per_task": per_task,
    }


def predictor_isolation_gate_decision(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the fixed, predeclared Predictor Isolation Gate thresholds."""

    if not isinstance(comparisons, Mapping):
        raise TypeError("comparisons must be a mapping")
    missing = sorted(set(BASELINE_METHODS).difference(comparisons))
    if missing:
        raise ValueError(f"required predictor comparisons are absent: {missing}")

    task_counts = {
        _strict_integer(
            comparisons[baseline].get("task_count"),
            label=f"{baseline} task_count",
            minimum=1,
        )
        for baseline in BASELINE_METHODS
    }
    if len(task_counts) != 1:
        raise ValueError("all predictor comparisons must contain the same tasks")
    task_count = next(iter(task_counts))
    required_positive_tasks = math.ceil(
        MINIMUM_TASK_DIRECTION_FRACTION * task_count
    )

    checks: dict[str, dict[str, Any]] = {
        "at_least_four_tasks": {
            "passed": task_count >= MINIMUM_TASK_COUNT,
            "observed": task_count,
            "operator": ">=",
            "threshold": MINIMUM_TASK_COUNT,
        }
    }
    for baseline in BASELINE_METHODS:
        row = comparisons[baseline]
        improvement = _squared_error_or_effect(
            row.get("relative_improvement"), label=f"{baseline} improvement"
        )
        ci = _effect_ci(
            row.get("relative_improvement_95_ci"), label=f"{baseline} CI"
        )
        positive_tasks = _strict_integer(
            row.get("positive_task_count"),
            label=f"{baseline} positive_task_count",
            minimum=0,
        )
        minimum = MINIMUM_IMPROVEMENT[baseline]
        checks[f"factual_over_{baseline}_at_least_minimum"] = {
            "passed": improvement >= minimum,
            "observed": improvement,
            "operator": ">=",
            "threshold": minimum,
        }
        checks[f"factual_over_{baseline}_ci_lower_positive"] = {
            "passed": ci[0] > 0.0,
            "observed": ci[0],
            "operator": ">",
            "threshold": 0.0,
        }
        checks[f"factual_over_{baseline}_task_direction"] = {
            "passed": positive_tasks >= required_positive_tasks,
            "observed": positive_tasks,
            "operator": ">=",
            "threshold": required_positive_tasks,
        }

    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": 1,
        "evaluated": True,
        "thresholds": {
            "minimum_relative_improvement": dict(MINIMUM_IMPROVEMENT),
            "ci_lower_strictly_positive": True,
            "minimum_task_count": MINIMUM_TASK_COUNT,
            "minimum_positive_task_fraction": MINIMUM_TASK_DIRECTION_FRACTION,
            "required_positive_tasks": required_positive_tasks,
        },
        "checks": checks,
        "passed": passed,
        "decision": "Go" if passed else "No-Go",
        "failed_checks": [
            name for name, check in checks.items() if not bool(check["passed"])
        ],
    }


def _squared_error_or_effect(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _effect_ci(value: Any, *, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a two-element sequence")
    if len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    lower = _squared_error_or_effect(value[0], label=f"{label}[0]")
    upper = _squared_error_or_effect(value[1], label=f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} must be ordered")
    return lower, upper


def summarize_predictor_isolation_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    factual_method: str = FACTUAL_METHOD,
    hold_last_method: str = "hold_last",
    constant_velocity_method: str = "constant_velocity",
    zero_action_method: str = "zero_action",
    shuffled_action_method: str = "shuffled_action",
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize one paired Predictor Isolation Gate panel."""

    method_roles = {
        "factual": _nonempty_string(factual_method, label="factual_method"),
        "hold_last": _nonempty_string(
            hold_last_method, label="hold_last_method"
        ),
        "constant_velocity": _nonempty_string(
            constant_velocity_method, label="constant_velocity_method"
        ),
        "zero_action": _nonempty_string(
            zero_action_method, label="zero_action_method"
        ),
        "shuffled_action": _nonempty_string(
            shuffled_action_method, label="shuffled_action_method"
        ),
    }
    if len(set(method_roles.values())) != len(method_roles):
        raise ValueError("all Predictor Isolation Gate method names must differ")
    samples = _strict_integer(
        bootstrap_samples, label="bootstrap_samples", minimum=1
    )
    seed = _strict_integer(bootstrap_seed, label="bootstrap_seed", minimum=0)

    (
        by_method,
        methods,
        sources_by_task,
        source_views,
        phases_by_event,
    ) = _parse_records(records)
    missing = sorted(set(method_roles.values()).difference(methods))
    if missing:
        raise ValueError(f"required Predictor Isolation methods are absent: {missing}")
    source_values = _source_values(
        by_method,
        methods,
        sources_by_task,
        source_views,
        phases_by_event,
    )
    draws = _bootstrap_draws(sources_by_task, samples=samples, seed=seed)
    comparisons = {
        role: _comparison(
            source_values,
            factual_method=method_roles["factual"],
            baseline_method=method_roles[role],
            draws=draws,
        )
        for role in BASELINE_METHODS
    }
    decision = predictor_isolation_gate_decision(comparisons)

    view_count = sum(len(views) for views in source_views.values())
    source_count = sum(len(sources) for sources in sources_by_task.values())
    camera_views_by_task = {
        task: sorted(
            {
                camera
                for source in sources
                for camera, _view in source_views[(task, source)]
            }
        )
        for task, sources in sources_by_task.items()
    }
    return {
        "schema_version": 1,
        "metric": "squared_error",
        "lower_is_better": True,
        "methods": method_roles,
        "aggregation": {
            "phase_reduction": "equal_mean_within_event",
            "event_reduction": "equal_mean_within_view_episode",
            "view_reduction": "equal_mean_of_paired_camera_views_within_source",
            "source_reduction": "equal_mean_within_task",
            "task_reduction": "equal_mean_of_task_relative_improvements",
        },
        "bootstrap": {
            "method": "nonparametric_percentile",
            "unit": ["task", "source_trajectory_id"],
            "stratified_by": "task",
            "shared_draws_across_methods": True,
            "camera_views_are_paired_not_resampled": True,
            "samples": samples,
            "seed": seed,
            "confidence": 0.95,
        },
        "coverage": {
            "support_identical_across_methods": True,
            "task_count": len(sources_by_task),
            "view_episode_count": view_count,
            "source_trajectory_count": source_count,
            "independent_sampling_unit_count": source_count,
            "source_trajectories_per_task": {
                task: len(sources) for task, sources in sources_by_task.items()
            },
            "camera_views_per_task": camera_views_by_task,
            "record_count": sum(len(values) for values in by_method.values()),
        },
        "comparisons": comparisons,
        "gate": decision,
        "decision": decision["decision"],
        "passed": decision["passed"],
    }


summarize_predictor_isolation = summarize_predictor_isolation_statistics


__all__ = [
    "BASELINE_METHODS",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "FACTUAL_METHOD",
    "MINIMUM_IMPROVEMENT",
    "MINIMUM_TASK_COUNT",
    "MINIMUM_TASK_DIRECTION_FRACTION",
    "predictor_isolation_gate_decision",
    "summarize_predictor_isolation",
    "summarize_predictor_isolation_statistics",
]
