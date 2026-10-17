"""Task-macro trajectory statistics for the randomized posterior oracle.

Rows are already scored on the direct anchor. ``physical_episode_id`` identifies
an evaluated view, while optional ``resampling_unit_id`` identifies the parent
trajectory shared by paired views. The latter defaults to the view identity for
backward compatibility. This module reduces each view first, averages paired
views within their parent, and bootstraps independent parents within task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np


WINDOWS = ("active", "recovery", "clean")
PHASES = (0, 1, 2, 3, 4)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 15_731
_MAX_BOOTSTRAP_INDEX_ELEMENTS = 1_000_000

Identity: TypeAlias = int | str
ViewKey: TypeAlias = tuple[str, Identity]
ResamplingUnitKey: TypeAlias = tuple[str, Identity]
EventKey: TypeAlias = tuple[str, Identity, str, int]
RowKey: TypeAlias = tuple[str, Identity, str, int, int]
ResamplingUnitValues: TypeAlias = dict[str, dict[str, np.ndarray]]


def _strict_integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


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


def _finite_mse(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _finite_threshold(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


def _identity_order(value: Identity) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _view_order(key: ViewKey) -> tuple[Any, ...]:
    return (key[0], _identity_order(key[1]))


def _resampling_unit_order(key: ResamplingUnitKey) -> tuple[Any, ...]:
    return (key[0], _identity_order(key[1]))


def _row_order(key: RowKey) -> tuple[Any, ...]:
    return (
        key[0],
        _identity_order(key[1]),
        WINDOWS.index(key[2]),
        key[3],
        key[4],
    )


def _stable_mean(values: Sequence[float] | np.ndarray, *, label: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} inputs must be finite")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    result = scale * float(np.mean(array / scale, dtype=np.float64))
    if not math.isfinite(result):
        raise ValueError(f"{label} must remain finite")
    return result


def _parse_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[RowKey, float]],
    tuple[str, ...],
    dict[str, tuple[Identity, ...]],
    dict[ViewKey, dict[str, tuple[int, ...]]],
    dict[str, tuple[Identity, ...]],
    dict[ResamplingUnitKey, tuple[Identity, ...]],
    dict[ViewKey, Identity],
]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of mappings")
    if not records:
        raise ValueError("records must not be empty")

    by_method: dict[str, dict[RowKey, float]] = {}
    resampling_unit_by_view: dict[ViewKey, Identity] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        task = _nonempty_string(record.get("task"), label=f"record {index} task")
        episode_id = _identity(
            record.get("physical_episode_id"),
            label=f"record {index} physical_episode_id",
        )
        resampling_unit_id = _identity(
            record.get("resampling_unit_id", episode_id),
            label=f"record {index} resampling_unit_id",
        )
        view_key = (task, episode_id)
        previous_resampling_unit = resampling_unit_by_view.setdefault(
            view_key, resampling_unit_id
        )
        if previous_resampling_unit != resampling_unit_id:
            raise ValueError(
                "resampling_unit_id changes within view episode "
                f"{view_key!r}: {previous_resampling_unit!r} != "
                f"{resampling_unit_id!r}"
            )
        phase = _strict_integer(
            record.get("phase_index"), label=f"record {index} phase_index", minimum=0
        )
        if phase not in PHASES:
            raise ValueError(f"record {index} phase_index must lie in [0, 4]")
        window = _nonempty_string(
            record.get("window"), label=f"record {index} window"
        )
        if window not in WINDOWS:
            raise ValueError(f"record {index} window must be one of {list(WINDOWS)!r}")
        event_index = _strict_integer(
            record.get("event_index"),
            label=f"record {index} event_index",
            minimum=0,
        )
        method = _nonempty_string(
            record.get("method"), label=f"record {index} method"
        )
        mse = _finite_mse(record.get("mse"), label=f"record {index} mse")
        key = (task, episode_id, window, event_index, phase)
        values = by_method.setdefault(method, {})
        if key in values:
            raise ValueError(
                "duplicate randomized oracle statistic record: "
                f"{(task, episode_id, window, event_index, phase, method)!r}"
            )
        values[key] = mse

    methods = tuple(sorted(by_method))
    reference = methods[0]
    reference_support = set(by_method[reference])
    for method in methods[1:]:
        support = set(by_method[method])
        if support != reference_support:
            missing = sorted(reference_support - support, key=_row_order)
            unexpected = sorted(support - reference_support, key=_row_order)
            raise ValueError(
                "method support differs: "
                f"reference={reference!r} method={method!r} "
                f"missing={missing[:3]} unexpected={unexpected[:3]}"
            )

    event_phases: dict[EventKey, set[int]] = {}
    for task, episode_id, window, event_index, phase in reference_support:
        event_phases.setdefault((task, episode_id, window, event_index), set()).add(
            phase
        )
    incomplete = [
        (key, tuple(sorted(phases)))
        for key, phases in event_phases.items()
        if phases != set(PHASES)
    ]
    if incomplete:
        incomplete.sort(key=lambda item: _row_order((*item[0], 0)))
        raise ValueError(
            "every task/episode/window/event/method must contain exactly phases "
            f"0..4; incomplete={incomplete[:3]}"
        )

    view_windows: dict[ViewKey, dict[str, set[int]]] = {}
    for task, episode_id, window, event_index in event_phases:
        view_windows.setdefault((task, episode_id), {}).setdefault(
            window, set()
        ).add(event_index)
    missing_windows = [
        (view, sorted(set(WINDOWS) - set(windows)))
        for view, windows in view_windows.items()
        if set(windows) != set(WINDOWS)
    ]
    if missing_windows:
        missing_windows.sort(key=lambda item: _view_order(item[0]))
        raise ValueError(
            "every view episode must contain active, recovery, and clean "
            f"windows; missing={missing_windows[:3]}"
        )
    noncontiguous = [
        (view, window, tuple(sorted(indices)))
        for view, windows in view_windows.items()
        for window, indices in windows.items()
        if sorted(indices) != list(range(max(indices) + 1))
    ]
    if noncontiguous:
        noncontiguous.sort(
            key=lambda item: (*_view_order(item[0]), item[1])
        )
        raise ValueError(
            "event indices must start at zero and remain contiguous within "
            f"each view/window; invalid={noncontiguous[:3]}"
        )
    no_primary = [
        view
        for view, windows in view_windows.items()
        if not any(event_index != 0 for event_index in windows["active"])
    ]
    if no_primary:
        no_primary.sort(key=_view_order)
        raise ValueError(
            "every view episode needs an active event after event_index 0; "
            f"views={no_primary[:3]}"
        )

    views: dict[str, set[Identity]] = {}
    for task, episode_id in view_windows:
        views.setdefault(task, set()).add(episode_id)
    views_by_task = {
        task: tuple(sorted(task_views, key=_identity_order))
        for task, task_views in sorted(views.items())
    }
    immutable_windows = {
        view: {
            window: tuple(sorted(indices)) for window, indices in windows.items()
        }
        for view, windows in sorted(
            view_windows.items(), key=lambda item: _view_order(item[0])
        )
    }
    resampling_units: dict[str, set[Identity]] = {}
    view_sets_by_resampling_unit: dict[ResamplingUnitKey, set[Identity]] = {}
    for (task, view_id), resampling_unit_id in resampling_unit_by_view.items():
        resampling_units.setdefault(task, set()).add(resampling_unit_id)
        view_sets_by_resampling_unit.setdefault(
            (task, resampling_unit_id), set()
        ).add(view_id)
    resampling_units_by_task = {
        task: tuple(sorted(task_units, key=_identity_order))
        for task, task_units in sorted(resampling_units.items())
    }
    views_by_resampling_unit = {
        unit: tuple(sorted(unit_views, key=_identity_order))
        for unit, unit_views in sorted(
            view_sets_by_resampling_unit.items(),
            key=lambda item: _resampling_unit_order(item[0]),
        )
    }
    return (
        by_method,
        methods,
        views_by_task,
        immutable_windows,
        resampling_units_by_task,
        views_by_resampling_unit,
        resampling_unit_by_view,
    )


def _resampling_unit_values(
    by_method: Mapping[str, Mapping[RowKey, float]],
    methods: Sequence[str],
    resampling_units_by_task: Mapping[str, Sequence[Identity]],
    views_by_resampling_unit: Mapping[ResamplingUnitKey, Sequence[Identity]],
    events_by_view: Mapping[ViewKey, Mapping[str, Sequence[int]]],
    *,
    window: str,
    exclude_event_zero: bool,
) -> ResamplingUnitValues:
    result: ResamplingUnitValues = {method: {} for method in methods}
    for method in methods:
        values = by_method[method]
        for task, resampling_units in resampling_units_by_task.items():
            reduced: list[float] = []
            for resampling_unit_id in resampling_units:
                view_means: list[float] = []
                for view_id in views_by_resampling_unit[
                    (task, resampling_unit_id)
                ]:
                    events = events_by_view[(task, view_id)][window]
                    selected_events = tuple(
                        event
                        for event in events
                        if not (exclude_event_zero and event == 0)
                    )
                    event_means = [
                        _stable_mean(
                            [
                                values[(task, view_id, window, event_index, phase)]
                                for phase in PHASES
                            ],
                            label=(
                                f"{task}/{view_id}/{window}/{event_index}/{method} "
                                "phase MSE"
                            ),
                        )
                        for event_index in selected_events
                    ]
                    view_means.append(
                        _stable_mean(
                            event_means,
                            label=f"{task}/{view_id}/{window}/{method} event MSE",
                        )
                    )
                reduced.append(
                    _stable_mean(
                        view_means,
                        label=(
                            f"{task}/{resampling_unit_id}/{window}/{method} "
                            "paired view MSE"
                        ),
                    )
                )
            result[method][task] = np.asarray(reduced, dtype=np.float64)
    return result


def _task_point_effects(
    primary: ResamplingUnitValues,
    recovery: ResamplingUnitValues,
    clean: ResamplingUnitValues,
    *,
    view_counts_by_task: Mapping[str, int],
    measurement_method: str,
    candidate_method: str,
    time_only_method: str,
) -> tuple[dict[str, dict[str, float | int]], dict[str, float]]:
    tasks = tuple(primary[measurement_method])
    rows: dict[str, dict[str, float | int]] = {}
    method_task_mse: dict[str, list[float]] = {
        method: [] for method in primary
    }
    for task in tasks:
        means = {
            method: _stable_mean(values[task], label=f"{task}/{method} primary MSE")
            for method, values in primary.items()
        }
        measurement = means[measurement_method]
        if measurement <= 0.0:
            raise ValueError(
                f"task {task!r} has a zero primary measurement MSE denominator"
            )
        for method, value in means.items():
            method_task_mse[method].append(value)
        candidate = means[candidate_method]
        time_only = means[time_only_method]
        recovery_excess = _stable_mean(
            recovery[candidate_method][task] - recovery[measurement_method][task],
            label=f"{task} recovery candidate excess",
        )
        clean_excess = _stable_mean(
            clean[candidate_method][task] - clean[measurement_method][task],
            label=f"{task} clean candidate excess",
        )
        independent_count = int(primary[measurement_method][task].size)
        rows[task] = {
            "measurement_mse": measurement,
            "candidate_mse": candidate,
            "time_only_mse": time_only,
            "candidate_relative_improvement": (measurement - candidate) / measurement,
            "time_only_relative_improvement": (measurement - time_only) / measurement,
            "candidate_incremental_improvement_over_time_only": (
                time_only - candidate
            )
            / measurement,
            "recovery_excess_mse": recovery_excess,
            "recovery_normalized_excess_mse": recovery_excess / measurement,
            "clean_excess_mse": clean_excess,
            "clean_normalized_excess_mse": clean_excess / measurement,
            "view_episode_count": int(view_counts_by_task[task]),
            "physical_episode_count": independent_count,
            "physical_trajectory_count": independent_count,
            "independent_trajectory_count": independent_count,
            "resampling_unit_count": independent_count,
        }
    macro_mse = {
        method: _stable_mean(task_values, label=f"{method} task-macro primary MSE")
        for method, task_values in method_task_mse.items()
    }
    return rows, macro_mse


def _bootstrap_effects(
    primary: ResamplingUnitValues,
    recovery: ResamplingUnitValues,
    clean: ResamplingUnitValues,
    *,
    measurement_method: str,
    candidate_method: str,
    time_only_method: str,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    names = (
        "candidate_relative_improvement",
        "time_only_relative_improvement",
        "candidate_incremental_improvement_over_time_only",
        "recovery_excess_mse",
        "recovery_normalized_excess_mse",
        "clean_excess_mse",
        "clean_normalized_excess_mse",
    )
    output = {name: np.zeros(samples, dtype=np.float64) for name in names}
    tasks = tuple(primary[measurement_method])
    rng = np.random.default_rng(seed)

    for task in tasks:
        measurement = primary[measurement_method][task]
        candidate = primary[candidate_method][task]
        time_only = primary[time_only_method][task]
        if np.any(measurement <= 0.0):
            raise ValueError(
                f"task {task!r} contains a zero primary measurement MSE "
                "independent-trajectory denominator"
            )
        recovery_difference = (
            recovery[candidate_method][task] - recovery[measurement_method][task]
        )
        clean_difference = clean[candidate_method][task] - clean[measurement_method][
            task
        ]
        independent_count = int(measurement.size)
        arrays = (candidate, time_only, recovery_difference, clean_difference)
        if any(array.shape != measurement.shape for array in arrays):
            raise RuntimeError(
                f"paired independent-trajectory arrays differ for task {task!r}"
            )
        chunk_size = max(
            1,
            min(samples, _MAX_BOOTSTRAP_INDEX_ELEMENTS // max(1, independent_count)),
        )
        for start in range(0, samples, chunk_size):
            stop = min(start + chunk_size, samples)
            indices = rng.integers(
                0, independent_count, size=(stop - start, independent_count)
            )
            measurement_mean = np.mean(measurement[indices], axis=1, dtype=np.float64)
            if np.any(measurement_mean <= 0.0):
                raise ValueError(
                    f"bootstrap produced a zero primary measurement denominator "
                    f"for task {task!r}"
                )
            candidate_mean = np.mean(candidate[indices], axis=1, dtype=np.float64)
            time_mean = np.mean(time_only[indices], axis=1, dtype=np.float64)
            weight = 1.0 / len(tasks)
            output["candidate_relative_improvement"][start:stop] += (
                (measurement_mean - candidate_mean) / measurement_mean * weight
            )
            output["time_only_relative_improvement"][start:stop] += (
                (measurement_mean - time_mean) / measurement_mean * weight
            )
            output[
                "candidate_incremental_improvement_over_time_only"
            ][start:stop] += (
                (time_mean - candidate_mean) / measurement_mean * weight
            )
            recovery_excess = np.mean(
                recovery_difference[indices], axis=1, dtype=np.float64
            )
            clean_excess = np.mean(
                clean_difference[indices], axis=1, dtype=np.float64
            )
            output["recovery_excess_mse"][start:stop] += recovery_excess * weight
            output["recovery_normalized_excess_mse"][start:stop] += (
                recovery_excess / measurement_mean * weight
            )
            output["clean_excess_mse"][start:stop] += clean_excess * weight
            output["clean_normalized_excess_mse"][start:stop] += (
                clean_excess / measurement_mean * weight
            )
    if any(not np.isfinite(values).all() for values in output.values()):
        raise ValueError("randomized oracle bootstrap effects must remain finite")
    return output


def _ci(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _diagnostic(
    name: str,
    values: ResamplingUnitValues,
    per_task: Mapping[str, Mapping[str, float | int]],
    bootstrap: Mapping[str, np.ndarray],
    *,
    measurement_method: str,
    candidate_method: str,
) -> dict[str, Any]:
    excess_field = f"{name}_excess_mse"
    normalized_field = f"{name}_normalized_excess_mse"
    candidate_task_mse = [
        _stable_mean(values[candidate_method][task], label=f"{task}/{name} candidate")
        for task in per_task
    ]
    measurement_task_mse = [
        _stable_mean(
            values[measurement_method][task], label=f"{task}/{name} measurement"
        )
        for task in per_task
    ]
    excess = [float(row[excess_field]) for row in per_task.values()]
    normalized = [float(row[normalized_field]) for row in per_task.values()]
    return {
        "window": name,
        "candidate_method": candidate_method,
        "measurement_method": measurement_method,
        "orientation": "candidate_mse_minus_measurement_mse",
        "normalization": "each_task_primary_measurement_mse",
        "feasibility_only": True,
        "non_evidentiary": True,
        "hard_go_check": True,
        "candidate_task_macro_mse": _stable_mean(
            candidate_task_mse, label=f"{name} candidate task-macro MSE"
        ),
        "measurement_task_macro_mse": _stable_mean(
            measurement_task_mse, label=f"{name} measurement task-macro MSE"
        ),
        "excess_mse": _stable_mean(excess, label=f"{name} task-macro excess MSE"),
        "excess_mse_95_ci": _ci(bootstrap[excess_field]),
        "excess_mse_one_sided_95_upper": float(
            np.quantile(bootstrap[excess_field], 0.95)
        ),
        "normalized_excess_mse": _stable_mean(
            normalized, label=f"{name} task-macro normalized excess MSE"
        ),
        "normalized_excess_mse_95_ci": _ci(bootstrap[normalized_field]),
        "normalized_excess_mse_one_sided_95_upper": float(
            np.quantile(bootstrap[normalized_field], 0.95)
        ),
        "per_task": {
            task: {
                "excess_mse": float(row[excess_field]),
                "normalized_excess_mse": float(row[normalized_field]),
                "primary_measurement_mse": float(row["measurement_mse"]),
                "view_episode_count": int(row["view_episode_count"]),
                "physical_episode_count": int(row["physical_episode_count"]),
                "physical_trajectory_count": int(row["physical_trajectory_count"]),
                "independent_trajectory_count": int(
                    row["independent_trajectory_count"]
                ),
                "resampling_unit_count": int(row["resampling_unit_count"]),
            }
            for task, row in per_task.items()
        },
    }


def summarize_randomized_oracle(
    records: Sequence[Mapping[str, Any]],
    *,
    measurement_method: str,
    candidate_method: str,
    time_only_method: str,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize a variable-duration randomized posterior-oracle panel."""

    measurement_name = _nonempty_string(
        measurement_method, label="measurement_method"
    )
    candidate_name = _nonempty_string(candidate_method, label="candidate_method")
    time_name = _nonempty_string(time_only_method, label="time_only_method")
    if len({measurement_name, candidate_name, time_name}) != 3:
        raise ValueError("measurement, candidate, and time-only methods must differ")
    samples = _strict_integer(
        bootstrap_samples, label="bootstrap_samples", minimum=1
    )
    seed = _strict_integer(bootstrap_seed, label="bootstrap_seed", minimum=0)

    (
        by_method,
        methods,
        views_by_task,
        events_by_view,
        resampling_units_by_task,
        views_by_resampling_unit,
        resampling_unit_by_view,
    ) = _parse_records(records)
    missing_methods = sorted(
        {measurement_name, candidate_name, time_name}.difference(methods)
    )
    if missing_methods:
        raise ValueError(f"required randomized oracle methods are absent: {missing_methods}")

    primary = _resampling_unit_values(
        by_method,
        methods,
        resampling_units_by_task,
        views_by_resampling_unit,
        events_by_view,
        window="active",
        exclude_event_zero=True,
    )
    recovery = _resampling_unit_values(
        by_method,
        methods,
        resampling_units_by_task,
        views_by_resampling_unit,
        events_by_view,
        window="recovery",
        exclude_event_zero=False,
    )
    clean = _resampling_unit_values(
        by_method,
        methods,
        resampling_units_by_task,
        views_by_resampling_unit,
        events_by_view,
        window="clean",
        exclude_event_zero=False,
    )
    per_task, macro_mse = _task_point_effects(
        primary,
        recovery,
        clean,
        view_counts_by_task={
            task: len(views) for task, views in views_by_task.items()
        },
        measurement_method=measurement_name,
        candidate_method=candidate_name,
        time_only_method=time_name,
    )
    bootstrap = _bootstrap_effects(
        primary,
        recovery,
        clean,
        measurement_method=measurement_name,
        candidate_method=candidate_name,
        time_only_method=time_name,
        samples=samples,
        seed=seed,
    )

    candidate_effect = _stable_mean(
        [float(row["candidate_relative_improvement"]) for row in per_task.values()],
        label="candidate task-macro relative improvement",
    )
    time_effect = _stable_mean(
        [float(row["time_only_relative_improvement"]) for row in per_task.values()],
        label="time-only task-macro relative improvement",
    )
    incremental_effect = _stable_mean(
        [
            float(row["candidate_incremental_improvement_over_time_only"])
            for row in per_task.values()
        ],
        label="candidate task-macro incremental improvement",
    )
    candidate_ci = _ci(bootstrap["candidate_relative_improvement"])
    time_ci = _ci(bootstrap["time_only_relative_improvement"])
    incremental_ci = _ci(
        bootstrap["candidate_incremental_improvement_over_time_only"]
    )

    view_coverage = []
    for (task, episode_id), windows in events_by_view.items():
        view_coverage.append(
            {
                "task": task,
                "physical_episode_id": episode_id,
                "resampling_unit_id": resampling_unit_by_view[(task, episode_id)],
                "active_event_count": len(windows["active"]),
                "primary_active_event_count": sum(
                    event != 0 for event in windows["active"]
                ),
                "recovery_event_count": len(windows["recovery"]),
                "clean_event_count": len(windows["clean"]),
            }
        )
    minimum_active = min(row["active_event_count"] for row in view_coverage)
    minimum_recovery = min(row["recovery_event_count"] for row in view_coverage)
    view_episode_count = len(events_by_view)
    independent_trajectory_count = sum(
        len(units) for units in resampling_units_by_task.values()
    )
    view_episodes_per_task = {
        task: len(views) for task, views in views_by_task.items()
    }
    independent_trajectories_per_task = {
        task: len(units) for task, units in resampling_units_by_task.items()
    }
    coverage = {
        "support_identical_across_methods": True,
        "exactly_five_phases_per_event_method": True,
        "task_count": len(resampling_units_by_task),
        "view_episode_count": view_episode_count,
        "physical_episode_count": independent_trajectory_count,
        "physical_trajectory_count": independent_trajectory_count,
        "independent_trajectory_count": independent_trajectory_count,
        "resampling_unit_count": independent_trajectory_count,
        "record_count": sum(len(values) for values in by_method.values()),
        "all_methods": list(methods),
        "episodes_per_task": independent_trajectories_per_task,
        "view_episodes_per_task": view_episodes_per_task,
        "physical_trajectories_per_task": independent_trajectories_per_task,
        "independent_trajectories_per_task": independent_trajectories_per_task,
        "resampling_units_per_task": independent_trajectories_per_task,
        "minimum_active_events_per_view_episode": minimum_active,
        "minimum_recovery_events_per_view_episode": minimum_recovery,
        "minimum_active_events_per_episode": minimum_active,
        "minimum_recovery_events_per_episode": minimum_recovery,
        "at_least_two_active_events_per_view_episode": minimum_active >= 2,
        "at_least_two_recovery_events_per_view_episode": minimum_recovery >= 2,
        "at_least_two_active_events_per_episode": minimum_active >= 2,
        "at_least_two_recovery_events_per_episode": minimum_recovery >= 2,
        "per_episode": view_coverage,
        "per_view_episode": view_coverage,
    }
    primary_summary = {
        "window": "active",
        "excluded_event_indices": [0],
        "event_weighting": "equal_within_view_episode",
        "view_weighting": "equal_within_resampling_unit",
        "resampling_unit_weighting": "equal_within_task",
        "episode_weighting": "equal_resampling_units_within_task",
        "task_weighting": "equal",
        "method_task_macro_mse": macro_mse,
        "measurement_mse": macro_mse[measurement_name],
        "candidate_mse": macro_mse[candidate_name],
        "time_only_mse": macro_mse[time_name],
        "candidate_relative_improvement": candidate_effect,
        "candidate_relative_improvement_95_ci": candidate_ci,
        "time_only_relative_improvement": time_effect,
        "time_only_relative_improvement_95_ci": time_ci,
        "candidate_incremental_improvement_over_time_only": incremental_effect,
        "candidate_incremental_improvement_over_time_only_95_ci": incremental_ci,
        "candidate_vs_measurement": {
            "candidate_method": candidate_name,
            "control_method": measurement_name,
            "relative_improvement": candidate_effect,
            "relative_improvement_95_ci": candidate_ci,
        },
        "time_only_vs_measurement": {
            "candidate_method": time_name,
            "control_method": measurement_name,
            "relative_improvement": time_effect,
            "relative_improvement_95_ci": time_ci,
        },
        "candidate_over_time_only": {
            "candidate_method": candidate_name,
            "time_only_method": time_name,
            "normalization": "each_task_primary_measurement_mse",
            "incremental_improvement": incremental_effect,
            "incremental_improvement_95_ci": incremental_ci,
        },
        "per_task": per_task,
    }
    recovery_diagnostic = _diagnostic(
        "recovery",
        recovery,
        per_task,
        bootstrap,
        measurement_method=measurement_name,
        candidate_method=candidate_name,
    )
    clean_diagnostic = _diagnostic(
        "clean",
        clean,
        per_task,
        bootstrap,
        measurement_method=measurement_name,
        candidate_method=candidate_name,
    )
    return {
        "schema_version": 1,
        "metric": "mse",
        "lower_is_better": True,
        "methods": {
            "measurement": measurement_name,
            "candidate": candidate_name,
            "time_only": time_name,
        },
        "aggregation": {
            "phase_reduction": "mean_5_phases_first",
            "active_reduction": "exclude_event_0_then_mean_events_per_view",
            "recovery_reduction": "mean_events_per_view",
            "clean_reduction": "mean_all_events_per_view",
            "view_reduction": "paired_equal_mean_within_resampling_unit",
            "episode_reduction": "equal_resampling_unit_mean_within_task",
            "task_reduction": "equal_mean_of_task_effects",
        },
        "bootstrap": {
            "method": "nonparametric_percentile",
            "unit": ["task", "resampling_unit_id"],
            "view_identity": ["task", "physical_episode_id"],
            "physical_episode_id_role": "view_identity",
            "resampling_unit_id_default": "physical_episode_id",
            "stratified_by": "task",
            "shared_draws_across_methods_and_windows": True,
            "vectorized_in_bounded_chunks": True,
            "samples": samples,
            "seed": seed,
            "confidence": 0.95,
            "interval": "two_sided_percentile",
        },
        "coverage": coverage,
        "primary": primary_summary,
        "per_task": per_task,
        "recovery": recovery_diagnostic,
        "clean": clean_diagnostic,
        "diagnostics": {
            "recovery": recovery_diagnostic,
            "clean": clean_diagnostic,
        },
    }


def randomized_oracle_decision(
    summary: Mapping[str, Any],
    minimum_primary: float = 0.10,
    minimum_over_time_only: float = 0.05,
    maximum_recovery_excess: float = 0.02,
    maximum_clean_excess: float = 0.02,
) -> dict[str, Any]:
    """Apply randomized-oracle benefit and privileged-feasibility criteria."""

    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping")
    primary_minimum = _finite_threshold(minimum_primary, label="minimum_primary")
    incremental_minimum = _finite_threshold(
        minimum_over_time_only, label="minimum_over_time_only"
    )
    recovery_maximum = _finite_threshold(
        maximum_recovery_excess, label="maximum_recovery_excess"
    )
    clean_maximum = _finite_threshold(
        maximum_clean_excess, label="maximum_clean_excess"
    )
    primary = summary.get("primary")
    coverage = summary.get("coverage")
    recovery = summary.get("recovery")
    clean = summary.get("clean")
    if (
        not isinstance(primary, Mapping)
        or not isinstance(coverage, Mapping)
        or not isinstance(recovery, Mapping)
        or not isinstance(clean, Mapping)
    ):
        raise ValueError(
            "summary must contain primary, coverage, recovery, and clean mappings"
        )
    per_task = primary.get("per_task")
    if not isinstance(per_task, Mapping) or not per_task:
        raise ValueError("summary primary.per_task must be a non-empty mapping")

    candidate_effect = _finite_mse_or_effect(
        primary.get("candidate_relative_improvement"),
        label="primary candidate_relative_improvement",
    )
    candidate_ci = _effect_ci(
        primary.get("candidate_relative_improvement_95_ci"),
        label="primary candidate_relative_improvement_95_ci",
    )
    incremental_effect = _finite_mse_or_effect(
        primary.get("candidate_incremental_improvement_over_time_only"),
        label="primary candidate_incremental_improvement_over_time_only",
    )
    incremental_ci = _effect_ci(
        primary.get("candidate_incremental_improvement_over_time_only_95_ci"),
        label="primary candidate_incremental_improvement_over_time_only_95_ci",
    )
    task_effects = {
        str(task): _finite_mse_or_effect(
            row.get("candidate_relative_improvement")
            if isinstance(row, Mapping)
            else None,
            label=f"primary per_task[{task!r}] candidate_relative_improvement",
        )
        for task, row in per_task.items()
    }
    task_incremental_effects = {
        str(task): _finite_mse_or_effect(
            row.get("candidate_incremental_improvement_over_time_only")
            if isinstance(row, Mapping)
            else None,
            label=(
                f"primary per_task[{task!r}] "
                "candidate_incremental_improvement_over_time_only"
            ),
        )
        for task, row in per_task.items()
    }
    minimum_active = _strict_integer(
        coverage.get("minimum_active_events_per_episode"),
        label="coverage minimum_active_events_per_episode",
        minimum=0,
    )
    minimum_recovery = _strict_integer(
        coverage.get("minimum_recovery_events_per_episode"),
        label="coverage minimum_recovery_events_per_episode",
        minimum=0,
    )
    recovery_upper = _finite_mse_or_effect(
        recovery.get("normalized_excess_mse_one_sided_95_upper"),
        label="recovery normalized_excess_mse_one_sided_95_upper",
    )
    clean_upper = _finite_mse_or_effect(
        clean.get("normalized_excess_mse_one_sided_95_upper"),
        label="clean normalized_excess_mse_one_sided_95_upper",
    )
    checks = {
        "candidate_primary_at_least_minimum": {
            "passed": candidate_effect >= primary_minimum,
            "observed": candidate_effect,
            "operator": ">=",
            "threshold": primary_minimum,
        },
        "candidate_primary_ci_lower_positive": {
            "passed": candidate_ci[0] > 0.0,
            "observed": candidate_ci[0],
            "operator": ">",
            "threshold": 0.0,
        },
        "candidate_incremental_over_time_only_at_least_minimum": {
            "passed": incremental_effect >= incremental_minimum,
            "observed": incremental_effect,
            "operator": ">=",
            "threshold": incremental_minimum,
        },
        "candidate_incremental_over_time_only_ci_lower_positive": {
            "passed": incremental_ci[0] > 0.0,
            "observed": incremental_ci[0],
            "operator": ">",
            "threshold": 0.0,
        },
        "every_task_primary_point_effect_nonnegative": {
            "passed": all(value >= 0.0 for value in task_effects.values()),
            "observed": task_effects,
            "operator": ">=",
            "threshold": 0.0,
        },
        "every_task_incremental_over_time_only_nonnegative": {
            "passed": all(
                value >= 0.0 for value in task_incremental_effects.values()
            ),
            "observed": task_incremental_effects,
            "operator": ">=",
            "threshold": 0.0,
        },
        "recovery_feasibility_upper_below_maximum": {
            "passed": recovery_upper < recovery_maximum,
            "observed": recovery_upper,
            "operator": "<",
            "threshold": recovery_maximum,
        },
        "clean_feasibility_upper_below_maximum": {
            "passed": clean_upper < clean_maximum,
            "observed": clean_upper,
            "operator": "<",
            "threshold": clean_maximum,
        },
        "at_least_two_active_events_per_episode": {
            "passed": minimum_active >= 2,
            "observed": minimum_active,
            "operator": ">=",
            "threshold": 2,
        },
        "at_least_two_recovery_events_per_episode": {
            "passed": minimum_recovery >= 2,
            "observed": minimum_recovery,
            "operator": ">=",
            "threshold": 2,
        },
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": 1,
        "evaluated": True,
        "thresholds": {
            "minimum_primary": primary_minimum,
            "minimum_over_time_only": incremental_minimum,
            "maximum_recovery_excess": recovery_maximum,
            "maximum_clean_excess": clean_maximum,
            "ci_lower_strictly_positive": True,
            "minimum_active_events_per_episode": 2,
            "minimum_recovery_events_per_episode": 2,
        },
        "checks": checks,
        "recovery_and_clean": {
            "hard_go_checks": True,
            "feasibility_only": True,
            "non_evidentiary": True,
            "reason": (
                "These privileged checks gate recursive feasibility only; they "
                "do not establish deployable recovery detection."
            ),
        },
        "passed": passed,
        "decision": "Go" if passed else "No-Go",
        "failed_checks": [name for name, check in checks.items() if not check["passed"]],
    }


def _finite_mse_or_effect(value: Any, *, label: str) -> float:
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
    lower = _finite_mse_or_effect(value[0], label=f"{label}[0]")
    upper = _finite_mse_or_effect(value[1], label=f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} must be ordered")
    return lower, upper


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "PHASES",
    "WINDOWS",
    "randomized_oracle_decision",
    "summarize_randomized_oracle",
]
