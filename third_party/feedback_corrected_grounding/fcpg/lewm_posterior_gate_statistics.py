"""Formal paired statistics for the bounded LeWM posterior Gate.

The module consumes already-scored MSE rows and has no model dependency.  A
physical episode is the resampling unit.  The primary estimand is frozen to the
direct anchor's ``post_update`` error over ``o1`` and ``o2``; recovery events are
reported only as separate diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


ESTIMANDS = ("post_update", "pre_prior")
EVENTS = ("o0", "o1", "o2", "r0", "r1")
PRIMARY_ESTIMAND = "post_update"
PRIMARY_EVENTS = ("o1", "o2")
CLEAN_ESTIMAND = "post_update"
CLEAN_EVENTS = EVENTS
SECONDARY_EVENTS = ("o0", "r0", "r1")
RECOVERY_EVENTS = ("r1",)

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 12_091
DEFAULT_ADAPTER_METHOD = "adapter"
DEFAULT_MEASUREMENT_METHOD = "measurement_writeback"
DEFAULT_FIXED_METHOD = "fixed_beta"
DEFAULT_NO_LEWM_METHOD = "no_lewm"
PROTOCOL_NAME = "fcpg_lewm_bounded_posterior_gate_statistics_v1"

Identity = int | str
EpisodeKey = tuple[str, Identity]
UnitKey = tuple[str, Identity, Identity]
RowKey = tuple[str, Identity, Identity, str, str, str]
EpisodeValues = dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class _Panel:
    values: dict[RowKey, float]
    methods: tuple[str, ...]
    units: tuple[UnitKey, ...]
    episodes_by_task: dict[str, tuple[Identity, ...]]
    anchors_by_episode: dict[EpisodeKey, tuple[Identity, ...]]
    direct_anchor_by_episode: dict[EpisodeKey, Identity]
    direct_anchor_source: str


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


def _identity(value: Any, *, label: str) -> Identity:
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


def _identity_order(value: Identity) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _episode_order(key: EpisodeKey) -> tuple[Any, ...]:
    return (key[0], _identity_order(key[1]))


def _unit_order(key: UnitKey) -> tuple[Any, ...]:
    return (key[0], _identity_order(key[1]), _identity_order(key[2]))


def _row_order(key: RowKey) -> tuple[Any, ...]:
    return (
        key[0],
        _identity_order(key[1]),
        _identity_order(key[2]),
        ESTIMANDS.index(key[3]),
        EVENTS.index(key[4]),
        key[5],
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


def _direct_marker(record: Mapping[str, Any], *, index: int) -> bool | None:
    """Read optional direct-anchor annotations without requiring extra fields."""

    markers: list[bool] = []
    for field in ("is_direct_anchor", "directly_occluded"):
        if field not in record:
            continue
        value = record[field]
        if not isinstance(value, bool):
            raise TypeError(f"record {index} {field} must be boolean")
        markers.append(value)
    if "anchor_scope" in record:
        scope = _nonempty_string(
            record["anchor_scope"], label=f"record {index} anchor_scope"
        )
        if scope in {"direct", "directly_occluded", "primary_direct"}:
            markers.append(True)
        elif scope in {"indirect", "indirectly_affected", "secondary"}:
            markers.append(False)
        else:
            raise ValueError(f"record {index} has unknown anchor_scope {scope!r}")
    if not markers:
        return None
    if len(set(markers)) != 1:
        raise ValueError(f"record {index} has conflicting direct-anchor annotations")
    return markers[0]


def _normalize_direct_mapping(
    value: Mapping[Any, Any],
) -> dict[EpisodeKey, Identity]:
    result: dict[EpisodeKey, Identity] = {}
    for raw_key, raw_anchor in value.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise TypeError(
                "direct_anchor mapping keys must be (task, physical_episode_id)"
            )
        task = _nonempty_string(raw_key[0], label="direct_anchor task")
        episode_id = _identity(
            raw_key[1], label="direct_anchor physical_episode_id"
        )
        key = (task, episode_id)
        if key in result:
            raise ValueError(f"duplicate direct_anchor mapping key: {key}")
        result[key] = _identity(raw_anchor, label=f"direct_anchor for {key}")
    return result


def _resolve_direct_anchors(
    *,
    units: Sequence[UnitKey],
    unit_markers: Mapping[UnitKey, bool] | None,
    direct_anchor: Identity | Mapping[EpisodeKey, Identity] | None,
) -> tuple[
    dict[EpisodeKey, tuple[Identity, ...]],
    dict[EpisodeKey, Identity],
    str,
]:
    anchors: dict[EpisodeKey, set[Identity]] = {}
    for task, episode_id, anchor in units:
        anchors.setdefault((task, episode_id), set()).add(anchor)
    anchors_by_episode = {
        key: tuple(sorted(values, key=_identity_order))
        for key, values in sorted(anchors.items(), key=lambda item: _episode_order(item[0]))
    }

    explicit: dict[EpisodeKey, Identity] | None = None
    if isinstance(direct_anchor, Mapping):
        explicit = _normalize_direct_mapping(direct_anchor)
        expected = set(anchors_by_episode)
        if set(explicit) != expected:
            missing = sorted(expected - set(explicit), key=_episode_order)
            unexpected = sorted(set(explicit) - expected, key=_episode_order)
            raise ValueError(
                "direct_anchor mapping must cover every physical episode exactly; "
                f"missing={missing[:3]} unexpected={unexpected[:3]}"
            )
    elif direct_anchor is not None:
        anchor = _identity(direct_anchor, label="direct_anchor")
        explicit = {key: anchor for key in anchors_by_episode}

    annotated: dict[EpisodeKey, Identity] | None = None
    if unit_markers is not None:
        annotated = {}
        for episode_key, episode_anchors in anchors_by_episode.items():
            marked = [
                anchor
                for anchor in episode_anchors
                if unit_markers[(episode_key[0], episode_key[1], anchor)]
            ]
            if len(marked) != 1:
                raise ValueError(
                    "each physical episode must mark exactly one primary direct "
                    f"anchor; episode={episode_key!r} marked={marked!r}"
                )
            annotated[episode_key] = marked[0]

    if explicit is not None and annotated is not None and explicit != annotated:
        raise ValueError("direct_anchor argument disagrees with record annotations")
    if explicit is not None:
        selected = explicit
        source = "argument"
    elif annotated is not None:
        selected = annotated
        source = "record_annotation"
    else:
        selected = {}
        for episode_key, episode_anchors in anchors_by_episode.items():
            if len(episode_anchors) != 1:
                raise ValueError(
                    "primary direct anchor is ambiguous; provide direct_anchor or "
                    "mark exactly one direct anchor per physical episode"
                )
            selected[episode_key] = episode_anchors[0]
        source = "single_anchor_inference"

    for episode_key, anchor in selected.items():
        if anchor not in anchors_by_episode[episode_key]:
            raise ValueError(
                f"direct anchor {anchor!r} is absent from episode {episode_key!r}"
            )
    return anchors_by_episode, selected, source


def _parse_panel(
    records: Sequence[Mapping[str, Any]],
    *,
    direct_anchor: Identity | Mapping[EpisodeKey, Identity] | None,
) -> _Panel:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of mappings")
    if not records:
        raise ValueError("records must not be empty")

    values: dict[RowKey, float] = {}
    marker_sets: dict[UnitKey, set[bool | None]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        task = _nonempty_string(record.get("task"), label=f"record {index} task")
        episode_id = _identity(
            record.get("physical_episode_id"),
            label=f"record {index} physical_episode_id",
        )
        anchor = _identity(record.get("anchor"), label=f"record {index} anchor")
        estimand = _nonempty_string(
            record.get("estimand"), label=f"record {index} estimand"
        )
        if estimand not in ESTIMANDS:
            raise ValueError(
                f"record {index} estimand must be one of {list(ESTIMANDS)!r}"
            )
        event = _nonempty_string(
            record.get("event"), label=f"record {index} event"
        )
        if event not in EVENTS:
            raise ValueError(f"record {index} event must be one of {list(EVENTS)!r}")
        method = _nonempty_string(
            record.get("method"), label=f"record {index} method"
        )
        mse = _finite_mse(record.get("mse"), label=f"record {index} mse")
        key = (task, episode_id, anchor, estimand, event, method)
        if key in values:
            raise ValueError(f"duplicate posterior Gate statistic record: {key}")
        values[key] = mse
        unit = (task, episode_id, anchor)
        marker_sets.setdefault(unit, set()).add(_direct_marker(record, index=index))

    has_annotations = any(
        marker is not None for markers in marker_sets.values() for marker in markers
    )
    unit_markers: dict[UnitKey, bool] | None = None
    if has_annotations:
        unit_markers = {}
        for unit, markers in marker_sets.items():
            if None in markers:
                raise ValueError(
                    "direct-anchor annotations must be present on every panel row"
                )
            concrete = {bool(marker) for marker in markers}
            if len(concrete) != 1:
                raise ValueError(
                    f"direct-anchor annotation changes within panel unit {unit!r}"
                )
            unit_markers[unit] = concrete.pop()

    methods = tuple(sorted({key[5] for key in values}))
    units = tuple(sorted({key[:3] for key in values}, key=_unit_order))
    expected = {
        (task, episode_id, anchor, estimand, event, method)
        for task, episode_id, anchor in units
        for estimand in ESTIMANDS
        for event in EVENTS
        for method in methods
    }
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual, key=_row_order)
        unexpected = sorted(actual - expected, key=_row_order)
        raise ValueError(
            "incomplete estimand/event/method phase-window panel; "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )

    anchors_by_episode, direct_by_episode, direct_source = _resolve_direct_anchors(
        units=units,
        unit_markers=unit_markers,
        direct_anchor=direct_anchor,
    )
    episodes: dict[str, set[Identity]] = {}
    for task, episode_id in anchors_by_episode:
        episodes.setdefault(task, set()).add(episode_id)
    episodes_by_task = {
        task: tuple(sorted(task_episodes, key=_identity_order))
        for task, task_episodes in sorted(episodes.items())
    }
    return _Panel(
        values=values,
        methods=methods,
        units=units,
        episodes_by_task=episodes_by_task,
        anchors_by_episode=anchors_by_episode,
        direct_anchor_by_episode=direct_by_episode,
        direct_anchor_source=direct_source,
    )


def _validate_primary(
    primary_estimand: Any, primary_events: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    estimand = _nonempty_string(primary_estimand, label="primary_estimand")
    if estimand != PRIMARY_ESTIMAND:
        raise ValueError(f"primary_estimand must be exactly {PRIMARY_ESTIMAND!r}")
    if isinstance(primary_events, (str, bytes)) or not isinstance(
        primary_events, Sequence
    ):
        raise TypeError("primary_events must be a sequence")
    events = tuple(primary_events)
    if any(event in RECOVERY_EVENTS for event in events):
        raise ValueError("r0/r1 recovery events cannot be primary")
    if events != PRIMARY_EVENTS:
        raise ValueError(f"primary_events must be exactly {PRIMARY_EVENTS!r}")
    return estimand, events


def _episode_values(
    panel: _Panel, *, estimand: str, events: Sequence[str]
) -> EpisodeValues:
    result: EpisodeValues = {method: {} for method in panel.methods}
    for task, episodes in panel.episodes_by_task.items():
        for method in panel.methods:
            task_values: list[float] = []
            for episode_id in episodes:
                anchor = panel.direct_anchor_by_episode[(task, episode_id)]
                event_values = [
                    panel.values[
                        (task, episode_id, anchor, estimand, event, method)
                    ]
                    for event in events
                ]
                task_values.append(
                    _stable_mean(
                        event_values,
                        label=(
                            f"{task}/{episode_id}/{method}/{estimand}/"
                            f"{','.join(events)} MSE"
                        ),
                    )
                )
            result[method][task] = np.asarray(task_values, dtype=np.float64)
    return result


def _indirect_episode_values(
    panel: _Panel, *, estimand: str, events: Sequence[str]
) -> EpisodeValues:
    """Average all non-direct anchor/event cells into one value per episode."""

    result: EpisodeValues = {method: {} for method in panel.methods}
    for task, episodes in panel.episodes_by_task.items():
        for method in panel.methods:
            task_values: list[float] = []
            for episode_id in episodes:
                episode_key = (task, episode_id)
                direct_anchor = panel.direct_anchor_by_episode[episode_key]
                indirect_anchors = tuple(
                    anchor
                    for anchor in panel.anchors_by_episode[episode_key]
                    if anchor != direct_anchor
                )
                if not indirect_anchors:
                    raise ValueError(
                        "each physical episode must contain at least one indirect "
                        f"anchor; episode={episode_key!r}"
                    )
                cell_values = [
                    panel.values[
                        (task, episode_id, anchor, estimand, event, method)
                    ]
                    for anchor in indirect_anchors
                    for event in events
                ]
                task_values.append(
                    _stable_mean(
                        cell_values,
                        label=(
                            f"{task}/{episode_id}/{method}/{estimand}/indirect/"
                            f"{','.join(events)} MSE"
                        ),
                    )
                )
            result[method][task] = np.asarray(task_values, dtype=np.float64)
    return result


def _episode_draws(
    episodes_by_task: Mapping[str, Sequence[Identity]],
    *,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        task: rng.integers(0, len(episodes), size=(samples, len(episodes)))
        for task, episodes in episodes_by_task.items()
    }


def _bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return np.zeros(indices.shape[0], dtype=np.float64)
    return np.mean((values / scale)[indices], axis=1, dtype=np.float64) * scale


def _finite_ratio(numerator: float, denominator: float, *, label: str) -> float:
    if denominator <= 0.0:
        raise ValueError(f"{label} denominator MSE must be positive")
    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError(f"{label} ratio must remain finite")
    return result


def _paired_effect(
    episode_values: EpisodeValues,
    *,
    candidate_method: str,
    control_method: str,
    draws: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    candidate_by_task = episode_values[candidate_method]
    control_by_task = episode_values[control_method]
    tasks = tuple(candidate_by_task)
    bootstrap_ratios = np.zeros(next(iter(draws.values())).shape[0], dtype=np.float64)
    candidate_task_means: list[float] = []
    control_task_means: list[float] = []
    task_ratios: list[float] = []
    per_task: dict[str, dict[str, Any]] = {}

    for task in tasks:
        candidate = candidate_by_task[task]
        control = control_by_task[task]
        if candidate.shape != control.shape or candidate.ndim != 1:
            raise RuntimeError(f"paired episode arrays differ for task {task!r}")
        # A zero episode-level denominator gives the empirical bootstrap
        # distribution positive mass at an undefined ratio.
        if np.any(control <= 0.0):
            raise ValueError(
                f"{control_method} episode MSE must be positive for ratio "
                f"statistics in task {task!r}"
            )
        candidate_mean = _stable_mean(
            candidate, label=f"{task}/{candidate_method} task MSE"
        )
        control_mean = _stable_mean(
            control, label=f"{task}/{control_method} task MSE"
        )
        ratio = _finite_ratio(
            candidate_mean,
            control_mean,
            label=f"{task} {candidate_method}/{control_method}",
        )
        candidate_bootstrap = _bootstrap_mean(candidate, draws[task])
        control_bootstrap = _bootstrap_mean(control, draws[task])
        if np.any(control_bootstrap <= 0.0):
            raise ValueError(
                f"bootstrap produced a non-positive {control_method} denominator "
                f"for task {task!r}"
            )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            task_bootstrap_ratios = candidate_bootstrap / control_bootstrap
        if not np.isfinite(task_bootstrap_ratios).all():
            raise ValueError(
                f"bootstrap {candidate_method}/{control_method} ratios must be finite"
            )
        bootstrap_ratios += task_bootstrap_ratios / len(tasks)
        candidate_task_means.append(candidate_mean)
        control_task_means.append(control_mean)
        task_ratios.append(ratio)
        per_task[task] = {
            "candidate_mse": candidate_mean,
            "control_mse": control_mean,
            "candidate_to_control_ratio": ratio,
            "candidate_to_control_ratio_one_sided_95_upper": float(
                np.quantile(task_bootstrap_ratios, 0.95)
            ),
            "relative_improvement": 1.0 - ratio,
            "episode_count": int(candidate.size),
        }

    if not np.isfinite(bootstrap_ratios).all():
        raise ValueError("task-equal bootstrap ratios must remain finite")
    ratio = _stable_mean(task_ratios, label="task-equal candidate/control ratio")
    improvement_samples = 1.0 - bootstrap_ratios
    improvement = 1.0 - ratio
    return {
        "candidate_method": candidate_method,
        "control_method": control_method,
        "candidate_macro_mse": _stable_mean(
            candidate_task_means, label=f"{candidate_method} task-macro MSE"
        ),
        "control_macro_mse": _stable_mean(
            control_task_means, label=f"{control_method} task-macro MSE"
        ),
        "candidate_to_control_ratio": ratio,
        "candidate_to_control_ratio_one_sided_95_upper": float(
            np.quantile(bootstrap_ratios, 0.95)
        ),
        "relative_improvement": improvement,
        "relative_improvement_95_ci": [
            float(np.quantile(improvement_samples, 0.025)),
            float(np.quantile(improvement_samples, 0.975)),
        ],
        "improvement": improvement,
        "improvement_95_ci": [
            float(np.quantile(improvement_samples, 0.025)),
            float(np.quantile(improvement_samples, 0.975)),
        ],
        "per_task": per_task,
        "task_count": len(tasks),
        "physical_episode_count": int(
            sum(values.size for values in candidate_by_task.values())
        ),
    }


def _paired_additive_effect(
    episode_values: EpisodeValues,
    *,
    candidate_method: str,
    control_method: str,
    draws: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    candidate_by_task = episode_values[candidate_method]
    control_by_task = episode_values[control_method]
    tasks = tuple(candidate_by_task)
    bootstrap_excess = np.zeros(
        next(iter(draws.values())).shape[0], dtype=np.float64
    )
    candidate_task_means: list[float] = []
    control_task_means: list[float] = []
    task_excesses: list[float] = []
    per_task: dict[str, dict[str, Any]] = {}

    for task in tasks:
        candidate = candidate_by_task[task]
        control = control_by_task[task]
        if candidate.shape != control.shape or candidate.ndim != 1:
            raise RuntimeError(f"paired episode arrays differ for task {task!r}")
        difference = candidate - control
        candidate_mean = _stable_mean(
            candidate, label=f"{task}/{candidate_method} additive MSE"
        )
        control_mean = _stable_mean(
            control, label=f"{task}/{control_method} additive MSE"
        )
        excess = _stable_mean(difference, label=f"{task} excess MSE")
        task_bootstrap = _bootstrap_mean(difference, draws[task])
        if not np.isfinite(task_bootstrap).all():
            raise ValueError("additive excess-MSE bootstrap must remain finite")
        bootstrap_excess += task_bootstrap / len(tasks)
        candidate_task_means.append(candidate_mean)
        control_task_means.append(control_mean)
        task_excesses.append(excess)
        per_task[task] = {
            "candidate_mse": candidate_mean,
            "control_mse": control_mean,
            "excess_mse": excess,
            "excess_mse_95_ci": [
                float(np.quantile(task_bootstrap, 0.025)),
                float(np.quantile(task_bootstrap, 0.975)),
            ],
            "excess_mse_one_sided_95_upper": float(
                np.quantile(task_bootstrap, 0.95)
            ),
            "episode_count": int(candidate.size),
        }

    return {
        "candidate_method": candidate_method,
        "control_method": control_method,
        "orientation": "candidate_mse_minus_control_mse",
        "candidate_macro_mse": _stable_mean(
            candidate_task_means, label=f"{candidate_method} additive macro MSE"
        ),
        "control_macro_mse": _stable_mean(
            control_task_means, label=f"{control_method} additive macro MSE"
        ),
        "excess_mse": _stable_mean(task_excesses, label="task-macro excess MSE"),
        "excess_mse_95_ci": [
            float(np.quantile(bootstrap_excess, 0.025)),
            float(np.quantile(bootstrap_excess, 0.975)),
        ],
        "excess_mse_one_sided_95_upper": float(
            np.quantile(bootstrap_excess, 0.95)
        ),
        "per_task": per_task,
        "task_count": len(tasks),
        "physical_episode_count": int(
            sum(values.size for values in candidate_by_task.values())
        ),
    }


def _normalized_excess_safety(
    safety_values: EpisodeValues,
    primary_values: EpisodeValues,
    *,
    candidate_method: str,
    control_method: str,
    scale_method: str,
    safety_draws: Mapping[str, np.ndarray],
    primary_draws: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    candidate_by_task = safety_values[candidate_method]
    control_by_task = safety_values[control_method]
    scale_by_task = primary_values[scale_method]
    tasks = tuple(candidate_by_task)
    if set(tasks) != set(control_by_task) or set(tasks) != set(scale_by_task):
        raise ValueError("safety and primary scale task sets must match")

    sample_count = next(iter(safety_draws.values())).shape[0]
    normalized_bootstrap = np.zeros(sample_count, dtype=np.float64)
    raw_bootstrap = np.zeros(sample_count, dtype=np.float64)
    candidate_task_means: list[float] = []
    control_task_means: list[float] = []
    raw_task_excesses: list[float] = []
    normalized_task_excesses: list[float] = []
    per_task: dict[str, dict[str, Any]] = {}

    for task in tasks:
        candidate = candidate_by_task[task]
        control = control_by_task[task]
        scale_values = scale_by_task[task]
        if candidate.shape != control.shape or candidate.ndim != 1:
            raise RuntimeError(f"paired safety arrays differ for task {task!r}")
        if scale_values.ndim != 1:
            raise RuntimeError(f"primary scale array is invalid for task {task!r}")
        difference = candidate - control
        primary_scale = _stable_mean(
            scale_values, label=f"{task} primary measurement scale"
        )
        if primary_scale <= 0.0:
            raise ValueError(
                f"task {task!r} primary measurement scale must be positive"
            )
        raw_excess = _stable_mean(difference, label=f"{task} safety excess MSE")
        normalized_excess = _finite_ratio(
            raw_excess,
            primary_scale,
            label=f"{task} normalized excess MSE",
        )

        raw_task_bootstrap = _bootstrap_mean(difference, safety_draws[task])
        scale_bootstrap = _bootstrap_mean(scale_values, primary_draws[task])
        if np.any(scale_bootstrap <= 0.0):
            raise ValueError(
                f"bootstrap primary measurement scale is non-positive for task {task!r}"
            )
        normalized_task_bootstrap = raw_task_bootstrap / scale_bootstrap
        if not np.isfinite(normalized_task_bootstrap).all():
            raise ValueError("normalized safety bootstrap must remain finite")
        raw_bootstrap += raw_task_bootstrap / len(tasks)
        normalized_bootstrap += normalized_task_bootstrap / len(tasks)

        candidate_mean = _stable_mean(
            candidate, label=f"{task}/{candidate_method} safety MSE"
        )
        control_mean = _stable_mean(
            control, label=f"{task}/{control_method} safety MSE"
        )
        candidate_task_means.append(candidate_mean)
        control_task_means.append(control_mean)
        raw_task_excesses.append(raw_excess)
        normalized_task_excesses.append(normalized_excess)
        per_task[task] = {
            "candidate_mse": candidate_mean,
            "control_mse": control_mean,
            "excess_mse": raw_excess,
            "primary_measurement_scale_mse": primary_scale,
            "normalized_excess_mse": normalized_excess,
            "normalized_excess_mse_95_ci": [
                float(np.quantile(normalized_task_bootstrap, 0.025)),
                float(np.quantile(normalized_task_bootstrap, 0.975)),
            ],
            "normalized_excess_mse_one_sided_95_upper": float(
                np.quantile(normalized_task_bootstrap, 0.95)
            ),
            "safety_episode_count": int(candidate.size),
            "scale_episode_count": int(scale_values.size),
        }

    return {
        "candidate_method": candidate_method,
        "control_method": control_method,
        "orientation": "candidate_mse_minus_control_mse",
        "normalization": "each_task_primary_measurement_mse",
        "candidate_macro_mse": _stable_mean(
            candidate_task_means, label=f"{candidate_method} safety macro MSE"
        ),
        "control_macro_mse": _stable_mean(
            control_task_means, label=f"{control_method} safety macro MSE"
        ),
        "excess_mse": _stable_mean(
            raw_task_excesses, label="task-macro safety excess MSE"
        ),
        "excess_mse_95_ci": [
            float(np.quantile(raw_bootstrap, 0.025)),
            float(np.quantile(raw_bootstrap, 0.975)),
        ],
        "normalized_excess_mse": _stable_mean(
            normalized_task_excesses,
            label="task-macro normalized safety excess MSE",
        ),
        "normalized_excess_mse_95_ci": [
            float(np.quantile(normalized_bootstrap, 0.025)),
            float(np.quantile(normalized_bootstrap, 0.975)),
        ],
        "normalized_excess_mse_one_sided_95_upper": float(
            np.quantile(normalized_bootstrap, 0.95)
        ),
        "per_task": per_task,
        "task_count": len(tasks),
        "safety_physical_episode_count": int(
            sum(values.size for values in candidate_by_task.values())
        ),
        "scale_physical_episode_count": int(
            sum(values.size for values in scale_by_task.values())
        ),
    }


def _task_macro_mse(episode_values: EpisodeValues) -> dict[str, float]:
    return {
        method: _stable_mean(
            [
                _stable_mean(values, label=f"{task}/{method} diagnostic MSE")
                for task, values in by_task.items()
            ],
            label=f"{method} diagnostic task-macro MSE",
        )
        for method, by_task in episode_values.items()
    }


def _diagnostic_cell(
    panel: _Panel,
    *,
    estimand: str,
    event: str,
    adapter_method: str,
    measurement_method: str,
    fixed_methods: Sequence[str],
    no_lewm_method: str,
    draws: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    episode_values = _episode_values(panel, estimand=estimand, events=(event,))
    return {
        "estimand": estimand,
        "event": event,
        "primary_eligible": False,
        "method_task_macro_mse": _task_macro_mse(episode_values),
        "effect_type": "paired_additive_excess_mse",
        "adapter_vs_measurement": _paired_additive_effect(
            episode_values,
            candidate_method=adapter_method,
            control_method=measurement_method,
            draws=draws,
        ),
        "adapter_vs_fixed": {
            method: _paired_additive_effect(
                episode_values,
                candidate_method=adapter_method,
                control_method=method,
                draws=draws,
            )
            for method in fixed_methods
        },
        "adapter_vs_no_lewm": _paired_additive_effect(
            episode_values,
            candidate_method=adapter_method,
            control_method=no_lewm_method,
            draws=draws,
        ),
    }


def _named_ratio_rows(
    effect: Mapping[str, Any], *, candidate: str, control: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task, row_value in effect["per_task"].items():
        row = dict(row_value)
        result[task] = {
            f"{candidate}_mse": row["candidate_mse"],
            f"{control}_mse": row["control_mse"],
            f"{candidate}_to_{control}_ratio": row["candidate_to_control_ratio"],
            f"{candidate}_to_{control}_ratio_one_sided_95_upper": row[
                "candidate_to_control_ratio_one_sided_95_upper"
            ],
            "relative_improvement": row["relative_improvement"],
            "physical_episode_count": row["episode_count"],
        }
    return result


def _fixed_method_names(
    fixed_methods: Sequence[str] | None, fixed_method: str | None
) -> tuple[str, ...]:
    if fixed_methods is not None and fixed_method is not None:
        raise ValueError("provide fixed_methods or fixed_method, not both")
    if fixed_methods is None:
        values: Sequence[str] = (
            DEFAULT_FIXED_METHOD if fixed_method is None else fixed_method,
        )
    else:
        values = fixed_methods
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("fixed_methods must be a sequence of method names")
    result = tuple(
        _nonempty_string(value, label=f"fixed_methods[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError("fixed_methods must contain at least one method")
    if len(result) != len(set(result)):
        raise ValueError("fixed_methods must not contain duplicates")
    return result


def summarize_posterior_gate_statistics(
    records: Sequence[Mapping[str, Any]],
    clean_records: Sequence[Mapping[str, Any]],
    *,
    adapter_method: str = DEFAULT_ADAPTER_METHOD,
    measurement_method: str = DEFAULT_MEASUREMENT_METHOD,
    fixed_methods: Sequence[str] | None = None,
    fixed_method: str | None = None,
    no_lewm_method: str = DEFAULT_NO_LEWM_METHOD,
    direct_anchor: Identity | Mapping[EpisodeKey, Identity] | None = None,
    clean_direct_anchor: Identity | Mapping[EpisodeKey, Identity] | None = None,
    primary_estimand: str = PRIMARY_ESTIMAND,
    primary_events: Sequence[str] = PRIMARY_EVENTS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate and summarize one frozen bounded-posterior Gate panel.

    The primary effect is

    ``1 - mean_task(mean_episode(adapter_mse) / mean_episode(measurement_mse))``.

    ``o1`` and ``o2`` are averaged before each physical episode enters either
    the point estimate or bootstrap.  Positive primary effects therefore favor
    the Adapter.  Recovery and clean safety use additive Adapter-minus-
    measurement excess MSE, normalized by each task's direct-primary measurement
    MSE; their measurement MSE is allowed to be exactly zero.  Indirect safety
    first averages every non-direct anchor and event within a physical episode.

    ``clean_records`` is a required, physically disjoint clean cohort.  No
    event from the occlusion panel is re-labelled as clean.
    """

    estimand, events = _validate_primary(primary_estimand, primary_events)
    samples = _strict_integer(
        bootstrap_samples, label="bootstrap_samples", minimum=1
    )
    seed = _strict_integer(bootstrap_seed, label="bootstrap_seed", minimum=0)
    adapter_name = _nonempty_string(adapter_method, label="adapter_method")
    measurement_name = _nonempty_string(
        measurement_method, label="measurement_method"
    )
    no_lewm_name = _nonempty_string(
        no_lewm_method, label="no_lewm_method"
    )
    fixed_names = _fixed_method_names(fixed_methods, fixed_method)
    all_role_methods = (
        adapter_name,
        measurement_name,
        no_lewm_name,
        *fixed_names,
    )
    if len(all_role_methods) != len(set(all_role_methods)):
        raise ValueError(
            "Adapter, measurement, fixed, and no-LeWM methods must differ"
        )
    role_methods: dict[str, Any] = {
        "adapter": adapter_name,
        "measurement": measurement_name,
        "fixed": list(fixed_names),
        "no_lewm": no_lewm_name,
    }

    panel = _parse_panel(records, direct_anchor=direct_anchor)
    clean_anchor = clean_direct_anchor
    if clean_anchor is None and not isinstance(direct_anchor, Mapping):
        clean_anchor = direct_anchor
    clean_panel = _parse_panel(clean_records, direct_anchor=clean_anchor)
    missing_methods = sorted(set(all_role_methods) - set(panel.methods))
    if missing_methods:
        raise ValueError(f"required Gate methods are absent: {missing_methods}")
    missing_clean_methods = sorted(
        set(all_role_methods) - set(clean_panel.methods)
    )
    if missing_clean_methods:
        raise ValueError(
            f"required clean-cohort methods are absent: {missing_clean_methods}"
        )
    if set(clean_panel.episodes_by_task) != set(panel.episodes_by_task):
        raise ValueError("clean cohort task set must match the primary cohort")
    primary_episode_keys = set(panel.direct_anchor_by_episode)
    clean_episode_keys = set(clean_panel.direct_anchor_by_episode)
    overlap = sorted(primary_episode_keys & clean_episode_keys, key=_episode_order)
    if overlap:
        raise ValueError(
            "clean_records must be a physically disjoint cohort; "
            f"overlap={overlap[:3]}"
        )

    draws = _episode_draws(panel.episodes_by_task, samples=samples, seed=seed)
    clean_draws = _episode_draws(
        clean_panel.episodes_by_task, samples=samples, seed=seed + 1
    )

    primary_values = _episode_values(panel, estimand=estimand, events=events)
    primary = _paired_effect(
        primary_values,
        candidate_method=adapter_name,
        control_method=measurement_name,
        draws=draws,
    )
    primary.update(
        {
            "estimand": estimand,
            "events": list(events),
            "anchor_scope": "primary_direct",
            "theta": primary["relative_improvement"],
            "theta_95_ci": primary["relative_improvement_95_ci"],
            "adapter_mse": primary["candidate_macro_mse"],
            "measurement_mse": primary["control_macro_mse"],
            "adapter_to_measurement_ratio": primary[
                "candidate_to_control_ratio"
            ],
            "adapter_to_measurement_ratio_one_sided_95_upper": primary[
                "candidate_to_control_ratio_one_sided_95_upper"
            ],
        }
    )
    comparators = {
        "adapter_vs_fixed": {
            method: _paired_effect(
                primary_values,
                candidate_method=adapter_name,
                control_method=method,
                draws=draws,
            )
            for method in fixed_names
        },
        "adapter_vs_no_lewm": _paired_effect(
            primary_values,
            candidate_method=adapter_name,
            control_method=no_lewm_name,
            draws=draws,
        ),
    }
    task_safety = _named_ratio_rows(
        primary, candidate="adapter", control="measurement"
    )

    recovery_values = _episode_values(
        panel, estimand="post_update", events=RECOVERY_EVENTS
    )
    recovery_safety = _normalized_excess_safety(
        recovery_values,
        primary_values,
        candidate_method=adapter_name,
        control_method=measurement_name,
        scale_method=measurement_name,
        safety_draws=draws,
        primary_draws=draws,
    )
    recovery_safety.update(
        {
            "estimand": "post_update",
            "events": list(RECOVERY_EVENTS),
            "cohort": "primary_recovery_windows",
        }
    )

    indirect_values = _indirect_episode_values(
        panel, estimand="post_update", events=EVENTS
    )
    indirect_safety = _normalized_excess_safety(
        indirect_values,
        primary_values,
        candidate_method=adapter_name,
        control_method=measurement_name,
        scale_method=measurement_name,
        safety_draws=draws,
        primary_draws=draws,
    )
    indirect_safety.update(
        {
            "estimand": "post_update",
            "events": list(EVENTS),
            "anchor_scope": "all_except_primary_direct",
            "cohort": "primary_indirect_anchor_windows",
        }
    )

    clean_values = _episode_values(
        clean_panel, estimand=CLEAN_ESTIMAND, events=CLEAN_EVENTS
    )
    clean_safety = _normalized_excess_safety(
        clean_values,
        primary_values,
        candidate_method=adapter_name,
        control_method=measurement_name,
        scale_method=measurement_name,
        safety_draws=clean_draws,
        primary_draws=draws,
    )
    clean_safety.update(
        {
            "estimand": CLEAN_ESTIMAND,
            "events": list(CLEAN_EVENTS),
            "cohort": "explicit_independent_clean_records",
            "interpretation": (
                "additive clean excess as a fraction of perturbed primary "
                "measurement MSE"
            ),
        }
    )

    secondary: dict[str, dict[str, Any]] = {}
    for event in SECONDARY_EVENTS:
        secondary[event] = {}
        for estimand_name in ESTIMANDS:
            cell = _diagnostic_cell(
                panel,
                estimand=estimand_name,
                event=event,
                adapter_method=adapter_name,
                measurement_method=measurement_name,
                fixed_methods=fixed_names,
                no_lewm_method=no_lewm_name,
                draws=draws,
            )
            if estimand_name == "post_update" and event in RECOVERY_EVENTS:
                event_values = _episode_values(
                    panel, estimand=estimand_name, events=(event,)
                )
                cell["adapter_vs_measurement_normalized_safety"] = (
                    _normalized_excess_safety(
                        event_values,
                        primary_values,
                        candidate_method=adapter_name,
                        control_method=measurement_name,
                        scale_method=measurement_name,
                        safety_draws=draws,
                        primary_draws=draws,
                    )
                )
            secondary[event][estimand_name] = cell

    direct_rows = [
        {
            "task": task,
            "physical_episode_id": episode_id,
            "anchor": panel.direct_anchor_by_episode[(task, episode_id)],
        }
        for task, episodes in panel.episodes_by_task.items()
        for episode_id in episodes
    ]
    clean_direct_rows = [
        {
            "task": task,
            "physical_episode_id": episode_id,
            "anchor": clean_panel.direct_anchor_by_episode[(task, episode_id)],
        }
        for task, episodes in clean_panel.episodes_by_task.items()
        for episode_id in episodes
    ]
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "metric": "mse",
        "lower_is_better": True,
        "methods": role_methods,
        "aggregation": {
            "record_key": [
                "task",
                "physical_episode_id",
                "anchor",
                "estimand",
                "event",
                "method",
            ],
            "primary_event_reduction": "mean_o1_o2_within_physical_episode",
            "indirect_safety_reduction": (
                "mean_non_direct_anchors_and_events_within_physical_episode"
            ),
            "task_reduction": "ratio_of_equal_episode_means",
            "cross_task_reduction": "equal_task_mean_of_ratios",
            "effect": (
                "1-mean_task(mean_episode(adapter_mse)/"
                "mean_episode(measurement_mse))"
            ),
        },
        "bootstrap": {
            "method": "nonparametric_percentile",
            "unit": ["task", "physical_episode_id"],
            "stratified_by": "task",
            "whole_episode_panel": True,
            "samples": samples,
            "seed": seed,
            "clean_seed": seed + 1,
            "two_sided_confidence": 0.95,
            "one_sided_ratio_confidence": 0.95,
        },
        "coverage": {
            "complete_paired_support": True,
            "complete_estimand_event_method_panels": True,
            "record_count": len(panel.values),
            "expected_record_count": (
                len(panel.units) * len(ESTIMANDS) * len(EVENTS) * len(panel.methods)
            ),
            "task_count": len(panel.episodes_by_task),
            "physical_episode_count": sum(
                len(episodes) for episodes in panel.episodes_by_task.values()
            ),
            "episode_anchor_count": len(panel.units),
            "unique_anchor_count": len({unit[2] for unit in panel.units}),
            "episodes_per_task": {
                task: len(episodes)
                for task, episodes in panel.episodes_by_task.items()
            },
            "all_methods": list(panel.methods),
            "estimands": list(ESTIMANDS),
            "events": list(EVENTS),
            "direct_anchor_source": panel.direct_anchor_source,
            "primary_direct_anchors": direct_rows,
            "clean_record_count": len(clean_panel.values),
            "clean_expected_record_count": (
                len(clean_panel.units)
                * len(ESTIMANDS)
                * len(EVENTS)
                * len(clean_panel.methods)
            ),
            "clean_physical_episode_count": sum(
                len(episodes)
                for episodes in clean_panel.episodes_by_task.values()
            ),
            "clean_complete_paired_support": True,
            "clean_physically_disjoint": True,
            "clean_direct_anchor_source": clean_panel.direct_anchor_source,
            "clean_direct_anchors": clean_direct_rows,
        },
        "primary": primary,
        "comparators": comparators,
        "task_safety_ratios": task_safety,
        "recovery_safety": recovery_safety,
        "indirect_safety": indirect_safety,
        "clean_safety": clean_safety,
        "secondary_events": secondary,
    }


summarize_bounded_gate_statistics = summarize_posterior_gate_statistics
paired_episode_stratified_bootstrap = summarize_posterior_gate_statistics


__all__ = [
    "CLEAN_ESTIMAND",
    "CLEAN_EVENTS",
    "DEFAULT_NO_LEWM_METHOD",
    "DEFAULT_ADAPTER_METHOD",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_FIXED_METHOD",
    "DEFAULT_MEASUREMENT_METHOD",
    "ESTIMANDS",
    "EVENTS",
    "PRIMARY_ESTIMAND",
    "PRIMARY_EVENTS",
    "PROTOCOL_NAME",
    "RECOVERY_EVENTS",
    "SECONDARY_EVENTS",
    "paired_episode_stratified_bootstrap",
    "summarize_bounded_gate_statistics",
    "summarize_posterior_gate_statistics",
]
