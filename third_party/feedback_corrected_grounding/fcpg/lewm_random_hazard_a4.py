"""Random visual-interruption utilities for the seven-task A4 LeWM.

The interruption schedule is sampled only from a random seed and sequence
length. It cannot inspect task state, actions, rewards, or success. This keeps
the predictor-isolation smoke separate from the later problem of detecting a
visual failure from deployment inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np


FACTUAL_METHOD: Final = "factual_lewm"
HOLD_LAST_METHOD: Final = "hold_last"
LATENT_CV_METHOD: Final = "latent_constant_velocity"
ZERO_ACTION_METHOD: Final = "matched_no_action"
WRONG_ACTION_METHOD: Final = "wrong_action_lewm"

METHODS: Final = (
    FACTUAL_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CV_METHOD,
    ZERO_ACTION_METHOD,
    WRONG_ACTION_METHOD,
)


@dataclass(frozen=True)
class RandomHazardSchedule:
    """A model-step interruption schedule independent of episode semantics."""

    active: np.ndarray
    event_id: np.ndarray
    event_age: np.ndarray
    events: tuple[dict[str, int], ...]

    def __post_init__(self) -> None:
        active = np.asarray(self.active)
        event_id = np.asarray(self.event_id)
        event_age = np.asarray(self.event_age)
        if active.ndim != 1 or active.dtype != np.bool_:
            raise TypeError("active must be a one-dimensional boolean array")
        if event_id.shape != active.shape or event_age.shape != active.shape:
            raise ValueError("hazard schedule arrays must have matching shapes")
        if not np.issubdtype(event_id.dtype, np.integer) or not np.issubdtype(
            event_age.dtype, np.integer
        ):
            raise TypeError("event identifiers and ages must be integer arrays")
        if np.any(event_id[~active] != -1) or np.any(event_age[~active] != 0):
            raise ValueError("inactive steps must use event_id=-1 and event_age=0")
        if np.any(event_id[active] < 0) or np.any(event_age[active] < 1):
            raise ValueError("active steps require non-negative ids and positive ages")


def sample_random_hazard_schedule(
    step_count: int,
    *,
    seed: int,
    trigger_probability: float = 0.12,
    minimum_duration: int = 2,
    maximum_duration: int = 8,
    refractory_steps: int = 2,
    maximum_events: int = 3,
    warmup_steps: int = 2,
    require_event: bool = True,
) -> RandomHazardSchedule:
    """Sample random onsets and durations without reading trajectory content."""

    count = int(step_count)
    minimum = int(minimum_duration)
    maximum = int(maximum_duration)
    refractory = int(refractory_steps)
    max_events = int(maximum_events)
    warmup = int(warmup_steps)
    probability = float(trigger_probability)
    if count <= 0:
        raise ValueError("step_count must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("trigger_probability must be in [0,1]")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("hazard durations must satisfy 0 < minimum <= maximum")
    if refractory < 0 or max_events <= 0 or warmup < 0:
        raise ValueError("refractory/warmup must be non-negative and events positive")

    active = np.zeros(count, dtype=bool)
    event_id = np.full(count, -1, dtype=np.int64)
    event_age = np.zeros(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    events: list[dict[str, int]] = []

    latest_start = count - minimum
    cursor = min(warmup, count)
    while cursor <= latest_start and len(events) < max_events:
        if rng.random() >= probability:
            cursor += 1
            continue
        duration = int(rng.integers(minimum, maximum + 1))
        duration = min(duration, count - cursor)
        identifier = len(events)
        stop = cursor + duration
        active[cursor:stop] = True
        event_id[cursor:stop] = identifier
        event_age[cursor:stop] = np.arange(1, duration + 1, dtype=np.int64)
        events.append({"event_id": identifier, "start": cursor, "duration": duration})
        cursor = stop + refractory

    if require_event and not events:
        if latest_start < warmup:
            raise ValueError("sequence is too short for the requested hazard")
        start = int(rng.integers(warmup, latest_start + 1))
        duration = int(rng.integers(minimum, maximum + 1))
        duration = min(duration, count - start)
        active[start : start + duration] = True
        event_id[start : start + duration] = 0
        event_age[start : start + duration] = np.arange(
            1, duration + 1, dtype=np.int64
        )
        events.append({"event_id": 0, "start": start, "duration": duration})

    return RandomHazardSchedule(
        active=active,
        event_id=event_id,
        event_age=event_age,
        events=tuple(events),
    )


def latent_constant_velocity(history: Sequence[np.ndarray]) -> np.ndarray:
    """Propagate the latest latent with a one-step constant-velocity prior."""

    values = [np.asarray(value, dtype=np.float32) for value in history]
    if not values:
        raise ValueError("latent history must not be empty")
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("latent history entries must share a shape")
    if len(values) == 1:
        return values[-1].copy()
    result = values[-1] + (values[-1] - values[-2])
    if not np.isfinite(result).all():
        raise ValueError("constant-velocity latent prior is not finite")
    return result


def summarize_random_hazard_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize hidden-step anchor MSE by method, task, and interruption age."""

    rows = tuple(records)
    if not rows:
        raise ValueError("random-hazard records must not be empty")
    methods = tuple(sorted({str(row["method"]) for row in rows}))
    if set(methods) != set(METHODS):
        raise ValueError("random-hazard records contain an unexpected method set")
    tasks = tuple(sorted({str(row["task"]) for row in rows}))
    ages = tuple(sorted({int(row["event_age"]) for row in rows}))
    if not ages or ages[0] < 1:
        raise ValueError("random-hazard records require positive event ages")

    per_task: dict[str, dict[str, float]] = {}
    task_macro: dict[str, float] = {}
    for task in tasks:
        per_task[task] = {}
        for method in methods:
            values = [
                float(row["anchor_mse"])
                for row in rows
                if str(row["task"]) == task and str(row["method"]) == method
            ]
            if not values or not np.isfinite(values).all():
                raise ValueError(f"missing finite values for {task}/{method}")
            per_task[task][method] = float(np.mean(values))
    for method in methods:
        task_macro[method] = float(
            np.mean([per_task[task][method] for task in tasks])
        )

    comparisons: dict[str, dict[str, Any]] = {}
    factual = task_macro[FACTUAL_METHOD]
    for baseline in METHODS[1:]:
        baseline_value = task_macro[baseline]
        task_effects = {
            task: 1.0
            - per_task[task][FACTUAL_METHOD]
            / max(per_task[task][baseline], 1.0e-12)
            for task in tasks
        }
        comparisons[baseline] = {
            "relative_improvement": 1.0
            - factual / max(baseline_value, 1.0e-12),
            "positive_task_count": int(
                sum(value > 0.0 for value in task_effects.values())
            ),
            "task_count": len(tasks),
            "per_task_relative_improvement": task_effects,
        }

    by_age: dict[str, dict[str, float]] = {}
    for age in ages:
        by_age[f"age_{age}"] = {
            method: float(
                np.mean(
                    [
                        float(row["anchor_mse"])
                        for row in rows
                        if int(row["event_age"]) == age
                        and str(row["method"]) == method
                    ]
                )
            )
            for method in methods
        }

    per_task_stage: dict[str, dict[str, dict[str, float]]] = {}
    if all("task_stage" in row for row in rows):
        for task in tasks:
            stages = tuple(
                sorted(
                    {
                        str(row["task_stage"])
                        for row in rows
                        if str(row["task"]) == task
                    }
                )
            )
            per_task_stage[task] = {
                stage: {
                    method: float(
                        np.mean(
                            [
                                float(row["anchor_mse"])
                                for row in rows
                                if str(row["task"]) == task
                                and str(row["task_stage"]) == stage
                                and str(row["method"]) == method
                            ]
                        )
                    )
                    for method in methods
                }
                for stage in stages
            }

    signal = bool(
        comparisons[HOLD_LAST_METHOD]["relative_improvement"] >= 0.10
        and comparisons[LATENT_CV_METHOD]["relative_improvement"] >= 0.10
        and comparisons[ZERO_ACTION_METHOD]["relative_improvement"] >= 0.05
        and comparisons[WRONG_ACTION_METHOD]["relative_improvement"] >= 0.05
        and comparisons[HOLD_LAST_METHOD]["positive_task_count"] >= max(
            len(tasks) - 2, 1
        )
        and comparisons[ZERO_ACTION_METHOD]["positive_task_count"] >= max(
            len(tasks) - 2, 1
        )
    )
    return {
        "decision": "screen_signal" if signal else "screen_no_signal",
        "task_count": len(tasks),
        "physical_episode_count": len(
            {
                (str(row["task"]), int(row.get("task_instance", -1)))
                for row in rows
            }
        ),
        "hidden_record_count": len(rows) // len(METHODS),
        "task_macro_anchor_mse": task_macro,
        "comparisons": comparisons,
        "per_task_anchor_mse": per_task,
        "by_event_age": by_age,
        "per_task_stage_anchor_mse": per_task_stage,
        "scope": "causal formal-A4b random-hazard predictor screen",
    }


__all__ = [
    "FACTUAL_METHOD",
    "HOLD_LAST_METHOD",
    "LATENT_CV_METHOD",
    "METHODS",
    "RandomHazardSchedule",
    "WRONG_ACTION_METHOD",
    "ZERO_ACTION_METHOD",
    "latent_constant_velocity",
    "sample_random_hazard_schedule",
    "summarize_random_hazard_records",
]
