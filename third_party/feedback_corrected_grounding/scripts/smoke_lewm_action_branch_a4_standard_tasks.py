#!/usr/bin/env python3
"""Smoke trajectories for A4's five-task standard MetaWorld subpanel.

The trajectory scope is reset through a stable success signal or timeout. Local
action-branch horizons are a separate transition diagnostic and are not used as
a substitute for this reset-to-outcome rollout.

This script is not the formal A4 full panel: the remaining two tasks are
repo-local custom environments exercised through the unified A4 task factory.
Formal A4 training always uses one shared seven-task model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from numbers import Integral, Real
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROTOCOL = "lewm_action_branch_a4_standard_task_full_source_trajectory_smoke_v2"
FULL_TRAJECTORY_MODE = "full_source_trajectory"
AVAILABILITY_ONLY_MODE = "availability_only"
STANDARD_FIVE_TASK_SUBPANEL = "standard_five_task_subpanel"
DIAGNOSTIC_SUBSET = "diagnostic_subset"
STANDARD_METAWORLD_TASKS = (
    "pick-place-v3",
    "reach-v3",
    "push-v3",
    "door-open-v3",
    "window-open-v3",
)
OBJECT_REFERENCE_TASKS = frozenset(task for task in STANDARD_METAWORLD_TASKS if task != "reach-v3")
SUCCESS_STREAK_FRAMES = 3
DEFAULT_MAX_STEPS = 500
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/lewm_action_branch_a4/standard_task_full_source_trajectory_smoke.json"
)

EnvFactory = Callable[..., Any]
PolicyFactory = Callable[[str], Any]


def _validate_task_selection(
    tasks: Sequence[str], *, diagnostic_subset: bool
) -> tuple[str, ...]:
    selected = tuple(tasks)
    if not selected:
        raise ValueError("at least one task is required")
    if len(set(selected)) != len(selected):
        raise ValueError("tasks must not contain duplicates")
    unknown = [task for task in selected if task not in STANDARD_METAWORLD_TASKS]
    if unknown:
        raise ValueError(
            f"tasks must belong to the five-task MetaWorld smoke subpanel; "
            f"unknown: {unknown}"
        )

    if diagnostic_subset:
        if len(selected) >= len(STANDARD_METAWORLD_TASKS):
            raise ValueError(
                "--diagnostic-subset requires a proper subset of the five-task "
                "MetaWorld smoke subpanel"
            )
    elif selected != STANDARD_METAWORLD_TASKS:
        raise ValueError(
            "standard subpanel smoke requires the exact ordered five-task "
            "STANDARD_METAWORLD_TASKS subpanel; "
            "use --diagnostic-subset for a labeled subset"
        )
    return selected


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tasks:
        raise argparse.ArgumentTypeError("at least one task is required")
    if len(set(tasks)) != len(tasks):
        raise argparse.ArgumentTypeError("tasks must not contain duplicates")
    unknown = [task for task in tasks if task not in STANDARD_METAWORLD_TASKS]
    if unknown:
        raise argparse.ArgumentTypeError(
            "tasks must belong to the five-task MetaWorld smoke subpanel; "
            f"unknown: {unknown}"
        )
    return tasks


def _success_from_info(info: Mapping[str, Any]) -> bool:
    value = info.get("success", False)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"success must be numeric, got {value!r}") from exc
    if not np.isfinite(numeric):
        raise ValueError("success must be finite")
    return numeric > 0.0


def _reset_success(
    env: Any,
    observation: np.ndarray,
    reset_info: Mapping[str, Any],
) -> bool | None:
    if "success" in reset_info:
        return _success_from_info(reset_info)

    evaluate_state = getattr(env, "evaluate_state", None)
    action_space = getattr(env, "action_space", None)
    if not callable(evaluate_state) or action_space is None:
        return None
    low = np.asarray(action_space.low, dtype=np.float32)
    try:
        _, info = evaluate_state(observation, np.zeros_like(low))
    except (AttributeError, KeyError, NotImplementedError):
        return None
    if not isinstance(info, Mapping) or "success" not in info:
        return None
    return _success_from_info(info)


def _finite_xyz(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size < 3 or not np.isfinite(vector[:3]).all():
        return None
    return vector[:3]


def reset_distance_fields(env: Any, task: str) -> dict[str, Any]:
    """Return the task-relevant reset TCP-to-object or TCP-to-goal distance."""

    tcp = _finite_xyz(getattr(env, "tcp_center", None))
    reference_kind = "object" if task in OBJECT_REFERENCE_TASKS else "goal"
    reference = None
    if reference_kind == "object":
        get_object = getattr(env, "_get_pos_objects", None)
        if callable(get_object):
            try:
                reference = _finite_xyz(get_object())
            except (AttributeError, KeyError, RuntimeError, ValueError):
                reference = None
    else:
        reference = _finite_xyz(getattr(env, "_target_pos", None))

    distance = None
    if tcp is not None and reference is not None:
        distance = float(np.linalg.norm(tcp - reference))
    return {
        "reset_distance_reference": reference_kind if distance is not None else None,
        "reset_tcp_reference_distance_m": distance,
        "reset_tcp_object_distance_m": distance if reference_kind == "object" else None,
        "reset_tcp_goal_distance_m": distance if reference_kind == "goal" else None,
    }


def _bounded_action(env: Any, policy: Any, observation: np.ndarray) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    action = np.asarray(policy.get_action(observation), dtype=np.float32)
    if action.shape != low.shape or high.shape != low.shape:
        raise ValueError(
            f"scripted action/bounds shape mismatch: {action.shape}, {low.shape}, {high.shape}"
        )
    if not np.isfinite(action).all():
        raise ValueError("scripted policy returned a non-finite action")
    return np.clip(action, low, high).astype(np.float32, copy=False)


def run_full_source_trajectory(
    task: str,
    env: Any,
    policy: Any,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    env_seed: int = 0,
    nonzero_action_epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Run one reset-origin trajectory until three successes or a hard bound."""

    if task not in STANDARD_METAWORLD_TASKS:
        raise ValueError(f"unsupported standard task: {task}")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if nonzero_action_epsilon < 0.0 or not np.isfinite(nonzero_action_epsilon):
        raise ValueError("nonzero_action_epsilon must be finite and non-negative")

    observation, reset_info = env.reset(seed=int(env_seed))
    observation = np.asarray(observation, dtype=np.float64)
    if not isinstance(reset_info, Mapping):
        raise TypeError("environment reset info must be a mapping")
    reset_success = _reset_success(env, observation, reset_info)
    distance_fields = reset_distance_fields(env, task)

    # Reset is an initial-condition check, not a policy-produced success frame.
    current_success = False
    consecutive_successes = 0
    max_success_streak = 0
    first_success_step: int | None = None
    nonzero_action_count = 0
    steps_executed = 0
    terminated = False
    truncated = False
    stop_reason = (
        "success_streak"
        if consecutive_successes >= SUCCESS_STREAK_FRAMES
        else "max_steps"
    )

    for step in range(1, max_steps + 1):
        if consecutive_successes >= SUCCESS_STREAK_FRAMES:
            break
        action = _bounded_action(env, policy, observation)
        nonzero_action_count += int(
            bool(np.any(np.abs(action) > nonzero_action_epsilon))
        )
        observation, _, terminated, truncated, info = env.step(action)
        observation = np.asarray(observation, dtype=np.float64)
        if not isinstance(info, Mapping):
            raise TypeError("environment step info must be a mapping")

        steps_executed = step
        current_success = _success_from_info(info)
        if current_success:
            if first_success_step is None:
                first_success_step = step
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        max_success_streak = max(max_success_streak, consecutive_successes)

        if consecutive_successes >= SUCCESS_STREAK_FRAMES:
            stop_reason = "success_streak"
            break
        if terminated:
            stop_reason = "terminated"
            break
        if truncated:
            stop_reason = "truncated"
            break

    fraction = nonzero_action_count / steps_executed if steps_executed else 0.0
    return {
        "task": task,
        "mode": FULL_TRAJECTORY_MODE,
        "available": True,
        "env_seed": int(env_seed),
        "reset_success": reset_success,
        **distance_fields,
        "steps_executed": steps_executed,
        "max_steps": int(max_steps),
        "first_success_step": first_success_step,
        "final_success": bool(current_success),
        "success_streak_frames_required": SUCCESS_STREAK_FRAMES,
        "success_streak_reached": consecutive_successes >= SUCCESS_STREAK_FRAMES,
        "max_success_streak": int(max_success_streak),
        "nonzero_action_count": int(nonzero_action_count),
        "nonzero_action_fraction": float(fraction),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "stop_reason": stop_reason,
        "error_type": None,
        "error": None,
        "close_error_type": None,
        "close_error": None,
    }


def probe_task_availability(
    task: str,
    env: Any,
    policy: Any,
    *,
    env_seed: int = 0,
) -> dict[str, Any]:
    """Construct and reset a real task/policy pair without executing actions."""

    observation, reset_info = env.reset(seed=int(env_seed))
    observation = np.asarray(observation, dtype=np.float64)
    if not isinstance(reset_info, Mapping):
        raise TypeError("environment reset info must be a mapping")
    return {
        "task": task,
        "mode": AVAILABILITY_ONLY_MODE,
        "available": True,
        "env_seed": int(env_seed),
        "policy_class": type(policy).__name__,
        "observation_shape": list(observation.shape),
        "reset_success": _reset_success(env, observation, reset_info),
        **reset_distance_fields(env, task),
        "first_success_step": None,
        "final_success": None,
        "success_streak_reached": None,
        "nonzero_action_fraction": None,
        "stop_reason": None,
        "error_type": None,
        "error": None,
        "close_error_type": None,
        "close_error": None,
    }


def _error_row(
    task: str,
    *,
    availability_only: bool,
    env_seed: int,
    max_steps: int,
    error: Exception,
) -> dict[str, Any]:
    row = {
        "task": task,
        "mode": (
            AVAILABILITY_ONLY_MODE if availability_only else FULL_TRAJECTORY_MODE
        ),
        "available": False,
        "env_seed": int(env_seed),
        "error_type": type(error).__name__,
        "error": str(error),
        "reset_success": None,
        "reset_distance_reference": None,
        "reset_tcp_reference_distance_m": None,
        "reset_tcp_object_distance_m": None,
        "reset_tcp_goal_distance_m": None,
        "first_success_step": None,
        "final_success": None,
        "success_streak_reached": None,
        "nonzero_action_fraction": None,
        "stop_reason": None,
        "close_error_type": None,
        "close_error": None,
    }
    if availability_only:
        row.update(
            {
                "policy_class": None,
                "observation_shape": None,
            }
        )
    else:
        row.update(
            {
                "steps_executed": None,
                "max_steps": int(max_steps),
                "success_streak_frames_required": SUCCESS_STREAK_FRAMES,
                "max_success_streak": None,
                "nonzero_action_count": None,
                "terminated": None,
                "truncated": None,
            }
        )
    return row


def collect_task_rows(
    tasks: Sequence[str],
    *,
    camera: str,
    width: int,
    height: int,
    task_index: int,
    benchmark_seed: int,
    env_seed: int,
    max_steps: int,
    nonzero_action_epsilon: float,
    availability_only: bool,
    env_factory: EnvFactory | None = None,
    policy_factory: PolicyFactory | None = None,
) -> list[dict[str, Any]]:
    if env_factory is None or policy_factory is None:
        from fcpg.metaworld_utils import make_env, make_scripted_policy

        env_factory = env_factory or make_env
        policy_factory = policy_factory or make_scripted_policy

    rows: list[dict[str, Any]] = []
    for task in tasks:
        env = None
        row: dict[str, Any] | None = None
        try:
            env = env_factory(
                task,
                camera,
                int(width),
                int(height),
                task_index=int(task_index),
                benchmark_seed=int(benchmark_seed),
            )
            policy = policy_factory(task)
            if availability_only:
                row = probe_task_availability(
                    task,
                    env,
                    policy,
                    env_seed=int(env_seed),
                )
            else:
                row = run_full_source_trajectory(
                    task,
                    env,
                    policy,
                    max_steps=int(max_steps),
                    env_seed=int(env_seed),
                    nonzero_action_epsilon=float(nonzero_action_epsilon),
                )
        except Exception as error:  # Report every task instead of hiding later failures.
            row = _error_row(
                task,
                availability_only=availability_only,
                env_seed=int(env_seed),
                max_steps=int(max_steps),
                error=error,
            )
        finally:
            try:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            except Exception as close_error:
                if row is not None:
                    row["close_error_type"] = type(close_error).__name__
                    row["close_error"] = str(close_error)
        if row is None:
            raise RuntimeError(f"task {task} completed without a result row")
        rows.append(row)
    return rows


_COMMON_ROW_FIELDS = (
    "task",
    "mode",
    "available",
    "env_seed",
    "reset_success",
    "reset_distance_reference",
    "reset_tcp_reference_distance_m",
    "reset_tcp_object_distance_m",
    "reset_tcp_goal_distance_m",
    "first_success_step",
    "final_success",
    "success_streak_reached",
    "nonzero_action_fraction",
    "stop_reason",
    "error_type",
    "error",
    "close_error_type",
    "close_error",
)
_FULL_TRAJECTORY_ROW_FIELDS = (
    "steps_executed",
    "max_steps",
    "success_streak_frames_required",
    "max_success_streak",
    "nonzero_action_count",
    "terminated",
    "truncated",
)
_AVAILABILITY_ROW_FIELDS = ("policy_class", "observation_shape")


def _optional_fraction(row: Mapping[str, Any], key: str) -> float | None:
    value = row[key]
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{key} must be numeric or null")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{key} must be a finite fraction")
    return numeric


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be boolean")
    return bool(value)


def _optional_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    return _strict_bool(value, label)


def _optional_nonnegative_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric or null")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return numeric


def _optional_nonnegative_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer or null")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{label} must be non-negative")
    return numeric


def _validate_error_pair(row: Mapping[str, Any], prefix: str, label: str) -> bool:
    error_type = row[f"{prefix}error_type"]
    error = row[f"{prefix}error"]
    if (error_type is None) != (error is None):
        raise ValueError(f"{label} error type and message must both be set or null")
    if error_type is None:
        return False
    if not isinstance(error_type, str) or not error_type:
        raise TypeError(f"{label} error type must be a non-empty string")
    if not isinstance(error, str):
        raise TypeError(f"{label} error message must be a string")
    return True


def _validate_row(row: Mapping[str, Any], index: int) -> tuple[str, str, bool]:
    missing_common = [key for key in _COMMON_ROW_FIELDS if key not in row]
    if missing_common:
        raise ValueError(
            f"row {index} is missing required fields: {', '.join(missing_common)}"
        )

    task = row["task"]
    if not isinstance(task, str) or not task:
        raise TypeError(f"row {index} task must be a non-empty string")
    if task not in STANDARD_METAWORLD_TASKS:
        raise ValueError(f"row {index} has unsupported task {task!r}")

    mode = row["mode"]
    if mode not in (FULL_TRAJECTORY_MODE, AVAILABILITY_ONLY_MODE):
        raise ValueError(f"row {index} has unsupported mode {mode!r}")
    mode_fields = (
        _FULL_TRAJECTORY_ROW_FIELDS
        if mode == FULL_TRAJECTORY_MODE
        else _AVAILABILITY_ROW_FIELDS
    )
    missing_mode = [key for key in mode_fields if key not in row]
    if missing_mode:
        raise ValueError(
            f"row {index} is missing required {mode} fields: "
            + ", ".join(missing_mode)
        )

    available = _strict_bool(row["available"], f"row {index} available")
    _optional_nonnegative_integer(row["env_seed"], f"row {index} env_seed")
    reset_success = _optional_bool(
        row["reset_success"], f"row {index} reset_success"
    )
    primary_error = _validate_error_pair(row, "", f"row {index} primary")
    close_error = _validate_error_pair(row, "close_", f"row {index} close")
    if available == primary_error:
        raise ValueError(
            f"row {index} available must be false exactly when a primary error exists"
        )

    reference = row["reset_distance_reference"]
    if reference not in (None, "object", "goal"):
        raise ValueError(f"row {index} has invalid reset distance reference")
    reference_distance = _optional_nonnegative_number(
        row["reset_tcp_reference_distance_m"],
        f"row {index} reset_tcp_reference_distance_m",
    )
    object_distance = _optional_nonnegative_number(
        row["reset_tcp_object_distance_m"],
        f"row {index} reset_tcp_object_distance_m",
    )
    goal_distance = _optional_nonnegative_number(
        row["reset_tcp_goal_distance_m"],
        f"row {index} reset_tcp_goal_distance_m",
    )
    expected_reference = "goal" if task == "reach-v3" else "object"
    if reference is not None and reference != expected_reference:
        raise ValueError(f"row {index} uses the wrong reset reference for {task}")
    if reference is None:
        if any(
            value is not None
            for value in (reference_distance, object_distance, goal_distance)
        ):
            raise ValueError(f"row {index} has a distance without a reference")
    else:
        selected_distance = object_distance if reference == "object" else goal_distance
        other_distance = goal_distance if reference == "object" else object_distance
        if (
            reference_distance is None
            or selected_distance is None
            or other_distance is not None
        ):
            raise ValueError(f"row {index} reset distance fields are inconsistent")
        if not np.isclose(reference_distance, selected_distance, rtol=0.0, atol=1e-12):
            raise ValueError(f"row {index} reset reference distances disagree")

    first_success_step = _optional_nonnegative_integer(
        row["first_success_step"], f"row {index} first_success_step"
    )
    final_success = _optional_bool(
        row["final_success"], f"row {index} final_success"
    )
    streak_reached = _optional_bool(
        row["success_streak_reached"],
        f"row {index} success_streak_reached",
    )
    fraction = _optional_fraction(row, "nonzero_action_fraction")
    stop_reason = row["stop_reason"]

    if mode == AVAILABILITY_ONLY_MODE:
        if any(
            value is not None
            for value in (
                first_success_step,
                final_success,
                streak_reached,
                fraction,
                stop_reason,
            )
        ):
            raise ValueError(f"row {index} availability-only outcomes must be null")
        policy_class = row["policy_class"]
        observation_shape = row["observation_shape"]
        if available:
            if not isinstance(policy_class, str) or not policy_class:
                raise TypeError(
                    f"row {index} policy_class must be a non-empty string"
                )
            if (
                not isinstance(observation_shape, list)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, Integral)
                    or int(value) < 0
                    for value in observation_shape
                )
            ):
                raise TypeError(
                    f"row {index} observation_shape must be non-negative integers"
                )
    elif available:
        steps = _optional_nonnegative_integer(
            row["steps_executed"], f"row {index} steps_executed"
        )
        max_steps = _optional_nonnegative_integer(
            row["max_steps"], f"row {index} max_steps"
        )
        required_streak = _optional_nonnegative_integer(
            row["success_streak_frames_required"],
            f"row {index} success_streak_frames_required",
        )
        max_streak = _optional_nonnegative_integer(
            row["max_success_streak"], f"row {index} max_success_streak"
        )
        nonzero_count = _optional_nonnegative_integer(
            row["nonzero_action_count"], f"row {index} nonzero_action_count"
        )
        terminated = _optional_bool(row["terminated"], f"row {index} terminated")
        truncated = _optional_bool(row["truncated"], f"row {index} truncated")
        if any(
            value is None
            for value in (
                steps,
                max_steps,
                required_streak,
                max_streak,
                nonzero_count,
                final_success,
                streak_reached,
                fraction,
                terminated,
                truncated,
            )
        ):
            raise ValueError(f"row {index} available trajectory fields cannot be null")
        assert steps is not None
        assert max_steps is not None
        assert required_streak is not None
        assert max_streak is not None
        assert nonzero_count is not None
        if max_steps <= 0 or steps > max_steps:
            raise ValueError(f"row {index} has invalid trajectory step bounds")
        if required_streak != SUCCESS_STREAK_FRAMES:
            raise ValueError(f"row {index} has the wrong required success streak")
        if nonzero_count > steps or max_streak > steps:
            raise ValueError(f"row {index} trajectory counts exceed executed steps")
        if first_success_step is not None and not 1 <= first_success_step <= steps:
            raise ValueError(f"row {index} first success must be a stepped frame")
        if not isinstance(stop_reason, str) or stop_reason not in {
            "success_streak",
            "max_steps",
            "terminated",
            "truncated",
        }:
            raise ValueError(f"row {index} has an invalid stop reason")
        if streak_reached and (
            not final_success
            or first_success_step is None
            or max_streak < SUCCESS_STREAK_FRAMES
            or stop_reason != "success_streak"
        ):
            raise ValueError(f"row {index} has inconsistent success-streak evidence")
        if final_success and first_success_step is None:
            raise ValueError(f"row {index} final success lacks a first-success step")

    passed = (
        available
        and not primary_error
        and not close_error
        and reset_success is False
        and (
            mode == AVAILABILITY_ONLY_MODE
            or (streak_reached is True and final_success is True)
        )
    )
    return task, mode, bool(passed)


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_tasks: Sequence[str] | None = None,
    expected_mode: str | None = None,
) -> dict[str, Any]:
    """Pure aggregation for mock tests and the command-line report."""

    if not rows:
        raise ValueError("cannot summarize an empty task subpanel")
    validated = [_validate_row(row, index) for index, row in enumerate(rows)]
    tasks = [task for task, _, _ in validated]
    modes = [mode for _, mode, _ in validated]
    task_passes = [passed for _, _, passed in validated]
    if len(set(tasks)) != len(tasks):
        raise ValueError("rows must have unique task names")
    requested_tasks = list(STANDARD_METAWORLD_TASKS if expected_tasks is None else expected_tasks)
    if tasks != requested_tasks:
        raise ValueError("task rows do not match the requested ordered task subpanel")
    if len(set(modes)) != 1:
        raise ValueError("task rows must use one consistent mode")
    mode = modes[0]
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(
            f"task row mode {mode!r} does not match expected mode {expected_mode!r}"
        )

    available = [row["available"] is True for row in rows]
    fractions = [
        value
        for row in rows
        if (value := _optional_fraction(row, "nonzero_action_fraction")) is not None
    ]
    distance_available = [
        row["reset_tcp_object_distance_m"] is not None
        or row["reset_tcp_goal_distance_m"] is not None
        for row in rows
    ]
    per_task_keys = (
        "mode",
        "available",
        "reset_success",
        "reset_distance_reference",
        "reset_tcp_reference_distance_m",
        "reset_tcp_object_distance_m",
        "reset_tcp_goal_distance_m",
        "first_success_step",
        "final_success",
        "success_streak_reached",
        "nonzero_action_fraction",
        "stop_reason",
        "error_type",
        "error",
        "close_error_type",
        "close_error",
    )
    per_task = {
        task: {
            **{key: row[key] for key in per_task_keys},
            "passed": passed,
        }
        for task, row, passed in zip(tasks, rows, task_passes, strict=True)
    }
    is_complete_standard_task_subpanel = tasks == list(STANDARD_METAWORLD_TASKS)
    all_tasks_passed = bool(all(task_passes))
    return {
        "mode": mode,
        "tasks": tasks,
        "task_count": len(rows),
        "is_complete_standard_task_subpanel": is_complete_standard_task_subpanel,
        "available_task_count": int(sum(available)),
        "all_tasks_available": bool(all(available)),
        "reset_success_count": int(sum(row["reset_success"] is True for row in rows)),
        "valid_reset_task_count": int(
            sum(row["reset_success"] is False for row in rows)
        ),
        "reset_distance_available_count": int(sum(distance_available)),
        "first_success_task_count": int(
            sum(row["first_success_step"] is not None for row in rows)
        ),
        "final_success_count": int(sum(row["final_success"] is True for row in rows)),
        "success_streak_reached_count": int(
            sum(row["success_streak_reached"] is True for row in rows)
        ),
        "primary_error_count": int(sum(row["error_type"] is not None for row in rows)),
        "close_error_count": int(
            sum(row["close_error_type"] is not None for row in rows)
        ),
        "passed_task_count": int(sum(task_passes)),
        "all_tasks_passed": all_tasks_passed,
        "standard_subpanel_pass": bool(
            mode == FULL_TRAJECTORY_MODE
            and is_complete_standard_task_subpanel
            and all_tasks_passed
        ),
        "mean_nonzero_action_fraction": (
            float(np.mean(fractions)) if fractions else None
        ),
        "per_task": per_task,
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    availability_only: bool,
    max_steps: int,
    nonzero_action_epsilon: float,
    diagnostic_subset: bool = False,
) -> dict[str, Any]:
    selected_tasks = _validate_task_selection(
        tasks, diagnostic_subset=bool(diagnostic_subset)
    )
    mode = AVAILABILITY_ONLY_MODE if availability_only else FULL_TRAJECTORY_MODE
    summary = summarize_rows(
        rows,
        expected_tasks=selected_tasks,
        expected_mode=mode,
    )
    panel_kind = (
        DIAGNOSTIC_SUBSET if diagnostic_subset else STANDARD_FIVE_TASK_SUBPANEL
    )
    standard_subpanel_pass = bool(
        not availability_only
        and not diagnostic_subset
        and summary["standard_subpanel_pass"]
    )
    diagnostic_subset_pass = (
        bool(summary["all_tasks_passed"])
        if diagnostic_subset and not availability_only
        else None
    )
    availability_pass = (
        bool(summary["all_tasks_passed"]) if availability_only else None
    )
    run_pass = (
        availability_pass
        if availability_only
        else diagnostic_subset_pass
        if diagnostic_subset
        else standard_subpanel_pass
    )
    return {
        "protocol": PROTOCOL,
        "mode": mode,
        "panel_kind": panel_kind,
        "is_complete_standard_task_subpanel_smoke": bool(
            not availability_only and not diagnostic_subset
        ),
        "scope": (
            "environment_policy_reset_availability"
            if availability_only
            else "reset_to_three_success_frames_or_timeout"
        ),
        "reset_frame_step": 0,
        "reset_frame_counts_toward_success_streak": False,
        "required_reset_success": False,
        "success_streak_frames_required": SUCCESS_STREAK_FRAMES,
        "max_steps": None if availability_only else int(max_steps),
        "nonzero_action_epsilon": (
            None if availability_only else float(nonzero_action_epsilon)
        ),
        "standard_subpanel_pass": standard_subpanel_pass,
        "diagnostic_subset_pass": diagnostic_subset_pass,
        "availability_pass": availability_pass,
        "run_pass": bool(run_pass),
        "summary": summary,
        "rows": [dict(row) for row in rows],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=parse_tasks, default=STANDARD_METAWORLD_TASKS)
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--benchmark-seed", type=int, default=31404)
    parser.add_argument("--env-seed", type=int, default=31404)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--nonzero-action-epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--diagnostic-subset",
        action="store_true",
        help=(
            "label --tasks as a diagnostic subset of the five standard tasks; "
            "subset results never count as a standard-subpanel pass"
        ),
    )
    parser.add_argument(
        "--availability-only",
        action="store_true",
        help="construct and reset each environment/policy pair without stepping",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if not 0 <= args.task_index < 50:
        raise ValueError("task_index must be in [0,49]")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.nonzero_action_epsilon < 0.0 or not np.isfinite(
        args.nonzero_action_epsilon
    ):
        raise ValueError("nonzero_action_epsilon must be finite and non-negative")

    tasks = _validate_task_selection(
        args.tasks, diagnostic_subset=bool(args.diagnostic_subset)
    )
    rows = collect_task_rows(
        tasks,
        camera=args.camera,
        width=args.width,
        height=args.height,
        task_index=args.task_index,
        benchmark_seed=args.benchmark_seed,
        env_seed=args.env_seed,
        max_steps=args.max_steps,
        nonzero_action_epsilon=args.nonzero_action_epsilon,
        availability_only=bool(args.availability_only),
    )
    report = build_report(
        rows,
        tasks=tasks,
        availability_only=bool(args.availability_only),
        max_steps=args.max_steps,
        nonzero_action_epsilon=args.nonzero_action_epsilon,
        diagnostic_subset=bool(args.diagnostic_subset),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)

    if not report["run_pass"]:
        return 1 if args.availability_only else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
