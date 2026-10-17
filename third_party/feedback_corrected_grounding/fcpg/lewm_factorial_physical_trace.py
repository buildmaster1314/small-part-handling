"""Physical-state traces for interpreting the matched FCPG x LeWM panel."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from typing import Any

import numpy as np


PICK_SUCCESS_RADIUS_M = 0.07
PICK_TERMINAL_RADIUS_M = 0.05
RAW_ACTION_STEPS = 5


def _vector(value: object, *, size: int = 3) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < size or not bool(np.isfinite(array[:size]).all()):
        return None
    return array[:size].copy()


def _json_vector(value: np.ndarray | None) -> list[float] | None:
    return None if value is None else [float(item) for item in value]


def _numeric_info(info: object) -> dict[str, float]:
    if not isinstance(info, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in info.items():
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            scalar = float(value)
            if np.isfinite(scalar):
                result[str(key)] = scalar
    return result


class PhysicalTraceEnv:
    """Transparent environment proxy that records post-step task geometry."""

    def __init__(
        self,
        env: object,
        sink: MutableSequence[dict[str, object]],
        *,
        drop_start: int,
        drop_duration: int,
    ) -> None:
        self._env = env
        self._sink = sink
        self._drop_start = int(drop_start)
        self._drop_duration = int(drop_duration)
        self._raw_step = 0
        self._previous_object: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _object_position(self) -> np.ndarray | None:
        getter = getattr(self._env, "_get_pos_objects", None)
        if not callable(getter):
            return None
        return _vector(getter())

    def step(self, action: np.ndarray):
        result = self._env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("MetaWorld step must return a five-item tuple")
        observation, reward, terminated, truncated, info = result
        self._raw_step += 1
        decision = (self._raw_step - 1) // RAW_ACTION_STEPS
        if decision < self._drop_start:
            phase = "prefix"
        elif decision < self._drop_start + self._drop_duration:
            phase = "dropout"
        else:
            phase = "recovery"

        object_position = self._object_position()
        target_position = _vector(getattr(self._env, "_target_pos", None))
        tcp_position = _vector(getattr(self._env, "tcp_center", None))
        object_to_target = (
            None
            if object_position is None or target_position is None
            else float(np.linalg.norm(object_position - target_position))
        )
        tcp_to_object = (
            None
            if object_position is None or tcp_position is None
            else float(np.linalg.norm(tcp_position - object_position))
        )
        object_step_speed = (
            None
            if object_position is None or self._previous_object is None
            else float(np.linalg.norm(object_position - self._previous_object))
        )
        if object_position is not None:
            self._previous_object = object_position.copy()

        numeric_info = _numeric_info(info)
        if object_to_target is None and "obj_to_target" in numeric_info:
            object_to_target = numeric_info["obj_to_target"]
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size != 4 or not bool(np.isfinite(action_array).all()):
            raise ValueError("recorded MetaWorld action must be one finite 4-vector")
        self._sink.append(
            {
                "raw_step": self._raw_step,
                "decision": decision,
                "decision_offset": (self._raw_step - 1) % RAW_ACTION_STEPS,
                "phase": phase,
                "action_xyz": [float(value) for value in action_array[:3]],
                "gripper_action": float(action_array[3]),
                "reward": float(reward),
                "success": float(numeric_info.get("success", 0.0)),
                "object_position": _json_vector(object_position),
                "target_position": _json_vector(target_position),
                "tcp_position": _json_vector(tcp_position),
                "object_to_target": object_to_target,
                "tcp_to_object": tcp_to_object,
                "object_step_speed": object_step_speed,
                "within_terminal_5cm": bool(
                    object_to_target is not None
                    and object_to_target <= PICK_TERMINAL_RADIUS_M
                ),
                "within_success_7cm": bool(
                    object_to_target is not None
                    and object_to_target <= PICK_SUCCESS_RADIUS_M
                ),
                "info": numeric_info,
            }
        )
        return observation, reward, terminated, truncated, info


def _maximum_true_streak(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def summarize_physical_trace(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize whether a trajectory reaches, retains, or escapes the goal."""

    if not rows:
        raise ValueError("physical trace is empty")
    expected_steps = list(range(1, len(rows) + 1))
    if [int(row["raw_step"]) for row in rows] != expected_steps:
        raise ValueError("physical trace raw steps must be consecutive and one-indexed")

    distances = [
        None if row["object_to_target"] is None else float(row["object_to_target"])
        for row in rows
    ]
    finite_distances = [value for value in distances if value is not None]
    success = [bool(float(row["success"]) > 0.0) for row in rows]
    core = [bool(row["within_terminal_5cm"]) for row in rows]
    shell = [
        bool(row["within_success_7cm"]) and not bool(row["within_terminal_5cm"])
        for row in rows
    ]
    successful_steps = [index + 1 for index, value in enumerate(success) if value]

    if not finite_distances:
        mechanism = "distance_unavailable"
    else:
        minimum = min(finite_distances)
        final = finite_distances[-1]
        if minimum > PICK_SUCCESS_RADIUS_M:
            mechanism = "never_entered_success_region"
        elif final <= PICK_TERMINAL_RADIUS_M:
            mechanism = "retained_terminal_core"
        elif final <= PICK_SUCCESS_RADIUS_M:
            mechanism = "retained_success_shell"
        elif minimum <= PICK_TERMINAL_RADIUS_M:
            mechanism = "escaped_terminal_core"
        else:
            mechanism = "escaped_success_shell"

    object_rows = [
        (index, _vector(row["object_position"]))
        for index, row in enumerate(rows)
        if row["object_position"] is not None
    ]
    recovery_indices = [
        index for index, row in enumerate(rows) if row["phase"] == "recovery"
    ]
    recovery_displacement = None
    if object_rows and recovery_indices:
        recovery_index = recovery_indices[0]
        recovery_object = next(
            (value for index, value in object_rows if index >= recovery_index), None
        )
        final_object = object_rows[-1][1]
        if recovery_object is not None and final_object is not None:
            recovery_displacement = float(np.linalg.norm(final_object - recovery_object))

    return {
        "raw_steps": len(rows),
        "success_steps": len(successful_steps),
        "first_success_step": successful_steps[0] if successful_steps else None,
        "last_success_step": successful_steps[-1] if successful_steps else None,
        "maximum_success_streak": _maximum_true_streak(success),
        "terminal_core_steps": int(sum(core)),
        "success_shell_steps": int(sum(shell)),
        "minimum_object_to_target_m": (
            None if not finite_distances else float(min(finite_distances))
        ),
        "final_object_to_target_m": (
            None if not finite_distances else float(finite_distances[-1])
        ),
        "post_recovery_object_displacement_m": recovery_displacement,
        "retention_mechanism": mechanism,
    }


__all__ = [
    "PICK_SUCCESS_RADIUS_M",
    "PICK_TERMINAL_RADIUS_M",
    "PhysicalTraceEnv",
    "summarize_physical_trace",
]
