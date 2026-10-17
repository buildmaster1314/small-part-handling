"""Unified environment and expert factories for the frozen A4 task panel."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any, Protocol, cast

import numpy as np

from fcpg import metaworld_utils
from fcpg.lewm_action_branch_a4 import (
    FORMAL_TASK_INSTANCES,
    TASKS as _A4_TASKS,
)


TASKS = _A4_TASKS
DRAWER_RETRIEVE_TASK = "drawer-retrieve-v0"
TOWEL_WIPE_TASK = "towel-wipe-v0"
CUSTOM_TASKS = (DRAWER_RETRIEVE_TASK, TOWEL_WIPE_TASK)
STANDARD_TASKS = tuple(task for task in TASKS if task not in CUSTOM_TASKS)


class ClosableEnvironment(Protocol):
    """Minimum environment interface required by A4 collectors."""

    def close(self) -> Any: ...


class ActionExpert(Protocol):
    """Minimum expert interface required by A4 collectors."""

    def get_action(self, observation: Any) -> Any: ...


def _validate_task_name(task_name: str) -> str:
    if not isinstance(task_name, str):
        raise TypeError("A4 task_name must be a string")
    if task_name not in TASKS:
        raise ValueError(
            f"Unknown A4 task {task_name!r}; expected one of the exact TASKS panel: "
            f"{TASKS}"
        )
    return task_name


def _validate_task_instance(task_instance: int) -> int:
    if isinstance(task_instance, (bool, np.bool_)):
        raise TypeError("A4 task_instance must be an integer, not bool")
    try:
        instance = operator.index(task_instance)
    except TypeError as exc:
        raise TypeError("A4 task_instance must be an integer") from exc
    if instance not in FORMAL_TASK_INSTANCES:
        raise ValueError(
            "A4 task_instance must be in the inclusive range "
            f"{FORMAL_TASK_INSTANCES[0]}..{FORMAL_TASK_INSTANCES[-1]}"
        )
    return instance


def _custom_task_backend() -> Any:
    # Keep the local MuJoCo/MetaWorld task import lazy so standard task users do
    # not load the custom environment until it is selected.
    from fcpg.custom_tasks import registry

    return registry


def _require_custom_backend(task_name: str) -> Any:
    backend = _custom_task_backend()
    registry = getattr(backend, "CUSTOM_TASK_REGISTRY", None)
    if not isinstance(registry, Mapping) or task_name not in registry:
        raise RuntimeError(
            f"Required A4 custom task {task_name!r} is not registered"
        )
    return backend


def _require_callable_method(value: Any, method: str, label: str) -> None:
    if not callable(getattr(value, method, None)):
        raise TypeError(f"A4 {label} must provide callable {method}()")


def make_a4_env(
    task_name: str,
    camera_name: str,
    width: int,
    height: int,
    *,
    task_instance: int,
    benchmark_seed: int | None = None,
) -> ClosableEnvironment:
    """Create one A4 environment with a common task-instance interface."""

    task = _validate_task_name(task_name)
    instance = _validate_task_instance(task_instance)
    if task in CUSTOM_TASKS:
        backend = _require_custom_backend(task)
        custom_kwargs: dict[str, Any] = {
            "task_instance": instance,
            "render_mode": "rgb_array",
            "camera_name": camera_name,
            "width": width,
            "height": height,
        }
        if benchmark_seed is not None:
            custom_kwargs["seed"] = benchmark_seed
        env = backend.make_custom_task(task, **custom_kwargs)
    else:
        env = metaworld_utils.make_env(
            task,
            camera_name,
            width,
            height,
            task_index=instance,
            benchmark_seed=benchmark_seed,
        )
    _require_callable_method(env, "close", "environment")
    return cast(ClosableEnvironment, env)


def make_a4_expert(task_name: str) -> ActionExpert:
    """Create the registered expert for one task in the exact A4 panel."""

    task = _validate_task_name(task_name)
    if task in CUSTOM_TASKS:
        expert = _require_custom_backend(task).make_custom_task_expert(task)
    else:
        expert = metaworld_utils.make_scripted_policy(task)
    _require_callable_method(expert, "get_action", "expert")
    return cast(ActionExpert, expert)


def make_a4_env_and_expert(
    task_name: str,
    camera_name: str,
    width: int,
    height: int,
    *,
    task_instance: int,
    benchmark_seed: int | None = None,
) -> tuple[ClosableEnvironment, ActionExpert]:
    """Create a matched A4 environment/expert pair without leaking on failure."""

    env = make_a4_env(
        task_name,
        camera_name,
        width,
        height,
        task_instance=task_instance,
        benchmark_seed=benchmark_seed,
    )
    try:
        expert = make_a4_expert(task_name)
    except Exception:
        try:
            env.close()
        except Exception:
            pass
        raise
    return env, expert


__all__ = [
    "ActionExpert",
    "ClosableEnvironment",
    "CUSTOM_TASKS",
    "DRAWER_RETRIEVE_TASK",
    "STANDARD_TASKS",
    "TASKS",
    "TOWEL_WIPE_TASK",
    "make_a4_env",
    "make_a4_env_and_expert",
    "make_a4_expert",
]
