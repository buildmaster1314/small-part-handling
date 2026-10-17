"""Side-effect-free registry for repo-local custom tasks."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from fcpg.custom_tasks.drawer_retrieve import (
    DRAWER_RETRIEVE_TASK_NAME,
    SawyerDrawerRetrieveEnvV0,
)
from fcpg.custom_tasks.drawer_retrieve_expert import (
    DrawerRetrievePhysicalStateExpert,
)
from fcpg.custom_tasks.towel_wipe import (
    TOWEL_WIPE_TASK_NAME,
    SawyerTowelWipeEnvV0,
)
from fcpg.custom_tasks.towel_wipe_expert import TowelWipePhysicalStateExpert


CUSTOM_TASK_REGISTRY = MappingProxyType(
    {
        DRAWER_RETRIEVE_TASK_NAME: SawyerDrawerRetrieveEnvV0,
        TOWEL_WIPE_TASK_NAME: SawyerTowelWipeEnvV0,
    }
)
TASK_REGISTRY = CUSTOM_TASK_REGISTRY


def available_custom_tasks() -> tuple[str, ...]:
    return tuple(CUSTOM_TASK_REGISTRY)


def get_custom_task_class(task_name: str) -> type[Any]:
    try:
        return CUSTOM_TASK_REGISTRY[task_name]
    except KeyError as exc:
        available = ", ".join(available_custom_tasks())
        raise KeyError(f"Unknown custom task {task_name!r}; available: {available}") from exc


def make_custom_task(
    task_name: str,
    *,
    task_instance: int = 0,
    **kwargs: Any,
) -> Any:
    return get_custom_task_class(task_name)(task_instance=task_instance, **kwargs)


def make_drawer_retrieve_env(
    *, task_instance: int = 0, **kwargs: Any
) -> SawyerDrawerRetrieveEnvV0:
    return make_custom_task(
        DRAWER_RETRIEVE_TASK_NAME, task_instance=task_instance, **kwargs
    )


def make_towel_wipe_env(*, task_instance: int = 0, **kwargs: Any) -> SawyerTowelWipeEnvV0:
    return make_custom_task(TOWEL_WIPE_TASK_NAME, task_instance=task_instance, **kwargs)


def make_custom_task_expert(
    task_name: str,
) -> DrawerRetrievePhysicalStateExpert | TowelWipePhysicalStateExpert:
    get_custom_task_class(task_name)
    if task_name == DRAWER_RETRIEVE_TASK_NAME:
        return DrawerRetrievePhysicalStateExpert()
    if task_name == TOWEL_WIPE_TASK_NAME:
        return TowelWipePhysicalStateExpert()
    raise AssertionError(f"unhandled registered custom task {task_name!r}")


make = make_custom_task
