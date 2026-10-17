from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import fcpg.lewm_a4_task_factory as factory


EXPECTED_TASKS = (
    "pick-place-v3",
    "drawer-retrieve-v0",
    "reach-v3",
    "push-v3",
    "door-open-v3",
    "window-open-v3",
    "towel-wipe-v0",
)
EXPECTED_STANDARD_TASKS = tuple(
    task
    for task in EXPECTED_TASKS
    if task not in ("drawer-retrieve-v0", "towel-wipe-v0")
)


@dataclass
class FakeEnv:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


class FakeExpert:
    def get_action(self, observation: Any) -> tuple[str, Any]:
        return "action", observation


@dataclass
class FakeCustomBackend:
    CUSTOM_TASK_REGISTRY: dict[str, object] = field(
        default_factory=lambda: {
            factory.DRAWER_RETRIEVE_TASK: object(),
            factory.TOWEL_WIPE_TASK: object(),
        }
    )
    env_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=list
    )
    expert_calls: list[str] = field(default_factory=list)

    def make_custom_task(self, *args: Any, **kwargs: Any) -> FakeEnv:
        self.env_calls.append((args, kwargs))
        return FakeEnv()

    def make_custom_task_expert(self, task_name: str) -> FakeExpert:
        self.expert_calls.append(task_name)
        return FakeExpert()


def test_factory_uses_the_exact_frozen_a4_task_order() -> None:
    assert factory.TASKS == EXPECTED_TASKS
    assert factory.STANDARD_TASKS == EXPECTED_STANDARD_TASKS


@pytest.mark.parametrize("task_name", EXPECTED_STANDARD_TASKS)
def test_standard_tasks_reuse_metaworld_factories_and_map_instance_to_task_index(
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
) -> None:
    env = FakeEnv()
    expert = FakeExpert()
    env_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    expert_calls: list[str] = []

    def make_env(*args: Any, **kwargs: Any) -> FakeEnv:
        env_calls.append((args, kwargs))
        return env

    def make_policy(task: str) -> FakeExpert:
        expert_calls.append(task)
        return expert

    monkeypatch.setattr(factory.metaworld_utils, "make_env", make_env)
    monkeypatch.setattr(
        factory.metaworld_utils, "make_scripted_policy", make_policy
    )

    made_env = factory.make_a4_env(
        task_name,
        "corner",
        224,
        192,
        task_instance=17,
        benchmark_seed=30_103,
    )
    made_expert = factory.make_a4_expert(task_name)

    assert made_env is env
    assert made_expert is expert
    assert env_calls == [
        (
            (task_name, "corner", 224, 192),
            {"task_index": 17, "benchmark_seed": 30_103},
        )
    ]
    assert expert_calls == [task_name]
    made_env.close()
    assert env.closed
    assert made_expert.get_action("observation") == ("action", "observation")


@pytest.mark.parametrize(
    "task_name", (factory.DRAWER_RETRIEVE_TASK, factory.TOWEL_WIPE_TASK)
)
def test_custom_tasks_use_registry_expert_and_preserve_task_instance(
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
) -> None:
    backend = FakeCustomBackend()
    monkeypatch.setattr(factory, "_custom_task_backend", lambda: backend)

    env, expert = factory.make_a4_env_and_expert(
        task_name,
        "corner",
        160,
        128,
        task_instance=23,
        benchmark_seed=30_103,
    )

    assert backend.env_calls == [
        (
            (task_name,),
            {
                "task_instance": 23,
                "render_mode": "rgb_array",
                "camera_name": "corner",
                "width": 160,
                "height": 128,
                "seed": 30_103,
            },
        )
    ]
    assert backend.expert_calls == [task_name]
    assert expert.get_action(4) == ("action", 4)
    env.close()
    assert env.closed


def test_drawer_without_benchmark_seed_does_not_invent_a_constructor_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeCustomBackend()
    monkeypatch.setattr(factory, "_custom_task_backend", lambda: backend)

    factory.make_a4_env(
        factory.DRAWER_RETRIEVE_TASK,
        "corner",
        32,
        32,
        task_instance=5,
    ).close()

    assert "seed" not in backend.env_calls[0][1]
    assert backend.env_calls[0][1]["task_instance"] == 5


@pytest.mark.parametrize("task_name", ("", "drawer-open-v3", "sweep-v3", None))
def test_unknown_or_missing_task_fails_before_any_backend_call(
    monkeypatch: pytest.MonkeyPatch,
    task_name: Any,
) -> None:
    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an invalid task must not reach a backend")

    monkeypatch.setattr(factory.metaworld_utils, "make_env", unexpected)
    monkeypatch.setattr(factory.metaworld_utils, "make_scripted_policy", unexpected)
    monkeypatch.setattr(factory, "_custom_task_backend", unexpected)

    with pytest.raises((TypeError, ValueError), match="A4 task|Unknown A4 task"):
        factory.make_a4_env(
            task_name,
            "corner",
            32,
            32,
            task_instance=0,
        )
    with pytest.raises((TypeError, ValueError), match="A4 task|Unknown A4 task"):
        factory.make_a4_expert(task_name)


def test_omitted_task_argument_fails_closed() -> None:
    with pytest.raises(TypeError):
        factory.make_a4_expert()  # type: ignore[call-arg]


def test_missing_required_drawer_registration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeCustomBackend(CUSTOM_TASK_REGISTRY={})
    monkeypatch.setattr(factory, "_custom_task_backend", lambda: backend)

    with pytest.raises(RuntimeError, match="not registered"):
        factory.make_a4_env(
            factory.DRAWER_RETRIEVE_TASK,
            "corner",
            32,
            32,
            task_instance=0,
        )
    with pytest.raises(RuntimeError, match="not registered"):
        factory.make_a4_expert(factory.DRAWER_RETRIEVE_TASK)
    assert backend.env_calls == []
    assert backend.expert_calls == []


@pytest.mark.parametrize(
    "task_instance", (-1, 48, True, np.bool_(True), 1.5, "1")
)
def test_invalid_task_instance_fails_before_environment_creation(
    monkeypatch: pytest.MonkeyPatch,
    task_instance: Any,
) -> None:
    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an invalid instance must not reach a backend")

    monkeypatch.setattr(factory.metaworld_utils, "make_env", unexpected)

    with pytest.raises((TypeError, ValueError), match="task_instance"):
        factory.make_a4_env(
            "reach-v3",
            "corner",
            32,
            32,
            task_instance=task_instance,
        )


def test_factories_reject_backends_without_the_common_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factory.metaworld_utils, "make_env", lambda *_args, **_kwargs: object()
    )
    with pytest.raises(TypeError, match=r"callable close\(\)"):
        factory.make_a4_env(
            "reach-v3", "corner", 32, 32, task_instance=0
        )

    monkeypatch.setattr(
        factory.metaworld_utils, "make_scripted_policy", lambda _task: object()
    )
    with pytest.raises(TypeError, match=r"callable get_action\(\)"):
        factory.make_a4_expert("reach-v3")


def test_pair_factory_closes_environment_when_expert_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = FakeEnv()
    monkeypatch.setattr(factory, "make_a4_env", lambda *_args, **_kwargs: env)

    def fail(_task_name: str) -> FakeExpert:
        raise RuntimeError("expert unavailable")

    monkeypatch.setattr(factory, "make_a4_expert", fail)

    with pytest.raises(RuntimeError, match="expert unavailable"):
        factory.make_a4_env_and_expert(
            "reach-v3", "corner", 32, 32, task_instance=0
        )
    assert env.closed
