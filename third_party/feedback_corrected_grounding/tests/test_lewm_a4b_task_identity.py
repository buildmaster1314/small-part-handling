from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.custom_tasks.drawer_retrieve import SawyerDrawerRetrieveEnvV0
from fcpg.custom_tasks.towel_wipe import SawyerTowelWipeEnvV0
from fcpg.lewm_a4_task_factory import DRAWER_RETRIEVE_TASK, TOWEL_WIPE_TASK
from fcpg.lewm_a4b_task_identity import (
    A4bDrawerRetrieveEnvV1,
    A4bTowelWipeEnvV1,
    capture_reset_physical_identity,
    make_a4b_env,
    reset_physical_state_goal_digest,
)
from fcpg.lewm_action_branch_a4b import BENCHMARK_SEED


def _fake_env(
    *, qpos: np.ndarray, goal: np.ndarray, task_data: np.ndarray
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            qpos=np.asarray(qpos, dtype=np.float64),
            qvel=np.zeros_like(qpos, dtype=np.float64),
        ),
        _target_pos=np.asarray(goal, dtype=np.float64),
        _last_rand_vec=np.asarray(task_data, dtype=np.float64),
    )


@pytest.mark.parametrize("raw_task_index", [0, 1, 23, 47])
def test_drawer_a4b_continuous_parameters_change_physical_digest(
    raw_task_index: int,
) -> None:
    old_block, old_goal = SawyerDrawerRetrieveEnvV0.task_instance_reset_parameters(
        raw_task_index
    )
    fresh_block, fresh_goal = A4bDrawerRetrieveEnvV1.task_instance_reset_parameters(
        raw_task_index
    )
    repeated = A4bDrawerRetrieveEnvV1.task_instance_reset_parameters(raw_task_index)

    assert np.array_equal(fresh_block, repeated[0])
    assert np.array_equal(fresh_goal, repeated[1])
    assert not np.array_equal(old_block, fresh_block)
    assert not np.array_equal(old_goal, fresh_goal)
    old_env = _fake_env(
        qpos=old_block,
        goal=old_goal,
        task_data=np.concatenate((old_block, old_goal)),
    )
    fresh_env = _fake_env(
        qpos=fresh_block,
        goal=fresh_goal,
        task_data=np.concatenate((fresh_block, fresh_goal)),
    )
    assert reset_physical_state_goal_digest(old_env) != (
        reset_physical_state_goal_digest(fresh_env)
    )


@pytest.mark.parametrize("raw_task_index", [0, 1, 23, 47])
def test_towel_a4b_continuous_parameters_change_physical_digest(
    raw_task_index: int,
) -> None:
    old_left, old_right = SawyerTowelWipeEnvV0.task_instance_endpoints(
        raw_task_index
    )
    fresh_left, fresh_right = A4bTowelWipeEnvV1.task_instance_endpoints(
        raw_task_index
    )
    repeated = A4bTowelWipeEnvV1.task_instance_endpoints(raw_task_index)

    assert np.array_equal(fresh_left, repeated[0])
    assert np.array_equal(fresh_right, repeated[1])
    assert not np.array_equal(old_left, fresh_left)
    assert not np.array_equal(old_right, fresh_right)
    old_env = _fake_env(
        qpos=old_left,
        goal=old_right,
        task_data=np.concatenate((old_left, old_right)),
    )
    fresh_env = _fake_env(
        qpos=fresh_left,
        goal=fresh_right,
        task_data=np.concatenate((fresh_left, fresh_right)),
    )
    assert reset_physical_state_goal_digest(old_env) != (
        reset_physical_state_goal_digest(fresh_env)
    )


def test_reset_digest_cannot_be_changed_by_renaming_identity_metadata() -> None:
    env = _fake_env(
        qpos=np.asarray([1.0, 2.0]),
        goal=np.asarray([0.2, 0.3, 0.4]),
        task_data=np.asarray([4.0, 5.0, 6.0]),
    )
    first = capture_reset_physical_identity(
        env,
        task_name=DRAWER_RETRIEVE_TASK,
        raw_task_index=0,
        benchmark_seed=30_103,
        environment_seed=620_000,
        camera_name="corner",
    )
    renamed = capture_reset_physical_identity(
        env,
        task_name=TOWEL_WIPE_TASK,
        raw_task_index=47,
        benchmark_seed=BENCHMARK_SEED,
        environment_seed=720_000,
        camera_name="different-camera",
    )

    assert first.digest == renamed.digest
    env._target_pos[0] += 1.0e-6
    assert reset_physical_state_goal_digest(env) != first.digest


def test_standard_a4b_env_uses_fresh_mt1_benchmark_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    expected = SimpleNamespace()

    def fake_make_env(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return expected

    from fcpg import lewm_a4b_task_identity as identity

    monkeypatch.setattr(identity.metaworld_utils, "make_env", fake_make_env)
    observed = make_a4b_env(
        "pick-place-v3", "corner", 224, 224, raw_task_index=17
    )

    assert observed is expected
    assert calls == [
        (
            ("pick-place-v3", "corner", 224, 224),
            {"task_index": 17, "benchmark_seed": BENCHMARK_SEED},
        )
    ]


def test_digest_format_is_stable_lowercase_sha256() -> None:
    env = _fake_env(
        qpos=np.asarray([0.0]),
        goal=np.asarray([1.0]),
        task_data=np.asarray([2.0]),
    )
    digest = reset_physical_state_goal_digest(env)

    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
