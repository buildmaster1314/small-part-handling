from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


pytest.importorskip("metaworld")
pytest.importorskip("mujoco")

from fcpg.metaworld_snapshot import (  # noqa: E402
    MetaWorldSnapshot,
    capture_metaworld_snapshot,
    restore_metaworld_snapshot,
)
from fcpg.metaworld_utils import make_env, make_scripted_policy  # noqa: E402


_TASK = "pick-place-v3"
_ACTION_STEPS = 5
_LEARNER_ACTIONS = np.asarray(
    [
        [0.12, -0.08, 0.04, 0.60],
        [-0.06, 0.10, -0.03, -0.40],
        [0.03, 0.02, 0.08, 0.20],
        [-0.08, -0.04, 0.02, -0.70],
        [0.05, 0.06, -0.05, 0.10],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class _StepRecord:
    observation: np.ndarray
    reward: float
    success: float
    terminated: bool
    truncated: bool


@pytest.fixture
def pick_branch_point() -> tuple[Any, Any, np.ndarray]:
    env = make_env(
        _TASK,
        "corner",
        64,
        64,
        task_index=0,
        benchmark_seed=7419,
    )
    expert = make_scripted_policy(_TASK)
    try:
        observation, _info = env.reset(seed=7420)
        observation = np.array(observation, dtype=np.float64, copy=True, order="C")

        # Move away from reset so the snapshot covers MetaWorld's observation history.
        for _ in range(2):
            action = _bounded_action(env, expert.get_action(observation.copy()))
            observation, _reward, terminated, truncated, _info = env.step(action)
            assert not terminated and not truncated
            observation = np.array(observation, dtype=np.float64, copy=True, order="C")

        yield env, expert, observation
    finally:
        env.close()


def _bounded_action(env: Any, action: Any) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    assert value.shape == (4,)
    assert np.isfinite(value).all()
    return np.clip(value, env.action_space.low, env.action_space.high).astype(
        np.float32, copy=False
    )


def _run_expert_branch(env: Any, expert: Any, observation: np.ndarray) -> np.ndarray:
    current = np.array(observation, copy=True, order="C")
    actions: list[np.ndarray] = []
    for index in range(_ACTION_STEPS):
        action = _bounded_action(env, expert.get_action(current.copy()))
        actions.append(action.copy())
        current, _reward, terminated, truncated, _info = env.step(action)
        current = np.array(current, copy=True, order="C")
        assert not (terminated or truncated), f"expert branch ended at step {index + 1}"
    return np.ascontiguousarray(np.stack(actions), dtype=np.float32)


def _run_learner_chunk(
    env: Any, observation: np.ndarray, actions: np.ndarray
) -> tuple[_StepRecord, ...]:
    current = np.array(observation, copy=True, order="C")
    records: list[_StepRecord] = []
    for index, action in enumerate(actions):
        current, reward, terminated, truncated, info = env.step(action)
        current = np.array(current, copy=True, order="C")
        records.append(
            _StepRecord(
                observation=current,
                reward=float(reward),
                success=float(info["success"]),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )
        assert not (terminated or truncated), f"learner chunk ended at step {index + 1}"
    return tuple(records)


def _assert_records_equal(
    expected: tuple[_StepRecord, ...], actual: tuple[_StepRecord, ...]
) -> None:
    assert len(expected) == len(actual) == _ACTION_STEPS
    for expected_step, actual_step in zip(expected, actual):
        np.testing.assert_array_equal(
            expected_step.observation, actual_step.observation
        )
        assert (
            expected_step.observation.dtype == actual_step.observation.dtype
            and expected_step.observation.tobytes(order="C")
            == actual_step.observation.tobytes(order="C")
        )
        assert expected_step.reward == actual_step.reward
        assert expected_step.success == actual_step.success
        assert expected_step.terminated == actual_step.terminated
        assert expected_step.truncated == actual_step.truncated


def _assert_mujoco_state_equal(
    expected: MetaWorldSnapshot, actual: MetaWorldSnapshot
) -> None:
    for expected_array, actual_array in (
        (expected.integration_state, actual.integration_state),
        (expected.warmstart_state, actual.warmstart_state),
        (expected.model_arrays.body_pos, actual.model_arrays.body_pos),
        (expected.model_arrays.site_pos, actual.model_arrays.site_pos),
        (expected.model_arrays.cam_pos, actual.model_arrays.cam_pos),
        (expected.model_arrays.cam_quat, actual.model_arrays.cam_quat),
    ):
        np.testing.assert_array_equal(expected_array, actual_array)
        assert expected_array.dtype == actual_array.dtype
        assert expected_array.tobytes(order="C") == actual_array.tobytes(order="C")


def test_five_step_expert_branch_does_not_change_learner_path(
    pick_branch_point: tuple[Any, Any, np.ndarray],
) -> None:
    env, expert, observation = pick_branch_point
    caller_observation = np.array(observation, copy=True, order="C")
    caller_observation_bytes = caller_observation.tobytes(order="C")
    branch_point = capture_metaworld_snapshot(env, include_process_rng=True)
    assert branch_point.process_rng is not None

    # Control trajectory: no expert branch is evaluated before the learner chunk.
    control_records = _run_learner_chunk(env, caller_observation, _LEARNER_ACTIONS)
    control_final = capture_metaworld_snapshot(env)

    restore_metaworld_snapshot(env, branch_point, restore_process_rng=True)
    _run_expert_branch(env, expert, caller_observation)
    restore_metaworld_snapshot(env, branch_point, restore_process_rng=True)

    assert caller_observation.tobytes(order="C") == caller_observation_bytes
    branched_records = _run_learner_chunk(env, caller_observation, _LEARNER_ACTIONS)
    branched_final = capture_metaworld_snapshot(env)

    _assert_records_equal(control_records, branched_records)
    _assert_mujoco_state_equal(control_final, branched_final)


def test_repeated_five_step_expert_labels_are_identical_after_restore(
    pick_branch_point: tuple[Any, Any, np.ndarray],
) -> None:
    env, expert, observation = pick_branch_point
    caller_observation = np.array(observation, copy=True, order="C")
    branch_point = capture_metaworld_snapshot(env, include_process_rng=True)
    assert branch_point.process_rng is not None

    first_labels = _run_expert_branch(env, expert, caller_observation)
    first_rng_draws = (random.random(), np.random.random(4))
    restore_metaworld_snapshot(env, branch_point, restore_process_rng=True)

    second_labels = _run_expert_branch(env, expert, caller_observation)
    second_rng_draws = (random.random(), np.random.random(4))
    restore_metaworld_snapshot(env, branch_point, restore_process_rng=True)

    np.testing.assert_array_equal(first_labels, second_labels)
    assert first_labels.tobytes(order="C") == second_labels.tobytes(order="C")
    assert first_rng_draws[0] == second_rng_draws[0]
    np.testing.assert_array_equal(first_rng_draws[1], second_rng_draws[1])
