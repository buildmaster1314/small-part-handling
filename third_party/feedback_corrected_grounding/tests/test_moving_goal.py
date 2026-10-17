from __future__ import annotations

import numpy as np
import pytest

from fcpg.moving_goal import (
    MovingGoalConfig,
    MovingGoalEnv,
    MovingPushExpert,
    SmoothMovingGoal,
)


class _Site:
    def __init__(self) -> None:
        self.pos = np.zeros(3, dtype=np.float64)


class _Model:
    def __init__(self) -> None:
        self._goal = _Site()

    def site(self, name: str) -> _Site:
        if name != "goal":
            raise KeyError(name)
        return self._goal


class _FakeEnv:
    def __init__(self) -> None:
        self._last_rand_vec = np.asarray(
            [0.0, 0.0, 0.0, 0.4, 0.1, 0.2], dtype=np.float64
        )
        self._target_pos = self._last_rand_vec[3:]
        self.model = _Model()
        self.data = None
        self.step_targets: list[np.ndarray] = []
        self.t = 0
        self.get_obs_calls = 0

    def reset(self, *, seed: int | None = None, options=None):
        del seed, options
        self.t = 0
        return self._get_obs(), {"reset": True}

    def _get_obs(self) -> np.ndarray:
        self.get_obs_calls += 1
        return np.asarray([float(self.t), *self._target_pos], dtype=np.float32)

    def _set_pos_site(self, name: str, position: np.ndarray) -> None:
        self.model.site(name).pos = np.asarray(position, dtype=np.float64)

    def step(self, action):
        del action
        self.step_targets.append(self._target_pos.copy())
        self.t += 1
        return self._get_obs(), 1.0, False, False, {}

    def close(self) -> None:
        return None


def test_smooth_goal_is_deterministic_and_starts_at_base() -> None:
    config = MovingGoalConfig(task="reach-v3", seed=9, period_raw_steps=40)
    first = SmoothMovingGoal(config)
    second = SmoothMovingGoal(config)
    base = np.asarray([0.5, 0.0, 0.2])
    np.testing.assert_allclose(first.reset(base), base)
    np.testing.assert_allclose(second.reset(base), base)
    first_positions = np.stack([first.position(step) for step in range(12)])
    second_positions = np.stack([second.position(step) for step in range(12)])
    np.testing.assert_allclose(first_positions, second_positions)
    assert np.max(np.linalg.norm(first_positions[1:, :2] - base[:2], axis=1)) > 1e-4
    assert np.all(np.isfinite(first_positions))


def test_push_goal_moves_forward_without_reversing() -> None:
    controller = SmoothMovingGoal(
        MovingGoalConfig(
            task="push-v3",
            seed=4,
            amplitude_xy=(0.02, 0.005),
            period_raw_steps=100,
        )
    )
    base = np.asarray([0.2, 0.8, 0.02])
    reference = np.asarray([0.1, 0.6])
    controller.reset(base, reference_xy=reference)
    positions = np.stack([controller.position(step) for step in range(121)])
    direction = base[:2] - reference
    direction /= np.linalg.norm(direction)
    forward = (positions[:, :2] - base[:2]) @ direction
    assert np.all(np.diff(forward) >= -1e-10)
    np.testing.assert_allclose(forward[-1], 0.02, atol=1e-8)
    np.testing.assert_allclose(positions[-1, 2], base[2])


def test_moving_goal_updates_target_site_observation_and_reward_target() -> None:
    base = _FakeEnv()
    env = MovingGoalEnv(base, task="push-v3", seed=3, period_raw_steps=20)
    observation, info = env.reset(seed=4)
    np.testing.assert_allclose(observation[-3:], base._target_pos)
    np.testing.assert_allclose(base.model.site("goal").pos, base._target_pos)
    assert info["moving_goal_raw_step"] == 0
    next_observation, _, _, _, next_info = env.step(np.zeros(4, dtype=np.float32))
    assert env.raw_step == 1
    np.testing.assert_allclose(base.step_targets[0], [0.4, 0.1, 0.2])
    np.testing.assert_allclose(
        next_info["moving_goal_reward_position"], base.step_targets[0]
    )
    np.testing.assert_allclose(next_observation[-3:], base._target_pos)
    np.testing.assert_allclose(next_info["moving_goal_position"], base._target_pos)
    assert not np.array_equal(base._target_pos, base.step_targets[0])
    assert base.get_obs_calls == 2


def test_moving_goal_breaks_task_alias_and_round_trips_state() -> None:
    base = _FakeEnv()
    env = MovingGoalEnv(base, task="reach-v3", seed=12, period_raw_steps=50)
    env.reset()
    original_task_target = base._last_rand_vec[3:].copy()
    env.step(np.zeros(4, dtype=np.float32))
    assert not np.shares_memory(base._last_rand_vec, base._target_pos)
    np.testing.assert_allclose(base._last_rand_vec[3:], original_task_target)
    state = env.state_dict()
    expected = base._target_pos.copy()
    env.step(np.zeros(4, dtype=np.float32))
    env.load_state_dict(state)
    np.testing.assert_allclose(base._target_pos, expected)
    assert env.raw_step == 1


def test_moving_push_expert_repositions_before_pushing() -> None:
    expert = MovingPushExpert()
    observation = np.zeros(39, dtype=np.float64)
    observation[:3] = [0.0, 0.7, 0.04]
    observation[4:7] = [0.0, 0.6, 0.02]
    observation[-3:] = [0.0, 0.85, 0.02]
    raise_action = expert.get_action(observation)
    assert raise_action[2] > 0.0

    observation[:3] = [0.0, 0.555, 0.045]
    push_action = expert.get_action(observation)
    assert push_action[1] > 0.0
    assert push_action[3] > 0.0


@pytest.mark.parametrize("task", ["reach-v3", "push-v3", "pick-place-v3"])
def test_supported_moving_goal_tasks(task: str) -> None:
    config = MovingGoalConfig(task=task, seed=1)
    assert config.task == task


def test_pick_goal_moves_smoothly_without_changing_height() -> None:
    controller = SmoothMovingGoal(
        MovingGoalConfig(
            task="pick-place-v3",
            seed=17,
            amplitude_xy=(0.01, 0.006),
            period_raw_steps=220,
        )
    )
    base = np.asarray([0.1, 0.75, 0.2])
    controller.reset(base)
    positions = np.stack([controller.position(step) for step in range(221)])
    step_distance = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    np.testing.assert_allclose(positions[0], base)
    np.testing.assert_allclose(positions[:, 2], base[2])
    assert np.max(step_distance) < 0.001
    assert np.max(np.linalg.norm(positions[:, :2] - base[:2], axis=1)) > 0.01


def test_unsupported_moving_goal_task_is_rejected() -> None:
    with pytest.raises(ValueError):
        MovingGoalConfig(task="button-press-v3", seed=1)
