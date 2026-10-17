import numpy as np
import pytest
from types import SimpleNamespace
import torch

from fcpg.lewm_goal_conditioned_mpc import (
    CANDIDATE_COUNT,
    HORIZON,
    build_smooth_candidate_bank,
    causal_action_history,
    predicted_goal_cost,
    repeated_action_prefix,
    selected_raw_chunk,
)
from scripts import evaluate_lewm_goal_conditioned_mpc as runner
from fcpg.lewm_goal_conditioned_proposer import (
    GoalConditionedActionPrefixProposer,
    advantage_weighted_prefix_target,
)


def test_candidate_bank_preserves_center_and_gripper() -> None:
    action = np.asarray([0.2, -0.1, 0.05, 0.7], dtype=np.float32)
    center = repeated_action_prefix(action)
    bank = build_smooth_candidate_bank(
        center,
        np.random.default_rng(17),
        amplitude=0.2,
    )
    assert bank.shape == (CANDIDATE_COUNT, HORIZON, 20)
    np.testing.assert_allclose(bank[0].reshape(40, 4), center)
    gripper = bank.reshape(CANDIDATE_COUNT, 40, 4)[:, :, 3]
    np.testing.assert_allclose(gripper, 0.7)
    assert not np.allclose(bank[1], bank[0])


def test_causal_history_left_pads_and_keeps_latest_actions() -> None:
    actions = [np.full(4, index, dtype=np.float32) for index in range(1, 4)]
    history = causal_action_history(actions).reshape(10, 4)
    np.testing.assert_allclose(history[:7], 0.0)
    np.testing.assert_allclose(history[-3:], np.asarray(actions))


def test_goal_cost_follows_moving_goal_and_selects_closest_terminal() -> None:
    predicted = np.zeros((2, HORIZON, 3), dtype=np.float32)
    velocity = np.asarray([0.001, 0.0, 0.0], dtype=np.float32)
    future_x = 5.0 * np.arange(1, HORIZON + 1, dtype=np.float32) * velocity[0]
    predicted[0, :, 0] = future_x
    predicted[1, :, 0] = future_x + 0.05
    cost = predicted_goal_cost(
        predicted,
        np.zeros(3, dtype=np.float32),
        velocity,
    )
    assert cost.shape == (2,)
    assert cost[0] < cost[1]


def test_goal_cost_can_score_only_the_executed_near_horizon() -> None:
    predicted = np.zeros((2, HORIZON, 3), dtype=np.float32)
    predicted[0, 0, 0] = 0.01
    predicted[0, -1, 0] = 1.0
    predicted[1, 0, 0] = 0.02
    predicted[1, -1, 0] = 0.0
    goal = np.zeros(3, dtype=np.float32)
    velocity = np.zeros(3, dtype=np.float32)
    near = predicted_goal_cost(
        predicted,
        goal,
        velocity,
        horizon_tokens=1,
        path_weight=0.0,
    )
    far = predicted_goal_cost(
        predicted,
        goal,
        velocity,
        horizon_tokens=HORIZON,
        path_weight=0.0,
    )
    assert near[0] < near[1]
    assert far[1] < far[0]


def test_selected_raw_chunk_returns_first_model_token() -> None:
    bank = np.arange(CANDIDATE_COUNT * HORIZON * 20, dtype=np.float32).reshape(
        CANDIDATE_COUNT, HORIZON, 20
    )
    selected = selected_raw_chunk(bank, 3)
    np.testing.assert_allclose(selected.reshape(-1), bank[3, 0])
    with pytest.raises(ValueError):
        selected_raw_chunk(bank, CANDIDATE_COUNT)


def test_oracle_expert_prefix_restores_source_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEnv:
        action_space = SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )

        def __init__(self) -> None:
            self.step_index = 7

        def step(self, action: np.ndarray):
            self.step_index += 1
            observation = np.full(3, self.step_index, dtype=np.float32)
            return observation, 0.0, False, False, {}

    class FakeExpert:
        def get_action(self, observation: np.ndarray) -> np.ndarray:
            return np.asarray(
                [0.01 * float(observation[0]), 0.0, 0.0, 0.5],
                dtype=np.float32,
            )

    env = FakeEnv()
    runtime = SimpleNamespace(env=env, expert=FakeExpert())
    monkeypatch.setattr(
        runner,
        "capture_branch_snapshot",
        lambda source, include_process_rng: int(source.step_index),
    )
    monkeypatch.setattr(
        runner,
        "restore_branch_snapshot",
        lambda source, snapshot: setattr(source, "step_index", int(snapshot)),
    )
    prefix = runner._oracle_expert_rollout_prefix(
        runtime,
        np.full(3, env.step_index, dtype=np.float32),
    )
    assert prefix.shape == (40, 4)
    assert env.step_index == 7
    assert prefix[-1, 0] > prefix[0, 0]


def test_advantage_weighted_prefix_prefers_better_terminal_outcome() -> None:
    actions = np.zeros((1, 2, HORIZON, 20), dtype=np.float32)
    actions[:, 1] = 1.0
    entity = np.zeros((1, 2, HORIZON, 3), dtype=np.float32)
    goal = np.zeros_like(entity)
    entity[:, 1, -1, 0] = 0.2
    target, weights = advantage_weighted_prefix_target(
        actions,
        entity,
        goal,
        temperature_m=0.01,
    )
    assert target.shape == (1, HORIZON, 20)
    assert weights[0, 0] > 0.99
    assert float(target.max()) < 0.01


def test_goal_conditioned_proposer_outputs_bounded_h8_prefix() -> None:
    model = GoalConditionedActionPrefixProposer(hidden_dim=64, slot_dim=32)
    output = model(
        current_state=torch.zeros(2, 4, 192),
        history_action=torch.zeros(2, 2, 20),
        current_goal_xyz=torch.zeros(2, 3),
        current_goal_velocity=torch.zeros(2, 3),
        current_world_xyz=torch.zeros(2, 3, 3),
        task_index=torch.tensor([2, 3]),
    )
    assert output.action_prefix.shape == (2, HORIZON, 20)
    assert bool((output.action_prefix.abs() <= 1.0).all())


def test_entity_conditioned_proposer_requires_and_uses_structured_entity() -> None:
    model = GoalConditionedActionPrefixProposer(
        hidden_dim=64,
        slot_dim=32,
        use_entity_estimate=True,
    )
    inputs = {
        "current_state": torch.zeros(2, 4, 192),
        "history_action": torch.zeros(2, 2, 20),
        "current_goal_xyz": torch.zeros(2, 3),
        "current_goal_velocity": torch.zeros(2, 3),
        "current_world_xyz": torch.zeros(2, 3, 3),
        "task_index": torch.tensor([2, 3]),
    }
    with pytest.raises(ValueError, match="current_entity_xyz"):
        model(**inputs)
    output = model(**inputs, current_entity_xyz=torch.zeros(2, 3))
    assert output.action_prefix.shape == (2, HORIZON, 20)
