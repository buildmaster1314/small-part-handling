from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from fcpg.lewm_action_branch_a1 import NPZ_KEYS, validate_a1_arrays
from fcpg.lewm_action_branch_collection import (
    CANDIDATE_COUNT,
    FORMAL_TASK_INSTANCES,
    SMOKE_TASK_INSTANCES,
    ActionBranchRow,
    EpisodeRuntime,
    HazardScheduleConfig,
    assign_source_episode_splits,
    build_action_branch_bank,
    collect_source_episode,
    encode_fcpg_anchor_records,
    execute_action_branches,
    sample_hazard_schedule,
    stack_action_branch_rows,
)
from fcpg.lewm_counterfactual_runtime import FrameRecord
from scripts import collect_lewm_action_branch_a1 as cli


def _contract() -> dict[str, Any]:
    return {
        "tasks": ["pick-place-v3", "reach-v3"],
        "split": {
            "protocol_seed": 18_301,
            "smoke_episode_namespace": "a1-smoke-disjoint",
            "assignment": "task-local-seeded-permutation",
        },
        "hazard": {
            "decision_steps_per_episode": 40,
            "history_steps": 3,
            "eligible_first_step": 3,
            "eligible_last_step": 34,
            "bernoulli_probability": 0.12,
            "refractory_steps": 5,
            "maximum_events_per_episode": 3,
            "condition_on_at_least_one_event": True,
            "schedule_drawn_before_environment_reset": True,
            "schedule_independent_of_state_action_reward_and_task_phase": True,
        },
        "matched_state_branches": {
            "branch_count": 8,
            "motion_delta_rms": 0.15,
            "preserve_policy_gripper_commands": True,
        },
    }


def _record(value: float, *, size: int = 8) -> FrameRecord:
    q = np.full((3, 2), min(float(size - 1), 2.0 + value * 0.01), dtype=np.float32)
    return FrameRecord(
        rgb=np.full((3, size, size), value * 0.001, dtype=np.float32),
        proprio=np.full(4, value, dtype=np.float32),
        keypoints_xyz=np.full((3, 3), value * 0.01, dtype=np.float32),
        q_nominal=q,
        valid_nominal=np.ones(3, dtype=bool),
        q_true=np.full((3, 2), 777.0, dtype=np.float32),
        valid_true=np.zeros(3, dtype=bool),
    )


def _fake_embedding(records: list[FrameRecord] | tuple[FrameRecord, ...]) -> np.ndarray:
    result = np.empty((len(records), 3, 192), dtype=np.float32)
    for index, record in enumerate(records):
        result[index] = float(record.proprio[0])
    return result


def test_hazard_schedule_is_deterministic_refractory_and_bounded() -> None:
    config = HazardScheduleConfig(
        decision_steps=40,
        eligible_first_step=3,
        eligible_last_step=34,
        probability=0.35,
        refractory_steps=5,
        maximum_events=3,
        condition_on_at_least_one_event=True,
    )
    first = sample_hazard_schedule(config, rng=np.random.default_rng(91))
    second = sample_hazard_schedule(config, rng=np.random.default_rng(91))

    assert first == second
    assert 1 <= len(first) <= 3
    assert all(3 <= step <= 34 for step in first)
    assert all(right - left >= 5 for left, right in zip(first, first[1:]))


def test_split_uses_formal_0_to_47_and_smoke_48_49_without_overlap() -> None:
    tasks = tuple(_contract()["tasks"])
    formal_instances = {task: FORMAL_TASK_INSTANCES for task in tasks}
    first = assign_source_episode_splits(tasks, formal_instances, _contract(), smoke=False)
    second = assign_source_episode_splits(tasks, formal_instances, _contract(), smoke=False)

    assert first == second
    for task in tasks:
        labels = [first[(task, instance)] for instance in FORMAL_TASK_INSTANCES]
        assert labels.count("train") == 24
        assert labels.count("development") == 12
        assert labels.count("test") == 12
        assert set(instance for name, instance in first if name == task) == set(range(48))

    smoke = assign_source_episode_splits(
        tasks,
        {task: SMOKE_TASK_INSTANCES for task in tasks},
        _contract(),
        smoke=True,
    )
    assert set(smoke.values()) == {"smoke"}
    assert {instance for _, instance in smoke} == {48, 49}
    with pytest.raises(ValueError, match="48 and 49"):
        assign_source_episode_splits(
            tasks,
            {task: (0,) for task in tasks},
            _contract(),
            smoke=True,
        )


class _FakeExpert:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        index = len(self.actions)
        action = np.asarray(
            [
                0.16 * np.sin(index * 0.11),
                0.14 * np.cos(index * 0.07),
                0.12 * np.sin(index * 0.05 + 0.3),
                -1.0 if index % 11 < 5 else 1.0,
            ],
            dtype=np.float32,
        )
        self.actions.append(action.copy())
        return action


class _FakeEnv:
    def __init__(self) -> None:
        self.action_space = SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )
        self.state = 0.0
        self.raw_step = 0
        self.closed = False

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.raw_step += 1
        self.state += float(action[0]) + 0.1 * float(action[1])
        observation = np.full(4, self.state, dtype=np.float32)
        return observation, self.state, False, False, {"ignored": self.raw_step}

    def close(self) -> None:
        self.closed = True


def test_source_episode_draws_hazards_before_reset_and_uses_exact_scripted_center() -> None:
    order: list[str] = []
    restore_states: list[tuple[float, int]] = []
    env = _FakeEnv()
    expert = _FakeExpert()
    branch_builder_calls: list[tuple[int, int]] = []

    def schedule(config: HazardScheduleConfig, *, rng: np.random.Generator) -> tuple[int, ...]:
        del config, rng
        order.append("schedule")
        return (3, 8)

    def runtime_factory(task: str, task_index: int, seed: int) -> EpisodeRuntime:
        assert (task, task_index, seed) == ("pick-place-v3", 48, 1_040_048)
        order.append("reset")
        return EpisodeRuntime(
            env=env,
            expert=expert,
            observation=np.zeros(4, dtype=np.float32),
            record_current=lambda observation: _record(float(observation[0])),
        )

    def capture(fake_env: _FakeEnv, *, include_process_rng: bool) -> tuple[float, int]:
        assert include_process_rng is True
        return fake_env.state, fake_env.raw_step

    def restore(fake_env: _FakeEnv, snapshot: tuple[float, int]) -> None:
        restore_states.append(snapshot)
        fake_env.state, fake_env.raw_step = snapshot

    def branch_builder(center_action: np.ndarray, **kwargs: Any) -> Any:
        branch_builder_calls.append(
            (int(kwargs["direction_seed"]), int(kwargs["permutation_seed"]))
        )
        return build_action_branch_bank(center_action, **kwargs)

    rows = collect_source_episode(
        task="pick-place-v3",
        task_seed_index=0,
        task_index_mod50=48,
        environment_seed=1_040_048,
        split="smoke",
        contract=_contract(),
        horizon_steps=8,
        runtime_factory=runtime_factory,
        encode_records=_fake_embedding,
        branch_bank_builder=branch_builder,
        capture_snapshot=capture,
        restore_snapshot=restore,
        schedule_sampler=schedule,
    )

    assert order == ["schedule", "reset"]
    assert len(rows) == 2
    assert len(branch_builder_calls) == 2
    assert branch_builder_calls[0] != branch_builder_calls[1]
    assert len(restore_states) == 2 * CANDIDATE_COUNT
    assert restore_states[:8] == [restore_states[0]] * 8
    assert restore_states[8:] == [restore_states[8]] * 8
    source_actions = np.stack(expert.actions)
    np.testing.assert_array_equal(rows[0].center_action, source_actions[15:55])
    np.testing.assert_array_equal(rows[1].center_action, source_actions[40:80])
    assert rows[0].center_action.shape == rows[1].center_action.shape == (40, 4)
    assert rows[0].history_action.shape == (2, 5, 4)
    assert rows[0].future_embedding.shape == (8, 8, 3, 192)
    assert rows[0].task_index == 0
    assert rows[0].task_instance_index == 48
    assert env.closed is True

    arrays = stack_action_branch_rows(rows)
    assert tuple(arrays) == NPZ_KEYS
    axes = validate_a1_arrays(arrays)
    assert axes.sample_count == 2
    assert axes.horizon == 8
    np.testing.assert_array_equal(arrays["task_index"], np.asarray([0, 0]))
    np.testing.assert_array_equal(
        arrays["task_instance_index"], np.asarray([48, 48])
    )
    assert arrays["center_action"].shape == (2, 8, 20)
    np.testing.assert_array_equal(
        arrays["center_action"][0].reshape(40, 4), source_actions[15:55]
    )
    assert set(arrays["source_episode_id"].tolist()) == {
        "pick-place-v3:task-instance-48"
    }
    assert not any(
        token in name
        for name in arrays
        for token in ("rgb", "snapshot", "q_true", "canonical")
    )


def test_branch_executor_restores_eight_times_and_encodes_current_endpoints() -> None:
    env = _FakeEnv()
    runtime = EpisodeRuntime(
        env=env,
        expert=None,
        observation=np.zeros(4, dtype=np.float32),
        record_current=lambda observation: _record(float(observation[0])),
    )
    actions = np.zeros((8, 40, 4), dtype=np.float32)
    for branch in range(8):
        actions[branch, :, 0] = 0.01 * (branch + 1)
    restored: list[int] = []

    def restore(fake_env: _FakeEnv, snapshot: tuple[float, int]) -> None:
        restored.append(1)
        fake_env.state, fake_env.raw_step = snapshot

    result = execute_action_branches(
        runtime,
        (0.0, 0),
        actions,
        encode_records=_fake_embedding,
        restore_snapshot=restore,
    )

    assert len(restored) == 8
    assert result.shape == (8, 8, 3, 192)
    np.testing.assert_allclose(result[0, :, 0, 0], np.arange(1, 9) * 0.05)
    np.testing.assert_allclose(result[7, :, 0, 0], np.arange(1, 9) * 0.4)


class _FakeAnchorModel:
    def __init__(self) -> None:
        self.batch: dict[str, torch.Tensor] | None = None

    def encode_frames(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.batch = batch
        assert set(batch) == {
            "pixels",
            "proprio",
            "q_nominal",
            "q_corrected",
            "keypoints_xyz",
            "valid",
        }
        count = int(batch["pixels"].shape[1])
        embedding = torch.zeros((1, count, 3, 192), dtype=torch.float32)
        embedding += batch["q_corrected"].mean()
        return embedding, embedding


def test_encoder_uses_fcpg_corrected_current_images_without_q_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [_record(1.0), _record(2.0), _record(3.0)]
    seen: list[FrameRecord] = []

    def correct(rows: list[FrameRecord], frozen: Any) -> np.ndarray:
        assert frozen.corrector == "frozen-fcpg"
        seen.extend(rows)
        return np.stack([record.q_nominal + 1.0 for record in rows])

    monkeypatch.setattr(
        "fcpg.lewm_action_branch_collection.corrected_points",
        correct,
    )
    model = _FakeAnchorModel()
    result = encode_fcpg_anchor_records(
        records,
        model=model,
        corrector="frozen-fcpg",
        device=torch.device("cpu"),
    )

    assert seen == records
    assert result.shape == (3, 3, 192)
    assert model.batch is not None
    assert "q_true" not in model.batch
    assert "canonical" not in model.batch
    np.testing.assert_array_equal(
        model.batch["q_corrected"][0].numpy(),
        np.stack([record.q_nominal + 1.0 for record in records]),
    )


def _row(*, split: str) -> ActionBranchRow:
    return ActionBranchRow(
        task="reach-v3",
        task_index=2,
        task_instance_index=7,
        source_episode_id="reach-v3:task-instance-07",
        environment_seed=50_007,
        hazard_step=3,
        split=split,
        center_action=np.zeros((40, 4), dtype=np.float32),
        history_action=np.zeros((2, 5, 4), dtype=np.float32),
        branch_action=np.zeros((8, 40, 4), dtype=np.float32),
        history_embedding=np.zeros((3, 3, 192), dtype=np.float32),
        future_embedding=np.zeros((8, 8, 3, 192), dtype=np.float32),
    )


def test_stacker_rejects_same_task_instance_crossing_splits() -> None:
    first = _row(split="train")
    second = replace(first, hazard_step=8, split="test")

    with pytest.raises(ValueError, match="task instance crosses splits"):
        stack_action_branch_rows([first, second])


def test_cli_exposes_smoke_bounds_and_never_accepts_an_action_policy() -> None:
    args = cli._parser().parse_args(
        [
            "--smoke",
            "--episodes-per-task",
            "2",
            "--max-events",
            "1",
            "--horizon",
            "8",
            "--device",
            "cpu",
            "--output",
            "smoke.npz",
        ]
    )

    assert args.episodes_per_task == 2
    assert args.max_events == 1
    assert args.horizon == 8
    assert args.device == "cpu"
    assert args.output == "smoke.npz"
    assert "anchor_persistent_sig002_seed9216_2k" in args.anchor_checkpoint
    assert "shared_4task_seed0_sig0" in args.fcpg_checkpoint
    assert not hasattr(args, "action_policy")
