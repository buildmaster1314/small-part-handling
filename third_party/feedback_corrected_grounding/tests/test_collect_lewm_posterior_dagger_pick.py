from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import pytest
import torch

from scripts.collect_lewm_posterior_dagger_pick import (
    _dagger_api,
    _merge_rows,
    _validate_config,
    apply_fixed_camera_condition,
    behavior_head_contract,
    choose_learner_action,
    collection_contract,
    execute_learner_action_block,
    load_behavior_head,
    parse_camera_pose_condition,
    query_expert_action_block,
)


class _Space:
    low = np.full(4, -1.0, dtype=np.float32)
    high = np.full(4, 1.0, dtype=np.float32)


class _FakeEnv:
    def __init__(self) -> None:
        self.state = 2.0
        self.action_space = _Space()

    def step(self, action: np.ndarray):
        random.random()
        np.random.random()
        torch.rand(1)
        self.state += float(action[0])
        observation = np.asarray([self.state, 7.0], dtype=np.float32)
        return observation, self.state, False, False, {"success": 0.0}


class _MutatingExpert:
    def __init__(self) -> None:
        self.calls = 0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        self.calls += 1
        observation[...] = -99.0
        random.random()
        np.random.random()
        torch.rand(1)
        return np.asarray([0.75, 0.0, 0.0, 0.0], dtype=np.float32)


def _capture_fake(env: _FakeEnv, *, include_process_rng: bool) -> dict[str, Any]:
    assert include_process_rng is True
    return {
        "state": env.state,
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }


def _restore_fake(
    env: _FakeEnv, snapshot: dict[str, Any], *, restore_process_rng: bool
) -> None:
    assert restore_process_rng is True
    env.state = float(snapshot["state"])
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])


def _next_rng_values() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(1))


def test_expert_query_restores_environment_rng_observation_and_expert() -> None:
    env = _FakeEnv()
    expert = _MutatingExpert()
    observation = np.asarray([env.state, 7.0], dtype=np.float32)
    original = observation.copy()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    expected = _next_rng_values()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    block = query_expert_action_block(
        env,
        expert,
        observation,
        capture_snapshot=_capture_fake,
        restore_snapshot=_restore_fake,
    )

    assert block.shape == (5, 4)
    np.testing.assert_allclose(block[:, 0], 0.75)
    assert env.state == 2.0
    assert expert.calls == 0
    np.testing.assert_array_equal(observation, original)
    assert _next_rng_values() == pytest.approx(expected)


def test_learner_not_expert_advances_the_main_trajectory() -> None:
    env = _FakeEnv()
    observation = np.asarray([env.state, 7.0], dtype=np.float32)
    expert_block = query_expert_action_block(
        env,
        _MutatingExpert(),
        observation,
        capture_snapshot=_capture_fake,
        restore_snapshot=_restore_fake,
    )
    assert env.state == 2.0

    learner_block = np.zeros((5, 4), dtype=np.float32)
    learner_block[:, 0] = -0.25
    result = execute_learner_action_block(env, observation, learner_block)

    assert env.state == pytest.approx(0.75)
    assert float(result.observation[0]) == pytest.approx(0.75)
    assert env.state != pytest.approx(2.0 + float(expert_block[:, 0].sum()))


def test_expert_query_restores_after_branch_failure() -> None:
    class FailingExpert(_MutatingExpert):
        def get_action(self, observation: np.ndarray) -> np.ndarray:
            if self.calls == 1:
                raise RuntimeError("branch failure")
            return super().get_action(observation)

    env = _FakeEnv()
    expert = FailingExpert()
    observation = np.asarray([2.0, 7.0], dtype=np.float32)
    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    expected = _next_rng_values()
    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)

    with pytest.raises(RuntimeError, match="branch failure"):
        query_expert_action_block(
            env,
            expert,
            observation,
            capture_snapshot=_capture_fake,
            restore_snapshot=_restore_fake,
        )

    assert env.state == 2.0
    assert expert.calls == 0
    assert _next_rng_values() == pytest.approx(expected)


def test_valid_measurement_exactly_bypasses_loaded_head() -> None:
    class ExplodingHead(torch.nn.Module):
        def forward(self, **_kwargs):
            raise AssertionError("valid visual input must not call the residual head")

    base = torch.randn(1, 5, 4)
    result = choose_learner_action(
        base_action=base,
        state_latent=torch.randn(1, 8),
        action_latent=torch.randn(1, 5, 8),
        posterior_token=torch.randn(1, 3, 8),
        missing_age=0,
        sensor_valid=True,
        head=ExplodingHead(),
    )
    assert result is base
    assert torch.equal(result, base)


def test_camera_condition_validation_and_pose_hook() -> None:
    condition = parse_camera_pose_condition(
        {
            "camera_condition": "fixed_rotation",
            "camera_rotation_axis_world": [0.0, 0.0, 2.0],
            "camera_rotation_deg": 5.0,
        }
    )
    np.testing.assert_allclose(condition.axis_world, [0.0, 0.0, 1.0])
    assert condition.rotation_deg == 5.0

    class Model:
        cam_pos = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
        cam_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    class Env:
        model = Model()
        data = object()

    calls: list[tuple[Any, ...]] = []

    def setter(*args: Any) -> None:
        calls.append(args)

    apply_fixed_camera_condition(Env(), 0, condition, setter=setter)
    assert len(calls) == 1
    assert calls[0][2] == 0
    np.testing.assert_array_equal(calls[0][3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(calls[0][4], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(calls[0][5], np.zeros(3))
    np.testing.assert_array_equal(calls[0][6], [0.0, 0.0, 1.0])
    assert calls[0][7] == 5.0


@pytest.mark.parametrize(
    "config",
    [
        {
            "camera_condition": "nominal",
            "camera_rotation_axis_world": [0.0, 0.0, 1.0],
            "camera_rotation_deg": 5.0,
        },
        {
            "camera_condition": "fixed_rotation",
            "camera_rotation_axis_world": [0.0, 0.0, 0.0],
            "camera_rotation_deg": 5.0,
        },
        {
            "camera_condition": "fixed_rotation",
            "camera_rotation_axis_world": [0.0, 0.0, 1.0],
            "camera_rotation_deg": 0.0,
        },
    ],
)
def test_camera_condition_rejects_non_joint_conditions(config) -> None:
    with pytest.raises(ValueError):
        parse_camera_pose_condition(config)


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": "fcpg_lewm_posterior_dagger_v1",
        "task": "pick-place-v3",
        "camera": "corner",
        "width": 224,
        "height": 224,
        "action_steps_per_decision": 5,
        "drop_duration_decisions": 6,
        "drop_start_min_decision": 6,
        "drop_start_max_decision": 9,
        "occlusion_center_key": "q_nominal",
        "known_sensor_validity_only": True,
        "camera_condition": "fixed_rotation",
        "camera_rotation_axis_world": [0.0, 0.0, 1.0],
        "camera_rotation_deg": 5.0,
        "policy_seed": 0,
        "model_seed": 15801,
        "source_fresh_gate_config": "source.json",
        "occlusion_radius_px": 24,
        "occlusion_value_uint8": 0,
        "training": {"steps": 2000, "seeds": [64301]},
        "splits": {
            name: {
                "benchmark_seed": seed,
                "environment_seed_base": seed + 1,
                "task_index_base": seed + 2,
                "episodes": 2,
                "schedule_seed": seed + 3,
            }
            for name, seed in (("train", 10), ("development", 20))
        },
    }


def test_joint_drift_config_is_strict() -> None:
    config = _valid_config()
    _validate_config(config, protocol="fcpg_lewm_posterior_dagger_v1")
    config["occlusion_center_key"] = "q_true"
    with pytest.raises(ValueError, match="q_nominal"):
        _validate_config(config, protocol="fcpg_lewm_posterior_dagger_v1")


@pytest.mark.parametrize("policy_seed", [1, -1, None, False])
def test_collector_config_is_hard_bound_to_policy_seed_zero(
    policy_seed: object,
) -> None:
    config = _valid_config()
    config["policy_seed"] = policy_seed
    with pytest.raises(ValueError, match="policy_seed=0"):
        _validate_config(config, protocol="fcpg_lewm_posterior_dagger_v1")


@pytest.mark.parametrize("policy_seed", [None, False])
def test_collection_contract_reports_policy_seed_contract_before_types(
    policy_seed: object,
) -> None:
    config = _valid_config()
    config["policy_seed"] = policy_seed
    with pytest.raises(ValueError, match="policy_seed=0"):
        collection_contract(config)


def test_collection_contract_records_plain_matched_fields() -> None:
    config = _valid_config()
    assert collection_contract(config) == {
        "policy_seed": 0,
        "model_seed": 15801,
        "drop_duration_decisions": 6,
        "drop_start_min_decision": 6,
        "drop_start_max_decision": 9,
        "occlusion_radius_px": 24,
        "occlusion_value_uint8": 0,
        "occlusion_center_key": "q_nominal",
        "camera_condition": {
            "kind": "fixed_rotation",
            "camera_rotation_deg": 5.0,
            "camera_rotation_axis_world": [0.0, 0.0, 1.0],
        },
    }


def test_collection_contract_records_camera_axis_without_normalizing() -> None:
    config = _valid_config()
    config["camera_rotation_axis_world"] = [0.0, 0.0, 2.0]
    assert collection_contract(config)["camera_condition"] == {
        "kind": "fixed_rotation",
        "camera_rotation_deg": 5.0,
        "camera_rotation_axis_world": [0.0, 0.0, 2.0],
    }


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("model_seed", "15801", "model_seed must be an integer"),
        ("occlusion_radius_px", False, "occlusion_radius_px must be an integer"),
        ("camera_rotation_deg", "5.0", "camera_rotation_deg must be numeric"),
        (
            "camera_rotation_axis_world",
            [0.0, False, 1.0],
            "camera_rotation_axis_world must contain three numbers",
        ),
    ],
)
def test_collection_contract_rejects_silent_type_coercion(
    key: str, value: object, error: str
) -> None:
    config = _valid_config()
    config[key] = value
    with pytest.raises(TypeError, match=error):
        collection_contract(config)


def test_round_zero_forbids_behavior_head() -> None:
    config = _valid_config()
    contract = behavior_head_contract(
        config, dagger_iteration=0, head_checkpoint=None
    )
    assert contract.required is False
    assert contract.expected_seed == 64301
    assert contract.expected_steps == 2000
    assert contract.expected_dagger_iteration is None
    with pytest.raises(ValueError, match="iteration 0 forbids"):
        behavior_head_contract(
            config, dagger_iteration=0, head_checkpoint="unexpected.pt"
        )


def test_later_round_requires_preceding_head() -> None:
    config = _valid_config()
    with pytest.raises(ValueError, match="iteration >0 requires"):
        behavior_head_contract(config, dagger_iteration=2, head_checkpoint=None)
    contract = behavior_head_contract(
        config, dagger_iteration=2, head_checkpoint="round1.pt"
    )
    assert contract.required is True
    assert contract.expected_dagger_iteration == 1


@pytest.mark.parametrize(
    ("training", "error"),
    [
        ({"steps": 2000, "seeds": []}, "exactly one"),
        ({"steps": 2000, "seeds": [1, 2]}, "exactly one"),
        ({"steps": 0, "seeds": [1]}, "must be positive"),
    ],
)
def test_behavior_head_contract_requires_unique_training_endpoint(
    training: dict[str, object], error: str
) -> None:
    config = _valid_config()
    config["training"] = training
    with pytest.raises(ValueError, match=error):
        behavior_head_contract(
            config, dagger_iteration=1, head_checkpoint="round0.pt"
        )


def test_behavior_head_loader_strictly_binds_previous_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    sentinel_head = torch.nn.Identity()
    sentinel_metadata = {"dagger_iteration": 2}

    class Api:
        @staticmethod
        def load_posterior_dagger_checkpoint(path, **kwargs):
            calls.append({"path": path, **kwargs})
            return sentinel_head, sentinel_metadata

    monkeypatch.setattr(
        "scripts.collect_lewm_posterior_dagger_pick._resolve",
        lambda path: path,
    )
    head, metadata, contract = load_behavior_head(
        Api,
        _valid_config(),
        method="complementary_factual",
        dagger_iteration=3,
        head_checkpoint="round2.pt",
        device=torch.device("cpu"),
    )

    assert head is sentinel_head
    assert metadata is sentinel_metadata
    assert contract.expected_dagger_iteration == 2
    assert calls == [
        {
            "path": "round2.pt",
            "device": torch.device("cpu"),
            "expected_method": "complementary_factual",
            "expected_seed": 64301,
            "expected_steps": 2000,
            "expected_dagger_iteration": 2,
        }
    ]


def test_collected_rows_match_shared_single_method_schema() -> None:
    api = _dagger_api()
    row = {
        "episode_id": 7,
        "decision_index": 4,
        "dagger_iteration": 1,
        "missing_age": 2,
        "sensor_valid": False,
        "state_latent": np.zeros(8, dtype=np.float32),
        "action_latent": np.zeros((5, 8), dtype=np.float32),
        "base_action": np.zeros((5, 4), dtype=np.float32),
        "posterior_token": np.zeros((3, 8), dtype=np.float32),
        "expert_action": np.zeros((5, 4), dtype=np.float32),
        "executed_action": np.zeros((5, 4), dtype=np.float32),
    }
    arrays = _merge_rows(api, [row], method="complementary_factual")
    validated = api.validate_posterior_dagger_arrays(arrays)
    assert validated.sample_count == 1
    assert validated.method_index.tolist() == [0]
