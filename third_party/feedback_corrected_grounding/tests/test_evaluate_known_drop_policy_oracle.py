from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.evaluate_known_drop_policy_oracle as evaluator


class _FakePolicy:
    num_keypoints = 2

    def _observation_tokens(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = batch["rgb"]
        batch_size, history = rgb.shape[:2]
        scalar = rgb.mean(dim=(2, 3, 4), keepdim=False)
        grounded = scalar[:, :, None, None].expand(-1, -1, 2, 4).clone()
        if "grounded_token_override" in batch:
            mask = batch["grounded_token_override_mask"]
            grounded = torch.where(
                mask.unsqueeze(-1), batch["grounded_token_override"], grounded
            )
        global_token = torch.zeros(batch_size, history, 1, 4)
        robot_token = torch.zeros(batch_size, history, 1, 4)
        memory = torch.cat([global_token, robot_token, grounded], dim=2)
        points = batch["q_nominal"]
        reliability = torch.ones(batch_size, history, 2)
        return memory, points, reliability

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]):
        memory, _, _ = self._observation_tokens(batch)
        value = memory[:, -1, 2:].mean(dim=(1, 2))
        action = value[:, None, None].expand(-1, 8, 4).clone()
        return SimpleNamespace(action=action)


def _batch() -> dict[str, torch.Tensor]:
    rgb = torch.ones(1, 2, 3, 8, 8)
    return {
        "rgb": rgb,
        "obs": torch.zeros(1, 2, 4),
        "q_true": torch.tensor([[[[2.0, 2.0], [5.0, 5.0]]] * 2]),
        "q_nominal": torch.tensor([[[[2.0, 2.0], [5.0, 5.0]]] * 2]),
        "keypoints_xyz": torch.zeros(1, 2, 2, 3),
    }


def test_active_oracle_restores_clean_anchor_action_while_measurement_does_not() -> None:
    model = _FakePolicy()
    batch = _batch()
    common = {
        "active": True,
        "last_valid_tokens": torch.full((1, 2, 4), 0.25),
        "occlusion_radius_px": 1,
        "occlusion_value": 0.0,
        "device": torch.device("cpu"),
    }

    measurement = evaluator.select_known_drop_action(
        model, batch, method="measurement", **common
    )
    oracle = evaluator.select_known_drop_action(
        model, batch, method="oracle", **common
    )
    held = evaluator.select_known_drop_action(
        model, batch, method="hold_last", **common
    )

    assert not torch.equal(measurement.action, measurement.clean_action)
    assert torch.equal(oracle.action, oracle.clean_action)
    assert torch.allclose(held.action, torch.full_like(held.action, 0.25))


@pytest.mark.parametrize("method", evaluator.KNOWN_DROP_METHODS)
def test_inactive_decision_always_uses_native_clean_policy(method: str) -> None:
    result = evaluator.select_known_drop_action(
        _FakePolicy(),
        _batch(),
        method=method,
        active=False,
        last_valid_tokens=None,
        occlusion_radius_px=1,
        occlusion_value=0.0,
        device=torch.device("cpu"),
    )

    assert torch.equal(result.action, result.clean_action)


def test_hold_last_fails_closed_without_predrop_state() -> None:
    with pytest.raises(RuntimeError, match="valid pre-drop"):
        evaluator.select_known_drop_action(
            _FakePolicy(),
            _batch(),
            method="hold_last",
            active=True,
            last_valid_tokens=None,
            occlusion_radius_px=1,
            occlusion_value=0.0,
            device=torch.device("cpu"),
        )


class _FakeEnv:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            cam_fovy=np.asarray([45.0]),
            cam_pos=np.zeros((1, 3), dtype=np.float64),
            cam_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        )
        self.data = SimpleNamespace()
        self.action_space = SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )
        self.step_count = 0
        self.closed = False

    def reset(self, *, seed: int):
        self.step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        return (
            np.full(4, self.step_count, dtype=np.float32),
            float(np.mean(action)),
            False,
            False,
            {"success": 0.0},
        )

    def close(self) -> None:
        self.closed = True


def test_rollout_smoke_observes_half_open_drop_and_immediate_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environments: list[_FakeEnv] = []

    def make_env(*args: object, **kwargs: object) -> _FakeEnv:
        env = _FakeEnv()
        environments.append(env)
        return env

    def collect(
        env: _FakeEnv,
        obs: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        level = np.float32(0.20 + 0.01 * env.step_count)
        return {
            "rgb": np.full((3, 8, 8), level, dtype=np.float32),
            "obs": np.asarray(obs, dtype=np.float32),
            "keypoints_xyz": np.zeros((2, 3), dtype=np.float32),
            "q_true": np.asarray([[2.0, 2.0], [5.0, 5.0]], dtype=np.float32),
            "q_nominal": np.asarray([[2.0, 2.0], [5.0, 5.0]], dtype=np.float32),
        }

    monkeypatch.setattr(evaluator, "make_env", make_env)
    monkeypatch.setattr(evaluator, "get_camera_id", lambda *args: 0)
    monkeypatch.setattr(
        evaluator,
        "camera_intrinsic_from_fovy",
        lambda *args: np.eye(3, dtype=np.float64),
    )
    monkeypatch.setattr(
        evaluator,
        "camera_extrinsic_cv_from_mujoco",
        lambda *args: np.eye(4, dtype=np.float64),
    )
    monkeypatch.setattr(evaluator, "set_fixed_camera_pose_offset", lambda *args: None)
    monkeypatch.setattr(evaluator, "collect_observation", collect)

    common = {
        "model": _FakePolicy(),
        "checkpoint_args": {"obs_mode": "robot", "obs_horizon": 2},
        "task_name": "fake-task",
        "camera_name": "corner",
        "width": 8,
        "height": 8,
        "horizon": 15,
        "camera_offset": (
            np.zeros(3, dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            0.0,
        ),
        "device": torch.device("cpu"),
        "task_index": 7,
        "env_seed": 13,
        "drop_start": 1,
        "drop_duration": 1,
        "occlusion_radius_px": 1,
        "occlusion_value": 0.0,
    }
    measurement = evaluator.rollout_once(method="measurement", **common)
    oracle = evaluator.rollout_once(method="oracle", **common)

    assert measurement["active_decisions_observed"] == 1
    assert measurement["first_active_decision"] == 1
    assert measurement["last_active_decision"] == 1
    assert measurement["action_distance"] > oracle["action_distance"]
    assert oracle["action_distance"] == pytest.approx(0.0)
    assert all(env.closed for env in environments)


def test_checkpoint_seed_requires_integer_or_explicit_override() -> None:
    assert evaluator._checkpoint_seed({"seed": 17}, None) == 17
    assert evaluator._checkpoint_seed({"seed": "17"}, 23) == 23
    with pytest.raises(ValueError, match="integer seed"):
        evaluator._checkpoint_seed({"seed": "17"}, None)
