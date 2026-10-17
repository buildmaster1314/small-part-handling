from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.lewm_action_branch_a3_physical import (
    replay_physical_action_branches,
    validate_reconstructed_source_actions,
    validate_a3_physical_arrays,
)
from fcpg.lewm_action_branch_collection import EpisodeRuntime


class _PhysicalEnv:
    def __init__(self) -> None:
        self.action_space = SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )
        self.tcp_center = np.zeros(3, dtype=np.float64)
        self._object = np.zeros(3, dtype=np.float64)

    def _get_pos_objects(self) -> np.ndarray:
        return self._object

    def step(self, action: np.ndarray):
        self.tcp_center += np.asarray(action[:3], dtype=np.float64) * 0.01
        self._object += np.asarray(action[:3], dtype=np.float64) * 0.002
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


def test_physical_replay_restores_each_branch_and_records_all_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _PhysicalEnv()
    runtime = EpisodeRuntime(env=env, expert=None, observation=np.zeros(4), record_current=None)
    restores: list[int] = []

    def restore(fake_env: _PhysicalEnv, snapshot: tuple[np.ndarray, np.ndarray]) -> None:
        restores.append(1)
        fake_env.tcp_center = snapshot[0].copy()
        fake_env._object = snapshot[1].copy()

    monkeypatch.setattr(
        "fcpg.lewm_action_branch_a3_physical.restore_branch_snapshot", restore
    )
    actions = np.zeros((8, 40, 4), dtype=np.float32)
    for branch in range(8):
        actions[branch, :, 0] = 0.1 * (branch + 1)
    tcp, object_position = replay_physical_action_branches(
        runtime,
        (np.zeros(3), np.zeros(3)),
        actions,
    )

    assert len(restores) == 8
    assert tcp.shape == object_position.shape == (8, 8, 3)
    np.testing.assert_allclose(tcp[0, :, 0], np.arange(1, 9) * 0.005)
    np.testing.assert_allclose(tcp[7, :, 0], np.arange(1, 9) * 0.04)
    np.testing.assert_allclose(object_position, tcp * 0.2)


def _physical_arrays() -> dict[str, np.ndarray]:
    return {
        "task_index": np.asarray([0, 1], dtype=np.int64),
        "task_instance_index": np.asarray([2, 3], dtype=np.int64),
        "source_episode_id": np.asarray(["pick-place-v3:a", "drawer-open-v3:b"]),
        "hazard_id": np.asarray([4, 5], dtype=np.int64),
        "split": np.asarray(["train", "development"]),
        "source_tcp": np.zeros((2, 3), dtype=np.float32),
        "source_object_position": np.zeros((2, 3), dtype=np.float32),
        "tcp": np.zeros((2, 8, 8, 3), dtype=np.float32),
        "object_position": np.zeros((2, 8, 8, 3), dtype=np.float32),
    }


def test_physical_identity_must_match_latent_rows_exactly() -> None:
    physical = _physical_arrays()
    latent = {
        name: value.copy()
        for name, value in physical.items()
        if name
        in {
            "task_index",
            "task_instance_index",
            "source_episode_id",
            "hazard_id",
            "split",
        }
    }
    validate_a3_physical_arrays(physical, latent_arrays=latent)

    latent["hazard_id"] = latent["hazard_id"][::-1]
    with pytest.raises(ValueError, match="hazard_id"):
        validate_a3_physical_arrays(physical, latent_arrays=latent)


def test_physical_arrays_reject_wrong_endpoint_shape() -> None:
    physical = _physical_arrays()
    physical["tcp"] = np.zeros((2, 8, 7, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="tcp"):
        validate_a3_physical_arrays(physical)


def test_reconstructed_source_actions_must_match_cached_history_and_center() -> None:
    source = np.arange(70 * 4, dtype=np.float32).reshape(70, 4) / 1000.0
    validate_reconstructed_source_actions(
        history_action=source[5:15].reshape(2, 20),
        center_action=source[15:55].reshape(8, 20),
        hazard_step=3,
        source_actions=source,
    )

    wrong = source[15:55].copy()
    wrong[0, 0] += 1.0
    with pytest.raises(ValueError, match="center actions"):
        validate_reconstructed_source_actions(
            history_action=source[5:15].reshape(2, 20),
            center_action=wrong.reshape(8, 20),
            hazard_step=3,
            source_actions=source,
        )
