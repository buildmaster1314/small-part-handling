from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_lewm_a4b_trajectory_coverage",
    ROOT / "scripts" / "audit_lewm_a4b_trajectory_coverage.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_raw_dataset_loader_deduplicates_multiview_rows(tmp_path: Path) -> None:
    index = tmp_path / "index.npz"
    np.savez(
        index,
        task_name=np.asarray(["pick-place-v3", "pick-place-v3", "push-v3"]),
        task_instance=np.asarray([1, 1, 2], dtype=np.int64),
        environment_seed=np.asarray([620001, 620001, 620002], dtype=np.int64),
        proposal_raw_step=np.asarray([10, 10, 15], dtype=np.int64),
        source_episode_id=np.asarray(["pick:e1", "pick:e1", "push:e2"]),
        split=np.asarray(["train", "train", "development"]),
    )

    episodes = MODULE.load_raw_dataset_episodes(tmp_path)

    assert episodes["pick:e1"]["proposal_steps"] == [10]
    assert episodes["push:e2"]["proposal_steps"] == [15]
    assert episodes["push:e2"]["split"] == "development"


def test_coverage_summary_uses_bundle_level_rows() -> None:
    rows = [
        {
            "task": "pick-place-v3",
            "episode_id": "e0",
            "proposal_step": 10,
            "first_success_step": 40,
            "trajectory_end_step": 42,
            "active_at_proposal": True,
            "post_success": False,
            "h8_reaches_first_success": True,
            "h8_reaches_stable_success": True,
        },
        {
            "task": "pick-place-v3",
            "episode_id": "e1",
            "proposal_step": 10,
            "first_success_step": 80,
            "trajectory_end_step": 82,
            "active_at_proposal": True,
            "post_success": False,
            "h8_reaches_first_success": False,
            "h8_reaches_stable_success": False,
        },
    ]

    summary = MODULE.summarize(rows)["pick-place-v3"]

    assert summary["bundle_count"] == 2
    assert summary["episode_count"] == 2
    assert summary["h8_reaches_first_success_fraction"] == 0.5
    assert summary["h8_reaches_stable_success_fraction"] == 0.5


def test_replay_matches_collector_float32_observations(monkeypatch) -> None:
    observed_dtypes: list[np.dtype] = []

    class Expert:
        def get_action(self, observation):
            observed_dtypes.append(np.asarray(observation).dtype)
            return np.zeros(4, dtype=np.float32)

    class ActionSpace:
        low = np.full(4, -1.0, dtype=np.float32)
        high = np.full(4, 1.0, dtype=np.float32)

    class Env:
        max_path_length = 4
        action_space = ActionSpace()

        def reset(self, seed):
            del seed
            return np.zeros(39, dtype=np.float64), {}

        def step(self, action):
            del action
            return np.zeros(39, dtype=np.float64), 0.0, False, False, {
                "success": 1.0
            }

        def close(self):
            return None

    monkeypatch.setattr(MODULE, "make_a4_expert", lambda task: Expert())
    result = MODULE.replay_expert_episode(
        {
            "task": "pick-place-v3",
            "task_instance": 0,
            "environment_seed": 1,
        },
        env_builder=lambda *args: Env(),
    )

    assert result["stable_success_step"] == 3
    assert observed_dtypes == [np.dtype(np.float32)] * 3
