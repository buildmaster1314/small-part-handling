from __future__ import annotations

import numpy as np
import pytest

from scripts.render_towel_policy_dataset import (
    count_wipe_runs,
    episode_indices,
    validate_dataset,
)


def _dataset() -> dict[str, np.ndarray]:
    count = 6
    return {
        "rgb": np.zeros((count, 8, 8, 3), dtype=np.uint8),
        "obs": np.zeros((count, 39), dtype=np.float32),
        "action": np.zeros((count, 4), dtype=np.float32),
        "success": np.zeros(count, dtype=np.float32),
        "task_name": np.asarray(["towel-wipe-v0"] * count),
        "task_index": np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int32),
        "episode_id": np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int32),
        "env_seed": np.arange(count, dtype=np.int64),
        "frame_id": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32),
        "q_true": np.zeros((count, 3, 2), dtype=np.float32),
        "q_nominal": np.zeros((count, 3, 2), dtype=np.float32),
        "valid_true": np.ones((count, 3), dtype=bool),
        "valid_nominal": np.ones((count, 3), dtype=bool),
        "observation_action_alignment": np.asarray("pre_action"),
    }


def test_wipe_run_count_requires_alternating_phase_entries() -> None:
    phases = [
        "lower_to_start",
        "wipe_right",
        "wipe_right",
        "wipe_left",
        "wipe_right",
        "wipe_left",
        "wipe_right",
        "wipe_left",
        "lift_after_wipe",
    ]
    count, order = count_wipe_runs(phases)
    assert count == 6
    assert order == (
        "wipe_right",
        "wipe_left",
        "wipe_right",
        "wipe_left",
        "wipe_right",
        "wipe_left",
    )


def test_episode_groups_are_sorted_and_contiguous() -> None:
    data = _dataset()
    validate_dataset(data)
    groups = episode_indices(data)
    assert [group.tolist() for group in groups] == [[3, 4, 5], [0, 1, 2]]


def test_dataset_validation_rejects_alignment_and_anchor_shape() -> None:
    data = _dataset()
    data["observation_action_alignment"] = np.asarray("post_action")
    with pytest.raises(ValueError, match="pre_action"):
        validate_dataset(data)
    data = _dataset()
    data["q_true"] = np.zeros((6, 1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="three"):
        validate_dataset(data)
