from __future__ import annotations

import numpy as np
import pytest

from scripts.prepare_towel_policy_dataset import (
    active_control_indices,
    first_stable_success,
    prepare_arrays,
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


def test_first_stable_success_ignores_isolated_success() -> None:
    success = np.asarray([0, 1, 0, 1, 1, 1, 1], dtype=np.float32)
    assert first_stable_success(success, 3) == 3
    assert first_stable_success(success, 5) is None


def test_active_control_indices_keep_stable_run_and_tail() -> None:
    data = _dataset()
    data["success"] = np.asarray([0, 1, 1, 0, 1, 1], dtype=np.float32)
    selected, counts = active_control_indices(
        data,
        stable_success_frames=2,
        post_success_tail=0,
    )
    assert selected.tolist() == [3, 4, 5, 0, 1, 2]
    assert counts == {0: 3, 1: 3}


def test_prepare_arrays_slices_frame_fields_and_updates_counts() -> None:
    data = _dataset()
    data["success"] = np.asarray([0, 1, 1, 0, 1, 1], dtype=np.float32)
    data["episode_recorded_frame_count"] = np.full(6, 3, dtype=np.int32)
    data["scalar_metadata"] = np.asarray("kept")
    prepared = prepare_arrays(
        data,
        stable_success_frames=2,
        post_success_tail=0,
    )
    assert prepared["rgb"].shape[0] == 6
    assert prepared["episode_recorded_frame_count"].tolist() == [3] * 6
    assert prepared["scalar_metadata"].item() == "kept"
    assert prepared["dataset_type"].item() == "towel_policy_active_control_v1"


def test_missing_stable_success_is_rejected() -> None:
    data = _dataset()
    with pytest.raises(ValueError, match="never reaches"):
        active_control_indices(
            data,
            stable_success_frames=2,
            post_success_tail=0,
        )
