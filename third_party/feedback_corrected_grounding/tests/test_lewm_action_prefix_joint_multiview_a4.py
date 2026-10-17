from __future__ import annotations

import numpy as np
import torch

from fcpg.lewm_action_branch_a4 import TASKS
from fcpg.lewm_action_prefix_joint_a4 import (
    gripper_aperture_loss,
    task_balanced_sample,
)
from fcpg.lewm_action_prefix_joint_data import validate_replay_index
from scripts.collect_lewm_action_prefix_joint_a4 import (
    MULTIVIEW_CAMERA_NAMES,
    augment_gripper_action_branches,
    expand_replay_index_orbit_views,
    expand_replay_index_views,
)
from scripts.train_lewm_action_prefix_joint_a4 import _move


def _base_index() -> dict[str, np.ndarray]:
    count = len(TASKS)
    return {
        "cache_index": np.arange(count, dtype=np.int64),
        "proposal_id": np.asarray([f"p{index}" for index in range(count)]),
        "task_index": np.arange(count, dtype=np.int64),
        "task_name": np.asarray(TASKS),
        "task_instance": np.zeros(count, dtype=np.int64),
        "environment_seed": np.arange(count, dtype=np.int64) + 10,
        "proposal_raw_step": np.arange(count, dtype=np.int64),
        "source_episode_id": np.asarray([f"e{index}" for index in range(count)]),
        "split": np.asarray(["train"] * count),
        "future_action": np.zeros((count, 8, 8, 20), dtype=np.float32),
    }


def test_expand_replay_index_views_repeats_physics_and_labels_camera() -> None:
    expanded = expand_replay_index_views(_base_index(), MULTIVIEW_CAMERA_NAMES)

    assert validate_replay_index(expanded) == len(TASKS) * len(
        MULTIVIEW_CAMERA_NAMES
    )
    assert expanded["camera_name"][:5].tolist() == list(MULTIVIEW_CAMERA_NAMES)
    assert expanded["physical_bundle_index"][:5].tolist() == [0] * 5
    assert expanded["task_name"][:5].tolist() == [TASKS[0]] * 5


def test_expand_replay_index_orbit_views_holds_out_odd_clock_hours() -> None:
    base = _base_index()
    base["split"] = np.asarray(
        ["development" if index == 1 else "train" for index in range(len(TASKS))]
    )

    expanded = expand_replay_index_orbit_views(base)

    assert validate_replay_index(expanded) == len(TASKS) * 6
    first = np.flatnonzero(expanded["physical_bundle_index"] == 0)
    second = np.flatnonzero(expanded["physical_bundle_index"] == 1)
    assert expanded["camera_clock_hour"][first].tolist() == [12, 2, 4, 6, 8, 10]
    assert expanded["camera_clock_hour"][second].tolist() == [1, 3, 5, 7, 9, 11]
    assert set(expanded["camera_name"][first].tolist()) == {
        "clock12_e35",
        "clock02_e35",
        "clock04_e35",
        "clock06_e35",
        "clock08_e35",
        "clock10_e35",
    }
    drawer_rows = np.flatnonzero(expanded["task_name"] == "drawer-retrieve-v0")
    towel_rows = np.flatnonzero(expanded["task_name"] == "towel-wipe-v0")
    assert set(expanded["camera_base_name"][drawer_rows].tolist()) == {"drawer_view"}
    assert set(expanded["camera_base_name"][towel_rows].tolist()) == {"towel_view"}


def test_gripper_augmentation_preserves_xyz_and_adds_release_schedules() -> None:
    source = np.zeros((2, 8, 8, 20), dtype=np.float32)
    source[..., 0::4] = 0.25
    source[..., 1::4] = -0.5
    source[..., 2::4] = 0.75
    source[..., 3::4] = 0.5

    result = augment_gripper_action_branches(source)

    np.testing.assert_array_equal(result[..., 0::4], source[..., 0::4])
    np.testing.assert_array_equal(result[..., 1::4], source[..., 1::4])
    np.testing.assert_array_equal(result[..., 2::4], source[..., 2::4])
    assert np.all(result[:, 2, :, 3::4] == -1.0)
    assert np.all(result[:, 3, :, 3::4] == 1.0)
    assert np.all(result[:, 4, :, 3::4] == 0.0)
    assert np.all(result[:, 7, :4, 3::4] == 1.0)
    assert np.all(result[:, 7, 4:, 3::4] == -1.0)


def test_task_balanced_sampling_spreads_views_within_each_task() -> None:
    index = expand_replay_index_views(_base_index(), MULTIVIEW_CAMERA_NAMES)
    bundles, branches = task_balanced_sample(
        index,
        split="train",
        batch_size=len(TASKS) * len(MULTIVIEW_CAMERA_NAMES),
        rng=np.random.default_rng(7),
    )

    assert branches.shape == bundles.shape
    for task_index in range(len(TASKS)):
        selected = bundles[index["task_index"][bundles] == task_index]
        assert set(index["camera_index"][selected].tolist()) == set(range(5))


def test_orbit_batch_84_samples_each_train_view_twice_per_task() -> None:
    index = expand_replay_index_orbit_views(_base_index())
    bundles, branches = task_balanced_sample(
        index,
        split="train",
        batch_size=84,
        rng=np.random.default_rng(13),
    )

    assert branches.shape == bundles.shape == (84,)
    for task_index in range(len(TASKS)):
        selected = bundles[index["task_index"][bundles] == task_index]
        assert selected.shape == (12,)
        _, counts = np.unique(index["camera_index"][selected], return_counts=True)
        assert counts.tolist() == [2] * 6


def test_gripper_aperture_loss_uses_fingertip_separation() -> None:
    target = torch.zeros(1, 2, 3, 2)
    predicted = target.clone()
    target[:, :, 1, 0] = 20.0
    predicted[:, :, 1, 0] = 10.0
    valid = torch.ones(1, 2, 3, dtype=torch.bool)

    value = gripper_aperture_loss(
        predicted,
        target,
        valid,
        image_size=(101, 101),
    )

    torch.testing.assert_close(value, torch.tensor(0.01))


def test_gripper_aperture_loss_can_emphasize_late_horizons() -> None:
    target = torch.zeros(1, 2, 3, 2)
    predicted = target.clone()
    target[:, -1, 1, 0] = 20.0
    predicted[:, -1, 1, 0] = 10.0
    valid = torch.ones(1, 2, 3, dtype=torch.bool)

    uniform = gripper_aperture_loss(
        predicted,
        target,
        valid,
        image_size=(101, 101),
    )
    late = gripper_aperture_loss(
        predicted,
        target,
        valid,
        image_size=(101, 101),
        horizon_weights=torch.tensor([1.0, 3.0]),
    )

    torch.testing.assert_close(late, uniform * 1.5)


def test_joint_training_move_only_transfers_batch_tensors() -> None:
    batch = {
        "current_rgb": torch.zeros(2, 3, 4, 4),
        "task_index": torch.arange(2),
    }

    moved = _move(batch, torch.device("cpu"))

    assert set(moved) == set(batch)
    torch.testing.assert_close(moved["current_rgb"], batch["current_rgb"])
    torch.testing.assert_close(moved["task_index"], batch["task_index"])
