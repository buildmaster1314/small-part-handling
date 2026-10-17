from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_action_branch_a1 import (
    ACTION_DIM,
    ANONYMOUS_BRANCH_NAMES,
    CANDIDATE_COUNT,
    EMBEDDING_DIM,
    HISTORY_STEPS,
    KEYPOINT_COUNT,
    MIN_PAIRWISE_MOTION_RMS,
    NPZ_KEYS,
    TARGET_MOTION_RMS,
    build_anonymous_action_branches,
    load_a1_npz,
    motion_delta_rms,
    reshape_action_branches,
    save_a1_npz,
    validate_a1_arrays,
    validate_action_branches,
)


def _arrays(*, sample_count: int = 4, horizon: int = 8) -> dict[str, np.ndarray]:
    task_index = np.arange(sample_count, dtype=np.int32) % 2
    source_episode_id = np.asarray(
        [f"task-{int(task)}-episode-{index // 2}" for index, task in enumerate(task_index)]
    )
    split = np.asarray(["train", "train", "development", "test"][:sample_count])
    return {
        "history_embedding": np.zeros(
            (sample_count, HISTORY_STEPS, KEYPOINT_COUNT, EMBEDDING_DIM),
            dtype=np.float32,
        ),
        "history_action": np.zeros(
            (sample_count, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float32
        ),
        "center_action": np.zeros(
            (sample_count, horizon, ACTION_DIM), dtype=np.float32
        ),
        "future_action": np.zeros(
            (sample_count, CANDIDATE_COUNT, horizon, ACTION_DIM), dtype=np.float32
        ),
        "target_embedding": np.zeros(
            (
                sample_count,
                CANDIDATE_COUNT,
                horizon,
                KEYPOINT_COUNT,
                EMBEDDING_DIM,
            ),
            dtype=np.float32,
        ),
        "task_index": task_index,
        "task_instance_index": np.arange(sample_count, dtype=np.int16),
        "source_episode_id": source_episode_id,
        "hazard_id": np.arange(sample_count, dtype=np.int16),
        "split": split,
    }


def _scripted_center(horizon: int = 8, *, near_bounds: bool = False) -> np.ndarray:
    rng = np.random.default_rng(1701 + horizon)
    limit = 0.82 if near_bounds else 0.45
    center = rng.uniform(
        -limit, limit, size=(horizon * 5, 4)
    ).astype(np.float32)
    center[:, 3] = np.where(np.arange(center.shape[0]) % 9 < 4, -1.0, 1.0)
    return center


def _branch_payloads(branches: np.ndarray) -> list[bytes]:
    return sorted(np.ascontiguousarray(branch).tobytes() for branch in branches)


def test_a1_array_contract_and_atomic_npz_round_trip(tmp_path: Path) -> None:
    arrays = _arrays(horizon=9)
    axes = validate_a1_arrays(arrays)

    assert axes.sample_count == 4
    assert axes.history_steps == 3
    assert axes.candidate_count == 8
    assert axes.horizon == 9
    assert axes.action_dim == 20
    assert axes.keypoint_count == 3
    assert axes.embedding_dim == 192

    path = tmp_path / "a1.npz"
    assert save_a1_npz(path, arrays) == path
    assert {item.name for item in tmp_path.iterdir()} == {"a1.npz"}
    loaded = load_a1_npz(path)

    assert tuple(loaded) == NPZ_KEYS
    for name in NPZ_KEYS:
        np.testing.assert_array_equal(loaded[name], arrays[name])
        assert loaded[name].flags.writeable


def test_a1_contract_rejects_episode_split_leakage() -> None:
    arrays = _arrays()
    arrays["task_index"][:2] = 0
    arrays["source_episode_id"][:2] = "shared-physical-episode"
    arrays["split"][:2] = np.asarray(["train", "test"])

    with pytest.raises(ValueError, match="source physical episode crosses splits"):
        validate_a1_arrays(arrays)


def test_episode_ids_may_be_reused_by_different_tasks() -> None:
    arrays = _arrays(sample_count=2)
    arrays["task_index"] = np.asarray([0, 1], dtype=np.int32)
    arrays["task_instance_index"] = np.asarray([7, 7], dtype=np.int16)
    arrays["source_episode_id"] = np.asarray(["episode-7", "episode-7"])
    arrays["split"] = np.asarray(["train", "test"])

    assert validate_a1_arrays(arrays).sample_count == 2


def test_a1_contract_rejects_task_instance_split_leakage() -> None:
    arrays = _arrays(sample_count=2)
    arrays["task_index"] = np.asarray([0, 0], dtype=np.int32)
    arrays["task_instance_index"] = np.asarray([17, 17], dtype=np.int16)
    arrays["source_episode_id"] = np.asarray(["episode-a", "episode-b"])
    arrays["split"] = np.asarray(["train", "development"])

    with pytest.raises(ValueError, match="MetaWorld task instance crosses splits"):
        validate_a1_arrays(arrays)


@pytest.mark.parametrize("instance", [-1, 50])
def test_a1_contract_rejects_task_instance_outside_zero_to_49(
    instance: int,
) -> None:
    arrays = _arrays()
    arrays["task_instance_index"][0] = instance

    with pytest.raises(ValueError, match="0..49"):
        validate_a1_arrays(arrays)


@pytest.mark.parametrize(
    "name",
    [
        "history_embedding",
        "history_action",
        "center_action",
        "future_action",
        "target_embedding",
    ],
)
def test_a1_contract_rejects_nonfinite_numeric_arrays(name: str) -> None:
    arrays = _arrays()
    arrays[name].reshape(-1)[0] = np.nan

    with pytest.raises(ValueError, match="finite numeric"):
        validate_a1_arrays(arrays)


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        (
            "history_embedding",
            np.zeros((4, 2, KEYPOINT_COUNT, EMBEDDING_DIM), dtype=np.float32),
        ),
        ("history_action", np.zeros((4, 3, ACTION_DIM), dtype=np.float32)),
        ("center_action", np.zeros((4, 7, ACTION_DIM), dtype=np.float32)),
        ("future_action", np.zeros((4, 7, 8, ACTION_DIM), dtype=np.float32)),
        (
            "target_embedding",
            np.zeros((4, 8, 8, 2, EMBEDDING_DIM), dtype=np.float32),
        ),
        ("task_index", np.zeros((4, 1), dtype=np.int32)),
        ("task_instance_index", np.zeros((4, 1), dtype=np.int16)),
    ],
)
def test_a1_contract_rejects_bad_shapes(
    name: str, replacement: np.ndarray
) -> None:
    arrays = _arrays()
    arrays[name] = replacement

    with pytest.raises(ValueError, match="shape"):
        validate_a1_arrays(arrays)


def test_a1_contract_rejects_horizon_below_eight() -> None:
    arrays = _arrays(horizon=7)

    with pytest.raises(ValueError, match="P>=8"):
        validate_a1_arrays(arrays)


@pytest.mark.parametrize("horizon", [8, 11])
def test_action_branches_are_deterministic_equal_rms_and_reshapable(
    horizon: int,
) -> None:
    center = _scripted_center(horizon)
    left = build_anonymous_action_branches(
        center,
        direction_seed=103,
        permutation_seed=211,
        horizon=horizon,
    )
    right = build_anonymous_action_branches(
        center,
        direction_seed=103,
        permutation_seed=211,
        horizon=horizon,
    )

    assert left.shape == (8, horizon * 5, 4)
    assert left.dtype == np.float32
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(left, np.clip(left, -1.0, 1.0))
    np.testing.assert_array_equal(
        left[:, :, 3], np.broadcast_to(center[None, :, 3], (8, horizon * 5))
    )
    np.testing.assert_allclose(
        motion_delta_rms(left, center),
        TARGET_MOTION_RMS,
        rtol=0.0,
        atol=2.0e-6,
    )

    blocks = reshape_action_branches(left, horizon=horizon)
    assert blocks.shape == (8, horizon, 20)
    np.testing.assert_array_equal(blocks.reshape(8, horizon * 5, 4), left)


def test_branch_slots_are_anonymous_and_independently_permuted() -> None:
    center = np.zeros((40, 4), dtype=np.float32)
    center[:, 3] = np.where(np.arange(40) < 20, -1.0, 1.0)
    left = build_anonymous_action_branches(
        center, direction_seed=73, permutation_seed=101
    )
    permuted = build_anonymous_action_branches(
        center, direction_seed=73, permutation_seed=102
    )

    assert ANONYMOUS_BRANCH_NAMES == tuple(f"candidate_{index}" for index in range(8))
    assert not np.array_equal(left, permuted)
    assert _branch_payloads(left) == _branch_payloads(permuted)

    delta = left[:, :, :3] - center[None, :, :3]
    for index in range(CANDIDATE_COUNT):
        assert any(
            other != index
            and np.allclose(delta[other], -delta[index], rtol=0.0, atol=1.0e-7)
            for other in range(CANDIDATE_COUNT)
        )


def test_clipping_preserves_bounds_radius_and_branch_diversity() -> None:
    center = _scripted_center(8, near_bounds=True)
    branches = build_anonymous_action_branches(
        center,
        direction_seed=307,
        permutation_seed=401,
        action_low=-1.0,
        action_high=1.0,
    )
    validation = validate_action_branches(branches, center)

    assert np.all(branches >= -1.0)
    assert np.all(branches <= 1.0)
    assert validation.horizon == 8
    np.testing.assert_allclose(
        validation.realized_motion_rms,
        TARGET_MOTION_RMS,
        rtol=0.0,
        atol=2.0e-6,
    )
    assert validation.minimum_pairwise_motion_rms >= MIN_PAIRWISE_MOTION_RMS
    assert len(_branch_payloads(branches)) == len(set(_branch_payloads(branches))) == 8


@pytest.mark.parametrize(
    ("center", "horizon", "message"),
    [
        (np.zeros((40, 3), dtype=np.float32), 8, "shape"),
        (np.zeros((32, 4), dtype=np.float32), 8, "P\\*5"),
        (np.zeros((35, 4), dtype=np.float32), 7, "P>=8"),
        (np.zeros((40, 4), dtype=np.float32), 9, "exactly horizon\\*5"),
    ],
)
def test_generator_rejects_incomplete_or_mismatched_scripted_centers(
    center: np.ndarray, horizon: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_anonymous_action_branches(
            center,
            direction_seed=1,
            permutation_seed=2,
            horizon=horizon,
        )


def test_generator_rejects_nan_without_filling_the_center() -> None:
    center = np.zeros((40, 4), dtype=np.float32)
    center[17, 2] = np.nan

    with pytest.raises(ValueError, match="finite numeric"):
        build_anonymous_action_branches(
            center, direction_seed=1, permutation_seed=2, horizon=8
        )
