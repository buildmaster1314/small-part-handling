from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fcpg.visual_trust_corruptions import CORRUPTION_TYPES
from scripts import build_visual_trust_corruption_dataset as builder


def _source_arrays(
    *,
    include_camera_only: bool = True,
    frame_count: int = 5,
    task_names: tuple[str, ...] = ("assembly-v3", "button-press-v3"),
) -> dict[str, np.ndarray]:
    rows = [(task, frame) for frame in range(frame_count) for task in task_names]
    sample_count = len(rows)
    height, width, anchor_count = 11, 13, 3
    image_rng = np.random.default_rng(901)
    clean = image_rng.integers(
        0, 256, size=(sample_count, height, width, 3), dtype=np.uint8
    )
    q_true = np.empty((sample_count, anchor_count, 2), dtype=np.float32)
    q_true[:] = np.array(((2.0, 2.0), (6.0, 5.0), (10.0, 8.0)))
    valid = np.zeros((sample_count, anchor_count), dtype=bool)
    valid[:, 1] = True
    task_name = np.asarray([task for task, _ in rows])
    frame_id = np.asarray([frame for _, frame in rows], dtype=np.int32)
    task_offsets = {task: index for index, task in enumerate(task_names)}

    arrays: dict[str, np.ndarray] = {
        "rgb": np.zeros_like(clean),
        "obs": np.arange(sample_count * 4, dtype=np.float32).reshape(
            sample_count, 4
        ),
        "q_true": q_true,
        "q_nominal": q_true + np.float32(0.25),
        "keypoints_xyz": np.zeros(
            (sample_count, anchor_count, 3), dtype=np.float32
        ),
        "valid_true": valid,
        "valid_nominal": valid.copy(),
        "task_name": task_name,
        "task_index": np.asarray(
            [task_offsets[task] for task, _ in rows], dtype=np.int32
        ),
        "episode_id": np.zeros(sample_count, dtype=np.int32),
        "env_seed": np.asarray(
            [100 + task_offsets[task] for task, _ in rows], dtype=np.int64
        ),
        "frame_id": frame_id,
        "episode_recorded_frame_count": np.full(
            sample_count, frame_count, dtype=np.int32
        ),
        "action": np.ones((sample_count, 2), dtype=np.float32),
    }
    if include_camera_only:
        arrays["rgb_camera_only"] = clean
    else:
        arrays["rgb"] = clean
    return arrays


def _episode_rows(arrays: dict[str, np.ndarray], task: str) -> np.ndarray:
    rows = np.flatnonzero(arrays["task_name"].astype(str) == task)
    return rows[np.argsort(arrays["frame_id"][rows])]


def test_builds_one_deterministic_local_event_per_episode_without_mutation() -> None:
    source = _source_arrays()
    snapshots = {name: value.copy() for name, value in source.items()}

    first = builder.build_visual_trust_arrays(
        source, seed=73, minimum_duration=2, maximum_duration=2
    )
    second = builder.build_visual_trust_arrays(
        source, seed=73, minimum_duration=2, maximum_duration=2
    )

    expected_keys = {
        "rgb",
        "rgb_camera_only",
        *builder.CACHE_REQUIRED_KEYS,
        *builder.EPISODE_METADATA_KEYS,
        *builder.ANNOTATION_KEYS,
    }
    assert set(first) == expected_keys
    assert "action" not in first
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    for name in source:
        np.testing.assert_array_equal(source[name], snapshots[name])
    np.testing.assert_array_equal(first["rgb_camera_only"], source["rgb_camera_only"])

    assert first["corruption_active"].shape == (10, 3)
    assert first["corruption_active"].dtype == np.bool_
    assert first["corruption_family"].shape == (10,)
    assert first["corruption_family"].dtype.kind == "U"
    assert first["corruption_target_anchor"].shape == (10,)
    assert first["corruption_target_anchor"].dtype == np.int32
    assert first["corruption_recovery_onset"].shape == (10, 3)
    assert first["frames_since_recovery"].shape == (10, 3)
    assert first["frames_since_recovery"].dtype == np.int32

    for task in ("assembly-v3", "button-press-v3"):
        rows = _episode_rows(first, task)
        episode_active = first["corruption_active"][rows]
        active_positions = np.flatnonzero(np.any(episode_active, axis=1))
        np.testing.assert_array_equal(
            active_positions, np.arange(active_positions[0], active_positions[0] + 2)
        )
        assert active_positions[-1] < len(rows) - 1
        assert np.all(episode_active[active_positions, 1])
        assert int(episode_active.sum()) == 2

        active_rows = rows[active_positions]
        inactive_rows = np.setdiff1d(rows, active_rows)
        assert set(first["corruption_family"][active_rows]) <= set(CORRUPTION_TYPES)
        assert len(set(first["corruption_family"][active_rows])) == 1
        assert np.all(first["corruption_family"][inactive_rows] == "")
        assert np.all(first["corruption_target_anchor"][active_rows] == 1)
        assert np.all(first["corruption_target_anchor"][inactive_rows] == -1)

        recovery_position = int(active_positions[-1]) + 1
        recovery_row = int(rows[recovery_position])
        expected_onset = np.zeros((len(rows), 3), dtype=bool)
        expected_onset[recovery_position, 1] = True
        np.testing.assert_array_equal(
            first["corruption_recovery_onset"][rows], expected_onset
        )
        expected_since = np.full((len(rows), 3), -1, dtype=np.int32)
        expected_since[recovery_position:, 1] = np.arange(
            len(rows) - recovery_position, dtype=np.int32
        )
        np.testing.assert_array_equal(
            first["frames_since_recovery"][rows], expected_since
        )

        np.testing.assert_array_equal(
            first["rgb"][inactive_rows], first["rgb_camera_only"][inactive_rows]
        )
        changed = np.any(
            first["rgb"][active_rows] != first["rgb_camera_only"][active_rows],
            axis=(1, 2, 3),
        )
        assert bool(np.all(changed))
        np.testing.assert_array_equal(
            first["rgb"][recovery_row], first["rgb_camera_only"][recovery_row]
        )


def test_cli_falls_back_to_rgb_and_writes_cache_compatible_minimal_npz(
    tmp_path: Path,
) -> None:
    source = _source_arrays(
        include_camera_only=False,
        frame_count=4,
        task_names=("pick-place-v3",),
    )
    input_path = tmp_path / "source.npz"
    output_path = tmp_path / "derived" / "visual_trust.npz"
    np.savez_compressed(input_path, **source)
    input_bytes = input_path.read_bytes()

    assert (
        builder.main(
            [
                "--data",
                str(input_path),
                "--output",
                str(output_path),
                "--seed",
                "19",
                "--min-duration",
                "1",
                "--max-duration",
                "1",
                "--minimum-recovery-frames",
                "2",
            ]
        )
        == 0
    )
    assert input_path.read_bytes() == input_bytes

    with np.load(output_path, allow_pickle=False) as derived:
        np.testing.assert_array_equal(derived["rgb_camera_only"], source["rgb"])
        active_rows = np.flatnonzero(np.any(derived["corruption_active"], axis=1))
        assert active_rows.size == 1
        recovery_rows = np.flatnonzero(
            np.any(derived["corruption_recovery_onset"], axis=1)
        )
        np.testing.assert_array_equal(recovery_rows, active_rows + 1)
        np.testing.assert_array_equal(
            derived["rgb"][recovery_rows],
            derived["rgb_camera_only"][recovery_rows],
        )
        for name in builder.CACHE_REQUIRED_KEYS:
            assert name in derived.files


@pytest.mark.parametrize(
    ("update", "kwargs", "error"),
    [
        ({"valid_true": None}, {}, "missing required arrays"),
        ({}, {"minimum_duration": 0}, "minimum_duration must be positive"),
        (
            {},
            {"minimum_duration": 3, "maximum_duration": 2},
            "maximum_duration must be at least",
        ),
        (
            {},
            {"minimum_recovery_frames": 0},
            "minimum_recovery_frames must be positive",
        ),
        (
            {"frame_id": np.array([0, 0, 1, 1, 3, 3, 4, 4, 5, 5])},
            {},
            "contiguous frame IDs",
        ),
    ],
)
def test_rejects_invalid_source_and_schedule_bounds(
    update: dict[str, np.ndarray | None],
    kwargs: dict[str, int],
    error: str,
) -> None:
    source = _source_arrays()
    for name, value in update.items():
        if value is None:
            source.pop(name)
        else:
            source[name] = value

    with pytest.raises((KeyError, TypeError, ValueError), match=error):
        builder.build_visual_trust_arrays(source, **kwargs)


def test_rejects_episodes_without_recovery_or_valid_in_bounds_anchor() -> None:
    one_frame = _source_arrays(frame_count=1, task_names=("reach-v3",))
    with pytest.raises(ValueError, match="clean recovery"):
        builder.build_visual_trust_arrays(one_frame)

    no_anchor = _source_arrays(task_names=("reach-v3",))
    no_anchor["valid_true"][:] = False
    with pytest.raises(ValueError, match="no valid in-bounds anchor window"):
        builder.build_visual_trust_arrays(no_anchor)

    out_of_bounds = _source_arrays(task_names=("reach-v3",))
    out_of_bounds["q_true"][:, 1, 0] = out_of_bounds["rgb_camera_only"].shape[2]
    with pytest.raises(ValueError, match="no valid in-bounds anchor window"):
        builder.build_visual_trust_arrays(out_of_bounds)


def test_event_reserves_requested_number_of_clean_recovery_frames() -> None:
    source = _source_arrays(frame_count=12, task_names=("reach-v3",))
    result = builder.build_visual_trust_arrays(
        source,
        seed=11,
        minimum_duration=3,
        maximum_duration=3,
        minimum_recovery_frames=8,
    )
    active = np.flatnonzero(np.any(result["corruption_active"], axis=1))
    assert active.size == 3
    assert result["rgb"].shape[0] - int(active[-1]) - 1 >= 8


def test_rejects_overwriting_the_input_npz(tmp_path: Path) -> None:
    path = tmp_path / "source.npz"
    np.savez(path, **_source_arrays(task_names=("reach-v3",)))
    original = path.read_bytes()

    with pytest.raises(ValueError, match="input and output paths must differ"):
        builder.build_dataset(path, path)

    assert path.read_bytes() == original
