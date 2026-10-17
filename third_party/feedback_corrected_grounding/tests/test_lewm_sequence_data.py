from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_sequence_data import LeWMSequenceDataset, frame_identity_sha256


def _write_dataset(
    path: Path,
    episodes: list[tuple[str, int, list[int]]],
    *,
    omit: set[str] | None = None,
    alignment: str | None = "pre_action",
) -> dict[str, np.ndarray]:
    tasks: list[str] = []
    episode_ids: list[int] = []
    frame_ids: list[int] = []
    for task, episode_id, frames in episodes:
        tasks.extend([task] * len(frames))
        episode_ids.extend([episode_id] * len(frames))
        frame_ids.extend(frames)
    count = len(frame_ids)
    values = np.arange(count, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "rgb": np.broadcast_to(
            values.astype(np.uint8)[:, None, None, None], (count, 2, 3, 3)
        ).copy(),
        "obs": np.broadcast_to(values[:, None], (count, 6)).copy(),
        "action": np.arange(count * 4, dtype=np.float32).reshape(count, 4),
        "task_name": np.asarray(tasks),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "frame_id": np.asarray(frame_ids, dtype=np.int32),
        "keypoints_xyz": np.broadcast_to(values[:, None, None], (count, 3, 3)).copy(),
        "q_nominal": np.broadcast_to(values[:, None, None], (count, 3, 2)).copy(),
        "q_true": np.broadcast_to((values + 0.5)[:, None, None], (count, 3, 2)).copy(),
        "valid_true": np.ones((count, 3), dtype=bool),
        "valid_nominal": np.ones((count, 3), dtype=bool),
    }
    arrays["valid_nominal"][1::2, 0] = False
    if alignment is not None:
        arrays["observation_action_alignment"] = np.asarray(alignment)
    for field in omit or set():
        arrays.pop(field)
    np.savez(path, **arrays)
    return arrays


def test_windows_are_episode_local_unpadded_and_expose_grounding(tmp_path) -> None:
    path = tmp_path / "tasks.npz"
    arrays = _write_dataset(
        path,
        [
            ("pick-place-v3", 7, list(range(5))),
            ("button-press-v3", 7, list(range(5))),
        ],
    )
    dataset = LeWMSequenceDataset(
        path,
        history_size=2,
        rollout_steps=2,
        source_index=4,
        source_name="shared",
    )

    assert len(dataset) == 4
    first, other_task = dataset[0], dataset[2]
    assert first["task_name"] != other_task["task_name"]
    assert first["episode_id"] == other_task["episode_id"] == 7
    torch.testing.assert_close(first["frame_id"], torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(other_task["frame_id"], torch.tensor([0, 1, 2, 3]))
    assert first["pixels"].shape == (4, 3, 2, 3)
    assert first["rgb"].shape == (4, 2, 3, 3)
    assert first["actions"].shape == (3, 4)
    assert first["proprio"].shape == (4, 4)
    assert first["obs"].shape == (4, 6)
    assert first["q_nominal"].shape == (4, 3, 2)
    assert "q_true" not in first
    assert first["keypoints_xyz"].shape == (4, 3, 3)
    assert first["valid"].dtype == torch.bool
    assert first["source"] == "shared" and int(first["source_index"]) == 4
    torch.testing.assert_close(first["goal_pixels"], first["pixels"][-1])
    torch.testing.assert_close(first["goal_rgb"], first["rgb"][-1])
    assert int(first["goal_frame_id"]) == 3
    start = int(first["raw_index"][0])
    torch.testing.assert_close(
        first["actions"], torch.from_numpy(arrays["action"][start : start + 3])
    )


def test_frame_skip_concatenates_transition_action_blocks(tmp_path) -> None:
    path = tmp_path / "skip.npz"
    arrays = _write_dataset(path, [("reach-v3", 0, list(range(8)))])
    dataset = LeWMSequenceDataset(
        path, history_size=2, rollout_steps=2, frame_skip=2
    )

    assert dataset.raw_window_span == 7
    assert dataset.transition_action_dim == 8
    assert len(dataset) == 2
    sample = dataset[0]
    torch.testing.assert_close(sample["frame_id"], torch.tensor([0, 2, 4, 6]))
    expected = np.stack(
        [arrays["action"][start : start + 2].reshape(-1) for start in (0, 2, 4)]
    )
    torch.testing.assert_close(sample["actions"], torch.from_numpy(expected))


def test_optional_occlusion_annotations_follow_sequence_windows(tmp_path) -> None:
    path = tmp_path / "annotated.npz"
    arrays = _write_dataset(path, [("reach-v3", 0, list(range(6)))])
    dropout = np.zeros((6, 3), dtype=bool)
    dropout[2:4, 1] = True
    recovery_onset = np.zeros_like(dropout)
    recovery_onset[4, 1] = True
    recovery_segment = np.zeros_like(dropout)
    recovery_segment[4:, 1] = True
    arrays.update(
        {
            "anchor_dropout_active": dropout,
            "anchor_recovery_onset": recovery_onset,
            "anchor_recovery_segment": recovery_segment,
            "measurement_valid": arrays["valid_nominal"] & ~dropout,
        }
    )
    np.savez(path, **arrays)

    dataset = LeWMSequenceDataset(path, history_size=2, rollout_steps=2)
    sample = dataset[1]
    for field in (
        "anchor_dropout_active",
        "anchor_recovery_onset",
        "anchor_recovery_segment",
        "measurement_valid",
    ):
        assert sample[field].dtype == torch.bool
        assert sample[field].shape == (4, 3)
        torch.testing.assert_close(
            sample[field], torch.from_numpy(arrays[field][1:5])
        )


def test_optional_occlusion_annotations_must_match_anchor_shape(tmp_path) -> None:
    path = tmp_path / "bad_annotation.npz"
    arrays = _write_dataset(path, [("reach-v3", 0, list(range(4)))])
    arrays["anchor_dropout_active"] = np.zeros((4, 2), dtype=bool)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="anchor_dropout_active must have shape"):
        LeWMSequenceDataset(path, history_size=2, rollout_steps=2)


def test_short_episodes_are_omitted_instead_of_terminal_padded(tmp_path) -> None:
    path = tmp_path / "short.npz"
    _write_dataset(
        path,
        [("short-v3", 0, [0, 1, 2]), ("long-v3", 1, [0, 1, 2, 3])],
    )
    dataset = LeWMSequenceDataset(path, history_size=2, rollout_steps=2)

    assert len(dataset) == 1
    assert dataset[0]["task_name"] == "long-v3"
    torch.testing.assert_close(dataset[0]["frame_id"], torch.tensor([0, 1, 2, 3]))


def test_rejects_non_contiguous_missing_and_unaligned_data(tmp_path) -> None:
    gap_path = tmp_path / "gap.npz"
    _write_dataset(gap_path, [("reach-v3", 0, [0, 1, 3, 4])])
    with pytest.raises(ValueError, match="not strictly contiguous"):
        LeWMSequenceDataset(gap_path, history_size=2, rollout_steps=1)

    oracle_free_path = tmp_path / "oracle_free.npz"
    _write_dataset(
        oracle_free_path,
        [("reach-v3", 0, [0, 1, 2])],
        omit={"q_true", "valid_true"},
    )
    oracle_free = LeWMSequenceDataset(
        oracle_free_path, history_size=2, rollout_steps=1
    )
    assert "q_true" not in oracle_free[0]

    missing_path = tmp_path / "missing.npz"
    _write_dataset(
        missing_path, [("reach-v3", 0, [0, 1, 2])], omit={"q_nominal"}
    )
    with pytest.raises(KeyError, match="q_nominal"):
        LeWMSequenceDataset(missing_path, history_size=2, rollout_steps=1)

    alignment_path = tmp_path / "alignment.npz"
    _write_dataset(
        alignment_path,
        [("reach-v3", 0, [0, 1, 2])],
        alignment="post_action",
    )
    with pytest.raises(ValueError, match="pre_action"):
        LeWMSequenceDataset(alignment_path, history_size=2, rollout_steps=1)


def test_all_episodes_too_short_raise_without_padding(tmp_path) -> None:
    path = tmp_path / "all_short.npz"
    _write_dataset(path, [("reach-v3", 0, [0, 1, 2])])
    with pytest.raises(ValueError, match="no episode is long enough"):
        LeWMSequenceDataset(
            path, history_size=2, rollout_steps=2, frame_skip=2
        )


def test_correction_cache_directory_is_validated_and_indexed(tmp_path) -> None:
    path = tmp_path / "corrected.npz"
    arrays = _write_dataset(path, [("pick-place-v3", 0, list(range(5)))])
    corrected = arrays["q_true"].astype(np.float32) + 7.0
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "q_corrected.npy", corrected)
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(path.resolve()),
                "sample_count": 5,
                "num_keypoints": 3,
            }
        ),
        encoding="utf-8",
    )

    dataset = LeWMSequenceDataset(
        path,
        history_size=2,
        rollout_steps=2,
        correction_cache=cache,
    )
    sample = dataset[0]
    indices = sample["raw_index"].numpy()
    torch.testing.assert_close(
        sample["q_corrected"],
        torch.from_numpy(corrected[indices]),
    )
    assert "q_true" not in sample

    without_cache = LeWMSequenceDataset(path, history_size=2, rollout_steps=2)
    assert "q_corrected" not in without_cache[0]


def test_schema_v3_correction_cache_rejects_tampering(tmp_path) -> None:
    path = tmp_path / "sealed.npz"
    arrays = _write_dataset(path, [("pick-place-v3", 0, list(range(5)))])
    cache = tmp_path / "sealed_cache"
    cache.mkdir()
    corrected_path = cache / "q_corrected.npy"
    np.save(corrected_path, arrays["q_true"].astype(np.float32))
    checkpoint = tmp_path / "corrector.pt"
    checkpoint.write_bytes(b"frozen-corrector")
    metadata = {
        "schema_version": 3,
        "source_data": str(path.resolve()),
        "source_data_sha256": sha256_file(path),
        "sample_count": 5,
        "num_keypoints": 3,
        "frame_identity_sha256": frame_identity_sha256(
            arrays["episode_id"], arrays["frame_id"]
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "q_corrected_sha256": sha256_file(corrected_path),
        "q_true_usage": "offline_metrics_only",
    }
    metadata_path = cache / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    LeWMSequenceDataset(
        path,
        history_size=2,
        rollout_steps=2,
        correction_cache=cache,
    )

    corrected = np.load(corrected_path)
    corrected[0, 0, 0] += 1.0
    np.save(corrected_path, corrected)
    with pytest.raises(ValueError, match="correction array hash"):
        LeWMSequenceDataset(
            path,
            history_size=2,
            rollout_steps=2,
            correction_cache=cache,
        )


def test_validity_uses_deployable_geometry_not_valid_true(tmp_path) -> None:
    path = tmp_path / "validity.npz"
    arrays = _write_dataset(path, [("reach-v3", 0, list(range(4)))])
    arrays["valid_true"][:] = False
    arrays["valid_nominal"][:] = True
    arrays["q_nominal"][:] = np.asarray([1.0, 1.0], dtype=np.float32)
    np.savez(path, **arrays)

    corrected = arrays["q_nominal"].astype(np.float32, copy=True)
    corrected[0, 0] = np.asarray([-1.0, 1.0], dtype=np.float32)
    cache = tmp_path / "validity_corrected.npy"
    np.save(cache, corrected)
    dataset = LeWMSequenceDataset(
        path, history_size=2, rollout_steps=1, correction_cache=cache
    )
    valid = dataset[0]["valid"]

    assert not bool(valid[0, 0])
    assert bool(valid[0, 1])
    assert bool(valid[1:].all())


def test_direct_correction_array_and_invalid_cache_are_rejected(tmp_path) -> None:
    path = tmp_path / "direct.npz"
    arrays = _write_dataset(path, [("reach-v3", 0, list(range(4)))])
    direct = tmp_path / "corrected.npy"
    corrected = (arrays["q_true"] + 2.0).astype(np.float16)
    np.save(direct, corrected)
    dataset = LeWMSequenceDataset(
        path,
        history_size=2,
        rollout_steps=2,
        correction_cache=direct,
    )
    assert dataset[0]["q_corrected"].dtype == torch.float32

    bad_shape = tmp_path / "bad.npy"
    np.save(bad_shape, np.zeros((4, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="q_corrected must have shape"):
        LeWMSequenceDataset(
            path,
            history_size=2,
            rollout_steps=2,
            correction_cache=bad_shape,
        )

    wrong_cache = tmp_path / "wrong_cache"
    wrong_cache.mkdir()
    np.save(wrong_cache / "q_corrected.npy", corrected)
    (wrong_cache / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str((tmp_path / "other.npz").resolve()),
                "sample_count": 4,
                "num_keypoints": 3,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match dataset"):
        LeWMSequenceDataset(
            path,
            history_size=2,
            rollout_steps=2,
            correction_cache=wrong_cache,
        )
