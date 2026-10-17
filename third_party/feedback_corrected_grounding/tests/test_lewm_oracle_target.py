from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fcpg.lewm_oracle_target import (
    FrozenOracleTargetEncoder,
    load_frozen_oracle_target_encoder,
    save_frozen_oracle_target_encoder,
)
from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_sequence_data import LeWMSequenceDataset, frame_identity_sha256


def _write_source(path: Path) -> int:
    count = 8
    episode_id = np.asarray([10] * 4 + [11] * 4, dtype=np.int64)
    frame_id = np.tile(np.arange(4, dtype=np.int64), 2)
    rng = np.random.default_rng(11)
    q_nominal = rng.uniform(0.0, 7.0, size=(count, 3, 2)).astype(np.float32)
    np.savez(
        path,
        rgb=rng.integers(0, 255, size=(count, 8, 8, 3), dtype=np.uint8),
        obs=rng.normal(size=(count, 6)).astype(np.float32),
        action=rng.normal(size=(count, 4)).astype(np.float32),
        task_name=np.asarray(["reach-v3"] * count),
        episode_id=episode_id,
        frame_id=frame_id,
        keypoints_xyz=rng.normal(size=(count, 3, 3)).astype(np.float32),
        q_nominal=q_nominal,
        q_true=q_nominal + 0.25,
        valid_true=np.ones((count, 3), dtype=bool),
        valid_nominal=np.ones((count, 3), dtype=bool),
        observation_action_alignment=np.asarray("pre_action"),
    )
    return count


def _write_target_cache(
    directory: Path,
    source: Path,
    encoder_path: Path,
    *,
    target_dim: int = 16,
) -> np.ndarray:
    with np.load(source, allow_pickle=False) as raw:
        episode_id = np.asarray(raw["episode_id"])
        frame_id = np.asarray(raw["frame_id"])
    rng = np.random.default_rng(17)
    target = rng.normal(size=(len(frame_id), target_dim)).astype(np.float32)
    directory.mkdir()
    target_path = directory / "oracle_target.npy"
    np.save(target_path, target)
    encoder, _ = load_frozen_oracle_target_encoder(encoder_path)
    metadata = {
        "schema_version": 1,
        "cache_type": "fcpg_oracle_target_v1",
        "source_data": str(source.resolve()),
        "source_data_sha256": sha256_file(source),
        "sample_count": len(frame_id),
        "num_keypoints": 3,
        "target_dim": target_dim,
        "encoder_checkpoint": str(encoder_path.resolve()),
        "encoder_checkpoint_sha256": sha256_file(encoder_path),
        "encoder_spec": encoder.spec(),
        "frame_identity_sha256": frame_identity_sha256(episode_id, frame_id),
        "oracle_target_sha256": sha256_file(target_path),
        "q_true_usage": "target_construction_only",
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return target


def test_frozen_encoder_is_deterministic_and_masks_invalid_anchors(tmp_path) -> None:
    encoder = FrozenOracleTargetEncoder(
        visual_dim=8,
        num_keypoints=3,
        target_dim=16,
        projection_seed=9,
    )
    feature_map = torch.randn(2, 8, 4, 4)
    global_feature = torch.randn(2, 8)
    q_true = torch.tensor(
        [
            [[1.0, 1.0], [4.0, 4.0], [6.0, 6.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ]
    )
    valid = torch.tensor([[True, False, True], [True, True, True]])

    first = encoder.forward_from_features(
        feature_map, global_feature, q_true, valid, image_size=(8, 8)
    )
    second = encoder.forward_from_features(
        feature_map, global_feature, q_true, valid, image_size=(8, 8)
    )

    assert first.shape == (2, 16)
    torch.testing.assert_close(first, second)
    assert bool(torch.all(first[0, 8:12] == 0.0))
    assert not list(encoder.parameters())

    checkpoint = tmp_path / "target_encoder.pt"
    save_frozen_oracle_target_encoder(checkpoint, encoder)
    restored, payload = load_frozen_oracle_target_encoder(checkpoint)
    assert payload["spec"] == encoder.spec()
    torch.testing.assert_close(restored.projection, encoder.projection)

    first_hash = frame_identity_sha256(
        np.asarray(["episode-a", "episode-b"]), np.asarray([0, 1])
    )
    second_hash = frame_identity_sha256(
        np.asarray(["episode-a", "episode-b"]), np.asarray([0, 1])
    )
    assert first_hash == second_hash


def test_dataset_exposes_only_cached_target_and_verifies_hashes(tmp_path) -> None:
    source = tmp_path / "source.npz"
    _write_source(source)
    encoder_path = tmp_path / "encoder.pt"
    save_frozen_oracle_target_encoder(
        encoder_path,
        FrozenOracleTargetEncoder(visual_dim=8, num_keypoints=3, target_dim=16),
    )
    cache = tmp_path / "target_cache"
    target = _write_target_cache(cache, source, encoder_path)

    dataset = LeWMSequenceDataset(
        source,
        history_size=2,
        rollout_steps=1,
        oracle_target_cache=cache,
    )
    sample = dataset[0]
    assert "q_true" not in sample
    assert sample["oracle_target"].shape == (3, 16)
    torch.testing.assert_close(
        sample["oracle_target"],
        torch.from_numpy(target[sample["raw_index"].numpy()]),
    )

    target[0, 0] += 1.0
    np.save(cache / "oracle_target.npy", target)
    with pytest.raises(ValueError, match="array hash"):
        LeWMSequenceDataset(
            source,
            history_size=2,
            rollout_steps=1,
            oracle_target_cache=cache,
        )
