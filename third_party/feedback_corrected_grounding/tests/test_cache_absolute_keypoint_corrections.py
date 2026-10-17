from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fcpg.absolute_keypoint_baseline import AbsoluteKeypointOutput
from scripts import cache_absolute_keypoint_corrections as cache_absolute


def _write_deployment_only_dataset(path: Path, count: int = 3) -> None:
    np.savez(
        path,
        obs=np.zeros((count, 4), dtype=np.float32),
        q_nominal=np.arange(count * 3 * 2, dtype=np.float32).reshape(count, 3, 2),
        keypoints_xyz=np.zeros((count, 3, 3), dtype=np.float32),
        episode_id=np.zeros(count, dtype=np.int64),
        frame_id=np.arange(count, dtype=np.int64),
    )


def test_inference_dataset_does_not_require_or_return_q_true(tmp_path: Path) -> None:
    data_path = tmp_path / "deployment.npz"
    _write_deployment_only_dataset(data_path)
    feature_path = tmp_path / "features.npy"
    prefix_path = tmp_path / "prefix.npy"
    np.save(feature_path, np.zeros((3, 1, 4, 2, 2), dtype=np.float16))
    np.save(prefix_path, np.zeros((3, 4), dtype=np.float16))
    dataset = cache_absolute.AbsoluteKeypointInferenceDataset(
        data_path,
        {
            "feature_path": str(feature_path),
            "prefix_path": str(prefix_path),
            "image_size": (32, 24),
        },
        expected_obs_dim=4,
        expected_num_keypoints=3,
    )

    sample = dataset[0]

    assert set(sample) == {
        "vision_backbone_features",
        "vision_prefix",
        "q_nominal",
        "keypoints_xyz_fk",
        "robot_obs",
        "image_size",
    }
    assert all("true" not in key for key in sample)


def test_cache_is_action_chunk_compatible_and_uses_no_q_true(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "deployment.npz"
    _write_deployment_only_dataset(data_path)
    vision_cache = tmp_path / "vision"
    vision_cache.mkdir()
    feature_path = vision_cache / "backbone_features.npy"
    prefix_path = vision_cache / "prefix.npy"
    vision_metadata_path = vision_cache / "metadata.json"
    np.save(feature_path, np.zeros((3, 1, 4, 2, 2), dtype=np.float16))
    np.save(prefix_path, np.zeros((3, 4), dtype=np.float16))
    vision_metadata_path.write_text("{}", encoding="utf-8")
    checkpoint_path = tmp_path / "absolute.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    reference_path = tmp_path / "reference.pt"
    reference_path.write_bytes(b"reference")
    output_dir = tmp_path / "corrections"

    class FakeAbsoluteModel:
        obs_dim = 4
        num_keypoints = 3
        coordinate_parameterization = "direct-test"

        def eval(self):
            return self

        def forward_from_backbone_features(
            self,
            _features,
            _prefix,
            q_nominal,
            _xyz,
            _obs,
            *,
            image_size,
        ):
            assert image_size == (32, 24)
            predicted = torch.full_like(q_nominal, 7.0)
            return AbsoluteKeypointOutput(
                q_absolute_norm=predicted / 31.0,
                q_absolute_px=predicted,
                keypoint_latent=torch.zeros(
                    q_nominal.shape[0], q_nominal.shape[1], 4
                ),
            )

        def parameter_report(self):
            return {"trainable_parameters": 10}

        def sampling_report(self):
            return {"sampled_visual_tokens_per_frame": 75}

    model = FakeAbsoluteModel()
    checkpoint = {
        "schema_version": 1,
        "model_config": {},
        "training_provenance": {
            "reference_fcpg": {"checkpoint": str(reference_path)}
        },
    }
    monkeypatch.setattr(
        cache_absolute,
        "load_absolute_keypoint_baseline",
        lambda *_args, **_kwargs: (model, checkpoint),
    )
    monkeypatch.setattr(
        cache_absolute,
        "validate_deployment_vision_cache",
        lambda *_args, **_kwargs: {
            "feature_path": str(feature_path),
            "prefix_path": str(prefix_path),
            "image_size": (32, 24),
            "source_data_sha256": "1" * 64,
            "selected_image_sha256": "2" * 64,
            "metadata_path": str(vision_metadata_path),
            "backbone_features_sha256": "3" * 64,
            "prefix_sha256": "4" * 64,
        },
    )

    assert cache_absolute.main(
        [
            "--data",
            str(data_path),
            "--vision-cache",
            str(vision_cache),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    ) == 0

    corrected = np.load(output_dir / "q_corrected.npy", allow_pickle=False)
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    np.testing.assert_array_equal(corrected, np.full((3, 3, 2), 7.0, dtype=np.float32))
    assert metadata["has_reliability"] is False
    assert metadata["q_true_usage"] == "none"
    assert metadata["corrector_arch"] == "anchor_conditioned_absolute_keypoint"
    assert metadata["deployment_cache_eligible"] is True
    assert not (output_dir / "reliability.npy").exists()
