from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_action_prefix_joint_a4 import joint_batch_from_memmaps
from fcpg.lewm_pose_conditioned_joint_a4 import (
    CameraLatentConditioner,
    camera_features_from_batch,
)
from fcpg.lewm_pose_conditioned_readout import orbit_camera_basis_features
from scripts.train_lewm_pose_conditioned_joint_a4 import _cuda0, _parser


def test_camera_latent_conditioner_is_exact_identity_at_initialization() -> None:
    module = CameraLatentConditioner(hidden_dim=32)
    embedding = torch.randn(3, 9, 4, 192)
    camera = torch.randn(3, 10)

    result = module(embedding, camera)

    torch.testing.assert_close(result, embedding)


def test_camera_latent_conditioner_responds_to_continuous_pose_after_learning() -> None:
    module = CameraLatentConditioner(hidden_dim=16)
    with torch.no_grad():
        module.network[0].weight.zero_()
        module.network[0].bias.zero_()
        module.network[0].weight[0, 0] = 1.0
        module.network[-1].weight[192, 0] = 1.0
    embedding = torch.zeros(2, 1, 4, 192)
    camera = torch.zeros(2, 10)
    camera[1, 0] = 1.0

    result = module(embedding, camera)

    assert float(result[0].abs().max()) == 0.0
    assert float(result[1, 0, 0, 0]) > 0.0


def test_camera_features_from_batch_matches_orbit_basis() -> None:
    batch = {
        "camera_clock_hour": torch.tensor([3, 11]),
        "camera_elevation_degrees": torch.tensor([35.0, 35.0]),
        "camera_radius_m": torch.tensor([1.1, 1.1]),
    }

    result = camera_features_from_batch(batch)
    expected = orbit_camera_basis_features(
        batch["camera_clock_hour"],
        batch["camera_elevation_degrees"],
        batch["camera_radius_m"],
    )

    torch.testing.assert_close(result, expected)
    assert not torch.allclose(result[0], result[1])


def test_joint_memmap_batch_carries_continuous_camera_metadata() -> None:
    index = {
        "task_index": np.asarray([2], dtype=np.int64),
        "future_action": np.zeros((1, 8, 8, 20), dtype=np.float32),
        "camera_index": np.asarray([3], dtype=np.int64),
        "camera_clock_hour": np.asarray([3], dtype=np.int64),
        "camera_elevation_degrees": np.asarray([35.0], dtype=np.float32),
        "camera_radius_m": np.asarray([1.1], dtype=np.float32),
    }
    arrays = {
        "current_rgb": np.zeros((1, 3, 8, 8), dtype=np.uint8),
        "current_proprio": np.zeros((1, 4), dtype=np.float32),
        "current_q_nominal": np.zeros((1, 3, 2), dtype=np.float32),
        "current_q_corrected": np.zeros((1, 3, 2), dtype=np.float32),
        "current_keypoints_xyz": np.zeros((1, 3, 3), dtype=np.float32),
        "current_valid": np.ones((1, 3), dtype=np.bool_),
        "current_q_true": np.zeros((1, 3, 2), dtype=np.float32),
        "future_rgb": np.zeros((1, 8, 8, 3, 8, 8), dtype=np.uint8),
        "future_proprio": np.zeros((1, 8, 8, 4), dtype=np.float32),
        "future_q_nominal": np.zeros((1, 8, 8, 3, 2), dtype=np.float32),
        "future_q_corrected": np.zeros((1, 8, 8, 3, 2), dtype=np.float32),
        "future_keypoints_xyz": np.zeros((1, 8, 8, 3, 3), dtype=np.float32),
        "future_valid": np.ones((1, 8, 8, 3), dtype=np.bool_),
        "future_q_true": np.zeros((1, 8, 8, 3, 2), dtype=np.float32),
    }

    batch = joint_batch_from_memmaps(
        index,
        arrays,
        bundle_indices=np.asarray([0]),
        branch_indices=np.asarray([4]),
    )

    assert int(batch["camera_clock_hour"][0]) == 3
    assert float(batch["camera_elevation_degrees"][0]) == 35.0
    assert float(batch["camera_radius_m"][0]) == pytest.approx(1.1)


def test_pose_conditioned_training_rejects_cuda1() -> None:
    with pytest.raises(ValueError, match="CUDA0"):
        _cuda0("cuda:1")


def test_pose_conditioned_sigreg_accepts_only_registered_ablation_weights() -> None:
    parsed = _parser(["--output-dir", "/tmp/out", "--sigreg-weight", "0.09"])
    assert parsed.sigreg_weight == pytest.approx(0.09)
    with pytest.raises(SystemExit):
        _parser(["--output-dir", "/tmp/out", "--sigreg-weight", "0.05"])
