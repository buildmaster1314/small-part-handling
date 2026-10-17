from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_action_prefix_joint_a4 import (
    ANCHOR_COUNT,
    HORIZON,
    LATENT_DIM,
    STATE_SLOTS,
    AnchorMetricReadout,
)
from fcpg.lewm_pose_conditioned_readout import (
    CAMERA_FEATURE_DIM,
    POSE_READOUT_CHECKPOINT_TYPE,
    PoseConditionedResidualReadout,
    load_pose_readout_checkpoint,
    orbit_camera_basis_features,
    pose_readout_checkpoint,
)


def _inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {
        "predicted_state": torch.randn(
            batch_size,
            HORIZON,
            STATE_SLOTS,
            LATENT_DIM,
            generator=generator,
        ),
        "current_state": torch.randn(
            batch_size,
            STATE_SLOTS,
            LATENT_DIM,
            generator=generator,
        ),
        "base_prediction_q_px": 112.0
        + 20.0
        * torch.randn(
            batch_size,
            HORIZON,
            ANCHOR_COUNT,
            2,
            generator=generator,
        ),
        "current_q_px": 112.0
        + 10.0
        * torch.randn(batch_size, ANCHOR_COUNT, 2, generator=generator),
        "current_keypoints_xyz": torch.randn(
            batch_size, ANCHOR_COUNT, 3, generator=generator
        ),
        "task_index": torch.arange(batch_size) % 7,
        "camera_features": orbit_camera_basis_features(
            torch.tensor([3, 9][:batch_size]),
            torch.full((batch_size,), 35.0),
            torch.full((batch_size,), 1.1),
        ),
    }


def test_orbit_camera_basis_matches_clock_axes_and_is_periodic() -> None:
    features = orbit_camera_basis_features(
        torch.tensor([12, 3, 6, 9]),
        torch.full((4,), 35.0),
        torch.full((4,), 1.1),
    )
    assert features.shape == (4, CAMERA_FEATURE_DIM)
    np.testing.assert_allclose(features[:, -1].numpy(), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        features[:, :3].norm(dim=-1).numpy(), 1.0, atol=1.0e-6
    )
    np.testing.assert_allclose(
        features[:, 3:6].norm(dim=-1).numpy(), 1.0, atol=1.0e-6
    )
    np.testing.assert_allclose(
        features[:, 6:9].norm(dim=-1).numpy(), 1.0, atol=1.0e-6
    )
    assert features[0, 6] == pytest.approx(0.0, abs=1.0e-6)
    assert float(features[0, 7]) > 0.0
    assert float(features[1, 6]) < 0.0
    assert features[1, 7] == pytest.approx(0.0, abs=1.0e-6)


def test_orbit_camera_basis_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match=r"\[1,12\]"):
        orbit_camera_basis_features(
            torch.tensor([0]), torch.tensor([35.0]), torch.tensor([1.1])
        )
    with pytest.raises(ValueError, match="positive"):
        orbit_camera_basis_features(
            torch.tensor([3]), torch.tensor([35.0]), torch.tensor([0.0])
        )


def test_pose_readout_is_exactly_backward_compatible_at_initialization() -> None:
    base = AnchorMetricReadout()
    model = PoseConditionedResidualReadout()
    model.initialize_embeddings_from(base)
    values = _inputs()
    output = model(**values, image_size=(224, 224))
    torch.testing.assert_close(
        output.predicted_q_px,
        values["base_prediction_q_px"].float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.residual_q_px,
        torch.zeros_like(output.residual_q_px),
        rtol=0.0,
        atol=0.0,
    )


def test_pose_readout_checkpoint_round_trip() -> None:
    model = PoseConditionedResidualReadout(hidden_dim=64)
    checkpoint = pose_readout_checkpoint(
        model,
        step=2000,
        seed=40541,
        base_checkpoint="base.pt",
        raw_dataset="raw",
        reference_mode="current_q",
    )
    assert checkpoint["checkpoint_type"] == POSE_READOUT_CHECKPOINT_TYPE
    loaded = load_pose_readout_checkpoint(checkpoint)
    values = _inputs()
    torch.testing.assert_close(
        loaded(**values, image_size=(224, 224)).predicted_q_px,
        model(**values, image_size=(224, 224)).predicted_q_px,
    )
