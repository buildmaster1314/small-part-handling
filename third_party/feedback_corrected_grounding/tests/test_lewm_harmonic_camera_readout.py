from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_action_prefix_joint_a4 import ANCHOR_COUNT, HORIZON
from fcpg.lewm_harmonic_camera_readout import (
    HARMONIC_DIM,
    HarmonicCameraReadout,
    harmonic_checkpoint,
    load_harmonic_checkpoint,
    orbit_harmonic_features,
)


def test_harmonic_features_are_periodic_and_low_order() -> None:
    features = orbit_harmonic_features(torch.tensor([12, 3, 6, 9]))
    assert features.shape == (4, HARMONIC_DIM)
    np.testing.assert_allclose(features[:, 0].numpy(), 1.0)
    assert features[0, 1] == pytest.approx(0.0, abs=1.0e-6)
    assert features[1, 1] == pytest.approx(1.0, abs=1.0e-6)
    assert features[2, 2] == pytest.approx(-1.0, abs=1.0e-6)


def test_zero_harmonic_readout_preserves_base_prediction() -> None:
    model = HarmonicCameraReadout()
    base = torch.randn(2, HORIZON, ANCHOR_COUNT, 2)
    current = torch.randn(2, ANCHOR_COUNT, 2)
    output = model(
        base_prediction_q_px=base,
        current_q_px=current,
        task_index=torch.tensor([0, 1]),
        clock_hour=torch.tensor([3, 9]),
        image_size=(224, 224),
    )
    torch.testing.assert_close(output.predicted_q_px, base)
    torch.testing.assert_close(output.residual_q_px, torch.zeros_like(base))


def test_harmonic_checkpoint_round_trip() -> None:
    model = HarmonicCameraReadout()
    model.weight.data.normal_(generator=torch.Generator().manual_seed(9))
    payload = harmonic_checkpoint(
        model,
        base_checkpoint="base.pt",
        raw_dataset="raw",
        ridge=0.01,
    )
    loaded = load_harmonic_checkpoint(payload)
    torch.testing.assert_close(loaded.weight, model.weight)
