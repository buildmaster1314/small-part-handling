from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.evaluate_lewm_camera_action_error_decomposition import (
    _accumulate,
    _cuda0,
    _empty_accumulator,
    _finalize,
    _parse_checkpoints,
)


def test_decomposition_separates_substitution_and_dynamics_error() -> None:
    accumulator = _empty_accumulator()
    target_q = torch.zeros(1, 8, 3, 2)
    current_q = torch.zeros(1, 3, 2)
    predicted_q = target_q.clone()
    predicted_q[..., 0] = 3.0
    predicted_q[..., 1] = 4.0
    substitution_q = target_q.clone()
    substitution_q[..., 0] = 1.0
    predicted_state = torch.ones(1, 8, 4, 192)
    target_state = torch.zeros_like(predicted_state)

    _accumulate(
        accumulator,
        predicted_q=predicted_q,
        substitution_q=substitution_q,
        target_q=target_q,
        current_q=current_q,
        valid=torch.ones(1, 8, 3, dtype=torch.bool),
        predicted_state=predicted_state,
        target_state=target_state,
    )
    result = _finalize(accumulator)

    assert result["predicted_anchor_epe_px"] == pytest.approx(5.0)
    assert result["target_latent_substitution_anchor_epe_px"] == pytest.approx(1.0)
    assert result["dynamics_interface_excess_epe_px"] == pytest.approx(4.0)
    assert result["target_substitution_fraction_of_predicted_epe"] == pytest.approx(
        0.2
    )
    assert result["latent_prediction_mse"] == pytest.approx(1.0)
    assert len(result["per_horizon"]) == 8


def test_decomposition_ignores_invalid_anchor_points() -> None:
    accumulator = _empty_accumulator()
    target_q = torch.zeros(1, 8, 3, 2)
    predicted_q = torch.zeros_like(target_q)
    predicted_q[:, :, 0, 0] = 10.0
    valid = torch.ones(1, 8, 3, dtype=torch.bool)
    valid[:, :, 0] = False

    _accumulate(
        accumulator,
        predicted_q=predicted_q,
        substitution_q=target_q,
        target_q=target_q,
        current_q=torch.zeros(1, 3, 2),
        valid=valid,
        predicted_state=torch.zeros(1, 8, 4, 192),
        target_state=torch.zeros(1, 8, 4, 192),
    )
    result = _finalize(accumulator)

    assert result["predicted_anchor_epe_px"] == pytest.approx(0.0)
    assert result["point_count"] == 16


def test_checkpoint_arguments_require_unique_labels(tmp_path) -> None:
    parsed = _parse_checkpoints([f"two={tmp_path / 'two.pt'}", f"ten={tmp_path / 'ten.pt'}"])
    assert set(parsed) == {"two", "ten"}
    with pytest.raises(ValueError, match="duplicate"):
        _parse_checkpoints(["same=a.pt", "same=b.pt"])


def test_decomposition_rejects_cuda1() -> None:
    with pytest.raises(ValueError, match="CUDA0"):
        _cuda0("cuda:1")


def test_empty_accumulator_uses_eight_horizons() -> None:
    accumulator = _empty_accumulator()
    assert accumulator["point_count"].shape == (8,)
    np.testing.assert_array_equal(accumulator["point_count"], 0)
