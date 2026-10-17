from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fcpg.lewm_pixel_observer import (
    LeWMPixelInnovationObserver,
    PixelResidualPriorAdapter,
    fixed_gain_update,
    sample_posterior_features,
)


class _DeterministicBaseLeWM(nn.Module):
    """Small CPU-only LeWM contract whose latent is the sampled anchor mean."""

    def __init__(self) -> None:
        super().__init__()
        self.history_size = 2
        self.latent_dim = 2
        self.action_dim = 2
        self.frame_encoder = SimpleNamespace(num_keypoints=2)

    def encode_frames(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_posterior = batch["q_corrected"].float()
        embedding = q_posterior.mean(dim=2)
        return embedding, embedding

    def predict(
        self,
        embedding: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if action.shape[:2] != embedding.shape[:2]:
            raise ValueError("test LeWM received misaligned actions")
        return embedding + action


class _EmbeddingPriorAdapter(nn.Module):
    """Decode the latest LeWM prior directly as each anchor residual."""

    latent_dim = 2
    num_keypoints = 2
    image_size = (5, 5)

    def forward(
        self,
        predicted_embedding: torch.Tensor,
        last_posterior_residual_px: torch.Tensor,
        last_q_nominal_px: torch.Tensor,
        last_keypoints_xyz: torch.Tensor,
        last_valid: torch.Tensor,
    ) -> torch.Tensor:
        del (
            last_posterior_residual_px,
            last_q_nominal_px,
            last_keypoints_xyz,
            last_valid,
        )
        return predicted_embedding.unsqueeze(1).expand(-1, self.num_keypoints, -1)


class _HoldLastPriorAdapter(_EmbeddingPriorAdapter):
    def forward(
        self,
        predicted_embedding: torch.Tensor,
        last_posterior_residual_px: torch.Tensor,
        last_q_nominal_px: torch.Tensor,
        last_keypoints_xyz: torch.Tensor,
        last_valid: torch.Tensor,
    ) -> torch.Tensor:
        del predicted_embedding, last_q_nominal_px, last_keypoints_xyz, last_valid
        return last_posterior_residual_px


def _observer(*, gain: float = 0.5) -> LeWMPixelInnovationObserver:
    return LeWMPixelInnovationObserver(
        _DeterministicBaseLeWM(),
        _EmbeddingPriorAdapter(),
        gain=gain,
        freeze_base=True,
    )


def _sequence_batch() -> dict[str, torch.Tensor]:
    batch_size, frames, keypoints = 1, 4, 2
    q_nominal = torch.tensor(
        [
            [
                [[1.0, 1.0], [2.0, 1.0]],
                [[1.0, 1.0], [2.0, 1.0]],
                [[1.0, 1.0], [2.0, 1.0]],
                [[1.0, 1.0], [2.0, 1.0]],
            ]
        ]
    )
    residual = torch.tensor(
        [
            [
                [[0.25, 0.50], [0.75, 1.00]],
                [[0.50, 0.75], [1.00, 1.25]],
                [[0.75, 1.00], [1.25, 1.50]],
                [[1.00, 1.25], [1.50, 1.75]],
            ]
        ]
    )
    return {
        "pixels": torch.zeros(batch_size, frames, 3, 5, 5),
        "proprio": torch.zeros(batch_size, frames, 4),
        "q_nominal": q_nominal,
        "q_corrected": q_nominal + residual,
        "keypoints_xyz": torch.zeros(batch_size, frames, keypoints, 3),
        "valid": torch.ones(batch_size, frames, keypoints, dtype=torch.bool),
        "action": torch.zeros(batch_size, frames - 1, 2),
    }


def _clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in batch.items()}


def test_fixed_gain_update_has_exact_numeric_and_endpoint_semantics() -> None:
    prior = torch.full((1, 2, 2), 2.0)
    measurement = torch.full_like(prior, 10.0)

    innovation, posterior, effective_gain = fixed_gain_update(
        prior, measurement, 0.25
    )
    torch.testing.assert_close(innovation, torch.full_like(prior, 8.0))
    torch.testing.assert_close(posterior, torch.full_like(prior, 4.0))
    torch.testing.assert_close(effective_gain, torch.full_like(prior, 0.25))

    _, zero_gain, _ = fixed_gain_update(prior, measurement, 0.0)
    _, unit_gain, _ = fixed_gain_update(prior, measurement, 1.0)
    torch.testing.assert_close(zero_gain, prior, rtol=0.0, atol=0.0)
    torch.testing.assert_close(unit_gain, measurement, rtol=0.0, atol=0.0)


def test_fixed_gain_masks_invalid_measurements_and_has_correct_gradients() -> None:
    prior = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
    )
    measurement = torch.tensor(
        [[[11.0, 12.0], [13.0, 14.0]]], requires_grad=True
    )
    valid = torch.tensor([[True, False]])

    _, posterior, effective_gain = fixed_gain_update(
        prior, measurement, 0.25, valid
    )
    expected = torch.tensor([[[3.5, 4.5], [3.0, 4.0]]])
    torch.testing.assert_close(posterior, expected)
    torch.testing.assert_close(
        effective_gain,
        torch.tensor([[[0.25, 0.25], [0.0, 0.0]]]),
    )

    posterior.sum().backward()
    torch.testing.assert_close(
        prior.grad,
        torch.tensor([[[0.75, 0.75], [1.0, 1.0]]]),
    )
    torch.testing.assert_close(
        measurement.grad,
        torch.tensor([[[0.25, 0.25], [0.0, 0.0]]]),
    )


def test_fixed_gain_invalid_nan_measurement_retains_finite_prior() -> None:
    prior = torch.tensor([[[1.0, 2.0]]])
    measurement = torch.full_like(prior, float("nan"))
    innovation, posterior, gain = fixed_gain_update(
        prior,
        measurement,
        0.75,
        measurement_valid=torch.tensor([[False]]),
    )

    torch.testing.assert_close(innovation, torch.zeros_like(prior))
    torch.testing.assert_close(posterior, prior)
    torch.testing.assert_close(gain, torch.zeros_like(prior))


def test_fixed_gain_rejects_valid_nan_measurement() -> None:
    prior = torch.tensor([[[1.0, 2.0]]])
    measurement = torch.full_like(prior, float("nan"))
    with pytest.raises(ValueError, match="valid measurement residuals must be finite"):
        fixed_gain_update(
            prior,
            measurement,
            0.75,
            measurement_valid=torch.tensor([[True]]),
        )


@pytest.mark.parametrize("gain", [-0.01, 1.01, float("nan"), float("inf")])
def test_fixed_gain_rejects_invalid_gain(gain: float) -> None:
    residual = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="observer gain"):
        fixed_gain_update(residual, residual, gain)


def test_zero_initialized_adapter_is_exactly_hold_last() -> None:
    torch.manual_seed(17)
    adapter = PixelResidualPriorAdapter(
        latent_dim=4,
        num_keypoints=3,
        image_size=(16, 12),
        hidden_dim=8,
    )
    last = torch.randn(2, 3, 2)
    output = adapter(
        predicted_embedding=torch.randn(2, 4),
        last_posterior_residual_px=last,
        last_q_nominal_px=torch.rand(2, 3, 2) * 10.0,
        last_keypoints_xyz=torch.randn(2, 3, 3),
        last_valid=torch.ones(2, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(output, last, rtol=0.0, atol=0.0)

    target = last + 1.0
    (output - target).square().mean().backward()
    final = adapter.net[-1]
    assert final.weight.grad is not None
    assert final.bias.grad is not None
    assert float(final.weight.grad.abs().sum()) > 0.0
    assert float(final.bias.grad.abs().sum()) > 0.0


def test_target_frame_observation_cannot_change_its_prior() -> None:
    observer = _observer(gain=0.5).eval()
    original = _sequence_batch()
    changed = _clone_batch(original)
    target = observer.history_size
    changed["pixels"][:, target] = 99.0
    changed["proprio"][:, target] = -77.0
    changed["q_nominal"][:, target] += 3.0
    changed["q_corrected"][:, target] -= 2.0
    changed["keypoints_xyz"][:, target] = 55.0

    with torch.no_grad():
        reference_output = observer.filter_sequence(original)
        changed_output = observer.filter_sequence(changed)

    torch.testing.assert_close(
        changed_output.prior_residual_px[:, 0],
        reference_output.prior_residual_px[:, 0],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        changed_output.posterior_residual_px[:, 0],
        reference_output.posterior_residual_px[:, 0],
    )


def test_posterior_embedding_is_written_back_into_the_next_prior() -> None:
    observer = _observer(gain=1.0).eval()
    original = _sequence_batch()
    changed = _clone_batch(original)
    target = observer.history_size
    changed["q_corrected"][:, target] += torch.tensor([2.0, -1.0])

    with torch.no_grad():
        reference_output = observer.filter_sequence(original)
        changed_output = observer.filter_sequence(changed)

    torch.testing.assert_close(
        changed_output.prior_residual_px[:, 0],
        reference_output.prior_residual_px[:, 0],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        changed_output.posterior_residual_px[:, 0],
        reference_output.posterior_residual_px[:, 0],
    )
    assert not torch.equal(
        changed_output.posterior_embedding[:, 0],
        reference_output.posterior_embedding[:, 0],
    )
    assert not torch.equal(
        changed_output.prior_residual_px[:, 1],
        reference_output.prior_residual_px[:, 1],
    )


def test_unit_gain_is_equivalent_to_fcpg_on_all_observed_frames() -> None:
    observer = _observer(gain=1.0).eval()
    batch = _sequence_batch()

    with torch.no_grad():
        output = observer.filter_sequence(batch)

    start = observer.history_size
    expected_measurement = (
        batch["q_corrected"][:, start:] - batch["q_nominal"][:, start:]
    )
    torch.testing.assert_close(
        output.posterior_residual_px,
        expected_measurement,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.q_posterior_px,
        batch["q_corrected"][:, start:],
        rtol=0.0,
        atol=0.0,
    )


def test_invalid_nan_warmup_measurement_retains_previous_posterior() -> None:
    observer = LeWMPixelInnovationObserver(
        _DeterministicBaseLeWM(),
        _HoldLastPriorAdapter(),
        gain=0.5,
        freeze_base=True,
    ).eval()
    batch = _sequence_batch()
    first_measurement = batch["q_corrected"][:, 0] - batch["q_nominal"][:, 0]
    batch["q_corrected"][:, 1] = float("nan")
    batch["valid"][:, 1] = False

    with torch.no_grad():
        output = observer.filter_sequence(batch)

    torch.testing.assert_close(output.prior_residual_px[:, 0], first_measurement)
    assert bool(torch.isfinite(output.prior_residual_px).all())
    assert bool(torch.isfinite(output.posterior_embedding).all())


def test_default_forward_rejects_invalid_nan_measurement() -> None:
    observer = _observer(gain=0.5).eval()
    batch = _sequence_batch()
    target_index = observer.history_size
    batch["q_corrected"][:, target_index] = float("nan")
    batch["valid"][:, target_index] = False

    with torch.no_grad():
        output = observer(batch)

    assert bool((output.effective_gain[:, 0] == 0).all())
    torch.testing.assert_close(
        output.posterior_residual_px[:, 0], output.prior_residual_px[:, 0]
    )
    assert bool(torch.isfinite(output.posterior_residual_px).all())
    assert bool(torch.isfinite(output.posterior_embedding).all())


def test_posterior_feature_sampling_uses_nominal_plus_posterior_residual() -> None:
    width, height = 5, 4
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    feature_map = torch.stack([x, y], dim=0).unsqueeze(0)
    q_nominal = torch.tensor([[[1.0, 1.0], [2.0, 1.0]]])
    posterior = torch.tensor([[[1.0, 1.0], [-1.0, 2.0]]])

    sampled = sample_posterior_features(
        feature_map,
        q_nominal,
        posterior,
        image_size=(width, height),
    )

    torch.testing.assert_close(
        sampled,
        torch.tensor([[[2.0, 2.0], [1.0, 3.0]]]),
        rtol=0.0,
        atol=1e-6,
    )


def test_forecast_accepts_only_history_and_actions_and_has_expected_shapes() -> None:
    observer = _observer().eval()
    batch = _sequence_batch()
    history = {
        name: value[:, : observer.history_size]
        for name, value in batch.items()
        if name != "action"
    }
    history["action"] = batch["action"][:, : observer.history_size - 1]
    history_residual = history["q_corrected"] - history["q_nominal"]
    future_action = torch.zeros(1, 3, 2)

    with torch.no_grad():
        forecast = observer.forecast_residuals(
            history,
            history_residual,
            future_action,
        )

    assert forecast.prior_residual_px.shape == (1, 3, 2, 2)
    assert forecast.predicted_embedding.shape == (1, 3, 2)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        observer.forecast_residuals(
            history,
            history_residual,
            future_action,
            future_pixels=batch["pixels"][:, observer.history_size :],
        )


@pytest.mark.parametrize("oracle_key", ["q_true", "q_oracle", "oracle_grounding"])
def test_oracle_pixels_are_rejected_from_observer_input(oracle_key: str) -> None:
    observer = _observer().eval()
    batch = _sequence_batch()
    batch[oracle_key] = torch.zeros_like(batch["q_nominal"])

    with pytest.raises(ValueError, match="evaluator-only"):
        observer.filter_sequence(batch)
