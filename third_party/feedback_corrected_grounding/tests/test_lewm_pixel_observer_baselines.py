from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fcpg.lewm_pixel_observer import fixed_gain_update
from fcpg.lewm_pixel_observer_baselines import (
    ConstantVelocityResidualBaseline,
    HoldLastResidualBaseline,
    LinearConstantVelocityKalmanBaseline,
    LinearKalmanState,
    TemporalMeasurement,
    TemporalMeasurementSource,
    TemporalPixelBaseline,
    TemporalPrior,
    fcpg_measurement,
    run_temporal_baseline,
)


def _advance(
    baseline: TemporalPixelBaseline,
    state,
    measurement: torch.Tensor,
    valid: torch.Tensor | None = None,
):
    prior = baseline.prior(state)
    update = baseline.update(prior, fcpg_measurement(measurement, valid))
    return prior, update


def _baseline_factories():
    return (
        lambda: HoldLastResidualBaseline(),
        lambda: ConstantVelocityResidualBaseline(max_delta_px=8.0),
        lambda: LinearConstantVelocityKalmanBaseline(),
    )


@pytest.mark.parametrize("factory", _baseline_factories())
def test_initialize_uses_only_template_shape_device_and_dtype(factory) -> None:
    baseline = factory()
    finite = torch.full((2, 3, 2), 91.0, dtype=torch.float64)
    nonfinite = torch.full_like(finite, float("nan"))

    finite_prior = baseline.prior(baseline.initialize(finite))
    nonfinite_prior = baseline.prior(baseline.initialize(nonfinite))
    expected = torch.zeros_like(finite)
    torch.testing.assert_close(finite_prior.residual_px, expected)
    torch.testing.assert_close(nonfinite_prior.residual_px, expected)
    assert nonfinite_prior.residual_px.dtype == nonfinite.dtype
    assert nonfinite_prior.residual_px.device == nonfinite.device


@pytest.mark.parametrize("factory", _baseline_factories())
@pytest.mark.parametrize(
    "source",
    [TemporalMeasurementSource.ORACLE, TemporalMeasurementSource.UNKNOWN],
)
def test_update_rejects_non_fcpg_measurement_provenance(factory, source) -> None:
    baseline = factory()
    prior = baseline.prior(baseline.initialize(torch.zeros(1, 1, 2)))
    measurement = TemporalMeasurement(
        residual_px=torch.zeros(1, 1, 2),
        valid=torch.ones(1, 1, dtype=torch.bool),
        source=source,
    )
    with pytest.raises(ValueError, match="only FCPG_MEASUREMENT"):
        baseline.update(prior, measurement)


@pytest.mark.parametrize("factory", _baseline_factories())
def test_update_rejects_untyped_tensor_as_unknown_provenance(factory) -> None:
    baseline = factory()
    prior = baseline.prior(baseline.initialize(torch.zeros(1, 1, 2)))
    with pytest.raises(TypeError, match="typed TemporalMeasurement"):
        baseline.update(prior, torch.zeros(1, 1, 2))  # type: ignore[arg-type]


def test_sequence_runner_hold_last_matches_existing_fixed_gain_stepwise() -> None:
    residual = torch.tensor(
        [
            [
                [[0.0, 0.0], [5.0, -2.0]],
                [[1.0, 0.5], [float("nan"), float("nan")]],
                [[2.0, 1.0], [7.0, -1.0]],
                [[3.0, 1.5], [8.0, -0.5]],
                [[4.0, 2.0], [9.0, 0.0]],
                [[5.0, 2.5], [10.0, 0.5]],
            ]
        ],
        dtype=torch.float64,
    )
    valid = torch.ones(1, 6, 2, dtype=torch.bool)
    valid[:, 1, 1] = False
    warmup = 3
    output = run_temporal_baseline(
        HoldLastResidualBaseline(),
        fcpg_measurement(residual, valid),
        warmup_frames=warmup,
    )

    posterior = torch.zeros_like(residual[:, 0])
    expected_prior = []
    expected_posterior = []
    expected_gain = []
    for index in range(residual.shape[1]):
        prior = posterior
        _, posterior, gain = fixed_gain_update(
            prior, residual[:, index], 1.0, valid[:, index]
        )
        if index >= warmup:
            expected_prior.append(prior)
            expected_posterior.append(posterior)
            expected_gain.append(gain)

    torch.testing.assert_close(
        output.prior_residual_px, torch.stack(expected_prior, dim=1)
    )
    torch.testing.assert_close(
        output.posterior_residual_px, torch.stack(expected_posterior, dim=1)
    )
    torch.testing.assert_close(
        output.effective_gain, torch.stack(expected_gain, dim=1)
    )
    assert output.warmup_frames == warmup
    assert output.measurement_source is TemporalMeasurementSource.FCPG_MEASUREMENT


@pytest.mark.parametrize(
    ("factory", "has_kalman_gain"),
    [
        (lambda: HoldLastResidualBaseline(), False),
        (lambda: ConstantVelocityResidualBaseline(), False),
        (lambda: LinearConstantVelocityKalmanBaseline(), True),
    ],
)
def test_sequence_runner_returns_unified_scored_shapes(
    factory, has_kalman_gain
) -> None:
    residual = torch.randn(2, 7, 3, 2, dtype=torch.float64)
    output = run_temporal_baseline(
        factory(), fcpg_measurement(residual), warmup_frames=3
    )
    expected = (2, 4, 3, 2)
    assert output.measurement_residual_px.shape == expected
    assert output.measurement_valid.shape == expected[:-1]
    assert output.prior_residual_px.shape == expected
    assert output.posterior_residual_px.shape == expected
    assert output.innovation_px.shape == expected
    assert output.effective_gain.shape == expected
    assert (output.kalman_gain is not None) is has_kalman_gain
    if output.kalman_gain is not None:
        assert output.kalman_gain.shape == (2, 4, 3, 4, 2)


def test_sequence_runner_resets_state_on_every_call() -> None:
    baseline = ConstantVelocityResidualBaseline(max_delta_px=20.0)
    moving = torch.arange(6, dtype=torch.float64).view(1, 6, 1, 1)
    moving = torch.cat([moving, -moving], dim=-1)
    stationary = torch.zeros_like(moving)
    run_temporal_baseline(
        baseline, fcpg_measurement(moving), warmup_frames=2
    )
    reused = run_temporal_baseline(
        baseline, fcpg_measurement(stationary), warmup_frames=2
    )
    fresh = run_temporal_baseline(
        ConstantVelocityResidualBaseline(max_delta_px=20.0),
        fcpg_measurement(stationary),
        warmup_frames=2,
    )
    torch.testing.assert_close(
        reused.prior_residual_px, fresh.prior_residual_px, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        reused.posterior_residual_px,
        fresh.posterior_residual_px,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "source",
    [TemporalMeasurementSource.ORACLE, TemporalMeasurementSource.UNKNOWN],
)
def test_sequence_runner_rejects_oracle_and_unknown_sources(source) -> None:
    measurement = TemporalMeasurement(
        residual_px=torch.zeros(1, 4, 1, 2),
        valid=None,
        source=source,
    )
    with pytest.raises(ValueError, match="only FCPG_MEASUREMENT"):
        run_temporal_baseline(
            HoldLastResidualBaseline(), measurement, warmup_frames=2
        )


def test_sequence_runner_rejects_q_true_mapping_instead_of_inspecting_it() -> None:
    with pytest.raises(TypeError, match="typed TemporalMeasurement"):
        run_temporal_baseline(
            HoldLastResidualBaseline(),
            {"q_true": torch.zeros(1, 4, 1, 2)},  # type: ignore[arg-type]
            warmup_frames=2,
        )


def test_hold_last_uses_fixed_gain_and_retains_invalid_nan_measurement() -> None:
    baseline = HoldLastResidualBaseline(gain=0.25)
    state = baseline.initialize(torch.zeros(1, 2, 2))
    prior, update = _advance(
        baseline,
        state,
        torch.tensor([[[8.0, 4.0], [6.0, 2.0]]]),
        torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(prior.residual_px, torch.zeros_like(prior.residual_px))
    torch.testing.assert_close(
        update.posterior_residual_px,
        torch.tensor([[[2.0, 1.0], [0.0, 0.0]]]),
    )
    torch.testing.assert_close(
        update.effective_gain,
        torch.tensor([[[0.25, 0.25], [0.0, 0.0]]]),
    )

    prior = baseline.prior(update.state)
    invalid = torch.full_like(prior.residual_px, float("nan"))
    retained = baseline.update(
        prior,
        fcpg_measurement(invalid, torch.zeros(1, 2, dtype=torch.bool)),
    )
    torch.testing.assert_close(retained.innovation_px, torch.zeros_like(invalid))
    torch.testing.assert_close(
        retained.posterior_residual_px, prior.residual_px, rtol=0.0, atol=0.0
    )


def test_constant_velocity_exactly_extrapolates_synthetic_uniform_motion() -> None:
    baseline = ConstantVelocityResidualBaseline(max_delta_px=10.0)
    reference = torch.zeros(1, 2, 2, dtype=torch.float64)
    state = baseline.initialize(reference)
    velocity = torch.tensor(
        [[[2.0, -1.0], [-0.5, 1.5]]], dtype=reference.dtype
    )

    for step in range(7):
        measurement = velocity * step
        prior, update = _advance(baseline, state, measurement)
        if step >= 2:
            torch.testing.assert_close(
                prior.residual_px, measurement, rtol=0.0, atol=1e-12
            )
        state = update.state

    torch.testing.assert_close(
        state.velocity_px_per_step, velocity, rtol=0.0, atol=1e-12
    )


def test_constant_velocity_propagates_across_missing_measurements() -> None:
    baseline = ConstantVelocityResidualBaseline(max_delta_px=5.0)
    state = baseline.initialize(torch.zeros(1, 1, 2))
    valid = torch.ones(1, 1, dtype=torch.bool)

    _, first = _advance(baseline, state, torch.tensor([[[0.0, 0.0]]]), valid)
    _, second = _advance(
        baseline, first.state, torch.tensor([[[1.0, -2.0]]]), valid
    )
    missing_prior = baseline.prior(second.state)
    missing = baseline.update(
        missing_prior,
        fcpg_measurement(
            torch.full((1, 1, 2), float("nan")),
            torch.zeros(1, 1, dtype=torch.bool),
        ),
    )
    torch.testing.assert_close(
        missing.posterior_residual_px,
        torch.tensor([[[2.0, -4.0]]]),
        rtol=0.0,
        atol=0.0,
    )
    reacquired_prior = baseline.prior(missing.state)
    torch.testing.assert_close(
        reacquired_prior.residual_px,
        torch.tensor([[[3.0, -6.0]]]),
        rtol=0.0,
        atol=0.0,
    )
    reacquired = baseline.update(
        reacquired_prior,
        fcpg_measurement(torch.tensor([[[3.0, -6.0]]]), valid),
    )
    torch.testing.assert_close(
        reacquired.state.velocity_px_per_step,
        torch.tensor([[[1.0, -2.0]]]),
        rtol=0.0,
        atol=0.0,
    )


def test_constant_velocity_clips_per_step_delta_componentwise() -> None:
    baseline = ConstantVelocityResidualBaseline(max_delta_px=2.0)
    state = baseline.initialize(torch.zeros(1, 1, 2))
    _, first = _advance(baseline, state, torch.zeros(1, 1, 2))
    _, outlier = _advance(
        baseline, first.state, torch.tensor([[[100.0, -40.0]]])
    )
    torch.testing.assert_close(
        outlier.state.velocity_px_per_step,
        torch.tensor([[[2.0, -2.0]]]),
        rtol=0.0,
        atol=0.0,
    )
    next_prior = baseline.prior(outlier.state)
    torch.testing.assert_close(
        next_prior.residual_px,
        torch.tensor([[[102.0, -42.0]]]),
        rtol=0.0,
        atol=0.0,
    )


def test_linear_kalman_tracks_constant_velocity_without_future_observations() -> None:
    baseline = LinearConstantVelocityKalmanBaseline(
        acceleration_variance=0.0,
        measurement_variance=1e-3,
        initial_position_variance=100.0,
        initial_velocity_variance=100.0,
    )
    state = baseline.initialize(torch.zeros(1, 1, 2, dtype=torch.float64))
    velocity = torch.tensor([[[1.25, -0.75]]], dtype=torch.float64)

    for step in range(12):
        measurement = velocity * step
        prior, update = _advance(baseline, state, measurement)
        state = update.state

    final_prior = baseline.prior(state)
    expected = velocity * 12
    torch.testing.assert_close(final_prior.residual_px, expected, atol=2e-3, rtol=0.0)


def test_linear_kalman_invalid_measurement_skips_mean_and_covariance_update() -> None:
    baseline = LinearConstantVelocityKalmanBaseline()
    state = baseline.initialize(torch.zeros(1, 2, 2, dtype=torch.float64))
    prior = baseline.prior(state)
    invalid = torch.full((1, 2, 2), float("nan"), dtype=torch.float64)
    update = baseline.update(
        prior,
        fcpg_measurement(invalid, torch.zeros(1, 2, dtype=torch.bool)),
    )

    torch.testing.assert_close(update.state.mean, prior.state.mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        update.state.covariance, prior.state.covariance, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(update.innovation_px, torch.zeros_like(invalid))
    assert update.kalman_gain is not None
    torch.testing.assert_close(
        update.kalman_gain, torch.zeros_like(update.kalman_gain), rtol=0.0, atol=0.0
    )


def test_linear_kalman_first_measurement_does_not_invent_velocity() -> None:
    baseline = LinearConstantVelocityKalmanBaseline(
        measurement_variance=1e-6,
        initial_position_variance=100.0,
        initial_velocity_variance=100.0,
    )
    state = baseline.initialize(torch.zeros(1, 1, 2, dtype=torch.float64))
    prior = baseline.prior(state)
    torch.testing.assert_close(
        prior.state.covariance, state.covariance, rtol=0.0, atol=0.0
    )
    update = baseline.update(
        prior,
        fcpg_measurement(torch.tensor([[[9.0, -7.0]]], dtype=torch.float64)),
    )
    torch.testing.assert_close(
        update.state.mean[..., 2:], torch.zeros(1, 1, 2, dtype=torch.float64)
    )


def test_linear_kalman_covariance_stays_symmetric_psd() -> None:
    torch.manual_seed(118)
    baseline = LinearConstantVelocityKalmanBaseline(
        acceleration_variance=0.3,
        measurement_variance=2.0,
        psd_tolerance=1e-10,
    )
    state = baseline.initialize(torch.zeros(2, 3, 2, dtype=torch.float64))

    for step in range(40):
        prior = baseline.prior(state)
        measurement = torch.randn(2, 3, 2, dtype=torch.float64)
        valid = torch.rand(2, 3) > (0.2 if step % 3 else 0.7)
        measurement[~valid] = float("nan")
        state = baseline.update(prior, fcpg_measurement(measurement, valid)).state
        torch.testing.assert_close(
            state.covariance,
            state.covariance.transpose(-1, -2),
            rtol=0.0,
            atol=1e-12,
        )
        assert float(torch.linalg.eigvalsh(state.covariance).amin()) >= -1e-10


def test_linear_kalman_rejects_non_psd_state() -> None:
    baseline = LinearConstantVelocityKalmanBaseline()
    state = baseline.initialize(torch.zeros(1, 1, 2))
    bad_covariance = state.covariance.clone()
    bad_covariance[..., 0, 0] = -1.0
    bad_state = replace(state, covariance=bad_covariance)
    with pytest.raises(ValueError, match="positive semidefinite"):
        baseline.prior(bad_state)


@pytest.mark.parametrize("factory", _baseline_factories())
def test_common_interface_preserves_shape_device_and_dtype(factory) -> None:
    baseline = factory()
    assert isinstance(baseline, TemporalPixelBaseline)
    reference = torch.zeros(2, 3, 2, dtype=torch.float64)
    state = baseline.initialize(reference)
    prior = baseline.prior(state)
    update = baseline.update(prior, fcpg_measurement(torch.ones_like(reference)))

    for value in (
        prior.residual_px,
        update.prior_residual_px,
        update.innovation_px,
        update.posterior_residual_px,
        update.effective_gain,
    ):
        assert value.shape == reference.shape
        assert value.dtype == reference.dtype
        assert value.device == reference.device


@pytest.mark.parametrize("factory", _baseline_factories())
def test_common_interface_rejects_shape_and_dtype_mismatch(factory) -> None:
    baseline = factory()
    reference = torch.zeros(2, 3, 2)
    prior = baseline.prior(baseline.initialize(reference))
    with pytest.raises(ValueError, match="match the prior shape"):
        baseline.update(prior, fcpg_measurement(torch.zeros(2, 2, 2)))
    with pytest.raises(TypeError, match="dtypes must match"):
        baseline.update(
            prior,
            fcpg_measurement(torch.zeros(2, 3, 2, dtype=torch.float64)),
        )
    with pytest.raises(TypeError, match="torch.bool"):
        baseline.update(
            prior,
            fcpg_measurement(
                torch.zeros_like(reference),
                torch.ones(2, 3, dtype=torch.int64),
            ),
        )


@pytest.mark.parametrize("factory", _baseline_factories())
@pytest.mark.parametrize(
    "field",
    ["q_true", "target_rgb", "future_pixels", "true_extrinsic"],
)
def test_oracle_and_future_fields_fail_closed(factory, field: str) -> None:
    baseline = factory()
    state = baseline.initialize(torch.zeros(1, 1, 2))
    payload = {field: torch.zeros(1)}
    with pytest.raises(ValueError, match="must not enter"):
        baseline.prior(state, context=payload)

    prior = baseline.prior(state)
    with pytest.raises(ValueError, match="must not enter"):
        baseline.update(
            prior,
            fcpg_measurement(torch.zeros(1, 1, 2)),
            context=payload,
        )


@pytest.mark.parametrize("factory", _baseline_factories())
def test_unknown_context_fields_also_fail_closed(factory) -> None:
    baseline = factory()
    state = baseline.initialize(torch.zeros(1, 1, 2))
    with pytest.raises(ValueError, match="state-only"):
        baseline.prior(state, context={"unregistered_side_channel": 1})


@pytest.mark.parametrize("factory", _baseline_factories())
def test_manually_forged_prior_state_binding_fails_closed(factory) -> None:
    baseline = factory()
    state = baseline.initialize(torch.zeros(1, 1, 2))
    genuine = baseline.prior(state)
    forged = TemporalPrior(
        residual_px=genuine.residual_px + 1.0,
        state=genuine.state,
    )
    with pytest.raises(ValueError, match="does not match its state"):
        baseline.update(forged, fcpg_measurement(torch.zeros(1, 1, 2)))


@pytest.mark.parametrize("factory", _baseline_factories())
def test_valid_nan_is_rejected_but_invalid_nan_is_safe(factory) -> None:
    baseline = factory()
    state = baseline.initialize(torch.zeros(1, 1, 2))
    prior = baseline.prior(state)
    nan_measurement = torch.full((1, 1, 2), float("nan"))
    with pytest.raises(ValueError, match="valid measurement residuals must be finite"):
        baseline.update(
            prior,
            fcpg_measurement(nan_measurement, torch.ones(1, 1, dtype=torch.bool)),
        )
    safe = baseline.update(
        prior,
        fcpg_measurement(nan_measurement, torch.zeros(1, 1, dtype=torch.bool)),
    )
    assert bool(torch.isfinite(safe.posterior_residual_px).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("factory", _baseline_factories())
def test_measurement_device_mismatch_is_rejected(factory) -> None:
    baseline = factory()
    state = baseline.initialize(torch.zeros(1, 1, 2, device="cuda"))
    prior = baseline.prior(state)
    with pytest.raises(ValueError, match="devices must match"):
        baseline.update(
            prior, fcpg_measurement(torch.zeros(1, 1, 2, device="cpu"))
        )


def test_reference_contract_rejects_non_anchor_and_low_precision_shapes() -> None:
    baseline = HoldLastResidualBaseline()
    with pytest.raises(ValueError, match=r"\[B, K, 2\]"):
        baseline.initialize(torch.zeros(3, 2))
    with pytest.raises(TypeError, match="float32 or float64"):
        baseline.initialize(torch.zeros(1, 3, 2, dtype=torch.float16))


def test_kalman_state_uses_documented_dx_dy_vx_vy_order() -> None:
    baseline = LinearConstantVelocityKalmanBaseline(
        acceleration_variance=0.0
    )
    state = baseline.initialize(torch.zeros(1, 1, 2, dtype=torch.float64))
    state = LinearKalmanState(
        mean=torch.tensor([[[4.0, -3.0, 2.0, 1.5]]], dtype=torch.float64),
        covariance=state.covariance,
        has_valid_measurement=torch.ones(1, 1, dtype=torch.bool),
    )
    prior = baseline.prior(state)
    torch.testing.assert_close(
        prior.residual_px,
        torch.tensor([[[6.0, -1.5]]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
