"""Strictly causal, zero-training temporal baselines for PixelObserver.

Every baseline implements the same immutable transition API::

    state = baseline.initialize(template_residual_px)
    prediction = baseline.prior(state)
    correction = baseline.update(prediction, typed_fcpg_measurement)
    state = correction.state

The prior is state-only.  It cannot consume RGB, oracle geometry, or any logged
future field.  Measurements are admitted only by :meth:`update`, after the
prior for that time step has already been formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, TypeAlias, runtime_checkable

import torch

from fcpg.lewm_pixel_observer import fixed_gain_update


_FORBIDDEN_CONTEXT_FIELDS = frozenset(
    {
        "future_pixels",
        "future_rgb",
        "oracle_grounding",
        "q_oracle",
        "q_true",
        "target_pixels",
        "target_rgb",
        "target_true_extrinsic",
        "true_extrinsic",
    }
)


class TemporalMeasurementSource(str, Enum):
    """Typed provenance admitted by the formal causal baseline API."""

    FCPG_MEASUREMENT = "fcpg_measurement"
    ORACLE = "oracle"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TemporalMeasurement:
    """Pixel-residual measurement plus validity and explicit provenance."""

    residual_px: torch.Tensor
    valid: torch.Tensor | None
    source: TemporalMeasurementSource


def fcpg_measurement(
    residual_px: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> TemporalMeasurement:
    """Mark a residual produced by the deployed FCPG measurement path."""

    return TemporalMeasurement(
        residual_px=residual_px,
        valid=valid,
        source=TemporalMeasurementSource.FCPG_MEASUREMENT,
    )


def _validate_template(template: torch.Tensor) -> None:
    """Validate a shape/device/dtype template without reading its values."""

    if template.ndim != 3 or template.shape[-1] != 2:
        raise ValueError("template residual must have shape [B, K, 2]")
    if not template.is_floating_point():
        raise TypeError("template residual must use a floating point dtype")
    if template.dtype not in (torch.float32, torch.float64):
        raise TypeError("temporal baselines require float32 or float64 residuals")


def _validate_reference(reference: torch.Tensor) -> None:
    _validate_template(reference)
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("reference residual must contain only finite values")


def _validate_context(context: Mapping[str, object] | None) -> None:
    if context is None:
        return
    if not isinstance(context, Mapping):
        raise TypeError("baseline context must be a mapping")
    forbidden = sorted(_FORBIDDEN_CONTEXT_FIELDS.intersection(context))
    if forbidden:
        raise ValueError(
            "oracle or future observation fields must not enter a temporal "
            f"baseline: {forbidden}"
        )
    if context:
        raise ValueError(
            "temporal baseline transitions are state-only and accept no "
            f"external context fields: {sorted(str(name) for name in context)}"
        )


def _validate_measurement(
    measurement: TemporalMeasurement,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(measurement, TemporalMeasurement):
        raise TypeError("causal update requires a typed TemporalMeasurement")
    if measurement.source is not TemporalMeasurementSource.FCPG_MEASUREMENT:
        raise ValueError(
            "causal update accepts only FCPG_MEASUREMENT provenance; got "
            f"{measurement.source!r}"
        )
    residual = measurement.residual_px
    if not torch.is_tensor(residual):
        raise TypeError("measurement residual_px must be a tensor")
    if residual.shape != reference.shape:
        raise ValueError(
            "measurement residual must match the prior shape: "
            f"{tuple(residual.shape)} != {tuple(reference.shape)}"
        )
    if residual.dtype != reference.dtype:
        raise TypeError("measurement and prior residual dtypes must match")
    if residual.device != reference.device:
        raise ValueError("measurement and prior residual devices must match")
    measurement_valid = measurement.valid
    if measurement_valid is None:
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("unmasked measurement residuals must be finite")
        valid = torch.ones(
            reference.shape[:-1], dtype=torch.bool, device=reference.device
        )
        return residual, valid
    if not torch.is_tensor(measurement_valid):
        raise TypeError("measurement valid must be a tensor or None")
    if measurement_valid.shape != reference.shape[:-1]:
        raise ValueError(
            "measurement_valid must have shape "
            f"{tuple(reference.shape[:-1])}, got {tuple(measurement_valid.shape)}"
        )
    if measurement_valid.dtype != torch.bool:
        raise TypeError("measurement_valid must use torch.bool")
    if measurement_valid.device != reference.device:
        raise ValueError("measurement_valid and residual devices must match")
    valid_coordinates = measurement_valid.unsqueeze(-1).expand_as(residual)
    if bool((valid_coordinates & ~torch.isfinite(residual)).any()):
        raise ValueError("valid measurement residuals must be finite")
    return residual, measurement_valid


def _validate_prior_binding(
    prior: TemporalPrior,
    state_residual_px: torch.Tensor,
) -> None:
    _validate_reference(prior.residual_px)
    if (
        prior.residual_px.dtype != state_residual_px.dtype
        or prior.residual_px.device != state_residual_px.device
        or prior.residual_px.shape != state_residual_px.shape
    ):
        raise ValueError("temporal prior residual does not match its state")
    if not torch.equal(prior.residual_px, state_residual_px):
        raise ValueError("temporal prior residual does not match its state")


@dataclass(frozen=True)
class HoldLastState:
    residual_px: torch.Tensor
    has_valid_measurement: torch.Tensor


@dataclass(frozen=True)
class ConstantVelocityState:
    residual_px: torch.Tensor
    velocity_px_per_step: torch.Tensor
    last_valid_residual_px: torch.Tensor
    steps_since_valid: torch.Tensor
    has_valid_measurement: torch.Tensor


@dataclass(frozen=True)
class LinearKalmanState:
    """Per-anchor Gaussian state ordered as ``[dx, dy, vx, vy]``."""

    mean: torch.Tensor
    covariance: torch.Tensor
    has_valid_measurement: torch.Tensor


TemporalState: TypeAlias = HoldLastState | ConstantVelocityState | LinearKalmanState


@dataclass(frozen=True)
class TemporalPrior:
    residual_px: torch.Tensor
    state: TemporalState


@dataclass(frozen=True)
class TemporalUpdate:
    measurement_residual_px: torch.Tensor
    prior_residual_px: torch.Tensor
    innovation_px: torch.Tensor
    posterior_residual_px: torch.Tensor
    effective_gain: torch.Tensor
    state: TemporalState
    kalman_gain: torch.Tensor | None = None


@dataclass(frozen=True)
class TemporalSequenceResult:
    """Unified scored outputs after an explicitly executed warm-up prefix."""

    warmup_frames: int
    measurement_source: TemporalMeasurementSource
    measurement_residual_px: torch.Tensor
    measurement_valid: torch.Tensor
    prior_residual_px: torch.Tensor
    innovation_px: torch.Tensor
    posterior_residual_px: torch.Tensor
    effective_gain: torch.Tensor
    final_state: TemporalState
    kalman_gain: torch.Tensor | None = None


@runtime_checkable
class TemporalPixelBaseline(Protocol):
    """Common transition interface used by all zero-training baselines."""

    def initialize(
        self, template_residual_px: torch.Tensor
    ) -> TemporalState: ...

    def prior(
        self,
        state: TemporalState,
        *,
        context: Mapping[str, object] | None = None,
    ) -> TemporalPrior: ...

    def update(
        self,
        prior: TemporalPrior,
        measurement: TemporalMeasurement,
        *,
        context: Mapping[str, object] | None = None,
    ) -> TemporalUpdate: ...


def _gain_value(gain: float) -> float:
    value = float(gain)
    if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
        raise ValueError("baseline gain must be finite and lie in [0, 1]")
    return value


class HoldLastResidualBaseline:
    """Zero-order hold with an optional fixed-gain measurement correction."""

    def __init__(self, *, gain: float = 1.0) -> None:
        self.gain = _gain_value(gain)

    def initialize(
        self, template_residual_px: torch.Tensor
    ) -> HoldLastState:
        """Allocate zero state; template values are intentionally ignored."""

        _validate_template(template_residual_px)
        return HoldLastState(
            residual_px=torch.zeros_like(template_residual_px),
            has_valid_measurement=torch.zeros(
                template_residual_px.shape[:-1],
                dtype=torch.bool,
                device=template_residual_px.device,
            ),
        )

    def prior(
        self,
        state: TemporalState,
        *,
        context: Mapping[str, object] | None = None,
    ) -> TemporalPrior:
        _validate_context(context)
        if not isinstance(state, HoldLastState):
            raise TypeError("hold-last baseline requires HoldLastState")
        _validate_reference(state.residual_px)
        if state.has_valid_measurement.shape != state.residual_px.shape[:-1]:
            raise ValueError("hold-last validity shape is invalid")
        if state.has_valid_measurement.dtype != torch.bool:
            raise TypeError("hold-last validity must use torch.bool")
        if state.has_valid_measurement.device != state.residual_px.device:
            raise ValueError("hold-last state devices must match")
        return TemporalPrior(residual_px=state.residual_px, state=state)

    def update(
        self,
        prior: TemporalPrior,
        measurement: TemporalMeasurement,
        *,
        context: Mapping[str, object] | None = None,
    ) -> TemporalUpdate:
        _validate_context(context)
        if not isinstance(prior.state, HoldLastState):
            raise TypeError("hold-last baseline requires a HoldLastState prior")
        _validate_prior_binding(prior, prior.state.residual_px)
        measurement_residual_px, valid = _validate_measurement(
            measurement, prior.residual_px
        )
        innovation, posterior, effective_gain = fixed_gain_update(
            prior.residual_px,
            measurement_residual_px,
            self.gain,
            valid,
        )
        state = HoldLastState(
            residual_px=posterior,
            has_valid_measurement=prior.state.has_valid_measurement | valid,
        )
        return TemporalUpdate(
            measurement_residual_px=measurement_residual_px,
            prior_residual_px=prior.residual_px,
            innovation_px=innovation,
            posterior_residual_px=posterior,
            effective_gain=effective_gain,
            state=state,
        )


class ConstantVelocityResidualBaseline:
    """Clipped finite-difference velocity with causal missing-data propagation.

    A valid observation updates position through :func:`fixed_gain_update`.
    Velocity is the component-wise clipped, per-step displacement since the
    previous valid posterior.  Invalid observations retain the propagated prior
    and velocity; their numeric payload (including NaN) is never used.
    """

    def __init__(self, *, gain: float = 1.0, max_delta_px: float = 16.0) -> None:
        self.gain = _gain_value(gain)
        self.max_delta_px = float(max_delta_px)
        if (
            not torch.isfinite(torch.tensor(self.max_delta_px))
            or self.max_delta_px <= 0
        ):
            raise ValueError("max_delta_px must be finite and positive")

    def initialize(
        self, template_residual_px: torch.Tensor
    ) -> ConstantVelocityState:
        """Allocate zero state; template values are intentionally ignored."""

        _validate_template(template_residual_px)
        residual = torch.zeros_like(template_residual_px)
        anchor_shape = template_residual_px.shape[:-1]
        return ConstantVelocityState(
            residual_px=residual,
            velocity_px_per_step=torch.zeros_like(residual),
            last_valid_residual_px=torch.zeros_like(residual),
            steps_since_valid=torch.zeros(
                anchor_shape, dtype=torch.long, device=template_residual_px.device
            ),
            has_valid_measurement=torch.zeros(
                anchor_shape, dtype=torch.bool, device=template_residual_px.device
            ),
        )

    @staticmethod
    def _validate_state(state: ConstantVelocityState) -> None:
        _validate_reference(state.residual_px)
        if (
            state.velocity_px_per_step.shape != state.residual_px.shape
            or state.last_valid_residual_px.shape != state.residual_px.shape
        ):
            raise ValueError("constant-velocity residual state shapes are invalid")
        anchor_shape = state.residual_px.shape[:-1]
        if (
            state.steps_since_valid.shape != anchor_shape
            or state.steps_since_valid.dtype != torch.long
            or state.has_valid_measurement.shape != anchor_shape
            or state.has_valid_measurement.dtype != torch.bool
        ):
            raise ValueError("constant-velocity metadata state shapes are invalid")
        for tensor in (
            state.velocity_px_per_step,
            state.last_valid_residual_px,
            state.steps_since_valid,
            state.has_valid_measurement,
        ):
            if tensor.device != state.residual_px.device:
                raise ValueError("constant-velocity state devices must match")
        if state.velocity_px_per_step.dtype != state.residual_px.dtype:
            raise TypeError("constant-velocity state dtypes must match")
        if state.last_valid_residual_px.dtype != state.residual_px.dtype:
            raise TypeError("constant-velocity state dtypes must match")
        if not bool(torch.isfinite(state.velocity_px_per_step).all()) or not bool(
            torch.isfinite(state.last_valid_residual_px).all()
        ):
            raise ValueError("constant-velocity state must be finite")
        if bool(torch.any(state.steps_since_valid < 0)):
            raise ValueError("steps_since_valid must be non-negative")

    def prior(
        self,
        state: TemporalState,
        *,
        context: Mapping[str, object] | None = None,
    ) -> TemporalPrior:
        _validate_context(context)
        if not isinstance(state, ConstantVelocityState):
            raise TypeError("constant-velocity baseline requires ConstantVelocityState")
        self._validate_state(state)
        propagate = state.has_valid_measurement.unsqueeze(-1)
        predicted = torch.where(
            propagate,
            state.residual_px + state.velocity_px_per_step,
            state.residual_px,
        )
        predicted_state = ConstantVelocityState(
            residual_px=predicted,
            velocity_px_per_step=state.velocity_px_per_step,
            last_valid_residual_px=state.last_valid_residual_px,
            steps_since_valid=torch.where(
                state.has_valid_measurement,
                state.steps_since_valid + 1,
                state.steps_since_valid,
            ),
            has_valid_measurement=state.has_valid_measurement,
        )
        return TemporalPrior(residual_px=predicted, state=predicted_state)

    def update(
        self,
        prior: TemporalPrior,
        measurement: TemporalMeasurement,
        *,
        context: Mapping[str, object] | None = None,
    ) -> TemporalUpdate:
        _validate_context(context)
        if not isinstance(prior.state, ConstantVelocityState):
            raise TypeError(
                "constant-velocity baseline requires a ConstantVelocityState prior"
            )
        self._validate_state(prior.state)
        _validate_prior_binding(prior, prior.state.residual_px)
        measurement_residual_px, valid = _validate_measurement(
            measurement, prior.residual_px
        )
        innovation, posterior, effective_gain = fixed_gain_update(
            prior.residual_px,
            measurement_residual_px,
            self.gain,
            valid,
        )
        elapsed = prior.state.steps_since_valid.clamp_min(1).unsqueeze(-1)
        observed_velocity = (
            posterior - prior.state.last_valid_residual_px
        ) / elapsed.to(dtype=posterior.dtype)
        observed_velocity = observed_velocity.clamp(
            min=-self.max_delta_px, max=self.max_delta_px
        )
        can_estimate = valid & prior.state.has_valid_measurement
        velocity = torch.where(
            can_estimate.unsqueeze(-1),
            observed_velocity,
            prior.state.velocity_px_per_step,
        )
        state = ConstantVelocityState(
            residual_px=posterior,
            velocity_px_per_step=velocity,
            last_valid_residual_px=torch.where(
                valid.unsqueeze(-1),
                posterior,
                prior.state.last_valid_residual_px,
            ),
            steps_since_valid=torch.where(
                valid,
                torch.zeros_like(prior.state.steps_since_valid),
                prior.state.steps_since_valid,
            ),
            has_valid_measurement=prior.state.has_valid_measurement | valid,
        )
        return TemporalUpdate(
            measurement_residual_px=measurement_residual_px,
            prior_residual_px=prior.residual_px,
            innovation_px=innovation,
            posterior_residual_px=posterior,
            effective_gain=effective_gain,
            state=state,
        )


class LinearConstantVelocityKalmanBaseline:
    """Independent per-anchor linear constant-velocity Kalman filters."""

    def __init__(
        self,
        *,
        dt: float = 1.0,
        acceleration_variance: float = 1.0,
        measurement_variance: float = 4.0,
        initial_position_variance: float = 16.0,
        initial_velocity_variance: float = 16.0,
        psd_tolerance: float = 1e-6,
    ) -> None:
        values = {
            "dt": float(dt),
            "acceleration_variance": float(acceleration_variance),
            "measurement_variance": float(measurement_variance),
            "initial_position_variance": float(initial_position_variance),
            "initial_velocity_variance": float(initial_velocity_variance),
            "psd_tolerance": float(psd_tolerance),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise ValueError("Kalman baseline parameters must be finite")
        if values["dt"] <= 0.0:
            raise ValueError("dt must be positive")
        if values["acceleration_variance"] < 0.0:
            raise ValueError("acceleration_variance must be non-negative")
        if values["measurement_variance"] <= 0.0:
            raise ValueError("measurement_variance must be positive")
        if values["initial_position_variance"] <= 0.0:
            raise ValueError("initial_position_variance must be positive")
        if values["initial_velocity_variance"] <= 0.0:
            raise ValueError("initial_velocity_variance must be positive")
        if values["psd_tolerance"] < 0.0:
            raise ValueError("psd_tolerance must be non-negative")
        self.dt = values["dt"]
        self.acceleration_variance = values["acceleration_variance"]
        self.measurement_variance = values["measurement_variance"]
        self.initial_position_variance = values["initial_position_variance"]
        self.initial_velocity_variance = values["initial_velocity_variance"]
        self.psd_tolerance = values["psd_tolerance"]

    @staticmethod
    def _system_matrices(
        reference: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition = reference.new_tensor(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        observation = reference.new_tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        return transition, observation

    def _process_noise(self, reference: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        return self.acceleration_variance * reference.new_tensor(
            [
                [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
                [dt**3 / 2.0, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2.0, 0.0, dt**2],
            ]
        )

    def initialize(
        self, template_residual_px: torch.Tensor
    ) -> LinearKalmanState:
        """Allocate zero state; template values are intentionally ignored."""

        _validate_template(template_residual_px)
        batch, anchors, _ = template_residual_px.shape
        mean = template_residual_px.new_zeros(batch, anchors, 4)
        diagonal = template_residual_px.new_tensor(
            [
                self.initial_position_variance,
                self.initial_position_variance,
                self.initial_velocity_variance,
                self.initial_velocity_variance,
            ]
        )
        covariance = torch.diag(diagonal).expand(batch, anchors, 4, 4).clone()
        return LinearKalmanState(
            mean=mean,
            covariance=covariance,
            has_valid_measurement=torch.zeros(
                batch, anchors, dtype=torch.bool, device=template_residual_px.device
            ),
        )

    def _validate_state(self, state: LinearKalmanState) -> None:
        if state.mean.ndim != 3 or state.mean.shape[-1] != 4:
            raise ValueError("Kalman mean must have shape [B, K, 4]")
        expected_covariance = (*state.mean.shape[:-1], 4, 4)
        if state.covariance.shape != expected_covariance:
            raise ValueError("Kalman covariance must have shape [B, K, 4, 4]")
        if state.has_valid_measurement.shape != state.mean.shape[:-1]:
            raise ValueError("Kalman validity must have shape [B, K]")
        if state.mean.dtype not in (torch.float32, torch.float64):
            raise TypeError("Kalman state requires float32 or float64")
        if state.covariance.dtype != state.mean.dtype:
            raise TypeError("Kalman mean and covariance dtypes must match")
        if state.has_valid_measurement.dtype != torch.bool:
            raise TypeError("Kalman validity must use torch.bool")
        if (
            state.covariance.device != state.mean.device
            or state.has_valid_measurement.device != state.mean.device
        ):
            raise ValueError("Kalman state devices must match")
        if not bool(torch.isfinite(state.mean).all()) or not bool(
            torch.isfinite(state.covariance).all()
        ):
            raise ValueError("Kalman state must contain only finite values")
        symmetry_error = torch.max(
            torch.abs(state.covariance - state.covariance.transpose(-1, -2))
        )
        if float(symmetry_error) > self.psd_tolerance:
            raise ValueError("Kalman covariance must be symmetric")
        minimum_eigenvalue = torch.linalg.eigvalsh(state.covariance).amin()
        if float(minimum_eigenvalue) < -self.psd_tolerance:
            raise ValueError("Kalman covariance must be positive semidefinite")

    def prior(
        self,
        state: TemporalState,
        *,
        context: Mapping[str, object] | None = None,
    ) -> TemporalPrior:
        _validate_context(context)
        if not isinstance(state, LinearKalmanState):
            raise TypeError("Kalman baseline requires LinearKalmanState")
        self._validate_state(state)
        transition, _ = self._system_matrices(state.mean, self.dt)
        propagated_mean = torch.matmul(
            transition, state.mean.unsqueeze(-1)
        ).squeeze(-1)
        propagated_covariance = (
            transition @ state.covariance @ transition.transpose(-1, -2)
            + self._process_noise(state.mean)
        )
        propagate = state.has_valid_measurement.unsqueeze(-1)
        mean = torch.where(propagate, propagated_mean, state.mean)
        covariance = torch.where(
            propagate.unsqueeze(-1),
            propagated_covariance,
            state.covariance,
        )
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        predicted_state = LinearKalmanState(
            mean=mean,
            covariance=covariance,
            has_valid_measurement=state.has_valid_measurement,
        )
        self._validate_state(predicted_state)
        return TemporalPrior(residual_px=mean[..., :2], state=predicted_state)

    def update(
        self,
        prior: TemporalPrior,
        measurement: TemporalMeasurement,
        *,
        context: Mapping[str, object] | None = None,
    ) -> TemporalUpdate:
        _validate_context(context)
        if not isinstance(prior.state, LinearKalmanState):
            raise TypeError("Kalman baseline requires a LinearKalmanState prior")
        self._validate_state(prior.state)
        _validate_prior_binding(prior, prior.state.mean[..., :2])
        measurement_residual_px, valid = _validate_measurement(
            measurement, prior.residual_px
        )
        _, observation = self._system_matrices(prior.state.mean, self.dt)
        covariance = prior.state.covariance
        predicted_measurement = prior.state.mean[..., :2]
        safe_measurement = torch.where(
            valid.unsqueeze(-1), measurement_residual_px, predicted_measurement
        )
        innovation = safe_measurement - predicted_measurement
        measurement_noise = self.measurement_variance * torch.eye(
            2, dtype=covariance.dtype, device=covariance.device
        )
        covariance_observation_t = covariance @ observation.transpose(-1, -2)
        innovation_covariance = (
            observation @ covariance_observation_t + measurement_noise
        )
        kalman_gain = torch.linalg.solve(
            innovation_covariance,
            covariance_observation_t.transpose(-1, -2),
        ).transpose(-1, -2)
        kalman_gain = torch.where(
            valid[..., None, None], kalman_gain, torch.zeros_like(kalman_gain)
        )
        posterior_mean_candidate = prior.state.mean + torch.matmul(
            kalman_gain, innovation.unsqueeze(-1)
        ).squeeze(-1)
        identity = torch.eye(4, dtype=covariance.dtype, device=covariance.device)
        residual_operator = identity - kalman_gain @ observation
        posterior_covariance_candidate = (
            residual_operator
            @ covariance
            @ residual_operator.transpose(-1, -2)
            + kalman_gain @ measurement_noise @ kalman_gain.transpose(-1, -2)
        )
        posterior_mean = torch.where(
            valid.unsqueeze(-1), posterior_mean_candidate, prior.state.mean
        )
        posterior_covariance = torch.where(
            valid[..., None, None],
            posterior_covariance_candidate,
            covariance,
        )
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.transpose(-1, -2)
        )
        state = LinearKalmanState(
            mean=posterior_mean,
            covariance=posterior_covariance,
            has_valid_measurement=prior.state.has_valid_measurement | valid,
        )
        self._validate_state(state)
        effective_gain = torch.diagonal(
            kalman_gain[..., :2, :], dim1=-2, dim2=-1
        )
        return TemporalUpdate(
            measurement_residual_px=measurement_residual_px,
            prior_residual_px=prior.residual_px,
            innovation_px=innovation,
            posterior_residual_px=posterior_mean[..., :2],
            effective_gain=effective_gain,
            state=state,
            kalman_gain=kalman_gain,
        )


def _validate_sequence_measurement(
    measurement: TemporalMeasurement,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(measurement, TemporalMeasurement):
        raise TypeError("sequence runner requires a typed TemporalMeasurement")
    if measurement.source is not TemporalMeasurementSource.FCPG_MEASUREMENT:
        raise ValueError(
            "sequence runner accepts only FCPG_MEASUREMENT provenance; got "
            f"{measurement.source!r}"
        )
    residual = measurement.residual_px
    if not torch.is_tensor(residual):
        raise TypeError("sequence residual_px must be a tensor")
    if residual.ndim != 4 or residual.shape[-1] != 2:
        raise ValueError("sequence residual must have shape [B, T, K, 2]")
    if min(residual.shape[:3]) <= 0:
        raise ValueError("sequence batch, time, and anchor dimensions must be positive")
    if not residual.is_floating_point() or residual.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise TypeError("sequence residual must use float32 or float64")
    valid = measurement.valid
    if valid is None:
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("unmasked sequence residuals must be finite")
        valid = torch.ones(
            residual.shape[:-1], dtype=torch.bool, device=residual.device
        )
    else:
        if not torch.is_tensor(valid):
            raise TypeError("sequence valid must be a tensor or None")
        if valid.shape != residual.shape[:-1]:
            raise ValueError("sequence valid must have shape [B, T, K]")
        if valid.dtype != torch.bool:
            raise TypeError("sequence valid must use torch.bool")
        if valid.device != residual.device:
            raise ValueError("sequence residual and valid devices must match")
        valid_coordinates = valid.unsqueeze(-1).expand_as(residual)
        if bool((valid_coordinates & ~torch.isfinite(residual)).any()):
            raise ValueError("valid sequence residuals must be finite")
    return residual, valid


def run_temporal_baseline(
    baseline: TemporalPixelBaseline,
    measurement: TemporalMeasurement,
    *,
    warmup_frames: int,
) -> TemporalSequenceResult:
    """Reset and run one causal baseline over a complete observed sequence.

    Every call allocates a fresh zero state from a shape/device/dtype template.
    Each of the first ``warmup_frames`` is processed by ``prior`` followed by
    ``update`` exactly once; warm-up outputs are not scored.  For later frames,
    the prior is always formed before the current FCPG measurement is sliced
    and admitted to ``update``.  No mapping or evaluator-only field is accepted.
    """

    if not isinstance(baseline, TemporalPixelBaseline):
        raise TypeError("baseline must implement TemporalPixelBaseline")
    residual, valid = _validate_sequence_measurement(measurement)
    frames = residual.shape[1]
    if isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int):
        raise TypeError("warmup_frames must be an integer")
    if warmup_frames < 1 or warmup_frames >= frames:
        raise ValueError("warmup_frames must satisfy 1 <= H < T")

    state = baseline.initialize(residual[:, 0])
    prior_steps: list[torch.Tensor] = []
    innovation_steps: list[torch.Tensor] = []
    posterior_steps: list[torch.Tensor] = []
    gain_steps: list[torch.Tensor] = []
    kalman_steps: list[torch.Tensor | None] = []

    for index in range(frames):
        prior = baseline.prior(state)
        current_measurement = fcpg_measurement(
            residual[:, index], valid[:, index]
        )
        update = baseline.update(prior, current_measurement)
        state = update.state
        if index < warmup_frames:
            continue
        prior_steps.append(update.prior_residual_px)
        innovation_steps.append(update.innovation_px)
        posterior_steps.append(update.posterior_residual_px)
        gain_steps.append(update.effective_gain)
        kalman_steps.append(update.kalman_gain)

    if any(value is None for value in kalman_steps):
        if not all(value is None for value in kalman_steps):
            raise RuntimeError("baseline returned inconsistent Kalman gain fields")
        kalman_gain = None
    else:
        kalman_gain = torch.stack(
            [value for value in kalman_steps if value is not None], dim=1
        )
    target = slice(warmup_frames, None)
    return TemporalSequenceResult(
        warmup_frames=warmup_frames,
        measurement_source=measurement.source,
        measurement_residual_px=residual[:, target],
        measurement_valid=valid[:, target],
        prior_residual_px=torch.stack(prior_steps, dim=1),
        innovation_px=torch.stack(innovation_steps, dim=1),
        posterior_residual_px=torch.stack(posterior_steps, dim=1),
        effective_gain=torch.stack(gain_steps, dim=1),
        final_state=state,
        kalman_gain=kalman_gain,
    )


__all__ = [
    "ConstantVelocityResidualBaseline",
    "ConstantVelocityState",
    "HoldLastResidualBaseline",
    "HoldLastState",
    "LinearConstantVelocityKalmanBaseline",
    "LinearKalmanState",
    "TemporalMeasurement",
    "TemporalMeasurementSource",
    "TemporalPixelBaseline",
    "TemporalPrior",
    "TemporalSequenceResult",
    "TemporalState",
    "TemporalUpdate",
    "fcpg_measurement",
    "run_temporal_baseline",
]
