"""Second-order pixel-residual prior for the LeWM/FCPG observer.

This module is intentionally separate from :mod:`fcpg.lewm_pixel_observer`.
The first observer and its frozen confirmatory protocol remain unchanged.

The v2 prior uses the explicitly observable state ``[residual, velocity]`` and
known kinematics on both sides of the transition. It learns a bounded discrete
velocity innovation, not a physical acceleration. The semi-implicit update
``v' = v + dv; r' = r + v'`` can damp a spurious velocity after a camera
step without forcing the residual position to move. A zero-initialized adapter
is exactly the constant-velocity baseline:

``r_prior[t+1] = r_posterior[t] + velocity[t]``.

Target-frame RGB, FCPG corrections, and oracle coordinates are deliberately
absent from the prior API.  They may be supplied later to the separate
supervision builder, after the causal prior has already been computed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn


PIXEL_OBSERVER_V2_CHECKPOINT_TYPE = "fcpg_lewm_pixel_observer_v2"
PAIRED_DEPLOYMENT_TARGET_MODE = "paired_deployment"
DEPLOYMENT_TARGET_MODE = PAIRED_DEPLOYMENT_TARGET_MODE
SINGLE_MEASUREMENT_BASELINE_TARGET_MODE = "single_measurement_baseline"
ORACLE_CEILING_TARGET_MODE = "oracle_ceiling"

_PRIOR_INPUT_NAMES = frozenset(
    {
        "predicted_embedding",
        "posterior_residual_px",
        "residual_velocity_px",
        "previous_q_nominal_px",
        "current_q_nominal_px",
        "previous_keypoints_xyz",
        "current_keypoints_xyz",
        "previous_nominal_valid",
        "current_nominal_valid",
        "previous_measurement_valid",
        "measurement_age_steps",
    }
)
_FORBIDDEN_PRIOR_INPUT_NAMES = frozenset(
    {
        "rgb",
        "pixels",
        "image",
        "current_rgb",
        "current_pixels",
        "q_corrected",
        "q_grounded",
        "current_q_corrected",
        "q_true",
        "q_oracle",
        "oracle_grounding",
        "true_extrinsic",
        "extrinsic",
        "frame_id",
        "time",
        "timestamp",
    }
)


@dataclass(frozen=True)
class PixelObserverV2Prior:
    """One causal second-order residual prediction."""

    residual_px: torch.Tensor
    velocity_px: torch.Tensor
    velocity_innovation_px: torch.Tensor


@dataclass(frozen=True)
class PixelObserverV2Supervision:
    """A target constructed only after the causal prior is available."""

    target_residual_px: torch.Tensor
    valid: torch.Tensor
    target_mode: str
    diagnostic_only: bool
    deployment_allowed: bool
    target_contract: str


@dataclass(frozen=True)
class PixelObserverV2PairedSupervision:
    """Independent A/B measurements used as cross-view deployment targets."""

    target_for_a_history_px: torch.Tensor
    target_for_b_history_px: torch.Tensor
    consensus_target_px: torch.Tensor
    valid: torch.Tensor
    target_mode: str = PAIRED_DEPLOYMENT_TARGET_MODE
    diagnostic_only: bool = False
    deployment_allowed: bool = True
    target_contract: str = "independent_a_b_mutual_supervision"


def _require_floating_finite(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


class PixelResidualVelocityInnovationAdapter(nn.Module):
    """Predict a bounded velocity innovation around a constant-velocity prior.

    ``predicted_embedding`` is always detached.  Setting
    ``use_predicted_latent=False`` preserves the exact architecture and
    parameter count but replaces that input with zeros, providing a matched
    geometry-only baseline.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        num_keypoints: int,
        image_size: tuple[int, int],
        hidden_dim: int = 256,
        max_velocity_innovation_px: float = 1.0,
        residual_scale_px: float = 8.0,
        velocity_scale_px: float = 1.0,
        use_predicted_latent: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or num_keypoints <= 0 or hidden_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        width, height = (int(image_size[0]), int(image_size[1]))
        if width <= 1 or height <= 1:
            raise ValueError("image_size dimensions must be larger than one")
        scales = (max_velocity_innovation_px, residual_scale_px, velocity_scale_px)
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0 for value in scales
        ):
            raise ValueError("adapter pixel scales must be positive")

        self.latent_dim = int(latent_dim)
        self.num_keypoints = int(num_keypoints)
        self.image_size = (width, height)
        self.hidden_dim = int(hidden_dim)
        self.max_velocity_innovation_px = float(max_velocity_innovation_px)
        self.residual_scale_px = float(residual_scale_px)
        self.velocity_scale_px = float(velocity_scale_px)
        self.use_predicted_latent = bool(use_predicted_latent)

        self.anchor_embedding = nn.Parameter(
            torch.empty(1, self.num_keypoints, self.latent_dim)
        )
        # latent + state(r, v) + previous/current q + previous/current XYZ
        # + previous/current nominal-valid + previous measurement validity
        # + normalized age since the last valid measurement.
        input_dim = self.latent_dim + 4 + 4 + 6 + 2 + 2
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.normal_(self.anchor_embedding, std=0.02)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.register_buffer("xyz_mean", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("xyz_std", torch.ones(3, dtype=torch.float32))

    def set_geometry_normalization(
        self,
        *,
        xyz_mean: torch.Tensor | list[float],
        xyz_std: torch.Tensor | list[float],
    ) -> None:
        mean = torch.as_tensor(
            xyz_mean, dtype=self.xyz_mean.dtype, device=self.xyz_mean.device
        )
        std = torch.as_tensor(
            xyz_std, dtype=self.xyz_std.dtype, device=self.xyz_std.device
        )
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("xyz_mean and xyz_std must each have shape [3]")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("geometry normalization must be finite")
        if bool(torch.any(std <= 0.0)):
            raise ValueError("xyz_std must be strictly positive")
        with torch.no_grad():
            self.xyz_mean.copy_(mean)
            self.xyz_std.copy_(std)

    def export_config(self) -> dict[str, Any]:
        """Return all behavior-affecting settings for checkpoint provenance."""

        return {
            "latent_dim": self.latent_dim,
            "num_keypoints": self.num_keypoints,
            "image_size": list(self.image_size),
            "hidden_dim": self.hidden_dim,
            "max_velocity_innovation_px": self.max_velocity_innovation_px,
            "residual_scale_px": self.residual_scale_px,
            "velocity_scale_px": self.velocity_scale_px,
            "use_predicted_latent": self.use_predicted_latent,
            "dynamics_state": "posterior_residual_and_velocity",
            "learned_output": "bounded_velocity_innovation",
            "zero_initialization_baseline": "constant_velocity",
            "measurement_age_contract": "min(sampled_steps_since_valid,H)/H",
        }

    def _normalized_nominal(self, value: torch.Tensor) -> torch.Tensor:
        width, height = self.image_size
        scale = value.new_tensor((float(width - 1), float(height - 1)))
        return 2.0 * value / scale - 1.0

    def _validate_inputs(self, inputs: Mapping[str, torch.Tensor]) -> int:
        predicted = inputs["predicted_embedding"]
        if predicted.ndim != 2:
            raise ValueError("predicted_embedding must have shape [B, D]")
        batch_size = predicted.shape[0]
        expected_residual = (batch_size, self.num_keypoints, 2)
        expected_xyz = (batch_size, self.num_keypoints, 3)
        expected_valid = (batch_size, self.num_keypoints)
        if predicted.shape != (batch_size, self.latent_dim):
            raise ValueError(
                "predicted_embedding must have shape "
                f"{(batch_size, self.latent_dim)}, got {tuple(predicted.shape)}"
            )
        for name in (
            "posterior_residual_px",
            "residual_velocity_px",
            "previous_q_nominal_px",
            "current_q_nominal_px",
        ):
            value = inputs[name]
            if value.shape != expected_residual:
                raise ValueError(
                    f"{name} must have shape {expected_residual}, got "
                    f"{tuple(value.shape)}"
                )
        for name in ("previous_keypoints_xyz", "current_keypoints_xyz"):
            value = inputs[name]
            if value.shape != expected_xyz:
                raise ValueError(
                    f"{name} must have shape {expected_xyz}, got {tuple(value.shape)}"
                )
        for name in (
            "previous_nominal_valid",
            "current_nominal_valid",
            "previous_measurement_valid",
        ):
            value = inputs[name]
            if value.shape != expected_valid:
                raise ValueError(
                    f"{name} must have shape {expected_valid}, got {tuple(value.shape)}"
                )
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must have boolean dtype")
        for name in (
            "predicted_embedding",
            "posterior_residual_px",
            "residual_velocity_px",
            "previous_q_nominal_px",
            "current_q_nominal_px",
            "previous_keypoints_xyz",
            "current_keypoints_xyz",
            "measurement_age_steps",
        ):
            _require_floating_finite(name, inputs[name])
            if inputs[name].dtype != predicted.dtype:
                raise TypeError(
                    f"{name} must use predicted_embedding dtype {predicted.dtype}"
                )
            if inputs[name].device != predicted.device:
                raise ValueError(
                    f"{name} must use predicted_embedding device {predicted.device}"
                )
        if predicted.dtype != self.anchor_embedding.dtype:
            raise TypeError(
                "predicted_embedding dtype must match the adapter parameter dtype"
            )
        if predicted.device != self.anchor_embedding.device:
            raise ValueError(
                "predicted_embedding device must match the adapter parameter device"
            )
        age = inputs["measurement_age_steps"]
        if age.shape != expected_valid:
            raise ValueError(
                "measurement_age_steps must have shape "
                f"{expected_valid}, got {tuple(age.shape)}"
            )
        if bool(((age < 0.0) | (age > 1.0)).any()):
            raise ValueError("measurement_age_steps must be normalized to [0,1]")
        for name in (
            "previous_nominal_valid",
            "current_nominal_valid",
            "previous_measurement_valid",
        ):
            if inputs[name].device != predicted.device:
                raise ValueError(
                    f"{name} must use predicted_embedding device {predicted.device}"
                )
        return batch_size

    def predict_prior(self, inputs: Mapping[str, torch.Tensor]) -> PixelObserverV2Prior:
        """Predict from a closed, observation-free causal input mapping."""

        names = frozenset(inputs)
        forbidden = sorted(names & _FORBIDDEN_PRIOR_INPUT_NAMES)
        if forbidden:
            raise ValueError(
                "current visual/correction/oracle values are forbidden in the "
                f"v2 prior API: {forbidden}"
            )
        missing = sorted(_PRIOR_INPUT_NAMES - names)
        unknown = sorted(names - _PRIOR_INPUT_NAMES)
        if missing or unknown:
            raise ValueError(
                f"invalid v2 prior inputs; missing={missing}, unknown={unknown}"
            )
        self._validate_inputs(inputs)

        predicted = inputs["predicted_embedding"].detach()
        if not self.use_predicted_latent:
            predicted = torch.zeros_like(predicted).detach()
        latent = predicted.unsqueeze(1) + self.anchor_embedding
        previous_xyz = (
            inputs["previous_keypoints_xyz"] - self.xyz_mean
        ) / self.xyz_std
        current_xyz = (
            inputs["current_keypoints_xyz"] - self.xyz_mean
        ) / self.xyz_std
        features = torch.cat(
            (
                latent,
                inputs["posterior_residual_px"] / self.residual_scale_px,
                inputs["residual_velocity_px"] / self.velocity_scale_px,
                self._normalized_nominal(inputs["previous_q_nominal_px"]),
                self._normalized_nominal(inputs["current_q_nominal_px"]),
                previous_xyz,
                current_xyz,
                inputs["previous_nominal_valid"]
                .to(dtype=predicted.dtype)
                .unsqueeze(-1),
                inputs["current_nominal_valid"]
                .to(dtype=predicted.dtype)
                .unsqueeze(-1),
                inputs["previous_measurement_valid"]
                .to(dtype=predicted.dtype)
                .unsqueeze(-1),
                inputs["measurement_age_steps"].unsqueeze(-1),
            ),
            dim=-1,
        )
        raw_velocity_innovation = self.max_velocity_innovation_px * torch.tanh(
            self.net(features)
        )
        velocity_innovation = torch.where(
            inputs["current_nominal_valid"].unsqueeze(-1),
            raw_velocity_innovation,
            torch.zeros_like(raw_velocity_innovation),
        )
        velocity = inputs["residual_velocity_px"] + velocity_innovation
        residual = inputs["posterior_residual_px"] + velocity
        return PixelObserverV2Prior(
            residual_px=residual,
            velocity_px=velocity,
            velocity_innovation_px=velocity_innovation,
        )

    def forward(
        self,
        *,
        predicted_embedding: torch.Tensor,
        posterior_residual_px: torch.Tensor,
        residual_velocity_px: torch.Tensor,
        previous_q_nominal_px: torch.Tensor,
        current_q_nominal_px: torch.Tensor,
        previous_keypoints_xyz: torch.Tensor,
        current_keypoints_xyz: torch.Tensor,
        previous_nominal_valid: torch.Tensor,
        current_nominal_valid: torch.Tensor,
        previous_measurement_valid: torch.Tensor,
        measurement_age_steps: torch.Tensor,
    ) -> PixelObserverV2Prior:
        return self.predict_prior(
            {
                "predicted_embedding": predicted_embedding,
                "posterior_residual_px": posterior_residual_px,
                "residual_velocity_px": residual_velocity_px,
                "previous_q_nominal_px": previous_q_nominal_px,
                "current_q_nominal_px": current_q_nominal_px,
                "previous_keypoints_xyz": previous_keypoints_xyz,
                "current_keypoints_xyz": current_keypoints_xyz,
                "previous_nominal_valid": previous_nominal_valid,
                "current_nominal_valid": current_nominal_valid,
                "previous_measurement_valid": previous_measurement_valid,
                "measurement_age_steps": measurement_age_steps,
            }
        )


def build_pixel_observer_v2_supervision(
    *,
    q_nominal_px: torch.Tensor,
    nominal_valid: torch.Tensor,
    target_mode: str,
    q_corrected_px: torch.Tensor | None = None,
    q_true_px: torch.Tensor | None = None,
    prior_supervision_valid: torch.Tensor | None = None,
) -> PixelObserverV2Supervision:
    """Build a single-measurement baseline or oracle diagnostic target.

    ``oracle_ceiling`` is diagnostic-only and its metadata explicitly forbids
    deployment.  A known jump frame can be excluded by setting
    ``prior_supervision_valid=False`` without exposing the jump schedule, frame
    index, or camera extrinsic to the adapter.
    """

    if q_nominal_px.ndim < 3 or q_nominal_px.shape[-1] != 2:
        raise ValueError("q_nominal_px must end in [K, 2]")
    if nominal_valid.shape != q_nominal_px.shape[:-1]:
        raise ValueError("nominal_valid must match q_nominal_px without coordinate axis")
    if nominal_valid.device != q_nominal_px.device:
        raise ValueError("nominal_valid must match q_nominal_px device")
    if nominal_valid.dtype != torch.bool:
        raise TypeError("nominal_valid must have boolean dtype")
    _require_floating_finite("q_nominal_px", q_nominal_px)

    if target_mode == PAIRED_DEPLOYMENT_TARGET_MODE:
        raise ValueError(
            "paired_deployment requires "
            "build_pixel_observer_v2_paired_supervision"
        )
    if target_mode == SINGLE_MEASUREMENT_BASELINE_TARGET_MODE:
        if q_true_px is not None:
            raise ValueError("single-measurement supervision must not receive q_true_px")
        if q_corrected_px is None:
            raise ValueError("single-measurement supervision requires q_corrected_px")
        target_points = q_corrected_px
        diagnostic_only = True
        target_contract = "single_noisy_measurement_baseline"
    elif target_mode == ORACLE_CEILING_TARGET_MODE:
        if q_corrected_px is not None:
            raise ValueError("oracle_ceiling must isolate q_true from FCPG targets")
        if q_true_px is None:
            raise ValueError("oracle_ceiling supervision requires q_true_px")
        target_points = q_true_px
        diagnostic_only = True
        target_contract = "oracle_true_history_and_target_ceiling_v1"
    else:
        raise ValueError(
            "target_mode must be 'single_measurement_baseline' or "
            "diagnostic-only 'oracle_ceiling'"
        )
    if target_points.shape != q_nominal_px.shape:
        raise ValueError("supervision target points must match q_nominal_px")
    if target_points.dtype != q_nominal_px.dtype:
        raise TypeError("supervision target points must match q_nominal_px dtype")
    if target_points.device != q_nominal_px.device:
        raise ValueError("supervision target points must match q_nominal_px device")

    valid = nominal_valid
    if prior_supervision_valid is not None:
        if prior_supervision_valid.shape != nominal_valid.shape:
            raise ValueError("prior_supervision_valid must match nominal_valid")
        if prior_supervision_valid.dtype != torch.bool:
            raise TypeError("prior_supervision_valid must have boolean dtype")
        if prior_supervision_valid.device != q_nominal_px.device:
            raise ValueError("prior_supervision_valid must match q_nominal_px device")
        valid = valid & prior_supervision_valid
    selected = valid.unsqueeze(-1).expand_as(target_points)
    if not target_points.is_floating_point():
        raise TypeError("supervision target points must use floating point dtype")
    if bool((selected & ~torch.isfinite(target_points)).any()):
        raise ValueError("valid supervision target points must be finite")
    target_residual = torch.where(
        selected,
        target_points - q_nominal_px,
        torch.zeros_like(q_nominal_px),
    )
    return PixelObserverV2Supervision(
        target_residual_px=target_residual,
        valid=valid,
        target_mode=target_mode,
        diagnostic_only=diagnostic_only,
        deployment_allowed=False,
        target_contract=target_contract,
    )


def build_pixel_observer_v2_paired_supervision(
    *,
    q_nominal_px: torch.Tensor,
    nominal_valid: torch.Tensor,
    q_corrected_a_px: torch.Tensor,
    q_corrected_b_px: torch.Tensor,
    prior_supervision_valid: torch.Tensor | None = None,
) -> PixelObserverV2PairedSupervision:
    """Build formal deployment supervision from independent A/B measurements.

    A history is supervised by B and B history by A.  The consensus target is
    exposed for evaluation only; training should use the two cross-view terms
    via :func:`paired_mutual_huber_pixel_loss`.  This contract prevents a
    single jittery FCPG measurement from supervising its own prediction.
    """

    if q_nominal_px.ndim < 3 or q_nominal_px.shape[-1] != 2:
        raise ValueError("q_nominal_px must end in [K, 2]")
    if nominal_valid.shape != q_nominal_px.shape[:-1]:
        raise ValueError("nominal_valid must match q_nominal_px without coordinate axis")
    if nominal_valid.dtype != torch.bool:
        raise TypeError("nominal_valid must have boolean dtype")
    if nominal_valid.device != q_nominal_px.device:
        raise ValueError("nominal_valid must match q_nominal_px device")
    _require_floating_finite("q_nominal_px", q_nominal_px)
    for name, value in (
        ("q_corrected_a_px", q_corrected_a_px),
        ("q_corrected_b_px", q_corrected_b_px),
    ):
        if value.shape != q_nominal_px.shape:
            raise ValueError(f"{name} must match q_nominal_px")
        _require_floating_finite(name, value)
        if value.dtype != q_nominal_px.dtype:
            raise TypeError(f"{name} must match q_nominal_px dtype")
        if value.device != q_nominal_px.device:
            raise ValueError(f"{name} must match q_nominal_px device")
    valid = nominal_valid
    if prior_supervision_valid is not None:
        if prior_supervision_valid.shape != nominal_valid.shape:
            raise ValueError("prior_supervision_valid must match nominal_valid")
        if prior_supervision_valid.dtype != torch.bool:
            raise TypeError("prior_supervision_valid must have boolean dtype")
        if prior_supervision_valid.device != q_nominal_px.device:
            raise ValueError("prior_supervision_valid must match q_nominal_px device")
        valid = valid & prior_supervision_valid
    selected = valid.unsqueeze(-1).expand_as(q_nominal_px)
    residual_a = torch.where(
        selected, q_corrected_a_px - q_nominal_px, torch.zeros_like(q_nominal_px)
    )
    residual_b = torch.where(
        selected, q_corrected_b_px - q_nominal_px, torch.zeros_like(q_nominal_px)
    )
    return PixelObserverV2PairedSupervision(
        target_for_a_history_px=residual_b,
        target_for_b_history_px=residual_a,
        consensus_target_px=0.5 * (residual_a + residual_b),
        valid=valid,
    )


def masked_huber_pixel_loss(
    prediction: torch.Tensor,
    supervision: PixelObserverV2Supervision,
    *,
    delta_px: float = 0.25,
) -> torch.Tensor:
    """Huber loss over valid coordinates, including the jump-frame mask."""

    if prediction.shape != supervision.target_residual_px.shape:
        raise ValueError("prediction and supervision target shapes must match")
    if supervision.valid.shape != prediction.shape[:-1]:
        raise ValueError("supervision valid mask has an incompatible shape")
    if not prediction.is_floating_point() or not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction must be a finite floating point tensor")
    if prediction.dtype != supervision.target_residual_px.dtype:
        raise TypeError("prediction and supervision target dtypes must match")
    if prediction.device != supervision.target_residual_px.device:
        raise ValueError("prediction and supervision target devices must match")
    if supervision.valid.device != prediction.device:
        raise ValueError("supervision valid mask and prediction devices must match")
    if not math.isfinite(float(delta_px)) or not float(delta_px) > 0.0:
        raise ValueError("delta_px must be positive")
    selected = supervision.valid.unsqueeze(-1).expand_as(prediction)
    count = selected.sum()
    if int(count.detach().item()) == 0:
        return prediction.sum() * 0.0
    safe_target = torch.where(
        selected,
        supervision.target_residual_px,
        prediction.detach(),
    )
    difference = (prediction - safe_target).abs()
    quadratic = torch.clamp(difference, max=float(delta_px))
    elementwise = 0.5 * quadratic.square() + float(delta_px) * (
        difference - quadratic
    )
    return torch.where(selected, elementwise, torch.zeros_like(elementwise)).sum() / count


def paired_mutual_huber_pixel_loss(
    prediction_from_a: torch.Tensor,
    prediction_from_b: torch.Tensor,
    supervision: PixelObserverV2PairedSupervision,
    *,
    delta_px: float = 0.25,
) -> torch.Tensor:
    """Average the A->B and B->A robust cross-measurement losses."""

    first = PixelObserverV2Supervision(
        target_residual_px=supervision.target_for_a_history_px,
        valid=supervision.valid,
        target_mode=supervision.target_mode,
        diagnostic_only=False,
        deployment_allowed=True,
        target_contract=supervision.target_contract,
    )
    second = PixelObserverV2Supervision(
        target_residual_px=supervision.target_for_b_history_px,
        valid=supervision.valid,
        target_mode=supervision.target_mode,
        diagnostic_only=False,
        deployment_allowed=True,
        target_contract=supervision.target_contract,
    )
    return 0.5 * (
        masked_huber_pixel_loss(prediction_from_a, first, delta_px=delta_px)
        + masked_huber_pixel_loss(prediction_from_b, second, delta_px=delta_px)
    )


def pixel_observer_v2_checkpoint_metadata(
    adapter: PixelResidualVelocityInnovationAdapter,
    *,
    target_mode: str,
) -> dict[str, Any]:
    """Create the mandatory supervision/deployment portion of a checkpoint."""

    if target_mode not in {
        PAIRED_DEPLOYMENT_TARGET_MODE,
        SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
        ORACLE_CEILING_TARGET_MODE,
    }:
        raise ValueError("unknown pixel observer v2 target_mode")
    deployment_allowed = target_mode == PAIRED_DEPLOYMENT_TARGET_MODE
    diagnostic_only = not deployment_allowed
    target_contract = (
        "independent_a_b_mutual_supervision"
        if deployment_allowed
        else (
            "single_noisy_measurement_baseline"
            if target_mode == SINGLE_MEASUREMENT_BASELINE_TARGET_MODE
            else "oracle_ceiling"
        )
    )
    return {
        "schema_version": 1,
        "checkpoint_type": PIXEL_OBSERVER_V2_CHECKPOINT_TYPE,
        "objective": "masked_huber(next_residual_prior, selected_target_residual)",
        "adapter_config": adapter.export_config(),
        "supervision_contract": {
            "target_mode": target_mode,
            "diagnostic_only": diagnostic_only,
            "deployment_allowed": deployment_allowed,
            "formal_deployment_claim_allowed": deployment_allowed,
            "target_contract": target_contract,
            "oracle_target_used": target_mode == ORACLE_CEILING_TARGET_MODE,
            "prior_current_rgb_forbidden": True,
            "prior_current_correction_forbidden": True,
            "prior_current_oracle_forbidden": True,
            "current_known_geometry_allowed": True,
            "jump_frame_prior_supervision_mask_supported": True,
        },
    }
