"""Parameter-free deployment features for :class:`LeWMPosteriorAdapter`.

The builder accepts individual online tensors rather than a dataset/batch
mapping.  Its output names are exactly the adapter's four feature groups.
Privileged labels, target-view values, and experiment annotations therefore
cannot be forwarded accidentally through an open-ended mapping or ``**kwargs``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch import Tensor


# These are fixed protocol constants, not fitted or learned parameters.  They
# match the frozen corrector used by the posterior Gate: a 20 px output bound
# and a nominal-centered 5 x 5 visual grid spanning a 24 px radius.
MAX_RESIDUAL_PX = 20.0
LOCAL_SUPPORT_RADIUS_PX = 24.0
LOCAL_SUPPORT_GRID_SIZE = 5
LATENT_INNOVATION_CLIP = 1.0

POSTERIOR_FEATURE_GROUPS = (
    "innovation",
    "confidence",
    "mask_history",
    "action_summary",
)
POSTERIOR_CONFIDENCE_FIELDS = (
    "dx_over_max_residual",
    "dy_over_max_residual",
    "norm_over_sqrt2_max_residual",
    "local_support_fraction",
    "valid_nominal",
    "corrected_in_bounds",
)
POSTERIOR_FEATURE_ALLOWED_FIELDS = (
    "z_measurement",
    "z_prior",
    "q_nominal",
    "q_corrected",
    "valid_nominal",
    "image_width",
    "image_height",
    "online_validity_history",
    "causal_action_summary",
)
POSTERIOR_FEATURE_FORBIDDEN_FIELDS = (
    "dataset",
    "dataset_mapping",
    "batch",
    "task_id",
    "oracle_validity",
    "q_true",
    "measurement_valid",
    "anchor_dropout_active",
    "camera_step_frame",
    "canonical",
    "canonical_embedding",
    "canonical_rgb",
    "canonical_q_corrected",
    "canonical_target",
    "oracle_target",
    "event_name",
    "target",
    "target_latent",
)

# Input aliases make the whitelist terminology explicit while retaining the
# requested field-oriented names.
POSTERIOR_FEATURE_ALLOWED_INPUTS = POSTERIOR_FEATURE_ALLOWED_FIELDS
POSTERIOR_FEATURE_FORBIDDEN_INPUTS = POSTERIOR_FEATURE_FORBIDDEN_FIELDS


@dataclass(frozen=True)
class LeWMPosteriorFeatures:
    """The four feature groups consumed by ``LeWMPosteriorAdapter``."""

    innovation: Tensor
    confidence: Tensor
    mask_history: Tensor
    action_summary: Tensor

    def as_adapter_kwargs(self) -> dict[str, Tensor]:
        """Return only the adapter's deployment feature keyword arguments."""

        return {
            "innovation": self.innovation,
            "confidence": self.confidence,
            "mask_history": self.mask_history,
            "action_summary": self.action_summary,
        }


def _require_tensor(value: object, name: str) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    return value


def _require_floating_finite(value: object, name: str) -> Tensor:
    tensor = _require_tensor(value, name)
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _validate_latents(z_measurement: object, z_prior: object) -> tuple[Tensor, Tensor]:
    measurement = _require_floating_finite(z_measurement, "z_measurement")
    prior = _require_floating_finite(z_prior, "z_prior")
    if measurement.ndim not in (3, 4):
        raise ValueError(
            "z_measurement must have shape [B,K,D] or [B,T,K,D]"
        )
    if any(size <= 0 for size in measurement.shape):
        raise ValueError("z_measurement dimensions must be positive")
    if prior.shape != measurement.shape:
        raise ValueError("z_prior must match z_measurement shape")
    if prior.dtype != measurement.dtype:
        raise TypeError("z_prior dtype must match z_measurement")
    if prior.device != measurement.device:
        raise ValueError("z_prior device must match z_measurement")
    return measurement, prior


def _validate_coordinates(value: object, name: str, reference: Tensor) -> Tensor:
    tensor = _require_floating_finite(value, name)
    expected = (*reference.shape[:-1], 2)
    if tensor.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
    if tensor.dtype != reference.dtype:
        raise TypeError(f"{name} dtype must match z_measurement")
    if tensor.device != reference.device:
        raise ValueError(f"{name} device must match z_measurement")
    return tensor


def _validate_valid_nominal(value: object, reference: Tensor) -> Tensor:
    tensor = _require_tensor(value, "valid_nominal")
    expected = reference.shape[:-1]
    if tensor.shape != expected:
        raise ValueError(
            f"valid_nominal must have shape {expected}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.bool:
        raise TypeError("valid_nominal must have boolean dtype")
    if tensor.device != reference.device:
        raise ValueError("valid_nominal device must match z_measurement")
    return tensor


def _image_dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 2:
        raise ValueError(f"{name} must be at least 2")
    return result


def _validate_history(
    value: object,
    name: str,
    reference: Tensor,
    *,
    binary: bool,
) -> Tensor:
    tensor = _require_tensor(value, name)
    expected_prefix = reference.shape[:-1]
    if tensor.ndim != reference.ndim or tensor.shape[:-1] != expected_prefix:
        raise ValueError(
            f"{name} must have shape {expected_prefix} + [H], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] <= 0:
        raise ValueError(f"{name} history dimension must be positive")
    if tensor.device != reference.device:
        raise ValueError(f"{name} device must match z_measurement")
    if tensor.dtype == torch.bool:
        return tensor.to(dtype=reference.dtype)
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a boolean or floating dtype")
    if tensor.dtype != reference.dtype:
        raise TypeError(f"{name} floating dtype must match z_measurement")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    if binary:
        in_range = (tensor == 0) | (tensor == 1)
        qualifier = "binary"
    else:
        in_range = (tensor >= 0) & (tensor <= 1)
        qualifier = "in [0, 1]"
    if not bool(in_range.all()):
        raise ValueError(f"{name} values must be {qualifier}")
    return tensor


def _validate_action_summary(value: object, reference: Tensor) -> Tensor:
    action = _require_floating_finite(value, "causal_action_summary")
    leading_shape = reference.shape[:-2]
    if action.ndim != len(leading_shape) + 1 or action.shape[:-1] != leading_shape:
        raise ValueError(
            "causal_action_summary must have shape "
            f"{leading_shape} + [A], got {tuple(action.shape)}"
        )
    if action.shape[-1] <= 0:
        raise ValueError("causal_action_summary dimension must be positive")
    if action.dtype != reference.dtype:
        raise TypeError("causal_action_summary dtype must match z_measurement")
    if action.device != reference.device:
        raise ValueError("causal_action_summary device must match z_measurement")
    return action


def _local_support_fraction(
    q_nominal: Tensor,
    *,
    image_width: int,
    image_height: int,
) -> Tensor:
    """Return the fraction of local visual samples inside the image bounds.

    For ``q=(x,y)`` and fixed ``r=LOCAL_SUPPORT_RADIUS_PX``, let
    ``S={-r + 2*r*i/(g-1) | i=0,...,g-1}``, where
    ``g=LOCAL_SUPPORT_GRID_SIZE``.  This returns
    ``|{(u,v) in S^2 : q+(u,v) in [0,W-1] x [0,H-1]}| / g^2``.
    """

    radius = float(LOCAL_SUPPORT_RADIUS_PX)
    offsets = torch.linspace(
        -radius,
        radius,
        LOCAL_SUPPORT_GRID_SIZE,
        dtype=q_nominal.dtype,
        device=q_nominal.device,
    )
    sample_x = q_nominal[..., 0, None] + offsets
    sample_y = q_nominal[..., 1, None] + offsets
    support_x = (sample_x >= 0.0) & (sample_x <= float(image_width - 1))
    support_y = (sample_y >= 0.0) & (sample_y <= float(image_height - 1))
    return (
        support_x.to(dtype=q_nominal.dtype).mean(dim=-1)
        * support_y.to(dtype=q_nominal.dtype).mean(dim=-1)
    )


def build_lewm_posterior_features(
    *,
    z_measurement: Tensor,
    z_prior: Tensor,
    q_nominal: Tensor,
    q_corrected: Tensor,
    valid_nominal: Tensor,
    image_width: int,
    image_height: int,
    online_validity_history: Tensor,
    causal_action_summary: Tensor,
) -> LeWMPosteriorFeatures:
    """Build deterministic, deployment-only posterior-adapter features.

    Latents and pixel coordinates use ``[B,K,*]`` or ``[B,T,K,*]``.  History
    tensors use the same leading and anchor axes plus their history channels;
    the causal action summary has no anchor axis.  No mapping or extra keyword
    field is accepted.
    """

    measurement, prior = _validate_latents(z_measurement, z_prior)
    nominal = _validate_coordinates(q_nominal, "q_nominal", measurement)
    corrected = _validate_coordinates(q_corrected, "q_corrected", measurement)
    nominal_valid = _validate_valid_nominal(valid_nominal, measurement)
    width = _image_dimension(image_width, "image_width")
    height = _image_dimension(image_height, "image_height")
    validity_history = _validate_history(
        online_validity_history,
        "online_validity_history",
        measurement,
        binary=True,
    )
    action_summary = _validate_action_summary(causal_action_summary, measurement)

    latent_delta = prior - measurement
    innovation = latent_delta.clamp(
        min=-LATENT_INNOVATION_CLIP,
        max=LATENT_INNOVATION_CLIP,
    ) / LATENT_INNOVATION_CLIP

    delta_px = corrected - nominal
    if not bool(torch.isfinite(delta_px).all()):
        raise ValueError("q_corrected - q_nominal must be finite")
    dx = delta_px[..., 0] / MAX_RESIDUAL_PX
    dy = delta_px[..., 1] / MAX_RESIDUAL_PX
    normalized_norm = delta_px.square().sum(dim=-1).sqrt() / (
        math.sqrt(2.0) * MAX_RESIDUAL_PX
    )
    support = _local_support_fraction(
        nominal,
        image_width=width,
        image_height=height,
    )
    corrected_in_bounds = (
        (corrected[..., 0] >= 0.0)
        & (corrected[..., 0] <= float(width - 1))
        & (corrected[..., 1] >= 0.0)
        & (corrected[..., 1] <= float(height - 1))
    )
    confidence = torch.stack(
        (
            dx,
            dy,
            normalized_norm,
            support,
            nominal_valid.to(dtype=measurement.dtype),
            corrected_in_bounds.to(dtype=measurement.dtype),
        ),
        dim=-1,
    )
    mask_history = validity_history

    for name, value in (
        ("innovation", innovation),
        ("confidence", confidence),
        ("mask_history", mask_history),
        ("action_summary", action_summary),
    ):
        if value.dtype != measurement.dtype:
            raise RuntimeError(f"{name} builder dtype contract was violated")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"built {name} must be finite")

    return LeWMPosteriorFeatures(
        innovation=innovation,
        confidence=confidence,
        mask_history=mask_history,
        action_summary=action_summary,
    )


__all__ = [
    "LATENT_INNOVATION_CLIP",
    "LOCAL_SUPPORT_GRID_SIZE",
    "LOCAL_SUPPORT_RADIUS_PX",
    "LeWMPosteriorFeatures",
    "MAX_RESIDUAL_PX",
    "POSTERIOR_CONFIDENCE_FIELDS",
    "POSTERIOR_FEATURE_ALLOWED_FIELDS",
    "POSTERIOR_FEATURE_ALLOWED_INPUTS",
    "POSTERIOR_FEATURE_FORBIDDEN_FIELDS",
    "POSTERIOR_FEATURE_FORBIDDEN_INPUTS",
    "POSTERIOR_FEATURE_GROUPS",
    "build_lewm_posterior_features",
]
