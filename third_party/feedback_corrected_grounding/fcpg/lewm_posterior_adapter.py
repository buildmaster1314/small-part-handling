"""Restricted measurement-dominant posterior adapter for anchor LeWM states.

The adapter has exactly four deployment feature groups and no privileged
inputs.  One shared network emits one scalar gain per anchor; it cannot emit a
per-latent gain or a learned residual.  The posterior update is therefore
always the bounded interpolation

``z_measurement + beta * (z_prior - z_measurement)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


POSTERIOR_ADAPTER_CHECKPOINT_TYPE = "fcpg_lewm_posterior_adapter_v1"
POSTERIOR_ADAPTER_SCHEMA_VERSION = 1
MAX_POSTERIOR_ADAPTER_HIDDEN_DIM = 32
POSTERIOR_ADAPTER_DEPLOYMENT_INPUTS = (
    "innovation",
    "confidence",
    "mask_history",
    "action_summary",
)
POSTERIOR_ADAPTER_FORBIDDEN_INPUTS = (
    "task_id",
    "oracle_validity",
    "canonical",
    "q_true",
    "measurement_valid",
)


def _integer_dimension(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


@dataclass(frozen=True)
class LeWMPosteriorAdapterConfig:
    """Reconstructable architecture configuration stored with checkpoints."""

    latent_dim: int
    innovation_dim: int
    confidence_dim: int
    mask_history_dim: int
    action_summary_dim: int
    hidden_dim: int = MAX_POSTERIOR_ADAPTER_HIDDEN_DIM

    def __post_init__(self) -> None:
        for name in (
            "latent_dim",
            "innovation_dim",
            "confidence_dim",
            "mask_history_dim",
            "action_summary_dim",
        ):
            _integer_dimension(getattr(self, name), name)
        hidden_dim = _integer_dimension(
            self.hidden_dim, "hidden_dim", allow_zero=True
        )
        if hidden_dim > MAX_POSTERIOR_ADAPTER_HIDDEN_DIM:
            raise ValueError(
                "hidden_dim must be at most "
                f"{MAX_POSTERIOR_ADAPTER_HIDDEN_DIM}"
            )

    def to_checkpoint_config(self) -> dict[str, Any]:
        return {
            "checkpoint_type": POSTERIOR_ADAPTER_CHECKPOINT_TYPE,
            "schema_version": POSTERIOR_ADAPTER_SCHEMA_VERSION,
            "latent_dim": self.latent_dim,
            "innovation_dim": self.innovation_dim,
            "confidence_dim": self.confidence_dim,
            "mask_history_dim": self.mask_history_dim,
            "action_summary_dim": self.action_summary_dim,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_checkpoint_config(
        cls, value: Mapping[str, object]
    ) -> "LeWMPosteriorAdapterConfig":
        if not isinstance(value, Mapping):
            raise TypeError("posterior adapter checkpoint config must be an object")
        expected = {
            "checkpoint_type",
            "schema_version",
            "latent_dim",
            "innovation_dim",
            "confidence_dim",
            "mask_history_dim",
            "action_summary_dim",
            "hidden_dim",
        }
        missing = expected.difference(value)
        unexpected = set(value).difference(expected)
        if missing:
            raise ValueError(
                "posterior adapter checkpoint config is missing fields: "
                f"{sorted(missing)}"
            )
        if unexpected:
            raise ValueError(
                "posterior adapter checkpoint config has unexpected fields: "
                f"{sorted(unexpected)}"
            )
        if value["checkpoint_type"] != POSTERIOR_ADAPTER_CHECKPOINT_TYPE:
            raise ValueError("posterior adapter checkpoint_type mismatch")
        if value["schema_version"] != POSTERIOR_ADAPTER_SCHEMA_VERSION:
            raise ValueError("posterior adapter schema_version mismatch")
        return cls(
            latent_dim=_integer_dimension(value["latent_dim"], "latent_dim"),
            innovation_dim=_integer_dimension(
                value["innovation_dim"], "innovation_dim"
            ),
            confidence_dim=_integer_dimension(
                value["confidence_dim"], "confidence_dim"
            ),
            mask_history_dim=_integer_dimension(
                value["mask_history_dim"], "mask_history_dim"
            ),
            action_summary_dim=_integer_dimension(
                value["action_summary_dim"], "action_summary_dim"
            ),
            hidden_dim=_integer_dimension(
                value["hidden_dim"], "hidden_dim", allow_zero=True
            ),
        )


@dataclass(frozen=True)
class LeWMPosteriorAdapterOutput:
    """Posterior state and its shared-network, per-anchor scalar gain."""

    z_post: Tensor
    beta: Tensor


class LeWMPosteriorAdapter(nn.Module):
    """Measurement-first fusion with a bounded scalar gate per anchor.

    Latents may have shape ``[B, K, D]`` or ``[B, T, K, D]``.  Innovation,
    confidence, and mask history preserve the same leading and anchor axes.
    Action summaries have no anchor axis and are broadcast to all anchors, so
    the same gate parameters and action context are shared across anchors.

    Non-finite prior or feature values fail closed locally: the affected
    anchor receives ``beta=0`` and its posterior is the measurement.  A
    non-finite measurement is rejected because no finite measurement-identity
    fallback exists.
    """

    checkpoint_type = POSTERIOR_ADAPTER_CHECKPOINT_TYPE

    def __init__(
        self,
        latent_dim: int,
        innovation_dim: int,
        confidence_dim: int,
        mask_history_dim: int,
        action_summary_dim: int,
        hidden_dim: int = MAX_POSTERIOR_ADAPTER_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self._config = LeWMPosteriorAdapterConfig(
            latent_dim=latent_dim,
            innovation_dim=innovation_dim,
            confidence_dim=confidence_dim,
            mask_history_dim=mask_history_dim,
            action_summary_dim=action_summary_dim,
            hidden_dim=hidden_dim,
        )
        self.latent_dim = self._config.latent_dim
        self.innovation_dim = self._config.innovation_dim
        self.confidence_dim = self._config.confidence_dim
        self.mask_history_dim = self._config.mask_history_dim
        self.action_summary_dim = self._config.action_summary_dim
        self.hidden_dim = self._config.hidden_dim

        gate_input_dim = (
            self.innovation_dim
            + self.confidence_dim
            + self.mask_history_dim
            + self.action_summary_dim
        )
        if self.hidden_dim == 0:
            self.gate = nn.Sequential(nn.Linear(gate_input_dim, 1))
        else:
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    @property
    def config(self) -> LeWMPosteriorAdapterConfig:
        return self._config

    @property
    def output_layer(self) -> nn.Linear:
        layer = self.gate[-1]
        if not isinstance(layer, nn.Linear):
            raise RuntimeError("posterior adapter output layer contract was violated")
        return layer

    def checkpoint_config(self) -> dict[str, Any]:
        """Return the complete architecture config needed for reconstruction."""

        return self._config.to_checkpoint_config()

    def to_config(self) -> dict[str, Any]:
        return self.checkpoint_config()

    @classmethod
    def from_checkpoint_config(
        cls, value: Mapping[str, object]
    ) -> "LeWMPosteriorAdapter":
        config = LeWMPosteriorAdapterConfig.from_checkpoint_config(value)
        return cls(
            latent_dim=config.latent_dim,
            innovation_dim=config.innovation_dim,
            confidence_dim=config.confidence_dim,
            mask_history_dim=config.mask_history_dim,
            action_summary_dim=config.action_summary_dim,
            hidden_dim=config.hidden_dim,
        )

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> "LeWMPosteriorAdapter":
        return cls.from_checkpoint_config(value)

    @staticmethod
    def _validate_latents(z_measurement: Tensor, z_prior: Tensor, latent_dim: int) -> None:
        if not torch.is_tensor(z_measurement):
            raise TypeError("z_measurement must be a tensor")
        if not torch.is_tensor(z_prior):
            raise TypeError("z_prior must be a tensor")
        if z_measurement.ndim not in (3, 4):
            raise ValueError(
                "z_measurement must have shape [B,K,D] or [B,T,K,D]"
            )
        if z_measurement.shape[-1] != latent_dim:
            raise ValueError(
                f"z_measurement must end with latent_dim={latent_dim}, "
                f"got {tuple(z_measurement.shape)}"
            )
        if any(size <= 0 for size in z_measurement.shape[:-1]):
            raise ValueError("z_measurement batch/time/anchor dimensions must be positive")
        if z_prior.shape != z_measurement.shape:
            raise ValueError(
                "z_prior must match z_measurement shape; got "
                f"{tuple(z_prior.shape)} and {tuple(z_measurement.shape)}"
            )
        if not torch.is_floating_point(z_measurement):
            raise TypeError("z_measurement must have a floating dtype")
        if z_prior.dtype != z_measurement.dtype:
            raise TypeError("z_prior and z_measurement dtypes must match")
        if z_prior.device != z_measurement.device:
            raise ValueError("z_prior and z_measurement devices must match")
        if not bool(torch.isfinite(z_measurement).all()):
            raise ValueError("z_measurement must be finite")

    @staticmethod
    def _validate_feature(
        value: Tensor,
        *,
        name: str,
        expected_shape: tuple[int, ...],
        reference: Tensor,
        allow_bool: bool = False,
    ) -> Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
            )
        if value.device != reference.device:
            raise ValueError(f"{name} must be on the z_measurement device")
        if allow_bool and value.dtype == torch.bool:
            return value.to(dtype=reference.dtype)
        if not torch.is_floating_point(value):
            suffix = " or bool" if allow_bool else ""
            raise TypeError(f"{name} must have a floating dtype{suffix}")
        if value.dtype != reference.dtype:
            raise TypeError(f"{name} dtype must match z_measurement")
        return value

    def _prepare_inputs(
        self,
        z_measurement: Tensor,
        z_prior: Tensor,
        innovation: Tensor,
        confidence: Tensor,
        mask_history: Tensor,
        action_summary: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_latents(z_measurement, z_prior, self.latent_dim)
        leading_shape = tuple(z_measurement.shape[:-2])
        anchors = z_measurement.shape[-2]
        anchor_shape = (*leading_shape, anchors)

        innovation = self._validate_feature(
            innovation,
            name="innovation",
            expected_shape=(*anchor_shape, self.innovation_dim),
            reference=z_measurement,
        )
        confidence = self._validate_feature(
            confidence,
            name="confidence",
            expected_shape=(*anchor_shape, self.confidence_dim),
            reference=z_measurement,
        )
        mask_history = self._validate_feature(
            mask_history,
            name="mask_history",
            expected_shape=(*anchor_shape, self.mask_history_dim),
            reference=z_measurement,
            allow_bool=True,
        )
        action_summary = self._validate_feature(
            action_summary,
            name="action_summary",
            expected_shape=(*leading_shape, self.action_summary_dim),
            reference=z_measurement,
        )

        per_anchor_finite = torch.isfinite(z_prior).all(dim=-1, keepdim=True)
        safe_prior = torch.where(torch.isfinite(z_prior), z_prior, z_measurement)
        delta = safe_prior - z_measurement
        delta_finite = torch.isfinite(delta)
        per_anchor_finite = per_anchor_finite & delta_finite.all(
            dim=-1, keepdim=True
        )
        safe_delta = torch.where(delta_finite, delta, torch.zeros_like(delta))

        safe_anchor_features: list[Tensor] = []
        for feature in (innovation, confidence, mask_history):
            finite = torch.isfinite(feature)
            per_anchor_finite = per_anchor_finite & finite.all(
                dim=-1, keepdim=True
            )
            safe_anchor_features.append(
                torch.where(finite, feature, torch.zeros_like(feature))
            )

        action_finite = torch.isfinite(action_summary)
        safe_action = torch.where(
            action_finite, action_summary, torch.zeros_like(action_summary)
        )
        action_finite = action_finite.all(dim=-1, keepdim=True).unsqueeze(-2)
        per_anchor_finite = per_anchor_finite & action_finite
        expanded_action = safe_action.unsqueeze(-2).expand(
            *leading_shape, anchors, self.action_summary_dim
        )
        gate_input = torch.cat((*safe_anchor_features, expanded_action), dim=-1)
        return gate_input, safe_delta, per_anchor_finite

    def _bounded_beta(self, gate_input: Tensor, input_finite: Tensor) -> Tensor:
        raw_beta = self.gate(gate_input)
        raw_finite = torch.isfinite(raw_beta)
        safe_raw_beta = torch.where(
            raw_finite, raw_beta, torch.zeros_like(raw_beta)
        )
        # A centered one-sided sigmoid is exactly zero for a zero logit while
        # retaining a non-zero boundary gradient for the zero-initialized head.
        beta = torch.clamp(2.0 * torch.sigmoid(safe_raw_beta) - 1.0, 0.0, 1.0)
        return torch.where(
            input_finite & raw_finite, beta, torch.zeros_like(beta)
        )

    def compute_beta(
        self,
        z_measurement: Tensor,
        z_prior: Tensor,
        innovation: Tensor,
        confidence: Tensor,
        mask_history: Tensor,
        action_summary: Tensor,
    ) -> Tensor:
        """Compute ``[..., K, 1]`` gains from deployment-eligible inputs."""

        gate_input, _, input_finite = self._prepare_inputs(
            z_measurement,
            z_prior,
            innovation,
            confidence,
            mask_history,
            action_summary,
        )
        return self._bounded_beta(gate_input, input_finite)

    def forward(
        self,
        z_measurement: Tensor,
        z_prior: Tensor,
        innovation: Tensor,
        confidence: Tensor,
        mask_history: Tensor,
        action_summary: Tensor,
    ) -> LeWMPosteriorAdapterOutput:
        gate_input, safe_delta, input_finite = self._prepare_inputs(
            z_measurement,
            z_prior,
            innovation,
            confidence,
            mask_history,
            action_summary,
        )
        beta = self._bounded_beta(gate_input, input_finite)
        candidate = z_measurement + beta * safe_delta
        candidate_finite = torch.isfinite(candidate).all(dim=-1, keepdim=True)
        beta = torch.where(candidate_finite, beta, torch.zeros_like(beta))
        z_post = torch.where(candidate_finite, candidate, z_measurement)
        return LeWMPosteriorAdapterOutput(z_post=z_post, beta=beta)


__all__ = [
    "LeWMPosteriorAdapter",
    "LeWMPosteriorAdapterConfig",
    "LeWMPosteriorAdapterOutput",
    "MAX_POSTERIOR_ADAPTER_HIDDEN_DIM",
    "POSTERIOR_ADAPTER_CHECKPOINT_TYPE",
    "POSTERIOR_ADAPTER_DEPLOYMENT_INPUTS",
    "POSTERIOR_ADAPTER_FORBIDDEN_INPUTS",
    "POSTERIOR_ADAPTER_SCHEMA_VERSION",
]
