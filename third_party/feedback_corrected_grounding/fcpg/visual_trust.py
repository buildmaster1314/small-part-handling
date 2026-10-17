"""Small per-anchor trust head for current FCPG visual evidence."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn


DEFAULT_VISUAL_TRUST_HIDDEN_DIM = 32
MAX_VISUAL_TRUST_HIDDEN_DIM = 32
MAX_VISUAL_TRUST_PARAMETERS = 20_000


def _positive_dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class VisualTrustOutput:
    """Per-anchor raw logits and sigmoid trust values, both shaped ``[B,K]``."""

    logits: Tensor
    trust: Tensor


class VisualTrustHead(nn.Module):
    """Estimate trust from current, per-anchor FCPG features only.

    ``current_visual_features`` has shape ``[B,K,C]`` and
    ``auxiliary_descriptors`` has shape ``[B,K,G]``. The same normalized MLP
    is applied independently to every anchor; no temporal, policy, or target
    context is accepted by the forward interface.
    """

    def __init__(
        self,
        visual_dim: int,
        auxiliary_dim: int,
        hidden_dim: int = DEFAULT_VISUAL_TRUST_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.visual_dim = _positive_dimension(visual_dim, "visual_dim")
        self.auxiliary_dim = _positive_dimension(auxiliary_dim, "auxiliary_dim")
        self.hidden_dim = _positive_dimension(hidden_dim, "hidden_dim")
        if self.hidden_dim > MAX_VISUAL_TRUST_HIDDEN_DIM:
            raise ValueError(
                f"hidden_dim must be at most {MAX_VISUAL_TRUST_HIDDEN_DIM}"
            )

        input_dim = self.visual_dim + self.auxiliary_dim
        parameter_count = input_dim * (self.hidden_dim + 2) + 2 * self.hidden_dim + 1
        if parameter_count >= MAX_VISUAL_TRUST_PARAMETERS:
            raise ValueError(
                "VisualTrustHead must have fewer than "
                f"{MAX_VISUAL_TRUST_PARAMETERS} parameters; requested architecture "
                f"has {parameter_count}"
            )

        self.input_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self._parameter_count = parameter_count

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters in the head."""

        return self._parameter_count

    @staticmethod
    def _validate_tensor(
        value: object,
        *,
        name: str,
        feature_dim: int,
    ) -> Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B,K,{feature_dim}], "
                f"got {tuple(value.shape)}"
            )
        if any(size <= 0 for size in value.shape):
            raise ValueError(f"{name} dimensions must be positive")
        if value.shape[-1] != feature_dim:
            raise ValueError(
                f"{name} must have shape [B,K,{feature_dim}], "
                f"got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        return value

    def forward(
        self,
        current_visual_features: Tensor,
        auxiliary_descriptors: Tensor,
    ) -> VisualTrustOutput:
        visual = self._validate_tensor(
            current_visual_features,
            name="current_visual_features",
            feature_dim=self.visual_dim,
        )
        auxiliary = self._validate_tensor(
            auxiliary_descriptors,
            name="auxiliary_descriptors",
            feature_dim=self.auxiliary_dim,
        )
        if visual.shape[:2] != auxiliary.shape[:2]:
            raise ValueError(
                "current_visual_features and auxiliary_descriptors must have "
                "matching batch and anchor dimensions"
            )
        if visual.dtype != auxiliary.dtype:
            raise TypeError(
                "current_visual_features and auxiliary_descriptors dtypes must match"
            )
        if visual.device != auxiliary.device:
            raise ValueError(
                "current_visual_features and auxiliary_descriptors devices must match"
            )

        parameter = self.input_norm.weight
        if visual.dtype != parameter.dtype:
            raise TypeError(
                "input dtype must match VisualTrustHead parameter dtype"
            )
        if visual.device != parameter.device:
            raise ValueError(
                "input device must match VisualTrustHead parameter device"
            )

        features = torch.cat((visual, auxiliary), dim=-1)
        logits = self.mlp(self.input_norm(features)).squeeze(-1)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("VisualTrustHead logits must be finite")
        trust = torch.sigmoid(logits)
        if not bool(torch.isfinite(trust).all()):
            raise ValueError("VisualTrustHead trust must be finite")
        return VisualTrustOutput(logits=logits, trust=trust)
