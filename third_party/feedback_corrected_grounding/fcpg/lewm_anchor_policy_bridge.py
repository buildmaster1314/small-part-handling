"""Residual bridge from persistent-anchor latents to policy grounding tokens.

The Anchor-Persistent LeWM and the frozen action policy use different anchor
spaces.  LeWM predicts 192-dimensional projected anchor embeddings, whereas
the policy consumes 256-dimensional outputs of its ``GroundedTokenAdapter``.
This module maps *changes* in the former to changes in the latter while using
the last valid policy token as an exact residual base.

The bridge intentionally has no image, task, validity, schedule, or privileged
target input.  A known-validity router lives outside this module.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE = "fcpg_anchor_policy_token_bridge_v1"


def _finite_vector(
    value: torch.Tensor | Iterable[float],
    *,
    size: int,
    name: str,
    positive: bool = False,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    if positive and bool(torch.any(tensor <= 0.0)):
        raise ValueError(f"{name} must be strictly positive")
    return tensor


class ResidualAnchorPolicyBridge(nn.Module):
    """Map one LeWM anchor prior to a frozen policy's native token space.

    For anchor ``k``, the bridge computes

    ``g_hat = g_valid + W2(GELU(W1([z_valid, z_hat-z_valid, dx, id_k])))``.

    All continuous inputs are standardized by fixed train-split statistics.
    The last layer is zero initialized, so a fresh bridge is exactly the
    hold-last baseline.  Parameters are shared across anchors.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        token_dim: int = 256,
        num_keypoints: int = 3,
        hidden_dim: int = 64,
        zero_initialize_output: bool = True,
    ) -> None:
        super().__init__()
        dimensions = {
            "latent_dim": latent_dim,
            "token_dim": token_dim,
            "num_keypoints": num_keypoints,
            "hidden_dim": hidden_dim,
        }
        if any(isinstance(value, bool) or int(value) <= 0 for value in dimensions.values()):
            raise ValueError("all bridge dimensions must be positive integers")
        self.latent_dim = int(latent_dim)
        self.token_dim = int(token_dim)
        self.num_keypoints = int(num_keypoints)
        self.hidden_dim = int(hidden_dim)
        input_dim = 2 * self.latent_dim + 3 + self.num_keypoints

        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.token_dim)
        if zero_initialize_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

        self.register_buffer("context_mean", torch.zeros(self.latent_dim))
        self.register_buffer("context_std", torch.ones(self.latent_dim))
        self.register_buffer("latent_delta_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_delta_std", torch.ones(self.latent_dim))
        self.register_buffer("xyz_delta_mean", torch.zeros(3))
        self.register_buffer("xyz_delta_std", torch.ones(3))
        self.register_buffer("token_error_std", torch.ones(self.token_dim))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_input_normalization(
        self,
        *,
        context_mean: torch.Tensor | Iterable[float],
        context_std: torch.Tensor | Iterable[float],
        latent_delta_mean: torch.Tensor | Iterable[float],
        latent_delta_std: torch.Tensor | Iterable[float],
        xyz_delta_mean: torch.Tensor | Iterable[float],
        xyz_delta_std: torch.Tensor | Iterable[float],
        token_error_std: torch.Tensor | Iterable[float] | None = None,
    ) -> None:
        values = {
            "context_mean": _finite_vector(
                context_mean, size=self.latent_dim, name="context_mean"
            ),
            "context_std": _finite_vector(
                context_std,
                size=self.latent_dim,
                name="context_std",
                positive=True,
            ),
            "latent_delta_mean": _finite_vector(
                latent_delta_mean,
                size=self.latent_dim,
                name="latent_delta_mean",
            ),
            "latent_delta_std": _finite_vector(
                latent_delta_std,
                size=self.latent_dim,
                name="latent_delta_std",
                positive=True,
            ),
            "xyz_delta_mean": _finite_vector(
                xyz_delta_mean, size=3, name="xyz_delta_mean"
            ),
            "xyz_delta_std": _finite_vector(
                xyz_delta_std, size=3, name="xyz_delta_std", positive=True
            ),
        }
        if token_error_std is not None:
            values["token_error_std"] = _finite_vector(
                token_error_std,
                size=self.token_dim,
                name="token_error_std",
                positive=True,
            )
        with torch.no_grad():
            for name, value in values.items():
                target = getattr(self, name)
                target.copy_(value.to(device=target.device, dtype=target.dtype))

    def _validate_inputs(
        self,
        predicted_embedding: torch.Tensor,
        last_valid_embedding: torch.Tensor,
        last_valid_token: torch.Tensor,
        current_xyz: torch.Tensor,
        last_valid_xyz: torch.Tensor,
    ) -> tuple[int, ...]:
        values = {
            "predicted_embedding": predicted_embedding,
            "last_valid_embedding": last_valid_embedding,
            "last_valid_token": last_valid_token,
            "current_xyz": current_xyz,
            "last_valid_xyz": last_valid_xyz,
        }
        if any(not isinstance(value, torch.Tensor) for value in values.values()):
            raise TypeError("all bridge inputs must be torch.Tensor instances")
        leading = tuple(predicted_embedding.shape[:-2])
        expected = {
            "predicted_embedding": (*leading, self.num_keypoints, self.latent_dim),
            "last_valid_embedding": (*leading, self.num_keypoints, self.latent_dim),
            "last_valid_token": (*leading, self.num_keypoints, self.token_dim),
            "current_xyz": (*leading, self.num_keypoints, 3),
            "last_valid_xyz": (*leading, self.num_keypoints, 3),
        }
        for name, value in values.items():
            if value.shape != expected[name]:
                raise ValueError(
                    f"{name} must have shape {expected[name]}, got {tuple(value.shape)}"
                )
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be floating point")
            if value.device != predicted_embedding.device:
                raise ValueError("all bridge inputs must be on the same device")
            if value.dtype != predicted_embedding.dtype:
                raise TypeError("all bridge inputs must use the same dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        return leading

    def forward(
        self,
        predicted_embedding: torch.Tensor,
        last_valid_embedding: torch.Tensor,
        last_valid_token: torch.Tensor,
        current_xyz: torch.Tensor,
        last_valid_xyz: torch.Tensor,
    ) -> torch.Tensor:
        leading = self._validate_inputs(
            predicted_embedding,
            last_valid_embedding,
            last_valid_token,
            current_xyz,
            last_valid_xyz,
        )
        dtype = predicted_embedding.dtype
        context = (last_valid_embedding - self.context_mean.to(dtype=dtype)) / (
            self.context_std.to(dtype=dtype).clamp_min(1.0e-6)
        )
        latent_delta = predicted_embedding - last_valid_embedding
        latent_delta = (latent_delta - self.latent_delta_mean.to(dtype=dtype)) / (
            self.latent_delta_std.to(dtype=dtype).clamp_min(1.0e-6)
        )
        xyz_delta = current_xyz - last_valid_xyz
        xyz_delta = (xyz_delta - self.xyz_delta_mean.to(dtype=dtype)) / (
            self.xyz_delta_std.to(dtype=dtype).clamp_min(1.0e-6)
        )
        identity = torch.eye(
            self.num_keypoints,
            device=predicted_embedding.device,
            dtype=dtype,
        ).reshape(*(1 for _ in leading), self.num_keypoints, self.num_keypoints)
        identity = identity.expand(*leading, -1, -1)
        features = torch.cat((context, latent_delta, xyz_delta, identity), dim=-1)
        token_delta = self.output_projection(F.gelu(self.input_projection(features)))
        return last_valid_token + token_delta

    def standardized_token_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        beta: float = 0.1,
    ) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.shape[-1] != self.token_dim:
            raise ValueError("prediction and target must share shape [..., K, token_dim]")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        scale = self.token_error_std.to(
            device=prediction.device, dtype=prediction.dtype
        ).clamp_min(1.0e-6)
        return F.smooth_l1_loss(
            prediction / scale,
            target.detach() / scale,
            beta=float(beta),
        )


def build_matched_bridge_pair(
    *,
    initialization_seed: int,
    latent_dim: int = 192,
    token_dim: int = 256,
    num_keypoints: int = 3,
    hidden_dim: int = 64,
) -> tuple[ResidualAnchorPolicyBridge, ResidualAnchorPolicyBridge]:
    """Create byte-identical factual/no-action bridges before training."""

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(initialization_seed))
        factual = ResidualAnchorPolicyBridge(
            latent_dim=latent_dim,
            token_dim=token_dim,
            num_keypoints=num_keypoints,
            hidden_dim=hidden_dim,
        )
    return factual, deepcopy(factual)


__all__ = [
    "ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE",
    "ResidualAnchorPolicyBridge",
    "build_matched_bridge_pair",
]
