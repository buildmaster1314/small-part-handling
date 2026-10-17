"""Frozen oracle-anchor targets for matched grounded LeWM comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.modules import sample_feature_map_at_pixels


ORACLE_TARGET_CHECKPOINT_TYPE = "fcpg_frozen_oracle_target_v1"


def _orthonormal_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
) -> torch.Tensor:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("projection dimensions must be positive")
    if output_dim > input_dim:
        raise ValueError("output_dim must not exceed input_dim")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float64)
    basis, triangular = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.sign(torch.diagonal(triangular))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (basis * signs.unsqueeze(0)).to(dtype=torch.float32)


class FrozenOracleTargetEncoder(nn.Module):
    """Build a shared target from frozen visual features at oracle anchors.

    The target contains one global visual token followed by one token per
    kinematic keypoint. Each token is normalized independently and projected by
    the same fixed orthonormal matrix. There are no trainable parameters.
    """

    def __init__(
        self,
        *,
        visual_dim: int,
        num_keypoints: int,
        target_dim: int = 192,
        projection_seed: int = 3072,
    ) -> None:
        super().__init__()
        if num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
        token_count = 1 + int(num_keypoints)
        if target_dim % token_count:
            raise ValueError(
                f"target_dim={target_dim} must be divisible by token_count={token_count}"
            )
        token_dim = int(target_dim) // token_count
        self.visual_dim = int(visual_dim)
        self.num_keypoints = int(num_keypoints)
        self.target_dim = int(target_dim)
        self.token_dim = token_dim
        self.projection_seed = int(projection_seed)
        self.register_buffer(
            "projection",
            _orthonormal_projection(self.visual_dim, self.token_dim, self.projection_seed),
        )

    def spec(self) -> dict[str, int | str]:
        return {
            "checkpoint_type": ORACLE_TARGET_CHECKPOINT_TYPE,
            "visual_dim": self.visual_dim,
            "num_keypoints": self.num_keypoints,
            "target_dim": self.target_dim,
            "token_dim": self.token_dim,
            "projection_seed": self.projection_seed,
            "normalization": "per_token_layer_norm_no_affine",
            "token_order": "global_then_keypoints",
        }

    def forward_from_features(
        self,
        feature_map: torch.Tensor,
        global_feature: torch.Tensor,
        q_true_px: torch.Tensor,
        valid_true: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if feature_map.ndim != 4 or feature_map.shape[1] != self.visual_dim:
            raise ValueError(
                "feature_map must have shape "
                f"[B, {self.visual_dim}, H, W], got {tuple(feature_map.shape)}"
            )
        batch = int(feature_map.shape[0])
        if global_feature.shape != (batch, self.visual_dim):
            raise ValueError(
                f"global_feature must have shape {(batch, self.visual_dim)}, "
                f"got {tuple(global_feature.shape)}"
            )
        if q_true_px.shape != (batch, self.num_keypoints, 2):
            raise ValueError(
                f"q_true_px must have shape {(batch, self.num_keypoints, 2)}, "
                f"got {tuple(q_true_px.shape)}"
            )
        if valid_true.shape != (batch, self.num_keypoints):
            raise ValueError(
                f"valid_true must have shape {(batch, self.num_keypoints)}, "
                f"got {tuple(valid_true.shape)}"
            )
        width, height = image_size
        inside = (
            (q_true_px[..., 0] >= 0.0)
            & (q_true_px[..., 0] <= float(width - 1))
            & (q_true_px[..., 1] >= 0.0)
            & (q_true_px[..., 1] <= float(height - 1))
        )
        local = sample_feature_map_at_pixels(
            feature_map,
            q_true_px,
            image_size=image_size,
            padding_mode="border",
        )
        tokens = torch.cat([global_feature.unsqueeze(1), local], dim=1)
        tokens = F.layer_norm(tokens.float(), (self.visual_dim,))
        projected = tokens @ self.projection.to(dtype=tokens.dtype)
        local_valid = (valid_true.bool() & inside).to(dtype=projected.dtype)
        mask = torch.cat(
            [torch.ones(batch, 1, device=projected.device, dtype=projected.dtype), local_valid],
            dim=1,
        )
        projected = projected * mask.unsqueeze(-1)
        return projected.reshape(batch, self.target_dim)


def save_frozen_oracle_target_encoder(
    path: str | Path,
    encoder: FrozenOracleTargetEncoder,
) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_type": ORACLE_TARGET_CHECKPOINT_TYPE,
        "spec": encoder.spec(),
        "state_dict": encoder.state_dict(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_frozen_oracle_target_encoder(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[FrozenOracleTargetEncoder, Mapping[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("oracle target checkpoint must be a mapping")
    if payload.get("checkpoint_type") != ORACLE_TARGET_CHECKPOINT_TYPE:
        raise ValueError("unexpected oracle target checkpoint type")
    spec = payload.get("spec")
    state_dict = payload.get("state_dict")
    if not isinstance(spec, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("oracle target checkpoint is missing spec or state_dict")
    encoder = FrozenOracleTargetEncoder(
        visual_dim=int(spec["visual_dim"]),
        num_keypoints=int(spec["num_keypoints"]),
        target_dim=int(spec["target_dim"]),
        projection_seed=int(spec["projection_seed"]),
    )
    encoder.load_state_dict(state_dict, strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder, payload
