"""Continuous-camera residual readout for the joint A4 LeWM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix_joint_a4 import ANCHOR_COUNT, LATENT_DIM, STATE_SLOTS


POSE_READOUT_PROTOCOL = "a4_orbit_pose_residual_readout_v1"
POSE_READOUT_CHECKPOINT_TYPE = "a4_orbit_pose_residual_readout"
CAMERA_FEATURE_DIM = 10
REFERENCE_MODES = ("base_prediction", "current_q")


@dataclass(frozen=True)
class PoseReadoutOutput:
    predicted_q_px: torch.Tensor
    residual_q_px: torch.Tensor


def orbit_camera_basis_features(
    clock_hour: torch.Tensor,
    elevation_degrees: torch.Tensor,
    radius_m: torch.Tensor,
) -> torch.Tensor:
    """Return a smooth world-to-camera orientation descriptor.

    The first nine values are the camera right, up, and forward axes in world
    coordinates. The final value is log radius relative to the A4 default.
    """

    if clock_hour.shape != elevation_degrees.shape or clock_hour.shape != radius_m.shape:
        raise ValueError("orbit camera metadata tensors must share one shape")
    if clock_hour.ndim != 1:
        raise ValueError("orbit camera metadata must have shape [B]")
    if not bool(torch.isfinite(elevation_degrees).all()) or not bool(
        torch.isfinite(radius_m).all()
    ):
        raise ValueError("orbit camera metadata contains non-finite values")
    if bool(torch.any((clock_hour < 1) | (clock_hour > 12))):
        raise ValueError("clock_hour must be in [1,12]")
    if bool(torch.any(radius_m <= 0.0)):
        raise ValueError("camera radius must be positive")

    dtype = elevation_degrees.dtype
    device = elevation_degrees.device
    theta = torch.remainder(clock_hour.to(dtype=dtype), 12.0) * (math.pi / 6.0)
    elevation = elevation_degrees * (math.pi / 180.0)
    offset = torch.stack(
        [
            torch.sin(theta),
            -torch.cos(theta),
            torch.tan(elevation),
        ],
        dim=-1,
    )
    forward = F.normalize(-offset, dim=-1)
    world_up = torch.zeros_like(forward)
    world_up[:, 2] = 1.0
    right = F.normalize(torch.cross(forward, world_up, dim=-1), dim=-1)
    camera_up = F.normalize(torch.cross(right, forward, dim=-1), dim=-1)
    radius_feature = torch.log(radius_m.to(dtype=dtype) / 1.1).unsqueeze(-1)
    features = torch.cat([right, camera_up, forward, radius_feature], dim=-1)
    if features.shape != (clock_hour.shape[0], CAMERA_FEATURE_DIM):
        raise RuntimeError("orbit camera feature shape drifted")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("orbit camera features are non-finite")
    return features.to(device=device)


class PoseConditionedResidualReadout(nn.Module):
    """Correct a frozen pixel readout using continuous camera geometry."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        max_residual_fraction: float = 0.35,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 < float(max_residual_fraction) <= 1.0:
            raise ValueError("max_residual_fraction must be in (0,1]")
        self.hidden_dim = int(hidden_dim)
        self.max_residual_fraction = float(max_residual_fraction)
        self.task_embedding = nn.Embedding(len(TASKS), 32)
        self.anchor_embedding = nn.Embedding(ANCHOR_COUNT, 16)
        self.horizon_embedding = nn.Embedding(HORIZON, 16)
        input_dim = (
            3 * LATENT_DIM
            + 2
            + 2
            + 3
            + CAMERA_FEATURE_DIM
            + 32
            + 16
            + 16
        )
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def initialize_embeddings_from(self, base_readout: nn.Module) -> None:
        """Copy task/anchor/horizon identities from the frozen base readout."""

        for name in ("task_embedding", "anchor_embedding", "horizon_embedding"):
            source = getattr(base_readout, name, None)
            target = getattr(self, name)
            if not isinstance(source, nn.Embedding):
                raise TypeError(f"base readout does not provide {name}")
            if source.weight.shape != target.weight.shape:
                raise ValueError(f"base readout {name} shape drifted")
            target.weight.data.copy_(source.weight.detach())

    def forward(
        self,
        *,
        predicted_state: torch.Tensor,
        current_state: torch.Tensor,
        base_prediction_q_px: torch.Tensor,
        current_q_px: torch.Tensor,
        current_keypoints_xyz: torch.Tensor,
        task_index: torch.Tensor,
        camera_features: torch.Tensor,
        image_size: tuple[int, int],
    ) -> PoseReadoutOutput:
        batch_size = int(predicted_state.shape[0])
        if predicted_state.shape != (batch_size, HORIZON, STATE_SLOTS, LATENT_DIM):
            raise ValueError("predicted_state must have shape [B,8,4,192]")
        if current_state.shape != (batch_size, STATE_SLOTS, LATENT_DIM):
            raise ValueError("current_state must have shape [B,4,192]")
        if base_prediction_q_px.shape != (batch_size, HORIZON, ANCHOR_COUNT, 2):
            raise ValueError("base_prediction_q_px must have shape [B,8,3,2]")
        if current_q_px.shape != (batch_size, ANCHOR_COUNT, 2):
            raise ValueError("current_q_px must have shape [B,3,2]")
        if current_keypoints_xyz.shape != (batch_size, ANCHOR_COUNT, 3):
            raise ValueError("current_keypoints_xyz must have shape [B,3,3]")
        if task_index.shape != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        if bool(torch.any((task_index < 0) | (task_index >= len(TASKS)))):
            raise ValueError("task_index is outside the seven-task panel")
        if camera_features.shape != (batch_size, CAMERA_FEATURE_DIM):
            raise ValueError(f"camera_features must have shape [B,{CAMERA_FEATURE_DIM}]")
        for value in (
            predicted_state,
            current_state,
            base_prediction_q_px,
            current_q_px,
            current_keypoints_xyz,
            camera_features,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("pose readout input contains non-finite values")

        width, height = (int(image_size[0]), int(image_size[1]))
        if width <= 1 or height <= 1:
            raise ValueError("image_size must be at least 2x2")
        scale = current_q_px.new_tensor([width - 1, height - 1]).float()
        current_q = current_q_px.float() / scale
        base_q = base_prediction_q_px.float() / scale
        predicted_anchor = predicted_state[:, :, 1:].float()
        current_anchor = current_state[:, None, 1:].float().expand(
            -1, HORIZON, -1, -1
        )
        current_q = current_q[:, None].expand(-1, HORIZON, -1, -1)
        current_xyz = current_keypoints_xyz[:, None].float().expand(
            -1, HORIZON, -1, -1
        )
        camera = camera_features[:, None, None].float().expand(
            -1, HORIZON, ANCHOR_COUNT, -1
        )
        task = self.task_embedding(task_index.long())[:, None, None].expand(
            -1, HORIZON, ANCHOR_COUNT, -1
        )
        anchor_ids = torch.arange(ANCHOR_COUNT, device=predicted_state.device)
        anchor = self.anchor_embedding(anchor_ids)[None, None].expand(
            batch_size, HORIZON, -1, -1
        )
        horizon_ids = torch.arange(HORIZON, device=predicted_state.device)
        horizon = self.horizon_embedding(horizon_ids)[None, :, None].expand(
            batch_size, -1, ANCHOR_COUNT, -1
        )
        features = torch.cat(
            [
                predicted_anchor,
                current_anchor,
                predicted_anchor - current_anchor,
                base_q,
                current_q,
                current_xyz,
                camera,
                task,
                anchor,
                horizon,
            ],
            dim=-1,
        )
        residual_normalized = self.max_residual_fraction * torch.tanh(
            self.head(features)
        )
        residual_q_px = residual_normalized * scale
        predicted_q_px = base_prediction_q_px.float() + residual_q_px
        return PoseReadoutOutput(
            predicted_q_px=predicted_q_px,
            residual_q_px=residual_q_px,
        )


def pose_readout_checkpoint(
    model: PoseConditionedResidualReadout,
    *,
    step: int,
    seed: int,
    base_checkpoint: str,
    raw_dataset: str,
    reference_mode: str = "base_prediction",
) -> dict[str, Any]:
    if str(reference_mode) not in REFERENCE_MODES:
        raise ValueError("unknown pose readout reference mode")
    return {
        "checkpoint_type": POSE_READOUT_CHECKPOINT_TYPE,
        "protocol": POSE_READOUT_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "base_checkpoint": str(base_checkpoint),
        "raw_dataset": str(raw_dataset),
        "reference_mode": str(reference_mode),
        "base_sigreg_weight": SIGREG_WEIGHT,
        "hidden_dim": int(model.hidden_dim),
        "max_residual_fraction": float(model.max_residual_fraction),
        "model_state_dict": model.state_dict(),
    }


def load_pose_readout_checkpoint(
    checkpoint: Mapping[str, Any],
) -> PoseConditionedResidualReadout:
    if checkpoint.get("checkpoint_type") != POSE_READOUT_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a pose-conditioned readout")
    if checkpoint.get("protocol") != POSE_READOUT_PROTOCOL:
        raise ValueError("pose readout protocol drifted")
    if float(checkpoint.get("base_sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("pose readout base SIGReg drifted")
    if str(checkpoint.get("reference_mode", "base_prediction")) not in REFERENCE_MODES:
        raise ValueError("pose readout reference mode drifted")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("pose readout checkpoint is incomplete")
    model = PoseConditionedResidualReadout(
        hidden_dim=int(checkpoint["hidden_dim"]),
        max_residual_fraction=float(checkpoint["max_residual_fraction"]),
    )
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "CAMERA_FEATURE_DIM",
    "POSE_READOUT_CHECKPOINT_TYPE",
    "POSE_READOUT_PROTOCOL",
    "REFERENCE_MODES",
    "PoseConditionedResidualReadout",
    "PoseReadoutOutput",
    "load_pose_readout_checkpoint",
    "orbit_camera_basis_features",
    "pose_readout_checkpoint",
]
