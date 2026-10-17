"""View-invariant world-anchor readout with deterministic camera projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix_joint_a4 import ANCHOR_COUNT, LATENT_DIM, STATE_SLOTS
from fcpg.structured_grounding import project_points_torch


WORLD_ANCHOR_PROTOCOL = "a4_world_anchor_camera_command_residual_v3"
WORLD_ANCHOR_CHECKPOINT_TYPE = "a4_world_anchor_command_residual"


@dataclass(frozen=True)
class ProjectedWorldAnchors:
    pixels: torch.Tensor
    depth: torch.Tensor
    valid_depth: torch.Tensor


@dataclass(frozen=True)
class WorldAnchorOutput:
    predicted_xyz_world: torch.Tensor
    delta_xyz_world: torch.Tensor


def project_world_points_torch(
    points_xyz_world: torch.Tensor,
    camera_intrinsic: torch.Tensor,
    extrinsic_cam_from_world: torch.Tensor,
    *,
    minimum_depth: float = 1.0e-6,
) -> ProjectedWorldAnchors:
    """Project world points with batched OpenCV-style camera matrices.

    ``points_xyz_world`` may contain arbitrary axes between the batch and XYZ
    dimensions. Camera matrices are shared over those axes.
    """

    if points_xyz_world.ndim < 3 or points_xyz_world.shape[-1] != 3:
        raise ValueError("points_xyz_world must have shape [B,...,3]")
    batch_size = int(points_xyz_world.shape[0])
    if camera_intrinsic.shape != (batch_size, 3, 3):
        raise ValueError("camera_intrinsic must have shape [B,3,3]")
    if extrinsic_cam_from_world.shape != (batch_size, 4, 4):
        raise ValueError("extrinsic_cam_from_world must have shape [B,4,4]")
    for value in (points_xyz_world, camera_intrinsic, extrinsic_cam_from_world):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("camera projection inputs must be finite")
    if float(minimum_depth) <= 0.0:
        raise ValueError("minimum_depth must be positive")

    points = points_xyz_world.float()
    intrinsic = camera_intrinsic.to(device=points.device, dtype=points.dtype)
    extrinsic = extrinsic_cam_from_world.to(
        device=points.device, dtype=points.dtype
    )
    flat = points.reshape(batch_size, -1, 3)
    pixels, depth = project_points_torch(intrinsic, extrinsic, flat)
    output_shape = points.shape[:-1]
    return ProjectedWorldAnchors(
        pixels=pixels.reshape(*output_shape, 2),
        depth=depth.reshape(*output_shape),
        valid_depth=(depth > float(minimum_depth)).reshape(*output_shape),
    )


class WorldAnchorDynamicsReadout(nn.Module):
    """Decode one camera-independent 3D trajectory from LeWM anchor states."""

    def __init__(
        self,
        *,
        hidden_dim: int = 2048,
        max_delta_m: float = 0.5,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 < float(max_delta_m) <= 2.0:
            raise ValueError("max_delta_m must be in (0,2]")
        self.hidden_dim = int(hidden_dim)
        self.max_delta_m = float(max_delta_m)
        self.task_embedding = nn.Embedding(len(TASKS), 32)
        self.anchor_embedding = nn.Embedding(ANCHOR_COUNT, 16)
        self.horizon_embedding = nn.Embedding(HORIZON, 16)
        input_dim = 3 * LATENT_DIM + 3 + 3 + 3 + 9 + 32 + 16 + 16
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def initialize_embeddings_from(self, base_readout: nn.Module) -> None:
        """Copy the trained task/anchor/horizon identities from the 2D head."""

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
        current_keypoints_xyz: torch.Tensor,
        camera_rotation_world_to_camera: torch.Tensor,
        task_index: torch.Tensor,
        reference_keypoints_xyz: torch.Tensor | None = None,
    ) -> WorldAnchorOutput:
        batch_size = int(predicted_state.shape[0])
        if predicted_state.shape != (batch_size, HORIZON, STATE_SLOTS, LATENT_DIM):
            raise ValueError("predicted_state must have shape [B,8,4,192]")
        if current_state.shape != (batch_size, STATE_SLOTS, LATENT_DIM):
            raise ValueError("current_state must have shape [B,4,192]")
        if current_keypoints_xyz.shape != (batch_size, ANCHOR_COUNT, 3):
            raise ValueError("current_keypoints_xyz must have shape [B,3,3]")
        if camera_rotation_world_to_camera.shape != (batch_size, 3, 3):
            raise ValueError(
                "camera_rotation_world_to_camera must have shape [B,3,3]"
            )
        if reference_keypoints_xyz is not None and reference_keypoints_xyz.shape != (
            batch_size,
            HORIZON,
            ANCHOR_COUNT,
            3,
        ):
            raise ValueError("reference_keypoints_xyz must have shape [B,8,3,3]")
        if task_index.shape != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        if bool(torch.any((task_index < 0) | (task_index >= len(TASKS)))):
            raise ValueError("task_index is outside the seven-task panel")
        for value in (
            predicted_state,
            current_state,
            current_keypoints_xyz,
            camera_rotation_world_to_camera,
            reference_keypoints_xyz,
        ):
            if value is None:
                continue
            if not bool(torch.isfinite(value).all()):
                raise ValueError("world-anchor readout input contains non-finite values")

        predicted_anchor = predicted_state[:, :, 1:].float()
        current_anchor = current_state[:, None, 1:].float().expand(
            -1, HORIZON, -1, -1
        )
        current_xyz = current_keypoints_xyz[:, None].float().expand(
            -1, HORIZON, -1, -1
        )
        reference_xyz = (
            current_xyz
            if reference_keypoints_xyz is None
            else reference_keypoints_xyz.float()
        )
        camera_rotation = camera_rotation_world_to_camera.float().reshape(
            batch_size, 1, 1, 9
        ).expand(-1, HORIZON, ANCHOR_COUNT, -1)
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
                current_xyz,
                reference_xyz,
                reference_xyz - current_xyz,
                camera_rotation,
                task,
                anchor,
                horizon,
            ],
            dim=-1,
        )
        delta_xyz = self.max_delta_m * torch.tanh(self.head(features))
        return WorldAnchorOutput(
            predicted_xyz_world=reference_xyz + delta_xyz,
            delta_xyz_world=delta_xyz,
        )


def world_anchor_checkpoint(
    model: WorldAnchorDynamicsReadout,
    *,
    step: int,
    seed: int,
    base_checkpoint: str,
    raw_dataset: str,
) -> dict[str, Any]:
    return {
        "checkpoint_type": WORLD_ANCHOR_CHECKPOINT_TYPE,
        "protocol": WORLD_ANCHOR_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "base_checkpoint": str(base_checkpoint),
        "raw_dataset": str(raw_dataset),
        "base_sigreg_weight": SIGREG_WEIGHT,
        "camera_conditioning": "continuous_world_to_camera_rotation_3x3",
        "hidden_dim": int(model.hidden_dim),
        "max_delta_m": float(model.max_delta_m),
        "model_state_dict": model.state_dict(),
    }


def load_world_anchor_checkpoint(
    checkpoint: Mapping[str, Any],
) -> WorldAnchorDynamicsReadout:
    if checkpoint.get("checkpoint_type") != WORLD_ANCHOR_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a world-anchor readout")
    if checkpoint.get("protocol") != WORLD_ANCHOR_PROTOCOL:
        raise ValueError("world-anchor protocol drifted")
    if float(checkpoint.get("base_sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("world-anchor base SIGReg drifted")
    if checkpoint.get("camera_conditioning") != (
        "continuous_world_to_camera_rotation_3x3"
    ):
        raise ValueError("world-anchor camera conditioning drifted")
    model = WorldAnchorDynamicsReadout(
        hidden_dim=int(checkpoint["hidden_dim"]),
        max_delta_m=float(checkpoint["max_delta_m"]),
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("world-anchor checkpoint lacks model_state_dict")
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "ProjectedWorldAnchors",
    "WORLD_ANCHOR_CHECKPOINT_TYPE",
    "WORLD_ANCHOR_PROTOCOL",
    "WorldAnchorDynamicsReadout",
    "WorldAnchorOutput",
    "load_world_anchor_checkpoint",
    "project_world_points_torch",
    "world_anchor_checkpoint",
]
