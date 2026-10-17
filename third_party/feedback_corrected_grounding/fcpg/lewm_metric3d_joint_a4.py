"""Known-geometry 3D anchor supervision for the camera-action LeWM.

The original v1 protocol predicts a bounded residual in camera coordinates.
The v2 world protocol keeps the analytic FK/action reference in the robot
frame and lets LeWM predict only the interaction/execution residual there.
The camera is used only for the final exact projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix_joint_a4 import (
    ANCHOR_COUNT,
    LATENT_DIM,
    STATE_SLOTS,
    JointA4Output,
    gripper_aperture_loss,
)
from fcpg.lewm_camera_action_joint_a4 import (
    CAMERA_ACTION_CHECKPOINT_TYPE,
    CameraActionJointSevenTaskLeWM,
    load_camera_action_joint_checkpoint,
)
from fcpg.lewm_world_anchor_projection import project_world_points_torch


METRIC3D_PROTOCOL = "a4_camera_action_joint_metric3d_v1"
METRIC3D_WORLD_PROTOCOL = "a4_camera_action_joint_metric3d_world_v2"
METRIC3D_CHECKPOINT_TYPE = "a4_camera_action_joint_metric3d"
WORLD_SCALE_M = 0.1
WORLD_LOSS_WEIGHT = 1.0
PROJECTED_METRIC_LOSS_WEIGHT = 10.0
PROJECTED_APERTURE_LOSS_WEIGHT = 10.0


@dataclass(frozen=True)
class CameraAnchor3DOutput:
    predicted_camera_xyz: torch.Tensor
    predicted_world_xyz: torch.Tensor
    predicted_q_px: torch.Tensor
    residual_camera_xyz: torch.Tensor | None = None
    residual_world_xyz: torch.Tensor | None = None


@dataclass(frozen=True)
class Metric3DJointOutput:
    current_state: torch.Tensor
    target_state: torch.Tensor
    predicted_state: torch.Tensor
    predicted_q_px: torch.Tensor
    base_predicted_q_px: torch.Tensor
    predicted_camera_xyz: torch.Tensor
    predicted_world_xyz: torch.Tensor
    latent_loss: torch.Tensor
    metric_loss: torch.Tensor
    gripper_aperture_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    world_loss: torch.Tensor
    base_loss: torch.Tensor
    loss: torch.Tensor


def _validate_points_and_extrinsic(
    points: torch.Tensor, extrinsic_cam_from_world: torch.Tensor
) -> None:
    if points.ndim < 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,...,3]")
    if extrinsic_cam_from_world.shape != (points.shape[0], 4, 4):
        raise ValueError("extrinsic must have shape [B,4,4]")
    if not bool(torch.isfinite(points).all()) or not bool(
        torch.isfinite(extrinsic_cam_from_world).all()
    ):
        raise ValueError("point transform inputs must be finite")


def world_to_camera_points_torch(
    world_xyz: torch.Tensor, extrinsic_cam_from_world: torch.Tensor
) -> torch.Tensor:
    """Transform arbitrary batched point axes into camera coordinates."""

    _validate_points_and_extrinsic(world_xyz, extrinsic_cam_from_world)
    points = world_xyz.float()
    extrinsic = extrinsic_cam_from_world.to(device=points.device, dtype=points.dtype)
    rotation = extrinsic[:, :3, :3]
    translation_shape = (points.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    translation = extrinsic[:, :3, 3].reshape(translation_shape)
    return torch.einsum("bij,b...j->b...i", rotation, points) + translation


def camera_to_world_points_torch(
    camera_xyz: torch.Tensor, extrinsic_cam_from_world: torch.Tensor
) -> torch.Tensor:
    """Transform arbitrary batched point axes into world coordinates."""

    _validate_points_and_extrinsic(camera_xyz, extrinsic_cam_from_world)
    points = camera_xyz.float()
    extrinsic = extrinsic_cam_from_world.to(device=points.device, dtype=points.dtype)
    rotation = extrinsic[:, :3, :3]
    translation_shape = (points.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    centered = points - extrinsic[:, :3, 3].reshape(translation_shape)
    return torch.einsum("bji,b...j->b...i", rotation, centered)


def command_integrated_world_xyz(
    current_keypoints_xyz: torch.Tensor, future_action: torch.Tensor
) -> torch.Tensor:
    """Integrate five raw Cartesian commands represented by each model token."""

    if current_keypoints_xyz.ndim != 3 or current_keypoints_xyz.shape[1:] != (
        ANCHOR_COUNT,
        3,
    ):
        raise ValueError("current_keypoints_xyz must have shape [B,3,3]")
    if future_action.shape != (current_keypoints_xyz.shape[0], HORIZON, 20):
        raise ValueError("future_action must have shape [B,8,20]")
    raw = future_action.float().reshape(-1, HORIZON, 5, 4)
    token_delta = 0.01 * raw[..., :3].sum(dim=2)
    cumulative = torch.cumsum(token_delta, dim=1)
    return current_keypoints_xyz[:, None].float() + cumulative[:, :, None]


class CameraAnchor3DReadout(nn.Module):
    """Decode bounded residuals in whichever metric frame the caller supplies."""

    def __init__(self, *, hidden_dim: int = 256, max_residual_m: float = 0.15) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 < float(max_residual_m) <= 0.5:
            raise ValueError("max_residual_m must be in (0,0.5]")
        self.hidden_dim = int(hidden_dim)
        self.max_residual_m = float(max_residual_m)
        self.task_embedding = nn.Embedding(len(TASKS), 32)
        self.anchor_embedding = nn.Embedding(ANCHOR_COUNT, 16)
        self.horizon_embedding = nn.Embedding(HORIZON, 16)
        input_dim = 3 * LATENT_DIM + 3 + 3 + 32 + 16 + 16
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

    def initialize_embeddings_from(self, metric_readout: nn.Module) -> None:
        for name in ("task_embedding", "anchor_embedding", "horizon_embedding"):
            source = getattr(metric_readout, name, None)
            target = getattr(self, name)
            if not isinstance(source, nn.Embedding):
                raise TypeError(f"metric readout lacks {name}")
            if source.weight.shape != target.weight.shape:
                raise ValueError(f"metric readout {name} shape drifted")
            target.weight.data.copy_(source.weight.detach())

    def forward(
        self,
        *,
        predicted_state: torch.Tensor,
        current_state: torch.Tensor,
        current_camera_xyz: torch.Tensor | None = None,
        reference_camera_xyz: torch.Tensor | None = None,
        current_world_xyz: torch.Tensor | None = None,
        reference_world_xyz: torch.Tensor | None = None,
        task_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(predicted_state.shape[0])
        if predicted_state.shape != (batch_size, HORIZON, STATE_SLOTS, LATENT_DIM):
            raise ValueError("predicted_state must have shape [B,8,4,192]")
        if current_state.shape != (batch_size, STATE_SLOTS, LATENT_DIM):
            raise ValueError("current_state must have shape [B,4,192]")
        current_xyz = current_world_xyz if current_world_xyz is not None else current_camera_xyz
        reference_xyz = (
            reference_world_xyz
            if reference_world_xyz is not None
            else reference_camera_xyz
        )
        if current_xyz is None or reference_xyz is None:
            raise ValueError("one complete metric-frame XYZ pair is required")
        if current_xyz.shape != (batch_size, ANCHOR_COUNT, 3):
            raise ValueError("current XYZ must have shape [B,3,3]")
        if reference_xyz.shape != (batch_size, HORIZON, ANCHOR_COUNT, 3):
            raise ValueError("reference XYZ must have shape [B,8,3,3]")
        if task_index.shape != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        if bool(torch.any((task_index < 0) | (task_index >= len(TASKS)))):
            raise ValueError("task_index is outside the seven-task panel")
        for value in (
            predicted_state,
            current_state,
            current_xyz,
            reference_xyz,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("3D readout input contains non-finite values")

        predicted_anchor = predicted_state[:, :, 1:].float()
        current_anchor = current_state[:, None, 1:].float().expand(
            -1, HORIZON, -1, -1
        )
        current_metric = current_xyz[:, None].float().expand(
            -1, HORIZON, -1, -1
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
                current_metric,
                reference_xyz.float(),
                task,
                anchor,
                horizon,
            ],
            dim=-1,
        )
        residual = self.max_residual_m * torch.tanh(self.head(features))
        return reference_xyz.float() + residual, residual


class Metric3DCameraActionLeWM(nn.Module):
    """Fine-tune LeWM dynamics with an explicitly projectable 3D anchor state."""

    def __init__(
        self,
        camera_action_model: CameraActionJointSevenTaskLeWM,
        *,
        hidden_dim: int = 256,
        max_residual_m: float = 0.15,
        residual_frame: str = "camera",
    ) -> None:
        super().__init__()
        if not isinstance(camera_action_model, CameraActionJointSevenTaskLeWM):
            raise TypeError("camera_action_model has the wrong type")
        if residual_frame not in {"camera", "world"}:
            raise ValueError("residual_frame must be 'camera' or 'world'")
        self.camera_action_model = camera_action_model
        self.residual_frame = str(residual_frame)
        self.metric3d_readout = CameraAnchor3DReadout(
            hidden_dim=int(hidden_dim), max_residual_m=float(max_residual_m)
        )
        self.metric3d_readout.initialize_embeddings_from(self.metric_readout)

    @property
    def frame_encoder(self) -> nn.Module:
        return self.camera_action_model.frame_encoder

    @property
    def state_projector(self) -> nn.Module:
        return self.camera_action_model.state_projector

    @property
    def dynamics(self) -> nn.Module:
        return self.camera_action_model.dynamics

    @property
    def metric_readout(self) -> nn.Module:
        return self.camera_action_model.metric_readout

    def encode_observations(self, **kwargs: Any) -> torch.Tensor:
        """Expose the frozen/base observation interface for deployment tools."""

        return self.camera_action_model.encode_observations(**kwargs)

    def transform_future_action(
        self, future_action: torch.Tensor, batch: Mapping[str, Any]
    ) -> torch.Tensor:
        return self.camera_action_model.transform_future_action(future_action, batch)

    def decode_3d(
        self,
        *,
        predicted_state: torch.Tensor,
        current_state: torch.Tensor,
        current_keypoints_xyz: torch.Tensor,
        future_action: torch.Tensor,
        task_index: torch.Tensor,
        camera_intrinsic: torch.Tensor,
        extrinsic_cam_from_world: torch.Tensor,
    ) -> CameraAnchor3DOutput:
        reference_world = command_integrated_world_xyz(
            current_keypoints_xyz, future_action
        )
        if self.residual_frame == "world":
            predicted_world, residual_world = self.metric3d_readout(
                predicted_state=predicted_state,
                current_state=current_state,
                current_world_xyz=current_keypoints_xyz,
                reference_world_xyz=reference_world,
                task_index=task_index,
            )
            predicted_camera = world_to_camera_points_torch(
                predicted_world, extrinsic_cam_from_world
            )
            residual_camera = None
        else:
            current_camera = world_to_camera_points_torch(
                current_keypoints_xyz, extrinsic_cam_from_world
            )
            reference_camera = world_to_camera_points_torch(
                reference_world, extrinsic_cam_from_world
            )
            predicted_camera, residual_camera = self.metric3d_readout(
                predicted_state=predicted_state,
                current_state=current_state,
                current_camera_xyz=current_camera,
                reference_camera_xyz=reference_camera,
                task_index=task_index,
            )
            predicted_world = camera_to_world_points_torch(
                predicted_camera, extrinsic_cam_from_world
            )
            residual_world = predicted_world - reference_world
        projected = project_world_points_torch(
            predicted_world, camera_intrinsic, extrinsic_cam_from_world
        ).pixels
        return CameraAnchor3DOutput(
            predicted_camera_xyz=predicted_camera,
            predicted_world_xyz=predicted_world,
            predicted_q_px=projected,
            residual_camera_xyz=residual_camera,
            residual_world_xyz=residual_world,
        )

    def forward(self, batch: Mapping[str, Any]) -> Metric3DJointOutput:
        for name in (
            "current_keypoints_xyz",
            "future_keypoints_xyz",
            "future_action",
            "future_q_true",
            "future_valid",
            "task_index",
            "camera_intrinsic",
            "camera_extrinsic_cam_from_world",
        ):
            if name not in batch or not torch.is_tensor(batch[name]):
                raise KeyError(f"metric3d batch requires tensor {name!r}")
        base: JointA4Output = self.camera_action_model(batch)
        decoded = self.decode_3d(
            predicted_state=base.predicted_state,
            current_state=base.current_state,
            current_keypoints_xyz=batch["current_keypoints_xyz"].float(),
            future_action=batch["future_action"].float(),
            task_index=batch["task_index"].long(),
            camera_intrinsic=batch["camera_intrinsic"].float(),
            extrinsic_cam_from_world=batch[
                "camera_extrinsic_cam_from_world"
            ].float(),
        )
        target_world = batch["future_keypoints_xyz"].float()
        future_valid = batch["future_valid"].bool()
        world_squared = (
            (decoded.predicted_world_xyz - target_world) / WORLD_SCALE_M
        ).square().mean(dim=-1)
        world_weights = future_valid.float()
        world_loss = (world_squared * world_weights).sum() / world_weights.sum().clamp_min(1.0)

        target_q = batch["future_q_true"].float()
        image_size = (
            int(batch["current_rgb"].shape[-1]),
            int(batch["current_rgb"].shape[-2]),
        )
        scale = target_q.new_tensor(
            [max(image_size[0] - 1, 1), max(image_size[1] - 1, 1)]
        )
        metric_squared = (
            (decoded.predicted_q_px - target_q) / scale
        ).square().mean(dim=-1)
        metric_loss = (
            metric_squared * world_weights
        ).sum() / world_weights.sum().clamp_min(1.0)
        aperture_loss = gripper_aperture_loss(
            decoded.predicted_q_px,
            target_q,
            future_valid,
            image_size=image_size,
        )
        loss = (
            base.loss
            + WORLD_LOSS_WEIGHT * world_loss
            + PROJECTED_METRIC_LOSS_WEIGHT * metric_loss
            + PROJECTED_APERTURE_LOSS_WEIGHT * aperture_loss
        )
        return Metric3DJointOutput(
            current_state=base.current_state,
            target_state=base.target_state,
            predicted_state=base.predicted_state,
            predicted_q_px=decoded.predicted_q_px,
            base_predicted_q_px=base.predicted_q_px,
            predicted_camera_xyz=decoded.predicted_camera_xyz,
            predicted_world_xyz=decoded.predicted_world_xyz,
            latent_loss=base.latent_loss,
            metric_loss=metric_loss,
            gripper_aperture_loss=aperture_loss,
            sigreg_loss=base.sigreg_loss,
            world_loss=world_loss,
            base_loss=base.loss,
            loss=loss,
        )


def build_metric3d_model(
    camera_action_checkpoint: Mapping[str, Any],
    *,
    seed: int,
    hidden_dim: int = 256,
    max_residual_m: float = 0.15,
    residual_frame: str = "camera",
) -> Metric3DCameraActionLeWM:
    if camera_action_checkpoint.get("checkpoint_type") != CAMERA_ACTION_CHECKPOINT_TYPE:
        raise ValueError("parent is not a camera-action checkpoint")
    if int(camera_action_checkpoint.get("step", -1)) != 10_000:
        raise ValueError("metric3d requires the fixed 10k camera-action parent")
    if float(camera_action_checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("metric3d requires SIGReg=0.02")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        base = load_camera_action_joint_checkpoint(camera_action_checkpoint)
        return Metric3DCameraActionLeWM(
            base,
            hidden_dim=int(hidden_dim),
            max_residual_m=float(max_residual_m),
            residual_frame=str(residual_frame),
        )


def metric3d_checkpoint_payload(
    model: Metric3DCameraActionLeWM,
    *,
    step: int,
    seed: int,
    parent_checkpoint: str,
    raw_dataset: str,
    protocol: str | None = None,
) -> dict[str, Any]:
    selected_protocol = str(protocol or (
        METRIC3D_WORLD_PROTOCOL
        if model.residual_frame == "world"
        else METRIC3D_PROTOCOL
    ))
    return {
        "checkpoint_type": METRIC3D_CHECKPOINT_TYPE,
        "protocol": selected_protocol,
        "step": int(step),
        "seed": int(seed),
        "tasks": list(TASKS),
        "sigreg_weight": SIGREG_WEIGHT,
        "parent_checkpoint": str(parent_checkpoint),
        "raw_dataset": str(raw_dataset),
        "hidden_dim": int(model.metric3d_readout.hidden_dim),
        "max_residual_m": float(model.metric3d_readout.max_residual_m),
        "residual_frame": str(model.residual_frame),
        "coordinate_contract": (
            "world_frame_known_geometry_reference_plus_bounded_residual_then_exact_projection"
            if model.residual_frame == "world"
            else "camera_frame_3d_residual_then_exact_world_projection"
        ),
        "model_state": model.state_dict(),
    }


def load_metric3d_checkpoint(checkpoint: Mapping[str, Any]) -> Metric3DCameraActionLeWM:
    if checkpoint.get("checkpoint_type") != METRIC3D_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not metric3d")
    protocol = str(checkpoint.get("protocol", ""))
    if protocol not in {METRIC3D_PROTOCOL, METRIC3D_WORLD_PROTOCOL}:
        raise ValueError("metric3d protocol drifted")
    if tuple(checkpoint.get("tasks", ())) != TASKS:
        raise ValueError("metric3d task panel drifted")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("metric3d SIGReg drifted")
    parent_path = str(checkpoint.get("parent_checkpoint", ""))
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping):
        raise TypeError("metric3d parent checkpoint must contain a mapping")
    model = build_metric3d_model(
        parent,
        seed=int(checkpoint.get("seed", 0)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        max_residual_m=float(checkpoint["max_residual_m"]),
        residual_frame=str(
            checkpoint.get(
                "residual_frame",
                "world" if protocol == METRIC3D_WORLD_PROTOCOL else "camera",
            )
        ),
    )
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("metric3d checkpoint lacks model state")
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "CameraAnchor3DOutput",
    "CameraAnchor3DReadout",
    "METRIC3D_CHECKPOINT_TYPE",
    "METRIC3D_PROTOCOL",
    "METRIC3D_WORLD_PROTOCOL",
    "Metric3DCameraActionLeWM",
    "Metric3DJointOutput",
    "build_metric3d_model",
    "camera_to_world_points_torch",
    "command_integrated_world_xyz",
    "load_metric3d_checkpoint",
    "metric3d_checkpoint_payload",
    "world_to_camera_points_torch",
]
