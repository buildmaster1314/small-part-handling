"""Continuous-camera conditioned seven-task A4 LeWM.

The original joint model encoded camera-specific RGB observations but never
received the camera coordinate frame explicitly. This module preserves that
checkpoint as an initialization and injects a smooth orbit-camera basis into
every persistent state slot before action-conditioned dynamics. A jointly
trained residual metric head then maps the pose-aware latent motion back to
image coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix import dense_prefix_mse
from fcpg.lewm_action_prefix_joint_a4 import (
    ANCHOR_COUNT,
    GRIPPER_APERTURE_LOSS_WEIGHT,
    LATENT_DIM,
    METRIC_LOSS_WEIGHT,
    STATE_SLOTS,
    JointA4Output,
    JointSevenTaskActionPrefixLeWM,
    gripper_aperture_loss,
    load_joint_checkpoint,
)
from fcpg.lewm_pose_conditioned_readout import (
    CAMERA_FEATURE_DIM,
    PoseConditionedResidualReadout,
    orbit_camera_basis_features,
)


POSE_JOINT_PROTOCOL = "a4_joint_visual_action_prefix_pose_v2"
POSE_JOINT_CHECKPOINT_TYPE = "a4_joint_visual_action_prefix_pose"
POSE_JOINT_SIGREG_WEIGHTS = (0.02, 0.09)


class CameraLatentConditioner(nn.Module):
    """Apply zero-initialized, slot-wise camera FiLM to LeWM states."""

    def __init__(self, hidden_dim: int = 256, max_modulation: float = 0.5) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("camera conditioner hidden_dim must be positive")
        if not 0.0 < float(max_modulation) <= 1.0:
            raise ValueError("max_modulation must be in (0,1]")
        self.hidden_dim = int(hidden_dim)
        self.max_modulation = float(max_modulation)
        self.state_norm = nn.LayerNorm(LATENT_DIM, elementwise_affine=False)
        self.network = nn.Sequential(
            nn.Linear(CAMERA_FEATURE_DIM, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2 * STATE_SLOTS * LATENT_DIM),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        embedding: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.ndim != 4 or embedding.shape[2:] != (STATE_SLOTS, LATENT_DIM):
            raise ValueError("embedding must have shape [B,T,4,192]")
        batch_size = int(embedding.shape[0])
        if camera_features.shape != (batch_size, CAMERA_FEATURE_DIM):
            raise ValueError(
                f"camera_features must have shape [B,{CAMERA_FEATURE_DIM}]"
            )
        if not bool(torch.isfinite(camera_features).all()):
            raise ValueError("camera_features contain non-finite values")
        modulation = self.network(camera_features.float()).reshape(
            batch_size, STATE_SLOTS, 2, LATENT_DIM
        )
        scale, shift = modulation.unbind(dim=2)
        scale = self.max_modulation * torch.tanh(scale)
        shift = self.max_modulation * torch.tanh(shift)
        normalized = self.state_norm(embedding.float())
        return embedding.float() + scale[:, None] * normalized + shift[:, None]


def camera_features_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    values = []
    for name in (
        "camera_clock_hour",
        "camera_elevation_degrees",
        "camera_radius_m",
    ):
        value = batch.get(name)
        if not torch.is_tensor(value):
            raise KeyError(f"pose-conditioned A4 batch requires tensor {name!r}")
        values.append(value)
    clock_hour, elevation, radius = values
    return orbit_camera_basis_features(
        clock_hour.long(),
        elevation.float(),
        radius.float(),
    )


class PoseConditionedJointSevenTaskActionPrefixLeWM(nn.Module):
    """Joint LeWM with explicit continuous camera coordinates."""

    def __init__(
        self,
        *,
        base_model: JointSevenTaskActionPrefixLeWM,
        camera_hidden_dim: int = 256,
        readout_hidden_dim: int = 256,
        sigreg_weight: float = SIGREG_WEIGHT,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, JointSevenTaskActionPrefixLeWM):
            raise TypeError("base_model must be the joint seven-task A4 LeWM")
        self.base_model = base_model
        self.camera_conditioner = CameraLatentConditioner(
            hidden_dim=int(camera_hidden_dim)
        )
        self.pose_readout = PoseConditionedResidualReadout(
            hidden_dim=int(readout_hidden_dim),
            max_residual_fraction=0.35,
        )
        self.pose_readout.initialize_embeddings_from(base_model.metric_readout)
        self.camera_hidden_dim = int(camera_hidden_dim)
        self.readout_hidden_dim = int(readout_hidden_dim)
        self.sigreg_weight = float(sigreg_weight)
        if self.sigreg_weight not in POSE_JOINT_SIGREG_WEIGHTS:
            raise ValueError(
                f"pose-conditioned SIGReg must be one of {POSE_JOINT_SIGREG_WEIGHTS}"
            )

    @property
    def frame_encoder(self) -> nn.Module:
        return self.base_model.frame_encoder

    @property
    def state_projector(self) -> nn.Module:
        return self.base_model.state_projector

    @property
    def dynamics(self) -> nn.Module:
        return self.base_model.dynamics

    @property
    def metric_readout(self) -> nn.Module:
        return self.base_model.metric_readout

    def encode_observations(
        self,
        *,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        q_nominal: torch.Tensor,
        q_corrected: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        valid: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.base_model.encode_observations(
            rgb=rgb,
            proprio=proprio,
            q_nominal=q_nominal,
            q_corrected=q_corrected,
            keypoints_xyz=keypoints_xyz,
            valid=valid,
        )
        return self.camera_conditioner(embedding, camera_features)

    def decode_metric(
        self,
        *,
        predicted_state: torch.Tensor,
        current_state: torch.Tensor,
        current_q_px: torch.Tensor,
        current_keypoints_xyz: torch.Tensor,
        task_index: torch.Tensor,
        camera_features: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        base_prediction = self.base_model.metric_readout(
            predicted_state,
            current_state,
            current_q_px,
            task_index,
            image_size=image_size,
        )
        return self.pose_readout(
            predicted_state=predicted_state,
            current_state=current_state,
            base_prediction_q_px=base_prediction,
            current_q_px=current_q_px,
            current_keypoints_xyz=current_keypoints_xyz,
            task_index=task_index,
            camera_features=camera_features,
            image_size=image_size,
        ).predicted_q_px

    def forward(self, batch: Mapping[str, Any]) -> JointA4Output:
        camera_features = camera_features_from_batch(batch)
        current_rgb = batch["current_rgb"][:, None]
        future_rgb = batch["future_rgb"]
        current_proprio = batch["current_proprio"][:, None]
        future_proprio = batch["future_proprio"]
        current_q_nominal = batch["current_q_nominal"][:, None]
        future_q_nominal = batch["future_q_nominal"]
        current_q_corrected = batch["current_q_corrected"][:, None]
        future_q_corrected = batch["future_q_corrected"]
        current_xyz = batch["current_keypoints_xyz"][:, None]
        future_xyz = batch["future_keypoints_xyz"]
        current_valid = batch["current_valid"][:, None]
        future_valid = batch["future_valid"].bool()
        future_action = batch["future_action"].float()
        task_index = batch["task_index"].long()
        future_q_true = batch["future_q_true"].float()

        for name, value in (
            ("current_rgb", current_rgb),
            ("future_rgb", future_rgb),
            ("current_proprio", current_proprio),
            ("future_proprio", future_proprio),
            ("future_action", future_action),
            ("future_q_true", future_q_true),
        ):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"pose-conditioned batch tensor {name!r} is non-finite")

        rgb = torch.cat([current_rgb, future_rgb], dim=1)
        proprio = torch.cat([current_proprio, future_proprio], dim=1)
        q_nominal = torch.cat([current_q_nominal, future_q_nominal], dim=1)
        q_corrected = torch.cat([current_q_corrected, future_q_corrected], dim=1)
        xyz = torch.cat([current_xyz, future_xyz], dim=1)
        valid = torch.cat([current_valid, future_valid], dim=1)
        embedding = self.encode_observations(
            rgb=rgb,
            proprio=proprio,
            q_nominal=q_nominal,
            q_corrected=q_corrected,
            keypoints_xyz=xyz,
            valid=valid,
            camera_features=camera_features,
        )
        current_state = embedding[:, 0]
        target_state = embedding[:, 1:]
        predicted_state = self.dynamics(
            current_state,
            future_action.to(dtype=current_state.dtype),
            task_index=task_index,
        )
        latent_loss = dense_prefix_mse(predicted_state, target_state)
        image_size = (int(rgb.shape[-1]), int(rgb.shape[-2]))
        predicted_q = self.decode_metric(
            predicted_state=predicted_state,
            current_state=current_state,
            current_q_px=current_q_corrected[:, 0].float(),
            current_keypoints_xyz=current_xyz[:, 0].float(),
            task_index=task_index,
            camera_features=camera_features,
            image_size=image_size,
        )
        scale = future_q_true.new_tensor(
            [max(image_size[0] - 1, 1), max(image_size[1] - 1, 1)]
        )
        squared = ((predicted_q - future_q_true) / scale).square().mean(dim=-1)
        weights = future_valid.float()
        metric_loss = (squared * weights).sum() / weights.sum().clamp_min(1.0)
        aperture_loss = gripper_aperture_loss(
            predicted_q,
            future_q_true,
            future_valid,
            image_size=image_size,
        )
        sigreg_loss = self.base_model._sigreg_loss(embedding)
        loss = (
            latent_loss
            + METRIC_LOSS_WEIGHT * metric_loss
            + GRIPPER_APERTURE_LOSS_WEIGHT * aperture_loss
            + self.sigreg_weight * sigreg_loss
        )
        return JointA4Output(
            current_state=current_state,
            target_state=target_state,
            predicted_state=predicted_state,
            predicted_q_px=predicted_q,
            latent_loss=latent_loss,
            metric_loss=metric_loss,
            gripper_aperture_loss=aperture_loss,
            sigreg_loss=sigreg_loss,
            loss=loss,
        )


def build_pose_conditioned_joint_model(
    parent_checkpoint: Mapping[str, Any],
    *,
    seed: int,
    camera_hidden_dim: int = 256,
    readout_hidden_dim: int = 256,
    sigreg_weight: float = SIGREG_WEIGHT,
) -> PoseConditionedJointSevenTaskActionPrefixLeWM:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        base_model = load_joint_checkpoint(parent_checkpoint)
        return PoseConditionedJointSevenTaskActionPrefixLeWM(
            base_model=base_model,
            camera_hidden_dim=int(camera_hidden_dim),
            readout_hidden_dim=int(readout_hidden_dim),
            sigreg_weight=float(sigreg_weight),
        )


def pose_joint_checkpoint_payload(
    model: PoseConditionedJointSevenTaskActionPrefixLeWM,
    *,
    step: int,
    seed: int,
    parent_checkpoint: str,
    raw_dataset: str,
) -> dict[str, Any]:
    return {
        "checkpoint_type": POSE_JOINT_CHECKPOINT_TYPE,
        "protocol": POSE_JOINT_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "tasks": list(TASKS),
        "sigreg_weight": float(model.sigreg_weight),
        "camera_feature_dim": CAMERA_FEATURE_DIM,
        "camera_hidden_dim": int(model.camera_hidden_dim),
        "readout_hidden_dim": int(model.readout_hidden_dim),
        "parent_checkpoint": str(parent_checkpoint),
        "raw_dataset": str(raw_dataset),
        "model_state": model.state_dict(),
    }


def load_pose_conditioned_joint_checkpoint(
    checkpoint: Mapping[str, Any],
) -> PoseConditionedJointSevenTaskActionPrefixLeWM:
    if checkpoint.get("checkpoint_type") != POSE_JOINT_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a pose-conditioned joint A4 model")
    if checkpoint.get("protocol") != POSE_JOINT_PROTOCOL:
        raise ValueError("pose-conditioned joint A4 protocol drifted")
    if tuple(checkpoint.get("tasks", ())) != TASKS:
        raise ValueError("pose-conditioned joint A4 task panel drifted")
    sigreg_weight = float(checkpoint.get("sigreg_weight", -1.0))
    if sigreg_weight not in POSE_JOINT_SIGREG_WEIGHTS:
        raise ValueError("pose-conditioned joint A4 SIGReg is unsupported")
    parent_path = str(checkpoint.get("parent_checkpoint", ""))
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping):
        raise TypeError("pose-conditioned parent checkpoint must contain a mapping")
    model = build_pose_conditioned_joint_model(
        parent,
        seed=int(checkpoint.get("seed", 0)),
        camera_hidden_dim=int(checkpoint.get("camera_hidden_dim", 256)),
        readout_hidden_dim=int(checkpoint.get("readout_hidden_dim", 256)),
        sigreg_weight=sigreg_weight,
    )
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("pose-conditioned joint A4 checkpoint is incomplete")
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "POSE_JOINT_CHECKPOINT_TYPE",
    "POSE_JOINT_PROTOCOL",
    "POSE_JOINT_SIGREG_WEIGHTS",
    "CameraLatentConditioner",
    "PoseConditionedJointSevenTaskActionPrefixLeWM",
    "build_pose_conditioned_joint_model",
    "camera_features_from_batch",
    "load_pose_conditioned_joint_checkpoint",
    "pose_joint_checkpoint_payload",
]
