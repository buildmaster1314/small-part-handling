"""Joint seven-task visual encoder and dense action-prefix LeWM for A4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS
from fcpg.lewm_action_prefix import ActionPrefixDynamics, dense_prefix_mse
from third_party.lewm.module import SIGReg


JOINT_PROTOCOL = "a4_joint_visual_action_prefix_v1"
JOINT_CHECKPOINT_TYPE = "a4_joint_visual_action_prefix"
STATE_SLOTS = 4
ANCHOR_COUNT = 3
LATENT_DIM = 192
ACTION_DIM = 20
METRIC_LOSS_WEIGHT = 10.0
GRIPPER_APERTURE_LOSS_WEIGHT = 10.0
ENDPOINT_METRIC_LOSS_WEIGHT = 5.0
LOSS_PROFILE_UNIFORM = "uniform_v1"
LOSS_PROFILE_ENDPOINT_HORIZON = "endpoint_horizon_v1"
LOSS_PROFILES = (LOSS_PROFILE_UNIFORM, LOSS_PROFILE_ENDPOINT_HORIZON)


@dataclass(frozen=True)
class JointA4Output:
    current_state: torch.Tensor
    target_state: torch.Tensor
    predicted_state: torch.Tensor
    predicted_q_px: torch.Tensor
    latent_loss: torch.Tensor
    metric_loss: torch.Tensor
    gripper_aperture_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    loss: torch.Tensor
    endpoint_metric_loss: torch.Tensor | None = None


def _finite_tensor(batch: Mapping[str, Any], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not torch.is_tensor(value):
        raise KeyError(f"joint A4 batch requires tensor {name!r}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"joint A4 batch tensor {name!r} contains non-finite values")
    return value


def gripper_aperture_loss(
    predicted_q_px: torch.Tensor,
    target_q_px: torch.Tensor,
    valid: torch.Tensor,
    *,
    image_size: tuple[int, int],
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise the projected distance between the two fingertip anchors."""

    if predicted_q_px.shape != target_q_px.shape or predicted_q_px.ndim != 4:
        raise ValueError("gripper aperture points must share shape [B,H,3,2]")
    if predicted_q_px.shape[2:] != (ANCHOR_COUNT, 2):
        raise ValueError("gripper aperture points must end in [3,2]")
    if valid.shape != predicted_q_px.shape[:-1]:
        raise ValueError("gripper aperture validity must align with anchor points")
    width, height = (int(image_size[0]), int(image_size[1]))
    scale = predicted_q_px.new_tensor(
        [max(width - 1, 1), max(height - 1, 1)]
    )
    predicted_aperture = torch.linalg.vector_norm(
        (predicted_q_px[:, :, 0] - predicted_q_px[:, :, 1]) / scale,
        dim=-1,
    )
    target_aperture = torch.linalg.vector_norm(
        (target_q_px[:, :, 0] - target_q_px[:, :, 1]) / scale,
        dim=-1,
    )
    aperture_valid = (valid[:, :, 0] & valid[:, :, 1]).float()
    squared = (predicted_aperture - target_aperture).square()
    if horizon_weights is None:
        resolved_horizon_weights = torch.ones(
            predicted_q_px.shape[1],
            device=predicted_q_px.device,
            dtype=squared.dtype,
        )
    else:
        if horizon_weights.ndim != 1 or horizon_weights.shape[0] != predicted_q_px.shape[1]:
            raise ValueError("horizon_weights must have shape [H]")
        if horizon_weights.device != predicted_q_px.device:
            raise ValueError("horizon_weights must share the prediction device")
        if not horizon_weights.is_floating_point() or not bool(torch.isfinite(horizon_weights).all()):
            raise ValueError("horizon_weights must be finite floating point")
        if bool(torch.any(horizon_weights < 0.0)) or float(horizon_weights.sum()) <= 0.0:
            raise ValueError("horizon_weights must be non-negative with a positive value")
        resolved_horizon_weights = horizon_weights.to(squared.dtype)
        resolved_horizon_weights = resolved_horizon_weights / resolved_horizon_weights.mean()
    weighted_valid = aperture_valid * resolved_horizon_weights[None]
    return (squared * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)


def _resolve_loss_profile(value: str) -> str:
    if value not in LOSS_PROFILES:
        raise ValueError(f"unknown joint LeWM loss profile: {value!r}")
    return value


def _loss_horizon_weights(
    profile: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    profile = _resolve_loss_profile(profile)
    if profile == LOSS_PROFILE_UNIFORM:
        return None
    # A modest late-horizon emphasis keeps the first predictions supervised.
    weights = torch.linspace(1.0, 2.5, HORIZON, device=device, dtype=dtype)
    return weights / weights.mean()


class AnchorMetricReadout(nn.Module):
    """Decode future metric anchor locations only from predicted anchor states."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(len(TASKS), 32)
        self.anchor_embedding = nn.Embedding(ANCHOR_COUNT, 16)
        self.horizon_embedding = nn.Embedding(HORIZON, 16)
        input_dim = 3 * LATENT_DIM + 2 + 32 + 16 + 16
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        predicted_state: torch.Tensor,
        current_state: torch.Tensor,
        current_q_px: torch.Tensor,
        task_index: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if predicted_state.ndim != 4 or predicted_state.shape[1:] != (
            HORIZON,
            STATE_SLOTS,
            LATENT_DIM,
        ):
            raise ValueError("predicted_state must have shape [B,8,4,192]")
        batch_size = int(predicted_state.shape[0])
        if current_state.shape != (batch_size, STATE_SLOTS, LATENT_DIM):
            raise ValueError("current_state must have shape [B,4,192]")
        if current_q_px.shape != (batch_size, ANCHOR_COUNT, 2):
            raise ValueError("current_q_px must have shape [B,3,2]")
        if task_index.shape != (batch_size,):
            raise ValueError("task_index must have shape [B]")
        width, height = (int(image_size[0]), int(image_size[1]))
        scale = current_q_px.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        base_q = current_q_px.float() / scale
        predicted_anchor = predicted_state[:, :, 1:].float()
        current_anchor = current_state[:, None, 1:].float().expand(
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
        base = base_q[:, None].expand(-1, HORIZON, -1, -1)
        features = torch.cat(
            [
                predicted_anchor,
                current_anchor,
                predicted_anchor - current_anchor,
                base,
                task,
                anchor,
                horizon,
            ],
            dim=-1,
        )
        delta = 0.5 * torch.tanh(self.head(features))
        return (base + delta) * scale


class JointSevenTaskActionPrefixLeWM(nn.Module):
    """Fine-tune one shared frame encoder and action-prefix dynamics on seven tasks."""

    def __init__(
        self,
        *,
        frame_encoder: nn.Module,
        state_projector: nn.Module,
        dynamics: ActionPrefixDynamics,
        proprio_mean: torch.Tensor,
        proprio_std: torch.Tensor,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        loss_profile: str = LOSS_PROFILE_UNIFORM,
    ) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder
        self.state_projector = state_projector
        self.dynamics = dynamics
        self.loss_profile = _resolve_loss_profile(loss_profile)
        self.metric_readout = AnchorMetricReadout()
        self.sigreg = SIGReg(knots=int(sigreg_knots), num_proj=int(sigreg_num_proj))
        self.register_buffer("proprio_mean", proprio_mean.detach().float().clone())
        self.register_buffer("proprio_std", proprio_std.detach().float().clone())
        if self.proprio_mean.shape != (4,) or self.proprio_std.shape != (4,):
            raise ValueError("joint A4 proprio statistics must have shape [4]")
        if bool(torch.any(self.proprio_std <= 0.0)):
            raise ValueError("joint A4 proprio std must be positive")

    def encode_observations(
        self,
        *,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        q_nominal: torch.Tensor,
        q_corrected: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape [B,T,3,H,W]")
        batch_size, sequence_length, _, height, width = rgb.shape
        if proprio.shape != (batch_size, sequence_length, 4):
            raise ValueError("proprio must have shape [B,T,4]")
        expected_q = (batch_size, sequence_length, ANCHOR_COUNT, 2)
        if q_nominal.shape != expected_q or q_corrected.shape != expected_q:
            raise ValueError("q_nominal/q_corrected must have shape [B,T,3,2]")
        if keypoints_xyz.shape != (batch_size, sequence_length, ANCHOR_COUNT, 3):
            raise ValueError("keypoints_xyz must have shape [B,T,3,3]")
        if valid.shape != (batch_size, sequence_length, ANCHOR_COUNT):
            raise ValueError("valid must have shape [B,T,3]")
        pixels = rgb.reshape(-1, 3, height, width).float()
        if pixels.numel() and pixels.detach().amax() > 1.5:
            pixels = pixels / 255.0
        normalized_proprio = (
            proprio.reshape(-1, 4).float() - self.proprio_mean
        ) / self.proprio_std
        tokens = self.frame_encoder.forward_token_sequence(
            pixels=pixels,
            robot_obs=normalized_proprio,
            q_nominal_px=q_nominal.reshape(-1, ANCHOR_COUNT, 2).float(),
            q_grounded_px=q_corrected.reshape(-1, ANCHOR_COUNT, 2).float(),
            keypoints_xyz=keypoints_xyz.reshape(-1, ANCHOR_COUNT, 3).float(),
            valid=valid.reshape(-1, ANCHOR_COUNT).bool(),
        )
        if tokens.shape[1:] != (3 + ANCHOR_COUNT, LATENT_DIM):
            raise ValueError("frame encoder token sequence contract drifted")
        selected = torch.cat([tokens[:, :1], tokens[:, -ANCHOR_COUNT:]], dim=1)
        projected = self.state_projector(selected.reshape(-1, LATENT_DIM))
        return projected.reshape(batch_size, sequence_length, STATE_SLOTS, LATENT_DIM)

    def _sigreg_loss(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 4 or embedding.shape[2:] != (STATE_SLOTS, LATENT_DIM):
            raise ValueError("SIGReg embedding must have shape [B,T,4,192]")
        losses = [
            self.sigreg(embedding[:, :, slot].transpose(0, 1))
            for slot in range(STATE_SLOTS)
        ]
        return torch.stack(losses).mean()

    def forward(self, batch: Mapping[str, Any]) -> JointA4Output:
        current_rgb = _finite_tensor(batch, "current_rgb")[:, None]
        future_rgb = _finite_tensor(batch, "future_rgb")
        current_proprio = _finite_tensor(batch, "current_proprio")[:, None]
        future_proprio = _finite_tensor(batch, "future_proprio")
        current_q_nominal = _finite_tensor(batch, "current_q_nominal")[:, None]
        future_q_nominal = _finite_tensor(batch, "future_q_nominal")
        current_q_corrected = _finite_tensor(batch, "current_q_corrected")[:, None]
        future_q_corrected = _finite_tensor(batch, "future_q_corrected")
        current_xyz = _finite_tensor(batch, "current_keypoints_xyz")[:, None]
        future_xyz = _finite_tensor(batch, "future_keypoints_xyz")
        current_valid = _finite_tensor(batch, "current_valid")[:, None]
        future_valid = _finite_tensor(batch, "future_valid")
        future_action = _finite_tensor(batch, "future_action").float()
        task_index = _finite_tensor(batch, "task_index").long()
        future_q_true = _finite_tensor(batch, "future_q_true").float()

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
        )
        current_state = embedding[:, 0]
        target_state = embedding[:, 1:]
        predicted_state = self.dynamics(
            current_state,
            future_action.to(dtype=current_state.dtype),
            task_index=task_index,
        )
        horizon_weights = _loss_horizon_weights(
            self.loss_profile,
            device=predicted_state.device,
            dtype=predicted_state.dtype,
        )
        latent_loss = dense_prefix_mse(
            predicted_state,
            target_state,
            horizon_weights=horizon_weights,
        )
        image_size = (int(rgb.shape[-1]), int(rgb.shape[-2]))
        predicted_q = self.metric_readout(
            predicted_state,
            current_state,
            current_q_corrected[:, 0].float(),
            task_index,
            image_size=image_size,
        )
        scale = future_q_true.new_tensor(
            [max(image_size[0] - 1, 1), max(image_size[1] - 1, 1)]
        )
        squared = ((predicted_q - future_q_true) / scale).square().mean(dim=-1)
        weights = future_valid.float()
        metric_horizon_weights = (
            torch.ones(HORIZON, device=squared.device, dtype=squared.dtype)
            if horizon_weights is None
            else horizon_weights.to(dtype=squared.dtype)
        )
        weighted_metric_valid = weights * metric_horizon_weights[None, :, None]
        metric_loss = (squared * weighted_metric_valid).sum() / weighted_metric_valid.sum().clamp_min(1.0)
        if self.loss_profile == LOSS_PROFILE_ENDPOINT_HORIZON:
            endpoint_valid = weights[:, -1]
            endpoint_metric_loss = (squared[:, -1] * endpoint_valid).sum() / endpoint_valid.sum().clamp_min(1.0)
        else:
            endpoint_metric_loss = torch.zeros_like(metric_loss)
        gripper_aperture_loss_value = gripper_aperture_loss(
            predicted_q,
            future_q_true,
            future_valid,
            image_size=image_size,
            horizon_weights=horizon_weights,
        )
        sigreg_loss = self._sigreg_loss(embedding)
        loss = (
            latent_loss
            + METRIC_LOSS_WEIGHT * metric_loss
            + GRIPPER_APERTURE_LOSS_WEIGHT * gripper_aperture_loss_value
            + ENDPOINT_METRIC_LOSS_WEIGHT * endpoint_metric_loss
            + SIGREG_WEIGHT * sigreg_loss
        )
        return JointA4Output(
            current_state=current_state,
            target_state=target_state,
            predicted_state=predicted_state,
            predicted_q_px=predicted_q,
            latent_loss=latent_loss,
            metric_loss=metric_loss,
            gripper_aperture_loss=gripper_aperture_loss_value,
            sigreg_loss=sigreg_loss,
            loss=loss,
            endpoint_metric_loss=endpoint_metric_loss,
        )


def joint_batch_from_memmaps(
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    bundle_indices: np.ndarray,
    branch_indices: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Gather one branch per selected bundle from memory-mapped raw replay data."""

    bundles = np.asarray(bundle_indices, dtype=np.int64)
    branches = np.asarray(branch_indices, dtype=np.int64)
    if bundles.ndim != 1 or branches.shape != bundles.shape or bundles.size <= 0:
        raise ValueError("bundle_indices and branch_indices must be equal non-empty vectors")
    if int(branches.min()) < 0 or int(branches.max()) >= 8:
        raise ValueError("branch index is outside [0,7]")
    result: dict[str, torch.Tensor] = {
        "task_index": torch.from_numpy(
            np.ascontiguousarray(np.asarray(index["task_index"])[bundles])
        ).long(),
        "future_action": torch.from_numpy(
            np.ascontiguousarray(np.asarray(index["future_action"])[bundles, branches])
        ).float(),
    }
    if "camera_index" in index:
        result["camera_index"] = torch.from_numpy(
            np.ascontiguousarray(np.asarray(index["camera_index"])[bundles])
        ).long()
    for name in (
        "camera_clock_hour",
        "camera_elevation_degrees",
        "camera_radius_m",
    ):
        if name in index:
            result[name] = torch.from_numpy(
                np.ascontiguousarray(np.asarray(index[name])[bundles])
            )
    for name in (
        "current_rgb",
        "current_proprio",
        "current_q_nominal",
        "current_q_corrected",
        "current_keypoints_xyz",
        "current_valid",
        "current_q_true",
    ):
        result[name] = torch.from_numpy(
            np.ascontiguousarray(np.asarray(arrays[name])[bundles])
        )
    for name in (
        "future_rgb",
        "future_proprio",
        "future_q_nominal",
        "future_q_corrected",
        "future_keypoints_xyz",
        "future_valid",
        "future_q_true",
    ):
        result[name] = torch.from_numpy(
            np.ascontiguousarray(np.asarray(arrays[name])[bundles, branches])
        )
    camera_matrix_names = (
        "camera_intrinsic",
        "camera_extrinsic_cam_from_world",
    )
    present_camera_matrices = {
        name for name in camera_matrix_names if name in arrays
    }
    if present_camera_matrices and len(present_camera_matrices) != len(
        camera_matrix_names
    ):
        raise ValueError("joint A4 camera matrices must be supplied together")
    for name in camera_matrix_names:
        if name in arrays:
            result[name] = torch.from_numpy(
                np.ascontiguousarray(np.asarray(arrays[name])[bundles])
            ).float()
    return result


def task_balanced_sample(
    index: Mapping[str, np.ndarray],
    *,
    split: str,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample every task each step and spread samples over available cameras."""

    if batch_size < len(TASKS):
        raise ValueError("joint A4 batch size must be at least seven")
    splits = np.asarray(index["split"]).astype(str)
    tasks = np.asarray(index["task_index"], dtype=np.int64)
    cameras = (
        np.asarray(index["camera_index"], dtype=np.int64)
        if "camera_index" in index
        else None
    )
    selected = []
    equal_count = batch_size // len(TASKS) if batch_size % len(TASKS) == 0 else 1
    for task_index in range(len(TASKS)):
        task_mask = (splits == split) & (tasks == task_index)
        candidates = np.flatnonzero(task_mask)
        if candidates.size == 0:
            raise ValueError(f"split {split!r} has no bundles for task {TASKS[task_index]}")
        if cameras is None:
            chosen = rng.choice(candidates, size=equal_count, replace=True)
        else:
            available_views = np.unique(cameras[candidates])
            view_order = rng.permutation(available_views)
            chosen_rows = []
            for row in range(equal_count):
                view = int(view_order[row % len(view_order)])
                view_candidates = np.flatnonzero(task_mask & (cameras == view))
                chosen_rows.append(int(rng.choice(view_candidates)))
            chosen = np.asarray(chosen_rows, dtype=np.int64)
        selected.extend(int(value) for value in chosen)
    all_candidates = np.flatnonzero(splits == split)
    while len(selected) < batch_size:
        selected.append(int(rng.choice(all_candidates)))
    rng.shuffle(selected)
    branches = rng.integers(0, 8, size=batch_size, dtype=np.int64)
    return np.asarray(selected, dtype=np.int64), branches


def input_statistics(
    index: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    train = np.flatnonzero(np.asarray(index["split"]).astype(str) == "train")
    if train.size == 0:
        raise ValueError("joint A4 raw dataset has no train bundles")
    proprio = np.concatenate(
        [
            np.asarray(arrays["current_proprio"])[train].reshape(-1, 4),
            np.asarray(arrays["future_proprio"])[train].reshape(-1, 4),
        ],
        axis=0,
    ).astype(np.float64)
    actions = np.asarray(index["future_action"])[train].reshape(-1, ACTION_DIM).astype(np.float64)
    return {
        "proprio_mean": proprio.mean(axis=0).astype(np.float32),
        "proprio_std": np.maximum(proprio.std(axis=0), 1.0e-6).astype(np.float32),
        "action_mean": actions.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(actions.std(axis=0), 1.0e-6).astype(np.float32),
    }


def build_joint_model_from_checkpoints(
    *,
    base_checkpoint: str,
    prefix_checkpoint: Mapping[str, Any],
    statistics: Mapping[str, np.ndarray | torch.Tensor],
    seed: int,
    loss_profile: str = LOSS_PROFILE_UNIFORM,
) -> JointSevenTaskActionPrefixLeWM:
    """Initialize the joint model from the four-task visual base and seven-task head."""

    from fcpg.lewm_action_prefix_a4 import load_action_prefix_checkpoint
    from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM
    from fcpg.lewm_runtime import load_lewm_grounded_checkpoint

    base, _resolved, _payload = load_lewm_grounded_checkpoint(
        base_checkpoint,
        device=torch.device("cpu"),
    )
    if not isinstance(base, AnchorPersistentLeWM):
        raise TypeError("joint A4 base checkpoint must be AnchorPersistentLeWM")
    dynamics = load_action_prefix_checkpoint(prefix_checkpoint)
    dynamics.set_action_normalization(
        statistics["action_mean"], statistics["action_std"]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = JointSevenTaskActionPrefixLeWM(
            frame_encoder=base.frame_encoder,
            state_projector=base.projector,
            dynamics=dynamics,
            proprio_mean=torch.as_tensor(statistics["proprio_mean"]),
            proprio_std=torch.as_tensor(statistics["proprio_std"]),
            loss_profile=loss_profile,
        )
    return model


def joint_checkpoint_payload(
    model: JointSevenTaskActionPrefixLeWM,
    *,
    step: int,
    seed: int,
    raw_dataset: str,
    base_checkpoint: str,
    prefix_checkpoint: str,
    statistics: Mapping[str, np.ndarray],
    loss_profile: str | None = None,
) -> dict[str, Any]:
    resolved_loss_profile = _resolve_loss_profile(
        model.loss_profile if loss_profile is None else loss_profile
    )
    return {
        "checkpoint_type": JOINT_CHECKPOINT_TYPE,
        "protocol": JOINT_PROTOCOL,
        "step": int(step),
        "seed": int(seed),
        "tasks": list(TASKS),
        "sigreg_weight": SIGREG_WEIGHT,
        "metric_loss_weight": METRIC_LOSS_WEIGHT,
        "gripper_aperture_loss_weight": GRIPPER_APERTURE_LOSS_WEIGHT,
        "endpoint_metric_loss_weight": ENDPOINT_METRIC_LOSS_WEIGHT,
        "loss_profile": resolved_loss_profile,
        "raw_dataset": str(raw_dataset),
        "base_checkpoint": str(base_checkpoint),
        "prefix_checkpoint": str(prefix_checkpoint),
        "statistics": {
            name: np.asarray(value, dtype=np.float32).tolist()
            for name, value in statistics.items()
        },
        "model_state": model.state_dict(),
    }


def load_joint_checkpoint(
    checkpoint: Mapping[str, Any],
) -> JointSevenTaskActionPrefixLeWM:
    if checkpoint.get("checkpoint_type") != JOINT_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a joint A4 action-prefix model")
    if checkpoint.get("protocol") != JOINT_PROTOCOL:
        raise ValueError("joint A4 protocol drifted")
    if tuple(checkpoint.get("tasks", ())) != TASKS:
        raise ValueError("joint A4 task panel drifted")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("joint A4 SIGReg must be exactly 0.02")
    statistics_value = checkpoint.get("statistics")
    state = checkpoint.get("model_state")
    if not isinstance(statistics_value, Mapping) or not isinstance(state, Mapping):
        raise ValueError("joint A4 checkpoint is incomplete")
    statistics = {
        name: np.asarray(statistics_value[name], dtype=np.float32)
        for name in ("proprio_mean", "proprio_std", "action_mean", "action_std")
    }
    prefix_path = str(checkpoint.get("prefix_checkpoint", ""))
    prefix_payload = torch.load(prefix_path, map_location="cpu", weights_only=False)
    if not isinstance(prefix_payload, Mapping):
        raise TypeError("joint A4 prefix checkpoint must contain a mapping")
    model = build_joint_model_from_checkpoints(
        base_checkpoint=str(checkpoint.get("base_checkpoint", "")),
        prefix_checkpoint=prefix_payload,
        statistics=statistics,
        seed=int(checkpoint.get("seed", 0)),
        loss_profile=str(checkpoint.get("loss_profile", LOSS_PROFILE_UNIFORM)),
    )
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "ACTION_DIM",
    "ANCHOR_COUNT",
    "JOINT_CHECKPOINT_TYPE",
    "JOINT_PROTOCOL",
    "JointA4Output",
    "JointSevenTaskActionPrefixLeWM",
    "METRIC_LOSS_WEIGHT",
    "GRIPPER_APERTURE_LOSS_WEIGHT",
    "ENDPOINT_METRIC_LOSS_WEIGHT",
    "LOSS_PROFILE_UNIFORM",
    "LOSS_PROFILE_ENDPOINT_HORIZON",
    "LOSS_PROFILES",
    "build_joint_model_from_checkpoints",
    "input_statistics",
    "joint_checkpoint_payload",
    "joint_batch_from_memmaps",
    "load_joint_checkpoint",
    "task_balanced_sample",
]
