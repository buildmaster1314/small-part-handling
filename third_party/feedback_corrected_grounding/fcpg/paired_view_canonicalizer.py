"""Small adapters for canonicalizing paired-camera anchor states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


FEATURE_MODES = ("latent_only", "residual_context", "camera_oracle")
CONTEXT_DIM = 32


@dataclass(frozen=True)
class PairedViewAnchorCache:
    task_name: np.ndarray
    episode_id: np.ndarray
    frame_id: np.ndarray
    q_nominal: np.ndarray
    q_true: np.ndarray
    q_dynamic_corrected: np.ndarray
    q_clean_corrected: np.ndarray
    valid_nominal: np.ndarray
    valid_true: np.ndarray
    keypoints_xyz: np.ndarray
    true_extrinsic: np.ndarray
    base_extrinsic: np.ndarray
    trajectory_phase: np.ndarray
    camera_intervention_onset: np.ndarray
    latent_dynamic_corrected: np.ndarray
    latent_clean_corrected: np.ndarray
    latent_clean_nominal: np.ndarray
    latent_dynamic_oracle: np.ndarray
    image_size: tuple[int, int]
    trajectory_mode: str
    source_data_sha256: str

    @property
    def frame_count(self) -> int:
        return int(self.q_nominal.shape[0])

    @property
    def anchor_count(self) -> int:
        return int(self.q_nominal.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.latent_dynamic_corrected.shape[-1])


def _scalar(data: np.lib.npyio.NpzFile, name: str) -> object:
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return value.item()


def load_paired_view_anchor_cache(path: str | Path) -> PairedViewAnchorCache:
    """Load and fail closed on a paired-camera anchor cache."""

    with np.load(path, allow_pickle=False) as data:
        if int(_scalar(data, "schema_version")) != 1:
            raise ValueError("unsupported paired-view cache schema")
        if str(_scalar(data, "protocol")) != "fcpg_paired_camera_anchor_cache_v1":
            raise ValueError("invalid paired-view cache protocol")
        arrays = {
            name: np.array(data[name], copy=True)
            for name in (
                "task_name",
                "episode_id",
                "frame_id",
                "q_nominal",
                "q_true",
                "q_dynamic_corrected",
                "q_clean_corrected",
                "valid_nominal",
                "valid_true",
                "keypoints_xyz",
                "true_extrinsic",
                "base_extrinsic",
                "trajectory_phase",
                "camera_intervention_onset",
                "latent_dynamic_corrected",
                "latent_clean_corrected",
                "latent_clean_nominal",
                "latent_dynamic_oracle",
            )
        }
        image_size = (
            int(_scalar(data, "image_width")),
            int(_scalar(data, "image_height")),
        )
        trajectory_mode = str(_scalar(data, "trajectory_mode"))
        source_sha = str(_scalar(data, "source_data_sha256"))

    q = arrays["q_nominal"]
    if q.ndim != 3 or q.shape[-1] != 2:
        raise ValueError("q_nominal must have shape [N,K,2]")
    frames, anchors, _ = q.shape
    for name in ("q_true", "q_dynamic_corrected", "q_clean_corrected"):
        if arrays[name].shape != q.shape:
            raise ValueError(f"{name} must match q_nominal")
    for name in ("valid_nominal", "valid_true"):
        if arrays[name].shape != (frames, anchors):
            raise ValueError(f"{name} must have shape [N,K]")
    if arrays["keypoints_xyz"].shape != (frames, anchors, 3):
        raise ValueError("keypoints_xyz must have shape [N,K,3]")
    for name in ("true_extrinsic", "base_extrinsic"):
        if arrays[name].shape != (frames, 4, 4):
            raise ValueError(f"{name} must have shape [N,4,4]")
    if arrays["trajectory_phase"].shape != (frames,):
        raise ValueError("trajectory_phase must have one value per frame")
    if arrays["camera_intervention_onset"].shape != (frames,):
        raise ValueError("camera_intervention_onset must have one value per frame")
    latent_shape = arrays["latent_dynamic_corrected"].shape
    if len(latent_shape) != 3 or latent_shape[:2] != (frames, anchors):
        raise ValueError("anchor latents must have shape [N,K,D]")
    for name in (
        "latent_clean_corrected",
        "latent_clean_nominal",
        "latent_dynamic_oracle",
    ):
        if arrays[name].shape != latent_shape:
            raise ValueError(f"{name} must match dynamic anchor latents")
    if arrays["task_name"].shape != (frames,):
        raise ValueError("task_name must have one value per frame")
    if arrays["episode_id"].shape != (frames,) or arrays["frame_id"].shape != (
        frames,
    ):
        raise ValueError("episode and frame IDs must have one value per frame")
    numeric = [
        value
        for name, value in arrays.items()
        if name
        not in {
            "task_name",
            "valid_nominal",
            "valid_true",
            "camera_intervention_onset",
        }
    ]
    if not all(np.isfinite(value).all() for value in numeric):
        raise ValueError("paired-view cache contains non-finite values")
    identities = list(
        zip(
            arrays["task_name"].astype(str).tolist(),
            arrays["episode_id"].tolist(),
            arrays["frame_id"].tolist(),
            strict=True,
        )
    )
    if len(set(identities)) != frames:
        raise ValueError("paired-view cache contains duplicate frame identities")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("paired-view image size must be positive")
    return PairedViewAnchorCache(
        **arrays,
        image_size=image_size,
        trajectory_mode=trajectory_mode,
        source_data_sha256=source_sha,
    )


def relative_camera_features(
    true_extrinsic: np.ndarray, base_extrinsic: np.ndarray
) -> np.ndarray:
    """Return a 12D zero-at-identity relative camera transform."""

    true = np.asarray(true_extrinsic, dtype=np.float64)
    base = np.asarray(base_extrinsic, dtype=np.float64)
    if true.shape != base.shape or true.ndim != 3 or true.shape[1:] != (4, 4):
        raise ValueError("camera extrinsics must have matching [N,4,4] shapes")
    relative = true @ np.linalg.inv(base)
    identity = np.eye(4, dtype=np.float64)[None]
    return (relative - identity)[:, :3, :4].reshape(len(relative), 12).astype(
        np.float32
    )


def build_anchor_context(
    cache: PairedViewAnchorCache,
    *,
    mode: str,
    clean: bool,
) -> np.ndarray:
    """Build fixed-width deployment or privileged per-anchor context."""

    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode {mode!r}")
    frames, anchors = cache.frame_count, cache.anchor_count
    result = np.zeros((frames, anchors, CONTEXT_DIM), dtype=np.float32)
    if mode == "latent_only":
        return result
    corrected = cache.q_clean_corrected if clean else cache.q_dynamic_corrected
    scale = np.asarray(
        [max(cache.image_size[0] - 1, 1), max(cache.image_size[1] - 1, 1)],
        dtype=np.float32,
    )
    residual = ((corrected - cache.q_nominal) / scale).reshape(frames, -1)
    nominal = cache.q_nominal / scale
    onehot = np.eye(anchors, dtype=np.float32)[None].repeat(frames, axis=0)
    valid = cache.valid_nominal.astype(np.float32)[..., None]
    shared_residual = np.repeat(residual[:, None], anchors, axis=1)
    deployment = np.concatenate(
        [shared_residual, nominal, cache.keypoints_xyz, onehot, valid], axis=-1
    )
    result[..., : deployment.shape[-1]] = deployment
    if mode == "camera_oracle":
        camera = (
            np.zeros((frames, 12), dtype=np.float32)
            if clean
            else relative_camera_features(cache.true_extrinsic, cache.base_extrinsic)
        )
        start = deployment.shape[-1]
        result[..., start : start + 12] = np.repeat(
            camera[:, None], anchors, axis=1
        )
    return result


class PairedViewCanonicalizer(nn.Module):
    """Zero-initialized residual map from view-specific to canonical anchors."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent and hidden dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_norm = nn.LayerNorm(self.latent_dim, elementwise_affine=False)
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim + CONTEXT_DIM, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError("latent dimension mismatch")
        if context.shape != (*latent.shape[:-1], CONTEXT_DIM):
            raise ValueError("context shape mismatch")
        values = torch.cat([self.latent_norm(latent), context], dim=-1)
        return latent + self.network(values)
