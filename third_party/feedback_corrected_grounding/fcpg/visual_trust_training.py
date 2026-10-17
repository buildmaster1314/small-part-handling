"""Data preparation and small-head training for the VisualTrust probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from fcpg.visual_trust import VisualTrustHead


DEFAULT_UNSAFE_RMS_THRESHOLD = 0.10


@dataclass(frozen=True)
class VisualTrustSplit:
    visual: np.ndarray
    auxiliary: np.ndarray
    safe_target: np.ndarray
    unsafe: np.ndarray
    clean: np.ndarray
    early_reopen: np.ndarray
    valid: np.ndarray
    active: np.ndarray
    recovery_onset: np.ndarray
    task: np.ndarray
    episode: np.ndarray
    frame: np.ndarray
    family: np.ndarray
    canonical_rms: np.ndarray


def build_auxiliary_descriptors(
    q_nominal: np.ndarray,
    q_corrected: np.ndarray,
    valid_nominal: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    patch_radius: float = 24.0,
) -> np.ndarray:
    """Build current-only geometry descriptors with shape ``[N,K,7]``."""

    nominal = np.asarray(q_nominal, dtype=np.float32)
    corrected = np.asarray(q_corrected, dtype=np.float32)
    valid = np.asarray(valid_nominal)
    if nominal.ndim != 3 or nominal.shape[-1] != 2:
        raise ValueError("q_nominal must have shape [N,K,2]")
    if corrected.shape != nominal.shape:
        raise ValueError("q_corrected must match q_nominal")
    if valid.shape != nominal.shape[:2] or valid.dtype != np.bool_:
        raise TypeError("valid_nominal must be boolean with shape [N,K]")
    if image_width <= 1 or image_height <= 1:
        raise ValueError("image dimensions must be greater than one")
    if not np.isfinite(patch_radius) or patch_radius <= 0.0:
        raise ValueError("patch_radius must be positive and finite")
    if not bool(np.isfinite(nominal).all() and np.isfinite(corrected).all()):
        raise ValueError("grounding coordinates must be finite")

    scale = np.asarray([image_width - 1, image_height - 1], dtype=np.float32)
    corrected_normalized = 2.0 * corrected / scale - 1.0
    delta = (corrected - nominal) / float(patch_radius)
    delta_norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    in_bounds = (
        (corrected[..., 0] >= 0.0)
        & (corrected[..., 0] <= image_width - 1)
        & (corrected[..., 1] >= 0.0)
        & (corrected[..., 1] <= image_height - 1)
    )[..., None]
    return np.concatenate(
        (
            corrected_normalized,
            delta,
            delta_norm,
            valid[..., None].astype(np.float32),
            in_bounds.astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def canonical_feature_rms(
    observed: np.ndarray, canonical: np.ndarray
) -> np.ndarray:
    observed_array = np.asarray(observed, dtype=np.float32)
    canonical_array = np.asarray(canonical, dtype=np.float32)
    if observed_array.ndim != 3 or observed_array.shape[-1] <= 0:
        raise ValueError("observed features must have shape [N,K,C]")
    if canonical_array.shape != observed_array.shape:
        raise ValueError("canonical features must match observed features")
    if not bool(
        np.isfinite(observed_array).all() and np.isfinite(canonical_array).all()
    ):
        raise ValueError("visual features must be finite")
    return np.sqrt(
        np.mean((observed_array - canonical_array) ** 2, axis=-1)
    ).astype(np.float32)


def _episode_family(
    task: np.ndarray,
    episode: np.ndarray,
    frame_family: np.ndarray,
) -> np.ndarray:
    result = np.empty(task.shape, dtype=frame_family.dtype)
    for key in sorted(set(zip(task.astype(str), episode.astype(int), strict=True))):
        mask = (task.astype(str) == key[0]) & (episode.astype(int) == key[1])
        labels = np.unique(frame_family[mask].astype(str))
        labels = labels[labels != ""]
        if labels.size != 1:
            raise ValueError(f"episode {key} must contain exactly one corruption family")
        result[mask] = labels[0]
    return result


def load_visual_trust_split(
    data_path: str | Path,
    observed_visual_path: str | Path,
    canonical_visual_path: str | Path,
    observed_grounding_path: str | Path,
    *,
    unsafe_rms_threshold: float = DEFAULT_UNSAFE_RMS_THRESHOLD,
    patch_radius: float = 24.0,
    clean_only: bool = False,
) -> VisualTrustSplit:
    """Load deployable inputs and label-only canonical features for one split."""

    if unsafe_rms_threshold <= 0.0 or not np.isfinite(unsafe_rms_threshold):
        raise ValueError("unsafe_rms_threshold must be positive and finite")
    visual = np.asarray(np.load(observed_visual_path, mmap_mode="r"), dtype=np.float32)
    canonical = np.asarray(
        np.load(canonical_visual_path, mmap_mode="r"), dtype=np.float32
    )
    q_corrected = np.asarray(
        np.load(observed_grounding_path, mmap_mode="r"), dtype=np.float32
    )
    with np.load(data_path, allow_pickle=False) as data:
        required = {
            "rgb",
            "q_nominal",
            "valid_nominal",
            "task_name",
            "episode_id",
            "frame_id",
        }
        if not clean_only:
            required.update(
                {
                    "corruption_active",
                    "corruption_family",
                    "corruption_recovery_onset",
                }
            )
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"VisualTrust dataset is missing arrays: {missing}")
        image_shape = data["rgb"].shape
        q_nominal = data["q_nominal"].astype(np.float32)
        valid_nominal = data["valid_nominal"].astype(bool)
        task = data["task_name"].astype(str)
        episode = data["episode_id"].astype(np.int64)
        frame = data["frame_id"].astype(np.int64)
        if clean_only:
            active = np.zeros(q_nominal.shape[:2], dtype=bool)
            recovery_onset = np.zeros_like(active)
            frame_family = np.full(q_nominal.shape[0], "clean")
        else:
            active = data["corruption_active"].astype(bool)
            recovery_onset = data["corruption_recovery_onset"].astype(bool)
            frame_family = data["corruption_family"].astype(str)

    if len(image_shape) != 4 or image_shape[-1] != 3:
        raise ValueError("rgb must have shape [N,H,W,3]")
    count, height, width, _ = image_shape
    if visual.shape != canonical.shape:
        raise ValueError("observed and canonical visual caches must match")
    if visual.shape[:2] != q_nominal.shape[:2] or visual.shape[0] != count:
        raise ValueError("visual cache and dataset sample/anchor shapes differ")
    if q_corrected.shape != q_nominal.shape:
        raise ValueError("q_corrected and q_nominal shapes differ")
    if active.shape != q_nominal.shape[:2] or recovery_onset.shape != active.shape:
        raise ValueError("corruption annotations must have shape [N,K]")
    for name, value in (("task", task), ("episode", episode), ("frame", frame)):
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape [N]")

    error = canonical_feature_rms(visual, canonical)
    unsafe = error > float(unsafe_rms_threshold)
    frame_clean = ~np.any(active, axis=1)
    clean = frame_clean[:, None] & ~unsafe
    family_by_frame = (
        frame_family
        if clean_only
        else _episode_family(task, episode, frame_family)
    )
    family = np.broadcast_to(family_by_frame[:, None], active.shape).copy()
    auxiliary = build_auxiliary_descriptors(
        q_nominal,
        q_corrected,
        valid_nominal,
        image_width=int(width),
        image_height=int(height),
        patch_radius=patch_radius,
    )
    valid = valid_nominal & (auxiliary[..., -1] > 0.5) & np.isfinite(error)
    return VisualTrustSplit(
        visual=visual,
        auxiliary=auxiliary,
        safe_target=(~unsafe).astype(np.float32),
        unsafe=unsafe,
        clean=clean,
        early_reopen=active & unsafe,
        valid=valid,
        active=active,
        recovery_onset=recovery_onset,
        task=np.broadcast_to(task[:, None], active.shape).copy(),
        episode=np.broadcast_to(episode[:, None], active.shape).copy(),
        frame=np.broadcast_to(frame[:, None], active.shape).copy(),
        family=family,
        canonical_rms=error,
    )


def temporal_feature_distance(split: VisualTrustSplit) -> np.ndarray:
    """Causal RMS distance to the previous observed feature in each episode."""

    distance = np.zeros(split.visual.shape[:2], dtype=np.float32)
    task = split.task[:, 0].astype(str)
    episode = split.episode[:, 0].astype(np.int64)
    frame = split.frame[:, 0].astype(np.int64)
    for key in sorted(set(zip(task, episode, strict=True))):
        rows = np.flatnonzero((task == key[0]) & (episode == key[1]))
        rows = rows[np.argsort(frame[rows], kind="stable")]
        if rows.size > 1:
            difference = split.visual[rows[1:]] - split.visual[rows[:-1]]
            distance[rows[1:]] = np.sqrt(np.mean(difference**2, axis=-1))
    return distance


def threshold_for_clean_false_reject(
    trust: np.ndarray,
    clean: np.ndarray,
    valid: np.ndarray,
    *,
    target_rate: float = 0.05,
) -> float:
    values = np.asarray(trust, dtype=np.float64)
    mask = np.asarray(clean, dtype=bool) & np.asarray(valid, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("trust and clean/valid masks must match")
    if not 0.0 < target_rate < 1.0:
        raise ValueError("target_rate must lie in (0,1)")
    selected = values[mask]
    if selected.size == 0 or not bool(np.isfinite(selected).all()):
        raise ValueError("clean calibration set must be nonempty and finite")
    # Rejection uses the strict comparison ``trust < threshold``. Selecting
    # the order statistic at the maximum allowed reject count therefore uses
    # the clean budget without exceeding it (ties remain conservative).
    ordered = np.sort(selected)
    allowed_rejects = int(np.floor(target_rate * ordered.size))
    allowed_rejects = min(allowed_rejects, ordered.size - 1)
    return float(ordered[allowed_rejects])


def train_visual_trust_head(
    visual: np.ndarray,
    auxiliary: np.ndarray,
    safe_target: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    visual_enabled: bool,
) -> VisualTrustHead:
    """Fit one tiny current-only head with class-balanced BCE."""

    if steps <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    visual_array = np.asarray(visual, dtype=np.float32)
    auxiliary_array = np.asarray(auxiliary, dtype=np.float32)
    targets = np.asarray(safe_target, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if visual_array.shape[:2] != targets.shape or auxiliary_array.shape[:2] != targets.shape:
        raise ValueError("training arrays must share [N,K]")
    indices = np.flatnonzero(mask.reshape(-1))
    if indices.size == 0:
        raise ValueError("training split has no valid anchors")
    flat_visual = visual_array.reshape(-1, visual_array.shape[-1])[indices]
    flat_auxiliary = auxiliary_array.reshape(-1, auxiliary_array.shape[-1])[indices]
    flat_target = targets.reshape(-1)[indices]
    if not bool(np.any(flat_target == 0.0) and np.any(flat_target == 1.0)):
        raise ValueError("training split must contain safe and unsafe anchors")

    torch.manual_seed(int(seed))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # Keep the auxiliary-only control parameter matched by retaining the full
    # visual input width and replacing its values with zeros.
    model_visual_dim = visual_array.shape[-1]
    model = VisualTrustHead(
        visual_dim=int(model_visual_dim),
        auxiliary_dim=int(auxiliary_array.shape[-1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    safe_fraction = float(np.mean(flat_target))
    weights = np.where(
        flat_target > 0.5,
        0.5 / max(safe_fraction, 1e-6),
        0.5 / max(1.0 - safe_fraction, 1e-6),
    ).astype(np.float32)

    model.train()
    for _ in range(steps):
        selected = torch.randint(
            indices.size, (batch_size,), generator=generator, device="cpu"
        ).numpy()
        if visual_enabled:
            batch_visual = torch.from_numpy(flat_visual[selected]).to(device)
        else:
            batch_visual = torch.zeros(
                (batch_size, visual_array.shape[-1]), device=device
            )
        batch_auxiliary = torch.from_numpy(flat_auxiliary[selected]).to(device)
        batch_target = torch.from_numpy(flat_target[selected]).to(device)
        batch_weight = torch.from_numpy(weights[selected]).to(device)
        output = model(batch_visual[:, None], batch_auxiliary[:, None])
        loss = F.binary_cross_entropy_with_logits(
            output.logits[:, 0], batch_target, weight=batch_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict_visual_trust(
    model: VisualTrustHead,
    visual: np.ndarray,
    auxiliary: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    visual_enabled: bool,
) -> np.ndarray:
    visual_array = np.asarray(visual, dtype=np.float32)
    auxiliary_array = np.asarray(auxiliary, dtype=np.float32)
    if visual_array.shape[:2] != auxiliary_array.shape[:2]:
        raise ValueError("visual and auxiliary arrays must share [N,K]")
    result = np.empty(visual_array.shape[:2], dtype=np.float32)
    for start in range(0, visual_array.shape[0], batch_size):
        stop = min(start + batch_size, visual_array.shape[0])
        if visual_enabled:
            batch_visual = torch.from_numpy(visual_array[start:stop]).to(device)
        else:
            batch_visual = torch.zeros(
                (
                    stop - start,
                    visual_array.shape[1],
                    visual_array.shape[-1],
                ),
                device=device,
            )
        batch_auxiliary = torch.from_numpy(auxiliary_array[start:stop]).to(device)
        result[start:stop] = (
            model(batch_visual, batch_auxiliary).trust.float().cpu().numpy()
        )
    return result


__all__ = [
    "DEFAULT_UNSAFE_RMS_THRESHOLD",
    "VisualTrustSplit",
    "build_auxiliary_descriptors",
    "canonical_feature_rms",
    "load_visual_trust_split",
    "predict_visual_trust",
    "temporal_feature_distance",
    "threshold_for_clean_false_reject",
    "train_visual_trust_head",
]
