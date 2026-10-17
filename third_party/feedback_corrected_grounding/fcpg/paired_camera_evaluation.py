"""Metrics for same-state clean/dynamic camera pairs.

The paired renderer changes only the camera.  These helpers therefore measure
how much FCPG recovers the image-space anchors and how much camera variation
remains in the downstream anchor representation without treating camera motion
as robot dynamics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def camera_phase_labels(phase: np.ndarray) -> np.ndarray:
    """Map trajectory phase to stable pre/moving/post labels."""

    values = np.asarray(phase, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("trajectory phase must be a finite vector")
    labels = np.full(values.shape, "moving", dtype="<U8")
    labels[values <= 1e-6] = "pre"
    labels[values >= 1.0 - 1e-6] = "post"
    return labels


def _metric(values: np.ndarray, valid: np.ndarray) -> dict[str, float | int | None]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(selected.size),
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "p90": float(np.percentile(selected, 90)),
    }


def _mean(metric: Mapping[str, Any]) -> float | None:
    value = metric.get("mean")
    return None if value is None else float(value)


def _ratio(numerator: Mapping[str, Any], denominator: Mapping[str, Any]) -> float | None:
    top = _mean(numerator)
    bottom = _mean(denominator)
    if top is None or bottom is None or bottom <= 1e-12:
        return None
    return float(top / bottom)


def summarize_paired_camera_arrays(
    *,
    q_nominal: np.ndarray,
    q_true: np.ndarray,
    q_dynamic_corrected: np.ndarray,
    q_clean_corrected: np.ndarray,
    valid_nominal: np.ndarray,
    valid_true: np.ndarray,
    latent_dynamic_nominal: np.ndarray,
    latent_clean_nominal: np.ndarray,
    latent_dynamic_corrected: np.ndarray,
    latent_dynamic_oracle: np.ndarray,
    latent_clean_corrected: np.ndarray,
    task_names: Sequence[object],
    trajectory_phase: np.ndarray,
    intervention_onset: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Summarize FCPG and downstream latent invariance for paired frames."""

    q_nominal = np.asarray(q_nominal, dtype=np.float64)
    q_true = np.asarray(q_true, dtype=np.float64)
    q_dynamic_corrected = np.asarray(q_dynamic_corrected, dtype=np.float64)
    q_clean_corrected = np.asarray(q_clean_corrected, dtype=np.float64)
    if q_nominal.ndim != 3 or q_nominal.shape[-1] != 2:
        raise ValueError("pixel points must have shape [N,K,2]")
    expected_q = q_nominal.shape
    for name, value in (
        ("q_true", q_true),
        ("q_dynamic_corrected", q_dynamic_corrected),
        ("q_clean_corrected", q_clean_corrected),
    ):
        if value.shape != expected_q:
            raise ValueError(f"{name} must have shape {expected_q}")

    frames, anchors, _ = expected_q
    valid_nominal = np.asarray(valid_nominal, dtype=bool)
    valid_true = np.asarray(valid_true, dtype=bool)
    if valid_nominal.shape != (frames, anchors) or valid_true.shape != (
        frames,
        anchors,
    ):
        raise ValueError("valid masks must have shape [N,K]")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    def inside(points: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(points).all(axis=-1)
            & (points[..., 0] >= 0.0)
            & (points[..., 0] <= width - 1)
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= height - 1)
        )

    dynamic_valid = valid_nominal & valid_true & inside(q_dynamic_corrected)
    clean_valid = valid_nominal & inside(q_clean_corrected)
    paired_valid = dynamic_valid & clean_valid

    latent_arrays = {
        "latent_dynamic_nominal": np.asarray(latent_dynamic_nominal, dtype=np.float64),
        "latent_clean_nominal": np.asarray(latent_clean_nominal, dtype=np.float64),
        "latent_dynamic_corrected": np.asarray(
            latent_dynamic_corrected, dtype=np.float64
        ),
        "latent_dynamic_oracle": np.asarray(latent_dynamic_oracle, dtype=np.float64),
        "latent_clean_corrected": np.asarray(latent_clean_corrected, dtype=np.float64),
    }
    latent_shape = latent_arrays["latent_dynamic_nominal"].shape
    if len(latent_shape) != 3 or latent_shape[:2] != (frames, anchors):
        raise ValueError("latent arrays must have shape [N,K,D]")
    for name, value in latent_arrays.items():
        if value.shape != latent_shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {latent_shape}")

    values = {
        "nominal_dynamic_epe_px": np.linalg.norm(q_nominal - q_true, axis=-1),
        "fcpg_dynamic_epe_px": np.linalg.norm(
            q_dynamic_corrected - q_true, axis=-1
        ),
        "fcpg_clean_epe_px": np.linalg.norm(
            q_clean_corrected - q_nominal, axis=-1
        ),
        "nominal_cross_view_latent_l2": np.linalg.norm(
            latent_arrays["latent_dynamic_nominal"]
            - latent_arrays["latent_clean_nominal"],
            axis=-1,
        ),
        "fcpg_cross_view_latent_l2": np.linalg.norm(
            latent_arrays["latent_dynamic_corrected"]
            - latent_arrays["latent_clean_corrected"],
            axis=-1,
        ),
        "oracle_cross_view_latent_l2": np.linalg.norm(
            latent_arrays["latent_dynamic_oracle"]
            - latent_arrays["latent_clean_nominal"],
            axis=-1,
        ),
        "clean_fcpg_latent_shift_l2": np.linalg.norm(
            latent_arrays["latent_clean_corrected"]
            - latent_arrays["latent_clean_nominal"],
            axis=-1,
        ),
    }
    validity = {
        "nominal_dynamic_epe_px": valid_nominal & valid_true,
        "fcpg_dynamic_epe_px": dynamic_valid,
        "fcpg_clean_epe_px": clean_valid,
        "nominal_cross_view_latent_l2": valid_nominal & valid_true,
        "fcpg_cross_view_latent_l2": paired_valid,
        "oracle_cross_view_latent_l2": valid_nominal & valid_true,
        "clean_fcpg_latent_shift_l2": clean_valid,
    }

    tasks = np.asarray([str(value) for value in task_names])
    phase = np.asarray(trajectory_phase, dtype=np.float64)
    onset = np.asarray(intervention_onset, dtype=bool)
    if tasks.shape != (frames,) or phase.shape != (frames,) or onset.shape != (frames,):
        raise ValueError("task, phase, and onset arrays must have one value per frame")
    labels = camera_phase_labels(phase)

    def summarize(frame_mask: np.ndarray) -> dict[str, Any]:
        mask = np.asarray(frame_mask, dtype=bool)[:, None]
        metrics = {
            name: _metric(value, valid & mask)
            for name, value in values.items()
            for valid in (validity[name],)
        }
        metrics["fcpg_pixel_ratio"] = _ratio(
            metrics["fcpg_dynamic_epe_px"], metrics["nominal_dynamic_epe_px"]
        )
        metrics["fcpg_latent_ratio"] = _ratio(
            metrics["fcpg_cross_view_latent_l2"],
            metrics["nominal_cross_view_latent_l2"],
        )
        metrics["oracle_latent_ratio"] = _ratio(
            metrics["oracle_cross_view_latent_l2"],
            metrics["nominal_cross_view_latent_l2"],
        )
        metrics["fcpg_to_oracle_latent_ratio"] = _ratio(
            metrics["fcpg_cross_view_latent_l2"],
            metrics["oracle_cross_view_latent_l2"],
        )
        metrics["frames"] = int(np.asarray(frame_mask, dtype=bool).sum())
        return metrics

    groups: dict[str, Any] = {"all": summarize(np.ones(frames, dtype=bool))}
    groups["by_task"] = {
        task: summarize(tasks == task) for task in sorted(set(tasks.tolist()))
    }
    groups["by_phase"] = {
        label: summarize(labels == label)
        for label in ("pre", "moving", "post")
        if bool(np.any(labels == label))
    }
    if bool(onset.any()):
        groups["intervention_onset"] = summarize(onset)
    return groups
