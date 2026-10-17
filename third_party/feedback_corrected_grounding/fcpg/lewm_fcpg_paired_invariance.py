"""Utilities for FCPG-supervised paired-view LeWM invariance training."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS


PROTOCOL = "fcpg_paired_view_lewm_invariance_v1"
GROUNDING_MODES = ("nominal", "fcpg")
RAW_ACTION_DIM = 4
RAW_STEPS_PER_TOKEN = 5


def canonical_task_indices(task_names: Sequence[str] | np.ndarray) -> np.ndarray:
    """Map task names onto the shared seven-task LeWM task vocabulary."""

    lookup = {task: index for index, task in enumerate(TASKS)}
    values = np.asarray(task_names).astype(str)
    result = np.empty(values.shape, dtype=np.int64)
    for index, task in np.ndenumerate(values):
        try:
            result[index] = lookup[str(task)]
        except KeyError as exc:
            raise ValueError(f"paired-view data contains unknown task {task!r}") from exc
    return result


def grounded_valid_mask(
    valid_nominal: np.ndarray,
    points_xy: np.ndarray,
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Keep only finite corrected points that remain inside the image."""

    valid = np.asarray(valid_nominal, dtype=bool)
    points = np.asarray(points_xy, dtype=np.float32)
    if points.shape != (*valid.shape, 2):
        raise ValueError("points_xy must align with valid_nominal and end in XY")
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    finite = np.isfinite(points).all(axis=-1)
    inside = (
        (points[..., 0] >= 0.0)
        & (points[..., 0] <= float(width - 1))
        & (points[..., 1] >= 0.0)
        & (points[..., 1] <= float(height - 1))
    )
    return valid & finite & inside


def select_dynamic_grounding(
    q_nominal: np.ndarray,
    q_fcpg: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Select the only experimental input that differs across matched runs."""

    if mode not in GROUNDING_MODES:
        raise ValueError(f"unknown grounding mode {mode!r}")
    nominal = np.asarray(q_nominal, dtype=np.float32)
    corrected = np.asarray(q_fcpg, dtype=np.float32)
    if nominal.shape != corrected.shape or nominal.shape[-1] != 2:
        raise ValueError("nominal and FCPG points must share an XY shape")
    return nominal if mode == "nominal" else corrected


def repeated_action_prefix(raw_action: torch.Tensor) -> torch.Tensor:
    """Build a diagnostic H8 action prefix from one deployment-time action."""

    if raw_action.ndim != 2 or raw_action.shape[-1] != RAW_ACTION_DIM:
        raise ValueError("raw_action must have shape [B,4]")
    token = raw_action.repeat(1, RAW_STEPS_PER_TOKEN)
    return token[:, None].expand(-1, HORIZON, -1).contiguous()


def paired_alignment_loss(
    dynamic_state: torch.Tensor,
    clean_teacher_state: torch.Tensor,
    *,
    summary_weight: float = 0.25,
) -> torch.Tensor:
    """Align FCPG-grounded dynamic-view states to clean same-state targets."""

    if dynamic_state.shape != clean_teacher_state.shape:
        raise ValueError("paired states must have identical shapes")
    if dynamic_state.ndim != 3 or dynamic_state.shape[1:] != (4, 192):
        raise ValueError("paired states must have shape [B,4,192]")
    if not 0.0 <= float(summary_weight) <= 1.0:
        raise ValueError("summary_weight must lie in [0,1]")
    summary = F.mse_loss(dynamic_state[:, 0], clean_teacher_state[:, 0])
    anchors = F.mse_loss(dynamic_state[:, 1:], clean_teacher_state[:, 1:])
    return float(summary_weight) * summary + anchors


def clean_identity_loss(
    clean_state: torch.Tensor,
    clean_teacher_state: torch.Tensor,
) -> torch.Tensor:
    """Protect the source model on clean observations while adapting its encoder."""

    if clean_state.shape != clean_teacher_state.shape:
        raise ValueError("clean identity states must have identical shapes")
    return F.mse_loss(clean_state, clean_teacher_state)


def mean_state_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the mean per-slot Euclidean latent gap."""

    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("state tensors must share a non-scalar shape")
    return float(torch.linalg.vector_norm(left.float() - right.float(), dim=-1).mean())


__all__ = [
    "GROUNDING_MODES",
    "PROTOCOL",
    "canonical_task_indices",
    "clean_identity_loss",
    "grounded_valid_mask",
    "mean_state_l2",
    "paired_alignment_loss",
    "repeated_action_prefix",
    "select_dynamic_grounding",
]
