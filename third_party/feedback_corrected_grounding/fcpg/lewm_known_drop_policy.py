"""Utilities for the known-drop policy-token control preflight."""

from __future__ import annotations

from collections.abc import Mapping

import torch


KNOWN_DROP_METHODS = ("clean", "measurement", "hold_last", "oracle")


def drop_is_active(decision_index: int, *, start: int, duration: int) -> bool:
    if decision_index < 0:
        raise ValueError("decision_index must be non-negative")
    if start < 0:
        raise ValueError("drop start must be non-negative")
    if duration < 0:
        raise ValueError("drop duration must be non-negative")
    return start <= decision_index < start + duration


def occlude_rgb_at_keypoints(
    batch: Mapping[str, torch.Tensor],
    *,
    radius_px: int,
    value: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return a shallow batch copy with RGB squares centered at ``q_true``."""

    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    if not 0.0 <= value <= 1.0:
        raise ValueError("occlusion value must lie in [0, 1]")
    if "rgb" not in batch or "q_true" not in batch:
        raise KeyError("known-drop occlusion requires rgb and q_true")
    rgb = batch["rgb"]
    points = batch["q_true"]
    if rgb.ndim != 5 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape [B,H,3,Y,X], got {tuple(rgb.shape)}")
    if (
        points.ndim != 4
        or points.shape[:2] != rgb.shape[:2]
        or points.shape[-1] != 2
    ):
        raise ValueError(
            "q_true must have shape [B,H,K,2] aligned with rgb, "
            f"got {tuple(points.shape)}"
        )
    if not torch.is_floating_point(rgb) or not torch.is_floating_point(points):
        raise TypeError("rgb and q_true must be floating tensors")
    if rgb.device != points.device:
        raise ValueError("rgb and q_true must be on the same device")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("q_true contains non-finite coordinates")

    result = dict(batch)
    occluded = rgb.clone()
    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    rounded = torch.round(points).to(dtype=torch.long)
    for batch_index in range(int(rgb.shape[0])):
        for history_index in range(int(rgb.shape[1])):
            for anchor_index in range(int(points.shape[2])):
                x = int(rounded[batch_index, history_index, anchor_index, 0])
                y = int(rounded[batch_index, history_index, anchor_index, 1])
                x0, x1 = max(0, x - radius_px), min(width, x + radius_px + 1)
                y0, y1 = max(0, y - radius_px), min(height, y + radius_px + 1)
                if x0 < x1 and y0 < y1:
                    occluded[batch_index, history_index, :, y0:y1, x0:x1] = value
    result["rgb"] = occluded
    return result


def grounded_tokens_from_memory(
    memory: torch.Tensor,
    *,
    num_keypoints: int,
) -> torch.Tensor:
    """Extract the identity-preserving anchor block from policy memory."""

    if memory.ndim != 4:
        raise ValueError(
            f"policy memory must have shape [B,H,N,D], got {tuple(memory.shape)}"
        )
    expected_tokens = 2 + int(num_keypoints)
    if memory.shape[2] != expected_tokens:
        raise ValueError(
            f"policy memory contains {memory.shape[2]} tokens, expected {expected_tokens}"
        )
    return memory[:, :, 2:, :]


def attach_grounded_override(
    batch: Mapping[str, torch.Tensor],
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Select every supplied anchor token while preserving all other inputs."""

    if tokens.ndim != 4:
        raise ValueError(
            f"grounded override must have shape [B,H,K,D], got {tuple(tokens.shape)}"
        )
    result = dict(batch)
    result["grounded_token_override"] = tokens
    result["grounded_token_override_mask"] = torch.ones(
        tokens.shape[:-1], device=tokens.device, dtype=torch.bool
    )
    return result


def repeated_last_valid_tokens(
    last_valid: torch.Tensor,
    *,
    history_steps: int,
) -> torch.Tensor:
    """Repeat a last-valid ``[B,K,D]`` token set across policy history."""

    if last_valid.ndim != 3:
        raise ValueError(
            f"last_valid must have shape [B,K,D], got {tuple(last_valid.shape)}"
        )
    if history_steps <= 0:
        raise ValueError("history_steps must be positive")
    return last_valid.unsqueeze(1).expand(-1, history_steps, -1, -1)


__all__ = [
    "KNOWN_DROP_METHODS",
    "attach_grounded_override",
    "drop_is_active",
    "grounded_tokens_from_memory",
    "occlude_rgb_at_keypoints",
    "repeated_last_valid_tokens",
]
