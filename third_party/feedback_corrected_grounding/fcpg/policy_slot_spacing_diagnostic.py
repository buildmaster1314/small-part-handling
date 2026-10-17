"""Privileged policy-token slot-spacing diagnostic.

The diagnostic isolates one interface question: which two clean grounded-token
timestamps should fill a two-slot policy history when the RGB/global context is
corrupted?  Hidden-clean tokens are labels used only by this diagnostic.  No
LeWM state, learned bridge, or deployable validity estimate is involved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from fcpg.lewm_known_drop_policy import attach_grounded_override


SLOT_SPACING_METHODS = (
    "exact_adjacent",
    "decision_spaced",
    "repeat_current",
    "hold_last",
)


@dataclass(frozen=True)
class SlotSpacingState:
    """Privileged clean token state maintained by the evaluator only."""

    previous_decision: torch.Tensor | None = None
    last_valid: torch.Tensor | None = None


def _validate_current(tokens: torch.Tensor, *, label: str) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"{label} must have shape [B,K,D], got {tuple(tokens.shape)}")
    if not torch.is_floating_point(tokens):
        raise TypeError(f"{label} must be floating point")
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError(f"{label} contains non-finite values")


def update_slot_spacing_state(
    state: SlotSpacingState,
    clean_current: torch.Tensor,
    *,
    active: bool,
) -> SlotSpacingState:
    """Advance hidden-clean diagnostic state after one policy decision.

    ``previous_decision`` follows every hidden-clean decision so that the
    decision-spaced condition remains exactly one controller period apart.
    ``last_valid`` freezes at the final pre-drop measurement during an active
    interval, matching conventional hold-last behavior.
    """

    _validate_current(clean_current, label="clean_current")
    last_valid = state.last_valid if active else clean_current.detach()
    return SlotSpacingState(
        previous_decision=clean_current.detach(),
        last_valid=last_valid,
    )


def slot_spacing_override_tokens(
    clean_adjacent: torch.Tensor,
    *,
    method: str,
    state: SlotSpacingState,
) -> torch.Tensor:
    """Construct one privileged two-slot override for an active decision."""

    if method not in SLOT_SPACING_METHODS:
        raise ValueError(f"unknown slot-spacing method {method!r}")
    if clean_adjacent.ndim != 4 or int(clean_adjacent.shape[1]) != 2:
        raise ValueError(
            "clean_adjacent must have shape [B,2,K,D], got "
            f"{tuple(clean_adjacent.shape)}"
        )
    if not torch.is_floating_point(clean_adjacent):
        raise TypeError("clean_adjacent must be floating point")
    if not bool(torch.isfinite(clean_adjacent).all()):
        raise ValueError("clean_adjacent contains non-finite values")

    current = clean_adjacent[:, -1]
    if method == "exact_adjacent":
        return clean_adjacent
    if method == "repeat_current":
        return current.unsqueeze(1).expand(-1, 2, -1, -1)

    source = (
        state.previous_decision
        if method == "decision_spaced"
        else state.last_valid
    )
    if source is None:
        raise RuntimeError(f"{method} requires an initialized pre-drop token state")
    _validate_current(source, label=f"{method} source")
    if tuple(source.shape) != tuple(current.shape):
        raise ValueError(
            f"{method} source shape {tuple(source.shape)} does not match current "
            f"shape {tuple(current.shape)}"
        )
    if source.device != current.device or source.dtype != current.dtype:
        raise ValueError(f"{method} source must match current device and dtype")
    if method == "decision_spaced":
        return torch.stack((source, current), dim=1)
    return source.unsqueeze(1).expand(-1, 2, -1, -1)


def attach_slot_spacing_override(
    corrupted_batch: Mapping[str, torch.Tensor],
    clean_adjacent: torch.Tensor,
    *,
    method: str,
    state: SlotSpacingState,
) -> dict[str, torch.Tensor]:
    """Attach only grounded-token overrides to a corrupted observation batch."""

    tokens = slot_spacing_override_tokens(
        clean_adjacent,
        method=method,
        state=state,
    )
    return attach_grounded_override(corrupted_batch, tokens)


__all__ = [
    "SLOT_SPACING_METHODS",
    "SlotSpacingState",
    "attach_slot_spacing_override",
    "slot_spacing_override_tokens",
    "update_slot_spacing_state",
]
