"""Deterministic SO-101 V9 composition helpers.

V9 keeps the task-transferred temporal backbone used by the real FCPG bridge
and adds the already-trained multi-scale memory layers from V6.  The visual
base must be byte-identical before the temporal backbone can be transplanted.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


V9_PROTOCOL = "so101-state-residual-lewm-v9-local-multiscale"
V6_PROTOCOL = "so101-state-residual-lewm-v6-multiscale-memory"
MEMORY_PROTOCOLS = frozenset(
    {
        V6_PROTOCOL,
        "so101-state-residual-lewm-v7-reliability-memory",
        V9_PROTOCOL,
    }
)


def compose_v9_state_dict(
    memory_state: Mapping[str, torch.Tensor],
    local_backbone_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Replace a V6 backbone with a compatible task-local temporal backbone."""

    prefix = "backbone."
    embedded_keys = {
        key.removeprefix(prefix) for key in memory_state if key.startswith(prefix)
    }
    local_keys = set(local_backbone_state)
    if embedded_keys != local_keys:
        missing = sorted(local_keys - embedded_keys)
        unexpected = sorted(embedded_keys - local_keys)
        raise ValueError(
            "V9 backbone keys differ: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    for key, local_value in local_backbone_state.items():
        memory_value = memory_state[prefix + key]
        if memory_value.shape != local_value.shape:
            raise ValueError(f"V9 backbone shape differs for {key}")
        if key.startswith("base.") and not torch.equal(memory_value, local_value):
            raise ValueError(
                f"V9 visual base differs for {key}; FCPG latent compatibility is not proven"
            )

    composed = dict(memory_state)
    for key, value in local_backbone_state.items():
        composed[prefix + key] = value.detach().clone()
    return composed


def assert_v9_backbone_matches(
    memory_state: Mapping[str, torch.Tensor],
    local_backbone_state: Mapping[str, torch.Tensor],
) -> None:
    """Fail closed unless a loaded V9 checkpoint contains its declared backbone."""

    prefix = "backbone."
    for key, expected in local_backbone_state.items():
        actual = memory_state.get(prefix + key)
        if actual is None or actual.shape != expected.shape or not torch.equal(
            actual, expected
        ):
            raise ValueError(f"V9 checkpoint does not contain the local backbone: {key}")


def should_use_memory(
    policy: str,
    *,
    model_available: bool,
    memory_available: bool,
    measurement_reliable: bool,
) -> bool:
    """Resolve V9 memory use without changing the clean local-backbone path."""

    if policy not in {"auto", "off", "always"}:
        raise ValueError("memory policy must be auto, off, or always")
    if not model_available or not memory_available or policy == "off":
        return False
    return policy == "always" or not measurement_reliable


__all__ = [
    "MEMORY_PROTOCOLS",
    "V6_PROTOCOL",
    "V9_PROTOCOL",
    "assert_v9_backbone_matches",
    "compose_v9_state_dict",
    "should_use_memory",
]
