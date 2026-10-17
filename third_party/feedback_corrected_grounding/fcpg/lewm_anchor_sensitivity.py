"""Pure diagnostics for anchor-token attenuation and batch interventions.

The token intervention helpers assume the ordering used by
``GroundedFrameEncoder``: summary, global visual, robot, then anchor tokens.
They never mutate their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


TokenGroup = Literal[
    "summary",
    "global",
    "robot",
    "all_anchors",
    "single_anchor",
]

_TOKEN_GROUPS = frozenset(
    {"summary", "global", "robot", "all_anchors", "single_anchor"}
)


@dataclass(frozen=True)
class AnchorSensitivity:
    """Per-sample nominal/corrected token differences."""

    pre_anchor_rms: torch.Tensor
    post_cls_l2: torch.Tensor
    attenuation_ratio: torch.Tensor


@dataclass(frozen=True)
class AnchorSensitivitySummary:
    """JSON-friendly aggregate statistics for an anchor sensitivity batch."""

    num_samples: int
    pre_anchor_rms_mean: float
    pre_anchor_rms_median: float
    post_cls_l2_mean: float
    post_cls_l2_median: float
    attenuation_ratio_mean: float
    attenuation_ratio_median: float


def _validate_float_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_pair(
    nominal: torch.Tensor,
    corrected: torch.Tensor,
    name: str,
) -> None:
    _validate_float_tensor(nominal, f"nominal_{name}")
    _validate_float_tensor(corrected, f"corrected_{name}")
    if nominal.shape != corrected.shape:
        raise ValueError(
            f"nominal_{name} and corrected_{name} must have identical shapes"
        )
    if nominal.dtype != corrected.dtype:
        raise TypeError(
            f"nominal_{name} and corrected_{name} must have identical dtypes"
        )
    if nominal.device != corrected.device:
        raise ValueError(
            f"nominal_{name} and corrected_{name} must be on the same device"
        )


def compute_anchor_sensitivity(
    nominal_anchor_tokens: torch.Tensor,
    corrected_anchor_tokens: torch.Tensor,
    nominal_cls: torch.Tensor,
    corrected_cls: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> AnchorSensitivity:
    """Measure how nominal/corrected anchor changes survive CLS aggregation.

    Anchor tensors must have shape ``[B, K, D]`` and CLS tensors ``[B, D]``.
    The pre-aggregation magnitude is the root mean square of the ``K`` anchor
    L2 norms. The attenuation ratio is ``post_cls_l2 / max(pre_anchor_rms,
    eps)`` and is returned independently for every sample.
    """

    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a real scalar")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    _validate_pair(nominal_anchor_tokens, corrected_anchor_tokens, "anchor_tokens")
    _validate_pair(nominal_cls, corrected_cls, "cls")
    if nominal_anchor_tokens.ndim != 3:
        raise ValueError("anchor tokens must have shape [B, K, D]")
    if nominal_cls.ndim != 2:
        raise ValueError("CLS tensors must have shape [B, D]")
    batch_size, num_anchors, feature_dim = nominal_anchor_tokens.shape
    if batch_size <= 0 or num_anchors <= 0 or feature_dim <= 0:
        raise ValueError("anchor token dimensions must all be positive")
    if nominal_cls.shape != (batch_size, feature_dim):
        raise ValueError(
            "CLS tensors must have shape [B, D] matching the anchor tokens"
        )
    if nominal_anchor_tokens.dtype != nominal_cls.dtype:
        raise TypeError("anchor tokens and CLS tensors must have identical dtypes")
    if nominal_anchor_tokens.device != nominal_cls.device:
        raise ValueError("anchor tokens and CLS tensors must be on the same device")

    anchor_delta = corrected_anchor_tokens - nominal_anchor_tokens
    cls_delta = corrected_cls - nominal_cls
    pre_anchor_rms = anchor_delta.square().sum(dim=-1).mean(dim=-1).sqrt()
    post_cls_l2 = cls_delta.square().sum(dim=-1).sqrt()
    denominator = pre_anchor_rms.clamp_min(
        torch.as_tensor(eps, dtype=pre_anchor_rms.dtype, device=pre_anchor_rms.device)
    )
    attenuation_ratio = post_cls_l2 / denominator
    if not bool(torch.isfinite(attenuation_ratio).all()):
        raise ValueError("attenuation ratio is non-finite; increase eps for this dtype")
    return AnchorSensitivity(
        pre_anchor_rms=pre_anchor_rms,
        post_cls_l2=post_cls_l2,
        attenuation_ratio=attenuation_ratio,
    )


def summarize_anchor_sensitivity(
    sensitivity: AnchorSensitivity,
) -> AnchorSensitivitySummary:
    """Summarize a non-empty per-sample sensitivity result."""

    values = {
        "pre_anchor_rms": sensitivity.pre_anchor_rms,
        "post_cls_l2": sensitivity.post_cls_l2,
        "attenuation_ratio": sensitivity.attenuation_ratio,
    }
    batch_size: int | None = None
    for name, value in values.items():
        _validate_float_tensor(value, name)
        if value.ndim != 1 or value.numel() == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
        if batch_size is None:
            batch_size = int(value.numel())
        elif value.numel() != batch_size:
            raise ValueError("all sensitivity tensors must have the same length")
    assert batch_size is not None

    def mean(name: str) -> float:
        return float(values[name].detach().double().mean().cpu())

    def median(name: str) -> float:
        return float(values[name].detach().double().median().cpu())

    return AnchorSensitivitySummary(
        num_samples=batch_size,
        pre_anchor_rms_mean=mean("pre_anchor_rms"),
        pre_anchor_rms_median=median("pre_anchor_rms"),
        post_cls_l2_mean=mean("post_cls_l2"),
        post_cls_l2_median=median("post_cls_l2"),
        attenuation_ratio_mean=mean("attenuation_ratio"),
        attenuation_ratio_median=median("attenuation_ratio"),
    )


def batch_permutation_indices(
    batch_size: int,
    *,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a reproducible derangement of ``range(batch_size)``.

    No sample maps to itself, so every selected token is actually intervened
    on. Randomness is generated on CPU for device-independent reproducibility.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for batch shuffling")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63)")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(batch_size, generator=generator)
    fixed = torch.nonzero(
        permutation == torch.arange(batch_size), as_tuple=False
    ).flatten()
    if fixed.numel() == 1:
        index = int(fixed[0])
        other = (index + 1) % batch_size
        permutation[index], permutation[other] = (
            permutation[other].clone(),
            permutation[index].clone(),
        )
    elif fixed.numel() > 1:
        permutation[fixed] = permutation[fixed].roll(1)
    if bool(torch.any(permutation == torch.arange(batch_size))):
        raise RuntimeError("failed to construct a derangement")
    return permutation.to(device=device)


def _validate_permutation(permutation: torch.Tensor, batch_size: int) -> torch.Tensor:
    if not torch.is_tensor(permutation):
        raise TypeError("permutation must be a torch.Tensor")
    if permutation.ndim != 1 or permutation.shape[0] != batch_size:
        raise ValueError(f"permutation must have shape {(batch_size,)}")
    if permutation.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("permutation must have an integer dtype")
    permutation = permutation.to(dtype=torch.long, device="cpu")
    if not torch.equal(
        torch.sort(permutation).values,
        torch.arange(batch_size, dtype=torch.long),
    ):
        raise ValueError("permutation must contain each batch index exactly once")
    return permutation


def shuffle_token_group(
    tokens: torch.Tensor,
    permutation: torch.Tensor,
    *,
    group: TokenGroup,
    anchor_index: int | None = None,
) -> torch.Tensor:
    """Batch-shuffle one token group without mutating ``tokens``.

    ``tokens`` must have shape ``[B, 3 + K, D]`` with ``K >= 1``. For
    ``single_anchor``, ``anchor_index`` is zero-based within the anchor group.
    """

    _validate_float_tensor(tokens, "tokens")
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, 3 + K, D]")
    batch_size, num_tokens, feature_dim = tokens.shape
    if batch_size < 2 or num_tokens < 4 or feature_dim <= 0:
        raise ValueError("tokens require B >= 2, at least one anchor, and D > 0")
    if group not in _TOKEN_GROUPS:
        choices = ", ".join(sorted(_TOKEN_GROUPS))
        raise ValueError(f"group must be one of: {choices}")
    permutation = _validate_permutation(permutation, batch_size).to(tokens.device)

    if group == "single_anchor":
        if isinstance(anchor_index, bool) or not isinstance(anchor_index, int):
            raise TypeError("single_anchor requires an integer anchor_index")
        num_anchors = num_tokens - 3
        if not 0 <= anchor_index < num_anchors:
            raise ValueError(f"anchor_index must be in [0, {num_anchors})")
        selected = slice(3 + anchor_index, 4 + anchor_index)
    else:
        if anchor_index is not None:
            raise ValueError("anchor_index is only valid for single_anchor")
        selected = {
            "summary": slice(0, 1),
            "global": slice(1, 2),
            "robot": slice(2, 3),
            "all_anchors": slice(3, num_tokens),
        }[group]

    result = tokens.clone()
    shuffled = tokens.index_select(0, permutation)
    result[:, selected] = shuffled[:, selected]
    return result
