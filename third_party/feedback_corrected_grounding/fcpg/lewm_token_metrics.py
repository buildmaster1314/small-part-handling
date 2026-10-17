"""Token-wise metrics for frozen oracle-anchor LeWM targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch


@dataclass(frozen=True)
class OracleTokenLayout:
    """Validated layout of one global token followed by anchor tokens."""

    num_anchors: int
    token_dim: int

    @property
    def token_count(self) -> int:
        return 1 + self.num_anchors

    @property
    def target_dim(self) -> int:
        return self.token_count * self.token_dim


@dataclass(frozen=True)
class OracleTokenBlocks:
    """Views of a flattened oracle target split by semantic token type."""

    global_token: torch.Tensor
    anchor_tokens: torch.Tensor


@dataclass(frozen=True)
class OracleTokenErrorStats:
    """Mergeable squared-error totals before conversion to MSE."""

    layout: OracleTokenLayout
    prefix_count: int
    token_squared_error: tuple[float, ...]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"oracle encoder spec {name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"oracle encoder spec {name} must be positive")
    return result


def oracle_token_layout(spec: Mapping[str, Any]) -> OracleTokenLayout:
    """Validate an oracle encoder spec and return its token layout."""

    if not isinstance(spec, Mapping):
        raise TypeError("oracle encoder spec must be a mapping")
    if "num_keypoints" not in spec or "target_dim" not in spec:
        raise KeyError("oracle encoder spec requires num_keypoints and target_dim")

    num_anchors = _positive_integer(spec["num_keypoints"], "num_keypoints")
    target_dim = _positive_integer(spec["target_dim"], "target_dim")
    token_count = 1 + num_anchors
    if target_dim % token_count:
        raise ValueError(
            f"oracle target_dim={target_dim} must be divisible by "
            f"token_count={token_count}"
        )
    token_dim = target_dim // token_count

    declared_token_dim = spec.get("token_dim")
    if declared_token_dim is not None:
        declared_token_dim = _positive_integer(declared_token_dim, "token_dim")
        if declared_token_dim != token_dim:
            raise ValueError(
                f"oracle encoder spec token_dim={declared_token_dim} disagrees "
                f"with target_dim/token_count={token_dim}"
            )
    token_order = spec.get("token_order")
    if token_order is not None and token_order != "global_then_keypoints":
        raise ValueError(
            "oracle encoder spec token_order must be 'global_then_keypoints'"
        )
    return OracleTokenLayout(num_anchors=num_anchors, token_dim=token_dim)


def _validate_tensor(
    value: torch.Tensor,
    layout: OracleTokenLayout,
    *,
    name: str,
) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if value.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if value.shape[-1] != layout.target_dim:
        raise ValueError(
            f"{name} last dimension must be {layout.target_dim}, "
            f"got {value.shape[-1]}"
        )
    if value.numel() == 0:
        raise ValueError(f"{name} must contain at least one target vector")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def split_oracle_tokens(
    value: torch.Tensor,
    spec: Mapping[str, Any],
    *,
    name: str = "value",
) -> OracleTokenBlocks:
    """Split ``[..., D]`` into one global token and ``K`` anchor tokens."""

    layout = oracle_token_layout(spec)
    _validate_tensor(value, layout, name=name)
    tokens = value.reshape(*value.shape[:-1], layout.token_count, layout.token_dim)
    return OracleTokenBlocks(
        global_token=tokens[..., 0, :],
        anchor_tokens=tokens[..., 1:, :],
    )


def oracle_token_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spec: Mapping[str, Any],
) -> OracleTokenErrorStats:
    """Compute mergeable per-token SSE totals for tensors shaped ``[..., D]``."""

    layout = oracle_token_layout(spec)
    _validate_tensor(prediction, layout, name="prediction")
    _validate_tensor(target, layout, name="target")
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes must match exactly: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")

    prefix_count = prediction.numel() // layout.target_dim
    difference = prediction.detach().to(torch.float64) - target.detach().to(torch.float64)
    token_error = difference.reshape(
        prefix_count, layout.token_count, layout.token_dim
    ).square().sum(dim=(0, 2))
    return OracleTokenErrorStats(
        layout=layout,
        prefix_count=prefix_count,
        token_squared_error=tuple(float(value) for value in token_error.cpu().tolist()),
    )


def _validate_stats(stats: OracleTokenErrorStats) -> None:
    if not isinstance(stats, OracleTokenErrorStats):
        raise TypeError("token error statistics must be OracleTokenErrorStats")
    if stats.layout.num_anchors <= 0:
        raise ValueError("token error statistics num_anchors must be positive")
    if stats.layout.token_dim <= 0:
        raise ValueError("token error statistics token_dim must be positive")
    if stats.prefix_count <= 0:
        raise ValueError("token error statistics prefix_count must be positive")
    if len(stats.token_squared_error) != stats.layout.token_count:
        raise ValueError("token error statistics have the wrong token count")
    values = torch.tensor(stats.token_squared_error, dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
        raise ValueError("token squared-error totals must be finite and non-negative")


def merge_oracle_token_errors(
    values: Iterable[OracleTokenErrorStats],
) -> OracleTokenErrorStats:
    """Merge batch statistics using element-weighted SSE accumulation."""

    items = tuple(values)
    if not items:
        raise ValueError("at least one token error statistic is required")
    for item in items:
        _validate_stats(item)
    layout = items[0].layout
    if any(item.layout != layout for item in items[1:]):
        raise ValueError("cannot merge token statistics with different layouts")
    return OracleTokenErrorStats(
        layout=layout,
        prefix_count=sum(item.prefix_count for item in items),
        token_squared_error=tuple(
            sum(item.token_squared_error[index] for item in items)
            for index in range(layout.token_count)
        ),
    )


def summarize_oracle_token_mse(
    stats: OracleTokenErrorStats,
) -> dict[str, object]:
    """Return global, per-anchor, and anchor-macro MSE values."""

    _validate_stats(stats)
    elements_per_token = stats.prefix_count * stats.layout.token_dim
    token_mse = [
        squared_error / elements_per_token
        for squared_error in stats.token_squared_error
    ]
    anchor_mse = token_mse[1:]
    return {
        "global_mse": token_mse[0],
        "anchor_mse": anchor_mse,
        "anchor_macro_mse": sum(anchor_mse) / len(anchor_mse),
        "overall_mse": sum(stats.token_squared_error)
        / (elements_per_token * stats.layout.token_count),
        "num_anchors": stats.layout.num_anchors,
        "token_dim": stats.layout.token_dim,
        "prefix_count": stats.prefix_count,
        "elements_per_token": elements_per_token,
    }


__all__ = [
    "OracleTokenBlocks",
    "OracleTokenErrorStats",
    "OracleTokenLayout",
    "merge_oracle_token_errors",
    "oracle_token_layout",
    "oracle_token_squared_error",
    "split_oracle_tokens",
    "summarize_oracle_token_mse",
]
