"""Zero-training posterior-fusion headroom on anchor-persistent sequences.

All numeric inputs use ``[episode, time, anchor, D]``.  The module is deliberately
model-free: it scores already-produced priors and measurements against one
canonical target and aggregates event slices at the episode level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import TypeAlias

import numpy as np
import torch


Array: TypeAlias = np.ndarray | torch.Tensor

METHOD_NAMES = (
    "prior_only",
    "measurement_only",
    "hold_last",
    "fixed_alpha_fusion",
    "oracle_convex_alpha",
)
EVENT_NAMES = ("o0", "o1", "o2", "r0", "r1")
DEFAULT_PRIMARY_EVENTS = ("o1", "o2", "r1")


@dataclass(frozen=True)
class HeadroomMethod:
    """Prediction and per-``[episode,time,anchor]`` errors."""

    prediction: Array
    squared_error: Array
    sse: Array
    mse: Array
    episode_sse: Array
    episode_mse: Array


@dataclass(frozen=True)
class PosteriorHeadroom:
    """Validity-aware endpoint and fusion scores."""

    fixed_alpha: float
    measurement_valid: Array
    fixed_alpha_effective: Array
    oracle_alpha: Array
    hold_last_available: Array
    prior_only: HeadroomMethod
    measurement_only: HeadroomMethod
    hold_last: HeadroomMethod
    fixed_alpha_fusion: HeadroomMethod
    oracle_convex_alpha: HeadroomMethod

    @property
    def methods(self) -> Mapping[str, HeadroomMethod]:
        return {name: getattr(self, name) for name in METHOD_NAMES}


@dataclass(frozen=True)
class EventMasks:
    """Event-relative masks, each shaped ``[episode,time,anchor]``."""

    o0: Array
    o1: Array
    o2: Array
    r0: Array
    r1: Array

    @property
    def by_name(self) -> Mapping[str, Array]:
        return {name: getattr(self, name) for name in EVENT_NAMES}


@dataclass(frozen=True)
class EpisodeEventRecord:
    """One episode-level statistic for one event and one anchor."""

    episode_index: int
    episode_id: str
    task: str
    anchor_index: int
    event: str
    event_samples: int
    element_count: int
    sse: Mapping[str, float]
    mse: Mapping[str, float]
    is_primary: bool
    same_frame_measurement: bool

    def __post_init__(self) -> None:
        if self.event not in EVENT_NAMES:
            raise ValueError(f"unknown event {self.event!r}")
        if self.event_samples <= 0 or self.element_count <= 0:
            raise ValueError("episode event records require positive support")
        if set(self.sse) != set(METHOD_NAMES) or set(self.mse) != set(METHOD_NAMES):
            raise ValueError("episode event records require every headroom method")
        if self.is_primary and self.event == "r0":
            raise ValueError("r0 same-frame measurement cannot be primary")
        if self.same_frame_measurement != (self.event == "r0"):
            raise ValueError("same_frame_measurement must identify r0 exactly")
        values = [*self.sse.values(), *self.mse.values()]
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("episode event errors must be finite and non-negative")


@dataclass(frozen=True)
class TaskAnchorMacro:
    """Equal-episode means followed by an equal task x anchor macro mean."""

    event: str
    method: str
    task_anchor_macro_mse: float
    per_task_anchor_mse: Mapping[str, Mapping[int, float]]
    per_task_mse: Mapping[str, float]
    per_anchor_mse: Mapping[int, float]
    episodes_per_task_anchor: Mapping[str, Mapping[int, int]]
    episode_count: int
    statistical_unit: str = "episode"


def _backend(value: object, name: str) -> str:
    if torch.is_tensor(value):
        return "torch"
    if isinstance(value, np.ndarray):
        return "numpy"
    raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray")


def _all(value: Array) -> bool:
    return bool(value.all().item()) if torch.is_tensor(value) else bool(value.all())


def _finite(value: Array) -> Array:
    return torch.isfinite(value) if torch.is_tensor(value) else np.isfinite(value)


def _where(mask: Array, yes: Array, no: Array) -> Array:
    return torch.where(mask, yes, no) if torch.is_tensor(mask) else np.where(mask, yes, no)


def _validate_inputs(
    prior: Array,
    measurement: Array,
    canonical_target: Array,
    measurement_valid: Array | None,
) -> Array:
    backend = _backend(prior, "prior")
    for name, value in (
        ("measurement", measurement),
        ("canonical_target", canonical_target),
    ):
        if _backend(value, name) != backend:
            raise TypeError("prior, measurement, and canonical_target must share a backend")
        if value.shape != prior.shape:
            raise ValueError(f"{name} must match prior shape {tuple(prior.shape)}")
    if prior.ndim != 4 or any(size <= 0 for size in prior.shape):
        raise ValueError("numeric inputs must have nonempty shape [episode,time,anchor,D]")
    if torch.is_tensor(prior):
        if not prior.is_floating_point():
            raise TypeError("numeric inputs must use floating dtypes")
        if measurement.dtype != prior.dtype or canonical_target.dtype != prior.dtype:
            raise TypeError("numeric input dtypes must match")
        if measurement.device != prior.device or canonical_target.device != prior.device:
            raise ValueError("torch numeric inputs must share a device")
    else:
        if not np.issubdtype(prior.dtype, np.floating):
            raise TypeError("numeric inputs must use floating dtypes")
        if measurement.dtype != prior.dtype or canonical_target.dtype != prior.dtype:
            raise TypeError("numeric input dtypes must match")
    if not _all(_finite(prior)) or not _all(_finite(canonical_target)):
        raise ValueError("prior and canonical_target must be finite")

    if measurement_valid is None:
        if torch.is_tensor(prior):
            valid: Array = torch.ones(
                prior.shape[:-1], dtype=torch.bool, device=prior.device
            )
        else:
            valid = np.ones(prior.shape[:-1], dtype=np.bool_)
    else:
        if _backend(measurement_valid, "measurement_valid") != backend:
            raise TypeError("measurement_valid must share the numeric backend")
        if measurement_valid.shape != prior.shape[:-1]:
            raise ValueError("measurement_valid must have shape [episode,time,anchor]")
        if torch.is_tensor(measurement_valid):
            if measurement_valid.dtype != torch.bool:
                raise TypeError("measurement_valid must be boolean")
            if measurement_valid.device != prior.device:
                raise ValueError("measurement_valid must share the torch device")
        elif measurement_valid.dtype != np.bool_:
            raise TypeError("measurement_valid must be boolean")
        valid = measurement_valid
    if not _all((~valid[..., None]) | _finite(measurement)):
        raise ValueError("valid measurements must be finite")
    return valid


def _alpha(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("fixed_alpha must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("fixed_alpha must be finite and lie in [0, 1]")
    return result


def _oracle_alpha(
    prior: Array, safe_measurement: Array, target: Array, valid: Array
) -> Array:
    delta = safe_measurement - prior
    if torch.is_tensor(prior):
        denominator = delta.square().sum(dim=-1)
        numerator = ((target - prior) * delta).sum(dim=-1)
        positive = denominator > 0
        raw = numerator / torch.where(positive, denominator, torch.ones_like(denominator))
        return torch.where(valid & positive, raw.clamp(0.0, 1.0), torch.zeros_like(raw))
    denominator = np.sum(np.square(delta), axis=-1)
    numerator = np.sum((target - prior) * delta, axis=-1)
    positive = denominator > 0
    raw = numerator / np.where(positive, denominator, np.ones_like(denominator))
    return np.where(valid & positive, np.clip(raw, 0.0, 1.0), np.zeros_like(raw))


def oracle_convex_alpha(
    prior: Array,
    measurement: Array,
    canonical_target: Array,
    measurement_valid: Array | None = None,
) -> Array:
    """Return the clipped closed-form alpha for every ``[episode,time,anchor]``."""

    valid = _validate_inputs(prior, measurement, canonical_target, measurement_valid)
    safe = _where(valid[..., None], measurement, prior)
    return _oracle_alpha(prior, safe, canonical_target, valid)


def _hold_last(prior: Array, safe: Array, valid: Array) -> tuple[Array, Array]:
    if torch.is_tensor(prior):
        last = torch.zeros_like(prior[:, 0])
        has_last: Array = torch.zeros(
            prior.shape[0], prior.shape[2], dtype=torch.bool, device=prior.device
        )
    else:
        last = np.zeros_like(prior[:, 0])
        has_last = np.zeros((prior.shape[0], prior.shape[2]), dtype=np.bool_)
    predictions: list[Array] = []
    availability: list[Array] = []
    for time in range(prior.shape[1]):
        current_valid = valid[:, time]
        fallback = _where(has_last[..., None], last, prior[:, time])
        prediction = _where(current_valid[..., None], safe[:, time], fallback)
        last = _where(current_valid[..., None], safe[:, time], last)
        has_last = has_last | current_valid
        predictions.append(prediction)
        availability.append(has_last)
    if torch.is_tensor(prior):
        return torch.stack(predictions, dim=1), torch.stack(availability, dim=1)
    return np.stack(predictions, axis=1), np.stack(availability, axis=1)


def _score(prediction: Array, target: Array) -> HeadroomMethod:
    squared = (prediction - target) ** 2
    if torch.is_tensor(prediction):
        sse = squared.sum(dim=-1)
        mse = squared.mean(dim=-1)
        episode_sse = squared.sum(dim=(1, 2, 3))
        episode_mse = squared.mean(dim=(1, 2, 3))
    else:
        sse = np.sum(squared, axis=-1)
        mse = np.mean(squared, axis=-1)
        episode_sse = np.sum(squared, axis=(1, 2, 3))
        episode_mse = np.mean(squared, axis=(1, 2, 3))
    return HeadroomMethod(prediction, squared, sse, mse, episode_sse, episode_mse)


def compute_posterior_headroom(
    prior: Array,
    measurement: Array,
    canonical_target: Array,
    *,
    fixed_alpha: Real,
    measurement_valid: Array | None = None,
) -> PosteriorHeadroom:
    """Compute all zero-training endpoints and errors.

    Invalid measurements are never read numerically. Measurement-only and fixed
    fusion fall back to the prior; hold-last falls back to the most recent valid
    measurement, or to the current prior before any measurement has been seen.
    """

    alpha = _alpha(fixed_alpha)
    valid = _validate_inputs(prior, measurement, canonical_target, measurement_valid)
    safe = _where(valid[..., None], measurement, prior)
    hold_last, hold_available = _hold_last(prior, safe, valid)
    fixed = prior + alpha * (safe - prior)
    oracle_alpha = _oracle_alpha(prior, safe, canonical_target, valid)
    oracle = prior + oracle_alpha[..., None] * (safe - prior)
    if torch.is_tensor(prior):
        fixed_effective = valid.to(dtype=prior.dtype) * alpha
    else:
        fixed_effective = valid.astype(prior.dtype) * alpha
    return PosteriorHeadroom(
        fixed_alpha=alpha,
        measurement_valid=valid,
        fixed_alpha_effective=fixed_effective,
        oracle_alpha=oracle_alpha,
        hold_last_available=hold_available,
        prior_only=_score(prior, canonical_target),
        measurement_only=_score(safe, canonical_target),
        hold_last=_score(hold_last, canonical_target),
        fixed_alpha_fusion=_score(fixed, canonical_target),
        oracle_convex_alpha=_score(oracle, canonical_target),
    )


def _shift(value: Array) -> Array:
    if torch.is_tensor(value):
        return torch.cat((torch.zeros_like(value[:, :1]), value[:, :-1]), dim=1)
    return np.concatenate((np.zeros_like(value[:, :1]), value[:, :-1]), axis=1)


def event_relative_masks(masked_windows: Array) -> EventMasks:
    """Derive entry, masked writeback, and recovery positions.

    ``o0`` is the first masked frame, ``o1/o2`` the next two contiguous masked
    frames, ``r0`` the first unmasked frame, and ``r1`` its unmasked successor.
    """

    _backend(masked_windows, "masked_windows")
    if masked_windows.ndim != 3 or any(size <= 0 for size in masked_windows.shape):
        raise ValueError("masked_windows must have nonempty [episode,time,anchor] shape")
    expected_bool = (
        masked_windows.dtype == torch.bool
        if torch.is_tensor(masked_windows)
        else masked_windows.dtype == np.bool_
    )
    if not expected_bool:
        raise TypeError("masked_windows must be boolean")
    previous_masked = _shift(masked_windows)
    o0 = masked_windows & ~previous_masked
    o1 = masked_windows & _shift(o0)
    o2 = masked_windows & _shift(o1)
    r0 = ~masked_windows & previous_masked
    r1 = ~masked_windows & _shift(r0)
    return EventMasks(o0=o0, o1=o1, o2=o2, r0=r0, r1=r1)


def _labels(values: Sequence[object], size: int, name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or len(values) != size:
        raise ValueError(f"{name} must contain one value per episode")
    labels = [str(value) for value in values]
    if any(not value for value in labels):
        raise ValueError(f"{name} values must be non-empty")
    return labels


def _count(mask: Array) -> int:
    return int(mask.sum().item()) if torch.is_tensor(mask) else int(mask.sum())


def _masked_sum(value: Array, mask: Array) -> float:
    selected = value[mask]
    total = selected.sum()
    return float(total.detach().cpu().item()) if torch.is_tensor(total) else float(total)


def build_episode_event_records(
    result: PosteriorHeadroom,
    masked_windows: Array,
    *,
    task_names: Sequence[object],
    episode_ids: Sequence[object],
    primary_events: Sequence[str] = DEFAULT_PRIMARY_EVENTS,
) -> tuple[EpisodeEventRecord, ...]:
    """Aggregate event samples before emitting any statistical record."""

    if not isinstance(result, PosteriorHeadroom):
        raise TypeError("result must be PosteriorHeadroom")
    events = event_relative_masks(masked_windows)
    if _backend(masked_windows, "masked_windows") != _backend(
        result.measurement_valid, "measurement_valid"
    ) or masked_windows.shape != result.measurement_valid.shape:
        raise ValueError("masked_windows must match result measurement_valid")
    if torch.is_tensor(masked_windows) and masked_windows.device != result.measurement_valid.device:
        raise ValueError("masked_windows must share the torch device")
    if not _all(~masked_windows | ~result.measurement_valid):
        raise ValueError("masked windows must be measurement-invalid")
    primary = tuple(primary_events)
    if len(primary) != len(set(primary)) or any(name not in EVENT_NAMES for name in primary):
        raise ValueError("primary_events must be unique known event names")
    if "r0" in primary:
        raise ValueError("r0 same-frame measurement cannot be primary")

    episodes, _, anchors = result.measurement_valid.shape
    tasks = _labels(task_names, episodes, "task_names")
    ids = _labels(episode_ids, episodes, "episode_ids")
    if len(set(zip(tasks, ids, strict=True))) != episodes:
        raise ValueError("each input row must identify a unique task episode")
    dimension = result.prior_only.prediction.shape[-1]
    records: list[EpisodeEventRecord] = []
    for episode in range(episodes):
        for event, event_mask in events.by_name.items():
            for anchor in range(anchors):
                selected = event_mask[episode, :, anchor]
                sample_count = _count(selected)
                if sample_count == 0:
                    continue
                element_count = sample_count * dimension
                sums = {
                    name: _masked_sum(score.sse[episode, :, anchor], selected)
                    for name, score in result.methods.items()
                }
                records.append(
                    EpisodeEventRecord(
                        episode_index=episode,
                        episode_id=ids[episode],
                        task=tasks[episode],
                        anchor_index=anchor,
                        event=event,
                        event_samples=sample_count,
                        element_count=element_count,
                        sse=sums,
                        mse={name: value / element_count for name, value in sums.items()},
                        is_primary=event in primary,
                        same_frame_measurement=event == "r0",
                    )
                )
    return tuple(records)


def task_anchor_macro(
    records: Sequence[EpisodeEventRecord],
    *,
    event: str,
    method: str,
    require_complete: bool = True,
) -> TaskAnchorMacro:
    """Average episodes within task x anchor cells, then average cells equally."""

    if event not in EVENT_NAMES or method not in METHOD_NAMES:
        raise ValueError("unknown event or method")
    if not records or any(not isinstance(row, EpisodeEventRecord) for row in records):
        raise ValueError("records must contain EpisodeEventRecord values")
    tasks = sorted({row.task for row in records})
    anchors = sorted({row.anchor_index for row in records})
    selected = [row for row in records if row.event == event]
    if not selected:
        raise ValueError("no records match the requested event")
    keys = [(row.task, row.episode_id, row.anchor_index) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate episode event records")
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in selected:
        grouped.setdefault((row.task, row.anchor_index), []).append(row.mse[method])
    missing = [(task, anchor) for task in tasks for anchor in anchors if (task, anchor) not in grouped]
    if require_complete and missing:
        raise ValueError(f"incomplete task x anchor grid: {missing}")
    cells = {
        key: math.fsum(values) / len(values) for key, values in grouped.items()
    }
    per_task_anchor = {
        task: {anchor: cells[(task, anchor)] for anchor in anchors if (task, anchor) in cells}
        for task in tasks
    }
    per_task = {
        task: math.fsum(values.values()) / len(values)
        for task, values in per_task_anchor.items()
        if values
    }
    per_anchor = {
        anchor: math.fsum(cells[(task, anchor)] for task in tasks if (task, anchor) in cells)
        / sum((task, anchor) in cells for task in tasks)
        for anchor in anchors
        if any((task, anchor) in cells for task in tasks)
    }
    return TaskAnchorMacro(
        event=event,
        method=method,
        task_anchor_macro_mse=math.fsum(cells.values()) / len(cells),
        per_task_anchor_mse=per_task_anchor,
        per_task_mse=per_task,
        per_anchor_mse=per_anchor,
        episodes_per_task_anchor={
            task: {
                anchor: len(grouped[(task, anchor)])
                for anchor in anchors
                if (task, anchor) in grouped
            }
            for task in tasks
        },
        episode_count=len({(row.task, row.episode_id) for row in selected}),
    )


__all__ = [
    "DEFAULT_PRIMARY_EVENTS",
    "EVENT_NAMES",
    "METHOD_NAMES",
    "EpisodeEventRecord",
    "EventMasks",
    "HeadroomMethod",
    "PosteriorHeadroom",
    "TaskAnchorMacro",
    "build_episode_event_records",
    "compute_posterior_headroom",
    "event_relative_masks",
    "oracle_convex_alpha",
    "task_anchor_macro",
]
