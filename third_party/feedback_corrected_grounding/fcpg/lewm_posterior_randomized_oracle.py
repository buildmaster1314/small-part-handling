"""Recursive privileged bounds for randomized posterior source selection.

The fixed-schedule posterior Gate uses five hard-coded events.  This module is
an isolated variable-duration evaluator: every method owns its own recursive
history, while canonical targets are used only by privileged diagnostics and
offline scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any, Final, TypeAlias

import numpy as np
import torch
from torch import Tensor

from fcpg.lewm_posterior_gate_data import PosteriorGateExampleMetadata
from fcpg.lewm_posterior_gate_runtime import (
    MEASUREMENT_METHOD,
    PosteriorGateMethodTrajectory,
    PosteriorGateRecursiveResult,
    _frozen_evaluation,
    _predict_prior,
    _runtime_device_dtype,
    _runtime_tensors,
    infer_direct_anchors,
)
from fcpg.lewm_posterior_randomized_sequence import (
    PosteriorRandomizedSequenceExamples,
)
from fcpg.lewm_posterior_time_only import TimeOnlyPolicy, predict_beta


PRIOR_METHOD: Final = "prior_writeback"
HARD_ORACLE_METHOD: Final = "oracle_hard"
CONVEX_ORACLE_METHOD: Final = "oracle_convex"
SCHEDULE_SHORTCUT_METHOD: Final = "privileged_schedule_shortcut"
FSM_ORACLE_METHOD: Final = "oracle_fsm_cautious_enter_fast_exit"
TIME_ONLY_METHOD: Final = "time_only_locked"

RANDOMIZED_ORACLE_METHODS: Final = (
    MEASUREMENT_METHOD,
    PRIOR_METHOD,
    HARD_ORACLE_METHOD,
    CONVEX_ORACLE_METHOD,
    SCHEDULE_SHORTCUT_METHOD,
    FSM_ORACLE_METHOD,
    TIME_ONLY_METHOD,
)

EpisodeKey: TypeAlias = tuple[str, int | str]
DirectAnchors: TypeAlias = Mapping[EpisodeKey, int]


def _episode_id(value: object) -> int | str:
    if isinstance(value, bool):
        raise TypeError("episode_id must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise TypeError("episode_id must be an integer or non-empty string")


def _methods(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return RANDOMIZED_ORACLE_METHODS
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("methods must be a sequence")
    methods = tuple(value)
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be non-empty and unique")
    unexpected = set(methods).difference(RANDOMIZED_ORACLE_METHODS)
    if unexpected:
        raise ValueError(f"unsupported randomized oracle methods: {sorted(unexpected)}")
    return methods


def _canonical(
    value: np.ndarray | Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if torch.is_tensor(value):
        target = value.detach().to(device=device, dtype=dtype, copy=True)
    elif isinstance(value, np.ndarray):
        target = torch.as_tensor(np.array(value, copy=True), device=device, dtype=dtype)
    else:
        raise TypeError("canonical_current must be a tensor or numpy array")
    if tuple(target.shape) != shape:
        raise ValueError(f"canonical_current must have shape {shape}")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("canonical_current must be finite")
    return target


def _direct_anchors(
    examples: PosteriorRandomizedSequenceExamples,
) -> dict[EpisodeKey, int]:
    inferred = infer_direct_anchors(
        examples.metadata, examples.evaluation.dropout_active
    )
    result: dict[EpisodeKey, int] = {}
    anchor_count = int(examples.deployment.measurement_current.shape[1])
    for raw_key, raw_value in inferred.items():
        key = (raw_key[0], _episode_id(raw_key[1]))
        if isinstance(raw_value, bool) or not isinstance(raw_value, Integral):
            raise TypeError("direct anchor must be an integer")
        anchor = int(raw_value)
        if not 0 <= anchor < anchor_count:
            raise ValueError("direct anchor is outside the anchor dimension")
        result[key] = anchor
    return result


def normalized_episode_time(
    metadata: Sequence[PosteriorGateExampleMetadata],
    episode_frame_bounds: Mapping[EpisodeKey, tuple[int, int]],
) -> np.ndarray:
    """Normalize target frames by full raw-episode bounds without event timing."""

    rows = tuple(metadata)
    if not rows:
        raise ValueError("metadata must not be empty")
    values = np.empty(len(rows), dtype=np.float64)
    seen: set[EpisodeKey] = set()
    for index, row in enumerate(rows):
        key = (row.task, _episode_id(row.episode_id))
        seen.add(key)
        if key not in episode_frame_bounds:
            raise KeyError(f"missing raw frame bounds for episode {key!r}")
        start, stop = episode_frame_bounds[key]
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, Integral)
            or not isinstance(stop, Integral)
        ):
            raise TypeError("episode frame bounds must contain integer pairs")
        first, last = int(start), int(stop)
        if last <= first:
            raise ValueError("episode frame bounds must have positive span")
        target = int(row.target_frame)
        if not first <= target <= last:
            raise ValueError("target frame lies outside its raw episode bounds")
        values[index] = (target - first) / (last - first)
    if set(episode_frame_bounds) != seen:
        raise ValueError("episode_frame_bounds must cover exactly the evaluated episodes")
    return values


def fit_time_only_from_teacher_forced(
    lewm: object,
    examples: PosteriorRandomizedSequenceExamples,
    normalized_time: np.ndarray,
    *,
    resampling_units: Mapping[EpisodeKey, int | str] | None = None,
    n_time_bins: int = 10,
    min_count: int = 3,
    dead_zone: float = 0.0,
    device: torch.device | str | None = None,
) -> TimeOnlyPolicy:
    """Fit the privileged event-aligned schedule shortcut.

    Canonical targets determine calibration labels only.  The returned policy
    consumes normalized episode time, phase, and anchor identity at evaluation
    time. Paired camera views can share a ``resampling_units`` identity so they
    count as one independent calibration trajectory within each table cell.
    """

    from fcpg.lewm_posterior_time_only import fit_time_only_policy

    runtime_device, runtime_dtype = _runtime_device_dtype(
        lewm, (), examples.deployment.measurement_history, device
    )
    tensors = _runtime_tensors(
        examples.deployment, device=runtime_device, dtype=runtime_dtype
    )
    target = _canonical(
        examples.labels.canonical_current,
        shape=tuple(tensors.measurement_current.shape),
        device=runtime_device,
        dtype=runtime_dtype,
    )
    times = np.asarray(normalized_time, dtype=np.float64)
    if times.shape != (tensors.sample_count,) or not np.isfinite(times).all():
        raise ValueError("normalized_time must be finite [N]")
    direct = _direct_anchors(examples)
    with _frozen_evaluation((lewm,)):
        prior = _predict_prior(
            lewm, tensors.measurement_history, tensors.action_chunks
        )
    measurement_error: list[float] = []
    prior_error: list[float] = []
    anchors: list[int] = []
    phases: list[int] = []
    episode_keys: list[EpisodeKey] = []
    for index, row in enumerate(examples.metadata):
        anchor = direct[(row.task, _episode_id(row.episode_id))]
        anchors.append(anchor)
        phases.append(int(row.frame_phase))
        episode_keys.append((row.task, _episode_id(row.episode_id)))
        measurement_error.append(
            float(torch.square(tensors.measurement_current[index, anchor] - target[index, anchor]).mean())
        )
        prior_error.append(
            float(torch.square(prior[index, anchor] - target[index, anchor]).mean())
        )
    phase_array = np.asarray(phases, dtype=np.int64)
    anchor_array = np.asarray(anchors, dtype=np.int64)
    evaluated = set(episode_keys)
    if resampling_units is None:
        unit_by_episode: dict[EpisodeKey, int | str] = {
            key: key[1] for key in evaluated
        }
    else:
        unit_by_episode = {
            (str(task), _episode_id(episode)): _episode_id(unit)
            for (task, episode), unit in resampling_units.items()
        }
        if set(unit_by_episode) != evaluated:
            raise ValueError(
                "resampling_units must cover exactly the calibration view episodes"
            )
    parent_ids: dict[tuple[str, int | str], int] = {}
    view_ids: dict[EpisodeKey, int] = {}
    sample_groups: list[str] = []
    sample_subgroups: list[str] = []
    for key in episode_keys:
        parent = (key[0], unit_by_episode[key])
        if parent not in parent_ids:
            parent_ids[parent] = len(parent_ids)
        if key not in view_ids:
            view_ids[key] = len(view_ids)
        sample_groups.append(str(parent_ids[parent]))
        sample_subgroups.append(str(view_ids[key]))
    return fit_time_only_policy(
        times,
        phase_array,
        anchor_array,
        np.asarray(measurement_error, dtype=np.float64),
        np.asarray(prior_error, dtype=np.float64),
        sample_group=np.asarray(sample_groups),
        sample_subgroup=np.asarray(sample_subgroups),
        n_anchors=tensors.anchors,
        n_time_bins=n_time_bins,
        min_count=min_count,
        dead_zone=dead_zone,
    )


def run_randomized_recursive_oracle(
    lewm: object,
    examples: PosteriorRandomizedSequenceExamples,
    *,
    methods: Sequence[str] | None = None,
    time_only_policy: TimeOnlyPolicy | None = None,
    normalized_time: np.ndarray | None = None,
    device: torch.device | str | None = None,
) -> PosteriorGateRecursiveResult:
    """Evaluate variable streams with independent recursive method histories."""

    if not isinstance(examples, PosteriorRandomizedSequenceExamples):
        raise TypeError("examples must be PosteriorRandomizedSequenceExamples")
    selected = _methods(methods)
    if TIME_ONLY_METHOD in selected:
        if time_only_policy is None or normalized_time is None:
            raise ValueError("time-only evaluation requires a fitted policy and times")
    runtime_device, runtime_dtype = _runtime_device_dtype(
        lewm, (), examples.deployment.measurement_history, device
    )
    tensors = _runtime_tensors(
        examples.deployment, device=runtime_device, dtype=runtime_dtype
    )
    rows = tuple(examples.metadata)
    target = _canonical(
        examples.labels.canonical_current,
        shape=tuple(tensors.measurement_current.shape),
        device=runtime_device,
        dtype=runtime_dtype,
    )
    direct = _direct_anchors(examples)
    raw_dropout = examples.evaluation.dropout_active
    if torch.is_tensor(raw_dropout):
        dropout_active = raw_dropout.detach().cpu().numpy()
    else:
        dropout_active = np.asarray(raw_dropout)
    if dropout_active.shape != (
        tensors.sample_count,
        tensors.anchors,
    ) or dropout_active.dtype != np.bool_:
        raise ValueError("evaluation.dropout_active must be boolean [N,K]")
    row_anchors = np.asarray(
        [direct[(row.task, _episode_id(row.episode_id))] for row in rows],
        dtype=np.int64,
    )
    time_beta: np.ndarray | None = None
    if TIME_ONLY_METHOD in selected:
        times = np.asarray(normalized_time, dtype=np.float64)
        if times.shape != (len(rows),):
            raise ValueError("normalized_time must have one value per row")
        time_beta = predict_beta(
            time_only_policy,
            times,
            np.asarray([row.frame_phase for row in rows], dtype=np.int64),
            row_anchors,
        )

    trajectories: dict[str, PosteriorGateMethodTrajectory] = {}
    with _frozen_evaluation((lewm,)):
        for method in selected:
            prior_table = torch.empty_like(tensors.measurement_current)
            post_table = torch.empty_like(tensors.measurement_current)
            beta_table = torch.zeros(
                tensors.sample_count,
                tensors.anchors,
                1,
                dtype=runtime_dtype,
                device=runtime_device,
            )
            for stream_slice in examples.stream_slices:
                history = tensors.measurement_history[stream_slice.start].unsqueeze(0).clone()
                prior_mode = False
                prior_streak = 0
                for row_index in range(stream_slice.start, stream_slice.stop):
                    row = rows[row_index]
                    anchor = int(row_anchors[row_index])
                    measurement = tensors.measurement_current[row_index].unsqueeze(0)
                    prior = _predict_prior(
                        lewm,
                        history,
                        tensors.action_chunks[row_index].unsqueeze(0),
                    )
                    beta_value = 0.0
                    if method == PRIOR_METHOD:
                        beta_value = 1.0
                    elif method == SCHEDULE_SHORTCUT_METHOD:
                        beta_value = float(dropout_active[row_index, anchor])
                    elif method == TIME_ONLY_METHOD:
                        assert time_beta is not None
                        beta_value = float(time_beta[row_index])
                    elif method in {
                        HARD_ORACLE_METHOD,
                        CONVEX_ORACLE_METHOD,
                        FSM_ORACLE_METHOD,
                    }:
                        measurement_error = torch.square(
                            measurement[0, anchor] - target[row_index, anchor]
                        ).mean()
                        prior_error = torch.square(
                            prior[0, anchor] - target[row_index, anchor]
                        ).mean()
                        if method == HARD_ORACLE_METHOD:
                            beta_value = float(prior_error < measurement_error)
                        elif method == CONVEX_ORACLE_METHOD:
                            delta = prior[0, anchor] - measurement[0, anchor]
                            denominator = torch.square(delta).sum()
                            if float(denominator) > 0.0:
                                numerator = torch.dot(
                                    target[row_index, anchor] - measurement[0, anchor],
                                    delta,
                                )
                                beta_value = float((numerator / denominator).clamp(0.0, 1.0))
                        else:
                            prior_better = bool(prior_error < measurement_error)
                            if prior_mode and not prior_better:
                                prior_mode = False
                                prior_streak = 0
                            elif not prior_mode:
                                prior_streak = prior_streak + 1 if prior_better else 0
                                if prior_streak >= 2:
                                    prior_mode = True
                            beta_value = float(prior_mode)
                    elif method != MEASUREMENT_METHOD:
                        raise AssertionError(f"unhandled method {method}")

                    beta = torch.zeros(
                        1,
                        tensors.anchors,
                        1,
                        dtype=runtime_dtype,
                        device=runtime_device,
                    )
                    beta[0, anchor, 0] = beta_value
                    post = measurement + beta * (prior - measurement)
                    prior_table[row_index] = prior[0]
                    post_table[row_index] = post[0]
                    beta_table[row_index] = beta[0]
                    history = torch.cat((history[:, 1:], post.unsqueeze(1)), dim=1)

            cpu_prior = prior_table.detach().cpu().clone()
            cpu_post = post_table.detach().cpu().clone()
            trajectories[method] = PosteriorGateMethodTrajectory(
                pre_prior=cpu_prior,
                post_update=cpu_post,
                writeback=cpu_post.clone(),
                beta=beta_table.detach().cpu().clone(),
                action_source_index=torch.arange(tensors.sample_count, dtype=torch.long),
            )
    return PosteriorGateRecursiveResult(trajectories=trajectories, metadata=rows)


def build_randomized_oracle_records(
    result: PosteriorGateRecursiveResult,
    canonical_current: np.ndarray | Tensor,
    direct_anchors: DirectAnchors,
    *,
    clean: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Score direct-anchor rows without assuming a fixed event vocabulary."""

    reference = next(iter(result.trajectories.values()), None)
    if reference is None:
        raise ValueError("result must contain at least one method")
    target = torch.as_tensor(canonical_current, dtype=reference.post_update.dtype)
    if target.shape != reference.post_update.shape or not bool(torch.isfinite(target).all()):
        raise ValueError("canonical_current must be finite and match trajectories")
    records: list[dict[str, Any]] = []
    for method, trajectory in result.trajectories.items():
        values = trajectory.post_update
        if values.shape != target.shape or not bool(torch.isfinite(values).all()):
            raise ValueError(f"invalid trajectory for method {method}")
        mse = torch.square(values - target).mean(dim=-1)
        for index, row in enumerate(result.metadata):
            key = (row.task, _episode_id(row.episode_id))
            if key not in direct_anchors:
                raise KeyError(f"missing direct anchor for episode {key!r}")
            anchor = int(direct_anchors[key])
            if clean:
                window = "clean"
            elif row.event.startswith("o"):
                window = "active"
            elif row.event.startswith("r"):
                window = "recovery"
            else:
                raise ValueError(f"unsupported randomized event {row.event!r}")
            try:
                event_index = (
                    int(row.event_index) if clean else int(row.event[1:])
                )
            except ValueError as exc:
                raise ValueError(f"invalid randomized event {row.event!r}") from exc
            records.append(
                {
                    "task": row.task,
                    "physical_episode_id": _episode_id(row.episode_id),
                    "phase_index": int(row.phase_index),
                    "window": window,
                    "event_index": event_index,
                    "method": method,
                    "mse": float(mse[index, anchor]),
                }
            )
    return tuple(records)


__all__ = [
    "CONVEX_ORACLE_METHOD",
    "FSM_ORACLE_METHOD",
    "HARD_ORACLE_METHOD",
    "PRIOR_METHOD",
    "RANDOMIZED_ORACLE_METHODS",
    "SCHEDULE_SHORTCUT_METHOD",
    "TIME_ONLY_METHOD",
    "build_randomized_oracle_records",
    "fit_time_only_from_teacher_forced",
    "normalized_episode_time",
    "run_randomized_recursive_oracle",
]
