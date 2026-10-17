"""Privileged recursive upper bounds for posterior source selection.

This module is deliberately separate from the deployment-only posterior
runtime. Canonical targets are used only to choose an oracle source and score
the resulting recursive trajectory; they are never exposed to a deployable
adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from numbers import Integral
from typing import Final

import numpy as np
import torch
from torch import Tensor

from fcpg.lewm_posterior_gate_cache import PosteriorGateCache
from fcpg.lewm_posterior_gate_data import (
    PosteriorGateDeploymentTensors,
    PosteriorGateExampleMetadata,
)
from fcpg.lewm_posterior_gate_runtime import (
    MEASUREMENT_METHOD,
    PosteriorGateMethodTrajectory,
    PosteriorGateRecursiveResult,
    PosteriorGateRuntimeResult,
    _frozen_evaluation,
    _predict_prior,
    _runtime_device_dtype,
    _runtime_tensors,
    _stream_slices,
    build_posterior_gate_row_records,
    infer_direct_anchors,
    posterior_gate_examples_from_cache,
)


PRIOR_METHOD: Final = "prior_writeback"
HARD_ORACLE_METHOD: Final = "oracle_hard"
CONVEX_ORACLE_METHOD: Final = "oracle_convex"
SCHEDULE_ORACLE_METHOD: Final = "oracle_schedule"
SEQUENCE_ORACLE_METHOD: Final = "oracle_sequence_exact"
FSM_ORACLE_METHOD: Final = "oracle_fsm_cautious_enter_fast_exit"

PREFERENCE_ORACLE_METHODS: Final = (
    MEASUREMENT_METHOD,
    PRIOR_METHOD,
    HARD_ORACLE_METHOD,
    CONVEX_ORACLE_METHOD,
    SCHEDULE_ORACLE_METHOD,
    SEQUENCE_ORACLE_METHOD,
    FSM_ORACLE_METHOD,
)
DEFAULT_PREFERENCE_ORACLE_METHODS: Final = tuple(
    method for method in PREFERENCE_ORACLE_METHODS if method != SEQUENCE_ORACLE_METHOD
)
MAX_EXACT_SEQUENCE_PATHS: Final = 1_024

DirectAnchor = int | Mapping[tuple[str, int | str], int]


def _canonical_tensor(
    value: np.ndarray | Tensor,
    *,
    sample_count: int,
    anchors: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if torch.is_tensor(value):
        result = value.detach().to(device=device, dtype=dtype, copy=True)
    elif isinstance(value, np.ndarray):
        result = torch.as_tensor(np.array(value, copy=True), device=device, dtype=dtype)
    else:
        raise TypeError("canonical_current must be a tensor or numpy array")
    expected = (sample_count, anchors, latent_dim)
    if tuple(result.shape) != expected:
        raise ValueError(f"canonical_current must have shape {expected}")
    if not result.is_floating_point() or not bool(torch.isfinite(result).all()):
        raise ValueError("canonical_current must be finite floating point")
    return result


def _valid_tensor(
    value: np.ndarray | Tensor | None,
    *,
    sample_count: int,
    anchors: int,
    device: torch.device,
) -> Tensor:
    if value is None:
        return torch.ones(sample_count, anchors, dtype=torch.bool, device=device)
    if torch.is_tensor(value):
        result = value.detach().to(device=device, copy=True)
    elif isinstance(value, np.ndarray):
        result = torch.as_tensor(np.array(value, copy=True), device=device)
    else:
        raise TypeError("canonical_valid_current must be a tensor or numpy array")
    if tuple(result.shape) != (sample_count, anchors):
        raise ValueError(
            "canonical_valid_current must have shape [sample_count, anchors]"
        )
    if result.dtype != torch.bool:
        raise TypeError("canonical_valid_current must be boolean")
    return result


def _normalize_methods(methods: Sequence[str] | None) -> tuple[str, ...]:
    if methods is None:
        return DEFAULT_PREFERENCE_ORACLE_METHODS
    if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
        raise TypeError("methods must be a sequence")
    result = tuple(methods)
    if not result:
        raise ValueError("methods must not be empty")
    if any(not isinstance(method, str) for method in result):
        raise TypeError("every method must be a string")
    if len(set(result)) != len(result):
        raise ValueError("methods must not contain duplicates")
    unexpected = set(result).difference(PREFERENCE_ORACLE_METHODS)
    if unexpected:
        raise ValueError(f"unsupported preference oracle methods: {sorted(unexpected)}")
    return result


def _direct_anchor_mask(
    metadata: Sequence[PosteriorGateExampleMetadata],
    *,
    anchors: int,
    direct_anchor: DirectAnchor | None,
    device: torch.device,
) -> Tensor:
    rows = tuple(metadata)
    if direct_anchor is None:
        return torch.ones(len(rows), anchors, 1, dtype=torch.bool, device=device)

    if isinstance(direct_anchor, bool):
        raise TypeError("direct_anchor must be an integer or episode mapping")
    if isinstance(direct_anchor, Integral):
        value = int(direct_anchor)
        if not 0 <= value < anchors:
            raise ValueError("direct_anchor is outside the anchor dimension")
        indices = [value] * len(rows)
    elif isinstance(direct_anchor, Mapping):
        episode_keys = {(row.task, row.episode_id) for row in rows}
        if set(direct_anchor) != episode_keys:
            missing = sorted(episode_keys - set(direct_anchor), key=str)
            unexpected = sorted(set(direct_anchor) - episode_keys, key=str)
            raise ValueError(
                "direct_anchor mapping must cover every episode exactly; "
                f"missing={missing[:3]} unexpected={unexpected[:3]}"
            )
        indices = []
        for row in rows:
            value = direct_anchor[(row.task, row.episode_id)]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("direct_anchor mapping values must be integers")
            index = int(value)
            if not 0 <= index < anchors:
                raise ValueError("direct_anchor mapping value is outside the anchor dimension")
            indices.append(index)
    else:
        raise TypeError("direct_anchor must be an integer or episode mapping")

    result = torch.zeros(len(rows), anchors, 1, dtype=torch.bool, device=device)
    result[torch.arange(len(rows), device=device), indices, 0] = True
    return result


def _oracle_beta(
    method: str,
    *,
    measurement: Tensor,
    prior: Tensor,
    target: Tensor,
    target_valid: Tensor,
    event: str,
) -> Tensor:
    shape = (*measurement.shape[:-1], 1)
    if method == MEASUREMENT_METHOD:
        return torch.zeros(shape, dtype=measurement.dtype, device=measurement.device)
    if method == PRIOR_METHOD:
        return torch.ones(shape, dtype=measurement.dtype, device=measurement.device)
    if method == SCHEDULE_ORACLE_METHOD:
        use_prior = event in {"o0", "o1", "o2"}
        if not use_prior:
            return torch.zeros(shape, dtype=measurement.dtype, device=measurement.device)
        return torch.ones(shape, dtype=measurement.dtype, device=measurement.device)

    delta = prior - measurement
    if method == HARD_ORACLE_METHOD:
        measurement_error = torch.square(measurement - target).mean(dim=-1)
        prior_error = torch.square(prior - target).mean(dim=-1)
        # Ties and invalid targets fail closed to the current measurement.
        choose_prior = target_valid & (prior_error < measurement_error)
        return choose_prior.unsqueeze(-1).to(dtype=measurement.dtype)
    if method == CONVEX_ORACLE_METHOD:
        denominator = torch.square(delta).sum(dim=-1)
        numerator = ((target - measurement) * delta).sum(dim=-1)
        positive = denominator > 0.0
        raw = numerator / torch.where(
            positive, denominator, torch.ones_like(denominator)
        )
        beta = torch.where(
            target_valid & positive,
            raw.clamp(0.0, 1.0),
            torch.zeros_like(raw),
        )
        return beta.unsqueeze(-1)
    raise AssertionError(f"unhandled preference oracle method: {method}")


def _exact_sequence_oracle_trajectory(
    lewm: object,
    tensors: object,
    rows: tuple[PosteriorGateExampleMetadata, ...],
    streams: Sequence[slice],
    target: Tensor,
    target_valid: Tensor,
    selectable: Tensor,
) -> PosteriorGateMethodTrajectory:
    """Enumerate a hindsight-only direct-anchor sequence bound per stream."""

    sample_count = int(tensors.sample_count)
    anchors = int(tensors.anchors)
    runtime_device = tensors.measurement_current.device
    runtime_dtype = tensors.measurement_current.dtype
    prior_table = torch.empty_like(tensors.measurement_current)
    post_table = torch.empty_like(tensors.measurement_current)
    beta_table = torch.zeros(
        sample_count, anchors, 1, dtype=runtime_dtype, device=runtime_device
    )
    action_sources = torch.arange(sample_count, dtype=torch.long, device=runtime_device)

    for stream_slice in streams:
        width = stream_slice.stop - stream_slice.start
        path_count = 2**width
        if path_count > MAX_EXACT_SEQUENCE_PATHS:
            raise ValueError(
                "exact sequence path count exceeds the bounded diagnostic limit"
            )
        choices = torch.tensor(
            tuple(product((0.0, 1.0), repeat=width)),
            dtype=runtime_dtype,
            device=runtime_device,
        )
        selected = torch.nonzero(
            selectable[stream_slice.start, :, 0], as_tuple=False
        ).flatten()
        if selected.numel() != 1:
            raise ValueError(
                "exact sequence oracle requires exactly one direct anchor per stream"
            )
        direct = int(selected.item())
        batch = int(choices.shape[0])
        history = tensors.measurement_history[stream_slice.start].unsqueeze(0)
        history = history.expand(batch, *history.shape[1:]).clone()
        cumulative = torch.zeros(batch, dtype=torch.float64, device=runtime_device)
        candidate_priors: list[Tensor] = []
        candidate_posts: list[Tensor] = []
        candidate_betas: list[Tensor] = []

        for local_index, row_index in enumerate(
            range(stream_slice.start, stream_slice.stop)
        ):
            actions = tensors.action_chunks[row_index].unsqueeze(0)
            actions = actions.expand(batch, *actions.shape[1:]).contiguous()
            measurement = tensors.measurement_current[row_index].unsqueeze(0)
            measurement = measurement.expand(batch, -1, -1)
            prior = _predict_prior(lewm, history, actions)
            beta = torch.zeros(
                batch, anchors, 1, dtype=runtime_dtype, device=runtime_device
            )
            valid = bool(target_valid[row_index, direct].item())
            if valid:
                beta[:, direct, 0] = choices[:, local_index]
            post = measurement + beta * (prior - measurement)
            if valid:
                direct_target = target[row_index, direct].unsqueeze(0)
                cumulative = cumulative + torch.square(
                    post[:, direct] - direct_target
                ).mean(dim=-1).to(torch.float64)
            candidate_priors.append(prior)
            candidate_posts.append(post)
            candidate_betas.append(beta)
            history = torch.cat((history[:, 1:], post.unsqueeze(1)), dim=1)

        # Resolve exact ties toward fewer prior writes, then fewer switches,
        # then the measurement-first lexicographic path.
        minimum = torch.min(cumulative)
        tied = torch.nonzero(cumulative == minimum, as_tuple=False).flatten().tolist()
        winner = min(
            tied,
            key=lambda index: (
                int(choices[index].sum().item()),
                int((choices[index, 1:] != choices[index, :-1]).sum().item()),
                int(index),
            ),
        )
        for local_index, row_index in enumerate(
            range(stream_slice.start, stream_slice.stop)
        ):
            prior_table[row_index] = candidate_priors[local_index][winner]
            post_table[row_index] = candidate_posts[local_index][winner]
            beta_table[row_index] = candidate_betas[local_index][winner]

    cpu_post = post_table.detach().cpu().clone()
    return PosteriorGateMethodTrajectory(
        pre_prior=prior_table.detach().cpu().clone(),
        post_update=cpu_post,
        writeback=cpu_post.clone(),
        beta=beta_table.detach().cpu().clone(),
        action_source_index=action_sources.detach().cpu().clone(),
    )


def run_recursive_posterior_preference_oracle(
    lewm: object,
    deployment: PosteriorGateDeploymentTensors,
    metadata: Sequence[PosteriorGateExampleMetadata],
    canonical_current: np.ndarray | Tensor,
    *,
    canonical_valid_current: np.ndarray | Tensor | None = None,
    methods: Sequence[str] | None = None,
    direct_anchor: DirectAnchor | None = None,
    device: torch.device | str | None = None,
) -> PosteriorGateRecursiveResult:
    """Run privileged source-selection upper bounds with recursive writeback.

    Every method owns an independent length-three history. The hard oracle
    selects either the full measurement or full prior vector per anchor. The
    convex oracle selects one scalar prior weight per anchor. No per-dimension
    oracle is provided. ``oracle_sequence_exact`` is a non-causal hindsight
    bound; all other oracle decisions use only the current canonical target.
    """

    runtime_device, runtime_dtype = _runtime_device_dtype(
        lewm, (), deployment.measurement_history, device
    )
    tensors = _runtime_tensors(
        deployment, device=runtime_device, dtype=runtime_dtype
    )
    rows = tuple(metadata)
    streams = _stream_slices(rows, tensors)
    target = _canonical_tensor(
        canonical_current,
        sample_count=tensors.sample_count,
        anchors=tensors.anchors,
        latent_dim=tensors.latent_dim,
        device=runtime_device,
        dtype=runtime_dtype,
    )
    target_valid = _valid_tensor(
        canonical_valid_current,
        sample_count=tensors.sample_count,
        anchors=tensors.anchors,
        device=runtime_device,
    )
    selected_methods = _normalize_methods(methods)
    selectable = _direct_anchor_mask(
        rows,
        anchors=tensors.anchors,
        direct_anchor=direct_anchor,
        device=runtime_device,
    )

    trajectories: dict[str, PosteriorGateMethodTrajectory] = {}
    with _frozen_evaluation((lewm,)):
        for method in selected_methods:
            if method == SEQUENCE_ORACLE_METHOD:
                trajectories[method] = _exact_sequence_oracle_trajectory(
                    lewm,
                    tensors,
                    rows,
                    streams,
                    target,
                    target_valid,
                    selectable,
                )
                continue
            prior_table = torch.empty_like(tensors.measurement_current)
            post_table = torch.empty_like(tensors.measurement_current)
            beta_table = torch.empty(
                tensors.sample_count,
                tensors.anchors,
                1,
                dtype=runtime_dtype,
                device=runtime_device,
            )
            action_sources = torch.arange(
                tensors.sample_count, dtype=torch.long, device=runtime_device
            )

            for stream_slice in streams:
                history = tensors.measurement_history[
                    stream_slice.start
                ].unsqueeze(0).clone()
                prior_mode = torch.zeros(
                    tensors.anchors, dtype=torch.bool, device=runtime_device
                )
                prior_streak = torch.zeros(
                    tensors.anchors, dtype=torch.long, device=runtime_device
                )
                for row_index in range(stream_slice.start, stream_slice.stop):
                    actions = tensors.action_chunks[row_index].unsqueeze(0)
                    measurement = tensors.measurement_current[row_index].unsqueeze(0)
                    prior = _predict_prior(lewm, history, actions)
                    if method == FSM_ORACLE_METHOD:
                        measurement_error = torch.square(
                            measurement[0] - target[row_index]
                        ).mean(dim=-1)
                        prior_error = torch.square(
                            prior[0] - target[row_index]
                        ).mean(dim=-1)
                        eligible = (
                            selectable[row_index, :, 0]
                            & target_valid[row_index]
                        )
                        prior_better = eligible & (prior_error < measurement_error)
                        leave = prior_mode & ~prior_better
                        prior_mode = prior_mode & ~leave
                        prior_streak = torch.where(
                            leave, torch.zeros_like(prior_streak), prior_streak
                        )
                        waiting = ~prior_mode
                        prior_streak = torch.where(
                            waiting & prior_better,
                            prior_streak + 1,
                            torch.where(
                                waiting, torch.zeros_like(prior_streak), prior_streak
                            ),
                        )
                        enter = waiting & prior_better & (prior_streak >= 2)
                        prior_mode = prior_mode | enter
                        beta = prior_mode.to(runtime_dtype).view(1, -1, 1)
                    else:
                        beta = _oracle_beta(
                            method,
                            measurement=measurement,
                            prior=prior,
                            target=target[row_index].unsqueeze(0),
                            target_valid=target_valid[row_index].unsqueeze(0),
                            event=rows[row_index].event,
                        )
                        beta = beta * selectable[row_index].unsqueeze(0).to(
                            dtype=runtime_dtype
                        )
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
                action_source_index=action_sources.detach().cpu().clone(),
            )

    return PosteriorGateRecursiveResult(
        trajectories=trajectories,
        metadata=rows,
    )


def evaluate_posterior_preference_oracle_cache(
    lewm: object,
    cache: PosteriorGateCache,
    *,
    methods: Sequence[str] | None = None,
    device: torch.device | str | None = None,
) -> PosteriorGateRuntimeResult:
    """Evaluate a verified cache while keeping labels out of deployment inputs."""

    if not bool(np.asarray(cache.labels["canonical_valid_current"]).all()):
        raise ValueError("oracle scoring requires valid canonical targets for every row")
    examples = posterior_gate_examples_from_cache(cache)
    direct_anchor: Mapping[tuple[str, int | str], int] = infer_direct_anchors(
        examples.metadata, examples.evaluation.dropout_active
    )
    recursive = run_recursive_posterior_preference_oracle(
        lewm,
        examples.deployment,
        examples.metadata,
        examples.labels.canonical_current,
        canonical_valid_current=cache.labels["canonical_valid_current"],
        methods=_normalize_methods(methods),
        direct_anchor=direct_anchor,
        device=device,
    )
    records = build_posterior_gate_row_records(
        recursive,
        examples.labels.canonical_current,
        direct_anchor=direct_anchor,
    )
    return PosteriorGateRuntimeResult(
        recursive=recursive,
        row_records=records,
    )


def evaluate_teacher_forced_preference_oracle_cache(
    cache: PosteriorGateCache,
    *,
    methods: Sequence[str] = (
        MEASUREMENT_METHOD,
        PRIOR_METHOD,
        HARD_ORACLE_METHOD,
        CONVEX_ORACLE_METHOD,
        SCHEDULE_ORACLE_METHOD,
    ),
) -> PosteriorGateRuntimeResult:
    """Reproduce one-step source-selection headroom from the cached factual prior."""

    if not bool(np.asarray(cache.labels["canonical_valid_current"]).all()):
        raise ValueError("oracle scoring requires valid canonical targets for every row")
    selected = _normalize_methods(methods)
    unsupported = set(selected).intersection(
        {SEQUENCE_ORACLE_METHOD, FSM_ORACLE_METHOD}
    )
    if unsupported:
        raise ValueError(
            "teacher-forced evaluation does not support recursive methods: "
            f"{sorted(unsupported)}"
        )
    examples = posterior_gate_examples_from_cache(cache)
    rows = tuple(examples.metadata)
    measurement = torch.as_tensor(
        np.array(examples.deployment.measurement_current, copy=True),
        dtype=torch.float32,
    )
    prior = torch.as_tensor(
        np.array(cache.deployment["teacher_forced_factual_prior"], copy=True),
        dtype=measurement.dtype,
    )
    target = torch.as_tensor(
        np.array(examples.labels.canonical_current, copy=True),
        dtype=measurement.dtype,
    )
    valid = torch.as_tensor(
        np.array(cache.labels["canonical_valid_current"], copy=True),
        dtype=torch.bool,
    )
    if prior.shape != measurement.shape or target.shape != measurement.shape:
        raise ValueError("teacher-forced prior, measurement, and target must align")
    if valid.shape != measurement.shape[:-1]:
        raise ValueError("canonical validity must align with teacher-forced rows")
    if not bool(torch.isfinite(prior).all()):
        raise ValueError("teacher-forced factual prior must be finite")

    direct_anchor: Mapping[tuple[str, int | str], int] = infer_direct_anchors(
        examples.metadata, examples.evaluation.dropout_active
    )
    selectable = _direct_anchor_mask(
        rows,
        anchors=measurement.shape[1],
        direct_anchor=direct_anchor,
        device=measurement.device,
    )
    trajectories: dict[str, PosteriorGateMethodTrajectory] = {}
    action_sources = torch.arange(measurement.shape[0], dtype=torch.long)
    for method in selected:
        beta_rows: list[Tensor] = []
        for row_index, row in enumerate(rows):
            beta = _oracle_beta(
                method,
                measurement=measurement[row_index].unsqueeze(0),
                prior=prior[row_index].unsqueeze(0),
                target=target[row_index].unsqueeze(0),
                target_valid=valid[row_index].unsqueeze(0),
                event=row.event,
            )[0]
            beta_rows.append(
                beta * selectable[row_index].to(dtype=measurement.dtype)
            )
        beta_table = torch.stack(beta_rows)
        post = measurement + beta_table * (prior - measurement)
        trajectories[method] = PosteriorGateMethodTrajectory(
            pre_prior=prior.clone(),
            post_update=post,
            writeback=post.clone(),
            beta=beta_table,
            action_source_index=action_sources.clone(),
        )

    recursive = PosteriorGateRecursiveResult(
        trajectories=trajectories,
        metadata=rows,
    )
    records = build_posterior_gate_row_records(
        recursive,
        target,
        direct_anchor=direct_anchor,
    )
    return PosteriorGateRuntimeResult(
        recursive=recursive,
        row_records=records,
    )


__all__ = [
    "CONVEX_ORACLE_METHOD",
    "DEFAULT_PREFERENCE_ORACLE_METHODS",
    "HARD_ORACLE_METHOD",
    "FSM_ORACLE_METHOD",
    "PREFERENCE_ORACLE_METHODS",
    "PRIOR_METHOD",
    "SCHEDULE_ORACLE_METHOD",
    "SEQUENCE_ORACLE_METHOD",
    "evaluate_posterior_preference_oracle_cache",
    "evaluate_teacher_forced_preference_oracle_cache",
    "run_recursive_posterior_preference_oracle",
]
