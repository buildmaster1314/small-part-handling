"""Causal predictor-isolation runtime for randomized visual interruptions.

This module isolates the dynamics question from source selection.  The true
dropout mask is an offline routing oracle: an interrupted anchor receives the
method-specific prior, while every other anchor receives the current visual
measurement.  No canonical target or recovery annotation enters prediction.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Final

import numpy as np
import torch
from torch import Tensor

from fcpg.lewm_posterior_gate_data import PosteriorGateExampleMetadata
from fcpg.lewm_posterior_gate_runtime import (
    PosteriorGateMethodTrajectory,
    PosteriorGateRecursiveResult,
    _frozen_evaluation,
    _predict_prior,
    _runtime_device_dtype,
    _runtime_tensors,
)
from fcpg.lewm_posterior_randomized_sequence import (
    PosteriorRandomizedSequenceExamples,
)


FACTUAL_LEWM_METHOD: Final = "factual_lewm"
HOLD_LAST_METHOD: Final = "hold_last"
LATENT_CONSTANT_VELOCITY_METHOD: Final = "latent_constant_velocity"
ZERO_ACTION_LEWM_METHOD: Final = "zero_action_lewm"
MATCHED_NO_ACTION_METHOD: Final = "matched_no_action_recurrent"
SHUFFLED_ACTION_LEWM_METHOD: Final = "shuffled_action_lewm"

PREDICTOR_ISOLATION_METHODS: Final = (
    FACTUAL_LEWM_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    ZERO_ACTION_LEWM_METHOD,
    MATCHED_NO_ACTION_METHOD,
    SHUFFLED_ACTION_LEWM_METHOD,
)
DEFAULT_PREDICTOR_ISOLATION_METHODS: Final = (
    FACTUAL_LEWM_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    ZERO_ACTION_LEWM_METHOD,
)


def validate_shuffled_action_row_mapping(
    value: Sequence[int] | np.ndarray | Tensor,
    metadata: Sequence[PosteriorGateExampleMetadata],
) -> tuple[int, ...]:
    """Validate a same-task/phase/event, cross-episode donor permutation."""

    rows = tuple(metadata)
    if not rows:
        raise ValueError("metadata must not be empty")
    if torch.is_tensor(value):
        if value.ndim != 1 or value.dtype == torch.bool or value.is_floating_point():
            raise TypeError("shuffled_action_row_mapping must be a 1D integer tensor")
        raw = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
            raise TypeError("shuffled_action_row_mapping must be a 1D integer array")
        raw = value.tolist()
    elif not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
        raw = list(value)
    else:
        raise TypeError("shuffled_action_row_mapping must be an integer sequence")

    if len(raw) != len(rows):
        raise ValueError(
            "shuffled_action_row_mapping must contain one donor per row"
        )
    donors: list[int] = []
    for position, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(
                f"shuffled_action_row_mapping[{position}] must be an integer"
            )
        donor = int(item)
        if not 0 <= donor < len(rows):
            raise IndexError("shuffled action donor is outside the row table")
        donors.append(donor)
    if len(set(donors)) != len(donors):
        raise ValueError("shuffled action donors must form a permutation")

    for target_index, donor_index in enumerate(donors):
        target = rows[target_index]
        donor = rows[donor_index]
        if target.task != donor.task:
            raise ValueError("shuffled action donors must remain within task")
        if target.episode_id == donor.episode_id:
            raise ValueError("shuffled action donors must cross physical episodes")
        if (
            int(target.phase_index) != int(donor.phase_index)
            or target.event != donor.event
        ):
            raise ValueError(
                "shuffled action donors must preserve phase and event"
            )
    return tuple(donors)


def _normalize_methods(
    methods: Sequence[str] | None,
    *,
    has_shuffle_mapping: bool,
    has_matched_no_action: bool,
) -> tuple[str, ...]:
    if methods is None:
        selected = DEFAULT_PREDICTOR_ISOLATION_METHODS
        if has_matched_no_action:
            selected += (MATCHED_NO_ACTION_METHOD,)
        if has_shuffle_mapping:
            selected += (SHUFFLED_ACTION_LEWM_METHOD,)
    else:
        if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
            raise TypeError("methods must be a sequence of method names")
        selected = tuple(methods)
    if not selected:
        raise ValueError("methods must not be empty")
    if any(not isinstance(method, str) for method in selected):
        raise TypeError("every method name must be a string")
    if len(set(selected)) != len(selected):
        raise ValueError("methods must not contain duplicates")
    unsupported = set(selected).difference(PREDICTOR_ISOLATION_METHODS)
    if unsupported:
        raise ValueError(f"unsupported predictor methods: {sorted(unsupported)}")

    uses_shuffle = SHUFFLED_ACTION_LEWM_METHOD in selected
    if uses_shuffle != has_shuffle_mapping:
        if uses_shuffle:
            raise ValueError(
                "shuffled_action_lewm requires shuffled_action_row_mapping"
            )
        raise ValueError(
            "shuffled_action_row_mapping was provided but shuffled_action_lewm "
            "was not selected"
        )
    uses_matched = MATCHED_NO_ACTION_METHOD in selected
    if uses_matched != has_matched_no_action:
        if uses_matched:
            raise ValueError(
                "matched_no_action_recurrent requires matched_no_action_lewm"
            )
        raise ValueError(
            "matched_no_action_lewm was provided but its method was not selected"
        )
    return selected


def _dropout_mask(
    examples: PosteriorRandomizedSequenceExamples,
    *,
    device: torch.device,
    sample_count: int,
    anchors: int,
) -> Tensor:
    raw = examples.evaluation.dropout_active
    if torch.is_tensor(raw):
        mask = raw.detach().to(device=device, copy=True)
    elif isinstance(raw, np.ndarray):
        mask = torch.as_tensor(np.array(raw, copy=True), device=device)
    else:
        raise TypeError("evaluation.dropout_active must be a tensor or numpy array")
    if mask.shape != (sample_count, anchors):
        raise ValueError(
            f"evaluation.dropout_active must have shape {(sample_count, anchors)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError("evaluation.dropout_active must use boolean dtype")
    return mask


def _validate_prior(prior: Tensor, history: Tensor, method: str) -> Tensor:
    expected = history[:, -1].shape
    if prior.shape != expected:
        raise ValueError(f"{method} prior must have shape {tuple(expected)}")
    if prior.dtype != history.dtype or prior.device != history.device:
        raise TypeError(f"{method} prior must share history dtype and device")
    if not bool(torch.isfinite(prior).all()):
        raise ValueError(f"{method} prior must be finite")
    return prior


def _method_prior(
    method: str,
    lewm: object,
    matched_no_action_lewm: object | None,
    history: Tensor,
    action_chunks: Tensor,
) -> Tensor:
    if method in {FACTUAL_LEWM_METHOD, SHUFFLED_ACTION_LEWM_METHOD}:
        prior = _predict_prior(lewm, history, action_chunks)
    elif method == ZERO_ACTION_LEWM_METHOD:
        prior = _predict_prior(lewm, history, torch.zeros_like(action_chunks))
    elif method == MATCHED_NO_ACTION_METHOD:
        if matched_no_action_lewm is None:
            raise AssertionError("matched no-action model was not supplied")
        prior = _predict_prior(matched_no_action_lewm, history, action_chunks)
    elif method == HOLD_LAST_METHOD:
        prior = history[:, -1]
    elif method == LATENT_CONSTANT_VELOCITY_METHOD:
        prior = history[:, -1] + (history[:, -1] - history[:, -2])
    else:
        raise AssertionError(f"unhandled predictor method {method}")
    return _validate_prior(prior, history, method)


def run_randomized_predictor_isolation(
    lewm: object,
    examples: PosteriorRandomizedSequenceExamples,
    *,
    matched_no_action_lewm: object | None = None,
    methods: Sequence[str] | None = None,
    shuffled_action_row_mapping: Sequence[int] | np.ndarray | Tensor | None = None,
    device: torch.device | str | None = None,
) -> PosteriorGateRecursiveResult:
    """Run isolated recursive predictors under the true interruption mask.

    Each predictor owns an independent recursive history.  At every row, only
    anchors marked by ``dropout_active`` receive the predictor prior; all other
    anchors use the current measurement.  Recovery rows therefore re-anchor to
    measurement immediately because their dropout mask is false.
    """

    if not isinstance(examples, PosteriorRandomizedSequenceExamples):
        raise TypeError("examples must be PosteriorRandomizedSequenceExamples")
    has_mapping = shuffled_action_row_mapping is not None
    selected = _normalize_methods(
        methods,
        has_shuffle_mapping=has_mapping,
        has_matched_no_action=matched_no_action_lewm is not None,
    )
    rows = tuple(examples.metadata)
    shuffle_mapping = (
        validate_shuffled_action_row_mapping(shuffled_action_row_mapping, rows)
        if shuffled_action_row_mapping is not None
        else None
    )

    auxiliary_modules = (
        () if matched_no_action_lewm is None else (matched_no_action_lewm,)
    )
    runtime_device, runtime_dtype = _runtime_device_dtype(
        lewm, auxiliary_modules, examples.deployment.measurement_history, device
    )
    tensors = _runtime_tensors(
        examples.deployment, device=runtime_device, dtype=runtime_dtype
    )
    dropout = _dropout_mask(
        examples,
        device=runtime_device,
        sample_count=tensors.sample_count,
        anchors=tensors.anchors,
    )

    trajectories: dict[str, PosteriorGateMethodTrajectory] = {}
    with _frozen_evaluation((lewm, *auxiliary_modules)):
        for method in selected:
            prior_table = torch.empty_like(tensors.measurement_current)
            posterior_table = torch.empty_like(tensors.measurement_current)
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

            for stream_slice in examples.stream_slices:
                history = tensors.measurement_history[
                    stream_slice.start
                ].unsqueeze(0).clone()
                for row_index in range(stream_slice.start, stream_slice.stop):
                    source_index = (
                        shuffle_mapping[row_index]
                        if method == SHUFFLED_ACTION_LEWM_METHOD
                        and shuffle_mapping is not None
                        else row_index
                    )
                    action_sources[row_index] = source_index
                    actions = tensors.action_chunks[source_index].unsqueeze(0)
                    prior = _method_prior(
                        method,
                        lewm,
                        matched_no_action_lewm,
                        history,
                        actions,
                    )
                    measurement = tensors.measurement_current[row_index].unsqueeze(0)
                    beta = dropout[row_index].view(1, tensors.anchors, 1).to(
                        dtype=runtime_dtype
                    )
                    posterior = torch.where(beta.bool(), prior, measurement)
                    if not bool(torch.isfinite(posterior).all()):
                        raise ValueError(f"{method} posterior must be finite")

                    prior_table[row_index] = prior[0]
                    posterior_table[row_index] = posterior[0]
                    beta_table[row_index] = beta[0]
                    history = torch.cat(
                        (history[:, 1:], posterior.unsqueeze(1)), dim=1
                    )

            cpu_prior = prior_table.detach().cpu().clone()
            cpu_posterior = posterior_table.detach().cpu().clone()
            trajectories[method] = PosteriorGateMethodTrajectory(
                pre_prior=cpu_prior,
                post_update=cpu_posterior,
                writeback=cpu_posterior.clone(),
                beta=beta_table.detach().cpu().clone(),
                action_source_index=action_sources.detach().cpu().clone(),
            )

    return PosteriorGateRecursiveResult(
        trajectories=trajectories,
        metadata=rows,
    )


__all__ = [
    "DEFAULT_PREDICTOR_ISOLATION_METHODS",
    "FACTUAL_LEWM_METHOD",
    "HOLD_LAST_METHOD",
    "LATENT_CONSTANT_VELOCITY_METHOD",
    "MATCHED_NO_ACTION_METHOD",
    "PREDICTOR_ISOLATION_METHODS",
    "SHUFFLED_ACTION_LEWM_METHOD",
    "ZERO_ACTION_LEWM_METHOD",
    "run_randomized_predictor_isolation",
    "validate_shuffled_action_row_mapping",
]
