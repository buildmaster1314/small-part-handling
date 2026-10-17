"""Causal recursive evaluation for the frozen LeWM posterior gate.

The recursive kernel accepts only the deployment partition and row identity.
Canonical targets and direct-anchor annotations enter a separate scoring
function after every model forward has completed.  Each raw-frame phase is an
independent history stream, and every posterior update writes all anchors back
to that method's own length-three history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
import torch
from torch import Tensor, nn

from fcpg.lewm_posterior_adapter import (
    LeWMPosteriorAdapter,
    LeWMPosteriorAdapterOutput,
)
from fcpg.lewm_posterior_features import build_lewm_posterior_features
from fcpg.lewm_posterior_gate_cache import PosteriorGateCache
from fcpg.lewm_posterior_gate_data import (
    FRESH_EVENT_NAMES,
    FRESH_FRAME_SKIP,
    FRESH_HISTORY_SIZE,
    FRESH_PHASE_COUNT,
    PosteriorGateDeploymentTensors,
    PosteriorGateExampleMetadata,
    PosteriorGateExamples,
    PosteriorGateEvaluationMasks,
    PosteriorGateLabels,
)
from fcpg.lewm_posterior_gate_statistics import (
    DEFAULT_ADAPTER_METHOD,
    DEFAULT_FIXED_METHOD,
    DEFAULT_MEASUREMENT_METHOD,
    ESTIMANDS,
)


Identity: TypeAlias = int | str
EpisodeKey: TypeAlias = tuple[str, Identity]

ADAPTER_METHOD = DEFAULT_ADAPTER_METHOD
MEASUREMENT_METHOD = DEFAULT_MEASUREMENT_METHOD
FIXED_BETA_METHOD = DEFAULT_FIXED_METHOD
ACTION_BLIND_METHOD = "action_blind"
NO_LEWM_METHOD = "no_lewm"
NO_WRITEBACK_METHOD = "no_writeback"
ACTION_SHUFFLE_METHOD = "action_shuffle"

POSTERIOR_GATE_RUNTIME_METHODS = (
    MEASUREMENT_METHOD,
    FIXED_BETA_METHOD,
    ADAPTER_METHOD,
    NO_LEWM_METHOD,
    ACTION_BLIND_METHOD,
    NO_WRITEBACK_METHOD,
    ACTION_SHUFFLE_METHOD,
)
DEFAULT_POSTERIOR_GATE_RUNTIME_METHODS = POSTERIOR_GATE_RUNTIME_METHODS[:-1]
POSTERIOR_GATE_RUNTIME_MODEL_INPUTS = (
    "history_embedding",
    "causal_action_chunks",
    "z_measurement",
    "z_prior",
    "innovation",
    "confidence",
    "mask_history",
    "action_summary",
)
POSTERIOR_GATE_RUNTIME_FORBIDDEN_MODEL_INPUTS = (
    "q_true",
    "canonical",
    "schedule",
    "measurement_valid",
)


@dataclass(frozen=True)
class PosteriorGateMethodTrajectory:
    """Per-event tensors in the same flat order as the deployment rows."""

    pre_prior: Tensor
    post_update: Tensor
    writeback: Tensor
    beta: Tensor
    action_source_index: Tensor

    @property
    def prior(self) -> Tensor:
        return self.pre_prior

    @property
    def posterior(self) -> Tensor:
        return self.post_update


@dataclass(frozen=True)
class PosteriorGateRecursiveResult:
    """Model-only result; it deliberately contains no canonical target."""

    trajectories: Mapping[str, PosteriorGateMethodTrajectory]
    metadata: tuple[PosteriorGateExampleMetadata, ...]

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.trajectories)


@dataclass(frozen=True)
class PosteriorGateRuntimeResult:
    """Recursive trajectories plus phase-reduced statistics row records."""

    recursive: PosteriorGateRecursiveResult
    row_records: tuple[dict[str, Any], ...]

    @property
    def trajectories(self) -> Mapping[str, PosteriorGateMethodTrajectory]:
        return self.recursive.trajectories

    @property
    def metadata(self) -> tuple[PosteriorGateExampleMetadata, ...]:
        return self.recursive.metadata

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return self.row_records


@dataclass(frozen=True)
class _RuntimeTensors:
    measurement_history: Tensor
    action_chunks: Tensor
    measurement_current: Tensor
    q_nominal_history: Tensor
    q_corrected_history: Tensor
    valid_nominal_history: Tensor
    q_nominal_current: Tensor
    q_corrected_current: Tensor
    valid_nominal_current: Tensor

    @property
    def sample_count(self) -> int:
        return int(self.measurement_current.shape[0])

    @property
    def anchors(self) -> int:
        return int(self.measurement_current.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.measurement_current.shape[2])


def _identity(value: object, *, label: str) -> Identity:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise TypeError(f"{label} must be an integer or non-empty string")


def _identity_order(value: Identity) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _image_dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 2:
        raise ValueError(f"{name} must be at least 2")
    return result


def _fixed_beta(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("fixed_beta must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("fixed_beta must be finite and lie in [0, 1]")
    return result


def _first_floating_parameter_or_buffer(module: object) -> Tensor | None:
    if not isinstance(module, nn.Module):
        return None
    for value in (*module.parameters(), *module.buffers()):
        if value.is_floating_point():
            return value
    return None


def _runtime_device_dtype(
    lewm: object,
    adapters: Sequence[LeWMPosteriorAdapter],
    measurement_history: object,
    requested_device: torch.device | str | None,
) -> tuple[torch.device, torch.dtype]:
    references = [
        value
        for value in (
            _first_floating_parameter_or_buffer(lewm),
            *(
                _first_floating_parameter_or_buffer(adapter)
                for adapter in adapters
            ),
        )
        if value is not None
    ]
    if references:
        dtype = references[0].dtype
        if any(value.dtype != dtype for value in references[1:]):
            raise TypeError("LeWM and posterior adapters must share one floating dtype")
        module_device = references[0].device
        if any(value.device != module_device for value in references[1:]):
            raise ValueError("LeWM and posterior adapters must share one device")
    elif torch.is_tensor(measurement_history):
        if not measurement_history.is_floating_point():
            raise TypeError("measurement_history must use a floating dtype")
        dtype = measurement_history.dtype
        module_device = measurement_history.device
    else:
        array = np.asarray(measurement_history)
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError("measurement_history must use a floating dtype")
        dtype = torch.float64 if array.dtype == np.float64 else torch.float32
        module_device = torch.device("cpu")

    device = (
        module_device
        if requested_device is None
        else torch.device(requested_device)
    )
    if references and device != module_device:
        raise ValueError(
            "requested device differs from the frozen LeWM/adapter device"
        )
    return device, dtype


def _array_tensor(
    value: object,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=dtype, copy=True)
    elif isinstance(value, np.ndarray):
        tensor = torch.as_tensor(np.array(value, copy=True), device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
    else:
        raise TypeError(f"deployment.{name} must be a tensor or numpy array")
    return tensor


def _runtime_tensors(
    deployment: PosteriorGateDeploymentTensors,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _RuntimeTensors:
    if not isinstance(deployment, PosteriorGateDeploymentTensors):
        raise TypeError("deployment must be PosteriorGateDeploymentTensors")
    floating_names = (
        "measurement_history",
        "action_chunks",
        "measurement_current",
        "q_nominal_history",
        "q_corrected_history",
        "q_nominal_current",
        "q_corrected_current",
    )
    values = {
        name: _array_tensor(
            getattr(deployment, name), name, device=device, dtype=dtype
        )
        for name in floating_names
    }
    values["valid_nominal_history"] = _array_tensor(
        deployment.valid_nominal_history,
        "valid_nominal_history",
        device=device,
        dtype=None,
    )
    values["valid_nominal_current"] = _array_tensor(
        deployment.valid_nominal_current,
        "valid_nominal_current",
        device=device,
        dtype=None,
    )
    tensors = _RuntimeTensors(**values)  # type: ignore[arg-type]

    current = tensors.measurement_current
    if current.ndim != 3 or any(size <= 0 for size in current.shape):
        raise ValueError("measurement_current must have nonempty shape [N,K,D]")
    count, anchors, latent_dim = current.shape
    expected_history = (count, FRESH_HISTORY_SIZE, anchors, latent_dim)
    if tensors.measurement_history.shape != expected_history:
        raise ValueError(
            f"measurement_history must have shape {expected_history}"
        )
    if tensors.action_chunks.ndim != 3 or tensors.action_chunks.shape[:2] != (
        count,
        FRESH_HISTORY_SIZE,
    ) or tensors.action_chunks.shape[-1] <= 0:
        raise ValueError("action_chunks must have shape [N,3,A] with A > 0")
    expected_q_history = (count, FRESH_HISTORY_SIZE, anchors, 2)
    for name in ("q_nominal_history", "q_corrected_history"):
        if getattr(tensors, name).shape != expected_q_history:
            raise ValueError(f"{name} must have shape {expected_q_history}")
    expected_q = (count, anchors, 2)
    for name in ("q_nominal_current", "q_corrected_current"):
        if getattr(tensors, name).shape != expected_q:
            raise ValueError(f"{name} must have shape {expected_q}")
    if tensors.valid_nominal_history.shape != (
        count,
        FRESH_HISTORY_SIZE,
        anchors,
    ):
        raise ValueError("valid_nominal_history must have shape [N,3,K]")
    if tensors.valid_nominal_current.shape != (count, anchors):
        raise ValueError("valid_nominal_current must have shape [N,K]")
    if tensors.valid_nominal_history.dtype != torch.bool:
        raise TypeError("valid_nominal_history must have boolean dtype")
    if tensors.valid_nominal_current.dtype != torch.bool:
        raise TypeError("valid_nominal_current must have boolean dtype")
    for name in floating_names:
        tensor = getattr(tensors, name)
        if not tensor.is_floating_point() or tensor.dtype != dtype:
            raise TypeError(f"{name} must use the runtime floating dtype")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
    return tensors


def _stream_slices(
    metadata: Sequence[PosteriorGateExampleMetadata],
    tensors: _RuntimeTensors,
) -> tuple[slice, ...]:
    if isinstance(metadata, (str, bytes)) or not isinstance(metadata, Sequence):
        raise TypeError("metadata must be a sequence")
    rows = tuple(metadata)
    if len(rows) != tensors.sample_count:
        raise ValueError("metadata length must match deployment sample count")
    if len(rows) % len(FRESH_EVENT_NAMES):
        raise ValueError("rows must form complete five-event phase streams")

    width = len(FRESH_EVENT_NAMES)
    slices = tuple(slice(start, start + width) for start in range(0, len(rows), width))
    seen_streams: set[tuple[str, Identity, int]] = set()
    phases_by_episode: dict[EpisodeKey, set[int]] = {}
    for stream_slice in slices:
        stream = rows[stream_slice]
        first = stream[0]
        if not all(isinstance(row, PosteriorGateExampleMetadata) for row in stream):
            raise TypeError("metadata rows must be PosteriorGateExampleMetadata")
        identity = (first.task, _identity(first.episode_id, label="episode_id"))
        if tuple(row.event for row in stream) != FRESH_EVENT_NAMES:
            raise ValueError("each phase stream must be ordered o0,o1,o2,r0,r1")
        if tuple(row.event_index for row in stream) != tuple(range(width)):
            raise ValueError("event_index must be 0..4 within every phase stream")
        if any(
            row.task != first.task
            or row.episode_id != first.episode_id
            or row.phase_index != first.phase_index
            or row.frame_phase != first.phase_index
            or row.stream_offset != first.stream_offset
            for row in stream
        ):
            raise ValueError("task, episode, and phase must remain fixed in a stream")
        if not 0 <= first.phase_index < FRESH_PHASE_COUNT:
            raise ValueError("phase_index must lie in 0..4")
        stream_key = (identity[0], identity[1], int(first.phase_index))
        if stream_key in seen_streams:
            raise ValueError(f"duplicate phase stream: {stream_key}")
        seen_streams.add(stream_key)
        phases_by_episode.setdefault(identity, set()).add(int(first.phase_index))
        target_frames = tuple(int(row.target_frame) for row in stream)
        if any(
            right - left != FRESH_FRAME_SKIP
            for left, right in zip(target_frames, target_frames[1:])
        ):
            raise ValueError("successive events must be one frame_skip apart")

        for local_index, row in enumerate(stream):
            history_frames = tuple(int(value) for value in row.history_frames)
            if len(history_frames) != FRESH_HISTORY_SIZE:
                raise ValueError("every event must contain exactly three history frames")
            expected_history = tuple(
                int(row.target_frame) - offset * FRESH_FRAME_SKIP
                for offset in range(FRESH_HISTORY_SIZE, 0, -1)
            )
            if history_frames != expected_history:
                raise ValueError("history_frames violate the causal history=3 contract")
            if len(row.action_frames) != FRESH_HISTORY_SIZE:
                raise ValueError("every event must contain three causal action chunks")
            for chunk_index, action_frames in enumerate(row.action_frames):
                stop = (
                    history_frames[chunk_index + 1]
                    if chunk_index + 1 < FRESH_HISTORY_SIZE
                    else int(row.target_frame)
                )
                expected_actions = tuple(range(history_frames[chunk_index], stop))
                if tuple(int(value) for value in action_frames) != expected_actions:
                    raise ValueError(
                        "action_frames must map each history state to its successor"
                    )
                if int(row.target_frame) in action_frames:
                    raise ValueError("target-frame action is forbidden")
            raw_chunk_length = len(row.action_frames[0])
            if raw_chunk_length <= 0 or tensors.action_chunks.shape[-1] % raw_chunk_length:
                raise ValueError("flattened action chunk has an invalid dimension")

            if local_index:
                row_index = stream_slice.start + local_index
                previous_index = row_index - 1
                if not torch.equal(
                    tensors.measurement_history[row_index, :-1],
                    tensors.measurement_history[previous_index, 1:],
                ) or not torch.equal(
                    tensors.measurement_history[row_index, -1],
                    tensors.measurement_current[previous_index],
                ):
                    raise ValueError(
                        "measurement histories do not advance causally within a phase"
                    )

    expected_phases = set(range(FRESH_PHASE_COUNT))
    for identity, phases in phases_by_episode.items():
        if phases != expected_phases:
            raise ValueError(
                f"physical episode {identity!r} must contain all five phases"
            )
    return slices


def build_action_shuffle_indices(
    metadata: Sequence[PosteriorGateExampleMetadata],
) -> tuple[int, ...]:
    """Build a deterministic same-task/phase cross-episode action derangement."""

    rows = tuple(metadata)
    width = len(FRESH_EVENT_NAMES)
    if not rows or len(rows) % width:
        raise ValueError("metadata must contain complete phase streams")
    stream_starts = tuple(range(0, len(rows), width))
    grouped: dict[tuple[str, int], list[int]] = {}
    seen: set[tuple[str, Identity, int]] = set()
    for start in stream_starts:
        stream = rows[start : start + width]
        if tuple(row.event for row in stream) != FRESH_EVENT_NAMES:
            raise ValueError("action shuffle requires ordered complete streams")
        first = stream[0]
        episode = _identity(first.episode_id, label="episode_id")
        key = (first.task, episode, int(first.phase_index))
        if key in seen:
            raise ValueError(f"duplicate phase stream: {key}")
        seen.add(key)
        grouped.setdefault((first.task, int(first.phase_index)), []).append(start)

    result = [-1] * len(rows)
    for group_key, starts in sorted(grouped.items()):
        starts.sort(
            key=lambda start: _identity_order(
                _identity(rows[start].episode_id, label="episode_id")
            )
        )
        if len(starts) < 2:
            raise ValueError(
                "action shuffle requires at least two physical episodes for "
                f"task/phase {group_key!r}"
            )
        donors = starts[1:] + starts[:1]
        for target_start, donor_start in zip(starts, donors, strict=True):
            for event_index in range(width):
                result[target_start + event_index] = donor_start + event_index
    if any(index < 0 for index in result):
        raise RuntimeError("action shuffle did not cover every row")
    return tuple(result)


def _normalize_shuffle_indices(
    value: Sequence[int] | np.ndarray | Tensor,
    metadata: tuple[PosteriorGateExampleMetadata, ...],
) -> tuple[int, ...]:
    if torch.is_tensor(value):
        if value.ndim != 1 or value.dtype == torch.bool or value.is_floating_point():
            raise TypeError("action_shuffle_indices must be a 1D integer tensor")
        raw = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
            raise TypeError("action_shuffle_indices must be a 1D integer array")
        raw = value.tolist()
    elif not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
        raw = list(value)
    else:
        raise TypeError("action_shuffle_indices must be an integer sequence")
    if len(raw) != len(metadata):
        raise ValueError("action_shuffle_indices must contain one donor per row")
    indices: list[int] = []
    for position, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"action_shuffle_indices[{position}] must be an integer")
        index = int(item)
        if not 0 <= index < len(metadata):
            raise IndexError("action shuffle donor is outside the row table")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("action shuffle donors must form a permutation")

    for target_index, donor_index in enumerate(indices):
        target = metadata[target_index]
        donor = metadata[donor_index]
        if target_index == donor_index:
            raise ValueError("action shuffle must be a derangement")
        if target.task != donor.task:
            raise ValueError("action shuffle donors must remain within task")
        if target.episode_id == donor.episode_id:
            raise ValueError("action shuffle donors must cross physical episodes")
        if target.phase_index != donor.phase_index or target.event != donor.event:
            raise ValueError("action shuffle must preserve phase and event")
    return tuple(indices)


def _normalize_methods(
    methods: Sequence[str] | None,
    *,
    has_shuffle: bool,
) -> tuple[str, ...]:
    if methods is None:
        result = DEFAULT_POSTERIOR_GATE_RUNTIME_METHODS + (
            (ACTION_SHUFFLE_METHOD,) if has_shuffle else ()
        )
    else:
        if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
            raise TypeError("methods must be a sequence")
        result = tuple(methods)
    if not result:
        raise ValueError("methods must not be empty")
    if any(not isinstance(method, str) for method in result):
        raise TypeError("every method must be a string")
    if len(set(result)) != len(result):
        raise ValueError("methods must not contain duplicates")
    unexpected = set(result).difference(POSTERIOR_GATE_RUNTIME_METHODS)
    if unexpected:
        raise ValueError(f"unsupported posterior runtime methods: {sorted(unexpected)}")
    return result


def _validate_adapter(
    adapter: object,
    *,
    name: str,
    tensors: _RuntimeTensors,
) -> LeWMPosteriorAdapter:
    if not isinstance(adapter, LeWMPosteriorAdapter):
        raise TypeError(f"{name} must be LeWMPosteriorAdapter")
    if adapter.latent_dim != tensors.latent_dim:
        raise ValueError(f"{name} latent_dim differs from deployment")
    if adapter.mask_history_dim != FRESH_HISTORY_SIZE:
        raise ValueError(f"{name} mask_history_dim must be {FRESH_HISTORY_SIZE}")
    if adapter.action_summary_dim != tensors.action_chunks.shape[-1]:
        raise ValueError(f"{name} action_summary_dim differs from action chunk")
    return adapter


@contextmanager
def _frozen_evaluation(modules: Sequence[object]):
    unique: list[nn.Module] = []
    for value in modules:
        if isinstance(value, nn.Module) and all(value is not item for item in unique):
            unique.append(value)
    states = [module.training for module in unique]
    try:
        for module in unique:
            module.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, training in zip(unique, states, strict=True):
            module.train(training)


def _predict_prior(
    lewm: object,
    history: Tensor,
    action_chunks: Tensor,
) -> Tensor:
    predict = getattr(lewm, "predict", None)
    if not callable(predict):
        raise TypeError("lewm must expose a callable predict(history, action)")
    predicted = predict(history, action_chunks)
    if not torch.is_tensor(predicted):
        raise TypeError("lewm.predict must return a tensor")
    if predicted.shape != history.shape:
        raise ValueError(
            "lewm.predict must return one prediction per causal history step"
        )
    if predicted.dtype != history.dtype or predicted.device != history.device:
        raise TypeError("lewm prediction must share history dtype and device")
    prior = predicted[:, -1]
    if not bool(torch.isfinite(prior).all()):
        raise ValueError("lewm prior must be finite")
    return prior


def _online_validity_history(
    tensors: _RuntimeTensors,
    row_index: int,
    *,
    image_width: int,
    image_height: int,
) -> Tensor:
    corrected = tensors.q_corrected_history[row_index]
    inside = (
        (corrected[..., 0] >= 0.0)
        & (corrected[..., 0] <= float(image_width - 1))
        & (corrected[..., 1] >= 0.0)
        & (corrected[..., 1] <= float(image_height - 1))
    )
    valid = tensors.valid_nominal_history[row_index] & inside
    return valid.transpose(0, 1).unsqueeze(0)


def _adapter_output(
    adapter: LeWMPosteriorAdapter,
    *,
    measurement: Tensor,
    prior: Tensor,
    tensors: _RuntimeTensors,
    row_index: int,
    action_summary: Tensor,
    image_width: int,
    image_height: int,
) -> LeWMPosteriorAdapterOutput:
    features = build_lewm_posterior_features(
        z_measurement=measurement,
        z_prior=prior,
        q_nominal=tensors.q_nominal_current[row_index].unsqueeze(0),
        q_corrected=tensors.q_corrected_current[row_index].unsqueeze(0),
        valid_nominal=tensors.valid_nominal_current[row_index].unsqueeze(0),
        image_width=image_width,
        image_height=image_height,
        online_validity_history=_online_validity_history(
            tensors,
            row_index,
            image_width=image_width,
            image_height=image_height,
        ),
        causal_action_summary=action_summary,
    )
    output = adapter(
        measurement,
        prior,
        features.innovation,
        features.confidence,
        features.mask_history,
        features.action_summary,
    )
    if not isinstance(output, LeWMPosteriorAdapterOutput):
        raise TypeError("posterior adapter must return LeWMPosteriorAdapterOutput")
    expected_beta = (*measurement.shape[:-1], 1)
    if output.z_post.shape != measurement.shape or output.beta.shape != expected_beta:
        raise ValueError("posterior adapter returned invalid output shapes")
    if output.z_post.dtype != measurement.dtype or output.z_post.device != measurement.device:
        raise TypeError("posterior output must share measurement dtype and device")
    if output.beta.dtype != measurement.dtype or output.beta.device != measurement.device:
        raise TypeError("posterior beta must share measurement dtype and device")
    if not bool(torch.isfinite(output.z_post).all()) or not bool(
        torch.isfinite(output.beta).all()
    ):
        raise ValueError("posterior adapter output must be finite")
    if not bool(((output.beta >= 0.0) & (output.beta <= 1.0)).all()):
        raise ValueError("posterior beta must lie in [0, 1]")
    return output


def run_recursive_posterior_gate(
    lewm: object,
    deployment: PosteriorGateDeploymentTensors,
    metadata: Sequence[PosteriorGateExampleMetadata],
    *,
    adapter: LeWMPosteriorAdapter,
    image_width: int,
    image_height: int,
    fixed_beta: Real = 0.75,
    action_blind_adapter: LeWMPosteriorAdapter | None = None,
    no_lewm_adapter: LeWMPosteriorAdapter | None = None,
    action_shuffle_indices: Sequence[int] | np.ndarray | Tensor | None = None,
    methods: Sequence[str] | None = None,
    device: torch.device | str | None = None,
) -> PosteriorGateRecursiveResult:
    """Evaluate independent phase streams with causal recursive writeback.

    ``deployment`` is the complete data boundary.  In particular, this
    function has no target, annotation, schedule, or measurement-validity
    argument.  ``action_shuffle_indices`` maps each target row to a causal
    action row from another physical episode with the same task, phase, and
    event.
    """

    action_blind = adapter if action_blind_adapter is None else action_blind_adapter
    no_lewm = adapter if no_lewm_adapter is None else no_lewm_adapter
    candidate_adapters = (adapter, action_blind, no_lewm)
    runtime_device, runtime_dtype = _runtime_device_dtype(
        lewm,
        candidate_adapters,
        deployment.measurement_history,
        device,
    )
    tensors = _runtime_tensors(
        deployment, device=runtime_device, dtype=runtime_dtype
    )
    main_adapter = _validate_adapter(adapter, name="adapter", tensors=tensors)
    blind_adapter = _validate_adapter(
        action_blind, name="action_blind_adapter", tensors=tensors
    )
    hold_last_adapter = _validate_adapter(
        no_lewm, name="no_lewm_adapter", tensors=tensors
    )
    rows = tuple(metadata)
    streams = _stream_slices(rows, tensors)
    width = _image_dimension(image_width, "image_width")
    height = _image_dimension(image_height, "image_height")
    gain = _fixed_beta(fixed_beta)

    selected_methods = _normalize_methods(
        methods, has_shuffle=action_shuffle_indices is not None
    )
    shuffle_indices: tuple[int, ...] | None = None
    if ACTION_SHUFFLE_METHOD in selected_methods:
        raw_shuffle = (
            build_action_shuffle_indices(rows)
            if action_shuffle_indices is None
            else action_shuffle_indices
        )
        shuffle_indices = _normalize_shuffle_indices(raw_shuffle, rows)

    modules = (lewm, main_adapter, blind_adapter, hold_last_adapter)
    trajectories: dict[str, PosteriorGateMethodTrajectory] = {}
    with _frozen_evaluation(modules):
        for method in selected_methods:
            prior_table = torch.empty_like(tensors.measurement_current)
            post_table = torch.empty_like(tensors.measurement_current)
            writeback_table = torch.empty_like(tensors.measurement_current)
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
                for row_index in range(stream_slice.start, stream_slice.stop):
                    source_index = (
                        shuffle_indices[row_index]
                        if method == ACTION_SHUFFLE_METHOD
                        and shuffle_indices is not None
                        else row_index
                    )
                    action_sources[row_index] = source_index
                    causal_actions = tensors.action_chunks[source_index].unsqueeze(0)
                    measurement = tensors.measurement_current[row_index].unsqueeze(0)

                    if method == NO_LEWM_METHOD:
                        prior = history[:, -1]
                    else:
                        prior = _predict_prior(lewm, history, causal_actions)

                    if method == MEASUREMENT_METHOD:
                        beta = torch.zeros(
                            1,
                            tensors.anchors,
                            1,
                            dtype=runtime_dtype,
                            device=runtime_device,
                        )
                        post = measurement
                    elif method == FIXED_BETA_METHOD:
                        beta = torch.full(
                            (1, tensors.anchors, 1),
                            gain,
                            dtype=runtime_dtype,
                            device=runtime_device,
                        )
                        post = measurement + beta * (prior - measurement)
                    else:
                        feature_action = causal_actions[:, -1]
                        method_adapter = main_adapter
                        if method == ACTION_BLIND_METHOD:
                            feature_action = torch.zeros_like(feature_action)
                            method_adapter = blind_adapter
                        elif method == NO_LEWM_METHOD:
                            method_adapter = hold_last_adapter
                        output = _adapter_output(
                            method_adapter,
                            measurement=measurement,
                            prior=prior,
                            tensors=tensors,
                            row_index=row_index,
                            action_summary=feature_action,
                            image_width=width,
                            image_height=height,
                        )
                        post = output.z_post
                        beta = output.beta

                    writeback = (
                        measurement if method == NO_WRITEBACK_METHOD else post
                    )
                    prior_table[row_index] = prior[0]
                    post_table[row_index] = post[0]
                    writeback_table[row_index] = writeback[0]
                    beta_table[row_index] = beta[0]
                    history = torch.cat(
                        (history[:, 1:], writeback.unsqueeze(1)), dim=1
                    )

            trajectories[method] = PosteriorGateMethodTrajectory(
                pre_prior=prior_table.detach().cpu().clone(),
                post_update=post_table.detach().cpu().clone(),
                writeback=writeback_table.detach().cpu().clone(),
                beta=beta_table.detach().cpu().clone(),
                action_source_index=action_sources.detach().cpu().clone(),
            )

    return PosteriorGateRecursiveResult(
        trajectories=trajectories,
        metadata=rows,
    )


def infer_direct_anchors(
    metadata: Sequence[PosteriorGateExampleMetadata],
    dropout_active: np.ndarray | Tensor,
) -> dict[EpisodeKey, Identity]:
    """Infer one offline direct anchor per physical episode for row annotation."""

    rows = tuple(metadata)
    if torch.is_tensor(dropout_active):
        active = dropout_active.detach().cpu().numpy()
    elif isinstance(dropout_active, np.ndarray):
        active = np.asarray(dropout_active)
    else:
        raise TypeError("dropout_active must be a tensor or numpy array")
    if active.shape[0] != len(rows) or active.ndim != 2:
        raise ValueError("dropout_active must have shape [N,K]")
    if active.dtype != np.bool_:
        raise TypeError("dropout_active must have boolean dtype")
    selected: dict[EpisodeKey, set[int]] = {}
    for row_index, row in enumerate(rows):
        key = (row.task, _identity(row.episode_id, label="episode_id"))
        anchors = np.flatnonzero(active[row_index])
        selected.setdefault(key, set()).update(int(value) for value in anchors)
    result: dict[EpisodeKey, Identity] = {}
    for key, anchors in selected.items():
        if len(anchors) != 1:
            raise ValueError(
                f"physical episode {key!r} must have exactly one direct anchor"
            )
        result[key] = next(iter(anchors))
    return result


def _direct_anchor_mapping(
    direct_anchor: Identity | Mapping[EpisodeKey, Identity] | None,
    metadata: Sequence[PosteriorGateExampleMetadata],
) -> dict[EpisodeKey, Identity] | None:
    if direct_anchor is None:
        return None
    episode_keys = {
        (row.task, _identity(row.episode_id, label="episode_id")) for row in metadata
    }
    if isinstance(direct_anchor, Mapping):
        result: dict[EpisodeKey, Identity] = {}
        for raw_key, raw_anchor in direct_anchor.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise TypeError("direct_anchor keys must be (task, physical_episode_id)")
            if not isinstance(raw_key[0], str) or not raw_key[0]:
                raise ValueError("direct_anchor task must be a non-empty string")
            key = (
                raw_key[0],
                _identity(raw_key[1], label="direct_anchor episode"),
            )
            result[key] = _identity(raw_anchor, label=f"direct anchor for {key}")
        if set(result) != episode_keys:
            raise ValueError("direct_anchor mapping must cover every physical episode")
        return result
    anchor = _identity(direct_anchor, label="direct_anchor")
    return {key: anchor for key in episode_keys}


def build_posterior_gate_row_records(
    recursive: PosteriorGateRecursiveResult,
    canonical_target: np.ndarray | Tensor,
    *,
    direct_anchor: Identity | Mapping[EpisodeKey, Identity] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Score trajectories and average the five phases into statistics rows."""

    if not isinstance(recursive, PosteriorGateRecursiveResult):
        raise TypeError("recursive must be PosteriorGateRecursiveResult")
    reference = next(iter(recursive.trajectories.values()), None)
    if reference is None:
        raise ValueError("recursive result must contain at least one method")
    if torch.is_tensor(canonical_target):
        canonical = canonical_target.detach().cpu().to(
            dtype=reference.pre_prior.dtype, copy=True
        )
    elif isinstance(canonical_target, np.ndarray):
        canonical = torch.as_tensor(
            np.array(canonical_target, copy=True), dtype=reference.pre_prior.dtype
        )
    else:
        raise TypeError("canonical_target must be a tensor or numpy array")
    if canonical.shape != reference.pre_prior.shape:
        raise ValueError("canonical_target must have shape [N,K,D]")
    if not canonical.is_floating_point() or not bool(torch.isfinite(canonical).all()):
        raise ValueError("canonical_target must be finite floating point")

    direct_by_episode = _direct_anchor_mapping(
        direct_anchor, recursive.metadata
    )
    grouped: dict[
        tuple[str, Identity, int, str, str, str], list[tuple[int, float]]
    ] = {}
    for method, trajectory in recursive.trajectories.items():
        for name, values in (
            ("pre_prior", trajectory.pre_prior),
            ("post_update", trajectory.post_update),
        ):
            if values.shape != canonical.shape or not bool(torch.isfinite(values).all()):
                raise ValueError(f"trajectory {method}/{name} is invalid")
            per_anchor_mse = torch.square(values - canonical).mean(dim=-1)
            for row_index, row in enumerate(recursive.metadata):
                episode = _identity(row.episode_id, label="episode_id")
                for anchor in range(per_anchor_mse.shape[1]):
                    key = (
                        row.task,
                        episode,
                        anchor,
                        name,
                        row.event,
                        method,
                    )
                    grouped.setdefault(key, []).append(
                        (int(row.phase_index), float(per_anchor_mse[row_index, anchor]))
                    )

    method_order = {name: index for index, name in enumerate(recursive.methods)}
    event_order = {name: index for index, name in enumerate(FRESH_EVENT_NAMES)}

    def key_order(
        key: tuple[str, Identity, int, str, str, str]
    ) -> tuple[Any, ...]:
        task, episode, anchor, estimand, event, method = key
        return (
            task,
            _identity_order(episode),
            anchor,
            ESTIMANDS.index(estimand),
            event_order[event],
            method_order[method],
        )

    records: list[dict[str, Any]] = []
    expected_phases = set(range(FRESH_PHASE_COUNT))
    for key in sorted(grouped, key=key_order):
        task, episode, anchor, estimand, event, method = key
        phase_values = grouped[key]
        phases = [phase for phase, _ in phase_values]
        if len(phases) != FRESH_PHASE_COUNT or set(phases) != expected_phases:
            raise ValueError("every statistics row must aggregate all five phases")
        values = [value for _, value in phase_values]
        mse = math.fsum(values) / len(values)
        if not math.isfinite(mse) or mse < 0.0:
            raise ValueError("phase-averaged MSE must be finite and non-negative")
        record: dict[str, Any] = {
            "task": task,
            "physical_episode_id": episode,
            "anchor": anchor,
            "estimand": estimand,
            "event": event,
            "method": method,
            "mse": mse,
        }
        if direct_by_episode is not None:
            record["is_direct_anchor"] = (
                anchor == direct_by_episode[(task, episode)]
            )
        records.append(record)
    return tuple(records)


def posterior_gate_examples_from_cache(
    cache: PosteriorGateCache,
) -> PosteriorGateExamples:
    """Rebuild the typed runtime boundary from a verified cache partition."""

    if not isinstance(cache, PosteriorGateCache):
        raise TypeError("cache must be PosteriorGateCache")
    count = cache.sample_count
    deployment = PosteriorGateDeploymentTensors(
        measurement_history=cache.deployment["measurement_history"],
        action_chunks=cache.deployment["action_chunks"],
        measurement_current=cache.deployment["measurement_current"],
        q_nominal_history=cache.deployment["q_nominal_history"],
        q_corrected_history=cache.deployment["q_corrected_history"],
        valid_nominal_history=cache.deployment["valid_nominal_history"],
        q_nominal_current=cache.deployment["q_nominal_current"],
        q_corrected_current=cache.deployment["q_corrected_current"],
        valid_nominal_current=cache.deployment["valid_nominal_current"],
    )

    required_evaluation = (
        "task_name",
        "episode_id",
        "phase_index",
        "stream_offset",
        "frame_phase",
        "event",
        "event_index",
        "target_frame",
        "target_raw_index",
        "history_frames",
        "action_frames",
        "dropout_active",
        "recovery_active",
    )
    evaluation_arrays: dict[str, np.ndarray] = {}
    for name in required_evaluation:
        if name not in cache.evaluation:
            raise ValueError(f"posterior cache evaluation is missing {name!r}")
        value = np.asarray(cache.evaluation[name])
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"posterior cache evaluation field {name!r} is invalid")
        evaluation_arrays[name] = value

    physical_identity_fields = ("task_index", "env_seed")
    present_identity_fields = tuple(
        name for name in physical_identity_fields if name in cache.evaluation
    )
    if present_identity_fields and present_identity_fields != physical_identity_fields:
        raise ValueError(
            "posterior cache evaluation must provide task_index and env_seed together"
        )
    for name in present_identity_fields:
        value = np.asarray(cache.evaluation[name])
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(
                f"posterior cache evaluation field {name!r} is invalid"
            )
        evaluation_arrays[name] = value

    rows: list[PosteriorGateExampleMetadata] = []
    for index in range(count):
        history_frames = tuple(
            int(value) for value in evaluation_arrays["history_frames"][index]
        )
        action_frames = tuple(
            tuple(int(value) for value in chunk)
            for chunk in evaluation_arrays["action_frames"][index]
        )
        local_episode_id = _identity(
            evaluation_arrays["episode_id"][index], label="episode_id"
        )
        if present_identity_fields:
            task_index = int(evaluation_arrays["task_index"][index])
            env_seed = int(evaluation_arrays["env_seed"][index])
            episode_id: str | int = (
                f"task_index={task_index}|env_seed={env_seed}|"
                f"episode_id={local_episode_id}"
            )
        else:
            episode_id = local_episode_id
        rows.append(
            PosteriorGateExampleMetadata(
                task=str(evaluation_arrays["task_name"][index]),
                episode_id=episode_id,
                phase_index=int(evaluation_arrays["phase_index"][index]),
                stream_offset=int(evaluation_arrays["stream_offset"][index]),
                frame_phase=int(evaluation_arrays["frame_phase"][index]),
                event=str(evaluation_arrays["event"][index]),
                event_index=int(evaluation_arrays["event_index"][index]),
                target_frame=int(evaluation_arrays["target_frame"][index]),
                target_raw_index=int(
                    evaluation_arrays["target_raw_index"][index]
                ),
                history_frames=history_frames,
                action_frames=action_frames,
            )
        )

    if "canonical_current" not in cache.labels:
        raise ValueError("posterior cache labels are missing canonical_current")
    return PosteriorGateExamples(
        deployment=deployment,
        labels=PosteriorGateLabels(
            canonical_current=cache.labels["canonical_current"]
        ),
        evaluation=PosteriorGateEvaluationMasks(
            dropout_active=evaluation_arrays["dropout_active"],
            recovery_active=evaluation_arrays["recovery_active"],
        ),
        metadata=tuple(rows),
    )


def evaluate_recursive_posterior_gate(
    lewm: object,
    examples: PosteriorGateExamples,
    *,
    adapter: LeWMPosteriorAdapter,
    image_width: int,
    image_height: int,
    fixed_beta: Real = 0.75,
    action_blind_adapter: LeWMPosteriorAdapter | None = None,
    no_lewm_adapter: LeWMPosteriorAdapter | None = None,
    action_shuffle_indices: Sequence[int] | np.ndarray | Tensor | None = None,
    methods: Sequence[str] | None = None,
    direct_anchor: Identity | Mapping[EpisodeKey, Identity] | None = None,
    device: torch.device | str | None = None,
) -> PosteriorGateRuntimeResult:
    """Run the deployment-only kernel, then score with isolated offline labels."""

    if not isinstance(examples, PosteriorGateExamples):
        raise TypeError("examples must be PosteriorGateExamples")
    recursive = run_recursive_posterior_gate(
        lewm,
        examples.deployment,
        examples.metadata,
        adapter=adapter,
        image_width=image_width,
        image_height=image_height,
        fixed_beta=fixed_beta,
        action_blind_adapter=action_blind_adapter,
        no_lewm_adapter=no_lewm_adapter,
        action_shuffle_indices=action_shuffle_indices,
        methods=methods,
        device=device,
    )
    scoring_anchor = direct_anchor
    if scoring_anchor is None:
        scoring_anchor = infer_direct_anchors(
            examples.metadata, examples.evaluation.dropout_active
        )
    records = build_posterior_gate_row_records(
        recursive,
        examples.labels.canonical_current,
        direct_anchor=scoring_anchor,
    )
    return PosteriorGateRuntimeResult(recursive=recursive, row_records=records)


def evaluate_posterior_gate_cache(
    lewm: object,
    cache: PosteriorGateCache,
    *,
    adapter: LeWMPosteriorAdapter,
    fixed_beta: Real = 0.75,
    action_blind_adapter: LeWMPosteriorAdapter | None = None,
    no_lewm_adapter: LeWMPosteriorAdapter | None = None,
    action_shuffle_indices: Sequence[int] | np.ndarray | Tensor | None = None,
    methods: Sequence[str] | None = None,
    direct_anchor: Identity | Mapping[EpisodeKey, Identity] | None = None,
    device: torch.device | str | None = None,
) -> PosteriorGateRuntimeResult:
    """Evaluate one verified cache without exposing its privileged fields."""

    examples = posterior_gate_examples_from_cache(cache)
    return evaluate_recursive_posterior_gate(
        lewm,
        examples,
        adapter=adapter,
        image_width=cache.image_width,
        image_height=cache.image_height,
        fixed_beta=fixed_beta,
        action_blind_adapter=action_blind_adapter,
        no_lewm_adapter=no_lewm_adapter,
        action_shuffle_indices=action_shuffle_indices,
        methods=methods,
        direct_anchor=direct_anchor,
        device=device,
    )


# Public spelling aliases for callers that name the operation by subsystem.
evaluate_lewm_posterior_gate = evaluate_recursive_posterior_gate
run_lewm_posterior_gate_runtime = run_recursive_posterior_gate


__all__ = [
    "ACTION_BLIND_METHOD",
    "ACTION_SHUFFLE_METHOD",
    "ADAPTER_METHOD",
    "DEFAULT_POSTERIOR_GATE_RUNTIME_METHODS",
    "FIXED_BETA_METHOD",
    "MEASUREMENT_METHOD",
    "NO_LEWM_METHOD",
    "NO_WRITEBACK_METHOD",
    "POSTERIOR_GATE_RUNTIME_FORBIDDEN_MODEL_INPUTS",
    "POSTERIOR_GATE_RUNTIME_METHODS",
    "POSTERIOR_GATE_RUNTIME_MODEL_INPUTS",
    "PosteriorGateMethodTrajectory",
    "PosteriorGateRecursiveResult",
    "PosteriorGateRuntimeResult",
    "build_action_shuffle_indices",
    "build_posterior_gate_row_records",
    "evaluate_lewm_posterior_gate",
    "evaluate_posterior_gate_cache",
    "evaluate_recursive_posterior_gate",
    "infer_direct_anchors",
    "posterior_gate_examples_from_cache",
    "run_lewm_posterior_gate_runtime",
    "run_recursive_posterior_gate",
]
