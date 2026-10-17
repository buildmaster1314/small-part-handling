"""Pure variable-duration sequences for the randomized posterior diagnostic.

This module deliberately does not use the fixed-width ``15/10`` posterior
Gate builder.  It accepts the same frame-aligned inputs, but derives each
phase-stream width from the annotated active and recovery durations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TypeAlias

import numpy as np
import torch

from fcpg.lewm_posterior_gate_data import (
    PosteriorGateDeploymentTensors,
    PosteriorGateEvaluationMasks,
    PosteriorGateExampleMetadata,
    PosteriorGateLabels,
)


Array: TypeAlias = np.ndarray | torch.Tensor
EpisodeId: TypeAlias = int | str

RANDOMIZED_FRAME_SKIP = 5
RANDOMIZED_HISTORY_SIZE = 3
RANDOMIZED_MIN_ACTIVE_FRAMES = 10
RANDOMIZED_MIN_RECOVERY_FRAMES = 10
RANDOMIZED_PHASE_COUNT = RANDOMIZED_FRAME_SKIP


@dataclass(frozen=True)
class PosteriorRandomizedSequenceExamples:
    """Variable-width phase streams with privileged data kept separate."""

    deployment: PosteriorGateDeploymentTensors
    labels: PosteriorGateLabels
    evaluation: PosteriorGateEvaluationMasks
    metadata: tuple[PosteriorGateExampleMetadata, ...]
    stream_slices: tuple[slice, ...]

    def __post_init__(self) -> None:
        count = len(self.metadata)
        for container_name, container in (
            ("deployment", self.deployment),
            ("labels", self.labels),
            ("evaluation", self.evaluation),
        ):
            for name, value in container.items():
                if value.ndim == 0 or int(value.shape[0]) != count:
                    raise ValueError(
                        f"{container_name}.{name} must have {count} leading rows"
                    )

        expected_start = 0
        for stream_slice in self.stream_slices:
            if not isinstance(stream_slice, slice):
                raise TypeError("stream_slices must contain slice objects")
            if stream_slice.step not in (None, 1):
                raise ValueError("stream slices must have unit step")
            if stream_slice.start != expected_start:
                raise ValueError("stream slices must be contiguous and ordered")
            if stream_slice.stop is None or stream_slice.stop <= expected_start:
                raise ValueError("stream slices must be nonempty")
            expected_start = stream_slice.stop
        if expected_start != count:
            raise ValueError("stream slices must cover every row exactly once")

        phases_by_episode: dict[tuple[str, EpisodeId], set[int]] = {}
        for stream_slice in self.stream_slices:
            stream = self.metadata[stream_slice]
            if not all(
                isinstance(row, PosteriorGateExampleMetadata) for row in stream
            ):
                raise TypeError(
                    "metadata rows must be PosteriorGateExampleMetadata"
                )
            first = stream[0]
            if any(
                row.task != first.task
                or row.episode_id != first.episode_id
                or row.phase_index != first.phase_index
                or row.stream_offset != first.stream_offset
                for row in stream
            ):
                raise ValueError(
                    "each stream slice must contain exactly one task, episode, "
                    "phase, and stream_offset"
                )

            if any(
                isinstance(row.phase_index, bool)
                or not isinstance(row.phase_index, Integral)
                for row in stream
            ):
                raise TypeError("phase_index must be an integer")
            phase = int(first.phase_index)
            if not 0 <= phase < RANDOMIZED_PHASE_COUNT:
                raise ValueError("phase_index must lie in 0..4")
            if any(
                isinstance(row.stream_offset, bool)
                or not isinstance(row.stream_offset, Integral)
                for row in stream
            ):
                raise TypeError("stream_offset must be an integer")
            if not 0 <= int(first.stream_offset) < RANDOMIZED_PHASE_COUNT:
                raise ValueError("stream_offset must lie in 0..4")
            if any(
                isinstance(row.frame_phase, bool)
                or not isinstance(row.frame_phase, Integral)
                for row in stream
            ):
                raise TypeError("frame_phase must be an integer")
            if any(int(row.frame_phase) != phase for row in stream):
                raise ValueError(
                    "frame_phase must match phase_index throughout every stream"
                )

            events = tuple(row.event for row in stream)
            active_count = 0
            while (
                active_count < len(events)
                and events[active_count] == f"o{active_count}"
            ):
                active_count += 1
            recovery_count = len(events) - active_count
            expected_events = tuple(
                [f"o{index}" for index in range(active_count)]
                + [f"r{index}" for index in range(recovery_count)]
            )
            if (
                active_count == 0
                or recovery_count == 0
                or events != expected_events
            ):
                raise ValueError(
                    "each phase stream must be ordered o0..oN followed by r0..rM"
                )
            event_indices = tuple(row.event_index for row in stream)
            if any(
                isinstance(event_index, bool)
                or not isinstance(event_index, Integral)
                for event_index in event_indices
            ):
                raise TypeError("event_index must be an integer")
            if tuple(int(value) for value in event_indices) != tuple(
                range(len(stream))
            ):
                raise ValueError(
                    "event_index must enumerate the full phase stream from zero"
                )

            target_frames = tuple(row.target_frame for row in stream)
            if any(
                isinstance(target_frame, bool)
                or not isinstance(target_frame, Integral)
                for target_frame in target_frames
            ):
                raise TypeError("target_frame must be an integer")
            if any(
                int(right) - int(left) != RANDOMIZED_FRAME_SKIP
                for left, right in zip(target_frames, target_frames[1:])
            ):
                raise ValueError(
                    "successive target_frame values must be five frames apart"
                )
            if any(
                int(target_frame) % RANDOMIZED_FRAME_SKIP != phase
                for target_frame in target_frames
            ):
                raise ValueError(
                    "target_frame phase must match phase_index throughout every stream"
                )

            identity = (first.task, first.episode_id)
            phases = phases_by_episode.setdefault(identity, set())
            if phase in phases:
                raise ValueError(
                    f"physical episode {identity!r} contains duplicate phase {phase}"
                )
            phases.add(phase)

        expected_phases = set(range(RANDOMIZED_PHASE_COUNT))
        for identity, phases in phases_by_episode.items():
            if phases != expected_phases:
                raise ValueError(
                    f"physical episode {identity!r} must contain phases 0..4 "
                    "exactly once"
                )

    @property
    def sample_count(self) -> int:
        return len(self.metadata)

    @property
    def phase_stream_count(self) -> int:
        return len(self.stream_slices)

    @property
    def deployment_tensors(self) -> Mapping[str, Array]:
        return self.deployment

    @property
    def label_tensors(self) -> Mapping[str, Array]:
        return self.labels


def _backend(value: object, name: str) -> str:
    if torch.is_tensor(value):
        return "torch"
    if isinstance(value, np.ndarray):
        return "numpy"
    raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray")


def _device(value: Array) -> torch.device | None:
    return value.device if torch.is_tensor(value) else None


def _is_floating(value: Array) -> bool:
    if torch.is_tensor(value):
        return bool(torch.is_floating_point(value))
    return bool(np.issubdtype(value.dtype, np.floating))


def _is_bool(value: Array) -> bool:
    if torch.is_tensor(value):
        return value.dtype == torch.bool
    return value.dtype == np.bool_


def _all_finite(value: Array) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(value).all())


def _strict_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _sequence_values(value: object, name: str) -> list[object]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional per-frame sequence")
    if torch.is_tensor(value):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return value.detach().cpu().tolist()
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array.tolist()


def _task_values(value: object, count: int) -> list[str]:
    raw = _sequence_values(value, "task")
    if len(raw) != count:
        raise ValueError(f"task must contain {count} per-frame values")
    if any(not isinstance(item, str) for item in raw):
        raise TypeError("task values must be strings")
    tasks = [item for item in raw if isinstance(item, str)]
    if any(not item for item in tasks):
        raise ValueError("task values must be non-empty strings")
    return tasks


def _episode_value(value: object, name: str) -> EpisodeId:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise TypeError(f"{name} must be an integer or non-empty string")


def _episode_values(value: object, count: int) -> list[EpisodeId]:
    raw = _sequence_values(value, "episode")
    if len(raw) != count:
        raise ValueError(f"episode must contain {count} per-frame values")
    return [
        _episode_value(item, f"episode[{index}]")
        for index, item in enumerate(raw)
    ]


def _frame_values(value: object, count: int) -> np.ndarray:
    raw = _sequence_values(value, "frame")
    if len(raw) != count:
        raise ValueError(f"frame must contain {count} per-frame values")
    frames: list[int] = []
    for index, item in enumerate(raw):
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"frame[{index}] must be an integer")
        frames.append(int(item))
    return np.asarray(frames, dtype=np.int64)


def _validate_frame_arrays(
    *,
    action: Array,
    measurement_embedding: Array,
    canonical_label: Array,
    q_nominal: Array,
    q_corrected: Array,
    valid_nominal: Array,
    anchor_dropout_active: Array,
    anchor_recovery_segment: Array,
) -> int:
    backend = _backend(measurement_embedding, "measurement_embedding")
    arrays = {
        "action": action,
        "canonical_label": canonical_label,
        "q_nominal": q_nominal,
        "q_corrected": q_corrected,
        "valid_nominal": valid_nominal,
        "anchor_dropout_active": anchor_dropout_active,
        "anchor_recovery_segment": anchor_recovery_segment,
    }
    for name, value in arrays.items():
        if _backend(value, name) != backend:
            raise TypeError("all frame tensors must share one numpy or torch backend")
        if _device(value) != _device(measurement_embedding):
            raise ValueError("all torch frame tensors must share one device")

    if measurement_embedding.ndim != 3 or any(
        int(size) <= 0 for size in measurement_embedding.shape
    ):
        raise ValueError("measurement_embedding must have nonempty shape [N,K,D]")
    count, anchors, _ = (int(size) for size in measurement_embedding.shape)
    if canonical_label.shape != measurement_embedding.shape:
        raise ValueError(
            "canonical_label must match measurement_embedding shape [N,K,D]"
        )
    if measurement_embedding.dtype != canonical_label.dtype:
        raise TypeError("measurement_embedding and canonical_label dtypes must match")
    if not _is_floating(measurement_embedding):
        raise TypeError("measurement_embedding and canonical_label must be floating")

    if action.ndim != 2 or int(action.shape[0]) != count or int(action.shape[1]) <= 0:
        raise ValueError("action must have nonempty shape [N,A]")
    if not _is_floating(action):
        raise TypeError("action must use a floating dtype")
    if action.dtype != measurement_embedding.dtype:
        raise TypeError("action dtype must match measurement_embedding")

    expected_q = (count, anchors, 2)
    if q_nominal.shape != expected_q or q_corrected.shape != expected_q:
        raise ValueError(f"q_nominal and q_corrected must have shape {expected_q}")
    if q_nominal.dtype != q_corrected.dtype or not _is_floating(q_nominal):
        raise TypeError("q_nominal and q_corrected must share a floating dtype")
    if q_nominal.dtype != measurement_embedding.dtype:
        raise TypeError(
            "q_nominal and q_corrected dtype must match measurement_embedding"
        )

    expected_mask = (count, anchors)
    for name, value in (
        ("valid_nominal", valid_nominal),
        ("anchor_dropout_active", anchor_dropout_active),
        ("anchor_recovery_segment", anchor_recovery_segment),
    ):
        if value.shape != expected_mask:
            raise ValueError(f"{name} must have shape {expected_mask}")
        if not _is_bool(value):
            raise TypeError(f"{name} must use boolean dtype")

    for name, value in (
        ("action", action),
        ("measurement_embedding", measurement_embedding),
        ("canonical_label", canonical_label),
        ("q_nominal", q_nominal),
        ("q_corrected", q_corrected),
    ):
        if not _all_finite(value):
            raise ValueError(f"{name} must be finite")
    return count


def _bool_numpy(value: Array) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.bool_, copy=False)
    return np.asarray(value, dtype=np.bool_)


def _gather(value: Array, indices: np.ndarray) -> Array:
    shape = tuple(int(size) for size in indices.shape)
    flat = np.ascontiguousarray(indices.reshape(-1), dtype=np.int64)
    if torch.is_tensor(value):
        index = torch.as_tensor(flat, dtype=torch.long, device=value.device)
        selected = value.index_select(0, index)
        return selected.reshape(*shape, *value.shape[1:])
    selected = np.asarray(value)[flat]
    return np.array(selected.reshape(*shape, *value.shape[1:]), copy=True)


def _flatten_action_chunks(action: Array, indices: np.ndarray) -> Array:
    selected = _gather(action, indices)
    samples, history, frame_skip, action_dim = selected.shape
    return selected.reshape(samples, history, frame_skip * action_dim)


def _detached_copy(value: Array) -> Array:
    if torch.is_tensor(value):
        return value.detach().clone()
    return np.array(value, copy=True)


def _episode_sort_key(
    identity: tuple[str, EpisodeId],
) -> tuple[str, tuple[int, int | str]]:
    task, episode = identity
    if isinstance(episode, int):
        episode_key: tuple[int, int | str] = (0, episode)
    else:
        episode_key = (1, episode)
    return task, episode_key


def _validate_duration(duration: int, minimum: int, label: str) -> None:
    if duration < minimum or duration % RANDOMIZED_FRAME_SKIP:
        raise ValueError(
            f"{label} duration must be a positive multiple of "
            f"{RANDOMIZED_FRAME_SKIP} and at least {minimum} frames"
        )


def _randomized_interval(
    dropout: np.ndarray,
    recovery: np.ndarray,
    *,
    identity: tuple[str, EpisodeId],
) -> tuple[int, int, int] | None:
    has_dropout = bool(dropout.any())
    has_recovery = bool(recovery.any())
    if not has_dropout and not has_recovery:
        return None
    if has_dropout != has_recovery:
        raise ValueError(
            f"episode {identity} contains a partial active/recovery schedule"
        )
    if bool(np.any(dropout & recovery)):
        raise ValueError(f"episode {identity} active and recovery masks overlap")

    active_positions = np.flatnonzero(dropout.any(axis=1))
    if not bool(np.all(np.diff(active_positions) == 1)):
        raise ValueError(f"episode {identity} active interval must be contiguous")
    active_duration = int(active_positions.size)
    _validate_duration(active_duration, RANDOMIZED_MIN_ACTIVE_FRAMES, "active")
    active_start = int(active_positions[0])
    active_stop = active_start + active_duration

    active_anchors = np.flatnonzero(dropout[active_positions].any(axis=0))
    if active_anchors.size != 1:
        raise ValueError(f"episode {identity} must have exactly one active anchor")
    anchor = int(active_anchors[0])
    expected_dropout = np.zeros_like(dropout)
    expected_dropout[active_start:active_stop, anchor] = True
    if not np.array_equal(dropout, expected_dropout):
        raise ValueError(
            f"episode {identity} must keep exactly one anchor active throughout"
        )

    recovery_positions = np.flatnonzero(recovery.any(axis=1))
    if not bool(np.all(np.diff(recovery_positions) == 1)):
        raise ValueError(f"episode {identity} recovery interval must be contiguous")
    recovery_duration = int(recovery_positions.size)
    _validate_duration(
        recovery_duration, RANDOMIZED_MIN_RECOVERY_FRAMES, "recovery"
    )
    recovery_start = int(recovery_positions[0])
    recovery_stop = recovery_start + recovery_duration
    if recovery_start != active_stop:
        raise ValueError(
            f"episode {identity} recovery must begin immediately after active"
        )
    expected_recovery = np.zeros_like(recovery)
    expected_recovery[recovery_start:recovery_stop, anchor] = True
    if not np.array_equal(recovery, expected_recovery):
        raise ValueError(
            f"episode {identity} recovery must use the single active anchor"
        )
    return active_start, active_duration, recovery_duration


def build_fresh_posterior_randomized_sequence(
    *,
    task: Sequence[object] | np.ndarray | torch.Tensor,
    episode: Sequence[object] | np.ndarray | torch.Tensor,
    frame: Sequence[object] | np.ndarray | torch.Tensor,
    action: Array,
    measurement_embedding: Array,
    canonical_label: Array,
    q_nominal: Array,
    q_corrected: Array,
    valid_nominal: Array,
    anchor_dropout_active: Array,
    anchor_recovery_segment: Array,
    observation_action_alignment: str,
    frame_skip: int = RANDOMIZED_FRAME_SKIP,
    history_size: int = RANDOMIZED_HISTORY_SIZE,
) -> PosteriorRandomizedSequenceExamples:
    """Build all active and recovery events for every raw-frame phase.

    Each annotated physical episode must have one active anchor, followed by a
    recovery interval on that anchor.  Both interval lengths may vary by
    episode in five-frame units.  Episodes without either annotation are not
    selected; malformed or partial annotations fail closed.
    """

    if not isinstance(observation_action_alignment, str):
        raise TypeError("observation_action_alignment must be an explicit string")
    if observation_action_alignment != "pre_action":
        raise ValueError(
            "observation_action_alignment must be 'pre_action' so action_t maps "
            "state_t to state_t+1"
        )
    actual_skip = _strict_positive_integer(frame_skip, "frame_skip")
    actual_history = _strict_positive_integer(history_size, "history_size")
    if actual_skip != RANDOMIZED_FRAME_SKIP:
        raise ValueError(
            f"randomized posterior frame_skip must be {RANDOMIZED_FRAME_SKIP}"
        )
    if actual_history != RANDOMIZED_HISTORY_SIZE:
        raise ValueError(
            f"randomized posterior history_size must be {RANDOMIZED_HISTORY_SIZE}"
        )

    count = _validate_frame_arrays(
        action=action,
        measurement_embedding=measurement_embedding,
        canonical_label=canonical_label,
        q_nominal=q_nominal,
        q_corrected=q_corrected,
        valid_nominal=valid_nominal,
        anchor_dropout_active=anchor_dropout_active,
        anchor_recovery_segment=anchor_recovery_segment,
    )
    tasks = _task_values(task, count)
    episodes = _episode_values(episode, count)
    frames = _frame_values(frame, count)
    dropout_numpy = _bool_numpy(anchor_dropout_active)
    recovery_numpy = _bool_numpy(anchor_recovery_segment)

    grouped: dict[tuple[str, EpisodeId], list[int]] = {}
    for index, identity in enumerate(zip(tasks, episodes, strict=True)):
        grouped.setdefault(identity, []).append(index)

    history_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    target_rows: list[int] = []
    metadata: list[PosteriorGateExampleMetadata] = []
    stream_slices: list[slice] = []
    required_history_span = actual_history * actual_skip

    for identity in sorted(grouped, key=_episode_sort_key):
        sequence = np.asarray(
            sorted(grouped[identity], key=lambda index: int(frames[index])),
            dtype=np.int64,
        )
        episode_frames = frames[sequence]
        if np.unique(episode_frames).size != episode_frames.size:
            raise ValueError(f"episode {identity} contains duplicate frame values")
        if episode_frames.size > 1 and not bool(np.all(np.diff(episode_frames) == 1)):
            raise ValueError(f"episode {identity} frames must be strictly contiguous")

        interval = _randomized_interval(
            dropout_numpy[sequence],
            recovery_numpy[sequence],
            identity=identity,
        )
        if interval is None:
            continue
        active_start, active_duration, recovery_duration = interval
        if active_start < required_history_span:
            raise ValueError(
                f"episode {identity} needs {required_history_span} contiguous "
                "pre-active frames for measurement-forced history"
            )
        recovery_start = active_start + active_duration
        recovery_stop = recovery_start + recovery_duration
        if recovery_stop > int(sequence.size):
            raise ValueError(f"episode {identity} has insufficient recovery tail")

        active_events = active_duration // actual_skip
        recovery_events = recovery_duration // actual_skip
        onset_phase = int(episode_frames[active_start]) % actual_skip
        for frame_phase in range(RANDOMIZED_PHASE_COUNT):
            stream_offset = (frame_phase - onset_phase) % actual_skip
            active_targets = active_start + stream_offset + (
                np.arange(active_events, dtype=np.int64) * actual_skip
            )
            recovery_targets = recovery_start + stream_offset + (
                np.arange(recovery_events, dtype=np.int64) * actual_skip
            )
            target_positions = np.concatenate((active_targets, recovery_targets))
            events = tuple(
                [f"o{index}" for index in range(active_events)]
                + [f"r{index}" for index in range(recovery_events)]
            )
            stream_start = len(metadata)

            for event_index, (event, target_position) in enumerate(
                zip(events, target_positions.tolist(), strict=True)
            ):
                target_raw_index = int(sequence[target_position])
                history_positions = target_position - (
                    np.arange(actual_history, 0, -1, dtype=np.int64) * actual_skip
                )
                transition_positions = history_positions[:, None] + np.arange(
                    actual_skip, dtype=np.int64
                )[None, :]
                history_indices = sequence[history_positions]
                transition_indices = sequence[transition_positions]
                history_frame_values = frames[history_indices]
                transition_frame_values = frames[transition_indices]
                target_frame = int(frames[target_raw_index])

                if target_frame % actual_skip != frame_phase:
                    raise RuntimeError("target frame does not match its phase stream")
                if int(transition_frame_values[-1, -1]) != target_frame - 1:
                    raise RuntimeError(
                        "action chunk unexpectedly reached the target action"
                    )

                history_rows.append(history_indices)
                action_rows.append(transition_indices)
                target_rows.append(target_raw_index)
                metadata.append(
                    PosteriorGateExampleMetadata(
                        task=identity[0],
                        episode_id=identity[1],
                        phase_index=frame_phase,
                        stream_offset=stream_offset,
                        frame_phase=frame_phase,
                        event=event,
                        event_index=event_index,
                        target_frame=target_frame,
                        target_raw_index=target_raw_index,
                        history_frames=tuple(
                            int(value) for value in history_frame_values.tolist()
                        ),
                        action_frames=tuple(
                            tuple(int(value) for value in row.tolist())
                            for row in transition_frame_values
                        ),
                    )
                )

            stream_stop = len(metadata)
            stream_frames = [
                row.target_frame for row in metadata[stream_start:stream_stop]
            ]
            if any(
                right - left != actual_skip
                for left, right in zip(stream_frames, stream_frames[1:])
            ):
                raise RuntimeError("successive event targets must be five frames apart")
            stream_slices.append(slice(stream_start, stream_stop))

    if not metadata:
        raise ValueError("no episode contains a complete randomized posterior window")

    history_indices = np.stack(history_rows).astype(np.int64, copy=False)
    transition_indices = np.stack(action_rows).astype(np.int64, copy=False)
    target_indices = np.asarray(target_rows, dtype=np.int64)
    deployment = PosteriorGateDeploymentTensors(
        measurement_history=_gather(measurement_embedding, history_indices),
        action_chunks=_flatten_action_chunks(action, transition_indices),
        measurement_current=_gather(measurement_embedding, target_indices),
        q_nominal_history=_gather(q_nominal, history_indices),
        q_corrected_history=_gather(q_corrected, history_indices),
        valid_nominal_history=_gather(valid_nominal, history_indices),
        q_nominal_current=_gather(q_nominal, target_indices),
        q_corrected_current=_gather(q_corrected, target_indices),
        valid_nominal_current=_gather(valid_nominal, target_indices),
    )
    labels = PosteriorGateLabels(
        canonical_current=_detached_copy(_gather(canonical_label, target_indices))
    )
    evaluation = PosteriorGateEvaluationMasks(
        dropout_active=_gather(anchor_dropout_active, target_indices),
        recovery_active=_gather(anchor_recovery_segment, target_indices),
    )
    return PosteriorRandomizedSequenceExamples(
        deployment=deployment,
        labels=labels,
        evaluation=evaluation,
        metadata=tuple(metadata),
        stream_slices=tuple(stream_slices),
    )


# Explicit aliases make the examples/result naming available without coupling
# this protocol to the fixed-width Gate builder.
build_fresh_posterior_randomized_sequence_examples = (
    build_fresh_posterior_randomized_sequence
)
build_posterior_randomized_sequence = build_fresh_posterior_randomized_sequence
build_posterior_randomized_sequence_examples = (
    build_fresh_posterior_randomized_sequence
)


__all__ = [
    "PosteriorRandomizedSequenceExamples",
    "RANDOMIZED_FRAME_SKIP",
    "RANDOMIZED_HISTORY_SIZE",
    "RANDOMIZED_MIN_ACTIVE_FRAMES",
    "RANDOMIZED_MIN_RECOVERY_FRAMES",
    "RANDOMIZED_PHASE_COUNT",
    "build_fresh_posterior_randomized_sequence",
    "build_fresh_posterior_randomized_sequence_examples",
    "build_posterior_randomized_sequence",
    "build_posterior_randomized_sequence_examples",
]
