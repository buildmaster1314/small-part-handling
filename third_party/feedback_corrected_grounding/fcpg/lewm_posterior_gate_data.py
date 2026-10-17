"""Pure sequence construction for the fresh LeWM posterior Gate protocol.

The builder in this module has no file or CLI responsibilities.  It turns
precomputed, frame-aligned tensors into measurement-forced LeWM examples while
keeping offline canonical labels and annotation-derived metadata outside the
deployment input container.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import TypeAlias

import numpy as np
import torch


Array: TypeAlias = np.ndarray | torch.Tensor
EpisodeId: TypeAlias = int | str

FRESH_FRAME_SKIP = 5
FRESH_HISTORY_SIZE = 3
FRESH_DROPOUT_FRAMES = 15
FRESH_RECOVERY_FRAMES = 10
FRESH_PHASE_COUNT = FRESH_FRAME_SKIP
FRESH_EVENT_NAMES = ("o0", "o1", "o2", "r0", "r1")

DEPLOYMENT_TENSOR_KEYS = (
    "measurement_history",
    "action_chunks",
    "measurement_current",
    "q_nominal_history",
    "q_corrected_history",
    "valid_nominal_history",
    "q_nominal_current",
    "q_corrected_current",
    "valid_nominal_current",
)
LABEL_TENSOR_KEYS = ("canonical_current",)
EVALUATION_MASK_KEYS = ("dropout_active", "recovery_active")
PHYSICAL_EPISODE_SPLITS = ("train", "dev", "sealed")


class _ArrayMapping(Mapping[str, Array]):
    _keys: tuple[str, ...]

    def __getitem__(self, key: str) -> Array:
        if key not in self._keys:
            raise KeyError(key)
        value = getattr(self, key)
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            raise RuntimeError(f"{key} is not an array")
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def as_dict(self) -> dict[str, Array]:
        return {key: self[key] for key in self}


@dataclass(frozen=True)
class PosteriorGateDeploymentTensors(_ArrayMapping):
    """The complete allowlist of tensors that a deployed model may consume."""

    measurement_history: Array
    action_chunks: Array
    measurement_current: Array
    q_nominal_history: Array
    q_corrected_history: Array
    valid_nominal_history: Array
    q_nominal_current: Array
    q_corrected_current: Array
    valid_nominal_current: Array

    _keys = DEPLOYMENT_TENSOR_KEYS

    @property
    def action(self) -> Array:
        """Upstream LeWM alias for the transition action chunks."""

        return self.action_chunks


@dataclass(frozen=True)
class PosteriorGateLabels(_ArrayMapping):
    """Offline supervision, deliberately separate from deployment inputs."""

    canonical_current: Array

    _keys = LABEL_TENSOR_KEYS


@dataclass(frozen=True)
class PosteriorGateEvaluationMasks(_ArrayMapping):
    """Current-frame annotations available only for windowed evaluation."""

    dropout_active: Array
    recovery_active: Array

    _keys = EVALUATION_MASK_KEYS


@dataclass(frozen=True)
class PosteriorGateExampleMetadata:
    """Physical and phase-stream identity for one flat training row."""

    task: str
    episode_id: EpisodeId
    phase_index: int
    stream_offset: int
    frame_phase: int
    event: str
    event_index: int
    target_frame: int
    target_raw_index: int
    history_frames: tuple[int, ...]
    action_frames: tuple[tuple[int, ...], ...]

    @property
    def physical_episode(self) -> tuple[str, EpisodeId]:
        return (self.task, self.episode_id)


@dataclass(frozen=True)
class PosteriorGateExamples:
    """Fresh Gate rows with privileged information kept outside deployment."""

    deployment: PosteriorGateDeploymentTensors
    labels: PosteriorGateLabels
    evaluation: PosteriorGateEvaluationMasks
    metadata: tuple[PosteriorGateExampleMetadata, ...]

    @property
    def sample_count(self) -> int:
        return len(self.metadata)

    @property
    def phase_stream_count(self) -> int:
        if self.sample_count % len(FRESH_EVENT_NAMES):
            raise RuntimeError("posterior Gate rows do not form complete phase streams")
        return self.sample_count // len(FRESH_EVENT_NAMES)

    @property
    def deployment_tensors(self) -> Mapping[str, Array]:
        return self.deployment

    @property
    def label_tensors(self) -> Mapping[str, Array]:
        return self.labels

    @property
    def stream_slices(self) -> tuple[slice, ...]:
        width = len(FRESH_EVENT_NAMES)
        return tuple(
            slice(start, start + width)
            for start in range(0, self.sample_count, width)
        )


def _backend(value: object, name: str) -> str:
    if torch.is_tensor(value):
        return "torch"
    if isinstance(value, np.ndarray):
        return "numpy"
    raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray")


def _is_floating(value: Array) -> bool:
    if torch.is_tensor(value):
        return bool(torch.is_floating_point(value))
    return bool(np.issubdtype(value.dtype, np.floating))


def _all_finite(value: Array) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(value).all())


def _is_bool(value: Array) -> bool:
    if torch.is_tensor(value):
        return value.dtype == torch.bool
    return value.dtype == np.bool_


def _device(value: Array) -> torch.device | None:
    return value.device if torch.is_tensor(value) else None


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
) -> tuple[int, int]:
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
        size <= 0 for size in measurement_embedding.shape
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

    if action.ndim != 2 or action.shape[0] != count or action.shape[1] <= 0:
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
    return count, anchors


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
    episode_key: tuple[int, int | str]
    if isinstance(episode, int):
        episode_key = (0, episode)
    else:
        episode_key = (1, episode)
    return task, episode_key


def _fresh_interval(
    dropout: np.ndarray,
    recovery: np.ndarray,
    *,
    identity: tuple[str, EpisodeId],
) -> tuple[int, np.ndarray] | None:
    has_dropout = bool(dropout.any())
    has_recovery = bool(recovery.any())
    if not has_dropout and not has_recovery:
        return None
    if has_dropout != has_recovery:
        raise ValueError(f"episode {identity} must contain both dropout and recovery")
    if bool(np.any(dropout & recovery)):
        raise ValueError(f"episode {identity} dropout and recovery masks overlap")

    dropout_positions = np.flatnonzero(dropout.any(axis=1))
    if dropout_positions.size != FRESH_DROPOUT_FRAMES:
        raise ValueError(
            f"episode {identity} must contain exactly {FRESH_DROPOUT_FRAMES} "
            "dropout frames"
        )
    if not bool(np.all(np.diff(dropout_positions) == 1)):
        raise ValueError(f"episode {identity} dropout frames must be consecutive")
    start = int(dropout_positions[0])
    dropout_stop = start + FRESH_DROPOUT_FRAMES
    active_anchors = np.flatnonzero(dropout[dropout_positions].any(axis=0))
    if active_anchors.size == 0:
        raise ValueError(f"episode {identity} dropout must affect at least one anchor")
    expected_dropout = np.zeros_like(dropout)
    expected_dropout[start:dropout_stop, active_anchors] = True
    if not np.array_equal(dropout, expected_dropout):
        raise ValueError(
            f"episode {identity} must use one fixed anchor set for all "
            f"{FRESH_DROPOUT_FRAMES} dropout frames"
        )

    recovery_positions = np.flatnonzero(recovery.any(axis=1))
    if recovery_positions.size != FRESH_RECOVERY_FRAMES:
        raise ValueError(
            f"episode {identity} must contain exactly {FRESH_RECOVERY_FRAMES} "
            "finite recovery frames"
        )
    if not bool(np.all(np.diff(recovery_positions) == 1)):
        raise ValueError(f"episode {identity} recovery frames must be consecutive")
    recovery_stop = dropout_stop + FRESH_RECOVERY_FRAMES
    if int(recovery_positions[0]) != dropout_stop:
        raise ValueError(
            f"episode {identity} recovery must begin immediately after dropout"
        )
    expected_recovery = np.zeros_like(recovery)
    expected_recovery[dropout_stop:recovery_stop, active_anchors] = True
    if not np.array_equal(recovery, expected_recovery):
        raise ValueError(
            f"episode {identity} recovery must use the dropout anchor set for "
            f"exactly {FRESH_RECOVERY_FRAMES} frames"
        )
    return start, active_anchors


def build_fresh_posterior_gate_examples(
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
    frame_skip: int = FRESH_FRAME_SKIP,
    history_size: int = FRESH_HISTORY_SIZE,
) -> PosteriorGateExamples:
    """Build exact ``o0/o1/o2/r0/r1`` rows for each raw-frame phase stream.

    Every annotated physical episode contributes five streams, one for each raw
    frame phase.  Each stream contributes the five events in
    :data:`FRESH_EVENT_NAMES`, for 25 flat rows.  A row targeting state ``t``
    uses deployment measurements at ``t-15, t-10, t-5`` and action chunks
    ``a[t-15:t-10]``, ``a[t-10:t-5]``, and ``a[t-5:t]``.  Thus ``a[t]`` is never
    part of the example.

    Episodes with neither annotation are ignored.  Partially or incorrectly
    annotated episodes fail closed.
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
    if actual_skip != FRESH_FRAME_SKIP:
        raise ValueError(f"fresh posterior Gate frame_skip must be {FRESH_FRAME_SKIP}")
    if actual_history != FRESH_HISTORY_SIZE:
        raise ValueError(
            f"fresh posterior Gate history_size must be {FRESH_HISTORY_SIZE}"
        )

    count, _ = _validate_frame_arrays(
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

        interval = _fresh_interval(
            dropout_numpy[sequence],
            recovery_numpy[sequence],
            identity=identity,
        )
        if interval is None:
            continue
        dropout_start, _ = interval
        required_history_span = actual_history * actual_skip
        if dropout_start < required_history_span:
            raise ValueError(
                f"episode {identity} needs {required_history_span} contiguous "
                "pre-dropout frames for measurement-forced history"
            )

        for stream_offset in range(FRESH_PHASE_COUNT):
            for event_index, event in enumerate(FRESH_EVENT_NAMES):
                target_position = (
                    dropout_start + stream_offset + event_index * actual_skip
                )
                target_raw_index = int(sequence[target_position])
                history_positions = target_position - (
                    np.arange(actual_history, 0, -1, dtype=np.int64) * actual_skip
                )
                transition_positions = (
                    history_positions[:, None]
                    + np.arange(actual_skip, dtype=np.int64)[None, :]
                )
                history_indices = sequence[history_positions]
                transition_indices = sequence[transition_positions]
                history_frame_values = frames[history_indices]
                transition_frame_values = frames[transition_indices]
                target_frame = int(frames[target_raw_index])
                if int(transition_frame_values[-1, -1]) != target_frame - 1:
                    raise RuntimeError(
                        "action chunk unexpectedly reached the target action"
                    )
                phase_index = target_frame % actual_skip

                history_rows.append(history_indices)
                action_rows.append(transition_indices)
                target_rows.append(target_raw_index)
                metadata.append(
                    PosteriorGateExampleMetadata(
                        task=identity[0],
                        episode_id=identity[1],
                        phase_index=phase_index,
                        stream_offset=stream_offset,
                        frame_phase=phase_index,
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

    if not metadata:
        raise ValueError("no episode contains a complete fresh posterior Gate window")

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
    return PosteriorGateExamples(
        deployment=deployment,
        labels=labels,
        evaluation=evaluation,
        metadata=tuple(metadata),
    )


# The shorter name is the public default; the explicit name documents that this
# builder is not compatible with the older development-only posterior protocol.
build_posterior_gate_examples = build_fresh_posterior_gate_examples


def _strict_integer(value: object, name: str) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validated_physical_identity_values(
    task: object,
    task_index: object,
    env_seed: object,
    task_generation_seed: object,
) -> tuple[str, int, int, int]:
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    return (
        task,
        _strict_integer(task_index, "task_index"),
        _strict_integer(env_seed, "env_seed"),
        _strict_integer(task_generation_seed, "task_generation_seed"),
    )


@dataclass(frozen=True)
class PhysicalEpisodeIdentity:
    """Simulator identity fields that define one physical episode."""

    task: str
    task_index: int
    env_seed: int
    task_generation_seed: int

    def __post_init__(self) -> None:
        task, task_index, env_seed, task_generation_seed = (
            _validated_physical_identity_values(
                self.task,
                self.task_index,
                self.env_seed,
                self.task_generation_seed,
            )
        )
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "task_index", task_index)
        object.__setattr__(self, "env_seed", env_seed)
        object.__setattr__(self, "task_generation_seed", task_generation_seed)

    @property
    def sha256(self) -> str:
        return physical_episode_sha256(
            self.task,
            self.task_index,
            self.env_seed,
            self.task_generation_seed,
        )


def physical_episode_sha256(
    task: str,
    task_index: int,
    env_seed: int,
    task_generation_seed: int,
) -> str:
    """Hash the complete fresh-Gate physical episode identity."""

    task, task_index, env_seed, task_generation_seed = (
        _validated_physical_identity_values(
            task,
            task_index,
            env_seed,
            task_generation_seed,
        )
    )
    payload = json.dumps(
        {
            "env_seed": env_seed,
            "task": task,
            "task_generation_seed": task_generation_seed,
            "task_index": task_index,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


IdentityLike: TypeAlias = PhysicalEpisodeIdentity | Mapping[str, object]
_IDENTITY_KEYS = frozenset(
    {"task", "task_index", "env_seed", "task_generation_seed"}
)


def _coerce_identity(value: IdentityLike, label: str) -> PhysicalEpisodeIdentity:
    if isinstance(value, PhysicalEpisodeIdentity):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a PhysicalEpisodeIdentity or mapping")
    if set(value) != set(_IDENTITY_KEYS):
        missing = sorted(_IDENTITY_KEYS.difference(value))
        extra = sorted(set(value).difference(_IDENTITY_KEYS))
        raise ValueError(
            f"{label} identity keys changed: missing={missing}, extra={extra}"
        )
    return PhysicalEpisodeIdentity(
        task=value["task"],  # type: ignore[arg-type]
        task_index=value["task_index"],  # type: ignore[arg-type]
        env_seed=value["env_seed"],  # type: ignore[arg-type]
        task_generation_seed=value["task_generation_seed"],  # type: ignore[arg-type]
    )


def build_physical_episode_sha_manifest(
    records: Sequence[IdentityLike],
    *,
    split: str | None = None,
) -> tuple[str, ...]:
    """Return a deterministic SHA-sorted manifest for physical episode rows."""

    label = "manifest" if split is None else f"{split} manifest"
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Sequence):
        raise TypeError(f"{label} records must be a sequence")
    if not records:
        raise ValueError(f"{label} must contain at least one physical episode")
    identities = [
        _coerce_identity(record, f"{label}[{index}]")
        for index, record in enumerate(records)
    ]
    digests = [identity.sha256 for identity in identities]
    if len(set(digests)) != len(digests):
        raise ValueError(f"{label} contains duplicate physical episodes")
    return tuple(sorted(digests))


@dataclass(frozen=True)
class PhysicalEpisodeShaManifests(Mapping[str, tuple[str, ...]]):
    """The three mandatory, pairwise-disjoint physical episode manifests."""

    train: tuple[str, ...]
    dev: tuple[str, ...]
    sealed: tuple[str, ...]

    def __getitem__(self, key: str) -> tuple[str, ...]:
        if key not in PHYSICAL_EPISODE_SPLITS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(PHYSICAL_EPISODE_SPLITS)

    def __len__(self) -> int:
        return len(PHYSICAL_EPISODE_SPLITS)

    def as_dict(self) -> dict[str, list[str]]:
        return {split: list(self[split]) for split in self}


def _validated_sha_manifest(
    value: object,
    *,
    split: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{split} episode SHA manifest must be a sequence")
    digests = tuple(value)
    if not digests:
        raise ValueError(f"{split} episode SHA manifest must not be empty")
    for index, digest in enumerate(digests):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"{split} episode SHA manifest[{index}] is not lowercase SHA-256"
            )
    if len(set(digests)) != len(digests):
        raise ValueError(f"{split} episode SHA manifest contains duplicates")
    return digests


def validate_disjoint_episode_sha_manifests(
    manifests: Mapping[str, Sequence[str]] | PhysicalEpisodeShaManifests,
) -> None:
    """Strictly validate exact split names, SHA syntax, and pairwise isolation."""

    if not isinstance(manifests, Mapping):
        raise TypeError("episode SHA manifests must be a mapping")
    if set(manifests) != set(PHYSICAL_EPISODE_SPLITS):
        missing = sorted(set(PHYSICAL_EPISODE_SPLITS).difference(manifests))
        extra = sorted(set(manifests).difference(PHYSICAL_EPISODE_SPLITS))
        raise ValueError(
            f"episode SHA manifest splits changed: missing={missing}, extra={extra}"
        )
    validated = {
        split: _validated_sha_manifest(manifests[split], split=split)
        for split in PHYSICAL_EPISODE_SPLITS
    }
    for left_index, left in enumerate(PHYSICAL_EPISODE_SPLITS):
        for right in PHYSICAL_EPISODE_SPLITS[left_index + 1 :]:
            overlap = sorted(set(validated[left]).intersection(validated[right]))
            if overlap:
                raise ValueError(
                    f"physical episode overlap between {left} and {right}: "
                    f"{overlap[:3]}"
                )


def build_disjoint_episode_sha_manifests(
    *,
    train: Sequence[IdentityLike],
    dev: Sequence[IdentityLike],
    sealed: Sequence[IdentityLike],
) -> PhysicalEpisodeShaManifests:
    """Build and validate train/dev/sealed physical episode SHA manifests."""

    result = PhysicalEpisodeShaManifests(
        train=build_physical_episode_sha_manifest(train, split="train"),
        dev=build_physical_episode_sha_manifest(dev, split="dev"),
        sealed=build_physical_episode_sha_manifest(sealed, split="sealed"),
    )
    validate_disjoint_episode_sha_manifests(result)
    return result


build_disjoint_physical_episode_sha_manifests = (
    build_disjoint_episode_sha_manifests
)
validate_disjoint_physical_episode_sha_manifests = (
    validate_disjoint_episode_sha_manifests
)
