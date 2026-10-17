"""Frozen task identity contract for the seven-task LeWM A4 panel.

``task_class_index`` is the position of ``task_name`` in :data:`TASKS` and
therefore has exactly seven possible values.  MetaWorld's ``task_index`` has a
different meaning: it selects one of the 48 environment/task instances.  That
value belongs in ``task_instance_index`` and must never be used as an A4 task
class.

The scalar API is intended for collectors.  The vector API additionally
checks the cross-row invariants required by storage and training: within one
task, source episode and task instance are one-to-one and neither identity may
cross a split.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


TASKS = (
    "pick-place-v3",
    "drawer-retrieve-v0",
    "reach-v3",
    "push-v3",
    "door-open-v3",
    "window-open-v3",
    "towel-wipe-v0",
)
TASK_CLASS_COUNT = len(TASKS)
TASK_INSTANCE_COUNT = 48
TASK_CLASS_INDICES = tuple(range(TASK_CLASS_COUNT))
TASK_INSTANCE_INDICES = tuple(range(TASK_INSTANCE_COUNT))
VALID_SPLITS = ("train", "development", "test")
IDENTITY_KEYS = (
    "task_name",
    "task_class_index",
    "task_instance_index",
    "source_episode_id",
    "split",
)

TASK_NAME_TO_CLASS_INDEX: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(TASKS)}
)
TASK_CLASS_INDEX_TO_NAME: Mapping[int, str] = MappingProxyType(
    {index: name for index, name in enumerate(TASKS)}
)

SourceEpisodeId = str | int
VectorInput = Sequence[object] | np.ndarray


def _text(value: object, *, label: str) -> str:
    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} must contain valid UTF-8") from error
    raise TypeError(f"{label} must be a string")


def _bounded_index(value: object, *, label: str, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if not 0 <= result < upper:
        raise ValueError(f"{label} must be in the inclusive range 0..{upper - 1}")
    return int(result)


def validate_task_name(task_name: object) -> str:
    """Return one exact task name from the frozen ordered panel."""

    name = _text(task_name, label="task_name")
    if name not in TASK_NAME_TO_CLASS_INDEX:
        raise ValueError(
            f"unknown A4 task_name {name!r}; expected one of {TASKS}"
        )
    return name


def validate_task_class_index(task_class_index: object) -> int:
    """Validate an index into ``TASKS``, never a MetaWorld task index."""

    return _bounded_index(
        task_class_index,
        label="task_class_index",
        upper=TASK_CLASS_COUNT,
    )


def validate_task_instance_index(task_instance_index: object) -> int:
    """Validate the 0..47 environment instance selected for one A4 task."""

    return _bounded_index(
        task_instance_index,
        label="task_instance_index",
        upper=TASK_INSTANCE_COUNT,
    )


def task_name_to_class_index(task_name: object) -> int:
    """Map an exact A4 task name to its frozen class index."""

    return TASK_NAME_TO_CLASS_INDEX[validate_task_name(task_name)]


def task_class_index_to_name(task_class_index: object) -> str:
    """Map a frozen A4 class index to its exact task name."""

    return TASK_CLASS_INDEX_TO_NAME[
        validate_task_class_index(task_class_index)
    ]


def validate_source_episode_id(source_episode_id: object) -> SourceEpisodeId:
    """Normalize a non-empty text or non-negative integer episode identity."""

    if isinstance(source_episode_id, (bool, np.bool_)):
        raise TypeError("source_episode_id must not be bool")
    if isinstance(source_episode_id, (str, np.str_, bytes, np.bytes_)):
        identity = _text(source_episode_id, label="source_episode_id")
        if not identity.strip():
            raise ValueError("source_episode_id must be non-empty")
        return identity
    try:
        identity = operator.index(source_episode_id)
    except TypeError as error:
        raise TypeError(
            "source_episode_id must be a string or integer"
        ) from error
    if identity < 0:
        raise ValueError("integer source_episode_id must be non-negative")
    if identity > np.iinfo(np.int64).max:
        raise ValueError("integer source_episode_id must fit signed int64")
    return int(identity)


def validate_split(split: object) -> str:
    """Return one exact formal A4 split label."""

    label = _text(split, label="split")
    if label not in VALID_SPLITS:
        raise ValueError(
            f"unsupported A4 split {label!r}; expected one of {VALID_SPLITS}"
        )
    return label


@dataclass(frozen=True, slots=True)
class A4TaskIdentity:
    """One fully validated A4 task/source/split identity row."""

    task_name: str
    task_class_index: int
    task_instance_index: int
    source_episode_id: SourceEpisodeId
    split: str

    def __post_init__(self) -> None:
        name = validate_task_name(self.task_name)
        class_index = validate_task_class_index(self.task_class_index)
        expected_index = task_name_to_class_index(name)
        if class_index != expected_index:
            raise ValueError(
                "task_name/task_class_index mismatch: "
                f"{name!r} maps to {expected_index}, got {class_index}; "
                "task_class_index indexes TASKS and is not MetaWorld "
                "task_index/task instance"
            )
        object.__setattr__(self, "task_name", name)
        object.__setattr__(self, "task_class_index", class_index)
        object.__setattr__(
            self,
            "task_instance_index",
            validate_task_instance_index(self.task_instance_index),
        )
        object.__setattr__(
            self,
            "source_episode_id",
            validate_source_episode_id(self.source_episode_id),
        )
        object.__setattr__(self, "split", validate_split(self.split))


def validate_a4_task_identity(
    task_name: object,
    task_class_index: object,
    task_instance_index: object,
    source_episode_id: object,
    split: object,
) -> A4TaskIdentity:
    """Validate and normalize one collector identity row."""

    return A4TaskIdentity(
        task_name=task_name,  # type: ignore[arg-type]
        task_class_index=task_class_index,  # type: ignore[arg-type]
        task_instance_index=task_instance_index,  # type: ignore[arg-type]
        source_episode_id=source_episode_id,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
    )


def _vector(value: VectorInput, *, label: str) -> np.ndarray:
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        raise TypeError(f"{label} must be a one-dimensional vector")
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a one-dimensional vector") from error
    if array.ndim != 1:
        raise ValueError(f"{label} must have shape [N]")
    return array


def _readonly(value: Sequence[Any], *, dtype: Any = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class A4TaskIdentityVectors:
    """Normalized, read-only aligned identity vectors for an A4 row batch."""

    task_name: np.ndarray
    task_class_index: np.ndarray
    task_instance_index: np.ndarray
    source_episode_id: np.ndarray
    split: np.ndarray

    def __post_init__(self) -> None:
        raw = {
            "task_name": _vector(self.task_name, label="task_name"),
            "task_class_index": _vector(
                self.task_class_index, label="task_class_index"
            ),
            "task_instance_index": _vector(
                self.task_instance_index, label="task_instance_index"
            ),
            "source_episode_id": _vector(
                self.source_episode_id, label="source_episode_id"
            ),
            "split": _vector(self.split, label="split"),
        }
        lengths = {name: int(value.shape[0]) for name, value in raw.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "A4 task identity vectors must have equal length: "
                f"{lengths}"
            )
        count = next(iter(lengths.values()))
        if count == 0:
            raise ValueError("A4 task identity vectors must not be empty")

        identities: list[A4TaskIdentity] = []
        for row, values in enumerate(
            zip(*(raw[name].tolist() for name in IDENTITY_KEYS), strict=True)
        ):
            try:
                identities.append(validate_a4_task_identity(*values))
            except (TypeError, ValueError) as error:
                raise type(error)(f"A4 identity row {row}: {error}") from error

        source_kinds = {
            "integer" if isinstance(item.source_episode_id, int) else "string"
            for item in identities
        }
        if len(source_kinds) != 1:
            raise TypeError(
                "source_episode_id must not mix string and integer identities"
            )

        instance_by_episode: dict[tuple[int, SourceEpisodeId], int] = {}
        episode_by_instance: dict[tuple[int, int], SourceEpisodeId] = {}
        split_by_episode: dict[tuple[int, SourceEpisodeId], str] = {}
        split_by_instance: dict[tuple[int, int], str] = {}
        for item in identities:
            episode_key = (item.task_class_index, item.source_episode_id)
            instance_key = (item.task_class_index, item.task_instance_index)

            previous_instance = instance_by_episode.setdefault(
                episode_key, item.task_instance_index
            )
            if previous_instance != item.task_instance_index:
                raise ValueError(
                    "one source_episode_id changes task_instance_index "
                    "within one A4 task"
                )
            previous_episode = episode_by_instance.setdefault(
                instance_key, item.source_episode_id
            )
            if previous_episode != item.source_episode_id:
                raise ValueError(
                    "one task_instance_index maps to multiple "
                    "source_episode_id values within one A4 task"
                )

            previous_split = split_by_episode.setdefault(episode_key, item.split)
            if previous_split != item.split:
                raise ValueError(
                    "one task/source_episode_id appears in multiple splits"
                )
            previous_instance_split = split_by_instance.setdefault(
                instance_key, item.split
            )
            if previous_instance_split != item.split:
                raise ValueError(
                    "one task/task_instance_index appears in multiple splits"
                )

        names = [item.task_name for item in identities]
        class_indices = [item.task_class_index for item in identities]
        instance_indices = [item.task_instance_index for item in identities]
        source_ids = [item.source_episode_id for item in identities]
        splits = [item.split for item in identities]
        source_dtype = np.int64 if source_kinds == {"integer"} else str

        object.__setattr__(self, "task_name", _readonly(names, dtype=str))
        object.__setattr__(
            self,
            "task_class_index",
            _readonly(class_indices, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "task_instance_index",
            _readonly(instance_indices, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "source_episode_id",
            _readonly(source_ids, dtype=source_dtype),
        )
        object.__setattr__(self, "split", _readonly(splits, dtype=str))

    @property
    def sample_count(self) -> int:
        """Number of aligned rows represented by every vector."""

        return int(self.task_name.shape[0])

    def as_arrays(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        """Return identity columns in the canonical storage field order."""

        arrays = {name: getattr(self, name) for name in IDENTITY_KEYS}
        if copy:
            return {name: value.copy() for name, value in arrays.items()}
        return arrays


def validate_a4_task_identity_vectors(
    task_name: VectorInput,
    task_class_index: VectorInput,
    task_instance_index: VectorInput,
    source_episode_id: VectorInput,
    split: VectorInput,
) -> A4TaskIdentityVectors:
    """Validate five aligned A4 identity vectors and all cross-row invariants."""

    return A4TaskIdentityVectors(
        task_name=task_name,  # type: ignore[arg-type]
        task_class_index=task_class_index,  # type: ignore[arg-type]
        task_instance_index=task_instance_index,  # type: ignore[arg-type]
        source_episode_id=source_episode_id,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
    )


def validate_a4_task_identity_arrays(
    arrays: Mapping[str, VectorInput],
) -> A4TaskIdentityVectors:
    """Validate identity columns from a larger collector/storage array mapping."""

    if not isinstance(arrays, Mapping):
        raise TypeError("A4 task identity arrays must be a mapping")
    if "task_index" in arrays:
        raise ValueError(
            "ambiguous task_index is forbidden in the A4 identity contract; "
            "use task_class_index for TASKS and task_instance_index for the "
            "MetaWorld task index/task instance"
        )
    missing = [name for name in IDENTITY_KEYS if name not in arrays]
    if missing:
        raise ValueError(f"A4 task identity arrays are missing keys: {missing}")
    return validate_a4_task_identity_vectors(
        *(arrays[name] for name in IDENTITY_KEYS)
    )


__all__ = [
    "A4TaskIdentity",
    "A4TaskIdentityVectors",
    "IDENTITY_KEYS",
    "TASKS",
    "TASK_CLASS_COUNT",
    "TASK_CLASS_INDICES",
    "TASK_CLASS_INDEX_TO_NAME",
    "TASK_INSTANCE_COUNT",
    "TASK_INSTANCE_INDICES",
    "TASK_NAME_TO_CLASS_INDEX",
    "VALID_SPLITS",
    "task_class_index_to_name",
    "task_name_to_class_index",
    "validate_a4_task_identity",
    "validate_a4_task_identity_arrays",
    "validate_a4_task_identity_vectors",
    "validate_source_episode_id",
    "validate_split",
    "validate_task_class_index",
    "validate_task_instance_index",
    "validate_task_name",
]
