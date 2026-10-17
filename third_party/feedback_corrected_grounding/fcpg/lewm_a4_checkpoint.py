"""Fail-closed metadata for one shared seven-task A4 checkpoint."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from fcpg.lewm_action_branch_a4 import (
    BUNDLE_BATCH_SIZE,
    BUNDLES_PER_TASK_PER_BATCH,
    SIGREG_WEIGHT,
    SPLIT_COUNTS,
    TASKS,
)
from fcpg.lewm_task_context_persistent import (
    TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT,
    TASK_CONTEXT_PERSISTENT_VARIANT,
)


A4_CHECKPOINT_TYPE = "fcpg_lewm_action_branch_a4_shared_v1"
A4_CHECKPOINT_SCHEMA_VERSION = 1
A4_NUM_TASKS = 7
A4_MODELS_PER_SEED = 1
A4_SPLITS = tuple(name for name, _ in SPLIT_COUNTS)
A4_TASK_TO_INDEX = {task: index for index, task in enumerate(TASKS)}
A4_STATE_KEYS = ("model_state", "no_action_state")

_METADATA_KEYS = (
    "schema_version",
    "checkpoint_type",
    "variant",
    "num_tasks",
    "tasks",
    "task_to_index",
    "models_per_seed",
    "bundle_batch_size",
    "bundles_per_task_per_batch",
    "sigreg_weight",
    "seed",
    "step",
    "episode_counts",
    "bundle_counts",
    "cache_identifier",
    "base_identifier",
    "state_keys",
)


def _validate_module_contract() -> None:
    if len(TASKS) != A4_NUM_TASKS or len(set(TASKS)) != A4_NUM_TASKS:
        raise RuntimeError("A4 checkpoint metadata requires seven distinct tasks")
    if BUNDLE_BATCH_SIZE != A4_NUM_TASKS * BUNDLES_PER_TASK_PER_BATCH:
        raise RuntimeError("A4 checkpoint batch constants are inconsistent")
    if BUNDLES_PER_TASK_PER_BATCH != 4:
        raise RuntimeError("A4 checkpoint requires four bundles per task")
    if SIGREG_WEIGHT != TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT:
        raise RuntimeError("A4 and task-context-persistent SIGReg constants disagree")


_validate_module_contract()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: Sequence[str],
    *,
    label: str,
    ordered: bool = False,
) -> None:
    actual_keys = tuple(value)
    expected_keys = tuple(expected)
    if set(actual_keys) != set(expected_keys):
        missing = [key for key in expected_keys if key not in value]
        unexpected = [key for key in actual_keys if key not in expected_keys]
        raise ValueError(
            f"{label} keys must be exact; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if ordered and actual_keys != expected_keys:
        raise ValueError(f"{label} keys must follow the frozen A4 order")


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty stable identifier")
    return value


def _frozen_integer(value: object, expected: int, *, label: str) -> int:
    result = _integer(value, label=label)
    if result != expected:
        raise ValueError(f"{label} must be exactly {expected}")
    return result


def _sigreg_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("A4 checkpoint sigreg_weight must be numeric")
    result = float(value)
    if not math.isfinite(result) or result != SIGREG_WEIGHT:
        raise ValueError(
            f"A4 checkpoint SIGReg weight must be exactly {SIGREG_WEIGHT}"
        )
    return result


def _tasks(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("A4 checkpoint tasks must be an ordered sequence")
    result = list(value)
    if result != list(TASKS):
        raise ValueError(
            "A4 checkpoint tasks must contain the complete seven-task panel "
            "in frozen order"
        )
    if any(not isinstance(task, str) for task in result):
        raise TypeError("A4 checkpoint task names must be strings")
    return result


def _task_index(value: object) -> dict[str, int]:
    mapping = _mapping(value, label="A4 checkpoint task_to_index")
    _exact_keys(
        mapping,
        TASKS,
        label="A4 checkpoint task_to_index",
        ordered=True,
    )
    result: dict[str, int] = {}
    for index, task in enumerate(TASKS):
        task_index = _integer(
            mapping[task], label=f"A4 checkpoint task_to_index[{task!r}]"
        )
        if task_index != index:
            raise ValueError(
                "A4 checkpoint task_to_index must match the frozen TASKS order"
            )
        result[task] = task_index
    return result


def _count_table(value: object, *, label: str) -> dict[str, dict[str, int]]:
    split_mapping = _mapping(value, label=label)
    _exact_keys(split_mapping, A4_SPLITS, label=label, ordered=True)
    result: dict[str, dict[str, int]] = {}
    for split in A4_SPLITS:
        task_mapping = _mapping(
            split_mapping[split], label=f"{label}[{split!r}]"
        )
        _exact_keys(
            task_mapping,
            TASKS,
            label=f"{label}[{split!r}]",
            ordered=True,
        )
        result[split] = {
            task: _integer(
                task_mapping[task],
                label=f"{label}[{split!r}][{task!r}]",
            )
            for task in TASKS
        }
    return result


def _state_keys(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("A4 checkpoint state_keys must be an ordered sequence")
    result = list(value)
    if result != list(A4_STATE_KEYS):
        raise ValueError(
            "A4 checkpoint state_keys must be exactly model_state and "
            "no_action_state"
        )
    return result


def validate_a4_checkpoint_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a canonical copy of shared A4 metadata.

    The schema is intentionally exact.  A task subset, reordered panel, one
    model per task, alternative batch shape, or alternative SIGReg value is a
    hard error rather than a recoverable compatibility mode.
    """

    values = _mapping(metadata, label="A4 checkpoint metadata")
    _exact_keys(values, _METADATA_KEYS, label="A4 checkpoint metadata")

    schema_version = _frozen_integer(
        values["schema_version"],
        A4_CHECKPOINT_SCHEMA_VERSION,
        label="A4 checkpoint schema_version",
    )
    if values["checkpoint_type"] != A4_CHECKPOINT_TYPE:
        raise ValueError(
            f"A4 checkpoint_type must be exactly {A4_CHECKPOINT_TYPE!r}"
        )
    if values["variant"] != TASK_CONTEXT_PERSISTENT_VARIANT:
        raise ValueError(
            "A4 checkpoint variant must be exactly "
            f"{TASK_CONTEXT_PERSISTENT_VARIANT!r}"
        )
    num_tasks = _frozen_integer(
        values["num_tasks"], A4_NUM_TASKS, label="A4 checkpoint num_tasks"
    )
    tasks = _tasks(values["tasks"])
    task_to_index = _task_index(values["task_to_index"])
    models_per_seed = _frozen_integer(
        values["models_per_seed"],
        A4_MODELS_PER_SEED,
        label="A4 checkpoint models_per_seed",
    )
    bundle_batch_size = _frozen_integer(
        values["bundle_batch_size"],
        BUNDLE_BATCH_SIZE,
        label="A4 checkpoint bundle_batch_size",
    )
    bundles_per_task = _frozen_integer(
        values["bundles_per_task_per_batch"],
        BUNDLES_PER_TASK_PER_BATCH,
        label="A4 checkpoint bundles_per_task_per_batch",
    )
    if bundle_batch_size != num_tasks * bundles_per_task:
        raise ValueError("A4 checkpoint batch must remain 28 = 7 x 4")
    sigreg_weight = _sigreg_weight(values["sigreg_weight"])
    seed = _integer(values["seed"], label="A4 checkpoint seed")
    step = _integer(values["step"], label="A4 checkpoint step")
    episode_counts = _count_table(
        values["episode_counts"], label="A4 checkpoint episode_counts"
    )
    bundle_counts = _count_table(
        values["bundle_counts"], label="A4 checkpoint bundle_counts"
    )
    for split in A4_SPLITS:
        for task in TASKS:
            if episode_counts[split][task] == 0 and bundle_counts[split][task] != 0:
                raise ValueError(
                    "A4 checkpoint cannot record bundles for a split/task with "
                    f"zero episodes: split={split!r}, task={task!r}"
                )

    return {
        "schema_version": schema_version,
        "checkpoint_type": A4_CHECKPOINT_TYPE,
        "variant": TASK_CONTEXT_PERSISTENT_VARIANT,
        "num_tasks": num_tasks,
        "tasks": tasks,
        "task_to_index": task_to_index,
        "models_per_seed": models_per_seed,
        "bundle_batch_size": bundle_batch_size,
        "bundles_per_task_per_batch": bundles_per_task,
        "sigreg_weight": sigreg_weight,
        "seed": seed,
        "step": step,
        "episode_counts": episode_counts,
        "bundle_counts": bundle_counts,
        "cache_identifier": _identifier(
            values["cache_identifier"], label="A4 checkpoint cache_identifier"
        ),
        "base_identifier": _identifier(
            values["base_identifier"], label="A4 checkpoint base_identifier"
        ),
        "state_keys": _state_keys(values["state_keys"]),
    }


def build_a4_checkpoint_metadata(
    *,
    seed: int,
    step: int,
    episode_counts: Mapping[str, Mapping[str, int]],
    bundle_counts: Mapping[str, Mapping[str, int]],
    cache_identifier: str,
    base_identifier: str,
    variant: str = TASK_CONTEXT_PERSISTENT_VARIANT,
    num_tasks: int = A4_NUM_TASKS,
    tasks: Sequence[str] = TASKS,
    task_to_index: Mapping[str, int] | None = None,
    models_per_seed: int = A4_MODELS_PER_SEED,
    bundle_batch_size: int = BUNDLE_BATCH_SIZE,
    bundles_per_task_per_batch: int = BUNDLES_PER_TASK_PER_BATCH,
    sigreg_weight: float = SIGREG_WEIGHT,
    state_keys: Sequence[str] = A4_STATE_KEYS,
) -> dict[str, Any]:
    """Construct metadata and immediately enforce the frozen A4 contract."""

    supplied_tasks = list(tasks) if not isinstance(tasks, (str, bytes)) else tasks
    supplied_index = (
        {task: index for index, task in enumerate(supplied_tasks)}
        if task_to_index is None and not isinstance(supplied_tasks, str)
        else task_to_index
    )
    metadata = {
        "schema_version": A4_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": A4_CHECKPOINT_TYPE,
        "variant": variant,
        "num_tasks": num_tasks,
        "tasks": supplied_tasks,
        "task_to_index": supplied_index,
        "models_per_seed": models_per_seed,
        "bundle_batch_size": bundle_batch_size,
        "bundles_per_task_per_batch": bundles_per_task_per_batch,
        "sigreg_weight": sigreg_weight,
        "seed": seed,
        "step": step,
        "episode_counts": episode_counts,
        "bundle_counts": bundle_counts,
        "cache_identifier": cache_identifier,
        "base_identifier": base_identifier,
        "state_keys": list(state_keys),
    }
    return validate_a4_checkpoint_metadata(metadata)


def _checkpoint_state(value: object, *, label: str) -> dict[str, Any]:
    state = _mapping(value, label=label)
    if not state:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(key, str) or not key for key in state):
        raise TypeError(f"{label} keys must be non-empty strings")
    if set(state) == set(TASKS):
        raise ValueError(
            f"{label} must be one shared model state, not seven task-specific models"
        )
    return dict(state)


def build_a4_checkpoint(
    metadata: Mapping[str, Any],
    *,
    model_state: Mapping[str, Any],
    no_action_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the two required shared states to validated A4 metadata."""

    result = validate_a4_checkpoint_metadata(metadata)
    result["model_state"] = _checkpoint_state(
        model_state, label="A4 checkpoint model_state"
    )
    result["no_action_state"] = _checkpoint_state(
        no_action_state, label="A4 checkpoint no_action_state"
    )
    return result


def validate_a4_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate complete metadata plus factual and matched no-action states."""

    values = _mapping(checkpoint, label="A4 checkpoint")
    _exact_keys(
        values,
        (*_METADATA_KEYS, *A4_STATE_KEYS),
        label="A4 checkpoint",
    )
    metadata = validate_a4_checkpoint_metadata(
        {key: values[key] for key in _METADATA_KEYS}
    )
    metadata["model_state"] = _checkpoint_state(
        values["model_state"], label="A4 checkpoint model_state"
    )
    metadata["no_action_state"] = _checkpoint_state(
        values["no_action_state"], label="A4 checkpoint no_action_state"
    )
    return metadata


__all__ = [
    "A4_CHECKPOINT_SCHEMA_VERSION",
    "A4_CHECKPOINT_TYPE",
    "A4_MODELS_PER_SEED",
    "A4_NUM_TASKS",
    "A4_SPLITS",
    "A4_STATE_KEYS",
    "A4_TASK_TO_INDEX",
    "TASKS",
    "build_a4_checkpoint",
    "build_a4_checkpoint_metadata",
    "validate_a4_checkpoint",
    "validate_a4_checkpoint_metadata",
]
