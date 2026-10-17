"""Compact storage contracts for the seven-task A4 experiment.

The proposal ledger records every pre-scheduled decision before branch
outcomes are known.  The state cache contains only accepted proposals encoded
by the frozen four-slot A4 state encoder.  Raw images are intentionally absent:
A4 trains temporal dynamics on a frozen visual encoder rather than updating
the encoder from branch frames.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    HORIZON,
    HISTORY_STEPS,
    TASKS,
)
from fcpg.lewm_task_context_persistent import (
    TASK_CONTEXT_PERSISTENT_STATE_SLOTS,
)


ACTION_DIM = 20
LATENT_DIM = 192
STATE_SLOT_COUNT = len(TASK_CONTEXT_PERSISTENT_STATE_SLOTS)
VALID_SPLITS = ("train", "development", "test")

PROPOSAL_LEDGER_KEYS = (
    "proposal_id",
    "task_class_index",
    "task_name",
    "task_instance_index",
    "source_episode_id",
    "environment_seed",
    "proposal_raw_step",
    "split",
    "proposal",
    "complete_history",
    "success",
    "ever_success_before",
    "active_stage",
    "expert_xyz_norm",
    "recent_action_rms",
    "workspace_and_joint_margin",
    "unexpected_collision",
    "remaining_raw_steps",
    "eligible",
    "rejection_reasons",
    "accepted_bundle_index",
)

STATE_CACHE_KEYS = (
    "proposal_id",
    "task_class_index",
    "task_name",
    "task_instance_index",
    "source_episode_id",
    "split",
    "history_state",
    "history_action",
    "center_action",
    "future_action",
    "target_state",
    "branch_semantic_index",
)


@dataclass(frozen=True, slots=True)
class A4ProposalLedgerAxes:
    proposal_count: int
    accepted_count: int


@dataclass(frozen=True, slots=True)
class A4StateCacheAxes:
    bundle_count: int
    history_steps: int
    branch_count: int
    horizon: int
    state_slot_count: int
    latent_dim: int


def _exact_arrays(
    arrays: Mapping[str, np.ndarray], expected: tuple[str, ...], label: str
) -> None:
    if not isinstance(arrays, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = set(arrays)
    required = set(expected)
    if actual != required:
        raise ValueError(
            f"{label} keys changed: missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )
    if any(not isinstance(value, np.ndarray) for value in arrays.values()):
        raise TypeError(f"every {label} value must already be a numpy ndarray")


def _vector(array: np.ndarray, count: int, name: str) -> None:
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")


def _integer_vector(array: np.ndarray, count: int, name: str) -> np.ndarray:
    _vector(array, count, name)
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integers")
    return array.astype(np.int64, copy=False)


def _bool_vector(array: np.ndarray, count: int, name: str) -> np.ndarray:
    _vector(array, count, name)
    if array.dtype.kind != "b":
        raise TypeError(f"{name} must contain booleans")
    return array.astype(bool, copy=False)


def _text_vector(array: np.ndarray, count: int, name: str) -> np.ndarray:
    _vector(array, count, name)
    if array.dtype.kind not in {"U", "S"}:
        raise TypeError(f"{name} must contain fixed-width strings")
    values = array.astype(str, copy=False)
    if name != "rejection_reasons" and any(not value.strip() for value in values):
        raise ValueError(f"{name} must not contain empty strings")
    return values


def _identifier_vector(array: np.ndarray, count: int, name: str) -> np.ndarray:
    _vector(array, count, name)
    if array.dtype.kind in {"i", "u"}:
        return array
    return _text_vector(array, count, name)


def _finite_float_vector(array: np.ndarray, count: int, name: str) -> np.ndarray:
    _vector(array, count, name)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must use a floating dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _split_vector(array: np.ndarray, count: int) -> np.ndarray:
    values = _text_vector(array, count, "split")
    unknown = sorted(set(values) - set(VALID_SPLITS))
    if unknown:
        raise ValueError(f"split contains unsupported labels: {unknown}")
    return values


def _validate_task_identity(
    task_class_index: np.ndarray,
    task_name: np.ndarray,
    task_instance_index: np.ndarray,
    count: int,
) -> None:
    class_index = _integer_vector(
        task_class_index, count, "task_class_index"
    )
    names = _text_vector(task_name, count, "task_name")
    instances = _integer_vector(
        task_instance_index, count, "task_instance_index"
    )
    if not np.logical_and(class_index >= 0, class_index < len(TASKS)).all():
        raise ValueError("task_class_index must be in the inclusive range 0..6")
    expected_names = np.asarray(TASKS, dtype=str)[class_index]
    if not np.array_equal(names, expected_names):
        raise ValueError("task_name must equal TASKS[task_class_index]")
    if not np.logical_and(instances >= 0, instances < 48).all():
        raise ValueError("task_instance_index must be in the inclusive range 0..47")


def _validate_episode_split_isolation(
    task_class_index: np.ndarray,
    source_episode_id: np.ndarray,
    split: np.ndarray,
) -> None:
    seen: dict[tuple[int, str], str] = {}
    for task, episode, label in zip(
        task_class_index, source_episode_id, split, strict=True
    ):
        key = (int(task), str(episode))
        previous = seen.setdefault(key, str(label))
        if previous != str(label):
            raise ValueError(
                "one task/source episode appears in multiple splits: "
                f"task_class_index={key[0]}, source_episode_id={key[1]!r}"
            )


def validate_a4_proposal_ledger(
    arrays: Mapping[str, np.ndarray],
) -> A4ProposalLedgerAxes:
    """Validate the complete pre-outcome proposal ledger."""

    _exact_arrays(arrays, PROPOSAL_LEDGER_KEYS, "A4 proposal ledger")
    proposal_id = arrays["proposal_id"]
    if proposal_id.ndim != 1 or proposal_id.shape[0] <= 0:
        raise ValueError("proposal_id must have non-empty shape [M]")
    count = int(proposal_id.shape[0])
    proposal_ids = _identifier_vector(proposal_id, count, "proposal_id")
    if len({str(value) for value in proposal_ids}) != count:
        raise ValueError("proposal_id values must be unique")

    _validate_task_identity(
        arrays["task_class_index"],
        arrays["task_name"],
        arrays["task_instance_index"],
        count,
    )
    source_episode = _identifier_vector(
        arrays["source_episode_id"], count, "source_episode_id"
    )
    _integer_vector(arrays["environment_seed"], count, "environment_seed")
    proposal_step = _integer_vector(
        arrays["proposal_raw_step"], count, "proposal_raw_step"
    )
    if bool(np.any(proposal_step < 0)):
        raise ValueError("proposal_raw_step must be non-negative")
    split = _split_vector(arrays["split"], count)

    proposal = _bool_vector(arrays["proposal"], count, "proposal")
    if not bool(np.all(proposal)):
        raise ValueError("the proposal ledger may contain only scheduled proposals")
    for name in (
        "complete_history",
        "success",
        "ever_success_before",
        "active_stage",
        "unexpected_collision",
        "eligible",
    ):
        _bool_vector(arrays[name], count, name)
    for name in (
        "expert_xyz_norm",
        "recent_action_rms",
        "workspace_and_joint_margin",
    ):
        values = _finite_float_vector(arrays[name], count, name)
        if bool(np.any(values < 0.0)):
            raise ValueError(f"{name} must be non-negative")
    remaining = _integer_vector(
        arrays["remaining_raw_steps"], count, "remaining_raw_steps"
    )
    if bool(np.any(remaining < 0)):
        raise ValueError("remaining_raw_steps must be non-negative")

    eligible = arrays["eligible"].astype(bool, copy=False)
    reasons = _text_vector(
        arrays["rejection_reasons"], count, "rejection_reasons"
    )
    accepted = _integer_vector(
        arrays["accepted_bundle_index"], count, "accepted_bundle_index"
    )
    if bool(np.any(eligible & (reasons != ""))):
        raise ValueError("eligible proposals must have no rejection reasons")
    if bool(np.any((~eligible) & (reasons == ""))):
        raise ValueError("rejected proposals must record rejection reasons")
    if bool(np.any(eligible & (accepted < 0))):
        raise ValueError("eligible proposals require an accepted bundle index")
    if bool(np.any((~eligible) & (accepted != -1))):
        raise ValueError("rejected proposals must use accepted_bundle_index=-1")
    accepted_values = accepted[eligible]
    if not np.array_equal(
        np.sort(accepted_values), np.arange(accepted_values.size, dtype=np.int64)
    ):
        raise ValueError("accepted bundle indices must be contiguous and unique")

    _validate_episode_split_isolation(
        arrays["task_class_index"], source_episode, split
    )
    return A4ProposalLedgerAxes(count, int(eligible.sum()))


def validate_a4_state_cache(
    arrays: Mapping[str, np.ndarray],
) -> A4StateCacheAxes:
    """Validate the frozen-encoder four-slot branch cache."""

    _exact_arrays(arrays, STATE_CACHE_KEYS, "A4 state cache")
    history_state = arrays["history_state"]
    if history_state.ndim != 4 or history_state.shape[0] <= 0:
        raise ValueError("history_state must have non-empty shape [N,3,4,192]")
    count = int(history_state.shape[0])
    expected_shapes = {
        "history_state": (
            count,
            HISTORY_STEPS,
            STATE_SLOT_COUNT,
            LATENT_DIM,
        ),
        "history_action": (count, HISTORY_STEPS - 1, ACTION_DIM),
        "center_action": (count, HORIZON, ACTION_DIM),
        "future_action": (count, BRANCH_COUNT, HORIZON, ACTION_DIM),
        "target_state": (
            count,
            BRANCH_COUNT,
            HORIZON,
            STATE_SLOT_COUNT,
            LATENT_DIM,
        ),
        "branch_semantic_index": (count, BRANCH_COUNT),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in (
        "history_state",
        "history_action",
        "center_action",
        "future_action",
        "target_state",
    ):
        value = arrays[name]
        if value.dtype.kind != "f":
            raise TypeError(f"{name} must use a floating dtype")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")

    proposal_id = _identifier_vector(arrays["proposal_id"], count, "proposal_id")
    if len({str(value) for value in proposal_id}) != count:
        raise ValueError("state-cache proposal_id values must be unique")
    _validate_task_identity(
        arrays["task_class_index"],
        arrays["task_name"],
        arrays["task_instance_index"],
        count,
    )
    source_episode = _identifier_vector(
        arrays["source_episode_id"], count, "source_episode_id"
    )
    split = _split_vector(arrays["split"], count)
    _validate_episode_split_isolation(
        arrays["task_class_index"], source_episode, split
    )

    semantic = _integer_vector(
        arrays["branch_semantic_index"].reshape(-1),
        count * BRANCH_COUNT,
        "branch_semantic_index",
    ).reshape(count, BRANCH_COUNT)
    expected = np.arange(BRANCH_COUNT, dtype=np.int64)
    if any(not np.array_equal(np.sort(row), expected) for row in semantic):
        raise ValueError("each branch_semantic_index row must be a permutation 0..7")

    return A4StateCacheAxes(
        bundle_count=count,
        history_steps=HISTORY_STEPS,
        branch_count=BRANCH_COUNT,
        horizon=HORIZON,
        state_slot_count=STATE_SLOT_COUNT,
        latent_dim=LATENT_DIM,
    )


def validate_a4_ledger_cache_pair(
    ledger: Mapping[str, np.ndarray], cache: Mapping[str, np.ndarray]
) -> None:
    """Require one state-cache row for every and only eligible proposal."""

    ledger_axes = validate_a4_proposal_ledger(ledger)
    cache_axes = validate_a4_state_cache(cache)
    if ledger_axes.accepted_count != cache_axes.bundle_count:
        raise ValueError("eligible proposal count does not match state-cache rows")
    eligible = ledger["eligible"].astype(bool, copy=False)
    bundle_index = ledger["accepted_bundle_index"][eligible].astype(np.int64)
    ordered_ids = ledger["proposal_id"][eligible][np.argsort(bundle_index)].astype(str)
    if not np.array_equal(ordered_ids, cache["proposal_id"].astype(str)):
        raise ValueError("state-cache proposal order does not match the ledger")


def _save_npz(
    path: str | os.PathLike[str],
    arrays: Mapping[str, np.ndarray],
    keys: tuple[str, ...],
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **{name: arrays[name] for name in keys})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_a4_proposal_ledger(
    path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]
) -> Path:
    validate_a4_proposal_ledger(arrays)
    return _save_npz(path, arrays, PROPOSAL_LEDGER_KEYS)


def save_a4_state_cache(
    path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]
) -> Path:
    validate_a4_state_cache(arrays)
    return _save_npz(path, arrays, STATE_CACHE_KEYS)


def _load_npz(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def load_a4_proposal_ledger(
    path: str | os.PathLike[str],
) -> dict[str, np.ndarray]:
    arrays = _load_npz(path)
    validate_a4_proposal_ledger(arrays)
    return arrays


def load_a4_state_cache(
    path: str | os.PathLike[str],
) -> dict[str, np.ndarray]:
    arrays = _load_npz(path)
    validate_a4_state_cache(arrays)
    return arrays


__all__ = [
    "A4ProposalLedgerAxes",
    "A4StateCacheAxes",
    "ACTION_DIM",
    "LATENT_DIM",
    "PROPOSAL_LEDGER_KEYS",
    "STATE_CACHE_KEYS",
    "STATE_SLOT_COUNT",
    "load_a4_proposal_ledger",
    "load_a4_state_cache",
    "save_a4_proposal_ledger",
    "save_a4_state_cache",
    "validate_a4_ledger_cache_pair",
    "validate_a4_proposal_ledger",
    "validate_a4_state_cache",
]
