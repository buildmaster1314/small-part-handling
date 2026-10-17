from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fcpg.lewm_a4_storage import (
    ACTION_DIM,
    LATENT_DIM,
    PROPOSAL_LEDGER_KEYS,
    STATE_CACHE_KEYS,
    STATE_SLOT_COUNT,
    load_a4_proposal_ledger,
    load_a4_state_cache,
    save_a4_proposal_ledger,
    save_a4_state_cache,
    validate_a4_ledger_cache_pair,
    validate_a4_proposal_ledger,
    validate_a4_state_cache,
)
from fcpg.lewm_action_branch_a4 import BRANCH_COUNT, HISTORY_STEPS, HORIZON, TASKS


def _ledger() -> dict[str, np.ndarray]:
    count = 4
    return {
        "proposal_id": np.asarray(["p0", "p1", "p2", "p3"]),
        "task_class_index": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "task_name": np.asarray([TASKS[0], TASKS[0], TASKS[1], TASKS[1]]),
        "task_instance_index": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "source_episode_id": np.asarray(["e0", "e0", "e1", "e1"]),
        "environment_seed": np.asarray([10, 10, 11, 11], dtype=np.int64),
        "proposal_raw_step": np.asarray([4, 9, 5, 10], dtype=np.int64),
        "split": np.asarray(["train"] * count),
        "proposal": np.ones(count, dtype=bool),
        "complete_history": np.ones(count, dtype=bool),
        "success": np.zeros(count, dtype=bool),
        "ever_success_before": np.zeros(count, dtype=bool),
        "active_stage": np.ones(count, dtype=bool),
        "expert_xyz_norm": np.full(count, 0.2, dtype=np.float32),
        "recent_action_rms": np.full(count, 0.1, dtype=np.float32),
        "workspace_and_joint_margin": np.full(count, 0.1, dtype=np.float32),
        "unexpected_collision": np.zeros(count, dtype=bool),
        "remaining_raw_steps": np.full(count, 100, dtype=np.int64),
        "eligible": np.asarray([True, False, True, False]),
        "rejection_reasons": np.asarray(["", "inactive", "", "collision"]),
        "accepted_bundle_index": np.asarray([0, -1, 1, -1], dtype=np.int64),
    }


def _cache() -> dict[str, np.ndarray]:
    count = 2
    return {
        "proposal_id": np.asarray(["p0", "p2"]),
        "task_class_index": np.asarray([0, 1], dtype=np.int64),
        "task_name": np.asarray([TASKS[0], TASKS[1]]),
        "task_instance_index": np.asarray([0, 1], dtype=np.int64),
        "source_episode_id": np.asarray(["e0", "e1"]),
        "split": np.asarray(["train", "train"]),
        "history_state": np.zeros(
            (count, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float32
        ),
        "history_action": np.zeros(
            (count, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float32
        ),
        "center_action": np.zeros((count, HORIZON, ACTION_DIM), dtype=np.float32),
        "future_action": np.zeros(
            (count, BRANCH_COUNT, HORIZON, ACTION_DIM), dtype=np.float32
        ),
        "target_state": np.zeros(
            (count, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM),
            dtype=np.float32,
        ),
        "branch_semantic_index": np.stack(
            [np.arange(BRANCH_COUNT), np.arange(BRANCH_COUNT)[::-1]]
        ).astype(np.int64),
    }


def test_valid_ledger_cache_pair_and_atomic_roundtrip(tmp_path) -> None:
    ledger = _ledger()
    cache = _cache()

    ledger_axes = validate_a4_proposal_ledger(ledger)
    cache_axes = validate_a4_state_cache(cache)
    validate_a4_ledger_cache_pair(ledger, cache)

    assert ledger_axes.proposal_count == 4
    assert ledger_axes.accepted_count == 2
    assert cache_axes.bundle_count == 2
    assert cache_axes.state_slot_count == 4

    ledger_path = save_a4_proposal_ledger(tmp_path / "ledger.npz", ledger)
    cache_path = save_a4_state_cache(tmp_path / "cache.npz", cache)
    loaded_ledger = load_a4_proposal_ledger(ledger_path)
    loaded_cache = load_a4_state_cache(cache_path)
    assert tuple(loaded_ledger) == PROPOSAL_LEDGER_KEYS
    assert tuple(loaded_cache) == STATE_CACHE_KEYS
    for key in PROPOSAL_LEDGER_KEYS:
        np.testing.assert_array_equal(loaded_ledger[key], ledger[key])
    for key in STATE_CACHE_KEYS:
        np.testing.assert_array_equal(loaded_cache[key], cache[key])


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda x: x.pop("proposal_raw_step"), "keys changed"),
        (lambda x: x.__setitem__("task_name", np.asarray([TASKS[1]] * 4)), "TASKS"),
        (lambda x: x["proposal"].__setitem__(0, False), "scheduled proposals"),
        (lambda x: x["eligible"].__setitem__(1, True), "no rejection reasons"),
        (lambda x: x["rejection_reasons"].__setitem__(1, ""), "record rejection"),
        (
            lambda x: x["accepted_bundle_index"].__setitem__(2, 4),
            "contiguous and unique",
        ),
    ],
)
def test_ledger_contract_fails_closed(mutation, match: str) -> None:
    arrays = deepcopy(_ledger())
    mutation(arrays)
    with pytest.raises((TypeError, ValueError), match=match):
        validate_a4_proposal_ledger(arrays)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda x: x.pop("center_action"), "keys changed"),
        (
            lambda x: x.__setitem__(
                "history_state", np.zeros((2, 3, 3, 192), dtype=np.float32)
            ),
            "history_state must have shape",
        ),
        (
            lambda x: x["branch_semantic_index"].__setitem__(0, np.zeros(8)),
            "permutation",
        ),
        (lambda x: x["target_state"].__setitem__((0, 0, 0, 0, 0), np.nan), "finite"),
    ],
)
def test_state_cache_contract_fails_closed(mutation, match: str) -> None:
    arrays = deepcopy(_cache())
    mutation(arrays)
    with pytest.raises((TypeError, ValueError), match=match):
        validate_a4_state_cache(arrays)


def test_pair_rejects_missing_or_reordered_accepted_bundle() -> None:
    cache = _cache()
    cache["proposal_id"] = cache["proposal_id"][::-1].copy()
    with pytest.raises(ValueError, match="proposal order"):
        validate_a4_ledger_cache_pair(_ledger(), cache)


def test_object_arrays_and_split_leakage_are_rejected() -> None:
    ledger = _ledger()
    ledger["proposal_id"] = np.asarray(["p0", "p1", "p2", "p3"], dtype=object)
    with pytest.raises(TypeError, match="strings or integers|fixed-width"):
        validate_a4_proposal_ledger(ledger)

    cache = _cache()
    cache["source_episode_id"] = np.asarray(["shared", "shared"])
    cache["task_class_index"] = np.asarray([0, 0], dtype=np.int64)
    cache["task_name"] = np.asarray([TASKS[0], TASKS[0]])
    cache["split"] = np.asarray(["train", "development"])
    with pytest.raises(ValueError, match="multiple splits"):
        validate_a4_state_cache(cache)
