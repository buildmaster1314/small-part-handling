from __future__ import annotations

import json

import numpy as np
import pytest

from fcpg.lewm_action_branch_a2_confirmation import (
    TASKS,
    assign_confirmation_splits,
    confirmation_support,
    validate_confirmation_contract,
)
from scripts.collect_lewm_action_branch_a2_confirm import DEFAULT_CONFIG


def _contract() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_confirmation_split_is_24_train_24_test_per_task() -> None:
    contract = _contract()
    assert validate_confirmation_contract(contract) == TASKS
    assignments = assign_confirmation_splits(contract)
    for task in TASKS:
        labels = [assignments[(task, instance)] for instance in range(48)]
        assert labels.count("train") == 24
        assert labels.count("test") == 24


def test_confirmation_support_uses_h6_h8_and_all_tasks() -> None:
    rows = 4 * 24
    target = np.zeros((rows, 8, 8, 1, 4), dtype=np.float32)
    task_index = np.repeat(np.arange(4), 24)
    episodes = np.asarray(
        [f"{TASKS[task]}:episode-{episode}" for task in range(4) for episode in range(24)]
    )
    branch = np.arange(8, dtype=np.float32)[:, None, None, None]
    target[:, :, 5:8] = 0.2 * branch[:, :3]
    arrays = {
        "target_embedding": target,
        "task_index": task_index,
        "task_instance_index": np.tile(np.arange(24, dtype=np.int64), 4),
        "source_episode_id": episodes,
        "split": np.full(rows, "train"),
    }
    summary, moments = confirmation_support(arrays)
    assert summary["decision"] == "Go"
    assert moments.count > 0
    assert set(summary["per_task"]) == set(TASKS)


def test_confirmation_contract_locks_protocol_seed() -> None:
    contract = _contract()
    contract["split"]["protocol_seed"] = 999

    with pytest.raises(ValueError, match="seed 27103"):
        validate_confirmation_contract(contract)


def test_confirmation_support_rejects_non_train_rows() -> None:
    rows = 4 * 24
    arrays = {
        "target_embedding": np.zeros((rows, 8, 8, 1, 4), dtype=np.float32),
        "task_index": np.repeat(np.arange(4), 24),
        "task_instance_index": np.tile(np.arange(24, dtype=np.int64), 4),
        "source_episode_id": np.asarray(
            [
                f"{TASKS[task]}:episode-{episode}"
                for task in range(4)
                for episode in range(24)
            ]
        ),
        "split": np.full(rows, "test"),
    }

    with pytest.raises(ValueError, match="train rows only"):
        confirmation_support(arrays)


def test_confirmation_support_binds_task_index_to_episode_prefix() -> None:
    rows = 4 * 24
    target = np.zeros((rows, 8, 8, 1, 4), dtype=np.float32)
    target[:, :, 5:8] = 0.2 * np.arange(8, dtype=np.float32)[:, None, None, None]
    episodes = np.asarray(
        [
            f"{TASKS[task]}:episode-{episode}"
            for task in range(4)
            for episode in range(24)
        ]
    )
    episodes[:24] = np.char.replace(episodes[:24], TASKS[0], TASKS[1])
    arrays = {
        "target_embedding": target,
        "task_index": np.repeat(np.arange(4), 24),
        "task_instance_index": np.tile(np.arange(24, dtype=np.int64), 4),
        "source_episode_id": episodes,
        "split": np.full(rows, "train"),
    }

    with pytest.raises(ValueError, match="not bound"):
        confirmation_support(arrays)
