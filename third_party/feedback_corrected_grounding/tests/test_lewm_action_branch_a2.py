from __future__ import annotations

import numpy as np
import pytest

from fcpg.lewm_action_branch_a2 import (
    PROTOCOL,
    assign_a2_splits,
    summarize_intervention_support,
    validate_a2_contract,
)


def _contract() -> dict:
    return {
        "protocol": PROTOCOL,
        "tasks": ["pick-place-v3", "drawer-open-v3", "reach-v3", "push-v3"],
        "split": {
            "protocol_seed": 25701,
            "train_episodes_per_task": 24,
            "development_episodes_per_task": 12,
            "test_episodes_per_task": 12,
        },
        "collection": {
            "horizon_steps": 8,
            "test_instances_not_collected_by_this_protocol": True,
        },
    }


def test_contract_requires_drawer_and_excludes_button() -> None:
    assert validate_a2_contract(_contract())[1] == "drawer-open-v3"
    contract = _contract()
    contract["tasks"][1] = "button-press-v3"
    with pytest.raises(ValueError, match="replace Button"):
        validate_a2_contract(contract)
def test_split_assignment_is_complete_and_task_local() -> None:
    contract = _contract()
    tasks = validate_a2_contract(contract)
    assignment = assign_a2_splits(tasks, contract)
    assert len(assignment) == 4 * 48
    for task in tasks:
        labels = [assignment[(task, instance)] for instance in range(48)]
        assert labels.count("train") == 24
        assert labels.count("development") == 12
        assert labels.count("test") == 12


def _support_arrays(*, weak_task: int | None = None) -> dict[str, np.ndarray]:
    tasks, train_episodes, dev_episodes = 4, 24, 12
    rows = tasks * (train_episodes + dev_episodes)
    target = np.zeros((rows, 8, 8, 1, 4), dtype=np.float32)
    split: list[str] = []
    task_index: list[int] = []
    episode: list[str] = []
    rng = np.random.default_rng(4)
    cursor = 0
    for task in range(tasks):
        for local in range(train_episodes + dev_episodes):
            label = "train" if local < train_episodes else "development"
            split.append(label)
            task_index.append(task)
            episode.append(f"task-{task}-episode-{local}")
            scale = 0.02 if task != weak_task or label == "train" else 1e-5
            branch = np.arange(8, dtype=np.float32)[:, None, None, None]
            target[cursor] = scale * branch + rng.normal(
                0.0, 1e-4, size=target[cursor].shape
            )
            cursor += 1
    return {
        "target_embedding": target,
        "split": np.asarray(split),
        "task_index": np.asarray(task_index, dtype=np.int64),
        "source_episode_id": np.asarray(episode),
    }


def test_support_gate_requires_every_task_and_twelve_episodes() -> None:
    tasks = ("pick", "drawer", "reach", "push")
    go = summarize_intervention_support(
        _support_arrays(), task_names=tasks, minimum_branch_variance=1e-4
    )
    assert go["decision"] == "Go"
    no_go = summarize_intervention_support(
        _support_arrays(weak_task=1),
        task_names=tasks,
        minimum_branch_variance=1e-4,
    )
    assert no_go["decision"] == "No-Go"
    assert not no_go["per_task"]["drawer"]["passed"]


def test_late_horizon_support_can_differ_from_full_horizon() -> None:
    arrays = _support_arrays()
    development = arrays["split"].astype(str) == "development"
    drawer = arrays["task_index"] == 1
    selected = development & drawer
    arrays["target_embedding"][selected, :, :5] = 0.0

    full = summarize_intervention_support(
        arrays,
        task_names=("pick", "drawer", "reach", "push"),
        minimum_branch_variance=1e-4,
    )
    late = summarize_intervention_support(
        arrays,
        task_names=("pick", "drawer", "reach", "push"),
        primary_horizons=(6, 7, 8),
        minimum_branch_variance=1e-4,
    )
    assert full["per_task"]["drawer"]["median_branch_variance"] < late["per_task"][
        "drawer"
    ]["median_branch_variance"]
    assert late["primary_horizons"] == [6, 7, 8]


def test_support_gate_refuses_test_rows() -> None:
    arrays = _support_arrays()
    arrays["split"][0] = "test"
    with pytest.raises(ValueError, match="test rows"):
        summarize_intervention_support(
            arrays,
            task_names=("pick", "drawer", "reach", "push"),
        )
