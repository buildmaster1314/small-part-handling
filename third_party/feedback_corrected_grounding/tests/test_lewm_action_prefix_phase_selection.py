from __future__ import annotations

import numpy as np

from fcpg.lewm_action_branch_a4 import TASKS
from fcpg.lewm_action_prefix_joint_data import (
    phase_labels_for_bundle_indices,
    select_phase_balanced_bundle_indices,
)


def _fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    cache_rows = []
    schedule_rows = []
    for task_index, task in enumerate(TASKS):
        for split, episode_count in (("train", 2), ("development", 1)):
            for episode in range(episode_count):
                instance = task_index * 10 + episode + (100 if split == "development" else 0)
                steps = [10, 20, 30, 40]
                schedule_rows.append(
                    {
                        "task": task,
                        "task_instance": instance,
                        "proposal_steps": steps,
                        "coverage": {
                            "selected_phase_names": [
                                f"phase_{index}" for index in range(len(steps))
                            ]
                        },
                    }
                )
                for step in steps:
                    cache_rows.append((task_index, task, instance, split, episode, step))
    count = len(cache_rows)
    cache = {
        "proposal_id": np.asarray([f"p{index}" for index in range(count)]),
        "task_class_index": np.asarray([row[0] for row in cache_rows]),
        "task_name": np.asarray([row[1] for row in cache_rows]),
        "task_instance_index": np.asarray([row[2] for row in cache_rows]),
        "source_episode_id": np.asarray(
            [f"{row[1]}_{row[3]}_{row[4]}" for row in cache_rows]
        ),
        "split": np.asarray([row[3] for row in cache_rows]),
    }
    ledger = {
        "proposal_id": np.array(cache["proposal_id"], copy=True),
        "accepted_bundle_index": np.arange(count, dtype=np.int64),
        "proposal_raw_step": np.asarray([row[5] for row in cache_rows]),
    }
    return cache, ledger, {"episodes": schedule_rows}


def test_phase_balanced_selection_spreads_each_episode_over_trajectory() -> None:
    cache, ledger, schedule = _fixture()

    selected = select_phase_balanced_bundle_indices(
        cache,
        ledger,
        schedule,
        train_per_task=4,
        development_per_task=2,
    )
    labels = phase_labels_for_bundle_indices(cache, ledger, selected, schedule)

    assert selected.shape == (len(TASKS) * 6,)
    for task in TASKS:
        for split, expected_count in (("train", 4), ("development", 2)):
            mask = (
                (cache["task_name"][selected] == task)
                & (cache["split"][selected] == split)
            )
            assert int(mask.sum()) == expected_count
            assert len(set(cache["source_episode_id"][selected][mask].tolist())) == (
                2 if split == "train" else 1
            )
            assert len(set(labels[mask].tolist())) >= 2


def test_phase_balanced_selection_rejects_schedule_drift() -> None:
    cache, ledger, schedule = _fixture()
    schedule["episodes"][0]["proposal_steps"][0] = 11

    try:
        select_phase_balanced_bundle_indices(
            cache,
            ledger,
            schedule,
            train_per_task=4,
            development_per_task=2,
        )
    except ValueError as exc:
        assert "absent from the phase schedule" in str(exc)
    else:
        raise AssertionError("schedule drift must be rejected")
