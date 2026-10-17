from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from fcpg.lewm_a4_sampling import A4EpisodeBalancedBatchSampler
from fcpg.lewm_action_branch_a4 import (
    BUNDLE_BATCH_SIZE,
    BUNDLES_PER_TASK_PER_BATCH,
    TASKS,
)


def _metadata(
    *,
    episode_count: int = 6,
    rows_per_episode: int = 2,
) -> tuple[list[str], list[str]]:
    rows = [
        (task, f"{task}:episode-{episode}")
        for task in TASKS
        for episode in range(episode_count)
        for _ in range(rows_per_episode)
    ]
    rows.reverse()
    return [row[0] for row in rows], [row[1] for row in rows]


def _assert_strict_batch(
    batch: list[int],
    task_name: list[str],
    source_episode_id: list[str],
) -> None:
    assert len(batch) == BUNDLE_BATCH_SIZE
    assert [task_name[index] for index in batch] == [
        task
        for task in TASKS
        for _ in range(BUNDLES_PER_TASK_PER_BATCH)
    ]
    for task_offset in range(len(TASKS)):
        start = task_offset * BUNDLES_PER_TASK_PER_BATCH
        task_indices = batch[start : start + BUNDLES_PER_TASK_PER_BATCH]
        assert len({source_episode_id[index] for index in task_indices}) == 4


def test_batches_use_frozen_task_order_and_four_distinct_episodes() -> None:
    task_name, source_episode_id = _metadata()
    sampler = A4EpisodeBalancedBatchSampler(
        task_name,
        source_episode_id,
        seed=17,
    )

    assert len(sampler) == 3
    assert sampler.batches_per_epoch == 3
    for batch in sampler:
        _assert_strict_batch(batch, task_name, source_episode_id)


def test_duplicate_episode_rows_are_balanced_and_repeat_deterministically() -> None:
    task_name, source_episode_id = _metadata(
        episode_count=5,
        rows_per_episode=2,
    )
    sampler = A4EpisodeBalancedBatchSampler(
        task_name,
        source_episode_id,
        seed=29,
        batches_per_epoch=4,
    )

    batches = list(sampler)
    assert batches == list(sampler)
    for batch in batches:
        _assert_strict_batch(batch, task_name, source_episode_id)

    for task in TASKS:
        selected = [
            index
            for batch in batches
            for index in batch
            if task_name[index] == task
        ]
        assert len(selected) == 16
        assert len(set(selected)) == 10
        episode_counts = Counter(source_episode_id[index] for index in selected)
        assert max(episode_counts.values()) - min(episode_counts.values()) <= 1


def test_seed_reproduction_and_epoch_changes_are_stateless() -> None:
    task_name, source_episode_id = _metadata(episode_count=8)
    first = A4EpisodeBalancedBatchSampler(
        task_name,
        source_episode_id,
        seed=101,
        batches_per_epoch=5,
    )
    second = A4EpisodeBalancedBatchSampler(
        np.asarray(task_name, dtype=object),
        np.asarray(source_episode_id, dtype=object),
        seed=101,
        batches_per_epoch=5,
    )

    epoch_zero = list(first)
    assert epoch_zero == list(first) == list(second)

    first.set_epoch(1)
    second.set_epoch(1)
    epoch_one = list(first)
    assert first.epoch == 1
    assert epoch_one == list(second)
    assert epoch_one != epoch_zero

    first.set_epoch(0)
    assert list(first) == epoch_zero


def test_missing_task_and_insufficient_episode_coverage_fail_closed() -> None:
    task_name, source_episode_id = _metadata()
    keep = [index for index, task in enumerate(task_name) if task != TASKS[-1]]
    with pytest.raises(ValueError, match="cover the frozen A4 tasks exactly"):
        A4EpisodeBalancedBatchSampler(
            [task_name[index] for index in keep],
            [source_episode_id[index] for index in keep],
            seed=1,
        )

    task_name, source_episode_id = _metadata(episode_count=3)
    with pytest.raises(ValueError, match="at least 4 distinct source episodes"):
        A4EpisodeBalancedBatchSampler(task_name, source_episode_id, seed=1)


def test_invalid_metadata_and_control_values_are_rejected() -> None:
    task_name, source_episode_id = _metadata()

    with pytest.raises(ValueError, match="equal length"):
        A4EpisodeBalancedBatchSampler(task_name[:-1], source_episode_id, seed=1)
    with pytest.raises(TypeError, match="task_name must contain strings"):
        A4EpisodeBalancedBatchSampler(
            list(range(len(task_name))), source_episode_id, seed=1
        )
    invalid_episode = list(source_episode_id)
    invalid_episode[0] = ""
    with pytest.raises(ValueError, match="must be non-empty"):
        A4EpisodeBalancedBatchSampler(task_name, invalid_episode, seed=1)
    with pytest.raises(TypeError, match="strings or integers"):
        A4EpisodeBalancedBatchSampler(
            task_name,
            np.arange(len(task_name), dtype=np.float64),
            seed=1,
        )
    with pytest.raises(TypeError, match="seed must be an integer"):
        A4EpisodeBalancedBatchSampler(task_name, source_episode_id, seed=True)
    with pytest.raises(ValueError, match="batches_per_epoch must be positive"):
        A4EpisodeBalancedBatchSampler(
            task_name,
            source_episode_id,
            seed=1,
            batches_per_epoch=0,
        )

    sampler = A4EpisodeBalancedBatchSampler(task_name, source_episode_id, seed=1)
    with pytest.raises(ValueError, match="epoch must be non-negative"):
        sampler.set_epoch(-1)
    with pytest.raises(TypeError, match="epoch must be an integer"):
        sampler.set_epoch(False)
