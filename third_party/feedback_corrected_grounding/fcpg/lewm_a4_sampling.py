"""Strict task- and episode-balanced batch sampling for Action-Branch A4."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from numbers import Integral
from typing import Any

import numpy as np
from torch.utils.data import Sampler

from fcpg.lewm_action_branch_a4 import (
    BUNDLE_BATCH_SIZE,
    BUNDLES_PER_TASK_PER_BATCH,
    TASKS,
)


def _validate_frozen_batch_contract() -> None:
    if len(TASKS) != 7 or len(set(TASKS)) != 7:
        raise RuntimeError("A4 sampling requires exactly seven distinct tasks")
    if BUNDLES_PER_TASK_PER_BATCH != 4:
        raise RuntimeError("A4 sampling requires exactly four bundles per task")
    if BUNDLE_BATCH_SIZE != len(TASKS) * BUNDLES_PER_TASK_PER_BATCH:
        raise RuntimeError("A4 batch constants are internally inconsistent")


def _integer(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    minimum = 1 if positive else 0
    if result < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _vector(value: Sequence[object] | np.ndarray, *, label: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a one-dimensional sample vector")
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a one-dimensional sample vector") from error
    if array.ndim != 1:
        raise ValueError(f"{label} must have shape [N]")
    return array


def _text(value: object, *, label: str) -> str:
    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} contains invalid UTF-8") from error
    raise TypeError(f"{label} must contain strings")


def _metadata(
    task_name: Sequence[object] | np.ndarray,
    source_episode_id: Sequence[object] | np.ndarray,
) -> tuple[tuple[str, ...], tuple[str | int, ...]]:
    task_array = _vector(task_name, label="task_name")
    episode_array = _vector(source_episode_id, label="source_episode_id")
    if task_array.shape[0] != episode_array.shape[0]:
        raise ValueError("task_name and source_episode_id must have equal length")
    if task_array.shape[0] == 0:
        raise ValueError("A4 sampling metadata must not be empty")

    tasks = tuple(
        _text(value, label="task_name") for value in task_array.tolist()
    )
    observed = set(tasks)
    expected = set(TASKS)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "task_name must cover the frozen A4 tasks exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )

    episodes_list: list[str | int] = []
    for value in episode_array.tolist():
        if isinstance(value, (str, np.str_, bytes, np.bytes_)):
            episode = _text(value, label="source_episode_id")
            if not episode.strip():
                raise ValueError("source_episode_id values must be non-empty")
            episodes_list.append(episode)
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("source_episode_id must contain strings or integers")
        else:
            episodes_list.append(int(value))
    episodes = tuple(episodes_list)
    return tasks, episodes


class A4EpisodeBalancedBatchSampler(Sampler[list[int]]):
    """Yield strict 28-sample A4 batches from one shared dataset.

    Every batch contains four samples for each task in the frozen ``TASKS``
    order. Within a task, the four samples always come from distinct source
    episodes. Episode selection is round-robin after a seeded shuffle, while
    rows within each episode are shuffled and cycled independently.

    By default, the largest task determines the number of batches in an epoch;
    smaller tasks repeat deterministically to preserve exact task balance.
    ``batches_per_epoch`` can instead set a fixed training budget.
    """

    def __init__(
        self,
        task_name: Sequence[object] | np.ndarray,
        source_episode_id: Sequence[object] | np.ndarray,
        *,
        seed: int,
        batches_per_epoch: int | None = None,
    ) -> None:
        _validate_frozen_batch_contract()
        self._seed = _integer(seed, label="seed")
        tasks, episodes = _metadata(task_name, source_episode_id)

        grouped: dict[str, dict[str | int, list[int]]] = {
            task: {} for task in TASKS
        }
        for index, (task, episode) in enumerate(zip(tasks, episodes, strict=True)):
            grouped[task].setdefault(episode, []).append(index)

        groups: list[tuple[tuple[int, ...], ...]] = []
        sample_counts: list[int] = []
        for task in TASKS:
            task_episodes = grouped[task]
            if len(task_episodes) < BUNDLES_PER_TASK_PER_BATCH:
                raise ValueError(
                    f"task {task!r} requires at least "
                    f"{BUNDLES_PER_TASK_PER_BATCH} distinct source episodes; "
                    f"got {len(task_episodes)}"
                )
            groups.append(tuple(tuple(indices) for indices in task_episodes.values()))
            sample_counts.append(
                sum(len(indices) for indices in task_episodes.values())
            )

        self._groups = tuple(groups)
        if batches_per_epoch is None:
            largest_task = max(sample_counts)
            self._batches_per_epoch = (
                largest_task + BUNDLES_PER_TASK_PER_BATCH - 1
            ) // BUNDLES_PER_TASK_PER_BATCH
        else:
            self._batches_per_epoch = _integer(
                batches_per_epoch,
                label="batches_per_epoch",
                positive=True,
            )
        self._epoch = 0

    @property
    def epoch(self) -> int:
        """Current epoch used to derive deterministic permutations."""

        return self._epoch

    @property
    def batches_per_epoch(self) -> int:
        """Number of complete A4 batches yielded per epoch."""

        return self._batches_per_epoch

    def __len__(self) -> int:
        return self._batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch-specific deterministic permutation."""

        self._epoch = _integer(epoch, label="epoch")

    def __iter__(self) -> Iterator[list[int]]:
        return self._iter_epoch(self._epoch)

    def _iter_epoch(self, epoch: int) -> Iterator[list[int]]:
        rng = random.Random(f"fcpg-a4-sampling:{self._seed}:{epoch}")

        episode_orders: list[list[int]] = []
        episode_offsets = [0 for _ in TASKS]
        sample_orders: list[list[list[int]]] = []
        sample_offsets: list[list[int]] = []
        for task_groups in self._groups:
            episode_order = list(range(len(task_groups)))
            rng.shuffle(episode_order)
            episode_orders.append(episode_order)

            task_sample_orders: list[list[int]] = []
            for indices in task_groups:
                order = list(indices)
                rng.shuffle(order)
                task_sample_orders.append(order)
            sample_orders.append(task_sample_orders)
            sample_offsets.append([0 for _ in task_groups])

        for _ in range(self._batches_per_epoch):
            batch: list[int] = []
            for task_index, task_groups in enumerate(self._groups):
                order = episode_orders[task_index]
                selected_episodes: list[int] = []
                for _ in range(BUNDLES_PER_TASK_PER_BATCH):
                    offset = episode_offsets[task_index]
                    episode_index = order[offset % len(order)]
                    episode_offsets[task_index] = offset + 1
                    selected_episodes.append(episode_index)

                    row_order = sample_orders[task_index][episode_index]
                    row_offset = sample_offsets[task_index][episode_index]
                    if row_offset == len(row_order):
                        rng.shuffle(row_order)
                        row_offset = 0
                    batch.append(row_order[row_offset])
                    sample_offsets[task_index][episode_index] = row_offset + 1

                if len(set(selected_episodes)) != BUNDLES_PER_TASK_PER_BATCH:
                    raise RuntimeError("A4 episode balancing produced a duplicate")

            if len(batch) != BUNDLE_BATCH_SIZE:
                raise RuntimeError("A4 sampler produced an invalid batch size")
            yield batch


__all__ = [
    "A4EpisodeBalancedBatchSampler",
    "BUNDLE_BATCH_SIZE",
    "BUNDLES_PER_TASK_PER_BATCH",
    "TASKS",
]
