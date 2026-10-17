"""Fresh continuous-task action-branch screen for A2.

A2 keeps A1's state-independent random-hazard intervention and replaces the
near-absorbing Button task with Drawer Open.  This module only collects smoke
or train/development rows; reserved test instances are not reset or encoded.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_collection import (
    FORMAL_TASK_INSTANCES,
    SMOKE_TASK_INSTANCES,
    CollectionPayload,
    EpisodeRuntime,
    collect_source_episode,
    stack_action_branch_rows,
)
from fcpg.lewm_action_branch_metrics import center_whitened_branches, fit_train_moments


PROTOCOL = "fcpg_action_branch_continuous_task_a2_v1"


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def validate_a2_contract(contract: Mapping[str, Any]) -> tuple[str, ...]:
    if contract.get("protocol") != PROTOCOL:
        raise ValueError(f"A2 protocol must be {PROTOCOL!r}")
    tasks = tuple(str(value) for value in contract.get("tasks", ()))
    if len(tasks) != 4 or len(set(tasks)) != 4:
        raise ValueError("A2 requires exactly four distinct tasks")
    if "button-press-v3" in tasks or "drawer-open-v3" not in tasks:
        raise ValueError("A2 must replace Button with Drawer Open")
    split = _mapping(contract.get("split"), "split")
    counts = (
        int(split["train_episodes_per_task"]),
        int(split["development_episodes_per_task"]),
        int(split["test_episodes_per_task"]),
    )
    if counts != (24, 12, 12):
        raise ValueError("A2 split must remain 24 train / 12 development / 12 test")
    collection = _mapping(contract.get("collection"), "collection")
    if int(collection["horizon_steps"]) != 8:
        raise ValueError("A2 horizon must remain H=8")
    if not bool(collection.get("test_instances_not_collected_by_this_protocol", False)):
        raise ValueError("A2 collection must leave test instances untouched")
    return tasks


def assign_a2_splits(
    tasks: Sequence[str], contract: Mapping[str, Any]
) -> dict[tuple[str, int], str]:
    """Assign the frozen 24/12/12 task-local split."""

    protocol_seed = int(_mapping(contract.get("split"), "split")["protocol_seed"])
    result: dict[tuple[str, int], str] = {}
    counts = (("train", 24), ("development", 12), ("test", 12))
    for task in tasks:
        generator = np.random.default_rng(
            np.random.SeedSequence([protocol_seed, zlib.crc32(task.encode("utf-8"))])
        )
        ranked = generator.permutation(FORMAL_TASK_INSTANCES).tolist()
        offset = 0
        for label, count in counts:
            for instance in ranked[offset : offset + count]:
                result[(str(task), int(instance))] = label
            offset += count
    return result


def run_a2_collection(
    contract: Mapping[str, Any],
    *,
    encoder: Any,
    runtime_factory: Callable[[str, int, int], EpisodeRuntime],
    smoke: bool,
    max_tasks: int | None = None,
    max_episodes_per_task: int | None = None,
    max_events_per_episode: int | None = None,
) -> CollectionPayload:
    """Collect A2 smoke or train/development rows without touching test."""

    configured_tasks = validate_a2_contract(contract)
    if max_tasks is not None:
        if not smoke or int(max_tasks) <= 0:
            raise ValueError("max_tasks is a positive smoke-only bound")
        tasks = configured_tasks[: int(max_tasks)]
    else:
        tasks = configured_tasks
    collection = _mapping(contract.get("collection"), "collection")
    horizon_steps = int(collection["horizon_steps"])
    environment_seed_base = int(collection["environment_seed_base"])
    assignments = assign_a2_splits(configured_tasks, contract)

    rows = []
    for task in tasks:
        task_index = configured_tasks.index(task)
        if smoke:
            instances = SMOKE_TASK_INSTANCES
            if max_episodes_per_task is not None:
                if int(max_episodes_per_task) <= 0:
                    raise ValueError("max_episodes_per_task must be positive")
                instances = instances[: int(max_episodes_per_task)]
        else:
            if max_episodes_per_task is not None or max_events_per_episode is not None:
                raise ValueError("episode/event limits are smoke-only")
            instances = tuple(
                instance
                for instance in FORMAL_TASK_INSTANCES
                if assignments[(task, int(instance))] in {"train", "development"}
            )
        for instance in instances:
            split = "smoke" if smoke else assignments[(task, int(instance))]
            environment_seed = (
                environment_seed_base
                + (1_000_000 if smoke else 0)
                + task_index * 10_000
                + int(instance)
            )
            rows.extend(
                collect_source_episode(
                    task=task,
                    task_seed_index=task_index,
                    task_index_mod50=int(instance),
                    environment_seed=environment_seed,
                    split=split,
                    contract=contract,
                    horizon_steps=horizon_steps,
                    runtime_factory=runtime_factory,
                    encode_records=encoder.encode,
                    max_events=max_events_per_episode,
                )
            )
    arrays = stack_action_branch_rows(rows)
    if np.any(arrays["split"].astype(str) == "test"):
        raise RuntimeError("A2 screen unexpectedly collected test rows")
    return CollectionPayload(
        arrays=arrays,
        metadata={
            "protocol": PROTOCOL,
            "smoke": bool(smoke),
            "task_count": len(tasks),
            "horizon_steps": horizon_steps,
        },
    )


def summarize_intervention_support(
    arrays: Mapping[str, np.ndarray],
    *,
    task_names: Sequence[str],
    primary_horizons: Sequence[int] | None = None,
    minimum_branch_variance: float = 0.01,
    minimum_episode_macro_fraction_per_task: float = 0.8,
    minimum_development_episodes_per_task: int = 12,
) -> dict[str, Any]:
    """Apply the collection-only A2 Gate with physical episodes weighted equally."""

    split = np.asarray(arrays["split"]).astype(str)
    if np.any(split == "test"):
        raise ValueError("A2 support Gate refuses arrays containing test rows")
    train = split == "train"
    development = split == "development"
    if not np.any(train) or not np.any(development):
        raise ValueError("A2 support Gate requires train and development rows")
    target = np.asarray(arrays["target_embedding"], dtype=np.float32)
    if target.ndim != 5:
        raise ValueError("A2 target_embedding must have shape [N,C,H,K,D]")
    if primary_horizons is None:
        horizon_indices = np.arange(target.shape[2], dtype=np.int64)
    else:
        horizons = tuple(int(value) for value in primary_horizons)
        if not horizons or len(set(horizons)) != len(horizons):
            raise ValueError("primary_horizons must be distinct and non-empty")
        if any(value < 1 or value > target.shape[2] for value in horizons):
            raise ValueError("primary_horizons are outside the cached horizon")
        horizon_indices = np.asarray([value - 1 for value in horizons], dtype=np.int64)
    task_values = np.asarray(arrays["task_index"], dtype=np.int64)
    expected_task_indices = set(range(len(task_names)))
    observed_task_indices = set(task_values.tolist())
    if observed_task_indices != expected_task_indices:
        raise ValueError(
            f"A2 task_index must cover {sorted(expected_task_indices)}, "
            f"got {sorted(observed_task_indices)}"
        )
    moments = fit_train_moments(target[train])
    centered = center_whitened_branches(target[development], moments)
    centered = centered[:, :, horizon_indices]
    variance = np.mean(np.square(centered), axis=(1, 2, 3, 4), dtype=np.float64)
    task_index = task_values[development]
    episode = np.asarray(arrays["source_episode_id"])[development].astype(str)

    all_episode = np.asarray(arrays["source_episode_id"]).astype(str)
    for index, task in enumerate(task_names):
        train_episode_count = int(
            np.unique(all_episode[train & (task_values == index)]).size
        )
        development_episode_count = int(
            np.unique(all_episode[development & (task_values == index)]).size
        )
        if train_episode_count != 24 or development_episode_count != 12:
            raise ValueError(
                f"A2 {task} requires exactly 24 train and 12 development "
                f"episodes, got {train_episode_count}/{development_episode_count}"
            )

    per_task: dict[str, Any] = {}
    for index, task in enumerate(task_names):
        task_mask = task_index == index
        selected = variance[task_mask]
        selected_episode = episode[task_mask]
        if selected.size == 0:
            raise ValueError(f"A2 development has no rows for {task}")
        informative = selected >= float(minimum_branch_variance)
        identities = np.unique(selected_episode)
        episode_fractions = np.asarray(
            [
                np.mean(informative[selected_episode == identity])
                for identity in identities
            ],
            dtype=np.float64,
        )
        episode_macro = float(np.mean(episode_fractions))
        passed = (
            int(episode_fractions.size) == int(minimum_development_episodes_per_task)
            and episode_macro >= float(minimum_episode_macro_fraction_per_task)
        )
        per_task[str(task)] = {
            "bundle_count": int(selected.size),
            "informative_bundle_count": int(np.count_nonzero(informative)),
            "bundle_fraction": float(np.mean(informative)),
            "development_episode_count": int(episode_fractions.size),
            "episode_macro_informative_fraction": episode_macro,
            "median_branch_variance": float(np.median(selected)),
            "passed": bool(passed),
        }
    passed = all(bool(row["passed"]) for row in per_task.values())
    return {
        "protocol": PROTOCOL,
        "stage": "collection_only_intervention_support",
        "decision": "Go" if passed else "No-Go",
        "model_training_started": False,
        "input_contains_test_rows": False,
        "primary_horizons": [int(value) + 1 for value in horizon_indices.tolist()],
        "minimum_branch_variance": float(minimum_branch_variance),
        "minimum_episode_macro_fraction_per_task": float(
            minimum_episode_macro_fraction_per_task
        ),
        "minimum_development_episodes_per_task": int(
            minimum_development_episodes_per_task
        ),
        "per_task": per_task,
    }


__all__ = [
    "PROTOCOL",
    "assign_a2_splits",
    "run_a2_collection",
    "summarize_intervention_support",
    "validate_a2_contract",
]
