"""Fresh H6-H8 confirmation data for continuous action-conditioned grounding."""

from __future__ import annotations

from dataclasses import replace
import zlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_collection import (
    FORMAL_TASK_INSTANCES,
    CollectionPayload,
    EpisodeRuntime,
    collect_source_episode,
    stack_action_branch_rows,
)
from fcpg.lewm_action_branch_metrics import (
    TrainMoments,
    center_whitened_branches,
    fit_train_moments,
)


PROTOCOL = "fcpg_action_branch_continuous_task_a2_confirmation_v1"
TASKS = (
    "pick-place-v3",
    "drawer-open-v3",
    "reach-v3",
    "push-v3",
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def validate_confirmation_contract(contract: Mapping[str, Any]) -> tuple[str, ...]:
    if contract.get("protocol") != PROTOCOL:
        raise ValueError(f"confirmation protocol must be {PROTOCOL}")
    tasks = tuple(str(value) for value in contract.get("tasks", ()))
    if tasks != TASKS:
        raise ValueError(f"confirmation tasks must remain {TASKS}")
    split = _mapping(contract.get("split"), "split")
    if (
        int(split.get("protocol_seed", -1)),
        int(split.get("train_episodes_per_task", -1)),
        int(split.get("test_episodes_per_task", -1)),
    ) != (27_103, 24, 24):
        raise ValueError(
            "confirmation split must keep seed 27103 and 24 train / 24 test"
        )
    collection = _mapping(contract.get("collection"), "collection")
    if (
        int(collection.get("benchmark_seed", -1)),
        int(collection.get("environment_seed_base", -1)),
        int(collection.get("horizon_steps", -1)),
    ) != (27103, 420000, 8):
        raise ValueError("confirmation collection seed and H=8 are fixed")
    hazard = _mapping(contract.get("hazard"), "hazard")
    if (
        int(hazard.get("decision_steps_per_episode", -1)),
        int(hazard.get("eligible_first_step", -1)),
        int(hazard.get("eligible_last_step", -1)),
        float(hazard.get("bernoulli_probability", -1.0)),
        int(hazard.get("refractory_steps", -1)),
        int(hazard.get("maximum_events_per_episode", -1)),
    ) != (40, 3, 31, 0.12, 5, 3):
        raise ValueError("confirmation random-hazard contract drifted")
    branches = _mapping(contract.get("matched_state_branches"), "branches")
    if (
        int(branches.get("branch_count", -1)),
        int(branches.get("interruption_horizon_steps", -1)),
        int(branches.get("raw_steps_per_model_step", -1)),
        float(branches.get("motion_delta_rms", -1.0)),
        branches.get("preserve_policy_gripper_commands"),
    ) != (8, 8, 5, 0.15, True):
        raise ValueError("confirmation action-branch contract drifted")
    support = _mapping(contract.get("support_gate"), "support_gate")
    if (
        tuple(int(value) for value in support.get("primary_horizons", ())),
        float(support.get("minimum_branch_variance", -1.0)),
        float(
            support.get(
                "minimum_episode_macro_informative_fraction_per_task", -1.0
            )
        ),
        int(support.get("episodes_per_task", -1)),
    ) != ((6, 7, 8), 0.01, 0.8, 24):
        raise ValueError("confirmation H6-H8 support Gate drifted")
    return tasks


def assign_confirmation_splits(
    contract: Mapping[str, Any],
) -> dict[tuple[str, int], str]:
    tasks = validate_confirmation_contract(contract)
    seed = int(_mapping(contract.get("split"), "split")["protocol_seed"])
    result: dict[tuple[str, int], str] = {}
    for task in tasks:
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, zlib.crc32(task.encode("utf-8"))])
        )
        ranked = rng.permutation(FORMAL_TASK_INSTANCES).tolist()
        for instance in ranked[:24]:
            result[(task, int(instance))] = "train"
        for instance in ranked[24:]:
            result[(task, int(instance))] = "test"
    return result


def run_confirmation_collection(
    contract: Mapping[str, Any],
    *,
    split: str,
    encoder: Any,
    runtime_factory: Callable[[str, int, int], EpisodeRuntime],
) -> CollectionPayload:
    tasks = validate_confirmation_contract(contract)
    if split not in {"train", "test"}:
        raise ValueError("confirmation collection split must be train or test")
    assignments = assign_confirmation_splits(contract)
    collection = _mapping(contract.get("collection"), "collection")
    environment_seed_base = int(collection["environment_seed_base"])
    horizon_steps = int(collection["horizon_steps"])
    rows = []
    for task_index, task in enumerate(tasks):
        instances = [
            instance
            for instance in FORMAL_TASK_INSTANCES
            if assignments[(task, int(instance))] == split
        ]
        if len(instances) != 24:
            raise RuntimeError(f"{task} confirmation split is not 24 episodes")
        for instance in instances:
            environment_seed = environment_seed_base + task_index * 10_000 + int(instance)
            collected = collect_source_episode(
                task=task,
                task_seed_index=task_index,
                task_index_mod50=int(instance),
                environment_seed=environment_seed,
                split=split,
                contract=contract,
                horizon_steps=horizon_steps,
                runtime_factory=runtime_factory,
                encode_records=encoder.encode,
            )
            rows.extend(
                replace(
                    row,
                    source_episode_id=(
                        f"{task}:a2-confirm:{split}:env-{environment_seed}:"
                        f"instance-{int(instance):02d}"
                    ),
                )
                for row in collected
            )
    arrays = stack_action_branch_rows(rows)
    if set(np.asarray(arrays["split"]).astype(str).tolist()) != {split}:
        raise RuntimeError("confirmation collector mixed split labels")
    return CollectionPayload(
        arrays=arrays,
        metadata={"protocol": PROTOCOL, "split": split, "task_count": len(tasks)},
    )


def confirmation_support(
    arrays: Mapping[str, np.ndarray],
    *,
    task_names: Sequence[str] = TASKS,
    moments: TrainMoments | None = None,
    minimum_branch_variance: float = 0.01,
    minimum_episode_macro_fraction_per_task: float = 0.8,
    expected_episodes_per_task: int = 24,
) -> tuple[dict[str, Any], TrainMoments]:
    split = np.asarray(arrays["split"]).astype(str)
    if set(split.tolist()) != {"train"}:
        raise ValueError("confirmation support accepts train rows only")
    target = np.asarray(arrays["target_embedding"], dtype=np.float32)
    if target.ndim != 5 or target.shape[2] != 8:
        raise ValueError("confirmation target must have shape [N,C,8,K,D]")
    fitted = fit_train_moments(target) if moments is None else moments
    centered = center_whitened_branches(target, fitted)[:, :, 5:8]
    variance = np.mean(np.square(centered), axis=(1, 2, 3, 4), dtype=np.float64)
    task_index = np.asarray(arrays["task_index"], dtype=np.int64)
    if set(task_index.tolist()) != set(range(len(task_names))):
        raise ValueError("confirmation task_index must cover exactly four tasks")
    episode = np.asarray(arrays["source_episode_id"]).astype(str)
    instance = np.asarray(arrays["task_instance_index"], dtype=np.int64)
    per_task: dict[str, Any] = {}
    for index, task in enumerate(task_names):
        selected = task_index == index
        if not all(
            episode_id.startswith(f"{task}:")
            for episode_id in episode[selected].tolist()
        ):
            raise ValueError(
                f"confirmation task_index {index} is not bound to task {task}"
            )
        identities = np.unique(episode[selected])
        if identities.size != expected_episodes_per_task:
            raise ValueError(
                f"{task} confirmation requires {expected_episodes_per_task} episodes"
            )
        fractions = np.asarray(
            [
                np.mean(variance[selected & (episode == identity)] >= minimum_branch_variance)
                for identity in identities
            ],
            dtype=np.float64,
        )
        macro = float(np.mean(fractions))
        episode_to_instance: dict[str, int] = {}
        instance_to_episode: dict[int, str] = {}
        for episode_id, instance_id in zip(
            episode[selected].tolist(), instance[selected].tolist(), strict=True
        ):
            previous_instance = episode_to_instance.setdefault(
                episode_id, int(instance_id)
            )
            previous_episode = instance_to_episode.setdefault(
                int(instance_id), episode_id
            )
            if previous_instance != int(instance_id) or previous_episode != episode_id:
                raise ValueError(
                    f"{task} confirmation episode/instance mapping is not one-to-one"
                )
        per_task[task] = {
            "episode_count": int(identities.size),
            "bundle_count": int(np.count_nonzero(selected)),
            "episode_macro_informative_fraction": macro,
            "median_branch_variance": float(np.median(variance[selected])),
            "passed": macro >= minimum_episode_macro_fraction_per_task,
        }
    passed = all(bool(row["passed"]) for row in per_task.values())
    return (
        {
            "protocol": PROTOCOL,
            "primary_horizons": [6, 7, 8],
            "decision": "Go" if passed else "No-Go",
            "passed": passed,
            "per_task": per_task,
        },
        fitted,
    )


__all__ = [
    "PROTOCOL",
    "TASKS",
    "assign_confirmation_splits",
    "confirmation_support",
    "run_confirmation_collection",
    "validate_confirmation_contract",
]
