"""Fresh A3 collection with physically diverse absolute action branches."""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_a1 import validate_a1_arrays
from fcpg.lewm_action_branch_a3 import build_diverse_action_branch_bank
from fcpg.lewm_action_branch_collection import (
    FORMAL_TASK_INSTANCES,
    SMOKE_TASK_INSTANCES,
    CollectionPayload,
    EpisodeRuntime,
    HazardScheduleConfig,
    collect_source_episode,
    sample_hazard_schedule,
    stack_action_branch_rows,
)


PROTOCOL = "fcpg_action_branch_physical_support_a3_v1"
TASKS = (
    "pick-place-v3",
    "drawer-open-v3",
    "reach-v3",
    "push-v3",
)
SEMANTIC_BRANCHES = (
    "expert",
    "reverse",
    "hold",
    "half_speed",
    "transverse_u_positive",
    "transverse_u_negative",
    "transverse_v_positive",
    "transverse_v_negative",
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def validate_a3_contract(contract: Mapping[str, Any]) -> tuple[str, ...]:
    if contract.get("protocol") != PROTOCOL:
        raise ValueError(f"A3 protocol must be {PROTOCOL!r}")
    tasks = tuple(str(value) for value in contract.get("tasks", ()))
    if tasks != TASKS:
        raise ValueError(f"A3 tasks must remain {TASKS}")

    split = _mapping(contract.get("split"), "split")
    counts = (
        int(split.get("train_episodes_per_task", -1)),
        int(split.get("development_episodes_per_task", -1)),
        int(split.get("test_episodes_per_task", -1)),
    )
    if int(split.get("protocol_seed", -1)) != 29_103 or counts != (24, 12, 12):
        raise ValueError("A3 split must remain seed 29103 and 24/12/12")

    collection = _mapping(contract.get("collection"), "collection")
    if (
        int(collection.get("benchmark_seed", -1)),
        int(collection.get("environment_seed_base", -1)),
        int(collection.get("horizon_steps", -1)),
    ) != (29_103, 520_000, 8):
        raise ValueError("A3 benchmark/environment seeds and H=8 are fixed")
    if not bool(collection.get("test_instances_not_collected_by_this_protocol", False)):
        raise ValueError("A3 zero-training screen must not collect reserved test")

    hazard = _mapping(contract.get("hazard"), "hazard")
    if (
        int(hazard.get("decision_steps_per_episode", -1)),
        int(hazard.get("history_steps", -1)),
        int(hazard.get("eligible_first_step", -1)),
        int(hazard.get("eligible_last_step", -1)),
        float(hazard.get("bernoulli_probability", -1.0)),
        int(hazard.get("refractory_steps", -1)),
        int(hazard.get("maximum_events_per_episode", -1)),
        hazard.get("condition_on_at_least_one_event"),
        hazard.get("schedule_drawn_before_environment_reset"),
        hazard.get("schedule_independent_of_state_action_reward_and_task_phase"),
    ) != (40, 3, 3, 31, 0.12, 5, 3, True, True, True):
        raise ValueError("A3 random-hazard contract drifted")

    branches = _mapping(contract.get("matched_state_branches"), "branches")
    if (
        int(branches.get("branch_count", -1)),
        int(branches.get("interruption_horizon_steps", -1)),
        int(branches.get("raw_steps_per_model_step", -1)),
        branches.get("preserve_policy_gripper_commands"),
        branches.get("motion_delta_rms"),
        tuple(str(value) for value in branches.get("semantic_bank_before_permutation", ())),
    ) != (8, 8, 5, True, None, SEMANTIC_BRANCHES):
        raise ValueError("A3 absolute action-bank contract drifted")

    support = _mapping(contract.get("support_gate"), "support_gate")
    if (
        tuple(int(value) for value in support.get("primary_horizons", ())),
        float(support.get("minimum_tcp_max_pairwise_m", -1.0)),
        float(support.get("minimum_object_max_pairwise_m", -1.0)),
        float(support.get("minimum_latent_branch_variance", -1.0)),
        float(support.get("minimum_episode_macro_informative_fraction_per_task", -1.0)),
    ) != ((6, 7, 8), 0.01, 0.005, 0.01, 0.8):
        raise ValueError("A3 support Gate drifted")
    return tasks


def assign_a3_splits(
    tasks: Sequence[str], contract: Mapping[str, Any]
) -> dict[tuple[str, int], str]:
    """Assign each MetaWorld task instance wholly to one frozen split."""

    protocol_seed = int(_mapping(contract.get("split"), "split")["protocol_seed"])
    result: dict[tuple[str, int], str] = {}
    for task in tasks:
        rng = np.random.default_rng(
            np.random.SeedSequence([protocol_seed, zlib.crc32(task.encode("utf-8"))])
        )
        ranked = rng.permutation(FORMAL_TASK_INSTANCES).tolist()
        offset = 0
        for label, count in (("train", 24), ("development", 12), ("test", 12)):
            for instance in ranked[offset : offset + count]:
                result[(str(task), int(instance))] = label
            offset += count
    return result


def _a3_schedule_rng(
    *, protocol_seed: int, namespace: str, task_index: int, instance: int
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(protocol_seed),
                int(zlib.crc32(namespace.encode("utf-8"))),
                int(task_index),
                int(instance),
                0,
            ]
        )
    )


def _a3_candidate_seeds(
    protocol_seed: int, task_index: int, instance: int, hazard_step: int
) -> tuple[int, int]:
    values = np.random.SeedSequence(
        [int(protocol_seed), int(task_index), int(instance), int(hazard_step)]
    ).generate_state(2, dtype=np.uint32)
    return int(values[0]), int(values[1])


def validate_a3_cache_conformance(
    arrays: Mapping[str, np.ndarray], contract: Mapping[str, Any]
) -> None:
    """Regenerate the frozen split, hazards, and anonymous branch bank."""

    validate_a1_arrays(arrays)
    tasks = validate_a3_contract(contract)
    split_values = np.asarray(arrays["split"]).astype(str)
    if set(split_values.tolist()) - {"train", "development"}:
        raise ValueError("A3 formal cache must contain only train/development rows")
    task_index = np.asarray(arrays["task_index"], dtype=np.int64)
    instance = np.asarray(arrays["task_instance_index"], dtype=np.int64)
    hazard = np.asarray(arrays["hazard_id"], dtype=np.int64)
    source_episode = np.asarray(arrays["source_episode_id"]).astype(str)
    center = np.asarray(arrays["center_action"], dtype=np.float32)
    future = np.asarray(arrays["future_action"], dtype=np.float32)
    assignments = assign_a3_splits(tasks, contract)
    split_contract = _mapping(contract.get("split"), "split")
    protocol_seed = int(split_contract["protocol_seed"])
    namespace = str(split_contract["assignment"])
    hazard_config = HazardScheduleConfig.from_contract(contract)

    for task_position, task in enumerate(tasks):
        selected_task = task_index == task_position
        if not np.any(selected_task):
            raise ValueError(f"A3 cache has no rows for {task}")
        if not all(
            value.startswith(f"{task}:") for value in source_episode[selected_task]
        ):
            raise ValueError("A3 task index and source identity disagree")
        expected_instances = {
            value
            for value in FORMAL_TASK_INSTANCES
            if assignments[(task, int(value))] in {"train", "development"}
        }
        observed_instances = set(instance[selected_task].tolist())
        if observed_instances != expected_instances:
            raise ValueError(f"A3 {task} instance membership drifted")
        for current_instance in sorted(expected_instances):
            selected = selected_task & (instance == current_instance)
            expected_split = assignments[(task, current_instance)]
            if set(split_values[selected].tolist()) != {expected_split}:
                raise ValueError("A3 cached split assignment drifted")
            expected_hazards = sample_hazard_schedule(
                hazard_config,
                rng=_a3_schedule_rng(
                    protocol_seed=protocol_seed,
                    namespace=namespace,
                    task_index=task_position,
                    instance=current_instance,
                ),
            )
            if tuple(sorted(hazard[selected].tolist())) != tuple(expected_hazards):
                raise ValueError("A3 cached hazard schedule is incomplete or altered")

    for row in range(task_index.size):
        direction_seed, permutation_seed = _a3_candidate_seeds(
            protocol_seed,
            int(task_index[row]),
            int(instance[row]),
            int(hazard[row]),
        )
        expected_bank = build_diverse_action_branch_bank(
            center[row].reshape(40, 4),
            direction_seed=direction_seed,
            permutation_seed=permutation_seed,
            motion_delta_rms=None,
        )
        if not np.array_equal(
            expected_bank.actions,
            future[row].reshape(8, 40, 4),
        ):
            raise ValueError("A3 cached action bank does not match the frozen builder")


def run_a3_collection(
    contract: Mapping[str, Any],
    *,
    encoder: Any,
    runtime_factory: Callable[[str, int, int], EpisodeRuntime],
    smoke: bool,
    max_tasks: int | None = None,
    max_episodes_per_task: int | None = None,
    max_events_per_episode: int | None = None,
) -> CollectionPayload:
    """Collect A3 latent/action rows; physical state is replayed separately."""

    configured_tasks = validate_a3_contract(contract)
    if max_tasks is not None:
        if not smoke or int(max_tasks) <= 0:
            raise ValueError("max_tasks is a positive smoke-only bound")
        tasks = configured_tasks[: int(max_tasks)]
    else:
        tasks = configured_tasks
    collection = _mapping(contract.get("collection"), "collection")
    assignments = assign_a3_splits(configured_tasks, contract)
    environment_seed_base = int(collection["environment_seed_base"])
    horizon_steps = int(collection["horizon_steps"])

    rows = []
    for task_index, task in enumerate(tasks):
        if smoke:
            instances = SMOKE_TASK_INSTANCES
            if max_episodes_per_task is not None:
                if not 0 < int(max_episodes_per_task) <= len(SMOKE_TASK_INSTANCES):
                    raise ValueError("smoke episodes_per_task must be one or two")
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
                    branch_bank_builder=build_diverse_action_branch_bank,
                )
            )
    arrays = stack_action_branch_rows(rows)
    if np.any(np.asarray(arrays["split"]).astype(str) == "test"):
        raise RuntimeError("A3 zero-training screen unexpectedly collected test")
    return CollectionPayload(
        arrays=arrays,
        metadata={
            "protocol": PROTOCOL,
            "smoke": bool(smoke),
            "task_count": len(tasks),
            "horizon_steps": horizon_steps,
        },
    )


__all__ = [
    "PROTOCOL",
    "SEMANTIC_BRANCHES",
    "TASKS",
    "assign_a3_splits",
    "run_a3_collection",
    "validate_a3_cache_conformance",
    "validate_a3_contract",
]
