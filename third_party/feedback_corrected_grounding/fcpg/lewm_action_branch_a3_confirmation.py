"""Authorized reserved-test collection and canonical assembly for A3.

The A3 screen keeps its test episodes unopened.  This module opens exactly
those frozen episodes only after validating the 2k development decision, then
combines the original train rows with the fresh test rows.  Development rows
are never admitted to the confirmation canonical.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_a1 import NPZ_KEYS, validate_a1_arrays
from fcpg.lewm_action_branch_a3 import build_diverse_action_branch_bank
from fcpg.lewm_action_branch_a3_collection import (
    PROTOCOL,
    TASKS,
    _a3_candidate_seeds,
    _a3_schedule_rng,
    assign_a3_splits,
    validate_a3_cache_conformance,
    validate_a3_contract,
)
from fcpg.lewm_action_branch_a3_metrics import (
    CHECKPOINT_STEP,
    PRIMARY_HORIZONS,
    TRAINING_SEEDS,
)
from fcpg.lewm_action_branch_collection import (
    FORMAL_TASK_INSTANCES,
    CollectionPayload,
    EpisodeRuntime,
    HazardScheduleConfig,
    collect_source_episode,
    sample_hazard_schedule,
    stack_action_branch_rows,
)


DEVELOPMENT_STAGE = "a3_2k_development"
DEVELOPMENT_GO = "Development-Go"
CONFIRMATION_TRAINING_SEEDS = (29_301, 29_302, 29_303)
EXPECTED_GATE_CHECKS = frozenset(
    {
        "late_effect_vs_no_action",
        "late_effect_vs_shuffled_action",
        "late_effect_vs_every_shuffled_shift",
        "jointly_positive_tasks",
        "jointly_positive_seeds",
        "cross_seed_task_replication",
        "raw_not_worse_than_no_action",
        "raw_better_than_hold_last",
        "raw_better_than_constant_velocity",
    }
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _same_typed_json(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _same_typed_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_typed_json(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _member_subset(
    value: object,
    *,
    allowed: Sequence[str | int],
    minimum: int,
    label: str,
) -> tuple[str | int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of distinct frozen members")
    try:
        distinct = len(set(value)) == len(value)
    except TypeError as error:
        raise ValueError(
            f"{label} must be a list of distinct frozen members"
        ) from error
    allowed_types = {type(item) for item in allowed}
    if (
        not distinct
        or any(type(item) not in allowed_types for item in value)
        or not set(value).issubset(set(allowed))
        or len(value) < minimum
    ):
        raise ValueError(f"{label} does not satisfy the frozen A3 minimum")
    return tuple(value)


def validate_a3_confirmation_contract(
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the frozen post-development A3 confirmation section."""

    tasks = validate_a3_contract(contract)
    split = _mapping(contract.get("split"), "split")
    screen_training = _mapping(
        contract.get("training_if_gate_go"), "training_if_gate_go"
    )
    if not _same_typed_json(
        [screen_training.get("optimizer_steps"), screen_training.get("seeds")],
        [CHECKPOINT_STEP, list(TRAINING_SEEDS)],
    ):
        raise ValueError("A3 confirmation requires the frozen 2k screen contract")

    development_gate = _mapping(
        contract.get("development_gate"), "development_gate"
    )
    development_observed = (
        development_gate.get("evaluation_split"),
        development_gate.get("checkpoint_step"),
        development_gate.get("seeds"),
        development_gate.get("primary_horizons"),
        development_gate.get("minimum_relative_improvement"),
        development_gate.get("minimum_positive_tasks"),
        development_gate.get("minimum_positive_seeds"),
        development_gate.get("minimum_cross_replicating_seeds"),
        development_gate.get("minimum_positive_tasks_per_replicating_seed"),
        development_gate.get("every_shuffled_shift_must_reach_minimum"),
        development_gate.get("raw_factual_not_worse_than_no_action"),
        development_gate.get(
            "raw_factual_better_than_hold_last_and_constant_velocity"
        ),
    )
    development_expected = (
        "development",
        CHECKPOINT_STEP,
        list(TRAINING_SEEDS),
        list(PRIMARY_HORIZONS),
        0.05,
        3,
        2,
        2,
        3,
        True,
        True,
        True,
    )
    if not _same_typed_json(
        list(development_observed), list(development_expected)
    ):
        raise ValueError("A3 development Gate contract drifted")

    confirmation = _mapping(
        contract.get("confirmation_after_development_go"),
        "confirmation_after_development_go",
    )
    observed = (
        confirmation.get("reserved_test_episodes_per_task"),
        confirmation.get("test_collection_requires_development_go"),
        confirmation.get("optimizer_steps"),
        confirmation.get("seeds"),
        confirmation.get("fresh_start_from_frozen_base"),
        confirmation.get("train_split"),
        confirmation.get("evaluation_split"),
        confirmation.get("bundle_batch_size"),
        confirmation.get("amp_dtype"),
        confirmation.get("anchor_encoder_sigreg"),
        confirmation.get("additional_sigreg_values_allowed"),
        confirmation.get("selection"),
        confirmation.get("primary_horizons"),
        confirmation.get("minimum_relative_improvement"),
        confirmation.get("minimum_positive_tasks"),
        confirmation.get("minimum_positive_seeds"),
        confirmation.get("minimum_cross_replicating_seeds"),
        confirmation.get("minimum_positive_tasks_per_replicating_seed"),
        confirmation.get("every_shuffled_shift_must_reach_minimum"),
        confirmation.get("raw_factual_not_worse_than_no_action"),
        confirmation.get("raw_factual_better_than_hold_last_and_constant_velocity"),
    )
    expected = (
        12,
        True,
        10_000,
        list(CONFIRMATION_TRAINING_SEEDS),
        True,
        "train",
        "test",
        16,
        "float16",
        0.02,
        [],
        "fixed_latest",
        list(PRIMARY_HORIZONS),
        0.05,
        3,
        2,
        2,
        3,
        True,
        True,
        True,
    )
    if not _same_typed_json(list(observed), list(expected)):
        raise ValueError("A3 confirmation contract drifted")
    if type(split.get("test_episodes_per_task")) is not int or (
        split.get("test_episodes_per_task") != expected[0]
    ):
        raise ValueError("A3 confirmation must open exactly 12 reserved tests per task")
    return tasks


def validate_a3_development_go_summary(
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Authorize test collection from one complete A3 2k gate summary."""

    tasks = validate_a3_confirmation_contract(contract)
    if not isinstance(summary, Mapping):
        raise TypeError("A3 2k summary must be a mapping")
    if (
        summary.get("protocol") != PROTOCOL
        or summary.get("stage") != DEVELOPMENT_STAGE
        or summary.get("decision") != DEVELOPMENT_GO
        or summary.get("passed") is not True
        or summary.get("input_contains_test_rows") is not False
        or type(summary.get("checkpoint_step")) is not int
        or summary.get("checkpoint_step") != CHECKPOINT_STEP
        or not _same_typed_json(
            summary.get("primary_horizons"), list(PRIMARY_HORIZONS)
        )
    ):
        raise ValueError(
            "A3 reserved-test collection requires the 2k Development-Go summary"
        )

    coverage = _mapping(summary.get("coverage"), "A3 summary coverage")
    if not _same_typed_json(coverage.get("tasks"), list(tasks)):
        raise ValueError("A3 Development-Go summary task coverage drifted")
    if not _same_typed_json(
        coverage.get("training_seeds"), list(TRAINING_SEEDS)
    ):
        raise ValueError("A3 Development-Go summary training seeds drifted")
    per_task = _mapping(
        coverage.get("per_task"), "A3 summary per-task coverage"
    )
    if set(per_task) != set(tasks):
        raise ValueError("A3 Development-Go summary must cover every frozen task")
    split = _mapping(contract.get("split"), "split")
    train_count = int(split["train_episodes_per_task"])
    development_count = int(split["development_episodes_per_task"])
    development_bundles = 0
    for task in tasks:
        row = _mapping(per_task[task], f"A3 summary coverage for {task}")
        bundle_count = _positive_integer(
            row.get("development_hazard_bundle_count"),
            f"A3 development hazard coverage for {task}",
        )
        if (
            _positive_integer(
                row.get("train_physical_episode_count"),
                f"A3 train episode coverage for {task}",
            )
            != train_count
            or _positive_integer(
                row.get("development_physical_episode_count"),
                f"A3 development episode coverage for {task}",
            )
            != development_count
        ):
            raise ValueError(f"A3 Development-Go coverage drifted for {task}")
        development_bundles += bundle_count
    if _positive_integer(
        coverage.get("development_hazard_bundle_count"),
        "A3 development hazard coverage",
    ) != development_bundles:
        raise ValueError("A3 Development-Go development bundle coverage drifted")

    development_gate = _mapping(
        contract.get("development_gate"),
        "development_gate",
    )
    minimum_effect = float(development_gate["minimum_relative_improvement"])
    minimum_tasks = int(development_gate["minimum_positive_tasks"])
    minimum_seeds = int(development_gate["minimum_positive_seeds"])
    minimum_cross_seeds = int(
        development_gate["minimum_cross_replicating_seeds"]
    )
    minimum_tasks_per_cross_seed = int(
        development_gate["minimum_positive_tasks_per_replicating_seed"]
    )

    whitening = _mapping(summary.get("whitening"), "A3 summary whitening")
    if (
        whitening.get("source") != "canonical_train_rows"
        or whitening.get("per_task") is not True
        or whitening.get("episode_equal") is not True
        or not _same_typed_json(
            whitening.get("fit_horizons"), list(PRIMARY_HORIZONS)
        )
    ):
        raise ValueError("A3 Development-Go whitening contract drifted")
    whitening_tasks = _mapping(
        whitening.get("tasks"), "A3 summary whitening tasks"
    )
    if set(whitening_tasks) != set(tasks):
        raise ValueError("A3 Development-Go whitening must cover every task")
    for task in tasks:
        row = _mapping(whitening_tasks[task], f"A3 whitening for {task}")
        if _positive_integer(
            row.get("train_physical_episode_count"),
            f"A3 whitening episode coverage for {task}",
        ) != train_count:
            raise ValueError(f"A3 Development-Go whitening drifted for {task}")

    comparisons = _mapping(summary.get("comparisons"), "A3 summary comparisons")
    for baseline in ("no_action", "shuffled_action"):
        row = _mapping(comparisons.get(baseline), f"A3 {baseline} comparison")
        if (
            row.get("baseline_method") != baseline
            or row.get("factual_method") != "factual"
            or _positive_integer(row.get("task_count"), "A3 comparison task count")
            != len(tasks)
            or _positive_integer(row.get("seed_count"), "A3 comparison seed count")
            != len(TRAINING_SEEDS)
            or _positive_integer(
                row.get("positive_task_count"), "A3 positive task count"
            )
            < minimum_tasks
            or _positive_integer(
                row.get("positive_seed_count"), "A3 positive seed count"
            )
            < minimum_seeds
            or _finite_number(row.get("effect"), f"A3 {baseline} effect")
            < minimum_effect
        ):
            raise ValueError(f"A3 Development-Go {baseline} comparison failed")

    shifts = _mapping(
        summary.get("shuffled_action_shifts"), "A3 summary shuffled shifts"
    )
    per_shift = shifts.get("per_shift")
    if (
        shifts.get("count") != 7
        or not isinstance(per_shift, list)
        or len(per_shift) != 7
        or _finite_number(shifts.get("minimum_effect"), "A3 minimum shift effect")
        < minimum_effect
    ):
        raise ValueError("A3 Development-Go must pass all seven shuffled shifts")
    for expected_shift, raw_row in enumerate(per_shift, start=1):
        row = _mapping(raw_row, f"A3 shuffled shift {expected_shift}")
        if (
            _positive_integer(row.get("donor_shift"), "A3 donor shift")
            != expected_shift
            or _positive_integer(
                row.get("positive_task_count"), "A3 shift positive task count"
            )
            < minimum_tasks
            or _positive_integer(
                row.get("positive_seed_count"), "A3 shift positive seed count"
            )
            < minimum_seeds
            or _finite_number(row.get("effect"), "A3 shuffled shift effect")
            < minimum_effect
        ):
            raise ValueError(
                f"A3 Development-Go shuffled shift {expected_shift} failed"
            )

    raw = _mapping(summary.get("raw_h6_h8"), "A3 summary raw H6-H8")
    raw_methods = _mapping(raw.get("methods"), "A3 summary raw methods")
    raw_mse: dict[str, float] = {}
    for method in (
        "factual",
        "no_action",
        "hold_last",
        "latent_constant_velocity",
    ):
        row = _mapping(raw_methods.get(method), f"A3 raw {method}")
        raw_mse[method] = _finite_number(
            row.get("task_macro_mse"), f"A3 raw {method} MSE"
        )
        if raw_mse[method] < 0.0:
            raise ValueError("A3 Development-Go raw MSE must be non-negative")
    if (
        raw_mse["factual"] > raw_mse["no_action"]
        or raw_mse["factual"] >= raw_mse["hold_last"]
        or raw_mse["factual"] >= raw_mse["latent_constant_velocity"]
    ):
        raise ValueError("A3 Development-Go raw factual baselines failed")

    gate = _mapping(summary.get("gate"), "A3 summary gate")
    checks = _mapping(gate.get("checks"), "A3 summary gate checks")
    if (
        gate.get("decision") != DEVELOPMENT_GO
        or gate.get("passed") is not True
        or set(checks) != EXPECTED_GATE_CHECKS
        or any(value is not True for value in checks.values())
        or gate.get("failed_checks") != []
        or _finite_number(
            gate.get("minimum_relative_improvement"), "A3 gate minimum effect"
        )
        != minimum_effect
    ):
        raise ValueError("A3 2k gate does not authorize reserved-test collection")

    _member_subset(
        gate.get("jointly_positive_tasks"),
        allowed=tasks,
        minimum=minimum_tasks,
        label="A3 jointly positive tasks",
    )
    _member_subset(
        gate.get("jointly_positive_seeds"),
        allowed=TRAINING_SEEDS,
        minimum=minimum_seeds,
        label="A3 jointly positive seeds",
    )
    cross_seeds = _member_subset(
        gate.get("cross_replicating_seeds"),
        allowed=TRAINING_SEEDS,
        minimum=minimum_cross_seeds,
        label="A3 cross-replicating seeds",
    )
    cross_counts = _mapping(
        gate.get("cross_replicating_seed_task_counts"),
        "A3 cross-replicating seed task counts",
    )
    expected_seed_keys = {str(seed) for seed in TRAINING_SEEDS}
    if set(cross_counts) != expected_seed_keys:
        raise ValueError("A3 cross-replicating seed task counts are incomplete")
    qualifying_seeds: list[int] = []
    for seed in TRAINING_SEEDS:
        count = _nonnegative_integer(
            cross_counts[str(seed)],
            f"A3 cross-replicating task count for seed {seed}",
        )
        if count > len(tasks):
            raise ValueError("A3 cross-replicating task count exceeds task coverage")
        if count >= minimum_tasks_per_cross_seed:
            qualifying_seeds.append(seed)
    if tuple(qualifying_seeds) != cross_seeds:
        raise ValueError("A3 cross-replicating seed evidence is inconsistent")


def load_a3_development_go_summary(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and validate the only artifact that may unseal A3 reserved test."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("A3 2k summary must contain a JSON object")
    validate_a3_development_go_summary(payload, contract)
    return payload


def _expected_hazards(
    contract: Mapping[str, Any],
    *,
    task_index: int,
    instance: int,
) -> tuple[int, ...]:
    split = _mapping(contract.get("split"), "split")
    return sample_hazard_schedule(
        HazardScheduleConfig.from_contract(contract),
        rng=_a3_schedule_rng(
            protocol_seed=int(split["protocol_seed"]),
            namespace=str(split["assignment"]),
            task_index=int(task_index),
            instance=int(instance),
        ),
    )


def _validate_frozen_split_rows(
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    *,
    expected_splits: Sequence[str],
) -> None:
    validate_a1_arrays(arrays)
    tasks = validate_a3_confirmation_contract(contract)
    required_splits = tuple(str(value) for value in expected_splits)
    if len(required_splits) != len(set(required_splits)) or not required_splits:
        raise ValueError("A3 expected splits must be distinct and non-empty")

    split_values = np.asarray(arrays["split"]).astype(str)
    if set(split_values.tolist()) != set(required_splits):
        raise ValueError(
            f"A3 rows must contain exactly splits {required_splits}; "
            "development is excluded from confirmation"
        )
    task_index = np.asarray(arrays["task_index"], dtype=np.int64)
    instance = np.asarray(arrays["task_instance_index"], dtype=np.int64)
    source_episode = np.asarray(arrays["source_episode_id"]).astype(str)
    hazard = np.asarray(arrays["hazard_id"], dtype=np.int64)
    center = np.asarray(arrays["center_action"], dtype=np.float32)
    future = np.asarray(arrays["future_action"], dtype=np.float32)
    if set(task_index.tolist()) != set(range(len(tasks))):
        raise ValueError("A3 confirmation task indices must cover exactly four tasks")

    assignments = assign_a3_splits(tasks, contract)
    split_contract = _mapping(contract.get("split"), "split")
    protocol_seed = int(split_contract["protocol_seed"])
    for task_position, task in enumerate(tasks):
        selected_task = task_index == task_position
        expected_instances = {
            int(value)
            for value in FORMAL_TASK_INSTANCES
            if assignments[(task, int(value))] in required_splits
        }
        if set(instance[selected_task].tolist()) != expected_instances:
            raise ValueError(f"A3 {task} confirmation episode membership drifted")
        for current_instance in sorted(expected_instances):
            selected = selected_task & (instance == current_instance)
            expected_split = assignments[(task, current_instance)]
            if set(split_values[selected].tolist()) != {expected_split}:
                raise ValueError("A3 confirmation split assignment drifted")
            expected_episode = f"{task}:task-instance-{current_instance:02d}"
            if set(source_episode[selected].tolist()) != {expected_episode}:
                raise ValueError("A3 confirmation source episode identity drifted")
            expected_hazards = _expected_hazards(
                contract,
                task_index=task_position,
                instance=current_instance,
            )
            if tuple(sorted(hazard[selected].tolist())) != tuple(expected_hazards):
                raise ValueError("A3 confirmation hazard schedule drifted")

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
            raise ValueError("A3 confirmation action bank drifted")


def validate_a3_confirmation_test_conformance(
    arrays: Mapping[str, np.ndarray], contract: Mapping[str, Any]
) -> None:
    """Validate a complete fresh reserved-test cache."""

    _validate_frozen_split_rows(arrays, contract, expected_splits=("test",))


def validate_a3_confirmation_canonical_conformance(
    arrays: Mapping[str, np.ndarray], contract: Mapping[str, Any]
) -> None:
    """Validate a confirmation canonical containing train and test only."""

    _validate_frozen_split_rows(
        arrays,
        contract,
        expected_splits=("train", "test"),
    )


def run_a3_confirmation_collection(
    contract: Mapping[str, Any],
    *,
    development_summary: Mapping[str, Any],
    encoder: Any,
    runtime_factory: Callable[[str, int, int], EpisodeRuntime],
) -> CollectionPayload:
    """Collect all and only the 12 frozen reserved-test episodes per task."""

    tasks = validate_a3_confirmation_contract(contract)
    validate_a3_development_go_summary(development_summary, contract)
    assignments = assign_a3_splits(tasks, contract)
    collection = _mapping(contract.get("collection"), "collection")
    environment_seed_base = int(collection["environment_seed_base"])
    horizon_steps = int(collection["horizon_steps"])
    expected_per_task = int(
        _mapping(
            contract.get("confirmation_after_development_go"),
            "confirmation_after_development_go",
        )["reserved_test_episodes_per_task"]
    )

    rows = []
    for task_position, task in enumerate(tasks):
        instances = tuple(
            int(instance)
            for instance in FORMAL_TASK_INSTANCES
            if assignments[(task, int(instance))] == "test"
        )
        if len(instances) != expected_per_task:
            raise RuntimeError(
                f"{task} A3 confirmation split is not {expected_per_task} episodes"
            )
        for instance in instances:
            environment_seed = (
                environment_seed_base + task_position * 10_000 + instance
            )
            rows.extend(
                collect_source_episode(
                    task=task,
                    task_seed_index=task_position,
                    task_index_mod50=instance,
                    environment_seed=environment_seed,
                    split="test",
                    contract=contract,
                    horizon_steps=horizon_steps,
                    runtime_factory=runtime_factory,
                    encode_records=encoder.encode,
                    branch_bank_builder=build_diverse_action_branch_bank,
                )
            )
    arrays = stack_action_branch_rows(rows)
    validate_a3_confirmation_test_conformance(arrays, contract)
    return CollectionPayload(
        arrays=arrays,
        metadata={
            "protocol": PROTOCOL,
            "split": "test",
            "task_count": len(tasks),
            "episodes_per_task": expected_per_task,
        },
    )


def merge_a3_confirmation_canonical(
    screen_arrays: Mapping[str, np.ndarray],
    test_arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return canonical A3 train+test rows, discarding screen development."""

    validate_a3_confirmation_contract(contract)
    validate_a3_cache_conformance(screen_arrays, contract)
    validate_a3_confirmation_test_conformance(test_arrays, contract)
    screen_split = np.asarray(screen_arrays["split"]).astype(str)
    train_rows = screen_split == "train"
    if not np.any(train_rows):
        raise ValueError("A3 screen contains no train rows")
    canonical = {
        name: np.concatenate(
            (
                np.asarray(screen_arrays[name])[train_rows],
                np.asarray(test_arrays[name]),
            ),
            axis=0,
        )
        for name in NPZ_KEYS
    }
    validate_a3_confirmation_canonical_conformance(canonical, contract)
    if np.any(np.asarray(canonical["split"]).astype(str) == "development"):
        raise RuntimeError("A3 confirmation canonical retained development rows")
    return canonical


# Short aliases retain the terminology used by the command-line entry point.
load_development_go_summary = load_a3_development_go_summary
validate_development_go_summary = validate_a3_development_go_summary


__all__ = [
    "CONFIRMATION_TRAINING_SEEDS",
    "DEVELOPMENT_GO",
    "DEVELOPMENT_STAGE",
    "EXPECTED_GATE_CHECKS",
    "load_a3_development_go_summary",
    "load_development_go_summary",
    "merge_a3_confirmation_canonical",
    "run_a3_confirmation_collection",
    "validate_a3_confirmation_canonical_conformance",
    "validate_a3_confirmation_contract",
    "validate_a3_confirmation_test_conformance",
    "validate_a3_development_go_summary",
    "validate_development_go_summary",
]
