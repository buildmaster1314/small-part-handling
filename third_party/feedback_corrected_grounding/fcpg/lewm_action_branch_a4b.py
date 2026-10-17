"""Frozen protocol primitives for the independent A4b fresh confirmation."""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from fcpg import lewm_action_branch_a4 as _a4


PROTOCOL = "fcpg_action_branch_active_control_a4b_fresh_confirmation_v1"
STATUS = "preregistered"
SPLIT_MANIFEST_PROTOCOL = "fcpg_action_branch_a4b_fresh_candidate_split_v1"
SPLIT_MANIFEST_ID = "a4b-fresh-preflight8-final40-seed31401-v1"
SPLIT_NAMESPACE = "a4b-fresh-preflight8-final40-v1"
TASK_IDENTITY_NAMESPACE = "a4b-task-instance-v1"

SOURCE_A4_PROTOCOL = _a4.PROTOCOL
SOURCE_A4_CONFIG_PATH = "config/lewm/eval/action_branch_active_control_a4.json"
SPLIT_MANIFEST_PATH = (
    "config/lewm/splits/action_branch_a4b_fresh_confirmation_seed31401.json"
)

TASKS = _a4.TASKS
RAW_TASK_INDICES = _a4.FORMAL_TASK_INSTANCES
SPLIT_SEED = 31_401
BENCHMARK_SEED = 31_411
PROPOSAL_SEED = 31_421
CUSTOM_TASK_PARAMETER_SEED = 31_431
PREFLIGHT_ENVIRONMENT_SEED_BASE = 710_000
FINAL_ENVIRONMENT_SEED_BASE = 720_000
ENVIRONMENT_SEED_TASK_STRIDE = 100
PREFLIGHT_CANDIDATES_PER_TASK = 8
FINAL_CANDIDATES_PER_TASK = 40
TARGET_ELIGIBLE_CANDIDATES_PER_TASK = 12
CONFIRMATION_TRAINING_SEEDS = _a4.CONFIRMATION_TRAINING_SEEDS

POOL_PREFLIGHT = "preflight"
POOL_FINAL = "final"
POOLS = (POOL_PREFLIGHT, POOL_FINAL)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
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


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if not _same_typed_json(value, expected):
        raise ValueError(f"{label} drifted from the frozen A4b protocol")


def _expected_candidate_order(task: str) -> tuple[int, ...]:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                SPLIT_SEED,
                zlib.crc32(SPLIT_NAMESPACE.encode("utf-8")),
                zlib.crc32(task.encode("utf-8")),
            ]
        )
    )
    return tuple(int(value) for value in rng.permutation(RAW_TASK_INDICES))


def validate_a4b_split_manifest(
    manifest: Mapping[str, Any],
) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
    """Validate the explicit, result-independent candidate order and limits."""

    root = _mapping(manifest, label="A4b split manifest")
    expected_keys = {
        "schema_version",
        "protocol",
        "a4b_protocol",
        "manifest_id",
        "split_seed",
        "namespace",
        "task_identity_namespace",
        "task_instance_universe",
        "pool_limits_per_task",
        "candidate_order_by_task",
    }
    if set(root) != expected_keys:
        raise ValueError("A4b split manifest keys drifted")
    expected_header = {
        "schema_version": 1,
        "protocol": SPLIT_MANIFEST_PROTOCOL,
        "a4b_protocol": PROTOCOL,
        "manifest_id": SPLIT_MANIFEST_ID,
        "split_seed": SPLIT_SEED,
        "namespace": SPLIT_NAMESPACE,
        "task_identity_namespace": TASK_IDENTITY_NAMESPACE,
    }
    for name, expected in expected_header.items():
        _require_exact(root.get(name), expected, label=f"A4b manifest {name}")
    _require_exact(
        root.get("task_instance_universe"),
        {"minimum": 0, "maximum": 47, "count": 48},
        label="A4b raw task-index universe",
    )
    _require_exact(
        root.get("pool_limits_per_task"),
        {
            POOL_PREFLIGHT: PREFLIGHT_CANDIDATES_PER_TASK,
            POOL_FINAL: FINAL_CANDIDATES_PER_TASK,
        },
        label="A4b pool limits",
    )

    rows = _mapping(
        root.get("candidate_order_by_task"), label="A4b candidate order"
    )
    if tuple(rows) != TASKS:
        raise ValueError("A4b candidate-order task panel or order drifted")
    validated: dict[str, dict[str, tuple[int, ...]]] = {}
    for task in TASKS:
        pools = _mapping(rows[task], label=f"A4b pools for {task}")
        if tuple(pools) != POOLS:
            raise ValueError(f"A4b pool names or order drifted for {task}")
        preflight = tuple(int(value) for value in pools[POOL_PREFLIGHT])
        final = tuple(int(value) for value in pools[POOL_FINAL])
        if len(preflight) != PREFLIGHT_CANDIDATES_PER_TASK:
            raise ValueError(f"A4b preflight candidate limit drifted for {task}")
        if len(final) != FINAL_CANDIDATES_PER_TASK:
            raise ValueError(f"A4b final candidate limit drifted for {task}")
        if preflight + final != _expected_candidate_order(task):
            raise ValueError(
                f"A4b candidate order differs from preregistered seed for {task}"
            )
        if set(preflight) & set(final) or set(preflight + final) != set(
            RAW_TASK_INDICES
        ):
            raise ValueError(f"A4b pools are not disjoint/exhaustive for {task}")
        validated[task] = {POOL_PREFLIGHT: preflight, POOL_FINAL: final}
    return validated


def _old_eligibility_projection(source_contract: Mapping[str, Any]) -> dict[str, Any]:
    eligibility = _mapping(
        source_contract.get("eligibility"), label="A4 v2 eligibility"
    )
    return {
        "source_protocol": SOURCE_A4_PROTOCOL,
        "reuse": "exact_function_and_thresholds",
        "allowed_inputs": eligibility["allowed_inputs"],
        "history_steps": eligibility["history_steps"],
        "minimum_expert_xyz_norm": eligibility["minimum_expert_xyz_norm"],
        "recent_action_window_raw_steps": eligibility[
            "recent_action_window_raw_steps"
        ],
        "minimum_recent_action_rms": eligibility["minimum_recent_action_rms"],
        "minimum_workspace_and_joint_margin": eligibility[
            "minimum_workspace_and_joint_margin"
        ],
        "branch_horizon_raw_steps": eligibility["branch_horizon_raw_steps"],
        "record_every_proposal": eligibility["record_every_proposal"],
        "collect_all_rejection_reasons": eligibility[
            "collect_all_rejection_reasons"
        ],
        "future_or_outcome_row_filtering_allowed": eligibility[
            "future_or_outcome_row_filtering_allowed"
        ],
    }


def _old_confirmation_projection(source_contract: Mapping[str, Any]) -> dict[str, Any]:
    gate = _mapping(
        source_contract.get("confirmation_after_development_go"),
        label="A4 v2 confirmation Gate",
    )
    names = (
        "evaluation_split",
        "primary_horizons",
        "primary_metric",
        "minimum_relative_improvement",
        "paired_ci_lower_must_exceed",
        "bootstrap_unit",
        "bootstrap_samples",
        "task_macro",
        "minimum_positive_tasks",
        "minimum_positive_seeds",
        "must_beat_matched_no_action",
        "must_beat_all_seven_wrong_action_shifts",
        "must_beat_hold_last_and_constant_velocity",
    )
    return {name: gate[name] for name in names}


def validate_a4b_contract(
    contract: Mapping[str, Any],
    *,
    source_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate A4b and prove eligibility/statistical thresholds equal A4 v2."""

    _a4.validate_a4_contract(source_contract)
    validate_a4b_split_manifest(split_manifest)
    root = _mapping(contract, label="A4b contract")
    expected_top_keys = {
        "schema_version",
        "protocol",
        "status",
        "source_a4_v2",
        "tasks",
        "split",
        "identity_audit",
        "collection",
        "proposal_schedule",
        "eligibility_source",
        "preflight_pool",
        "final_pool",
        "endpoint_reuse",
        "confirmation_gate",
    }
    if set(root) != expected_top_keys:
        raise ValueError("A4b top-level keys drifted")
    _require_exact(root.get("schema_version"), 1, label="A4b schema_version")
    _require_exact(root.get("protocol"), PROTOCOL, label="A4b protocol")
    _require_exact(root.get("status"), STATUS, label="A4b status")
    _require_exact(root.get("tasks"), list(TASKS), label="A4b tasks")
    _require_exact(
        root.get("source_a4_v2"),
        {"protocol": SOURCE_A4_PROTOCOL, "config_path": SOURCE_A4_CONFIG_PATH},
        label="A4b source A4 v2 identity",
    )
    _require_exact(
        root.get("split"),
        {
            "manifest_protocol": SPLIT_MANIFEST_PROTOCOL,
            "manifest_id": SPLIT_MANIFEST_ID,
            "manifest_path": SPLIT_MANIFEST_PATH,
            "split_seed": SPLIT_SEED,
            "unit": "source_physical_episode",
            "task_identity_namespace": TASK_IDENTITY_NAMESPACE,
            "raw_task_index_minimum": 0,
            "raw_task_index_maximum": 47,
            "preflight_candidates_per_task": PREFLIGHT_CANDIDATES_PER_TASK,
            "final_candidates_per_task": FINAL_CANDIDATES_PER_TASK,
            "candidate_pools_disjoint": True,
            "candidate_order_preregistered": True,
            "candidate_limit_preregistered": True,
            "old_a4_benchmark_instances_reused": False,
        },
        label="A4b split",
    )
    _require_exact(
        root.get("identity_audit"),
        {
            "legacy_identity_inventory_required": True,
            "legacy_scopes": [
                "a4_v2_formal_train",
                "a4_v2_formal_development",
                "a4_v2_attempted_test",
            ],
            "reset_digest_fields": ["qpos", "qvel", "goal", "task_data"],
            "persist_episode_uid": True,
            "persist_reset_physical_state_goal_digest": True,
            "digest_intersection_must_be_zero": True,
            "custom_task_fresh_parameters": (
                "a4b_protocol_seeded_continuous_offsets"
            ),
        },
        label="A4b identity audit",
    )
    _require_exact(
        root.get("collection"),
        {
            "benchmark_seed": BENCHMARK_SEED,
            "proposal_seed": PROPOSAL_SEED,
            "preflight_environment_seed_base": PREFLIGHT_ENVIRONMENT_SEED_BASE,
            "final_environment_seed_base": FINAL_ENVIRONMENT_SEED_BASE,
            "environment_seed_task_stride": ENVIRONMENT_SEED_TASK_STRIDE,
            "camera": "task_registered_a4_camera",
            "width": 224,
            "height": 224,
            "horizon_steps": _a4.HORIZON,
            "raw_steps_per_model_step": _a4.RAW_STEPS_PER_MODEL_STEP,
            "eligible_candidate_definition": "at_least_one_eligible_proposal",
            "target_eligible_candidates_per_task": (
                TARGET_ELIGIBLE_CANDIDATES_PER_TASK
            ),
            "selection": "first_eligible_in_preregistered_order",
            "future_or_outcome_candidate_filtering_allowed": False,
        },
        label="A4b collection",
    )
    old_collection = _mapping(
        source_contract.get("collection"), label="A4 v2 collection"
    )
    if BENCHMARK_SEED == int(old_collection["benchmark_seed"]):
        raise ValueError("A4b benchmark seed must be fresh relative to A4 v2")
    old_seed_base = int(old_collection["environment_seed_base"])
    old_environment_seeds = {
        old_seed_base + task_index * len(RAW_TASK_INDICES) + raw_index
        for task_index in range(len(TASKS))
        for raw_index in RAW_TASK_INDICES
    }
    new_environment_seeds = {
        candidate_environment_seed(pool=pool, task_index=task_index, rank=rank)
        for pool, count in (
            (POOL_PREFLIGHT, PREFLIGHT_CANDIDATES_PER_TASK),
            (POOL_FINAL, FINAL_CANDIDATES_PER_TASK),
        )
        for task_index in range(len(TASKS))
        for rank in range(count)
    }
    if old_environment_seeds & new_environment_seeds:
        raise ValueError("A4b environment seeds overlap A4 v2")
    _require_exact(
        root.get("proposal_schedule"),
        source_contract.get("proposal_schedule"),
        label="A4b proposal schedule",
    )
    _require_exact(
        root.get("eligibility_source"),
        _old_eligibility_projection(source_contract),
        label="A4b eligibility",
    )
    _require_exact(
        root.get("preflight_pool"),
        {
            "pool": POOL_PREFLIGHT,
            "attempt_all_candidates": True,
            "materialize_branch_bundles": False,
            "final_pool_access_allowed": False,
            "complete_result_required_before_final": True,
        },
        label="A4b preflight pool",
    )
    _require_exact(
        root.get("final_pool"),
        {
            "pool": POOL_FINAL,
            "candidate_limit_per_task": FINAL_CANDIDATES_PER_TASK,
            "target_eligible_candidates_per_task": (
                TARGET_ELIGIBLE_CANDIDATES_PER_TASK
            ),
            "selection": "first_eligible_in_preregistered_order",
            "stop_after_target_per_task": True,
            "complete_preflight_result_required": True,
            "complete_action": "write_exclusive_success_result",
            "insufficient_candidate_action": "write_exclusive_failure_result",
            "partial_cache_publish_allowed": False,
        },
        label="A4b final pool",
    )
    _require_exact(
        root.get("endpoint_reuse"),
        {
            "source_protocol": SOURCE_A4_PROTOCOL,
            "training_seeds": list(CONFIRMATION_TRAINING_SEEDS),
            "checkpoint_step": 10_000,
            "selection": "fixed_latest",
            "reuse_frozen_endpoints_allowed": True,
            "endpoint_retraining_for_a4b_allowed": False,
            "formal_train_cache_reused": True,
            "development_checkpoint_or_optimizer_reused": False,
        },
        label="A4b endpoint reuse",
    )
    _require_exact(
        root.get("confirmation_gate"),
        _old_confirmation_projection(source_contract),
        label="A4b confirmation Gate",
    )
    return TASKS


def candidate_order(
    split_manifest: Mapping[str, Any], *, pool: str, task: str
) -> tuple[int, ...]:
    validated = validate_a4b_split_manifest(split_manifest)
    if pool not in POOLS:
        raise ValueError(f"A4b pool must be one of {POOLS}")
    if task not in TASKS:
        raise ValueError(f"A4b task must be one of {TASKS}")
    return validated[task][pool]


def candidate_environment_seed(*, pool: str, task_index: int, rank: int) -> int:
    if pool not in POOLS:
        raise ValueError(f"A4b pool must be one of {POOLS}")
    if isinstance(task_index, bool) or not isinstance(task_index, Integral):
        raise TypeError("A4b task_index must be an integer")
    if isinstance(rank, bool) or not isinstance(rank, Integral):
        raise TypeError("A4b candidate rank must be an integer")
    task_value = int(task_index)
    rank_value = int(rank)
    limit = (
        PREFLIGHT_CANDIDATES_PER_TASK
        if pool == POOL_PREFLIGHT
        else FINAL_CANDIDATES_PER_TASK
    )
    if not 0 <= task_value < len(TASKS):
        raise ValueError("A4b task_index must be in 0..6")
    if not 0 <= rank_value < limit:
        raise ValueError(f"A4b {pool} candidate rank must be in 0..{limit - 1}")
    base = (
        PREFLIGHT_ENVIRONMENT_SEED_BASE
        if pool == POOL_PREFLIGHT
        else FINAL_ENVIRONMENT_SEED_BASE
    )
    return base + task_value * ENVIRONMENT_SEED_TASK_STRIDE + rank_value


def candidate_episode_uid(
    *, pool: str, task: str, rank: int, raw_task_index: int, environment_seed: int
) -> str:
    if pool not in POOLS or task not in TASKS:
        raise ValueError("invalid A4b pool or task identity")
    if int(raw_task_index) not in RAW_TASK_INDICES:
        raise ValueError("A4b raw task index must be in 0..47")
    return (
        f"{TASK_IDENTITY_NAMESPACE}:benchmark-{BENCHMARK_SEED}:{pool}:{task}:"
        f"rank-{int(rank):02d}:raw-{int(raw_task_index):02d}:"
        f"env-{int(environment_seed)}"
    )


def sample_a4b_proposal_schedule(
    task: str,
    raw_task_index: int,
    *,
    pool: str,
    candidate_rank: int,
    contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> tuple[int, ...]:
    """Draw the A4-identical schedule in A4b's frozen seed namespace."""

    validate_a4b_contract(
        contract, source_contract=source_contract, split_manifest=split_manifest
    )
    if int(raw_task_index) != candidate_order(
        split_manifest, pool=pool, task=task
    )[int(candidate_rank)]:
        raise ValueError("A4b raw task index differs from its candidate rank")
    proposal = _mapping(
        contract.get("proposal_schedule"), label="A4b proposal schedule"
    )
    count = int(proposal["proposals_per_episode"])
    minimum = int(proposal["minimum_raw_step"])
    maximum = int(proposal["maximum_raw_step_inclusive"])
    separation = int(proposal["minimum_separation_raw_steps"])
    compressed_maximum = maximum - (separation - 1) * (count - 1)
    population = np.arange(minimum, compressed_maximum + 1)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                PROPOSAL_SEED,
                zlib.crc32(SPLIT_MANIFEST_ID.encode("utf-8")),
                zlib.crc32(pool.encode("utf-8")),
                zlib.crc32(task.encode("utf-8")),
                int(raw_task_index),
                int(candidate_rank),
            ]
        )
    )
    compressed = np.sort(rng.choice(population, size=count, replace=False))
    expanded = compressed + (separation - 1) * np.arange(count)
    schedule = tuple(int(value) for value in expanded)
    if (
        len(schedule) != _a4.PROPOSALS_PER_EPISODE
        or schedule[0] < _a4.PROPOSAL_MINIMUM_RAW_STEP
        or schedule[-1] > _a4.PROPOSAL_MAXIMUM_RAW_STEP
        or any(
            right - left < _a4.PROPOSAL_MINIMUM_SEPARATION
            for left, right in zip(schedule, schedule[1:])
        )
    ):
        raise RuntimeError("internal A4b proposal schedule construction failed")
    return schedule


def branch_seeds(
    *,
    pool: str,
    task_index: int,
    raw_task_index: int,
    candidate_rank: int,
    proposal_step: int,
) -> tuple[int, int]:
    values = np.random.SeedSequence(
        [
            PROPOSAL_SEED,
            zlib.crc32(SPLIT_MANIFEST_ID.encode("utf-8")),
            zlib.crc32(pool.encode("utf-8")),
            int(task_index),
            int(raw_task_index),
            int(candidate_rank),
            int(proposal_step),
        ]
    ).generate_state(2, dtype=np.uint32)
    return int(values[0]), int(values[1])


__all__ = [
    "BENCHMARK_SEED",
    "CONFIRMATION_TRAINING_SEEDS",
    "CUSTOM_TASK_PARAMETER_SEED",
    "FINAL_CANDIDATES_PER_TASK",
    "POOL_FINAL",
    "POOL_PREFLIGHT",
    "PREFLIGHT_CANDIDATES_PER_TASK",
    "PROTOCOL",
    "RAW_TASK_INDICES",
    "SOURCE_A4_PROTOCOL",
    "SPLIT_MANIFEST_ID",
    "SPLIT_SEED",
    "TARGET_ELIGIBLE_CANDIDATES_PER_TASK",
    "TASK_IDENTITY_NAMESPACE",
    "TASKS",
    "branch_seeds",
    "candidate_environment_seed",
    "candidate_episode_uid",
    "candidate_order",
    "sample_a4b_proposal_schedule",
    "validate_a4b_contract",
    "validate_a4b_split_manifest",
]
