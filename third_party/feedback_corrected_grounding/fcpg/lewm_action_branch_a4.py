"""Fail-closed active-control preflight protocol for Action-Branch A4.

A4 proposals are scheduled before an episode starts.  This module decides
whether a proposal can become a branch bundle using only values available at
the proposal step or earlier.  Future branch measurements and outcomes are
deliberately outside the eligibility input schema.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


PROTOCOL = "fcpg_action_branch_active_control_a4_v2"
STATUS = "preflight_draft"
TASKS = (
    "pick-place-v3",
    "drawer-retrieve-v0",
    "reach-v3",
    "push-v3",
    "door-open-v3",
    "window-open-v3",
    "towel-wipe-v0",
)

FORMAL_TASK_INSTANCES = tuple(range(48))
SPLIT_SEED = 30_103
SPLIT_COUNTS = (("train", 24), ("development", 12), ("test", 12))

HISTORY_STEPS = 3
RECENT_ACTION_WINDOW_RAW_STEPS = 5
MINIMUM_EXPERT_XYZ_NORM = 0.05
MINIMUM_RECENT_ACTION_RMS = 0.05
MINIMUM_WORKSPACE_AND_JOINT_MARGIN = 0.05
BRANCH_COUNT = 8
HORIZON = 8
RAW_STEPS_PER_MODEL_STEP = 5
BRANCH_HORIZON_RAW_STEPS = HORIZON * RAW_STEPS_PER_MODEL_STEP

BUNDLE_BATCH_SIZE = 28
BUNDLES_PER_TASK_PER_BATCH = 4
DEVELOPMENT_TRAINING_SEEDS = (30_201, 30_202, 30_203)
CONFIRMATION_TRAINING_SEEDS = (30_301, 30_302, 30_303)
SIGREG_WEIGHT = 0.02

PROPOSALS_PER_EPISODE = 12
PROPOSAL_MINIMUM_RAW_STEP = 3
PROPOSAL_MAXIMUM_RAW_STEP = 160
PROPOSAL_MINIMUM_SEPARATION = 5

ELIGIBILITY_INPUT_FIELDS = (
    "proposal",
    "complete_history",
    "success",
    "ever_success_before",
    "active_stage",
    "expert_xyz_norm",
    "recent_action_rms",
    "workspace_and_joint_margin",
    "unexpected_collision",
    "remaining_raw_steps",
)
FUTURE_OR_OUTCOME_FIELDS = (
    "future_latent_variance",
    "future_physical_spread",
    "future_reward",
    "branch_success",
)

_RESULT_KEYS = frozenset(
    {
        *FUTURE_OR_OUTCOME_FIELDS,
        "actual_return",
        "observed_success",
        "decision",
        "execution_result",
        "failed_checks",
        "observed_result",
        "passed",
        "result",
        "results",
    }
)
_SEMANTIC_BRANCHES = (
    "expert",
    "reverse",
    "hold",
    "half_speed",
    "transverse_u_positive",
    "transverse_u_negative",
    "transverse_v_positive",
    "transverse_v_negative",
)
_PREFLIGHT_GATES = (
    "environment_and_expert_rollout",
    "snapshot_restore",
    "zero_severe_penetration",
    "fcpg_anchor_smoke",
    "active_control_support",
    "physical_support",
    "latent_support",
    "task_state_sufficiency",
)


@dataclass(frozen=True)
class ActiveControlEligibility:
    """Auditable acceptance decision for one pre-scheduled proposal."""

    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.eligible == bool(self.rejection_reasons):
            raise ValueError(
                "eligible decisions require no rejection reasons and rejected "
                "decisions require at least one"
            )

    def as_record(self) -> dict[str, Any]:
        """Return the JSON-compatible fields persisted for every proposal."""

        return {
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
        }


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


def _require_exact(
    section: Mapping[str, Any],
    key: str,
    expected: object,
    *,
    label: str,
) -> None:
    if key not in section or not _same_typed_json(section[key], expected):
        raise ValueError(f"{label} {key!r} drifted from the frozen A4 contract")


def _require_exact_keys(
    section: Mapping[str, Any], expected: Mapping[str, object], *, label: str
) -> None:
    actual_keys = set(section)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{label} keys drifted; missing={missing}, unexpected={unexpected}"
        )
    for key, value in expected.items():
        _require_exact(section, key, value, label=label)


def _result_paths(value: object, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            current = (*path, key)
            if key in _RESULT_KEYS:
                found.append(".".join(current))
            found.extend(_result_paths(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_result_paths(item, (*path, str(index))))
    return found


def _strict_bool(value: object, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be boolean")
    return bool(value)


def _nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def validate_a4_contract(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the frozen, result-free A4 preflight contract."""

    contract = _mapping(contract, "contract")
    result_paths = _result_paths(contract)
    if result_paths:
        raise ValueError(
            "A4 preflight draft must not contain observed result fields: "
            + ", ".join(result_paths)
        )
    top_level_expected = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": STATUS,
        "question": (
            "Can one shared seven-task LeWM propagate action-conditioned robot "
            "anchors and task context from genuinely active-control states?"
        ),
        "tasks": list(TASKS),
    }
    required_sections = {
        "split",
        "collection",
        "proposal_schedule",
        "eligibility",
        "matched_state_branches",
        "preflight",
        "support_gate",
        "task_state_sufficiency_gate",
        "training_if_all_preflight_gates_go",
        "development_gate",
        "confirmation_after_development_go",
    }
    actual_top_keys = set(contract)
    expected_top_keys = set(top_level_expected) | required_sections
    if actual_top_keys != expected_top_keys:
        raise ValueError("A4 top-level keys drifted from the frozen contract")
    for key, value in top_level_expected.items():
        _require_exact(contract, key, value, label="A4")

    split = _mapping(contract.get("split"), "split")
    split_expected = {
        "protocol_seed": SPLIT_SEED,
        "assignment": (
            "a4-task-local-fresh-24-train-12-development-12-reserved-test"
        ),
        "unit": "source_physical_episode",
        "train_episodes_per_task": 24,
        "development_episodes_per_task": 12,
        "test_episodes_per_task": 12,
        "task_local": True,
        "episode_disjoint": True,
        "a3_episode_reuse_allowed": False,
    }
    _require_exact_keys(split, split_expected, label="A4 split")

    collection = _mapping(contract.get("collection"), "collection")
    collection_expected = {
        "benchmark_seed": SPLIT_SEED,
        "environment_seed_base": 620_000,
        "camera": "corner",
        "width": 224,
        "height": 224,
        "horizon_steps": HORIZON,
        "raw_steps_per_model_step": RAW_STEPS_PER_MODEL_STEP,
        "test_instances_not_collected_before_development_go": True,
    }
    _require_exact_keys(collection, collection_expected, label="A4 collection")

    proposal = _mapping(contract.get("proposal_schedule"), "proposal_schedule")
    proposal_expected = {
        "drawn_before_environment_reset": True,
        "independent_of_state_action_reward_task_phase_and_outcomes": True,
        "sampling": "uniform_without_replacement",
        "proposals_per_episode": 12,
        "minimum_raw_step": 3,
        "maximum_raw_step_inclusive": 160,
        "minimum_separation_raw_steps": 5,
    }
    _require_exact_keys(proposal, proposal_expected, label="A4 proposal schedule")

    eligibility = _mapping(contract.get("eligibility"), "eligibility")
    eligibility_expected = {
        "allowed_inputs": list(ELIGIBILITY_INPUT_FIELDS),
        "history_steps": HISTORY_STEPS,
        "minimum_expert_xyz_norm": MINIMUM_EXPERT_XYZ_NORM,
        "recent_action_window_raw_steps": RECENT_ACTION_WINDOW_RAW_STEPS,
        "minimum_recent_action_rms": MINIMUM_RECENT_ACTION_RMS,
        "minimum_workspace_and_joint_margin": (
            MINIMUM_WORKSPACE_AND_JOINT_MARGIN
        ),
        "branch_horizon_raw_steps": BRANCH_HORIZON_RAW_STEPS,
        "record_every_proposal": True,
        "collect_all_rejection_reasons": True,
        "future_or_outcome_fields_for_task_level_support_only": list(
            FUTURE_OR_OUTCOME_FIELDS
        ),
        "future_or_outcome_row_filtering_allowed": False,
    }
    _require_exact_keys(eligibility, eligibility_expected, label="A4 eligibility")

    branches = _mapping(
        contract.get("matched_state_branches"), "matched_state_branches"
    )
    branch_expected = {
        "branch_count": BRANCH_COUNT,
        "interruption_horizon_steps": HORIZON,
        "raw_steps_per_model_step": RAW_STEPS_PER_MODEL_STEP,
        "interruption_horizon_raw_steps": BRANCH_HORIZON_RAW_STEPS,
        "semantic_bank_before_permutation": list(_SEMANTIC_BRANCHES),
        "all_branches_executed_and_saved_after_acceptance": True,
        "slot_order_randomized_per_snapshot": True,
    }
    _require_exact_keys(branches, branch_expected, label="A4 branch")

    preflight = _mapping(contract.get("preflight"), "preflight")
    _require_exact_keys(
        preflight,
        {"required_gates_before_training": list(_PREFLIGHT_GATES)},
        label="A4 preflight",
    )

    support = _mapping(contract.get("support_gate"), "support_gate")
    support_expected = {
        "evaluation_split": "train",
        "minimum_active_bundle_episode_count_per_task": 18,
        "maximum_accepted_post_success_fraction": 0.0,
        "maximum_accepted_near_zero_action_fraction": 0.0,
        "maximum_invalid_branch_fraction": 0.0,
        "aggregation": "task_then_physical_episode",
        "source_expert_success_role": "dataset_quality_metric_only",
        "future_or_outcome_usage": "task_level_support_gate_only",
    }
    _require_exact_keys(support, support_expected, label="A4 support Gate")

    task_state = _mapping(
        contract.get("task_state_sufficiency_gate"),
        "task_state_sufficiency_gate",
    )
    task_state_expected = {
        "candidates": [
            "anchor_only",
            "summary_task_context_plus_anchors",
            "global_visual_plus_anchors_reference",
        ],
        "required_readouts": [
            "object_pose",
            "goal_relative_progress",
            "contact_or_grasp_state",
            "branch_outcome",
        ],
        "current_state_gate_readouts": [
            "object_pose",
            "goal_relative_progress",
            "contact_or_grasp_state",
        ],
        "future_diagnostic_readouts": ["branch_outcome"],
        "shared_deployable_context": [
            "task_onehot",
            "proprio",
            "anchor_validity",
            "history_actions",
            "candidate_actions",
        ],
        "fit_split": "train",
        "evaluation_split": "development",
        "readout": "closed_form_ridge",
        "neural_optimizer_steps": 0,
        "bootstrap_unit": "source_physical_episode",
        "bootstrap_samples": 10_000,
        "minimum_summary_relative_improvement": 0.05,
        "minimum_nonnegative_tasks": 6,
        "privileged_labels_role": "offline_targets_only",
        "reserved_test_access_allowed": False,
        "task_dynamics_claim_requires_task_context_sufficiency": True,
    }
    _require_exact_keys(task_state, task_state_expected, label="A4 task-state Gate")

    training = _mapping(
        contract.get("training_if_all_preflight_gates_go"),
        "training_if_all_preflight_gates_go",
    )
    training_expected = {
        "optimizer_steps": 2_000,
        "seeds": list(DEVELOPMENT_TRAINING_SEEDS),
        "models_per_seed": 1,
        "model_scope": "one_shared_model_across_all_seven_tasks",
        "tasks_per_model": len(TASKS),
        "task_balanced": True,
        "episode_balanced_within_task": True,
        "bundle_batch_size": BUNDLE_BATCH_SIZE,
        "bundles_per_task_per_batch": BUNDLES_PER_TASK_PER_BATCH,
        "amp_dtype": "float16",
        "anchor_encoder_sigreg": SIGREG_WEIGHT,
        "sigreg_values_allowed": [SIGREG_WEIGHT],
        "additional_sigreg_values_allowed": [],
        "fresh_start_from_frozen_base": True,
        "selection": "fixed_latest",
    }
    _require_exact_keys(training, training_expected, label="A4 training")
    if (
        int(training["bundles_per_task_per_batch"]) * len(TASKS)
        != int(training["bundle_batch_size"])
    ):
        raise ValueError("A4 batch must contain four bundles from each task")

    development = _mapping(contract.get("development_gate"), "development_gate")
    development_expected = {
        "evaluation_split": "development",
        "checkpoint_step": 2_000,
        "primary_horizons": [6, 7, 8],
        "primary_metric": "relative_prediction_mse_improvement",
        "minimum_relative_improvement": 0.1,
        "paired_ci_lower_must_exceed": 0.0,
        "bootstrap_unit": "source_physical_episode",
        "bootstrap_samples": 10_000,
        "task_macro": True,
        "minimum_positive_tasks": 6,
        "minimum_positive_seeds": 2,
        "must_beat_matched_no_action": True,
        "must_beat_all_seven_wrong_action_shifts": True,
        "must_beat_hold_last_and_constant_velocity": True,
    }
    _require_exact_keys(development, development_expected, label="A4 development Gate")

    confirmation = _mapping(
        contract.get("confirmation_after_development_go"),
        "confirmation_after_development_go",
    )
    confirmation_expected = {
        "test_collection_requires_development_go": True,
        "optimizer_steps": 10_000,
        "seeds": list(CONFIRMATION_TRAINING_SEEDS),
        "fresh_start_from_same_frozen_base": True,
        "development_checkpoint_or_optimizer_reused": False,
        "models_per_seed": 1,
        "model_scope": "one_shared_model_across_all_seven_tasks",
        "tasks_per_model": len(TASKS),
        "task_balanced": True,
        "episode_balanced_within_task": True,
        "bundle_batch_size": BUNDLE_BATCH_SIZE,
        "bundles_per_task_per_batch": BUNDLES_PER_TASK_PER_BATCH,
        "amp_dtype": "float16",
        "anchor_encoder_sigreg": SIGREG_WEIGHT,
        "sigreg_values_allowed": [SIGREG_WEIGHT],
        "additional_sigreg_values_allowed": [],
        "selection": "fixed_latest",
        "evaluation_split": "test",
        "checkpoint_step": 10_000,
        "primary_horizons": [6, 7, 8],
        "primary_metric": "relative_prediction_mse_improvement",
        "minimum_relative_improvement": 0.1,
        "paired_ci_lower_must_exceed": 0.0,
        "bootstrap_unit": "source_physical_episode",
        "bootstrap_samples": 10_000,
        "task_macro": True,
        "minimum_positive_tasks": 6,
        "minimum_positive_seeds": 2,
        "must_beat_matched_no_action": True,
        "must_beat_all_seven_wrong_action_shifts": True,
        "must_beat_hold_last_and_constant_velocity": True,
    }
    _require_exact_keys(confirmation, confirmation_expected, label="A4 confirmation")
    return TASKS


def assign_a4_splits(
    tasks: Sequence[str], contract: Mapping[str, Any]
) -> dict[tuple[str, int], str]:
    """Assign a fresh, deterministic 24/12/12 split per A4 task."""

    configured_tasks = validate_a4_contract(contract)
    supplied_tasks = tuple(tasks)
    if supplied_tasks != configured_tasks:
        raise ValueError(f"A4 split tasks must remain {configured_tasks}")
    split = _mapping(contract.get("split"), "split")
    seed = int(split["protocol_seed"])
    namespace = str(split["assignment"])
    result: dict[tuple[str, int], str] = {}
    for task in configured_tasks:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    seed,
                    zlib.crc32(namespace.encode("utf-8")),
                    zlib.crc32(task.encode("utf-8")),
                ]
            )
        )
        ranked = rng.permutation(FORMAL_TASK_INSTANCES).tolist()
        offset = 0
        for split_name, count in SPLIT_COUNTS:
            for instance in ranked[offset : offset + count]:
                result[(task, int(instance))] = split_name
            offset += count
    return result


def sample_a4_proposal_schedule(
    task_name: str,
    task_instance: int,
    *,
    contract: Mapping[str, Any],
) -> tuple[int, ...]:
    """Reconstruct the uniform pre-reset proposal schedule for one episode.

    There is a bijection between sorted proposal sets with minimum separation
    ``d`` and unconstrained sorted sets after subtracting ``(d - 1) * i`` from
    the item at rank ``i``.  Sampling in that compressed space therefore gives
    an exactly uniform draw over all valid proposal sets without rejection or
    any dependence on environment state.
    """

    configured_tasks = validate_a4_contract(contract)
    if not isinstance(task_name, str) or task_name not in configured_tasks:
        raise ValueError(f"task_name must be one of the frozen A4 tasks: {configured_tasks}")
    if isinstance(task_instance, (bool, np.bool_)) or not isinstance(
        task_instance, Integral
    ):
        raise TypeError("task_instance must be an integer")
    instance = int(task_instance)
    if instance not in FORMAL_TASK_INSTANCES:
        raise ValueError("task_instance must be in the inclusive range 0..47")

    proposal = _mapping(contract.get("proposal_schedule"), "proposal_schedule")
    count = int(proposal["proposals_per_episode"])
    minimum = int(proposal["minimum_raw_step"])
    maximum = int(proposal["maximum_raw_step_inclusive"])
    separation = int(proposal["minimum_separation_raw_steps"])
    if (
        count != PROPOSALS_PER_EPISODE
        or minimum != PROPOSAL_MINIMUM_RAW_STEP
        or maximum != PROPOSAL_MAXIMUM_RAW_STEP
        or separation != PROPOSAL_MINIMUM_SEPARATION
    ):
        raise ValueError("A4 proposal schedule constants drifted")

    compressed_maximum = maximum - (separation - 1) * (count - 1)
    compressed_population = np.arange(minimum, compressed_maximum + 1)
    if compressed_population.size < count:
        raise ValueError("A4 proposal schedule has no feasible combination")

    split = _mapping(contract.get("split"), "split")
    namespace = str(split["assignment"])
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(split["protocol_seed"]),
                zlib.crc32(namespace.encode("utf-8")),
                zlib.crc32(task_name.encode("utf-8")),
                instance,
            ]
        )
    )
    compressed = np.sort(
        rng.choice(compressed_population, size=count, replace=False)
    )
    expanded = compressed + (separation - 1) * np.arange(count)
    schedule = tuple(int(value) for value in expanded)
    if (
        len(schedule) != count
        or schedule[0] < minimum
        or schedule[-1] > maximum
        or any(
            right - left < separation
            for left, right in zip(schedule, schedule[1:])
        )
    ):
        raise RuntimeError("internal A4 proposal schedule construction failed")
    return schedule


def evaluate_active_control_eligibility(
    inputs: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> ActiveControlEligibility:
    """Evaluate one proposal from the exact current/history input schema.

    Extra fields are rejected before any value is read.  This prevents future
    branch spread, rewards, or outcomes from becoming accidental row filters.
    Valid proposals receive all applicable rejection reasons in conjunction
    order, rather than only the first failure.
    """

    validate_a4_contract(contract)
    values = _mapping(inputs, "eligibility inputs")
    names = frozenset(values)
    expected = frozenset(ELIGIBILITY_INPUT_FIELDS)
    unexpected = sorted(str(name) for name in names - expected)
    missing = [name for name in ELIGIBILITY_INPUT_FIELDS if name not in names]
    if unexpected:
        raise ValueError(
            "A4 eligibility accepts only registered current/past fields; "
            f"unexpected fields: {unexpected}"
        )
    if missing:
        raise ValueError(f"A4 eligibility inputs are missing fields: {missing}")

    proposal = _strict_bool(values["proposal"], "proposal")
    complete_history = _strict_bool(
        values["complete_history"], "complete_history"
    )
    success = _strict_bool(values["success"], "success")
    ever_success_before = _strict_bool(
        values["ever_success_before"], "ever_success_before"
    )
    active_stage = _strict_bool(values["active_stage"], "active_stage")
    expert_xyz_norm = _nonnegative_number(
        values["expert_xyz_norm"], "expert_xyz_norm"
    )
    recent_action_rms = _nonnegative_number(
        values["recent_action_rms"], "recent_action_rms"
    )
    margin = _nonnegative_number(
        values["workspace_and_joint_margin"], "workspace_and_joint_margin"
    )
    unexpected_collision = _strict_bool(
        values["unexpected_collision"], "unexpected_collision"
    )
    remaining_raw_steps = _nonnegative_integer(
        values["remaining_raw_steps"], "remaining_raw_steps"
    )

    eligibility = _mapping(contract.get("eligibility"), "eligibility")
    reasons: list[str] = []
    checks = (
        (proposal, "not_a_pre_scheduled_proposal"),
        (complete_history, "incomplete_history"),
        (not success, "current_success"),
        (not ever_success_before, "success_observed_before_proposal"),
        (active_stage, "inactive_task_stage"),
        (
            expert_xyz_norm
            >= float(eligibility["minimum_expert_xyz_norm"]),
            "expert_xyz_norm_below_minimum",
        ),
        (
            recent_action_rms
            >= float(eligibility["minimum_recent_action_rms"]),
            "recent_action_rms_below_minimum",
        ),
        (
            margin
            >= float(eligibility["minimum_workspace_and_joint_margin"]),
            "workspace_or_joint_margin_below_minimum",
        ),
        (not unexpected_collision, "unexpected_collision"),
        (
            remaining_raw_steps
            >= int(eligibility["branch_horizon_raw_steps"]),
            "insufficient_remaining_raw_steps",
        ),
    )
    reasons.extend(reason for condition, reason in checks if not condition)
    return ActiveControlEligibility(not reasons, tuple(reasons))


__all__ = [
    "ActiveControlEligibility",
    "BRANCH_HORIZON_RAW_STEPS",
    "BUNDLE_BATCH_SIZE",
    "BUNDLES_PER_TASK_PER_BATCH",
    "ELIGIBILITY_INPUT_FIELDS",
    "FUTURE_OR_OUTCOME_FIELDS",
    "PROTOCOL",
    "PROPOSALS_PER_EPISODE",
    "PROPOSAL_MAXIMUM_RAW_STEP",
    "PROPOSAL_MINIMUM_RAW_STEP",
    "PROPOSAL_MINIMUM_SEPARATION",
    "SIGREG_WEIGHT",
    "STATUS",
    "TASKS",
    "assign_a4_splits",
    "evaluate_active_control_eligibility",
    "sample_a4_proposal_schedule",
    "validate_a4_contract",
]
