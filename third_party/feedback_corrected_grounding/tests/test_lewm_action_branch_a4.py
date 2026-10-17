from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from fcpg.lewm_action_branch_a4 import (
    BRANCH_HORIZON_RAW_STEPS,
    BUNDLE_BATCH_SIZE,
    BUNDLES_PER_TASK_PER_BATCH,
    ELIGIBILITY_INPUT_FIELDS,
    FUTURE_OR_OUTCOME_FIELDS,
    SIGREG_WEIGHT,
    STATUS,
    TASKS,
    assign_a4_splits,
    evaluate_active_control_eligibility,
    sample_a4_proposal_schedule,
    validate_a4_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"


def _contract() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _eligible_inputs() -> dict[str, Any]:
    return {
        "proposal": True,
        "complete_history": True,
        "success": False,
        "ever_success_before": False,
        "active_stage": True,
        "expert_xyz_norm": 0.05,
        "recent_action_rms": 0.05,
        "workspace_and_joint_margin": 0.05,
        "unexpected_collision": False,
        "remaining_raw_steps": BRANCH_HORIZON_RAW_STEPS,
    }


def _decision(**updates: Any):
    inputs = _eligible_inputs()
    inputs.update(updates)
    return evaluate_active_control_eligibility(inputs, contract=_contract())


def _nested_result_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = (*path, str(key))
            if key in {
                "decision",
                "execution_result",
                "failed_checks",
                "observed_result",
                "passed",
                "result",
                "results",
            }:
                found.append(".".join(current))
            found.extend(_nested_result_paths(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_nested_result_paths(item, (*path, str(index))))
    return found


def test_contract_is_result_free_preflight_draft_with_exact_tasks() -> None:
    contract = _contract()

    assert validate_a4_contract(contract) == TASKS
    assert contract["status"] == STATUS == "preflight_draft"
    assert tuple(contract["tasks"]) == TASKS
    assert TASKS[-1] == "towel-wipe-v0"
    assert "sweep-v3" not in TASKS
    assert _nested_result_paths(contract) == []

    false_result = deepcopy(contract)
    false_result["result"] = False
    with pytest.raises(ValueError, match="must not contain observed result"):
        validate_a4_contract(false_result)

    for forbidden in (*FUTURE_OR_OUTCOME_FIELDS, "observed_success"):
        contaminated = deepcopy(contract)
        contaminated[forbidden] = 1
        with pytest.raises(ValueError, match="must not contain observed result"):
            validate_a4_contract(contaminated)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("collection", "camera"),
        ("proposal_schedule", "proposals_per_episode"),
        ("development_gate", "primary_horizons"),
        ("development_gate", "minimum_relative_improvement"),
        ("confirmation_after_development_go", "selection"),
        ("confirmation_after_development_go", "evaluation_split"),
    ],
)
def test_contract_rejects_missing_scientific_fields(
    section: str, field: str
) -> None:
    contract = _contract()
    contract[section].pop(field)

    with pytest.raises(ValueError, match="keys drifted"):
        validate_a4_contract(contract)


def test_contract_rejects_unregistered_extra_fields() -> None:
    contract = _contract()
    contract["development_gate"]["choose_best_checkpoint"] = True

    with pytest.raises(ValueError, match="keys drifted"):
        validate_a4_contract(contract)


@pytest.mark.parametrize(
    "tasks",
    [
        [*TASKS[:-1]],
        [*TASKS, "basketball-v3"],
        ["drawer-open-v3", *TASKS[1:]],
        [TASKS[1], TASKS[0], *TASKS[2:]],
    ],
)
def test_contract_rejects_any_task_panel_drift(tasks: list[str]) -> None:
    contract = _contract()
    contract["tasks"] = tasks

    with pytest.raises(ValueError, match="tasks.*drifted"):
        validate_a4_contract(contract)


def test_split_is_fresh_deterministic_task_local_24_12_12() -> None:
    contract = _contract()
    first = assign_a4_splits(TASKS, contract)
    repeated = assign_a4_splits(TASKS, contract)

    assert first == repeated
    assert len(first) == len(TASKS) * 48
    for task in TASKS:
        labels = [first[(task, instance)] for instance in range(48)]
        assert labels.count("train") == 24
        assert labels.count("development") == 12
        assert labels.count("test") == 12

    drifted = deepcopy(contract)
    drifted["split"]["development_episodes_per_task"] = 11
    with pytest.raises(ValueError, match="A4 split"):
        validate_a4_contract(drifted)


def test_proposal_schedule_is_deterministic_pre_reset_and_respects_constraints() -> None:
    contract = _contract()
    schedules: dict[tuple[str, int], tuple[int, ...]] = {}
    for task in TASKS:
        for instance in (0, 1, 47):
            schedule = sample_a4_proposal_schedule(
                task, instance, contract=contract
            )
            assert schedule == sample_a4_proposal_schedule(
                task, instance, contract=contract
            )
            assert len(schedule) == 12
            assert schedule == tuple(sorted(schedule))
            assert schedule[0] >= 3
            assert schedule[-1] <= 160
            assert all(
                right - left >= 5
                for left, right in zip(schedule, schedule[1:])
            )
            schedules[(task, instance)] = schedule

    assert len(set(schedules.values())) > len(TASKS)


@pytest.mark.parametrize("bad_instance", [-1, 48, True, 1.5])
def test_proposal_schedule_rejects_invalid_task_instance(bad_instance: Any) -> None:
    error = TypeError if isinstance(bad_instance, (bool, float)) else ValueError
    with pytest.raises(error):
        sample_a4_proposal_schedule(
            TASKS[0], bad_instance, contract=_contract()
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (
            "training_if_all_preflight_gates_go",
            "models_per_seed",
            7,
        ),
        (
            "training_if_all_preflight_gates_go",
            "model_scope",
            "one_model_per_task",
        ),
        (
            "training_if_all_preflight_gates_go",
            "tasks_per_model",
            1,
        ),
        (
            "training_if_all_preflight_gates_go",
            "task_balanced",
            False,
        ),
        (
            "training_if_all_preflight_gates_go",
            "bundle_batch_size",
            27,
        ),
        (
            "training_if_all_preflight_gates_go",
            "bundles_per_task_per_batch",
            3,
        ),
        (
            "training_if_all_preflight_gates_go",
            "anchor_encoder_sigreg",
            0.0,
        ),
        (
            "training_if_all_preflight_gates_go",
            "sigreg_values_allowed",
            [0.0, 0.02],
        ),
        (
            "training_if_all_preflight_gates_go",
            "additional_sigreg_values_allowed",
            [0.0],
        ),
        (
            "confirmation_after_development_go",
            "model_scope",
            "one_model_per_task",
        ),
        (
            "confirmation_after_development_go",
            "tasks_per_model",
            1,
        ),
        (
            "confirmation_after_development_go",
            "bundle_batch_size",
            21,
        ),
        (
            "confirmation_after_development_go",
            "additional_sigreg_values_allowed",
            [0.02],
        ),
    ],
)
def test_contract_locks_shared_balanced_model_and_only_sigreg_002(
    section: str, field: str, value: Any
) -> None:
    contract = _contract()
    contract[section][field] = value

    with pytest.raises(ValueError, match="A4 (training|confirmation)"):
        validate_a4_contract(contract)


def test_batch_is_exactly_four_bundles_from_each_of_seven_tasks() -> None:
    contract = _contract()
    training = contract["training_if_all_preflight_gates_go"]

    assert BUNDLE_BATCH_SIZE == 28
    assert BUNDLES_PER_TASK_PER_BATCH == 4
    assert BUNDLES_PER_TASK_PER_BATCH * len(TASKS) == BUNDLE_BATCH_SIZE
    assert training["bundle_batch_size"] == BUNDLE_BATCH_SIZE
    assert training["sigreg_values_allowed"] == [SIGREG_WEIGHT] == [0.02]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidates", ["anchor_only", "summary_task_context_plus_anchors"]),
        ("fit_split", "development"),
        ("evaluation_split", "test"),
        ("readout", "tiny_mlp"),
        ("neural_optimizer_steps", 1),
        ("minimum_summary_relative_improvement", 0.0),
        ("minimum_nonnegative_tasks", 4),
        ("privileged_labels_role", "deployment_input"),
        ("reserved_test_access_allowed", True),
    ],
)
def test_task_state_gate_is_zero_training_and_keeps_test_sealed(
    field: str, value: Any
) -> None:
    contract = _contract()
    contract["task_state_sufficiency_gate"][field] = value

    with pytest.raises(ValueError, match="A4 task-state Gate"):
        validate_a4_contract(contract)


def test_all_checks_accept_at_the_frozen_inclusive_boundaries() -> None:
    decision = _decision()

    assert decision.eligible
    assert decision.rejection_reasons == ()
    assert decision.as_record() == {
        "eligible": True,
        "rejection_reasons": [],
    }


def test_incomplete_history_is_rejected_with_reason() -> None:
    decision = _decision(complete_history=False)

    assert not decision.eligible
    assert decision.rejection_reasons == ("incomplete_history",)


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"success": True}, "current_success"),
        (
            {"ever_success_before": True},
            "success_observed_before_proposal",
        ),
    ],
)
def test_current_or_ever_success_is_rejected(
    updates: dict[str, bool], reason: str
) -> None:
    decision = _decision(**updates)

    assert not decision.eligible
    assert decision.rejection_reasons == (reason,)


def test_inactive_task_stage_is_rejected() -> None:
    decision = _decision(active_stage=False)

    assert decision.rejection_reasons == ("inactive_task_stage",)


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("expert_xyz_norm", "expert_xyz_norm_below_minimum"),
        ("recent_action_rms", "recent_action_rms_below_minimum"),
    ],
)
def test_near_zero_current_or_recent_action_is_rejected(
    field: str, reason: str
) -> None:
    decision = _decision(**{field: 0.049999})

    assert not decision.eligible
    assert decision.rejection_reasons == (reason,)


def test_workspace_or_joint_margin_below_minimum_is_rejected() -> None:
    decision = _decision(workspace_and_joint_margin=0.049999)

    assert decision.rejection_reasons == (
        "workspace_or_joint_margin_below_minimum",
    )


def test_unexpected_collision_is_rejected() -> None:
    decision = _decision(unexpected_collision=True)

    assert decision.rejection_reasons == ("unexpected_collision",)


def test_remaining_horizon_requires_all_40_raw_steps() -> None:
    rejected = _decision(remaining_raw_steps=BRANCH_HORIZON_RAW_STEPS - 1)
    accepted = _decision(remaining_raw_steps=BRANCH_HORIZON_RAW_STEPS)

    assert rejected.rejection_reasons == ("insufficient_remaining_raw_steps",)
    assert accepted.eligible
    assert BRANCH_HORIZON_RAW_STEPS == 40


def test_nonproposal_and_all_failed_conjuncts_are_auditable() -> None:
    decision = _decision(
        proposal=False,
        complete_history=False,
        success=True,
        ever_success_before=True,
        active_stage=False,
        expert_xyz_norm=0.0,
        recent_action_rms=0.0,
        workspace_and_joint_margin=0.0,
        unexpected_collision=True,
        remaining_raw_steps=0,
    )

    assert decision.rejection_reasons == (
        "not_a_pre_scheduled_proposal",
        "incomplete_history",
        "current_success",
        "success_observed_before_proposal",
        "inactive_task_stage",
        "expert_xyz_norm_below_minimum",
        "recent_action_rms_below_minimum",
        "workspace_or_joint_margin_below_minimum",
        "unexpected_collision",
        "insufficient_remaining_raw_steps",
    )


@pytest.mark.parametrize("field", FUTURE_OR_OUTCOME_FIELDS)
def test_future_branch_and_outcome_fields_fail_closed(field: str) -> None:
    inputs = _eligible_inputs()
    inputs[field] = 0.0

    assert set(ELIGIBILITY_INPUT_FIELDS).isdisjoint(FUTURE_OR_OUTCOME_FIELDS)
    with pytest.raises(ValueError, match="only registered current/past fields"):
        evaluate_active_control_eligibility(inputs, contract=_contract())


def test_missing_unknown_and_malformed_inputs_cannot_become_eligible() -> None:
    missing = _eligible_inputs()
    missing.pop("success")
    with pytest.raises(ValueError, match="missing fields"):
        evaluate_active_control_eligibility(missing, contract=_contract())

    unknown = {**_eligible_inputs(), "unregistered_side_channel": 1}
    with pytest.raises(ValueError, match="current/past fields"):
        evaluate_active_control_eligibility(unknown, contract=_contract())

    malformed = _eligible_inputs()
    malformed["recent_action_rms"] = float("nan")
    with pytest.raises(ValueError, match="finite and non-negative"):
        evaluate_active_control_eligibility(malformed, contract=_contract())
