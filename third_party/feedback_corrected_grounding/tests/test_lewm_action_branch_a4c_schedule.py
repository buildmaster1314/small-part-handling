from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from fcpg.lewm_action_branch_a4c_schedule import (
    PROTOCOL,
    REQUIRED_PHASE_IDS,
    a4c_base_contract,
    choose_phase_schedule,
    load_a4c_contract,
    load_schedule_map,
    make_schedule_sampler,
    validate_a4c_contract,
    validate_schedule_manifest,
)
from fcpg.lewm_action_branch_a4 import FORMAL_TASK_INSTANCES, assign_a4_splits


ROOT = Path(__file__).resolve().parents[1]


def _trajectory():
    rows = []
    for step in range(100):
        rows.append(
            {
                "raw_step": step,
                "success": step >= 65,
                "progress": min(step / 65.0, 1.0),
                "contact": step >= 35,
                "phase": "approach" if step < 35 else "contact",
                "action_xyz_norm": 0.2,
                "workspace_control_margin": 0.5,
                "unexpected_collision": False,
                "signal_error": None,
            }
        )
    return {
        "task": "pick-place-v3",
        "rows": rows,
        "first_success_step": 65,
        "trajectory_action_count": 100,
    }


def _formal_manifest(contract: dict) -> dict:
    tasks = tuple(contract["tasks"])
    assignments = assign_a4_splits(tasks, a4c_base_contract(contract))
    episodes = []
    seed_base = int(contract["collection"]["environment_seed_base"])
    for task_index, task in enumerate(tasks):
        for instance in FORMAL_TASK_INSTANCES:
            split = assignments[(task, instance)]
            if split not in {"train", "development"}:
                continue
            steps = list(range(10, 70, 5))
            required = list(REQUIRED_PHASE_IDS.get(task, ()))
            phases = [*required]
            phases.extend(
                f"{task}_phase_{index}"
                for index in range(len(phases), len(steps))
            )
            episodes.append(
                {
                    "task": task,
                    "task_class_index": task_index,
                    "task_instance": instance,
                    "split": split,
                    "environment_seed": seed_base + task_index * 48 + instance,
                    "proposal_steps": steps,
                    "first_success_step": 70,
                    "trajectory_action_count": 110,
                    "coverage": {
                        "candidate_count": len(steps),
                        "selected_count": len(steps),
                        "selected_phase_names": phases,
                        "selected_unique_phase_count": len(set(phases)),
                        "required_phase_ids": list(required),
                        "covered_required_phase_ids": list(required),
                        "missing_required_phase_ids": [],
                        "source_success": True,
                        "h8_crosses_success": True,
                        "h8_crosses_phase_transition": True,
                        "status": "ok",
                    },
                    "signal_fallback_count": 0,
                }
            )
    proposal = contract["proposal_schedule"]
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "planned",
        "tasks": list(tasks),
        "schedule_contract": {
            "schedule_materialized_before_branch_execution": True,
            "proposal_count_per_episode": 12,
            "minimum_raw_step": 10,
            "minimum_separation_raw_steps": 5,
            "branch_horizon_raw_steps": 40,
            "uses_branch_outcomes": False,
            "uses_expert_replay_for_collection_design": True,
            "required_phase_ids": proposal["required_phase_ids"],
            "require_success_crossing_for_successful_source": True,
        },
        "collection": {
            "benchmark_seed": contract["collection"]["benchmark_seed"],
            "environment_seed_base": seed_base,
            "width": 224,
            "height": 224,
            "horizon_steps": 8,
            "raw_steps_per_model_step": 5,
        },
        "episodes": episodes,
    }


def test_phase_schedule_is_sorted_and_crosses_success():
    schedule, coverage = choose_phase_schedule(_trajectory(), count=8)
    assert schedule == tuple(sorted(set(schedule)))
    assert 6 <= len(schedule) <= 8
    assert coverage["h8_crosses_success"]
    assert coverage["h8_crosses_phase_transition"]


def test_phase_schedule_excludes_multi_axis_saturation_and_collision():
    trajectory = _trajectory()
    trajectory["rows"][20]["workspace_control_margin"] = 0.0
    trajectory["rows"][40]["unexpected_collision"] = True

    schedule, _coverage = choose_phase_schedule(trajectory, count=20)

    assert 20 not in schedule
    assert 40 not in schedule


def test_success_boundary_is_reserved_before_many_phase_transitions():
    trajectory = _trajectory()
    trajectory["first_success_step"] = 90
    for step, row in enumerate(trajectory["rows"]):
        row["success"] = step >= 90
        row["phase"] = f"phase_{step // 5:02d}"

    schedule, coverage = choose_phase_schedule(trajectory, count=8)

    assert coverage["h8_crosses_success"]
    assert any(step < 90 <= step + 40 for step in schedule)


def test_towel_schedule_preserves_all_six_half_cycles() -> None:
    rows = []
    cycle_phases = REQUIRED_PHASE_IDS["towel-wipe-v0"][:-1]
    for step in range(320):
        if step >= 280:
            phase = "success"
        elif step >= 245:
            phase = "lift_after_wipe"
        elif step >= 40:
            phase = cycle_phases[min((step - 40) // 30, 5)]
        else:
            phase = "lower_to_start"
        rows.append(
            {
                "raw_step": step,
                "success": step >= 280,
                "progress": min(step / 280.0, 1.0),
                "contact": 40 <= step < 245,
                "phase": phase,
                "action_xyz_norm": 0.2,
                "workspace_control_margin": 0.5,
                "unexpected_collision": False,
                "signal_error": None,
            }
        )
    trajectory = {
        "task": "towel-wipe-v0",
        "rows": rows,
        "first_success_step": 280,
        "trajectory_action_count": 320,
    }

    _schedule, coverage = choose_phase_schedule(trajectory, count=12)

    assert coverage["missing_required_phase_ids"] == []
    assert coverage["covered_required_phase_ids"] == list(
        REQUIRED_PHASE_IDS["towel-wipe-v0"]
    )


def test_schedule_map_and_sampler_round_trip():
    manifest = {
        "protocol": PROTOCOL,
        "episodes": [
            {"task": "pick-place-v3", "task_instance": 2, "proposal_steps": [10, 20]}
        ],
    }
    assert load_schedule_map(manifest) == {("pick-place-v3", 2): (10, 20)}
    sampler = make_schedule_sampler(manifest)
    assert sampler("pick-place-v3", 2, contract={}) == (10, 20)


def test_compact_a4c_contract_materializes_and_rejects_legacy_protocol():
    contract = load_a4c_contract(
        ROOT / "config/lewm/eval/action_branch_phase_aware_a4c.json"
    )
    assert validate_a4c_contract(contract) == tuple(contract["tasks"])
    legacy = json.loads(
        (ROOT / "config/lewm/eval/action_branch_active_control_a4.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(ValueError, match="protocol"):
        validate_a4c_contract(legacy)


def test_formal_schedule_manifest_matches_materialized_a4c_contract():
    contract = load_a4c_contract(
        ROOT / "config/lewm/eval/action_branch_phase_aware_a4c.json"
    )
    manifest = _formal_manifest(contract)
    validate_schedule_manifest(manifest, contract)
    sampler = make_schedule_sampler(manifest, contract=contract)
    assert sampler("drawer-retrieve-v0", 0, contract=contract)


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (
            lambda row: row.__setitem__("task_class_index", 999),
            "identity mismatch",
        ),
        (
            lambda row: row["coverage"].__setitem__("selected_count", 999),
            "selected_count mismatch",
        ),
        (
            lambda row: row["coverage"].__setitem__(
                "selected_phase_names", ["fabricated"] * len(row["proposal_steps"])
            ),
            "selected_unique_phase_count mismatch",
        ),
        (
            lambda row: row.__setitem__("first_success_step", -999),
            "out of range",
        ),
    ),
)
def test_schedule_manifest_rejects_semantic_mutation(mutator, message):
    contract = load_a4c_contract(
        ROOT / "config/lewm/eval/action_branch_phase_aware_a4c.json"
    )
    manifest = _formal_manifest(contract)
    mutated = deepcopy(manifest)
    mutator(mutated["episodes"][0])

    with pytest.raises((TypeError, ValueError), match=message):
        validate_schedule_manifest(mutated, contract)


def test_schedule_manifest_rejects_self_consistent_missing_towel_cycle() -> None:
    contract = load_a4c_contract(
        ROOT / "config/lewm/eval/action_branch_phase_aware_a4c.json"
    )
    manifest = _formal_manifest(contract)
    row = next(
        value
        for value in manifest["episodes"]
        if value["task"] == "towel-wipe-v0"
    )
    missing = REQUIRED_PHASE_IDS["towel-wipe-v0"][2]
    phases = row["coverage"]["selected_phase_names"]
    phases[phases.index(missing)] = "generic_wipe"
    row["coverage"]["selected_unique_phase_count"] = len(set(phases))
    row["coverage"]["covered_required_phase_ids"].remove(missing)
    row["coverage"]["missing_required_phase_ids"] = [missing]
    row["coverage"]["status"] = "sparse"

    with pytest.raises(ValueError, match="lacks required phases"):
        validate_schedule_manifest(manifest, contract)


def test_schedule_manifest_marks_failed_source_explicitly() -> None:
    contract = load_a4c_contract(
        ROOT / "config/lewm/eval/action_branch_phase_aware_a4c.json"
    )
    manifest = _formal_manifest(contract)
    row = manifest["episodes"][0]
    row["first_success_step"] = None
    row["coverage"]["source_success"] = False
    row["coverage"]["h8_crosses_success"] = False
    row["coverage"]["status"] = "source_failed"

    validate_schedule_manifest(manifest, contract)
