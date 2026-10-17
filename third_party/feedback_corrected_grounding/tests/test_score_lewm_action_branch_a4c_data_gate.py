from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.score_lewm_action_branch_a4c_data_gate import build_a4c_data_gate


def _preflight() -> dict[str, object]:
    return {
        "mode": "formal",
        "input_contains_test_rows": False,
        "checks": {
            "environment_and_expert_rollout": True,
            "snapshot_restore": True,
            "zero_severe_penetration": True,
            "fcpg_anchor_smoke": True,
            "active_control_support": True,
            "physical_support": True,
            "latent_support": True,
            "task_state_sufficiency": False,
        },
        "task_state_sufficiency": {
            "contextualized_state_gate": {
                "passed": True,
                "comparison": {
                    "relative_improvement": 0.14,
                    "nonnegative_task_count": 6,
                },
            }
        },
    }


def _gripper_support(*, passed: bool = True) -> dict[str, object]:
    return {
        "protocol": "fcpg_action_branch_a4c_gripper_support_v1",
        "passed": passed,
        "failed_groups": [] if passed else ["drawer-retrieve-v0/train/grasp"],
    }


def test_contextualized_state_gate_can_authorize_a4c_2k() -> None:
    report = build_a4c_data_gate(_preflight(), _gripper_support())

    assert report["decision"] == "Data-Go"
    assert report["training_authorized"]
    assert not report["reserved_test_access_authorized"]
    assert report["legacy_nested_summary_increment"]["passed"] is False
    assert report["checks"]["task_state_sufficiency"] is True


def test_other_preflight_failure_remains_blocking() -> None:
    value = _preflight()
    value["checks"]["physical_support"] = False

    report = build_a4c_data_gate(value, _gripper_support())

    assert report["decision"] == "Data-No-Go"
    assert not report["training_authorized"]
    assert report["failed_checks"] == ["physical_support"]


@pytest.mark.parametrize(
    "mutation, error",
    [
        (lambda value: value.update(mode="smoke"), ValueError),
        (lambda value: value.update(input_contains_test_rows=True), PermissionError),
        (
            lambda value: value["task_state_sufficiency"].pop(
                "contextualized_state_gate"
            ),
            ValueError,
        ),
    ],
)
def test_data_gate_fails_closed(mutation, error) -> None:
    value = deepcopy(_preflight())
    mutation(value)

    with pytest.raises(error):
        build_a4c_data_gate(value, _gripper_support())


def test_gripper_support_failure_blocks_training() -> None:
    report = build_a4c_data_gate(_preflight(), _gripper_support(passed=False))

    assert report["decision"] == "Data-No-Go"
    assert report["failed_checks"] == ["gripper_intervention_support"]
