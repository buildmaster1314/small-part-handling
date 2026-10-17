from __future__ import annotations

from copy import deepcopy

import pytest

from fcpg.lewm_gate_l1 import summarize_gate_l1


TASKS = ("pick-place-v3", "button-press-v3", "reach-v3", "push-v3")
CONDITIONS = ("clean", "rotation5", "translation4", "rotation10")


def _report(
    condition: str,
    train: str,
    evaluation: str,
    ratio: float,
) -> dict[str, object]:
    blocks = []
    for task in TASKS:
        for episode in range(6):
            blocks.append(
                {
                    "task_name": task,
                    "episode_id": episode,
                    "endpoints": {
                        str(horizon): {
                            "predicted_mse": ratio * horizon,
                            "hold_last_identity_mse": float(horizon),
                        }
                        for horizon in (1, 3, 5)
                    },
                }
            )
    representation = {
        "effective_rank": 20.0 if train == "nominal" else 19.0,
        "participation_ratio": 10.0 if train == "nominal" else 9.5,
    }
    return {
        "grounding_mode": evaluation,
        "checkpoint_sha256": f"checkpoint-{train}",
        "oracle_target_cache": [
            {
                "array_sha256": f"target-{condition}",
                "encoder_checkpoint_sha256": "shared-encoder",
            }
        ],
        "metrics": {
            "episode_blocks": blocks,
            "representation": {
                "overall": representation,
                "per_task": {task: representation for task in TASKS},
            },
            "h10_stress": {
                "enabled": condition == "rotation10",
                "ratio": ratio,
                "part_of_primary_gate": False,
            },
        },
    }


def _matrix() -> dict[str, object]:
    matrix = {}
    for condition in CONDITIONS:
        matrix[condition] = {}
        for train in ("nominal", "fcpg"):
            matrix[condition][train] = {}
            for evaluation in ("nominal", "fcpg"):
                if condition == "clean":
                    values = {
                        ("nominal", "nominal"): 0.50,
                        ("nominal", "fcpg"): 0.49,
                        ("fcpg", "nominal"): 0.55,
                        ("fcpg", "fcpg"): 0.51,
                    }
                else:
                    values = {
                        ("nominal", "nominal"): 0.80,
                        ("nominal", "fcpg"): 0.60,
                        ("fcpg", "nominal"): 0.75,
                        ("fcpg", "fcpg"): 0.55,
                    }
                matrix[condition][train][evaluation] = _report(
                    condition,
                    train,
                    evaluation,
                    values[(train, evaluation)],
                )
    return matrix


def test_gate_l1_go_requires_full_paired_evidence_chain() -> None:
    result = summarize_gate_l1(_matrix(), bootstrap_samples=500, bootstrap_seed=7)

    assert result["decision"] == "Go"
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["primary"]["relative_improvement"] > 0.10
    assert result["primary"]["bootstrap"]["difference_95_ci"][0] > 0.0


def test_gate_l1_no_go_when_matched_fcpg_is_worse() -> None:
    matrix = _matrix()
    for condition in ("rotation5", "translation4"):
        matrix[condition]["fcpg"]["fcpg"] = _report(
            condition, "fcpg", "fcpg", 0.85
        )

    result = summarize_gate_l1(matrix, bootstrap_samples=200, bootstrap_seed=8)

    assert result["decision"] == "No-Go"
    assert result["checks"]["relative_improvement_at_least_10_percent"] is False
    assert result["checks"]["matched_diagonal_direction_agrees"] is False


def test_gate_l1_rejects_non_common_hold_last_target() -> None:
    matrix = _matrix()
    tampered = deepcopy(matrix["rotation5"]["fcpg"]["nominal"])
    tampered["metrics"]["episode_blocks"][0]["endpoints"]["1"][
        "hold_last_identity_mse"
    ] = 2.0
    matrix["rotation5"]["fcpg"]["nominal"] = tampered

    with pytest.raises(ValueError, match="hold-last targets differ"):
        summarize_gate_l1(matrix, bootstrap_samples=10)
