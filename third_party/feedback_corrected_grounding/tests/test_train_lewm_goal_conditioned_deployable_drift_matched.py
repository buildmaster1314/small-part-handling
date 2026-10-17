from __future__ import annotations

import pytest

from scripts.train_lewm_goal_conditioned_deployable_drift_matched import (
    _paired_summary,
    parse_axis,
    parse_signed_float_list,
)


def _metric(value: float) -> dict[str, object]:
    task = {
        "entity_epe_mm": value,
        "spearman_mean": 1.0 - value / 100.0,
        "selected_actual_goal_distance_h8_mm": value + 5.0,
    }
    return {
        "entity_epe_mm": value,
        "spearman_mean": 1.0 - value / 100.0,
        "top1_action_accuracy": 0.5,
        "selected_actual_goal_distance_h8_mm": value + 5.0,
        "by_task": {"reach-v3": dict(task), "push-v3": dict(task)},
    }


def test_signed_rotation_parser_accepts_symmetric_drift() -> None:
    assert parse_signed_float_list("-10,-5,0,5,10", name="rotation") == (
        -10.0,
        -5.0,
        0.0,
        5.0,
        10.0,
    )
    with pytest.raises(ValueError, match="duplicates"):
        parse_signed_float_list("0,0", name="rotation")


def test_axis_parser_normalizes_vector() -> None:
    axis = parse_axis("0,2,0")
    assert axis.tolist() == [0.0, 1.0, 0.0]


def test_paired_summary_uses_matched_model_and_grounding() -> None:
    metrics = {
        "models": {
            "nominal": {
                "0.0": {"nominal": _metric(10.0), "fcpg": _metric(9.0)},
                "5.0": {"nominal": _metric(20.0), "fcpg": _metric(19.0)},
            },
            "fcpg": {
                "0.0": {"nominal": _metric(12.0), "fcpg": _metric(8.0)},
                "5.0": {"nominal": _metric(18.0), "fcpg": _metric(15.0)},
            },
        }
    }
    summary = _paired_summary(metrics, rotations=(0.0, 5.0))
    assert summary["drifted_mean_entity_epe_improvement_mm"] == pytest.approx(5.0)
    assert summary["drifted_mean_spearman_delta"] == pytest.approx(0.05)
    assert summary["drifted_mean_selected_h8_improvement_mm"] == pytest.approx(5.0)
