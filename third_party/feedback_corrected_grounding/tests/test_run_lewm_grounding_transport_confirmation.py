from __future__ import annotations

import numpy as np

from scripts import run_lewm_grounding_transport_confirmation as confirmation


def test_paired_bootstrap_reports_lower_right_error_as_positive() -> None:
    left = {
        "task-a": np.asarray([10.0, 12.0, 11.0]),
        "task-b": np.asarray([20.0, 18.0, 19.0]),
    }
    right = {
        "task-a": np.asarray([5.0, 6.0, 5.5]),
        "task-b": np.asarray([10.0, 9.0, 9.5]),
    }
    result = confirmation._paired_bootstrap(left, right, samples=1_000, seed=7)
    assert result["relative_improvement"] == 0.5
    assert result["relative_ci_low"] > 0.0
    assert result["positive_tasks"] == 2


def test_markdown_marks_clean_penalty_direction() -> None:
    comparison = {
        "left_task_macro_epe_px": 1.0,
        "right_task_macro_epe_px": 1.1,
        "absolute_improvement_px": -0.1,
        "absolute_ci_low_px": -0.2,
        "absolute_ci_high_px": 0.0,
        "relative_improvement": -0.1,
        "relative_ci_low": -0.2,
        "relative_ci_high": 0.0,
        "positive_tasks": 0,
        "task_count": 2,
    }
    document = confirmation._markdown(
        {
            "physical_episode_count": 2,
            "clock_hours": [1],
            "task_view_episode_cells": 2,
            "drift_gated_vs_nominal": comparison,
            "drift_current_vs_nominal": comparison,
            "clean_gated_vs_nominal": comparison,
        }
    )
    assert "negative value is a clean penalty" in document
