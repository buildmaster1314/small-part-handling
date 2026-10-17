from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_lewm_recursive_grounding_a4 import (
    METHODS,
    _centers_for_step,
    linear_camera_rotation_degrees,
    summarize_rows,
)


def test_linear_camera_rotation_is_bounded_and_piecewise_linear() -> None:
    values = [
        linear_camera_rotation_degrees(
            step,
            start_raw_step=10,
            end_raw_step=50,
            maximum_degrees=10.0,
        )
        for step in (0, 10, 20, 30, 50, 80)
    ]
    assert values == pytest.approx([0.0, 0.0, 2.5, 5.0, 10.0, 10.0])


def test_recursive_centers_use_lewm_prior_and_temporal_baselines() -> None:
    nominal = np.full((3, 2), 10.0, dtype=np.float32)
    previous = np.ones((3, 2), dtype=np.float32)
    latest = np.full((3, 2), 3.0, dtype=np.float32)
    prior = np.full((3, 2), 7.0, dtype=np.float32)
    history = {method: [] for method in METHODS}
    history["hold_last"] = [latest]
    history["constant_velocity"] = [previous, latest]

    kalman = np.full((3, 2), 5.0, dtype=np.float32)
    centers = _centers_for_step(nominal, history, prior, kalman)

    np.testing.assert_allclose(centers["one_shot"], nominal)
    np.testing.assert_allclose(centers["hold_last"], nominal + latest)
    np.testing.assert_allclose(
        centers["constant_velocity"], nominal + 2.0 * latest - previous
    )
    np.testing.assert_allclose(centers["linear_kalman"], nominal + kalman)
    np.testing.assert_allclose(centers["lewm_recursive"], prior)


def test_summary_reports_severe_recursive_improvement() -> None:
    rows = []
    for task in ("pick-place-v3", "towel-wipe-v0"):
        for method, error in {
            "one_shot": 20.0,
            "hold_last": 14.0,
            "constant_velocity": 12.0,
            "linear_kalman": 10.0,
            "lewm_recursive": 5.0,
        }.items():
            rows.append(
                {
                    "task": task,
                    "camera_rotation_degrees": 10.0,
                    "method": method,
                    "center_epe_px": error + 2.0,
                    "corrected_epe_px": error,
                }
            )

    summary = summarize_rows(rows)

    assert summary["severe_recursive_vs_one_shot_relative_improvement"] == pytest.approx(
        0.75
    )
    assert summary["methods"]["lewm_recursive"]["severe_capture_at_5px"] == 1.0
    assert summary["severe_evaluable_task_count"] == 2
    assert summary["severe_unevaluable_tasks"] == []
