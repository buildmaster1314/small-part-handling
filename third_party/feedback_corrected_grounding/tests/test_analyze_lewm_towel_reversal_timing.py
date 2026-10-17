from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_lewm_towel_reversal_timing as analysis


def _points(values: list[float]) -> np.ndarray:
    centers = np.asarray([[value, 2.0 * value] for value in values], dtype=np.float64)
    offsets = np.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
    return centers[:, None, :] + offsets[None]


def test_wipe_axis_is_oriented_toward_right_motion() -> None:
    truth = _points([0.0, 1.0, 2.0, 1.0, 0.0])
    stages = ("wipe_right", "wipe_right", "wipe_left", "wipe_left", "wipe_left")
    axis = analysis.wipe_axis(truth, stages)
    centers = truth.mean(axis=1)
    assert float((centers[1] - centers[0]) @ axis) > 0.0


def test_wipe_transitions_only_returns_direction_changes() -> None:
    stages = (
        "approach_towel",
        "wipe_right",
        "wipe_right",
        "wipe_left",
        "wipe_left",
        "lift_after_wipe",
    )
    assert analysis.wipe_transitions(stages) == (
        (3, "wipe_right", "wipe_left"),
    )


def test_reversal_rows_detects_early_short_forecast() -> None:
    starts = np.asarray((0,), dtype=np.int64)
    sampled_truth = np.asarray([0, 1, 2, 3, 4, 5, 4, 3, 2], dtype=np.float64)
    truth_values = np.interp(
        np.arange(41, dtype=np.float64),
        np.arange(0, 41, 5, dtype=np.float64),
        sampled_truth,
    ).tolist()
    truth = _points(truth_values)
    current = truth[[0]]
    predicted = _points([1, 2, 3, 3.5, 3.0, 2.5, 2.0, 1.5])[None]
    stages = tuple("wipe_right" if step < 25 else "wipe_left" for step in range(41))
    rows = analysis.reversal_rows(
        starts=starts,
        current_points=current,
        predicted_points=predicted,
        truth_points=truth,
        stages=stages,
        minimum_lead_steps=5,
    )
    assert len(rows) == 1
    assert rows[0]["prediction_timing_error_steps"] == -5
    assert rows[0]["predicted_to_true_endpoint_ratio"] == pytest.approx(0.7)


def test_cuda1_is_rejected() -> None:
    with pytest.raises(ValueError, match="CUDA device 0"):
        analysis._cuda0("cuda:1")


def test_model_label_distinguishes_checkpoints_at_same_step(tmp_path: Path) -> None:
    left = analysis._model_label(tmp_path / "base" / "model.pt", {"step": 2000})
    right = analysis._model_label(
        tmp_path / "history" / "model.pt", {"step": 2000}
    )
    assert left.endswith("base@step_2000")
    assert right.endswith("history@step_2000")
    assert left != right
