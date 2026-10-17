"""Tests for cross-seed action-chunk result aggregation."""

from __future__ import annotations

import csv
import json

import pytest

from scripts.summarize_action_chunk_seeds import (
    SeedPaths,
    render_markdown,
    summarize_seed_results,
)


def _write_offline(path, chunk_mse: float, first_action_mse: float) -> None:
    path.write_text(
        json.dumps(
            {
                "evaluated_sequences": 8,
                "chunk_mse": chunk_mse,
                "first_action_mse": first_action_mse,
                "gripper_accuracy": 0.9,
            }
        ),
        encoding="utf-8",
    )


def _write_rollout(path, success: list[int], corrected_error: float) -> None:
    fieldnames = [
        "task",
        "task_index",
        "env_seed",
        "success",
        "pixel_error_mean",
        "corrected_pixel_error_mean",
        "actual_camera_translation_x_cm",
        "actual_camera_translation_y_cm",
        "actual_camera_translation_z_cm",
        "actual_camera_rotation_axis_x",
        "actual_camera_rotation_axis_y",
        "actual_camera_rotation_axis_z",
        "actual_camera_rotation_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, value in enumerate(success):
            writer.writerow(
                {
                    "task": "pick-place-v3",
                    "task_index": index,
                    "env_seed": 1000 + index,
                    "success": value,
                    "pixel_error_mean": 10.0,
                    "corrected_pixel_error_mean": corrected_error,
                    "actual_camera_translation_x_cm": 0.0,
                    "actual_camera_translation_y_cm": 0.0,
                    "actual_camera_translation_z_cm": 0.0,
                    "actual_camera_rotation_axis_x": 0.0,
                    "actual_camera_rotation_axis_y": 0.0,
                    "actual_camera_rotation_axis_z": 1.0,
                    "actual_camera_rotation_deg": 5.0,
                }
            )


def test_summarize_seed_results_preserves_policy_seed_as_statistical_unit(tmp_path) -> None:
    specs = []
    for seed, nominal, fcpg in (
        (0, [0, 0, 0, 0], [1, 1, 0, 0]),
        (1, [0, 0, 0, 0], [1, 0, 0, 0]),
    ):
        geoprop_offline = tmp_path / f"geo_{seed}.json"
        fcpg_offline = tmp_path / f"fcpg_{seed}.json"
        nominal_rollout = tmp_path / f"nominal_{seed}.csv"
        fcpg_rollout = tmp_path / f"rollout_{seed}.csv"
        _write_offline(geoprop_offline, 0.02 + seed * 0.001, 0.03)
        _write_offline(fcpg_offline, 0.01 + seed * 0.001, 0.02)
        _write_rollout(nominal_rollout, nominal, corrected_error=10.0)
        _write_rollout(fcpg_rollout, fcpg, corrected_error=1.0)
        specs.append(
            SeedPaths(seed, geoprop_offline, fcpg_offline, nominal_rollout, fcpg_rollout)
        )

    summary = summarize_seed_results(specs, bootstrap_samples=2_000, bootstrap_seed=3)

    rollout = summary["aggregate"]["rollout"]
    assert rollout["success_rate_difference"]["mean"] == pytest.approx(0.375)
    assert rollout["success_rate_difference"]["sample_std"] == pytest.approx(0.1767767)
    assert rollout["hierarchical_bootstrap_95_ci"][0] >= 0.0
    assert summary["aggregate"]["offline"]["fcpg_chunk_mse"]["mean"] == pytest.approx(0.0105)

    markdown = render_markdown(summary, "Residual-only", "Full-FCPG")
    assert "Residual-only chunk MSE" in markdown
    assert "Full-FCPG rollout success" in markdown
