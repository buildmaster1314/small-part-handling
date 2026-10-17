from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fcpg.so101_realtime_planner as planner_module
from fcpg.so101_camera_calibration import ProjectionFit
from fcpg.so101_realtime_planner import (
    PreparedPlannerConfig,
    PreparedSO101RealtimePlanner,
    bounded_column_servo_target_xyz,
    cartesian_terminal_goal_cost,
    column_live_pixel_to_base_xyz,
)
from fcpg.so101_realtime_mpc import ManipulationPhase


def test_realtime_planner_defaults_to_target_conditioned_control() -> None:
    config = PreparedPlannerConfig(
        bridge_checkpoint=Path("bridge.pt"),
        calibration=Path("calibration.npz"),
        secondary_cache=Path("secondary.npz"),
    )

    assert config.control_mode == "phase_target"


def _planner(monkeypatch: pytest.MonkeyPatch) -> PreparedSO101RealtimePlanner:
    nominal = np.asarray(
        [
            [[100.0, 80.0], [120.0, 80.0], [80.0, 80.0]],
            [[110.0, 80.0], [120.0, 80.0], [100.0, 80.0]],
        ],
        dtype=np.float32,
    )

    def project_history(*_args, **_kwargs):
        return (
            np.zeros((2, 3, 3), dtype=np.float32),
            {"wrist": nominal.copy(), "column": nominal.copy()},
            {
                "wrist": np.ones((2, 3), dtype=bool),
                "column": np.ones((2, 3), dtype=bool),
            },
        )

    monkeypatch.setattr(planner_module, "project_history", project_history)
    planner = PreparedSO101RealtimePlanner.__new__(PreparedSO101RealtimePlanner)
    planner.fits = object()
    planner.kinematics = object()
    planner.live_anchor_offsets_px = {}
    planner.live_anchor_transforms_px = {}
    planner.model_contract = {
        "live_anchor_offsets_px": {},
        "live_anchor_transforms_px": {},
    }
    return planner


def test_live_geometry_selects_moving_tip_track_and_recovers_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _planner(monkeypatch)
    angle = np.deg2rad(10.0)
    linear = 0.75 * np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    translation = np.asarray([12.0, -7.0], dtype=np.float32)
    nominal_track = np.asarray([[80.0, 80.0], [100.0, 80.0]], dtype=np.float32)
    observed_track = nominal_track @ linear.T + translation
    candidates = np.asarray(
        [
            [[40.0, 20.0], [50.0, 20.0]],
            observed_track,
        ],
        dtype=np.float32,
    )

    result = planner.set_live_moving_tip_track(
        "column",
        candidates,
        np.asarray([0, 0, 0, 0, 0, 80], dtype=np.float32),
        np.asarray([0, 0, 0, 0, 0, 5], dtype=np.float32),
    )

    assert result["candidate_index"] == 1
    assert result["scale"] == pytest.approx(0.75)
    assert result["rotation_deg"] == pytest.approx(10.0)
    np.testing.assert_allclose(result["observed_moving_tip_track_px"], observed_track)
    np.testing.assert_allclose(result["nominal_moving_tip_track_px"], nominal_track)
    np.testing.assert_allclose(
        planner.live_anchor_transforms_px["column"][:, :2], linear, atol=1.0e-6
    )
    np.testing.assert_allclose(
        planner.live_anchor_transforms_px["column"][:, 2], translation, atol=2.0e-5
    )
    assert "column" not in planner.live_anchor_offsets_px


def test_live_geometry_fails_closed_when_no_candidate_is_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _planner(monkeypatch)
    candidates = np.asarray(
        [
            [[10.0, 10.0], [20.0, 10.0]],
            [[180.0, 180.0], [210.0, 180.0]],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="no automatic moving-tip track fit"):
        planner.set_live_moving_tip_track(
            "column",
            candidates,
            np.asarray([0, 0, 0, 0, 0, 80], dtype=np.float32),
            np.asarray([0, 0, 0, 0, 0, 5], dtype=np.float32),
        )

    assert planner.live_anchor_transforms_px == {}


def test_live_column_target_backprojection_undoes_similarity() -> None:
    fit = ProjectionFit(
        "column",
        ("x_base", "y_base", "z_base", "bias"),
        np.asarray(
            [[100.0, 0.0], [0.0, 100.0], [0.0, 0.0], [10.0, 20.0]]
        ),
        0.0,
    )
    transform = np.asarray(
        [[0.8, -0.2, 14.0], [0.2, 0.8, -9.0]], dtype=np.float32
    )
    calibration_pixel = np.asarray([30.0, 10.0], dtype=np.float32)
    live_pixel = np.concatenate((calibration_pixel, [1.0])) @ transform.T

    xyz = column_live_pixel_to_base_xyz(
        fit,
        live_pixel,
        z_base=-0.02,
        live_transform_px=transform,
    )

    np.testing.assert_allclose(xyz, [0.2, -0.1, -0.02], atol=1.0e-6)


def test_bounded_column_servo_uses_relative_pixels_and_standoff() -> None:
    fit = ProjectionFit(
        "column",
        ("x_base", "y_base", "z_base", "bias"),
        np.asarray(
            [[100.0, 0.0], [0.0, 100.0], [0.0, 0.0], [10.0, 20.0]]
        ),
        0.0,
    )
    transform = np.asarray(
        [[0.8, -0.2, 14.0], [0.2, 0.8, -9.0]], dtype=np.float32
    )
    current_calibration_px = np.asarray([30.0, 10.0], dtype=np.float32)
    target_calibration_px = np.asarray([36.0, 10.0], dtype=np.float32)
    current_live_px = np.concatenate((current_calibration_px, [1.0])) @ transform.T
    target_live_px = np.concatenate((target_calibration_px, [1.0])) @ transform.T

    planning, relative, error = bounded_column_servo_target_xyz(
        fit,
        current_live_px,
        target_live_px,
        np.asarray([0.15, 0.02, 0.03], dtype=np.float32),
        target_z_base=-0.02,
        max_planar_step_m=0.04,
        descent_radius_m=0.05,
        standoff_z_base=0.04,
        live_transform_px=transform,
    )

    assert error == pytest.approx(0.06, abs=1.0e-6)
    np.testing.assert_allclose(relative, [0.21, 0.02, -0.02], atol=1.0e-6)
    np.testing.assert_allclose(planning, [0.19, 0.02, 0.04], atol=1.0e-6)


def test_bounded_column_servo_descends_only_inside_local_radius() -> None:
    fit = ProjectionFit(
        "column",
        ("x_base", "y_base", "z_base", "bias"),
        np.asarray(
            [[100.0, 0.0], [0.0, 100.0], [0.0, 0.0], [10.0, 20.0]]
        ),
        0.0,
    )

    planning, relative, error = bounded_column_servo_target_xyz(
        fit,
        np.asarray([30.0, 10.0], dtype=np.float32),
        np.asarray([33.0, 10.0], dtype=np.float32),
        np.asarray([0.15, 0.02, 0.03], dtype=np.float32),
        target_z_base=-0.02,
        max_planar_step_m=0.04,
        descent_radius_m=0.05,
        standoff_z_base=0.04,
    )

    assert error == pytest.approx(0.03, abs=1.0e-6)
    np.testing.assert_allclose(relative, [0.18, 0.02, -0.02], atol=1.0e-6)
    np.testing.assert_allclose(planning, relative, atol=1.0e-6)


def test_cartesian_goal_cost_prefers_terminal_prediction_near_target() -> None:
    predicted = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.19, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    cost = cartesian_terminal_goal_cost(
        predicted,
        np.asarray([0.2, 0.0, 0.0]),
        final_steps=1,
        scale_m=0.2,
    )

    assert cost[1] < cost[0]


def test_demonstration_progress_advances_only_one_completed_chunk() -> None:
    planner = PreparedSO101RealtimePlanner.__new__(PreparedSO101RealtimePlanner)
    planner.config = SimpleNamespace(command_chunk=5)
    planner._demonstration_progress = None
    first_plan = SimpleNamespace(
        phase=ManipulationPhase.TRAJECTORY,
        selected_index=0,
        candidate_rows=[
            {
                "source": "implicit_successful_demonstration",
                "episode": 3,
                "frame": 10,
            }
        ],
    )

    first = planner.commit_executed_plan(first_plan, command_count=5)

    assert first == {
        "episode": 3,
        "executed_source_frame": 10,
        "next_source_frame": 15,
        "completed_chunks": 1,
    }
    assert planner._demonstration_progress.episode == 3
    assert planner._demonstration_progress.next_frame == 15

    continuation = SimpleNamespace(
        phase=ManipulationPhase.TRAJECTORY,
        selected_index=0,
        candidate_rows=[
            {
                "source": "locked_demonstration_continuation",
                "episode": 3,
                "frame": 15,
            }
        ],
    )
    second = planner.commit_executed_plan(continuation, command_count=5)
    assert second["next_source_frame"] == 20
    assert second["completed_chunks"] == 2


def test_demonstration_progress_rejects_a_backward_or_switched_candidate() -> None:
    planner = PreparedSO101RealtimePlanner.__new__(PreparedSO101RealtimePlanner)
    planner.config = SimpleNamespace(command_chunk=5)
    planner._demonstration_progress = SimpleNamespace(
        episode=3,
        next_frame=15,
        completed_chunks=1,
    )
    bad_plan = SimpleNamespace(
        phase=ManipulationPhase.TRAJECTORY,
        selected_index=0,
        candidate_rows=[
            {
                "source": "locked_demonstration_continuation",
                "episode": 3,
                "frame": 5,
            }
        ],
    )

    with pytest.raises(ValueError, match="does not match committed progress"):
        planner.commit_executed_plan(bad_plan, command_count=5)


def test_task_progress_reset_forgets_demonstration_alignment() -> None:
    planner = PreparedSO101RealtimePlanner.__new__(PreparedSO101RealtimePlanner)
    planner._demonstration_progress = SimpleNamespace(
        episode=3,
        next_frame=15,
        completed_chunks=1,
    )

    planner.reset_task_progress()

    assert planner._demonstration_progress is None
