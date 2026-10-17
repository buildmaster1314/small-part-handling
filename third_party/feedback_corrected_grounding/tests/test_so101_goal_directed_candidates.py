import numpy as np

from fcpg.so101_camera_calibration import ProjectionFit
from fcpg.so101_goal_directed_candidates import (
    ImageServoVariant,
    generate_fcpg_approach_prefixes,
    generate_fcpg_goal_prefixes,
    generate_gripper_phase_prefixes,
)
from fcpg.so101_realtime_mpc import validate_command_chunk


class _LinearKinematics:
    def forward_anchors(self, states: np.ndarray) -> np.ndarray:
        state = np.asarray(states, dtype=np.float64)
        center = state[:3] / 100.0
        return np.stack((center, center + [0.0, 0.01, 0.0], center - [0.0, 0.01, 0.0]))


def _fit() -> ProjectionFit:
    return ProjectionFit(
        view="column",
        feature_names=("x_base", "y_base", "z_base", "bias"),
        coefficients=np.asarray(
            [[100.0, 0.0], [0.0, 100.0], [0.0, 0.0], [100.0, 100.0]],
            dtype=np.float64,
        ),
        ridge=0.0,
    )


def test_fcpg_servo_reduces_target_error_and_preserves_tool_commands() -> None:
    state = np.asarray([0.0, 0.0, 0.0, 7.0, -12.0, 18.0], dtype=np.float32)
    variants = (ImageServoVariant("test", 0.5, 0.2, 1.0, 0.0),)

    prefixes, reports = generate_fcpg_approach_prefixes(
        state,
        np.asarray([104.0, 97.0]),
        np.asarray([124.0, 77.0]),
        _fit(),
        _LinearKinematics(),
        contact_height_m=0.0,
        position_lower=np.asarray([-100.0] * 5 + [0.0]),
        position_upper=np.asarray([100.0] * 6),
        horizon_raw=40,
        variants=variants,
    )

    assert prefixes.shape == (1, 40, 6)
    assert reports[0]["kinematic_final_pixel_error"] < reports[0]["initial_pixel_error"]
    assert np.max(np.abs(np.diff(np.concatenate((state[None], prefixes[0])), axis=0))) <= 1.0 + 1e-6
    np.testing.assert_array_equal(prefixes[0, :, 4], state[4])
    np.testing.assert_array_equal(prefixes[0, :, 5], state[5])


def test_fcpg_offset_changes_the_servo_solution() -> None:
    state = np.zeros(6, dtype=np.float32)
    bounds = np.asarray([-100.0] * 5 + [0.0]), np.asarray([100.0] * 6)
    variants = (ImageServoVariant("test", 0.4, 0.5, 0.8, 0.0),)

    nominal, _ = generate_fcpg_approach_prefixes(
        state,
        np.asarray([100.0, 100.0]),
        np.asarray([115.0, 100.0]),
        _fit(),
        _LinearKinematics(),
        contact_height_m=0.0,
        position_lower=bounds[0],
        position_upper=bounds[1],
        horizon_raw=10,
        variants=variants,
    )
    corrected, _ = generate_fcpg_approach_prefixes(
        state,
        np.asarray([110.0, 100.0]),
        np.asarray([115.0, 100.0]),
        _fit(),
        _LinearKinematics(),
        contact_height_m=0.0,
        position_lower=bounds[0],
        position_upper=bounds[1],
        horizon_raw=10,
        variants=variants,
    )

    assert corrected[0, -1, 0] < nominal[0, -1, 0]


def test_fcpg_servo_projects_motion_away_from_guarded_joint_limits() -> None:
    state = np.asarray([-98.0, 0.0, 0.0, 7.0, -12.0, 18.0], dtype=np.float32)
    variants = (ImageServoVariant("test", 0.5, 0.2, 1.0, 0.0),)

    prefixes, reports = generate_fcpg_approach_prefixes(
        state,
        np.asarray([2.0, 100.0]),
        np.asarray([-10.0, 130.0]),
        _fit(),
        _LinearKinematics(),
        contact_height_m=0.0,
        position_lower=np.asarray([-100.0] * 5 + [0.0]),
        position_upper=np.asarray([100.0] * 6),
        horizon_raw=10,
        variants=variants,
    )

    assert np.all(prefixes[0, :, 0] >= state[0])
    assert prefixes[0, -1, 1] > state[1]
    assert reports[0]["kinematic_final_pixel_error"] < reports[0]["initial_pixel_error"]
    assert validate_command_chunk(prefixes[0, :5], state).safe


def test_fcpg_servo_uses_live_column_similarity_transform() -> None:
    state = np.asarray([10.0, 20.0, 0.0, 7.0, -12.0, 18.0], dtype=np.float32)
    live_transform = np.asarray(
        [[0.0, -1.0, 160.0], [1.0, 0.0, 20.0]], dtype=np.float64
    )
    current_live_center = np.asarray([40.0, 130.0], dtype=np.float32)
    variants = (ImageServoVariant("test", 0.5, 0.2, 1.0, 0.0),)

    prefixes, reports = generate_fcpg_goal_prefixes(
        state,
        current_live_center,
        current_live_center + np.asarray([0.0, 24.0], dtype=np.float32),
        _fit(),
        _LinearKinematics(),
        contact_height_m=0.0,
        position_lower=np.asarray([-100.0] * 5 + [0.0]),
        position_upper=np.asarray([100.0] * 6),
        horizon_raw=20,
        variants=variants,
        live_transform_px=live_transform,
    )

    assert prefixes[0, -1, 0] > state[0]
    assert abs(float(prefixes[0, -1, 1] - state[1])) < 1e-4
    assert reports[0]["kinematic_final_pixel_error"] < reports[0]["initial_pixel_error"]


def test_transport_goal_can_use_data_derived_constant_clearance() -> None:
    state = np.asarray([0.0, 0.0, 2.0, 7.0, -12.0, 5.0], dtype=np.float32)
    variants = (ImageServoVariant("transport", 0.4, 0.5, 0.8, 0.03),)
    prefixes, reports = generate_fcpg_goal_prefixes(
        state,
        np.asarray([100.0, 102.0]),
        np.asarray([130.0, 82.0]),
        _fit(),
        _LinearKinematics(),
        contact_height_m=0.08,
        transit_height_m=0.08,
        position_lower=np.asarray([-100.0] * 5 + [0.0]),
        position_upper=np.asarray([100.0] * 6),
        horizon_raw=20,
        variants=variants,
        source="drop_zone",
    )
    assert reports[0]["goal_role"] == "drop_zone"
    assert reports[0]["kinematic_final_pixel_error"] < reports[0]["initial_pixel_error"]
    np.testing.assert_array_equal(prefixes[0, :, -1], state[-1])


def test_gripper_phase_prefixes_hold_arm_and_move_only_gripper() -> None:
    state = np.asarray([1, 2, 3, 4, 5, 12], dtype=np.float32)
    prefixes, reports = generate_gripper_phase_prefixes(
        state,
        (5.0, 2.0),
        position_lower=np.asarray([-100.0] * 5 + [0.0]),
        position_upper=np.asarray([100.0] * 6),
        horizon_raw=12,
        maximum_step=1.5,
        source="grasp",
    )
    assert prefixes.shape == (2, 12, 6)
    np.testing.assert_array_equal(prefixes[:, :, :5], np.broadcast_to(state[:5], (2, 12, 5)))
    assert np.max(np.abs(np.diff(np.concatenate((state[None], prefixes[0])), axis=0))) <= 1.5
    assert prefixes[0, -1, -1] == 5.0
    assert reports[0]["goal_role"] == "grasp"
