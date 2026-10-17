from pathlib import Path

import numpy as np
import pytest

from fcpg.so101_kinematics import (
    SO101_ANCHOR_NAMES,
    SO101MuJoCoKinematics,
    normalized_arm_positions_to_radians,
    normalized_positions_to_mujoco_qpos,
    radians_to_normalized_arm_positions,
)


def test_normalized_arm_positions_map_to_joint_range() -> None:
    ranges = np.asarray(
        [[-2.0, 2.0], [-1.0, 3.0], [-4.0, 0.0], [-3.0, 1.0], [-5.0, 5.0]]
    )
    values = np.asarray(
        [[-100.0] * 5, [0.0] * 5, [100.0] * 5, [200.0] * 5]
    )

    radians = normalized_arm_positions_to_radians(values, ranges)

    np.testing.assert_allclose(radians[0], ranges[:, 0])
    np.testing.assert_allclose(radians[1], ranges.mean(axis=1))
    np.testing.assert_allclose(radians[2], ranges[:, 1])
    np.testing.assert_allclose(radians[3], ranges[:, 1])


def test_normalized_arm_positions_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="five joints"):
        normalized_arm_positions_to_radians(np.zeros((2, 6)), np.zeros((5, 2)))
    with pytest.raises(ValueError, match=r"shape \(5, 2\)"):
        normalized_arm_positions_to_radians(np.zeros((2, 5)), np.zeros((6, 2)))


def test_normalized_positions_map_gripper_zero_and_hundred_to_hinge_range() -> None:
    ranges = np.asarray(
        [[-2.0, 2.0], [-1.0, 3.0], [-4.0, 0.0], [-3.0, 1.0], [-5.0, 5.0], [-0.2, 1.8]]
    )
    values = np.asarray(
        [[-100.0] * 5 + [0.0], [0.0] * 5 + [50.0], [100.0] * 6]
    )

    qpos = normalized_positions_to_mujoco_qpos(values, ranges)

    np.testing.assert_allclose(qpos[0], ranges[:, 0])
    np.testing.assert_allclose(qpos[1], ranges.mean(axis=1))
    np.testing.assert_allclose(qpos[2], ranges[:, 1])


def test_radian_and_normalized_arm_mappings_are_inverse() -> None:
    ranges = np.asarray(
        [[-2.0, 2.0], [-1.0, 3.0], [-4.0, 0.0], [-3.0, 1.0], [-5.0, 5.0]]
    )
    values = np.asarray([[-100.0, -40.0, 0.0, 55.0, 100.0]])
    radians = normalized_arm_positions_to_radians(values, ranges)
    reconstructed = radians_to_normalized_arm_positions(radians, ranges)
    np.testing.assert_allclose(reconstructed, values)


def test_official_so101_model_produces_finite_gripper_trajectory() -> None:
    model_path = Path(
        "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
    )
    if not model_path.is_file():
        pytest.skip("official SO-101 MuJoCo snapshot is not installed")

    kinematics = SO101MuJoCoKinematics(model_path)
    states = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-10.0, -7.0, 18.0, 60.0, -56.0, 18.0],
        ]
    )
    xyz = kinematics.forward(states)

    assert xyz.shape == (2, 3)
    assert np.all(np.isfinite(xyz))
    assert np.linalg.norm(xyz[1] - xyz[0]) > 0.05
    assert kinematics.clipping_fraction(states) == 0.0


def test_official_so101_model_produces_three_physical_gripper_anchors() -> None:
    model_path = Path(
        "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
    )
    if not model_path.is_file():
        pytest.skip("official SO-101 MuJoCo snapshot is not installed")

    kinematics = SO101MuJoCoKinematics(model_path)
    states = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
        ]
    )
    anchors = kinematics.forward_anchors(states)

    assert SO101_ANCHOR_NAMES == (
        "gripper_center",
        "fixed_finger_tip",
        "moving_finger_tip",
    )
    assert anchors.shape == (2, 3, 3)
    assert np.all(np.isfinite(anchors))
    np.testing.assert_allclose(anchors[:, 0], anchors[:, 1:].mean(axis=1))
    jaw_separation = np.linalg.norm(anchors[:, 1] - anchors[:, 2], axis=1)
    assert abs(float(jaw_separation[1] - jaw_separation[0])) > 0.005


def test_official_so101_model_exposes_eye_in_hand_anchor_frame() -> None:
    model_path = Path(
        "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
    )
    if not model_path.is_file():
        pytest.skip("official SO-101 MuJoCo snapshot is not installed")

    kinematics = SO101MuJoCoKinematics(model_path)
    states = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            [-20.0, -30.0, 45.0, 25.0, 60.0, 30.0],
        ]
    )
    base, gripper = kinematics.forward_anchor_frames(states)

    np.testing.assert_allclose(base, kinematics.forward_anchors(states), atol=1e-10)
    assert gripper.shape == (2, 3, 3)
    assert np.all(np.isfinite(gripper))
    np.testing.assert_allclose(gripper[:, 0], gripper[:, 1:].mean(axis=1))
    # Base-frame coordinates move with the arm while local coordinates remain
    # close to the rigid gripper assembly.
    assert np.linalg.norm(base[1, 0] - base[0, 0]) > 0.05
    assert np.linalg.norm(gripper[1, 0] - gripper[0, 0]) < 0.02


def test_official_so101_position_ik_reaches_fk_target_and_preserves_gripper() -> None:
    model_path = Path(
        "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
    )
    if not model_path.is_file():
        pytest.skip("official SO-101 MuJoCo snapshot is not installed")

    kinematics = SO101MuJoCoKinematics(model_path)
    initial = np.asarray([0.0, -20.0, 20.0, 20.0, 0.0, 73.0])
    target_state = np.asarray([25.0, -45.0, 55.0, 5.0, 10.0, 20.0])
    target_xyz = kinematics.forward(target_state)

    result = kinematics.inverse_position(
        target_xyz,
        initial,
        max_iterations=100,
        tolerance_m=1e-3,
    )

    assert result.converged
    assert result.error_m < 1e-3
    assert result.state[5] == initial[5]
    assert np.all(np.abs(result.state[:5]) <= 100.0)
    np.testing.assert_allclose(kinematics.forward(result.state), target_xyz, atol=1e-3)


def test_official_so101_position_ik_fails_bounded_for_unreachable_target() -> None:
    model_path = Path(
        "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
    )
    if not model_path.is_file():
        pytest.skip("official SO-101 MuJoCo snapshot is not installed")

    kinematics = SO101MuJoCoKinematics(model_path)
    initial = np.asarray([0.0, -20.0, 20.0, 20.0, 0.0, 31.0])
    result = kinematics.inverse_position(
        np.asarray([2.0, 2.0, 2.0]),
        initial,
        max_iterations=20,
        tolerance_m=1e-4,
    )

    assert not result.converged
    assert np.all(np.isfinite(result.state))
    assert np.all(np.abs(result.state[:5]) <= 100.0)
    assert result.state[5] == initial[5]


def test_official_so101_anchor_ik_recovers_pose_and_gripper_opening() -> None:
    model_path = Path(
        "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
    )
    if not model_path.is_file():
        pytest.skip("official SO-101 MuJoCo snapshot is not installed")

    kinematics = SO101MuJoCoKinematics(model_path)
    initial = np.asarray([2.0, -24.0, 29.0, 17.0, 4.0, 72.0])
    target_state = np.asarray([8.0, -31.0, 37.0, 12.0, 9.0, 26.0])
    target_anchors = kinematics.forward_anchors(target_state)

    result = kinematics.inverse_anchors(
        target_anchors,
        initial,
        max_iterations=100,
        tolerance_m=5e-4,
    )

    assert result.converged
    assert result.max_error_m < 5e-4
    np.testing.assert_allclose(
        kinematics.forward_anchors(result.state), target_anchors, atol=5e-4
    )
    target_aperture = np.linalg.norm(target_anchors[1] - target_anchors[2])
    solved = kinematics.forward_anchors(result.state)
    solved_aperture = np.linalg.norm(solved[1] - solved[2])
    assert solved_aperture == pytest.approx(target_aperture, abs=5e-4)
