from pathlib import Path

import numpy as np
import pytest

from fcpg.so101_ik_candidates import (
    IKPathSpec,
    REALTIME_CONTACT_PATH_SPECS,
    generate_ik_candidate_prefixes,
    track_anchor_trajectory,
)
from fcpg.so101_kinematics import SO101MuJoCoKinematics


def _kinematics() -> SO101MuJoCoKinematics:
    model_path = Path("third_party/so_arm100/Simulation/SO101/so101_new_calib.xml")
    if not model_path.is_file():
        pytest.skip("official SO-101 MuJoCo snapshot is not installed")
    return SO101MuJoCoKinematics(model_path)


def test_ik_candidates_preserve_gripper_and_match_horizon_contract() -> None:
    kinematics = _kinematics()
    current = np.asarray([0.0, -25.0, 35.0, 30.0, -20.0, 17.0])
    target_state = np.asarray([20.0, -45.0, 50.0, 10.0, -20.0, 17.0])
    target_surface = kinematics.forward(target_state)
    target_surface[2] -= 0.035

    batch = generate_ik_candidate_prefixes(
        kinematics, current, target_surface, frameskip=5, horizon=8
    )

    assert batch.raw_prefixes.shape == (8, 40, 6)
    assert batch.horizon_states.shape == (8, 8, 6)
    assert batch.horizon_xyz.shape == (8, 8, 3)
    assert batch.converged.shape == (8,)
    assert np.all(batch.raw_prefixes[..., 5] == current[5])
    assert np.all(batch.horizon_states[..., 5] == current[5])
    assert np.all(np.abs(batch.raw_prefixes[..., :5]) <= 100.0)
    np.testing.assert_allclose(
        batch.raw_prefixes[:, 4::5], batch.horizon_states, atol=1e-6
    )
    assert np.mean(batch.converged) > 0.8


def test_ik_candidate_raw_steps_are_smooth() -> None:
    kinematics = _kinematics()
    current = np.asarray([0.0, -20.0, 20.0, 20.0, 0.0, 70.0])
    target_state = np.asarray([15.0, -35.0, 40.0, 15.0, 0.0, 70.0])
    target_surface = kinematics.forward(target_state)
    target_surface[2] -= 0.025

    batch = generate_ik_candidate_prefixes(kinematics, current, target_surface)
    trajectory = np.concatenate(
        (np.broadcast_to(current, (len(batch.names), 1, 6)), batch.raw_prefixes), axis=1
    )

    assert float(np.max(np.abs(np.diff(trajectory, axis=1)))) < 4.0


def test_realtime_ik_candidates_include_contact_and_move_arm_chain() -> None:
    kinematics = _kinematics()
    current = np.asarray(
        [18.35, -97.61, 98.05, 41.02, -51.32, 1.64], dtype=np.float32
    )
    target_surface = np.asarray(
        [0.269524, -0.118163, -0.017118], dtype=np.float32
    )

    batch = generate_ik_candidate_prefixes(
        kinematics,
        current,
        target_surface,
        path_specs=REALTIME_CONTACT_PATH_SPECS,
    )

    contact = batch.names.index("direct_contact")
    np.testing.assert_allclose(
        batch.target_approach_xyz[contact], target_surface, atol=1.0e-8
    )
    assert batch.converged[contact]
    assert batch.terminal_ik_error_m[contact] < 3.0e-3
    first_chunk_delta = batch.raw_prefixes[contact, 4] - current
    assert abs(float(first_chunk_delta[1])) > 1.0
    assert abs(float(first_chunk_delta[2])) > 1.0


def test_realtime_ik_can_arrive_early_and_hold_for_remaining_horizon() -> None:
    kinematics = _kinematics()
    current = np.asarray([0.0, -20.0, 20.0, 20.0, 0.0, 70.0])
    target_state = np.asarray([15.0, -35.0, 40.0, 15.0, 0.0, 70.0])
    target_surface = kinematics.forward(target_state)

    batch = generate_ik_candidate_prefixes(
        kinematics,
        current,
        target_surface,
        arrival_raw_steps=20,
        path_specs=(IKPathSpec("direct", 0.0, 0.0, 0.0),),
    )

    np.testing.assert_allclose(
        batch.raw_prefixes[0, 19:],
        np.repeat(batch.raw_prefixes[0, 19][None], 21, axis=0),
    )
    np.testing.assert_allclose(
        batch.horizon_states[0, 3:],
        np.repeat(batch.horizon_states[0, 3][None], 5, axis=0),
    )


def test_realtime_ik_adaptive_arrival_finishes_short_moves_early() -> None:
    kinematics = _kinematics()
    current = np.asarray([0.0, -20.0, 20.0, 20.0, 0.0, 70.0])
    target_state = np.asarray([15.0, -35.0, 40.0, 15.0, 0.0, 70.0])
    target_surface = kinematics.forward(target_state)

    batch = generate_ik_candidate_prefixes(
        kinematics,
        current,
        target_surface,
        arrival_raw_steps=20,
        max_arrival_step=6.0,
        path_specs=(IKPathSpec("direct", 0.0, 0.0, 0.0),),
    )

    prefix = batch.raw_prefixes[0].astype(np.float64)
    trajectory = np.concatenate((current[None], prefix), axis=0)
    assert float(np.max(np.abs(np.diff(trajectory, axis=0)))) <= 6.0 + 1e-3

    travel = float(np.max(np.abs(prefix[-1] - current)))
    needed = int(np.ceil(travel / 6.0 - 1e-9))
    arrival = int(min(20, max(5, int(np.ceil(needed / 5)) * 5)))
    assert arrival < 20
    np.testing.assert_allclose(
        prefix[arrival - 1 :],
        np.repeat(prefix[arrival - 1][None], 41 - arrival, axis=0),
    )


def test_realtime_ik_adaptive_arrival_bounds_steps_across_contact_paths() -> None:
    kinematics = _kinematics()
    current = np.asarray([0.0, -20.0, 20.0, 20.0, 0.0, 70.0])
    target_state = np.asarray([15.0, -35.0, 40.0, 15.0, 0.0, 70.0])
    target_surface = kinematics.forward(target_state)

    batch = generate_ik_candidate_prefixes(
        kinematics,
        current,
        target_surface,
        arrival_raw_steps=20,
        max_arrival_step=6.0,
        path_specs=REALTIME_CONTACT_PATH_SPECS,
    )

    trajectory = np.concatenate(
        (np.broadcast_to(current, (len(batch.names), 1, 6)), batch.raw_prefixes),
        axis=1,
    )
    assert float(np.max(np.abs(np.diff(trajectory, axis=1)))) <= 6.0 + 1e-3


def test_realtime_ik_adaptive_arc_midpoint_lands_on_horizon_grid() -> None:
    kinematics = _kinematics()
    current = np.asarray([0.0, -20.0, 20.0, 20.0, 0.0, 70.0])
    target_state = np.asarray([15.0, -35.0, 40.0, 15.0, 0.0, 70.0])
    target_surface = kinematics.forward(target_state)

    batch = generate_ik_candidate_prefixes(
        kinematics,
        current,
        target_surface,
        arrival_raw_steps=20,
        max_arrival_step=6.0,
        path_specs=REALTIME_CONTACT_PATH_SPECS,
    )

    for index, name in enumerate(batch.names):
        if not name.startswith("arc"):
            continue
        prefix = batch.raw_prefixes[index].astype(np.float64)
        trajectory = np.concatenate((current[None], prefix), axis=0)
        second_difference = np.abs(np.diff(trajectory, n=2, axis=0)).max(axis=1)
        kinks = np.flatnonzero(second_difference > 1e-4)
        assert kinks.size >= 1, name
        midpoint_raw_index = int(kinks[0])
        assert midpoint_raw_index % 5 == 4, (name, midpoint_raw_index)


def test_realtime_ik_retries_near_limit_posture_without_relaxing_tolerance() -> None:
    kinematics = _kinematics()
    current = np.asarray(
        [-14.595376, -97.529846, 97.95918, 64.681206, -51.42857, 1.708279],
        dtype=np.float32,
    )
    target = np.asarray(
        [0.19565865, 0.02089234, 0.02378927], dtype=np.float32
    )

    batch = generate_ik_candidate_prefixes(
        kinematics,
        current,
        target,
        path_specs=(REALTIME_CONTACT_PATH_SPECS[0],),
    )

    assert batch.converged[0]
    assert batch.terminal_ik_error_m[0] <= 3.0e-3
    first_chunk_delta = batch.raw_prefixes[0, 4] - current
    assert np.count_nonzero(np.abs(first_chunk_delta[:5]) > 0.1) >= 3


def test_ik_candidate_paths_must_be_unique() -> None:
    kinematics = _kinematics()
    state = np.asarray([0.0, -25.0, 35.0, 30.0, -20.0, 17.0])
    target = kinematics.forward(state)
    duplicate = IKPathSpec("same", 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="unique"):
        generate_ik_candidate_prefixes(
            kinematics,
            state,
            target,
            path_specs=(duplicate, duplicate),
        )


def test_anchor_trajectory_ik_recovers_three_lines_and_gripper_channel() -> None:
    kinematics = _kinematics()
    current = np.asarray([0.0, -24.0, 28.0, 18.0, 3.0, 75.0])
    target_states = np.asarray(
        [
            [2.0, -26.0, 30.0, 17.0, 4.0, 62.0],
            [4.0, -28.0, 33.0, 15.0, 6.0, 45.0],
            [7.0, -31.0, 36.0, 13.0, 8.0, 24.0],
        ]
    )
    target_anchors = kinematics.forward_anchors(target_states)

    result = track_anchor_trajectory(
        kinematics,
        current,
        target_anchors,
        frameskip=5,
        tolerance_m=2e-3,
    )

    assert result.converged
    assert result.raw_prefix.shape == (15, 6)
    assert result.horizon_states.shape == (3, 6)
    np.testing.assert_allclose(result.raw_prefix[4::5], result.horizon_states)
    np.testing.assert_allclose(result.achieved_anchors, target_anchors, atol=2e-3)
    assert result.horizon_states[-1, 5] < result.horizon_states[0, 5]


def test_anchor_trajectory_ik_does_not_push_farther_past_a_joint_limit() -> None:
    kinematics = _kinematics()
    current = np.asarray([0.0, -98.0, 98.0, 20.0, 3.0, 35.0])
    outward_state = np.asarray([0.0, -100.0, 100.0, 20.0, 3.0, 35.0])
    target_anchors = kinematics.forward_anchors(outward_state)[None]

    result = track_anchor_trajectory(
        kinematics,
        current,
        target_anchors,
        frameskip=5,
        tolerance_m=3e-3,
        limit_guard=3.0,
    )

    assert np.all(result.raw_prefix[:, 1] >= current[1] - 1e-6)
    assert np.all(result.raw_prefix[:, 2] <= current[2] + 1e-6)
