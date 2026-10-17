import numpy as np
import pytest

from fcpg.so101_realtime_mpc import (
    ManipulationPhase,
    PhaseEvidence,
    PhaseThresholds,
    PhaseTracker,
    PlanningRoundGuard,
    candidate_phase_mask,
    phase_prediction_bounds_mask,
    target_kind_for_phase,
    validate_command_chunk,
)


def _thresholds() -> PhaseThresholds:
    return PhaseThresholds(
        pickup_near_px=18.0,
        drop_near_px=22.0,
        pickup_near_m=0.018,
        drop_near_m=0.025,
        gripper_closed=10.0,
        gripper_open=20.0,
        grasp_roi_px=16.0,
        confirmations=2,
        gripper_empty_closed=3.3,
    )


def test_thresholds_reject_empty_closed_at_or_above_closed() -> None:
    with pytest.raises(ValueError, match="gripper_empty_closed"):
        PhaseThresholds(
            pickup_near_px=18.0,
            drop_near_px=22.0,
            pickup_near_m=0.018,
            drop_near_m=0.025,
            gripper_closed=10.0,
            gripper_open=20.0,
            grasp_roi_px=16.0,
            confirmations=2,
            gripper_empty_closed=10.0,
        )


def test_grasp_attachment_requires_a_stall_above_the_empty_band() -> None:
    tracker = PhaseTracker(_thresholds(), initial=ManipulationPhase.GRASP)

    # Empty close (bottomed out at 2.4): never confirms TRANSPORT.
    for _ in range(4):
        phase = tracker.update(
            _evidence(
                gripper_position=2.4,
                wrist_pickup_distance_px=10.0,
                pickup_visible=True,
            )
        )
    assert phase == ManipulationPhase.GRASP

    # Stall against an object (8.0): confirms after two rounds.
    for _ in range(2):
        phase = tracker.update(
            _evidence(
                gripper_position=8.0,
                wrist_pickup_distance_px=10.0,
                pickup_visible=True,
            )
        )
    assert phase == ManipulationPhase.TRANSPORT


def _evidence(**overrides: object) -> PhaseEvidence:
    values = {
        "gripper_position": 25.0,
        "gripper_to_pickup_px": 40.0,
        "gripper_to_drop_px": 60.0,
        "gripper_to_pickup_m": 0.10,
        "gripper_to_drop_m": 0.15,
        "wrist_pickup_distance_px": 30.0,
        "pickup_visible": True,
        "drop_zone_visible": True,
        "pickup_inside_drop_zone": False,
    }
    values.update(overrides)
    return PhaseEvidence(**values)


def test_phase_tracker_requires_two_consecutive_confirmations() -> None:
    tracker = PhaseTracker(_thresholds())
    near = _evidence(gripper_to_pickup_px=10.0, gripper_to_pickup_m=0.010)
    assert tracker.update(near) == ManipulationPhase.APPROACH
    assert tracker.update(near) == ManipulationPhase.GRASP

    attached = _evidence(
        gripper_position=5.0,
        gripper_to_pickup_px=5.0,
        gripper_to_pickup_m=0.005,
        wrist_pickup_distance_px=8.0,
    )
    assert tracker.update(attached) == ManipulationPhase.GRASP
    assert tracker.update(attached) == ManipulationPhase.TRANSPORT

    at_box = _evidence(
        gripper_position=5.0,
        gripper_to_drop_px=12.0,
        gripper_to_drop_m=0.012,
        wrist_pickup_distance_px=8.0,
    )
    assert tracker.update(at_box) == ManipulationPhase.TRANSPORT
    assert tracker.update(at_box) == ManipulationPhase.RELEASE

    released = _evidence(
        gripper_position=25.0,
        pickup_inside_drop_zone=True,
        gripper_to_drop_px=8.0,
        gripper_to_drop_m=0.008,
    )
    assert tracker.update(released) == ManipulationPhase.RELEASE
    assert tracker.update(released) == ManipulationPhase.DONE


def test_phase_tracker_does_not_advance_on_missing_visual_evidence() -> None:
    tracker = PhaseTracker(_thresholds(), initial=ManipulationPhase.GRASP)
    missing = _evidence(
        gripper_position=5.0,
        wrist_pickup_distance_px=None,
        pickup_visible=False,
    )
    assert tracker.update(missing) == ManipulationPhase.GRASP
    assert tracker.update(missing) == ManipulationPhase.GRASP


def test_phase_tracker_does_not_grasp_when_pixel_near_but_metric_far() -> None:
    tracker = PhaseTracker(_thresholds())
    ambiguous = _evidence(
        gripper_to_pickup_px=8.0,
        gripper_to_pickup_m=0.080,
    )

    assert tracker.update(ambiguous) == ManipulationPhase.APPROACH
    assert tracker.update(ambiguous) == ManipulationPhase.APPROACH


def test_candidate_phase_mask_uses_gripper_direction() -> None:
    prefixes = np.zeros((4, 5, 6), dtype=np.float32)
    prefixes[0, :, -1] = [5, 10, 15, 20, 25]
    prefixes[1, :, -1] = [25, 20, 15, 10, 5]
    prefixes[2, :, -1] = [5, 6, 7, 8, 9]
    prefixes[3, :, -1] = [10, 12, 15, 20, 25]

    np.testing.assert_array_equal(
        candidate_phase_mask(prefixes, ManipulationPhase.APPROACH, _thresholds()),
        [True, False, True, True],
    )
    np.testing.assert_array_equal(
        candidate_phase_mask(prefixes, ManipulationPhase.GRASP, _thresholds()),
        [False, True, False, False],
    )
    np.testing.assert_array_equal(
        candidate_phase_mask(prefixes, ManipulationPhase.TRANSPORT, _thresholds()),
        [False, False, True, False],
    )
    np.testing.assert_array_equal(
        candidate_phase_mask(prefixes, ManipulationPhase.RELEASE, _thresholds()),
        [True, False, False, True],
    )


def test_approach_prediction_bounds_ignore_only_gripper_forecast() -> None:
    predictions = np.zeros((2, 8, 6), dtype=np.float32)
    predictions[:, :, -1] = -5.0
    predictions[1, :, 2] = 105.0

    approach = phase_prediction_bounds_mask(
        predictions, ManipulationPhase.APPROACH, margin=3.0
    )
    grasp = phase_prediction_bounds_mask(
        predictions, ManipulationPhase.GRASP, margin=3.0
    )

    np.testing.assert_array_equal(approach, [True, False])
    np.testing.assert_array_equal(grasp, [False, False])


def test_grasp_mask_admits_opening_recovery_when_gripper_already_closed() -> None:
    thresholds = _thresholds()
    base = np.zeros((1, 40, 6), dtype=np.float32)

    opening = base.copy()
    opening[0, :, 5] = np.linspace(1.9, 20.0, 40)
    closing = base.copy()
    closing[0, :, 5] = np.linspace(20.0, thresholds.gripper_closed - 1.0, 40)

    # Live gripper truly empty-closed: opening recovery becomes admissible.
    recovery = candidate_phase_mask(
        np.concatenate((opening, closing)),
        ManipulationPhase.GRASP,
        thresholds,
        current_gripper=1.9,
        recovery_below=3.3,
    )
    np.testing.assert_array_equal(recovery, [True, True])

    # Grasp stall against an object (aperture above the empty-closed band):
    # opening must NOT be admissible — it would drop the object.
    stalled = candidate_phase_mask(
        np.concatenate((opening, closing)),
        ManipulationPhase.GRASP,
        thresholds,
        current_gripper=9.6,
        recovery_below=3.3,
    )
    np.testing.assert_array_equal(stalled, [False, True])

    # Live gripper open: only closing candidates pass, as before.
    normal = candidate_phase_mask(
        np.concatenate((opening, closing)),
        ManipulationPhase.GRASP,
        thresholds,
        current_gripper=25.0,
        recovery_below=3.3,
    )
    np.testing.assert_array_equal(normal, [False, True])


def test_regress_to_approach_only_from_grasp() -> None:
    tracker = PhaseTracker(_thresholds(), initial=ManipulationPhase.GRASP)
    tracker.regress_to_approach()
    assert tracker.phase == ManipulationPhase.APPROACH

    transport = PhaseTracker(
        _thresholds(), initial=ManipulationPhase.TRANSPORT
    )
    transport.regress_to_approach()
    assert transport.phase == ManipulationPhase.TRANSPORT


def test_validate_command_chunk_rejects_outward_motion_near_limit() -> None:
    state = np.asarray([0, -98, 20, 20, 20, 10], dtype=np.float32)
    chunk = np.repeat(state[None], 5, axis=0)
    chunk[:, 1] = [-99, -99, -99, -99, -99]
    result = validate_command_chunk(chunk, state)
    assert not result.safe
    assert "moving_outward_at_joint_limit" in result.reasons


def test_validate_command_chunk_accepts_inward_five_command_token() -> None:
    state = np.asarray([0, -98, 20, 20, 20, 10], dtype=np.float32)
    chunk = np.repeat(state[None], 5, axis=0)
    chunk[:, 1] = [-97, -96, -95, -94, -93]
    result = validate_command_chunk(chunk, state)
    assert result.safe
    assert result.maximum_step_jump == pytest.approx(1.0)


def test_validate_command_chunk_outward_strict_by_default() -> None:
    state = np.asarray([0, -97.8, 97.9, 20, 20, 10], dtype=np.float32)
    chunk = np.repeat(state[None], 5, axis=0)
    chunk[:, 1] = -97.95
    chunk[:, 2] = 98.10

    result = validate_command_chunk(chunk, state)

    assert not result.safe
    assert "moving_outward_at_joint_limit" in result.reasons


def test_validate_command_chunk_opt_in_tolerance_covers_encoder_count() -> None:
    state = np.asarray([0, -97.8, 97.9, 20, 20, 10], dtype=np.float32)

    held = np.repeat(state[None], 5, axis=0)
    held[:, 1] = -97.883
    held[:, 2] = 97.983
    quantized = validate_command_chunk(held, state, outward_tolerance=0.1)
    assert quantized.safe

    beyond = np.repeat(state[None], 5, axis=0)
    beyond[:, 1] = -97.95
    beyond[:, 2] = 98.10
    rejected = validate_command_chunk(beyond, state, outward_tolerance=0.1)
    assert not rejected.safe
    assert "moving_outward_at_joint_limit" in rejected.reasons


def test_round_guard_rejects_stale_id_state_and_target() -> None:
    guard = PlanningRoundGuard(
        round_id="round-1",
        created_monotonic_s=10.0,
        state=np.zeros(6, dtype=np.float32),
        target_px=np.asarray([20.0, 30.0], dtype=np.float32),
        target_kind="pickup",
    )
    with pytest.raises(RuntimeError, match="stale planning round"):
        guard.validate_live(
            round_id="round-0",
            now_monotonic_s=11.0,
            live_state=np.zeros(6),
            live_target_px=np.asarray([20.0, 30.0]),
            max_age_s=5.0,
            max_state_drift=2.0,
            max_target_drift_px=10.0,
        )
    with pytest.raises(RuntimeError, match="state drift"):
        guard.validate_live(
            round_id="round-1",
            now_monotonic_s=11.0,
            live_state=np.asarray([3.0, 0, 0, 0, 0, 0]),
            live_target_px=np.asarray([20.0, 30.0]),
            max_age_s=5.0,
            max_state_drift=2.0,
            max_target_drift_px=10.0,
        )
    with pytest.raises(RuntimeError, match="target drift"):
        guard.validate_live(
            round_id="round-1",
            now_monotonic_s=11.0,
            live_state=np.zeros(6),
            live_target_px=np.asarray([40.0, 30.0]),
            max_age_s=5.0,
            max_state_drift=2.0,
            max_target_drift_px=10.0,
        )


def test_round_guard_zero_max_age_disables_time_expiry() -> None:
    guard = PlanningRoundGuard(
        round_id="round-1",
        created_monotonic_s=10.0,
        state=np.zeros(6, dtype=np.float32),
        target_px=np.asarray([20.0, 30.0], dtype=np.float32),
        target_kind="pickup",
    )

    result = guard.validate_live(
        round_id="round-1",
        now_monotonic_s=10_000.0,
        live_state=np.zeros(6),
        live_target_px=np.asarray([20.0, 30.0]),
        max_age_s=0.0,
        max_state_drift=2.0,
        max_target_drift_px=10.0,
    )

    assert result["age_s"] == pytest.approx(9_990.0)


def test_target_kind_tracks_phase() -> None:
    assert target_kind_for_phase(ManipulationPhase.APPROACH) == "pickup"
    assert target_kind_for_phase(ManipulationPhase.GRASP) == "pickup"
    assert target_kind_for_phase(ManipulationPhase.TRANSPORT) == "drop_zone"
    assert target_kind_for_phase(ManipulationPhase.RELEASE) == "drop_zone"
    assert target_kind_for_phase(ManipulationPhase.DONE) is None
