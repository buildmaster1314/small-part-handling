from __future__ import annotations

import numpy as np
import pytest

from fcpg.lewm_counterfactual_runtime import FrameRecord
from fcpg.lewm_towel_receding_selector import (
    CANDIDATE_NAMES,
    POLICY_CANDIDATE_NAMES,
    build_goal_centered_prefix,
    build_policy_centered_towel_candidates,
    build_policy_repeated_prefix,
    build_towel_candidate_prefixes,
    causal_history_chunks,
    center_calibrated_world_forecast,
    kinematic_world_goal_cost,
    lewm_world_goal_cost,
    selected_raw_chunk,
    tool_xy_goal_cost,
    tool_xy_goal_reach_cost,
)
from scripts.evaluate_lewm_towel_policy_headroom import _method_row
from scripts.evaluate_lewm_towel_receding_selector import (
    _anchor_occluded_record,
    _paired_nominal_fcpg_summary,
    _route_interruption_state,
)


def test_goal_centered_prefix_reaches_and_holds_current_goal() -> None:
    current = np.asarray((0.0, 0.5, 0.04), dtype=np.float32)
    goal = np.asarray((0.18, 0.5, 0.04), dtype=np.float32)
    prefix = build_goal_centered_prefix(current, goal, gripper_command=1.0)
    assert prefix.shape == (40, 4)
    assert np.all(prefix[:, 3] == 1.0)
    forecast = current + 0.01 * np.cumsum(prefix[:, :3], axis=0)
    assert np.linalg.norm(forecast[-1] - goal) < 5.0e-4
    assert np.linalg.norm(prefix[-1, :3]) < 1.0e-2


def test_candidate_bank_is_bounded_and_preserves_gripper() -> None:
    center = np.zeros((40, 4), dtype=np.float32)
    center[:, 0] = 0.9
    center[:, 3] = 0.7
    candidates = build_towel_candidate_prefixes(center)
    assert candidates.shape == (len(CANDIDATE_NAMES), 8, 20)
    raw = candidates.reshape(len(CANDIDATE_NAMES), 40, 4)
    assert np.max(raw) <= 1.0
    assert np.min(raw) >= -1.0
    np.testing.assert_array_equal(raw[:, :, 3], 0.7)
    np.testing.assert_array_equal(raw[2, :, :3], 0.0)
    np.testing.assert_allclose(raw[1, :, :3], -center[:, :3])
    np.testing.assert_allclose(raw[3, :, :3], 0.5 * center[:, :3])


def test_policy_centered_bank_changes_only_horizontal_policy_motion() -> None:
    action = np.asarray((0.8, 0.2, -0.15, 0.7), dtype=np.float32)
    prefix = build_policy_repeated_prefix(action)
    candidates = build_policy_centered_towel_candidates(prefix)
    raw = candidates.reshape(len(POLICY_CANDIDATE_NAMES), 40, 4)

    assert candidates.shape == (len(POLICY_CANDIDATE_NAMES), 8, 20)
    np.testing.assert_array_equal(raw[:, :, 2], action[2])
    np.testing.assert_array_equal(raw[:, :, 3], action[3])
    expected_xy = np.broadcast_to(action[:2], (40, 2))
    np.testing.assert_allclose(raw[0, :, :2], expected_xy)
    np.testing.assert_allclose(raw[1, :, :2], -expected_xy)
    np.testing.assert_array_equal(raw[2, :, :2], 0.0)
    np.testing.assert_allclose(raw[3, :, :2], 0.5 * expected_xy)
    np.testing.assert_allclose(raw[5, :, :2], 0.25 * expected_xy)
    np.testing.assert_allclose(raw[6, 0, :2], -0.5 * action[:2])
    np.testing.assert_allclose(raw[7, 0, :2], 0.75 * action[:2])


def test_policy_prefix_rejects_out_of_bounds_current_action() -> None:
    with pytest.raises(ValueError, match="outside"):
        build_policy_repeated_prefix(np.asarray((1.1, 0.0, 0.0, 1.0)))


def test_history_is_left_padded_and_contains_only_executed_actions() -> None:
    actions = [np.full(4, float(index), dtype=np.float32) for index in range(3)]
    history = causal_history_chunks(actions).reshape(10, 4)
    np.testing.assert_array_equal(history[:7], 0.0)
    np.testing.assert_array_equal(history[-3:], np.asarray(actions))


def test_world_costs_choose_forecast_closest_to_goal() -> None:
    current = np.zeros((3, 3), dtype=np.float32)
    candidates = np.zeros((2, 8, 20), dtype=np.float32)
    candidates[0, :, 0::4] = 1.0
    candidates[1, :, 0::4] = 0.5
    goal = np.asarray((0.15, 0.0, 0.0), dtype=np.float32)
    kinematic = kinematic_world_goal_cost(current, candidates, goal)
    assert int(np.argmin(kinematic)) == 1

    predicted = np.zeros((2, 8, 3, 3), dtype=np.float32)
    predicted[0, 5, 2, 0] = 0.1
    predicted[1, 5, 2, 0] = 0.15
    learned = lewm_world_goal_cost(predicted, goal)
    assert int(np.argmin(learned)) == 1


def test_tool_xy_cost_ignores_height_but_uses_requested_horizon() -> None:
    predicted = np.zeros((2, 8, 3, 3), dtype=np.float32)
    predicted[0, 5, 2] = (0.1, 0.2, 50.0)
    predicted[1, 5, 2] = (0.3, 0.2, 0.0)
    goal = np.asarray((0.1, 0.2, -50.0), dtype=np.float32)
    cost = tool_xy_goal_cost(predicted, goal)
    np.testing.assert_allclose(cost, (0.0, 0.2), atol=1.0e-7)


def test_reach_cost_rewards_arriving_before_the_terminal_horizon() -> None:
    predicted = np.zeros((2, 8, 3, 3), dtype=np.float32)
    predicted[0, :, 2, 0] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    predicted[1, :, 2, 0] = 0.15
    goal = np.asarray((0.2, 0.0, 0.0), dtype=np.float32)
    cost = tool_xy_goal_reach_cost(predicted, goal, terminal_weight=0.0)
    assert int(np.argmin(cost)) == 0
    assert cost[0] == pytest.approx(0.0)
    assert cost[1] == pytest.approx(0.05)


def test_method_row_reports_actual_selected_regret() -> None:
    row = _method_row(
        np.asarray((2.0, 0.0, 1.0)),
        np.asarray((0.2, 0.3, 0.1)),
    )
    assert row["selected_index"] == 1
    assert row["oracle_index"] == 2
    assert row["regret_m"] == pytest.approx(0.2)
    assert row["normalized_regret"] == pytest.approx(1.0)


def test_method_row_clamps_roundoff_below_zero() -> None:
    row = _method_row(
        np.asarray((0.0, 1.0)),
        np.asarray((0.1 - 1.0e-16, 0.1)),
    )
    assert row["selected_index"] == 0
    assert row["regret_m"] == 0.0
    assert row["normalized_regret"] == 0.0


def test_selected_chunk_never_executes_the_remaining_prefix() -> None:
    candidates = np.arange(2 * 8 * 20, dtype=np.float32).reshape(2, 8, 20)
    selected = selected_raw_chunk(candidates, 1)
    assert selected.shape == (5, 4)
    np.testing.assert_array_equal(selected.reshape(-1), candidates[1, 0])
    with pytest.raises(ValueError, match="outside"):
        selected_raw_chunk(candidates, 2)


def test_center_calibration_removes_only_common_absolute_bias() -> None:
    current = np.zeros((3, 3), dtype=np.float32)
    candidates = np.zeros((2, 8, 20), dtype=np.float32)
    candidates[0, :, 0::4] = 1.0
    candidates[1, :, 0::4] = 0.5
    predicted = np.zeros((2, 8, 3, 3), dtype=np.float32)
    common_bias = np.asarray((0.2, -0.1, 0.05), dtype=np.float32)
    predicted[:] = common_bias
    predicted[1, :, :, 0] += 0.03
    calibrated = center_calibrated_world_forecast(predicted, current, candidates)
    expected_center_x = 0.01 * np.arange(5, 41, 5, dtype=np.float32)
    np.testing.assert_allclose(
        calibrated[0, :, :, 0], np.repeat(expected_center_x[:, None], 3, axis=1)
    )
    np.testing.assert_allclose(
        calibrated[1] - calibrated[0], predicted[1] - predicted[0], atol=1.0e-7
    )


def test_paired_summary_tracks_success_grounding_and_choice_changes() -> None:
    def row(
        episode: int,
        controller: str,
        *,
        success: bool,
        steps: int,
        epe: float,
        choices: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "episode_index": episode,
            "controller": controller,
            "success": success,
            "raw_steps": steps,
            "mean_wipe_grounding_epe_px": epe,
            "decisions": [
                {"phase": "wipe_left", "choice": choice} for choice in choices
            ],
        }

    summary = _paired_nominal_fcpg_summary(
        [
            row(0, "lewm_nominal", success=True, steps=100, epe=10.0, choices=("center", "half")),
            row(0, "lewm_fcpg", success=True, steps=105, epe=1.0, choices=("center", "center")),
            row(1, "lewm_nominal", success=False, steps=120, epe=12.0, choices=("half",)),
            row(1, "lewm_fcpg", success=True, steps=110, epe=1.0, choices=("half",)),
        ]
    )
    assert summary is not None
    assert summary["both_success"] == 1
    assert summary["fcpg_only_success"] == 1
    assert summary["mean_raw_step_delta_fcpg_minus_nominal"] == pytest.approx(-2.5)
    assert summary["relative_grounding_epe_reduction"] == pytest.approx(10.0 / 11.0)
    assert summary["candidate_choice_disagreements"] == 1
    assert summary["aligned_wipe_decisions"] == 3
    assert summary["episodes_with_changed_choice_sequence"] == 1


def test_anchor_occlusion_changes_rgb_without_mutating_geometry() -> None:
    record = FrameRecord(
        rgb=np.ones((3, 32, 32), dtype=np.float32),
        proprio=np.zeros(4, dtype=np.float32),
        keypoints_xyz=np.zeros((3, 3), dtype=np.float32),
        q_nominal=np.asarray(((8, 8), (16, 16), (24, 24)), dtype=np.float32),
        valid_nominal=np.ones(3, dtype=bool),
        q_true=np.asarray(((8, 8), (16, 16), (24, 24)), dtype=np.float32),
        valid_true=np.ones(3, dtype=bool),
    )

    occluded = _anchor_occluded_record(record, radius_px=2, blur_radius=0.0)

    assert occluded is not record
    np.testing.assert_array_equal(occluded.q_nominal, record.q_nominal)
    np.testing.assert_array_equal(record.rgb, 1.0)
    assert float(occluded.rgb[:, 16, 16].max()) < 0.1
    assert float(occluded.rgb[:, 0, 0].min()) == pytest.approx(1.0)


def test_interruption_router_never_exposes_measurement_to_hold_or_prior() -> None:
    measurement = np.full((4, 192), 1.0, dtype=np.float32)
    trusted = np.full((4, 192), 2.0, dtype=np.float32)
    prior = np.full((4, 192), 3.0, dtype=np.float32)

    current, updated, source, used = _route_interruption_state(
        "lewm_fcpg_prior",
        hazard_active=False,
        measurement_state=measurement,
        last_valid_state=None,
        next_prior_state=None,
    )
    np.testing.assert_array_equal(current, measurement)
    np.testing.assert_array_equal(updated, measurement)
    assert source == "clean_measurement"
    assert used

    held, held_trusted, source, used = _route_interruption_state(
        "lewm_fcpg_hold",
        hazard_active=True,
        measurement_state=None,
        last_valid_state=trusted,
        next_prior_state=prior,
    )
    np.testing.assert_array_equal(held, trusted)
    np.testing.assert_array_equal(held_trusted, trusted)
    assert source == "hold_last"
    assert not used

    predicted, predicted_trusted, source, used = _route_interruption_state(
        "lewm_fcpg_prior",
        hazard_active=True,
        measurement_state=None,
        last_valid_state=trusted,
        next_prior_state=prior,
    )
    np.testing.assert_array_equal(predicted, prior)
    np.testing.assert_array_equal(predicted_trusted, trusted)
    assert source == "lewm_prior"
    assert not used

    with pytest.raises(RuntimeError, match="unavailable"):
        _route_interruption_state(
            "lewm_fcpg_prior",
            hazard_active=True,
            measurement_state=measurement,
            last_valid_state=trusted,
            next_prior_state=None,
        )
