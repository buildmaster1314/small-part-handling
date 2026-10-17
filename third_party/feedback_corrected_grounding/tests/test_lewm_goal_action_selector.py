import numpy as np
import pytest

from fcpg.lewm_goal_action_selector import (
    average_costs_by_physical_snapshot,
    goal_distance_cost,
    grouped_selector_metrics,
    integrated_cartesian_action_forecast,
    select_lowest_cost,
    selector_metrics,
    spearman_rank_correlation,
    world_goal_distance_cost,
)


def _forecast(values: list[list[float]]) -> np.ndarray:
    # [snapshot, candidate, horizon, anchor, xy]
    array = np.asarray(values, dtype=np.float32)
    return array[:, :, None, None, :]


def test_goal_distance_uses_tool_center_and_requested_horizon() -> None:
    values = np.zeros((1, 2, 2, 3, 2), dtype=np.float32)
    values[0, 0, 1, 2] = (3.0, 4.0)
    values[0, 1, 1, 2] = (0.0, 1.0)
    goals = np.zeros((1, 2), dtype=np.float32)

    result = goal_distance_cost(values, goals, horizon_index=1, anchor_index=2)

    np.testing.assert_allclose(result, [[5.0, 1.0]])


def test_goal_distance_can_average_all_anchors() -> None:
    values = np.zeros((1, 1, 1, 3, 2), dtype=np.float32)
    values[0, 0, :, :, :] = 1.0
    goals = np.zeros((1, 2), dtype=np.float32)

    result = goal_distance_cost(values, goals, anchor_index=None)

    np.testing.assert_allclose(result, [[np.sqrt(2.0)]])


def test_world_goal_distance_uses_requested_anchor_and_horizon() -> None:
    values = np.zeros((1, 2, 2, 3, 3), dtype=np.float32)
    values[0, 0, 1, 2] = (2.0, 3.0, 6.0)
    values[0, 1, 1, 2] = (0.0, 0.0, 2.0)

    result = world_goal_distance_cost(
        values, np.zeros((1, 3), dtype=np.float32), horizon_index=1
    )

    np.testing.assert_allclose(result, [[7.0, 2.0]])


def test_integrated_cartesian_action_forecast_accumulates_raw_actions() -> None:
    current = np.zeros((1, 3, 3), dtype=np.float32)
    actions = np.zeros((1, 2, 2, 8), dtype=np.float32)
    actions[0, 0, :, 0::4] = 1.0
    actions[0, 1, :, 1::4] = -0.5

    result = integrated_cartesian_action_forecast(current, actions)

    assert result.shape == (1, 2, 2, 3, 3)
    np.testing.assert_allclose(result[0, 0, :, 2, 0], [0.02, 0.04])
    np.testing.assert_allclose(result[0, 1, :, 1, 1], [-0.01, -0.02])
    np.testing.assert_allclose(result[..., 2], 0.0)


def test_average_costs_by_physical_snapshot_averages_views_only() -> None:
    predicted = np.asarray(
        [[1.0, 3.0], [3.0, 1.0], [5.0, 7.0]], dtype=np.float32
    )
    actual = np.asarray(
        [[2.0, 4.0], [2.0, 4.0], [6.0, 8.0]], dtype=np.float32
    )

    pred, target, identity = average_costs_by_physical_snapshot(
        predicted, actual, np.asarray([10, 10, 11])
    )

    np.testing.assert_allclose(pred, [[2.0, 2.0], [5.0, 7.0]])
    np.testing.assert_allclose(target, [[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(identity, [10, 11])


def test_average_costs_by_physical_snapshot_rejects_label_drift() -> None:
    with pytest.raises(ValueError, match="inconsistent actual costs"):
        average_costs_by_physical_snapshot(
            np.zeros((2, 2)),
            np.asarray([[0.0, 1.0], [0.0, 2.0]]),
            np.asarray([7, 7]),
        )


def test_selector_is_lower_cost_and_tie_breaks_to_first_candidate() -> None:
    cost = np.asarray([[2.0, 1.0, 1.0], [0.5, 0.5, 2.0]])

    np.testing.assert_array_equal(select_lowest_cost(cost), [1, 0])


def test_selector_metrics_reports_regret_against_actual_best() -> None:
    predicted = np.asarray([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
    actual = np.asarray([[1.0, 0.0, 2.0], [0.0, 2.0, 1.0]])

    result = selector_metrics(predicted, actual, expert_index=0)

    assert result["top1_accuracy"] == pytest.approx(0.0)
    assert result["top1_regret_mean"] == pytest.approx(1.5)
    assert result["expert_regret_mean"] == pytest.approx(0.5)
    assert len(result["selected_candidate_indices"]) == 2


def test_selector_metrics_does_not_assume_anonymous_slot_zero_is_expert() -> None:
    result = selector_metrics(
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
    )

    assert result["expert_baseline_available"] is False
    assert "expert_regret_mean" not in result


def test_grouped_metrics_requires_aligned_labels() -> None:
    predicted = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    actual = np.asarray([[0.0, 1.0], [1.0, 0.0]])

    result = grouped_selector_metrics(
        predicted, actual, {"clock": np.asarray([3, 5])}
    )

    assert set(result) == {"clock=3", "clock=5"}
    with pytest.raises(ValueError, match="align"):
        grouped_selector_metrics(predicted, actual, {"bad": np.asarray([3])})


def test_spearman_handles_reversed_order() -> None:
    assert spearman_rank_correlation(np.asarray([0.0, 1.0, 2.0]), np.asarray([2.0, 1.0, 0.0])) == pytest.approx(-1.0)
