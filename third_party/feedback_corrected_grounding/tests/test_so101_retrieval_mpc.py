import numpy as np
import pytest
from types import SimpleNamespace

from fcpg.so101_retrieval_mpc import (
    SO101_POSITION_LOWER,
    SO101_POSITION_UPPER,
    candidate_safety_checks,
    candidate_start_rows,
    chunk_and_normalize_actions,
    prefix_rmse,
    raw_action_prefixes,
    rebase_action_prefixes,
    retrieve_diverse_candidates,
    select_safe_candidate,
    terminal_goal_cost,
    terminal_goal_rows,
    terminal_state_cost,
    validate_paired_camera_data,
)


def test_candidate_rows_never_cross_episode_boundary() -> None:
    episodes = np.repeat([2, 4], 12)
    frames = np.tile(np.arange(12), 2)

    rows = candidate_start_rows(
        episodes,
        frames,
        {2, 4},
        horizon_raw=5,
        stride=2,
    )

    assert rows.tolist() == [0, 2, 4, 6, 12, 14, 16, 18]
    assert all(episodes[row] == episodes[row + 4] for row in rows)


def test_candidate_rows_reject_noncontiguous_episode_frames() -> None:
    with pytest.raises(ValueError, match="non-contiguous frame"):
        candidate_start_rows(
            np.zeros(8, dtype=np.int64),
            np.asarray([0, 1, 2, 4, 5, 6, 7, 8]),
            {0},
            horizon_raw=3,
            stride=1,
        )


def test_terminal_goal_rows_take_tail_from_each_episode() -> None:
    episodes = np.repeat([1, 3], [5, 4])
    rows = terminal_goal_rows(episodes, {1, 3}, tail_frames=2)
    assert rows.tolist() == [3, 4, 7, 8]


def test_retrieval_keeps_diverse_candidates_per_episode() -> None:
    rows = np.asarray([0, 5, 10, 20, 25, 30])
    episodes = np.asarray([0, 0, 0, 1, 1, 1])
    latents = np.asarray([[0.0], [0.01], [0.4], [0.02], [0.03], [0.5]])
    states = np.zeros((6, 6), dtype=np.float32)

    result = retrieve_diverse_candidates(
        np.asarray([0.0]),
        np.zeros(6),
        latents,
        states,
        rows,
        episodes,
        state_scale=np.ones(6),
        per_episode=2,
        minimum_row_gap=8,
    )

    assert result.rows.tolist() == [0, 20, 10, 30]
    assert result.episodes.tolist() == [0, 1, 0, 1]


def test_retrieval_can_use_deployment_phase_distance() -> None:
    rows = np.asarray([0, 10, 20])
    episodes = np.zeros(3, dtype=np.int64)
    latents = np.asarray([[0.0], [0.1], [0.2]], dtype=np.float32)
    states = np.zeros((3, 6), dtype=np.float32)

    result = retrieve_diverse_candidates(
        np.asarray([0.0], dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        latents,
        states,
        rows,
        episodes,
        state_scale=np.ones(6, dtype=np.float32),
        per_episode=1,
        minimum_row_gap=0,
        extra_distance=np.asarray([1.0, 0.0, 0.5], dtype=np.float32),
        extra_weight=2.0,
    )

    assert result.rows.tolist() == [10]


def test_action_prefix_chunking_preserves_raw_order() -> None:
    actions = np.arange(20 * 6, dtype=np.float32).reshape(20, 6)
    raw = raw_action_prefixes(actions, np.asarray([1, 8]), horizon_raw=6)
    chunks = chunk_and_normalize_actions(
        raw,
        frameskip=3,
        center=np.zeros(6),
        scale=np.full(6, 2.0),
    )

    assert chunks.shape == (2, 2, 18)
    np.testing.assert_allclose(chunks[0, 0], actions[1:4].reshape(-1) / 2.0)


def test_rebase_preserves_demo_relative_command_offset() -> None:
    raw = np.asarray([[[11.0] * 6, [12.0] * 6]], dtype=np.float32)
    rebased = rebase_action_prefixes(
        raw,
        np.asarray([[10.0] * 6]),
        np.asarray([20.0] * 6),
    )
    np.testing.assert_allclose(rebased, [[[21.0] * 6, [22.0] * 6]])


def test_goal_cost_uses_late_horizon_and_closest_goal() -> None:
    predicted = np.asarray(
        [
            [[9.0], [1.0], [0.0]],
            [[0.0], [2.0], [2.0]],
        ],
        dtype=np.float32,
    )
    goals = np.asarray([[0.0], [10.0]], dtype=np.float32)
    costs = terminal_goal_cost(predicted, goals, final_steps=2)

    np.testing.assert_allclose(costs, [0.5, 4.0])
    assert prefix_rmse(np.zeros((2, 6)), np.ones((2, 6))) == pytest.approx(1.0)


def test_state_cost_and_safe_selection() -> None:
    predicted = np.asarray(
        [
            [[2.0] * 6, [0.0] * 6],
            [[2.0] * 6, [1.0] * 6],
        ],
        dtype=np.float32,
    )
    costs = terminal_state_cost(predicted, np.zeros((1, 6)), final_steps=1)
    np.testing.assert_allclose(costs, [0.0, 1.0])

    selected, safe, jumps = select_safe_candidate(
        np.asarray([0.0, 1.0]),
        np.asarray([[50.0] * 6, [5.0] * 6]),
        np.zeros(6),
        max_joint_jump=20.0,
    )
    assert selected == 1
    assert safe.tolist() == [False, True]
    np.testing.assert_allclose(jumps, [50.0, 5.0])


def test_safe_selection_reports_stop_condition_without_feasible_candidate() -> None:
    selected, safe, _ = select_safe_candidate(
        np.asarray([0.0, 1.0]),
        np.asarray([[30.0] * 6, [25.0] * 6]),
        np.zeros(6),
        max_joint_jump=20.0,
    )
    assert selected == 1
    assert not np.any(safe)


def test_deployment_safety_rejects_absolute_limit_violation() -> None:
    current = np.asarray([0.0, -98.0, 98.0, 0.0, 0.0, 50.0], dtype=np.float32)
    prefixes = np.repeat(current[None, None], 2, axis=0)
    prefixes = np.repeat(prefixes, 3, axis=1)
    prefixes[0, :, 1] = -103.0
    prefixes[1, :, 1] = -95.0

    checks = candidate_safety_checks(
        prefixes[:, 0],
        current,
        max_joint_jump=8.0,
        raw_prefixes=prefixes,
        position_lower=SO101_POSITION_LOWER,
        position_upper=SO101_POSITION_UPPER,
    )

    assert checks.first_jump_ok.tolist() == [True, True]
    assert checks.command_bounds_ok.tolist() == [False, True]
    assert checks.safe.tolist() == [False, True]


def test_deployment_safety_rejects_internal_prefix_jump_and_prediction_oob() -> None:
    current = np.zeros(6, dtype=np.float32)
    prefixes = np.zeros((2, 3, 6), dtype=np.float32)
    prefixes[0, 1, 0] = 20.0
    predicted = np.zeros((2, 2, 6), dtype=np.float32)
    predicted[1, -1, 2] = 105.0

    checks = candidate_safety_checks(
        prefixes[:, 0],
        current,
        max_joint_jump=8.0,
        max_prefix_step_jump=8.0,
        raw_prefixes=prefixes,
        predicted_states=predicted,
        position_lower=SO101_POSITION_LOWER,
        position_upper=SO101_POSITION_UPPER,
    )

    assert checks.prefix_step_ok.tolist() == [False, True]
    assert checks.prediction_bounds_ok.tolist() == [True, False]
    assert checks.hard_command_safe.tolist() == [False, True]
    assert checks.prediction_plausible.tolist() == [True, False]
    assert checks.safe.tolist() == [False, False]


def test_prediction_margin_does_not_relax_command_bounds() -> None:
    current = np.asarray([0.0, -98.0, 98.0, 0.0, 0.0, 1.0], dtype=np.float32)
    prefixes = np.repeat(current[None, None], 2, axis=0)
    prefixes = np.repeat(prefixes, 2, axis=1)
    prefixes[0, :, 5] = -1.0
    predicted = np.repeat(current[None, None], 2, axis=0)
    predicted = np.repeat(predicted, 2, axis=1)
    predicted[:, -1, 5] = -2.0

    checks = candidate_safety_checks(
        prefixes[:, 0],
        current,
        max_joint_jump=8.0,
        raw_prefixes=prefixes,
        predicted_states=predicted,
        position_lower=SO101_POSITION_LOWER,
        position_upper=SO101_POSITION_UPPER,
        prediction_bound_margin=3.0,
    )

    assert checks.command_bounds_ok.tolist() == [False, True]
    assert checks.prediction_bounds_ok.tolist() == [True, True]
    assert checks.hard_command_safe.tolist() == [False, True]
    assert checks.prediction_plausible.tolist() == [True, True]
    assert checks.safe.tolist() == [False, True]


@pytest.mark.parametrize("margin", [-0.1, np.inf, np.nan])
def test_prediction_margin_must_be_finite_and_non_negative(margin: float) -> None:
    with pytest.raises(ValueError, match="prediction_bound_margin"):
        candidate_safety_checks(
            np.zeros((1, 6), dtype=np.float32),
            np.zeros(6, dtype=np.float32),
            max_joint_jump=8.0,
            prediction_bound_margin=margin,
        )


def _camera_data(camera: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_id="real_demo",
        camera=camera,
        frames=np.zeros((4, 8, 8, 3), dtype=np.uint8),
        actions=np.arange(24, dtype=np.float32).reshape(4, 6),
        states=np.arange(24, dtype=np.float32).reshape(4, 6) + 1,
        episodes=np.asarray([0, 0, 1, 1], dtype=np.int32),
        frame_indices=np.asarray([0, 1, 0, 1], dtype=np.int32),
        tasks=np.zeros(4, dtype=np.int16),
    )


def test_paired_camera_validation_allows_different_pixels() -> None:
    wrist = _camera_data("observation.images.wrist")
    column = _camera_data("observation.images.column")
    column.frames[:] = 255

    contract = validate_paired_camera_data(wrist, column)

    assert contract.frames == 4
    assert contract.primary_camera.endswith("wrist")
    assert contract.secondary_camera.endswith("column")


@pytest.mark.parametrize("field", ["actions", "states", "episodes", "frame_indices", "tasks"])
def test_paired_camera_validation_rejects_physical_row_mismatch(field: str) -> None:
    wrist = _camera_data("observation.images.wrist")
    column = _camera_data("observation.images.column")
    value = np.asarray(getattr(column, field)).copy()
    value.reshape(-1)[0] += 1
    setattr(column, field, value)

    with pytest.raises(ValueError, match=field):
        validate_paired_camera_data(wrist, column)
