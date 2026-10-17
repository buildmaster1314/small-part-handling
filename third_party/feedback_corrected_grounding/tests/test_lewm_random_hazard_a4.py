from __future__ import annotations

import numpy as np
import pytest

from fcpg.lewm_random_hazard_a4 import (
    FACTUAL_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CV_METHOD,
    METHODS,
    WRONG_ACTION_METHOD,
    ZERO_ACTION_METHOD,
    latent_constant_velocity,
    sample_random_hazard_schedule,
    summarize_random_hazard_records,
)
from scripts.evaluate_lewm_random_hazard_a4 import (
    _future_actions_through_age,
    _wrong_future_actions,
)


def test_random_hazard_schedule_is_deterministic_and_well_formed() -> None:
    left = sample_random_hazard_schedule(60, seed=17)
    right = sample_random_hazard_schedule(60, seed=17)

    np.testing.assert_array_equal(left.active, right.active)
    np.testing.assert_array_equal(left.event_id, right.event_id)
    np.testing.assert_array_equal(left.event_age, right.event_age)
    assert left.events == right.events
    assert left.active[:2].tolist() == [False, False]
    assert 1 <= len(left.events) <= 3
    for event in left.events:
        start = event["start"]
        duration = event["duration"]
        assert left.event_age[start : start + duration].tolist() == list(
            range(1, duration + 1)
        )


def test_random_hazard_schedule_forces_one_event_without_state_inputs() -> None:
    schedule = sample_random_hazard_schedule(
        20,
        seed=3,
        trigger_probability=0.0,
        minimum_duration=2,
        maximum_duration=4,
    )
    assert len(schedule.events) == 1
    assert 2 <= int(schedule.active.sum()) <= 4


def test_random_hazard_schedule_rejects_short_sequence() -> None:
    with pytest.raises(ValueError, match="too short"):
        sample_random_hazard_schedule(
            3,
            seed=1,
            warmup_steps=2,
            minimum_duration=2,
            trigger_probability=0.0,
        )


def test_latent_constant_velocity_propagates_latest_delta() -> None:
    first = np.asarray([[1.0, 2.0]], dtype=np.float32)
    second = np.asarray([[4.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(
        latent_constant_velocity([first, second]),
        np.asarray([[7.0, 14.0]], dtype=np.float32),
    )


def test_future_actions_contain_only_actions_executed_by_event_age() -> None:
    actions = np.arange(80 * 4, dtype=np.float32).reshape(80, 4)
    prefix = _future_actions_through_age(
        actions,
        first_transition=2,
        event_age=3,
    )

    assert prefix.shape == (1, 1, 3, 20)
    np.testing.assert_array_equal(prefix[0, 0, 0], actions[10:15].reshape(-1))
    np.testing.assert_array_equal(prefix[0, 0, 2], actions[20:25].reshape(-1))
    wrong = _wrong_future_actions(prefix).reshape(1, 1, 3, 5, 4)
    original = prefix.reshape(1, 1, 3, 5, 4)
    np.testing.assert_array_equal(wrong[..., :3], -original[..., :3])
    np.testing.assert_array_equal(wrong[..., 3], original[..., 3])


def test_summary_detects_factual_predictor_signal() -> None:
    errors = {
        FACTUAL_METHOD: 0.2,
        HOLD_LAST_METHOD: 1.0,
        LATENT_CV_METHOD: 2.0,
        ZERO_ACTION_METHOD: 0.8,
        WRONG_ACTION_METHOD: 0.9,
    }
    rows = []
    for task in ("pick", "towel"):
        for age in (1, 2, 3):
            for method in METHODS:
                rows.append(
                    {
                        "task": task,
                        "task_instance": 0,
                        "task_stage": "active",
                        "event_age": age,
                        "method": method,
                        "anchor_mse": errors[method],
                    }
                )

    summary = summarize_random_hazard_records(rows)

    assert summary["decision"] == "screen_signal"
    assert summary["physical_episode_count"] == 2
    assert summary["hidden_record_count"] == 6
    assert summary["comparisons"][HOLD_LAST_METHOD][
        "relative_improvement"
    ] == pytest.approx(0.8)
    assert set(summary["by_event_age"]) == {"age_1", "age_2", "age_3"}
    assert summary["per_task_stage_anchor_mse"]["pick"]["active"][
        FACTUAL_METHOD
    ] == pytest.approx(0.2)
