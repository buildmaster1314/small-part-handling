from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

import fcpg.lewm_posterior_randomized_statistics as statistics
from fcpg.lewm_posterior_randomized_statistics import (
    randomized_oracle_decision,
    summarize_randomized_oracle,
)


METHODS = ("measurement", "candidate", "time_only")


def _records(
    *, recovery_events: int = 2
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specifications = {
        ("a", "a0"): {
            "active": {
                "measurement": [999.0, 10.0],
                "candidate": [999.0, 8.0],
                "time_only": [999.0, 9.0],
            },
            "recovery": {
                "measurement": [2.0] * recovery_events,
                "candidate": [3.0] * recovery_events,
                "time_only": [2.5] * recovery_events,
            },
            "clean": {
                "measurement": [1.0],
                "candidate": [1.5],
                "time_only": [1.25],
            },
        },
        ("b", 0): {
            "active": {
                "measurement": [999.0, 100.0, 100.0],
                "candidate": [999.0, 50.0, 50.0],
                "time_only": [999.0, 80.0, 80.0],
            },
            "recovery": {
                "measurement": [4.0] * recovery_events,
                "candidate": [9.0] * recovery_events,
                "time_only": [5.0] * recovery_events,
            },
            "clean": {
                "measurement": [2.0, 2.0],
                "candidate": [4.0, 4.0],
                "time_only": [3.0, 3.0],
            },
        },
        ("b", 1): {
            "active": {
                "measurement": [999.0, 200.0, 200.0, 200.0, 200.0],
                "candidate": [999.0, 100.0, 100.0, 100.0, 100.0],
                "time_only": [999.0, 160.0, 160.0, 160.0, 160.0],
            },
            "recovery": {
                "measurement": [8.0] * recovery_events,
                "candidate": [18.0] * recovery_events,
                "time_only": [10.0] * recovery_events,
            },
            "clean": {
                "measurement": [4.0],
                "candidate": [8.0],
                "time_only": [6.0],
            },
        },
        ("b", 2): {
            "active": {
                "measurement": [999.0, 300.0],
                "candidate": [999.0, 150.0],
                "time_only": [999.0, 240.0],
            },
            "recovery": {
                "measurement": [12.0] * recovery_events,
                "candidate": [27.0] * recovery_events,
                "time_only": [15.0] * recovery_events,
            },
            "clean": {
                "measurement": [6.0, 6.0, 6.0],
                "candidate": [12.0, 12.0, 12.0],
                "time_only": [9.0, 9.0, 9.0],
            },
        },
    }
    for (task, episode_id), windows in specifications.items():
        for window, methods in windows.items():
            for method, event_values in methods.items():
                for event_index, mse in enumerate(event_values):
                    for phase_index in range(5):
                        rows.append(
                            {
                                "task": task,
                                "physical_episode_id": episode_id,
                                "phase_index": phase_index,
                                "window": window,
                                "event_index": event_index,
                                "method": method,
                                "mse": mse,
                            }
                        )
    return rows


def _parented_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    views = (
        (
            "view-0",
            "parent-0",
            {
                "measurement": [999.0, 2.0],
                "candidate": [999.0, 2.0],
                "time_only": [999.0, 2.0],
            },
        ),
        (
            "view-1",
            "parent-0",
            {
                "measurement": [999.0, 8.0, 8.0, 8.0],
                "candidate": [999.0, 8.0, 8.0, 8.0],
                "time_only": [999.0, 8.0, 8.0, 8.0],
            },
        ),
        (
            "view-2",
            "parent-1",
            {
                "measurement": [999.0, 10.0],
                "candidate": [999.0, 0.0],
                "time_only": [999.0, 5.0],
            },
        ),
    )
    recovery_candidate = {"view-0": 1.0, "view-1": 3.0, "view-2": 5.0}
    clean_candidate = {"view-0": 1.0, "view-1": 5.0, "view-2": 1.0}
    for view_id, parent_id, active in views:
        windows = {
            "active": active,
            "recovery": {
                "measurement": [1.0, 1.0],
                "candidate": [recovery_candidate[view_id]] * 2,
                "time_only": [1.0, 1.0],
            },
            "clean": {
                "measurement": [1.0],
                "candidate": [clean_candidate[view_id]],
                "time_only": [1.0],
            },
        }
        for window, methods in windows.items():
            for method, event_values in methods.items():
                for event_index, mse in enumerate(event_values):
                    for phase_index in range(5):
                        rows.append(
                            {
                                "task": "a",
                                "physical_episode_id": view_id,
                                "resampling_unit_id": parent_id,
                                "phase_index": phase_index,
                                "window": window,
                                "event_index": event_index,
                                "method": method,
                                "mse": mse,
                            }
                        )
    return rows


def _summarize(records=None, **kwargs):
    options = {"bootstrap_samples": 200, "bootstrap_seed": 17, **kwargs}
    return summarize_randomized_oracle(
        _records() if records is None else records,
        measurement_method="measurement",
        candidate_method="candidate",
        time_only_method="time_only",
        **options,
    )


def test_phase_event_episode_then_equal_task_macro_reduction() -> None:
    summary = _summarize()
    primary = summary["primary"]

    # Task a contributes 20%; task b contributes 50%. The three task-b
    # episodes and their unequal active durations do not increase task-b weight.
    assert primary["candidate_relative_improvement"] == pytest.approx(0.35)
    assert primary["time_only_relative_improvement"] == pytest.approx(0.15)
    assert primary[
        "candidate_incremental_improvement_over_time_only"
    ] == pytest.approx(0.20)
    assert primary["per_task"]["a"]["candidate_relative_improvement"] == pytest.approx(
        0.20
    )
    assert primary["per_task"]["b"]["candidate_relative_improvement"] == pytest.approx(
        0.50
    )
    assert primary["per_task"]["b"]["measurement_mse"] == pytest.approx(200.0)
    assert primary["method_task_macro_mse"]["measurement"] == pytest.approx(105.0)
    assert summary["recovery"]["normalized_excess_mse"] == pytest.approx(0.075)
    assert summary["recovery"]["excess_mse"] == pytest.approx(5.5)
    assert summary["clean"]["normalized_excess_mse"] == pytest.approx(0.035)
    assert summary["recovery"][
        "normalized_excess_mse_one_sided_95_upper"
    ] == pytest.approx(0.075)
    assert summary["clean"][
        "normalized_excess_mse_one_sided_95_upper"
    ] == pytest.approx(0.035)
    assert summary["coverage"]["view_episode_count"] == 4
    assert summary["coverage"]["physical_episode_count"] == 4
    assert summary["coverage"]["physical_trajectory_count"] == 4
    assert summary["coverage"]["independent_trajectory_count"] == 4
    assert summary["coverage"]["episodes_per_task"] == {"a": 1, "b": 3}
    assert summary["coverage"]["view_episodes_per_task"] == {"a": 1, "b": 3}
    assert summary["bootstrap"]["unit"] == ["task", "resampling_unit_id"]
    assert all(
        row["resampling_unit_id"] == row["physical_episode_id"]
        for row in summary["coverage"]["per_view_episode"]
    )


def test_active_event_zero_is_excluded_and_clean_uses_all_events() -> None:
    summary = _summarize()
    assert summary["primary"]["per_task"]["a"]["measurement_mse"] == 10.0
    assert summary["primary"]["excluded_event_indices"] == [0]
    assert summary["clean"]["candidate_task_macro_mse"] == pytest.approx(4.75)


def test_fixed_seed_is_input_order_and_chunk_size_invariant(monkeypatch) -> None:
    records = _records()
    first = _summarize(records)
    monkeypatch.setattr(statistics, "_MAX_BOOTSTRAP_INDEX_ELEMENTS", 3)
    chunked = _summarize(list(reversed(records)))
    assert first == chunked


def test_bootstrap_matches_shared_task_stratified_episode_draws() -> None:
    samples = 200
    seed = 17
    summary = _summarize()
    rng = np.random.default_rng(seed)
    rng.integers(0, 1, size=(samples, 1))
    draws_b = rng.integers(0, 3, size=(samples, 3))
    measurement = np.asarray([100.0, 200.0, 300.0])
    candidate = np.asarray([50.0, 100.0, 150.0])
    time_only = np.asarray([80.0, 160.0, 240.0])
    expected_candidate = 0.5 * (
        0.2
        + (measurement[draws_b].mean(axis=1) - candidate[draws_b].mean(axis=1))
        / measurement[draws_b].mean(axis=1)
    )
    expected_incremental = 0.5 * (
        0.1
        + (time_only[draws_b].mean(axis=1) - candidate[draws_b].mean(axis=1))
        / measurement[draws_b].mean(axis=1)
    )
    assert summary["primary"][
        "candidate_relative_improvement_95_ci"
    ] == pytest.approx(np.quantile(expected_candidate, [0.025, 0.975]))
    assert summary["primary"][
        "candidate_incremental_improvement_over_time_only_95_ci"
    ] == pytest.approx(np.quantile(expected_incremental, [0.025, 0.975]))


def test_views_are_averaged_within_parent_before_task_effects() -> None:
    summary = _summarize(_parented_records())
    primary = summary["primary"]

    # parent-0 averages view means 2 and 8 to 5; parent-1 contributes 10.
    # Equal parent weighting therefore gives measurement/candidate/time-only
    # task means 7.5/2.5/5.0. Direct view weighting would instead report a
    # 50% candidate improvement.
    assert primary["measurement_mse"] == pytest.approx(7.5)
    assert primary["candidate_mse"] == pytest.approx(2.5)
    assert primary["time_only_mse"] == pytest.approx(5.0)
    assert primary["candidate_relative_improvement"] == pytest.approx(2.0 / 3.0)
    assert primary[
        "candidate_incremental_improvement_over_time_only"
    ] == pytest.approx(1.0 / 3.0)
    assert summary["recovery"]["excess_mse"] == pytest.approx(2.5)
    assert summary["recovery"]["normalized_excess_mse"] == pytest.approx(1.0 / 3.0)
    assert summary["clean"]["excess_mse"] == pytest.approx(1.0)
    assert summary["clean"]["normalized_excess_mse"] == pytest.approx(2.0 / 15.0)
    assert primary["per_task"]["a"]["view_episode_count"] == 3
    assert primary["per_task"]["a"]["independent_trajectory_count"] == 2

    coverage = summary["coverage"]
    assert coverage["view_episode_count"] == 3
    assert coverage["physical_episode_count"] == 2
    assert coverage["physical_trajectory_count"] == 2
    assert coverage["independent_trajectory_count"] == 2
    assert coverage["view_episodes_per_task"] == {"a": 3}
    assert coverage["independent_trajectories_per_task"] == {"a": 2}
    assert {
        row["physical_episode_id"]: row["resampling_unit_id"]
        for row in coverage["per_view_episode"]
    } == {
        "view-0": "parent-0",
        "view-1": "parent-0",
        "view-2": "parent-1",
    }


def test_bootstrap_resamples_independent_parents_not_views() -> None:
    samples = 200
    seed = 29
    summary = _summarize(
        list(reversed(_parented_records())),
        bootstrap_samples=samples,
        bootstrap_seed=seed,
    )
    draws = np.random.default_rng(seed).integers(0, 2, size=(samples, 2))
    measurement = np.asarray([5.0, 10.0])
    candidate = np.asarray([5.0, 0.0])
    time_only = np.asarray([5.0, 5.0])
    measurement_mean = measurement[draws].mean(axis=1)
    expected_candidate = (
        measurement_mean - candidate[draws].mean(axis=1)
    ) / measurement_mean
    expected_incremental = (
        time_only[draws].mean(axis=1) - candidate[draws].mean(axis=1)
    ) / measurement_mean
    assert summary["primary"][
        "candidate_relative_improvement_95_ci"
    ] == pytest.approx(np.quantile(expected_candidate, [0.025, 0.975]))
    assert summary["primary"][
        "candidate_incremental_improvement_over_time_only_95_ci"
    ] == pytest.approx(np.quantile(expected_incremental, [0.025, 0.975]))


def test_rejects_resampling_unit_changes_within_view() -> None:
    records = _parented_records()
    records[0]["resampling_unit_id"] = "different-parent"
    with pytest.raises(ValueError, match="resampling_unit_id changes within view"):
        _summarize(records)


def test_diagnostics_are_explicitly_non_evidentiary() -> None:
    summary = _summarize()
    for name in ("recovery", "clean"):
        diagnostic = summary[name]
        assert diagnostic["feasibility_only"] is True
        assert diagnostic["non_evidentiary"] is True
        assert diagnostic["hard_go_check"] is True
        assert len(diagnostic["normalized_excess_mse_95_ci"]) == 2


def _feasible_decision_summary():
    summary = _summarize()
    summary["recovery"]["normalized_excess_mse_one_sided_95_upper"] = 0.0
    summary["clean"]["normalized_excess_mse_one_sided_95_upper"] = 0.0
    return summary


def test_decision_applies_primary_incremental_and_feasibility_checks() -> None:
    summary = _feasible_decision_summary()
    decision = randomized_oracle_decision(summary)
    assert decision["passed"] is True
    assert decision["decision"] == "Go"
    assert decision["recovery_and_clean"]["hard_go_checks"] is True

    unsafe = deepcopy(summary)
    unsafe["recovery"]["normalized_excess_mse_one_sided_95_upper"] = 0.03
    decision = randomized_oracle_decision(unsafe)
    assert decision["passed"] is False
    assert (
        "recovery_feasibility_upper_below_maximum" in decision["failed_checks"]
    )


def test_decision_rejects_task_harm_and_short_recovery_coverage() -> None:
    summary = _feasible_decision_summary()
    harmed = deepcopy(summary)
    harmed["primary"]["per_task"]["a"]["candidate_relative_improvement"] = -0.01
    decision = randomized_oracle_decision(harmed)
    assert decision["passed"] is False
    assert "every_task_primary_point_effect_nonnegative" in decision["failed_checks"]

    incremental_harm = deepcopy(summary)
    incremental_harm["primary"]["per_task"]["a"][
        "candidate_incremental_improvement_over_time_only"
    ] = -0.01
    decision = randomized_oracle_decision(incremental_harm)
    assert decision["passed"] is False
    assert (
        "every_task_incremental_over_time_only_nonnegative"
        in decision["failed_checks"]
    )

    short = _summarize(_records(recovery_events=1))
    decision = randomized_oracle_decision(short)
    assert decision["passed"] is False
    assert "at_least_two_recovery_events_per_episode" in decision["failed_checks"]


def test_requires_exact_paired_support_and_five_phases() -> None:
    records = _records()
    missing_method_row = [
        row
        for row in records
        if not (
            row["task"] == "a"
            and row["physical_episode_id"] == "a0"
            and row["window"] == "active"
            and row["event_index"] == 1
            and row["phase_index"] == 4
            and row["method"] == "candidate"
        )
    ]
    with pytest.raises(ValueError, match="method support differs"):
        _summarize(missing_method_row)

    missing_all_methods = [
        row
        for row in records
        if not (
            row["task"] == "a"
            and row["physical_episode_id"] == "a0"
            and row["window"] == "active"
            and row["event_index"] == 1
            and row["phase_index"] == 4
        )
    ]
    with pytest.raises(ValueError, match="exactly phases 0..4"):
        _summarize(missing_all_methods)

    noncontiguous = [
        {**row, "event_index": row["event_index"] + 1}
        if row["window"] == "active"
        else row
        for row in records
    ]
    with pytest.raises(ValueError, match="start at zero and remain contiguous"):
        _summarize(noncontiguous)


def test_rejects_duplicates_zero_denominator_and_invalid_rows() -> None:
    records = _records()
    records.append(deepcopy(records[0]))
    with pytest.raises(ValueError, match="duplicate randomized oracle"):
        _summarize(records)

    zero = _records()
    for row in zero:
        if row["task"] == "a" and row["window"] == "active" and row[
            "event_index"
        ] == 1 and row["method"] == "measurement":
            row["mse"] = 0.0
    with pytest.raises(ValueError, match="zero primary measurement"):
        _summarize(zero)

    for value in (math.nan, math.inf, -1.0):
        invalid = _records()
        invalid[0]["mse"] = value
        with pytest.raises(ValueError, match="mse must"):
            _summarize(invalid)


def test_validates_roles_and_scalar_arguments() -> None:
    with pytest.raises(ValueError, match="methods must differ"):
        summarize_randomized_oracle(
            _records(),
            measurement_method="measurement",
            candidate_method="measurement",
            time_only_method="time_only",
            bootstrap_samples=10,
        )
    with pytest.raises(ValueError, match="methods are absent"):
        summarize_randomized_oracle(
            _records(),
            measurement_method="missing",
            candidate_method="candidate",
            time_only_method="time_only",
            bootstrap_samples=10,
        )
    with pytest.raises(ValueError, match="bootstrap_samples must be at least 1"):
        _summarize(bootstrap_samples=0)
    with pytest.raises(ValueError, match="minimum_primary must lie"):
        randomized_oracle_decision(_summarize(), minimum_primary=1.1)
    with pytest.raises(ValueError, match="maximum_clean_excess must lie"):
        randomized_oracle_decision(_summarize(), maximum_clean_excess=1.1)
