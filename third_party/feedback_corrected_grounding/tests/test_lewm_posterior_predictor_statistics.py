from __future__ import annotations

from copy import deepcopy

import pytest

from fcpg.lewm_posterior_predictor_statistics import (
    predictor_isolation_gate_decision,
    summarize_predictor_isolation_statistics,
)


METHODS = (
    "factual",
    "hold_last",
    "constant_velocity",
    "zero_action",
    "shuffled_action",
)


def _records(
    *,
    camera_views: int = 2,
    source_count: int = 6,
    task_effects: dict[str, tuple[float, float, float, float, float]] | None = None,
) -> list[dict[str, object]]:
    effects = task_effects or {
        "button": (7.0, 10.0, 9.0, 8.0, 8.0),
        "pick": (6.5, 10.0, 9.0, 8.0, 8.0),
        "push": (7.5, 10.0, 9.0, 8.0, 8.0),
        "reach": (6.8, 10.0, 9.0, 8.0, 8.0),
    }
    rows: list[dict[str, object]] = []
    for task_index, (task, method_means) in enumerate(effects.items()):
        for source in range(source_count):
            # Source-specific variation makes the bootstrap interval nonzero.
            multiplier = 0.65 + 0.14 * source + 0.02 * task_index
            for camera_index in range(camera_views):
                camera = f"camera-{camera_index}"
                view_id = f"{task}-source-{source}-{camera}"
                for event_id in ("early", "late"):
                    for phase_index in range(3):
                        for method, mean in zip(METHODS, method_means, strict=True):
                            rows.append(
                                {
                                    "task": task,
                                    "view_episode_id": view_id,
                                    "source_trajectory_id": source,
                                    "camera_view": camera,
                                    "event_id": event_id,
                                    "phase_index": phase_index,
                                    "method": method,
                                    "squared_error": mean * multiplier,
                                }
                            )
    return rows


def _summary(records=None, **kwargs):
    options = {"bootstrap_samples": 600, "bootstrap_seed": 29, **kwargs}
    return summarize_predictor_isolation_statistics(
        _records() if records is None else records, **options
    )


def test_fixed_gate_passes_all_four_comparisons() -> None:
    summary = _summary()

    assert summary["decision"] == "Go"
    assert summary["passed"] is True
    assert summary["coverage"]["task_count"] == 4
    assert summary["coverage"]["source_trajectory_count"] == 24
    assert summary["coverage"]["view_episode_count"] == 48
    assert summary["bootstrap"]["unit"] == ["task", "source_trajectory_id"]
    assert summary["aggregation"]["view_reduction"] == (
        "equal_mean_of_paired_camera_views_within_source"
    )

    comparisons = summary["comparisons"]
    assert comparisons["hold_last"]["relative_improvement"] > 0.10
    assert comparisons["constant_velocity"]["relative_improvement"] > 0.10
    assert comparisons["zero_action"]["relative_improvement"] > 0.05
    assert comparisons["shuffled_action"]["relative_improvement"] > 0.05
    assert all(
        row["relative_improvement_95_ci"][0] > 0.0
        for row in comparisons.values()
    )
    assert all(row["positive_task_count"] == 4 for row in comparisons.values())
    assert summary["gate"]["thresholds"]["required_positive_tasks"] == 3


def test_reduction_is_phase_then_event_then_equal_camera_view() -> None:
    rows: list[dict[str, object]] = []
    values = {
        "view-a": {"event-a": [0.0, 4.0], "event-b": [20.0]},
        "view-b": {"event-a": [8.0, 12.0, 16.0]},
    }
    # view-a = mean(mean(0,4), 20) = 11; view-b = 12; source = 11.5.
    for task in ("button", "pick", "push", "reach"):
        for view_id, events in values.items():
            camera = "left" if view_id == "view-a" else "right"
            for event_id, phases in events.items():
                for phase_index, factual in enumerate(phases):
                    for method, scale in {
                        "factual": 1.0,
                        "hold_last": 2.0,
                        "constant_velocity": 2.0,
                        "zero_action": 2.0,
                        "shuffled_action": 2.0,
                    }.items():
                        rows.append(
                            {
                                "task": task,
                                "view_episode_id": f"{task}-{view_id}",
                                "source_trajectory_id": "source",
                                "camera_view": camera,
                                "event_id": event_id,
                                "phase_index": phase_index,
                                "method": method,
                                "squared_error": factual * scale,
                            }
                        )

    summary = _summary(rows, bootstrap_samples=20)
    hold = summary["comparisons"]["hold_last"]
    assert hold["per_task"]["button"]["factual_mse"] == pytest.approx(11.5)
    assert hold["per_task"]["button"]["baseline_mse"] == pytest.approx(23.0)
    assert hold["relative_improvement"] == pytest.approx(0.5)


def test_pseudoreplicated_camera_views_do_not_narrow_confidence_interval() -> None:
    paired = _summary(_records(camera_views=2), bootstrap_samples=2_000)
    replicated = _summary(_records(camera_views=20), bootstrap_samples=2_000)

    assert paired["coverage"]["source_trajectory_count"] == 24
    assert replicated["coverage"]["source_trajectory_count"] == 24
    assert paired["coverage"]["view_episode_count"] == 48
    assert replicated["coverage"]["view_episode_count"] == 480
    for baseline in ("hold_last", "constant_velocity", "zero_action", "shuffled_action"):
        left = paired["comparisons"][baseline]
        right = replicated["comparisons"][baseline]
        assert right["relative_improvement"] == pytest.approx(
            left["relative_improvement"], abs=1e-15
        )
        assert right["relative_improvement_95_ci"] == pytest.approx(
            left["relative_improvement_95_ci"], abs=1e-15
        )


def test_gate_fails_when_only_two_of_four_tasks_improve() -> None:
    records = _records(
        task_effects={
            "button": (5.0, 10.0, 10.0, 10.0, 10.0),
            "pick": (5.0, 10.0, 10.0, 10.0, 10.0),
            "push": (10.5, 10.0, 10.0, 10.0, 10.0),
            "reach": (10.5, 10.0, 10.0, 10.0, 10.0),
        }
    )
    summary = _summary(records)

    assert summary["decision"] == "No-Go"
    assert summary["comparisons"]["hold_last"]["positive_task_count"] == 2
    check = summary["gate"]["checks"]["factual_over_hold_last_task_direction"]
    assert check == {
        "passed": False,
        "observed": 2,
        "operator": ">=",
        "threshold": 3,
    }


def test_support_mismatch_and_camera_pairing_are_rejected() -> None:
    rows = _records()
    rows.pop()
    with pytest.raises(ValueError, match="method support differs"):
        _summary(rows)

    rows = _records()
    rows = [
        row
        for row in rows
        if not (
            row["task"] == "button"
            and row["source_trajectory_id"] == 0
            and row["camera_view"] == "camera-1"
        )
    ]
    with pytest.raises(ValueError, match="at least two paired camera views"):
        _summary(rows)


def test_conflicting_aliases_and_nonpositive_baseline_are_rejected() -> None:
    rows = _records()
    rows[0] = {**rows[0], "physical_episode_id": "different-view"}
    with pytest.raises(ValueError, match="conflicting record 0 view episode aliases"):
        _summary(rows)

    rows = _records()
    for row in rows:
        if row["method"] == "hold_last" and row["task"] == "button":
            row["squared_error"] = 0.0
    with pytest.raises(ValueError, match="hold_last source-trajectory squared error"):
        _summary(rows)


def test_decision_validates_required_comparisons() -> None:
    summary = _summary()
    incomplete = deepcopy(summary["comparisons"])
    incomplete.pop("zero_action")
    with pytest.raises(ValueError, match="required predictor comparisons"):
        predictor_isolation_gate_decision(incomplete)
