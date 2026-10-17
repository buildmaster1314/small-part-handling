from __future__ import annotations

from copy import deepcopy
import math

import pytest

from fcpg.lewm_posterior_statistics import (
    DEFAULT_BOOTSTRAP_SEED,
    PRIMARY_WINDOW,
    paired_task_anchor_macro_bootstrap,
    summarize_posterior_statistics,
)


def _records(*, include_r0: bool = True) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    cells = (
        ("large", 0, 0, 10.0, 8.0, 9.0),
        ("large", 1, 0, 10.0, 8.0, 9.0),
        ("large", 0, 1, 100.0, 90.0, 95.0),
        ("large", 1, 1, 100.0, 90.0, 95.0),
        ("small", "episode-a", "tip", 1.0, 2.0, 1.5),
    )
    windows = (PRIMARY_WINDOW, "r0") if include_r0 else (PRIMARY_WINDOW,)
    for task, episode_id, anchor, baseline, posterior, control in cells:
        for window in windows:
            multiplier = 100.0 if window == "r0" else 1.0
            for method, mse in (
                ("baseline", baseline),
                ("posterior", posterior),
                ("control", control),
            ):
                records.append(
                    {
                        "task": task,
                        "episode_id": episode_id,
                        "anchor": anchor,
                        "method": method,
                        "window": window,
                        "mse": mse * multiplier,
                    }
                )
    return records


def test_task_anchor_macro_is_equal_weighted_and_reports_coverage() -> None:
    summary = summarize_posterior_statistics(
        _records(), "baseline", bootstrap_samples=100, bootstrap_seed=7
    )
    posterior = summary["comparisons"]["posterior"]

    # The three task-anchor improvements are +2, +10, and -1. Episode counts
    # and MSE scales do not change their macro weights.
    assert posterior["improvement"] == pytest.approx(11.0 / 3.0)
    assert posterior["baseline_macro_mse"] == pytest.approx(111.0 / 3.0)
    assert posterior["method_macro_mse"] == pytest.approx(100.0 / 3.0)
    assert posterior["improvement_95_ci"] == pytest.approx([11.0 / 3.0] * 2)
    assert [row["improvement"] for row in posterior["per_task_anchor"]] == [
        2.0,
        10.0,
        -1.0,
    ]

    coverage = summary["coverage"]
    assert coverage["support_identical_across_methods"] is True
    assert coverage["task_count"] == 2
    assert coverage["task_anchor_count"] == 3
    assert coverage["physical_episode_count"] == 3
    assert coverage["primary_episode_anchor_count_per_method"] == 5
    assert coverage["primary_episode_anchor_fraction"] == 1.0
    assert posterior["coverage"]["method_to_baseline_support_fraction"] == 1.0


def test_whole_episode_draw_is_shared_across_anchors() -> None:
    records = []
    # Within each episode, the two anchor effects cancel. A whole-episode
    # cluster bootstrap is therefore identically zero. Independent per-anchor
    # resampling would produce a non-degenerate interval.
    effects = {
        (0, 0): 1.0,
        (0, 1): -1.0,
        (1, 0): -1.0,
        (1, 1): 1.0,
    }
    for (episode_id, anchor), improvement in effects.items():
        for method, mse in (
            ("baseline", 10.0),
            ("posterior", 10.0 - improvement),
        ):
            records.append(
                {
                    "task": "pick",
                    "episode_id": episode_id,
                    "anchor": anchor,
                    "method": method,
                    "window": PRIMARY_WINDOW,
                    "mse": mse,
                }
            )

    summary = paired_task_anchor_macro_bootstrap(
        records, "baseline", bootstrap_samples=1_000, bootstrap_seed=19
    )
    comparison = summary["comparisons"]["posterior"]
    assert comparison["improvement"] == 0.0
    assert comparison["improvement_95_ci"] == [0.0, 0.0]


def test_fixed_seed_is_reproducible_and_record_order_independent() -> None:
    records = _records(include_r0=False)
    records[3]["mse"] = 7.0
    first = summarize_posterior_statistics(
        records, "baseline", bootstrap_samples=300, bootstrap_seed=23
    )
    second = summarize_posterior_statistics(
        list(reversed(records)),
        "baseline",
        bootstrap_samples=300,
        bootstrap_seed=23,
    )
    assert first == second
    assert first["bootstrap"]["seed"] == 23
    assert first["bootstrap"]["interval"] == "percentile"


def test_incomplete_anchor_panel_reports_coverage_and_keeps_cluster_draws_valid() -> None:
    records = []
    for episode_id in range(3):
        anchors = (0, 1) if episode_id == 0 else (0,)
        for anchor in anchors:
            for method, mse in (("baseline", 4.0), ("posterior", 3.0)):
                records.append(
                    {
                        "task": "pick",
                        "episode_id": episode_id,
                        "anchor": anchor,
                        "method": method,
                        "window": PRIMARY_WINDOW,
                        "mse": mse,
                    }
                )

    summary = summarize_posterior_statistics(
        records, "baseline", bootstrap_samples=200, bootstrap_seed=29
    )
    assert summary["coverage"]["primary_episode_anchor_fraction"] == pytest.approx(
        2.0 / 3.0
    )
    assert summary["coverage"]["complete_physical_episode_count"] == 1
    assert summary["bootstrap"]["incomplete_panel_cluster_redraws"] > 0
    assert summary["comparisons"]["posterior"]["improvement_95_ci"] == [1.0, 1.0]


def test_r0_is_rejected_as_primary_but_allowed_as_secondary_data() -> None:
    summary = summarize_posterior_statistics(
        _records(), "baseline", bootstrap_samples=10
    )
    assert summary["primary_window"] == PRIMARY_WINDOW
    assert summary["bootstrap"]["seed"] == DEFAULT_BOOTSTRAP_SEED == 9_531
    assert summary["coverage"]["windows"] == [PRIMARY_WINDOW, "r0"]

    with pytest.raises(ValueError, match="cannot be primary"):
        summarize_posterior_statistics(
            _records(), "baseline", primary_window="r0", bootstrap_samples=10
        )
    with pytest.raises(ValueError, match="must be exactly"):
        summarize_posterior_statistics(
            _records(), "baseline", primary_window="other", bootstrap_samples=10
        )


def test_rejects_duplicate_records() -> None:
    records = _records()
    records.append(deepcopy(records[0]))
    with pytest.raises(ValueError, match="duplicate posterior statistic record"):
        summarize_posterior_statistics(records, "baseline", bootstrap_samples=10)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_mse(value: float) -> None:
    records = _records()
    records[0]["mse"] = value
    with pytest.raises(ValueError, match="mse must be finite"):
        summarize_posterior_statistics(records, "baseline", bootstrap_samples=10)


def test_rejects_negative_mse() -> None:
    records = _records()
    records[0]["mse"] = -1.0
    with pytest.raises(ValueError, match="mse must be non-negative"):
        summarize_posterior_statistics(records, "baseline", bootstrap_samples=10)


def test_rejects_method_support_mismatch_in_primary_or_secondary_window() -> None:
    records = _records()
    primary_missing = [
        row
        for row in records
        if not (
            row["method"] == "posterior"
            and row["window"] == PRIMARY_WINDOW
            and row["task"] == "small"
        )
    ]
    with pytest.raises(ValueError, match="method support differs"):
        summarize_posterior_statistics(
            primary_missing, "baseline", bootstrap_samples=10
        )

    secondary_missing = [
        row
        for row in records
        if not (
            row["method"] == "control"
            and row["window"] == "r0"
            and row["task"] == "small"
        )
    ]
    with pytest.raises(ValueError, match="method support differs"):
        summarize_posterior_statistics(
            secondary_missing, "baseline", bootstrap_samples=10
        )


def test_rejects_absent_baseline_and_invalid_bootstrap_parameters() -> None:
    records = _records(include_r0=False)
    with pytest.raises(ValueError, match="baseline method .* is absent"):
        summarize_posterior_statistics(records, "missing", bootstrap_samples=10)
    with pytest.raises(ValueError, match="bootstrap_samples must be at least 1"):
        summarize_posterior_statistics(records, "baseline", bootstrap_samples=0)
    with pytest.raises(ValueError, match="bootstrap_seed must be at least 0"):
        summarize_posterior_statistics(
            records, "baseline", bootstrap_samples=10, bootstrap_seed=-1
        )
