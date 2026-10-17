from __future__ import annotations

import copy

import numpy as np
import pytest

from fcpg.lewm_known_drop_statistics import (
    PROTOCOL,
    summarize_known_drop_comparison,
)


def _row(
    method: str,
    *,
    task: str = "pick",
    model_seed: int = 1,
    physical_episode: str | int = "episode-0",
    drop_duration: float = 2.0,
    success: float = 0.0,
    action_distance: float = 1.0,
) -> dict[str, object]:
    return {
        "method": method,
        "task": task,
        "model_seed": model_seed,
        "physical_episode": physical_episode,
        "drop_duration": drop_duration,
        "success": success,
        "action_distance": action_distance,
    }


def _complete_panel() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values = {
        "lewm": (1.0, 1.0),
        "baseline": (0.0, 3.0),
        "no_action": (0.0, 2.0),
        "oracle": (1.0, 0.5),
    }
    for task in ("pick", "push"):
        for episode in (0, 1):
            for duration in (1.0, 4.0):
                for method, (success, distance) in values.items():
                    rows.append(
                        _row(
                            method,
                            task=task,
                            physical_episode=episode,
                            drop_duration=duration,
                            success=success,
                            action_distance=distance,
                        )
                    )
    return rows


def test_summarizes_candidate_no_action_and_oracle_headroom() -> None:
    result = summarize_known_drop_comparison(
        _complete_panel(),
        candidate="lewm",
        baseline="baseline",
        matched_no_action="no_action",
        oracle="oracle",
        bootstrap_samples=200,
        seed=19,
    )

    assert result["protocol"] == PROTOCOL
    assert result["method_panel"] == ["baseline", "lewm", "no_action", "oracle"]
    comparisons = result["comparisons"]
    primary = comparisons["candidate_vs_baseline"]
    assert primary["success"]["task_macro_delta"] == pytest.approx(1.0)
    assert primary["success"]["ci"] == pytest.approx([1.0, 1.0])
    assert primary["action_distance"]["task_macro_improvement"] == pytest.approx(
        2.0
    )
    assert primary["action_distance"]["ci"] == pytest.approx([2.0, 2.0])
    assert primary["task_count"] == 2
    assert primary["physical_episode_count"] == 4
    assert primary["paired_row_count"] == 8
    assert primary["per_task"]["pick"]["physical_episode_count"] == 2
    assert primary["per_task"]["pick"]["paired_row_count"] == 4

    versus_no_action = comparisons["candidate_vs_matched_no_action"]
    assert versus_no_action["success"]["task_macro_delta"] == pytest.approx(1.0)
    assert versus_no_action["action_distance"][
        "task_macro_improvement"
    ] == pytest.approx(1.0)

    oracle = comparisons["oracle_headroom_vs_baseline"]
    assert oracle["success"]["task_macro_delta"] == pytest.approx(1.0)
    assert oracle["action_distance"]["task_macro_improvement"] == pytest.approx(
        2.5
    )


def test_aggregates_within_episode_before_task_macro() -> None:
    rows = [
        _row("lewm", physical_episode="one", success=1.0, action_distance=1.0),
        _row("base", physical_episode="one", success=0.0, action_distance=3.0),
    ]
    for duration in (1.0, 2.0, 3.0):
        rows.extend(
            [
                _row(
                    "lewm",
                    physical_episode="many",
                    drop_duration=duration,
                    success=0.0,
                    action_distance=2.0,
                ),
                _row(
                    "base",
                    physical_episode="many",
                    drop_duration=duration,
                    success=0.0,
                    action_distance=2.0,
                ),
            ]
        )

    result = summarize_known_drop_comparison(
        rows,
        candidate="lewm",
        baseline="base",
        bootstrap_samples=300,
        seed=7,
    )
    comparison = result["comparisons"]["candidate_vs_baseline"]
    assert comparison["success"]["task_macro_delta"] == pytest.approx(0.5)
    assert comparison["action_distance"]["task_macro_improvement"] == pytest.approx(
        1.0
    )
    assert comparison["physical_episode_count"] == 2
    assert comparison["paired_row_count"] == 4


def test_bootstrap_is_deterministic_for_fixed_seed() -> None:
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(("pick", "push")):
        for episode in range(8):
            baseline_success = float((episode + task_index) % 3 == 0)
            candidate_success = float(episode % 2 == 0)
            rows.extend(
                [
                    _row(
                        "lewm",
                        task=task,
                        physical_episode=episode,
                        success=candidate_success,
                        action_distance=0.5 + episode / 10.0,
                    ),
                    _row(
                        "base",
                        task=task,
                        physical_episode=episode,
                        success=baseline_success,
                        action_distance=1.0 + episode / 10.0,
                    ),
                ]
            )
    first = summarize_known_drop_comparison(
        rows,
        candidate="lewm",
        baseline="base",
        bootstrap_samples=500,
        seed=101,
    )
    second = summarize_known_drop_comparison(
        rows,
        candidate="lewm",
        baseline="base",
        bootstrap_samples=500,
        seed=101,
    )
    assert first == second


def test_rejects_duplicate_method_for_paired_key() -> None:
    rows = _complete_panel()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate method row"):
        summarize_known_drop_comparison(
            rows,
            candidate="lewm",
            baseline="baseline",
            bootstrap_samples=10,
        )


def test_rejects_incomplete_method_panel() -> None:
    rows = _complete_panel()
    rows.pop()
    with pytest.raises(ValueError, match="incomplete method panel"):
        summarize_known_drop_comparison(
            rows,
            candidate="lewm",
            baseline="baseline",
            bootstrap_samples=10,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("success", np.nan, "success must be finite"),
        ("success", 1.1, r"success must lie in \[0, 1\]"),
        ("action_distance", np.inf, "action_distance must be finite"),
        ("action_distance", -0.1, "action_distance must be non-negative"),
        ("drop_duration", -1.0, "drop_duration must be non-negative"),
    ],
)
def test_rejects_invalid_numeric_values(
    field: str,
    value: float,
    message: str,
) -> None:
    rows = _complete_panel()
    rows[0][field] = value
    with pytest.raises(ValueError, match=message):
        summarize_known_drop_comparison(
            rows,
            candidate="lewm",
            baseline="baseline",
            bootstrap_samples=10,
        )


def test_rejects_missing_field_and_absent_requested_method() -> None:
    rows = _complete_panel()
    del rows[0]["action_distance"]
    with pytest.raises(ValueError, match="missing fields"):
        summarize_known_drop_comparison(
            rows,
            candidate="lewm",
            baseline="baseline",
            bootstrap_samples=10,
        )

    two_method_rows = [
        row for row in _complete_panel() if row["method"] in {"lewm", "baseline"}
    ]
    with pytest.raises(ValueError, match="requested methods are absent"):
        summarize_known_drop_comparison(
            two_method_rows,
            candidate="lewm",
            baseline="baseline",
            matched_no_action="no_action",
            bootstrap_samples=10,
        )


def test_rejects_invalid_comparison_and_bootstrap_contract() -> None:
    rows = _complete_panel()
    with pytest.raises(ValueError, match="candidate and baseline must differ"):
        summarize_known_drop_comparison(
            rows,
            candidate="lewm",
            baseline="lewm",
            bootstrap_samples=10,
        )
    with pytest.raises(ValueError, match="bootstrap_samples must be positive"):
        summarize_known_drop_comparison(
            rows,
            candidate="lewm",
            baseline="baseline",
            bootstrap_samples=0,
        )
    with pytest.raises(ValueError, match="confidence must lie"):
        summarize_known_drop_comparison(
            rows,
            candidate="lewm",
            baseline="baseline",
            bootstrap_samples=10,
            confidence=1.0,
        )
