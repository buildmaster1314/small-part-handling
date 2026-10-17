from __future__ import annotations

from scripts.evaluate_lewm_rule_observer_rollout import (
    METHODS,
    method_specifications,
    summarize_rows,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode in range(2):
        for index, method in enumerate(METHODS):
            rows.append(
                {
                    "physical_episode": episode,
                    "method": method,
                    "sustained_success": float(method == "lewm_rule"),
                    "success": float(method == "lewm_rule"),
                    "final_success": float(method == "lewm_rule"),
                    "success_fraction": float(index) / 10.0,
                    "reward_sum": float(index),
                    "post_drop_success_fraction": float(index) / 10.0,
                    "selector_true_positive": 4 if method.endswith("rule") else 0,
                    "selector_false_positive": 0,
                    "selector_false_negative": 2 if method.endswith("rule") else 0,
                    "selector_true_negative": 8 if method.endswith("rule") else 0,
                }
            )
    return rows


def test_method_panel_keeps_known_upper_and_rule_matched_baselines() -> None:
    specs = method_specifications(15801)
    assert tuple(value[0] for value in specs) == METHODS
    assert specs[2][1] == specs[3][1]
    assert not specs[2][2]
    assert specs[3][2]


def test_summary_uses_paired_physical_episodes() -> None:
    summary = summarize_rows(_rows(), bootstrap_samples=100, seed=7)
    assert summary["physical_episodes"] == 2
    assert summary["comparisons"]["lewm_rule_vs_measurement_only"][
        "sustained_success"
    ]["delta"] == 1.0
    assert summary["selector_summary"]["lewm_rule"]["recall"] == 2.0 / 3.0
