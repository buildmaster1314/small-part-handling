from __future__ import annotations

import pytest
import torch

from fcpg.lewm_pixel_observer_eval import (
    _EpisodeErrors,
    build_g0_gate,
    paired_episode_task_bootstrap,
)


TASKS = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)


def _paired_rows(
    improvements_by_source: dict[int, float],
    *,
    tasks: tuple[str, ...] = TASKS,
    episodes_by_source: dict[int, int] | None = None,
    support_by_cell: dict[tuple[int, str], dict[str, int]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate = []
    baseline = []
    episode_counts = episodes_by_source or {
        source: 30 for source in improvements_by_source
    }
    for source, improvement in improvements_by_source.items():
        for task in tasks:
            support = {
                "possible_anchor_count": 100,
                "eligible_anchor_count": 100,
                "possible_window_count": 10,
                "contributing_window_count": 10,
                "measurement_eligible_anchor_count": 100,
                "measurement_valid_anchor_count": 100,
            }
            if support_by_cell is not None:
                support.update(support_by_cell.get((source, task), {}))
            for episode in range(episode_counts[source]):
                baseline.append(
                    {
                        "task": task,
                        "source_index": source,
                        "episode": episode,
                        "mean_epe_px": 10.0,
                        **support,
                    }
                )
                candidate.append(
                    {
                        "task": task,
                        "source_index": source,
                        "episode": episode,
                        "mean_epe_px": 10.0 - improvement,
                        **support,
                    }
                )
    return candidate, baseline


def _gate(comparison: dict[str, object]) -> dict[str, object]:
    return build_g0_gate(
        phase="confirmatory",
        alphas=(0.5,),
        comparisons={
            "lewm_vs_hold_last_same_alpha|alpha=0.5|H=5": comparison,
            "lewm_vs_raw_fcpg|alpha=0.5|H=5": {
                "point": {"relative_improvement": 0.0},
                "bootstrap": {
                    "relative_improvement_95_ci": [0.0, 0.0]
                },
            },
        },
        sanity_checks={"integrity": {"passed": True}},
    )


def test_episode_metrics_preserve_support_numerators_and_denominators() -> None:
    episode = _EpisodeErrors(source_index=0, task="pick-place-v3", episode=7)
    episode.update(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([True, True, False]),
        torch.tensor([True, False, True]),
        torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
    )
    episode.update(
        torch.tensor([4.0, 5.0, 6.0]),
        torch.tensor([False, False, False]),
        torch.tensor([True, True, True]),
        torch.ones(3, 2),
    )

    metrics = episode.metrics()

    assert metrics["possible_anchor_count"] == 6
    assert metrics["eligible_anchor_count"] == 2
    assert metrics["eligible_anchor_fraction"] == pytest.approx(2 / 6)
    assert metrics["possible_window_count"] == 2
    assert metrics["contributing_window_count"] == 1
    assert metrics["contributing_window_fraction"] == pytest.approx(0.5)
    assert metrics["measurement_valid_anchor_count"] == 1
    assert metrics["measurement_eligible_anchor_count"] == 2
    assert metrics["measurement_coverage"] == pytest.approx(0.5)


def test_macro_weights_sources_equally_despite_episode_count_imbalance() -> None:
    candidate, baseline = _paired_rows(
        {0: 2.0, 1: -1.0},
        tasks=("pick-place-v3",),
        episodes_by_source={0: 100, 1: 1},
    )

    result = paired_episode_task_bootstrap(
        candidate,
        baseline,
        samples=500,
        seed=6144,
    )

    assert result["point"]["improvement_px"] == pytest.approx(0.5)
    assert result["per_task"]["pick-place-v3"]["improvement_px"] == pytest.approx(
        0.5
    )
    assert result["per_source"]["source_0"]["improvement_px"] == pytest.approx(2.0)
    assert result["per_source"]["source_1"]["improvement_px"] == pytest.approx(-1.0)
    assert result["bootstrap"]["difference_95_ci"] == pytest.approx([0.5, 0.5])


def test_development_style_task_specific_sources_remain_equal_task_macro() -> None:
    candidate = []
    baseline = []
    for source, (task, improvement) in enumerate(
        zip(TASKS, (1.0, 2.0, 3.0, 4.0), strict=True)
    ):
        source_candidate, source_baseline = _paired_rows(
            {source: improvement},
            tasks=(task,),
            episodes_by_source={source: 5},
        )
        candidate.extend(source_candidate)
        baseline.extend(source_baseline)

    result = paired_episode_task_bootstrap(
        candidate,
        baseline,
        samples=500,
        seed=6144,
    )

    assert result["point"]["improvement_px"] == pytest.approx(2.5)
    assert set(result["per_task"]) == set(TASKS)
    assert set(result["per_source"]) == {
        "source_0",
        "source_1",
        "source_2",
        "source_3",
    }


def test_confirmatory_gate_rejects_positive_pooled_effect_with_harmed_cohort() -> None:
    candidate, baseline = _paired_rows({0: 2.0, 1: -0.8})
    comparison = paired_episode_task_bootstrap(
        candidate,
        baseline,
        samples=1_000,
        seed=6144,
    )

    gate = _gate(comparison)

    assert comparison["bootstrap"]["difference_95_ci"][0] > 0.0
    assert gate["conditions"]["h5_paired_ci_lower_positive"]["passed"] is True
    assert gate["conditions"]["no_task_harmful_degradation"]["passed"] is True
    cohort = gate["conditions"][
        "each_camera_cohort_h5_paired_ci_lower_positive"
    ]
    assert cohort["values"]["rotation5"] > 0.0
    assert cohort["values"]["translation4"] < 0.0
    assert cohort["passed"] is False
    assert gate["decision"] == "No-Go"


def test_confirmatory_gate_accepts_only_when_both_camera_cohorts_pass() -> None:
    candidate, baseline = _paired_rows({0: 1.0, 1: 0.5})
    comparison = paired_episode_task_bootstrap(
        candidate,
        baseline,
        samples=1_000,
        seed=6144,
    )

    gate = _gate(comparison)

    assert gate["conditions"][
        "each_camera_cohort_h5_paired_ci_lower_positive"
    ]["passed"] is True
    assert gate["conditions"][
        "each_source_task_eligible_anchor_fraction"
    ]["passed"] is True
    assert gate["conditions"][
        "each_source_task_contributing_window_fraction"
    ]["passed"] is True
    assert gate["conditions"][
        "each_source_task_measurement_coverage"
    ]["passed"] is True
    assert gate["conditions"][
        "each_source_task_relative_improvement_ci_noninferiority"
    ]["passed"] is True
    assert gate["decision"] == "Go"


@pytest.mark.parametrize(
    ("support_override", "condition"),
    (
        (
            {
                "eligible_anchor_count": 94,
                "measurement_eligible_anchor_count": 94,
                "measurement_valid_anchor_count": 94,
            },
            "each_source_task_eligible_anchor_fraction",
        ),
        (
            {
                "eligible_anchor_count": 90,
                "contributing_window_count": 9,
                "measurement_eligible_anchor_count": 90,
                "measurement_valid_anchor_count": 90,
            },
            "each_source_task_contributing_window_fraction",
        ),
        (
            {"measurement_valid_anchor_count": 94},
            "each_source_task_measurement_coverage",
        ),
    ),
)
def test_confirmatory_gate_rejects_one_low_support_cell_despite_go_mean(
    support_override: dict[str, int], condition: str
) -> None:
    candidate, baseline = _paired_rows(
        {0: 1.0, 1: 1.0},
        support_by_cell={(1, "push-v3"): support_override},
    )
    comparison = paired_episode_task_bootstrap(
        candidate,
        baseline,
        samples=1_000,
        seed=6144,
    )

    gate = _gate(comparison)

    assert comparison["bootstrap"]["difference_95_ci"][0] > 0.0
    assert gate["conditions"]["h5_paired_ci_lower_positive"]["passed"] is True
    assert gate["conditions"][condition]["passed"] is False
    assert gate["decision"] == "No-Go"


def test_confirmatory_gate_rejects_one_degraded_source_task_ci() -> None:
    candidate, baseline = _paired_rows({0: 2.0, 1: 1.0})
    for row in candidate:
        if row["source_index"] == 0 and row["task"] == "push-v3":
            row["mean_epe_px"] = 11.2
    comparison = paired_episode_task_bootstrap(
        candidate,
        baseline,
        samples=1_000,
        seed=6144,
    )

    gate = _gate(comparison)

    assert comparison["bootstrap"]["difference_95_ci"][0] > 0.0
    assert gate["conditions"][
        "each_camera_cohort_h5_paired_ci_lower_positive"
    ]["passed"] is True
    assert gate["conditions"]["no_task_harmful_degradation"]["passed"] is True
    source_task_guard = gate["conditions"][
        "each_source_task_relative_improvement_ci_noninferiority"
    ]
    assert source_task_guard["values"]["rotation5|push-v3"] == pytest.approx(
        -0.12
    )
    assert source_task_guard["passed"] is False
    assert gate["decision"] == "No-Go"


def test_raw_fcpg_guard_uses_bootstrap_ci_not_point_estimate() -> None:
    candidate, baseline = _paired_rows({0: 1.0, 1: 1.0})
    comparison = paired_episode_task_bootstrap(
        candidate,
        baseline,
        samples=1_000,
        seed=6144,
    )
    gate = build_g0_gate(
        phase="confirmatory",
        alphas=(0.5,),
        comparisons={
            "lewm_vs_hold_last_same_alpha|alpha=0.5|H=5": comparison,
            "lewm_vs_raw_fcpg|alpha=0.5|H=5": {
                "point": {"relative_improvement": 0.01},
                "bootstrap": {
                    "relative_improvement_95_ci": [-0.06, 0.08]
                },
            },
        },
        sanity_checks={"integrity": {"passed": True}},
    )

    guard = gate["conditions"]["raw_fcpg_practical_noninferiority"]
    assert guard["value"] == pytest.approx(-0.06)
    assert guard["passed"] is False
    assert gate["decision"] == "No-Go"
