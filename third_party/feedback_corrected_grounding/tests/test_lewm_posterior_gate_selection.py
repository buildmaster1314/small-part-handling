from __future__ import annotations

from copy import deepcopy

import pytest

from fcpg.lewm_posterior_gate_selection import (
    FIXED_BETA_CANDIDATES,
    NO_LEWM_METHOD,
    OPTIMIZATION_SEEDS,
    TASKS,
    evaluate_full_gate,
    select_fixed_beta,
    select_median_seed,
    select_validation_only_posterior_gate,
)


FIXED_METHODS = {
    0.25: "fixed_beta_025",
    0.50: "fixed_beta_050",
    0.75: "fixed_beta_075",
    1.00: "fixed_beta_100",
}


def _effect(
    candidate: str,
    control: str,
    *,
    improvement: float = 0.05,
    ci_lower: float = 0.01,
) -> dict[str, object]:
    return {
        "candidate_method": candidate,
        "control_method": control,
        "relative_improvement": improvement,
        "relative_improvement_95_ci": [ci_lower, max(ci_lower, improvement + 0.05)],
    }


def _summary(
    *,
    method: str = "adapter",
    primary_mse: float = 0.8,
    primary_improvement: float = 0.12,
    primary_ci_lower: float = 0.02,
    task_ratio: float = 1.0,
    task_ratio_upper: float = 1.02,
    recovery: float = 0.01,
    recovery_upper: float | None = None,
    indirect: float = 0.01,
    indirect_upper: float | None = None,
    clean: float = 0.01,
    clean_upper: float | None = None,
) -> dict[str, object]:
    recovery_upper = recovery if recovery_upper is None else recovery_upper
    indirect_upper = indirect if indirect_upper is None else indirect_upper
    clean_upper = clean if clean_upper is None else clean_upper
    return {
        "split": "validation",
        "primary": {
            "candidate_method": method,
            "control_method": "measurement_writeback",
            "candidate_macro_mse": primary_mse,
            "relative_improvement": primary_improvement,
            "relative_improvement_95_ci": [
                primary_ci_lower,
                max(primary_ci_lower, primary_improvement + 0.05),
            ],
        },
        "comparators": {
            "adapter_vs_fixed": {
                fixed_method: _effect(method, fixed_method)
                for fixed_method in FIXED_METHODS.values()
            },
            "adapter_vs_no_lewm": _effect(method, NO_LEWM_METHOD),
        },
        "task_safety_ratios": {
            task: {
                "adapter_to_measurement_ratio": task_ratio,
                "adapter_to_measurement_ratio_one_sided_95_upper": (
                    task_ratio_upper
                ),
            }
            for task in TASKS
        },
        "recovery_safety": {
            "normalized_excess_mse": recovery,
            "normalized_excess_mse_one_sided_95_upper": recovery_upper,
        },
        "indirect_safety": {
            "normalized_excess_mse": indirect,
            "normalized_excess_mse_one_sided_95_upper": indirect_upper,
        },
        "clean_safety": {
            "cohort": "explicit_independent_clean_records",
            "normalized_excess_mse": clean,
            "normalized_excess_mse_one_sided_95_upper": clean_upper,
        },
        "coverage": {
            "complete_paired_support": True,
            "complete_estimand_event_method_panels": True,
            "clean_complete_paired_support": True,
            "clean_physically_disjoint": True,
        },
    }


def _fixed_summaries() -> dict[float, dict[str, object]]:
    primary_mse = {0.25: 0.90, 0.50: 0.70, 0.75: 0.80, 1.00: 1.00}
    return {
        beta: _summary(method=FIXED_METHODS[beta], primary_mse=primary_mse[beta])
        for beta in FIXED_BETA_CANDIDATES
    }


def _seed_summaries() -> dict[int, dict[str, object]]:
    primary_mse = {
        12011: 0.50,
        12037: 0.10,
        12041: 0.30,
        12049: 0.20,
        12071: 0.40,
    }
    return {
        seed: _summary(primary_mse=primary_mse[seed])
        for seed in OPTIMIZATION_SEEDS
    }


def _reason_codes(decision: dict[str, object]) -> set[str]:
    reasons = decision["reasons"]
    assert isinstance(reasons, list)
    return {str(reason["code"]) for reason in reasons}


def test_fixed_beta_is_safety_filtered_before_primary_selection() -> None:
    summaries = _fixed_summaries()
    summaries[0.25] = _summary(
        method=FIXED_METHODS[0.25],
        primary_mse=0.01,
        recovery=0.021,
        recovery_upper=0.021,
    )
    summaries[0.50]["primary"]["candidate_macro_mse"] = 0.70  # type: ignore[index]
    summaries[0.75]["primary"]["candidate_macro_mse"] = 0.70  # type: ignore[index]

    selected = select_fixed_beta(summaries)

    assert selected["decision"] == "Selected"
    assert selected["selected_beta"] == 0.50
    assert selected["selected_method"] == FIXED_METHODS[0.50]
    assert 0.25 not in selected["safety_qualified_betas"]


def test_inclusive_gate_boundaries_pass() -> None:
    summary = _summary(
        primary_improvement=0.10,
        primary_ci_lower=1.0e-12,
        task_ratio=1.05,
        task_ratio_upper=1.10,
        recovery=0.02,
        recovery_upper=0.02,
        indirect=0.02,
        indirect_upper=0.02,
        clean=0.019,
        clean_upper=0.019,
    )

    decision = evaluate_full_gate(
        summary,
        selected_fixed_method=FIXED_METHODS[0.50],
        selected_fixed_beta=0.50,
    )

    assert decision["decision"] == "Go"
    assert decision["checks"]["primary_relative_improvement"]["passed"] is True
    assert decision["checks"]["per_task_point_degradation"]["passed"] is True
    assert decision["checks"]["recovery_additive_ci_safety"]["passed"] is True
    assert decision["checks"]["indirect_additive_ci_safety"]["passed"] is True
    assert decision["checks"]["beats_no_lewm"]["passed"] is True
    assert decision["checks"]["no_lewm_paired_ci_lower_positive"]["passed"] is True


def test_action_blind_diagnostic_is_not_a_formal_gate_comparator() -> None:
    summary = _summary()
    summary["comparators"]["adapter_vs_action_blind"] = _effect(  # type: ignore[index]
        "adapter", "action_blind", improvement=-1.0, ci_lower=-1.0
    )

    decision = evaluate_full_gate(
        summary, selected_fixed_method=FIXED_METHODS[0.50]
    )

    assert NO_LEWM_METHOD == "no_lewm"
    assert decision["decision"] == "Go"
    assert "beats_action_blind" not in decision["checks"]
    assert decision["comparators"]["no_lewm"]["control_method"] == NO_LEWM_METHOD


def test_strict_equal_boundaries_fail_with_specific_reasons() -> None:
    summary = _summary(
        primary_ci_lower=0.0,
        clean=0.02,
        clean_upper=0.02,
    )
    fixed = summary["comparators"]["adapter_vs_fixed"][  # type: ignore[index]
        FIXED_METHODS[0.50]
    ]
    fixed["relative_improvement"] = 0.0
    fixed["relative_improvement_95_ci"] = [0.0, 0.05]

    decision = evaluate_full_gate(
        summary, selected_fixed_method=FIXED_METHODS[0.50]
    )

    assert decision["decision"] == "No-Go"
    assert {
        "primary_paired_ci_lower_positive",
        "beats_selected_fixed_beta",
        "selected_fixed_beta_paired_ci_lower_positive",
        "clean_point_degradation",
        "clean_ci_degradation",
    }.issubset(_reason_codes(decision))


def test_missing_decision_fields_are_rejected() -> None:
    missing_indirect = _summary()
    del missing_indirect["indirect_safety"]
    with pytest.raises(ValueError, match="indirect_safety"):
        evaluate_full_gate(
            missing_indirect, selected_fixed_method=FIXED_METHODS[0.50]
        )

    missing_ci = _summary()
    del missing_ci["primary"]["relative_improvement_95_ci"]  # type: ignore[index]
    with pytest.raises(ValueError, match="relative_improvement_95_ci"):
        evaluate_full_gate(missing_ci, selected_fixed_method=FIXED_METHODS[0.50])

    missing_fixed = _fixed_summaries()
    del missing_fixed[1.00]
    with pytest.raises(ValueError, match="exactly beta"):
        select_fixed_beta(missing_fixed)


def test_seed_selection_is_deterministic_and_uses_upper_median_endpoint() -> None:
    summaries = _seed_summaries()
    summaries[12011] = _summary(primary_mse=0.50, clean=0.02, clean_upper=0.02)
    original = deepcopy(summaries)

    forward = select_median_seed(summaries)
    reverse = select_median_seed(dict(reversed(list(summaries.items()))))

    assert forward == reverse
    assert forward["safety_qualified_seeds"] == [12037, 12049, 12041, 12071]
    assert forward["selected_seed"] == 12041
    assert forward["selected_primary_mse"] == 0.30
    assert forward["median_policy"] == "upper_median_lower_is_better"
    assert summaries == original


def test_four_of_five_positive_is_inclusive_but_zero_is_not_positive() -> None:
    summaries = _seed_summaries()
    summaries[12011]["primary"]["relative_improvement"] = 0.0  # type: ignore[index]

    boundary = select_median_seed(summaries)

    assert boundary["positive_primary_seed_count"] == 4
    assert boundary["passed"] is True

    summaries[12037]["primary"]["relative_improvement"] = 0.0  # type: ignore[index]
    failed = select_median_seed(summaries)
    assert failed["decision"] == "No-Go"
    assert "five_seed_primary_direction" in _reason_codes(failed)


def test_full_no_go_reports_all_failed_conjuncts() -> None:
    summary = _summary(
        primary_improvement=0.09,
        primary_ci_lower=-0.01,
        task_ratio=1.051,
        recovery=0.021,
        recovery_upper=0.021,
        indirect=0.021,
        indirect_upper=0.021,
        clean=0.02,
        clean_upper=0.02,
    )
    fixed = summary["comparators"]["adapter_vs_fixed"][  # type: ignore[index]
        FIXED_METHODS[0.50]
    ]
    fixed["relative_improvement"] = -0.01
    fixed["relative_improvement_95_ci"] = [-0.03, 0.01]
    no_lewm = summary["comparators"]["adapter_vs_no_lewm"]  # type: ignore[index]
    no_lewm["relative_improvement"] = 0.0
    no_lewm["relative_improvement_95_ci"] = [0.0, 0.02]

    decision = evaluate_full_gate(
        summary, selected_fixed_method=FIXED_METHODS[0.50]
    )

    assert decision["decision"] == "No-Go"
    assert {
        "primary_relative_improvement",
        "primary_paired_ci_lower_positive",
        "beats_selected_fixed_beta",
        "selected_fixed_beta_paired_ci_lower_positive",
        "beats_no_lewm",
        "no_lewm_paired_ci_lower_positive",
        "per_task_point_degradation",
        "recovery_additive_point_safety",
        "recovery_additive_ci_safety",
        "indirect_additive_point_safety",
        "indirect_additive_ci_safety",
        "clean_point_degradation",
        "clean_ci_degradation",
    }.issubset(_reason_codes(decision))


def test_validation_only_orchestrator_returns_structured_go_decision() -> None:
    decision = select_validation_only_posterior_gate(
        _fixed_summaries(), _seed_summaries()
    )

    assert decision["decision"] == "Go"
    assert decision["fixed_beta_selection"]["selected_beta"] == 0.50
    assert decision["seed_selection"]["selected_seed"] == 12041
    assert decision["selection_scope"] == {
        "source_split": "validation",
        "validation_only": True,
        "sealed_data_used": False,
    }
    assert decision["reasons"] == []


def test_sealed_or_non_validation_summaries_are_rejected() -> None:
    with pytest.raises(ValueError, match="validation summaries only"):
        select_median_seed(_seed_summaries(), source_split="sealed_test")

    summary = _summary()
    summary["split"] = "sealed_test"
    with pytest.raises(ValueError, match="sealed summaries are forbidden"):
        evaluate_full_gate(summary, selected_fixed_method=FIXED_METHODS[0.50])
