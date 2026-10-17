"""Validation-only selection for the bounded LeWM posterior Gate.

The functions in this module consume only aggregate statistic summaries.  They
do not open datasets, checkpoints, or sealed reports.  Thresholds and candidate
sets mirror ``anchor_persistent_posterior_adapter_gate_v1.yaml`` so selection is
deterministic and cannot become an implicit sweep.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any


PROTOCOL_NAME = "fcpg_lewm_posterior_gate_validation_selection_v1"
VALIDATION_SPLIT = "validation"

FIXED_BETA_CANDIDATES = (0.25, 0.50, 0.75, 1.00)
OPTIMIZATION_SEEDS = (12011, 12037, 12041, 12049, 12071)
TASKS = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)

MEASUREMENT_METHOD = "measurement_writeback"
NO_LEWM_METHOD = "no_lewm"
REQUIRED_POSITIVE_SEEDS = 4

PRIMARY_RELATIVE_IMPROVEMENT_MIN = 0.10
PER_TASK_POINT_DEGRADATION_MAX = 0.05
PER_TASK_RATIO_CI_UPPER_MAX = 1.10
ADDITIVE_SAFETY_MAX = 0.02
CLEAN_DEGRADATION_MAX = 0.02


def _mapping(value: Any, *, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _required(mapping: Mapping[Any, Any], key: str, *, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label} is missing required field {key!r}")
    return mapping[key]


def _section(
    summary: Mapping[Any, Any], key: str, *, label: str
) -> Mapping[Any, Any]:
    return _mapping(
        _required(summary, key, label=label),
        label=f"{label}.{key}",
    )


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _confidence_interval(value: Any, *, label: str) -> tuple[float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(f"{label} must contain exactly two numeric endpoints")
    lower = _finite(value[0], label=f"{label}[0]")
    upper = _finite(value[1], label=f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} lower endpoint must not exceed upper endpoint")
    return lower, upper


def _validate_source_split(source_split: Any) -> str:
    split = _string(source_split, label="source_split")
    if split != VALIDATION_SPLIT:
        raise ValueError("posterior Gate selection accepts validation summaries only")
    return split


def _validate_summary_split(summary: Mapping[Any, Any], *, label: str) -> None:
    for field in ("split", "source_split", "development_split"):
        if field not in summary:
            continue
        split = _string(summary[field], label=f"{label}.{field}")
        if split != VALIDATION_SPLIT:
            raise ValueError(
                f"{label}.{field} must be {VALIDATION_SPLIT!r}; sealed summaries "
                "are forbidden during selection"
            )


def _primary_snapshot(summary: Mapping[Any, Any], *, label: str) -> dict[str, Any]:
    primary = _section(summary, "primary", label=label)
    candidate_method = _string(
        _required(primary, "candidate_method", label=f"{label}.primary"),
        label=f"{label}.primary.candidate_method",
    )
    control_method = _string(
        _required(primary, "control_method", label=f"{label}.primary"),
        label=f"{label}.primary.control_method",
    )
    if control_method != MEASUREMENT_METHOD:
        raise ValueError(
            f"{label}.primary.control_method must be {MEASUREMENT_METHOD!r}"
        )
    primary_mse = _finite(
        _required(primary, "candidate_macro_mse", label=f"{label}.primary"),
        label=f"{label}.primary.candidate_macro_mse",
        nonnegative=True,
    )
    improvement = _finite(
        _required(primary, "relative_improvement", label=f"{label}.primary"),
        label=f"{label}.primary.relative_improvement",
    )
    ci = _confidence_interval(
        _required(primary, "relative_improvement_95_ci", label=f"{label}.primary"),
        label=f"{label}.primary.relative_improvement_95_ci",
    )
    return {
        "candidate_method": candidate_method,
        "control_method": control_method,
        "primary_mse": primary_mse,
        "relative_improvement": improvement,
        "relative_improvement_95_ci": list(ci),
    }


def _effect_snapshot(effect: Any, *, label: str) -> dict[str, Any]:
    row = _mapping(effect, label=label)
    candidate_method = _string(
        _required(row, "candidate_method", label=label),
        label=f"{label}.candidate_method",
    )
    control_method = _string(
        _required(row, "control_method", label=label),
        label=f"{label}.control_method",
    )
    improvement = _finite(
        _required(row, "relative_improvement", label=label),
        label=f"{label}.relative_improvement",
    )
    ci = _confidence_interval(
        _required(row, "relative_improvement_95_ci", label=label),
        label=f"{label}.relative_improvement_95_ci",
    )
    return {
        "candidate_method": candidate_method,
        "control_method": control_method,
        "relative_improvement": improvement,
        "relative_improvement_95_ci": list(ci),
    }


def _check(
    *,
    passed: bool,
    observed: Any,
    required: str,
    message: str,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "message": message,
    }


def _failed_reasons(checks: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "code": code,
            "message": check["message"],
            "observed": check["observed"],
            "required": check["required"],
        }
        for code, check in checks.items()
        if not check["passed"]
    ]


def _normalized_safety_values(
    summary: Mapping[Any, Any], section_name: str, *, label: str
) -> tuple[float, float]:
    section = _section(summary, section_name, label=label)
    point = _finite(
        _required(section, "normalized_excess_mse", label=f"{label}.{section_name}"),
        label=f"{label}.{section_name}.normalized_excess_mse",
    )
    upper = _finite(
        _required(
            section,
            "normalized_excess_mse_one_sided_95_upper",
            label=f"{label}.{section_name}",
        ),
        label=(
            f"{label}.{section_name}."
            "normalized_excess_mse_one_sided_95_upper"
        ),
    )
    return point, upper


def _indirect_safety_values(
    summary: Mapping[Any, Any], *, label: str
) -> tuple[float, float]:
    names = (
        "indirect_safety",
        "indirectly_affected_anchor_safety",
    )
    present = [name for name in names if name in summary]
    if not present:
        _required(summary, "indirect_safety", label=label)
    if len(present) != 1:
        raise ValueError(
            f"{label} must provide exactly one indirect-anchor safety section"
        )
    return _normalized_safety_values(summary, present[0], label=label)


def _safety_evaluation(summary: Mapping[Any, Any], *, label: str) -> dict[str, Any]:
    coverage = _section(summary, "coverage", label=label)
    complete_paired_support = _boolean(
        _required(coverage, "complete_paired_support", label=f"{label}.coverage"),
        label=f"{label}.coverage.complete_paired_support",
    )
    complete_panels = _boolean(
        _required(
            coverage,
            "complete_estimand_event_method_panels",
            label=f"{label}.coverage",
        ),
        label=f"{label}.coverage.complete_estimand_event_method_panels",
    )
    complete_primary = complete_paired_support and complete_panels
    complete_clean = _boolean(
        _required(
            coverage, "clean_complete_paired_support", label=f"{label}.coverage"
        ),
        label=f"{label}.coverage.clean_complete_paired_support",
    )
    clean_disjoint = _boolean(
        _required(coverage, "clean_physically_disjoint", label=f"{label}.coverage"),
        label=f"{label}.coverage.clean_physically_disjoint",
    )

    task_rows = _section(summary, "task_safety_ratios", label=label)
    task_names = set(task_rows)
    if task_names != set(TASKS):
        missing = sorted(set(TASKS) - task_names)
        unexpected = sorted(task_names - set(TASKS), key=str)
        raise ValueError(
            f"{label}.task_safety_ratios must contain exactly the four "
            f"preregistered tasks; missing={missing} unexpected={unexpected}"
        )
    point_ratios: dict[str, float] = {}
    ratio_uppers: dict[str, float] = {}
    for task in TASKS:
        row = _mapping(task_rows[task], label=f"{label}.task_safety_ratios.{task}")
        point_ratios[task] = _finite(
            _required(
                row,
                "adapter_to_measurement_ratio",
                label=f"{label}.task_safety_ratios.{task}",
            ),
            label=(
                f"{label}.task_safety_ratios.{task}."
                "adapter_to_measurement_ratio"
            ),
            nonnegative=True,
        )
        ratio_uppers[task] = _finite(
            _required(
                row,
                "adapter_to_measurement_ratio_one_sided_95_upper",
                label=f"{label}.task_safety_ratios.{task}",
            ),
            label=(
                f"{label}.task_safety_ratios.{task}."
                "adapter_to_measurement_ratio_one_sided_95_upper"
            ),
            nonnegative=True,
        )

    recovery_point, recovery_upper = _normalized_safety_values(
        summary, "recovery_safety", label=label
    )
    indirect_point, indirect_upper = _indirect_safety_values(summary, label=label)
    clean_point, clean_upper = _normalized_safety_values(
        summary, "clean_safety", label=label
    )
    clean = _section(summary, "clean_safety", label=label)
    clean_cohort = _string(
        _required(clean, "cohort", label=f"{label}.clean_safety"),
        label=f"{label}.clean_safety.cohort",
    )

    checks = {
        "complete_validation_support": _check(
            passed=complete_primary,
            observed=complete_primary,
            required="true",
            message="validation statistic panels must have complete paired support",
        ),
        "independent_clean_cohort": _check(
            passed=(
                complete_clean
                and clean_disjoint
                and clean_cohort == "explicit_independent_clean_records"
            ),
            observed={
                "complete_paired_support": complete_clean,
                "physically_disjoint": clean_disjoint,
                "cohort": clean_cohort,
            },
            required=(
                "complete and physically disjoint "
                "explicit_independent_clean_records"
            ),
            message="clean safety must come from the independent clean cohort",
        ),
        "per_task_point_degradation": _check(
            passed=all(
                ratio <= 1.0 + PER_TASK_POINT_DEGRADATION_MAX
                for ratio in point_ratios.values()
            ),
            observed=point_ratios,
            required=(
                "every candidate/measurement ratio <= "
                f"{1.0 + PER_TASK_POINT_DEGRADATION_MAX:.2f}"
            ),
            message="at least one task degrades by more than five percent",
        ),
        "per_task_ratio_ci_upper": _check(
            passed=all(
                upper <= PER_TASK_RATIO_CI_UPPER_MAX
                for upper in ratio_uppers.values()
            ),
            observed=ratio_uppers,
            required=(
                "every one-sided ratio upper <= "
                f"{PER_TASK_RATIO_CI_UPPER_MAX:.2f}"
            ),
            message="at least one task exceeds the preregistered ratio upper bound",
        ),
        "recovery_additive_point_safety": _check(
            passed=recovery_point <= ADDITIVE_SAFETY_MAX,
            observed=recovery_point,
            required=f"normalized excess MSE <= {ADDITIVE_SAFETY_MAX:.2f}",
            message="recovery additive degradation exceeds two percent",
        ),
        "recovery_additive_ci_safety": _check(
            passed=recovery_upper <= ADDITIVE_SAFETY_MAX,
            observed=recovery_upper,
            required=f"one-sided normalized excess upper <= {ADDITIVE_SAFETY_MAX:.2f}",
            message="recovery additive safety confidence bound exceeds two percent",
        ),
        "indirect_additive_point_safety": _check(
            passed=indirect_point <= ADDITIVE_SAFETY_MAX,
            observed=indirect_point,
            required=f"normalized excess MSE <= {ADDITIVE_SAFETY_MAX:.2f}",
            message="indirect-anchor additive degradation exceeds two percent",
        ),
        "indirect_additive_ci_safety": _check(
            passed=indirect_upper <= ADDITIVE_SAFETY_MAX,
            observed=indirect_upper,
            required=f"one-sided normalized excess upper <= {ADDITIVE_SAFETY_MAX:.2f}",
            message=(
                "indirect-anchor additive safety confidence bound exceeds two percent"
            ),
        ),
        "clean_point_degradation": _check(
            passed=clean_point < CLEAN_DEGRADATION_MAX,
            observed=clean_point,
            required=(
                "additive clean excess / perturbed primary measurement MSE < "
                f"{CLEAN_DEGRADATION_MAX:.2f}"
            ),
            message=(
                "independent clean-cohort additive excess is not below two "
                "percent of the perturbed primary measurement scale"
            ),
        ),
        "clean_ci_degradation": _check(
            passed=clean_upper < CLEAN_DEGRADATION_MAX,
            observed=clean_upper,
            required=(
                "one-sided upper for additive clean excess / perturbed primary "
                f"measurement MSE < {CLEAN_DEGRADATION_MAX:.2f}"
            ),
            message=(
                "independent clean-cohort additive-excess confidence bound is not "
                "below two percent of the perturbed primary measurement scale"
            ),
        ),
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "reasons": _failed_reasons(checks),
    }


def _parse_beta(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must identify a numeric fixed beta")
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError as error:
            raise ValueError(f"{label} must identify a numeric fixed beta") from error
    elif isinstance(value, Real):
        numeric = float(value)
    else:
        raise TypeError(f"{label} must identify a numeric fixed beta")
    if not math.isfinite(numeric) or numeric not in FIXED_BETA_CANDIDATES:
        raise ValueError(
            f"{label} must be one of {list(FIXED_BETA_CANDIDATES)!r}"
        )
    return numeric


def _fixed_summaries(
    value: Any,
) -> dict[float, Mapping[Any, Any]]:
    raw = _mapping(value, label="fixed_beta_summaries")
    result: dict[float, Mapping[Any, Any]] = {}
    for raw_beta, raw_summary in raw.items():
        beta = _parse_beta(raw_beta, label=f"fixed beta key {raw_beta!r}")
        if beta in result:
            raise ValueError(f"duplicate normalized fixed beta {beta}")
        result[beta] = _mapping(
            raw_summary, label=f"fixed_beta_summaries[{beta}]"
        )
    if set(result) != set(FIXED_BETA_CANDIDATES):
        missing = sorted(set(FIXED_BETA_CANDIDATES) - set(result))
        raise ValueError(
            "fixed_beta_summaries must contain exactly beta "
            f"{list(FIXED_BETA_CANDIDATES)!r}; missing={missing}"
        )
    return result


def _parse_seed(value: Any, *, label: str) -> int:
    if isinstance(value, str):
        if not value.isdecimal():
            raise ValueError(f"{label} must identify an integer seed")
        return int(value)
    return _integer(value, label=label)


def _seed_summaries(value: Any) -> dict[int, Mapping[Any, Any]]:
    raw = _mapping(value, label="seed_summaries")
    result: dict[int, Mapping[Any, Any]] = {}
    for raw_seed, raw_summary in raw.items():
        seed = _parse_seed(raw_seed, label=f"seed key {raw_seed!r}")
        if seed in result:
            raise ValueError(f"duplicate normalized seed {seed}")
        result[seed] = _mapping(raw_summary, label=f"seed_summaries[{seed}]")
    if set(result) != set(OPTIMIZATION_SEEDS):
        missing = sorted(set(OPTIMIZATION_SEEDS) - set(result))
        unexpected = sorted(set(result) - set(OPTIMIZATION_SEEDS))
        raise ValueError(
            "seed_summaries must contain exactly the five preregistered seeds; "
            f"missing={missing} unexpected={unexpected}"
        )
    return result


def select_fixed_beta(
    fixed_beta_summaries: Mapping[Any, Any],
    *,
    source_split: str = VALIDATION_SPLIT,
) -> dict[str, Any]:
    """Safety-filter fixed betas, then select the lowest primary-MSE candidate."""

    split = _validate_source_split(source_split)
    summaries = _fixed_summaries(fixed_beta_summaries)
    candidates: list[dict[str, Any]] = []
    methods: set[str] = set()
    for beta in FIXED_BETA_CANDIDATES:
        label = f"fixed_beta_summaries[{beta}]"
        summary = summaries[beta]
        _validate_summary_split(summary, label=label)
        primary = _primary_snapshot(summary, label=label)
        method = primary["candidate_method"]
        if method in methods:
            raise ValueError("each fixed beta summary must name a distinct method")
        methods.add(method)
        safety = _safety_evaluation(summary, label=label)
        candidates.append(
            {
                "beta": beta,
                "method": method,
                "primary_mse": primary["primary_mse"],
                "safety_qualified": safety["passed"],
                "safety": safety,
            }
        )

    qualified = [row for row in candidates if row["safety_qualified"]]
    selected = min(
        qualified,
        key=lambda row: (row["primary_mse"], row["beta"]),
        default=None,
    )
    checks = {
        "fixed_beta_safety_candidate_available": _check(
            passed=selected is not None,
            observed=[row["beta"] for row in qualified],
            required="at least one preregistered fixed beta passes every safety check",
            message="no fixed beta is safety-qualified",
        )
    }
    passed = selected is not None
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "selection_split": split,
        "sealed_data_used": False,
        "selection_rule": "safety_filter_then_lowest_primary_mse_then_lower_beta",
        "candidates": candidates,
        "safety_qualified_betas": [row["beta"] for row in qualified],
        "selected_beta": None if selected is None else selected["beta"],
        "selected_method": None if selected is None else selected["method"],
        "selected_primary_mse": (
            None if selected is None else selected["primary_mse"]
        ),
        "checks": checks,
        "passed": passed,
        "decision": "Selected" if passed else "No-Go",
        "reasons": _failed_reasons(checks),
    }


def select_median_seed(
    seed_summaries: Mapping[Any, Any],
    *,
    source_split: str = VALIDATION_SPLIT,
) -> dict[str, Any]:
    """Select the conservative median endpoint among safety-qualified seeds.

    Endpoints are ordered by primary candidate MSE (lower is better), then seed.
    If an even number qualify, the upper median is used so the rule never chooses
    the more optimistic of the two central endpoints.
    """

    split = _validate_source_split(source_split)
    summaries = _seed_summaries(seed_summaries)
    endpoints: list[dict[str, Any]] = []
    candidate_methods: set[str] = set()
    for seed in OPTIMIZATION_SEEDS:
        label = f"seed_summaries[{seed}]"
        summary = summaries[seed]
        _validate_summary_split(summary, label=label)
        primary = _primary_snapshot(summary, label=label)
        candidate_methods.add(primary["candidate_method"])
        safety = _safety_evaluation(summary, label=label)
        endpoints.append(
            {
                "seed": seed,
                "primary_mse": primary["primary_mse"],
                "primary_relative_improvement": primary["relative_improvement"],
                "primary_positive": primary["relative_improvement"] > 0.0,
                "safety_qualified": safety["passed"],
                "safety": safety,
            }
        )
    if len(candidate_methods) != 1:
        raise ValueError("all seed summaries must evaluate the same full method")

    positive_count = sum(row["primary_positive"] for row in endpoints)
    qualified = sorted(
        (row for row in endpoints if row["safety_qualified"]),
        key=lambda row: (row["primary_mse"], row["seed"]),
    )
    selected = qualified[len(qualified) // 2] if qualified else None
    checks = {
        "five_seed_primary_direction": _check(
            passed=positive_count >= REQUIRED_POSITIVE_SEEDS,
            observed={"positive": positive_count, "total": len(endpoints)},
            required=(
                f">= {REQUIRED_POSITIVE_SEEDS}/5 seeds with "
                "primary improvement > 0"
            ),
            message="fewer than four of five seeds have positive primary effect",
        ),
        "safety_qualified_seed_available": _check(
            passed=selected is not None,
            observed=[row["seed"] for row in qualified],
            required="at least one seed passes every safety check",
            message="no seed endpoint is safety-qualified",
        ),
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "selection_split": split,
        "sealed_data_used": False,
        "selection_rule": (
            "safety_filter_then_primary_mse_upper_median_then_seed_tiebreak"
        ),
        "median_policy": "upper_median_lower_is_better",
        "endpoints": endpoints,
        "positive_primary_seed_count": positive_count,
        "required_positive_primary_seed_count": REQUIRED_POSITIVE_SEEDS,
        "safety_qualified_seeds": [row["seed"] for row in qualified],
        "selected_seed": None if selected is None else selected["seed"],
        "selected_primary_mse": (
            None if selected is None else selected["primary_mse"]
        ),
        "checks": checks,
        "passed": passed,
        "decision": "Selected" if passed else "No-Go",
        "reasons": _failed_reasons(checks),
    }


def evaluate_full_gate(
    summary: Mapping[str, Any],
    *,
    selected_fixed_method: str,
    selected_fixed_beta: float | None = None,
    source_split: str = VALIDATION_SPLIT,
) -> dict[str, Any]:
    """Apply every preregistered full-model efficacy and safety condition."""

    split = _validate_source_split(source_split)
    report = _mapping(summary, label="summary")
    _validate_summary_split(report, label="summary")
    fixed_method = _string(selected_fixed_method, label="selected_fixed_method")
    fixed_beta = (
        None
        if selected_fixed_beta is None
        else _parse_beta(selected_fixed_beta, label="selected_fixed_beta")
    )
    primary = _primary_snapshot(report, label="summary")
    safety = _safety_evaluation(report, label="summary")

    comparators = _section(report, "comparators", label="summary")
    fixed_effects = _mapping(
        _required(comparators, "adapter_vs_fixed", label="summary.comparators"),
        label="summary.comparators.adapter_vs_fixed",
    )
    if fixed_method not in fixed_effects:
        raise ValueError(
            "summary.comparators.adapter_vs_fixed is missing selected method "
            f"{fixed_method!r}"
        )
    fixed_effect = _effect_snapshot(
        fixed_effects[fixed_method],
        label=f"summary.comparators.adapter_vs_fixed.{fixed_method}",
    )
    no_lewm_effect = _effect_snapshot(
        _required(
            comparators,
            "adapter_vs_no_lewm",
            label="summary.comparators",
        ),
        label="summary.comparators.adapter_vs_no_lewm",
    )
    for effect, control, label in (
        (fixed_effect, fixed_method, "selected fixed beta"),
        (no_lewm_effect, NO_LEWM_METHOD, "no-LeWM"),
    ):
        if effect["candidate_method"] != primary["candidate_method"]:
            raise ValueError(f"{label} comparison candidate disagrees with primary")
        if effect["control_method"] != control:
            raise ValueError(f"{label} comparison names the wrong control method")

    checks: dict[str, dict[str, Any]] = {
        "primary_relative_improvement": _check(
            passed=(
                primary["relative_improvement"]
                >= PRIMARY_RELATIVE_IMPROVEMENT_MIN
            ),
            observed=primary["relative_improvement"],
            required=f">= {PRIMARY_RELATIVE_IMPROVEMENT_MIN:.2f} versus measurement",
            message="full method does not improve primary MSE by at least ten percent",
        ),
        "primary_paired_ci_lower_positive": _check(
            passed=primary["relative_improvement_95_ci"][0] > 0.0,
            observed=primary["relative_improvement_95_ci"][0],
            required="> 0",
            message="primary paired confidence interval does not exclude zero",
        ),
        "beats_selected_fixed_beta": _check(
            passed=fixed_effect["relative_improvement"] > 0.0,
            observed=fixed_effect["relative_improvement"],
            required="> 0",
            message="full method does not beat the selected fixed beta",
        ),
        "selected_fixed_beta_paired_ci_lower_positive": _check(
            passed=fixed_effect["relative_improvement_95_ci"][0] > 0.0,
            observed=fixed_effect["relative_improvement_95_ci"][0],
            required="> 0 at sequential one-sided alpha 0.025",
            message="full versus selected fixed beta does not exclude zero",
        ),
        "beats_no_lewm": _check(
            passed=no_lewm_effect["relative_improvement"] > 0.0,
            observed=no_lewm_effect["relative_improvement"],
            required="> 0",
            message="full method does not beat the no-LeWM hold-last-prior control",
        ),
        "no_lewm_paired_ci_lower_positive": _check(
            passed=no_lewm_effect["relative_improvement_95_ci"][0] > 0.0,
            observed=no_lewm_effect["relative_improvement_95_ci"][0],
            required="> 0 at sequential one-sided alpha 0.025",
            message="full versus no-LeWM does not exclude zero",
        ),
    }
    checks.update(safety["checks"])
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "selection_split": split,
        "sealed_data_used": False,
        "evaluated": True,
        "selected_fixed_beta": fixed_beta,
        "selected_fixed_method": fixed_method,
        "full_method": primary["candidate_method"],
        "primary_mse": primary["primary_mse"],
        "primary": primary,
        "comparators": {
            "selected_fixed_beta": fixed_effect,
            "no_lewm": no_lewm_effect,
        },
        "thresholds": {
            "primary_relative_improvement_min": PRIMARY_RELATIVE_IMPROVEMENT_MIN,
            "primary_ci_lower_strictly_positive": True,
            "comparator_effect_strictly_positive": True,
            "comparator_ci_lower_strictly_positive": True,
            "per_task_point_degradation_max": PER_TASK_POINT_DEGRADATION_MAX,
            "per_task_ratio_ci_upper_max": PER_TASK_RATIO_CI_UPPER_MAX,
            "additive_safety_max": ADDITIVE_SAFETY_MAX,
            "clean_degradation_strict_max": CLEAN_DEGRADATION_MAX,
        },
        "checks": checks,
        "passed": passed,
        "decision": "Go" if passed else "No-Go",
        "reasons": _failed_reasons(checks),
    }


def _staged_reasons(
    stage: str, reasons: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [{"stage": stage, **dict(reason)} for reason in reasons]


def select_validation_only_posterior_gate(
    fixed_beta_summaries: Mapping[Any, Any],
    seed_summaries: Mapping[Any, Any],
    *,
    source_split: str = VALIDATION_SPLIT,
) -> dict[str, Any]:
    """Select fixed beta and seed endpoint, then evaluate the conjunctive Gate."""

    split = _validate_source_split(source_split)
    normalized_seeds = _seed_summaries(seed_summaries)
    fixed_selection = select_fixed_beta(
        fixed_beta_summaries, source_split=split
    )
    seed_selection = select_median_seed(normalized_seeds, source_split=split)

    can_evaluate = (
        fixed_selection["selected_method"] is not None
        and seed_selection["selected_seed"] is not None
    )
    if can_evaluate:
        selected_seed = seed_selection["selected_seed"]
        full_gate = evaluate_full_gate(
            normalized_seeds[selected_seed],
            selected_fixed_method=fixed_selection["selected_method"],
            selected_fixed_beta=fixed_selection["selected_beta"],
            source_split=split,
        )
    else:
        prerequisite = _check(
            passed=False,
            observed={
                "selected_fixed_method": fixed_selection["selected_method"],
                "selected_seed": seed_selection["selected_seed"],
            },
            required="a safety-qualified fixed beta and seed endpoint",
            message="full Gate cannot be evaluated without both selections",
        )
        full_gate = {
            "schema_version": 1,
            "protocol": PROTOCOL_NAME,
            "selection_split": split,
            "sealed_data_used": False,
            "evaluated": False,
            "checks": {"selection_prerequisite": prerequisite},
            "passed": False,
            "decision": "No-Go",
            "reasons": _failed_reasons({"selection_prerequisite": prerequisite}),
        }

    checks = {
        "fixed_beta_selection": _check(
            passed=fixed_selection["passed"],
            observed=fixed_selection["selected_beta"],
            required="one safety-qualified fixed beta selected by primary MSE",
            message="fixed-beta validation selection failed",
        ),
        "seed_selection": _check(
            passed=seed_selection["passed"],
            observed={
                "selected_seed": seed_selection["selected_seed"],
                "positive_primary_seed_count": seed_selection[
                    "positive_primary_seed_count"
                ],
            },
            required=(
                "at least 4/5 positive seeds and a safety-qualified "
                "median endpoint"
            ),
            message="multi-seed validation selection failed",
        ),
        "full_conjunctive_gate": _check(
            passed=full_gate["passed"],
            observed=full_gate["decision"],
            required="Go",
            message=(
                "selected full endpoint fails at least one conjunctive "
                "Gate condition"
            ),
        ),
    }
    passed = all(check["passed"] for check in checks.values())
    reasons = _failed_reasons(checks)
    reasons.extend(_staged_reasons("fixed_beta_selection", fixed_selection["reasons"]))
    reasons.extend(_staged_reasons("seed_selection", seed_selection["reasons"]))
    reasons.extend(_staged_reasons("full_gate", full_gate["reasons"]))
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "selection_scope": {
            "source_split": split,
            "validation_only": True,
            "sealed_data_used": False,
        },
        "fixed_beta_selection": fixed_selection,
        "seed_selection": seed_selection,
        "full_gate": full_gate,
        "checks": checks,
        "passed": passed,
        "decision": "Go" if passed else "No-Go",
        "reasons": reasons,
    }


select_validation_fixed_beta = select_fixed_beta
select_validation_seed = select_median_seed
evaluate_validation_full_gate = evaluate_full_gate


__all__ = [
    "NO_LEWM_METHOD",
    "ADDITIVE_SAFETY_MAX",
    "CLEAN_DEGRADATION_MAX",
    "FIXED_BETA_CANDIDATES",
    "MEASUREMENT_METHOD",
    "OPTIMIZATION_SEEDS",
    "PER_TASK_POINT_DEGRADATION_MAX",
    "PER_TASK_RATIO_CI_UPPER_MAX",
    "PRIMARY_RELATIVE_IMPROVEMENT_MIN",
    "PROTOCOL_NAME",
    "REQUIRED_POSITIVE_SEEDS",
    "TASKS",
    "VALIDATION_SPLIT",
    "evaluate_full_gate",
    "evaluate_validation_full_gate",
    "select_fixed_beta",
    "select_median_seed",
    "select_validation_fixed_beta",
    "select_validation_only_posterior_gate",
    "select_validation_seed",
]
