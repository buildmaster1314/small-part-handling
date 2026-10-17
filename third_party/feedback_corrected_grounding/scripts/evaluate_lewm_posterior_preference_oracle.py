#!/usr/bin/env python3
"""Run the diagnostic teacher-forced and recursive posterior oracle panel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM  # noqa: E402
from fcpg.lewm_posterior_gate_data import FRESH_FRAME_SKIP  # noqa: E402
from fcpg.lewm_posterior_gate_orchestration import (  # noqa: E402
    load_posterior_gate_inputs,
)
from fcpg.lewm_posterior_gate_runtime import (  # noqa: E402
    MEASUREMENT_METHOD,
    infer_direct_anchors,
    posterior_gate_examples_from_cache,
)
from fcpg.lewm_posterior_gate_statistics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    summarize_posterior_gate_statistics,
)
from fcpg.lewm_posterior_preference_oracle import (  # noqa: E402
    CONVEX_ORACLE_METHOD,
    FSM_ORACLE_METHOD,
    HARD_ORACLE_METHOD,
    PREFERENCE_ORACLE_METHODS,
    PRIOR_METHOD,
    SCHEDULE_ORACLE_METHOD,
    SEQUENCE_ORACLE_METHOD,
    evaluate_posterior_preference_oracle_cache,
    evaluate_teacher_forced_preference_oracle_cache,
)
from fcpg.lewm_runtime import load_lewm_grounded_checkpoint  # noqa: E402


RECURSIVE_CANDIDATES = (
    PRIOR_METHOD,
    HARD_ORACLE_METHOD,
    CONVEX_ORACLE_METHOD,
    SCHEDULE_ORACLE_METHOD,
    SEQUENCE_ORACLE_METHOD,
    FSM_ORACLE_METHOD,
)
TEACHER_FORCED_CANDIDATES = (
    PRIOR_METHOD,
    HARD_ORACLE_METHOD,
    CONVEX_ORACLE_METHOD,
    SCHEDULE_ORACLE_METHOD,
)
SAFETY_LIMIT = 0.02
PRIMARY_MINIMUM = 0.10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/lewm/train/anchor_persistent_posterior_adapter_gate_v1.yaml"
        ),
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--validation-clean-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def _compact_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "primary": summary["primary"],
        "recovery_safety": summary["recovery_safety"],
        "clean_safety": summary["clean_safety"],
        "indirect_safety": summary["indirect_safety"],
        "task_safety_ratios": summary["task_safety_ratios"],
        "coverage": {
            key: summary["coverage"][key]
            for key in (
                "task_count",
                "physical_episode_count",
                "clean_physical_episode_count",
                "episodes_per_task",
            )
        },
    }


def _candidate_summaries(
    records: Sequence[Mapping[str, Any]],
    clean_records: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in candidates:
        controls = [
            method
            for method in candidates
            if method not in {candidate, MEASUREMENT_METHOD}
        ]
        if not controls:
            raise ValueError("oracle panel requires at least two candidate methods")
        no_lewm = PRIOR_METHOD if candidate != PRIOR_METHOD else controls[0]
        fixed = tuple(method for method in controls if method != no_lewm)
        if not fixed:
            raise ValueError("oracle panel requires a separate fixed comparator")
        summary = summarize_posterior_gate_statistics(
            records,
            clean_records,
            adapter_method=candidate,
            measurement_method=MEASUREMENT_METHOD,
            fixed_methods=fixed,
            no_lewm_method=no_lewm,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        result[candidate] = _compact_summary(summary)
    return result


def _beta_by_event(runtime: Any, cache: Any) -> dict[str, dict[str, float]]:
    examples = posterior_gate_examples_from_cache(cache)
    direct = infer_direct_anchors(
        examples.metadata, examples.evaluation.dropout_active
    )
    values: dict[str, dict[str, list[float]]] = {}
    for method, trajectory in runtime.trajectories.items():
        values[method] = {}
        for index, row in enumerate(runtime.metadata):
            anchor = direct[(row.task, row.episode_id)]
            values[method].setdefault(row.event, []).append(
                float(trajectory.beta[index, int(anchor), 0])
            )
    return {
        method: {
            event: math.fsum(event_values) / len(event_values)
            for event, event_values in events.items()
        }
        for method, events in values.items()
    }


def _global_event_gain(
    runtime: Any, cache: Any
) -> dict[str, dict[str, float | None]]:
    """Reproduce the original pooled direct-anchor one-step diagnostic."""

    examples = posterior_gate_examples_from_cache(cache)
    direct = infer_direct_anchors(
        examples.metadata, examples.evaluation.dropout_active
    )
    target = np.asarray(examples.labels.canonical_current, dtype=np.float64)
    errors: dict[str, dict[str, list[float]]] = {
        method: {} for method in runtime.trajectories
    }
    for method, trajectory in runtime.trajectories.items():
        post = trajectory.post_update.detach().cpu().numpy().astype(np.float64)
        for index, row in enumerate(runtime.metadata):
            anchor = int(direct[(row.task, row.episode_id)])
            errors[method].setdefault(row.event, []).append(
                float(np.square(post[index, anchor] - target[index, anchor]).mean())
            )
    baseline = errors[MEASUREMENT_METHOD]
    result: dict[str, dict[str, float | None]] = {}
    for method, events in errors.items():
        result[method] = {}
        for event, values in events.items():
            control = math.fsum(baseline[event]) / len(baseline[event])
            candidate = math.fsum(values) / len(values)
            result[method][event] = (
                None if control == 0.0 else 1.0 - candidate / control
            )
        pooled_candidate = events["o1"] + events["o2"]
        pooled_control = baseline["o1"] + baseline["o2"]
        result[method]["o1_o2"] = 1.0 - (
            (math.fsum(pooled_candidate) / len(pooled_candidate))
            / (math.fsum(pooled_control) / len(pooled_control))
        )
    return result


def _oracle_decision(summaries: Mapping[str, Any]) -> dict[str, Any]:
    fsm = summaries[FSM_ORACLE_METHOD]
    primary = fsm["primary"]
    recovery = fsm["recovery_safety"]
    clean = fsm["clean_safety"]
    checks = {
        "primary_improvement_at_least_10pct": (
            float(primary["relative_improvement"]) >= PRIMARY_MINIMUM
        ),
        "primary_two_sided_ci_excludes_zero": (
            float(primary["relative_improvement_95_ci"][0]) > 0.0
        ),
        "recovery_one_sided_upper_below_2pct": (
            float(recovery["normalized_excess_mse_one_sided_95_upper"])
            < SAFETY_LIMIT
        ),
        "clean_one_sided_upper_below_2pct": (
            float(clean["normalized_excess_mse_one_sided_95_upper"])
            < SAFETY_LIMIT
        ),
    }
    passed = all(checks.values())
    return {
        "diagnostic_decision": "Go" if passed else "No-Go",
        "formal_gate_decision": "NotEvaluated",
        "checks": checks,
        "oracle_safety_is_non_evidentiary": True,
        "formal_claim_allowed": False,
        "authorizes_learned_comparator_training": False,
        "authorizes_fresh_randomized_collection": passed,
        "reason": (
            "The current cache has fixed corruption timing and anchor identity; "
            "it can test recursive feasibility but cannot validate a deployable selector."
        ),
    }


def _retention(summaries: Mapping[str, Any]) -> dict[str, float | None]:
    continuous = float(
        summaries[CONVEX_ORACLE_METHOD]["primary"]["relative_improvement"]
    )
    result: dict[str, float | None] = {}
    for method in (HARD_ORACLE_METHOD, SEQUENCE_ORACLE_METHOD, FSM_ORACLE_METHOD):
        value = float(summaries[method]["primary"]["relative_improvement"])
        result[method] = None if continuous <= 0.0 else value / continuous
    return result


def _markdown(payload: Mapping[str, Any]) -> str:
    recursive = payload["recursive"]["methods"]
    lines = [
        "# Posterior Preference Oracle Diagnostic",
        "",
        f"Diagnostic decision: **{payload['decision']['diagnostic_decision']}**",
        f"Formal Gate: **{payload['decision']['formal_gate_decision']}**",
        "",
        "This is a privileged upper-bound diagnostic on the existing fixed-schedule "
        "development cache. It is not deployable evidence and cannot authorize "
        "comparator training.",
        "Recovery and clean safety are zero for target-aware selectors by construction; "
        "they are feasibility bounds, not recovery-detection evidence.",
        "",
        "| Method | Primary improvement | 95% CI | Recovery upper | Clean upper |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in RECURSIVE_CANDIDATES:
        summary = recursive[method]
        primary = summary["primary"]
        lines.append(
            "| {method} | {effect:.2%} | [{low:.2%}, {high:.2%}] | "
            "{recovery:.2%} | {clean:.2%} |".format(
                method=method,
                effect=float(primary["relative_improvement"]),
                low=float(primary["relative_improvement_95_ci"][0]),
                high=float(primary["relative_improvement_95_ci"][1]),
                recovery=float(
                    summary["recovery_safety"][
                        "normalized_excess_mse_one_sided_95_upper"
                    ]
                ),
                clean=float(
                    summary["clean_safety"][
                        "normalized_excess_mse_one_sided_95_upper"
                    ]
                ),
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    inputs = load_posterior_gate_inputs(
        args.config,
        args.train_cache,
        args.validation_cache,
        args.validation_clean_cache,
        repo_root=REPO_ROOT,
    )
    model, _, checkpoint = load_lewm_grounded_checkpoint(
        inputs.lewm_checkpoint, device=inputs.device
    )
    if not isinstance(model, AnchorPersistentLeWM):
        raise TypeError("preference oracle requires AnchorPersistentLeWM")
    if checkpoint.get("step") != 2_000:
        raise ValueError("preference oracle requires the frozen step-2000 LeWM")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    recursive_validation = evaluate_posterior_preference_oracle_cache(
        model,
        inputs.validation,
        methods=PREFERENCE_ORACLE_METHODS,
        device=inputs.device,
    )
    recursive_clean = evaluate_posterior_preference_oracle_cache(
        model,
        inputs.validation_clean,
        methods=PREFERENCE_ORACLE_METHODS,
        device=inputs.device,
    )
    recursive_summaries = _candidate_summaries(
        recursive_validation.records,
        recursive_clean.records,
        RECURSIVE_CANDIDATES,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    teacher_validation = evaluate_teacher_forced_preference_oracle_cache(
        inputs.validation
    )
    teacher_clean = evaluate_teacher_forced_preference_oracle_cache(
        inputs.validation_clean
    )
    teacher_summaries = _candidate_summaries(
        teacher_validation.records,
        teacher_clean.records,
        TEACHER_FORCED_CANDIDATES,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    teacher_event_gain = _global_event_gain(teacher_validation, inputs.validation)
    teacher_continuous_gain = teacher_event_gain[CONVEX_ORACLE_METHOD]["o1_o2"]
    if teacher_continuous_gain is None:
        raise RuntimeError("teacher-forced o1/o2 control MSE unexpectedly vanished")
    teacher_hard_gain = teacher_event_gain[HARD_ORACLE_METHOD]["o1_o2"]
    if teacher_hard_gain is None:
        raise RuntimeError("teacher-forced hard o1/o2 gain is undefined")
    teacher_hard_retention = (
        None
        if teacher_continuous_gain <= 0.0
        else teacher_hard_gain / teacher_continuous_gain
    )
    payload = {
        "schema_version": 1,
        "status": "privileged_fixed_schedule_development_diagnostic",
        "decision": _oracle_decision(recursive_summaries),
        "teacher_forced": {
            "methods": teacher_summaries,
            "beta_mean_by_event": _beta_by_event(
                teacher_validation, inputs.validation
            ),
            "global_direct_anchor_relative_gain_by_event": teacher_event_gain,
            "hard_gain_retention_of_continuous_o1_o2": teacher_hard_retention,
        },
        "recursive": {
            "methods": recursive_summaries,
            "beta_mean_by_event": _beta_by_event(
                recursive_validation, inputs.validation
            ),
            "continuous_primary_gain_retention": _retention(recursive_summaries),
        },
        "limits": [
            "canonical targets and fixed corruption events select privileged oracle actions",
            "zero recovery and clean excess for target-aware selectors is guaranteed by the oracle decision rule and is not deployment evidence",
            "corrupted anchor, onset, duration, and event schedule are fixed in this cache",
            f"FSM streaks count sampled events spaced {FRESH_FRAME_SKIP} raw frames apart, not consecutive raw frames",
            "the exact sequence oracle minimizes cumulative error across the whole five-event stream rather than the o1/o2 primary alone",
            "results may authorize fresh randomized collection only, not comparator training",
        ],
    }
    destination.mkdir(parents=True)
    (destination / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "summary.md").write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "diagnostic_decision": payload["decision"]["diagnostic_decision"],
                "formal_gate_decision": payload["decision"]["formal_gate_decision"],
                "formal_claim_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
