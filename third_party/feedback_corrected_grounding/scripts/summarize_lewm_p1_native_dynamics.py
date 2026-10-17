#!/usr/bin/env python3
"""Validate and aggregate the three frozen P1-native dynamics runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a3_p1_bridge_collection import sha256_file  # noqa: E402
from fcpg.lewm_p1_native_dynamics import (  # noqa: E402
    P1_NATIVE_PROTOCOL,
    P1_NATIVE_TARGET_HORIZONS,
    P1_NATIVE_WRONG_ACTION_SHIFTS,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/p1_native_dynamics_dev.json"
EXPECTED_SEEDS = (15801, 15802, 15803)
CHECKPOINT_TYPE = "fcpg_p1_native_dynamics_pair_v1"
STATIC_PREDICTION_KEYS = (
    "target_token",
    "wrong_action_shift",
    "hold_last_prediction",
    "geometry_hold_exact_fk_prediction",
    "visual_cv_exact_fk_prediction",
    "physical_episode_id",
    "source_row_index",
    "branch_index",
    "hazard_id",
    "horizon",
)
PRIMARY_COMPARISONS = (
    "factual_vs_matched_no_action",
    "factual_vs_hold_last",
    "factual_vs_geometry_hold_exact_fk",
    "factual_vs_visual_cv_exact_fk",
)


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_artifact(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _finite_float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load_fixed_config(path: Path) -> dict[str, Any]:
    canonical = _read_mapping(DEFAULT_CONFIG.resolve(), "frozen config")
    source = path.expanduser().resolve()
    payload = canonical if source == DEFAULT_CONFIG.resolve() else _read_mapping(source, "config")
    if payload != canonical:
        raise ValueError("P1-native summary requires the checked-in frozen config")
    training = _mapping(payload.get("training"), "config.training")
    gate = _mapping(payload.get("gate"), "config.gate")
    if (
        payload.get("protocol") != P1_NATIVE_PROTOCOL
        or tuple(int(seed) for seed in training.get("allowed_seeds", ())) != EXPECTED_SEEDS
        or int(training.get("optimizer_steps", -1)) != 2000
        or int(gate.get("minimum_passing_seeds", -1)) != 2
        or gate.get("remaining_seed_point_estimates_must_be_positive") is not True
    ):
        raise ValueError("frozen P1-native cross-seed contract drifted")
    return payload


def _array_digest(archive: np.lib.npyio.NpzFile) -> str:
    digest = hashlib.sha256()
    missing = [name for name in STATIC_PREDICTION_KEYS if name not in archive.files]
    if missing:
        raise ValueError(f"prediction archive lacks static provenance arrays: {missing}")
    for name in STATIC_PREDICTION_KEYS:
        value = np.ascontiguousarray(archive[name])
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _recompute_single_seed_screen(
    report: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    gate = _mapping(config["gate"], "config.gate")
    comparisons = _mapping(report.get("comparisons"), "report.comparisons")
    screen = _mapping(report.get("screen"), "report.screen")
    minimum_effect = float(gate["minimum_relative_improvement"])
    ci_threshold = float(gate["paired_bootstrap_ci_lower_must_exceed"])

    primary: dict[str, bool] = {}
    for name in PRIMARY_COMPARISONS:
        comparison = _mapping(comparisons.get(name), name)
        interval = comparison.get("confidence_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{name} confidence_interval must contain two values")
        primary[name] = (
            _finite_float(comparison.get("relative_improvement"), f"{name}.effect")
            >= minimum_effect
            and _finite_float(interval[0], f"{name}.ci_lower") > ci_threshold
        )

    wrong: dict[str, bool] = {}
    for shift in P1_NATIVE_WRONG_ACTION_SHIFTS:
        name = f"factual_vs_wrong_action_cyclic_shift_{shift}"
        comparison = _mapping(comparisons.get(name), name)
        interval = comparison.get("confidence_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{name} confidence_interval must contain two values")
        wrong[name] = (
            _finite_float(comparison.get("relative_improvement"), f"{name}.effect") > 0.0
            and _finite_float(interval[0], f"{name}.ci_lower") > ci_threshold
        )

    horizon_values = _mapping(
        screen.get("strongest_nonlearning_per_horizon_relative_improvement"),
        "screen.strongest_nonlearning_per_horizon_relative_improvement",
    )
    expected_horizon_keys = tuple(f"H{value}" for value in P1_NATIVE_TARGET_HORIZONS)
    if tuple(horizon_values.keys()) != expected_horizon_keys:
        raise ValueError("single-seed screen must report ordered H3-H8 horizon values")
    positive_horizons = sum(
        _finite_float(horizon_values[key], f"screen.{key}") > 0.0
        for key in expected_horizon_keys
    )
    horizon_pass = positive_horizons >= int(
        gate["minimum_horizons_beating_strongest_nonlearning"]
    )
    h3_pass = _finite_float(horizon_values["H3"], "screen.H3") >= -float(
        gate["h3_maximum_relative_regression"]
    )
    passed = all(primary.values()) and all(wrong.values()) and horizon_pass and h3_pass

    expected_fields = {
        "primary_pass": primary,
        "wrong_action_pass": wrong,
        "positive_horizon_count": positive_horizons,
        "horizon_pass": horizon_pass,
        "h3_noninferiority_pass": h3_pass,
        "passed": passed,
    }
    for name, expected in expected_fields.items():
        if name == "primary_pass":
            observed = screen.get(name)
            legacy_primary = {
                key: value
                for key, value in primary.items()
                if key != "factual_vs_hold_last"
            }
            # Seed 15801 finished before hold-last became a required primary
            # Gate. Its immutable report still contains the hold comparison,
            # so recompute that missing failure without retraining the model.
            if (
                int(report.get("seed", -1)) == 15801
                and primary["factual_vs_hold_last"] is False
                and observed == legacy_primary
            ):
                continue
        if screen.get(name) != expected:
            raise ValueError(f"single-seed screen field {name} is inconsistent with report metrics")
    return {**expected_fields, "horizon_values": dict(horizon_values)}


def _validate_run(
    report_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    report = _read_mapping(report_path, "report")
    seed = int(report.get("seed", -1))
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unexpected P1-native seed: {seed}")
    if (
        report.get("protocol") != P1_NATIVE_PROTOCOL
        or report.get("scope") != "exposed_train_development_only"
        or tuple(int(value) for value in report.get("forecast_horizons", ()))
        != P1_NATIVE_TARGET_HORIZONS
        or report.get("statistical_unit") != "physical_episode"
        or int(report.get("optimizer_steps", -1)) != 2000
        or report.get("smoke") is not False
        or report.get("selection") != "fixed_latest"
    ):
        raise ValueError(f"seed {seed} is not an exact non-smoke 2,000-step P1-native run")

    resolved = _read_mapping(report_path.parent / "resolved_config.json", "resolved config")
    if resolved != config:
        raise ValueError(f"seed {seed} resolved config differs from the frozen config")
    screen = _recompute_single_seed_screen(report, config)

    artifacts = _mapping(report.get("artifacts"), "report.artifacts")
    checkpoint_path = _resolve_artifact(artifacts.get("checkpoint"))
    predictions_path = _resolve_artifact(artifacts.get("predictions"))
    for label, path, expected in (
        ("checkpoint", checkpoint_path, artifacts.get("checkpoint_sha256")),
        ("predictions", predictions_path, artifacts.get("predictions_sha256")),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(expected):
            raise ValueError(f"seed {seed} {label} SHA-256 mismatch")

    # Training checkpoints include NumPy action-normalization arrays, which the
    # weights-only loader rejects. The file digest is verified before this load.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("P1-native checkpoint must contain a mapping")
    training = _mapping(config["training"], "config.training")
    expected_checkpoint = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "protocol": P1_NATIVE_PROTOCOL,
        "selection": "fixed_latest",
        "completed": True,
        "step": int(training["optimizer_steps"]),
        "seed": seed,
        "config": config,
        "strict_cache_sha256": _mapping(config["data"], "config.data")[
            "strict_cache_sha256"
        ],
        "p1_policy_checkpoint_sha256": _mapping(config["p1"], "config.p1")[
            "policy_checkpoint_sha256"
        ],
        "report_screen": report["screen"],
    }
    for name, expected in expected_checkpoint.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"seed {seed} checkpoint field {name} violates provenance")
    expected_predictor_config = {
        name: value
        for name, value in _mapping(config["model"], "config.model").items()
        if name != "sigreg_weight"
    }
    if checkpoint.get("predictor_config") != expected_predictor_config:
        raise ValueError(f"seed {seed} checkpoint predictor_config violates provenance")
    if int(checkpoint.get("parameter_count_per_head", -1)) != int(
        report.get("parameter_count_per_head", -2)
    ):
        raise ValueError(f"seed {seed} parameter count differs between checkpoint and report")

    with np.load(predictions_path, allow_pickle=False) as archive:
        provenance_digest = _array_digest(archive)

    metrics = _mapping(report.get("metrics"), "report.metrics")
    factual = _mapping(metrics.get("factual"), "report.metrics.factual")
    effects = {
        name: _finite_float(
            _mapping(_mapping(report["comparisons"], "comparisons")[name], name)[
                "relative_improvement"
            ],
            name,
        )
        for name in PRIMARY_COMPARISONS
    }
    return {
        "seed": seed,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "predictions_sha256": sha256_file(predictions_path),
        "development_provenance_sha256": provenance_digest,
        "config_sha256": _canonical_digest(config),
        "predictor_config_sha256": _canonical_digest(checkpoint.get("predictor_config")),
        "parameter_count_per_head": int(report["parameter_count_per_head"]),
        "development_physical_episode_count": int(
            report["development_physical_episode_count"]
        ),
        "factual_physical_episode_macro_mse": _finite_float(
            factual.get("physical_episode_macro_mse"), "factual MSE"
        ),
        "primary_relative_improvement": effects,
        "remaining_seed_point_estimates_positive": all(value > 0.0 for value in effects.values()),
        "positive_horizon_count": int(screen["positive_horizon_count"]),
        "passed": bool(screen["passed"]),
    }


def summarize_reports(
    report_paths: Sequence[Path],
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Validate exactly three fixed-seed reports and apply the cross-seed Gate."""

    if len(report_paths) != len(EXPECTED_SEEDS):
        raise ValueError("P1-native summary requires exactly three reports")
    config = _load_fixed_config(config_path)
    rows = [_validate_run(path, config) for path in report_paths]
    seeds = tuple(sorted(int(row["seed"]) for row in rows))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"reports must contain each fixed seed exactly once: {EXPECTED_SEEDS}")
    rows.sort(key=lambda row: int(row["seed"]))

    for field in (
        "config_sha256",
        "predictor_config_sha256",
        "parameter_count_per_head",
        "development_physical_episode_count",
        "development_provenance_sha256",
    ):
        observed = {row[field] for row in rows}
        if len(observed) != 1:
            raise ValueError(f"cross-seed {field} mismatch")

    passing = [int(row["seed"]) for row in rows if row["passed"]]
    remaining = [row for row in rows if not row["passed"]]
    minimum_passing = int(_mapping(config["gate"], "config.gate")["minimum_passing_seeds"])
    enough_passing = len(passing) >= minimum_passing
    remaining_positive = all(
        bool(row["remaining_seed_point_estimates_positive"]) for row in remaining
    )
    passed = enough_passing and remaining_positive
    return {
        "schema_version": 1,
        "protocol": P1_NATIVE_PROTOCOL,
        "scope": "bounded_exposed_train_development_cross_seed_gate",
        "seeds": list(EXPECTED_SEEDS),
        "config_sha256": rows[0]["config_sha256"],
        "development_provenance_sha256": rows[0]["development_provenance_sha256"],
        "per_seed": rows,
        "cross_seed_gate": {
            "minimum_passing_seeds": minimum_passing,
            "passing_seed_count": len(passing),
            "passing_seeds": passing,
            "remaining_seeds": [int(row["seed"]) for row in remaining],
            "at_least_two_of_three_pass": enough_passing,
            "remaining_seed_point_estimates_positive": remaining_positive,
            "passed": passed,
            "decision": "Development-Go" if passed else "Development-No-Go",
        },
        "claim_boundary": {
            "independent_test_claim": False,
            "policy_rollout_authorized": False,
            "ten_thousand_step_training_authorized": False,
            "positive_result_only_merits_fresh_policy_confirmation": True,
        },
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    gate = _mapping(summary["cross_seed_gate"], "cross_seed_gate")
    lines = [
        "# P1-Native Dynamics Cross-Seed Gate",
        "",
        f"**Decision: {gate['decision']}** "
        f"({gate['passing_seed_count']}/3 single-seed Gates passed).",
        "",
        "| seed | single-seed Gate | factual MSE | vs no-action | vs hold-last | "
        "vs geometry-hold | vs visual-CV | horizons won |",
        "| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["per_seed"]:
        effects = row["primary_relative_improvement"]
        lines.append(
            f"| {row['seed']} | {'Go' if row['passed'] else 'No-Go'} | "
            f"{row['factual_physical_episode_macro_mse']:.6f} | "
            f"{effects['factual_vs_matched_no_action']:+.2%} | "
            f"{effects['factual_vs_hold_last']:+.2%} | "
            f"{effects['factual_vs_geometry_hold_exact_fk']:+.2%} | "
            f"{effects['factual_vs_visual_cv_exact_fk']:+.2%} | "
            f"{row['positive_horizon_count']}/6 |"
        )
    lines.extend(
        [
            "",
            "Cross-seed success requires at least two single-seed passes and positive "
            "point estimates against matched no-action, hold-last, exact-FK geometry "
            "hold, and exact-FK visual CV for every remaining seed.",
            "",
            "This is an exposed train/development Gate only. It does not authorize a "
            "policy, independent-test, or 10k claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        help="report.json from one fixed seed; omit to use the frozen output template",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/lewm_p1_native_dynamics/cross_seed_summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_fixed_config(args.config)
    if args.report is None:
        template = str(config["output_dir_template"])
        reports = [ROOT / template.format(seed=seed) / "report.json" for seed in EXPECTED_SEEDS]
    else:
        reports = args.report
    summary = summarize_reports(reports, config_path=args.config)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "summary.json"
    markdown_path = output / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"), end="")
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
