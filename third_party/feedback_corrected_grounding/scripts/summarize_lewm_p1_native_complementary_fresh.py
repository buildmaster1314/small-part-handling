#!/usr/bin/env python3
"""Validate and summarize the formal fresh P1 complementary-observer Gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_p1_native_complementary_statistics import (  # noqa: E402
    CANDIDATE,
    GEOMETRY_HOLD,
    HOLD_LAST,
    HORIZONS,
    MATCHED_NO_ACTION,
    MODEL_SEEDS,
    PROTOCOL,
    REQUIRED_METHODS,
    TASK,
    WRONG_ACTIONS,
    summarize_complementary_fresh_gate,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/p1_native_complementary_fresh.json"
COLLECTOR_PROTOCOL = "fcpg_p1_native_complementary_fresh_v1"
FORMAL_CLASSIFICATION = "formal"
CONFIG_SCOPE = "fresh_pick_known_drop_confirmation_gate"
REQUIRED_FORMAL_FIELDS = (
    "schema_version",
    "classification",
    "collector_protocol",
    "config_sha256",
    "deployment_cache_sha256",
    "label_cache_sha256",
    "metadata_sha256",
)


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("artifact path must be a non-empty string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _require_exact_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(mapping)
    if observed != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _load_fixed_config(path: Path) -> dict[str, Any]:
    canonical_path = DEFAULT_CONFIG.expanduser().resolve()
    canonical = _read_mapping(canonical_path, "frozen config")
    source = path.expanduser().resolve()
    payload = canonical if source == canonical_path else _read_mapping(source, "config")
    if payload != canonical:
        raise ValueError("fresh complementary summary requires the checked-in frozen config")

    scope = _mapping(payload.get("scope"), "config.scope")
    fresh = _mapping(payload.get("fresh_data"), "config.fresh_data")
    observer = _mapping(payload.get("observer"), "config.observer")
    evaluation = _mapping(payload.get("evaluation"), "config.evaluation")
    gate = _mapping(payload.get("gate"), "config.gate")
    predictors = payload.get("frozen_predictors")
    if not isinstance(predictors, list):
        raise TypeError("config.frozen_predictors must be a list")

    gains = _mapping(
        observer.get("gain_by_missing_horizon"),
        "config.observer.gain_by_missing_horizon",
    )
    expected_gains = dict(zip((f"H{horizon}" for horizon in HORIZONS), (0.25, 0.5, 0.75, 0.75, 0.75, 0.75)))
    predictor_seeds = tuple(int(_mapping(item, "frozen predictor")["seed"]) for item in predictors)
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol") != COLLECTOR_PROTOCOL
        or scope.get("classification") != CONFIG_SCOPE
        or scope.get("task") != TASK
        or fresh.get("protocol") != "p1-native-fresh-v1"
        or int(fresh.get("physical_episode_count", -1)) != 30
        or tuple(int(value) for value in fresh.get("offline_label_horizons", ())) != HORIZONS
        or int(fresh.get("branch_count", -1)) != 8
        or predictor_seeds != MODEL_SEEDS
        or dict(gains) != expected_gains
        or int(evaluation.get("bootstrap_samples", -1)) != 10_000
        or int(evaluation.get("bootstrap_seed", -1)) != 41_890
        or float(evaluation.get("confidence", -1.0)) != 0.95
        or evaluation.get("statistical_unit") != "physical_episode"
        or evaluation.get("shared_bootstrap_draws_across_methods_horizons_and_model_seeds") is not True
        or float(gate.get("minimum_relative_improvement", -1.0)) != 0.10
        or float(gate.get("paired_two_sided_ci_lower_must_exceed", -1.0)) != 0.0
        or float(gate.get("h3_one_sided_ci_quantile", -1.0)) != 0.05
        or float(gate.get("h3_maximum_relative_regression", -1.0)) != 0.05
        or int(gate.get("minimum_horizons_beating_all_nonlearning_baselines", -1)) != 5
        or gate.get("all_seven_wrong_action_shifts_required") is not True
        or int(gate.get("minimum_passing_model_seeds", -1)) != 2
        or gate.get("remaining_seed_geometry_and_no_action_points_must_be_positive") is not True
    ):
        raise ValueError("frozen fresh complementary Gate contract drifted")
    return payload


def _validate_live_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    fresh = _mapping(config["fresh_data"], "config.fresh_data")
    paths = {
        "deployment_cache": _resolve_artifact(fresh["deployment_cache"]),
        "label_cache": _resolve_artifact(fresh["label_cache"]),
        "metadata": _resolve_artifact(fresh["metadata"]),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"fresh {label} does not exist: {path}")
    hashes = {f"{label}_sha256": _sha256_file(path) for label, path in paths.items()}

    checkpoint_hashes: dict[int, str] = {}
    checkpoint_paths: dict[int, str] = {}
    for item_value in config["frozen_predictors"]:
        item = _mapping(item_value, "frozen predictor")
        seed = int(item["seed"])
        path = _resolve_artifact(item["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(f"frozen predictor checkpoint does not exist: {path}")
        observed = _sha256_file(path)
        expected = str(item["sha256"])
        if observed != expected:
            raise ValueError(f"frozen predictor checkpoint SHA-256 mismatch for seed {seed}")
        checkpoint_hashes[seed] = observed
        checkpoint_paths[seed] = str(path)

    p1 = _mapping(config["p1"], "config.p1")
    p1_hashes: dict[str, str] = {}
    p1_paths: dict[str, str] = {}
    for label in ("policy", "corrector"):
        path = _resolve_artifact(p1[f"{label}_checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(f"P1 {label} checkpoint does not exist: {path}")
        observed = _sha256_file(path)
        if observed != str(p1[f"{label}_checkpoint_sha256"]):
            raise ValueError(f"P1 {label} checkpoint SHA-256 mismatch")
        p1_hashes[f"{label}_checkpoint_sha256"] = observed
        p1_paths[f"{label}_checkpoint"] = str(path)

    config_sha = _sha256_file(DEFAULT_CONFIG.expanduser().resolve())
    dataset_identity_payload = {
        "config_sha256": config_sha,
        "deployment_cache_sha256": hashes["deployment_cache_sha256"],
        "label_cache_sha256": hashes["label_cache_sha256"],
        "metadata_sha256": hashes["metadata_sha256"],
    }
    return {
        "config_sha256": config_sha,
        **hashes,
        "dataset_sha256": _canonical_digest(dataset_identity_payload),
        "checkpoint_sha256": checkpoint_hashes,
        "paths": {
            **{label: str(path) for label, path in paths.items()},
            **p1_paths,
            "frozen_predictors": checkpoint_paths,
        },
        **p1_hashes,
    }


def _load_formal_rows(
    path: Path,
    *,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"formal JSONL contains a blank line at {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row at line {line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row {line_number} must be an object")
            missing = [field for field in REQUIRED_FORMAL_FIELDS if field not in row]
            if missing:
                raise ValueError(f"formal JSONL row {line_number} is missing fields: {missing}")
            if (
                row["schema_version"] != 1
                or row["classification"] != FORMAL_CLASSIFICATION
                or row["collector_protocol"] != COLLECTOR_PROTOCOL
            ):
                raise ValueError(f"JSONL row {line_number} is not a formal fresh evaluation row")
            for field in (
                "config_sha256",
                "deployment_cache_sha256",
                "label_cache_sha256",
                "metadata_sha256",
            ):
                if row[field] != provenance[field]:
                    raise ValueError(f"JSONL row {line_number} {field} provenance mismatch")
            rows.append(row)

    expected_rows = len(MODEL_SEEDS) * len(REQUIRED_METHODS) * 30 * len(HORIZONS)
    if len(rows) != expected_rows:
        raise ValueError(
            f"formal JSONL must contain exactly {expected_rows} rows; observed {len(rows)}"
        )
    return rows


def summarize_formal_rows(
    rows_path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = _load_fixed_config(config_path)
    provenance = _validate_live_artifacts(config)
    rows = _load_formal_rows(rows_path, provenance=provenance)
    evaluation = _mapping(config["evaluation"], "config.evaluation")
    summary = summarize_complementary_fresh_gate(
        rows,
        expected_dataset_sha256=str(provenance["dataset_sha256"]),
        expected_checkpoint_sha256=provenance["checkpoint_sha256"],
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
    )
    summary["formal_input"] = {
        "classification": FORMAL_CLASSIFICATION,
        "collector_protocol": COLLECTOR_PROTOCOL,
        "row_count": len(rows),
        "rows_path": str(rows_path.expanduser().resolve()),
        "rows_sha256": _sha256_file(rows_path.expanduser().resolve()),
        "config_path": str(config_path.expanduser().resolve()),
        **{key: value for key, value in provenance.items() if key != "checkpoint_sha256"},
    }
    return summary


def _pct(value: object) -> str:
    return f"{100.0 * float(value):.2f}%"


def render_markdown(summary: Mapping[str, Any]) -> str:
    gate = _mapping(summary["cross_seed_gate"], "summary.cross_seed_gate")
    lines = [
        "# Fresh P1 Native Complementary Observer Gate",
        "",
        f"**Decision: {gate['decision']}** ({gate['passing_seed_count']}/3 model seeds passed)",
        "",
        "| Seed | Gate | MSE | vs no-action | vs hold-last | vs geometry | H3 point / q05 | Positive horizons | Wrong-action passes |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    comparison_names = (MATCHED_NO_ACTION, HOLD_LAST, GEOMETRY_HOLD)
    for value in summary["per_seed"]:
        seed = _mapping(value, "per-seed result")
        comparisons = _mapping(seed["comparisons"], "per-seed comparisons")
        effects = [
            _pct(_mapping(comparisons[f"factual_vs_{name}"], name)["relative_improvement"])
            for name in comparison_names
        ]
        h3 = _mapping(
            _mapping(seed["strongest_nonlearning_per_horizon"], "per-horizon")["H3"],
            "H3",
        )
        screen = _mapping(seed["screen"], "per-seed screen")
        wrong_pass = _mapping(screen["wrong_action_pass"], "wrong-action screen")
        lines.append(
            "| "
            f"{seed['model_seed']} | {'Go' if screen['passed'] else 'No-Go'} | "
            f"{float(seed['physical_episode_macro_mse']):.6f} | "
            f"{effects[0]} | {effects[1]} | {effects[2]} | "
            f"{_pct(h3['simultaneous_strongest_relative_improvement'])} / "
            f"{_pct(h3['simultaneous_one_sided_95_lower'])} | "
            f"{screen['positive_horizon_count']}/6 | "
            f"{sum(bool(item) for item in wrong_pass.values())}/7 |"
        )
    lines.extend(
        [
            "",
            "The statistical unit is the physical episode. One shared physical-episode "
            "bootstrap draw matrix is used across methods, horizons, and model seeds.",
            "",
            "This formal Gate covers known visual dropout only. A positive result authorizes "
            "paired known-drop policy rollout; it does not establish an automatic validity "
            "selector, unknown-corruption robustness, CEM, or new 10k training.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_fsync(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_summary_atomic(summary: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        json_path = temporary / "summary.json"
        markdown_path = temporary / "summary.md"
        _write_fsync(json_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _write_fsync(markdown_path, render_markdown(summary))
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "summary.json", output / "summary.md"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rows", type=Path, required=True, help="formal evaluator JSONL rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_formal_rows(args.rows, config_path=args.config)
    json_path, markdown_path = write_summary_atomic(summary, args.output_dir)
    print(markdown_path.read_text(encoding="utf-8"), end="")
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
