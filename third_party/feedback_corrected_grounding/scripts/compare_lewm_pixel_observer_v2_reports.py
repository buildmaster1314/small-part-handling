#!/usr/bin/env python3
"""Compare matched PixelObserver v2 latent-on and latent-off reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_pixel_observer_v2_report import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_RELATIVE_PATHS,
    PRIMARY_CONTROL_METRIC_FAMILY,
    condition_primary_metric,
    matched_training_protocol,
    validate_episode_rows,
)
from fcpg.lewm_pixel_observer_v2_runtime import canonical_sha256  # noqa: E402
from fcpg.lewm_provenance import sha256_file  # noqa: E402


MATCH_FIELDS = (
    "target_mode",
    "base_checkpoint_sha256",
    "checkpoint_training_budget",
    "checkpoint_matched_training_protocol_sha256",
    "training_data_provenance_sha256",
    "implementation_sha256",
    "evaluation_config_sha256",
    "evaluation_data_provenance_sha256",
    "split",
    "evaluation_condition",
    "evaluation_matrix_profile",
    "max_episodes_per_source",
    "evaluation_episode_selection_sha256",
    "observer_gains",
    "classic_baselines",
    "synthetic_measurement_dropout_contract",
    "hidden_step_negative_control",
    "primary_control_metric_family",
    "primary_control_metric",
)

METRICS = (
    "prior_qtrue_dropout_epe_px",
    "prior_qtrue_epe_px",
    "post_update_qtrue_dropout_epe_px",
    "post_update_qtrue_epe_px",
)

def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report root must be a mapping: {path}")
    return value


def _require_sha256(report: Mapping[str, Any], field: str) -> str:
    value = report.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"report lacks a valid {field}")
    return value


def _require_mapping(
    report: Mapping[str, Any], field: str
) -> Mapping[str, Any]:
    value = report.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"report lacks mapping field: {field}")
    return value


def _validate_implementation_manifest(report: Mapping[str, Any]) -> None:
    manifest = dict(_require_mapping(report, "implementation_files"))
    required = set(EVALUATION_IMPLEMENTATION_RELATIVE_PATHS)
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(
            "implementation_files lacks required evaluation files: "
            + ", ".join(missing)
        )
    for path, value in manifest.items():
        _require_sha256({"value": value}, "value")
        if not isinstance(path, str) or not path:
            raise ValueError("implementation_files contains an invalid path")
    declared = _require_sha256(report, "implementation_sha256")
    if canonical_sha256(manifest) != declared:
        raise ValueError("implementation_files hash binding is invalid")


def _validated_training_protocol(
    report: Mapping[str, Any], *, expected_use_predicted_latent: bool
) -> dict[str, Any]:
    resolved = dict(_require_mapping(report, "checkpoint_resolved_config"))
    resolved_sha256 = _require_sha256(
        report, "checkpoint_resolved_config_sha256"
    )
    if canonical_sha256(resolved) != resolved_sha256:
        raise ValueError("checkpoint resolved_config hash binding is invalid")
    protocol = matched_training_protocol(
        resolved,
        expected_use_predicted_latent=expected_use_predicted_latent,
    )
    declared_sha256 = _require_sha256(
        report, "checkpoint_matched_training_protocol_sha256"
    )
    if canonical_sha256(protocol) != declared_sha256:
        raise ValueError("checkpoint matched training protocol hash is invalid")
    return protocol


def _validate_matched(
    latent: Mapping[str, Any], no_latent: Mapping[str, Any]
) -> str:
    if latent.get("target_mode") != "paired_deployment":
        raise ValueError("latent report is not a paired-deployment checkpoint")
    if no_latent.get("target_mode") != "paired_deployment":
        raise ValueError("no-latent report is not a paired-deployment checkpoint")
    _validate_implementation_manifest(latent)
    _validate_implementation_manifest(no_latent)
    latent_protocol = _validated_training_protocol(
        latent, expected_use_predicted_latent=True
    )
    no_latent_protocol = _validated_training_protocol(
        no_latent, expected_use_predicted_latent=False
    )
    if latent_protocol != no_latent_protocol:
        raise ValueError(
            "matched training protocols differ beyond output_dir and "
            "use_predicted_latent"
        )
    for report in (latent, no_latent):
        if (
            report.get("primary_control_metric_family")
            != PRIMARY_CONTROL_METRIC_FAMILY
        ):
            raise ValueError(
                "report has an invalid primary_control_metric_family"
            )
    for name in MATCH_FIELDS:
        if name not in latent or name not in no_latent:
            raise ValueError(f"matched report field is missing: {name}")
        if latent.get(name) != no_latent.get(name):
            raise ValueError(f"matched report field differs: {name}")
    latent_config = dict(
        _require_mapping(latent, "checkpoint_adapter_config")
    )
    no_latent_config = dict(
        _require_mapping(no_latent, "checkpoint_adapter_config")
    )
    if latent_config.pop("use_predicted_latent", None) is not True:
        raise ValueError("latent report does not enable predicted latent input")
    if no_latent_config.pop("use_predicted_latent", None) is not False:
        raise ValueError("no-latent report does not disable predicted latent input")
    if latent_config != no_latent_config:
        raise ValueError("adapter configs differ beyond use_predicted_latent")
    dropout = _require_mapping(
        latent, "synthetic_measurement_dropout_contract"
    )
    hidden = _require_mapping(latent, "hidden_step_negative_control")
    dropout_length = dropout.get("length")
    hidden_length = hidden.get("sampled_steps")
    if (
        isinstance(dropout_length, bool)
        or not isinstance(dropout_length, int)
        or dropout_length < 0
        or isinstance(hidden_length, bool)
        or not isinstance(hidden_length, int)
        or hidden_length < 0
    ):
        raise ValueError("evaluation condition lengths are invalid")
    expected_primary = condition_primary_metric(
        dropout_length=dropout_length,
        hidden_step_length=hidden_length,
    )
    if latent.get("primary_control_metric") != expected_primary:
        raise ValueError(
            "primary_control_metric does not match the evaluation condition"
        )
    return expected_primary


def _episode_map(
    report: Mapping[str, Any],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    rows = report.get("episode_rows")
    if not isinstance(rows, list):
        raise ValueError("report episode_rows must be a list")
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("episode row must be a mapping")
        row = dict(raw)
        try:
            key = (
                str(row["source"]),
                str(row["task"]),
                str(row["episode"]),
            )
        except KeyError:
            raise ValueError("episode row lacks an identity field") from None
        if key in output:
            raise ValueError(f"duplicate episode row: {key}")
        output[key] = row
    if not output:
        raise ValueError("report contains no episode rows")
    return output


def _validate_formal_episode_selection(
    report: Mapping[str, Any],
    rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    primary_metric: str,
) -> None:
    maximum = report.get("max_episodes_per_source")
    if isinstance(maximum, bool) or maximum != 0:
        raise ValueError(
            "formal comparison requires max_episodes_per_source=0"
        )
    selection = dict(
        _require_mapping(report, "evaluation_episode_selection")
    )
    selection_sha256 = _require_sha256(
        report, "evaluation_episode_selection_sha256"
    )
    if canonical_sha256(selection) != selection_sha256:
        raise ValueError("evaluation episode selection hash is invalid")
    validate_episode_rows(
        list(rows.values()),
        coverage=selection,
        primary_metric=primary_metric,
    )


def _paired_metric(
    latent_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    no_latent_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any] | None:
    grouped: dict[str, list[tuple[float, float]]] = {}
    missing = 0
    for key in sorted(latent_rows):
        latent_modes = latent_rows[key].get("modes")
        no_latent_modes = no_latent_rows[key].get("modes")
        latent_full = (
            latent_modes.get("full_writeback")
            if isinstance(latent_modes, Mapping)
            else None
        )
        no_latent_full = (
            no_latent_modes.get("full_writeback")
            if isinstance(no_latent_modes, Mapping)
            else None
        )
        latent_value = (
            latent_full.get(metric)
            if isinstance(latent_full, Mapping)
            else None
        )
        no_latent_value = (
            no_latent_full.get(metric)
            if isinstance(no_latent_full, Mapping)
            else None
        )
        if (latent_value is None) != (no_latent_value is None):
            raise ValueError(f"metric availability differs for {key}: {metric}")
        if latent_value is None:
            missing += 1
            continue
        if (
            isinstance(latent_value, bool)
            or not isinstance(latent_value, (int, float))
            or isinstance(no_latent_value, bool)
            or not isinstance(no_latent_value, (int, float))
        ):
            raise ValueError(f"metric is not numeric for {key}: {metric}")
        latent_float = float(latent_value)
        no_latent_float = float(no_latent_value)
        if not np.isfinite(latent_float) or not np.isfinite(no_latent_float):
            raise ValueError(f"metric is not finite for {key}: {metric}")
        grouped.setdefault(key[1], []).append(
            (latent_float, no_latent_float)
        )
    if missing:
        if grouped:
            raise ValueError(
                f"metric coverage is incomplete across episodes: {metric}"
            )
        return None
    if not grouped:
        return None
    arrays = {
        task: np.asarray(values, dtype=np.float64)
        for task, values in sorted(grouped.items())
    }
    task_latent = np.asarray([values[:, 0].mean() for values in arrays.values()])
    task_no_latent = np.asarray([values[:, 1].mean() for values in arrays.values()])
    differences = task_latent - task_no_latent
    relative = np.divide(
        task_no_latent - task_latent,
        task_no_latent,
        out=np.full_like(task_latent, np.nan),
        where=task_no_latent != 0.0,
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap = np.empty(int(bootstrap_samples), dtype=np.float64)
    for index in range(bootstrap.size):
        sampled = []
        for values in arrays.values():
            chosen = values[rng.integers(0, len(values), size=len(values))]
            sampled.append(float(chosen[:, 0].mean() - chosen[:, 1].mean()))
        bootstrap[index] = float(np.mean(sampled))
    pooled = np.concatenate(list(arrays.values()), axis=0)
    finite_relative = relative[np.isfinite(relative)]
    return {
        "metric": metric,
        "episodes": int(len(pooled)),
        "task_macro_latent_px": float(task_latent.mean()),
        "task_macro_no_latent_px": float(task_no_latent.mean()),
        "task_macro_difference_latent_minus_no_latent_px": float(differences.mean()),
        "task_macro_relative_improvement_from_latent": (
            float(finite_relative.mean()) if finite_relative.size else None
        ),
        "episode_pooled_difference_latent_minus_no_latent_px": float(
            (pooled[:, 0] - pooled[:, 1]).mean()
        ),
        "task_differences_latent_minus_no_latent_px": {
            task: float(values[:, 0].mean() - values[:, 1].mean())
            for task, values in arrays.items()
        },
        "task_stratified_bootstrap95_ci_px": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "negative_favors_latent": True,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_samples": int(bootstrap_samples),
    }


def compare_reports(
    latent: Mapping[str, Any],
    no_latent: Mapping[str, Any],
    *,
    latent_report_sha256: str,
    no_latent_report_sha256: str,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 7168,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    _require_sha256(
        {"latent_report_sha256": latent_report_sha256},
        "latent_report_sha256",
    )
    _require_sha256(
        {"no_latent_report_sha256": no_latent_report_sha256},
        "no_latent_report_sha256",
    )
    primary_control_metric = _validate_matched(latent, no_latent)
    latent_rows = _episode_map(latent)
    no_latent_rows = _episode_map(no_latent)
    if latent_rows.keys() != no_latent_rows.keys():
        raise ValueError("latent/no-latent episode sets differ")
    _validate_formal_episode_selection(
        latent, latent_rows, primary_control_metric
    )
    _validate_formal_episode_selection(
        no_latent, no_latent_rows, primary_control_metric
    )
    comparisons = {}
    for metric in METRICS:
        value = _paired_metric(
            latent_rows,
            no_latent_rows,
            metric,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if value is not None:
            comparisons[metric] = value
    if primary_control_metric not in comparisons:
        raise ValueError("primary metric comparison is missing")
    comparison_script = Path(__file__).resolve()
    comparison_script_sha256 = sha256_file(comparison_script)
    return {
        "schema_version": 2,
        "comparison": "matched_predicted_latent_ablation",
        "comparison_script": str(comparison_script),
        "comparison_script_sha256": comparison_script_sha256,
        "latent_report_sha256": latent_report_sha256,
        "no_latent_report_sha256": no_latent_report_sha256,
        "latent_checkpoint": latent["checkpoint"],
        "latent_checkpoint_sha256": latent["checkpoint_sha256"],
        "no_latent_checkpoint": no_latent["checkpoint"],
        "no_latent_checkpoint_sha256": no_latent["checkpoint_sha256"],
        "evaluation_data_provenance_sha256": latent[
            "evaluation_data_provenance_sha256"
        ],
        "checkpoint_matched_training_protocol_sha256": latent[
            "checkpoint_matched_training_protocol_sha256"
        ],
        "evaluation_episode_selection_sha256": latent[
            "evaluation_episode_selection_sha256"
        ],
        "synthetic_measurement_dropout_contract": latent[
            "synthetic_measurement_dropout_contract"
        ],
        "hidden_step_negative_control": latent["hidden_step_negative_control"],
        "primary_control_metric_family": PRIMARY_CONTROL_METRIC_FAMILY,
        "primary_control_metric": primary_control_metric,
        "primary_metric": comparisons[primary_control_metric],
        "metrics": comparisons,
        "claim_rule": (
            "A LeWM contribution requires the latent-on full-writeback error to be "
            "lower than the parameter-matched latent-off adapter on the same episodes; "
            "negative paired differences favor latent-on."
        ),
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite comparison report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(
        f".{resolved.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-report", type=Path, required=True)
    parser.add_argument("--no-latent-report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=7168)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latent_report = args.latent_report.expanduser().resolve()
    no_latent_report = args.no_latent_report.expanduser().resolve()
    payload = compare_reports(
        _load(latent_report),
        _load(no_latent_report),
        latent_report_sha256=sha256_file(latent_report),
        no_latent_report_sha256=sha256_file(no_latent_report),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _atomic_write(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
