#!/usr/bin/env python3
"""Aggregate matched FCPG x LeWM reports across model seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.evaluate_fcpg_lewm_factorial import (
    CELL_SPECS,
    CANONICAL_BASELINE_CELLS,
    PREFIX_HASH_FIELDS,
    PROTOCOL,
    _validate_physical_pairing,
    crossed_seed_factorial_effects,
)


METRICS = (
    "success",
    "sustained_success",
    "final_success",
    "post_drop_success_fraction",
    "post_drop_maximum_success_streak",
    "reward_sum",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _without_model_seed(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("model_seed", None)
    return value


def _require_sha256(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _indexed_report_rows(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    episodes = int(report["episodes"])
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != episodes * len(CELL_SPECS):
        raise ValueError("factorial report must contain exactly episodes * 4 rows")
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise TypeError("factorial rows must be mappings")
        key = (int(raw["physical_episode"]), str(raw["method"]))
        if key in indexed:
            raise ValueError(f"duplicate factorial row {key}")
        indexed[key] = raw
    expected = {
        (episode, cell)
        for episode in range(episodes)
        for cell in CELL_SPECS
    }
    if set(indexed) != expected:
        raise ValueError("factorial report has missing or unexpected episode/cell rows")
    _validate_physical_pairing(rows)
    return indexed


def _validate_report_contract(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    if report.get("schema_version") != 2:
        raise ValueError("factorial report schema_version must be 2")
    if report.get("protocol") != PROTOCOL:
        raise ValueError("factorial report protocol does not match the paired-prefix protocol")
    if report.get("pairing_audit") != "passed":
        raise ValueError("factorial report did not pass its pairing audit")
    if report.get("run_role") not in {"canonical", "dynamics_only"}:
        raise ValueError("factorial report has an invalid run_role")
    expected_semantics = {
        cell: {"grounding": grounding, "dropout_state": method}
        for cell, (grounding, method) in CELL_SPECS.items()
    }
    if report.get("cell_semantics") != expected_semantics:
        raise ValueError("factorial report cell semantics mismatch")
    if not isinstance(report.get("classification"), str):
        raise TypeError("factorial report classification must be a string")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("factorial report is missing provenance")
    required_provenance = {
        "protocol",
        "git_commit",
        "evaluator_sha256",
        "rollout_evaluator_sha256",
        "p1_runtime_sha256",
        "prefix_bundle_module_sha256",
        "device",
        "config_sha256",
        "source_config_sha256",
        "policy_checkpoint_sha256",
        "corrector_checkpoint_sha256",
        "task",
        "camera",
        "physical_episode_count",
        "model_seeds",
        "canonical_model_seed",
        "benchmark_seed",
        "environment_seed_base",
        "task_instance_indices",
        "drop_start_decisions",
        "drop_duration_decisions",
        "horizon_raw_steps",
        "occlusion_radius_px",
        "occlusion_value_uint8",
        "camera_rotation_axis_world",
        "camera_rotation_deg",
    }
    missing = sorted(required_provenance.difference(provenance))
    if missing:
        raise ValueError(f"factorial report provenance is missing {missing}")
    if provenance["protocol"] != PROTOCOL:
        raise ValueError("factorial provenance protocol mismatch")
    moving_goal = provenance.get("moving_goal", {"enabled": False})
    if not isinstance(moving_goal, dict) or not isinstance(
        moving_goal.get("enabled"), bool
    ):
        raise ValueError("factorial provenance has an invalid moving-goal contract")
    if moving_goal["enabled"]:
        required_moving_goal = {
            "enabled",
            "seed_base",
            "amplitude_xy",
            "period_raw_steps",
        }
        if set(moving_goal) != required_moving_goal:
            raise ValueError("factorial provenance has an incomplete moving-goal contract")
    model_seed = int(report["model_seed"])
    canonical_model_seed = int(report.get("canonical_model_seed", -1))
    if canonical_model_seed != int(provenance["canonical_model_seed"]):
        raise ValueError("canonical model seed disagrees with provenance")
    expected_role = "canonical" if model_seed == canonical_model_seed else "dynamics_only"
    if report["run_role"] != expected_role:
        raise ValueError("factorial report run_role disagrees with model seeds")
    predictors = report.get("predictor_matching")
    if not isinstance(predictors, dict) or set(predictors) != {"nominal", "fcpg"}:
        raise ValueError("factorial report must bind nominal and fcpg predictors")
    for grounding, record in predictors.items():
        if not isinstance(record, dict):
            raise TypeError(f"{grounding} predictor provenance must be a mapping")
        for field in (
            "seed",
            "training_config_sha256",
            "checkpoint",
            "checkpoint_sha256",
        ):
            if field not in record:
                raise ValueError(f"{grounding} predictor provenance is missing {field}")
        if int(record["seed"]) != model_seed:
            raise ValueError(f"{grounding} predictor seed does not match report seed")
        _require_sha256(record["training_config_sha256"], f"{grounding} training config")
        _require_sha256(record["checkpoint_sha256"], f"{grounding} checkpoint")
    manifests = report.get("prefix_bundle_manifests")
    if not isinstance(manifests, dict) or len(manifests) != int(report["episodes"]) * 2:
        raise ValueError("factorial report has an incomplete prefix manifest mapping")
    indexed = _indexed_report_rows(report)
    for (episode, cell), row in indexed.items():
        if int(row.get("model_seed", -1)) != model_seed:
            raise ValueError(f"row model_seed mismatch for episode {episode} {cell}")
        for metric in METRICS:
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"non-finite {metric} for episode {episode} {cell}")
        for field in PREFIX_HASH_FIELDS:
            _require_sha256(row[field], f"episode {episode} {cell} {field}")
        grounding = CELL_SPECS[cell][0]
        manifest_key = f"episode_{episode:03d}/{grounding}"
        manifest_sha = _require_sha256(
            manifests[manifest_key], f"prefix manifest {manifest_key}"
        )
        if manifest_sha != row["prefix_manifest_sha256"]:
            raise ValueError(f"prefix manifest row mismatch for {manifest_key}")
    return indexed


def validate_cross_report_contract(
    reports: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Fail closed unless all reports share one canonical paired-prefix panel."""

    if not reports:
        raise ValueError("at least one factorial report is required")
    indexed = [(path, report, _validate_report_contract(report)) for path, report in reports]
    canonical = [item for item in indexed if item[1]["run_role"] == "canonical"]
    if len(canonical) != 1:
        raise ValueError("exactly one canonical factorial report is required")
    canonical_path, canonical_report, canonical_rows = canonical[0]
    canonical_sha = _sha256_file(canonical_path)
    canonical_seed = int(canonical_report["model_seed"])
    expected_provenance = canonical_report["provenance"]
    expected_manifests = canonical_report["prefix_bundle_manifests"]
    predictor_training_hashes = {
        grounding: canonical_report["predictor_matching"][grounding][
            "training_config_sha256"
        ]
        for grounding in ("nominal", "fcpg")
    }
    expected_classification = canonical_report["classification"]
    prefix_bundle_hashes: dict[str, str] = {}
    for episode in range(int(canonical_report["episodes"])):
        for grounding, cells in {
            "nominal": ("f0_l0", "f0_l1"),
            "fcpg": ("f1_l0", "f1_l1"),
        }.items():
            values = {
                str(canonical_rows[(episode, cell)]["prefix_bundle_sha256"])
                for cell in cells
            }
            if len(values) != 1:
                raise ValueError(
                    f"canonical {grounding} bundle mismatch for episode {episode}"
                )
            prefix_bundle_hashes[f"episode_{episode:03d}/{grounding}"] = values.pop()
    expected_seeds = sorted(int(value) for value in expected_provenance["model_seeds"])
    report_seeds = sorted(int(report["model_seed"]) for _, report, _ in indexed)
    if report_seeds != expected_seeds:
        raise ValueError(
            f"source reports must cover configured model_seeds {expected_seeds}, got {report_seeds}"
        )

    for path, report, rows in indexed:
        if report["provenance"] != expected_provenance:
            raise ValueError(f"report provenance mismatch: {path}")
        if report["classification"] != expected_classification:
            raise ValueError(f"report classification mismatch: {path}")
        if report["prefix_bundle_manifests"] != expected_manifests:
            raise ValueError(f"prefix bundle manifest mismatch: {path}")
        if int(report.get("canonical_model_seed", -1)) != canonical_seed:
            raise ValueError(f"canonical model seed mismatch: {path}")
        for grounding in ("nominal", "fcpg"):
            actual = report["predictor_matching"][grounding]["training_config_sha256"]
            if actual != predictor_training_hashes[grounding]:
                raise ValueError(f"{grounding} predictor config mismatch: {path}")
        if report["run_role"] == "canonical":
            if report.get("canonical_baseline_source") is not None:
                raise ValueError("canonical report cannot cite another baseline report")
        else:
            source = report.get("canonical_baseline_source")
            if not isinstance(source, dict) or source.get("sha256") != canonical_sha:
                raise ValueError(f"canonical baseline source hash mismatch: {path}")
            if Path(str(source.get("path", ""))).resolve() != canonical_path.resolve():
                raise ValueError(f"canonical baseline source path mismatch: {path}")

        for episode in range(int(report["episodes"])):
            for cell in CELL_SPECS:
                current = rows[(episode, cell)]
                reference = canonical_rows[(episode, cell)]
                for field in PREFIX_HASH_FIELDS:
                    if str(current[field]) != str(reference[field]):
                        raise ValueError(
                            f"cross-seed prefix mismatch for episode {episode} {cell} {field}"
                        )
            for cell in CANONICAL_BASELINE_CELLS:
                if _canonical_json(_without_model_seed(rows[(episode, cell)])) != _canonical_json(
                    _without_model_seed(canonical_rows[(episode, cell)])
                ):
                    raise ValueError(
                        f"canonical L0 row changed for episode {episode} {cell}"
                    )

    return {
        "canonical_report": str(canonical_path),
        "canonical_report_sha256": canonical_sha,
        "prefix_bundle_sha256": prefix_bundle_hashes,
        "prefix_bundle_manifests": expected_manifests,
    }


def _effect_direction(values: list[float], *, tolerance: float = 1.0e-12) -> str:
    """Classify whether an episode-level effect agrees across model seeds."""

    if not values:
        raise ValueError("episode-level effects require at least one model seed")
    signs = [
        0 if abs(value) <= tolerance else (1 if value > 0.0 else -1)
        for value in values
    ]
    if all(sign == 0 for sign in signs):
        return "zero_all_seeds"
    if all(sign > 0 for sign in signs):
        return "positive_all_seeds"
    if all(sign < 0 for sign in signs):
        return "negative_all_seeds"
    return "mixed_across_seeds"


def episode_level_lewm_effects(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit Full minus FCPG-only effects for each shared physical episode.

    The crossed factorial reports population averages.  This audit keeps the
    physical episode fixed and asks whether the LeWM direction repeats across
    model seeds, which distinguishes stable conditional effects from optimizer
    noise without treating model seeds as new physical episodes.
    """

    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    model_seeds: set[int] = set()
    episodes: set[int] = set()
    for row in rows:
        model_seed = int(row["model_seed"])
        episode = int(row["physical_episode"])
        method = str(row["method"])
        key = (model_seed, episode, method)
        if key in indexed:
            raise ValueError(f"duplicate factorial row for {key}")
        indexed[key] = row
        model_seeds.add(model_seed)
        episodes.add(episode)

    ordered_seeds = sorted(model_seeds)
    episode_rows: list[dict[str, Any]] = []
    direction_counts = {
        metric: {
            direction: 0
            for direction in (
                "positive_all_seeds",
                "negative_all_seeds",
                "zero_all_seeds",
                "mixed_across_seeds",
            )
        }
        for metric in (
            "sustained_success",
            "final_success",
            "post_drop_success_fraction",
            "post_drop_maximum_success_streak",
            "reward_sum",
        )
    }
    for episode in sorted(episodes):
        per_seed: dict[str, Any] = {}
        effects_by_metric: dict[str, list[float]] = {
            metric: [] for metric in direction_counts
        }
        drop_starts: set[int] = set()
        for model_seed in ordered_seeds:
            try:
                baseline = indexed[(model_seed, episode, "f1_l0")]
                full = indexed[(model_seed, episode, "f1_l1")]
            except KeyError as error:
                raise ValueError(
                    "episode audit requires F1L0 and F1L1 for every seed/episode"
                ) from error
            drop_starts.update((int(baseline["drop_start"]), int(full["drop_start"])))
            effects = {
                metric: float(full[metric]) - float(baseline[metric])
                for metric in effects_by_metric
            }
            for metric, value in effects.items():
                effects_by_metric[metric].append(value)
            per_seed[str(model_seed)] = effects
        if len(drop_starts) != 1:
            raise ValueError("model seeds do not share the episode dropout onset")

        directions = {
            metric: _effect_direction(values)
            for metric, values in effects_by_metric.items()
        }
        for metric, direction in directions.items():
            direction_counts[metric][direction] += 1
        episode_rows.append(
            {
                "physical_episode": episode,
                "drop_start": next(iter(drop_starts)),
                "directions": directions,
                "mean_effects": {
                    metric: float(sum(values) / len(values))
                    for metric, values in effects_by_metric.items()
                },
                "per_model_seed_effects": per_seed,
            }
        )

    return {
        "comparison": "f1_l1_minus_f1_l0",
        "model_seeds": ordered_seeds,
        "physical_episode_count": len(episodes),
        "direction_counts": direction_counts,
        "episodes": episode_rows,
    }


def _read_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise TypeError(f"{path} is not a factorial report")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=99600)
    args = parser.parse_args()

    report_paths = [path.expanduser().resolve() for path in args.report]
    reports = [_read_report(path) for path in report_paths]
    pairing_contract = validate_cross_report_contract(list(zip(report_paths, reports)))
    seeds = [int(report["model_seed"]) for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("model_seed reports must be unique")
    episode_counts = {int(report["episodes"]) for report in reports}
    if len(episode_counts) != 1:
        raise ValueError("all model seeds must use the same episode count")

    all_rows: list[dict[str, Any]] = []
    physical_identity: dict[tuple[int, str], tuple[str, ...]] = {}
    for report in reports:
        rows = report["rows"]
        _validate_physical_pairing(rows)
        model_seed = int(report["model_seed"])
        for row in rows:
            copied = dict(row)
            copied["model_seed"] = model_seed
            key = (int(copied["physical_episode"]), str(copied["method"]))
            identity = tuple(
                str(copied[field])
                for field in (
                    "reset_digest",
                    "camera_sha256",
                    "schedule_sha256",
                    "drop_onset_state_sha256",
                )
            )
            prior = physical_identity.setdefault(key, identity)
            if identity != prior:
                raise RuntimeError(f"model seeds do not share physical identity for {key}")
            all_rows.append(copied)

    output = {
        "classification": "paired_prefix_crossed_seed_development",
        "model_seeds": sorted(seeds),
        "physical_episode_count": next(iter(episode_counts)),
        "cell_semantics": {
            cell: {"grounding": grounding, "dropout_state": method}
            for cell, (grounding, method) in CELL_SPECS.items()
        },
        "factorial": {
            metric: crossed_seed_factorial_effects(
                all_rows,
                metric=metric,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + offset,
            )
            for offset, metric in enumerate(METRICS)
        },
        "episode_level_lewm_effects": episode_level_lewm_effects(all_rows),
        "pairing_contract": pairing_contract,
        "source_reports": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in report_paths
        ],
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
