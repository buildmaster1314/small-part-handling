#!/usr/bin/env python3
"""Aggregate offline and paired rollout metrics across action-policy seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PAIR_KEYS = (
    "task",
    "task_index",
    "env_seed",
    "actual_camera_translation_x_cm",
    "actual_camera_translation_y_cm",
    "actual_camera_translation_z_cm",
    "actual_camera_rotation_axis_x",
    "actual_camera_rotation_axis_y",
    "actual_camera_rotation_axis_z",
    "actual_camera_rotation_deg",
)


@dataclass(frozen=True)
class SeedPaths:
    seed: int
    geoprop_offline: Path
    fcpg_offline: Path
    nominal_rollout: Path
    fcpg_rollout: Path


def parse_seed_result(value: str) -> SeedPaths:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--seed-result must use SEED=GEOPROP_JSON,FCPG_JSON,NOMINAL_CSV,FCPG_CSV"
        )
    seed_text, paths_text = value.split("=", 1)
    paths = [Path(item.strip()) for item in paths_text.split(",")]
    if len(paths) != 4 or any(not str(path) for path in paths):
        raise argparse.ArgumentTypeError(
            "--seed-result must contain exactly four comma-separated paths"
        )
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("policy seed must be an integer") from error
    return SeedPaths(seed, *paths)


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    missing = [key for key in PAIR_KEYS if key not in row]
    if missing:
        raise KeyError(f"rollout CSV is missing deterministic pair keys: {missing}")
    return tuple(row[key] for key in PAIR_KEYS)


def _read_rollout(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = _row_key(row)
            if key in rows:
                raise ValueError(f"duplicate paired episode in {path}: {key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"rollout CSV is empty: {path}")
    return rows


def _exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left_only = int(np.sum((left == 1.0) & (right == 0.0)))
    right_only = int(np.sum((left == 0.0) & (right == 1.0)))
    discordant = left_only + right_only
    smaller = min(left_only, right_only)
    probability = 1.0
    if discordant:
        probability = min(
            1.0,
            2.0 * sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant),
        )
    return {
        "baseline_only_success": left_only,
        "fcpg_only_success": right_only,
        "discordant": discordant,
        "p_value_two_sided": probability,
    }


def hierarchical_bootstrap_ci(
    paired_differences: list[np.ndarray],
    samples: int,
    seed: int,
) -> list[float]:
    if not paired_differences:
        raise ValueError("at least one policy seed is required")
    episode_counts = {len(values) for values in paired_differences}
    if len(episode_counts) != 1:
        raise ValueError("all policy seeds must have the same number of paired episodes")
    values = np.stack(paired_differences)
    num_seeds, num_episodes = values.shape
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    batch_size = min(5_000, samples)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        seed_indices = rng.integers(0, num_seeds, size=(count, num_seeds))
        episode_indices = rng.integers(
            0,
            num_episodes,
            size=(count, num_seeds, num_episodes),
        )
        sampled = values[seed_indices[:, :, None], episode_indices]
        estimates.append(sampled.mean(axis=(1, 2)))
    distribution = np.concatenate(estimates)
    return [
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    ]


def summarize_seed_results(
    seed_paths: list[SeedPaths],
    bootstrap_samples: int = 100_000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    if not seed_paths:
        raise ValueError("at least one --seed-result is required")
    seeds = [item.seed for item in seed_paths]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate policy seeds: {seeds}")

    per_seed: list[dict[str, object]] = []
    paired_differences: list[np.ndarray] = []
    for item in sorted(seed_paths, key=lambda value: value.seed):
        geoprop = json.loads(item.geoprop_offline.read_text(encoding="utf-8"))
        fcpg = json.loads(item.fcpg_offline.read_text(encoding="utf-8"))
        if geoprop["evaluated_sequences"] != fcpg["evaluated_sequences"]:
            raise ValueError(f"offline sequence mismatch for seed {item.seed}")

        nominal_rows = _read_rollout(item.nominal_rollout)
        fcpg_rows = _read_rollout(item.fcpg_rollout)
        if set(nominal_rows) != set(fcpg_rows):
            raise ValueError(f"paired rollout keys differ for seed {item.seed}")
        keys = sorted(nominal_rows)
        nominal_success = np.asarray(
            [float(nominal_rows[key]["success"]) for key in keys], dtype=np.float64
        )
        fcpg_success = np.asarray(
            [float(fcpg_rows[key]["success"]) for key in keys], dtype=np.float64
        )
        differences = fcpg_success - nominal_success
        paired_differences.append(differences)
        per_seed.append(
            {
                "seed": item.seed,
                "offline": {
                    "geoprop_chunk_mse": float(geoprop["chunk_mse"]),
                    "fcpg_chunk_mse": float(fcpg["chunk_mse"]),
                    "geoprop_first_action_mse": float(geoprop["first_action_mse"]),
                    "fcpg_first_action_mse": float(fcpg["first_action_mse"]),
                    "geoprop_gripper_accuracy": float(geoprop["gripper_accuracy"]),
                    "fcpg_gripper_accuracy": float(fcpg["gripper_accuracy"]),
                },
                "rollout": {
                    "episodes": len(keys),
                    "nominal_success_rate": float(nominal_success.mean()),
                    "fcpg_success_rate": float(fcpg_success.mean()),
                    "success_rate_difference": float(differences.mean()),
                    "nominal_pixel_error_mean": float(
                        np.mean([float(nominal_rows[key]["pixel_error_mean"]) for key in keys])
                    ),
                    "fcpg_corrected_pixel_error_mean": float(
                        np.mean([float(fcpg_rows[key]["corrected_pixel_error_mean"]) for key in keys])
                    ),
                    "mcnemar_exact": _exact_mcnemar(nominal_success, fcpg_success),
                },
            }
        )

    def values(section: str, metric: str) -> list[float]:
        return [float(item[section][metric]) for item in per_seed]

    offline_metrics: dict[str, dict[str, float]] = {}
    for metric in (
        "geoprop_chunk_mse",
        "fcpg_chunk_mse",
        "geoprop_first_action_mse",
        "fcpg_first_action_mse",
        "geoprop_gripper_accuracy",
        "fcpg_gripper_accuracy",
    ):
        metric_values = values("offline", metric)
        offline_metrics[metric] = {
            "mean": float(np.mean(metric_values)),
            "sample_std": _sample_std(metric_values),
        }

    rollout_metrics: dict[str, object] = {}
    for metric in (
        "nominal_success_rate",
        "fcpg_success_rate",
        "success_rate_difference",
        "nominal_pixel_error_mean",
        "fcpg_corrected_pixel_error_mean",
    ):
        metric_values = values("rollout", metric)
        rollout_metrics[metric] = {
            "mean": float(np.mean(metric_values)),
            "sample_std": _sample_std(metric_values),
        }
    rollout_metrics["hierarchical_bootstrap_95_ci"] = hierarchical_bootstrap_ci(
        paired_differences,
        bootstrap_samples,
        bootstrap_seed,
    )

    return {
        "policy_seeds": sorted(seeds),
        "num_policy_seeds": len(seeds),
        "per_seed": per_seed,
        "aggregate": {
            "offline": offline_metrics,
            "rollout": rollout_metrics,
        },
    }


def render_markdown(
    summary: dict[str, object],
    baseline_label: str = "GeoProp",
    model_label: str = "FCPG",
) -> str:
    lines = [
        "# Action-Chunk Multi-Seed Summary",
        "",
        f"Policy seeds: {summary['policy_seeds']}",
        "",
        "## Per-Seed Results",
        "",
        f"| seed | {baseline_label} chunk MSE | {model_label} chunk MSE | "
        f"{baseline_label} success | {model_label} success | delta | McNemar p |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["per_seed"]:
        offline = item["offline"]
        rollout = item["rollout"]
        lines.append(
            f"| {item['seed']} | {offline['geoprop_chunk_mse']:.6f} | {offline['fcpg_chunk_mse']:.6f} | "
            f"{rollout['nominal_success_rate']:.4f} | {rollout['fcpg_success_rate']:.4f} | "
            f"{rollout['success_rate_difference']:+.4f} | "
            f"{rollout['mcnemar_exact']['p_value_two_sided']:.6f} |"
        )

    aggregate = summary["aggregate"]
    offline = aggregate["offline"]
    rollout = aggregate["rollout"]
    ci = rollout["hierarchical_bootstrap_95_ci"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- {baseline_label} chunk MSE: {offline['geoprop_chunk_mse']['mean']:.6f} +/- "
            f"{offline['geoprop_chunk_mse']['sample_std']:.6f}.",
            f"- {model_label} chunk MSE: {offline['fcpg_chunk_mse']['mean']:.6f} +/- "
            f"{offline['fcpg_chunk_mse']['sample_std']:.6f}.",
            f"- {baseline_label} rollout success: {rollout['nominal_success_rate']['mean']:.4f} +/- "
            f"{rollout['nominal_success_rate']['sample_std']:.4f}.",
            f"- {model_label} rollout success: {rollout['fcpg_success_rate']['mean']:.4f} +/- "
            f"{rollout['fcpg_success_rate']['sample_std']:.4f}.",
            f"- Paired success delta: {rollout['success_rate_difference']['mean']:+.4f} +/- "
            f"{rollout['success_rate_difference']['sample_std']:.4f}; hierarchical bootstrap 95% CI "
            f"[{ci[0]:.4f}, {ci[1]:.4f}].",
            "",
            "The confidence interval resamples policy seeds first and paired episodes within each sampled seed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-result",
        action="append",
        required=True,
        type=parse_seed_result,
        metavar="SEED=GEOPROP_JSON,FCPG_JSON,NOMINAL_CSV,FCPG_CSV",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--baseline-label", default="GeoProp")
    parser.add_argument("--model-label", default="FCPG")
    parser.add_argument("--output-dir", default="outputs/action_chunk/multi_seed_summary")
    args = parser.parse_args()

    summary = summarize_seed_results(
        args.seed_result,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["labels"] = {
        "baseline": args.baseline_label,
        "model": args.model_label,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "action_chunk_seed_summary.json"
    markdown_path = output_dir / "action_chunk_seed_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(
        render_markdown(summary, args.baseline_label, args.model_label),
        encoding="utf-8",
    )
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
