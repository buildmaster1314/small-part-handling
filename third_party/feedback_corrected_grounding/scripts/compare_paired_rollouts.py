#!/usr/bin/env python3
"""Compare deterministic rollout CSVs with paired statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use MODEL=CSV_PATH")
    model, path = value.split("=", 1)
    if not model.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must use MODEL=CSV_PATH")
    return model.strip(), Path(path.strip())


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    missing = [key for key in PAIR_KEYS if key not in row]
    if missing:
        raise KeyError(f"rollout CSV is missing deterministic pair keys: {missing}")
    return tuple(row[key] for key in PAIR_KEYS)


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left_only = int(np.sum((left == 1.0) & (right == 0.0)))
    right_only = int(np.sum((left == 0.0) & (right == 1.0)))
    discordant = left_only + right_only
    smaller = min(left_only, right_only)
    if discordant:
        probability = min(
            1.0,
            2.0 * sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant),
        )
    else:
        probability = 1.0
    return {
        "left_only_success": left_only,
        "right_only_success": right_only,
        "discordant": discordant,
        "p_value_two_sided": probability,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_run, metavar="MODEL=CSV")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/bc_rollout/paired_comparison")
    args = parser.parse_args()

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for model, path in args.run:
        grouped_paths[model].append(path)
    if args.baseline not in grouped_paths:
        parser.error(f"baseline {args.baseline!r} has no --run entry")

    rows_by_model: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
    for model, paths in grouped_paths.items():
        keyed_rows: dict[tuple[str, ...], dict[str, str]] = {}
        for path in paths:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = row_key(row)
                    if key in keyed_rows:
                        raise ValueError(f"duplicate paired episode for {model}: {key}")
                    keyed_rows[key] = row
        rows_by_model[model] = keyed_rows

    baseline_keys = set(rows_by_model[args.baseline])
    for model, keyed_rows in rows_by_model.items():
        keys = set(keyed_rows)
        if keys != baseline_keys:
            raise ValueError(
                f"pair-key mismatch for {model}: missing={len(baseline_keys - keys)} extra={len(keys - baseline_keys)}"
            )
    ordered_keys = sorted(baseline_keys)

    summary: dict[str, object] = {
        "baseline": args.baseline,
        "episodes": len(ordered_keys),
        "pair_keys": list(PAIR_KEYS),
        "models": {},
        "comparisons": {},
    }
    success: dict[str, np.ndarray] = {}
    for model, keyed_rows in rows_by_model.items():
        rows = [keyed_rows[key] for key in ordered_keys]
        values = np.asarray([float(row["success"]) for row in rows], dtype=np.float64)
        success[model] = values
        summary["models"][model] = {
            "success_rate": float(values.mean()),
            "reward_sum_mean": float(np.mean([float(row["reward_sum"]) for row in rows])),
            "pixel_error_mean": float(np.mean([float(row["pixel_error_mean"]) for row in rows])),
            "corrected_pixel_error_mean": float(
                np.mean([float(row["corrected_pixel_error_mean"]) for row in rows])
            ),
        }

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(
        0,
        len(ordered_keys),
        size=(args.bootstrap_samples, len(ordered_keys)),
    )
    baseline_success = success[args.baseline]
    for model, values in success.items():
        if model == args.baseline:
            continue
        paired_differences = (values[sample_indices] - baseline_success[sample_indices]).mean(axis=1)
        summary["comparisons"][f"{model}_vs_{args.baseline}"] = {
            "success_rate_difference": float(values.mean() - baseline_success.mean()),
            "bootstrap_95_ci": [
                float(np.quantile(paired_differences, 0.025)),
                float(np.quantile(paired_differences, 0.975)),
            ],
            "mcnemar_exact": exact_mcnemar(baseline_success, values),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paired_rollout_summary.json"
    markdown_path = output_dir / "paired_rollout_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Paired Rollout Comparison", "", f"Episodes: {len(ordered_keys)}", "", "## Models", ""]
    lines.append("| model | success | reward | pixel error | corrected error |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for model, metrics in summary["models"].items():
        lines.append(
            f"| {model} | {metrics['success_rate']:.4f} | {metrics['reward_sum_mean']:.3f} | "
            f"{metrics['pixel_error_mean']:.3f} | {metrics['corrected_pixel_error_mean']:.3f} |"
        )
    lines.extend(["", "## Paired Comparisons", ""])
    for name, metrics in summary["comparisons"].items():
        ci = metrics["bootstrap_95_ci"]
        test = metrics["mcnemar_exact"]
        lines.append(
            f"- `{name}`: delta={metrics['success_rate_difference']:.4f}, "
            f"bootstrap95=[{ci[0]:.4f}, {ci[1]:.4f}], "
            f"McNemar p={test['p_value_two_sided']:.6f} "
            f"(baseline-only={test['left_only_success']}, model-only={test['right_only_success']})."
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
