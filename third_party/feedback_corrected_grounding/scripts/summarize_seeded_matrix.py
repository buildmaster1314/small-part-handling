#!/usr/bin/env python3
"""Aggregate rollout matrices across training seeds without flattening repeats."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def parse_seeded_policy(label: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_s(\d+)", label)
    if not match:
        raise ValueError(f"policy label must end in _s<seed>: {label!r}")
    return match.group(1), int(match.group(2))


def parse_checkpoint_seed(label: str) -> int:
    match = re.search(r"(\d+)$", label)
    if not match:
        raise ValueError(f"checkpoint label must end in a seed number: {label!r}")
    return int(match.group(1))


def normalize_rows(payload: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    auc_rows = []
    if "native_policy" in payload:
        for raw in payload["native_policy"]:
            method, seed = parse_seeded_policy(str(raw["policy"]))
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "task": raw["task"],
                    "perturbation": raw["perturbation"],
                    "severity": float(raw["severity_deg"]),
                    "success": float(raw["success"]),
                    "pixel_error": float(raw["grounding_error"]),
                }
            )
        for raw in payload["auc"]:
            method, seed = parse_seeded_policy(str(raw["policy"]))
            auc_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "task": raw["task"],
                    "auc": float(raw["native_success_auc"]),
                }
            )
    elif "rows" in payload:
        for raw in payload["rows"]:
            rows.append(
                {
                    "method": raw["method"],
                    "seed": parse_checkpoint_seed(str(raw["checkpoint"])),
                    "task": raw["task"],
                    "perturbation": raw["perturbation"],
                    "severity": float(raw["severity"]),
                    "success": float(raw["success_rate"]),
                    "pixel_error": float(raw["corrected_pixel_error_mean"]),
                }
            )
        for raw in payload["auc"]:
            auc_rows.append(
                {
                    "method": raw["method"],
                    "seed": parse_checkpoint_seed(str(raw["checkpoint"])),
                    "task": raw["task"],
                    "auc": float(raw["success_auc"]),
                }
            )
    else:
        raise ValueError("unrecognized matrix summary schema")
    return rows, auc_rows


def aggregate(
    rows: list[dict[str, object]],
    auc_rows: list[dict[str, object]],
    baseline: str,
) -> dict[str, object]:
    grouped: dict[tuple[str, str, float, str], list[dict[str, object]]] = defaultdict(list)
    by_seed: dict[tuple[str, str, float, int, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["task"]),
            str(row["perturbation"]),
            float(row["severity"]),
            str(row["method"]),
        )
        grouped[key].append(row)
        by_seed[(key[0], key[1], key[2], int(row["seed"]), key[3])] = row

    summaries = []
    differences = []
    methods = sorted({str(row["method"]) for row in rows})
    for (task, perturbation, severity, method), method_rows in sorted(grouped.items()):
        successes = [float(row["success"]) for row in method_rows]
        errors = [float(row["pixel_error"]) for row in method_rows]
        success_mean, success_std = mean_std(successes)
        error_mean, error_std = mean_std(errors)
        seeds = sorted(int(row["seed"]) for row in method_rows)
        summaries.append(
            {
                "task": task,
                "perturbation": perturbation,
                "severity": severity,
                "method": method,
                "seeds": seeds,
                "success_mean": success_mean,
                "success_std": success_std,
                "pixel_error_mean": error_mean,
                "pixel_error_std": error_std,
            }
        )
        if method == baseline:
            continue
        seed_differences = []
        for seed in seeds:
            baseline_key = (task, perturbation, severity, seed, baseline)
            if baseline_key not in by_seed:
                raise ValueError(
                    f"missing baseline={baseline!r} for task={task} severity={severity} seed={seed}"
                )
            seed_differences.append(
                float(by_seed[(task, perturbation, severity, seed, method)]["success"])
                - float(by_seed[baseline_key]["success"])
            )
        difference_mean, difference_std = mean_std(seed_differences)
        differences.append(
            {
                "task": task,
                "perturbation": perturbation,
                "severity": severity,
                "baseline": baseline,
                "candidate": method,
                "seed_differences": seed_differences,
                "difference_mean": difference_mean,
                "difference_std": difference_std,
            }
        )

    auc_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in auc_rows:
        auc_grouped[(str(row["task"]), str(row["method"]))].append(float(row["auc"]))
    auc_summary = []
    for (task, method), values in sorted(auc_grouped.items()):
        value_mean, value_std = mean_std(values)
        auc_summary.append(
            {
                "task": task,
                "method": method,
                "auc_by_seed": values,
                "auc_mean": value_mean,
                "auc_std": value_std,
            }
        )
    return {
        "baseline": baseline,
        "methods": methods,
        "summary": summaries,
        "paired_seed_differences": differences,
        "auc": auc_summary,
    }


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Seed-Aggregated Rollout Matrix",
        "",
        "Values are mean +/- sample standard deviation across training seeds.",
        "Episode-level paired bootstrap intervals remain in the source matrix.",
        "",
        "| task | perturbation | severity | method | success | pixel error |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['task']} | {row['perturbation']} | {float(row['severity']):g} | "
            f"{row['method']} | {float(row['success_mean']):.4f} +/- "
            f"{float(row['success_std']):.4f} | {float(row['pixel_error_mean']):.3f} +/- "
            f"{float(row['pixel_error_std']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Same-seed success differences",
            "",
            "| task | perturbation | severity | comparison | delta | per-seed |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in payload["paired_seed_differences"]:
        per_seed = ", ".join(f"{float(value):+.4f}" for value in row["seed_differences"])
        lines.append(
            f"| {row['task']} | {row['perturbation']} | {float(row['severity']):g} | "
            f"{row['candidate']} - {row['baseline']} | {float(row['difference_mean']):+.4f} +/- "
            f"{float(row['difference_std']):.4f} | {per_seed} |"
        )
    lines.extend(
        [
            "",
            "## Success AUC",
            "",
            "| task | method | normalized AUC | per-seed |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in payload["auc"]:
        per_seed = ", ".join(f"{float(value):.4f}" for value in row["auc_by_seed"])
        lines.append(
            f"| {row['task']} | {row['method']} | {float(row['auc_mean']):.4f} +/- "
            f"{float(row['auc_std']):.4f} | {per_seed} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.matrix)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows, auc_rows = normalize_rows(payload)
    aggregated = aggregate(rows, auc_rows, args.baseline)
    aggregated["source_matrix"] = str(source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed_aggregate.json").write_text(
        json.dumps(aggregated, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(aggregated)
    (output_dir / "seed_aggregate.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
