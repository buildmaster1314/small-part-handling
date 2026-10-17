#!/usr/bin/env python3
"""Summarize strict leave-one-task-out corrector evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_evaluation(value: str) -> tuple[str, Path]:
    task, separator, path = value.partition("=")
    if not separator or not task or not path:
        raise argparse.ArgumentTypeError("evaluation must use TASK=PATH")
    return task, Path(path)


def summarize_evaluations(
    evaluations: list[tuple[str, Path]],
    ratio_threshold: float,
) -> dict[str, object]:
    if not evaluations:
        raise ValueError("at least one evaluation is required")
    if ratio_threshold <= 0:
        raise ValueError("ratio_threshold must be positive")
    rows = []
    for task, path in evaluations:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["results"]["normal"]
        ratio = float(metrics["ratio"])
        rows.append(
            {
                "task": task,
                "evaluation": str(path),
                "validation_frames": int(payload["validation_frames"]),
                "e_before_mean_px": float(metrics["e_before_mean_px"]),
                "e_after_mean_px": float(metrics["e_after_mean_px"]),
                "e_after_p90_px": float(metrics["e_after_p90_px"]),
                "ratio": ratio,
                "pass": ratio <= ratio_threshold,
            }
        )
    ratios = np.asarray([row["ratio"] for row in rows], dtype=np.float64)
    after = np.asarray([row["e_after_mean_px"] for row in rows], dtype=np.float64)
    pass_count = sum(bool(row["pass"]) for row in rows)
    return {
        "schema_version": 1,
        "protocol": "strict_leave_one_task_out",
        "ratio_threshold": float(ratio_threshold),
        "tasks": rows,
        "aggregate": {
            "task_count": len(rows),
            "pass_count": pass_count,
            "all_pass": pass_count == len(rows),
            "ratio_mean": float(ratios.mean()),
            "ratio_std": float(ratios.std()),
            "e_after_mean_px": float(after.mean()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        action="append",
        type=parse_evaluation,
        required=True,
        help="Repeat TASK=PATH for every held-out task.",
    )
    parser.add_argument("--ratio-threshold", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = summarize_evaluations(args.evaluation, args.ratio_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("| held-out task | E_before | E_after | ratio | pass |")
    print("|---|---:|---:|---:|:---:|")
    for row in summary["tasks"]:
        print(
            f"| {row['task']} | {row['e_before_mean_px']:.3f} | "
            f"{row['e_after_mean_px']:.3f} | {row['ratio']:.3f} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )
    aggregate = summary["aggregate"]
    print(
        f"pass={aggregate['pass_count']}/{aggregate['task_count']} "
        f"ratio_mean={aggregate['ratio_mean']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
