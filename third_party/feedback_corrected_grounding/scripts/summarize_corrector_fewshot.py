#!/usr/bin/env python3
"""Summarize target-task corrector adaptation over episode budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_evaluation(value: str) -> tuple[int, Path]:
    episodes, separator, path = value.partition("=")
    if not separator or not episodes or not path:
        raise argparse.ArgumentTypeError("evaluation must use EPISODES=PATH")
    try:
        count = int(episodes)
    except ValueError as error:
        raise argparse.ArgumentTypeError("EPISODES must be an integer") from error
    if count < 0:
        raise argparse.ArgumentTypeError("EPISODES must be non-negative")
    return count, Path(path)


def summarize_fewshot(
    evaluations: list[tuple[int, Path]],
    ratio_threshold: float,
) -> dict[str, object]:
    if not evaluations:
        raise ValueError("at least one evaluation is required")
    if ratio_threshold <= 0:
        raise ValueError("ratio_threshold must be positive")
    counts = [count for count, _ in evaluations]
    if len(set(counts)) != len(counts):
        raise ValueError("episode budgets must be unique")
    rows = []
    for episodes, path in sorted(evaluations):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["results"]["normal"]
        ratio = float(metrics["ratio"])
        rows.append(
            {
                "episodes": episodes,
                "evaluation": str(path),
                "validation_frames": int(payload["validation_frames"]),
                "e_before_mean_px": float(metrics["e_before_mean_px"]),
                "e_after_mean_px": float(metrics["e_after_mean_px"]),
                "e_after_p90_px": float(metrics["e_after_p90_px"]),
                "ratio": ratio,
                "pass": ratio <= ratio_threshold,
            }
        )
    return {
        "schema_version": 1,
        "protocol": "target_task_fewshot_final_checkpoint",
        "ratio_threshold": float(ratio_threshold),
        "points": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        action="append",
        type=parse_evaluation,
        required=True,
        help="Repeat EPISODES=PATH for every adaptation budget.",
    )
    parser.add_argument("--ratio-threshold", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = summarize_fewshot(args.evaluation, args.ratio_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("| episodes | E_before | E_after | ratio | pass |")
    print("|---:|---:|---:|---:|:---:|")
    for row in summary["points"]:
        print(
            f"| {row['episodes']} | {row['e_before_mean_px']:.3f} | "
            f"{row['e_after_mean_px']:.3f} | {row['ratio']:.3f} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
