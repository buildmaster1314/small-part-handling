#!/usr/bin/env python3
"""Run a small residual-prediction benchmark across directions and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from fcpg.observations import OBSERVATION_MODES


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    return value.replace("+", "p").replace("-", "m").replace(",", "_")


def load_best(metrics_path: Path) -> dict:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return data.get("best") or data["final"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/residual_benchmark")
    parser.add_argument("--directions", default="x-,y-")
    parser.add_argument("--translations-cm", default="")
    parser.add_argument("--yaws-deg", default="")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--models", default="no-image,local-image")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-radius", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-obs", action="store_true")
    parser.add_argument("--obs-mode", choices=OBSERVATION_MODES, default="full")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    split_cases: list[tuple[str, str]] = []
    split_cases.extend(("direction", direction) for direction in parse_list(args.directions))
    split_cases.extend(("translation_cm", value) for value in parse_list(args.translations_cm))
    split_cases.extend(("yaw_deg", value) for value in parse_list(args.yaws_deg))
    if not split_cases:
        split_cases.append(("random", "val"))
    seeds = [int(seed) for seed in parse_list(args.seeds)]
    models = parse_list(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for split_type, heldout in split_cases:
        for seed in seeds:
            for model in models:
                run_name = f"{split_type}_{safe_name(heldout)}_seed_{seed}_{model}"
                run_dir = output_dir / run_name
                metrics_path = run_dir / f"{model}_metrics.json"
                if not (args.skip_existing and metrics_path.exists()):
                    cmd = [
                        sys.executable,
                        "scripts/train_residual_predictor.py",
                        "--data",
                        args.data,
                        "--model",
                        model,
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--patch-radius",
                        str(args.patch_radius),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                        "--obs-mode",
                        args.obs_mode,
                        "--output-dir",
                        str(run_dir),
                    ]
                    if split_type == "direction":
                        cmd.extend(["--heldout-directions", heldout])
                    elif split_type == "translation_cm":
                        cmd.extend(["--heldout-directions", "", "--heldout-translations-cm", heldout])
                    elif split_type == "yaw_deg":
                        cmd.extend(["--heldout-directions", "", "--heldout-yaws-deg", heldout])
                    else:
                        cmd.extend(["--heldout-directions", ""])
                    if args.no_obs:
                        cmd.append("--no-obs")
                    print("RUN", " ".join(cmd), flush=True)
                    subprocess.run(cmd, check=True)
                best = load_best(metrics_path)
                rows.append(
                    {
                        "split_type": split_type,
                        "heldout": heldout,
                        "seed": seed,
                        "model": model,
                        "best_epoch": int(best["epoch"]),
                        "e_before": float(best["e_before"]),
                        "e_after": float(best["e_after"]),
                        "ratio": float(best["ratio"]),
                        "e_after_p90": float(best["e_after_p90"]),
                        "metrics_path": str(metrics_path),
                    }
                )

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "summary.md"
    lines = ["# Residual Benchmark Summary", "", "| split_type | heldout | seed | model | best_epoch | E_before | E_after | ratio |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['split_type']} | {row['heldout']} | {row['seed']} | {row['model']} | {row['best_epoch']} | "
            f"{row['e_before']:.4f} | {row['e_after']:.4f} | {row['ratio']:.4f} |"
        )

    aggregate: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        aggregate.setdefault((str(row["split_type"]), str(row["heldout"]), str(row["model"])), []).append(float(row["ratio"]))
    lines.extend(["", "## Mean Ratio", "", "| split_type | heldout | model | mean_ratio |", "|---|---|---|---:|"])
    for (split_type, heldout, model), values in sorted(aggregate.items()):
        lines.append(f"| {split_type} | {heldout} | {model} | {sum(values) / len(values):.4f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    print("summary:")
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
