#!/usr/bin/env python3
"""Compare minimal BC policy metric JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metric(path: str | Path) -> dict[str, object]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    best = data.get("best", {})
    final = data.get("final", {})
    return {
        "path": str(p),
        "run": p.parent.name,
        "model": p.name.replace("_metrics.json", ""),
        "best_epoch": best.get("epoch"),
        "best_mse": best.get("mse"),
        "best_mae": best.get("mae"),
        "final_mse": final.get("mse"),
        "final_mae": final.get("mae"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = [load_metric(path) for path in args.metrics]
    lines = [
        "# BC Policy Metric Comparison",
        "",
        "| run | model | best epoch | best MSE | best MAE | final MSE | final MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['model']} | {row['best_epoch']} | "
            f"{float(row['best_mse']):.6f} | {float(row['best_mae']):.6f} | "
            f"{float(row['final_mse']):.6f} | {float(row['final_mae']):.6f} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
