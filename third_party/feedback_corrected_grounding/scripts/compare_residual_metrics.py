#!/usr/bin/env python3
"""Compare residual predictor metrics JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metric(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    best = data.get("best") or data.get("final")
    final = data.get("final")
    return {"path": str(path), "best": best, "final": final}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+")
    args = parser.parse_args()
    rows = [load_metric(path) for path in args.metrics]
    print("model,path,best_epoch,best_e_before,best_e_after,best_ratio,final_ratio")
    for row in rows:
        path = Path(row["path"])
        model = path.name.replace("_metrics.json", "")
        best = row["best"]
        final = row["final"]
        print(
            f"{model},{path},{best['epoch']},"
            f"{best['e_before']:.4f},{best['e_after']:.4f},{best['ratio']:.4f},"
            f"{final['ratio']:.4f}"
        )
    if len(rows) >= 2:
        best_sorted = sorted(rows, key=lambda r: r["best"]["ratio"])
        winner = Path(best_sorted[0]["path"]).name.replace("_metrics.json", "")
        print(f"best_model: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
