#!/usr/bin/env python3
"""Compare MetaWorld rollout summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: str | Path) -> dict[str, object]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    name = p.parent.name
    return {"name": name, "path": str(p), **summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = [load_summary(path) for path in args.summaries]
    lines = [
        "# BC Rollout Summary Comparison",
        "",
        "| run | model | task | episodes | horizon | perturbation | success | reward mean | reward std | nominal pixel error | corrected pixel error |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        perturb = f"trans={row.get('translations_cm')}cm, dir={row.get('directions')}, yaw={row.get('yaws_deg')}deg"
        lines.append(
            f"| {row['name']} | {row.get('model')} | {row.get('task')} | {row.get('episodes')} | {row.get('horizon')} | "
            f"{perturb} | {float(row.get('success_rate', 0.0)):.4f} | "
            f"{float(row.get('reward_sum_mean', 0.0)):.4f} | {float(row.get('reward_sum_std', 0.0)):.4f} | "
            f"{float(row.get('pixel_error_mean', 0.0)):.4f} | "
            f"{float(row.get('corrected_pixel_error_mean', row.get('pixel_error_mean', 0.0))):.4f} |"
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
