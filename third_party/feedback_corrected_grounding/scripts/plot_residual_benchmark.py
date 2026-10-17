#!/usr/bin/env python3
"""Plot residual benchmark summaries as paper-ready sanity-check figures."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["no-image", "local-image"]
MODEL_LABEL = {"no-image": "No image", "local-image": "Local image"}
MODEL_COLOR = {"no-image": "#8a8f98", "local-image": "#1f77b4"}


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_rows(rows: list[dict[str, str]], label_prefix: str = "") -> list[dict[str, object]]:
    normalized = []
    for row in rows:
        if "split_type" in row and row.get("split_type"):
            split_type = row["split_type"]
            heldout = row["heldout"]
            label = f"{split_type}={heldout}"
        else:
            split_type = "direction"
            heldout = row["direction"]
            label = f"direction={heldout}"
        if label_prefix:
            label = f"{label_prefix}:{label}"
        normalized.append(
            {
                "label": label,
                "split_type": split_type,
                "heldout": heldout,
                "seed": int(row["seed"]),
                "model": row["model"],
                "ratio": float(row["ratio"]),
                "e_before": float(row["e_before"]),
                "e_after": float(row["e_after"]),
                "e_after_p90": float(row["e_after_p90"]),
            }
        )
    return normalized


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["label"]), str(row["model"]))].append(row)
    output = []
    for (label, model), values in sorted(buckets.items()):
        ratios = np.asarray([float(v["ratio"]) for v in values], dtype=np.float64)
        e_before = np.asarray([float(v["e_before"]) for v in values], dtype=np.float64)
        e_after = np.asarray([float(v["e_after"]) for v in values], dtype=np.float64)
        output.append(
            {
                "label": label,
                "model": model,
                "n": len(values),
                "ratio_mean": float(ratios.mean()),
                "ratio_std": float(ratios.std(ddof=0)),
                "e_before_mean": float(e_before.mean()),
                "e_after_mean": float(e_after.mean()),
            }
        )
    return output


def write_summary(agg: list[dict[str, object]], output: Path) -> None:
    lines = [
        "# Residual Benchmark Plot Summary",
        "",
        "| split | model | n | E_before | E_after | ratio mean | ratio std |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| {row['label']} | {row['model']} | {row['n']} | "
            f"{row['e_before_mean']:.4f} | {row['e_after_mean']:.4f} | "
            f"{row['ratio_mean']:.4f} | {row['ratio_std']:.4f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_ratio(agg: list[dict[str, object]], output: Path, title: str) -> None:
    labels = sorted({str(row["label"]) for row in agg})
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.34
    fig_width = max(7.0, 2.4 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    lookup = {(str(row["label"]), str(row["model"])): row for row in agg}
    for idx, model in enumerate(MODEL_ORDER):
        offset = (idx - 0.5) * width
        means = []
        stds = []
        for label in labels:
            row = lookup.get((label, model))
            means.append(float(row["ratio_mean"]) if row else 0.0)
            stds.append(float(row["ratio_std"]) if row else 0.0)
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=MODEL_LABEL.get(model, model),
            color=MODEL_COLOR.get(model),
            edgecolor="black",
            linewidth=0.6,
        )
    ax.axhline(1.0, linestyle="--", color="#444444", linewidth=1.0, label="No correction")
    ax.set_title(title)
    ax.set_ylabel("E_after / E_before (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="Path to benchmark summary.csv; can be repeated")
    parser.add_argument("--labels", default="", help="Optional comma-separated prefixes for repeated summaries")
    parser.add_argument("--output-dir", default="outputs/residual_figures")
    parser.add_argument("--title", default="Residual correction benchmark")
    args = parser.parse_args()

    prefixes = [item.strip() for item in args.labels.split(",") if item.strip()]
    all_rows = []
    for idx, summary in enumerate(args.summary):
        prefix = prefixes[idx] if idx < len(prefixes) else ""
        all_rows.extend(normalize_rows(read_rows(summary), label_prefix=prefix))
    agg = aggregate(all_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.md"
    figure_path = output_dir / "ratio_bar.png"
    write_summary(agg, summary_path)
    plot_ratio(agg, figure_path, title=args.title)
    print(f"summary: {summary_path}")
    print(f"figure: {figure_path}")
    print(summary_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
