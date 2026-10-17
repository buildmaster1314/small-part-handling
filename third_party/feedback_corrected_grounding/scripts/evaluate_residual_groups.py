#!/usr/bin/env python3
"""Evaluate residual predictors by dataset metadata groups."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_residual_curve import (  # noqa: E402
    evaluate_group,
    load_model,
    make_float_mask,
    parse_checkpoint,
)
from scripts.train_residual_predictor import make_split  # noqa: E402


def group_sort_key(value: object) -> tuple[int, float | str]:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def make_group_mask(values: np.ndarray, group_value: object) -> np.ndarray:
    if np.issubdtype(values.dtype, np.number):
        return np.isclose(values.astype(np.float32), np.float32(float(group_value)), atol=1e-4)
    return values.astype(str) == str(group_value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="label=path; can be repeated")
    parser.add_argument(
        "--group-by",
        choices=["task_name", "task_index", "translation_cm", "yaw_deg", "translation_direction"],
        default="task_name",
    )
    parser.add_argument("--filter-translation-cm", default="")
    parser.add_argument("--filter-yaw-deg", default="")
    parser.add_argument("--filter-direction", default="")
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--random-split-unit",
        choices=["sample", "task-episode"],
        default="sample",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/residual_group_eval")
    parser.add_argument("--title", default="Residual correction group evaluation")
    args = parser.parse_args()

    data_path = Path(args.data)
    data = np.load(data_path, allow_pickle=False)
    if args.group_by not in data.files:
        raise KeyError(f"{args.group_by} not found in {data_path}")
    n = int(data["rgb"].shape[0])
    base_mask = np.ones(n, dtype=bool)
    if args.split != "all":
        split = make_split(
            data_path,
            heldout_directions="",
            heldout_translations_cm="",
            heldout_yaws_deg="",
            val_fraction=args.val_fraction,
            seed=args.split_seed,
            random_split_unit=args.random_split_unit,
        )
        selected = split.train if args.split == "train" else split.test
        split_mask = np.zeros(n, dtype=bool)
        split_mask[selected] = True
        base_mask &= split_mask
    if args.filter_translation_cm:
        base_mask &= make_float_mask(data["translation_cm"], args.filter_translation_cm)
    if args.filter_yaw_deg:
        base_mask &= make_float_mask(data["yaw_deg"], args.filter_yaw_deg)
    if args.filter_direction:
        base_mask &= data["translation_direction"].astype(str) == args.filter_direction

    values = data[args.group_by]
    groups = sorted(np.unique(values[base_mask]).tolist(), key=group_sort_key)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows: list[dict[str, object]] = []
    for spec in args.checkpoint:
        label, checkpoint_path = parse_checkpoint(spec)
        model, model_type, patch_radius, ckpt_args = load_model(label, checkpoint_path, device)
        include_obs = not bool(ckpt_args.get("no_obs", False))
        for group in groups:
            mask = base_mask & make_group_mask(values, group)
            indices = np.arange(n, dtype=np.int64)[mask]
            metrics = evaluate_group(data_path, model, device, patch_radius, indices, args.batch_size, include_obs)
            if metrics is None:
                continue
            rows.append(
                {
                    "model_label": label,
                    "model_type": model_type,
                    "group_by": args.group_by,
                    "split": args.split,
                    "group": str(group),
                    "frames": int(metrics["frames"]),
                    "pairs": int(metrics["pairs"]),
                    "e_before": float(metrics["e_before"]),
                    "e_after": float(metrics["e_after"]),
                    "ratio": float(metrics["ratio"]),
                    "e_after_p90": float(metrics["e_after_p90"]),
                    "clean_count": int(metrics.get("clean_count", 0)),
                    "clean_e_after": float(metrics.get("clean_e_after", float("nan"))),
                    "checkpoint": str(checkpoint_path),
                }
            )

    if not rows:
        raise RuntimeError("No rows produced")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.group_by
    csv_path = output_dir / f"{stem}_groups.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / f"{stem}_groups.md"
    lines = [
        f"# {args.title}",
        "",
        f"data: `{data_path}`",
        f"split: `{args.split}` (`{args.random_split_unit}`, seed={args.split_seed})",
        "",
        "| model | group | frames | pairs | E_before | E_after | ratio | E_after p90 | clean E_after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        clean = row["clean_e_after"]
        clean_str = "nan" if not np.isfinite(float(clean)) else f"{float(clean):.4f}"
        model_label = row["model_label"]
        group = row["group"]
        frames = row["frames"]
        pairs = row["pairs"]
        e_before = float(row["e_before"])
        e_after = float(row["e_after"])
        ratio = float(row["ratio"])
        e_after_p90 = float(row["e_after_p90"])
        lines.append(
            f"| {model_label} | {group} | {frames} | {pairs} | "
            f"{e_before:.4f} | {e_after:.4f} | {ratio:.4f} | {e_after_p90:.4f} | {clean_str} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
