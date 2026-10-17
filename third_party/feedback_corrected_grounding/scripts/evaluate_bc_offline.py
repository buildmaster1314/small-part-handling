#!/usr/bin/env python3
"""Offline BC checkpoint evaluation with per-task breakdown.

This script reuses the validation split stored by train_bc_policy.py arguments
(seed + val_fraction), evaluates a saved checkpoint on that split, and reports
overall/per-task MSE and MAE. It is intended for multi-task dataset debugging,
where a single average can hide task-specific failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_bc_rollout import load_policy_checkpoint  # noqa: E402
from scripts.train_bc_policy import BCProjectionDataset, choose_device, make_split, move_batch  # noqa: E402


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="", help="Override dataset path; defaults to checkpoint args['data'].")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/bc_policy/offline_eval")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    data_path = args.data or str(ckpt_args.get("data", ""))
    if not data_path:
        raise ValueError("Dataset path was not provided and checkpoint args do not contain 'data'.")

    raw = np.load(data_path, allow_pickle=False)
    split = make_split(
        raw,
        float(ckpt_args.get("val_fraction", 0.2)),
        int(ckpt_args.get("split_seed", ckpt_args.get("seed", 0))),
        str(ckpt_args.get("random_split_unit", "sample")),
        str(ckpt_args.get("tasks", "")),
    )
    patch_radius = int(np.asarray(raw["patch_radius"]).item()) if "patch_radius" in raw.files else 16
    if str(ckpt_args.get("model")) == "fcpg" and ckpt_args.get("residual_checkpoint"):
        residual_ckpt = torch.load(str(ckpt_args["residual_checkpoint"]), map_location="cpu", weights_only=False)
        patch_radius = int(residual_ckpt.get("args", {}).get("patch_radius", patch_radius))

    dataset = BCProjectionDataset(
        data_path,
        split.test,
        patch_radius=patch_radius,
        include_residual_inputs=str(ckpt_args.get("model")) == "fcpg",
        include_residual_patch=False,
        obs_mode=str(ckpt_args.get("obs_mode", "full")),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model, loaded_args = load_policy_checkpoint(args.checkpoint, device)

    task_names = raw["task_name"].astype(str) if "task_name" in raw.files else np.asarray(["unknown"] * int(raw["rgb"].shape[0]))
    task_indices = raw["task_index"].astype(np.int32) if "task_index" in raw.files else np.full(int(raw["rgb"].shape[0]), -1, dtype=np.int32)

    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"mse": [], "mae": []})
    task_index_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"mse": [], "mae": []})
    overall_mse: list[float] = []
    overall_mae: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            sample_idx = batch["sample_idx"].cpu().numpy().astype(np.int64)
            batch_device = move_batch(batch, device)
            pred = model(batch_device)
            target = batch_device["action"]
            mse_per = (pred - target).square().mean(dim=1).detach().cpu().numpy()
            mae_per = (pred - target).abs().mean(dim=1).detach().cpu().numpy()
            for idx, mse, mae in zip(sample_idx, mse_per, mae_per):
                task = str(task_names[idx])
                task_index = int(task_indices[idx])
                groups[task]["mse"].append(float(mse))
                groups[task]["mae"].append(float(mae))
                task_index_groups[f"{task}:{task_index}"]["mse"].append(float(mse))
                task_index_groups[f"{task}:{task_index}"]["mae"].append(float(mae))
                overall_mse.append(float(mse))
                overall_mae.append(float(mae))

    rows = []
    rows.append({"group": "ALL", "count": len(overall_mse), "mse": _mean(overall_mse), "mae": _mean(overall_mae)})
    for task in sorted(groups):
        rows.append({"group": task, "count": len(groups[task]["mse"]), "mse": _mean(groups[task]["mse"]), "mae": _mean(groups[task]["mae"])})

    task_index_rows = []
    for key in sorted(task_index_groups):
        vals = task_index_groups[key]
        task_index_rows.append({"group": key, "count": len(vals["mse"]), "mse": _mean(vals["mse"]), "mae": _mean(vals["mae"])})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).parent.name + "_" + Path(args.checkpoint).stem
    csv_path = output_dir / f"{stem}_offline.csv"
    json_path = output_dir / f"{stem}_offline.json"
    md_path = output_dir / f"{stem}_offline.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "count", "mse", "mae"])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "checkpoint": args.checkpoint,
        "data": data_path,
        "model": loaded_args.get("model"),
        "policy_arch": loaded_args.get("policy_arch", "mlp"),
        "obs_mode": loaded_args.get("obs_mode", "full"),
        "model_seed": int(ckpt_args.get("seed", 0)),
        "split_seed": int(ckpt_args.get("split_seed", ckpt_args.get("seed", 0))),
        "val_fraction": float(ckpt_args.get("val_fraction", 0.2)),
        "overall": rows[0],
        "per_task": rows[1:],
        "per_task_index": task_index_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Offline BC Evaluation",
        "",
        f"checkpoint: `{args.checkpoint}`",
        f"data: `{data_path}`",
        "",
        "| group | count | MSE | MAE |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['group']} | {row['count']} | {row['mse']:.6f} | {row['mae']:.6f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("# Offline BC Evaluation")
    for row in rows:
        print(f"{row['group']}: n={row['count']} mse={row['mse']:.6f} mae={row['mae']:.6f}")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
