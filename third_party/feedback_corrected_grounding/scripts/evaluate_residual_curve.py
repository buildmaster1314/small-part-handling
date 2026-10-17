#!/usr/bin/env python3
"""Evaluate trained residual predictors over perturbation severity levels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_residual_predictor import (  # noqa: E402
    LocalImageResidualNet,
    NoImageResidualNet,
    ProjectionResidualDataset,
    evaluate,
)


MODEL_COLOR = {"no-image": "#8a8f98", "local-image": "#1f77b4"}
MODEL_LABEL = {"no-image": "No image", "local-image": "Local image"}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Checkpoint spec must be label=path")
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip())


def make_float_mask(values: np.ndarray, target: str, atol: float = 1e-4) -> np.ndarray:
    if target == "":
        return np.ones(values.shape, dtype=bool)
    return np.isclose(values.astype(np.float32), np.float32(float(target)), atol=atol)


def load_model(label: str, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model_type = args.get("model", label)
    vector_dim = int(checkpoint["vector_dim"])
    hidden_dim = int(args.get("hidden_dim", 128))
    patch_radius = int(args.get("patch_radius", 16))
    if model_type == "no-image":
        model = NoImageResidualNet(vector_dim, hidden_dim=hidden_dim)
    elif model_type == "local-image":
        model = LocalImageResidualNet(vector_dim, hidden_dim=hidden_dim, patch_radius=patch_radius)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, model_type, patch_radius, args


def evaluate_group(
    data_path: Path,
    model,
    device: torch.device,
    patch_radius: int,
    indices: np.ndarray,
    batch_size: int,
    include_obs: bool,
    obs_mode: str,
) -> dict[str, float] | None:
    if len(indices) == 0:
        return None
    try:
        dataset = ProjectionResidualDataset(
            data_path,
            indices,
            patch_radius=patch_radius,
            include_obs=include_obs,
            require_inside_patch=True,
            obs_mode=obs_mode,
        )
    except RuntimeError:
        return None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, device, patch_radius)
    metrics["pairs"] = len(dataset)
    metrics["frames"] = len(indices)
    return metrics


def aggregate_curve_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["model_label"]), str(row["model_type"]), float(row["severity"]))
        buckets.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (label, model_type, severity), values in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][2])):
        ratios = np.asarray([float(v["ratio"]) for v in values], dtype=np.float64)
        e_before = np.asarray([float(v["e_before"]) for v in values], dtype=np.float64)
        e_after = np.asarray([float(v["e_after"]) for v in values], dtype=np.float64)
        output.append(
            {
                "model_label": label,
                "model_type": model_type,
                "group_by": values[0]["group_by"],
                "severity": severity,
                "runs": len(values),
                "frames": int(sum(int(v["frames"]) for v in values)),
                "pairs": int(sum(int(v["pairs"]) for v in values)),
                "e_before": float(e_before.mean()),
                "e_after": float(e_after.mean()),
                "ratio": float(ratios.mean()),
                "ratio_std": float(ratios.std(ddof=0)),
            }
        )
    return output


def plot_curve(rows: list[dict[str, object]], output: Path, group_by: str, title: str) -> None:
    labels = sorted({str(row["model_label"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label in labels:
        label_rows = [row for row in rows if str(row["model_label"]) == label]
        label_rows.sort(key=lambda row: float(row["severity"]))
        x = [float(row["severity"]) for row in label_rows]
        y = [float(row["ratio"]) for row in label_rows]
        yerr = [float(row.get("ratio_std", 0.0)) for row in label_rows]
        model_type = str(label_rows[0]["model_type"])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            capsize=4,
            label=MODEL_LABEL.get(model_type, label),
            color=MODEL_COLOR.get(model_type),
        )
    ax.axhline(1.0, linestyle="--", color="#444444", linewidth=1.0, label="No correction")
    ax.set_title(title)
    ax.set_xlabel(group_by)
    ax.set_ylabel("E_after / E_before (lower is better)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="label=path; can be repeated")
    parser.add_argument("--group-by", choices=["translation_cm", "yaw_deg"], required=True)
    parser.add_argument("--filter-translation-cm", default="")
    parser.add_argument("--filter-yaw-deg", default="")
    parser.add_argument("--filter-direction", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/residual_curves")
    parser.add_argument("--title", default="Residual correction severity curve")
    args = parser.parse_args()

    data_path = Path(args.data)
    data = np.load(data_path, allow_pickle=False)
    n = int(data["rgb"].shape[0])
    base_mask = np.ones(n, dtype=bool)
    if args.filter_translation_cm:
        base_mask &= make_float_mask(data["translation_cm"], args.filter_translation_cm)
    if args.filter_yaw_deg:
        base_mask &= make_float_mask(data["yaw_deg"], args.filter_yaw_deg)
    if args.filter_direction:
        base_mask &= data["translation_direction"].astype(str) == args.filter_direction

    severities = sorted(float(v) for v in np.unique(data[args.group_by][base_mask].astype(np.float32)))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows: list[dict[str, object]] = []
    for spec in args.checkpoint:
        label, checkpoint_path = parse_checkpoint(spec)
        model, model_type, patch_radius, ckpt_args = load_model(label, checkpoint_path, device)
        include_obs = not bool(ckpt_args.get("no_obs", False))
        obs_mode = str(ckpt_args.get("obs_mode", "full"))
        for severity in severities:
            mask = base_mask & make_float_mask(data[args.group_by], str(severity))
            indices = np.arange(n, dtype=np.int64)[mask]
            metrics = evaluate_group(data_path, model, device, patch_radius, indices, args.batch_size, include_obs, obs_mode)
            if metrics is None:
                continue
            rows.append(
                {
                    "model_label": label,
                    "model_type": model_type,
                    "group_by": args.group_by,
                    "severity": severity,
                    "frames": int(metrics["frames"]),
                    "pairs": int(metrics["pairs"]),
                    "e_before": float(metrics["e_before"]),
                    "e_after": float(metrics["e_after"]),
                    "ratio": float(metrics["ratio"]),
                    "e_after_p90": float(metrics["e_after_p90"]),
                    "checkpoint": str(checkpoint_path),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.group_by}_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    agg_rows = aggregate_curve_rows(rows)

    md_path = output_dir / f"{args.group_by}_curve.md"
    lines = [
        f"# {args.title}",
        "",
        "| model | severity | runs | frames | E_before | E_after | ratio | ratio std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        lines.append(
            f"| {row['model_label']} | {float(row['severity']):g} | {row['runs']} | {row['frames']} | "
            f"{row['e_before']:.4f} | {row['e_after']:.4f} | {row['ratio']:.4f} | {row['ratio_std']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure_path = output_dir / f"{args.group_by}_curve.png"
    plot_curve(agg_rows, figure_path, args.group_by, args.title)
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    print(f"figure: {figure_path}")
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
