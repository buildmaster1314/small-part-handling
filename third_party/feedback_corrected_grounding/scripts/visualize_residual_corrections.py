#!/usr/bin/env python3
"""Visualize q_nominal -> q_corrected -> q_true for FCPG residual models."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_residual_predictor import ProjectionResidualDataset, make_split  # noqa: E402
from scripts.train_reliability_head import load_residual_model  # noqa: E402


COLOR_NOM = "#d62728"
COLOR_CORR = "#1f77b4"
COLOR_TRUE = "#2ca02c"


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_raw(data_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(data_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def collect_predictions(
    data_path: str | Path,
    checkpoint_path: str | Path,
    split_indices: np.ndarray,
    device: torch.device,
) -> tuple[ProjectionResidualDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model, residual_args, _, patch_radius = load_residual_model(Path(checkpoint_path), device)
    include_obs = not bool(residual_args.get("no_obs", False))
    obs_mode = str(residual_args.get("obs_mode", "full"))
    dataset = ProjectionResidualDataset(
        data_path, split_indices, patch_radius=patch_radius, include_obs=include_obs, obs_mode=obs_mode
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    pred_px = []
    target_px = []
    pairs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            vector = batch["vector"].to(device)
            patch = batch["patch"].to(device)
            pred = model(vector, patch) * float(patch_radius)
            pred_px.append(pred.cpu().numpy())
            target_px.append(batch["target_px"].cpu().numpy())
    pred_t = np.concatenate(pred_px, axis=0)
    target_t = np.concatenate(target_px, axis=0)
    pairs_t = dataset.pairs.copy()
    before = np.linalg.norm(target_t, axis=1)
    after = np.linalg.norm(pred_t - target_t, axis=1)
    return dataset, pairs_t, pred_t, before, after


def select_indices(before: np.ndarray, after: np.ndarray, mode: str, count: int, min_before: float) -> np.ndarray:
    valid = np.where(before >= min_before)[0]
    if valid.size == 0:
        valid = np.arange(before.shape[0])
    improvement = before - after
    if mode == "best":
        order = valid[np.argsort(-improvement[valid])]
    elif mode == "worst":
        order = valid[np.argsort(improvement[valid])]
    elif mode == "largest-before":
        order = valid[np.argsort(-before[valid])]
    else:
        raise ValueError(f"unknown selection mode: {mode}")
    return order[: max(1, count)]


def draw_point(ax, xy: np.ndarray, color: str, label: str) -> None:
    ax.scatter([xy[0]], [xy[1]], s=46, c=color, edgecolors="white", linewidths=0.8, label=label, zorder=4)
    ax.add_patch(Circle((xy[0], xy[1]), radius=2.6, fill=False, color=color, linewidth=1.2, zorder=3))


def draw_sample(
    ax,
    image: np.ndarray,
    q_nom: np.ndarray,
    q_corr: np.ndarray,
    q_true: np.ndarray,
    title: str,
    show_patch_radius: int | None,
) -> None:
    ax.imshow(image)
    if show_patch_radius is not None:
        ax.add_patch(
            plt.Rectangle(
                (q_nom[0] - show_patch_radius, q_nom[1] - show_patch_radius),
                2 * show_patch_radius + 1,
                2 * show_patch_radius + 1,
                fill=False,
                color=COLOR_NOM,
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
            )
        )
    ax.annotate(
        "",
        xy=q_corr,
        xytext=q_nom,
        arrowprops={"arrowstyle": "->", "color": COLOR_CORR, "linewidth": 1.6},
        zorder=3,
    )
    draw_point(ax, q_nom, COLOR_NOM, "nominal")
    draw_point(ax, q_corr, COLOR_CORR, "corrected")
    draw_point(ax, q_true, COLOR_TRUE, "true")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def save_grid(rows: list[dict[str, object]], output: Path, columns: int, patch_radius: int | None) -> None:
    n = len(rows)
    columns = max(1, min(columns, n))
    rows_count = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(4.0 * columns, 3.5 * rows_count), squeeze=False)
    for ax in axes.flatten()[n:]:
        ax.set_axis_off()
    for ax, row in zip(axes.flatten(), rows):
        draw_sample(
            ax,
            row["image"],
            row["q_nom"],
            row["q_corr"],
            row["q_true"],
            row["title"],
            patch_radius,
        )
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    unique = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique.append((handle, label))
    if unique:
        fig.legend([h for h, _ in unique], [l for _, l in unique], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    fieldnames = [
        "rank",
        "sample_idx",
        "keypoint_idx",
        "task_name",
        "keypoint_site",
        "before_error",
        "after_error",
        "improvement",
        "ratio",
        "q_nom_x",
        "q_nom_y",
        "q_corr_x",
        "q_corr_y",
        "q_true_x",
        "q_true_y",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_summary(rows: list[dict[str, object]], output: Path, figure_name: str) -> None:
    lines = [
        "# Residual Correction Visualization Summary",
        "",
        f"Figure: `{figure_name}`",
        "",
        "| rank | task | keypoint | E_before | E_after | improvement | ratio |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['task_name']} | {row['keypoint_site']} | "
            f"{row['before_error']:.4f} | {row['after_error']:.4f} | "
            f"{row['improvement']:.4f} | {row['ratio']:.4f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/residual_visualizations")
    parser.add_argument("--heldout-directions", default="")
    parser.add_argument("--heldout-translations-cm", default="4")
    parser.add_argument("--heldout-yaws-deg", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--random-split-unit",
        choices=["sample", "task-episode"],
        default="sample",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=["best", "worst", "largest-before"], default="best")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--min-before", type=float, default=1.0)
    parser.add_argument("--show-patch", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    raw = load_raw(args.data)
    split = make_split(
        args.data,
        args.heldout_directions,
        args.heldout_translations_cm,
        args.heldout_yaws_deg,
        args.val_fraction,
        args.seed,
        random_split_unit=args.random_split_unit,
    )
    dataset, pairs, pred_delta, before, after = collect_predictions(args.data, args.checkpoint, split.test, device)
    selected = select_indices(before, after, args.selection, args.count, args.min_before)

    keypoint_sites = raw.get("keypoint_site")
    task_names = raw.get("task_name")
    rows = []
    for rank, pair_idx in enumerate(selected.tolist(), start=1):
        sample_idx, keypoint_idx = pairs[pair_idx]
        q_nom = raw["q_nominal"][sample_idx, keypoint_idx].astype(np.float64)
        q_true = raw["q_true"][sample_idx, keypoint_idx].astype(np.float64)
        q_corr = q_nom + pred_delta[pair_idx].astype(np.float64)
        before_error = float(before[pair_idx])
        after_error = float(after[pair_idx])
        task = str(task_names[sample_idx]) if task_names is not None else str(sample_idx)
        keypoint = str(keypoint_sites[keypoint_idx]) if keypoint_sites is not None else str(keypoint_idx)
        ratio = after_error / max(before_error, 1e-8)
        row = {
            "rank": rank,
            "sample_idx": int(sample_idx),
            "keypoint_idx": int(keypoint_idx),
            "task_name": task,
            "keypoint_site": keypoint,
            "before_error": before_error,
            "after_error": after_error,
            "improvement": before_error - after_error,
            "ratio": ratio,
            "q_nom_x": float(q_nom[0]),
            "q_nom_y": float(q_nom[1]),
            "q_corr_x": float(q_corr[0]),
            "q_corr_y": float(q_corr[1]),
            "q_true_x": float(q_true[0]),
            "q_true_y": float(q_true[1]),
            "q_nom": q_nom,
            "q_corr": q_corr,
            "q_true": q_true,
            "image": raw["rgb"][sample_idx],
            "title": f"{rank}. {task} / {keypoint}\n{before_error:.2f}px -> {after_error:.2f}px",
        }
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{args.selection}_grid.png"
    csv_path = output_dir / f"{args.selection}_samples.csv"
    summary_path = output_dir / f"{args.selection}_summary.md"
    patch_radius = dataset.patch_radius if args.show_patch else None
    save_grid(rows, figure_path, args.columns, patch_radius)
    write_csv(rows, csv_path)
    write_summary(rows, summary_path, figure_path.name)
    print(f"device: {device}")
    print(f"split_test_frames: {len(split.test)} pairs: {len(dataset)}")
    print(f"figure: {figure_path}")
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")
    print(summary_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
