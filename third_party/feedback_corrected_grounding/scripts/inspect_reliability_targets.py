#!/usr/bin/env python3
"""Inspect reliability targets derived from a trained residual predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_residual_predictor import ProjectionResidualDataset, make_split  # noqa: E402
from scripts.train_reliability_head import load_residual_model, pearson_corr  # noqa: E402


def parse_sigmas(value: str) -> list[float]:
    sigmas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not sigmas:
        raise ValueError("at least one sigma is required")
    return sigmas


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten().cpu()
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p10": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p90": float(quantiles[4].item()),
        "max": float(values.max().item()),
    }


def collect_stats(dataset: ProjectionResidualDataset, residual_model: torch.nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    before = []
    after = []
    pred_delta = []
    target_delta = []
    residual_model.eval()
    with torch.no_grad():
        for batch in loader:
            vector = batch["vector"].to(device)
            patch = batch["patch"].to(device)
            target_px = batch["target_px"].to(device)
            pred_px = residual_model(vector, patch) * float(dataset.patch_radius)
            before.append(torch.linalg.norm(target_px, dim=1).cpu())
            after.append(torch.linalg.norm(pred_px - target_px, dim=1).cpu())
            pred_delta.append(pred_px.cpu())
            target_delta.append(target_px.cpu())
    pred_delta_t = torch.cat(pred_delta, dim=0).float()
    target_delta_t = torch.cat(target_delta, dim=0).float()
    return {
        "before_error": torch.cat(before).float(),
        "after_error": torch.cat(after).float(),
        "pred_delta": pred_delta_t,
        "target_delta": target_delta_t,
        "pred_delta_norm": torch.linalg.norm(pred_delta_t, dim=1),
        "target_delta_norm": torch.linalg.norm(target_delta_t, dim=1),
    }


def split_report(name: str, stats: dict[str, torch.Tensor], sigmas: list[float]) -> dict[str, object]:
    before = stats["before_error"]
    after = stats["after_error"]
    pred_norm = stats["pred_delta_norm"]
    target_norm = stats["target_delta_norm"]
    report: dict[str, object] = {
        "before_error": summarize(before),
        "after_error": summarize(after),
        "pred_delta_norm": summarize(pred_norm),
        "target_delta_norm": summarize(target_norm),
        "ratio_mean_after_before": float((after.mean() / torch.clamp(before.mean(), min=1e-8)).item()),
        "corr_pred_norm_neg_after_error": pearson_corr(pred_norm, -after),
        "corr_before_neg_after_error": pearson_corr(before, -after),
        "corr_target_norm_neg_after_error": pearson_corr(target_norm, -after),
        "rho_targets": {},
    }
    rho_targets: dict[str, object] = {}
    for sigma in sigmas:
        rho_nom = torch.exp(-before / max(sigma, 1e-6))
        rho_corr = torch.exp(-after / max(sigma, 1e-6))
        rho_targets[str(sigma)] = {
            "rho_nom": summarize(rho_nom),
            "rho_corr": summarize(rho_corr),
            "corr_rho_nom_neg_after_error": pearson_corr(rho_nom, -after),
            "corr_rho_corr_neg_after_error": pearson_corr(rho_corr, -after),
        }
    report["rho_targets"] = rho_targets
    print(f"[{name}] pairs={int(before.numel())} ratio={report['ratio_mean_after_before']:.4f}")
    print(
        f"  before mean={report['before_error']['mean']:.4f} after mean={report['after_error']['mean']:.4f} "
        f"after p10/p50/p90={report['after_error']['p10']:.4f}/{report['after_error']['p50']:.4f}/{report['after_error']['p90']:.4f}"
    )
    print(
        f"  corr(pred_delta_norm, -after)={report['corr_pred_norm_neg_after_error']:.4f} "
        f"corr(before, -after)={report['corr_before_neg_after_error']:.4f}"
    )
    for sigma in sigmas:
        rho = rho_targets[str(sigma)]["rho_corr"]
        print(
            f"  sigma={sigma:g} rho_corr mean={rho['mean']:.4f} "
            f"p10/p50/p90={rho['p10']:.4f}/{rho['p50']:.4f}/{rho['p90']:.4f}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--heldout-directions", default="")
    parser.add_argument("--heldout-translations-cm", default="4")
    parser.add_argument("--heldout-yaws-deg", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-px", default="0.5,1,2,4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = choose_device(args.device)
    residual_model, residual_args, _, patch_radius = load_residual_model(Path(args.residual_checkpoint), device)
    include_obs = not bool(residual_args.get("no_obs", False))
    obs_mode = str(residual_args.get("obs_mode", "full"))
    split = make_split(
        args.data,
        args.heldout_directions,
        args.heldout_translations_cm,
        args.heldout_yaws_deg,
        args.val_fraction,
        args.seed,
    )
    train_ds = ProjectionResidualDataset(args.data, split.train, patch_radius=patch_radius, include_obs=include_obs, obs_mode=obs_mode)
    test_ds = ProjectionResidualDataset(args.data, split.test, patch_radius=patch_radius, include_obs=include_obs, obs_mode=obs_mode)
    sigmas = parse_sigmas(args.sigma_px)
    print(f"device: {device}")
    print(f"train_pairs: {len(train_ds)} test_pairs: {len(test_ds)} patch_radius: {patch_radius}")
    train_stats = collect_stats(train_ds, residual_model, device)
    test_stats = collect_stats(test_ds, residual_model, device)
    report = {
        "data": args.data,
        "residual_checkpoint": args.residual_checkpoint,
        "heldout_directions": args.heldout_directions,
        "heldout_translations_cm": args.heldout_translations_cm,
        "heldout_yaws_deg": args.heldout_yaws_deg,
        "sigmas": sigmas,
        "train": split_report("train", train_stats, sigmas),
        "test": split_report("test", test_stats, sigmas),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
