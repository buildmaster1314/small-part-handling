#!/usr/bin/env python3
"""Stage-1 synthetic heatmap sanity check.

This script verifies the coordinate and local-patch convention before any
simulator or policy code is involved. It creates a Gaussian heatmap at q_true,
crops around q_nominal, estimates the local maximum by soft-argmax, and reports
whether E_after / E_before is below the stage-1 threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.geometry import crop_patch, make_gaussian_heatmap, soft_argmax_patch


def run_trial(rng: np.random.Generator, width: int, height: int, radius: int, sigma: float, max_offset: float) -> tuple[float, float, bool]:
    margin = radius + int(max_offset) + 4
    q_true = np.array(
        [
            rng.uniform(margin, width - margin),
            rng.uniform(margin, height - margin),
        ],
        dtype=np.float64,
    )
    offset = rng.uniform(-max_offset, max_offset, size=2)
    q_nominal = q_true + offset
    heatmap = make_gaussian_heatmap(width, height, q_true, sigma=sigma)
    patch, origin_xy = crop_patch(heatmap, q_nominal, radius)
    q_est = soft_argmax_patch(patch, origin_xy, temperature=0.05)
    e_before = float(np.linalg.norm(q_nominal - q_true))
    e_after = float(np.linalg.norm(q_est - q_true))
    inside_patch = bool(np.all(np.abs(q_true - np.round(q_nominal)) <= radius))
    return e_before, e_after, inside_patch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--max-offset", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    before = []
    after = []
    inside = []
    for _ in range(args.num_trials):
        e_before, e_after, inside_patch = run_trial(
            rng,
            width=args.width,
            height=args.height,
            radius=args.radius,
            sigma=args.sigma,
            max_offset=args.max_offset,
        )
        before.append(e_before)
        after.append(e_after)
        inside.append(inside_patch)

    before_arr = np.asarray(before)
    after_arr = np.asarray(after)
    ratio = float(after_arr.mean() / max(before_arr.mean(), 1e-12))
    inside_rate = float(np.mean(inside))
    print(f"E_before_mean: {before_arr.mean():.4f}")
    print(f"E_after_mean:  {after_arr.mean():.4f}")
    print(f"E_after/E_before: {ratio:.4f}")
    print(f"q_true_inside_patch_rate: {inside_rate:.4f}")
    if ratio >= args.threshold:
        print(f"FAIL: ratio >= threshold ({args.threshold})")
        return 1
    print(f"PASS: ratio < threshold ({args.threshold})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
