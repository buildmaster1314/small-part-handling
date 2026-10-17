#!/usr/bin/env python3
"""Fit train-only 2D/3D blend weights and evaluate held-out camera views."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import joint_batch_from_memmaps  # noqa: E402
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_metric3d_joint_a4 import (  # noqa: E402
    METRIC3D_PROTOCOL,
    load_metric3d_checkpoint,
)
from scripts.train_lewm_action_prefix_joint_a4 import _atomic_json, _move  # noqa: E402
from scripts.train_lewm_camera_action_joint_a4 import _validate_dataset  # noqa: E402
from scripts.train_lewm_metric3d_joint_a4 import _cuda0  # noqa: E402


def optimal_blend_weight(numerator: float, denominator: float) -> float:
    """Return the bounded least-squares weight for base + weight*(3D-base)."""

    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError("blend statistics must be finite")
    if float(denominator) <= 1.0e-12:
        return 0.0
    return float(np.clip(float(numerator) / float(denominator), 0.0, 1.0))


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if int(args.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    return args


def _pairs(index: Mapping[str, np.ndarray], split: str) -> list[tuple[int, int]]:
    bundles = np.flatnonzero(np.asarray(index["split"]).astype(str) == str(split))
    return [(int(bundle), branch) for bundle in bundles for branch in range(8)]


@torch.inference_mode()
def _fit_weights(
    model: torch.nn.Module,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    numerators = {task: 0.0 for task in TASKS}
    denominators = {task: 0.0 for task in TASKS}
    pairs = _pairs(index, "train")
    model.eval()
    for start in range(0, len(pairs), int(batch_size)):
        rows = pairs[start : start + int(batch_size)]
        bundles = np.asarray([row[0] for row in rows], dtype=np.int64)
        branches = np.asarray([row[1] for row in rows], dtype=np.int64)
        batch = _move(
            joint_batch_from_memmaps(
                index,
                arrays,
                bundle_indices=bundles,
                branch_indices=branches,
            ),
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
        base = output.base_predicted_q_px.float()
        delta = output.predicted_q_px.float() - base
        target_residual = batch["future_q_true"].float() - base
        valid = batch["future_valid"].float()[..., None]
        numerator = (delta * target_residual * valid).sum(dim=(1, 2, 3))
        denominator = (delta.square() * valid).sum(dim=(1, 2, 3))
        for row, task_index in enumerate(batch["task_index"].cpu().tolist()):
            task = TASKS[int(task_index)]
            numerators[task] += float(numerator[row])
            denominators[task] += float(denominator[row])
    result = {
        task: optimal_blend_weight(numerators[task], denominators[task])
        for task in TASKS
    }
    result["__global__"] = optimal_blend_weight(
        sum(numerators.values()), sum(denominators.values())
    )
    return result


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    names = ("base_2d", "metric3d", "task_blend", "global_blend")
    totals = {
        task: {"count": 0, **{name: 0.0 for name in names}}
        for task in TASKS
    }
    pairs = _pairs(index, "development")
    model.eval()
    for start in range(0, len(pairs), int(batch_size)):
        rows = pairs[start : start + int(batch_size)]
        bundles = np.asarray([row[0] for row in rows], dtype=np.int64)
        branches = np.asarray([row[1] for row in rows], dtype=np.int64)
        batch = _move(
            joint_batch_from_memmaps(
                index,
                arrays,
                bundle_indices=bundles,
                branch_indices=branches,
            ),
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
        base = output.base_predicted_q_px.float()
        metric3d = output.predicted_q_px.float()
        target = batch["future_q_true"].float()
        valid = batch["future_valid"].float()
        for row, task_index in enumerate(batch["task_index"].cpu().tolist()):
            task = TASKS[int(task_index)]
            task_prediction = base[row] + float(weights[task]) * (
                metric3d[row] - base[row]
            )
            global_prediction = base[row] + float(weights["__global__"]) * (
                metric3d[row] - base[row]
            )
            predictions = {
                "base_2d": base[row],
                "metric3d": metric3d[row],
                "task_blend": task_prediction,
                "global_blend": global_prediction,
            }
            denominator = valid[row].sum().clamp_min(1.0)
            totals[task]["count"] = int(totals[task]["count"]) + 1
            for name, prediction in predictions.items():
                error = torch.linalg.vector_norm(prediction - target[row], dim=-1)
                epe = (error * valid[row]).sum() / denominator
                totals[task][name] = float(totals[task][name]) + float(epe)
    per_task: dict[str, dict[str, float | int]] = {}
    for task in TASKS:
        count = int(totals[task]["count"])
        if count <= 0:
            raise RuntimeError(f"hybrid evaluation missed task {task}")
        per_task[task] = {
            "sequence_count": count,
            "train_only_blend_weight": float(weights[task]),
            **{
                f"{name}_pixel_epe": float(totals[task][name]) / count
                for name in names
            },
        }
    task_macro = {
        f"{name}_pixel_epe": float(
            np.mean([float(per_task[task][f"{name}_pixel_epe"]) for task in TASKS])
        )
        for name in names
    }
    return {
        "sequence_count": len(pairs),
        "global_train_only_blend_weight": float(weights["__global__"]),
        "task_macro": task_macro,
        "per_task": per_task,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("metric3d checkpoint must contain a mapping")
    model = load_metric3d_checkpoint(checkpoint).to(device).eval()
    raw_path = Path(str(checkpoint["raw_dataset"])).expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    weights = _fit_weights(
        model, index, arrays, device=device, batch_size=int(args.batch_size)
    )
    evaluation = _evaluate(
        model,
        index,
        arrays,
        weights=weights,
        device=device,
        batch_size=int(args.batch_size),
    )
    payload = {
        "protocol": f"{METRIC3D_PROTOCOL}_train_only_scalar_hybrid_v1",
        "checkpoint": str(checkpoint_path),
        "fit_split": "train_even_orbit_views",
        "evaluation_split": "development_odd_orbit_views",
        "blend_formula": "base_2d + lambda * (metric3d - base_2d)",
        "weights": weights,
        "evaluation": evaluation,
    }
    output = args.output.expanduser().resolve()
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "weights": weights,
                "task_macro": evaluation["task_macro"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
