#!/usr/bin/env python3
"""Measure spurious FCPG corrections on clean A4 multiview observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402


DEFAULT_DATA = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/fcpg_a4_clean_identity_matrix.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("clean-identity evaluation permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _summary(delta: np.ndarray) -> dict[str, Any]:
    if delta.ndim != 3 or delta.shape[1:] != (3, 2):
        raise ValueError("delta must have shape [N,3,2]")
    magnitude = np.linalg.norm(delta, axis=-1)
    flattened = magnitude.reshape(-1)
    return {
        "frames": int(delta.shape[0]),
        "anchors": int(flattened.size),
        "mean_epe_px": float(flattened.mean()),
        "median_epe_px": float(np.median(flattened)),
        "p90_epe_px": float(np.quantile(flattened, 0.90)),
        "p95_epe_px": float(np.quantile(flattened, 0.95)),
        "maximum_epe_px": float(flattened.max()),
        "fraction_over_1px": float(np.mean(flattened > 1.0)),
        "fraction_over_2px": float(np.mean(flattened > 2.0)),
        "fraction_over_5px": float(np.mean(flattened > 5.0)),
        "fraction_over_10px": float(np.mean(flattened > 10.0)),
        "mean_delta_xy_px": [float(value) for value in delta.mean(axis=(0, 1))],
        "per_anchor_mean_epe_px": [
            float(value) for value in magnitude.mean(axis=0)
        ],
    }


def _grouped(
    delta: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    return {
        str(label): _summary(delta[labels == label])
        for label in np.unique(labels)
    }


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    count = int(arrays["rgb"].shape[0])
    chunks: list[np.ndarray] = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        rgb = torch.from_numpy(
            np.array(arrays["rgb"][start:stop], dtype=np.float32, copy=True)
        ).to(device)
        rgb.div_(255.0)
        nominal = torch.from_numpy(
            np.array(arrays["nominal"][start:stop], dtype=np.float32, copy=True)
        ).to(device)
        xyz = torch.from_numpy(
            np.array(arrays["xyz"][start:stop], dtype=np.float32, copy=True)
        ).to(device)
        proprio = torch.from_numpy(
            np.array(arrays["proprio"][start:stop], dtype=np.float32, copy=True)
        ).to(device)
        output = model(rgb, nominal, xyz, proprio)
        chunks.append(output.delta_px.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    data = args.data_dir.expanduser().resolve()
    arrays = {
        "rgb": np.load(data / "current_rgb.npy", mmap_mode="r"),
        "nominal": np.load(data / "current_q_nominal.npy", mmap_mode="r"),
        "truth": np.load(data / "current_q_true.npy", mmap_mode="r"),
        "xyz": np.load(data / "current_keypoints_xyz.npy", mmap_mode="r"),
        "proprio": np.load(data / "current_proprio.npy", mmap_mode="r"),
    }
    count = int(arrays["rgb"].shape[0])
    if any(int(value.shape[0]) != count for value in arrays.values()):
        raise ValueError("A4 arrays have inconsistent leading dimensions")
    nominal_truth_error = np.linalg.norm(
        np.asarray(arrays["nominal"]) - np.asarray(arrays["truth"]), axis=-1
    )
    if float(nominal_truth_error.max()) > 1.0e-6:
        raise ValueError("clean identity evaluation requires q_nominal == q_true")
    with np.load(data / "index.npz", allow_pickle=False) as archive:
        task = np.array(archive["task_name"], copy=True).astype(str)
        clock = np.array(archive["camera_clock_hour"], copy=True).astype(np.int64)
        split = np.array(archive["split"], copy=True).astype(str)
    if any(value.shape != (count,) for value in (task, clock, split)):
        raise ValueError("A4 index labels do not match frame count")

    model, checkpoint_args = load_fcpg_vit_adapter(
        args.checkpoint.expanduser().resolve(), device=device
    )
    try:
        delta = _predict(
            model,
            arrays,
            batch_size=int(args.batch_size),
            device=device,
        )
    finally:
        del model
        torch.cuda.empty_cache()
    task_clock = np.asarray(
        [f"{name}/clock{hour:02d}" for name, hour in zip(task, clock, strict=True)]
    )
    result = {
        "protocol": "fcpg_a4_clean_identity_matrix_v1",
        "interpretation": (
            "q_nominal equals q_true, so every nonzero FCPG delta is a spurious "
            "clean-condition correction"
        ),
        "data_dir": str(data),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_max_residual_px": float(
            checkpoint_args.get("max_residual_px", 20.0)
        ),
        "overall": _summary(delta),
        "by_task": _grouped(delta, task),
        "by_camera_clock": _grouped(delta, clock),
        "by_split": _grouped(delta, split),
        "by_task_camera": _grouped(delta, task_clock),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "overall": result["overall"],
                "by_task_mean_epe_px": {
                    name: metrics["mean_epe_px"]
                    for name, metrics in result["by_task"].items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
