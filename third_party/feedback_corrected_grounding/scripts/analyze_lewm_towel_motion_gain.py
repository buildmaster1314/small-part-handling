#!/usr/bin/env python3
"""Audit action-to-pixel motion gain for Towel on held-out clock03."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_prefix_joint_a4 import joint_batch_from_memmaps  # noqa: E402
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_camera_action_joint_a4 import (  # noqa: E402
    CAMERA_ACTION_PROTOCOL,
    load_camera_action_joint_checkpoint,
)
from scripts.train_lewm_action_prefix_joint_a4 import _move  # noqa: E402


SEMANTIC_NAMES = (
    "expert",
    "reverse",
    "hold",
    "half_speed",
    "transverse_a_positive",
    "transverse_a_negative",
    "transverse_b_positive",
    "transverse_b_negative",
)
DEFAULT_RAW = ROOT / (
    "outputs/lewm_action_prefix_a4/joint_orbit12_gripper_24train_8dev"
)
DEFAULT_SOURCE = ROOT / "outputs/lewm_action_branch_a4/formal_v2/state_cache.npz"
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/towel_motion_gain_clock03/summary.json"
)
DEFAULT_CHECKPOINTS = (
    ROOT
    / "outputs/lewm_action_prefix_a4/camera_action_seed40571_2k/"
    "lewm_camera_action_joint_latest.pt",
    ROOT
    / "outputs/lewm_action_prefix_a4/camera_action_seed40571_10k/"
    "lewm_camera_action_joint_latest.pt",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", action="append", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _cuda0(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("towel motion-gain analysis permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def motion_gain_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    semantic: np.ndarray,
    *,
    minimum_motion_px: float = 1.0,
) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share shape [B,8,3,2]")
    if current.shape != (prediction.shape[0], 3, 2):
        raise ValueError("current must have shape [B,3,2]")
    labels = np.asarray(semantic, dtype=np.int64)
    if labels.shape != (prediction.shape[0],):
        raise ValueError("semantic labels must have shape [B]")
    if not 0.0 < float(minimum_motion_px):
        raise ValueError("minimum motion must be positive")

    target_motion = target.float() - current[:, None].float()
    predicted_motion = prediction.float() - current[:, None].float()
    denominator = target_motion.square().sum(dim=-1)
    support = denominator >= float(minimum_motion_px) ** 2
    gain = (
        (predicted_motion * target_motion).sum(dim=-1)
        / denominator.clamp_min(1.0e-6)
    ).detach().cpu().numpy()
    norm_ratio = (
        torch.linalg.vector_norm(predicted_motion, dim=-1)
        / torch.linalg.vector_norm(target_motion, dim=-1).clamp_min(1.0e-6)
    ).detach().cpu().numpy()
    supported = support.detach().cpu().numpy()

    result: dict[str, Any] = {}
    for semantic_id, name in enumerate(SEMANTIC_NAMES):
        mask = (labels == semantic_id)[:, None, None] & supported
        count = int(mask.sum())
        if not count:
            result[name] = {"point_count": 0}
            continue
        per_horizon = []
        for horizon in range(prediction.shape[1]):
            horizon_mask = mask[:, horizon]
            per_horizon.append(
                float(np.mean(gain[:, horizon][horizon_mask]))
                if bool(horizon_mask.any())
                else None
            )
        result[name] = {
            "point_count": count,
            "mean_signed_motion_gain": float(np.mean(gain[mask])),
            "median_signed_motion_gain": float(np.median(gain[mask])),
            "median_motion_norm_ratio": float(np.median(norm_ratio[mask])),
            "mean_signed_motion_gain_by_horizon": per_horizon,
        }
    return result


def _checkpoint_label(payload: Mapping[str, Any]) -> str:
    return f"step_{int(payload.get('step', -1))}"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoints = tuple(args.checkpoint or DEFAULT_CHECKPOINTS)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    raw_path = args.raw_dataset.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    source = np.load(args.source_cache.expanduser().resolve(), allow_pickle=False)

    selected_bundles = np.flatnonzero(
        (np.asarray(index["task_name"]).astype(str) == "towel-wipe-v0")
        & (np.asarray(index["camera_name"]).astype(str) == "clock03_e35")
        & (np.asarray(index["split"]).astype(str) == "development")
    )
    if selected_bundles.size == 0:
        raise RuntimeError("no Towel clock03 development bundles were found")
    bundles = np.repeat(selected_bundles, 8)
    branches = np.tile(np.arange(8, dtype=np.int64), selected_bundles.size)
    cache_indices = np.asarray(index["cache_index"], dtype=np.int64)[bundles]
    semantic = np.asarray(source["branch_semantic_index"])[cache_indices, branches]
    batch = _move(
        joint_batch_from_memmaps(
            index,
            arrays,
            bundle_indices=bundles,
            branch_indices=branches,
        ),
        device,
    )

    models: dict[str, Any] = {}
    for checkpoint_path in checkpoints:
        payload = torch.load(
            checkpoint_path.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, Mapping) or payload.get("protocol") != CAMERA_ACTION_PROTOCOL:
            raise ValueError("all checkpoints must be camera-action LeWM models")
        model = load_camera_action_joint_checkpoint(payload).to(device).eval()
        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                output = model(batch)
            models[_checkpoint_label(payload)] = {
                "checkpoint": str(checkpoint_path.expanduser().resolve()),
                "sigreg_weight": float(payload.get("sigreg_weight", -1.0)),
                "semantics": motion_gain_rows(
                    output.predicted_q_px,
                    batch["future_q_true"],
                    batch["current_q_corrected"],
                    semantic,
                ),
            }
        finally:
            del model
            torch.cuda.empty_cache()

    result = {
        "protocol": "lewm_towel_motion_gain_clock03_v1",
        "scope": "development",
        "task": "towel-wipe-v0",
        "camera": "clock03_e35",
        "bundle_count": int(selected_bundles.size),
        "sequence_count": int(bundles.size),
        "models": models,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
