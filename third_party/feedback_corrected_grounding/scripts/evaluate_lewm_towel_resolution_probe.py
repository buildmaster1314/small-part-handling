#!/usr/bin/env python3
"""Paired 224/448 input-resolution probe for the frozen towel LeWM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import HORIZON, RAW_STEPS_PER_MODEL_STEP, TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import a4_camera_name, load_frozen_a4_state_encoder  # noqa: E402
from fcpg.lewm_camera_action_joint_a4 import load_camera_action_joint_checkpoint  # noqa: E402
from fcpg.lewm_counterfactual_runtime import corrected_points  # noqa: E402
from fcpg.lewm_orbit_camera import OrbitCameraView  # noqa: E402
from scripts import render_lewm_action_branch_a4_accuracy_gallery as legacy  # noqa: E402
from scripts import render_lewm_action_branch_a4_trajectory_comparison as trajectory  # noqa: E402
from scripts import render_lewm_action_prefix_joint_a4_accuracy_gallery as gallery  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_BASE_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/camera_action_phaseaware_a4c_v4_seed40571_10k/"
    "lewm_camera_action_joint_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/towel_resolution_probe_seed40571_clock12.json"
)
RESOLUTIONS = (224, 448)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--orbit-clock-hour", type=int, default=12)
    parser.add_argument("--orbit-elevation-degrees", type=float, default=35.0)
    parser.add_argument("--orbit-radius-m", type=float, default=1.1)
    parser.add_argument("--max-raw-steps", type=int, default=500)
    parser.add_argument("--post-success-steps", type=int, default=40)
    return parser


def _device(value: str) -> torch.device:
    requested = torch.device(value)
    if requested.type != "cuda" or requested.index not in (None, 0):
        raise ValueError("resolution probe permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _equivalent_metrics(
    episode: dict[str, Any],
    starts: np.ndarray,
    predictions: np.ndarray,
    *,
    resolution: int,
) -> dict[str, dict[str, float]]:
    truth = np.stack([row.q_true for row in episode["records"]]).astype(np.float32)
    scale = float(resolution - 1)
    result: dict[str, dict[str, float]] = {}
    for horizon in range(1, HORIZON + 1):
        target = truth[starts + horizon * RAW_STEPS_PER_MODEL_STEP]
        estimate = np.asarray(predictions[:, horizon - 1], dtype=np.float32)
        anchor_raw = np.linalg.norm(estimate - target, axis=-1)
        center_raw = np.linalg.norm(estimate.mean(1) - target.mean(1), axis=-1)
        result[f"H{horizon}"] = {
            "anchor_epe_px": float(anchor_raw.mean()),
            "center_epe_px": float(center_raw.mean()),
            "anchor_epe_224eq_px": float((anchor_raw / scale * 223.0).mean()),
            "center_epe_224eq_px": float((center_raw / scale * 223.0).mean()),
        }
    return result


def _current_grounding_metrics(
    selected: list[Any],
    points: np.ndarray,
    *,
    resolution: int,
) -> dict[str, float]:
    estimate = np.asarray(points, dtype=np.float32)
    truth = np.stack([row.q_true for row in selected]).astype(np.float32)
    nominal = np.stack([row.q_nominal for row in selected]).astype(np.float32)
    scale = float(resolution - 1)
    error = np.linalg.norm(estimate - truth, axis=-1)
    correction = np.linalg.norm(estimate - nominal, axis=-1)
    out_of_bounds = (
        (estimate[..., 0] < 0.0)
        | (estimate[..., 0] > scale)
        | (estimate[..., 1] < 0.0)
        | (estimate[..., 1] > scale)
    )
    return {
        "epe_px": float(error.mean()),
        "epe_224eq_px": float((error / scale * 223.0).mean()),
        "correction_from_nominal_px": float(correction.mean()),
        "correction_from_nominal_224eq_px": float(
            (correction / scale * 223.0).mean()
        ),
        "out_of_bounds_fraction": float(out_of_bounds.mean()),
    }


def _paired_episode_audit(episodes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    low, high = (episodes[value] for value in RESOLUTIONS)
    actions_low = np.asarray(low["actions"], dtype=np.float32)
    actions_high = np.asarray(high["actions"], dtype=np.float32)
    if actions_low.shape != actions_high.shape:
        raise RuntimeError("paired resolutions produced different action lengths")
    maximum_action_delta = float(np.max(np.abs(actions_low - actions_high), initial=0.0))
    if maximum_action_delta > 1.0e-6:
        raise RuntimeError("paired resolutions produced different expert actions")
    if low["stages"] != high["stages"]:
        raise RuntimeError("paired resolutions produced different task stages")
    if low["successes"] != high["successes"]:
        raise RuntimeError("paired resolutions produced different success sequences")
    normalized = {}
    for resolution, episode in episodes.items():
        normalized[resolution] = (
            np.stack([row.q_true for row in episode["records"]]).astype(np.float32)
            / float(resolution - 1)
        )
    maximum_normalized_q_delta = float(
        np.max(np.abs(normalized[224] - normalized[448]), initial=0.0)
    )
    return {
        "same_action_sequence": True,
        "same_stage_sequence": True,
        "same_success_sequence": True,
        "max_action_abs_delta": maximum_action_delta,
        "max_normalized_q_true_abs_delta": maximum_normalized_q_delta,
        "raw_frames": int(len(low["frames"])),
    }


def _resolution_delta(
    metrics: dict[int, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for grounding in ("nominal", "fcpg"):
        rows: dict[str, float] = {}
        for horizon in range(1, HORIZON + 1):
            key = f"H{horizon}"
            low = metrics[224][grounding][key]["center_epe_224eq_px"]
            high = metrics[448][grounding][key]["center_epe_224eq_px"]
            rows[key] = float((high - low) / max(low, 1.0e-12))
        result[grounding] = rows
    return result


def main() -> int:
    args = _parser().parse_args()
    device = _device(args.device)
    torch.cuda.set_device(device)
    contract = legacy._safe_read_json(args.config)
    legacy.validate_a4_contract(contract)
    task = "towel-wipe-v0"
    task_index = TASKS.index(task)
    instance = legacy._first_test_instances(contract)[task]
    orbit = OrbitCameraView(
        clock_hour=int(args.orbit_clock_hour),
        elevation_degrees=float(args.orbit_elevation_degrees),
        radius_m=float(args.orbit_radius_m),
        base_camera_name=a4_camera_name(task),
    )
    environment_seed = legacy._environment_seed(
        contract, task=task, task_instance=instance
    )
    collection = contract["collection"]
    episodes = {
        resolution: trajectory._collect_full_episode(
            task,
            task_instance=instance,
            environment_seed=environment_seed,
            width=resolution,
            height=resolution,
            benchmark_seed=int(collection["benchmark_seed"]),
            max_raw_steps=int(args.max_raw_steps),
            post_success_steps=int(args.post_success_steps),
            camera_name=a4_camera_name(task),
            orbit_camera_view=orbit,
        )
        for resolution in RESOLUTIONS
    }
    audit = _paired_episode_audit(episodes)

    parent_payload = torch.load(
        args.parent_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    parent = load_camera_action_joint_checkpoint(parent_payload).to(device).eval()
    fcpg = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=device,
    )
    all_metrics: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    grounding_metrics: dict[int, dict[str, dict[str, float]]] = {}
    try:
        for resolution, episode in episodes.items():
            starts = gallery._decision_steps(len(episode["actions"]))
            selected = [episode["records"][int(index)] for index in starts]
            grounding_points = {
                "nominal": np.stack([row.q_nominal for row in selected]).astype(
                    np.float32
                ),
                "fcpg": corrected_points(selected, fcpg).astype(np.float32),
            }
            all_metrics[resolution] = {}
            grounding_metrics[resolution] = {}
            for grounding, points in grounding_points.items():
                grounding_metrics[resolution][grounding] = _current_grounding_metrics(
                    selected,
                    points,
                    resolution=resolution,
                )
                predicted_starts, predictions = gallery._predict_episode_windows(
                    parent,
                    episode,
                    points,
                    task_index=task_index,
                    device=device,
                    orbit_view=orbit,
                )
                if not np.array_equal(starts, predicted_starts):
                    raise RuntimeError("prediction decision grid drifted")
                all_metrics[resolution][grounding] = _equivalent_metrics(
                    episode,
                    starts,
                    predictions,
                    resolution=resolution,
                )
    finally:
        del fcpg, parent
        legacy._clear_runtime_memory(device)

    result = {
        "protocol": "paired_towel_input_resolution_probe_v1",
        "parent_step": int(parent_payload["step"]),
        "task": task,
        "task_instance": int(instance),
        "camera": orbit.label,
        "resolutions": list(RESOLUTIONS),
        "backbone_internal_image_size": 224,
        "backbone_patch_size": 14,
        "paired_episode_audit": audit,
        "current_grounding": {
            str(key): value for key, value in grounding_metrics.items()
        },
        "metrics": {str(key): value for key, value in all_metrics.items()},
        "relative_center_epe_change_448_vs_224": _resolution_delta(all_metrics),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
