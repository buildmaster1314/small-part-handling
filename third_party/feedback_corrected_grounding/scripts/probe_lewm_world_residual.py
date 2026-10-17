#!/usr/bin/env python3
"""Probe whether a frozen LeWM latent supports view-invariant 3D motion.

This is deliberately a low-capacity diagnostic, not a replacement readout.  A
ridge probe predicts the residual left by direct Cartesian command integration.
It is trained on the even camera hours and evaluated on the odd camera hours.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import HORIZON, SIGREG_WEIGHT, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    ANCHOR_COUNT,
    LATENT_DIM,
    STATE_SLOTS,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_camera_action_joint_a4 import (  # noqa: E402
    CAMERA_ACTION_PROTOCOL,
    load_camera_action_joint_checkpoint,
)


BRANCH_COUNT = 8
RAW_ACTIONS_PER_TOKEN = 5
RAW_ACTION_DIM = 4
DEFAULT_RAW = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_CHECKPOINT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_phaseaware_a4c_v4_seed40571_10k/"
    "lewm_camera_action_joint_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_phaseaware_a4c_v4_seed40571_10k_world_linear_probe.json"
)


def command_integrated_xyz_numpy(
    current_xyz: np.ndarray, future_action: np.ndarray
) -> np.ndarray:
    """Integrate MetaWorld Cartesian commands into an anchor trajectory."""

    current = np.asarray(current_xyz, dtype=np.float32)
    action = np.asarray(future_action, dtype=np.float32)
    if current.ndim != 3 or current.shape[1:] != (ANCHOR_COUNT, 3):
        raise ValueError("current_xyz must have shape [N,3,3]")
    if action.ndim != 4 or action.shape[1:] != (
        BRANCH_COUNT,
        HORIZON,
        RAW_ACTIONS_PER_TOKEN * RAW_ACTION_DIM,
    ):
        raise ValueError("future_action must have shape [N,8,8,20]")
    raw = action.reshape(
        action.shape[0], BRANCH_COUNT, HORIZON, RAW_ACTIONS_PER_TOKEN, RAW_ACTION_DIM
    )
    token_delta = 0.01 * raw[..., :3].sum(axis=3)
    cumulative = np.cumsum(token_delta, axis=2)
    return current[:, None, None] + cumulative[:, :, :, None]


def project_world_numpy(
    world_xyz: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_cam_from_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project [N,...,3] world points with one calibrated camera per row."""

    points = np.asarray(world_xyz, dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    extrinsic = np.asarray(extrinsic_cam_from_world, dtype=np.float64)
    if points.ndim < 3 or points.shape[0] != k.shape[0] or points.shape[-1] != 3:
        raise ValueError("world_xyz must have shape [N,...,3]")
    if k.shape != (points.shape[0], 3, 3):
        raise ValueError("intrinsic must have shape [N,3,3]")
    if extrinsic.shape != (points.shape[0], 4, 4):
        raise ValueError("extrinsic must have shape [N,4,4]")
    translation_shape = (points.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    translation = extrinsic[:, :3, 3].reshape(translation_shape)
    camera = (
        np.einsum("nij,n...j->n...i", extrinsic[:, :3, :3], points)
        + translation
    )
    homogeneous = np.einsum("nij,n...j->n...i", k, camera)
    depth = homogeneous[..., 2]
    pixels = homogeneous[..., :2] / np.maximum(depth[..., None], 1.0e-8)
    return pixels.astype(np.float32), (depth > 1.0e-6)


def world_to_camera_numpy(
    world_xyz: np.ndarray, extrinsic_cam_from_world: np.ndarray
) -> np.ndarray:
    """Transform [N,...,3] points from world into camera coordinates."""

    points = np.asarray(world_xyz, dtype=np.float64)
    extrinsic = np.asarray(extrinsic_cam_from_world, dtype=np.float64)
    if points.ndim < 3 or points.shape[-1] != 3:
        raise ValueError("world_xyz must have shape [N,...,3]")
    if extrinsic.shape != (points.shape[0], 4, 4):
        raise ValueError("extrinsic must have shape [N,4,4]")
    translation_shape = (points.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    return (
        np.einsum("nij,n...j->n...i", extrinsic[:, :3, :3], points)
        + extrinsic[:, :3, 3].reshape(translation_shape)
    ).astype(np.float32)


def camera_to_world_numpy(
    camera_xyz: np.ndarray, extrinsic_cam_from_world: np.ndarray
) -> np.ndarray:
    """Transform [N,...,3] points from camera into world coordinates."""

    points = np.asarray(camera_xyz, dtype=np.float64)
    extrinsic = np.asarray(extrinsic_cam_from_world, dtype=np.float64)
    if points.ndim < 3 or points.shape[-1] != 3:
        raise ValueError("camera_xyz must have shape [N,...,3]")
    if extrinsic.shape != (points.shape[0], 4, 4):
        raise ValueError("extrinsic must have shape [N,4,4]")
    translation_shape = (points.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    centered = points - extrinsic[:, :3, 3].reshape(translation_shape)
    return np.einsum(
        "nji,n...j->n...i", extrinsic[:, :3, :3], centered
    ).astype(np.float32)


def ridge_solve(
    xtx: np.ndarray,
    xty: np.ndarray,
    *,
    sample_count: int,
    alpha: float,
) -> np.ndarray:
    """Solve a normalized ridge system while leaving the intercept unpenalized."""

    matrix = np.asarray(xtx, dtype=np.float64) / float(sample_count)
    target = np.asarray(xty, dtype=np.float64) / float(sample_count)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("xtx must be square")
    if target.shape != (matrix.shape[0], 3):
        raise ValueError("xty must have shape [feature_dim,3]")
    if int(sample_count) <= 0 or float(alpha) < 0.0:
        raise ValueError("sample_count and alpha are invalid")
    penalty = np.eye(matrix.shape[0], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(matrix + penalty, target)


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-batch-size", type=int, default=42)
    parser.add_argument("--analysis-row-batch-size", type=int, default=24)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-3)
    parser.add_argument(
        "--target-frame",
        choices=("world", "camera"),
        default="world",
        help="Coordinate frame used by the linear residual target.",
    )
    args = parser.parse_args(argv)
    if int(args.cache_batch_size) <= 0 or int(args.analysis_row_batch_size) <= 0:
        parser.error("batch sizes must be positive")
    if float(args.ridge_alpha) < 0.0:
        parser.error("--ridge-alpha must be non-negative")
    return args


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("the 3D probe permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    if payload.get("protocol") != CAMERA_ACTION_PROTOCOL:
        raise ValueError("checkpoint is not the camera-action LeWM")
    if int(payload.get("step", -1)) != 10_000:
        raise ValueError("the 3D probe requires the fixed 10k endpoint")
    if float(payload.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("the 3D probe requires SIGReg=0.02")
    model = load_camera_action_joint_checkpoint(payload).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dict(payload)


@torch.inference_mode()
def _cache_predicted_anchor_latents(
    model: torch.nn.Module,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    count = int(np.asarray(index["task_index"]).shape[0])
    cache = np.empty(
        (count, BRANCH_COUNT, HORIZON, ANCHOR_COUNT, LATENT_DIM),
        dtype=np.float16,
    )
    task_all = np.asarray(index["task_index"], dtype=np.int64)
    action_all = np.asarray(index["future_action"], dtype=np.float32)
    for start in range(0, count, int(batch_size)):
        stop = min(start + int(batch_size), count)
        rows = np.arange(start, stop, dtype=np.int64)
        values = {
            name: torch.from_numpy(
                np.ascontiguousarray(np.asarray(arrays[name])[rows])
            )[:, None]
            for name in (
                "current_rgb",
                "current_proprio",
                "current_q_nominal",
                "current_q_corrected",
                "current_keypoints_xyz",
                "current_valid",
            )
        }
        extrinsic = torch.from_numpy(
            np.ascontiguousarray(
                np.asarray(arrays["camera_extrinsic_cam_from_world"])[rows]
            )
        ).to(device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            current_state = model.encode_observations(
                rgb=values["current_rgb"].to(device),
                proprio=values["current_proprio"].to(device),
                q_nominal=values["current_q_nominal"].to(device),
                q_corrected=values["current_q_corrected"].to(device),
                keypoints_xyz=values["current_keypoints_xyz"].to(device),
                valid=values["current_valid"].to(device),
            )[:, 0]
            expanded_state = current_state[:, None].expand(
                -1, BRANCH_COUNT, -1, -1
            ).reshape(-1, STATE_SLOTS, LATENT_DIM)
            actions = torch.from_numpy(
                np.ascontiguousarray(action_all[start:stop])
            ).reshape(-1, HORIZON, RAW_ACTIONS_PER_TOKEN * RAW_ACTION_DIM)
            actions = actions.to(device=device, dtype=current_state.dtype)
            expanded_extrinsic = extrinsic[:, None].expand(
                -1, BRANCH_COUNT, -1, -1
            ).reshape(-1, 4, 4)
            transformed = model.transform_future_action(
                actions,
                {"camera_extrinsic_cam_from_world": expanded_extrinsic},
            )
            tasks = torch.from_numpy(task_all[start:stop]).to(device)
            tasks = tasks[:, None].expand(-1, BRANCH_COUNT).reshape(-1)
            predicted = model.dynamics(
                expanded_state, transformed, task_index=tasks
            )[:, :, 1:]
        cache[start:stop] = (
            predicted.reshape(
                stop - start,
                BRANCH_COUNT,
                HORIZON,
                ANCHOR_COUNT,
                LATENT_DIM,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float16, copy=False)
        )
        print(
            json.dumps({"cached_rows": stop, "total_rows": count}),
            flush=True,
        )
    return cache


def _categorical_features(task: np.ndarray, row_count: int) -> np.ndarray:
    task_eye = np.eye(len(TASKS), dtype=np.float32)[task]
    task_values = np.broadcast_to(
        task_eye[:, None, None, None],
        (row_count, BRANCH_COUNT, HORIZON, ANCHOR_COUNT, len(TASKS)),
    )
    horizon_eye = np.eye(HORIZON, dtype=np.float32)
    horizon_values = np.broadcast_to(
        horizon_eye[None, None, :, None],
        (row_count, BRANCH_COUNT, HORIZON, ANCHOR_COUNT, HORIZON),
    )
    anchor_eye = np.eye(ANCHOR_COUNT, dtype=np.float32)
    anchor_values = np.broadcast_to(
        anchor_eye[None, None, None],
        (row_count, BRANCH_COUNT, HORIZON, ANCHOR_COUNT, ANCHOR_COUNT),
    )
    return np.concatenate(
        [task_values, horizon_values, anchor_values], axis=-1
    ).reshape(-1, len(TASKS) + HORIZON + ANCHOR_COUNT)


def _continuous_features(
    rows: np.ndarray,
    *,
    current_xyz: np.ndarray,
    command_xyz: np.ndarray,
    latent: np.ndarray | None,
) -> np.ndarray:
    count = int(rows.size)
    current = np.asarray(current_xyz[rows], dtype=np.float32)
    current_expanded = np.broadcast_to(
        current[:, None, None],
        (count, BRANCH_COUNT, HORIZON, ANCHOR_COUNT, 3),
    )
    command_delta = np.asarray(command_xyz[rows], dtype=np.float32) - current_expanded
    values = [current_expanded.reshape(-1, 3), command_delta.reshape(-1, 3)]
    if latent is not None:
        values.append(np.asarray(latent[rows], dtype=np.float32).reshape(-1, LATENT_DIM))
    return np.concatenate(values, axis=-1)


def _target_residual(
    rows: np.ndarray,
    *,
    target_xyz: np.ndarray,
    command_xyz: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(target_xyz[rows], dtype=np.float32)
        - np.asarray(command_xyz[rows], dtype=np.float32)
    ).reshape(-1, 3)


def _fit_feature_scaler(
    rows: np.ndarray,
    *,
    current_xyz: np.ndarray,
    command_xyz: np.ndarray,
    latent: np.ndarray | None,
    row_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = None
    squared = None
    sample_count = 0
    for start in range(0, rows.size, int(row_batch_size)):
        batch_rows = rows[start : start + int(row_batch_size)]
        values = _continuous_features(
            batch_rows,
            current_xyz=current_xyz,
            command_xyz=command_xyz,
            latent=latent,
        ).astype(np.float64)
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            squared = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
        sample_count += int(values.shape[0])
    if total is None or squared is None or sample_count <= 0:
        raise RuntimeError("feature scaler received no samples")
    mean = total / sample_count
    variance = np.maximum(squared / sample_count - np.square(mean), 1.0e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _design_matrix(
    rows: np.ndarray,
    *,
    index: Mapping[str, np.ndarray],
    current_xyz: np.ndarray,
    command_xyz: np.ndarray,
    latent: np.ndarray | None,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    continuous = _continuous_features(
        rows,
        current_xyz=current_xyz,
        command_xyz=command_xyz,
        latent=latent,
    )
    normalized = (continuous - mean[None]) / std[None]
    categorical = _categorical_features(
        np.asarray(index["task_index"], dtype=np.int64)[rows], int(rows.size)
    )
    intercept = np.ones((normalized.shape[0], 1), dtype=np.float32)
    return np.concatenate([normalized, categorical, intercept], axis=1)


def _fit_probe(
    rows: np.ndarray,
    *,
    index: Mapping[str, np.ndarray],
    current_xyz: np.ndarray,
    command_xyz: np.ndarray,
    target_xyz: np.ndarray,
    latent: np.ndarray | None,
    row_batch_size: int,
    alpha: float,
) -> dict[str, np.ndarray]:
    mean, std = _fit_feature_scaler(
        rows,
        current_xyz=current_xyz,
        command_xyz=command_xyz,
        latent=latent,
        row_batch_size=row_batch_size,
    )
    feature_dim = int(mean.size + len(TASKS) + HORIZON + ANCHOR_COUNT + 1)
    xtx = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    xty = np.zeros((feature_dim, 3), dtype=np.float64)
    sample_count = 0
    for start in range(0, rows.size, int(row_batch_size)):
        batch_rows = rows[start : start + int(row_batch_size)]
        design = _design_matrix(
            batch_rows,
            index=index,
            current_xyz=current_xyz,
            command_xyz=command_xyz,
            latent=latent,
            mean=mean,
            std=std,
        ).astype(np.float64)
        target = _target_residual(
            batch_rows, target_xyz=target_xyz, command_xyz=command_xyz
        ).astype(np.float64) / 0.1
        xtx += design.T @ design
        xty += design.T @ target
        sample_count += int(design.shape[0])
    weight = ridge_solve(xtx, xty, sample_count=sample_count, alpha=alpha)
    return {"mean": mean, "std": std, "weight": weight.astype(np.float32)}


def _predict_probe(
    rows: np.ndarray,
    *,
    probe: Mapping[str, np.ndarray],
    index: Mapping[str, np.ndarray],
    current_xyz: np.ndarray,
    command_xyz: np.ndarray,
    latent: np.ndarray | None,
) -> np.ndarray:
    design = _design_matrix(
        rows,
        index=index,
        current_xyz=current_xyz,
        command_xyz=command_xyz,
        latent=latent,
        mean=np.asarray(probe["mean"]),
        std=np.asarray(probe["std"]),
    )
    residual = 0.1 * (design @ np.asarray(probe["weight"]))
    return np.asarray(command_xyz[rows], dtype=np.float32) + residual.reshape(
        rows.size, BRANCH_COUNT, HORIZON, ANCHOR_COUNT, 3
    )


def _view_swap_rows(index: Mapping[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    physical = np.asarray(index["physical_bundle_index"], dtype=np.int64)
    split = np.asarray(index["split"]).astype(str)
    result = rows.copy()
    row_position = {int(row): position for position, row in enumerate(rows.tolist())}
    for physical_id in np.unique(physical[rows]):
        members = np.flatnonzero(
            (physical == physical_id) & (split == split[rows[0]])
        )
        if members.size < 2:
            continue
        ordered = members[np.argsort(np.asarray(index["camera_clock_hour"])[members])]
        positions = np.asarray([row_position[int(row)] for row in ordered], dtype=np.int64)
        result[positions] = np.roll(ordered, -1)
    return result


def _metric_block(
    prediction: np.ndarray,
    rows: np.ndarray,
    *,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    target_xyz: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(target_xyz[rows], dtype=np.float32)
    valid = np.asarray(arrays["future_valid"])[rows].astype(bool)
    world_error = 1000.0 * np.linalg.norm(prediction - target, axis=-1)
    prediction_q, depth_valid = project_world_numpy(
        prediction,
        np.asarray(arrays["camera_intrinsic"])[rows],
        np.asarray(arrays["camera_extrinsic_cam_from_world"])[rows],
    )
    target_q = np.asarray(arrays["future_q_true"])[rows]
    pixel_error = np.linalg.norm(prediction_q - target_q, axis=-1)
    pixel_valid = valid & depth_valid
    tasks = np.asarray(index["task_index"], dtype=np.int64)[rows]
    clocks = np.asarray(index["camera_clock_hour"], dtype=np.int64)[rows]

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        expanded = np.broadcast_to(mask[:, None, None, None], valid.shape)
        world_mask = valid & expanded
        image_mask = pixel_valid & expanded
        return {
            "row_count": int(mask.sum()),
            "world_epe_mm": float(world_error[world_mask].mean()),
            "pixel_epe": float(pixel_error[image_mask].mean()),
        }

    per_task = {
        task: summarize(tasks == task_index)
        for task_index, task in enumerate(TASKS)
    }
    towel_index = TASKS.index("towel-wipe-v0")
    towel_clock03 = summarize((tasks == towel_index) & (clocks == 3))
    return {
        "overall": summarize(np.ones(rows.size, dtype=bool)),
        "task_macro_world_epe_mm": float(
            np.mean([value["world_epe_mm"] for value in per_task.values()])
        ),
        "task_macro_pixel_epe": float(
            np.mean([value["pixel_epe"] for value in per_task.values()])
        ),
        "per_task": per_task,
        "towel_clock03": towel_clock03,
    }


def _cross_view_dispersion_mm(
    prediction: np.ndarray,
    rows: np.ndarray,
    index: Mapping[str, np.ndarray],
) -> float:
    physical = np.asarray(index["physical_bundle_index"], dtype=np.int64)[rows]
    values = []
    for physical_id in np.unique(physical):
        selected = prediction[physical == physical_id]
        if selected.shape[0] < 2:
            continue
        center = selected.mean(axis=0, keepdims=True)
        values.append(np.linalg.norm(selected - center, axis=-1).reshape(-1))
    if not values:
        raise RuntimeError("cross-view dispersion has no multiview group")
    return float(1000.0 * np.concatenate(values).mean())


def _markdown(result: Mapping[str, Any]) -> str:
    metrics = result["development"]
    lines = [
        "# Frozen LeWM 3D linear probe",
        "",
        "| Readout | Task-macro world EPE (mm) | Pixel EPE (px) | Towel clock03 world (mm) | Towel clock03 pixel (px) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("hold", "command", "geometry", "lewm", "lewm_cyclic", "lewm_view_swap"):
        value = metrics[name]
        lines.append(
            f"| {name} | {value['task_macro_world_epe_mm']:.3f} | "
            f"{value['task_macro_pixel_epe']:.3f} | "
            f"{value['towel_clock03']['world_epe_mm']:.3f} | "
            f"{value['towel_clock03']['pixel_epe']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"LeWM gain over geometry: {result['diagnostic']['lewm_gain_over_geometry_percent']:.2f}%",
            f"Cyclic-latent penalty: {result['diagnostic']['cyclic_penalty_mm']:.3f} mm",
            f"Cross-view dispersion: {result['diagnostic']['cross_view_dispersion_mm']:.3f} mm",
            f"Diagnostic decision: **{result['diagnostic']['decision']}**",
            "",
            "This is a development-only linear sufficiency probe, not a paper result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    raw_path = args.raw_dataset.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    required = ("camera_intrinsic", "camera_extrinsic_cam_from_world")
    if any(name not in arrays for name in required):
        raise ValueError("the 3D probe requires cached camera matrices")
    split_hours = {
        split: sorted(
            set(
                np.asarray(index["camera_clock_hour"])[
                    np.asarray(index["split"]).astype(str) == split
                ].tolist()
            )
        )
        for split in ("train", "development")
    }
    if split_hours != {
        "train": [2, 4, 6, 8, 10, 12],
        "development": [1, 3, 5, 7, 9, 11],
    }:
        raise ValueError("camera split is not even-train/odd-development")
    model, payload = _load_model(checkpoint_path, device)
    checkpoint_raw = Path(str(payload.get("raw_dataset", ""))).resolve()
    if checkpoint_raw != raw_path:
        raise ValueError("checkpoint and probe raw dataset differ")
    latent = _cache_predicted_anchor_latents(
        model,
        index,
        arrays,
        device=device,
        batch_size=int(args.cache_batch_size),
    )
    del model
    torch.cuda.empty_cache()

    current_world_xyz = np.asarray(
        arrays["current_keypoints_xyz"], dtype=np.float32
    )
    target_world_xyz = np.asarray(
        arrays["future_keypoints_xyz"], dtype=np.float32
    )
    command_world_xyz = command_integrated_xyz_numpy(
        current_world_xyz, np.asarray(index["future_action"], dtype=np.float32)
    )
    extrinsic = np.asarray(
        arrays["camera_extrinsic_cam_from_world"], dtype=np.float32
    )
    if args.target_frame == "camera":
        current_xyz = world_to_camera_numpy(current_world_xyz, extrinsic)
        target_xyz = world_to_camera_numpy(target_world_xyz, extrinsic)
        command_xyz = world_to_camera_numpy(command_world_xyz, extrinsic)
    else:
        current_xyz = current_world_xyz
        target_xyz = target_world_xyz
        command_xyz = command_world_xyz
    split = np.asarray(index["split"]).astype(str)
    train_rows = np.flatnonzero(split == "train")
    development_rows = np.flatnonzero(split == "development")
    geometry_probe = _fit_probe(
        train_rows,
        index=index,
        current_xyz=current_xyz,
        command_xyz=command_xyz,
        target_xyz=target_xyz,
        latent=None,
        row_batch_size=int(args.analysis_row_batch_size),
        alpha=float(args.ridge_alpha),
    )
    lewm_probe = _fit_probe(
        train_rows,
        index=index,
        current_xyz=current_xyz,
        command_xyz=command_xyz,
        target_xyz=target_xyz,
        latent=latent,
        row_batch_size=int(args.analysis_row_batch_size),
        alpha=float(args.ridge_alpha),
    )
    current_development = current_xyz[development_rows]
    hold_prediction = np.broadcast_to(
        current_development[:, None, None],
        (
            development_rows.size,
            BRANCH_COUNT,
            HORIZON,
            ANCHOR_COUNT,
            3,
        ),
    ).copy()
    command_prediction = command_xyz[development_rows]
    geometry_prediction = _predict_probe(
        development_rows,
        probe=geometry_probe,
        index=index,
        current_xyz=current_xyz,
        command_xyz=command_xyz,
        latent=None,
    )
    lewm_prediction = _predict_probe(
        development_rows,
        probe=lewm_probe,
        index=index,
        current_xyz=current_xyz,
        command_xyz=command_xyz,
        latent=latent,
    )
    cyclic_latent = latent.copy()
    cyclic_latent[:, np.arange(BRANCH_COUNT)] = latent[
        :, np.roll(np.arange(BRANCH_COUNT), -1)
    ]
    cyclic_prediction = _predict_probe(
        development_rows,
        probe=lewm_probe,
        index=index,
        current_xyz=current_xyz,
        command_xyz=command_xyz,
        latent=cyclic_latent,
    )
    view_swap = _view_swap_rows(index, development_rows)
    view_swapped_latent = latent.copy()
    view_swapped_latent[development_rows] = latent[view_swap]
    view_swap_prediction = _predict_probe(
        development_rows,
        probe=lewm_probe,
        index=index,
        current_xyz=current_xyz,
        command_xyz=command_xyz,
        latent=view_swapped_latent,
    )
    frame_predictions = {
        "hold": hold_prediction,
        "command": command_prediction,
        "geometry": geometry_prediction,
        "lewm": lewm_prediction,
        "lewm_cyclic": cyclic_prediction,
        "lewm_view_swap": view_swap_prediction,
    }
    predictions = (
        {
            name: camera_to_world_numpy(value, extrinsic[development_rows])
            for name, value in frame_predictions.items()
        }
        if args.target_frame == "camera"
        else frame_predictions
    )
    development = {
        name: _metric_block(
            prediction,
            development_rows,
            index=index,
            arrays=arrays,
            target_xyz=target_world_xyz,
        )
        for name, prediction in predictions.items()
    }
    geometry_error = float(development["geometry"]["task_macro_world_epe_mm"])
    lewm_error = float(development["lewm"]["task_macro_world_epe_mm"])
    cyclic_error = float(development["lewm_cyclic"]["task_macro_world_epe_mm"])
    gain = 100.0 * (geometry_error - lewm_error) / geometry_error
    cyclic_penalty = cyclic_error - lewm_error
    dispersion = _cross_view_dispersion_mm(
        lewm_prediction, development_rows, index
    )
    decision = (
        "3D_LATENT_SIGNAL"
        if gain >= 5.0 and cyclic_penalty > 0.5
        else "NO_RELIABLE_3D_LATENT_SIGNAL"
    )
    result = {
        "schema_version": 1,
        "scope": "development_only_linear_3d_sufficiency_probe",
        "formal_claim_allowed": False,
        "raw_dataset": str(raw_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "sigreg_weight": float(payload["sigreg_weight"]),
        "target_coordinate_frame": str(args.target_frame),
        "ridge_alpha": float(args.ridge_alpha),
        "train_camera_hours": split_hours["train"],
        "development_camera_hours": split_hours["development"],
        "development": development,
        "diagnostic": {
            "lewm_gain_over_geometry_percent": gain,
            "cyclic_penalty_mm": cyclic_penalty,
            "cross_view_dispersion_mm": dispersion,
            "decision": decision,
            "criterion": "gain>=5% and cyclic penalty>0.5mm",
        },
    }
    for value in _walk_floats(result):
        if not math.isfinite(value):
            raise RuntimeError("probe output contains a non-finite value")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result["diagnostic"], sort_keys=True), flush=True)
    print(output)
    return 0


def _walk_floats(value: Any):
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_floats(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_floats(nested)
    elif isinstance(value, float):
        yield value


if __name__ == "__main__":
    raise SystemExit(main())
