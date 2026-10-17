#!/usr/bin/env python3
"""Evaluate frozen goal-conditioned LeWM under grounding and blur interventions.

The physical state, candidate actions, and future outcomes remain fixed.  The
intervention changes only the current visual measurement:

* a synthetic nominal-calibration rotation moves the projected sampling points;
* deterministic Gaussian blur degrades the current RGB frame;
* nominal, FCPG, and oracle points are used to encode the current LeWM state.

The model keeps the recorded actual camera basis for action coordinates.  This
is an observation-update ablation, not a claim that FCPG estimates the complete
camera extrinsic used elsewhere in the dynamics model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter
import torch

from fcpg.geometry import Camera, project_points
from fcpg.lewm_goal_conditioned_metric3d import (
    load_goal_conditioned_checkpoint,
)
from fcpg.lewm_history_conditioned_metric3d_a4 import (
    load_history_conditioned_checkpoint,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter
from scripts.train_lewm_goal_conditioned_direct import (
    GROUNDINGS,
    _encode_states,
    _evaluate,
    _fcpg_q,
    _load,
)


PROTOCOL = "goal_conditioned_visual_robustness_v1"
DEFAULT_AXIS_WORLD = (0.0, 1.0, 0.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/metaworld_projection/moving_goal_reach_push_branches_train.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/lewm_goal_conditioned_direct_seed93001_2k/"
            "goal_conditioned_direct_latest.pt"
        ),
    )
    parser.add_argument(
        "--fcpg-checkpoint",
        type=Path,
        default=Path(
            "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig002/"
            "fcpg_transformer_residual_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/lewm_goal_conditioned_direct_seed93001_2k/"
            "visual_robustness_rotation_blur"
        ),
    )
    parser.add_argument("--rotation-degrees", default="0,5,10")
    parser.add_argument("--blur-sigma", default="0,2,4")
    parser.add_argument("--rotation-axis-world", default="0,1,0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def _parse_float_list(value: str, *, name: str, minimum: float = 0.0) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or not np.isfinite(values).all() or any(item < minimum for item in values):
        raise ValueError(f"{name} must contain finite values >= {minimum}")
    return values


def _parse_axis(value: str) -> np.ndarray:
    axis = np.asarray(
        [float(item.strip()) for item in value.split(",") if item.strip()],
        dtype=np.float64,
    )
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("rotation-axis-world must contain three finite values")
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise ValueError("rotation-axis-world must be non-zero")
    return axis / norm


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index == 1:
        raise ValueError("CUDA1 is disabled")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def axis_angle_rotation_matrix(axis: Sequence[float], degrees: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    if unit.shape != (3,) or not np.isfinite(unit).all():
        raise ValueError("axis must contain three finite values")
    norm = float(np.linalg.norm(unit))
    if norm <= 1.0e-12 or not np.isfinite(float(degrees)):
        raise ValueError("axis must be non-zero and degrees finite")
    x, y, z = unit / norm
    angle = np.deg2rad(float(degrees))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    complement = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ],
            [
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ],
            [
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ],
        ],
        dtype=np.float64,
    )


def rotate_extrinsic_about_world_axis(
    extrinsic_cam_from_world: np.ndarray,
    axis_world: Sequence[float],
    degrees: float,
) -> np.ndarray:
    """Rotate a calibrated camera orientation around its fixed optical center."""

    extrinsic = np.asarray(extrinsic_cam_from_world, dtype=np.float64)
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("extrinsic must have shape [4,4] with finite values")
    rotation_world_to_camera = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    camera_center_world = -rotation_world_to_camera.T @ translation
    delta_world = axis_angle_rotation_matrix(axis_world, degrees)
    rotated_camera_to_world = delta_world @ rotation_world_to_camera.T
    rotated_world_to_camera = rotated_camera_to_world.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotated_world_to_camera
    result[:3, 3] = -rotated_world_to_camera @ camera_center_world
    return result


def global_gaussian_blur(images: np.ndarray, sigma: float) -> np.ndarray:
    source = np.asarray(images)
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError("images must have shape [N,H,W,3] and dtype uint8")
    radius = float(sigma)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("sigma must be finite and non-negative")
    if radius == 0.0:
        return np.array(source, copy=True, order="C")
    return np.stack(
        [
            np.asarray(
                Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius)),
                dtype=np.uint8,
            )
            for image in source
        ]
    )


def nominal_projection_for_rotation(
    data: dict[str, np.ndarray],
    *,
    axis_world: Sequence[float],
    degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    height, width = data["current_rgb"].shape[1:3]
    for xyz, intrinsic, actual_extrinsic in zip(
        data["current_xyz"],
        data["camera_intrinsic"],
        data["true_extrinsic"],
    ):
        nominal_extrinsic = rotate_extrinsic_about_world_axis(
            actual_extrinsic,
            axis_world,
            degrees,
        )
        q, valid = project_points(
            xyz,
            Camera(intrinsic, nominal_extrinsic, int(width), int(height)),
        )
        q_rows.append(np.asarray(q, dtype=np.float32))
        valid_rows.append(np.asarray(valid, dtype=bool))
    return np.stack(q_rows), np.stack(valid_rows)


def _grounding_report(
    q: np.ndarray,
    q_true: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    support = np.asarray(valid, dtype=bool)
    error = np.linalg.norm(np.asarray(q) - np.asarray(q_true), axis=-1)
    supported = error[support]
    return {
        "supported_anchors": int(support.sum()),
        "total_anchors": int(support.size),
        "epe_px": float(supported.mean()) if supported.size else float("nan"),
        "median_px": float(np.median(supported)) if supported.size else float("nan"),
        "p90_px": float(np.percentile(supported, 90.0)) if supported.size else float("nan"),
    }


def _markdown(report: dict[str, Any]) -> str:
    rows = [
        "# Frozen goal-conditioned LeWM visual robustness",
        "",
        "The physical branch outcomes and candidate actions are fixed. The matrix changes only current RGB blur and nominal projection error. The LeWM camera basis remains the recorded actual basis, so this isolates FCPG's observation-update contribution.",
        "",
        "| Rotation | Blur sigma | Grounding | Current EPE (px) | Entity EPE (mm) | Spearman | Top-1 | Selected H8 goal distance (mm) |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in report["conditions"]:
        for grounding in GROUNDINGS:
            current = condition["grounding"][grounding]
            future = condition["future"][grounding]
            rows.append(
                "| {rotation:g} | {blur:g} | {grounding} | {epe:.3f} | {entity:.3f} | {spearman:.4f} | {top1:.3f} | {selected:.3f} |".format(
                    rotation=float(condition["rotation_degrees"]),
                    blur=float(condition["blur_sigma"]),
                    grounding=grounding,
                    epe=float(current["epe_px"]),
                    entity=float(future["entity_epe_mm"]),
                    spearman=float(future["spearman_mean"]),
                    top1=float(future["top1_action_accuracy"]),
                    selected=float(future["selected_actual_goal_distance_h8_mm"]),
                )
            )
    rows.append("")
    summary = report["summary"]
    rows.extend(
        [
            "## Cross-condition summary",
            "",
            f"- Mean current-grounding EPE reduction across drifted cells: {summary['mean_current_epe_reduction_percent']:.2f}%.",
            f"- Mean robot-anchor EPE improvement: {summary['mean_robot_anchor_epe_improvement_mm']:.4f} mm.",
            f"- Mean entity EPE improvement: {summary['mean_entity_epe_improvement_mm']:.4f} mm.",
            f"- Mean Spearman change: {summary['mean_spearman_delta']:+.6f}.",
            f"- Mean selected H8 goal-distance improvement: {summary['mean_selected_goal_distance_improvement_mm']:.4f} mm.",
            "",
        ]
    )
    rows.append("This is a development mechanism matrix, not rollout-success evidence.")
    return "\n".join(rows) + "\n"


def aggregate_summary(conditions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    drifted = [row for row in conditions if float(row["rotation_degrees"]) > 0.0]
    if not drifted:
        raise ValueError("visual-robustness summary requires a drifted condition")
    current_reduction = []
    for row in drifted:
        nominal = float(row["grounding"]["nominal"]["epe_px"])
        fcpg = float(row["grounding"]["fcpg"]["epe_px"])
        if nominal <= 0.0:
            raise ValueError("drifted nominal EPE must be positive")
        current_reduction.append((nominal - fcpg) / nominal)

    def mean_improvement(metric: str) -> float:
        return float(
            np.mean(
                [
                    float(row["future"]["nominal"][metric])
                    - float(row["future"]["fcpg"][metric])
                    for row in drifted
                ]
            )
        )

    blur_oracle_entity = {
        str(float(row["blur_sigma"])): float(row["future"]["oracle"]["entity_epe_mm"])
        for row in conditions
        if float(row["rotation_degrees"]) == 0.0
    }
    return {
        "drifted_condition_count": len(drifted),
        "mean_current_epe_reduction_percent": float(np.mean(current_reduction) * 100.0),
        "mean_robot_anchor_epe_improvement_mm": mean_improvement("robot_anchor_epe_mm"),
        "mean_entity_epe_improvement_mm": mean_improvement("entity_epe_mm"),
        "mean_spearman_delta": float(
            np.mean(
                [
                    float(row["future"]["fcpg"]["spearman_mean"])
                    - float(row["future"]["nominal"]["spearman_mean"])
                    for row in drifted
                ]
            )
        ),
        "mean_selected_goal_distance_improvement_mm": mean_improvement(
            "selected_actual_goal_distance_h8_mm"
        ),
        "oracle_entity_epe_mm_by_blur_at_zero_rotation": blur_oracle_entity,
    }


def main() -> int:
    args = _parser().parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    rotations = _parse_float_list(args.rotation_degrees, name="rotation-degrees")
    blur_sigmas = _parse_float_list(args.blur_sigma, name="blur-sigma")
    axis_world = _parse_axis(args.rotation_axis_world)
    device = _device(args.device)

    dataset_path = args.dataset.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    fcpg_path = args.fcpg_checkpoint.expanduser().resolve()
    data = _load(dataset_path)
    direct_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parent_path = Path(str(direct_payload["parent_checkpoint"])).expanduser().resolve()
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent = load_history_conditioned_checkpoint(parent_payload).to(device).eval()
    model = load_goal_conditioned_checkpoint(direct_payload, parent).to(device).eval()
    corrector, _ = load_fcpg_vit_adapter(fcpg_path, device=device)
    corrector.eval()
    dev_rows = np.flatnonzero(data["split"].astype(str) == "development")
    if not len(dev_rows):
        raise ValueError("dataset has no development rows")

    conditions: list[dict[str, Any]] = []
    for rotation in rotations:
        q_nominal, valid_nominal = nominal_projection_for_rotation(
            data,
            axis_world=axis_world,
            degrees=float(rotation),
        )
        for blur_sigma in blur_sigmas:
            condition_data = dict(data)
            condition_data["current_rgb"] = global_gaussian_blur(
                data["current_rgb"], float(blur_sigma)
            )
            condition_data["current_q_nominal"] = q_nominal
            condition_data["current_valid_nominal"] = valid_nominal
            q_fcpg = _fcpg_q(
                corrector,
                condition_data,
                device,
                batch_size=int(args.batch_size),
            )
            states = _encode_states(
                parent,
                condition_data,
                q_fcpg,
                device,
                batch_size=max(1, int(args.batch_size) // 2),
            )
            q_true = condition_data["current_q_true"][dev_rows]
            valid_true = condition_data["current_valid_true"][dev_rows]
            valid_fcpg = (
                valid_nominal[dev_rows]
                & (q_fcpg[dev_rows, :, 0] >= 0.0)
                & (q_fcpg[dev_rows, :, 1] >= 0.0)
                & (q_fcpg[dev_rows, :, 0] < condition_data["current_rgb"].shape[2])
                & (q_fcpg[dev_rows, :, 1] < condition_data["current_rgb"].shape[1])
            )
            grounding = {
                "nominal": _grounding_report(
                    q_nominal[dev_rows], q_true, valid_nominal[dev_rows] & valid_true
                ),
                "fcpg": _grounding_report(q_fcpg[dev_rows], q_true, valid_fcpg & valid_true),
                "oracle": _grounding_report(q_true, q_true, valid_true),
            }
            future: dict[str, Any] = {}
            for grounding_name in GROUNDINGS:
                future[grounding_name], _ = _evaluate(
                    model,
                    condition_data,
                    states,
                    dev_rows,
                    grounding_name,
                    device,
                    int(args.batch_size),
                )
            condition = {
                "rotation_degrees": float(rotation),
                "blur_sigma": float(blur_sigma),
                "grounding": grounding,
                "future": future,
            }
            conditions.append(condition)
            print(json.dumps(condition), flush=True)

    report = {
        "protocol": PROTOCOL,
        "scope": "development_observation_update_ablation",
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(direct_payload["step"]),
        "checkpoint_seed": int(direct_payload["seed"]),
        "fcpg_checkpoint": str(fcpg_path),
        "rotation_axis_world": axis_world.tolist(),
        "model_camera_basis": "recorded_actual_extrinsic_fixed_across_groundings",
        "future_rgb_used_as_input": False,
        "future_state_used_as_input": False,
        "conditions": conditions,
        "summary": aggregate_summary(conditions),
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "conditions": len(conditions)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
