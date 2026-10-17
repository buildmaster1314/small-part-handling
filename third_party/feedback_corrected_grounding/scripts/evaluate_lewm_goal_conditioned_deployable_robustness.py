#!/usr/bin/env python3
"""Evaluate deployable goal-conditioned LeWM under drift and visual blur."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from fcpg.geometry import Camera, project_points
from fcpg.lewm_goal_conditioned_deployable import load_deployable_checkpoint
from fcpg.lewm_history_conditioned_metric3d_a4 import (
    load_history_conditioned_checkpoint,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter
from scripts.evaluate_lewm_goal_conditioned_visual_robustness import (
    _grounding_report,
    _parse_axis,
    _parse_float_list,
    global_gaussian_blur,
    rotate_extrinsic_about_world_axis,
)
from scripts.train_lewm_goal_conditioned_deployable import evaluate_model
from scripts.train_lewm_goal_conditioned_direct import (
    GROUNDINGS,
    _encode_states,
    _fcpg_q,
    _load,
)


PROTOCOL = "goal_conditioned_deployable_visual_robustness_v1"
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
            "outputs/lewm_goal_conditioned_deployable_seed93011_2k/"
            "goal_conditioned_deployable_latest.pt"
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
            "outputs/lewm_goal_conditioned_deployable_seed93011_2k/"
            "visual_robustness_rotation_blur"
        ),
    )
    parser.add_argument("--rotation-degrees", default="0,5,10")
    parser.add_argument("--blur-sigma", default="0,2,4")
    parser.add_argument("--rotation-axis-world", default="0,1,0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index == 1:
        raise ValueError("CUDA1 is disabled")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def nominal_camera_for_rotation(
    data: dict[str, np.ndarray],
    *,
    axis_world: Sequence[float],
    degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    extrinsic_rows: list[np.ndarray] = []
    height, width = data["current_rgb"].shape[1:3]
    for xyz, intrinsic, actual_extrinsic in zip(
        data["current_xyz"], data["camera_intrinsic"], data["true_extrinsic"]
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
        extrinsic_rows.append(np.asarray(nominal_extrinsic, dtype=np.float32))
    return np.stack(q_rows), np.stack(valid_rows), np.stack(extrinsic_rows)


def aggregate_summary(conditions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    drifted = [row for row in conditions if float(row["rotation_degrees"]) > 0.0]
    if not drifted:
        raise ValueError("summary requires at least one drifted condition")

    def future_improvement(metric: str, *, task: str | None = None) -> float:
        values = []
        for row in drifted:
            nominal = row["future"]["nominal"]
            fcpg = row["future"]["fcpg"]
            if task is not None:
                nominal = nominal["by_task"][task]
                fcpg = fcpg["by_task"][task]
            values.append(float(nominal[metric]) - float(fcpg[metric]))
        return float(np.mean(values))

    current_reductions = []
    for row in drifted:
        nominal = float(row["grounding"]["nominal"]["epe_px"])
        fcpg = float(row["grounding"]["fcpg"]["epe_px"])
        current_reductions.append((nominal - fcpg) / nominal)
    return {
        "drifted_condition_count": len(drifted),
        "mean_current_epe_reduction_percent": float(
            np.mean(current_reductions) * 100.0
        ),
        "mean_robot_anchor_epe_improvement_mm": future_improvement(
            "robot_anchor_epe_mm"
        ),
        "mean_entity_epe_improvement_mm": future_improvement("entity_epe_mm"),
        "mean_spearman_delta": -future_improvement("spearman_mean"),
        "mean_selected_goal_distance_improvement_mm": future_improvement(
            "selected_actual_goal_distance_h8_mm"
        ),
        "by_task": {
            task: {
                "mean_entity_epe_improvement_mm": future_improvement(
                    "entity_epe_mm", task=task
                ),
                "mean_spearman_delta": -future_improvement(
                    "spearman_mean", task=task
                ),
                "mean_selected_goal_distance_improvement_mm": future_improvement(
                    "selected_actual_goal_distance_h8_mm", task=task
                ),
            }
            for task in ("reach-v3", "push-v3")
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Deployable goal-conditioned LeWM visual robustness",
        "",
        "The LeWM backbone is frozen. The deployable head receives no current-entity ground truth and no actual camera extrinsic.",
        "",
        "| Rotation | Blur | Grounding | Current EPE px | Entity EPE mm | Spearman | Top-1 | Selected H8 mm |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in report["conditions"]:
        for grounding in GROUNDINGS:
            current = condition["grounding"][grounding]
            future = condition["future"][grounding]
            lines.append(
                "| {rotation:g} | {blur:g} | {grounding} | {current:.3f} | {entity:.3f} | {spearman:.4f} | {top1:.3f} | {selected:.3f} |".format(
                    rotation=float(condition["rotation_degrees"]),
                    blur=float(condition["blur_sigma"]),
                    grounding=grounding,
                    current=float(current["epe_px"]),
                    entity=float(future["entity_epe_mm"]),
                    spearman=float(future["spearman_mean"]),
                    top1=float(future["top1_action_accuracy"]),
                    selected=float(future["selected_actual_goal_distance_h8_mm"]),
                )
            )
    summary = report["summary"]
    lines.extend(
        [
            "",
            "## Drifted-cell mean FCPG improvement",
            "",
            f"- Current grounding EPE reduction: {summary['mean_current_epe_reduction_percent']:.2f}%.",
            f"- Robot-anchor EPE: {summary['mean_robot_anchor_epe_improvement_mm']:+.4f} mm.",
            f"- Entity EPE: {summary['mean_entity_epe_improvement_mm']:+.4f} mm.",
            f"- Spearman: {summary['mean_spearman_delta']:+.6f}.",
            f"- Selected H8 goal distance: {summary['mean_selected_goal_distance_improvement_mm']:+.4f} mm.",
            "",
        ]
    )
    for task, values in summary["by_task"].items():
        lines.append(
            f"- {task}: entity {values['mean_entity_epe_improvement_mm']:+.4f} mm, "
            f"Spearman {values['mean_spearman_delta']:+.6f}, selected H8 "
            f"{values['mean_selected_goal_distance_improvement_mm']:+.4f} mm."
        )
    lines.extend(
        [
            "",
            "This is a development mechanism matrix, not rollout-success evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parser().parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    rotations = _parse_float_list(args.rotation_degrees, name="rotation-degrees")
    blur_sigmas = _parse_float_list(args.blur_sigma, name="blur-sigma")
    axis_world = _parse_axis(args.rotation_axis_world)
    device = _device(args.device)

    data = _load(args.dataset.expanduser().resolve())
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parent_path = Path(str(checkpoint["parent_checkpoint"])).expanduser().resolve()
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent = load_history_conditioned_checkpoint(parent_payload).to(device).eval()
    model = load_deployable_checkpoint(checkpoint, parent).to(device).eval()
    corrector, _ = load_fcpg_vit_adapter(
        args.fcpg_checkpoint.expanduser().resolve(), device=device
    )
    corrector.eval()
    dev_rows = np.flatnonzero(data["split"].astype(str) == "development")
    if not len(dev_rows):
        raise ValueError("dataset has no development rows")

    conditions: list[dict[str, Any]] = []
    for rotation in rotations:
        q_nominal, valid_nominal, nominal_extrinsic = nominal_camera_for_rotation(
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
                "fcpg": _grounding_report(
                    q_fcpg[dev_rows], q_true, valid_fcpg & valid_true
                ),
                "oracle": _grounding_report(q_true, q_true, valid_true),
            }
            future = {
                grounding_name: evaluate_model(
                    model,
                    condition_data,
                    states,
                    dev_rows,
                    grounding_name,
                    nominal_extrinsic,
                    device,
                    int(args.batch_size),
                )
                for grounding_name in GROUNDINGS
            }
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
        "scope": "development_deployable_input_ablation",
        "dataset": str(args.dataset.expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "current_entity_ground_truth_input": False,
        "actual_camera_extrinsic_input": False,
        "goal_is_known_task_input": True,
        "future_rgb_used_as_input": False,
        "future_state_used_as_input": False,
        "rotation_axis_world": axis_world.tolist(),
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
