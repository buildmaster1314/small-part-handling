#!/usr/bin/env python3
"""Evaluate recursive LeWM-to-FCPG search-center transport under camera drift."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    HORIZON,
    RAW_STEPS_PER_MODEL_STEP,
    TASKS,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    a4_camera_name,
    load_frozen_a4_state_encoder,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    corrected_points,
    grounded_valid_mask,
)
from fcpg.lewm_orbit_camera import OrbitCameraView  # noqa: E402
from fcpg.lewm_pixel_observer_baselines import (  # noqa: E402
    LinearConstantVelocityKalmanBaseline,
    fcpg_measurement,
)
from scripts import render_lewm_action_branch_a4_accuracy_gallery as legacy  # noqa: E402
from scripts import render_lewm_action_branch_a4_trajectory_comparison as trajectory  # noqa: E402
from scripts import render_lewm_action_prefix_joint_a4_accuracy_gallery as gallery  # noqa: E402


DEFAULT_BASE = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG = ROOT / (
    "outputs/transformer_residual/a4_multiview_projection_seed40601_2k/"
    "fcpg_a4_multiview_best.pt"
)
DEFAULT_LEWM = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_metric3d_world_history_seed40583_10k/"
    "lewm_history_conditioned_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/recursive_grounding_ramp_pilot_v1"
)
METHODS = (
    "one_shot",
    "hold_last",
    "constant_velocity",
    "linear_kalman",
    "lewm_recursive",
)
CV_MAX_DELTA_PX = 16.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=legacy.DEFAULT_CONFIG)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_LEWM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--test-instance-rank", type=int, default=4)
    parser.add_argument("--orbit-clock-hour", type=int, default=3)
    parser.add_argument("--orbit-elevation-degrees", type=float, default=35.0)
    parser.add_argument("--orbit-radius-m", type=float, default=1.1)
    parser.add_argument("--ramp-start-raw-step", type=int, default=10)
    parser.add_argument("--ramp-end-raw-step", type=int, default=50)
    parser.add_argument("--max-rotation-degrees", type=float, default=10.0)
    parser.add_argument(
        "--camera-drift-axis-world",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--max-raw-steps", type=int, default=legacy.DEFAULT_MAX_RAW_STEPS)
    parser.add_argument(
        "--post-success-steps",
        type=int,
        default=legacy.DEFAULT_POST_SUCCESS_RAW_STEPS,
    )
    parser.add_argument("--transport-gate-low-px", type=float, default=4.0)
    parser.add_argument("--transport-gate-high-px", type=float, default=8.0)
    parser.add_argument(
        "--observation-interval-model-steps",
        type=int,
        default=1,
        help="number of five-raw-step model horizons between visual updates",
    )
    parser.add_argument(
        "--summary-minimum-rotation-degrees",
        type=float,
        help="minimum drift magnitude included in primary summary; auto is 0 for clean and 8 otherwise",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.device not in {"cuda", "cuda:0"}:
        raise ValueError("recursive grounding evaluation permits only CUDA device 0")
    if not 0 <= int(args.test_instance_rank) <= 11:
        raise ValueError("test instance rank must be in 0..11")
    if int(args.orbit_clock_hour) not in range(1, 13):
        raise ValueError("orbit clock hour must be in 1..12")
    if int(args.ramp_start_raw_step) < 0:
        raise ValueError("ramp start must be non-negative")
    if int(args.ramp_end_raw_step) <= int(args.ramp_start_raw_step):
        raise ValueError("ramp end must be greater than ramp start")
    if not np.isfinite(float(args.max_rotation_degrees)):
        raise ValueError("maximum camera rotation must be finite")
    axis = np.asarray(args.camera_drift_axis_world, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0.0:
        raise ValueError("camera drift axis must be one finite non-zero 3D vector")
    if not 0.0 <= float(args.transport_gate_low_px) < float(
        args.transport_gate_high_px
    ):
        raise ValueError("transport gate thresholds must satisfy 0 <= low < high")
    if not 1 <= int(args.observation_interval_model_steps) <= HORIZON:
        raise ValueError("observation interval must be in 1..HORIZON")
    if args.summary_minimum_rotation_degrees is not None:
        minimum = float(args.summary_minimum_rotation_degrees)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("summary minimum rotation must be finite and non-negative")


def linear_camera_rotation_degrees(
    raw_step: int,
    *,
    start_raw_step: int,
    end_raw_step: int,
    maximum_degrees: float,
) -> float:
    """Return a bounded piecewise-linear 0-to-maximum camera drift."""

    step = int(raw_step)
    start = int(start_raw_step)
    end = int(end_raw_step)
    if start < 0 or end <= start:
        raise ValueError("camera ramp requires 0 <= start < end")
    maximum = float(maximum_degrees)
    if not np.isfinite(maximum):
        raise ValueError("camera ramp maximum must be finite")
    if step <= start:
        return 0.0
    if step >= end:
        return maximum
    return maximum * float(step - start) / float(end - start)


def _task_macro(rows: list[dict[str, Any]], field: str, *, minimum_drift: float) -> float:
    task_values = []
    for task in sorted({str(row["task"]) for row in rows}):
        values = [
            float(row[field])
            for row in rows
            if str(row["task"]) == task
            and float(row["camera_rotation_degrees"]) >= minimum_drift
        ]
        if values:
            task_values.append(float(np.mean(values)))
    if not task_values:
        raise ValueError("no rows satisfy the requested drift range")
    return float(np.mean(task_values))


def summarize_rows(rows: list[dict[str, Any]], *, severe_degrees: float = 8.0) -> dict[str, Any]:
    if not rows:
        raise ValueError("recursive grounding summary requires rows")
    methods = tuple(sorted({str(row["method"]) for row in rows}))
    if set(methods) != set(METHODS):
        raise ValueError("recursive grounding rows contain an unexpected method set")
    all_tasks = tuple(sorted({str(row["task"]) for row in rows}))
    severe_tasks = tuple(
        task
        for task in all_tasks
        if any(
            str(row["task"]) == task
            and float(row["camera_rotation_degrees"]) >= float(severe_degrees)
            for row in rows
        )
    )
    if not severe_tasks:
        raise ValueError("no task contains severe-drift rows")
    per_method: dict[str, dict[str, float]] = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        severe = [
            row
            for row in selected
            if float(row["camera_rotation_degrees"]) >= float(severe_degrees)
        ]
        if not severe:
            raise ValueError(f"method {method} has no severe-drift rows")
        per_method[method] = {
            "all_task_macro_corrected_epe_px": _task_macro(
                selected, "corrected_epe_px", minimum_drift=-np.inf
            ),
            "severe_task_macro_center_epe_px": _task_macro(
                selected, "center_epe_px", minimum_drift=float(severe_degrees)
            ),
            "severe_task_macro_corrected_epe_px": _task_macro(
                selected, "corrected_epe_px", minimum_drift=float(severe_degrees)
            ),
            "severe_capture_at_5px": float(
                np.mean([float(row["corrected_epe_px"]) <= 5.0 for row in severe])
            ),
            "severe_capture_at_10px": float(
                np.mean([float(row["corrected_epe_px"]) <= 10.0 for row in severe])
            ),
        }
    baseline = per_method["one_shot"]["severe_task_macro_corrected_epe_px"]
    recursive = per_method["lewm_recursive"]["severe_task_macro_corrected_epe_px"]
    return {
        "schema_version": 1,
        "protocol": "a4_recursive_lewm_fcpg_camera_ramp_v1",
        "severe_rotation_degrees": float(severe_degrees),
        "row_count": len(rows),
        "task_count": len(all_tasks),
        "severe_evaluable_task_count": len(severe_tasks),
        "severe_unevaluable_tasks": sorted(set(all_tasks) - set(severe_tasks)),
        "methods": per_method,
        "severe_recursive_vs_one_shot_relative_improvement": (
            baseline - recursive
        )
        / max(baseline, 1.0e-12),
        "severe_recursive_vs_one_shot_absolute_improvement_px": baseline
        - recursive,
        "per_task": {
            task: {
                method: float(
                    np.mean(
                        [
                            float(row["corrected_epe_px"])
                            for row in rows
                            if row["task"] == task
                            and row["method"] == method
                            and float(row["camera_rotation_degrees"])
                            >= float(severe_degrees)
                        ]
                    )
                )
                for method in METHODS
            }
            for task in severe_tasks
        },
    }


def _centers_for_step(
    nominal: np.ndarray,
    residual_history: Mapping[str, list[np.ndarray]],
    recursive_prior: np.ndarray | None,
    kalman_residual_prior: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    nominal_value = np.asarray(nominal, dtype=np.float32)
    if nominal_value.shape != (3, 2):
        raise ValueError("nominal anchor center must have shape [3,2]")
    result = {"one_shot": nominal_value}
    hold_history = residual_history["hold_last"]
    result["hold_last"] = (
        nominal_value + hold_history[-1] if hold_history else nominal_value
    )
    velocity_history = residual_history["constant_velocity"]
    if len(velocity_history) >= 2:
        velocity = np.clip(
            velocity_history[-1] - velocity_history[-2],
            -CV_MAX_DELTA_PX,
            CV_MAX_DELTA_PX,
        )
        result["constant_velocity"] = (
            nominal_value + velocity_history[-1] + velocity
        )
    elif velocity_history:
        result["constant_velocity"] = nominal_value + velocity_history[-1]
    else:
        result["constant_velocity"] = nominal_value
    result["linear_kalman"] = (
        nominal_value
        if kalman_residual_prior is None
        else nominal_value + np.asarray(kalman_residual_prior)
    )
    result["lewm_recursive"] = (
        nominal_value if recursive_prior is None else np.asarray(recursive_prior)
    )
    for method, value in result.items():
        if value.shape != (3, 2) or not np.isfinite(value).all():
            raise ValueError(f"{method} search center is invalid")
    return result


@torch.inference_mode()
def _predict_prior(
    model: Any,
    episode: Mapping[str, Any],
    record: Any,
    corrected: np.ndarray,
    *,
    task_index: int,
    raw_step: int,
    device: torch.device,
    gate_low_px: float,
    gate_high_px: float,
    horizon_steps: int,
) -> np.ndarray:
    current = np.asarray(corrected, dtype=np.float32)
    if current.shape != (3, 2) or not np.isfinite(current).all():
        raise ValueError("recursive current grounding must be finite [3,2]")
    future_actions = gallery._chunk_future_actions(
        np.asarray(episode["actions"], dtype=np.float32),
        np.asarray([raw_step], dtype=np.int64),
    )
    history_actions = gallery._chunk_history_actions(
        np.asarray(episode["actions"], dtype=np.float32),
        np.asarray([raw_step], dtype=np.int64),
    )
    rgb = torch.from_numpy(np.asarray(record.rgb, dtype=np.float32))[None, None]
    proprio = torch.from_numpy(np.asarray(record.proprio, dtype=np.float32))[None, None]
    nominal = torch.from_numpy(np.asarray(record.q_nominal, dtype=np.float32))[None, None]
    grounded = torch.from_numpy(current)[None, None]
    xyz = torch.from_numpy(np.asarray(record.keypoints_xyz, dtype=np.float32))[None, None]
    valid_np = grounded_valid_mask(
        np.asarray(record.valid_nominal, dtype=bool)[None],
        current[None],
        width=int(rgb.shape[-1]),
        height=int(rgb.shape[-2]),
    )
    valid = torch.from_numpy(valid_np)[None]
    extrinsic = torch.as_tensor(
        gallery._model_camera_extrinsic(episode),
        dtype=torch.float32,
        device=device,
    )[None]
    task = torch.full((1,), int(task_index), dtype=torch.long, device=device)
    raw_future = torch.from_numpy(future_actions).to(device=device, dtype=torch.float32)
    raw_history = torch.from_numpy(history_actions).to(device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        state = model.encode_observations(
            rgb=rgb.to(device),
            proprio=proprio.to(device),
            q_nominal=nominal.to(device),
            q_corrected=grounded.to(device),
            keypoints_xyz=xyz.to(device),
            valid=valid.to(device),
        )[:, 0]
        predicted_state = model.predict_with_history(
            current_state=state,
            history_action=raw_history.to(dtype=state.dtype),
            future_action=raw_future.to(dtype=state.dtype),
            task_index=task,
            batch={"camera_extrinsic_cam_from_world": extrinsic},
        )
        intrinsic = torch.as_tensor(
            np.asarray(episode["camera_intrinsic"], dtype=np.float32),
            device=device,
        )[None]
        predicted = model.decode_3d(
            predicted_state=predicted_state,
            current_state=state,
            current_keypoints_xyz=xyz[:, 0].to(device),
            future_action=raw_future.to(dtype=state.dtype),
            task_index=task,
            camera_intrinsic=intrinsic,
            extrinsic_cam_from_world=extrinsic,
        ).predicted_q_px
        transported = gallery._transport_current_grounding_residual(
            predicted,
            nominal.to(device),
            grounded.to(device),
            mode="magnitude-gated",
            gate_low_px=float(gate_low_px),
            gate_high_px=float(gate_high_px),
        )
    horizon = int(horizon_steps)
    if not 1 <= horizon <= HORIZON:
        raise ValueError("prediction horizon must be in 1..HORIZON")
    result = (
        transported[0, horizon - 1]
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if result.shape != (3, 2) or not np.isfinite(result).all():
        raise RuntimeError("LeWM prior is not finite [3,2]")
    return result


def _evaluate_episode(
    episode: Mapping[str, Any],
    *,
    fcpg: Any,
    model: Any,
    device: torch.device,
    gate_low_px: float,
    gate_high_px: float,
    observation_interval_model_steps: int,
) -> list[dict[str, Any]]:
    task = str(episode["task"])
    interval = int(observation_interval_model_steps)
    if not 1 <= interval <= HORIZON:
        raise ValueError("observation interval must be in 1..HORIZON")
    starts = gallery._decision_steps(len(episode["actions"]))[::interval]
    residual_history: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    recursive_prior: np.ndarray | None = None
    kalman = LinearConstantVelocityKalmanBaseline()
    kalman_state = kalman.initialize(torch.zeros(1, 3, 2, dtype=torch.float32))
    rows: list[dict[str, Any]] = []
    rotations = np.asarray(episode["camera_rotation_degrees_by_frame"], dtype=np.float32)
    for decision_index, raw_step_value in enumerate(starts):
        raw_step = int(raw_step_value)
        record = episode["records"][raw_step]
        kalman_transition = kalman.prior(kalman_state)
        kalman_residual_prior = (
            None
            if decision_index == 0
            else kalman_transition.residual_px[0].cpu().numpy()
        )
        centers = _centers_for_step(
            np.asarray(record.q_nominal, dtype=np.float32),
            residual_history,
            recursive_prior,
            kalman_residual_prior,
        )
        center_batch = np.stack([centers[method] for method in METHODS]).astype(
            np.float32, copy=False
        )
        corrected_batch = corrected_points(
            [record] * len(METHODS),
            fcpg,
            centers_px=center_batch,
        )
        truth = np.asarray(record.q_true, dtype=np.float32)
        for method, center, corrected in zip(
            METHODS, center_batch, corrected_batch, strict=True
        ):
            residual_history[method].append(
                corrected.copy() - np.asarray(record.q_nominal, dtype=np.float32)
            )
            rows.append(
                {
                    "task": task,
                    "task_instance": int(episode["task_instance"]),
                    "environment_seed": int(episode["environment_seed"]),
                    "decision_index": decision_index,
                    "raw_step": raw_step,
                    "camera_rotation_degrees": float(rotations[raw_step]),
                    "method": method,
                    "center_epe_px": float(
                        np.linalg.norm(center - truth, axis=-1).mean()
                    ),
                    "corrected_epe_px": float(
                        np.linalg.norm(corrected - truth, axis=-1).mean()
                    ),
                }
            )
        kalman_corrected = corrected_batch[METHODS.index("linear_kalman")]
        kalman_valid = grounded_valid_mask(
            np.asarray(record.valid_nominal, dtype=bool)[None],
            kalman_corrected[None],
            width=int(record.rgb.shape[-1]),
            height=int(record.rgb.shape[-2]),
        )
        kalman_update = kalman.update(
            kalman_transition,
            fcpg_measurement(
                torch.from_numpy(
                    (
                        kalman_corrected
                        - np.asarray(record.q_nominal, dtype=np.float32)
                    )[None]
                ),
                torch.from_numpy(kalman_valid),
            ),
        )
        kalman_state = kalman_update.state
        recursive_prior = _predict_prior(
            model,
            episode,
            record,
            corrected_batch[METHODS.index("lewm_recursive")],
            task_index=TASKS.index(task),
            raw_step=raw_step,
            device=device,
            gate_low_px=gate_low_px,
            gate_high_px=gate_high_px,
            horizon_steps=interval,
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty recursive grounding table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    device = gallery._cuda0(args.device)
    torch.cuda.set_device(device)
    contract = legacy._safe_read_json(args.config)
    validate_a4_contract(contract)
    tasks = tuple(args.task) if args.task else TASKS
    instances = legacy._test_instances_by_rank(
        contract, rank=int(args.test_instance_rank)
    )
    collection = contract["collection"]
    axis = np.asarray(args.camera_drift_axis_world, dtype=np.float64)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for task in tasks:
        task_instance = int(instances[task])
        environment_seed = legacy._environment_seed(
            contract, task=task, task_instance=task_instance
        )
        camera_name = a4_camera_name(task)
        orbit = OrbitCameraView(
            clock_hour=int(args.orbit_clock_hour),
            elevation_degrees=float(args.orbit_elevation_degrees),
            radius_m=float(args.orbit_radius_m),
            base_camera_name=camera_name,
        )
        schedule = lambda step, args=args: linear_camera_rotation_degrees(
            step,
            start_raw_step=int(args.ramp_start_raw_step),
            end_raw_step=int(args.ramp_end_raw_step),
            maximum_degrees=float(args.max_rotation_degrees),
        )
        episode = trajectory._collect_full_episode(
            task,
            task_instance=task_instance,
            environment_seed=environment_seed,
            width=int(collection["width"]),
            height=int(collection["height"]),
            benchmark_seed=int(collection["benchmark_seed"]),
            max_raw_steps=int(args.max_raw_steps),
            post_success_steps=int(args.post_success_steps),
            camera_name=camera_name,
            orbit_camera_view=orbit,
            camera_rotation_axis_world=axis,
            camera_rotation_schedule=schedule,
        )
        episode["model_camera_extrinsic_cam_from_world"] = np.asarray(
            episode["nominal_camera_extrinsic_cam_from_world"], dtype=np.float32
        ).copy()
        episodes.append(episode)
        print(
            json.dumps(
                {
                    "task": task,
                    "frames": len(episode["records"]),
                    "stable_success_step": episode["stable_success_step"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    fcpg = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=device,
    )
    model, checkpoint = gallery._load_checkpoint(
        args.checkpoint.expanduser().resolve(), device=device
    )
    if not callable(getattr(model, "predict_with_history", None)):
        raise TypeError("recursive grounding requires a history-conditioned LeWM")
    rows: list[dict[str, Any]] = []
    try:
        for episode in episodes:
            rows.extend(
                _evaluate_episode(
                    episode,
                    fcpg=fcpg,
                    model=model,
                    device=device,
                    gate_low_px=float(args.transport_gate_low_px),
                    gate_high_px=float(args.transport_gate_high_px),
                    observation_interval_model_steps=int(
                        args.observation_interval_model_steps
                    ),
                )
            )
    finally:
        del fcpg, model
        gallery.legacy._clear_runtime_memory(device)

    summary_minimum_rotation = (
        float(args.summary_minimum_rotation_degrees)
        if args.summary_minimum_rotation_degrees is not None
        else (0.0 if abs(float(args.max_rotation_degrees)) <= 1.0e-12 else 8.0)
    )
    summary = summarize_rows(rows, severe_degrees=summary_minimum_rotation)
    summary["metadata"] = {
        "tasks": list(tasks),
        "test_instance_rank": int(args.test_instance_rank),
        "orbit_clock_hour": int(args.orbit_clock_hour),
        "ramp_start_raw_step": int(args.ramp_start_raw_step),
        "ramp_end_raw_step": int(args.ramp_end_raw_step),
        "max_rotation_degrees": float(args.max_rotation_degrees),
        "summary_minimum_rotation_degrees": summary_minimum_rotation,
        "camera_drift_axis_world": axis.tolist(),
        "transport_gate_low_px": float(args.transport_gate_low_px),
        "transport_gate_high_px": float(args.transport_gate_high_px),
        "constant_velocity_max_delta_px": CV_MAX_DELTA_PX,
        "temporal_baseline_reference": "current_fk_nominal_plus_filtered_camera_residual",
        "linear_kalman": {
            "dt": 1.0,
            "acceleration_variance": 1.0,
            "measurement_variance": 4.0,
        },
        "observation_interval_model_steps": int(
            args.observation_interval_model_steps
        ),
        "checkpoint_protocol": str(checkpoint.get("protocol", "")),
        "uses_future_rgb": False,
        "uses_q_true_for_inference": False,
        "interpretation": "exploratory_dynamic_camera_mechanism",
    }
    _write_rows(output_dir / "per_decision.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
