#!/usr/bin/env python3
"""Measure endpoint undershoot and reversal timing in full towel forecasts."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import TASKS, validate_a4_contract  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import a4_camera_name  # noqa: E402
from fcpg.lewm_orbit_camera import OrbitCameraView  # noqa: E402
from scripts import render_lewm_action_branch_a4_accuracy_gallery as legacy  # noqa: E402
from scripts import render_lewm_action_branch_a4_trajectory_comparison as trajectory  # noqa: E402
from scripts import render_lewm_action_prefix_joint_a4_accuracy_gallery as gallery  # noqa: E402


TASK = "towel-wipe-v0"
RIGHT = "wipe_right"
LEFT = "wipe_left"
WIPE_STAGES = frozenset((RIGHT, LEFT))
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/towel_reversal_timing_clock03/summary.json"
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=legacy.DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        help="camera-action checkpoint; repeat to compare models",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clock-hour", type=int, default=3)
    parser.add_argument("--elevation-degrees", type=float, default=35.0)
    parser.add_argument("--radius-m", type=float, default=1.1)
    parser.add_argument(
        "--minimum-lead-steps",
        type=int,
        default=15,
        help="only diagnose forecasts issued at least this long before reversal",
    )
    return parser


def _cuda0(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("towel reversal analysis permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _centers(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim < 3 or values.shape[-2:] != (3, 2):
        raise ValueError("anchor points must end in [3,2]")
    if not np.isfinite(values).all():
        raise ValueError("anchor points must be finite")
    return values.mean(axis=-2)


def wipe_axis(truth_points: np.ndarray, stages: Sequence[str]) -> np.ndarray:
    """Return a unit image axis oriented toward the right-wipe endpoint."""

    centers = _centers(truth_points)
    labels = np.asarray(tuple(str(value) for value in stages), dtype=object)
    if labels.shape != (centers.shape[0],):
        raise ValueError("stages must align with truth points")
    mask = np.isin(labels, tuple(WIPE_STAGES))
    selected = centers[mask]
    if selected.shape[0] < 3:
        raise ValueError("at least three wipe frames are required")
    centered = selected - selected.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = np.asarray(vh[0], dtype=np.float64)
    axis /= np.linalg.norm(axis)

    right_steps = np.flatnonzero(labels[:-1] == RIGHT)
    right_steps = right_steps[np.isin(labels[right_steps + 1], tuple(WIPE_STAGES))]
    if right_steps.size:
        velocity = centers[right_steps + 1] - centers[right_steps]
        if float(np.mean(velocity @ axis)) < 0.0:
            axis = -axis
    return axis


def wipe_transitions(stages: Sequence[str]) -> tuple[tuple[int, str, str], ...]:
    labels = tuple(str(value) for value in stages)
    transitions = []
    for step in range(1, len(labels)):
        before = labels[step - 1]
        after = labels[step]
        if before in WIPE_STAGES and after in WIPE_STAGES and before != after:
            transitions.append((step, before, after))
    return tuple(transitions)


def _extreme_index(values: np.ndarray, stage: str) -> int:
    if stage == RIGHT:
        return int(np.argmax(values))
    if stage == LEFT:
        return int(np.argmin(values))
    raise ValueError(f"unsupported wipe stage {stage!r}")


def reversal_rows(
    *,
    starts: np.ndarray,
    current_points: np.ndarray,
    predicted_points: np.ndarray,
    truth_points: np.ndarray,
    stages: Sequence[str],
    minimum_lead_steps: int,
) -> list[dict[str, Any]]:
    """Compare sampled predicted and true extrema around every wipe reversal."""

    decision_steps = np.asarray(starts, dtype=np.int64)
    current = np.asarray(current_points, dtype=np.float64)
    predicted = np.asarray(predicted_points, dtype=np.float64)
    truth = np.asarray(truth_points, dtype=np.float64)
    if current.shape != (len(decision_steps), 3, 2):
        raise ValueError("current points must have shape [N,3,2]")
    if predicted.shape != (len(decision_steps), 8, 3, 2):
        raise ValueError("predicted points must have shape [N,8,3,2]")
    if truth.ndim != 3 or truth.shape[1:] != (3, 2):
        raise ValueError("truth points must have shape [T,3,2]")
    if minimum_lead_steps < 5 or minimum_lead_steps > 40:
        raise ValueError("minimum lead must be in [5,40]")

    axis = wipe_axis(truth, stages)
    truth_projection = _centers(truth) @ axis
    current_projection = _centers(current) @ axis
    predicted_projection = _centers(predicted) @ axis
    rows: list[dict[str, Any]] = []

    for transition_step, before, after in wipe_transitions(stages):
        candidates = np.flatnonzero(
            (decision_steps <= transition_step - minimum_lead_steps)
            & (decision_steps + 40 >= transition_step)
        )
        if not candidates.size:
            continue
        decision_index = int(candidates[-1])
        origin = int(decision_steps[decision_index])
        sample_steps = origin + np.arange(0, 41, 5, dtype=np.int64)
        if int(sample_steps[-1]) >= truth.shape[0]:
            continue

        true_values = truth_projection[sample_steps]
        pred_values = np.concatenate(
            (
                current_projection[decision_index : decision_index + 1],
                predicted_projection[decision_index],
            )
        )
        true_extreme = _extreme_index(true_values, before)
        pred_extreme = _extreme_index(pred_values, before)
        direction = 1.0 if before == RIGHT else -1.0
        true_displacement = direction * float(
            true_values[true_extreme] - true_values[0]
        )
        predicted_displacement = direction * float(
            pred_values[pred_extreme] - pred_values[0]
        )
        endpoint_ratio = (
            predicted_displacement / true_displacement
            if true_displacement > 1.0e-6
            else None
        )
        rows.append(
            {
                "transition_step": transition_step,
                "from_stage": before,
                "to_stage": after,
                "forecast_origin": origin,
                "transition_lead_steps": transition_step - origin,
                "true_extreme_step": int(sample_steps[true_extreme]),
                "predicted_extreme_step": int(sample_steps[pred_extreme]),
                "prediction_timing_error_steps": int(
                    sample_steps[pred_extreme] - sample_steps[true_extreme]
                ),
                "true_endpoint_displacement_px": true_displacement,
                "predicted_endpoint_displacement_px": predicted_displacement,
                "predicted_to_true_endpoint_ratio": endpoint_ratio,
                "endpoint_shortfall_px": true_displacement - predicted_displacement,
            }
        )
    return rows


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"event_count": 0}
    ratios = np.asarray(
        [row["predicted_to_true_endpoint_ratio"] for row in rows], dtype=np.float64
    )
    timing = np.asarray(
        [row["prediction_timing_error_steps"] for row in rows], dtype=np.float64
    )
    shortfall = np.asarray(
        [row["endpoint_shortfall_px"] for row in rows], dtype=np.float64
    )
    return {
        "event_count": len(rows),
        "mean_endpoint_ratio": float(np.mean(ratios)),
        "median_endpoint_ratio": float(np.median(ratios)),
        "mean_endpoint_shortfall_px": float(np.mean(shortfall)),
        "mean_timing_error_steps": float(np.mean(timing)),
        "early_reversal_fraction": float(np.mean(timing < 0.0)),
    }


def _model_label(checkpoint_path: Path, metadata: Mapping[str, Any]) -> str:
    return (
        f"{checkpoint_path.expanduser().resolve().parent.name}"
        f"@step_{int(metadata.get('step', -1))}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoints = tuple(args.checkpoint or DEFAULT_CHECKPOINTS)
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    device = _cuda0(args.device)
    torch.cuda.set_device(device)

    contract = legacy._safe_read_json(args.config)
    validate_a4_contract(contract)
    instance = legacy._first_test_instances(contract)[TASK]
    environment_seed = legacy._environment_seed(
        contract, task=TASK, task_instance=instance
    )
    collection = contract["collection"]
    orbit_view = OrbitCameraView(
        clock_hour=int(args.clock_hour),
        elevation_degrees=float(args.elevation_degrees),
        radius_m=float(args.radius_m),
        base_camera_name=a4_camera_name(TASK),
    )
    episode = trajectory._collect_full_episode(
        TASK,
        task_instance=instance,
        environment_seed=environment_seed,
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
        max_raw_steps=legacy.DEFAULT_MAX_RAW_STEPS,
        post_success_steps=legacy.DEFAULT_POST_SUCCESS_RAW_STEPS,
        camera_name=a4_camera_name(TASK),
        orbit_camera_view=orbit_view,
    )
    starts = gallery._decision_steps(len(episode["actions"]))
    selected = [episode["records"][int(index)] for index in starts]
    current = np.stack([record.q_nominal for record in selected]).astype(np.float32)
    truth = np.stack([record.q_true for record in episode["records"]]).astype(
        np.float32
    )

    models: dict[str, Any] = {}
    for checkpoint_path in checkpoints:
        model, metadata = gallery._load_checkpoint(checkpoint_path, device=device)
        try:
            model_starts, predictions = gallery._predict_episode_windows(
                model,
                episode,
                current,
                task_index=TASKS.index(TASK),
                device=device,
                orbit_view=orbit_view,
            )
            if not np.array_equal(model_starts, starts):
                raise RuntimeError("prediction decision grid changed unexpectedly")
            rows = reversal_rows(
                starts=starts,
                current_points=current,
                predicted_points=predictions,
                truth_points=truth,
                stages=episode["stages"],
                minimum_lead_steps=int(args.minimum_lead_steps),
            )
            label = _model_label(checkpoint_path, metadata)
            if label in models:
                raise ValueError(f"duplicate checkpoint label: {label}")
            models[label] = {
                "checkpoint": str(checkpoint_path.expanduser().resolve()),
                "sigreg_weight": float(metadata.get("sigreg_weight", -1.0)),
                "summary": _summary(rows),
                "events": rows,
            }
        finally:
            del model
            legacy._clear_runtime_memory(device)

    payload = {
        "protocol": "lewm_towel_reversal_timing_v1",
        "task": TASK,
        "task_instance": int(instance),
        "environment_seed": int(environment_seed),
        "camera": orbit_view.label,
        "raw_frame_count": len(episode["frames"]),
        "wipe_transitions": len(wipe_transitions(episode["stages"])),
        "minimum_lead_steps": int(args.minimum_lead_steps),
        "interpretation": (
            "negative timing error means the forecast reverses before the sampled "
            "true trajectory; endpoint ratio below one means it turns short"
        ),
        "models": models,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
