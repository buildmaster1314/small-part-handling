#!/usr/bin/env python3
"""Render full-episode accuracy GIFs for the joint seven-task A4 LeWM.

Every prediction uses one current observation, its selected grounded anchors,
the recorded H1-H8 action prefix, and task identity. History-conditioned
checkpoints additionally receive the two preceding executed action chunks.
Fixed-camera rendering uses
FCPG anchors by default; orbit-camera rendering uses the exact nominal anchors
used by the multi-view training protocol. Future RGB and q_true are used only
after inference for overlays and metrics.
"""

from __future__ import annotations

import argparse
from functools import partial
import html
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_task_factory import CUSTOM_TASKS  # noqa: E402
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
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    JOINT_PROTOCOL,
    load_joint_checkpoint,
)
from fcpg.lewm_camera_action_joint_a4 import (  # noqa: E402
    CAMERA_ACTION_PROTOCOL,
    load_camera_action_joint_checkpoint,
)
from fcpg.lewm_metric3d_joint_a4 import (  # noqa: E402
    METRIC3D_PROTOCOL,
    METRIC3D_WORLD_PROTOCOL,
    load_metric3d_checkpoint,
)
from fcpg.lewm_history_conditioned_metric3d_a4 import (  # noqa: E402
    HISTORY_CONDITIONED_PROTOCOL,
    load_history_conditioned_checkpoint,
)
from fcpg.lewm_orbit_camera import OrbitCameraView  # noqa: E402
from fcpg.lewm_pose_conditioned_readout import (  # noqa: E402
    load_pose_readout_checkpoint,
    orbit_camera_basis_features,
)
from fcpg.lewm_pose_conditioned_joint_a4 import (  # noqa: E402
    POSE_JOINT_PROTOCOL,
    load_pose_conditioned_joint_checkpoint,
)
from fcpg.lewm_counterfactual_runtime import corrected_points  # noqa: E402
from scripts import render_lewm_action_branch_a4_accuracy_gallery as legacy  # noqa: E402
from scripts import render_lewm_action_branch_a4_trajectory_comparison as trajectory  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / (
    "outputs/lewm_action_prefix_a4/shared_7task_joint_seed40501_2k/"
    "lewm_action_prefix_joint_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/joint_accuracy_gallery_seed40501_2k"
)
PROTOCOL = "a4_joint_visual_action_prefix_full_episode_gallery_v2"
GROUND_TRUTH_COLOR = (31, 151, 91)
PREDICTION_COLOR = (35, 105, 191)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=legacy.DEFAULT_CONFIG)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=legacy.DEFAULT_BASE_CHECKPOINT
    )
    parser.add_argument(
        "--fcpg-checkpoint", type=Path, default=legacy.DEFAULT_FCPG_CHECKPOINT
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--pose-readout-checkpoint",
        type=Path,
        help="optional continuous-camera residual readout trained on the frozen LeWM",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument(
        "--test-instance-rank",
        type=int,
        default=0,
        help="zero-based rank within each task's 12 reserved-test instances",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="compute and save metrics without rendering MP4/GIF media",
    )
    parser.add_argument(
        "--max-raw-steps", type=int, default=legacy.DEFAULT_MAX_RAW_STEPS
    )
    parser.add_argument(
        "--post-success-steps",
        type=int,
        default=legacy.DEFAULT_POST_SUCCESS_RAW_STEPS,
    )
    parser.add_argument(
        "--orbit-clock-hour",
        type=int,
        help="render from one clock-face orbit view; odd hours are held out",
    )
    parser.add_argument("--orbit-elevation-degrees", type=float, default=35.0)
    parser.add_argument("--orbit-radius-m", type=float, default=1.1)
    parser.add_argument(
        "--camera-drift-rotation-degrees",
        type=float,
        default=0.0,
        help="rotate the actual camera while retaining the pre-drift nominal calibration",
    )
    parser.add_argument(
        "--camera-drift-axis-world",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="world-frame axis for the actual-camera rotation",
    )
    parser.add_argument(
        "--camera-drift-translation-cm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="world-frame actual-camera translation in centimeters",
    )
    parser.add_argument(
        "--camera-dynamic-drift-degrees",
        type=float,
        default=0.0,
        help="linearly accumulate this actual-camera rotation during the ramp",
    )
    parser.add_argument(
        "--camera-jitter-degrees",
        type=float,
        default=0.0,
        help="add a sinusoidal actual-camera rotation around the drift axis",
    )
    parser.add_argument(
        "--camera-drift-ramp-steps",
        type=int,
        default=120,
        help="raw steps over which dynamic camera drift reaches its final angle",
    )
    parser.add_argument(
        "--camera-jitter-period-steps",
        type=int,
        default=24,
        help="raw-step period of the dynamic camera oscillation",
    )
    parser.add_argument(
        "--model-camera-calibration",
        choices=("nominal", "actual"),
        default="nominal",
        help=(
            "camera extrinsic exposed to LeWM; actual is an offline upper bound "
            "when camera drift is present"
        ),
    )
    parser.add_argument(
        "--grounding-residual-transport",
        choices=("none", "constant", "magnitude-gated"),
        default="none",
        help="transport the current grounding innovation across H1-H8",
    )
    parser.add_argument(
        "--transport-gate-low-px",
        type=float,
        default=4.0,
        help="magnitude-gated transport is zero at or below this innovation",
    )
    parser.add_argument(
        "--transport-gate-high-px",
        type=float,
        default=8.0,
        help="magnitude-gated transport is full at or above this innovation",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("auto", "fcpg", "nominal", "oracle"),
        default="auto",
        help="auto selects nominal grounding for orbit views and FCPG otherwise",
    )
    parser.add_argument(
        "--allow-orbit-fcpg",
        action="store_true",
        help=(
            "permit FCPG on an orbit view; use only with a corrector trained for "
            "the requested orbit-camera distribution"
        ),
    )
    return parser


def _dynamic_camera_rotation(
    step: int,
    *,
    drift_degrees: float,
    jitter_degrees: float,
    ramp_steps: int,
    jitter_period_steps: int,
) -> float:
    """Return a deterministic slow drift with a visible smooth camera jitter."""

    if step < 0:
        raise ValueError("camera schedule step must be non-negative")
    if ramp_steps <= 0 or jitter_period_steps <= 0:
        raise ValueError("camera schedule periods must be positive")
    values = (drift_degrees, jitter_degrees)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("dynamic camera angles must be finite")
    progress = min(float(step) / float(ramp_steps), 1.0)
    phase = 2.0 * np.pi * float(step) / float(jitter_period_steps)
    return float(drift_degrees * progress + jitter_degrees * np.sin(phase))


def _cuda0(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("joint A4 gallery permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _resolve_grounding_mode(
    requested: str,
    *,
    orbit_clock_hour: int | None,
    allow_orbit_fcpg: bool,
) -> str:
    mode = str(requested)
    if mode == "auto":
        mode = "nominal" if orbit_clock_hour is not None else "fcpg"
    if mode not in {"fcpg", "nominal", "oracle"}:
        raise ValueError(f"unsupported grounding mode: {mode}")
    if orbit_clock_hour is not None and mode == "fcpg" and not allow_orbit_fcpg:
        raise ValueError(
            "orbit FCPG requires --allow-orbit-fcpg and a matched multiview corrector"
        )
    return mode


def _load_checkpoint(path: Path, *, device: torch.device) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, Mapping):
        raise TypeError("joint seven-task checkpoint must contain a mapping")
    protocol = payload.get("protocol")
    if protocol == JOINT_PROTOCOL:
        model = load_joint_checkpoint(payload).to(device).eval()
    elif protocol == CAMERA_ACTION_PROTOCOL:
        model = load_camera_action_joint_checkpoint(payload).to(device).eval()
    elif protocol == POSE_JOINT_PROTOCOL:
        model = load_pose_conditioned_joint_checkpoint(payload).to(device).eval()
    elif protocol in {METRIC3D_PROTOCOL, METRIC3D_WORLD_PROTOCOL}:
        model = load_metric3d_checkpoint(payload).to(device).eval()
    elif protocol == HISTORY_CONDITIONED_PROTOCOL:
        model = load_history_conditioned_checkpoint(payload).to(device).eval()
    else:
        raise ValueError("checkpoint is not a supported joint seven-task A4 protocol")
    return model, dict(payload)


def _transform_future_actions_for_model(
    model: Any,
    future_action: torch.Tensor,
    episode: Mapping[str, Any],
) -> torch.Tensor:
    transform = getattr(model, "transform_future_action", None)
    if not callable(transform):
        return future_action
    extrinsic = _model_camera_extrinsic(episode)
    camera_batch = torch.as_tensor(
        extrinsic,
        device=future_action.device,
        dtype=future_action.dtype,
    )[None].expand(future_action.shape[0], -1, -1)
    return transform(
        future_action,
        {"camera_extrinsic_cam_from_world": camera_batch},
    )


def _model_camera_extrinsic(episode: Mapping[str, Any]) -> np.ndarray:
    value = episode.get(
        "model_camera_extrinsic_cam_from_world",
        episode.get("camera_extrinsic_cam_from_world"),
    )
    extrinsic = np.asarray(value, dtype=np.float32)
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("model camera extrinsic must be one finite [4,4] matrix")
    return extrinsic


def _transport_current_grounding_residual(
    predicted_q: torch.Tensor,
    q_nominal: torch.Tensor,
    q_grounded: torch.Tensor,
    *,
    mode: str,
    gate_low_px: float = 4.0,
    gate_high_px: float = 8.0,
) -> torch.Tensor:
    if mode == "none":
        return predicted_q
    if mode not in {"constant", "magnitude-gated"}:
        raise ValueError(f"unsupported grounding residual transport: {mode}")
    if q_nominal.shape != q_grounded.shape or q_nominal.ndim != 4:
        raise ValueError("current anchors must match [B,1,K,2]")
    if predicted_q.ndim != 4 or predicted_q.shape[0] != q_nominal.shape[0]:
        raise ValueError("predicted anchors must have shape [B,H,K,2]")
    innovation = (
        q_grounded[:, 0].to(device=predicted_q.device, dtype=predicted_q.dtype)
        - q_nominal[:, 0].to(device=predicted_q.device, dtype=predicted_q.dtype)
    )
    if innovation.shape != predicted_q.shape[:1] + predicted_q.shape[2:]:
        raise ValueError("grounding innovation does not align with predictions")
    if mode == "constant":
        transported = innovation
    else:
        low = float(gate_low_px)
        high = float(gate_high_px)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("transport gate requires finite high > low")
        magnitude = torch.linalg.vector_norm(innovation, dim=-1, keepdim=True)
        gain = ((magnitude - low) / (high - low)).clamp(0.0, 1.0)
        transported = gain * innovation
    return predicted_q + transported[:, None]


def _load_pose_readout(
    path: Path,
    *,
    base_checkpoint: Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, Mapping):
        raise TypeError("pose readout checkpoint must contain a mapping")
    recorded_base = Path(str(payload.get("base_checkpoint", ""))).expanduser()
    if recorded_base.resolve() != base_checkpoint.expanduser().resolve():
        raise ValueError("pose readout was trained against a different base checkpoint")
    model = load_pose_readout_checkpoint(payload).to(device).eval()
    model.reference_mode = str(payload.get("reference_mode", "base_prediction"))
    return model, dict(payload)


def _display_points(
    points: np.ndarray,
    *,
    task: str,
    height: int,
    flip_vertical: bool | None = None,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).copy()
    if values.ndim < 2 or values.shape[-2:] != (3, 2):
        raise ValueError("anchor points must end in [3,2]")
    if not np.isfinite(values).all():
        raise ValueError("anchor points must be finite")
    should_flip = task not in CUSTOM_TASKS if flip_vertical is None else flip_vertical
    if should_flip:
        values[..., 1] = float(height - 1) - values[..., 1]
    return values


def _decision_steps(raw_action_count: int) -> np.ndarray:
    last = int(raw_action_count) - HORIZON * RAW_STEPS_PER_MODEL_STEP
    if last < 0:
        raise ValueError("episode is shorter than the H8 prediction horizon")
    return np.arange(0, last + 1, RAW_STEPS_PER_MODEL_STEP, dtype=np.int64)


def _chunk_future_actions(actions: np.ndarray, starts: np.ndarray) -> np.ndarray:
    raw = np.asarray(actions, dtype=np.float32)
    result = []
    for start in np.asarray(starts, dtype=np.int64):
        stop = int(start) + HORIZON * RAW_STEPS_PER_MODEL_STEP
        window = raw[int(start) : stop]
        expected = (HORIZON * RAW_STEPS_PER_MODEL_STEP, 4)
        if window.shape != expected:
            raise ValueError("joint gallery action window is incomplete")
        result.append(window.reshape(HORIZON, -1))
    return np.ascontiguousarray(np.stack(result), dtype=np.float32)


def _chunk_history_actions(actions: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Return the two causal action chunks preceding each decision step."""

    raw = np.asarray(actions, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError("episode actions must have shape [T,4]")
    result = np.zeros((len(starts), 2, 5, 4), dtype=np.float32)
    for row, start_value in enumerate(np.asarray(starts, dtype=np.int64)):
        start = int(start_value)
        if start < 0 or start > len(raw):
            raise ValueError("decision step lies outside episode actions")
        available = raw[max(0, start - 10) : start]
        if len(available):
            result[row].reshape(10, 4)[-len(available) :] = available
    return np.ascontiguousarray(result.reshape(len(starts), 2, 20))


@torch.inference_mode()
def _predict_episode_windows(
    model: Any,
    episode: Mapping[str, Any],
    corrected: np.ndarray,
    *,
    task_index: int,
    device: torch.device,
    batch_size: int = 16,
    pose_readout: Any | None = None,
    orbit_view: OrbitCameraView | None = None,
    grounding_residual_transport: str = "none",
    transport_gate_low_px: float = 4.0,
    transport_gate_high_px: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    records = episode["records"]
    starts = _decision_steps(len(episode["actions"]))
    if corrected.shape != (len(starts), 3, 2):
        raise ValueError("corrected current anchors differ from decision grid")
    action_prefix = _chunk_future_actions(episode["actions"], starts)
    history_prefix = _chunk_history_actions(episode["actions"], starts)
    predictions = []
    for offset in range(0, len(starts), int(batch_size)):
        rows = starts[offset : offset + int(batch_size)]
        q_corrected = corrected[offset : offset + len(rows)]
        selected = [records[int(index)] for index in rows]
        rgb = torch.from_numpy(np.stack([row.rgb for row in selected]))[:, None]
        proprio = torch.from_numpy(
            np.stack([row.proprio for row in selected])
        )[:, None]
        q_nominal = torch.from_numpy(
            np.stack([row.q_nominal for row in selected])
        )[:, None]
        xyz = torch.from_numpy(
            np.stack([row.keypoints_xyz for row in selected])
        )[:, None]
        valid = torch.from_numpy(
            np.stack([row.valid_nominal for row in selected])
        )[:, None]
        q_grounded = torch.from_numpy(q_corrected)[:, None]
        camera_features = None
        if hasattr(model, "decode_metric"):
            if orbit_view is None:
                raise ValueError("pose-conditioned joint model requires an orbit view")
            camera_features = orbit_camera_basis_features(
                torch.full(
                    (len(rows),),
                    int(orbit_view.clock_hour),
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (len(rows),),
                    float(orbit_view.elevation_degrees),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.full(
                    (len(rows),),
                    float(orbit_view.radius_m),
                    dtype=torch.float32,
                    device=device,
                ),
            )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            encode_kwargs = dict(
                rgb=rgb.to(device),
                proprio=proprio.to(device),
                q_nominal=q_nominal.to(device),
                q_corrected=q_grounded.to(device),
                keypoints_xyz=xyz.to(device),
                valid=valid.to(device),
            )
            if camera_features is not None:
                encode_kwargs["camera_features"] = camera_features
            current_state = model.encode_observations(**encode_kwargs)[:, 0]
            task = torch.full(
                (len(rows),), int(task_index), dtype=torch.long, device=device
            )
            raw_future_action = torch.from_numpy(
                action_prefix[offset : offset + len(rows)]
            ).to(device=device, dtype=current_state.dtype)
            predict_with_history = getattr(model, "predict_with_history", None)
            if callable(predict_with_history):
                raw_history_action = torch.from_numpy(
                    history_prefix[offset : offset + len(rows)]
                ).to(device=device, dtype=current_state.dtype)
                extrinsic = torch.as_tensor(
                    _model_camera_extrinsic(episode),
                    device=device,
                    dtype=torch.float32,
                )[None].expand(len(rows), -1, -1)
                predicted_state = predict_with_history(
                    current_state=current_state,
                    history_action=raw_history_action,
                    future_action=raw_future_action,
                    task_index=task,
                    batch={"camera_extrinsic_cam_from_world": extrinsic},
                )
            else:
                transformed_future_action = _transform_future_actions_for_model(
                    model,
                    raw_future_action,
                    episode,
                )
                predicted_state = model.dynamics(
                    current_state,
                    transformed_future_action,
                    task_index=task,
                )
            image_size = (int(rgb.shape[-1]), int(rgb.shape[-2]))
            decode_3d = getattr(model, "decode_3d", None)
            if callable(decode_3d):
                intrinsic = torch.as_tensor(
                    np.asarray(episode["camera_intrinsic"], dtype=np.float32),
                    device=device,
                    dtype=torch.float32,
                )[None].expand(len(rows), -1, -1)
                extrinsic = torch.as_tensor(
                    _model_camera_extrinsic(episode),
                    device=device,
                    dtype=torch.float32,
                )[None].expand(len(rows), -1, -1)
                predicted_q = decode_3d(
                    predicted_state=predicted_state,
                    current_state=current_state,
                    current_keypoints_xyz=xyz[:, 0].to(device),
                    future_action=raw_future_action,
                    task_index=task,
                    camera_intrinsic=intrinsic,
                    extrinsic_cam_from_world=extrinsic,
                ).predicted_q_px
            elif camera_features is not None:
                predicted_q = model.decode_metric(
                    predicted_state=predicted_state,
                    current_state=current_state,
                    current_q_px=q_grounded[:, 0].to(device),
                    current_keypoints_xyz=xyz[:, 0].to(device),
                    task_index=task,
                    camera_features=camera_features,
                    image_size=image_size,
                )
            else:
                predicted_q = model.metric_readout(
                    predicted_state,
                    current_state,
                    q_grounded[:, 0].to(device),
                    task,
                    image_size=image_size,
                )
            if pose_readout is not None:
                if camera_features is not None:
                    raise ValueError(
                        "standalone pose readout cannot be stacked on the pose-conditioned joint model"
                    )
                if orbit_view is None:
                    raise ValueError("pose-conditioned readout requires an orbit view")
                camera_features = orbit_camera_basis_features(
                    torch.full(
                        (len(rows),),
                        int(orbit_view.clock_hour),
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.full(
                        (len(rows),),
                        float(orbit_view.elevation_degrees),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.full(
                        (len(rows),),
                        float(orbit_view.radius_m),
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                reference_mode = str(
                    getattr(pose_readout, "reference_mode", "base_prediction")
                )
                reference_q = (
                    q_grounded[:, 0]
                    .to(device)
                    .float()[:, None]
                    .expand_as(predicted_q)
                    if reference_mode == "current_q"
                    else predicted_q
                )
                predicted_q = pose_readout(
                    predicted_state=predicted_state,
                    current_state=current_state,
                    base_prediction_q_px=reference_q,
                    current_q_px=q_grounded[:, 0].to(device),
                    current_keypoints_xyz=xyz[:, 0].to(device),
                    task_index=task,
                    camera_features=camera_features,
                    image_size=(
                        int(rgb.shape[-1]),
                        int(rgb.shape[-2]),
                    ),
                ).predicted_q_px
            predicted_q = _transport_current_grounding_residual(
                predicted_q,
                q_nominal,
                q_grounded,
                mode=grounding_residual_transport,
                gate_low_px=transport_gate_low_px,
                gate_high_px=transport_gate_high_px,
            )
        predictions.append(predicted_q.float().cpu().numpy())
    result = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
    if result.shape != (len(starts), HORIZON, 3, 2):
        raise RuntimeError("joint gallery prediction shape drifted")
    if not np.isfinite(result).all():
        raise RuntimeError("joint gallery prediction contains non-finite values")
    return starts, result


def _stitch_window_predictions(
    decision_steps: np.ndarray,
    predictions: np.ndarray,
    current_q: np.ndarray,
    *,
    raw_frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefer H1 until the final decision, then consume H2-H8 to the end."""

    starts = np.asarray(decision_steps, dtype=np.int64)
    forecast = np.asarray(predictions, dtype=np.float32)
    current = np.asarray(current_q, dtype=np.float32)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("decision_steps must be non-empty")
    if forecast.shape != (starts.size, HORIZON, 3, 2):
        raise ValueError("predictions must have shape [N,8,3,2]")
    if current.shape != (3, 2):
        raise ValueError("current_q must have shape [3,2]")
    if not np.array_equal(
        starts,
        np.arange(starts.size, dtype=np.int64) * RAW_STEPS_PER_MODEL_STEP,
    ):
        raise ValueError("decision grid must begin at zero and be contiguous")
    last_target = min(
        int(raw_frame_count) - 1,
        int(starts[-1]) + HORIZON * RAW_STEPS_PER_MODEL_STEP,
    )
    grid_steps = np.arange(
        0, last_target + 1, RAW_STEPS_PER_MODEL_STEP, dtype=np.int64
    )
    grid_points = [current]
    source_horizons = [0]
    for target in grid_steps[1:]:
        decision = min(int(target) - RAW_STEPS_PER_MODEL_STEP, int(starts[-1]))
        decision_index = decision // RAW_STEPS_PER_MODEL_STEP
        horizon = (int(target) - decision) // RAW_STEPS_PER_MODEL_STEP
        if not 1 <= horizon <= HORIZON:
            raise RuntimeError("stitched prediction horizon left H1-H8")
        grid_points.append(forecast[decision_index, horizon - 1])
        source_horizons.append(horizon)
    model_points = np.stack(grid_points).astype(np.float32, copy=False)
    dense = np.empty((int(raw_frame_count), 3, 2), dtype=np.float32)
    raw_steps = np.arange(int(raw_frame_count), dtype=np.float64)
    for anchor in range(3):
        for axis in range(2):
            dense[:, anchor, axis] = np.interp(
                raw_steps,
                grid_steps.astype(np.float64),
                model_points[:, anchor, axis].astype(np.float64),
            ).astype(np.float32)
    return dense, grid_steps, np.asarray(source_horizons, dtype=np.int64)


def _stitch_max_lookahead_predictions(
    decision_steps: np.ndarray,
    predictions: np.ndarray,
    current_q: np.ndarray,
    *,
    raw_frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one continuous trajectory using the farthest available forecast."""

    starts = np.asarray(decision_steps, dtype=np.int64)
    forecast = np.asarray(predictions, dtype=np.float32)
    current = np.asarray(current_q, dtype=np.float32)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("decision_steps must be non-empty")
    if forecast.shape != (starts.size, HORIZON, 3, 2):
        raise ValueError("predictions must have shape [N,8,3,2]")
    if current.shape != (3, 2):
        raise ValueError("current_q must have shape [3,2]")
    expected_starts = (
        np.arange(starts.size, dtype=np.int64) * RAW_STEPS_PER_MODEL_STEP
    )
    if not np.array_equal(starts, expected_starts):
        raise ValueError("decision grid must begin at zero and be contiguous")

    last_target = min(
        int(raw_frame_count) - 1,
        int(starts[-1]) + HORIZON * RAW_STEPS_PER_MODEL_STEP,
    )
    grid_steps = np.arange(
        0, last_target + 1, RAW_STEPS_PER_MODEL_STEP, dtype=np.int64
    )
    grid_points = [current]
    source_horizons = [0]
    maximum_offset = HORIZON * RAW_STEPS_PER_MODEL_STEP
    for target in grid_steps[1:]:
        decision = max(0, int(target) - maximum_offset)
        decision = min(decision, int(starts[-1]))
        decision_index = decision // RAW_STEPS_PER_MODEL_STEP
        horizon = (int(target) - decision) // RAW_STEPS_PER_MODEL_STEP
        if not 1 <= horizon <= HORIZON:
            raise RuntimeError("max-lookahead prediction left H1-H8")
        grid_points.append(forecast[decision_index, horizon - 1])
        source_horizons.append(horizon)

    model_points = np.stack(grid_points).astype(np.float32, copy=False)
    dense = np.empty((int(raw_frame_count), 3, 2), dtype=np.float32)
    raw_steps = np.arange(int(raw_frame_count), dtype=np.float64)
    for anchor in range(3):
        for axis in range(2):
            dense[:, anchor, axis] = np.interp(
                raw_steps,
                grid_steps.astype(np.float64),
                model_points[:, anchor, axis].astype(np.float64),
            ).astype(np.float32)
    return dense, grid_steps, np.asarray(source_horizons, dtype=np.int64)


def _sample_frame_indices(frame_count: int, max_frames: int) -> np.ndarray:
    if frame_count <= 0 or max_frames <= 1:
        raise ValueError("frame_count and max_frames must be positive")
    if frame_count <= max_frames:
        return np.arange(frame_count, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, frame_count - 1, max_frames)).astype(np.int64)
    )


def _draw_path(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    *,
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    values = np.asarray(points, dtype=np.float32)
    if len(values) >= 2:
        draw.line([tuple(row) for row in values], fill=color, width=width)


def _render_split_frame(
    frame: Image.Image,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    raw_step: int,
    end_step: int,
    task_label: str,
    success: bool,
    source_horizon: int,
) -> Image.Image:
    if truth.shape != prediction.shape or truth.shape[1:] != (3, 2):
        raise ValueError("render trajectories must be aligned [T,3,2]")
    left = frame.convert("RGB").copy()
    right = frame.convert("L").convert("RGB")
    right = Image.blend(right, Image.new("RGB", right.size, (232, 235, 237)), 0.28)
    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    truth_center = truth.mean(axis=1)
    prediction_center = prediction.mean(axis=1)
    _draw_path(
        left_draw,
        truth_center[: raw_step + 1],
        color=GROUND_TRUTH_COLOR,
    )
    _draw_path(
        right_draw,
        prediction_center[: raw_step + 1],
        color=PREDICTION_COLOR,
    )
    for point in truth[raw_step]:
        legacy._draw_marker(
            left_draw, point, fill=GROUND_TRUTH_COLOR, radius=4, diamond=False
        )
    for point in prediction[raw_step]:
        legacy._draw_marker(
            right_draw, point, fill=PREDICTION_COLOR, radius=5, diamond=True
        )

    width, height = left.size
    margin, gap, header, footer = 12, 12, 66, 38
    canvas = Image.new(
        "RGB",
        (margin * 2 + width * 2 + gap, header + height + footer),
        (238, 241, 243),
    )
    right_x = margin + width + gap
    canvas.paste(left, (margin, header))
    canvas.paste(right, (right_x, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 7), task_label, fill=(23, 32, 40), font=legacy._safe_font(17))
    draw.text(
        (margin, 35),
        "ACTUAL FULL TRAJECTORY",
        fill=GROUND_TRUTH_COLOR,
        font=legacy._safe_font(12),
    )
    draw.text(
        (right_x, 35),
        "LeWM PREDICTED TRAJECTORY",
        fill=PREDICTION_COLOR,
        font=legacy._safe_font(12),
    )
    epe = float(
        np.linalg.norm(prediction[raw_step] - truth[raw_step], axis=-1).mean()
    )
    draw.text(
        (margin, header + height + 10),
        (
            f"raw {raw_step:03d}/{end_step:03d} | success {int(success)} | "
            f"anchor EPE {epe:.2f} px | source H{source_horizon}"
        ),
        fill=(55, 66, 75),
        font=legacy._safe_font(11),
    )
    return canvas


def _interpolate_prediction_window(
    current_q: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    """Interpolate H1-H8 model outputs into 40 smooth raw-step positions."""

    current = np.asarray(current_q, dtype=np.float32)
    forecast = np.asarray(prediction, dtype=np.float32)
    if current.shape != (3, 2) or forecast.shape != (HORIZON, 3, 2):
        raise ValueError("prediction window requires [3,2] current and [8,3,2] future")
    model_steps = np.arange(
        0,
        HORIZON * RAW_STEPS_PER_MODEL_STEP + 1,
        RAW_STEPS_PER_MODEL_STEP,
        dtype=np.float64,
    )
    points = np.concatenate([current[None], forecast], axis=0)
    raw_steps = np.arange(
        HORIZON * RAW_STEPS_PER_MODEL_STEP + 1, dtype=np.float64
    )
    dense = np.empty((len(raw_steps), 3, 2), dtype=np.float32)
    for anchor in range(3):
        for axis in range(2):
            dense[:, anchor, axis] = np.interp(
                raw_steps,
                model_steps,
                points[:, anchor, axis].astype(np.float64),
            ).astype(np.float32)
    return dense


def _render_prediction_cycle_frame(
    left_frame: Image.Image,
    right_frame: Image.Image,
    truth_history: np.ndarray,
    prediction_path: np.ndarray,
    *,
    actual_step: int,
    prediction_origin: int,
    prediction_target: int,
    end_step: int,
    task_label: str,
    success: bool,
    phase: str,
) -> Image.Image:
    truth = np.asarray(truth_history, dtype=np.float32)
    forecast = np.asarray(prediction_path, dtype=np.float32)
    if truth.ndim != 3 or truth.shape[1:] != (3, 2) or len(truth) == 0:
        raise ValueError("truth_history must have shape [T,3,2]")
    if forecast.ndim != 3 or forecast.shape[1:] != (3, 2) or len(forecast) == 0:
        raise ValueError("prediction_path must have shape [T,3,2]")

    left = left_frame.convert("RGB").copy()
    right = right_frame.convert("L").convert("RGB")
    right = Image.blend(right, Image.new("RGB", right.size, (232, 235, 237)), 0.28)
    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    _draw_path(left_draw, truth.mean(axis=1), color=GROUND_TRUTH_COLOR)
    _draw_path(right_draw, forecast.mean(axis=1), color=PREDICTION_COLOR)
    for point in truth[-1]:
        legacy._draw_marker(
            left_draw, point, fill=GROUND_TRUTH_COLOR, radius=4, diamond=False
        )
    for point in forecast[-1]:
        legacy._draw_marker(
            right_draw, point, fill=PREDICTION_COLOR, radius=5, diamond=True
        )

    width, height = left.size
    margin, gap, header, footer = 12, 12, 66, 38
    canvas = Image.new(
        "RGB",
        (margin * 2 + width * 2 + gap, header + height + footer),
        (238, 241, 243),
    )
    right_x = margin + width + gap
    canvas.paste(left, (margin, header))
    canvas.paste(right, (right_x, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 7), task_label, fill=(23, 32, 40), font=legacy._safe_font(17))
    draw.text(
        (margin, 35),
        f"ACTUAL t={actual_step}",
        fill=GROUND_TRUTH_COLOR,
        font=legacy._safe_font(12),
    )
    draw.text(
        (right_x, 35),
        f"PRED t={prediction_origin}->{prediction_target}",
        fill=PREDICTION_COLOR,
        font=legacy._safe_font(12),
    )
    draw.text(
        (margin, header + height + 10),
        (
            f"{phase} | actual {actual_step:03d}/{end_step:03d} | "
            f"forecast +{prediction_target - prediction_origin:02d}/40 | "
            f"success {int(success)}"
        ),
        fill=(55, 66, 75),
        font=legacy._safe_font(11),
    )
    return canvas


def _prediction_cycle_frames(
    episode: Mapping[str, Any],
    starts: np.ndarray,
    corrected: np.ndarray,
    predictions: np.ndarray,
    truth: np.ndarray,
    *,
    task_label: str,
) -> Iterable[Image.Image]:
    """Advance actual video five steps, then unroll a 40-step forecast."""

    decision_steps = np.asarray(starts, dtype=np.int64)
    grounded = np.asarray(corrected, dtype=np.float32)
    forecast = np.asarray(predictions, dtype=np.float32)
    if grounded.shape != (len(decision_steps), 3, 2):
        raise ValueError("corrected anchors do not match decision steps")
    if forecast.shape != (len(decision_steps), HORIZON, 3, 2):
        raise ValueError("predictions do not match decision steps")
    frames = episode["frames"]
    successes = episode["successes"]
    end_step = len(frames) - 1

    previous_dense: np.ndarray | None = None
    previous_step: int | None = None
    for decision_index, value in enumerate(decision_steps):
        current_step = int(value)
        if previous_dense is not None and previous_step is not None:
            for actual_step in range(previous_step + 1, current_step + 1):
                yield _render_prediction_cycle_frame(
                    frames[actual_step],
                    frames[previous_step],
                    truth[: actual_step + 1],
                    previous_dense,
                    actual_step=actual_step,
                    prediction_origin=previous_step,
                    prediction_target=previous_step + 40,
                    end_step=end_step,
                    task_label=task_label,
                    success=bool(successes[actual_step]),
                    phase="ACTUAL ADVANCE",
                )

        dense = _interpolate_prediction_window(
            grounded[decision_index], forecast[decision_index]
        )
        for offset in range(1, len(dense)):
            yield _render_prediction_cycle_frame(
                frames[current_step],
                frames[current_step],
                truth[: current_step + 1],
                dense[: offset + 1],
                actual_step=current_step,
                prediction_origin=current_step,
                prediction_target=current_step + offset,
                end_step=end_step,
                task_label=task_label,
                success=bool(successes[current_step]),
                phase="FORECAST PLAYBACK",
            )
        previous_dense = dense
        previous_step = current_step

    if previous_dense is not None and previous_step is not None:
        for actual_step in range(previous_step + 1, end_step + 1):
            yield _render_prediction_cycle_frame(
                frames[actual_step],
                frames[previous_step],
                truth[: actual_step + 1],
                previous_dense,
                actual_step=actual_step,
                prediction_origin=previous_step,
                prediction_target=previous_step + 40,
                end_step=end_step,
                task_label=task_label,
                success=bool(successes[actual_step]),
                phase="FINAL ACTUAL CATCH-UP",
            )


def _prediction_cycle_frame_count(starts: np.ndarray, *, end_step: int) -> int:
    decisions = np.asarray(starts, dtype=np.int64)
    if decisions.ndim != 1 or decisions.size == 0:
        raise ValueError("prediction-cycle decisions must be non-empty")
    if int(end_step) < int(decisions[-1]):
        raise ValueError("episode end precedes the final prediction decision")
    return int(HORIZON * RAW_STEPS_PER_MODEL_STEP * decisions.size + end_step)


def _save_prediction_cycle_media(
    output_dir: Path,
    *,
    task: str,
    frames: Iterable[Image.Image],
    fps: float,
    expected_frame_count: int,
    gif_max_frames: int = 200,
) -> tuple[str, str, int, int]:
    stem = f"{task.replace('-', '_')}_joint_prediction_cycle"
    video_name = f"{stem}.mp4"
    gif_name = f"{stem}.gif"
    video_writer = imageio.get_writer(
        output_dir / video_name,
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    )
    gif_count = min(int(gif_max_frames), int(expected_frame_count))
    gif_indices = _sample_frame_indices(int(expected_frame_count), gif_count)
    gif_duration = float(expected_frame_count) / (float(fps) * float(gif_count))
    gif_writer = imageio.get_writer(
        output_dir / gif_name,
        format="GIF",
        mode="I",
        duration=gif_duration,
        loop=0,
    )
    count = 0
    gif_written = 0
    try:
        for frame in frames:
            pixels = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            video_writer.append_data(pixels)
            if gif_written < gif_count and count == int(gif_indices[gif_written]):
                gif_writer.append_data(pixels)
                gif_written += 1
            count += 1
    finally:
        video_writer.close()
        gif_writer.close()
    if count != int(expected_frame_count):
        raise RuntimeError(
            f"prediction-cycle frame count drifted: {count} != {expected_frame_count}"
        )
    if gif_written != gif_count:
        raise RuntimeError("prediction-cycle GIF sampling did not reach its endpoint")
    if count <= 0:
        raise ValueError("prediction-cycle video requires frames")
    return video_name, gif_name, count, gif_written


def _horizon_metrics(
    episode: Mapping[str, Any], starts: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    records = episode["records"]
    truth = np.stack([row.q_true for row in records]).astype(np.float32)
    result: dict[str, Any] = {}
    for horizon in range(1, HORIZON + 1):
        indices = starts + horizon * RAW_STEPS_PER_MODEL_STEP
        target = truth[indices]
        estimate = predictions[:, horizon - 1]
        anchor_error = np.linalg.norm(estimate - target, axis=-1)
        center_error = np.linalg.norm(
            estimate.mean(axis=1) - target.mean(axis=1), axis=-1
        )
        result[f"H{horizon}"] = {
            "sample_count": int(len(indices)),
            "anchor_mean_epe_px": float(anchor_error.mean()),
            "anchor_p95_epe_px": float(np.quantile(anchor_error, 0.95)),
            "center_mean_epe_px": float(center_error.mean()),
            "center_p95_epe_px": float(np.quantile(center_error, 0.95)),
            "anchor_pck_at_4": float(np.mean(anchor_error <= 4.0)),
            "anchor_pck_at_8": float(np.mean(anchor_error <= 8.0)),
        }
    return result


def _save_gif(
    output_dir: Path,
    *,
    task: str,
    frames: Sequence[Image.Image],
    fps: float,
) -> str:
    if not frames:
        raise ValueError("joint accuracy GIF requires frames")
    name = f"{task.replace('-', '_')}_joint_full_episode.gif"
    duration = max(40, int(round(1000.0 / float(fps))))
    frames[0].save(
        output_dir / name,
        format="GIF",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration,
        loop=0,
        disposal=2,
    )
    return name


def _render_html(entries: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> str:
    history_clause = (
        ", the preceding 10 executed actions"
        if bool(metadata.get("uses_action_history", False))
        else ""
    )
    sections = []
    for entry in entries:
        rows = []
        for horizon in range(1, HORIZON + 1):
            metric = entry["horizon_metrics"][f"H{horizon}"]
            rows.append(
                "<tr>"
                f"<td>H{horizon}</td>"
                f"<td>{metric['anchor_mean_epe_px']:.2f}</td>"
                f"<td>{metric['anchor_p95_epe_px']:.2f}</td>"
                f"<td>{metric['center_mean_epe_px']:.2f}</td>"
                f"<td>{100.0 * metric['anchor_pck_at_4']:.1f}%</td>"
                f"<td>{100.0 * metric['anchor_pck_at_8']:.1f}%</td>"
                "</tr>"
            )
        media = (
            f'<video controls muted loop playsinline preload="metadata"><source src="{html.escape(str(entry["video"]), quote=True)}" type="video/mp4"></video>'
            if entry.get("video")
            else '<div class="metrics-only">Metrics-only run: prediction media was not rendered.</div>'
        )
        sections.append(
            f"""
<section><div class="head"><div><h2>{html.escape(str(entry['task_label']))}</h2>
<p>reserved-test instance {entry['task_instance']} &middot; camera {html.escape(str(entry.get('camera_label', 'task_default')))} &middot; grounding {html.escape(str(entry.get('grounding_mode', 'fcpg')))} &middot; raw frames {entry['raw_frame_count']} &middot; rendered frames {entry['rendered_frame_count']} &middot; stable success raw {entry['stable_success_step']}</p></div>
<div class="summary">current grounding / max-lookahead EPE<br><b>{entry.get('current_grounding_anchor_mean_epe_px', float('nan')):.2f} / {entry['stitched_anchor_mean_epe_px']:.2f} px</b></div></div>
<p class="view-title"><b>Repeated 40-step prediction:</b> the left side advances five real frames and pauses. The right side then plays the new 40-frame anchor forecast before the left advances again. The model outputs H1-H8; intermediate raw frames are linear interpolation for smooth playback.</p>
{media}
{f'<details><summary>Open the sampled {entry["gif_frame_count"]}-frame GIF</summary><img class="preview-gif" loading="lazy" src="{html.escape(str(entry["gif"]), quote=True)}" alt="sampled prediction cycle"></details>' if entry.get('gif') else ''}
<table><thead><tr><th>Horizon</th><th>Anchor mean</th><th>Anchor P95</th><th>Center mean</th><th>PCK@4</th><th>PCK@8</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"""
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Joint seven-task LeWM full-episode accuracy</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf1f3;color:#17212a;font-family:Inter,Arial,sans-serif;letter-spacing:0}}header{{background:#17252e;color:white;padding:24px max(20px,calc((100vw - 880px)/2))}}h1,h2,p{{margin:0}}header p{{margin-top:8px;color:#c4cfd5;line-height:1.5}}main{{max-width:880px;margin:auto;padding:20px 20px 60px}}.contract{{background:white;border-left:5px solid #2976a8;padding:15px 17px;line-height:1.55}}section{{padding:28px 0 34px;border-bottom:1px solid #c8d0d5}}.head{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:12px}}.head p{{margin-top:5px;color:#66737d;font-size:13px}}.summary{{text-align:right;color:#52616b;font-size:12px}}.summary b{{font-size:18px;color:#23333e}}.view-title{{margin:15px 0 8px;color:#52616b;line-height:1.45}}video,.preview-gif{{display:block;width:100%;height:auto;border:1px solid #bbc5cb;background:#20272c}}.metrics-only{{padding:18px;border:1px solid #bbc5cb;background:white;color:#52616b}}details{{margin-top:10px}}summary{{cursor:pointer;color:#52616b}}.preview-gif{{margin-top:8px}}table{{width:100%;border-collapse:collapse;background:white;margin-top:14px}}th,td{{padding:9px 10px;text-align:right;border-bottom:1px solid #dde3e6;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}th{{color:#60707a;font-size:12px}}@media(max-width:700px){{.head{{display:block}}.summary{{text-align:left;margin-top:8px}}table{{font-size:12px}}th,td{{padding:7px 5px}}}}
</style></head><body><header><h1>Joint Seven-Task LeWM: Full-Episode Accuracy</h1><p>One shared model, seed {metadata['seed']}, fixed step {metadata['step']}, SIGReg {metadata['sigreg_weight']}. Seven tasks are trained together with task-balanced batches. Camera {html.escape(str(metadata.get('camera', 'task_default')))}, grounding {html.escape(str(metadata.get('grounding_mode', 'fcpg')))}.</p></header><main>
<div class="contract"><b>What the model sees:</b> the paused left observation, proprioception, {html.escape(str(metadata.get('grounding_mode', 'fcpg')))} current anchors, task identity{history_clause}, and the next 40 executed actions. <b>What it never sees:</b> future RGB or q_true. Oracle mode uses current-frame q_true only as a diagnostic input. The right background is the same paused input frame; only the moving blue anchors and path are predictions.</div>
{''.join(sections)}</main></body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_raw_steps <= 0:
        raise ValueError("frame limit must be positive")
    if args.post_success_steps < 0 or not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("post-success steps and fps are invalid")
    if args.orbit_clock_hour is not None and not 1 <= args.orbit_clock_hour <= 12:
        raise ValueError("orbit clock hour must be in [1,12]")
    if (
        not np.isfinite(args.transport_gate_low_px)
        or not np.isfinite(args.transport_gate_high_px)
        or args.transport_gate_high_px <= args.transport_gate_low_px
    ):
        raise ValueError("transport gate requires finite high > low")
    drift_axis = np.asarray(args.camera_drift_axis_world, dtype=np.float64)
    drift_translation_cm = np.asarray(
        args.camera_drift_translation_cm, dtype=np.float64
    )
    drift_rotation_degrees = float(args.camera_drift_rotation_degrees)
    dynamic_drift_degrees = float(args.camera_dynamic_drift_degrees)
    jitter_degrees = float(args.camera_jitter_degrees)
    if (
        drift_axis.shape != (3,)
        or drift_translation_cm.shape != (3,)
        or not np.isfinite(drift_axis).all()
        or not np.isfinite(drift_translation_cm).all()
        or not np.isfinite(drift_rotation_degrees)
        or not np.isfinite(dynamic_drift_degrees)
        or not np.isfinite(jitter_degrees)
    ):
        raise ValueError("camera drift parameters must be finite 3D values")
    has_dynamic_rotation = bool(
        abs(dynamic_drift_degrees) > 0.0 or abs(jitter_degrees) > 0.0
    )
    if int(args.camera_drift_ramp_steps) <= 0:
        raise ValueError("camera drift ramp steps must be positive")
    if int(args.camera_jitter_period_steps) <= 0:
        raise ValueError("camera jitter period steps must be positive")
    if has_dynamic_rotation and abs(drift_rotation_degrees) > 0.0:
        raise ValueError("fixed and dynamic camera rotations cannot be combined")
    if has_dynamic_rotation and args.model_camera_calibration == "actual":
        raise ValueError(
            "dynamic camera rendering currently supports nominal model calibration only"
        )
    if (
        abs(drift_rotation_degrees) > 0.0 or has_dynamic_rotation
    ) and np.linalg.norm(drift_axis) <= 1.0e-12:
        raise ValueError("camera drift rotation axis must be non-zero")
    drift_translation_world = drift_translation_cm / 100.0
    has_camera_drift = bool(
        np.linalg.norm(drift_translation_world) > 0.0
        or abs(drift_rotation_degrees) > 0.0
        or has_dynamic_rotation
    )
    camera_rotation_schedule = None
    if has_dynamic_rotation:
        camera_rotation_schedule = partial(
            _dynamic_camera_rotation,
            drift_degrees=dynamic_drift_degrees,
            jitter_degrees=jitter_degrees,
            ramp_steps=int(args.camera_drift_ramp_steps),
            jitter_period_steps=int(args.camera_jitter_period_steps),
        )
    grounding_mode = _resolve_grounding_mode(
        args.grounding_mode,
        orbit_clock_hour=args.orbit_clock_hour,
        allow_orbit_fcpg=bool(args.allow_orbit_fcpg),
    )
    if args.pose_readout_checkpoint is not None and args.orbit_clock_hour is None:
        raise ValueError("pose-conditioned readout requires --orbit-clock-hour")
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    contract = legacy._safe_read_json(args.config)
    validate_a4_contract(contract)
    selected_tasks = tuple(args.task) if args.task else TASKS
    if len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError("--task values must be unique")
    instances = legacy._test_instances_by_rank(
        contract, rank=int(args.test_instance_rank)
    )
    collection = contract["collection"]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for task in selected_tasks:
        task_instance = instances[task]
        environment_seed = legacy._environment_seed(
            contract, task=task, task_instance=task_instance
        )
        orbit_view = None
        camera_name = None
        if args.orbit_clock_hour is not None:
            camera_name = a4_camera_name(task)
            orbit_view = OrbitCameraView(
                clock_hour=int(args.orbit_clock_hour),
                elevation_degrees=float(args.orbit_elevation_degrees),
                radius_m=float(args.orbit_radius_m),
                base_camera_name=camera_name,
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
            orbit_camera_view=orbit_view,
            camera_translation_world=drift_translation_world,
            camera_rotation_axis_world=drift_axis,
            camera_rotation_degrees=drift_rotation_degrees,
            camera_rotation_schedule=camera_rotation_schedule,
        )
        base_camera_label = (
            orbit_view.label if orbit_view is not None else a4_camera_name(task)
        )
        if has_dynamic_rotation:
            episode["camera_label"] = (
                f"{base_camera_label} + dynamic(drift={dynamic_drift_degrees:g}deg,"
                f" jitter={jitter_degrees:g}deg,"
                f" period={int(args.camera_jitter_period_steps)})"
            )
        elif has_camera_drift:
            episode["camera_label"] = (
                f"{base_camera_label} + drift({drift_rotation_degrees:g}deg)"
            )
        else:
            episode["camera_label"] = base_camera_label
        calibration_key = (
            "nominal_camera_extrinsic_cam_from_world"
            if args.model_camera_calibration == "nominal"
            else "camera_extrinsic_cam_from_world"
        )
        episode["model_camera_extrinsic_cam_from_world"] = np.asarray(
            episode[calibration_key], dtype=np.float32
        ).copy()
        episode["orbit_camera_view"] = orbit_view
        episodes.append(episode)
        print(
            json.dumps(
                {
                    "task": task,
                    "raw_frames": len(episode["frames"]),
                    "stable_success_step": episode["stable_success_step"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    corrected_by_task: dict[str, np.ndarray] = {}
    if grounding_mode in {"nominal", "oracle"}:
        for episode in episodes:
            starts = _decision_steps(len(episode["actions"]))
            selected = [episode["records"][int(index)] for index in starts]
            corrected_by_task[str(episode["task"])] = np.stack(
                [
                    record.q_true if grounding_mode == "oracle" else record.q_nominal
                    for record in selected
                ]
            ).astype(np.float32)
    else:
        fcpg = load_frozen_a4_state_encoder(
            args.base_checkpoint.expanduser().resolve(),
            args.fcpg_checkpoint.expanduser().resolve(),
            device=device,
        )
        try:
            for episode in episodes:
                starts = _decision_steps(len(episode["actions"]))
                selected = [episode["records"][int(index)] for index in starts]
                corrected_by_task[str(episode["task"])] = corrected_points(
                    selected, fcpg
                )
        finally:
            del fcpg
            legacy._clear_runtime_memory(device)

    model, checkpoint = _load_checkpoint(args.checkpoint, device=device)
    pose_readout = None
    pose_checkpoint: dict[str, Any] | None = None
    if args.pose_readout_checkpoint is not None:
        pose_readout, pose_checkpoint = _load_pose_readout(
            args.pose_readout_checkpoint,
            base_checkpoint=args.checkpoint,
            device=device,
        )
    entries = []
    manifests = []
    try:
        for episode in episodes:
            task = str(episode["task"])
            starts, predictions = _predict_episode_windows(
                model,
                episode,
                corrected_by_task[task],
                task_index=TASKS.index(task),
                device=device,
                pose_readout=pose_readout,
                orbit_view=episode.get("orbit_camera_view"),
                grounding_residual_transport=str(
                    args.grounding_residual_transport
                ),
                transport_gate_low_px=float(args.transport_gate_low_px),
                transport_gate_high_px=float(args.transport_gate_high_px),
            )
            dense_raw, grid_steps, source_horizons = (
                _stitch_max_lookahead_predictions(
                    starts,
                    predictions,
                    corrected_by_task[task][0],
                    raw_frame_count=len(episode["frames"]),
                )
            )
            truth_raw = np.stack([row.q_true for row in episode["records"]]).astype(
                np.float32
            )
            current_truth_raw = truth_raw[np.asarray(starts, dtype=np.int64)]
            current_grounding_error = np.linalg.norm(
                corrected_by_task[task] - current_truth_raw,
                axis=-1,
            )
            current_grounding_center_error = np.linalg.norm(
                corrected_by_task[task].mean(axis=1)
                - current_truth_raw.mean(axis=1),
                axis=-1,
            )
            current_nominal_raw = np.stack(
                [
                    episode["records"][int(index)].q_nominal
                    for index in np.asarray(starts, dtype=np.int64)
                ]
            ).astype(np.float32)
            current_grounding_innovation = np.linalg.norm(
                corrected_by_task[task] - current_nominal_raw,
                axis=-1,
            )
            height = int(collection["height"])
            flip_vertical = args.orbit_clock_hour is None and task not in CUSTOM_TASKS
            truth = _display_points(
                truth_raw,
                task=task,
                height=height,
                flip_vertical=flip_vertical,
            )
            dense = _display_points(
                dense_raw,
                task=task,
                height=height,
                flip_vertical=flip_vertical,
            )
            displayed_predictions = _display_points(
                predictions,
                task=task,
                height=height,
                flip_vertical=flip_vertical,
            )
            displayed_corrected = _display_points(
                corrected_by_task[task],
                task=task,
                height=height,
                flip_vertical=flip_vertical,
            )
            horizon_metrics = _horizon_metrics(episode, starts, predictions)
            anchor_error = np.linalg.norm(dense - truth, axis=-1)
            center_error = np.linalg.norm(
                dense.mean(axis=1) - truth.mean(axis=1), axis=-1
            )
            if args.metrics_only:
                video = ""
                gif = ""
                video_frame_count = 0
                gif_frame_count = 0
            else:
                expected_video_frames = _prediction_cycle_frame_count(
                    starts,
                    end_step=len(episode["frames"]) - 1,
                )
                video, gif, video_frame_count, gif_frame_count = (
                    _save_prediction_cycle_media(
                        output_dir,
                        task=task,
                        frames=_prediction_cycle_frames(
                            episode,
                            starts,
                            displayed_corrected,
                            displayed_predictions,
                            truth,
                            task_label=trajectory.TASK_LABELS[task],
                        ),
                        fps=float(args.fps),
                        expected_frame_count=expected_video_frames,
                    )
                )
            common = {
                "task": task,
                "task_label": trajectory.TASK_LABELS[task],
                "task_instance": int(episode["task_instance"]),
                "environment_seed": int(episode["environment_seed"]),
                "raw_frame_count": len(episode["frames"]),
                "rendered_frame_count": int(video_frame_count),
                "stable_success_step": int(episode["stable_success_step"]),
                "video": video,
                "gif": gif,
                "video_fps": float(args.fps),
                "gif_frame_count": int(gif_frame_count),
                "prediction_playback": "advance_actual_5_then_forecast_40",
                "camera_label": str(episode["camera_label"]),
                "grounding_mode": grounding_mode,
                "horizon_metrics": horizon_metrics,
                "current_grounding_anchor_mean_epe_px": float(
                    current_grounding_error.mean()
                ),
                "current_grounding_anchor_p95_epe_px": float(
                    np.quantile(current_grounding_error, 0.95)
                ),
                "current_grounding_center_mean_epe_px": float(
                    current_grounding_center_error.mean()
                ),
                "current_grounding_innovation_mean_px": float(
                    current_grounding_innovation.mean()
                ),
                "current_grounding_innovation_p95_px": float(
                    np.quantile(current_grounding_innovation, 0.95)
                ),
                "stitched_anchor_mean_epe_px": float(anchor_error.mean()),
                "stitched_anchor_p95_epe_px": float(np.quantile(anchor_error, 0.95)),
                "stitched_center_mean_epe_px": float(center_error.mean()),
            }
            entries.append(common)
            manifests.append(
                {
                    **common,
                    "decision_window_count": int(len(starts)),
                    "first_success_step": episode["first_success_step"],
                    "stop_reason": episode["stop_reason"],
                }
            )
            print(
                json.dumps(
                    {
                        "task": task,
                        "stitched_anchor_epe_px": float(anchor_error.mean()),
                        "video": video,
                        "video_frames": int(video_frame_count),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del model
        if pose_readout is not None:
            del pose_readout
        legacy._clear_runtime_memory(device)

    metadata = {
        "seed": int(checkpoint["seed"]),
        "step": int(checkpoint["step"]),
        "sigreg_weight": float(checkpoint["sigreg_weight"]),
        "camera": (
            "task_default"
            if args.orbit_clock_hour is None
            else OrbitCameraView(
                clock_hour=int(args.orbit_clock_hour),
                elevation_degrees=float(args.orbit_elevation_degrees),
                radius_m=float(args.orbit_radius_m),
            ).label
        ),
        "grounding_mode": grounding_mode,
        "model_camera_calibration": str(args.model_camera_calibration),
        "grounding_residual_transport": str(args.grounding_residual_transport),
        "transport_gate_low_px": float(args.transport_gate_low_px),
        "transport_gate_high_px": float(args.transport_gate_high_px),
        "metrics_only": bool(args.metrics_only),
        "test_instance_rank": int(args.test_instance_rank),
        "camera_drift": {
            "rotation_degrees": drift_rotation_degrees,
            "rotation_axis_world": drift_axis.tolist(),
            "translation_cm": drift_translation_cm.tolist(),
            "dynamic_drift_degrees": dynamic_drift_degrees,
            "jitter_degrees": jitter_degrees,
            "drift_ramp_steps": int(args.camera_drift_ramp_steps),
            "jitter_period_steps": int(args.camera_jitter_period_steps),
        },
        "checkpoint_protocol": str(checkpoint.get("protocol", "")),
        "uses_action_history": (
            checkpoint.get("protocol") == HISTORY_CONDITIONED_PROTOCOL
        ),
        "pose_conditioned_readout": pose_checkpoint is not None,
        "pose_conditioned_joint_model": (
            checkpoint.get("protocol") == POSE_JOINT_PROTOCOL
        ),
        "pose_readout_step": (
            None if pose_checkpoint is None else int(pose_checkpoint["step"])
        ),
    }
    (output_dir / "index.html").write_text(
        _render_html(entries, metadata), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "split": "reserved_test",
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "pose_readout_checkpoint": (
                    None
                    if args.pose_readout_checkpoint is None
                    else str(args.pose_readout_checkpoint.expanduser().resolve())
                ),
                "metadata": metadata,
                "tasks": {entry["task"]: entry for entry in entries},
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "model_input": [
                    "current_rgb",
                    "current_proprio",
                    (
                        "current_oracle_anchors"
                        if grounding_mode == "oracle"
                        else "current_grounded_anchors"
                    ),
                    *(
                        ["past_two_executed_action_chunks"]
                        if checkpoint.get("protocol")
                        == HISTORY_CONDITIONED_PROTOCOL
                        else []
                    ),
                    "causal_H1_H8_action_prefix",
                    "task_identity",
                    *(
                        ["continuous_orbit_camera_basis"]
                        if pose_checkpoint is not None
                        or checkpoint.get("protocol") == POSE_JOINT_PROTOCOL
                        else []
                    ),
                ],
                "camera": metadata["camera"],
                "grounding_mode": grounding_mode,
                "model_camera_calibration": str(args.model_camera_calibration),
                "grounding_residual_transport": str(
                    args.grounding_residual_transport
                ),
                "transport_gate_low_px": float(args.transport_gate_low_px),
                "transport_gate_high_px": float(args.transport_gate_high_px),
                "metrics_only": bool(args.metrics_only),
                "actual_camera_extrinsic": (
                    "model_input"
                    if args.model_camera_calibration == "actual"
                    else "evaluation_only"
                ),
                "uses_current_q_true": grounding_mode == "oracle",
                "future_rgb": "evaluation_only",
                "future_q_true": "evaluation_only",
                "entries": manifests,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
