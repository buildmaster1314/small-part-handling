#!/usr/bin/env python3
"""Render full reset-to-success GT versus decoded A4 LeWM trajectories.

The A4 mechanism Gate uses H=8 (40 raw environment steps), but the model's
recursive rollout API is not limited to H=8.  This diagnostic deliberately
runs each scripted Expert from reset through stable success plus a short tail,
then recursively forecasts every complete five-action chunk after a three-frame
history.  H9+ is shown as open-loop extrapolation, not as validated evidence.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_checkpoint import validate_a4_checkpoint  # noqa: E402
from fcpg.lewm_a4_task_factory import CUSTOM_TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    HISTORY_STEPS,
    HORIZON,
    RAW_STEPS_PER_MODEL_STEP,
    TASKS,
    assign_a4_splits,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    _bounded_expert_action,
    _success,
    load_frozen_a4_state_encoder,
    make_a4_episode_runtime,
)
from fcpg.lewm_action_branch_a4_training import (  # noqa: E402
    initialize_a4_model_from_anchor_base,
)
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    get_camera_id,
    set_fixed_camera_pose_offset,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_BASE_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_A4_CHECKPOINTS = tuple(
    ROOT
    / f"outputs/lewm_action_branch_a4/shared_7task_seed{seed}_2k/"
    "lewm_action_branch_a4_latest.pt"
    for seed in (30_201, 30_202, 30_203)
)
DEFAULT_DECODER = ROOT / "outputs/lewm_action_branch_a4/direction_readout/decoder.npz"
DEFAULT_METRICS = ROOT / "outputs/lewm_action_branch_a4/direction_readout/metrics.json"
DEFAULT_DEVELOPMENT_GATE = (
    ROOT / "outputs/lewm_action_branch_a4/development_gate/summary.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/full_trajectory_preview"

MAX_RAW_STEPS = 500
SUCCESS_STREAK = 3
POST_SUCCESS_RAW_STEPS = 8
GROUND_TRUTH_COLOR = (31, 151, 91)
LEWM_COLOR = (35, 105, 191)
TASK_LABELS = {
    "pick-place-v3": "Pick & Place",
    "drawer-retrieve-v0": "Drawer Retrieve",
    "reach-v3": "Reach",
    "push-v3": "Push",
    "door-open-v3": "Door Open",
    "window-open-v3": "Window Open",
    "towel-wipe-v0": "Towel Wipe",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument(
        "--a4-checkpoint",
        type=Path,
        action="append",
        dest="a4_checkpoints",
        help="2k A4 checkpoint; repeat to form the prediction ensemble",
    )
    parser.add_argument(
        "--expected-checkpoint-step",
        type=int,
        default=2_000,
        help="Require every A4 ensemble checkpoint to have this optimizer step",
    )
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--development-gate", type=Path, default=DEFAULT_DEVELOPMENT_GATE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--layout",
        choices=("overlay", "split"),
        default="overlay",
        help=(
            "overlay draws GT and prediction on actual RGB; split keeps actual "
            "future RGB on the left and freezes the prediction background at t0"
        ),
    )
    parser.add_argument("--max-raw-steps", type=int, default=MAX_RAW_STEPS)
    parser.add_argument(
        "--post-success-steps", type=int, default=POST_SUCCESS_RAW_STEPS
    )
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument(
        "--allow-failed-development-diagnostic",
        action="store_true",
        help="Required because the train-only latent-to-pixel readout Gate failed",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _torch_load(path: Path) -> dict[str, Any]:
    value = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one checkpoint mapping")
    return value


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _stage(expert: Any, observation: np.ndarray) -> str:
    get_phase = getattr(expert, "get_phase", None)
    if not callable(get_phase):
        return "active control"
    phase = get_phase(observation)
    return str(getattr(phase, "value", phase))


def _upright_frame_and_center(
    record: Any,
    *,
    task: str,
    flip_vertical: bool | None = None,
) -> tuple[Image.Image, np.ndarray]:
    image = np.asarray(record.rgb, dtype=np.float32)
    points = np.asarray(record.q_true, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("A4 frame RGB must have shape [3,H,W]")
    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise ValueError("A4 q_true must be finite [3,2]")
    rgb = np.clip(np.transpose(image, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    center = np.mean(points, axis=0).astype(np.float32)
    should_flip = task not in CUSTOM_TASKS if flip_vertical is None else bool(flip_vertical)
    if should_flip:
        rgb = np.flip(rgb, axis=0).copy()
        center[1] = float(rgb.shape[0] - 1) - center[1]
    return Image.fromarray(rgb, mode="RGB"), center


def _first_development_instances(contract: Mapping[str, Any]) -> dict[str, int]:
    assignment = assign_a4_splits(TASKS, contract)
    result: dict[str, int] = {}
    for task in TASKS:
        candidates = sorted(
            instance
            for (name, instance), split in assignment.items()
            if name == task and split == "development"
        )
        if not candidates:
            raise RuntimeError(f"A4 has no development instance for {task}")
        result[task] = int(candidates[0])
    return result


def _collect_full_episode(
    task: str,
    *,
    task_instance: int,
    environment_seed: int,
    width: int,
    height: int,
    benchmark_seed: int,
    max_raw_steps: int,
    post_success_steps: int,
    camera_name: str | None = None,
    orbit_camera_view: Any | None = None,
    camera_translation_world: np.ndarray | None = None,
    camera_rotation_axis_world: np.ndarray | None = None,
    camera_rotation_degrees: float = 0.0,
    camera_rotation_schedule: Callable[[int], float] | None = None,
) -> dict[str, Any]:
    runtime = make_a4_episode_runtime(
        task,
        task_instance,
        environment_seed,
        width=width,
        height=height,
        benchmark_seed=benchmark_seed,
        camera_name=camera_name,
        orbit_camera_view=orbit_camera_view,
    )
    nominal_camera_extrinsic = np.asarray(
        runtime.camera_extrinsic_cam_from_world, dtype=np.float32
    ).copy()
    translation = np.zeros(3, dtype=np.float64)
    if camera_translation_world is not None:
        translation = np.asarray(camera_translation_world, dtype=np.float64)
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if camera_rotation_axis_world is not None:
        axis = np.asarray(camera_rotation_axis_world, dtype=np.float64)
    if translation.shape != (3,) or axis.shape != (3,):
        runtime.env.close()
        raise ValueError("camera translation and rotation axis must have shape [3]")
    if not np.isfinite(translation).all() or not np.isfinite(axis).all():
        runtime.env.close()
        raise ValueError("camera offset values must be finite")
    rotation_degrees = float(camera_rotation_degrees)
    if not np.isfinite(rotation_degrees):
        runtime.env.close()
        raise ValueError("camera rotation must be finite")
    if camera_rotation_schedule is not None and abs(rotation_degrees) > 0.0:
        runtime.env.close()
        raise ValueError(
            "camera_rotation_schedule cannot be combined with a fixed rotation"
        )
    has_offset = bool(
        np.linalg.norm(translation) > 0.0
        or abs(rotation_degrees) > 0.0
        or camera_rotation_schedule is not None
    )
    camera_id: int | None = None
    base_position: np.ndarray | None = None
    base_quaternion: np.ndarray | None = None
    if has_offset:
        if abs(rotation_degrees) > 0.0 and float(np.linalg.norm(axis)) <= 1.0e-12:
            runtime.env.close()
            raise ValueError("camera rotation axis must be non-zero")
        camera_id = get_camera_id(runtime.env.model, runtime.camera_name)
        base_position = np.asarray(
            runtime.env.model.cam_pos[camera_id], dtype=np.float64
        ).copy()
        base_quaternion = np.asarray(
            runtime.env.model.cam_quat[camera_id], dtype=np.float64
        ).copy()
        if camera_rotation_schedule is None:
            set_fixed_camera_pose_offset(
                runtime.env.model,
                runtime.env.data,
                camera_id,
                base_position,
                base_quaternion,
                translation,
                axis,
                rotation_degrees,
            )
            runtime.camera_extrinsic_cam_from_world = np.asarray(
                camera_extrinsic_cv_from_mujoco(runtime.env.data, camera_id),
                dtype=np.float32,
            )
    records = []
    frames: list[Image.Image] = []
    centers = []
    actions = []
    stages = []
    successes = []
    rewards = []
    camera_rotation_by_frame: list[float] = []
    observation = np.asarray(runtime.observation, dtype=np.float32)
    info = dict(runtime.reset_info)
    reward = 0.0
    first_success: int | None = None
    stable_success: int | None = None
    streak = 0
    stop_reason = "max_raw_steps"
    try:
        for step in range(max_raw_steps + 1):
            frame_rotation_degrees = rotation_degrees
            if camera_rotation_schedule is not None:
                frame_rotation_degrees = float(camera_rotation_schedule(step))
                if not np.isfinite(frame_rotation_degrees):
                    raise ValueError("camera rotation schedule returned a non-finite angle")
                if abs(frame_rotation_degrees) > 0.0 and float(np.linalg.norm(axis)) <= 1.0e-12:
                    raise ValueError("camera rotation axis must be non-zero")
                if camera_id is None or base_position is None or base_quaternion is None:
                    raise RuntimeError("dynamic camera state was not initialized")
                set_fixed_camera_pose_offset(
                    runtime.env.model,
                    runtime.env.data,
                    camera_id,
                    base_position,
                    base_quaternion,
                    translation,
                    axis,
                    frame_rotation_degrees,
                )
                runtime.camera_extrinsic_cam_from_world = np.asarray(
                    camera_extrinsic_cv_from_mujoco(runtime.env.data, camera_id),
                    dtype=np.float32,
                )
            record = runtime.record_current(observation)
            frame, center = _upright_frame_and_center(
                record,
                task=task,
                flip_vertical=(task not in CUSTOM_TASKS and orbit_camera_view is None),
            )
            current_success = _success(info, runtime.env)
            records.append(record)
            frames.append(frame)
            centers.append(center)
            stages.append(_stage(runtime.expert, observation))
            successes.append(current_success)
            rewards.append(float(reward))
            camera_rotation_by_frame.append(frame_rotation_degrees)
            if current_success:
                first_success = step if first_success is None else first_success
                streak += 1
            else:
                streak = 0
            if streak >= SUCCESS_STREAK and stable_success is None:
                stable_success = step
            if stable_success is not None and step >= stable_success + post_success_steps:
                stop_reason = "post_success_tail"
                break
            if step >= max_raw_steps:
                break
            action = _bounded_expert_action(runtime, observation)
            actions.append(action)
            observation, reward, terminated, truncated, step_info = runtime.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            info = dict(step_info) if isinstance(step_info, Mapping) else {}
            if terminated or truncated:
                stop_reason = "terminated" if terminated else "truncated"
                break
    finally:
        runtime.env.close()
    action_array = np.asarray(actions, dtype=np.float32)
    center_array = np.asarray(centers, dtype=np.float32)
    if len(records) != len(frames) or center_array.shape != (len(frames), 2):
        raise RuntimeError("full episode frame/record/center lengths disagree")
    if action_array.shape != (len(frames) - 1, 4):
        raise RuntimeError("full episode must contain T frames and T-1 actions")
    if stable_success is None:
        raise RuntimeError(f"{task} did not reach stable success by raw step {max_raw_steps}")
    return {
        "task": task,
        "task_instance": task_instance,
        "environment_seed": environment_seed,
        "camera_name": str(runtime.camera_name),
        "camera_intrinsic": np.asarray(
            runtime.camera_intrinsic, dtype=np.float32
        ).copy(),
        "nominal_camera_extrinsic_cam_from_world": nominal_camera_extrinsic,
        "camera_extrinsic_cam_from_world": np.asarray(
            runtime.camera_extrinsic_cam_from_world, dtype=np.float32
        ).copy(),
        "camera_translation_world": translation.astype(np.float32),
        "camera_rotation_axis_world": axis.astype(np.float32),
        "camera_rotation_degrees": rotation_degrees,
        "camera_rotation_degrees_by_frame": np.asarray(
            camera_rotation_by_frame, dtype=np.float32
        ),
        "records": records,
        "frames": frames,
        "q_center_display": center_array,
        "actions": action_array,
        "stages": stages,
        "successes": successes,
        "rewards": rewards,
        "first_success_step": first_success,
        "stable_success_step": stable_success,
        "stop_reason": stop_reason,
    }


def _model_steps(raw_action_count: int) -> np.ndarray:
    last_complete = (int(raw_action_count) // RAW_STEPS_PER_MODEL_STEP) * RAW_STEPS_PER_MODEL_STEP
    result = np.arange(0, last_complete + 1, RAW_STEPS_PER_MODEL_STEP, dtype=np.int64)
    if result.size < HISTORY_STEPS + 1:
        raise ValueError("full trajectory is too short for three history states and one prediction")
    return result


def _chunk_actions(actions: np.ndarray, starts: Sequence[int]) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    chunks = [
        values[start : start + RAW_STEPS_PER_MODEL_STEP].reshape(-1)
        for start in starts
    ]
    result = np.asarray(chunks, dtype=np.float32)
    expected = (len(chunks), RAW_STEPS_PER_MODEL_STEP * 4)
    if result.shape != expected:
        raise RuntimeError(f"action chunk shape drifted: {result.shape} != {expected}")
    return result


@torch.inference_mode()
def _rollout_one_model(
    model: Any,
    *,
    history_embedding: np.ndarray,
    actions: np.ndarray,
    task_index: int,
    prediction_count: int,
    device: torch.device,
) -> np.ndarray:
    history = torch.from_numpy(np.asarray(history_embedding, dtype=np.float32))[None].to(device)
    history_actions = _chunk_actions(actions, (0, 5))
    future_starts = tuple(10 + 5 * index for index in range(prediction_count))
    future_actions = _chunk_actions(actions, future_starts)
    prediction = model.rollout_state_embeddings(
        history,
        torch.from_numpy(history_actions)[None].to(device),
        torch.from_numpy(future_actions)[None, None].to(device),
        torch.tensor([task_index], dtype=torch.long, device=device),
    )
    expected = (1, 1, prediction_count, 4, 192)
    if tuple(prediction.shape) != expected:
        raise RuntimeError(f"full LeWM prediction shape drifted: {tuple(prediction.shape)}")
    return prediction[0, 0].float().cpu().numpy()


def _load_decoder(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        required = {"coefficient_raw", "intercept_raw", "fit_split"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"direction decoder is missing {missing}")
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if str(values["fit_split"].item()) != "train":
        raise PermissionError("full visualization accepts only a train-only decoder")
    coefficient = np.asarray(values["coefficient_raw"])
    intercept = np.asarray(values["intercept_raw"])
    if coefficient.shape == (3 * 192, 2):
        if intercept.shape != (2,):
            raise ValueError("global direction decoder intercept must have shape [2]")
    elif coefficient.shape == (len(TASKS), 3 * 192, 2):
        if intercept.shape != (len(TASKS), 2):
            raise ValueError(
                "task-conditioned direction decoder intercept must have shape [7,2]"
            )
        task_names = np.asarray(values.get("task_names"))
        if task_names.shape != (len(TASKS),) or tuple(task_names.astype(str)) != TASKS:
            raise ValueError("task-conditioned decoder task_names differ from A4")
    else:
        raise ValueError(
            "direction decoder coefficient must have shape [576,2] or [7,576,2]"
        )
    return values


def _decoder_parameters(
    decoder: Mapping[str, np.ndarray], task: str
) -> tuple[np.ndarray, np.ndarray]:
    coefficient = np.asarray(decoder["coefficient_raw"], dtype=np.float64)
    intercept = np.asarray(decoder["intercept_raw"], dtype=np.float64)
    if coefficient.ndim == 2:
        return coefficient, intercept
    task_names = tuple(np.asarray(decoder["task_names"]).astype(str).tolist())
    try:
        index = task_names.index(task)
    except ValueError as error:
        raise ValueError(f"direction decoder does not contain task {task!r}") from error
    return coefficient[index], intercept[index]


def _decode_open_loop(
    prediction: np.ndarray,
    history: np.ndarray,
    q_start_display: np.ndarray,
    *,
    task: str,
    height: int,
    decoder: Mapping[str, np.ndarray],
) -> np.ndarray:
    future = np.asarray(prediction, dtype=np.float64)
    baseline = np.asarray(history[-1], dtype=np.float64)
    feature = (future[:, 1:] - baseline[None, 1:]).reshape(future.shape[0], -1)
    coefficient, intercept = _decoder_parameters(decoder, task)
    displacement_raw = feature @ coefficient + intercept
    if task not in CUSTOM_TASKS:
        displacement_raw[:, 1] *= -1.0
    result = np.concatenate(
        [np.asarray(q_start_display, dtype=np.float64)[None], q_start_display + displacement_raw],
        axis=0,
    )
    if result.shape != (future.shape[0] + 1, 2) or not np.isfinite(result).all():
        raise RuntimeError("decoded full LeWM path is invalid")
    return result.astype(np.float32)


def _dense_prediction(
    model_steps: np.ndarray,
    model_points: np.ndarray,
    *,
    raw_frame_count: int,
) -> np.ndarray:
    steps = np.asarray(model_steps, dtype=np.int64)
    points = np.asarray(model_points, dtype=np.float64)
    if steps.ndim != 1 or points.shape != (steps.size, 2) or steps.size < 2:
        raise ValueError("model steps and points must be aligned non-empty paths")
    if not np.all(np.diff(steps) == RAW_STEPS_PER_MODEL_STEP):
        raise ValueError("LeWM model nodes must be spaced by five raw frames")
    result = np.full((raw_frame_count, 2), np.nan, dtype=np.float32)
    visible = np.arange(steps[0], raw_frame_count, dtype=np.float64)
    for axis in range(2):
        result[steps[0] :, axis] = np.interp(
            visible,
            steps.astype(np.float64),
            points[:, axis],
        ).astype(np.float32)
    return result


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    *,
    fill: tuple[int, int, int],
    width: int,
    dash: float = 7.0,
    gap: float = 4.0,
) -> None:
    values = np.asarray(points, dtype=np.float64)
    for start, end in zip(values[:-1], values[1:], strict=True):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        direction = delta / length
        cursor = 0.0
        while cursor < length:
            segment_end = min(cursor + dash, length)
            draw.line(
                (tuple(start + direction * cursor), tuple(start + direction * segment_end)),
                fill=fill,
                width=width,
            )
            cursor += dash + gap


def _draw_frame(
    frame: Image.Image,
    *,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    raw_step: int,
    end_step: int,
    task_label: str,
    success: bool,
) -> Image.Image:
    canvas = frame.copy()
    draw = ImageDraw.Draw(canvas)
    gt = np.asarray(ground_truth[: raw_step + 1], dtype=np.float64)
    if gt.shape[0] >= 2:
        draw.line([tuple(value) for value in gt], fill=GROUND_TRUTH_COLOR, width=4)
    pred = np.asarray(predicted[: raw_step + 1], dtype=np.float64)
    pred = pred[np.isfinite(pred).all(axis=1)]
    if pred.shape[0] >= 2:
        _dashed_line(draw, pred, fill=LEWM_COLOR, width=4)
    current_gt = gt[-1]
    draw.ellipse(
        (current_gt[0] - 5, current_gt[1] - 5, current_gt[0] + 5, current_gt[1] + 5),
        fill=GROUND_TRUTH_COLOR,
        outline=(255, 255, 255),
        width=2,
    )
    if pred.shape[0]:
        current_pred = pred[-1]
        draw.ellipse(
            (current_pred[0] - 5, current_pred[1] - 5, current_pred[0] + 5, current_pred[1] + 5),
            fill=LEWM_COLOR,
            outline=(255, 255, 255),
            width=2,
        )
    draw.rectangle((5, 5, 219, 49), fill=(247, 249, 250))
    draw.text((10, 8), task_label, fill=(25, 31, 37), font=_font(12))
    draw.text(
        (10, 27),
        f"raw {raw_step:03d}/{end_step:03d} | success {int(success)}",
        fill=(65, 73, 81),
        font=_font(10),
    )
    draw.line((128, 14, 148, 14), fill=GROUND_TRUTH_COLOR, width=4)
    draw.text((153, 8), "GT", fill=(35, 45, 54), font=_font(9))
    _dashed_line(
        draw,
        np.asarray([[128.0, 35.0], [148.0, 35.0]]),
        fill=LEWM_COLOR,
        width=4,
    )
    draw.text((153, 27), "LeWM", fill=(35, 45, 54), font=_font(9))
    return canvas


def _draw_split_frame(
    frame: Image.Image,
    *,
    context_frame: Image.Image,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    raw_step: int,
    end_step: int,
    context_step: int,
    task_label: str,
    success: bool,
) -> Image.Image:
    """Render actual future and recursive prediction without future-RGB leakage."""

    if frame.size != context_frame.size:
        raise ValueError("actual and context frames must have identical dimensions")
    width, height = frame.size
    if context_step < 0 or context_step > end_step:
        raise ValueError("context_step must lie inside the rendered episode")

    gt_panel = frame.copy()
    gt_draw = ImageDraw.Draw(gt_panel)
    gt = np.asarray(ground_truth[: raw_step + 1], dtype=np.float64)
    if gt.shape[0] >= 2:
        gt_draw.line([tuple(value) for value in gt], fill=GROUND_TRUTH_COLOR, width=3)
    current_gt = gt[-1]
    gt_draw.ellipse(
        (
            current_gt[0] - 5,
            current_gt[1] - 5,
            current_gt[0] + 5,
            current_gt[1] + 5,
        ),
        fill=GROUND_TRUTH_COLOR,
        outline=(255, 255, 255),
        width=2,
    )

    if raw_step <= context_step:
        prediction_panel = frame.copy()
    else:
        gray = context_frame.convert("L").convert("RGB")
        prediction_panel = Image.blend(gray, Image.new("RGB", frame.size, (226, 230, 232)), 0.35)
    pred_draw = ImageDraw.Draw(prediction_panel)
    pred = np.asarray(predicted[: raw_step + 1], dtype=np.float64)
    pred = pred[np.isfinite(pred).all(axis=1)]
    if pred.shape[0] >= 2:
        _dashed_line(pred_draw, pred, fill=LEWM_COLOR, width=3)
    if pred.shape[0]:
        current_pred = pred[-1]
        pred_draw.ellipse(
            (
                current_pred[0] - 5,
                current_pred[1] - 5,
                current_pred[0] + 5,
                current_pred[1] + 5,
            ),
            fill=LEWM_COLOR,
            outline=(255, 255, 255),
            width=2,
        )

    scale = 2
    panel_width = width * scale
    panel_height = height * scale
    margin = 12
    gap = 12
    header_height = 66
    footer_height = 44
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + gap + panel_width * 2,
            header_height + panel_height + footer_height,
        ),
        (238, 241, 243),
    )
    gt_panel = gt_panel.resize((panel_width, panel_height), Image.Resampling.BICUBIC)
    prediction_panel = prediction_panel.resize(
        (panel_width, panel_height), Image.Resampling.BICUBIC
    )
    left_x = margin
    right_x = margin + panel_width + gap
    canvas.paste(gt_panel, (left_x, header_height))
    canvas.paste(prediction_panel, (right_x, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(17)
    label_font = _font(13)
    small_font = _font(11)
    draw.text((margin, 8), task_label, fill=(23, 32, 40), font=title_font)
    draw.text(
        (margin, 35),
        "ACTUAL FUTURE (evaluation only)",
        fill=GROUND_TRUTH_COLOR,
        font=label_font,
    )
    draw.text(
        (right_x, 35),
        "LEWM RECURSIVE ANCHOR FORECAST",
        fill=LEWM_COLOR,
        font=label_font,
    )
    draw.rectangle(
        (
            right_x + panel_width - 166,
            header_height + 8,
            right_x + panel_width - 8,
            header_height + 31,
        ),
        fill=(248, 249, 250),
    )
    right_status = (
        "collecting context"
        if raw_step < context_step
        else f"RGB frozen at raw {context_step:03d}"
    )
    draw.text(
        (right_x + panel_width - 159, header_height + 12),
        right_status,
        fill=(55, 66, 75),
        font=small_font,
    )
    if raw_step <= context_step:
        horizon = 0
        phase = "observed context"
    else:
        horizon = int(np.ceil((raw_step - context_step) / RAW_STEPS_PER_MODEL_STEP))
        phase = "H1-H8 trained horizon" if horizon <= HORIZON else "H9+ extrapolation"
    footer_y = header_height + panel_height + 12
    draw.text(
        (margin, footer_y),
        f"raw {raw_step:03d}/{end_step:03d}  |  forecast H{horizon:02d}  |  {phase}",
        fill=(49, 60, 69),
        font=label_font,
    )
    draw.text(
        (margin + panel_width - 140, footer_y),
        f"actual success {int(success)}",
        fill=(49, 60, 69),
        font=label_font,
    )
    return canvas


def _save_animation(
    output_dir: Path,
    *,
    task: str,
    frames: Sequence[Image.Image],
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    successes: Sequence[bool],
    fps: float,
    layout: str = "overlay",
    context_step: int = (HISTORY_STEPS - 1) * RAW_STEPS_PER_MODEL_STEP,
) -> str:
    if layout == "overlay":
        animation = [
            _draw_frame(
                frame,
                ground_truth=ground_truth,
                predicted=predicted,
                raw_step=index,
                end_step=len(frames) - 1,
                task_label=TASK_LABELS[task],
                success=bool(successes[index]),
            )
            for index, frame in enumerate(frames)
        ]
        suffix = "full_gt_vs_lewm"
    elif layout == "split":
        if context_step >= len(frames):
            raise ValueError("context step exceeds the full episode")
        context_frame = frames[context_step]
        animation = [
            _draw_split_frame(
                frame,
                context_frame=context_frame,
                ground_truth=ground_truth,
                predicted=predicted,
                raw_step=index,
                end_step=len(frames) - 1,
                context_step=context_step,
                task_label=TASK_LABELS[task],
                success=bool(successes[index]),
            )
            for index, frame in enumerate(frames)
        ]
        suffix = "full_actual_vs_recursive_forecast"
    else:
        raise ValueError(f"unknown animation layout {layout!r}")
    name = f"{task.replace('-', '_')}_{suffix}.gif"
    duration = max(40, int(round(1000.0 / fps)))
    animation[0].save(
        output_dir / name,
        format="GIF",
        save_all=True,
        append_images=animation[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    return name


def _path_metrics(
    ground_truth: np.ndarray,
    model_steps: np.ndarray,
    model_points: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(ground_truth, dtype=np.float64)[model_steps]
    estimate = np.asarray(model_points, dtype=np.float64)
    error = np.linalg.norm(estimate[1:] - truth[1:], axis=-1)
    near_count = min(HORIZON, error.size)
    extrapolated = error[near_count:]
    return {
        "model_prediction_nodes": int(error.size),
        "in_training_horizon_nodes": int(near_count),
        "extrapolation_nodes": int(extrapolated.size),
        "mean_epe_px": float(np.mean(error)),
        "h1_h8_mean_epe_px": float(np.mean(error[:near_count])),
        "h9_plus_mean_epe_px": (
            None if extrapolated.size == 0 else float(np.mean(extrapolated))
        ),
    }


def _percentage(comparisons: Mapping[str, Any], name: str) -> str:
    row = comparisons.get(name, {})
    value = row.get("relative_improvement") if isinstance(row, Mapping) else None
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _render_html(
    entries: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    development_gate: Mapping[str, Any],
    *,
    checkpoint_step: int,
) -> str:
    a4c_methods = metrics.get("methods", {})
    if not isinstance(a4c_methods, Mapping):
        a4c_methods = {}
    if not a4c_methods and isinstance(metrics.get("primary"), Mapping):
        a4c_methods = metrics["primary"].get("methods", {})
    is_a4c_summary = isinstance(a4c_methods, Mapping) and "factual" in a4c_methods
    target = metrics.get("target_ceiling", {})
    factual = metrics.get("factual", {})
    primary = development_gate.get("primary", {})
    comparisons = primary.get("comparisons", {}) if isinstance(primary, Mapping) else {}
    if is_a4c_summary:
        factual_task_macro = float(
            a4c_methods["factual"].get("task_macro_mse", float("nan"))
        )
        no_action_task_macro = float(
            a4c_methods.get("no_action", {}).get("task_macro_mse", float("nan"))
        )
        wrong_action_task_macro = float(
            a4c_methods.get("wrong_action", {}).get(
                "task_macro_mse", float("nan")
            )
        )
        no_action_relative = float(
            comparisons.get("no_action", {}).get("relative_improvement", float("nan"))
        )
        wrong_action_relative = float(
            comparisons.get("wrong_action_mean", {}).get(
                "relative_improvement", float("nan")
            )
        )
    panels = []
    for entry in entries:
        long_epe = entry["h9_plus_mean_epe_px"]
        panels.append(
            f"""
      <section>
        <div class="heading"><div><h2>{html.escape(TASK_LABELS[str(entry['task'])])}</h2>
        <p>development instance {entry['task_instance']} &middot; {entry['frame_count']} continuous raw frames &middot; first success {entry['first_success_step']} &middot; stable success {entry['stable_success_step']}</p></div>
        <div class="numbers"><b>H1-H8 {entry['h1_h8_mean_epe_px']:.2f}px</b><span>H9+ {'n/a' if long_epe is None else f'{float(long_epe):.2f}px'}</span></div></div>
        <img src="{html.escape(str(entry['gif']))}" alt="{html.escape(str(entry['task']))} full GT versus LeWM trajectory">
      </section>"""
        )
    if is_a4c_summary:
        metric_cards = (
            f'<div><b>{factual_task_macro:.4f}</b><span>A4c pooled factual H6-H8 MSE</span></div>'
            f'<div><b>{no_action_task_macro:.4f}</b><span>pooled no-action MSE</span></div>'
            f'<div><b>{wrong_action_task_macro:.4f}</b><span>pooled wrong-action MSE</span></div>'
            f'<div><b>{100.0 * no_action_relative:.1f}% / {100.0 * wrong_action_relative:.1f}%</b><span>factual improvement vs no-action / wrong-action</span></div>'
        )
        warning = (
            "<b>DEVELOPMENT VISUALIZATION - NOT SEALED TEST EVIDENCE</b><br>"
            "蓝线是 train-only 线性读出器从 LeWM latent 解码的像素轨迹；它用于人眼检查连续预测，不是 RGB 生成。"
            "A4c dynamics Gate 的 pooled 统计来自独立 branch 数据，但本页面使用 development episode 回放，不能替代正式 test。"
            "前 10 个 raw step 用于三帧历史，之后每 5 个真实动作产生一个 LeWM 节点；H9+ 是 open-loop extrapolation。"
        )
    else:
        metric_cards = (
            f"<div><b>{float(target.get('macro_mean_epe', float('nan'))):.2f}px</b><span>原 H8 target ceiling</span></div>"
            f"<div><b>{float(factual.get('macro_mean_epe', float('nan'))):.2f}px</b><span>原 2k H8 factual readout</span></div>"
            f"<div><b>{_percentage(comparisons, 'wrong_action_mean')}</b><span>原 2k vs wrong-action mean</span></div>"
            "<div><b>diagnostic</b><span>readout Gate boundary</span></div>"
        )
        warning = (
            "<b>DIAGNOSTIC - GATE FAILED - NOT PAPER EVIDENCE</b><br>"
            "蓝线是 train-only 线性读出器从 LeWM latent 解码的像素轨迹；该读出器没有通过冻结 Gate。"
            "前 10 个 raw step 用于三帧历史，之后每 5 个真实动作产生一个 LeWM 节点，节点间仅作显示插值。"
            "H1-H8 属于训练时域，H9+ 是诚实展示的 open-loop extrapolation。最后不足 5 步继续播放真实画面，"
            "蓝线保持最后一个完整预测节点。"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full episode GT versus LeWM</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef1f3;color:#17212a;font-family:Inter,"Noto Sans SC",system-ui,sans-serif;letter-spacing:0}}header{{background:#18252e;color:white;padding:25px max(20px,calc((100vw - 980px)/2))}}h1,h2,p{{margin:0}}header p{{margin-top:8px;color:#bdc8cf;line-height:1.5}}main{{max-width:980px;margin:auto;padding:22px 20px 60px}}.warning{{background:#fff0ef;border-left:4px solid #c43d32;padding:14px 16px;line-height:1.55}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:#ccd3d8;border:1px solid #ccd3d8;margin:18px 0}}.metrics div{{background:white;padding:12px}}.metrics span,.numbers span{{display:block;color:#66737d;font-size:12px;margin-top:3px}}section{{padding:24px 0 32px;border-bottom:1px solid #c9d0d5}}.heading{{display:flex;justify-content:space-between;gap:18px;margin-bottom:12px}}.heading p{{color:#65717b;font-size:13px;margin-top:5px}}.numbers{{text-align:right;white-space:nowrap}}img{{display:block;width:min(100%,672px);height:auto;border:1px solid #c4ccd2;background:#808080}}table{{width:100%;border-collapse:collapse;background:white;margin-top:18px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dde2e5}}th{{font-size:12px;color:#65717b}}@media(max-width:650px){{.metrics{{grid-template-columns:1fr 1fr}}.heading{{display:block}}.numbers{{text-align:left;margin-top:8px}}}}
</style></head><body><header><h1>完整 Episode：Ground Truth 与 LeWM 预测</h1><p>每个 raw simulator frame 都连续显示。绿色实线是真实末端中心轨迹；蓝色虚线是 LeWM 在已录制 Expert 动作序列条件下的递归预测。</p></header><main>
<div class="warning">{warning}</div>
<div class="metrics"><div><b>{checkpoint_step:,}</b><span>rendered checkpoint step</span></div>{metric_cards}</div>
{''.join(panels)}</main></body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.fps <= 0 or args.max_raw_steps <= 0 or args.post_success_steps < 0:
        raise ValueError("fps/max steps must be positive and post-success steps non-negative")
    if args.expected_checkpoint_step <= 0:
        raise ValueError("expected checkpoint step must be positive")
    contract = _read_json(args.config)
    validate_a4_contract(contract)
    metrics = _read_json(args.metrics)
    development_gate = _read_json(args.development_gate)
    if bool(metrics.get("render_factual_path_authorized", False)):
        raise RuntimeError("expected the frozen direction-readout Gate to remain failed")
    if not args.allow_failed_development_diagnostic:
        raise RuntimeError("pass --allow-failed-development-diagnostic to render this page")
    target_device = torch.device(args.device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {target_device}")
    tasks = tuple(args.task) if args.task else TASKS
    collection = contract["collection"]
    instances = _first_development_instances(contract)
    episodes = []
    for task in tasks:
        task_index = TASKS.index(task)
        instance = instances[task]
        episode = _collect_full_episode(
            task,
            task_instance=instance,
            environment_seed=int(collection["benchmark_seed"]) + 590_000 + task_index * 10_000 + instance,
            width=int(collection["width"]),
            height=int(collection["height"]),
            benchmark_seed=int(collection["benchmark_seed"]),
            max_raw_steps=int(args.max_raw_steps),
            post_success_steps=int(args.post_success_steps),
        )
        episode["model_steps"] = _model_steps(len(episode["actions"]))
        episodes.append(episode)
        print(json.dumps({"task": task, "frames": len(episode["frames"]), "stable_success": episode["stable_success_step"]}), flush=True)

    encoder = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=target_device,
    )
    try:
        for episode in episodes:
            indices = np.asarray(episode["model_steps"], dtype=np.int64)
            sampled = [episode["records"][int(index)] for index in indices]
            episode["encoded_state"] = encoder.encode(sampled)
    finally:
        del encoder
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    checkpoint_paths = tuple(args.a4_checkpoints or DEFAULT_A4_CHECKPOINTS)
    base_checkpoint = _torch_load(args.base_checkpoint)
    ensemble: dict[str, list[np.ndarray]] = {task: [] for task in tasks}
    for checkpoint_path in checkpoint_paths:
        checkpoint = validate_a4_checkpoint(_torch_load(checkpoint_path))
        if int(checkpoint["step"]) != int(args.expected_checkpoint_step):
            raise ValueError(
                "full development visualization checkpoint step mismatch: "
                f"{checkpoint['step']} != {args.expected_checkpoint_step}"
            )
        model, _ = initialize_a4_model_from_anchor_base(
            base_checkpoint, seed=int(checkpoint["seed"])
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(target_device).eval()
        for episode in episodes:
            state = np.asarray(episode["encoded_state"], dtype=np.float32)
            prediction_count = int(state.shape[0] - HISTORY_STEPS)
            ensemble[episode["task"]].append(
                _rollout_one_model(
                    model,
                    history_embedding=state[:HISTORY_STEPS],
                    actions=episode["actions"],
                    task_index=TASKS.index(episode["task"]),
                    prediction_count=prediction_count,
                    device=target_device,
                )
            )
        del model
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    decoder = _load_decoder(args.decoder)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for episode in episodes:
        task = episode["task"]
        prediction = np.mean(np.stack(ensemble[task], axis=0), axis=0)
        state = np.asarray(episode["encoded_state"], dtype=np.float32)
        all_model_steps = np.asarray(episode["model_steps"], dtype=np.int64)
        prediction_steps = all_model_steps[HISTORY_STEPS - 1 :]
        q_start = np.asarray(episode["q_center_display"], dtype=np.float32)[prediction_steps[0]]
        decoded = _decode_open_loop(
            prediction,
            state[:HISTORY_STEPS],
            q_start,
            task=task,
            height=int(collection["height"]),
            decoder=decoder,
        )
        dense = _dense_prediction(
            prediction_steps,
            decoded,
            raw_frame_count=len(episode["frames"]),
        )
        path_metrics = _path_metrics(
            episode["q_center_display"], prediction_steps, decoded
        )
        gif = _save_animation(
            output_dir,
            task=task,
            frames=episode["frames"],
            ground_truth=episode["q_center_display"],
            predicted=dense,
            successes=episode["successes"],
            fps=float(args.fps),
            layout=str(args.layout),
        )
        entries.append(
            {
                "task": task,
                "task_instance": episode["task_instance"],
                "environment_seed": episode["environment_seed"],
                "frame_count": len(episode["frames"]),
                "raw_actions": len(episode["actions"]),
                "first_success_step": episode["first_success_step"],
                "stable_success_step": episode["stable_success_step"],
                "post_success_steps": int(args.post_success_steps),
                "last_model_node_raw_step": int(prediction_steps[-1]),
                "layout": str(args.layout),
                "gif": gif,
                **path_metrics,
            }
        )
    (output_dir / "index.html").write_text(
        _render_html(
            entries,
            metrics,
            development_gate,
            checkpoint_step=int(args.expected_checkpoint_step),
        ),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "a4_full_episode_open_loop_trajectory_diagnostic",
                "reserved_test_opened": False,
                "readout_gate_passed": False,
                "layout": str(args.layout),
                "checkpoint_step": int(args.expected_checkpoint_step),
                "checkpoints": [str(path.expanduser().resolve()) for path in checkpoint_paths],
                "entries": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
