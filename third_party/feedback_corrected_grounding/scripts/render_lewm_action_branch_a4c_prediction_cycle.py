#!/usr/bin/env python3
"""Render the A4c seven-task prediction-cycle visualization.

Each section follows the original camera-action page layout: the left panel
advances through the real episode, while the right panel stays on the latest
observation and plays a fresh 40-raw-step LeWM forecast.  The model contract is
one summary slot plus three persistent anchor slots.  The human-facing probe
therefore decodes all three anchors independently; a legacy center-only
decoder is rejected instead of silently hiding the gripper geometry.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_checkpoint import validate_a4_checkpoint  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    HORIZON,
    HISTORY_STEPS,
    RAW_STEPS_PER_MODEL_STEP,
    TASKS,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    load_frozen_a4_state_encoder,
)
from fcpg.lewm_action_branch_a4_training import (  # noqa: E402
    initialize_a4_model_from_anchor_base,
)
from scripts import render_lewm_action_branch_a4_trajectory_comparison as trajectory  # noqa: E402


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
    / f"outputs/lewm_action_branch_a4c_v4/shared_7task_seed{seed}_2k/"
    "lewm_action_branch_a4_latest.pt"
    for seed in (30201, 30202, 30203)
)
DEFAULT_DECODER = ROOT / (
    "outputs/lewm_action_branch_a4/three_anchor_readout/decoder.npz"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_branch_a4c_v4/shared_7task_seed30201_2k_prediction_cycle_full"
)

ANCHOR_COLORS = ((236, 78, 68), (55, 176, 112), (57, 126, 214))
GROUND_TRUTH_COLOR = (31, 151, 91)
PREDICTION_COLOR = (35, 105, 191)
ANCHOR_COUNT = 3
ANCHOR_SLOT_START = 1
FEATURE_DIM = ANCHOR_COUNT * 192
OUTPUT_DIM = ANCHOR_COUNT * 2
SUCCESS_STREAK = 3
POST_SUCCESS_RAW_STEPS = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument(
        "--a4-checkpoint", type=Path, action="append", dest="a4_checkpoints"
    )
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-raw-steps", type=int, default=500)
    parser.add_argument("--post-success-steps", type=int, default=POST_SUCCESS_RAW_STEPS)
    parser.add_argument("--task", choices=TASKS, action="append")
    return parser


def _cuda0(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("A4c prediction-cycle rendering permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(name)
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_base(path: Path) -> dict[str, Any]:
    value = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError("A4 base checkpoint must contain a mapping")
    return value


def _load_decoder(path: Path) -> dict[str, np.ndarray]:
    """Load only the train-only six-output (three-anchor) readout."""

    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        required = {"coefficient_raw", "intercept_raw", "fit_split", "task_names"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"three-anchor decoder is missing {missing}")
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if str(np.asarray(values["fit_split"]).item()) != "train":
        raise PermissionError("prediction-cycle renderer accepts only a train-only decoder")
    coefficient = np.asarray(values["coefficient_raw"])
    intercept = np.asarray(values["intercept_raw"])
    expected_coefficient = (len(TASKS), FEATURE_DIM, OUTPUT_DIM)
    expected_intercept = (len(TASKS), OUTPUT_DIM)
    if coefficient.shape != expected_coefficient:
        raise ValueError(
            "A4c renderer requires task-conditioned three-anchor coefficients "
            f"{expected_coefficient}; got {coefficient.shape}. The legacy center "
            "decoder is intentionally rejected."
        )
    if intercept.shape != expected_intercept:
        raise ValueError(
            f"three-anchor decoder intercept must have shape {expected_intercept}, "
            f"got {intercept.shape}"
        )
    task_names = tuple(np.asarray(values["task_names"]).astype(str).tolist())
    if task_names != TASKS:
        raise ValueError("three-anchor decoder task_names differ from the A4 contract")
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
        raise ValueError("three-anchor decoder parameters must be finite")
    if int(np.asarray(values.get("anchor_count", ANCHOR_COUNT)).item()) != ANCHOR_COUNT:
        raise ValueError("three-anchor decoder must declare anchor_count=3")
    return values


def _decoder_parameters(
    decoder: Mapping[str, np.ndarray], task: str
) -> tuple[np.ndarray, np.ndarray]:
    task_names = tuple(np.asarray(decoder["task_names"]).astype(str).tolist())
    try:
        index = task_names.index(task)
    except ValueError as error:
        raise ValueError(f"three-anchor decoder does not contain task {task!r}") from error
    return (
        np.asarray(decoder["coefficient_raw"])[index].astype(np.float64, copy=False),
        np.asarray(decoder["intercept_raw"])[index].astype(np.float64, copy=False),
    )


def _prediction_starts(raw_action_count: int) -> np.ndarray:
    """Return starts with three history states and a complete H8 future."""

    last = int(raw_action_count) - HORIZON * RAW_STEPS_PER_MODEL_STEP
    if last < (HISTORY_STEPS - 1) * RAW_STEPS_PER_MODEL_STEP:
        raise ValueError("episode is too short for A4c history and H8 forecast")
    return np.arange(
        (HISTORY_STEPS - 1) * RAW_STEPS_PER_MODEL_STEP,
        last + 1,
        RAW_STEPS_PER_MODEL_STEP,
        dtype=np.int64,
    )


def _chunk_actions(actions: np.ndarray, starts: Sequence[int]) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    chunks = [values[int(start) : int(start) + RAW_STEPS_PER_MODEL_STEP].reshape(-1) for start in starts]
    result = np.asarray(chunks, dtype=np.float32)
    expected = (len(chunks), RAW_STEPS_PER_MODEL_STEP * 4)
    if result.shape != expected:
        raise RuntimeError(f"action chunk shape drifted: {result.shape} != {expected}")
    return result


def _prepare_window_inputs(
    state: np.ndarray, actions: np.ndarray, starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_indices = np.stack(
        [starts - 2 * RAW_STEPS_PER_MODEL_STEP, starts - RAW_STEPS_PER_MODEL_STEP, starts],
        axis=1,
    ) // RAW_STEPS_PER_MODEL_STEP
    history = np.asarray(state[history_indices], dtype=np.float32)
    history_action = np.stack(
        [
            _chunk_actions(actions, starts - 2 * RAW_STEPS_PER_MODEL_STEP),
            _chunk_actions(actions, starts - RAW_STEPS_PER_MODEL_STEP),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    future_action = np.stack(
        [
            _chunk_actions(
                actions,
                starts + horizon * RAW_STEPS_PER_MODEL_STEP,
            )
            for horizon in range(HORIZON)
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    expected_history = (len(starts), HISTORY_STEPS, 4, 192)
    expected_future = (len(starts), HORIZON, 20)
    if history.shape != expected_history or history_action.shape != (len(starts), 2, 20):
        raise RuntimeError("A4c history window shape drifted")
    if future_action.shape != expected_future:
        raise RuntimeError("A4c future action window shape drifted")
    return history, history_action, future_action


@torch.inference_mode()
def _predict_windows(
    model: Any,
    *,
    state: np.ndarray,
    actions: np.ndarray,
    starts: np.ndarray,
    task_index: int,
    device: torch.device,
) -> np.ndarray:
    history, history_action, future_action = _prepare_window_inputs(state, actions, starts)
    output = model.rollout_state_embeddings(
        torch.from_numpy(history).to(device),
        torch.from_numpy(history_action).to(device),
        torch.from_numpy(future_action[:, None]).to(device),
        torch.full((len(starts),), int(task_index), dtype=torch.long, device=device),
    )
    result = output[:, 0].float().cpu().numpy()
    expected = (len(starts), HORIZON, 4, 192)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"A4c prediction shape/value drifted: {result.shape}")
    return result


def _decode_anchor_windows(
    predictions: np.ndarray,
    history: np.ndarray,
    starts: np.ndarray,
    episode: Mapping[str, Any],
    *,
    task: str,
    decoder: Mapping[str, np.ndarray],
) -> np.ndarray:
    future = np.asarray(predictions, dtype=np.float64)
    past = np.asarray(history, dtype=np.float64)
    if future.ndim != 4 or future.shape[1:] != (HORIZON, 4, 192):
        raise ValueError("A4c predictions must have shape [N,8,4,192]")
    if past.shape != (len(starts), HISTORY_STEPS, 4, 192):
        raise ValueError("A4c history must have shape [N,3,4,192]")
    anchor_features = (
        future[:, :, ANCHOR_SLOT_START:, :] - past[:, None, -1, ANCHOR_SLOT_START:, :]
    )
    feature = anchor_features.reshape(len(starts), HORIZON, FEATURE_DIM)
    coefficient, intercept = _decoder_parameters(decoder, task)
    displacement = np.einsum("nhf,fo->nho", feature, coefficient) + intercept[None, None]
    displacement = displacement.reshape(len(starts), HORIZON, ANCHOR_COUNT, 2)
    # The readout is fitted in runtime projection coordinates.  The displayed
    # frames for standard MetaWorld tasks are vertically flipped.
    if task not in trajectory.CUSTOM_TASKS:
        displacement[:, :, :, 1] *= -1.0
    q_anchor = np.asarray(episode["q_anchor_display"], dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    if q_anchor.ndim != 3 or q_anchor.shape[1:] != (ANCHOR_COUNT, 2):
        raise ValueError("episode q_anchor_display must have shape [T,3,2]")
    result = q_anchor[starts, None] + displacement
    expected = (len(starts), HORIZON, ANCHOR_COUNT, 2)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"decoded three-anchor path shape drifted: {result.shape}")
    return result.astype(np.float32)


def _interpolate_anchors(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    points = np.concatenate([np.asarray(current, dtype=np.float32)[None], future], axis=0)
    model_steps = np.arange(0, HORIZON * RAW_STEPS_PER_MODEL_STEP + 1, RAW_STEPS_PER_MODEL_STEP)
    raw_steps = np.arange(HORIZON * RAW_STEPS_PER_MODEL_STEP + 1)
    if points.shape != (HORIZON + 1, ANCHOR_COUNT, 2):
        raise ValueError("anchor interpolation requires [9,3,2] model nodes")
    dense = np.empty((len(raw_steps), ANCHOR_COUNT, 2), dtype=np.float32)
    for anchor in range(ANCHOR_COUNT):
        for axis in range(2):
            dense[:, anchor, axis] = np.interp(
                raw_steps, model_steps, points[:, anchor, axis]
            ).astype(np.float32)
    return dense


def _line(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int], width: int = 4) -> None:
    values = np.asarray(points, dtype=np.float64)
    if len(values) >= 2:
        draw.line([tuple(row) for row in values], fill=color, width=width)


def _dashed_line(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int], width: int = 4) -> None:
    values = np.asarray(points, dtype=np.float64)
    for start, end in zip(values[:-1], values[1:], strict=True):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            continue
        direction = delta / length
        cursor = 0.0
        while cursor < length:
            end_cursor = min(cursor + 8.0, length)
            draw.line(
                (tuple(start + direction * cursor), tuple(start + direction * end_cursor)),
                fill=color,
                width=width,
            )
            cursor += 13.0


def _marker(draw: ImageDraw.ImageDraw, point: np.ndarray, color: tuple[int, int, int], diamond: bool) -> None:
    x, y = np.asarray(point, dtype=np.float64)
    radius = 6.0
    if diamond:
        polygon = ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y))
        draw.polygon(polygon, fill=color, outline=(255, 255, 255))
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=2)


def _cycle_frame(
    left_frame: Image.Image,
    right_frame: Image.Image,
    truth_anchors: np.ndarray,
    predicted_anchors: np.ndarray | None,
    *,
    actual_step: int,
    prediction_origin: int | None,
    prediction_target: int | None,
    end_step: int,
    task_label: str,
    success: bool,
    phase: str,
) -> Image.Image:
    left = left_frame.convert("RGB").copy()
    right = right_frame.convert("L").convert("RGB")
    right = Image.blend(right, Image.new("RGB", right.size, (232, 235, 237)), 0.28)
    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    truth = np.asarray(truth_anchors, dtype=np.float32)
    if truth.shape != (len(truth), ANCHOR_COUNT, 2):
        raise ValueError("truth anchors must have shape [T,3,2]")
    for anchor, color in enumerate(ANCHOR_COLORS):
        _line(left_draw, truth[: actual_step + 1, anchor], color, width=3)
        _marker(left_draw, truth[actual_step, anchor], color, diamond=False)
        left_draw.text(
            (float(truth[actual_step, anchor, 0]) + 7, float(truth[actual_step, anchor, 1]) - 7),
            f"A{anchor}",
            fill=color,
            font=_font(10),
        )
    if predicted_anchors is not None and len(predicted_anchors):
        predicted = np.asarray(predicted_anchors, dtype=np.float32)
        if predicted.ndim != 3 or predicted.shape[1:] != (ANCHOR_COUNT, 2):
            raise ValueError("predicted anchors must have shape [T,3,2]")
        for anchor, color in enumerate(ANCHOR_COLORS):
            _dashed_line(right_draw, predicted[:, anchor], color, width=3)
            _marker(right_draw, predicted[-1, anchor], color, diamond=True)
            right_draw.text(
                (float(predicted[-1, anchor, 0]) + 7, float(predicted[-1, anchor, 1]) - 7),
                f"A{anchor}",
                fill=color,
                font=_font(10),
            )

    width, height = left.size
    margin, gap, header, footer = 12, 12, 66, 38
    canvas = Image.new("RGB", (margin * 2 + width * 2 + gap, header + height + footer), (238, 241, 243))
    right_x = margin + width + gap
    canvas.paste(left, (margin, header))
    canvas.paste(right, (right_x, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 7), task_label, fill=(23, 32, 40), font=_font(17))
    draw.text((margin, 35), f"ACTUAL t={actual_step}", fill=GROUND_TRUTH_COLOR, font=_font(12))
    if prediction_origin is None:
        pred_label = "PREDICTOR WARMUP"
    else:
        pred_label = f"PRED t={prediction_origin}->{prediction_target}"
    draw.text((right_x, 35), pred_label, fill=PREDICTION_COLOR, font=_font(12))
    if prediction_origin is None:
        forecast_text = "waiting for 3-frame history"
    else:
        forecast_text = f"forecast +{int(prediction_target - prediction_origin):02d}/40"
    draw.text(
        (margin, header + height + 10),
        f"{phase} | actual {actual_step:03d}/{end_step:03d} | {forecast_text} | success {int(success)}",
        fill=(55, 66, 75),
        font=_font(11),
    )
    return canvas


def _cycle_frames(
    episode: Mapping[str, Any],
    starts: np.ndarray,
    predicted_windows: np.ndarray,
    *,
    task_label: str,
) -> Iterable[Image.Image]:
    frames = episode["frames"]
    successes = episode["successes"]
    truth = np.asarray(episode["q_anchor_display"], dtype=np.float32)
    end_step = len(frames) - 1
    starts = np.asarray(starts, dtype=np.int64)
    for actual_step in range(int(starts[0]) + 1):
        yield _cycle_frame(
            frames[actual_step],
            frames[0],
            truth,
            None,
            actual_step=actual_step,
            prediction_origin=None,
            prediction_target=None,
            end_step=end_step,
            task_label=task_label,
            success=bool(successes[actual_step]),
            phase="ACTUAL WARMUP",
        )
    previous_step = None
    previous_dense = None
    for index, start in enumerate(starts):
        start = int(start)
        if previous_step is not None and previous_dense is not None:
            for actual_step in range(previous_step + 1, start + 1):
                yield _cycle_frame(
                    frames[actual_step],
                    frames[previous_step],
                    truth,
                    previous_dense,
                    actual_step=actual_step,
                    prediction_origin=previous_step,
                    prediction_target=previous_step + 40,
                    end_step=end_step,
                    task_label=task_label,
                    success=bool(successes[actual_step]),
                    phase="ACTUAL ADVANCE",
                )
        dense = _interpolate_anchors(truth[start], predicted_windows[index])
        for offset in range(1, len(dense)):
            yield _cycle_frame(
                frames[start],
                frames[start],
                truth,
                dense[: offset + 1],
                actual_step=start,
                prediction_origin=start,
                prediction_target=start + offset,
                end_step=end_step,
                task_label=task_label,
                success=bool(successes[start]),
                phase="FORECAST PLAYBACK",
            )
        previous_step = start
        previous_dense = dense
    if previous_step is not None and previous_dense is not None:
        for actual_step in range(previous_step + 1, end_step + 1):
            yield _cycle_frame(
                frames[actual_step],
                frames[previous_step],
                truth,
                previous_dense,
                actual_step=actual_step,
                prediction_origin=previous_step,
                prediction_target=previous_step + 40,
                end_step=end_step,
                task_label=task_label,
                success=bool(successes[actual_step]),
                phase="FINAL ACTUAL CATCH-UP",
            )


def _save_media(
    output_dir: Path,
    task: str,
    frames: Iterable[Image.Image],
    *,
    fps: float,
    gif_max_frames: int = 200,
) -> tuple[str, str, int, int]:
    stem = f"{task.replace('-', '_')}_a4c_prediction_cycle"
    video_name = f"{stem}.mp4"
    gif_name = f"{stem}.gif"
    video = imageio.get_writer(
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
    # Keep the GIF light while leaving the full-resolution MP4 available.
    frame_count = 0
    gif_buffer: list[np.ndarray] = []
    try:
        for frame in frames:
            pixels = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            video.append_data(pixels)
            if len(gif_buffer) < int(gif_max_frames):
                gif_buffer.append(pixels.copy())
            frame_count += 1
    finally:
        video.close()
    if frame_count <= 0:
        raise RuntimeError("prediction-cycle produced no frames")
    # Uniformly sample the full video after writing it, rather than keeping a
    # biased prefix when a task has a long trajectory.
    if frame_count > gif_max_frames:
        # Re-read the compact MP4 only for GIF generation; this avoids retaining
        # thousands of full-size RGB frames in memory.
        reader = imageio.get_reader(output_dir / video_name, format="FFMPEG")
        indices = np.unique(np.rint(np.linspace(0, frame_count - 1, gif_max_frames)).astype(int))
        gif_buffer = [reader.get_data(int(index)) for index in indices]
        reader.close()
    duration = float(frame_count) / (float(fps) * float(len(gif_buffer)))
    gif = imageio.get_writer(output_dir / gif_name, format="GIF", mode="I", duration=duration, loop=0)
    try:
        for pixels in gif_buffer:
            gif.append_data(np.asarray(pixels, dtype=np.uint8))
    finally:
        gif.close()
    return video_name, gif_name, frame_count, len(gif_buffer)


def _horizon_metrics(
    episode: Mapping[str, Any], starts: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray(episode["q_anchor_display"], dtype=np.float32)
    rows: dict[str, Any] = {}
    for horizon in range(1, HORIZON + 1):
        target = truth[starts + horizon * RAW_STEPS_PER_MODEL_STEP]
        estimate = predictions[:, horizon - 1]
        per_anchor_error = np.linalg.norm(estimate - target, axis=-1)
        error = per_anchor_error.mean(axis=-1)
        rows[f"H{horizon}"] = {
            "sample_count": int(len(error)),
            "anchor_mean_epe_px": float(error.mean()),
            "anchor_p95_epe_px": float(np.quantile(error, 0.95)),
            "anchor_mean_epe_by_slot_px": per_anchor_error.mean(axis=0).tolist(),
            "anchor_p95_epe_by_slot_px": np.quantile(per_anchor_error, 0.95, axis=0).tolist(),
            "pck_at_4": float(np.mean(error <= 4.0)),
            "pck_at_8": float(np.mean(error <= 8.0)),
        }
    return rows


def _render_html(entries: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> str:
    sections = []
    for entry in entries:
        rows = []
        for horizon in range(1, HORIZON + 1):
            metric = entry["horizon_metrics"][f"H{horizon}"]
            rows.append(
                "<tr>"
                f"<td>H{horizon}</td><td>{metric['anchor_mean_epe_px']:.2f}</td>"
                f"<td>{metric['anchor_p95_epe_px']:.2f}</td>"
                f"<td>{100.0 * metric['pck_at_4']:.1f}%</td>"
                f"<td>{100.0 * metric['pck_at_8']:.1f}%</td></tr>"
            )
        sections.append(
            f"""
<section><div class="head"><div><h2>{html.escape(str(entry['task_label']))}</h2>
<p>development instance {entry['task_instance']} &middot; camera {html.escape(str(entry['camera_label']))} &middot; raw frames {entry['raw_frame_count']} &middot; rendered frames {entry['rendered_frame_count']} &middot; stable success raw {entry['stable_success_step']}</p></div>
<div class="summary">H1-H8 mean over 3 anchors<br><b>{entry['h1_h8_anchor_mean_epe_px']:.2f} px</b></div></div>
<p class="view-title"><b>Repeated 40-step prediction:</b> the left side advances five real frames and pauses. The right side then plays a new 40-frame forecast from the current three-frame history before the left advances again. Red/green/blue are fixed anchor identities A0/A1/A2; solid trails are measured and dashed trails are predicted.</p>
<video controls muted loop playsinline preload="metadata"><source src="{html.escape(str(entry['video']), quote=True)}" type="video/mp4"></video>
<details><summary>Open the sampled {entry['gif_frame_count']}-frame GIF</summary><img class="preview-gif" loading="lazy" src="{html.escape(str(entry['gif']), quote=True)}" alt="sampled prediction cycle"></details>
<table><thead><tr><th>Horizon</th><th>3-anchor mean EPE</th><th>3-anchor P95</th><th>PCK@4</th><th>PCK@8</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"""
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A4c seven-task LeWM prediction cycle</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf1f3;color:#17212a;font-family:Inter,Arial,sans-serif;letter-spacing:0}}header{{background:#17252e;color:white;padding:24px max(20px,calc((100vw - 880px)/2))}}h1,h2,p{{margin:0}}header p{{margin-top:8px;color:#c4cfd5;line-height:1.5}}main{{max-width:880px;margin:auto;padding:20px 20px 60px}}.contract{{background:white;border-left:5px solid #2976a8;padding:15px 17px;line-height:1.55}}.warning{{background:#fff5e8;border-left:5px solid #c47b21;padding:14px 17px;line-height:1.55;margin-top:14px}}section{{padding:28px 0 34px;border-bottom:1px solid #c8d0d5}}.head{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:12px}}.head p{{margin-top:5px;color:#66737d;font-size:13px}}.summary{{text-align:right;color:#52616b;font-size:12px}}.summary b{{font-size:18px;color:#23333e}}.view-title{{margin:15px 0 8px;color:#52616b;line-height:1.45}}video,.preview-gif{{display:block;width:100%;height:auto;border:1px solid #bbc5cb;background:#20272c}}details{{margin-top:10px}}summary{{cursor:pointer;color:#52616b}}.preview-gif{{margin-top:8px}}table{{width:100%;border-collapse:collapse;background:white;margin-top:14px}}th,td{{padding:9px 10px;text-align:right;border-bottom:1px solid #dde3e6;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}th{{color:#60707a;font-size:12px}}@media(max-width:700px){{.head{{display:block}}.summary{{text-align:left;margin-top:8px}}table{{font-size:12px}}th,td{{padding:7px 5px}}}}
</style></head><body><header><h1>A4c Shared Seven-Task LeWM: Prediction Cycle</h1><p>One shared model per seed, seeds {html.escape(str(metadata['seeds']))}, fixed step 2,000, SIGReg 0.02. Camera {html.escape(str(metadata['camera']))}.</p></header><main>
<div class="contract"><b>Playback:</b> the left panel advances through the real episode. Every five real raw steps it pauses; the right panel plays a fresh 40-step prediction from the current three-frame history, then the real episode advances again. The right panel never receives future RGB or q_true.</div>
<div class="warning"><b>Development visualization only.</b> This page uses the 2K development checkpoints and development episodes. The three colored dashed traces are a train-only latent-to-pixel readout of the persistent A0/A1/A2 slots, not generated RGB and not sealed-test evidence. The decoder is required to emit six values (three points), so the legacy center-only probe cannot be used here.</div>
{''.join(sections)}</main></body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.fps <= 0 or args.max_raw_steps <= 0 or args.post_success_steps < 0:
        raise ValueError("fps/max_raw_steps/post_success_steps must be valid")
    device = _cuda0(args.device)
    config = _load_json(args.config)
    validate_a4_contract(config)
    tasks = tuple(args.task) if args.task else TASKS
    if len(set(tasks)) != len(tasks):
        raise ValueError("--task values must be unique")
    collection = config["collection"]
    instances = trajectory._first_development_instances(config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = tuple(args.a4_checkpoints or DEFAULT_A4_CHECKPOINTS)
    if len(checkpoint_paths) != 3:
        raise ValueError("A4c cycle visualization requires exactly three checkpoints")

    episodes: list[dict[str, Any]] = []
    for task in tasks:
        task_index = TASKS.index(task)
        instance = instances[task]
        episode = trajectory._collect_full_episode(
            task,
            task_instance=instance,
            environment_seed=int(collection["benchmark_seed"]) + 590_000 + task_index * 10_000 + instance,
            width=int(collection["width"]),
            height=int(collection["height"]),
            benchmark_seed=int(collection["benchmark_seed"]),
            max_raw_steps=int(args.max_raw_steps),
            post_success_steps=int(args.post_success_steps),
        )
        anchor_rows = []
        for record, frame in zip(episode["records"], episode["frames"], strict=True):
            points = np.asarray(record.q_true, dtype=np.float32).copy()
            if points.shape != (ANCHOR_COUNT, 2) or not np.isfinite(points).all():
                raise RuntimeError("episode record must contain three finite q_true anchors")
            if task not in trajectory.CUSTOM_TASKS:
                points[:, 1] = float(frame.height - 1) - points[:, 1]
            anchor_rows.append(points)
        episode["q_anchor_display"] = np.asarray(anchor_rows, dtype=np.float32)
        if episode["q_anchor_display"].shape != (len(episode["frames"]), ANCHOR_COUNT, 2):
            raise RuntimeError("episode anchor trajectory shape drifted")
        episode["model_steps"] = trajectory._model_steps(len(episode["actions"]))
        episode["prediction_starts"] = _prediction_starts(len(episode["actions"]))
        episodes.append(episode)
        print(json.dumps({"task": task, "raw_frames": len(episode["frames"]), "starts": len(episode["prediction_starts"])}), flush=True)

    base_checkpoint = _load_base(args.base_checkpoint)
    encoder = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=device,
    )
    try:
        for episode in episodes:
            indices = np.asarray(episode["model_steps"], dtype=np.int64)
            episode["state"] = np.asarray(encoder.encode([episode["records"][int(i)] for i in indices]), dtype=np.float32)
    finally:
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

    predictions_by_task: dict[str, list[np.ndarray]] = {task: [] for task in tasks}
    decoder = _load_decoder(args.decoder)
    for checkpoint_path in checkpoint_paths:
        payload = validate_a4_checkpoint(torch.load(checkpoint_path.expanduser().resolve(), map_location="cpu", weights_only=False))
        if int(payload["step"]) != 2_000 or float(payload["sigreg_weight"]) != 0.02:
            raise ValueError("A4c cycle renderer requires step=2000 and SIGReg=0.02")
        model, _ = initialize_a4_model_from_anchor_base(base_checkpoint, seed=int(payload["seed"]))
        model.load_state_dict(payload["model_state"], strict=True)
        model.to(device).eval()
        try:
            for episode in episodes:
                task = str(episode["task"])
                starts = np.asarray(episode["prediction_starts"], dtype=np.int64)
                history, _, _ = _prepare_window_inputs(episode["state"], episode["actions"], starts)
                predicted = _predict_windows(
                    model,
                    state=episode["state"],
                    actions=episode["actions"],
                    starts=starts,
                    task_index=TASKS.index(task),
                    device=device,
                )
                decoded = _decode_anchor_windows(
                    predicted,
                    history,
                    starts,
                    episode,
                    task=task,
                    decoder=decoder,
                )
                predictions_by_task[task].append(decoded)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    entries = []
    manifest_entries = []
    for episode in episodes:
        task = str(episode["task"])
        starts = np.asarray(episode["prediction_starts"], dtype=np.int64)
        history, _, _ = _prepare_window_inputs(episode["state"], episode["actions"], starts)
        # Decode once more only to keep the per-task loop self-contained and to
        # make the decoder contract explicit in the output manifest.
        _ = decoder
        predicted = np.mean(np.stack(predictions_by_task[task], axis=0), axis=0)
        frames = _cycle_frames(
            episode,
            starts,
            predicted,
            task_label=trajectory.TASK_LABELS[task],
        )
        video, gif, frame_count, gif_count = _save_media(
            output_dir,
            task,
            frames,
            fps=float(args.fps),
        )
        metrics = _horizon_metrics(episode, starts, predicted)
        h1_h8 = float(np.mean([metrics[f"H{h}"]["anchor_mean_epe_px"] for h in range(1, HORIZON + 1)]))
        entry = {
            "task": task,
            "task_label": trajectory.TASK_LABELS[task],
            "task_instance": int(episode["task_instance"]),
            "environment_seed": int(episode["environment_seed"]),
            "camera_label": str(episode["camera_name"]),
            "raw_frame_count": len(episode["frames"]),
            "rendered_frame_count": frame_count,
            "stable_success_step": int(episode["stable_success_step"]),
            "video": video,
            "gif": gif,
            "gif_frame_count": gif_count,
            "horizon_metrics": metrics,
            "h1_h8_anchor_mean_epe_px": h1_h8,
            "anchor_count": ANCHOR_COUNT,
            "prediction_playback": "advance_actual_5_then_forecast_40",
        }
        entries.append(entry)
        manifest_entries.append(entry)
        print(json.dumps({"task": task, "video": video, "frames": frame_count}), flush=True)

    metadata = {
        "seeds": [int(torch.load(path, map_location="cpu", weights_only=False)["seed"]) for path in checkpoint_paths],
        "step": 2_000,
        "sigreg_weight": 0.02,
        "camera": "task_default",
        "tasks": list(tasks),
        "decoder": str(args.decoder.expanduser().resolve()),
        "anchor_count": ANCHOR_COUNT,
    }
    (output_dir / "index.html").write_text(_render_html(entries, metadata), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({"protocol": "a4c_prediction_cycle_visualization_v2_three_anchor", "metadata": metadata, "tasks": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({"protocol": "a4c_prediction_cycle_visualization_v2_three_anchor", "reserved_test_opened": False, "anchor_count": ANCHOR_COUNT, "state_slots": ["visual_summary", "anchor0", "anchor1", "anchor2"], "model_input": ["frozen_A4_state_history", "two_history_action_chunks", "eight_future_action_chunks", "task_identity"], "future_rgb": "not_used_for_prediction", "future_q_true": "not_used_for_prediction", "entries": manifest_entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
