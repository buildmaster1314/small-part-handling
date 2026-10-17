#!/usr/bin/env python3
"""Render a synchronized camera-panel forecast from the joint A4 LeWM.

The top row shows evaluation-only future RGB with true anchors. The bottom row
keeps the current observation frozen and overlays the model forecast. Every
view shares one physical replay bundle and one action branch.
"""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_task_factory import CUSTOM_TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4 import HORIZON, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    JOINT_PROTOCOL,
    joint_batch_from_memmaps,
    load_joint_checkpoint,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_camera_action_adapter_a4 import (  # noqa: E402
    ADAPTER_PROTOCOL,
    load_camera_action_adapter_checkpoint,
)
from fcpg.lewm_camera_action_joint_a4 import (  # noqa: E402
    CAMERA_ACTION_PROTOCOL,
    EXPANDED_CAMERA_ACTION_PROTOCOL,
    load_camera_action_joint_checkpoint,
    load_expanded_camera_action_joint_checkpoint,
)


DEFAULT_DATASET = ROOT / (
    "outputs/lewm_action_prefix_a4/joint_multiview5_gripper_8train_2dev"
)
DEFAULT_CHECKPOINT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "shared_7task_multiview5_gripper_seed40521_2k/"
    "lewm_action_prefix_joint_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "shared_7task_multiview5_gripper_seed40521_2k_visual"
)
CAMERAS = ("corner", "corner2", "corner3", "corner4", "topview")
TRUE_COLOR = (24, 166, 95)
PREDICTED_COLOR = (24, 103, 190)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", choices=TASKS, default="pick-place-v3")
    parser.add_argument(
        "--camera-split",
        choices=("train", "development"),
        default="development",
    )
    parser.add_argument(
        "--branch",
        type=int,
        default=7,
        help="Action branch; branch 7 closes first and then opens.",
    )
    parser.add_argument("--fps", type=float, default=1.5)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("multi-view gallery permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _load_checkpoint(path: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    protocol = payload.get("protocol")
    if protocol == JOINT_PROTOCOL:
        model = load_joint_checkpoint(payload)
    elif protocol == CAMERA_ACTION_PROTOCOL:
        model = load_camera_action_joint_checkpoint(payload)
    elif protocol == EXPANDED_CAMERA_ACTION_PROTOCOL:
        model = load_expanded_camera_action_joint_checkpoint(payload)
    elif protocol == ADAPTER_PROTOCOL:
        model = load_camera_action_adapter_checkpoint(payload)
    else:
        raise ValueError("checkpoint is not a supported joint seven-task model")
    return model.to(device).eval(), dict(payload)


def _aperture(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.shape[-2:] != (3, 2):
        raise ValueError("anchor points must end in [3,2]")
    return np.linalg.norm(values[..., 0, :] - values[..., 1, :], axis=-1)


def _select_physical_bundle(
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    task: str,
    branch: int,
    cameras: Sequence[str] = CAMERAS,
    split: str = "development",
) -> tuple[int, np.ndarray]:
    if not 0 <= int(branch) < 8:
        raise ValueError("branch must be in [0,7]")
    task_name = np.asarray(index["task_name"]).astype(str)
    split_values = np.asarray(index["split"]).astype(str)
    physical = np.asarray(index["physical_bundle_index"], dtype=np.int64)
    panel = tuple(str(value) for value in cameras)
    if not panel or len(panel) != len(set(panel)):
        raise ValueError("camera panel must be non-empty and unique")
    candidates = np.unique(
        physical[(task_name == task) & (split_values == str(split))]
    )
    if candidates.size == 0:
        raise ValueError(f"no {split} physical bundle for {task}")
    best: tuple[float, int, np.ndarray] | None = None
    for physical_index in candidates:
        rows = np.flatnonzero(physical == int(physical_index))
        row_cameras = np.asarray(index["camera_name"])[rows].astype(str)
        if set(row_cameras.tolist()) != set(panel):
            continue
        reference = int(rows[0])
        truth = np.asarray(arrays["future_q_true"])[reference, int(branch)]
        change = float(np.ptp(_aperture(truth)))
        ranked = (change, int(physical_index), rows)
        if best is None or ranked[0] > best[0]:
            best = ranked
    if best is None:
        raise ValueError("no complete multi-camera physical bundle")
    return best[1], best[2]


def _upright(
    task: str,
    rgb: np.ndarray,
    points: np.ndarray,
    *,
    flip_vertical: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(rgb, dtype=np.uint8)
    anchors = np.asarray(points, dtype=np.float32).copy()
    if image.shape != (3, 224, 224):
        raise ValueError("RGB must have shape [3,224,224]")
    if anchors.shape != (3, 2):
        raise ValueError("points must have shape [3,2]")
    image = np.moveaxis(image, 0, -1)
    should_flip = task not in CUSTOM_TASKS if flip_vertical is None else flip_vertical
    if should_flip:
        image = np.flip(image, axis=0).copy()
        anchors[:, 1] = float(image.shape[0] - 1) - anchors[:, 1]
    return image, anchors


def _camera_panel(
    index: Mapping[str, np.ndarray],
    *,
    task: str,
    split: str,
) -> tuple[str, ...]:
    task_name = np.asarray(index["task_name"]).astype(str)
    split_name = np.asarray(index["split"]).astype(str)
    rows = np.flatnonzero((task_name == task) & (split_name == split))
    if rows.size == 0:
        raise ValueError(f"no {split} rows for {task}")
    names = np.asarray(index["camera_name"])[rows].astype(str)
    if "camera_clock_hour" not in index:
        values = set(names.tolist())
        if values == set(CAMERAS):
            return CAMERAS
        return tuple(sorted(values))
    hours = np.asarray(index["camera_clock_hour"], dtype=np.int64)[rows]
    by_name: dict[str, int] = {}
    for name, hour in zip(names.tolist(), hours.tolist(), strict=True):
        previous = by_name.setdefault(str(name), int(hour))
        if previous != int(hour):
            raise ValueError("camera label maps to multiple clock hours")
    return tuple(sorted(by_name, key=lambda name: by_name[name]))


def _draw_anchor_panel(
    rgb: np.ndarray,
    points: np.ndarray,
    *,
    color: tuple[int, int, int],
    title: str,
    aperture_px: float,
    reference_points: np.ndarray | None = None,
) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    if reference_points is not None:
        reference = np.asarray(reference_points, dtype=np.float32)
        if reference.shape != (3, 2):
            raise ValueError("reference points must have shape [3,2]")
        for predicted, target in zip(points, reference):
            draw.line([tuple(predicted), tuple(target)], fill=(215, 63, 57), width=2)
            x, y = (float(target[0]), float(target[1]))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=TRUE_COLOR, width=2)
    if len(points) >= 2:
        draw.line([tuple(points[0]), tuple(points[1])], fill=color, width=3)
    for anchor_index, point in enumerate(points):
        x, y = (float(point[0]), float(point[1]))
        radius = 5 if anchor_index < 2 else 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.rectangle((0, 0, image.width, 36), fill=(15, 23, 29))
    draw.text((8, 4), title, fill=(245, 247, 248), font=_font(12))
    draw.text(
        (8, 20),
        f"finger aperture {aperture_px:.2f} px",
        fill=color,
        font=_font(10),
    )
    return image


def _render_montage_frame(
    *,
    task: str,
    camera_rows: Mapping[str, int],
    arrays: Mapping[str, np.ndarray],
    branch: int,
    predicted_q: Mapping[str, np.ndarray],
    cameras: Sequence[str],
    horizon: int,
    gripper_command: float,
    flip_vertical: bool | None = None,
) -> Image.Image:
    if not 0 <= horizon < HORIZON:
        raise ValueError("horizon is outside H1-H8")
    margin, gap, header = 12, 8, 58
    panel_size = 224
    panel = tuple(str(value) for value in cameras)
    width = margin * 2 + len(panel) * panel_size + (len(panel) - 1) * gap
    height = header + margin + panel_size * 2 + gap + margin
    canvas = Image.new("RGB", (width, height), (232, 236, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 8),
        f"{task} | branch {branch} | H{horizon + 1} | gripper command {gripper_command:+.2f}",
        fill=(20, 29, 36),
        font=_font(17),
    )
    draw.text(
        (margin, 34),
        "top: actual future   bottom: LeWM forecast on the same evaluation-only future frame",
        fill=(75, 88, 98),
        font=_font(11),
    )
    for column, camera in enumerate(panel):
        row = int(camera_rows[camera])
        true_rgb, true_q = _upright(
            task,
            np.asarray(arrays["future_rgb"])[row, branch, horizon],
            np.asarray(arrays["future_q_true"])[row, branch, horizon],
            flip_vertical=flip_vertical,
        )
        predicted_rgb, predicted = _upright(
            task,
            np.asarray(arrays["future_rgb"])[row, branch, horizon],
            np.asarray(predicted_q[camera])[horizon],
            flip_vertical=flip_vertical,
        )
        actual_panel = _draw_anchor_panel(
            true_rgb,
            true_q,
            color=TRUE_COLOR,
            title=f"{camera}: actual H{horizon + 1}",
            aperture_px=float(_aperture(true_q)),
        )
        predicted_panel = _draw_anchor_panel(
            predicted_rgb,
            predicted,
            color=PREDICTED_COLOR,
            title=f"{camera}: prediction vs truth H{horizon + 1}",
            aperture_px=float(_aperture(predicted)),
            reference_points=true_q,
        )
        x = margin + column * (panel_size + gap)
        canvas.paste(actual_panel, (x, header + margin))
        canvas.paste(predicted_panel, (x, header + margin + panel_size + gap))
    return canvas


def _save_video(path: Path, frames: Sequence[Image.Image], fps: float) -> None:
    writer = imageio.get_writer(
        path,
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
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _save_gif(path: Path, frames: Sequence[Image.Image], fps: float) -> None:
    duration = max(120, int(round(1000.0 / float(fps))))
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration,
        loop=0,
        disposal=2,
    )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 0 <= args.branch < 8 or not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("branch or fps is invalid")
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    dataset = args.raw_dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    index, arrays = open_raw_replay_arrays(dataset)
    cameras = _camera_panel(
        index,
        task=args.task,
        split=str(args.camera_split),
    )
    orbit_panel = "camera_clock_hour" in index
    physical_index, rows = _select_physical_bundle(
        index,
        arrays,
        task=args.task,
        branch=args.branch,
        cameras=cameras,
        split=str(args.camera_split),
    )
    camera_rows = {
        str(np.asarray(index["camera_name"])[row]): int(row) for row in rows
    }
    if tuple(camera for camera in cameras if camera in camera_rows) != cameras:
        raise RuntimeError("selected bundle does not contain the requested cameras")

    model, checkpoint = _load_checkpoint(args.checkpoint.expanduser().resolve(), device)
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    try:
        for camera in cameras:
            row = camera_rows[camera]
            batch = joint_batch_from_memmaps(
                index,
                arrays,
                bundle_indices=np.asarray([row], dtype=np.int64),
                branch_indices=np.asarray([args.branch], dtype=np.int64),
            )
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = model(batch)
            predicted = result.predicted_q_px[0].float().cpu().numpy()
            truth = batch["future_q_true"][0].float().cpu().numpy()
            predictions[camera] = predicted
            epe = np.linalg.norm(predicted - truth, axis=-1)
            aperture_error = np.abs(_aperture(predicted) - _aperture(truth))
            metrics[camera] = {
                "anchor_epe_px": float(epe.mean()),
                "anchor_epe_by_horizon_px": epe.mean(axis=-1).tolist(),
                "aperture_mae_px": float(aperture_error.mean()),
                "aperture_error_by_horizon_px": aperture_error.tolist(),
            }
    finally:
        del model
        torch.cuda.empty_cache()

    reference = camera_rows[cameras[0]]
    action = np.asarray(index["future_action"])[reference, args.branch]
    gripper = action[:, 3::4].mean(axis=-1)
    frames = [
        _render_montage_frame(
            task=args.task,
            camera_rows=camera_rows,
            arrays=arrays,
            branch=args.branch,
            predicted_q=predictions,
            cameras=cameras,
            horizon=horizon,
            gripper_command=float(gripper[horizon]),
            flip_vertical=False if orbit_panel else None,
        )
        for horizon in range(HORIZON)
    ]
    video_name = "multi_view_gripper_forecast.mp4"
    gif_name = "multi_view_gripper_forecast.gif"
    _save_video(output / video_name, frames, args.fps)
    _save_gif(output / gif_name, frames, args.fps)
    summary = {
        "schema_version": 1,
        "task": args.task,
        "branch": int(args.branch),
        "physical_bundle_index": int(physical_index),
        "camera_split": str(args.camera_split),
        "camera_panel": list(cameras),
        "camera_rows": camera_rows,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_protocol": str(checkpoint["protocol"]),
        "sigreg_weight": float(checkpoint["sigreg_weight"]),
        "grounding_mode": "nominal_clean_calibration",
        "future_rgb": "evaluation_only",
        "future_q_true": "evaluation_only",
        "metrics": metrics,
        "video": video_name,
        "gif": gif_name,
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows_html = "".join(
        "<tr>"
        f"<td>{html.escape(camera)}</td>"
        f"<td>{metrics[camera]['anchor_epe_px']:.2f}</td>"
        f"<td>{metrics[camera]['aperture_mae_px']:.2f}</td>"
        "</tr>"
        for camera in cameras
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Multi-view LeWM forecast</title>
<style>body{{margin:0;background:#edf1f3;color:#17212a;font-family:Arial,sans-serif;letter-spacing:0}}main{{max-width:1440px;margin:auto;padding:24px}}h1{{margin:0 0 8px}}p{{line-height:1.5;color:#56646e}}video,img{{width:100%;display:block;background:#111;border:1px solid #bcc7cd}}table{{margin-top:16px;width:100%;border-collapse:collapse;background:white}}th,td{{padding:9px 12px;border-bottom:1px solid #dce3e7;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head>
<body><main><h1>{len(cameras)}-View Joint LeWM Gripper Forecast</h1>
<p>One shared seven-task model, step {int(checkpoint['step'])}, SIGReg {float(checkpoint['sigreg_weight']):.2f}, protocol {html.escape(str(checkpoint['protocol']))}. Camera split {html.escape(str(args.camera_split))}. All columns use the same physical replay and action branch. Future RGB and true anchors are displayed only after inference.</p>
<video controls autoplay muted loop playsinline><source src="{video_name}" type="video/mp4"></video>
<p><a href="{gif_name}">Open GIF</a></p>
<table><thead><tr><th>Camera</th><th>Anchor EPE (px)</th><th>Finger-aperture MAE (px)</th></tr></thead><tbody>{rows_html}</tbody></table>
</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
