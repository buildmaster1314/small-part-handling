#!/usr/bin/env python3
"""Serve a bounded 4x5-command SO-101 FCPG + LeWM MPC loop."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image, ImageDraw

from fcpg.so101_active_gripper import (
    localize_gripper_from_pulse,
    next_gripper_position,
)
from fcpg.so101_kinematics import SO101_JOINT_NAMES
from fcpg.so101_realtime_planner import (
    PreparedPlannerConfig,
    PreparedSO101RealtimePlanner,
    RealtimePlan,
    plan_json,
)
from fcpg.so101_realtime_runtime import (
    ChunkExecution,
    LiveObservation,
    PlanRejectedError,
    RealtimeMPCController,
    ReplaySO101IO,
)
from fcpg.so101_retrieval_mpc import SO101_POSITION_LOWER, SO101_POSITION_UPPER
from fcpg.so101_shadow_session import observation_state, resize_rgb
from scripts.evaluate_so101_real_shadow_fcpg_lewm import (
    VIEWS,
    draw_grounding,
    load_paired_session,
)
from scripts.render_so101_mujoco_prediction import (
    DEFAULT_SCENE,
    SO101ReplayRenderer,
    fit_panel_image,
)
from scripts.visualize_so101_lewm import load_font


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_fcpg_context_bridge_val2_seed6126_2k/latest.pt"
)
DEFAULT_MODEL = REPO / (
    "outputs/lerobot/so101/"
    "so101_v9_local_multiscale_seed5131/latest.pt"
)
DEFAULT_CALIBRATION = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_dual_camera_calibration/calibration.json"
)
DEFAULT_SECONDARY = REPO / (
    "data/lerobot/so101/cache/my_first_task_clean_v1_column_224"
)
DEFAULT_REPLAY = REPO / (
    "outputs/lerobot/so101/real_dual_shadow_20260724_pre_traj02"
)
DEFAULT_OUTPUT = REPO / "outputs/lerobot/so101/realtime_fcpg_lewm_mpc"
DEFAULT_ROBOT_PORT = (
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115543-if00"
)
DEFAULT_WRIST_DEVICE = (
    "/dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0"
)
DEFAULT_COLUMN_DEVICE = (
    "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
)
COMMAND_COLOR = "#087e8b"
PREDICTION_COLOR = "#d1495b"
STATE_READ_RETRIES = 4
STATE_WRITE_RETRIES = 4
MONITOR_STREAM_FPS = 10.0
CAMERA_READ_TIMEOUT_MS = 1000.0
CAMERA_RECOVERY_TIMEOUT_MS = 2000.0
CAMERA_RECOVERY_ATTEMPTS = 3
CAMERA_RECOVERY_DELAY_S = 0.25


def _save_gripper_calibration_capture(
    root: Path,
    frames: dict[str, np.ndarray],
    states: np.ndarray,
    commands: np.ndarray,
    phase_ranges: list[dict[str, int | str]],
    *,
    sample_hz: float,
    closed_position: float,
    open_position: float,
) -> dict[str, str]:
    """Persist one complete pulse before attempting visual localization."""

    capture_dir = root / f"pulse_{time.time_ns()}"
    capture_dir.mkdir(parents=True, exist_ok=False)
    arrays = {view: np.asarray(frames[view], dtype=np.uint8) for view in VIEWS}
    state_array = np.asarray(states, dtype=np.float32)
    command_array = np.asarray(commands, dtype=np.float32)
    npz_path = capture_dir / "pulse.npz"
    np.savez_compressed(
        npz_path,
        wrist_frames=arrays["wrist"],
        column_frames=arrays["column"],
        states=state_array,
        commands=command_array,
        sample_hz=np.asarray(sample_hz, dtype=np.float32),
        closed_position=np.asarray(closed_position, dtype=np.float32),
        open_position=np.asarray(open_position, dtype=np.float32),
    )
    metadata = {
        "frame_count": int(len(state_array)),
        "sample_hz": float(sample_hz),
        "closed_position": float(closed_position),
        "open_position": float(open_position),
        "phase_ranges": phase_ranges,
    }
    metadata_path = capture_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    duration_ms = max(20, int(round(1000.0 / sample_hz)))
    gif_paths: dict[str, Path] = {}
    for view in VIEWS:
        images = [Image.fromarray(frame) for frame in arrays[view]]
        gif_path = capture_dir / f"{view}_pulse.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        gif_paths[view] = gif_path

    dual = [
        Image.fromarray(np.concatenate((wrist, column), axis=1))
        for wrist, column in zip(
            arrays["wrist"], arrays["column"], strict=True
        )
    ]
    dual_path = capture_dir / "dual_pulse.gif"
    dual[0].save(
        dual_path,
        save_all=True,
        append_images=dual[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    open_index = int(np.argmax(state_array[:, 5]))
    closed_index = int(np.argmin(state_array[:, 5]))
    for view in VIEWS:
        Image.fromarray(arrays[view][open_index]).save(capture_dir / f"{view}_open.png")
        Image.fromarray(arrays[view][closed_index]).save(
            capture_dir / f"{view}_closed.png"
        )
    index_path = capture_dir / "index.html"
    index_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>SO-101 gripper pulse</title>\n"
        "<style>body{font-family:sans-serif;margin:20px;background:#f5f6f7;color:#17202a}"
        "img{image-rendering:auto;max-width:100%;border:1px solid #ccd1d1}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}</style></head><body>"
        "<h1>SO-101 gripper calibration capture</h1><img src=\"dual_pulse.gif\">"
        "<div class=\"grid\"><section><h2>Wrist</h2><img src=\"wrist_pulse.gif\"></section>"
        "<section><h2>Column</h2><img src=\"column_pulse.gif\"></section></div>"
        "<p><a href=\"pulse.npz\">pulse.npz</a> | "
        "<a href=\"metadata.json\">metadata.json</a></p></body></html>",
        encoding="utf-8",
    )
    return {
        "directory": str(capture_dir),
        "npz": str(npz_path),
        "metadata": str(metadata_path),
        "index_html": str(index_path),
        "dual_gif": str(dual_path),
        "wrist_gif": str(gif_paths["wrist"]),
        "column_gif": str(gif_paths["column"]),
    }


def _preview_safe_state(state: np.ndarray) -> tuple[np.ndarray, bool]:
    """Clamp a copy for MuJoCo display without changing the executable command."""

    values = np.asarray(state, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("preview state must be one finite six-joint vector")
    safe = values.copy()
    safe[:5] = np.clip(safe[:5], -100.0, 100.0)
    safe[5] = np.clip(safe[5], 0.0, 100.0)
    return safe, not np.array_equal(safe, values)


def _draw_live_xy_trace(
    draw: ImageDraw.ImageDraw,
    commanded: np.ndarray,
    predicted: np.ndarray,
    current: int,
) -> None:
    """Draw command-reference and LeWM-predicted EEF paths in base-frame XY."""

    left, top, width, height = 60, 655, 520, 120
    both = np.concatenate([commanded[:, :2], predicted[:, :2]], axis=0) * 1000.0
    low = np.min(both, axis=0)
    high = np.max(both, axis=0)
    span = np.maximum(high - low, 1.0)
    margin = np.maximum(0.08 * span, 2.0)
    low -= margin
    high += margin

    def points(values: np.ndarray) -> list[tuple[float, float]]:
        xy = values[:, :2] * 1000.0
        return [
            (
                left + width * float((point[0] - low[0]) / (high[0] - low[0])),
                top + height * (1.0 - float((point[1] - low[1]) / (high[1] - low[1]))),
            )
            for point in xy[: current + 1]
        ]

    draw.rectangle(
        (left, top, left + width, top + height),
        fill="#f7f8f9",
        outline="#bcc5ca",
    )
    command_points = points(commanded)
    prediction_points = points(predicted)
    if len(command_points) > 1:
        draw.line(command_points, fill=COMMAND_COLOR, width=4)
        draw.line(prediction_points, fill=PREDICTION_COLOR, width=4)
    if command_points:
        cx, cy = command_points[-1]
        px, py = prediction_points[-1]
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=COMMAND_COLOR)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=PREDICTION_COLOR)
    draw.text(
        (left, top - 24),
        "Current H1-H8 base-frame XY path",
        fill="#20242a",
        font=load_font(18, True),
    )


def _compose_live_mujoco_frame(
    real_wrist: np.ndarray,
    predicted_image: np.ndarray,
    *,
    phase: str,
    candidate_index: int,
    horizon: int,
    raw_ahead: int,
    command_xyz: np.ndarray,
    predicted_xyz: np.ndarray,
    command_path: np.ndarray,
    prediction_path: np.ndarray,
    predicted_clipped: bool,
) -> Image.Image:
    """Match the established frozen-real / moving-prediction replay layout."""

    canvas = Image.new("RGB", (1280, 800), "#eef1f2")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, True)
    label_font = load_font(20, True)
    text_font = load_font(16)
    draw.text(
        (40, 22),
        "SO-ARM101 MuJoCo: live observation vs LeWM forecast",
        fill="#182024",
        font=title_font,
    )
    draw.text(
        (40, 58),
        f"phase {phase} | selected candidate #{candidate_index} | human-confirmed MPC preview",
        fill="#4a5358",
        font=text_font,
    )
    draw.text(
        (40, 84),
        f"H{horizon}: {raw_ahead} raw control steps ahead from the current RGB/state and candidate actions",
        fill="#4a5358",
        font=text_font,
    )
    left = Image.fromarray(fit_panel_image(real_wrist), mode="RGB")
    right = Image.fromarray(fit_panel_image(predicted_image), mode="RGB")
    canvas.paste(left, (40, 130))
    canvas.paste(right, (670, 130))
    draw.rectangle((40, 130, 610, 610), outline=COMMAND_COLOR, width=5)
    draw.rectangle((670, 130, 1240, 610), outline=PREDICTION_COLOR, width=5)
    draw.text(
        (56, 145),
        "REAL WRIST CAMERA OBSERVATION (FROZEN)",
        fill=COMMAND_COLOR,
        font=label_font,
    )
    draw.text(
        (686, 145),
        "LEWM -> MUJOCO PREDICTION",
        fill=PREDICTION_COLOR,
        font=label_font,
    )

    divergence_mm = 1000.0 * float(np.linalg.norm(predicted_xyz - command_xyz))
    draw.text(
        (630, 666),
        f"Pred-command EEF gap {divergence_mm:.1f} mm\n"
        + (
            "Prediction clipped for display only"
            if predicted_clipped
            else "No prediction display clipping"
        ),
        fill="#20242a",
        font=text_font,
    )
    _draw_live_xy_trace(draw, command_path, prediction_path, horizon)
    draw.text(
        (910, 678),
        "Teal: candidate command prefix via FK\n"
        "Red: LeWM-predicted state via FK\n"
        "Real RGB is frozen; MuJoCo is preview only",
        fill="#4a5358",
        font=text_font,
    )
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--model-checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--memory-policy",
        choices=("auto", "off", "always"),
        default="auto",
        help=(
            "V9 history policy: auto preserves task-local parity while current "
            "FCPG is reliable and uses memory only as a fallback."
        ),
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--live-anchor-calibration",
        type=Path,
        help="Previously captured open/closed moving-tip track for column geometry.",
    )
    parser.add_argument("--secondary-cache", type=Path, default=DEFAULT_SECONDARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-profile",
        type=Path,
        help="Persistent appearance templates; defaults to OUTPUT/target_profile.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--control-mode",
        choices=("phase_target", "anchor_trajectory"),
        default="phase_target",
        help=(
            "phase_target generates target-conditioned IK candidates and ranks "
            "their consequences with LeWM; anchor_trajectory is replay-only "
            "demonstration continuation."
        ),
    )
    parser.add_argument(
        "--max-plan-age-s",
        type=float,
        default=120.0,
        help=(
            "Maximum plan age in seconds; 0 disables time-based expiry and is "
            "only accepted without hardware motion permissions."
        ),
    )
    parser.add_argument("--max-state-drift", type=float, default=2.0)
    parser.add_argument("--max-target-drift-px", type=float, default=15.0)
    parser.add_argument("--max-command-jump", type=float, default=8.0)
    parser.add_argument(
        "--ik-arrival-raw-steps",
        type=int,
        default=20,
        help="Raw commands used by realtime IK to reach its terminal target.",
    )
    parser.add_argument(
        "--max-planar-step-m",
        type=float,
        default=0.025,
        help="Maximum XY travel of the tool per planning round, meters.",
    )
    parser.add_argument(
        "--max-z-step-m",
        type=float,
        default=0.020,
        help="Maximum vertical travel of the tool per planning round, meters.",
    )
    parser.add_argument(
        "--minimum-expected-motion",
        type=float,
        default=3.0,
        help=(
            "No-follow watchdog floor. Must not be below "
            "--settle-position-tolerance: the settle loop accepts any pose "
            "within that band, so a commanded move smaller than the band can "
            "legitimately produce zero observed motion."
        ),
    )
    parser.add_argument("--minimum-observed-motion", type=float, default=0.1)
    parser.add_argument("--command-guard-margin", type=float, default=0.5)
    parser.add_argument("--settle-timeout-s", type=float, default=3.0)
    parser.add_argument("--settle-position-tolerance", type=float, default=3.0)
    parser.add_argument("--settle-step-tolerance", type=float, default=0.25)
    parser.add_argument("--settle-samples", type=int, default=3)
    parser.add_argument("--replay-session", type=Path)
    parser.add_argument("--replay-index", type=int, default=54)
    parser.add_argument(
        "--home-state-report",
        "--object-prior-report",
        dest="home_state_report",
        type=Path,
        help="Load only the six-joint task-start state from a prior report.",
    )
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default="myfollower01")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--wrist-device", default=DEFAULT_WRIST_DEVICE)
    parser.add_argument("--column-device", default=DEFAULT_COLUMN_DEVICE)
    parser.add_argument("--execute-enabled", action="store_true")
    parser.add_argument("--gripper-calibration-enabled", action="store_true")
    parser.add_argument("--acknowledge-workspace-clear", action="store_true")
    parser.add_argument("--no-mujoco-preview", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    if args.http_port <= 0 or args.replay_index < 0:
        parser.error("HTTP port and replay index are invalid")
    for value in (
        args.control_hz,
        args.max_state_drift,
        args.max_target_drift_px,
        args.max_command_jump,
        args.command_guard_margin,
        args.settle_timeout_s,
        args.settle_position_tolerance,
        args.settle_step_tolerance,
        args.max_planar_step_m,
        args.max_z_step_m,
    ):
        if not np.isfinite(value) or value <= 0:
            parser.error("runtime thresholds must be finite and positive")
    if not np.isfinite(args.max_plan_age_s) or args.max_plan_age_s < 0:
        parser.error("--max-plan-age-s must be finite and non-negative")
    if args.settle_samples <= 0:
        parser.error("--settle-samples must be positive")
    if args.ik_arrival_raw_steps <= 0 or args.ik_arrival_raw_steps % 5:
        parser.error("--ik-arrival-raw-steps must be a positive multiple of 5")
    if args.command_guard_margin >= args.max_command_jump:
        parser.error("--command-guard-margin must be below --max-command-jump")
    if (
        args.execute_enabled or args.gripper_calibration_enabled
    ) and not args.acknowledge_workspace_clear:
        parser.error(
            "motion permissions require --acknowledge-workspace-clear"
        )
    if args.replay_session is not None and (
        args.execute_enabled or args.gripper_calibration_enabled
    ):
        parser.error("replay mode does not accept hardware motion permissions")
    if args.max_plan_age_s == 0 and (
        args.execute_enabled or args.gripper_calibration_enabled
    ):
        parser.error(
            "--max-plan-age-s 0 (no expiry) is not allowed with hardware "
            "motion permissions"
        )
    if args.control_mode == "anchor_trajectory" and (
        args.execute_enabled or args.gripper_calibration_enabled
    ):
        parser.error(
            "anchor_trajectory is replay-only; hardware motion requires "
            "--control-mode phase_target"
        )
    for path in (
        args.bridge_checkpoint,
        args.model_checkpoint,
        args.calibration,
        args.secondary_cache,
        args.live_anchor_calibration,
    ):
        if path is None:
            continue
        if not path.expanduser().exists():
            parser.error(f"required planner input does not exist: {path}")
    return args


def _load_live_anchor_calibration(
    planner: PreparedSO101RealtimePlanner, path: Path
) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("view") != "column":
        raise ValueError("unsupported live-anchor calibration contract")
    result = planner.set_live_moving_tip_track(
        "column",
        np.asarray(payload["moving_tip_track_candidates_px"], dtype=np.float32),
        np.asarray(payload["open_state"], dtype=np.float32),
        np.asarray(payload["closed_state"], dtype=np.float32),
    )
    planner.model_contract["live_anchor_calibration_source"] = str(
        path.expanduser().resolve()
    )
    return result


def _save_live_anchor_calibration(
    output: Path, calibration: dict[str, Any]
) -> Path:
    views = calibration.get("views")
    geometry = calibration.get("geometry")
    if not isinstance(views, dict) or not isinstance(geometry, dict):
        raise ValueError("calibration result is missing views or geometry")
    column_view = views.get("column")
    column_geometry = geometry.get("column")
    if not isinstance(column_view, dict) or not isinstance(column_geometry, dict):
        raise ValueError("calibration result is missing column geometry")
    payload = {
        "version": 1,
        "view": "column",
        "moving_tip_track_candidates_px": column_view[
            "moving_tip_track_candidates_px"
        ],
        "open_state": column_view["open_state"],
        "closed_state": column_view["closed_state"],
        "selected_candidate_index": column_geometry["candidate_index"],
        "source_capture": calibration.get("capture"),
    }
    path = output.expanduser().resolve() / "live_anchor_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _load_home_state_report(path: Path) -> np.ndarray:
    """Read a task-start state without importing stale image coordinates."""

    report = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if "current_state" not in report:
        raise ValueError("home-state report does not contain current_state")
    state = np.asarray(report["current_state"], dtype=np.float32)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("home-state report has an invalid current_state")
    return state.copy()


class HardwareSO101IO:
    """Persistent follower connection with bounded per-camera recovery."""

    is_hardware = True

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        gripper_calibration_closed: float,
        gripper_calibration_open: float,
    ) -> None:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        cameras = {
            "wrist": OpenCVCameraConfig(
                index_or_path=args.wrist_device,
                fps=60,
                width=640,
                height=480,
                fourcc="MJPG",
            ),
            "column": OpenCVCameraConfig(
                index_or_path=args.column_device,
                fps=30,
                width=640,
                height=480,
                fourcc="YUYV",
            ),
        }
        config = SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            calibration_dir=args.calibration_dir,
            use_degrees=False,
            disable_torque_on_disconnect=False,
            max_relative_target=args.max_command_jump,
            cameras=cameras,
        )
        self.follower = SO101Follower(config)
        self.max_command_jump = float(args.max_command_jump)
        self.settle_timeout_s = float(args.settle_timeout_s)
        self.settle_position_tolerance = float(args.settle_position_tolerance)
        self.settle_step_tolerance = float(args.settle_step_tolerance)
        self.settle_samples = int(args.settle_samples)
        calibration_positions = np.asarray(
            [gripper_calibration_closed, gripper_calibration_open],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(calibration_positions))
            or np.any(calibration_positions < 0.0)
            or np.any(calibration_positions > 100.0)
            or calibration_positions[1] - calibration_positions[0] < 20.0
        ):
            raise ValueError(
                "gripper calibration positions must be in [0,100] with "
                "at least 20 units of opening span"
            )
        self.gripper_calibration_closed = float(calibration_positions[0])
        self.gripper_calibration_open = float(calibration_positions[1])
        self.calibration_capture_root = (
            args.output.expanduser().resolve() / "gripper_calibration_captures"
        )
        self.camera_lock = threading.Lock()
        self.preview_cache_lock = threading.Lock()
        self.preview_cache: dict[str, np.ndarray] = {}
        self.camera_reconnect_counts = {view: 0 for view in VIEWS}
        self.camera_last_errors: dict[str, str | None] = {
            view: None for view in VIEWS
        }
        if not self.follower.calibration_fpath.is_file():
            raise FileNotFoundError(
                f"missing follower calibration {self.follower.calibration_fpath}"
            )
        try:
            self.follower.connect(calibrate=False)
        except BaseException:
            if self.follower.is_connected:
                self.follower.disconnect()
            raise
        if not self.follower.is_calibrated:
            self.close()
            raise RuntimeError("follower calibration does not match the connected bus")

    def observe(self) -> LiveObservation:
        state = self._read_state_only()
        with self.camera_lock:
            frames = {view: self._read_camera(view) for view in VIEWS}
        return LiveObservation(
            timestamp_s=time.perf_counter(),
            state=state,
            frames=frames,
        )

    def _read_camera(self, view: str) -> np.ndarray:
        """Read one camera while the caller owns ``camera_lock``."""

        camera = self.follower.cameras[view]
        try:
            raw_frame = camera.async_read(timeout_ms=CAMERA_READ_TIMEOUT_MS)
        except (TimeoutError, ConnectionError, RuntimeError) as error:
            raw_frame = self._recover_camera(view, error)
        frame = resize_rgb(np.asarray(raw_frame), 224)
        with self.preview_cache_lock:
            self.preview_cache[view] = frame.copy()
        return frame

    @staticmethod
    def _clear_camera_async_state(camera: Any) -> None:
        """Discard frames/events belonging to a released OpenCV handle."""

        event = getattr(camera, "new_frame_event", None)
        if event is not None:
            event.clear()
        if not hasattr(camera, "latest_frame"):
            return
        frame_lock = getattr(camera, "frame_lock", None)
        if frame_lock is None:
            camera.latest_frame = None
            return
        with frame_lock:
            camera.latest_frame = None

    @classmethod
    def _disconnect_camera(cls, camera: Any) -> None:
        """Stop a stale read thread and release its device handle."""

        try:
            camera.disconnect()
        except (ConnectionError, RuntimeError):
            pass
        finally:
            cls._clear_camera_async_state(camera)

    def _recover_camera(self, view: str, cause: Exception) -> np.ndarray:
        """Reconnect one failed camera through its stable configured path."""

        camera = self.follower.cameras[view]
        last_error: Exception = cause
        for attempt in range(1, CAMERA_RECOVERY_ATTEMPTS + 1):
            self._disconnect_camera(camera)
            if CAMERA_RECOVERY_DELAY_S > 0:
                time.sleep(CAMERA_RECOVERY_DELAY_S)
            try:
                camera.connect(warmup=False)
                # connect() does not start the asynchronous reader. Clearing here
                # guarantees async_read waits for a frame from the new handle.
                self._clear_camera_async_state(camera)
                frame = camera.async_read(
                    timeout_ms=CAMERA_RECOVERY_TIMEOUT_MS
                )
            except (TimeoutError, ConnectionError, RuntimeError) as error:
                last_error = error
                self._disconnect_camera(camera)
                continue
            self.camera_reconnect_counts[view] += 1
            self.camera_last_errors[view] = None
            return np.asarray(frame)

        self.camera_last_errors[view] = str(last_error)
        raise TimeoutError(
            f"camera {view!r} did not recover after "
            f"{CAMERA_RECOVERY_ATTEMPTS} reconnect attempts; "
            f"last error: {last_error}"
        ) from last_error

    def _cached_preview(self, view: str) -> np.ndarray | None:
        with self.preview_cache_lock:
            cached = self.preview_cache.get(view)
            return None if cached is None else cached.copy()

    def preview_frame(self, view: str) -> np.ndarray:
        """Read a monitor-only frame without advancing task/model state."""

        if view not in VIEWS:
            raise ValueError(f"unsupported preview view: {view}")
        if not self.camera_lock.acquire(blocking=False):
            cached = self._cached_preview(view)
            if cached is not None:
                return cached
            self.camera_lock.acquire()
        try:
            try:
                return self._read_camera(view)
            except TimeoutError:
                cached = self._cached_preview(view)
                if cached is not None:
                    return cached
                raise
        finally:
            self.camera_lock.release()

    def _read_state_only(self) -> np.ndarray:
        """Read joint positions without polling either camera."""

        positions = self.follower.bus.sync_read(
            "Present_Position", num_retry=STATE_READ_RETRIES
        )
        observation = {
            f"{motor}.pos": value for motor, value in positions.items()
        }
        return observation_state(observation)

    def _write_gripper_position(self, position: float) -> float:
        """Write one already-bounded gripper target without a redundant state read."""

        target = float(position)
        if not np.isfinite(target) or not 0.0 <= target <= 100.0:
            raise ValueError("gripper calibration target must lie in [0,100]")
        self.follower.bus.sync_write(
            "Goal_Position",
            {"gripper": target},
            num_retry=STATE_WRITE_RETRIES,
        )
        return target

    def _write_position_command(
        self,
        command: np.ndarray,
        previous: np.ndarray,
    ) -> np.ndarray:
        """Write one prevalidated arm command without LeRobot's zero-retry read."""

        target = np.asarray(command, dtype=np.float32)
        reference = np.asarray(previous, dtype=np.float32)
        if target.shape != (6,) or reference.shape != (6,):
            raise ValueError("SO-101 command and reference must be six-vectors")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(reference)):
            raise ValueError("SO-101 command and reference must be finite")
        if np.any(target < SO101_POSITION_LOWER) or np.any(
            target > SO101_POSITION_UPPER
        ):
            raise RuntimeError("SO-101 command lies outside hard position bounds")
        jump = float(np.max(np.abs(target - reference)))
        if jump > self.max_command_jump + 1e-5:
            raise RuntimeError(
                "SO-101 command exceeds the configured relative limit: "
                f"jump={jump:.3f}, limit={self.max_command_jump:.3f}"
            )
        goals = {
            name.removesuffix(".pos"): float(value)
            for name, value in zip(SO101_JOINT_NAMES, target, strict=True)
        }
        self.follower.bus.sync_write(
            "Goal_Position",
            goals,
            num_retry=STATE_WRITE_RETRIES,
        )
        return target.copy()

    def capture_initial(self, count: int, period_s: float) -> list[LiveObservation]:
        if count <= 0 or period_s <= 0:
            raise ValueError("initial capture settings are invalid")
        started = time.perf_counter()
        rows: list[LiveObservation] = []
        for index in range(count):
            deadline = started + index * period_s
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            rows.append(self.observe())
        return rows

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: threading.Event,
    ) -> ChunkExecution:
        values = np.asarray(chunk, dtype=np.float32)
        if values.shape != (5, 6) or not np.all(np.isfinite(values)):
            raise ValueError("hardware command chunk must be finite [5,6]")
        sent: list[np.ndarray] = []
        started = time.perf_counter()
        previous_command = self._read_state_only()
        for index, command in enumerate(values):
            if stop_event.is_set():
                raise RuntimeError("execution stopped by operator")
            deadline = started + index * period_s
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            actual = self._write_position_command(command, previous_command)
            if not np.allclose(actual, command, atol=1e-4, rtol=0.0):
                raise RuntimeError("LeRobot clipped or changed an interpolated command")
            if np.any(actual < SO101_POSITION_LOWER) or np.any(
                actual > SO101_POSITION_UPPER
            ):
                raise RuntimeError("LeRobot returned a command outside hard bounds")
            sent.append(actual)
            previous_command = actual

        target = sent[-1]
        previous_state = self._read_state_only()
        stable_samples = 0
        settle_deadline = time.perf_counter() + self.settle_timeout_s
        final_position_error = float("inf")
        final_step_delta = float("inf")
        while stable_samples < self.settle_samples:
            if stop_event.is_set():
                raise RuntimeError("execution stopped while waiting for the robot to settle")
            if time.perf_counter() >= settle_deadline:
                raise RuntimeError(
                    "robot did not settle after the five-command chunk: "
                    f"target_error={final_position_error:.3f}, "
                    f"step_delta={final_step_delta:.3f}"
                )
            time.sleep(period_s)
            current_state = self._read_state_only()
            # The gripper is the one joint that legitimately STALLS against
            # an object instead of reaching its target — that position error
            # is the grasp force (no force control exists). "Settled" for the
            # gripper therefore means "stopped moving"; the arm joints must
            # still reach their targets.
            final_position_error = float(
                np.max(np.abs(current_state[:5] - target[:5]))
            )
            final_step_delta = float(
                np.max(np.abs(current_state - previous_state))
            )
            if (
                final_position_error <= self.settle_position_tolerance
                and final_step_delta <= self.settle_step_tolerance
            ):
                stable_samples += 1
            else:
                stable_samples = 0
            previous_state = current_state

        # Camera reads are intentionally outside the command and settle loops.
        # The controller immediately rebuilds an eleven-frame visual history;
        # these repeated receipt observations only carry the settled endpoint.
        settled_observation = self.observe()
        return ChunkExecution(
            requested=values,
            sent=np.stack(sent),
            observations=(settled_observation,) * 5,
            duration_s=time.perf_counter() - started,
        )

    def calibrate_gripper(
        self,
        *,
        stop_event: threading.Event,
    ) -> dict[str, Any]:
        """Move only the gripper through one slow close-open-close pulse."""

        sample_hz = 20.0
        period_s = 1.0 / sample_hz
        maximum_step = 1.5
        settle_tolerance = 2.0
        stable_required = 6
        maximum_phase_frames = 100
        maximum_arm_drift = 1.5
        initial = self.observe()
        initial_arm = initial.state[:5].copy()
        initial_aperture = float(initial.state[5])
        open_position = self.gripper_calibration_open
        closed_position = self.gripper_calibration_closed
        frames: dict[str, list[np.ndarray]] = {view: [] for view in VIEWS}
        states: list[np.ndarray] = []
        commands: list[float] = []
        phase_ranges: list[dict[str, int | str]] = []
        maximum_observed_arm_drift = 0.0

        # The pulse must end where it began: the task flow may need the
        # gripper open (approach requires it), and a calibration that always
        # parks the jaws closed silently re-blocks the next plan.
        for phase_name, target in (
            ("close", closed_position),
            ("open", open_position),
            ("restore", initial_aperture),
        ):
            phase_start = len(states)
            stable = 0
            for _ in range(maximum_phase_frames):
                started = time.perf_counter()
                if stop_event.is_set():
                    raise RuntimeError("grounding calibration stopped by operator")
                observation = self.observe()
                drift = float(
                    np.max(np.abs(observation.state[:5] - initial_arm))
                )
                maximum_observed_arm_drift = max(
                    maximum_observed_arm_drift, drift
                )
                if drift > maximum_arm_drift:
                    raise RuntimeError(
                        "arm moved during gripper calibration: "
                        f"{drift:.3f} > {maximum_arm_drift:.3f}"
                    )
                for view in VIEWS:
                    frames[view].append(observation.frames[view].copy())
                states.append(observation.state.copy())

                command = next_gripper_position(
                    float(observation.state[5]), target, maximum_step
                )
                sent = self._write_gripper_position(command)
                commands.append(sent)
                if abs(float(observation.state[5]) - target) <= settle_tolerance:
                    stable += 1
                else:
                    stable = 0
                if stable >= stable_required:
                    break
                delay = period_s - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
            else:
                phase_states = np.asarray(states[phase_start:], dtype=np.float32)
                phase_gripper = phase_states[:, 5]
                raise RuntimeError(
                    "gripper did not settle during calibration phase "
                    f"{phase_name}: target={target:.3f}, "
                    f"start={phase_gripper[0]:.3f}, "
                    f"final={phase_gripper[-1]:.3f}, "
                    f"observed_range=[{np.min(phase_gripper):.3f},"
                    f"{np.max(phase_gripper):.3f}], "
                    f"last_command={commands[-1]:.3f}"
                )
            phase_ranges.append(
                {
                    "name": phase_name,
                    "start": phase_start,
                    "stop": len(states),
                }
            )

        state_array = np.stack(states).astype(np.float32)
        frame_arrays = {
            view: np.stack(frames[view]).astype(np.uint8, copy=False)
            for view in VIEWS
        }
        capture_root = getattr(self, "calibration_capture_root", None)
        capture = None
        if capture_root is not None:
            capture = _save_gripper_calibration_capture(
                Path(capture_root),
                frame_arrays,
                state_array,
                np.asarray(commands, dtype=np.float32),
                phase_ranges,
                sample_hz=sample_hz,
                closed_position=closed_position,
                open_position=open_position,
            )
        # The column camera is external and needs a live image-plane offset.
        # The wrist camera is rigidly attached to the gripper; its nominal
        # moving-tip projection is already expressed in the local camera frame.
        try:
            localizations = {
                "column": localize_gripper_from_pulse(
                    frame_arrays["column"],
                    state_array,
                    maximum_arm_motion=maximum_arm_drift,
                )
            }
        except BaseException as exc:
            location = None if capture is None else capture["index_html"]
            raise RuntimeError(
                f"gripper localization failed; cached capture={location}: {exc}"
            ) from exc
        view_results = {}
        for view, localization in localizations.items():
            view_results[view] = {
                **localization.to_dict(),
                "open_state": state_array[localization.open_frame_index].copy(),
                "closed_state": state_array[localization.closed_frame_index].copy(),
            }
        column = localizations["column"]
        return {
            **column.to_dict(),
            "center_px": np.asarray(column.center_px, dtype=np.float32),
            "state": state_array[-1].copy(),
            "views": view_results,
            "sample_hz": sample_hz,
            "frame_count": len(state_array),
            "maximum_command_step": maximum_step,
            "closed_position": closed_position,
            "open_position": open_position,
            "initial_position": initial_aperture,
            "final_position": float(state_array[-1, 5]),
            "restored_initial": bool(
                abs(float(state_array[-1, 5]) - initial_aperture)
                <= settle_tolerance
            ),
            "observed_gripper_range": [
                float(np.min(state_array[:, 5])),
                float(np.max(state_array[:, 5])),
            ],
            "maximum_observed_arm_drift": maximum_observed_arm_drift,
            "other_joint_commands_sent": False,
            "command_count": len(commands),
            "phase_ranges": phase_ranges,
            "capture": capture,
        }

    def close(self) -> None:
        follower = getattr(self, "follower", None)
        if follower is None:
            return
        if follower.is_connected:
            follower.disconnect()
        elif follower.bus.is_connected:
            follower.bus.disconnect(disable_torque=False)


class RoundArtifactWriter:
    def __init__(self, root: Path, *, render_mujoco: bool) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.render_mujoco = bool(render_mujoco)

    def __call__(self, plan: RealtimePlan) -> dict[str, str]:
        round_root = self.root / "rounds" / plan.guard.round_id
        round_root.mkdir(parents=True, exist_ok=False)
        assets: dict[str, str] = {}
        for view in VIEWS:
            raw_name = f"{view}_raw.jpg"
            Image.fromarray(plan.frames[view][-1], mode="RGB").save(
                round_root / raw_name, quality=93
            )
            grounding_name = f"{view}_grounding.jpg"
            draw_grounding(
                plan.frames[view][-1],
                plan.nominal[view][-1],
                plan.corrected[view][-1],
                plan.valid[view][-1],
            ).save(round_root / grounding_name, quality=93)
            assets[f"{view}_raw"] = self._url(plan, raw_name)
            assets[f"{view}_grounding"] = self._url(plan, grounding_name)

        target_name = "column_target_paths.jpg"
        self._target_overlay(plan).save(round_root / target_name, quality=94)
        assets["target_paths"] = self._url(plan, target_name)
        if self.render_mujoco and plan.selected_index is not None:
            preview_name = "mujoco_selected.gif"
            try:
                self._mujoco_gif(plan, round_root / preview_name)
            except Exception as exc:
                error_name = "mujoco_preview_error.txt"
                (round_root / error_name).write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
                assets["mujoco_preview_error"] = self._url(plan, error_name)
                print(
                    f"MuJoCo preview unavailable for {plan.guard.round_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            else:
                assets["mujoco_preview"] = self._url(plan, preview_name)

        (round_root / "plan.json").write_text(
            json.dumps(plan_json(plan), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assets["plan_json"] = self._url(plan, "plan.json")
        return assets

    def _url(self, plan: RealtimePlan, name: str) -> str:
        return f"/assets/rounds/{plan.guard.round_id}/{name}"

    @staticmethod
    def _target_overlay(plan: RealtimePlan) -> Image.Image:
        base_image = Image.fromarray(plan.frames["column"][-1], mode="RGB")
        image = base_image.copy()
        draw = ImageDraw.Draw(image)
        font = load_font(11, bold=True)
        current_anchors = np.asarray(
            plan.corrected["column"][-1], dtype=np.float32
        )
        anchor_names = ("C", "L", "R")
        command_colors = ((37, 99, 235), (85, 135, 235), (30, 68, 170))
        prediction_colors = ((22, 163, 74), (46, 211, 255), (255, 92, 185))

        command_anchors = getattr(plan, "commanded_anchor_px", None)
        prediction_anchors = getattr(plan, "predicted_anchor_px", None)
        if command_anchors is None and plan.commanded_gripper_px is not None:
            command_anchors = np.asarray(plan.commanded_gripper_px)[:, None]
        if prediction_anchors is None and plan.predicted_gripper_px is not None:
            prediction_anchors = np.asarray(plan.predicted_gripper_px)[:, None]

        def draw_anchor_paths(
            values: np.ndarray | None,
            colors: tuple[tuple[int, int, int], ...],
            prefix: str,
            width: int,
        ) -> None:
            if values is None:
                return
            paths = np.asarray(values, dtype=np.float32)
            if paths.ndim != 3 or paths.shape[-1] != 2:
                raise ValueError("anchor paths must have shape [H,K,2]")
            if paths.shape[1] not in (1, 3):
                raise ValueError("anchor paths must contain one or three anchors")
            indices = range(paths.shape[1])
            for anchor_index in indices:
                start_index = 0 if paths.shape[1] == 1 else anchor_index
                line = [
                    tuple(map(float, current_anchors[start_index])),
                    *[
                        tuple(map(float, point))
                        for point in paths[:, anchor_index]
                    ],
                ]
                color = colors[start_index]
                draw.line(line, fill=color, width=width)
                for point in line[1:]:
                    draw.ellipse(
                        (
                            point[0] - 2,
                            point[1] - 2,
                            point[0] + 2,
                            point[1] + 2,
                        ),
                        fill=color,
                    )
                end = line[-1]
                label = (
                    f"{prefix} H8"
                    if paths.shape[1] == 1
                    else f"{prefix}-{anchor_names[anchor_index]}"
                )
                draw.text(
                    (end[0] + 4, end[1] - 17 + 12 * anchor_index),
                    label,
                    fill=color,
                    font=font,
                    stroke_width=2,
                    stroke_fill=(245, 245, 245),
                )

        draw_anchor_paths(command_anchors, command_colors, "IK", 2)
        draw_anchor_paths(prediction_anchors, prediction_colors, "PRED", 3)

        # Paths are intentionally drawn first. The target and current anchors
        # must remain visible even when a forecast starts at the current point.
        anchors = current_anchors
        anchor_colors = ((255, 255, 255), (46, 211, 255), (255, 92, 185))
        for index, (point, color) in enumerate(zip(anchors, anchor_colors, strict=True)):
            radius = 5 if index == 0 else 3
            x, y = map(float, point)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(20, 20, 20),
                width=2,
            )
        draw.text(
            (anchors[0, 0] + 6, anchors[0, 1] - 12),
            "CURRENT",
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(20, 20, 20),
        )

        if plan.target.get("visible", False):
            x, y = map(float, plan.target["centroid_px"])
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 210, 36), width=3)
            draw.line((x - 11, y, x + 11, y), fill=(255, 210, 36), width=2)
            draw.line((x, y - 11, x, y + 11), fill=(255, 210, 36), width=2)
            draw.text(
                (x + 9, y - 15),
                _target_overlay_label(plan.target.get("kind")),
                fill=(255, 220, 40),
                font=font,
                stroke_width=2,
                stroke_fill=(20, 20, 20),
            )

        if prediction_anchors is None and command_anchors is None:
            return image

        path_arrays = [current_anchors]
        for values in (prediction_anchors, command_anchors):
            if values is not None:
                path_arrays.append(np.asarray(values, dtype=np.float32).reshape(-1, 2))
        all_points = np.concatenate(path_arrays, axis=0)
        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)
        center = 0.5 * (minimum + maximum)
        crop_size = max(32.0, float(np.max(maximum - minimum)) + 18.0)
        crop_size = min(crop_size, 112.0)
        left = int(np.clip(round(center[0] - crop_size / 2), 0, 224 - crop_size))
        top = int(np.clip(round(center[1] - crop_size / 2), 0, 224 - crop_size))
        right = int(round(left + crop_size))
        bottom = int(round(top + crop_size))
        zoom = base_image.crop((left, top, right, bottom)).resize(
            (224, 224), resample=Image.Resampling.BILINEAR
        )
        zoom_draw = ImageDraw.Draw(zoom)

        def zoom_point(point: np.ndarray) -> tuple[float, float]:
            return (
                float((point[0] - left) * 224.0 / (right - left)),
                float((point[1] - top) * 224.0 / (bottom - top)),
            )

        def draw_zoom_paths(
            values: np.ndarray | None,
            colors: tuple[tuple[int, int, int], ...],
            width: int,
        ) -> None:
            if values is None:
                return
            paths = np.asarray(values, dtype=np.float32)
            for anchor_index in range(paths.shape[1]):
                line = [
                    zoom_point(current_anchors[anchor_index]),
                    *[zoom_point(point) for point in paths[:, anchor_index]],
                ]
                zoom_draw.line(line, fill=colors[anchor_index], width=width)
                for point in line[1:]:
                    zoom_draw.ellipse(
                        (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                        fill=colors[anchor_index],
                    )

        draw_zoom_paths(command_anchors, command_colors, 4)
        draw_zoom_paths(prediction_anchors, prediction_colors, 6)
        for index, point in enumerate(current_anchors):
            x, y = zoom_point(point)
            radius = 6 if index == 0 else 4
            zoom_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=anchor_colors[index],
                outline=(20, 20, 20),
                width=2,
            )
        zoom_draw.rectangle((0, 0, 223, 223), outline=(255, 210, 36), width=3)
        zoom_draw.rectangle((5, 5, 218, 51), fill=(20, 20, 20))
        zoom_draw.text(
            (9, 7),
            f"ZOOM {224 / (right - left):.1f}x",
            fill=(255, 235, 120),
            font=font,
        )
        legend_x = 56
        for row, (row_label, colors) in enumerate(
            (("PRED", prediction_colors), ("IK", command_colors))
        ):
            y = 23 + 15 * row
            zoom_draw.text((9, y - 6), row_label, fill=(245, 245, 245), font=font)
            for index, label in enumerate(anchor_names):
                x = legend_x + index * 48
                zoom_draw.line((x, y, x + 12, y), fill=colors[index], width=5)
                zoom_draw.text((x + 15, y - 6), label, fill=(245, 245, 245), font=font)
        composite = Image.new("RGB", (448, 224), color=(20, 20, 20))
        composite.paste(image, (0, 0))
        composite.paste(zoom, (224, 0))
        return composite

    def _mujoco_gif(self, plan: RealtimePlan, output: Path) -> None:
        if plan.commanded_states is None or plan.predicted_states is None:
            raise RuntimeError("MuJoCo preview requires selected command and prediction")
        renderer = SO101ReplayRenderer(
            DEFAULT_SCENE,
            azimuth=135.0,
            elevation=-25.0,
            distance=0.72,
        )
        commanded = np.concatenate(
            (plan.current_state[None], np.asarray(plan.commanded_states)), axis=0
        )
        predicted = np.concatenate(
            (plan.current_state[None], np.asarray(plan.predicted_states)), axis=0
        )
        command_xyz: list[np.ndarray] = []
        prediction_images: list[np.ndarray] = []
        prediction_xyz: list[np.ndarray] = []
        prediction_clipped: list[bool] = []
        try:
            for command_state, prediction_state in zip(
                commanded, predicted, strict=True
            ):
                command_display, command_was_clipped = _preview_safe_state(
                    command_state
                )
                prediction_display, prediction_was_clipped = _preview_safe_state(
                    prediction_state
                )
                command_image, command_position, command_renderer_clipped = (
                    renderer.render(command_display)
                )
                prediction_image, prediction_position, prediction_renderer_clipped = (
                    renderer.render(prediction_display)
                )
                del command_image
                command_xyz.append(command_position)
                prediction_images.append(prediction_image)
                prediction_xyz.append(prediction_position)
                prediction_clipped.append(
                    prediction_was_clipped or prediction_renderer_clipped
                )
                if command_was_clipped or command_renderer_clipped:
                    prediction_clipped[-1] = True
        finally:
            renderer.close()

        command_path = np.asarray(command_xyz, dtype=np.float64)
        prediction_path = np.asarray(prediction_xyz, dtype=np.float64)
        frozen_wrist = np.asarray(plan.frames["wrist"][-1], dtype=np.uint8)
        frames = [
            _compose_live_mujoco_frame(
                frozen_wrist,
                prediction_images[index],
                phase=plan.phase.value,
                candidate_index=int(plan.selected_index),
                horizon=index,
                raw_ahead=index * 5,
                command_xyz=command_path[index],
                predicted_xyz=prediction_path[index],
                command_path=command_path,
                prediction_path=prediction_path,
                predicted_clipped=prediction_clipped[index],
            )
            for index in range(len(predicted))
        ]
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=320,
            loop=0,
            optimize=False,
        )

    def close(self) -> None:
        return None


def page() -> str:
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 FCPG + LeWM MPC</title>
<style>
:root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#eef0f2;color:#17191c}
*{box-sizing:border-box}body{margin:0}header{height:58px;background:#181b1f;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 20px;position:sticky;top:0;z-index:3}
h1{font-size:18px;margin:0;font-weight:650}#badge{font:600 13px ui-monospace,monospace;padding:6px 9px;border:1px solid #59616b;border-radius:4px}
main{max-width:1320px;margin:auto;padding:14px 18px 94px}.statusbar{display:grid;grid-template-columns:repeat(7,minmax(100px,1fr));gap:1px;background:#c8cdd2;border:1px solid #c8cdd2}.metric{background:#fff;padding:9px 11px;min-height:57px}.metric span{display:block;font-size:11px;color:#616970;text-transform:uppercase}.metric strong{font-size:15px;overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.panel{background:#fff;border:1px solid #c8cdd2;border-radius:6px;overflow:hidden}.panel h2{font-size:13px;margin:0;padding:9px 11px;border-bottom:1px solid #d7dadd}.panel img{width:100%;display:block;aspect-ratio:4/3;object-fit:contain;background:#151719}.panel.planning img{aspect-ratio:1/1;cursor:crosshair}.panel.wide{grid-column:1/-1}.panel.wide img{aspect-ratio:auto;max-height:490px}
.live-panel{border-color:#31835d}.live-panel h2{display:flex;align-items:center;justify-content:space-between}.live-tag{background:#17653a;color:#fff;border-radius:3px;padding:2px 6px;font:700 10px ui-monospace,monospace}.clock-note{color:#59616b;font-weight:500}
.target-tools{grid-column:1/-1;background:#fff;border:1px solid #c8cdd2;border-radius:6px;padding:10px 12px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}.target-tools[hidden]{display:none}.target-tools strong{font-size:13px}.target-modes{display:flex;border:1px solid #8b949c;border-radius:5px;overflow:hidden}.target-modes button{height:34px;border-radius:0;background:#fff;color:#2a3035;padding:0 13px}.target-modes button[aria-pressed="true"]{background:#235c82;color:#fff}.target-tools label{display:flex;align-items:center;gap:7px;font-size:12px}.target-tools input{width:130px}.target-state{margin-left:auto;font:12px ui-monospace,monospace;color:#4d555c}.trajectory-mode .panel.planning img{cursor:default}.trajectory-mode #column{aspect-ratio:2/1}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid #e1e4e7}tr.selected{background:#ecfdf3}tr.unsafe{color:#8a1c1c}.log{font:12px ui-monospace,monospace;white-space:pre-wrap;max-height:210px;overflow:auto;padding:10px;margin:0;background:#111418;color:#d8e4df;user-select:text;cursor:text}
.request-error{display:none;grid-column:1/-1;margin-top:12px;border:1px solid #b42318;background:#fff4f2;color:#731b14;padding:10px}.request-error.visible{display:block}.request-error header{height:auto;position:static;background:transparent;color:inherit;padding:0;display:flex;gap:10px}.request-error pre{white-space:pre-wrap;overflow:auto;user-select:text;cursor:text;margin:8px 0 0;max-height:180px}.copy-error,.copy-diagnostics{height:32px;color:#fff;padding:0 11px;font-size:12px}.copy-error{background:#731b14;margin-left:auto}.copy-diagnostics{background:#235c82;margin-left:auto}.runtime-title{display:flex;align-items:center;gap:8px}
.controls{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #aeb4ba;padding:12px 18px;display:flex;justify-content:center;gap:10px;align-items:center;z-index:4}button{height:44px;border:0;border-radius:5px;padding:0 18px;font-size:14px;font-weight:650;cursor:pointer;white-space:nowrap}.execute{background:#17653a;color:#fff}.auto{background:#0f766e;color:#fff}.auto[aria-pressed="true"]{background:#b45309}.replan{background:#235c82;color:#fff}.calibrate{background:#5d4777;color:#fff}.home{background:#6b4f1d;color:#fff}.stop{background:#b42318;color:#fff}button:disabled{opacity:.4;cursor:not-allowed}#busy{min-width:150px;color:#4d555c;font-size:13px}
@media(max-width:800px){.statusbar{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.panel.wide{grid-column:auto}.controls{justify-content:flex-start;overflow-x:auto}}
</style></head><body>
<header><h1>SO-101 FCPG + LeWM Receding-Horizon MPC</h1><div id="badge">LOADING</div></header>
<main><section class="statusbar">
<div class="metric"><span>Mode</span><strong id="mode">-</strong></div><div class="metric"><span>Phase</span><strong id="phase">-</strong></div><div class="metric"><span>Round</span><strong id="round">-</strong></div><div class="metric"><span>Candidate</span><strong id="candidate">-</strong></div><div class="metric"><span>Operation</span><strong id="operation">-</strong></div><div class="metric"><span>State</span><strong id="state">-</strong></div><div class="metric"><span>LeWM Clock</span><strong id="modelClock">-</strong></div>
</section><section class="grid">
<section class="request-error" id="requestError"><header><strong id="requestErrorTitle">请求失败</strong><button class="copy-error" id="copyError">复制错误</button></header><pre id="requestErrorText" tabindex="0"></pre></section>
<section class="target-tools" id="targetTools"><strong>规划图点击目标</strong><div class="target-modes"><button id="pickupMode" aria-pressed="true">抓取目标</button><button id="dropMode" aria-pressed="false">放置区域</button></div><label>模板半径 <input id="targetRadius" type="range" min="8" max="40" step="1" value="18"><output id="radiusValue">18 px</output></label><span class="target-state" id="targetState">pickup - / drop -</span></section>
<article class="panel live-panel"><h2>Wrist live monitor <span class="clock-note">not model history</span><span class="live-tag">LIVE</span></h2><img id="liveWrist" alt="Wrist live camera"></article>
<article class="panel live-panel"><h2>Column live monitor <span class="clock-note">not model history</span><span class="live-tag">LIVE</span></h2><img id="liveColumn" alt="Column live camera"></article>
<article class="panel planning"><h2>Wrist planning snapshot: FCPG anchors / pickup reference</h2><img id="wrist"></article>
<article class="panel planning"><h2>Column planning snapshot: current anchors / LeWM forecast / IK trajectory</h2><img id="column"></article>
<article class="panel wide"><h2>Frozen real wrist observation vs LeWM MuJoCo prediction <span id="previewWarning"></span></h2><img id="mujoco"></article>
<article class="panel wide"><h2>Ranked action prefixes</h2><table><thead><tr><th>#</th><th>Demo</th><th>Cost</th><th>Future</th><th>Retrieval</th><th>IK mm</th><th>Safe</th><th>Reason</th></tr></thead><tbody id="candidates"></tbody></table></article>
<article class="panel wide"><h2 class="runtime-title">Runtime <button class="copy-diagnostics" id="copyDiagnostics">复制完整诊断</button></h2><pre class="log" id="log" tabindex="0"></pre></article>
</section></main>
<div class="controls"><button class="execute" id="execute">执行 5 步</button><button class="auto" id="auto" aria-pressed="false">自动连续：关</button><button class="replan" id="replan">重新观测并规划</button><button class="calibrate" id="calibrate">标定夹爪位置</button><button class="home" id="home">回到初始位置</button><button class="stop" id="stop">停止</button><span id="busy"></span></div>
<script>
let latest=null,requestInFlight=false,refreshInFlight=false,stopInFlight=false,refreshTimer=null,targetRole='pickup';const $=id=>document.getElementById(id);
function fmt(v,n=3){return Number.isFinite(v)?v.toFixed(n):'-'}
function setAsset(id,url,round){const el=$(id);if(!url){el.style.display='none';el.removeAttribute('src');delete el.dataset.assetKey;return}const key=url+'?r='+encodeURIComponent(round||'none');if(el.dataset.assetKey!==key){el.src=key;el.dataset.assetKey=key}el.style.display='block'}
function renderCandidates(rows,selected){const body=$('candidates');body.replaceChildren();rows.slice().sort((x,y)=>x.cost-y.cost).forEach(r=>{const tr=document.createElement('tr');const source=Number.isInteger(r.episode)?`ep${r.episode}/f${r.frame}`:(r.source||'-');const ikMm=Number.isFinite(r.anchor_ik_max_error_m)?1000*r.anchor_ik_max_error_m:NaN;tr.className=r.index===selected?'selected':(r.safe?'':'unsafe');[r.index,source,fmt(r.cost),fmt(r.demonstration_future_cost),fmt(r.retrieval_cost),fmt(ikMm),r.safe?'yes':'no',(r.reasons||[]).join(', ')].forEach(value=>{const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td)});body.appendChild(tr)})}
function controls(s){const p=(s&&s.plan)||{};const backendBusy=Boolean(s&&s.active_operation);const busy=requestInFlight||backendBusy;$('execute').disabled=busy||!s||!s.execute_enabled||!s.execution_armed||s.stopped||!p.executable;$('replan').disabled=busy||!s||s.stopped;$('calibrate').disabled=busy||!s||!s.calibration_available;$('home').disabled=busy||!s||!s.execute_enabled||!s.home_available||s.stopped;$('pickupMode').disabled=busy;$('dropMode').disabled=busy;$('targetRadius').disabled=busy;$('auto').disabled=!s||!s.execute_enabled||s.stopped;$('stop').disabled=!s||s.stopped||stopInFlight}
function render(s){latest=s;const p=s.plan||{};const trajectoryMode=p.phase==='trajectory';document.body.classList.toggle('trajectory-mode',trajectoryMode);$('targetTools').hidden=trajectoryMode;const ready=Boolean(s.execute_enabled&&s.execution_armed&&p.executable&&!s.stopped);const operation=s.active_operation||'-';const permission=s.execute_enabled?'armed':(s.calibration_enabled?'calibration-only':'read-only');$('mode').textContent=s.mode+' / '+permission;$('phase').textContent=p.phase||'-';$('round').textContent=p.round_id||'-';$('candidate').textContent=p.selected_index===null||p.selected_index===undefined?'BLOCKED':'#'+p.selected_index;$('operation').textContent=operation;$('state').textContent=s.input_exhausted?'REPLAY COMPLETE':(s.stopped?'STOPPED':(ready?'READY':(s.execute_enabled&&!s.execution_armed?'REPLAN REQUIRED':'BLOCKED')));$('modelClock').textContent=(s.observation_clock||'-')+' / '+(s.model_history_frames??'-')+' frames';$('badge').textContent=s.error?'FAULT':(s.active_operation?String(s.active_operation).toUpperCase():(s.input_exhausted?'COMPLETE':(ready?'READY':'BLOCKED')));$('badge').style.background=s.error?'#8f1d18':(ready?'#17653a':'#5a626b');
const a=s.assets||{};setAsset('wrist',a.wrist_grounding,p.round_id);setAsset('column',a.target_paths,p.round_id);setAsset('mujoco',a.mujoco_preview,p.round_id);$('previewWarning').textContent=a.mujoco_preview_error?'(preview unavailable)':'';renderCandidates(p.candidates||[],p.selected_index);
const profile=p.target_profile||{};$('targetState').textContent='pickup '+(profile.pickup_ready?'ready':'-')+' / wrist '+(profile.wrist_pickup_ready?'ready':'-')+' / drop '+(profile.drop_zone_ready?'ready':'-');const diagnostics={request_error:s.request_error||null,request_error_code:s.request_error_code||null,error:s.error,active_operation:s.active_operation,observation_clock:s.observation_clock,monitor_frames_feed_model:s.monitor_frames_feed_model,model_history_frames:s.model_history_frames,blocked:p.blocked_reasons,phase:p.phase_status,target:p.target,target_profile:profile,grounding:p.grounding,calibration:s.calibration,preview_error_asset:a.mujoco_preview_error||null,last_execution_batch:s.last_execution_batch,latest_receipt:s.latest_receipt};$('log').textContent=JSON.stringify(diagnostics,null,2);controls(s);if(s.error&&!$('requestError').classList.contains('visible'))showRequestError(s.error)}
function errorTitle(message){const text=String(message||'');if(text.includes('status packet'))return '机械臂通信中断';if(text.includes('Timed out waiting for frame'))return '相机取帧超时';if(text.includes('live state drift')||text.includes('fresh plan')||text.includes('stale planning'))return '现场状态已变化，请重新规划';if(text.includes('did not settle'))return '机械臂未在时限内稳定';if(text.includes('clipped or changed'))return 'LeRobot 修改了命令，执行已停止';if(text.includes('another planning'))return '已有操作正在进行';if(text.includes('no complete next five-frame'))return '回放数据已经结束';return '请求失败'}
function showRequestError(message){$('requestErrorTitle').textContent=errorTitle(message);$('requestErrorText').textContent=String(message);$('requestError').classList.add('visible')}
function clearRequestError(){$('requestErrorText').textContent='';$('requestError').classList.remove('visible')}
async function readJson(response){const text=await response.text();try{return JSON.parse(text)}catch(e){throw new Error(`HTTP ${response.status}: ${text.slice(0,500)}`)}}
function scheduleRefresh(delay=1500){clearTimeout(refreshTimer);refreshTimer=setTimeout(refresh,delay)}
async function refresh(){if(requestInFlight||refreshInFlight){scheduleRefresh();return}refreshInFlight=true;try{const r=await fetch('/api/status',{cache:'no-store'});const d=await readJson(r);if(!r.ok)throw new Error(d.request_error||`HTTP ${r.status}`);render(d)}catch(e){showRequestError(String(e));$('badge').textContent='DISCONNECTED';$('badge').style.background='#8f1d18'}finally{refreshInFlight=false;scheduleRefresh()}}
async function post(path,payload={},label='处理中'){if(requestInFlight){showRequestError('another planning or execution request is active');return}requestInFlight=true;clearTimeout(refreshTimer);$('busy').textContent=label;clearRequestError();controls(latest);try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await readJson(r);render(d);if(!r.ok)showRequestError(d.request_error||d.error||`HTTP ${r.status}`)}catch(e){showRequestError(String(e))}finally{requestInFlight=false;$('busy').textContent='';controls(latest);scheduleRefresh(250)}}
let autoRun=false,autoIdle=0,autoChunks=-1,autoBlockedKey='',autoBlockedCount=0;
function setAuto(v){autoRun=v;if(!v){autoIdle=0;autoBlockedKey='';autoBlockedCount=0}$('auto').setAttribute('aria-pressed',String(v));$('auto').textContent='自动连续：'+(v?'开':'关')}
function autoTick(){if(!autoRun)return;if(requestInFlight||stopInFlight)return;const s=latest;if(!s)return;if(s.stopped||s.error){setAuto(false);return}if(s.active_operation)return;
const cc=s.completed_chunks??-1;if(cc!==autoChunks){autoIdle=0;autoChunks=cc}
const p=s.plan||{};const blockedKey=(p.phase||'')+':'+JSON.stringify(p.blocked_reasons||[]);
if((p.blocked_reasons||[]).includes('no_safe_phase_compatible_candidate')||(p.blocked_reasons||[]).includes('grasp_empty_close_retry_exhausted')){if(blockedKey===autoBlockedKey){autoBlockedCount++}else{autoBlockedKey=blockedKey;autoBlockedCount=1}if(autoBlockedCount>=3){setAuto(false);showRequestError('自动模式已停止：'+(p.blocked_reasons||[]).join(', ')+'（确定性阻塞，重规划无法解除）');return}}else{autoBlockedKey='';autoBlockedCount=0}
if(p.executable&&s.execution_armed){post('/api/execute',{round_id:p.round_id,confirmed:true,max_chunks:24},'自动连续执行中')}else{autoIdle++;if(autoIdle>8){setAuto(false);showRequestError('自动模式已暂停：连续多轮无法执行，请查看 blocked 原因');return}post('/api/plan',{},'自动重新规划')}}
setInterval(autoTick,1200);
async function stopNow(){if(stopInFlight)return;setAuto(false);stopInFlight=true;$('busy').textContent='正在停止';controls(latest);try{const r=await fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const d=await readJson(r);render(d);if(!r.ok)showRequestError(d.request_error||d.error||`HTTP ${r.status}`)}catch(e){showRequestError(String(e))}finally{stopInFlight=false;$('busy').textContent='';controls(latest);scheduleRefresh(250)}}
function selectTargetRole(role){targetRole=role;$('pickupMode').setAttribute('aria-pressed',String(role==='pickup'));$('dropMode').setAttribute('aria-pressed',String(role==='drop_zone'));$('targetRadius').value=role==='pickup'?'18':'28';$('radiusValue').textContent=$('targetRadius').value+' px'}
function registerTarget(view,event){if(requestInFlight||latest?.plan?.phase==='trajectory')return;if(targetRole==='drop_zone'&&view!=='column'){showRequestError('放置区域只能在 column 规划图中设置');return}const rect=event.currentTarget.getBoundingClientRect();const x=(event.clientX-rect.left)*224/rect.width;const y=(event.clientY-rect.top)*224/rect.height;const radius=Number($('targetRadius').value);const label=targetRole==='pickup'?'抓取目标':'放置区域';if(confirm('使用当前 '+view+' 规划图中的点击区域设置'+label+'？'))post('/api/target-reference',{role:targetRole,view:view,centroid_px:[x,y],radius_px:radius},'正在保存目标外观并重新规划')}
function attachLiveStream(id,path){const el=$(id);let retry=null;const connect=()=>{clearTimeout(retry);el.src=path+'?t='+Date.now()};el.onerror=()=>{clearTimeout(retry);retry=setTimeout(connect,1000)};connect()}
$('pickupMode').onclick=()=>selectTargetRole('pickup');$('dropMode').onclick=()=>selectTargetRole('drop_zone');$('targetRadius').oninput=()=>$('radiusValue').textContent=$('targetRadius').value+' px';$('wrist').onclick=e=>registerTarget('wrist',e);$('column').onclick=e=>registerTarget('column',e);$('execute').onclick=()=>{if(!latest||!latest.plan)return;post('/api/execute',{round_id:latest.plan.round_id,confirmed:true},'正在执行 5 步并重新观测')};$('replan').onclick=()=>post('/api/plan',{},'正在读取双相机并规划');$('auto').onclick=()=>setAuto(!autoRun);$('calibrate').onclick=()=>{setAuto(false);if(confirm('保持工作区清空。仅让夹爪慢速开合一次并标定 column 中的夹爪位置？'))post('/api/calibrate',{confirmed:true},'正在开合夹爪并标定')};$('home').onclick=()=>{setAuto(false);if(confirm('缓慢回到记录的任务初始位置？'))post('/api/home',{confirmed:true},'正在返回初始位置')};$('stop').onclick=stopNow;attachLiveStream('liveWrist','/stream/wrist.mjpg');attachLiveStream('liveColumn','/stream/column.mjpg');refresh();
async function copyText(value,label){try{await navigator.clipboard.writeText(value)}catch(e){const area=document.createElement('textarea');area.value=value;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();document.execCommand('copy');area.remove()}$('busy').textContent=label;setTimeout(()=>$('busy').textContent='',1200)}
$('copyError').onclick=()=>copyText($('requestErrorText').textContent,'错误已复制');
$('copyDiagnostics').onclick=()=>copyText($('log').textContent,'完整诊断已复制');
</script></body></html>"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _jpeg_bytes(frame: np.ndarray) -> bytes:
    values = np.asarray(frame)
    if values.shape != (224, 224, 3) or values.dtype != np.uint8:
        raise ValueError("monitor frame must be uint8 [224,224,3]")
    output = BytesIO()
    Image.fromarray(values, mode="RGB").save(
        output,
        format="JPEG",
        quality=82,
        optimize=False,
    )
    return output.getvalue()


def _target_overlay_label(kind: str | None) -> str:
    return {
        "pickup": "PICKUP TARGET",
        "drop_zone": "DROP ZONE",
    }.get(kind, "TARGET")


def _request_error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, PlanRejectedError):
        return "plan_rejected"
    if isinstance(exc, (TypeError, ValueError, json.JSONDecodeError)):
        return "invalid_request"
    if "status packet" in message:
        return "robot_communication"
    if "timed out waiting for frame" in message:
        return "camera_timeout"
    if "did not settle" in message:
        return "robot_not_settled"
    if "clipped or changed" in message:
        return "command_changed"
    if "another planning or execution" in message:
        return "operation_busy"
    return "runtime_error"


def handler_factory(
    controller: RealtimeMPCController, asset_root: Path
) -> type[BaseHTTPRequestHandler]:
    root = asset_root.expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(_jsonable(payload), indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream_monitor(self, view: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            period_s = 1.0 / MONITOR_STREAM_FPS
            try:
                while True:
                    started = time.perf_counter()
                    payload = _jpeg_bytes(controller.preview_frame(view))
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    delay = period_s - (time.perf_counter() - started)
                    if delay > 0:
                        time.sleep(delay)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                print(
                    f"{view} monitor stream stopped: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = page().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/status":
                self.send_json(controller.status())
                return
            if parsed.path in ("/stream/wrist.mjpg", "/stream/column.mjpg"):
                view = Path(parsed.path).stem
                self.stream_monitor(view)
                return
            if parsed.path.startswith("/assets/"):
                relative = Path(unquote(parsed.path.removeprefix("/assets/")))
                target = (root / relative).resolve()
                if root not in target.parents or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = target.read_bytes()
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4096:
                raise ValueError("request body is too large")
            if length == 0:
                return {}
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_POST(self) -> None:
            try:
                payload = self._body()
                if self.path == "/api/plan":
                    result = controller.replan()
                elif self.path == "/api/execute":
                    requested_chunks = payload.get("max_chunks", 1)
                    if (
                        not isinstance(requested_chunks, int)
                        or isinstance(requested_chunks, bool)
                        or not 1 <= requested_chunks <= 24
                    ):
                        raise ValueError("max_chunks must be an int in 1..24")
                    result = controller.execute(
                        round_id=str(payload.get("round_id", "")),
                        confirmed=payload.get("confirmed") is True,
                        max_chunks=requested_chunks,
                    )
                elif self.path == "/api/home":
                    result = controller.return_home(
                        confirmed=payload.get("confirmed") is True,
                    )
                elif self.path == "/api/calibrate":
                    result = controller.calibrate_grounding(
                        confirmed=payload.get("confirmed") is True,
                    )
                    calibration_path = _save_live_anchor_calibration(
                        root, result["calibration"]
                    )
                    result["calibration"]["live_anchor_calibration"] = str(
                        calibration_path
                    )
                elif self.path == "/api/stop":
                    result = controller.stop()
                elif self.path == "/api/target-reference":
                    result = controller.set_target_reference(
                        str(payload.get("role", "")),
                        str(payload.get("view", "")),
                        np.asarray(payload.get("centroid_px"), dtype=np.float32),
                        radius_px=float(payload.get("radius_px", 18.0)),
                    )
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
            except Exception as exc:
                print(
                    f"request failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                code = _request_error_code(exc)
                status = (
                    HTTPStatus.BAD_REQUEST
                    if code == "invalid_request"
                    else HTTPStatus.CONFLICT
                )
                self.send_json(
                    {
                        **controller.status(),
                        "request_error_code": code,
                        "request_error": f"{type(exc).__name__}: {exc}",
                    },
                    status=status,
                )

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.client_address[0]} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    args = parse_args()
    guarded_command_jump = args.max_command_jump - args.command_guard_margin
    target_profile = (
        args.target_profile
        if args.target_profile is not None
        else args.output / "target_profile"
    )
    planner = PreparedSO101RealtimePlanner(
        PreparedPlannerConfig(
            bridge_checkpoint=args.bridge_checkpoint,
            model_checkpoint=args.model_checkpoint,
            calibration=args.calibration,
            secondary_cache=args.secondary_cache,
            device=args.device,
            max_first_command_jump=guarded_command_jump,
            max_prefix_step_jump=guarded_command_jump,
            target_profile=target_profile,
            memory_policy=args.memory_policy,
            require_live_column_calibration=True,
            control_mode=args.control_mode,
            ik_arrival_raw_steps=args.ik_arrival_raw_steps,
            max_column_servo_step_m=args.max_planar_step_m,
            max_column_z_step_m=args.max_z_step_m,
        )
    )
    # In-session calibrations are saved to <output>/live_anchor_calibration
    # .json; loading only the CLI-provided file silently discarded every one
    # of them on restart. Prefer whichever calibration is newest.
    calibration_candidates: list[Path] = []
    if args.live_anchor_calibration is not None:
        calibration_candidates.append(
            args.live_anchor_calibration.expanduser().resolve()
        )
    session_calibration = (
        args.output.expanduser().resolve() / "live_anchor_calibration.json"
    )
    if session_calibration.exists():
        calibration_candidates.append(session_calibration)
    if calibration_candidates:
        newest = max(calibration_candidates, key=lambda p: p.stat().st_mtime)
        _load_live_anchor_calibration(planner, newest)
        print(f"live anchor calibration loaded from: {newest}")
    home_state: np.ndarray | None = None
    if args.home_state_report is not None:
        home_state = _load_home_state_report(args.home_state_report)
        # The report snapshots the arm POSE at task start, but its gripper
        # value reflects whatever the calibration pulse left behind (closed).
        # A grasp is only physically possible from an open gripper, and no
        # phase opens it during APPROACH, so returning home must restore the
        # task-ready aperture.
        home_state[5] = max(
            float(home_state[5]), planner.gripper_calibration_open
        )
    replay = args.replay_session is not None
    if replay:
        sessions = load_paired_session(args.replay_session)
        io = ReplaySO101IO(sessions, start_index=args.replay_index)
    else:
        io = HardwareSO101IO(
            args,
            gripper_calibration_closed=planner.gripper_calibration_closed,
            gripper_calibration_open=planner.gripper_calibration_open,
        )
    writer = RoundArtifactWriter(
        args.output, render_mujoco=not args.no_mujoco_preview
    )
    controller = RealtimeMPCController(
        planner,
        io,
        artifact_writer=writer,
        execute_enabled=replay or args.execute_enabled,
        calibration_enabled=args.execute_enabled
        or args.gripper_calibration_enabled,
        control_hz=args.control_hz,
        max_plan_age_s=args.max_plan_age_s,
        max_state_drift=args.max_state_drift,
        max_target_drift_px=args.max_target_drift_px,
        max_command_jump=guarded_command_jump,
        minimum_expected_motion=args.minimum_expected_motion,
        minimum_observed_motion=args.minimum_observed_motion,
        home_state=home_state,
    )
    server: ThreadingHTTPServer | None = None
    try:
        controller.bootstrap()
        server = ThreadingHTTPServer(
            (args.host, args.http_port), handler_factory(controller, args.output)
        )
        server.daemon_threads = True
        mode = (
            "replay"
            if replay
            else (
                "hardware-armed"
                if args.execute_enabled
                else (
                    "hardware-calibration-only"
                    if args.gripper_calibration_enabled
                    else "hardware-read-only"
                )
            )
        )
        print(
            f"SO-101 real-time FCPG + LeWM ({mode}): "
            f"http://{args.host}:{args.http_port}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("SO-101 real-time server stopped by operator", flush=True)
    finally:
        if server is not None:
            server.server_close()
        controller.close()
        writer.close()


if __name__ == "__main__":
    main()
