#!/usr/bin/env python3
"""Render a goal-aware towel replay from frozen LeWM plus a sidecar.

The left panel is the actual rollout.  On the right, the orange dashed path is
the frozen LeWM forecast and the blue path is the goal-relative sidecar
forecast.  Every five real raw steps the current frame pauses while the next
40 raw steps are played from the H1-H8 prediction. The diagnostic sidecar gets
an external 2-D goal coordinate, but no goal marker is drawn into either image.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_task_factory import CUSTOM_TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4 import HORIZON, RAW_STEPS_PER_MODEL_STEP, TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import a4_camera_name, load_frozen_a4_state_encoder  # noqa: E402
from fcpg.lewm_camera_action_joint_a4 import load_camera_action_joint_checkpoint  # noqa: E402
from fcpg.lewm_goal_relative_sidecar import load_sidecar_checkpoint, towel_goal_projection  # noqa: E402
from fcpg.lewm_orbit_camera import OrbitCameraView  # noqa: E402
from fcpg.lewm_counterfactual_runtime import corrected_points  # noqa: E402
from scripts import render_lewm_action_branch_a4_accuracy_gallery as legacy  # noqa: E402
from scripts import render_lewm_action_branch_a4_trajectory_comparison as trajectory  # noqa: E402
from scripts import render_lewm_action_prefix_joint_a4_accuracy_gallery as gallery  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_BASE_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_phaseaware_a4c_v4_seed40571_10k/"
    "lewm_camera_action_joint_latest.pt"
)
DEFAULT_SIDECAR = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "goal_relative_towel_sidecar_seed40572_2k/"
    "goal_relative_sidecar_latest.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "goal_relative_towel_sidecar_seed40572_2k_nominal_clock03_full"
)
ACTUAL_COLOR = (31, 151, 91)
BASE_COLOR = (224, 138, 44)
SIDECAR_COLOR = (35, 105, 191)


def _device(value: str) -> torch.device:
    result = torch.device(value)
    if result.type != "cuda" or result.index not in (None, 0):
        raise ValueError("goal replay permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--sidecar-checkpoint", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--orbit-clock-hour", type=int, default=3)
    parser.add_argument("--orbit-elevation-degrees", type=float, default=35.0)
    parser.add_argument("--orbit-radius-m", type=float, default=1.1)
    parser.add_argument(
        "--grounding-mode",
        choices=("nominal", "fcpg"),
        default="nominal",
        help="orbit replay uses nominal by default to match the frozen training distribution",
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-raw-steps", type=int, default=500)
    parser.add_argument("--post-success-steps", type=int, default=40)
    return parser


def _draw_path(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int], width: int = 3, dashed: bool = False) -> None:
    values = np.asarray(points, dtype=np.float32)
    if len(values) < 2:
        return
    rows = [tuple(row) for row in values]
    if not dashed:
        draw.line(rows, fill=color, width=width, joint="curve")
        return
    for start, stop in zip(rows[:-1], rows[1:]):
        a = np.asarray(start, dtype=np.float32)
        b = np.asarray(stop, dtype=np.float32)
        for begin in np.linspace(0.0, 1.0, 6)[:-1:2]:
            end = min(float(begin) + 0.18, 1.0)
            p = tuple((a * (1.0 - begin) + b * begin).tolist())
            q = tuple((a * (1.0 - end) + b * end).tolist())
            draw.line([p, q], fill=color, width=width)


def _draw_anchor_links(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    width: int = 2,
) -> None:
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (3, 2) or not np.isfinite(values).all():
        return
    # Anchor order is left fingertip, right fingertip, and tool center.
    draw.line(
        [tuple(values[0]), tuple(values[2]), tuple(values[1])],
        fill=color,
        width=width,
        joint="curve",
    )


def _render_frame(
    left_frame: Image.Image,
    right_frame: Image.Image,
    truth_history: np.ndarray,
    base_path: np.ndarray,
    sidecar_path: np.ndarray,
    *,
    actual_step: int,
    prediction_origin: int,
    prediction_target: int,
    end_step: int,
    phase: str,
    success: bool,
) -> Image.Image:
    left = left_frame.convert("RGB").copy()
    right = right_frame.convert("L").convert("RGB")
    right = Image.blend(right, Image.new("RGB", right.size, (232, 235, 237)), 0.28)
    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    _draw_path(left_draw, np.asarray(truth_history).mean(axis=1), ACTUAL_COLOR)
    _draw_path(right_draw, np.asarray(base_path).mean(axis=1), BASE_COLOR, width=3, dashed=True)
    _draw_path(right_draw, np.asarray(sidecar_path).mean(axis=1), SIDECAR_COLOR, width=4)
    for point in np.asarray(truth_history[-1]):
        legacy._draw_marker(left_draw, point, fill=ACTUAL_COLOR, radius=4, diamond=False)
    _draw_anchor_links(left_draw, truth_history[-1], ACTUAL_COLOR)
    for point in np.asarray(base_path[-1]):
        legacy._draw_marker(right_draw, point, fill=BASE_COLOR, radius=4, diamond=False)
    _draw_anchor_links(right_draw, base_path[-1], BASE_COLOR)
    for point in np.asarray(sidecar_path[-1]):
        legacy._draw_marker(right_draw, point, fill=SIDECAR_COLOR, radius=5, diamond=True)
    _draw_anchor_links(right_draw, sidecar_path[-1], SIDECAR_COLOR, width=3)
    width, height = left.size
    margin, gap, header, footer = 12, 12, 74, 58
    canvas = Image.new("RGB", (margin * 2 + width * 2 + gap, header + height + footer), (238, 241, 243))
    right_x = margin + width + gap
    canvas.paste(left, (margin, header))
    canvas.paste(right, (right_x, header))
    draw = ImageDraw.Draw(canvas)
    font17 = legacy._safe_font(17)
    font12 = legacy._safe_font(12)
    font11 = legacy._safe_font(11)
    draw.text((margin, 7), "Towel wipe: goal-relative LeWM sidecar", fill=(23, 32, 40), font=font17)
    draw.text((margin, 38), "ACTUAL ROLLOUT", fill=ACTUAL_COLOR, font=font12)
    draw.text((right_x, 32), "FORECAST", fill=SIDECAR_COLOR, font=font12)
    draw.text(
        (right_x, 50),
        "orange base | blue sidecar",
        fill=(73, 91, 104),
        font=font11,
    )
    base_epe = float(np.linalg.norm(base_path[-1] - truth_history[-1], axis=-1).mean()) if len(truth_history) else float("nan")
    side_epe = float(np.linalg.norm(sidecar_path[-1] - truth_history[-1], axis=-1).mean()) if len(truth_history) else float("nan")
    draw.text(
        (margin, header + height + 7),
        f"{phase} | actual {actual_step:03d}/{end_step:03d} | forecast {prediction_origin}->{prediction_target} | success {int(success)}",
        fill=(55, 66, 75),
        font=font11,
    )
    draw.text(
        (margin, header + height + 27),
        f"forecast endpoint EPE: base {base_epe:.1f}px | sidecar {side_epe:.1f}px",
        fill=(55, 66, 75),
        font=font11,
    )
    return canvas


def _cycle_frames(
    episode: Mapping[str, Any],
    starts: np.ndarray,
    corrected: np.ndarray,
    base_predictions: np.ndarray,
    sidecar_predictions: np.ndarray,
) -> Iterable[Image.Image]:
    frames = episode["frames"]
    truth = np.stack([row.q_true for row in episode["records"]]).astype(np.float32)
    successes = episode["successes"]
    end_step = len(frames) - 1
    previous_base = previous_side = None
    previous_step = None
    for i, start_value in enumerate(np.asarray(starts, dtype=np.int64)):
        current_step = int(start_value)
        if previous_base is not None and previous_side is not None and previous_step is not None:
            for actual_step in range(previous_step + 1, current_step + 1):
                yield _render_frame(
                    frames[actual_step], frames[previous_step], truth[: actual_step + 1],
                    previous_base, previous_side,
                    actual_step=actual_step, prediction_origin=previous_step,
                    prediction_target=previous_step + 40, end_step=end_step,
                    phase="ACTUAL ADVANCE", success=bool(successes[actual_step]),
                )
        base_dense = gallery._interpolate_prediction_window(corrected[i], base_predictions[i])
        side_dense = gallery._interpolate_prediction_window(corrected[i], sidecar_predictions[i])
        for offset in range(1, len(side_dense)):
            yield _render_frame(
                frames[current_step], frames[current_step], truth[: current_step + 1],
                base_dense[: offset + 1], side_dense[: offset + 1],
                actual_step=current_step, prediction_origin=current_step,
                prediction_target=current_step + offset, end_step=end_step,
                phase="GOAL-AWARE FORECAST", success=bool(successes[current_step]),
            )
        previous_base, previous_side, previous_step = base_dense, side_dense, current_step
    if previous_base is not None and previous_side is not None and previous_step is not None:
        for actual_step in range(previous_step + 1, end_step + 1):
            yield _render_frame(
                frames[actual_step], frames[previous_step], truth[: actual_step + 1],
                previous_base, previous_side,
                actual_step=actual_step, prediction_origin=previous_step,
                prediction_target=previous_step + 40, end_step=end_step,
                phase="FINAL ACTUAL CATCH-UP", success=bool(successes[actual_step]),
            )


def _horizon_metrics(episode: Mapping[str, Any], starts: np.ndarray, predictions: np.ndarray) -> dict[str, dict[str, float]]:
    truth = np.stack([row.q_true for row in episode["records"]]).astype(np.float32)
    result: dict[str, dict[str, float]] = {}
    for horizon in range(1, HORIZON + 1):
        target = truth[np.asarray(starts) + horizon * RAW_STEPS_PER_MODEL_STEP]
        estimate = predictions[:, horizon - 1]
        anchor = np.linalg.norm(estimate - target, axis=-1)
        center = np.linalg.norm(estimate.mean(axis=1) - target.mean(axis=1), axis=-1)
        result[f"H{horizon}"] = {
            "base_anchor_mean_epe_px": float(anchor.mean()),
            "base_center_mean_epe_px": float(center.mean()),
        }
    return result


def _apply_sidecar(
    model: Any,
    base: np.ndarray,
    current: np.ndarray,
    goals: np.ndarray,
    task_index: int,
    device: torch.device,
    *,
    future_action: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(base), 256):
            stop = min(len(base), start + 256)
            b = torch.from_numpy(base[start:stop]).to(device)
            c = torch.from_numpy(current[start:stop]).to(device)
            g = torch.from_numpy(goals[start:stop]).to(device)
            t = torch.full((stop - start,), int(task_index), dtype=torch.long, device=device)
            kwargs = {"image_size": (224, 224)}
            if future_action is not None:
                kwargs["future_action"] = torch.from_numpy(
                    future_action[start:stop]
                ).to(device)
            output.append(model(b, c, g, t, **kwargs).float().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _html(entry: Mapping[str, Any]) -> str:
    rows = []
    for h in range(1, HORIZON + 1):
        m = entry["horizon_metrics"][f"H{h}"]
        rows.append(
            f"<tr><td>H{h}</td><td>{m['base_anchor_mean_epe_px']:.2f}</td><td>{m['base_center_mean_epe_px']:.2f}</td><td>{m['sidecar_anchor_mean_epe_px']:.2f}</td><td>{m['sidecar_center_mean_epe_px']:.2f}</td></tr>"
        )
    h8 = entry["horizon_metrics"]["H8"]
    h8_delta = float(h8["sidecar_center_mean_epe_px"]) - float(
        h8["base_center_mean_epe_px"]
    )
    endpoint_delta = float(entry["sidecar_endpoint_epe"]) - float(
        entry["base_endpoint_epe"]
    )
    verdict_class = "pass" if h8_delta < 0.0 and endpoint_delta < 0.0 else "fail"
    verdict = (
        "This sidecar improves both H8 and the stitched endpoint."
        if verdict_class == "pass"
        else "This sidecar does not improve the held-out full-task replay."
    )
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Goal-relative LeWM towel replay</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#edf1f3;color:#17212a;font-family:Arial,sans-serif}}header{{background:#17252e;color:white;padding:24px max(20px,calc((100vw - 1000px)/2))}}header p{{color:#c4cfd5;line-height:1.5}}main{{max-width:1000px;margin:auto;padding:22px 20px 60px}}.note,.verdict{{background:white;border-left:5px solid #2976a8;padding:16px;line-height:1.55}}.verdict{{margin-top:12px}}.verdict.pass{{border-color:#24815d}}.verdict.fail{{border-color:#b24b3f}}video,img{{display:block;width:100%;height:auto;background:#20272c;border:1px solid #bbc5cb;margin-top:14px}}table{{width:100%;border-collapse:collapse;background:white;margin-top:16px}}th,td{{padding:9px;border-bottom:1px solid #dde3e6;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{font-size:12px;color:#60707a}}.small{{color:#60707a;font-size:13px;line-height:1.5}}.legend{{color:#52616b;line-height:1.6}}</style></head><body><header><h1>Goal-Relative LeWM: Towel Endpoint Replay</h1><p>Frozen seven-task camera-action LeWM step {entry['parent_step']} + fixed 2K goal sidecar. Clock-{entry['clock']} orbit view, {html.escape(entry['grounding_mode'])} current grounding.</p></header><main><div class='note'><b>How to read it:</b> left is the actual rollout. On the right, orange dashed is the frozen LeWM forecast and blue is the goal-relative sidecar forecast. Every five actual raw steps pause while the next 40 predicted steps play. No target marker is drawn into either image. The sidecar receives an external phase-derived 2-D goal coordinate; the frozen parent LeWM does not receive that coordinate, future RGB, or q_true.</div><div class='verdict {verdict_class}'><b>Diagnostic verdict:</b> {verdict} H8 center EPE delta: {h8_delta:+.2f} px; stitched endpoint delta: {endpoint_delta:+.2f} px. Negative is better.</div><video controls muted loop playsinline preload='metadata'><source src='{html.escape(entry['video'], quote=True)}' type='video/mp4'></video><details><summary>Open sampled GIF ({entry['gif_frames']} frames)</summary><img src='{html.escape(entry['gif'], quote=True)}' alt='goal relative towel replay'></details><p class='legend'>Orange dashed = frozen parent LeWM. Blue = goal-relative residual sidecar. The sidecar goal coordinate is derived from the recorded towel phase for development diagnosis; it is not a learned visual goal and is deliberately not shown as an image marker.</p><table><thead><tr><th>Horizon</th><th>Base anchor EPE</th><th>Base center EPE</th><th>Sidecar anchor EPE</th><th>Sidecar center EPE</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p class='small'>Base stitched endpoint center EPE: {entry['base_endpoint_epe']:.2f} px. Sidecar stitched endpoint center EPE: {entry['sidecar_endpoint_epe']:.2f} px. This is a frozen-model development visualization, not a rollout-success claim.</p></main></body></html>"""


def main() -> int:
    args = _parser().parse_args()
    device = _device(args.device)
    torch.cuda.set_device(device)
    if not 1 <= int(args.orbit_clock_hour) <= 12:
        raise ValueError("orbit clock must be in [1,12]")
    contract = legacy._safe_read_json(args.config)
    legacy.validate_a4_contract(contract)
    task = "towel-wipe-v0"
    task_index = TASKS.index(task)
    instance = legacy._first_test_instances(contract)[task]
    collection = contract["collection"]
    orbit = OrbitCameraView(
        clock_hour=int(args.orbit_clock_hour),
        elevation_degrees=float(args.orbit_elevation_degrees),
        radius_m=float(args.orbit_radius_m),
        base_camera_name=a4_camera_name(task),
    )
    environment_seed = legacy._environment_seed(contract, task=task, task_instance=instance)
    episode = trajectory._collect_full_episode(
        task,
        task_instance=instance,
        environment_seed=environment_seed,
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
        max_raw_steps=int(args.max_raw_steps),
        post_success_steps=int(args.post_success_steps),
        camera_name=a4_camera_name(task),
        orbit_camera_view=orbit,
    )
    episode["camera_label"] = orbit.label
    episode["orbit_camera_view"] = orbit
    starts = gallery._decision_steps(len(episode["actions"]))
    selected = [episode["records"][int(i)] for i in starts]
    if args.grounding_mode == "nominal":
        corrected = np.stack([record.q_nominal for record in selected]).astype(
            np.float32
        )
    else:
        fcpg = load_frozen_a4_state_encoder(
            args.base_checkpoint.expanduser().resolve(),
            args.fcpg_checkpoint.expanduser().resolve(),
            device=device,
        )
        try:
            corrected = corrected_points(selected, fcpg).astype(np.float32)
        finally:
            del fcpg
            legacy._clear_runtime_memory(device)
    payload = torch.load(args.parent_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    parent = load_camera_action_joint_checkpoint(payload).to(device).eval()
    starts, base_predictions = gallery._predict_episode_windows(
        parent, episode, corrected, task_index=task_index, device=device, orbit_view=orbit
    )
    side_payload = torch.load(args.sidecar_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    sidecar = load_sidecar_checkpoint(side_payload).to(device).eval()
    sidecar_protocol = str(side_payload.get("protocol", ""))
    sidecar_actions = None
    if sidecar_protocol == "goal_relative_action_sidecar_v1":
        raw_action_prefix = gallery._chunk_future_actions(episode["actions"], starts)
        transformed_action_prefix = gallery._transform_future_actions_for_model(
            parent,
            torch.from_numpy(raw_action_prefix).to(device),
            episode,
        )
        sidecar_actions = transformed_action_prefix.float().cpu().numpy()
    goals = np.asarray(
        [
            towel_goal_projection(
                instance,
                episode["stages"][int(step)],
                episode["camera_intrinsic"],
                episode["camera_extrinsic_cam_from_world"],
            )[0]
            for step in starts
        ],
        dtype=np.float32,
    )
    sidecar_predictions = _apply_sidecar(
        sidecar,
        base_predictions,
        corrected,
        goals,
        task_index,
        device,
        future_action=sidecar_actions,
    )
    truth = np.stack([row.q_true for row in episode["records"]]).astype(np.float32)
    metrics = _horizon_metrics(episode, starts, base_predictions)
    for h in range(1, HORIZON + 1):
        target = truth[starts + h * RAW_STEPS_PER_MODEL_STEP]
        for name, prediction in (("base", base_predictions), ("sidecar", sidecar_predictions)):
            error = np.linalg.norm(prediction[:, h - 1] - target, axis=-1)
            center = np.linalg.norm(prediction[:, h - 1].mean(1) - target.mean(1), axis=-1)
            metrics[f"H{h}"][f"{name}_anchor_mean_epe_px"] = float(error.mean())
            metrics[f"H{h}"][f"{name}_center_mean_epe_px"] = float(center.mean())
    sidecar_dense, _, _ = gallery._stitch_max_lookahead_predictions(starts, sidecar_predictions, corrected[0], raw_frame_count=len(episode["frames"]))
    base_dense, _, _ = gallery._stitch_max_lookahead_predictions(starts, base_predictions, corrected[0], raw_frame_count=len(episode["frames"]))
    base_endpoint = np.linalg.norm(base_dense[-1].mean(0) - truth[-1].mean(0))
    side_endpoint = np.linalg.norm(sidecar_dense[-1].mean(0) - truth[-1].mean(0))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected = gallery._prediction_cycle_frame_count(starts, end_step=len(episode["frames"]) - 1)
    video, gif, count, gif_count = gallery._save_prediction_cycle_media(
        output,
        task=task,
        frames=_cycle_frames(episode, starts, corrected, base_predictions, sidecar_predictions),
        fps=float(args.fps),
        expected_frame_count=expected,
        gif_max_frames=240,
    )
    entry = {
        "video": video,
        "gif": gif,
        "gif_frames": int(gif_count),
        "parent_step": int(payload["step"]),
        "clock": int(args.orbit_clock_hour),
        "grounding_mode": str(args.grounding_mode),
        "sidecar_protocol": sidecar_protocol,
        "horizon_metrics": metrics,
        "base_endpoint_epe": float(base_endpoint),
        "sidecar_endpoint_epe": float(side_endpoint),
        "raw_frames": len(episode["frames"]),
        "rendered_frames": int(count),
        "stable_success_step": int(episode["stable_success_step"]),
    }
    (output / "index.html").write_text(_html(entry), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(entry, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"protocol": "goal_relative_towel_replay_v1", **entry}, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output / "index.html"), **entry}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
