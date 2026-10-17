#!/usr/bin/env python3
"""Evaluate a learned action-chunk policy on the custom Towel task."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import deque
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.custom_tasks.towel_wipe_expert import infer_towel_wipe_phase  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    A4EpisodeRuntime,
    _success,
    make_a4_episode_runtime,
)
from fcpg.lewm_counterfactual_runtime import FrameRecord  # noqa: E402
from fcpg.mujoco_camera import (  # noqa: E402
    get_camera_id,
    set_fixed_camera_pose_offset,
)
from fcpg.observations import select_policy_observation  # noqa: E402
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402


TASK = "towel-wipe-v0"
ANCHOR_COLORS = ((255, 210, 55), (42, 207, 239), (235, 80, 148))
GOAL_MODES = ("native", "hold-initial")
GROUNDING_MODES = ("checkpoint", "fcpg", "oracle")


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def record_to_mapping(record: FrameRecord) -> dict[str, np.ndarray]:
    return {
        "rgb": np.asarray(record.rgb, dtype=np.float32),
        "obs": np.asarray(record.proprio, dtype=np.float32),
        "keypoints_xyz": np.asarray(record.keypoints_xyz, dtype=np.float32),
        "q_nominal": np.asarray(record.q_nominal, dtype=np.float32),
        "q_true": np.asarray(record.q_true, dtype=np.float32),
    }


def policy_frame_record(
    record: FrameRecord,
    observation: np.ndarray,
    obs_mode: str,
    goal_override: np.ndarray | None = None,
) -> FrameRecord:
    """Attach the policy-safe observation selected by its checkpoint."""

    policy_observation = np.asarray(observation, dtype=np.float32)
    if goal_override is not None:
        override = np.asarray(goal_override, dtype=np.float32)
        if override.shape != (3,) or not np.isfinite(override).all():
            raise ValueError("goal_override must contain three finite values")
        policy_observation = policy_observation.copy()
        policy_observation[-3:] = override
    proprio = select_policy_observation(policy_observation, obs_mode).astype(
        np.float32, copy=False
    )
    return replace(record, proprio=proprio)


def history_batch(
    history: deque[FrameRecord],
    obs_horizon: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    records = list(history)
    if not records:
        raise ValueError("history must not be empty")
    records = [records[0]] * max(0, obs_horizon - len(records)) + records[-obs_horizon:]
    mappings = [record_to_mapping(record) for record in records]
    return {
        name: torch.from_numpy(
            np.stack([mapping[name] for mapping in mappings], axis=0)
        )
        .unsqueeze(0)
        .to(device)
        for name in mappings[0]
    }


def configure_grounding(
    model: Any,
    grounding_mode: str,
    fcpg_checkpoint: Path | None,
    device: torch.device,
) -> None:
    """Apply a same-checkpoint grounding intervention to a frozen policy."""

    if grounding_mode == "checkpoint":
        return
    if grounding_mode == "oracle":
        model.mode = "oracle"
        model.set_reliability_override("ones")
        model.eval()
        return
    if grounding_mode != "fcpg":
        raise ValueError(f"unknown grounding mode {grounding_mode!r}")
    if fcpg_checkpoint is None:
        raise ValueError("fcpg grounding requires an FCPG checkpoint")
    adapter, _ = load_fcpg_vit_adapter(fcpg_checkpoint, device)
    adapter.eval()
    adapter.requires_grad_(False)
    model.fcpg_adapter = adapter
    model.mode = "fcpg"
    # The frozen GeoProp policy only saw unit reliability during training.
    # Keep this intervention limited to the corrected sampling locations.
    model.set_reliability_override("ones")
    model.eval()


def append_phase_run(runs: list[str], phase: str) -> None:
    if phase.startswith("wipe_") and (not runs or runs[-1] != phase):
        runs.append(phase)


def _draw_frame(
    record: FrameRecord,
    *,
    step: int,
    phase: str,
    action: np.ndarray,
    half_cycles: int,
    success: bool,
) -> np.ndarray:
    rgb = np.clip(np.transpose(record.rgb, (1, 2, 0)) * 255.0, 0, 255).astype(
        np.uint8
    )
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 45), fill=(12, 18, 24, 195))
    draw.text(
        (7, 5),
        f"step {step:03d} | {phase} | half-cycles {half_cycles}/6 | success {int(success)}",
        fill=(255, 255, 255, 255),
        font=_font(10),
    )
    draw.text(
        (7, 25),
        "policy action " + " ".join(f"{value:+.2f}" for value in action),
        fill=(205, 218, 228, 255),
        font=_font(10),
    )
    for point, valid, color in zip(
        record.q_true, record.valid_true, ANCHOR_COLORS, strict=True
    ):
        if not bool(valid) or not np.isfinite(point).all():
            continue
        x, y = float(point[0]), float(point[1])
        draw.ellipse(
            (x - 3, y - 3, x + 3, y + 3),
            fill=(*color, 225),
            outline=(12, 18, 24, 255),
            width=1,
        )
    return np.asarray(image)


def rollout_episode(
    model: Any,
    checkpoint_args: dict[str, object],
    *,
    episode: int,
    task_instance: int,
    environment_seed: int,
    benchmark_seed: int,
    width: int,
    height: int,
    max_raw_steps: int,
    action_steps: int,
    stable_success_frames: int,
    device: torch.device,
    video_path: Path | None,
    fps: float,
    goal_mode: str,
    grounding_mode: str,
    camera_rotation_axis_world: np.ndarray,
    camera_rotation_degrees: float,
) -> dict[str, object]:
    runtime: A4EpisodeRuntime = make_a4_episode_runtime(
        TASK,
        task_instance,
        environment_seed,
        width=width,
        height=height,
        benchmark_seed=benchmark_seed,
        camera_name="towel_view",
    )
    camera_id = get_camera_id(runtime.env.model, runtime.camera_name)
    base_position = np.asarray(
        runtime.env.model.cam_pos[camera_id], dtype=np.float64
    ).copy()
    base_quaternion = np.asarray(
        runtime.env.model.cam_quat[camera_id], dtype=np.float64
    ).copy()
    set_fixed_camera_pose_offset(
        runtime.env.model,
        runtime.env.data,
        camera_id,
        base_position,
        base_quaternion,
        np.zeros(3, dtype=np.float64),
        camera_rotation_axis_world,
        camera_rotation_degrees,
    )
    obs_horizon = int(checkpoint_args.get("obs_horizon", 2))
    obs_mode = str(checkpoint_args.get("obs_mode", "robot"))
    if goal_mode not in GOAL_MODES:
        raise ValueError(f"unknown goal mode {goal_mode!r}")
    history: deque[FrameRecord] = deque(maxlen=obs_horizon)
    observation = np.asarray(runtime.observation, dtype=np.float32)
    initial_goal = observation[-3:].copy() if goal_mode == "hold-initial" else None
    history.append(
        policy_frame_record(
            runtime.record_current(observation),
            observation,
            obs_mode,
            initial_goal,
        )
    )
    writer = (
        imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        if video_path is not None
        else None
    )
    phase_runs: list[str] = []
    success_streak = 0
    first_success_step: int | None = None
    raw_step = 0
    terminated = False
    truncated = False
    last_action = np.zeros(4, dtype=np.float32)
    grounding_errors: list[float] = []
    try:
        while raw_step < max_raw_steps and success_streak < stable_success_frames:
            phase = infer_towel_wipe_phase(observation).value
            append_phase_run(phase_runs, phase)
            batch = history_batch(history, obs_horizon, device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model.predict_action_chunk(batch)
            predicted_points = output.grounding_points_px[0, -1].float().cpu().numpy()
            true_points = batch["q_true"][0, -1].float().cpu().numpy()
            finite = np.isfinite(predicted_points).all(axis=-1) & np.isfinite(
                true_points
            ).all(axis=-1)
            if bool(finite.any()):
                grounding_errors.extend(
                    np.linalg.norm(
                        predicted_points[finite] - true_points[finite], axis=-1
                    ).tolist()
                )
            action_chunk = output.action[0, :action_steps].float().cpu().numpy()
            for action in action_chunk:
                action = np.clip(
                    action,
                    runtime.env.action_space.low,
                    runtime.env.action_space.high,
                ).astype(np.float32)
                last_action = action
                if writer is not None:
                    writer.append_data(
                        _draw_frame(
                            history[-1],
                            step=raw_step,
                            phase=phase,
                            action=action,
                            half_cycles=int(runtime.env.wipe_half_cycles),
                            success=success_streak > 0,
                        )
                    )
                observation, _reward, terminated, truncated, info = runtime.env.step(
                    action
                )
                observation = np.asarray(observation, dtype=np.float32)
                raw_step += 1
                succeeded = _success(info, runtime.env)
                if succeeded and first_success_step is None:
                    first_success_step = raw_step
                success_streak = success_streak + 1 if succeeded else 0
                history.append(
                    policy_frame_record(
                        runtime.record_current(observation),
                        observation,
                        obs_mode,
                        initial_goal,
                    )
                )
                if (
                    raw_step >= max_raw_steps
                    or success_streak >= stable_success_frames
                    or terminated
                    or truncated
                ):
                    break
            if terminated or truncated:
                break
        final_phase = infer_towel_wipe_phase(observation).value
        append_phase_run(phase_runs, final_phase)
        if writer is not None:
            writer.append_data(
                _draw_frame(
                    history[-1],
                    step=raw_step,
                    phase=final_phase,
                    action=last_action,
                    half_cycles=int(runtime.env.wipe_half_cycles),
                    success=success_streak > 0,
                )
            )
    finally:
        if writer is not None:
            writer.close()
        runtime.env.close()
    return {
        "episode": episode,
        "task_instance": task_instance,
        "environment_seed": environment_seed,
        "steps": raw_step,
        "success": bool(success_streak >= stable_success_frames),
        "first_success_step": first_success_step,
        "wipe_half_cycles": int(len(phase_runs)),
        "wipe_order": phase_runs,
        "final_phase": final_phase,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "video": None if video_path is None else video_path.name,
        "goal_mode": goal_mode,
        "grounding_mode": grounding_mode,
        "camera_rotation_degrees": float(camera_rotation_degrees),
        "mean_grounding_epe_px": (
            float(np.mean(grounding_errors)) if grounding_errors else None
        ),
    }


def write_html(
    output_dir: Path,
    checkpoint: Path,
    rows: list[dict[str, object]],
) -> Path:
    cards: list[str] = []
    for row in rows:
        video = row.get("video")
        media = (
            f'<video controls muted loop playsinline preload="metadata"><source src="{html.escape(str(video), quote=True)}" type="video/mp4"></video>'
            if video
            else "<p>Video not rendered for this episode.</p>"
        )
        cards.append(
            f"<article><h2>Episode {row['episode']} | instance {row['task_instance']}</h2>{media}"
            f"<p><b>success:</b> {row['success']} | <b>steps:</b> {row['steps']} | "
            f"<b>half-cycles:</b> {row['wipe_half_cycles']}/6 | "
            f"<b>grounding EPE:</b> {float(row['mean_grounding_epe_px']):.2f} px | "
            f"<b>final phase:</b> {html.escape(str(row['final_phase']))}</p></article>"
        )
    success_rate = np.mean([bool(row["success"]) for row in rows]) if rows else 0.0
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Towel learned-policy rollout</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#edf1f3;color:#17212a;font:14px Arial,sans-serif}}header{{padding:22px max(20px,calc((100vw - 1180px)/2));background:#18252d;color:white}}header p{{color:#cbd5da}}main{{max-width:1180px;margin:auto;padding:20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}article{{background:white;border:1px solid #d4dce0;border-radius:6px;padding:12px}}h2{{font-size:17px}}video{{width:100%;background:#111}}</style></head><body><header><h1>Towel Learned-Policy Rollout</h1><p>Actual closed-loop RGB. The policy predicts an action chunk, executes the configured leading action(s), then observes again. Success: {100.0 * success_rate:.1f}%. Checkpoint: {html.escape(str(checkpoint))}</p></header><main>{''.join(cards)}</main></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--task-index-offset", type=int, default=32)
    parser.add_argument("--env-seed-base", type=int, default=732000)
    parser.add_argument("--benchmark-seed", type=int, default=40590)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--max-raw-steps", type=int, default=350)
    parser.add_argument("--action-steps", type=int, default=1)
    parser.add_argument("--stable-success-frames", type=int, default=3)
    parser.add_argument("--render-episodes", type=int, default=4)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="native")
    parser.add_argument(
        "--grounding-mode", choices=GROUNDING_MODES, default="checkpoint"
    )
    parser.add_argument("--fcpg-checkpoint", type=Path)
    parser.add_argument("--camera-rotation-degrees", type=float, default=0.0)
    parser.add_argument(
        "--camera-rotation-axis-world",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    positive = (
        args.episodes,
        args.width,
        args.height,
        args.max_raw_steps,
        args.action_steps,
        args.stable_success_frames,
    )
    if any(int(value) <= 0 for value in positive):
        parser.error("episode, image, horizon, and action arguments must be positive")
    if args.render_episodes < 0 or args.fps <= 0:
        parser.error("--render-episodes must be non-negative and --fps positive")

    checkpoint = args.checkpoint.expanduser().resolve()
    fcpg_checkpoint = (
        None
        if args.fcpg_checkpoint is None
        else args.fcpg_checkpoint.expanduser().resolve()
    )
    camera_axis = np.asarray(args.camera_rotation_axis_world, dtype=np.float64)
    if not np.isfinite(args.camera_rotation_degrees):
        parser.error("--camera-rotation-degrees must be finite")
    if not np.isfinite(camera_axis).all() or (
        abs(float(args.camera_rotation_degrees)) > 0.0
        and float(np.linalg.norm(camera_axis)) <= 1.0e-12
    ):
        parser.error("--camera-rotation-axis-world must be finite and non-zero")
    if args.grounding_mode == "fcpg" and fcpg_checkpoint is None:
        parser.error("--grounding-mode fcpg requires --fcpg-checkpoint")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint_args = load_action_chunk_policy(checkpoint, device)
    configure_grounding(model, args.grounding_mode, fcpg_checkpoint, device)
    rows: list[dict[str, object]] = []
    for episode in range(args.episodes):
        video_path = (
            output_dir / f"episode_{episode:03d}.mp4"
            if episode < args.render_episodes
            else None
        )
        row = rollout_episode(
            model,
            checkpoint_args,
            episode=episode,
            task_instance=args.task_index_offset + episode,
            environment_seed=args.env_seed_base + episode,
            benchmark_seed=args.benchmark_seed,
            width=args.width,
            height=args.height,
            max_raw_steps=args.max_raw_steps,
            action_steps=args.action_steps,
            stable_success_frames=args.stable_success_frames,
            device=device,
            video_path=video_path,
            fps=args.fps,
            goal_mode=args.goal_mode,
            grounding_mode=args.grounding_mode,
            camera_rotation_axis_world=camera_axis,
            camera_rotation_degrees=float(args.camera_rotation_degrees),
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    index = write_html(output_dir, checkpoint, rows)
    print(f"success_rate: {np.mean([bool(row['success']) for row in rows]):.6f}")
    print(f"output: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
