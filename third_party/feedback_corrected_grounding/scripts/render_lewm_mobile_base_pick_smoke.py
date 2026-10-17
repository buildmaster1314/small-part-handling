#!/usr/bin/env python3
"""Render a fixed-base and slowly moving-base Pick episode side by side."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_task_factory import make_a4_env, make_a4_expert  # noqa: E402
from fcpg.lewm_mobile_base import (  # noqa: E402
    PlanarBaseTracker,
    apply_planar_base_translation,
)


DEFAULT_OUTPUT = ROOT / "outputs/lewm_mobile_base/pick_smoke"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-instance", type=int, default=0)
    parser.add_argument("--benchmark-seed", type=int, default=20260722)
    parser.add_argument("--environment-seed", type=int, default=20260723)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--success-tail", type=int, default=12)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    return parser.parse_args()


def _command(step: int) -> np.ndarray:
    """A visible but slow world-frame base translation in metres per raw step."""

    return np.asarray(
        [
            5.0e-4,
            3.0e-4 * np.sin(2.0 * np.pi * float(step) / 70.0),
        ],
        dtype=np.float64,
    )


def _slip(step: int, command: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            0.18 * command[0],
            -0.12 * command[1] + 5.0e-5 * np.cos(2.0 * np.pi * step / 31.0),
        ],
        dtype=np.float64,
    )


def _odometry_error(step: int, command: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            -0.04 * command[0],
            3.0e-5 * np.sin(2.0 * np.pi * step / 43.0),
        ],
        dtype=np.float64,
    )


def _render_rgb(env: Any) -> np.ndarray:
    frame = np.asarray(env.render())
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("environment render must return an RGB image")
    return np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)


def _run_episode(args: argparse.Namespace, *, moving: bool) -> dict[str, Any]:
    env = make_a4_env(
        "pick-place-v3",
        "corner",
        int(args.width),
        int(args.height),
        task_instance=int(args.task_instance),
        benchmark_seed=int(args.benchmark_seed),
    )
    expert = make_a4_expert("pick-place-v3")
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    try:
        observation, _ = env.reset(seed=int(args.environment_seed))
        tracker = PlanarBaseTracker()
        first_success: int | None = None
        for step in range(int(args.max_steps)):
            if moving:
                command = _command(step)
                transition = tracker.step(
                    command,
                    slip_xy=_slip(step, command),
                    odometry_error_xy=_odometry_error(step, command),
                )
                apply_planar_base_translation(env, transition.true_delta_xy)
                observation = np.asarray(env._get_obs(), dtype=np.float32)
            else:
                transition = tracker.step((0.0, 0.0))
            action = np.asarray(expert.get_action(observation), dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            frames.append(_render_rgb(env))
            success = bool(info.get("success", False))
            if success and first_success is None:
                first_success = step
            rows.append(
                {
                    "step": step,
                    "success": success,
                    "reward": float(reward),
                    "true_base_xy": transition.true_position_xy.tolist(),
                    "odometry_base_xy": transition.odometry_position_xy.tolist(),
                    "command_xy": transition.command_xy.tolist(),
                }
            )
            if terminated or truncated:
                break
            if first_success is not None and step >= first_success + int(args.success_tail):
                break
        return {
            "moving": moving,
            "frames": frames,
            "rows": rows,
            "success": bool(any(row["success"] for row in rows)),
            "first_success_step": first_success,
            "final_true_base_xy": tracker.true_position_xy.tolist(),
            "final_odometry_base_xy": tracker.odometry_position_xy.tolist(),
        }
    finally:
        env.close()


def _annotated_panel(
    fixed: dict[str, Any], moving: dict[str, Any], frame_index: int
) -> Image.Image:
    left_index = min(frame_index, len(fixed["frames"]) - 1)
    right_index = min(frame_index, len(moving["frames"]) - 1)
    left = Image.fromarray(fixed["frames"][left_index]).convert("RGB")
    right = Image.fromarray(moving["frames"][right_index]).convert("RGB")
    margin = 42
    panel = Image.new("RGB", (left.width + right.width, left.height + margin), "white")
    panel.paste(left, (0, margin))
    panel.paste(right, (left.width, margin))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), f"Fixed base | step {left_index}", fill="black")
    row = moving["rows"][right_index]
    true_xy = np.asarray(row["true_base_xy"]) * 100.0
    odom_xy = np.asarray(row["odometry_base_xy"]) * 100.0
    draw.text(
        (left.width + 8, 5),
        f"Moving base | step {right_index}",
        fill="black",
    )
    draw.text(
        (left.width + 8, 21),
        f"true=({true_xy[0]:+.1f},{true_xy[1]:+.1f}) cm  "
        f"odom=({odom_xy[0]:+.1f},{odom_xy[1]:+.1f}) cm",
        fill="black",
    )
    return panel


def main() -> None:
    args = _parser()
    if args.max_steps <= 0 or args.success_tail < 0 or args.fps <= 0.0:
        raise ValueError("render limits and fps must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixed = _run_episode(args, moving=False)
    moving = _run_episode(args, moving=True)
    frame_count = max(len(fixed["frames"]), len(moving["frames"]))
    panels = [_annotated_panel(fixed, moving, index) for index in range(frame_count)]
    gif_name = "fixed_vs_moving_base_pick.gif"
    imageio.mimsave(
        output / gif_name,
        [np.asarray(panel) for panel in panels],
        duration=1000.0 / float(args.fps),
        loop=0,
    )
    metrics = {
        "protocol": "mobile_base_pick_visual_smoke_v1",
        "task": "pick-place-v3",
        "fixed": {key: value for key, value in fixed.items() if key != "frames"},
        "moving": {key: value for key, value in moving.items() if key != "frames"},
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    page = f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>Moving-base Pick smoke</title>
<style>body{{font:16px system-ui;margin:24px;background:#f4f4f1;color:#181818}}img{{max-width:100%;border:1px solid #aaa}}code{{background:#fff;padding:2px 5px}}</style></head>
<body><h1>Fixed vs moving robot base</h1>
<p>右侧底座每个 raw step 都移动。<code>true</code> 是物理位移，<code>odom</code> 是所有可部署方法可见的带误差里程计。</p>
<img src=\"{gif_name}\" alt=\"fixed and moving base Pick episodes\">
<p>Fixed success: {fixed['success']} at {fixed['first_success_step']}; moving success: {moving['success']} at {moving['first_success_step']}.</p>
</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
