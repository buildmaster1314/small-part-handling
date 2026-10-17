#!/usr/bin/env python3
"""Render A4's seven full reset-to-success task trajectories as a gallery."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_task_factory import (  # noqa: E402
    CUSTOM_TASKS,
    DRAWER_RETRIEVE_TASK,
    TASKS,
    TOWEL_WIPE_TASK,
    make_a4_env_and_expert,
)


DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/visual_replays"
DEFAULT_INSTANCE = 0
DEFAULT_BENCHMARK_SEED = 30_103
DEFAULT_ENVIRONMENT_SEED_BASE = 620_000
DEFAULT_MAX_STEPS = 500
SUCCESS_STREAK = 3
DEFAULT_POST_SUCCESS_STEPS = 8
FINAL_FRAME_HOLD_COPIES = 6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--instance", type=int, default=DEFAULT_INSTANCE)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument(
        "--post-success-steps",
        type=int,
        default=DEFAULT_POST_SUCCESS_STEPS,
        help="real environment steps rendered after the stable success streak",
    )
    parser.add_argument("--task", choices=TASKS, action="append")
    return parser


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _success(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    value = float(info.get("success", 0.0))
    if not np.isfinite(value):
        raise ValueError("task success must be finite")
    return value > 0.0


def _bounded_action(env: Any, expert: Any, observation: np.ndarray) -> np.ndarray:
    action = np.asarray(expert.get_action(observation), dtype=np.float32)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    if action.shape != low.shape or high.shape != low.shape:
        raise ValueError("expert action and environment bounds do not align")
    if not np.isfinite(action).all():
        raise ValueError("expert returned a non-finite action")
    return np.clip(action, low, high).astype(np.float32, copy=False)


def _stage(task: str, expert: Any, observation: np.ndarray) -> str:
    get_phase = getattr(expert, "get_phase", None)
    if not callable(get_phase):
        return "active control"
    phase = get_phase(observation)
    return str(getattr(phase, "value", phase))


def _render_upright(env: Any, *, task: str) -> Image.Image:
    image = np.asarray(env.render(), dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rendered image must be HWC RGB, got {image.shape}")
    # MetaWorld's offscreen buffer needs the A3 row-origin correction. The
    # custom drawer camera already returns top-origin RGB.
    if task not in CUSTOM_TASKS:
        image = np.flip(image, axis=0)
    return Image.fromarray(image.copy(), mode="RGB")


def _annotated_frame(
    image: Image.Image,
    *,
    task: str,
    step: int,
    stage: str,
    reward: float,
    success: bool,
) -> Image.Image:
    header = 46
    canvas = Image.new("RGB", (image.width, image.height + header), (241, 244, 247))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"{task} | raw step {step}", fill=(20, 27, 36), font=_font(13))
    draw.text(
        (8, 25),
        f"stage {stage} | reward {reward:.2f} | success {int(success)}",
        fill=(74, 84, 96),
        font=_font(10),
    )
    return canvas


def _keyframe_indices(stages: list[str], count: int) -> list[int]:
    if count <= 0:
        raise ValueError("trajectory must contain at least one frame")
    seen_stages: set[str] = set()
    unique = []
    for index, stage in enumerate(stages):
        if stage in seen_stages:
            continue
        seen_stages.add(stage)
        unique.append(index)
    unique.append(count - 1)
    unique = list(dict.fromkeys(unique))
    if len(unique) <= 8:
        evenly = np.linspace(0, count - 1, min(8, count), dtype=np.int64).tolist()
        unique = sorted(set(unique).union(int(value) for value in evenly))
    if len(unique) > 8:
        selected = np.linspace(0, len(unique) - 1, 8, dtype=np.int64)
        unique = [unique[int(index)] for index in selected]
    return unique


def _contact_sheet(
    frames: list[Image.Image],
    steps: list[int],
    stages: list[str],
    rewards: list[float],
    successes: list[bool],
    *,
    task: str,
    instance: int,
) -> Image.Image:
    selected = _keyframe_indices(stages, len(frames))
    cell = 224
    label = 42
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    title = 62
    canvas = Image.new(
        "RGB",
        (columns * cell, title + rows * (cell + label)),
        (235, 239, 243),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), f"A4 full trajectory: {task}", fill=(18, 25, 34), font=_font(20))
    draw.text(
        (12, 36),
        f"instance {instance:02d} | reset through stable success or timeout",
        fill=(78, 88, 101),
        font=_font(12),
    )
    for tile_index, frame_index in enumerate(selected):
        row, column = divmod(tile_index, columns)
        x = column * cell
        y = title + row * (cell + label)
        tile = frames[frame_index].resize((cell, cell), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y + label))
        draw.text(
            (x + 7, y + 4),
            f"step {steps[frame_index]} | {stages[frame_index]}",
            fill=(24, 32, 42),
            font=_font(11),
        )
        draw.text(
            (x + 7, y + 22),
            f"reward {rewards[frame_index]:.2f} | success {int(successes[frame_index])}",
            fill=(80, 90, 103),
            font=_font(10),
        )
    return canvas


def render_task(
    task: str,
    *,
    task_index: int,
    instance: int,
    width: int,
    height: int,
    max_steps: int,
    frame_stride: int,
    gif_fps: int,
    post_success_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    if task == DRAWER_RETRIEVE_TASK:
        camera = "drawer_view"
    elif task == TOWEL_WIPE_TASK:
        camera = "towel_view"
    else:
        camera = "corner"
    environment_seed = DEFAULT_ENVIRONMENT_SEED_BASE + task_index * 10_000 + instance
    env, expert = make_a4_env_and_expert(
        task,
        camera,
        width,
        height,
        task_instance=instance,
        benchmark_seed=DEFAULT_BENCHMARK_SEED,
    )
    frames: list[Image.Image] = []
    gif_frames: list[Image.Image] = []
    steps: list[int] = []
    stages: list[str] = []
    rewards: list[float] = []
    successes: list[bool] = []
    first_success: int | None = None
    stable_success_step: int | None = None
    streak = 0
    stop_reason = "max_steps"
    try:
        observation, reset_info = env.reset(seed=environment_seed)
        observation = np.asarray(observation, dtype=np.float32)
        info = dict(reset_info) if isinstance(reset_info, dict) else {}
        reward = 0.0
        for step in range(max_steps + 1):
            current_success = _success(info)
            current_stage = _stage(task, expert, observation)
            image = _render_upright(env, task=task)
            frames.append(image)
            steps.append(step)
            stages.append(current_stage)
            rewards.append(float(reward))
            successes.append(current_success)
            if step % frame_stride == 0 or current_success:
                gif_frames.append(
                    _annotated_frame(
                        image,
                        task=task,
                        step=step,
                        stage=current_stage,
                        reward=float(reward),
                        success=current_success,
                    )
                )
            if current_success:
                first_success = step if first_success is None else first_success
                streak += 1
            else:
                streak = 0
            if streak >= SUCCESS_STREAK and stable_success_step is None:
                stable_success_step = step
            if (
                stable_success_step is not None
                and step >= stable_success_step + post_success_steps
            ):
                stop_reason = "post_success_tail"
                break
            if step == max_steps:
                break
            action = _bounded_action(env, expert, observation)
            observation, reward, terminated, truncated, step_info = env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            info = dict(step_info) if isinstance(step_info, dict) else {}
            if terminated or truncated:
                stop_reason = "terminated" if terminated else "truncated"
                break
    finally:
        env.close()

    if not gif_frames:
        raise RuntimeError(f"{task} produced no rendered frames")
    slug = task.replace("-", "_")
    gif_name = f"{slug}_instance{instance:02d}_full.gif"
    sheet_name = f"{slug}_instance{instance:02d}_timeline.png"
    duration = max(20, int(round(1000 / gif_fps)))
    held_gif_frames = [*gif_frames, *([gif_frames[-1]] * FINAL_FRAME_HOLD_COPIES)]
    held_gif_frames[0].save(
        output_dir / gif_name,
        format="GIF",
        save_all=True,
        append_images=held_gif_frames[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    _contact_sheet(
        frames,
        steps,
        stages,
        rewards,
        successes,
        task=task,
        instance=instance,
    ).save(output_dir / sheet_name, format="PNG", optimize=True)
    return {
        "task": task,
        "task_index": task_index,
        "instance": instance,
        "camera": camera,
        "environment_seed": environment_seed,
        "steps_executed": steps[-1],
        "first_success_step": first_success,
        "stable_success_step": stable_success_step,
        "post_success_steps": post_success_steps,
        "success_streak_reached": stable_success_step is not None,
        "stop_reason": stop_reason,
        "stages": list(dict.fromkeys(stages)),
        "gif": gif_name,
        "timeline": sheet_name,
    }


def _write_gallery(output_dir: Path, entries: list[dict[str, Any]]) -> Path:
    sections: list[str] = []
    for entry in entries:
        task = html.escape(str(entry["task"]))
        stages = " -> ".join(html.escape(str(value)) for value in entry["stages"])
        status = "success" if entry["success_streak_reached"] else entry["stop_reason"]
        sections.append(
            f"""
      <section>
        <div class="heading">
          <div><h2>{task}</h2><p class="stages">{stages}</p></div>
          <p>instance {entry['instance']:02d} &middot; {entry['camera']} &middot; step {entry['steps_executed']} &middot; {status}</p>
        </div>
        <img class="animation" src="{html.escape(entry['gif'])}" alt="{task} full trajectory">
        <a href="{html.escape(entry['timeline'])}">Open full-resolution trajectory sheet</a>
      </section>"""
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A4 seven-task full trajectory preview</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #eef1f4; color: #141a21; letter-spacing: 0; }}
    header {{ padding: 24px max(24px, calc((100vw - 1120px) / 2)); background: #18222d; color: white; }}
    h1, h2, p {{ margin: 0; }}
    header p {{ margin-top: 8px; color: #bec9d2; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 26px 24px 56px; }}
    section {{ padding: 22px 0 34px; border-bottom: 1px solid #c8d0d8; }}
    .heading {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; margin-bottom: 13px; }}
    .heading h2 {{ font-size: 20px; }}
    .heading > p, .stages {{ color: #65717d; font-size: 13px; margin-top: 4px; }}
    .animation {{ display: block; width: min(100%, 720px); height: auto; background: #dfe4e9; }}
    a {{ display: inline-block; margin-top: 12px; color: #155f91; font-weight: 650; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 680px) {{ .heading {{ display: block; }} .heading > p {{ margin-top: 7px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>LeWM A4: seven-task full trajectories</h1>
    <p>Reset-to-success expert rollouts. These are task/data semantics previews, not LeWM predictions or local H8 branches.</p>
  </header>
  <main>{''.join(sections)}</main>
</body>
</html>
"""
    destination = output_dir / "index.html"
    destination.write_text(page, encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.max_steps <= 0:
        raise ValueError("image dimensions and max steps must be positive")
    if args.frame_stride <= 0 or args.gif_fps <= 0:
        raise ValueError("frame stride and GIF FPS must be positive")
    if args.post_success_steps < 0:
        raise ValueError("post-success steps must be non-negative")
    tasks = tuple(args.task) if args.task else TASKS
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for task in tasks:
        entry = render_task(
            task,
            task_index=TASKS.index(task),
            instance=args.instance,
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
            frame_stride=args.frame_stride,
            gif_fps=args.gif_fps,
            post_success_steps=args.post_success_steps,
            output_dir=output_dir,
        )
        entries.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gallery = _write_gallery(output_dir, entries)
    print(json.dumps({"gallery": str(gallery), "rendered": len(entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
