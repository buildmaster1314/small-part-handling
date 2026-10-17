#!/usr/bin/env python3
"""Render the exact A3 matched-state action branches as images and GIFs."""

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

from fcpg.lewm_action_branch_a3 import build_diverse_action_branch_bank  # noqa: E402
from fcpg.lewm_action_branch_a3_collection import (  # noqa: E402
    SEMANTIC_BRANCHES,
    TASKS,
    _a3_candidate_seeds,
)
from fcpg.lewm_action_branch_a3_physical import (  # noqa: E402
    validate_reconstructed_source_actions,
)
from fcpg.lewm_action_branch_collection import (  # noqa: E402
    FRAME_SKIP,
    make_metaworld_episode_runtime,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    capture_branch_snapshot,
    restore_branch_snapshot,
)
from fcpg.lewm_counterfactual_v2_runtime import (  # noqa: E402
    _close_episode_env_safely,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
DEFAULT_INPUT = ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_confirmation.npz"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a3/visual_replays"
ANCHOR_COLORS = ((236, 78, 68), (55, 176, 112), (57, 126, 214))
CONTACT_STEPS = (0, 10, 20, 30, 40)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--instance", type=int)
    parser.add_argument("--hazard", type=int)
    parser.add_argument("--gif-fps", type=int, default=10)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A3 config root must be an object")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _select_row(
    arrays: dict[str, np.ndarray],
    *,
    task: str,
    split: str,
    instance: int | None,
    hazard: int | None,
) -> int:
    task_index = TASKS.index(task)
    selected = (np.asarray(arrays["task_index"]) == task_index) & (
        np.asarray(arrays["split"]).astype(str) == split
    )
    if instance is not None:
        selected &= np.asarray(arrays["task_instance_index"]) == int(instance)
    if hazard is not None:
        selected &= np.asarray(arrays["hazard_id"]) == int(hazard)
    rows = np.flatnonzero(selected)
    if rows.size == 0:
        raise ValueError(
            f"no A3 row matches task={task}, split={split}, "
            f"instance={instance}, hazard={hazard}"
        )
    return int(rows[0])


def _semantic_labels(
    *,
    protocol_seed: int,
    task_index: int,
    instance: int,
    hazard: int,
    center_actions: np.ndarray,
    branch_actions: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[str, ...]:
    direction_seed, permutation_seed = _a3_candidate_seeds(
        protocol_seed,
        task_index,
        instance,
        hazard,
    )
    expected = build_diverse_action_branch_bank(
        center_actions,
        direction_seed=direction_seed,
        permutation_seed=permutation_seed,
        action_low=action_low,
        action_high=action_high,
        motion_delta_rms=None,
    )
    if not np.array_equal(expected.actions, branch_actions):
        raise RuntimeError("reconstructed A3 branch bank does not match the cache")
    permutation = np.random.default_rng(permutation_seed).permutation(len(SEMANTIC_BRANCHES))
    return tuple(SEMANTIC_BRANCHES[int(index)] for index in permutation)


def _rgb(record: Any) -> np.ndarray:
    image = np.asarray(record.rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("rendered record must contain CHW RGB")
    image = np.clip(np.transpose(image, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    return np.flip(image, axis=0).copy()


def _display_points(record: Any, *, height: int) -> np.ndarray:
    points = np.array(record.q_true, dtype=np.float32, copy=True)
    if points.shape != (3, 2):
        raise ValueError("rendered record must contain three 2D anchor points")
    points[:, 1] = float(height - 1) - points[:, 1]
    return points


def _draw_anchors(
    image: np.ndarray,
    points: np.ndarray,
    trails: list[list[tuple[float, float]]],
) -> Image.Image:
    canvas = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for anchor, color in enumerate(ANCHOR_COLORS):
        trail = trails[anchor]
        if len(trail) >= 2:
            draw.line(trail, fill=color, width=2)
        x, y = (float(points[anchor, 0]), float(points[anchor, 1]))
        if np.isfinite((x, y)).all():
            trail.append((x, y))
            radius = 4
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return canvas


def _render_branch_frames(
    runtime: Any,
    *,
    snapshot: Any,
    observation: np.ndarray,
    actions: np.ndarray,
    start_success: float,
) -> tuple[list[Image.Image], list[float], list[float]]:
    restore_branch_snapshot(runtime.env, snapshot)
    current_observation = np.asarray(observation, dtype=np.float32).copy()
    trails: list[list[tuple[float, float]]] = [[], [], []]
    start = runtime.record_current(current_observation)
    start_rgb = _rgb(start)
    frames = [
        _draw_anchors(
            start_rgb,
            _display_points(start, height=start_rgb.shape[0]),
            trails,
        )
    ]
    rewards = [0.0]
    successes = [float(start_success)]
    for action in np.asarray(actions, dtype=np.float32):
        current_observation, reward, terminated, truncated, info = runtime.env.step(action)
        current_observation = np.asarray(current_observation, dtype=np.float32)
        record = runtime.record_current(current_observation)
        record_rgb = _rgb(record)
        frames.append(
            _draw_anchors(
                record_rgb,
                _display_points(record, height=record_rgb.shape[0]),
                trails,
            )
        )
        rewards.append(float(reward))
        successes.append(float(info.get("success", 0.0)))
        if terminated or truncated:
            if len(frames) != actions.shape[0] + 1:
                raise RuntimeError("A3 visual branch ended before the cached horizon")
    return frames, rewards, successes


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _fit_label(value: str, width: int = 25) -> str:
    return value if len(value) <= width else value[: width - 1] + "~"


def _tile(
    frame: Image.Image,
    *,
    label: str,
    slot: int,
    raw_step: int,
    reward: float,
    success: float,
    size: int,
) -> Image.Image:
    header = 42
    result = Image.new("RGB", (size, size + header), (245, 247, 250))
    resized = frame.resize((size, size), Image.Resampling.LANCZOS)
    result.paste(resized, (0, header))
    draw = ImageDraw.Draw(result)
    draw.text((7, 4), f"{slot}: {_fit_label(label)}", fill=(24, 31, 42), font=_font(13))
    horizon = min(8, int(np.ceil(raw_step / FRAME_SKIP))) if raw_step else 0
    draw.text(
        (7, 23),
        f"H{horizon}  reward {reward:.2f}  success {success:.0f}",
        fill=(78, 88, 102),
        font=_font(11),
    )
    return result


def _montage_frames(
    branch_frames: list[list[Image.Image]],
    labels: tuple[str, ...],
    rewards: list[list[float]],
    successes: list[list[float]],
    *,
    task: str,
    instance: int,
    hazard: int,
    tile_size: int = 180,
) -> list[Image.Image]:
    header = 54
    tile_height = tile_size + 42
    result: list[Image.Image] = []
    for raw_step in range(len(branch_frames[0])):
        canvas = Image.new("RGB", (tile_size * 4, header + tile_height * 2), (228, 232, 238))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (14, 10),
            f"{task} | instance {instance:02d} | hazard {hazard} | raw {raw_step:02d}/40",
            fill=(20, 27, 38),
            font=_font(17),
        )
        draw.text(
            (14, 33),
            "Red/green/blue trails are the three grounded robot anchors.",
            fill=(76, 86, 101),
            font=_font(11),
        )
        for slot in range(8):
            row, column = divmod(slot, 4)
            tile = _tile(
                branch_frames[slot][raw_step],
                label=labels[slot],
                slot=slot,
                raw_step=raw_step,
                reward=rewards[slot][raw_step],
                success=successes[slot][raw_step],
                size=tile_size,
            )
            canvas.paste(tile, (column * tile_size, header + row * tile_height))
        result.append(canvas)
    return result


def _contact_sheet(
    branch_frames: list[list[Image.Image]],
    labels: tuple[str, ...],
    rewards: list[list[float]],
    successes: list[list[float]],
    *,
    task: str,
    instance: int,
    hazard: int,
    cell_size: int = 170,
) -> Image.Image:
    label_width = 220
    top = 92
    row_height = cell_size + 18
    canvas = Image.new(
        "RGB",
        (label_width + cell_size * len(CONTACT_STEPS), top + row_height * 8),
        (245, 247, 250),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 12),
        f"A3 matched-state action branches: {task}",
        fill=(18, 25, 36),
        font=_font(22),
    )
    draw.text(
        (16, 42),
        f"instance {instance:02d} | hazard step {hazard} | "
        f"branch-start success {successes[0][0]:.0f} | same physical start for every row",
        fill=(74, 84, 99),
        font=_font(13),
    )
    for column, raw_step in enumerate(CONTACT_STEPS):
        horizon = 0 if raw_step == 0 else raw_step // FRAME_SKIP
        draw.text(
            (label_width + column * cell_size + 8, 70),
            f"Branch @ raw {hazard * FRAME_SKIP}" if horizon == 0 else f"H{horizon}",
            fill=(35, 44, 58),
            font=_font(14),
        )
    for slot in range(8):
        y = top + slot * row_height
        draw.text((12, y + 10), f"slot {slot}", fill=(41, 50, 64), font=_font(12))
        draw.text(
            (12, y + 31),
            _fit_label(labels[slot], 27),
            fill=(16, 22, 32),
            font=_font(14),
        )
        draw.text(
            (12, y + 55),
            f"H8 reward {rewards[slot][-1]:.2f}",
            fill=(79, 89, 104),
            font=_font(11),
        )
        draw.text(
            (12, y + 73),
            f"H8 success {successes[slot][-1]:.0f}",
            fill=(79, 89, 104),
            font=_font(11),
        )
        for column, raw_step in enumerate(CONTACT_STEPS):
            image = branch_frames[slot][raw_step].resize(
                (cell_size, cell_size), Image.Resampling.LANCZOS
            )
            canvas.paste(image, (label_width + column * cell_size, y))
    return canvas


def _write_gallery(output_dir: Path, entries: list[dict[str, Any]]) -> Path:
    sections = []
    for entry in entries:
        task = html.escape(str(entry["task"]))
        gif_name = html.escape(str(entry["gif"]))
        png_name = html.escape(str(entry["overview"]))
        sections.append(
            f"""
            <section>
              <div class="heading">
                <h2>{task}</h2>
                <p>Instance {entry['instance']:02d} &middot; hazard {entry['hazard']} &middot; {entry['split']}</p>
              </div>
              <img class="animation" src="{gif_name}" alt="{task} action branches">
              <a class="overview" href="{png_name}">Open full-resolution time sheet</a>
            </section>
            """
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A3 action branch replays</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #eef1f4; color: #121821; letter-spacing: 0; }}
    header {{ padding: 24px max(24px, calc((100vw - 1120px) / 2)); background: #18222d; color: white; }}
    h1, h2, p {{ margin: 0; }}
    header p {{ margin-top: 8px; color: #b9c5cf; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 24px 60px; }}
    section {{ padding: 22px 0 34px; border-bottom: 1px solid #cbd2d9; }}
    .heading {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 14px; }}
    .heading h2 {{ font-size: 20px; }}
    .heading p {{ color: #66717d; font-size: 14px; }}
    .animation {{ display: block; width: min(100%, 900px); height: auto; background: #dfe4e9; }}
    .overview {{ display: inline-block; margin-top: 12px; color: #165a8d; font-weight: 650; text-decoration: none; }}
    .overview:hover {{ text-decoration: underline; }}
    @media (max-width: 640px) {{ .heading {{ display: block; }} .heading p {{ margin-top: 5px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>LeWM A3: matched-state action branches</h1>
    <p>Eight physically executed futures from the same robot state. Colored trails mark the three persistent grounding anchors.</p>
  </header>
  <main>{''.join(sections)}</main>
</body>
</html>
"""
    destination = output_dir / "index.html"
    destination.write_text(page, encoding="utf-8")
    return destination


def render_one(
    arrays: dict[str, np.ndarray],
    contract: dict[str, Any],
    *,
    row: int,
    output_dir: Path,
    gif_fps: int,
) -> dict[str, Any]:
    task_index = int(arrays["task_index"][row])
    task = TASKS[task_index]
    instance = int(arrays["task_instance_index"][row])
    hazard = int(arrays["hazard_id"][row])
    split = str(arrays["split"][row])
    collection = contract["collection"]
    environment_seed = int(collection["environment_seed_base"]) + task_index * 10_000 + instance
    runtime = make_metaworld_episode_runtime(
        task,
        instance,
        environment_seed,
        camera=str(collection["camera"]),
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    center = np.asarray(arrays["center_action"][row], dtype=np.float32).reshape(40, 4)
    branches = np.asarray(arrays["future_action"][row], dtype=np.float32).reshape(8, 40, 4)
    try:
        observation = np.asarray(runtime.observation, dtype=np.float32)
        raw_hazard = hazard * FRAME_SKIP
        final_raw_step = raw_hazard + 40
        source_actions: list[np.ndarray] = []
        snapshot = None
        hazard_observation = None
        hazard_success = None
        latest_info: dict[str, Any] = {}
        for raw_step in range(final_raw_step):
            if raw_step == raw_hazard:
                snapshot = capture_branch_snapshot(runtime.env, include_process_rng=True)
                hazard_observation = observation.copy()
                hazard_success = float(latest_info.get("success", 0.0))
            action = np.asarray(runtime.expert.get_action(observation), dtype=np.float32)
            action = np.clip(action, runtime.env.action_space.low, runtime.env.action_space.high).astype(np.float32)
            source_actions.append(action.copy())
            observation, _, terminated, truncated, latest_info = runtime.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            if (terminated or truncated) and raw_step + 1 < final_raw_step:
                raise RuntimeError("source episode ended before the cached A3 center")
        if snapshot is None or hazard_observation is None or hazard_success is None:
            raise RuntimeError("failed to capture the selected A3 hazard")
        validate_reconstructed_source_actions(
            history_action=np.asarray(arrays["history_action"][row], dtype=np.float32),
            center_action=np.asarray(arrays["center_action"][row], dtype=np.float32),
            hazard_step=hazard,
            source_actions=np.asarray(source_actions, dtype=np.float32),
        )
        labels = _semantic_labels(
            protocol_seed=int(contract["split"]["protocol_seed"]),
            task_index=task_index,
            instance=instance,
            hazard=hazard,
            center_actions=center,
            branch_actions=branches,
            action_low=np.asarray(runtime.env.action_space.low, dtype=np.float32),
            action_high=np.asarray(runtime.env.action_space.high, dtype=np.float32),
        )
        branch_frames: list[list[Image.Image]] = []
        rewards: list[list[float]] = []
        successes: list[list[float]] = []
        for actions in branches:
            frames, branch_rewards, branch_successes = _render_branch_frames(
                runtime,
                snapshot=snapshot,
                observation=hazard_observation,
                actions=actions,
                start_success=hazard_success,
            )
            branch_frames.append(frames)
            rewards.append(branch_rewards)
            successes.append(branch_successes)
    finally:
        _close_episode_env_safely(runtime.env)

    slug = task.removesuffix("-v3")
    overview_name = f"{slug}_instance{instance:02d}_hazard{hazard:02d}_overview.png"
    gif_name = f"{slug}_instance{instance:02d}_hazard{hazard:02d}_branches.gif"
    overview = _contact_sheet(
        branch_frames,
        labels,
        rewards,
        successes,
        task=task,
        instance=instance,
        hazard=hazard,
    )
    overview.save(output_dir / overview_name, format="PNG", optimize=True)
    montage = _montage_frames(
        branch_frames,
        labels,
        rewards,
        successes,
        task=task,
        instance=instance,
        hazard=hazard,
    )
    frame_duration_ms = max(20, int(round(1000 / gif_fps)))
    montage[0].save(
        output_dir / gif_name,
        format="GIF",
        save_all=True,
        append_images=montage[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
    )
    return {
        "task": task,
        "task_index": task_index,
        "row": row,
        "split": split,
        "instance": instance,
        "hazard": hazard,
        "environment_seed": environment_seed,
        "labels_by_slot": list(labels),
        "overview": overview_name,
        "gif": gif_name,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be positive")
    contract = _load_json(args.config)
    arrays = _load_npz(args.input)
    tasks = tuple(args.task) if args.task else TASKS
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for task in tasks:
        row = _select_row(
            arrays,
            task=task,
            split=args.split,
            instance=args.instance,
            hazard=args.hazard,
        )
        result = render_one(
            arrays,
            contract,
            row=row,
            output_dir=output_dir,
            gif_fps=args.gif_fps,
        )
        entries.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gallery = _write_gallery(output_dir, entries)
    print(json.dumps({"gallery": str(gallery), "rendered": len(entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
