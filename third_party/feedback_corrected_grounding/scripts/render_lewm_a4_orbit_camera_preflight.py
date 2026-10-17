#!/usr/bin/env python3
"""Render all seven A4 tasks from the 12 clock-face camera views."""

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

from fcpg.lewm_action_branch_a4 import TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    a4_camera_name,
    make_a4_episode_runtime,
)
from fcpg.lewm_orbit_camera import ORBIT_ALL_VIEWS, OrbitCameraView  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_prefix_a4/orbit_camera_preflight"
ANCHOR_COLORS = ((35, 193, 116), (236, 92, 74), (42, 132, 214))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--task-instance", type=int, default=0)
    parser.add_argument("--benchmark-seed", type=int, default=30_103)
    parser.add_argument("--environment-seed", type=int, default=140_000)
    parser.add_argument("--radius-m", type=float, default=1.1)
    parser.add_argument("--elevation-degrees", type=float, default=35.0)
    return parser.parse_args()


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _upright(task: str, rgb: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.moveaxis(np.asarray(rgb, dtype=np.float32), 0, -1)
    image = np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)
    anchors = np.asarray(points, dtype=np.float32).copy()
    return image, anchors


def _panel(task: str, record: Any, view: OrbitCameraView, split: str) -> Image.Image:
    rgb, anchors = _upright(task, record.rgb, record.q_true)
    panel = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(panel)
    for anchor_index, (x, y) in enumerate(anchors):
        color = ANCHOR_COLORS[anchor_index]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255), width=1)
    draw.rectangle((0, 0, panel.width, 28), fill=(13, 21, 27))
    draw.text(
        (7, 5),
        f"{view.clock_hour:02d}:00 | {split}",
        fill=(247, 248, 249),
        font=_font(13),
    )
    return panel


def _montage(task: str, panels: list[Image.Image]) -> Image.Image:
    columns, rows = 4, 3
    gap, margin, header = 8, 12, 42
    panel_w, panel_h = panels[0].size
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + columns * panel_w + (columns - 1) * gap,
            header + margin + rows * panel_h + (rows - 1) * gap + margin,
        ),
        (232, 235, 237),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), task, fill=(21, 29, 35), font=_font(19))
    for index, panel in enumerate(panels):
        column = index % columns
        row = index // columns
        x = margin + column * (panel_w + gap)
        y = header + margin + row * (panel_h + gap)
        canvas.paste(panel, (x, y))
    return canvas


def _view_metrics(task: str, record: Any, view: OrbitCameraView, width: int, height: int) -> dict[str, Any]:
    q_true = np.asarray(record.q_true, dtype=np.float64)
    q_nominal = np.asarray(record.q_nominal, dtype=np.float64)
    valid = np.asarray(record.valid_true, dtype=bool)
    margins = np.stack(
        [q_true[:, 0], width - 1 - q_true[:, 0], q_true[:, 1], height - 1 - q_true[:, 1]],
        axis=-1,
    )
    return {
        "task": task,
        "camera": view.label,
        "clock_hour": int(view.clock_hour),
        "split": "train" if int(view.clock_hour) % 2 == 0 else "development",
        "all_anchors_valid": bool(valid.all()),
        "valid_anchor_count": int(valid.sum()),
        "minimum_frame_margin_px": float(margins.min()),
        "nominal_true_max_error_px": float(np.linalg.norm(q_nominal - q_true, axis=-1).max()),
    }


def main() -> int:
    args = _parser()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    task_images: dict[str, str] = {}
    for task_index, task in enumerate(TASKS):
        panels: list[Image.Image] = []
        for view in ORBIT_ALL_VIEWS:
            task_view = OrbitCameraView(
                clock_hour=view.clock_hour,
                elevation_degrees=float(args.elevation_degrees),
                radius_m=float(args.radius_m),
                base_camera_name=a4_camera_name(task),
            )
            runtime = make_a4_episode_runtime(
                task,
                int(args.task_instance),
                int(args.environment_seed) + task_index,
                width=int(args.width),
                height=int(args.height),
                benchmark_seed=int(args.benchmark_seed),
                camera_name=task_view.base_camera_name,
                orbit_camera_view=task_view,
            )
            try:
                record = runtime.record_current(runtime.observation)
            finally:
                runtime.env.close()
            split = "train" if int(view.clock_hour) % 2 == 0 else "held-out"
            panels.append(_panel(task, record, task_view, split))
            rows.append(_view_metrics(task, record, task_view, args.width, args.height))
        filename = f"{task.replace('/', '_')}.png"
        _montage(task, panels).save(output / filename)
        task_images[task] = filename

    summary = {
        "tasks": list(TASKS),
        "train_clock_hours": [12, 2, 4, 6, 8, 10],
        "development_clock_hours": [1, 3, 5, 7, 9, 11],
        "all_task_views_valid": bool(all(row["all_anchors_valid"] for row in rows)),
        "minimum_frame_margin_px": float(min(row["minimum_frame_margin_px"] for row in rows)),
        "maximum_nominal_true_error_px": float(max(row["nominal_true_max_error_px"] for row in rows)),
        "views": rows,
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sections = []
    for task in TASKS:
        sections.append(
            f'<section><h2>{html.escape(task)}</h2><img src="{html.escape(task_images[task])}" alt="{html.escape(task)} orbit views"></section>'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>A4 orbit camera preflight</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#eef1f2;color:#182027}}
main{{max-width:980px;margin:auto}}h1{{font-size:26px}}p{{line-height:1.45}}
section{{margin:26px 0}}img{{display:block;width:100%;height:auto;border:1px solid #bcc5ca}}
code{{background:#dde3e6;padding:2px 5px}}
</style></head><body><main><h1>A4 12-view orbit camera preflight</h1>
<p>Even clock hours are training views; odd clock hours are held-out development views. Green/red/blue dots are the three robot anchors. Exact nominal calibration is used, so this checks camera framing and projection before collection.</p>
{''.join(sections)}</main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "views"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
