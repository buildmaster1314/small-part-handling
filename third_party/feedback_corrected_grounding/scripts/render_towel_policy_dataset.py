#!/usr/bin/env python3
"""Render complete Towel policy demonstrations from a projection dataset."""

from __future__ import annotations

import argparse
import html
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.custom_tasks.towel_wipe_expert import infer_towel_wipe_phase


REQUIRED_FIELDS = frozenset(
    {
        "rgb",
        "obs",
        "action",
        "success",
        "task_name",
        "task_index",
        "episode_id",
        "env_seed",
        "frame_id",
        "q_true",
        "q_nominal",
        "valid_true",
        "valid_nominal",
        "observation_action_alignment",
    }
)
WIPE_PHASES = frozenset({"wipe_left", "wipe_right"})
ANCHOR_COLORS = ((255, 210, 55), (42, 207, 239), (235, 80, 148))


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def validate_dataset(data: Mapping[str, np.ndarray]) -> None:
    fields = set(data.files) if hasattr(data, "files") else set(data)
    missing = REQUIRED_FIELDS.difference(fields)
    if missing:
        raise KeyError(f"towel dataset is missing fields: {sorted(missing)}")
    if str(np.asarray(data["observation_action_alignment"]).item()) != "pre_action":
        raise ValueError("towel policy data must use pre_action alignment")
    sample_count = int(np.asarray(data["rgb"]).shape[0])
    for name in REQUIRED_FIELDS.difference({"observation_action_alignment"}):
        if int(np.asarray(data[name]).shape[0]) != sample_count:
            raise ValueError(f"dataset field {name} does not match rgb sample count")
    if np.asarray(data["rgb"]).ndim != 4 or np.asarray(data["rgb"]).shape[-1] != 3:
        raise ValueError("rgb must have shape [N,H,W,3]")
    if np.asarray(data["q_true"]).shape[1:] != (3, 2):
        raise ValueError("q_true must contain exactly three 2-D robot anchors")
    if np.asarray(data["action"]).shape[1:] != (4,):
        raise ValueError("action must have shape [N,4]")


def episode_indices(data: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    tasks = np.asarray(data["task_name"]).astype(str)
    episodes = np.asarray(data["episode_id"], dtype=np.int64)
    frames = np.asarray(data["frame_id"], dtype=np.int64)
    keys = sorted(set(zip(tasks.tolist(), episodes.tolist(), strict=True)))
    result: list[np.ndarray] = []
    for task, episode in keys:
        indices = np.flatnonzero((tasks == task) & (episodes == episode))
        indices = indices[np.argsort(frames[indices])]
        ordered_frames = frames[indices]
        if ordered_frames.size == 0 or not np.array_equal(
            ordered_frames, np.arange(ordered_frames.size, dtype=np.int64)
        ):
            raise ValueError(f"episode {(task, episode)} does not contain contiguous frames")
        result.append(indices)
    return result


def count_wipe_runs(phases: list[str]) -> tuple[int, tuple[str, ...]]:
    runs: list[str] = []
    previous: str | None = None
    for phase in phases:
        if phase in WIPE_PHASES and phase != previous:
            runs.append(phase)
        previous = phase
    return len(runs), tuple(runs)


def summarize_episode(
    data: Mapping[str, np.ndarray], indices: np.ndarray
) -> dict[str, object]:
    obs = np.asarray(data["obs"])[indices]
    phases = [infer_towel_wipe_phase(row).value for row in obs]
    half_cycles, wipe_order = count_wipe_runs(phases)
    success = np.asarray(data["success"], dtype=np.float32)[indices]
    success_rows = np.flatnonzero(success > 0.5)
    nominal = np.asarray(data["q_nominal"], dtype=np.float32)[indices]
    true = np.asarray(data["q_true"], dtype=np.float32)[indices]
    valid = np.asarray(data["valid_true"], dtype=bool)[indices] & np.asarray(
        data["valid_nominal"], dtype=bool
    )[indices]
    return {
        "task": str(np.asarray(data["task_name"])[indices[0]]),
        "episode": int(np.asarray(data["episode_id"])[indices[0]]),
        "task_instance": int(np.asarray(data["task_index"])[indices[0]]),
        "env_seed": int(np.asarray(data["env_seed"])[indices[0]]),
        "frames": int(indices.size),
        "first_success_frame": (
            None if success_rows.size == 0 else int(success_rows[0])
        ),
        "final_success": bool(success[-1] > 0.5),
        "wipe_half_cycles": int(half_cycles),
        "wipe_round_trips": float(half_cycles / 2.0),
        "wipe_order": list(wipe_order),
        "all_anchors_valid": bool(valid.all()),
        "maximum_nominal_true_epe_px": float(
            np.linalg.norm(nominal - true, axis=-1).max()
        ),
        "phases": list(dict.fromkeys(phases)),
    }


def _draw_frame(
    rgb: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    action: np.ndarray,
    *,
    frame_id: int,
    phase: str,
    success: bool,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 42), fill=(12, 18, 24, 190))
    draw.text(
        (7, 5),
        f"frame {frame_id:03d} | {phase} | success {int(success)}",
        fill=(255, 255, 255, 255),
        font=_font(11),
    )
    draw.text(
        (7, 23),
        "action " + " ".join(f"{value:+.2f}" for value in action),
        fill=(205, 218, 228, 255),
        font=_font(10),
    )
    for anchor, (point, is_valid, color) in enumerate(
        zip(points, valid, ANCHOR_COLORS, strict=True)
    ):
        if not bool(is_valid) or not np.isfinite(point).all():
            continue
        x, y = (float(point[0]), float(point[1]))
        radius = 4
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*color, 235),
            outline=(15, 20, 24, 255),
            width=1,
        )
        draw.text((x + 5, y - 7), str(anchor), fill=(*color, 255), font=_font(9))
    return np.asarray(image)


def _render_episode(
    data: Mapping[str, np.ndarray],
    indices: np.ndarray,
    output: Path,
    *,
    fps: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(data["rgb"])[indices]
    obs = np.asarray(data["obs"])[indices]
    points = np.asarray(data["q_true"])[indices]
    valid = np.asarray(data["valid_true"])[indices]
    action = np.asarray(data["action"])[indices]
    frame_id = np.asarray(data["frame_id"])[indices]
    success = np.asarray(data["success"])[indices]
    phases = [infer_towel_wipe_phase(row).value for row in obs]
    writer = imageio.get_writer(
        output,
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for position in range(indices.size):
            writer.append_data(
                _draw_frame(
                    rgb[position],
                    points[position],
                    valid[position],
                    action[position],
                    frame_id=int(frame_id[position]),
                    phase=phases[position],
                    success=bool(float(success[position]) > 0.5),
                )
            )
    finally:
        writer.close()


def _write_html(
    output_dir: Path,
    source: Path,
    entries: list[dict[str, object]],
    *,
    fps: float,
) -> Path:
    cards = []
    for entry in entries:
        summary = entry["summary"]
        assert isinstance(summary, dict)
        order = " -> ".join(str(value).replace("wipe_", "") for value in summary["wipe_order"])
        cards.append(
            f"""<article><h2>Episode {summary['episode']} | instance {summary['task_instance']}</h2>
<video controls muted loop playsinline preload="metadata"><source src="{html.escape(str(entry['video']), quote=True)}" type="video/mp4"></video>
<dl><dt>Frames</dt><dd>{summary['frames']}</dd><dt>First success</dt><dd>{summary['first_success_frame']}</dd><dt>Round trips</dt><dd>{summary['wipe_round_trips']:.1f}</dd><dt>Wipe order</dt><dd>{html.escape(order)}</dd><dt>All anchors valid</dt><dd>{summary['all_anchors_valid']}</dd><dt>Max clean EPE</dt><dd>{summary['maximum_nominal_true_epe_px']:.3f} px</dd></dl></article>"""
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Towel policy demonstrations</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#eef1f3;color:#17212a;font-family:Arial,sans-serif}}header{{background:#18252d;color:white;padding:22px max(20px,calc((100vw - 1180px)/2))}}header p{{color:#cad4da;line-height:1.5}}main{{max-width:1180px;margin:auto;padding:22px 20px 60px;display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}article{{background:white;border:1px solid #d5dce0;padding:14px;border-radius:6px}}h2{{font-size:17px;margin:0 0 10px}}video{{width:100%;height:auto;background:#111}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:6px 14px;font-size:13px}}dt{{color:#65747d}}dd{{margin:0}}code{{color:#fff}}</style></head><body><header><h1>Towel Policy Dataset Smoke</h1><p>Actual pre-action RGB demonstrations at {fps:g} FPS. Colored circles 0/1/2 are the three recorded robot anchors and are diagnostic overlays, not model inputs. Source: <code>{html.escape(str(source))}</code>.</p></header><main>{''.join(cards)}</main></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-episodes", type=int, default=4)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")

    source = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    with np.load(source, allow_pickle=False) as data:
        validate_dataset(data)
        groups = episode_indices(data)[: args.max_episodes]
        for indices in groups:
            summary = summarize_episode(data, indices)
            video_name = f"episode_{int(summary['episode']):03d}.mp4"
            _render_episode(data, indices, output_dir / video_name, fps=args.fps)
            entries.append({"video": video_name, "summary": summary})
    (output_dir / "summary.json").write_text(
        json.dumps([entry["summary"] for entry in entries], indent=2),
        encoding="utf-8",
    )
    index = _write_html(output_dir, source, entries, fps=args.fps)
    print(f"output: {index}")
    print(f"episodes: {len(entries)}")


if __name__ == "__main__":
    main()
