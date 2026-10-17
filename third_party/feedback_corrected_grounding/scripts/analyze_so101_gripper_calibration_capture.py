#!/usr/bin/env python3
"""Replay and visualize one cached SO-101 gripper calibration pulse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fcpg.so101_active_gripper import localize_gripper_from_pulse


TRACK_COLORS = ("#d1495b", "#087e8b")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _draw_frame(
    frame: np.ndarray,
    tracks: np.ndarray,
    *,
    time_index: int,
    bbox_xyxy: tuple[int, int, int, int],
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_xyxy, outline="#f4d35e", width=2)
    for candidate_index, track in enumerate(tracks):
        color = TRACK_COLORS[candidate_index % len(TRACK_COLORS)]
        point = tuple(float(value) for value in track[time_index])
        radius = 4
        draw.ellipse(
            (
                point[0] - radius,
                point[1] - radius,
                point[0] + radius,
                point[1] + radius,
            ),
            fill=color,
            outline="white",
            width=1,
        )
        draw.text((point[0] + 6, point[1] - 7), str(candidate_index), fill=color)
    return image


def analyze_capture(capture: Path, output: Path | None = None) -> dict[str, str]:
    """Re-run localization from a persisted pulse and write a visual report."""

    source = capture.expanduser().resolve()
    npz_path = source / "pulse.npz" if source.is_dir() else source
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing calibration pulse: {npz_path}")
    output_dir = (
        output.expanduser().resolve()
        if output is not None
        else npz_path.parent / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=False) as payload:
        column_frames = np.asarray(payload["column_frames"], dtype=np.uint8)
        states = np.asarray(payload["states"], dtype=np.float32)
    localization = localize_gripper_from_pulse(column_frames, states)
    tracks = np.asarray(
        localization.moving_tip_track_candidates_px, dtype=np.float32
    )
    open_image = _draw_frame(
        column_frames[localization.open_frame_index],
        tracks,
        time_index=0,
        bbox_xyxy=localization.bbox_xyxy,
    )
    closed_image = _draw_frame(
        column_frames[localization.closed_frame_index],
        tracks,
        time_index=1,
        bbox_xyxy=localization.bbox_xyxy,
    )
    pair = Image.new("RGB", (open_image.width * 2, open_image.height), "white")
    pair.paste(open_image, (0, 0))
    pair.paste(closed_image, (open_image.width, 0))
    overlay_path = output_dir / "column_open_closed_tracks.png"
    pair.save(overlay_path)

    diagnostics = {
        **localization.to_dict(),
        "candidate_displacement_px": np.linalg.norm(
            tracks[:, 1] - tracks[:, 0], axis=1
        ),
        "source_npz": str(npz_path),
    }
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(_jsonable(diagnostics), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    calibration_path = output_dir / "live_anchor_calibration.json"
    calibration_path.write_text(
        json.dumps(
            _jsonable(
                {
                    "version": 1,
                    "view": "column",
                    "moving_tip_track_candidates_px": tracks,
                    "open_state": states[localization.open_frame_index],
                    "closed_state": states[localization.closed_frame_index],
                    "source_capture": str(npz_path),
                }
            ),
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    pulse_relative = Path("..") / "column_pulse.gif"
    index_path = output_dir / "index.html"
    index_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>SO-101 gripper-track replay</title>"
        "<style>body{font-family:sans-serif;margin:20px;background:#f5f6f7;color:#17202a}"
        "img{max-width:100%;border:1px solid #ccd1d1}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}</style>"
        "</head><body><h1>Moving fingertip calibration replay</h1>"
        f"<div class=\"grid\"><img src=\"{pulse_relative.as_posix()}\">"
        "<img src=\"column_open_closed_tracks.png\"></div>"
        "<p><a href=\"diagnostics.json\">diagnostics.json</a></p>"
        "</body></html>",
        encoding="utf-8",
    )
    return {
        "directory": str(output_dir),
        "index_html": str(index_path),
        "overlay": str(overlay_path),
        "diagnostics": str(diagnostics_path),
        "live_anchor_calibration": str(calibration_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="pulse directory or pulse.npz")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_capture(args.capture, args.output), indent=2))


if __name__ == "__main__":
    main()
