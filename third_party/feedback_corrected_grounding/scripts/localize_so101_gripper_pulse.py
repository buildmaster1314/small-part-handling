#!/usr/bin/env python3
"""Localize the SO-101 gripper from an already-recorded open/close pulse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fcpg.so101_active_gripper import localize_gripper_from_pulse
from fcpg.so101_shadow_session import load_shadow_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    column = load_shadow_session(
        args.session.expanduser().resolve() / "column",
        allow_control=True,
    )
    result = localize_gripper_from_pulse(column.frames, column.states)
    payload = {
        "format": "so101-active-gripper-localization-v1",
        "camera": "column",
        "session": str(args.session.expanduser().resolve()),
        **result.to_dict(),
    }
    output.mkdir(parents=True)
    (output / "localization.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frame = Image.fromarray(
        np.asarray(column.frames[result.closed_frame_index]), mode="RGB"
    )
    draw = ImageDraw.Draw(frame)
    x, y = result.center_px
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(250, 250, 250), width=4)
    draw.rectangle(result.bbox_xyxy, outline=(40, 190, 215), width=2)
    frame.save(output / "column_closed_localized.jpg", quality=94)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"preview={output / 'column_closed_localized.jpg'}")


if __name__ == "__main__":
    main()
