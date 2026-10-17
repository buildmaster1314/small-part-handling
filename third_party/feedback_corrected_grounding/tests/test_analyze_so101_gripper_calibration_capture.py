from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.analyze_so101_gripper_calibration_capture import analyze_capture


def test_cached_gripper_pulse_can_be_reanalyzed_without_hardware(tmp_path: Path) -> None:
    frames = np.full((7, 96, 96, 3), 190, dtype=np.uint8)
    states = np.zeros((7, 6), dtype=np.float32)
    states[:, 5] = [80, 60, 35, 5, 35, 60, 80]
    for frame, x in zip(frames, [68, 62, 56, 50, 56, 62, 68], strict=True):
        jaw = np.asarray([[30, 18], [x + 9, 58], [x, 66]], dtype=np.int32)
        cv2.fillConvexPoly(frame, jaw, (35, 35, 35))
    np.savez_compressed(
        tmp_path / "pulse.npz",
        wrist_frames=frames,
        column_frames=frames,
        states=states,
        commands=states[:, 5],
    )

    result = analyze_capture(tmp_path)

    assert Path(result["index_html"]).is_file()
    assert Path(result["overlay"]).is_file()
    diagnostics = json.loads(Path(result["diagnostics"]).read_text(encoding="utf-8"))
    assert len(diagnostics["moving_tip_track_candidates_px"]) == 2
    assert diagnostics["candidate_displacement_px"][0] > 5.0
    calibration = json.loads(
        Path(result["live_anchor_calibration"]).read_text(encoding="utf-8")
    )
    assert calibration["version"] == 1
    assert calibration["view"] == "column"
    assert calibration["open_state"][5] == 80.0
    assert calibration["closed_state"][5] == 5.0
