#!/usr/bin/env python3
"""Compose exact paired camera cohorts into a within-episode camera step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.paired_camera_step import compose_paired_camera_step


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose two physically identical camera cohorts. Frames before each "
            "episode's switch come from the reference cohort; the switch frame and "
            "later frames come from the intervention cohort."
        )
    )
    parser.add_argument("--reference-data", required=True)
    parser.add_argument("--intervention-data", required=True)
    parser.add_argument("--reference-correction-cache", required=True)
    parser.add_argument("--intervention-correction-cache", required=True)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--output-correction-cache", required=True)
    parser.add_argument("--output-metadata")
    parser.add_argument(
        "--switch-frame",
        type=int,
        help="Fixed per-episode switch frame (default: 50).",
    )
    parser.add_argument(
        "--switch-frame-min",
        type=int,
        help="Inclusive minimum for deterministic per-episode switches.",
    )
    parser.add_argument(
        "--switch-frame-max",
        type=int,
        help="Inclusive maximum for deterministic per-episode switches.",
    )
    parser.add_argument(
        "--switch-seed",
        type=int,
        default=0,
        help="Seed for the stable per-episode ranged schedule.",
    )
    args = parser.parse_args()

    range_requested = (
        args.switch_frame_min is not None or args.switch_frame_max is not None
    )
    if range_requested:
        if args.switch_frame is not None:
            parser.error("--switch-frame cannot be combined with a switch-frame range")
        if args.switch_frame_min is None or args.switch_frame_max is None:
            parser.error("--switch-frame-min and --switch-frame-max are both required")
    elif args.switch_frame is None:
        args.switch_frame = 50
    return args


def main() -> int:
    args = _parse_args()
    result = compose_paired_camera_step(
        reference_data=args.reference_data,
        intervention_data=args.intervention_data,
        reference_correction_cache=args.reference_correction_cache,
        intervention_correction_cache=args.intervention_correction_cache,
        switch_frame=args.switch_frame,
        switch_frame_min=args.switch_frame_min,
        switch_frame_max=args.switch_frame_max,
        switch_seed=args.switch_seed,
        output_data=args.output_data,
        output_correction_cache=args.output_correction_cache,
        output_metadata=args.output_metadata,
    )
    rendered = {
        "data": str(result.data_path),
        "metadata": str(result.data_metadata_path),
        "correction_cache": str(result.correction_cache),
        "data_sha256": result.data_sha256,
        "q_corrected_sha256": result.q_corrected_sha256,
        "sample_count": result.sample_count,
        "reference_frames": result.reference_frames,
        "intervention_frames": result.intervention_frames,
    }
    print(json.dumps(rendered, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
