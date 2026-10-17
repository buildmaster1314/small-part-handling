#!/usr/bin/env python3
"""Validate and record an exact paired camera intervention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.camera_pairing import validate_paired_camera_datasets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--output")
    parser.add_argument("--allow-identical-camera", action="store_true")
    args = parser.parse_args()

    report = validate_paired_camera_datasets(
        args.reference,
        args.intervention,
        require_camera_difference=not args.allow_identical_camera,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite validation report: {output}")
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
