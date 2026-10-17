#!/usr/bin/env python3
"""Seal a schema-3 FCPG cache as a checkpoint-seed-bound measurement view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.independent_fcpg_measurement import (  # noqa: E402
    seal_independent_fcpg_measurement,
    validate_independent_fcpg_measurement_view,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-overwriting independent-measurement view from an "
            "existing schema-3 FCPG correction cache."
        )
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--generation-seed", required=True, type=int)
    parser.add_argument(
        "--transfer",
        choices=("copy", "hardlink", "auto"),
        default="copy",
    )
    args = parser.parse_args()

    metadata = seal_independent_fcpg_measurement(
        args.cache_dir,
        args.output_dir,
        measurement_view_id=args.view_id,
        measurement_generation_seed=args.generation_seed,
        transfer=args.transfer,
    )
    validated = validate_independent_fcpg_measurement_view(args.output_dir)
    report = {
        "output_dir": str(validated.cache_dir),
        "measurement_view_id": metadata["measurement_view_id"],
        "measurement_generation_seed": metadata["measurement_generation_seed"],
        "checkpoint_sha256": validated.checkpoint_sha256,
        "q_corrected_sha256": validated.q_corrected_sha256,
        "metadata_sha256": validated.metadata_sha256,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
