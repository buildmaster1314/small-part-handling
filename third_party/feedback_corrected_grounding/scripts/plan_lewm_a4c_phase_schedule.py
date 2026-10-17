#!/usr/bin/env python3
"""Plan phase-aware seven-task A4c proposal windows without using a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4c_schedule import (  # noqa: E402
    PROPOSALS_PER_EPISODE,
    build_phase_schedule_manifest,
    load_a4c_contract,
    save_manifest,
    validate_schedule_manifest,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_phase_aware_a4c.json"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4c_v4/phase_schedule.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--proposals-per-episode", type=int, default=PROPOSALS_PER_EPISODE)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    config = load_a4c_contract(args.config)
    manifest = build_phase_schedule_manifest(
        config,
        width=int(args.width),
        height=int(args.height),
        count=int(args.proposals_per_episode),
        max_tasks=args.max_tasks,
    )
    if args.max_tasks is None:
        validate_schedule_manifest(manifest, config)
    path = save_manifest(manifest, args.output)
    summary: dict[str, dict[str, float | int]] = {}
    for task in manifest["tasks"]:
        rows = [row for row in manifest["episodes"] if row["task"] == task]
        summary[task] = {
            "episodes": len(rows),
            "mean_selected": float(sum(len(row["proposal_steps"]) for row in rows) / max(len(rows), 1)),
            "success_crossing_fraction": float(sum(row["coverage"]["h8_crosses_success"] for row in rows) / max(len(rows), 1)),
            "phase_transition_fraction": float(sum(row["coverage"]["h8_crosses_phase_transition"] for row in rows) / max(len(rows), 1)),
            "mean_first_success": float(sum(row["first_success_step"] or 0 for row in rows) / max(len(rows), 1)),
        }
    print(json.dumps({"output": str(path), "tasks": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
