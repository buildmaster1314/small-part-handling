#!/usr/bin/env python3
"""Build the audited seven-task A4c coarse-control index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4c_control_index import build_a4c_control_index  # noqa: E402


DEFAULT_RAW_ROOT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_A4C_ROOT = ROOT / "outputs/lewm_action_branch_a4c_v4"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_prefix_a4/a4c_control_index_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--a4c-root", type=Path, default=DEFAULT_A4C_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def main() -> int:
    args = _parser().parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    a4c_root = args.a4c_root.expanduser().resolve()
    schedule = json.loads((a4c_root / "phase_schedule.json").read_text(encoding="utf-8"))
    payload, summary = build_a4c_control_index(
        _npz(raw_root / "index.npz"),
        _npz(a4c_root / "state_cache.npz"),
        _npz(a4c_root / "proposal_ledger.npz"),
        _npz(a4c_root / "preflight_labels.npz"),
        schedule,
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "index.npz", **payload)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
