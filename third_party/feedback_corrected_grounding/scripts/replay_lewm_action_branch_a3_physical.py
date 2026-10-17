#!/usr/bin/env python3
"""Replay exact A3 actions and save aligned physical endpoint diagnostics."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a3_collection import validate_a3_contract  # noqa: E402
from fcpg.lewm_action_branch_a3_physical import replay_a3_physical_rows  # noqa: E402
from fcpg.lewm_action_branch_collection import make_metaworld_episode_runtime  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
DEFAULT_INPUT = ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_screen.npz"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_physical.npz"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A3 config root must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = _load_contract(args.config)
    validate_a3_contract(contract)
    latent = load_a1_npz(args.input.expanduser().resolve())
    collection = contract["collection"]
    runtime_factory = partial(
        make_metaworld_episode_runtime,
        camera=str(collection["camera"]),
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    physical = replay_a3_physical_rows(
        latent,
        contract,
        runtime_factory=runtime_factory,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **physical)
    print(json.dumps({"output": str(output), "rows": int(physical["tcp"].shape[0])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
