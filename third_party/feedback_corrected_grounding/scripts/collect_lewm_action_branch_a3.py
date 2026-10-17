#!/usr/bin/env python3
"""Collect fresh A3 latent/action branches without opening reserved test."""

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

from fcpg.lewm_action_branch_a1 import save_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a3_collection import (  # noqa: E402
    run_a3_collection,
    validate_a3_contract,
)
from fcpg.lewm_action_branch_collection import (  # noqa: E402
    load_frozen_fcpg_anchor_encoder,
    make_metaworld_episode_runtime,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
DEFAULT_ANCHOR_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_screen.npz"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    return parser


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A3 config root must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.smoke and any(
        value is not None
        for value in (args.max_tasks, args.episodes_per_task, args.max_events)
    ):
        raise ValueError("A3 episode/event/task bounds are smoke-only")
    contract = _load(args.config)
    validate_a3_contract(contract)
    collection = contract["collection"]
    encoder = load_frozen_fcpg_anchor_encoder(
        args.anchor_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=args.device,
    )
    runtime_factory = partial(
        make_metaworld_episode_runtime,
        camera=str(collection["camera"]),
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    payload = run_a3_collection(
        contract,
        encoder=encoder,
        runtime_factory=runtime_factory,
        smoke=bool(args.smoke),
        max_tasks=args.max_tasks,
        max_episodes_per_task=args.episodes_per_task,
        max_events_per_episode=args.max_events,
    )
    destination = save_a1_npz(args.output.expanduser().resolve(), payload.arrays)
    print(
        json.dumps(
            {
                "output": str(destination),
                "smoke": bool(args.smoke),
                "bundles": int(payload.arrays["target_embedding"].shape[0]),
                "test_rows": int(
                    np.count_nonzero(np.asarray(payload.arrays["split"]).astype(str) == "test")
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
