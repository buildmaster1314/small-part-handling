#!/usr/bin/env python3
"""Collect the fresh continuous-task A2 train/development screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a1 import save_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a2 import run_a2_collection  # noqa: E402
from fcpg.lewm_action_branch_collection import (  # noqa: E402
    load_frozen_fcpg_anchor_encoder,
    make_metaworld_episode_runtime,
)


DEFAULT_CONFIG = "config/lewm/eval/action_branch_continuous_task_a2.json"
DEFAULT_ANCHOR_CHECKPOINT = (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_OUTPUT = "outputs/lewm_action_branch_a2/action_branch_a2_screen.npz"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("A2 config root must be an object")
    return value


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--anchor-checkpoint", default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes-per-task", type=_positive, default=None)
    parser.add_argument("--max-events", type=_positive, default=None)
    parser.add_argument("--max-tasks", type=_positive, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.smoke and any(
        value is not None
        for value in (args.episodes_per_task, args.max_events, args.max_tasks)
    ):
        raise ValueError("episode/event/task bounds are smoke-only")
    contract = _load(_resolve(args.config))
    collection = contract["collection"]
    encoder = load_frozen_fcpg_anchor_encoder(
        _resolve(args.anchor_checkpoint),
        _resolve(args.fcpg_checkpoint),
        device=args.device,
    )
    runtime_factory = partial(
        make_metaworld_episode_runtime,
        camera=str(collection["camera"]),
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    payload = run_a2_collection(
        contract,
        encoder=encoder,
        runtime_factory=runtime_factory,
        smoke=bool(args.smoke),
        max_tasks=args.max_tasks,
        max_episodes_per_task=args.episodes_per_task,
        max_events_per_episode=args.max_events,
    )
    destination = save_a1_npz(_resolve(args.output), payload.arrays)
    print(
        json.dumps(
            {
                "output": str(destination),
                "smoke": bool(args.smoke),
                "bundles": int(payload.arrays["target_embedding"].shape[0]),
                "test_rows": int(
                    (payload.arrays["split"].astype(str) == "test").sum()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
