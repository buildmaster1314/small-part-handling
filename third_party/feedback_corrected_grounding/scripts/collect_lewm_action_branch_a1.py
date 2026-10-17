#!/usr/bin/env python3
"""Collect the A1 same-snapshot action-effect dataset without training."""

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
from fcpg.lewm_action_branch_collection import (  # noqa: E402
    CENTER_SOURCE,
    COLLECTION_PURPOSE,
    MIN_HORIZON_STEPS,
    load_frozen_fcpg_anchor_encoder,
    make_metaworld_episode_runtime,
    run_action_branch_collection,
)


DEFAULT_CONFIG = "config/lewm/eval/action_branch_random_hazard_a1.json"
DEFAULT_ANCHOR_CHECKPOINT = (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_OUTPUT = "outputs/lewm_action_branch_a1/action_branch_a1.npz"


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _load_contract(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("A1 config root must be a JSON object")
    return value


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect eight physical branches around the source scripted expert's "
            "exact future actions; this is not deployment CEM collection."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--anchor-checkpoint", default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--horizon",
        type=_positive,
        default=MIN_HORIZON_STEPS,
        help="endpoint horizon P; P>=8 supplies exactly P*5 scripted raw actions",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=_positive,
        default=None,
        help="smoke-only episode bound (the preregistered smoke default is 2)",
    )
    parser.add_argument(
        "--max-events",
        type=_positive,
        default=None,
        help="smoke-only maximum retained hazards per source episode",
    )
    parser.add_argument(
        "--max-tasks",
        type=_positive,
        default=None,
        help="optional smoke-only task bound",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=_positive, default=224)
    parser.add_argument("--height", type=_positive, default=224)
    parser.add_argument("--benchmark-seed", type=int, default=18_301)
    parser.add_argument("--environment-seed-base", type=int, default=40_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.horizon < MIN_HORIZON_STEPS:
        raise ValueError(f"--horizon must be at least {MIN_HORIZON_STEPS}")
    if not args.smoke and any(
        value is not None
        for value in (args.episodes_per_task, args.max_events, args.max_tasks)
    ):
        raise ValueError("episode/event/task bounds are smoke-only")

    contract = _load_contract(_resolve(args.config))
    encoder = load_frozen_fcpg_anchor_encoder(
        _resolve(args.anchor_checkpoint),
        _resolve(args.fcpg_checkpoint),
        device=args.device,
    )
    runtime_factory = partial(
        make_metaworld_episode_runtime,
        camera=str(args.camera),
        width=int(args.width),
        height=int(args.height),
        benchmark_seed=int(args.benchmark_seed),
    )
    payload = run_action_branch_collection(
        contract,
        encoder=encoder,
        runtime_factory=runtime_factory,
        horizon_steps=int(args.horizon),
        environment_seed_base=int(args.environment_seed_base),
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
                "samples": int(payload.arrays["future_action"].shape[0]),
                "horizon": int(payload.arrays["future_action"].shape[2]),
                "center_source": CENTER_SOURCE,
                "purpose": COLLECTION_PURPOSE,
                "action_policy_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
