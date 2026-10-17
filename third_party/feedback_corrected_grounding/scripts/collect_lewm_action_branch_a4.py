#!/usr/bin/env python3
"""Collect the frozen-encoder seven-task A4 train/development dataset."""

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

from fcpg.lewm_a4_storage import (  # noqa: E402
    save_a4_proposal_ledger,
    save_a4_state_cache,
)
from fcpg.lewm_action_branch_a4 import validate_a4_contract  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    load_frozen_a4_state_encoder,
    make_a4_episode_runtime,
    run_a4_collection,
    save_a4_diagnostics,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_BASE_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_LEDGER = ROOT / "outputs/lewm_action_branch_a4/proposal_ledger.npz"
DEFAULT_CACHE = ROOT / "outputs/lewm_action_branch_a4/state_cache.npz"
DEFAULT_DIAGNOSTICS = ROOT / "outputs/lewm_action_branch_a4/preflight_labels.npz"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--base-checkpoint",
        "--anchor-checkpoint",
        dest="base_checkpoint",
        type=Path,
        default=DEFAULT_BASE_CHECKPOINT,
    )
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument("--ledger", "--output-ledger", dest="ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cache", "--output-cache", dest="cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--diagnostics",
        "--output-diagnostics",
        dest="diagnostics",
        type=Path,
        default=DEFAULT_DIAGNOSTICS,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--max-proposals", type=int, default=None)
    return parser


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A4 config root must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bounds = (args.max_tasks, args.episodes_per_task, args.max_proposals)
    if not args.smoke and any(value is not None for value in bounds):
        raise ValueError("A4 task/episode/proposal bounds are smoke-only")
    contract = _load(args.config)
    validate_a4_contract(contract)
    collection = contract["collection"]
    encoder = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=args.device,
    )
    runtime_factory = partial(
        make_a4_episode_runtime,
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    payload = run_a4_collection(
        contract,
        encoder=encoder,
        runtime_factory=runtime_factory,
        smoke=bool(args.smoke),
        max_tasks=args.max_tasks,
        max_episodes_per_task=args.episodes_per_task,
        max_proposals_per_episode=args.max_proposals,
    )
    ledger_path = save_a4_proposal_ledger(
        args.ledger.expanduser().resolve(), payload.proposal_ledger
    )
    cache_path = save_a4_state_cache(
        args.cache.expanduser().resolve(), payload.state_cache
    )
    diagnostics_path = save_a4_diagnostics(
        args.diagnostics.expanduser().resolve(), payload.diagnostics
    )
    print(
        json.dumps(
            {
                "ledger": str(ledger_path),
                "cache": str(cache_path),
                "diagnostics": str(diagnostics_path),
                "smoke": bool(args.smoke),
                "tasks": payload.metadata["tasks"],
                "proposals": int(payload.proposal_ledger["proposal_id"].shape[0]),
                "bundles": int(payload.state_cache["proposal_id"].shape[0]),
                "test_rows": int(
                    np.count_nonzero(
                        np.asarray(payload.state_cache["split"]).astype(str) == "test"
                    )
                ),
                "camera_by_task": payload.metadata["camera_by_task"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
