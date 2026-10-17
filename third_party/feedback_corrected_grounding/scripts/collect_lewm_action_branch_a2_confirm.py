#!/usr/bin/env python3
"""Collect the fresh A2 confirmation training support split."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a1 import save_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a2_confirmation import (  # noqa: E402
    run_confirmation_collection,
    validate_confirmation_contract,
)
from fcpg.lewm_action_branch_collection import (  # noqa: E402
    load_frozen_fcpg_anchor_encoder,
    make_metaworld_episode_runtime,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_continuous_task_a2_confirm.json"
DEFAULT_ANCHOR_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    validate_confirmation_contract(contract)
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
    payload = run_confirmation_collection(
        contract,
        split=args.split,
        encoder=encoder,
        runtime_factory=runtime_factory,
    )
    output = save_a1_npz(args.output.expanduser().resolve(), payload.arrays)
    print(
        json.dumps(
            {
                "output": str(output),
                "split": args.split,
                "bundles": int(payload.arrays["target_embedding"].shape[0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
