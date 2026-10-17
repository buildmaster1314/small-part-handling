#!/usr/bin/env python3
"""Evaluate a frozen joint A4 checkpoint with matched action controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    JOINT_PROTOCOL,
    load_joint_checkpoint,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_pose_conditioned_joint_a4 import (  # noqa: E402
    POSE_JOINT_PROTOCOL,
    load_pose_conditioned_joint_checkpoint,
)
from scripts.train_lewm_action_prefix_joint_a4 import _evaluate  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=84)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("joint A4 endpoint evaluation permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def main() -> int:
    args = _parser().parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("batch size must be positive")
    device = _cuda0(str(args.device))
    torch.cuda.set_device(device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("joint A4 checkpoint must contain a mapping")
    protocol = payload.get("protocol")
    if protocol == JOINT_PROTOCOL:
        model = load_joint_checkpoint(payload).to(device).eval()
    elif protocol == POSE_JOINT_PROTOCOL:
        model = load_pose_conditioned_joint_checkpoint(payload).to(device).eval()
    else:
        raise ValueError("unsupported joint A4 checkpoint protocol")
    index, arrays = open_raw_replay_arrays(args.raw_dataset.expanduser().resolve())
    try:
        evaluation = _evaluate(
            model,
            index,
            arrays,
            device=device,
            batch_size=int(args.batch_size),
        )
    finally:
        del model
        torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "protocol": str(protocol),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "seed": int(payload["seed"]),
        "sigreg_weight": float(payload["sigreg_weight"]),
        "raw_dataset": str(args.raw_dataset.expanduser().resolve()),
        "action_control": "same_current_state_cyclic_branch_plus_one",
        "evaluation": evaluation,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
