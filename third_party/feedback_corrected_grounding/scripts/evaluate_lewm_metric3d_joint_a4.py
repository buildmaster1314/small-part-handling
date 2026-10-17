#!/usr/bin/env python3
"""Evaluate a metric3d checkpoint with latent-bypass controls on CUDA0."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_metric3d_joint_a4 import (  # noqa: E402
    METRIC3D_PROTOCOL,
    METRIC3D_WORLD_PROTOCOL,
    load_metric3d_checkpoint,
)
from scripts.train_lewm_camera_action_joint_a4 import _validate_dataset  # noqa: E402
from scripts.train_lewm_metric3d_joint_a4 import (  # noqa: E402
    _atomic_json,
    _cuda0,
    _evaluate,
)


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if int(args.batch_size) <= 1:
        parser.error("--batch-size must exceed one for shuffled-latent control")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("metric3d checkpoint must contain a mapping")
    model = load_metric3d_checkpoint(checkpoint).to(device).eval()
    raw_path = Path(str(checkpoint.get("raw_dataset", ""))).expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    evaluation = _evaluate(
        model,
        index,
        arrays,
        device=device,
        batch_size=int(args.batch_size),
    )
    payload = {
        "protocol": (
            f"{METRIC3D_WORLD_PROTOCOL}_latent_bypass_audit_v1"
            if checkpoint.get("protocol") == METRIC3D_WORLD_PROTOCOL
            else f"{METRIC3D_PROTOCOL}_latent_bypass_audit_v1"
        ),
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint["step"]),
        "seed": int(checkpoint["seed"]),
        "sigreg_weight": float(checkpoint["sigreg_weight"]),
        "controls": {
            "static_latent": "repeat current state while preserving action geometry",
            "shuffled_latent": (
                "cyclically shift predicted states across the eight branch-aligned "
                "rows while preserving each row's action geometry"
            ),
        },
        "evaluation": evaluation,
    }
    output = args.output.expanduser().resolve()
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "world_epe_mm": evaluation["task_macro_world_epe_mm"],
                "static_penalty_mm": evaluation[
                    "task_macro_static_latent_world_penalty_mm"
                ],
                "shuffled_penalty_mm": evaluation[
                    "task_macro_shuffled_latent_world_penalty_mm"
                ],
                "pixel_epe": evaluation["task_macro_pixel_epe"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
