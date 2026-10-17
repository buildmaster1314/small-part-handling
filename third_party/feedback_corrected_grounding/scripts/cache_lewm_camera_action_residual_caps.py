#!/usr/bin/env python3
"""Estimate train-only latent residual caps for bounded A4 adapters."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import SIGREG_WEIGHT  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    joint_batch_from_memmaps,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_camera_action_joint_a4 import (  # noqa: E402
    CameraActionJointSevenTaskLeWM,
    load_camera_action_joint_checkpoint,
)
from scripts.train_lewm_action_prefix_joint_a4 import _move  # noqa: E402
from scripts.train_lewm_camera_action_joint_a4 import (  # noqa: E402
    DEFAULT_RAW,
    _cuda0,
    _validate_dataset,
)


DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/camera_action_seed40571_2k/"
    "lewm_camera_action_joint_latest.pt"
)


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--floor", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not 0.5 <= args.quantile < 1.0:
        parser.error("--quantile must be in [0.5,1.0)")
    if args.floor <= 0.0 or args.batch_size <= 0:
        parser.error("--floor and --batch-size must be positive")
    return args


def _encode_states(
    model: CameraActionJointSevenTaskLeWM,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    base = model.base_model
    embedding = base.encode_observations(
        rgb=torch.cat([batch["current_rgb"][:, None], batch["future_rgb"]], dim=1),
        proprio=torch.cat(
            [batch["current_proprio"][:, None], batch["future_proprio"]], dim=1
        ),
        q_nominal=torch.cat(
            [batch["current_q_nominal"][:, None], batch["future_q_nominal"]],
            dim=1,
        ),
        q_corrected=torch.cat(
            [
                batch["current_q_corrected"][:, None],
                batch["future_q_corrected"],
            ],
            dim=1,
        ),
        keypoints_xyz=torch.cat(
            [
                batch["current_keypoints_xyz"][:, None],
                batch["future_keypoints_xyz"],
            ],
            dim=1,
        ),
        valid=torch.cat(
            [batch["current_valid"][:, None], batch["future_valid"]], dim=1
        ),
    )
    current_state = embedding[:, 0]
    target_state = embedding[:, 1:]
    transformed_action = model.transform_future_action(batch["future_action"], batch)
    predicted_state = base.dynamics(
        current_state,
        transformed_action.to(dtype=current_state.dtype),
        task_index=batch["task_index"],
    )
    return predicted_state, target_state


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    raw_path = args.raw_dataset.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"residual cap output already exists: {output_path}")

    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("camera-action checkpoint must contain a mapping")
    if int(checkpoint.get("step", -1)) != 2_000:
        raise ValueError("bounded adapters require the fixed camera-action 2k parent")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("residual cap parent SIGReg must be exactly 0.02")
    model = load_camera_action_joint_checkpoint(checkpoint).to(device).eval()

    train = np.flatnonzero(np.asarray(index["split"]).astype(str) == "train")
    pairs = np.asarray(
        [(int(bundle), branch) for bundle in train for branch in range(8)],
        dtype=np.int64,
    )
    residuals: list[np.ndarray] = []
    for start in range(0, len(pairs), int(args.batch_size)):
        selected = pairs[start : start + int(args.batch_size)]
        batch = _move(
            joint_batch_from_memmaps(
                index,
                arrays,
                bundle_indices=selected[:, 0],
                branch_indices=selected[:, 1],
            ),
            device,
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            predicted, target = _encode_states(model, batch)
        residuals.append(
            (target.float() - predicted.float()).abs().cpu().numpy().astype(
                np.float16
            )
        )
        if start == 0 or (start // int(args.batch_size) + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "processed": min(start + len(selected), len(pairs)),
                        "total": int(len(pairs)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    stacked = np.concatenate(residuals, axis=0).astype(np.float32)
    cap = np.maximum(
        np.quantile(stacked, float(args.quantile), axis=0),
        float(args.floor),
    ).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        residual_cap=cap,
        quantile=np.asarray(float(args.quantile), dtype=np.float64),
        floor=np.asarray(float(args.floor), dtype=np.float64),
        train_pair_count=np.asarray(len(pairs), dtype=np.int64),
        parent_checkpoint=np.asarray(str(parent_path)),
        raw_dataset=np.asarray(str(raw_path)),
    )
    summary = {
        "output": str(output_path),
        "shape": list(cap.shape),
        "quantile": float(args.quantile),
        "floor": float(args.floor),
        "train_pair_count": int(len(pairs)),
        "cap_min": float(cap.min()),
        "cap_median": float(np.median(cap)),
        "cap_max": float(cap.max()),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
