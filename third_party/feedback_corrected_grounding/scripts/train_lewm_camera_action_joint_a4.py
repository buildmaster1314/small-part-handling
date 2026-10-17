#!/usr/bin/env python3
"""Fine-tune the seven-task A4 LeWM with camera-frame action prefixes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import SIGREG_WEIGHT, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    ENDPOINT_METRIC_LOSS_WEIGHT,
    GRIPPER_APERTURE_LOSS_WEIGHT,
    LOSS_PROFILE_UNIFORM,
    LOSS_PROFILES,
    METRIC_LOSS_WEIGHT,
    joint_batch_from_memmaps,
    task_balanced_sample,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_camera_action_joint_a4 import (  # noqa: E402
    CAMERA_ACTION_PROTOCOL,
    build_camera_action_joint_model,
    camera_action_checkpoint_payload,
    camera_action_statistics,
)
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402
from scripts.train_lewm_action_prefix_joint_a4 import (  # noqa: E402
    BACKBONE_LR,
    ENCODER_LR,
    GRAD_CLIP,
    HEAD_LR,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    _atomic_json,
    _atomic_torch,
    _evaluate,
    _move,
    _parameter_groups,
)


DEFAULT_RAW = ROOT / (
    "outputs/lewm_action_prefix_a4/joint_orbit12_gripper_24train_8dev"
)
DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/"
    "lewm_action_prefix_joint_latest.pt"
)


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=40_571)
    parser.add_argument("--batch-size", type=int, default=84)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--loss-profile",
        choices=LOSS_PROFILES,
        default=LOSS_PROFILE_UNIFORM,
    )
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.seed <= 0:
        parser.error("--steps and --seed must be positive")
    if args.batch_size < len(TASKS):
        parser.error("--batch-size must be at least seven")
    if args.eval_batch_size <= 0 or args.log_every <= 0:
        parser.error("evaluation batch size and log interval must be positive")
    return args


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("camera-action joint A4 training permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _validate_dataset(
    index: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray]
) -> None:
    required_arrays = {
        "camera_intrinsic",
        "camera_extrinsic_cam_from_world",
    }
    if not required_arrays <= set(arrays):
        raise ValueError("camera-action dataset lacks exact camera matrices")
    if "camera_clock_hour" not in index or set(
        np.unique(np.asarray(index["camera_clock_hour"])).tolist()
    ) != set(range(1, 13)):
        raise ValueError("camera-action dataset must cover all twelve orbit views")
    for split in ("train", "development"):
        mask = np.asarray(index["split"]).astype(str) == split
        names = set(np.asarray(index["task_name"])[mask].astype(str))
        if names != set(TASKS):
            raise ValueError(f"split {split!r} does not cover all seven tasks")
    rotation = np.asarray(
        arrays["camera_extrinsic_cam_from_world"], dtype=np.float64
    )[:, :3, :3]
    identity = np.einsum("nij,nkj->nik", rotation, rotation)
    if float(np.max(np.abs(identity - np.eye(3)))) > 1.0e-4:
        raise ValueError("camera extrinsics contain a non-orthonormal rotation")
    determinant = np.linalg.det(rotation)
    if float(np.max(np.abs(determinant - 1.0))) > 1.0e-4:
        raise ValueError("camera extrinsics contain an improper rotation")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    seed_everything(int(args.seed))

    raw_path = args.raw_dataset.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping):
        raise TypeError("camera-action parent checkpoint must contain a mapping")
    if int(parent.get("step", -1)) != 10_000:
        raise ValueError("camera-action run requires the fixed 10k parent")
    if float(parent.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("camera-action parent SIGReg must be exactly 0.02")

    statistics = camera_action_statistics(index, arrays)
    model = build_camera_action_joint_model(
        parent,
        action_statistics=statistics,
        loss_profile=args.loss_profile,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        _parameter_groups(model),
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            min(WARMUP_STEPS, int(args.steps)),
            int(args.steps),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("camera-action output directory must be empty")
    resolved = {
        "protocol": CAMERA_ACTION_PROTOCOL,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "raw_dataset": str(raw_path),
        "parent_checkpoint": str(parent_path),
        "parent_step": int(parent["step"]),
        "parameter_count": int(parameter_count),
        "sigreg_weight": SIGREG_WEIGHT,
        "metric_loss_weight": METRIC_LOSS_WEIGHT,
        "gripper_aperture_loss_weight": GRIPPER_APERTURE_LOSS_WEIGHT,
        "endpoint_metric_loss_weight": ENDPOINT_METRIC_LOSS_WEIGHT,
        "loss_profile": args.loss_profile,
        "backbone_lr": BACKBONE_LR,
        "encoder_lr": ENCODER_LR,
        "head_lr": HEAD_LR,
        "camera_conditioning": "continuous_world_to_camera_rotation_on_action",
        "action_statistics": {
            name: value.tolist() for name, value in statistics.items()
        },
        "task_balanced_every_batch": True,
        "selection": "fixed_latest",
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)

    rng = np.random.default_rng(int(args.seed))
    model.train()
    started = time.perf_counter()
    final: dict[str, Any] = {}
    for step in range(1, int(args.steps) + 1):
        bundles, branches = task_balanced_sample(
            index,
            split="train",
            batch_size=int(args.batch_size),
            rng=rng,
        )
        batch = _move(
            joint_batch_from_memmaps(
                index,
                arrays,
                bundle_indices=bundles,
                branch_indices=branches,
            ),
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = model(batch)
        if not bool(torch.isfinite(result.loss)):
            raise RuntimeError("camera-action joint A4 loss is non-finite")
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final = {
            "step": step,
            "train_loss": float(result.loss.detach()),
            "train_latent_loss": float(result.latent_loss.detach()),
            "train_metric_loss": float(result.metric_loss.detach()),
            "train_gripper_aperture_loss": float(
                result.gripper_aperture_loss.detach()
            ),
            "train_endpoint_metric_loss": float(
                result.endpoint_metric_loss.detach()
                if result.endpoint_metric_loss is not None
                else 0.0
            ),
            "train_sigreg_loss": float(result.sigreg_loss.detach()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(float(value)) for value in final.values()):
            raise RuntimeError("camera-action training metrics are non-finite")
        if step % int(args.log_every) == 0 or step == int(args.steps):
            print(json.dumps(final, sort_keys=True), flush=True)

    evaluation = _evaluate(
        model,
        index,
        arrays,
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    metrics = {
        **final,
        "protocol": CAMERA_ACTION_PROTOCOL,
        "parameter_count": int(parameter_count),
        "evaluation": evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    checkpoint = camera_action_checkpoint_payload(
        model,
        step=int(args.steps),
        seed=int(args.seed),
        parent_checkpoint=str(parent_path),
        raw_dataset=str(raw_path),
        action_statistics=statistics,
        loss_profile=args.loss_profile,
    )
    _atomic_torch(
        output_dir / "lewm_camera_action_joint_latest.pt", checkpoint
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
