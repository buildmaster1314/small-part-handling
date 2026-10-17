#!/usr/bin/env python3
"""Jointly fine-tune the seven-task A4 LeWM with continuous camera pose."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
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
    GRIPPER_APERTURE_LOSS_WEIGHT,
    METRIC_LOSS_WEIGHT,
    joint_batch_from_memmaps,
    task_balanced_sample,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_pose_conditioned_joint_a4 import (  # noqa: E402
    POSE_JOINT_PROTOCOL,
    POSE_JOINT_SIGREG_WEIGHTS,
    build_pose_conditioned_joint_model,
    pose_joint_checkpoint_payload,
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
    parser.add_argument("--seed", type=int, default=40_551)
    parser.add_argument("--batch-size", type=int, default=84)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--camera-hidden-dim", type=int, default=256)
    parser.add_argument("--readout-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        choices=POSE_JOINT_SIGREG_WEIGHTS,
        default=SIGREG_WEIGHT,
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.seed <= 0:
        parser.error("--steps and --seed must be positive")
    if args.batch_size < len(TASKS):
        parser.error("--batch-size must be at least seven")
    if args.eval_batch_size <= 0 or args.log_every <= 0:
        parser.error("evaluation batch size and log interval must be positive")
    if args.camera_hidden_dim <= 0 or args.readout_hidden_dim <= 0:
        parser.error("camera/readout hidden dimensions must be positive")
    return args


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("pose-conditioned joint A4 training permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _parameter_groups(model: nn.Module) -> list[dict[str, Any]]:
    vision = getattr(model.frame_encoder, "vision_encoder", None)
    backbone = getattr(vision, "backbone", None)
    backbone_parameters = (
        list(backbone.parameters()) if isinstance(backbone, nn.Module) else []
    )
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    encoder_modules = (
        model.frame_encoder,
        model.state_projector,
        model.camera_conditioner,
    )
    encoder_parameters = [
        parameter
        for module in encoder_modules
        for parameter in module.parameters()
        if id(parameter) not in backbone_ids
    ]
    head_parameters = [
        parameter
        for module in (model.dynamics, model.metric_readout, model.pose_readout)
        for parameter in module.parameters()
    ]
    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append(
            {"params": backbone_parameters, "lr": BACKBONE_LR, "name": "vit_backbone"}
        )
    groups.append(
        {"params": encoder_parameters, "lr": ENCODER_LR, "name": "pose_encoder"}
    )
    groups.append(
        {"params": head_parameters, "lr": HEAD_LR, "name": "dynamics_and_readout"}
    )
    flattened = [parameter for group in groups for parameter in group["params"]]
    if len(flattened) != len({id(parameter) for parameter in flattened}):
        raise RuntimeError("pose-conditioned optimizer parameter groups overlap")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in flattened} != {
        id(parameter) for parameter in trainable
    }:
        raise RuntimeError("pose-conditioned optimizer misses trainable parameters")
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    seed_everything(int(args.seed))
    raw_path = args.raw_dataset.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    required_camera = {
        "camera_clock_hour",
        "camera_elevation_degrees",
        "camera_radius_m",
    }
    if not required_camera <= set(index):
        raise ValueError("pose-conditioned dataset lacks continuous camera metadata")
    if set(np.unique(np.asarray(index["camera_clock_hour"])).tolist()) != set(
        range(1, 13)
    ):
        raise ValueError("pose-conditioned dataset must cover all twelve orbit views")
    for split in ("train", "development"):
        mask = np.asarray(index["split"]).astype(str) == split
        names = set(np.asarray(index["task_name"])[mask].astype(str))
        if names != set(TASKS):
            raise ValueError(f"split {split!r} does not cover all seven tasks")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping):
        raise TypeError("parent joint checkpoint must contain a mapping")
    if int(parent.get("step", -1)) != 10_000:
        raise ValueError("pose-conditioned run requires the fixed 10k parent")
    if float(parent.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("parent SIGReg must be exactly 0.02")
    model = build_pose_conditioned_joint_model(
        parent,
        seed=int(args.seed),
        camera_hidden_dim=int(args.camera_hidden_dim),
        readout_hidden_dim=int(args.readout_hidden_dim),
        sigreg_weight=float(args.sigreg_weight),
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
        raise FileExistsError("pose-conditioned output directory must be empty")
    resolved = {
        "protocol": POSE_JOINT_PROTOCOL,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "raw_dataset": str(raw_path),
        "parent_checkpoint": str(parent_path),
        "parent_step": int(parent["step"]),
        "parameter_count": int(parameter_count),
        "sigreg_weight": float(args.sigreg_weight),
        "metric_loss_weight": METRIC_LOSS_WEIGHT,
        "gripper_aperture_loss_weight": GRIPPER_APERTURE_LOSS_WEIGHT,
        "backbone_lr": BACKBONE_LR,
        "encoder_lr": ENCODER_LR,
        "head_lr": HEAD_LR,
        "camera_hidden_dim": int(args.camera_hidden_dim),
        "readout_hidden_dim": int(args.readout_hidden_dim),
        "camera_conditioning": "continuous_world_to_camera_basis_slot_film",
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
            raise RuntimeError("pose-conditioned joint A4 loss is non-finite")
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
            "train_sigreg_loss": float(result.sigreg_loss.detach()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(float(value)) for value in final.values()):
            raise RuntimeError("pose-conditioned training metrics are non-finite")
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
        "protocol": POSE_JOINT_PROTOCOL,
        "parameter_count": int(parameter_count),
        "evaluation": evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    checkpoint = pose_joint_checkpoint_payload(
        model,
        step=int(args.steps),
        seed=int(args.seed),
        parent_checkpoint=str(parent_path),
        raw_dataset=str(raw_path),
    )
    _atomic_torch(output_dir / "lewm_pose_conditioned_joint_latest.pt", checkpoint)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
