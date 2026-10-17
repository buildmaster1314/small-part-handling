#!/usr/bin/env python3
"""Train the expanded camera-action seven-task A4 LeWM on CUDA0."""

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
    GRIPPER_APERTURE_LOSS_WEIGHT,
    METRIC_LOSS_WEIGHT,
    joint_batch_from_memmaps,
    task_balanced_sample,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_camera_action_joint_a4 import (  # noqa: E402
    EXPANDED_CAMERA_ACTION_PROTOCOL,
    EXPANDED_DYNAMICS_CONFIG,
    EXPANDED_RESIDUAL_SCALE,
    ExpandedActionPrefixDynamics,
    build_expanded_camera_action_joint_model,
    expanded_camera_action_checkpoint_payload,
)
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402
from scripts.train_lewm_action_prefix_joint_a4 import (  # noqa: E402
    GRAD_CLIP,
    WEIGHT_DECAY,
    _atomic_json,
    _atomic_torch,
    _evaluate,
    _move,
)
from scripts.train_lewm_camera_action_joint_a4 import (  # noqa: E402
    DEFAULT_RAW,
    _cuda0,
    _validate_dataset,
)


DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/camera_action_seed40571_2k/"
    "lewm_camera_action_joint_latest.pt"
)
EXPANDED_BACKBONE_LR = 5.0e-6
EXPANDED_ENCODER_LR = 1.0e-5
BASE_DYNAMICS_LR = 1.0e-5
EXPANSION_LR = 1.0e-4
READOUT_LR = 5.0e-5
EXPANDED_WARMUP_STEPS = 200


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=40_572)
    parser.add_argument("--micro-batch-size", type=int, default=42)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.seed <= 0:
        parser.error("--steps and --seed must be positive")
    if args.micro_batch_size < len(TASKS):
        parser.error("--micro-batch-size must be at least seven")
    if args.gradient_accumulation <= 0:
        parser.error("--gradient-accumulation must be positive")
    if args.eval_batch_size <= 0 or args.log_every <= 0:
        parser.error("evaluation batch size and log interval must be positive")
    return args


def _parameter_groups(model: nn.Module) -> list[dict[str, Any]]:
    dynamics = model.dynamics
    if not isinstance(dynamics, ExpandedActionPrefixDynamics):
        raise TypeError("expanded trainer requires ExpandedActionPrefixDynamics")
    vision = getattr(model.frame_encoder, "vision_encoder", None)
    backbone = getattr(vision, "backbone", None)
    backbone_parameters = (
        list(backbone.parameters()) if isinstance(backbone, nn.Module) else []
    )
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    encoder_parameters = [
        parameter
        for module in (model.frame_encoder, model.state_projector)
        for parameter in module.parameters()
        if id(parameter) not in backbone_ids
    ]
    groups = []
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": EXPANDED_BACKBONE_LR,
                "name": "vit",
            }
        )
    groups.extend(
        [
            {
                "params": encoder_parameters,
                "lr": EXPANDED_ENCODER_LR,
                "name": "frame_encoder",
            },
            {
                "params": list(dynamics.base_dynamics.parameters()),
                "lr": BASE_DYNAMICS_LR,
                "name": "proven_dynamics",
            },
            {
                "params": list(dynamics.expansion.parameters()),
                "lr": EXPANSION_LR,
                "name": "expanded_dynamics",
            },
            {
                "params": list(model.metric_readout.parameters()),
                "lr": READOUT_LR,
                "name": "metric_readout",
            },
        ]
    )
    flattened = [parameter for group in groups for parameter in group["params"]]
    if len(flattened) != len({id(parameter) for parameter in flattened}):
        raise RuntimeError("expanded optimizer parameter groups overlap")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in flattened} != {
        id(parameter) for parameter in trainable
    }:
        raise RuntimeError("expanded optimizer misses trainable parameters")
    return groups


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
        raise TypeError("expanded parent checkpoint must contain a mapping")
    if int(parent.get("step", -1)) != 2_000:
        raise ValueError("expanded run requires the fixed camera-action 2k parent")
    if float(parent.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("expanded parent SIGReg must be exactly 0.02")
    model = build_expanded_camera_action_joint_model(
        parent,
        seed=int(args.seed),
        expansion_config=EXPANDED_DYNAMICS_CONFIG,
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
            step, min(EXPANDED_WARMUP_STEPS, int(args.steps)), int(args.steps)
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("expanded camera-action output directory must be empty")
    effective_batch = int(args.micro_batch_size) * int(args.gradient_accumulation)
    resolved = {
        "protocol": EXPANDED_CAMERA_ACTION_PROTOCOL,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation": int(args.gradient_accumulation),
        "effective_batch_size": effective_batch,
        "raw_dataset": str(raw_path),
        "parent_checkpoint": str(parent_path),
        "parent_step": int(parent["step"]),
        "parameter_count": int(parameter_count),
        "sigreg_weight": SIGREG_WEIGHT,
        "metric_loss_weight": METRIC_LOSS_WEIGHT,
        "gripper_aperture_loss_weight": GRIPPER_APERTURE_LOSS_WEIGHT,
        "camera_conditioning": "continuous_world_to_camera_rotation_on_action",
        "capacity_strategy": "proven_base_plus_zero_initialized_residual",
        "residual_scale": EXPANDED_RESIDUAL_SCALE,
        "expansion_config": dict(EXPANDED_DYNAMICS_CONFIG),
        "backbone_lr": EXPANDED_BACKBONE_LR,
        "encoder_lr": EXPANDED_ENCODER_LR,
        "base_dynamics_lr": BASE_DYNAMICS_LR,
        "expansion_lr": EXPANSION_LR,
        "readout_lr": READOUT_LR,
        "warmup_steps": min(EXPANDED_WARMUP_STEPS, int(args.steps)),
        "selection": "fixed_latest",
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)

    rng = np.random.default_rng(int(args.seed))
    model.train()
    started = time.perf_counter()
    final: dict[str, Any] = {}
    accumulation = int(args.gradient_accumulation)
    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_metrics = []
        for _micro_step in range(accumulation):
            bundles, branches = task_balanced_sample(
                index,
                split="train",
                batch_size=int(args.micro_batch_size),
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = model(batch)
                scaled_loss = result.loss / accumulation
            if not bool(torch.isfinite(result.loss)):
                raise RuntimeError("expanded camera-action loss is non-finite")
            scaler.scale(scaled_loss).backward()
            micro_metrics.append(result)
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final = {
            "step": step,
            "train_loss": float(
                np.mean([float(value.loss.detach()) for value in micro_metrics])
            ),
            "train_latent_loss": float(
                np.mean(
                    [float(value.latent_loss.detach()) for value in micro_metrics]
                )
            ),
            "train_metric_loss": float(
                np.mean(
                    [float(value.metric_loss.detach()) for value in micro_metrics]
                )
            ),
            "train_sigreg_loss": float(
                np.mean(
                    [float(value.sigreg_loss.detach()) for value in micro_metrics]
                )
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(float(value)) for value in final.values()):
            raise RuntimeError("expanded camera-action metrics are non-finite")
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
        "protocol": EXPANDED_CAMERA_ACTION_PROTOCOL,
        "parameter_count": int(parameter_count),
        "evaluation": evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    checkpoint = expanded_camera_action_checkpoint_payload(
        model,
        step=int(args.steps),
        seed=int(args.seed),
        parent_checkpoint=str(parent_path),
        raw_dataset=str(raw_path),
        expansion_config=EXPANDED_DYNAMICS_CONFIG,
    )
    _atomic_torch(
        output_dir / "lewm_camera_action_expanded_joint_latest.pt", checkpoint
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
