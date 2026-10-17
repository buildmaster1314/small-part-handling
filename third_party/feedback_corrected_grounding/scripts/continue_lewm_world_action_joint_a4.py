#!/usr/bin/env python3
"""Matched world-action continuation for the camera-frame action control."""

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
    JOINT_PROTOCOL,
    METRIC_LOSS_WEIGHT,
    joint_batch_from_memmaps,
    joint_checkpoint_payload,
    load_joint_checkpoint,
    task_balanced_sample,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402
from scripts.train_lewm_action_prefix_joint_a4 import (  # noqa: E402
    GRAD_CLIP,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    _atomic_json,
    _atomic_torch,
    _evaluate,
    _move,
    _parameter_groups,
)
from scripts.train_lewm_camera_action_joint_a4 import (  # noqa: E402
    DEFAULT_PARENT,
    DEFAULT_RAW,
    _cuda0,
    _validate_dataset,
)


WORLD_CONTINUATION_PROTOCOL = "a4_world_action_matched_continuation_v1"


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=40_571)
    parser.add_argument("--batch-size", type=int, default=84)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.seed <= 0:
        parser.error("--steps and --seed must be positive")
    if args.batch_size < len(TASKS):
        parser.error("--batch-size must be at least seven")
    if args.eval_batch_size <= 0 or args.log_every <= 0:
        parser.error("evaluation batch size and log interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    seed_everything(int(args.seed))

    raw_path = args.raw_dataset.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("world-action continuation parent must contain a mapping")
    if checkpoint.get("protocol") != JOINT_PROTOCOL:
        raise ValueError("world-action continuation requires the joint A4 protocol")
    if int(checkpoint.get("step", -1)) != 10_000:
        raise ValueError("world-action continuation requires the fixed 10k parent")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("world-action continuation SIGReg must be exactly 0.02")
    if Path(str(checkpoint.get("raw_dataset", ""))).resolve() != raw_path:
        raise ValueError("world-action parent dataset differs from requested dataset")

    model = load_joint_checkpoint(checkpoint).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    trainable_count = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    if parameter_count != trainable_count:
        raise RuntimeError("matched world-action continuation must train every parameter")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("world-action continuation output directory must be empty")
    total_step = int(checkpoint["step"]) + int(args.steps)
    resolved = {
        "protocol": WORLD_CONTINUATION_PROTOCOL,
        "seed": int(args.seed),
        "additional_steps": int(args.steps),
        "total_step": total_step,
        "batch_size": int(args.batch_size),
        "raw_dataset": str(raw_path),
        "parent_checkpoint": str(parent_path),
        "parent_step": int(checkpoint["step"]),
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_count),
        "sigreg_weight": SIGREG_WEIGHT,
        "metric_loss_weight": METRIC_LOSS_WEIGHT,
        "gripper_aperture_loss_weight": GRIPPER_APERTURE_LOSS_WEIGHT,
        "action_frame": "world",
        "matched_camera_action_seed": True,
        "selection": "fixed_latest",
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)

    rng = np.random.default_rng(int(args.seed))
    model.train()
    started = time.perf_counter()
    final: dict[str, Any] = {}
    for local_step in range(1, int(args.steps) + 1):
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
            raise RuntimeError("world-action continuation loss is non-finite")
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final = {
            "local_step": local_step,
            "total_step": int(checkpoint["step"]) + local_step,
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
            raise RuntimeError("world-action continuation metrics are non-finite")
        if local_step % int(args.log_every) == 0 or local_step == int(args.steps):
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
        "protocol": WORLD_CONTINUATION_PROTOCOL,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_count),
        "evaluation": evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    statistics = checkpoint.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("world-action parent lacks input statistics")
    payload = joint_checkpoint_payload(
        model,
        step=total_step,
        seed=int(args.seed),
        raw_dataset=str(raw_path),
        base_checkpoint=str(checkpoint.get("base_checkpoint", "")),
        prefix_checkpoint=str(checkpoint.get("prefix_checkpoint", "")),
        statistics={name: np.asarray(value) for name, value in statistics.items()},
    )
    payload["continuation_protocol"] = WORLD_CONTINUATION_PROTOCOL
    payload["continuation_source"] = str(parent_path)
    payload["additional_steps"] = int(args.steps)
    _atomic_torch(output_dir / "lewm_action_prefix_joint_latest.pt", payload)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
