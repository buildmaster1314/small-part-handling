#!/usr/bin/env python3
"""Continue the existing A4 camera-action LeWM without adding parameters."""

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
    CAMERA_ACTION_PROTOCOL,
    camera_action_checkpoint_payload,
    load_camera_action_joint_checkpoint,
)
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
    DEFAULT_RAW,
    _cuda0,
    _validate_dataset,
)


CONTINUATION_PROTOCOL = "a4_camera_action_full_finetune_continuation_v1"
DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/camera_action_seed40571_2k/"
    "lewm_camera_action_joint_latest.pt"
)


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=40_581)
    parser.add_argument("--batch-size", type=int, default=84)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
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
    source_path = args.parent_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("camera-action continuation source must be a mapping")
    if checkpoint.get("protocol") != CAMERA_ACTION_PROTOCOL:
        raise ValueError("continuation source is not a camera-action checkpoint")
    if int(checkpoint.get("step", -1)) != 2_000:
        raise ValueError("continuation requires the fixed camera-action 2k source")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("continuation SIGReg must be exactly 0.02")
    if Path(str(checkpoint.get("raw_dataset", ""))).resolve() != raw_path:
        raise ValueError("continuation source dataset differs from requested dataset")
    model = load_camera_action_joint_checkpoint(checkpoint).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    trainable_count = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    if parameter_count != trainable_count:
        raise RuntimeError("full continuation must train every existing parameter")
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
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=1024.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("continuation output directory must be empty")
    total_step = int(checkpoint["step"]) + int(args.steps)
    resolved = {
        "protocol": CONTINUATION_PROTOCOL,
        "seed": int(args.seed),
        "additional_steps": int(args.steps),
        "total_step": total_step,
        "batch_size": int(args.batch_size),
        "raw_dataset": str(raw_path),
        "continuation_source": str(source_path),
        "source_step": int(checkpoint["step"]),
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_count),
        "sigreg_weight": SIGREG_WEIGHT,
        "sigreg_gradient_active": True,
        "metric_loss_weight": METRIC_LOSS_WEIGHT,
        "gripper_aperture_loss_weight": GRIPPER_APERTURE_LOSS_WEIGHT,
        "selection": "fixed_latest",
        "task_balanced_every_batch": True,
        "optimizer_state_restarted": True,
        "amp_initial_scale": 1024.0,
        "cuda_device": 0,
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
            raise RuntimeError("camera-action continuation loss is non-finite")
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("camera-action continuation gradient is non-finite")
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
            "gradient_norm": float(gradient_norm.detach()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(float(value)) for value in final.values()):
            raise RuntimeError("camera-action continuation metrics are non-finite")
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
        "protocol": CONTINUATION_PROTOCOL,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_count),
        "evaluation": evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    statistics = checkpoint.get("action_statistics")
    source_parent = checkpoint.get("parent_checkpoint")
    if not isinstance(statistics, Mapping) or not isinstance(source_parent, str):
        raise ValueError("continuation source lacks reconstruction metadata")
    payload = camera_action_checkpoint_payload(
        model,
        step=total_step,
        seed=int(args.seed),
        parent_checkpoint=source_parent,
        raw_dataset=str(raw_path),
        action_statistics=statistics,
    )
    payload["continuation_protocol"] = CONTINUATION_PROTOCOL
    payload["continuation_source"] = str(source_path)
    payload["additional_steps"] = int(args.steps)
    _atomic_torch(output_dir / "lewm_camera_action_joint_latest.pt", payload)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
