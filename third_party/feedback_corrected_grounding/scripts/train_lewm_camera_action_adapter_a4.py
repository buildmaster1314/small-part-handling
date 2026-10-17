#!/usr/bin/env python3
"""Train a bounded, parameter-matched adapter on the frozen A4 LeWM."""

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
from fcpg.lewm_camera_action_adapter_a4 import (  # noqa: E402
    ADAPTER_PROTOCOL,
    ADAPTER_VARIANTS,
    BoundedAnchorAdapterDynamics,
    adapter_checkpoint_payload,
    build_camera_action_adapter_model,
)
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402
from scripts.train_lewm_action_prefix_joint_a4 import (  # noqa: E402
    GRAD_CLIP,
    HEAD_LR,
    WARMUP_STEPS,
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
DEFAULT_CAP = ROOT / (
    "outputs/lewm_action_prefix_a4/camera_action_seed40571_2k/"
    "train_residual_cap_q99.npz"
)


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--residual-cap", type=Path, default=DEFAULT_CAP)
    parser.add_argument("--variant", choices=ADAPTER_VARIANTS, required=True)
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


def _load_cap(path: Path, *, parent_path: Path, raw_path: Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "residual_cap",
            "quantile",
            "floor",
            "train_pair_count",
            "parent_checkpoint",
            "raw_dataset",
        }
        if not required <= set(payload.files):
            raise ValueError("residual cap artifact is incomplete")
        if Path(str(payload["parent_checkpoint"].item())).resolve() != parent_path:
            raise ValueError("residual cap parent differs from training parent")
        if Path(str(payload["raw_dataset"].item())).resolve() != raw_path:
            raise ValueError("residual cap dataset differs from training dataset")
        cap = torch.from_numpy(np.asarray(payload["residual_cap"], dtype=np.float32))
    if cap.shape != (8, 4, 192):
        raise ValueError("residual cap must have shape [8,4,192]")
    if not bool(torch.isfinite(cap).all()) or bool(torch.any(cap <= 0.0)):
        raise ValueError("residual cap must be finite and positive")
    return cap


def _trainable_adapter(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    dynamics = getattr(model, "dynamics", None)
    if not isinstance(dynamics, BoundedAnchorAdapterDynamics):
        raise TypeError("adapter training requires bounded adapter dynamics")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    expected = list(dynamics.adapter.parameters())
    if {id(value) for value in trainable} != {id(value) for value in expected}:
        raise RuntimeError("only adapter parameters may be trainable")
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    return trainable, names


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    seed_everything(int(args.seed))

    raw_path = args.raw_dataset.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    cap_path = args.residual_cap.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("camera-action parent checkpoint must contain a mapping")
    if int(checkpoint.get("step", -1)) != 2_000:
        raise ValueError("adapter training requires the fixed camera-action 2k parent")
    if float(checkpoint.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("adapter parent SIGReg must be exactly 0.02")
    cap = _load_cap(cap_path, parent_path=parent_path, raw_path=raw_path)
    model = build_camera_action_adapter_model(
        checkpoint,
        variant=str(args.variant),
        residual_cap=cap,
        seed=int(args.seed),
    ).to(device)
    trainable, trainable_names = _trainable_adapter(model)
    trainable_count = sum(value.numel() for value in trainable)
    total_count = sum(value.numel() for value in model.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=HEAD_LR,
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
        raise FileExistsError("adapter output directory must be empty")
    resolved = {
        "protocol": ADAPTER_PROTOCOL,
        "variant": str(args.variant),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "raw_dataset": str(raw_path),
        "parent_checkpoint": str(parent_path),
        "parent_step": int(checkpoint["step"]),
        "residual_cap": str(cap_path),
        "residual_cap_min": float(cap.min()),
        "residual_cap_median": float(cap.median()),
        "residual_cap_max": float(cap.max()),
        "parameter_count": int(total_count),
        "trainable_parameter_count": int(trainable_count),
        "trainable_parameter_names": trainable_names,
        "sigreg_weight": SIGREG_WEIGHT,
        "sigreg_gradient_active": False,
        "metric_loss_weight": METRIC_LOSS_WEIGHT,
        "gripper_aperture_loss_weight": GRIPPER_APERTURE_LOSS_WEIGHT,
        "learning_rate": HEAD_LR,
        "selection": "fixed_latest",
        "task_balanced_every_batch": True,
        "cuda_device": 0,
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)

    rng = np.random.default_rng(int(args.seed))
    model.eval()
    dynamics = model.dynamics
    assert isinstance(dynamics, BoundedAnchorAdapterDynamics)
    dynamics.adapter.train()
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
            raise RuntimeError("bounded adapter loss is non-finite")
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("bounded adapter gradient is non-finite")
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
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(float(value)) for value in final.values()):
            raise RuntimeError("bounded adapter metrics are non-finite")
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
        "protocol": ADAPTER_PROTOCOL,
        "variant": str(args.variant),
        "parameter_count": int(total_count),
        "trainable_parameter_count": int(trainable_count),
        "evaluation": evaluation,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    payload = adapter_checkpoint_payload(
        model,
        step=int(args.steps),
        seed=int(args.seed),
        parent_checkpoint=str(parent_path),
        raw_dataset=str(raw_path),
    )
    _atomic_torch(output_dir / "lewm_camera_action_adapter_latest.pt", payload)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
