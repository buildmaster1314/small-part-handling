#!/usr/bin/env python3
"""Train one shared seven-task Fast-LeWM-style A4 dynamics head."""

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

from fcpg.lewm_a4_storage import load_a4_state_cache  # noqa: E402
from fcpg.lewm_a4_training_data import build_a4_state_cache_loader  # noqa: E402
from fcpg.lewm_action_branch_a4 import BUNDLE_BATCH_SIZE, SIGREG_WEIGHT  # noqa: E402
from fcpg.lewm_action_branch_a4_training import (  # noqa: E402
    build_a4_evaluation_loader,
)
from fcpg.lewm_action_prefix_a4 import (  # noqa: E402
    ACTION_PREFIX_MODEL_CONFIG,
    ACTION_PREFIX_PROTOCOL,
    GEOMETRY_LOSS_WEIGHT,
    action_normalization_from_cache,
    action_prefix_loss,
    build_action_prefix_model,
    checkpoint_payload,
    decoder_aligned_geometry_loss,
    evaluate_action_prefix_model,
    flatten_action_prefix_batch,
)
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402


DEFAULT_INPUT = ROOT / "outputs/lewm_action_branch_a4/formal_v2/state_cache.npz"
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
GEOMETRY_IMAGE_SIZE = (224, 224)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2_000, 10_000), default=2_000)
    parser.add_argument("--seed", type=int, default=40_401)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--geometry-decoder",
        type=Path,
        default=None,
        help=(
            "Optional train-only task-conditioned anchor-latent decoder. "
            "When supplied, the fixed decoder adds normalized pixel-space "
            "alignment with weight 10.0; it is never updated."
        ),
    )
    args = parser.parse_args(argv)
    if args.seed <= 0:
        parser.error("--seed must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    return args


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or (device.index not in (None, 0)):
        raise ValueError("action-prefix training permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _load_geometry_coefficient(path: Path) -> np.ndarray:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as values:
        required = {
            "fit_split",
            "evaluation_split",
            "task_names",
            "anchor_slots",
            "coefficient_raw",
        }
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(f"geometry decoder is missing fields: {missing}")
        if str(values["fit_split"].item()) != "train":
            raise ValueError("geometry decoder must be fitted on train only")
        if str(values["evaluation_split"].item()) != "development":
            raise ValueError("geometry decoder must be selected on development")
        task_names = tuple(np.asarray(values["task_names"]).astype(str).tolist())
        from fcpg.lewm_action_branch_a4 import TASKS

        if task_names != TASKS:
            raise ValueError("geometry decoder task panel drifted")
        anchor_slots = tuple(
            int(value) for value in np.asarray(values["anchor_slots"]).tolist()
        )
        if anchor_slots != (1, 2, 3):
            raise ValueError("geometry decoder must read the three anchor slots")
        coefficient = np.asarray(values["coefficient_raw"], dtype=np.float32)
    expected = (7, 3 * 192, 2)
    if coefficient.shape != expected:
        raise ValueError(
            f"geometry coefficient must have shape {expected}, got {coefficient.shape}"
        )
    if not np.isfinite(coefficient).all():
        raise ValueError("geometry coefficient contains non-finite values")
    return np.ascontiguousarray(coefficient)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = _cuda0(args.device)
    seed_everything(int(args.seed))
    torch.cuda.set_device(device)

    input_path = args.input_npz.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    cache = load_a4_state_cache(input_path)
    if bool((cache["split"].astype(str) == "test").any()):
        raise ValueError("action-prefix training must not open reserved test rows")
    action_mean, action_std = action_normalization_from_cache(cache)
    geometry_coefficient = None
    geometry_decoder_path = None
    if args.geometry_decoder is not None:
        geometry_decoder_path = args.geometry_decoder.expanduser().resolve()
        geometry_coefficient = torch.from_numpy(
            _load_geometry_coefficient(geometry_decoder_path)
        ).to(device=device)
    model = build_action_prefix_model(
        seed=int(args.seed),
        action_mean=action_mean,
        action_std=action_std,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_loader = build_a4_state_cache_loader(
        cache,
        split="train",
        seed=int(args.seed),
        dtype=torch.float32,
        batches_per_epoch=int(args.steps),
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    development_loader = build_a4_evaluation_loader(
        cache,
        split="development",
        batch_size=BUNDLE_BATCH_SIZE,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
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
    existing = sorted(path.name for path in output_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"action-prefix output directory must be empty, found {existing}"
        )
    resolved = {
        "schema_version": 1,
        "protocol": ACTION_PREFIX_PROTOCOL,
        "input_npz": str(input_path),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "optimizer_steps": int(args.steps),
        "model_config": dict(ACTION_PREFIX_MODEL_CONFIG),
        "parameter_count": int(parameter_count),
        "bundle_batch_size": BUNDLE_BATCH_SIZE,
        "effective_action_sequence_batch_size": int(BUNDLE_BATCH_SIZE * 8),
        "objective": (
            "dense_prefix_h1_h8_weighted_latent_mse"
            if geometry_coefficient is None
            else "dense_prefix_latent_mse_plus_fixed_decoder_geometry"
        ),
        "geometry_decoder": (
            None if geometry_decoder_path is None else str(geometry_decoder_path)
        ),
        "geometry_loss_weight": (
            0.0 if geometry_coefficient is None else GEOMETRY_LOSS_WEIGHT
        ),
        "geometry_image_size": list(GEOMETRY_IMAGE_SIZE),
        "anchor_encoder_sigreg": SIGREG_WEIGHT,
        "prediction_head_adds_sigreg": False,
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "amp_dtype": "float16",
        "selection": "fixed_latest",
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)

    model.train()
    started = time.perf_counter()
    last_loss = float("nan")
    last_latent_loss = float("nan")
    last_geometry_loss = 0.0
    for step, host_batch in enumerate(train_loader, start=1):
        batch = {
            name: (
                value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
            )
            for name, value in host_batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            latent_loss, prediction = action_prefix_loss(model, batch)
        geometry_loss = latent_loss.new_zeros(())
        if geometry_coefficient is not None:
            flat = flatten_action_prefix_batch(batch)
            geometry_loss = decoder_aligned_geometry_loss(
                prediction,
                flat.target_state,
                flat.task_index,
                geometry_coefficient,
                image_size=GEOMETRY_IMAGE_SIZE,
            )
        loss = latent_loss + GEOMETRY_LOSS_WEIGHT * geometry_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("action-prefix training loss is non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        last_loss = float(loss.detach())
        last_latent_loss = float(latent_loss.detach())
        last_geometry_loss = float(geometry_loss.detach())
        if step % int(args.log_every) == 0 or step == int(args.steps):
            record = {
                "step": int(step),
                "optimizer_steps": int(args.steps),
                "train_loss": last_loss,
                "train_latent_loss": last_latent_loss,
                "train_geometry_loss": last_geometry_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            if not all(math.isfinite(value) for value in record.values()):
                raise RuntimeError("action-prefix training metrics are non-finite")
            print(json.dumps(record, sort_keys=True), flush=True)
    if step != int(args.steps):
        raise RuntimeError("action-prefix loader ended before fixed endpoint")

    evaluation = evaluate_action_prefix_model(
        model,
        development_loader,
        device=device,
        amp_enabled=True,
        geometry_coefficient=geometry_coefficient,
    )
    metrics = {
        "protocol": ACTION_PREFIX_PROTOCOL,
        "seed": int(args.seed),
        "step": int(args.steps),
        "parameter_count": int(parameter_count),
        "train_loss": last_loss,
        "train_latent_loss": last_latent_loss,
        "train_geometry_loss": last_geometry_loss,
        "evaluation": evaluation.metrics,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    checkpoint = checkpoint_payload(
        model,
        seed=int(args.seed),
        step=int(args.steps),
        input_npz=str(input_path),
    )
    checkpoint["geometry_decoder"] = (
        None if geometry_decoder_path is None else str(geometry_decoder_path)
    )
    checkpoint["geometry_loss_weight"] = (
        0.0 if geometry_coefficient is None else GEOMETRY_LOSS_WEIGHT
    )
    _atomic_torch(output_dir / "lewm_action_prefix_latest.pt", checkpoint)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
