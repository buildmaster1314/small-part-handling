#!/usr/bin/env python3
"""Train matched factual/no-action dynamics heads in one frozen latent space."""

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

import torch
from torch import nn
from torch.nn import functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_anchor_dynamics_pair import (  # noqa: E402
    MATCHED_DYNAMICS_PAIR_CHECKPOINT_TYPE,
    MatchedAnchorDynamicsPredictor,
    build_matched_predictor_pair,
    matched_pair_forward,
    predictor_kwargs_from_base_checkpoint,
    trainable_parameter_count,
)
from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM  # noqa: E402
from fcpg.lewm_runtime import (  # noqa: E402
    build_lewm_data,
    choose_device,
    cosine_warmup_lambda,
    load_lewm_grounded_checkpoint,
    move_batch_to_device,
    seed_everything,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("pair training config must contain a mapping")
    required = {"seed", "base_checkpoint", "output_dir", "optimizer", "trainer"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"pair training config is missing {sorted(missing)}")
    return dict(payload)


def _amp_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    raise ValueError("amp_dtype must be bfloat16 or float16")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _freeze_base(model: AnchorPersistentLeWM) -> AnchorPersistentLeWM:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _latent_batch(
    base: AnchorPersistentLeWM,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        _, embedding = base.encode_frames(batch)
    history_size = int(base.history_size)
    context = embedding[:, :history_size].detach()
    target = embedding[:, 1 : history_size + 1].detach()
    action = batch["action"][:, :history_size].float()
    return context, target, action


@torch.no_grad()
def _evaluate_pair(
    base: AnchorPersistentLeWM,
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    loader: Any,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> dict[str, float]:
    base.eval()
    factual.eval()
    no_action.eval()
    totals = {"factual_mse": 0.0, "no_action_mse": 0.0, "copy_mse": 0.0}
    samples = 0
    batches = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch_to_device(host_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            context, target, action = _latent_batch(base, batch)
            factual_loss = factual(context, action, target).prediction_loss
            no_action_loss = no_action(context, action, target).prediction_loss
            copy_loss = F.mse_loss(context, target)
        batch_size = int(context.shape[0])
        totals["factual_mse"] += float(factual_loss) * batch_size
        totals["no_action_mse"] += float(no_action_loss) * batch_size
        totals["copy_mse"] += float(copy_loss) * batch_size
        samples += batch_size
        batches += 1
    if samples == 0:
        raise RuntimeError("validation loader produced no samples")
    metrics = {key: value / samples for key, value in totals.items()}
    metrics.update({"samples": float(samples), "batches": float(batches)})
    metrics["factual_relative_improvement"] = (
        metrics["no_action_mse"] - metrics["factual_mse"]
    ) / max(metrics["no_action_mse"], 1e-12)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(args.device)

    base_checkpoint = Path(str(config["base_checkpoint"])).expanduser().resolve()
    base, resolved, checkpoint = load_lewm_grounded_checkpoint(
        base_checkpoint, device=device
    )
    if not isinstance(base, AnchorPersistentLeWM):
        raise TypeError("base checkpoint must contain AnchorPersistentLeWM")
    base = _freeze_base(base)

    loader_config = config.get("loader")
    if loader_config is not None:
        if not isinstance(loader_config, Mapping):
            raise TypeError("loader override must be a mapping")
        resolved = dict(resolved)
        resolved["loader"] = {**dict(resolved["loader"]), **dict(loader_config)}
    bundle = build_lewm_data(resolved)

    factual, no_action = build_matched_predictor_pair(
        checkpoint, initialization_seed=seed
    )
    for model in (factual, no_action):
        model.set_action_normalization(
            action_mean=base.action_mean,
            action_std=base.action_std,
        )
        model.to(device)
    if trainable_parameter_count(factual) != trainable_parameter_count(no_action):
        raise RuntimeError("matched predictor parameter counts differ")
    for left, right in zip(
        factual.state_dict().values(), no_action.state_dict().values(), strict=True
    ):
        if not torch.equal(left, right):
            raise RuntimeError("matched predictors did not start identically")

    optimizer_config = dict(config["optimizer"])
    optimizer = torch.optim.AdamW(
        list(factual.parameters()) + list(no_action.parameters()),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config.get("betas", (0.9, 0.999))),
        eps=float(optimizer_config.get("eps", 1e-8)),
    )
    trainer = dict(config["trainer"])
    configured_steps = int(trainer["max_steps"])
    total_steps = int(args.smoke_steps) if args.smoke_steps > 0 else configured_steps
    if total_steps <= 0:
        raise ValueError("max_steps must be positive")
    warmup_steps = min(int(optimizer_config.get("warmup_steps", 0)), total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(step, warmup_steps, total_steps),
    )
    amp_dtype = _amp_dtype(str(trainer.get("amp_dtype", "bfloat16")))
    amp_enabled = bool(trainer.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    grad_clip = float(trainer.get("grad_clip", 1.0))
    max_val_batches = int(trainer.get("max_val_batches", 0))
    if args.smoke_steps > 0:
        max_val_batches = 1

    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    if args.smoke_steps > 0:
        output_dir = output_dir.with_name(output_dir.name + f"_smoke{args.smoke_steps}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "predictor_pair_latest.pt"
    step = 0
    epoch = 0
    records: list[dict[str, Any]] = []
    while step < total_steps:
        epoch += 1
        factual.train()
        no_action.train()
        base.eval()
        totals = {"factual_mse": 0.0, "no_action_mse": 0.0}
        samples = 0
        started = time.perf_counter()
        for host_batch in bundle.train_loader:
            if step >= total_steps:
                break
            batch = move_batch_to_device(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                context, target, action = _latent_batch(base, batch)
                factual_output, no_action_output = matched_pair_forward(
                    factual, no_action, context, action, target
                )
                loss = factual_output.prediction_loss + no_action_output.prediction_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(factual.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(no_action.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            batch_size = int(context.shape[0])
            totals["factual_mse"] += (
                float(factual_output.prediction_loss.detach()) * batch_size
            )
            totals["no_action_mse"] += (
                float(no_action_output.prediction_loss.detach()) * batch_size
            )
            samples += batch_size
        if samples == 0:
            raise RuntimeError("training loader produced no samples")
        train_metrics = {key: value / samples for key, value in totals.items()}
        train_metrics["samples"] = float(samples)
        train_metrics["seconds"] = time.perf_counter() - started
        val_metrics = _evaluate_pair(
            base,
            factual,
            no_action,
            bundle.val_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_batches=max_val_batches,
        )
        record = {
            "epoch": epoch,
            "step": step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        if not all(
            math.isfinite(float(value))
            for group in (train_metrics, val_metrics)
            for value in group.values()
        ):
            raise RuntimeError("pair training produced non-finite metrics")
        records.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        _atomic_torch_save(
            {
                "checkpoint_type": MATCHED_DYNAMICS_PAIR_CHECKPOINT_TYPE,
                "step": step,
                "epoch": epoch,
                "seed": seed,
                "base_checkpoint": str(base_checkpoint),
                "base_checkpoint_step": int(checkpoint["step"]),
                "predictor_config": predictor_kwargs_from_base_checkpoint(checkpoint),
                "parameter_count_per_head": trainable_parameter_count(factual),
                "factual_state": factual.state_dict(),
                "no_action_state": no_action.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "metrics": record,
                "config": config,
            },
            checkpoint_path,
        )
        print(json.dumps(record, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
