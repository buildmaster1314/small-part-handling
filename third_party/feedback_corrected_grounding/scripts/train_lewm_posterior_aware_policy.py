#!/usr/bin/env python3
"""Fine-tune one matched Pick policy on measurement or P1 posterior tokens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_posterior_aware_policy import (  # noqa: E402
    POSTERIOR_POLICY_METHODS,
    PosteriorAwarePolicyDataset,
    load_posterior_policy_cache,
    merge_posterior_trained_state,
)
from fcpg.sequence_data import ProjectionSequenceDataset  # noqa: E402
from scripts.train_bc_policy import make_split  # noqa: E402


DEFAULT_DATA = ROOT / "data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz"
DEFAULT_POLICY_CACHE = ROOT / "data/policy_visual_cache/lewm_stage1/pick_seed0_sig0_pick"
DEFAULT_GROUNDING_CACHE = ROOT / "data/grounding_cache/lewm_stage1/pick_seed0_sig0"
DEFAULT_BASE = ROOT / "outputs/action_chunk/p1/pick_fcpg_pureres_seed0_retry1/fcpg_action_chunk_best.pt"
DEFAULT_POSTERIOR = ROOT / "outputs/lewm_posterior_aware_policy/pick_train_cache_seed63101.npz"


def _resolve(path: str | Path) -> Path:
    target = Path(path).expanduser()
    return target.resolve() if target.is_absolute() else (ROOT / target).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def _set_train_mode(policy: torch.nn.Module) -> None:
    policy.train()
    for name in (
        "vision_encoder",
        "grounding_adapter",
        "global_visual_projection",
        "robot_projection",
        "fcpg_adapter",
    ):
        module = getattr(policy, name, None)
        if isinstance(module, torch.nn.Module):
            module.eval()


@torch.inference_mode()
def _evaluate(
    policy: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    policy.eval()
    totals = {"loss": 0.0, "motion": 0.0, "gripper": 0.0, "rows": 0.0}
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        moved = _move(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss, parts, _ = policy.compute_loss(moved, gripper_weight=0.5)
        rows = float(moved["action"].shape[0])
        totals["loss"] += float(loss) * rows
        totals["motion"] += float(parts["motion_loss"]) * rows
        totals["gripper"] += float(parts["gripper_loss"]) * rows
        totals["rows"] += rows
    if totals["rows"] <= 0:
        raise RuntimeError("validation loader produced no rows")
    return {
        name: totals[name] / totals["rows"]
        for name in ("loss", "motion", "gripper")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=POSTERIOR_POLICY_METHODS, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--policy-cache", type=Path, default=DEFAULT_POLICY_CACHE)
    parser.add_argument("--grounding-cache", type=Path, default=DEFAULT_GROUNDING_CACHE)
    parser.add_argument("--posterior-cache", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=24)
    parser.add_argument("--seed", type=int, default=63201)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("steps and batch size must be positive")
    if not str(args.device).startswith("cuda:0") and str(args.device) != "cpu":
        parser.error("this experiment may use cuda:0 or cpu only")

    data_path = _resolve(args.data)
    policy_cache_path = _resolve(args.policy_cache)
    grounding_cache_path = _resolve(args.grounding_cache)
    posterior_cache_path = _resolve(args.posterior_cache)
    base_path = _resolve(args.base_checkpoint)
    output_dir = _resolve(
        args.output_dir
        if args.output_dir is not None
        else ROOT / f"outputs/lewm_posterior_aware_policy/{args.method}_seed{args.seed}_2k"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _seed_everything(int(args.seed))
    device = torch.device(args.device)

    with np.load(data_path, allow_pickle=False) as raw:
        split = make_split(raw, 0.2, int(args.split_seed), "task-episode")
    posterior = load_posterior_policy_cache(posterior_cache_path)
    train_base = ProjectionSequenceDataset(
        data_path,
        split.train,
        obs_mode="robot",
        obs_horizon=2,
        action_horizon=32,
        grounding_cache=grounding_cache_path,
        policy_cache=policy_cache_path,
    )
    val_base = ProjectionSequenceDataset(
        data_path,
        split.test,
        obs_mode="robot",
        obs_horizon=2,
        action_horizon=32,
        grounding_cache=grounding_cache_path,
        policy_cache=policy_cache_path,
    )
    train_data = PosteriorAwarePolicyDataset(
        train_base, posterior, method=args.method, policy_history_steps=2
    )
    val_data = PosteriorAwarePolicyDataset(
        val_base, posterior, method=args.method, policy_history_steps=2
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    loader_kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    train_loader = DataLoader(
        train_data,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(args.batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    if len(train_loader) == 0:
        raise RuntimeError("training loader has no full batch")

    policy, _ = load_action_chunk_policy(base_path, device)
    policy.set_reliability_override("ones")
    policy.requires_grad_(True)
    for name in (
        "vision_encoder",
        "grounding_adapter",
        "global_visual_projection",
        "robot_projection",
        "fcpg_adapter",
    ):
        module = getattr(policy, name, None)
        if isinstance(module, torch.nn.Module):
            module.requires_grad_(False)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        betas=(0.95, 0.999),
        eps=1.0e-8,
        weight_decay=float(args.weight_decay),
    )

    def schedule(step: int) -> float:
        if step < int(args.warmup_steps):
            return float(step + 1) / max(1, int(args.warmup_steps))
        progress = (step - int(args.warmup_steps)) / max(
            1, int(args.steps) - int(args.warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_loader)
    metrics: list[dict[str, object]] = []
    print(
        json.dumps(
            {
                "method": args.method,
                "train_rows": len(train_data),
                "val_rows": len(val_data),
                "steps": int(args.steps),
                "trainable_parameters": sum(p.numel() for p in trainable),
                "device": str(device),
            }
        ),
        flush=True,
    )
    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        moved = _move(batch, device)
        _set_train_mode(policy)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss, parts, _ = policy.compute_loss(moved, gripper_weight=0.5)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step % int(args.val_every) == 0 or step == int(args.steps):
            validation = _evaluate(
                policy,
                val_loader,
                device=device,
                max_batches=int(args.val_batches),
            )
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "train_motion": float(parts["motion_loss"].detach()),
                "train_gripper": float(parts["gripper_loss"].detach()),
                "val": validation,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            metrics.append(row)
            print(json.dumps(row), flush=True)

    original = torch.load(base_path, map_location="cpu", weights_only=False)
    checkpoint = dict(original)
    checkpoint["model_state"] = merge_posterior_trained_state(
        original["model_state"],
        policy.state_dict(),
    )
    checkpoint["posterior_aware_policy"] = {
        "method": args.method,
        "base_checkpoint": str(base_path),
        "posterior_cache": str(posterior_cache_path),
        "optimizer_steps": int(args.steps),
        "seed": int(args.seed),
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "frontend_frozen": True,
        "selection": "fixed_latest",
    }
    checkpoint_path = output_dir / f"{args.method}_policy_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "method": args.method,
        "checkpoint": str(checkpoint_path),
        "base_checkpoint": str(base_path),
        "posterior_cache": str(posterior_cache_path),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "metrics": metrics,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
