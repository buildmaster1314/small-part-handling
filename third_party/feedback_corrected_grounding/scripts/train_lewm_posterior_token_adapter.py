#!/usr/bin/env python3
"""Distill a frozen clean policy through factual or no-action posterior tokens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_posterior_aware_policy import (  # noqa: E402
    PosteriorAwarePolicyDataset,
    PosteriorTokenResidualAdapter,
    load_posterior_policy_cache,
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


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=device.type == "cuda") for name, value in batch.items()}


def _teacher_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in batch.items()
        if name not in {
            "grounded_token_override",
            "grounded_token_override_mask",
            "posterior_drop_active",
            "posterior_missing_age",
        }
    }


def _student_batch(
    batch: dict[str, torch.Tensor],
    adapter: PosteriorTokenResidualAdapter,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    result = dict(batch)
    token = result["grounded_token_override"]
    mask = result["grounded_token_override_mask"]
    adapted = adapter(token, mask)
    result["grounded_token_override"] = adapted
    return result, adapted - token


def _distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    first = F.smooth_l1_loss(student[:, :5], teacher[:, :5], beta=0.02)
    full = F.smooth_l1_loss(student, teacher, beta=0.02)
    return first + 0.1 * full


@torch.no_grad()
def _evaluate(policy, adapter, loader, *, device: torch.device) -> dict[str, float]:
    policy.eval()
    adapter.eval()
    totals = {"base_mse": 0.0, "adapted_mse": 0.0, "rows": 0.0}
    for raw in loader:
        batch = _move(raw, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            teacher = policy(_teacher_batch(batch)).action.detach().float()
            unadapted = policy(batch).action.detach().float()
            student_batch, _ = _student_batch(batch, adapter)
            adapted = policy(student_batch).action.detach().float()
        rows = float(teacher.shape[0])
        totals["base_mse"] += float((unadapted[:, :5] - teacher[:, :5]).square().mean()) * rows
        totals["adapted_mse"] += float((adapted[:, :5] - teacher[:, :5]).square().mean()) * rows
        totals["rows"] += rows
    return {
        "base_mse": totals["base_mse"] / totals["rows"],
        "adapted_mse": totals["adapted_mse"] / totals["rows"],
        "relative_improvement": 1.0 - totals["adapted_mse"] / max(totals["base_mse"], 1e-12),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("factual", "no_action"), required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--policy-cache", type=Path, default=DEFAULT_POLICY_CACHE)
    parser.add_argument("--grounding-cache", type=Path, default=DEFAULT_GROUNDING_CACHE)
    parser.add_argument("--posterior-cache", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=63401)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not str(args.device).startswith("cuda:0") and str(args.device) != "cpu":
        parser.error("this experiment may use cuda:0 or cpu only")
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("steps and batch size must be positive")

    _seed(int(args.seed))
    device = torch.device(args.device)
    data_path = _resolve(args.data)
    posterior = load_posterior_policy_cache(_resolve(args.posterior_cache))
    with np.load(data_path, allow_pickle=False) as raw:
        split = make_split(raw, 0.2, 0, "task-episode")

    def dataset(indices) -> PosteriorAwarePolicyDataset:
        base = ProjectionSequenceDataset(
            data_path,
            indices,
            obs_mode="robot",
            obs_horizon=2,
            action_horizon=32,
            grounding_cache=_resolve(args.grounding_cache),
            policy_cache=_resolve(args.policy_cache),
        )
        return PosteriorAwarePolicyDataset(base, posterior, method=args.method, policy_history_steps=2)

    train_all = dataset(split.train)
    val_all = dataset(split.test)

    def active_subset(value: PosteriorAwarePolicyDataset) -> Subset:
        active = np.flatnonzero(value.cache.drop_active[value.rows]).tolist()
        if not active:
            raise RuntimeError("posterior split has no active interruption rows")
        return Subset(value, active)

    train_data = active_subset(train_all)
    val_data = active_subset(val_all)
    common = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(train_data, shuffle=True, drop_last=False, generator=generator, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)

    policy, policy_args = load_action_chunk_policy(_resolve(args.base_checkpoint), device)
    policy.set_reliability_override("ones")
    policy.requires_grad_(False).eval()
    adapter = PosteriorTokenResidualAdapter(
        token_dim=int(policy.model_dim),
        num_keypoints=int(policy.num_keypoints),
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=1e-6)

    def schedule(step: int) -> float:
        if step < int(args.warmup_steps):
            return float(step + 1) / max(1, int(args.warmup_steps))
        progress = (step - int(args.warmup_steps)) / max(1, int(args.steps) - int(args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_loader)
    metrics: list[dict[str, object]] = []
    for step in range(1, int(args.steps) + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw = next(iterator)
        batch = _move(raw, device)
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            teacher = policy(_teacher_batch(batch)).action.detach().float()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            student_batch, residual = _student_batch(batch, adapter)
            student = policy(student_batch).action.float()
            action_loss = _distillation_loss(student, teacher)
            regularization = residual.square().mean()
            loss = action_loss + 1e-3 * regularization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step % 200 == 0 or step == int(args.steps):
            validation = _evaluate(policy, adapter, val_loader, device=device)
            row = {
                "step": step,
                "train_action_loss": float(action_loss.detach()),
                "train_residual_rms": float(regularization.detach().sqrt()),
                "validation": validation,
            }
            metrics.append(row)
            print(json.dumps(row), flush=True)

    output_dir = _resolve(
        args.output_dir
        if args.output_dir is not None
        else ROOT / f"outputs/lewm_posterior_token_adapter/{args.method}_seed{args.seed}_2k"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "checkpoint_type": "lewm_posterior_token_residual_adapter",
        "method": args.method,
        "base_checkpoint": str(_resolve(args.base_checkpoint)),
        "adapter_config": {
            "token_dim": int(policy.model_dim),
            "num_keypoints": int(policy.num_keypoints),
            "hidden_dim": 128,
            "identity_dim": 16,
            "residual_cap": 0.25,
        },
        "adapter_state": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
        "policy_args": policy_args,
        "steps": int(args.steps),
        "seed": int(args.seed),
    }
    checkpoint_path = output_dir / "adapter_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "method": args.method,
        "checkpoint": str(checkpoint_path),
        "train_rows": len(train_data),
        "val_rows": len(val_data),
        "trainable_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "metrics": metrics,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report | {"metrics": metrics[-1:]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
