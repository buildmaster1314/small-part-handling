#!/usr/bin/env python3
"""Train a bounded paired-view adapter over frozen anchor embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.paired_view_canonicalizer import (  # noqa: E402
    FEATURE_MODES,
    PairedViewAnchorCache,
    PairedViewCanonicalizer,
    build_anchor_context,
    load_paired_view_anchor_cache,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1, value.shape[-1])


def _effective_rank(value: np.ndarray) -> float:
    centered = np.asarray(value, dtype=np.float64) - np.asarray(
        value, dtype=np.float64
    ).mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    if float(energy.sum()) <= 1e-12:
        return 0.0
    probability = energy / energy.sum()
    return float(np.exp(-(probability * np.log(probability + 1e-12)).sum()))


def _mean_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, axis=-1).mean())


def _temporal_delta_error(
    value: np.ndarray,
    target: np.ndarray,
    cache: PairedViewAnchorCache,
) -> float:
    value = value.reshape(cache.frame_count, cache.anchor_count, -1)
    target = target.reshape(cache.frame_count, cache.anchor_count, -1)
    errors: list[np.ndarray] = []
    tasks = cache.task_name.astype(str)
    for task in sorted(set(tasks.tolist())):
        episodes = sorted(set(cache.episode_id[tasks == task].tolist()))
        for episode in episodes:
            rows = np.flatnonzero((tasks == task) & (cache.episode_id == episode))
            rows = rows[np.argsort(cache.frame_id[rows])]
            if len(rows) < 2:
                continue
            value_delta = np.diff(value[rows], axis=0)
            target_delta = np.diff(target[rows], axis=0)
            errors.append(np.linalg.norm(value_delta - target_delta, axis=-1).reshape(-1))
    if not errors:
        raise ValueError("cache has no consecutive frames for temporal evaluation")
    return float(np.concatenate(errors).mean())


def _task_anchor_means(
    cache: PairedViewAnchorCache, target: np.ndarray
) -> dict[tuple[str, int], np.ndarray]:
    target = target.reshape(cache.frame_count, cache.anchor_count, -1)
    tasks = cache.task_name.astype(str)
    result: dict[tuple[str, int], np.ndarray] = {}
    for task in sorted(set(tasks.tolist())):
        rows = tasks == task
        for anchor in range(cache.anchor_count):
            result[(task, anchor)] = target[rows, anchor].mean(axis=0)
    return result


def _mean_baseline(
    cache: PairedViewAnchorCache,
    means: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    rows = []
    for task in cache.task_name.astype(str).tolist():
        rows.extend(means[(task, anchor)] for anchor in range(cache.anchor_count))
    return np.stack(rows).astype(np.float32)


@torch.inference_mode()
def _predict(
    model: PairedViewCanonicalizer,
    latent: np.ndarray,
    context: np.ndarray,
    *,
    context_mean: np.ndarray,
    context_std: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    latent_flat = _flatten(latent)
    context_flat = _flatten(context)
    parts: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(latent_flat), batch_size):
        stop = min(start + batch_size, len(latent_flat))
        z = torch.from_numpy(latent_flat[start:stop]).to(device)
        c = torch.from_numpy(
            (context_flat[start:stop] - context_mean) / context_std
        ).to(device)
        parts.append(model(z, c).float().cpu().numpy())
    return np.concatenate(parts)


def _evaluate(
    model: PairedViewCanonicalizer,
    cache: PairedViewAnchorCache,
    *,
    mode: str,
    context_mean: np.ndarray,
    context_std: np.ndarray,
    train_means: dict[tuple[str, int], np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    dynamic = _flatten(cache.latent_dynamic_corrected)
    clean_input = _flatten(cache.latent_clean_corrected)
    target = _flatten(cache.latent_clean_nominal)
    oracle = _flatten(cache.latent_dynamic_oracle)
    dynamic_context = build_anchor_context(cache, mode=mode, clean=False)
    clean_context = build_anchor_context(cache, mode=mode, clean=True)
    adapted = _predict(
        model,
        dynamic,
        dynamic_context,
        context_mean=context_mean,
        context_std=context_std,
        device=device,
    )
    clean_adapted = _predict(
        model,
        clean_input,
        clean_context,
        context_mean=context_mean,
        context_std=context_std,
        device=device,
    )
    mean_baseline = _mean_baseline(cache, train_means)
    raw_gap = _mean_l2(dynamic, target)
    adapted_gap = _mean_l2(adapted, target)
    tasks = cache.task_name.astype(str)

    def metrics_for_rows(flat_rows: np.ndarray) -> dict[str, float]:
        group_raw = _mean_l2(dynamic[flat_rows], target[flat_rows])
        group_adapted = _mean_l2(adapted[flat_rows], target[flat_rows])
        return {
            "raw_gap": group_raw,
            "adapted_gap": group_adapted,
            "relative_improvement": float(
                (group_raw - group_adapted) / max(group_raw, 1e-12)
            ),
        }

    def anchor_rows(frame_rows: np.ndarray) -> np.ndarray:
        return (
            frame_rows[:, None] * cache.anchor_count
            + np.arange(cache.anchor_count)[None]
        ).reshape(-1)

    by_task: dict[str, Any] = {}
    for task in sorted(set(tasks.tolist())):
        frame_rows = np.flatnonzero(tasks == task)
        by_task[task] = metrics_for_rows(anchor_rows(frame_rows))
    by_phase: dict[str, Any] = {}
    phase = np.asarray(cache.trajectory_phase, dtype=np.float32)
    phase_masks = {
        "pre": phase <= 0.0,
        "moving": (phase > 0.0) & (phase < 1.0),
        "post": phase >= 1.0,
    }
    for name, mask in phase_masks.items():
        frame_rows = np.flatnonzero(mask)
        if frame_rows.size:
            by_phase[name] = {
                "frames": int(frame_rows.size),
                **metrics_for_rows(anchor_rows(frame_rows)),
            }
    target_std = float(target.std(axis=0).mean())
    return {
        "frames": cache.frame_count,
        "trajectory_mode": cache.trajectory_mode,
        "raw_gap": raw_gap,
        "adapted_gap": adapted_gap,
        "relative_improvement": float(
            (raw_gap - adapted_gap) / max(raw_gap, 1e-12)
        ),
        "q_true_oracle_gap": _mean_l2(oracle, target),
        "task_anchor_mean_gap": _mean_l2(mean_baseline, target),
        "clean_adapter_gap": _mean_l2(clean_adapted, target),
        "clean_input_gap": _mean_l2(clean_input, target),
        "raw_temporal_delta_error": _temporal_delta_error(dynamic, target, cache),
        "adapted_temporal_delta_error": _temporal_delta_error(
            adapted, target, cache
        ),
        "target_mean_std": target_std,
        "adapted_mean_std": float(adapted.std(axis=0).mean()),
        "std_ratio": float(adapted.std(axis=0).mean() / max(target_std, 1e-12)),
        "target_effective_rank": _effective_rank(target),
        "adapted_effective_rank": _effective_rank(adapted),
        "by_task": by_task,
        "by_phase": by_phase,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument(
        "--eval-cache",
        action="append",
        required=True,
        help="Named evaluation cache as name=path; may be repeated.",
    )
    parser.add_argument("--mode", choices=FEATURE_MODES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-weight", type=float, default=0.5)
    parser.add_argument("--moment-weight", type=float, default=0.02)
    return parser


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("eval caches must use name=path")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError("eval cache names must be non-empty and unique")
        result[name] = Path(raw_path).resolve()
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.steps <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise ValueError("steps, batch size, and hidden dimension must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("optimizer settings are invalid")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_path = Path(args.train_cache).resolve()
    eval_paths = _named_paths(args.eval_cache)
    train_cache = load_paired_view_anchor_cache(train_path)
    eval_caches = {
        name: load_paired_view_anchor_cache(path) for name, path in eval_paths.items()
    }
    for name, cache in eval_caches.items():
        if cache.latent_dim != train_cache.latent_dim:
            raise ValueError(f"eval cache {name!r} latent dimension mismatch")
        if cache.source_data_sha256 == train_cache.source_data_sha256:
            raise ValueError(f"eval cache {name!r} reuses the training source")

    dynamic_context = _flatten(
        build_anchor_context(train_cache, mode=args.mode, clean=False)
    )
    clean_context = _flatten(
        build_anchor_context(train_cache, mode=args.mode, clean=True)
    )
    context_values = np.concatenate([dynamic_context, clean_context])
    context_mean = context_values.mean(axis=0).astype(np.float32)
    context_std = context_values.std(axis=0).astype(np.float32)
    context_std[context_std < 1e-4] = 1.0

    dynamic = torch.from_numpy(_flatten(train_cache.latent_dynamic_corrected)).to(
        device
    )
    clean_input = torch.from_numpy(_flatten(train_cache.latent_clean_corrected)).to(
        device
    )
    target = torch.from_numpy(_flatten(train_cache.latent_clean_nominal)).to(device)
    dynamic_context_tensor = torch.from_numpy(
        (dynamic_context - context_mean) / context_std
    ).to(device)
    clean_context_tensor = torch.from_numpy(
        (clean_context - context_mean) / context_std
    ).to(device)
    model = PairedViewCanonicalizer(
        train_cache.latent_dim, hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    logs: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        rows = torch.randint(
            len(dynamic),
            (args.batch_size,),
            generator=generator,
            device=device,
        )
        predicted = model(dynamic[rows], dynamic_context_tensor[rows])
        clean_predicted = model(clean_input[rows], clean_context_tensor[rows])
        pair_loss = F.mse_loss(predicted, target[rows])
        identity_loss = F.mse_loss(clean_predicted, target[rows])
        predicted_std = predicted.std(dim=0, unbiased=False)
        target_std = target[rows].std(dim=0, unbiased=False)
        moment_loss = F.smooth_l1_loss(predicted_std, target_std)
        loss = (
            pair_loss
            + args.identity_weight * identity_loss
            + args.moment_weight * moment_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "pair_loss": float(pair_loss.detach().cpu()),
                "identity_loss": float(identity_loss.detach().cpu()),
                "moment_loss": float(moment_loss.detach().cpu()),
            }
            logs.append(row)
            print(json.dumps(row), flush=True)

    train_target = _flatten(train_cache.latent_clean_nominal)
    train_means = _task_anchor_means(train_cache, train_target)
    evaluations = {
        "train": _evaluate(
            model,
            train_cache,
            mode=args.mode,
            context_mean=context_mean,
            context_std=context_std,
            train_means=train_means,
            device=device,
        )
    }
    evaluations.update(
        {
            name: _evaluate(
                model,
                cache,
                mode=args.mode,
                context_mean=context_mean,
                context_std=context_std,
                train_means=train_means,
                device=device,
            )
            for name, cache in eval_caches.items()
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "canonicalizer_latest.pt"
    torch.save(
        {
            "checkpoint_type": "fcpg_paired_view_canonicalizer_v1",
            "model_state_dict": model.state_dict(),
            "latent_dim": train_cache.latent_dim,
            "hidden_dim": args.hidden_dim,
            "feature_mode": args.mode,
            "context_mean": context_mean,
            "context_std": context_std,
            "seed": args.seed,
            "steps": args.steps,
            "train_cache_sha256": _sha256(train_path),
        },
        checkpoint_path,
    )
    payload = {
        "protocol": "fcpg_paired_view_canonicalizer_development_v1",
        "mode": args.mode,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_cache": str(train_path),
        "train_cache_sha256": _sha256(train_path),
        "eval_caches": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in eval_paths.items()
        },
        "loss_weights": {
            "identity": args.identity_weight,
            "moment": args.moment_weight,
        },
        "logs": logs,
        "evaluations": evaluations,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
