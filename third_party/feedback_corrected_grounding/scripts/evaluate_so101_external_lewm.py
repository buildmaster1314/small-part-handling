#!/usr/bin/env python3
"""Evaluate a frozen SO-101 state-residual LeWM on fully external episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fcpg.so101_state_residual_lewm import StateResidualLeWM
from scripts.train_so101_lewm import Cache, FixedIndexSampler, SO101Sequences
from scripts.train_so101_state_residual_lewm import evaluate
from scripts.visualize_so101_lewm import (
    build_checkpoint_model,
    checkpoint_action_normalizer,
    checkpoint_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen state-residual LeWM on every eligible sequence in one "
            "or more external SO-101 caches. No train/validation split is fitted."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-base", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    if args.device not in {"cpu", "cuda:0"}:
        parser.error("use CPU or CUDA0")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch-size must be positive and workers non-negative")
    return args


def all_validation_assignments(caches: list[Cache]) -> dict[tuple[str, int], str]:
    assignments: dict[tuple[str, int], str] = {}
    for cache in caches:
        for raw_episode in np.unique(np.asarray(cache.episodes)):
            key = (cache.dataset_id, int(raw_episode))
            previous = assignments.setdefault(key, "val")
            if previous != "val":
                raise ValueError(f"external episode assignment conflict for {key}")
    if not assignments:
        raise ValueError("external caches contain no episodes")
    return assignments


def relative_gain(prediction: float, baseline: float) -> float:
    if not np.isfinite(prediction) or not np.isfinite(baseline) or baseline <= 0:
        return float("nan")
    return (baseline - prediction) / baseline


def build_summary(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    visual_prediction = float(metrics["visual_prediction_mse"])
    state_prediction = float(metrics["state_prediction_mse"])
    rows = [
        "# Frozen LeWM external-source evaluation",
        "",
        f"- Checkpoint step: `{payload['checkpoint']['step']}`",
        f"- External physical episodes: `{payload['external']['episodes']}`",
        f"- Evaluated sequences: `{payload['external']['sequences']}`",
        f"- Source-task groups: `{metrics['macro_groups']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Visual prediction MSE | `{visual_prediction:.8f}` |",
        f"| Visual base MSE | `{metrics['visual_base_mse']:.8f}` |",
        f"| Visual copy MSE | `{metrics['visual_copy_mse']:.8f}` |",
        f"| Visual gain vs base | `{100.0 * relative_gain(visual_prediction, metrics['visual_base_mse']):.2f}%` |",
        f"| Visual gain vs copy | `{100.0 * relative_gain(visual_prediction, metrics['visual_copy_mse']):.2f}%` |",
        f"| Visual action-shuffle delta | `{metrics['visual_action_delta']:.8f}` |",
        f"| Visual state-shuffle delta | `{metrics['visual_state_delta']:.8f}` |",
        f"| Six-state prediction MSE | `{state_prediction:.8f}` |",
        f"| Six-state command baseline MSE | `{metrics['state_command_mse']:.8f}` |",
        f"| Six-state hold-current MSE | `{metrics['state_copy_mse']:.8f}` |",
        f"| State gain vs command | `{100.0 * relative_gain(state_prediction, metrics['state_command_mse']):.2f}%` |",
        f"| State gain vs hold-current | `{100.0 * relative_gain(state_prediction, metrics['state_copy_mse']):.2f}%` |",
        f"| State action-shuffle delta | `{metrics['state_action_delta']:.8f}` |",
        f"| State input-shuffle delta | `{metrics['state_input_delta']:.8f}` |",
        "",
        "This is a zero-training external-source test. Every eligible sequence from the listed external caches is evaluated; none is used to fit or select the checkpoint.",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is non-empty; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.workers))
    try:
        torch.set_num_interop_threads(max(1, min(args.workers, 4)))
    except RuntimeError:
        pass

    checkpoint_path = args.checkpoint.expanduser().resolve()
    (
        checkpoint,
        embedded_args,
        _,
        frameskip,
        history,
        checkpoint_sha256,
    ) = checkpoint_contract(checkpoint_path, args.cache_base)
    contract = checkpoint.get("run_config", {}).get("contract", {})
    if contract.get("protocol") not in {
        "so101-state-residual-lewm-v4",
        "so101-state-residual-lewm-v5-temporal",
    }:
        raise ValueError("external evaluation requires a state-residual LeWM checkpoint")

    cache_paths = [path.expanduser().resolve() for path in args.cache]
    external_caches = [Cache(path) for path in cache_paths]
    assignments = all_validation_assignments(external_caches)
    center, scale, normalization = checkpoint_action_normalizer(
        checkpoint, embedded_args
    )
    dataset = SO101Sequences(
        cache_paths,
        "val",
        frameskip,
        history,
        exclusions=set(),
        split_seed=int(embedded_args.get("split_seed", 7301)),
        val_fraction=float(embedded_args.get("val_fraction", 0.1)),
        joint_scale=float(embedded_args.get("joint_scale", 100.0)),
        motion_threshold=float(embedded_args.get("motion_threshold", 0.01)),
        action_center=center,
        action_scale=scale,
        episode_assignments=assignments,
    )
    indices = dataset.balanced_eval_indices(None)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA0 was requested but CUDA is unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    if not isinstance(model, StateResidualLeWM):
        raise ValueError("checkpoint did not reconstruct a state-residual LeWM")

    workers = max(0, min(2, args.workers))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=FixedIndexSampler(torch.as_tensor(indices, dtype=torch.int64)),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    metrics = evaluate(
        model,
        loader,
        device,
        dataset.group_labels,
        torch.as_tensor(center, device=device),
        torch.as_tensor(scale, device=device),
    )
    payload = {
        "format": "so101-frozen-external-evaluation-v1",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "step": int(checkpoint.get("step", -1)),
            "protocol": contract.get("protocol"),
            "normalization": normalization,
        },
        "external": {
            "cache_paths": [str(path) for path in cache_paths],
            "dataset_ids": sorted({cache.dataset_id for cache in external_caches}),
            "episodes": len(assignments),
            "sequences": len(dataset),
            "frameskip": frameskip,
            "history": history,
            "training_performed": False,
        },
        "metrics": metrics,
    }
    (output / "evaluation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(build_summary(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"summary: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
