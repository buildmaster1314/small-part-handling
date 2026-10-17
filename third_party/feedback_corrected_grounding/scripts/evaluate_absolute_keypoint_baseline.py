#!/usr/bin/env python3
"""Evaluate direct absolute-q predictions on the matched FCPG validation split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.absolute_keypoint_baseline import (  # noqa: E402
    BASELINE_VARIANT,
    load_absolute_keypoint_baseline,
    load_absolute_keypoint_config,
    validate_deployment_vision_cache,
)
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from scripts.train_absolute_keypoint_baseline import (  # noqa: E402
    DEFAULT_CONFIG,
    AbsoluteKeypointDataset,
    choose_device,
    move_batch,
    predict_from_batch,
    resolve_repo_path,
)
from scripts.train_bc_policy import make_split  # noqa: E402


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_errors(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    if before.ndim != 1 or after.ndim != 1 or before.size != after.size:
        raise ValueError("error vectors must be matched non-empty 1-D arrays")
    if before.size == 0:
        raise ValueError("cannot summarize an empty validation split")
    return {
        "e_before_mean_px": float(before.mean()),
        "e_absolute_mean_px": float(after.mean()),
        "ratio": float(after.mean() / max(before.mean(), 1e-8)),
        "e_absolute_median_px": float(np.median(after)),
        "e_absolute_p90_px": float(np.percentile(after, 90)),
        "valid_keypoints": int(after.size),
    }


@torch.inference_mode()
def evaluate_loader(
    model,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    before_parts: list[np.ndarray] = []
    after_parts: list[np.ndarray] = []
    timed_seconds = 0.0
    timed_frames = 0
    all_seconds = 0.0
    all_frames = 0
    for batch_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        synchronize(device)
        start = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            # q_true remains outside this call and is used only below for metrics.
            output = predict_from_batch(model, batch)
        synchronize(device)
        elapsed = time.perf_counter() - start
        all_seconds += elapsed
        all_frames += int(batch["robot_obs"].shape[0])
        if batch_index >= warmup_batches:
            timed_seconds += elapsed
            timed_frames += int(batch["robot_obs"].shape[0])
        valid = batch["target_valid"]
        before = torch.linalg.vector_norm(
            batch["q_nominal"] - batch["target_q_true"], dim=-1
        )
        after = torch.linalg.vector_norm(
            output.q_absolute_px - batch["target_q_true"], dim=-1
        )
        before_parts.append(before[valid].float().cpu().numpy())
        after_parts.append(after[valid].float().cpu().numpy())
    before_values = np.concatenate(before_parts)
    after_values = np.concatenate(after_parts)
    if timed_frames == 0:
        timed_seconds = all_seconds
        timed_frames = all_frames
    metrics = summarize_errors(before_values, after_values)
    metrics.update(
        {
            "timed_frames": int(timed_frames),
            "model_latency_ms_per_frame": (
                1000.0 * timed_seconds / timed_frames
                if timed_frames > 0
                else float("nan")
            ),
            "model_throughput_frames_per_second": (
                timed_frames / timed_seconds
                if timed_seconds > 0
                else float("nan")
            ),
        }
    )
    return metrics, before_values, after_values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--warmup-batches", type=int, default=2)
    args = parser.parse_args(argv)
    if args.batch_size < 0 or args.num_workers < -1 or args.warmup_batches < 0:
        parser.error("batch size, worker count, and warmup count are invalid")

    config_path = resolve_repo_path(args.config)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    config = load_absolute_keypoint_config(config_path)
    device = choose_device(args.device)
    model, checkpoint = load_absolute_keypoint_baseline(checkpoint_path, device)
    model_config = config["model"]
    if checkpoint["model_config"] != model_config:
        raise ValueError("evaluation config model differs from checkpoint model")
    provenance = checkpoint["training_provenance"]
    assert isinstance(provenance, Mapping)
    if provenance["config_sha256"] != sha256_file(config_path):
        raise ValueError("evaluation config hash differs from training provenance")
    reference = config["reference_fcpg"]
    assert isinstance(reference, Mapping)
    reference_path = resolve_repo_path(str(reference["checkpoint"]))
    split_config = config["split"]
    budget = config["budget"]
    assert isinstance(split_config, Mapping)
    assert isinstance(budget, Mapping)
    batch_size = args.batch_size or int(budget["batch_size"])
    num_workers = (
        int(budget["num_workers"])
        if args.num_workers < 0
        else args.num_workers
    )

    per_source: dict[str, dict[str, float]] = {}
    all_before: list[np.ndarray] = []
    all_after: list[np.ndarray] = []
    source_specs = config["data_sources"]
    assert isinstance(source_specs, list)
    for source in source_specs:
        assert isinstance(source, Mapping)
        data_path = resolve_repo_path(str(source["path"]))
        cache_info = validate_deployment_vision_cache(
            data_path,
            resolve_repo_path(str(source["vision_cache"])),
            model_config,
            expected_weight_source=reference_path,
        )
        with np.load(data_path, allow_pickle=False) as raw:
            split = make_split(
                raw,
                float(split_config["val_fraction"]),
                int(split_config["seed"]),
                str(split_config["unit"]),
            )
        dataset = AbsoluteKeypointDataset(
            data_path,
            cache_info,
            split.test,
            expected_obs_dim=model.obs_dim,
            expected_num_keypoints=model.num_keypoints,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        metrics, before, after = evaluate_loader(
            model, loader, device, args.warmup_batches
        )
        per_source[str(source["name"])] = metrics
        all_before.append(before)
        all_after.append(after)

    overall = summarize_errors(np.concatenate(all_before), np.concatenate(all_after))
    timed_frames = sum(int(row["timed_frames"]) for row in per_source.values())
    inferred_seconds = sum(
        float(row["model_latency_ms_per_frame"])
        * int(row["timed_frames"])
        / 1000.0
        for row in per_source.values()
    )
    overall.update(
        {
            "timed_frames": timed_frames,
            "model_latency_ms_per_frame": (
                1000.0 * inferred_seconds / timed_frames
                if timed_frames > 0
                else float("nan")
            ),
            "model_throughput_frames_per_second": (
                timed_frames / inferred_seconds
                if inferred_seconds > 0
                else float("nan")
            ),
        }
    )
    payload = {
        "schema_version": 1,
        "variant": BASELINE_VARIANT,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split": dict(split_config),
        "q_true_usage": "offline_metrics_only_never_model_input",
        "parameter_report": model.parameter_report(),
        "sampling_report": model.sampling_report(),
        "latency_scope": "adapter_forward_from_cached_frozen_vit_features",
        "warmup_batches_per_source": args.warmup_batches,
        "per_source": per_source,
        "overall": overall,
    }
    print(json.dumps(payload, indent=2))
    output_path = (
        resolve_repo_path(args.output)
        if args.output
        else resolve_repo_path(str(config["output_dir"])) / "evaluation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
