#!/usr/bin/env python3
"""Evaluate whether FCPG context correction improves frozen LeWM rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fcpg.so101_fcpg_lewm_bridge import (
    SO101FCPGConditionedLeWM,
    build_so101_fcpg_corrector,
    infer_so101_fcpg_corrector_variant,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from fcpg.so101_synthetic_dataset import SO101SyntheticFCPGDataset
from scripts.train_so101_fcpg_context_bridge import (
    cache_vision,
    correct_points,
)
from scripts.visualize_so101_lewm import (
    CacheView,
    build_checkpoint_model,
    checkpoint_action_normalizer,
    checkpoint_contract,
    load_caches,
    normalized_action_chunk,
    predict_state_residual_batch,
)


METHODS = ("clean_base", "clean_fcpg", "drift_base", "drift_fcpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--origin-stride", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch_size, args.horizon, args.origin_stride) <= 0:
        parser.error("batch size, horizon, and stride must be positive")
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    return args


def dataset_row_lookup(dataset: SO101SyntheticFCPGDataset) -> np.ndarray:
    arrays = dataset.arrays
    rows = np.asarray(arrays["source_row"], dtype=np.int64)
    cameras = np.asarray(arrays["camera_index"], dtype=np.int64)
    variants = np.asarray(arrays["drift_variant"], dtype=np.int64)
    camera_count = len(dataset.manifest["camera"]["views"])
    variant_count = int(variants.max()) + 1
    lookup = np.full((int(rows.max()) + 1, camera_count, variant_count), -1, dtype=np.int64)
    for index, (row, camera, variant) in enumerate(
        zip(rows, cameras, variants, strict=True)
    ):
        key = (int(row), int(camera), int(variant))
        if lookup[key] >= 0:
            raise ValueError(f"duplicate augmented row identity {key}")
        lookup[key] = index
    if np.any(lookup < 0):
        raise ValueError("augmented dataset lacks a row/view/variant combination")
    return lookup


@torch.inference_mode()
def bridge_all_contexts(
    system: SO101FCPGConditionedLeWM,
    dataset: SO101SyntheticFCPGDataset,
    contexts: torch.Tensor,
    patches: torch.Tensor,
    corrected: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    arrays = dataset.arrays
    width = int(dataset.manifest["width"])
    height = int(dataset.manifest["height"])
    result = np.empty((len(dataset), system.model.latent_dim), dtype=np.float32)
    system.bridge.eval()
    for begin in range(0, len(dataset), batch_size):
        end = min(begin + batch_size, len(dataset))
        rows = slice(begin, end)
        output = system.bridge(
            contexts[rows].to(device=device, dtype=torch.float32).unsqueeze(1),
            patches[rows].to(device=device, dtype=torch.float32).unsqueeze(1),
            torch.from_numpy(np.asarray(arrays["q_nominal"][rows]).copy()).to(device).unsqueeze(1),
            torch.from_numpy(corrected[rows].copy()).to(device).unsqueeze(1),
            torch.from_numpy(np.asarray(arrays["anchors_xyz"][rows]).copy()).to(device).unsqueeze(1),
            torch.from_numpy(np.asarray(arrays["valid"][rows]).copy()).to(device).unsqueeze(1),
            image_size=(width, height),
        )
        result[rows] = output.context_embedding[:, 0].cpu().numpy()
    return result


def rollout(
    model: StateResidualLeWM,
    context_latents: np.ndarray,
    context_states: np.ndarray,
    past_action_chunks: np.ndarray,
    future_action_chunks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    latent_history = np.asarray(context_latents, dtype=np.float32)
    state_history = np.asarray(context_states, dtype=np.float32)
    action_history = np.asarray(past_action_chunks, dtype=np.float32)
    latent_predictions: list[np.ndarray] = []
    state_predictions: list[np.ndarray] = []
    for step in range(future_action_chunks.shape[1]):
        action_history = np.concatenate(
            (action_history, future_action_chunks[:, step : step + 1]), axis=1
        )
        next_latent, next_state = predict_state_residual_batch(
            model,
            latent_history[:, -3:],
            action_history[:, -3:],
            state_history[:, -3:],
        )
        latent_history = np.concatenate((latent_history, next_latent[:, None]), axis=1)
        state_history = np.concatenate((state_history, next_state[:, None]), axis=1)
        latent_predictions.append(next_latent)
        state_predictions.append(next_state)
    return np.stack(latent_predictions, axis=1), np.stack(state_predictions, axis=1)


def metric_summary(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    per_origin_horizon = np.mean(np.square(prediction - target), axis=-1)
    horizon = per_origin_horizon.mean(axis=0)
    return {
        "origins": int(len(prediction)),
        "all_horizon_mse": float(per_origin_horizon.mean()),
        "per_horizon_mse": {
            f"H{index + 1}": float(value) for index, value in enumerate(horizon)
        },
    }


def relative_effect(reference: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    reference_all = float(reference["all_horizon_mse"])
    method_all = float(method["all_horizon_mse"])
    per_horizon = {}
    for name, reference_value in reference["per_horizon_mse"].items():
        method_value = float(method["per_horizon_mse"][name])
        per_horizon[name] = (
            (float(reference_value) - method_value) / float(reference_value)
            if float(reference_value) > 1.0e-12
            else None
        )
    return {
        "all_horizon_relative_improvement": (
            (reference_all - method_all) / reference_all
            if reference_all > 1.0e-12
            else None
        ),
        "per_horizon_relative_improvement": per_horizon,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    bridge_payload = torch.load(
        args.bridge_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    bridge_args = bridge_payload["args"]
    dataset = SO101SyntheticFCPGDataset(Path(bridge_args["dataset"]))
    checkpoint_path = Path(bridge_args["checkpoint"]).resolve()
    checkpoint, embedded_args, _, frameskip, history, checkpoint_sha = checkpoint_contract(
        checkpoint_path, None
    )
    if history != 3:
        raise ValueError("future evaluation requires history=3")
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    if not isinstance(model, StateResidualLeWM):
        raise TypeError("future evaluation requires a state-residual LeWM")
    corrector = build_so101_fcpg_corrector(
        model,
        camera_context_dim=5,
        variant=infer_so101_fcpg_corrector_variant(bridge_payload),
    )
    system = SO101FCPGConditionedLeWM(
        model,
        corrector=corrector,
        camera_context_dim=5,
        freeze_model=True,
    ).to(device)
    system.corrector.load_state_dict(bridge_payload["corrector"], strict=True)
    system.bridge.load_state_dict(bridge_payload["bridge"], strict=True)
    system.eval()

    contexts, patches = cache_vision(system, dataset, device, args.batch_size)
    contract = checkpoint["run_config"]["contract"]
    state_center = np.asarray(contract["normalization_center"], dtype=np.float32)
    state_scale = np.asarray(contract["normalization_scale"], dtype=np.float32)
    normalized_dataset_state = (
        np.asarray(dataset.arrays["states"], dtype=np.float32) - state_center
    ) / state_scale
    corrected = correct_points(
        system,
        dataset,
        patches,
        normalized_dataset_state,
        device,
        args.batch_size,
    )
    bridged = bridge_all_contexts(
        system, dataset, contexts, patches, corrected, device, args.batch_size
    )
    base = contexts.float().numpy()
    lookup = dataset_row_lookup(dataset)

    source_cache_path = Path(dataset.manifest["sources"][0])
    cache: CacheView = load_caches([source_cache_path])[0]
    validation_episode = int(dataset.manifest["validation_episode"])
    episode_rows = np.flatnonzero(
        np.asarray(cache.data.episodes, dtype=np.int64) == validation_episode
    )
    if len(episode_rows) == 0 or not np.all(np.diff(episode_rows) == 1):
        raise ValueError("validation episode must be present and contiguous")
    horizon_raw = frameskip * args.horizon
    first_local = frameskip * (history - 1)
    final_local = len(episode_rows) - horizon_raw - 1
    origins = episode_rows[
        np.arange(first_local, final_local + 1, args.origin_stride, dtype=np.int64)
    ]
    if len(origins) == 0:
        raise ValueError("validation episode is too short for the requested horizon")
    action_center, action_scale, action_normalization = checkpoint_action_normalizer(
        checkpoint, embedded_args
    )
    normalized_states = (
        np.asarray(cache.data.states, dtype=np.float32) - state_center
    ) / state_scale
    context_rows = np.stack(
        [origins - 2 * frameskip, origins - frameskip, origins], axis=1
    )
    context_states = normalized_states[context_rows]
    past_actions = np.stack(
        [
            np.stack(
                [
                    normalized_action_chunk(
                        cache,
                        int(origin - 2 * frameskip),
                        frameskip,
                        action_center,
                        action_scale,
                    ),
                    normalized_action_chunk(
                        cache,
                        int(origin - frameskip),
                        frameskip,
                        action_center,
                        action_scale,
                    ),
                ]
            )
            for origin in origins
        ]
    )
    future_actions = np.stack(
        [
            np.stack(
                [
                    normalized_action_chunk(
                        cache,
                        int(origin + step * frameskip),
                        frameskip,
                        action_center,
                        action_scale,
                    )
                    for step in range(args.horizon)
                ]
            )
            for origin in origins
        ]
    )
    future_rows = np.stack(
        [origins + (step + 1) * frameskip for step in range(args.horizon)], axis=1
    )
    target_states = normalized_states[future_rows]

    view_metrics: dict[str, Any] = {}
    pooled_predictions: dict[str, list[np.ndarray]] = {name: [] for name in METHODS}
    pooled_targets: list[np.ndarray] = []
    views = dataset.manifest["camera"]["views"]
    for camera_index, view in enumerate(views):
        clean_context_indices = lookup[context_rows, camera_index, 0]
        drift_context_indices = lookup[context_rows, camera_index, 1]
        clean_target_indices = lookup[future_rows, camera_index, 0]
        target_latents = base[clean_target_indices]
        method_contexts = {
            "clean_base": base[clean_context_indices],
            "clean_fcpg": bridged[clean_context_indices],
            "drift_base": base[drift_context_indices],
            "drift_fcpg": bridged[drift_context_indices],
        }
        predictions: dict[str, np.ndarray] = {}
        state_predictions: dict[str, np.ndarray] = {}
        for name in METHODS:
            predictions[name], state_predictions[name] = rollout(
                model,
                method_contexts[name],
                context_states,
                past_actions,
                future_actions,
            )
            pooled_predictions[name].append(predictions[name])
        pooled_targets.append(target_latents)
        method_metrics = {
            name: metric_summary(predictions[name], target_latents) for name in METHODS
        }
        method_metrics["drift_fcpg_effect"] = relative_effect(
            method_metrics["drift_base"], method_metrics["drift_fcpg"]
        )
        method_metrics["clean_fcpg_effect"] = relative_effect(
            method_metrics["clean_base"], method_metrics["clean_fcpg"]
        )
        state_target = target_states
        method_metrics["state_mse"] = {
            name: float(np.mean(np.square(state_predictions[name] - state_target)))
            for name in METHODS
        }
        view_metrics[str(view["name"])] = method_metrics

    pooled_target = np.concatenate(pooled_targets, axis=0)
    pooled_method_metrics = {
        name: metric_summary(np.concatenate(pooled_predictions[name], axis=0), pooled_target)
        for name in METHODS
    }
    pooled_method_metrics["drift_fcpg_effect"] = relative_effect(
        pooled_method_metrics["drift_base"], pooled_method_metrics["drift_fcpg"]
    )
    pooled_method_metrics["clean_fcpg_effect"] = relative_effect(
        pooled_method_metrics["clean_base"], pooled_method_metrics["clean_fcpg"]
    )
    report = {
        "format": "so101-fcpg-lewm-future-gate-v1",
        "scope": "offline held-out episode with real images and synthetic affine drift",
        "robot_commands_sent": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "bridge_checkpoint": str(args.bridge_checkpoint.resolve()),
        "dataset": str(dataset.root),
        "validation_episode": validation_episode,
        "origins": len(origins),
        "frameskip": frameskip,
        "history": history,
        "horizon": args.horizon,
        "action_normalization": action_normalization,
        "metrics": {
            "pooled": pooled_method_metrics,
            "views": view_metrics,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
