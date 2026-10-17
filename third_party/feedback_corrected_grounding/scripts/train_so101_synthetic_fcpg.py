#!/usr/bin/env python3
"""Train a small SO-101 FCPG corrector on synthetic camera drift."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fcpg.modules import sample_feature_map_at_pixels
from fcpg.so101_fcpg_lewm_bridge import (
    SO101_FCPG_CORRECTOR_VARIANTS,
    SO101FCPGConditionedLeWM,
    build_so101_fcpg_corrector,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from fcpg.so101_synthetic_dataset import SO101SyntheticFCPGDataset
from scripts.train_so101_lewm import IMAGENET_MEAN, IMAGENET_STD
from scripts.visualize_so101_lewm import (
    build_checkpoint_model,
    checkpoint_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=53102)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--corrector-variant",
        choices=SO101_FCPG_CORRECTOR_VARIANTS,
        default="independent3",
    )
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.feature_batch_size, args.log_every) <= 0:
        parser.error("step and batch arguments must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("optimizer arguments are invalid")
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    return args


def normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    pixels = torch.from_numpy(np.asarray(rgb).copy()).permute(0, 3, 1, 2)
    pixels = pixels.float().div_(255.0)
    return (pixels - IMAGENET_MEAN) / IMAGENET_STD


def masked_smooth_l1(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if predicted_delta.shape != target_delta.shape or predicted_delta.shape[-1] != 2:
        raise ValueError("delta tensors must share shape [B, N, 2]")
    if valid.shape != predicted_delta.shape[:-1]:
        raise ValueError("valid mask must align with delta tensors")
    per_coordinate = F.smooth_l1_loss(
        predicted_delta,
        target_delta,
        beta=1.0,
        reduction="none",
    )
    weights = valid.to(dtype=per_coordinate.dtype).unsqueeze(-1)
    denominator = weights.sum() * predicted_delta.shape[-1]
    if float(denominator.detach().cpu()) <= 0:
        raise ValueError("training batch has no valid anchors")
    return (per_coordinate * weights).sum() / denominator


def metric_row(
    q_nominal: np.ndarray,
    q_corrected: np.ndarray,
    q_true: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    support = np.asarray(valid, dtype=bool)
    if not np.any(support):
        return {"anchors": 0, "nominal_epe_px": float("nan"), "corrected_epe_px": float("nan")}
    nominal = np.linalg.norm(q_nominal[support] - q_true[support], axis=-1)
    corrected = np.linalg.norm(q_corrected[support] - q_true[support], axis=-1)
    return {
        "anchors": int(len(nominal)),
        "nominal_epe_px": float(nominal.mean()),
        "corrected_epe_px": float(corrected.mean()),
    }


def aggregate_metrics(
    dataset: SO101SyntheticFCPGDataset,
    indices: np.ndarray,
    corrected: np.ndarray,
) -> dict[str, Any]:
    arrays = dataset.arrays
    views = dataset.manifest["camera"]["views"]
    buckets: dict[str, list[int]] = defaultdict(list)
    for raw_index in indices:
        index = int(raw_index)
        camera = str(views[int(arrays["camera_index"][index])]["name"])
        clean = not bool(np.any(np.abs(np.asarray(arrays["drift"][index])) > 1.0e-8))
        buckets["all"].append(index)
        buckets["clean" if clean else "drift"].append(index)
        buckets[f"{camera}/all"].append(index)
        buckets[f"{camera}/{'clean' if clean else 'drift'}"].append(index)

    metrics: dict[str, Any] = {}
    for name, rows in sorted(buckets.items()):
        valid = np.asarray(arrays["valid"][rows], dtype=bool)
        nominal = np.asarray(arrays["q_nominal"][rows])
        target = np.asarray(arrays["q_true"][rows])
        row = metric_row(nominal, corrected[rows], target, valid)
        nominal_epe = float(row["nominal_epe_px"])
        corrected_epe = float(row["corrected_epe_px"])
        row["frames"] = len(rows)
        row["relative_improvement"] = (
            (nominal_epe - corrected_epe) / nominal_epe
            if nominal_epe > 1.0e-8
            else float("nan")
        )
        metrics[name] = row
    return metrics


@torch.inference_mode()
def token_sampling_metrics(
    dataset: SO101SyntheticFCPGDataset,
    indices: np.ndarray,
    features: torch.Tensor,
    corrected: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    arrays = dataset.arrays
    width = int(dataset.manifest["width"])
    height = int(dataset.manifest["height"])
    nominal_mse: list[float] = []
    corrected_mse: list[float] = []
    nominal_cosine: list[float] = []
    corrected_cosine: list[float] = []
    for begin in range(0, len(indices), batch_size):
        rows = np.asarray(indices[begin : begin + batch_size], dtype=np.int64)
        row_tensor = torch.from_numpy(rows)
        feature = features[row_tensor].to(device=device, dtype=torch.float32)
        nominal = torch.from_numpy(np.asarray(arrays["q_nominal"][rows]).copy()).to(device)
        target = torch.from_numpy(np.asarray(arrays["q_true"][rows]).copy()).to(device)
        predicted = torch.from_numpy(np.asarray(corrected[rows]).copy()).to(device)
        valid = torch.from_numpy(np.asarray(arrays["valid"][rows]).copy()).to(device)
        nominal_tokens = sample_feature_map_at_pixels(
            feature, nominal, image_size=(width, height), padding_mode="border"
        )
        corrected_tokens = sample_feature_map_at_pixels(
            feature, predicted, image_size=(width, height), padding_mode="border"
        )
        target_tokens = sample_feature_map_at_pixels(
            feature, target, image_size=(width, height), padding_mode="border"
        )
        mse_nominal = (nominal_tokens - target_tokens).square().mean(dim=-1)
        mse_corrected = (corrected_tokens - target_tokens).square().mean(dim=-1)
        cosine_nominal = 1.0 - F.cosine_similarity(
            nominal_tokens, target_tokens, dim=-1
        )
        cosine_corrected = 1.0 - F.cosine_similarity(
            corrected_tokens, target_tokens, dim=-1
        )
        nominal_mse.extend(mse_nominal[valid].cpu().tolist())
        corrected_mse.extend(mse_corrected[valid].cpu().tolist())
        nominal_cosine.extend(cosine_nominal[valid].cpu().tolist())
        corrected_cosine.extend(cosine_corrected[valid].cpu().tolist())
    if not nominal_mse:
        raise ValueError("token metric split has no valid anchors")
    nominal_mse_mean = float(np.mean(nominal_mse))
    corrected_mse_mean = float(np.mean(corrected_mse))
    nominal_cosine_mean = float(np.mean(nominal_cosine))
    corrected_cosine_mean = float(np.mean(corrected_cosine))
    return {
        "anchors": len(nominal_mse),
        "nominal_feature_mse": nominal_mse_mean,
        "corrected_feature_mse": corrected_mse_mean,
        "feature_mse_relative_improvement": (
            (nominal_mse_mean - corrected_mse_mean) / nominal_mse_mean
            if nominal_mse_mean > 1.0e-12
            else float("nan")
        ),
        "nominal_cosine_distance": nominal_cosine_mean,
        "corrected_cosine_distance": corrected_cosine_mean,
        "cosine_relative_improvement": (
            (nominal_cosine_mean - corrected_cosine_mean) / nominal_cosine_mean
            if nominal_cosine_mean > 1.0e-12
            else float("nan")
        ),
    }


@torch.inference_mode()
def cache_feature_maps(
    system: SO101FCPGConditionedLeWM,
    dataset: SO101SyntheticFCPGDataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    rgb = dataset.arrays["rgb"]
    features: list[torch.Tensor] = []
    system.eval()
    for begin in range(0, len(dataset), batch_size):
        end = min(begin + batch_size, len(dataset))
        pixels = normalize_rgb(np.asarray(rgb[begin:end])).to(device)
        pixels = pixels.unsqueeze(1)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            _, _, feature_map = system._encode_shared_vision(pixels)
        features.append(feature_map.detach().to(device="cpu", dtype=torch.float16))
        print(f"encoded features {end}/{len(dataset)}", flush=True)
    return torch.cat(features, dim=0)


def correct_all(
    system: SO101FCPGConditionedLeWM,
    dataset: SO101SyntheticFCPGDataset,
    features: torch.Tensor,
    normalized_state: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    arrays = dataset.arrays
    result = np.empty_like(np.asarray(arrays["q_nominal"]), dtype=np.float32)
    system.corrector.eval()
    with torch.inference_mode():
        for begin in range(0, len(dataset), batch_size):
            end = min(begin + batch_size, len(dataset))
            rows = slice(begin, end)
            feature = features[rows].to(device=device, dtype=torch.float32)
            nominal = torch.from_numpy(np.asarray(arrays["q_nominal"][rows]).copy()).to(device)
            xyz = torch.from_numpy(np.asarray(arrays["anchors_xyz"][rows]).copy()).to(device)
            state = torch.from_numpy(normalized_state[rows]).to(device)
            camera = torch.from_numpy(np.asarray(arrays["camera_context"][rows]).copy()).to(device)
            observation = torch.cat((state, camera), dim=-1)
            output = system.corrector(
                feature,
                nominal,
                xyz,
                observation,
                image_size=(
                    int(dataset.manifest["width"]),
                    int(dataset.manifest["height"]),
                ),
            )
            result[rows] = output.q_corrected_px.detach().cpu().numpy()
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = SO101SyntheticFCPGDataset(args.dataset)
    checkpoint, _, _, frameskip, history, checkpoint_sha = checkpoint_contract(
        args.checkpoint.resolve(), None
    )
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    if not isinstance(model, StateResidualLeWM):
        raise TypeError("SO-101 FCPG training requires a state-residual LeWM checkpoint")
    corrector = build_so101_fcpg_corrector(
        model,
        camera_context_dim=5,
        variant=args.corrector_variant,
    )
    system = SO101FCPGConditionedLeWM(
        model,
        corrector=corrector,
        camera_context_dim=5,
        freeze_model=True,
    ).to(device)
    system.bridge.requires_grad_(False)
    features = cache_feature_maps(
        system, dataset, device, args.feature_batch_size
    )

    contract = checkpoint["run_config"]["contract"]
    center = np.asarray(contract["normalization_center"], dtype=np.float32)
    scale = np.asarray(contract["normalization_scale"], dtype=np.float32)
    states = np.asarray(dataset.arrays["states"], dtype=np.float32)
    normalized_state = (states - center) / scale
    train_indices = torch.from_numpy(dataset.indices("train")).long()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    optimizer = torch.optim.AdamW(
        system.corrector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    width = int(dataset.manifest["width"])
    height = int(dataset.manifest["height"])
    arrays = dataset.arrays
    history_rows: list[dict[str, float | int]] = []
    system.corrector.train()
    for step in range(1, args.steps + 1):
        sampled = train_indices[
            torch.randint(
                len(train_indices),
                (args.batch_size,),
                generator=generator,
            )
        ]
        rows = sampled.numpy()
        feature = features[sampled].to(device=device, dtype=torch.float32)
        nominal = torch.from_numpy(np.asarray(arrays["q_nominal"][rows]).copy()).to(device)
        target = torch.from_numpy(np.asarray(arrays["q_true"][rows]).copy()).to(device)
        xyz = torch.from_numpy(np.asarray(arrays["anchors_xyz"][rows]).copy()).to(device)
        valid = torch.from_numpy(np.asarray(arrays["valid"][rows]).copy()).to(device)
        state = torch.from_numpy(normalized_state[rows]).to(device)
        camera = torch.from_numpy(np.asarray(arrays["camera_context"][rows]).copy()).to(device)
        observation = torch.cat((state, camera), dim=-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            prediction = system.corrector(
                feature,
                nominal,
                xyz,
                observation,
                image_size=(width, height),
            )
            loss = masked_smooth_l1(
                prediction.delta_px,
                target - nominal,
                valid,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.corrector.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"step": step, "train_loss": float(loss.detach().cpu())}
            history_rows.append(row)
            print(json.dumps(row), flush=True)

    corrected = correct_all(
        system,
        dataset,
        features,
        normalized_state,
        device,
        args.feature_batch_size,
    )
    np.save(output / "corrected_q.npy", corrected)
    metrics = {
        "train": aggregate_metrics(dataset, dataset.indices("train"), corrected),
        "val": aggregate_metrics(dataset, dataset.indices("val"), corrected),
    }
    token_metrics = {
        "train": token_sampling_metrics(
            dataset,
            dataset.indices("train"),
            features,
            corrected,
            device,
            args.feature_batch_size,
        ),
        "val": token_sampling_metrics(
            dataset,
            dataset.indices("val"),
            features,
            corrected,
            device,
            args.feature_batch_size,
        ),
    }
    payload = {
        "format": "so101-synthetic-fcpg-corrector-v1",
        "step": args.steps,
        "seed": args.seed,
        "dataset": str(dataset.root),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha,
        "camera_context_dim": 5,
        "corrector_variant": args.corrector_variant,
        "corrector": {
            "parameters": sum(parameter.numel() for parameter in system.corrector.parameters()),
            "local_radius_px": system.corrector.local_radius_px,
            "max_residual_px": system.corrector.max_residual_px,
        },
        "scope": "synthetic dual-camera pixel-correction gate; not real-camera evidence",
        "metrics": metrics,
        "token_sampling_metrics": token_metrics,
        "history": history_rows,
    }
    torch.save(
        {
            "corrector": system.corrector.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "token_sampling_metrics": token_metrics,
            "source_checkpoint_sha256": checkpoint_sha,
            "corrector_variant": args.corrector_variant,
        },
        output / "latest.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(metrics["val"], indent=2), flush=True)
    print(json.dumps({"val_token_sampling": token_metrics["val"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
