#!/usr/bin/env python3
"""Evaluate deterministic action chunks on held-out complete episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.sequence_data import ProjectionSequenceDataset  # noqa: E402
from scripts.train_bc_policy import make_split  # noqa: E402


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def representation_statistics(latent: np.ndarray) -> dict[str, float | int]:
    """Summarize held-out latent spread without fitting another estimator."""
    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"latent must have shape [N, D] with N > 0, got {values.shape}")
    centered = values - values.mean(axis=0, keepdims=True)
    std = values.std(axis=0)
    singular = np.linalg.svd(
        centered / np.sqrt(max(values.shape[0] - 1, 1)),
        full_matrices=False,
        compute_uv=False,
    )
    spectrum = np.square(singular)
    total = float(spectrum.sum())
    if total > 0.0:
        probability = spectrum / total
        nonzero = probability > 0.0
        entropy = -(probability[nonzero] * np.log(probability[nonzero])).sum()
        effective_rank = float(np.exp(entropy))
        participation_ratio = float(total * total / np.square(spectrum).sum())
    else:
        effective_rank = 0.0
        participation_ratio = 0.0
    return {
        "latent_samples": int(values.shape[0]),
        "latent_dim": int(values.shape[1]),
        "latent_mean_norm": float(np.linalg.norm(values.mean(axis=0))),
        "latent_rms": float(np.sqrt(np.square(values).mean())),
        "latent_std_mean": float(std.mean()),
        "latent_std_min": float(std.min()),
        "latent_std_max": float(std.max()),
        "latent_effective_rank": effective_rank,
        "latent_participation_ratio": participation_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vision-cache", default="")
    parser.add_argument("--grounding-cache", default="")
    parser.add_argument("--policy-cache", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, checkpoint_args = load_action_chunk_policy(args.checkpoint, device)
    with np.load(args.data, allow_pickle=False) as raw:
        split = make_split(raw, 0.2, args.split_seed, "task-episode")
    dataset = ProjectionSequenceDataset(
        args.data,
        split.test,
        obs_mode=str(checkpoint_args.get("obs_mode", "robot")),
        obs_horizon=int(checkpoint_args.get("obs_horizon", 2)),
        action_horizon=int(checkpoint_args.get("action_horizon", 32)),
        vision_cache=args.vision_cache or None,
        grounding_cache=args.grounding_cache or None,
        policy_cache=args.policy_cache or None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    squared: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    first_squared: list[np.ndarray] = []
    gripper_correct: list[np.ndarray] = []
    state_latents: list[np.ndarray] = []
    next_latent_squared: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            batch = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model.predict_action_chunk(batch)
            error = output.action.float() - batch["action"]
            mask = batch["action_mask"].unsqueeze(-1).expand_as(error)
            squared.append(error.square()[mask].cpu().numpy())
            absolute.append(error.abs()[mask].cpu().numpy())
            first_squared.append(error[:, 0].square().cpu().numpy().reshape(-1))
            valid_gripper = batch["action_mask"]
            gripper_correct.append(
                ((output.action[..., 3] >= 0.5) == (batch["action"][..., 3] >= 0.5))[valid_gripper]
                .cpu()
                .numpy()
            )
            if output.state_latent is not None:
                state_latents.append(output.state_latent.float().cpu().numpy())
            if output.next_latent is not None and output.next_latent_target is not None:
                next_mask = batch["next_mask"].bool()
                if bool(next_mask.any()):
                    next_error = (
                        output.next_latent.float() - output.next_latent_target.float()
                    ).square().mean(dim=-1)
                    next_latent_squared.append(next_error[next_mask].cpu().numpy())
    payload = {
        "data": args.data,
        "checkpoint": args.checkpoint,
        "mode": model.mode,
        "policy_arch": model.policy_arch,
        "latent_pred_weight": float(checkpoint_args.get("latent_pred_weight", 0.0)),
        "sigreg_weight": float(checkpoint_args.get("sigreg_weight", 0.0)),
        "reliability_override": str(checkpoint_args.get("reliability_override", "native")),
        "evaluated_sequences": int(sum(values.size for values in first_squared) // model.action_dim),
        "chunk_mse": float(np.concatenate(squared).mean()),
        "chunk_mae": float(np.concatenate(absolute).mean()),
        "first_action_mse": float(np.concatenate(first_squared).mean()),
        "gripper_accuracy": float(np.concatenate(gripper_correct).mean()),
    }
    if state_latents:
        payload.update(representation_statistics(np.concatenate(state_latents, axis=0)))
    if next_latent_squared:
        next_values = np.concatenate(next_latent_squared)
        payload["next_latent_samples"] = int(next_values.size)
        payload["next_latent_mse"] = float(next_values.mean())
    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
