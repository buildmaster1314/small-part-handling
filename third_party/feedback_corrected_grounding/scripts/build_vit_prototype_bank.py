#!/usr/bin/env python3
"""Build a training-split ViT descriptor bank for local-refinement baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.modules import sample_feature_map_at_pixels  # noqa: E402
from scripts.train_bc_policy import make_split  # noqa: E402


def load_keypoint_source(
    data: np.lib.npyio.NpzFile,
    keypoint_source: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load keypoints together with their semantic site order."""

    prefix = "" if keypoint_source == "primary" else "control_"
    q_key = f"{prefix}q_true"
    valid_key = f"{prefix}valid_true"
    site_key = f"{prefix}keypoint_site"
    missing = [key for key in (q_key, valid_key, site_key) if key not in data.files]
    if missing:
        raise ValueError(
            f"dataset does not provide {keypoint_source} keypoints: missing {missing}"
        )
    q_true = np.asarray(data[q_key], dtype=np.float32)
    valid = np.asarray(data[valid_key], dtype=bool)
    sites = tuple(str(value) for value in np.asarray(data[site_key]).tolist())
    if q_true.ndim != 3 or q_true.shape[-1] != 2:
        raise ValueError(f"{q_key} must have shape [N,K,2]")
    if valid.shape != q_true.shape[:-1]:
        raise ValueError(f"{valid_key} shape does not match {q_key}")
    if len(sites) != q_true.shape[1] or len(set(sites)) != len(sites):
        raise ValueError("keypoint site metadata is missing, duplicated, or misaligned")
    return q_true, valid, sites


def select_prototype_indices(
    train_indices: np.ndarray,
    valid: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    train_indices = np.asarray(train_indices, dtype=np.int64)
    eligible = train_indices[np.asarray(valid[train_indices]).all(axis=1)]
    if eligible.size == 0:
        raise ValueError("training split contains no frames with all keypoints valid")
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, size=min(count, eligible.size), replace=False)
    return np.sort(selected.astype(np.int64))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--keypoint-source", choices=("primary", "control"), default="primary"
    )
    parser.add_argument("--vision-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prototypes-per-keypoint", type=int, default=128)
    parser.add_argument("--feature-level", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prototypes_path = output_dir / "prototypes.npy"
    indices_path = output_dir / "source_indices.npy"
    metadata_path = output_dir / "metadata.json"
    for path in (prototypes_path, indices_path, metadata_path):
        if path.exists():
            parser.error(f"prototype output already exists: {path}")

    data_path = Path(args.data)
    cache_path = Path(args.vision_cache)
    cache_metadata = json.loads((cache_path / "metadata.json").read_text(encoding="utf-8"))
    source_data = cache_metadata.get("source_data")
    if source_data and Path(source_data).resolve() != data_path.resolve():
        raise ValueError("vision cache source does not match --data")
    feature_maps = np.load(cache_path / "backbone_features.npy", mmap_mode="r")
    level = args.feature_level % feature_maps.shape[1]
    with np.load(data_path, allow_pickle=False) as data:
        if "task_name" not in data.files:
            raise ValueError("prototype source must record task_name")
        training_tasks = tuple(
            sorted(
                {
                    str(value)
                    for value in np.asarray(data["task_name"]).reshape(-1).tolist()
                }
            )
        )
        if not training_tasks:
            raise ValueError("prototype source records no training tasks")
        split = make_split(data, 0.2, args.split_seed, "task-episode")
        q_true, valid, keypoint_sites = load_keypoint_source(
            data, args.keypoint_source
        )
    selected = select_prototype_indices(
        split.train,
        valid,
        args.prototypes_per_keypoint,
        args.seed,
    )
    image_size = (
        int(cache_metadata["image_width"]),
        int(cache_metadata["image_height"]),
    )
    device = torch.device(args.device)
    descriptors: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, selected.size, args.batch_size):
            indices = selected[start : start + args.batch_size]
            features = torch.from_numpy(
                np.array(feature_maps[indices, level], dtype=np.float32, copy=True)
            ).to(device)
            points = torch.from_numpy(q_true[indices]).to(device)
            sampled = sample_feature_map_at_pixels(
                features,
                points,
                image_size=image_size,
                padding_mode="border",
            )
            descriptors.append(F.normalize(sampled.float(), dim=-1).cpu())
    bank = torch.cat(descriptors, dim=0).permute(1, 0, 2).contiguous().numpy()
    np.save(prototypes_path, bank.astype(np.float32))
    np.save(indices_path, selected)
    metadata = {
        "version": 1,
        "baseline": "frozen_vit_prototype_refinement",
        "keypoint_source": args.keypoint_source,
        "keypoint_site": list(keypoint_sites),
        "training_tasks": list(training_tasks),
        "source_data": str(data_path.resolve()),
        "vision_cache": str(cache_path.resolve()),
        "model_name": cache_metadata["model_name"],
        "selected_layers": cache_metadata["selected_layers"],
        "feature_level": int(level),
        "feature_layer": int(cache_metadata["selected_layers"][level]),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "split_unit": "task-episode",
        "split_seed": args.split_seed,
        "prototype_seed": args.seed,
        "prototype_count": int(selected.size),
        "num_keypoints": int(bank.shape[0]),
        "feature_dim": int(bank.shape[-1]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"prototypes: {prototypes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
