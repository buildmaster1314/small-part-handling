#!/usr/bin/env python3
"""Cache a shared frozen oracle-anchor target for grounded LeWM Gate L1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_oracle_target import (  # noqa: E402
    FrozenOracleTargetEncoder,
    load_frozen_oracle_target_encoder,
    save_frozen_oracle_target_encoder,
)
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.lewm_sequence_data import frame_identity_sha256  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--vision-cache", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-dim", type=int, default=192)
    parser.add_argument("--projection-seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _same_path(value: object, expected: Path) -> bool:
    return bool(value) and Path(str(value)).expanduser().resolve() == expected.resolve()


def _load_or_create_encoder(
    path: Path,
    *,
    visual_dim: int,
    num_keypoints: int,
    target_dim: int,
    projection_seed: int,
) -> FrozenOracleTargetEncoder:
    if path.exists():
        encoder, _ = load_frozen_oracle_target_encoder(path)
        expected = (visual_dim, num_keypoints, target_dim, projection_seed)
        actual = (
            encoder.visual_dim,
            encoder.num_keypoints,
            encoder.target_dim,
            encoder.projection_seed,
        )
        if actual != expected:
            raise ValueError(
                f"oracle target encoder contract mismatch: expected={expected} actual={actual}"
            )
        return encoder
    encoder = FrozenOracleTargetEncoder(
        visual_dim=visual_dim,
        num_keypoints=num_keypoints,
        target_dim=target_dim,
        projection_seed=projection_seed,
    )
    save_frozen_oracle_target_encoder(path, encoder)
    return encoder


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = args.data.expanduser().resolve()
    vision_cache = args.vision_cache.expanduser().resolve()
    encoder_path = args.encoder_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    target_path = output_dir / "oracle_target.npy"
    metadata_path = output_dir / "metadata.json"
    existing = [path for path in (target_path, metadata_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite oracle target cache: {existing}")
    vision_metadata_path = vision_cache / "metadata.json"
    vision_metadata = json.loads(vision_metadata_path.read_text(encoding="utf-8"))
    if not _same_path(vision_metadata.get("source_data"), data_path):
        raise ValueError("vision cache source_data does not match --data")

    feature_maps = np.load(
        vision_cache / "backbone_features.npy", mmap_mode="r", allow_pickle=False
    )
    global_features = np.load(
        vision_cache / "prefix.npy", mmap_mode="r", allow_pickle=False
    )
    with np.load(data_path, allow_pickle=False) as raw:
        required = {"q_true", "valid_true", "rgb", "episode_id", "frame_id"}
        missing = required.difference(raw.files)
        if missing:
            raise KeyError(f"dataset is missing oracle target fields: {sorted(missing)}")
        q_true = np.asarray(raw["q_true"], dtype=np.float32)
        valid_true = np.asarray(raw["valid_true"], dtype=bool)
        sample_count = int(raw["rgb"].shape[0])
        image_size = (int(raw["rgb"].shape[2]), int(raw["rgb"].shape[1]))
        episode_id = np.asarray(raw["episode_id"])
        frame_id = np.asarray(raw["frame_id"], dtype=np.int64)

    if feature_maps.ndim != 5 or feature_maps.shape[1] != 1:
        raise ValueError(
            "oracle target cache currently requires exactly one cached ViT layer, "
            f"got {feature_maps.shape}"
        )
    visual_dim = int(feature_maps.shape[2])
    if feature_maps.shape[0] != sample_count or global_features.shape != (
        sample_count,
        visual_dim,
    ):
        raise ValueError("vision cache sample count or feature dimension mismatch")
    if q_true.shape[:2] != valid_true.shape or q_true.shape != (
        sample_count,
        valid_true.shape[1],
        2,
    ):
        raise ValueError("q_true/valid_true shapes are inconsistent")

    encoder = _load_or_create_encoder(
        encoder_path,
        visual_dim=visual_dim,
        num_keypoints=int(q_true.shape[1]),
        target_dim=int(args.target_dim),
        projection_seed=int(args.projection_seed),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    encoder = encoder.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, encoder.target_dim),
    )
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    with torch.inference_mode():
        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            feature = torch.from_numpy(
                np.array(feature_maps[start:stop, 0], dtype=np.float16, copy=True)
            ).to(device)
            global_feature = torch.from_numpy(
                np.array(global_features[start:stop], dtype=np.float16, copy=True)
            ).to(device)
            q = torch.from_numpy(np.array(q_true[start:stop], copy=True)).to(device)
            valid = torch.from_numpy(np.array(valid_true[start:stop], copy=True)).to(device)
            target = encoder.forward_from_features(
                feature,
                global_feature,
                q,
                valid,
                image_size=image_size,
            )
            store[start:stop] = target.cpu().numpy().astype(np.float32, copy=False)
    store.flush()

    metadata = {
        "schema_version": 1,
        "cache_type": "fcpg_oracle_target_v1",
        "source_data": str(data_path),
        "source_size_bytes": data_path.stat().st_size,
        "source_data_sha256": sha256_file(data_path),
        "sample_count": sample_count,
        "num_keypoints": encoder.num_keypoints,
        "target_dim": encoder.target_dim,
        "dtype": "float32",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "vision_cache": str(vision_cache),
        "vision_metadata_sha256": sha256_file(vision_metadata_path),
        "backbone_features_sha256": sha256_file(
            vision_cache / "backbone_features.npy"
        ),
        "prefix_sha256": sha256_file(vision_cache / "prefix.npy"),
        "encoder_checkpoint": str(encoder_path),
        "encoder_checkpoint_sha256": sha256_file(encoder_path),
        "encoder_spec": encoder.spec(),
        "frame_identity_sha256": frame_identity_sha256(episode_id, frame_id),
        "oracle_target_sha256": sha256_file(target_path),
        "q_true_usage": "target_construction_only",
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
