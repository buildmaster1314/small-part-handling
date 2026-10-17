#!/usr/bin/env python3
"""Cache compact policy-ready visual features at corrected keypoints."""

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

from fcpg.modules import sample_feature_map_at_pixels  # noqa: E402
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402


def _read_metadata(path: Path) -> dict[str, object]:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _same_path(value: object, expected: Path) -> bool:
    return value is not None and Path(str(value)).resolve() == expected.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--vision-cache", required=True)
    parser.add_argument("--grounding-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    vision_cache = Path(args.vision_cache).resolve()
    grounding_cache = Path(args.grounding_cache).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_path = output_dir / "sampled_visual.npy"
    global_path = output_dir / "global_visual.npy"
    metadata_path = output_dir / "metadata.json"
    existing = [path for path in (sampled_path, global_path, metadata_path) if path.exists()]
    if existing:
        parser.error(f"cache output already exists: {existing[0]}")

    vision_metadata = _read_metadata(vision_cache)
    grounding_metadata = _read_metadata(grounding_cache)
    if not _same_path(vision_metadata.get("source_data"), data_path):
        raise ValueError("vision cache source does not match --data")
    if not _same_path(grounding_metadata.get("source_data"), data_path):
        raise ValueError("grounding cache source does not match --data")
    if not _same_path(grounding_metadata.get("vision_cache"), vision_cache):
        raise ValueError("grounding cache was generated from a different vision cache")
    if not _same_path(grounding_metadata.get("checkpoint"), checkpoint):
        raise ValueError("grounding cache was generated from a different checkpoint")

    feature_maps = np.load(vision_cache / "backbone_features.npy", mmap_mode="r")
    prefix = np.load(vision_cache / "prefix.npy", mmap_mode="r")
    q_corrected = np.load(grounding_cache / "q_corrected.npy", mmap_mode="r")
    sample_count, num_keypoints, coordinate_dim = q_corrected.shape
    if coordinate_dim != 2:
        raise ValueError("q_corrected must have shape [N, K, 2]")
    if feature_maps.shape[0] != sample_count or prefix.shape[0] != sample_count:
        raise ValueError("vision and grounding cache sample counts differ")

    device = torch.device(args.device)
    model, checkpoint_args = load_fcpg_vit_adapter(checkpoint, device)
    model.eval()
    model.vision.eval()
    checkpoint_layers = tuple(
        int(layer)
        for layer in checkpoint_args.get("vision_selected_layers", (2, 5, 8, 11))
    )
    cache_layers = tuple(int(layer) for layer in vision_metadata["selected_layers"])
    if checkpoint_layers != cache_layers:
        raise ValueError(
            f"checkpoint layers {checkpoint_layers} do not match cache layers {cache_layers}"
        )
    model_name = str(checkpoint_args.get("vision_model_name", model.vision.model_name))
    if str(vision_metadata["model_name"]) != model_name:
        raise ValueError("checkpoint and vision cache model names differ")

    visual_dim = int(model.vision.output_dim)
    image_size = (
        int(vision_metadata["image_width"]),
        int(vision_metadata["image_height"]),
    )
    sampled_store = np.lib.format.open_memmap(
        sampled_path,
        mode="w+",
        dtype=np.float16,
        shape=(sample_count, num_keypoints, visual_dim),
    )
    global_store = np.lib.format.open_memmap(
        global_path,
        mode="w+",
        dtype=np.float16,
        shape=(sample_count, visual_dim),
    )

    torch.set_float32_matmul_precision("high")
    with torch.inference_mode():
        for start in range(0, sample_count, args.batch_size):
            stop = min(start + args.batch_size, sample_count)
            batch_features = torch.from_numpy(
                np.array(feature_maps[start:stop], dtype=np.float16, copy=True)
            ).to(device, non_blocking=True)
            batch_prefix = torch.from_numpy(
                np.array(prefix[start:stop], dtype=np.float16, copy=True)
            ).to(device, non_blocking=True)
            batch_points = torch.from_numpy(
                np.array(q_corrected[start:stop], dtype=np.float32, copy=True)
            ).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                vision = model.vision.project_backbone_features(
                    batch_features,
                    batch_prefix,
                )
                sampled = sample_feature_map_at_pixels(
                    vision.feature_map,
                    batch_points,
                    image_size=image_size,
                )
            sampled_store[start:stop] = sampled.cpu().numpy().astype(np.float16)
            global_store[start:stop] = (
                vision.global_feature.cpu().numpy().astype(np.float16)
            )
            if stop == sample_count or stop % (args.batch_size * 10) == 0:
                print(f"cached {stop}/{sample_count}", flush=True)
    sampled_store.flush()
    global_store.flush()

    raw_bytes = int(feature_maps.nbytes + prefix.nbytes)
    compact_bytes = int(sampled_store.nbytes + global_store.nbytes)
    metadata = {
        "version": 1,
        "schema_version": 1,
        "cache_type": "fcpg_policy_visual",
        "point_source": "fcpg_q_corrected",
        "source_data": str(data_path),
        "vision_cache": str(vision_cache),
        "grounding_cache": str(grounding_cache),
        "checkpoint": str(checkpoint),
        "sample_count": int(sample_count),
        "num_keypoints": int(num_keypoints),
        "visual_dim": visual_dim,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "model_name": model_name,
        "selected_layers": list(cache_layers),
        "dtype": "float16",
        "raw_cache_bytes": raw_bytes,
        "compact_cache_bytes": compact_bytes,
        "compression_ratio": compact_bytes / max(raw_bytes, 1),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"sampled_visual: {sampled_path}")
    print(f"global_visual: {global_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
