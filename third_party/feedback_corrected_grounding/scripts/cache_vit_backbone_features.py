#!/usr/bin/env python3
"""Cache frozen ViT intermediate maps for repeated policy training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.transformer_grounding import DEFAULT_VIT_MODEL, ViTDenseEncoder  # noqa: E402
from fcpg.lewm_provenance import sha256_file  # noqa: E402


IMAGE_VIEW_ROLES = {
    "rgb": "deployment_observation",
    "rgb_camera_only": "target_only",
}
_HASH_CHUNK_BYTES = 1024 * 1024


def _validate_image_array(
    value: np.ndarray,
    *,
    image_key: str,
    expected_sample_count: int | None = None,
) -> np.ndarray:
    images = np.asarray(value)
    if images.ndim != 4 or images.shape[-1] != 3 or any(
        int(size) <= 0 for size in images.shape[:3]
    ):
        raise ValueError(
            f"dataset image array {image_key!r} must have shape [N, H, W, 3] "
            f"with positive dimensions, got {images.shape}"
        )
    if images.dtype != np.uint8:
        raise ValueError(
            f"dataset image array {image_key!r} must have dtype uint8, "
            f"got {images.dtype}"
        )
    if (
        expected_sample_count is not None
        and images.shape[0] != expected_sample_count
    ):
        raise ValueError(
            f"dataset image array {image_key!r} has {images.shape[0]} samples; "
            f"expected {expected_sample_count}"
        )
    return images


def _selected_image_sha256(value: np.ndarray) -> str:
    """Hash logical C-order image bytes without materializing another full array."""

    array = np.ascontiguousarray(np.asarray(value))
    byte_view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(byte_view), _HASH_CHUNK_BYTES):
        digest.update(byte_view[start : start + _HASH_CHUNK_BYTES])
    return digest.hexdigest()


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("at least one ViT layer is required")
    return layers


def _load_embedded_vision_weights(
    model: ViTDenseEncoder, checkpoint_path: Path
) -> None:
    """Load the exact frozen vision module already embedded in an FCPG checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a model_state mapping")
    prefix = "vision."
    vision_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if not vision_state:
        raise ValueError("checkpoint model_state does not contain vision weights")
    model.load_state_dict(vision_state, strict=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-key",
        choices=tuple(IMAGE_VIEW_ROLES),
        default="rgb",
        help="NPZ image array to encode",
    )
    parser.add_argument("--model-name", default=DEFAULT_VIT_MODEL)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="FCPG checkpoint containing vision.* weights; avoids network lookup",
    )
    parser.add_argument("--selected-layers", type=parse_layers, default=(11,))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "backbone_features.npy"
    prefix_path = output_dir / "prefix.npy"
    metadata_path = output_dir / "metadata.json"
    existing = [path for path in (feature_path, prefix_path, metadata_path) if path.exists()]
    if existing:
        parser.error(f"cache output already exists: {existing[0]}")

    data_path = Path(args.data).expanduser().resolve()
    with np.load(data_path, allow_pickle=False) as data:
        if args.image_key not in data.files:
            raise KeyError(
                f"dataset does not contain requested image key {args.image_key!r}"
            )
        rgb = _validate_image_array(data[args.image_key], image_key=args.image_key)
        sample_count = int(rgb.shape[0])
        for count_key in ("q_nominal", "episode_id", "frame_id"):
            if count_key not in data.files:
                continue
            count_value = np.asarray(data[count_key])
            if count_value.ndim == 0 or count_value.shape[0] != sample_count:
                actual = "scalar" if count_value.ndim == 0 else count_value.shape[0]
                raise ValueError(
                    f"dataset image array {args.image_key!r} has {sample_count} samples, "
                    f"but {count_key!r} has {actual}"
                )
        selected_image_sha256 = _selected_image_sha256(rgb)
    sample_count, image_height, image_width, channels = rgb.shape
    assert channels == 3
    view_role = IMAGE_VIEW_ROLES[args.image_key]

    device = torch.device(args.device)
    model = ViTDenseEncoder(
        model_name=args.model_name,
        pretrained=args.checkpoint is None,
        selected_layers=args.selected_layers,
        output_dim=256,
        freeze_backbone=True,
    )
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        _load_embedded_vision_weights(model, checkpoint_path)
    model = model.to(device).eval()
    feature_store = None
    prefix_store = None
    with torch.inference_mode():
        for start in range(0, sample_count, args.batch_size):
            stop = min(start + args.batch_size, sample_count)
            image = torch.from_numpy(rgb[start:stop]).permute(0, 3, 1, 2).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ) / 255.0
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                features, prefix = model.extract_backbone_features(image)
            features_np = features.to(dtype=torch.float16).cpu().numpy()
            prefix_np = prefix.to(dtype=torch.float16).cpu().numpy()
            if feature_store is None:
                feature_store = np.lib.format.open_memmap(
                    feature_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(sample_count, *features_np.shape[1:]),
                )
                prefix_store = np.lib.format.open_memmap(
                    prefix_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(sample_count, prefix_np.shape[-1]),
                )
            feature_store[start:stop] = features_np
            assert prefix_store is not None
            prefix_store[start:stop] = prefix_np
            if stop == sample_count or stop % (args.batch_size * 10) == 0:
                print(f"cached {stop}/{sample_count}", flush=True)

    assert feature_store is not None and prefix_store is not None
    feature_store.flush()
    prefix_store.flush()
    metadata = {
        "version": 1,
        "source_data": str(data_path),
        "source_size_bytes": int(data_path.stat().st_size),
        "source_data_sha256": sha256_file(data_path),
        "image_key": args.image_key,
        "selected_image_sha256": selected_image_sha256,
        "view_role": view_role,
        "deployment_cache_eligible": view_role == "deployment_observation",
        "sample_count": int(sample_count),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "model_name": args.model_name,
        "weight_source": (
            "pretrained_registry"
            if args.checkpoint is None
            else str(args.checkpoint.expanduser().resolve())
        ),
        "selected_layers": list(args.selected_layers),
        "feature_shape": list(feature_store.shape[1:]),
        "prefix_dim": int(prefix_store.shape[-1]),
        "dtype": "float16",
        "backbone_features_sha256": sha256_file(feature_path),
        "prefix_sha256": sha256_file(prefix_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"features: {feature_path} {feature_store.shape}")
    print(f"prefix: {prefix_path} {prefix_store.shape}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
