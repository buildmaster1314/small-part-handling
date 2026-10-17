#!/usr/bin/env python3
"""Cache FCPG corrected keypoints and optional reliability for policy training."""

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

from fcpg.observations import select_policy_observation  # noqa: E402
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.lewm_sequence_data import frame_identity_sha256  # noqa: E402
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402


IMAGE_VIEW_ROLES = {
    "rgb": "deployment_observation",
    "rgb_camera_only": "target_only",
}
_HASH_CHUNK_BYTES = 1024 * 1024


def _validate_image_array(
    value: np.ndarray,
    *,
    image_key: str,
    expected_sample_count: int,
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
    if images.shape[0] != expected_sample_count:
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


def _extract_grounding_output(
    output: object,
    q_nominal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize legacy and residual-only adapter outputs without type coupling."""

    corrected = getattr(output, "q_corrected_px", None)
    if corrected is None:
        delta = getattr(output, "delta_px", None)
        if delta is None:
            raise AttributeError("grounding output must provide q_corrected_px or delta_px")
        corrected = q_nominal + delta
    reliability = getattr(output, "reliability", None)
    return corrected, reliability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--vision-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-key",
        choices=tuple(IMAGE_VIEW_ROLES),
        default="rgb",
        help="NPZ image array used to build the vision cache",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    q_path = output_dir / "q_corrected.npy"
    rho_path = output_dir / "reliability.npy"
    metadata_path = output_dir / "metadata.json"
    existing = [path for path in (q_path, rho_path, metadata_path) if path.exists()]
    if existing:
        parser.error(f"cache output already exists: {existing[0]}")

    data_path = Path(args.data).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    vision_cache = Path(args.vision_cache).expanduser().resolve()
    vision_metadata_path = vision_cache / "metadata.json"
    feature_path = vision_cache / "backbone_features.npy"
    prefix_path = vision_cache / "prefix.npy"
    vision_metadata = json.loads(vision_metadata_path.read_text(encoding="utf-8"))
    view_role = IMAGE_VIEW_ROLES[args.image_key]
    image_lineage_fields = {"image_key", "view_role", "selected_image_sha256"}
    missing_image_lineage_fields = sorted(
        image_lineage_fields.difference(vision_metadata)
    )
    legacy_default_rgb = bool(missing_image_lineage_fields) and args.image_key == "rgb"
    if missing_image_lineage_fields and not legacy_default_rgb:
        raise ValueError(
            "vision cache requires explicit image lineage for "
            f"{args.image_key!r}; missing {missing_image_lineage_fields}"
        )
    if (
        "image_key" in vision_metadata
        and vision_metadata["image_key"] != args.image_key
    ):
        raise ValueError(
            "vision cache image_key does not match requested dataset image key"
        )
    if "view_role" in vision_metadata and vision_metadata["view_role"] != view_role:
        raise ValueError(
            f"vision cache view_role must be {view_role!r} for image key "
            f"{args.image_key!r}"
        )
    deployment_cache_eligible = view_role == "deployment_observation"
    if (
        "deployment_cache_eligible" in vision_metadata
        and vision_metadata["deployment_cache_eligible"]
        is not deployment_cache_eligible
    ):
        raise ValueError(
            "vision cache deployment_cache_eligible contradicts its view_role"
        )
    source_data = vision_metadata.get("source_data")
    if source_data and Path(source_data).resolve() != data_path:
        raise ValueError(
            f"vision cache source {source_data} does not match dataset {data_path}"
        )
    source_sha256 = sha256_file(data_path)
    if vision_metadata.get("source_data_sha256") not in (None, source_sha256):
        raise ValueError("vision cache source_data_sha256 differs from dataset")
    feature_sha256 = sha256_file(feature_path)
    prefix_sha256 = sha256_file(prefix_path)
    if vision_metadata.get("backbone_features_sha256") not in (None, feature_sha256):
        raise ValueError("vision cache backbone feature hash mismatch")
    if vision_metadata.get("prefix_sha256") not in (None, prefix_sha256):
        raise ValueError("vision cache prefix hash mismatch")
    feature_maps = np.load(feature_path, mmap_mode="r")
    prefix = np.load(prefix_path, mmap_mode="r")
    with np.load(data_path, allow_pickle=False) as data:
        if args.image_key not in data.files:
            raise KeyError(
                f"dataset does not contain requested image key {args.image_key!r}"
            )
        q_nominal = data["q_nominal"].astype(np.float32)
        q_true = data["q_true"].astype(np.float32)
        keypoints_xyz = data["keypoints_xyz"].astype(np.float32)
        valid = (data["valid_true"] & data["valid_nominal"]).astype(bool)
        raw_obs = data["obs"]
        episode_id = np.asarray(data["episode_id"])
        frame_id = np.asarray(data["frame_id"])
        if q_nominal.ndim != 3 or q_nominal.shape[-1] != 2:
            raise ValueError(
                f"q_nominal must have shape [N, K, 2], got {q_nominal.shape}"
            )
        selected_images = _validate_image_array(
            data[args.image_key],
            image_key=args.image_key,
            expected_sample_count=int(q_nominal.shape[0]),
        )
        selected_image_shape = tuple(int(size) for size in selected_images.shape)
        selected_image_sha256 = _selected_image_sha256(selected_images)
    if (
        "selected_image_sha256" in vision_metadata
        and vision_metadata["selected_image_sha256"] != selected_image_sha256
    ):
        raise ValueError(
            "vision cache selected_image_sha256 differs from dataset image array"
        )
    if vision_metadata.get("sample_count") != selected_image_shape[0]:
        raise ValueError("vision cache sample count does not match selected image array")
    if (
        vision_metadata.get("image_height") != selected_image_shape[1]
        or vision_metadata.get("image_width") != selected_image_shape[2]
    ):
        raise ValueError("vision cache image shape does not match selected image array")
    device = torch.device(args.device)
    model, checkpoint_args = load_fcpg_vit_adapter(checkpoint_path, device)
    checkpoint_layers = tuple(
        int(layer) for layer in checkpoint_args.get("vision_selected_layers", (2, 5, 8, 11))
    )
    cache_layers = tuple(int(layer) for layer in vision_metadata["selected_layers"])
    if checkpoint_layers != cache_layers:
        raise ValueError(
            f"checkpoint layers {checkpoint_layers} do not match cache layers {cache_layers}"
        )
    obs = select_policy_observation(raw_obs, str(checkpoint_args.get("obs_mode", "robot")))
    sample_count, num_keypoints, _ = q_nominal.shape
    if feature_maps.shape[0] != sample_count or prefix.shape[0] != sample_count:
        raise ValueError("vision cache sample count does not match grounding dataset")
    image_size = (int(vision_metadata["image_width"]), int(vision_metadata["image_height"]))
    q_store = np.lib.format.open_memmap(
        q_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, num_keypoints, 2),
    )
    rho_store: np.memmap | None = None
    has_reliability: bool | None = None
    before_sum = 0.0
    after_sum = 0.0
    valid_count = 0
    rho_sum = 0.0
    with torch.inference_mode():
        for start in range(0, sample_count, args.batch_size):
            stop = min(start + args.batch_size, sample_count)
            batch_features = torch.from_numpy(
                np.array(feature_maps[start:stop], dtype=np.float16, copy=True)
            ).to(device, non_blocking=True)
            batch_prefix = torch.from_numpy(
                np.array(prefix[start:stop], dtype=np.float16, copy=True)
            ).to(device, non_blocking=True)
            batch_q = torch.from_numpy(q_nominal[start:stop]).to(device, non_blocking=True)
            batch_xyz = torch.from_numpy(keypoints_xyz[start:stop]).to(device, non_blocking=True)
            batch_obs = torch.from_numpy(obs[start:stop]).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model.forward_from_backbone_features(
                    batch_features,
                    batch_prefix,
                    batch_q,
                    batch_xyz,
                    batch_obs,
                    image_size=image_size,
                )
            corrected_tensor, reliability_tensor = _extract_grounding_output(output, batch_q)
            corrected = corrected_tensor.float().cpu().numpy()
            reliability = (
                reliability_tensor.float().cpu().numpy()
                if reliability_tensor is not None
                else None
            )
            batch_has_reliability = reliability is not None
            if has_reliability is None:
                has_reliability = batch_has_reliability
                if has_reliability:
                    rho_store = np.lib.format.open_memmap(
                        rho_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(sample_count, num_keypoints),
                    )
            elif has_reliability != batch_has_reliability:
                raise RuntimeError("adapter output reliability changed between cache batches")
            q_store[start:stop] = corrected
            if rho_store is not None:
                assert reliability is not None
                rho_store[start:stop] = reliability
            batch_valid = valid[start:stop]
            before = np.linalg.norm(q_nominal[start:stop] - q_true[start:stop], axis=-1)
            after = np.linalg.norm(corrected - q_true[start:stop], axis=-1)
            before_sum += float(before[batch_valid].sum())
            after_sum += float(after[batch_valid].sum())
            if reliability is not None:
                rho_sum += float(reliability[batch_valid].sum())
            valid_count += int(batch_valid.sum())
            if stop == sample_count or stop % (args.batch_size * 10) == 0:
                print(f"cached {stop}/{sample_count}", flush=True)
    q_store.flush()
    if rho_store is not None:
        rho_store.flush()
    has_reliability = bool(has_reliability)
    metrics = {
        "e_before": before_sum / max(valid_count, 1),
        "e_after": after_sum / max(valid_count, 1),
        "ratio": after_sum / max(before_sum, 1e-8),
    }
    if has_reliability:
        metrics["reliability_mean"] = rho_sum / max(valid_count, 1)
    metadata = {
        "version": 3,
        "schema_version": 3,
        "has_reliability": has_reliability,
        "corrector_arch": str(checkpoint_args.get("corrector_arch", "legacy")),
        "checkpoint_schema_version": int(checkpoint_args.get("schema_version", 1)),
        "obs_mode": str(checkpoint_args.get("obs_mode", "robot")),
        "lewm_upstream_commit": checkpoint_args.get("lewm_upstream_commit"),
        "source_data": str(data_path),
        "source_data_sha256": source_sha256,
        "image_key": args.image_key,
        "selected_image_sha256": selected_image_sha256,
        "view_role": view_role,
        "deployment_cache_eligible": deployment_cache_eligible,
        "vision_image_lineage_verified": not legacy_default_rgb,
        "vision_cache": str(vision_cache),
        "vision_metadata_sha256": sha256_file(vision_metadata_path),
        "backbone_features_sha256": feature_sha256,
        "prefix_sha256": prefix_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "sample_count": int(sample_count),
        "num_keypoints": int(num_keypoints),
        "frame_identity_sha256": frame_identity_sha256(episode_id, frame_id),
        "q_corrected_sha256": sha256_file(q_path),
        "q_true_usage": "offline_metrics_only",
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"q_corrected: {q_path}")
    if has_reliability:
        print(f"reliability: {rho_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
