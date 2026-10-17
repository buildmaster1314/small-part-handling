#!/usr/bin/env python3
"""Cache absolute-keypoint predictions in the existing correction-cache format."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.absolute_keypoint_baseline import (  # noqa: E402
    BASELINE_VARIANT,
    load_absolute_keypoint_baseline,
    validate_deployment_vision_cache,
)
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.lewm_sequence_data import frame_identity_sha256  # noqa: E402
from fcpg.observations import select_policy_observation  # noqa: E402


class AbsoluteKeypointInferenceDataset(Dataset):
    """Deployment-only inputs; target arrays are neither required nor loaded."""

    REQUIRED_FIELDS = {
        "obs",
        "q_nominal",
        "keypoints_xyz",
        "episode_id",
        "frame_id",
    }

    def __init__(
        self,
        data_path: str | Path,
        cache_info: Mapping[str, object],
        *,
        expected_obs_dim: int,
        expected_num_keypoints: int,
    ) -> None:
        self.path = Path(data_path).resolve()
        with np.load(self.path, allow_pickle=False) as raw:
            missing = sorted(self.REQUIRED_FIELDS.difference(raw.files))
            if missing:
                raise ValueError(f"inference dataset missing deployment fields: {missing}")
            self.robot_obs = select_policy_observation(raw["obs"], "robot")
            self.q_nominal = np.asarray(raw["q_nominal"], dtype=np.float32)
            self.keypoints_xyz_fk = np.asarray(
                raw["keypoints_xyz"], dtype=np.float32
            )
            self.episode_id = np.asarray(raw["episode_id"])
            self.frame_id = np.asarray(raw["frame_id"])
        sample_count = int(self.robot_obs.shape[0])
        if self.robot_obs.shape != (sample_count, expected_obs_dim):
            raise ValueError("robot obs shape does not match checkpoint")
        if self.q_nominal.shape != (sample_count, expected_num_keypoints, 2):
            raise ValueError("q_nominal shape does not match checkpoint")
        if self.keypoints_xyz_fk.shape != (
            sample_count,
            expected_num_keypoints,
            3,
        ):
            raise ValueError("FK XYZ shape does not match checkpoint")
        if self.episode_id.shape[0] != sample_count or self.frame_id.shape[0] != sample_count:
            raise ValueError("frame identity count does not match deployment inputs")
        for name, value in (
            ("robot_obs", self.robot_obs),
            ("q_nominal", self.q_nominal),
            ("keypoints_xyz", self.keypoints_xyz_fk),
        ):
            if not bool(np.isfinite(value).all()):
                raise ValueError(f"deployment input {name} contains non-finite values")
        self.features = np.load(
            str(cache_info["feature_path"]), mmap_mode="r", allow_pickle=False
        )
        self.prefix = np.load(
            str(cache_info["prefix_path"]), mmap_mode="r", allow_pickle=False
        )
        if self.features.shape[0] != sample_count or self.prefix.shape[0] != sample_count:
            raise ValueError("vision cache count does not match inference dataset")
        self.image_size = tuple(int(v) for v in cache_info["image_size"])
        self.sample_count = sample_count
        self.num_keypoints = expected_num_keypoints

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "vision_backbone_features": torch.from_numpy(
                np.array(self.features[index], dtype=np.float16, copy=True)
            ),
            "vision_prefix": torch.from_numpy(
                np.array(self.prefix[index], dtype=np.float16, copy=True)
            ),
            "q_nominal": torch.from_numpy(self.q_nominal[index]),
            "keypoints_xyz_fk": torch.from_numpy(self.keypoints_xyz_fk[index]),
            "robot_obs": torch.from_numpy(self.robot_obs[index]),
            "image_size": torch.tensor(self.image_size, dtype=torch.long),
        }


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--vision-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers non-negative")

    data_path = Path(args.data).expanduser().resolve()
    vision_cache = Path(args.vision_cache).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    q_path = output_dir / "q_corrected.npy"
    metadata_path = output_dir / "metadata.json"
    reliability_path = output_dir / "reliability.npy"
    for path in (q_path, metadata_path, reliability_path):
        if path.exists():
            parser.error(f"cache output already exists: {path}")

    device = choose_device(args.device)
    model, checkpoint = load_absolute_keypoint_baseline(checkpoint_path, device)
    model_config = checkpoint["model_config"]
    provenance = checkpoint["training_provenance"]
    assert isinstance(model_config, Mapping)
    assert isinstance(provenance, Mapping)
    reference = provenance["reference_fcpg"]
    if not isinstance(reference, Mapping) or not isinstance(reference.get("checkpoint"), str):
        raise ValueError("checkpoint lacks reference FCPG vision provenance")
    cache_info = validate_deployment_vision_cache(
        data_path,
        vision_cache,
        model_config,
        expected_weight_source=str(reference["checkpoint"]),
    )
    dataset = AbsoluteKeypointInferenceDataset(
        data_path,
        cache_info,
        expected_obs_dim=model.obs_dim,
        expected_num_keypoints=model.num_keypoints,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    q_store = np.lib.format.open_memmap(
        q_path,
        mode="w+",
        dtype=np.float32,
        shape=(dataset.sample_count, dataset.num_keypoints, 2),
    )
    start_index = 0
    elapsed_seconds = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {
                name: value.to(device, non_blocking=device.type == "cuda")
                for name, value in batch.items()
            }
            image_sizes = batch["image_size"]
            if not bool(torch.all(image_sizes == image_sizes[0])):
                raise ValueError("cache batch contains mixed image sizes")
            image_size = tuple(int(v) for v in image_sizes[0].tolist())
            synchronize(device)
            start = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model.forward_from_backbone_features(
                    batch["vision_backbone_features"],
                    batch["vision_prefix"],
                    batch["q_nominal"],
                    batch["keypoints_xyz_fk"],
                    batch["robot_obs"],
                    image_size=image_size,
                )
            synchronize(device)
            elapsed_seconds += time.perf_counter() - start
            predicted = output.q_absolute_px.float().cpu().numpy()
            stop_index = start_index + int(predicted.shape[0])
            q_store[start_index:stop_index] = predicted
            start_index = stop_index
            if start_index == dataset.sample_count or start_index % (args.batch_size * 10) == 0:
                print(f"cached {start_index}/{dataset.sample_count}", flush=True)
    if start_index != dataset.sample_count:
        raise RuntimeError("correction cache stopped before all frames were written")
    q_store.flush()
    metadata = {
        "version": 3,
        "schema_version": 3,
        "has_reliability": False,
        "corrector_arch": BASELINE_VARIANT,
        "checkpoint_schema_version": int(checkpoint["schema_version"]),
        "obs_mode": "robot",
        "source_data": str(data_path),
        "source_data_sha256": str(cache_info["source_data_sha256"]),
        "image_key": "rgb",
        "selected_image_sha256": str(cache_info["selected_image_sha256"]),
        "view_role": "deployment_observation",
        "deployment_cache_eligible": True,
        "vision_image_lineage_verified": True,
        "vision_cache": str(vision_cache),
        "vision_metadata_sha256": sha256_file(cache_info["metadata_path"]),
        "backbone_features_sha256": str(cache_info["backbone_features_sha256"]),
        "prefix_sha256": str(cache_info["prefix_sha256"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "sample_count": dataset.sample_count,
        "num_keypoints": dataset.num_keypoints,
        "frame_identity_sha256": frame_identity_sha256(
            dataset.episode_id, dataset.frame_id
        ),
        "q_corrected_sha256": sha256_file(q_path),
        "q_true_usage": "none",
        "coordinate_parameterization": str(model.coordinate_parameterization),
        "parameter_report": model.parameter_report(),
        "sampling_report": model.sampling_report(),
        "latency": {
            "scope": "adapter_forward_from_cached_frozen_vit_features",
            "total_seconds": elapsed_seconds,
            "milliseconds_per_frame": 1000.0 * elapsed_seconds / dataset.sample_count,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "sample_count": dataset.sample_count,
        "milliseconds_per_frame": metadata["latency"]["milliseconds_per_frame"],
        "q_true_usage": "none",
    }, indent=2))
    print(f"q_corrected: {q_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
