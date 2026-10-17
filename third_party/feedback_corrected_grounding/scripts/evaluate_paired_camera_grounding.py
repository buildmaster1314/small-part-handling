#!/usr/bin/env python3
"""Evaluate FCPG and anchor-state invariance on paired camera renders."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_action_branch_collection import (  # noqa: E402
    load_frozen_fcpg_anchor_encoder,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    FrameRecord,
    corrected_points,
    grounded_valid_mask,
)
from fcpg.observations import select_policy_observation  # noqa: E402
from fcpg.paired_camera_evaluation import (  # noqa: E402
    summarize_paired_camera_arrays,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _records(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    *,
    image_key: str,
    clean: bool,
) -> list[FrameRecord]:
    images = data[image_key][indices]
    rows: list[FrameRecord] = []
    for local, index in enumerate(indices.tolist()):
        q_nominal = np.asarray(data["q_nominal"][index], dtype=np.float32)
        valid_nominal = np.asarray(data["valid_nominal"][index], dtype=bool)
        rows.append(
            FrameRecord(
                rgb=np.transpose(images[local].astype(np.float32) / 255.0, (2, 0, 1)),
                proprio=select_policy_observation(data["obs"][index], "robot").astype(
                    np.float32
                ),
                keypoints_xyz=np.asarray(
                    data["keypoints_xyz"][index], dtype=np.float32
                ),
                q_nominal=q_nominal,
                valid_nominal=valid_nominal,
                q_true=(
                    q_nominal
                    if clean
                    else np.asarray(data["q_true"][index], dtype=np.float32)
                ),
                valid_true=(
                    valid_nominal
                    if clean
                    else np.asarray(data["valid_true"][index], dtype=bool)
                ),
            )
        )
    return rows


@torch.inference_mode()
def _encode_at_points(
    records: list[FrameRecord],
    points: np.ndarray,
    *,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    pixels = np.stack([record.rgb for record in records]).astype(np.float32)
    proprio = np.stack([record.proprio for record in records]).astype(np.float32)
    q_nominal = np.stack([record.q_nominal for record in records]).astype(np.float32)
    xyz = np.stack([record.keypoints_xyz for record in records]).astype(np.float32)
    valid_nominal = np.stack([record.valid_nominal for record in records]).astype(bool)
    grounded = np.asarray(points, dtype=np.float32)
    valid = grounded_valid_mask(
        valid_nominal,
        grounded,
        width=int(pixels.shape[-1]),
        height=int(pixels.shape[-2]),
    )
    batch = {
        "pixels": torch.from_numpy(pixels).unsqueeze(0).to(device),
        "proprio": torch.from_numpy(proprio).unsqueeze(0).to(device),
        "q_nominal": torch.from_numpy(q_nominal).unsqueeze(0).to(device),
        "q_corrected": torch.from_numpy(grounded).unsqueeze(0).to(device),
        "keypoints_xyz": torch.from_numpy(xyz).unsqueeze(0).to(device),
        "valid": torch.from_numpy(valid).unsqueeze(0).to(device),
    }
    _, embedding = model.encode_frames(batch)
    result = embedding[0].float().cpu().numpy()
    if result.shape[:2] != grounded.shape[:2] or not np.isfinite(result).all():
        raise ValueError("anchor embedding has invalid shape or values")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--fcpg-checkpoint", required=True)
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cache-output",
        default="",
        help="Optional compressed anchor-state cache for downstream diagnostics.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    data_path = Path(args.data).resolve()
    fcpg_path = Path(args.fcpg_checkpoint).resolve()
    anchor_path = Path(args.anchor_checkpoint).resolve()
    device = torch.device(args.device)
    encoder = load_frozen_fcpg_anchor_encoder(
        anchor_path,
        fcpg_path,
        device=device,
    )
    frozen = SimpleNamespace(corrector=encoder.corrector, device=device)

    outputs: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "q_dynamic_corrected",
            "q_clean_corrected",
            "latent_dynamic_nominal",
            "latent_clean_nominal",
            "latent_dynamic_corrected",
            "latent_dynamic_oracle",
            "latent_clean_corrected",
        )
    }
    with np.load(data_path, allow_pickle=False) as data:
        if "paired_clean_rgb" not in data.files:
            raise ValueError("dataset must contain paired_clean_rgb")
        count = int(data["rgb"].shape[0])
        if data["paired_clean_rgb"].shape != data["rgb"].shape:
            raise ValueError("dynamic and clean images must have identical shapes")
        for start in range(0, count, args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, count))
            dynamic = _records(data, indices, image_key="rgb", clean=False)
            clean = _records(data, indices, image_key="paired_clean_rgb", clean=True)
            dynamic_corrected = corrected_points(dynamic, frozen)
            clean_corrected = corrected_points(clean, frozen)
            outputs["q_dynamic_corrected"].append(dynamic_corrected)
            outputs["q_clean_corrected"].append(clean_corrected)
            dynamic_nominal = np.stack([record.q_nominal for record in dynamic])
            clean_nominal = np.stack([record.q_nominal for record in clean])
            outputs["latent_dynamic_nominal"].append(
                _encode_at_points(
                    dynamic,
                    dynamic_nominal,
                    model=encoder.model,
                    device=device,
                )
            )
            outputs["latent_clean_nominal"].append(
                _encode_at_points(
                    clean,
                    clean_nominal,
                    model=encoder.model,
                    device=device,
                )
            )
            outputs["latent_dynamic_corrected"].append(
                _encode_at_points(
                    dynamic,
                    dynamic_corrected,
                    model=encoder.model,
                    device=device,
                )
            )
            dynamic_oracle = np.stack([record.q_true for record in dynamic])
            outputs["latent_dynamic_oracle"].append(
                _encode_at_points(
                    dynamic,
                    dynamic_oracle,
                    model=encoder.model,
                    device=device,
                )
            )
            outputs["latent_clean_corrected"].append(
                _encode_at_points(
                    clean,
                    clean_corrected,
                    model=encoder.model,
                    device=device,
                )
            )

        joined = {name: np.concatenate(parts) for name, parts in outputs.items()}
        width = int(np.asarray(data["image_width"]).item())
        height = int(np.asarray(data["image_height"]).item())
        metrics = summarize_paired_camera_arrays(
            q_nominal=data["q_nominal"],
            q_true=data["q_true"],
            valid_nominal=data["valid_nominal"],
            valid_true=data["valid_true"],
            task_names=data["task_name"],
            trajectory_phase=data["actual_camera_trajectory_phase"],
            intervention_onset=data["camera_intervention_onset"],
            image_size=(width, height),
            **joined,
        )
        payload = {
            "protocol": "fcpg_paired_camera_grounding_probe_v1",
            "data": str(data_path),
            "data_sha256": _sha256(data_path),
            "fcpg_checkpoint": str(fcpg_path),
            "fcpg_checkpoint_sha256": _sha256(fcpg_path),
            "anchor_checkpoint": str(anchor_path),
            "anchor_checkpoint_sha256": _sha256(anchor_path),
            "frames": count,
            "trajectory_mode": str(
                np.asarray(data["actual_camera_trajectory_mode"]).item()
            ),
            "metrics": metrics,
        }
        if args.cache_output:
            cache_path = Path(args.cache_output)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                schema_version=np.asarray(1, dtype=np.int32),
                protocol=np.asarray("fcpg_paired_camera_anchor_cache_v1"),
                source_data=np.asarray(str(data_path)),
                source_data_sha256=np.asarray(payload["data_sha256"]),
                fcpg_checkpoint_sha256=np.asarray(payload["fcpg_checkpoint_sha256"]),
                anchor_checkpoint_sha256=np.asarray(
                    payload["anchor_checkpoint_sha256"]
                ),
                trajectory_mode=np.asarray(payload["trajectory_mode"]),
                image_width=np.asarray(width, dtype=np.int32),
                image_height=np.asarray(height, dtype=np.int32),
                task_name=np.asarray(data["task_name"]),
                episode_id=np.asarray(data["episode_id"], dtype=np.int32),
                frame_id=np.asarray(data["frame_id"], dtype=np.int32),
                q_nominal=np.asarray(data["q_nominal"], dtype=np.float32),
                q_true=np.asarray(data["q_true"], dtype=np.float32),
                valid_nominal=np.asarray(data["valid_nominal"], dtype=bool),
                valid_true=np.asarray(data["valid_true"], dtype=bool),
                keypoints_xyz=np.asarray(data["keypoints_xyz"], dtype=np.float32),
                true_extrinsic=np.asarray(data["true_extrinsic"], dtype=np.float32),
                base_extrinsic=np.asarray(data["base_extrinsic"], dtype=np.float32),
                trajectory_phase=np.asarray(
                    data["actual_camera_trajectory_phase"], dtype=np.float32
                ),
                camera_intervention_onset=np.asarray(
                    data["camera_intervention_onset"], dtype=bool
                ),
                q_dynamic_corrected=joined["q_dynamic_corrected"].astype(
                    np.float32, copy=False
                ),
                q_clean_corrected=joined["q_clean_corrected"].astype(
                    np.float32, copy=False
                ),
                latent_dynamic_corrected=joined[
                    "latent_dynamic_corrected"
                ].astype(np.float32, copy=False),
                latent_clean_corrected=joined["latent_clean_corrected"].astype(
                    np.float32, copy=False
                ),
                latent_clean_nominal=joined["latent_clean_nominal"].astype(
                    np.float32, copy=False
                ),
                latent_dynamic_oracle=joined["latent_dynamic_oracle"].astype(
                    np.float32, copy=False
                ),
            )
            payload["cache_output"] = str(cache_path.resolve())
            payload["cache_output_sha256"] = _sha256(cache_path)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
