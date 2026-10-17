#!/usr/bin/env python3
"""Compare frozen FCPG and LeWM local features on paired camera views.

This is a zero-training diagnostic.  It keeps the two encoders untouched and
measures whether a corrected image-space anchor gives a more view-stable local
descriptor than the current LeWM ViT-Tiny input.  The paired images are
rendered from the same physical snapshot, so descriptor gaps are not mixed
with robot motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_action_branch_collection import load_frozen_fcpg_anchor_encoder  # noqa: E402
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    FrameRecord,
    corrected_points,
    grounded_valid_mask,
)
from fcpg.modules import sample_feature_map_at_pixels  # noqa: E402
from fcpg.observations import select_policy_observation  # noqa: E402


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
        rows.append(
            FrameRecord(
                rgb=np.transpose(images[local].astype(np.float32) / 255.0, (2, 0, 1)),
                proprio=select_policy_observation(data["obs"][index], "robot").astype(
                    np.float32
                ),
                keypoints_xyz=np.asarray(data["keypoints_xyz"][index], dtype=np.float32),
                q_nominal=q_nominal,
                valid_nominal=np.asarray(data["valid_nominal"][index], dtype=bool),
                q_true=(
                    q_nominal
                    if clean
                    else np.asarray(data["q_true"][index], dtype=np.float32)
                ),
                valid_true=(
                    np.asarray(data["valid_nominal"][index], dtype=bool)
                    if clean
                    else np.asarray(data["valid_true"][index], dtype=bool)
                ),
            )
        )
    return rows


def _stack_rgb(records: list[FrameRecord], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack([record.rgb for record in records])).to(device)


def _sample(
    feature_map: torch.Tensor,
    points: np.ndarray,
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    sampled = sample_feature_map_at_pixels(
        feature_map,
        torch.from_numpy(np.asarray(points, dtype=np.float32)).to(feature_map.device),
        image_size=image_size,
        padding_mode="border",
    )
    return sampled.float().cpu().numpy()


def _cosine_gap(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-8)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8)
    return 1.0 - np.sum(left_norm * right_norm, axis=-1)


def _normalized_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-8)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8)
    return np.linalg.norm(left_norm - right_norm, axis=-1)


def _summary(values: np.ndarray, valid: np.ndarray) -> dict[str, float | int | None]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(selected.size),
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "p90": float(np.percentile(selected, 90)),
    }


def _relative_gain(nominal: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    n = nominal.get("mean")
    c = candidate.get("mean")
    if n is None or c is None or float(n) <= 1e-12:
        return None
    return float((float(n) - float(c)) / float(n))


def _apply_frame_mask(frame_mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Broadcast a frame-level selection over the anchor dimension safely."""

    frame_mask = np.asarray(frame_mask, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or frame_mask.shape != (valid.shape[0],):
        raise ValueError("frame mask and valid mask must have shapes [N] and [N,K]")
    return frame_mask[:, None] & valid


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--fcpg-checkpoint", required=True)
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional smoke limit; zero evaluates every frame.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.max_frames < 0:
        raise ValueError("max-frames must be non-negative")
    data_path = Path(args.data).resolve()
    fcpg_path = Path(args.fcpg_checkpoint).resolve()
    anchor_path = Path(args.anchor_checkpoint).resolve()
    device = torch.device(args.device)
    encoder = load_frozen_fcpg_anchor_encoder(
        anchor_path,
        fcpg_path,
        device=device,
    )
    corrector = encoder.corrector
    tiny_encoder = encoder.model.frame_encoder.vision_encoder
    fcpg_encoder = corrector.vision
    corrector.eval()
    tiny_encoder.eval()
    fcpg_encoder.eval()

    rows: dict[str, list[np.ndarray]] = {
        "task": [],
        "phase": [],
        "valid": [],
        "tiny_nominal": [],
        "tiny_corrected": [],
        "tiny_oracle": [],
        "tiny_clean": [],
        "fcpg_nominal": [],
        "fcpg_corrected": [],
        "fcpg_oracle": [],
        "fcpg_clean": [],
    }

    with np.load(data_path, allow_pickle=False) as data:
        if "paired_clean_rgb" not in data.files:
            raise ValueError("dataset must contain paired_clean_rgb")
        count = int(data["rgb"].shape[0])
        if args.max_frames:
            count = min(count, int(args.max_frames))
        width = int(np.asarray(data["image_width"]).item())
        height = int(np.asarray(data["image_height"]).item())
        for start in range(0, count, args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, count))
            dynamic = _records(data, indices, image_key="rgb", clean=False)
            clean = _records(data, indices, image_key="paired_clean_rgb", clean=True)
            q_nominal = np.stack([record.q_nominal for record in dynamic]).astype(np.float32)
            q_oracle = np.stack([record.q_true for record in dynamic]).astype(np.float32)
            q_dynamic_corrected = corrected_points(dynamic, encoder)
            q_clean_corrected = corrected_points(clean, encoder)
            valid_nominal = np.stack([record.valid_nominal for record in dynamic])
            valid_true = np.stack([record.valid_true for record in dynamic])
            valid = (
                valid_nominal
                & valid_true
                & grounded_valid_mask(
                    valid_nominal,
                    q_dynamic_corrected,
                    width=width,
                    height=height,
                )
                & grounded_valid_mask(
                    valid_nominal,
                    q_clean_corrected,
                    width=width,
                    height=height,
                )
                & np.isfinite(q_oracle).all(axis=-1)
                & (q_oracle[..., 0] >= 0.0)
                & (q_oracle[..., 0] <= width - 1)
                & (q_oracle[..., 1] >= 0.0)
                & (q_oracle[..., 1] <= height - 1)
            )

            dynamic_rgb = _stack_rgb(dynamic, device)
            clean_rgb = _stack_rgb(clean, device)
            tiny_dynamic = tiny_encoder(dynamic_rgb).feature_map
            tiny_clean = tiny_encoder(clean_rgb).feature_map
            fcpg_dynamic = fcpg_encoder(dynamic_rgb).feature_map
            fcpg_clean = fcpg_encoder(clean_rgb).feature_map

            for name, feature_dynamic, feature_clean in (
                ("tiny", tiny_dynamic, tiny_clean),
                ("fcpg", fcpg_dynamic, fcpg_clean),
            ):
                rows[f"{name}_nominal"].append(
                    _sample(feature_dynamic, q_nominal, image_size=(width, height))
                )
                rows[f"{name}_corrected"].append(
                    _sample(feature_dynamic, q_dynamic_corrected, image_size=(width, height))
                )
                rows[f"{name}_oracle"].append(
                    _sample(feature_dynamic, q_oracle, image_size=(width, height))
                )
                rows[f"{name}_clean"].append(
                    _sample(feature_clean, q_nominal, image_size=(width, height))
                )

            rows["task"].append(np.asarray(data["task_name"][indices]).astype(str))
            phase = np.asarray(data["actual_camera_trajectory_phase"][indices], dtype=np.float32)
            rows["phase"].append(
                np.where(phase <= 1e-6, "pre", np.where(phase >= 1.0 - 1e-6, "post", "moving"))
            )
            rows["valid"].append(valid)

        joined = {name: np.concatenate(values, axis=0) for name, values in rows.items()}

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        mask = _apply_frame_mask(mask, joined["valid"])
        result: dict[str, Any] = {"frames": int(np.asarray(mask).sum())}
        for name in ("tiny", "fcpg"):
            nominal = _summary(
                _normalized_l2(joined[f"{name}_nominal"], joined[f"{name}_clean"]), mask
            )
            corrected = _summary(
                _normalized_l2(joined[f"{name}_corrected"], joined[f"{name}_clean"]), mask
            )
            oracle = _summary(
                _normalized_l2(joined[f"{name}_oracle"], joined[f"{name}_clean"]), mask
            )
            nominal_cos = _summary(
                _cosine_gap(joined[f"{name}_nominal"], joined[f"{name}_clean"]), mask
            )
            corrected_cos = _summary(
                _cosine_gap(joined[f"{name}_corrected"], joined[f"{name}_clean"]), mask
            )
            oracle_cos = _summary(
                _cosine_gap(joined[f"{name}_oracle"], joined[f"{name}_clean"]), mask
            )
            result[name] = {
                "normalized_l2": {
                    "nominal": nominal,
                    "corrected": corrected,
                    "oracle": oracle,
                    "corrected_relative_gain": _relative_gain(nominal, corrected),
                    "oracle_relative_gain": _relative_gain(nominal, oracle),
                },
                "cosine_gap": {
                    "nominal": nominal_cos,
                    "corrected": corrected_cos,
                    "oracle": oracle_cos,
                    "corrected_relative_gain": _relative_gain(nominal_cos, corrected_cos),
                    "oracle_relative_gain": _relative_gain(nominal_cos, oracle_cos),
                },
            }
        return result

    payload = {
        "protocol": "frozen_local_feature_pair_probe_v1",
        "data": str(data_path),
        "data_sha256": _sha256(data_path),
        "fcpg_checkpoint": str(fcpg_path),
        "fcpg_checkpoint_sha256": _sha256(fcpg_path),
        "anchor_checkpoint": str(anchor_path),
        "anchor_checkpoint_sha256": _sha256(anchor_path),
        "frames": int(joined["valid"].shape[0]),
        "valid_anchor_count": int(joined["valid"].sum()),
        "metrics": {
            "all": metrics(np.ones(joined["valid"].shape[0], dtype=bool)),
            "by_task": {
                task: metrics(joined["task"] == task)
                for task in sorted(set(joined["task"].tolist()))
            },
            "by_phase": {
                phase: metrics(joined["phase"] == phase)
                for phase in ("pre", "moving", "post")
                if bool(np.any(joined["phase"] == phase))
            },
        },
        "interpretation": {
            "nominal": "dynamic RGB sampled at nominal q, clean RGB sampled at clean nominal q",
            "corrected": "dynamic RGB sampled at frozen FCPG q_corrected, clean RGB at clean nominal q",
            "oracle": "dynamic RGB sampled at q_true, clean RGB at clean nominal q",
            "feature_gate": "diagnostic only; no training or LeWM input replacement is authorized by this script",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
