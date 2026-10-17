#!/usr/bin/env python3
"""Build real-appearance SO-101 FCPG data with exact synthetic pixel drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fcpg.so101_kinematics import SO101MuJoCoKinematics
from fcpg.so101_real_image_drift import (
    SO101ImageAffineDrift,
    points_in_bounds,
    sample_bounded_drift,
    transform_points,
    warp_rgb,
)
from fcpg.so101_synthetic_dataset import ARRAY_NAMES, DATASET_FORMAT
from scripts.visualize_so101_lewm import DEFAULT_SO101_MODEL


VIEWS = ("wrist", "column")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrist-cache", type=Path, required=True)
    parser.add_argument("--column-cache", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_SO101_MODEL)
    parser.add_argument("--validation-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=6121)
    parser.add_argument("--max-rotation", type=float, default=4.0)
    parser.add_argument("--max-translation", type=float, default=8.0)
    parser.add_argument("--min-mean-residual", type=float, default=3.0)
    parser.add_argument("--max-residual", type=float, default=18.0)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"SO-101 model not found: {args.model}")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def load_cache(path: Path) -> dict[str, object]:
    root = path.resolve()
    return {
        "root": root,
        "manifest": json.loads((root / "manifest.json").read_text()),
        "frames": np.load(root / "frames.npy", mmap_mode="r"),
        "states": np.load(root / "states.npy", mmap_mode="r"),
        "episodes": np.load(root / "episodes.npy", mmap_mode="r"),
        "frame_indices": np.load(root / "frame_indices.npy", mmap_mode="r"),
        "tasks": np.load(root / "tasks.npy", mmap_mode="r"),
    }


def allocate_arrays(root: Path, samples: int, height: int, width: int):
    specifications = {
        "rgb": ((samples, height, width, 3), np.uint8),
        "states": ((samples, 6), np.float32),
        "anchors_xyz": ((samples, 3, 3), np.float32),
        "q_nominal": ((samples, 3, 2), np.float32),
        "q_true": ((samples, 3, 2), np.float32),
        "valid": ((samples, 3), np.bool_),
        "drift": ((samples, 3), np.float32),
        "camera_index": ((samples,), np.uint8),
        "camera_context": ((samples, 5), np.float32),
        "drift_variant": ((samples,), np.uint8),
        "split": ((samples,), np.uint8),
        "source_index": ((samples,), np.int16),
        "source_row": ((samples,), np.int64),
        "episode": ((samples,), np.int32),
        "task": ((samples,), np.int32),
    }
    if set(specifications) != set(ARRAY_NAMES):
        raise AssertionError("FCPG dataset array contract drifted")
    return {
        name: np.lib.format.open_memmap(
            root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in specifications.items()
    }


def main() -> None:
    args = parse_args()
    wrist = load_cache(args.wrist_cache)
    column = load_cache(args.column_cache)
    for key in ("states", "episodes", "frame_indices", "tasks"):
        if not np.array_equal(wrist[key], column[key]):
            raise ValueError(f"paired caches disagree on {key}")
    if wrist["manifest"]["dataset_id"] != column["manifest"]["dataset_id"]:
        raise ValueError("paired caches have different dataset IDs")
    frames_by_view = (wrist["frames"], column["frames"])
    frame_shape = tuple(np.asarray(frames_by_view[0]).shape[1:])
    if frame_shape != tuple(np.asarray(frames_by_view[1]).shape[1:]):
        raise ValueError("paired camera frames have different shapes")
    height, width, channels = frame_shape
    if channels != 3:
        raise ValueError("camera frames must be RGB")
    rows = len(wrist["states"])
    episodes = np.asarray(wrist["episodes"])
    if args.validation_episode not in set(map(int, np.unique(episodes))):
        raise ValueError("validation episode is absent from the cache")

    calibration = args.calibration.resolve()
    q_by_view = tuple(
        np.load(calibration / f"q_nominal_px_{view}.npy", mmap_mode="r")
        for view in VIEWS
    )
    source_valid_by_view = tuple(
        np.load(calibration / f"valid_{view}.npy", mmap_mode="r")
        for view in VIEWS
    )
    for view, q_values, valid_values in zip(
        VIEWS, q_by_view, source_valid_by_view, strict=True
    ):
        if q_values.shape != (rows, 3, 2) or valid_values.shape != (rows, 3):
            raise ValueError(f"calibration arrays have invalid shape for {view}")

    output = args.output.resolve()
    output.mkdir(parents=True)
    samples = rows * len(VIEWS) * 2
    arrays = allocate_arrays(output, samples, height, width)
    kinematics = SO101MuJoCoKinematics(args.model)
    states = np.asarray(wrist["states"], dtype=np.float32)
    anchors_xyz = kinematics.forward_anchors(states).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    identity = SO101ImageAffineDrift()
    index = 0
    for row in range(rows):
        drift = sample_bounded_drift(
            rng,
            tuple(np.asarray(q[row]) for q in q_by_view),
            tuple(np.asarray(valid[row]) for valid in source_valid_by_view),
            width=width,
            height=height,
            max_rotation_degrees=args.max_rotation,
            max_translation_px=args.max_translation,
            min_mean_residual_px=args.min_mean_residual,
            max_residual_px=args.max_residual,
        )
        for drift_variant, perturbation in enumerate((identity, drift)):
            matrix = perturbation.matrix(width, height)
            for camera_index, view in enumerate(VIEWS):
                source_rgb = np.asarray(frames_by_view[camera_index][row])
                nominal = np.asarray(q_by_view[camera_index][row], dtype=np.float32)
                target = transform_points(nominal, matrix)
                valid = (
                    np.asarray(source_valid_by_view[camera_index][row], dtype=bool)
                    & points_in_bounds(nominal, width, height)
                    & points_in_bounds(target, width, height)
                )
                arrays["rgb"][index] = (
                    source_rgb if drift_variant == 0 else warp_rgb(source_rgb, matrix)
                )
                arrays["states"][index] = states[row]
                arrays["anchors_xyz"][index] = anchors_xyz[row]
                arrays["q_nominal"][index] = nominal
                arrays["q_true"][index] = target
                arrays["valid"][index] = valid
                arrays["drift"][index] = perturbation.as_array()
                arrays["camera_index"][index] = camera_index
                context = np.zeros(5, dtype=np.float32)
                context[camera_index] = 1.0
                arrays["camera_context"][index] = context
                arrays["drift_variant"][index] = drift_variant
                arrays["split"][index] = int(episodes[row] == args.validation_episode)
                arrays["source_index"][index] = 0
                arrays["source_row"][index] = row
                arrays["episode"][index] = int(episodes[row])
                arrays["task"][index] = int(wrist["tasks"][row])
                index += 1
        if (row + 1) % 100 == 0 or row + 1 == rows:
            print(f"warped physical frames {row + 1}/{rows}", flush=True)
    if index != samples:
        raise AssertionError("dataset record count mismatch")
    for values in arrays.values():
        values.flush()

    split = np.asarray(arrays["split"])
    valid = np.asarray(arrays["valid"])
    drift_variant = np.asarray(arrays["drift_variant"])
    residual = np.linalg.norm(
        np.asarray(arrays["q_true"]) - np.asarray(arrays["q_nominal"]), axis=-1
    )
    nonclean_support = (drift_variant[:, None] == 1) & valid
    camera_views = [
        {"name": "wrist", "identity": [1.0, 0.0, 0.0, 0.0, 0.0]},
        {"name": "column", "identity": [0.0, 1.0, 0.0, 0.0, 0.0]},
    ]
    manifest = {
        "format": DATASET_FORMAT,
        "samples": samples,
        "train_samples": int(np.sum(split == 0)),
        "val_samples": int(np.sum(split == 1)),
        "train_physical_states": int(np.sum(episodes != args.validation_episode)),
        "val_physical_states": int(np.sum(episodes == args.validation_episode)),
        "width": width,
        "height": height,
        "seed": args.seed,
        "dataset_id": wrist["manifest"]["dataset_id"],
        "sources": [str(wrist["root"]), str(column["root"])],
        "calibration": str(calibration),
        "model": str(args.model.resolve()),
        "validation_episode": args.validation_episode,
        "task_names": wrist["manifest"].get("task_names", {}),
        "camera": {
            "views": camera_views,
            "context": "fixed two-view identity; synthetic drift parameters are hidden",
            "drifts_per_view": 2,
            "perturbation_type": "image_affine_rotation_translation",
            "max_rotation_degrees": args.max_rotation,
            "max_translation_px": args.max_translation,
            "min_mean_residual_px": args.min_mean_residual,
            "max_residual_px": args.max_residual,
            "pairing": "the same affine perturbation is applied to both views at each row",
        },
        "valid_anchor_fraction": float(valid.mean()),
        "nonclean_mean_epe_px": float(residual[nonclean_support].mean()),
        "nonclean_p95_epe_px": float(np.quantile(residual[nonclean_support], 0.95)),
        "scope": (
            "real dual-camera appearance with exact synthetic affine drift; "
            "development evidence, not real camera-motion confirmation"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
