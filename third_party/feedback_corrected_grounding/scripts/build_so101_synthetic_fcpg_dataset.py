#!/usr/bin/env python3
"""Build a balanced synthetic camera-drift dataset from SO-101 trajectories."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from fcpg.so101_synthetic_dataset import (
    ARRAY_NAMES,
    DATASET_FORMAT,
    balanced_counts,
    sample_camera_drift,
)
from fcpg.so101_synthetic_grounding import SO101SyntheticGroundingRenderer
from scripts.render_so101_mujoco_prediction import DEFAULT_SCENE
from scripts.train_so101_lewm import Cache, episode_splits, parse_exclusions
from scripts.visualize_so101_lewm import checkpoint_contract


CAMERAS = (
    {"name": "front", "azimuth": 135.0, "elevation": -25.0, "distance": 0.72},
    {"name": "side", "azimuth": 45.0, "elevation": -25.0, "distance": 0.72},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--cache-base", type=Path)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=53101)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--clean-probability", type=float, default=0.15)
    parser.add_argument(
        "--drifts-per-view",
        type=int,
        default=1,
        help=(
            "Camera realizations per physical state and view. With values above "
            "one, variant zero is clean and every remaining variant is drifted."
        ),
    )
    parser.add_argument("--max-azimuth", type=float, default=10.0)
    parser.add_argument("--max-elevation", type=float, default=4.0)
    parser.add_argument("--max-distance", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(
        args.train_samples,
        args.val_samples,
        args.width,
        args.height,
        args.drifts_per_view,
    ) <= 0:
        parser.error("sample counts and render dimensions must be positive")
    if not args.scene.is_file():
        parser.error(f"SO-101 scene not found: {args.scene}")
    return args


def grouped_rows(
    caches: list[Cache],
    assignments: dict[tuple[str, int], str],
    split: str,
) -> list[tuple[int, int, str, np.ndarray]]:
    groups: list[tuple[int, int, str, np.ndarray]] = []
    for cache_index, cache in enumerate(caches):
        episodes = np.asarray(cache.episodes)
        tasks = np.asarray(cache.tasks)
        by_task: dict[int, list[np.ndarray]] = defaultdict(list)
        for raw_episode in np.unique(episodes):
            episode = int(raw_episode)
            if assignments.get((cache.dataset_id, episode)) != split:
                continue
            rows = np.flatnonzero(episodes == episode)
            task_ids = np.unique(tasks[rows])
            if len(task_ids) != 1:
                raise ValueError(f"episode spans tasks: {cache.name}:{episode}")
            by_task[int(task_ids[0])].append(rows)
        for task, row_groups in sorted(by_task.items()):
            rows = np.concatenate(row_groups)
            groups.append((cache_index, task, cache.task_key(task), rows))
    if not groups:
        raise ValueError(f"no {split} rows available")
    return groups


def select_rows(
    groups: list[tuple[int, int, str, np.ndarray]],
    total: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, str]]:
    selected: list[tuple[int, int, int, str]] = []
    for (cache_index, task, task_name, rows), count in zip(
        groups,
        balanced_counts(total, len(groups)),
        strict=True,
    ):
        replace = int(count) > len(rows)
        choices = rng.choice(rows, size=int(count), replace=replace)
        selected.extend(
            (cache_index, int(row), task, task_name) for row in choices
        )
    rng.shuffle(selected)
    return selected


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
        raise AssertionError("synthetic FCPG array contract drifted")
    return {
        name: np.lib.format.open_memmap(
            root / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        for name, (shape, dtype) in specifications.items()
    }


def camera_context(camera: dict[str, float | str]) -> np.ndarray:
    azimuth = np.deg2rad(float(camera["azimuth"]))
    elevation = np.deg2rad(float(camera["elevation"]))
    return np.asarray(
        (
            np.sin(azimuth),
            np.cos(azimuth),
            np.sin(elevation),
            np.cos(elevation),
            float(camera["distance"]),
        ),
        dtype=np.float32,
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _, embedded_args, cache_paths, _, _, checkpoint_sha = checkpoint_contract(
        args.checkpoint.resolve(), args.cache_base
    )
    caches = [Cache(path) for path in cache_paths]
    assignments = episode_splits(
        caches,
        parse_exclusions(embedded_args.get("exclude_episode", [])),
        split_seed=int(embedded_args["split_seed"]),
        val_fraction=float(embedded_args["val_fraction"]),
    )
    rng = np.random.default_rng(args.seed)
    samples_per_state = len(CAMERAS) * args.drifts_per_view
    if args.train_samples % samples_per_state or args.val_samples % samples_per_state:
        raise ValueError(
            "train and validation sample counts must be divisible by camera "
            "count times drifts-per-view"
        )
    train_states = select_rows(
        grouped_rows(caches, assignments, "train"),
        args.train_samples // samples_per_state,
        rng,
    )
    val_states = select_rows(
        grouped_rows(caches, assignments, "val"),
        args.val_samples // samples_per_state,
        rng,
    )
    records = []
    for split, physical_records in ((0, train_states), (1, val_states)):
        for record in physical_records:
            drifts = []
            for drift_variant in range(args.drifts_per_view):
                clean_variant = args.drifts_per_view > 1 and drift_variant == 0
                drifts.append(
                    sample_camera_drift(
                        rng,
                        clean_probability=(
                            1.0
                            if clean_variant
                            else 0.0
                            if args.drifts_per_view > 1
                            else args.clean_probability
                        ),
                        max_azimuth_degrees=args.max_azimuth,
                        max_elevation_degrees=args.max_elevation,
                        max_distance_meters=args.max_distance,
                    )
                )
            for camera_index in range(len(CAMERAS)):
                for drift_variant, drift in enumerate(drifts):
                    records.append(
                        (
                            split,
                            camera_index,
                            drift_variant,
                            drift,
                            *record,
                        )
                    )
    arrays = allocate_arrays(output, len(records), args.height, args.width)
    task_names: dict[str, str] = {}
    renderers = [
        SO101SyntheticGroundingRenderer(
            args.scene,
            width=args.width,
            height=args.height,
            azimuth=float(camera["azimuth"]),
            elevation=float(camera["elevation"]),
            distance=float(camera["distance"]),
        )
        for camera in CAMERAS
    ]
    try:
        for index, (
            split,
            camera_index,
            drift_variant,
            drift,
            cache_index,
            row,
            task,
            task_name,
        ) in enumerate(records):
            cache = caches[cache_index]
            state = np.asarray(cache.states[row], dtype=np.float32)
            frame = renderers[camera_index].render(state, drift)
            arrays["rgb"][index] = frame.rgb
            arrays["states"][index] = state
            arrays["anchors_xyz"][index] = frame.anchors_xyz
            arrays["q_nominal"][index] = frame.q_nominal_px
            arrays["q_true"][index] = frame.q_true_px
            arrays["valid"][index] = frame.valid_nominal & frame.valid_true
            arrays["drift"][index] = (
                drift.azimuth_degrees,
                drift.elevation_degrees,
                drift.distance_meters,
            )
            arrays["camera_index"][index] = camera_index
            arrays["camera_context"][index] = camera_context(CAMERAS[camera_index])
            arrays["drift_variant"][index] = drift_variant
            arrays["split"][index] = split
            arrays["source_index"][index] = cache_index
            arrays["source_row"][index] = row
            arrays["episode"][index] = int(cache.episodes[row])
            arrays["task"][index] = task
            task_names[f"{cache_index}:{task}"] = task_name
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                print(f"rendered {index + 1}/{len(records)}", flush=True)
    finally:
        for renderer in renderers:
            renderer.close()
    for values in arrays.values():
        values.flush()
    valid = np.asarray(arrays["valid"])
    residual = np.asarray(arrays["q_true"]) - np.asarray(arrays["q_nominal"])
    epe = np.linalg.norm(residual, axis=-1)
    nonclean = np.any(np.abs(np.asarray(arrays["drift"])) > 1.0e-8, axis=1)
    manifest = {
        "format": DATASET_FORMAT,
        "samples": len(records),
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "train_physical_states": len(train_states),
        "val_physical_states": len(val_states),
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "scene": str(args.scene.resolve()),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha,
        "sources": [str(path) for path in cache_paths],
        "task_names": task_names,
        "camera": {
            "views": list(CAMERAS),
            "context": "sin/cos azimuth, sin/cos elevation, distance",
            "drifts_per_view": args.drifts_per_view,
            "clean_probability": args.clean_probability,
            "max_azimuth": args.max_azimuth,
            "max_elevation": args.max_elevation,
            "max_distance": args.max_distance,
            "perturbation": (
                "common-rig fixed-position yaw/pitch plus optical-axis translation; "
                "the same drift variant is shared by both cameras"
            ),
        },
        "valid_anchor_fraction": float(valid.mean()),
        "nonclean_mean_epe_px": float(epe[nonclean[:, None] & valid].mean()),
        "gripper_caveat": "official MJCF lacks the LeRobot linear gripper calibration",
        "scope": "synthetic geometry supervision; not real-camera evidence",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
