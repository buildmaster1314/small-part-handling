#!/usr/bin/env python3
"""Fit real SO-101 nominal projections from the annotation HTML export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fcpg.so101_camera_calibration import (
    ProjectionFit,
    fit_projection,
    predict_projection,
    projection_metrics,
    valid_projection_mask,
)
from fcpg.so101_kinematics import SO101MuJoCoKinematics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wrist-cache", type=Path, required=True)
    parser.add_argument("--column-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-every", type=int, default=6)
    parser.add_argument("--max-validation-epe", type=float, default=8.0)
    parser.add_argument("--max-cross-episode-p95", type=float, default=16.0)
    return parser.parse_args()


def _fit_dict(fit: ProjectionFit) -> dict[str, object]:
    return {
        "view": fit.view,
        "feature_names": list(fit.feature_names),
        "coefficients": np.asarray(fit.coefficients).tolist(),
        "ridge": fit.ridge,
    }


def _load_cache(path: Path) -> dict[str, object]:
    return {
        "manifest": json.loads((path / "manifest.json").read_text()),
        "states": np.load(path / "states.npy", mmap_mode="r"),
        "episodes": np.load(path / "episodes.npy", mmap_mode="r"),
        "frame_indices": np.load(path / "frame_indices.npy", mmap_mode="r"),
    }


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text())
    if annotations.get("version") != "so101-camera-annotations-v1":
        raise ValueError("unsupported annotation export version")
    samples = annotations.get("samples")
    points = annotations.get("points")
    if not isinstance(samples, list) or not isinstance(points, dict) or not samples:
        raise ValueError("annotations must contain samples and points")
    if (
        args.validation_every <= 1
        or args.max_validation_epe <= 0
        or args.max_cross_episode_p95 <= 0
    ):
        raise ValueError("validation settings are invalid")

    wrist = _load_cache(args.wrist_cache.resolve())
    column = _load_cache(args.column_cache.resolve())
    if wrist["manifest"]["dataset_id"] != column["manifest"]["dataset_id"]:
        raise ValueError("paired caches must have the same dataset_id")
    for key in ("episodes", "frame_indices", "states"):
        if not np.array_equal(wrist[key], column[key]):
            raise ValueError(f"paired caches disagree on {key}")

    all_rows = np.asarray([int(sample["row"]) for sample in samples], dtype=np.int64)
    if len(np.unique(all_rows)) != len(all_rows):
        raise ValueError("annotation samples contain duplicate rows")
    if np.any(all_rows < 0) or np.any(all_rows >= len(wrist["states"])):
        raise ValueError("annotation row is outside the cache")
    observed: dict[str, np.ndarray] = {}
    complete_indices: dict[str, np.ndarray] = {}
    for view in ("wrist", "column"):
        values: list[list[list[float]]] = []
        indices: list[int] = []
        for sample_index, sample in enumerate(samples):
            key = f"{sample['id']}:{view}"
            value = points.get(key)
            if value is None or value == []:
                continue
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"{key} must be empty or contain exactly three clicks")
            values.append(value)
            indices.append(sample_index)
        if len(indices) <= args.validation_every:
            raise ValueError(
                f"{view} needs more than {args.validation_every} complete frames; "
                f"got {len(indices)}"
            )
        observed[view] = np.asarray(values, dtype=np.float64)
        complete_indices[view] = np.asarray(indices, dtype=np.int64)
    if not all(np.all(np.isfinite(value)) for value in observed.values()):
        raise ValueError("clicked pixels must be finite")

    kinematics = SO101MuJoCoKinematics(args.model)
    states = np.asarray(wrist["states"])[all_rows].astype(np.float64)
    base_xyz, gripper_xyz = kinematics.forward_anchor_frames(states)
    gripper = states[:, 5]
    episodes = np.asarray(wrist["episodes"])[all_rows].astype(np.int64)
    result: dict[str, object] = {
        "version": "so101-camera-calibration-v1",
        "dataset_id": str(wrist["manifest"]["dataset_id"]),
        "image_size": [int(annotations["width"]), int(annotations["height"])],
        "model": str(args.model.resolve()),
        "annotation_rows": {
            view: all_rows[complete_indices[view]].tolist()
            for view in ("wrist", "column")
        },
        "validation_rows": {},
        "fits": {},
        "metrics": {},
    }
    fits: dict[str, ProjectionFit] = {}
    fit_payload = result["fits"]
    metric_payload = result["metrics"]
    assert isinstance(fit_payload, dict) and isinstance(metric_payload, dict)
    for view in ("wrist", "column"):
        indices = complete_indices[view]
        train = np.arange(len(indices)) % args.validation_every != 0
        validation = ~train
        view_base = base_xyz[indices]
        view_gripper_xyz = gripper_xyz[indices]
        view_gripper = gripper[indices]
        validation_rows = result["validation_rows"]
        assert isinstance(validation_rows, dict)
        validation_rows[view] = all_rows[indices[validation]].tolist()
        diagnostic_fit = fit_projection(
            view,
            view_base[train],
            view_gripper_xyz[train],
            view_gripper[train],
            observed[view][train],
        )
        predicted_train = predict_projection(
            diagnostic_fit,
            view_base[train],
            view_gripper_xyz[train],
            view_gripper[train],
        )
        predicted_val = predict_projection(
            diagnostic_fit,
            view_base[validation],
            view_gripper_xyz[validation],
            view_gripper[validation],
        )
        train_metrics = projection_metrics(predicted_train, observed[view][train])
        val_metrics = projection_metrics(
            predicted_val,
            observed[view][validation],
            image_size=(int(annotations["width"]), int(annotations["height"])),
        )
        cross_predictions: list[np.ndarray] = []
        cross_targets: list[np.ndarray] = []
        cross_episodes: dict[str, object] = {}
        view_episodes = episodes[indices]
        for episode in np.unique(view_episodes):
            fold_validation = view_episodes == episode
            fold_train = ~fold_validation
            if int(np.sum(fold_train)) <= args.validation_every:
                raise ValueError(f"{view} episode fold has too few training frames")
            fold_fit = fit_projection(
                view,
                view_base[fold_train],
                view_gripper_xyz[fold_train],
                view_gripper[fold_train],
                observed[view][fold_train],
            )
            fold_prediction = predict_projection(
                fold_fit,
                view_base[fold_validation],
                view_gripper_xyz[fold_validation],
                view_gripper[fold_validation],
            )
            cross_predictions.append(fold_prediction)
            cross_targets.append(observed[view][fold_validation])
            cross_episodes[str(int(episode))] = projection_metrics(
                fold_prediction, observed[view][fold_validation]
            )
        cross_metrics = projection_metrics(
            np.concatenate(cross_predictions, axis=0),
            np.concatenate(cross_targets, axis=0),
            image_size=(int(annotations["width"]), int(annotations["height"])),
        )
        cross_metrics["per_episode"] = cross_episodes

        # Once the independent episode folds pass, refit on every clicked frame
        # to maximize the quality of the deployable nominal projection.
        fit = fit_projection(
            view, view_base, view_gripper_xyz, view_gripper, observed[view]
        )
        fits[view] = fit
        fit_all_prediction = predict_projection(
            fit, view_base, view_gripper_xyz, view_gripper
        )
        fit_all_metrics = projection_metrics(fit_all_prediction, observed[view])
        fit_payload[view] = _fit_dict(fit)
        metric_payload[view] = {
            "diagnostic_train": train_metrics,
            "diagnostic_validation": val_metrics,
            "cross_episode": cross_metrics,
            "fit_all": fit_all_metrics,
        }
        if float(cross_metrics["mean_epe_px"]) > args.max_validation_epe:
            raise RuntimeError(
                f"{view} cross-episode projection error is too high: "
                f"{cross_metrics['mean_epe_px']:.3f}px > {args.max_validation_epe:.3f}px"
            )
        if float(cross_metrics["p95_epe_px"]) > args.max_cross_episode_p95:
            raise RuntimeError(
                f"{view} cross-episode P95 projection error is too high: "
                f"{cross_metrics['p95_epe_px']:.3f}px > "
                f"{args.max_cross_episode_p95:.3f}px"
            )

    # Materialize deployable nominal points only after both held-out fits pass.
    all_states = np.asarray(wrist["states"], dtype=np.float64)
    all_base, all_gripper_xyz = kinematics.forward_anchor_frames(all_states)
    all_gripper = all_states[:, 5]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for view, fit in fits.items():
        q_nominal = predict_projection(
            fit, all_base, all_gripper_xyz, all_gripper
        ).astype(np.float32)
        valid = valid_projection_mask(
            q_nominal,
            image_size=(int(annotations["width"]), int(annotations["height"])),
            margin=1.0,
        )
        np.save(output / f"q_nominal_px_{view}.npy", q_nominal)
        np.save(output / f"valid_{view}.npy", valid)
        view_metrics = metric_payload[view]
        assert isinstance(view_metrics, dict)
        view_metrics["all_in_bounds_fraction"] = float(np.mean(valid))
    result["status"] = "passed"
    (output / "calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"calibration={output / 'calibration.json'}")
    for view in fits:
        view_metrics = metric_payload[view]
        print(
            f"{view}: cross_episode_mean_epe_px="
            f"{view_metrics['cross_episode']['mean_epe_px']:.3f}, "
            f"p95_epe_px={view_metrics['cross_episode']['p95_epe_px']:.3f}"
        )


if __name__ == "__main__":
    main()
