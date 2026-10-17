#!/usr/bin/env python3
"""Evaluate learned observer correspondences on held-out control anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.geometric_baselines import refine_points_with_prototypes  # noqa: E402
from fcpg.observations import select_policy_observation  # noqa: E402
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402
from scripts.evaluate_kineparity_stress import (  # noqa: E402
    CONSTELLATIONS,
    SOLVERS,
    aggregate_rows,
    append_observer_history,
    evaluate_solver,
    parse_choices,
)
from scripts.evaluate_transformer_residual import (  # noqa: E402
    choose_device,
    load_prototype_bank,
    validate_vision_cache,
)


INTERVENTION_MODES = ("normal", "zero-features", "shuffled-features")
REQUIRED_DATA_KEYS = {
    "obs",
    "keypoints_xyz",
    "q_true",
    "q_nominal",
    "valid_true",
    "valid_nominal",
    "control_keypoints_xyz",
    "control_q_true",
    "control_q_nominal",
    "control_valid_true",
    "control_valid_nominal",
    "camera_intrinsic",
    "nominal_extrinsic",
    "task_name",
    "episode_id",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_observer_mask(
    predicted_points_px: np.ndarray,
    valid_nominal: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Build a deployable mask without consulting true pixels or visibility."""

    points = np.asarray(predicted_points_px)
    valid = np.asarray(valid_nominal, dtype=bool)
    if points.shape[:-1] != valid.shape or points.shape[-1] != 2:
        raise ValueError("predicted points and valid_nominal shapes are inconsistent")
    return (
        valid
        & np.isfinite(points).all(axis=-1)
        & (points[..., 0] >= 0.0)
        & (points[..., 0] < width)
        & (points[..., 1] >= 0.0)
        & (points[..., 1] < height)
    )


def frame_group_key(raw, index: int) -> tuple[object, ...]:
    key: list[object] = [str(raw["task_name"][index])]
    for field in ("task_index", "episode_id", "env_seed"):
        if field in raw.files:
            key.append(int(raw[field][index]))
    return tuple(key)


@torch.inference_mode()
def predict_cached_correspondences(
    model,
    raw,
    cache_dir: str | Path,
    obs_mode: str,
    mode: str,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Predict q_observer while intervening only on cached visual features."""

    if mode not in INTERVENTION_MODES:
        raise ValueError(f"unknown intervention mode: {mode}")
    cache_dir = Path(cache_dir)
    metadata = json.loads(
        (cache_dir / "metadata.json").read_text(encoding="utf-8")
    )
    backbone = np.load(cache_dir / "backbone_features.npy", mmap_mode="r")
    prefix = np.load(cache_dir / "prefix.npy", mmap_mode="r")
    sample_count = int(raw["q_nominal"].shape[0])
    if backbone.shape[0] != sample_count or prefix.shape[0] != sample_count:
        raise ValueError("vision cache and projection dataset lengths differ")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    image_size = (int(metadata["image_width"]), int(metadata["image_height"]))
    robot_obs = select_policy_observation(raw["obs"], obs_mode).astype(np.float32)
    source_indices = np.arange(sample_count, dtype=np.int64)
    if mode == "shuffled-features":
        source_indices = np.random.default_rng(seed).permutation(source_indices)

    predicted = np.empty_like(raw["q_nominal"], dtype=np.float32)
    model.eval()
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        target = np.arange(start, stop, dtype=np.int64)
        source = source_indices[target]
        feature_batch = np.array(backbone[source], dtype=np.float16, copy=True)
        prefix_batch = np.array(prefix[source], dtype=np.float16, copy=True)
        if mode == "zero-features":
            feature_batch.fill(0)
            prefix_batch.fill(0)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(value)).to(
                device,
                non_blocking=device.type == "cuda",
            )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model.forward_from_backbone_features(
                tensor(feature_batch),
                tensor(prefix_batch),
                tensor(np.asarray(raw["q_nominal"][target], dtype=np.float32)),
                tensor(np.asarray(raw["keypoints_xyz"][target], dtype=np.float32)),
                tensor(robot_obs[target]),
                image_size=image_size,
            )
        predicted[target] = output.q_corrected_px.float().cpu().numpy()
    return predicted


@torch.inference_mode()
def predict_prototype_correspondences(
    raw,
    q_nominal_px: np.ndarray,
    cache_dir: str | Path,
    descriptors: torch.Tensor,
    feature_level: int,
    image_size: tuple[int, int],
    radius_px: float,
    grid_size: int,
    temperature: float,
    min_confidence: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict observer points with the matched frozen-descriptor baseline."""

    backbone = np.load(
        Path(cache_dir) / "backbone_features.npy", mmap_mode="r"
    )
    sample_count = int(q_nominal_px.shape[0])
    if backbone.shape[0] != sample_count:
        raise ValueError("vision cache and projection dataset lengths differ")
    level = feature_level % int(backbone.shape[1])
    predicted = np.empty_like(q_nominal_px, dtype=np.float32)
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        indices = np.arange(start, stop, dtype=np.int64)
        features = torch.from_numpy(
            np.array(backbone[indices, level], dtype=np.float16, copy=True)
        ).to(device, non_blocking=device.type == "cuda")
        q_nominal = torch.from_numpy(
            np.asarray(q_nominal_px[indices], dtype=np.float32)
        ).to(device, non_blocking=device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            result = refine_points_with_prototypes(
                features,
                q_nominal,
                descriptors,
                image_size=image_size,
                radius_px=radius_px,
                grid_size=grid_size,
                temperature=temperature,
                min_confidence=min_confidence,
            )
        predicted[indices] = result.points_px.float().cpu().numpy()
    return predicted


def point_error_summary(
    q_nominal_px: np.ndarray,
    q_true_px: np.ndarray,
    predicted_px: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    mask = np.asarray(valid, dtype=bool)
    before = np.linalg.norm(q_nominal_px - q_true_px, axis=-1)[mask]
    after = np.linalg.norm(predicted_px - q_true_px, axis=-1)[mask]
    return {
        "measurements": int(after.size),
        "before_error_mean_px": float(before.mean()),
        "after_error_mean_px": float(after.mean()),
        "after_error_median_px": float(np.median(after)),
        "after_error_p90_px": float(np.percentile(after, 90)),
        "after_error_p95_px": float(np.percentile(after, 95)),
        "after_error_max_px": float(after.max()),
        "error_ratio": float(after.mean() / max(before.mean(), 1e-8)),
    }


def correspondence_summary(
    raw,
    predicted_points_px: np.ndarray,
) -> dict[str, float | int]:
    valid = np.asarray(raw["valid_nominal"], dtype=bool) & np.asarray(
        raw["valid_true"], dtype=bool
    )
    before = np.linalg.norm(raw["q_nominal"] - raw["q_true"], axis=-1)[valid]
    after = np.linalg.norm(predicted_points_px - raw["q_true"], axis=-1)[valid]
    return {
        "observer_measurements": int(after.size),
        "before_error_mean_px": float(before.mean()),
        "after_error_mean_px": float(after.mean()),
        "after_error_median_px": float(np.median(after)),
        "after_error_p90_px": float(np.percentile(after, 90)),
        "after_error_p95_px": float(np.percentile(after, 95)),
        "after_error_max_px": float(after.max()),
        "error_ratio": float(after.mean() / max(before.mean(), 1e-8)),
    }


def infer_severity(raw) -> tuple[str, float]:
    rotation = (
        np.abs(np.asarray(raw["actual_camera_rotation_deg"], dtype=np.float64))
        if "actual_camera_rotation_deg" in raw.files
        else np.zeros(len(raw["q_nominal"]), dtype=np.float64)
    )
    translation = (
        100.0
        * np.linalg.norm(
            np.asarray(raw["actual_camera_translation_xyz"], dtype=np.float64),
            axis=-1,
        )
        if "actual_camera_translation_xyz" in raw.files
        else np.zeros_like(rotation)
    )
    if float(rotation.mean()) >= float(translation.mean()):
        return "rotation_deg", float(rotation.mean())
    return "translation_cm", float(translation.mean())


def constellation_indices(
    observer_names: list[str],
    body_names: list[str],
) -> np.ndarray:
    by_body = {name.split(":", 1)[0]: index for index, name in enumerate(observer_names)}
    missing = [name for name in body_names if name not in by_body]
    if missing:
        raise ValueError(f"dataset lacks observer bodies: {missing}")
    return np.asarray([by_body[name] for name in body_names], dtype=np.int64)


def evaluate_visual_transfer(
    raw,
    predicted_points_px: np.ndarray,
    solvers: tuple[str, ...],
    observer_indices: np.ndarray,
    constellation_name: str,
    history_frames: int,
    prior_std: torch.Tensor,
    observation_std_px: float,
    width: int,
    height: int,
    observer_mask: str,
    min_whitened_singular_value: float,
    relative_singular_value: float,
    min_control_variance_reduction_fraction: float,
    min_innovation_z_score: float,
    min_global_innovation_rms_z: float,
    innovation_false_alarm_probability: float,
) -> list[dict[str, object]]:
    """Transfer learned observer correspondences to held-out control anchors."""

    if history_frames < 1:
        raise ValueError("history_frames must be positive")
    deployable = deployment_observer_mask(
        predicted_points_px,
        raw["valid_nominal"],
        width,
        height,
    )
    if observer_mask == "oracle-visibility":
        deployable &= np.asarray(raw["valid_true"], dtype=bool)
    elif observer_mask != "deployment":
        raise ValueError(f"unknown observer mask: {observer_mask}")
    observer_indices = np.asarray(observer_indices, dtype=np.int64)
    predicted_points_px = predicted_points_px[:, observer_indices]
    deployable = deployable[:, observer_indices]

    severity_name, severity = infer_severity(raw)
    rows: dict[str, list[dict[str, object]]] = {solver: [] for solver in solvers}
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    previous_group = None
    sample_count = int(predicted_points_px.shape[0])
    for index in range(sample_count):
        group = frame_group_key(raw, index)
        if group != previous_group:
            history = []
            previous_group = group
        observer_xyz, observer_px, observer_valid = append_observer_history(
            history,
            raw["keypoints_xyz"][index, observer_indices],
            predicted_points_px[index],
            deployable[index],
            history_frames,
        )
        control_valid = np.asarray(
            raw["control_valid_nominal"][index]
            & raw["control_valid_true"][index],
            dtype=bool,
        )
        if not bool(control_valid.any()):
            continue
        for solver in solvers:
            rows[solver].append(
                evaluate_solver(
                    solver,
                    raw["camera_intrinsic"][index],
                    raw["nominal_extrinsic"][index],
                    observer_xyz,
                    observer_px,
                    observer_valid,
                    raw["control_keypoints_xyz"][index],
                    raw["control_q_true"][index],
                    control_valid,
                    prior_std,
                    observation_std_px,
                    width,
                    height,
                    min_whitened_singular_value,
                    relative_singular_value,
                    min_control_variance_reduction_fraction,
                    min_innovation_z_score,
                    min_global_innovation_rms_z,
                    innovation_false_alarm_probability,
                )
            )
    return [
        aggregate_rows(
            rows[solver], severity_name, severity, constellation_name, solver
        )
        for solver in solvers
        if rows[solver]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vision-cache", required=True)
    parser.add_argument("--prototype-bank", default="")
    parser.add_argument("--control-prototype-bank", default="")
    parser.add_argument("--prototype-radius-px", type=float, default=64.0)
    parser.add_argument("--prototype-grid-size", type=int, default=17)
    parser.add_argument("--prototype-temperature", type=float, default=0.0)
    parser.add_argument("--prototype-min-confidence", type=float, default=-1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--modes",
        type=lambda value: parse_choices(
            value, set(INTERVENTION_MODES), "intervention mode"
        ),
        default="normal,zero-features,shuffled-features",
    )
    parser.add_argument(
        "--solvers",
        type=lambda value: parse_choices(value, set(SOLVERS), "solver"),
        default="kineparity,map_all,linear_6d,pnp,pnp_ransac",
    )
    parser.add_argument(
        "--constellations",
        type=lambda value: parse_choices(
            value, set(CONSTELLATIONS), "constellation"
        ),
        default="full9,uniform5,distal3",
    )
    parser.add_argument("--observer-mask", choices=("deployment", "oracle-visibility"), default="deployment")
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--prior-translation-std-cm", type=float, default=8.0)
    parser.add_argument("--prior-rotation-std-deg", type=float, default=15.0)
    parser.add_argument("--observation-std-px", type=float, default=2.0)
    parser.add_argument("--min-whitened-singular-value", type=float, default=1.0)
    parser.add_argument("--relative-singular-value", type=float, default=1e-2)
    parser.add_argument("--min-control-variance-reduction-fraction", type=float, default=1e-3)
    parser.add_argument("--min-innovation-z-score", type=float, default=0.0)
    parser.add_argument("--min-global-innovation-rms-z", type=float, default=0.0)
    parser.add_argument("--innovation-false-alarm-prob", type=float, default=0.01)
    args = parser.parse_args()
    if args.history_frames < 1 or args.batch_size < 1:
        parser.error("history-frames and batch-size must be positive")

    cv2.setRNGSeed(args.seed)
    device = choose_device(args.device)
    model, checkpoint_args = load_fcpg_vit_adapter(args.checkpoint, device)
    validate_vision_cache(args.vision_cache, args.data, checkpoint_args)
    obs_mode = str(checkpoint_args.get("obs_mode", "robot"))
    prior_std = torch.tensor(
        [
            *([args.prior_translation_std_cm / 100.0] * 3),
            *([np.deg2rad(args.prior_rotation_std_deg)] * 3),
        ],
        dtype=torch.float64,
    )

    prototype_metadata = None
    prototype_sha256 = None
    control_prototype_metadata = None
    control_prototype_sha256 = None
    with np.load(args.data, allow_pickle=False) as raw:
        missing = sorted(REQUIRED_DATA_KEYS.difference(raw.files))
        if missing:
            raise ValueError(f"dataset lacks Gate B fields: {missing}")
        layout = str(raw["keypoint_layout"].item()) if "keypoint_layout" in raw.files else "unknown"
        if layout != "observer":
            raise ValueError(f"Gate B requires observer layout, got {layout!r}")
        height, width = map(int, raw["rgb"].shape[1:3])
        observer_names = [str(value) for value in raw["keypoint_site"]]
        selected_constellations = {
            name: constellation_indices(observer_names, CONSTELLATIONS[name])
            for name in args.constellations
        }

        def transfer(predicted_points: np.ndarray) -> list[dict[str, object]]:
            summaries: list[dict[str, object]] = []
            for name, indices in selected_constellations.items():
                summaries.extend(
                    evaluate_visual_transfer(
                        raw,
                        predicted_points,
                        tuple(args.solvers),
                        indices,
                        name,
                        args.history_frames,
                        prior_std,
                        args.observation_std_px,
                        width,
                        height,
                        args.observer_mask,
                        args.min_whitened_singular_value,
                        args.relative_singular_value,
                        args.min_control_variance_reduction_fraction,
                        args.min_innovation_z_score,
                        args.min_global_innovation_rms_z,
                        args.innovation_false_alarm_prob,
                    )
                )
            return summaries

        results: dict[str, object] = {}
        for mode in args.modes:
            predicted = predict_cached_correspondences(
                model,
                raw,
                args.vision_cache,
                obs_mode,
                mode,
                args.batch_size,
                device,
                args.seed,
            )
            results[mode] = {
                "observer_correspondence": correspondence_summary(raw, predicted),
                "held_out_control_transfer": transfer(predicted),
            }
        if args.prototype_bank:
            descriptors, prototype_metadata, prototype_sha256 = load_prototype_bank(
                args.prototype_bank,
                checkpoint_args,
                (width, height),
                device,
            )
            prototype_predicted = predict_prototype_correspondences(
                raw,
                raw["q_nominal"],
                args.vision_cache,
                descriptors,
                int(prototype_metadata["feature_level"]),
                (width, height),
                args.prototype_radius_px,
                args.prototype_grid_size,
                args.prototype_temperature,
                args.prototype_min_confidence,
                args.batch_size,
                device,
            )
            results["prototype"] = {
                "observer_correspondence": correspondence_summary(
                    raw, prototype_predicted
                ),
                "held_out_control_transfer": transfer(prototype_predicted),
            }
        if args.control_prototype_bank:
            (
                control_descriptors,
                control_prototype_metadata,
                control_prototype_sha256,
            ) = load_prototype_bank(
                args.control_prototype_bank,
                checkpoint_args,
                (width, height),
                device,
            )
            if control_prototype_metadata.get("keypoint_source") != "control":
                raise ValueError(
                    "--control-prototype-bank must declare keypoint_source=control"
                )
            direct_control = predict_prototype_correspondences(
                raw,
                raw["control_q_nominal"],
                args.vision_cache,
                control_descriptors,
                int(control_prototype_metadata["feature_level"]),
                (width, height),
                args.prototype_radius_px,
                args.prototype_grid_size,
                args.prototype_temperature,
                args.prototype_min_confidence,
                args.batch_size,
                device,
            )
            results["direct_control_prototype"] = {
                "control_anchor_error": point_error_summary(
                    raw["control_q_nominal"],
                    raw["control_q_true"],
                    direct_control,
                    raw["control_valid_nominal"]
                    & raw["control_valid_true"],
                ),
                "control_visual_correspondences_used": True,
                "used_for_kineparity_solver": False,
            }
        task_names = sorted({str(value) for value in raw["task_name"]})
        observer_names = [str(value) for value in raw["keypoint_site"]]
        control_names = [str(value) for value in raw["control_keypoint_site"]]

    payload = {
        "schema_version": 1,
        "method": "kineparity_learned_observer_gate_b",
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "vision_cache": str(Path(args.vision_cache).resolve()),
        "prototype_bank": (
            str(Path(args.prototype_bank).resolve()) if args.prototype_bank else None
        ),
        "prototype_bank_sha256": prototype_sha256,
        "prototype_metadata": prototype_metadata,
        "control_prototype_bank": (
            str(Path(args.control_prototype_bank).resolve())
            if args.control_prototype_bank
            else None
        ),
        "control_prototype_bank_sha256": control_prototype_sha256,
        "control_prototype_metadata": control_prototype_metadata,
        "tasks": task_names,
        "observer_names": observer_names,
        "held_out_control_names": control_names,
        "observer_mask": args.observer_mask,
        "observer_true_pixels_used_for_solver": False,

        "control_visual_correspondences_used": False,
        "control_true_pixels_used_for_solver": False,
        "settings": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
