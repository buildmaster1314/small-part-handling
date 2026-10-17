#!/usr/bin/env python3
"""Stress-test observer-to-control transfer under noisy, missing correspondences."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.geometry import Camera, project_points
from fcpg.metaworld_utils import (
    DEFAULT_KEYPOINT_SITES,
    choose_action,
    get_link_observer_points,
    get_site_keypoints,
    make_env,
    make_scripted_policy,
)
from fcpg.mujoco_camera import (
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    set_fixed_camera_pose_offset,
)
from fcpg.structured_grounding import (
    project_points_torch,
    projection_jacobian_left_se3,
    se3_exp,
    solve_observer_control_transfer,
)
from scripts.evaluate_kineparity_oracle import (
    fixed_camera_offset,
    parse_severity,
    tensor_batch,
)


CONSTELLATIONS = {
    "full9": [
        "right_arm_base_link",
        "right_l0",
        "right_l1",
        "right_l2",
        "right_l3",
        "right_l4",
        "right_l5",
        "right_l6",
        "right_hand",
    ],
    "uniform5": [
        "right_arm_base_link",
        "right_l1",
        "right_l3",
        "right_l5",
        "right_hand",
    ],
    "distal3": ["right_l5", "right_l6", "right_hand"],
}
SOLVERS = ("kineparity", "map_all", "linear_6d", "pnp", "pnp_ransac")


def parse_choices(value: str, choices: set[str], label: str) -> list[str]:
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in choices:
            raise argparse.ArgumentTypeError(
                f"unknown {label} {item!r}; choose from {sorted(choices)}"
            )
        if item not in parsed:
            parsed.append(item)
    if not parsed:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    return parsed


def corrupt_observer_correspondences(
    true_points_px: np.ndarray,
    valid: np.ndarray,
    noise_std_px: float,
    dropout_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if noise_std_px < 0:
        raise ValueError("noise_std_px must be non-negative")
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("dropout_probability must lie in [0, 1]")
    measured = np.asarray(true_points_px, dtype=np.float64).copy()
    measured += rng.normal(0.0, noise_std_px, size=measured.shape)
    keep = rng.random(valid.shape) >= dropout_probability
    return measured, np.asarray(valid, dtype=bool) & keep


def append_observer_history(
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    observer_xyz: np.ndarray,
    observer_points_px: np.ndarray,
    observer_valid: np.ndarray,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    history.append(
        (
            np.asarray(observer_xyz, dtype=np.float64).copy(),
            np.asarray(observer_points_px, dtype=np.float64).copy(),
            np.asarray(observer_valid, dtype=bool).copy(),
        )
    )
    del history[:-max_frames]
    return tuple(np.concatenate(items, axis=0) for items in zip(*history))


def transform_update_magnitude(
    corrected_extrinsic: np.ndarray,
    nominal_extrinsic: np.ndarray,
) -> tuple[float, float]:
    delta = corrected_extrinsic @ np.linalg.inv(nominal_extrinsic)
    translation_cm = 100.0 * float(np.linalg.norm(delta[:3, 3]))
    cosine = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    return translation_cm, rotation_deg


def solve_linear_6d(
    intrinsic: np.ndarray,
    nominal_extrinsic: np.ndarray,
    observer_xyz: np.ndarray,
    observer_points_px: np.ndarray,
    observer_valid: np.ndarray,
    control_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    _, depth, jacobian = projection_jacobian_left_se3(
        tensor_batch(intrinsic),
        tensor_batch(nominal_extrinsic),
        tensor_batch(observer_xyz),
    )
    nominal_observers, _ = project_points_torch(
        tensor_batch(intrinsic),
        tensor_batch(nominal_extrinsic),
        tensor_batch(observer_xyz),
    )
    valid = torch.from_numpy(observer_valid).to(dtype=torch.bool)
    valid = valid & (depth[0] > 1e-8)
    coordinate_valid = valid[:, None].expand(-1, 2).reshape(-1)
    design = jacobian[0].reshape(-1, 6)[coordinate_valid]
    target = (
        torch.from_numpy(observer_points_px).to(dtype=torch.float64)
        - nominal_observers[0]
    ).reshape(-1)[coordinate_valid]
    rank = int(torch.linalg.matrix_rank(design).item()) if design.numel() else 0
    if design.numel():
        twist = torch.linalg.pinv(design, rtol=1e-12) @ target
    else:
        twist = torch.zeros(6, dtype=torch.float64)
    corrected_extrinsic = (
        se3_exp(twist.unsqueeze(0))[0] @ tensor_batch(nominal_extrinsic)[0]
    )
    corrected_control, _ = project_points_torch(
        tensor_batch(intrinsic),
        corrected_extrinsic.unsqueeze(0),
        tensor_batch(control_xyz),
    )
    return (
        corrected_control[0].cpu().numpy(),
        corrected_extrinsic.cpu().numpy(),
        rank,
        bool(design.numel()),
    )


def solve_pnp(
    intrinsic: np.ndarray,
    nominal_extrinsic: np.ndarray,
    observer_xyz: np.ndarray,
    observer_points_px: np.ndarray,
    observer_valid: np.ndarray,
    control_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    mask = (
        np.asarray(observer_valid, dtype=bool)
        & np.isfinite(observer_xyz).all(axis=1)
        & np.isfinite(observer_points_px).all(axis=1)
    )
    corrected = np.asarray(nominal_extrinsic, dtype=np.float64).copy()
    solved = False
    if int(mask.sum()) >= 4:
        rvec, _ = cv2.Rodrigues(corrected[:3, :3])
        tvec = corrected[:3, 3:4].copy()
        try:
            ok, rvec, tvec = cv2.solvePnP(
                np.asarray(observer_xyz[mask], dtype=np.float64),
                np.asarray(observer_points_px[mask], dtype=np.float64),
                np.asarray(intrinsic, dtype=np.float64),
                None,
                rvec,
                tvec,
                True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            ok = False
        if ok:
            rotation, _ = cv2.Rodrigues(rvec)
            corrected[:3, :3] = rotation
            corrected[:3, 3] = tvec.reshape(3)
            solved = True
    camera = Camera(intrinsic, corrected, width=1, height=1)
    corrected_control, _ = project_points(control_xyz, camera)
    return corrected_control, corrected, min(int(mask.sum()) * 2, 6), solved


def solve_pnp_ransac(
    intrinsic: np.ndarray,
    nominal_extrinsic: np.ndarray,
    observer_xyz: np.ndarray,
    observer_points_px: np.ndarray,
    observer_valid: np.ndarray,
    control_xyz: np.ndarray,
    reprojection_error_px: float = 8.0,
    iterations: int = 100,
    confidence: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Estimate camera extrinsics with RANSAC and refine on its inliers."""

    mask = (
        np.asarray(observer_valid, dtype=bool)
        & np.isfinite(observer_xyz).all(axis=1)
        & np.isfinite(observer_points_px).all(axis=1)
    )
    corrected = np.asarray(nominal_extrinsic, dtype=np.float64).copy()
    solved = False
    inlier_count = 0
    if int(mask.sum()) >= 4:
        object_points = np.asarray(observer_xyz[mask], dtype=np.float64)
        image_points = np.asarray(observer_points_px[mask], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(corrected[:3, :3])
        tvec = corrected[:3, 3:4].copy()
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                np.asarray(intrinsic, dtype=np.float64),
                None,
                rvec,
                tvec,
                True,
                iterationsCount=iterations,
                reprojectionError=reprojection_error_px,
                confidence=confidence,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            ok = False
            inliers = None
        if ok and inliers is not None and int(inliers.size) >= 4:
            inlier_indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
            inlier_count = int(inlier_indices.size)
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points[inlier_indices],
                    image_points[inlier_indices],
                    np.asarray(intrinsic, dtype=np.float64),
                    None,
                    rvec,
                    tvec,
                    True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                ok = False
            if ok:
                rotation, _ = cv2.Rodrigues(rvec)
                corrected[:3, :3] = rotation
                corrected[:3, 3] = tvec.reshape(3)
                solved = True
    camera = Camera(intrinsic, corrected, width=1, height=1)
    corrected_control, _ = project_points(control_xyz, camera)
    return corrected_control, corrected, min(inlier_count * 2, 6), solved


def evaluate_solver(
    solver: str,
    intrinsic: np.ndarray,
    nominal_extrinsic: np.ndarray,
    observer_xyz: np.ndarray,
    observer_points_px: np.ndarray,
    observer_valid: np.ndarray,
    control_xyz: np.ndarray,
    control_true_px: np.ndarray,
    control_valid: np.ndarray,
    prior_std: torch.Tensor,
    observation_std_px: float,
    width: int,
    height: int,
    min_whitened_singular_value: float,
    relative_singular_value: float,
    min_control_variance_reduction_fraction: float,
    min_innovation_z_score: float,
    min_global_innovation_rms_z: float,
    innovation_false_alarm_probability: float,
) -> dict[str, object]:
    nominal_camera = Camera(intrinsic, nominal_extrinsic, width, height)
    control_nominal, _ = project_points(control_xyz, nominal_camera)
    singular_values = np.zeros(6, dtype=np.float64)
    retained_modes = np.zeros(6, dtype=bool)
    prior_variance = float("nan")
    posterior_variance = float("nan")
    global_innovation_rms_z = float("nan")
    global_innovation_threshold_rms_z = float("nan")

    if solver in {"kineparity", "map_all"}:
        filtered = solver == "kineparity"
        output = solve_observer_control_transfer(
            tensor_batch(intrinsic),
            tensor_batch(nominal_extrinsic),
            tensor_batch(observer_xyz),
            tensor_batch(observer_points_px),
            tensor_batch(control_xyz),
            prior_std=prior_std,
            observation_std_px=observation_std_px,
            observer_valid=torch.from_numpy(observer_valid).unsqueeze(0),
            min_whitened_singular_value=(
                min_whitened_singular_value if filtered else 1e-12
            ),
            relative_singular_value=(relative_singular_value if filtered else 0.0),
            min_control_variance_reduction_fraction=(
                min_control_variance_reduction_fraction if filtered else 0.0
            ),
            min_innovation_z_score=(min_innovation_z_score if filtered else 0.0),
            min_global_innovation_rms_z=(
                min_global_innovation_rms_z if filtered else 0.0
            ),
            innovation_false_alarm_probability=(
                innovation_false_alarm_probability if filtered else 0.0
            ),
        )
        corrected_control = output.control_corrected_points_px[0].cpu().numpy()
        corrected_extrinsic = output.corrected_extrinsic[0].cpu().numpy()
        rank = int(output.update.effective_rank.item())
        solved = rank > 0
        singular_values = output.update.singular_values[0].cpu().numpy()
        retained_modes = output.update.retained_modes[0].cpu().numpy()
        prior_variance = float(output.update.prior_control_variance.item())
        posterior_variance = float(output.update.posterior_control_variance.item())
        global_innovation_rms_z = float(
            output.update.global_innovation_rms_z.item()
        )
        global_innovation_threshold_rms_z = float(
            output.update.global_innovation_threshold_rms_z.item()
        )
    elif solver == "linear_6d":
        corrected_control, corrected_extrinsic, rank, solved = solve_linear_6d(
            intrinsic,
            nominal_extrinsic,
            observer_xyz,
            observer_points_px,
            observer_valid,
            control_xyz,
        )
    elif solver == "pnp":
        corrected_control, corrected_extrinsic, rank, solved = solve_pnp(
            intrinsic,
            nominal_extrinsic,
            observer_xyz,
            observer_points_px,
            observer_valid,
            control_xyz,
        )
    elif solver == "pnp_ransac":
        corrected_control, corrected_extrinsic, rank, solved = solve_pnp_ransac(
            intrinsic,
            nominal_extrinsic,
            observer_xyz,
            observer_points_px,
            observer_valid,
            control_xyz,
        )
    else:
        raise ValueError(f"unknown solver: {solver}")

    valid = np.asarray(control_valid, dtype=bool)
    before = np.linalg.norm(control_nominal[valid] - control_true_px[valid], axis=1)
    after = np.linalg.norm(corrected_control[valid] - control_true_px[valid], axis=1)
    displacement = np.linalg.norm(
        corrected_control[valid] - control_nominal[valid],
        axis=1,
    )
    failure_penalty = 2.0 * math.hypot(width, height)
    after = np.where(np.isfinite(after), after, failure_penalty)
    displacement = np.where(
        np.isfinite(displacement),
        displacement,
        failure_penalty,
    )
    translation_cm, rotation_deg = transform_update_magnitude(
        corrected_extrinsic,
        nominal_extrinsic,
    )
    if not np.isfinite(translation_cm):
        translation_cm = failure_penalty
    if not np.isfinite(rotation_deg):
        rotation_deg = 180.0
    return {
        "before": before,
        "after": after,
        "displacement": displacement,
        "valid_observer_count": int(np.asarray(observer_valid).sum()),
        "effective_rank": rank,
        "singular_values": singular_values,
        "retained_modes": retained_modes,
        "prior_control_variance": prior_variance,
        "posterior_control_variance": posterior_variance,
        "global_innovation_rms_z": global_innovation_rms_z,
        "global_innovation_threshold_rms_z": global_innovation_threshold_rms_z,
        "solver_success": bool(solved),
        "update_translation_cm": translation_cm,
        "update_rotation_deg": rotation_deg,
    }


def aggregate_rows(
    rows: list[dict[str, object]],
    perturbation: str,
    severity: float,
    constellation: str,
    solver: str,
) -> dict[str, object]:
    before = np.concatenate([np.asarray(row["before"]) for row in rows])
    after = np.concatenate([np.asarray(row["after"]) for row in rows])
    displacement = np.concatenate(
        [np.asarray(row["displacement"]) for row in rows]
    )
    ranks = np.asarray([row["effective_rank"] for row in rows], dtype=np.float64)
    valid_counts = np.asarray(
        [row["valid_observer_count"] for row in rows],
        dtype=np.float64,
    )
    success = np.asarray([row["solver_success"] for row in rows], dtype=bool)
    translations = np.asarray(
        [row["update_translation_cm"] for row in rows],
        dtype=np.float64,
    )
    rotations = np.asarray(
        [row["update_rotation_deg"] for row in rows],
        dtype=np.float64,
    )
    global_innovation = np.asarray(
        [row["global_innovation_rms_z"] for row in rows],
        dtype=np.float64,
    )
    global_innovation_threshold = np.asarray(
        [row["global_innovation_threshold_rms_z"] for row in rows],
        dtype=np.float64,
    )
    singular_values = np.stack(
        [np.asarray(row["singular_values"]) for row in rows]
    )
    retained_modes = np.stack(
        [np.asarray(row["retained_modes"]) for row in rows]
    )
    prior_variance = np.asarray(
        [row["prior_control_variance"] for row in rows],
        dtype=np.float64,
    )
    posterior_variance = np.asarray(
        [row["posterior_control_variance"] for row in rows],
        dtype=np.float64,
    )
    finite_variance = (
        np.isfinite(prior_variance)
        & np.isfinite(posterior_variance)
        & (prior_variance > 0)
    )
    before_mean = float(before.mean())
    after_mean = float(after.mean())
    return {
        "perturbation": perturbation,
        "severity": severity,
        "constellation": constellation,
        "solver": solver,
        "samples": len(rows),
        "control_anchor_observations": int(after.size),
        "before_error_mean_px": before_mean,
        "after_error_mean_px": after_mean,
        "after_error_median_px": float(np.median(after)),
        "after_error_p95_px": float(np.percentile(after, 95)),
        "after_error_max_px": float(after.max()),
        "error_ratio": after_mean / before_mean if before_mean > 1e-9 else None,
        "worsened_anchor_fraction": float(np.mean(after > before)),
        "catastrophic_anchor_fraction_64px": float(np.mean(after > 64.0)),
        "control_displacement_p95_px": float(np.percentile(displacement, 95)),
        "valid_observer_count_mean": float(valid_counts.mean()),
        "all_observers_dropped_fraction": float(np.mean(valid_counts == 0)),
        "effective_rank_mean": float(ranks.mean()),
        "effective_rank_min": int(ranks.min()),
        "solver_success_fraction": float(success.mean()),
        "singular_values_mean": singular_values.mean(axis=0).tolist(),
        "retained_mode_fraction": retained_modes.mean(axis=0).tolist(),
        "global_innovation_rms_z_mean": (
            float(np.nanmean(global_innovation))
            if bool(np.isfinite(global_innovation).any())
            else None
        ),
        "global_innovation_rms_z_p95": (
            float(np.nanpercentile(global_innovation, 95))
            if bool(np.isfinite(global_innovation).any())
            else None
        ),
        "global_innovation_threshold_rms_z_mean": (
            float(np.nanmean(global_innovation_threshold))
            if bool(np.isfinite(global_innovation_threshold).any())
            else None
        ),
        "posterior_to_prior_variance_ratio": (
            float(
                np.mean(
                    posterior_variance[finite_variance]
                    / prior_variance[finite_variance]
                )
            )
            if bool(finite_variance.any())
            else None
        ),
        "update_translation_p95_cm": float(np.percentile(translations, 95)),
        "update_translation_max_cm": float(translations.max()),
        "update_rotation_p95_deg": float(np.percentile(rotations, 95)),
        "update_rotation_max_deg": float(rotations.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="pick-place-v3")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--perturbation",
        choices=("rotation", "translation"),
        default="rotation",
    )
    parser.add_argument("--severities", type=parse_severity, default="10")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=20)
    parser.add_argument("--trials-per-frame", type=int, default=10)
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument("--task-index-offset", type=int, default=3000)
    parser.add_argument("--env-seed-base", type=int, default=90000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--observer-noise-std-px", type=float, default=2.0)
    parser.add_argument("--observer-dropout-prob", type=float, default=0.25)
    parser.add_argument(
        "--constellations",
        type=lambda value: parse_choices(
            value,
            set(CONSTELLATIONS),
            "constellation",
        ),
        default="full9,uniform5,distal3",
    )
    parser.add_argument(
        "--solvers",
        type=lambda value: parse_choices(value, set(SOLVERS), "solver"),
        default=",".join(SOLVERS),
    )
    parser.add_argument("--prior-translation-std-cm", type=float, default=8.0)
    parser.add_argument("--prior-rotation-std-deg", type=float, default=15.0)
    parser.add_argument("--observation-std-px", type=float, default=2.0)
    parser.add_argument("--min-whitened-singular-value", type=float, default=1.0)
    parser.add_argument("--relative-singular-value", type=float, default=1e-2)
    parser.add_argument(
        "--min-control-variance-reduction-fraction",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--min-innovation-z-score", type=float, default=0.0)
    parser.add_argument("--min-global-innovation-rms-z", type=float, default=0.0)
    parser.add_argument("--innovation-false-alarm-prob", type=float, default=0.01)
    parser.add_argument(
        "--output",
        default="outputs/kineparity/gate_a/pick_rotation_stress.json",
    )
    args = parser.parse_args()
    if min(
        args.episodes,
        args.steps_per_episode,
        args.trials_per_frame,
        args.history_frames,
    ) < 1:
        parser.error(
            "episodes, steps-per-episode, trials-per-frame and history-frames "
            "must be positive"
        )
    if args.observer_noise_std_px < 0:
        parser.error("observer-noise-std-px must be non-negative")
    if not 0.0 <= args.observer_dropout_prob <= 1.0:
        parser.error("observer-dropout-prob must lie in [0, 1]")

    os.environ.setdefault("MUJOCO_GL", "egl")
    policy = make_scripted_policy(args.task)
    prior_std = torch.tensor(
        [
            *([args.prior_translation_std_cm / 100.0] * 3),
            *([np.deg2rad(args.prior_rotation_std_deg)] * 3),
        ],
        dtype=torch.float64,
    )
    rows: dict[tuple[float, str, str], list[dict[str, object]]] = {}
    manifests: dict[str, list[str]] = {}
    control_names = None
    for severity in args.severities:
        for episode in range(args.episodes):
            env_seed = args.env_seed_base + episode
            env = make_env(
                args.task,
                args.camera,
                args.width,
                args.height,
                task_index=args.task_index_offset + episode,
            )
            obs, _ = env.reset(seed=env_seed)
            camera_id = get_camera_id(env.model, args.camera)
            intrinsic = camera_intrinsic_from_fovy(
                args.width,
                args.height,
                float(env.model.cam_fovy[camera_id]),
            )
            base_position = np.asarray(env.model.cam_pos[camera_id]).copy()
            base_quaternion = np.asarray(env.model.cam_quat[camera_id]).copy()
            nominal_extrinsic = camera_extrinsic_cv_from_mujoco(
                env.data,
                camera_id,
            )
            camera_offset = fixed_camera_offset(
                args.perturbation,
                severity,
                args.seed + episode,
            )
            set_fixed_camera_pose_offset(
                env.model,
                env.data,
                camera_id,
                base_position,
                base_quaternion,
                *camera_offset,
            )
            observer_histories: dict[
                tuple[str, int],
                list[tuple[np.ndarray, np.ndarray, np.ndarray]],
            ] = {}
            for frame in range(args.steps_per_episode):
                true_extrinsic = camera_extrinsic_cv_from_mujoco(
                    env.data,
                    camera_id,
                )
                controls = get_site_keypoints(
                    env.model,
                    env.data,
                    DEFAULT_KEYPOINT_SITES,
                )
                control_names = controls.names
                true_camera = Camera(
                    intrinsic,
                    true_extrinsic,
                    args.width,
                    args.height,
                )
                nominal_camera = Camera(
                    intrinsic,
                    nominal_extrinsic,
                    args.width,
                    args.height,
                )
                control_true, control_true_valid = project_points(
                    controls.xyz,
                    true_camera,
                )
                _, control_nominal_valid = project_points(
                    controls.xyz,
                    nominal_camera,
                )
                control_valid = control_true_valid & control_nominal_valid
                if bool(control_valid.any()):
                    for constellation in args.constellations:
                        observers = get_link_observer_points(
                            env.model,
                            env.data,
                            CONSTELLATIONS[constellation],
                        )
                        manifests[constellation] = observers.names
                        observer_true, observer_true_valid = project_points(
                            observers.xyz,
                            true_camera,
                        )
                        _, observer_nominal_valid = project_points(
                            observers.xyz,
                            nominal_camera,
                        )
                        base_valid = observer_true_valid & observer_nominal_valid
                        for trial in range(args.trials_per_frame):
                            trial_seed = (
                                args.seed
                                + int(round(float(severity) * 1000)) * 1_000_003
                                + episode * 10_007
                                + frame * 101
                                + trial
                            )
                            measured, observer_valid = (
                                corrupt_observer_correspondences(
                                    observer_true,
                                    base_valid,
                                    args.observer_noise_std_px,
                                    args.observer_dropout_prob,
                                    np.random.default_rng(trial_seed),
                                )
                            )
                            history_key = (constellation, trial)
                            observer_xyz_window, measured_window, valid_window = (
                                append_observer_history(
                                    observer_histories.setdefault(history_key, []),
                                    observers.xyz,
                                    measured,
                                    observer_valid,
                                    args.history_frames,
                                )
                            )
                            for solver in args.solvers:
                                row = evaluate_solver(
                                    solver,
                                    intrinsic,
                                    nominal_extrinsic,
                                    observer_xyz_window,
                                    measured_window,
                                    valid_window,
                                    controls.xyz,
                                    control_true,
                                    control_valid,
                                    prior_std,
                                    args.observation_std_px,
                                    args.width,
                                    args.height,
                                    args.min_whitened_singular_value,
                                    args.relative_singular_value,
                                    args.min_control_variance_reduction_fraction,
                                    args.min_innovation_z_score,
                                    args.min_global_innovation_rms_z,
                                    args.innovation_false_alarm_prob,
                                )
                                rows.setdefault(
                                    (float(severity), constellation, solver),
                                    [],
                                ).append(row)
                action = choose_action(
                    env.action_space,
                    np.random.default_rng(env_seed + frame),
                    obs,
                    policy=policy,
                )
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            if hasattr(env, "close"):
                env.close()

    summaries = [
        aggregate_rows(group, args.perturbation, severity, constellation, solver)
        for (severity, constellation, solver), group in sorted(rows.items())
    ]
    payload = {
        "schema_version": 1,
        "method": "kineparity_oracle_observer_stress",
        "task": args.task,
        "camera": args.camera,
        "observer_manifests": manifests,
        "control_names": control_names,
        "control_visual_correspondences_used": False,
        "control_true_pixels_used_for_solver": False,
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key != "output"
        },
        "summary": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
