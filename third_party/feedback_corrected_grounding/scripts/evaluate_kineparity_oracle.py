#!/usr/bin/env python3
"""Evaluate oracle observer-to-control transfer in MetaWorld."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.geometry import Camera, project_points
from fcpg.metaworld_utils import (
    DEFAULT_KEYPOINT_SITES,
    DEFAULT_OBSERVER_BODY_NAMES,
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
    sample_camera_pose_offset,
    set_fixed_camera_pose_offset,
)
from fcpg.structured_grounding import solve_observer_control_transfer


def parse_severity(value: str) -> list[float]:
    severities = sorted({float(item) for item in value.split(",") if item.strip()})
    if not severities or severities[0] < 0:
        raise argparse.ArgumentTypeError("severity values must be non-negative")
    return severities


def fixed_camera_offset(
    perturbation: str,
    severity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    if perturbation == "rotation":
        return sample_camera_pose_offset(
            rng,
            translation_cm_max=0.0,
            rotation_deg_max=severity,
            rotation_deg_min=severity,
            clean_probability=0.0,
        )
    return sample_camera_pose_offset(
        rng,
        translation_cm_max=severity,
        translation_cm_min=severity,
        rotation_deg_max=0.0,
        clean_probability=0.0,
    )


def tensor_batch(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float64)).unsqueeze(0)


def evaluate_frame(
    intrinsic: np.ndarray,
    nominal_extrinsic: np.ndarray,
    true_extrinsic: np.ndarray,
    observer_xyz: np.ndarray,
    control_xyz: np.ndarray,
    prior_std: torch.Tensor,
    observation_std_px: float,
    width: int,
    height: int,
    min_whitened_singular_value: float,
    relative_singular_value: float,
    min_control_variance_reduction_fraction: float,
) -> dict[str, object] | None:
    true_camera = Camera(intrinsic, true_extrinsic, width, height)
    nominal_camera = Camera(intrinsic, nominal_extrinsic, width, height)
    observer_true, observer_true_valid = project_points(observer_xyz, true_camera)
    _, observer_nominal_valid = project_points(observer_xyz, nominal_camera)
    control_true, control_true_valid = project_points(control_xyz, true_camera)
    _, control_nominal_valid = project_points(control_xyz, nominal_camera)
    observer_valid = observer_true_valid & observer_nominal_valid
    control_valid = control_true_valid & control_nominal_valid
    if not bool(control_valid.any()):
        return None

    output = solve_observer_control_transfer(
        tensor_batch(intrinsic),
        tensor_batch(nominal_extrinsic),
        tensor_batch(observer_xyz),
        tensor_batch(observer_true),
        tensor_batch(control_xyz),
        prior_std=prior_std,
        observation_std_px=observation_std_px,
        observer_valid=torch.from_numpy(observer_valid).unsqueeze(0),
        min_whitened_singular_value=min_whitened_singular_value,
        relative_singular_value=relative_singular_value,
        min_control_variance_reduction_fraction=(
            min_control_variance_reduction_fraction
        ),
    )
    control_true_tensor = tensor_batch(control_true)
    valid_tensor = torch.from_numpy(control_valid).unsqueeze(0)
    before = torch.linalg.vector_norm(
        output.control_nominal_points_px - control_true_tensor,
        dim=-1,
    )[valid_tensor]
    after = torch.linalg.vector_norm(
        output.control_corrected_points_px - control_true_tensor,
        dim=-1,
    )[valid_tensor]
    displacement = torch.linalg.vector_norm(
        output.control_corrected_points_px - output.control_nominal_points_px,
        dim=-1,
    )[valid_tensor]
    update = output.update
    return {
        "before": before.cpu().numpy(),
        "after": after.cpu().numpy(),
        "displacement": displacement.cpu().numpy(),
        "valid_observer_count": int(update.valid_observer_count.item()),
        "effective_rank": int(update.effective_rank.item()),
        "singular_values": update.singular_values[0].cpu().numpy(),
        "retained_modes": update.retained_modes[0].cpu().numpy(),
        "prior_control_variance": float(update.prior_control_variance.item()),
        "posterior_control_variance": float(
            update.posterior_control_variance.item()
        ),
        "twist": update.twist[0].cpu().numpy(),
    }


def aggregate_rows(
    rows: list[dict[str, object]],
    perturbation: str,
    severity: float,
) -> dict[str, object]:
    if not rows:
        raise RuntimeError(f"no valid rows for {perturbation} severity={severity}")
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
    singular_values = np.stack(
        [np.asarray(row["singular_values"]) for row in rows]
    )
    retained = np.stack([np.asarray(row["retained_modes"]) for row in rows])
    prior_variance = np.asarray(
        [row["prior_control_variance"] for row in rows],
        dtype=np.float64,
    )
    posterior_variance = np.asarray(
        [row["posterior_control_variance"] for row in rows],
        dtype=np.float64,
    )
    before_mean = float(before.mean())
    after_mean = float(after.mean())
    ratio = after_mean / before_mean if before_mean > 1e-9 else None
    summary = {
        "perturbation": perturbation,
        "severity": severity,
        "frames": len(rows),
        "control_anchor_observations": int(after.size),
        "before_error_mean_px": before_mean,
        "before_error_median_px": float(np.median(before)),
        "before_error_p95_px": float(np.percentile(before, 95)),
        "after_error_mean_px": after_mean,
        "after_error_median_px": float(np.median(after)),
        "after_error_p95_px": float(np.percentile(after, 95)),
        "after_error_max_px": float(after.max()),
        "error_ratio": ratio,
        "control_displacement_mean_px": float(displacement.mean()),
        "control_displacement_p95_px": float(np.percentile(displacement, 95)),
        "valid_observer_count_mean": float(valid_counts.mean()),
        "effective_rank_mean": float(ranks.mean()),
        "effective_rank_min": int(ranks.min()),
        "effective_rank_max": int(ranks.max()),
        "singular_values_mean": singular_values.mean(axis=0).tolist(),
        "retained_mode_fraction": retained.mean(axis=0).tolist(),
        "prior_control_variance_mean": float(prior_variance.mean()),
        "posterior_control_variance_mean": float(posterior_variance.mean()),
        "posterior_to_prior_variance_ratio": float(
            np.mean(posterior_variance / np.maximum(prior_variance, 1e-12))
        ),
    }
    if severity == 0:
        summary["gate_clean_displacement_le_0_5px"] = (
            summary["control_displacement_mean_px"] <= 0.5
        )
    elif perturbation == "rotation" and severity == 10:
        summary["gate_ratio_le_0_25"] = ratio is not None and ratio <= 0.25
        summary["gate_p95_le_12px"] = summary["after_error_p95_px"] <= 12.0
    elif perturbation == "translation" and severity == 4:
        summary["gate_ratio_le_0_30"] = ratio is not None and ratio <= 0.30
    return summary


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
    parser.add_argument("--severities", type=parse_severity, default="0,5,10")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--task-index-offset", type=int, default=2000)
    parser.add_argument("--env-seed-base", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--prior-translation-std-cm", type=float, default=8.0)
    parser.add_argument("--prior-rotation-std-deg", type=float, default=15.0)
    parser.add_argument("--observation-std-px", type=float, default=0.5)
    parser.add_argument(
        "--min-whitened-singular-value",
        type=float,
        default=0.25,
    )
    parser.add_argument("--relative-singular-value", type=float, default=1e-3)
    parser.add_argument(
        "--min-control-variance-reduction-fraction",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--output",
        default="outputs/kineparity/oracle_geometry_rotation.json",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.steps_per_episode < 1:
        parser.error("episodes and steps-per-episode must be positive")
    if args.warmup_steps < 0:
        parser.error("warmup-steps must be non-negative")

    os.environ.setdefault("MUJOCO_GL", "egl")
    policy = make_scripted_policy(args.task)
    prior_std = torch.tensor(
        [
            args.prior_translation_std_cm / 100.0,
            args.prior_translation_std_cm / 100.0,
            args.prior_translation_std_cm / 100.0,
            np.deg2rad(args.prior_rotation_std_deg),
            np.deg2rad(args.prior_rotation_std_deg),
            np.deg2rad(args.prior_rotation_std_deg),
        ],
        dtype=torch.float64,
    )
    summaries = []
    observer_names = None
    control_names = None
    for severity in args.severities:
        rows = []
        for episode in range(args.episodes):
            task_index = args.task_index_offset + episode
            env_seed = args.env_seed_base + episode
            env = make_env(
                args.task,
                args.camera,
                args.width,
                args.height,
                task_index=task_index,
            )
            obs, _ = env.reset(seed=env_seed)
            camera_id = get_camera_id(env.model, args.camera)
            intrinsic = camera_intrinsic_from_fovy(
                args.width,
                args.height,
                float(env.model.cam_fovy[camera_id]),
            )
            base_position = np.asarray(
                env.model.cam_pos[camera_id],
                dtype=np.float64,
            ).copy()
            base_quaternion = np.asarray(
                env.model.cam_quat[camera_id],
                dtype=np.float64,
            ).copy()
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
            for frame in range(args.warmup_steps + args.steps_per_episode):
                if frame >= args.warmup_steps:
                    observers = get_link_observer_points(env.model, env.data)
                    controls = get_site_keypoints(
                        env.model,
                        env.data,
                        DEFAULT_KEYPOINT_SITES,
                    )
                    if observer_names is None:
                        observer_names = observers.names
                        control_names = controls.names
                    elif (
                        observer_names != observers.names
                        or control_names != controls.names
                    ):
                        raise RuntimeError("observer/control point manifest changed")
                    row = evaluate_frame(
                        intrinsic,
                        nominal_extrinsic,
                        camera_extrinsic_cv_from_mujoco(env.data, camera_id),
                        observers.xyz,
                        controls.xyz,
                        prior_std,
                        args.observation_std_px,
                        args.width,
                        args.height,
                        args.min_whitened_singular_value,
                        args.relative_singular_value,
                        args.min_control_variance_reduction_fraction,
                    )
                    if row is not None:
                        rows.append(row)
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
        summaries.append(
            aggregate_rows(rows, args.perturbation, float(severity))
        )

    payload = {
        "schema_version": 1,
        "method": "kineparity_oracle_observer_global_only",
        "task": args.task,
        "camera": args.camera,
        "image_size": [args.width, args.height],
        "observer_names": observer_names,
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

