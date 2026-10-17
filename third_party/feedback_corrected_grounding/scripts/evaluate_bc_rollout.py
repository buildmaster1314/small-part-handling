#!/usr/bin/env python3
"""Roll out a minimal BC policy in MetaWorld.

This is a Stage-4 smoke evaluator for the lightweight BC checkpoints produced
by `scripts/train_bc_policy.py`. It is not a final Diffusion Policy evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.geometry import Camera, perturb_extrinsic, project_points  # noqa: E402
from fcpg.observations import OBSERVATION_MODES, select_policy_observation  # noqa: E402
from fcpg.metaworld_utils import DEFAULT_KEYPOINT_SITES, get_site_keypoints, make_env, parse_direction_list  # noqa: E402
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    sample_camera_pose_offset,
    set_fixed_camera_pose_offset,
)
from scripts.train_bc_policy import MinimalBCPolicy, choose_device, load_residual_checkpoint  # noqa: E402


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def choose_perturbation(
    rng: np.random.Generator,
    translations_cm: list[float],
    directions: dict[str, np.ndarray],
    yaws_deg: list[float],
) -> tuple[float, str, np.ndarray, float]:
    trans_cm = float(rng.choice(translations_cm))
    direction_name = str(rng.choice(list(directions.keys())))
    yaw = float(rng.choice(yaws_deg))
    translation_m = (trans_cm / 100.0) * directions[direction_name]
    return trans_cm, direction_name, translation_m, yaw


def make_residual_vectors(obs: np.ndarray, keypoints_xyz: np.ndarray, q_nominal: np.ndarray, width: int, height: int) -> np.ndarray:
    num_keypoints = int(q_nominal.shape[0])
    q_norm = np.stack([q_nominal[:, 0] / width, q_nominal[:, 1] / height], axis=-1).astype(np.float32)
    onehot = np.eye(num_keypoints, dtype=np.float32)
    obs_rep = np.repeat(obs.astype(np.float32)[None, :], num_keypoints, axis=0)
    return np.concatenate([q_norm, keypoints_xyz.astype(np.float32), onehot, obs_rep], axis=-1)


def make_proprio(keypoints_xyz: np.ndarray) -> np.ndarray:
    num_keypoints = int(keypoints_xyz.shape[0])
    return np.concatenate([keypoints_xyz.astype(np.float32), np.eye(num_keypoints, dtype=np.float32)], axis=-1)


def make_batch(
    obs: np.ndarray,
    image: np.ndarray,
    keypoints_xyz: np.ndarray,
    q_true: np.ndarray,
    q_nominal: np.ndarray,
    device: torch.device,
    obs_mode: str,
) -> dict[str, torch.Tensor]:
    policy_obs = select_policy_observation(obs, obs_mode)
    batch = {
        "rgb": torch.from_numpy(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0),
        "obs": torch.from_numpy(policy_obs).unsqueeze(0),
        "q_nominal": torch.from_numpy(q_nominal.astype(np.float32)).unsqueeze(0),
        "q_true": torch.from_numpy(q_true.astype(np.float32)).unsqueeze(0),
        "proprio": torch.from_numpy(make_proprio(keypoints_xyz)).unsqueeze(0),
        "residual_vector": torch.from_numpy(make_residual_vectors(policy_obs, keypoints_xyz, q_nominal, image.shape[1], image.shape[0])).unsqueeze(0),
    }
    return {key: value.to(device) for key, value in batch.items()}


def load_policy_checkpoint(path: str | Path, device: torch.device) -> tuple[MinimalBCPolicy, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    model_type = str(ckpt_args.get("model"))
    residual_model = None
    if model_type == "fcpg":
        residual_path = ckpt_args.get("residual_checkpoint", "")
        if not residual_path:
            raise ValueError("FCPG BC checkpoint is missing residual_checkpoint in args")
        residual_model, _, _ = load_residual_checkpoint(residual_path, device)
    model = MinimalBCPolicy(
        model_type=model_type,
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        num_keypoints=int(checkpoint["num_keypoints"]),
        proprio_dim=int(checkpoint["proprio_dim"]),
        token_dim=int(ckpt_args.get("token_dim", 32)),
        hidden_dim=int(ckpt_args.get("hidden_dim", 128)),
        residual_model=residual_model,
        image_size=(int(ckpt_args.get("width", 128)), int(ckpt_args.get("height", 128))),
        freeze_residual=not bool(ckpt_args.get("train_residual", False)),
        policy_arch=str(ckpt_args.get("policy_arch", "mlp")),
        image_patch_size=int(ckpt_args.get("image_patch_size", 16)),
        transformer_layers=int(ckpt_args.get("transformer_layers", 4)),
        transformer_heads=int(ckpt_args.get("transformer_heads", 4)),
        transformer_mlp_ratio=int(ckpt_args.get("transformer_mlp_ratio", 4)),
        transformer_dropout=float(ckpt_args.get("transformer_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, ckpt_args


def rollout_once(
    model: MinimalBCPolicy,
    task_name: str,
    camera_name: str,
    width: int,
    height: int,
    horizon: int,
    rng: np.random.Generator,
    translations_cm: list[float],
    directions: dict[str, np.ndarray],
    yaws_deg: list[float],
    actual_camera_translation_cm_min: float,
    actual_camera_translation_cm_max: float,
    actual_camera_rotation_deg_min: float,
    actual_camera_rotation_deg_max: float,
    actual_camera_clean_probability: float,
    device: torch.device,
    obs_mode: str,
    task_index: int,
    env_seed: int,
) -> dict[str, object]:
    random.seed(env_seed)
    np.random.seed(env_seed)
    env = make_env(task_name, camera_name, width, height, task_index=task_index)
    obs, info = env.reset(seed=env_seed)
    camera_id = get_camera_id(env.model, camera_name)
    intrinsic = camera_intrinsic_from_fovy(width, height, float(env.model.cam_fovy[camera_id]))
    base_camera_position = np.asarray(env.model.cam_pos[camera_id], dtype=np.float64).copy()
    base_camera_quaternion = np.asarray(env.model.cam_quat[camera_id], dtype=np.float64).copy()
    base_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
    actual_camera_offset = sample_camera_pose_offset(
        rng,
        actual_camera_translation_cm_max,
        actual_camera_rotation_deg_max,
        actual_camera_clean_probability,
        translation_cm_min=actual_camera_translation_cm_min,
        rotation_deg_min=actual_camera_rotation_deg_min,
    )
    set_fixed_camera_pose_offset(
        env.model,
        env.data,
        camera_id,
        base_camera_position,
        base_camera_quaternion,
        *actual_camera_offset,
    )
    actual_translation_cm = float(np.linalg.norm(actual_camera_offset[0]) * 100.0)
    actual_rotation_deg = float(actual_camera_offset[2])
    successes = []
    rewards = []
    actions = []
    pixel_errors = []
    corrected_pixel_errors = []
    for step in range(horizon):
        image = env.render()
        true_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
        true_camera = Camera(intrinsic=intrinsic, extrinsic_cam_from_world=true_extrinsic, width=width, height=height)
        keypoint_state = get_site_keypoints(env.model, env.data, DEFAULT_KEYPOINT_SITES)
        q_true, _ = project_points(keypoint_state.xyz, true_camera)
        trans_cm, direction_name, translation_m, yaw = choose_perturbation(rng, translations_cm, directions, yaws_deg)
        nominal_extrinsic = perturb_extrinsic(base_extrinsic, translation_xyz=translation_m, yaw_degrees=yaw)
        nominal_camera = Camera(intrinsic=intrinsic, extrinsic_cam_from_world=nominal_extrinsic, width=width, height=height)
        q_nominal, _ = project_points(keypoint_state.xyz, nominal_camera)
        pixel_errors.append(float(np.linalg.norm(q_nominal - q_true, axis=1).mean()))
        batch = make_batch(obs, image, keypoint_state.xyz, q_true, q_nominal, device, obs_mode)
        with torch.no_grad():
            if model.model_type == "state":
                q_corrected = batch["q_nominal"]
            else:
                q_corrected = model.corrected_points(batch)
            corrected_err = torch.linalg.norm(q_corrected - batch["q_true"], dim=-1).mean()
            corrected_pixel_errors.append(float(corrected_err.detach().cpu().item()))
            action = model(batch).squeeze(0).detach().cpu().numpy().astype(np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        successes.append(float(info.get("success", 0.0)) if isinstance(info, dict) else 0.0)
        if terminated or truncated:
            break
    return {
        "task": task_name,
        "task_index": task_index,
        "env_seed": env_seed,
        "steps": len(rewards),
        "success": float(np.max(successes) if successes else 0.0),
        "final_success": float(successes[-1] if successes else 0.0),
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards) if rewards else 0.0),
        "action_abs_mean": float(np.mean(np.abs(np.asarray(actions))) if actions else 0.0),
        "pixel_error_mean": float(np.mean(pixel_errors) if pixel_errors else 0.0),
        "corrected_pixel_error_mean": float(np.mean(corrected_pixel_errors) if corrected_pixel_errors else 0.0),
        "actual_camera_translation_cm": actual_translation_cm,
        "actual_camera_translation_x_cm": float(actual_camera_offset[0][0] * 100.0),
        "actual_camera_translation_y_cm": float(actual_camera_offset[0][1] * 100.0),
        "actual_camera_translation_z_cm": float(actual_camera_offset[0][2] * 100.0),
        "actual_camera_rotation_axis_x": float(actual_camera_offset[1][0]),
        "actual_camera_rotation_axis_y": float(actual_camera_offset[1][1]),
        "actual_camera_rotation_axis_z": float(actual_camera_offset[1][2]),
        "actual_camera_rotation_deg": actual_rotation_deg,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="reach-v3")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--task-index-offset",
        type=int,
        default=0,
        help="Add this offset to episode indices when selecting MetaWorld task instances.",
    )
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--translations-cm", default="0")
    parser.add_argument("--directions", default="x+")
    parser.add_argument("--yaws-deg", default="0")
    parser.add_argument(
        "--actual-camera-translation-cm-min",
        type=float,
        default=0.0,
        help="Minimum rendered-camera translation magnitude; set equal to max for a fixed severity.",
    )
    parser.add_argument(
        "--actual-camera-translation-cm-max",
        type=float,
        default=0.0,
        help="Maximum continuous world-frame translation of the rendered camera, sampled once per episode.",
    )
    parser.add_argument(
        "--actual-camera-rotation-deg-min",
        type=float,
        default=0.0,
        help="Minimum absolute rendered-camera rotation; set equal to max for a fixed severity.",
    )
    parser.add_argument(
        "--actual-camera-rotation-deg-max",
        type=float,
        default=0.0,
        help="Maximum continuous axis-angle rotation of the rendered camera, sampled once per episode.",
    )
    parser.add_argument(
        "--actual-camera-clean-probability",
        type=float,
        default=0.0,
        help="Probability that an episode keeps the rendered camera at the base pose.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for actual-camera pose sampling.")
    parser.add_argument("--env-seed-base", type=int, default=1000, help="Episode environment seed is base + episode.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--oracle-grounding",
        action="store_true",
        help="Legacy shortcut for --grounding-override oracle on a GeoProp checkpoint.",
    )
    parser.add_argument(
        "--grounding-override",
        choices=("native", "nominal", "oracle"),
        default="native",
        help="Replace grounding points at test time while keeping all learned policy weights unchanged.",
    )
    parser.add_argument("--output-dir", default="outputs/bc_rollout")
    args = parser.parse_args()
    if not 0.0 <= args.actual_camera_translation_cm_min <= args.actual_camera_translation_cm_max:
        parser.error("actual camera translation requires 0 <= min <= max")
    if not 0.0 <= args.actual_camera_rotation_deg_min <= args.actual_camera_rotation_deg_max:
        parser.error("actual camera rotation requires 0 <= min <= max")
    if not 0.0 <= args.actual_camera_clean_probability <= 1.0:
        parser.error("--actual-camera-clean-probability must be in [0, 1]")

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = choose_device(args.device)
    rng = np.random.default_rng(args.seed)
    translations_cm = parse_csv_floats(args.translations_cm)
    directions = parse_direction_list(args.directions)
    yaws_deg = parse_csv_floats(args.yaws_deg)
    model, ckpt_args = load_policy_checkpoint(args.checkpoint, device)
    obs_mode = str(ckpt_args.get("obs_mode", "full"))
    if obs_mode not in OBSERVATION_MODES:
        parser.error(f"checkpoint has unsupported obs_mode={obs_mode!r}")
    source_model_type = str(ckpt_args.get("model"))
    grounding_mode = args.grounding_override
    if args.oracle_grounding:
        if source_model_type != "geoprop":
            parser.error("--oracle-grounding requires a GeoProp checkpoint")
        if grounding_mode != "native":
            parser.error("do not combine --oracle-grounding with --grounding-override")
        grounding_mode = "oracle"
    if grounding_mode != "native" and source_model_type == "state":
        parser.error("grounding overrides require a grounded policy checkpoint")
    if grounding_mode == "nominal":
        model.model_type = "geoprop"
    elif grounding_mode == "oracle":
        model.model_type = "oracle"
    if args.oracle_grounding:
        model_type = "oracle"
    elif grounding_mode == "native":
        model_type = source_model_type
    else:
        model_type = f"{source_model_type}_{grounding_mode}"
    rows = []
    for episode in range(args.episodes):
        result = rollout_once(
            model=model,
            task_name=args.task,
            camera_name=args.camera,
            width=args.width,
            height=args.height,
            horizon=args.horizon,
            rng=rng,
            translations_cm=translations_cm,
            directions=directions,
            yaws_deg=yaws_deg,
            actual_camera_translation_cm_min=args.actual_camera_translation_cm_min,
            actual_camera_translation_cm_max=args.actual_camera_translation_cm_max,
            actual_camera_rotation_deg_min=args.actual_camera_rotation_deg_min,
            actual_camera_rotation_deg_max=args.actual_camera_rotation_deg_max,
            actual_camera_clean_probability=args.actual_camera_clean_probability,
            device=device,
            obs_mode=obs_mode,
            task_index=args.task_index_offset + episode,
            env_seed=args.env_seed_base + episode,
        )
        result = {"episode": episode, "model": model_type, **result}
        rows.append(result)
        print(
            f"episode {episode:03d} model={model_type} success={result['success']:.1f} "
            f"reward_sum={result['reward_sum']:.3f} pixel_err={result['pixel_error_mean']:.3f} "
            f"corr_pixel_err={result['corrected_pixel_error_mean']:.3f}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{model_type}_rollout.csv"
    json_path = output_dir / f"{model_type}_rollout_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": args.checkpoint,
        "source_model": source_model_type,
        "model": model_type,
        "obs_mode": obs_mode,
        "grounding_mode": grounding_mode,
        "task": args.task,
        "episodes": args.episodes,
        "task_index_offset": args.task_index_offset,
        "env_seed_base": args.env_seed_base,
        "camera_seed": args.seed,
        "horizon": args.horizon,
        "translations_cm": args.translations_cm,
        "directions": args.directions,
        "yaws_deg": args.yaws_deg,
        "actual_camera_translation_cm_min": args.actual_camera_translation_cm_min,
        "actual_camera_translation_cm_max": args.actual_camera_translation_cm_max,
        "actual_camera_rotation_deg_min": args.actual_camera_rotation_deg_min,
        "actual_camera_rotation_deg_max": args.actual_camera_rotation_deg_max,
        "actual_camera_clean_probability": args.actual_camera_clean_probability,
        "actual_camera_translation_cm_mean": float(np.mean([row["actual_camera_translation_cm"] for row in rows])),
        "actual_camera_rotation_deg_abs_mean": float(np.mean([abs(row["actual_camera_rotation_deg"]) for row in rows])),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "reward_sum_mean": float(np.mean([row["reward_sum"] for row in rows])),
        "reward_sum_std": float(np.std([row["reward_sum"] for row in rows])),
        "pixel_error_mean": float(np.mean([row["pixel_error_mean"] for row in rows])),
        "corrected_pixel_error_mean": float(np.mean([row["corrected_pixel_error_mean"] for row in rows])),
    }
    json_path.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2), encoding="utf-8")
    print(f"csv: {csv_path}")
    print(f"summary: {json_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
