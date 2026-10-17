#!/usr/bin/env python3
"""Roll out a deterministic action-chunk transformer in MetaWorld."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.geometric_baselines import (  # noqa: E402
    OnlinePnPCalibrator,
    refine_points_with_prototypes,
)
from fcpg.geometry import Camera, project_points  # noqa: E402
from fcpg.metaworld_utils import DEFAULT_KEYPOINT_SITES, get_site_keypoints, make_env  # noqa: E402
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    sample_camera_pose_offset,
    set_fixed_camera_pose_offset,
)
from fcpg.observations import (  # noqa: E402
    apply_keypoint_square_occlusion,
    select_policy_observation,
)


@dataclass(frozen=True)
class PrototypeBank:
    descriptors: torch.Tensor
    feature_level: int
    metadata: dict[str, object]
    sha256: str


class OnlineVisualGroundingBaseline:
    """Apply image-only local refinement, optionally followed by shared-pose PnP."""

    def __init__(
        self,
        mode: str,
        model,
        prototype_bank: PrototypeBank,
        image_size: tuple[int, int],
        intrinsic: np.ndarray,
        nominal_extrinsic: np.ndarray,
        device: torch.device,
        radius_px: float,
        grid_size: int,
        temperature: float,
        min_confidence: float,
        pnp_history_frames: int,
        pnp_max_reprojection_error_px: float,
        pnp_max_pose_rotation_deg: float,
        pnp_max_pose_translation_cm: float,
    ) -> None:
        if mode not in {"prototype", "pnp"}:
            raise ValueError(f"unsupported external grounding mode {mode!r}")
        self.mode = mode
        self.model = model
        self.prototype_bank = prototype_bank
        self.image_size = image_size
        self.device = device
        self.radius_px = float(radius_px)
        self.grid_size = int(grid_size)
        self.temperature = float(temperature)
        self.min_confidence = float(min_confidence)
        self.pnp = (
            OnlinePnPCalibrator(
                intrinsic,
                nominal_extrinsic,
                image_size=image_size,
                history_frames=pnp_history_frames,
                max_reprojection_error_px=pnp_max_reprojection_error_px,
                max_pose_rotation_deg=pnp_max_pose_rotation_deg,
                max_pose_translation_cm=pnp_max_pose_translation_cm,
            )
            if mode == "pnp"
            else None
        )

    @torch.inference_mode()
    def enrich(self, record: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rgb = torch.from_numpy(record["rgb"]).unsqueeze(0).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            feature_maps, _ = self.model.vision_encoder.extract_backbone_features(rgb)
        level = self.prototype_bank.feature_level % feature_maps.shape[1]
        q_nominal = torch.from_numpy(record["q_nominal"]).unsqueeze(0).to(self.device)
        refinement = refine_points_with_prototypes(
            feature_maps[:, level],
            q_nominal,
            self.prototype_bank.descriptors,
            image_size=self.image_size,
            radius_px=self.radius_px,
            grid_size=self.grid_size,
            temperature=self.temperature,
            min_confidence=self.min_confidence,
        )
        matched_points = refinement.points_px[0].float().cpu().numpy()
        confidence = refinement.confidence[0].float().cpu().numpy()
        valid = refinement.valid[0].cpu().numpy()
        corrected = matched_points
        pnp_solved = False
        pnp_error = float("nan")
        pnp_correspondences = 0
        if self.pnp is not None:
            update = self.pnp.update(record["keypoints_xyz"], matched_points, valid)
            corrected = update.points_px
            pnp_solved = update.solved
            pnp_error = update.reprojection_error_px
            pnp_correspondences = update.correspondence_count

        enriched = dict(record)
        enriched["fcpg_q_corrected"] = corrected.astype(np.float32)
        enriched["fcpg_reliability"] = np.ones(corrected.shape[0], dtype=np.float32)
        enriched["external_match_confidence"] = confidence.astype(np.float32)
        enriched["external_match_valid"] = valid.astype(np.float32)
        enriched["external_pnp_solved"] = np.asarray(float(pnp_solved), dtype=np.float32)
        enriched["external_pnp_error"] = np.asarray(pnp_error, dtype=np.float32)
        enriched["external_pnp_correspondences"] = np.asarray(
            pnp_correspondences,
            dtype=np.float32,
        )
        return enriched


def load_prototype_bank(
    directory: str | Path,
    model,
    image_size: tuple[int, int],
    device: torch.device,
) -> PrototypeBank:
    directory = Path(directory)
    descriptors_path = directory / "prototypes.npy"
    metadata_path = directory / "metadata.json"
    if not descriptors_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"invalid prototype bank directory: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_size = (int(metadata["image_width"]), int(metadata["image_height"]))
    if expected_size != image_size:
        raise ValueError(
            f"prototype bank image size {expected_size} does not match rollout {image_size}"
        )
    expected_model = str(metadata["model_name"])
    if expected_model != model.vision_model_name:
        raise ValueError(
            f"prototype model {expected_model!r} does not match policy {model.vision_model_name!r}"
        )
    bank_layers = tuple(int(value) for value in metadata["selected_layers"])
    if bank_layers != model.vision_selected_layers:
        raise ValueError(
            f"prototype layers {bank_layers} do not match policy {model.vision_selected_layers}"
        )
    descriptors = torch.from_numpy(np.load(descriptors_path)).to(device)
    if descriptors.shape[0] != model.num_keypoints:
        raise ValueError(
            f"prototype keypoints {descriptors.shape[0]} do not match policy {model.num_keypoints}"
        )
    digest = hashlib.sha256(descriptors_path.read_bytes()).hexdigest()
    return PrototypeBank(
        descriptors=descriptors,
        feature_level=int(metadata["feature_level"]),
        metadata=metadata,
        sha256=digest,
    )


def collect_observation(
    env,
    obs: np.ndarray,
    camera_id: int,
    intrinsic: np.ndarray,
    nominal_extrinsic: np.ndarray,
    width: int,
    height: int,
    obs_mode: str,
    keypoint_occlusion_radius_px: int,
    occlusion_value: int,
) -> dict[str, np.ndarray]:
    image = env.render().astype(np.uint8)
    true_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
    true_camera = Camera(intrinsic, true_extrinsic, width, height)
    nominal_camera = Camera(intrinsic, nominal_extrinsic, width, height)
    keypoints = get_site_keypoints(env.model, env.data, DEFAULT_KEYPOINT_SITES).xyz
    q_true, _ = project_points(keypoints, true_camera)
    q_nominal, _ = project_points(keypoints, nominal_camera)
    image = apply_keypoint_square_occlusion(
        image,
        q_true,
        radius_px=keypoint_occlusion_radius_px,
        value=occlusion_value,
    )
    return {
        "rgb": np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)),
        "obs": select_policy_observation(obs, obs_mode),
        "keypoints_xyz": keypoints.astype(np.float32),
        "q_true": q_true.astype(np.float32),
        "q_nominal": q_nominal.astype(np.float32),
    }


def history_batch(
    history: deque[dict[str, np.ndarray]],
    obs_horizon: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    records = list(history)
    if not records:
        raise ValueError("history must not be empty")
    records = [records[0]] * max(0, obs_horizon - len(records)) + records[-obs_horizon:]
    keys = ["rgb", "obs", "keypoints_xyz", "q_nominal", "q_true"]
    for key in ("fcpg_q_corrected", "fcpg_reliability"):
        present = [key in record for record in records]
        if any(present) and not all(present):
            raise ValueError(f"optional history field {key!r} must be present in every record")
        if all(present):
            keys.append(key)
    return {
        key: torch.from_numpy(np.stack([record[key] for record in records], axis=0))
        .unsqueeze(0)
        .to(device)
        for key in keys
    }


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def optional_mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def rollout_once(
    model,
    checkpoint_args: dict[str, object],
    task_name: str,
    camera_name: str,
    width: int,
    height: int,
    horizon: int,
    action_steps: int,
    rng: np.random.Generator,
    translation_cm_min: float,
    translation_cm_max: float,
    rotation_deg_min: float,
    rotation_deg_max: float,
    clean_probability: float,
    device: torch.device,
    task_index: int,
    env_seed: int,
    keypoint_occlusion_radius_px: int,
    occlusion_value: int,
    external_grounding: str,
    prototype_bank: PrototypeBank | None,
    prototype_radius_px: float,
    prototype_grid_size: int,
    prototype_temperature: float,
    prototype_min_confidence: float,
    pnp_history_frames: int,
    pnp_max_reprojection_error_px: float,
    pnp_max_pose_rotation_deg: float,
    pnp_max_pose_translation_cm: float,
) -> dict[str, object]:
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)
    env = make_env(task_name, camera_name, width, height, task_index=task_index)
    obs, _ = env.reset(seed=env_seed)
    camera_id = get_camera_id(env.model, camera_name)
    intrinsic = camera_intrinsic_from_fovy(width, height, float(env.model.cam_fovy[camera_id]))
    base_position = np.asarray(env.model.cam_pos[camera_id], dtype=np.float64).copy()
    base_quaternion = np.asarray(env.model.cam_quat[camera_id], dtype=np.float64).copy()
    base_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
    camera_offset = sample_camera_pose_offset(
        rng,
        translation_cm_max,
        rotation_deg_max,
        clean_probability,
        translation_cm_min=translation_cm_min,
        rotation_deg_min=rotation_deg_min,
    )
    set_fixed_camera_pose_offset(
        env.model,
        env.data,
        camera_id,
        base_position,
        base_quaternion,
        *camera_offset,
    )
    obs_mode = str(checkpoint_args.get("obs_mode", "robot"))
    obs_horizon = int(checkpoint_args.get("obs_horizon", 2))
    external_runtime = None
    if external_grounding != "none":
        if prototype_bank is None:
            raise ValueError("external grounding requires a prototype bank")
        external_runtime = OnlineVisualGroundingBaseline(
            external_grounding,
            model,
            prototype_bank,
            image_size=(width, height),
            intrinsic=intrinsic,
            nominal_extrinsic=base_extrinsic,
            device=device,
            radius_px=prototype_radius_px,
            grid_size=prototype_grid_size,
            temperature=prototype_temperature,
            min_confidence=prototype_min_confidence,
            pnp_history_frames=pnp_history_frames,
            pnp_max_reprojection_error_px=pnp_max_reprojection_error_px,
            pnp_max_pose_rotation_deg=pnp_max_pose_rotation_deg,
            pnp_max_pose_translation_cm=pnp_max_pose_translation_cm,
        )
    history: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)
    initial_record = collect_observation(
        env,
        obs,
        camera_id,
        intrinsic,
        base_extrinsic,
        width,
        height,
        obs_mode,
        keypoint_occlusion_radius_px,
        occlusion_value,
    )
    if external_runtime is not None:
        initial_record = external_runtime.enrich(initial_record)
    history.append(initial_record)
    rewards: list[float] = []
    successes: list[float] = []
    actions: list[np.ndarray] = []
    pixel_errors: list[float] = []
    grounding_errors: list[float] = []
    reliabilities: list[float] = []
    external_confidences: list[float] = []
    external_valid_fractions: list[float] = []
    pnp_solved_steps: list[float] = []
    pnp_errors: list[float] = []
    terminated = False
    truncated = False
    while len(rewards) < horizon and not (terminated or truncated):
        batch = history_batch(history, obs_horizon, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model.predict_action_chunk(batch)
        q_true = batch["q_true"][:, -1]
        q_nominal = batch["q_nominal"][:, -1]
        q_grounded = output.grounding_points_px[:, -1]
        pixel_errors.append(float(torch.linalg.vector_norm(q_nominal - q_true, dim=-1).mean().cpu()))
        grounding_errors.append(float(torch.linalg.vector_norm(q_grounded - q_true, dim=-1).mean().cpu()))
        reliabilities.append(float(output.reliability[:, -1].mean().cpu()))
        latest = history[-1]
        if "external_match_confidence" in latest:
            external_confidences.append(float(np.mean(latest["external_match_confidence"])))
            external_valid_fractions.append(float(np.mean(latest["external_match_valid"])))
            pnp_solved_steps.append(float(latest["external_pnp_solved"]))
            pnp_error = float(latest["external_pnp_error"])
            if np.isfinite(pnp_error):
                pnp_errors.append(pnp_error)
        for action in output.action[0, :action_steps].float().cpu().numpy():
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            success = float(info.get("success", 0.0)) if isinstance(info, dict) else 0.0
            successes.append(success)
            next_record = collect_observation(
                env,
                obs,
                camera_id,
                intrinsic,
                base_extrinsic,
                width,
                height,
                obs_mode,
                keypoint_occlusion_radius_px,
                occlusion_value,
            )
            if external_runtime is not None:
                next_record = external_runtime.enrich(next_record)
            history.append(next_record)
            if success > 0 or terminated or truncated or len(rewards) >= horizon:
                break
    if hasattr(env, "close"):
        env.close()
    return {
        "task": task_name,
        "task_index": task_index,
        "env_seed": env_seed,
        "steps": len(rewards),
        "success": float(max(successes, default=0.0)),
        "reward_sum": float(np.sum(rewards)),
        "action_abs_mean": float(np.mean(np.abs(np.asarray(actions)))) if actions else 0.0,
        "pixel_error_mean": float(np.mean(pixel_errors)) if pixel_errors else 0.0,
        "grounding_error_mean": float(np.mean(grounding_errors)) if grounding_errors else 0.0,
        "corrected_pixel_error_mean": (
            float(np.mean(grounding_errors)) if grounding_errors else 0.0
        ),
        "reliability_mean": float(np.mean(reliabilities)) if reliabilities else 0.0,
        "external_match_confidence_mean": (
            float(np.mean(external_confidences)) if external_confidences else None
        ),
        "external_match_valid_fraction": (
            float(np.mean(external_valid_fractions)) if external_valid_fractions else None
        ),
        "external_pnp_solved_fraction": (
            float(np.mean(pnp_solved_steps)) if pnp_solved_steps else None
        ),
        "external_pnp_error_mean": float(np.mean(pnp_errors)) if pnp_errors else None,
        "actual_camera_translation_cm": float(np.linalg.norm(camera_offset[0]) * 100.0),
        "actual_camera_translation_x_cm": float(camera_offset[0][0] * 100.0),
        "actual_camera_translation_y_cm": float(camera_offset[0][1] * 100.0),
        "actual_camera_translation_z_cm": float(camera_offset[0][2] * 100.0),
        "actual_camera_rotation_axis_x": float(camera_offset[1][0]),
        "actual_camera_rotation_axis_y": float(camera_offset[1][1]),
        "actual_camera_rotation_axis_z": float(camera_offset[1][2]),
        "actual_camera_rotation_deg": float(camera_offset[2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="pick-place-v3")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--task-index-offset", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--translation-cm-min", type=float, default=0.0)
    parser.add_argument("--translation-cm-max", type=float, default=0.0)
    parser.add_argument("--rotation-deg-min", type=float, default=0.0)
    parser.add_argument("--rotation-deg-max", type=float, default=0.0)
    parser.add_argument("--clean-probability", type=float, default=0.0)
    parser.add_argument("--grounding-override", choices=("native", "nominal", "oracle"), default="native")
    parser.add_argument(
        "--external-grounding",
        choices=("none", "prototype", "pnp"),
        default="none",
        help="replace native FCPG corrections with an image-only matcher or matcher+PnP",
    )
    parser.add_argument("--prototype-bank")
    parser.add_argument("--prototype-radius-px", type=float, default=32.0)
    parser.add_argument("--prototype-grid-size", type=int, default=9)
    parser.add_argument("--prototype-temperature", type=float, default=0.0)
    parser.add_argument("--prototype-min-confidence", type=float, default=-1.0)
    parser.add_argument("--pnp-history-frames", type=int, default=3)
    parser.add_argument("--pnp-max-reprojection-error-px", type=float, default=8.0)
    parser.add_argument("--pnp-max-pose-rotation-deg", type=float, default=30.0)
    parser.add_argument("--pnp-max-pose-translation-cm", type=float, default=20.0)
    parser.add_argument(
        "--reliability-override",
        choices=("native", "ones", "zeros"),
        default="native",
    )
    parser.add_argument("--keypoint-occlusion-radius-px", type=int, default=0)
    parser.add_argument("--occlusion-value", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-seed-base", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/action_chunk_rollout")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = choose_device(args.device)
    model, checkpoint_args = load_action_chunk_policy(args.checkpoint, device)
    source_mode = model.mode
    if args.external_grounding != "none":
        if args.grounding_override != "native":
            parser.error("external grounding cannot be combined with grounding override")
        if source_mode != "fcpg":
            parser.error("external grounding currently requires an FCPG policy checkpoint")
        if not args.prototype_bank:
            parser.error("external grounding requires --prototype-bank")
        if args.reliability_override != "native":
            parser.error("external grounding fixes reliability to one")
    if args.grounding_override == "nominal":
        model.mode = "geoprop"
    elif args.grounding_override == "oracle":
        model.mode = "oracle"
    checkpoint_reliability_override = str(
        checkpoint_args.get("reliability_override", "native")
    )
    if args.reliability_override != "native":
        if source_mode != "fcpg" or args.grounding_override != "native":
            parser.error("reliability override requires a native FCPG checkpoint")
        model.set_reliability_override(args.reliability_override)
    active_reliability_override = (
        checkpoint_reliability_override
        if args.reliability_override == "native"
        else args.reliability_override
    )
    model_name = source_mode if args.grounding_override == "native" else f"{source_mode}_{args.grounding_override}"
    if args.external_grounding != "none":
        model_name = f"{source_mode}_{args.external_grounding}"
    if (
        source_mode == "fcpg"
        and args.grounding_override == "native"
        and active_reliability_override != "native"
    ):
        model_name = f"{model_name}_rho_{active_reliability_override}"
    prototype_bank = (
        load_prototype_bank(
            args.prototype_bank,
            model,
            image_size=(args.width, args.height),
            device=device,
        )
        if args.external_grounding != "none"
        else None
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for episode in range(args.episodes):
        result = rollout_once(
            model,
            checkpoint_args,
            args.task,
            args.camera,
            args.width,
            args.height,
            args.horizon,
            args.action_steps,
            rng,
            args.translation_cm_min,
            args.translation_cm_max,
            args.rotation_deg_min,
            args.rotation_deg_max,
            args.clean_probability,
            device,
            args.task_index_offset + episode,
            args.env_seed_base + episode,
            args.keypoint_occlusion_radius_px,
            args.occlusion_value,
            args.external_grounding,
            prototype_bank,
            args.prototype_radius_px,
            args.prototype_grid_size,
            args.prototype_temperature,
            args.prototype_min_confidence,
            args.pnp_history_frames,
            args.pnp_max_reprojection_error_px,
            args.pnp_max_pose_rotation_deg,
            args.pnp_max_pose_translation_cm,
        )
        row = {"episode": episode, "model": model_name, **result}
        rows.append(row)
        print(
            f"episode {episode:03d} model={model_name} success={row['success']:.1f} "
            f"reward={row['reward_sum']:.2f} pixel={row['pixel_error_mean']:.2f} "
            f"grounded={row['grounding_error_mean']:.2f} rho={row['reliability_mean']:.3f}",
            flush=True,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{model_name}_rollout.csv"
    json_path = output_dir / f"{model_name}_rollout_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": args.checkpoint,
        "model": model_name,
        "source_mode": source_mode,
        "grounding_override": args.grounding_override,
        "external_grounding": args.external_grounding,
        "prototype_bank": (
            str(Path(args.prototype_bank).resolve()) if args.prototype_bank else None
        ),
        "prototype_bank_sha256": (
            prototype_bank.sha256 if prototype_bank is not None else None
        ),
        "prototype_radius_px": args.prototype_radius_px,
        "prototype_grid_size": args.prototype_grid_size,
        "prototype_temperature": args.prototype_temperature,
        "prototype_min_confidence": args.prototype_min_confidence,
        "pnp_history_frames": args.pnp_history_frames,
        "pnp_max_reprojection_error_px": args.pnp_max_reprojection_error_px,
        "pnp_max_pose_rotation_deg": args.pnp_max_pose_rotation_deg,
        "pnp_max_pose_translation_cm": args.pnp_max_pose_translation_cm,
        "pnp_pose_prior_mode": "camera_center_v1",
        "reliability_override_request": args.reliability_override,
        "checkpoint_reliability_override": checkpoint_reliability_override,
        "active_reliability_override": active_reliability_override,
        "keypoint_occlusion_radius_px": args.keypoint_occlusion_radius_px,
        "occlusion_value": args.occlusion_value,
        "task": args.task,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "action_steps": args.action_steps,
        "translation_cm_min": args.translation_cm_min,
        "translation_cm_max": args.translation_cm_max,
        "rotation_deg_min": args.rotation_deg_min,
        "rotation_deg_max": args.rotation_deg_max,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "reward_sum_mean": float(np.mean([row["reward_sum"] for row in rows])),
        "pixel_error_mean": float(np.mean([row["pixel_error_mean"] for row in rows])),
        "grounding_error_mean": float(np.mean([row["grounding_error_mean"] for row in rows])),
        "reliability_mean": float(np.mean([row["reliability_mean"] for row in rows])),
        "external_match_confidence_mean": optional_mean(
            rows, "external_match_confidence_mean"
        ),
        "external_match_valid_fraction": optional_mean(
            rows, "external_match_valid_fraction"
        ),
        "external_pnp_solved_fraction": optional_mean(
            rows, "external_pnp_solved_fraction"
        ),
        "external_pnp_error_mean": optional_mean(rows, "external_pnp_error_mean"),
    }
    json_path.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2), encoding="utf-8")
    print(f"csv: {csv_path}")
    print(f"summary: {json_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
