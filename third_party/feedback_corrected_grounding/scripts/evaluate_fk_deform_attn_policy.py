#!/usr/bin/env python3
"""Evaluate the matched policy-only grounding pair in MetaWorld rollouts."""

from __future__ import annotations

import argparse
import csv
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.fk_deform_attn_baseline import (  # noqa: E402
    DEPLOYMENT_INPUT_KEYS,
    PolicyReadoutOutput,
    VARIANTS,
    load_matched_policy_only_grounding,
    validate_structural_pair,
)
from fcpg.geometry import Camera, project_points  # noqa: E402
from fcpg.metaworld_utils import (  # noqa: E402
    DEFAULT_KEYPOINT_SITES,
    get_site_keypoints,
    make_env,
)
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    sample_camera_pose_offset,
    set_fixed_camera_pose_offset,
)
from fcpg.observations import select_policy_observation  # noqa: E402


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _hash_arrays(*values: tuple[str, object]) -> str:
    digest = hashlib.sha256()
    for name, raw_value in values:
        digest.update(name.encode("utf-8"))
        if isinstance(raw_value, np.ndarray):
            value = np.ascontiguousarray(raw_value)
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.tobytes())
        else:
            digest.update(str(raw_value).encode("utf-8"))
    return digest.hexdigest()


def collect_deployment_observation(
    env,
    observation: np.ndarray,
    nominal_camera: Camera,
    obs_mode: str,
    *,
    diagnostic_camera: Camera | None = None,
    image_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Collect only information available to a deployed FK-anchored policy."""

    image = (
        env.render().astype(np.uint8)
        if image_override is None
        else np.asarray(image_override, dtype=np.uint8)
    )
    keypoints = get_site_keypoints(
        env.model,
        env.data,
        DEFAULT_KEYPOINT_SITES,
    ).xyz
    q_nominal, _ = project_points(keypoints, nominal_camera)
    record = {
        "rgb": np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)),
        "obs": select_policy_observation(observation, obs_mode),
        "keypoints_xyz": keypoints.astype(np.float32),
        "q_nominal": q_nominal.astype(np.float32),
    }
    if diagnostic_camera is not None:
        q_true, q_true_valid = project_points(keypoints, diagnostic_camera)
        record["diagnostic_q_true"] = q_true.astype(np.float32)
        record["diagnostic_q_true_valid"] = q_true_valid.astype(np.bool_)
    return record


def stable_initial_render(env, max_attempts: int = 8) -> np.ndarray:
    """Return the first byte-stable render without advancing physics."""

    if max_attempts < 2:
        raise ValueError("max_attempts must be at least two")
    previous = env.render().astype(np.uint8)
    maximum_lsb_delta = 0
    maximum_changed_values = 0
    for _ in range(1, max_attempts):
        current = env.render().astype(np.uint8)
        if np.array_equal(previous, current):
            return current
        difference = np.abs(current.astype(np.int16) - previous.astype(np.int16))
        maximum_lsb_delta = max(maximum_lsb_delta, int(difference.max()))
        maximum_changed_values = max(
            maximum_changed_values,
            int(np.count_nonzero(difference)),
        )
        previous = current
    raise RuntimeError(
        "initial renderer did not stabilize without a physics step: "
        f"max_lsb_delta={maximum_lsb_delta}, "
        f"max_changed_values={maximum_changed_values}"
    )


def deployment_history_batch(
    history: deque[dict[str, np.ndarray]],
    obs_horizon: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    records = list(history)
    if not records:
        raise ValueError("history must not be empty")
    records = [records[0]] * max(0, obs_horizon - len(records)) + records[
        -obs_horizon:
    ]
    required = ("rgb", "obs", "keypoints_xyz", "q_nominal")
    if not set(required).issubset(DEPLOYMENT_INPUT_KEYS):
        raise RuntimeError("rollout fields violate the deployment input contract")
    return {
        key: torch.from_numpy(
            np.stack([record[key] for record in records], axis=0)
        )
        .unsqueeze(0)
        .to(device)
        for key in required
    }


def diagnostic_history_batch(
    history: deque[dict[str, np.ndarray]],
    obs_horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    records = list(history)
    if not records:
        raise ValueError("history must not be empty")
    records = [records[0]] * max(0, obs_horizon - len(records)) + records[
        -obs_horizon:
    ]
    required = ("diagnostic_q_true", "diagnostic_q_true_valid")
    if any(key not in record for record in records for key in required):
        raise KeyError("diagnostic history is missing actual-camera projection targets")
    q_true = torch.from_numpy(
        np.stack([record["diagnostic_q_true"] for record in records], axis=0)
    ).unsqueeze(0).to(device)
    valid = torch.from_numpy(
        np.stack(
            [record["diagnostic_q_true_valid"] for record in records], axis=0
        )
    ).unsqueeze(0).to(device)
    return q_true, valid


def _unavailable_coordinate_diagnostic(reason: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "name": "attention_barycenter_epe_px",
        "epe_claim_supported": False,
        "reason": reason,
        "latest_epe_sum_px": None,
        "latest_valid_count": 0,
        "all_epe_sum_px": None,
        "all_valid_count": 0,
    }


def _coordinate_diagnostic_from_readout(
    readout: object,
    q_true: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, object]:
    """Score only the attention barycenter of the model's actual sample points."""

    if not isinstance(readout, PolicyReadoutOutput):
        return _unavailable_coordinate_diagnostic(
            "forward_with_readout did not return PolicyReadoutOutput"
        )
    if q_true.ndim != 4 or q_true.shape[-1] != 2:
        return _unavailable_coordinate_diagnostic(
            "actual-camera q_true does not have shape [B,H,K,2]"
        )
    if valid.shape != q_true.shape[:-1]:
        return _unavailable_coordinate_diagnostic(
            "actual-camera validity mask does not align with q_true"
        )
    if readout.sample_points_px.ndim != 4 or readout.sample_points_px.shape[-1] != 2:
        return _unavailable_coordinate_diagnostic(
            "readout sample points do not have shape [B*H,K,S,2]"
        )
    if readout.attention_weights.shape != readout.sample_points_px.shape[:-1]:
        return _unavailable_coordinate_diagnostic(
            "readout attention weights do not align with actual sample points"
        )
    batch, horizon, keypoints, _ = q_true.shape
    expected_prefix = (batch * horizon, keypoints)
    if tuple(readout.sample_points_px.shape[:2]) != expected_prefix:
        return _unavailable_coordinate_diagnostic(
            "readout sample points do not align with q_true history"
        )
    points = readout.sample_points_px.float()
    weights = readout.attention_weights.float()
    if not bool(torch.isfinite(points).all()) or not bool(torch.isfinite(weights).all()):
        return _unavailable_coordinate_diagnostic(
            "readout sample coordinates or attention weights are non-finite"
        )
    if bool((weights < 0).any()) or not torch.allclose(
        weights.sum(dim=-1),
        torch.ones_like(weights[..., 0]),
        atol=1e-4,
        rtol=1e-4,
    ):
        return _unavailable_coordinate_diagnostic(
            "readout attention weights are not normalized probabilities"
        )
    barycenter = (points * weights.unsqueeze(-1)).sum(dim=-2).reshape(
        batch,
        horizon,
        keypoints,
        2,
    )
    reference = q_true.to(device=barycenter.device, dtype=barycenter.dtype)
    mask = valid.to(device=barycenter.device, dtype=torch.bool)
    finite_reference = torch.isfinite(reference).all(dim=-1)
    mask = mask & finite_reference
    latest_mask = mask[:, -1]
    if not bool(latest_mask.any()):
        return _unavailable_coordinate_diagnostic(
            "latest model readout has no valid actual-camera q_true projection"
        )
    error = torch.linalg.vector_norm(barycenter - reference, dim=-1)
    return {
        "status": "available",
        "name": "attention_barycenter_epe_px",
        "epe_claim_supported": True,
        "reason": "",
        "latest_epe_sum_px": float(error[:, -1][latest_mask].sum().cpu()),
        "latest_valid_count": int(latest_mask.sum().cpu()),
        "all_epe_sum_px": float(error[mask].sum().cpu()),
        "all_valid_count": int(mask.sum().cpu()),
    }


@torch.inference_mode()
def predict_action_chunk_with_coordinate_diagnostic(
    model,
    batch: dict[str, torch.Tensor],
    q_true: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[object, dict[str, object]]:
    """Run policy inference without ever exposing q_true to the model batch."""

    if "q_true" in batch or "diagnostic_q_true" in batch:
        raise RuntimeError("projection targets must not enter the deployment batch")
    forward_with_readout = getattr(model, "forward_with_readout", None)
    if not callable(forward_with_readout):
        output = model.predict_action_chunk(batch)
        return output, _unavailable_coordinate_diagnostic(
            "model does not expose its actual policy readout sample coordinates"
        )
    model.eval()
    output, readout = forward_with_readout(batch)
    return output, _coordinate_diagnostic_from_readout(readout, q_true, valid)


def camera_diagnostics(
    camera_offset: tuple[np.ndarray, np.ndarray, float],
) -> dict[str, float]:
    translation, axis, rotation = camera_offset
    return {
        "actual_camera_translation_cm": float(np.linalg.norm(translation) * 100.0),
        "actual_camera_translation_x_cm": float(translation[0] * 100.0),
        "actual_camera_translation_y_cm": float(translation[1] * 100.0),
        "actual_camera_translation_z_cm": float(translation[2] * 100.0),
        "actual_camera_rotation_axis_x": float(axis[0]),
        "actual_camera_rotation_axis_y": float(axis[1]),
        "actual_camera_rotation_axis_z": float(axis[2]),
        "actual_camera_rotation_deg": float(rotation),
    }


def rollout_once(
    *,
    model,
    checkpoint: dict[str, object],
    task_name: str,
    camera_name: str,
    width: int,
    height: int,
    horizon: int,
    action_steps: int,
    task_index: int,
    env_seed: int,
    camera_offset: tuple[np.ndarray, np.ndarray, float],
    device: torch.device,
    amp: bool,
) -> dict[str, object]:
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)
    env = make_env(
        task_name,
        camera_name,
        width,
        height,
        task_index=task_index,
    )
    try:
        observation, _ = env.reset(seed=env_seed)
        camera_id = get_camera_id(env.model, camera_name)
        intrinsic = camera_intrinsic_from_fovy(
            width,
            height,
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
        nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
        nominal_camera = Camera(
            intrinsic,
            nominal_extrinsic,
            width,
            height,
        )
        set_fixed_camera_pose_offset(
            env.model,
            env.data,
            camera_id,
            base_position,
            base_quaternion,
            *camera_offset,
        )
        actual_camera = Camera(
            intrinsic,
            camera_extrinsic_cv_from_mujoco(env.data, camera_id),
            width,
            height,
        )

        build = checkpoint["build_contract"]
        if not isinstance(build, dict):
            raise TypeError("checkpoint build contract must be a dictionary")
        policy_config = build["policy_config"]
        if not isinstance(policy_config, dict):
            raise TypeError("checkpoint policy config must be a dictionary")
        obs_mode = str(policy_config["obs_mode"])
        obs_horizon = int(policy_config["obs_horizon"])
        history: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)
        initial_image = stable_initial_render(env)
        initial_record = collect_deployment_observation(
            env,
            observation,
            nominal_camera,
            obs_mode,
            diagnostic_camera=actual_camera,
            image_override=initial_image,
        )
        history.append(initial_record)
        physical_episode_sha256 = _hash_arrays(
            ("task", task_name),
            ("task_index", task_index),
            ("env_seed", env_seed),
            ("initial_observation", np.asarray(observation)),
            ("initial_keypoints_xyz", initial_record["keypoints_xyz"]),
        )
        paired_initial_view_sha256 = _hash_arrays(
            ("physical_episode_sha256", physical_episode_sha256),
            ("camera_translation", np.asarray(camera_offset[0])),
            ("camera_axis", np.asarray(camera_offset[1])),
            ("camera_rotation", camera_offset[2]),
            ("initial_rgb", initial_record["rgb"]),
            ("initial_q_nominal", initial_record["q_nominal"]),
            ("initial_q_true", initial_record["diagnostic_q_true"]),
            (
                "initial_q_true_valid",
                initial_record["diagnostic_q_true_valid"],
            ),
        )
        initial_rgb_sha256 = _hash_arrays(("initial_rgb", initial_record["rgb"]))
        initial_q_nominal_sha256 = _hash_arrays(
            ("initial_q_nominal", initial_record["q_nominal"])
        )
        initial_q_true_sha256 = _hash_arrays(
            ("initial_q_true", initial_record["diagnostic_q_true"])
        )
        initial_q_true_valid_sha256 = _hash_arrays(
            (
                "initial_q_true_valid",
                initial_record["diagnostic_q_true_valid"],
            )
        )
        rewards: list[float] = []
        successes: list[float] = []
        actions: list[np.ndarray] = []
        terminated = False
        truncated = False
        inference_calls = 0
        coordinate_failure_reason: str | None = None
        initial_epe_sum_px = 0.0
        initial_valid_count = 0
        on_policy_epe_sum_px = 0.0
        on_policy_valid_count = 0
        while len(rewards) < horizon and not (terminated or truncated):
            batch = deployment_history_batch(history, obs_horizon, device)
            q_true, q_true_valid = diagnostic_history_batch(
                history,
                obs_horizon,
                device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                output, coordinate_diagnostic = (
                    predict_action_chunk_with_coordinate_diagnostic(
                        model,
                        batch,
                        q_true,
                        q_true_valid,
                    )
                )
            if bool(coordinate_diagnostic["epe_claim_supported"]):
                latest_sum = coordinate_diagnostic["latest_epe_sum_px"]
                all_sum = coordinate_diagnostic["all_epe_sum_px"]
                if not isinstance(latest_sum, float) or not isinstance(all_sum, float):
                    raise RuntimeError("available coordinate diagnostic has no EPE sums")
                if inference_calls == 0:
                    initial_epe_sum_px = latest_sum
                    initial_valid_count = int(
                        coordinate_diagnostic["latest_valid_count"]
                    )
                on_policy_epe_sum_px += all_sum
                on_policy_valid_count += int(
                    coordinate_diagnostic["all_valid_count"]
                )
            elif coordinate_failure_reason is None:
                coordinate_failure_reason = str(coordinate_diagnostic["reason"])
            inference_calls += 1
            for action in output.action[0, :action_steps].float().cpu().numpy():
                action = np.clip(
                    action,
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)
                actions.append(action)
                observation, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                success = (
                    float(info.get("success", 0.0))
                    if isinstance(info, dict)
                    else 0.0
                )
                successes.append(success)
                history.append(
                    collect_deployment_observation(
                        env,
                        observation,
                        nominal_camera,
                        obs_mode,
                        diagnostic_camera=actual_camera,
                    )
                )
                if success > 0 or terminated or truncated or len(rewards) >= horizon:
                    break
        coordinate_supported = (
            coordinate_failure_reason is None
            and initial_valid_count > 0
            and on_policy_valid_count > 0
        )
        if not coordinate_supported and coordinate_failure_reason is None:
            coordinate_failure_reason = "rollout produced no valid coordinate diagnostic"
        return {
            "task": task_name,
            "task_index": int(task_index),
            "env_seed": int(env_seed),
            "physical_episode_sha256": physical_episode_sha256,
            "paired_initial_view_sha256": paired_initial_view_sha256,
            "initial_rgb_sha256": initial_rgb_sha256,
            "initial_q_nominal_sha256": initial_q_nominal_sha256,
            "initial_q_true_sha256": initial_q_true_sha256,
            "initial_q_true_valid_sha256": initial_q_true_valid_sha256,
            "steps": len(rewards),
            "inference_calls": inference_calls,
            "success": float(max(successes, default=0.0)),
            "reward_sum": float(np.sum(rewards)),
            "action_abs_mean": (
                float(np.mean(np.abs(np.asarray(actions)))) if actions else 0.0
            ),
            "coordinate_diagnostic_status": (
                "available" if coordinate_supported else "unavailable"
            ),
            "coordinate_diagnostic_name": "attention_barycenter_epe_px",
            "coordinate_diagnostic_reason": (
                "" if coordinate_supported else coordinate_failure_reason
            ),
            "coordinate_epe_claim_supported": coordinate_supported,
            "initial_attention_barycenter_epe_sum_px": (
                initial_epe_sum_px if coordinate_supported else None
            ),
            "initial_attention_barycenter_valid_count": (
                initial_valid_count if coordinate_supported else 0
            ),
            "initial_attention_barycenter_epe_px": (
                initial_epe_sum_px / initial_valid_count
                if coordinate_supported
                else None
            ),
            "on_policy_attention_barycenter_epe_sum_px": (
                on_policy_epe_sum_px if coordinate_supported else None
            ),
            "on_policy_attention_barycenter_valid_count": (
                on_policy_valid_count if coordinate_supported else 0
            ),
            "on_policy_attention_barycenter_epe_px": (
                on_policy_epe_sum_px / on_policy_valid_count
                if coordinate_supported
                else None
            ),
            **camera_diagnostics(camera_offset),
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--deform-checkpoint", required=True)
    parser.add_argument("--policy-contract-checkpoint")
    parser.add_argument("--fcpg-corrector-checkpoint")
    parser.add_argument("--task", default="pick-place-v3")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--task-index-offset", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--translation-cm-min", type=float, default=0.0)
    parser.add_argument("--translation-cm-max", type=float, default=4.0)
    parser.add_argument("--rotation-deg-min", type=float, default=0.0)
    parser.add_argument("--rotation-deg-max", type=float, default=5.0)
    parser.add_argument("--clean-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-seed-base", type=int, default=1000)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/fk_deform_attn_rollout")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.episodes <= 0 or args.horizon <= 0 or args.action_steps <= 0:
        raise ValueError("episodes, horizon, and action-steps must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = choose_device(args.device)
    load_kwargs = {
        "policy_contract_checkpoint": args.policy_contract_checkpoint,
        "fcpg_corrector_checkpoint": args.fcpg_corrector_checkpoint,
    }
    residual_model, residual_checkpoint = load_matched_policy_only_grounding(
        args.residual_checkpoint,
        device,
        **load_kwargs,
    )
    deform_model, deform_checkpoint = load_matched_policy_only_grounding(
        args.deform_checkpoint,
        device,
        **load_kwargs,
    )
    pair_validation = validate_structural_pair(
        residual_checkpoint,
        deform_checkpoint,
    )
    deform_model.policy.vision_encoder.backbone = (
        residual_model.policy.vision_encoder.backbone
    )
    models = {
        "policy_only_residual": (residual_model, residual_checkpoint),
        "fk_deform_attn": (deform_model, deform_checkpoint),
    }

    rng = np.random.default_rng(args.seed)
    camera_offsets = [
        sample_camera_pose_offset(
            rng,
            args.translation_cm_max,
            args.rotation_deg_max,
            args.clean_probability,
            translation_cm_min=args.translation_cm_min,
            rotation_deg_min=args.rotation_deg_min,
        )
        for _ in range(args.episodes)
    ]
    rows: list[dict[str, object]] = []
    for episode, camera_offset in enumerate(camera_offsets):
        for variant in VARIANTS:
            model, checkpoint = models[variant]
            result = rollout_once(
                model=model,
                checkpoint=checkpoint,
                task_name=args.task,
                camera_name=args.camera,
                width=args.width,
                height=args.height,
                horizon=args.horizon,
                action_steps=args.action_steps,
                task_index=args.task_index_offset + episode,
                env_seed=args.env_seed_base + episode,
                camera_offset=(
                    camera_offset[0].copy(),
                    camera_offset[1].copy(),
                    camera_offset[2],
                ),
                device=device,
                amp=args.amp,
            )
            row = {"episode": episode, "variant": variant, **result}
            rows.append(row)
            print(
                f"episode {episode:03d} variant={variant} "
                f"success={row['success']:.1f} reward={row['reward_sum']:.2f} "
                f"inference_calls={row['inference_calls']}",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "matched_pair_rollout.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate: dict[str, dict[str, float | int]] = {}
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        diagnostic_supported = all(
            bool(row["coordinate_epe_claim_supported"])
            for row in variant_rows
        )
        initial_epe_sum = sum(
            float(row["initial_attention_barycenter_epe_sum_px"])
            for row in variant_rows
            if diagnostic_supported
        )
        initial_epe_count = sum(
            int(row["initial_attention_barycenter_valid_count"])
            for row in variant_rows
            if diagnostic_supported
        )
        aggregate[variant] = {
            "episodes": len(variant_rows),
            "success_rate": float(np.mean([row["success"] for row in variant_rows])),
            "reward_mean": float(np.mean([row["reward_sum"] for row in variant_rows])),
            "rollout_inference_calls": int(
                    sum(int(row["inference_calls"]) for row in variant_rows)
                ),
            "coordinate_epe_claim_supported": diagnostic_supported,
            "initial_attention_barycenter_epe_px": (
                initial_epe_sum / initial_epe_count
                if diagnostic_supported and initial_epe_count > 0
                else None
            ),
        }
    structural_latency = {
        "source": "checkpoint_fixed_input_equal_count_interleaved_benchmark",
        "comparison_contract": pair_validation["structural_latency_contract"],
        "rollout_call_counts_used": False,
        "measurements": {
            variant: models[variant][1]["runtime_budget"]["structural_latency"][
                "measurement"
            ]
            for variant in VARIANTS
        },
    }
    summary = {
        "complete": True,
        "evaluation_contract": {
            "paired_environment_seed": True,
            "paired_task_index": True,
            "paired_camera_offset": True,
            "model_input_fields": (
                "rgb",
                "obs",
                "keypoints_xyz",
                "q_nominal",
            ),
            "projection_target_access": False,
            "coordinate_diagnostics": {
                "name": "attention_barycenter_epe_px",
                "coordinate_source": (
                    "PolicyReadoutOutput.sample_points_px attention-weighted by "
                    "PolicyReadoutOutput.attention_weights"
                ),
                "reference": "q_true_from_actual_camera_projection",
                "reference_is_model_input": False,
                "validity": "in_front_of_camera_and_in_image",
                "fail_closed": True,
                "unqualified_keypoint_epe_claim_supported": False,
            },
            "rollout_inference_call_counts_are_behavior_dependent": True,
            "rollout_call_counts_used_for_structural_latency": False,
            "structural_latency_requires_fixed_input_equal_counts": True,
        },
        "pair_validation": pair_validation,
        "selection_summary": {
            "interpretation": pair_validation["selection_interpretation"],
            "selected_epochs": pair_validation["selected_epochs"],
            "same_selected_step_claim_supported": False,
            "matched_maximum_budget": pair_validation["matched_maximum_budget"],
        },
        "claim_contract": pair_validation["claim_contract"],
        "task": args.task,
        "camera": args.camera,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "action_steps": args.action_steps,
        "device": str(device),
        "amp": bool(args.amp),
        "checkpoints": {
            "policy_only_residual": str(Path(args.residual_checkpoint).resolve()),
            "fk_deform_attn": str(Path(args.deform_checkpoint).resolve()),
        },
        "parameter_counts": {
            variant: models[variant][1]["parameter_counts"] for variant in VARIANTS
        },
        "training_runtime_budget": {
            variant: models[variant][1]["runtime_budget"] for variant in VARIANTS
        },
        "structural_latency": structural_latency,
        "rollout": aggregate,
        "rows": str(csv_path.resolve()),
    }
    summary_path = output_dir / "matched_pair_rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
