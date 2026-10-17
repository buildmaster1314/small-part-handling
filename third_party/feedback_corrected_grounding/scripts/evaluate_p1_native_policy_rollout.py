#!/usr/bin/env python3
"""Paired policy rollout for the frozen P1-native complementary observer.

This is a known-validity development Gate.  ``q_true`` centers a synthetic
local sensor interruption and hidden-clean actions are retained only as
same-state offline labels.  Neither enters the policy or P1 dynamics runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import sys
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_a4b_task_identity import (  # noqa: E402
    reset_physical_state_goal_digest,
)
from fcpg.lewm_known_drop_policy_evaluator import (  # noqa: E402
    deployment_policy_batch,
    extract_policy_anchor_observation,
)
from fcpg.lewm_p1_native_complementary import (  # noqa: E402
    FrozenP1NativeCheckpointSpec,
    load_frozen_p1_native_checkpoint,
)
from fcpg.lewm_p1_native_fresh_collection import sha256_file  # noqa: E402
from fcpg.lewm_p1_native_policy_runtime import (  # noqa: E402
    P1_POLICY_ACTION_STEPS,
    P1NativeComplementaryPolicyRuntime,
)
from fcpg.metaworld_utils import make_env  # noqa: E402
from fcpg.moving_goal import (  # noqa: E402
    DEFAULT_AMPLITUDE_XY,
    DEFAULT_PERIOD_RAW_STEPS,
    MovingGoalEnv,
)
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    set_fixed_camera_pose_offset,
)
from fcpg.observations import apply_keypoint_square_occlusion  # noqa: E402
from scripts.evaluate_action_chunk_rollout import (  # noqa: E402
    choose_device,
    collect_observation,
    history_batch,
)


PROTOCOL = "fcpg_p1_native_known_drop_policy_development_v1"
JOINT_DRIFT_DROP_PROTOCOL = "fcpg_p1_native_joint_drift_drop_screen_v1"
JOINT_DRIFT_DROP_CONFIRMATION_PROTOCOL = (
    "fcpg_p1_native_joint_drift_drop_confirmation_v1"
)
SUPPORTED_PROTOCOLS = (
    PROTOCOL,
    JOINT_DRIFT_DROP_PROTOCOL,
    JOINT_DRIFT_DROP_CONFIRMATION_PROTOCOL,
)
BASELINE_METHODS = (
    "clean_native",
    "measurement",
    "hold_last",
    "geometry_hold",
    "visual_cv",
)
VISIBLE_GROUNDING_MODES = ("nominal", "fcpg")


def _resolve(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = REPO_ROOT / target
    return target.resolve()


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be a mapping: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_hash(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _maximum_success_streak(values: Sequence[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        if float(value) > 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _validate_config(config: Mapping[str, object]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("protocol") not in SUPPORTED_PROTOCOLS
    ):
        raise ValueError("policy rollout config has the wrong schema or protocol")
    episodes = int(config["physical_episode_count"])
    task_indices = list(config["task_instance_indices"])
    starts = list(config["drop_start_decisions"])
    if episodes <= 0 or len(task_indices) != episodes or len(starts) != episodes:
        raise ValueError("episode count, task indices, and drop starts must align")
    duration = int(config["drop_duration_decisions"])
    horizon = int(config["horizon_raw_steps"])
    action_steps = int(config["action_steps_per_decision"])
    if action_steps != P1_POLICY_ACTION_STEPS or horizon % action_steps:
        raise ValueError("horizon must contain complete frozen five-action decisions")
    if duration != 6:
        raise ValueError("the frozen P1 observer is scoped to H3-H8 (six decisions)")
    if any(int(start) < 3 for start in starts):
        raise ValueError("every known drop requires at least three warmup decisions")
    if any(int(start) + duration >= horizon // action_steps for start in starts):
        raise ValueError("every rollout must observe the first recovery decision")
    camera_condition = config.get("camera_condition")
    if camera_condition not in {"nominal", "fixed_rotation"}:
        raise ValueError("camera_condition must be nominal or fixed_rotation")
    if camera_condition == "fixed_rotation":
        axis = np.asarray(config.get("camera_rotation_axis_world"), dtype=np.float64)
        degrees = float(config.get("camera_rotation_deg", 0.0))
        if (
            axis.shape != (3,)
            or not np.isfinite(axis).all()
            or np.linalg.norm(axis) <= 0.0
        ):
            raise ValueError(
                "camera_rotation_axis_world must be one finite non-zero 3-vector"
            )
        if not np.isfinite(degrees) or abs(degrees) <= 0.0:
            raise ValueError(
                "fixed_rotation requires a finite non-zero camera_rotation_deg"
            )
    if config.get("known_sensor_validity_only") is not True:
        raise ValueError("this evaluator only supports externally known validity")
    if config.get("protocol") == JOINT_DRIFT_DROP_CONFIRMATION_PROTOCOL:
        if config.get("classification") != "frozen_confirmation_not_policy_success":
            raise ValueError("joint confirmation classification is not frozen")
        if episodes != 30 or task_indices != list(range(30)):
            raise ValueError(
                "joint confirmation is frozen to 30 fresh task-instance episodes"
            )
        expected_methods = [
            "clean_native",
            "measurement",
            "geometry_hold",
            "complementary_factual",
            "complementary_no_action",
        ]
        if list(config.get("methods", [])) != expected_methods:
            raise ValueError("joint confirmation method panel is frozen")
        seeds = tuple(int(value) for value in config.get("full_model_seeds_after_development_go", []))
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("joint confirmation requires exactly three frozen model seeds")
        gate = config.get("confirmation_gate")
        if not isinstance(gate, Mapping):
            raise ValueError("joint confirmation gate is missing")
        if float(gate.get("minimum_relative_improvement", 0.0)) != 0.10:
            raise ValueError("joint confirmation primary threshold is frozen at 10 percent")
        if float(gate.get("primary_ci_lower_must_exceed", 1.0)) != 0.0:
            raise ValueError("joint confirmation primary CI rule is not frozen")


def _corrupt_record(
    clean: Mapping[str, np.ndarray],
    *,
    radius_px: int,
    value_uint8: int,
    center_key: str = "q_true",
) -> dict[str, np.ndarray]:
    """Materialize one corrupted raw frame without mutating the clean label copy."""

    result = {name: np.asarray(value).copy() for name, value in clean.items()}
    rgb = np.asarray(clean["rgb"], dtype=np.float32)
    image = np.rint(np.clip(np.transpose(rgb, (1, 2, 0)), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    image = apply_keypoint_square_occlusion(
        image,
        np.asarray(clean[center_key], dtype=np.float32),
        radius_px=radius_px,
        value=value_uint8,
    )
    result["rgb"] = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
    return result


def _trace_rgb_uint8(record: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert one rollout observation into an immutable HWC display frame."""

    rgb = np.asarray(record["rgb"], dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("rollout trace RGB must have shape [3,H,W]")
    return np.ascontiguousarray(
        np.rint(np.clip(np.transpose(rgb, (1, 2, 0)), 0.0, 1.0) * 255.0),
        dtype=np.uint8,
    )


def _emit_rollout_trace(
    callback: Callable[[Mapping[str, object]], None] | None,
    *,
    raw_step: int,
    decision: int,
    substep: int,
    corruption_active: bool,
    prior_active: bool,
    clean_record: Mapping[str, np.ndarray],
    observed_record: Mapping[str, np.ndarray],
    selected_points: np.ndarray | None,
    action: np.ndarray | None,
    reward: float,
    cumulative_reward: float,
    success: float,
    moving_goal_position: np.ndarray | None,
) -> None:
    """Publish a detached visualization trace without changing rollout state."""

    if callback is None:
        return
    callback(
        {
            "raw_step": int(raw_step),
            "decision": int(decision),
            "substep": int(substep),
            "corruption_active": bool(corruption_active),
            "prior_active": bool(prior_active),
            "clean_rgb": _trace_rgb_uint8(clean_record),
            "observed_rgb": _trace_rgb_uint8(observed_record),
            "q_nominal": np.array(clean_record["q_nominal"], dtype=np.float32, copy=True),
            "q_true": np.array(clean_record["q_true"], dtype=np.float32, copy=True),
            "selected_points": (
                None
                if selected_points is None
                else np.array(selected_points, dtype=np.float32, copy=True)
            ),
            "keypoints_xyz": np.array(
                clean_record["keypoints_xyz"], dtype=np.float32, copy=True
            ),
            "action": (
                None if action is None else np.array(action, dtype=np.float32, copy=True)
            ),
            "reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "success": float(success),
            "moving_goal_position": (
                None
                if moving_goal_position is None
                else np.array(moving_goal_position, dtype=np.float32, copy=True)
            ),
        }
    )


def _policy_action(policy, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    output = policy.predict_action_chunk(dict(batch))
    action = output.action[:, :P1_POLICY_ACTION_STEPS].detach().float()
    if tuple(action.shape) != (1, P1_POLICY_ACTION_STEPS, 4):
        raise ValueError(f"policy action has unexpected shape {tuple(action.shape)}")
    if not bool(torch.isfinite(action).all()):
        raise ValueError("policy returned non-finite action")
    return action


def visible_grounding_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    """Select nominal or FCPG points without changing the frozen policy.

    The P1 policy is an FCPG-mode checkpoint.  Supplying its existing compact
    correction fields is therefore the least invasive way to run a nominal
    intervention: nominal points replace the FCPG correction while every
    policy parameter and all other observations remain identical.
    """

    deployable = deployment_policy_batch(batch)
    if mode not in VISIBLE_GROUNDING_MODES:
        raise ValueError(f"unsupported visible grounding mode {mode!r}")
    if mode == "fcpg":
        return deployable
    if "q_nominal" not in deployable:
        raise KeyError("nominal grounding requires q_nominal")
    nominal = deployable["q_nominal"]
    if nominal.ndim != 4 or nominal.shape[-1] != 2:
        raise ValueError("q_nominal must have shape [B,H,K,2]")
    result = dict(deployable)
    result["fcpg_q_corrected"] = nominal
    result["fcpg_reliability"] = torch.ones_like(nominal[..., 0])
    return result


def _apply_action_residual_head(
    head: torch.nn.Module,
    *,
    base_action: torch.Tensor,
    action_latent: torch.Tensor,
    posterior_token: torch.Tensor,
    missing_age: int,
) -> torch.Tensor:
    """Call legacy and posterior-aware residual heads through one interface."""

    kwargs = {
        "base_action": base_action,
        "state_latent": posterior_token.mean(dim=1),
        "action_latent": action_latent,
        "missing_age": torch.full(
            (base_action.shape[0],),
            int(missing_age),
            dtype=torch.long,
            device=base_action.device,
        ),
        "sensor_valid": torch.zeros(
            (base_action.shape[0],),
            dtype=torch.bool,
            device=base_action.device,
        ),
    }
    if "posterior_token" in inspect.signature(head.forward).parameters:
        kwargs["posterior_token"] = posterior_token
    return head(**kwargs)


def _runtime_method(method: str) -> str:
    if method == "clean_native":
        return "measurement"
    if method.startswith("complementary_factual_trusted_global_s"):
        return "complementary_factual"
    if method.startswith("complementary_factual_s"):
        return "complementary_factual"
    if method.startswith("complementary_no_action_s"):
        return "complementary_no_action"
    return method


def _uses_trusted_global_visual(method: str) -> bool:
    """Enable whole-frame freezing only for the explicit development ablation."""

    return method.startswith("complementary_factual_trusted_global_s")


def _make_runtime(method: str, *, policy, pair) -> P1NativeComplementaryPolicyRuntime:
    return P1NativeComplementaryPolicyRuntime(
        method=_runtime_method(method),
        policy=policy,
        factual_predictor=(None if pair is None else pair.factual),
        no_action_predictor=(None if pair is None else pair.no_action),
        freeze_global_visual_during_drop=_uses_trusted_global_visual(method),
    )


def _set_current_observation(
    observed: deque[dict[str, np.ndarray]],
    clean: deque[dict[str, np.ndarray]],
    *,
    active: bool,
    radius_px: int,
    value_uint8: int,
    center_key: str,
) -> None:
    current = clean[-1]
    observed[-1] = (
        _corrupt_record(
            current,
            radius_px=radius_px,
            value_uint8=value_uint8,
            center_key=center_key,
        )
        if active
        else {name: np.asarray(value).copy() for name, value in current.items()}
    )


def _register_action(
    runtime: P1NativeComplementaryPolicyRuntime,
    actions: np.ndarray,
    *,
    device: torch.device,
) -> None:
    runtime.register_executed_action(
        torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))
        .unsqueeze(0)
        .to(device)
    )


@torch.inference_mode()
def rollout_once(
    policy,
    policy_args: Mapping[str, object],
    *,
    pair,
    method: str,
    model_seed: int,
    task: str,
    camera: str,
    width: int,
    height: int,
    benchmark_seed: int,
    task_index: int,
    environment_seed: int,
    drop_start: int,
    drop_duration: int,
    horizon: int,
    radius_px: int,
    value_uint8: int,
    device: torch.device,
    camera_rotation_axis_world: Sequence[float] = (0.0, 0.0, 1.0),
    camera_rotation_deg: float = 0.0,
    moving_goal_seed: int | None = None,
    moving_goal_amplitude_xy: Sequence[float] = DEFAULT_AMPLITUDE_XY,
    moving_goal_period_raw_steps: float = DEFAULT_PERIOD_RAW_STEPS,
    forced_prefix_actions: np.ndarray | None = None,
    forced_prefix_tokens: np.ndarray | None = None,
    forced_prefix_visual: np.ndarray | None = None,
    action_residual_head: torch.nn.Module | None = None,
    result_method: str | None = None,
    occlusion_center_key: str = "q_true",
    compute_clean_action_metric: bool = True,
    visual_validity_fsm: object | None = None,
    visible_grounding_mode: str = "fcpg",
    trace_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run one method from a deterministic shared episode contract."""

    random.seed(environment_seed)
    np.random.seed(environment_seed)
    torch.manual_seed(environment_seed)
    env = make_env(
        task,
        camera,
        width,
        height,
        task_index=task_index,
        benchmark_seed=benchmark_seed,
    )
    try:
        if moving_goal_seed is not None:
            amplitude = tuple(float(value) for value in moving_goal_amplitude_xy)
            if len(amplitude) != 2:
                raise ValueError("moving_goal_amplitude_xy must contain two values")
            env = MovingGoalEnv(
                env,
                task=task,
                seed=int(moving_goal_seed),
                amplitude_xy=amplitude,
                period_raw_steps=float(moving_goal_period_raw_steps),
            )
        observation, reset_info = env.reset(seed=environment_seed)
        moving_goal_positions: list[np.ndarray] = []
        if moving_goal_seed is not None:
            if not isinstance(reset_info, dict) or "moving_goal_position" not in reset_info:
                raise RuntimeError("moving-goal reset did not expose its target position")
            moving_goal_positions.append(
                np.asarray(reset_info["moving_goal_position"], dtype=np.float32).copy()
            )
        reset_digest = reset_physical_state_goal_digest(env)
        camera_id = get_camera_id(env.model, camera)
        intrinsic = camera_intrinsic_from_fovy(
            width, height, float(env.model.cam_fovy[camera_id])
        )
        base_position = np.asarray(env.model.cam_pos[camera_id], dtype=np.float64).copy()
        base_quaternion = np.asarray(env.model.cam_quat[camera_id], dtype=np.float64).copy()
        nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
        set_fixed_camera_pose_offset(
            env.model,
            env.data,
            camera_id,
            base_position,
            base_quaternion,
            np.zeros(3, dtype=np.float64),
            np.asarray(camera_rotation_axis_world, dtype=np.float64),
            float(camera_rotation_deg),
        )
        actual_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
        camera_sha = _sha256_bytes(
            np.ascontiguousarray(actual_extrinsic, dtype=np.float64).tobytes()
        )
        obs_mode = str(policy_args.get("obs_mode", "robot"))
        obs_horizon = int(policy_args.get("obs_horizon", 2))
        if obs_horizon != 2:
            raise ValueError("the frozen P1 policy must use raw-adjacent history=2")
        clean_history: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)
        observed_history: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)
        initial = collect_observation(
            env,
            observation,
            camera_id,
            intrinsic,
            nominal_extrinsic,
            width,
            height,
            obs_mode,
            0,
            0,
        )
        clean_history.append(initial)
        observed_history.append(
            {name: np.asarray(value).copy() for name, value in initial.items()}
        )
        runtime = _make_runtime(method, policy=policy, pair=pair)
        rewards: list[float] = []
        successes: list[float] = []
        action_blocks: list[np.ndarray] = []
        active_action_mse: list[float] = []
        active_decisions: list[int] = []
        prefix_native_tokens: list[np.ndarray] = []
        prefix_sampled_visual: list[np.ndarray] = []
        recovery_exact = False
        selector_true_positive = 0
        selector_false_positive = 0
        selector_false_negative = 0
        selector_true_negative = 0
        selector_prior_decisions = 0
        selector_reasons: dict[str, int] = {}
        selector_trace: list[dict[str, object]] = []
        first_success_step: int | None = None
        drop_onset_state_digest: str | None = None
        decision_index = 0

        _emit_rollout_trace(
            trace_callback,
            raw_step=0,
            decision=0,
            substep=-1,
            corruption_active=False,
            prior_active=False,
            clean_record=clean_history[-1],
            observed_record=observed_history[-1],
            selected_points=None,
            action=None,
            reward=0.0,
            cumulative_reward=0.0,
            success=0.0,
            moving_goal_position=(
                None if not moving_goal_positions else moving_goal_positions[-1]
            ),
        )

        while len(rewards) < horizon:
            if decision_index == drop_start:
                drop_onset_state_digest = reset_physical_state_goal_digest(env)
            scheduled_active = drop_start <= decision_index < drop_start + drop_duration
            corruption_active = scheduled_active and method != "clean_native"
            _set_current_observation(
                observed_history,
                clean_history,
                active=corruption_active,
                radius_px=radius_px,
                value_uint8=value_uint8,
                center_key=occlusion_center_key,
            )
            observed_batch = history_batch(observed_history, obs_horizon, device)
            observed_deploy = visible_grounding_batch(
                observed_batch,
                mode=visible_grounding_mode,
            )
            current_xyz = observed_deploy["keypoints_xyz"][:, -1].float()

            native_anchor = None
            active = corruption_active
            if visual_validity_fsm is not None and method != "clean_native":
                update_selector = getattr(visual_validity_fsm, "update", None)
                if not callable(update_selector):
                    raise TypeError("visual_validity_fsm must expose update(**signals)")
                native_anchor = extract_policy_anchor_observation(
                    policy, observed_deploy
                )
                current_record = observed_history[-1]
                image = np.asarray(current_record["rgb"], dtype=np.float32)
                nominal = np.asarray(current_record["q_nominal"], dtype=np.float32)
                corrected = (
                    native_anchor.points[:, -1].detach().float().cpu().numpy()[0]
                )
                image_height, image_width = int(image.shape[-2]), int(image.shape[-1])
                valid_nominal = (
                    np.isfinite(nominal).all(axis=-1)
                    & (nominal[:, 0] >= 0.0)
                    & (nominal[:, 0] <= float(image_width - 1))
                    & (nominal[:, 1] >= 0.0)
                    & (nominal[:, 1] <= float(image_height - 1))
                )
                executed_action_rms = (
                    float(np.sqrt(np.mean(np.square(action_blocks[-1][..., :3]))))
                    if action_blocks
                    else 0.0
                )
                selector = update_selector(
                    image=image,
                    frame_present=True,
                    frame_fresh=True,
                    q_nominal=nominal,
                    q_corrected=corrected,
                    valid_nominal=valid_nominal.astype(bool),
                    executed_action_rms=executed_action_rms,
                )
                selected_prior = bool(getattr(selector, "use_prior"))
                if selected_prior and not runtime.ready_for_drop:
                    selected_prior = False
                    selector_reasons["warmup_measurement_override"] = (
                        selector_reasons.get("warmup_measurement_override", 0) + 1
                    )
                active = selected_prior
                horizon_fallback = bool(
                    active and runtime.missing_age >= len(runtime.gains)
                )
                if horizon_fallback:
                    selector_reasons["prior_horizon_geometry_fallback"] = (
                        selector_reasons.get("prior_horizon_geometry_fallback", 0) + 1
                    )
                selector_prior_decisions += int(active)
                for reason in tuple(getattr(selector, "reasons", ())):
                    label = str(reason)
                    selector_reasons[label] = selector_reasons.get(label, 0) + 1
                selector_trace.append(
                    {
                        "decision": int(decision_index),
                        "corruption_active": bool(corruption_active),
                        "use_prior": bool(active),
                        "horizon_geometry_fallback": horizon_fallback,
                        "reasons": [
                            str(value) for value in tuple(getattr(selector, "reasons", ()))
                        ],
                        "sharpness_ratio": float(
                            getattr(selector, "minimum_sharpness_ratio", 1.0)
                        ),
                        "contrast_ratio": float(
                            getattr(selector, "minimum_contrast_ratio", 1.0)
                        ),
                        "maximum_residual_px": float(
                            getattr(selector, "maximum_residual_px", 0.0)
                        ),
                        "maximum_nominal_patch_mean": (
                            None
                            if getattr(selector, "maximum_nominal_patch_mean", None)
                            is None
                            else float(selector.maximum_nominal_patch_mean)
                        ),
                    }
                )
                if active and corruption_active:
                    selector_true_positive += 1
                elif active:
                    selector_false_positive += 1
                elif corruption_active:
                    selector_false_negative += 1
                else:
                    selector_true_negative += 1

            selected_points: np.ndarray | None = None
            if active:
                clean_action = None
                if compute_clean_action_metric:
                    clean_batch = history_batch(clean_history, obs_horizon, device)
                    clean_deploy = visible_grounding_batch(
                        clean_batch,
                        mode=visible_grounding_mode,
                    )
                    clean_action = _policy_action(policy, clean_deploy)
                runtime_name = _runtime_method(method)
                if runtime_name == "measurement":
                    observed_anchor = extract_policy_anchor_observation(
                        policy, observed_deploy
                    )
                    selected_points = (
                        observed_anchor.points[:, -1].detach().float().cpu().numpy()[0]
                    )
                    decision = runtime.begin_invalid(
                        current_xyz,
                        corrupted_token=observed_anchor.native_token[:, -1],
                    )
                else:
                    decision = runtime.begin_invalid(
                        current_xyz,
                        allow_geometry_fallback=visual_validity_fsm is not None,
                    )
                policy_batch = decision.attach(observed_deploy)
                if action_residual_head is None:
                    action = _policy_action(policy, policy_batch)
                else:
                    if action_residual_head.training or any(
                        parameter.requires_grad
                        for parameter in action_residual_head.parameters()
                    ):
                        raise ValueError("action residual head must be frozen and in eval mode")
                    output = policy.predict_action_chunk(dict(policy_batch))
                    base_action = output.action[:, :P1_POLICY_ACTION_STEPS].detach().float()
                    action_latent = output.latent[:, :P1_POLICY_ACTION_STEPS].detach().float()
                    action = _apply_action_residual_head(
                        action_residual_head,
                        base_action=base_action,
                        action_latent=action_latent,
                        posterior_token=decision.policy_token,
                        missing_age=int(decision.missing_age),
                    ).detach().float()
                    if tuple(action.shape) != (1, P1_POLICY_ACTION_STEPS, 4):
                        raise ValueError(
                            "action residual head returned unexpected shape "
                            f"{tuple(action.shape)}"
                        )
                    if not bool(torch.isfinite(action).all()):
                        raise ValueError("action residual head returned non-finite actions")
                if clean_action is not None:
                    active_action_mse.append(
                        float(torch.mean((action - clean_action).square()).cpu())
                    )
                active_decisions.append(decision_index)
            else:
                use_forced_prefix = (
                    decision_index < drop_start
                    and forced_prefix_tokens is not None
                    and forced_prefix_visual is not None
                )
                if native_anchor is None:
                    native_anchor = extract_policy_anchor_observation(
                        policy, observed_deploy
                    )
                selected_points = (
                    native_anchor.points[:, -1].detach().float().cpu().numpy()[0]
                )
                global_visual_token = native_anchor.global_visual_token[:, -1].float()
                if use_forced_prefix:
                    expected_token_prefix = (
                        drop_start,
                        1,
                        int(policy.num_keypoints),
                        int(policy.model_dim),
                    )
                    expected_visual_prefix = (
                        drop_start,
                        1,
                        int(policy.num_keypoints),
                        int(policy.vision_dim),
                    )
                    if forced_prefix_tokens.shape != expected_token_prefix:
                        raise ValueError(
                            "forced native-token prefix has shape "
                            f"{forced_prefix_tokens.shape}, expected {expected_token_prefix}"
                        )
                    if forced_prefix_visual.shape != expected_visual_prefix:
                        raise ValueError(
                            "forced sampled-visual prefix has shape "
                            f"{forced_prefix_visual.shape}, expected {expected_visual_prefix}"
                        )
                    native_token = torch.from_numpy(
                        np.array(forced_prefix_tokens[decision_index], copy=True)
                    ).to(device)
                    sampled_visual = torch.from_numpy(
                        np.array(forced_prefix_visual[decision_index], copy=True)
                    ).to(device)
                else:
                    native_token = native_anchor.native_token[:, -1].float()
                    sampled_visual = native_anchor.sampled_visual[:, -1].float()
                if decision_index < drop_start:
                    prefix_native_tokens.append(native_token.cpu().numpy().copy())
                    prefix_sampled_visual.append(sampled_visual.cpu().numpy().copy())
                decision = runtime.begin_valid(
                    native_token,
                    sampled_visual,
                    global_visual_token,
                )
                if decision.override_required:
                    raise RuntimeError("valid/recovery decision attempted a token override")
                action = (
                    torch.from_numpy(
                        np.array(forced_prefix_actions[decision_index], copy=True)
                    )
                    .unsqueeze(0)
                    .to(device)
                    if use_forced_prefix and forced_prefix_actions is not None
                    else _policy_action(policy, observed_deploy)
                )
                if decision_index == drop_start + drop_duration and method != "clean_native":
                    recovery_exact = (
                        "grounded_token_override" not in decision.attach(observed_deploy)
                        and runtime.missing_age == 0
                    )

            raw_actions = action[0].cpu().numpy()
            if decision_index < drop_start and forced_prefix_actions is not None:
                expected_prefix = (drop_start, P1_POLICY_ACTION_STEPS, 4)
                if forced_prefix_actions.shape != expected_prefix:
                    raise ValueError(
                        f"forced prefix must have shape {expected_prefix}, got "
                        f"{forced_prefix_actions.shape}"
                    )
                raw_actions = forced_prefix_actions[decision_index]
            clipped = np.clip(
                raw_actions,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
            action_blocks.append(clipped.copy())
            for substep_index, raw_action in enumerate(clipped):
                observation, reward, terminated, truncated, info = env.step(raw_action)
                if moving_goal_seed is not None:
                    if not isinstance(info, dict) or "moving_goal_position" not in info:
                        raise RuntimeError("moving-goal step did not expose its target position")
                    moving_goal_positions.append(
                        np.asarray(info["moving_goal_position"], dtype=np.float32).copy()
                    )
                rewards.append(float(reward))
                success = float(info.get("success", 0.0)) if isinstance(info, dict) else 0.0
                successes.append(success)
                if success > 0.0 and first_success_step is None:
                    first_success_step = len(rewards)
                clean_record = collect_observation(
                    env,
                    observation,
                    camera_id,
                    intrinsic,
                    nominal_extrinsic,
                    width,
                    height,
                    obs_mode,
                    0,
                    0,
                )
                clean_history.append(clean_record)
                observed_history.append(
                    _corrupt_record(
                        clean_record,
                        radius_px=radius_px,
                        value_uint8=value_uint8,
                        center_key=occlusion_center_key,
                    )
                    if corruption_active
                    else {
                        name: np.asarray(value).copy()
                        for name, value in clean_record.items()
                    }
                )
                _emit_rollout_trace(
                    trace_callback,
                    raw_step=len(rewards),
                    decision=decision_index,
                    substep=substep_index,
                    corruption_active=corruption_active,
                    prior_active=active,
                    clean_record=clean_record,
                    observed_record=observed_history[-1],
                    selected_points=selected_points,
                    action=raw_action,
                    reward=float(reward),
                    cumulative_reward=float(np.sum(rewards)),
                    success=success,
                    moving_goal_position=(
                        None if not moving_goal_positions else moving_goal_positions[-1]
                    ),
                )
                if terminated or truncated:
                    raise RuntimeError(
                        "environment terminated before the fixed policy horizon"
                    )
            _register_action(runtime, clipped, device=device)
            decision_index += 1

        if method != "clean_native" and visual_validity_fsm is None:
            expected_active = list(range(drop_start, drop_start + drop_duration))
            if active_decisions != expected_active:
                raise RuntimeError(
                    f"incomplete dropout decisions: {active_decisions} != {expected_active}"
                )
            if not recovery_exact:
                raise RuntimeError("first recovery decision did not hard-bypass prediction")
        if drop_onset_state_digest is None:
            raise RuntimeError("drop-onset physical state was not captured")
        actions_array = np.ascontiguousarray(np.stack(action_blocks), dtype=np.float32)
        pre_drop = actions_array[:drop_start]
        if forced_prefix_tokens is not None and forced_prefix_visual is not None:
            prefix_native_tokens = [
                np.asarray(value).copy() for value in forced_prefix_tokens
            ]
            prefix_sampled_visual = [
                np.asarray(value).copy() for value in forced_prefix_visual
            ]
        if len(prefix_native_tokens) != drop_start or len(prefix_sampled_visual) != drop_start:
            raise RuntimeError("pre-drop native history was not collected exactly once")
        native_tail = np.ascontiguousarray(
            np.stack(prefix_native_tokens[-3:]), dtype=np.float32
        )
        visual_tail = np.ascontiguousarray(
            np.stack(prefix_sampled_visual[-2:]), dtype=np.float32
        )
        post_drop_successes = successes[drop_start * P1_POLICY_ACTION_STEPS :]
        post_drop_streak = _maximum_success_streak(post_drop_successes)
        moving_goal_trajectory: np.ndarray | None = None
        moving_goal_trajectory_sha256: str | None = None
        moving_goal_path_length_m = 0.0
        moving_goal_max_displacement_m = 0.0
        moving_goal_start_xyz: list[float] | None = None
        moving_goal_final_xyz: list[float] | None = None
        if moving_goal_seed is not None:
            moving_goal_trajectory = np.ascontiguousarray(
                np.stack(moving_goal_positions), dtype=np.float32
            )
            expected_shape = (len(rewards) + 1, 3)
            if moving_goal_trajectory.shape != expected_shape:
                raise RuntimeError(
                    "moving-goal trajectory has shape "
                    f"{moving_goal_trajectory.shape}, expected {expected_shape}"
                )
            moving_goal_trajectory_sha256 = _sha256_bytes(
                moving_goal_trajectory.tobytes()
            )
            increments = np.diff(moving_goal_trajectory, axis=0)
            moving_goal_path_length_m = float(
                np.linalg.norm(increments, axis=1).sum()
            )
            moving_goal_max_displacement_m = float(
                np.linalg.norm(
                    moving_goal_trajectory - moving_goal_trajectory[0], axis=1
                ).max()
            )
            moving_goal_start_xyz = moving_goal_trajectory[0].tolist()
            moving_goal_final_xyz = moving_goal_trajectory[-1].tolist()
        return {
            "method": method if result_method is None else str(result_method),
            "model_seed": model_seed,
            "task": task,
            "task_index": task_index,
            "environment_seed": environment_seed,
            "reset_digest": reset_digest,
            "camera_sha256": camera_sha,
            "camera_rotation_axis_world": [
                float(value) for value in camera_rotation_axis_world
            ],
            "camera_rotation_deg": float(camera_rotation_deg),
            "visible_grounding_mode": visible_grounding_mode,
            "moving_goal_enabled": moving_goal_seed is not None,
            "moving_goal_seed": (
                None if moving_goal_seed is None else int(moving_goal_seed)
            ),
            "moving_goal_amplitude_xy": (
                None
                if moving_goal_seed is None
                else [float(value) for value in moving_goal_amplitude_xy]
            ),
            "moving_goal_period_raw_steps": (
                None
                if moving_goal_seed is None
                else float(moving_goal_period_raw_steps)
            ),
            "moving_goal_trajectory_sha256": moving_goal_trajectory_sha256,
            "moving_goal_trajectory_points": (
                0
                if moving_goal_trajectory is None
                else int(moving_goal_trajectory.shape[0])
            ),
            "moving_goal_path_length_m": moving_goal_path_length_m,
            "moving_goal_max_displacement_m": moving_goal_max_displacement_m,
            "moving_goal_start_xyz": moving_goal_start_xyz,
            "moving_goal_final_xyz": moving_goal_final_xyz,
            "schedule_sha256": _stable_hash(
                {"start": drop_start, "duration": drop_duration, "radius": radius_px}
            ),
            "drop_onset_state_sha256": drop_onset_state_digest,
            "pre_drop_action_sha256": _sha256_bytes(pre_drop.tobytes()),
            "pre_drop_action_round6_sha256": _sha256_bytes(
                np.ascontiguousarray(np.round(pre_drop, decimals=6), dtype=np.float32).tobytes()
            ),
            "pre_drop_native_tail_round6_sha256": _sha256_bytes(
                np.ascontiguousarray(np.round(native_tail, decimals=6), dtype=np.float32).tobytes()
            ),
            "pre_drop_visual_tail_round6_sha256": _sha256_bytes(
                np.ascontiguousarray(np.round(visual_tail, decimals=6), dtype=np.float32).tobytes()
            ),
            "pre_drop_action_values": pre_drop.reshape(-1).tolist(),
            "_prefix_native_tokens": np.ascontiguousarray(
                np.stack(prefix_native_tokens), dtype=np.float32
            ),
            "_prefix_sampled_visual": np.ascontiguousarray(
                np.stack(prefix_sampled_visual), dtype=np.float32
            ),
            "all_action_sha256": _sha256_bytes(actions_array.tobytes()),
            "drop_start": drop_start,
            "drop_duration": drop_duration,
            "steps": len(rewards),
            "decisions": decision_index,
            "success": float(max(successes, default=0.0)),
            "final_success": float(successes[-1]) if successes else 0.0,
            "success_fraction": float(np.mean(successes)) if successes else 0.0,
            "maximum_success_streak": _maximum_success_streak(successes),
            "post_drop_success_fraction": (
                float(np.mean(post_drop_successes)) if post_drop_successes else 0.0
            ),
            "post_drop_maximum_success_streak": post_drop_streak,
            "sustained_success": float(post_drop_streak >= P1_POLICY_ACTION_STEPS),
            "reward_sum": float(np.sum(rewards)),
            "first_success_step": first_success_step,
            "action_distance": float(np.mean(active_action_mse)) if active_action_mse else 0.0,
            "active_decisions_observed": len(active_decisions),
            "recovery_exact_native_bypass": (
                True if method == "clean_native" else recovery_exact
            ),
            "selector_enabled": visual_validity_fsm is not None,
            "selector_prior_decisions": int(selector_prior_decisions),
            "selector_true_positive": int(selector_true_positive),
            "selector_false_positive": int(selector_false_positive),
            "selector_false_negative": int(selector_false_negative),
            "selector_true_negative": int(selector_true_negative),
            "selector_reasons": dict(sorted(selector_reasons.items())),
            "selector_trace": selector_trace,
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def _percentile(samples: np.ndarray, confidence: float = 0.95) -> list[float]:
    tail = (1.0 - confidence) * 50.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return [float(low), float(high)]


def _exact_mcnemar(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, object]:
    better = int(np.sum((candidate == 1.0) & (baseline == 0.0)))
    worse = int(np.sum((candidate == 0.0) & (baseline == 1.0)))
    discordant = better + worse
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(better, worse) + 1))
        p_value = min(1.0, 2.0 * tail / (2.0**discordant))
    return {
        "candidate_only_success": better,
        "baseline_only_success": worse,
        "discordant": discordant,
        "two_sided_exact_p": float(p_value),
    }


def paired_comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate: str,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    indexed = {(int(row["physical_episode"]), str(row["method"])): row for row in rows}
    episodes = sorted(
        episode for episode, method in indexed if method == candidate
    )
    if not episodes or any((episode, baseline) not in indexed for episode in episodes):
        raise ValueError(f"incomplete paired comparison {candidate} vs {baseline}")
    candidate_success = np.asarray(
        [indexed[(episode, candidate)]["success"] for episode in episodes], dtype=np.float64
    )
    baseline_success = np.asarray(
        [indexed[(episode, baseline)]["success"] for episode in episodes], dtype=np.float64
    )
    candidate_final_success = np.asarray(
        [indexed[(episode, candidate)]["final_success"] for episode in episodes],
        dtype=np.float64,
    )
    baseline_final_success = np.asarray(
        [indexed[(episode, baseline)]["final_success"] for episode in episodes],
        dtype=np.float64,
    )
    candidate_success_fraction = np.asarray(
        [indexed[(episode, candidate)]["success_fraction"] for episode in episodes],
        dtype=np.float64,
    )
    baseline_success_fraction = np.asarray(
        [indexed[(episode, baseline)]["success_fraction"] for episode in episodes],
        dtype=np.float64,
    )
    candidate_success_streak = np.asarray(
        [indexed[(episode, candidate)]["maximum_success_streak"] for episode in episodes],
        dtype=np.float64,
    )
    baseline_success_streak = np.asarray(
        [indexed[(episode, baseline)]["maximum_success_streak"] for episode in episodes],
        dtype=np.float64,
    )
    candidate_distance = np.asarray(
        [indexed[(episode, candidate)]["action_distance"] for episode in episodes],
        dtype=np.float64,
    )
    baseline_distance = np.asarray(
        [indexed[(episode, baseline)]["action_distance"] for episode in episodes],
        dtype=np.float64,
    )
    candidate_reward = np.asarray(
        [indexed[(episode, candidate)]["reward_sum"] for episode in episodes],
        dtype=np.float64,
    )
    baseline_reward = np.asarray(
        [indexed[(episode, baseline)]["reward_sum"] for episode in episodes],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(episodes), size=(bootstrap_samples, len(episodes)))
    success_samples = np.mean((candidate_success - baseline_success)[draws], axis=1)
    final_success_samples = np.mean(
        (candidate_final_success - baseline_final_success)[draws], axis=1
    )
    success_fraction_samples = np.mean(
        (candidate_success_fraction - baseline_success_fraction)[draws], axis=1
    )
    success_streak_samples = np.mean(
        (candidate_success_streak - baseline_success_streak)[draws], axis=1
    )
    reward_samples = np.mean((candidate_reward - baseline_reward)[draws], axis=1)
    base_distance_samples = np.mean(baseline_distance[draws], axis=1)
    candidate_distance_samples = np.mean(candidate_distance[draws], axis=1)
    relative_samples = np.divide(
        base_distance_samples - candidate_distance_samples,
        base_distance_samples,
        out=np.zeros_like(base_distance_samples),
        where=base_distance_samples > 0.0,
    )
    baseline_mean = float(np.mean(baseline_distance))
    relative = (
        (baseline_mean - float(np.mean(candidate_distance))) / baseline_mean
        if baseline_mean > 0.0
        else 0.0
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "physical_episode_count": len(episodes),
        "success_delta": float(np.mean(candidate_success - baseline_success)),
        "success_delta_ci": _percentile(success_samples),
        "final_success_delta": float(
            np.mean(candidate_final_success - baseline_final_success)
        ),
        "final_success_delta_ci": _percentile(final_success_samples),
        "success_fraction_delta": float(
            np.mean(candidate_success_fraction - baseline_success_fraction)
        ),
        "success_fraction_delta_ci": _percentile(success_fraction_samples),
        "maximum_success_streak_delta": float(
            np.mean(candidate_success_streak - baseline_success_streak)
        ),
        "maximum_success_streak_delta_ci": _percentile(success_streak_samples),
        "reward_delta": float(np.mean(candidate_reward - baseline_reward)),
        "reward_delta_ci": _percentile(reward_samples),
        "action_distance_candidate": float(np.mean(candidate_distance)),
        "action_distance_baseline": baseline_mean,
        "action_distance_relative_improvement": relative,
        "action_distance_relative_improvement_ci": _percentile(relative_samples),
        "mcnemar": _exact_mcnemar(candidate_success, baseline_success),
    }


def clustered_seed_macro_comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    model_seeds: Sequence[int],
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Compare model seeds while resampling physical episodes as clusters."""

    seeds = tuple(int(value) for value in model_seeds)
    if not seeds:
        raise ValueError("clustered seed-macro comparison requires model seeds")
    if baseline not in {"geometry_hold", "matched_no_action"}:
        raise ValueError(f"unsupported clustered baseline: {baseline}")
    indexed = {(int(row["physical_episode"]), str(row["method"])): row for row in rows}
    episodes = sorted(
        episode
        for episode, method in indexed
        if method == f"complementary_factual_s{seeds[0]}"
    )
    if not episodes:
        raise ValueError("clustered seed-macro comparison has no physical episodes")

    def values(metric: str, *, factual: bool) -> np.ndarray:
        columns: list[list[float]] = []
        for model_seed in seeds:
            if factual:
                method = f"complementary_factual_s{model_seed}"
            elif baseline == "matched_no_action":
                method = f"complementary_no_action_s{model_seed}"
            else:
                method = "geometry_hold"
            if any((episode, method) not in indexed for episode in episodes):
                raise ValueError(
                    f"incomplete clustered comparison for model seed {model_seed}: {method}"
                )
            columns.append(
                [float(indexed[(episode, method)][metric]) for episode in episodes]
            )
        return np.asarray(columns, dtype=np.float64).T

    candidate_success = values("success", factual=True)
    baseline_success = values("success", factual=False)
    candidate_final_success = values("final_success", factual=True)
    baseline_final_success = values("final_success", factual=False)
    candidate_success_fraction = values("success_fraction", factual=True)
    baseline_success_fraction = values("success_fraction", factual=False)
    candidate_success_streak = values("maximum_success_streak", factual=True)
    baseline_success_streak = values("maximum_success_streak", factual=False)
    candidate_reward = values("reward_sum", factual=True)
    baseline_reward = values("reward_sum", factual=False)
    candidate_distance = values("action_distance", factual=True)
    baseline_distance = values("action_distance", factual=False)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(episodes), size=(bootstrap_samples, len(episodes)))
    success_by_episode = np.mean(candidate_success - baseline_success, axis=1)
    reward_by_episode = np.mean(candidate_reward - baseline_reward, axis=1)
    success_samples = np.mean(success_by_episode[draws], axis=1)
    final_success_by_episode = np.mean(
        candidate_final_success - baseline_final_success, axis=1
    )
    success_fraction_by_episode = np.mean(
        candidate_success_fraction - baseline_success_fraction, axis=1
    )
    success_streak_by_episode = np.mean(
        candidate_success_streak - baseline_success_streak, axis=1
    )
    final_success_samples = np.mean(final_success_by_episode[draws], axis=1)
    success_fraction_samples = np.mean(success_fraction_by_episode[draws], axis=1)
    success_streak_samples = np.mean(success_streak_by_episode[draws], axis=1)
    reward_samples = np.mean(reward_by_episode[draws], axis=1)
    candidate_distance_samples = np.mean(candidate_distance[draws, :], axis=(1, 2))
    baseline_distance_samples = np.mean(baseline_distance[draws, :], axis=(1, 2))
    relative_samples = np.divide(
        baseline_distance_samples - candidate_distance_samples,
        baseline_distance_samples,
        out=np.zeros_like(baseline_distance_samples),
        where=baseline_distance_samples > 0.0,
    )
    candidate_distance_mean = float(np.mean(candidate_distance))
    baseline_distance_mean = float(np.mean(baseline_distance))
    relative = (
        (baseline_distance_mean - candidate_distance_mean) / baseline_distance_mean
        if baseline_distance_mean > 0.0
        else 0.0
    )
    return {
        "candidate": "seed_macro_factual",
        "baseline": baseline,
        "model_seeds": list(seeds),
        "model_seed_count": len(seeds),
        "physical_episode_count": len(episodes),
        "independent_unit": "physical_episode",
        "success_delta": float(np.mean(success_by_episode)),
        "success_delta_ci": _percentile(success_samples),
        "final_success_delta": float(np.mean(final_success_by_episode)),
        "final_success_delta_ci": _percentile(final_success_samples),
        "success_fraction_delta": float(np.mean(success_fraction_by_episode)),
        "success_fraction_delta_ci": _percentile(success_fraction_samples),
        "maximum_success_streak_delta": float(np.mean(success_streak_by_episode)),
        "maximum_success_streak_delta_ci": _percentile(success_streak_samples),
        "reward_delta": float(np.mean(reward_by_episode)),
        "reward_delta_ci": _percentile(reward_samples),
        "action_distance_candidate": candidate_distance_mean,
        "action_distance_baseline": baseline_distance_mean,
        "action_distance_relative_improvement": relative,
        "action_distance_relative_improvement_ci": _percentile(relative_samples),
    }


def _add_seed_macro_comparisons(
    report: dict[str, object],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    rows = report.get("rows")
    seeds = report.get("model_seeds")
    methods = set(report.get("methods", []))
    if not isinstance(rows, list) or not isinstance(seeds, list):
        raise TypeError("rollout report is missing rows or model seeds")
    required = {
        *(f"complementary_factual_s{int(value)}" for value in seeds),
        *(f"complementary_no_action_s{int(value)}" for value in seeds),
        "geometry_hold",
    }
    if not required.issubset(methods):
        report["seed_macro_comparisons"] = {}
        return
    report["seed_macro_comparisons"] = {
        "factual_vs_geometry_hold": clustered_seed_macro_comparison(
            rows,
            model_seeds=[int(value) for value in seeds],
            baseline="geometry_hold",
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + 1000,
        ),
        "factual_vs_matched_no_action": clustered_seed_macro_comparison(
            rows,
            model_seeds=[int(value) for value in seeds],
            baseline="matched_no_action",
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + 1001,
        ),
    }


def _summarize_existing_report(path: str | Path) -> None:
    report_path = _resolve(path)
    report = _load_json(report_path)
    if (
        report.get("protocol") not in SUPPORTED_PROTOCOLS
        or report.get("pairing_audit") != "passed"
    ):
        raise ValueError("existing report is not a paired P1 rollout artifact")
    config = _load_json(str(report["config"]))
    _add_seed_macro_comparisons(
        report,
        bootstrap_samples=int(config["bootstrap_samples"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, report_path)
    print(f"updated seed-macro summary: {report_path}")


def _validate_pairing(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["physical_episode"]), []).append(row)
    for episode, panel in grouped.items():
        for field in (
            "reset_digest",
            "camera_sha256",
            "schedule_sha256",
            "drop_onset_state_sha256",
            "pre_drop_native_tail_round6_sha256",
            "pre_drop_visual_tail_round6_sha256",
        ):
            values = {row[field] for row in panel}
            if len(values) != 1:
                raise RuntimeError(
                    f"paired episode {episode} differs before intervention in {field}"
                )
        reference = np.asarray(panel[0]["pre_drop_action_values"], dtype=np.float32)
        for row in panel[1:]:
            current = np.asarray(row["pre_drop_action_values"], dtype=np.float32)
            if not np.allclose(current, reference, rtol=0.0, atol=1.0e-6):
                maximum = float(np.max(np.abs(current - reference)))
                raise RuntimeError(
                    f"paired episode {episode} pre-drop actions differ by {maximum:.9g}"
                )
        if not all(bool(row["recovery_exact_native_bypass"]) for row in panel):
            raise RuntimeError(f"paired episode {episode} failed native recovery bypass")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/lewm/eval/p1_native_policy_rollout_development.json",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--model-seeds", type=int, nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--summarize-existing-report")
    args = parser.parse_args(argv)

    if args.summarize_existing_report is not None:
        _summarize_existing_report(args.summarize_existing_report)
        return 0

    os.environ.setdefault("MUJOCO_GL", "egl")
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    _validate_config(config)
    is_confirmation = (
        config.get("protocol") == JOINT_DRIFT_DROP_CONFIRMATION_PROTOCOL
    )
    if is_confirmation and any(
        value is not None for value in (args.episodes, args.model_seeds, args.methods)
    ):
        parser.error(
            "joint confirmation does not allow --episodes, --model-seeds, or --methods overrides"
        )
    source_config_path = _resolve(str(config["source_fresh_gate_config"]))
    source = _load_json(source_config_path)
    p1 = source["p1"]
    if not isinstance(p1, Mapping):
        raise TypeError("source P1 config is malformed")
    training_contract = source["frozen_predictor_training_contract"]
    if not isinstance(training_contract, Mapping):
        raise TypeError("source training contract is malformed")
    expected_config_path = _resolve(str(training_contract["config"]))
    expected_config = _load_json(expected_config_path)
    if sha256_file(expected_config_path) != training_contract["config_sha256"]:
        raise ValueError("frozen predictor training config SHA mismatch")

    policy_path = _resolve(str(p1["policy_checkpoint"]))
    if sha256_file(policy_path) != p1["policy_checkpoint_sha256"]:
        raise ValueError("P1 policy checkpoint SHA mismatch")
    corrector_path = _resolve(str(p1["corrector_checkpoint"]))
    if sha256_file(corrector_path) != p1["corrector_checkpoint_sha256"]:
        raise ValueError("P1 corrector checkpoint SHA mismatch")

    if is_confirmation:
        selected_seeds = tuple(
            int(value) for value in config["full_model_seeds_after_development_go"]
        )
    else:
        selected_seeds = tuple(
            args.model_seeds
            if args.model_seeds is not None
            else config["development_model_seeds"]
        )
    frozen_records = source["frozen_predictors"]
    if not isinstance(frozen_records, list):
        raise TypeError("source frozen predictor records are malformed")
    by_seed = {int(record["seed"]): record for record in frozen_records}
    unknown = set(selected_seeds).difference(by_seed)
    if unknown:
        raise ValueError(f"unknown frozen model seeds: {sorted(unknown)}")
    device = choose_device(args.device)
    policy, policy_args = load_action_chunk_policy(policy_path, device)
    embedded_corrector_path = _resolve(str(policy_args.get("fcpg_checkpoint", "")))
    if embedded_corrector_path != corrector_path:
        raise ValueError(
            "P1 policy embedded corrector differs from the validated corrector: "
            f"{embedded_corrector_path} != {corrector_path}"
        )
    policy.set_reliability_override("ones")
    policy.requires_grad_(False).eval()
    pairs = {}
    for seed in selected_seeds:
        record = by_seed[int(seed)]
        pairs[int(seed)] = load_frozen_p1_native_checkpoint(
            FrozenP1NativeCheckpointSpec(
                seed=int(seed),
                path=_resolve(str(record["checkpoint"])),
                sha256=str(record["sha256"]),
            ),
            expected_config=expected_config,
            device=device,
        )

    total_episodes = int(config["physical_episode_count"])
    episodes = total_episodes if args.episodes is None else int(args.episodes)
    if not 0 < episodes <= total_episodes:
        parser.error(f"--episodes must lie in [1,{total_episodes}]")
    output_dir = _resolve(
        args.output_dir if args.output_dir is not None else str(config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "policy_rollout_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    if is_confirmation:
        methods = [
            method
            for method in config["methods"]
            if method in BASELINE_METHODS
        ]
    else:
        methods = list(BASELINE_METHODS)
    for seed in selected_seeds:
        methods.extend(
            (f"complementary_factual_s{seed}", f"complementary_no_action_s{seed}")
        )
    development_ablation_methods = [] if is_confirmation else [
        f"complementary_factual_trusted_global_s{seed}"
        for seed in selected_seeds
    ]
    if args.methods is not None:
        unknown_methods = set(args.methods).difference(
            [*methods, *development_ablation_methods]
        )
        if unknown_methods:
            parser.error(f"unknown --methods: {sorted(unknown_methods)}")
        methods = list(args.methods)
    rows: list[dict[str, object]] = []
    for episode in range(episodes):
        shared_prefix: np.ndarray | None = None
        shared_prefix_tokens: np.ndarray | None = None
        shared_prefix_visual: np.ndarray | None = None
        for method in methods:
            suffix = method.rsplit("_s", 1)
            model_seed = int(suffix[1]) if len(suffix) == 2 else int(selected_seeds[0])
            pair = pairs.get(model_seed)
            result = rollout_once(
                policy,
                policy_args,
                pair=pair,
                method=method,
                model_seed=model_seed,
                task=str(config["task"]),
                camera=str(config["camera"]),
                width=int(config["width"]),
                height=int(config["height"]),
                benchmark_seed=int(config["benchmark_seed"]),
                task_index=int(config["task_instance_indices"][episode]),
                environment_seed=int(config["environment_seed_base"]) + episode,
                drop_start=int(config["drop_start_decisions"][episode]),
                drop_duration=int(config["drop_duration_decisions"]),
                horizon=int(config["horizon_raw_steps"]),
                radius_px=int(config["occlusion_radius_px"]),
                value_uint8=int(config["occlusion_value_uint8"]),
                device=device,
                camera_rotation_axis_world=config.get(
                    "camera_rotation_axis_world", (0.0, 0.0, 1.0)
                ),
                camera_rotation_deg=float(config.get("camera_rotation_deg", 0.0)),
                forced_prefix_actions=shared_prefix,
                forced_prefix_tokens=shared_prefix_tokens,
                forced_prefix_visual=shared_prefix_visual,
            )
            prefix_tokens = result.pop("_prefix_native_tokens")
            prefix_visual = result.pop("_prefix_sampled_visual")
            row = {"physical_episode": episode, **result}
            if shared_prefix is None:
                if method != "clean_native":
                    raise RuntimeError(
                        "clean_native must run first to define the shared prefix"
                    )
                shared_prefix = np.asarray(
                    result["pre_drop_action_values"], dtype=np.float32
                ).reshape(
                    int(config["drop_start_decisions"][episode]),
                    P1_POLICY_ACTION_STEPS,
                    4,
                )
                shared_prefix_tokens = np.asarray(prefix_tokens, dtype=np.float32)
                shared_prefix_visual = np.asarray(prefix_visual, dtype=np.float32)
            rows.append(row)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(
                f"episode={episode:03d} method={method:35s} "
                f"success={row['success']:.0f} reward={row['reward_sum']:.2f} "
                f"action_mse={row['action_distance']:.6f}",
                flush=True,
            )

    _validate_pairing(rows)
    comparisons: dict[str, object] = {}
    bootstrap_samples = int(config["bootstrap_samples"])
    bootstrap_seed = int(config["bootstrap_seed"])
    for offset, seed in enumerate(selected_seeds):
        factual = f"complementary_factual_s{seed}"
        trusted_global = f"complementary_factual_trusted_global_s{seed}"
        no_action = f"complementary_no_action_s{seed}"
        if factual not in methods:
            continue
        for baseline in ("geometry_hold", no_action, "measurement", "hold_last"):
            if baseline not in methods:
                continue
            key = f"{factual}_vs_{baseline}"
            comparisons[key] = paired_comparison(
                rows,
                candidate=factual,
                baseline=baseline,
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed + offset * 10 + len(comparisons),
            )
        if trusted_global in methods:
            key = f"{factual}_vs_{trusted_global}"
            comparisons[key] = paired_comparison(
                rows,
                candidate=factual,
                baseline=trusted_global,
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed + offset * 10 + len(comparisons),
            )

    method_summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        method_summary[method] = {
            "success_rate": float(np.mean([row["success"] for row in selected])),
            "reward_mean": float(np.mean([row["reward_sum"] for row in selected])),
            "action_distance_mean": float(
                np.mean([row["action_distance"] for row in selected])
            ),
        }
    report = {
        "protocol": str(config["protocol"]),
        "classification": str(
            config.get("classification", "development_only_not_confirmatory")
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_fresh_gate_config": str(source_config_path),
        "policy_checkpoint": str(policy_path),
        "policy_checkpoint_sha256": sha256_file(policy_path),
        "corrector_checkpoint": str(corrector_path),
        "corrector_checkpoint_sha256": sha256_file(corrector_path),
        "model_seeds": list(selected_seeds),
        "episodes": episodes,
        "methods": methods,
        "method_summary": method_summary,
        "comparisons": comparisons,
        "pairing_audit": "passed",
        "incremental_progress": str(progress_path),
        "claim_boundaries": config["claim_boundaries"],
        "rows": rows,
    }
    _add_seed_macro_comparisons(
        report,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    csv_path = output_dir / "policy_rollout_long_table.csv"
    report_path = output_dir / "policy_rollout_report.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"csv: {csv_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
