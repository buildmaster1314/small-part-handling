#!/usr/bin/env python3
"""Collect on-policy Pick states with restored scripted-expert labels.

The learner owns the main trajectory.  At an invalid-visual decision, this
collector snapshots the exact environment and process RNG state, rolls the
scripted expert for five raw steps to obtain a training label, restores the
snapshot, and only then executes the learner's action block.  Consequently an
expert query cannot move, randomize, or otherwise alter the learner rollout.
"""

from __future__ import annotations

import argparse
import copy
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_known_drop_policy_evaluator import (  # noqa: E402
    deployment_policy_batch,
    extract_policy_anchor_observation,
)
from fcpg.lewm_p1_native_complementary import (  # noqa: E402
    FrozenP1NativeCheckpointSpec,
    load_frozen_p1_native_checkpoint,
)
from fcpg.lewm_p1_native_policy_runtime import (  # noqa: E402
    P1NativeComplementaryPolicyRuntime,
)
from fcpg.metaworld_snapshot import (  # noqa: E402
    capture_metaworld_snapshot,
    restore_metaworld_snapshot,
)
from fcpg.metaworld_utils import make_env, make_scripted_policy  # noqa: E402
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    set_fixed_camera_pose_offset,
)
from scripts.evaluate_action_chunk_rollout import (  # noqa: E402
    choose_device,
    collect_observation,
    history_batch,
)
from scripts.evaluate_p1_native_policy_rollout import _corrupt_record  # noqa: E402


ACTION_STEPS = 5
ACTION_DIM = 4
COLLECTION_METHODS = (
    "complementary_factual",
    "complementary_no_action",
    "geometry_hold",
)
DEFAULT_CONFIG = "config/lewm/eval/posterior_dagger_pick_dev.json"


@dataclass(frozen=True, slots=True)
class CameraPoseCondition:
    """One fixed post-calibration camera rotation."""

    axis_world: np.ndarray
    rotation_deg: float


@dataclass(frozen=True, slots=True)
class LearnerBlockResult:
    """Observed outcome of exactly one five-action learner block."""

    observation: np.ndarray
    rewards: tuple[float, ...]
    successes: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BehaviorHeadContract:
    """Expected training endpoint for the policy that collected this round."""

    required: bool
    expected_seed: int
    expected_steps: int
    expected_dagger_iteration: int | None


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _dagger_api() -> Any:
    """Load the separately owned DAgger data/head API only when needed."""

    from fcpg import lewm_posterior_dagger as api

    required = (
        "ACTION_STEPS",
        "METHODS",
        "PROTOCOL",
        "PosteriorDaggerArrays",
        "load_posterior_dagger_checkpoint",
        "method_index",
        "save_posterior_dagger_arrays",
    )
    missing = [name for name in required if not hasattr(api, name)]
    if missing:
        raise RuntimeError(f"posterior DAgger API is incomplete: {missing}")
    if int(api.ACTION_STEPS) != ACTION_STEPS:
        raise ValueError("posterior DAgger action horizon is not five")
    if tuple(api.METHODS) != COLLECTION_METHODS:
        raise ValueError("posterior DAgger method ordering drifted")
    return api


def _split_config(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    splits = config.get("splits")
    if not isinstance(splits, Mapping) or not isinstance(splits.get(name), Mapping):
        raise ValueError(f"unknown collection split {name!r}")
    return splits[name]


def parse_camera_pose_condition(config: Mapping[str, object]) -> CameraPoseCondition:
    """Validate the required joint camera-drift condition."""

    if config.get("camera_condition") != "fixed_rotation":
        raise ValueError("posterior DAgger requires camera_condition=fixed_rotation")
    axis = np.asarray(config.get("camera_rotation_axis_world"), dtype=np.float64)
    degrees = float(config.get("camera_rotation_deg", 0.0))
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("camera_rotation_axis_world must be one finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("camera_rotation_axis_world must be non-zero")
    if not np.isfinite(degrees) or abs(degrees) <= 0.0:
        raise ValueError("camera_rotation_deg must be finite and non-zero")
    normalized = np.ascontiguousarray(axis / norm, dtype=np.float64)
    normalized.setflags(write=False)
    return CameraPoseCondition(normalized, degrees)


def _validate_config(config: Mapping[str, object], *, protocol: str) -> None:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("posterior DAgger config schema is unsupported")
    if str(config.get("protocol", "")) != protocol:
        raise ValueError("posterior DAgger config protocol is unsupported")
    if str(config.get("task", "")) != "pick-place-v3":
        raise ValueError("this collector is fixed to pick-place-v3")
    if int(config.get("action_steps_per_decision", -1)) != ACTION_STEPS:
        raise ValueError("action_steps_per_decision must equal five")
    duration = int(config.get("drop_duration_decisions", -1))
    if duration <= 0 or duration > 6:
        raise ValueError("drop_duration_decisions must lie in [1,6]")
    start_min = int(config.get("drop_start_min_decision", -1))
    start_max = int(config.get("drop_start_max_decision", -1))
    if start_min < 3 or start_max < start_min:
        raise ValueError("drop starts must be ordered and leave three visible decisions")
    if config.get("known_sensor_validity_only") is not True:
        raise ValueError("posterior DAgger requires the known-validity development gate")
    if str(config.get("occlusion_center_key", "")) != "q_nominal":
        raise ValueError("deployable local interruption must be centered on q_nominal")
    if int(config.get("width", 0)) <= 0 or int(config.get("height", 0)) <= 0:
        raise ValueError("image dimensions must be positive")
    if int(config.get("model_seed", -1)) < 0:
        raise ValueError("model_seed must be non-negative")
    policy_seed = config.get("policy_seed")
    if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
        raise ValueError("this collector requires policy_seed=0")
    if policy_seed != 0:
        raise ValueError("this collector requires policy_seed=0")
    collection_contract(config)
    if "source_fresh_gate_config" not in config:
        raise ValueError("source_fresh_gate_config is required")
    _training_endpoint(config)
    parse_camera_pose_condition(config)
    for name in ("train", "development"):
        split = _split_config(config, name)
        for key in (
            "benchmark_seed",
            "environment_seed_base",
            "task_index_base",
            "episodes",
            "schedule_seed",
        ):
            if int(split.get(key, -1)) < 0:
                raise ValueError(f"split {name!r} has invalid {key!r}")
        if int(split["episodes"]) <= 0:
            raise ValueError(f"split {name!r} must contain episodes")


def _training_endpoint(config: Mapping[str, object]) -> tuple[int, int]:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training must be an object")
    seeds = training.get("seeds")
    if not isinstance(seeds, (list, tuple)):
        raise TypeError("training.seeds must be a list containing one seed")
    if len(seeds) != 1:
        raise ValueError("training.seeds must contain exactly one expected seed")
    seed_value = seeds[0]
    steps_value = training.get("steps")
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise TypeError("training.seeds must contain one integer seed")
    if isinstance(steps_value, bool) or not isinstance(steps_value, int):
        raise TypeError("training.steps must be an integer")
    seed = int(seed_value)
    steps = int(steps_value)
    if seed < 0:
        raise ValueError("training seed must be non-negative")
    if steps <= 0:
        raise ValueError("training.steps must be positive")
    return seed, steps


def behavior_head_contract(
    config: Mapping[str, object],
    *,
    dagger_iteration: int,
    head_checkpoint: str | None,
) -> BehaviorHeadContract:
    """Bind a collection round to exactly the preceding trained head."""

    iteration = int(dagger_iteration)
    if iteration < 0:
        raise ValueError("DAgger iteration must be non-negative")
    expected_seed, expected_steps = _training_endpoint(config)
    if iteration == 0:
        if head_checkpoint is not None:
            raise ValueError("DAgger iteration 0 forbids --head-checkpoint")
        return BehaviorHeadContract(
            required=False,
            expected_seed=expected_seed,
            expected_steps=expected_steps,
            expected_dagger_iteration=None,
        )
    if head_checkpoint is None:
        raise ValueError("DAgger iteration >0 requires --head-checkpoint")
    return BehaviorHeadContract(
        required=True,
        expected_seed=expected_seed,
        expected_steps=expected_steps,
        expected_dagger_iteration=iteration - 1,
    )


def load_behavior_head(
    api: Any,
    config: Mapping[str, object],
    *,
    method: str,
    dagger_iteration: int,
    head_checkpoint: str | None,
    device: torch.device,
) -> tuple[torch.nn.Module | None, Mapping[str, object] | None, BehaviorHeadContract]:
    """Validate and, when required, load the preceding-round behavior head."""

    contract = behavior_head_contract(
        config,
        dagger_iteration=dagger_iteration,
        head_checkpoint=head_checkpoint,
    )
    if not contract.required:
        return None, None, contract
    assert head_checkpoint is not None
    head, metadata = api.load_posterior_dagger_checkpoint(
        _resolve(head_checkpoint),
        device=device,
        expected_method=method,
        expected_seed=contract.expected_seed,
        expected_steps=contract.expected_steps,
        expected_dagger_iteration=contract.expected_dagger_iteration,
    )
    return head, metadata, contract


def collection_contract(config: Mapping[str, object]) -> dict[str, object]:
    """Return minimally typed, config-native fields shared across methods."""

    policy_seed = config.get("policy_seed")
    if (
        isinstance(policy_seed, bool)
        or not isinstance(policy_seed, int)
        or policy_seed != 0
    ):
        raise ValueError("this collector requires policy_seed=0")
    integer_keys = (
        "model_seed",
        "drop_duration_decisions",
        "drop_start_min_decision",
        "drop_start_max_decision",
        "occlusion_radius_px",
        "occlusion_value_uint8",
    )
    integers: dict[str, int] = {"policy_seed": policy_seed}
    for key in integer_keys:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer")
        integers[key] = value

    center_key = config.get("occlusion_center_key")
    if not isinstance(center_key, str) or not center_key:
        raise TypeError("occlusion_center_key must be a non-empty string")
    camera_kind = config.get("camera_condition")
    if not isinstance(camera_kind, str) or not camera_kind:
        raise TypeError("camera_condition must be a non-empty string")
    degrees_value = config.get("camera_rotation_deg")
    if isinstance(degrees_value, bool) or not isinstance(degrees_value, (int, float)):
        raise TypeError("camera_rotation_deg must be numeric")
    degrees = float(degrees_value)
    if not np.isfinite(degrees):
        raise ValueError("camera_rotation_deg must be finite")
    axis_value = config.get("camera_rotation_axis_world")
    if not isinstance(axis_value, (list, tuple)) or len(axis_value) != 3:
        raise TypeError("camera_rotation_axis_world must contain three numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in axis_value):
        raise TypeError("camera_rotation_axis_world must contain three numbers")
    axis = [float(value) for value in axis_value]
    if not np.isfinite(np.asarray(axis, dtype=np.float64)).all():
        raise ValueError("camera_rotation_axis_world must be finite")

    return {
        **integers,
        "occlusion_center_key": center_key,
        "camera_condition": {
            "kind": camera_kind,
            "camera_rotation_deg": degrees,
            "camera_rotation_axis_world": axis,
        },
    }


def apply_fixed_camera_condition(
    env: Any,
    camera_id: int,
    condition: CameraPoseCondition,
    *,
    setter: Callable[..., None] = set_fixed_camera_pose_offset,
) -> None:
    """Apply drift after the caller has captured the nominal extrinsic."""

    base_position = np.asarray(env.model.cam_pos[camera_id], dtype=np.float64).copy()
    base_quaternion = np.asarray(env.model.cam_quat[camera_id], dtype=np.float64).copy()
    setter(
        env.model,
        env.data,
        int(camera_id),
        base_position,
        base_quaternion,
        np.zeros(3, dtype=np.float64),
        condition.axis_world,
        condition.rotation_deg,
    )


def _capture_exact_branch(env: Any, *, include_process_rng: bool) -> Any:
    if include_process_rng is not True:
        raise ValueError("expert branches must include process RNG state")
    return capture_metaworld_snapshot(env, include_process_rng=True)


def _restore_exact_branch(
    env: Any, snapshot: Any, *, restore_process_rng: bool
) -> None:
    if restore_process_rng is not True:
        raise ValueError("expert branches must restore process RNG state")
    restore_metaworld_snapshot(env, snapshot, restore_process_rng=True)


def _bounded_action(env: Any, action: Any, *, owner: str) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
        raise ValueError(f"{owner} returned an invalid action")
    return np.clip(value, env.action_space.low, env.action_space.high).astype(
        np.float32, copy=False
    )


def query_expert_action_block(
    env: Any,
    expert: Any,
    observation: np.ndarray,
    *,
    action_steps: int = ACTION_STEPS,
    capture_snapshot: Callable[..., Any] = _capture_exact_branch,
    restore_snapshot: Callable[..., None] = _restore_exact_branch,
) -> np.ndarray:
    """Query a branch expert without changing the caller's trajectory or RNG."""

    if int(action_steps) != ACTION_STEPS:
        raise ValueError("expert labels must contain exactly five raw actions")
    caller_observation = np.array(observation, dtype=np.float32, copy=True, order="C")
    snapshot = capture_snapshot(env, include_process_rng=True)
    expert_state = copy.deepcopy(vars(expert)) if hasattr(expert, "__dict__") else None
    actions: list[np.ndarray] = []
    try:
        branch_observation = caller_observation.copy()
        for index in range(ACTION_STEPS):
            action = _bounded_action(
                env,
                expert.get_action(branch_observation.copy()),
                owner="scripted expert",
            )
            actions.append(action.copy())
            branch_observation, _reward, terminated, truncated, _info = env.step(action)
            branch_observation = np.asarray(branch_observation, dtype=np.float32)
            if (terminated or truncated) and index + 1 < ACTION_STEPS:
                raise RuntimeError("expert branch ended before five raw actions")
    finally:
        restore_snapshot(env, snapshot, restore_process_rng=True)
        if expert_state is not None:
            vars(expert).clear()
            vars(expert).update(expert_state)
    return np.ascontiguousarray(np.stack(actions), dtype=np.float32)


def execute_learner_action_block(
    env: Any,
    observation: np.ndarray,
    action_block: np.ndarray,
    *,
    on_step: Callable[[np.ndarray, float, Mapping[str, object]], None] | None = None,
) -> LearnerBlockResult:
    """Advance the main environment with the learner block exactly once."""

    actions = np.asarray(action_block, dtype=np.float32)
    if actions.shape != (ACTION_STEPS, ACTION_DIM) or not np.isfinite(actions).all():
        raise ValueError("learner action block must have shape [5,4] and be finite")
    current = np.array(observation, dtype=np.float32, copy=True, order="C")
    rewards: list[float] = []
    successes: list[float] = []
    for index, raw_action in enumerate(actions):
        current, reward, terminated, truncated, info = env.step(raw_action)
        current = np.asarray(current, dtype=np.float32)
        mapping = info if isinstance(info, Mapping) else {}
        success = float(mapping.get("success", 0.0))
        rewards.append(float(reward))
        successes.append(success)
        if on_step is not None:
            on_step(current, float(reward), mapping)
        if (terminated or truncated) and index + 1 < ACTION_STEPS:
            raise RuntimeError("learner environment ended before five raw actions")
    return LearnerBlockResult(
        observation=np.array(current, dtype=np.float32, copy=True, order="C"),
        rewards=tuple(rewards),
        successes=tuple(successes),
    )


def choose_learner_action(
    *,
    base_action: torch.Tensor,
    state_latent: torch.Tensor,
    action_latent: torch.Tensor,
    posterior_token: torch.Tensor,
    missing_age: int,
    sensor_valid: bool,
    head: torch.nn.Module | None,
) -> torch.Tensor:
    """Apply a learned residual only when the visual measurement is invalid."""

    if sensor_valid or head is None:
        return base_action
    if head.training or any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("posterior DAgger head must be frozen and in eval mode")
    result = head(
        base_action=base_action,
        state_latent=state_latent,
        action_latent=action_latent,
        posterior_token=posterior_token,
        missing_age=torch.full(
            (base_action.shape[0],),
            int(missing_age),
            dtype=torch.long,
            device=base_action.device,
        ),
        sensor_valid=torch.zeros(
            (base_action.shape[0],),
            dtype=torch.bool,
            device=base_action.device,
        ),
    ).detach().float()
    if result.shape != base_action.shape or not bool(torch.isfinite(result).all()):
        raise ValueError("posterior DAgger head returned an invalid action block")
    return result


def _load_frozen_components(
    config: Mapping[str, object], device: torch.device
) -> tuple[torch.nn.Module, Mapping[str, object], Any]:
    source = _json(str(config["source_fresh_gate_config"]))
    p1 = source.get("p1")
    if not isinstance(p1, Mapping):
        raise TypeError("source fresh config lacks P1 policy settings")
    policy, policy_args = load_action_chunk_policy(
        _resolve(str(p1["policy_checkpoint"])), device
    )
    policy.set_reliability_override("ones")
    policy.requires_grad_(False).eval()

    training = source.get("frozen_predictor_training_contract")
    if not isinstance(training, Mapping):
        raise TypeError("source fresh config lacks predictor training contract")
    expected_config = _json(str(training["config"]))
    records = source.get("frozen_predictors")
    if not isinstance(records, list):
        raise TypeError("source fresh config lacks frozen predictor records")
    seed = int(config["model_seed"])
    matches = [record for record in records if int(record["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"frozen predictor seed {seed} is unavailable or ambiguous")
    record = matches[0]
    pair = load_frozen_p1_native_checkpoint(
        FrozenP1NativeCheckpointSpec(
            seed=seed,
            path=_resolve(str(record["checkpoint"])),
            sha256=str(record["sha256"]),
        ),
        expected_config=expected_config,
        device=device,
    )
    return policy, policy_args, pair


def _make_runtime(
    method: str, *, policy: torch.nn.Module, pair: Any
) -> P1NativeComplementaryPolicyRuntime:
    if method not in COLLECTION_METHODS:
        raise ValueError(f"unsupported posterior DAgger method {method!r}")
    return P1NativeComplementaryPolicyRuntime(
        method=method,
        policy=policy,
        factual_predictor=pair.factual,
        no_action_predictor=pair.no_action,
        freeze_global_visual_during_drop=False,
    )


@torch.inference_mode()
def _collect_episode(
    *,
    policy: torch.nn.Module,
    policy_args: Mapping[str, object],
    pair: Any,
    head: torch.nn.Module | None,
    method: str,
    dagger_iteration: int,
    episode_id: int,
    task: str,
    camera: str,
    width: int,
    height: int,
    benchmark_seed: int,
    task_index: int,
    environment_seed: int,
    drop_start: int,
    drop_duration: int,
    radius_px: int,
    value_uint8: int,
    occlusion_center_key: str,
    camera_condition: CameraPoseCondition,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    random.seed(environment_seed)
    np.random.seed(environment_seed)
    torch.random.default_generator.manual_seed(environment_seed)
    if device.type == "cuda":
        torch.cuda.default_generators[int(device.index or 0)].manual_seed(environment_seed)
    env = make_env(
        task,
        camera,
        width,
        height,
        task_index=task_index,
        benchmark_seed=benchmark_seed,
    )
    expert = make_scripted_policy(task)
    rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    successes: list[float] = []
    try:
        observation, _ = env.reset(seed=environment_seed)
        observation = np.asarray(observation, dtype=np.float32)
        camera_id = get_camera_id(env.model, camera)
        intrinsic = camera_intrinsic_from_fovy(
            width, height, float(env.model.cam_fovy[camera_id])
        )
        nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
        apply_fixed_camera_condition(env, camera_id, camera_condition)

        obs_horizon = int(policy_args.get("obs_horizon", 2))
        if obs_horizon != 2:
            raise ValueError("posterior DAgger requires policy history=2")
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
            str(policy_args.get("obs_mode", "robot")),
            0,
            0,
        )
        clean_history.append(initial)
        observed_history.append({name: value.copy() for name, value in initial.items()})
        runtime = _make_runtime(method, policy=policy, pair=pair)
        first_success_raw_step: int | None = None
        decision_index = 0

        while decision_index < drop_start + drop_duration:
            if first_success_raw_step is not None:
                break
            active = drop_start <= decision_index < drop_start + drop_duration
            current_clean = clean_history[-1]
            observed_history[-1] = (
                _corrupt_record(
                    current_clean,
                    radius_px=radius_px,
                    value_uint8=value_uint8,
                    center_key=occlusion_center_key,
                )
                if active
                else {name: value.copy() for name, value in current_clean.items()}
            )
            observed_batch = deployment_policy_batch(
                history_batch(observed_history, obs_horizon, device)
            )

            pending_row: dict[str, Any] | None = None
            if active:
                current_xyz = observed_batch["keypoints_xyz"][:, -1].float()
                decision = runtime.begin_invalid(current_xyz)
                policy_batch = decision.attach(observed_batch)
                output = policy.predict_action_chunk(dict(policy_batch))
                base_action = output.action[:, :ACTION_STEPS].detach().float()
                action_latent = output.latent[:, :ACTION_STEPS].detach().float()
                if base_action.shape != (1, ACTION_STEPS, ACTION_DIM):
                    raise ValueError("frozen learner returned an invalid action block")
                learner_action = choose_learner_action(
                    base_action=base_action,
                    state_latent=decision.policy_token.mean(dim=1),
                    action_latent=action_latent,
                    posterior_token=decision.policy_token,
                    missing_age=int(decision.missing_age),
                    sensor_valid=False,
                    head=head,
                )
                expert_action = query_expert_action_block(env, expert, observation)
                pending_row = {
                    "episode_id": int(episode_id),
                    "decision_index": int(decision_index),
                    "dagger_iteration": int(dagger_iteration),
                    "missing_age": int(decision.missing_age),
                    "sensor_valid": False,
                    "state_latent": decision.policy_token.mean(dim=1).cpu().numpy()[0],
                    "action_latent": action_latent.cpu().numpy()[0],
                    "base_action": base_action.cpu().numpy()[0],
                    "posterior_token": decision.policy_token.cpu().numpy()[0],
                    "expert_action": expert_action,
                }
            else:
                native = extract_policy_anchor_observation(policy, observed_batch)
                decision = runtime.begin_valid(
                    native.native_token[:, -1].float(),
                    native.sampled_visual[:, -1].float(),
                    native.global_visual_token[:, -1].float(),
                )
                if decision.override_required:
                    raise RuntimeError("valid visual input did not exactly bypass posterior")
                base_action = policy.predict_action_chunk(dict(observed_batch)).action[
                    :, :ACTION_STEPS
                ].detach().float()
                learner_action = choose_learner_action(
                    base_action=base_action,
                    state_latent=decision.policy_token.mean(dim=1),
                    action_latent=torch.zeros(
                        1,
                        ACTION_STEPS,
                        int(policy.model_dim),
                        dtype=base_action.dtype,
                        device=base_action.device,
                    ),
                    posterior_token=decision.policy_token,
                    missing_age=0,
                    sensor_valid=True,
                    head=head,
                )
                if not torch.equal(learner_action, base_action):
                    raise RuntimeError("valid visual input failed exact action bypass")

            raw = learner_action[0].cpu().numpy()
            clipped = np.clip(raw, env.action_space.low, env.action_space.high).astype(
                np.float32
            )

            def record_step(
                next_observation: np.ndarray,
                reward: float,
                info: Mapping[str, object],
            ) -> None:
                nonlocal first_success_raw_step
                clean_record = collect_observation(
                    env,
                    next_observation,
                    camera_id,
                    intrinsic,
                    nominal_extrinsic,
                    width,
                    height,
                    str(policy_args.get("obs_mode", "robot")),
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
                    if active
                    else {name: value.copy() for name, value in clean_record.items()}
                )
                success = float(info.get("success", 0.0))
                rewards.append(float(reward))
                successes.append(success)
                if success > 0.0 and first_success_raw_step is None:
                    first_success_raw_step = len(rewards)

            outcome = execute_learner_action_block(
                env,
                observation,
                clipped,
                on_step=record_step,
            )
            observation = outcome.observation
            runtime.register_executed_action(
                torch.from_numpy(np.ascontiguousarray(clipped))
                .unsqueeze(0)
                .to(device)
            )
            if pending_row is not None:
                pending_row["executed_action"] = clipped.copy()
                rows.append(pending_row)
            decision_index += 1

        metrics: dict[str, object] = {
            "episode_id": int(episode_id),
            "task_index": int(task_index),
            "environment_seed": int(environment_seed),
            "drop_start_decision": int(drop_start),
            "decisions_executed": int(decision_index),
            "label_count": len(rows),
            "success": bool(any(value > 0.0 for value in successes)),
            "first_success_raw_step": first_success_raw_step,
            "reward_sum": float(sum(rewards)),
            "camera_rotation_deg": float(camera_condition.rotation_deg),
            "camera_rotation_axis_world": camera_condition.axis_world.tolist(),
        }
        return rows, metrics
    finally:
        env.close()


def _merge_rows(api: Any, rows: list[dict[str, Any]], *, method: str) -> Any:
    if not rows:
        raise RuntimeError("posterior DAgger collection produced no invalid-state labels")
    return api.PosteriorDaggerArrays(
        episode_id=np.asarray([row["episode_id"] for row in rows], dtype=np.int64),
        decision_index=np.asarray(
            [row["decision_index"] for row in rows], dtype=np.int64
        ),
        dagger_iteration=np.asarray(
            [row["dagger_iteration"] for row in rows], dtype=np.int64
        ),
        method_index=np.full(
            len(rows), int(api.method_index(method)), dtype=np.int64
        ),
        missing_age=np.asarray([row["missing_age"] for row in rows], dtype=np.int64),
        sensor_valid=np.asarray([row["sensor_valid"] for row in rows], dtype=bool),
        state_latent=np.asarray([row["state_latent"] for row in rows], dtype=np.float32),
        action_latent=np.asarray(
            [row["action_latent"] for row in rows], dtype=np.float32
        ),
        base_action=np.asarray([row["base_action"] for row in rows], dtype=np.float32),
        posterior_token=np.asarray(
            [row["posterior_token"] for row in rows], dtype=np.float32
        ),
        expert_action=np.asarray(
            [row["expert_action"] for row in rows], dtype=np.float32
        ),
        executed_action=np.asarray(
            [row["executed_action"] for row in rows], dtype=np.float32
        ),
    )


def _output_path(
    *, split: str, method: str, dagger_iteration: int, explicit: str | None
) -> Path:
    if explicit is not None:
        return _resolve(explicit)
    return _resolve(
        "outputs/lewm_posterior_dagger_pick/"
        f"iteration_{dagger_iteration}/{split}_{method}.npz"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--method", choices=COLLECTION_METHODS, required=True)
    parser.add_argument("--dagger-iteration", type=int, required=True)
    parser.add_argument("--head-checkpoint")
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.dagger_iteration < 0:
        parser.error("--dagger-iteration must be non-negative")
    if args.device != "cpu" and args.device != "cuda:0":
        parser.error("this collector may use only cpu or cuda:0")

    os.environ.setdefault("MUJOCO_GL", "egl")
    api = _dagger_api()
    config = _json(args.config)
    _validate_config(config, protocol=str(api.PROTOCOL))
    try:
        contract = behavior_head_contract(
            config,
            dagger_iteration=int(args.dagger_iteration),
            head_checkpoint=args.head_checkpoint,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    split = _split_config(config, args.split)
    available = int(split["episodes"])
    episodes = available if args.episodes is None else int(args.episodes)
    if not 0 < episodes <= available:
        parser.error(f"--episodes must lie in [1,{available}]")
    output = _output_path(
        split=args.split,
        method=args.method,
        dagger_iteration=int(args.dagger_iteration),
        explicit=args.output,
    )
    metadata_path = output.with_suffix(".metadata.json")
    if not args.overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"refusing to overwrite {output} or its metadata")

    device = choose_device(args.device)
    policy, policy_args, pair = _load_frozen_components(config, device)
    head, head_metadata, loaded_contract = load_behavior_head(
        api,
        config,
        method=args.method,
        dagger_iteration=int(args.dagger_iteration),
        head_checkpoint=args.head_checkpoint,
        device=device,
    )
    if loaded_contract != contract:
        raise RuntimeError("behavior-head contract changed during collector setup")
    camera_condition = parse_camera_pose_condition(config)
    rng = np.random.default_rng(int(split["schedule_seed"]))
    starts = rng.integers(
        int(config["drop_start_min_decision"]),
        int(config["drop_start_max_decision"]) + 1,
        size=episodes,
    )

    all_rows: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, object]] = []
    for episode in range(episodes):
        rows, metrics = _collect_episode(
            policy=policy,
            policy_args=policy_args,
            pair=pair,
            head=head,
            method=args.method,
            dagger_iteration=int(args.dagger_iteration),
            episode_id=int(split["environment_seed_base"]) + episode,
            task=str(config["task"]),
            camera=str(config["camera"]),
            width=int(config["width"]),
            height=int(config["height"]),
            benchmark_seed=int(split["benchmark_seed"]),
            task_index=int(split["task_index_base"]) + episode,
            environment_seed=int(split["environment_seed_base"]) + episode,
            drop_start=int(starts[episode]),
            drop_duration=int(config["drop_duration_decisions"]),
            radius_px=int(config["occlusion_radius_px"]),
            value_uint8=int(config["occlusion_value_uint8"]),
            occlusion_center_key=str(config["occlusion_center_key"]),
            camera_condition=camera_condition,
            device=device,
        )
        all_rows.extend(rows)
        episode_metrics.append(metrics)
        print(
            f"[{args.split}] episode {episode + 1}/{episodes} "
            f"drop_start={int(starts[episode])} labels={len(rows)}",
            flush=True,
        )

    arrays = _merge_rows(api, all_rows, method=args.method)
    output.parent.mkdir(parents=True, exist_ok=True)
    api.save_posterior_dagger_arrays(output, arrays)
    metadata = {
        "schema_version": 1,
        "protocol": str(api.PROTOCOL),
        "split": args.split,
        "method": args.method,
        "dagger_iteration": int(args.dagger_iteration),
        "episodes": episodes,
        "rows": int(arrays.sample_count),
        "collection_contract": collection_contract(config),
        "head_checkpoint": (
            None if args.head_checkpoint is None else str(_resolve(args.head_checkpoint))
        ),
        "head_metadata": None if head_metadata is None else dict(head_metadata),
        "behavior_head_contract": {
            "required": bool(contract.required),
            "expected_method": args.method if contract.required else None,
            "expected_seed": int(contract.expected_seed),
            "expected_steps": int(contract.expected_steps),
            "expected_dagger_iteration": contract.expected_dagger_iteration,
        },
        "camera_condition": {
            "kind": "fixed_rotation",
            "camera_rotation_deg": float(camera_condition.rotation_deg),
            "camera_rotation_axis_world": camera_condition.axis_world.tolist(),
        },
        "episode_metrics": episode_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
