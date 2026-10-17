"""Causal seven-task collection for the Action-Branch A4 protocol."""

from __future__ import annotations

import os
import tempfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from fcpg.lewm_a4_storage import (
    ACTION_DIM as CHUNK_ACTION_DIM,
    LATENT_DIM,
    PROPOSAL_LEDGER_KEYS,
    STATE_CACHE_KEYS,
    STATE_SLOT_COUNT,
    validate_a4_ledger_cache_pair,
)
from fcpg.lewm_a4_task_signals import (
    A4TaskSignals,
    extract_a4_task_signals,
    workspace_control_margin,
)
from fcpg.lewm_action_branch_a3 import build_diverse_action_branch_bank
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    BRANCH_HORIZON_RAW_STEPS,
    FORMAL_TASK_INSTANCES,
    HISTORY_STEPS,
    HORIZON,
    RAW_STEPS_PER_MODEL_STEP,
    SIGREG_WEIGHT,
    TASKS,
    assign_a4_splits,
    evaluate_active_control_eligibility,
    sample_a4_proposal_schedule,
    validate_a4_contract,
)
from fcpg.lewm_counterfactual_runtime import (
    FrameRecord,
    _collect_frame_record,
    capture_branch_snapshot,
    corrected_points,
    grounded_valid_mask,
    restore_branch_snapshot,
)


RAW_ACTION_DIM = 4
MAX_QPOS_DIM = 17
STATE_SLOT_NAMES = ("visual_summary", "anchor0", "anchor1", "anchor2")
CONTACT_OR_GRASP_FIELDS = ("left_contact", "right_contact", "grasp")
OBJECT_POSE_FIELDS = ("x", "y", "z", "qw", "qx", "qy", "qz")

A4_CAMERA_BY_TASK = {
    task: (
        "drawer_view"
        if task == "drawer-retrieve-v0"
        else "towel_view"
        if task == "towel-wipe-v0"
        else "corner"
    )
    for task in TASKS
}

DIAGNOSTIC_KEYS = (
    "proposal_id",
    "task_class_index",
    "task_name",
    "task_instance_index",
    "source_episode_id",
    "split",
    "branch_valid",
    "future_physical_spread",
    "source_expert_success",
    "proprio",
    "anchor_validity",
    "object_pose",
    "goal_relative_progress",
    "contact_or_grasp_state",
    "branch_outcome",
    "physical_state",
    "branch_gripper_aperture",
    "branch_contact_or_grasp",
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _record_value(record: FrameRecord | Mapping[str, Any], name: str) -> np.ndarray:
    value = record[name] if isinstance(record, Mapping) else getattr(record, name)
    return np.asarray(value)


def a4_camera_name(task: str) -> str:
    try:
        return A4_CAMERA_BY_TASK[task]
    except KeyError as error:
        raise ValueError(f"unknown A4 task {task!r}") from error


@dataclass
class A4EpisodeRuntime:
    env: Any
    expert: Any
    observation: np.ndarray
    reset_info: Mapping[str, Any]
    record_current: Callable[[np.ndarray], FrameRecord]
    max_raw_steps: int
    camera_name: str
    camera_intrinsic: np.ndarray
    camera_extrinsic_cam_from_world: np.ndarray


def make_a4_episode_runtime(
    task: str,
    task_instance: int,
    environment_seed: int,
    *,
    width: int,
    height: int,
    benchmark_seed: int,
    camera_name: str | None = None,
    orbit_camera_view: Any | None = None,
) -> A4EpisodeRuntime:
    """Build and reset an A4 runtime with the task default or an explicit camera."""

    from fcpg.lewm_a4_task_factory import make_a4_env_and_expert
    from fcpg.mujoco_camera import (
        camera_extrinsic_cv_from_mujoco,
        camera_intrinsic_from_fovy,
        get_camera_id,
    )
    from fcpg.lewm_orbit_camera import (
        OrbitCameraView,
        a4_orbit_focus,
        orbit_camera_position,
        set_fixed_camera_look_at,
    )

    camera = a4_camera_name(task) if camera_name is None else str(camera_name)
    if not camera:
        raise ValueError("A4 camera_name must be non-empty")
    env, expert = make_a4_env_and_expert(
        task,
        camera,
        int(width),
        int(height),
        task_instance=int(task_instance),
        benchmark_seed=int(benchmark_seed),
    )
    try:
        observation, reset_info = env.reset(seed=int(environment_seed))
        if not isinstance(reset_info, Mapping):
            raise TypeError("A4 environment reset info must be a mapping")
        camera_id = get_camera_id(env.model, camera)
        if orbit_camera_view is not None:
            if not isinstance(orbit_camera_view, OrbitCameraView):
                raise TypeError("orbit_camera_view must be an OrbitCameraView")
            focus = a4_orbit_focus(
                np.asarray(observation, dtype=np.float64),
                getattr(env, "_target_pos", None),
            )
            set_fixed_camera_look_at(
                env.model,
                env.data,
                camera_id,
                orbit_camera_position(focus, orbit_camera_view),
                focus,
            )
        intrinsic = camera_intrinsic_from_fovy(
            int(width), int(height), float(env.model.cam_fovy[camera_id])
        )
        nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)

        def record_current(current_observation: np.ndarray) -> FrameRecord:
            return _collect_frame_record(
                env,
                current_observation,
                camera_id=camera_id,
                intrinsic=intrinsic,
                nominal_extrinsic=nominal_extrinsic,
                width=int(width),
                height=int(height),
            )

        return A4EpisodeRuntime(
            env=env,
            expert=expert,
            observation=np.asarray(observation, dtype=np.float32),
            reset_info=dict(reset_info),
            record_current=record_current,
            max_raw_steps=int(getattr(env, "max_path_length", 500)),
            camera_name=camera,
            camera_intrinsic=np.asarray(intrinsic, dtype=np.float32).copy(),
            camera_extrinsic_cam_from_world=np.asarray(
                nominal_extrinsic, dtype=np.float32
            ).copy(),
        )
    except Exception:
        env.close()
        raise


@torch.inference_mode()
def encode_a4_state_records(
    records: Sequence[FrameRecord],
    *,
    frame_encoder: Any,
    projector: Any,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
    corrector: Any,
    device: torch.device,
) -> np.ndarray:
    """Encode only ``[summary, anchor0, anchor1, anchor2]`` with frozen modules."""

    rows = list(records)
    if not rows:
        raise ValueError("A4 state encoding requires at least one frame")
    corrected = corrected_points(
        rows, SimpleNamespace(corrector=corrector, device=device)
    ).astype(np.float32, copy=False)
    pixels = np.stack([_record_value(row, "rgb") for row in rows]).astype(
        np.float32, copy=False
    )
    proprio = np.stack([_record_value(row, "proprio") for row in rows]).astype(
        np.float32, copy=False
    )
    nominal = np.stack([_record_value(row, "q_nominal") for row in rows]).astype(
        np.float32, copy=False
    )
    xyz = np.stack([_record_value(row, "keypoints_xyz") for row in rows]).astype(
        np.float32, copy=False
    )
    valid_nominal = np.stack(
        [_record_value(row, "valid_nominal") for row in rows]
    ).astype(bool, copy=False)
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError("A4 RGB records must have shape [T,3,H,W]")
    valid = grounded_valid_mask(
        valid_nominal,
        corrected,
        width=int(pixels.shape[-1]),
        height=int(pixels.shape[-2]),
    )
    pixel_tensor = torch.from_numpy(np.ascontiguousarray(pixels)).to(device).float()
    proprio_tensor = torch.from_numpy(np.ascontiguousarray(proprio)).to(device).float()
    proprio_tensor = (proprio_tensor - proprio_mean) / proprio_std
    tokens = frame_encoder.forward_token_sequence(
        pixels=pixel_tensor,
        robot_obs=proprio_tensor,
        q_nominal_px=torch.from_numpy(np.ascontiguousarray(nominal)).to(device),
        q_grounded_px=torch.from_numpy(np.ascontiguousarray(corrected)).to(device),
        keypoints_xyz=torch.from_numpy(np.ascontiguousarray(xyz)).to(device),
        valid=torch.from_numpy(np.ascontiguousarray(valid)).to(device),
    )
    if tokens.ndim != 3 or tokens.shape[1] < 4 or int(tokens.shape[-1]) != LATENT_DIM:
        raise ValueError("frozen frame encoder did not emit the expected token sequence")
    selected = torch.cat((tokens[:, :1], tokens[:, -3:]), dim=1)
    projected = projector(selected.reshape(-1, LATENT_DIM)).reshape(
        len(rows), STATE_SLOT_COUNT, LATENT_DIM
    )
    if not bool(torch.isfinite(projected).all()):
        raise ValueError("A4 frozen state encoding produced non-finite values")
    return projected.float().cpu().numpy().astype(np.float32, copy=False)


@dataclass(frozen=True)
class FrozenA4StateEncoder:
    frame_encoder: Any
    projector: Any
    proprio_mean: torch.Tensor
    proprio_std: torch.Tensor
    corrector: Any
    device: torch.device

    def encode(self, records: Sequence[FrameRecord]) -> np.ndarray:
        return encode_a4_state_records(
            records,
            frame_encoder=self.frame_encoder,
            projector=self.projector,
            proprio_mean=self.proprio_mean,
            proprio_std=self.proprio_std,
            corrector=self.corrector,
            device=self.device,
        )


def load_frozen_a4_state_encoder(
    base_checkpoint: str | Path,
    fcpg_checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> FrozenA4StateEncoder:
    """Load a SIGReg=0.02 base and retain only its frozen visual state path."""

    from fcpg.lewm_runtime import load_lewm_grounded_checkpoint
    from fcpg.transformer_grounding import load_fcpg_vit_adapter

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {target}")
    model, resolved, checkpoint = load_lewm_grounded_checkpoint(
        base_checkpoint, device=target
    )
    loss = _mapping(resolved.get("loss"), "base checkpoint loss config")
    weight = loss.get("sigreg_weight")
    if isinstance(weight, bool) or not isinstance(weight, Real) or float(weight) != SIGREG_WEIGHT:
        raise ValueError(f"A4 frozen base checkpoint must use SIGReg={SIGREG_WEIGHT}")
    if checkpoint.get("resolved_config") is None:
        raise ValueError("A4 frozen base checkpoint lacks resolved configuration")
    if (
        int(getattr(model, "latent_dim", -1)) != LATENT_DIM
        or int(getattr(model.frame_encoder, "num_keypoints", -1)) != 3
        or int(getattr(model, "history_size", -1)) != HISTORY_STEPS
        or int(getattr(model, "action_dim", -1)) != CHUNK_ACTION_DIM
    ):
        raise ValueError("A4 frozen base dimensions must be H=3, slots=4, D=192, A=20")
    corrector, _ = load_fcpg_vit_adapter(fcpg_checkpoint, target)
    model.frame_encoder.requires_grad_(False).eval()
    model.projector.requires_grad_(False).eval()
    corrector.requires_grad_(False).eval()
    return FrozenA4StateEncoder(
        frame_encoder=model.frame_encoder,
        projector=model.projector,
        proprio_mean=model.proprio_mean.detach(),
        proprio_std=model.proprio_std.detach(),
        corrector=corrector,
        device=target,
    )


def _success(info: Mapping[str, Any], env: Any) -> bool:
    value = info.get("success")
    if value is None:
        candidate = getattr(env, "success", False)
        value = candidate() if callable(candidate) else candidate
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("A4 success must be numeric") from error
    if not np.isfinite(numeric):
        raise ValueError("A4 success must be finite")
    return numeric > 0.0


def _bounded_expert_action(runtime: A4EpisodeRuntime, observation: np.ndarray) -> np.ndarray:
    action = np.asarray(runtime.expert.get_action(observation), dtype=np.float32)
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if action.shape != (RAW_ACTION_DIM,) or low.shape != action.shape or high.shape != action.shape:
        raise ValueError("A4 expert and action bounds must have shape [4]")
    if not np.isfinite(action).all():
        raise ValueError("A4 expert action must be finite")
    return np.clip(action, low, high).astype(np.float32, copy=False)


def workspace_and_joint_margin(env: Any) -> float:
    """Return normalized directional mocap-control headroom.

    The historical field name is retained for artifact compatibility. Task
    joints and Sawyer joint-range proxies are not mixed into this scalar.  See
    :func:`workspace_control_margin` for the single-boundary contact rule.
    """

    result = float(workspace_control_margin(env))
    if not np.isfinite(result):
        raise ValueError("A4 workspace/joint margin must be finite")
    return result


def recent_xyz_action_rms(action_xyz_norms: Sequence[float]) -> float:
    """Return RMS Cartesian action magnitude for one causal action window."""

    values = np.asarray(action_xyz_norms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("action_xyz_norms must be a non-empty vector")
    if not np.isfinite(values).all() or bool(np.any(values < 0.0)):
        raise ValueError("action_xyz_norms must be finite and non-negative")
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def unexpected_collision(env: Any, *, severe_penetration: float = -0.002) -> bool:
    """Flag severe robot/environment penetration, excluding intended object grasp."""

    object_id = _main_object_geom_id(env)
    pad_ids = set()
    for name in ("leftpad_geom", "rightpad_geom"):
        try:
            pad_ids.add(int(env.model.geom(name).id))
        except (KeyError, ValueError):
            pass
    for index in range(int(env.data.ncon)):
        contact = env.data.contact[index]
        if float(contact.dist) >= severe_penetration:
            continue
        first, second = int(contact.geom1), int(contact.geom2)
        first_robot = int(env.model.geom_group[first]) == 4
        second_robot = int(env.model.geom_group[second]) == 4
        if first_robot == second_robot:
            continue
        pair = {first, second}
        if object_id is not None and object_id in pair and pair & pad_ids:
            continue
        return True
    return False


def _main_object_geom_id(env: Any) -> int | None:
    getter = getattr(env, "_get_id_main_object", None)
    if callable(getter):
        try:
            value = int(getter())
            if value >= 0:
                return value
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return int(env.model.geom("objGeom").id)
    except (KeyError, ValueError):
        return None


def _object_position(env: Any) -> np.ndarray:
    values = np.asarray(env._get_pos_objects(), dtype=np.float64).reshape(-1)
    if values.size < 3 or not np.isfinite(values[:3]).all():
        raise ValueError("A4 object position must provide at least three finite values")
    return values[:3]


def _object_pose(env: Any) -> np.ndarray:
    position = _object_position(env)
    geom_id = _main_object_geom_id(env)
    if geom_id is None:
        quaternion = np.asarray(env._get_quat_objects(), dtype=np.float64).reshape(-1)[:4]
    else:
        body_id = int(env.model.geom_bodyid[geom_id])
        quaternion = np.asarray(env.data.xquat[body_id], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or norm <= 0.0:
        raise ValueError("A4 object quaternion must be finite and non-zero")
    quaternion = quaternion / norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return np.concatenate((position, quaternion)).astype(np.float32)


def _contacts_and_grasp(env: Any, info: Mapping[str, Any]) -> np.ndarray:
    object_id = _main_object_geom_id(env)
    contacts = [False, False]
    if object_id is not None:
        for pad_index, name in enumerate(("leftpad_geom", "rightpad_geom")):
            try:
                pad_id = int(env.model.geom(name).id)
            except (KeyError, ValueError):
                continue
            for index in range(int(env.data.ncon)):
                contact = env.data.contact[index]
                if {int(contact.geom1), int(contact.geom2)} == {object_id, pad_id}:
                    contacts[pad_index] = True
                    break
    grasp_value = info.get("grasp_success")
    if grasp_value is None:
        for name in ("is_block_grasped", "towel_grasped", "touching_main_object"):
            if hasattr(env, name):
                grasp_value = getattr(env, name)
                grasp_value = grasp_value() if callable(grasp_value) else grasp_value
                break
    grasp = bool(float(grasp_value or 0.0) > 0.0)
    return np.asarray((*contacts, grasp), dtype=bool)


def _goal_distance(env: Any, task: str) -> float:
    goal = np.asarray(env._target_pos, dtype=np.float64).reshape(-1)
    if goal.size < 3 or not np.isfinite(goal[:3]).all():
        raise ValueError("A4 goal position must provide three finite values")
    reference = np.asarray(env.tcp_center, dtype=np.float64) if task == "reach-v3" else _object_position(env)
    return float(np.linalg.norm(reference - goal[:3]))


def _goal_progress(
    env: Any, task: str, info: Mapping[str, Any], initial_distance: float
) -> float:
    if task == "towel-wipe-v0" and hasattr(env, "wipe_half_cycles"):
        required = max(int(getattr(env, "REQUIRED_HALF_CYCLES", 6)), 1)
        return float(np.clip(int(env.wipe_half_cycles) / required, 0.0, 1.0))
    distance = _goal_distance(env, task)
    if initial_distance <= np.finfo(np.float64).eps:
        return float(distance <= np.finfo(np.float64).eps)
    progress = 1.0 - distance / initial_distance
    if task == "drawer-retrieve-v0" and hasattr(env, "drawer_open_fraction"):
        progress = 0.25 * float(env.drawer_open_fraction) + 0.75 * progress
    return float(np.clip(progress, 0.0, 1.0))


@dataclass(frozen=True)
class PhysicalTarget:
    object_pose: np.ndarray
    reference_pose: np.ndarray
    goal_progress: float
    contact_or_grasp: bool
    branch_outcome: np.ndarray
    tcp: np.ndarray
    qpos: np.ndarray
    qpos_size: int
    gripper_aperture: float


def read_physical_target(
    env: Any,
    task: str,
    observation: np.ndarray,
    info: Mapping[str, Any],
    *,
    signal_extractor: Callable[[str, Any, Any, Mapping[str, Any] | None], A4TaskSignals] = extract_a4_task_signals,
) -> PhysicalTarget:
    qpos_raw = np.asarray(env.data.qpos, dtype=np.float32)
    if qpos_raw.ndim != 1 or qpos_raw.size > MAX_QPOS_DIM or not np.isfinite(qpos_raw).all():
        raise ValueError(f"A4 qpos must be finite with at most {MAX_QPOS_DIM} values")
    qpos = np.zeros(MAX_QPOS_DIM, dtype=np.float32)
    qpos[: qpos_raw.size] = qpos_raw
    tcp = np.asarray(env.tcp_center, dtype=np.float32)
    if tcp.shape != (3,) or not np.isfinite(tcp).all():
        raise ValueError("A4 TCP must be finite [3]")
    signals = signal_extractor(task, env, observation, info)
    diagnostics = signals.diagnostics
    return PhysicalTarget(
        object_pose=np.asarray(diagnostics.object_pose, dtype=np.float32).copy(),
        reference_pose=np.asarray(diagnostics.reference_pose, dtype=np.float32).copy(),
        goal_progress=float(diagnostics.goal_relative_progress),
        contact_or_grasp=bool(diagnostics.contact_or_grasp_state),
        branch_outcome=np.asarray(diagnostics.branch_outcome, dtype=np.float32).copy(),
        tcp=tcp.copy(),
        qpos=qpos,
        qpos_size=int(qpos_raw.size),
        gripper_aperture=float(signals.deployment.proprio[3]),
    )


@dataclass(frozen=True)
class ProposalRow:
    proposal_id: str
    task_class_index: int
    task_name: str
    task_instance_index: int
    source_episode_id: str
    environment_seed: int
    proposal_raw_step: int
    split: str
    inputs: Mapping[str, Any]
    eligible: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class BundleRow:
    proposal: ProposalRow
    history_state: np.ndarray
    history_action: np.ndarray
    center_action: np.ndarray
    future_action: np.ndarray
    target_state: np.ndarray
    branch_semantic_index: np.ndarray
    current_target: PhysicalTarget
    current_proprio: np.ndarray
    anchor_validity: np.ndarray
    branch_targets: tuple[tuple[PhysicalTarget, ...], ...]
    branch_reward: np.ndarray
    branch_success: np.ndarray
    source_expert_success: bool


@dataclass(frozen=True)
class A4CollectionPayload:
    proposal_ledger: Mapping[str, np.ndarray]
    state_cache: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


def _branch_seeds(
    contract: Mapping[str, Any], task_class_index: int, instance: int, proposal_step: int
) -> tuple[int, int]:
    split = _mapping(contract.get("split"), "split")
    values = np.random.SeedSequence(
        [
            int(split["protocol_seed"]),
            zlib.crc32(str(split["assignment"]).encode("utf-8")),
            int(task_class_index),
            int(instance),
            int(proposal_step),
        ]
    ).generate_state(2, dtype=np.uint32)
    return int(values[0]), int(values[1])


def _execute_branches(
    runtime: A4EpisodeRuntime,
    snapshot: Any,
    actions: np.ndarray,
    *,
    task: str,
    encode_records: Callable[[Sequence[FrameRecord]], np.ndarray],
    restore_snapshot: Callable[[Any, Any], None],
    signal_extractor: Callable[[str, Any, Any, Mapping[str, Any] | None], A4TaskSignals],
) -> tuple[np.ndarray, tuple[tuple[PhysicalTarget, ...], ...], np.ndarray, np.ndarray]:
    action_array = np.asarray(actions, dtype=np.float32)
    expected = (BRANCH_COUNT, BRANCH_HORIZON_RAW_STEPS, RAW_ACTION_DIM)
    if action_array.shape != expected:
        raise ValueError(f"A4 branch actions must have shape {expected}")
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if not np.array_equal(action_array, np.clip(action_array, low, high)):
        raise ValueError("A4 branch executor refuses actions requiring clipping")
    state_rows = []
    target_rows = []
    reward_rows = []
    success_rows = []
    for branch in action_array:
        restore_snapshot(runtime.env, snapshot)
        records = []
        targets = []
        rewards = []
        successes = []
        observation = None
        for raw_index, action in enumerate(branch, start=1):
            observation, reward, terminated, truncated, info = runtime.env.step(action)
            if not isinstance(info, Mapping):
                raise TypeError("A4 branch info must be a mapping")
            if terminated or truncated:
                raise RuntimeError("A4 branch ended before the complete H=8 horizon")
            if raw_index % RAW_STEPS_PER_MODEL_STEP == 0:
                observation = np.asarray(observation, dtype=np.float32)
                records.append(runtime.record_current(observation))
                targets.append(
                    read_physical_target(
                        runtime.env,
                        task,
                        observation,
                        info,
                        signal_extractor=signal_extractor,
                    )
                )
                rewards.append(float(reward))
                successes.append(_success(info, runtime.env))
        encoded = np.asarray(encode_records(records), dtype=np.float32)
        if encoded.shape != (HORIZON, STATE_SLOT_COUNT, LATENT_DIM) or not np.isfinite(encoded).all():
            raise ValueError("A4 branch encoder must emit finite [8,4,192] state")
        state_rows.append(encoded)
        target_rows.append(tuple(targets))
        reward_rows.append(rewards)
        success_rows.append(successes)
    return (
        np.stack(state_rows).astype(np.float32, copy=False),
        tuple(target_rows),
        np.asarray(reward_rows, dtype=np.float32),
        np.asarray(success_rows, dtype=bool),
    )


def _model_actions(raw_actions: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != RAW_ACTION_DIM:
        raise ValueError("raw actions must have shape [T,4]")
    if values.shape[0] % RAW_STEPS_PER_MODEL_STEP:
        raise ValueError("raw action count must be divisible by five")
    return np.ascontiguousarray(
        values.reshape(-1, RAW_STEPS_PER_MODEL_STEP * RAW_ACTION_DIM),
        dtype=np.float32,
    )


def _nominal_anchor_validity(records: Sequence[FrameRecord]) -> np.ndarray:
    validity = np.stack([_record_value(row, "valid_nominal") for row in records])
    if validity.shape != (HISTORY_STEPS, STATE_SLOT_COUNT - 1):
        raise ValueError("A4 history anchor validity must have shape [3,3]")
    return validity.astype(bool, copy=False)


def collect_a4_source_episode(
    *,
    task: str,
    task_class_index: int,
    task_instance_index: int,
    environment_seed: int,
    split: str,
    contract: Mapping[str, Any],
    runtime_factory: Callable[[str, int, int], A4EpisodeRuntime],
    encode_records: Callable[[Sequence[FrameRecord]], np.ndarray],
    max_proposals: int | None = None,
    schedule_sampler: Callable[..., tuple[int, ...]] = sample_a4_proposal_schedule,
    contract_validator: Callable[[Mapping[str, Any]], tuple[str, ...]] = validate_a4_contract,
    eligibility_contract: Mapping[str, Any] | None = None,
    snapshot_capture: Callable[..., Any] = capture_branch_snapshot,
    snapshot_restore: Callable[[Any, Any], None] = restore_branch_snapshot,
    branch_bank_builder: Callable[..., Any] = build_diverse_action_branch_bank,
    signal_extractor: Callable[[str, Any, Any, Mapping[str, Any] | None], A4TaskSignals] = extract_a4_task_signals,
) -> tuple[list[ProposalRow], list[BundleRow]]:
    """Collect one causal source episode and all accepted matched branches."""

    configured_tasks = contract_validator(contract)
    if task not in configured_tasks or configured_tasks[task_class_index] != task:
        raise ValueError("A4 task identity differs from the frozen seven-task panel")
    if split not in {"train", "development"}:
        raise ValueError("A4 collection is restricted to train and development")

    # Sampling is intentionally the first stateful operation, before env creation/reset.
    formal_schedule = tuple(
        int(value)
        for value in schedule_sampler(
            task, int(task_instance_index), contract=contract
        )
    )
    if tuple(sorted(set(formal_schedule))) != formal_schedule or not formal_schedule:
        raise ValueError("A4 proposal schedule must be non-empty, sorted, and unique")
    schedule = formal_schedule
    if max_proposals is not None:
        if isinstance(max_proposals, bool) or not 1 <= int(max_proposals) <= len(schedule):
            raise ValueError("max_proposals must be in the schedule range")
        complete = tuple(step for step in schedule if step >= 2 * RAW_STEPS_PER_MODEL_STEP)
        schedule = complete[: int(max_proposals)]
        if not schedule:
            schedule = formal_schedule[: int(max_proposals)]

    runtime = runtime_factory(task, int(task_instance_index), int(environment_seed))
    episode_id = f"{task}:task-instance-{int(task_instance_index):02d}"
    proposal_rows: list[ProposalRow] = []
    source_actions: list[np.ndarray] = []
    records: dict[int, FrameRecord] = {}
    accepted: dict[str, tuple[Any, PhysicalTarget, np.ndarray]] = {}
    needed_records = {
        step - offset
        for step in schedule
        for offset in (2 * RAW_STEPS_PER_MODEL_STEP, RAW_STEPS_PER_MODEL_STEP, 0)
        if step - offset >= 0
    }
    observation = np.asarray(runtime.observation, dtype=np.float32)
    info: Mapping[str, Any] = runtime.reset_info
    ever_success = False
    source_success = False
    last_raw_step = -1
    try:
        for raw_step in range(runtime.max_raw_steps):
            last_raw_step = raw_step
            if raw_step in needed_records:
                records[raw_step] = runtime.record_current(observation)

            expert_action = _bounded_expert_action(runtime, observation)
            current_success = _success(info, runtime.env)
            source_success |= current_success
            if raw_step in schedule:
                signals = signal_extractor(task, runtime.env, observation, info)
                signal_fields = signals.eligibility_fields()
                current_success = bool(signal_fields["success"])
                history_start = raw_step - 2 * RAW_STEPS_PER_MODEL_STEP
                complete_history = (
                    history_start >= 0
                    and all(
                        index in records
                        for index in (
                            history_start,
                            raw_step - RAW_STEPS_PER_MODEL_STEP,
                            raw_step,
                        )
                    )
                    and len(source_actions) >= raw_step
                )
                recent = np.asarray(
                    [
                        *source_actions[-(RAW_STEPS_PER_MODEL_STEP - 1) :],
                        expert_action,
                    ],
                    dtype=np.float32,
                )
                recent_rms = (
                    recent_xyz_action_rms(np.linalg.norm(recent[:, :3], axis=1))
                    if recent.shape == (RAW_STEPS_PER_MODEL_STEP, RAW_ACTION_DIM)
                    else 0.0
                )
                inputs = {
                    "proposal": True,
                    "complete_history": bool(complete_history),
                    "success": current_success,
                    "ever_success_before": bool(ever_success),
                    "active_stage": bool(signal_fields["active_stage"]),
                    "expert_xyz_norm": float(np.linalg.norm(expert_action[:3])),
                    "recent_action_rms": recent_rms,
                    "workspace_and_joint_margin": float(
                        signal_fields["workspace_and_joint_margin"]
                    ),
                    "unexpected_collision": bool(
                        signal_fields["unexpected_collision"]
                    ),
                    "remaining_raw_steps": int(runtime.max_raw_steps - raw_step),
                }
                decision = evaluate_active_control_eligibility(
                    inputs,
                    contract=(
                        contract
                        if eligibility_contract is None
                        else eligibility_contract
                    ),
                )
                proposal_id = f"{episode_id}:proposal-{raw_step:03d}"
                row = ProposalRow(
                    proposal_id=proposal_id,
                    task_class_index=int(task_class_index),
                    task_name=task,
                    task_instance_index=int(task_instance_index),
                    source_episode_id=episode_id,
                    environment_seed=int(environment_seed),
                    proposal_raw_step=int(raw_step),
                    split=split,
                    inputs=inputs,
                    eligible=decision.eligible,
                    rejection_reasons=decision.rejection_reasons,
                )
                proposal_rows.append(row)
                if decision.eligible:
                    accepted[proposal_id] = (
                        snapshot_capture(runtime.env, include_process_rng=True),
                        read_physical_target(
                            runtime.env,
                            task,
                            observation,
                            info,
                            signal_extractor=signal_extractor,
                        ),
                        np.asarray(signals.deployment.proprio, dtype=np.float32).copy(),
                    )

            ever_success |= current_success
            next_observation, _reward, terminated, truncated, next_info = runtime.env.step(
                expert_action
            )
            if not isinstance(next_info, Mapping):
                raise TypeError("A4 source step info must be a mapping")
            source_actions.append(expert_action.copy())
            observation = np.asarray(next_observation, dtype=np.float32)
            info = next_info
            source_success |= _success(info, runtime.env)
            if terminated or truncated:
                break

        missing = sorted(set(schedule) - {row.proposal_raw_step for row in proposal_rows})
        if missing:
            raise RuntimeError(
                f"A4 source episode ended before scheduled proposals {missing} "
                f"(last raw step {last_raw_step})"
            )

        action_array = np.asarray(source_actions, dtype=np.float32)
        bundles: list[BundleRow] = []
        row_by_id = {row.proposal_id: row for row in proposal_rows}
        for proposal_id, (snapshot, current_target, current_proprio) in accepted.items():
            row = row_by_id[proposal_id]
            step = row.proposal_raw_step
            end = step + BRANCH_HORIZON_RAW_STEPS
            if action_array.shape[0] < end:
                raise RuntimeError("accepted A4 proposal lacks a complete expert center")
            history_records = (
                records[step - 2 * RAW_STEPS_PER_MODEL_STEP],
                records[step - RAW_STEPS_PER_MODEL_STEP],
                records[step],
            )
            history_state = np.asarray(encode_records(history_records), dtype=np.float32)
            if history_state.shape != (HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM):
                raise ValueError("A4 history encoder must emit [3,4,192]")
            history_action = np.stack(
                (
                    _model_actions(action_array[step - 10 : step - 5])[0],
                    _model_actions(action_array[step - 5 : step])[0],
                )
            ).astype(np.float32, copy=False)
            raw_center = np.ascontiguousarray(action_array[step:end], dtype=np.float32)
            direction_seed, permutation_seed = _branch_seeds(
                contract, task_class_index, task_instance_index, step
            )
            bank = branch_bank_builder(
                raw_center,
                direction_seed=direction_seed,
                permutation_seed=permutation_seed,
                action_low=runtime.env.action_space.low,
                action_high=runtime.env.action_space.high,
            )
            target_state, branch_targets, branch_reward, branch_success = _execute_branches(
                runtime,
                snapshot,
                bank.actions,
                task=task,
                encode_records=encode_records,
                restore_snapshot=snapshot_restore,
                signal_extractor=signal_extractor,
            )
            semantic = np.random.default_rng(permutation_seed).permutation(BRANCH_COUNT)
            bundles.append(
                BundleRow(
                    proposal=row,
                    history_state=history_state,
                    history_action=history_action,
                    center_action=_model_actions(bank.center_action),
                    future_action=np.stack(
                        [_model_actions(branch) for branch in bank.actions]
                    ).astype(np.float32, copy=False),
                    target_state=target_state,
                    branch_semantic_index=semantic.astype(np.int64, copy=False),
                    current_target=current_target,
                    current_proprio=current_proprio,
                    anchor_validity=_nominal_anchor_validity(history_records),
                    branch_targets=branch_targets,
                    branch_reward=branch_reward,
                    branch_success=branch_success,
                    source_expert_success=bool(source_success),
                )
            )
        return proposal_rows, bundles
    finally:
        runtime.env.close()


def _text(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(value) for value in values])


def _future_physical_spread(physical_state: np.ndarray) -> np.ndarray:
    positions = np.asarray(physical_state, dtype=np.float64)[..., :3]
    flat = positions.reshape(positions.shape[0], BRANCH_COUNT, -1)
    delta = flat[:, :, None] - flat[:, None, :]
    return np.max(np.linalg.norm(delta, axis=-1), axis=(1, 2)).astype(np.float32)


def _physical_support_pose(task: str, target: PhysicalTarget) -> np.ndarray:
    """Return the task-controlled pose used by the physical-support gate."""

    if task == "reach-v3":
        pose = np.concatenate(
            (target.tcp, np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        )
    else:
        pose = np.asarray(target.object_pose, dtype=np.float32)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("A4 physical-support pose must be finite [7]")
    return pose.astype(np.float32, copy=False)


def validate_a4_diagnostics(arrays: Mapping[str, np.ndarray]) -> None:
    if tuple(arrays) != DIAGNOSTIC_KEYS or set(arrays) != set(DIAGNOSTIC_KEYS):
        raise ValueError("A4 diagnostics must use the exact companion schema")
    count = int(np.asarray(arrays["proposal_id"]).shape[0])
    if count <= 0:
        raise ValueError("A4 diagnostics must contain at least one accepted bundle")
    shapes = {
        "proposal_id": (count,),
        "task_class_index": (count,),
        "task_name": (count,),
        "task_instance_index": (count,),
        "source_episode_id": (count,),
        "split": (count,),
        "branch_valid": (count, BRANCH_COUNT),
        "future_physical_spread": (count,),
        "source_expert_success": (count,),
        "proprio": (count, 4),
        "anchor_validity": (count, HISTORY_STEPS, STATE_SLOT_COUNT - 1),
        "object_pose": (count, 7),
        "goal_relative_progress": (count, 1),
        "contact_or_grasp_state": (count, 1),
        "branch_outcome": (count, BRANCH_COUNT, HORIZON, 2),
        "physical_state": (count, BRANCH_COUNT, HORIZON, 7),
        "branch_gripper_aperture": (count, BRANCH_COUNT, HORIZON),
        "branch_contact_or_grasp": (count, BRANCH_COUNT, HORIZON),
    }
    for name, shape in shapes.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ValueError(f"A4 diagnostic {name} must have shape {shape}")
    for name in (
        "future_physical_spread",
        "proprio",
        "object_pose",
        "goal_relative_progress",
        "branch_outcome",
        "physical_state",
        "branch_gripper_aperture",
    ):
        value = np.asarray(arrays[name])
        if value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"A4 diagnostic {name} must be finite floating point")
    for name in (
        "branch_valid",
        "source_expert_success",
        "anchor_validity",
        "contact_or_grasp_state",
        "branch_contact_or_grasp",
    ):
        if np.asarray(arrays[name]).dtype.kind != "b":
            raise TypeError(f"A4 diagnostic {name} must contain booleans")
    if not set(np.asarray(arrays["split"]).astype(str)) <= {"train", "development"}:
        raise ValueError("A4 diagnostics cannot contain reserved-test rows")


def stack_a4_collection(
    proposals: Sequence[ProposalRow], bundles: Sequence[BundleRow]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    proposal_rows = list(proposals)
    bundle_rows = list(bundles)
    if not proposal_rows:
        raise ValueError("A4 proposal ledger cannot be empty")
    if not bundle_rows:
        raise RuntimeError("A4 collection produced no eligible bundles")
    accepted_index = {
        row.proposal.proposal_id: index for index, row in enumerate(bundle_rows)
    }
    ledger: dict[str, np.ndarray] = {
        "proposal_id": _text([row.proposal_id for row in proposal_rows]),
        "task_class_index": np.asarray([row.task_class_index for row in proposal_rows], dtype=np.int64),
        "task_name": _text([row.task_name for row in proposal_rows]),
        "task_instance_index": np.asarray([row.task_instance_index for row in proposal_rows], dtype=np.int64),
        "source_episode_id": _text([row.source_episode_id for row in proposal_rows]),
        "environment_seed": np.asarray([row.environment_seed for row in proposal_rows], dtype=np.int64),
        "proposal_raw_step": np.asarray([row.proposal_raw_step for row in proposal_rows], dtype=np.int64),
        "split": _text([row.split for row in proposal_rows]),
        "proposal": np.ones(len(proposal_rows), dtype=bool),
        "complete_history": np.asarray([row.inputs["complete_history"] for row in proposal_rows], dtype=bool),
        "success": np.asarray([row.inputs["success"] for row in proposal_rows], dtype=bool),
        "ever_success_before": np.asarray([row.inputs["ever_success_before"] for row in proposal_rows], dtype=bool),
        "active_stage": np.asarray([row.inputs["active_stage"] for row in proposal_rows], dtype=bool),
        "expert_xyz_norm": np.asarray([row.inputs["expert_xyz_norm"] for row in proposal_rows], dtype=np.float32),
        "recent_action_rms": np.asarray([row.inputs["recent_action_rms"] for row in proposal_rows], dtype=np.float32),
        "workspace_and_joint_margin": np.asarray([row.inputs["workspace_and_joint_margin"] for row in proposal_rows], dtype=np.float32),
        "unexpected_collision": np.asarray([row.inputs["unexpected_collision"] for row in proposal_rows], dtype=bool),
        "remaining_raw_steps": np.asarray([row.inputs["remaining_raw_steps"] for row in proposal_rows], dtype=np.int64),
        "eligible": np.asarray([row.eligible for row in proposal_rows], dtype=bool),
        "rejection_reasons": _text(["|".join(row.rejection_reasons) for row in proposal_rows]),
        "accepted_bundle_index": np.asarray([accepted_index.get(row.proposal_id, -1) for row in proposal_rows], dtype=np.int64),
    }
    if tuple(ledger) != PROPOSAL_LEDGER_KEYS:
        raise RuntimeError("internal A4 proposal ledger key order drifted")
    cache: dict[str, np.ndarray] = {
        "proposal_id": _text([row.proposal.proposal_id for row in bundle_rows]),
        "task_class_index": np.asarray([row.proposal.task_class_index for row in bundle_rows], dtype=np.int64),
        "task_name": _text([row.proposal.task_name for row in bundle_rows]),
        "task_instance_index": np.asarray([row.proposal.task_instance_index for row in bundle_rows], dtype=np.int64),
        "source_episode_id": _text([row.proposal.source_episode_id for row in bundle_rows]),
        "split": _text([row.proposal.split for row in bundle_rows]),
        "history_state": np.stack([row.history_state for row in bundle_rows]).astype(np.float32, copy=False),
        "history_action": np.stack([row.history_action for row in bundle_rows]).astype(np.float32, copy=False),
        "center_action": np.stack([row.center_action for row in bundle_rows]).astype(np.float32, copy=False),
        "future_action": np.stack([row.future_action for row in bundle_rows]).astype(np.float32, copy=False),
        "target_state": np.stack([row.target_state for row in bundle_rows]).astype(np.float32, copy=False),
        "branch_semantic_index": np.stack([row.branch_semantic_index for row in bundle_rows]).astype(np.int64, copy=False),
    }
    if tuple(cache) != STATE_CACHE_KEYS:
        raise RuntimeError("internal A4 state cache key order drifted")
    physical_state = np.asarray(
        [
            [
                [
                    _physical_support_pose(row.proposal.task_name, target)
                    for target in branch
                ]
                for branch in row.branch_targets
            ]
            for row in bundle_rows
        ],
        dtype=np.float32,
    )
    diagnostics: dict[str, np.ndarray] = {
        "proposal_id": cache["proposal_id"].copy(),
        "task_class_index": cache["task_class_index"].copy(),
        "task_name": cache["task_name"].copy(),
        "task_instance_index": cache["task_instance_index"].copy(),
        "source_episode_id": cache["source_episode_id"].copy(),
        "split": cache["split"].copy(),
        "branch_valid": np.ones((len(bundle_rows), BRANCH_COUNT), dtype=bool),
        "future_physical_spread": _future_physical_spread(physical_state),
        "source_expert_success": np.asarray([row.source_expert_success for row in bundle_rows], dtype=bool),
        "proprio": np.stack([row.current_proprio for row in bundle_rows]).astype(np.float32, copy=False),
        "anchor_validity": np.stack([row.anchor_validity for row in bundle_rows]).astype(bool, copy=False),
        "object_pose": np.stack([row.current_target.object_pose for row in bundle_rows]).astype(np.float32, copy=False),
        "goal_relative_progress": np.asarray([[row.current_target.goal_progress] for row in bundle_rows], dtype=np.float32),
        "contact_or_grasp_state": np.asarray([[row.current_target.contact_or_grasp] for row in bundle_rows], dtype=bool),
        "branch_outcome": np.asarray(
            [
                [
                    [target.branch_outcome for target in branch]
                    for branch in row.branch_targets
                ]
                for row in bundle_rows
            ],
            dtype=np.float32,
        ),
        "physical_state": physical_state,
        "branch_gripper_aperture": np.asarray(
            [
                [
                    [target.gripper_aperture for target in branch]
                    for branch in row.branch_targets
                ]
                for row in bundle_rows
            ],
            dtype=np.float32,
        ),
        "branch_contact_or_grasp": np.asarray(
            [
                [
                    [target.contact_or_grasp for target in branch]
                    for branch in row.branch_targets
                ]
                for row in bundle_rows
            ],
            dtype=bool,
        ),
    }
    validate_a4_ledger_cache_pair(ledger, cache)
    validate_a4_diagnostics(diagnostics)
    if not np.array_equal(diagnostics["proposal_id"], cache["proposal_id"]):
        raise RuntimeError("A4 diagnostics and state cache are not row aligned")
    return ledger, cache, diagnostics


def save_a4_diagnostics(
    path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]
) -> Path:
    validate_a4_diagnostics(arrays)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run_a4_collection(
    contract: Mapping[str, Any],
    *,
    encoder: FrozenA4StateEncoder,
    runtime_factory: Callable[[str, int, int], A4EpisodeRuntime],
    smoke: bool,
    max_tasks: int | None = None,
    max_episodes_per_task: int | None = None,
    max_proposals_per_episode: int | None = None,
    schedule_sampler: Callable[..., tuple[int, ...]] = sample_a4_proposal_schedule,
    contract_validator: Callable[[Mapping[str, Any]], tuple[str, ...]] = validate_a4_contract,
    assignment_contract: Mapping[str, Any] | None = None,
    eligibility_contract: Mapping[str, Any] | None = None,
    branch_bank_builder: Callable[..., Any] = build_diverse_action_branch_bank,
) -> A4CollectionPayload:
    """Run either the bounded diagnostic smoke or formal train+development panel."""

    tasks = contract_validator(contract)
    bounds = (max_tasks, max_episodes_per_task, max_proposals_per_episode)
    if not smoke and any(value is not None for value in bounds):
        raise ValueError("A4 task/episode/proposal bounds are smoke-only")
    if smoke:
        task_count = len(tasks) if max_tasks is None else int(max_tasks)
        episode_count = 2 if max_episodes_per_task is None else int(max_episodes_per_task)
        proposal_count = 4 if max_proposals_per_episode is None else int(max_proposals_per_episode)
        if not 1 <= task_count <= len(tasks):
            raise ValueError("A4 smoke max_tasks must be in 1..7")
        if not 1 <= episode_count <= 4:
            raise ValueError("A4 smoke max_episodes_per_task must be in 1..4")
        if not 1 <= proposal_count <= 4:
            raise ValueError("A4 smoke max_proposals_per_episode must be in 1..4")
        selected_tasks = tasks[:task_count]
    else:
        episode_count = 36
        proposal_count = None
        selected_tasks = tasks

    assignments = assign_a4_splits(
        tasks, contract if assignment_contract is None else assignment_contract
    )
    collection = _mapping(contract.get("collection"), "collection")
    seed_base = int(collection["environment_seed_base"])
    proposals: list[ProposalRow] = []
    bundles: list[BundleRow] = []
    for task_index, task in enumerate(selected_tasks):
        train = sorted(
            instance
            for instance in FORMAL_TASK_INSTANCES
            if assignments[(task, instance)] == "train"
        )
        development = sorted(
            instance
            for instance in FORMAL_TASK_INSTANCES
            if assignments[(task, instance)] == "development"
        )
        if smoke:
            instances = [value for pair in zip(train, development) for value in pair][
                :episode_count
            ]
        else:
            instances = train + development
        for instance in instances:
            split = assignments[(task, instance)]
            environment_seed = seed_base + task_index * len(FORMAL_TASK_INSTANCES) + instance
            episode_proposals, episode_bundles = collect_a4_source_episode(
                task=task,
                task_class_index=task_index,
                task_instance_index=instance,
                environment_seed=environment_seed,
                split=split,
                contract=contract,
                runtime_factory=runtime_factory,
                encode_records=encoder.encode,
                max_proposals=proposal_count,
                schedule_sampler=schedule_sampler,
                contract_validator=contract_validator,
                eligibility_contract=eligibility_contract,
                branch_bank_builder=branch_bank_builder,
            )
            proposals.extend(episode_proposals)
            bundles.extend(episode_bundles)
    ledger, cache, diagnostics = stack_a4_collection(proposals, bundles)
    metadata = {
        "mode": "smoke" if smoke else "formal",
        "tasks": list(selected_tasks),
        "source_episode_count": len(
            {row.source_episode_id for row in proposals}
        ),
        "proposal_count": len(proposals),
        "bundle_count": len(bundles),
        "state_slots": list(STATE_SLOT_NAMES),
        "camera_by_task": {task: a4_camera_name(task) for task in selected_tasks},
        "reserved_test_rows": 0,
    }
    return A4CollectionPayload(ledger, cache, diagnostics, metadata)


__all__ = [
    "A4_CAMERA_BY_TASK",
    "A4CollectionPayload",
    "A4EpisodeRuntime",
    "DIAGNOSTIC_KEYS",
    "FrozenA4StateEncoder",
    "STATE_SLOT_NAMES",
    "a4_camera_name",
    "collect_a4_source_episode",
    "encode_a4_state_records",
    "load_frozen_a4_state_encoder",
    "make_a4_episode_runtime",
    "recent_xyz_action_rms",
    "run_a4_collection",
    "save_a4_diagnostics",
    "stack_a4_collection",
    "validate_a4_diagnostics",
]
