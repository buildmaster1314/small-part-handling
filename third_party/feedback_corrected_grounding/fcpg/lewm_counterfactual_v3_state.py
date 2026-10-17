"""Privileged physical-state diagnostics for Counterfactual v3.

This module is deliberately independent of visual features, LeWM latents, and
environment rewards.  It preserves an interpretable semantic prefix and then
appends the padded raw MuJoCo state needed to make the privileged diagnostic
Markov.  Its deterministic task oracle is an offline upper bound; none of its
inputs are deployment inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np


SEMANTIC_STATE_DIM = 18
RAW_QPOS_DIM = 16
RAW_QVEL_DIM = 15
RAW_CTRL_DIM = 2
RAW_MOCAP_POSITION_DIM = 3
RAW_MOCAP_QUATERNION_DIM = 4
STATE_DIM = (
    SEMANTIC_STATE_DIM
    + RAW_QPOS_DIM
    + RAW_QVEL_DIM
    + RAW_CTRL_DIM
    + RAW_MOCAP_POSITION_DIM
    + RAW_MOCAP_QUATERNION_DIM
)
ORACLE_HORIZON = 5
TASK_NAMES = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)

STATE_FIELD_NAMES = (
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "gripper_open",
    "object_x",
    "object_y",
    "object_z",
    "object_qx",
    "object_qy",
    "object_qz",
    "object_qw",
    "left_pad_contact",
    "right_pad_contact",
    "left_pad_contact_force",
    "right_pad_contact_force",
    "object_init_x",
    "object_init_y",
    "object_init_z",
    *(f"raw_qpos_{index:02d}" for index in range(RAW_QPOS_DIM)),
    *(f"raw_qvel_{index:02d}" for index in range(RAW_QVEL_DIM)),
    *(f"raw_ctrl_{index:02d}" for index in range(RAW_CTRL_DIM)),
    *(f"raw_mocap_pos_{index:02d}" for index in range(RAW_MOCAP_POSITION_DIM)),
    *(
        f"raw_mocap_quat_{index:02d}"
        for index in range(RAW_MOCAP_QUATERNION_DIM)
    ),
)

TCP_SLICE = slice(0, 3)
GRIPPER_OPEN_INDEX = 3
OBJECT_POSITION_SLICE = slice(4, 7)
OBJECT_QUATERNION_SLICE = slice(7, 11)
LEFT_CONTACT_INDEX = 11
RIGHT_CONTACT_INDEX = 12
LEFT_FORCE_INDEX = 13
RIGHT_FORCE_INDEX = 14
OBJECT_INITIAL_POSITION_SLICE = slice(15, 18)
RAW_QPOS_SLICE = slice(18, 34)
RAW_QVEL_SLICE = slice(34, 49)
RAW_CTRL_SLICE = slice(49, 51)
RAW_MOCAP_POSITION_SLICE = slice(51, 54)
RAW_MOCAP_QUATERNION_SLICE = slice(54, 58)

_TASK_RAW_DIMENSIONS = {
    "pick-place-v3": (16, 15, 2, 1),
    "button-press-v3": (10, 10, 2, 1),
    "reach-v3": (16, 15, 2, 1),
    "push-v3": (16, 15, 2, 1),
}
_FREE_OBJECT_QPOS_QUATERNION_SLICE = slice(
    RAW_QPOS_SLICE.start + 12,
    RAW_QPOS_SLICE.start + 16,
)

STATE_ORACLE_INPUT_KEYS = frozenset(
    {"task", "current_state", "future_state", "goal_world"}
)
FORBIDDEN_ORACLE_KEY_TOKENS = frozenset(
    {
        "actual_return",
        "embedding",
        "future_return",
        "image",
        "label",
        "latent",
        "pixel",
        "progress",
        "q_true",
        "return",
        "reward",
        "rgb",
        "success",
        "visual",
    }
)

_OBJECT_GEOM_BY_TASK = {
    "pick-place-v3": "objGeom",
    "button-press-v3": "btnGeom",
    "reach-v3": "objGeom",
    "push-v3": "objGeom",
}


@dataclass(frozen=True)
class PadObjectContacts:
    """Per-pad contact existence and accumulated contact-frame force norm."""

    left: bool
    right: bool
    left_force: float
    right_force: float


@dataclass(frozen=True)
class StateOracleInput:
    """One true simulator state and its separately represented task goal."""

    state: np.ndarray
    goal_world: np.ndarray


@dataclass(frozen=True)
class DeterministicOracleResult:
    """Pre-registered physical costs and their deterministic candidate order.

    ``score`` is a monotone transform of ``physical_cost``, not a rank-derived
    label.  Physically identical trajectories therefore receive exactly equal
    scores and tied ranks.  ``order`` is stable only to make serialization
    deterministic; its ordering within a tie has no scoring semantics.
    """

    task_distance: np.ndarray
    stage_penalty: np.ndarray
    contact_penalty: np.ndarray
    reach_distance: np.ndarray
    trajectory_task_distance: np.ndarray
    physical_cost: np.ndarray
    order: np.ndarray
    rank: np.ndarray
    score: np.ndarray


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_raw_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = _finite_vector(value, size, name)
    return result.astype(np.float32)


def _padded_raw_vector(
    value: Any,
    *,
    actual_size: int,
    padded_size: int,
    name: str,
) -> np.ndarray:
    if actual_size > padded_size:
        raise ValueError(
            f"{name} actual size {actual_size} exceeds padded size {padded_size}"
        )
    raw = _finite_raw_vector(value, actual_size, name)
    result = np.zeros(padded_size, dtype=np.float32)
    result[:actual_size] = raw
    return result


def _canonical_task(task: str) -> str:
    result = str(task)
    if result not in TASK_NAMES:
        raise ValueError(f"unsupported v3 StateOracle task: {result!r}")
    return result


def normalize_object_quaternion(task: str, quaternion: Any) -> np.ndarray:
    """Return a finite, unit ``xyzw`` quaternion with a canonical sign.

    MetaWorld's button task exposes a MuJoCo body quaternion in ``wxyz`` order;
    the other three tasks expose SciPy ``xyzw`` quaternions.
    """

    task = _canonical_task(task)
    raw = _finite_vector(quaternion, 4, "object quaternion")
    if task == "button-press-v3":
        raw = raw[[1, 2, 3, 0]]
    norm = float(np.linalg.norm(raw))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("object quaternion must have non-zero norm")
    result = raw / norm
    if result[3] < 0.0:
        result = -result
    return result.astype(np.float32)


def _mujoco_module(module: Any | None) -> Any:
    return import_module("mujoco") if module is None else module


def _geom_id(model: Any, name: str, mujoco_module: Any) -> int:
    try:
        object_type = mujoco_module.mjtObj.mjOBJ_GEOM
        value = mujoco_module.mj_name2id(model, object_type, name)
    except (AttributeError, TypeError) as error:
        raise TypeError("mujoco module must provide mj_name2id and mjOBJ_GEOM") from error
    geom_id = int(value)
    if geom_id < 0:
        raise ValueError(f"required MuJoCo geom is missing: {name!r}")
    return geom_id


def query_pad_object_contacts(
    model: Any,
    data: Any,
    object_geom_name: str,
    *,
    mujoco_module: Any | None = None,
) -> PadObjectContacts:
    """Query real pad-object contacts through MuJoCo's name and force APIs.

    This intentionally does not call MetaWorld's ``touching_main_object``
    property.  The force is the sum of Euclidean norms of the three contact
    force components for every matching contact involving a pad.
    """

    mujoco = _mujoco_module(mujoco_module)
    left_id = _geom_id(model, "leftpad_geom", mujoco)
    right_id = _geom_id(model, "rightpad_geom", mujoco)
    object_id = _geom_id(model, str(object_geom_name), mujoco)

    contacts = getattr(data, "contact", None)
    if contacts is None:
        raise TypeError("MuJoCo data must expose contact records")
    ncon = int(getattr(data, "ncon", len(contacts)))
    if ncon < 0 or ncon > len(contacts):
        raise ValueError("MuJoCo data.ncon is inconsistent with contact storage")

    left = False
    right = False
    left_force = 0.0
    right_force = 0.0
    for index in range(ncon):
        contact = contacts[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        pad: str | None = None
        if object_id in pair and left_id in pair:
            left = True
            pad = "left"
        elif object_id in pair and right_id in pair:
            right = True
            pad = "right"
        if pad is None:
            continue

        wrench = np.zeros(6, dtype=np.float64)
        try:
            mujoco.mj_contactForce(model, data, index, wrench)
        except (AttributeError, TypeError) as error:
            raise TypeError("mujoco module must provide mj_contactForce") from error
        if not np.isfinite(wrench).all():
            raise ValueError("MuJoCo returned a non-finite contact force")
        magnitude = float(np.linalg.norm(wrench[:3]))
        if pad == "left":
            left_force += magnitude
        else:
            right_force += magnitude

    return PadObjectContacts(
        left=left,
        right=right,
        left_force=left_force,
        right_force=right_force,
    )


def extract_true_simulator_state(
    env: Any,
    observation: Any,
    task: str,
    *,
    mujoco_module: Any | None = None,
) -> StateOracleInput:
    """Extract the v3 full physical state without reward or visual values."""

    task = _canonical_task(task)
    observation_array = np.asarray(observation, dtype=np.float64)
    if observation_array.ndim != 1 or observation_array.shape[0] < 4:
        raise ValueError("MetaWorld observation must be a flat vector of length >= 4")
    if not np.isfinite(observation_array[:4]).all():
        raise ValueError("MetaWorld proprioceptive observation must be finite")

    tcp = _finite_vector(getattr(env, "tcp_center"), 3, "tcp_center")
    object_position = _finite_vector(
        env._get_pos_objects(), 3, "object position"
    )
    object_quaternion = normalize_object_quaternion(task, env._get_quat_objects())
    object_initial_position = _finite_vector(
        getattr(env, "obj_init_pos"), 3, "obj_init_pos"
    )
    goal_world = _finite_vector(getattr(env, "_target_pos"), 3, "_target_pos")
    contacts = query_pad_object_contacts(
        env.model,
        env.data,
        _OBJECT_GEOM_BY_TASK[task],
        mujoco_module=mujoco_module,
    )

    expected_nq, expected_nv, expected_nu, expected_nmocap = (
        _TASK_RAW_DIMENSIONS[task]
    )
    actual_model_dimensions = (
        int(getattr(env.model, "nq", -1)),
        int(getattr(env.model, "nv", -1)),
        int(getattr(env.model, "nu", -1)),
        int(getattr(env.model, "nmocap", -1)),
    )
    expected_model_dimensions = (
        expected_nq,
        expected_nv,
        expected_nu,
        expected_nmocap,
    )
    if actual_model_dimensions != expected_model_dimensions:
        raise ValueError(
            f"{task} MuJoCo dimensions changed: expected "
            f"nq/nv/nu/nmocap={expected_model_dimensions}, got "
            f"{actual_model_dimensions}"
        )

    raw_qpos = _padded_raw_vector(
        env.data.qpos,
        actual_size=expected_nq,
        padded_size=RAW_QPOS_DIM,
        name="data.qpos",
    )
    raw_qvel = _padded_raw_vector(
        env.data.qvel,
        actual_size=expected_nv,
        padded_size=RAW_QVEL_DIM,
        name="data.qvel",
    )
    raw_ctrl = _finite_raw_vector(env.data.ctrl, RAW_CTRL_DIM, "data.ctrl")
    mocap_position = np.asarray(env.data.mocap_pos, dtype=np.float64)
    mocap_quaternion = np.asarray(env.data.mocap_quat, dtype=np.float64)
    if mocap_position.shape != (expected_nmocap, RAW_MOCAP_POSITION_DIM):
        raise ValueError(
            "data.mocap_pos must have shape "
            f"({expected_nmocap}, {RAW_MOCAP_POSITION_DIM}), got "
            f"{mocap_position.shape}"
        )
    if mocap_quaternion.shape != (
        expected_nmocap,
        RAW_MOCAP_QUATERNION_DIM,
    ):
        raise ValueError(
            "data.mocap_quat must have shape "
            f"({expected_nmocap}, {RAW_MOCAP_QUATERNION_DIM}), got "
            f"{mocap_quaternion.shape}"
        )
    raw_mocap_position = _finite_raw_vector(
        mocap_position[0], RAW_MOCAP_POSITION_DIM, "data.mocap_pos[0]"
    )
    raw_mocap_quaternion = _finite_raw_vector(
        mocap_quaternion[0],
        RAW_MOCAP_QUATERNION_DIM,
        "data.mocap_quat[0]",
    )
    # MetaWorld deliberately stores Sawyer's raw mocap orientation as
    # [1, 0, 1, 0], so preserving raw state precludes unit normalization here.
    # Non-zero norm is still required for it to denote an orientation.
    if float(np.linalg.norm(raw_mocap_quaternion)) <= np.finfo(np.float32).eps:
        raise ValueError("data.mocap_quat[0] must have non-zero norm")

    state = np.concatenate(
        (
            tcp,
            observation_array[3:4],
            object_position,
            object_quaternion.astype(np.float64),
            np.asarray(
                [
                    float(contacts.left),
                    float(contacts.right),
                    contacts.left_force,
                    contacts.right_force,
                ],
                dtype=np.float64,
            ),
            object_initial_position,
            raw_qpos,
            raw_qvel,
            raw_ctrl,
            raw_mocap_position,
            raw_mocap_quaternion,
        )
    ).astype(np.float32)
    validate_state_array(state, name="state")
    _validate_task_raw_state(task, state, name="state")
    return StateOracleInput(
        state=state,
        goal_world=goal_world.astype(np.float32),
    )


def validate_state_array(value: Any, *, name: str) -> np.ndarray:
    """Validate one or more full states and return a float32 view."""

    result = np.asarray(value)
    if result.ndim < 1 or result.shape[-1] != STATE_DIM:
        raise ValueError(f"{name} must end in state dimension {STATE_DIM}")
    if result.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    contacts = result[..., [LEFT_CONTACT_INDEX, RIGHT_CONTACT_INDEX]]
    if not np.logical_or(contacts == 0.0, contacts == 1.0).all():
        raise ValueError(f"{name} contact flags must be binary")
    forces = result[..., [LEFT_FORCE_INDEX, RIGHT_FORCE_INDEX]]
    if np.any(forces < 0.0):
        raise ValueError(f"{name} contact forces must be non-negative")
    quaternion = result[..., OBJECT_QUATERNION_SLICE]
    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1.0, atol=1e-5):
        raise ValueError(f"{name} object quaternions must be unit length")
    mocap_quaternion = result[..., RAW_MOCAP_QUATERNION_SLICE]
    if np.any(
        np.linalg.norm(mocap_quaternion, axis=-1)
        <= np.finfo(np.float32).eps
    ):
        raise ValueError(f"{name} raw mocap quaternions must have non-zero norm")
    return result


def _validate_task_raw_state(task: str, value: Any, *, name: str) -> np.ndarray:
    """Validate task-specific padding and free-joint quaternion invariants."""

    task = _canonical_task(task)
    result = validate_state_array(value, name=name)
    expected_nq, expected_nv, _, _ = _TASK_RAW_DIMENSIONS[task]
    padded_qpos = result[..., RAW_QPOS_SLICE]
    padded_qvel = result[..., RAW_QVEL_SLICE]
    if expected_nq < RAW_QPOS_DIM and np.any(padded_qpos[..., expected_nq:] != 0.0):
        raise ValueError(f"{name} {task} qpos padding must be exactly zero")
    if expected_nv < RAW_QVEL_DIM and np.any(padded_qvel[..., expected_nv:] != 0.0):
        raise ValueError(f"{name} {task} qvel padding must be exactly zero")
    if expected_nq == RAW_QPOS_DIM:
        free_quaternion = result[..., _FREE_OBJECT_QPOS_QUATERNION_SLICE]
        if not np.allclose(
            np.linalg.norm(free_quaternion, axis=-1),
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                f"{name} {task} raw free-joint quaternion must be unit length"
            )
    return result


def validate_state_oracle_inputs(inputs: Mapping[str, Any]) -> None:
    """Fail closed if labels or visual representations enter StateOracle."""

    if not isinstance(inputs, Mapping):
        raise TypeError("StateOracle inputs must be a mapping")
    keys = {str(key) for key in inputs}
    unexpected = keys - STATE_ORACLE_INPUT_KEYS
    if unexpected:
        forbidden = sorted(
            key
            for key in unexpected
            if any(token in key.lower() for token in FORBIDDEN_ORACLE_KEY_TOKENS)
        )
        if forbidden:
            raise ValueError(
                "offline labels or visual representations are forbidden in "
                f"StateOracle inputs: {forbidden}"
            )
        raise ValueError(f"unknown StateOracle input keys: {sorted(unexpected)}")
    missing = STATE_ORACLE_INPUT_KEYS - keys
    if missing:
        raise KeyError(f"StateOracle inputs are missing keys: {sorted(missing)}")

    _canonical_task(str(inputs["task"]))
    current = _validate_task_raw_state(
        str(inputs["task"]), inputs["current_state"], name="current_state"
    )
    future = _validate_task_raw_state(
        str(inputs["task"]), inputs["future_state"], name="future_state"
    )
    if current.shape != (STATE_DIM,):
        raise ValueError(f"current_state must have shape ({STATE_DIM},)")
    if future.ndim != 3 or future.shape[1:] != (ORACLE_HORIZON, STATE_DIM):
        raise ValueError(
            "future_state must have shape "
            f"[candidates, {ORACLE_HORIZON}, {STATE_DIM}]"
        )
    goal = np.asarray(inputs["goal_world"])
    if goal.dtype != np.float32 or goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("goal_world must be one finite float32 [3] vector")


def _stage_penalty(
    task: str,
    task_distance: np.ndarray,
    reach_distance: np.ndarray,
    final_state: np.ndarray,
    any_contact: np.ndarray,
    goal_world: np.ndarray,
) -> np.ndarray:
    """Return a fixed coarse task stage, where lower is better."""

    if task == "reach-v3":
        return (task_distance > 0.05).astype(np.float64)
    if task == "button-press-v3":
        success = task_distance <= 0.02
        near_or_contact = np.logical_or(reach_distance <= 0.05, any_contact)
        return np.where(success, 0.0, np.where(near_or_contact, 1.0, 2.0))
    if task == "push-v3":
        success = task_distance <= 0.05
        initial_position = final_state[:, OBJECT_INITIAL_POSITION_SLICE]
        initial_distance = np.linalg.norm(
            initial_position - goal_world[None], axis=-1
        )
        moved_toward_goal = task_distance + 0.005 < initial_distance
        near_or_contact = np.logical_or(reach_distance <= 0.03, any_contact)
        return np.where(
            success,
            0.0,
            np.where(moved_toward_goal, 1.0, np.where(near_or_contact, 2.0, 3.0)),
        )
    if task == "pick-place-v3":
        success = task_distance <= 0.07
        lifted = (
            final_state[:, OBJECT_POSITION_SLICE][:, 2]
            - final_state[:, OBJECT_INITIAL_POSITION_SLICE][:, 2]
            > 0.02
        )
        near_or_contact = np.logical_or(reach_distance <= 0.03, any_contact)
        return np.where(
            success,
            0.0,
            np.where(lifted, 1.0, np.where(near_or_contact, 2.0, 3.0)),
        )
    raise AssertionError(task)


def deterministic_oracle_scores(
    task: str,
    current_state: Any,
    future_state: Any,
    goal_world: Any,
) -> DeterministicOracleResult:
    """Score candidates using only fixed H=5 physical trajectory costs.

    The utility is a monotone transform of a task-aware staged physical cost.
    Candidate identity and array index never enter the score.  No reward,
    return, progress, success label, RGB, or latent is accepted.
    """

    task = _canonical_task(task)
    inputs = {
        "task": task,
        "current_state": np.asarray(current_state),
        "future_state": np.asarray(future_state),
        "goal_world": np.asarray(goal_world),
    }
    validate_state_oracle_inputs(inputs)
    current = inputs["current_state"]
    future = inputs["future_state"]
    goal = inputs["goal_world"].astype(np.float64)
    del current  # Contract validation is intentional; ranking uses future physics.

    positions = future[..., OBJECT_POSITION_SLICE].astype(np.float64)
    tcp = future[..., TCP_SLICE].astype(np.float64)
    if task == "reach-v3":
        trajectory_task_distance = np.linalg.norm(tcp - goal[None, None], axis=-1)
        reach_trajectory = trajectory_task_distance
    elif task == "button-press-v3":
        trajectory_task_distance = np.abs(positions[..., 1] - goal[1])
        reach_trajectory = np.linalg.norm(tcp - positions, axis=-1)
    else:
        trajectory_task_distance = np.linalg.norm(
            positions - goal[None, None], axis=-1
        )
        reach_trajectory = np.linalg.norm(tcp - positions, axis=-1)

    final = future[:, -1]
    task_distance = trajectory_task_distance[:, -1]
    reach_distance = reach_trajectory[:, -1]
    any_contact_trajectory = np.logical_or(
        future[..., LEFT_CONTACT_INDEX] > 0.5,
        future[..., RIGHT_CONTACT_INDEX] > 0.5,
    )
    any_contact = np.any(any_contact_trajectory, axis=1)
    contact_penalty = np.where(any_contact, 0.0, 1.0)
    if task == "reach-v3":
        contact_penalty.fill(0.0)

    stage_penalty = _stage_penalty(
        task,
        task_distance,
        reach_distance,
        final,
        any_contact,
        goal,
    )

    mean_distance = np.mean(trajectory_task_distance, axis=1)
    # Fixed units and weights make the diagnostic reproducible.  A full stage
    # dominates centimetre-scale distance differences, while terminal and
    # trajectory distance still distinguish candidates within a stage.
    if task == "reach-v3":
        stage_weight, reach_weight, contact_weight = 0.25, 0.0, 0.0
    elif task == "button-press-v3":
        stage_weight, reach_weight, contact_weight = 0.25, 0.05, 0.01
    elif task == "push-v3":
        stage_weight, reach_weight, contact_weight = 0.25, 0.05, 0.01
    elif task == "pick-place-v3":
        stage_weight, reach_weight, contact_weight = 0.25, 0.05, 0.02
    else:  # pragma: no cover - guarded by _canonical_task.
        raise AssertionError(task)
    physical_cost = (
        stage_weight * stage_penalty
        + task_distance
        + 0.25 * mean_distance
        + reach_weight * reach_distance
        + contact_weight * contact_penalty
    )
    if not np.isfinite(physical_cost).all() or np.any(physical_cost < 0.0):
        raise ValueError("StateOracle physical costs must be finite and non-negative")

    # Stable ordering is only a serialization convenience.  Scores and ranks
    # preserve exact ties, so candidate position cannot create artificial
    # supervision when two physical outcomes are identical.
    order = np.argsort(physical_cost, kind="stable").astype(np.int64)
    rank = np.sum(
        physical_cost[None, :] < physical_cost[:, None], axis=1
    ).astype(np.int64)
    score = (1.0 / (1.0 + physical_cost)).astype(np.float32)

    return DeterministicOracleResult(
        task_distance=task_distance.astype(np.float32),
        stage_penalty=stage_penalty.astype(np.float32),
        contact_penalty=contact_penalty.astype(np.float32),
        reach_distance=reach_distance.astype(np.float32),
        trajectory_task_distance=mean_distance.astype(np.float32),
        physical_cost=physical_cost.astype(np.float32),
        order=order,
        rank=rank,
        score=score,
    )


def stack_state_oracle_inputs(
    values: Sequence[StateOracleInput],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack extracted states while preserving the separate goal channel."""

    if not values:
        raise ValueError("at least one StateOracleInput is required")
    states = np.stack([value.state for value in values]).astype(np.float32)
    goals = np.stack([value.goal_world for value in values]).astype(np.float32)
    validate_state_array(states, name="states")
    if goals.shape != (len(values), 3) or not np.isfinite(goals).all():
        raise ValueError("stacked goals must have shape [N, 3]")
    return states, goals


__all__ = [
    "DeterministicOracleResult",
    "FORBIDDEN_ORACLE_KEY_TOKENS",
    "GRIPPER_OPEN_INDEX",
    "LEFT_CONTACT_INDEX",
    "LEFT_FORCE_INDEX",
    "OBJECT_INITIAL_POSITION_SLICE",
    "OBJECT_POSITION_SLICE",
    "OBJECT_QUATERNION_SLICE",
    "ORACLE_HORIZON",
    "PadObjectContacts",
    "RAW_CTRL_DIM",
    "RAW_CTRL_SLICE",
    "RAW_MOCAP_POSITION_DIM",
    "RAW_MOCAP_POSITION_SLICE",
    "RAW_MOCAP_QUATERNION_DIM",
    "RAW_MOCAP_QUATERNION_SLICE",
    "RAW_QPOS_DIM",
    "RAW_QPOS_SLICE",
    "RAW_QVEL_DIM",
    "RAW_QVEL_SLICE",
    "RIGHT_CONTACT_INDEX",
    "RIGHT_FORCE_INDEX",
    "SEMANTIC_STATE_DIM",
    "STATE_DIM",
    "STATE_FIELD_NAMES",
    "STATE_ORACLE_INPUT_KEYS",
    "StateOracleInput",
    "TASK_NAMES",
    "TCP_SLICE",
    "deterministic_oracle_scores",
    "extract_true_simulator_state",
    "normalize_object_quaternion",
    "query_pad_object_contacts",
    "stack_state_oracle_inputs",
    "validate_state_array",
    "validate_state_oracle_inputs",
]
