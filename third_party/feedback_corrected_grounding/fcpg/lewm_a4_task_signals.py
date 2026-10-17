"""Fixed-shape state signals for the frozen seven-task A4 panel.

Only task identity and proprioception are deployment-visible.  Geometry,
progress, contacts, outcomes, and safety checks are privileged collection-time
diagnostics and offline readout targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_a4 import TASKS as _PROTOCOL_TASKS


TASKS = (
    "pick-place-v3",
    "drawer-retrieve-v0",
    "reach-v3",
    "push-v3",
    "door-open-v3",
    "window-open-v3",
    "towel-wipe-v0",
)
if tuple(_PROTOCOL_TASKS) != TASKS:
    raise RuntimeError("A4 task-signal panel differs from the frozen protocol")

OBSERVATION_DIM = 39
PROPRIO_DIM = 4
POSE_DIM = 7
BRANCH_OUTCOME_DIM = 2
SEVERE_PENETRATION_METERS = 0.002

_GOAL_AXES = {
    "pick-place-v3": (0, 1, 2),
    "drawer-retrieve-v0": (0, 1, 2),
    "reach-v3": (0, 1, 2),
    "push-v3": (0, 1),
    "door-open-v3": (0, 1),
    "window-open-v3": (0,),
}


class A4SignalError(RuntimeError):
    """Raised when an A4 environment omits a required physical signal."""


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise A4SignalError(f"{name} must be numeric") from exc
    if result.shape != shape:
        raise A4SignalError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise A4SignalError(f"{name} must contain only finite values")
    output = result.astype(np.float32)
    output.setflags(write=False)
    return output


def _scalar(value: Any, name: str, *, unit: bool = False) -> np.float32:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise A4SignalError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise A4SignalError(f"{name} must be finite")
    if unit and not 0.0 <= result <= 1.0:
        raise A4SignalError(f"{name} must be in [0, 1], got {result}")
    return np.float32(result)


def _flag(value: Any, name: str) -> np.bool_:
    if isinstance(value, (bool, np.bool_)):
        return np.bool_(value)
    numeric = _scalar(value, name)
    if float(numeric) not in (0.0, 1.0):
        raise A4SignalError(f"{name} must be boolean or binary")
    return np.bool_(bool(numeric))


def _fixed(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.float32:
        raise TypeError(f"{name} must be float32 with shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class A4DeploymentSignals:
    task_onehot: np.ndarray
    proprio: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_onehot", _fixed(self.task_onehot, (len(TASKS),), "task_onehot")
        )
        object.__setattr__(self, "proprio", _fixed(self.proprio, (PROPRIO_DIM,), "proprio"))


@dataclass(frozen=True, slots=True)
class A4PrivilegedSignals:
    active_stage: np.bool_
    task_progress: np.float32
    object_pose: np.ndarray
    reference_pose: np.ndarray
    goal_relative_progress: np.float32
    contact_or_grasp_state: np.bool_
    success: np.bool_
    branch_outcome: np.ndarray
    workspace_and_joint_margin: np.float32
    unexpected_collision: np.bool_
    severe_penetration: np.bool_

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_stage", _flag(self.active_stage, "active_stage"))
        object.__setattr__(self, "task_progress", _scalar(self.task_progress, "task_progress", unit=True))
        object.__setattr__(self, "object_pose", _fixed(self.object_pose, (POSE_DIM,), "object_pose"))
        object.__setattr__(self, "reference_pose", _fixed(self.reference_pose, (POSE_DIM,), "reference_pose"))
        object.__setattr__(
            self,
            "goal_relative_progress",
            _scalar(self.goal_relative_progress, "goal_relative_progress", unit=True),
        )
        object.__setattr__(
            self,
            "contact_or_grasp_state",
            _flag(self.contact_or_grasp_state, "contact_or_grasp_state"),
        )
        object.__setattr__(self, "success", _flag(self.success, "success"))
        object.__setattr__(
            self,
            "branch_outcome",
            _fixed(self.branch_outcome, (BRANCH_OUTCOME_DIM,), "branch_outcome"),
        )
        margin = _scalar(self.workspace_and_joint_margin, "workspace_and_joint_margin", unit=True)
        object.__setattr__(self, "workspace_and_joint_margin", margin)
        object.__setattr__(
            self, "unexpected_collision", _flag(self.unexpected_collision, "unexpected_collision")
        )
        object.__setattr__(
            self, "severe_penetration", _flag(self.severe_penetration, "severe_penetration")
        )


@dataclass(frozen=True, slots=True)
class A4TaskSignals:
    deployment: A4DeploymentSignals
    diagnostics: A4PrivilegedSignals

    def eligibility_fields(self) -> dict[str, bool | float]:
        """Return only the signal fields consumed by A4 eligibility."""

        return {
            "success": bool(self.diagnostics.success),
            "active_stage": bool(self.diagnostics.active_stage),
            "workspace_and_joint_margin": float(
                self.diagnostics.workspace_and_joint_margin
            ),
            "unexpected_collision": bool(self.diagnostics.unexpected_collision),
        }


def _task_name(task_name: str) -> str:
    if not isinstance(task_name, str):
        raise TypeError("A4 task_name must be a string")
    if task_name not in TASKS:
        raise ValueError(f"unsupported A4 task {task_name!r}; expected one of {TASKS}")
    return task_name


def _required(env: Any, name: str) -> Any:
    try:
        value = getattr(env, name)
    except (AttributeError, RuntimeError) as exc:
        raise A4SignalError(f"environment is missing required signal {name!r}") from exc
    if value is None:
        raise A4SignalError(f"environment signal {name!r} is not initialized")
    return value


def _resolved_info(
    env: Any, observation: np.ndarray, info: Mapping[str, Any] | None
) -> dict[str, Any]:
    if info is not None and not isinstance(info, Mapping):
        raise TypeError("A4 step info must be a mapping")
    result = dict(info or {})
    missing = {"success", "in_place_reward"} - set(result)
    if missing:
        evaluate = getattr(env, "evaluate_state", None)
        action_space = getattr(env, "action_space", None)
        if not callable(evaluate) or action_space is None:
            raise A4SignalError(
                f"step info is missing {sorted(missing)} and evaluate_state() is unavailable"
            )
        low = np.asarray(getattr(action_space, "low", None), dtype=np.float32)
        if low.ndim != 1 or not np.isfinite(low).all():
            raise A4SignalError("action_space.low must be a finite vector")
        try:
            _, evaluated = evaluate(observation, np.zeros_like(low))
        except Exception as exc:
            raise A4SignalError("evaluate_state() could not recover required task signals") from exc
        if not isinstance(evaluated, Mapping):
            raise A4SignalError("evaluate_state() info must be a mapping")
        result = {**evaluated, **result}
    missing = {"success", "in_place_reward"} - set(result)
    if missing:
        raise A4SignalError(f"task info is missing required signals {sorted(missing)}")
    return result


def _xyzw(value: Any, name: str, *, wxyz: bool = False) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise A4SignalError(f"{name} must be a finite quaternion with shape (4,)")
    if wxyz:
        quat = quat[[1, 2, 3, 0]]
    norm = float(np.linalg.norm(quat))
    if norm <= np.finfo(np.float64).eps:
        raise A4SignalError(f"{name} quaternion has zero norm")
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def _body_pose(env: Any, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    data = _required(env, "data")
    body = getattr(data, "body", None)
    if not callable(body):
        raise A4SignalError("environment data must provide body(name)")
    try:
        state = body(body_name)
        return (
            _array(state.xpos, (3,), f"body {body_name!r} position"),
            _xyzw(state.xquat, f"body {body_name!r}", wxyz=True),
        )
    except A4SignalError:
        raise
    except Exception as exc:
        raise A4SignalError(f"required body {body_name!r} is unavailable") from exc


def _object_pose(env: Any, task_name: str) -> tuple[np.ndarray, np.ndarray]:
    if task_name == "drawer-retrieve-v0":
        return _body_pose(env, "block")
    if task_name == "towel-wipe-v0":
        return _body_pose(env, "towel")

    get_position = getattr(env, "_get_pos_objects", None)
    if not callable(get_position):
        raise A4SignalError("environment must provide _get_pos_objects()")
    try:
        position = _array(get_position(), (3,), "object position")
    except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
        raise A4SignalError("could not read object position") from exc

    if task_name == "window-open-v3":
        _, quaternion = _body_pose(env, "windowb_a")
        return position, quaternion
    get_quaternion = getattr(env, "_get_quat_objects", None)
    if not callable(get_quaternion):
        raise A4SignalError("environment must provide _get_quat_objects()")
    try:
        quaternion = _xyzw(get_quaternion(), "object")
    except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
        raise A4SignalError("could not read object quaternion") from exc
    return position, quaternion


def _reference_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    position = _array(_required(env, "_target_pos"), (3,), "_target_pos")
    quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quaternion.setflags(write=False)
    return position, quaternion


def _goal_state(
    env: Any,
    task_name: str,
    object_position: np.ndarray,
    target: np.ndarray,
    task_progress: float,
) -> np.float32:
    controlled = (
        _array(_required(env, "tcp_center"), (3,), "tcp_center")
        if task_name == "reach-v3"
        else object_position
    )
    delta = target - controlled
    if task_name == "towel-wipe-v0":
        return np.float32(task_progress)

    axes = _GOAL_AXES[task_name]
    current_distance = float(np.linalg.norm(delta[list(axes)]))
    if task_name == "reach-v3":
        initial_distance = float(_required(env, "maxReachDist"))
    elif task_name == "door-open-v3":
        initial_distance = float(_required(env, "maxPullDist"))
    elif task_name == "window-open-v3":
        initial = _array(
            _required(env, "window_handle_pos_init"),
            (3,),
            "window_handle_pos_init",
        )
        initial_distance = float(abs(initial[0] - target[0]))
    else:
        initial = _array(_required(env, "obj_init_pos"), (3,), "obj_init_pos")
        initial_distance = float(np.linalg.norm((target - initial)[list(axes)]))
    if not np.isfinite(initial_distance) or initial_distance <= 0.0:
        raise A4SignalError(f"{task_name} initial goal distance must be positive")
    return np.float32(
        np.clip(1.0 - current_distance / initial_distance, 0.0, 1.0)
    )


def _task_progress(env: Any, task_name: str, info: Mapping[str, Any]) -> np.float32:
    in_place = _scalar(info["in_place_reward"], "in_place_reward", unit=True)
    if task_name != "drawer-retrieve-v0":
        return in_place
    drawer = _scalar(_required(env, "drawer_open_fraction"), "drawer_open_fraction", unit=True)
    if drawer < np.float32(0.95):
        return np.float32(0.25 * float(drawer))
    return np.float32(0.25 + 0.75 * float(in_place))


def _model_name(model: Any, kind: str, index: int) -> str:
    accessor = getattr(model, kind, None)
    if not callable(accessor):
        raise A4SignalError(f"MuJoCo model must provide {kind}(id)")
    try:
        return str(getattr(accessor(index), "name", "") or "")
    except Exception as exc:
        raise A4SignalError(f"could not resolve MuJoCo {kind} id {index}") from exc


def _contacts(env: Any) -> list[tuple[str, str, str, str, float]]:
    model = _required(env, "model")
    data = _required(env, "data")
    storage = getattr(data, "contact", None)
    body_ids = getattr(model, "geom_bodyid", None)
    if storage is None or body_ids is None:
        raise A4SignalError("MuJoCo contact and geom_bodyid state are required")
    try:
        ncon = int(getattr(data, "ncon", len(storage)))
        if ncon < 0 or ncon > len(storage):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise A4SignalError("MuJoCo data.ncon is inconsistent with contacts") from exc
    result = []
    for index in range(ncon):
        contact = storage[index]
        try:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            distance = float(contact.dist)
            body1, body2 = int(body_ids[geom1]), int(body_ids[geom2])
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise A4SignalError("malformed MuJoCo contact record") from exc
        if not np.isfinite(distance):
            raise A4SignalError("MuJoCo contact distance must be finite")
        result.append(
            (
                _model_name(model, "geom", geom1),
                _model_name(model, "body", body1),
                _model_name(model, "geom", geom2),
                _model_name(model, "body", body2),
                distance,
            )
        )
    return result


def _robot(geom: str, body: str) -> bool:
    return (
        body in {"right_hand", "hand", "rightclaw", "leftclaw", "rightpad", "leftpad"}
        or body.startswith("right_l")
        or any(token in geom.lower() for token in ("claw", "pad_geom", "hand_geom"))
    )


def _expected(task_name: str, geom: str, body: str) -> bool:
    if task_name in {"pick-place-v3", "push-v3"}:
        return geom == "objGeom" and body == "obj"
    if task_name == "drawer-retrieve-v0":
        return (geom == "drawer_handle_geom" and body == "drawer") or body == "block"
    if task_name == "door-open-v3":
        return body == "door_link"
    if task_name == "window-open-v3":
        return body == "windowb_a"
    if task_name == "towel-wipe-v0":
        return body == "towel"
    return False


def _contact_and_safety(
    env: Any, task_name: str
) -> tuple[np.bool_, np.bool_, np.bool_]:
    expected_contact = False
    worst_unexpected = 0.0
    for geom1, body1, geom2, body2, distance in _contacts(env):
        robot1, robot2 = _robot(geom1, body1), _robot(geom2, body2)
        if robot1 == robot2:
            if robot1:
                worst_unexpected = min(worst_unexpected, distance)
            continue
        other_geom, other_body = (geom2, body2) if robot1 else (geom1, body1)
        if _expected(task_name, other_geom, other_body):
            expected_contact |= distance <= 0.0
        else:
            worst_unexpected = min(worst_unexpected, distance)

    if task_name == "reach-v3":
        contact = False
    elif task_name == "drawer-retrieve-v0":
        contact = bool(_flag(_required(env, "is_block_grasped"), "is_block_grasped")) or expected_contact
    elif task_name == "towel-wipe-v0":
        contact = bool(_flag(_required(env, "towel_grasped"), "towel_grasped"))
    elif task_name in {"pick-place-v3", "push-v3"}:
        touching = getattr(env, "touching_main_object", expected_contact)
        contact = bool(_flag(touching, "touching_main_object"))
    else:
        contact = expected_contact

    sim_exception = bool(
        _flag(_required(env, "_did_see_sim_exception"), "_did_see_sim_exception")
    )
    unexpected = sim_exception or worst_unexpected < -1e-5
    severe = sim_exception or worst_unexpected < -SEVERE_PENETRATION_METERS
    return np.bool_(contact), np.bool_(unexpected), np.bool_(severe)


def workspace_control_margin(env: Any) -> np.float32:
    """Return directional headroom for the action-controlled mocap target.

    MetaWorld clips Cartesian actions against ``mocap_low/high``.  Comparing
    those bounds with ``tcp_center`` is invalid because the fingertips have a
    task-dependent offset from the mocap body and legitimately extend below
    the mocap workspace during contact.  The historical scalar also used the
    minimum of all six signed directions, which rejected every planar contact
    whenever the downward direction was saturated.

    This metric is the second-smallest of the six normalized signed
    headrooms.  A state may therefore have one intentionally constrained
    direction (for example downward at the table) while states constrained in
    two or more directions remain below the frozen 0.05 safety threshold.
    """

    data = _required(env, "data")
    positions = np.asarray(_required(data, "mocap_pos"), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
        raise A4SignalError(
            f"data.mocap_pos must have shape [M,3] with M>=1, got {positions.shape}"
        )
    if not np.isfinite(positions).all():
        raise A4SignalError("data.mocap_pos must contain only finite values")
    position = positions[0]
    low = _array(_required(env, "mocap_low"), (3,), "mocap_low").astype(np.float64)
    high = _array(_required(env, "mocap_high"), (3,), "mocap_high").astype(np.float64)
    width = high - low
    if np.any(width <= 0.0):
        raise A4SignalError("mocap workspace bounds must be strictly ordered")
    signed_headroom = np.concatenate(
        ((position - low) / width, (high - position) / width)
    )
    if bool(np.any(signed_headroom < -1e-6)):
        return np.float32(0.0)
    signed_headroom = np.clip(signed_headroom, 0.0, 1.0)
    return np.float32(np.partition(signed_headroom, 1)[1])


def _margin(env: Any, task_name: str) -> np.float32:
    """Return task-independent controller headroom for eligibility."""

    del task_name
    return workspace_control_margin(env)


def extract_a4_task_signals(
    task_name: str,
    env: Any,
    observation: Any,
    info: Mapping[str, Any] | None = None,
) -> A4TaskSignals:
    """Extract one deployment/privileged signal pair from the current state."""

    task_name = _task_name(task_name)
    observation_array = _array(observation, (OBSERVATION_DIM,), "A4 observation")
    task_info = _resolved_info(env, observation_array, info)
    success = _flag(task_info["success"], "success")
    task_progress = _task_progress(env, task_name, task_info)

    object_position, object_quaternion = _object_pose(env, task_name)
    target_position, target_quaternion = _reference_pose(env)
    goal_progress = _goal_state(
        env,
        task_name,
        object_position,
        target_position,
        float(task_progress),
    )
    contact, unexpected, severe = _contact_and_safety(env, task_name)

    onehot = np.zeros(len(TASKS), dtype=np.float32)
    onehot[TASKS.index(task_name)] = 1.0
    object_pose = np.concatenate((object_position, object_quaternion)).astype(np.float32)
    reference_pose = np.concatenate((target_position, target_quaternion)).astype(np.float32)
    outcome = np.asarray([float(task_progress), float(success)], dtype=np.float32)

    return A4TaskSignals(
        deployment=A4DeploymentSignals(
            task_onehot=onehot,
            proprio=observation_array[:PROPRIO_DIM].copy(),
        ),
        diagnostics=A4PrivilegedSignals(
            active_stage=np.bool_(not bool(success)),
            task_progress=task_progress,
            object_pose=object_pose,
            reference_pose=reference_pose,
            goal_relative_progress=goal_progress,
            contact_or_grasp_state=contact,
            success=success,
            branch_outcome=outcome,
            workspace_and_joint_margin=_margin(env, task_name),
            unexpected_collision=unexpected,
            severe_penetration=severe,
        ),
    )


__all__ = [
    "A4DeploymentSignals",
    "A4PrivilegedSignals",
    "A4SignalError",
    "A4TaskSignals",
    "BRANCH_OUTCOME_DIM",
    "OBSERVATION_DIM",
    "POSE_DIM",
    "PROPRIO_DIM",
    "SEVERE_PENETRATION_METERS",
    "TASKS",
    "extract_a4_task_signals",
    "workspace_control_margin",
]
