"""Joint-space motion and a fixed pick-transport-place sequence."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.scene import ARM_JOINTS, GRIPPER_CLOSED, GRIPPER_JOINT, GRIPPER_OPEN, HOME_QPOS


def joint_index(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(name)
    return int(model.jnt_qposadr[joint_id])


def actuator_index(model: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise KeyError(name)
    return int(actuator_id)


def set_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name, value in zip((*ARM_JOINTS, GRIPPER_JOINT), HOME_QPOS, strict=True):
        data.qpos[joint_index(model, name)] = value
        data.ctrl[actuator_index(model, name)] = value
    mujoco.mj_forward(model, data)


def move_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: dict[str, float],
    *,
    duration: float = 1.0,
    speed: float = 1.0,
) -> None:
    """Linearly interpolate position targets. Speed scales the motion time."""

    if speed <= 0:
        raise ValueError("speed must be positive")
    steps = max(1, int(duration / speed / model.opt.timestep))
    start = {name: float(data.ctrl[actuator_index(model, name)]) for name in target}
    for step in range(1, steps + 1):
        alpha = step / steps
        for name, goal in target.items():
            data.ctrl[actuator_index(model, name)] = (1 - alpha) * start[name] + alpha * goal
        mujoco.mj_step(model, data)


def gripper(model: mujoco.MjModel, data: mujoco.MjData, closed: bool, *, duration: float = 0.4) -> None:
    value = GRIPPER_CLOSED if closed else GRIPPER_OPEN
    move_joints(model, data, {GRIPPER_JOINT: value}, duration=duration)


def pick_transport_place(model: mujoco.MjModel, data: mujoco.MjData, *, speed: float = 1.0) -> None:
    """Run one scripted handling cycle from the known home pose.

    The joint targets are a first motion scaffold for the table scene. Inverse
    kinematics and vision-based target updates replace these constants next.
    """

    set_home(model, data)
    approach = {
        "shoulder_pan": -0.35,
        "shoulder_lift": -0.15,
        "elbow_flex": 0.85,
        "wrist_flex": 0.55,
        "wrist_roll": 0.0,
    }
    place = {
        "shoulder_pan": 0.45,
        "shoulder_lift": -0.15,
        "elbow_flex": 0.85,
        "wrist_flex": 0.55,
        "wrist_roll": 0.0,
    }
    gripper(model, data, closed=False, duration=0.3 / speed)
    move_joints(model, data, approach, duration=1.2, speed=speed)
    gripper(model, data, closed=True, duration=0.4 / speed)
    move_joints(model, data, place, duration=1.4, speed=speed)
    gripper(model, data, closed=False, duration=0.4 / speed)


def part_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part")
    return np.array(data.xpos[body_id], dtype=float)


def place_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "place_target")
    return float(np.linalg.norm(part_position(model, data) - data.site_xpos[site_id]))
