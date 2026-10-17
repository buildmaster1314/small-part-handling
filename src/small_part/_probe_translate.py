"""After the tube is half out and upright, slide the hand sideways off the hole."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _clamp, _ranges, _set_friction, _tooth_midpoint
from small_part.tube_scene import (
    RACK_Z,
    SOURCE_ROW,
    TUBE_CENTER_Z,
    TUBE_HALF_LENGTH,
    hole_xy,
    load_tube_model,
    tube_position,
    tube_tilt,
)


def go(model, data, actuators, goal, steps):
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, steps + 1):
        data.ctrl[actuators] = start + (np.asarray(goal, float) - start) * i / steps
        mujoco.mj_step(model, data)


def move_tooth(model, data, arm, ranges, delta, grip):
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    q = np.array([data.qpos[a] for a in arm], float)
    parked = data.qpos.copy()
    anchor = [None]

    def pinch(qv):
        for address, value in zip(arm, qv, strict=True):
            data.qpos[address] = float(value)
        mujoco.mj_kinematics(model, data)
        origin = data.xpos[gripper]
        rot = data.xmat[gripper].reshape(3, 3)
        if anchor[0] is None:
            anchor[0] = rot.T @ (_tooth_midpoint(model, data) - origin)
        return origin + rot @ anchor[0]

    start = pinch(q).copy()
    goal = start + np.asarray(delta, float)
    for _ in range(12):
        point = pinch(q)
        error = goal - point
        if np.linalg.norm(error) < 5e-4:
            break
        cols = []
        for j in range(4):
            probe = q.copy()
            probe[j] += 0.005
            cols.append((pinch(probe) - point) / 0.005)
        jac = np.column_stack(cols)
        step, *_ = np.linalg.lstsq(jac, error, rcond=None)
        q = q.copy()
        q[:4] += np.clip(step, -0.015, 0.015)
        _clamp(q, ranges)
    data.qpos[:] = parked
    mujoco.mj_kinematics(model, data)
    q[5] = grip
    return q


def prep(model, data):
    _set_friction(model, 2.2, 0.08, 0.015, 6)
    actuators = [actuator_index(model, n) for n in JOINTS]
    arm = [joint_index(model, n) for n in JOINTS]
    ranges = _ranges(model)
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    q = np.array([-0.30, 0.60, -0.40, 1.371, 1.60, 0.70])
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    data.qpos[arm[5]] = 0.70
    data.ctrl[actuators[5]] = 0.70
    mujoco.mj_forward(model, data)
    for _ in range(20):
        mujoco.mj_step(model, data)
    hover = q.copy(); hover[1] -= 0.45; hover[3] += 0.45
    go(model, data, actuators, hover, 120)
    go(model, data, actuators, q, 300)
    closed = q.copy(); closed[5] = -0.17
    go(model, data, actuators, closed, 350)
    hold = np.array([data.ctrl[a] for a in actuators])
    for _ in range(8):
        hold = hold.copy(); hold[0] += 0.055; hold[5] = -0.17
        go(model, data, actuators, hold, 28)
    return actuators, arm, ranges


def main():
    model, data = load_tube_model()
    actuators, arm, ranges = prep(model, data)
    print("start", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3))
    # slide toward +x, away from the shoulder, which is the open side of the rack
    for k in range(18):
        goal = move_tooth(model, data, arm, ranges, (0.006, 0.0, 0.001), -0.17)
        go(model, data, actuators, goal, 24)
        z = float(tube_position(data)[2])
        tilt = tube_tilt(data)
        bottom = z - TUBE_HALF_LENGTH * np.cos(tilt)
        print(f"  {k:02d} pos={np.round(tube_position(data), 3)} bottom={bottom:.3f} tilt={tilt:.3f}")
        if bottom > RACK_Z + 0.078 and tilt < 0.10 and tube_position(data)[0] > 0.50:
            print("OFF")
            return
        if tilt > 0.45:
            print("spill")
            return
    print("fail")


if __name__ == "__main__":
    main()
