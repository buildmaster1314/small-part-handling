"""Squeeze harder, then lift the pinch straight up."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _ranges, _set_friction, _tooth_midpoint
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


def jaw(model, data):
    per = [0, 0]
    for i in range(data.ncon):
        c = data.contact[i]
        g = c.geom1 if c.geom1 >= 0 else c.geom2
        if g < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
        if name.startswith("tube"):
            for f in (int(c.flex[0]), int(c.flex[1])):
                if f >= 0:
                    per[f] += 1
    return per


def go(model, data, actuators, goal, steps):
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, steps + 1):
        data.ctrl[actuators] = start + (np.asarray(goal) - start) * i / steps
        mujoco.mj_step(model, data)


def ik_up(model, data, arm, ranges, height, grip):
    """Raise the tooth midpoint to ``height`` without sliding it sideways."""

    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    q = np.array([data.qpos[a] for a in arm])
    parked = data.qpos.copy()
    anchor = None

    def pinch(qv):
        nonlocal anchor
        for address, value in zip(arm, qv, strict=True):
            data.qpos[address] = float(value)
        mujoco.mj_kinematics(model, data)
        origin = data.xpos[gripper]
        rot = data.xmat[gripper].reshape(3, 3)
        if anchor is None:
            anchor = rot.T @ (_tooth_midpoint(model, data) - origin)
        return origin + rot @ anchor

    target_xy = pinch(q)[:2].copy()
    for _ in range(25):
        point = pinch(q)
        if point[2] >= height - 1e-3:
            break
        error = np.array([target_xy[0] - point[0], target_xy[1] - point[1], min(point[2] + 0.004, height) - point[2]])
        cols = []
        for j in range(4):
            probe = q.copy()
            probe[j] += 0.006
            cols.append((pinch(probe) - point) / 0.006)
        jac = np.column_stack(cols)
        step, *_ = np.linalg.lstsq(jac, error, rcond=None)
        q = q.copy()
        q[:4] += np.clip(step, -0.02, 0.02)
        for i in range(4):
            q[i] = float(np.clip(q[i], ranges[i][0] + 0.02, ranges[i][1] - 0.02))
    data.qpos[:] = parked
    mujoco.mj_kinematics(model, data)
    q[5] = grip
    return q


def main():
    model, data = load_tube_model()
    _set_friction(model, 2.0, 0.06, 0.01, 6)
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
    go(model, data, actuators, closed, 400)
    print(
        "closed", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3),
        "jaw", jaw(model, data), "gq", round(float(data.qpos[arm[5]]), 3),
        "tooth", np.round(_tooth_midpoint(model, data), 3),
    )
    need = RACK_Z + 0.074 + TUBE_HALF_LENGTH + 0.02
    worst = tube_tilt(data)
    for k in range(30):
        here = float(_tooth_midpoint(model, data)[2])
        goal = ik_up(model, data, arm, ranges, here + 0.003, -0.17)
        go(model, data, actuators, goal, 40)
        z = float(tube_position(data)[2])
        tilt = tube_tilt(data)
        worst = max(worst, tilt)
        if k % 3 == 0:
            print(f"  {k:02d} tube_z={z:.3f} tilt={tilt:.3f} tooth_z={_tooth_midpoint(model, data)[2]:.3f} jaw={jaw(model, data)}")
        # Stop while the mouth is still up. Past here the teeth outrun the tube.
        if z >= TUBE_CENTER_Z + 0.030 or tilt > 0.08:
            break
        if tilt > 0.5:
            print("spill", round(worst, 3))
            return
    print("parked", round(float(tube_position(data)[2]), 3), "tilt", round(tube_tilt(data), 3), "worst", round(worst, 3))

    # Swing the shoulder. The tube is already partly out; yawing drags the
    # bottom up the hole instead of pulling the teeth up the wall.
    hold = np.array([data.ctrl[a] for a in actuators])
    for k in range(9):
        hold = hold.copy()
        hold[0] += 0.06
        hold[5] = -0.17
        go(model, data, actuators, hold, 30)
        z = float(tube_position(data)[2])
        tilt = tube_tilt(data)
        worst = max(worst, tilt)
        bottom = z - TUBE_HALF_LENGTH * np.cos(tilt)
        print(f"  yaw {k:02d} z={z:.3f} bottom={bottom:.3f} tilt={tilt:.3f} xy={np.round(tube_position(data)[:2], 3)}")
        if bottom > RACK_Z + 0.074 + 0.004 and tilt < 0.10:
            print("CLEAR worst", round(worst, 3))
            return
        if tilt > 0.45:
            print("spill yaw")
            return
    print("after yaw", round(float(tube_position(data)[2]), 3), round(tube_tilt(data), 3), "worst", round(worst, 3))
    for k in range(20):
        here = float(_tooth_midpoint(model, data)[2])
        goal = ik_up(model, data, arm, ranges, here + 0.004, -0.17)
        goal[0] = float(data.ctrl[actuators[0]])
        go(model, data, actuators, goal, 30)
        z = float(tube_position(data)[2])
        tilt = tube_tilt(data)
        worst = max(worst, tilt)
        bottom = z - TUBE_HALF_LENGTH * np.cos(tilt)
        if k % 2 == 0:
            print(f"  relift {k:02d} z={z:.3f} bottom={bottom:.3f} tilt={tilt:.3f} tooth={_tooth_midpoint(model, data)[2]:.3f}")
        if bottom > RACK_Z + 0.080 and tilt < 0.10:
            print("CLEAR", round(worst, 3))
            return
        if tilt > 0.4:
            print("spill relift")
            return
    print("relift fail", round(float(tube_position(data)[2]), 3), round(tube_tilt(data), 3))


if __name__ == "__main__":
    main()
