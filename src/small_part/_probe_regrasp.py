"""Close, then walk the pinch down the wall before lifting out."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _set_friction
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
        if not name.startswith("tube"):
            continue
        for f in (int(c.flex[0]), int(c.flex[1])):
            if f >= 0:
                per[f] += 1
    return per


def go(model, data, actuators, goal, steps):
    start = np.array([data.ctrl[a] for a in actuators])
    goal = np.asarray(goal, float)
    worst = 0.0
    for i in range(1, steps + 1):
        data.ctrl[actuators] = start + (goal - start) * i / steps
        mujoco.mj_step(model, data)
        worst = max(worst, tube_tilt(data))
    return worst


def main():
    model, data = load_tube_model()
    _set_friction(model, 2.2, 0.08, 0.02, 6)
    actuators = [actuator_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    q = np.array([-0.30, 0.60, -0.40, 1.371, 1.60, 0.70])
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    data.qpos[joint_index(model, "gripper")] = 0.70
    data.ctrl[actuators[5]] = 0.70
    mujoco.mj_forward(model, data)
    for _ in range(30):
        mujoco.mj_step(model, data)
    hover = q.copy(); hover[1] -= 0.45; hover[3] += 0.45
    go(model, data, actuators, hover, 140)
    go(model, data, actuators, q, 360)
    goal = q.copy(); goal[5] = -0.12
    w = go(model, data, actuators, goal, 500)
    print("closed", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3), "jaw", jaw(model, data), "w", round(w, 3))

    # slide the hand down the tube: increase shoulder lift, flex cancels pitch
    for k in range(8):
        g = np.array([data.ctrl[a] for a in actuators])
        g[1] += 0.025
        g[3] -= 0.025
        g[5] = -0.12
        w = max(w, go(model, data, actuators, g, 40))
        print(f"  down {k} z={tube_position(data)[2]:.3f} tilt={tube_tilt(data):.3f} jaw={jaw(model, data)} q1={data.qpos[joint_index(model, 'shoulder_lift')]:.3f}")
        if tube_tilt(data) > 0.3:
            break

    need = RACK_Z + 0.074 + TUBE_HALF_LENGTH + 0.015
    for k in range(16):
        g = np.array([data.ctrl[a] for a in actuators])
        g[1] -= 0.02
        g[3] += 0.02
        g[5] = -0.12
        w = max(w, go(model, data, actuators, g, 30))
        z = float(tube_position(data)[2])
        print(f"  up {k} z={z:.3f} tilt={tube_tilt(data):.3f} jaw={jaw(model, data)}")
        if z >= need and tube_tilt(data) < 0.10:
            print("LIFTED", round(w, 3))
            return
        if tube_tilt(data) > 0.45:
            print("spill")
            return
    print("fail", round(float(tube_position(data)[2]), 3), round(tube_tilt(data), 3), "worst", round(w, 3))


if __name__ == "__main__":
    main()
