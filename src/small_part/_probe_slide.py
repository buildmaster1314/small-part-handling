"""Draw the gripped tube sideways through the open side of the hole."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _set_friction
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, hole_xy, load_tube_model, tube_position, tube_tilt


def go(model, data, actuators, goal, steps):
    start = np.array([data.ctrl[a] for a in actuators], float)
    goal = np.asarray(goal, float)
    for i in range(1, steps + 1):
        data.ctrl[actuators] = start + (goal - start) * i / steps
        mujoco.mj_step(model, data)


def main():
    model, data = load_tube_model()
    _set_friction(model, 4.0, 0.2, 0.04, 6)
    actuators = [actuator_index(model, name) for name in JOINTS]
    arm = [joint_index(model, name) for name in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    q = np.array([-0.30, 0.23, -0.15, 1.57, 1.60, 0.70])
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1, 0, 0, 0)
    data.qpos[arm[5]] = 0.70
    data.ctrl[actuators] = q
    mujoco.mj_forward(model, data)
    for _ in range(30):
        mujoco.mj_step(model, data)
    go(model, data, actuators, q, 200)
    print("down", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3))
    closed = q.copy()
    closed[5] = -0.17
    go(model, data, actuators, closed, 400)
    print("shut", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3), "gq", round(float(data.qpos[arm[5]]), 3))
    hold = closed.copy()
    for k in range(30):
        hold = hold.copy()
        hold[0] += 0.006
        go(model, data, actuators, hold, 30)
        print(f"  {k:02d}", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3))
        if tube_tilt(data) > 0.4:
            break


if __name__ == "__main__":
    main()
