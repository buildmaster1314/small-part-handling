"""Who is saturating while the hand tries to rise."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _set_friction
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, hole_xy, load_tube_model, tube_position, tube_tilt


def go(model, data, actuators, goal, steps):
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, steps + 1):
        data.ctrl[actuators] = start + (np.asarray(goal, float) - start) * i / steps
        mujoco.mj_step(model, data)


def main():
    model, data = load_tube_model()
    for name in ("shoulder_lift", "elbow_flex", "wrist_flex", "gripper"):
        a = actuator_index(model, name)
        print(name, "forcerange", model.actuator_forcerange[a], "kp", model.actuator_gainprm[a, 0])
    _set_friction(model, 2.4, 0.1, 0.02, 6)
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
    for _ in range(10):
        mujoco.mj_step(model, data)
    hover = q.copy(); hover[1] -= 0.45; hover[3] += 0.45
    go(model, data, actuators, hover, 80)
    go(model, data, actuators, q, 200)
    closed = q.copy(); closed[5] = -0.17
    go(model, data, actuators, closed, 200)
    # command a clearly higher shoulder and watch the force
    goal = np.array([data.ctrl[a] for a in actuators])
    goal[1] -= 0.30
    goal[3] += 0.30
    start = goal.copy() * 0 + np.array([data.ctrl[a] for a in actuators])
    for i in range(1, 201):
        data.ctrl[actuators] = start + (goal - start) * i / 200
        mujoco.mj_step(model, data)
        if i % 40 == 0:
            qq = [data.qpos[joint_index(model, n)] for n in JOINTS]
            print(
                f"i={i} q={np.round(qq, 3)} ctrl={np.round(data.ctrl[actuators], 3)} "
                f"force={np.round(data.actuator_force, 2)} z={tube_position(data)[2]:.3f} tilt={tube_tilt(data):.3f}"
            )


if __name__ == "__main__":
    main()
