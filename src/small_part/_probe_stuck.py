"""Reproduce the stall and print every contact on the tube."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part._probe_crest import bottom, go, move_tooth
from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _ranges, _set_friction
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, hole_xy, load_tube_model, tube_position, tube_tilt


def who(model, data):
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        def side(geom, flex):
            if geom >= 0:
                return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or "?"
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, int(flex)) or "?"
        a, b = side(c.geom1, c.flex[0]), side(c.geom2, c.flex[1])
        if "tube" in a or "tube" in b:
            rows.append(f"{a:20s} {b:20s} z={c.pos[2]:.3f}")
    return rows


def main():
    model, data = load_tube_model()
    _set_friction(model, 2.4, 0.1, 0.02, 6)
    actuators = [actuator_index(model, n) for n in JOINTS]
    arm = [joint_index(model, n) for n in JOINTS]
    ranges = _ranges(model)
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    grip = -0.17
    q = np.array([-0.30, 0.35, -0.15, 1.45, 1.60, 0.70])
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1, 0, 0, 0)
    data.qpos[arm[5]] = 0.70
    data.ctrl[actuators[5]] = 0.70
    mujoco.mj_forward(model, data)
    for _ in range(10):
        mujoco.mj_step(model, data)
    hover = q.copy(); hover[1] -= 0.85; hover[3] += 0.85
    go(model, data, actuators, hover, 80)
    go(model, data, actuators, q, 200)
    closed = q.copy(); closed[5] = grip
    go(model, data, actuators, closed, 300)
    for _ in range(16):
        goal = move_tooth(model, data, arm, ranges, (0.0, 0.0, 0.0012), grip)
        go(model, data, actuators, goal, 30)
    print("z", round(float(tube_position(data)[2]), 3), "tilt", round(tube_tilt(data), 3), "ncon", data.ncon)
    from collections import Counter
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        def side(geom, flex):
            if geom >= 0:
                return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or f"g{int(geom)}"
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, int(flex)) or f"f{int(flex)}"
        rows.append(f"{side(c.geom1, c.flex[0])} | {side(c.geom2, c.flex[1])}")
    for key, n in Counter(rows).most_common(20):
        print(f"  {n:3d} {key}")


if __name__ == "__main__":
    main()
