"""Check contype bits and who still blocks the descent."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, hole_xy, load_tube_model


def main():
    model, data = load_tube_model()
    print("nflex", model.nflex)
    for i in range(model.nflex):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, i)
        print(f"flex {name} contype={model.flex_contype[i]} conaff={model.flex_conaffinity[i]}")
    interesting = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if any(k in name for k in ("tube", "lip", "post", "wall", "shelf")):
            interesting.append(
                f"{name:28s} type={model.geom_contype[i]} aff={model.geom_conaffinity[i]}"
            )
    print("sample", len(interesting))
    for line in interesting[:8]:
        print(" ", line)
    lips = [l for l in interesting if "lip" in l]
    print("lip example", lips[0] if lips else None, "count", len(lips))
    print("tube", [l for l in interesting if l.startswith("tube")])

    # descend and print blocking contacts + actuator force
    actuators = [actuator_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    q = np.array([-0.30, 0.60, -0.40, 1.371, 1.60, 0.55])
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    data.ctrl[actuators[5]] = 0.55
    data.qpos[joint_index(model, "gripper")] = 0.55
    mujoco.mj_forward(model, data)
    for _ in range(30):
        mujoco.mj_step(model, data)
    hover = q.copy(); hover[1] -= 0.45; hover[3] += 0.45
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, 101):
        data.ctrl[actuators] = start + (hover - start) * i / 100
        mujoco.mj_step(model, data)
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, 501):
        data.ctrl[actuators] = start + (q - start) * i / 500
        mujoco.mj_step(model, data)
    print("q", np.round([data.qpos[joint_index(model, n)] for n in JOINTS], 3))
    print("ctrl", np.round([data.ctrl[a] for a in actuators], 3))
    # actuator force
    print("actuator force", np.round(data.actuator_force, 3))
    for i in range(data.ncon):
        c = data.contact[i]
        def side(geom, flex):
            if geom >= 0:
                return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or f"g{int(geom)}"
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, int(flex)) or f"f{int(flex)}"
        print(f"  {side(c.geom1, c.flex[0]):24s} {side(c.geom2, c.flex[1]):24s} {np.round(c.pos, 3)}")


if __name__ == "__main__":
    main()
