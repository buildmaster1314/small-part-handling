"""Kinematic snap of candidate pinches, then one physics close."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index
from small_part.tube_cycle import JOINTS, _tooth_midpoint
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, hole_xy, load_tube_model, tube_position, tube_tilt


def report(model, data, tag):
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    tube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tube")
    print(
        f"{tag} tooth={np.round(_tooth_midpoint(model, data), 4)} "
        f"tube={np.round(data.xpos[tube], 4)} grip={np.round(data.xpos[gripper], 4)} "
        f"jaw={np.round(data.xpos[jaw], 4)} q={np.round([data.qpos[joint_index(model, n)] for n in JOINTS], 3)}"
    )
    # flex vertex z range
    for fi in range(model.nflex):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fi)
        adr = int(model.flex_vertadr[fi])
        n = int(model.flex_vertnum[fi])
        # world positions of flex verts are in data.flexvert_xpos? 
        xpos = data.flexvert_xpos[adr : adr + n]
        print(
            f"  {name} z[{xpos[:, 2].min():.3f},{xpos[:, 2].max():.3f}] "
            f"x[{xpos[:, 0].min():.3f},{xpos[:, 0].max():.3f}] "
            f"y[{xpos[:, 1].min():.3f},{xpos[:, 1].max():.3f}] n={n}"
        )


def jaw_on_tube(model, data):
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if int(c.flex[0]) < 0 and int(c.flex[1]) < 0:
            continue
        g = c.geom1 if c.geom1 >= 0 else c.geom2
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) if g >= 0 else "?"
        if name and name.startswith("tube"):
            n += 1
    return n


def main():
    model, data = load_tube_model()
    actuators = [actuator_index(model, n) for n in JOINTS]
    arm = [joint_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    print("has flexvert", hasattr(data, "flexvert_xpos"), "nflex", model.nflex)

    candidates = [
        np.array([-0.30, 0.60, -0.40, 1.371, 1.60, 0.40]),
        np.array([-0.30, 0.50, -0.20, 1.271, 1.60, 0.40]),
        np.array([-0.30, 0.40, -0.20, 1.371, 1.60, 0.40]),
        np.array([-0.2966, 0.4721, -0.3742, 1.4729, 1.5781, 0.40]),  # IN_SOCKET open
    ]
    for q in candidates:
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
        for address, value, act in zip(arm, q, actuators, strict=True):
            data.qpos[address] = float(value)
            data.ctrl[act] = float(value)
        mujoco.mj_forward(model, data)
        report(model, data, "snap")
        # close the jaw kinematically a bit and see penetration
        address = joint_index(model, "gripper")
        for g in (0.20, 0.05, -0.10):
            data.qpos[address] = g
            mujoco.mj_forward(model, data)
            print(f"    grip q={g:+.2f} tube contacts={jaw_on_tube(model, data)} ncon={data.ncon}")


if __name__ == "__main__":
    main()
