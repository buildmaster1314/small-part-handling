"""What stops the shoulder, and how high are the tooth tips."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _set_friction
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, hole_xy, load_tube_model, tube_tilt


def flex_body(model, flex):
    # first vertex body
    adr = int(model.flex_vertadr[flex])
    body = int(model.flex_vertbodyid[adr]) if hasattr(model, "flex_vertbodyid") else -1
    return body


def contacts(model, data):
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        def side(geom, flex):
            if geom >= 0:
                return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or f"g{geom}"
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, int(flex)) or f"flex{flex}"
            return name
        a = side(c.geom1, c.flex[0])
        b = side(c.geom2, c.flex[1])
        rows.append((a, b, round(float(c.dist), 5), np.round(c.pos, 3)))
    return rows


def main():
    model, data = load_tube_model()
    print("flex_vertbodyid", hasattr(model, "flex_vertbodyid"))
    print("flex fields", [a for a in dir(model) if "flex" in a.lower()][:40])
    _set_friction(model, 1.6, 0.04, 0.008, 6)
    actuators = [actuator_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    q = np.array([-0.30, 0.60, -0.40, 1.371, 1.60, 0.55])
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    data.qpos[joint_index(model, "gripper")] = 0.55
    data.ctrl[actuators[5]] = 0.55
    mujoco.mj_forward(model, data)
    for _ in range(40):
        mujoco.mj_step(model, data)

    hover = q.copy()
    hover[1] -= 0.45
    hover[3] += 0.45
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, 121):
        data.ctrl[actuators] = start + (hover - start) * i / 120
        mujoco.mj_step(model, data)
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, 401):
        data.ctrl[actuators] = start + (q - start) * i / 400
        mujoco.mj_step(model, data)
    print("arrived q", np.round([data.qpos[joint_index(model, n)] for n in JOINTS], 3))
    print("tilt", round(tube_tilt(data), 3), "ncon", data.ncon)
    # group contacts
    from collections import Counter
    cnt = Counter()
    for a, b, dist, pos in contacts(model, data):
        key = tuple(sorted((a, b)))
        cnt[key] += 1
    for key, n in cnt.most_common(25):
        print(f"  {n:3d} {key}")

    # tooth tip: lowest 15 flex verts of each jaw and their xy vs tube
    for fi in range(model.nflex):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fi)
        adr = int(model.flex_vertadr[fi])
        n = int(model.flex_vertnum[fi])
        xpos = np.asarray(data.flexvert_xpos[adr : adr + n])
        order = np.argsort(xpos[:, 2])[:8]
        print(name, "lowest verts")
        for k in order:
            p = xpos[k]
            print(f"    {np.round(p, 4)}  dxy={np.linalg.norm(p[:2] - source):.4f}")


if __name__ == "__main__":
    main()
