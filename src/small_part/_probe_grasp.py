"""Diagnose a friction-only side pinch. Not part of the task."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import (
    GRIP,
    IN_SOCKET,
    JOINTS,
    OPEN,
    PICK_LIFT,
    PINCH_PAN,
    ROLL,
    _clamp,
    _ranges,
    _set_friction,
    _tooth_midpoint,
)
from small_part.tube_scene import (
    HOLE_RADIUS,
    RACK_HEIGHT,
    RACK_Z,
    SOURCE_ROW,
    TUBE_CENTER_Z,
    TUBE_HALF_LENGTH,
    TUBE_RADIUS,
    hole_xy,
    load_tube_model,
    tube_position,
    tube_tilt,
)


def contacts(model, data, label):
    print(f"\n== {label} ncon={data.ncon} tilt={tube_tilt(data):.4f} pos={np.round(tube_position(data), 4)}")
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
        # flex contacts use geom id -1 sometimes? check
        names = []
        for g, b in ((c.geom1, b1), (c.geom2, b2)):
            if g >= 0:
                names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}")
            else:
                names.append(f"flex/body:{b}")
        # flex: geom is -1, flex is set
        f1 = int(c.flex[0]) if hasattr(c, "flex") else -1
        print(
            f"  {names[0]:28s} {names[1]:28s} dist={c.dist:.5f} "
            f"pos={np.round(c.pos, 4)} flex={tuple(int(x) for x in c.flex)} "
            f"geom=({c.geom1},{c.geom2})"
        )


def main():
    model, data = load_tube_model()
    actuators = [actuator_index(model, name) for name in JOINTS]
    arm = [joint_index(model, name) for name in JOINTS]
    tube = joint_index(model, "tube_free")
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    source = hole_xy(SOURCE_ROW, 1)
    print("source", source, "center_z", TUBE_CENTER_Z, "half", TUBE_HALF_LENGTH, "radius", TUBE_RADIUS)
    print("rack top", RACK_Z + RACK_HEIGHT, "hole r", HOLE_RADIUS)
    print("upper lip z ~", RACK_Z + RACK_HEIGHT + 0.001)
    print("mouth z", TUBE_CENTER_Z + TUBE_HALF_LENGTH)

    # flex info
    for i in range(model.nflex):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, i)
        print(f"flex {i} {name} condim={model.flex_condim[i]} friction={model.flex_friction[i]} "
              f"contype={model.flex_contype[i]} conaffinity={model.flex_conaffinity[i]}")

    _set_friction(model, 1.4, 0.02, 0.002, 4)
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    # settle tube in rack
    for _ in range(400):
        mujoco.mj_step(model, data)
    print("settled", np.round(tube_position(data), 4), "tilt", tube_tilt(data))

    # park the recorded pinch kinematically and report tooth vs tube
    ranges = _ranges(model)
    seed = IN_SOCKET.copy()
    seed[0] = PINCH_PAN
    seed[1] = PICK_LIFT
    seed[4] = ROLL
    seed[5] = OPEN
    for address, value in zip(arm, seed, strict=True):
        data.qpos[address] = float(value)
        # also ctrl
    for a, v in zip(actuators, seed, strict=True):
        data.ctrl[a] = float(v)
    mujoco.mj_forward(model, data)
    mid = _tooth_midpoint(model, data)
    axis = data.xmat[gripper].reshape(3, 3)
    print("seed tooth mid", np.round(mid, 4))
    print("gripper xpos", np.round(data.xpos[gripper], 4))
    print("gripper xmat cols\n", np.round(axis, 3))
    print("tube", np.round(tube_position(data), 4))
    print("gap tooth-tube xy", np.round(mid[:2] - tube_position(data)[:2], 4), "dz", mid[2] - tube_position(data)[2])

    # jaw geoms z range at this pose
    for index in range(model.ngeom):
        body = int(model.geom_bodyid[index])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name in ("gripper", "moving_jaw_so101_v1", "wrist"):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
            print(f"  geom {index} {gname:20s} body={name:22s} contype={model.geom_contype[index]} posz={data.geom_xpos[index][2]:.4f}")

    # now physically approach: open jaw, go to seed from above, close slowly
    # reset tube settled, arm at home open, then interpolate ctrl to seed
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    for a, name in zip(actuators, JOINTS, strict=True):
        if name == "gripper":
            data.ctrl[a] = OPEN
            data.qpos[joint_index(model, name)] = OPEN
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)

    goal = seed.copy()
    goal[5] = OPEN
    start = np.array([data.ctrl[a] for a in actuators], float)
    for i in range(1, 401):
        data.ctrl[actuators] = start + (goal - start) * i / 400
        mujoco.mj_step(model, data)
    print("\napproached q", np.round([data.qpos[joint_index(model, n)] for n in JOINTS], 3))
    print("tube", np.round(tube_position(data), 4), "tilt", round(tube_tilt(data), 4))
    print("tooth", np.round(_tooth_midpoint(model, data), 4))
    contacts(model, data, "open at pinch height")

    # close
    start = np.array([data.ctrl[a] for a in actuators], float)
    goal = start.copy()
    goal[5] = GRIP
    for i in range(1, 801):
        data.ctrl[actuators] = start + (goal - start) * i / 800
        mujoco.mj_step(model, data)
    print("\nclosed q grip", round(float(data.qpos[joint_index(model, "gripper")]), 4), "cmd", GRIP)
    print("tube", np.round(tube_position(data), 4), "tilt", round(tube_tilt(data), 4))
    contacts(model, data, "closed")

    # try a small lift
    start = np.array([data.ctrl[a] for a in actuators], float)
    goal = start.copy()
    goal[1] -= 0.12
    goal[2] -= 0.12
    goal[3] += 0.12
    for i in range(1, 401):
        data.ctrl[actuators] = start + (goal - start) * i / 400
        mujoco.mj_step(model, data)
        if i % 100 == 0:
            print(f"  lift {i} z={tube_position(data)[2]:.4f} tilt={tube_tilt(data):.3f} xy={np.round(tube_position(data)[:2], 4)}")
    contacts(model, data, "after lift attempt")


if __name__ == "__main__":
    main()
