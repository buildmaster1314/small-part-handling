"""Close onto the seated pad and lift straight up. Diagnostic only."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _set_friction
from small_part.tube_scene import (
    RACK_HEIGHT,
    RACK_Z,
    SOURCE_ROW,
    TUBE_CENTER_Z,
    TUBE_HALF_LENGTH,
    hole_xy,
    load_tube_model,
    tube_position,
    tube_tilt,
)

RIM = RACK_Z + RACK_HEIGHT


def contacts(model, data):
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        names = []
        for geom, flex in ((c.geom1, c.flex[0]), (c.geom2, c.flex[1])):
            if geom >= 0:
                names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or "?")
            else:
                names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, int(flex)) or "?")
        if any("tube" in name or "pad" in name for name in names):
            rows.append(f"{names[0]}|{names[1]}@{c.pos[2]:.3f}")
    return rows


def go(model, data, actuators, goal, steps):
    start = np.array([data.ctrl[a] for a in actuators], float)
    goal = np.asarray(goal, float)
    for i in range(1, steps + 1):
        data.ctrl[actuators] = start + (goal - start) * i / steps
        mujoco.mj_step(model, data)


def main():
    model, data = load_tube_model()
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_pad")
    model.geom_friction[pad] = [2.4, 0.1, 0.02]
    _set_friction(model, 2.4, 0.1, 0.02, 6)
    actuators = [actuator_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    q = np.array([-0.30, 0.35, -0.15, 1.45, 1.60, 0.70])
    grip = 0.20
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1, 0, 0, 0)
    data.qpos[joint_index(model, "gripper")] = 0.70
    data.ctrl[actuators[5]] = 0.70
    mujoco.mj_forward(model, data)
    for _ in range(20):
        mujoco.mj_step(model, data)
    hover = q.copy()
    hover[1] -= 0.85
    hover[3] += 0.85
    go(model, data, actuators, hover, 100)
    print("hover", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3))
    go(model, data, actuators, q, 280)
    print(
        "down",
        np.round(tube_position(data), 3),
        "tilt",
        round(tube_tilt(data), 3),
        "pad",
        np.round(data.geom_xpos[pad], 3),
    )
    closed = q.copy()
    closed[5] = grip
    go(model, data, actuators, closed, 500)
    print(
        "closed tilt",
        round(tube_tilt(data), 3),
        "gq",
        round(float(data.qpos[joint_index(model, "gripper")]), 3),
        "force",
        np.round(data.actuator_force, 1),
    )
    from collections import Counter

    print(Counter(contacts(model, data)).most_common(8))
    worst = tube_tilt(data)
    hold = np.array([data.ctrl[a] for a in actuators], float)
    for k in range(48):
        hold = hold.copy()
        hold[1] -= 0.010
        hold[3] += 0.010
        hold[5] = grip
        go(model, data, actuators, hold, 24)
        z = float(tube_position(data)[2])
        tilt = tube_tilt(data)
        worst = max(worst, tilt)
        bottom = z - TUBE_HALF_LENGTH * np.cos(tilt)
        if k % 3 == 0 or k > 32:
            print(
                f"  up {k:02d} z={z:.3f} bot={bottom:.3f} tilt={tilt:.3f} "
                f"padz={data.geom_xpos[pad][2]:.3f} q1={data.qpos[joint_index(model, 'shoulder_lift')]:.3f}"
            )
        if z > 0.185 and tilt < 0.08:
            print("high", round(worst, 3), np.round(tube_position(data), 3))
            break
        if tilt > 0.25:
            print("spill during lift", round(worst, 3))
            return
    else:
        print("never rose", round(float(tube_position(data)[2]), 3))
        return
    # Reverse the lift and open. If the tube stands back up in the hole, the
    # pinch itself is not wrecking it; only the last centimetre is.
    back = hold.copy()
    back[1] += 0.25
    back[3] -= 0.25
    back[5] = 0.70
    go(model, data, actuators, back, 400)
    for _ in range(200):
        mujoco.mj_step(model, data)
    print("after release", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3), "worst", round(worst, 3))


if __name__ == "__main__":
    main()
