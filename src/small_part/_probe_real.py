"""Physically close and lift from a pose whose teeth meet the neck."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, OPEN, _set_friction, _tooth_midpoint
from small_part.tube_scene import (
    RACK_Z,
    SOURCE_ROW,
    TARGET_ROW,
    TUBE_CENTER_Z,
    TUBE_HALF_LENGTH,
    hole_xy,
    load_tube_model,
    tube_position,
    tube_tilt,
)


def jaw_contacts(model, data):
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        if int(c.flex[0]) < 0 and int(c.flex[1]) < 0:
            continue
        g = c.geom1 if c.geom1 >= 0 else c.geom2
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) if g >= 0 else "?"
        rows.append((name, float(c.dist), np.round(c.pos, 4), int(c.flex[0]), int(c.flex[1])))
    return rows


def trial(q_pinch, grip, friction, label):
    model, data = load_tube_model()
    actuators = [actuator_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    slide, spin, roll, condim = friction
    _set_friction(model, slide, spin, roll, condim)
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    gi = joint_index(model, "gripper")
    data.qpos[gi] = OPEN
    data.ctrl[actuators[5]] = OPEN
    mujoco.mj_forward(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)

    # approach from a raised version of the same pinch (smaller shoulder lift)
    raised = q_pinch.copy()
    raised[1] -= 0.35
    raised[3] += 0.35  # keep pitch
    raised[5] = OPEN
    start = np.array([data.ctrl[a] for a in actuators], float)
    for i in range(1, 201):
        data.ctrl[actuators] = start + (raised - start) * i / 200
        mujoco.mj_step(model, data)
    start = np.array([data.ctrl[a] for a in actuators], float)
    goal = q_pinch.copy()
    goal[5] = OPEN
    for i in range(1, 251):
        data.ctrl[actuators] = start + (goal - start) * i / 250
        mujoco.mj_step(model, data)
    tooth = _tooth_midpoint(model, data)
    pos = tube_position(data)
    print(
        f"\n{label} open tooth={np.round(tooth, 4)} tube={np.round(pos, 4)} "
        f"tilt={tube_tilt(data):.3f} dz={tooth[2] - pos[2]:+.3f}"
    )

    start = np.array([data.ctrl[a] for a in actuators], float)
    goal = start.copy()
    goal[5] = grip
    worst = tube_tilt(data)
    for i in range(1, 701):
        data.ctrl[actuators] = start + (goal - start) * i / 700
        mujoco.mj_step(model, data)
        worst = max(worst, tube_tilt(data))
    print(
        f"  closed z={tube_position(data)[2]:.4f} tilt={tube_tilt(data):.3f} "
        f"worst={worst:.3f} grip_q={data.qpos[gi]:.3f}"
    )
    rows = jaw_contacts(model, data)
    print(f"  jaw contacts {len(rows)}")
    for row in rows[:6]:
        print("   ", row)

    need = RACK_Z + 0.074 + TUBE_HALF_LENGTH + 0.012
    for k in range(20):
        s0 = np.array([data.ctrl[a] for a in actuators], float)
        g = s0.copy()
        g[1] -= 0.02
        g[2] -= 0.02
        g[3] += 0.02
        g[5] = grip
        for i in range(1, 26):
            data.ctrl[actuators] = s0 + (g - s0) * i / 25
            mujoco.mj_step(model, data)
            worst = max(worst, tube_tilt(data))
        z = float(tube_position(data)[2])
        tilt = tube_tilt(data)
        if k % 3 == 0 or z > need or tilt > 0.25:
            print(f"  up {k:02d} z={z:.4f} tilt={tilt:.3f} xy={np.round(tube_position(data)[:2], 4)}")
        if z >= need and tilt < 0.12:
            print("  LIFTED")
            return True
        if tilt > 0.4:
            print("  spilled")
            return False
    print(f"  stop z={tube_position(data)[2]:.4f} tilt={tube_tilt(data):.3f} worst={worst:.3f}")
    return False


def main():
    q = np.array([-0.30, 0.60, -0.40, 1.371, 1.60, 0.40])
    frictions = {
        "rubber": (1.8, 0.05, 0.01, 6),
        "sticky": (2.6, 0.1, 0.02, 6),
        "mild": (1.0, 0.02, 0.002, 4),
    }
    for name, fr in frictions.items():
        for grip in (0.05, -0.05, -0.15):
            trial(q, grip, fr, f"{name} grip={grip:+.2f}")


if __name__ == "__main__":
    main()
