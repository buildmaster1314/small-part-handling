"""Descend onto the neck under physics and close. Optionally ignore the lip."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _set_friction, _tooth_midpoint
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


def silence_lips(model):
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "lip" in name or "post" in name:
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0


def tube_jaw(model, data):
    n = 0
    z = []
    for i in range(data.ncon):
        c = data.contact[i]
        if int(c.flex[0]) < 0 and int(c.flex[1]) < 0:
            continue
        g = c.geom1 if c.geom1 >= 0 else c.geom2
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) if g >= 0 else ""
        if name and name.startswith("tube"):
            n += 1
            z.append(c.pos[2])
    return n, (float(np.mean(z)) if z else 0.0)


def run(q_pinch, *, grip, no_lip, friction=(1.6, 0.04, 0.008, 6)):
    model, data = load_tube_model()
    if no_lip:
        silence_lips(model)
    slide, spin, roll, condim = friction
    _set_friction(model, slide, spin, roll, condim)
    actuators = [actuator_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    gi = joint_index(model, "gripper")
    data.qpos[gi] = 0.55
    data.ctrl[actuators[5]] = 0.55
    mujoco.mj_forward(model, data)
    for _ in range(60):
        mujoco.mj_step(model, data)

    # come in from a higher shoulder (smaller lift value) with the jaw open
    hover = q_pinch.copy()
    hover[1] -= 0.45
    hover[3] += 0.45
    hover[5] = 0.55
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, 181):
        data.ctrl[actuators] = start + (hover - start) * i / 180
        mujoco.mj_step(model, data)

    # descend while beginning to close in the last third
    start = np.array([data.ctrl[a] for a in actuators])
    goal = q_pinch.copy()
    goal[5] = grip
    steps = 500
    worst = 0.0
    for i in range(1, steps + 1):
        alpha = i / steps
        cmd = start + (goal - start) * alpha
        # keep the jaw open until the hand is most of the way down
        if alpha < 0.65:
            cmd[5] = 0.55 + (0.30 - 0.55) * (alpha / 0.65)
        data.ctrl[actuators] = cmd
        mujoco.mj_step(model, data)
        worst = max(worst, tube_tilt(data))
        if i % 100 == 0:
            n, zc = tube_jaw(model, data)
            tooth = _tooth_midpoint(model, data)
            print(
                f"  a={alpha:.2f} tooth_z={tooth[2]:.3f} tube_z={tube_position(data)[2]:.3f} "
                f"tilt={tube_tilt(data):.3f} jaw_on_tube={n} zc={zc:.3f} "
                f"q1={data.qpos[joint_index(model, 'shoulder_lift')]:.3f}/{goal[1]:.3f}"
            )
    n, zc = tube_jaw(model, data)
    print(
        f"CLOSED lip_off={no_lip} grip={grip:+.2f} z={tube_position(data)[2]:.3f} "
        f"tilt={tube_tilt(data):.3f} worst={worst:.3f} on_tube={n} zc={zc:.3f} "
        f"q={np.round([data.qpos[joint_index(model, n)] for n in JOINTS], 3)}"
    )
    need = RACK_Z + 0.074 + TUBE_HALF_LENGTH + 0.015
    for k in range(24):
        s0 = np.array([data.ctrl[a] for a in actuators])
        g = s0.copy()
        g[1] -= 0.018
        g[2] -= 0.012
        g[3] += 0.012
        g[5] = grip
        for i in range(1, 21):
            data.ctrl[actuators] = s0 + (g - s0) * i / 20
            mujoco.mj_step(model, data)
            worst = max(worst, tube_tilt(data))
        z = float(tube_position(data)[2])
        tilt = tube_tilt(data)
        if k % 4 == 0:
            n, zc = tube_jaw(model, data)
            print(f"  up{k:02d} z={z:.3f} tilt={tilt:.3f} on_tube={n} zc={zc:.3f}")
        if z >= need and tilt < 0.10:
            print("  LIFTED", "worst", round(worst, 3))
            return True
        if tilt > 0.45:
            print("  spill")
            return False
    print("  fail", round(float(tube_position(data)[2]), 3), round(tube_tilt(data), 3), "worst", round(worst, 3))
    return False


def main():
    q = np.array([-0.30, 0.60, -0.40, 1.371, 1.60, 0.40])
    for no_lip in (True, False):
        for grip in (0.02, -0.08):
            print("\n====", "nolip" if no_lip else "lip", grip)
            run(q, grip=grip, no_lip=no_lip)


if __name__ == "__main__":
    main()
