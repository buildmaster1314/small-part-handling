"""Yaw as soon as the teeth are on, while the hole still holds the tube."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _set_friction
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


def go(model, data, actuators, goal, steps):
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, steps + 1):
        data.ctrl[actuators] = start + (np.asarray(goal, float) - start) * i / steps
        mujoco.mj_step(model, data)


def run(sign, step, grip):
    model, data = load_tube_model()
    _set_friction(model, 2.2, 0.08, 0.015, 6)
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
    for _ in range(20):
        mujoco.mj_step(model, data)
    hover = q.copy(); hover[1] -= 0.45; hover[3] += 0.45
    go(model, data, actuators, hover, 100)
    go(model, data, actuators, q, 260)
    closed = q.copy(); closed[5] = grip
    go(model, data, actuators, closed, 300)
    print(f"\nsign={sign:+} step={step} closed tilt={tube_tilt(data):.3f} z={tube_position(data)[2]:.3f}")
    hold = np.array([data.ctrl[a] for a in actuators])
    worst = tube_tilt(data)
    best_bottom = 0.0
    for k in range(22):
        hold = hold.copy()
        hold[0] += sign * step
        hold[5] = grip
        go(model, data, actuators, hold, 24)
        z = float(tube_position(data)[2])
        tilt = tube_tilt(data)
        worst = max(worst, tilt)
        bottom = z - TUBE_HALF_LENGTH * np.cos(tilt)
        best_bottom = max(best_bottom, bottom)
        if k % 2 == 0:
            print(f"  {k:02d} z={z:.3f} bot={bottom:.3f} tilt={tilt:.3f} xy={np.round(tube_position(data)[:2], 3)}")
        if bottom > RACK_Z + 0.078 and tilt < 0.10:
            print("  CLEAR", round(worst, 3))
            return
        if tilt > 0.55:
            print("  spill", round(worst, 3), "best bot", round(best_bottom, 3))
            return
    print("  end", round(float(tube_position(data)[2]), 3), round(tube_tilt(data), 3), "best", round(best_bottom, 3))


if __name__ == "__main__":
    for sign in (1, -1):
        run(sign, 0.05, -0.17)
