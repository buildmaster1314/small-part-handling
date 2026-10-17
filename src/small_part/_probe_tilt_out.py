"""Pinch, then tip the tube out over the rim instead of pulling it straight up."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part._probe_crest import bottom, go, move_tooth
from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _ranges, _set_friction
from small_part.tube_scene import RACK_HEIGHT, RACK_Z, SOURCE_ROW, TUBE_CENTER_Z, hole_xy, load_tube_model, tube_position

RIM = RACK_Z + RACK_HEIGHT


def run(direction):
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
    print(f"\ndir={direction}")
    dx, dy = direction
    for k in range(30):
        goal = move_tooth(model, data, arm, ranges, (dx, dy, 0.0015), grip)
        go(model, data, actuators, goal, 30)
        bot, tilt = bottom(data)
        if k % 3 == 0:
            print(f"  {k:02d} bot={bot:.3f} tilt={tilt:.3f} xy={np.round(tube_position(data)[:2], 3)}")
        if bot > RIM + 0.006 and tilt < 0.35:
            print("  OUT", round(tilt, 3))
            return True
        if tilt > 1.2:
            print("  fell")
            return False
    print("  stuck", round(bot, 3), round(tilt, 3))
    return False


if __name__ == "__main__":
    for direction in ((0.003, 0.0), (-0.003, 0.0), (0.0, 0.003), (0.0, -0.003)):
        run(direction)
