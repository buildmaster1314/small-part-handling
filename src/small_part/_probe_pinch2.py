"""Physics pinch from the pose where both jaws meet the wall."""

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


def on_tube(model, data):
    per = [0, 0]
    zs = [[], []]
    for i in range(data.ncon):
        c = data.contact[i]
        flexes = [int(c.flex[0]), int(c.flex[1])]
        g = c.geom1 if c.geom1 >= 0 else c.geom2
        if g < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
        if not name.startswith("tube"):
            continue
        for f in flexes:
            if f >= 0:
                per[f] += 1
                zs[f].append(c.pos[2])
    return per, [float(np.mean(z)) if z else 0.0 for z in zs]


def trial(grip_goal, friction):
    model, data = load_tube_model()
    _set_friction(model, *friction)
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
    for _ in range(40):
        mujoco.mj_step(model, data)

    hover = q.copy()
    hover[1] -= 0.40
    hover[3] += 0.40
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, 161):
        data.ctrl[actuators] = start + (hover - start) * i / 160
        mujoco.mj_step(model, data)
    start = np.array([data.ctrl[a] for a in actuators])
    down = q.copy()
    for i in range(1, 401):
        data.ctrl[actuators] = start + (down - start) * i / 400
        mujoco.mj_step(model, data)
    per, zs = on_tube(model, data)
    print(
        f"\ngrip->{grip_goal:+.2f} down q={np.round([data.qpos[joint_index(model, n)] for n in JOINTS], 3)} "
        f"tilt={tube_tilt(data):.3f} jaw={per} z={np.round(zs, 3)}"
    )
    start = np.array([data.ctrl[a] for a in actuators])
    goal = start.copy()
    goal[5] = grip_goal
    worst = tube_tilt(data)
    for i in range(1, 601):
        data.ctrl[actuators] = start + (goal - start) * i / 600
        mujoco.mj_step(model, data)
        worst = max(worst, tube_tilt(data))
    per, zs = on_tube(model, data)
    print(
        f"  closed tilt={tube_tilt(data):.3f} worst={worst:.3f} z={tube_position(data)[2]:.3f} "
        f"jaw={per} zc={np.round(zs, 3)} gq={data.qpos[joint_index(model, 'gripper')]:.3f}"
    )
    need = RACK_Z + 0.074 + TUBE_HALF_LENGTH + 0.012
    for k in range(18):
        s0 = np.array([data.ctrl[a] for a in actuators])
        g = s0.copy()
        g[1] -= 0.02
        g[3] += 0.02
        g[5] = grip_goal
        for i in range(1, 21):
            data.ctrl[actuators] = s0 + (g - s0) * i / 20
            mujoco.mj_step(model, data)
            worst = max(worst, tube_tilt(data))
        z = float(tube_position(data)[2])
        tilt = tube_tilt(data)
        if k % 3 == 0:
            per, zs = on_tube(model, data)
            print(f"  up{k:02d} z={z:.3f} tilt={tilt:.3f} jaw={per} zc={np.round(zs, 3)}")
        if z >= need and tilt < 0.10:
            print("  LIFTED", round(worst, 3))
            return
        if tilt > 0.5:
            print("  spill")
            return
    print("  fail", round(float(tube_position(data)[2]), 3), round(tube_tilt(data), 3))


def main():
    for fr in ((1.8, 0.05, 0.01, 6), (2.8, 0.1, 0.02, 6)):
        for grip in (0.05, -0.05, -0.15):
            trial(grip, fr)


if __name__ == "__main__":
    main()
