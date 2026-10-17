"""Yaw until the tube crests, then lift before the teeth slip off."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import actuator_index, joint_index, set_home
from small_part.tube_cycle import JOINTS, _clamp, _ranges, _set_friction, _tooth_midpoint
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


def go(model, data, actuators, goal, steps):
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, steps + 1):
        data.ctrl[actuators] = start + (np.asarray(goal, float) - start) * i / steps
        mujoco.mj_step(model, data)


def bottom(data):
    tilt = tube_tilt(data)
    return float(tube_position(data)[2] - TUBE_HALF_LENGTH * np.cos(tilt)), tilt


def move_tooth(model, data, arm, ranges, delta, grip):
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    q = np.array([data.qpos[a] for a in arm], float)
    parked = data.qpos.copy()
    anchor = [None]

    def pinch(qv):
        for address, value in zip(arm, qv, strict=True):
            data.qpos[address] = float(value)
        mujoco.mj_kinematics(model, data)
        origin = data.xpos[gripper]
        rot = data.xmat[gripper].reshape(3, 3)
        if anchor[0] is None:
            anchor[0] = rot.T @ (_tooth_midpoint(model, data) - origin)
        return origin + rot @ anchor[0]

    goal = pinch(q) + np.asarray(delta, float)
    for _ in range(10):
        point = pinch(q)
        error = goal - point
        if np.linalg.norm(error) < 6e-4:
            break
        cols = []
        for j in range(4):
            probe = q.copy()
            probe[j] += 0.005
            cols.append((pinch(probe) - point) / 0.005)
        jac = np.column_stack(cols)
        step, *_ = np.linalg.lstsq(jac, error, rcond=None)
        q = q.copy()
        q[:4] += np.clip(step, -0.012, 0.012)
        _clamp(q, ranges)
    data.qpos[:] = parked
    mujoco.mj_kinematics(model, data)
    q[5] = grip
    return q


def main():
    model, data = load_tube_model()
    _set_friction(model, 2.4, 0.1, 0.02, 6)
    actuators = [actuator_index(model, n) for n in JOINTS]
    arm = [joint_index(model, n) for n in JOINTS]
    ranges = _ranges(model)
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    grip = -0.17
    # Wrist flex saturates at its 1.658 stop, so the descent stops where the
    # wrist still has travel: lift and elbow give the height, flex stays inside.
    q = np.array([-0.30, 0.35, -0.15, 1.45, 1.60, 0.70])
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    data.qpos[arm[5]] = 0.70
    data.ctrl[actuators[5]] = 0.70
    mujoco.mj_forward(model, data)
    for _ in range(15):
        mujoco.mj_step(model, data)
    hover = q.copy(); hover[1] -= 0.85; hover[3] += 0.85
    go(model, data, actuators, hover, 120)
    print("hover", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3))
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, 241):
        data.ctrl[actuators] = start + (q - start) * i / 240
        mujoco.mj_step(model, data)
        if i in (160, 240):
            qq = [data.qpos[joint_index(model, n)] for n in JOINTS]
            print(f"  down {i} tilt={tube_tilt(data):.3f} q={np.round(qq, 3)} force={np.round(data.actuator_force, 1)}")
    closed = q.copy(); closed[5] = grip
    start = np.array([data.ctrl[a] for a in actuators])
    for i in range(1, 901):
        data.ctrl[actuators] = start + (closed - start) * i / 900
        mujoco.mj_step(model, data)
        if i % 300 == 0:
            qq = [data.qpos[joint_index(model, n)] for n in JOINTS]
            print(
                f"  close {i} tilt={tube_tilt(data):.3f} q={np.round(qq, 3)} "
                f"force={np.round(data.actuator_force, 2)}"
            )

    hold = np.array([data.ctrl[a] for a in actuators])
    worst = tube_tilt(data)
    best = 0.0
    for k in range(14):
        hold = hold.copy(); hold[0] += 0.045; hold[5] = grip
        go(model, data, actuators, hold, 22)
        bot, tilt = bottom(data)
        worst = max(worst, tilt)
        if bot < best - 0.004:
            print(f"crest at {k} bot={bot:.3f} best={best:.3f} tilt={tilt:.3f}")
            break
        best = max(best, bot)
        print(f"  yaw {k:02d} bot={bot:.3f} tilt={tilt:.3f}")
    print("at crest", np.round(tube_position(data), 3), "tilt", round(tube_tilt(data), 3), "worst", round(worst, 3))

    for k in range(40):
        goal = move_tooth(model, data, arm, ranges, (0.0, 0.0, 0.0010), grip)
        go(model, data, actuators, goal, 40)
        bot, tilt = bottom(data)
        worst = max(worst, tilt)
        if k % 3 == 0:
            print(f"  up {k:02d} bot={bot:.3f} tilt={tilt:.3f} z={tube_position(data)[2]:.3f}")
        if tilt > 0.16 and k > 4:
            print(f"  leaning {k} bot={bot:.3f} tilt={tilt:.3f}")
            break
        if bot > RIM + 0.006 and tilt < 0.10:
            print("CLEAR", round(worst, 3), np.round(tube_position(data), 3))
            return
        if tilt > 0.45:
            print("spill")
            return
    print("stuck", round(float(tube_position(data)[2]), 3), round(tube_tilt(data), 3), "worst", round(worst, 3))
    # Slide the hand, and the tube with it, off the hole. Yawing tips it.
    for k in range(20):
        goal = move_tooth(model, data, arm, ranges, (0.004, -0.002, 0.0), grip)
        go(model, data, actuators, goal, 30)
        bot, tilt = bottom(data)
        worst = max(worst, tilt)
        print(f"  slide {k:02d} bot={bot:.3f} tilt={tilt:.3f} xy={np.round(tube_position(data)[:2], 3)}")
        if bot > RIM + 0.004 and tilt < 0.10 and np.linalg.norm(tube_position(data)[:2] - source) > 0.03:
            print("CLEAR", round(worst, 3))
            return
        if tilt > 0.45:
            print("spill")
            return
    print("slide fail")


if __name__ == "__main__":
    main()
