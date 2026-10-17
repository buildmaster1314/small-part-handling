"""Where can the teeth actually meet the neck? Not part of the task."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import joint_index, set_home
from small_part.tube_cycle import JOINTS, OPEN, ROLL, _clamp, _ranges, _tooth_midpoint
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, TUBE_RADIUS, hole_xy, load_tube_model


def main():
    model, data = load_tube_model()
    arm = [joint_index(model, name) for name in JOINTS]
    ranges = _ranges(model)
    tube_j = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    set_home(model, data)
    data.qpos[tube_j : tube_j + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)

    # neck band: from just above the upper plate (0.175) to the mouth (0.216)
    # tube axis xy is source
    target_xy = source
    heights = np.linspace(0.178, 0.210, 5)

    def measure(q):
        saved = data.qpos.copy()
        for address, value in zip(arm, q, strict=True):
            data.qpos[address] = float(value)
        mujoco.mj_kinematics(model, data)
        mid = _tooth_midpoint(model, data).copy()
        # jaw opening axis: gripper x axis from earlier was mostly horizontal
        axis = data.xmat[gripper].reshape(3, 3).copy()
        data.qpos[:] = saved
        return mid, axis

    # grid over pan, lift, elbow. flex cancels lift+elbow so pitch of the hand
    # stays at the recorded side-pinch pitch. roll fixed.
    from small_part.tube_cycle import IN_SOCKET

    pitch = float(IN_SOCKET[1] + IN_SOCKET[2] + IN_SOCKET[3])
    best = []
    pans = np.linspace(-0.7, 0.2, 10)
    lifts = np.linspace(-0.2, 1.2, 15)
    elbows = np.linspace(-1.2, 1.2, 13)
    for pan in pans:
        for lift in lifts:
            for elbow in elbows:
                q = np.array([pan, lift, elbow, pitch - lift - elbow, ROLL, OPEN])
                if q[3] < ranges[3][0] + 0.02 or q[3] > ranges[3][1] - 0.02:
                    continue
                _clamp(q, ranges)
                mid, axis = measure(q)
                # horizontal miss of the tooth centre from the tube axis
                miss = float(np.linalg.norm(mid[:2] - target_xy))
                # the teeth close along a horizontal direction; we want the
                # centre within ~tube radius of the axis so closing hits the wall
                if miss < 0.018 and 0.170 < mid[2] < 0.220:
                    # approach side: gripper x is the jaw plane normal-ish
                    best.append((miss, mid[2], q.copy(), axis[:, 2].copy()))
    best.sort(key=lambda item: (item[0], abs(item[1] - 0.195)))
    print("hits", len(best))
    for miss, z, q, up in best[:12]:
        print(
            f"  miss={miss:.4f} z={z:.3f} q={np.round(q, 3)} hand_up={np.round(up, 2)}"
        )

    # also: lowest tooth z achievable anywhere near the hole, regardless of pitch
    low = []
    for pan in np.linspace(-0.8, 0.3, 8):
        for lift in np.linspace(-0.4, 1.4, 10):
            for elbow in np.linspace(-1.4, 1.4, 10):
                for flex in np.linspace(-1.4, 1.4, 8):
                    q = np.array([pan, lift, elbow, flex, ROLL, OPEN])
                    ok = True
                    for i in range(4):
                        if q[i] < ranges[i][0] + 0.02 or q[i] > ranges[i][1] - 0.02:
                            ok = False
                    if not ok:
                        continue
                    mid, axis = measure(q)
                    miss = float(np.linalg.norm(mid[:2] - target_xy))
                    if miss < 0.025:
                        low.append((mid[2], miss, q.copy(), axis[:, 2].copy()))
    low.sort(key=lambda item: item[0])
    print("\nlowest near hole", len(low))
    for z, miss, q, up in low[:15]:
        print(f"  z={z:.3f} miss={miss:.4f} q={np.round(q, 3)} up={np.round(up, 2)}")


if __name__ == "__main__":
    main()
