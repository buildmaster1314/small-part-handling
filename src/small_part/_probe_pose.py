"""Distance from each jaw to the tube wall at a candidate pose."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import joint_index
from small_part.tube_cycle import JOINTS
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, TUBE_RADIUS, hole_xy, load_tube_model


def main():
    model, data = load_tube_model()
    arm = [joint_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    candidates = [
        np.array([-0.30, lift, -0.15, 1.45, 1.60, 0.5]) for lift in (0.40, 0.48, 0.55, 0.62)
    ]
    for q in candidates:
        data.qpos[:] = 0
        data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
        for address, value in zip(arm, q, strict=True):
            data.qpos[address] = float(value)
        mujoco.mj_forward(model, data)
        print(f"\ngrip {q[5]:+.2f}")
        for fi in range(model.nflex):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fi)
            adr = int(model.flex_vertadr[fi])
            n = int(model.flex_vertnum[fi])
            xpos = np.asarray(data.flexvert_xpos[adr : adr + n])
            d = np.linalg.norm(xpos[:, :2] - source, axis=1) - TUBE_RADIUS
            k = int(np.argmin(d))
            print(f"  {name:12s} gap={d[k]:+.4f} at z={xpos[k, 2]:.3f} dy={xpos[k, 1] - source[1]:+.3f}")


if __name__ == "__main__":
    main()
