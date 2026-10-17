"""Jaw coverage of the neck at the pose that reaches it. Uses mj_forward."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import joint_index
from small_part.tube_cycle import JOINTS
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, hole_xy, load_tube_model


def main():
    model, data = load_tube_model()
    arm = [joint_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    base = np.array([-0.30, 0.60, -0.40, 1.371, 1.60, 0.70])
    for grip in (0.70, 0.30, 0.10, 0.0, -0.10):
        q = base.copy()
        q[5] = grip
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
        for address, value in zip(arm, q, strict=True):
            data.qpos[address] = float(value)
        mujoco.mj_forward(model, data)
        print(f"\ngrip {grip:+.2f}")
        for fi in range(model.nflex):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fi)
            adr = int(model.flex_vertadr[fi])
            n = int(model.flex_vertnum[fi])
            xpos = np.asarray(data.flexvert_xpos[adr : adr + n])
            d = np.linalg.norm(xpos[:, :2] - source, axis=1)
            k = int(np.argmin(d))
            band = (xpos[:, 2] > 0.145) & (xpos[:, 2] < 0.212)
            db = d[band]
            print(
                f"  {name:12s} closest {d[k]:.4f} at z={xpos[k, 2]:.3f} "
                f"dy={xpos[k, 1] - source[1]:+.3f} dx={xpos[k, 0] - source[0]:+.3f} "
                f"neck_min={db.min() if len(db) else -1:.4f} n_neck={band.sum()}"
            )


if __name__ == "__main__":
    main()
