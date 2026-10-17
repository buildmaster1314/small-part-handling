"""Where on the jaw meshes is the surface that can touch the tube."""

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
    q = np.array([-0.30, 0.60, -0.40, 1.371, 1.60, 0.70])
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
    for address, value in zip(arm, q, strict=True):
        data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)

    for fi in range(model.nflex):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fi)
        adr = int(model.flex_vertadr[fi])
        n = int(model.flex_vertnum[fi])
        xpos = np.asarray(data.flexvert_xpos[adr : adr + n])
        dxy = np.linalg.norm(xpos[:, :2] - source, axis=1)
        close = xpos[dxy < 0.016]
        print(f"\n{name} verts {n} within 16mm of axis: {len(close)}")
        if len(close):
            print(f"  z [{close[:, 2].min():.3f}, {close[:, 2].max():.3f}]")
            print(f"  y [{close[:, 1].min():.3f}, {close[:, 1].max():.3f}] tube y {source[1]:.3f}")
            for band in np.linspace(close[:, 2].min(), close[:, 2].max(), 6):
                sel = close[np.abs(close[:, 2] - band) < 0.004]
                if len(sel):
                    print(
                        f"  z~{band:.3f} n={len(sel):3d} y[{sel[:, 1].min():.3f},{sel[:, 1].max():.3f}] "
                        f"x[{sel[:, 0].min():.3f},{sel[:, 0].max():.3f}]"
                    )
        # also the verts below the lip
        low = xpos[xpos[:, 2] < 0.176]
        near_low = low[np.linalg.norm(low[:, :2] - source, axis=1) < 0.02] if len(low) else low
        print(f"  verts below lip: {len(low)}, of which within 20mm xy: {len(near_low)}")


if __name__ == "__main__":
    main()
