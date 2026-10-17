"""Where a pad on the moving jaw would land, in that jaw's own frame."""

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
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    source = hole_xy(SOURCE_ROW, 1)
    q = np.array([-0.30, 0.35, -0.15, 1.45, 1.60, 0.70])
    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1, 0, 0, 0)
    for address, value in zip(arm, q, strict=True):
        data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)
    rot = data.xmat[jaw].reshape(3, 3)
    origin = data.xpos[jaw]
    # a point on the tube wall, on the side facing the jaw, at neck height
    wall = np.array([source[0], source[1] + 0.011, 0.195])
    local = rot.T @ (wall - origin)
    print("jaw origin", np.round(origin, 4))
    print("wall", np.round(wall, 4), "in jaw frame", np.round(local, 4))
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_pad")
    print("pad world", np.round(data.geom_xpos[pad], 4), "size", model.geom_size[pad])


if __name__ == "__main__":
    main()
