"""Can the moving jaw touch the tube anywhere below the rack rim?"""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import joint_index
from small_part.tube_cycle import JOINTS
from small_part.tube_scene import RACK_HEIGHT, RACK_Z, SOURCE_ROW, TUBE_CENTER_Z, TUBE_RADIUS, hole_xy, load_tube_model

RIM = RACK_Z + RACK_HEIGHT


def main():
    model, data = load_tube_model()
    arm = [joint_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    source = hole_xy(SOURCE_ROW, 1)
    best = None
    for lift in np.linspace(0.30, 0.70, 17):
        for elbow in np.linspace(-0.5, 0.3, 9):
            for flex in np.linspace(1.0, 1.62, 7):
                for grip in (0.6, 0.3, 0.05):
                    q = np.array([-0.30, lift, elbow, flex, 1.60, grip])
                    data.qpos[:] = 0
                    data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1, 0, 0, 0)
                    for address, value in zip(arm, q, strict=True):
                        data.qpos[address] = float(value)
                    mujoco.mj_forward(model, data)
                    gaps = []
                    for fi in range(model.nflex):
                        adr = int(model.flex_vertadr[fi])
                        n = int(model.flex_vertnum[fi])
                        xpos = np.asarray(data.flexvert_xpos[adr : adr + n])
                        low = xpos[xpos[:, 2] < RIM - 0.002]
                        if len(low) == 0:
                            gaps.append(1.0)
                            continue
                        d = np.linalg.norm(low[:, :2] - source, axis=1) - TUBE_RADIUS
                        gaps.append(float(d.min()))
                    if gaps[0] < 0 and gaps[1] < 0:
                        score = -(gaps[0] + gaps[1])
                        if best is None or score > best[0]:
                            best = (score, q.copy(), gaps)
                            print(f"  both below rim gaps={np.round(gaps, 4)} q={np.round(q, 3)}")
    print("BEST", None if best is None else (np.round(best[1], 3), np.round(best[2], 4)))


if __name__ == "__main__":
    main()
