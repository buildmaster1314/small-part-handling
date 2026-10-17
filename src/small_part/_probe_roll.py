"""Sweep wrist roll for a pinch where both jaws reach the tube wall."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import joint_index
from small_part.tube_cycle import JOINTS, _ranges
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, TUBE_RADIUS, hole_xy, load_tube_model


def main():
    model, data = load_tube_model()
    arm = [joint_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    ranges = _ranges(model)
    source = hole_xy(SOURCE_ROW, 1)
    base = np.array([-0.30, 0.55, -0.35, 1.40, 1.60, 0.90])
    print("roll range", ranges[4])

    def jaw_reach(q):
        for address, value in zip(arm, q, strict=True):
            data.qpos[address] = float(value)
        data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
        mujoco.mj_kinematics(model, data)
        out = []
        for fi in range(model.nflex):
            adr = int(model.flex_vertadr[fi])
            n = int(model.flex_vertnum[fi])
            xpos = np.asarray(data.flexvert_xpos[adr : adr + n])
            # only the part of the tooth that can sit on the neck: above the
            # lower plate, and not buried in the table
            band = xpos[(xpos[:, 2] > 0.125) & (xpos[:, 2] < 0.214)]
            if len(band) == 0:
                out.append((1.0, 0))
                continue
            dxy = np.linalg.norm(band[:, :2] - source, axis=1)
            out.append((float(dxy.min()), int(np.sum(dxy < TUBE_RADIUS + 0.004))))
        return out

    hits = []
    for roll in np.linspace(ranges[4][0] + 0.05, ranges[4][1] - 0.05, 19):
        for pan in np.linspace(-0.7, 0.15, 8):
            for lift in np.linspace(0.2, 1.1, 7):
                for elbow in np.linspace(-1.0, 0.6, 7):
                    flex = 1.45 - lift - elbow
                    if not (ranges[3][0] < flex < ranges[3][1]):
                        continue
                    q = np.array([pan, lift, elbow, flex, roll, 0.9])
                    (d0, n0), (d1, n1) = jaw_reach(q)
                    if d0 < 0.014 and d1 < 0.014 and n0 > 2 and n1 > 2:
                        hits.append((d0 + d1, n0 + n1, q.copy(), d0, d1, n0, n1))
    hits.sort(key=lambda h: h[0])
    print("hits", len(hits))
    for row in hits[:10]:
        _, _, q, d0, d1, n0, n1 = row
        print(f"  sum={d0 + d1:.4f} fixed d={d0:.4f} n={n0} moving d={d1:.4f} n={n1} q={np.round(q, 3)}")


if __name__ == "__main__":
    main()
