"""Find a pose where the two jaws sit on opposite sides of the tube."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.control import joint_index
from small_part.tube_cycle import JOINTS, _ranges
from small_part.tube_scene import SOURCE_ROW, TUBE_CENTER_Z, hole_xy, load_tube_model


def main():
    model, data = load_tube_model()
    arm = [joint_index(model, n) for n in JOINTS]
    tube = joint_index(model, "tube_free")
    ranges = _ranges(model)
    source = hole_xy(SOURCE_ROW, 1)

    def sides(q, grip):
        qq = q.copy()
        qq[5] = grip
        for address, value in zip(arm, qq, strict=True):
            data.qpos[address] = float(value)
        data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
        mujoco.mj_kinematics(model, data)
        info = []
        for fi in range(model.nflex):
            adr = int(model.flex_vertadr[fi])
            n = int(model.flex_vertnum[fi])
            xpos = np.asarray(data.flexvert_xpos[adr : adr + n])
            band = xpos[(xpos[:, 2] > 0.150) & (xpos[:, 2] < 0.210)]
            if len(band) == 0:
                info.append(None)
                continue
            delta = band[:, :2] - source
            # signed side along the jaw's own spread. Use y of the closest vert
            # and the spread of y.
            d = np.linalg.norm(delta, axis=1)
            k = int(np.argmin(d))
            info.append((float(d[k]), float(band[k, 1] - source[1]), float(band[k, 2]), int(np.sum(d < 0.02))))
        return info

    hits = []
    rolls = np.linspace(-2.4, 2.6, 14)
    pans = np.linspace(-1.2, 0.8, 9)
    lifts = np.linspace(-0.2, 1.3, 8)
    elbows = np.linspace(-1.3, 1.2, 8)
    flexes = np.linspace(-1.2, 1.4, 8)
    for roll in rolls:
        for pan in pans:
            for lift in lifts:
                for elbow in elbows:
                    for flex in flexes:
                        q = np.array([pan, lift, elbow, flex, roll, 0.7])
                        if any(q[i] < ranges[i][0] + 0.03 or q[i] > ranges[i][1] - 0.03 for i in range(5)):
                            continue
                        open_j = sides(q, 0.7)
                        if any(v is None for v in open_j):
                            continue
                        d0, y0, z0, _ = open_j[0]
                        d1, y1, z1, _ = open_j[1]
                        # opposite sides of the tube, both within a couple of cm
                        if y0 * y1 < 0 and d0 < 0.03 and d1 < 0.03 and abs(z0 - z1) < 0.02:
                            shut = sides(q, -0.05)
                            if any(v is None for v in shut):
                                continue
                            hits.append((d0 + d1, q.copy(), open_j, shut))
    hits.sort(key=lambda h: h[0])
    print("hits", len(hits))
    for score, q, opened, shut in hits[:8]:
        print(f"  score={score:.3f} q={np.round(q, 2)}")
        print(f"    open  fixed={opened[0]} moving={opened[1]}")
        print(f"    shut  fixed={shut[0]} moving={shut[1]}")


if __name__ == "__main__":
    main()
