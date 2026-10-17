"""Closest each jaw can get to the neck, and which side it is on."""

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
    best = [(1.0, None), (1.0, None)]
    both = []

    rolls = np.linspace(-2.2, 2.4, 10)
    pans = np.linspace(-1.0, 0.6, 7)
    lifts = np.linspace(0.0, 1.3, 7)
    elbows = np.linspace(-1.2, 0.8, 7)
    flexes = np.linspace(-0.6, 1.5, 6)
    for roll in rolls:
        for pan in pans:
            for lift in lifts:
                for elbow in elbows:
                    for flex in flexes:
                        q = np.array([pan, lift, elbow, flex, roll, 0.15])
                        if any(q[i] < ranges[i][0] + 0.03 or q[i] > ranges[i][1] - 0.03 for i in range(5)):
                            continue
                        for address, value in zip(arm, q, strict=True):
                            data.qpos[address] = float(value)
                        data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
                        mujoco.mj_kinematics(model, data)
                        ds = []
                        for fi in range(model.nflex):
                            adr = int(model.flex_vertadr[fi])
                            n = int(model.flex_vertnum[fi])
                            xpos = np.asarray(data.flexvert_xpos[adr : adr + n])
                            band = xpos[(xpos[:, 2] > 0.145) & (xpos[:, 2] < 0.212)]
                            if len(band) == 0:
                                ds.append(None)
                                continue
                            delta = band[:, :2] - source
                            d = np.linalg.norm(delta, axis=1)
                            k = int(np.argmin(d))
                            ds.append((float(d[k]), float(delta[k, 0]), float(delta[k, 1]), float(band[k, 2])))
                            if d[k] < best[fi][0]:
                                best[fi] = (float(d[k]), q.copy(), ds[-1])
                        if ds[0] and ds[1] and ds[0][0] < 0.025 and ds[1][0] < 0.025:
                            # opposite half-planes?
                            dot = ds[0][1] * ds[1][1] + ds[0][2] * ds[1][2]
                            both.append((ds[0][0] + ds[1][0], dot, q.copy(), ds))
    for label, row in (("fixed", best[0]), ("moving", best[1])):
        q = None if row[1] is None else np.round(row[1], 3)
        detail = row[2] if len(row) > 2 else None
        print(label, row[0], q, detail)
    both.sort(key=lambda b: b[0])
    print("both near", len(both))
    opposed = [b for b in both if b[1] < 0]
    print("opposed", len(opposed))
    for row in (opposed or both)[:8]:
        score, dot, q, ds = row
        print(f"  sum={score:.3f} dot={dot:.4f} q={np.round(q, 2)} fixed={ds[0]} moving={ds[1]}")


if __name__ == "__main__":
    main()
