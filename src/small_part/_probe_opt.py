"""Local search for a two-sided pinch on the neck."""

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
    rng = np.random.default_rng(0)

    def score(q):
        for address, value in zip(arm, q, strict=True):
            data.qpos[address] = float(value)
        data.qpos[tube : tube + 7] = (*source, TUBE_CENTER_Z, 1.0, 0.0, 0.0, 0.0)
        mujoco.mj_kinematics(model, data)
        pts = []
        for fi in range(model.nflex):
            adr = int(model.flex_vertadr[fi])
            n = int(model.flex_vertnum[fi])
            xpos = np.asarray(data.flexvert_xpos[adr : adr + n])
            band = (xpos[:, 2] > 0.140) & (xpos[:, 2] < 0.214)
            if not np.any(band):
                return 1.0, None
            delta = xpos[band, :2] - source
            d = np.linalg.norm(delta, axis=1)
            k = int(np.argmin(d))
            pts.append((float(d[k]), delta[k].copy(), float(xpos[band][k, 2])))
        d0, v0, z0 = pts[0]
        d1, v1, z1 = pts[1]
        dot = float(v0 @ v1)
        # both close, on opposite sides, at similar height
        value = max(d0, d1) + 0.15 * max(0.0, dot) + 0.05 * abs(z0 - z1)
        return value, (d0, d1, dot, z0, z1)

    seeds = [
        np.array([-0.30, 0.60, -0.40, 1.37, roll, 0.25])
        for roll in np.linspace(-2.2, 2.4, 16)
    ]
    seeds.append(np.array([-0.30, 0.47, -0.37, 1.47, 1.58, 0.25]))
    best_q = None
    best_s = 1.0
    best_info = None
    for seed in seeds:
        q = seed.copy()
        s, info = score(q)
        if info is None:
            continue
        for _ in range(40):
            step = rng.normal(scale=[0.04, 0.05, 0.05, 0.05, 0.08, 0.05])
            trial = q + step
            ok = True
            for i in range(6):
                if trial[i] < ranges[i][0] + 0.02 or trial[i] > ranges[i][1] - 0.02:
                    ok = False
            if not ok:
                continue
            st, inf = score(trial)
            if inf is not None and st < s:
                q, s, info = trial, st, inf
        if s < best_s:
            best_q, best_s, best_info = q.copy(), s, info
            print(f"  {best_s:.4f} q={np.round(best_q, 3)} {best_info}")
    print("BEST", best_s, np.round(best_q, 3), best_info)


if __name__ == "__main__":
    main()
