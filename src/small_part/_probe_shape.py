"""Shape of each jaw in its own body frame, so the tooth profile is visible."""

from __future__ import annotations

import mujoco
import numpy as np

from small_part.tube_scene import load_tube_model


def main():
    model, _ = load_tube_model()
    for fi in range(model.nflex):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fi)
        adr = int(model.flex_vertadr[fi])
        n = int(model.flex_vertnum[fi])
        local = np.asarray(model.flex_vert[adr : adr + n]).reshape(-1, 3)
        print(f"\n{name} n={n}")
        print(f"  x[{local[:, 0].min():.4f},{local[:, 0].max():.4f}]")
        print(f"  y[{local[:, 1].min():.4f},{local[:, 1].max():.4f}]")
        print(f"  z[{local[:, 2].min():.4f},{local[:, 2].max():.4f}]")
        # the tooth is the part far from the hinge. Bin by the long axis.
        span = local.max(0) - local.min(0)
        axis = int(np.argmax(span))
        print("  long axis", axis, "span", np.round(span, 4))
        edges = np.linspace(local[:, axis].min(), local[:, axis].max(), 8)
        for a, b in zip(edges[:-1], edges[1:]):
            sel = local[(local[:, axis] >= a) & (local[:, axis] < b + 1e-9)]
            if len(sel) == 0:
                continue
            other = [i for i in range(3) if i != axis]
            print(
                f"  {axis}~{(a + b) / 2:+.3f} n={len(sel):4d} "
                + " ".join(
                    f"{'xyz'[k]}[{sel[:, k].min():+.3f},{sel[:, k].max():+.3f}]" for k in other
                )
            )


if __name__ == "__main__":
    main()
