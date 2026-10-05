"""Load the SO-101 workcell and run one pick-transport-place cycle."""

from __future__ import annotations

import argparse
import time

import mujoco

from small_part.control import part_position, pick_transport_place, place_error
from small_part.scene import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.0, help="Motion speed scale.")
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, data = load_model()
    started = time.perf_counter()
    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            pick_transport_place(model, data, speed=args.speed)
            viewer.sync()
    else:
        pick_transport_place(model, data, speed=args.speed)
    elapsed = time.perf_counter() - started
    position = part_position(model, data)
    print(
        "cycle finished: "
        f"time={elapsed:.2f}s "
        f"part_xyz=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) "
        f"place_error={place_error(model, data):.3f}m"
    )


if __name__ == "__main__":
    main()
