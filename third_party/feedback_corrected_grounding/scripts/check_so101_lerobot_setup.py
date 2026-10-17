#!/usr/bin/env python3
"""Inspect local SO-101 ports and camera devices without connecting to hardware."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--leader-id", default="so101_leader")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def calibration_file(kind: str, identifier: str, root: Path | None) -> Path:
    if root is None:
        from lerobot.utils.constants import (
            HF_LEROBOT_CALIBRATION,
            ROBOTS,
            TELEOPERATORS,
        )

        category, implementation = kind.split("/", maxsplit=1)
        folder = ROBOTS if category == "robots" else TELEOPERATORS
        root = HF_LEROBOT_CALIBRATION / folder / implementation
    return root.expanduser().resolve() / f"{identifier}.json"


def main() -> None:
    args = parse_args()
    from serial.tools import list_ports

    serial_ports = [
        {
            "device": port.device,
            "description": port.description,
            "manufacturer": port.manufacturer,
            "serial_number": port.serial_number,
        }
        for port in list_ports.comports()
        if port.device.startswith(("/dev/ttyACM", "/dev/ttyUSB"))
        or port.manufacturer is not None
    ]
    video_devices = [str(path) for path in sorted(Path("/dev").glob("video*"))]
    follower_calibration = calibration_file(
        "robots/so101_follower", args.robot_id, args.calibration_dir
    )
    leader_calibration = calibration_file(
        "teleoperators/so101_leader", args.leader_id, args.calibration_dir
    )
    report = {
        "lerobot_version": importlib.metadata.version("lerobot"),
        "serial_ports": serial_ports,
        "video_devices": video_devices,
        "follower_calibration": str(follower_calibration),
        "follower_calibrated": follower_calibration.is_file(),
        "leader_calibration": str(leader_calibration),
        "leader_calibrated": leader_calibration.is_file(),
        "hardware_ready": bool(serial_ports and video_devices and follower_calibration.is_file()),
        "control_sent": False,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
