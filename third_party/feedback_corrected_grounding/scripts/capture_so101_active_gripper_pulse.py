#!/usr/bin/env python3
"""Capture a slow SO-101 gripper-only pulse for live camera localization."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fcpg.so101_active_gripper import next_gripper_position
from fcpg.so101_shadow_session import (
    observation_rgb,
    observation_state,
    resize_rgb,
    save_shadow_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="myfollower01")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--wrist-index", default="/dev/video0")
    parser.add_argument("--column-index", default="/dev/video2")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--open-position", type=float, default=60.0)
    parser.add_argument("--closed-position", type=float, default=5.0)
    parser.add_argument("--maximum-step", type=float, default=1.5)
    parser.add_argument("--settle-tolerance", type=float, default=2.0)
    parser.add_argument("--hold-frames", type=int, default=12)
    parser.add_argument("--maximum-phase-frames", type=int, default=120)
    parser.add_argument("--maximum-arm-drift", type=float, default=1.5)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.fps <= 0 or args.image_size <= 0:
        raise ValueError("fps and image-size must be positive")
    if args.hold_frames < 2 or args.maximum_phase_frames <= args.hold_frames:
        raise ValueError("phase frame limits are invalid")
    if args.maximum_arm_drift <= 0 or args.settle_tolerance <= 0:
        raise ValueError("drift and settle tolerances must be positive")
    next_gripper_position(args.closed_position, args.open_position, args.maximum_step)


def _disconnect_without_torque_change(follower: object) -> None:
    for camera in follower.cameras.values():
        if camera.is_connected:
            camera.disconnect()
    if follower.bus.is_connected:
        follower.bus.disconnect(disable_torque=False)


def _save_capture(
    output: Path,
    *,
    images: dict[str, list[np.ndarray]],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    timestamps: list[float],
    args: argparse.Namespace,
    initial_torque: int,
    maximum_arm_drift: float,
) -> None:
    output.mkdir(parents=True)
    common = {
        "active_localization": True,
        "pulse_positions": [
            float(args.open_position),
            float(args.closed_position),
            float(args.open_position),
        ],
        "maximum_command_step": float(args.maximum_step),
        "maximum_observed_arm_drift": float(maximum_arm_drift),
        "initial_gripper_torque_enabled": bool(initial_torque),
        "other_joint_commands_sent": False,
    }
    camera_fps = {"wrist": 60, "column": 30}
    for camera, frames in images.items():
        save_shadow_session(
            output / camera,
            frames=np.stack(frames),
            states=np.stack(states),
            timestamps_s=np.asarray(timestamps, dtype=np.float64),
            actions=np.stack(actions),
            source="real-so101-active-gripper-pulse",
            camera=camera,
            fps=args.fps,
            action_source="bounded gripper-only position pulse",
            control_sent_by_recorder=True,
            extra_manifest={**common, "camera_device_fps": camera_fps[camera]},
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "format": "so101-paired-active-gripper-pulse-v1",
                "cameras": list(images),
                "frames": len(states),
                "sample_fps": float(args.fps),
                "control_sent_by_recorder": True,
                **common,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def capture(args: argparse.Namespace) -> None:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    cameras = {
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_index,
            fps=60,
            width=640,
            height=480,
        ),
        "column": OpenCVCameraConfig(
            index_or_path=args.column_index,
            fps=30,
            width=640,
            height=480,
        ),
    }
    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            calibration_dir=args.calibration_dir,
            use_degrees=False,
            disable_torque_on_disconnect=False,
            cameras=cameras,
        )
    )
    if not follower.calibration_fpath.is_file():
        raise FileNotFoundError(f"missing calibration: {follower.calibration_fpath}")

    images: dict[str, list[np.ndarray]] = {"wrist": [], "column": []}
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    timestamps: list[float] = []
    initial_torque: int | None = None
    initial_arm: np.ndarray | None = None
    maximum_arm_drift = 0.0
    failed = True
    started = time.perf_counter()
    try:
        follower.bus.connect()
        for camera in follower.cameras.values():
            camera.connect()
        initial_observation = follower.get_observation()
        initial_state = observation_state(initial_observation)
        initial_arm = initial_state[:5].copy()
        torque = follower.bus.sync_read(
            "Torque_Enable", motors="gripper", normalize=False, num_retry=1
        )
        initial_torque = int(torque["gripper"])
        follower.bus.sync_write(
            "Goal_Position", {"gripper": float(initial_state[5])}
        )
        if not initial_torque:
            follower.bus.enable_torque("gripper", num_retry=1)

        period = 1.0 / args.fps
        for phase_name, target in (
            ("open", args.open_position),
            ("close", args.closed_position),
            ("reopen", args.open_position),
        ):
            stable_frames = 0
            for _ in range(args.maximum_phase_frames):
                loop_started = time.perf_counter()
                observation = follower.get_observation()
                state = observation_state(observation)
                drift = float(np.max(np.abs(state[:5] - initial_arm)))
                maximum_arm_drift = max(maximum_arm_drift, drift)
                if drift > args.maximum_arm_drift:
                    raise RuntimeError(
                        f"arm drift {drift:.3f} exceeds {args.maximum_arm_drift:.3f}"
                    )
                for camera in images:
                    images[camera].append(
                        resize_rgb(observation_rgb(observation, camera), args.image_size)
                    )
                timestamps.append(time.perf_counter() - started)

                if abs(float(state[5]) - target) <= args.settle_tolerance:
                    stable_frames += 1
                else:
                    stable_frames = 0
                command = next_gripper_position(
                    float(state[5]), float(target), float(args.maximum_step)
                )
                sent = follower.send_action({"gripper.pos": command})
                if set(sent) != {"gripper.pos"} or not np.isfinite(sent["gripper.pos"]):
                    raise RuntimeError("LeRobot returned an invalid gripper command")
                full_action = state.copy()
                full_action[5] = float(sent["gripper.pos"])
                states.append(state)
                actions.append(full_action)
                if stable_frames >= args.hold_frames:
                    break
                delay = period - (time.perf_counter() - loop_started)
                if delay > 0:
                    time.sleep(delay)
            else:
                raise RuntimeError(
                    f"gripper did not settle during {phase_name} phase at {target:.1f}"
                )
        failed = False
    finally:
        if not failed and initial_torque == 0 and follower.bus.is_connected:
            follower.bus.disable_torque("gripper", num_retry=1)
        _disconnect_without_torque_change(follower)

    if initial_torque is None or initial_arm is None:
        raise RuntimeError("capture ended before the initial hardware state was read")
    _save_capture(
        args.output.expanduser().resolve(),
        images=images,
        states=states,
        actions=actions,
        timestamps=timestamps,
        args=args,
        initial_torque=initial_torque,
        maximum_arm_drift=maximum_arm_drift,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "frames": len(states),
                "maximum_arm_drift": maximum_arm_drift,
                "final_gripper_position": float(states[-1][5]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    _validate_args(args)
    capture(args)


if __name__ == "__main__":
    main()
