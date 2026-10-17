#!/usr/bin/env python3
"""Record SO-101 observations for shadow inference without sending motor actions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fcpg.so101_shadow_session import (
    observation_rgb,
    observation_state,
    ordered_action,
    resize_rgb,
    save_shadow_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--port")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--primary-camera-fps", type=int)
    parser.add_argument("--primary-camera-fourcc")
    parser.add_argument("--secondary-camera-name")
    parser.add_argument("--secondary-camera-index", type=int)
    parser.add_argument("--secondary-camera-fps", type=int)
    parser.add_argument("--secondary-camera-fourcc")
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--leader-port")
    parser.add_argument("--leader-id", default="so101_leader")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--dry-run-cache", type=Path)
    parser.add_argument("--dry-run-episode", type=int)
    return parser.parse_args()


def camera_specs(args: argparse.Namespace) -> list[tuple[str, int, int, str | None]]:
    if args.camera_index is None:
        raise ValueError("hardware capture requires --camera-index")
    primary_fps = args.primary_camera_fps or int(round(args.fps))
    specs = [
        (args.camera_name, args.camera_index, primary_fps, args.primary_camera_fourcc)
    ]
    secondary = (
        args.secondary_camera_name,
        args.secondary_camera_index,
        args.secondary_camera_fps,
    )
    if any(value is not None for value in secondary):
        if any(value is None for value in secondary):
            raise ValueError(
                "secondary camera requires --secondary-camera-name, "
                "--secondary-camera-index, and --secondary-camera-fps"
            )
        specs.append(
            (
                str(args.secondary_camera_name),
                int(args.secondary_camera_index),
                int(args.secondary_camera_fps),
                args.secondary_camera_fourcc,
            )
        )
    names = [name for name, _, _, _ in specs]
    indexes = [index for _, index, _, _ in specs]
    if len(names) != len(set(names)) or len(indexes) != len(set(indexes)):
        raise ValueError("camera names and indexes must be unique")
    if any(camera_fps <= 0 for _, _, camera_fps, _ in specs):
        raise ValueError("camera FPS values must be positive")
    for _, _, _, fourcc in specs:
        if fourcc is not None and len(fourcc) != 4:
            raise ValueError("camera FOURCC must contain exactly four characters")
    return specs


def save_captured_sessions(
    output: Path,
    *,
    images_by_camera: dict[str, list[np.ndarray]],
    camera_fps: dict[str, int],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    timestamps: list[float],
    sample_fps: float,
    robot_id: str,
    leader_id: str | None,
) -> None:
    names = list(images_by_camera)
    paired = len(names) > 1
    if paired:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"shadow-session output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    common = {
        "dry_run": False,
        "follower_id": robot_id,
        "leader_id": leader_id,
        "paired_capture": paired,
        "paired_camera_names": names,
    }
    for name in names:
        root = output / name if paired else output
        save_shadow_session(
            root,
            frames=np.stack(images_by_camera[name]),
            states=np.stack(states),
            timestamps_s=np.asarray(timestamps),
            actions=np.stack(actions) if actions else None,
            source="real-so101-read-only",
            camera=name,
            fps=sample_fps,
            action_source="leader read-only telemetry" if actions else "none",
            extra_manifest={
                **common,
                "camera_device_fps": int(camera_fps[name]),
            },
        )
    if paired:
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "so101-paired-shadow-session-v1",
                    "cameras": names,
                    "frames": len(states),
                    "sample_fps": float(sample_fps),
                    "control_sent_by_recorder": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def connect_read_only(follower: object, leader: object | None) -> None:
    """Open buses and cameras without configuring motors or enabling control."""

    follower.bus.connect()
    try:
        for camera in follower.cameras.values():
            camera.connect()
        if leader is not None:
            leader.bus.connect()
    except BaseException:
        disconnect_read_only(follower, leader)
        raise


def disconnect_read_only(follower: object, leader: object | None) -> None:
    if leader is not None and leader.bus.is_connected:
        leader.bus.disconnect(disable_torque=False)
    for camera in follower.cameras.values():
        if camera.is_connected:
            camera.disconnect()
    if follower.bus.is_connected:
        follower.bus.disconnect(disable_torque=False)


def capture_cache(args: argparse.Namespace) -> None:
    root = args.dry_run_cache.expanduser().resolve()
    manifest = __import__("json").loads((root / "manifest.json").read_text())
    frames = np.load(root / "frames.npy", mmap_mode="r")
    states = np.load(root / "states.npy", mmap_mode="r")
    actions = np.load(root / "actions.npy", mmap_mode="r")
    episodes = np.load(root / "episodes.npy", mmap_mode="r")
    episode = (
        int(args.dry_run_episode)
        if args.dry_run_episode is not None
        else int(np.unique(episodes)[0])
    )
    rows = np.flatnonzero(np.asarray(episodes) == episode)[: args.frames]
    if len(rows) < 2:
        raise ValueError(f"dry-run episode {episode} has fewer than two frames")
    selected_frames = np.stack(
        [resize_rgb(np.asarray(frames[row]), args.image_size) for row in rows]
    )
    fps = float(manifest["fps"])
    save_shadow_session(
        args.output,
        frames=selected_frames,
        states=np.asarray(states[rows]),
        timestamps_s=np.arange(len(rows), dtype=np.float64) / fps,
        actions=np.asarray(actions[rows]),
        source=f"dry-run:{manifest.get('name', root.name)}:episode-{episode}",
        camera=str(manifest.get("camera", args.camera_name)),
        fps=fps,
        action_source="recorded LeRobot actions",
        extra_manifest={"dry_run": True, "source_episode": episode},
    )


def capture_hardware(args: argparse.Namespace) -> None:
    if args.port is None:
        raise ValueError("hardware capture requires --port")
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    specs = camera_specs(args)
    configured_cameras = {
        name: OpenCVCameraConfig(
            index_or_path=index,
            fps=camera_fps,
            width=640,
            height=480,
            fourcc=fourcc,
        )
        for name, index, camera_fps, fourcc in specs
    }

    follower_config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir,
        use_degrees=False,
        disable_torque_on_disconnect=False,
        cameras=configured_cameras,
    )
    follower = SO101Follower(follower_config)
    if not follower.calibration_fpath.is_file():
        raise FileNotFoundError(
            f"missing follower calibration {follower.calibration_fpath}; run the official "
            "lerobot-calibrate command first"
        )

    leader = None
    if args.leader_port:
        from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

        leader = SO101Leader(
            SO101LeaderConfig(
                port=args.leader_port,
                id=args.leader_id,
                calibration_dir=args.calibration_dir,
                use_degrees=False,
            )
        )
        if not leader.calibration_fpath.is_file():
            raise FileNotFoundError(
                f"missing leader calibration {leader.calibration_fpath}; run the official "
                "lerobot-calibrate command first"
            )

    images_by_camera: dict[str, list[np.ndarray]] = {
        name: [] for name, _, _, _ in specs
    }
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    timestamps: list[float] = []
    period = 1.0 / args.fps
    started = 0.0
    try:
        connect_read_only(follower, leader)
        started = time.perf_counter()
        for index in range(args.frames):
            deadline = started + index * period
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            observation = follower.get_observation()
            for name in images_by_camera:
                images_by_camera[name].append(
                    resize_rgb(observation_rgb(observation, name), args.image_size)
                )
            states.append(observation_state(observation))
            timestamps.append(time.perf_counter() - started)
            if leader is not None:
                actions.append(ordered_action(leader.get_action()))
    finally:
        disconnect_read_only(follower, leader)

    save_captured_sessions(
        args.output,
        images_by_camera=images_by_camera,
        camera_fps={name: camera_fps for name, _, camera_fps, _ in specs},
        states=states,
        actions=actions,
        timestamps=timestamps,
        sample_fps=args.fps,
        robot_id=args.robot_id,
        leader_id=args.leader_id if leader is not None else None,
    )


def main() -> None:
    args = parse_args()
    if args.frames <= 1 or args.fps <= 0 or args.image_size <= 0:
        raise ValueError("frames, fps and image-size must be positive")
    if args.dry_run_cache:
        capture_cache(args)
    else:
        capture_hardware(args)


if __name__ == "__main__":
    main()
