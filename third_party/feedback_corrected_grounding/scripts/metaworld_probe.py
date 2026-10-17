#!/usr/bin/env python3
"""Probe MetaWorld rendering, camera names, and keypoint projection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.geometry import Camera, project_points
from fcpg.metaworld_utils import DEFAULT_KEYPOINT_SITES, get_site_keypoints, make_env
from fcpg.mujoco_camera import (
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    list_mujoco_names,
)


def draw_cross(image: np.ndarray, xy: np.ndarray, color: tuple[int, int, int], radius: int = 4) -> None:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    h, w = image.shape[:2]
    for dx in range(-radius, radius + 1):
        xx = x + dx
        if 0 <= xx < w and 0 <= y < h:
            image[y, xx] = color
    for dy in range(-radius, radius + 1):
        yy = y + dy
        if 0 <= x < w and 0 <= yy < h:
            image[yy, x] = color


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach-v3")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/metaworld_probe")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco

    env = make_env(args.task, args.camera, args.width, args.height)
    obs, info = env.reset()
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()

    image = env.render()
    model = env.model
    data = env.data

    camera_names = list_mujoco_names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
    site_names = list_mujoco_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    body_names = list_mujoco_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)

    camera_id = get_camera_id(model, args.camera)
    intrinsic = camera_intrinsic_from_fovy(args.width, args.height, float(model.cam_fovy[camera_id]))
    extrinsic = camera_extrinsic_cv_from_mujoco(data, camera_id)
    camera = Camera(intrinsic=intrinsic, extrinsic_cam_from_world=extrinsic, width=args.width, height=args.height)

    keypoint_state = get_site_keypoints(model, data, DEFAULT_KEYPOINT_SITES)
    pixels, valid = project_points(keypoint_state.xyz, camera)

    overlay = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255)]
    for idx, pixel in enumerate(pixels):
        if valid[idx]:
            draw_cross(overlay, pixel, colors[idx % len(colors)])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.task}_{args.camera}.png"
    overlay_path = output_dir / f"{args.task}_{args.camera}_keypoints.png"
    imageio.imwrite(image_path, image)
    imageio.imwrite(overlay_path, overlay)

    print(f"task: {args.task}")
    print(f"camera: {args.camera}")
    print(f"obs_shape: {getattr(obs, 'shape', None)}")
    print(f"action_space: {env.action_space}")
    print(f"camera_names: {camera_names}")
    print(f"site_names: {site_names}")
    print(f"body_names_sample: {body_names[:20]}")
    print("keypoint_projections:")
    for name, xyz, pixel, is_valid in zip(keypoint_state.names, keypoint_state.xyz, pixels, valid):
        print(f"  {name}: xyz={xyz.round(4).tolist()} pixel={pixel.round(2).tolist()} valid={bool(is_valid)}")
    print(f"image: {image_path}")
    print(f"overlay: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
