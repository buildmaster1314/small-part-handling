#!/usr/bin/env python3
"""Measure nominal projection drift under synthetic camera extrinsic errors."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.geometry import Camera, perturb_extrinsic, project_points
from fcpg.metaworld_utils import (
    DEFAULT_KEYPOINT_SITES,
    get_site_keypoints,
    make_env,
    parse_direction_list,
    parse_float_list,
)
from fcpg.mujoco_camera import camera_extrinsic_cv_from_mujoco, camera_intrinsic_from_fovy, get_camera_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach-v3")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--translations-cm", default="0,0.5,1,2,4")
    parser.add_argument("--yaws-deg", default="0,1,2,5,10")
    parser.add_argument("--directions", default="x+,x-,y+,y-")
    parser.add_argument("--output", default="outputs/stage0_projection_drift.csv")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    env = make_env(args.task, args.camera, args.width, args.height)
    env.reset()
    for _ in range(args.steps):
        env.step(env.action_space.sample())

    model = env.model
    data = env.data
    camera_id = get_camera_id(model, args.camera)
    intrinsic = camera_intrinsic_from_fovy(args.width, args.height, float(model.cam_fovy[camera_id]))
    true_extrinsic = camera_extrinsic_cv_from_mujoco(data, camera_id)
    true_camera = Camera(intrinsic=intrinsic, extrinsic_cam_from_world=true_extrinsic, width=args.width, height=args.height)

    keypoint_state = get_site_keypoints(model, data, DEFAULT_KEYPOINT_SITES)
    points = keypoint_state.xyz
    q_true, valid_true = project_points(points, true_camera)
    rows: list[dict[str, object]] = []

    translations_cm = parse_float_list(args.translations_cm)
    yaws_deg = parse_float_list(args.yaws_deg)
    directions = parse_direction_list(args.directions)

    for trans_cm in translations_cm:
        for direction_name, direction in directions.items():
            trans_m = (trans_cm / 100.0) * direction
            for yaw in yaws_deg:
                nominal_extrinsic = perturb_extrinsic(true_extrinsic, translation_xyz=trans_m, yaw_degrees=yaw)
                nominal_camera = Camera(
                    intrinsic=intrinsic,
                    extrinsic_cam_from_world=nominal_extrinsic,
                    width=args.width,
                    height=args.height,
                )
                q_nominal, valid_nominal = project_points(points, nominal_camera)
                errors = np.linalg.norm(q_nominal - q_true, axis=1)
                for idx, name in enumerate(keypoint_state.names):
                    rows.append(
                        {
                            "task": args.task,
                            "camera": args.camera,
                            "site": name,
                            "translation_cm": trans_cm,
                            "translation_direction": direction_name,
                            "yaw_deg": yaw,
                            "q_true_x": q_true[idx, 0],
                            "q_true_y": q_true[idx, 1],
                            "q_nominal_x": q_nominal[idx, 0],
                            "q_nominal_y": q_nominal[idx, 1],
                            "pixel_error": errors[idx],
                            "valid_true": bool(valid_true[idx]),
                            "valid_nominal": bool(valid_nominal[idx]),
                        }
                    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errors = np.asarray([float(row["pixel_error"]) for row in rows], dtype=np.float64)
    print(f"task: {args.task}")
    print(f"camera: {args.camera}")
    print(f"sites: {keypoint_state.names}")
    print(f"rows: {len(rows)}")
    print(f"pixel_error_mean: {errors.mean():.4f}")
    print(f"pixel_error_p50: {np.percentile(errors, 50):.4f}")
    print(f"pixel_error_p90: {np.percentile(errors, 90):.4f}")
    print(f"pixel_error_max: {errors.max():.4f}")
    print(f"csv: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
