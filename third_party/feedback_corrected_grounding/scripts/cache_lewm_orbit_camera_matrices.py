#!/usr/bin/env python3
"""Reconstruct and cache exact camera matrices for an orbit A4 replay dataset."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.geometry import Camera, project_points  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    make_a4_episode_runtime,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_orbit_camera import (  # noqa: E402
    a4_orbit_focus,
    orbit_camera_position,
    orbit_view_from_values,
    set_fixed_camera_look_at,
)
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
)


DEFAULT_RAW = ROOT / "outputs/lewm_action_prefix_a4/joint_orbit12_gripper_24train_8dev"
DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)
    if int(args.log_every) <= 0:
        parser.error("--log-every must be positive")
    return args


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def reconstruct_camera_matrices(
    index: Mapping[str, np.ndarray],
    *,
    width: int,
    height: int,
    benchmark_seed: int,
    log_every: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay only environment reset to recover exact orbit camera K/T."""

    required = (
        "task_name",
        "task_instance",
        "environment_seed",
        "camera_base_name",
        "camera_clock_hour",
        "camera_elevation_degrees",
        "camera_radius_m",
    )
    if any(name not in index for name in required):
        raise ValueError("camera reconstruction requires orbit replay metadata")
    count = int(np.asarray(index["task_name"]).shape[0])
    intrinsic = np.empty((count, 3, 3), dtype=np.float32)
    extrinsic = np.empty((count, 4, 4), dtype=np.float32)
    physical = np.asarray(index.get("physical_bundle_index"), dtype=np.int64)
    if physical.shape != (count,):
        raise ValueError("camera reconstruction requires physical_bundle_index")
    completed = 0
    for physical_id in np.unique(physical):
        rows = np.flatnonzero(physical == int(physical_id))
        first = int(rows[0])
        identity_names = (
            "task_name",
            "task_instance",
            "environment_seed",
            "camera_base_name",
        )
        for name in identity_names:
            if np.unique(np.asarray(index[name])[rows]).size != 1:
                raise ValueError(f"physical bundle disagrees on {name}")
        base_name = str(index["camera_base_name"][first])
        first_view = orbit_view_from_values(
            int(index["camera_clock_hour"][first]),
            float(index["camera_elevation_degrees"][first]),
            float(index["camera_radius_m"][first]),
            base_name,
        )
        runtime = make_a4_episode_runtime(
            str(index["task_name"][first]),
            int(index["task_instance"][first]),
            int(index["environment_seed"][first]),
            width=int(width),
            height=int(height),
            benchmark_seed=int(benchmark_seed),
            camera_name=base_name,
            orbit_camera_view=first_view,
        )
        try:
            camera_id = get_camera_id(runtime.env.model, base_name)
            focus = a4_orbit_focus(
                runtime.observation,
                getattr(runtime.env, "_target_pos", None),
            )
            row_intrinsic = camera_intrinsic_from_fovy(
                int(width), int(height), float(runtime.env.model.cam_fovy[camera_id])
            ).astype(np.float32)
            for row_value in rows:
                row = int(row_value)
                view = orbit_view_from_values(
                    int(index["camera_clock_hour"][row]),
                    float(index["camera_elevation_degrees"][row]),
                    float(index["camera_radius_m"][row]),
                    base_name,
                )
                set_fixed_camera_look_at(
                    runtime.env.model,
                    runtime.env.data,
                    camera_id,
                    orbit_camera_position(focus, view),
                    focus,
                )
                intrinsic[row] = row_intrinsic
                extrinsic[row] = camera_extrinsic_cv_from_mujoco(
                    runtime.env.data, camera_id
                ).astype(np.float32)
                completed += 1
        finally:
            runtime.env.close()
        if completed % int(log_every) < len(rows) or completed == count:
            print(json.dumps({"camera_rows": completed, "total": count}), flush=True)
    return intrinsic, extrinsic


def validate_reprojection(
    *,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    current_xyz: np.ndarray,
    current_q_true: np.ndarray,
    width: int,
    height: int,
) -> dict[str, float]:
    if intrinsic.shape[0] != current_xyz.shape[0] or extrinsic.shape[0] != current_xyz.shape[0]:
        raise ValueError("camera and replay row counts differ")
    errors = []
    for row in range(current_xyz.shape[0]):
        camera = Camera(
            intrinsic=np.asarray(intrinsic[row], dtype=np.float64),
            extrinsic_cam_from_world=np.asarray(extrinsic[row], dtype=np.float64),
            width=int(width),
            height=int(height),
        )
        projected, _valid = project_points(
            np.asarray(current_xyz[row], dtype=np.float64), camera
        )
        errors.append(np.linalg.norm(projected - current_q_true[row], axis=-1))
    value = np.concatenate(errors).astype(np.float64)
    result = {
        "mean_reprojection_error_px": float(value.mean()),
        "max_reprojection_error_px": float(value.max()),
    }
    if result["max_reprojection_error_px"] > 1.0e-3:
        raise RuntimeError(f"camera reconstruction mismatch: {result}")
    return result


def _atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, np.asarray(value))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    raw = args.raw_dataset.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw)
    config = _json(args.config)
    collection = config.get("collection")
    if not isinstance(collection, Mapping):
        raise TypeError("A4 config collection must be a mapping")
    width = int(arrays["current_rgb"].shape[-1])
    height = int(arrays["current_rgb"].shape[-2])
    intrinsic, extrinsic = reconstruct_camera_matrices(
        index,
        width=width,
        height=height,
        benchmark_seed=int(collection["benchmark_seed"]),
        log_every=int(args.log_every),
    )
    metrics = validate_reprojection(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        current_xyz=np.asarray(arrays["current_keypoints_xyz"]),
        current_q_true=np.asarray(arrays["current_q_true"]),
        width=width,
        height=height,
    )
    _atomic_save(raw / "camera_intrinsic.npy", intrinsic)
    _atomic_save(raw / "camera_extrinsic_cam_from_world.npy", extrinsic)
    print(json.dumps({"rows": int(intrinsic.shape[0]), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
