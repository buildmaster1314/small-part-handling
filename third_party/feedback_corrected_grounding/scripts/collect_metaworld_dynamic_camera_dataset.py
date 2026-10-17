#!/usr/bin/env python3
"""Collect pre-action MetaWorld data with auditable within-episode camera motion.

The existing projection collector is part of a frozen protocol bundle. This
standalone collector reuses its stable helpers while adding persistent step and
continuous linear-drift camera schedules. The nominal camera extrinsic is the
fixed base calibration for every frame in an episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.geometry import Camera, project_points
from fcpg.metaworld_utils import (
    DEFAULT_KEYPOINT_SITES,
    choose_action,
    get_site_keypoints,
    make_env,
    make_scripted_policy,
)
from fcpg.mujoco_camera import (
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    sample_camera_pose_offset,
    set_fixed_camera_pose_offset,
)
from fcpg.mujoco_camera_trajectory import (
    CAMERA_TRAJECTORY_MODES,
    anchor_observation_schedule,
    trajectory_frame,
    trajectory_phase,
    trajectory_supervision_masks,
)
from fcpg.observations import apply_keypoint_square_occlusion
from scripts.collect_metaworld_projection_dataset import make_rng_streams


DYNAMIC_CAMERA_DATASET_TYPE = "fcpg_metaworld_dynamic_camera_v2"
DYNAMIC_CAMERA_SCHEMA_VERSION = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="reach-v3")
    parser.add_argument("--camera", default="corner")
    parser.add_argument(
        "--a4-task-factory",
        action="store_true",
        help=(
            "Create tasks through the seven-task A4 factory. Use "
            "--camera a4-default to select each task's native camera."
        ),
    )
    parser.add_argument(
        "--a4-orbit-clock-hour",
        type=int,
        default=None,
        help=(
            "Place each A4 task's base camera at this clock-face orbit view "
            "before applying the dynamic camera trajectory."
        ),
    )
    parser.add_argument("--a4-orbit-elevation-deg", type=float, default=35.0)
    parser.add_argument("--a4-orbit-radius-m", type=float, default=1.1)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--steps-per-episode", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument(
        "--action-source", choices=("random", "scripted"), default="scripted"
    )
    parser.add_argument(
        "--actual-camera-translation-cm-min", type=float, default=0.0
    )
    parser.add_argument(
        "--actual-camera-translation-cm-max", type=float, default=0.0
    )
    parser.add_argument(
        "--actual-camera-rotation-deg-min", type=float, default=0.0
    )
    parser.add_argument(
        "--actual-camera-rotation-deg-max", type=float, default=0.0
    )
    parser.add_argument(
        "--actual-camera-trajectory",
        "--actual-camera-randomization",
        dest="actual_camera_trajectory",
        choices=CAMERA_TRAJECTORY_MODES,
        default="episode",
    )
    parser.add_argument("--actual-camera-step-fraction", type=float, default=0.5)
    parser.add_argument("--actual-camera-step-fraction-min", type=float, default=None)
    parser.add_argument("--actual-camera-step-fraction-max", type=float, default=None)
    parser.add_argument(
        "--actual-camera-drift-start-fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--actual-camera-drift-end-fraction", type=float, default=1.0
    )
    parser.add_argument(
        "--actual-camera-clean-probability", type=float, default=0.0
    )
    parser.add_argument("--patch-radius", type=int, default=16)
    parser.add_argument("--paired-clean-view", action="store_true")
    parser.add_argument(
        "--anchor-dropout-mode",
        choices=("none", "measurement", "visual"),
        default="none",
    )
    parser.add_argument("--anchor-dropout-indices", default="0")
    parser.add_argument("--anchor-dropout-start-offset", type=int, default=1)
    parser.add_argument("--anchor-dropout-length", type=int, default=0)
    parser.add_argument("--anchor-dropout-maximum-length", type=int, default=6)
    parser.add_argument("--anchor-recovery-window-length", type=int, default=0)
    parser.add_argument("--anchor-occlusion-radius-px", type=int, default=12)
    parser.add_argument("--anchor-occlusion-value", type=int, default=0)
    parser.add_argument("--minimum-recovery-frames", type=int, default=4)
    parser.add_argument("--minimum-history-size", type=int, default=4)
    parser.add_argument("--vary-task-index", action="store_true")
    parser.add_argument("--task-index-offset", type=int, default=0)
    parser.add_argument("--env-seed-base", type=int, default=0)
    parser.add_argument("--task-generation-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--independent-rng-streams", action="store_true")
    parser.add_argument(
        "--output",
        default="data/metaworld_projection/reach_v3_dynamic_camera.npz",
    )
    return parser


def _validated_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    for name in (
        "width",
        "height",
        "episodes_per_task",
        "steps_per_episode",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.patch_radius < 0:
        parser.error("--patch-radius must be non-negative")
    observer_protocol = bool(
        args.paired_clean_view or args.anchor_dropout_mode != "none"
    )
    if args.minimum_history_size < 0:
        parser.error("--minimum-history-size must be non-negative")
    if observer_protocol and args.minimum_history_size < 4:
        parser.error("observer/dropout data require --minimum-history-size >= 4")
    if (
        observer_protocol
        and args.steps_per_episode < args.minimum_history_size + 1
    ):
        parser.error("observer/dropout data require history plus a target frame")
    if args.anchor_dropout_start_offset < 0:
        parser.error("--anchor-dropout-start-offset must be non-negative")
    if args.anchor_recovery_window_length < 0:
        parser.error("--anchor-recovery-window-length must be non-negative")
    if args.minimum_recovery_frames < 1:
        parser.error("--minimum-recovery-frames must be positive")
    if args.anchor_occlusion_radius_px < 0:
        parser.error("--anchor-occlusion-radius-px must be non-negative")
    if not 0 <= args.anchor_occlusion_value <= 255:
        parser.error("--anchor-occlusion-value must lie in [0, 255]")
    camera_limits = (
        args.actual_camera_translation_cm_min,
        args.actual_camera_translation_cm_max,
        args.actual_camera_rotation_deg_min,
        args.actual_camera_rotation_deg_max,
        args.actual_camera_clean_probability,
    )
    if not all(np.isfinite(float(value)) for value in camera_limits):
        parser.error("actual camera limits and probability must be finite")
    if not (
        0.0
        <= args.actual_camera_translation_cm_min
        <= args.actual_camera_translation_cm_max
    ):
        parser.error("actual camera translation min/max are inconsistent")
    if not (
        0.0
        <= args.actual_camera_rotation_deg_min
        <= args.actual_camera_rotation_deg_max
    ):
        parser.error("actual camera rotation min/max are inconsistent")
    if not 0.0 <= args.actual_camera_clean_probability <= 1.0:
        parser.error("--actual-camera-clean-probability must be in [0, 1]")
    for name in (
        "actual_camera_step_fraction",
        "actual_camera_drift_start_fraction",
        "actual_camera_drift_end_fraction",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and in [0, 1]")
    supplied_step_bounds = (
        args.actual_camera_step_fraction_min is not None,
        args.actual_camera_step_fraction_max is not None,
    )
    if supplied_step_bounds[0] != supplied_step_bounds[1]:
        parser.error("step-fraction min and max must be provided together")
    if not any(supplied_step_bounds):
        args.actual_camera_step_fraction_min = float(args.actual_camera_step_fraction)
        args.actual_camera_step_fraction_max = float(args.actual_camera_step_fraction)
    step_min = float(args.actual_camera_step_fraction_min)
    step_max = float(args.actual_camera_step_fraction_max)
    if not np.isfinite(step_min) or not np.isfinite(step_max):
        parser.error("step-fraction bounds must be finite")
    if not 0.0 <= step_min <= step_max <= 1.0:
        parser.error("step-fraction min/max are inconsistent")
    if step_min != step_max:
        if args.actual_camera_trajectory != "step":
            parser.error("random step timing requires --actual-camera-trajectory step")
        if not args.independent_rng_streams:
            parser.error("random step timing requires --independent-rng-streams")
    if (
        args.actual_camera_drift_start_fraction
        >= args.actual_camera_drift_end_fraction
    ):
        parser.error("actual camera drift start must be smaller than its end")
    if observer_protocol:
        earliest_step_fraction = (
            step_min
            if args.actual_camera_trajectory == "step"
            else args.actual_camera_step_fraction
        )
        earliest_phases = np.asarray(
            [
                trajectory_phase(
                    args.actual_camera_trajectory,
                    frame_index,
                    args.steps_per_episode,
                    step_fraction=earliest_step_fraction,
                    drift_start_fraction=args.actual_camera_drift_start_fraction,
                    drift_end_fraction=args.actual_camera_drift_end_fraction,
                )
                for frame_index in range(args.steps_per_episode)
            ],
            dtype=np.float64,
        )
        earliest_events = np.flatnonzero(
            trajectory_supervision_masks(
                args.actual_camera_trajectory, earliest_phases
            ).intervention_onset
        )
        if (
            earliest_events.size > 0
            and int(earliest_events[0]) < int(args.minimum_history_size)
        ):
            parser.error(
                "observer/dropout data require every within-episode camera "
                "intervention to follow --minimum-history-size clean frames"
            )
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        parser.error("--tasks must contain at least one task")
    if len(set(tasks)) != len(tasks):
        parser.error("--tasks must not contain duplicate task names")
    if args.a4_task_factory:
        from fcpg.lewm_action_branch_a4 import FORMAL_TASK_INSTANCES, TASKS

        unknown = sorted(set(tasks) - set(TASKS))
        if unknown:
            parser.error(f"A4 task factory received unknown tasks: {unknown}")
        largest_task_index = args.task_index_offset + (
            args.episodes_per_task - 1 if args.vary_task_index else 0
        )
        if (
            args.task_index_offset not in FORMAL_TASK_INSTANCES
            or largest_task_index not in FORMAL_TASK_INSTANCES
        ):
            parser.error(
                "A4 task instances must remain in the inclusive range "
                f"{FORMAL_TASK_INSTANCES[0]}..{FORMAL_TASK_INSTANCES[-1]}"
            )
    elif args.camera == "a4-default":
        parser.error("--camera a4-default requires --a4-task-factory")
    if args.a4_orbit_clock_hour is not None:
        if not args.a4_task_factory or args.camera != "a4-default":
            parser.error(
                "A4 orbit views require --a4-task-factory and --camera a4-default"
            )
        from fcpg.lewm_orbit_camera import OrbitCameraView

        try:
            OrbitCameraView(
                clock_hour=int(args.a4_orbit_clock_hour),
                elevation_degrees=float(args.a4_orbit_elevation_deg),
                radius_m=float(args.a4_orbit_radius_m),
            )
        except (TypeError, ValueError) as error:
            parser.error(f"invalid A4 orbit camera: {error}")
    if Path(args.output).suffix != ".npz":
        parser.error("--output must use the .npz suffix")
    if args.anchor_dropout_mode == "none":
        if args.anchor_dropout_length != 0:
            parser.error("dropout length requires a non-none anchor dropout mode")
        args.anchor_dropout_index_list = []
    else:
        if args.actual_camera_trajectory not in {"step", "linear_drift"}:
            parser.error("anchor dropout requires a step or linear-drift trajectory")
        if args.anchor_dropout_maximum_length < 4:
            parser.error("anchor dropout maximum length must be at least 4")
        if not 4 <= args.anchor_dropout_length <= args.anchor_dropout_maximum_length:
            parser.error(
                "anchor dropout length must lie in "
                f"[4, {args.anchor_dropout_maximum_length}]"
            )
        try:
            indices = [
                int(value.strip())
                for value in args.anchor_dropout_indices.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("anchor dropout indices must be comma-separated integers")
        if not indices or len(set(indices)) != len(indices):
            parser.error("anchor dropout indices must be non-empty and unique")
        if any(index < 0 or index >= len(DEFAULT_KEYPOINT_SITES) for index in indices):
            parser.error("anchor dropout index is outside the control-anchor layout")
        if args.anchor_dropout_mode == "visual" and args.anchor_occlusion_radius_px == 0:
            parser.error("visual anchor dropout requires a positive occlusion radius")
        args.anchor_dropout_index_list = indices
        worst_case_step_fraction = (
            step_max
            if args.actual_camera_trajectory == "step"
            else args.actual_camera_step_fraction
        )
        planned = np.asarray(
            [
                trajectory_phase(
                    args.actual_camera_trajectory,
                    frame_index,
                    args.steps_per_episode,
                    step_fraction=worst_case_step_fraction,
                    drift_start_fraction=args.actual_camera_drift_start_fraction,
                    drift_end_fraction=args.actual_camera_drift_end_fraction,
                )
                for frame_index in range(args.steps_per_episode)
            ],
            dtype=np.float64,
        )
        try:
            anchor_observation_schedule(
                args.steps_per_episode,
                len(DEFAULT_KEYPOINT_SITES),
                trajectory_supervision_masks(
                    args.actual_camera_trajectory, planned
                ).intervention_onset,
                anchor_indices=tuple(indices),
                start_offset=args.anchor_dropout_start_offset,
                length=args.anchor_dropout_length,
                maximum_length=args.anchor_dropout_maximum_length,
                minimum_recovery_frames=args.minimum_recovery_frames,
                recovery_length=(
                    args.anchor_recovery_window_length or None
                ),
            )
        except (TypeError, ValueError) as error:
            parser.error(f"invalid anchor dropout schedule: {error}")
    args.task_list = tasks
    return args


def _collection_config(
    args: argparse.Namespace,
    *,
    task_generation_seed: int,
) -> dict[str, object]:
    return {
        "dataset_type": DYNAMIC_CAMERA_DATASET_TYPE,
        "schema_version": DYNAMIC_CAMERA_SCHEMA_VERSION,
        "tasks": list(args.task_list),
        "camera": str(args.camera),
        "a4_task_factory": bool(args.a4_task_factory),
        "camera_by_task": {
            task: _camera_name(args, task) for task in args.task_list
        },
        "a4_orbit_clock_hour": (
            None
            if args.a4_orbit_clock_hour is None
            else int(args.a4_orbit_clock_hour)
        ),
        "a4_orbit_elevation_degrees": float(args.a4_orbit_elevation_deg),
        "a4_orbit_radius_m": float(args.a4_orbit_radius_m),
        "width": int(args.width),
        "height": int(args.height),
        "episodes_per_task": int(args.episodes_per_task),
        "steps_per_episode": int(args.steps_per_episode),
        "warmup_steps": int(args.warmup_steps),
        "action_source": str(args.action_source),
        "actual_camera_translation_cm_min": float(
            args.actual_camera_translation_cm_min
        ),
        "actual_camera_translation_cm_max": float(
            args.actual_camera_translation_cm_max
        ),
        "actual_camera_rotation_deg_min": float(
            args.actual_camera_rotation_deg_min
        ),
        "actual_camera_rotation_deg_max": float(
            args.actual_camera_rotation_deg_max
        ),
        "actual_camera_trajectory": str(args.actual_camera_trajectory),
        "actual_camera_step_fraction": float(args.actual_camera_step_fraction),
        "actual_camera_step_fraction_min": float(
            args.actual_camera_step_fraction_min
        ),
        "actual_camera_step_fraction_max": float(
            args.actual_camera_step_fraction_max
        ),
        "actual_camera_drift_start_fraction": float(
            args.actual_camera_drift_start_fraction
        ),
        "actual_camera_drift_end_fraction": float(
            args.actual_camera_drift_end_fraction
        ),
        "actual_camera_clean_probability": float(
            args.actual_camera_clean_probability
        ),
        "patch_radius": int(args.patch_radius),
        "paired_clean_view": bool(args.paired_clean_view),
        "anchor_dropout_mode": str(args.anchor_dropout_mode),
        "anchor_dropout_indices": list(args.anchor_dropout_index_list),
        "anchor_dropout_start_offset": int(args.anchor_dropout_start_offset),
        "anchor_dropout_length": int(args.anchor_dropout_length),
        "anchor_dropout_maximum_length": int(
            args.anchor_dropout_maximum_length
        ),
        "anchor_recovery_window_length": int(
            args.anchor_recovery_window_length
        ),
        "anchor_occlusion_radius_px": int(args.anchor_occlusion_radius_px),
        "anchor_occlusion_value": int(args.anchor_occlusion_value),
        "visual_occlusion_center_source": (
            "q_true_offline_intervention_only"
            if args.anchor_dropout_mode == "visual"
            else "not_applicable"
        ),
        "minimum_recovery_frames": int(args.minimum_recovery_frames),
        "minimum_history_size": int(args.minimum_history_size),
        "vary_task_index": bool(args.vary_task_index),
        "task_index_offset": int(args.task_index_offset),
        "env_seed_base": int(args.env_seed_base),
        "task_generation_seed": int(task_generation_seed),
        "seed": int(args.seed),
        "independent_rng_streams": bool(args.independent_rng_streams),
        "output": str(Path(args.output).expanduser().resolve()),
    }


def _camera_name(args: argparse.Namespace, task_name: str) -> str:
    if args.a4_task_factory and args.camera == "a4-default":
        from fcpg.lewm_action_branch_a4_collection import a4_camera_name

        return a4_camera_name(task_name)
    return str(args.camera)


def _orbit_camera_view(
    args: argparse.Namespace, camera_name: str
):
    if args.a4_orbit_clock_hour is None:
        return None
    from fcpg.lewm_orbit_camera import OrbitCameraView

    return OrbitCameraView(
        clock_hour=int(args.a4_orbit_clock_hour),
        elevation_degrees=float(args.a4_orbit_elevation_deg),
        radius_m=float(args.a4_orbit_radius_m),
        base_camera_name=str(camera_name),
    )


def _make_policy(args: argparse.Namespace, task_name: str):
    if args.action_source != "scripted":
        return None
    if args.a4_task_factory:
        from fcpg.lewm_a4_task_factory import make_a4_expert

        return make_a4_expert(task_name)
    return make_scripted_policy(task_name)


def _make_environment(
    args: argparse.Namespace,
    task_name: str,
    camera_name: str,
    *,
    task_index: int,
    task_generation_seed: int,
):
    if args.a4_task_factory:
        from fcpg.lewm_a4_task_factory import make_a4_env

        return make_a4_env(
            task_name,
            camera_name,
            args.width,
            args.height,
            task_instance=task_index,
            benchmark_seed=task_generation_seed,
        )
    return make_env(
        task_name,
        camera_name,
        args.width,
        args.height,
        task_index=task_index,
        benchmark_seed=task_generation_seed,
    )


_FRAME_ARRAY_NAMES = (
    "rgb",
    "obs",
    "action",
    "reward",
    "success",
    "task_name",
    "task_index",
    "episode_id",
    "env_seed",
    "frame_id",
    "episode_recorded_frame_count",
    "keypoints_xyz",
    "q_true",
    "q_nominal",
    "valid_true",
    "valid_nominal",
    "camera_intrinsic",
    "true_extrinsic",
    "base_extrinsic",
    "nominal_extrinsic",
    "actual_camera_translation_xyz",
    "actual_camera_rotation_axis",
    "actual_camera_rotation_deg",
    "actual_camera_endpoint_translation_xyz",
    "actual_camera_endpoint_rotation_axis",
    "actual_camera_endpoint_rotation_deg",
    "actual_camera_trajectory_phase",
    "camera_intervention_onset",
    "prior_supervision_valid",
    "camera_step_frame",
    "actual_camera_realized_step_fraction",
    "anchor_dropout_active",
    "anchor_recovery_onset",
    "anchor_recovery_segment",
    "measurement_valid",
    "pixel_error",
    "q_true_inside_patch",
)


def _validate_collected_arrays(
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    missing = sorted(set(_FRAME_ARRAY_NAMES) - set(arrays))
    if missing:
        raise ValueError(f"dynamic camera output is missing frame arrays: {missing}")
    count = int(arrays["rgb"].shape[0]) if arrays["rgb"].ndim > 0 else 0
    if count <= 0:
        raise ValueError("dynamic camera collection produced no recorded frames")
    mismatched = {
        name: tuple(arrays[name].shape)
        for name in _FRAME_ARRAY_NAMES
        if arrays[name].ndim == 0 or int(arrays[name].shape[0]) != count
    }
    if mismatched:
        raise ValueError(
            f"all dynamic camera frame arrays must contain {count} rows: {mismatched}"
        )

    keypoints = int(np.asarray(arrays["keypoint_site"]).size)
    expected_shapes = {
        "rgb": (count, int(args.height), int(args.width), 3),
        "keypoints_xyz": (count, keypoints, 3),
        "q_true": (count, keypoints, 2),
        "q_nominal": (count, keypoints, 2),
        "valid_true": (count, keypoints),
        "valid_nominal": (count, keypoints),
        "camera_intrinsic": (count, 3, 3),
        "true_extrinsic": (count, 4, 4),
        "base_extrinsic": (count, 4, 4),
        "nominal_extrinsic": (count, 4, 4),
        "actual_camera_translation_xyz": (count, 3),
        "actual_camera_rotation_axis": (count, 3),
        "actual_camera_endpoint_translation_xyz": (count, 3),
        "actual_camera_endpoint_rotation_axis": (count, 3),
        "anchor_dropout_active": (count, keypoints),
        "anchor_recovery_onset": (count, keypoints),
        "anchor_recovery_segment": (count, keypoints),
        "measurement_valid": (count, keypoints),
    }
    if "paired_clean_rgb" in arrays:
        expected_shapes["paired_clean_rgb"] = (
            count,
            int(args.height),
            int(args.width),
            3,
        )
    if "paired_clean_rgb_unmodified" in arrays:
        expected_shapes["paired_clean_rgb_unmodified"] = (
            count,
            int(args.height),
            int(args.width),
            3,
        )
    if "rgb_camera_only" in arrays:
        expected_shapes["rgb_camera_only"] = (
            count,
            int(args.height),
            int(args.width),
            3,
        )
    bad_shapes = {
        name: tuple(arrays[name].shape)
        for name, expected in expected_shapes.items()
        if tuple(arrays[name].shape) != expected
    }
    if arrays["obs"].ndim != 2 or arrays["action"].ndim != 2:
        bad_shapes["obs/action"] = (
            tuple(arrays["obs"].shape),
            tuple(arrays["action"].shape),
        )
    if keypoints <= 0 or bad_shapes:
        raise ValueError(f"dynamic camera arrays have invalid shapes: {bad_shapes}")
    if arrays["rgb"].dtype != np.uint8:
        raise ValueError("dynamic camera RGB must use uint8 storage")
    for name in (
        "paired_clean_rgb",
        "paired_clean_rgb_unmodified",
        "rgb_camera_only",
    ):
        if name in arrays and arrays[name].dtype != np.uint8:
            raise ValueError(f"{name} must use uint8 storage")
    if args.paired_clean_view != ("paired_clean_rgb" in arrays):
        raise ValueError("paired_clean_rgb presence disagrees with collector config")
    expects_raw_clean = bool(
        args.paired_clean_view and args.anchor_dropout_mode == "visual"
    )
    if expects_raw_clean != ("paired_clean_rgb_unmodified" in arrays):
        raise ValueError(
            "paired_clean_rgb_unmodified presence disagrees with visual pairing config"
        )
    if (args.anchor_dropout_mode == "visual") != ("rgb_camera_only" in arrays):
        raise ValueError("rgb_camera_only presence disagrees with visual dropout config")
    if args.anchor_dropout_mode == "visual":
        active_frames = np.any(arrays["anchor_dropout_active"], axis=1)
        changed_frames = np.any(
            arrays["rgb"] != arrays["rgb_camera_only"], axis=(1, 2, 3)
        )
        if not bool(np.all(changed_frames[active_frames])):
            raise ValueError(
                "every active visual-dropout frame must differ from rgb_camera_only"
            )
    if not np.issubdtype(arrays["camera_step_frame"].dtype, np.integer):
        raise TypeError("camera_step_frame must use integer storage")
    for name in (
        "valid_true",
        "valid_nominal",
        "q_true_inside_patch",
        "anchor_dropout_active",
        "anchor_recovery_onset",
        "anchor_recovery_segment",
        "measurement_valid",
    ):
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"{name} must use boolean storage")
    for name in ("camera_intervention_onset", "prior_supervision_valid"):
        if arrays[name].shape != (count,) or arrays[name].dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean vector with one value per frame")

    finite_names = (
        "obs",
        "action",
        "reward",
        "success",
        "keypoints_xyz",
        "q_true",
        "q_nominal",
        "camera_intrinsic",
        "true_extrinsic",
        "base_extrinsic",
        "nominal_extrinsic",
        "actual_camera_translation_xyz",
        "actual_camera_rotation_axis",
        "actual_camera_rotation_deg",
        "actual_camera_endpoint_translation_xyz",
        "actual_camera_endpoint_rotation_axis",
        "actual_camera_endpoint_rotation_deg",
        "actual_camera_trajectory_phase",
        "actual_camera_realized_step_fraction",
        "pixel_error",
    )
    nonfinite = [name for name in finite_names if not np.isfinite(arrays[name]).all()]
    if nonfinite:
        raise ValueError(f"dynamic camera arrays contain non-finite values: {nonfinite}")
    realized_step = arrays["actual_camera_realized_step_fraction"]
    if args.actual_camera_trajectory == "step":
        if bool(np.any(realized_step < 0.0) or np.any(realized_step > 1.0)):
            raise ValueError("realized step fractions must lie in [0, 1]")
    elif not np.array_equal(realized_step, np.full_like(realized_step, -1.0)):
        raise ValueError("non-step trajectories must use -1 realized-step sentinels")
    if not np.array_equal(arrays["nominal_extrinsic"], arrays["base_extrinsic"]):
        raise ValueError("nominal_extrinsic must remain equal to base_extrinsic")
    if bool(np.any(arrays["measurement_valid"] & arrays["anchor_dropout_active"])):
        raise ValueError("withheld anchors must not be valid measurements")
    expected_measurement_valid = (
        arrays["valid_nominal"] & ~arrays["anchor_dropout_active"]
    )
    if not np.array_equal(arrays["measurement_valid"], expected_measurement_valid):
        raise ValueError("measurement_valid must encode nominal validity and dropout")

    groups: dict[tuple[str, int], list[int]] = {}
    identities: set[tuple[str, int, int]] = set()
    for index in range(count):
        key = (str(arrays["task_name"][index]), int(arrays["episode_id"][index]))
        identity = (*key, int(arrays["frame_id"][index]))
        if identity in identities:
            raise ValueError(f"duplicate task/episode/frame identity: {identity}")
        identities.add(identity)
        groups.setdefault(key, []).append(index)
    expected_groups = {
        (task, episode_id)
        for task in args.task_list
        for episode_id in range(args.episodes_per_task)
    }
    if set(groups) != expected_groups:
        raise ValueError(
            "one or more requested task episodes produced no recorded frames"
        )
    for key, indices in groups.items():
        ordered = sorted(indices, key=lambda index: int(arrays["frame_id"][index]))
        frames = arrays["frame_id"][ordered]
        if not np.array_equal(frames, np.arange(len(ordered), dtype=frames.dtype)):
            raise ValueError(f"episode {key} has non-contiguous frame IDs")
        expected_count = np.full(len(ordered), len(ordered), dtype=np.int32)
        if not np.array_equal(
            arrays["episode_recorded_frame_count"][ordered], expected_count
        ):
            raise ValueError(f"episode {key} has incorrect recorded frame counts")
        expected_masks = trajectory_supervision_masks(
            args.actual_camera_trajectory,
            arrays["actual_camera_trajectory_phase"][ordered],
        )
        if not np.array_equal(
            arrays["camera_intervention_onset"][ordered],
            expected_masks.intervention_onset,
        ) or not np.array_equal(
            arrays["prior_supervision_valid"][ordered],
            expected_masks.prior_supervision_valid,
        ):
            raise ValueError(f"episode {key} has inconsistent supervision masks")
        step_frames = np.unique(arrays["camera_step_frame"][ordered])
        if step_frames.size != 1:
            raise ValueError(f"episode {key} has inconsistent camera_step_frame")
        expected_step_frame = (
            int(np.flatnonzero(expected_masks.intervention_onset)[0])
            if args.actual_camera_trajectory == "step"
            else -1
        )
        if int(step_frames[0]) != expected_step_frame:
            raise ValueError(f"episode {key} has incorrect camera_step_frame")
        realized_fractions = np.unique(
            arrays["actual_camera_realized_step_fraction"][ordered]
        )
        if realized_fractions.size != 1:
            raise ValueError(f"episode {key} has inconsistent realized step fraction")
        observer_protocol = bool(
            args.paired_clean_view or args.anchor_dropout_mode != "none"
        )
        if (
            observer_protocol
            and len(ordered) < int(args.minimum_history_size) + 1
        ):
            raise ValueError(f"episode {key} cannot support the requested history")
        intervention_frames = np.flatnonzero(expected_masks.intervention_onset)
        if (
            observer_protocol
            and intervention_frames.size > 0
            and int(intervention_frames[0]) < int(args.minimum_history_size)
        ):
            raise ValueError(
                f"episode {key} camera intervention precedes the requested history"
            )
        expected_dropout = anchor_observation_schedule(
            len(ordered),
            keypoints,
            expected_masks.intervention_onset,
            anchor_indices=tuple(args.anchor_dropout_index_list),
            start_offset=args.anchor_dropout_start_offset,
            length=args.anchor_dropout_length,
            maximum_length=args.anchor_dropout_maximum_length,
            minimum_recovery_frames=args.minimum_recovery_frames,
            recovery_length=(args.anchor_recovery_window_length or None),
        )
        for name, expected in (
            ("anchor_dropout_active", expected_dropout.dropout_active),
            ("anchor_recovery_onset", expected_dropout.recovery_onset),
            ("anchor_recovery_segment", expected_dropout.recovery_segment),
        ):
            if not np.array_equal(arrays[name][ordered], expected):
                raise ValueError(f"episode {key} has inconsistent {name}")


def collect(args: argparse.Namespace) -> Path:
    os.environ.setdefault("MUJOCO_GL", "egl")
    task_generation_seed = (
        args.seed if args.task_generation_seed is None else args.task_generation_seed
    )
    action_rng, camera_rng, schedule_rng = make_rng_streams(
        args.seed, args.independent_rng_streams
    )

    records: dict[str, list[object]] = {
        name: []
        for name in (
            "rgb",
            "obs",
            "action",
            "reward",
            "success",
            "task_name",
            "task_index",
            "episode_id",
            "env_seed",
            "frame_id",
            "episode_recorded_frame_count",
            "keypoints_xyz",
            "q_true",
            "q_nominal",
            "valid_true",
            "valid_nominal",
            "camera_intrinsic",
            "true_extrinsic",
            "base_extrinsic",
            "nominal_extrinsic",
            "actual_translation",
            "actual_axis",
            "actual_degrees",
            "endpoint_translation",
            "endpoint_axis",
            "endpoint_degrees",
            "trajectory_phase",
            "intervention_onset",
            "prior_supervision_valid",
            "camera_step_frame",
            "realized_step_fraction",
            "anchor_dropout_active",
            "anchor_recovery_onset",
            "anchor_recovery_segment",
            "measurement_valid",
            "pixel_error",
            "inside_patch",
        )
    }
    paired_clean_rgb: list[np.ndarray] = []
    paired_clean_rgb_unmodified: list[np.ndarray] = []
    rgb_camera_only: list[np.ndarray] = []
    keypoint_names: list[str] | None = None

    for task_offset, task_name in enumerate(args.task_list):
        policy = _make_policy(args, task_name)
        camera_name = _camera_name(args, task_name)
        for episode_id in range(args.episodes_per_task):
            task_index = args.task_index_offset + (
                episode_id if args.vary_task_index else 0
            )
            env_seed = args.env_seed_base + task_offset * 1_000_000 + episode_id
            env = _make_environment(
                args,
                task_name,
                camera_name,
                task_index=task_index,
                task_generation_seed=task_generation_seed,
            )
            try:
                camera_id = get_camera_id(env.model, camera_name)
                intrinsic = camera_intrinsic_from_fovy(
                    args.width,
                    args.height,
                    float(env.model.cam_fovy[camera_id]),
                )
                obs, _ = env.reset(seed=env_seed)
                orbit_view = _orbit_camera_view(args, camera_name)
                if orbit_view is not None:
                    from fcpg.lewm_orbit_camera import (
                        a4_orbit_focus,
                        orbit_camera_position,
                        set_fixed_camera_look_at,
                    )

                    orbit_focus = a4_orbit_focus(
                        np.asarray(obs, dtype=np.float64),
                        getattr(env, "_target_pos", None),
                    )
                    set_fixed_camera_look_at(
                        env.model,
                        env.data,
                        camera_id,
                        orbit_camera_position(orbit_focus, orbit_view),
                        orbit_focus,
                    )
                base_position = np.asarray(
                    env.model.cam_pos[camera_id], dtype=np.float64
                ).copy()
                base_quaternion = np.asarray(
                    env.model.cam_quat[camera_id], dtype=np.float64
                ).copy()
                base_extrinsic = camera_extrinsic_cv_from_mujoco(
                    env.data, camera_id
                )
                nominal_camera = Camera(
                    intrinsic=intrinsic,
                    extrinsic_cam_from_world=base_extrinsic,
                    width=args.width,
                    height=args.height,
                )
                episode_endpoint = sample_camera_pose_offset(
                    camera_rng,
                    args.actual_camera_translation_cm_max,
                    args.actual_camera_rotation_deg_max,
                    args.actual_camera_clean_probability,
                    translation_cm_min=args.actual_camera_translation_cm_min,
                    rotation_deg_min=args.actual_camera_rotation_deg_min,
                )
                if (
                    args.actual_camera_trajectory == "step"
                    and args.actual_camera_step_fraction_min
                    != args.actual_camera_step_fraction_max
                ):
                    episode_step_fraction = float(
                        schedule_rng.uniform(
                            args.actual_camera_step_fraction_min,
                            args.actual_camera_step_fraction_max,
                        )
                    )
                else:
                    episode_step_fraction = float(
                        args.actual_camera_step_fraction_min
                        if args.actual_camera_trajectory == "step"
                        else args.actual_camera_step_fraction
                    )
                planned_phases = np.asarray(
                    [
                        trajectory_phase(
                            args.actual_camera_trajectory,
                            frame_index,
                            args.steps_per_episode,
                            step_fraction=episode_step_fraction,
                            drift_start_fraction=(
                                args.actual_camera_drift_start_fraction
                            ),
                            drift_end_fraction=args.actual_camera_drift_end_fraction,
                        )
                        for frame_index in range(args.steps_per_episode)
                    ],
                    dtype=np.float64,
                )
                supervision_masks = trajectory_supervision_masks(
                    args.actual_camera_trajectory, planned_phases
                )
                dropout_schedule = anchor_observation_schedule(
                    args.steps_per_episode,
                    len(DEFAULT_KEYPOINT_SITES),
                    supervision_masks.intervention_onset,
                    anchor_indices=tuple(args.anchor_dropout_index_list),
                    start_offset=args.anchor_dropout_start_offset,
                    length=args.anchor_dropout_length,
                    maximum_length=args.anchor_dropout_maximum_length,
                    minimum_recovery_frames=args.minimum_recovery_frames,
                    recovery_length=(args.anchor_recovery_window_length or None),
                )
                step_events = np.flatnonzero(supervision_masks.intervention_onset)
                episode_step_frame = (
                    int(step_events[0])
                    if args.actual_camera_trajectory == "step" and step_events.size == 1
                    else -1
                )
                if args.actual_camera_trajectory == "episode":
                    set_fixed_camera_pose_offset(
                        env.model,
                        env.data,
                        camera_id,
                        base_position,
                        base_quaternion,
                        *episode_endpoint,
                    )

                recorded_frame_id = 0
                episode_record_start = len(records["frame_id"])
                total_raw_frames = args.warmup_steps + args.steps_per_episode
                for raw_frame_id in range(total_raw_frames):
                    action = choose_action(
                        env.action_space, action_rng, obs, policy=policy
                    )
                    should_record = raw_frame_id >= args.warmup_steps
                    if should_record:
                        obs_before_step = np.asarray(obs, dtype=np.float32).copy()
                        if args.actual_camera_trajectory == "frame":
                            endpoint = sample_camera_pose_offset(
                                camera_rng,
                                args.actual_camera_translation_cm_max,
                                args.actual_camera_rotation_deg_max,
                                args.actual_camera_clean_probability,
                                translation_cm_min=(
                                    args.actual_camera_translation_cm_min
                                ),
                                rotation_deg_min=args.actual_camera_rotation_deg_min,
                            )
                        else:
                            endpoint = episode_endpoint
                        camera_frame = trajectory_frame(
                            args.actual_camera_trajectory,
                            recorded_frame_id,
                            args.steps_per_episode,
                            endpoint,
                            step_fraction=episode_step_fraction,
                            drift_start_fraction=(
                                args.actual_camera_drift_start_fraction
                            ),
                            drift_end_fraction=(
                                args.actual_camera_drift_end_fraction
                            ),
                        )
                        if args.actual_camera_trajectory != "episode":
                            set_fixed_camera_pose_offset(
                                env.model,
                                env.data,
                                camera_id,
                                base_position,
                                base_quaternion,
                                *camera_frame.as_pose_offset(),
                            )
                        image_camera_only = env.render()
                        true_extrinsic = camera_extrinsic_cv_from_mujoco(
                            env.data, camera_id
                        )
                        true_camera = Camera(
                            intrinsic=intrinsic,
                            extrinsic_cam_from_world=true_extrinsic,
                            width=args.width,
                            height=args.height,
                        )
                        keypoint_state = get_site_keypoints(
                            env.model, env.data, DEFAULT_KEYPOINT_SITES
                        )
                        if keypoint_names is None:
                            keypoint_names = list(keypoint_state.names)
                        elif keypoint_names != list(keypoint_state.names):
                            raise RuntimeError("control-anchor site order changed")
                        q_true, valid_true = project_points(
                            keypoint_state.xyz, true_camera
                        )
                        q_nominal, valid_nominal = project_points(
                            keypoint_state.xyz, nominal_camera
                        )
                        error = np.linalg.norm(q_nominal - q_true, axis=1)
                        inside = np.all(
                            np.abs(q_true - np.round(q_nominal))
                            <= args.patch_radius,
                            axis=1,
                        )
                        active_dropout = dropout_schedule.dropout_active[
                            recorded_frame_id
                        ]
                        image = np.asarray(image_camera_only, dtype=np.uint8)
                        if args.anchor_dropout_mode == "visual" and bool(
                            active_dropout.any()
                        ):
                            image = apply_keypoint_square_occlusion(
                                image,
                                q_true[active_dropout],
                                radius_px=args.anchor_occlusion_radius_px,
                                value=args.anchor_occlusion_value,
                            )
                        if args.anchor_dropout_mode == "visual":
                            rgb_camera_only.append(
                                np.asarray(image_camera_only, dtype=np.uint8).copy()
                            )
                        if args.paired_clean_view:
                            set_fixed_camera_pose_offset(
                                env.model,
                                env.data,
                                camera_id,
                                base_position,
                                base_quaternion,
                                np.zeros(3, dtype=np.float64),
                                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                                0.0,
                            )
                            clean_image_unmodified = np.asarray(
                                env.render(), dtype=np.uint8
                            ).copy()
                            clean_image = clean_image_unmodified
                            if args.anchor_dropout_mode == "visual" and bool(
                                active_dropout.any()
                            ):
                                clean_image = apply_keypoint_square_occlusion(
                                    clean_image_unmodified,
                                    q_nominal[active_dropout],
                                    radius_px=args.anchor_occlusion_radius_px,
                                    value=args.anchor_occlusion_value,
                                )
                            paired_clean_rgb.append(clean_image)
                            if args.anchor_dropout_mode == "visual":
                                paired_clean_rgb_unmodified.append(
                                    clean_image_unmodified
                                )
                            set_fixed_camera_pose_offset(
                                env.model,
                                env.data,
                                camera_id,
                                base_position,
                                base_quaternion,
                                *camera_frame.as_pose_offset(),
                            )
                            restored_extrinsic = camera_extrinsic_cv_from_mujoco(
                                env.data, camera_id
                            )
                            if not np.allclose(
                                restored_extrinsic,
                                true_extrinsic,
                                rtol=0.0,
                                atol=1e-10,
                            ):
                                raise RuntimeError(
                                    "paired clean re-render did not restore the actual camera"
                                )

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    if should_record:
                        success = (
                            float(info.get("success", 0.0))
                            if isinstance(info, dict)
                            else 0.0
                        )
                        values = {
                            "rgb": image.astype(np.uint8),
                            "obs": obs_before_step,
                            "action": np.asarray(action, dtype=np.float32),
                            "reward": float(reward),
                            "success": success,
                            "task_name": task_name,
                            "task_index": task_index,
                            "episode_id": episode_id,
                            "env_seed": env_seed,
                            "frame_id": recorded_frame_id,
                            "keypoints_xyz": keypoint_state.xyz.astype(np.float32),
                            "q_true": q_true.astype(np.float32),
                            "q_nominal": q_nominal.astype(np.float32),
                            "valid_true": valid_true.astype(bool),
                            "valid_nominal": valid_nominal.astype(bool),
                            "camera_intrinsic": intrinsic.astype(np.float32),
                            "true_extrinsic": true_extrinsic.astype(np.float32),
                            "base_extrinsic": base_extrinsic.astype(np.float32),
                            "nominal_extrinsic": base_extrinsic.astype(np.float32),
                            "actual_translation": camera_frame.translation_world.astype(
                                np.float32
                            ),
                            "actual_axis": camera_frame.rotation_axis_world.astype(
                                np.float32
                            ),
                            "actual_degrees": camera_frame.rotation_degrees,
                            "endpoint_translation": np.asarray(
                                endpoint[0], dtype=np.float32
                            ),
                            "endpoint_axis": np.asarray(
                                endpoint[1], dtype=np.float32
                            ),
                            "endpoint_degrees": float(endpoint[2]),
                            "trajectory_phase": camera_frame.phase,
                            "intervention_onset": bool(
                                supervision_masks.intervention_onset[recorded_frame_id]
                            ),
                            "prior_supervision_valid": bool(
                                supervision_masks.prior_supervision_valid[recorded_frame_id]
                            ),
                            "camera_step_frame": episode_step_frame,
                            "realized_step_fraction": (
                                episode_step_fraction
                                if args.actual_camera_trajectory == "step"
                                else -1.0
                            ),
                            "anchor_dropout_active": active_dropout.copy(),
                            "anchor_recovery_onset": dropout_schedule.recovery_onset[
                                recorded_frame_id
                            ].copy(),
                            "anchor_recovery_segment": dropout_schedule.recovery_segment[
                                recorded_frame_id
                            ].copy(),
                            "measurement_valid": (
                                valid_nominal & ~active_dropout
                            ).astype(bool),
                            "pixel_error": error.astype(np.float32),
                            "inside_patch": inside.astype(bool),
                        }
                        for name, value in values.items():
                            records[name].append(value)
                        recorded_frame_id += 1
                    obs = next_obs
                    if terminated or truncated:
                        break
                episode_record_count = len(records["frame_id"]) - episode_record_start
                records["episode_recorded_frame_count"].extend(
                    [episode_record_count] * episode_record_count
                )
            finally:
                if hasattr(env, "close"):
                    env.close()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    collection_config = _collection_config(
        args, task_generation_seed=int(task_generation_seed)
    )
    collector_source = Path(__file__).resolve()
    collector_source_sha256 = hashlib.sha256(
        collector_source.read_bytes()
    ).hexdigest()
    arrays = {
        "dataset_type": np.asarray(DYNAMIC_CAMERA_DATASET_TYPE),
        "schema_version": np.asarray(DYNAMIC_CAMERA_SCHEMA_VERSION, dtype=np.int32),
        "collector_source": np.asarray(str(collector_source)),
        "collector_source_sha256": np.asarray(collector_source_sha256),
        "collection_config_json": np.asarray(
            json.dumps(collection_config, sort_keys=True, separators=(",", ":"))
        ),
        "requested_task_list": np.asarray(args.task_list),
        "camera_name": np.asarray(args.camera),
        "camera_name_by_task_json": np.asarray(
            json.dumps(
                {task: _camera_name(args, task) for task in args.task_list},
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "a4_task_factory": np.asarray(args.a4_task_factory),
        "a4_orbit_clock_hour": np.asarray(
            -1 if args.a4_orbit_clock_hour is None else args.a4_orbit_clock_hour,
            dtype=np.int32,
        ),
        "a4_orbit_elevation_degrees": np.asarray(
            args.a4_orbit_elevation_deg, dtype=np.float32
        ),
        "a4_orbit_radius_m": np.asarray(args.a4_orbit_radius_m, dtype=np.float32),
        "base_camera_pose_family": np.asarray(
            "a4_clock_orbit"
            if args.a4_orbit_clock_hour is not None
            else "environment_native"
        ),
        "image_width": np.asarray(args.width, dtype=np.int32),
        "image_height": np.asarray(args.height, dtype=np.int32),
        "episodes_per_task": np.asarray(args.episodes_per_task, dtype=np.int32),
        "warmup_steps": np.asarray(args.warmup_steps, dtype=np.int32),
        "env_seed_base": np.asarray(args.env_seed_base, dtype=np.int64),
        "task_index_offset": np.asarray(args.task_index_offset, dtype=np.int32),
        "vary_task_index": np.asarray(args.vary_task_index),
        "rgb": np.asarray(records["rgb"], dtype=np.uint8),
        "obs": np.asarray(records["obs"], dtype=np.float32),
        "action": np.asarray(records["action"], dtype=np.float32),
        "reward": np.asarray(records["reward"], dtype=np.float32),
        "success": np.asarray(records["success"], dtype=np.float32),
        "action_source": np.asarray(args.action_source),
        "observation_action_alignment": np.asarray("pre_action"),
        "task_name": np.asarray(records["task_name"]),
        "task_index": np.asarray(records["task_index"], dtype=np.int32),
        "episode_id": np.asarray(records["episode_id"], dtype=np.int32),
        "env_seed": np.asarray(records["env_seed"], dtype=np.int64),
        "frame_id": np.asarray(records["frame_id"], dtype=np.int32),
        "episode_recorded_frame_count": np.asarray(
            records["episode_recorded_frame_count"], dtype=np.int32
        ),
        "keypoint_layout": np.asarray("control"),
        "keypoint_site": np.asarray(keypoint_names or []),
        "keypoints_xyz": np.asarray(records["keypoints_xyz"], dtype=np.float32),
        "q_true": np.asarray(records["q_true"], dtype=np.float32),
        "q_nominal": np.asarray(records["q_nominal"], dtype=np.float32),
        "valid_true": np.asarray(records["valid_true"], dtype=bool),
        "valid_nominal": np.asarray(records["valid_nominal"], dtype=bool),
        "control_keypoint_site": np.asarray(keypoint_names or []),
        "control_keypoints_xyz": np.asarray(
            records["keypoints_xyz"], dtype=np.float32
        ),
        "control_q_true": np.asarray(records["q_true"], dtype=np.float32),
        "control_q_nominal": np.asarray(records["q_nominal"], dtype=np.float32),
        "control_valid_true": np.asarray(records["valid_true"], dtype=bool),
        "control_valid_nominal": np.asarray(records["valid_nominal"], dtype=bool),
        "camera_intrinsic": np.asarray(
            records["camera_intrinsic"], dtype=np.float32
        ),
        "true_extrinsic": np.asarray(records["true_extrinsic"], dtype=np.float32),
        "base_extrinsic": np.asarray(records["base_extrinsic"], dtype=np.float32),
        "nominal_extrinsic": np.asarray(
            records["nominal_extrinsic"], dtype=np.float32
        ),
        "actual_camera_translation_xyz": np.asarray(
            records["actual_translation"], dtype=np.float32
        ),
        "actual_camera_rotation_axis": np.asarray(
            records["actual_axis"], dtype=np.float32
        ),
        "actual_camera_rotation_deg": np.asarray(
            records["actual_degrees"], dtype=np.float32
        ),
        "actual_camera_endpoint_translation_xyz": np.asarray(
            records["endpoint_translation"], dtype=np.float32
        ),
        "actual_camera_endpoint_rotation_axis": np.asarray(
            records["endpoint_axis"], dtype=np.float32
        ),
        "actual_camera_endpoint_rotation_deg": np.asarray(
            records["endpoint_degrees"], dtype=np.float32
        ),
        "actual_camera_trajectory_phase": np.asarray(
            records["trajectory_phase"], dtype=np.float32
        ),
        "camera_intervention_onset": np.asarray(
            records["intervention_onset"], dtype=bool
        ),
        "prior_supervision_valid": np.asarray(
            records["prior_supervision_valid"], dtype=bool
        ),
        "camera_step_frame": np.asarray(records["camera_step_frame"], dtype=np.int32),
        "actual_camera_realized_step_fraction": np.asarray(
            records["realized_step_fraction"], dtype=np.float32
        ),
        "anchor_dropout_active": np.asarray(
            records["anchor_dropout_active"], dtype=bool
        ),
        "anchor_recovery_onset": np.asarray(
            records["anchor_recovery_onset"], dtype=bool
        ),
        "anchor_recovery_segment": np.asarray(
            records["anchor_recovery_segment"], dtype=bool
        ),
        "measurement_valid": np.asarray(records["measurement_valid"], dtype=bool),
        "actual_camera_trajectory_mode": np.asarray(
            args.actual_camera_trajectory
        ),
        "actual_camera_randomization": np.asarray(args.actual_camera_trajectory),
        "actual_camera_step_fraction": np.asarray(
            args.actual_camera_step_fraction, dtype=np.float32
        ),
        "actual_camera_step_fraction_min": np.asarray(
            args.actual_camera_step_fraction_min, dtype=np.float32
        ),
        "actual_camera_step_fraction_max": np.asarray(
            args.actual_camera_step_fraction_max, dtype=np.float32
        ),
        "actual_camera_drift_start_fraction": np.asarray(
            args.actual_camera_drift_start_fraction, dtype=np.float32
        ),
        "actual_camera_drift_end_fraction": np.asarray(
            args.actual_camera_drift_end_fraction, dtype=np.float32
        ),
        "actual_camera_planned_frame_count": np.asarray(
            args.steps_per_episode, dtype=np.int32
        ),
        "nominal_extrinsic_schedule": np.asarray("fixed_base_per_episode"),
        "actual_camera_translation_cm_min": np.asarray(
            args.actual_camera_translation_cm_min, dtype=np.float32
        ),
        "actual_camera_translation_cm_max": np.asarray(
            args.actual_camera_translation_cm_max, dtype=np.float32
        ),
        "actual_camera_rotation_deg_min": np.asarray(
            args.actual_camera_rotation_deg_min, dtype=np.float32
        ),
        "actual_camera_rotation_deg_max": np.asarray(
            args.actual_camera_rotation_deg_max, dtype=np.float32
        ),
        "actual_camera_clean_probability": np.asarray(
            args.actual_camera_clean_probability, dtype=np.float32
        ),
        "independent_rng_streams": np.asarray(args.independent_rng_streams),
        "rng_seed": np.asarray(args.seed, dtype=np.int64),
        "task_generation_seed": np.asarray(task_generation_seed, dtype=np.int64),
        "pixel_error": np.asarray(records["pixel_error"], dtype=np.float32),
        "q_true_inside_patch": np.asarray(records["inside_patch"], dtype=bool),
        "patch_radius": np.asarray(args.patch_radius, dtype=np.int32),
        "paired_clean_view": np.asarray(args.paired_clean_view),
        "rgb_observation_contract": np.asarray(
            "actual_camera_with_visual_anchor_occlusion"
            if args.anchor_dropout_mode == "visual"
            else "actual_camera_unmodified"
        ),
        "measurement_valid_contract": np.asarray(
            "valid_nominal_and_not_anchor_dropout"
        ),
        "paired_view_contract": np.asarray(
            (
                "same_pre_action_physical_state_matched_anchor_visibility_"
                "dynamic_then_clean_rerender_v2"
                if args.anchor_dropout_mode == "visual"
                else "same_pre_action_physical_state_dynamic_then_clean_rerender_v1"
            )
            if args.paired_clean_view
            else "disabled"
        ),
        "paired_clean_rgb_contract": np.asarray(
            (
                "clean_camera_with_matched_visual_anchor_occlusion"
                if args.anchor_dropout_mode == "visual"
                else "clean_camera_unmodified"
            )
            if args.paired_clean_view
            else "disabled"
        ),
        "paired_clean_rgb_unmodified_contract": np.asarray(
            "clean_camera_unmodified_raw_view"
            if args.paired_clean_view and args.anchor_dropout_mode == "visual"
            else "disabled"
        ),
        "anchor_dropout_mode": np.asarray(args.anchor_dropout_mode),
        "anchor_dropout_indices": np.asarray(
            args.anchor_dropout_index_list, dtype=np.int32
        ),
        "anchor_dropout_start_offset": np.asarray(
            args.anchor_dropout_start_offset, dtype=np.int32
        ),
        "anchor_dropout_length": np.asarray(args.anchor_dropout_length, dtype=np.int32),
        "anchor_dropout_maximum_length": np.asarray(
            args.anchor_dropout_maximum_length, dtype=np.int32
        ),
        "anchor_recovery_window_length": np.asarray(
            args.anchor_recovery_window_length, dtype=np.int32
        ),
        "anchor_occlusion_radius_px": np.asarray(
            args.anchor_occlusion_radius_px, dtype=np.int32
        ),
        "anchor_occlusion_value": np.asarray(
            args.anchor_occlusion_value, dtype=np.uint8
        ),
        "minimum_recovery_frames": np.asarray(
            args.minimum_recovery_frames, dtype=np.int32
        ),
        "minimum_history_size": np.asarray(args.minimum_history_size, dtype=np.int32),
        "q_true_usage": np.asarray(
            "offline_target_metrics_and_visual_intervention_only"
            if args.anchor_dropout_mode == "visual"
            else "offline_target_and_metrics_only"
        ),
        "visual_occlusion_center_source": np.asarray(
            "q_true_offline_intervention_only"
            if args.anchor_dropout_mode == "visual"
            else "not_applicable"
        ),
    }
    if args.paired_clean_view:
        arrays["paired_clean_rgb"] = np.asarray(paired_clean_rgb, dtype=np.uint8)
    if args.paired_clean_view and args.anchor_dropout_mode == "visual":
        arrays["paired_clean_rgb_unmodified"] = np.asarray(
            paired_clean_rgb_unmodified, dtype=np.uint8
        )
    if args.anchor_dropout_mode == "visual":
        arrays["rgb_camera_only"] = np.asarray(rgb_camera_only, dtype=np.uint8)
    _validate_collected_arrays(arrays, args)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"output: {output}")
    print(f"samples: {len(records['frame_id'])}")
    print(f"tasks: {args.task_list}")
    print(f"actual_camera_trajectory: {args.actual_camera_trajectory}")
    print("nominal_extrinsic_schedule: fixed_base_per_episode")
    print("observation_action_alignment: pre_action")
    return output


def main(argv: list[str] | None = None) -> int:
    collect(_validated_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
