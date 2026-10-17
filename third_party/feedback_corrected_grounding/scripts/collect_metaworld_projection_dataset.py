#!/usr/bin/env python3
"""Collect MetaWorld frames for FCPG residual-correction experiments.

The output is a compressed `.npz` file where each sample pairs one rendered
RGB frame with one nominal camera projection. The rendered camera can be
physically randomized while the nominal calibration remains fixed, preventing
models from recovering the true projection from robot state alone.

For BC/policy integration, the saved observation, RGB frame, projection, and
action are aligned at the pre-action state: action_t is chosen from obs_t and
paired with image/projections rendered before env.step(action_t). Reward and
success fields are the immediate outcome after applying that action.
"""

from __future__ import annotations

import argparse
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
    DEFAULT_OBSERVER_BODY_NAMES,
    choose_action,
    get_link_observer_points,
    get_site_keypoints,
    observer_visibility_from_segmentation,
    render_segmentation,
    make_env,
    make_scripted_policy,
    parse_direction_list,
    parse_float_list,
)
from fcpg.mujoco_camera import (
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    sample_camera_pose_offset,
    set_fixed_camera_pose_offset,
)
from fcpg.moving_goal import MovingGoalEnv, MovingPushExpert


def choose_perturbation(
    rng: np.random.Generator,
    translations_cm: list[float],
    directions: dict[str, np.ndarray],
    yaws_deg: list[float],
) -> tuple[float, str, np.ndarray, float]:
    trans_cm = float(rng.choice(translations_cm))
    direction_name = str(rng.choice(list(directions.keys())))
    yaw = float(rng.choice(yaws_deg))
    translation_m = (trans_cm / 100.0) * directions[direction_name]
    return trans_cm, direction_name, translation_m, yaw


def make_rng_streams(
    seed: int,
    independent: bool,
) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Return action, camera, and nominal-projection RNG streams."""

    if not independent:
        shared = np.random.default_rng(seed)
        return shared, shared, shared
    action_seed, camera_seed, projection_seed = np.random.SeedSequence(seed).spawn(3)
    return (
        np.random.default_rng(action_seed),
        np.random.default_rng(camera_seed),
        np.random.default_rng(projection_seed),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="reach-v3")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--steps-per-episode", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--perturbations-per-frame", type=int, default=2)
    parser.add_argument("--action-source", choices=["random", "scripted"], default="random")
    parser.add_argument("--translations-cm", default="0,0.5,1,2,4")
    parser.add_argument("--directions", default="x+,x-,y+,y-")
    parser.add_argument("--yaws-deg", default="0,1,2,5,10")
    parser.add_argument("--patch-radius", type=int, default=16)
    parser.add_argument(
        "--keypoint-layout",
        choices=("control", "observer"),
        default="control",
        help="Use end-effector control anchors or full-arm observer points as q arrays.",
    )
    parser.add_argument(
        "--observer-body-names",
        default=",".join(DEFAULT_OBSERVER_BODY_NAMES),
        help="Comma-separated robot bodies used when --keypoint-layout observer.",
    )
    parser.add_argument(
        "--observer-visibility",
        choices=("frustum", "segmentation"),
        default="frustum",
        help="Optionally require each observer projection to land on its own link.",
    )
    parser.add_argument("--visibility-patch-radius-px", type=int, default=4)
    parser.add_argument(
        "--actual-camera-translation-cm-min",
        type=float,
        default=0.0,
        help="Minimum non-clean translation magnitude of the rendered camera.",
    )
    parser.add_argument(
        "--actual-camera-translation-cm-max",
        type=float,
        default=0.0,
        help="Maximum continuous world-frame translation of the rendered camera.",
    )
    parser.add_argument(
        "--actual-camera-rotation-deg-min",
        type=float,
        default=0.0,
        help="Minimum non-clean rotation magnitude of the rendered camera.",
    )
    parser.add_argument(
        "--actual-camera-rotation-deg-max",
        type=float,
        default=0.0,
        help="Maximum continuous axis-angle rotation of the rendered camera.",
    )
    parser.add_argument(
        "--actual-camera-randomization",
        choices=["episode", "frame"],
        default="episode",
        help="Resample the rendered camera offset once per episode or recorded frame.",
    )
    parser.add_argument(
        "--actual-camera-clean-probability",
        type=float,
        default=0.0,
        help="Probability that an actual-camera sample uses the unperturbed base pose.",
    )
    parser.add_argument(
        "--vary-task-index",
        action="store_true",
        help="Cycle MetaWorld train task index across episodes.",
    )
    parser.add_argument("--task-index-offset", type=int, default=0)
    parser.add_argument("--env-seed-base", type=int, default=0)
    parser.add_argument(
        "--task-generation-seed",
        type=int,
        default=None,
        help=(
            "Seed used by MetaWorld to generate MT1 task definitions. Defaults "
            "to --seed; keep it fixed across camera conditions for paired data."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--moving-goal",
        action="store_true",
        help="Move the Reach/Push target smoothly throughout each episode.",
    )
    parser.add_argument("--moving-goal-amplitude-xy", default="0.02,0.01")
    parser.add_argument("--moving-goal-period-raw-steps", type=float, default=300.0)
    parser.add_argument("--moving-goal-seed-base", type=int, default=90_000)
    parser.add_argument(
        "--independent-rng-streams",
        action="store_true",
        help=(
            "Separate action, actual-camera, and nominal-projection RNG streams so "
            "camera-condition changes preserve the physical trajectory."
        ),
    )
    parser.add_argument("--output", default="data/metaworld_projection/reach_v3_small.npz")
    args = parser.parse_args()
    if not 0.0 <= args.actual_camera_clean_probability <= 1.0:
        parser.error("--actual-camera-clean-probability must be in [0, 1]")
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
    if args.visibility_patch_radius_px < 0:
        parser.error("--visibility-patch-radius-px must be non-negative")

    os.environ.setdefault("MUJOCO_GL", "egl")

    task_generation_seed = (
        args.seed if args.task_generation_seed is None else args.task_generation_seed
    )
    action_rng, camera_rng, projection_rng = make_rng_streams(
        args.seed, args.independent_rng_streams
    )
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    moving_goal_amplitude = parse_float_list(args.moving_goal_amplitude_xy)
    if len(moving_goal_amplitude) != 2:
        parser.error("--moving-goal-amplitude-xy must contain exactly two values")
    if args.moving_goal and not set(tasks) <= {"reach-v3", "push-v3"}:
        parser.error("--moving-goal supports only reach-v3 and push-v3")
    if args.moving_goal and args.perturbations_per_frame != 1:
        parser.error("--moving-goal requires --perturbations-per-frame 1")
    observer_body_names = [
        name.strip() for name in args.observer_body_names.split(",") if name.strip()
    ]
    if args.keypoint_layout == "observer" and not observer_body_names:
        parser.error("--observer-body-names must not be empty")
    translations_cm = parse_float_list(args.translations_cm)
    directions = parse_direction_list(args.directions)
    yaws_deg = parse_float_list(args.yaws_deg)

    rgbs = []
    obs_list = []
    actions = []
    rewards = []
    successes = []
    action_sources = []
    task_names = []
    task_indices = []
    episode_ids = []
    env_seeds = []
    frame_ids = []
    keypoints_xyz = []
    q_true_list = []
    q_nominal_list = []
    valid_true_list = []
    valid_nominal_list = []
    camera_intrinsics = []
    true_extrinsics = []
    base_extrinsics = []
    nominal_extrinsics = []
    actual_camera_translations = []
    actual_camera_rotation_axes = []
    actual_camera_rotation_degrees = []
    translation_cms = []
    direction_names = []
    yaw_degs = []
    pixel_errors = []
    inside_patch_list = []
    observer_visibility_list = []
    control_keypoints_xyz = []
    control_q_true_list = []
    control_q_nominal_list = []
    control_valid_true_list = []
    control_valid_nominal_list = []
    moving_goal_positions = []
    moving_goal_next_positions = []
    controlled_entity_xyz = []
    policy_keypoints_xyz = []
    policy_q_true_list = []
    policy_q_nominal_list = []
    policy_valid_true_list = []
    policy_valid_nominal_list = []
    site_names_reference: list[str] | None = None
    control_names_reference: list[str] | None = None

    sample_count = 0
    for task_offset, task_name in enumerate(tasks):
        if args.action_source == "scripted":
            policy = (
                MovingPushExpert()
                if args.moving_goal and task_name == "push-v3"
                else make_scripted_policy(task_name)
            )
        else:
            policy = None
        for episode_id in range(args.episodes_per_task):
            task_index = args.task_index_offset
            if args.vary_task_index:
                task_index += episode_id
            env_seed = (
                args.env_seed_base + task_offset * 1_000_000 + episode_id
            )
            base_env = make_env(
                task_name,
                args.camera,
                args.width,
                args.height,
                task_index=task_index,
                benchmark_seed=task_generation_seed,
            )
            env = (
                MovingGoalEnv(
                    base_env,
                    task=task_name,
                    seed=args.moving_goal_seed_base + task_offset * 1_000_000 + episode_id,
                    amplitude_xy=tuple(moving_goal_amplitude),
                    period_raw_steps=args.moving_goal_period_raw_steps,
                )
                if args.moving_goal
                else base_env
            )
            camera_id = get_camera_id(env.model, args.camera)
            intrinsic = camera_intrinsic_from_fovy(args.width, args.height, float(env.model.cam_fovy[camera_id]))
            obs, info = env.reset(seed=env_seed)
            base_camera_position = np.asarray(env.model.cam_pos[camera_id], dtype=np.float64).copy()
            base_camera_quaternion = np.asarray(env.model.cam_quat[camera_id], dtype=np.float64).copy()
            base_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
            episode_camera_offset = sample_camera_pose_offset(
                camera_rng,
                args.actual_camera_translation_cm_max,
                args.actual_camera_rotation_deg_max,
                args.actual_camera_clean_probability,
                translation_cm_min=args.actual_camera_translation_cm_min,
                rotation_deg_min=args.actual_camera_rotation_deg_min,
            )
            if args.actual_camera_randomization == "episode":
                set_fixed_camera_pose_offset(
                    env.model,
                    env.data,
                    camera_id,
                    base_camera_position,
                    base_camera_quaternion,
                    *episode_camera_offset,
                )
            recorded_frame_id = 0
            for raw_frame_id in range(args.steps_per_episode + args.warmup_steps):
                action = choose_action(
                    env.action_space, action_rng, obs, policy=policy
                )

                should_record = raw_frame_id >= args.warmup_steps
                if should_record:
                    # Record the state that the action was computed from. This
                    # keeps BC samples aligned as (obs_t, image_t, q_t, action_t).
                    obs_before_step = np.asarray(obs, dtype=np.float32).copy()
                    if args.actual_camera_randomization == "frame":
                        actual_camera_offset = sample_camera_pose_offset(
                            camera_rng,
                            args.actual_camera_translation_cm_max,
                            args.actual_camera_rotation_deg_max,
                            args.actual_camera_clean_probability,
                            translation_cm_min=args.actual_camera_translation_cm_min,
                            rotation_deg_min=args.actual_camera_rotation_deg_min,
                        )
                        set_fixed_camera_pose_offset(
                            env.model,
                            env.data,
                            camera_id,
                            base_camera_position,
                            base_camera_quaternion,
                            *actual_camera_offset,
                        )
                    else:
                        actual_camera_offset = episode_camera_offset
                    image = env.render()
                    true_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
                    true_camera = Camera(
                        intrinsic=intrinsic,
                        extrinsic_cam_from_world=true_extrinsic,
                        width=args.width,
                        height=args.height,
                    )
                    control_state = get_site_keypoints(
                        env.model, env.data, DEFAULT_KEYPOINT_SITES
                    )
                    if control_names_reference is None:
                        control_names_reference = control_state.names
                    elif control_names_reference != control_state.names:
                        raise RuntimeError(
                            "Control-anchor mismatch: "
                            f"{control_names_reference} vs {control_state.names}"
                        )
                    if args.keypoint_layout == "observer":
                        keypoint_state = get_link_observer_points(
                            env.model, env.data, observer_body_names
                        )
                    else:
                        keypoint_state = control_state
                    if site_names_reference is None:
                        site_names_reference = keypoint_state.names
                    elif site_names_reference != keypoint_state.names:
                        raise RuntimeError(
                            "Keypoint layout mismatch: "
                            f"{site_names_reference} vs {keypoint_state.names}"
                        )

                    q_true, valid_true = project_points(
                        keypoint_state.xyz, true_camera
                    )
                    control_q_true, control_valid_true = project_points(
                        control_state.xyz, true_camera
                    )
                    entity_xyz = np.asarray(
                        obs_before_step[:3]
                        if task_name == "reach-v3"
                        else obs_before_step[4:7],
                        dtype=np.float64,
                    )
                    goal_xyz = np.asarray(env._target_pos, dtype=np.float64).copy()
                    task_anchor_xyz = np.stack([entity_xyz, goal_xyz])
                    task_q_true, task_valid_true = project_points(
                        task_anchor_xyz, true_camera
                    )
                    visual_visibility = valid_true.copy()
                    if (
                        args.keypoint_layout == "observer"
                        and args.observer_visibility == "segmentation"
                    ):
                        segmentation = render_segmentation(env)
                        visual_visibility = observer_visibility_from_segmentation(
                            env.model,
                            segmentation,
                            q_true,
                            observer_body_names,
                            radius_px=args.visibility_patch_radius_px,
                        )
                        valid_true = valid_true & visual_visibility
                    perturbation_records = []
                    for _ in range(args.perturbations_per_frame):
                        trans_cm, direction_name, translation_m, yaw = choose_perturbation(
                            projection_rng, translations_cm, directions, yaws_deg
                        )
                        nominal_extrinsic = perturb_extrinsic(
                            base_extrinsic,
                            translation_xyz=translation_m,
                            yaw_degrees=yaw,
                        )
                        nominal_camera = Camera(
                            intrinsic=intrinsic,
                            extrinsic_cam_from_world=nominal_extrinsic,
                            width=args.width,
                            height=args.height,
                        )
                        q_nominal, valid_nominal = project_points(
                            keypoint_state.xyz, nominal_camera
                        )
                        control_q_nominal, control_valid_nominal = project_points(
                            control_state.xyz, nominal_camera
                        )
                        task_q_nominal, task_valid_nominal = project_points(
                            task_anchor_xyz, nominal_camera
                        )
                        err = np.linalg.norm(q_nominal - q_true, axis=1)
                        inside_patch = np.all(np.abs(q_true - np.round(q_nominal)) <= args.patch_radius, axis=1)
                        perturbation_records.append(
                            (
                                trans_cm,
                                direction_name,
                                yaw,
                                nominal_extrinsic,
                                q_nominal,
                                valid_nominal,
                                control_q_nominal,
                                control_valid_nominal,
                                task_q_nominal,
                                task_valid_nominal,
                                err,
                                inside_patch,
                            )
                        )

                next_obs, reward, terminated, truncated, step_info = env.step(action)

                if should_record:
                    success_value = float(step_info.get("success", 0.0)) if isinstance(step_info, dict) else 0.0
                    for (
                        trans_cm,
                        direction_name,
                        yaw,
                        nominal_extrinsic,
                        q_nominal,
                        valid_nominal,
                        control_q_nominal,
                        control_valid_nominal,
                        task_q_nominal,
                        task_valid_nominal,
                        err,
                        inside_patch,
                    ) in perturbation_records:
                        rgbs.append(image.astype(np.uint8))
                        obs_list.append(obs_before_step)
                        actions.append(action.astype(np.float32))
                        rewards.append(float(reward))
                        successes.append(success_value)
                        action_sources.append(args.action_source)
                        task_names.append(task_name)
                        task_indices.append(task_index)
                        episode_ids.append(episode_id)
                        env_seeds.append(env_seed)
                        frame_ids.append(recorded_frame_id)
                        keypoints_xyz.append(keypoint_state.xyz.astype(np.float32))
                        q_true_list.append(q_true.astype(np.float32))
                        q_nominal_list.append(q_nominal.astype(np.float32))
                        valid_true_list.append(valid_true.astype(bool))
                        valid_nominal_list.append(valid_nominal.astype(bool))
                        observer_visibility_list.append(
                            visual_visibility.astype(bool)
                        )
                        control_keypoints_xyz.append(
                            control_state.xyz.astype(np.float32)
                        )
                        control_q_true_list.append(
                            control_q_true.astype(np.float32)
                        )
                        control_q_nominal_list.append(
                            control_q_nominal.astype(np.float32)
                        )
                        control_valid_true_list.append(
                            control_valid_true.astype(bool)
                        )
                        control_valid_nominal_list.append(
                            control_valid_nominal.astype(bool)
                        )
                        moving_goal_positions.append(goal_xyz.astype(np.float32))
                        next_goal = np.asarray(
                            env._target_pos if args.moving_goal else goal_xyz,
                            dtype=np.float32,
                        )
                        moving_goal_next_positions.append(
                            next_goal.copy()
                        )
                        controlled_entity_xyz.append(entity_xyz.astype(np.float32))
                        policy_keypoints_xyz.append(
                            np.concatenate(
                                [control_state.xyz, task_anchor_xyz], axis=0
                            ).astype(np.float32)
                        )
                        policy_q_true_list.append(
                            np.concatenate([control_q_true, task_q_true], axis=0).astype(
                                np.float32
                            )
                        )
                        policy_q_nominal_list.append(
                            np.concatenate(
                                [control_q_nominal, task_q_nominal], axis=0
                            ).astype(np.float32)
                        )
                        policy_valid_true_list.append(
                            np.concatenate(
                                [control_valid_true, task_valid_true], axis=0
                            ).astype(bool)
                        )
                        policy_valid_nominal_list.append(
                            np.concatenate(
                                [control_valid_nominal, task_valid_nominal], axis=0
                            ).astype(bool)
                        )
                        camera_intrinsics.append(intrinsic.astype(np.float32))
                        true_extrinsics.append(true_extrinsic.astype(np.float32))
                        base_extrinsics.append(base_extrinsic.astype(np.float32))
                        nominal_extrinsics.append(nominal_extrinsic.astype(np.float32))
                        actual_camera_translations.append(actual_camera_offset[0].astype(np.float32))
                        actual_camera_rotation_axes.append(actual_camera_offset[1].astype(np.float32))
                        actual_camera_rotation_degrees.append(float(actual_camera_offset[2]))
                        translation_cms.append(trans_cm)
                        direction_names.append(direction_name)
                        yaw_degs.append(yaw)
                        pixel_errors.append(err.astype(np.float32))
                        inside_patch_list.append(inside_patch.astype(bool))
                        sample_count += 1
                    recorded_frame_id += 1

                obs = next_obs
                info = step_info
                if terminated or truncated:
                    break
            if hasattr(env, "close"):
                env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rgb=np.asarray(rgbs, dtype=np.uint8),
        obs=np.asarray(obs_list, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        success=np.asarray(successes, dtype=np.float32),
        action_source=np.asarray(action_sources),
        observation_action_alignment=np.asarray("pre_action"),
        task_name=np.asarray(task_names),
        task_index=np.asarray(task_indices, dtype=np.int32),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        env_seed=np.asarray(env_seeds, dtype=np.int64),
        frame_id=np.asarray(frame_ids, dtype=np.int32),
        keypoint_layout=np.asarray(args.keypoint_layout),
        keypoint_site=np.asarray(site_names_reference or []),
        observer_body_name=np.asarray(
            observer_body_names if args.keypoint_layout == "observer" else []
        ),
        observer_visibility_mode=np.asarray(args.observer_visibility),
        observer_visibility=np.asarray(observer_visibility_list, dtype=bool),
        keypoints_xyz=np.asarray(keypoints_xyz, dtype=np.float32),
        q_true=np.asarray(q_true_list, dtype=np.float32),
        q_nominal=np.asarray(q_nominal_list, dtype=np.float32),
        valid_true=np.asarray(valid_true_list, dtype=bool),
        valid_nominal=np.asarray(valid_nominal_list, dtype=bool),
        control_keypoint_site=np.asarray(control_names_reference or []),
        control_keypoints_xyz=np.asarray(control_keypoints_xyz, dtype=np.float32),
        control_q_true=np.asarray(control_q_true_list, dtype=np.float32),
        control_q_nominal=np.asarray(control_q_nominal_list, dtype=np.float32),
        control_valid_true=np.asarray(control_valid_true_list, dtype=bool),
        control_valid_nominal=np.asarray(control_valid_nominal_list, dtype=bool),
        moving_goal=np.asarray(args.moving_goal),
        moving_goal_amplitude_xy=np.asarray(
            moving_goal_amplitude, dtype=np.float32
        ),
        moving_goal_period_raw_steps=np.asarray(
            args.moving_goal_period_raw_steps, dtype=np.float32
        ),
        moving_goal_seed_base=np.asarray(
            args.moving_goal_seed_base, dtype=np.int64
        ),
        moving_goal_position=np.asarray(moving_goal_positions, dtype=np.float32),
        moving_goal_next_position=np.asarray(
            moving_goal_next_positions, dtype=np.float32
        ),
        controlled_entity_xyz=np.asarray(
            controlled_entity_xyz, dtype=np.float32
        ),
        policy_anchor_role=np.asarray(
            [
                *(control_names_reference or []),
                "controlled_entity",
                "moving_goal",
            ]
        ),
        policy_keypoints_xyz=np.asarray(policy_keypoints_xyz, dtype=np.float32),
        policy_q_true=np.asarray(policy_q_true_list, dtype=np.float32),
        policy_q_nominal=np.asarray(policy_q_nominal_list, dtype=np.float32),
        policy_valid_true=np.asarray(policy_valid_true_list, dtype=bool),
        policy_valid_nominal=np.asarray(policy_valid_nominal_list, dtype=bool),
        camera_intrinsic=np.asarray(camera_intrinsics, dtype=np.float32),
        true_extrinsic=np.asarray(true_extrinsics, dtype=np.float32),
        base_extrinsic=np.asarray(base_extrinsics, dtype=np.float32),
        nominal_extrinsic=np.asarray(nominal_extrinsics, dtype=np.float32),
        actual_camera_translation_xyz=np.asarray(actual_camera_translations, dtype=np.float32),
        actual_camera_rotation_axis=np.asarray(actual_camera_rotation_axes, dtype=np.float32),
        actual_camera_rotation_deg=np.asarray(actual_camera_rotation_degrees, dtype=np.float32),
        actual_camera_translation_cm_min=np.asarray(
            args.actual_camera_translation_cm_min, dtype=np.float32
        ),
        actual_camera_translation_cm_max=np.asarray(
            args.actual_camera_translation_cm_max, dtype=np.float32
        ),
        actual_camera_rotation_deg_min=np.asarray(
            args.actual_camera_rotation_deg_min, dtype=np.float32
        ),
        actual_camera_rotation_deg_max=np.asarray(
            args.actual_camera_rotation_deg_max, dtype=np.float32
        ),
        actual_camera_randomization=np.asarray(args.actual_camera_randomization),
        actual_camera_clean_probability=np.asarray(args.actual_camera_clean_probability, dtype=np.float32),
        independent_rng_streams=np.asarray(args.independent_rng_streams),
        rng_seed=np.asarray(args.seed, dtype=np.int64),
        task_generation_seed=np.asarray(task_generation_seed, dtype=np.int64),
        translation_cm=np.asarray(translation_cms, dtype=np.float32),
        translation_direction=np.asarray(direction_names),
        yaw_deg=np.asarray(yaw_degs, dtype=np.float32),
        pixel_error=np.asarray(pixel_errors, dtype=np.float32),
        q_true_inside_patch=np.asarray(inside_patch_list, dtype=bool),
        patch_radius=np.asarray(args.patch_radius, dtype=np.int32),
    )

    errors = np.asarray(pixel_errors, dtype=np.float32).reshape(sample_count, -1)
    inside = np.asarray(inside_patch_list, dtype=bool).reshape(sample_count, -1)
    visibility = np.asarray(observer_visibility_list, dtype=bool).reshape(
        sample_count, -1
    )
    print(f"output: {output}")
    print(f"samples: {sample_count}")
    print(f"tasks: {tasks}")
    print(f"action_source: {args.action_source}")
    print(f"vary_task_index: {args.vary_task_index}")
    print(f"task_index_offset: {args.task_index_offset}")
    print(f"env_seed_base: {args.env_seed_base}")
    print(f"task_generation_seed: {task_generation_seed}")
    print(
        "actual_camera_translation_cm_range: "
        f"[{args.actual_camera_translation_cm_min}, "
        f"{args.actual_camera_translation_cm_max}]"
    )
    print(
        "actual_camera_rotation_deg_range: "
        f"[{args.actual_camera_rotation_deg_min}, "
        f"{args.actual_camera_rotation_deg_max}]"
    )
    print(f"actual_camera_randomization: {args.actual_camera_randomization}")
    print(f"actual_camera_clean_probability: {args.actual_camera_clean_probability}")
    print("observation_action_alignment: pre_action")
    print(f"keypoint_layout: {args.keypoint_layout}")
    print(f"keypoint_sites: {site_names_reference}")
    print(f"control_keypoint_sites: {control_names_reference}")
    print(f"observer_visibility_mode: {args.observer_visibility}")
    print(f"rgb_shape: {np.asarray(rgbs).shape}")
    print(f"obs_shape: {np.asarray(obs_list).shape}")
    print(f"q_shape: {np.asarray(q_true_list).shape}")
    print(f"pixel_error_mean: {float(errors.mean()):.4f}")
    print(f"pixel_error_p90: {float(np.percentile(errors, 90)):.4f}")
    print(f"pixel_error_max: {float(errors.max()):.4f}")
    print(f"inside_patch_rate: {float(inside.mean()):.4f}")
    print(f"visual_visibility_rate: {float(visibility.mean()):.4f}")
    if successes:
        print(f"success_mean: {float(np.mean(successes)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
