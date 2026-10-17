#!/usr/bin/env python3
"""Collect exact causal expert H8 prefixes for a deployable LeWM proposer."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.geometry import Camera, project_points  # noqa: E402
from fcpg.metaworld_snapshot import (  # noqa: E402
    capture_metaworld_snapshot,
    restore_metaworld_snapshot,
)
from fcpg.metaworld_utils import (  # noqa: E402
    DEFAULT_KEYPOINT_SITES,
    get_site_keypoints,
    make_env,
    make_scripted_policy,
)
from fcpg.moving_goal import MovingGoalEnv, MovingPushExpert  # noqa: E402
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    sample_camera_pose_offset,
    set_fixed_camera_pose_offset,
)


TASKS = ("reach-v3", "push-v3")
TASK_INDEX = {"reach-v3": 2, "push-v3": 3}
HORIZON = 8
RAW_STEPS_PER_TOKEN = 5
PREFIX_RAW_STEPS = HORIZON * RAW_STEPS_PER_TOKEN


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--snapshots-per-episode", type=int, default=8)
    parser.add_argument("--snapshot-stride", type=int, default=5)
    parser.add_argument("--max-source-steps", type=int, default=120)
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--camera-rotation-deg", type=float, default=5.0)
    parser.add_argument("--camera-clean-probability", type=float, default=0.25)
    parser.add_argument("--amplitude-xy", type=float, nargs=2, default=(0.02, 0.01))
    parser.add_argument("--period-raw-steps", type=float, default=220.0)
    parser.add_argument("--moving-goal-seed-base", type=int, default=194000)
    parser.add_argument("--environment-seed-base", type=int, default=294000)
    parser.add_argument("--benchmark-seed", type=int, default=30103)
    parser.add_argument("--seed", type=int, default=94001)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/metaworld_projection/goal_conditioned_action_prefixes_v1.npz",
    )
    return parser


def _bounded_action(policy: Any, env: Any, observation: np.ndarray) -> np.ndarray:
    value = np.asarray(policy.get_action(observation), dtype=np.float32)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("expert action must be one finite 4-vector")
    return np.clip(value, low, high).astype(np.float32, copy=False)


def _history(actions: list[np.ndarray]) -> np.ndarray:
    result = np.zeros((2, 20), dtype=np.float32)
    recent = np.asarray(actions[-10:], dtype=np.float32)
    if len(recent):
        result.reshape(10, 4)[-len(recent) :] = recent
    return result


def _restore(
    base_env: Any,
    moving_env: MovingGoalEnv,
    snapshot: Any,
    moving_state: dict[str, Any],
    observation: np.ndarray,
) -> None:
    restore_metaworld_snapshot(base_env, snapshot, restore_process_rng=True)
    moving_env.load_state_dict(moving_state)
    moving_env._last_observation = np.asarray(observation, dtype=np.float32).copy()


def _expert_prefix(
    base_env: Any,
    moving_env: MovingGoalEnv,
    policy: Any,
    observation: np.ndarray,
) -> np.ndarray | None:
    snapshot = capture_metaworld_snapshot(base_env, include_process_rng=True)
    moving_state = moving_env.state_dict()
    branch_observation = np.asarray(observation, dtype=np.float32).copy()
    branch_policy = copy.deepcopy(policy)
    actions: list[np.ndarray] = []
    try:
        for _ in range(PREFIX_RAW_STEPS):
            action = _bounded_action(branch_policy, moving_env, branch_observation)
            branch_observation, _reward, terminated, truncated, _info = moving_env.step(action)
            actions.append(action.copy())
            if terminated or truncated:
                return None
            branch_observation = np.asarray(branch_observation, dtype=np.float32)
    finally:
        _restore(base_env, moving_env, snapshot, moving_state, observation)
    result = np.asarray(actions, dtype=np.float32)
    return (
        np.ascontiguousarray(result.reshape(HORIZON, 20))
        if result.shape == (PREFIX_RAW_STEPS, 4)
        else None
    )


def main() -> int:
    args = _parser().parse_args()
    if (
        int(args.episodes) < 4
        or int(args.episodes) > 48
        or int(args.snapshots_per_episode) <= 0
        or int(args.snapshot_stride) <= 0
    ):
        raise ValueError("episodes must be 4..48 and snapshot settings positive")
    rng = np.random.default_rng(int(args.seed))
    outputs: dict[str, list[Any]] = {
        "current_rgb": [],
        "current_obs": [],
        "current_xyz": [],
        "current_q_true": [],
        "current_q_nominal": [],
        "current_valid_true": [],
        "current_valid_nominal": [],
        "current_goal": [],
        "current_goal_velocity": [],
        "history_action": [],
        "expert_action_prefix": [],
        "camera_intrinsic": [],
        "true_extrinsic": [],
        "task_name": [],
        "task_index": [],
        "episode_id": [],
        "snapshot_step": [],
        "split": [],
    }
    train_episodes = int(round(0.75 * int(args.episodes)))
    for task_offset, task in enumerate(TASKS):
        for episode in range(int(args.episodes)):
            base_env = make_env(
                task,
                args.camera,
                int(args.width),
                int(args.height),
                task_index=episode,
                benchmark_seed=int(args.benchmark_seed),
            )
            moving_env = MovingGoalEnv(
                base_env,
                task=task,
                seed=int(args.moving_goal_seed_base) + 100_000 * task_offset + episode,
                amplitude_xy=tuple(float(v) for v in args.amplitude_xy),
                period_raw_steps=float(args.period_raw_steps),
            )
            policy = MovingPushExpert() if task == "push-v3" else make_scripted_policy(task)
            observation, _info = moving_env.reset(
                seed=int(args.environment_seed_base) + 100_000 * task_offset + episode
            )
            observation = np.asarray(observation, dtype=np.float32)
            camera_id = get_camera_id(moving_env.model, args.camera)
            intrinsic = camera_intrinsic_from_fovy(
                int(args.width),
                int(args.height),
                float(moving_env.model.cam_fovy[camera_id]),
            )
            base_position = np.asarray(moving_env.model.cam_pos[camera_id]).copy()
            base_quaternion = np.asarray(moving_env.model.cam_quat[camera_id]).copy()
            nominal_extrinsic = camera_extrinsic_cv_from_mujoco(moving_env.data, camera_id)
            offset = sample_camera_pose_offset(
                rng,
                0.0,
                float(args.camera_rotation_deg),
                float(args.camera_clean_probability),
            )
            set_fixed_camera_pose_offset(
                moving_env.model,
                moving_env.data,
                camera_id,
                base_position,
                base_quaternion,
                *offset,
            )
            source_actions: list[np.ndarray] = []
            accepted = 0
            try:
                for raw_step in range(int(args.max_source_steps)):
                    source_action = _bounded_action(policy, moving_env, observation)
                    if (
                        raw_step % int(args.snapshot_stride) == 0
                        and accepted < int(args.snapshots_per_episode)
                    ):
                        prefix = _expert_prefix(
                            base_env,
                            moving_env,
                            policy,
                            observation,
                        )
                        if prefix is not None:
                            image = np.asarray(moving_env.render(), dtype=np.uint8)
                            xyz = get_site_keypoints(
                                moving_env.model,
                                moving_env.data,
                                DEFAULT_KEYPOINT_SITES,
                            ).xyz.astype(np.float32)
                            true_extrinsic = camera_extrinsic_cv_from_mujoco(
                                moving_env.data, camera_id
                            )
                            q_true, valid_true = project_points(
                                xyz,
                                Camera(
                                    intrinsic,
                                    true_extrinsic,
                                    int(args.width),
                                    int(args.height),
                                ),
                            )
                            q_nominal, valid_nominal = project_points(
                                xyz,
                                Camera(
                                    intrinsic,
                                    nominal_extrinsic,
                                    int(args.width),
                                    int(args.height),
                                ),
                            )
                            goal = np.asarray(moving_env._target_pos, dtype=np.float32).copy()
                            next_goal = moving_env.controller.position(
                                moving_env.raw_step + 1
                            ).astype(np.float32)
                            outputs["current_rgb"].append(image)
                            outputs["current_obs"].append(observation.copy())
                            outputs["current_xyz"].append(xyz)
                            outputs["current_q_true"].append(np.asarray(q_true, dtype=np.float32))
                            outputs["current_q_nominal"].append(
                                np.asarray(q_nominal, dtype=np.float32)
                            )
                            outputs["current_valid_true"].append(np.asarray(valid_true, dtype=bool))
                            outputs["current_valid_nominal"].append(
                                np.asarray(valid_nominal, dtype=bool)
                            )
                            outputs["current_goal"].append(goal)
                            outputs["current_goal_velocity"].append(next_goal - goal)
                            outputs["history_action"].append(_history(source_actions))
                            outputs["expert_action_prefix"].append(prefix)
                            outputs["camera_intrinsic"].append(
                                np.asarray(intrinsic, dtype=np.float32)
                            )
                            outputs["true_extrinsic"].append(
                                np.asarray(true_extrinsic, dtype=np.float32)
                            )
                            outputs["task_name"].append(task)
                            outputs["task_index"].append(TASK_INDEX[task])
                            outputs["episode_id"].append(episode)
                            outputs["snapshot_step"].append(raw_step)
                            outputs["split"].append(
                                "train" if episode < train_episodes else "development"
                            )
                            accepted += 1
                    observation, _reward, terminated, truncated, _info = moving_env.step(
                        source_action
                    )
                    observation = np.asarray(observation, dtype=np.float32)
                    source_actions.append(source_action.copy())
                    if terminated or truncated or accepted >= int(args.snapshots_per_episode):
                        break
            finally:
                moving_env.close()
        print(f"completed {task}", flush=True)

    if not outputs["current_rgb"]:
        raise RuntimeError("no proposal rows were collected")
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **{name: np.asarray(value) for name, value in outputs.items()},
        protocol=np.asarray("goal_conditioned_action_prefix_collection_v1"),
        horizon=np.asarray(HORIZON),
        raw_steps_per_token=np.asarray(RAW_STEPS_PER_TOKEN),
    )
    print(f"saved {len(outputs['current_rgb'])} rows to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
