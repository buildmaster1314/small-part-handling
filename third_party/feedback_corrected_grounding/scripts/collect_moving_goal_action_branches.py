#!/usr/bin/env python3
"""Collect small same-state action branches for a moving-goal LeWM pilot.

The collector is intentionally a development tool.  It replays one scripted
trajectory, snapshots a few active states, and executes several different
open-loop action prefixes from each exact state.  The resulting file contains
the current observation plus future robot/entity/goal trajectories; no future
visual observation is used as a model input.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.geometry import Camera, perturb_extrinsic, project_points  # noqa: E402
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


RAW_ACTION_DIM = 4
RAW_STEPS_PER_TOKEN = 5
HORIZON = 8
CANDIDATE_COUNT = 8
BRANCH_RAW_STEPS = HORIZON * RAW_STEPS_PER_TOKEN
TASK_INDEX = {"reach-v3": 2, "push-v3": 3}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--tasks", default="reach-v3,push-v3")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--snapshot-steps", default="20,60,100,140")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--amplitude-xy", default="0.02,0.01")
    parser.add_argument("--period-raw-steps", type=float, default=220.0)
    parser.add_argument(
        "--goal-amplitude-jitter",
        type=float,
        default=0.25,
        help="episode-wise relative jitter for moving-goal amplitudes",
    )
    parser.add_argument(
        "--goal-period-jitter",
        type=float,
        default=0.20,
        help="episode-wise relative jitter for the moving-goal period",
    )
    parser.add_argument("--candidate-amplitude", type=float, default=0.16)
    parser.add_argument("--moving-goal-seed-base", type=int, default=92000)
    parser.add_argument("--env-seed-base", type=int, default=72000)
    parser.add_argument("--task-generation-seed", type=int, default=62000)
    parser.add_argument("--seed", type=int, default=82000)
    parser.add_argument("--camera-rotation-deg", type=float, default=5.0)
    parser.add_argument("--camera-clean-probability", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metaworld_projection/moving_goal_reach_push_branches_dev.npz"),
    )
    return parser


def _parse_pair(value: str, label: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 2 or not np.isfinite(values).all() or any(v <= 0 for v in values):
        raise ValueError(f"{label} must contain two positive finite values")
    return float(values[0]), float(values[1])


def _parse_steps(value: str, *, max_step: int) -> tuple[int, ...]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(step < 0 or step + BRANCH_RAW_STEPS >= max_step for step in values):
        raise ValueError("snapshot steps must leave a complete 40-step future")
    return tuple(values)


def _entity(observation: np.ndarray, task: str) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float32)
    start = 0 if task == "reach-v3" else 4
    return np.ascontiguousarray(values[start : start + 3], dtype=np.float32)


def _history(actions: list[np.ndarray]) -> np.ndarray:
    result = np.zeros((2, RAW_STEPS_PER_TOKEN * RAW_ACTION_DIM), dtype=np.float32)
    if actions:
        flat = np.asarray(actions[-2 * RAW_STEPS_PER_TOKEN :], dtype=np.float32).reshape(-1)
        result.reshape(-1)[-flat.size :] = flat
    return result


def _candidate_actions(
    expert: np.ndarray,
    rng: np.random.Generator,
    *,
    amplitude: float,
) -> np.ndarray:
    """Create anonymous smooth perturbations around one executable prefix."""

    actions = np.repeat(np.asarray(expert, dtype=np.float32)[None], CANDIDATE_COUNT, axis=0)
    if actions.shape != (CANDIDATE_COUNT, BRANCH_RAW_STEPS, RAW_ACTION_DIM):
        raise ValueError("expert prefix has an unexpected shape")
    knot_x = np.linspace(0.0, 1.0, 5)
    query_x = np.linspace(0.0, 1.0, BRANCH_RAW_STEPS)
    directions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    for candidate in range(1, CANDIDATE_COUNT):
        if candidate <= len(directions):
            knots = np.tile(directions[candidate - 1][None], (5, 1))
            knots += 0.15 * rng.normal(size=knots.shape).astype(np.float32)
        else:
            knots = rng.normal(size=(5, 3)).astype(np.float32)
        knots /= max(float(np.sqrt(np.mean(knots * knots))), 1.0e-6)
        noise = np.stack(
            [np.interp(query_x, knot_x, knots[:, axis]) for axis in range(3)], axis=-1
        )
        # Keep branches executable but large enough to create physical outcome
        # differences.  Gripper commands remain tied to the expert prefix.
        actions[candidate, :, :3] = np.clip(
            actions[candidate, :, :3] + float(amplitude) * noise, -1.0, 1.0
        )
    return np.ascontiguousarray(actions, dtype=np.float32)


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


def _run_branch(
    moving_env: MovingGoalEnv,
    actions: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_rows: list[np.ndarray] = []
    entity_rows: list[np.ndarray] = []
    goal_rows: list[np.ndarray] = []
    observation: np.ndarray | None = None
    for raw_step, action in enumerate(actions, start=1):
        observation, _reward, terminated, truncated, _info = moving_env.step(action)
        if terminated or truncated:
            raise RuntimeError(f"{task} branch terminated before {BRANCH_RAW_STEPS} steps")
        if raw_step % RAW_STEPS_PER_TOKEN == 0:
            xyz_rows.append(get_site_keypoints(moving_env.model, moving_env.data, DEFAULT_KEYPOINT_SITES).xyz.astype(np.float32))
            entity_rows.append(_entity(np.asarray(observation), task))
            goal_rows.append(np.asarray(moving_env._target_pos, dtype=np.float32).copy())
    if len(xyz_rows) != HORIZON:
        raise RuntimeError("branch did not produce eight future states")
    return np.stack(xyz_rows), np.stack(entity_rows), np.stack(goal_rows)


def main() -> int:
    args = _parser().parse_args()
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not tasks or any(task not in TASK_INDEX for task in tasks):
        raise ValueError(f"tasks must be drawn from {tuple(TASK_INDEX)}")
    if args.episodes < 2 or args.steps <= 0:
        raise ValueError("episodes must be >=2 and steps must be positive")
    amplitude = _parse_pair(args.amplitude_xy, "amplitude-xy")
    if not 0.0 <= float(args.goal_amplitude_jitter) < 1.0:
        raise ValueError("goal-amplitude-jitter must be in [0,1)")
    if not 0.0 <= float(args.goal_period_jitter) < 1.0:
        raise ValueError("goal-period-jitter must be in [0,1)")
    if not 0.0 < float(args.candidate_amplitude) <= 1.0:
        raise ValueError("candidate-amplitude must be in (0,1]")
    snapshot_steps = _parse_steps(args.snapshot_steps, max_step=args.steps)
    rng = np.random.default_rng(int(args.seed))
    camera_rng = np.random.default_rng(int(args.seed) + 1)
    outputs: dict[str, list[Any]] = {
        "current_rgb": [],
        "current_obs": [],
        "current_xyz": [],
        "current_q_true": [],
        "current_q_nominal": [],
        "current_valid_true": [],
        "current_valid_nominal": [],
        "current_entity_xyz": [],
        "current_goal": [],
        "current_goal_velocity": [],
        "history_action": [],
        "candidate_action": [],
        "target_xyz": [],
        "target_entity_xyz": [],
        "target_goal": [],
        "camera_intrinsic": [],
        "true_extrinsic": [],
        "task_name": [],
        "task_index": [],
        "episode_id": [],
        "snapshot_step": [],
        "split": [],
    }
    for task_offset, task in enumerate(tasks):
        for episode_id in range(int(args.episodes)):
            episode_rng = np.random.default_rng(
                int(args.seed) + task_offset * 1_000_000 + episode_id
            )
            amplitude_scale = episode_rng.uniform(
                1.0 - float(args.goal_amplitude_jitter),
                1.0 + float(args.goal_amplitude_jitter),
            )
            period_scale = episode_rng.uniform(
                1.0 - float(args.goal_period_jitter),
                1.0 + float(args.goal_period_jitter),
            )
            task_index = episode_id
            base_env = make_env(
                task,
                args.camera,
                int(args.width),
                int(args.height),
                task_index=task_index,
                benchmark_seed=int(args.task_generation_seed),
            )
            moving_env = MovingGoalEnv(
                base_env,
                task=task,
                seed=int(args.moving_goal_seed_base) + task_offset * 1_000_000 + episode_id,
                amplitude_xy=tuple(float(value) * amplitude_scale for value in amplitude),
                period_raw_steps=float(args.period_raw_steps) * period_scale,
            )
            policy = MovingPushExpert() if task == "push-v3" else make_scripted_policy(task)
            observation, _info = moving_env.reset(
                seed=int(args.env_seed_base) + task_offset * 1_000_000 + episode_id
            )
            camera_id = get_camera_id(moving_env.model, args.camera)
            intrinsic = camera_intrinsic_from_fovy(
                int(args.width), int(args.height), float(moving_env.model.cam_fovy[camera_id])
            )
            base_position = np.asarray(moving_env.model.cam_pos[camera_id], dtype=np.float64).copy()
            base_quaternion = np.asarray(moving_env.model.cam_quat[camera_id], dtype=np.float64).copy()
            base_extrinsic = camera_extrinsic_cv_from_mujoco(moving_env.data, camera_id)
            offset = sample_camera_pose_offset(
                camera_rng,
                0.0,
                float(args.camera_rotation_deg),
                float(args.camera_clean_probability),
            )
            set_fixed_camera_pose_offset(
                moving_env.model, moving_env.data, camera_id, base_position, base_quaternion, *offset
            )
            source_actions: list[np.ndarray] = []
            for raw_step in range(int(args.steps)):
                action = np.asarray(policy.get_action(observation), dtype=np.float32)
                if raw_step in snapshot_steps:
                    current_xyz = get_site_keypoints(moving_env.model, moving_env.data, DEFAULT_KEYPOINT_SITES).xyz.astype(np.float32)
                    current_goal = np.asarray(moving_env._target_pos, dtype=np.float32).copy()
                    next_goal = moving_env.controller.position(moving_env.raw_step + 1).astype(np.float32)
                    current_entity = _entity(np.asarray(observation), task)
                    # Keep the learning set in the active-control regime. A
                    # near-terminal snapshot has little action-conditioned
                    # information and would teach a trivial fixed point.
                    if float(np.linalg.norm(current_entity - current_goal)) < 0.04:
                        observation, _r, term, trunc, _i = moving_env.step(action)
                        source_actions.append(action)
                        if term or trunc:
                            break
                        continue
                    image = np.asarray(moving_env.render(), dtype=np.uint8)
                    true_extrinsic = camera_extrinsic_cv_from_mujoco(moving_env.data, camera_id)
                    true_camera = Camera(intrinsic, true_extrinsic, int(args.width), int(args.height))
                    nominal_camera = Camera(intrinsic, base_extrinsic, int(args.width), int(args.height))
                    q_true, valid_true = project_points(current_xyz, true_camera)
                    q_nominal, valid_nominal = project_points(current_xyz, nominal_camera)
                    snapshot = capture_metaworld_snapshot(base_env, include_process_rng=True)
                    moving_state = moving_env.state_dict()
                    expert_actions: list[np.ndarray] = []
                    branch_observation = np.asarray(observation, dtype=np.float32).copy()
                    for _ in range(BRANCH_RAW_STEPS):
                        branch_action = np.asarray(policy.get_action(branch_observation), dtype=np.float32)
                        expert_actions.append(np.clip(branch_action, -1.0, 1.0))
                        branch_observation, _r, term, trunc, _i = moving_env.step(expert_actions[-1])
                        if term or trunc:
                            break
                    if len(expert_actions) != BRANCH_RAW_STEPS:
                        _restore(base_env, moving_env, snapshot, moving_state, np.asarray(observation))
                        observation, _r, term, trunc, _i = moving_env.step(action)
                        source_actions.append(action)
                        continue
                    candidates = _candidate_actions(
                        np.stack(expert_actions),
                        rng,
                        amplitude=float(args.candidate_amplitude),
                    )
                    # Candidate slots carry no semantic identity. Randomize
                    # their order so a constant scorer cannot win by always
                    # choosing the expert's historical slot zero.
                    candidates = candidates[rng.permutation(CANDIDATE_COUNT)]
                    targets_xyz: list[np.ndarray] = []
                    targets_entity: list[np.ndarray] = []
                    targets_goal: list[np.ndarray] = []
                    for candidate in candidates:
                        _restore(base_env, moving_env, snapshot, moving_state, np.asarray(observation))
                        xyz, entity, goal = _run_branch(moving_env, candidate, task)
                        targets_xyz.append(xyz)
                        targets_entity.append(entity)
                        targets_goal.append(goal)
                    _restore(base_env, moving_env, snapshot, moving_state, np.asarray(observation))
                    outputs["current_rgb"].append(image)
                    outputs["current_obs"].append(np.asarray(observation, dtype=np.float32).copy())
                    outputs["current_xyz"].append(current_xyz)
                    outputs["current_q_true"].append(np.asarray(q_true, dtype=np.float32))
                    outputs["current_q_nominal"].append(np.asarray(q_nominal, dtype=np.float32))
                    outputs["current_valid_true"].append(np.asarray(valid_true, dtype=bool))
                    outputs["current_valid_nominal"].append(np.asarray(valid_nominal, dtype=bool))
                    outputs["current_entity_xyz"].append(current_entity)
                    outputs["current_goal"].append(current_goal)
                    outputs["current_goal_velocity"].append(next_goal - current_goal)
                    outputs["history_action"].append(_history(source_actions))
                    outputs["candidate_action"].append(candidates.reshape(CANDIDATE_COUNT, HORIZON, -1))
                    outputs["target_xyz"].append(np.stack(targets_xyz))
                    outputs["target_entity_xyz"].append(np.stack(targets_entity))
                    outputs["target_goal"].append(np.stack(targets_goal))
                    outputs["camera_intrinsic"].append(np.asarray(intrinsic, dtype=np.float32))
                    outputs["true_extrinsic"].append(np.asarray(true_extrinsic, dtype=np.float32))
                    outputs["task_name"].append(task)
                    outputs["task_index"].append(TASK_INDEX[task])
                    outputs["episode_id"].append(episode_id)
                    outputs["snapshot_step"].append(raw_step)
                    outputs["split"].append("train" if episode_id < args.episodes // 2 else "development")
                observation, _reward, terminated, truncated, _info = moving_env.step(action)
                source_actions.append(action)
                if terminated or truncated:
                    break
            moving_env.close()
        print(f"completed {task}", flush=True)
    if not outputs["current_rgb"]:
        raise RuntimeError("no branch snapshots were collected")
    path = args.output.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{key: np.asarray(value) for key, value in outputs.items()},
        protocol=np.asarray("moving_goal_action_branches_v1"),
        horizon=np.asarray(HORIZON),
        candidate_count=np.asarray(CANDIDATE_COUNT),
    )
    print(f"saved {len(outputs['current_rgb'])} snapshots to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
