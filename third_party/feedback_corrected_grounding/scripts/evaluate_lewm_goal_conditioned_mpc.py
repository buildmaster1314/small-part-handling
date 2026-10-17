#!/usr/bin/env python3
"""Run goal-conditioned LeWM as a receding-horizon controller.

The scripted policy supplies only the current causal proposal. LeWM predicts
the consequences of a matched candidate bank, selects one prefix, executes its
first five raw actions, and then replans from a fresh visual observation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import html
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    make_a4_episode_runtime,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    capture_branch_snapshot,
    corrected_points,
    grounded_valid_mask,
    restore_branch_snapshot,
)
from fcpg.lewm_goal_conditioned_deployable import (  # noqa: E402
    load_deployable_checkpoint,
)
from fcpg.lewm_goal_conditioned_mpc import (  # noqa: E402
    CANDIDATE_NAMES,
    RAW_STEPS_PER_TOKEN,
    build_smooth_candidate_bank,
    causal_action_history,
    predicted_goal_cost,
    repeated_action_prefix,
    selected_raw_chunk,
)
from fcpg.lewm_goal_conditioned_proposer import (  # noqa: E402
    load_goal_conditioned_prefix_proposer,
)
from fcpg.lewm_history_conditioned_metric3d_a4 import (  # noqa: E402
    load_history_conditioned_checkpoint,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / (
    "outputs/lewm_goal_conditioned_deployable_seed93011_2k/"
    "goal_conditioned_deployable_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig002/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_PROPOSER_CHECKPOINT = ROOT / (
    "outputs/lewm_goal_conditioned_proposer_seed93031_2k/"
    "goal_conditioned_proposer_latest.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_goal_conditioned_mpc/clean_oracle_pilot"
SUPPORTED_TASKS = ("reach-v3", "push-v3")
CONTROLLERS = ("expert", "center", "lewm")
GROUNDINGS = ("nominal", "fcpg", "oracle")
PROPOSAL_MODES = ("repeat-current", "learned-prefix", "oracle-expert-rollout")
DEFAULT_COST_HORIZON = {"reach-v3": 3, "push-v3": 1}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT
    )
    parser.add_argument(
        "--proposal-checkpoint", type=Path, default=DEFAULT_PROPOSER_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS, default=SUPPORTED_TASKS)
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=CONTROLLERS)
    parser.add_argument("--grounding", choices=GROUNDINGS, default="oracle")
    parser.add_argument(
        "--proposal-mode",
        choices=PROPOSAL_MODES,
        default="repeat-current",
        help=(
            "Center-prefix source. oracle-expert-rollout branches the simulator "
            "for 40 steps and is diagnostic-only, not deployable."
        ),
    )
    parser.add_argument(
        "--proposal-entity-source",
        choices=("lewm", "oracle"),
        default="lewm",
        help="Entity input for entity-conditioned proposers; oracle is diagnostic-only.",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--task-instance-start", type=int, default=36)
    parser.add_argument("--environment-seed-base", type=int, default=190_000)
    parser.add_argument("--benchmark-seed", type=int, default=30_103)
    parser.add_argument("--candidate-seed", type=int, default=194_001)
    parser.add_argument("--candidate-amplitude", type=float, default=0.25)
    parser.add_argument("--reach-cost-horizon", type=int, default=3)
    parser.add_argument("--push-cost-horizon", type=int, default=1)
    parser.add_argument("--max-raw-steps", type=int, default=220)
    parser.add_argument(
        "--execute-raw-steps",
        type=int,
        default=RAW_STEPS_PER_TOKEN,
        help="Raw actions executed before the next visual replan (1..5).",
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--frame-duration-ms", type=int, default=300)
    return parser


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("goal-conditioned MPC permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _bounded_expert_action(runtime: Any, observation: np.ndarray) -> np.ndarray:
    return _bounded_policy_action(runtime.expert, runtime.env, observation)


def _bounded_policy_action(expert: Any, env: Any, observation: np.ndarray) -> np.ndarray:
    action = np.asarray(expert.get_action(observation), dtype=np.float32)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    if action.shape != (4,) or not np.isfinite(action).all():
        raise ValueError("expert must emit one finite 4D action")
    return np.clip(action, low, high).astype(np.float32, copy=False)


def _oracle_expert_rollout_prefix(runtime: Any, observation: np.ndarray) -> np.ndarray:
    """Build a training-matched expert prefix, then restore the source state.

    This deliberately reads simulated future observations and is therefore an
    interface diagnostic only. It must never be described as deployable MPC.
    """

    snapshot = capture_branch_snapshot(runtime.env, include_process_rng=True)
    branch_observation = np.asarray(observation, dtype=np.float32).copy()
    branch_expert = copy.deepcopy(runtime.expert)
    actions: list[np.ndarray] = []
    try:
        for _ in range(8 * RAW_STEPS_PER_TOKEN):
            action = _bounded_policy_action(
                branch_expert,
                runtime.env,
                branch_observation,
            )
            branch_observation, _reward, terminated, truncated, _info = (
                runtime.env.step(action)
            )
            actions.append(action.copy())
            if terminated or truncated:
                raise RuntimeError(
                    "oracle expert proposal ended before the 40-step horizon"
                )
            branch_observation = np.asarray(branch_observation, dtype=np.float32)
    finally:
        restore_branch_snapshot(runtime.env, snapshot)
    result = np.asarray(actions, dtype=np.float32)
    if result.shape != (8 * RAW_STEPS_PER_TOKEN, 4):
        raise RuntimeError("oracle expert proposal has an invalid shape")
    return np.ascontiguousarray(result)


def _center_prefix(
    runtime: Any,
    observation: np.ndarray,
    *,
    proposal_mode: str,
) -> np.ndarray:
    if proposal_mode == "repeat-current":
        return repeated_action_prefix(
            _bounded_expert_action(runtime, observation),
            action_low=runtime.env.action_space.low,
            action_high=runtime.env.action_space.high,
        )
    if proposal_mode == "oracle-expert-rollout":
        return _oracle_expert_rollout_prefix(runtime, observation)
    raise ValueError(f"unsupported proposal mode: {proposal_mode}")


@torch.inference_mode()
def _learned_prefix(
    proposer: Any,
    model: Any,
    record: Any,
    history_action: np.ndarray,
    goal: np.ndarray,
    goal_velocity: np.ndarray,
    task: str,
    *,
    q_grounded: np.ndarray,
    valid_grounded: np.ndarray,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    entity_source: str,
    oracle_entity_xyz: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    state = _encode_current_state(
        model,
        record,
        q_grounded,
        valid_grounded,
        device=device,
    )
    entity_estimate = None
    if bool(getattr(proposer, "use_entity_estimate", False)):
        if entity_source == "oracle":
            entity_estimate = torch.from_numpy(
                np.ascontiguousarray(oracle_entity_xyz)
            )[None].to(device)
        elif entity_source == "lewm":
            count = 1
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                entity_output = model(
                    current_state=state.float(),
                    history_action=torch.from_numpy(
                        np.ascontiguousarray(history_action)
                    )[None].to(device),
                    future_action=torch.zeros(
                        (count, 8, 20), dtype=torch.float32, device=device
                    ),
                    current_goal_xyz=torch.from_numpy(np.ascontiguousarray(goal))[None].to(device),
                    current_goal_velocity=torch.from_numpy(
                        np.ascontiguousarray(goal_velocity)
                    )[None].to(device),
                    task_index=torch.asarray(
                        [TASKS.index(task)], dtype=torch.long, device=device
                    ),
                    batch={
                        "current_keypoints_xyz": torch.from_numpy(
                            np.ascontiguousarray(record.keypoints_xyz)
                        )[None].to(device),
                        "camera_intrinsic": torch.from_numpy(
                            np.ascontiguousarray(camera_intrinsic)
                        )[None].to(device),
                        "camera_extrinsic_cam_from_world": torch.from_numpy(
                            np.ascontiguousarray(camera_extrinsic)
                        )[None].to(device),
                    },
                )
            entity_estimate = entity_output.predicted_entity_xyz[:, 0].float()
        else:
            raise ValueError(f"unsupported proposer entity source: {entity_source}")
    output = proposer(
        current_state=state.float(),
        history_action=torch.from_numpy(np.ascontiguousarray(history_action))[None].to(device),
        current_goal_xyz=torch.from_numpy(np.ascontiguousarray(goal))[None].to(device),
        current_goal_velocity=torch.from_numpy(np.ascontiguousarray(goal_velocity))[None].to(device),
        current_world_xyz=torch.from_numpy(
            np.ascontiguousarray(record.keypoints_xyz)
        )[None].to(device),
        task_index=torch.asarray([TASKS.index(task)], dtype=torch.long, device=device),
        current_entity_xyz=entity_estimate,
    )
    result = output.action_prefix[0].float().cpu().numpy().reshape(40, 4)
    if result.shape != (40, 4) or not np.isfinite(result).all():
        raise RuntimeError("learned proposal returned an invalid action prefix")
    return np.ascontiguousarray(result, dtype=np.float32)


def _success(info: Mapping[str, Any], env: Any) -> bool:
    value = info.get("success")
    if value is None:
        value = getattr(env, "success", False)
        value = value() if callable(value) else value
    return bool(float(value or 0.0) > 0.0)


def _entity_world(runtime: Any, record: Any, task: str) -> np.ndarray:
    if task == "reach-v3":
        result = np.asarray(record.keypoints_xyz[2], dtype=np.float32)
    else:
        result = np.asarray(runtime.env._get_pos_objects(), dtype=np.float32).reshape(-1)[:3]
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("controlled entity must be one finite world 3-vector")
    return result.copy()


def _goal_world(runtime: Any) -> np.ndarray:
    result = np.asarray(runtime.env._target_pos, dtype=np.float32).reshape(-1)[:3]
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("task goal must be one finite world 3-vector")
    return result.copy()


def _grounding(
    record: Any,
    *,
    mode: str,
    corrector: Any | None,
    device: torch.device,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if mode == "nominal":
        q = np.asarray(record.q_nominal, dtype=np.float32).copy()
        valid = np.asarray(record.valid_nominal, dtype=bool).copy()
    elif mode == "oracle":
        q = np.asarray(record.q_true, dtype=np.float32).copy()
        valid = np.asarray(record.valid_true, dtype=bool).copy()
    elif mode == "fcpg":
        if corrector is None:
            raise ValueError("FCPG grounding requires a corrector")
        q = corrected_points(
            [record], SimpleNamespace(corrector=corrector, device=device)
        )[0]
        valid = grounded_valid_mask(
            np.asarray(record.valid_nominal, dtype=bool)[None],
            q[None],
            width=int(width),
            height=int(height),
        )[0]
    else:
        raise ValueError(f"unsupported grounding mode: {mode}")
    support = valid & np.asarray(record.valid_true, dtype=bool)
    epe = (
        float(
            np.linalg.norm(
                q[support] - np.asarray(record.q_true, dtype=np.float32)[support],
                axis=-1,
            ).mean()
        )
        if np.any(support)
        else float("nan")
    )
    return np.ascontiguousarray(q), np.ascontiguousarray(valid), epe


@torch.inference_mode()
def _encode_current_state(
    model: Any,
    record: Any,
    q_grounded: np.ndarray,
    valid_grounded: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    rgb = torch.from_numpy(np.ascontiguousarray(record.rgb))[None, None].to(device)
    proprio = torch.from_numpy(np.ascontiguousarray(record.proprio))[None, None].to(device)
    q_nominal = torch.from_numpy(np.ascontiguousarray(record.q_nominal))[None, None].to(device)
    q_corrected = torch.from_numpy(np.ascontiguousarray(q_grounded))[None, None].to(device)
    xyz = torch.from_numpy(np.ascontiguousarray(record.keypoints_xyz))[None, None].to(device)
    valid = torch.from_numpy(np.ascontiguousarray(valid_grounded))[None, None].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        state = model.parent.encode_observations(
            rgb=rgb,
            proprio=proprio,
            q_nominal=q_nominal,
            q_corrected=q_corrected,
            keypoints_xyz=xyz,
            valid=valid,
        )[:, 0]
    if state.shape != (1, 4, 192) or not bool(torch.isfinite(state).all()):
        raise RuntimeError("current state encoder returned an invalid tensor")
    return state


@torch.inference_mode()
def _predict_candidate_costs(
    model: Any,
    record: Any,
    candidates: np.ndarray,
    history_action: np.ndarray,
    goal: np.ndarray,
    goal_velocity: np.ndarray,
    task: str,
    *,
    q_grounded: np.ndarray,
    valid_grounded: np.ndarray,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    cost_horizon: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = _encode_current_state(
        model,
        record,
        q_grounded,
        valid_grounded,
        device=device,
    )
    count = int(candidates.shape[0])
    state = state.expand(count, -1, -1)
    future = torch.from_numpy(np.ascontiguousarray(candidates)).to(device)
    history = torch.from_numpy(np.ascontiguousarray(history_action))[None].expand(
        count, -1, -1
    ).to(device)
    goal_t = torch.from_numpy(np.ascontiguousarray(goal))[None].expand(count, -1).to(device)
    velocity_t = torch.from_numpy(np.ascontiguousarray(goal_velocity))[None].expand(
        count, -1
    ).to(device)
    task_index = torch.full(
        (count,), TASKS.index(task), dtype=torch.long, device=device
    )
    xyz = torch.from_numpy(np.ascontiguousarray(record.keypoints_xyz))[None].expand(
        count, -1, -1
    ).to(device)
    intrinsic = torch.from_numpy(np.ascontiguousarray(camera_intrinsic))[None].expand(
        count, -1, -1
    ).to(device)
    extrinsic = torch.from_numpy(np.ascontiguousarray(camera_extrinsic))[None].expand(
        count, -1, -1
    ).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(
            current_state=state,
            history_action=history,
            future_action=future,
            current_goal_xyz=goal_t,
            current_goal_velocity=velocity_t,
            task_index=task_index,
            batch={
                "current_keypoints_xyz": xyz,
                "camera_intrinsic": intrinsic,
                "camera_extrinsic_cam_from_world": extrinsic,
            },
        )
    entity = output.predicted_entity_xyz.float().cpu().numpy()
    robot = output.predicted_world_xyz.float().cpu().numpy()
    cost = predicted_goal_cost(
        entity,
        goal,
        goal_velocity,
        horizon_tokens=int(cost_horizon),
    )
    return cost, entity, robot


def _frame(record: Any, lines: Sequence[str]) -> Image.Image:
    rgb = np.asarray(record.rgb, dtype=np.float32)
    image = Image.fromarray(
        np.clip(np.transpose(rgb, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8),
        mode="RGB",
    )
    draw = ImageDraw.Draw(image)
    box_height = 8 + 15 * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((5, 4 + 15 * index), str(line), fill=(255, 255, 255))
    return image


def _run_episode(
    model: Any,
    *,
    task: str,
    controller: str,
    grounding_mode: str,
    corrector: Any | None,
    proposer: Any | None,
    task_instance: int,
    environment_seed: int,
    benchmark_seed: int,
    candidate_seed: int,
    candidate_amplitude: float,
    proposal_mode: str,
    proposal_entity_source: str,
    cost_horizon: int,
    execute_raw_steps: int,
    max_raw_steps: int,
    width: int,
    height: int,
    device: torch.device,
    gif_path: Path,
    frame_duration_ms: int,
) -> dict[str, Any]:
    runtime = make_a4_episode_runtime(
        task,
        int(task_instance),
        int(environment_seed),
        width=int(width),
        height=int(height),
        benchmark_seed=int(benchmark_seed),
    )
    observation = np.asarray(runtime.observation, dtype=np.float32)
    info = dict(runtime.reset_info)
    executed: list[np.ndarray] = []
    decisions: list[dict[str, Any]] = []
    frames: list[Image.Image] = []
    reward_sum = 0.0
    raw_step = 0
    success_streak = 0
    stable_success_step: int | None = None
    previous_goal: np.ndarray | None = None
    terminated_reason = "max_raw_steps"
    try:
        while raw_step < min(int(max_raw_steps), int(runtime.max_raw_steps)):
            record = runtime.record_current(observation)
            goal = _goal_world(runtime)
            goal_velocity = (
                np.zeros(3, dtype=np.float32)
                if previous_goal is None
                else (goal - previous_goal) / float(RAW_STEPS_PER_TOKEN)
            )
            previous_goal = goal.copy()
            entity = _entity_world(runtime, record, task)
            distance_before = float(np.linalg.norm(entity - goal))
            choice = "expert"
            selected_index: int | None = None
            costs: np.ndarray | None = None
            grounding_epe = float("nan")

            if controller == "expert":
                action_chunk = np.empty((RAW_STEPS_PER_TOKEN, 4), dtype=np.float32)
            else:
                q_grounded: np.ndarray | None = None
                valid_grounded: np.ndarray | None = None
                if proposal_mode == "learned-prefix":
                    if proposer is None:
                        raise ValueError("learned-prefix mode requires a proposer")
                    q_grounded, valid_grounded, grounding_epe = _grounding(
                        record,
                        mode=grounding_mode,
                        corrector=corrector,
                        device=device,
                        width=width,
                        height=height,
                    )
                    center = _learned_prefix(
                        proposer,
                        model,
                        record,
                        causal_action_history(executed),
                        goal,
                        goal_velocity,
                        task,
                        q_grounded=q_grounded,
                        valid_grounded=valid_grounded,
                        camera_intrinsic=runtime.camera_intrinsic,
                        camera_extrinsic=runtime.camera_extrinsic_cam_from_world,
                        entity_source=proposal_entity_source,
                        oracle_entity_xyz=entity,
                        device=device,
                    )
                else:
                    center = _center_prefix(
                        runtime,
                        observation,
                        proposal_mode=proposal_mode,
                    )
                rng = np.random.default_rng(
                    int(candidate_seed)
                    + 1_000_003 * int(task_instance)
                    + 10_007 * int(raw_step // RAW_STEPS_PER_TOKEN)
                )
                candidates = build_smooth_candidate_bank(
                    center,
                    rng,
                    amplitude=float(candidate_amplitude),
                    action_low=runtime.env.action_space.low,
                    action_high=runtime.env.action_space.high,
                )
                if controller == "center":
                    selected_index = 0
                elif controller == "lewm":
                    if q_grounded is None or valid_grounded is None:
                        q_grounded, valid_grounded, grounding_epe = _grounding(
                            record,
                            mode=grounding_mode,
                            corrector=corrector,
                            device=device,
                            width=width,
                            height=height,
                        )
                    costs, _predicted_entity, _predicted_robot = _predict_candidate_costs(
                        model,
                        record,
                        candidates,
                        causal_action_history(executed),
                        goal,
                        goal_velocity,
                        task,
                        q_grounded=q_grounded,
                        valid_grounded=valid_grounded,
                        camera_intrinsic=runtime.camera_intrinsic,
                        camera_extrinsic=runtime.camera_extrinsic_cam_from_world,
                        cost_horizon=int(cost_horizon),
                        device=device,
                    )
                    selected_index = int(np.argmin(costs))
                else:
                    raise ValueError(f"unsupported controller: {controller}")
                choice = CANDIDATE_NAMES[selected_index]
                action_chunk = selected_raw_chunk(candidates, selected_index)

            lines = [
                f"{task} | {controller} | {grounding_mode}",
                f"step {raw_step} | choice {choice}",
                f"entity-goal {100.0 * distance_before:.2f} cm",
                (
                    f"predicted cost {100.0 * float(costs[selected_index]):.2f} cm"
                    if costs is not None and selected_index is not None
                    else "predicted cost n/a"
                ),
                f"ground EPE {grounding_epe:.2f} px",
            ]
            frames.append(_frame(record, lines))
            decisions.append(
                {
                    "raw_step": int(raw_step),
                    "choice": choice,
                    "selected_index": selected_index,
                    "entity_goal_distance_m": distance_before,
                    "predicted_cost_m": (
                        float(costs[selected_index])
                        if costs is not None and selected_index is not None
                        else None
                    ),
                    "all_predicted_cost_m": costs.astype(float).tolist() if costs is not None else None,
                    "grounding_epe_px": grounding_epe,
                    "cost_horizon_tokens": int(cost_horizon),
                    "proposal_mode": proposal_mode,
                }
            )

            terminated = truncated = False
            for chunk_step in range(int(execute_raw_steps)):
                action = (
                    _bounded_expert_action(runtime, observation)
                    if controller == "expert"
                    else action_chunk[chunk_step]
                )
                observation, reward, terminated, truncated, step_info = runtime.env.step(action)
                observation = np.asarray(observation, dtype=np.float32)
                info = dict(step_info) if isinstance(step_info, Mapping) else {}
                executed.append(np.asarray(action, dtype=np.float32).copy())
                reward_sum += float(reward)
                raw_step += 1
                if _success(info, runtime.env):
                    success_streak += 1
                    if success_streak >= 3:
                        stable_success_step = raw_step
                        terminated_reason = "stable_success"
                        break
                else:
                    success_streak = 0
                if terminated or truncated:
                    terminated_reason = "terminated" if terminated else "truncated"
                    break
                if raw_step >= min(int(max_raw_steps), int(runtime.max_raw_steps)):
                    break
            if stable_success_step is not None or terminated or truncated:
                break

        final_record = runtime.record_current(observation)
        final_distance = float(
            np.linalg.norm(_entity_world(runtime, final_record, task) - _goal_world(runtime))
        )
        final_success = bool(_success(info, runtime.env))
    finally:
        runtime.env.close()

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(frame_duration_ms),
            loop=0,
        )
    counts = Counter(row["choice"] for row in decisions)
    return {
        "task": task,
        "controller": controller,
        "grounding": grounding_mode,
        "proposal_mode": proposal_mode,
        "proposal_is_deployable": (
            proposal_mode != "oracle-expert-rollout"
            and proposal_entity_source != "oracle"
        ),
        "task_instance": int(task_instance),
        "environment_seed": int(environment_seed),
        "success": final_success,
        "stable_success_step": stable_success_step,
        "raw_steps": int(raw_step),
        "terminated_reason": terminated_reason,
        "cumulative_reward": float(reward_sum),
        "final_entity_goal_distance_m": final_distance,
        "decisions": decisions,
        "choice_counts": dict(sorted(counts.items())),
        "gif": gif_path.name,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in SUPPORTED_TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        result[task] = {}
        for controller in CONTROLLERS:
            selected = [row for row in task_rows if row["controller"] == controller]
            if not selected:
                continue
            result[task][controller] = {
                "episodes": len(selected),
                "success_rate": float(np.mean([bool(row["success"]) for row in selected])),
                "mean_final_goal_distance_m": float(
                    np.mean([float(row["final_entity_goal_distance_m"]) for row in selected])
                ),
                "mean_cumulative_reward": float(
                    np.mean([float(row["cumulative_reward"]) for row in selected])
                ),
                "mean_raw_steps": float(np.mean([int(row["raw_steps"]) for row in selected])),
            }
    return result


def _write_html(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    cards = []
    for row in rows:
        title = (
            f"{row['task']} | {row['controller']} | ep {row['task_instance']} | "
            f"success={int(bool(row['success']))} | final="
            f"{100.0 * float(row['final_entity_goal_distance_m']):.2f} cm"
        )
        cards.append(
            "<article><h2>{}</h2><img src=\"{}\" alt=\"{}\"></article>".format(
                html.escape(title),
                html.escape(str(row["gif"])),
                html.escape(title),
            )
        )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>LeWM MPC</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:20px}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}"
        "article{background:#1d1d1d;padding:12px;border:1px solid #444}h2{font-size:15px}"
        "img{width:100%;image-rendering:auto}</style></head><body>"
        "<h1>Goal-conditioned LeWM receding-horizon control</h1><main>"
        + "".join(cards)
        + "</main></body></html>\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parser().parse_args()
    if int(args.episodes) <= 0 or int(args.max_raw_steps) <= 0:
        raise ValueError("episodes and max-raw-steps must be positive")
    if not 1 <= int(args.execute_raw_steps) <= RAW_STEPS_PER_TOKEN:
        raise ValueError("execute-raw-steps must be in 1..5")
    cost_horizon = {
        "reach-v3": int(args.reach_cost_horizon),
        "push-v3": int(args.push_cost_horizon),
    }
    if any(not 1 <= value <= 8 for value in cost_horizon.values()):
        raise ValueError("cost horizons must be in 1..8")
    if int(args.task_instance_start) < 0 or int(args.task_instance_start) + int(args.episodes) > 48:
        raise ValueError("requested task instances must remain within 0..47")
    device = _cuda0(args.device)
    torch.cuda.set_device(device)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parent_path = Path(str(checkpoint["parent_checkpoint"])).expanduser().resolve()
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent = load_history_conditioned_checkpoint(parent_payload).to(device).eval()
    model = load_deployable_checkpoint(checkpoint, parent).to(device).eval()
    corrector = None
    if args.grounding == "fcpg":
        corrector, _ = load_fcpg_vit_adapter(
            args.fcpg_checkpoint.expanduser().resolve(), device=device
        )
        corrector.eval()
    proposer = None
    proposer_path: Path | None = None
    if args.proposal_mode == "learned-prefix":
        proposer_path = args.proposal_checkpoint.expanduser().resolve()
        proposer_payload = torch.load(
            proposer_path, map_location="cpu", weights_only=False
        )
        proposer = load_goal_conditioned_prefix_proposer(proposer_payload).to(device).eval()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task_offset, task in enumerate(args.tasks):
        for episode in range(int(args.episodes)):
            task_instance = int(args.task_instance_start) + episode
            environment_seed = (
                int(args.environment_seed_base) + 100_000 * task_offset + episode
            )
            for controller in args.controllers:
                gif_path = output_dir / f"{task}_ep{episode:02d}_{controller}.gif"
                row = _run_episode(
                    model,
                    task=task,
                    controller=controller,
                    grounding_mode=args.grounding,
                    corrector=corrector,
                    proposer=proposer,
                    task_instance=task_instance,
                    environment_seed=environment_seed,
                    benchmark_seed=int(args.benchmark_seed),
                    candidate_seed=int(args.candidate_seed),
                    candidate_amplitude=float(args.candidate_amplitude),
                    proposal_mode=args.proposal_mode,
                    proposal_entity_source=args.proposal_entity_source,
                    cost_horizon=cost_horizon[task],
                    execute_raw_steps=int(args.execute_raw_steps),
                    max_raw_steps=int(args.max_raw_steps),
                    width=int(args.width),
                    height=int(args.height),
                    device=device,
                    gif_path=gif_path,
                    frame_duration_ms=int(args.frame_duration_ms),
                )
                rows.append(row)
                print(json.dumps({key: value for key, value in row.items() if key != "decisions"}), flush=True)

    report = {
        "protocol": "goal_conditioned_lewm_receding_horizon_control_v1",
        "checkpoint": str(checkpoint_path),
        "grounding": args.grounding,
        "proposal_mode": args.proposal_mode,
        "proposal_is_deployable": (
            args.proposal_mode != "oracle-expert-rollout"
            and args.proposal_entity_source != "oracle"
        ),
        "proposal_entity_source": args.proposal_entity_source,
        "proposal_checkpoint": str(proposer_path) if proposer_path is not None else None,
        "tasks": list(args.tasks),
        "controllers": list(args.controllers),
        "episodes_per_task": int(args.episodes),
        "candidate_count": len(CANDIDATE_NAMES),
        "model_horizon_tokens": 8,
        "raw_steps_per_token": RAW_STEPS_PER_TOKEN,
        "executed_raw_steps_per_replan": int(args.execute_raw_steps),
        "cost_horizon_by_task": cost_horizon,
        "executes_only_first_token_then_reobserves": True,
        "scripted_policy_role": (
            "current_causal_proposal_only"
            if args.proposal_mode == "repeat-current"
            else "simulated_future_proposal_diagnostic_only"
        ),
        "lewm_selects_executed_candidate": True,
        "summary": _summary(rows),
        "episodes": rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_html(output_dir / "index.html", rows)
    print(json.dumps(report["summary"], indent=2))
    print(f"HTML: {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
