#!/usr/bin/env python3
"""Privileged two-slot policy-clock diagnostic on Pick P1.

At active local RGB drops, every method receives the same corrupted RGB/global
context and differs only in the two grounded-token history slots:

* exact_adjacent: hidden-clean raw-adjacent tokens ``[t-1, t]``;
* decision_spaced: hidden-clean controller-spaced tokens ``[t-5, t]``;
* repeat_current: hidden-clean current token ``[t, t]``;
* hold_last: final pre-drop token repeated in both slots.

This is an interface-clock upper-bound diagnostic.  It uses hidden clean tokens
and the known drop schedule, trains nothing, and authorizes no deployment claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_known_drop_policy import (  # noqa: E402
    drop_is_active,
    grounded_tokens_from_memory,
    occlude_rgb_at_keypoints,
)
from fcpg.lewm_known_drop_statistics import (  # noqa: E402
    summarize_known_drop_comparison,
)
from fcpg.metaworld_utils import make_env  # noqa: E402
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    sample_camera_pose_offset,
    set_fixed_camera_pose_offset,
)
from fcpg.policy_slot_spacing_diagnostic import (  # noqa: E402
    SLOT_SPACING_METHODS,
    SlotSpacingState,
    attach_slot_spacing_override,
    update_slot_spacing_state,
)
from scripts.evaluate_action_chunk_rollout import (  # noqa: E402
    choose_device,
    collect_observation,
    history_batch,
)


PROTOCOL = "privileged_policy_slot_spacing_diagnostic_v1"
ACTION_STEPS = 5


class DecisionResult(NamedTuple):
    action: torch.Tensor
    clean_action: torch.Tensor
    clean_current_tokens: torch.Tensor


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    )


@torch.inference_mode()
def select_slot_spacing_action(
    model,
    clean_batch: dict[str, torch.Tensor],
    *,
    method: str,
    active: bool,
    state: SlotSpacingState,
    occlusion_radius_px: int,
    occlusion_value: float,
    device: torch.device,
) -> DecisionResult:
    """Return one slot-spacing action and exact-clean same-state reference."""

    if method not in SLOT_SPACING_METHODS:
        raise ValueError(f"unknown slot-spacing method {method!r}")
    with _autocast(device):
        clean_memory, _, _ = model._observation_tokens(clean_batch)
        clean_tokens = grounded_tokens_from_memory(
            clean_memory,
            num_keypoints=model.num_keypoints,
        )
        clean_output = model.predict_action_chunk(clean_batch)
    clean_action = clean_output.action[:, :ACTION_STEPS]
    clean_current = clean_tokens[:, -1].detach()
    if not active:
        return DecisionResult(clean_action, clean_action, clean_current)

    corrupted_batch = occlude_rgb_at_keypoints(
        clean_batch,
        radius_px=occlusion_radius_px,
        value=occlusion_value,
    )
    policy_batch = attach_slot_spacing_override(
        corrupted_batch,
        clean_tokens,
        method=method,
        state=state,
    )
    with _autocast(device):
        output = model.predict_action_chunk(policy_batch)
    return DecisionResult(
        output.action[:, :ACTION_STEPS],
        clean_action,
        clean_current,
    )


def _camera_diagnostics(
    camera_offset: tuple[np.ndarray, np.ndarray, float],
) -> dict[str, float]:
    translation, axis, rotation = camera_offset
    return {
        "actual_camera_translation_cm": float(np.linalg.norm(translation) * 100.0),
        "actual_camera_rotation_deg": float(rotation),
        "actual_camera_rotation_axis_x": float(axis[0]),
        "actual_camera_rotation_axis_y": float(axis[1]),
        "actual_camera_rotation_axis_z": float(axis[2]),
    }


def rollout_once(
    model,
    checkpoint_args: dict[str, object],
    *,
    method: str,
    task_name: str,
    camera_name: str,
    width: int,
    height: int,
    horizon: int,
    camera_offset: tuple[np.ndarray, np.ndarray, float],
    device: torch.device,
    task_index: int,
    env_seed: int,
    drop_start: int,
    drop_duration: int,
    occlusion_radius_px: int,
    occlusion_value: float,
) -> dict[str, object]:
    """Run one paired method trajectory with an identical seed and schedule."""

    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)
    env = make_env(
        task_name,
        camera_name,
        width,
        height,
        task_index=task_index,
        benchmark_seed=env_seed,
    )
    try:
        obs, _ = env.reset(seed=env_seed)
        camera_id = get_camera_id(env.model, camera_name)
        intrinsic = camera_intrinsic_from_fovy(
            width, height, float(env.model.cam_fovy[camera_id])
        )
        base_position = np.asarray(env.model.cam_pos[camera_id], dtype=np.float64).copy()
        base_quaternion = np.asarray(env.model.cam_quat[camera_id], dtype=np.float64).copy()
        nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
        set_fixed_camera_pose_offset(
            env.model,
            env.data,
            camera_id,
            base_position,
            base_quaternion,
            *camera_offset,
        )

        obs_mode = str(checkpoint_args.get("obs_mode", "robot"))
        obs_horizon = int(checkpoint_args.get("obs_horizon", 2))
        if obs_horizon != 2:
            raise ValueError("slot-spacing diagnostic requires policy obs_horizon=2")
        history: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)
        history.append(
            collect_observation(
                env,
                obs,
                camera_id,
                intrinsic,
                nominal_extrinsic,
                width,
                height,
                obs_mode,
                0,
                0,
            )
        )
        rewards: list[float] = []
        successes: list[float] = []
        action_distances: list[float] = []
        active_decisions: list[int] = []
        decision_index = 0
        slot_state = SlotSpacingState()
        terminated = False
        truncated = False

        while len(rewards) < horizon and not (terminated or truncated):
            clean_batch = history_batch(history, obs_horizon, device)
            active = drop_is_active(
                decision_index,
                start=drop_start,
                duration=drop_duration,
            )
            decision = select_slot_spacing_action(
                model,
                clean_batch,
                method=method,
                active=active,
                state=slot_state,
                occlusion_radius_px=occlusion_radius_px,
                occlusion_value=occlusion_value,
                device=device,
            )
            if active:
                action_distances.append(
                    float(
                        torch.mean(
                            (decision.action.float() - decision.clean_action.float()).square()
                        ).cpu()
                    )
                )
                active_decisions.append(decision_index)
            slot_state = update_slot_spacing_state(
                slot_state,
                decision.clean_current_tokens,
                active=active,
            )

            actions = decision.action[0].float().cpu().numpy()
            if actions.shape != (ACTION_STEPS, 4):
                raise ValueError(
                    "policy must emit exactly five selected 4D actions, got "
                    f"{actions.shape}"
                )
            for action in actions:
                clipped = np.clip(
                    action, env.action_space.low, env.action_space.high
                ).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(clipped)
                rewards.append(float(reward))
                successes.append(
                    float(info.get("success", 0.0)) if isinstance(info, dict) else 0.0
                )
                history.append(
                    collect_observation(
                        env,
                        obs,
                        camera_id,
                        intrinsic,
                        nominal_extrinsic,
                        width,
                        height,
                        obs_mode,
                        0,
                        0,
                    )
                )
                if successes[-1] > 0.0 or terminated or truncated or len(rewards) >= horizon:
                    break
            decision_index += 1

        if not action_distances:
            raise RuntimeError("episode ended before the slot-spacing drop interval")
        return {
            "method": method,
            "task": task_name,
            "task_index": task_index,
            "env_seed": env_seed,
            "steps": len(rewards),
            "decisions": decision_index,
            "success": float(max(successes, default=0.0)),
            "reward_sum": float(np.sum(rewards)),
            "action_distance": float(np.mean(action_distances)),
            "active_decisions_observed": len(action_distances),
            "first_active_decision": int(active_decisions[0]),
            "last_active_decision": int(active_decisions[-1]),
            **_camera_diagnostics(camera_offset),
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="pick-place-v3")
    parser.add_argument("--camera", default="corner")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--task-index-offset", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=337)
    parser.add_argument("--drop-start", type=int, default=1)
    parser.add_argument("--drop-duration", type=int, default=4)
    parser.add_argument("--occlusion-radius-px", type=int, default=24)
    parser.add_argument("--occlusion-value", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=15_719)
    parser.add_argument("--env-seed-base", type=int, default=37_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="outputs/lewm_known_drop_policy/slot_spacing_diagnostic",
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.horizon <= 0:
        parser.error("episodes and horizon must be positive")
    if args.drop_start < 1:
        parser.error("drop-start must initialize previous-decision and hold-last state")
    if args.drop_duration <= 0:
        parser.error("drop-duration must be positive")
    if args.occlusion_radius_px <= 0:
        parser.error("occlusion-radius-px must be positive")
    if not 0.0 <= args.occlusion_value <= 1.0:
        parser.error("occlusion-value must lie in [0,1]")

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = choose_device(args.device)
    model, checkpoint_args = load_action_chunk_policy(args.checkpoint, device)
    model.set_reliability_override("ones")
    model_seed = int(checkpoint_args.get("seed", 0))
    checkpoint_sha256 = _sha256(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    offsets = [
        sample_camera_pose_offset(rng, 0.0, 0.0, 0.0)
        for _ in range(args.episodes)
    ]

    rows: list[dict[str, object]] = []
    for episode, offset in enumerate(offsets):
        for method in SLOT_SPACING_METHODS:
            result = rollout_once(
                model,
                checkpoint_args,
                method=method,
                task_name=args.task,
                camera_name=args.camera,
                width=args.width,
                height=args.height,
                horizon=args.horizon,
                camera_offset=offset,
                device=device,
                task_index=args.task_index_offset + episode,
                env_seed=args.env_seed_base + episode,
                drop_start=args.drop_start,
                drop_duration=args.drop_duration,
                occlusion_radius_px=args.occlusion_radius_px,
                occlusion_value=args.occlusion_value,
            )
            row = {
                "physical_episode": episode,
                "model_seed": model_seed,
                "drop_duration": float(args.drop_duration),
                "drop_duration_unit": "policy_decisions",
                **result,
            }
            rows.append(row)
            print(
                f"episode={episode:03d} method={method:16s} "
                f"success={row['success']:.1f} action_distance={row['action_distance']:.6f}",
                flush=True,
            )

    comparisons = {}
    for baseline_index, baseline in enumerate(SLOT_SPACING_METHODS[1:]):
        comparisons[f"exact_adjacent_vs_{baseline}"] = summarize_known_drop_comparison(
            rows,
            candidate="exact_adjacent",
            baseline=baseline,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + baseline_index,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "policy_slot_spacing_long_table.csv"
    json_path = output_dir / "policy_slot_spacing_report.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": PROTOCOL,
        "status": "privileged_interface_clock_diagnostic_only",
        "deployment_claim_authorized": False,
        "lewm_used": False,
        "training_performed": False,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "task": args.task,
        "episodes": args.episodes,
        "action_steps": ACTION_STEPS,
        "methods": list(SLOT_SPACING_METHODS),
        "primary_metric": "active-decision action MSE to exact clean policy",
        "success_role": "trajectory-level diagnostic only",
        "slot_semantics": {
            "exact_adjacent": "hidden-clean raw-adjacent [t-1,t]",
            "decision_spaced": "hidden-clean previous decision/current [t-5,t]",
            "repeat_current": "hidden-clean current repeated [t,t]",
            "hold_last": "last pre-drop current repeated and frozen",
        },
        "pairing_note": (
            "methods share task, task index, environment seed, clean camera, and drop "
            "schedule; each method executes its own actions, while its clean reference is "
            "recomputed at that method's same physical state"
        ),
        "global_context_note": (
            "every active method uses the locally corrupted RGB/global token and differs "
            "only in grounded-token overrides"
        ),
        "comparisons": comparisons,
        "rows": rows,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"csv: {csv_path}")
    print(f"report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
