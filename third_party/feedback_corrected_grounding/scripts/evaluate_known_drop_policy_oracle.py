#!/usr/bin/env python3
"""Known-drop policy-token oracle preflight for local anchor corruption.

This is deliberately a privileged diagnostic.  The fixed drop schedule routes
the methods and ``q_true`` centers the synthetic local occlusion; neither is a
deployable input.  The oracle restores only the policy's grounded anchor
tokens.  Its global image token still comes from the corrupted image, so the
experiment measures the policy-level headroom of recovering local grounding,
not full-image reconstruction.
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
    KNOWN_DROP_METHODS,
    attach_grounded_override,
    drop_is_active,
    grounded_tokens_from_memory,
    occlude_rgb_at_keypoints,
    repeated_last_valid_tokens,
)
from fcpg.lewm_known_drop_statistics import (  # noqa: E402
    summarize_known_drop_comparison,
)
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
    sample_camera_pose_offset,
    set_fixed_camera_pose_offset,
)
from fcpg.metaworld_utils import make_env  # noqa: E402
from scripts.evaluate_action_chunk_rollout import (  # noqa: E402
    choose_device,
    collect_observation,
    history_batch,
)


PROTOCOL = "known_drop_policy_token_oracle_preflight_v1"
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
def select_known_drop_action(
    model,
    clean_batch: dict[str, torch.Tensor],
    *,
    method: str,
    active: bool,
    last_valid_tokens: torch.Tensor | None,
    occlusion_radius_px: int,
    occlusion_value: float,
    device: torch.device,
) -> DecisionResult:
    """Return one method action and its hidden-clean same-state reference."""

    if method not in KNOWN_DROP_METHODS:
        raise ValueError(f"unknown known-drop method {method!r}")
    with _autocast(device):
        clean_memory, _, _ = model._observation_tokens(clean_batch)
        clean_tokens = grounded_tokens_from_memory(
            clean_memory,
            num_keypoints=model.num_keypoints,
        )
        clean_output = model.predict_action_chunk(clean_batch)

    clean_action = clean_output.action[:, :ACTION_STEPS]
    clean_current_tokens = clean_tokens[:, -1].detach()
    if not active or method == "clean":
        return DecisionResult(
            action=clean_action,
            clean_action=clean_action,
            clean_current_tokens=clean_current_tokens,
        )

    corrupted_batch = occlude_rgb_at_keypoints(
        clean_batch,
        radius_px=occlusion_radius_px,
        value=occlusion_value,
    )
    if method == "oracle":
        policy_batch = attach_grounded_override(corrupted_batch, clean_tokens)
    elif method == "hold_last":
        if last_valid_tokens is None:
            raise RuntimeError("hold_last requires a valid pre-drop token state")
        held = repeated_last_valid_tokens(
            last_valid_tokens,
            history_steps=int(clean_batch["obs"].shape[1]),
        )
        policy_batch = attach_grounded_override(corrupted_batch, held)
    else:
        policy_batch = corrupted_batch

    with _autocast(device):
        output = model.predict_action_chunk(policy_batch)
    return DecisionResult(
        action=output.action[:, :ACTION_STEPS],
        clean_action=clean_action,
        clean_current_tokens=clean_current_tokens,
    )


def _camera_diagnostics(
    camera_offset: tuple[np.ndarray, np.ndarray, float],
) -> dict[str, float]:
    translation, axis, rotation = camera_offset
    return {
        "actual_camera_translation_cm": float(np.linalg.norm(translation) * 100.0),
        "actual_camera_translation_x_cm": float(translation[0] * 100.0),
        "actual_camera_translation_y_cm": float(translation[1] * 100.0),
        "actual_camera_translation_z_cm": float(translation[2] * 100.0),
        "actual_camera_rotation_axis_x": float(axis[0]),
        "actual_camera_rotation_axis_y": float(axis[1]),
        "actual_camera_rotation_axis_z": float(axis[2]),
        "actual_camera_rotation_deg": float(rotation),
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
    """Run one deterministic method from a shared episode/camera seed."""

    if method not in KNOWN_DROP_METHODS:
        raise ValueError(f"unknown known-drop method {method!r}")
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
            width,
            height,
            float(env.model.cam_fovy[camera_id]),
        )
        base_position = np.asarray(
            env.model.cam_pos[camera_id], dtype=np.float64
        ).copy()
        base_quaternion = np.asarray(
            env.model.cam_quat[camera_id], dtype=np.float64
        ).copy()
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
        active_action_distances: list[float] = []
        active_decisions: list[int] = []
        decision_index = 0
        last_valid: torch.Tensor | None = None
        terminated = False
        truncated = False

        while len(rewards) < horizon and not (terminated or truncated):
            clean_batch = history_batch(history, obs_horizon, device)
            active = drop_is_active(
                decision_index,
                start=drop_start,
                duration=drop_duration,
            )
            decision = select_known_drop_action(
                model,
                clean_batch,
                method=method,
                active=active,
                last_valid_tokens=last_valid,
                occlusion_radius_px=occlusion_radius_px,
                occlusion_value=occlusion_value,
                device=device,
            )
            if active:
                distance = torch.mean(
                    (decision.action.float() - decision.clean_action.float()).square()
                )
                active_action_distances.append(float(distance.cpu()))
                active_decisions.append(decision_index)
            else:
                # Recovery is immediate: the first valid decision uses native
                # FCPG tokens and replaces the held state without blending.
                last_valid = decision.clean_current_tokens

            actions = decision.action[0].float().cpu().numpy()
            if actions.shape != (ACTION_STEPS, 4):
                raise ValueError(
                    "policy must emit at least five 4D actions for this protocol, "
                    f"got {actions.shape}"
                )
            for action in actions:
                action = np.clip(
                    action, env.action_space.low, env.action_space.high
                ).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                success = (
                    float(info.get("success", 0.0))
                    if isinstance(info, dict)
                    else 0.0
                )
                successes.append(success)
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
                if success > 0.0 or terminated or truncated or len(rewards) >= horizon:
                    break
            decision_index += 1

        if not active_action_distances:
            raise RuntimeError(
                "episode ended before the fixed known-drop interval was observed"
            )
        return {
            "method": method,
            "task": task_name,
            "task_index": task_index,
            "env_seed": env_seed,
            "steps": len(rewards),
            "decisions": decision_index,
            "success": float(max(successes, default=0.0)),
            "reward_sum": float(np.sum(rewards)),
            "action_distance": float(np.mean(active_action_distances)),
            "active_decisions_observed": len(active_action_distances),
            "first_active_decision": int(active_decisions[0]),
            "last_active_decision": int(active_decisions[-1]),
            **_camera_diagnostics(camera_offset),
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def _validate_cli(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.action_steps != ACTION_STEPS:
        parser.error(f"--action-steps must equal {ACTION_STEPS} for the frozen protocol")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.drop_start < 1:
        parser.error("--drop-start must be at least one to initialize hold-last")
    if args.drop_duration <= 0:
        parser.error("--drop-duration must be positive")
    if args.occlusion_radius_px <= 0:
        parser.error("--occlusion-radius-px must be positive")
    if not 0.0 <= args.occlusion_value <= 1.0:
        parser.error("--occlusion-value must lie in [0, 1]")
    if args.width <= 0 or args.height <= 0:
        parser.error("image dimensions must be positive")


def _checkpoint_seed(
    checkpoint_args: dict[str, object], override: int | None
) -> int:
    if override is not None:
        return int(override)
    value = checkpoint_args.get("seed", 0)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("checkpoint args do not contain an integer seed; use --model-seed")
    return int(value)


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
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--task-index-offset", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=337)
    parser.add_argument("--action-steps", type=int, default=ACTION_STEPS)
    parser.add_argument("--drop-start", type=int, default=1)
    parser.add_argument("--drop-duration", type=int, default=4)
    parser.add_argument("--occlusion-radius-px", type=int, default=24)
    parser.add_argument("--occlusion-value", type=float, default=0.0)
    parser.add_argument("--translation-cm-min", type=float, default=0.0)
    parser.add_argument("--translation-cm-max", type=float, default=0.0)
    parser.add_argument("--rotation-deg-min", type=float, default=0.0)
    parser.add_argument("--rotation-deg-max", type=float, default=0.0)
    parser.add_argument("--clean-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=15_701)
    parser.add_argument("--env-seed-base", type=int, default=31_000)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument(
        "--reliability-override",
        choices=("native", "ones"),
        default="ones",
        help=(
            "freeze policy grounding reliability; the Stage-I residual-only "
            "reference protocol uses ones"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="outputs/lewm_known_drop_policy/oracle_preflight",
    )
    args = parser.parse_args(argv)
    _validate_cli(parser, args)

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = choose_device(args.device)
    model, checkpoint_args = load_action_chunk_policy(args.checkpoint, device)
    model.set_reliability_override(args.reliability_override)
    model_seed = _checkpoint_seed(checkpoint_args, args.model_seed)
    checkpoint_sha256 = _sha256(args.checkpoint)

    rng = np.random.default_rng(args.seed)
    camera_offsets = [
        sample_camera_pose_offset(
            rng,
            args.translation_cm_max,
            args.rotation_deg_max,
            args.clean_probability,
            translation_cm_min=args.translation_cm_min,
            rotation_deg_min=args.rotation_deg_min,
        )
        for _ in range(args.episodes)
    ]
    rows: list[dict[str, object]] = []
    for episode, camera_offset in enumerate(camera_offsets):
        for method in KNOWN_DROP_METHODS:
            result = rollout_once(
                model,
                checkpoint_args,
                method=method,
                task_name=args.task,
                camera_name=args.camera,
                width=args.width,
                height=args.height,
                horizon=args.horizon,
                camera_offset=camera_offset,
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
                "drop_duration_raw_steps": args.drop_duration * ACTION_STEPS,
                **result,
            }
            rows.append(row)
            print(
                f"episode={episode:03d} method={method:11s} "
                f"success={row['success']:.1f} reward={row['reward_sum']:.2f} "
                f"action_distance={row['action_distance']:.6f}",
                flush=True,
            )

    comparisons = {
        "oracle_vs_measurement": summarize_known_drop_comparison(
            rows,
            candidate="oracle",
            baseline="measurement",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "oracle_vs_hold_last": summarize_known_drop_comparison(
            rows,
            candidate="oracle",
            baseline="hold_last",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 1,
        ),
        "oracle_vs_clean": summarize_known_drop_comparison(
            rows,
            candidate="oracle",
            baseline="clean",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 2,
        ),
        "measurement_vs_clean": summarize_known_drop_comparison(
            rows,
            candidate="measurement",
            baseline="clean",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 3,
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "known_drop_policy_oracle_long_table.csv"
    json_path = output_dir / "known_drop_policy_oracle_report.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": PROTOCOL,
        "status": "privileged_oracle_preflight_only",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "task": args.task,
        "camera": args.camera,
        "model_seed": model_seed,
        "reliability_override": args.reliability_override,
        "methods": list(KNOWN_DROP_METHODS),
        "episodes": args.episodes,
        "horizon_raw_steps": args.horizon,
        "action_steps": ACTION_STEPS,
        "drop_start_decision": args.drop_start,
        "drop_duration_decisions": args.drop_duration,
        "occlusion_radius_px": args.occlusion_radius_px,
        "occlusion_value": args.occlusion_value,
        "privileged_inputs": {
            "known_valid_mask": "fixed decision-level half-open schedule",
            "q_true": "centers synthetic local anchor occlusion only",
            "hidden_clean_tokens": "oracle policy-token upper bound only",
        },
        "deployment_claim_authorized": False,
        "global_token_note": (
            "measurement, hold_last, and oracle retain the corrupted global image "
            "token; oracle restores grounded anchor tokens only"
        ),
        "pairing_note": (
            "methods share task, task index, environment seed, camera offset, and "
            "drop schedule; clean reference actions are recomputed at each method's "
            "own current physical state"
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
