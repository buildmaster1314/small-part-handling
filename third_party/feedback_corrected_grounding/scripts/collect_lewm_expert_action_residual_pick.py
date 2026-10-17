#!/usr/bin/env python3
"""Collect causal Pick features with same-state scripted-expert action labels."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_expert_action_residual import (  # noqa: E402
    ACTION_STEPS,
    ExpertResidualArrays,
    METHODS,
    PROTOCOL,
    save_expert_residual_arrays,
)
from fcpg.lewm_known_drop_policy_evaluator import (  # noqa: E402
    deployment_policy_batch,
    extract_policy_anchor_observation,
)
from fcpg.lewm_p1_native_complementary import (  # noqa: E402
    FrozenP1NativeCheckpointSpec,
    load_frozen_p1_native_checkpoint,
)
from fcpg.lewm_p1_native_policy_runtime import (  # noqa: E402
    P1NativeComplementaryPolicyRuntime,
)
from fcpg.metaworld_utils import make_env, make_scripted_policy  # noqa: E402
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
)
from scripts.evaluate_action_chunk_rollout import (  # noqa: E402
    choose_device,
    collect_observation,
    history_batch,
)
from scripts.evaluate_p1_native_policy_rollout import _corrupt_record  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _split_config(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    splits = config.get("splits")
    if not isinstance(splits, Mapping) or not isinstance(splits.get(name), Mapping):
        raise ValueError(f"unknown collection split {name!r}")
    return splits[name]


def _load_frozen_components(
    config: Mapping[str, object], device: torch.device
) -> tuple[torch.nn.Module, Mapping[str, object], object]:
    source = _json(str(config["source_fresh_gate_config"]))
    p1 = source.get("p1")
    if not isinstance(p1, Mapping):
        raise TypeError("source fresh config lacks P1 policy settings")
    policy, policy_args = load_action_chunk_policy(
        _resolve(str(p1["policy_checkpoint"])), device
    )
    policy.set_reliability_override("ones")
    policy.requires_grad_(False).eval()

    training = source.get("frozen_predictor_training_contract")
    if not isinstance(training, Mapping):
        raise TypeError("source fresh config lacks predictor training contract")
    expected_config = _json(str(training["config"]))
    records = source.get("frozen_predictors")
    if not isinstance(records, list):
        raise TypeError("source fresh config lacks frozen predictor records")
    seed = int(config["model_seed"])
    matches = [record for record in records if int(record["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"frozen predictor seed {seed} is unavailable or ambiguous")
    record = matches[0]
    pair = load_frozen_p1_native_checkpoint(
        FrozenP1NativeCheckpointSpec(
            seed=seed,
            path=_resolve(str(record["checkpoint"])),
            sha256=str(record["sha256"]),
        ),
        expected_config=expected_config,
        device=device,
    )
    return policy, policy_args, pair


def _runtime_panel(policy: torch.nn.Module, pair: object) -> dict[str, object]:
    return {
        "complementary_factual": P1NativeComplementaryPolicyRuntime(
            method="complementary_factual",
            policy=policy,
            factual_predictor=pair.factual,
            no_action_predictor=pair.no_action,
        ),
        "complementary_no_action": P1NativeComplementaryPolicyRuntime(
            method="complementary_no_action",
            policy=policy,
            factual_predictor=pair.factual,
            no_action_predictor=pair.no_action,
        ),
        "geometry_hold": P1NativeComplementaryPolicyRuntime(
            method="geometry_hold",
            policy=policy,
        ),
    }


def _expert_action(expert: object, env: object, observation: np.ndarray) -> np.ndarray:
    action = np.asarray(expert.get_action(observation), dtype=np.float32)
    if action.shape != (4,) or not np.isfinite(action).all():
        raise ValueError("scripted Pick expert returned an invalid action")
    return np.clip(action, env.action_space.low, env.action_space.high).astype(
        np.float32, copy=False
    )


@torch.inference_mode()
def _collect_episode(
    *,
    policy: torch.nn.Module,
    policy_args: Mapping[str, object],
    pair: object,
    episode_id: int,
    task: str,
    camera: str,
    width: int,
    height: int,
    benchmark_seed: int,
    task_index: int,
    environment_seed: int,
    drop_start: int,
    drop_duration: int,
    radius_px: int,
    value_uint8: int,
    occlusion_center_key: str,
    device: torch.device,
) -> dict[str, list[np.ndarray | int | bool]]:
    random.seed(environment_seed)
    np.random.seed(environment_seed)
    torch.manual_seed(environment_seed)
    env = make_env(
        task,
        camera,
        width,
        height,
        task_index=task_index,
        benchmark_seed=benchmark_seed,
    )
    expert = make_scripted_policy(task)
    rows: dict[str, list[np.ndarray | int | bool]] = {
        "episode_id": [],
        "decision_index": [],
        "missing_age": [],
        "sensor_valid": [],
        "state_latent": [],
        "action_latent": [],
        "base_action": [],
        "posterior_token": [],
        "expert_action": [],
    }
    try:
        observation, _ = env.reset(seed=environment_seed)
        observation = np.asarray(observation, dtype=np.float32)
        camera_id = get_camera_id(env.model, camera)
        intrinsic = camera_intrinsic_from_fovy(
            width, height, float(env.model.cam_fovy[camera_id])
        )
        nominal_extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
        obs_horizon = int(policy_args.get("obs_horizon", 2))
        if obs_horizon != 2:
            raise ValueError("expert residual collector requires policy history=2")
        clean_history: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)
        observed_history: deque[dict[str, np.ndarray]] = deque(maxlen=obs_horizon)
        initial = collect_observation(
            env,
            observation,
            camera_id,
            intrinsic,
            nominal_extrinsic,
            width,
            height,
            str(policy_args.get("obs_mode", "robot")),
            0,
            0,
        )
        clean_history.append(initial)
        observed_history.append({key: value.copy() for key, value in initial.items()})
        runtimes = _runtime_panel(policy, pair)

        for decision_index in range(drop_start + drop_duration):
            active = drop_start <= decision_index < drop_start + drop_duration
            observed_history[-1] = (
                _corrupt_record(
                    clean_history[-1],
                    radius_px=radius_px,
                    value_uint8=value_uint8,
                    center_key=occlusion_center_key,
                )
                if active
                else {key: value.copy() for key, value in clean_history[-1].items()}
            )
            observed_batch = deployment_policy_batch(
                history_batch(observed_history, obs_horizon, device)
            )

            method_features: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            if active:
                current_xyz = observed_batch["keypoints_xyz"][:, -1].float()
                ages: list[int] = []
                for method in METHODS:
                    decision = runtimes[method].begin_invalid(current_xyz)
                    output = policy.predict_action_chunk(decision.attach(observed_batch))
                    if output.latent.shape[1] < ACTION_STEPS:
                        raise RuntimeError("frozen policy action horizon is shorter than five")
                    posterior = decision.policy_token.detach().float()
                    method_features.append(
                        (
                            posterior.mean(dim=1).cpu().numpy()[0],
                            output.latent[:, :ACTION_STEPS].detach().float().cpu().numpy()[0],
                            output.action[:, :ACTION_STEPS].detach().float().cpu().numpy()[0],
                            posterior.cpu().numpy()[0],
                        )
                    )
                    ages.append(int(decision.missing_age))
                if len(set(ages)) != 1:
                    raise RuntimeError("matched observer methods disagree on missing age")
            else:
                native = extract_policy_anchor_observation(policy, observed_batch)
                for runtime in runtimes.values():
                    runtime.begin_valid(
                        native.native_token[:, -1].float(),
                        native.sampled_visual[:, -1].float(),
                    )

            expert_block: list[np.ndarray] = []
            for raw_step in range(ACTION_STEPS):
                action = _expert_action(expert, env, observation)
                expert_block.append(action.copy())
                observation, _reward, terminated, truncated, _info = env.step(action)
                observation = np.asarray(observation, dtype=np.float32)
                if terminated or truncated:
                    raise RuntimeError(
                        "Pick episode ended before the expert residual interval"
                    )
                clean_record = collect_observation(
                    env,
                    observation,
                    camera_id,
                    intrinsic,
                    nominal_extrinsic,
                    width,
                    height,
                    str(policy_args.get("obs_mode", "robot")),
                    0,
                    0,
                )
                clean_history.append(clean_record)
                observed_history.append(
                    _corrupt_record(
                        clean_record,
                        radius_px=radius_px,
                        value_uint8=value_uint8,
                        center_key=occlusion_center_key,
                    )
                    if active
                    else {key: value.copy() for key, value in clean_record.items()}
                )
            block = np.ascontiguousarray(np.stack(expert_block), dtype=np.float32)
            action_tensor = torch.from_numpy(block).unsqueeze(0).to(device)
            for runtime in runtimes.values():
                runtime.register_executed_action(action_tensor)

            if active:
                state, action_latent, base, posterior = zip(*method_features, strict=True)
                rows["episode_id"].append(int(episode_id))
                rows["decision_index"].append(int(decision_index))
                rows["missing_age"].append(int(ages[0]))
                rows["sensor_valid"].append(False)
                rows["state_latent"].append(np.stack(state))
                rows["action_latent"].append(np.stack(action_latent))
                rows["base_action"].append(np.stack(base))
                rows["posterior_token"].append(np.stack(posterior))
                rows["expert_action"].append(block)
    finally:
        env.close()
    return rows


def _merge(rows: list[dict[str, list[np.ndarray | int | bool]]]) -> ExpertResidualArrays:
    names = tuple(rows[0])
    flattened = {name: [value for row in rows for value in row[name]] for name in names}
    return ExpertResidualArrays(
        episode_id=np.asarray(flattened["episode_id"], dtype=np.int64),
        decision_index=np.asarray(flattened["decision_index"], dtype=np.int64),
        missing_age=np.asarray(flattened["missing_age"], dtype=np.int64),
        sensor_valid=np.asarray(flattened["sensor_valid"], dtype=bool),
        state_latent=np.asarray(flattened["state_latent"], dtype=np.float32),
        action_latent=np.asarray(flattened["action_latent"], dtype=np.float32),
        base_action=np.asarray(flattened["base_action"], dtype=np.float32),
        posterior_token=np.asarray(flattened["posterior_token"], dtype=np.float32),
        expert_action=np.asarray(flattened["expert_action"], dtype=np.float32),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/lewm/eval/expert_action_residual_pick_dev.json",
    )
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")
    config = _json(args.config)
    if config.get("schema_version") != 1 or config.get("protocol") != PROTOCOL:
        raise ValueError("expert residual collection config is incompatible")
    split = _split_config(config, args.split)
    total = int(split["episodes"])
    episodes = total if args.episodes is None else int(args.episodes)
    if not 0 < episodes <= total:
        parser.error(f"--episodes must lie in [1,{total}]")
    output = _resolve(args.output or str(split["output"]))
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")

    device = choose_device(args.device)
    policy, policy_args, pair = _load_frozen_components(config, device)
    rng = np.random.default_rng(int(split["schedule_seed"]))
    starts = rng.integers(
        int(config["drop_start_min_decision"]),
        int(config["drop_start_max_decision"]) + 1,
        size=episodes,
    )
    collected = []
    for episode in range(episodes):
        collected.append(
            _collect_episode(
                policy=policy,
                policy_args=policy_args,
                pair=pair,
                episode_id=int(split["environment_seed_base"]) + episode,
                task=str(config["task"]),
                camera=str(config["camera"]),
                width=int(config["width"]),
                height=int(config["height"]),
                benchmark_seed=int(split["benchmark_seed"]),
                task_index=int(split["task_index_base"]) + episode,
                environment_seed=int(split["environment_seed_base"]) + episode,
                drop_start=int(starts[episode]),
                drop_duration=int(config["drop_duration_decisions"]),
                radius_px=int(config["occlusion_radius_px"]),
                value_uint8=int(config["occlusion_value_uint8"]),
                occlusion_center_key=str(config["occlusion_center_key"]),
                device=device,
            )
        )
        print(
            f"[{args.split}] episode {episode + 1}/{episodes} "
            f"drop_start={int(starts[episode])}",
            flush=True,
        )
    arrays = _merge(collected)
    save_expert_residual_arrays(output, arrays)
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": episodes,
                "rows": arrays.sample_count,
                "methods": list(METHODS),
                "model_seed": int(config["model_seed"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
