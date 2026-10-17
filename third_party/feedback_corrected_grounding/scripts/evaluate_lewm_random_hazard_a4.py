#!/usr/bin/env python3
"""Evaluate the formal seven-task A4b LeWM under random visual interruptions."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_checkpoint import validate_a4_checkpoint  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    CONFIRMATION_TRAINING_SEEDS,
    HISTORY_STEPS,
    RAW_STEPS_PER_MODEL_STEP,
    TASKS,
    assign_a4_splits,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    a4_camera_name,
    load_frozen_a4_state_encoder,
)
from fcpg.lewm_action_branch_a4_training import (  # noqa: E402
    build_matched_a4_models,
)
from fcpg.lewm_random_hazard_a4 import (  # noqa: E402
    FACTUAL_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CV_METHOD,
    METHODS,
    WRONG_ACTION_METHOD,
    ZERO_ACTION_METHOD,
    latent_constant_velocity,
    sample_random_hazard_schedule,
    summarize_random_hazard_records,
)
from scripts import render_lewm_action_branch_a4_accuracy_gallery as legacy  # noqa: E402
from scripts import render_lewm_action_branch_a4_trajectory_comparison as trajectory  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_BASE_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_CHECKPOINTS = tuple(
    ROOT
    / f"outputs/lewm_action_branch_a4/shared_7task_seed{seed}_10k/"
    "lewm_action_branch_a4_latest.pt"
    for seed in CONFIRMATION_TRAINING_SEEDS
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_branch_a4/"
    "random_hazard_a4b_7task_smoke_seed42601"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        dest="checkpoints",
        help="Formal A4b 10k checkpoint; repeat for an ensemble",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42_601)
    parser.add_argument("--hazard-probability", type=float, default=0.12)
    parser.add_argument("--minimum-duration", type=int, default=2)
    parser.add_argument("--maximum-duration", type=int, default=8)
    parser.add_argument("--maximum-events", type=int, default=3)
    parser.add_argument("--max-raw-steps", type=int, default=500)
    parser.add_argument("--post-success-steps", type=int, default=8)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.device not in {"cuda", "cuda:0"}:
        raise ValueError("random-hazard A4b evaluation permits only CUDA device 0")
    if not 0.0 <= float(args.hazard_probability) <= 1.0:
        raise ValueError("hazard probability must be in [0,1]")
    if not 1 <= int(args.minimum_duration) <= int(args.maximum_duration) <= 8:
        raise ValueError("hazard duration must satisfy 1 <= min <= max <= 8")
    if not 1 <= int(args.episodes_per_task) <= 12:
        raise ValueError("episodes-per-task must be in 1..12")


def _torch_load(path: Path) -> dict[str, Any]:
    value = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one checkpoint mapping")
    return value


def _decision_steps(action_count: int) -> np.ndarray:
    complete = int(action_count) // RAW_STEPS_PER_MODEL_STEP
    if complete < HISTORY_STEPS + 1:
        raise ValueError("episode is too short for A4b random-hazard evaluation")
    return np.arange(complete + 1, dtype=np.int64) * RAW_STEPS_PER_MODEL_STEP


def _development_instances(
    contract: Mapping[str, Any], *, episodes_per_task: int
) -> dict[str, tuple[int, ...]]:
    assignment = assign_a4_splits(TASKS, contract)
    result: dict[str, tuple[int, ...]] = {}
    for task in TASKS:
        candidates = tuple(
            sorted(
                instance
                for (name, instance), split in assignment.items()
                if name == task and split == "development"
            )
        )
        selected = candidates[: int(episodes_per_task)]
        if len(selected) != int(episodes_per_task):
            raise RuntimeError(f"A4 has too few development instances for {task}")
        result[task] = selected
    return result


def _chunk_actions(actions: np.ndarray, transition_indices: Sequence[int]) -> np.ndarray:
    raw = np.asarray(actions, dtype=np.float32)
    chunks = []
    for transition_index in transition_indices:
        start = int(transition_index) * RAW_STEPS_PER_MODEL_STEP
        stop = start + RAW_STEPS_PER_MODEL_STEP
        chunk = raw[start:stop]
        if chunk.shape != (RAW_STEPS_PER_MODEL_STEP, 4):
            raise ValueError("transition action chunk is incomplete")
        chunks.append(chunk.reshape(-1))
    result = np.asarray(chunks, dtype=np.float32)
    expected = (len(chunks), RAW_STEPS_PER_MODEL_STEP * 4)
    if result.shape != expected:
        raise RuntimeError(f"chunked action shape drifted: {result.shape} != {expected}")
    return np.ascontiguousarray(result)


def _future_actions_through_age(
    actions: np.ndarray,
    *,
    first_transition: int,
    event_age: int,
) -> np.ndarray:
    """Return only the actions executed since the last valid measurement."""

    age = int(event_age)
    if not 1 <= age <= 8:
        raise ValueError("event age must be in 1..8 for the formal A4b model")
    chunks = _chunk_actions(
        actions,
        tuple(int(first_transition) + offset for offset in range(age)),
    )
    return chunks[None, None]


def _wrong_future_actions(actions: np.ndarray) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32).copy().reshape(
        *np.asarray(actions).shape[:-1], RAW_STEPS_PER_MODEL_STEP, 4
    )
    result[..., :3] *= -1.0
    return np.ascontiguousarray(result.reshape(np.asarray(actions).shape))


@torch.inference_mode()
def _predict_at_event_age(
    model: Any,
    history_embedding: np.ndarray,
    history_action: np.ndarray,
    future_action: np.ndarray,
    *,
    task_index: int,
    device: torch.device,
) -> np.ndarray:
    history = torch.from_numpy(
        np.asarray(history_embedding, dtype=np.float32)
    )[None].to(device)
    history_actions = torch.from_numpy(
        np.asarray(history_action, dtype=np.float32)
    )[None].to(device)
    future_actions = torch.from_numpy(
        np.asarray(future_action, dtype=np.float32)
    ).to(device)
    task = torch.tensor([int(task_index)], dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model.rollout_state_embeddings(
            history,
            history_actions,
            future_actions,
            task,
        )
    expected = (1, 1, int(future_actions.shape[2]), 4, 192)
    if tuple(prediction.shape) != expected:
        raise RuntimeError(
            f"formal A4b prediction shape drifted: {tuple(prediction.shape)} != {expected}"
        )
    result = prediction[0, 0, -1].float().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("formal A4b prediction contains non-finite values")
    return result


def _evaluate_episode_for_models(
    factual: Any,
    no_action: Any,
    episode: Mapping[str, Any],
    *,
    task_index: int,
    checkpoint_seed: int,
    schedule_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions = np.asarray(episode["actions"], dtype=np.float32)
    starts = np.asarray(episode["model_steps"], dtype=np.int64)
    measurement = np.asarray(episode["encoded_state"], dtype=np.float32)
    if measurement.shape != (len(starts), 4, 192):
        raise ValueError("formal A4b encoded state must have shape [T,4,192]")
    schedule = sample_random_hazard_schedule(
        len(starts),
        seed=int(schedule_seed),
        trigger_probability=float(args.hazard_probability),
        minimum_duration=int(args.minimum_duration),
        maximum_duration=int(args.maximum_duration),
        maximum_events=int(args.maximum_events),
        warmup_steps=HISTORY_STEPS,
    )
    rows: list[dict[str, Any]] = []
    for index, raw_step in enumerate(starts):
        if not bool(schedule.active[index]):
            continue
        age = int(schedule.event_age[index])
        base_index = index - age
        history_start = base_index - HISTORY_STEPS + 1
        if history_start < 0:
            raise RuntimeError("hazard schedule did not preserve A4b history")
        history = measurement[history_start : base_index + 1]
        history_action = _chunk_actions(
            actions,
            tuple(range(history_start, base_index)),
        )
        future_action = _future_actions_through_age(
            actions,
            first_transition=base_index,
            event_age=age,
        )
        wrong_action = _wrong_future_actions(future_action)
        target = measurement[index]
        predictions = {
            FACTUAL_METHOD: _predict_at_event_age(
                factual,
                history,
                history_action,
                future_action,
                task_index=task_index,
                device=device,
            ),
            ZERO_ACTION_METHOD: _predict_at_event_age(
                no_action,
                history,
                np.zeros_like(history_action),
                np.zeros_like(future_action),
                task_index=task_index,
                device=device,
            ),
            WRONG_ACTION_METHOD: _predict_at_event_age(
                factual,
                history,
                history_action,
                wrong_action,
                task_index=task_index,
                device=device,
            ),
            HOLD_LAST_METHOD: measurement[base_index].copy(),
            LATENT_CV_METHOD: measurement[base_index]
            + float(age)
            * (
                latent_constant_velocity(measurement[base_index - 1 : base_index + 1])
                - measurement[base_index]
            ),
        }
        for method in METHODS:
            predicted = predictions[method]
            rows.append(
                {
                    "task": str(episode["task"]),
                    "task_instance": int(episode["task_instance"]),
                    "checkpoint_seed": int(checkpoint_seed),
                    "raw_step": int(raw_step),
                    "event_id": int(schedule.event_id[index]),
                    "event_age": age,
                    "task_stage": str(episode["stages"][int(raw_step)]),
                    "method": method,
                    "anchor_mse": float(np.square(predicted[1:] - target[1:]).mean()),
                }
            )
    metadata = {
        "task": str(episode["task"]),
        "task_instance": int(episode["task_instance"]),
        "decision_count": len(starts),
        "hidden_decision_count": int(schedule.active.sum()),
        "events": list(schedule.events),
    }
    return rows, metadata


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA0 is required for the formal A4b random-hazard smoke")
    torch.cuda.set_device(device)

    contract = legacy._safe_read_json(args.config)
    validate_a4_contract(contract)
    tasks = tuple(args.task) if args.task else TASKS
    collection = contract["collection"]
    instances = _development_instances(
        contract, episodes_per_task=int(args.episodes_per_task)
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    episodes: list[dict[str, Any]] = []
    for task in tasks:
        task_index = TASKS.index(task)
        for episode_offset, task_instance in enumerate(instances[task]):
            environment_seed = (
                int(collection["benchmark_seed"])
                + 640_000
                + task_index * 10_000
                + int(task_instance)
            )
            episode = trajectory._collect_full_episode(
                task,
                task_instance=int(task_instance),
                environment_seed=environment_seed,
                width=int(collection["width"]),
                height=int(collection["height"]),
                benchmark_seed=int(collection["benchmark_seed"]),
                max_raw_steps=int(args.max_raw_steps),
                post_success_steps=int(args.post_success_steps),
                camera_name=a4_camera_name(task),
            )
            episode["model_steps"] = _decision_steps(len(episode["actions"]))
            episode["episode_offset"] = int(episode_offset)
            episodes.append(episode)
            print(
                json.dumps(
                    {
                        "stage": "collected",
                        "task": task,
                        "task_instance": int(task_instance),
                        "frames": len(episode["records"]),
                        "stable_success": episode["stable_success_step"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    encoder = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=device,
    )
    try:
        for episode in episodes:
            sampled = [
                episode["records"][int(index)]
                for index in np.asarray(episode["model_steps"], dtype=np.int64)
            ]
            episode["encoded_state"] = encoder.encode(sampled)
    finally:
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

    base_checkpoint = _torch_load(args.base_checkpoint)
    checkpoint_paths = tuple(args.checkpoints or DEFAULT_CHECKPOINTS)
    all_rows: list[dict[str, Any]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    episode_metadata: list[dict[str, Any]] | None = None
    observed_seeds: list[int] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = validate_a4_checkpoint(_torch_load(checkpoint_path))
        seed = int(checkpoint["seed"])
        if int(checkpoint["step"]) != 10_000:
            raise ValueError("random-hazard A4b evaluation requires step-10000 latest")
        factual, no_action, _ = build_matched_a4_models(
            base_checkpoint,
            seed=seed,
        )
        factual.load_state_dict(checkpoint["model_state"], strict=True)
        no_action.load_state_dict(checkpoint["no_action_state"], strict=True)
        factual.to(device).eval()
        no_action.to(device).eval()
        current_episode_metadata: list[dict[str, Any]] = []
        try:
            for episode in episodes:
                task_index = TASKS.index(str(episode["task"]))
                rows, metadata = _evaluate_episode_for_models(
                    factual,
                    no_action,
                    episode,
                    task_index=task_index,
                    checkpoint_seed=seed,
                    schedule_seed=(
                        int(args.seed)
                        + 1009 * task_index
                        + 104_729 * int(episode["episode_offset"])
                    ),
                    args=args,
                    device=device,
                )
                all_rows.extend(rows)
                current_episode_metadata.append(metadata)
        finally:
            del factual, no_action
            gc.collect()
            torch.cuda.empty_cache()
        if episode_metadata is None:
            episode_metadata = current_episode_metadata
        elif current_episode_metadata != episode_metadata:
            raise RuntimeError("hazard schedule changed between A4b checkpoint seeds")
        observed_seeds.append(seed)
        checkpoint_metadata.append(
            {
                "path": str(checkpoint_path.expanduser().resolve()),
                "seed": seed,
                "step": int(checkpoint["step"]),
            }
        )
        print(json.dumps({"stage": "evaluated", "seed": seed}, sort_keys=True), flush=True)

    summary = summarize_random_hazard_records(all_rows)
    summary["checkpoint_seed_count"] = len(observed_seeds)
    summary["checkpoint_seeds"] = observed_seeds
    payload = {
        "schema_version": 2,
        "protocol": "a4b_formal_models_random_hazard_predictor_smoke_v2",
        "scope": {
            "predictor_only": True,
            "sensor_valid_mask_is_privileged": True,
            "selector_trained": False,
            "corruption_onset_is_random": True,
            "corruption_onset_reads_task_state": False,
            "camera": "formal_a4_task_camera",
            "formal_claim_allowed": False,
        },
        "config": {
            "seed": int(args.seed),
            "hazard_probability": float(args.hazard_probability),
            "minimum_duration": int(args.minimum_duration),
            "maximum_duration": int(args.maximum_duration),
            "maximum_events": int(args.maximum_events),
            "episodes_per_task": int(args.episodes_per_task),
        },
        "base_checkpoint": str(args.base_checkpoint.expanduser().resolve()),
        "fcpg_checkpoint": str(args.fcpg_checkpoint.expanduser().resolve()),
        "checkpoints": checkpoint_metadata,
        "episodes": episode_metadata or [],
        "summary": summary,
        "records": all_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
