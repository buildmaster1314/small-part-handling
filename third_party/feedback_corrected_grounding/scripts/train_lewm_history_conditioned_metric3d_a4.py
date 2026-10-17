#!/usr/bin/env python3
"""Train a causal two-action-chunk adapter on the frozen world-v2 LeWM."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import SIGREG_WEIGHT, TASKS  # noqa: E402
from fcpg.lewm_action_prefix_joint_a4 import (  # noqa: E402
    joint_batch_from_memmaps,
    task_balanced_sample,
)
from fcpg.lewm_action_prefix_joint_data import open_raw_replay_arrays  # noqa: E402
from fcpg.lewm_history_conditioned_metric3d_a4 import (  # noqa: E402
    HISTORY_ADAPTER_HIDDEN_DIM,
    HISTORY_CONDITIONED_PROTOCOL,
    HistoryConditionedMetric3DLeWM,
    build_history_conditioned_model,
    history_conditioned_checkpoint_payload,
)
from fcpg.lewm_runtime import cosine_warmup_lambda, seed_everything  # noqa: E402
from scripts.train_lewm_action_prefix_joint_a4 import (  # noqa: E402
    GRAD_CLIP,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    _atomic_json,
    _atomic_torch,
    _move,
)
from scripts.train_lewm_camera_action_joint_a4 import _validate_dataset  # noqa: E402


DEFAULT_RAW = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_SOURCE = ROOT / "outputs/lewm_action_branch_a4c_v4/state_cache.npz"
DEFAULT_PARENT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "camera_action_metric3d_world_seed40582_2k/"
    "lewm_metric3d_joint_latest.pt"
)
ADAPTER_LR = 3.0e-4


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=40_583)
    parser.add_argument("--batch-size", type=int, default=84)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=HISTORY_ADAPTER_HIDDEN_DIM)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-final-evaluation", action="store_true")
    args = parser.parse_args(argv)
    if int(args.steps) <= 0 or int(args.seed) <= 0:
        parser.error("--steps and --seed must be positive")
    if int(args.batch_size) < len(TASKS):
        parser.error("--batch-size must be at least seven")
    if int(args.eval_batch_size) <= 0 or int(args.hidden_dim) <= 0:
        parser.error("evaluation batch size and hidden dim must be positive")
    if int(args.log_every) <= 0:
        parser.error("--log-every must be positive")
    return args


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("history-conditioned training permits only CUDA0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _history_array(source: Mapping[str, np.ndarray]) -> np.ndarray:
    value = np.asarray(source["history_action"], dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (2, 20):
        raise ValueError("source history_action must have shape [N,2,20]")
    if not np.isfinite(value).all():
        raise ValueError("source history_action contains non-finite values")
    return value


def _batch_from_memmaps(
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    history: np.ndarray,
    *,
    bundles: np.ndarray,
    branches: np.ndarray,
) -> dict[str, torch.Tensor]:
    batch = joint_batch_from_memmaps(
        index,
        arrays,
        bundle_indices=bundles,
        branch_indices=branches,
    )
    cache_index = np.asarray(index["cache_index"], dtype=np.int64)[bundles]
    if cache_index.size and (
        int(cache_index.min()) < 0 or int(cache_index.max()) >= history.shape[0]
    ):
        raise ValueError("raw cache_index is outside source history cache")
    batch["history_action"] = torch.from_numpy(
        np.ascontiguousarray(history[cache_index])
    ).float()
    return batch


def _sequence_epe(
    value: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    error = float(scale) * torch.linalg.vector_norm(
        value.float() - target.float(), dim=-1
    )
    weights = valid.float()
    return (error * weights).sum(dim=(-1, -2)) / weights.sum(
        dim=(-1, -2)
    ).clamp_min(1.0)


def _mismatched_history_cache_indices(
    index: Mapping[str, np.ndarray], bundle_indices: np.ndarray
) -> np.ndarray:
    """Choose a same-task history from a different physical bundle."""

    bundles = np.asarray(bundle_indices, dtype=np.int64)
    task_names = np.asarray(index["task_name"]).astype(str)
    cache_indices = np.asarray(index["cache_index"], dtype=np.int64)
    split = np.asarray(index["split"]).astype(str)
    result = np.empty_like(bundles)
    development = np.flatnonzero(split == "development")
    for task in np.unique(task_names[bundles]):
        task_rows = development[task_names[development] == task]
        unique_cache = np.unique(cache_indices[task_rows])
        if unique_cache.size < 2:
            raise ValueError(f"task {task!r} needs two development histories")
        successor = {
            int(value): int(unique_cache[(offset + 1) % unique_cache.size])
            for offset, value in enumerate(unique_cache)
        }
        selected = np.flatnonzero(task_names[bundles] == task)
        result[selected] = [
            successor[int(cache_indices[bundles[row]])] for row in selected
        ]
    if np.any(result == cache_indices[bundles]):
        raise RuntimeError("mismatched history reused the factual physical bundle")
    return result


@torch.inference_mode()
def _evaluate(
    model: HistoryConditionedMetric3DLeWM,
    index: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    history: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    development = np.flatnonzero(
        np.asarray(index["split"]).astype(str) == "development"
    )
    pairs = [(int(bundle), branch) for bundle in development for branch in range(8)]
    metric_names = (
        "world_epe_mm",
        "base_world_epe_mm",
        "shuffled_history_world_epe_mm",
        "pixel_epe",
        "base_pixel_epe",
        "shuffled_history_pixel_epe",
        "latent_mse",
        "base_latent_mse",
    )
    totals = {
        task: {"count": 0, **{name: 0.0 for name in metric_names}}
        for task in TASKS
    }
    for start in range(0, len(pairs), int(batch_size)):
        current = pairs[start : start + int(batch_size)]
        bundles = np.asarray([row[0] for row in current], dtype=np.int64)
        branches = np.asarray([row[1] for row in current], dtype=np.int64)
        batch = _move(
            _batch_from_memmaps(
                index,
                arrays,
                history,
                bundles=bundles,
                branches=branches,
            ),
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
            mismatch_cache = _mismatched_history_cache_indices(index, bundles)
            shuffled_history = torch.from_numpy(
                np.ascontiguousarray(history[mismatch_cache])
            ).to(device=device, dtype=batch["history_action"].dtype)
            shuffled_state = model.predict_with_history(
                current_state=output.current_state,
                history_action=shuffled_history,
                future_action=batch["future_action"],
                task_index=batch["task_index"],
                batch=batch,
            )
            shuffled_decoded = model.decode_3d(
                predicted_state=shuffled_state,
                current_state=output.current_state,
                current_keypoints_xyz=batch["current_keypoints_xyz"],
                future_action=batch["future_action"],
                task_index=batch["task_index"],
                camera_intrinsic=batch["camera_intrinsic"],
                extrinsic_cam_from_world=batch[
                    "camera_extrinsic_cam_from_world"
                ],
            )

        valid = batch["future_valid"].bool()
        world_target = batch["future_keypoints_xyz"]
        pixel_target = batch["future_q_true"]
        factual_latent = output.predicted_state.float() - output.target_state.float()
        base_latent = output.base_predicted_state.float() - output.target_state.float()
        values = {
            "world_epe_mm": _sequence_epe(
                output.predicted_world_xyz, world_target, valid, scale=1000.0
            ),
            "base_world_epe_mm": _sequence_epe(
                output.base_predicted_world_xyz, world_target, valid, scale=1000.0
            ),
            "shuffled_history_world_epe_mm": _sequence_epe(
                shuffled_decoded.predicted_world_xyz,
                world_target,
                valid,
                scale=1000.0,
            ),
            "pixel_epe": _sequence_epe(
                output.predicted_q_px, pixel_target, valid, scale=1.0
            ),
            "base_pixel_epe": _sequence_epe(
                output.base_predicted_q_px, pixel_target, valid, scale=1.0
            ),
            "shuffled_history_pixel_epe": _sequence_epe(
                shuffled_decoded.predicted_q_px,
                pixel_target,
                valid,
                scale=1.0,
            ),
            "latent_mse": factual_latent.square().mean(dim=(-1, -2, -3)),
            "base_latent_mse": base_latent.square().mean(dim=(-1, -2, -3)),
        }
        for row, task_index in enumerate(batch["task_index"].cpu().tolist()):
            task = TASKS[int(task_index)]
            totals[task]["count"] = int(totals[task]["count"]) + 1
            for name in metric_names:
                totals[task][name] = float(totals[task][name]) + float(
                    values[name][row]
                )

    per_task: dict[str, dict[str, float | int]] = {}
    for task in TASKS:
        count = int(totals[task]["count"])
        if count <= 0:
            raise RuntimeError(f"history evaluation missed task {task}")
        per_task[task] = {"sequence_count": count}
        for name in metric_names:
            per_task[task][name] = float(totals[task][name]) / count
    result: dict[str, Any] = {
        "sequence_count": len(pairs),
        "shuffled_history_control": (
            "same_task_next_development_physical_bundle"
        ),
        "per_task": per_task,
    }
    for name in metric_names:
        result[f"task_macro_{name}"] = float(
            np.mean([float(per_task[task][name]) for task in TASKS])
        )
    result["task_macro_history_world_gain_mm"] = (
        result["task_macro_base_world_epe_mm"]
        - result["task_macro_world_epe_mm"]
    )
    result["task_macro_history_pixel_gain_px"] = (
        result["task_macro_base_pixel_epe"] - result["task_macro_pixel_epe"]
    )
    result["task_macro_shuffled_history_world_penalty_mm"] = (
        result["task_macro_shuffled_history_world_epe_mm"]
        - result["task_macro_world_epe_mm"]
    )
    result["task_macro_shuffled_history_pixel_penalty_px"] = (
        result["task_macro_shuffled_history_pixel_epe"]
        - result["task_macro_pixel_epe"]
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    seed_everything(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True

    raw_path = args.raw_dataset.expanduser().resolve()
    source_path = args.source_cache.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    index, arrays = open_raw_replay_arrays(raw_path)
    _validate_dataset(index, arrays)
    source = np.load(source_path, allow_pickle=False)
    history = _history_array(source)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping):
        raise TypeError("history parent checkpoint must contain a mapping")
    if Path(str(parent.get("raw_dataset", ""))).resolve() != raw_path:
        raise ValueError("history parent and raw dataset differ")
    if float(parent.get("sigreg_weight", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("history parent must inherit SIGReg=0.02")

    model = build_history_conditioned_model(
        parent, hidden_dim=int(args.hidden_dim)
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    expected = list(model.history_dynamics.condition_adapter.parameters())
    if {id(value) for value in trainable} != {id(value) for value in expected}:
        raise RuntimeError("only the causal history adapter may be trainable")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=ADAPTER_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            min(WARMUP_STEPS, int(args.steps)),
            int(args.steps),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("history output directory must be empty")
    resolved = {
        "protocol": HISTORY_CONDITIONED_PROTOCOL,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "raw_dataset": str(raw_path),
        "source_cache": str(source_path),
        "parent_checkpoint": str(parent_path),
        "parent_step": int(parent.get("step", -1)),
        "sigreg_weight": SIGREG_WEIGHT,
        "history_action_steps": 2,
        "history_raw_action_count": 10,
        "hidden_dim": int(args.hidden_dim),
        "adapter_lr": ADAPTER_LR,
        "parent_frozen": True,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "trainable_parameter_count": int(sum(value.numel() for value in trainable)),
    }
    _atomic_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"resolved": resolved}, sort_keys=True), flush=True)

    rng = np.random.default_rng(int(args.seed))
    model.train()
    started = time.perf_counter()
    final: dict[str, Any] = {}
    for step in range(1, int(args.steps) + 1):
        bundles, branches = task_balanced_sample(
            index,
            split="train",
            batch_size=int(args.batch_size),
            rng=rng,
        )
        batch = _move(
            _batch_from_memmaps(
                index,
                arrays,
                history,
                bundles=bundles,
                branches=branches,
            ),
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError("history-conditioned loss is non-finite")
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        final = {
            "step": step,
            "train_loss": float(output.loss.detach()),
            "train_latent_loss": float(output.latent_loss.detach()),
            "train_world_loss": float(output.world_loss.detach()),
            "train_projected_metric_loss": float(output.metric_loss.detach()),
            "train_projected_aperture_loss": float(
                output.gripper_aperture_loss.detach()
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(float(value)) for value in final.values()):
            raise RuntimeError("history-conditioned training metric is non-finite")
        if step % int(args.log_every) == 0 or step == int(args.steps):
            print(json.dumps(final, sort_keys=True), flush=True)

    evaluation = (
        None
        if bool(args.skip_final_evaluation)
        else _evaluate(
            model,
            index,
            arrays,
            history,
            device=device,
            batch_size=int(args.eval_batch_size),
        )
    )
    metrics = {**final, "protocol": HISTORY_CONDITIONED_PROTOCOL, "evaluation": evaluation}
    _atomic_json(output_dir / "metrics.json", metrics)
    _atomic_torch(
        output_dir / "lewm_history_conditioned_latest.pt",
        history_conditioned_checkpoint_payload(
            model,
            step=int(args.steps),
            seed=int(args.seed),
            parent_checkpoint=str(parent_path),
            raw_dataset=str(raw_path),
            source_cache=str(source_path),
        ),
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
