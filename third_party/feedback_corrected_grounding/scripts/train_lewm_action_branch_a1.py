#!/usr/bin/env python3
"""Train A1 closed-loop matched factual/no-action heads from cached latents."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_training import (  # noqa: E402
    ACTION_BRANCH_TRAINING_PROTOCOL,
    A1_ACTION_DIM,
    A1_BRANCH_COUNT,
    A1_HISTORY_SIZE,
    A1_LATENT_DIM,
    A1_MIN_HORIZON,
    A1_NUM_KEYPOINTS,
    BASE_SIGREG_WEIGHT,
    BUDGET_STEPS,
    CachedActionBranchDataset,
    action_normalization_from_base_checkpoint,
    all_cyclic_branch_derangements,
    assert_matched_predictor_contract,
    budget_steps,
    load_action_branch_cache,
    matched_action_branch_loss,
    permute_action_branches,
    raw_and_branch_centered_mse,
    rollout_action_branches,
    validate_frozen_base_checkpoint,
)
from fcpg.lewm_anchor_dynamics_pair import (  # noqa: E402
    MATCHED_DYNAMICS_PAIR_CHECKPOINT_TYPE,
    MatchedAnchorDynamicsPredictor,
    build_matched_predictor_pair,
    predictor_kwargs_from_base_checkpoint,
    trainable_parameter_count,
)
from fcpg.lewm_action_branch_metrics import fit_train_moments  # noqa: E402
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.lewm_runtime import (  # noqa: E402
    choose_device,
    cosine_warmup_lambda,
    seed_everything,
)


LATEST_CHECKPOINT_NAME = "latest.pt"
VAL_PREDICTIONS_NAME = "val_predictions.npz"
DEFAULT_BASE_CHECKPOINT = (
    ROOT
    / "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_BUNDLE_BATCH_SIZE = {"smoke": 4, "2k": 16, "10k": 16}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", choices=sorted(BUDGET_STEPS), required=True)
    parser.add_argument(
        "--screen-summary",
        type=Path,
        default=None,
        help="required passed screen summary before a fresh 10k run",
    )
    parser.add_argument("--seed", type=int, default=18_401)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="number of bundles per batch (each bundle contains eight branches)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    result = parser.parse_args(argv)
    for name in ("checkpoint_every", "log_every"):
        if int(getattr(result, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if result.batch_size is not None and result.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if result.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if result.lr <= 0.0 or result.weight_decay < 0.0:
        parser.error("optimizer learning rate/weight decay is invalid")
    if not (0.0 <= result.beta1 < 1.0 and 0.0 <= result.beta2 < 1.0):
        parser.error("optimizer betas must be in [0,1)")
    if result.eps <= 0.0 or result.warmup_steps < 0 or result.grad_clip < 0.0:
        parser.error("optimizer epsilon/warmup/gradient clipping is invalid")
    return result


def _default_splits(
    budget: str, train_split: str | None, val_split: str | None
) -> tuple[str, str]:
    expected = {
        "smoke": ("smoke", "smoke"),
        "2k": ("train", "development"),
        "10k": ("train", "test"),
    }[budget]
    resolved = (train_split or expected[0], val_split or expected[1])
    if resolved != expected:
        raise ValueError(
            f"A1 budget {budget} freezes train/val splits to {expected}, got {resolved}"
        )
    return resolved


def _validate_10k_authorization(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("screen summary must contain a JSON object")
    if (
        payload.get("protocol") != ACTION_BRANCH_TRAINING_PROTOCOL
        or payload.get("stage") != "screen"
        or payload.get("decision") != "Go"
        or payload.get("passed") is not True
    ):
        raise ValueError("10k requires a passed A1 screen summary")
    gate = payload.get("gate")
    coverage = payload.get("coverage")
    if not isinstance(gate, dict) or not isinstance(coverage, dict):
        raise ValueError("10k requires complete nested A1 screen evidence")
    if (
        gate.get("stage") != "screen"
        or gate.get("scientific_decision") is not True
        or gate.get("evaluated") is not True
        or gate.get("decision") != "Go"
        or gate.get("passed") is not True
        or gate.get("failed_checks") != []
    ):
        raise ValueError("10k requires a consistent nested A1 screen Go")
    checks = gate.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("10k requires complete nested A1 screen checks")
    support = checks.get("informative_bundle_fraction_per_task")
    if not isinstance(support, dict) or support.get("passed") is not True:
        raise ValueError("10k requires informative support in every A1 task")
    if any(
        not isinstance(check, dict) or check.get("passed") is not True
        for check in checks.values()
    ):
        raise ValueError("10k requires consistent nested A1 screen checks")
    task_names = coverage.get("task_names")
    if (
        coverage.get("task_count") != 4
        or not isinstance(task_names, list)
        or len(task_names) != 4
        or not all(isinstance(task, str) for task in task_names)
        or len(set(task_names)) != 4
    ):
        raise ValueError("10k requires exactly four-task A1 screen coverage")
    training_seeds = coverage.get("training_seeds")
    if (
        coverage.get("training_seed_count") != 3
        or not isinstance(training_seeds, list)
        or len(training_seeds) != 3
        or not all(
            isinstance(seed, int) and not isinstance(seed, bool)
            for seed in training_seeds
        )
        or len(set(training_seeds)) != 3
    ):
        raise ValueError("10k requires exactly three-seed A1 screen coverage")
    informative = payload.get("informative_bundles")
    per_task = (
        informative.get("per_task")
        if isinstance(informative, dict)
        else None
    )
    if (
        not isinstance(informative, dict)
        or informative.get("all_tasks_passed") is not True
        or not isinstance(per_task, dict)
        or set(per_task) != set(task_names)
        or any(
            not isinstance(row, dict) or row.get("passed") is not True
            for row in per_task.values()
        )
    ):
        raise ValueError("10k requires informative support in every A1 task")
    return payload


def _amp_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported AMP dtype: {name}")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must contain a mapping: {path}")
    return payload


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(value: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_a1_dimensions(predictor_config: Mapping[str, object]) -> None:
    expected = {
        "history_size": A1_HISTORY_SIZE,
        "num_keypoints": A1_NUM_KEYPOINTS,
        "latent_dim": A1_LATENT_DIM,
        "action_dim": A1_ACTION_DIM,
    }
    actual = {name: int(predictor_config[name]) for name in expected}
    if actual != expected:
        raise ValueError(f"A1 base predictor dimensions must be {expected}, got {actual}")


def _batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_index = torch.as_tensor(batch["sample_index"], dtype=torch.int64)
    tensors = []
    for name in (
        "history_embedding",
        "history_action",
        "future_action",
        "target_embedding",
    ):
        tensor = torch.as_tensor(batch[name]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"training batch {name} contains non-finite values")
        tensors.append(tensor)
    return sample_index, tensors[0], tensors[1], tensors[2], tensors[3]


def _metric_values(output: Any) -> dict[str, float]:
    return {
        "factual_raw_mse": float(output.factual_raw_mse.detach()),
        "no_action_raw_mse": float(output.no_action_raw_mse.detach()),
        "factual_branch_centered_mse": float(
            output.factual_branch_centered_mse.detach()
        ),
        "no_action_branch_centered_mse": float(
            output.no_action_branch_centered_mse.detach()
        ),
    }


def _checkpoint_payload(
    *,
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    resolved_config: Mapping[str, Any],
    predictor_config: Mapping[str, object],
    step: int,
    epoch: int,
    metrics: Mapping[str, Any],
    completed: bool,
) -> dict[str, Any]:
    target_steps = int(resolved_config["trainer"]["optimizer_steps"])
    if step > target_steps or completed != (step == target_steps):
        raise ValueError("checkpoint completion state is inconsistent")
    return {
        "checkpoint_type": MATCHED_DYNAMICS_PAIR_CHECKPOINT_TYPE,
        "training_protocol": ACTION_BRANCH_TRAINING_PROTOCOL,
        "schema_version": 1,
        "step": int(step),
        "epoch": int(epoch),
        "completed": bool(completed),
        "selection": "fixed_latest",
        "seed": int(resolved_config["seed"]),
        "base_checkpoint": resolved_config["base_checkpoint"],
        "base_checkpoint_sha256": resolved_config["base_checkpoint_sha256"],
        "base_sigreg_weight": BASE_SIGREG_WEIGHT,
        "encoder_frozen": True,
        "prediction_heads_add_sigreg": False,
        "predictor_config": dict(predictor_config),
        "parameter_count_per_head": trainable_parameter_count(factual),
        "factual_state": factual.state_dict(),
        "no_action_state": no_action.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_total_steps": target_steps,
        "metrics": dict(metrics),
        "resolved_config": dict(resolved_config),
    }


@torch.no_grad()
def _evaluate_and_collect(
    factual: MatchedAnchorDynamicsPredictor,
    no_action: MatchedAnchorDynamicsPredictor,
    loader: DataLoader,
    *,
    cache: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    step: int,
    training_seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    factual.eval()
    no_action.eval()
    metric_totals = {
        "factual_raw_mse": 0.0,
        "no_action_raw_mse": 0.0,
        "factual_branch_centered_mse": 0.0,
        "no_action_branch_centered_mse": 0.0,
        "shuffled_action_raw_mse": 0.0,
        "shuffled_action_branch_centered_mse": 0.0,
    }
    sample_count = 0
    indices: list[np.ndarray] = []
    factual_predictions: list[np.ndarray] = []
    no_action_predictions: list[np.ndarray] = []
    shuffled_action_predictions: list[np.ndarray] = []
    shuffle_permutations: list[np.ndarray] = []
    shuffled_raw_totals_by_shift = np.zeros(A1_BRANCH_COUNT - 1, dtype=np.float64)
    shuffled_centered_totals_by_shift = np.zeros(
        A1_BRANCH_COUNT - 1, dtype=np.float64
    )
    no_action_branch_spread = 0.0
    for host_batch in loader:
        batch_indices, history, history_action, future_action, target = _batch_to_device(
            host_batch, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = matched_action_branch_loss(
                factual,
                no_action,
                history,
                history_action,
                future_action,
                target,
            )
            permutations = all_cyclic_branch_derangements(
                int(history.shape[0]),
                branch_count=int(future_action.shape[1]),
                device=device,
            )
            shuffled_batch: list[np.ndarray] = []
            shuffled_raw_values: list[float] = []
            shuffled_centered_values: list[float] = []
            for shift_index in range(int(permutations.shape[1])):
                shuffled_action = permute_action_branches(
                    future_action, permutations[:, shift_index]
                )
                shuffled_prediction = rollout_action_branches(
                    factual,
                    history,
                    history_action,
                    shuffled_action,
                )
                shuffled_raw, shuffled_centered = raw_and_branch_centered_mse(
                    shuffled_prediction, target
                )
                shuffled_raw_values.append(float(shuffled_raw))
                shuffled_centered_values.append(float(shuffled_centered))
                shuffled_batch.append(
                    shuffled_prediction.detach().float().cpu().numpy()
                )
        values = _metric_values(output)
        values["shuffled_action_raw_mse"] = float(np.mean(shuffled_raw_values))
        values["shuffled_action_branch_centered_mse"] = float(
            np.mean(shuffled_centered_values)
        )
        batch_size = int(history.shape[0])
        for name, value in values.items():
            metric_totals[name] += value * batch_size
        shuffled_raw_totals_by_shift += np.asarray(
            shuffled_raw_values, dtype=np.float64
        ) * batch_size
        shuffled_centered_totals_by_shift += np.asarray(
            shuffled_centered_values, dtype=np.float64
        ) * batch_size
        sample_count += batch_size
        spread = (
            output.no_action_prediction
            - output.no_action_prediction[:, :1]
        ).abs().max()
        no_action_branch_spread = max(no_action_branch_spread, float(spread))
        indices.append(batch_indices.numpy())
        factual_predictions.append(
            output.factual_prediction.detach().float().cpu().numpy()
        )
        no_action_predictions.append(
            output.no_action_prediction.detach().float().cpu().numpy()
        )
        all_shift_batch = np.stack(shuffled_batch, axis=1)
        all_shift_permutations = permutations.detach().cpu().numpy()
        shuffled_action_predictions.append(all_shift_batch)
        shuffle_permutations.append(all_shift_permutations)
    if sample_count == 0:
        raise RuntimeError("validation loader produced no samples")
    if no_action_branch_spread > 1e-6:
        raise RuntimeError(
            "no-action predictions vary across action branches: "
            f"max_abs={no_action_branch_spread}"
        )
    metrics = {name: value / sample_count for name, value in metric_totals.items()}
    metrics["samples"] = float(sample_count)
    metrics["no_action_max_branch_spread"] = no_action_branch_spread
    metrics["factual_relative_raw_improvement"] = (
        metrics["no_action_raw_mse"] - metrics["factual_raw_mse"]
    ) / max(metrics["no_action_raw_mse"], 1e-12)
    for shift_index in range(A1_BRANCH_COUNT - 1):
        donor_shift = shift_index + 1
        metrics[f"shuffled_action_raw_mse_shift_{donor_shift}"] = float(
            shuffled_raw_totals_by_shift[shift_index] / sample_count
        )
        metrics[
            f"shuffled_action_branch_centered_mse_shift_{donor_shift}"
        ] = float(shuffled_centered_totals_by_shift[shift_index] / sample_count)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("validation produced non-finite metrics")

    sample_indices = np.concatenate(indices).astype(np.int64, copy=False)
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "protocol": np.asarray(ACTION_BRANCH_TRAINING_PROTOCOL),
        "checkpoint_step": np.asarray(step, dtype=np.int64),
        "selection": np.asarray("fixed_latest"),
        "prediction_heads_add_sigreg": np.asarray(False),
        "training_seed": np.full(sample_count, training_seed, dtype=np.int64),
        "seed": np.full(sample_count, training_seed, dtype=np.int64),
        "sample_index": sample_indices,
        "split": cache.split[sample_indices],
        "task_index": cache.task_index[sample_indices],
        "task_instance_index": cache.task_instance_index[sample_indices],
        "source_episode_id": cache.source_episode_id[sample_indices],
        "hazard_id": cache.hazard_id[sample_indices],
        "history": cache.history_embedding[sample_indices],
        "history_action": cache.history_action[sample_indices],
        "center_action": cache.center_action[sample_indices],
        "future_action": cache.future_action[sample_indices],
        "target": cache.target_embedding[sample_indices],
        "factual_prediction": np.concatenate(factual_predictions).astype(
            np.float32, copy=False
        ),
        "no_action_prediction": np.concatenate(no_action_predictions).astype(
            np.float32, copy=False
        ),
        "shuffled_action_prediction": np.concatenate(
            shuffled_action_predictions
        ).astype(np.float32, copy=False),
        "shuffle_permutation": np.concatenate(shuffle_permutations).astype(
            np.int64, copy=False
        ),
        "shuffled_action_donor_shifts": np.arange(
            1, A1_BRANCH_COUNT, dtype=np.int64
        ),
        "shuffled_action_loss_reduction": np.asarray(
            "mean_of_per_shift_raw_mse_over_all_nonidentity_cyclic_donor_shifts"
        ),
        "shuffled_action_raw_mse_by_shift": (
            shuffled_raw_totals_by_shift / sample_count
        ).astype(np.float64, copy=False),
        "shuffled_action_branch_centered_mse_by_shift": (
            shuffled_centered_totals_by_shift / sample_count
        ).astype(np.float64, copy=False),
    }
    expected_prediction_shape = (
        sample_count,
        A1_BRANCH_COUNT,
        cache.horizon,
        A1_NUM_KEYPOINTS,
        A1_LATENT_DIM,
    )
    if arrays["factual_prediction"].shape != expected_prediction_shape:
        raise RuntimeError("validation factual prediction shape is invalid")
    if arrays["no_action_prediction"].shape != expected_prediction_shape:
        raise RuntimeError("validation no-action prediction shape is invalid")
    expected_all_shifts_shape = (
        sample_count,
        A1_BRANCH_COUNT - 1,
        *expected_prediction_shape[1:],
    )
    if arrays["shuffled_action_prediction"].shape != expected_all_shifts_shape:
        raise RuntimeError("validation all-shifts prediction shape is invalid")
    expected_permutations_shape = (
        sample_count,
        A1_BRANCH_COUNT - 1,
        A1_BRANCH_COUNT,
    )
    if arrays["shuffle_permutation"].shape != expected_permutations_shape:
        raise RuntimeError("validation all-shifts permutations shape is invalid")
    return metrics, arrays


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    total_steps = budget_steps(args.budget)
    bundle_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else DEFAULT_BUNDLE_BATCH_SIZE[args.budget]
    )
    train_split, val_split = _default_splits(
        args.budget, args.train_split, args.val_split
    )
    screen_authorization = _validate_10k_authorization(args.screen_summary)
    if args.budget == "10k" and screen_authorization is None:
        raise ValueError("--screen-summary is required for a fresh 10k A1 run")
    if args.budget != "10k" and screen_authorization is not None:
        raise ValueError("--screen-summary is only valid for the 10k budget")
    seed_everything(int(args.seed))
    device = choose_device(args.device)
    amp_dtype = _amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.amp) and device.type == "cuda"

    base_path = args.base_checkpoint.expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    checkpoint = _torch_load(base_path)
    validate_frozen_base_checkpoint(checkpoint)
    predictor_config = predictor_kwargs_from_base_checkpoint(checkpoint)
    _validate_a1_dimensions(predictor_config)
    action_mean, action_std = action_normalization_from_base_checkpoint(
        checkpoint, action_dim=A1_ACTION_DIM
    )

    cache = load_action_branch_cache(args.input_npz)
    if cache.horizon != A1_MIN_HORIZON:
        raise ValueError(
            f"scientific A1 training requires exactly H={A1_MIN_HORIZON}, "
            f"got H={cache.horizon}"
        )
    train_indices = cache.indices_for_split(train_split)
    val_indices = cache.indices_for_split(val_split)
    train_moments = fit_train_moments(cache.target_embedding[train_indices])
    train_dataset = CachedActionBranchDataset(cache, train_indices)
    val_dataset = CachedActionBranchDataset(cache, val_indices)
    loader_generator = torch.Generator().manual_seed(int(args.seed) + 1)
    loader_kwargs = {
        "batch_size": bundle_batch_size,
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=loader_generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    factual, no_action = build_matched_predictor_pair(
        checkpoint, initialization_seed=int(args.seed)
    )
    for model in (factual, no_action):
        model.set_action_normalization(
            action_mean=action_mean,
            action_std=action_std,
        )
        model.to(device)
    assert_matched_predictor_contract(factual, no_action)

    parameter_groups = [
        {"params": list(factual.parameters()), "name": "factual"},
        {"params": list(no_action.parameters()), "name": "no_action"},
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(float(args.beta1), float(args.beta2)),
        eps=float(args.eps),
    )
    warmup_steps = min(int(args.warmup_steps), total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(step, warmup_steps, total_steps),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in output_dir.iterdir())
    if existing:
        raise FileExistsError(
            "A1 runs must start from scratch in an empty output directory; "
            f"found {existing}"
        )
    resolved_config: dict[str, Any] = {
        "schema_version": 1,
        "protocol": ACTION_BRANCH_TRAINING_PROTOCOL,
        "seed": int(args.seed),
        "budget": args.budget,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "input_npz": str(cache.path),
        "input_npz_sha256": sha256_file(cache.path),
        "data": {
            "history_size": A1_HISTORY_SIZE,
            "num_keypoints": A1_NUM_KEYPOINTS,
            "latent_dim": A1_LATENT_DIM,
            "action_dim": A1_ACTION_DIM,
            "branch_count": A1_BRANCH_COUNT,
            "minimum_horizon": A1_MIN_HORIZON,
            "horizon": cache.horizon,
            "train_split": train_split,
            "val_split": val_split,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "cached_latent_bundle_batch_size": bundle_batch_size,
            "recursive_trajectories_per_full_batch": (
                bundle_batch_size * A1_BRANCH_COUNT
            ),
        },
        "base": {
            "encoder_frozen": True,
            "sigreg_weight": BASE_SIGREG_WEIGHT,
        },
        "model": {
            "matched_architecture": True,
            "matched_initialization": True,
            "matched_full_rollout_dropout_rng": True,
            "prediction_heads_add_sigreg": False,
            "factual_action_conditioned": True,
            "no_action_channel": "constant_zero",
            "predictor_config": dict(predictor_config),
        },
        "loss": {
            "objective": "sum_of_factual_and_no_action_all_horizon_raw_mse",
            "closed_loop": True,
            "detach_between_steps": False,
            "branch_centered_diagnostic": True,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "betas": [float(args.beta1), float(args.beta2)],
            "eps": float(args.eps),
            "warmup_steps": warmup_steps,
        },
        "trainer": {
            "optimizer_steps": total_steps,
            "scheduler_total_steps": total_steps,
            "fresh_start": True,
            "resume_from_other_budget_allowed": False,
            "bundle_batch_size": bundle_batch_size,
            "grad_clip": float(args.grad_clip),
            "amp": amp_enabled,
            "amp_dtype": args.amp_dtype,
            "selection": "fixed_latest",
            "screen_authorized": screen_authorization is not None,
        },
        "artifacts": {
            "latest_checkpoint": LATEST_CHECKPOINT_NAME,
            "val_predictions": VAL_PREDICTIONS_NAME,
            "best_checkpoint": None,
            "shuffled_action_control": (
                "all_seven_nonidentity_cyclic_donor_shifts"
            ),
        },
    }
    _atomic_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
        output_dir / "resolved_config.json",
    )

    latest_path = output_dir / LATEST_CHECKPOINT_NAME
    metrics_path = output_dir / "metrics.jsonl"
    factual.train()
    no_action.train()
    step = 0
    epoch = 1
    iterator = iter(train_loader)
    interval_samples = 0
    interval_totals = {
        "factual_raw_mse": 0.0,
        "no_action_raw_mse": 0.0,
        "factual_branch_centered_mse": 0.0,
        "no_action_branch_centered_mse": 0.0,
    }
    last_record: dict[str, Any] = {}
    started = time.perf_counter()
    while step < total_steps:
        try:
            host_batch = next(iterator)
        except StopIteration:
            epoch += 1
            iterator = iter(train_loader)
            host_batch = next(iterator)
        _, history, history_action, future_action, target = _batch_to_device(
            host_batch, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = matched_action_branch_loss(
                factual,
                no_action,
                history,
                history_action,
                future_action,
                target,
            )
            loss = output.loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("A1 training produced a non-finite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(factual.parameters(), float(args.grad_clip))
            nn.utils.clip_grad_norm_(no_action.parameters(), float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        batch_size = int(history.shape[0])
        values = _metric_values(output)
        for name, value in values.items():
            interval_totals[name] += value * batch_size
        interval_samples += batch_size

        should_log = step % int(args.log_every) == 0 or step == total_steps
        if should_log:
            train_metrics = {
                name: value / interval_samples
                for name, value in interval_totals.items()
            }
            last_record = {
                "step": step,
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - started,
                "train": train_metrics,
            }
            if not all(math.isfinite(value) for value in train_metrics.values()):
                raise RuntimeError("A1 training produced non-finite metrics")
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_record, sort_keys=True) + "\n")
            print(json.dumps(last_record, sort_keys=True), flush=True)
            interval_samples = 0
            interval_totals = {name: 0.0 for name in interval_totals}

        if step % int(args.checkpoint_every) == 0 and step < total_steps:
            _atomic_torch_save(
                _checkpoint_payload(
                    factual=factual,
                    no_action=no_action,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    resolved_config=resolved_config,
                    predictor_config=predictor_config,
                    step=step,
                    epoch=epoch,
                    metrics=last_record,
                    completed=False,
                ),
                latest_path,
            )

    _atomic_torch_save(
        _checkpoint_payload(
            factual=factual,
            no_action=no_action,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resolved_config=resolved_config,
            predictor_config=predictor_config,
            step=step,
            epoch=epoch,
            metrics=last_record,
            completed=True,
        ),
        latest_path,
    )

    val_metrics, prediction_arrays = _evaluate_and_collect(
        factual,
        no_action,
        val_loader,
        cache=cache,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        step=step,
        training_seed=int(args.seed),
    )
    prediction_arrays.update(
        {
            "train_mean": train_moments.mean,
            "train_scale": train_moments.scale,
            "train_moment_count": np.asarray(
                train_moments.count, dtype=np.int64
            ),
        }
    )
    final_record = {
        **last_record,
        "step": step,
        "epoch": epoch,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "val": val_metrics,
        "selection": "fixed_latest",
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(final_record, sort_keys=True) + "\n")
    _atomic_npz(output_dir / VAL_PREDICTIONS_NAME, prediction_arrays)
    _atomic_torch_save(
        _checkpoint_payload(
            factual=factual,
            no_action=no_action,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resolved_config=resolved_config,
            predictor_config=predictor_config,
            step=step,
            epoch=epoch,
            metrics=final_record,
            completed=True,
        ),
        latest_path,
    )
    print(json.dumps(final_record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
