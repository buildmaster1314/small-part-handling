"""Seven-task A4 training helpers for dense action-prefix dynamics.

The visual encoder and its ``SIGReg=0.02`` representation are frozen in the
A4 state cache.  This module trains only a Fast-LeWM-style dynamics head that
predicts every H1-H8 target directly from the latest observed four-slot state
and the corresponding future action prefix.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch import nn

from fcpg.lewm_a4_storage import (
    ACTION_DIM,
    LATENT_DIM,
    STATE_SLOT_COUNT,
    validate_a4_state_cache,
)
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    HORIZON,
    SIGREG_WEIGHT,
    TASKS,
)
from fcpg.lewm_action_prefix import ActionPrefixDynamics, dense_prefix_mse


ACTION_PREFIX_PROTOCOL = "a4_fast_lewm_dense_prefix_v1"
ACTION_PREFIX_CHECKPOINT_TYPE = "a4_action_prefix_dynamics"
GEOMETRY_LOSS_WEIGHT = 10.0
ACTION_PREFIX_MODEL_CONFIG = {
    "action_dim": ACTION_DIM,
    "state_dim": LATENT_DIM,
    "num_slots": STATE_SLOT_COUNT,
    "num_tasks": len(TASKS),
    "max_horizon": HORIZON,
    "prefix_depth": 3,
    "prefix_heads": 6,
    "prefix_mlp_dim": 768,
    "predictor_depth": 6,
    "predictor_hidden_dim": 2048,
    "fusion_dim": 768,
    "state_mlp_dim": 768,
    "dropout": 0.1,
}


@dataclass(frozen=True)
class ActionPrefixBatch:
    """Flattened branch batch accepted by ``ActionPrefixDynamics``."""

    current_state: torch.Tensor
    future_action: torch.Tensor
    target_state: torch.Tensor
    task_index: torch.Tensor


@dataclass(frozen=True)
class ActionPrefixEvaluation:
    metrics: dict[str, Any]


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def action_normalization_from_cache(
    cache: Mapping[str, np.ndarray],
    *,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-split statistics over the future actions seen by the head."""

    validate_a4_state_cache(cache)
    split_values = np.asarray(cache["split"]).astype(str)
    selected = split_values == str(split)
    if not bool(np.any(selected)):
        raise ValueError(f"A4 cache has no rows for split {split!r}")
    values = np.asarray(cache["future_action"][selected], dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (
        BRANCH_COUNT,
        HORIZON,
        ACTION_DIM,
    ):
        raise ValueError("future_action cache shape drifted")
    flat = values.reshape(-1, ACTION_DIM)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.maximum(std, 1.0e-6)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("A4 action normalization is non-finite")
    return mean.astype(np.float32), std.astype(np.float32)


def build_action_prefix_model(
    *,
    seed: int,
    action_mean: np.ndarray | torch.Tensor,
    action_std: np.ndarray | torch.Tensor,
    model_config: Mapping[str, Any] | None = None,
) -> ActionPrefixDynamics:
    """Build the shared seven-task paper-aligned dynamics head."""

    resolved = dict(ACTION_PREFIX_MODEL_CONFIG)
    if model_config is not None:
        resolved.update(dict(model_config))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_positive_int(seed, label="seed"))
        model = ActionPrefixDynamics(**resolved)
    model.set_action_normalization(action_mean, action_std)
    return model


def flatten_action_prefix_batch(batch: Mapping[str, Any]) -> ActionPrefixBatch:
    """Convert ``[bundle, branch, ...]`` cache tensors to dense sequences."""

    required = (
        "history_embedding",
        "future_action",
        "target_embedding",
        "task_index",
    )
    values: dict[str, torch.Tensor] = {}
    for name in required:
        value = batch.get(name)
        if not torch.is_tensor(value):
            raise KeyError(f"action-prefix batch requires tensor {name!r}")
        values[name] = value
    history = values["history_embedding"]
    actions = values["future_action"]
    target = values["target_embedding"]
    task = values["task_index"]
    if history.ndim != 4 or history.shape[1:] != (
        3,
        STATE_SLOT_COUNT,
        LATENT_DIM,
    ):
        raise ValueError("history_embedding must have shape [B,3,4,192]")
    batch_size = int(history.shape[0])
    if actions.shape != (
        batch_size,
        BRANCH_COUNT,
        HORIZON,
        ACTION_DIM,
    ):
        raise ValueError("future_action must have shape [B,8,8,20]")
    if target.shape != (
        batch_size,
        BRANCH_COUNT,
        HORIZON,
        STATE_SLOT_COUNT,
        LATENT_DIM,
    ):
        raise ValueError("target_embedding must have shape [B,8,8,4,192]")
    if task.shape != (batch_size,):
        raise ValueError("task_index must have shape [B]")
    current = history[:, -1]
    repeated_current = current[:, None].expand(-1, BRANCH_COUNT, -1, -1)
    repeated_task = task[:, None].expand(-1, BRANCH_COUNT)
    return ActionPrefixBatch(
        current_state=repeated_current.reshape(
            batch_size * BRANCH_COUNT,
            STATE_SLOT_COUNT,
            LATENT_DIM,
        ),
        future_action=actions.reshape(
            batch_size * BRANCH_COUNT,
            HORIZON,
            ACTION_DIM,
        ),
        target_state=target.reshape(
            batch_size * BRANCH_COUNT,
            HORIZON,
            STATE_SLOT_COUNT,
            LATENT_DIM,
        ),
        task_index=repeated_task.reshape(batch_size * BRANCH_COUNT),
    )


def _slot_mse_by_horizon(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    difference = prediction.float() - target.float()
    context = difference[..., 0, :].square().mean(dim=-1)
    anchor = difference[..., 1:, :].square().mean(dim=(-1, -2))
    return 0.25 * context + 0.75 * anchor


def action_prefix_loss(
    model: ActionPrefixDynamics,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense H1-H8 prediction loss and predictions."""

    flat = flatten_action_prefix_batch(batch)
    prediction = model(
        flat.current_state,
        flat.future_action,
        task_index=flat.task_index,
    )
    loss = dense_prefix_mse(prediction, flat.target_state)
    return loss, prediction


def decoder_projected_anchor_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task_index: torch.Tensor,
    coefficient: torch.Tensor,
) -> torch.Tensor:
    """Project anchor-latent error into the fixed train-only pixel subspace."""

    if prediction.ndim != 4 or prediction.shape[2:] != (
        STATE_SLOT_COUNT,
        LATENT_DIM,
    ):
        raise ValueError("prediction must have shape [B,H,4,192]")
    if target.shape != prediction.shape:
        raise ValueError("target must match prediction")
    if task_index.shape != (prediction.shape[0],):
        raise ValueError("task_index must have shape [B]")
    expected_coefficient = (len(TASKS), 3 * LATENT_DIM, 2)
    if coefficient.shape != expected_coefficient:
        raise ValueError(
            "coefficient must have shape "
            f"{expected_coefficient}, got {tuple(coefficient.shape)}"
        )
    if coefficient.device != prediction.device:
        raise ValueError("coefficient and prediction must share device")
    if task_index.device != prediction.device:
        raise ValueError("task_index and prediction must share device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point")
    if not coefficient.is_floating_point():
        raise TypeError("coefficient must be floating point")
    if not bool(torch.isfinite(coefficient).all()):
        raise ValueError("coefficient contains non-finite values")
    task_index = task_index.to(dtype=torch.long)
    if task_index.numel() and (
        int(task_index.amin()) < 0 or int(task_index.amax()) >= len(TASKS)
    ):
        raise ValueError("task_index is outside the seven-task panel")

    difference = prediction.float()[..., 1:, :] - target.float()[..., 1:, :]
    flattened = difference.reshape(prediction.shape[0], prediction.shape[1], -1)
    selected = coefficient.float()[task_index]
    result = torch.einsum("bhd,bdc->bhc", flattened, selected)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("decoder-projected anchor error is non-finite")
    return result


def decoder_aligned_geometry_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    task_index: torch.Tensor,
    coefficient: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Return normalized pixel MSE induced by the frozen spatial decoder."""

    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 1 or height <= 1:
        raise ValueError("image_size dimensions must be greater than one")
    error = decoder_projected_anchor_error(
        prediction,
        target,
        task_index,
        coefficient,
    )
    scale = error.new_tensor([width - 1, height - 1])
    loss = (error / scale).square().mean()
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("decoder-aligned geometry loss is non-finite")
    return loss


def _baseline_predictions(
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    history = batch["history_embedding"].float()
    target = batch["target_embedding"].float()
    batch_size = int(history.shape[0])
    last = history[:, -1]
    hold = last[:, None, None].expand(
        -1, BRANCH_COUNT, HORIZON, -1, -1
    )
    velocity = last - history[:, -2]
    step = torch.arange(
        1,
        HORIZON + 1,
        device=history.device,
        dtype=history.dtype,
    )[None, None, :, None, None]
    constant_velocity = last[:, None, None] + step * velocity[:, None, None]
    constant_velocity = constant_velocity.expand(
        batch_size, BRANCH_COUNT, -1, -1, -1
    )
    if hold.shape != target.shape or constant_velocity.shape != target.shape:
        raise RuntimeError("action-prefix baseline shape drifted")
    return hold, constant_velocity


@torch.no_grad()
def evaluate_action_prefix_model(
    model: ActionPrefixDynamics,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    amp_enabled: bool = True,
    geometry_coefficient: torch.Tensor | None = None,
) -> ActionPrefixEvaluation:
    """Evaluate prediction, action dependence, and simple latent baselines."""

    target_device = torch.device(device)
    model.eval()
    totals = {
        "prediction": np.zeros(HORIZON, dtype=np.float64),
        "hold_last": np.zeros(HORIZON, dtype=np.float64),
        "constant_velocity": np.zeros(HORIZON, dtype=np.float64),
        "zero_action": np.zeros(HORIZON, dtype=np.float64),
        "shuffled_action": np.zeros(HORIZON, dtype=np.float64),
    }
    task_totals = {
        task: {
            name: np.zeros(HORIZON, dtype=np.float64) for name in totals
        }
        for task in TASKS
    }
    task_counts = {task: 0 for task in TASKS}
    projected_error_sum = 0.0
    projected_error_count = 0
    task_projected_error_sum = {task: 0.0 for task in TASKS}
    task_projected_error_count = {task: 0 for task in TASKS}
    if geometry_coefficient is not None:
        geometry_coefficient = geometry_coefficient.to(
            device=target_device,
            dtype=torch.float32,
        )
    sequence_count = 0
    for host_batch in loader:
        batch = {
            name: (
                value.to(target_device, non_blocking=target_device.type == "cuda")
                if torch.is_tensor(value)
                else value
            )
            for name, value in host_batch.items()
        }
        flat = flatten_action_prefix_batch(batch)
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=amp_enabled and target_device.type == "cuda",
        ):
            prediction = model(
                flat.current_state,
                flat.future_action,
                task_index=flat.task_index,
            )
            zero_prediction = model(
                flat.current_state,
                torch.zeros_like(flat.future_action),
                task_index=flat.task_index,
            )
            shuffled_action = batch["future_action"].roll(1, dims=1)
            shuffled_batch = dict(batch)
            shuffled_batch["future_action"] = shuffled_action
            shuffled = flatten_action_prefix_batch(shuffled_batch)
            shuffled_prediction = model(
                shuffled.current_state,
                shuffled.future_action,
                task_index=shuffled.task_index,
            )
        bundle_count = int(batch["history_embedding"].shape[0])
        prediction = prediction.reshape(
            bundle_count,
            BRANCH_COUNT,
            HORIZON,
            STATE_SLOT_COUNT,
            LATENT_DIM,
        )
        zero_prediction = zero_prediction.reshape_as(prediction)
        shuffled_prediction = shuffled_prediction.reshape_as(prediction)
        target = batch["target_embedding"]
        hold, velocity = _baseline_predictions(batch)
        values = {
            "prediction": _slot_mse_by_horizon(prediction, target),
            "hold_last": _slot_mse_by_horizon(hold, target),
            "constant_velocity": _slot_mse_by_horizon(velocity, target),
            "zero_action": _slot_mse_by_horizon(zero_prediction, target),
            "shuffled_action": _slot_mse_by_horizon(
                shuffled_prediction, target
            ),
        }
        projected_epe = None
        if geometry_coefficient is not None:
            flat_prediction = prediction.reshape(
                bundle_count * BRANCH_COUNT,
                HORIZON,
                STATE_SLOT_COUNT,
                LATENT_DIM,
            )
            flat_target = target.reshape_as(flat_prediction)
            projected_error = decoder_projected_anchor_error(
                flat_prediction,
                flat_target,
                flat.task_index,
                geometry_coefficient,
            )
            projected_epe = projected_error.square().sum(dim=-1).sqrt()
            projected_error_sum += float(projected_epe.sum())
            projected_error_count += int(projected_epe.numel())
        for name, tensor in values.items():
            totals[name] += tensor.sum(dim=(0, 1)).cpu().numpy()
        sequence_count += bundle_count * BRANCH_COUNT
        task_index = batch["task_index"].detach().cpu().numpy()
        for task_id, task_name in enumerate(TASKS):
            mask = np.flatnonzero(task_index == task_id)
            if mask.size == 0:
                continue
            task_counts[task_name] += int(mask.size * BRANCH_COUNT)
            for name, tensor in values.items():
                task_totals[task_name][name] += (
                    tensor[mask].sum(dim=(0, 1)).cpu().numpy()
                )
            if projected_epe is not None:
                sequence_mask = flat.task_index == task_id
                selected_epe = projected_epe[sequence_mask]
                task_projected_error_sum[task_name] += float(selected_epe.sum())
                task_projected_error_count[task_name] += int(selected_epe.numel())
    if sequence_count <= 0 or any(count <= 0 for count in task_counts.values()):
        raise RuntimeError("action-prefix evaluation did not cover seven tasks")

    def summarize(source: Mapping[str, np.ndarray], count: int) -> dict[str, Any]:
        means = {name: values / float(count) for name, values in source.items()}
        return {
            "mean": {name: float(values.mean()) for name, values in means.items()},
            "by_horizon": {
                name: {
                    f"H{index + 1}": float(value)
                    for index, value in enumerate(values)
                }
                for name, values in means.items()
            },
            "action_shuffle_delta": float(
                means["shuffled_action"].mean() - means["prediction"].mean()
            ),
            "zero_action_delta": float(
                means["zero_action"].mean() - means["prediction"].mean()
            ),
        }

    per_task = {
        task: summarize(task_totals[task], task_counts[task]) for task in TASKS
    }
    aggregate = summarize(totals, sequence_count)
    aggregate["task_macro_prediction_mse"] = float(
        np.mean([per_task[task]["mean"]["prediction"] for task in TASKS])
    )
    aggregate["task_macro_action_shuffle_delta"] = float(
        np.mean([per_task[task]["action_shuffle_delta"] for task in TASKS])
    )
    if geometry_coefficient is not None:
        if projected_error_count <= 0 or any(
            task_projected_error_count[task] <= 0 for task in TASKS
        ):
            raise RuntimeError("geometry evaluation did not cover seven tasks")
        aggregate["decoder_projected_anchor_epe_px"] = float(
            projected_error_sum / projected_error_count
        )
        aggregate["task_macro_decoder_projected_anchor_epe_px"] = float(
            np.mean(
                [
                    task_projected_error_sum[task]
                    / task_projected_error_count[task]
                    for task in TASKS
                ]
            )
        )
        for task in TASKS:
            per_task[task]["decoder_projected_anchor_epe_px"] = float(
                task_projected_error_sum[task]
                / task_projected_error_count[task]
            )
    return ActionPrefixEvaluation(
        metrics={
            "protocol": ACTION_PREFIX_PROTOCOL,
            "sequence_count": int(sequence_count),
            "anchor_encoder_sigreg": SIGREG_WEIGHT,
            "prediction_head_adds_sigreg": False,
            "aggregate": aggregate,
            "per_task": per_task,
        }
    )


def checkpoint_payload(
    model: ActionPrefixDynamics,
    *,
    seed: int,
    step: int,
    input_npz: str,
) -> dict[str, Any]:
    """Build a self-contained dynamics-head checkpoint."""

    return {
        "checkpoint_type": ACTION_PREFIX_CHECKPOINT_TYPE,
        "protocol": ACTION_PREFIX_PROTOCOL,
        "seed": _positive_int(seed, label="seed"),
        "step": _positive_int(step, label="step"),
        "tasks": list(TASKS),
        "model_config": dict(ACTION_PREFIX_MODEL_CONFIG),
        "model_state": model.state_dict(),
        "input_npz": str(input_npz),
        "anchor_encoder_sigreg": SIGREG_WEIGHT,
        "prediction_head_adds_sigreg": False,
    }


def load_action_prefix_checkpoint(
    checkpoint: Mapping[str, Any],
) -> ActionPrefixDynamics:
    """Validate and instantiate one saved action-prefix dynamics head."""

    if checkpoint.get("checkpoint_type") != ACTION_PREFIX_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not an A4 action-prefix checkpoint")
    if checkpoint.get("protocol") != ACTION_PREFIX_PROTOCOL:
        raise ValueError("action-prefix checkpoint protocol drifted")
    if tuple(checkpoint.get("tasks", ())) != TASKS:
        raise ValueError("action-prefix checkpoint task panel drifted")
    if float(checkpoint.get("anchor_encoder_sigreg", -1.0)) != SIGREG_WEIGHT:
        raise ValueError("action-prefix checkpoint SIGReg provenance drifted")
    config = checkpoint.get("model_config")
    state = checkpoint.get("model_state")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("action-prefix checkpoint is incomplete")
    model = ActionPrefixDynamics(**dict(config))
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "ACTION_PREFIX_CHECKPOINT_TYPE",
    "ACTION_PREFIX_MODEL_CONFIG",
    "ACTION_PREFIX_PROTOCOL",
    "ActionPrefixBatch",
    "ActionPrefixEvaluation",
    "GEOMETRY_LOSS_WEIGHT",
    "action_normalization_from_cache",
    "action_prefix_loss",
    "build_action_prefix_model",
    "checkpoint_payload",
    "decoder_aligned_geometry_loss",
    "decoder_projected_anchor_error",
    "evaluate_action_prefix_model",
    "flatten_action_prefix_batch",
    "load_action_prefix_checkpoint",
]
