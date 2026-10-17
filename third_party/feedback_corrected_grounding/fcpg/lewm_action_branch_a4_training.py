"""Matched recursive dynamics training for the shared seven-task A4 model.

A4 trains from cached outputs of a completed ``anchor_persistent@2000``
encoder run.  The cache makes the frame encoder, projector, and SIGReg path
strictly frozen.  Consequently the training objective contains only the
matched factual and constant-zero-action recursive prediction losses; the
``0.02`` SIGReg value describes and validates the frozen encoder source.
"""

from __future__ import annotations

import copy
import os
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fcpg.lewm_a4_checkpoint import (
    A4_SPLITS,
    build_a4_checkpoint,
    build_a4_checkpoint_metadata,
)
from fcpg.lewm_a4_storage import (
    ACTION_DIM,
    LATENT_DIM,
    STATE_SLOT_COUNT,
    validate_a4_state_cache,
)
from fcpg.lewm_a4_training_data import A4StateCacheDataset
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    BUNDLE_BATCH_SIZE,
    BUNDLES_PER_TASK_PER_BATCH,
    CONFIRMATION_TRAINING_SEEDS,
    DEVELOPMENT_TRAINING_SEEDS,
    HISTORY_STEPS,
    HORIZON,
    PROTOCOL,
    SIGREG_WEIGHT,
    TASKS,
)
from fcpg.lewm_action_branch_training import (
    BASE_SIGREG_WEIGHT,
    all_cyclic_branch_derangements,
    validate_frozen_base_checkpoint,
)
from fcpg.lewm_anchor_persistent import ANCHOR_PERSISTENT_VARIANT
from fcpg.lewm_task_context_persistent import (
    TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT,
    TASK_CONTEXT_PERSISTENT_VARIANT,
    TaskContextPersistentLeWM,
)


A4_LATEST_CHECKPOINT_NAME = "lewm_action_branch_a4_latest.pt"
A4_EVALUATION_PREDICTIONS_NAME = "evaluation_predictions.npz"
A4_BASE_INITIALIZATION_NAME = "base_initialization.json"
A4_RESOLVED_CONFIG_NAME = "resolved_config.json"
A4_METRICS_NAME = "metrics.json"

A4_AMP_DTYPE = torch.float16
A4_LEARNING_RATE = 3.0e-4
A4_WEIGHT_DECAY = 1.0e-4
A4_BETAS = (0.9, 0.999)
A4_EPS = 1.0e-8
A4_WARMUP_STEPS = 100
A4_GRAD_CLIP = 1.0

A4_FRESH_STATE_KEYS = (
    "task_embedding.weight",
    "state_schema_version",
)
A4_TRAINABLE_MODULES = (
    "predictor",
    "action_encoder",
    "task_embedding",
    "pred_proj",
)
A4_FROZEN_MODULES = (
    "frame_encoder",
    "projector",
    "sigreg",
)


@dataclass(frozen=True)
class A4TrainingBudget:
    """One of the two preregistered A4 training endpoints."""

    name: str
    optimizer_steps: int
    seeds: tuple[int, ...]
    train_split: str
    evaluation_split: str | None


A4_TRAINING_BUDGETS = {
    "2k": A4TrainingBudget(
        name="2k",
        optimizer_steps=2_000,
        seeds=DEVELOPMENT_TRAINING_SEEDS,
        train_split="train",
        evaluation_split="development",
    ),
    "10k": A4TrainingBudget(
        name="10k",
        optimizer_steps=10_000,
        seeds=CONFIRMATION_TRAINING_SEEDS,
        train_split="train",
        # Reserved test is opened only by the independent fixed-endpoint
        # evaluator after all three confirmation checkpoints exist.
        evaluation_split=None,
    ),
}


@dataclass(frozen=True)
class A4BaseInitializationReport:
    """Exact cross-variant state transfer from the frozen anchor base."""

    source_variant: str
    target_variant: str
    source_key_count: int
    target_key_count: int
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatch_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class A4MatchedDynamicsOutput:
    """Predictions and weighted four-slot losses for one matched pair."""

    factual_prediction: torch.Tensor
    no_action_prediction: torch.Tensor
    factual_raw_mse: torch.Tensor
    no_action_raw_mse: torch.Tensor
    factual_context_mse: torch.Tensor
    factual_anchor_mse: torch.Tensor
    no_action_context_mse: torch.Tensor
    no_action_anchor_mse: torch.Tensor
    factual_branch_centered_mse: torch.Tensor
    no_action_branch_centered_mse: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        """The complete A4 objective; cached predictor heads add no SIGReg."""

        return self.factual_raw_mse + self.no_action_raw_mse


@dataclass(frozen=True)
class A4EvaluationResult:
    metrics: dict[str, float]
    arrays: dict[str, np.ndarray]


def _integer(value: object, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    minimum = 1 if positive else 0
    if result < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def resolve_a4_training_budget(name: str, *, seed: int) -> A4TrainingBudget:
    """Resolve and validate the only legal A4 budget/seed combinations."""

    try:
        budget = A4_TRAINING_BUDGETS[str(name)]
    except KeyError as error:
        raise ValueError(
            f"unknown A4 budget {name!r}; expected {sorted(A4_TRAINING_BUDGETS)}"
        ) from error
    seed_value = _integer(seed, label="A4 training seed")
    if seed_value not in budget.seeds:
        raise ValueError(
            f"A4 {budget.name} seed must be one of {list(budget.seeds)}, "
            f"got {seed_value}"
        )
    return budget


def _model_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("frozen base checkpoint is missing model_state")
    return state


def a4_model_config_from_anchor_base(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert only the model identity fields of an anchor-persistent base."""

    validate_frozen_base_checkpoint(checkpoint)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("frozen base checkpoint is missing model_config")
    result = copy.deepcopy(dict(model_config))
    if result.get("variant") != ANCHOR_PERSISTENT_VARIANT:
        raise ValueError(
            "A4 base model_config.variant must be exactly anchor_persistent"
        )
    world_model = result.get("world_model")
    if not isinstance(world_model, Mapping):
        raise ValueError("frozen base model_config is missing world_model")
    result["world_model"] = copy.deepcopy(dict(world_model))
    result["world_model"]["num_tasks"] = len(TASKS)
    result["variant"] = TASK_CONTEXT_PERSISTENT_VARIANT
    result["task_names"] = list(TASKS)
    result["sigreg_weight"] = SIGREG_WEIGHT
    return result


def validate_a4_model_contract(model: TaskContextPersistentLeWM) -> None:
    """Require the formal shared seven-task cached-state dimensions."""

    if not isinstance(model, TaskContextPersistentLeWM):
        raise TypeError("A4 model must be TaskContextPersistentLeWM")
    observed = (
        model.model_variant,
        int(model.num_tasks),
        int(model.history_size),
        int(model.num_state_slots),
        int(model.latent_dim),
        int(model.action_dim),
        float(model.sigreg_weight),
    )
    expected = (
        TASK_CONTEXT_PERSISTENT_VARIANT,
        len(TASKS),
        HISTORY_STEPS,
        STATE_SLOT_COUNT,
        LATENT_DIM,
        ACTION_DIM,
        SIGREG_WEIGHT,
    )
    if observed != expected:
        raise ValueError(
            "A4 TaskContextPersistentLeWM contract drifted: "
            f"expected={expected}, observed={observed}"
        )
    if SIGREG_WEIGHT != BASE_SIGREG_WEIGHT or (
        SIGREG_WEIGHT != TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT
    ):
        raise RuntimeError("A4 SIGReg constants disagree")


def _default_model_builder(
    model_config: Mapping[str, Any],
) -> TaskContextPersistentLeWM:
    from fcpg import lewm_runtime

    model = lewm_runtime._build_from_model_config(model_config)
    if not isinstance(model, TaskContextPersistentLeWM):
        raise TypeError("A4 model builder did not create TaskContextPersistentLeWM")
    return model


def transfer_anchor_base_state(
    model: TaskContextPersistentLeWM,
    checkpoint: Mapping[str, Any],
) -> A4BaseInitializationReport:
    """Load every overlapping same-shape key, leaving exactly two fresh keys."""

    validate_frozen_base_checkpoint(checkpoint)
    validate_a4_model_contract(model)
    source = _model_state(checkpoint)
    target = model.state_dict()
    fresh = set(A4_FRESH_STATE_KEYS)
    absent_fresh = [name for name in A4_FRESH_STATE_KEYS if name not in target]
    if absent_fresh:
        raise ValueError(f"A4 target model is missing fresh keys {absent_fresh}")

    loaded: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for name, target_value in target.items():
        if name in fresh:
            missing.append(name)
            continue
        source_value = source.get(name)
        if not torch.is_tensor(source_value):
            missing.append(name)
            continue
        if source_value.shape != target_value.shape or (
            source_value.dtype != target_value.dtype
        ):
            mismatched.append(name)
            continue
        loaded[name] = source_value

    unexpected = tuple(name for name in source if name not in target)
    report = A4BaseInitializationReport(
        source_variant=ANCHOR_PERSISTENT_VARIANT,
        target_variant=TASK_CONTEXT_PERSISTENT_VARIANT,
        source_key_count=len(source),
        target_key_count=len(target),
        loaded_keys=tuple(name for name in target if name in loaded),
        missing_keys=tuple(
            name for name in A4_FRESH_STATE_KEYS if name in set(missing)
        ),
        unexpected_keys=unexpected,
        shape_mismatch_keys=tuple(mismatched),
    )
    nonfresh_missing = [name for name in missing if name not in fresh]
    if (
        nonfresh_missing
        or set(report.missing_keys) != fresh
        or report.unexpected_keys
        or report.shape_mismatch_keys
    ):
        raise ValueError(
            "A4 anchor-base transfer must leave only task_embedding.weight and "
            "state_schema_version fresh; "
            f"missing={missing}, unexpected={list(unexpected)}, "
            f"shape_mismatch={mismatched}"
        )

    incompatible = model.load_state_dict(loaded, strict=False)
    if set(incompatible.missing_keys) != fresh or incompatible.unexpected_keys:
        raise RuntimeError(
            "A4 state transfer load report disagrees with the validated transfer: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    loaded_state = model.state_dict()
    for name in report.loaded_keys:
        if not torch.equal(loaded_state[name].detach().cpu(), source[name].detach().cpu()):
            raise RuntimeError(f"A4 base state key {name!r} was not loaded exactly")
    return report


def freeze_a4_cached_encoder(model: TaskContextPersistentLeWM) -> None:
    """Train only transition conditioning/output modules used by the cache."""

    validate_a4_model_contract(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module_name in A4_TRAINABLE_MODULES:
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.eval()
    for module_name in A4_FROZEN_MODULES:
        getattr(model, module_name).eval()


def set_a4_model_training_mode(
    model: TaskContextPersistentLeWM, training: bool
) -> None:
    """Set transition mode without ever enabling frozen cached-state modules."""

    model.train(bool(training))
    for module_name in A4_FROZEN_MODULES:
        getattr(model, module_name).eval()


def _trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, value in model.named_parameters() if value.requires_grad)


def assert_a4_matched_capacity(
    factual: TaskContextPersistentLeWM,
    no_action: TaskContextPersistentLeWM,
) -> None:
    """Check shared-model identity, matched capacity, and frozen boundaries."""

    validate_a4_model_contract(factual)
    validate_a4_model_contract(no_action)
    left = factual.state_dict()
    right = no_action.state_dict()
    if left.keys() != right.keys() or any(
        left[name].shape != right[name].shape for name in left
    ):
        raise ValueError("A4 factual/no-action model architectures differ")
    left_names = _trainable_parameter_names(factual)
    right_names = _trainable_parameter_names(no_action)
    if left_names != right_names:
        raise ValueError("A4 factual/no-action trainable parameter sets differ")
    allowed_prefixes = tuple(f"{name}." for name in A4_TRAINABLE_MODULES)
    if not left_names or any(
        not name.startswith(allowed_prefixes) for name in left_names
    ):
        raise ValueError("A4 trainable parameters escaped the cached dynamics boundary")
    for model in (factual, no_action):
        for module_name in A4_FROZEN_MODULES:
            if any(
                parameter.requires_grad
                for parameter in getattr(model, module_name).parameters()
            ):
                raise ValueError(f"A4 {module_name} must remain frozen")


def assert_a4_matched_initialization(
    factual: TaskContextPersistentLeWM,
    no_action: TaskContextPersistentLeWM,
) -> None:
    """Additionally require bit-identical state before optimization."""

    assert_a4_matched_capacity(factual, no_action)
    left = factual.state_dict()
    right = no_action.state_dict()
    unequal = [name for name in left if not torch.equal(left[name], right[name])]
    if unequal:
        raise ValueError(
            "A4 factual/no-action models must start from the same exact state; "
            f"unequal={unequal}"
        )


def initialize_a4_model_from_anchor_base(
    checkpoint: Mapping[str, Any],
    *,
    seed: int,
    model_builder: Callable[
        [Mapping[str, Any]], TaskContextPersistentLeWM
    ] | None = None,
) -> tuple[TaskContextPersistentLeWM, A4BaseInitializationReport]:
    """Build one fresh task-context model and transfer the frozen A4 base."""

    seed_value = _integer(seed, label="A4 initialization seed")
    config = a4_model_config_from_anchor_base(checkpoint)
    builder = model_builder or _default_model_builder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_value)
        model = builder(config)
    if not isinstance(model, TaskContextPersistentLeWM):
        raise TypeError("A4 model_builder must return TaskContextPersistentLeWM")
    model.runtime_model_config = copy.deepcopy(config)
    report = transfer_anchor_base_state(model, checkpoint)
    freeze_a4_cached_encoder(model)
    return model, report


def build_matched_a4_models(
    checkpoint: Mapping[str, Any],
    *,
    seed: int,
    model_builder: Callable[
        [Mapping[str, Any]], TaskContextPersistentLeWM
    ] | None = None,
) -> tuple[
    TaskContextPersistentLeWM,
    TaskContextPersistentLeWM,
    A4BaseInitializationReport,
]:
    """Create one shared seven-task factual/no-action matched pair."""

    factual, report = initialize_a4_model_from_anchor_base(
        checkpoint,
        seed=seed,
        model_builder=model_builder,
    )
    no_action = copy.deepcopy(factual)
    assert_a4_matched_initialization(factual, no_action)
    return factual, no_action, report


@dataclass(frozen=True)
class _TorchRNGState:
    cpu: torch.Tensor
    cuda: torch.Tensor | None
    cuda_device: torch.device | None


def _capture_torch_rng(device: torch.device) -> _TorchRNGState:
    cuda_device = device if device.type == "cuda" else None
    cuda = (
        torch.cuda.get_rng_state(cuda_device).clone()
        if cuda_device is not None
        else None
    )
    return _TorchRNGState(
        cpu=torch.random.get_rng_state().clone(),
        cuda=cuda,
        cuda_device=cuda_device,
    )


def _restore_torch_rng(state: _TorchRNGState) -> None:
    torch.random.set_rng_state(state.cpu)
    if state.cuda is not None and state.cuda_device is not None:
        torch.cuda.set_rng_state(state.cuda, state.cuda_device)


def _require_tensor(batch: Mapping[str, Any], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not torch.is_tensor(value):
        raise KeyError(f"A4 batch requires tensor {name!r}")
    return value


def _validate_recursive_inputs(
    model: TaskContextPersistentLeWM,
    history_embedding: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    task_index: torch.Tensor,
) -> tuple[int, int, int]:
    validate_a4_model_contract(model)
    if history_embedding.ndim != 4:
        raise ValueError("history_embedding must have shape [B,3,4,192]")
    batch_size = int(history_embedding.shape[0])
    expected_history = (
        batch_size,
        model.history_size,
        model.num_state_slots,
        model.latent_dim,
    )
    if tuple(history_embedding.shape) != expected_history:
        raise ValueError(f"history_embedding must have shape {expected_history}")
    expected_history_action = (
        batch_size,
        model.history_size - 1,
        model.action_dim,
    )
    if tuple(history_action.shape) != expected_history_action:
        raise ValueError(f"history_action must have shape {expected_history_action}")
    expected_future = (batch_size, BRANCH_COUNT, HORIZON, model.action_dim)
    if tuple(future_action.shape) != expected_future:
        raise ValueError(f"future_action must have shape {expected_future}")
    if task_index.shape != (batch_size,):
        raise ValueError(f"task_index must have shape {(batch_size,)}")
    if not (
        history_embedding.device
        == history_action.device
        == future_action.device
        == task_index.device
    ):
        raise ValueError("all A4 recursive tensors must share one device")
    return batch_size, BRANCH_COUNT, HORIZON


def rollout_a4_branches(
    model: TaskContextPersistentLeWM,
    history_embedding: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    task_index: torch.Tensor,
    *,
    no_action: bool = False,
) -> torch.Tensor:
    """Recursively predict all eight branches with factual or zero action."""

    batch_size, branches, horizon = _validate_recursive_inputs(
        model,
        history_embedding,
        history_action,
        future_action,
        task_index,
    )
    if not isinstance(no_action, bool):
        raise TypeError("no_action must be boolean")
    rollout_history_action = (
        torch.zeros_like(history_action) if no_action else history_action
    )
    rollout_future_action = (
        torch.zeros_like(future_action) if no_action else future_action
    )
    prediction = model.rollout_state_embeddings(
        history_embedding,
        rollout_history_action,
        rollout_future_action,
        task_index,
    )
    expected = (
        batch_size,
        branches,
        horizon,
        model.num_state_slots,
        model.latent_dim,
    )
    if tuple(prediction.shape) != expected:
        raise RuntimeError(
            f"A4 recursive prediction must have shape {expected}, "
            f"got {tuple(prediction.shape)}"
        )
    return prediction


def matched_a4_recursive_rollout(
    factual: TaskContextPersistentLeWM,
    no_action: TaskContextPersistentLeWM,
    history_embedding: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    task_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run both full trajectories from one matched dropout RNG state."""

    assert_a4_matched_capacity(factual, no_action)
    starting_rng = _capture_torch_rng(history_embedding.device)
    factual_prediction = rollout_a4_branches(
        factual,
        history_embedding,
        history_action,
        future_action,
        task_index,
    )
    ending_rng = _capture_torch_rng(history_embedding.device)
    _restore_torch_rng(starting_rng)
    try:
        no_action_prediction = rollout_a4_branches(
            no_action,
            history_embedding,
            history_action,
            future_action,
            task_index,
            no_action=True,
        )
    finally:
        _restore_torch_rng(ending_rng)
    return factual_prediction, no_action_prediction


def _weighted_slot_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("A4 prediction and target must share shape [B,8,8,4,192]")
    prediction_float = prediction.float()
    target_float = target.float()
    context = (prediction_float[..., 0, :] - target_float[..., 0, :]).square().mean()
    anchors = (
        prediction_float[..., 1:, :] - target_float[..., 1:, :]
    ).square().mean()
    combined = 0.25 * context + 0.75 * anchors
    return combined, context, anchors


def _weighted_branch_centered_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction_float = prediction.float()
    target_float = target.float()
    centered_prediction = prediction_float - prediction_float.mean(
        dim=1, keepdim=True
    )
    centered_target = target_float - target_float.mean(dim=1, keepdim=True)
    combined, _, _ = _weighted_slot_mse(centered_prediction, centered_target)
    return combined


def matched_a4_recursive_loss(
    factual: TaskContextPersistentLeWM,
    no_action: TaskContextPersistentLeWM,
    history_embedding: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    target_embedding: torch.Tensor,
    task_index: torch.Tensor,
) -> A4MatchedDynamicsOutput:
    """Compute matched closed-loop prediction loss with no head SIGReg."""

    factual_prediction, no_action_prediction = matched_a4_recursive_rollout(
        factual,
        no_action,
        history_embedding,
        history_action,
        future_action,
        task_index,
    )
    factual_raw, factual_context, factual_anchor = _weighted_slot_mse(
        factual_prediction, target_embedding
    )
    no_action_raw, no_action_context, no_action_anchor = _weighted_slot_mse(
        no_action_prediction, target_embedding
    )
    return A4MatchedDynamicsOutput(
        factual_prediction=factual_prediction,
        no_action_prediction=no_action_prediction,
        factual_raw_mse=factual_raw,
        no_action_raw_mse=no_action_raw,
        factual_context_mse=factual_context,
        factual_anchor_mse=factual_anchor,
        no_action_context_mse=no_action_context,
        no_action_anchor_mse=no_action_anchor,
        factual_branch_centered_mse=_weighted_branch_centered_mse(
            factual_prediction, target_embedding
        ),
        no_action_branch_centered_mse=_weighted_branch_centered_mse(
            no_action_prediction, target_embedding
        ),
    )


def _sequence_values(value: object, *, label: str, length: int) -> list[Any]:
    if torch.is_tensor(value):
        result = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        result = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
    else:
        raise TypeError(f"{label} must be a batch sequence")
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} values")
    return result


def validate_a4_training_batch(batch: Mapping[str, Any]) -> None:
    """Require the optimizer batch to be exactly ordered ``28 = 7 x 4``."""

    if not isinstance(batch, Mapping):
        raise TypeError("A4 training batch must be a mapping")
    history = _require_tensor(batch, "history_embedding")
    history_action = _require_tensor(batch, "history_action")
    future_action = _require_tensor(batch, "future_action")
    target = _require_tensor(batch, "target_embedding")
    task_index = _require_tensor(batch, "task_index")
    expected_shapes = {
        "history_embedding": (
            BUNDLE_BATCH_SIZE,
            HISTORY_STEPS,
            STATE_SLOT_COUNT,
            LATENT_DIM,
        ),
        "history_action": (
            BUNDLE_BATCH_SIZE,
            HISTORY_STEPS - 1,
            ACTION_DIM,
        ),
        "future_action": (
            BUNDLE_BATCH_SIZE,
            BRANCH_COUNT,
            HORIZON,
            ACTION_DIM,
        ),
        "target_embedding": (
            BUNDLE_BATCH_SIZE,
            BRANCH_COUNT,
            HORIZON,
            STATE_SLOT_COUNT,
            LATENT_DIM,
        ),
        "task_index": (BUNDLE_BATCH_SIZE,),
    }
    observed = {
        "history_embedding": tuple(history.shape),
        "history_action": tuple(history_action.shape),
        "future_action": tuple(future_action.shape),
        "target_embedding": tuple(target.shape),
        "task_index": tuple(task_index.shape),
    }
    for name, expected in expected_shapes.items():
        if observed[name] != expected:
            raise ValueError(
                f"A4 training {name} must have shape {expected}, "
                f"got {observed[name]}"
            )
    if task_index.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("A4 training task_index must have an integer dtype")
    expected_task_index = [
        task_index_value
        for task_index_value in range(len(TASKS))
        for _ in range(BUNDLES_PER_TASK_PER_BATCH)
    ]
    if task_index.detach().cpu().to(torch.long).tolist() != expected_task_index:
        raise ValueError("A4 training batch must contain four ordered rows per task")
    if "task_name" in batch:
        task_names = _sequence_values(
            batch["task_name"], label="task_name", length=BUNDLE_BATCH_SIZE
        )
        expected_names = [
            task for task in TASKS for _ in range(BUNDLES_PER_TASK_PER_BATCH)
        ]
        if [str(value) for value in task_names] != expected_names:
            raise ValueError("A4 training task_name order drifted from TASKS")
    if "source_episode_id" in batch:
        episodes = _sequence_values(
            batch["source_episode_id"],
            label="source_episode_id",
            length=BUNDLE_BATCH_SIZE,
        )
        for task_offset in range(len(TASKS)):
            start = task_offset * BUNDLES_PER_TASK_PER_BATCH
            selected = episodes[start : start + BUNDLES_PER_TASK_PER_BATCH]
            if len(set(selected)) != BUNDLES_PER_TASK_PER_BATCH:
                raise ValueError(
                    "A4 training requires four distinct episodes within each task"
                )


def move_a4_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    target = torch.device(device)
    return {
        name: (
            value.to(target, non_blocking=target.type == "cuda")
            if torch.is_tensor(value)
            else value
        )
        for name, value in batch.items()
    }


class _IndexedA4Dataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: A4StateCacheDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = dict(self.dataset[index])
        result["sample_index"] = self.dataset.source_indices[index]
        return result


def build_a4_evaluation_loader(
    cache: Mapping[str, np.ndarray],
    *,
    split: str,
    batch_size: int = BUNDLE_BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Any]]:
    """Build a deterministic, row-complete evaluation pass over an A4 split."""

    size = _integer(batch_size, label="evaluation batch_size", positive=True)
    workers = _integer(num_workers, label="num_workers")
    if not isinstance(pin_memory, bool):
        raise TypeError("pin_memory must be boolean")
    dataset = _IndexedA4Dataset(A4StateCacheDataset(cache, split=split))
    return DataLoader(
        dataset,
        batch_size=size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def _gather_all_wrong_action_predictions(
    factual_prediction: torch.Tensor,
    permutations: torch.Tensor,
) -> torch.Tensor:
    batch_size, branches, horizon, slots, latent_dim = factual_prediction.shape
    shifts = int(permutations.shape[1])
    expected = (batch_size, BRANCH_COUNT - 1, branches)
    if tuple(permutations.shape) != expected:
        raise ValueError(f"wrong-action permutations must have shape {expected}")
    source = factual_prediction[:, None].expand(-1, shifts, -1, -1, -1, -1)
    gather_index = permutations[:, :, :, None, None, None].expand(
        -1, -1, -1, horizon, slots, latent_dim
    )
    return torch.gather(source, dim=2, index=gather_index)


def _baseline_predictions(
    history: torch.Tensor,
    *,
    branches: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    last = history[:, -1]
    hold_last = last[:, None, None].expand(-1, branches, horizon, -1, -1)
    velocity = last - history[:, -2]
    steps = torch.arange(
        1,
        horizon + 1,
        dtype=history.dtype,
        device=history.device,
    )[None, None, :, None, None]
    constant_velocity = last[:, None, None] + steps * velocity[:, None, None]
    return hold_last, constant_velocity.expand(-1, branches, -1, -1, -1)


def _numpy_metadata(values: list[Any]) -> np.ndarray:
    if values and all(
        isinstance(value, Integral) and not isinstance(value, bool) for value in values
    ):
        return np.asarray(values, dtype=np.int64)
    return np.asarray([str(value) for value in values])


@torch.no_grad()
def evaluate_a4_models(
    factual: TaskContextPersistentLeWM,
    no_action: TaskContextPersistentLeWM,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    step: int,
    training_seed: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype = A4_AMP_DTYPE,
) -> A4EvaluationResult:
    """Collect every prediction needed by the A4 development comparisons."""

    step_value = _integer(step, label="A4 evaluation step", positive=True)
    budget_name = {2_000: "2k", 10_000: "10k"}.get(step_value)
    if budget_name is None:
        raise ValueError("A4 evaluation step must be exactly 2000 or 10000")
    resolve_a4_training_budget(budget_name, seed=training_seed)
    assert_a4_matched_capacity(factual, no_action)
    target_device = torch.device(device)
    if amp_enabled and target_device.type != "cuda":
        raise ValueError("A4 AMP evaluation requires a CUDA device")
    if amp_enabled and amp_dtype != torch.float16:
        raise ValueError("A4 AMP dtype is fixed at float16")
    set_a4_model_training_mode(factual, False)
    set_a4_model_training_mode(no_action, False)

    prediction_lists: dict[str, list[np.ndarray]] = {
        "history_state": [],
        "history_action": [],
        "future_action": [],
        "target_state": [],
        "factual_prediction": [],
        "no_action_prediction": [],
        "shuffled_action_prediction": [],
        "hold_last_prediction": [],
        "constant_velocity_prediction": [],
        "shuffle_permutation": [],
    }
    sample_indices: list[int] = []
    task_indices: list[int] = []
    task_names: list[Any] = []
    episode_ids: list[Any] = []
    metric_totals: dict[str, float] = {
        "factual_raw_mse": 0.0,
        "no_action_raw_mse": 0.0,
        "hold_last_raw_mse": 0.0,
        "constant_velocity_raw_mse": 0.0,
        "factual_branch_centered_mse": 0.0,
        "no_action_branch_centered_mse": 0.0,
    }
    shifted_totals = np.zeros(BRANCH_COUNT - 1, dtype=np.float64)
    sample_count = 0
    no_action_spread = 0.0

    for host_batch in loader:
        batch = move_a4_batch_to_device(host_batch, target_device)
        history = _require_tensor(batch, "history_embedding").float()
        history_action = _require_tensor(batch, "history_action").float()
        future_action = _require_tensor(batch, "future_action").float()
        target = _require_tensor(batch, "target_embedding").float()
        task_index = _require_tensor(batch, "task_index")
        with torch.autocast(
            device_type=target_device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = matched_a4_recursive_loss(
                factual,
                no_action,
                history,
                history_action,
                future_action,
                target,
                task_index,
            )
        permutations = all_cyclic_branch_derangements(
            int(history.shape[0]),
            branch_count=BRANCH_COUNT,
            device=target_device,
        )
        shuffled = _gather_all_wrong_action_predictions(
            output.factual_prediction, permutations
        )
        hold_last, constant_velocity = _baseline_predictions(
            history,
            branches=BRANCH_COUNT,
            horizon=HORIZON,
        )
        hold_loss, _, _ = _weighted_slot_mse(hold_last, target)
        velocity_loss, _, _ = _weighted_slot_mse(constant_velocity, target)
        shift_losses = [
            _weighted_slot_mse(shuffled[:, shift], target)[0]
            for shift in range(BRANCH_COUNT - 1)
        ]

        batch_size = int(history.shape[0])
        values = {
            "factual_raw_mse": output.factual_raw_mse,
            "no_action_raw_mse": output.no_action_raw_mse,
            "hold_last_raw_mse": hold_loss,
            "constant_velocity_raw_mse": velocity_loss,
            "factual_branch_centered_mse": output.factual_branch_centered_mse,
            "no_action_branch_centered_mse": output.no_action_branch_centered_mse,
        }
        for name, value in values.items():
            metric_totals[name] += float(value) * batch_size
        shifted_totals += np.asarray(
            [float(value) for value in shift_losses], dtype=np.float64
        ) * batch_size
        spread = (
            output.no_action_prediction - output.no_action_prediction[:, :1]
        ).abs().amax()
        no_action_spread = max(no_action_spread, float(spread))

        prediction_lists["history_state"].append(history.cpu().numpy())
        prediction_lists["history_action"].append(history_action.cpu().numpy())
        prediction_lists["future_action"].append(future_action.cpu().numpy())
        prediction_lists["target_state"].append(target.cpu().numpy())
        prediction_lists["factual_prediction"].append(
            output.factual_prediction.float().cpu().numpy()
        )
        prediction_lists["no_action_prediction"].append(
            output.no_action_prediction.float().cpu().numpy()
        )
        prediction_lists["shuffled_action_prediction"].append(
            shuffled.float().cpu().numpy()
        )
        prediction_lists["hold_last_prediction"].append(
            hold_last.float().cpu().numpy()
        )
        prediction_lists["constant_velocity_prediction"].append(
            constant_velocity.float().cpu().numpy()
        )
        prediction_lists["shuffle_permutation"].append(
            permutations.cpu().numpy()
        )

        fallback = list(range(sample_count, sample_count + batch_size))
        raw_sample_index = host_batch.get("sample_index", fallback)
        sample_indices.extend(
            int(value)
            for value in _sequence_values(
                raw_sample_index, label="sample_index", length=batch_size
            )
        )
        task_indices.extend(int(value) for value in task_index.detach().cpu().tolist())
        task_names.extend(
            _sequence_values(
                host_batch.get(
                    "task_name",
                    [TASKS[index] for index in task_index.detach().cpu().tolist()],
                ),
                label="task_name",
                length=batch_size,
            )
        )
        episode_ids.extend(
            _sequence_values(
                host_batch.get("source_episode_id", fallback),
                label="source_episode_id",
                length=batch_size,
            )
        )
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("A4 evaluation loader produced no samples")
    if no_action_spread > 1.0e-6:
        raise RuntimeError(
            "A4 no-action prediction varies across branches: "
            f"max_abs={no_action_spread}"
        )
    metrics = {name: value / sample_count for name, value in metric_totals.items()}
    metrics["samples"] = float(sample_count)
    metrics["no_action_max_branch_spread"] = no_action_spread
    metrics["factual_relative_improvement"] = (
        metrics["no_action_raw_mse"] - metrics["factual_raw_mse"]
    ) / max(metrics["no_action_raw_mse"], 1.0e-12)
    for shift_index, total in enumerate(shifted_totals, start=1):
        metrics[f"shuffled_action_raw_mse_shift_{shift_index}"] = float(
            total / sample_count
        )
    metrics["shuffled_action_raw_mse"] = float(
        shifted_totals.mean() / sample_count
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("A4 evaluation produced non-finite metrics")

    arrays = {
        name: np.concatenate(values, axis=0)
        for name, values in prediction_lists.items()
    }
    arrays.update(
        {
            "schema_version": np.asarray(1, dtype=np.int64),
            "protocol": np.asarray(PROTOCOL),
            "checkpoint_step": np.asarray(step_value, dtype=np.int64),
            "selection": np.asarray("fixed_latest"),
            "anchor_encoder_sigreg": np.asarray(SIGREG_WEIGHT, dtype=np.float64),
            "prediction_heads_add_sigreg": np.asarray(False),
            "training_seed": np.full(sample_count, training_seed, dtype=np.int64),
            "seed": np.full(sample_count, training_seed, dtype=np.int64),
            "sample_index": np.asarray(sample_indices, dtype=np.int64),
            "task_class_index": np.asarray(task_indices, dtype=np.int64),
            "task_name": _numpy_metadata(task_names),
            "source_episode_id": _numpy_metadata(episode_ids),
            "shuffled_action_donor_shifts": np.arange(
                1, BRANCH_COUNT, dtype=np.int64
            ),
            "shuffled_action_loss_reduction": np.asarray(
                "mean_of_all_seven_nonidentity_cyclic_donor_shifts"
            ),
            "shuffled_action_raw_mse_by_shift": shifted_totals / sample_count,
        }
    )
    expected_prediction = (
        sample_count,
        BRANCH_COUNT,
        HORIZON,
        STATE_SLOT_COUNT,
        LATENT_DIM,
    )
    for name in (
        "target_state",
        "factual_prediction",
        "no_action_prediction",
        "hold_last_prediction",
        "constant_velocity_prediction",
    ):
        if arrays[name].shape != expected_prediction:
            raise RuntimeError(f"A4 {name} shape is invalid: {arrays[name].shape}")
    expected_shuffled = (
        sample_count,
        BRANCH_COUNT - 1,
        *expected_prediction[1:],
    )
    if arrays["shuffled_action_prediction"].shape != expected_shuffled:
        raise RuntimeError("A4 all-shifts prediction shape is invalid")
    return A4EvaluationResult(metrics=metrics, arrays=arrays)


def attach_a4_cache_metadata(
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Attach row-aligned canonical cache identity and branch metadata."""

    validate_a4_state_cache(cache)
    result = dict(arrays)
    indices = np.asarray(result.get("sample_index"), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("A4 evaluation sample_index must be a non-empty vector")
    if int(indices.min()) < 0 or int(indices.max()) >= len(cache["split"]):
        raise IndexError("A4 evaluation sample_index is outside the cache")
    canonical = {
        "proposal_id": cache["proposal_id"][indices],
        "task_class_index": cache["task_class_index"][indices],
        "task_name": cache["task_name"][indices],
        "task_instance_index": cache["task_instance_index"][indices],
        "source_episode_id": cache["source_episode_id"][indices],
        "split": cache["split"][indices],
        "center_action": cache["center_action"][indices],
        "branch_semantic_index": cache["branch_semantic_index"][indices],
    }
    for name in ("task_class_index", "task_name", "source_episode_id"):
        if name in result and not np.array_equal(np.asarray(result[name]), canonical[name]):
            raise ValueError(f"A4 evaluation {name} is not cache-row aligned")
    result.update({name: np.asarray(value) for name, value in canonical.items()})
    return result


def summarize_a4_cache_counts(
    cache: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Build ordered per-split/task episode and bundle checkpoint tables."""

    validate_a4_state_cache(cache)
    split_values = cache["split"].astype(str, copy=False)
    task_values = cache["task_name"].astype(str, copy=False)
    episode_values = cache["source_episode_id"]
    episode_counts: dict[str, dict[str, int]] = {}
    bundle_counts: dict[str, dict[str, int]] = {}
    for split in A4_SPLITS:
        episode_counts[split] = {}
        bundle_counts[split] = {}
        for task in TASKS:
            selected = (split_values == split) & (task_values == task)
            bundle_counts[split][task] = int(selected.sum())
            episode_counts[split][task] = len(
                {str(value) for value in episode_values[selected].tolist()}
            )
    return episode_counts, bundle_counts


def build_a4_latest_checkpoint(
    factual: TaskContextPersistentLeWM,
    no_action: TaskContextPersistentLeWM,
    *,
    seed: int,
    step: int,
    episode_counts: Mapping[str, Mapping[str, int]],
    bundle_counts: Mapping[str, Mapping[str, int]],
    cache_identifier: str,
    base_identifier: str,
) -> dict[str, Any]:
    """Build the fixed endpoint using the exact existing A4 schema."""

    step_value = _integer(step, label="A4 checkpoint step", positive=True)
    budget_name = {2_000: "2k", 10_000: "10k"}.get(step_value)
    if budget_name is None:
        raise ValueError("A4 latest checkpoint must be step 2000 or 10000")
    budget = resolve_a4_training_budget(budget_name, seed=seed)
    assert_a4_matched_capacity(factual, no_action)
    metadata = build_a4_checkpoint_metadata(
        seed=seed,
        step=budget.optimizer_steps,
        episode_counts=episode_counts,
        bundle_counts=bundle_counts,
        cache_identifier=cache_identifier,
        base_identifier=base_identifier,
        sigreg_weight=SIGREG_WEIGHT,
    )
    return build_a4_checkpoint(
        metadata,
        model_state=factual.state_dict(),
        no_action_state=no_action.state_dict(),
    )


def atomic_save_a4_checkpoint(
    payload: Mapping[str, Any], path: str | os.PathLike[str]
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_save_a4_predictions(
    arrays: Mapping[str, np.ndarray], path: str | os.PathLike[str]
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "A4_AMP_DTYPE",
    "A4_BASE_INITIALIZATION_NAME",
    "A4_BETAS",
    "A4_EPS",
    "A4_EVALUATION_PREDICTIONS_NAME",
    "A4_FRESH_STATE_KEYS",
    "A4_FROZEN_MODULES",
    "A4_GRAD_CLIP",
    "A4_LATEST_CHECKPOINT_NAME",
    "A4_LEARNING_RATE",
    "A4_METRICS_NAME",
    "A4_RESOLVED_CONFIG_NAME",
    "A4_TRAINABLE_MODULES",
    "A4_TRAINING_BUDGETS",
    "A4_WARMUP_STEPS",
    "A4_WEIGHT_DECAY",
    "A4BaseInitializationReport",
    "A4EvaluationResult",
    "A4MatchedDynamicsOutput",
    "A4TrainingBudget",
    "a4_model_config_from_anchor_base",
    "assert_a4_matched_capacity",
    "assert_a4_matched_initialization",
    "atomic_save_a4_checkpoint",
    "atomic_save_a4_predictions",
    "attach_a4_cache_metadata",
    "build_a4_evaluation_loader",
    "build_a4_latest_checkpoint",
    "build_matched_a4_models",
    "evaluate_a4_models",
    "freeze_a4_cached_encoder",
    "initialize_a4_model_from_anchor_base",
    "matched_a4_recursive_loss",
    "matched_a4_recursive_rollout",
    "move_a4_batch_to_device",
    "resolve_a4_training_budget",
    "rollout_a4_branches",
    "set_a4_model_training_mode",
    "summarize_a4_cache_counts",
    "transfer_anchor_base_state",
    "validate_a4_model_contract",
    "validate_a4_training_batch",
]
