"""Train-only physical-state decoding for frozen A4 latent trajectories.

The decoder is diagnostic: it fits one closed-form ridge map from the formal
train target latents to physical labels and applies that frozen map to
development target and model-predicted latents.  Branch bundles remain nested
inside their source physical episode for weighting and bootstrap inference.

Only future labels that are actually present in A4 ``preflight_labels`` are
decoded.  In particular, ``branch_outcome[..., 0]`` is task progress and
``physical_state[..., :3]`` is the task-controlled position (object position
for every task except ``reach-v3``, where it is TCP position).  The archive
does not contain a future goal-relative vector or all-task future robot/TCP
state, so those quantities are explicitly unavailable rather than inferred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any
import zlib

import numpy as np

from fcpg.lewm_a4_storage import (
    ACTION_DIM,
    LATENT_DIM,
    STATE_SLOT_COUNT,
    validate_a4_state_cache,
)
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    CONFIRMATION_TRAINING_SEEDS,
    HORIZON,
    PROTOCOL,
    TASKS,
)
from fcpg.lewm_action_branch_a4_collection import validate_a4_diagnostics
from fcpg.lewm_action_branch_a4_metrics import (
    RIDGE_ALPHA,
    closed_form_ridge_predict,
    validate_wrong_action_prediction_replay,
)


SCHEMA_VERSION = 2
STAGE = "exposed_development_exploratory_mechanism_diagnostic"
AUDIT_DESIGNATION = "exposed-development exploratory mechanism diagnostic"
FEATURE_DIM = STATE_SLOT_COUNT * LATENT_DIM
ANCHOR_START = 1
ANCHOR_FEATURE_DIM = (STATE_SLOT_COUNT - ANCHOR_START) * LATENT_DIM
MATCHED_DIRECT_ACTION_FEATURE_DIM = FEATURE_DIM
TARGET_DIM = 4
TARGET_FIELDS = ("task_progress", "task_controlled_position_xyz")
WRONG_ACTION_SHIFT_COUNT = BRANCH_COUNT - 1
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 30_701
NUMERIC_EPSILON = 1.0e-12

HORIZON_WINDOWS: Mapping[str, tuple[int, ...]] = {
    "h1_h8": tuple(range(HORIZON)),
    "h6_h8": tuple(range(5, HORIZON)),
}


@dataclass(frozen=True)
class ReadoutSpec:
    name: str
    dimensions: tuple[int, ...]
    supported_tasks: tuple[str, ...]
    unit: str
    semantic: str


READOUT_SPECS = (
    ReadoutSpec(
        name="task_progress",
        dimensions=(0,),
        supported_tasks=tuple(TASKS),
        unit="unit_interval",
        semantic=(
            "collector task progress from branch_outcome[...,0]; this is the "
            "task-specific in-place progress signal, not a reconstructed reward"
        ),
    ),
    ReadoutSpec(
        name="task_controlled_position",
        dimensions=(1, 2, 3),
        supported_tasks=tuple(TASKS),
        unit="meter",
        semantic=(
            "physical_state xyz: object position for six tasks and TCP position "
            "for reach-v3"
        ),
    ),
    ReadoutSpec(
        name="object_position",
        dimensions=(1, 2, 3),
        supported_tasks=tuple(task for task in TASKS if task != "reach-v3"),
        unit="meter",
        semantic="physical_state xyz restricted to tasks where it is object position",
    ),
    ReadoutSpec(
        name="tcp_position",
        dimensions=(1, 2, 3),
        supported_tasks=("reach-v3",),
        unit="meter",
        semantic="physical_state xyz restricted to reach-v3, where it is TCP position",
    ),
)


def _readonly_float(value: object, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_float(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _scalar(array: object, *, label: str) -> object:
    value = np.asarray(array)
    if value.ndim != 0:
        raise ValueError(f"{label} must be scalar")
    return value.item()


def _future_physical_targets(labels: Mapping[str, np.ndarray]) -> np.ndarray:
    progress = np.asarray(labels["branch_outcome"], dtype=np.float64)[..., :1]
    position = np.asarray(labels["physical_state"], dtype=np.float64)[..., :3]
    if progress.shape[:-1] != position.shape[:-1]:
        raise ValueError("A4 progress and physical-state future axes differ")
    result = np.concatenate((progress, position), axis=-1)
    if result.shape[1:] != (BRANCH_COUNT, HORIZON, TARGET_DIM):
        raise ValueError(
            "A4 future physical targets must have shape [N,8,8,4]"
        )
    if not np.isfinite(result).all():
        raise ValueError("A4 future physical targets must be finite")
    if np.any((result[..., 0] < 0.0) | (result[..., 0] > 1.0)):
        raise ValueError("A4 task progress labels must remain in [0,1]")
    success = np.asarray(labels["branch_outcome"], dtype=np.float64)[..., 1]
    if not np.all(np.isin(success, (0.0, 1.0))):
        raise ValueError("A4 branch success labels must be exactly binary")
    return result


def validate_a4_physical_decoding_inputs(
    cache: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Validate the exact formal train/development cache and label contract."""

    for name, values in (("state cache", cache), ("preflight labels", labels)):
        if "split" in values and np.any(
            np.asarray(values["split"]).astype(str, copy=False) == "test"
        ):
            raise PermissionError(
                f"A4 physical decoding refuses {name} containing reserved-test rows"
            )
    validate_a4_state_cache(cache)
    validate_a4_diagnostics(labels)
    split = np.asarray(cache["split"]).astype(str, copy=False)
    label_split = np.asarray(labels["split"]).astype(str, copy=False)
    if set(split.tolist()) != {"train", "development"}:
        if np.any(split == "test"):
            raise PermissionError(
                "A4 physical decoding refuses every cache containing reserved-test rows"
            )
        raise ValueError(
            "A4 physical decoding requires both train and development rows"
        )
    if not np.array_equal(split, label_split):
        raise ValueError("A4 preflight-label split differs from the state cache")

    identity_fields = (
        "proposal_id",
        "task_class_index",
        "task_name",
        "task_instance_index",
        "source_episode_id",
        "split",
    )
    for name in identity_fields:
        if not np.array_equal(np.asarray(cache[name]), np.asarray(labels[name])):
            raise ValueError(
                f"A4 preflight-label {name} is not exactly state-cache aligned"
            )
    if not bool(np.all(np.asarray(labels["branch_valid"], dtype=bool))):
        raise ValueError(
            "A4 physical decoding requires every saved branch to be valid"
        )

    task = np.asarray(cache["task_name"]).astype(str, copy=False)
    episode = np.asarray(cache["source_episode_id"]).astype(str, copy=False)
    per_task: dict[str, Any] = {}
    for task_name in TASKS:
        task_rows = task == task_name
        if not np.any(task_rows):
            raise ValueError(f"A4 physical decoding is missing task {task_name}")
        split_counts: dict[str, Any] = {}
        for split_name in ("train", "development"):
            selected = task_rows & (split == split_name)
            episode_count = int(np.unique(episode[selected]).size)
            if episode_count == 0:
                raise ValueError(
                    f"A4 physical decoding has no {split_name} episode for {task_name}"
                )
            split_counts[split_name] = {
                "bundle_count": int(np.count_nonzero(selected)),
                "physical_episode_count": episode_count,
            }
        per_task[task_name] = split_counts
    _future_physical_targets(labels)
    return {
        "bundle_count": int(split.size),
        "train_bundle_count": int(np.count_nonzero(split == "train")),
        "development_bundle_count": int(
            np.count_nonzero(split == "development")
        ),
        "per_task": per_task,
    }


def _episode_equal_point_weights(
    task: np.ndarray, episode: np.ndarray
) -> np.ndarray:
    task_index = np.asarray(task, dtype=np.int64)
    episode_id = np.asarray(episode).astype(str, copy=False)
    if task_index.ndim != 1 or episode_id.shape != task_index.shape:
        raise ValueError("ridge task and physical-episode vectors must align")
    weights = np.zeros(task_index.size, dtype=np.float64)
    for index, task_name in enumerate(TASKS):
        selected_task = task_index == index
        identities = np.unique(episode_id[selected_task])
        if identities.size == 0:
            raise ValueError(f"ridge train points are missing task {task_name}")
        for identity in identities:
            selected = selected_task & (episode_id == identity)
            point_count = int(np.count_nonzero(selected))
            if point_count == 0:
                raise RuntimeError("physical-episode point count unexpectedly vanished")
            weights[selected] = 1.0 / (
                len(TASKS) * identities.size * point_count
            )
    if not np.isclose(weights.sum(), 1.0, atol=1.0e-12, rtol=1.0e-10):
        raise RuntimeError("episode-equal ridge point weights do not sum to one")
    if np.any(weights <= 0.0):
        raise RuntimeError("episode-equal ridge produced a non-positive weight")
    return weights


def _fit_affine_ridge(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover an affine model using the existing A4 closed-form solver."""

    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("ridge feature must be a non-empty matrix")
    if y.ndim != 2 or y.shape != (x.shape[0], TARGET_DIM):
        raise ValueError(f"ridge target must have shape [N,{TARGET_DIM}]")

    # The shared solver returns predictions rather than coefficients.  An
    # affine basis recovers its exact fitted intercept and raw coefficients,
    # allowing large prediction panels to be decoded in bounded chunks.
    feature_dim = int(x.shape[1])
    basis = np.zeros((feature_dim + 1, feature_dim), dtype=np.float64)
    basis[1:] = np.eye(feature_dim, dtype=np.float64)
    basis_prediction = closed_form_ridge_predict(
        x,
        y,
        basis,
        sample_weight=np.asarray(sample_weight, dtype=np.float64),
        alpha=alpha,
    )
    intercept = np.asarray(basis_prediction[0], dtype=np.float64)
    coefficient = np.asarray(basis_prediction[1:] - intercept, dtype=np.float64)
    if coefficient.shape != (feature_dim, TARGET_DIM) or intercept.shape != (
        TARGET_DIM,
    ):
        raise RuntimeError("recovered A4 ridge coefficient axes drifted")
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
        raise RuntimeError("recovered A4 ridge model is non-finite")
    return coefficient, intercept


@dataclass(frozen=True)
class FrozenA4PhysicalDecoder:
    """Immutable train-fitted full-state and anchor-only physical readouts."""

    coefficient: np.ndarray
    intercept: np.ndarray
    anchor_coefficient: np.ndarray
    anchor_intercept: np.ndarray
    target_mean_by_task: np.ndarray
    target_scale_by_task: np.ndarray
    active_target_by_task: np.ndarray
    ridge_alpha: float
    train_bundle_count: int
    train_point_count: int
    train_episode_count_by_task: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficient",
            _readonly_float(
                self.coefficient,
                shape=(FEATURE_DIM, TARGET_DIM),
                label="decoder coefficient",
            ),
        )
        object.__setattr__(
            self,
            "intercept",
            _readonly_float(
                self.intercept,
                shape=(TARGET_DIM,),
                label="decoder intercept",
            ),
        )
        object.__setattr__(
            self,
            "anchor_coefficient",
            _readonly_float(
                self.anchor_coefficient,
                shape=(ANCHOR_FEATURE_DIM, TARGET_DIM),
                label="anchor-only decoder coefficient",
            ),
        )
        object.__setattr__(
            self,
            "anchor_intercept",
            _readonly_float(
                self.anchor_intercept,
                shape=(TARGET_DIM,),
                label="anchor-only decoder intercept",
            ),
        )
        object.__setattr__(
            self,
            "target_mean_by_task",
            _readonly_float(
                self.target_mean_by_task,
                shape=(len(TASKS), TARGET_DIM),
                label="decoder target means",
            ),
        )
        object.__setattr__(
            self,
            "target_scale_by_task",
            _readonly_float(
                self.target_scale_by_task,
                shape=(len(TASKS), TARGET_DIM),
                label="decoder target scales",
            ),
        )
        active = np.asarray(self.active_target_by_task)
        if active.shape != (len(TASKS), TARGET_DIM) or active.dtype.kind != "b":
            raise TypeError(
                "decoder active-target mask must be boolean with shape [7,4]"
            )
        active_copy = np.array(active, dtype=bool, copy=True)
        active_copy.setflags(write=False)
        object.__setattr__(self, "active_target_by_task", active_copy)
        alpha = _positive_float(self.ridge_alpha, label="decoder ridge alpha")
        if alpha != RIDGE_ALPHA:
            raise ValueError(
                f"decoder ridge alpha is source-fixed at {RIDGE_ALPHA}, got {alpha}"
            )
        object.__setattr__(self, "ridge_alpha", alpha)
        object.__setattr__(
            self,
            "train_bundle_count",
            _positive_integer(
                self.train_bundle_count, label="decoder train bundle count"
            ),
        )
        object.__setattr__(
            self,
            "train_point_count",
            _positive_integer(
                self.train_point_count, label="decoder train point count"
            ),
        )
        episode_counts = tuple(
            _positive_integer(value, label="decoder train episode count")
            for value in self.train_episode_count_by_task
        )
        if len(episode_counts) != len(TASKS):
            raise ValueError("decoder requires one train episode count per task")
        object.__setattr__(self, "train_episode_count_by_task", episode_counts)
        if np.any(self.target_scale_by_task <= 0.0):
            raise ValueError("decoder target scales must be strictly positive")
        if not bool(np.all(self.active_target_by_task[:, 0])):
            raise ValueError("task progress must vary in train for every A4 task")
        if not bool(np.all(np.any(self.active_target_by_task[:, 1:], axis=1))):
            raise ValueError(
                "task-controlled position must vary on at least one axis per task"
            )

    def decode_standardized(self, states: np.ndarray) -> np.ndarray:
        value = np.asarray(states)
        if value.ndim < 3 or value.shape[-2:] != (STATE_SLOT_COUNT, LATENT_DIM):
            raise ValueError("A4 latent states must end in [4,192]")
        if value.dtype.kind != "f" or not np.isfinite(value).all():
            raise TypeError("A4 latent states must be finite floating point")
        leading = value.shape[:-2]
        flat = value.astype(np.float64, copy=False).reshape(-1, FEATURE_DIM)
        prediction = flat @ self.coefficient + self.intercept
        if not np.isfinite(prediction).all():
            raise RuntimeError("A4 physical decoder produced non-finite output")
        return prediction.reshape(*leading, TARGET_DIM)

    def decode_anchor_standardized(self, states: np.ndarray) -> np.ndarray:
        value = np.asarray(states)
        if value.ndim < 3 or value.shape[-2:] != (STATE_SLOT_COUNT, LATENT_DIM):
            raise ValueError("A4 latent states must end in [4,192]")
        if value.dtype.kind != "f" or not np.isfinite(value).all():
            raise TypeError("A4 latent states must be finite floating point")
        leading = value.shape[:-2]
        flat = value[..., ANCHOR_START:, :].astype(
            np.float64, copy=False
        ).reshape(-1, ANCHOR_FEATURE_DIM)
        prediction = flat @ self.anchor_coefficient + self.anchor_intercept
        if not np.isfinite(prediction).all():
            raise RuntimeError("A4 anchor-only decoder produced non-finite output")
        return prediction.reshape(*leading, TARGET_DIM)

    def standardize_targets(
        self, target: np.ndarray, task_class_index: np.ndarray
    ) -> np.ndarray:
        value = np.asarray(target, dtype=np.float64)
        task = np.asarray(task_class_index, dtype=np.int64)
        if value.ndim < 2 or value.shape[0] != task.size or value.shape[-1] != TARGET_DIM:
            raise ValueError("physical targets and task indices must align on axis N")
        if np.any((task < 0) | (task >= len(TASKS))):
            raise ValueError("physical target task index must remain in 0..6")
        broadcast = (task.size,) + (1,) * (value.ndim - 2) + (TARGET_DIM,)
        mean = self.target_mean_by_task[task].reshape(broadcast)
        scale = self.target_scale_by_task[task].reshape(broadcast)
        result = (value - mean) / scale
        if not np.isfinite(result).all():
            raise RuntimeError("standardized physical targets are non-finite")
        return result

    def inverse_targets(
        self, standardized: np.ndarray, task_class_index: np.ndarray
    ) -> np.ndarray:
        value = np.asarray(standardized, dtype=np.float64)
        task = np.asarray(task_class_index, dtype=np.int64)
        if value.ndim < 2 or value.shape[-1] != TARGET_DIM:
            raise ValueError("standardized decoder output must end in target dimension 4")
        if value.shape[0] != task.size:
            raise ValueError("decoder output first axis must align with task indices")
        broadcast = (task.size,) + (1,) * (value.ndim - 2) + (TARGET_DIM,)
        return (
            value * self.target_scale_by_task[task].reshape(broadcast)
            + self.target_mean_by_task[task].reshape(broadcast)
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "protocol": np.asarray(PROTOCOL),
            "fit_split": np.asarray("train"),
            "evaluation_split": np.asarray("development"),
            "ridge_implementation": np.asarray(
                "fcpg.lewm_action_branch_a4_metrics.closed_form_ridge_predict"
            ),
            "ridge_alpha": np.asarray(self.ridge_alpha, dtype=np.float64),
            "coefficient": np.asarray(self.coefficient),
            "intercept": np.asarray(self.intercept),
            "anchor_coefficient": np.asarray(self.anchor_coefficient),
            "anchor_intercept": np.asarray(self.anchor_intercept),
            "target_mean_by_task": np.asarray(self.target_mean_by_task),
            "target_scale_by_task": np.asarray(self.target_scale_by_task),
            "active_target_by_task": np.asarray(self.active_target_by_task),
            "task_name": np.asarray(TASKS),
            "target_field": np.asarray(
                ("task_progress", "controlled_x", "controlled_y", "controlled_z")
            ),
            "train_bundle_count": np.asarray(
                self.train_bundle_count, dtype=np.int64
            ),
            "train_point_count": np.asarray(self.train_point_count, dtype=np.int64),
            "train_episode_count_by_task": np.asarray(
                self.train_episode_count_by_task, dtype=np.int64
            ),
        }


def fit_a4_physical_decoder(
    cache: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    ridge_alpha: float = RIDGE_ALPHA,
) -> FrozenA4PhysicalDecoder:
    """Fit the A4 physical readout using formal train rows only."""

    validate_a4_physical_decoding_inputs(cache, labels)
    alpha = _positive_float(ridge_alpha, label="ridge alpha")
    if alpha != RIDGE_ALPHA:
        raise ValueError(
            f"A4 physical decoding fixes ridge alpha at {RIDGE_ALPHA}; "
            "development tuning is prohibited"
        )
    split = np.asarray(cache["split"]).astype(str, copy=False)
    train = split == "train"
    target = _future_physical_targets(labels)[train]
    feature = np.asarray(cache["target_state"])[train].reshape(-1, FEATURE_DIM)
    raw_target = target.reshape(-1, TARGET_DIM)
    repeats = BRANCH_COUNT * HORIZON
    task = np.repeat(
        np.asarray(cache["task_class_index"], dtype=np.int64)[train], repeats
    )
    episode = np.repeat(
        np.asarray(cache["source_episode_id"])[train].astype(str), repeats
    )
    weights = _episode_equal_point_weights(task, episode)

    means = np.empty((len(TASKS), TARGET_DIM), dtype=np.float64)
    scales = np.empty_like(means)
    active = np.empty_like(means, dtype=bool)
    standardized = np.empty_like(raw_target, dtype=np.float64)
    episode_counts: list[int] = []
    for task_index, task_name in enumerate(TASKS):
        selected = task == task_index
        local_weight = weights[selected]
        local_weight = local_weight / local_weight.sum()
        values = raw_target[selected]
        mean = np.sum(local_weight[:, None] * values, axis=0)
        variance = np.sum(
            local_weight[:, None] * np.square(values - mean), axis=0
        )
        scale = np.sqrt(np.maximum(variance, 0.0))
        informative = scale > NUMERIC_EPSILON
        safe_scale = np.where(informative, scale, 1.0)
        means[task_index] = mean
        scales[task_index] = safe_scale
        active[task_index] = informative
        standardized[selected] = (values - mean) / safe_scale
        episode_counts.append(int(np.unique(episode[selected]).size))
        if not informative[0]:
            raise ValueError(
                f"train task-progress label has zero variance for {task_name}"
            )
        if not np.any(informative[1:]):
            raise ValueError(
                f"train controlled-position label has zero variance for {task_name}"
            )

    coefficient, intercept = _fit_affine_ridge(
        feature,
        standardized,
        sample_weight=weights,
        alpha=alpha,
    )
    anchor_coefficient, anchor_intercept = _fit_affine_ridge(
        np.asarray(cache["target_state"])[train, ..., ANCHOR_START:, :].reshape(
            -1, ANCHOR_FEATURE_DIM
        ),
        standardized,
        sample_weight=weights,
        alpha=alpha,
    )
    return FrozenA4PhysicalDecoder(
        coefficient=coefficient,
        intercept=intercept,
        anchor_coefficient=anchor_coefficient,
        anchor_intercept=anchor_intercept,
        target_mean_by_task=means,
        target_scale_by_task=scales,
        active_target_by_task=active,
        ridge_alpha=alpha,
        train_bundle_count=int(np.count_nonzero(train)),
        train_point_count=int(feature.shape[0]),
        train_episode_count_by_task=tuple(episode_counts),
    )


@dataclass(frozen=True)
class DecodedA4Seed:
    seed: int
    factual: np.ndarray
    anchor_only_factual: np.ndarray
    matched_no_action: np.ndarray
    wrong_action: np.ndarray

    def __post_init__(self) -> None:
        seed = _positive_integer(self.seed, label="decoded checkpoint seed")
        object.__setattr__(self, "seed", seed)
        factual = np.asarray(self.factual, dtype=np.float64)
        expected_tail = (BRANCH_COUNT, HORIZON, TARGET_DIM)
        if factual.ndim != 4 or factual.shape[1:] != expected_tail:
            raise ValueError("decoded factual panel must have shape [N,8,8,4]")
        count = factual.shape[0]
        expected = {
            "factual": (count, *expected_tail),
            "anchor_only_factual": (count, *expected_tail),
            "matched_no_action": (count, *expected_tail),
            "wrong_action": (
                count,
                WRONG_ACTION_SHIFT_COUNT,
                *expected_tail,
            ),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"decoded {name} must be finite with shape {shape}")
            copied = np.array(value, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def decode_a4_evaluation_arrays(
    decoder: FrozenA4PhysicalDecoder,
    cache: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
) -> DecodedA4Seed:
    """Validate and decode one frozen 10k development prediction panel."""

    if not isinstance(decoder, FrozenA4PhysicalDecoder):
        raise TypeError("decoder must be a FrozenA4PhysicalDecoder")
    validate_a4_state_cache(cache)
    split = np.asarray(cache["split"]).astype(str, copy=False)
    if np.any(split == "test"):
        raise PermissionError("A4 physical decoder refuses reserved-test cache rows")
    development_rows = np.flatnonzero(split == "development")
    count = int(development_rows.size)
    if count == 0:
        raise ValueError("A4 state cache has no development rows")
    values = dict(arrays)
    required = {
        "checkpoint_step",
        "selection",
        "training_seed",
        "sample_index",
        "target_state",
        "factual_prediction",
        "no_action_prediction",
        "shuffled_action_prediction",
        "shuffled_action_donor_shifts",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"A4 evaluation arrays are missing fields: {missing}")
    if int(_scalar(values["checkpoint_step"], label="checkpoint_step")) != 10_000:
        raise ValueError("A4 physical decoding requires checkpoint step exactly 10000")
    if str(_scalar(values["selection"], label="selection")) != "fixed_latest":
        raise ValueError("A4 physical decoding requires fixed_latest selection")
    sample_index = np.asarray(values["sample_index"], dtype=np.int64)
    if not np.array_equal(sample_index, development_rows):
        raise ValueError(
            "A4 evaluation sample_index must cover development rows in cache order"
        )
    seed_values = np.asarray(values["training_seed"], dtype=np.int64)
    if seed_values.shape != (count,) or np.unique(seed_values).size != 1:
        raise ValueError("A4 evaluation training_seed must be one constant row vector")
    seed = int(seed_values[0])

    expected = (count, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM)
    target = np.asarray(values["target_state"])
    factual = np.asarray(values["factual_prediction"])
    no_action = np.asarray(values["no_action_prediction"])
    for name, value in (
        ("target_state", target),
        ("factual_prediction", factual),
        ("no_action_prediction", no_action),
    ):
        if value.shape != expected or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"A4 {name} must be finite with shape {expected}")
    if not np.array_equal(target, np.asarray(cache["target_state"])[development_rows]):
        raise ValueError("A4 evaluation target_state differs from formal_v2 cache")
    wrong = np.asarray(values["shuffled_action_prediction"])
    expected_wrong = (count, WRONG_ACTION_SHIFT_COUNT, *expected[1:])
    if wrong.shape != expected_wrong or wrong.dtype.kind != "f" or not np.isfinite(wrong).all():
        raise ValueError(
            f"A4 wrong-action prediction must be finite with shape {expected_wrong}"
        )
    shifts = np.asarray(values["shuffled_action_donor_shifts"], dtype=np.int64)
    if not np.array_equal(shifts, np.arange(1, BRANCH_COUNT, dtype=np.int64)):
        raise ValueError("A4 wrong-action donor shifts must remain exactly 1..7")
    validate_wrong_action_prediction_replay(factual[None], wrong[None])
    no_action_spread = float(
        np.max(np.abs(no_action - no_action[:, :1]), initial=0.0)
    )
    if no_action_spread > 1.0e-6:
        raise ValueError(
            "matched no-action prediction must remain branch invariant; "
            f"max_abs={no_action_spread}"
        )
    return DecodedA4Seed(
        seed=seed,
        factual=decoder.decode_standardized(factual),
        anchor_only_factual=decoder.decode_anchor_standardized(factual),
        matched_no_action=decoder.decode_standardized(no_action),
        wrong_action=decoder.decode_standardized(wrong),
    )


def _one_hot(values: np.ndarray, count: int, *, label: str) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if np.any((indices < 0) | (indices >= count)):
        raise ValueError(f"{label} indices must remain in 0..{count - 1}")
    return np.eye(count, dtype=np.float32)[indices]


def _controlled_q0(
    labels: Mapping[str, np.ndarray], task_class_index: np.ndarray
) -> np.ndarray:
    """Return the real H0 state matching each task's future controlled xyz."""

    task = np.asarray(task_class_index, dtype=np.int64)
    object_pose = np.asarray(labels["object_pose"], dtype=np.float32)
    proprio = np.asarray(labels["proprio"], dtype=np.float32)
    if object_pose.shape != (task.size, 7) or proprio.shape != (task.size, 4):
        raise ValueError("H0 object-pose/proprio labels are not row aligned")
    result = np.array(object_pose[:, :3], dtype=np.float32, copy=True)
    reach_index = TASKS.index("reach-v3")
    result[task == reach_index] = proprio[task == reach_index, :3]
    if not np.isfinite(result).all():
        raise ValueError("controlled q0 contains non-finite values")
    return result


def _build_task_time_features(task_class_index: np.ndarray) -> np.ndarray:
    task = np.asarray(task_class_index, dtype=np.int64)
    task_one_hot = _one_hot(task, len(TASKS), label="task-time task")
    task_panel = np.broadcast_to(
        task_one_hot[:, None, None, :],
        (task.size, BRANCH_COUNT, HORIZON, len(TASKS)),
    )
    horizon = np.eye(HORIZON, dtype=np.float32)
    horizon_panel = np.broadcast_to(
        horizon[None, None, :, :],
        (task.size, BRANCH_COUNT, HORIZON, HORIZON),
    )
    return np.concatenate((task_panel, horizon_panel), axis=-1)


def _build_matched_direct_action_features(
    cache: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    rows: np.ndarray,
) -> np.ndarray:
    """Build the source-frozen 768-D DirectAction comparison contract."""

    selected = np.asarray(rows)
    task = np.asarray(cache["task_class_index"], dtype=np.int64)[selected]
    history_state = np.asarray(cache["history_state"], dtype=np.float32)[selected]
    history_action = np.asarray(cache["history_action"], dtype=np.float32)[selected]
    center_action = np.asarray(cache["center_action"], dtype=np.float32)[selected]
    future_action = np.asarray(cache["future_action"], dtype=np.float32)[selected]
    semantic = np.asarray(cache["branch_semantic_index"], dtype=np.int64)[selected]
    proprio = np.asarray(labels["proprio"], dtype=np.float32)[selected]
    q0 = _controlled_q0(labels, np.asarray(cache["task_class_index"]))[selected]
    count = int(task.size)
    if history_state.shape != (
        count,
        3,
        STATE_SLOT_COUNT,
        LATENT_DIM,
    ):
        raise ValueError("DirectAction history_state shape drifted")
    if history_action.shape != (count, 2, ACTION_DIM):
        raise ValueError("DirectAction history_action shape drifted")
    if center_action.shape != (count, HORIZON, ACTION_DIM):
        raise ValueError("DirectAction center_action shape drifted")
    if future_action.shape != (count, BRANCH_COUNT, HORIZON, ACTION_DIM):
        raise ValueError("DirectAction future_action shape drifted")
    expected_semantic = np.arange(BRANCH_COUNT, dtype=np.int64)
    if semantic.shape != (count, BRANCH_COUNT) or not np.all(
        np.sort(semantic, axis=1) == expected_semantic
    ):
        raise ValueError("DirectAction branch semantics must be row-wise 0..7 permutations")
    if proprio.shape != (count, 4) or not np.isfinite(proprio).all():
        raise ValueError("DirectAction H0 proprio shape drifted")

    panel_shape = (count, BRANCH_COUNT, HORIZON)

    def repeated(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            value[:, None, None, :], (*panel_shape, value.shape[-1])
        )

    latest_anchor = history_state[:, -1, ANCHOR_START:, :].reshape(
        count, ANCHOR_FEATURE_DIM
    )
    past_action = history_action.reshape(count, 2 * ACTION_DIM)
    past_delta = history_action[:, 1] - history_action[:, 0]
    cumulative = np.cumsum(future_action, axis=2, dtype=np.float32)
    cumulative_square = np.cumsum(
        np.square(future_action), axis=2, dtype=np.float32
    )
    first_action = np.broadcast_to(
        future_action[:, :, :1, :], future_action.shape
    )
    task_one_hot = repeated(_one_hot(task, len(TASKS), label="DirectAction task"))
    horizon_one_hot = np.broadcast_to(
        np.eye(HORIZON, dtype=np.float32)[None, None, :, :],
        (*panel_shape, HORIZON),
    )
    semantic_one_hot = np.broadcast_to(
        _one_hot(semantic, BRANCH_COUNT, label="DirectAction branch semantic")[:, :, None, :],
        (*panel_shape, BRANCH_COUNT),
    )
    cumulative_norm = np.sqrt(
        np.sum(cumulative_square, axis=-1, keepdims=True, dtype=np.float32)
    )
    center_panel = np.broadcast_to(
        center_action[:, None, :, :], (*panel_shape, ACTION_DIM)
    )
    horizon_scalar = np.broadcast_to(
        (
            np.arange(1, HORIZON + 1, dtype=np.float32)
            / np.float32(HORIZON)
        )[None, None, :, None],
        (*panel_shape, 1),
    )
    result = np.concatenate(
        (
            repeated(latest_anchor),
            repeated(past_action),
            future_action,
            cumulative,
            first_action,
            cumulative_square,
            repeated(q0),
            repeated(proprio),
            task_one_hot,
            horizon_one_hot,
            semantic_one_hot,
            cumulative_norm,
            repeated(past_delta),
            center_panel,
            horizon_scalar,
        ),
        axis=-1,
    )
    if result.shape != (*panel_shape, MATCHED_DIRECT_ACTION_FEATURE_DIM):
        raise RuntimeError(
            "matched DirectAction feature dimension drifted: "
            f"expected {MATCHED_DIRECT_ACTION_FEATURE_DIM}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise RuntimeError("matched DirectAction features are non-finite")
    return result


@dataclass(frozen=True)
class FixedA4Baselines:
    task_time: np.ndarray
    matched_direct_action: np.ndarray
    latent_hold: np.ndarray
    q0_position_raw: np.ndarray
    q0_position_standardized: np.ndarray

    def __post_init__(self) -> None:
        task_time = np.asarray(self.task_time, dtype=np.float64)
        expected = task_time.shape
        if task_time.ndim != 4 or task_time.shape[1:] != (
            BRANCH_COUNT,
            HORIZON,
            TARGET_DIM,
        ):
            raise ValueError("fixed baseline panels must have shape [N,8,8,4]")
        for name in ("task_time", "matched_direct_action", "latent_hold"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f"{name} baseline shape or values are invalid")
            copied = np.array(value, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        for name in ("q0_position_raw", "q0_position_standardized"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != expected:
                raise ValueError(f"{name} baseline shape is invalid")
            if not np.isnan(value[..., 0]).all() or not np.isfinite(value[..., 1:]).all():
                raise ValueError(
                    f"{name} must use NaN only for unsupported task progress"
                )
            copied = np.array(value, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def fit_fixed_a4_baselines(
    cache: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    decoder: FrozenA4PhysicalDecoder,
) -> FixedA4Baselines:
    """Fit fixed train-only comparison ridges and construct non-fitted holds."""

    validate_a4_physical_decoding_inputs(cache, labels)
    if not isinstance(decoder, FrozenA4PhysicalDecoder):
        raise TypeError("decoder must be a FrozenA4PhysicalDecoder")
    split = np.asarray(cache["split"]).astype(str, copy=False)
    train = split == "train"
    development = split == "development"
    task_all = np.asarray(cache["task_class_index"], dtype=np.int64)
    episode_all = np.asarray(cache["source_episode_id"]).astype(str, copy=False)
    standardized_target = decoder.standardize_targets(
        _future_physical_targets(labels), task_all
    )
    train_target = standardized_target[train].reshape(-1, TARGET_DIM)
    repeated_task = np.repeat(task_all[train], BRANCH_COUNT * HORIZON)
    repeated_episode = np.repeat(
        episode_all[train], BRANCH_COUNT * HORIZON
    )
    weights = _episode_equal_point_weights(repeated_task, repeated_episode)

    task_time_train = _build_task_time_features(task_all[train])
    task_time_development = _build_task_time_features(task_all[development])
    task_time = closed_form_ridge_predict(
        task_time_train.reshape(-1, task_time_train.shape[-1]),
        train_target,
        task_time_development.reshape(-1, task_time_development.shape[-1]),
        sample_weight=weights,
        alpha=RIDGE_ALPHA,
    ).reshape(-1, BRANCH_COUNT, HORIZON, TARGET_DIM)

    direct_train = _build_matched_direct_action_features(cache, labels, train)
    direct_development = _build_matched_direct_action_features(
        cache, labels, development
    )
    matched_direct_action = closed_form_ridge_predict(
        direct_train.reshape(-1, MATCHED_DIRECT_ACTION_FEATURE_DIM),
        train_target,
        direct_development.reshape(-1, MATCHED_DIRECT_ACTION_FEATURE_DIM),
        sample_weight=weights,
        alpha=RIDGE_ALPHA,
    ).reshape(-1, BRANCH_COUNT, HORIZON, TARGET_DIM)

    history_last = np.asarray(cache["history_state"])[development, -1]
    hold_state = np.broadcast_to(
        history_last[:, None, None, :, :],
        (
            history_last.shape[0],
            BRANCH_COUNT,
            HORIZON,
            STATE_SLOT_COUNT,
            LATENT_DIM,
        ),
    )
    latent_hold = decoder.decode_standardized(hold_state)

    development_task = task_all[development]
    q0 = _controlled_q0(labels, task_all)[development]
    q0_raw = np.full(
        (q0.shape[0], BRANCH_COUNT, HORIZON, TARGET_DIM),
        np.nan,
        dtype=np.float64,
    )
    q0_raw[..., 1:] = q0[:, None, None, :]
    q0_standardized = np.full_like(q0_raw, np.nan)
    q0_standardized[..., 1:] = (
        q0[:, None, None, :]
        - decoder.target_mean_by_task[development_task, 1:][:, None, None, :]
    ) / decoder.target_scale_by_task[development_task, 1:][:, None, None, :]
    return FixedA4Baselines(
        task_time=task_time,
        matched_direct_action=matched_direct_action,
        latent_hold=latent_hold,
        q0_position_raw=q0_raw,
        q0_position_standardized=q0_standardized,
    )


@dataclass(frozen=True)
class _EpisodeReduction:
    report: Mapping[str, Any]
    standardized_episode_values: Mapping[str, np.ndarray]
    raw_episode_values: Mapping[str, np.ndarray]
    episode_ids: Mapping[str, tuple[str, ...]]
    seeds: tuple[int, ...] | None


def _seed_panel_inverse(
    decoder: FrozenA4PhysicalDecoder,
    standardized: np.ndarray,
    task: np.ndarray,
) -> np.ndarray:
    value = np.asarray(standardized, dtype=np.float64)
    task_index = np.asarray(task, dtype=np.int64)
    if value.ndim != 5 or value.shape[1] != task_index.size or value.shape[-1] != TARGET_DIM:
        raise ValueError("seed decoder panel must have shape [S,N,8,8,4]")
    mean = decoder.target_mean_by_task[task_index][None, :, None, None, :]
    scale = decoder.target_scale_by_task[task_index][None, :, None, None, :]
    result = value * scale + mean
    if not np.isfinite(result).all():
        raise RuntimeError("inverse-decoded A4 physical panel is non-finite")
    return result


def _readout_losses(
    prediction_standardized: np.ndarray,
    prediction_raw: np.ndarray,
    truth_standardized: np.ndarray,
    truth_raw: np.ndarray,
    *,
    task: np.ndarray,
    decoder: FrozenA4PhysicalDecoder,
    spec: ReadoutSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction_standardized, dtype=np.float64)
    prediction_units = np.asarray(prediction_raw, dtype=np.float64)
    if prediction.ndim != 5 or prediction.shape != prediction_units.shape:
        raise ValueError("readout predictions must share shape [S,N,8,8,4]")
    if truth_standardized.shape != prediction.shape[1:] or truth_raw.shape != prediction.shape[1:]:
        raise ValueError("readout truth must align with seed prediction panels")
    task_index = np.asarray(task, dtype=np.int64)
    selected_rows = np.isin(
        np.asarray(TASKS, dtype=str)[task_index], np.asarray(spec.supported_tasks)
    )
    if not np.any(selected_rows):
        raise ValueError(f"readout {spec.name} has no supported development rows")
    dimensions = np.asarray(spec.dimensions, dtype=np.int64)
    active = decoder.active_target_by_task[task_index][:, dimensions]
    if np.any(selected_rows & ~np.any(active, axis=1)):
        missing = sorted(
            set(np.asarray(TASKS)[task_index[selected_rows & ~np.any(active, axis=1)]])
        )
        raise ValueError(
            f"readout {spec.name} has no variable train dimension for tasks {missing}"
        )
    mask = active[None, :, None, None, :]
    denominator = np.sum(active, axis=1, dtype=np.float64)[None, :, None, None]
    standardized_error = prediction[..., dimensions] - truth_standardized[None, ..., dimensions]
    raw_error = prediction_units[..., dimensions] - truth_raw[None, ..., dimensions]
    standardized_loss = np.sum(
        np.square(standardized_error) * mask, axis=-1, dtype=np.float64
    ) / denominator
    raw_loss = np.sum(
        np.square(raw_error) * mask, axis=-1, dtype=np.float64
    ) / denominator
    if not np.isfinite(standardized_loss).all() or not np.isfinite(raw_loss).all():
        raise RuntimeError(f"readout {spec.name} produced non-finite loss")
    return standardized_loss, raw_loss, selected_rows


def _reduce_method(
    standardized_loss: np.ndarray,
    raw_loss: np.ndarray,
    *,
    task: np.ndarray,
    episode: np.ndarray,
    selected_rows: np.ndarray,
    horizon_indices: Sequence[int],
    seeds: Sequence[int] | None,
    unit: str,
) -> _EpisodeReduction:
    standardized = np.asarray(standardized_loss, dtype=np.float64)
    raw = np.asarray(raw_loss, dtype=np.float64)
    if standardized.shape != raw.shape or standardized.ndim != 4:
        raise ValueError("method loss must share shape [S,N,8,8]")
    indices = np.asarray(tuple(horizon_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0 or np.any((indices < 0) | (indices >= HORIZON)):
        raise ValueError("horizon indices must be a non-empty subset of 0..7")
    seed_values = tuple(int(value) for value in seeds) if seeds is not None else ()
    if seeds is None:
        if standardized.shape[0] != 1:
            raise ValueError("unseeded method must have exactly one prediction panel")
    elif len(seed_values) != standardized.shape[0] or len(set(seed_values)) != len(seed_values):
        raise ValueError("method seeds must be distinct and align with loss axis S")

    row_standardized = np.mean(
        standardized[:, :, :, indices], axis=(2, 3), dtype=np.float64
    )
    row_raw = np.mean(raw[:, :, :, indices], axis=(2, 3), dtype=np.float64)
    task_index = np.asarray(task, dtype=np.int64)
    episode_id = np.asarray(episode).astype(str, copy=False)
    task_reports: dict[str, Any] = {}
    standardized_episode: dict[str, np.ndarray] = {}
    raw_episode: dict[str, np.ndarray] = {}
    episode_ids: dict[str, tuple[str, ...]] = {}
    for index, task_name in enumerate(TASKS):
        selected_task = selected_rows & (task_index == index)
        if not np.any(selected_task):
            continue
        identities = tuple(sorted(np.unique(episode_id[selected_task]).tolist()))
        episode_ids[task_name] = identities
        per_seed_standardized = np.stack(
            [
                np.mean(row_standardized[:, selected_task & (episode_id == identity)], axis=1)
                for identity in identities
            ],
            axis=1,
        )
        per_seed_raw = np.stack(
            [
                np.mean(row_raw[:, selected_task & (episode_id == identity)], axis=1)
                for identity in identities
            ],
            axis=1,
        )
        standardized_episode[task_name] = per_seed_standardized
        raw_episode[task_name] = per_seed_raw
        seed_task_report = (
            {
                str(seed): {
                    "standardized_mse": float(
                        np.mean(per_seed_standardized[seed_index])
                    ),
                    "raw_mse": float(np.mean(per_seed_raw[seed_index])),
                    "raw_rmse": float(
                        np.sqrt(np.mean(per_seed_raw[seed_index]))
                    ),
                }
                for seed_index, seed in enumerate(seed_values)
            }
            if seeds is not None
            else None
        )
        task_reports[task_name] = {
            "physical_episode_count": len(identities),
            "standardized_mse": float(np.mean(per_seed_standardized)),
            "raw_mse": float(np.mean(per_seed_raw)),
            "raw_rmse": float(np.sqrt(np.mean(per_seed_raw))),
        }
        if seed_task_report is not None:
            task_reports[task_name]["per_seed"] = seed_task_report

    if not task_reports:
        raise ValueError("method reduction has no supported tasks")
    task_macro_standardized = float(
        np.mean([row["standardized_mse"] for row in task_reports.values()])
    )
    task_macro_raw = float(np.mean([row["raw_mse"] for row in task_reports.values()]))
    report: dict[str, Any] = {
        "aggregation": "task_then_source_physical_episode",
        "seed_aggregation": (
            "per_seed_losses_first; predictions_are_never_averaged"
            if seeds is not None
            else None
        ),
        "seed_count": len(seed_values) if seeds is not None else 0,
        "task_count": len(task_reports),
        "physical_episode_count": int(
            sum(row["physical_episode_count"] for row in task_reports.values())
        ),
        "task_macro_standardized_mse": task_macro_standardized,
        "task_macro_raw_mse": task_macro_raw,
        "task_macro_raw_rmse": float(np.sqrt(task_macro_raw)),
        "raw_unit": unit,
        "per_task": task_reports,
    }
    if seeds is not None:
        per_seed: dict[str, Any] = {}
        for seed_index, seed in enumerate(seed_values):
            seed_task_standardized = [
                float(np.mean(values[seed_index]))
                for values in standardized_episode.values()
            ]
            seed_task_raw = [
                float(np.mean(values[seed_index]))
                for values in raw_episode.values()
            ]
            seed_macro_raw = float(np.mean(seed_task_raw))
            per_seed[str(seed)] = {
                "task_macro_standardized_mse": float(
                    np.mean(seed_task_standardized)
                ),
                "task_macro_raw_mse": seed_macro_raw,
                "task_macro_raw_rmse": float(np.sqrt(seed_macro_raw)),
            }
        report["per_seed"] = per_seed
    return _EpisodeReduction(
        report=report,
        standardized_episode_values=standardized_episode,
        raw_episode_values=raw_episode,
        episode_ids=episode_ids,
        seeds=seed_values if seeds is not None else None,
    )


def _comparison(
    comparison: _EpisodeReduction,
    baseline: _EpisodeReduction,
    *,
    comparison_name: str,
    baseline_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    samples = _positive_integer(bootstrap_samples, label="bootstrap samples")
    seed = _nonnegative_integer(bootstrap_seed, label="bootstrap seed")
    if tuple(comparison.episode_ids) != tuple(baseline.episode_ids):
        raise ValueError("paired comparison task panels differ")
    if (
        comparison.seeds is not None
        and baseline.seeds is not None
        and comparison.seeds != baseline.seeds
    ):
        raise ValueError("paired comparison LeWM seed identities differ")
    crossed_seeds = comparison.seeds or baseline.seeds
    seed_count = len(crossed_seeds) if crossed_seeds is not None else 1
    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(
        0, seed_count, size=(samples, seed_count)
    )
    task_draws: list[np.ndarray] = []
    per_task: dict[str, Any] = {}
    raw_differences = []
    relative_improvements = []
    per_seed_standardized: dict[str, list[float]] = {
        str(value): [] for value in crossed_seeds or ()
    }
    per_seed_raw: dict[str, list[float]] = {
        str(value): [] for value in crossed_seeds or ()
    }

    def expanded(value: np.ndarray, episode_count: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (seed_count, episode_count):
            return array
        if array.shape == (1, episode_count):
            return np.broadcast_to(array, (seed_count, episode_count))
        raise ValueError(
            "paired comparison seed/physical-episode axes cannot be crossed"
        )

    for task_name in comparison.episode_ids:
        left_ids = comparison.episode_ids[task_name]
        right_ids = baseline.episode_ids.get(task_name)
        if left_ids != right_ids:
            raise ValueError(
                f"paired comparison physical-episode identities differ for {task_name}"
            )
        episode_count = len(left_ids)
        left = expanded(
            comparison.standardized_episode_values[task_name], episode_count
        )
        right = expanded(
            baseline.standardized_episode_values[task_name], episode_count
        )
        left_raw = expanded(comparison.raw_episode_values[task_name], episode_count)
        right_raw = expanded(baseline.raw_episode_values[task_name], episode_count)
        episode_indices = rng.integers(
            0, episode_count, size=(samples, episode_count)
        )
        difference_panel = right - left
        draw = np.mean(
            difference_panel[
                seed_indices[:, :, None], episode_indices[:, None, :]
            ],
            axis=(1, 2),
            dtype=np.float64,
        )
        task_draws.append(draw)
        comparison_mse = float(np.mean(left))
        baseline_mse = float(np.mean(right))
        difference = baseline_mse - comparison_mse
        relative = (
            1.0 - comparison_mse / baseline_mse
            if baseline_mse > NUMERIC_EPSILON
            else (0.0 if comparison_mse <= NUMERIC_EPSILON else -math.inf)
        )
        if not math.isfinite(relative):
            raise ValueError(
                f"{task_name} comparison baseline MSE is zero with positive comparison MSE"
            )
        per_task[task_name] = {
            "comparison_standardized_mse": comparison_mse,
            "baseline_standardized_mse": baseline_mse,
            "standardized_mse_difference": difference,
            "relative_improvement": float(relative),
            "crossed_95_ci": {
                "lower": float(np.quantile(draw, 0.025)),
                "upper": float(np.quantile(draw, 0.975)),
            },
        }
        if crossed_seeds is not None:
            for seed_index, seed_name in enumerate(crossed_seeds):
                per_seed_standardized[str(seed_name)].append(
                    float(np.mean(difference_panel[seed_index]))
                )
                per_seed_raw[str(seed_name)].append(
                    float(np.mean(right_raw[seed_index] - left_raw[seed_index]))
                )
        relative_improvements.append(float(relative))
        raw_differences.append(float(np.mean(right_raw) - np.mean(left_raw)))

    distribution = np.mean(np.stack(task_draws, axis=1), axis=1)
    estimate = float(
        np.mean(
            [row["standardized_mse_difference"] for row in per_task.values()]
        )
    )
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    result = {
        "comparison_method": comparison_name,
        "baseline_method": baseline_name,
        "orientation": "positive_means_comparison_has_lower_mse",
        "task_macro_standardized_mse_difference": estimate,
        "task_macro_raw_mse_difference": float(np.mean(raw_differences)),
        "task_macro_relative_improvement": float(
            np.mean(relative_improvements)
        ),
        "positive_task_count": int(
            sum(row["standardized_mse_difference"] > 0.0 for row in per_task.values())
        ),
        "task_count": len(per_task),
        "per_task": per_task,
        "crossed_seed_physical_episode_bootstrap": {
            "unit": "lewm_seed_x_source_physical_episode_within_task",
            "task_macro": True,
            "paired": True,
            "seed_resampling": crossed_seeds is not None,
            "lewm_seed_count": len(crossed_seeds or ()),
            "physical_episode_resampling": "independent_within_each_task",
            "crossed_product_reduction": True,
            "samples": samples,
            "seed": seed,
            "confidence": 0.95,
            "lower": float(lower),
            "upper": float(upper),
            "probability_positive": float(np.mean(distribution > 0.0)),
        },
    }
    if crossed_seeds is not None:
        result["per_seed"] = {
            seed_name: {
                "task_macro_standardized_mse_difference": float(
                    np.mean(per_seed_standardized[seed_name])
                ),
                "task_macro_raw_mse_difference": float(
                    np.mean(per_seed_raw[seed_name])
                ),
                "positive_task_count": int(
                    np.count_nonzero(
                        np.asarray(per_seed_standardized[seed_name]) > 0.0
                    )
                ),
                "task_count": len(per_seed_standardized[seed_name]),
            }
            for seed_name in per_seed_standardized
        }
    return result


def _comparison_seed(base: int, *parts: str) -> int:
    token = "\x1f".join(parts).encode("utf-8")
    return int(base + zlib.crc32(token))


def _score_readout_window(
    *,
    decoder: FrozenA4PhysicalDecoder,
    spec: ReadoutSpec,
    truth_standardized: np.ndarray,
    truth_raw: np.ndarray,
    task: np.ndarray,
    episode: np.ndarray,
    target_ceiling: np.ndarray,
    factual: np.ndarray,
    anchor_only: np.ndarray,
    no_action: np.ndarray,
    wrong_action: np.ndarray,
    baselines: FixedA4Baselines,
    seeds: tuple[int, ...],
    horizon_indices: tuple[int, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
    window_name: str,
) -> dict[str, Any]:
    panels: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, ...] | None]] = {
        "target_latent_ceiling": (
            target_ceiling[None],
            _seed_panel_inverse(decoder, target_ceiling[None], task),
            None,
        ),
        "factual": (
            factual,
            _seed_panel_inverse(decoder, factual, task),
            seeds,
        ),
        "anchor_only_factual": (
            anchor_only,
            _seed_panel_inverse(decoder, anchor_only, task),
            seeds,
        ),
        "matched_no_action": (
            no_action,
            _seed_panel_inverse(decoder, no_action, task),
            seeds,
        ),
        "matched_direct_action": (
            baselines.matched_direct_action[None],
            _seed_panel_inverse(
                decoder, baselines.matched_direct_action[None], task
            ),
            None,
        ),
        "task_time": (
            baselines.task_time[None],
            _seed_panel_inverse(decoder, baselines.task_time[None], task),
            None,
        ),
        "latent_hold": (
            baselines.latent_hold[None],
            _seed_panel_inverse(decoder, baselines.latent_hold[None], task),
            None,
        ),
    }
    q0_available = 0 not in spec.dimensions
    if q0_available:
        panels["q0_physical_hold"] = (
            baselines.q0_position_standardized[None],
            baselines.q0_position_raw[None],
            None,
        )

    reductions: dict[str, _EpisodeReduction] = {}
    selected_rows: np.ndarray | None = None
    for method_name, (standardized_panel, raw_panel, method_seeds) in panels.items():
        method_loss, method_raw_loss, method_selected = _readout_losses(
            standardized_panel,
            raw_panel,
            truth_standardized,
            truth_raw,
            task=task,
            decoder=decoder,
            spec=spec,
        )
        if selected_rows is None:
            selected_rows = method_selected
        elif not np.array_equal(selected_rows, method_selected):
            raise RuntimeError("readout method task scope drifted")
        reductions[method_name] = _reduce_method(
            method_loss,
            method_raw_loss,
            task=task,
            episode=episode,
            selected_rows=method_selected,
            horizon_indices=horizon_indices,
            seeds=method_seeds,
            unit=spec.unit,
        )
    if selected_rows is None:
        raise RuntimeError("readout scoring did not construct a task mask")

    wrong_reductions: list[_EpisodeReduction] = []
    wrong_reports: list[dict[str, Any]] = []
    wrong_loss_by_shift: list[np.ndarray] = []
    wrong_raw_loss_by_shift: list[np.ndarray] = []
    for shift_index in range(WRONG_ACTION_SHIFT_COUNT):
        shifted = wrong_action[:, :, shift_index]
        shifted_raw = _seed_panel_inverse(decoder, shifted, task)
        shift_loss, shift_raw_loss, _ = _readout_losses(
            shifted,
            shifted_raw,
            truth_standardized,
            truth_raw,
            task=task,
            decoder=decoder,
            spec=spec,
        )
        reduction = _reduce_method(
            shift_loss,
            shift_raw_loss,
            task=task,
            episode=episode,
            selected_rows=selected_rows,
            horizon_indices=horizon_indices,
            seeds=seeds,
            unit=spec.unit,
        )
        wrong_reductions.append(reduction)
        wrong_loss_by_shift.append(shift_loss)
        wrong_raw_loss_by_shift.append(shift_raw_loss)
        wrong_reports.append(
            {"donor_shift": shift_index + 1, **dict(reduction.report)}
        )
    wrong_mean_reduction = _reduce_method(
        np.mean(np.stack(wrong_loss_by_shift, axis=2), axis=2),
        np.mean(np.stack(wrong_raw_loss_by_shift, axis=2), axis=2),
        task=task,
        episode=episode,
        selected_rows=selected_rows,
        horizon_indices=horizon_indices,
        seeds=seeds,
        unit=spec.unit,
    )

    factual_reduction = reductions["factual"]

    def compare(
        comparison_name: str, baseline_name: str
    ) -> dict[str, Any]:
        return _comparison(
            reductions[comparison_name],
            reductions[baseline_name],
            comparison_name=comparison_name,
            baseline_name=baseline_name,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=_comparison_seed(
                bootstrap_seed,
                window_name,
                spec.name,
                comparison_name,
                baseline_name,
            ),
        )

    baseline_comparisons = {
        name: compare("factual", name)
        for name in (
            "matched_no_action",
            "matched_direct_action",
            "task_time",
            "anchor_only_factual",
            "latent_hold",
        )
    }
    if q0_available:
        baseline_comparisons["q0_physical_hold"] = compare(
            "factual", "q0_physical_hold"
        )
    wrong_comparisons = [
        {
            "donor_shift": shift_index + 1,
            **_comparison(
                factual_reduction,
                reduction,
                comparison_name="factual",
                baseline_name=f"wrong_action_shift_{shift_index + 1}",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_comparison_seed(
                    bootstrap_seed,
                    window_name,
                    spec.name,
                    f"wrong_action_shift_{shift_index + 1}",
                ),
            ),
        }
        for shift_index, reduction in enumerate(wrong_reductions)
    ]
    ceiling_comparison = _comparison(
        reductions["target_latent_ceiling"],
        factual_reduction,
        comparison_name="target_latent_ceiling",
        baseline_name="factual",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=_comparison_seed(
            bootstrap_seed, window_name, spec.name, "target_latent_ceiling"
        ),
    )

    def descriptive_observation(value: Mapping[str, Any]) -> dict[str, Any]:
        interval = value["crossed_seed_physical_episode_bootstrap"]
        estimate = float(value["task_macro_standardized_mse_difference"])
        return {
            "orientation": value["orientation"],
            "point_estimate": estimate,
            "point_direction": (
                "comparison_lower_mse"
                if estimate > 0.0
                else "baseline_lower_mse"
                if estimate < 0.0
                else "equal"
            ),
            "crossed_95_ci": {
                "lower": interval["lower"],
                "upper": interval["upper"],
            },
            "descriptive_only": True,
        }

    return {
        "semantic": spec.semantic,
        "raw_unit": spec.unit,
        "supported_tasks": list(spec.supported_tasks),
        "active_train_dimensions_by_task": {
            task_name: [
                dimension
                for dimension in spec.dimensions
                if decoder.active_target_by_task[TASKS.index(task_name), dimension]
            ]
            for task_name in spec.supported_tasks
        },
        "methods": {
            **{
                name: dict(reduction.report)
                for name, reduction in reductions.items()
            },
            "wrong_action_shift_macro": dict(wrong_mean_reduction.report),
            "wrong_action_shifts": {
                "count": WRONG_ACTION_SHIFT_COUNT,
                "per_shift": wrong_reports,
            },
        },
        "unavailable_methods": (
            {
                "q0_physical_hold": (
                    "unavailable for task_progress: formal_v2 stores H0 "
                    "goal_relative_progress, not H0 collector task_progress; "
                    "the distinct field is not substituted"
                )
            }
            if not q0_available
            else {}
        ),
        "comparisons": {
            "factual_vs_fixed_baselines": baseline_comparisons,
            "factual_vs_wrong_action_shifts": {
                "count": WRONG_ACTION_SHIFT_COUNT,
                "per_shift": wrong_comparisons,
            },
            "target_latent_ceiling_vs_factual": ceiling_comparison,
        },
        "development_mechanism_evidence": {
            "designation": AUDIT_DESIGNATION,
            "evaluation_data_are_exposed_development": True,
            "decision": "descriptive_only_no_gate",
            "gate_configured": False,
            "confirmation_claim_authorized": False,
            "predictions_averaged_across_lewm_seeds": False,
            "factual_vs_fixed_baselines": {
                name: descriptive_observation(value)
                for name, value in baseline_comparisons.items()
            },
            "factual_vs_wrong_action_shifts": {
                str(row["donor_shift"]): descriptive_observation(row)
                for row in wrong_comparisons
            },
            "target_latent_ceiling_vs_factual": descriptive_observation(
                ceiling_comparison
            ),
        },
    }


def _label_availability() -> dict[str, Any]:
    return {
        "decoded": {
            "task_progress": {
                "source": "preflight_labels.branch_outcome[...,0]",
                "horizons": list(range(1, HORIZON + 1)),
                "tasks": list(TASKS),
            },
            "object_position": {
                "source": "preflight_labels.physical_state[...,:3]",
                "horizons": list(range(1, HORIZON + 1)),
                "tasks": [task for task in TASKS if task != "reach-v3"],
            },
            "tcp_position": {
                "source": "preflight_labels.physical_state[...,:3]",
                "horizons": list(range(1, HORIZON + 1)),
                "tasks": ["reach-v3"],
            },
        },
        "h0_comparison_fields": {
            "q0_object_position": {
                "source": "preflight_labels.object_pose[:,:3]",
                "tasks": [task for task in TASKS if task != "reach-v3"],
                "use": "q0_physical_hold and matched_direct_action only",
            },
            "q0_tcp_position": {
                "source": "preflight_labels.proprio[:,:3]",
                "tasks": ["reach-v3"],
                "use": "q0_physical_hold and matched_direct_action only",
            },
            "h0_robot_proprio": {
                "source": "preflight_labels.proprio[:,:4]",
                "tasks": list(TASKS),
                "use": "matched_direct_action feature only; not a future target",
            },
        },
        "excluded_fail_closed": {
            "future_goal_relative_vector": (
                "unavailable: the archive has H0 goal_relative_progress but no "
                "branch/horizon reference pose or future goal-relative vector"
            ),
            "future_goal_relative_progress": (
                "unavailable: H0 goal_relative_progress is not substituted for the "
                "collector's distinct future task-progress signal"
            ),
            "future_robot_proprio_all_tasks": (
                "unavailable: proprio is H0-only in preflight_labels"
            ),
            "future_tcp_position_non_reach_tasks": (
                "unavailable: physical_state stores object pose for non-reach tasks"
            ),
            "fabricated_or_zero_filled_labels": False,
        },
    }


def build_a4_physical_decoding_report(
    cache: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    decoder: FrozenA4PhysicalDecoder,
    decoded_seeds: Sequence[DecodedA4Seed],
    *,
    checkpoint_manifest: Sequence[Mapping[str, Any]],
    fixed_baselines: FixedA4Baselines | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the complete development report and compact decoded archive."""

    coverage = validate_a4_physical_decoding_inputs(cache, labels)
    if not isinstance(decoder, FrozenA4PhysicalDecoder):
        raise TypeError("decoder must be a FrozenA4PhysicalDecoder")
    samples = _positive_integer(bootstrap_samples, label="bootstrap samples")
    seed_value = _nonnegative_integer(bootstrap_seed, label="bootstrap seed")
    rows = tuple(decoded_seeds)
    seeds = tuple(row.seed for row in rows)
    if seeds != tuple(CONFIRMATION_TRAINING_SEEDS):
        raise ValueError(
            "A4 physical decoding seeds must remain exactly "
            f"{CONFIRMATION_TRAINING_SEEDS}"
        )
    if len(checkpoint_manifest) != len(seeds):
        raise ValueError("checkpoint manifest must contain one row per frozen seed")
    manifest_seeds = tuple(int(row.get("seed", -1)) for row in checkpoint_manifest)
    if manifest_seeds != seeds:
        raise ValueError("checkpoint manifest seed order differs from decoded panels")

    split = np.asarray(cache["split"]).astype(str, copy=False)
    development = split == "development"
    task = np.asarray(cache["task_class_index"], dtype=np.int64)[development]
    task_name = np.asarray(cache["task_name"])[development]
    episode = np.asarray(cache["source_episode_id"])[development].astype(str)
    proposal_id = np.asarray(cache["proposal_id"])[development]
    truth_raw = _future_physical_targets(labels)[development]
    truth_standardized = decoder.standardize_targets(truth_raw, task)
    target_ceiling = decoder.decode_standardized(
        np.asarray(cache["target_state"])[development]
    )
    baselines = (
        fit_fixed_a4_baselines(cache, labels, decoder)
        if fixed_baselines is None
        else fixed_baselines
    )
    if not isinstance(baselines, FixedA4Baselines):
        raise TypeError("fixed_baselines must be a FixedA4Baselines instance")
    factual = np.stack([row.factual for row in rows], axis=0)
    anchor_only = np.stack([row.anchor_only_factual for row in rows], axis=0)
    no_action = np.stack([row.matched_no_action for row in rows], axis=0)
    wrong_action = np.stack([row.wrong_action for row in rows], axis=0)
    expected_seeded = (
        len(seeds),
        int(np.count_nonzero(development)),
        BRANCH_COUNT,
        HORIZON,
        TARGET_DIM,
    )
    if (
        factual.shape != expected_seeded
        or anchor_only.shape != expected_seeded
        or no_action.shape != expected_seeded
    ):
        raise ValueError("decoded seeded panels have invalid axes")
    if wrong_action.shape != (
        len(seeds),
        expected_seeded[1],
        WRONG_ACTION_SHIFT_COUNT,
        *expected_seeded[2:],
    ):
        raise ValueError("decoded wrong-action panel must contain all seven shifts")

    windows: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for window_name, horizon_indices in HORIZON_WINDOWS.items():
        readouts: dict[str, Any] = {}
        for spec in READOUT_SPECS:
            readouts[spec.name] = _score_readout_window(
                decoder=decoder,
                spec=spec,
                truth_standardized=truth_standardized,
                truth_raw=truth_raw,
                task=task,
                episode=episode,
                target_ceiling=target_ceiling,
                factual=factual,
                anchor_only=anchor_only,
                no_action=no_action,
                wrong_action=wrong_action,
                baselines=baselines,
                seeds=seeds,
                horizon_indices=horizon_indices,
                bootstrap_samples=samples,
                bootstrap_seed=seed_value,
                window_name=window_name,
            )
        windows[window_name] = {
            "horizons": [index + 1 for index in horizon_indices],
            "readouts": readouts,
        }
        evidence[window_name] = {
            name: value["development_mechanism_evidence"]
            for name, value in readouts.items()
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "stage": STAGE,
        "audit_designation": AUDIT_DESIGNATION,
        "diagnostic_only": True,
        "exposed_development": True,
        "exploratory_mechanism_diagnostic": True,
        "confirmation_claim_authorized": False,
        "gate_configured": False,
        "decision": "descriptive_only_no_gate",
        "reserved_test_accessed": False,
        "fit_split": "train",
        "evaluation_split": "development",
        "model_optimizer_steps": 0,
        "checkpoint_parameters_frozen": True,
        "ridge": {
            "implementation": (
                "fcpg.lewm_action_branch_a4_metrics.closed_form_ridge_predict"
            ),
            "fit": "closed_form_train_only",
            "alpha": decoder.ridge_alpha,
            "alpha_fixed_in_source": True,
            "development_grid_search": False,
            "features_fixed_in_source": True,
            "features": {
                "full_state": {
                    "contract": "flattened_four_slot_latent_state_[4,192]",
                    "dimension": FEATURE_DIM,
                },
                "anchor_only": {
                    "contract": "flattened_anchor_slots_1_to_3_[3,192]",
                    "dimension": ANCHOR_FEATURE_DIM,
                },
                "task_time": {
                    "contract": "task_one_hot_7_plus_horizon_one_hot_8",
                    "dimension": len(TASKS) + HORIZON,
                },
                "matched_direct_action": {
                    "contract": (
                        "latest_history_anchor_latent; two_history_actions; "
                        "current/cumulative/first/cumulative_square_future_action; "
                        "controlled_q0; H0_proprio; task/horizon/branch_semantic; "
                        "cumulative_action_norm; past_action_delta; center_action; "
                        "normalized_horizon"
                    ),
                    "dimension": MATCHED_DIRECT_ACTION_FEATURE_DIM,
                    "parameter_matched_to_full_state_readout": True,
                },
            },
            "target_fields": list(TARGET_FIELDS),
            "targets_fixed_in_source": True,
            "train_bundle_count": decoder.train_bundle_count,
            "train_point_count": decoder.train_point_count,
            "train_physical_episode_count_by_task": {
                task_name: decoder.train_episode_count_by_task[index]
                for index, task_name in enumerate(TASKS)
            },
            "target_standardization": "per_task_episode_equal_train_points_only",
        },
        "checkpoint_panel": [dict(row) for row in checkpoint_manifest],
        "coverage": coverage,
        "label_availability": _label_availability(),
        "aggregation": {
            "point_loss": "mean_active_dimensions_then_branches_and_horizons",
            "task_macro": "equal_task_mean_after_equal_physical_episode_mean",
            "seed": (
                "predictions_scored_separately_per_lewm_seed; per_seed_losses "
                "reported before any task-macro summary"
            ),
            "predictions_averaged_across_lewm_seeds": False,
            "wrong_action": "seven_nonidentity_cyclic_donor_shifts_reported_separately",
            "bootstrap": {
                "unit": "crossed_lewm_seed_x_source_physical_episode_within_task",
                "paired": True,
                "task_macro": True,
                "seed_resampling": True,
                "physical_episode_resampling": "independent_within_task",
                "samples": samples,
                "seed": seed_value,
            },
        },
        "horizon_windows": windows,
        "development_mechanism_evidence": evidence,
    }
    archive = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "protocol": np.asarray(PROTOCOL),
        "proposal_id": np.asarray(proposal_id),
        "task_class_index": np.asarray(task, dtype=np.int64),
        "task_name": np.asarray(task_name),
        "source_episode_id": np.asarray(episode),
        "split": np.full(proposal_id.shape, "development"),
        "training_seed": np.asarray(seeds, dtype=np.int64),
        "wrong_action_donor_shift": np.arange(1, BRANCH_COUNT, dtype=np.int64),
        "physical_target_raw": truth_raw.astype(np.float32),
        "physical_target_standardized": truth_standardized.astype(np.float32),
        "target_latent_ceiling": target_ceiling.astype(np.float32),
        "factual": factual.astype(np.float32),
        "anchor_only_factual": anchor_only.astype(np.float32),
        "matched_no_action": no_action.astype(np.float32),
        "wrong_action": wrong_action.astype(np.float32),
        "matched_direct_action": baselines.matched_direct_action.astype(np.float32),
        "task_time": baselines.task_time.astype(np.float32),
        "latent_hold": baselines.latent_hold.astype(np.float32),
        "q0_position_raw": baselines.q0_position_raw[:, 0, 0, 1:].astype(
            np.float32
        ),
    }
    return report, archive


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DecodedA4Seed",
    "FEATURE_DIM",
    "FixedA4Baselines",
    "FrozenA4PhysicalDecoder",
    "HORIZON_WINDOWS",
    "READOUT_SPECS",
    "SCHEMA_VERSION",
    "STAGE",
    "build_a4_physical_decoding_report",
    "decode_a4_evaluation_arrays",
    "fit_a4_physical_decoder",
    "fit_fixed_a4_baselines",
    "validate_a4_physical_decoding_inputs",
]
