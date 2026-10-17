"""Development and confirmation scoring for the A3 action-branch protocol.

The scorer fits one whitening transform per task from canonical train rows.
Train physical episodes receive equal weight even when they contain different
numbers of random hazards.  Evaluation losses retain complete branch bundles
and are reduced in the frozen order: hazards, training seeds, physical episodes,
then an equal-weight task macro.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_metrics import (
    FACTUAL_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    NO_ACTION_METHOD,
    SHUFFLED_ACTION_METHOD,
    TrainMoments,
    build_secondary_predictions,
    whiten_latents,
)


PROTOCOL = "fcpg_action_branch_physical_support_a3_v1"
TASKS = (
    "pick-place-v3",
    "drawer-open-v3",
    "reach-v3",
    "push-v3",
)
TRAINING_SEEDS = (29_201, 29_202, 29_203)
CHECKPOINT_STEP = 2_000
CONFIRMATION_TRAINING_SEEDS = (29_301, 29_302, 29_303)
CONFIRMATION_SEEDS = CONFIRMATION_TRAINING_SEEDS
CONFIRMATION_CHECKPOINT_STEP = 10_000
PRIMARY_HORIZONS = (6, 7, 8)
BRANCH_COUNT = 8
SHUFFLE_SHIFT_COUNT = BRANCH_COUNT - 1
MINIMUM_RELATIVE_IMPROVEMENT = 0.05
MINIMUM_POSITIVE_TASKS = 3
MINIMUM_POSITIVE_SEEDS = 2
MINIMUM_CROSS_REPLICATING_SEEDS = 2
MINIMUM_POSITIVE_TASKS_PER_REPLICATING_SEED = 3
SHUFFLED_PREDICTION_RTOL = 1e-6
SHUFFLED_PREDICTION_ATOL = 1e-7


def _numeric(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _integer_vector(value: Any, *, label: str, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{label} must have shape [N]")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{label} must be a one-dimensional integer array")
    return np.asarray(array, dtype=np.int64)


def _string_vector(value: Any, *, label: str, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"{label} must have shape [N]")
    result = array.astype(str)
    if np.any(result == ""):
        raise ValueError(f"{label} values must be non-empty")
    return result


def _horizon_indices(
    horizons: Sequence[int], *, available: int
) -> tuple[tuple[int, ...], np.ndarray]:
    values: list[int] = []
    for value in horizons:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("horizons must contain integers")
        values.append(int(value))
    resolved = tuple(values)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("horizons must be distinct and non-empty")
    if any(value < 1 or value > available for value in resolved):
        raise ValueError(f"horizons must be in the range 1..{available}")
    return resolved, np.asarray([value - 1 for value in resolved], dtype=np.int64)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def validate_a3_scoring_contract(
    contract: Mapping[str, Any], *, phase: str
) -> Mapping[str, Any]:
    """Fail closed if a checked-in A3 Gate differs from scorer constants."""

    root = _mapping(contract, label="A3 contract")
    if root.get("protocol") != PROTOCOL:
        raise ValueError(f"A3 scoring protocol must remain {PROTOCOL}")
    if tuple(str(value) for value in root.get("tasks", ())) != TASKS:
        raise ValueError(f"A3 scoring tasks must remain {TASKS}")
    if phase == "development":
        section_name = "development_gate"
        evaluation_split = "development"
        checkpoint_field = "checkpoint_step"
        checkpoint_step = CHECKPOINT_STEP
        seeds = TRAINING_SEEDS
    elif phase == "confirmation":
        section_name = "confirmation_after_development_go"
        evaluation_split = "test"
        checkpoint_field = "optimizer_steps"
        checkpoint_step = CONFIRMATION_CHECKPOINT_STEP
        seeds = CONFIRMATION_TRAINING_SEEDS
    else:
        raise ValueError("A3 scoring phase must be development or confirmation")
    gate = _mapping(root.get(section_name), label=f"A3 {section_name}")
    observed = (
        gate.get("evaluation_split"),
        int(gate.get(checkpoint_field, -1)),
        tuple(int(value) for value in gate.get("seeds", ())),
        tuple(int(value) for value in gate.get("primary_horizons", ())),
        float(gate.get("minimum_relative_improvement", float("nan"))),
        int(gate.get("minimum_positive_tasks", -1)),
        int(gate.get("minimum_positive_seeds", -1)),
        int(gate.get("minimum_cross_replicating_seeds", -1)),
        int(gate.get("minimum_positive_tasks_per_replicating_seed", -1)),
        gate.get("every_shuffled_shift_must_reach_minimum"),
        gate.get("raw_factual_not_worse_than_no_action"),
        gate.get("raw_factual_better_than_hold_last_and_constant_velocity"),
    )
    expected = (
        evaluation_split,
        checkpoint_step,
        seeds,
        PRIMARY_HORIZONS,
        MINIMUM_RELATIVE_IMPROVEMENT,
        MINIMUM_POSITIVE_TASKS,
        MINIMUM_POSITIVE_SEEDS,
        MINIMUM_CROSS_REPLICATING_SEEDS,
        MINIMUM_POSITIVE_TASKS_PER_REPLICATING_SEED,
        True,
        True,
        True,
    )
    if observed != expected:
        raise ValueError(f"A3 {phase} scoring Gate contract drifted")
    if phase == "confirmation" and gate.get("selection") != "fixed_latest":
        raise ValueError("A3 confirmation scoring selection must be fixed_latest")
    return gate


def validate_shuffled_prediction_replay(
    factual_prediction: np.ndarray,
    shuffled_prediction: np.ndarray,
    shuffle_permutation: np.ndarray,
    *,
    label: str = "A3 prediction",
    rtol: float = SHUFFLED_PREDICTION_RTOL,
    atol: float = SHUFFLED_PREDICTION_ATOL,
) -> None:
    """Verify shuffled predictions are factual outputs in donor-branch order."""

    factual = _numeric(factual_prediction, label=f"{label} factual_prediction")
    shuffled = _numeric(
        shuffled_prediction, label=f"{label} shuffled_action_prediction"
    )
    permutation = np.asarray(shuffle_permutation)
    if factual.ndim != 5:
        raise ValueError(f"{label} factual_prediction must have shape [N,8,P,K,D]")
    expected_shuffled_shape = (
        factual.shape[0],
        factual.shape[1] - 1,
        *factual.shape[1:],
    )
    if shuffled.shape != expected_shuffled_shape:
        raise ValueError(
            f"{label} shuffled_action_prediction must have shape "
            f"{expected_shuffled_shape}"
        )
    expected_permutation_shape = (
        factual.shape[0],
        factual.shape[1] - 1,
        factual.shape[1],
    )
    if (
        permutation.shape != expected_permutation_shape
        or permutation.dtype == np.bool_
        or not np.issubdtype(permutation.dtype, np.integer)
    ):
        raise ValueError(
            f"{label} shuffle_permutation must have shape "
            f"{expected_permutation_shape} and use integers"
        )
    if isinstance(rtol, (bool, np.bool_)) or isinstance(atol, (bool, np.bool_)):
        raise TypeError("shuffled prediction tolerances must be numeric")
    relative = float(rtol)
    absolute = float(atol)
    if (
        not math.isfinite(relative)
        or not math.isfinite(absolute)
        or relative < 0.0
        or absolute < 0.0
    ):
        raise ValueError("shuffled prediction tolerances must be finite and non-negative")
    bundle_rows = np.arange(factual.shape[0], dtype=np.int64)[:, None]
    for shift in range(factual.shape[1] - 1):
        expected = factual[bundle_rows, permutation[:, shift]]
        observed = shuffled[:, shift]
        if not np.allclose(
            observed,
            expected,
            rtol=relative,
            atol=absolute,
            equal_nan=False,
        ):
            maximum_error = float(np.max(np.abs(observed - expected)))
            raise ValueError(
                f"{label} shuffled_action_prediction shift {shift + 1} does not "
                "equal factual_prediction in shuffle_permutation donor order; "
                f"max_abs_error={maximum_error:.9g}"
            )


def fit_task_whitening_moments(
    target: np.ndarray,
    *,
    task_index: np.ndarray,
    source_episode_id: np.ndarray,
    split: np.ndarray,
    task_names: Sequence[str] = TASKS,
    horizons: Sequence[int] = PRIMARY_HORIZONS,
    epsilon: float = 1e-8,
) -> dict[str, TrainMoments]:
    """Fit H6-H8 moments per task with equal train-episode weight."""

    values = _numeric(target, label="target")
    if values.ndim != 5 or values.shape[1] != BRANCH_COUNT:
        raise ValueError("target must have shape [N,8,P,K,D]")
    if any(size <= 0 for size in values.shape):
        raise ValueError("target axes must be non-empty")
    count = int(values.shape[0])
    tasks = tuple(str(name) for name in task_names)
    if not tasks or any(not name for name in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("task_names must be distinct and non-empty")
    task_values = _integer_vector(task_index, label="task_index", count=count)
    if np.any(task_values < 0) or np.any(task_values >= len(tasks)):
        raise ValueError("task_index is outside task_names")
    episodes = _string_vector(
        source_episode_id, label="source_episode_id", count=count
    )
    splits = _string_vector(split, label="split", count=count)
    _, indices = _horizon_indices(horizons, available=values.shape[2])
    if isinstance(epsilon, (bool, np.bool_)):
        raise TypeError("epsilon must be numeric")
    threshold = float(epsilon)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    result: dict[str, TrainMoments] = {}
    for task_id, task_name in enumerate(tasks):
        selected = (task_values == task_id) & (splits == "train")
        if not np.any(selected):
            raise ValueError(f"{task_name} has no canonical train rows")
        selected_target = np.asarray(
            values[selected][:, :, indices], dtype=np.float64
        )
        selected_episodes = episodes[selected]
        identities = np.unique(selected_episodes)
        episode_means: list[np.ndarray] = []
        episode_seconds: list[np.ndarray] = []
        for identity in identities:
            episode_values = selected_target[selected_episodes == identity]
            episode_means.append(
                np.mean(episode_values, axis=(0, 1, 2), dtype=np.float64)
            )
            episode_seconds.append(
                np.mean(
                    np.square(episode_values),
                    axis=(0, 1, 2),
                    dtype=np.float64,
                )
            )
        mean = np.mean(episode_means, axis=0, dtype=np.float64)
        second = np.mean(episode_seconds, axis=0, dtype=np.float64)
        variance = np.maximum(second - np.square(mean), 0.0)
        raw_scale = np.sqrt(variance)
        floored = raw_scale <= threshold
        result[task_name] = TrainMoments(
            mean=mean,
            scale=np.where(floored, 1.0, raw_scale),
            count=int(identities.size),
            epsilon=threshold,
            floored_dimensions=int(np.count_nonzero(floored)),
        )
    return result


# Explicit alias for callers that describe the weighting rather than the scope.
fit_episode_equal_task_moments = fit_task_whitening_moments


@dataclass(frozen=True)
class _CanonicalPanel:
    target: np.ndarray
    history: np.ndarray
    task_index: np.ndarray
    task_instance_index: np.ndarray | None
    episode: np.ndarray
    hazard: np.ndarray
    split: np.ndarray
    evaluation_rows: np.ndarray


def _validate_canonical(
    arrays: Mapping[str, np.ndarray],
    *,
    task_names: Sequence[str],
    evaluation_split: str,
    expected_train_episodes: int | None,
    expected_evaluation_episodes: int | None,
) -> _CanonicalPanel:
    if not isinstance(arrays, Mapping):
        raise TypeError("canonical arrays must be a mapping")
    required = {
        "target_embedding",
        "history_embedding",
        "task_index",
        "source_episode_id",
        "hazard_id",
        "split",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"canonical arrays are missing {missing}")
    target = _numeric(arrays["target_embedding"], label="target_embedding")
    if target.ndim != 5 or target.shape[1:3] != (BRANCH_COUNT, 8):
        raise ValueError("A3 target_embedding must have shape [N,8,8,K,D]")
    if any(size <= 0 for size in target.shape):
        raise ValueError("A3 target_embedding axes must be non-empty")
    count = int(target.shape[0])
    history = _numeric(arrays["history_embedding"], label="history_embedding")
    if (
        history.ndim != 4
        or history.shape[0] != count
        or history.shape[1] < 2
        or history.shape[2:] != target.shape[3:]
    ):
        raise ValueError(
            "history_embedding must have shape [N,H,K,D] with H>=2 and "
            "target-matched K,D"
        )
    names = tuple(str(name) for name in task_names)
    task = _integer_vector(arrays["task_index"], label="task_index", count=count)
    if set(task.tolist()) != set(range(len(names))):
        raise ValueError("canonical A3 rows must cover exactly the frozen tasks")
    episode = _string_vector(
        arrays["source_episode_id"], label="source_episode_id", count=count
    )
    hazard = _integer_vector(arrays["hazard_id"], label="hazard_id", count=count)
    split = _string_vector(arrays["split"], label="split", count=count)
    if evaluation_split not in {"development", "test"}:
        raise ValueError("A3 evaluation_split must be development or test")
    required_splits = {"train", evaluation_split}
    if set(split.tolist()) != required_splits:
        other_split = "test" if evaluation_split == "development" else "development"
        if np.any(split == other_split):
            raise ValueError(
                f"A3 {evaluation_split} scoring refuses canonical {other_split} rows"
            )
        raise ValueError(
            "canonical A3 "
            f"{evaluation_split} scoring requires only train/{evaluation_split} rows"
        )

    row_keys: set[tuple[int, str, int]] = set()
    episode_splits: dict[tuple[int, str], str] = {}
    for task_id, identity, hazard_id, split_name in zip(
        task.tolist(), episode.tolist(), hazard.tolist(), split.tolist(), strict=True
    ):
        row_key = (int(task_id), str(identity), int(hazard_id))
        if row_key in row_keys:
            raise ValueError(f"duplicate canonical A3 hazard row: {row_key!r}")
        row_keys.add(row_key)
        episode_key = row_key[:2]
        previous = episode_splits.setdefault(episode_key, str(split_name))
        if previous != split_name:
            raise ValueError(
                f"one physical episode crosses train/{evaluation_split}"
            )

    task_instance: np.ndarray | None = None
    if "task_instance_index" in arrays:
        task_instance = _integer_vector(
            arrays["task_instance_index"],
            label="task_instance_index",
            count=count,
        )
        instance_by_episode: dict[tuple[int, str], int] = {}
        episode_by_instance: dict[tuple[int, int], str] = {}
        for task_id, identity, instance in zip(
            task.tolist(), episode.tolist(), task_instance.tolist(), strict=True
        ):
            episode_key = (int(task_id), str(identity))
            previous_instance = instance_by_episode.setdefault(
                episode_key, int(instance)
            )
            instance_key = (int(task_id), int(instance))
            previous_episode = episode_by_instance.setdefault(
                instance_key, str(identity)
            )
            if previous_instance != instance or previous_episode != identity:
                raise ValueError(
                    "source_episode_id and task_instance_index must be one-to-one "
                    "within task"
                )

    for task_id, task_name in enumerate(names):
        for split_name, expected in (
            ("train", expected_train_episodes),
            (evaluation_split, expected_evaluation_episodes),
        ):
            identities = np.unique(
                episode[(task == task_id) & (split == split_name)]
            )
            if identities.size == 0:
                raise ValueError(f"{task_name} has no {split_name} physical episodes")
            if expected is not None and identities.size != expected:
                raise ValueError(
                    f"{task_name} requires exactly {expected} {split_name} "
                    f"physical episodes, got {identities.size}"
                )

    return _CanonicalPanel(
        target=target,
        history=history,
        task_index=task,
        task_instance_index=task_instance,
        episode=episode,
        hazard=hazard,
        split=split,
        evaluation_rows=np.flatnonzero(split == evaluation_split),
    )


def _prediction(
    predictions: Mapping[str, np.ndarray],
    method: str,
    aliases: Sequence[str],
) -> np.ndarray:
    present = [name for name in aliases if name in predictions]
    if not present:
        raise ValueError(f"predictions are missing {method!r}")
    reference = np.asarray(predictions[present[0]])
    for name in present[1:]:
        if not np.array_equal(reference, np.asarray(predictions[name])):
            raise ValueError(f"conflicting aliases were supplied for {method!r}")
    return _numeric(reference, label=f"prediction {method!r}")


def _validated_predictions(
    predictions: Mapping[str, np.ndarray],
    *,
    seed_count: int,
    evaluation_count: int,
    target_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping")
    factual = _prediction(
        predictions,
        FACTUAL_METHOD,
        (FACTUAL_METHOD, "factual_prediction"),
    )
    no_action = _prediction(
        predictions,
        NO_ACTION_METHOD,
        (NO_ACTION_METHOD, "no_action_prediction", "matched_no_action"),
    )
    shuffled = _prediction(
        predictions,
        SHUFFLED_ACTION_METHOD,
        (SHUFFLED_ACTION_METHOD, "shuffled_action_prediction"),
    )
    ordinary_shape = (seed_count, evaluation_count, *target_shape)
    shuffled_shape = (
        seed_count,
        evaluation_count,
        SHUFFLE_SHIFT_COUNT,
        *target_shape,
    )
    if factual.shape != ordinary_shape:
        raise ValueError(f"factual prediction must have shape {ordinary_shape}")
    if no_action.shape != ordinary_shape:
        raise ValueError(f"no-action prediction must have shape {ordinary_shape}")
    if shuffled.shape != shuffled_shape:
        raise ValueError(f"shuffled prediction must have shape {shuffled_shape}")
    if not np.allclose(
        no_action,
        no_action[:, :, :1],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("no-action prediction must be branch-invariant")
    return {
        FACTUAL_METHOD: factual,
        NO_ACTION_METHOD: no_action,
        SHUFFLED_ACTION_METHOD: shuffled,
    }


def _ordinary_losses(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    task_index: np.ndarray,
    task_names: Sequence[str],
    moments: Mapping[str, TrainMoments],
    horizon_indices: np.ndarray,
    evaluation_split: str,
) -> tuple[np.ndarray, np.ndarray]:
    seed_count, row_count = prediction.shape[:2]
    centered_loss = np.empty((seed_count, row_count), dtype=np.float64)
    raw_loss = np.empty((seed_count, row_count), dtype=np.float64)
    for task_id, task_name in enumerate(task_names):
        selected = task_index == task_id
        target_w = whiten_latents(
            target[selected][:, :, horizon_indices],
            moments[task_name],
            label=f"{task_name} {evaluation_split} target",
        )
        prediction_w = whiten_latents(
            prediction[:, selected][:, :, :, horizon_indices],
            moments[task_name],
            label=f"{task_name} {evaluation_split} prediction",
        )
        target_centered = target_w - np.mean(
            target_w, axis=1, keepdims=True, dtype=np.float64
        )
        prediction_centered = prediction_w - np.mean(
            prediction_w, axis=2, keepdims=True, dtype=np.float64
        )
        centered_loss[:, selected] = np.mean(
            np.square(prediction_centered - target_centered[None]),
            axis=(2, 3, 4, 5),
            dtype=np.float64,
        )
        raw_loss[:, selected] = np.mean(
            np.square(prediction_w - target_w[None]),
            axis=(2, 3, 4, 5),
            dtype=np.float64,
        )
    return centered_loss, raw_loss


def _shuffled_centered_losses(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    task_index: np.ndarray,
    task_names: Sequence[str],
    moments: Mapping[str, TrainMoments],
    horizon_indices: np.ndarray,
    evaluation_split: str,
) -> np.ndarray:
    seed_count, row_count, shift_count = prediction.shape[:3]
    centered_loss = np.empty(
        (seed_count, row_count, shift_count), dtype=np.float64
    )
    for task_id, task_name in enumerate(task_names):
        selected = task_index == task_id
        target_w = whiten_latents(
            target[selected][:, :, horizon_indices],
            moments[task_name],
            label=f"{task_name} {evaluation_split} target",
        )
        prediction_w = whiten_latents(
            prediction[:, selected][:, :, :, :, horizon_indices],
            moments[task_name],
            label=f"{task_name} {evaluation_split} shuffled prediction",
        )
        target_centered = target_w - np.mean(
            target_w, axis=1, keepdims=True, dtype=np.float64
        )
        prediction_centered = prediction_w - np.mean(
            prediction_w, axis=3, keepdims=True, dtype=np.float64
        )
        centered_loss[:, selected] = np.mean(
            np.square(prediction_centered - target_centered[None, :, None]),
            axis=(3, 4, 5, 6),
            dtype=np.float64,
        )
    return centered_loss


@dataclass(frozen=True)
class _Reduction:
    task_values: Mapping[str, float]
    task_seed_values: Mapping[str, Mapping[int, float]]
    episode_values: Mapping[str, np.ndarray]


def _reduce_losses(
    losses: np.ndarray,
    *,
    task_index: np.ndarray,
    episode: np.ndarray,
    task_names: Sequence[str],
    seeds: Sequence[int],
    evaluation_split: str,
) -> _Reduction:
    values = _numeric(losses, label=f"{evaluation_split} losses").astype(
        np.float64, copy=False
    )
    if values.shape != (len(seeds), task_index.size):
        raise ValueError(
            f"{evaluation_split} losses must have shape [seed,bundle]"
        )
    if np.any(values < 0.0):
        raise ValueError(f"{evaluation_split} losses must be non-negative")
    task_values: dict[str, float] = {}
    task_seed_values: dict[str, dict[int, float]] = {}
    episode_values: dict[str, np.ndarray] = {}
    for task_id, task_name in enumerate(task_names):
        selected = task_index == task_id
        identities = np.unique(episode[selected])
        by_seed = np.empty((len(seeds), identities.size), dtype=np.float64)
        for seed_position in range(len(seeds)):
            for episode_position, identity in enumerate(identities):
                rows = selected & (episode == identity)
                by_seed[seed_position, episode_position] = np.mean(
                    values[seed_position, rows], dtype=np.float64
                )
        episode_values[task_name] = np.mean(by_seed, axis=0, dtype=np.float64)
        task_seed_values[task_name] = {
            int(seed): float(np.mean(by_seed[position], dtype=np.float64))
            for position, seed in enumerate(seeds)
        }
        task_values[task_name] = float(
            np.mean(episode_values[task_name], dtype=np.float64)
        )
    return _Reduction(task_values, task_seed_values, episode_values)


def _method_report(
    reduction: _Reduction,
    *,
    task_names: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    return {
        "task_macro_mse": float(
            np.mean([reduction.task_values[task] for task in task_names])
        ),
        "per_task": {
            task: {
                "mse": reduction.task_values[task],
                "physical_episode_count": int(reduction.episode_values[task].size),
                "per_seed": {
                    str(seed): {"mse": reduction.task_seed_values[task][seed]}
                    for seed in seeds
                },
            }
            for task in task_names
        },
        "per_seed": {
            str(seed): {
                "task_macro_mse": float(
                    np.mean(
                        [
                            reduction.task_seed_values[task][seed]
                            for task in task_names
                        ]
                    )
                )
            }
            for seed in seeds
        },
    }


def _relative_effect(factual: float, baseline: float, *, label: str) -> float:
    if factual < 0.0 or baseline < 0.0:
        raise ValueError(f"{label} losses must be non-negative")
    if baseline == 0.0:
        if factual == 0.0:
            return 0.0
        raise ValueError(f"{label} has zero baseline loss and positive factual loss")
    result = 1.0 - factual / baseline
    if not math.isfinite(result):
        raise ValueError(f"{label} effect must be finite")
    return float(result)


def _comparison_report(
    factual: _Reduction,
    baseline: _Reduction,
    *,
    baseline_method: str,
    metric: str,
    task_names: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task in task_names:
        per_seed = {
            str(seed): {
                "factual_mse": factual.task_seed_values[task][seed],
                "baseline_mse": baseline.task_seed_values[task][seed],
                "effect": _relative_effect(
                    factual.task_seed_values[task][seed],
                    baseline.task_seed_values[task][seed],
                    label=f"{task}/{seed}/{baseline_method}",
                ),
            }
            for seed in seeds
        }
        effect = _relative_effect(
            factual.task_values[task],
            baseline.task_values[task],
            label=f"{task}/{baseline_method}",
        )
        per_task[task] = {
            "factual_mse": factual.task_values[task],
            "baseline_mse": baseline.task_values[task],
            "effect": effect,
            "positive": effect > 0.0,
            "per_seed": per_seed,
        }
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        effects = {
            task: float(per_task[task]["per_seed"][str(seed)]["effect"])
            for task in task_names
        }
        effect = float(np.mean(list(effects.values())))
        per_seed[str(seed)] = {
            "effect": effect,
            "positive": effect > 0.0,
            "per_task_effect": effects,
        }
    effect = float(np.mean([per_task[task]["effect"] for task in task_names]))
    return {
        "metric": metric,
        "factual_method": FACTUAL_METHOD,
        "baseline_method": baseline_method,
        "orientation": "positive_means_factual_has_lower_mse",
        "effect": effect,
        "relative_improvement": effect,
        "positive_task_count": sum(
            bool(per_task[task]["positive"]) for task in task_names
        ),
        "task_count": len(task_names),
        "positive_seed_count": sum(
            bool(per_seed[str(seed)]["positive"]) for seed in seeds
        ),
        "seed_count": len(seeds),
        "per_task": per_task,
        "per_seed": per_seed,
    }


def _summarize_a3_panel(
    canonical_arrays: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    training_seeds: Sequence[int],
    task_names: Sequence[str],
    primary_horizons: Sequence[int],
    minimum_relative_improvement: float,
    evaluation_split: str,
    stage: str,
    decision_prefix: str,
    checkpoint_step: int,
    expected_train_episodes: int | None = None,
    expected_evaluation_episodes: int | None = None,
) -> dict[str, Any]:
    """Score one complete A3 panel after its phase contract is fixed.

    Ordinary prediction arrays use ``[seed, evaluation_bundle, branch,
    horizon, anchor, latent]``.  The shuffled control inserts its seven-shift
    axis after ``evaluation_bundle``.
    """

    names = tuple(str(name) for name in task_names)
    if names != TASKS:
        raise ValueError(f"A3 scoring tasks must remain fixed at {TASKS}")
    seeds = tuple(int(seed) for seed in training_seeds)
    if len(seeds) != 3 or len(set(seeds)) != len(seeds):
        raise ValueError("A3 scoring requires exactly three distinct training seeds")
    if isinstance(minimum_relative_improvement, (bool, np.bool_)):
        raise TypeError("minimum_relative_improvement must be numeric")
    minimum_effect = float(minimum_relative_improvement)
    if not math.isfinite(minimum_effect) or minimum_effect != 0.05:
        raise ValueError("A3 minimum relative improvement is frozen at 0.05")

    panel = _validate_canonical(
        canonical_arrays,
        task_names=names,
        evaluation_split=evaluation_split,
        expected_train_episodes=expected_train_episodes,
        expected_evaluation_episodes=expected_evaluation_episodes,
    )
    horizons, horizon_indices = _horizon_indices(
        primary_horizons, available=panel.target.shape[2]
    )
    if horizons != PRIMARY_HORIZONS:
        raise ValueError("A3 scoring horizons must remain H6-H8")
    moments = fit_task_whitening_moments(
        panel.target,
        task_index=panel.task_index,
        source_episode_id=panel.episode,
        split=panel.split,
        task_names=names,
        horizons=horizons,
    )
    evaluation = panel.evaluation_rows
    evaluation_target = panel.target[evaluation]
    evaluation_history = panel.history[evaluation]
    evaluation_task = panel.task_index[evaluation]
    evaluation_episode = panel.episode[evaluation]
    checked = _validated_predictions(
        predictions,
        seed_count=len(seeds),
        evaluation_count=evaluation.size,
        target_shape=tuple(evaluation_target.shape[1:]),
    )

    centered_losses: dict[str, np.ndarray] = {}
    raw_losses: dict[str, np.ndarray] = {}
    for method in (FACTUAL_METHOD, NO_ACTION_METHOD):
        centered_losses[method], raw_losses[method] = _ordinary_losses(
            evaluation_target,
            checked[method],
            task_index=evaluation_task,
            task_names=names,
            moments=moments,
            horizon_indices=horizon_indices,
            evaluation_split=evaluation_split,
        )
    shuffled_centered = _shuffled_centered_losses(
        evaluation_target,
        checked[SHUFFLED_ACTION_METHOD],
        task_index=evaluation_task,
        task_names=names,
        moments=moments,
        horizon_indices=horizon_indices,
        evaluation_split=evaluation_split,
    )
    centered_losses[SHUFFLED_ACTION_METHOD] = np.mean(
        shuffled_centered, axis=2, dtype=np.float64
    )

    simple = build_secondary_predictions(
        evaluation_history,
        branch_count=BRANCH_COUNT,
        horizon_count=evaluation_target.shape[2],
    )
    for method in (HOLD_LAST_METHOD, LATENT_CONSTANT_VELOCITY_METHOD):
        prediction = np.broadcast_to(
            simple[method][None],
            (len(seeds), *simple[method].shape),
        )
        _, raw_losses[method] = _ordinary_losses(
            evaluation_target,
            prediction,
            task_index=evaluation_task,
            task_names=names,
            moments=moments,
            horizon_indices=horizon_indices,
            evaluation_split=evaluation_split,
        )

    centered_reductions = {
        method: _reduce_losses(
            losses,
            task_index=evaluation_task,
            episode=evaluation_episode,
            task_names=names,
            seeds=seeds,
            evaluation_split=evaluation_split,
        )
        for method, losses in centered_losses.items()
    }
    raw_reductions = {
        method: _reduce_losses(
            losses,
            task_index=evaluation_task,
            episode=evaluation_episode,
            task_names=names,
            seeds=seeds,
            evaluation_split=evaluation_split,
        )
        for method, losses in raw_losses.items()
    }
    shift_reductions = [
        _reduce_losses(
            shuffled_centered[:, :, shift],
            task_index=evaluation_task,
            episode=evaluation_episode,
            task_names=names,
            seeds=seeds,
            evaluation_split=evaluation_split,
        )
        for shift in range(SHUFFLE_SHIFT_COUNT)
    ]

    centered_metric = "h6_h8_within_bundle_branch_centered_whitened_mse"
    raw_metric = "h6_h8_raw_whitened_trajectory_mse"
    comparisons = {
        baseline: _comparison_report(
            centered_reductions[FACTUAL_METHOD],
            centered_reductions[baseline],
            baseline_method=baseline,
            metric=centered_metric,
            task_names=names,
            seeds=seeds,
        )
        for baseline in (NO_ACTION_METHOD, SHUFFLED_ACTION_METHOD)
    }
    shift_reports = [
        {
            "donor_shift": shift + 1,
            **_comparison_report(
                centered_reductions[FACTUAL_METHOD],
                reduction,
                baseline_method=f"shuffled_action_shift_{shift + 1}",
                metric=centered_metric,
                task_names=names,
                seeds=seeds,
            ),
        }
        for shift, reduction in enumerate(shift_reductions)
    ]
    raw_comparisons = {
        baseline: _comparison_report(
            raw_reductions[FACTUAL_METHOD],
            raw_reductions[baseline],
            baseline_method=baseline,
            metric=raw_metric,
            task_names=names,
            seeds=seeds,
        )
        for baseline in (
            NO_ACTION_METHOD,
            HOLD_LAST_METHOD,
            LATENT_CONSTANT_VELOCITY_METHOD,
        )
    }

    joint_tasks = [
        task
        for task in names
        if all(
            comparisons[baseline]["per_task"][task]["effect"] > 0.0
            for baseline in (NO_ACTION_METHOD, SHUFFLED_ACTION_METHOD)
        )
    ]
    joint_seeds = [
        seed
        for seed in seeds
        if all(
            comparisons[baseline]["per_seed"][str(seed)]["effect"] > 0.0
            for baseline in (NO_ACTION_METHOD, SHUFFLED_ACTION_METHOD)
        )
    ]
    cross_replicating_seed_task_counts = {
        seed: sum(
            all(
                comparisons[baseline]["per_task"][task]["per_seed"][str(seed)][
                    "effect"
                ]
                > 0.0
                for baseline in (NO_ACTION_METHOD, SHUFFLED_ACTION_METHOD)
            )
            for task in names
        )
        for seed in seeds
    }
    cross_replicating_seeds = [
        seed
        for seed in seeds
        if cross_replicating_seed_task_counts[seed]
        >= MINIMUM_POSITIVE_TASKS_PER_REPLICATING_SEED
    ]
    raw_reports = {
        method: _method_report(
            reduction, task_names=names, seeds=seeds
        )
        for method, reduction in raw_reductions.items()
    }
    checks = {
        "late_effect_vs_no_action": comparisons[NO_ACTION_METHOD]["effect"]
        >= minimum_effect,
        "late_effect_vs_shuffled_action": comparisons[SHUFFLED_ACTION_METHOD][
            "effect"
        ]
        >= minimum_effect,
        "late_effect_vs_every_shuffled_shift": min(
            report["effect"] for report in shift_reports
        )
        >= minimum_effect,
        "jointly_positive_tasks": len(joint_tasks) >= MINIMUM_POSITIVE_TASKS,
        "jointly_positive_seeds": len(joint_seeds) >= MINIMUM_POSITIVE_SEEDS,
        "cross_seed_task_replication": len(cross_replicating_seeds)
        >= MINIMUM_CROSS_REPLICATING_SEEDS,
        "raw_not_worse_than_no_action": raw_reports[FACTUAL_METHOD][
            "task_macro_mse"
        ]
        <= raw_reports[NO_ACTION_METHOD]["task_macro_mse"],
        "raw_better_than_hold_last": raw_reports[FACTUAL_METHOD]["task_macro_mse"]
        < raw_reports[HOLD_LAST_METHOD]["task_macro_mse"],
        "raw_better_than_constant_velocity": raw_reports[FACTUAL_METHOD][
            "task_macro_mse"
        ]
        < raw_reports[LATENT_CONSTANT_VELOCITY_METHOD]["task_macro_mse"],
    }
    passed = all(checks.values())
    decision = f"{decision_prefix}-Go" if passed else f"{decision_prefix}-No-Go"

    coverage_per_task: dict[str, Any] = {}
    for task_id, task_name in enumerate(names):
        coverage_per_task[task_name] = {
            "train_physical_episode_count": int(
                np.unique(
                    panel.episode[
                        (panel.task_index == task_id) & (panel.split == "train")
                    ]
                ).size
            ),
            f"{evaluation_split}_physical_episode_count": int(
                np.unique(evaluation_episode[evaluation_task == task_id]).size
            ),
            f"{evaluation_split}_hazard_bundle_count": int(
                np.count_nonzero(evaluation_task == task_id)
            ),
        }

    return {
        "protocol": PROTOCOL,
        "stage": stage,
        "decision": decision,
        "passed": bool(passed),
        "input_contains_test_rows": evaluation_split == "test",
        "checkpoint_step": checkpoint_step,
        "primary_horizons": list(horizons),
        "aggregation": [
            "hazard_within_seed_episode",
            "training_seed_within_physical_episode",
            "physical_episode_within_task",
            "equal_weight_task_macro",
        ],
        "coverage": {
            "tasks": list(names),
            "training_seeds": list(seeds),
            f"{evaluation_split}_hazard_bundle_count": int(evaluation.size),
            "per_task": coverage_per_task,
        },
        "whitening": {
            "source": "canonical_train_rows",
            "per_task": True,
            "episode_equal": True,
            "fit_horizons": list(horizons),
            "tasks": {
                task: {
                    "train_physical_episode_count": moments[task].count,
                    "floored_dimensions": moments[task].floored_dimensions,
                }
                for task in names
            },
        },
        "metric": centered_metric,
        "methods": {
            method: _method_report(
                reduction, task_names=names, seeds=seeds
            )
            for method, reduction in centered_reductions.items()
        },
        "comparisons": comparisons,
        "shuffled_action_shifts": {
            "count": SHUFFLE_SHIFT_COUNT,
            "minimum_effect": min(report["effect"] for report in shift_reports),
            "per_shift": shift_reports,
        },
        "raw_h6_h8": {
            "metric": raw_metric,
            "methods": raw_reports,
            "comparisons": raw_comparisons,
        },
        "gate": {
            "minimum_relative_improvement": minimum_effect,
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "jointly_positive_tasks": joint_tasks,
            "jointly_positive_seeds": joint_seeds,
            "cross_replicating_seeds": cross_replicating_seeds,
            "cross_replicating_seed_task_counts": {
                str(seed): cross_replicating_seed_task_counts[seed]
                for seed in seeds
            },
            "passed": bool(passed),
            "decision": decision,
        },
    }


def summarize_a3_metrics(
    canonical_arrays: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
    task_names: Sequence[str] = TASKS,
    primary_horizons: Sequence[int] = PRIMARY_HORIZONS,
    minimum_relative_improvement: float = MINIMUM_RELATIVE_IMPROVEMENT,
    expected_train_episodes: int | None = None,
    expected_development_episodes: int | None = None,
) -> dict[str, Any]:
    """Score the frozen A3 2k train/development panel."""

    seeds = tuple(int(seed) for seed in training_seeds)
    if seeds != TRAINING_SEEDS:
        raise ValueError(f"A3 scoring seeds must remain fixed at {TRAINING_SEEDS}")
    return _summarize_a3_panel(
        canonical_arrays,
        predictions,
        training_seeds=seeds,
        task_names=task_names,
        primary_horizons=primary_horizons,
        minimum_relative_improvement=minimum_relative_improvement,
        evaluation_split="development",
        stage="a3_2k_development",
        decision_prefix="Development",
        checkpoint_step=CHECKPOINT_STEP,
        expected_train_episodes=expected_train_episodes,
        expected_evaluation_episodes=expected_development_episodes,
    )


def summarize_a3_confirmation(
    canonical_arrays: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Score the frozen A3 10k train/test confirmation panel."""

    return _summarize_a3_panel(
        canonical_arrays,
        predictions,
        training_seeds=CONFIRMATION_TRAINING_SEEDS,
        task_names=TASKS,
        primary_horizons=PRIMARY_HORIZONS,
        minimum_relative_improvement=MINIMUM_RELATIVE_IMPROVEMENT,
        evaluation_split="test",
        stage="a3_10k_confirmation",
        decision_prefix="Confirmation",
        checkpoint_step=CONFIRMATION_CHECKPOINT_STEP,
        expected_train_episodes=24,
        expected_evaluation_episodes=12,
    )


summarize_a3_development = summarize_a3_metrics


__all__ = [
    "BRANCH_COUNT",
    "CHECKPOINT_STEP",
    "CONFIRMATION_CHECKPOINT_STEP",
    "CONFIRMATION_SEEDS",
    "CONFIRMATION_TRAINING_SEEDS",
    "MINIMUM_CROSS_REPLICATING_SEEDS",
    "MINIMUM_POSITIVE_SEEDS",
    "MINIMUM_POSITIVE_TASKS",
    "MINIMUM_POSITIVE_TASKS_PER_REPLICATING_SEED",
    "MINIMUM_RELATIVE_IMPROVEMENT",
    "PRIMARY_HORIZONS",
    "PROTOCOL",
    "SHUFFLE_SHIFT_COUNT",
    "TASKS",
    "TRAINING_SEEDS",
    "fit_episode_equal_task_moments",
    "fit_task_whitening_moments",
    "summarize_a3_development",
    "summarize_a3_confirmation",
    "summarize_a3_metrics",
    "validate_a3_scoring_contract",
    "validate_shuffled_prediction_replay",
]
