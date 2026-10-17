#!/usr/bin/env python3
"""Fit a train-only A4 latent-to-image-direction diagnostic readout.

The readout maps the three persistent anchor-token displacements for the
semantic Expert branch to the corresponding 2-D center displacement.  Ridge
regularization is selected with source-episode-grouped cross-validation using
train rows only.  Development rows are used exactly once to compare the
target-latent ceiling with the three-seed factual LeWM trajectory.

This is a visualization diagnostic, not a latent-to-RGB decoder.  It refuses
any archive containing the sealed reserved-test split.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_storage import LATENT_DIM, validate_a4_state_cache  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    HORIZON,
    PROTOCOL,
    TASKS,
)
from fcpg.lewm_action_branch_a4_metrics import FACTUAL_METHOD  # noqa: E402
from scripts.score_lewm_action_branch_a4 import (  # noqa: E402
    load_a4_prediction_panel,
)


SCHEMA_VERSION = 1
EXPERT_SEMANTIC_INDEX = 0
ANCHOR_SLOT_START = 1
ANCHOR_COUNT = 3
FEATURE_DIM = ANCHOR_COUNT * LATENT_DIM
ALPHA_GRID = (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
MAX_CV_FOLDS = 5
NUMERIC_EPSILON = 1.0e-12
INFORMATIVE_DISPLACEMENT_PX = 2.0
TARGET_MAX_MEAN_EPE_PX = 2.0
TARGET_MAX_P95_EPE_PX = 8.0
TARGET_MIN_PCK8 = 0.90
FACTUAL_MAX_MEAN_EPE_PX = 4.0
FACTUAL_MAX_P95_EPE_PX = 16.0
FACTUAL_MIN_POSITIVE_COSINE_RATE = 0.75
FACTUAL_MIN_TASKS_BEATING_STATIC = 6

DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/direction_readout"


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _load_npz_without_reserved_test(
    path: str | Path, *, label: str
) -> dict[str, np.ndarray]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        if "split" not in archive.files:
            raise ValueError(f"{label} is missing split metadata")
        split = np.asarray(archive["split"])
        if split.ndim != 1 or split.dtype.kind not in {"U", "S"}:
            raise TypeError(f"{label} split must be a string vector")
        names = split.astype(str, copy=False)
        if np.any(names == "test"):
            raise PermissionError(
                f"{label} contains reserved test rows; direction readout is development-only"
            )
        unknown = sorted(set(names.tolist()) - {"train", "development"})
        if unknown:
            raise ValueError(f"{label} contains unsupported split labels: {unknown}")
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _aligned_companion(
    companion_path: str | Path, cache: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    companion = _load_npz_without_reserved_test(
        companion_path, label="A4 direction companion"
    )
    required = {
        "proposal_id",
        "split",
        "task_name",
        "source_episode_id",
    }
    missing = sorted(required - set(companion))
    if missing:
        raise ValueError(f"A4 direction companion is missing fields: {missing}")

    cache_ids = np.asarray(cache["proposal_id"]).astype(str)
    companion_ids = np.asarray(companion["proposal_id"])
    if companion_ids.shape != cache_ids.shape or companion_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("companion proposal_id must be a cache-sized string vector")
    companion_ids = companion_ids.astype(str, copy=False)
    if len(set(companion_ids.tolist())) != companion_ids.size:
        raise ValueError("companion proposal_id values must be unique")
    if set(companion_ids.tolist()) != set(cache_ids.tolist()):
        raise ValueError("companion must cover every and only state-cache proposal_id")
    positions = {value: index for index, value in enumerate(companion_ids.tolist())}
    order = np.asarray([positions[value] for value in cache_ids.tolist()], dtype=np.int64)
    aligned = {
        name: np.asarray(value)[order]
        for name, value in companion.items()
        if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == cache_ids.size
    }
    for name in ("proposal_id", "split", "task_name", "source_episode_id"):
        observed = np.asarray(aligned[name]).astype(str)
        expected = np.asarray(cache[name]).astype(str)
        if not np.array_equal(observed, expected):
            raise ValueError(f"companion {name} differs from the state cache")
    q_aliases = [
        name for name in ("q_center_model_raw", "q_center_model") if name in aligned
    ]
    if not q_aliases:
        raise ValueError(
            "A4 direction companion is missing q_center_model_raw/q_center_model"
        )
    q_center = np.asarray(aligned[q_aliases[0]])
    for name in q_aliases[1:]:
        if not np.array_equal(q_center, np.asarray(aligned[name])):
            raise ValueError("companion q_center_model aliases disagree")
    expected_q_shape = (cache_ids.size, HORIZON + 1, 2)
    if q_center.shape != expected_q_shape:
        raise ValueError(f"q_center_model must have shape {expected_q_shape}")
    if q_center.dtype.kind != "f" or not np.isfinite(q_center).all():
        raise TypeError("q_center_model must be a finite floating array")
    aligned["q_center_model_raw"] = q_center.astype(np.float64, copy=False)
    aligned["q_center_source_field"] = np.asarray(q_aliases[0])
    return aligned


def _expert_branch_slots(cache: Mapping[str, np.ndarray]) -> np.ndarray:
    semantic = np.asarray(cache["branch_semantic_index"], dtype=np.int64)
    matches = semantic == EXPERT_SEMANTIC_INDEX
    if not np.all(np.sum(matches, axis=1) == 1):
        raise ValueError("each A4 cache row must contain exactly one semantic Expert branch")
    return np.argmax(matches, axis=1).astype(np.int64, copy=False)


def _expert_states(states: np.ndarray, expert_slots: np.ndarray) -> np.ndarray:
    values = np.asarray(states)
    expected_tail = (HORIZON, 4, LATENT_DIM)
    if values.ndim != 5 or values.shape[1:] != (8, *expected_tail):
        raise ValueError(
            "A4 states must have shape [N,8,8,4,192] before Expert selection"
        )
    if expert_slots.shape != (values.shape[0],):
        raise ValueError("Expert branch slots must align with state rows")
    return values[np.arange(values.shape[0]), expert_slots]


def _direction_features(
    expert_state: np.ndarray, history_state: np.ndarray
) -> np.ndarray:
    future = np.asarray(expert_state, dtype=np.float64)
    history = np.asarray(history_state, dtype=np.float64)
    if future.shape[1:] != (HORIZON, 4, LATENT_DIM):
        raise ValueError("Expert state must have shape [N,8,4,192]")
    if history.ndim != 4 or history.shape[0] != future.shape[0] or history.shape[2:] != (
        4,
        LATENT_DIM,
    ):
        raise ValueError("history_state must have aligned shape [N,T,4,192]")
    last_anchor = history[:, -1, ANCHOR_SLOT_START:, :]
    future_anchor = future[:, :, ANCHOR_SLOT_START:, :]
    feature = future_anchor - last_anchor[:, None]
    expected = (future.shape[0], HORIZON, ANCHOR_COUNT, LATENT_DIM)
    if feature.shape != expected:
        raise RuntimeError(f"direction feature shape drifted: {feature.shape} != {expected}")
    return feature.reshape(future.shape[0], HORIZON, FEATURE_DIM)


def _group_keys(task: np.ndarray, episode: np.ndarray) -> np.ndarray:
    task_name = np.asarray(task).astype(str)
    episode_id = np.asarray(episode).astype(str)
    if task_name.shape != episode_id.shape or task_name.ndim != 1:
        raise ValueError("task and source episode identity must be aligned vectors")
    return np.asarray(
        [f"{name}\x1f{identity}" for name, identity in zip(task_name, episode_id, strict=True)]
    )


def _episode_equal_weights(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups).astype(str)
    identities, counts = np.unique(values, return_counts=True)
    if identities.size == 0:
        raise ValueError("ridge fitting requires at least one source episode")
    count_by_group = dict(zip(identities.tolist(), counts.tolist(), strict=True))
    weights = np.asarray(
        [1.0 / (identities.size * count_by_group[value]) for value in values],
        dtype=np.float64,
    )
    if not np.isclose(weights.sum(), 1.0, atol=1.0e-12, rtol=1.0e-10):
        raise RuntimeError("episode-equal weights do not sum to one")
    return weights


def _grouped_folds(task: np.ndarray, episode: np.ndarray) -> tuple[np.ndarray, int]:
    task_name = np.asarray(task).astype(str)
    episode_id = np.asarray(episode).astype(str)
    if task_name.shape != episode_id.shape or task_name.ndim != 1:
        raise ValueError("CV task and source episode vectors must align")
    counts = []
    for name in sorted(set(task_name.tolist())):
        counts.append(np.unique(episode_id[task_name == name]).size)
    fold_count = min(MAX_CV_FOLDS, min(counts, default=0))
    if fold_count < 2:
        raise ValueError("episode-group CV requires at least two train episodes per task")
    fold = np.empty(task_name.size, dtype=np.int64)
    for name in sorted(set(task_name.tolist())):
        selected = task_name == name
        identities = sorted(np.unique(episode_id[selected]).tolist())
        assignment = {identity: index % fold_count for index, identity in enumerate(identities)}
        fold[selected] = np.asarray([assignment[value] for value in episode_id[selected]])
    return fold, fold_count


def _fit_ridge(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge feature/target must be aligned non-empty matrices")
    if weight.shape != (x.shape[0],) or np.any(weight <= 0.0):
        raise ValueError("ridge sample weights must be positive and row-aligned")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weight).all():
        raise ValueError("ridge inputs must be finite")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("ridge alpha must be positive and finite")
    weight = weight / weight.sum()
    x_mean = np.sum(weight[:, None] * x, axis=0)
    variance = np.sum(weight[:, None] * np.square(x - x_mean), axis=0)
    keep = variance > NUMERIC_EPSILON
    scale = np.ones(x.shape[1], dtype=np.float64)
    scale[keep] = np.sqrt(variance[keep])
    y_mean = np.sum(weight[:, None] * y, axis=0)
    coefficient_standardized = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    if np.any(keep):
        standardized = (x[:, keep] - x_mean[keep]) / scale[keep]
        root_weight = np.sqrt(weight)[:, None]
        weighted_x = standardized * root_weight
        weighted_y = (y - y_mean) * root_weight
        if weighted_x.shape[1] <= weighted_x.shape[0]:
            gram = weighted_x.T @ weighted_x
            gram.flat[:: gram.shape[0] + 1] += alpha
            active_coefficient = np.linalg.solve(gram, weighted_x.T @ weighted_y)
        else:
            gram = weighted_x @ weighted_x.T
            gram.flat[:: gram.shape[0] + 1] += alpha
            active_coefficient = weighted_x.T @ np.linalg.solve(gram, weighted_y)
        coefficient_standardized[keep] = active_coefficient
    coefficient_raw = coefficient_standardized / scale[:, None]
    intercept_raw = y_mean - x_mean @ coefficient_raw
    return {
        "feature_mean": x_mean,
        "feature_scale": scale,
        "active_feature_mask": keep,
        "coefficient_standardized": coefficient_standardized,
        "coefficient_raw": coefficient_raw,
        "intercept_raw": intercept_raw,
        "target_mean": y_mean,
    }


def _predict_ridge(model: Mapping[str, np.ndarray], feature: np.ndarray) -> np.ndarray:
    x = np.asarray(feature, dtype=np.float64)
    coefficient = np.asarray(model["coefficient_raw"], dtype=np.float64)
    intercept = np.asarray(model["intercept_raw"], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != coefficient.shape[0]:
        raise ValueError("ridge evaluation feature dimension differs from decoder")
    result = x @ coefficient + intercept
    if not np.isfinite(result).all():
        raise RuntimeError("direction decoder produced non-finite output")
    return result


def _select_alpha(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    task: np.ndarray,
    episode: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    fold, fold_count = _grouped_folds(task, episode)
    groups = _group_keys(task, episode)
    scores = np.empty(len(ALPHA_GRID), dtype=np.float64)
    for alpha_index, alpha in enumerate(ALPHA_GRID):
        fold_scores = []
        for fold_index in range(fold_count):
            validation = fold == fold_index
            training = ~validation
            model = _fit_ridge(
                feature[training],
                target[training],
                sample_weight=_episode_equal_weights(groups[training]),
                alpha=alpha,
            )
            prediction = _predict_ridge(model, feature[validation])
            squared = np.mean(np.square(prediction - target[validation]), axis=1)
            validation_groups = groups[validation]
            fold_scores.append(
                float(
                    np.mean(
                        [
                            np.mean(squared[validation_groups == identity])
                            for identity in np.unique(validation_groups)
                        ]
                    )
                )
            )
        scores[alpha_index] = float(np.mean(fold_scores))
    selected_index = min(range(len(ALPHA_GRID)), key=lambda index: (scores[index], ALPHA_GRID[index]))
    return float(ALPHA_GRID[selected_index]), scores, fold_count


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if data.shape != weight.shape or data.size == 0:
        raise ValueError("weighted quantile requires aligned non-empty vectors")
    if np.any(weight < 0.0) or not np.any(weight > 0.0):
        raise ValueError("weighted quantile weights must be non-negative and non-zero")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in 0..1")
    order = np.argsort(data, kind="stable")
    sorted_data = data[order]
    cumulative = np.cumsum(weight[order])
    threshold = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), data.size - 1)
    return float(sorted_data[index])


def _balanced_point_weights(
    task: np.ndarray,
    episode: np.ndarray,
    point_mask: np.ndarray,
) -> np.ndarray:
    task_name = np.asarray(task).astype(str)
    episode_id = np.asarray(episode).astype(str)
    mask = np.asarray(point_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != task_name.size or episode_id.shape != task_name.shape:
        raise ValueError("point mask must align with task/source episode rows")
    weights = np.zeros(mask.shape, dtype=np.float64)
    supported_tasks = [
        name for name in sorted(set(task_name.tolist())) if np.any(mask[task_name == name])
    ]
    if not supported_tasks:
        return weights
    for name in supported_tasks:
        task_rows = task_name == name
        supported_episodes = [
            identity
            for identity in np.unique(episode_id[task_rows]).tolist()
            if np.any(mask[task_rows & (episode_id == identity)])
        ]
        for identity in supported_episodes:
            rows = task_rows & (episode_id == identity)
            selected = mask & rows[:, None]
            count = int(np.count_nonzero(selected))
            weights[selected] = 1.0 / (
                len(supported_tasks) * len(supported_episodes) * count
            )
    if not np.isclose(weights.sum(), 1.0, atol=1.0e-12, rtol=1.0e-10):
        raise RuntimeError("task/episode-balanced point weights do not sum to one")
    return weights


def _trajectory_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    task: np.ndarray,
    episode: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(ground_truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 3 or truth.shape[1:] != (
        HORIZON + 1,
        2,
    ):
        raise ValueError("trajectory metrics require aligned [N,9,2] paths")
    delta_error = estimate[:, 1:] - truth[:, 1:]
    point_epe = np.linalg.norm(delta_error, axis=-1)
    error = np.mean(np.square(delta_error), axis=(1, 2))
    endpoint_error = np.mean(np.square(estimate[:, -1] - truth[:, -1]), axis=1)
    static = np.repeat(truth[:, :1], HORIZON + 1, axis=1)
    static_error = np.mean(np.square(static[:, 1:] - truth[:, 1:]), axis=(1, 2))
    true_displacement = truth[:, 1:] - truth[:, :1]
    predicted_displacement = estimate[:, 1:] - estimate[:, :1]
    static_point_epe = np.linalg.norm(true_displacement, axis=-1)
    all_points = np.ones(point_epe.shape, dtype=bool)
    point_weights = _balanced_point_weights(task, episode, all_points)
    informative = np.linalg.norm(true_displacement, axis=-1) >= INFORMATIVE_DISPLACEMENT_PX
    informative_weights = _balanced_point_weights(task, episode, informative)
    true_norm = np.linalg.norm(true_displacement, axis=-1)
    predicted_norm = np.linalg.norm(predicted_displacement, axis=-1)
    cosine = np.full(point_epe.shape, -1.0, dtype=np.float64)
    defined = informative & (predicted_norm > NUMERIC_EPSILON)
    cosine[defined] = np.sum(
        true_displacement[defined] * predicted_displacement[defined], axis=-1
    ) / (true_norm[defined] * predicted_norm[defined])
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(cosine))
    per_task: dict[str, Any] = {}
    task_mse = []
    task_endpoint_mse = []
    task_static_mse = []
    for name in sorted(set(np.asarray(task).astype(str).tolist())):
        selected_task = np.asarray(task).astype(str) == name
        task_episodes = np.asarray(episode).astype(str)[selected_task]
        identities = np.unique(task_episodes)
        episode_mse = np.asarray(
            [np.mean(error[selected_task][task_episodes == identity]) for identity in identities]
        )
        episode_endpoint = np.asarray(
            [
                np.mean(endpoint_error[selected_task][task_episodes == identity])
                for identity in identities
            ]
        )
        episode_static = np.asarray(
            [
                np.mean(static_error[selected_task][task_episodes == identity])
                for identity in identities
            ]
        )
        mse = float(np.mean(episode_mse))
        endpoint_mse = float(np.mean(episode_endpoint))
        baseline_mse = float(np.mean(episode_static))
        task_mse.append(mse)
        task_endpoint_mse.append(endpoint_mse)
        task_static_mse.append(baseline_mse)
        task_point_weights = _balanced_point_weights(
            np.asarray(task)[selected_task],
            np.asarray(episode)[selected_task],
            np.ones(point_epe[selected_task].shape, dtype=bool),
        )
        task_informative = informative[selected_task]
        task_informative_weights = _balanced_point_weights(
            np.asarray(task)[selected_task],
            np.asarray(episode)[selected_task],
            task_informative,
        )
        per_task[name] = {
            "proposal_count": int(np.count_nonzero(selected_task)),
            "episode_count": int(identities.size),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "endpoint_mse": endpoint_mse,
            "endpoint_rmse": float(np.sqrt(endpoint_mse)),
            "static_q0_mse": baseline_mse,
            "relative_improvement_vs_static_q0": float(
                1.0 - mse / max(baseline_mse, NUMERIC_EPSILON)
            ),
            "mean_epe": float(
                np.sum(task_point_weights * point_epe[selected_task])
            ),
            "median_epe": _weighted_quantile(
                point_epe[selected_task], task_point_weights, 0.5
            ),
            "p95_epe": _weighted_quantile(
                point_epe[selected_task], task_point_weights, 0.95
            ),
            "pck_at_4": float(
                np.sum(task_point_weights * (point_epe[selected_task] <= 4.0))
            ),
            "pck_at_8": float(
                np.sum(task_point_weights * (point_epe[selected_task] <= 8.0))
            ),
            "static_q0_mean_epe": float(
                np.sum(task_point_weights * static_point_epe[selected_task])
            ),
            "informative_point_count": int(np.count_nonzero(task_informative)),
            "positive_cosine_rate": (
                float(
                    np.sum(
                        task_informative_weights[task_informative]
                        * (cosine[selected_task][task_informative] > 0.0)
                    )
                )
                if np.any(task_informative)
                else None
            ),
            "median_angular_error_degrees": (
                _weighted_quantile(
                    angle_degrees[selected_task][task_informative],
                    task_informative_weights[task_informative],
                    0.5,
                )
                if np.any(task_informative)
                else None
            ),
        }
    macro_mse = float(np.mean(task_mse))
    macro_endpoint_mse = float(np.mean(task_endpoint_mse))
    macro_static_mse = float(np.mean(task_static_mse))
    informative_count = int(np.count_nonzero(informative))
    informative_summary = {
        "minimum_displacement_px": INFORMATIVE_DISPLACEMENT_PX,
        "point_count": informative_count,
        "task_count": int(
            len(
                {
                    str(name)
                    for name, row in zip(np.asarray(task).astype(str), informative, strict=True)
                    if np.any(row)
                }
            )
        ),
        "positive_cosine_rate": None,
        "median_cosine": None,
        "median_angular_error_degrees": None,
    }
    if informative_count:
        selected_weights = informative_weights[informative]
        selected_cosine = cosine[informative]
        selected_angle = angle_degrees[informative]
        informative_summary.update(
            {
                "positive_cosine_rate": float(
                    np.sum(selected_weights * (selected_cosine > 0.0))
                ),
                "median_cosine": _weighted_quantile(
                    selected_cosine, selected_weights, 0.5
                ),
                "median_angular_error_degrees": _weighted_quantile(
                    selected_angle, selected_weights, 0.5
                ),
            }
        )
    return {
        "aggregation": "task_then_source_physical_episode",
        "task_macro_mse": macro_mse,
        "task_macro_rmse": float(np.sqrt(macro_mse)),
        "task_macro_endpoint_mse": macro_endpoint_mse,
        "task_macro_endpoint_rmse": float(np.sqrt(macro_endpoint_mse)),
        "task_macro_static_q0_mse": macro_static_mse,
        "relative_improvement_vs_static_q0": float(
            1.0 - macro_mse / max(macro_static_mse, NUMERIC_EPSILON)
        ),
        "macro_mean_epe": float(np.sum(point_weights * point_epe)),
        "macro_median_epe": _weighted_quantile(
            point_epe, point_weights, 0.5
        ),
        "macro_p95_epe": _weighted_quantile(point_epe, point_weights, 0.95),
        "macro_pck_at_4": float(np.sum(point_weights * (point_epe <= 4.0))),
        "macro_pck_at_8": float(np.sum(point_weights * (point_epe <= 8.0))),
        "macro_static_q0_mean_epe": float(
            np.sum(point_weights * static_point_epe)
        ),
        "informative_direction": informative_summary,
        "per_task": per_task,
    }


def fit_direction_readout(
    cache_path: str | Path,
    companion_path: str | Path,
    prediction_paths: Sequence[str | Path],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Fit the train-only decoder and return decoder, paths, and metrics."""

    cache, predictions, seeds = load_a4_prediction_panel(
        cache_path, prediction_paths
    )
    validate_a4_state_cache(cache)
    split = np.asarray(cache["split"]).astype(str)
    if np.any(split == "test"):
        raise PermissionError("A4 direction readout must not open reserved-test rows")
    companion = _aligned_companion(companion_path, cache)
    train = split == "train"
    development = split == "development"
    if not np.any(train) or not np.any(development):
        raise ValueError("direction readout requires both train and development rows")

    expert_slots = _expert_branch_slots(cache)
    target_expert = _expert_states(np.asarray(cache["target_state"]), expert_slots)
    target_feature = _direction_features(target_expert, np.asarray(cache["history_state"]))
    q = np.asarray(companion["q_center_model_raw"], dtype=np.float64)
    displacement = q[:, 1:] - q[:, :1]

    train_feature = target_feature[train].reshape(-1, FEATURE_DIM)
    train_target = displacement[train].reshape(-1, 2)
    train_task = np.repeat(np.asarray(cache["task_name"])[train].astype(str), HORIZON)
    train_episode = np.repeat(
        np.asarray(cache["source_episode_id"])[train].astype(str), HORIZON
    )
    alpha, cv_mse, fold_count = _select_alpha(
        train_feature,
        train_target,
        task=train_task,
        episode=train_episode,
    )
    groups = _group_keys(train_task, train_episode)
    decoder = _fit_ridge(
        train_feature,
        train_target,
        sample_weight=_episode_equal_weights(groups),
        alpha=alpha,
    )

    q_development = q[development]
    target_delta = _predict_ridge(
        decoder, target_feature[development].reshape(-1, FEATURE_DIM)
    ).reshape(-1, HORIZON, 2)
    q_decoded_target = np.concatenate(
        [q_development[:, :1], q_development[:, :1] + target_delta], axis=1
    )

    development_expert_slots = expert_slots[development]
    factual = np.asarray(predictions[FACTUAL_METHOD], dtype=np.float64)
    seed_features = []
    for seed_index in range(factual.shape[0]):
        expert_prediction = _expert_states(
            factual[seed_index], development_expert_slots
        )
        seed_features.append(
            _direction_features(
                expert_prediction,
                np.asarray(cache["history_state"])[development],
            )
        )
    q_factual_by_seed = []
    for feature in seed_features:
        delta = _predict_ridge(
            decoder, feature.reshape(-1, FEATURE_DIM)
        ).reshape(-1, HORIZON, 2)
        q_factual_by_seed.append(
            np.concatenate(
                [q_development[:, :1], q_development[:, :1] + delta], axis=1
            )
        )
    q_decoded_factual_by_seed = np.stack(q_factual_by_seed, axis=0)
    q_decoded_factual = np.mean(q_decoded_factual_by_seed, axis=0)

    task_development = np.asarray(cache["task_name"])[development].astype(str)
    episode_development = np.asarray(cache["source_episode_id"])[development].astype(str)
    if set(task_development.tolist()) != set(TASKS):
        raise ValueError("formal direction readout development must cover all seven A4 tasks")
    target_metrics = _trajectory_metrics(
        q_development,
        q_decoded_target,
        task=task_development,
        episode=episode_development,
    )
    factual_metrics = _trajectory_metrics(
        q_development,
        q_decoded_factual,
        task=task_development,
        episode=episode_development,
    )
    target_checks = {
        "mean_epe_at_most_2px": target_metrics["macro_mean_epe"]
        <= TARGET_MAX_MEAN_EPE_PX,
        "p95_epe_at_most_8px": target_metrics["macro_p95_epe"]
        <= TARGET_MAX_P95_EPE_PX,
        "pck_at_8_at_least_0_90": target_metrics["macro_pck_at_8"]
        >= TARGET_MIN_PCK8,
    }
    factual_tasks_beating_static = sum(
        row["mean_epe"] < row["static_q0_mean_epe"]
        for row in factual_metrics["per_task"].values()
    )
    positive_cosine_rate = factual_metrics["informative_direction"][
        "positive_cosine_rate"
    ]
    factual_checks = {
        "mean_epe_at_most_4px": factual_metrics["macro_mean_epe"]
        <= FACTUAL_MAX_MEAN_EPE_PX,
        "p95_epe_at_most_16px": factual_metrics["macro_p95_epe"]
        <= FACTUAL_MAX_P95_EPE_PX,
        "positive_cosine_rate_at_least_0_75": positive_cosine_rate is not None
        and positive_cosine_rate >= FACTUAL_MIN_POSITIVE_COSINE_RATE,
        "at_least_6_of_7_tasks_beat_static_q0": factual_tasks_beating_static
        >= FACTUAL_MIN_TASKS_BEATING_STATIC,
    }
    target_check = all(target_checks.values())
    factual_check = all(factual_checks.values())
    per_task = {
        name: {
            "target_ceiling": target_metrics["per_task"][name],
            "factual": factual_metrics["per_task"][name],
        }
        for name in sorted(target_metrics["per_task"])
    }

    decoder_arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "protocol": np.asarray(PROTOCOL),
        "fit_split": np.asarray("train"),
        "evaluation_split": np.asarray("development"),
        "feature_definition": np.asarray(
            "expert_future_anchor_tokens_minus_last_history_anchor_tokens"
        ),
        "target_definition": np.asarray("q_center_model_raw_h_minus_h0"),
        "coordinate_contract": np.asarray(
            "raw_runtime_model_coordinates_no_display_transform"
        ),
        "companion_q_source_field": np.asarray(
            companion["q_center_source_field"]
        ),
        "anchor_slots": np.arange(
            ANCHOR_SLOT_START, ANCHOR_SLOT_START + ANCHOR_COUNT, dtype=np.int64
        ),
        "horizons": np.arange(1, HORIZON + 1, dtype=np.int64),
        "alpha_grid": np.asarray(ALPHA_GRID, dtype=np.float64),
        "selected_alpha": np.asarray(alpha, dtype=np.float64),
        "cv_grouped_mse": cv_mse,
        "cv_fold_count": np.asarray(fold_count, dtype=np.int64),
        "feature_mean": decoder["feature_mean"],
        "feature_scale": decoder["feature_scale"],
        "active_feature_mask": decoder["active_feature_mask"],
        "coefficient_standardized": decoder["coefficient_standardized"],
        "coefficient_raw": decoder["coefficient_raw"],
        "intercept_raw": decoder["intercept_raw"],
        "target_mean": decoder["target_mean"],
    }
    path_arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "coordinate_contract": np.asarray(
            "raw_runtime_model_coordinates_no_display_transform"
        ),
        "proposal_id": np.asarray(cache["proposal_id"])[development],
        "split": np.asarray(cache["split"])[development],
        "task_name": np.asarray(cache["task_name"])[development],
        "source_episode_id": np.asarray(cache["source_episode_id"])[development],
        "q_ground_truth_model": q_development.astype(np.float32),
        "q_decoded_target": q_decoded_target.astype(np.float32),
        "q_decoded_factual": q_decoded_factual.astype(np.float32),
        "q_decoded_factual_by_seed": q_decoded_factual_by_seed.astype(np.float32),
        "training_seed": np.asarray(seeds, dtype=np.int64),
    }
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "stage": "a4_direction_readout_development",
        "gate_pass": bool(target_check and factual_check),
        "render_factual_path_authorized": bool(target_check and factual_check),
        "reserved_test_accessed": False,
        "fit_split": "train",
        "evaluation_split": "development",
        "coordinate_contract": "raw_runtime_model_coordinates_no_display_transform",
        "training_seed": list(seeds),
        "train_proposal_count": int(np.count_nonzero(train)),
        "development_proposal_count": int(np.count_nonzero(development)),
        "train_episode_group_count": int(np.unique(groups).size),
        "selected_alpha": alpha,
        "alpha_grid": list(ALPHA_GRID),
        "cross_validation": {
            "group": "task_name/source_episode_id",
            "fold_count": fold_count,
            "mse_by_alpha": {
                str(value): float(score)
                for value, score in zip(ALPHA_GRID, cv_mse, strict=True)
            },
        },
        "gate": {
            "scope": "visualization_diagnostic_only",
            "criterion": (
                "strict raw-model-coordinate EPE, tail, PCK, direction, and "
                "per-task static-baseline checks before a factual path may be rendered"
            ),
            "checks": {
                "target_ceiling": {name: bool(value) for name, value in target_checks.items()},
                "factual": {name: bool(value) for name, value in factual_checks.items()},
            },
            "thresholds": {
                "target_max_macro_mean_epe_px": TARGET_MAX_MEAN_EPE_PX,
                "target_max_macro_p95_epe_px": TARGET_MAX_P95_EPE_PX,
                "target_min_macro_pck_at_8": TARGET_MIN_PCK8,
                "factual_max_macro_mean_epe_px": FACTUAL_MAX_MEAN_EPE_PX,
                "factual_max_macro_p95_epe_px": FACTUAL_MAX_P95_EPE_PX,
                "factual_min_positive_cosine_rate": FACTUAL_MIN_POSITIVE_COSINE_RATE,
                "factual_min_tasks_beating_static_q0": FACTUAL_MIN_TASKS_BEATING_STATIC,
            },
            "factual_tasks_beating_static_q0": factual_tasks_beating_static,
        },
        "target_ceiling": target_metrics,
        "factual": factual_metrics,
        "per_task": per_task,
    }
    return decoder_arrays, path_arrays, metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    decoder, paths, metrics = fit_direction_readout(
        args.cache, args.companion, args.predictions
    )
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "decoder.npz", **decoder)
    np.savez_compressed(output / "development_predictions.npz", **paths)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate_pass": metrics["gate_pass"],
                "output": str(output),
                "reserved_test_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
