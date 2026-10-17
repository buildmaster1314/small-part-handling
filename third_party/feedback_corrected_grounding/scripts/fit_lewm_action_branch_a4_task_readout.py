#!/usr/bin/env python3
"""Fit a train-only task-conditioned A4 latent-to-q-center probe.

The historical visualization probe shares one linear map across seven tasks.
That map has a high target-latent decoding floor, especially for custom camera
views.  This diagnostic fits one grouped-CV ridge map per task while retaining
the same three-anchor displacement features and train-only data boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS  # noqa: E402
from scripts.fit_lewm_action_branch_a4_direction_readout import (  # noqa: E402
    ALPHA_GRID,
    ANCHOR_COUNT,
    ANCHOR_SLOT_START,
    SCHEMA_VERSION,
    _aligned_companion,
    _direction_features,
    _episode_equal_weights,
    _expert_branch_slots,
    _expert_states,
    _fit_ridge,
    _group_keys,
    _load_npz_without_reserved_test,
    _predict_ridge,
    _resolve,
    _select_alpha,
)


DEFAULT_CACHE = ROOT / "outputs/lewm_action_branch_a4/formal_v2/state_cache.npz"
DEFAULT_COMPANION = ROOT / (
    "outputs/lewm_action_branch_a4/formal_v2/expert_direction_targets.npz"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_branch_a4/task_conditioned_direction_readout"
)
DECODER_TYPE = "task_conditioned_ridge"


def _task_metrics(error: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(error, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("task readout metrics require finite non-empty errors")
    return {
        "point_count": int(values.size),
        "mean_epe_px": float(np.mean(values)),
        "median_epe_px": float(np.median(values)),
        "p95_epe_px": float(np.quantile(values, 0.95)),
        "pck_at_4": float(np.mean(values <= 4.0)),
        "pck_at_8": float(np.mean(values <= 8.0)),
    }


def fit_task_conditioned_readout(
    cache_path: str | Path,
    companion_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit one train-only ridge decoder for each frozen A4 task."""

    cache = _load_npz_without_reserved_test(cache_path, label="A4 state cache")
    companion = _aligned_companion(companion_path, cache)
    expert_slots = _expert_branch_slots(cache)
    target_state = _expert_states(cache["target_state"], expert_slots)
    feature = _direction_features(target_state, cache["history_state"])
    q_center = np.asarray(companion["q_center_model_raw"], dtype=np.float64)
    target = q_center[:, 1:] - q_center[:, :1]
    if target.shape != (feature.shape[0], HORIZON, 2):
        raise RuntimeError("task readout q-center target shape drifted")

    split = np.asarray(cache["split"]).astype(str)
    task_name = np.asarray(cache["task_name"]).astype(str)
    source_episode = np.asarray(cache["source_episode_id"]).astype(str)
    coefficients = []
    intercepts = []
    selected_alphas = []
    cv_scores = []
    fold_counts = []
    per_task: dict[str, Any] = {}
    all_development_error = []

    for task in TASKS:
        train = (split == "train") & (task_name == task)
        development = (split == "development") & (task_name == task)
        if not bool(np.any(train)) or not bool(np.any(development)):
            raise ValueError(f"task readout requires train/development rows for {task}")

        x_train = feature[train].reshape(-1, feature.shape[-1])
        y_train = target[train].reshape(-1, 2)
        repeated_task = np.repeat(task_name[train], HORIZON)
        repeated_episode = np.repeat(source_episode[train], HORIZON)
        alpha, scores, fold_count = _select_alpha(
            x_train,
            y_train,
            task=repeated_task,
            episode=repeated_episode,
        )
        groups = _group_keys(repeated_task, repeated_episode)
        model = _fit_ridge(
            x_train,
            y_train,
            sample_weight=_episode_equal_weights(groups),
            alpha=alpha,
        )
        x_development = feature[development].reshape(-1, feature.shape[-1])
        y_development = target[development].reshape(-1, 2)
        prediction = _predict_ridge(model, x_development)
        error = np.linalg.norm(prediction - y_development, axis=1)

        coefficients.append(model["coefficient_raw"])
        intercepts.append(model["intercept_raw"])
        selected_alphas.append(alpha)
        cv_scores.append(scores)
        fold_counts.append(fold_count)
        all_development_error.append(error)
        per_task[task] = {
            "train_proposal_count": int(np.count_nonzero(train)),
            "development_proposal_count": int(np.count_nonzero(development)),
            "train_episode_count": int(np.unique(source_episode[train]).size),
            "selected_alpha": float(alpha),
            "cv_fold_count": int(fold_count),
            "cv_mse_by_alpha": {
                str(value): float(score)
                for value, score in zip(ALPHA_GRID, scores, strict=True)
            },
            "development": _task_metrics(error),
        }

    stacked_error = np.concatenate(all_development_error)
    task_macro_mean = float(
        np.mean([per_task[task]["development"]["mean_epe_px"] for task in TASKS])
    )
    decoder = {
        "schema_version": np.asarray(SCHEMA_VERSION + 1, dtype=np.int64),
        "decoder_type": np.asarray(DECODER_TYPE),
        "fit_split": np.asarray("train"),
        "evaluation_split": np.asarray("development"),
        "coordinate_contract": np.asarray(
            "raw_runtime_model_coordinates_no_display_transform"
        ),
        "task_names": np.asarray(TASKS),
        "anchor_slots": np.arange(
            ANCHOR_SLOT_START,
            ANCHOR_SLOT_START + ANCHOR_COUNT,
            dtype=np.int64,
        ),
        "horizons": np.arange(1, HORIZON + 1, dtype=np.int64),
        "alpha_grid": np.asarray(ALPHA_GRID, dtype=np.float64),
        "selected_alpha": np.asarray(selected_alphas, dtype=np.float64),
        "cv_grouped_mse": np.stack(cv_scores).astype(np.float64),
        "cv_fold_count": np.asarray(fold_counts, dtype=np.int64),
        "coefficient_raw": np.stack(coefficients).astype(np.float64),
        "intercept_raw": np.stack(intercepts).astype(np.float64),
    }
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION + 1,
        "decoder_type": DECODER_TYPE,
        "fit_split": "train",
        "evaluation_split": "development",
        "reserved_test_accessed": False,
        "task_macro_mean_epe_px": task_macro_mean,
        "sample_weighted_development": _task_metrics(stacked_error),
        "per_task": per_task,
    }
    return decoder, metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--companion", type=Path, default=DEFAULT_COMPANION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    decoder, metrics = fit_task_conditioned_readout(args.cache, args.companion)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "decoder.npz", **decoder)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decoder": str(output / "decoder.npz"),
                "reserved_test_accessed": False,
                "task_macro_mean_epe_px": metrics["task_macro_mean_epe_px"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
