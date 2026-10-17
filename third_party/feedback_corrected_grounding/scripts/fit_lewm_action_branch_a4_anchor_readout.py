#!/usr/bin/env python3
"""Fit a train-only three-anchor latent-to-pixel readout.

The A4c dynamics model predicts four persistent latent slots.  This probe maps
the three anchor-slot displacements to six pixel coordinates (three points x,
y).  It is deliberately separate from the LeWM checkpoint: LeWM is a latent
world model, while this readout is only for an honest human-facing trajectory
visualization.  It never opens reserved-test rows.
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
from fcpg.lewm_action_branch_a4 import HORIZON, PROTOCOL, TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4_metrics import FACTUAL_METHOD  # noqa: E402
from scripts.fit_lewm_action_branch_a4_direction_readout import (  # noqa: E402
    ALPHA_GRID,
    ANCHOR_COUNT,
    ANCHOR_SLOT_START,
    MAX_CV_FOLDS,
    _direction_features,
    _episode_equal_weights,
    _expert_branch_slots,
    _expert_states,
    _fit_ridge,
    _group_keys,
    _grouped_folds,
    _predict_ridge,
)
from scripts.score_lewm_action_branch_a4 import load_a4_prediction_panel  # noqa: E402


FEATURE_DIM = ANCHOR_COUNT * LATENT_DIM
OUTPUT_DIM = ANCHOR_COUNT * 2
DEFAULT_CACHE = ROOT / "outputs/lewm_action_branch_a4/formal_v2/state_cache.npz"
DEFAULT_TARGETS = ROOT / "outputs/lewm_action_branch_a4/formal_v2/anchor_targets.npz"
DEFAULT_PREDICTIONS = tuple(
    ROOT / f"outputs/lewm_action_branch_a4/shared_7task_seed3020{seed}_2k/evaluation_predictions.npz"
    for seed in (1, 2, 3)
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/three_anchor_readout"


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _load_archive(path: str | Path, *, label: str) -> dict[str, np.ndarray]:
    source = _resolve(path)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if "split" not in arrays:
        raise ValueError(f"{label} is missing split metadata")
    split = arrays["split"].astype(str)
    if np.any(split == "test"):
        raise PermissionError(f"{label} contains reserved-test rows")
    if set(split.tolist()) - {"train", "development"}:
        raise ValueError(f"{label} contains unsupported split labels")
    return arrays


def _align_targets(
    cache: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]
) -> np.ndarray:
    cache_ids = np.asarray(cache["proposal_id"]).astype(str)
    target_ids = np.asarray(targets["proposal_id"]).astype(str)
    if target_ids.ndim != 1 or len(set(target_ids.tolist())) != target_ids.size:
        raise ValueError("anchor target proposal_id values must be unique")
    if set(cache_ids.tolist()) != set(target_ids.tolist()):
        raise ValueError("anchor target archive must cover exactly the state cache")
    positions = {value: index for index, value in enumerate(target_ids.tolist())}
    order = np.asarray([positions[value] for value in cache_ids.tolist()], dtype=np.int64)
    for name in ("split", "task_name", "source_episode_id"):
        if not np.array_equal(
            np.asarray(targets[name])[order].astype(str), np.asarray(cache[name]).astype(str)
        ):
            raise ValueError(f"anchor target {name} is not aligned with the state cache")
    q = np.asarray(targets["q_anchor_model_raw"])[order]
    expected = (cache_ids.size, HORIZON + 1, ANCHOR_COUNT, 2)
    if q.shape != expected or q.dtype.kind not in "fc" or not np.isfinite(q).all():
        raise ValueError(f"q_anchor_model_raw must have finite shape {expected}")
    return q.astype(np.float64, copy=False)


def _select_alpha(feature: np.ndarray, target: np.ndarray, task: np.ndarray, episode: np.ndarray) -> float:
    fold, fold_count = _grouped_folds(task, episode)
    groups = _group_keys(task, episode)
    scores = []
    for alpha in ALPHA_GRID:
        fold_scores = []
        for fold_index in range(fold_count):
            validation = fold == fold_index
            model = _fit_ridge(
                feature[~validation],
                target[~validation],
                sample_weight=_episode_equal_weights(groups[~validation]),
                alpha=float(alpha),
            )
            prediction = _predict_ridge(model, feature[validation])
            error = np.mean(np.square(prediction - target[validation]), axis=1)
            validation_groups = groups[validation]
            fold_scores.append(
                float(
                    np.mean(
                        [np.mean(error[validation_groups == value]) for value in np.unique(validation_groups)]
                    )
                )
            )
        scores.append(float(np.mean(fold_scores)))
    return float(ALPHA_GRID[int(np.argmin(np.asarray(scores)))])


def _fit_task_decoder(
    feature: np.ndarray,
    target: np.ndarray,
    task: np.ndarray,
    episode: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficients = []
    intercepts = []
    selected = []
    for task_name in TASKS:
        mask = task == task_name
        if not np.any(mask):
            raise ValueError(f"train split has no rows for {task_name}")
        alpha = _select_alpha(feature[mask], target[mask], task[mask], episode[mask])
        groups = _group_keys(task[mask], episode[mask])
        model = _fit_ridge(
            feature[mask],
            target[mask],
            sample_weight=_episode_equal_weights(groups),
            alpha=alpha,
        )
        coefficients.append(np.asarray(model["coefficient_raw"], dtype=np.float64))
        intercepts.append(np.asarray(model["intercept_raw"], dtype=np.float64))
        selected.append(alpha)
    return np.stack(coefficients), np.stack(intercepts), np.asarray(selected, dtype=np.float64)


def _predict_task_decoder(
    feature: np.ndarray,
    task: np.ndarray,
    coefficient: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    result = np.empty((feature.shape[0], feature.shape[1], OUTPUT_DIM), dtype=np.float64)
    for task_index, task_name in enumerate(TASKS):
        mask = task == task_name
        if np.any(mask):
            result[mask] = np.einsum(
                "nhf,fo->nho",
                feature[mask],
                coefficient[task_index],
            ) + intercept[task_index][None, None]
    return result


def _trajectory_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(prediction - truth[:, 1:], axis=-1)
    return {
        "mean_epe_px": float(error.mean()),
        "p95_epe_px": float(np.quantile(error, 0.95)),
        "pck_at_4": float(np.mean(error <= 4.0)),
        "pck_at_8": float(np.mean(error <= 8.0)),
    }


def fit_anchor_readout(
    cache_path: str | Path,
    target_path: str | Path,
    prediction_paths: Sequence[str | Path],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache = _load_archive(cache_path, label="A4 state cache")
    validate_a4_state_cache(cache)
    targets = _load_archive(target_path, label="A4 anchor targets")
    q = _align_targets(cache, targets)
    split = np.asarray(cache["split"]).astype(str)
    train = split == "train"
    development = split == "development"
    if not np.any(train) or not np.any(development):
        raise ValueError("three-anchor readout needs train and development rows")

    predictions_cache, predictions, seeds = load_a4_prediction_panel(cache_path, prediction_paths)
    development_ids = np.asarray(cache["proposal_id"])[development].astype(str)
    if not np.array_equal(predictions_cache["proposal_id"].astype(str), development_ids):
        raise RuntimeError("prediction loader returned a differently ordered cache")
    expert_slots = _expert_branch_slots(cache)
    target_expert = _expert_states(np.asarray(cache["target_state"]), expert_slots)
    feature_all = _direction_features(target_expert, np.asarray(cache["history_state"]))
    target_delta = q[:, 1:] - q[:, :1]
    feature_train = feature_all[train]
    target_train = target_delta[train]
    task_all = np.asarray(cache["task_name"]).astype(str)
    episode_all = np.asarray(cache["source_episode_id"]).astype(str)
    task_train = np.repeat(task_all[train], HORIZON)
    episode_train = np.repeat(episode_all[train], HORIZON)
    coefficient, intercept, alpha = _fit_task_decoder(
        feature_train.reshape(-1, FEATURE_DIM),
        target_train.reshape(-1, OUTPUT_DIM),
        task_train,
        episode_train,
    )

    dev_features_by_seed = []
    factual = predictions[FACTUAL_METHOD]
    for seed_index in range(factual.shape[0]):
        expert_prediction = _expert_states(factual[seed_index], expert_slots[development])
        dev_features_by_seed.append(
            _direction_features(expert_prediction, np.asarray(cache["history_state"])[development])
        )
    dev_features = np.mean(np.stack(dev_features_by_seed), axis=0)
    dev_task = task_all[development]
    dev_delta = _predict_task_decoder(dev_features, dev_task, coefficient, intercept)
    dev_q = np.concatenate(
        [q[development, :1], q[development, :1] + dev_delta.reshape(-1, HORIZON, ANCHOR_COUNT, 2)],
        axis=1,
    )
    metrics = {
        "schema_version": 1,
        "protocol": "a4_three_anchor_visualization_readout_v1",
        "fit_split": "train",
        "evaluation_split": "development",
        "reserved_test_opened": False,
        "anchor_count": ANCHOR_COUNT,
        "latent_feature_dim": FEATURE_DIM,
        "output_dim": OUTPUT_DIM,
        "training_seeds": list(seeds),
        "selected_alpha_by_task": alpha.tolist(),
        "development_metrics": _trajectory_metrics(q[development], dev_q),
        "notes": (
            "The LeWM checkpoint remains a latent model. This train-only linear "
            "probe is used solely to render three persistent anchor coordinates; "
            "it is not a policy or a new sealed-test result."
        ),
    }
    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "protocol": np.asarray("a4_three_anchor_visualization_readout_v1"),
        "fit_split": np.asarray("train"),
        "evaluation_split": np.asarray("development"),
        "coordinate_contract": np.asarray("raw_runtime_model_coordinates_no_display_transform"),
        "target_definition": np.asarray("q_anchor_model_raw_h_minus_h0"),
        "task_names": np.asarray(TASKS),
        "anchor_slots": np.arange(ANCHOR_SLOT_START, ANCHOR_SLOT_START + ANCHOR_COUNT, dtype=np.int64),
        "horizons": np.arange(1, HORIZON + 1, dtype=np.int64),
        "alpha_grid": np.asarray(ALPHA_GRID, dtype=np.float64),
        "selected_alpha": alpha,
        "coefficient_raw": coefficient,
        "intercept_raw": intercept,
        "reserved_test_opened": np.asarray(False, dtype=bool),
    }
    return arrays, metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--predictions", type=Path, nargs="+", default=list(DEFAULT_PREDICTIONS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    arrays, metrics = fit_anchor_readout(args.cache, args.targets, args.predictions)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "decoder.npz", **arrays)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
