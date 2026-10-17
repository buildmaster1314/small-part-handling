#!/usr/bin/env python3
"""Phase-0 nonlinear geometry gate for A4 q-center prediction.

The geometry head is deliberately action- and time-blind.  It is trained only
on canonical target latents from physical train episodes, then frozen and
applied unchanged to target, factual, matched no-action, and all shuffled-action
development latents.  The sealed reserved test is never opened.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_storage import ACTION_DIM, LATENT_DIM, validate_a4_state_cache  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    BRANCH_COUNT,
    DEVELOPMENT_TRAINING_SEEDS,
    HORIZON,
    TASKS,
)
from fcpg.lewm_action_branch_a4_metrics import (  # noqa: E402
    FACTUAL_METHOD,
    NO_ACTION_METHOD,
    WRONG_ACTION_METHOD,
)
from scripts.fit_lewm_action_branch_a4_direction_readout import (  # noqa: E402
    _aligned_companion,
    _expert_branch_slots,
    _expert_states,
)
from scripts.score_lewm_action_branch_a4 import load_a4_prediction_panel  # noqa: E402


SCHEMA_VERSION = "fcpg_a4_qcenter_geometry_gate_v1"
ANCHOR_START = 1
ANCHOR_COUNT = 3
MAIN_FEATURE_DIM = ANCHOR_COUNT * LATENT_DIM + 2 + len(TASKS)
TASK_TIME_FEATURE_DIM = 2 + len(TASKS) + HORIZON
DIRECT_ACTION_FEATURE_DIM = (
    3 * ANCHOR_COUNT * LATENT_DIM + 2 * ACTION_DIM
    + 2 + len(TASKS) + HORIZON + HORIZON * ACTION_DIM
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/qcenter_geometry_gate"
GATE_THRESHOLDS = {
    "target_max_macro_mean_epe_px": 2.0,
    "target_max_macro_p95_epe_px": 8.0,
    "target_min_macro_pck_at_8": 0.90,
    "factual_max_macro_mean_epe_px": 4.0,
    "factual_max_macro_p95_epe_px": 16.0,
    "minimum_winning_lewm_seeds": 2,
    "minimum_winning_tasks": 6,
}


class GeometryHead(nn.Module):
    """Small nonlinear q-center displacement regressor."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _one_hot(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= count):
        raise ValueError("one-hot indices are invalid")
    return np.eye(count, dtype=np.float32)[indices]


def _anchor_delta(states: np.ndarray, history: np.ndarray) -> np.ndarray:
    future = np.asarray(states, dtype=np.float32)
    past = np.asarray(history, dtype=np.float32)
    if future.ndim != 4 or future.shape[1:] != (HORIZON, 4, LATENT_DIM):
        raise ValueError("selected state must have shape [N,8,4,192]")
    if past.ndim != 4 or past.shape[0] != future.shape[0] or past.shape[2:] != (4, LATENT_DIM):
        raise ValueError("history must have aligned shape [N,T,4,192]")
    delta = future[:, :, ANCHOR_START:, :] - past[:, -1, None, ANCHOR_START:, :]
    return delta.reshape(future.shape[0], HORIZON, ANCHOR_COUNT * LATENT_DIM)


def build_main_features(
    anchor_delta: np.ndarray,
    q0: np.ndarray,
    task_index: np.ndarray,
) -> np.ndarray:
    """Build the frozen main contract without action, horizon, or time."""

    latent = np.asarray(anchor_delta, dtype=np.float32)
    center = np.asarray(q0, dtype=np.float32)
    if latent.ndim != 3 or latent.shape[1:] != (HORIZON, ANCHOR_COUNT * LATENT_DIM):
        raise ValueError("anchor delta must have shape [N,8,576]")
    if center.shape != (latent.shape[0], 2):
        raise ValueError("q0 must have shape [N,2]")
    task = _one_hot(task_index, len(TASKS))
    repeated_q0 = np.broadcast_to(center[:, None], (latent.shape[0], HORIZON, 2))
    repeated_task = np.broadcast_to(task[:, None], (latent.shape[0], HORIZON, len(TASKS)))
    result = np.concatenate([latent, repeated_q0, repeated_task], axis=-1)
    if result.shape != (latent.shape[0], HORIZON, MAIN_FEATURE_DIM):
        raise RuntimeError("main feature contract drifted")
    return result


def build_task_time_q0_features(q0: np.ndarray, task_index: np.ndarray) -> np.ndarray:
    center = np.asarray(q0, dtype=np.float32)
    if center.ndim != 2 or center.shape[1] != 2:
        raise ValueError("q0 must have shape [N,2]")
    task = _one_hot(task_index, len(TASKS))
    horizon = np.eye(HORIZON, dtype=np.float32)[None].repeat(center.shape[0], axis=0)
    result = np.concatenate(
        [
            np.broadcast_to(center[:, None], (center.shape[0], HORIZON, 2)),
            np.broadcast_to(task[:, None], (center.shape[0], HORIZON, len(TASKS))),
            horizon,
        ],
        axis=-1,
    )
    return result


def build_direct_action_features(
    q0: np.ndarray,
    task_index: np.ndarray,
    expert_action: np.ndarray,
    history_state: np.ndarray,
    history_action: np.ndarray,
) -> np.ndarray:
    action = np.asarray(expert_action, dtype=np.float32)
    if action.ndim != 3 or action.shape[1:] != (HORIZON, ACTION_DIM):
        raise ValueError("expert action must have shape [N,8,20]")
    history = np.asarray(history_state, dtype=np.float32)
    if history.ndim != 4 or history.shape[0] != action.shape[0] or history.shape[2:] != (4, LATENT_DIM):
        raise ValueError("history_state must have aligned shape [N,T,4,192]")
    past_action = np.asarray(history_action, dtype=np.float32)
    if past_action.shape != (action.shape[0], 2, ACTION_DIM):
        raise ValueError("history_action must have shape [N,2,20]")
    base = build_task_time_q0_features(q0, task_index)
    prefix = np.zeros((action.shape[0], HORIZON, HORIZON, ACTION_DIM), dtype=np.float32)
    for horizon in range(HORIZON):
        prefix[:, horizon, : horizon + 1] = action[:, : horizon + 1]
    history_context = np.concatenate(
        [
            history[:, :, ANCHOR_START:].reshape(action.shape[0], -1),
            past_action.reshape(action.shape[0], -1),
        ],
        axis=1,
    )
    repeated_history = np.broadcast_to(
        history_context[:, None], (action.shape[0], HORIZON, history_context.shape[1])
    )
    result = np.concatenate(
        [repeated_history, base, prefix.reshape(action.shape[0], HORIZON, -1)], axis=-1
    )
    if result.shape[-1] != DIRECT_ACTION_FEATURE_DIM:
        raise RuntimeError("DirectAction feature contract drifted")
    return result


def _fit_normalizer(
    feature: np.ndarray, sample_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(feature, dtype=np.float32).reshape(-1, feature.shape[-1])
    weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    weight = weight / weight.sum()
    mean = np.sum(flat * weight[:, None], axis=0)
    scale = np.sqrt(np.sum(np.square(flat - mean) * weight[:, None], axis=0))
    scale[scale < 1.0e-5] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _point_sampling_weights(task: np.ndarray, episode: np.ndarray) -> np.ndarray:
    tasks = np.asarray(task).astype(str)
    episodes = np.asarray(episode).astype(str)
    if tasks.ndim != 1 or episodes.shape != tasks.shape:
        raise ValueError("sampling task/episode vectors must align")
    result = np.zeros((tasks.size, HORIZON), dtype=np.float64)
    names = sorted(set(tasks.tolist()))
    for name in names:
        task_rows = tasks == name
        identities = sorted(set(episodes[task_rows].tolist()))
        for identity in identities:
            rows = task_rows & (episodes == identity)
            count = int(np.count_nonzero(rows))
            result[rows] = 1.0 / (len(names) * len(identities) * count * HORIZON)
    if not np.isclose(result.sum(), 1.0):
        raise RuntimeError("task/episode-balanced point weights do not sum to one")
    return result.reshape(-1)


def _parameter_count(input_dim: int, hidden_dim: int) -> int:
    return (
        input_dim * hidden_dim + hidden_dim
        + hidden_dim * (hidden_dim // 2) + hidden_dim // 2
        + (hidden_dim // 2) * 2 + 2
    )


def _matched_hidden_dim(input_dim: int, target_parameters: int) -> int:
    candidates = range(8, 2049)
    return min(candidates, key=lambda value: abs(_parameter_count(input_dim, value) - target_parameters))


def _train_head(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    sample_weight: np.ndarray,
) -> tuple[GeometryHead, np.ndarray, np.ndarray, list[float]]:
    _seed_everything(seed)
    mean, scale = _fit_normalizer(feature, sample_weight)
    x = torch.from_numpy(
        ((feature - mean) / scale).reshape(-1, feature.shape[-1]).astype(np.float32)
    )
    y = torch.from_numpy(np.asarray(target, dtype=np.float32).reshape(-1, 2))
    weight = torch.from_numpy(np.asarray(sample_weight, dtype=np.float64).reshape(-1))
    if weight.shape != (x.shape[0],) or torch.any(weight <= 0):
        raise ValueError("training sample weights must be positive and point-aligned")
    model = GeometryHead(feature.shape[-1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    losses: list[float] = []
    for step in range(steps):
        index = torch.multinomial(
            weight, min(batch_size, x.shape[0]), replacement=True, generator=generator
        )
        prediction = model(x[index].to(device))
        loss = torch.nn.functional.smooth_l1_loss(prediction, y[index].to(device), beta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 20) == 0:
            losses.append(float(loss.detach().cpu()))
    model.eval()
    return model, mean, scale, losses


@torch.inference_mode()
def _predict(
    model: GeometryHead,
    feature: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    flat = ((feature - mean) / scale).reshape(-1, feature.shape[-1]).astype(np.float32)
    chunks = []
    for start in range(0, flat.shape[0], batch_size):
        chunks.append(model(torch.from_numpy(flat[start : start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0).reshape(feature.shape[0], HORIZON, 2)


def _episode_task_metric(
    predicted_delta: np.ndarray,
    target_delta: np.ndarray,
    task: np.ndarray,
    episode: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(predicted_delta, dtype=np.float64)
    target = np.asarray(target_delta, dtype=np.float64)
    epe = np.linalg.norm(prediction - target, axis=-1).mean(axis=1)
    point_epe = np.linalg.norm(prediction - target, axis=-1)
    squared = np.square(prediction - target).mean(axis=(1, 2))
    tasks = np.asarray(task).astype(str)
    episodes = np.asarray(episode).astype(str)
    per_task: dict[str, Any] = {}
    task_epes = []
    task_mses = []
    episode_rows: list[dict[str, Any]] = []
    for name in sorted(set(tasks.tolist())):
        selected = tasks == name
        episode_epe = []
        episode_mse = []
        for identity in sorted(set(episodes[selected].tolist())):
            rows = selected & (episodes == identity)
            ee = float(epe[rows].mean())
            em = float(squared[rows].mean())
            points = point_epe[rows].reshape(-1)
            episode_epe.append(ee)
            episode_mse.append(em)
            episode_rows.append({
                "task": name,
                "episode": identity,
                "mean_epe": ee,
                "mse": em,
                "p95_epe": float(np.quantile(points, 0.95)),
                "pck_at_4": float(np.mean(points <= 4.0)),
                "pck_at_8": float(np.mean(points <= 8.0)),
            })
        task_epes.append(float(np.mean(episode_epe)))
        task_mses.append(float(np.mean(episode_mse)))
        per_task[name] = {
            "episode_count": len(episode_epe),
            "mean_epe": task_epes[-1],
            "mse": task_mses[-1],
            "p95_epe": float(np.mean([row["p95_epe"] for row in episode_rows if row["task"] == name])),
            "pck_at_4": float(np.mean([row["pck_at_4"] for row in episode_rows if row["task"] == name])),
            "pck_at_8": float(np.mean([row["pck_at_8"] for row in episode_rows if row["task"] == name])),
        }
    return {
        "aggregation": "task_then_physical_episode_macro",
        "macro_mean_epe": float(np.mean(task_epes)),
        "task_macro_mse": float(np.mean(task_mses)),
        "macro_p95_epe": float(np.mean([row["p95_epe"] for row in per_task.values()])),
        "macro_pck_at_4": float(np.mean([row["pck_at_4"] for row in per_task.values()])),
        "macro_pck_at_8": float(np.mean([row["pck_at_8"] for row in per_task.values()])),
        "per_task": per_task,
        "episode_rows": episode_rows,
    }


def _bootstrap_difference(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if samples <= 0:
        return {}
    by_method = []
    for report in (candidate, baseline):
        grouped: dict[str, dict[str, float]] = {}
        for row in report["episode_rows"]:
            grouped.setdefault(str(row["task"]), {})[str(row["episode"])] = float(row["mean_epe"])
        by_method.append(grouped)
    if set(by_method[0]) != set(by_method[1]):
        raise ValueError("bootstrap task support differs")
    rng = np.random.default_rng(seed)
    delta = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        values = []
        for task in sorted(by_method[0]):
            identities = sorted(by_method[0][task])
            if identities != sorted(by_method[1][task]):
                raise ValueError("paired bootstrap episode support differs")
            left = np.asarray([by_method[0][task][identity] for identity in identities])
            right = np.asarray([by_method[1][task][identity] for identity in identities])
            index = rng.integers(0, left.size, size=left.size)
            values.append(float(np.mean(left[index] - right[index])))
        delta[draw] = float(np.mean(values))
    return {
        "mean_epe_difference": float(delta.mean()),
        "ci95_low": float(np.quantile(delta, 0.025)),
        "ci95_high": float(np.quantile(delta, 0.975)),
    }


def _mean_metric_reports(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not values:
        raise ValueError("at least one metric report is required")
    keyed = []
    for report in values:
        keyed.append(
            {
                (str(row["task"]), str(row["episode"])): row
                for row in report["episode_rows"]
            }
        )
    keys = sorted(keyed[0])
    if any(sorted(rows) != keys for rows in keyed[1:]):
        raise ValueError("metric reports have different physical episode support")
    episode_rows = []
    for task, episode in keys:
        episode_rows.append(
            {
                "task": task,
                "episode": episode,
                "mean_epe": float(np.mean([rows[(task, episode)]["mean_epe"] for rows in keyed])),
                "mse": float(np.mean([rows[(task, episode)]["mse"] for rows in keyed])),
                "p95_epe": float(np.mean([rows[(task, episode)]["p95_epe"] for rows in keyed])),
                "pck_at_4": float(np.mean([rows[(task, episode)]["pck_at_4"] for rows in keyed])),
                "pck_at_8": float(np.mean([rows[(task, episode)]["pck_at_8"] for rows in keyed])),
            }
        )
    per_task = {}
    for task in sorted({row["task"] for row in episode_rows}):
        rows = [row for row in episode_rows if row["task"] == task]
        per_task[task] = {
            "episode_count": len(rows),
            "mean_epe": float(np.mean([row["mean_epe"] for row in rows])),
            "mse": float(np.mean([row["mse"] for row in rows])),
            "p95_epe": float(np.mean([row["p95_epe"] for row in rows])),
            "pck_at_4": float(np.mean([row["pck_at_4"] for row in rows])),
            "pck_at_8": float(np.mean([row["pck_at_8"] for row in rows])),
        }
    return {
        "aggregation": "error_mean_then_task_then_physical_episode_macro",
        "macro_mean_epe": float(np.mean([row["mean_epe"] for row in per_task.values()])),
        "task_macro_mse": float(np.mean([row["mse"] for row in per_task.values()])),
        "macro_p95_epe": float(np.mean([row["p95_epe"] for row in per_task.values()])),
        "macro_pck_at_4": float(np.mean([row["pck_at_4"] for row in per_task.values()])),
        "macro_pck_at_8": float(np.mean([row["pck_at_8"] for row in per_task.values()])),
        "per_task": per_task,
        "episode_rows": episode_rows,
        "component_report_count": len(values),
    }


def run_geometry_gate(
    cache_path: str | Path,
    companion_path: str | Path,
    prediction_paths: Sequence[str | Path],
    *,
    device_name: str,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    head_seeds: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
    prediction_mode: str = "formal",
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache, predictions, prediction_seeds = load_a4_prediction_panel(
        cache_path, prediction_paths, mode=prediction_mode
    )
    validate_a4_state_cache(cache)
    split = np.asarray(cache["split"]).astype(str)
    if np.any(split == "test"):
        raise PermissionError("q-center geometry Gate refuses reserved test rows")
    if set(split.tolist()) != {"train", "development"}:
        raise ValueError("geometry Gate requires exactly train/development splits")
    train = split == "train"
    development = split == "development"
    train_episode = set(np.asarray(cache["source_episode_id"])[train].astype(str).tolist())
    dev_episode = set(np.asarray(cache["source_episode_id"])[development].astype(str).tolist())
    if train_episode & dev_episode:
        raise ValueError("train and development physical episodes overlap")

    companion = _aligned_companion(companion_path, cache)
    q = np.asarray(companion["q_center_model_raw"], dtype=np.float32)
    q0 = q[:, 0]
    target_delta = q[:, 1:] - q[:, :1]
    task_index = np.asarray(cache["task_class_index"], dtype=np.int64)
    task = np.asarray(cache["task_name"]).astype(str)
    episode = np.asarray(cache["source_episode_id"]).astype(str)
    slots = _expert_branch_slots(cache)
    target_expert = _expert_states(np.asarray(cache["target_state"]), slots)
    target_anchor_delta = _anchor_delta(target_expert, np.asarray(cache["history_state"]))
    main_feature = build_main_features(target_anchor_delta, q0, task_index)
    time_feature = build_task_time_q0_features(q0, task_index)
    expert_action = np.asarray(cache["future_action"])[np.arange(split.size), slots]
    history_state = np.asarray(cache["history_state"])
    action_feature = build_direct_action_features(
        q0, task_index, expert_action, history_state,
        np.asarray(cache["history_action"]),
    )

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(tuple(head_seeds)) != 3 or len(set(head_seeds)) != 3:
        raise ValueError("geometry Gate requires exactly three distinct head seeds")
    point_weight = _point_sampling_weights(task[train], episode[train])
    target_parameter_count = _parameter_count(main_feature.shape[-1], hidden_dim)
    hidden_by_name = {
        "geometry": hidden_dim,
        "task_time_q0": _matched_hidden_dim(time_feature.shape[-1], target_parameter_count),
        "direct_action": _matched_hidden_dim(action_feature.shape[-1], target_parameter_count),
    }
    heads: dict[str, list[tuple[GeometryHead, np.ndarray, np.ndarray, list[float]]]] = {}
    for name, feature in (
        (("geometry", main_feature), ("task_time_q0", time_feature), ("direct_action", action_feature))
    ):
        heads[name] = [
            _train_head(
                feature[train], target_delta[train], hidden_dim=hidden_by_name[name],
                steps=steps, batch_size=batch_size, learning_rate=learning_rate,
                seed=int(seed), device=device, sample_weight=point_weight,
            )
            for seed in head_seeds
        ]

    reports: dict[str, Any] = {}
    predictions_out: dict[str, np.ndarray] = {}
    static_delta = np.zeros_like(target_delta[development])
    predictions_out["static_q0"] = static_delta.astype(np.float32)
    reports["static_q0"] = _episode_task_metric(
        static_delta, target_delta[development], task[development], episode[development]
    )
    for name, feature in (
        ("target", main_feature),
        ("task_time_q0", time_feature),
        ("direct_action", action_feature),
    ):
        head_name = "geometry" if name == "target" else name
        estimates = []
        seed_reports = []
        for model, mean, scale, _ in heads[head_name]:
            estimate = _predict(
                model, feature[development], mean, scale,
                device=device, batch_size=batch_size,
            )
            estimates.append(estimate)
            seed_reports.append(_episode_task_metric(
                estimate, target_delta[development], task[development], episode[development]
            ))
        predictions_out[name] = np.asarray(estimates, dtype=np.float32)
        reports[name] = {
            "by_head_seed": {
                str(seed): report for seed, report in zip(head_seeds, seed_reports, strict=True)
            },
            "primary_error_mean": _mean_metric_reports(seed_reports),
            "coordinate_ensemble_auxiliary": _episode_task_metric(
                np.mean(estimates, axis=0), target_delta[development],
                task[development], episode[development],
            ),
        }

    development_slots = slots[development]
    development_history = np.asarray(cache["history_state"])[development]
    q0_dev = q0[development]
    task_index_dev = task_index[development]
    def evaluate_ordinary(source: str) -> None:
        values = np.asarray(predictions[source])
        all_estimates = []
        by_head_and_lewm: dict[str, Any] = {}
        primary_by_lewm: dict[str, Any] = {}
        for head_index, (geometry_model, geometry_mean, geometry_scale, _) in enumerate(heads["geometry"]):
            head_estimates = []
            head_reports = {}
            for seed_index, lewm_seed in enumerate(prediction_seeds):
                selected = _expert_states(values[seed_index], development_slots)
                feature = build_main_features(
                    _anchor_delta(selected, development_history), q0_dev, task_index_dev
                )
                estimate = _predict(
                    geometry_model, feature, geometry_mean, geometry_scale,
                    device=device, batch_size=batch_size,
                )
                head_estimates.append(estimate)
                head_reports[str(lewm_seed)] = _episode_task_metric(
                    estimate, target_delta[development], task[development], episode[development]
                )
            all_estimates.append(head_estimates)
            by_head_and_lewm[str(head_seeds[head_index])] = head_reports
        for seed_index, lewm_seed in enumerate(prediction_seeds):
            primary_by_lewm[str(lewm_seed)] = _mean_metric_reports(
                [by_head_and_lewm[str(head_seed)][str(lewm_seed)] for head_seed in head_seeds]
            )
        reports[source] = {
            "by_head_and_lewm_seed": by_head_and_lewm,
            "primary_by_lewm_seed_error_mean": primary_by_lewm,
            "primary_all_seed_error_mean": _mean_metric_reports(list(primary_by_lewm.values())),
            "coordinate_ensemble_auxiliary": _episode_task_metric(
                np.mean(all_estimates, axis=(0, 1)), target_delta[development],
                task[development], episode[development],
            ),
        }
        predictions_out[source] = np.asarray(all_estimates, dtype=np.float32)

    evaluate_ordinary(FACTUAL_METHOD)
    evaluate_ordinary(NO_ACTION_METHOD)

    wrong_values = np.asarray(predictions[WRONG_ACTION_METHOD])
    wrong_estimates = []
    wrong_reports: dict[str, Any] = {}
    primary_wrong_by_lewm: dict[str, Any] = {}
    for head_index, (geometry_model, geometry_mean, geometry_scale, _) in enumerate(heads["geometry"]):
        head_estimates = []
        head_reports = {}
        for seed_index, seed in enumerate(prediction_seeds):
            per_shift = []
            per_shift_reports = {}
            for shift in range(wrong_values.shape[2]):
                selected = _expert_states(wrong_values[seed_index, :, shift], development_slots)
                feature = build_main_features(
                    _anchor_delta(selected, development_history), q0_dev, task_index_dev
                )
                estimate = _predict(
                    geometry_model, feature, geometry_mean, geometry_scale,
                    device=device, batch_size=batch_size,
                )
                per_shift.append(estimate)
                per_shift_reports[str(shift + 1)] = _episode_task_metric(
                    estimate, target_delta[development], task[development], episode[development]
                )
            head_estimates.append(per_shift)
            head_reports[str(seed)] = per_shift_reports
        wrong_estimates.append(head_estimates)
        wrong_reports[str(head_seeds[head_index])] = head_reports
    for lewm_seed in prediction_seeds:
        components = [
            wrong_reports[str(head_seed)][str(lewm_seed)][str(shift + 1)]
            for head_seed in head_seeds for shift in range(wrong_values.shape[2])
        ]
        primary_wrong_by_lewm[str(lewm_seed)] = _mean_metric_reports(components)
    wrong_estimates_array = np.asarray(wrong_estimates, dtype=np.float32)
    reports[WRONG_ACTION_METHOD] = {
        "by_head_lewm_seed_and_shift": wrong_reports,
        "primary_by_lewm_seed_error_mean_over_head_and_shift": primary_wrong_by_lewm,
        "primary_all_seed_error_mean": _mean_metric_reports(list(primary_wrong_by_lewm.values())),
        "coordinate_ensemble_auxiliary": _episode_task_metric(
            wrong_estimates_array.mean(axis=(0, 1, 2)), target_delta[development],
            task[development], episode[development],
        ),
    }
    predictions_out[WRONG_ACTION_METHOD] = wrong_estimates_array

    comparisons = {}
    factual_report = reports[FACTUAL_METHOD]["primary_all_seed_error_mean"]
    for baseline_name, baseline_report in (
        (NO_ACTION_METHOD, reports[NO_ACTION_METHOD]["primary_all_seed_error_mean"]),
        (WRONG_ACTION_METHOD, reports[WRONG_ACTION_METHOD]["primary_all_seed_error_mean"]),
        ("static_q0", reports["static_q0"]),
        ("task_time_q0", reports["task_time_q0"]["primary_error_mean"]),
        ("direct_action", reports["direct_action"]["primary_error_mean"]),
    ):
        comparisons[f"factual_vs_{baseline_name}"] = {
            "relative_epe_improvement": float(
                1.0 - factual_report["macro_mean_epe"] / baseline_report["macro_mean_epe"]
            ),
            "paired_episode_bootstrap": _bootstrap_difference(
                factual_report, baseline_report, samples=bootstrap_samples,
                seed=bootstrap_seed + len(comparisons),
            ),
        }

    def comparison_screen(
        name: str,
        baseline_by_lewm: Mapping[str, Mapping[str, Any]] | None,
        fixed_baseline: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        seed_wins = 0
        task_win_counts = {task_name: 0 for task_name in TASKS}
        for lewm_seed in prediction_seeds:
            factual_seed = reports[FACTUAL_METHOD]["primary_by_lewm_seed_error_mean"][str(lewm_seed)]
            baseline_seed = (
                baseline_by_lewm[str(lewm_seed)] if baseline_by_lewm is not None else fixed_baseline
            )
            assert baseline_seed is not None
            if factual_seed["macro_mean_epe"] < baseline_seed["macro_mean_epe"]:
                seed_wins += 1
            for task_name in TASKS:
                if (
                    factual_seed["per_task"][task_name]["mean_epe"]
                    < baseline_seed["per_task"][task_name]["mean_epe"]
                ):
                    task_win_counts[task_name] += 1
        tasks_winning_majority = sum(
            count >= GATE_THRESHOLDS["minimum_winning_lewm_seeds"]
            for count in task_win_counts.values()
        )
        return {
            "name": name,
            "winning_lewm_seeds": seed_wins,
            "task_win_count_by_lewm_seed": task_win_counts,
            "tasks_winning_at_least_2_of_3_seeds": tasks_winning_majority,
            "pass": (
                seed_wins >= GATE_THRESHOLDS["minimum_winning_lewm_seeds"]
                and tasks_winning_majority >= GATE_THRESHOLDS["minimum_winning_tasks"]
            ),
        }

    target_primary = reports["target"]["primary_error_mean"]
    target_improvement = 1.0 - (
        target_primary["macro_mean_epe"] / reports["static_q0"]["macro_mean_epe"]
    )
    target_winning_tasks = sum(
        target_primary["per_task"][name]["mean_epe"]
        < reports["static_q0"]["per_task"][name]["mean_epe"]
        for name in TASKS
    )
    screen_checks = {
        "target_geometry_decodable": {
            "relative_epe_improvement_vs_static_q0": float(target_improvement),
            "winning_tasks": target_winning_tasks,
            "pass": (
                target_primary["macro_mean_epe"]
                <= GATE_THRESHOLDS["target_max_macro_mean_epe_px"]
                and target_primary["macro_p95_epe"]
                <= GATE_THRESHOLDS["target_max_macro_p95_epe_px"]
                and target_primary["macro_pck_at_8"]
                >= GATE_THRESHOLDS["target_min_macro_pck_at_8"]
                and target_winning_tasks >= GATE_THRESHOLDS["minimum_winning_tasks"]
            ),
        },
        "factual_vs_no_action": comparison_screen(
            "factual_vs_no_action",
            reports[NO_ACTION_METHOD]["primary_by_lewm_seed_error_mean"], None,
        ),
        "factual_vs_wrong_action": comparison_screen(
            "factual_vs_wrong_action", primary_wrong_by_lewm, None,
        ),
        "factual_vs_task_time_q0": comparison_screen(
            "factual_vs_task_time_q0", None,
            reports["task_time_q0"]["primary_error_mean"],
        ),
        "factual_vs_direct_action": comparison_screen(
            "factual_vs_direct_action", None,
            reports["direct_action"]["primary_error_mean"],
        ),
        "factual_absolute_coordinate_accuracy": {
            "macro_mean_epe": factual_report["macro_mean_epe"],
            "macro_p95_epe": factual_report["macro_p95_epe"],
            "pass": (
                factual_report["macro_mean_epe"]
                <= GATE_THRESHOLDS["factual_max_macro_mean_epe_px"]
                and factual_report["macro_p95_epe"]
                <= GATE_THRESHOLDS["factual_max_macro_p95_epe_px"]
            ),
        },
    }

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "a4_qcenter_phase0_nonlinear_geometry_gate",
        "scope": "exploratory_development_only_fixed_view_privileged_q0_center",
        "claims_not_authorized": [
            "deployable_q0_observation",
            "cross_view_generalization",
            "FCPG_policy_or_control_improvement",
            "persistent_per_anchor_identity",
            "confirmatory_or_reserved_test_gate",
        ],
        "q0_contract": (
            "q_center_model_raw_h0_from_simulator_companion; privileged diagnostic input"
        ),
        "reserved_test_accessed": False,
        "fit_split": "train_target_latent_only",
        "evaluation_split": "development",
        "main_input_contract": [
            "target_or_predicted_anchor_delta_latent",
            "q0_center",
            "task_onehot",
        ],
        "main_forbidden_inputs": ["horizon", "time", "action"],
        "prediction_seeds": list(prediction_seeds),
        "head_seeds": list(head_seeds),
        "device": str(device),
        "steps_per_head": steps,
        "train_episode_count": len(train_episode),
        "development_episode_count": len(dev_episode),
        "episode_disjoint": True,
        "training_sampling": "task_then_physical_episode_then_proposal_then_horizon_equal",
        "direct_action_baseline_contract": (
            "three_frame_anchor_history_plus_two_history_actions_plus_expert_future_action"
        ),
        "thresholds_frozen_in_source_before_first_formal_run": GATE_THRESHOLDS,
        "exploratory_screen_checks": screen_checks,
        "exploratory_screen_pass": bool(all(value["pass"] for value in screen_checks.values())),
        "hidden_dim": hidden_by_name,
        "parameter_count": {
            name: int(sum(parameter.numel() for parameter in values[0][0].parameters()))
            for name, values in heads.items()
        },
        "training_loss_trace": {
            name: {str(seed): value[3] for seed, value in zip(head_seeds, values, strict=True)}
            for name, values in heads.items()
        },
        "reports": reports,
        "comparisons": comparisons,
    }
    artifact = {
        "proposal_id": np.asarray(cache["proposal_id"])[development],
        "task_name": task[development],
        "source_episode_id": episode[development],
        "split": split[development],
        "q0": q0[development],
        "target_delta": target_delta[development],
        **predictions_out,
    }
    return summary, artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--head-seeds", type=int, nargs=3, default=(40_901, 40_902, 40_903))
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=40_903)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.steps <= 0 or args.batch_size <= 0 or args.hidden_dim < 4:
        raise ValueError("steps/batch size must be positive and hidden dim at least four")
    summary, artifact = run_geometry_gate(
        args.cache, args.companion, args.predictions,
        device_name=args.device, steps=args.steps, batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, learning_rate=args.learning_rate,
        head_seeds=args.head_seeds, bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        prediction_mode="smoke" if args.smoke else "formal",
    )
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output / "development_predictions.npz", **artifact)
    print(json.dumps({
        "output": str(output),
        "factual_macro_mean_epe": summary["reports"][FACTUAL_METHOD]["primary_all_seed_error_mean"]["macro_mean_epe"],
        "exploratory_screen_pass": summary["exploratory_screen_pass"],
        "reserved_test_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
