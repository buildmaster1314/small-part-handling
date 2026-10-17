"""Phase-resolved diagnostics for A4c seven-task development predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PRIMARY_HORIZON_INDICES = np.asarray((5, 6, 7), dtype=np.int64)


def branch_centered_prediction_losses(
    target: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Return H6-H8 branch-centered MSE with shape ``[seed, row]``."""

    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    if target_array.ndim != 5:
        raise ValueError("target must have shape [row,branch,horizon,slot,latent]")
    expected = (prediction_array.shape[0], *target_array.shape)
    if prediction_array.ndim != 6 or prediction_array.shape != expected:
        raise ValueError(f"prediction must have shape {expected}")
    if target_array.shape[2] < 8:
        raise ValueError("phase diagnostics require at least eight horizons")
    if not np.isfinite(target_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("phase diagnostic tensors must be finite")

    target_primary = target_array[:, :, PRIMARY_HORIZON_INDICES].astype(
        np.float64, copy=False
    )
    prediction_primary = prediction_array[:, :, :, PRIMARY_HORIZON_INDICES].astype(
        np.float64, copy=False
    )
    target_centered = target_primary - np.mean(
        target_primary, axis=1, keepdims=True, dtype=np.float64
    )
    prediction_centered = prediction_primary - np.mean(
        prediction_primary, axis=2, keepdims=True, dtype=np.float64
    )
    return np.mean(
        np.square(prediction_centered - target_centered[None]),
        axis=(2, 3, 4, 5),
        dtype=np.float64,
    )


def _group_report(
    factual_loss: np.ndarray,
    no_action_loss: np.ndarray,
    episode: np.ndarray,
    selected: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, Any]:
    identities = np.unique(episode[selected])
    if identities.size == 0:
        raise ValueError("phase group contains no physical episode")
    factual_episode = np.asarray(
        [
            [np.mean(factual_loss[s, selected & (episode == identity)]) for identity in identities]
            for s in range(len(seeds))
        ],
        dtype=np.float64,
    )
    no_action_episode = np.asarray(
        [
            [np.mean(no_action_loss[s, selected & (episode == identity)]) for identity in identities]
            for s in range(len(seeds))
        ],
        dtype=np.float64,
    )
    factual_mse = float(np.mean(factual_episode))
    no_action_mse = float(np.mean(no_action_episode))
    relative = 0.0 if no_action_mse <= 1e-12 else 1.0 - factual_mse / no_action_mse
    per_seed: dict[str, Any] = {}
    for index, seed in enumerate(seeds):
        factual_seed = float(np.mean(factual_episode[index]))
        no_action_seed = float(np.mean(no_action_episode[index]))
        effect = (
            0.0
            if no_action_seed <= 1e-12
            else 1.0 - factual_seed / no_action_seed
        )
        per_seed[str(int(seed))] = {
            "factual_mse": factual_seed,
            "no_action_mse": no_action_seed,
            "relative_improvement": float(effect),
            "positive": bool(effect > 0.0),
        }
    return {
        "row_count": int(np.count_nonzero(selected)),
        "physical_episode_count": int(identities.size),
        "factual_mse": factual_mse,
        "no_action_mse": no_action_mse,
        "relative_improvement": float(relative),
        "positive_seed_count": int(
            sum(row["positive"] for row in per_seed.values())
        ),
        "per_seed": per_seed,
    }


def summarize_phase_development(
    *,
    target: np.ndarray,
    factual_prediction: np.ndarray,
    no_action_prediction: np.ndarray,
    task_name: Sequence[str],
    episode_id: Sequence[str],
    phase_name: Sequence[str],
    success_crossing: Sequence[bool],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Summarize development errors by task phase and success crossing."""

    factual_loss = branch_centered_prediction_losses(target, factual_prediction)
    no_action_loss = branch_centered_prediction_losses(target, no_action_prediction)
    row_count = factual_loss.shape[1]
    task = np.asarray(task_name).astype(str, copy=False)
    episode = np.asarray(episode_id).astype(str, copy=False)
    phase = np.asarray(phase_name).astype(str, copy=False)
    crossing = np.asarray(success_crossing, dtype=bool)
    if any(value.shape != (row_count,) for value in (task, episode, phase, crossing)):
        raise ValueError("phase metadata must align with development rows")
    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) != factual_loss.shape[0] or len(set(seed_values)) != len(
        seed_values
    ):
        raise ValueError("training seeds must be distinct and align with predictions")

    tasks: dict[str, Any] = {}
    for task_value in sorted(set(task.tolist())):
        task_selected = task == task_value
        phases: dict[str, Any] = {}
        for phase_value in sorted(set(phase[task_selected].tolist())):
            selected = task_selected & (phase == phase_value)
            phases[phase_value] = _group_report(
                factual_loss, no_action_loss, episode, selected, seed_values
            )
        task_report: dict[str, Any] = {
            "overall": _group_report(
                factual_loss, no_action_loss, episode, task_selected, seed_values
            ),
            "phases": phases,
        }
        selected_crossing = task_selected & crossing
        task_report["success_crossing"] = (
            _group_report(
                factual_loss,
                no_action_loss,
                episode,
                selected_crossing,
                seed_values,
            )
            if np.any(selected_crossing)
            else None
        )
        tasks[task_value] = task_report
    return {
        "schema_version": 1,
        "metric": "h6_h8_unwhitened_branch_centered_latent_mse",
        "aggregation": "seed_then_physical_episode_within_task_phase",
        "training_seeds": list(seed_values),
        "tasks": tasks,
        "claim_boundary": (
            "Development diagnostic only. Phase rows are selected by the frozen "
            "expert-replay schedule and do not define an independent Gate."
        ),
    }


__all__ = [
    "PRIMARY_HORIZON_INDICES",
    "branch_centered_prediction_losses",
    "summarize_phase_development",
]
