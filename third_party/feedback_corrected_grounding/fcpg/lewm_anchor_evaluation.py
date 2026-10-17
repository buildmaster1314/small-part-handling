"""Evaluation utilities for Anchor-Persistent LeWM pilots."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM
from fcpg.lewm_runtime import move_batch_to_device


_SEGMENT_FIELDS = (
    "anchor_dropout_active",
    "anchor_recovery_onset",
    "anchor_recovery_segment",
)
_ERROR_NAMES = (
    "factual",
    "copy",
    "shuffled",
    "nominal_measurement",
    "nominal_intervention",
)


def _metadata_values(value: object, name: str, batch_size: int) -> list[str]:
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, str):
        values = [value] * batch_size
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise KeyError(f"same-task action shuffle requires batch {name}")
    if len(values) != batch_size:
        raise ValueError(f"{name} metadata must match the batch size")
    return [str(item) for item in values]


def same_task_shuffled_actions(
    action: torch.Tensor,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Derange actions across episodes without crossing a task boundary.

    A final or highly imbalanced batch can make a complete derangement
    impossible for one task.  Such rows retain their factual action and are
    excluded through the returned validity mask instead of contaminating the
    metric with a cross-task donor.
    """

    if action.ndim < 2:
        raise ValueError("action must have a batch axis and at least one value axis")
    batch_size = int(action.shape[0])
    tasks = _metadata_values(batch.get("task_name"), "task_name", batch_size)
    episodes = _metadata_values(batch.get("episode_id"), "episode_id", batch_size)
    permutation = list(range(batch_size))
    valid = torch.zeros(batch_size, dtype=torch.bool, device=action.device)

    for task in sorted(set(tasks)):
        groups: dict[str, list[int]] = {}
        for index, (row_task, episode) in enumerate(zip(tasks, episodes, strict=True)):
            if row_task == task:
                groups.setdefault(episode, []).append(index)
        if len(groups) < 2:
            continue
        largest_group = max(len(indices) for indices in groups.values())
        task_count = sum(len(indices) for indices in groups.values())
        if largest_group * 2 > task_count:
            continue
        ordered = [index for episode in sorted(groups) for index in groups[episode]]
        donors = ordered[largest_group:] + ordered[:largest_group]
        for target, donor in zip(ordered, donors, strict=True):
            if tasks[target] != tasks[donor] or episodes[target] == episodes[donor]:
                raise RuntimeError("failed to construct a same-task episode derangement")
            permutation[target] = donor
            valid[target] = True

    if not bool(valid.any()):
        return None, valid
    indices = torch.tensor(permutation, dtype=torch.long, device=action.device)
    return action.index_select(0, indices), valid


def _squared_error_sum(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[float, int, torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("anchor prediction and target must share shape [B, T, K, D]")
    squared = (prediction - target).square()
    support = torch.ones(
        squared.shape[:-1], dtype=torch.bool, device=squared.device
    )
    if mask is not None:
        mask = mask.to(device=squared.device, dtype=torch.bool)
        while mask.ndim < support.ndim:
            mask = mask.unsqueeze(-1)
        try:
            support = torch.broadcast_to(mask, support.shape)
        except RuntimeError as error:
            raise ValueError(
                f"error mask {tuple(mask.shape)} cannot cover {tuple(support.shape)}"
            ) from error
    weighted = squared * support.unsqueeze(-1)
    per_anchor_sum = weighted.sum(dim=(0, 1, 3)).detach().double().cpu()
    per_anchor_count = (
        support.sum(dim=(0, 1)).detach().double().cpu() * squared.shape[-1]
    )
    return (
        float(weighted.sum().detach()),
        int(per_anchor_count.sum().item()),
        per_anchor_sum,
        per_anchor_count,
    )


def _new_error_totals(num_keypoints: int) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sum": 0.0,
            "count": 0,
            "per_anchor_sum": torch.zeros(num_keypoints, dtype=torch.float64),
            "per_anchor_count": torch.zeros(num_keypoints, dtype=torch.float64),
        }
        for name in _ERROR_NAMES
    }


def _accumulate_error(
    totals: dict[str, dict[str, Any]],
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    value_sum, value_count, anchor_sum, anchor_count = _squared_error_sum(
        prediction.float(), target.float(), mask
    )
    block = totals[name]
    block["sum"] += value_sum
    block["count"] += value_count
    block["per_anchor_sum"] += anchor_sum
    block["per_anchor_count"] += anchor_count


def _finalize_error_totals(totals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, block in totals.items():
        count = int(block["count"])
        metrics[f"{name}_element_count"] = count
        metrics[f"{name}_mse"] = block["sum"] / count if count else math.nan
        anchor_count = block["per_anchor_count"]
        anchor_sum = block["per_anchor_sum"]
        metrics[f"{name}_per_anchor_mse"] = [
            float(value_sum / value_count) if value_count > 0 else math.nan
            for value_sum, value_count in zip(anchor_sum, anchor_count, strict=True)
        ]
    _add_derived_metrics(metrics)
    return metrics


def _add_derived_metrics(metrics: dict[str, Any]) -> None:
    def difference(left: str, right: str) -> float:
        left_value = float(metrics.get(left, math.nan))
        right_value = float(metrics.get(right, math.nan))
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            return math.nan
        return left_value - right_value

    metrics["action_shuffle_delta"] = difference("shuffled_mse", "factual_mse")
    metrics["copy_delta"] = difference("copy_mse", "factual_mse")
    metrics["fcpg_measurement_innovation_gain"] = difference(
        "nominal_measurement_mse", "factual_mse"
    )
    metrics["fcpg_intervention_gain"] = difference(
        "nominal_intervention_mse", "factual_mse"
    )
    nominal = float(metrics.get("nominal_intervention_mse", math.nan))
    gain = float(metrics["fcpg_intervention_gain"])
    metrics["fcpg_intervention_relative_gain"] = (
        gain / nominal if math.isfinite(gain) and nominal > 0.0 else math.nan
    )


@dataclass
class _SpectrumAccumulator:
    count: int = 0
    vector_sum: torch.Tensor | None = None
    gram_sum: torch.Tensor | None = None

    def update(self, embedding: torch.Tensor) -> None:
        values = embedding.detach().reshape(-1, embedding.shape[-1]).double().cpu()
        if self.vector_sum is None:
            dimension = int(values.shape[-1])
            self.vector_sum = torch.zeros(dimension, dtype=torch.float64)
            self.gram_sum = torch.zeros(dimension, dimension, dtype=torch.float64)
        assert self.vector_sum is not None and self.gram_sum is not None
        if values.shape[-1] != self.vector_sum.shape[0]:
            raise ValueError("representation dimension changed during evaluation")
        self.vector_sum += values.sum(dim=0)
        self.gram_sum += values.T @ values
        self.count += int(values.shape[0])

    def metrics(self) -> dict[str, float | int | None]:
        if self.count < 2 or self.vector_sum is None or self.gram_sum is None:
            return {
                "samples": self.count,
                "effective_rank": None,
                "participation_ratio": None,
                "numerical_rank": None,
                "trace": None,
            }
        mean = self.vector_sum / self.count
        covariance = self.gram_sum / self.count - torch.outer(mean, mean)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        total = float(eigenvalues.sum())
        if not math.isfinite(total) or total <= 0.0:
            effective_rank = 0.0
            participation_ratio = 0.0
            numerical_rank = 0
        else:
            probability = eigenvalues / total
            positive = probability > 0.0
            entropy = -(probability[positive] * probability[positive].log()).sum()
            effective_rank = float(entropy.exp())
            participation_ratio = float(
                eigenvalues.sum().square()
                / eigenvalues.square().sum().clamp_min(1e-30)
            )
            numerical_rank = int((eigenvalues > eigenvalues.max() * 1e-6).sum())
        return {
            "samples": self.count,
            "effective_rank": effective_rank,
            "participation_ratio": participation_ratio,
            "numerical_rank": numerical_rank,
            "trace": total,
        }


@torch.no_grad()
def evaluate_anchor_persistent(
    model: AnchorPersistentLeWM,
    loader: DataLoader,
    device: torch.device,
    *,
    sigreg_weight: float = 0.02,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int = 0,
) -> dict[str, Any]:
    """Evaluate dynamics and current-measurement innovation without q_true.

    FCPG and nominal intervention cells each use their own history, prediction,
    and target under the same frozen checkpoint.  This is a same-checkpoint
    input intervention, not a matched-training comparison or a control metric.
    """

    model.eval()
    totals = _new_error_totals(model.num_keypoints)
    segment_totals: dict[str, dict[str, dict[str, Any]]] = {}
    representation = _SpectrumAccumulator()
    per_anchor_representation = [
        _SpectrumAccumulator() for _ in range(model.num_keypoints)
    ]
    winner_count = torch.zeros(model.num_keypoints, dtype=torch.float64)
    winner_total = torch.zeros(model.num_keypoints, dtype=torch.float64)
    samples = 0
    batches = 0
    shuffled_samples = 0

    for batch_index, host_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch_to_device(host_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled and device.type == "cuda",
        ):
            output = model(batch, sigreg_weight=sigreg_weight)
            factual = output.predicted_embedding
            target = output.target_embedding
            copy = output.embedding[:, : model.history_size]

            nominal_batch = dict(batch)
            nominal_batch["q_corrected"] = batch["q_nominal"]
            _, nominal_embedding = model.encode_frames(nominal_batch)
            nominal_target = nominal_embedding[:, 1 : model.history_size + 1]

            action = batch["action"][:, : model.history_size].float()
            nominal_prediction = model.predict(
                nominal_embedding[:, : model.history_size], action
            )
            shuffled_action, shuffle_valid = same_task_shuffled_actions(action, batch)
            shuffled = (
                model.predict(output.embedding[:, : model.history_size], shuffled_action)
                if shuffled_action is not None
                else None
            )

        values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "factual": (factual, target),
            "copy": (copy, target),
            "nominal_measurement": (factual, nominal_target),
            "nominal_intervention": (nominal_prediction, nominal_target),
        }
        if shuffled is not None:
            values["shuffled"] = (shuffled, target)
            shuffled_samples += int(shuffle_valid.sum())
        for name, (prediction, comparison) in values.items():
            mask = shuffle_valid if name == "shuffled" else None
            _accumulate_error(totals, name, prediction, comparison, mask)

        segment_masks: dict[str, torch.Tensor] = {}
        for field in _SEGMENT_FIELDS:
            value = batch.get(field)
            if value is None:
                continue
            if not torch.is_tensor(value) or value.shape != output.embedding.shape[:-1]:
                raise ValueError(
                    f"{field} must have shape {tuple(output.embedding.shape[:-1])}"
                )
            segment_masks[field] = value[:, 1 : model.history_size + 1].bool()
        for field, segment_mask in segment_masks.items():
            block = segment_totals.setdefault(
                field, _new_error_totals(model.num_keypoints)
            )
            for name, (prediction, comparison) in values.items():
                mask = segment_mask
                if name == "shuffled":
                    mask = mask & shuffle_valid[:, None, None]
                _accumulate_error(block, name, prediction, comparison, mask)

        representation.update(output.embedding)
        for anchor, accumulator in enumerate(per_anchor_representation):
            accumulator.update(output.embedding[:, :, anchor])
        factual_distance = (factual.float() - target.float()).square().mean(dim=-1)
        nominal_distance = (
            factual.float() - nominal_target.float()
        ).square().mean(dim=-1)
        winner_count += (factual_distance < nominal_distance).sum(dim=(0, 1)).cpu()
        winner_total += torch.full(
            (model.num_keypoints,),
            factual_distance.shape[0] * factual_distance.shape[1],
            dtype=torch.float64,
        )
        samples += int(output.embedding.shape[0])
        batches += 1

    if samples == 0:
        raise ValueError("evaluation loader produced no batches")

    metrics = _finalize_error_totals(totals)
    metrics.update(
        {
            "samples": samples,
            "batches": batches,
            "num_keypoints": model.num_keypoints,
            "strict_shuffle_samples": shuffled_samples,
            "shuffle_contract": "same_task_different_episode_only",
            "fcpg_measurement_closer_rate_per_anchor": (
                winner_count / winner_total.clamp_min(1.0)
            ).tolist(),
            "representation": {
                "all_anchors": representation.metrics(),
                "per_anchor": [
                    accumulator.metrics()
                    for accumulator in per_anchor_representation
                ],
            },
            "segments": {
                field: _finalize_error_totals(block)
                for field, block in segment_totals.items()
            },
        }
    )
    return metrics
