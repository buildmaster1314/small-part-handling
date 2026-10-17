"""Fail-closed scorer primitives for the Counterfactual v2 G1--G3 gates.

The collection store deliberately separates deployable inputs from labels that
exist only after a physical counterfactual branch has run.  This module keeps
that boundary intact: feature builders accept exact probe views, while labels
are used only by dataset assembly, normalization, and losses.

One dataset item is one physical snapshot containing its complete candidate
bank.  Candidates are never split into independent examples, and
train/validation assignment is inherited from the audited physical-episode
split in the v2 store.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fcpg.lewm_counterfactual import counterfactual_ranking_metrics
from fcpg.lewm_counterfactual_v2 import (
    PROTOCOL,
    TASK_NAMES,
    ActionOnlyProbeInputs,
    AuditArrays,
    DeploymentInputs,
    DirectProbeInputs,
    OfflineLabels,
    OracleProbeInputs,
    PredictedProbeInputs,
    build_probe_inputs,
    canonical_sha256,
    physical_episode_sha256,
    validate_deployable_probe_inputs,
    validate_full_config,
    validate_manifest_hash,
)
from fcpg.lewm_counterfactual_v2_storage import (
    CounterfactualV2Dataset,
    _fsync_directory,
    _publish_directory_no_replace,
    load_counterfactual_v2_store,
)


ProbeName = Literal[
    "action_only", "direct", "predicted", "oracle", "shuffled_predicted"
]
TargetName = Literal["actual_return", "actual_progress", "actual_success"]
ArchitectureName = Literal["linear", "tiny"]
ProbeInputs = (
    ActionOnlyProbeInputs
    | DirectProbeInputs
    | PredictedProbeInputs
    | OracleProbeInputs
)

DEPLOYABLE_PROBE_NAMES = frozenset({"action_only", "direct", "predicted"})
PROBE_NAMES = frozenset(
    {"action_only", "direct", "predicted", "oracle", "shuffled_predicted"}
)
TARGET_NAMES = frozenset({"actual_return", "actual_progress", "actual_success"})
DEFAULT_TINY_PARAMETER_BUDGET = 6_930
SCORER_EXPERIMENT_PROTOCOL = "fcpg_lewm_counterfactual_v2_scorer_e1_e3_v1"
RANDOM_CANDIDATE_NAMES = frozenset({"random_0", "random_1"})


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, copy=True, order="C", dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SourceEligibility:
    """Claim boundary that must be persisted beside every scorer result."""

    g0_eligible: bool
    external_development_override: bool
    ineligibility_reasons: tuple[str, ...]


def _scorer_source_contract(
    dataset: CounterfactualV2Dataset,
    *,
    allow_ineligible_external_development: bool,
) -> tuple[Mapping[str, Any], SourceEligibility]:
    """Accept formal G0 by default or an explicitly non-claimable dev store."""

    if not isinstance(dataset, CounterfactualV2Dataset):
        raise TypeError(
            "dataset must come from load_counterfactual_v2_store; unchecked mappings "
            "are forbidden"
        )
    publication = dataset.provenance
    if set(publication) != {"scientific_provenance", "g0_manifest"}:
        raise ValueError("published store lacks the exact scientific/G0 provenance pair")
    scientific = publication["scientific_provenance"]
    g0_manifest = publication["g0_manifest"]
    if not isinstance(scientific, Mapping) or not isinstance(g0_manifest, Mapping):
        raise TypeError("scientific provenance and G0 manifest must be mappings")
    validate_manifest_hash(g0_manifest)
    if g0_manifest.get("protocol") != PROTOCOL or g0_manifest.get("mode") != "full":
        raise ValueError("scorer training requires a full Counterfactual v2 G0 store")
    audit = g0_manifest.get("g0_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("G0 audit is missing")
    if audit.get("passed") is not True:
        raise ValueError("scorer training requires a scientifically passed G0 audit")
    contract = audit.get("index_and_provenance")
    if not isinstance(contract, Mapping):
        raise ValueError("G0 index/provenance audit is missing")
    if contract.get("passed") is not True:
        raise ValueError("G0 index/provenance contract did not pass")
    config = scientific.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("scientific provenance does not contain a resolved config")
    validate_full_config(config)
    if g0_manifest.get("config_sha256") != scientific.get("config_sha256"):
        raise ValueError("G0 and scientific config hashes disagree")
    if canonical_sha256(config) != scientific.get("config_sha256"):
        raise ValueError("scientific config hash is invalid")
    scope = config.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("scientific config scope is malformed")
    eligible = bool(audit.get("g0_eligible") is True and contract.get("g0_eligible") is True)
    reasons_value = contract.get("ineligibility_reasons", [])
    if not isinstance(reasons_value, Sequence) or isinstance(reasons_value, (str, bytes)):
        raise ValueError("G0 ineligibility reasons must be a sequence")
    reasons = tuple(str(reason) for reason in reasons_value)
    if eligible:
        if reasons:
            raise ValueError("eligible G0 evidence cannot contain ineligibility reasons")
        upstream = contract.get("upstream_non_overlap")
        if not isinstance(upstream, Mapping) or upstream.get("status") != "verified":
            raise ValueError("eligible G0 lacks verified upstream episode non-overlap")
        return scientific, SourceEligibility(True, False, ())
    if not allow_ineligible_external_development:
        raise ValueError(
            "scorer training requires formally eligible G0 evidence by default; "
            "an external-development run needs explicit opt-in"
        )
    if (
        scope.get("classification") != "external_development"
        or scope.get("formal_claim_allowed") is not False
    ):
        raise ValueError(
            "ineligible override is restricted to non-claimable external development"
        )
    if not reasons:
        raise ValueError("ineligible external-development evidence needs explicit reasons")
    return scientific, SourceEligibility(False, True, reasons)


@dataclass(frozen=True)
class TaskLabelNormalizer:
    """Per-task normalization fitted exclusively on training labels."""

    mean: np.ndarray
    scale: np.ndarray
    count: np.ndarray
    epsilon: float

    def transform(self, labels: np.ndarray, task_index: np.ndarray) -> np.ndarray:
        values = np.asarray(labels)
        task = np.asarray(task_index)
        if values.ndim != 2 or task.shape != (values.shape[0],):
            raise ValueError("labels must be [N,C] and task_index must be [N]")
        if not np.isfinite(values).all():
            raise ValueError("labels contain non-finite values")
        if np.any(task < 0) or np.any(task >= len(self.mean)):
            raise ValueError("task_index is outside the fitted task range")
        normalized = (
            values.astype(np.float64, copy=False) - self.mean[task, None]
        ) / self.scale[task, None]
        return _readonly(normalized, dtype=np.dtype(np.float32))


def fit_task_label_normalizer(
    train_labels: np.ndarray,
    train_task_index: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> TaskLabelNormalizer:
    """Fit per-task statistics; no validation argument is accepted by design."""

    labels = np.asarray(train_labels)
    task = np.asarray(train_task_index)
    if labels.ndim != 2 or task.shape != (labels.shape[0],):
        raise ValueError("train_labels must be [N,C] and train_task_index must be [N]")
    if not np.isfinite(labels).all():
        raise ValueError("training labels contain non-finite values")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    means = np.empty(len(TASK_NAMES), dtype=np.float64)
    scales = np.empty(len(TASK_NAMES), dtype=np.float64)
    counts = np.empty(len(TASK_NAMES), dtype=np.int64)
    for index, name in enumerate(TASK_NAMES):
        selected = labels[task == index].astype(np.float64, copy=False).reshape(-1)
        if selected.size < 2:
            raise ValueError(f"training split has too few labels for task {name}")
        mean = float(np.mean(selected))
        scale = float(np.std(selected, ddof=0))
        if not np.isfinite(scale) or scale <= epsilon:
            raise ValueError(f"training labels are degenerate for task {name}")
        means[index] = mean
        scales[index] = scale
        counts[index] = selected.size
    means.setflags(write=False)
    scales.setflags(write=False)
    counts.setflags(write=False)
    return TaskLabelNormalizer(means, scales, counts, float(epsilon))


@dataclass(frozen=True)
class PairwiseTargets:
    left: np.ndarray
    right: np.ndarray
    preference: np.ndarray
    informative: np.ndarray
    ties: np.ndarray


def pairwise_targets(labels: np.ndarray, *, tolerance: float = 1e-6) -> PairwiseTargets:
    """Create upper-triangle preferences while explicitly retaining ties."""

    values = np.asarray(labels)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("labels must have shape [N,C] with C >= 2")
    if not np.isfinite(values).all():
        raise ValueError("labels contain non-finite values")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    left, right = np.triu_indices(values.shape[1], k=1)
    differences = values[:, left] - values[:, right]
    informative = np.abs(differences) > tolerance
    preference = np.sign(differences).astype(np.int8)
    preference[~informative] = 0
    ties = ~informative
    return PairwiseTargets(
        left=_readonly(left.astype(np.int64, copy=False)),
        right=_readonly(right.astype(np.int64, copy=False)),
        preference=_readonly(preference),
        informative=_readonly(informative),
        ties=_readonly(ties),
    )


@dataclass(frozen=True)
class GroupedScorerSplit:
    """One episode-safe split whose rows retain complete candidate banks."""

    name: Literal["train", "validation"]
    probe_name: ProbeName
    inputs: ProbeInputs
    labels: np.ndarray
    raw_labels: np.ndarray
    task_index: np.ndarray
    phase_index: np.ndarray
    sample_indices: np.ndarray
    physical_episode_sha256: tuple[str, ...]
    candidate_names: tuple[tuple[str, ...], ...]
    pairwise: PairwiseTargets

    @property
    def sample_count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.labels.shape[1])


@dataclass(frozen=True)
class GroupedScorerData:
    probe_name: ProbeName
    target_name: TargetName
    horizon_index: int
    normalizer: TaskLabelNormalizer
    train: GroupedScorerSplit
    validation: GroupedScorerSplit
    source_result_sha256: str
    source_eligibility: SourceEligibility


def _subset_probe(inputs: ProbeInputs, indices: np.ndarray) -> ProbeInputs:
    values = {
        name: _readonly(array[indices]) for name, array in inputs.as_mapping().items()
    }
    return type(inputs)(**values)


def _episode_derangement(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> np.ndarray:
    """Map each row to the same task/phase in another physical episode."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("shuffle seed must be an integer")
    row_count = len(rows)
    donors = np.full(row_count, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for task_index in range(len(TASK_NAMES)):
        selected = [
            index for index, row in enumerate(rows) if int(row["task_index"]) == task_index
        ]
        episodes = sorted({str(rows[index]["physical_episode_sha256"]) for index in selected})
        if len(episodes) < 2:
            raise ValueError(
                f"shuffled-predicted control needs at least two episodes for task "
                f"{TASK_NAMES[task_index]}"
            )
        randomized = [episodes[index] for index in rng.permutation(len(episodes))]
        donor_episode = {
            randomized[index]: randomized[(index + 1) % len(randomized)]
            for index in range(len(randomized))
        }
        row_by_episode_phase: dict[tuple[str, int], int] = {}
        for index in selected:
            key = (
                str(rows[index]["physical_episode_sha256"]),
                int(rows[index]["phase_index"]),
            )
            if key in row_by_episode_phase:
                raise ValueError("duplicate physical-episode/phase row in scorer split")
            row_by_episode_phase[key] = index
        for index in selected:
            own = str(rows[index]["physical_episode_sha256"])
            key = (donor_episode[own], int(rows[index]["phase_index"]))
            if key not in row_by_episode_phase:
                raise ValueError("episode groups do not share the same frozen phase set")
            donor = row_by_episode_phase[key]
            if str(rows[donor]["physical_episode_sha256"]) == own:
                raise RuntimeError("shuffled-predicted control selected a self donor")
            donors[index] = donor
    if np.any(donors < 0):
        raise RuntimeError("shuffled-predicted donor assignment is incomplete")
    return donors


def shuffled_predicted_inputs(
    inputs: PredictedProbeInputs,
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> PredictedProbeInputs:
    """Shuffle only predicted rollouts across same-task episode groups."""

    validate_deployable_probe_inputs(inputs)
    if len(rows) != inputs.candidate_action_raw.shape[0]:
        raise ValueError("split rows and predicted inputs are not aligned")
    donors = _episode_derangement(rows, seed=seed)
    values = {name: _readonly(value) for name, value in inputs.as_mapping().items()}
    values["z_pred_future"] = _readonly(inputs.z_pred_future[donors])
    result = PredictedProbeInputs(**values)
    validate_deployable_probe_inputs(result)
    return result


def _split_rows(
    dataset: CounterfactualV2Dataset,
) -> tuple[np.ndarray, np.ndarray, tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    train = []
    validation = []
    expected_hashes: list[str] = []
    for index, row in enumerate(dataset.index_rows):
        split = row.get("split")
        if split not in {"train", "validation"}:
            raise ValueError("formal scorer data contains a non-train/validation row")
        expected = physical_episode_sha256(
            int(row["task_index"]), int(row["episode_id"]), int(row["env_seed"])
        )
        if row.get("physical_episode_sha256") != expected:
            raise ValueError("physical episode hash disagrees with row identity")
        expected_hashes.append(expected)
        (train if split == "train" else validation).append(index)
    if not train or not validation:
        raise ValueError("formal scorer data requires non-empty train and validation splits")
    by_episode: dict[str, set[str]] = {}
    for index, digest in enumerate(expected_hashes):
        by_episode.setdefault(digest, set()).add(str(dataset.index_rows[index]["split"]))
    if any(len(splits) != 1 for splits in by_episode.values()):
        raise ValueError("one physical episode straddles train and validation")
    train_episodes = {expected_hashes[index] for index in train}
    validation_episodes = {expected_hashes[index] for index in validation}
    if train_episodes & validation_episodes:
        raise ValueError("train and validation physical episodes overlap")
    train_indices = np.asarray(train, dtype=np.int64)
    validation_indices = np.asarray(validation, dtype=np.int64)
    return (
        train_indices,
        validation_indices,
        tuple(dataset.index_rows[index] for index in train),
        tuple(dataset.index_rows[index] for index in validation),
    )


def assemble_grouped_scorer_data(
    dataset: CounterfactualV2Dataset,
    *,
    probe: ProbeName,
    model: str | int,
    grounding: str | int,
    view: str | int,
    target: TargetName = "actual_return",
    horizon_index: int = -1,
    tie_tolerance: float = 1e-6,
    shuffle_seed: int = 8192,
    allow_ineligible_external_development: bool = False,
) -> GroupedScorerData:
    """Assemble a formal G1--G3 dataset without crossing the label boundary."""

    _scientific, source_eligibility = _scorer_source_contract(
        dataset,
        allow_ineligible_external_development=allow_ineligible_external_development,
    )
    if probe not in PROBE_NAMES:
        raise ValueError(f"unknown scorer probe {probe!r}")
    if target not in TARGET_NAMES:
        raise ValueError(f"unknown scorer target {target!r}")
    deployment = DeploymentInputs.from_mapping(dataset.deployment_inputs)
    labels = OfflineLabels.from_mapping(dataset.offline_labels)
    audit = AuditArrays.from_mapping(dataset.audit_arrays)
    label_array = np.asarray(getattr(labels, target))
    if label_array.ndim != 3:
        raise ValueError(f"{target} must have shape [N,C,H]")
    resolved_horizon = int(horizon_index)
    if resolved_horizon < 0:
        resolved_horizon += label_array.shape[2]
    if not 0 <= resolved_horizon < label_array.shape[2]:
        raise IndexError("horizon_index is outside the collected horizon range")
    raw = label_array[:, :, resolved_horizon].astype(np.float32, copy=False)

    base_probe = "predicted" if probe == "shuffled_predicted" else probe
    inputs = build_probe_inputs(
        base_probe,
        deployment,
        labels=labels if base_probe == "oracle" else None,
        model=model,
        grounding=grounding,
        view=view,
    )
    train_indices, validation_indices, train_rows, validation_rows = _split_rows(dataset)
    train_inputs = _subset_probe(inputs, train_indices)
    validation_inputs = _subset_probe(inputs, validation_indices)
    if probe == "shuffled_predicted":
        if not isinstance(train_inputs, PredictedProbeInputs) or not isinstance(
            validation_inputs, PredictedProbeInputs
        ):
            raise RuntimeError("shuffled-predicted control did not build predicted inputs")
        train_inputs = shuffled_predicted_inputs(
            train_inputs, train_rows, seed=shuffle_seed
        )
        validation_inputs = shuffled_predicted_inputs(
            validation_inputs, validation_rows, seed=shuffle_seed + 1
        )

    train_raw = _readonly(raw[train_indices], dtype=np.dtype(np.float32))
    validation_raw = _readonly(raw[validation_indices], dtype=np.dtype(np.float32))
    train_task = _readonly(deployment.task_index[train_indices].astype(np.int64))
    validation_task = _readonly(deployment.task_index[validation_indices].astype(np.int64))
    normalizer = fit_task_label_normalizer(train_raw, train_task)
    train_labels = normalizer.transform(train_raw, train_task)
    validation_labels = normalizer.transform(validation_raw, validation_task)

    def make_split(
        name: Literal["train", "validation"],
        split_inputs: ProbeInputs,
        split_labels: np.ndarray,
        split_raw: np.ndarray,
        split_task: np.ndarray,
        indices: np.ndarray,
        rows: tuple[Mapping[str, Any], ...],
    ) -> GroupedScorerSplit:
        phase = _readonly(audit.phase_index[indices].astype(np.int64))
        episodes = tuple(str(row["physical_episode_sha256"]) for row in rows)
        names = tuple(tuple(str(item) for item in row["candidate_names"]) for row in rows)
        candidate_count = split_inputs.candidate_action_raw.shape[1]
        if any(len(group) != candidate_count for group in names):
            raise ValueError("candidate-name groups do not align with candidate tensors")
        return GroupedScorerSplit(
            name=name,
            probe_name=probe,
            inputs=split_inputs,
            labels=split_labels,
            raw_labels=split_raw,
            task_index=split_task,
            phase_index=phase,
            sample_indices=_readonly(indices),
            physical_episode_sha256=episodes,
            candidate_names=names,
            pairwise=pairwise_targets(split_labels, tolerance=tie_tolerance),
        )

    train_split = make_split(
        "train", train_inputs, train_labels, train_raw, train_task, train_indices, train_rows
    )
    validation_split = make_split(
        "validation",
        validation_inputs,
        validation_labels,
        validation_raw,
        validation_task,
        validation_indices,
        validation_rows,
    )
    return GroupedScorerData(
        probe_name=probe,
        target_name=target,
        horizon_index=resolved_horizon,
        normalizer=normalizer,
        train=train_split,
        validation=validation_split,
        source_result_sha256=str(dataset.manifest["result_sha256"]),
        source_eligibility=source_eligibility,
    )


def candidate_feature_array(inputs: ProbeInputs) -> np.ndarray:
    """Flatten an exact probe view into [snapshot,candidate,feature]."""

    if isinstance(inputs, (ActionOnlyProbeInputs, DirectProbeInputs, PredictedProbeInputs)):
        validate_deployable_probe_inputs(inputs)
    elif not isinstance(inputs, OracleProbeInputs):
        raise TypeError("feature builder accepts only exact Counterfactual v2 probe views")
    arrays = inputs.as_mapping()
    candidate = arrays["candidate_action_raw"]
    n, c = candidate.shape[:2]

    def shared(name: str) -> np.ndarray:
        value = np.asarray(arrays[name]).reshape(n, -1)
        return np.broadcast_to(value[:, None, :], (n, c, value.shape[1]))

    blocks: list[np.ndarray] = []
    if isinstance(inputs, (DirectProbeInputs, PredictedProbeInputs, OracleProbeInputs)):
        blocks.extend((shared("z_history"), shared("history_action_raw")))
    blocks.append(candidate.reshape(n, c, -1))
    blocks.extend((shared("goal_visual"), shared("task_onehot")))
    if isinstance(inputs, PredictedProbeInputs):
        blocks.append(inputs.z_pred_future.reshape(n, c, -1))
    elif isinstance(inputs, OracleProbeInputs):
        blocks.append(inputs.oracle_future_latent.reshape(n, c, -1))
    result = np.concatenate(blocks, axis=-1).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("scorer features contain non-finite values")
    return _readonly(result)


class LinearCandidateScorer(nn.Module):
    """Fixed linear CostAdapter v0 applied independently to each candidate."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.output = nn.Linear(self.input_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("candidate features must have shape [B,C,input_dim]")
        return self.output(features).squeeze(-1)


class TinyCandidateScorer(nn.Module):
    """One-hidden-layer nonlinear CostAdapter with a fixed parameter budget."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("candidate features must have shape [B,C,input_dim]")
        return self.network(features).squeeze(-1)


def trainable_parameter_count(module: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )


def hidden_width_for_budget(input_dim: int, parameter_budget: int) -> int:
    """Choose the nearest valid tiny-scorer width for a common budget."""

    if input_dim <= 0 or parameter_budget <= 0:
        raise ValueError("input_dim and parameter_budget must be positive")
    # TinyCandidateScorer has h*(input_dim + 2) + 1 trainable parameters.
    return max(1, int(round((parameter_budget - 1) / float(input_dim + 2))))


def build_fixed_scorer(
    probe: ProbeName,
    *,
    input_dim: int,
    architecture: ArchitectureName,
    parameter_budget: int = DEFAULT_TINY_PARAMETER_BUDGET,
) -> nn.Module:
    if probe not in PROBE_NAMES:
        raise ValueError(f"unknown scorer probe {probe!r}")
    if architecture == "linear":
        return LinearCandidateScorer(input_dim)
    if architecture == "tiny":
        width = hidden_width_for_budget(input_dim, parameter_budget)
        return TinyCandidateScorer(input_dim, width)
    raise ValueError("architecture must be 'linear' or 'tiny'")


def parameter_match_report(
    scorers: Mapping[str, nn.Module],
    *,
    tolerance_fraction: float = 0.02,
    direct_name: str = "direct",
    predicted_name: str = "predicted",
) -> dict[str, Any]:
    """Audit a matched scorer family and enforce a non-weaker DirectRanker."""

    if not scorers:
        raise ValueError("scorers must not be empty")
    if not np.isfinite(tolerance_fraction) or tolerance_fraction < 0.0:
        raise ValueError("tolerance_fraction must be finite and non-negative")
    counts = {name: trainable_parameter_count(model) for name, model in scorers.items()}
    minimum = min(counts.values())
    maximum = max(counts.values())
    if minimum <= 0:
        raise ValueError("every scorer must have trainable parameters")
    spread = (maximum - minimum) / float(minimum)
    if spread > tolerance_fraction:
        raise ValueError(
            f"scorer parameter counts are not matched: spread={spread:.6f}, "
            f"tolerance={tolerance_fraction:.6f}"
        )
    if direct_name in counts and predicted_name in counts:
        if counts[direct_name] < counts[predicted_name]:
            raise ValueError("DirectRanker parameter count is lower than predicted scorer")
    return {
        "passed": True,
        "counts": counts,
        "minimum": minimum,
        "maximum": maximum,
        "spread_fraction": float(spread),
        "tolerance_fraction": float(tolerance_fraction),
    }


@dataclass(frozen=True)
class RankingLoss:
    total: Tensor
    pairwise: Tensor
    huber: Tensor
    informative_pair_count: int
    tie_pair_count: int


def pairwise_ranking_huber_loss(
    scores: Tensor,
    labels: Tensor,
    *,
    tie_tolerance: float = 1e-6,
    pairwise_weight: float = 1.0,
    huber_weight: float = 0.1,
    huber_beta: float = 1.0,
) -> RankingLoss:
    """Logistic pairwise ranking plus normalized-label Smooth-L1 regression."""

    if scores.ndim != 2 or labels.shape != scores.shape or scores.shape[1] < 2:
        raise ValueError("scores and labels must share shape [B,C] with C >= 2")
    if not torch.isfinite(scores).all() or not torch.isfinite(labels).all():
        raise ValueError("scores and labels must be finite")
    for name, value in (
        ("tie_tolerance", tie_tolerance),
        ("pairwise_weight", pairwise_weight),
        ("huber_weight", huber_weight),
        ("huber_beta", huber_beta),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if huber_beta <= 0.0:
        raise ValueError("huber_beta must be positive")
    left, right = torch.triu_indices(
        scores.shape[1], scores.shape[1], offset=1, device=scores.device
    )
    target_difference = labels[:, left] - labels[:, right]
    informative = target_difference.abs() > tie_tolerance
    score_difference = scores[:, left] - scores[:, right]
    signs = target_difference.sign()
    terms = F.softplus(-signs * score_difference)
    informative_per_snapshot = informative.sum(dim=1)
    informative_snapshots = informative_per_snapshot > 0
    if bool(informative_snapshots.any()):
        masked_sum = (terms * informative.to(dtype=terms.dtype)).sum(dim=1)
        per_snapshot = masked_sum[informative_snapshots] / informative_per_snapshot[
            informative_snapshots
        ].to(dtype=terms.dtype)
        pairwise = per_snapshot.mean()
    else:
        pairwise = scores.sum() * 0.0
    huber = F.smooth_l1_loss(scores, labels, beta=huber_beta)
    total = pairwise_weight * pairwise + huber_weight * huber
    return RankingLoss(
        total=total,
        pairwise=pairwise,
        huber=huber,
        informative_pair_count=int(informative.sum().item()),
        tie_pair_count=int((~informative).sum().item()),
    )


@dataclass(frozen=True)
class FeatureScaler:
    """Feature standardization fitted on training rows only.

    Task identity is a categorical input, so its one-hot coordinates are
    deliberately left byte-for-byte unchanged.
    """

    mean: np.ndarray
    scale: np.ndarray
    passthrough_indices: np.ndarray
    fitted_scalar_count: int
    epsilon: float

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features)
        if values.ndim != 3 or values.shape[-1] != self.mean.size:
            raise ValueError("features must have shape [N,C,fitted_feature_dim]")
        if not np.isfinite(values).all():
            raise ValueError("features contain non-finite values")
        result = (
            values.astype(np.float64, copy=False) - self.mean[None, None, :]
        ) / self.scale[None, None, :]
        result = result.astype(np.float32, copy=False)
        if self.passthrough_indices.size:
            result[..., self.passthrough_indices] = values[
                ..., self.passthrough_indices
            ]
        return _readonly(result)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "passthrough_indices": self.passthrough_indices.tolist(),
            "fitted_scalar_count": int(self.fitted_scalar_count),
            "epsilon": float(self.epsilon),
            "feature_dim": int(self.mean.size),
            "fit_scope": "training_rows_only",
        }


def task_onehot_feature_indices(inputs: ProbeInputs) -> np.ndarray:
    """Return the exact task-onehot coordinates in ``candidate_feature_array``."""

    arrays = inputs.as_mapping()
    candidate = np.asarray(arrays["candidate_action_raw"])
    n, c = candidate.shape[:2]
    del n, c
    offset = 0
    if isinstance(inputs, (DirectProbeInputs, PredictedProbeInputs, OracleProbeInputs)):
        offset += int(np.prod(np.asarray(arrays["z_history"]).shape[1:]))
        offset += int(np.prod(np.asarray(arrays["history_action_raw"]).shape[1:]))
    offset += int(np.prod(candidate.shape[2:]))
    offset += int(np.prod(np.asarray(arrays["goal_visual"]).shape[1:]))
    width = int(np.prod(np.asarray(arrays["task_onehot"]).shape[1:]))
    if width != len(TASK_NAMES):
        raise ValueError("task_onehot width does not match the frozen task family")
    result = np.arange(offset, offset + width, dtype=np.int64)
    result.setflags(write=False)
    return result


def fit_feature_scaler(
    train_features: np.ndarray,
    *,
    passthrough_indices: np.ndarray,
    epsilon: float = 1e-8,
) -> FeatureScaler:
    """Fit scalar statistics without accepting a validation argument."""

    values = np.asarray(train_features)
    passthrough = np.asarray(passthrough_indices)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("train_features must have shape [N,C,D]")
    if not np.isfinite(values).all():
        raise ValueError("training features contain non-finite values")
    if passthrough.ndim != 1 or not np.issubdtype(passthrough.dtype, np.integer):
        raise ValueError("passthrough_indices must be a one-dimensional integer array")
    if np.unique(passthrough).size != passthrough.size:
        raise ValueError("passthrough_indices contain duplicates")
    if np.any(passthrough < 0) or np.any(passthrough >= values.shape[-1]):
        raise ValueError("passthrough index is outside the feature dimension")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    flattened = values.astype(np.float64, copy=False).reshape(-1, values.shape[-1])
    mean = flattened.mean(axis=0)
    scale = flattened.std(axis=0, ddof=0)
    scale[scale <= epsilon] = 1.0
    mean[passthrough] = 0.0
    scale[passthrough] = 1.0
    return FeatureScaler(
        mean=_readonly(mean, dtype=np.dtype(np.float64)),
        scale=_readonly(scale, dtype=np.dtype(np.float64)),
        passthrough_indices=_readonly(passthrough.astype(np.int64, copy=False)),
        fitted_scalar_count=int(flattened.shape[0]),
        epsilon=float(epsilon),
    )


@dataclass(frozen=True)
class PooledScorerTrainingData:
    probe_name: ProbeName
    features: np.ndarray
    labels: np.ndarray
    task_index: np.ndarray
    scaler: FeatureScaler
    input_dim: int
    candidate_count: int
    source_result_sha256: str
    source_eligibility: SourceEligibility
    groundings: tuple[str, ...]
    view: str
    model: str


def assemble_pooled_clean_training_data(
    dataset: CounterfactualV2Dataset,
    *,
    probe: ProbeName,
    model: str,
    groundings: Sequence[str],
    view: str,
    target: TargetName = "actual_return",
    horizon_index: int = -1,
    allow_ineligible_external_development: bool = False,
) -> PooledScorerTrainingData:
    """Pool equal complete-snapshot copies from each frozen grounding cell."""

    if probe == "shuffled_predicted":
        raise ValueError("shuffled predicted futures are validation-only")
    cells = tuple(str(value) for value in groundings)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("groundings must be a non-empty unique sequence")
    assembled = tuple(
        assemble_grouped_scorer_data(
            dataset,
            probe=probe,
            model=model,
            grounding=grounding,
            view=view,
            target=target,
            horizon_index=horizon_index,
            allow_ineligible_external_development=(
                allow_ineligible_external_development
            ),
        )
        for grounding in cells
    )
    reference = assembled[0]
    reference_features = candidate_feature_array(reference.train.inputs)
    feature_groups = [reference_features]
    for cell in assembled[1:]:
        if cell.source_result_sha256 != reference.source_result_sha256:
            raise ValueError("pooled cells do not share one immutable source store")
        if cell.source_eligibility != reference.source_eligibility:
            raise ValueError("pooled cells disagree on source eligibility")
        if not np.array_equal(cell.train.raw_labels, reference.train.raw_labels):
            raise ValueError("camera-invariant training labels differ across groundings")
        if not np.array_equal(cell.train.sample_indices, reference.train.sample_indices):
            raise ValueError("pooled grounding cells are not snapshot aligned")
        if cell.train.candidate_names != reference.train.candidate_names:
            raise ValueError("candidate identities differ across pooled groundings")
        feature_groups.append(candidate_feature_array(cell.train.inputs))
    features = _readonly(np.concatenate(feature_groups, axis=0))
    labels = _readonly(
        np.concatenate([cell.train.labels for cell in assembled], axis=0),
        dtype=np.dtype(np.float32),
    )
    task_index = _readonly(
        np.concatenate([cell.train.task_index for cell in assembled], axis=0),
        dtype=np.dtype(np.int64),
    )
    passthrough = task_onehot_feature_indices(reference.train.inputs)
    scaler = fit_feature_scaler(features, passthrough_indices=passthrough)
    scaled = scaler.transform(features)
    if not np.array_equal(scaled[..., passthrough], features[..., passthrough]):
        raise RuntimeError("task onehot changed during feature scaling")
    return PooledScorerTrainingData(
        probe_name=probe,
        features=features,
        labels=labels,
        task_index=task_index,
        scaler=scaler,
        input_dim=int(features.shape[-1]),
        candidate_count=int(features.shape[1]),
        source_result_sha256=reference.source_result_sha256,
        source_eligibility=reference.source_eligibility,
        groundings=cells,
        view=str(view),
        model=str(model),
    )


def cyclic_candidate_shuffled_predicted_inputs(
    inputs: PredictedProbeInputs,
    *,
    offset: int = 1,
) -> PredictedProbeInputs:
    """Cyclically misassign candidate futures while freezing every other input."""

    validate_deployable_probe_inputs(inputs)
    candidate_count = int(inputs.candidate_action_raw.shape[1])
    if candidate_count < 2:
        raise ValueError("cyclic shuffle requires at least two candidates")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise TypeError("cyclic shuffle offset must be an integer")
    resolved = offset % candidate_count
    if resolved == 0:
        raise ValueError("cyclic shuffle offset must not be an identity rotation")
    values = {name: _readonly(value) for name, value in inputs.as_mapping().items()}
    values["z_pred_future"] = _readonly(
        np.roll(inputs.z_pred_future, shift=resolved, axis=1)
    )
    result = PredictedProbeInputs(**values)
    validate_deployable_probe_inputs(result)
    return result


def scorer_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass
class TrainedCandidateScorer:
    probe_name: ProbeName
    seed: int
    model: nn.Module
    optimizer_state: Mapping[str, Any]
    state_sha256: str
    parameter_count: int
    steps: int
    initial_loss: float
    final_loss: float
    final_pairwise_loss: float
    final_huber_loss: float
    final_gradient_norm: float


def _seed_training(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_full_batch_candidate_scorer(
    data: PooledScorerTrainingData,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    pairwise_weight: float = 1.0,
    huber_weight: float = 0.1,
    huber_beta: float = 1.0,
    tie_tolerance: float = 1e-6,
    parameter_budget: int = DEFAULT_TINY_PARAMETER_BUDGET,
    device: str = "cpu",
) -> TrainedCandidateScorer:
    """Deterministic fixed-step, full-batch AdamW training."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    for name, value in (
        ("learning_rate", learning_rate),
        ("weight_decay", weight_decay),
        ("gradient_clip_norm", gradient_clip_norm),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if learning_rate <= 0.0 or gradient_clip_norm <= 0.0:
        raise ValueError("learning rate and gradient clip norm must be positive")
    _seed_training(seed)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = build_fixed_scorer(
            data.probe_name,
            input_dim=data.input_dim,
            architecture="tiny",
            parameter_budget=parameter_budget,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        features = torch.from_numpy(
            data.scaler.transform(data.features).copy()
        ).to(device=device)
        labels = torch.from_numpy(data.labels.copy()).to(device=device)

        def evaluate_loss() -> RankingLoss:
            scores = model(features)
            return pairwise_ranking_huber_loss(
                scores,
                labels,
                tie_tolerance=tie_tolerance,
                pairwise_weight=pairwise_weight,
                huber_weight=huber_weight,
                huber_beta=huber_beta,
            )

        model.train()
        with torch.no_grad():
            initial_loss = float(evaluate_loss().total.detach().cpu())
        final_gradient_norm = float("nan")
        for _step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = evaluate_loss()
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("scorer gradient norm became non-finite")
            final_gradient_norm = float(gradient_norm.detach().cpu())
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = evaluate_loss()
        model = model.cpu()
        return TrainedCandidateScorer(
            probe_name=data.probe_name,
            seed=int(seed),
            model=model,
            optimizer_state=optimizer.state_dict(),
            state_sha256=scorer_state_sha256(model),
            parameter_count=trainable_parameter_count(model),
            steps=int(steps),
            initial_loss=initial_loss,
            final_loss=float(final.total.detach().cpu()),
            final_pairwise_loss=float(final.pairwise.detach().cpu()),
            final_huber_loss=float(final.huber.detach().cpu()),
            final_gradient_norm=final_gradient_norm,
        )
    finally:
        torch.use_deterministic_algorithms(previous_determinism)


def higher_is_better_ranking_metrics(
    scores: np.ndarray,
    returns: np.ndarray,
    *,
    expert_index: int,
) -> dict[str, Any]:
    """Adapt reward-oriented scores to the legacy lower-cost metric exactly once."""

    score = np.asarray(scores, dtype=np.float64)
    actual_return = np.asarray(returns, dtype=np.float64)
    # The legacy implementation is cost-oriented.  Both axes must be negated;
    # passing raw score/return would silently reverse regret orientation.
    result = counterfactual_ranking_metrics(
        -score,
        -actual_return,
        expert_index=expert_index,
        top_k=min(3, score.size),
    )
    return {
        "spearman": float(result["spearman"]),
        "normalized_top1_regret": float(result["normalized_top1_regret"]),
    }


def score_validation_split(
    trained: TrainedCandidateScorer,
    scaler: FeatureScaler,
    data: GroupedScorerData,
    *,
    view: str,
    grounding: str,
    variant: str | None = None,
) -> list[dict[str, Any]]:
    """Score the near-expert validation family, excluding random candidates."""

    inputs = data.validation.inputs
    if variant == "cyclic_candidate_future":
        if not isinstance(inputs, PredictedProbeInputs):
            raise TypeError("cyclic candidate shuffle requires predicted inputs")
        inputs = cyclic_candidate_shuffled_predicted_inputs(inputs)
    elif variant not in {None, "standard"}:
        raise ValueError(f"unknown validation variant {variant!r}")
    features = scaler.transform(candidate_feature_array(inputs))
    trained.model.eval()
    with torch.no_grad():
        scores = trained.model(torch.from_numpy(features.copy())).cpu().numpy()
    if scores.shape != data.validation.raw_labels.shape:
        raise RuntimeError("scorer output does not align with validation labels")
    scaler_sha256 = canonical_sha256(scaler.as_mapping())
    rows: list[dict[str, Any]] = []
    for row_index, names in enumerate(data.validation.candidate_names):
        if names.count("expert") != 1:
            raise ValueError("each candidate family must contain exactly one expert")
        kept = [
            index for index, name in enumerate(names) if name not in RANDOM_CANDIDATE_NAMES
        ]
        if len(kept) < 2:
            raise ValueError("near-expert candidate family is too small")
        kept_names = [names[index] for index in kept]
        expert_index = kept_names.index("expert")
        metrics = higher_is_better_ranking_metrics(
            scores[row_index, kept],
            data.validation.raw_labels[row_index, kept],
            expert_index=expert_index,
        )
        rows.append(
            {
                "probe": trained.probe_name,
                "variant": variant or "standard",
                "seed": int(trained.seed),
                "state_sha256": trained.state_sha256,
                "scaler_sha256": scaler_sha256,
                "task": TASK_NAMES[int(data.validation.task_index[row_index])],
                "physical_episode_sha256": data.validation.physical_episode_sha256[
                    row_index
                ],
                "phase_index": int(data.validation.phase_index[row_index]),
                "sample_index": int(data.validation.sample_indices[row_index]),
                "view": str(view),
                "grounding": str(grounding),
                "candidate_subset": "nonrandom_without_random_0_random_1",
                "candidate_names": kept_names,
                "candidate_scores": [float(scores[row_index, index]) for index in kept],
                "actual_return_h5": [
                    float(data.validation.raw_labels[row_index, index]) for index in kept
                ],
                **metrics,
            }
        )
    return rows


def _metric_episode_groups(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        task = str(row["task"])
        episode = str(row["physical_episode_sha256"])
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError(f"non-finite {metric} in scorer evaluation row")
        grouped.setdefault(task, {}).setdefault(episode, []).append(value)
    if set(grouped) != set(TASK_NAMES):
        raise ValueError("task-macro scorer metrics require the complete task family")
    return {
        task: {episode: float(np.mean(values)) for episode, values in episodes.items()}
        for task, episodes in grouped.items()
    }


def episode_task_macro_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Average phases/views/seeds within episode, then resample physical episodes."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    grouped = _metric_episode_groups(rows, metric)
    arrays = {
        task: np.asarray(list(grouped[task].values()), dtype=np.float64)
        for task in TASK_NAMES
    }
    estimate = float(np.mean([values.mean() for values in arrays.values()]))
    rng = np.random.default_rng(bootstrap_seed)
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        task_values = []
        for task in TASK_NAMES:
            values = arrays[task]
            indices = rng.integers(0, values.size, size=values.size)
            task_values.append(float(values[indices].mean()))
        samples[sample_index] = float(np.mean(task_values))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci95": [float(low), float(high)],
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "resampling_unit": "physical_episode_within_task",
        "averaged_before_resampling": ["phase", "view", "training_seed"],
        "episodes_per_task": {
            task: int(arrays[task].size) for task in TASK_NAMES
        },
    }


def paired_episode_task_macro_bootstrap(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    higher_is_better: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
    require_same_scorer: bool = False,
) -> dict[str, Any]:
    """Matched-row delta followed by physical-episode task-macro bootstrap."""

    key_fields = (
        "task",
        "physical_episode_sha256",
        "phase_index",
        "view",
        "seed",
    )

    def index(
        values: Sequence[Mapping[str, Any]], label: str
    ) -> dict[tuple[Any, ...], Mapping[str, Any]]:
        result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in values:
            key = tuple(row[field] for field in key_fields)
            if key in result:
                raise ValueError(f"duplicate {label} scorer row: {key}")
            result[key] = row
        return result

    left = index(left_rows, "left")
    right = index(right_rows, "right")
    if set(left) != set(right):
        raise ValueError("paired scorer rows do not share exact physical snapshots")
    sign = 1.0 if higher_is_better else -1.0
    delta_rows: list[dict[str, Any]] = []
    for key in sorted(left):
        left_row, right_row = left[key], right[key]
        if require_same_scorer:
            for identity in ("state_sha256", "scaler_sha256"):
                if identity not in left_row or identity not in right_row:
                    raise ValueError(
                        f"same-scorer comparison is missing {identity}"
                    )
                if left_row[identity] != right_row[identity]:
                    raise ValueError(
                        "G3 must compare two groundings with the same scorer "
                        f"state and scaler ({identity} differs)"
                    )
        delta_rows.append(
            {
                "task": left_row["task"],
                "physical_episode_sha256": left_row["physical_episode_sha256"],
                metric: sign * (float(right_row[metric]) - float(left_row[metric])),
            }
        )
    result = episode_task_macro_bootstrap(
        delta_rows,
        metric,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    result["orientation"] = "positive_means_right_better"
    result["raw_delta"] = "right_minus_left" if higher_is_better else "left_minus_right"
    result["same_scorer_required"] = bool(require_same_scorer)
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def validate_scorer_experiment_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen E1--E3 development contract without soft defaults."""

    if config.get("protocol") != SCORER_EXPERIMENT_PROTOCOL:
        raise ValueError("unexpected Counterfactual v2 scorer protocol")
    scope = config.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("scorer config scope is missing")
    external = bool(scope.get("allow_ineligible_external_development") is True)
    if external and (
        scope.get("classification") != "external_development"
        or scope.get("formal_claim_allowed") is not False
    ):
        raise ValueError("ineligible scorer opt-in is restricted to external development")
    source = config.get("source")
    source_sha = str(source.get("result_sha256", "")) if isinstance(source, Mapping) else ""
    if (
        not isinstance(source, Mapping)
        or not source.get("directory")
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("scorer source must bind a 64-character result SHA-256")
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("scorer data contract is missing")
    if (
        data.get("target") != "actual_return"
        or data.get("horizon_index") != 4
        or data.get("model") != "fcpg"
    ):
        raise ValueError("primary scorer target must be FCPG-model H5 actual_return")
    training = data.get("training")
    if not isinstance(training, Mapping) or (
        training.get("view") != "clean"
        or training.get("groundings") != ["nominal", "fcpg"]
        or training.get("pooling") != "equal_concat"
    ):
        raise ValueError("training must equally pool clean nominal/FCPG groundings")
    model = config.get("model")
    if not isinstance(model, Mapping) or (
        model.get("architecture") != "tiny"
        or model.get("parameter_budget") != DEFAULT_TINY_PARAMETER_BUDGET
        or model.get("expected_input_dims")
        != {"action_only": 152, "direct": 768, "predicted": 1728, "oracle": 1728}
        or model.get("expected_parameter_counts")
        != {"action_only": 6931, "direct": 6931, "predicted": 6921, "oracle": 6921}
    ):
        raise ValueError("scorer architecture/dimension/parameter contract changed")
    feature_scaling = config.get("feature_scaling")
    if not isinstance(feature_scaling, Mapping) or dict(feature_scaling) != {
        "fit_scope": "training_only",
        "method": "per_coordinate_standardization",
        "task_onehot": "unchanged",
    }:
        raise ValueError("feature scaling must be train-only with unchanged task onehot")
    optimization = config.get("optimization")
    if not isinstance(optimization, Mapping) or (
        optimization.get("optimizer") != "AdamW"
        or optimization.get("learning_rate") != 0.001
        or optimization.get("weight_decay") != 0.0001
        or optimization.get("gradient_clip_norm") != 1.0
        or optimization.get("batching") != "full_batch"
        or optimization.get("steps") != 2000
        or optimization.get("scheduler") is not None
        or optimization.get("early_stopping") is not False
        or optimization.get("seeds") != [8201, 8202, 8203]
        or optimization.get("deterministic") is not True
        or optimization.get("device") != "cpu"
    ):
        raise ValueError("frozen scorer optimization contract changed")
    loss = config.get("loss")
    if not isinstance(loss, Mapping) or (
        loss.get("name") != "ranknet_plus_huber"
        or loss.get("pairwise_weight") != 1.0
        or loss.get("huber_weight") != 0.1
        or loss.get("huber_beta") != 1.0
        or loss.get("tie_tolerance") != 1e-6
    ):
        raise ValueError("scorer loss must be RankNet plus 0.1 Huber")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping) or (
        evaluation.get("bootstrap_samples") != 10_000
        or evaluation.get("bootstrap_seed") != 8299
        or evaluation.get("exclude_candidates") != ["random_0", "random_1"]
        or evaluation.get("drift_views") != ["rotation5", "translation4"]
        or evaluation.get("resampling_unit") != "physical_episode_within_task"
        or evaluation.get("metrics")
        != ["task_macro_spearman", "task_macro_normalized_top1_regret"]
    ):
        raise ValueError("frozen scorer evaluation contract changed")
    output = config.get("output")
    if not isinstance(output, Mapping) or output.get("write_mode") != "exclusive_atomic_directory":
        raise ValueError("scorer output must use exclusive atomic directory publication")
    code = config.get("code")
    if not isinstance(code, Mapping) or code.get("require_clean_git") is not True:
        raise ValueError("scorer experiment must bind a clean code commit")


def _resolve_path(repo_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def validate_source_result_sha256(
    dataset: CounterfactualV2Dataset, expected: str
) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("expected source result SHA-256 must contain 64 characters")
    actual = str(dataset.manifest.get("result_sha256", ""))
    if actual != expected:
        raise ValueError(
            f"source result SHA mismatch: expected {expected}, got {actual}"
        )
    return actual


def publish_scorer_directory_exclusive(staging: Path, destination: Path) -> None:
    """Atomically publish without replacing an existing result."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"exclusive scorer output already exists: {destination}")
    _fsync_directory(staging)
    _publish_directory_no_replace(staging, destination)
    _fsync_directory(destination.parent)


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical_json_bytes(row).decode("ascii") + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_torch(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        torch.save(dict(value), stream)
        stream.flush()
        os.fsync(stream.fileno())


def _condition_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    probe: str,
    views: Sequence[str],
    grounding: str,
    variant: str = "standard",
) -> list[Mapping[str, Any]]:
    view_set = set(views)
    return [
        row
        for row in rows
        if row["probe"] == probe
        and row["view"] in view_set
        and row["grounding"] == grounding
        and row["variant"] == variant
    ]


def summarize_e1_e3_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Build the frozen E1/E2/E3 comparisons and clean guardrails."""

    metrics = ("spearman", "normalized_top1_regret")
    comparisons: dict[str, Any] = {}

    def paired(
        name: str,
        left: Sequence[Mapping[str, Any]],
        right: Sequence[Mapping[str, Any]],
        *,
        same_scorer: bool = False,
    ) -> None:
        if not left or not right:
            raise ValueError(f"comparison {name} has an empty cell")
        comparisons[name] = {
            metric: paired_episode_task_macro_bootstrap(
                left,
                right,
                metric,
                higher_is_better=(metric == "spearman"),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                require_same_scorer=same_scorer,
            )
            for metric in metrics
        }

    clean_fcpg = {probe: _condition_rows(rows, probe=probe, views=["clean"], grounding="fcpg") for probe in ("action_only", "direct", "predicted", "oracle")}
    drift_fcpg = {probe: _condition_rows(rows, probe=probe, views=["rotation5", "translation4"], grounding="fcpg") for probe in ("action_only", "direct", "predicted")}
    predicted_shuffle = _condition_rows(
        rows,
        probe="predicted",
        views=["rotation5", "translation4"],
        grounding="fcpg",
        variant="cyclic_candidate_future",
    )
    predicted_drift_nominal = _condition_rows(
        rows, probe="predicted", views=["rotation5", "translation4"], grounding="nominal"
    )
    predicted_clean_nominal = _condition_rows(
        rows, probe="predicted", views=["clean"], grounding="nominal"
    )
    paired("E1_oracle_vs_direct_clean_fcpg", clean_fcpg["direct"], clean_fcpg["oracle"])
    paired("E2_predicted_vs_direct_drift_fcpg", drift_fcpg["direct"], drift_fcpg["predicted"])
    paired("E2_predicted_vs_action_drift_fcpg", drift_fcpg["action_only"], drift_fcpg["predicted"])
    paired("E2_predicted_vs_cyclic_shuffle_drift_fcpg", predicted_shuffle, drift_fcpg["predicted"], same_scorer=True)
    paired("E3_fcpg_vs_nominal_grounding_drift", predicted_drift_nominal, drift_fcpg["predicted"], same_scorer=True)
    paired("clean_predicted_vs_direct_fcpg", clean_fcpg["direct"], clean_fcpg["predicted"])
    paired("clean_fcpg_vs_nominal_grounding", predicted_clean_nominal, clean_fcpg["predicted"], same_scorer=True)
    cells: dict[str, Any] = {}
    for name, cell in {
        "clean_fcpg_action_only": clean_fcpg["action_only"],
        "clean_fcpg_direct": clean_fcpg["direct"],
        "clean_fcpg_predicted": clean_fcpg["predicted"],
        "clean_fcpg_oracle": clean_fcpg["oracle"],
        "drift_fcpg_action_only": drift_fcpg["action_only"],
        "drift_fcpg_direct": drift_fcpg["direct"],
        "drift_fcpg_predicted": drift_fcpg["predicted"],
        "drift_fcpg_predicted_cyclic_shuffle": predicted_shuffle,
        "drift_nominal_predicted": predicted_drift_nominal,
    }.items():
        cells[name] = {
            metric: episode_task_macro_bootstrap(
                cell,
                metric,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for metric in metrics
        }
    return {
        "primary_target": "H5 actual_return",
        "candidate_subset": "nonrandom_without_random_0_random_1",
        "metrics": ["task_macro_spearman", "task_macro_normalized_top1_regret"],
        "cells": cells,
        "comparisons": comparisons,
    }


def _checkpoint_payload(
    trained: TrainedCandidateScorer,
    *,
    config_sha256: str,
    source_result_sha256: str,
    code: Mapping[str, Any],
    scaler_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol": SCORER_EXPERIMENT_PROTOCOL,
        "probe": trained.probe_name,
        "seed": trained.seed,
        "steps": trained.steps,
        "parameter_count": trained.parameter_count,
        "model_state": trained.model.state_dict(),
        "model_state_sha256": trained.state_sha256,
        "optimizer_state": trained.optimizer_state,
        "config_sha256": config_sha256,
        "source_result_sha256": source_result_sha256,
        "code": dict(code),
        "scaler_sha256": scaler_sha256,
    }


def run_scorer_experiment(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    confirm_external_development: bool = False,
) -> dict[str, Any]:
    """Train and evaluate frozen E1--E3 scorers, then publish one sealed result."""

    validate_scorer_experiment_config(config)
    scope = config["scope"]
    external = bool(scope["allow_ineligible_external_development"])
    if external and not confirm_external_development:
        raise PermissionError(
            "ineligible external-development source requires explicit CLI confirmation"
        )
    source_path = _resolve_path(repo_root, config["source"]["directory"])
    output_path = _resolve_path(repo_root, config["output"]["directory"])
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"exclusive scorer output already exists: {output_path}")
    dataset = load_counterfactual_v2_store(source_path)
    expected_source = str(config["source"]["result_sha256"])
    actual_source = validate_source_result_sha256(dataset, expected_source)
    code = _git_provenance(repo_root)
    if config.get("code", {}).get("require_clean_git") is True and code["dirty"]:
        raise RuntimeError("scorer experiment requires a clean git worktree")
    config_sha = canonical_sha256(config)
    training_config = config["data"]["training"]
    optimization = config["optimization"]
    loss_config = config["loss"]
    model_config = config["model"]
    expected_dims = model_config["expected_input_dims"]
    expected_counts = model_config["expected_parameter_counts"]
    probes: tuple[ProbeName, ...] = ("action_only", "direct", "predicted", "oracle")
    pooled: dict[str, PooledScorerTrainingData] = {}
    for probe in probes:
        value = assemble_pooled_clean_training_data(
            dataset,
            probe=probe,
            model=str(config["data"]["model"]),
            groundings=training_config["groundings"],
            view=str(training_config["view"]),
            target=str(config["data"]["target"]),  # type: ignore[arg-type]
            horizon_index=int(config["data"]["horizon_index"]),
            allow_ineligible_external_development=external,
        )
        if value.input_dim != int(expected_dims[probe]):
            raise ValueError(f"{probe} feature dimension changed")
        pooled[probe] = value
    eligibility = pooled["direct"].source_eligibility
    if any(value.source_eligibility != eligibility for value in pooled.values()):
        raise RuntimeError("scorer probes disagree on source eligibility")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent)
    )
    evaluation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    try:
        (staging / "checkpoints").mkdir()
        (staging / "scalers").mkdir()
        _write_json(staging / "resolved_config.json", config)
        _write_json(
            staging / "source.json",
            {
                "directory": str(source_path),
                "result_sha256": actual_source,
                "eligibility": {
                    "g0_eligible": eligibility.g0_eligible,
                    "external_development_override": eligibility.external_development_override,
                    "ineligibility_reasons": list(eligibility.ineligibility_reasons),
                },
            },
        )
        scaler_hashes: dict[str, str] = {}
        for probe in probes:
            path = staging / "scalers" / f"{probe}.json"
            _write_json(path, pooled[probe].scaler.as_mapping())
            scaler_hashes[probe] = _sha256_file(path)

        evaluation_cache: dict[tuple[str, str, str], GroupedScorerData] = {}

        def eval_data(probe: str, grounding: str, view: str) -> GroupedScorerData:
            key = (probe, grounding, view)
            if key not in evaluation_cache:
                evaluation_cache[key] = assemble_grouped_scorer_data(
                    dataset,
                    probe=probe,  # type: ignore[arg-type]
                    model=str(config["data"]["model"]),
                    grounding=grounding,
                    view=view,
                    target=str(config["data"]["target"]),  # type: ignore[arg-type]
                    horizon_index=int(config["data"]["horizon_index"]),
                    allow_ineligible_external_development=external,
                )
            return evaluation_cache[key]

        states_by_seed: dict[str, dict[str, str]] = {}
        for seed in optimization["seeds"]:
            trained_by_probe: dict[str, TrainedCandidateScorer] = {}
            for probe in probes:
                trained = train_full_batch_candidate_scorer(
                    pooled[probe],
                    seed=int(seed),
                    steps=int(optimization["steps"]),
                    learning_rate=float(optimization["learning_rate"]),
                    weight_decay=float(optimization["weight_decay"]),
                    gradient_clip_norm=float(optimization["gradient_clip_norm"]),
                    pairwise_weight=float(loss_config["pairwise_weight"]),
                    huber_weight=float(loss_config["huber_weight"]),
                    huber_beta=float(loss_config["huber_beta"]),
                    tie_tolerance=float(loss_config["tie_tolerance"]),
                    parameter_budget=int(model_config["parameter_budget"]),
                    device=str(optimization["device"]),
                )
                if trained.parameter_count != int(expected_counts[probe]):
                    raise RuntimeError(f"{probe} parameter count changed")
                trained_by_probe[probe] = trained
                training_rows.append(
                    {
                        "probe": probe,
                        "seed": int(seed),
                        "steps": trained.steps,
                        "parameter_count": trained.parameter_count,
                        "state_sha256": trained.state_sha256,
                        "initial_loss": trained.initial_loss,
                        "final_loss": trained.final_loss,
                        "final_pairwise_loss": trained.final_pairwise_loss,
                        "final_huber_loss": trained.final_huber_loss,
                        "final_gradient_norm": trained.final_gradient_norm,
                    }
                )
                checkpoint_dir = staging / "checkpoints" / probe
                checkpoint_dir.mkdir(exist_ok=True)
                _write_torch(
                    checkpoint_dir / f"seed_{seed}.pt",
                    _checkpoint_payload(
                        trained,
                        config_sha256=config_sha,
                        source_result_sha256=actual_source,
                        code=code,
                        scaler_sha256=scaler_hashes[probe],
                    ),
                )
            parameter_match_report(
                {name: value.model for name, value in trained_by_probe.items()},
                tolerance_fraction=0.01,
            )
            states_by_seed[str(int(seed))] = {
                name: value.state_sha256 for name, value in trained_by_probe.items()
            }
            for probe in ("action_only", "direct", "predicted"):
                for view in ("clean", "rotation5", "translation4"):
                    evaluation_rows.extend(
                        score_validation_split(
                            trained_by_probe[probe],
                            pooled[probe].scaler,
                            eval_data(probe, "fcpg", view),
                            view=view,
                            grounding="fcpg",
                        )
                    )
            evaluation_rows.extend(
                score_validation_split(
                    trained_by_probe["oracle"],
                    pooled["oracle"].scaler,
                    eval_data("oracle", "fcpg", "clean"),
                    view="clean",
                    grounding="fcpg",
                )
            )
            for view in ("clean", "rotation5", "translation4"):
                evaluation_rows.extend(
                    score_validation_split(
                        trained_by_probe["predicted"],
                        pooled["predicted"].scaler,
                        eval_data("predicted", "nominal", view),
                        view=view,
                        grounding="nominal",
                    )
                )
            for view in ("rotation5", "translation4"):
                evaluation_rows.extend(
                    score_validation_split(
                        trained_by_probe["predicted"],
                        pooled["predicted"].scaler,
                        eval_data("predicted", "fcpg", view),
                        view=view,
                        grounding="fcpg",
                        variant="cyclic_candidate_future",
                    )
                )

        metrics = summarize_e1_e3_metrics(
            evaluation_rows,
            bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
            bootstrap_seed=int(config["evaluation"]["bootstrap_seed"]),
        )
        _write_jsonl(staging / "evaluation_rows.jsonl", evaluation_rows)
        _write_json(staging / "training_state.json", training_rows)
        _write_json(staging / "metrics.json", metrics)
        files: dict[str, Any] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                relative = path.relative_to(staging).as_posix()
                files[relative] = {
                    "sha256": _sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                }
        manifest: dict[str, Any] = {
            "protocol": SCORER_EXPERIMENT_PROTOCOL,
            "complete": True,
            "config_sha256": config_sha,
            "source_result_sha256": actual_source,
            "source_eligibility": {
                "g0_eligible": eligibility.g0_eligible,
                "external_development_override": eligibility.external_development_override,
                "ineligibility_reasons": list(eligibility.ineligibility_reasons),
            },
            "code": code,
            "model_state_sha256": states_by_seed,
            "files": files,
        }
        manifest["result_sha256"] = hashlib.sha256(
            _canonical_json_bytes(manifest)
        ).hexdigest()
        _write_json(staging / "manifest.json", manifest)
        publish_scorer_directory_exclusive(staging, output_path)
        published = audit_scorer_experiment_result(output_path)
        if published != manifest:
            raise RuntimeError("published scorer manifest changed after atomic rename")
        return published
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def audit_scorer_experiment_result(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Re-hash every sealed result artifact and validate its result identity."""

    path = Path(root).expanduser().absolute()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(manifest.pop("result_sha256"))
    actual = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    if actual != expected:
        raise ValueError("scorer result manifest hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("scorer result manifest lacks file inventory")
    actual_files = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if actual_files != set(files):
        raise ValueError("scorer result file inventory mismatch")
    for name, record in files.items():
        if _sha256_file(path / name) != record["sha256"]:
            raise ValueError(f"scorer result file hash mismatch: {name}")
    manifest["result_sha256"] = expected
    return manifest


__all__ = [
    "DEFAULT_TINY_PARAMETER_BUDGET",
    "FeatureScaler",
    "GroupedScorerData",
    "GroupedScorerSplit",
    "LinearCandidateScorer",
    "PairwiseTargets",
    "PooledScorerTrainingData",
    "ProbeName",
    "RANDOM_CANDIDATE_NAMES",
    "RankingLoss",
    "SCORER_EXPERIMENT_PROTOCOL",
    "SourceEligibility",
    "TaskLabelNormalizer",
    "TinyCandidateScorer",
    "TrainedCandidateScorer",
    "assemble_pooled_clean_training_data",
    "assemble_grouped_scorer_data",
    "audit_scorer_experiment_result",
    "build_fixed_scorer",
    "candidate_feature_array",
    "cyclic_candidate_shuffled_predicted_inputs",
    "episode_task_macro_bootstrap",
    "fit_feature_scaler",
    "fit_task_label_normalizer",
    "higher_is_better_ranking_metrics",
    "hidden_width_for_budget",
    "paired_episode_task_macro_bootstrap",
    "pairwise_ranking_huber_loss",
    "pairwise_targets",
    "parameter_match_report",
    "publish_scorer_directory_exclusive",
    "run_scorer_experiment",
    "score_validation_split",
    "scorer_state_sha256",
    "shuffled_predicted_inputs",
    "summarize_e1_e3_metrics",
    "task_onehot_feature_indices",
    "train_full_batch_candidate_scorer",
    "trainable_parameter_count",
    "validate_scorer_experiment_config",
    "validate_source_result_sha256",
]
