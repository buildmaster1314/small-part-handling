"""P0/P1 evaluation for the Counterfactual v3 StateOracle gate.

The module consumes an already loaded v3 array bundle.  It deliberately keeps
three channels separate:

* the privileged DirectRanker receives only current physical state, candidate
  action, explicit world goal, and task identity;
* the deterministic StateOracle receives true future physical trajectories;
* a deterministic physical-task score recomputed from future state is the
  primary target; branch return is only a secondary P0 informativeness check.

No visual feature, LeWM latent, reward, return, or success value is accepted by
either scorer.  This is an external-development protocol, not a deployable
controller.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from fcpg.lewm_counterfactual import spearman_rank_correlation
from fcpg.lewm_counterfactual_v2_scorer import (
    TaskLabelNormalizer,
    fit_task_label_normalizer,
    pairwise_ranking_huber_loss,
    scorer_state_sha256,
)
from fcpg.lewm_counterfactual_v3 import (
    CANDIDATE_COUNT,
    TASK_NAMES,
    V3Axes,
    validate_action_hash_bindings,
    validate_array_bundle,
    validate_index_rows,
)
from fcpg.lewm_counterfactual_v3_state import (
    STATE_DIM,
    deterministic_oracle_scores,
    validate_state_array,
)


DIRECT_TRAINING_SEEDS = (9241, 9242, 9243)
SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
TASK_COUNT = len(TASK_NAMES)
MINIMUM_PHYSICAL_SCORE_SPREAD = 0.10
MINIMUM_ACTUAL_RETURN_SPREAD = 0.01
DIRECT_MINIMUM_RELATIVE_LOSS_REDUCTION = 0.10
DIRECT_MINIMUM_TRAIN_SPEARMAN = 0.20


class StopGate(RuntimeError):
    """Raised when a fail-closed prerequisite invalidates the next comparison."""

    def __init__(
        self,
        report: Mapping[str, Any],
        *,
        direct_ensemble: DirectEnsembleResult | None = None,
    ) -> None:
        super().__init__("Counterfactual v3 prerequisite failed; later gates are forbidden")
        self.report = report
        self.direct_ensemble = direct_ensemble


def _readonly(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, copy=True, order="C", dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class V3GateBundle:
    """Validated aligned views of one already loaded v3 store."""

    deployment_inputs: Mapping[str, np.ndarray]
    offline_labels: Mapping[str, np.ndarray]
    audit_arrays: Mapping[str, np.ndarray]
    index_rows: tuple[Mapping[str, Any], ...]
    axes: V3Axes

    @property
    def train_mask(self) -> np.ndarray:
        return self.audit_arrays["split_index"] == SPLIT_TRAIN

    @property
    def validation_mask(self) -> np.ndarray:
        return self.audit_arrays["split_index"] == SPLIT_VALIDATION

    @property
    def terminal_return(self) -> np.ndarray:
        return self.offline_labels["actual_return"][:, :, -1]

    @property
    def physical_task_score(self) -> np.ndarray:
        """Task-aligned physical score used by every primary ranking comparison."""

        return self.offline_labels["physical_task_score"]


def prepare_gate_bundle(
    deployment_inputs: Mapping[str, np.ndarray],
    offline_labels: Mapping[str, np.ndarray],
    audit_arrays: Mapping[str, np.ndarray],
    index_rows: Sequence[Mapping[str, Any]],
) -> V3GateBundle:
    """Validate array/index alignment and the privileged-state schema."""

    axes = validate_array_bundle(deployment_inputs, offline_labels, audit_arrays)
    validate_index_rows(index_rows, axes=axes)
    validate_action_hash_bindings(deployment_inputs, index_rows)
    if axes.state_dim != STATE_DIM:
        raise ValueError(
            f"v3 StateOracle gate requires state_dim={STATE_DIM}, got {axes.state_dim}"
        )
    validate_state_array(offline_labels["current_state"], name="current_state")
    validate_state_array(offline_labels["future_state"], name="future_state")
    if not np.any(audit_arrays["split_index"] == SPLIT_TRAIN):
        raise ValueError("v3 gate has no training snapshots")
    if not np.any(audit_arrays["split_index"] == SPLIT_VALIDATION):
        raise ValueError("v3 gate has no validation snapshots")

    for sample, row in enumerate(index_rows):
        task_index = int(audit_arrays["task_index"][sample])
        split_index = int(audit_arrays["split_index"][sample])
        if row["task"] != TASK_NAMES[task_index]:
            raise ValueError("index task does not match task_index array")
        expected_split = "train" if split_index == SPLIT_TRAIN else "validation"
        if row["split"] != expected_split:
            raise ValueError("index split does not match split_index array")
        if int(row["episode_id"]) != int(audit_arrays["episode_id"][sample]):
            raise ValueError("index episode_id does not match episode_id array")
        if int(row["snapshot_step"]) != int(audit_arrays["snapshot_step"][sample]):
            raise ValueError("index snapshot_step does not match audit array")

    return V3GateBundle(
        deployment_inputs=deployment_inputs,
        offline_labels=offline_labels,
        audit_arrays=audit_arrays,
        index_rows=tuple(index_rows),
        axes=axes,
    )


def validate_physical_score_contract(bundle: V3GateBundle) -> None:
    """Recompute every stored primary label from privileged physical state.

    The collector stores ``physical_task_score`` for auditing and efficient
    training, but the formal gate must not trust that serialized label.  This
    check binds it exactly to the frozen StateOracle implementation before any
    shortcut audit or DirectRanker training can begin.
    """

    expected = np.empty_like(bundle.physical_task_score, dtype=np.float32)
    for sample in range(bundle.axes.sample_count):
        oracle = deterministic_oracle_scores(
            TASK_NAMES[int(bundle.audit_arrays["task_index"][sample])],
            bundle.offline_labels["current_state"][sample],
            bundle.offline_labels["future_state"][sample],
            bundle.offline_labels["goal_world"][sample],
        )
        expected[sample] = oracle.score
    stored = np.asarray(bundle.physical_task_score, dtype=np.float32)
    if not np.array_equal(stored, expected):
        mismatch = np.argwhere(stored != expected)
        first = tuple(int(value) for value in mismatch[0])
        raise ValueError(
            "stored physical_task_score does not match the deterministic "
            f"StateOracle at index {first}: stored={stored[first]!r}, "
            f"expected={expected[first]!r}"
        )


@dataclass(frozen=True)
class SnapshotRankingMetrics:
    spearman: float
    normalized_top1_regret: float
    selected_indices: tuple[int, ...]
    return_spread: float


def snapshot_ranking_metrics(
    scores: np.ndarray,
    actual_return: np.ndarray,
    *,
    tie_tolerance: float = 1e-6,
) -> SnapshotRankingMetrics:
    """Compute tie-aware ranking metrics for one anonymous candidate bank."""

    predicted = np.asarray(scores, dtype=np.float64)
    target = np.asarray(actual_return, dtype=np.float64)
    if predicted.shape != (CANDIDATE_COUNT,) or target.shape != predicted.shape:
        raise ValueError("scores and actual_return must both have shape [8]")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("ranking inputs must be finite")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    spread = float(np.max(target) - np.min(target))
    # Candidate selection follows the scorer exactly.  ``tie_tolerance`` is a
    # physical-return informativeness threshold, not permission to alter a
    # learned scorer's argmax after seeing its output.
    selected = np.flatnonzero(predicted == np.max(predicted))
    if selected.size == 0:
        raise RuntimeError("score tie handling selected no candidate")
    if spread <= tie_tolerance:
        regret = 0.0
    else:
        selected_return = float(np.mean(target[selected]))
        regret = (float(np.max(target)) - selected_return) / spread
    return SnapshotRankingMetrics(
        spearman=float(spearman_rank_correlation(predicted, target)),
        normalized_top1_regret=float(regret),
        selected_indices=tuple(int(value) for value in selected),
        return_spread=spread,
    )


def _candidate_spread_audit(
    bundle: V3GateBundle,
    labels: np.ndarray,
    *,
    label: str,
    minimum_fraction: float = 0.8,
    minimum_spread: float,
) -> dict[str, Any]:
    """Require a meaningful candidate spread separately in every task."""

    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must be in [0,1]")
    if not np.isfinite(minimum_spread) or minimum_spread <= 0.0:
        raise ValueError("minimum_spread must be finite and positive")
    mask = bundle.validation_mask
    target = np.asarray(labels, dtype=np.float64)[mask]
    if target.ndim != 2 or target.shape[1] != CANDIDATE_COUNT:
        raise ValueError(f"{label} must have shape [N,8]")
    tasks = bundle.audit_arrays["task_index"][mask]
    informative = np.ptp(target, axis=1) >= minimum_spread
    per_task: dict[str, float] = {}
    for task in range(TASK_COUNT):
        task_values = informative[tasks == task]
        if task_values.size == 0:
            raise ValueError(f"spread audit has no validation rows for {TASK_NAMES[task]}")
        per_task[TASK_NAMES[task]] = float(np.mean(task_values))
    fraction = float(np.mean(informative))
    per_task_passed = {
        task: bool(value >= minimum_fraction) for task, value in per_task.items()
    }
    return {
        "passed": bool(all(per_task_passed.values())),
        "label": str(label),
        "fraction": fraction,
        "minimum_fraction": float(minimum_fraction),
        "minimum_spread": float(minimum_spread),
        "informative_snapshots": int(np.sum(informative)),
        "validation_snapshots": int(informative.size),
        "per_task_fraction": per_task,
        "per_task_passed": per_task_passed,
        "definition": "candidate_range_meets_fixed_minimum_in_every_task",
    }


def candidate_return_spread_audit(
    bundle: V3GateBundle,
    *,
    minimum_fraction: float = 0.8,
    minimum_spread: float = 0.01,
    tie_tolerance: float | None = None,
) -> dict[str, Any]:
    """Parallel P0 diagnostic for the real H5 cumulative return label.

    ``tie_tolerance`` remains a compatibility alias for focused callers.  The
    protocol entry point always supplies the pre-registered absolute spread.
    """

    if tie_tolerance is not None:
        minimum_spread = float(tie_tolerance)
    return _candidate_spread_audit(
        bundle,
        bundle.terminal_return,
        label="actual_return_h5",
        minimum_fraction=minimum_fraction,
        minimum_spread=minimum_spread,
    )


def candidate_physical_score_spread_audit(
    bundle: V3GateBundle,
    *,
    minimum_fraction: float = 0.8,
    minimum_spread: float = 0.1,
) -> dict[str, Any]:
    """Primary P0 informativeness audit for task-aligned physical scores."""

    return _candidate_spread_audit(
        bundle,
        bundle.physical_task_score,
        label="physical_task_score",
        minimum_fraction=minimum_fraction,
        minimum_spread=minimum_spread,
    )


def _expected_uniform_regret(
    target: np.ndarray,
    *,
    tie_tolerance: float,
) -> float:
    spread = float(np.ptp(target))
    if spread <= tie_tolerance:
        return 0.0
    return float(np.mean((np.max(target) - target) / spread))


def _task_macro(values: np.ndarray, task_index: np.ndarray) -> float:
    if values.ndim != 1 or task_index.shape != values.shape:
        raise ValueError("task macro inputs must be aligned vectors")
    return float(
        np.mean(
            [float(np.mean(values[task_index == task])) for task in range(TASK_COUNT)]
        )
    )


def _episode_task_macro_bootstrap_values(
    values: np.ndarray,
    task_index: np.ndarray,
    episode_sha256: Sequence[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Average phases within episode, then resample episodes within each task."""

    metric = np.asarray(values, dtype=np.float64)
    task = np.asarray(task_index)
    if metric.ndim != 1 or task.shape != metric.shape:
        raise ValueError("bootstrap values and task_index must be aligned vectors")
    if len(episode_sha256) != metric.size:
        raise ValueError("bootstrap episode identifiers are not sample aligned")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    grouped: dict[int, dict[str, list[float]]] = {
        index: defaultdict(list) for index in range(TASK_COUNT)
    }
    for value, task_value, episode in zip(
        metric, task, episode_sha256, strict=True
    ):
        grouped[int(task_value)][str(episode)].append(float(value))
    episode_values: dict[int, np.ndarray] = {}
    for task_value in range(TASK_COUNT):
        if not grouped[task_value]:
            raise ValueError(
                f"bootstrap has no validation episodes for {TASK_NAMES[task_value]}"
            )
        episode_values[task_value] = np.asarray(
            [np.mean(samples) for samples in grouped[task_value].values()],
            dtype=np.float64,
        )
    estimate = float(
        np.mean([np.mean(episode_values[index]) for index in range(TASK_COUNT)])
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for bootstrap_index in range(bootstrap_samples):
        task_samples = []
        for task_value in range(TASK_COUNT):
            candidates = episode_values[task_value]
            selection = rng.integers(0, candidates.size, size=candidates.size)
            task_samples.append(float(np.mean(candidates[selection])))
        samples[bootstrap_index] = float(np.mean(task_samples))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci95": [float(low), float(high)],
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "resampling_unit": "physical_episode_within_task",
        "phase_averaging": "within_physical_episode_before_resampling",
    }


def _fit_lookup(
    labels: np.ndarray,
    task_index: np.ndarray,
    key_arrays: Sequence[np.ndarray],
) -> dict[tuple[int, ...], float]:
    grouped: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for sample in range(labels.shape[0]):
        for candidate in range(CANDIDATE_COUNT):
            key = (int(task_index[sample]),) + tuple(
                int(value[sample, candidate]) for value in key_arrays
            )
            grouped[key].append(float(labels[sample, candidate]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _lookup_scores(
    lookup: Mapping[tuple[int, ...], float],
    task_index: np.ndarray,
    key_arrays: Sequence[np.ndarray],
) -> np.ndarray:
    result = np.empty((task_index.size, CANDIDATE_COUNT), dtype=np.float64)
    for sample in range(task_index.size):
        for candidate in range(CANDIDATE_COUNT):
            key = (int(task_index[sample]),) + tuple(
                int(value[sample, candidate]) for value in key_arrays
            )
            if key not in lookup:
                raise ValueError(f"validation shortcut key was absent in training: {key}")
            result[sample, candidate] = lookup[key]
    return result


def _shortcut_report(
    scores: np.ndarray,
    labels: np.ndarray,
    task_index: np.ndarray,
    episode_sha256: Sequence[str],
    *,
    abs_spearman_max: float,
    regret_improvement_max: float,
    tie_tolerance: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics = [
        snapshot_ranking_metrics(score, target, tie_tolerance=tie_tolerance)
        for score, target in zip(scores, labels, strict=True)
    ]
    spearman = np.asarray([value.spearman for value in metrics])
    regret = np.asarray([value.normalized_top1_regret for value in metrics])
    random_regret = np.asarray(
        [
            _expected_uniform_regret(value, tie_tolerance=tie_tolerance)
            for value in labels
        ]
    )
    macro_spearman = _task_macro(spearman, task_index)
    macro_regret = _task_macro(regret, task_index)
    macro_random_regret = _task_macro(random_regret, task_index)
    improvement = macro_random_regret - macro_regret
    spearman_bootstrap = _episode_task_macro_bootstrap_values(
        spearman,
        task_index,
        episode_sha256,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    regret_improvement_bootstrap = _episode_task_macro_bootstrap_values(
        random_regret - regret,
        task_index,
        episode_sha256,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 10_000,
    )
    ci_low, ci_high = spearman_bootstrap["ci95"]
    regret_ci_low, regret_ci_high = regret_improvement_bootstrap["ci95"]
    spearman_equivalent = (
        ci_low >= -abs_spearman_max and ci_high <= abs_spearman_max
    )
    regret_equivalent = regret_ci_high <= regret_improvement_max
    passed = (
        abs(macro_spearman) <= abs_spearman_max
        and improvement <= regret_improvement_max
        and spearman_equivalent
        and regret_equivalent
    )
    return {
        "passed": bool(passed),
        "task_macro_spearman": macro_spearman,
        "abs_spearman_max": float(abs_spearman_max),
        "task_macro_normalized_top1_regret": macro_regret,
        "uniform_random_normalized_top1_regret": macro_random_regret,
        "regret_improvement_over_uniform": float(improvement),
        "regret_improvement_max": float(regret_improvement_max),
        "episode_within_task_spearman_bootstrap": spearman_bootstrap,
        "episode_within_task_regret_improvement_bootstrap": (
            regret_improvement_bootstrap
        ),
        "spearman_equivalence_interval": [
            -float(abs_spearman_max),
            float(abs_spearman_max),
        ],
        "spearman_ci_inside_equivalence_interval": bool(spearman_equivalent),
        "regret_improvement_ci_upper_must_not_exceed": float(
            regret_improvement_max
        ),
        "regret_improvement_ci_passed": bool(regret_equivalent),
        "regret_improvement_ci95": [float(regret_ci_low), float(regret_ci_high)],
        "fit_scope": "training_split_only",
        "evaluation_scope": "validation_split_only",
    }


def shortcut_lookup_audits(
    bundle: V3GateBundle,
    *,
    abs_spearman_max: float = 0.1,
    regret_improvement_max: float = 0.02,
    tie_tolerance: float = 1e-6,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 9231,
) -> dict[str, Any]:
    """Audit train-only task x slot and task x hidden-axis/sign shortcuts."""

    train = bundle.train_mask
    validation = bundle.validation_mask
    labels = bundle.physical_task_score
    task = bundle.audit_arrays["task_index"]
    slots = np.broadcast_to(
        np.arange(CANDIDATE_COUNT, dtype=np.int64),
        (bundle.axes.sample_count, CANDIDATE_COUNT),
    )
    axis = np.asarray(bundle.audit_arrays["candidate_axis"])
    sign = np.asarray(bundle.audit_arrays["candidate_sign"])
    validation_indices = np.flatnonzero(validation)
    validation_episode_sha256 = [
        str(bundle.index_rows[index]["physical_episode_sha256"])
        for index in validation_indices
    ]
    if not np.isin(axis, np.arange(CANDIDATE_COUNT // 2)).all():
        raise ValueError("candidate_axis must be in [0,3]")
    if not np.isin(sign, (-1, 1)).all():
        raise ValueError("candidate_sign must be -1 or +1")

    reports: dict[str, Any] = {}
    for audit_index, (name, keys) in enumerate({
        "task_x_anonymous_slot": (slots,),
        "task_x_hidden_axis_sign": (axis, sign),
    }.items()):
        lookup = _fit_lookup(labels[train], task[train], [value[train] for value in keys])
        score = _lookup_scores(
            lookup,
            task[validation],
            [value[validation] for value in keys],
        )
        reports[name] = _shortcut_report(
            score,
            labels[validation],
            task[validation],
            validation_episode_sha256,
            abs_spearman_max=abs_spearman_max,
            regret_improvement_max=regret_improvement_max,
            tie_tolerance=tie_tolerance,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + audit_index,
        )
        reports[name]["lookup_cells"] = len(lookup)
    return {
        "passed": bool(all(report["passed"] for report in reports.values())),
        "primary_target": "physical_task_score",
        **reports,
    }


@dataclass(frozen=True)
class DirectRankerInputs:
    """The exact privileged inputs accepted by the DirectRanker."""

    current_state: np.ndarray
    candidate_action_raw: np.ndarray
    goal_world: np.ndarray
    task_onehot: np.ndarray

    def validate(self) -> None:
        n = self.current_state.shape[0]
        if self.current_state.shape != (n, STATE_DIM):
            raise ValueError(f"current_state must have shape [N,{STATE_DIM}]")
        if self.candidate_action_raw.shape != (n, CANDIDATE_COUNT, 25, 4):
            raise ValueError("candidate_action_raw must have shape [N,8,25,4]")
        if self.goal_world.shape != (n, 3):
            raise ValueError("goal_world must have shape [N,3]")
        if self.task_onehot.shape != (n, TASK_COUNT):
            raise ValueError("task_onehot must have shape [N,4]")
        for value in (
            self.current_state,
            self.candidate_action_raw,
            self.goal_world,
            self.task_onehot,
        ):
            if not np.isfinite(value).all():
                raise ValueError("DirectRanker inputs must be finite")


def direct_ranker_inputs(bundle: V3GateBundle) -> DirectRankerInputs:
    task = bundle.audit_arrays["task_index"].astype(np.int64, copy=False)
    onehot = np.eye(TASK_COUNT, dtype=np.float32)[task]
    result = DirectRankerInputs(
        current_state=_readonly(
            bundle.offline_labels["current_state"], dtype=np.dtype(np.float32)
        ),
        candidate_action_raw=_readonly(
            bundle.deployment_inputs["candidate_action_raw"],
            dtype=np.dtype(np.float32),
        ),
        goal_world=_readonly(
            bundle.offline_labels["goal_world"], dtype=np.dtype(np.float32)
        ),
        task_onehot=_readonly(onehot),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class DirectInputScaler:
    state_mean: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray
    goal_mean: np.ndarray
    goal_scale: np.ndarray

    def transform(self, inputs: DirectRankerInputs) -> DirectRankerInputs:
        inputs.validate()

        def apply(value: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
            shape = (1,) * (value.ndim - 1) + (-1,)
            return _readonly(
                (value.astype(np.float64) - mean.reshape(shape)) / scale.reshape(shape),
                dtype=np.dtype(np.float32),
            )

        result = DirectRankerInputs(
            current_state=apply(inputs.current_state, self.state_mean, self.state_scale),
            candidate_action_raw=apply(
                inputs.candidate_action_raw, self.action_mean, self.action_scale
            ),
            goal_world=apply(inputs.goal_world, self.goal_mean, self.goal_scale),
            task_onehot=_readonly(inputs.task_onehot, dtype=np.dtype(np.float32)),
        )
        result.validate()
        return result


def fit_direct_input_scaler(inputs: DirectRankerInputs) -> DirectInputScaler:
    """Fit feature scales from training inputs only; no label argument exists."""

    inputs.validate()

    def moments(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = value.astype(np.float64).reshape(-1, value.shape[-1])
        mean = flat.mean(axis=0)
        scale = flat.std(axis=0, ddof=0)
        scale[scale <= 1e-8] = 1.0
        return _readonly(mean), _readonly(scale)

    state_mean, state_scale = moments(inputs.current_state)
    action_mean, action_scale = moments(inputs.candidate_action_raw)
    goal_mean, goal_scale = moments(inputs.goal_world)
    return DirectInputScaler(
        state_mean,
        state_scale,
        action_mean,
        action_scale,
        goal_mean,
        goal_scale,
    )


def _select_inputs(inputs: DirectRankerInputs, selection: np.ndarray) -> DirectRankerInputs:
    result = DirectRankerInputs(
        current_state=_readonly(inputs.current_state[selection]),
        candidate_action_raw=_readonly(inputs.candidate_action_raw[selection]),
        goal_world=_readonly(inputs.goal_world[selection]),
        task_onehot=_readonly(inputs.task_onehot[selection]),
    )
    result.validate()
    return result


class PrivilegedDirectRanker(nn.Module):
    """Shared candidate scorer with no slot-specific parameters."""

    def __init__(
        self,
        *,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 64,
        action_hidden_dim: int = 32,
        task_dim: int = TASK_COUNT,
    ) -> None:
        super().__init__()
        if min(state_dim, hidden_dim, action_hidden_dim, task_dim) <= 0:
            raise ValueError("DirectRanker dimensions must be positive")
        if hidden_dim < 2:
            raise ValueError("DirectRanker hidden_dim must be at least 2")
        self.state_dim = int(state_dim)
        self.task_dim = int(task_dim)
        self.action_encoder = nn.GRU(4, action_hidden_dim, batch_first=True)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2), nn.GELU()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + action_hidden_dim + task_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        current_state: Tensor,
        candidate_action_raw: Tensor,
        goal_world: Tensor,
        task_onehot: Tensor,
    ) -> Tensor:
        if current_state.ndim != 2 or current_state.shape[-1] != self.state_dim:
            raise ValueError("current_state tensor shape changed")
        batch = current_state.shape[0]
        if candidate_action_raw.ndim != 4 or candidate_action_raw.shape[0] != batch:
            raise ValueError("candidate_action_raw must have shape [B,C,25,4]")
        candidates = candidate_action_raw.shape[1]
        if candidate_action_raw.shape[2:] != (25, 4):
            raise ValueError("candidate action sequence shape changed")
        if goal_world.shape != (batch, 3) or task_onehot.shape != (
            batch,
            self.task_dim,
        ):
            raise ValueError("goal/task tensor shape changed")
        action = candidate_action_raw.reshape(batch * candidates, 25, 4)
        _, action_hidden = self.action_encoder(action)
        action_context = action_hidden[-1].reshape(batch, candidates, -1)
        state_context = self.state_encoder(current_state)[:, None].expand(
            -1, candidates, -1
        )
        goal_context = self.goal_encoder(goal_world)[:, None].expand(
            -1, candidates, -1
        )
        task_context = task_onehot[:, None].expand(-1, candidates, -1)
        feature = torch.cat(
            (state_context, action_context, goal_context, task_context), dim=-1
        )
        return self.head(feature).squeeze(-1)


def _tensor_inputs(inputs: DirectRankerInputs, device: str) -> tuple[Tensor, ...]:
    inputs.validate()
    return tuple(
        torch.from_numpy(getattr(inputs, name).copy()).to(device=device)
        for name in DirectRankerInputs.__dataclass_fields__
    )


@dataclass(frozen=True)
class DirectEnsembleResult:
    training_scores: np.ndarray
    validation_scores: np.ndarray
    training_seeds: tuple[int, ...]
    parameter_count: int
    initial_losses: tuple[float, ...]
    final_losses: tuple[float, ...]
    state_dicts: tuple[Mapping[str, Tensor], ...]
    state_sha256: tuple[str, ...]
    input_scaler: DirectInputScaler
    label_normalizer: TaskLabelNormalizer
    model_config: Mapping[str, int]
    ensemble_rule: str = "average_seed_scores_before_metrics"


@dataclass(frozen=True)
class StateOracleGateRun:
    """JSON-ready report plus exact frozen Direct ensemble artifacts."""

    report: Mapping[str, Any]
    direct_ensemble: DirectEnsembleResult


def train_direct_ranker_ensemble(
    bundle: V3GateBundle,
    *,
    seeds: Sequence[int] = DIRECT_TRAINING_SEEDS,
    steps: int = 2000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    hidden_dim: int = 64,
    action_hidden_dim: int = 32,
    pairwise_weight: float = 1.0,
    huber_weight: float = 0.1,
    huber_beta: float = 1.0,
    tie_tolerance: float = 1e-6,
    device: str = "cpu",
) -> DirectEnsembleResult:
    """Train fixed-seed DirectRankers and average scores before evaluation."""

    seed_tuple = tuple(int(value) for value in seeds)
    if not seed_tuple or len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("training seeds must be non-empty and unique")
    if steps <= 0 or learning_rate <= 0.0 or gradient_clip_norm <= 0.0:
        raise ValueError("DirectRanker optimization constants must be positive")
    all_inputs = direct_ranker_inputs(bundle)
    train_inputs = _select_inputs(all_inputs, bundle.train_mask)
    validation_inputs = _select_inputs(all_inputs, bundle.validation_mask)
    scaler = fit_direct_input_scaler(train_inputs)
    train_inputs = scaler.transform(train_inputs)
    validation_inputs = scaler.transform(validation_inputs)
    train_task = bundle.audit_arrays["task_index"][bundle.train_mask]
    raw_labels = bundle.physical_task_score[bundle.train_mask]
    normalizer = fit_task_label_normalizer(raw_labels, train_task)
    labels = normalizer.transform(raw_labels, train_task)
    train_tensors = _tensor_inputs(train_inputs, device)
    validation_tensors = _tensor_inputs(validation_inputs, device)
    target = torch.from_numpy(labels.copy()).to(device=device)

    train_scores_by_seed: list[np.ndarray] = []
    validation_scores_by_seed: list[np.ndarray] = []
    initial_losses: list[float] = []
    final_losses: list[float] = []
    state_dicts: list[Mapping[str, Tensor]] = []
    state_hashes: list[str] = []
    parameter_count: int | None = None
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        for seed in seed_tuple:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = PrivilegedDirectRanker(
                hidden_dim=hidden_dim,
                action_hidden_dim=action_hidden_dim,
            ).to(device=device)
            count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            if parameter_count is None:
                parameter_count = count
            elif parameter_count != count:
                raise RuntimeError("DirectRanker parameter count changed across seeds")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            model.train()
            with torch.no_grad():
                initial = pairwise_ranking_huber_loss(
                    model(*train_tensors),
                    target,
                    tie_tolerance=tie_tolerance,
                    pairwise_weight=pairwise_weight,
                    huber_weight=huber_weight,
                    huber_beta=huber_beta,
                )
            initial_loss = float(initial.total.detach().cpu())
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                scores = model(*train_tensors)
                loss = pairwise_ranking_huber_loss(
                    scores,
                    target,
                    tie_tolerance=tie_tolerance,
                    pairwise_weight=pairwise_weight,
                    huber_weight=huber_weight,
                    huber_beta=huber_beta,
                )
                loss.total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_score_tensor = model(*train_tensors)
                final = pairwise_ranking_huber_loss(
                    train_score_tensor,
                    target,
                    tie_tolerance=tie_tolerance,
                    pairwise_weight=pairwise_weight,
                    huber_weight=huber_weight,
                    huber_beta=huber_beta,
                )
                train_score = train_score_tensor.detach().cpu().numpy()
                validation_score = model(*validation_tensors).detach().cpu().numpy()
            final_loss = float(final.total.detach().cpu())
            if not np.isfinite(train_score).all() or not np.isfinite(
                validation_score
            ).all():
                raise RuntimeError("DirectRanker produced non-finite scores")
            if not np.isfinite(initial_loss) or not np.isfinite(final_loss):
                raise RuntimeError("DirectRanker produced non-finite training losses")
            state_hashes.append(scorer_state_sha256(model))
            state_dicts.append(
                {
                    name: tensor.detach().cpu().contiguous().clone()
                    for name, tensor in model.state_dict().items()
                }
            )
            train_scores_by_seed.append(train_score)
            validation_scores_by_seed.append(validation_score)
            initial_losses.append(initial_loss)
            final_losses.append(final_loss)
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before,
            warn_only=warn_only_before,
        )
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
    training_ensemble = np.mean(
        np.stack(train_scores_by_seed), axis=0, dtype=np.float64
    )
    validation_ensemble = np.mean(
        np.stack(validation_scores_by_seed), axis=0, dtype=np.float64
    )
    return DirectEnsembleResult(
        training_scores=_readonly(training_ensemble, dtype=np.dtype(np.float32)),
        validation_scores=_readonly(validation_ensemble, dtype=np.dtype(np.float32)),
        training_seeds=seed_tuple,
        parameter_count=int(parameter_count or 0),
        initial_losses=tuple(initial_losses),
        final_losses=tuple(final_losses),
        state_dicts=tuple(state_dicts),
        state_sha256=tuple(state_hashes),
        input_scaler=scaler,
        label_normalizer=normalizer,
        model_config={
            "state_dim": STATE_DIM,
            "hidden_dim": int(hidden_dim),
            "action_hidden_dim": int(action_hidden_dim),
            "task_dim": TASK_COUNT,
        },
    )


def _ranking_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    task_index: np.ndarray,
    *,
    tie_tolerance: float,
) -> dict[str, float]:
    values = np.asarray(scores)
    target = np.asarray(labels)
    task = np.asarray(task_index)
    if values.shape != target.shape or values.ndim != 2:
        raise ValueError("ranking summary scores and labels must share shape [N,8]")
    if task.shape != (values.shape[0],):
        raise ValueError("ranking summary task_index is not sample aligned")
    metrics = [
        snapshot_ranking_metrics(score, label, tie_tolerance=tie_tolerance)
        for score, label in zip(values, target, strict=True)
    ]
    spearman = np.asarray([metric.spearman for metric in metrics], dtype=np.float64)
    regret = np.asarray(
        [metric.normalized_top1_regret for metric in metrics], dtype=np.float64
    )
    return {
        "task_macro_spearman": _task_macro(spearman, task),
        "task_macro_normalized_top1_regret": _task_macro(regret, task),
    }


def direct_training_eligibility(
    bundle: V3GateBundle,
    ensemble: DirectEnsembleResult,
    *,
    minimum_relative_loss_reduction: float = DIRECT_MINIMUM_RELATIVE_LOSS_REDUCTION,
    minimum_train_spearman: float = DIRECT_MINIMUM_TRAIN_SPEARMAN,
    tie_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Qualify Direct optimization before it may serve as a P1 baseline."""

    if not 0.0 < minimum_relative_loss_reduction < 1.0:
        raise ValueError("minimum_relative_loss_reduction must be in (0,1)")
    if not -1.0 <= minimum_train_spearman <= 1.0:
        raise ValueError("minimum_train_spearman must be in [-1,1]")
    initial = np.asarray(ensemble.initial_losses, dtype=np.float64)
    final = np.asarray(ensemble.final_losses, dtype=np.float64)
    if initial.shape != final.shape or initial.shape != (len(ensemble.training_seeds),):
        raise ValueError("Direct loss records do not match training seeds")
    denominator = np.maximum(np.abs(initial), np.finfo(np.float64).eps)
    relative_reduction = (initial - final) / denominator

    train = bundle.train_mask
    labels = bundle.physical_task_score[train]
    task = bundle.audit_arrays["task_index"][train]
    slots = np.broadcast_to(
        np.arange(CANDIDATE_COUNT, dtype=np.int64), labels.shape
    )
    axis = bundle.audit_arrays["candidate_axis"][train]
    sign = bundle.audit_arrays["candidate_sign"][train]
    baselines: dict[str, np.ndarray] = {
        "uniform": np.zeros_like(labels, dtype=np.float64),
    }
    for name, keys in {
        "task_x_anonymous_slot": (slots,),
        "task_x_hidden_axis_sign": (axis, sign),
    }.items():
        lookup = _fit_lookup(labels, task, keys)
        baselines[name] = _lookup_scores(lookup, task, keys)

    direct_summary = _ranking_summary(
        ensemble.training_scores,
        labels,
        task,
        tie_tolerance=tie_tolerance,
    )
    baseline_summaries = {
        name: _ranking_summary(
            scores,
            labels,
            task,
            tie_tolerance=tie_tolerance,
        )
        for name, scores in baselines.items()
    }
    strongest_baseline_spearman = max(
        summary["task_macro_spearman"]
        for summary in baseline_summaries.values()
    )
    losses_passed = bool(
        np.all(relative_reduction >= minimum_relative_loss_reduction)
    )
    ranking_passed = bool(
        direct_summary["task_macro_spearman"] >= minimum_train_spearman
        and direct_summary["task_macro_spearman"] > strongest_baseline_spearman
    )
    return {
        "passed": bool(losses_passed and ranking_passed),
        "primary_target": "physical_task_score",
        "initial_loss_by_seed": initial.tolist(),
        "final_loss_by_seed": final.tolist(),
        "relative_loss_reduction_by_seed": relative_reduction.tolist(),
        "minimum_relative_loss_reduction": float(
            minimum_relative_loss_reduction
        ),
        "all_seeds_loss_reduction_passed": losses_passed,
        "direct_train_metrics": direct_summary,
        "training_baseline_metrics": baseline_summaries,
        "strongest_baseline_train_spearman": float(strongest_baseline_spearman),
        "minimum_train_spearman": float(minimum_train_spearman),
        "direct_strictly_exceeds_training_priors": bool(
            direct_summary["task_macro_spearman"] > strongest_baseline_spearman
        ),
        "ranking_passed": ranking_passed,
        "evaluation_scope": "training_split_qualification_only",
    }


def deterministic_state_oracle_score_matrix(
    bundle: V3GateBundle,
    *,
    cyclic_candidate_shift: int = 0,
) -> np.ndarray:
    """Score validation trajectories, optionally after a candidate-axis cycle."""

    if not isinstance(cyclic_candidate_shift, int):
        raise TypeError("cyclic_candidate_shift must be an integer")
    indices = np.flatnonzero(bundle.validation_mask)
    result = np.empty((indices.size, CANDIDATE_COUNT), dtype=np.float32)
    for output_index, sample in enumerate(indices):
        future = bundle.offline_labels["future_state"][sample]
        if cyclic_candidate_shift:
            # Public slots are independently anonymized at every snapshot.
            # Cycle canonical QR axes while preserving direction sign, then
            # map back to public slots; a raw np.roll over slots would not be
            # the pre-registered candidate-axis mismatch control.
            permutation = np.asarray(
                bundle.audit_arrays["candidate_permutation"][sample],
                dtype=np.int64,
            )
            if sorted(permutation.tolist()) != list(range(CANDIDATE_COUNT)):
                raise ValueError("candidate_permutation is not a permutation")
            inverse = np.empty(CANDIDATE_COUNT, dtype=np.int64)
            inverse[permutation] = np.arange(CANDIDATE_COUNT, dtype=np.int64)
            target_canonical = (
                ((permutation // 2 + cyclic_candidate_shift) % (CANDIDATE_COUNT // 2))
                * 2
                + permutation % 2
            )
            future = future[inverse[target_canonical]]
        oracle = deterministic_oracle_scores(
            TASK_NAMES[int(bundle.audit_arrays["task_index"][sample])],
            bundle.offline_labels["current_state"][sample],
            future,
            bundle.offline_labels["goal_world"][sample],
        )
        result[output_index] = oracle.score
    return _readonly(result)


def metric_rows(
    bundle: V3GateBundle,
    scores: np.ndarray,
    *,
    scorer: str,
    target_labels: np.ndarray | None = None,
    target_name: str = "physical_task_score",
    tie_tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """Return one metric row per validation physical snapshot."""

    indices = np.flatnonzero(bundle.validation_mask)
    values = np.asarray(scores)
    labels = (
        bundle.physical_task_score
        if target_labels is None
        else np.asarray(target_labels)
    )
    if labels.shape != bundle.physical_task_score.shape:
        raise ValueError("metric target labels must have shape [N,8]")
    if values.shape != (indices.size, CANDIDATE_COUNT):
        raise ValueError("validation score matrix shape changed")
    result: list[dict[str, Any]] = []
    for score, sample in zip(values, indices, strict=True):
        metric = snapshot_ranking_metrics(
            score,
            labels[sample],
            tie_tolerance=tie_tolerance,
        )
        row = bundle.index_rows[sample]
        result.append(
            {
                "sample_index": int(sample),
                "task": str(row["task"]),
                "physical_episode_sha256": str(row["physical_episode_sha256"]),
                "physical_snapshot_sha256": str(row["physical_snapshot_sha256"]),
                "phase_index": int(bundle.audit_arrays["phase_index"][sample]),
                "scorer": str(scorer),
                "primary_target": str(target_name),
                "spearman": metric.spearman,
                "normalized_top1_regret": metric.normalized_top1_regret,
                "return_spread": metric.return_spread,
                "selected_indices": list(metric.selected_indices),
            }
        )
    return result


def physical_return_alignment_audit(
    bundle: V3GateBundle,
    *,
    tie_tolerance: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Prove the frozen physical task-cost agrees with actual branch return."""

    physical = bundle.physical_task_score[bundle.validation_mask]
    uniform = np.zeros_like(physical, dtype=np.float32)
    physical_rows = metric_rows(
        bundle,
        physical,
        scorer="physical_task_score",
        target_labels=bundle.terminal_return,
        target_name="H5_cumulative_actual_return",
        tie_tolerance=tie_tolerance,
    )
    uniform_rows = metric_rows(
        bundle,
        uniform,
        scorer="uniform_control",
        target_labels=bundle.terminal_return,
        target_name="H5_cumulative_actual_return",
        tie_tolerance=tie_tolerance,
    )
    report = _comparison_report(
        physical_rows,
        uniform_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        **report,
        "required": True,
        "interpretation": (
            "The frozen physical task-cost must rank actual H5 return better "
            "than an uninformative uniform scorer before it may be a P1 target."
        ),
    }


def _index_metric_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row["physical_snapshot_sha256"])
        if key in result:
            raise ValueError("metric rows contain a duplicate physical snapshot")
        result[key] = row
    return result


def paired_episode_within_task_bootstrap(
    better_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    higher_is_better: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Bootstrap paired improvements after averaging phases within episodes."""

    if metric not in {"spearman", "normalized_top1_regret"}:
        raise ValueError("unsupported v3 gate metric")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    better = _index_metric_rows(better_rows)
    baseline = _index_metric_rows(baseline_rows)
    if set(better) != set(baseline):
        raise ValueError("paired scorers do not share exact validation snapshots")
    sign = 1.0 if higher_is_better else -1.0
    grouped: dict[str, dict[str, list[float]]] = {
        task: defaultdict(list) for task in TASK_NAMES
    }
    for key in sorted(better):
        left, right = better[key], baseline[key]
        if left["task"] != right["task"] or left["physical_episode_sha256"] != right[
            "physical_episode_sha256"
        ]:
            raise ValueError("paired metric row metadata changed")
        grouped[str(left["task"])][str(left["physical_episode_sha256"])].append(
            sign * (float(left[metric]) - float(right[metric]))
        )
    episode_values: dict[str, np.ndarray] = {}
    task_means: dict[str, float] = {}
    for task in TASK_NAMES:
        episodes = grouped[task]
        if not episodes:
            raise ValueError(f"bootstrap has no validation episodes for {task}")
        values = np.asarray(
            [np.mean(samples) for samples in episodes.values()], dtype=np.float64
        )
        episode_values[task] = values
        task_means[task] = float(np.mean(values))
    estimate = float(np.mean(list(task_means.values())))
    rng = np.random.default_rng(int(bootstrap_seed))
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for bootstrap_index in range(bootstrap_samples):
        task_samples = []
        for task in TASK_NAMES:
            values = episode_values[task]
            selected = rng.integers(0, values.size, size=values.size)
            task_samples.append(float(np.mean(values[selected])))
        samples[bootstrap_index] = float(np.mean(task_samples))
    low, high = np.percentile(samples, [2.5, 97.5])
    positive_task_count = sum(value > 0.0 for value in task_means.values())
    return {
        "estimate": estimate,
        "ci95": [float(low), float(high)],
        "task_mean_improvement": task_means,
        "positive_task_count": int(positive_task_count),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "resampling_unit": "physical_episode_within_task",
        "phase_averaging": "within_physical_episode_before_resampling",
        "orientation": "positive_means_better_scorer_improves_metric",
    }


def _comparison_report(
    better: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    spearman = paired_episode_within_task_bootstrap(
        better,
        baseline,
        metric="spearman",
        higher_is_better=True,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    regret = paired_episode_within_task_bootstrap(
        better,
        baseline,
        metric="normalized_top1_regret",
        higher_is_better=False,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 1,
    )
    jointly_positive_tasks = [
        task
        for task in TASK_NAMES
        if spearman["task_mean_improvement"][task] > 0.0
        and regret["task_mean_improvement"][task] > 0.0
    ]
    passed = (
        spearman["ci95"][0] > 0.0
        and regret["ci95"][0] > 0.0
        and len(jointly_positive_tasks) >= 3
    )
    return {
        "passed": bool(passed),
        "spearman": spearman,
        "normalized_top1_regret": regret,
        "jointly_positive_tasks": jointly_positive_tasks,
        "jointly_positive_task_count": len(jointly_positive_tasks),
        "criteria": {
            "paired_ci_lower_must_exceed": 0.0,
            "minimum_jointly_positive_tasks": 3,
        },
    }


def evaluate_state_oracle_gate(
    bundle: V3GateBundle,
    direct_validation_scores: np.ndarray,
    *,
    minimum_return_spread_fraction: float = 0.8,
    minimum_physical_score_spread: float = 0.1,
    minimum_actual_return_spread: float = 0.01,
    shortcut_abs_spearman_max: float = 0.1,
    shortcut_regret_improvement_max: float = 0.02,
    tie_tolerance: float = 1e-6,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 9231,
) -> dict[str, Any]:
    """Evaluate P0 shortcuts and the deterministic P1 StateOracle gate."""

    validate_physical_score_contract(bundle)

    physical_spread = candidate_physical_score_spread_audit(
        bundle,
        minimum_fraction=minimum_return_spread_fraction,
        minimum_spread=minimum_physical_score_spread,
    )
    actual_return_spread = candidate_return_spread_audit(
        bundle,
        minimum_fraction=minimum_return_spread_fraction,
        minimum_spread=minimum_actual_return_spread,
    )
    shortcuts = shortcut_lookup_audits(
        bundle,
        abs_spearman_max=shortcut_abs_spearman_max,
        regret_improvement_max=shortcut_regret_improvement_max,
        tie_tolerance=tie_tolerance,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 100,
    )
    return_alignment = physical_return_alignment_audit(
        bundle,
        tie_tolerance=tie_tolerance,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 200,
    )
    oracle_scores = deterministic_state_oracle_score_matrix(bundle)
    cyclic_scores = deterministic_state_oracle_score_matrix(
        bundle, cyclic_candidate_shift=1
    )
    rows = {
        "direct": metric_rows(
            bundle, direct_validation_scores, scorer="privileged_direct"
        ),
        "state_oracle": metric_rows(bundle, oracle_scores, scorer="state_oracle"),
        "cyclic_future": metric_rows(
            bundle, cyclic_scores, scorer="cyclic_candidate_axis_future"
        ),
    }
    versus_direct = _comparison_report(
        rows["state_oracle"],
        rows["direct"],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    versus_cyclic = _comparison_report(
        rows["state_oracle"],
        rows["cyclic_future"],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 10,
    )
    common_positive_tasks = sorted(
        set(versus_direct["jointly_positive_tasks"]).intersection(
            versus_cyclic["jointly_positive_tasks"]
        )
    )
    common_task_requirement_passed = len(common_positive_tasks) >= 3
    passed = (
        physical_spread["passed"]
        and actual_return_spread["passed"]
        and shortcuts["passed"]
        and return_alignment["passed"]
        and versus_direct["passed"]
        and versus_cyclic["passed"]
        and common_task_requirement_passed
    )
    return {
        "passed": bool(passed),
        "stage": "P1_state_oracle",
        "direct_training_started": True,
        "candidate_physical_score_spread": physical_spread,
        "candidate_actual_return_spread": actual_return_spread,
        "shortcut_audits": shortcuts,
        "physical_return_alignment": return_alignment,
        "comparisons": {
            "state_oracle_vs_privileged_direct": versus_direct,
            "state_oracle_vs_cyclic_future": versus_cyclic,
        },
        "cross_baseline_jointly_positive_tasks": common_positive_tasks,
        "cross_baseline_jointly_positive_task_count": len(common_positive_tasks),
        "cross_baseline_three_task_requirement_passed": (
            common_task_requirement_passed
        ),
        "snapshot_metrics": rows,
        "claim_boundary": (
            "Privileged external-development diagnostic only; no learned future "
            "delta, visual oracle, LeWM retraining, GoalToken, FCPG comparison, or CEM."
        ),
    }


def run_state_oracle_gate(
    bundle: V3GateBundle,
    gate_config: Mapping[str, Any],
    *,
    device: str,
) -> StateOracleGateRun:
    """Train the pre-registered Direct ensemble, then run the P0/P1 gate.

    This is the protocol-facing entry point.  The lower-level training helper
    accepts custom seeds for focused tests, whereas this function fails closed
    unless the three pre-declared seeds are used.
    """

    required = {
        "minimum_return_spread_fraction",
        "minimum_physical_score_spread",
        "minimum_actual_return_spread",
        "physical_return_alignment_required",
        "shortcut_abs_spearman_max",
        "shortcut_regret_improvement_max",
        "bootstrap_samples",
        "bootstrap_seed",
        "direct_training_seeds",
        "direct_steps",
        "direct_learning_rate",
        "direct_weight_decay",
        "direct_gradient_clip_norm",
        "direct_hidden_dim",
        "direct_action_hidden_dim",
        "direct_minimum_relative_loss_reduction",
        "direct_minimum_train_spearman",
        "pairwise_weight",
        "huber_weight",
        "huber_beta",
        "tie_tolerance",
    }
    if set(gate_config) != required:
        missing = sorted(required - set(gate_config))
        extra = sorted(set(gate_config) - required)
        raise ValueError(f"v3 gate config keys changed: missing={missing}, extra={extra}")
    validate_physical_score_contract(bundle)
    seeds = tuple(int(value) for value in gate_config["direct_training_seeds"])
    if seeds != DIRECT_TRAINING_SEEDS:
        raise ValueError(
            f"v3 DirectRanker seeds must remain {DIRECT_TRAINING_SEEDS}, got {seeds}"
        )
    frozen_values = {
        "minimum_physical_score_spread": MINIMUM_PHYSICAL_SCORE_SPREAD,
        "minimum_actual_return_spread": MINIMUM_ACTUAL_RETURN_SPREAD,
        "direct_minimum_relative_loss_reduction": (
            DIRECT_MINIMUM_RELATIVE_LOSS_REDUCTION
        ),
        "direct_minimum_train_spearman": DIRECT_MINIMUM_TRAIN_SPEARMAN,
    }
    for name, expected in frozen_values.items():
        if float(gate_config[name]) != expected:
            raise ValueError(f"v3 gate {name} must remain {expected}")
    if gate_config["physical_return_alignment_required"] is not True:
        raise ValueError("v3 physical-return alignment gate must remain enabled")
    p0_physical_spread = candidate_physical_score_spread_audit(
        bundle,
        minimum_fraction=float(gate_config["minimum_return_spread_fraction"]),
        minimum_spread=float(gate_config["minimum_physical_score_spread"]),
    )
    p0_actual_return_spread = candidate_return_spread_audit(
        bundle,
        minimum_fraction=float(gate_config["minimum_return_spread_fraction"]),
        minimum_spread=float(gate_config["minimum_actual_return_spread"]),
    )
    p0_shortcuts = shortcut_lookup_audits(
        bundle,
        abs_spearman_max=float(gate_config["shortcut_abs_spearman_max"]),
        regret_improvement_max=float(
            gate_config["shortcut_regret_improvement_max"]
        ),
        tie_tolerance=float(gate_config["tie_tolerance"]),
        bootstrap_samples=int(gate_config["bootstrap_samples"]),
        bootstrap_seed=int(gate_config["bootstrap_seed"]) + 100,
    )
    p0_return_alignment = physical_return_alignment_audit(
        bundle,
        tie_tolerance=float(gate_config["tie_tolerance"]),
        bootstrap_samples=int(gate_config["bootstrap_samples"]),
        bootstrap_seed=int(gate_config["bootstrap_seed"]) + 200,
    )
    if (
        not p0_physical_spread["passed"]
        or not p0_actual_return_spread["passed"]
        or not p0_shortcuts["passed"]
        or not p0_return_alignment["passed"]
    ):
        raise StopGate(
            {
                "passed": False,
                "stage": "P0_candidate_protocol",
                "candidate_physical_score_spread": p0_physical_spread,
                "candidate_actual_return_spread": p0_actual_return_spread,
                "shortcut_audits": p0_shortcuts,
                "physical_return_alignment": p0_return_alignment,
                "direct_training_started": False,
                "claim_boundary": (
                    "Candidate protocol failed P0; no DirectRanker or StateOracle "
                    "comparison was run."
                ),
            }
        )
    ensemble = train_direct_ranker_ensemble(
        bundle,
        seeds=seeds,
        steps=int(gate_config["direct_steps"]),
        learning_rate=float(gate_config["direct_learning_rate"]),
        weight_decay=float(gate_config["direct_weight_decay"]),
        gradient_clip_norm=float(gate_config["direct_gradient_clip_norm"]),
        hidden_dim=int(gate_config["direct_hidden_dim"]),
        action_hidden_dim=int(gate_config["direct_action_hidden_dim"]),
        pairwise_weight=float(gate_config["pairwise_weight"]),
        huber_weight=float(gate_config["huber_weight"]),
        huber_beta=float(gate_config["huber_beta"]),
        tie_tolerance=float(gate_config["tie_tolerance"]),
        device=device,
    )
    training_eligibility = direct_training_eligibility(
        bundle,
        ensemble,
        minimum_relative_loss_reduction=float(
            gate_config["direct_minimum_relative_loss_reduction"]
        ),
        minimum_train_spearman=float(
            gate_config["direct_minimum_train_spearman"]
        ),
        tie_tolerance=float(gate_config["tie_tolerance"]),
    )
    if not training_eligibility["passed"]:
        raise StopGate(
            {
                "passed": False,
                "stage": "P1_direct_training_invalid",
                "candidate_physical_score_spread": p0_physical_spread,
                "candidate_actual_return_spread": p0_actual_return_spread,
                "shortcut_audits": p0_shortcuts,
                "direct_training_started": True,
                "direct_training_eligibility": training_eligibility,
                "claim_boundary": (
                    "Direct baseline failed its frozen training qualification; "
                    "no StateOracle comparison is valid."
                ),
            },
            direct_ensemble=ensemble,
        )
    report = evaluate_state_oracle_gate(
        bundle,
        ensemble.validation_scores,
        minimum_return_spread_fraction=float(
            gate_config["minimum_return_spread_fraction"]
        ),
        minimum_physical_score_spread=float(
            gate_config["minimum_physical_score_spread"]
        ),
        minimum_actual_return_spread=float(
            gate_config["minimum_actual_return_spread"]
        ),
        shortcut_abs_spearman_max=float(gate_config["shortcut_abs_spearman_max"]),
        shortcut_regret_improvement_max=float(
            gate_config["shortcut_regret_improvement_max"]
        ),
        tie_tolerance=float(gate_config["tie_tolerance"]),
        bootstrap_samples=int(gate_config["bootstrap_samples"]),
        bootstrap_seed=int(gate_config["bootstrap_seed"]),
    )
    report = {
        **report,
        "direct_training_eligibility": training_eligibility,
        "direct_ensemble": {
            "training_seeds": list(ensemble.training_seeds),
            "ensemble_rule": ensemble.ensemble_rule,
            "parameter_count_per_seed": ensemble.parameter_count,
            "initial_training_loss_by_seed": list(ensemble.initial_losses),
            "final_training_loss_by_seed": list(ensemble.final_losses),
            "state_sha256_by_seed": list(ensemble.state_sha256),
            "model_config": dict(ensemble.model_config),
            "input_scaler_fit_scope": "training_inputs_only",
            "label_normalizer_fit_scope": "training_labels_only",
            "scores_averaged_before_metrics": True,
        },
    }
    return StateOracleGateRun(report=report, direct_ensemble=ensemble)


__all__ = [
    "DIRECT_MINIMUM_RELATIVE_LOSS_REDUCTION",
    "DIRECT_MINIMUM_TRAIN_SPEARMAN",
    "DIRECT_TRAINING_SEEDS",
    "MINIMUM_ACTUAL_RETURN_SPREAD",
    "MINIMUM_PHYSICAL_SCORE_SPREAD",
    "DirectEnsembleResult",
    "DirectInputScaler",
    "DirectRankerInputs",
    "PrivilegedDirectRanker",
    "SnapshotRankingMetrics",
    "StateOracleGateRun",
    "StopGate",
    "V3GateBundle",
    "candidate_physical_score_spread_audit",
    "candidate_return_spread_audit",
    "deterministic_state_oracle_score_matrix",
    "direct_ranker_inputs",
    "direct_training_eligibility",
    "evaluate_state_oracle_gate",
    "fit_direct_input_scaler",
    "metric_rows",
    "physical_return_alignment_audit",
    "paired_episode_within_task_bootstrap",
    "prepare_gate_bundle",
    "validate_physical_score_contract",
    "run_state_oracle_gate",
    "shortcut_lookup_audits",
    "snapshot_ranking_metrics",
    "train_direct_ranker_ensemble",
]
