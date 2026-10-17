"""Post-hoc actual-return diagnostic for the frozen Counterfactual v3 store.

This module deliberately does not reopen the formal v3 gate.  It reuses the
already collected train/validation split to answer one descriptive question:
does privileged future physical state improve H5 return ranking over a frozen
current-state/action Direct ensemble?

The target is always ``actual_return[:, :, -1]``.  ``physical_task_score`` is
neither a training target nor an evaluation target here.  Results from this
module are exploratory because the formal validation split has already been
exposed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fcpg.lewm_counterfactual_v2_scorer import (
    RankingLoss,
    TaskLabelNormalizer,
    fit_task_label_normalizer,
    scorer_state_sha256,
)
from fcpg.lewm_counterfactual_v3 import CANDIDATE_COUNT, HORIZON, TASK_NAMES
from fcpg.lewm_counterfactual_v3_gate import (
    DirectInputScaler,
    DirectRankerInputs,
    PrivilegedDirectRanker,
    V3GateBundle,
    direct_ranker_inputs,
    fit_direct_input_scaler,
    metric_rows,
    paired_episode_within_task_bootstrap,
    snapshot_ranking_metrics,
)
from fcpg.lewm_counterfactual_v3_state import STATE_DIM


TASK_COUNT = len(TASK_NAMES)
DEFAULT_DIRECT_SEEDS = (9241, 9242, 9243)
DEFAULT_RESIDUAL_SEEDS = (9251, 9252, 9253)
DEFAULT_RAW_RETURN_TIE_TOLERANCE = 0.01
CROSS_FIT_FOLDS = 4
CROSS_FIT_EPISODES_PER_TASK_PER_FOLD = 2


def _readonly(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, copy=True, order="C", dtype=dtype)
    result.setflags(write=False)
    return result


def _state_dict_copy(model: nn.Module) -> Mapping[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _parameter_count(model: nn.Module, *, trainable_only: bool = True) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def _validate_seeds(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{label} must be non-empty and unique")
    return seeds


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class _RngState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: tuple[Tensor, ...] | None


def _capture_rng_state() -> _RngState:
    numpy_state = np.random.get_state()
    return _RngState(
        python=random.getstate(),
        numpy=(
            numpy_state[0],
            np.array(numpy_state[1], copy=True),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=(
            tuple(value.clone() for value in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else None
        ),
    )


def _restore_rng_state(state: _RngState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(list(state.torch_cuda))


@dataclass(frozen=True)
class FutureResidualInputs:
    """Privileged inputs accepted by the exploratory future residual."""

    future_state: np.ndarray
    goal_world: np.ndarray
    task_onehot: np.ndarray

    def validate(self) -> None:
        n = int(self.future_state.shape[0])
        if self.future_state.shape != (
            n,
            CANDIDATE_COUNT,
            HORIZON,
            STATE_DIM,
        ):
            raise ValueError("future_state must have shape [N,8,5,state_dim]")
        if self.goal_world.shape != (n, 3):
            raise ValueError("goal_world must have shape [N,3]")
        if self.task_onehot.shape != (n, TASK_COUNT):
            raise ValueError("task_onehot must have shape [N,4]")
        for value in (self.future_state, self.goal_world, self.task_onehot):
            if not np.isfinite(value).all():
                raise ValueError("future residual inputs must be finite")


def future_residual_inputs(bundle: V3GateBundle) -> FutureResidualInputs:
    task = np.asarray(bundle.audit_arrays["task_index"], dtype=np.int64)
    if task.shape != (bundle.axes.sample_count,):
        raise ValueError("task_index shape changed")
    if np.any(task < 0) or np.any(task >= TASK_COUNT):
        raise ValueError("task_index is outside the task family")
    result = FutureResidualInputs(
        future_state=_readonly(
            bundle.offline_labels["future_state"], dtype=np.dtype(np.float32)
        ),
        goal_world=_readonly(
            bundle.offline_labels["goal_world"], dtype=np.dtype(np.float32)
        ),
        task_onehot=_readonly(np.eye(TASK_COUNT, dtype=np.float32)[task]),
    )
    result.validate()
    return result


def _select_direct_inputs(
    inputs: DirectRankerInputs, selection: np.ndarray
) -> DirectRankerInputs:
    result = DirectRankerInputs(
        current_state=_readonly(inputs.current_state[selection]),
        candidate_action_raw=_readonly(inputs.candidate_action_raw[selection]),
        goal_world=_readonly(inputs.goal_world[selection]),
        task_onehot=_readonly(inputs.task_onehot[selection]),
    )
    result.validate()
    return result


def _select_future_inputs(
    inputs: FutureResidualInputs, selection: np.ndarray
) -> FutureResidualInputs:
    result = FutureResidualInputs(
        future_state=_readonly(inputs.future_state[selection]),
        goal_world=_readonly(inputs.goal_world[selection]),
        task_onehot=_readonly(inputs.task_onehot[selection]),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class FutureInputScaler:
    """Training-only feature statistics for privileged future trajectories."""

    state_mean: np.ndarray
    state_scale: np.ndarray
    goal_mean: np.ndarray
    goal_scale: np.ndarray

    def transform(self, inputs: FutureResidualInputs) -> FutureResidualInputs:
        inputs.validate()
        future = (
            inputs.future_state.astype(np.float64)
            - self.state_mean.reshape(1, 1, 1, -1)
        ) / self.state_scale.reshape(1, 1, 1, -1)
        goal = (
            inputs.goal_world.astype(np.float64) - self.goal_mean.reshape(1, -1)
        ) / self.goal_scale.reshape(1, -1)
        result = FutureResidualInputs(
            future_state=_readonly(future, dtype=np.dtype(np.float32)),
            goal_world=_readonly(goal, dtype=np.dtype(np.float32)),
            task_onehot=_readonly(inputs.task_onehot, dtype=np.dtype(np.float32)),
        )
        result.validate()
        return result


def fit_future_input_scaler(inputs: FutureResidualInputs) -> FutureInputScaler:
    """Fit feature moments without accepting validation data or labels."""

    inputs.validate()

    def moments(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = value.astype(np.float64).reshape(-1, value.shape[-1])
        mean = flat.mean(axis=0)
        scale = flat.std(axis=0, ddof=0)
        scale[scale <= 1e-8] = 1.0
        return _readonly(mean), _readonly(scale)

    state_mean, state_scale = moments(inputs.future_state)
    goal_mean, goal_scale = moments(inputs.goal_world)
    return FutureInputScaler(state_mean, state_scale, goal_mean, goal_scale)


def _direct_tensors(inputs: DirectRankerInputs, device: str) -> tuple[Tensor, ...]:
    inputs.validate()
    return tuple(
        torch.from_numpy(getattr(inputs, name).copy()).to(device=device)
        for name in DirectRankerInputs.__dataclass_fields__
    )


def _future_tensors(inputs: FutureResidualInputs, device: str) -> tuple[Tensor, ...]:
    inputs.validate()
    return tuple(
        torch.from_numpy(getattr(inputs, name).copy()).to(device=device)
        for name in FutureResidualInputs.__dataclass_fields__
    )


class FutureStateResidualRanker(nn.Module):
    """Zero-initialized score delta over a frozen Direct ensemble score.

    The model is candidate-equivariant: every future trajectory is encoded by
    shared weights.  ``frozen_direct_score`` is detached inside ``forward`` so
    this module cannot update or emulate a jointly optimized Direct baseline.
    """

    def __init__(
        self,
        *,
        state_dim: int = STATE_DIM,
        horizon: int = HORIZON,
        hidden_dim: int = 64,
        trajectory_hidden_dim: int = 48,
        task_dim: int = TASK_COUNT,
    ) -> None:
        super().__init__()
        if min(state_dim, horizon, hidden_dim, trajectory_hidden_dim, task_dim) <= 0:
            raise ValueError("future residual dimensions must be positive")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.task_dim = int(task_dim)
        self.trajectory_encoder = nn.GRU(
            state_dim, trajectory_hidden_dim, batch_first=True
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2), nn.GELU()
        )
        self.delta_head = nn.Sequential(
            nn.Linear(trajectory_hidden_dim + hidden_dim // 2 + task_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.delta_head[-1]
        if not isinstance(final, nn.Linear):  # pragma: no cover - construction invariant.
            raise AssertionError("delta head must end in Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def delta(
        self,
        future_state: Tensor,
        goal_world: Tensor,
        task_onehot: Tensor,
    ) -> Tensor:
        if future_state.ndim != 4 or future_state.shape[2:] != (
            self.horizon,
            self.state_dim,
        ):
            raise ValueError("future_state must have shape [B,C,H,state_dim]")
        batch, candidates = future_state.shape[:2]
        if goal_world.shape != (batch, 3):
            raise ValueError("goal_world tensor shape changed")
        if task_onehot.shape != (batch, self.task_dim):
            raise ValueError("task_onehot tensor shape changed")
        trajectory = future_state.reshape(
            batch * candidates, self.horizon, self.state_dim
        )
        _, hidden = self.trajectory_encoder(trajectory)
        trajectory_context = hidden[-1].reshape(batch, candidates, -1)
        goal_context = self.goal_encoder(goal_world)[:, None].expand(
            -1, candidates, -1
        )
        task_context = task_onehot[:, None].expand(-1, candidates, -1)
        feature = torch.cat(
            (trajectory_context, goal_context, task_context), dim=-1
        )
        return self.delta_head(feature).squeeze(-1)

    def forward(
        self,
        frozen_direct_score: Tensor,
        future_state: Tensor,
        goal_world: Tensor,
        task_onehot: Tensor,
    ) -> Tensor:
        if frozen_direct_score.ndim != 2 or frozen_direct_score.shape[:2] != (
            future_state.shape[0],
            future_state.shape[1],
        ):
            raise ValueError("frozen_direct_score must have shape [B,C]")
        return frozen_direct_score.detach() + self.delta(
            future_state, goal_world, task_onehot
        )


def actual_return_ranking_loss(
    scores: Tensor,
    normalized_labels: Tensor,
    raw_labels: Tensor,
    *,
    raw_tie_tolerance: float = DEFAULT_RAW_RETURN_TIE_TOLERANCE,
    pairwise_weight: float = 1.0,
    huber_weight: float = 0.1,
    huber_beta: float = 1.0,
) -> RankingLoss:
    """Pairwise ranking with ties defined in raw H5-return units.

    Regression is performed on task-normalized labels, while the informative
    pair mask is computed before normalization.  This avoids accidentally
    interpreting ``0.01`` as a fraction of one task standard deviation.
    """

    if (
        scores.ndim != 2
        or normalized_labels.shape != scores.shape
        or raw_labels.shape != scores.shape
        or scores.shape[1] < 2
    ):
        raise ValueError("scores and both label tensors must share shape [B,C]")
    if not (
        torch.isfinite(scores).all()
        and torch.isfinite(normalized_labels).all()
        and torch.isfinite(raw_labels).all()
    ):
        raise ValueError("scores and labels must be finite")
    for name, value in (
        ("raw_tie_tolerance", raw_tie_tolerance),
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
    raw_difference = raw_labels[:, left] - raw_labels[:, right]
    informative = raw_difference.abs() >= raw_tie_tolerance
    signs = raw_difference.sign()
    score_difference = scores[:, left] - scores[:, right]
    terms = F.softplus(-signs * score_difference)
    informative_per_snapshot = informative.sum(dim=1)
    informative_snapshots = informative_per_snapshot > 0
    if bool(informative_snapshots.any()):
        masked_sum = (terms * informative.to(dtype=terms.dtype)).sum(dim=1)
        pairwise = (
            masked_sum[informative_snapshots]
            / informative_per_snapshot[informative_snapshots].to(dtype=terms.dtype)
        ).mean()
    else:
        pairwise = scores.sum() * 0.0
    huber = F.smooth_l1_loss(scores, normalized_labels, beta=huber_beta)
    total = pairwise_weight * pairwise + huber_weight * huber
    return RankingLoss(
        total=total,
        pairwise=pairwise,
        huber=huber,
        informative_pair_count=int(informative.sum().item()),
        tie_pair_count=int((~informative).sum().item()),
    )


@dataclass(frozen=True)
class LossRecord:
    total: float
    pairwise: float
    huber: float
    informative_pair_count: int
    tie_pair_count: int

    @classmethod
    def from_loss(cls, loss: RankingLoss) -> "LossRecord":
        return cls(
            total=float(loss.total.detach().cpu()),
            pairwise=float(loss.pairwise.detach().cpu()),
            huber=float(loss.huber.detach().cpu()),
            informative_pair_count=int(loss.informative_pair_count),
            tie_pair_count=int(loss.tie_pair_count),
        )

    def as_mapping(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "pairwise": self.pairwise,
            "huber": self.huber,
            "informative_pair_count": self.informative_pair_count,
            "tie_pair_count": self.tie_pair_count,
        }


@dataclass(frozen=True)
class DirectSeedResult:
    seed: int
    initial_loss: LossRecord
    final_loss: LossRecord
    training_scores: np.ndarray
    validation_scores: np.ndarray
    state_dict: Mapping[str, Tensor]
    state_sha256: str


@dataclass(frozen=True)
class ResidualSeedResult:
    seed: int
    initial_loss: LossRecord
    final_loss: LossRecord
    initial_delta_max_abs: float
    training_scores: np.ndarray
    validation_scores: np.ndarray
    training_delta: np.ndarray
    validation_delta: np.ndarray
    state_dict: Mapping[str, Tensor]
    state_sha256: str


@dataclass(frozen=True)
class ExploratoryActualReturnResult:
    direct_training_scores: np.ndarray
    direct_validation_scores: np.ndarray
    nested_training_scores: np.ndarray
    nested_validation_scores: np.ndarray
    direct_seed_results: tuple[DirectSeedResult, ...]
    residual_seed_results: tuple[ResidualSeedResult, ...]
    direct_parameter_count: int
    residual_parameter_count: int
    nested_total_parameter_count: int
    direct_input_scaler: DirectInputScaler
    future_input_scaler: FutureInputScaler
    label_normalizer: TaskLabelNormalizer
    raw_tie_tolerance: float
    target_name: str = "actual_return[:, :, -1]"


@dataclass(frozen=True)
class ExploratoryActualReturnRun:
    report: Mapping[str, Any]
    result: ExploratoryActualReturnResult


@dataclass(frozen=True)
class EpisodeCrossFitAssignment:
    """Fixed episode-level fold assignment over the formal training split."""

    fold_by_sample: np.ndarray
    episode_fold_by_task: Mapping[str, Mapping[str, int]]
    folds: int

    def validate(self, bundle: V3GateBundle) -> None:
        values = np.asarray(self.fold_by_sample)
        if values.shape != (bundle.axes.sample_count,):
            raise ValueError("cross-fit sample assignment shape changed")
        if not np.all(values[bundle.train_mask] >= 0):
            raise ValueError("every training sample must belong to one OOF fold")
        if not np.all(values[bundle.validation_mask] == -1):
            raise ValueError("validation samples must not enter cross-fitting")
        for task_index, task_name in enumerate(TASK_NAMES):
            rows = np.flatnonzero(
                bundle.train_mask
                & (np.asarray(bundle.audit_arrays["task_index"]) == task_index)
            )
            episodes_by_fold: dict[int, set[str]] = {
                fold: set() for fold in range(self.folds)
            }
            seen_episode_fold: dict[str, int] = {}
            for sample in rows:
                episode = str(
                    bundle.index_rows[int(sample)]["physical_episode_sha256"]
                )
                fold = int(values[sample])
                if fold < 0 or fold >= self.folds:
                    raise ValueError("training sample has an invalid OOF fold")
                previous = seen_episode_fold.setdefault(episode, fold)
                if previous != fold:
                    raise ValueError("one physical episode crosses OOF folds")
                episodes_by_fold[fold].add(episode)
            expected = CROSS_FIT_EPISODES_PER_TASK_PER_FOLD
            if any(len(episodes) != expected for episodes in episodes_by_fold.values()):
                raise ValueError(
                    f"{task_name} must contribute exactly {expected} episodes "
                    "to every OOF fold"
                )


def build_episode_crossfit_assignment(
    bundle: V3GateBundle,
    *,
    folds: int = CROSS_FIT_FOLDS,
) -> EpisodeCrossFitAssignment:
    """Assign SHA-sorted training episodes by ``rank % folds`` within task."""

    if folds != CROSS_FIT_FOLDS:
        raise ValueError(f"exploratory cross-fitting is fixed at {CROSS_FIT_FOLDS} folds")
    fold_by_sample = np.full(bundle.axes.sample_count, -1, dtype=np.int8)
    task_index = np.asarray(bundle.audit_arrays["task_index"], dtype=np.int64)
    episode_fold_by_task: dict[str, dict[str, int]] = {}
    expected_episodes = folds * CROSS_FIT_EPISODES_PER_TASK_PER_FOLD
    for task_value, task_name in enumerate(TASK_NAMES):
        rows = np.flatnonzero(bundle.train_mask & (task_index == task_value))
        episodes = sorted(
            {
                str(bundle.index_rows[int(sample)]["physical_episode_sha256"])
                for sample in rows
            }
        )
        if len(episodes) != expected_episodes:
            raise ValueError(
                f"{task_name} needs exactly {expected_episodes} training episodes "
                "for fixed four-fold cross-fitting"
            )
        mapping = {episode: rank % folds for rank, episode in enumerate(episodes)}
        episode_fold_by_task[task_name] = mapping
        for sample in rows:
            episode = str(
                bundle.index_rows[int(sample)]["physical_episode_sha256"]
            )
            fold_by_sample[sample] = mapping[episode]
    fold_by_sample.setflags(write=False)
    result = EpisodeCrossFitAssignment(
        fold_by_sample=fold_by_sample,
        episode_fold_by_task=episode_fold_by_task,
        folds=folds,
    )
    result.validate(bundle)
    return result


@dataclass(frozen=True)
class CrossFitFoldResult:
    fold_index: int
    training_sample_indices: np.ndarray
    heldout_sample_indices: np.ndarray
    training_episode_sha256_by_task: Mapping[str, tuple[str, ...]]
    heldout_episode_sha256_by_task: Mapping[str, tuple[str, ...]]
    direct_seed_results: tuple[DirectSeedResult, ...]
    heldout_scores: np.ndarray
    input_scaler: DirectInputScaler
    label_normalizer: TaskLabelNormalizer


@dataclass(frozen=True)
class ExploratoryCrossFitActualReturnResult:
    oof_direct_training_scores: np.ndarray
    oof_coverage: np.ndarray
    full_direct_training_scores: np.ndarray
    full_direct_validation_scores: np.ndarray
    nested_training_scores: np.ndarray
    nested_validation_scores: np.ndarray
    full_direct_seed_results: tuple[DirectSeedResult, ...]
    fold_results: tuple[CrossFitFoldResult, ...]
    residual_seed_results: tuple[ResidualSeedResult, ...]
    direct_parameter_count: int
    residual_parameter_count: int
    nested_total_parameter_count: int
    direct_input_scaler: DirectInputScaler
    future_input_scaler: FutureInputScaler
    label_normalizer: TaskLabelNormalizer
    assignment: EpisodeCrossFitAssignment
    raw_tie_tolerance: float
    target_name: str = "actual_return[:, :, -1]"


@dataclass(frozen=True)
class ExploratoryCrossFitActualReturnRun:
    report: Mapping[str, Any]
    result: ExploratoryCrossFitActualReturnResult


def _loss(
    scores: Tensor,
    normalized_target: Tensor,
    raw_target: Tensor,
    *,
    raw_tie_tolerance: float,
    pairwise_weight: float,
    huber_weight: float,
    huber_beta: float,
) -> RankingLoss:
    return actual_return_ranking_loss(
        scores,
        normalized_target,
        raw_target,
        raw_tie_tolerance=raw_tie_tolerance,
        pairwise_weight=pairwise_weight,
        huber_weight=huber_weight,
        huber_beta=huber_beta,
    )


def _train_direct_seed_models(
    train_inputs: DirectRankerInputs,
    evaluation_inputs: DirectRankerInputs,
    normalized_labels: np.ndarray,
    raw_labels: np.ndarray,
    *,
    seeds: Sequence[int],
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    hidden_dim: int,
    action_hidden_dim: int,
    pairwise_weight: float,
    huber_weight: float,
    huber_beta: float,
    raw_tie_tolerance: float,
    device: str,
) -> tuple[tuple[DirectSeedResult, ...], int]:
    """Train identical Direct models on one supplied train/evaluation split."""

    seed_tuple = _validate_seeds(seeds, label="direct_seeds")
    train_tensors = _direct_tensors(train_inputs, device)
    evaluation_tensors = _direct_tensors(evaluation_inputs, device)
    normalized_target = torch.from_numpy(
        np.asarray(normalized_labels, dtype=np.float32).copy()
    ).to(device)
    raw_target = torch.from_numpy(
        np.asarray(raw_labels, dtype=np.float32).copy()
    ).to(device)
    results: list[DirectSeedResult] = []
    parameter_count: int | None = None
    for seed in seed_tuple:
        _seed_everything(seed)
        model = PrivilegedDirectRanker(
            hidden_dim=hidden_dim, action_hidden_dim=action_hidden_dim
        ).to(device=device)
        count = _parameter_count(model)
        if parameter_count is None:
            parameter_count = count
        elif parameter_count != count:
            raise RuntimeError("Direct parameter count changed across seeds")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        model.train()
        with torch.no_grad():
            initial = _loss(
                model(*train_tensors),
                normalized_target,
                raw_target,
                raw_tie_tolerance=raw_tie_tolerance,
                pairwise_weight=pairwise_weight,
                huber_weight=huber_weight,
                huber_beta=huber_beta,
            )
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            score = model(*train_tensors)
            loss = _loss(
                score,
                normalized_target,
                raw_target,
                raw_tie_tolerance=raw_tie_tolerance,
                pairwise_weight=pairwise_weight,
                huber_weight=huber_weight,
                huber_beta=huber_beta,
            )
            loss.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            training_score = model(*train_tensors)
            evaluation_score = model(*evaluation_tensors)
            final = _loss(
                training_score,
                normalized_target,
                raw_target,
                raw_tie_tolerance=raw_tie_tolerance,
                pairwise_weight=pairwise_weight,
                huber_weight=huber_weight,
                huber_beta=huber_beta,
            )
        results.append(
            DirectSeedResult(
                seed=seed,
                initial_loss=LossRecord.from_loss(initial),
                final_loss=LossRecord.from_loss(final),
                training_scores=_readonly(
                    training_score.detach().cpu().numpy(),
                    dtype=np.dtype(np.float32),
                ),
                validation_scores=_readonly(
                    evaluation_score.detach().cpu().numpy(),
                    dtype=np.dtype(np.float32),
                ),
                state_dict=_state_dict_copy(model),
                state_sha256=scorer_state_sha256(model),
            )
        )
    return tuple(results), int(parameter_count or 0)


def _train_residual_seed_models(
    train_inputs: FutureResidualInputs,
    validation_inputs: FutureResidualInputs,
    frozen_training_scores: np.ndarray,
    frozen_validation_scores: np.ndarray,
    normalized_labels: np.ndarray,
    raw_labels: np.ndarray,
    *,
    seeds: Sequence[int],
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    hidden_dim: int,
    trajectory_hidden_dim: int,
    pairwise_weight: float,
    huber_weight: float,
    huber_beta: float,
    raw_tie_tolerance: float,
    device: str,
) -> tuple[tuple[ResidualSeedResult, ...], int]:
    """Train zero-initialized deltas over immutable supplied base scores."""

    seed_tuple = _validate_seeds(seeds, label="residual_seeds")
    train_tensors = _future_tensors(train_inputs, device)
    validation_tensors = _future_tensors(validation_inputs, device)
    frozen_train = torch.from_numpy(
        np.asarray(frozen_training_scores, dtype=np.float32).copy()
    ).to(device)
    frozen_validation = torch.from_numpy(
        np.asarray(frozen_validation_scores, dtype=np.float32).copy()
    ).to(device)
    frozen_train.requires_grad_(False)
    frozen_validation.requires_grad_(False)
    normalized_target = torch.from_numpy(
        np.asarray(normalized_labels, dtype=np.float32).copy()
    ).to(device)
    raw_target = torch.from_numpy(
        np.asarray(raw_labels, dtype=np.float32).copy()
    ).to(device)
    results: list[ResidualSeedResult] = []
    parameter_count: int | None = None
    for seed in seed_tuple:
        _seed_everything(seed)
        model = FutureStateResidualRanker(
            hidden_dim=hidden_dim, trajectory_hidden_dim=trajectory_hidden_dim
        ).to(device=device)
        count = _parameter_count(model)
        if parameter_count is None:
            parameter_count = count
        elif parameter_count != count:
            raise RuntimeError("residual parameter count changed across seeds")
        model.eval()
        with torch.no_grad():
            initial_delta = model.delta(*train_tensors)
            initial_score = model(frozen_train, *train_tensors)
            initial_validation_score = model(
                frozen_validation, *validation_tensors
            )
            initial = _loss(
                initial_score,
                normalized_target,
                raw_target,
                raw_tie_tolerance=raw_tie_tolerance,
                pairwise_weight=pairwise_weight,
                huber_weight=huber_weight,
                huber_beta=huber_beta,
            )
        initial_delta_max_abs = float(initial_delta.abs().max().cpu())
        if initial_delta_max_abs != 0.0:
            raise RuntimeError("future residual is not exactly zero initialized")
        if not torch.equal(initial_score, frozen_train):
            raise RuntimeError("nested score does not equal frozen Direct at init")
        if not torch.equal(initial_validation_score, frozen_validation):
            raise RuntimeError(
                "nested validation score does not equal full Direct at init"
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            score = model(frozen_train, *train_tensors)
            loss = _loss(
                score,
                normalized_target,
                raw_target,
                raw_tie_tolerance=raw_tie_tolerance,
                pairwise_weight=pairwise_weight,
                huber_weight=huber_weight,
                huber_beta=huber_beta,
            )
            loss.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            training_delta = model.delta(*train_tensors)
            validation_delta = model.delta(*validation_tensors)
            training_score = frozen_train + training_delta
            validation_score = frozen_validation + validation_delta
            final = _loss(
                training_score,
                normalized_target,
                raw_target,
                raw_tie_tolerance=raw_tie_tolerance,
                pairwise_weight=pairwise_weight,
                huber_weight=huber_weight,
                huber_beta=huber_beta,
            )
        results.append(
            ResidualSeedResult(
                seed=seed,
                initial_loss=LossRecord.from_loss(initial),
                final_loss=LossRecord.from_loss(final),
                initial_delta_max_abs=initial_delta_max_abs,
                training_scores=_readonly(
                    training_score.detach().cpu().numpy(),
                    dtype=np.dtype(np.float32),
                ),
                validation_scores=_readonly(
                    validation_score.detach().cpu().numpy(),
                    dtype=np.dtype(np.float32),
                ),
                training_delta=_readonly(
                    training_delta.detach().cpu().numpy(),
                    dtype=np.dtype(np.float32),
                ),
                validation_delta=_readonly(
                    validation_delta.detach().cpu().numpy(),
                    dtype=np.dtype(np.float32),
                ),
                state_dict=_state_dict_copy(model),
                state_sha256=scorer_state_sha256(model),
            )
        )
    return tuple(results), int(parameter_count or 0)


def train_actual_return_exploratory(
    bundle: V3GateBundle,
    *,
    direct_seeds: Sequence[int] = DEFAULT_DIRECT_SEEDS,
    residual_seeds: Sequence[int] = DEFAULT_RESIDUAL_SEEDS,
    direct_steps: int = 2000,
    residual_steps: int = 2000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    direct_hidden_dim: int = 64,
    direct_action_hidden_dim: int = 32,
    residual_hidden_dim: int = 64,
    residual_trajectory_hidden_dim: int = 48,
    pairwise_weight: float = 1.0,
    huber_weight: float = 0.1,
    huber_beta: float = 1.0,
    raw_tie_tolerance: float = DEFAULT_RAW_RETURN_TIE_TOLERANCE,
    device: str = "cpu",
) -> ExploratoryActualReturnResult:
    """Train Direct, freeze its ensemble score, then train future deltas."""

    direct_seed_tuple = _validate_seeds(direct_seeds, label="direct_seeds")
    residual_seed_tuple = _validate_seeds(residual_seeds, label="residual_seeds")
    if direct_steps <= 0 or residual_steps <= 0:
        raise ValueError("training steps must be positive")
    if learning_rate <= 0.0 or gradient_clip_norm <= 0.0:
        raise ValueError("optimization constants must be positive")
    if not np.isfinite(raw_tie_tolerance) or raw_tie_tolerance < 0.0:
        raise ValueError("raw_tie_tolerance must be finite and non-negative")

    all_direct = direct_ranker_inputs(bundle)
    train_direct = _select_direct_inputs(all_direct, bundle.train_mask)
    validation_direct = _select_direct_inputs(all_direct, bundle.validation_mask)
    direct_scaler = fit_direct_input_scaler(train_direct)
    train_direct = direct_scaler.transform(train_direct)
    validation_direct = direct_scaler.transform(validation_direct)
    train_direct_tensors = _direct_tensors(train_direct, device)
    validation_direct_tensors = _direct_tensors(validation_direct, device)

    all_future = future_residual_inputs(bundle)
    train_future = _select_future_inputs(all_future, bundle.train_mask)
    validation_future = _select_future_inputs(all_future, bundle.validation_mask)
    future_scaler = fit_future_input_scaler(train_future)
    train_future = future_scaler.transform(train_future)
    validation_future = future_scaler.transform(validation_future)
    train_future_tensors = _future_tensors(train_future, device)
    validation_future_tensors = _future_tensors(validation_future, device)

    train_task = np.asarray(bundle.audit_arrays["task_index"])[bundle.train_mask]
    raw_train_labels = np.asarray(bundle.terminal_return, dtype=np.float32)[
        bundle.train_mask
    ]
    label_normalizer = fit_task_label_normalizer(raw_train_labels, train_task)
    normalized_train_labels = label_normalizer.transform(raw_train_labels, train_task)
    normalized_target = torch.from_numpy(normalized_train_labels.copy()).to(device)
    raw_target = torch.from_numpy(raw_train_labels.copy()).to(device)

    direct_results: list[DirectSeedResult] = []
    direct_parameter_count: int | None = None
    rng_before = _capture_rng_state()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        for seed in direct_seed_tuple:
            _seed_everything(seed)
            model = PrivilegedDirectRanker(
                hidden_dim=direct_hidden_dim,
                action_hidden_dim=direct_action_hidden_dim,
            ).to(device=device)
            count = _parameter_count(model)
            if direct_parameter_count is None:
                direct_parameter_count = count
            elif count != direct_parameter_count:
                raise RuntimeError("Direct parameter count changed across seeds")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            model.train()
            with torch.no_grad():
                initial = _loss(
                    model(*train_direct_tensors),
                    normalized_target,
                    raw_target,
                    raw_tie_tolerance=raw_tie_tolerance,
                    pairwise_weight=pairwise_weight,
                    huber_weight=huber_weight,
                    huber_beta=huber_beta,
                )
            for _ in range(direct_steps):
                optimizer.zero_grad(set_to_none=True)
                score = model(*train_direct_tensors)
                loss = _loss(
                    score,
                    normalized_target,
                    raw_target,
                    raw_tie_tolerance=raw_tie_tolerance,
                    pairwise_weight=pairwise_weight,
                    huber_weight=huber_weight,
                    huber_beta=huber_beta,
                )
                loss.total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_score = model(*train_direct_tensors)
                validation_score = model(*validation_direct_tensors)
                final = _loss(
                    train_score,
                    normalized_target,
                    raw_target,
                    raw_tie_tolerance=raw_tie_tolerance,
                    pairwise_weight=pairwise_weight,
                    huber_weight=huber_weight,
                    huber_beta=huber_beta,
                )
            direct_results.append(
                DirectSeedResult(
                    seed=seed,
                    initial_loss=LossRecord.from_loss(initial),
                    final_loss=LossRecord.from_loss(final),
                    training_scores=_readonly(
                        train_score.detach().cpu().numpy(),
                        dtype=np.dtype(np.float32),
                    ),
                    validation_scores=_readonly(
                        validation_score.detach().cpu().numpy(),
                        dtype=np.dtype(np.float32),
                    ),
                    state_dict=_state_dict_copy(model),
                    state_sha256=scorer_state_sha256(model),
                )
            )

        direct_train_ensemble = np.mean(
            np.stack([value.training_scores for value in direct_results]),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        direct_validation_ensemble = np.mean(
            np.stack([value.validation_scores for value in direct_results]),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        frozen_train_score = torch.from_numpy(direct_train_ensemble.copy()).to(device)
        frozen_validation_score = torch.from_numpy(
            direct_validation_ensemble.copy()
        ).to(device)
        frozen_train_score.requires_grad_(False)
        frozen_validation_score.requires_grad_(False)

        residual_results: list[ResidualSeedResult] = []
        residual_parameter_count: int | None = None
        for seed in residual_seed_tuple:
            _seed_everything(seed)
            model = FutureStateResidualRanker(
                hidden_dim=residual_hidden_dim,
                trajectory_hidden_dim=residual_trajectory_hidden_dim,
            ).to(device=device)
            count = _parameter_count(model)
            if residual_parameter_count is None:
                residual_parameter_count = count
            elif count != residual_parameter_count:
                raise RuntimeError("residual parameter count changed across seeds")
            model.eval()
            with torch.no_grad():
                initial_delta = model.delta(*train_future_tensors)
                initial_score = model(frozen_train_score, *train_future_tensors)
                initial = _loss(
                    initial_score,
                    normalized_target,
                    raw_target,
                    raw_tie_tolerance=raw_tie_tolerance,
                    pairwise_weight=pairwise_weight,
                    huber_weight=huber_weight,
                    huber_beta=huber_beta,
                )
            initial_delta_max_abs = float(initial_delta.abs().max().cpu())
            if initial_delta_max_abs != 0.0:
                raise RuntimeError("future residual is not exactly zero initialized")
            if not torch.equal(initial_score, frozen_train_score):
                raise RuntimeError("nested score does not equal frozen Direct at init")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            model.train()
            for _ in range(residual_steps):
                optimizer.zero_grad(set_to_none=True)
                score = model(frozen_train_score, *train_future_tensors)
                loss = _loss(
                    score,
                    normalized_target,
                    raw_target,
                    raw_tie_tolerance=raw_tie_tolerance,
                    pairwise_weight=pairwise_weight,
                    huber_weight=huber_weight,
                    huber_beta=huber_beta,
                )
                loss.total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_delta = model.delta(*train_future_tensors)
                validation_delta = model.delta(*validation_future_tensors)
                train_score = frozen_train_score + train_delta
                validation_score = frozen_validation_score + validation_delta
                final = _loss(
                    train_score,
                    normalized_target,
                    raw_target,
                    raw_tie_tolerance=raw_tie_tolerance,
                    pairwise_weight=pairwise_weight,
                    huber_weight=huber_weight,
                    huber_beta=huber_beta,
                )
            residual_results.append(
                ResidualSeedResult(
                    seed=seed,
                    initial_loss=LossRecord.from_loss(initial),
                    final_loss=LossRecord.from_loss(final),
                    initial_delta_max_abs=initial_delta_max_abs,
                    training_scores=_readonly(
                        train_score.detach().cpu().numpy(),
                        dtype=np.dtype(np.float32),
                    ),
                    validation_scores=_readonly(
                        validation_score.detach().cpu().numpy(),
                        dtype=np.dtype(np.float32),
                    ),
                    training_delta=_readonly(
                        train_delta.detach().cpu().numpy(),
                        dtype=np.dtype(np.float32),
                    ),
                    validation_delta=_readonly(
                        validation_delta.detach().cpu().numpy(),
                        dtype=np.dtype(np.float32),
                    ),
                    state_dict=_state_dict_copy(model),
                    state_sha256=scorer_state_sha256(model),
                )
            )
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_only_before
        )
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
        _restore_rng_state(rng_before)

    nested_train_ensemble = np.mean(
        np.stack([value.training_scores for value in residual_results]),
        axis=0,
        dtype=np.float64,
    )
    nested_validation_ensemble = np.mean(
        np.stack([value.validation_scores for value in residual_results]),
        axis=0,
        dtype=np.float64,
    )
    direct_count = int(direct_parameter_count or 0)
    residual_count = int(residual_parameter_count or 0)
    return ExploratoryActualReturnResult(
        direct_training_scores=_readonly(
            direct_train_ensemble, dtype=np.dtype(np.float32)
        ),
        direct_validation_scores=_readonly(
            direct_validation_ensemble, dtype=np.dtype(np.float32)
        ),
        nested_training_scores=_readonly(
            nested_train_ensemble, dtype=np.dtype(np.float32)
        ),
        nested_validation_scores=_readonly(
            nested_validation_ensemble, dtype=np.dtype(np.float32)
        ),
        direct_seed_results=tuple(direct_results),
        residual_seed_results=tuple(residual_results),
        direct_parameter_count=direct_count,
        residual_parameter_count=residual_count,
        nested_total_parameter_count=direct_count + residual_count,
        direct_input_scaler=direct_scaler,
        future_input_scaler=future_scaler,
        label_normalizer=label_normalizer,
        raw_tie_tolerance=float(raw_tie_tolerance),
    )


def _convert_normalized_scores(
    scores: np.ndarray,
    task_index: np.ndarray,
    *,
    source: TaskLabelNormalizer,
    destination: TaskLabelNormalizer,
) -> np.ndarray:
    """Move model scores between per-task normalized return coordinates."""

    values = np.asarray(scores, dtype=np.float64)
    task = np.asarray(task_index, dtype=np.int64)
    if values.ndim != 2 or task.shape != (values.shape[0],):
        raise ValueError("score conversion expects [N,C] scores and [N] tasks")
    raw = values * source.scale[task, None] + source.mean[task, None]
    converted = (
        raw - destination.mean[task, None]
    ) / destination.scale[task, None]
    return _readonly(converted, dtype=np.dtype(np.float32))


def _convert_direct_seed_result_scores(
    result: DirectSeedResult,
    training_task_index: np.ndarray,
    evaluation_task_index: np.ndarray,
    *,
    source: TaskLabelNormalizer,
    destination: TaskLabelNormalizer,
) -> DirectSeedResult:
    return DirectSeedResult(
        seed=result.seed,
        initial_loss=result.initial_loss,
        final_loss=result.final_loss,
        training_scores=_convert_normalized_scores(
            result.training_scores,
            training_task_index,
            source=source,
            destination=destination,
        ),
        validation_scores=_convert_normalized_scores(
            result.validation_scores,
            evaluation_task_index,
            source=source,
            destination=destination,
        ),
        state_dict=result.state_dict,
        state_sha256=result.state_sha256,
    )


def _episode_sha256_by_task(
    bundle: V3GateBundle, sample_indices: np.ndarray
) -> Mapping[str, tuple[str, ...]]:
    task_index = np.asarray(bundle.audit_arrays["task_index"], dtype=np.int64)
    result: dict[str, tuple[str, ...]] = {}
    for task_value, task_name in enumerate(TASK_NAMES):
        rows = [
            int(sample)
            for sample in np.asarray(sample_indices, dtype=np.int64)
            if int(task_index[sample]) == task_value
        ]
        result[task_name] = tuple(
            sorted(
                {
                    str(bundle.index_rows[sample]["physical_episode_sha256"])
                    for sample in rows
                }
            )
        )
    return result


def train_actual_return_crossfit_exploratory(
    bundle: V3GateBundle,
    *,
    direct_seeds: Sequence[int] = DEFAULT_DIRECT_SEEDS,
    residual_seeds: Sequence[int] = DEFAULT_RESIDUAL_SEEDS,
    direct_steps: int = 2000,
    residual_steps: int = 2000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    direct_hidden_dim: int = 64,
    direct_action_hidden_dim: int = 32,
    residual_hidden_dim: int = 64,
    residual_trajectory_hidden_dim: int = 48,
    pairwise_weight: float = 1.0,
    huber_weight: float = 0.1,
    huber_beta: float = 1.0,
    raw_tie_tolerance: float = DEFAULT_RAW_RETURN_TIE_TOLERANCE,
    device: str = "cpu",
) -> ExploratoryCrossFitActualReturnResult:
    """Train a four-fold OOF Direct base before fitting the future residual."""

    direct_seed_tuple = _validate_seeds(direct_seeds, label="direct_seeds")
    residual_seed_tuple = _validate_seeds(residual_seeds, label="residual_seeds")
    if direct_steps <= 0 or residual_steps <= 0:
        raise ValueError("training steps must be positive")
    if learning_rate <= 0.0 or gradient_clip_norm <= 0.0:
        raise ValueError("optimization constants must be positive")
    if weight_decay < 0.0 or not np.isfinite(weight_decay):
        raise ValueError("weight_decay must be finite and non-negative")
    if not np.isfinite(raw_tie_tolerance) or raw_tie_tolerance < 0.0:
        raise ValueError("raw_tie_tolerance must be finite and non-negative")

    assignment = build_episode_crossfit_assignment(bundle)
    train_indices = np.flatnonzero(bundle.train_mask)
    validation_indices = np.flatnonzero(bundle.validation_mask)
    train_position = np.full(bundle.axes.sample_count, -1, dtype=np.int64)
    train_position[train_indices] = np.arange(train_indices.size)
    task_index = np.asarray(bundle.audit_arrays["task_index"], dtype=np.int64)
    raw_train_labels = np.asarray(bundle.terminal_return, dtype=np.float32)[
        train_indices
    ]
    global_label_normalizer = fit_task_label_normalizer(
        raw_train_labels, task_index[train_indices]
    )
    global_normalized_labels = global_label_normalizer.transform(
        raw_train_labels, task_index[train_indices]
    )

    all_direct_raw = direct_ranker_inputs(bundle)
    full_train_direct_raw = _select_direct_inputs(all_direct_raw, train_indices)
    direct_scaler = fit_direct_input_scaler(full_train_direct_raw)
    all_direct_global = direct_scaler.transform(all_direct_raw)
    full_train_direct = _select_direct_inputs(all_direct_global, train_indices)
    validation_direct = _select_direct_inputs(
        all_direct_global, validation_indices
    )

    all_future_raw = future_residual_inputs(bundle)
    full_train_future_raw = _select_future_inputs(all_future_raw, train_indices)
    future_scaler = fit_future_input_scaler(full_train_future_raw)
    all_future = future_scaler.transform(all_future_raw)
    full_train_future = _select_future_inputs(all_future, train_indices)
    validation_future = _select_future_inputs(all_future, validation_indices)

    rng_before = _capture_rng_state()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        full_direct_results, direct_parameter_count = _train_direct_seed_models(
            full_train_direct,
            validation_direct,
            global_normalized_labels,
            raw_train_labels,
            seeds=direct_seed_tuple,
            steps=direct_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            hidden_dim=direct_hidden_dim,
            action_hidden_dim=direct_action_hidden_dim,
            pairwise_weight=pairwise_weight,
            huber_weight=huber_weight,
            huber_beta=huber_beta,
            raw_tie_tolerance=raw_tie_tolerance,
            device=device,
        )
        full_direct_train_scores = np.mean(
            np.stack([value.training_scores for value in full_direct_results]),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        full_direct_validation_scores = np.mean(
            np.stack([value.validation_scores for value in full_direct_results]),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        oof_scores = np.full(
            (train_indices.size, CANDIDATE_COUNT), np.nan, dtype=np.float32
        )
        oof_coverage = np.zeros(train_indices.size, dtype=np.int8)
        fold_results: list[CrossFitFoldResult] = []
        for fold in range(CROSS_FIT_FOLDS):
            heldout_global = train_indices[
                assignment.fold_by_sample[train_indices] == fold
            ]
            fold_train_global = train_indices[
                assignment.fold_by_sample[train_indices] != fold
            ]
            heldout_positions = train_position[heldout_global]
            fold_train_positions = train_position[fold_train_global]
            if np.any(heldout_positions < 0) or np.any(fold_train_positions < 0):
                raise RuntimeError("OOF fold contains a non-training sample")

            fold_train_raw_inputs = _select_direct_inputs(
                all_direct_raw, fold_train_global
            )
            fold_scaler = fit_direct_input_scaler(fold_train_raw_inputs)
            fold_train_inputs = fold_scaler.transform(fold_train_raw_inputs)
            fold_heldout_inputs = fold_scaler.transform(
                _select_direct_inputs(all_direct_raw, heldout_global)
            )
            fold_raw_labels = raw_train_labels[fold_train_positions]
            fold_train_tasks = task_index[fold_train_global]
            fold_heldout_tasks = task_index[heldout_global]
            fold_normalizer = fit_task_label_normalizer(
                fold_raw_labels, fold_train_tasks
            )
            fold_normalized_labels = fold_normalizer.transform(
                fold_raw_labels, fold_train_tasks
            )
            raw_fold_results, fold_parameter_count = _train_direct_seed_models(
                fold_train_inputs,
                fold_heldout_inputs,
                fold_normalized_labels,
                fold_raw_labels,
                seeds=direct_seed_tuple,
                steps=direct_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                gradient_clip_norm=gradient_clip_norm,
                hidden_dim=direct_hidden_dim,
                action_hidden_dim=direct_action_hidden_dim,
                pairwise_weight=pairwise_weight,
                huber_weight=huber_weight,
                huber_beta=huber_beta,
                raw_tie_tolerance=raw_tie_tolerance,
                device=device,
            )
            if fold_parameter_count != direct_parameter_count:
                raise RuntimeError("full and fold Direct parameter counts differ")
            converted_fold_results = tuple(
                _convert_direct_seed_result_scores(
                    value,
                    fold_train_tasks,
                    fold_heldout_tasks,
                    source=fold_normalizer,
                    destination=global_label_normalizer,
                )
                for value in raw_fold_results
            )
            heldout_scores = np.mean(
                np.stack(
                    [value.validation_scores for value in converted_fold_results]
                ),
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
            oof_scores[heldout_positions] = heldout_scores
            oof_coverage[heldout_positions] += 1
            fold_results.append(
                CrossFitFoldResult(
                    fold_index=fold,
                    training_sample_indices=_readonly(
                        fold_train_global, dtype=np.dtype(np.int64)
                    ),
                    heldout_sample_indices=_readonly(
                        heldout_global, dtype=np.dtype(np.int64)
                    ),
                    training_episode_sha256_by_task=_episode_sha256_by_task(
                        bundle, fold_train_global
                    ),
                    heldout_episode_sha256_by_task=_episode_sha256_by_task(
                        bundle, heldout_global
                    ),
                    direct_seed_results=converted_fold_results,
                    heldout_scores=_readonly(heldout_scores),
                    input_scaler=fold_scaler,
                    label_normalizer=fold_normalizer,
                )
            )
        if not np.all(oof_coverage == 1):
            raise RuntimeError("every training sample must receive exactly one OOF score")
        if not np.isfinite(oof_scores).all():
            raise RuntimeError("OOF Direct scores contain unfilled values")

        residual_results, residual_parameter_count = _train_residual_seed_models(
            full_train_future,
            validation_future,
            oof_scores,
            full_direct_validation_scores,
            global_normalized_labels,
            raw_train_labels,
            seeds=residual_seed_tuple,
            steps=residual_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            hidden_dim=residual_hidden_dim,
            trajectory_hidden_dim=residual_trajectory_hidden_dim,
            pairwise_weight=pairwise_weight,
            huber_weight=huber_weight,
            huber_beta=huber_beta,
            raw_tie_tolerance=raw_tie_tolerance,
            device=device,
        )
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_only_before
        )
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
        _restore_rng_state(rng_before)

    nested_train_scores = np.mean(
        np.stack([value.training_scores for value in residual_results]),
        axis=0,
        dtype=np.float64,
    )
    nested_validation_scores = np.mean(
        np.stack([value.validation_scores for value in residual_results]),
        axis=0,
        dtype=np.float64,
    )
    oof_coverage.setflags(write=False)
    return ExploratoryCrossFitActualReturnResult(
        oof_direct_training_scores=_readonly(oof_scores),
        oof_coverage=oof_coverage,
        full_direct_training_scores=_readonly(full_direct_train_scores),
        full_direct_validation_scores=_readonly(full_direct_validation_scores),
        nested_training_scores=_readonly(
            nested_train_scores, dtype=np.dtype(np.float32)
        ),
        nested_validation_scores=_readonly(
            nested_validation_scores, dtype=np.dtype(np.float32)
        ),
        full_direct_seed_results=full_direct_results,
        fold_results=tuple(fold_results),
        residual_seed_results=residual_results,
        direct_parameter_count=direct_parameter_count,
        residual_parameter_count=residual_parameter_count,
        nested_total_parameter_count=(
            direct_parameter_count + residual_parameter_count
        ),
        direct_input_scaler=direct_scaler,
        future_input_scaler=future_scaler,
        label_normalizer=global_label_normalizer,
        assignment=assignment,
        raw_tie_tolerance=float(raw_tie_tolerance),
    )


def evaluate_actual_return_scores(
    bundle: V3GateBundle,
    scores: np.ndarray,
    *,
    split: Literal["train", "validation"] = "validation",
    tie_tolerance: float = DEFAULT_RAW_RETURN_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Return descriptive task-macro ranking metrics without inference tests."""

    if split == "train":
        mask = bundle.train_mask
    elif split == "validation":
        mask = bundle.validation_mask
    else:
        raise ValueError("split must be 'train' or 'validation'")
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(bundle.terminal_return, dtype=np.float64)[mask]
    tasks = np.asarray(bundle.audit_arrays["task_index"], dtype=np.int64)[mask]
    if values.shape != labels.shape:
        raise ValueError("score matrix shape does not match selected split")
    per_task: dict[str, dict[str, float | int]] = {}
    task_spearman: list[float] = []
    task_regret: list[float] = []
    for task_index, task_name in enumerate(TASK_NAMES):
        selected = np.flatnonzero(tasks == task_index)
        if selected.size == 0:
            raise ValueError(f"selected split has no rows for {task_name}")
        metrics = [
            snapshot_ranking_metrics(
                values[index], labels[index], tie_tolerance=tie_tolerance
            )
            for index in selected
        ]
        spearman = float(np.mean([value.spearman for value in metrics]))
        regret = float(
            np.mean([value.normalized_top1_regret for value in metrics])
        )
        spreads = np.asarray([value.return_spread for value in metrics])
        task_spearman.append(spearman)
        task_regret.append(regret)
        per_task[task_name] = {
            "snapshot_count": int(selected.size),
            "mean_spearman": spearman,
            "mean_normalized_top1_regret": regret,
            "median_raw_return_spread": float(np.median(spreads)),
            "minimum_raw_return_spread": float(np.min(spreads)),
            "maximum_raw_return_spread": float(np.max(spreads)),
        }
    return {
        "split": split,
        "snapshot_count": int(values.shape[0]),
        "target": "actual_return[:, :, -1]",
        "task_macro_spearman": float(np.mean(task_spearman)),
        "task_macro_normalized_top1_regret": float(np.mean(task_regret)),
        "per_task": per_task,
    }


def _train_only_lookup_scores(
    bundle: V3GateBundle,
    *,
    keys: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a task-conditioned lookup on train rows and score validation rows."""

    labels = np.asarray(bundle.terminal_return, dtype=np.float64)
    tasks = np.asarray(bundle.audit_arrays["task_index"], dtype=np.int64)
    normalized_keys = [np.asarray(value) for value in keys]
    for value in normalized_keys:
        if value.shape != labels.shape:
            raise ValueError("lookup key arrays must have shape [N,8]")
    grouped: dict[tuple[int, ...], list[float]] = {}
    for sample in np.flatnonzero(bundle.train_mask):
        for candidate in range(CANDIDATE_COUNT):
            key = (int(tasks[sample]),) + tuple(
                int(value[sample, candidate]) for value in normalized_keys
            )
            grouped.setdefault(key, []).append(float(labels[sample, candidate]))
    lookup = {key: float(np.mean(values)) for key, values in grouped.items()}
    validation_indices = np.flatnonzero(bundle.validation_mask)
    scores = np.empty(
        (validation_indices.size, CANDIDATE_COUNT), dtype=np.float32
    )
    for output_index, sample in enumerate(validation_indices):
        for candidate in range(CANDIDATE_COUNT):
            key = (int(tasks[sample]),) + tuple(
                int(value[sample, candidate]) for value in normalized_keys
            )
            if key not in lookup:
                raise ValueError(f"validation lookup key was absent in train: {key}")
            scores[output_index, candidate] = lookup[key]
    return _readonly(scores), np.asarray(sorted(lookup), dtype=np.int64)


def actual_return_lookup_baselines(
    bundle: V3GateBundle,
) -> dict[str, dict[str, Any]]:
    """Return train-only construction priors evaluated on terminal return."""

    samples = bundle.axes.sample_count
    slots = np.broadcast_to(
        np.arange(CANDIDATE_COUNT, dtype=np.int64),
        (samples, CANDIDATE_COUNT),
    )
    try:
        axis = np.asarray(bundle.audit_arrays["candidate_axis"])
        sign = np.asarray(bundle.audit_arrays["candidate_sign"])
    except KeyError as error:
        raise KeyError("exploratory report requires candidate_axis/sign audits") from error
    slot_scores, slot_cells = _train_only_lookup_scores(bundle, keys=(slots,))
    axis_scores, axis_cells = _train_only_lookup_scores(
        bundle, keys=(axis, sign)
    )
    return {
        "anonymous_slot": {
            "scores": slot_scores,
            "lookup_cell_count": int(slot_cells.shape[0]),
            "fit_scope": "train_split_only",
            "metadata": "task_x_public_anonymous_slot",
        },
        "axis_sign": {
            "scores": axis_scores,
            "lookup_cell_count": int(axis_cells.shape[0]),
            "fit_scope": "train_split_only",
            "metadata": "task_x_hidden_generator_axis_sign",
        },
    }


def _actual_return_metric_rows(
    bundle: V3GateBundle,
    scores: np.ndarray,
    *,
    scorer: str,
    tie_tolerance: float,
) -> list[dict[str, Any]]:
    return metric_rows(
        bundle,
        scores,
        scorer=scorer,
        target_labels=bundle.terminal_return,
        target_name="H5_terminal_actual_return",
        tie_tolerance=tie_tolerance,
    )


def _descriptive_paired_comparison(
    bundle: V3GateBundle,
    better_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    better_name: str,
    baseline_name: str,
    tie_tolerance: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    better_rows = _actual_return_metric_rows(
        bundle, better_scores, scorer=better_name, tie_tolerance=tie_tolerance
    )
    baseline_rows = _actual_return_metric_rows(
        bundle, baseline_scores, scorer=baseline_name, tie_tolerance=tie_tolerance
    )
    spearman = paired_episode_within_task_bootstrap(
        better_rows,
        baseline_rows,
        metric="spearman",
        higher_is_better=True,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    regret = paired_episode_within_task_bootstrap(
        better_rows,
        baseline_rows,
        metric="normalized_top1_regret",
        higher_is_better=False,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 1,
    )
    jointly_positive = [
        task
        for task in TASK_NAMES
        if spearman["task_mean_improvement"][task] > 0.0
        and regret["task_mean_improvement"][task] > 0.0
    ]
    point_estimate_signal = bool(
        spearman["estimate"] > 0.0
        and regret["estimate"] > 0.0
        and len(jointly_positive) >= 3
    )
    return {
        "better": better_name,
        "baseline": baseline_name,
        "spearman": spearman,
        "normalized_top1_regret": regret,
        "jointly_positive_tasks": jointly_positive,
        "jointly_positive_task_count": len(jointly_positive),
        "point_estimate_signal": point_estimate_signal,
        "criteria": {
            "both_task_macro_point_estimates_positive": True,
            "minimum_jointly_positive_tasks": 3,
            "confidence_intervals_are_descriptive_only": True,
        },
    }


def _seed_report(
    bundle: V3GateBundle,
    values: Sequence[DirectSeedResult] | Sequence[ResidualSeedResult],
) -> list[dict[str, Any]]:
    result = []
    for value in values:
        row: dict[str, Any] = {
            "seed": int(value.seed),
            "initial_loss": value.initial_loss.as_mapping(),
            "final_loss": value.final_loss.as_mapping(),
            "state_sha256": str(value.state_sha256),
            "train_metrics": evaluate_actual_return_scores(
                bundle, value.training_scores, split="train"
            ),
            "validation_metrics": evaluate_actual_return_scores(
                bundle, value.validation_scores, split="validation"
            ),
        }
        if isinstance(value, ResidualSeedResult):
            row["initial_delta_max_abs"] = value.initial_delta_max_abs
        result.append(row)
    return result


def run_exploratory_actual_return(
    bundle: V3GateBundle,
    *,
    device: str,
    direct_steps: int = 2000,
    residual_steps: int = 2000,
) -> ExploratoryActualReturnRun:
    """Stable entry point for the isolated post-hoc actual-return run."""

    result = train_actual_return_exploratory(
        bundle,
        direct_seeds=DEFAULT_DIRECT_SEEDS,
        residual_seeds=DEFAULT_RESIDUAL_SEEDS,
        direct_steps=direct_steps,
        residual_steps=residual_steps,
        device=device,
    )
    direct_train = evaluate_actual_return_scores(
        bundle, result.direct_training_scores, split="train"
    )
    direct_validation = evaluate_actual_return_scores(
        bundle, result.direct_validation_scores, split="validation"
    )
    nested_train = evaluate_actual_return_scores(
        bundle, result.nested_training_scores, split="train"
    )
    nested_validation = evaluate_actual_return_scores(
        bundle, result.nested_validation_scores, split="validation"
    )
    lookup_baselines = actual_return_lookup_baselines(bundle)
    slot_scores = lookup_baselines["anonymous_slot"]["scores"]
    axis_scores = lookup_baselines["axis_sign"]["scores"]
    slot_validation = evaluate_actual_return_scores(
        bundle, slot_scores, split="validation"
    )
    axis_validation = evaluate_actual_return_scores(
        bundle, axis_scores, split="validation"
    )
    direct_vs_axis = _descriptive_paired_comparison(
        bundle,
        result.direct_validation_scores,
        axis_scores,
        better_name="direct",
        baseline_name="axis_sign",
        tie_tolerance=result.raw_tie_tolerance,
        bootstrap_samples=10_000,
        bootstrap_seed=9361,
    )
    nested_vs_direct = _descriptive_paired_comparison(
        bundle,
        result.nested_validation_scores,
        result.direct_validation_scores,
        better_name="direct_plus_state_oracle_residual",
        baseline_name="direct",
        tie_tolerance=result.raw_tie_tolerance,
        bootstrap_samples=10_000,
        bootstrap_seed=9371,
    )
    exploratory_signal = bool(
        direct_vs_axis["point_estimate_signal"]
        and nested_vs_direct["point_estimate_signal"]
    )
    scorers = {
        "anonymous_slot": {
            "fit_scope": lookup_baselines["anonymous_slot"]["fit_scope"],
            "lookup_cell_count": lookup_baselines["anonymous_slot"][
                "lookup_cell_count"
            ],
            "validation_metrics": slot_validation,
        },
        "axis_sign": {
            "fit_scope": lookup_baselines["axis_sign"]["fit_scope"],
            "lookup_cell_count": lookup_baselines["axis_sign"][
                "lookup_cell_count"
            ],
            "validation_metrics": axis_validation,
        },
        "direct": {
            "inputs": "current_state+candidate_action_raw+goal_world+task_onehot",
            "parameter_count_per_seed": result.direct_parameter_count,
            "ensemble_parameter_count": (
                result.direct_parameter_count * len(result.direct_seed_results)
            ),
            "seed_results": _seed_report(bundle, result.direct_seed_results),
            "train_metrics": direct_train,
            "validation_metrics": direct_validation,
        },
        "direct_plus_state_oracle_residual": {
            "formula": "stopgrad(frozen_direct_ensemble_score)+delta(future_state,goal,task)",
            "delta_parameter_count_per_seed": result.residual_parameter_count,
            "nested_parameter_count_per_seed": result.nested_total_parameter_count,
            "ensemble_parameter_count": (
                result.direct_parameter_count * len(result.direct_seed_results)
                + result.residual_parameter_count * len(result.residual_seed_results)
            ),
            "zero_initialized": True,
            "seed_results": _seed_report(bundle, result.residual_seed_results),
            "train_metrics": nested_train,
            "validation_metrics": nested_validation,
        },
    }
    report: dict[str, Any] = {
        "stage": "post_hoc_exploratory_actual_h5_return",
        "classification": "post_hoc_exploratory",
        "formal_v3_gate_status": "unchanged_P0_No-Go",
        "formal_artifact_mutated": False,
        "target": "actual_return[:, :, -1]",
        "physical_task_score_used": False,
        "raw_return_tie_tolerance": result.raw_tie_tolerance,
        "loss": {
            "primary": "pairwise_logistic_ranking",
            "auxiliary": "task_normalized_Huber",
            "pairwise_weight": 1.0,
            "huber_weight": 0.1,
            "huber_beta": 1.0,
        },
        "scorers": scorers,
        "comparisons": {
            "direct_vs_axis_sign": direct_vs_axis,
            "nested_vs_direct": nested_vs_direct,
        },
        "exploratory_signal": exploratory_signal,
        "exploratory_signal_definition": (
            "Direct exceeds axis/sign and nested exceeds Direct with positive "
            "paired task-macro Spearman and regret point estimates and at least "
            "three jointly positive tasks in each comparison."
        ),
        "descriptive_validation_delta": {
            "task_macro_spearman": (
                nested_validation["task_macro_spearman"]
                - direct_validation["task_macro_spearman"]
            ),
            "normalized_top1_regret_improvement": (
                direct_validation["task_macro_normalized_top1_regret"]
                - nested_validation["task_macro_normalized_top1_regret"]
            ),
        },
        "claim_boundary": (
            "Exposed-validation post-hoc diagnostic only. A positive residual "
            "delta is capacity-confounded and cannot establish that future "
            "information, LeWM, FCPG planning, or CEM is necessary."
        ),
    }
    return ExploratoryActualReturnRun(report=report, result=result)


def _crossfit_fold_report(value: CrossFitFoldResult) -> dict[str, Any]:
    return {
        "fold_index": value.fold_index,
        "training_sample_count": int(value.training_sample_indices.size),
        "heldout_sample_count": int(value.heldout_sample_indices.size),
        "training_episode_sha256_by_task": {
            task: list(episodes)
            for task, episodes in value.training_episode_sha256_by_task.items()
        },
        "heldout_episode_sha256_by_task": {
            task: list(episodes)
            for task, episodes in value.heldout_episode_sha256_by_task.items()
        },
        "preprocessing_fit_scope": "fold_training_episodes_only",
        "score_coordinate_conversion": (
            "fold_normalized_to_raw_return_to_global_train_normalized"
        ),
        "shared_label_coordinate_note": (
            "The final affine coordinate uses the all-training label normalizer, "
            "including this fold's held-out labels; it does not change rankings."
        ),
        "direct_seed_results": [
            {
                "seed": seed.seed,
                "initial_loss": seed.initial_loss.as_mapping(),
                "final_loss": seed.final_loss.as_mapping(),
                "state_sha256": seed.state_sha256,
            }
            for seed in value.direct_seed_results
        ],
    }


def run_exploratory_actual_return_crossfit(
    bundle: V3GateBundle,
    *,
    device: str,
    direct_steps: int = 2000,
    residual_steps: int = 2000,
) -> ExploratoryCrossFitActualReturnRun:
    """Stable fixed-four-fold entry point for the OOF residual diagnostic."""

    result = train_actual_return_crossfit_exploratory(
        bundle,
        direct_seeds=DEFAULT_DIRECT_SEEDS,
        residual_seeds=DEFAULT_RESIDUAL_SEEDS,
        direct_steps=direct_steps,
        residual_steps=residual_steps,
        device=device,
    )
    oof_direct_train = evaluate_actual_return_scores(
        bundle, result.oof_direct_training_scores, split="train"
    )
    full_direct_train = evaluate_actual_return_scores(
        bundle, result.full_direct_training_scores, split="train"
    )
    full_direct_validation = evaluate_actual_return_scores(
        bundle, result.full_direct_validation_scores, split="validation"
    )
    nested_train = evaluate_actual_return_scores(
        bundle, result.nested_training_scores, split="train"
    )
    nested_validation = evaluate_actual_return_scores(
        bundle, result.nested_validation_scores, split="validation"
    )
    lookup_baselines = actual_return_lookup_baselines(bundle)
    slot_scores = lookup_baselines["anonymous_slot"]["scores"]
    axis_scores = lookup_baselines["axis_sign"]["scores"]
    slot_validation = evaluate_actual_return_scores(
        bundle, slot_scores, split="validation"
    )
    axis_validation = evaluate_actual_return_scores(
        bundle, axis_scores, split="validation"
    )
    direct_vs_axis = _descriptive_paired_comparison(
        bundle,
        result.full_direct_validation_scores,
        axis_scores,
        better_name="full_direct",
        baseline_name="axis_sign",
        tie_tolerance=result.raw_tie_tolerance,
        bootstrap_samples=10_000,
        bootstrap_seed=9461,
    )
    nested_vs_direct = _descriptive_paired_comparison(
        bundle,
        result.nested_validation_scores,
        result.full_direct_validation_scores,
        better_name="crossfit_direct_plus_state_oracle_residual",
        baseline_name="full_direct",
        tie_tolerance=result.raw_tie_tolerance,
        bootstrap_samples=10_000,
        bootstrap_seed=9471,
    )
    exploratory_signal = bool(
        direct_vs_axis["point_estimate_signal"]
        and nested_vs_direct["point_estimate_signal"]
    )
    scorers = {
        "anonymous_slot": {
            "fit_scope": lookup_baselines["anonymous_slot"]["fit_scope"],
            "lookup_cell_count": lookup_baselines["anonymous_slot"][
                "lookup_cell_count"
            ],
            "validation_metrics": slot_validation,
        },
        "axis_sign": {
            "fit_scope": lookup_baselines["axis_sign"]["fit_scope"],
            "lookup_cell_count": lookup_baselines["axis_sign"][
                "lookup_cell_count"
            ],
            "validation_metrics": axis_validation,
        },
        "direct": {
            "inputs": "current_state+candidate_action_raw+goal_world+task_onehot",
            "parameter_count_per_seed": result.direct_parameter_count,
            "ensemble_parameter_count": (
                result.direct_parameter_count
                * len(result.full_direct_seed_results)
            ),
            "seed_results": _seed_report(
                bundle, result.full_direct_seed_results
            ),
            "oof_train_metrics": oof_direct_train,
            "full_fit_train_metrics": full_direct_train,
            "validation_metrics": full_direct_validation,
        },
        "direct_plus_state_oracle_residual": {
            "formula": "stopgrad(OOF_direct_train/full_direct_validation)+delta(future_state,goal,task)",
            "delta_parameter_count_per_seed": result.residual_parameter_count,
            "nested_parameter_count_per_seed": result.nested_total_parameter_count,
            "ensemble_parameter_count": (
                result.direct_parameter_count
                * len(result.full_direct_seed_results)
                + result.residual_parameter_count * len(result.residual_seed_results)
            ),
            "zero_initialized": True,
            "seed_results": _seed_report(bundle, result.residual_seed_results),
            "train_metrics_over_oof_base": nested_train,
            "validation_metrics_over_full_direct_base": nested_validation,
        },
    }
    report: dict[str, Any] = {
        "stage": "post_hoc_exploratory_actual_h5_return_crossfit",
        "classification": "post_hoc_exploratory",
        "formal_v3_gate_status": "unchanged_P0_No-Go",
        "formal_artifact_mutated": False,
        "target": "actual_return[:, :, -1]",
        "physical_task_score_used": False,
        "raw_return_tie_tolerance": result.raw_tie_tolerance,
        "loss": {
            "primary": "pairwise_logistic_ranking",
            "auxiliary": "task_normalized_Huber",
            "pairwise_weight": 1.0,
            "huber_weight": 0.1,
            "huber_beta": 1.0,
        },
        "cross_fit": {
            "folds": CROSS_FIT_FOLDS,
            "assignment": "per_task_sha256_sorted_episode_rank_mod_4",
            "episodes_per_task_per_heldout_fold": (
                CROSS_FIT_EPISODES_PER_TASK_PER_FOLD
            ),
            "oof_training_sample_count": int(result.oof_coverage.size),
            "oof_coverage_min": int(np.min(result.oof_coverage)),
            "oof_coverage_max": int(np.max(result.oof_coverage)),
            "every_training_sample_scored_once": bool(
                np.all(result.oof_coverage == 1)
            ),
            "fold_preprocessing": "fit_on_six_fold-training_episodes_per_task_only",
            "fold_score_conversion": (
                "fold_normalized_to_raw_return_to_global_train_normalized"
            ),
            "strictly_fold_blind_score_coordinate": False,
            "shared_score_coordinate": (
                "all-training task label normalizer used only after held-out "
                "prediction as an affine fold-to-global conversion; candidate "
                "rankings are invariant to this conversion"
            ),
            "fold_results": [
                _crossfit_fold_report(value) for value in result.fold_results
            ],
        },
        "scorers": scorers,
        "comparisons": {
            "direct_vs_axis_sign": direct_vs_axis,
            "nested_vs_direct": nested_vs_direct,
        },
        "exploratory_signal": exploratory_signal,
        "exploratory_signal_definition": (
            "Full Direct exceeds axis/sign and cross-fitted nested exceeds full "
            "Direct with positive paired task-macro Spearman and regret point "
            "estimates and at least three jointly positive tasks in each comparison."
        ),
        "descriptive_validation_delta": {
            "task_macro_spearman": (
                nested_validation["task_macro_spearman"]
                - full_direct_validation["task_macro_spearman"]
            ),
            "normalized_top1_regret_improvement": (
                full_direct_validation["task_macro_normalized_top1_regret"]
                - nested_validation["task_macro_normalized_top1_regret"]
            ),
        },
        "claim_boundary": (
            "Exposed-validation post-hoc diagnostic only. Cross-fitting removes "
            "in-sample Direct model fitting from the residual's training base; "
            "the shared affine score coordinate is not strictly fold-blind. A "
            "positive residual remains "
            "capacity-confounded and cannot establish that future information, "
            "LeWM, FCPG planning, or CEM is necessary."
        ),
    }
    return ExploratoryCrossFitActualReturnRun(report=report, result=result)


__all__ = [
    "CROSS_FIT_EPISODES_PER_TASK_PER_FOLD",
    "CROSS_FIT_FOLDS",
    "CrossFitFoldResult",
    "DEFAULT_DIRECT_SEEDS",
    "DEFAULT_RAW_RETURN_TIE_TOLERANCE",
    "DEFAULT_RESIDUAL_SEEDS",
    "DirectSeedResult",
    "ExploratoryActualReturnResult",
    "ExploratoryActualReturnRun",
    "ExploratoryCrossFitActualReturnResult",
    "ExploratoryCrossFitActualReturnRun",
    "EpisodeCrossFitAssignment",
    "FutureInputScaler",
    "FutureResidualInputs",
    "FutureStateResidualRanker",
    "LossRecord",
    "ResidualSeedResult",
    "actual_return_ranking_loss",
    "actual_return_lookup_baselines",
    "evaluate_actual_return_scores",
    "build_episode_crossfit_assignment",
    "fit_future_input_scaler",
    "future_residual_inputs",
    "run_exploratory_actual_return",
    "run_exploratory_actual_return_crossfit",
    "train_actual_return_exploratory",
    "train_actual_return_crossfit_exploratory",
]
