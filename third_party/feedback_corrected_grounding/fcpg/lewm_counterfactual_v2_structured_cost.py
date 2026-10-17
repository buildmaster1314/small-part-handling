"""Structured Counterfactual v2 CostAdapter development comparison.

This module implements the single pre-declared architecture comparison used
after the flat tiny-head E1 failure.  Direct, predicted, and oracle scorers are
the *same* model: only the five-step state sequence changes.  In particular,
the direct control repeats the current ``z_last`` for all five steps while
retaining the same history, action-prefix, goal, and task inputs.

The experiment remains external development.  It does not retrain FCPG or
LeWM, does not consume privileged labels at deployment, and does not launch
CEM.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from fcpg.lewm_counterfactual_v2 import (
    TASK_NAMES,
    DirectProbeInputs,
    OracleProbeInputs,
    PredictedProbeInputs,
    canonical_sha256,
    sha256_array,
)
from fcpg.lewm_counterfactual_v2_scorer import (
    RANDOM_CANDIDATE_NAMES,
    GroupedScorerData,
    SourceEligibility,
    _canonical_json_bytes,
    _git_provenance,
    _resolve_path,
    _sha256_file,
    _write_json,
    _write_jsonl,
    _write_torch,
    audit_scorer_experiment_result,
    assemble_grouped_scorer_data,
    episode_task_macro_bootstrap,
    higher_is_better_ranking_metrics,
    paired_episode_task_macro_bootstrap,
    pairwise_ranking_huber_loss,
    publish_scorer_directory_exclusive,
    scorer_state_sha256,
    trainable_parameter_count,
    validate_source_result_sha256,
)
from fcpg.lewm_counterfactual_v2_storage import load_counterfactual_v2_store


StructuredProbeName = Literal["direct", "predicted", "oracle"]
STRUCTURED_PROBES: tuple[StructuredProbeName, ...] = (
    "direct",
    "predicted",
    "oracle",
)
STRUCTURED_SCORER_PROTOCOL = (
    "fcpg_lewm_counterfactual_v2_structured_cost_e1_e3_v1"
)
STRICT_NEAR_CANDIDATE_NAMES = frozenset(
    {"expert", "expert_noise_0.05", "expert_noise_0.15", "expert_noise_0.3"}
)


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, copy=True, order="C", dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StructuredCostInputs:
    """Exact deployable tensors consumed by the structured scorer."""

    z_history: np.ndarray
    history_action_raw: np.ndarray
    candidate_action_raw: np.ndarray
    goal_visual: np.ndarray
    task_onehot: np.ndarray
    future_state: np.ndarray

    def validate(
        self,
        *,
        latent_dim: int = 192,
        horizon: int = 5,
        action_dim: int = 4,
        action_prefix_stride: int = 5,
    ) -> None:
        arrays = (
            self.z_history,
            self.history_action_raw,
            self.candidate_action_raw,
            self.goal_visual,
            self.task_onehot,
            self.future_state,
        )
        if any(not isinstance(value, np.ndarray) for value in arrays):
            raise TypeError("structured scorer inputs must be numpy arrays")
        n, c = self.candidate_action_raw.shape[:2]
        if self.z_history.ndim != 3 or self.z_history.shape[0] != n:
            raise ValueError("z_history must have shape [N,T,D]")
        if self.z_history.shape[-1] != latent_dim:
            raise ValueError("z_history latent dimension changed")
        if (
            self.history_action_raw.ndim != 4
            or self.history_action_raw.shape[0] != n
            or self.history_action_raw.shape[-1] != action_dim
        ):
            raise ValueError("history_action_raw must have shape [N,T,S,A]")
        if (
            self.candidate_action_raw.ndim != 4
            or self.candidate_action_raw.shape[-1] != action_dim
            or self.candidate_action_raw.shape[2] != horizon * action_prefix_stride
        ):
            raise ValueError("candidate action sequence does not match H x prefix stride")
        if self.goal_visual.ndim != 2 or self.goal_visual.shape[0] != n:
            raise ValueError("goal_visual must have shape [N,G]")
        if self.task_onehot.shape != (n, len(TASK_NAMES)):
            raise ValueError("task_onehot shape changed")
        if self.future_state.shape != (n, c, horizon, latent_dim):
            raise ValueError("future_state must have shape [N,C,H,D]")
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("structured scorer inputs contain non-finite values")


def structured_inputs_from_probe(
    inputs: DirectProbeInputs | PredictedProbeInputs | OracleProbeInputs,
    *,
    probe: StructuredProbeName,
) -> StructuredCostInputs:
    """Build one matched structured view without exposing offline labels."""

    if probe == "direct":
        if not isinstance(inputs, DirectProbeInputs):
            raise TypeError("direct structured input requires DirectProbeInputs")
        n, c = inputs.candidate_action_raw.shape[:2]
        z_last = inputs.z_history[:, -1, :]
        future = np.broadcast_to(z_last[:, None, None, :], (n, c, 5, 192))
    elif probe == "predicted":
        if not isinstance(inputs, PredictedProbeInputs):
            raise TypeError("predicted structured input requires PredictedProbeInputs")
        future = inputs.z_pred_future
    elif probe == "oracle":
        if not isinstance(inputs, OracleProbeInputs):
            raise TypeError("oracle structured input requires OracleProbeInputs")
        future = inputs.oracle_future_latent
    else:
        raise ValueError(f"unknown structured probe {probe!r}")
    result = StructuredCostInputs(
        z_history=_readonly(inputs.z_history, dtype=np.dtype(np.float32)),
        history_action_raw=_readonly(
            inputs.history_action_raw, dtype=np.dtype(np.float32)
        ),
        candidate_action_raw=_readonly(
            inputs.candidate_action_raw, dtype=np.dtype(np.float32)
        ),
        goal_visual=_readonly(inputs.goal_visual, dtype=np.dtype(np.float32)),
        task_onehot=_readonly(inputs.task_onehot, dtype=np.dtype(np.float32)),
        future_state=_readonly(future, dtype=np.dtype(np.float32)),
    )
    result.validate()
    return result


def concatenate_structured_inputs(
    values: Sequence[StructuredCostInputs],
) -> StructuredCostInputs:
    if not values:
        raise ValueError("structured input sequence must not be empty")
    result = StructuredCostInputs(
        **{
            name: _readonly(
                np.concatenate([getattr(value, name) for value in values], axis=0),
                dtype=np.dtype(np.float32),
            )
            for name in StructuredCostInputs.__dataclass_fields__
        }
    )
    result.validate()
    return result


@dataclass(frozen=True)
class CoordinateScale:
    mean: np.ndarray
    scale: np.ndarray

    def as_mapping(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def _fit_coordinate_scale(values: np.ndarray, *, epsilon: float) -> CoordinateScale:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("coordinate scaler requires finite [N,D] values")
    mean = array.mean(axis=0)
    scale = array.std(axis=0, ddof=0)
    scale = np.where(scale > epsilon, scale, 1.0)
    return CoordinateScale(_readonly(mean), _readonly(scale))


@dataclass(frozen=True)
class StructuredFeatureScaler:
    """Train-only block scaler; task identity remains byte-for-byte unchanged."""

    latent: CoordinateScale
    action: CoordinateScale
    goal: CoordinateScale
    epsilon: float

    def transform(self, inputs: StructuredCostInputs) -> StructuredCostInputs:
        inputs.validate()

        def scaled(value: np.ndarray, block: CoordinateScale) -> np.ndarray:
            result = (
                value.astype(np.float64, copy=False)
                - block.mean.reshape((1,) * (value.ndim - 1) + (-1,))
            ) / block.scale.reshape((1,) * (value.ndim - 1) + (-1,))
            return _readonly(result, dtype=np.dtype(np.float32))

        result = StructuredCostInputs(
            z_history=scaled(inputs.z_history, self.latent),
            history_action_raw=scaled(inputs.history_action_raw, self.action),
            candidate_action_raw=scaled(inputs.candidate_action_raw, self.action),
            goal_visual=scaled(inputs.goal_visual, self.goal),
            task_onehot=_readonly(inputs.task_onehot, dtype=np.dtype(np.float32)),
            future_state=scaled(inputs.future_state, self.latent),
        )
        result.validate()
        if not np.array_equal(result.task_onehot, inputs.task_onehot):
            raise RuntimeError("structured scaling changed task identity")
        return result

    def as_mapping(self) -> dict[str, Any]:
        return {
            "latent": self.latent.as_mapping(),
            "action": self.action.as_mapping(),
            "goal": self.goal.as_mapping(),
            "epsilon": float(self.epsilon),
            "fit_scope": "pooled_clean_training_shared_across_probes",
            "task_onehot": "unchanged",
            "latent_scaling": "fit_z_history_only_applied_to_history_and_future_state",
            "action_scaling": "shared_across_history_and_candidate_actions",
        }


def fit_structured_feature_scaler(
    inputs: StructuredCostInputs, *, epsilon: float = 1e-8
) -> StructuredFeatureScaler:
    """Fit all feature statistics exclusively from the supplied training rows."""

    inputs.validate()
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("scaler epsilon must be finite and positive")
    # The three probes intentionally have different future-state distributions.
    # Fitting on future_state would therefore transform their otherwise identical
    # history differently.  The common latent coordinate system is fitted once
    # from the raw deployable history shared by every probe.
    latent_rows = inputs.z_history.reshape(-1, inputs.z_history.shape[-1])
    action_rows = np.concatenate(
        (
            inputs.history_action_raw.reshape(
                -1, inputs.history_action_raw.shape[-1]
            ),
            inputs.candidate_action_raw.reshape(
                -1, inputs.candidate_action_raw.shape[-1]
            ),
        ),
        axis=0,
    )
    return StructuredFeatureScaler(
        latent=_fit_coordinate_scale(latent_rows, epsilon=epsilon),
        action=_fit_coordinate_scale(action_rows, epsilon=epsilon),
        goal=_fit_coordinate_scale(inputs.goal_visual, epsilon=epsilon),
        epsilon=float(epsilon),
    )


@dataclass(frozen=True)
class StructuredPooledTrainingData:
    probe_name: StructuredProbeName
    inputs: StructuredCostInputs
    labels: np.ndarray
    task_index: np.ndarray
    scaler: StructuredFeatureScaler
    source_result_sha256: str
    source_eligibility: SourceEligibility
    groundings: tuple[str, ...]
    view: str
    model: str


def assemble_structured_pooled_training_data(
    dataset: Any,
    *,
    probe: StructuredProbeName,
    model: str,
    groundings: Sequence[str],
    view: str,
    target: str = "actual_return",
    horizon_index: int = 4,
    allow_ineligible_external_development: bool = False,
) -> StructuredPooledTrainingData:
    cells = tuple(str(value) for value in groundings)
    if not cells or len(cells) != len(set(cells)):
        raise ValueError("groundings must be a non-empty unique sequence")
    grouped = tuple(
        assemble_grouped_scorer_data(
            dataset,
            probe=probe,
            model=model,
            grounding=grounding,
            view=view,
            target=target,  # type: ignore[arg-type]
            horizon_index=horizon_index,
            allow_ineligible_external_development=(
                allow_ineligible_external_development
            ),
        )
        for grounding in cells
    )
    reference = grouped[0]
    structured = [
        structured_inputs_from_probe(cell.train.inputs, probe=probe)
        for cell in grouped
    ]
    for cell in grouped[1:]:
        if cell.source_result_sha256 != reference.source_result_sha256:
            raise ValueError("pooled cells do not share one immutable source store")
        if cell.source_eligibility != reference.source_eligibility:
            raise ValueError("pooled cells disagree on source eligibility")
        if not np.array_equal(cell.train.raw_labels, reference.train.raw_labels):
            raise ValueError("camera-invariant labels differ across pooled cells")
        if not np.array_equal(cell.train.sample_indices, reference.train.sample_indices):
            raise ValueError("pooled cells are not physical-snapshot aligned")
        if cell.train.candidate_names != reference.train.candidate_names:
            raise ValueError("pooled cells have different candidate identities")
    inputs = concatenate_structured_inputs(structured)
    labels = _readonly(
        np.concatenate([cell.train.labels for cell in grouped], axis=0),
        dtype=np.dtype(np.float32),
    )
    task_index = _readonly(
        np.concatenate([cell.train.task_index for cell in grouped], axis=0),
        dtype=np.dtype(np.int64),
    )
    return StructuredPooledTrainingData(
        probe_name=probe,
        inputs=inputs,
        labels=labels,
        task_index=task_index,
        scaler=fit_structured_feature_scaler(inputs),
        source_result_sha256=reference.source_result_sha256,
        source_eligibility=reference.source_eligibility,
        groundings=cells,
        view=str(view),
        model=str(model),
    )


STRUCTURED_SHARED_INPUT_FIELDS = (
    "z_history",
    "history_action_raw",
    "candidate_action_raw",
    "goal_visual",
    "task_onehot",
)


def bind_shared_structured_scaler(
    pooled: Mapping[str, StructuredPooledTrainingData],
) -> tuple[dict[str, StructuredPooledTrainingData], dict[str, Any]]:
    """Bind one scaler to all probes and prove their shared inputs stay identical."""

    if set(pooled) != set(STRUCTURED_PROBES):
        raise ValueError("shared scaler requires Direct, Predicted, and Oracle data")
    reference = pooled["direct"]
    shared_scaler = fit_structured_feature_scaler(reference.inputs)
    scaler_mapping = shared_scaler.as_mapping()
    scaler_sha = canonical_sha256(scaler_mapping)
    candidate_scalers = {
        probe: fit_structured_feature_scaler(value.inputs)
        for probe, value in pooled.items()
    }
    for probe, candidate_scaler in candidate_scalers.items():
        if candidate_scaler.as_mapping() != scaler_mapping:
            raise ValueError(
                f"structured probe {probe} derived a different shared scaler"
            )
    transformed_by_probe = {
        probe: shared_scaler.transform(value.inputs)
        for probe, value in pooled.items()
    }
    raw_hashes: dict[str, str] = {}
    transformed_hashes: dict[str, str] = {}
    transformed_reference = transformed_by_probe["direct"]
    for field in STRUCTURED_SHARED_INPUT_FIELDS:
        raw_reference = getattr(reference.inputs, field)
        transformed_value = getattr(transformed_reference, field)
        raw_hashes[field] = sha256_array(raw_reference)
        transformed_hashes[field] = sha256_array(transformed_value)
        for probe in STRUCTURED_PROBES[1:]:
            value = pooled[probe]
            if not np.array_equal(getattr(value.inputs, field), raw_reference):
                raise ValueError(
                    f"structured probes differ in shared raw input {field}"
                )
            transformed_candidate = transformed_by_probe[probe]
            if not np.array_equal(
                getattr(transformed_candidate, field), transformed_value
            ):
                raise ValueError(
                    f"structured probes differ in transformed shared input {field}"
                )
    rebound = {
        probe: replace(value, scaler=shared_scaler) for probe, value in pooled.items()
    }
    return rebound, {
        "passed": True,
        "fit_scope": "pooled_clean_training_shared_across_probes",
        "latent_fit_source": "raw_z_history_only",
        "shared_scaler_canonical_sha256": scaler_sha,
        "latent_block_canonical_sha256": canonical_sha256(
            shared_scaler.latent.as_mapping()
        ),
        "action_block_canonical_sha256": canonical_sha256(
            shared_scaler.action.as_mapping()
        ),
        "goal_block_canonical_sha256": canonical_sha256(
            shared_scaler.goal.as_mapping()
        ),
        "raw_shared_input_sha256": raw_hashes,
        "transformed_shared_input_sha256": transformed_hashes,
        "byte_identical_shared_inputs": list(STRUCTURED_SHARED_INPUT_FIELDS),
        "future_state_used_for_scaler_fit": False,
    }


class HorizonStructuredCostAdapter(nn.Module):
    """Shared horizon-wise state/action cost whose five step costs are summed."""

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        state_dim: int = 16,
        history_hidden_dim: int = 16,
        action_dim: int = 4,
        action_hidden_dim: int = 16,
        goal_dim: int = 48,
        goal_hidden_dim: int = 8,
        task_dim: int = 4,
        task_hidden_dim: int = 4,
        step_hidden_dim: int = 16,
        horizon: int = 5,
        action_prefix_stride: int = 5,
    ) -> None:
        super().__init__()
        dimensions = (
            latent_dim,
            state_dim,
            history_hidden_dim,
            action_dim,
            action_hidden_dim,
            goal_dim,
            goal_hidden_dim,
            task_dim,
            task_hidden_dim,
            step_hidden_dim,
            horizon,
            action_prefix_stride,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("structured adapter dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.state_dim = int(state_dim)
        self.history_hidden_dim = int(history_hidden_dim)
        self.action_dim = int(action_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        self.goal_dim = int(goal_dim)
        self.task_dim = int(task_dim)
        self.horizon = int(horizon)
        self.action_prefix_stride = int(action_prefix_stride)

        self.state_projection = nn.Linear(self.latent_dim, self.state_dim)
        self.history_encoder = nn.GRU(
            self.state_dim, self.history_hidden_dim, batch_first=True
        )
        self.action_prefix_encoder = nn.GRU(
            self.action_dim, self.action_hidden_dim, batch_first=True
        )
        self.goal_projection = nn.Linear(self.goal_dim, goal_hidden_dim)
        self.task_projection = nn.Linear(self.task_dim, task_hidden_dim)
        step_input_dim = (
            self.state_dim
            + self.history_hidden_dim
            + self.action_hidden_dim
            + goal_hidden_dim
            + task_hidden_dim
        )
        self.step_cost = nn.Sequential(
            nn.Linear(step_input_dim, step_hidden_dim),
            nn.GELU(),
            nn.Linear(step_hidden_dim, 1),
        )

    def forward(
        self,
        z_history: Tensor,
        history_action_raw: Tensor,
        candidate_action_raw: Tensor,
        goal_visual: Tensor,
        task_onehot: Tensor,
        future_state: Tensor,
    ) -> Tensor:
        if z_history.ndim != 3 or z_history.shape[-1] != self.latent_dim:
            raise ValueError("z_history must have shape [B,T,latent_dim]")
        batch = z_history.shape[0]
        if candidate_action_raw.ndim != 4:
            raise ValueError("candidate_action_raw must have shape [B,C,S,A]")
        candidates = candidate_action_raw.shape[1]
        if candidate_action_raw.shape != (
            batch,
            candidates,
            self.horizon * self.action_prefix_stride,
            self.action_dim,
        ):
            raise ValueError("candidate action shape changed")
        if history_action_raw.ndim != 4 or history_action_raw.shape[0] != batch:
            raise ValueError("history actions must have shape [B,T,S,A]")
        if history_action_raw.shape[-1] != self.action_dim:
            raise ValueError("history action dimension changed")
        if future_state.shape != (
            batch,
            candidates,
            self.horizon,
            self.latent_dim,
        ):
            raise ValueError("future state shape changed")
        if goal_visual.shape != (batch, self.goal_dim):
            raise ValueError("goal_visual shape changed")
        if task_onehot.shape != (batch, self.task_dim):
            raise ValueError("task_onehot shape changed")

        projected_history = self.state_projection(z_history)
        _, history_hidden = self.history_encoder(projected_history)
        history_context = history_hidden[-1]

        history_actions = history_action_raw.reshape(batch, -1, self.action_dim)
        history_steps = history_actions.shape[1]
        repeated_history_actions = history_actions[:, None].expand(
            -1, candidates, -1, -1
        )
        action_sequence = torch.cat(
            (repeated_history_actions, candidate_action_raw), dim=2
        ).reshape(batch * candidates, -1, self.action_dim)
        action_encoded, _ = self.action_prefix_encoder(action_sequence)
        prefix_indices = history_steps + (
            torch.arange(self.horizon, device=action_encoded.device) + 1
        ) * self.action_prefix_stride - 1
        action_prefix = action_encoded[:, prefix_indices, :].reshape(
            batch, candidates, self.horizon, self.action_hidden_dim
        )

        projected_future = self.state_projection(future_state)
        history_context = history_context[:, None, None, :].expand(
            -1, candidates, self.horizon, -1
        )
        goal_context = self.goal_projection(goal_visual)[:, None, None, :].expand(
            -1, candidates, self.horizon, -1
        )
        task_context = self.task_projection(task_onehot)[:, None, None, :].expand(
            -1, candidates, self.horizon, -1
        )
        step_features = torch.cat(
            (
                projected_future,
                history_context,
                action_prefix,
                goal_context,
                task_context,
            ),
            dim=-1,
        )
        step_scores = self.step_cost(step_features).squeeze(-1)
        return step_scores.sum(dim=-1)


def structured_model_from_config(config: Mapping[str, Any]) -> HorizonStructuredCostAdapter:
    return HorizonStructuredCostAdapter(
        latent_dim=int(config["latent_dim"]),
        state_dim=int(config["state_projection_dim"]),
        history_hidden_dim=int(config["history_hidden_dim"]),
        action_dim=int(config["action_dim"]),
        action_hidden_dim=int(config["action_hidden_dim"]),
        goal_dim=int(config["goal_dim"]),
        goal_hidden_dim=int(config["goal_hidden_dim"]),
        task_dim=int(config["task_dim"]),
        task_hidden_dim=int(config["task_hidden_dim"]),
        step_hidden_dim=int(config["step_hidden_dim"]),
        horizon=int(config["horizon"]),
        action_prefix_stride=int(config["action_prefix_stride"]),
    )


def structured_tensor_arguments(
    inputs: StructuredCostInputs, *, device: str
) -> tuple[Tensor, ...]:
    inputs.validate()
    return tuple(
        torch.from_numpy(getattr(inputs, name).copy()).to(device=device)
        for name in StructuredCostInputs.__dataclass_fields__
    )


@dataclass
class TrainedStructuredCostAdapter:
    probe_name: StructuredProbeName
    seed: int
    model: HorizonStructuredCostAdapter
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


def train_structured_cost_adapter(
    data: StructuredPooledTrainingData,
    *,
    model_config: Mapping[str, Any],
    seed: int,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    pairwise_weight: float = 1.0,
    huber_weight: float = 0.1,
    huber_beta: float = 1.0,
    tie_tolerance: float = 1e-6,
    device: str = "cpu",
) -> TrainedStructuredCostAdapter:
    """Deterministic fixed-step full-batch training for one structured probe."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    _seed_training(seed)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = structured_model_from_config(model_config).to(device)
        expected = int(model_config["expected_parameter_count"])
        actual = trainable_parameter_count(model)
        if actual != expected:
            raise RuntimeError(
                f"structured parameter contract changed: expected {expected}, got {actual}"
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scaled = data.scaler.transform(data.inputs)
        arguments = structured_tensor_arguments(scaled, device=device)
        labels = torch.from_numpy(data.labels.copy()).to(device=device)

        def loss_value() -> Any:
            scores = model(*arguments)
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
            initial_loss = float(loss_value().total.detach().cpu())
        final_gradient_norm = float("nan")
        for _step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_value()
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("structured scorer gradient became non-finite")
            final_gradient_norm = float(gradient_norm.detach().cpu())
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = loss_value()
        model = model.cpu()
        return TrainedStructuredCostAdapter(
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


def cyclic_structured_future(inputs: StructuredCostInputs, *, offset: int = 1) -> StructuredCostInputs:
    candidate_count = int(inputs.future_state.shape[1])
    resolved = int(offset) % candidate_count
    if resolved == 0:
        raise ValueError("cyclic structured future must not be identity")
    result = StructuredCostInputs(
        **{
            name: _readonly(
                np.roll(value, shift=resolved, axis=1)
                if name == "future_state"
                else value,
                dtype=np.dtype(np.float32),
            )
            for name, value in inputs.__dict__.items()
        }
    )
    result.validate()
    return result


def score_structured_validation_split(
    trained: TrainedStructuredCostAdapter,
    scaler: StructuredFeatureScaler,
    data: GroupedScorerData,
    *,
    view: str,
    grounding: str,
    variant: str = "standard",
) -> list[dict[str, Any]]:
    inputs = structured_inputs_from_probe(
        data.validation.inputs, probe=trained.probe_name
    )
    if variant == "cyclic_candidate_future":
        if trained.probe_name != "predicted":
            raise TypeError("cyclic future is only defined for predicted scorer")
        inputs = cyclic_structured_future(inputs)
    elif variant != "standard":
        raise ValueError(f"unknown structured validation variant {variant!r}")
    scaled = scaler.transform(inputs)
    trained.model.eval()
    with torch.no_grad():
        scores = trained.model(
            *structured_tensor_arguments(scaled, device="cpu")
        ).cpu().numpy()
    if scores.shape != data.validation.raw_labels.shape:
        raise RuntimeError("structured scores do not align with validation labels")
    scaler_sha256 = canonical_sha256(scaler.as_mapping())
    rows: list[dict[str, Any]] = []
    for row_index, names in enumerate(data.validation.candidate_names):
        kept = [
            index for index, name in enumerate(names)
            if name not in RANDOM_CANDIDATE_NAMES
        ]
        kept_names = [names[index] for index in kept]
        if kept_names.count("expert") != 1 or len(kept_names) < 2:
            raise ValueError("near-expert candidate family is malformed")
        metrics = higher_is_better_ranking_metrics(
            scores[row_index, kept],
            data.validation.raw_labels[row_index, kept],
            expert_index=kept_names.index("expert"),
        )
        strict_kept = [
            index for index, name in enumerate(names)
            if name in STRICT_NEAR_CANDIDATE_NAMES
        ]
        strict_names = [names[index] for index in strict_kept]
        if set(strict_names) != set(STRICT_NEAR_CANDIDATE_NAMES):
            raise ValueError("strict-near candidate family is malformed")
        strict_metrics = higher_is_better_ranking_metrics(
            scores[row_index, strict_kept],
            data.validation.raw_labels[row_index, strict_kept],
            expert_index=strict_names.index("expert"),
        )
        rows.append(
            {
                "probe": trained.probe_name,
                "architecture": "horizon_shared_cost_gru",
                "variant": variant,
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
                    float(data.validation.raw_labels[row_index, index])
                    for index in kept
                ],
                "strict_near_candidate_subset": "expert_plus_three_expert_noise",
                "strict_near_candidate_names": strict_names,
                "strict_near_spearman": strict_metrics["spearman"],
                "strict_near_normalized_top1_regret": strict_metrics[
                    "normalized_top1_regret"
                ],
                **metrics,
            }
        )
    return rows


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


def summarize_structured_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics = (
        "spearman",
        "normalized_top1_regret",
        "strict_near_spearman",
        "strict_near_normalized_top1_regret",
    )
    clean = {
        probe: _condition_rows(rows, probe=probe, views=["clean"], grounding="fcpg")
        for probe in STRUCTURED_PROBES
    }
    drift = {
        probe: _condition_rows(
            rows,
            probe=probe,
            views=["rotation5", "translation4"],
            grounding="fcpg",
        )
        for probe in ("direct", "predicted")
    }
    shuffled = _condition_rows(
        rows,
        probe="predicted",
        views=["rotation5", "translation4"],
        grounding="fcpg",
        variant="cyclic_candidate_future",
    )
    drift_nominal = _condition_rows(
        rows,
        probe="predicted",
        views=["rotation5", "translation4"],
        grounding="nominal",
    )
    clean_nominal = _condition_rows(
        rows, probe="predicted", views=["clean"], grounding="nominal"
    )

    comparisons: dict[str, Any] = {}

    def compare(
        name: str,
        left: Sequence[Mapping[str, Any]],
        right: Sequence[Mapping[str, Any]],
        *,
        same_scorer: bool = False,
    ) -> None:
        if not left or not right:
            raise ValueError(f"structured comparison {name} has an empty cell")
        comparisons[name] = {
            metric: paired_episode_task_macro_bootstrap(
                left,
                right,
                metric,
                higher_is_better=metric.endswith("spearman"),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                require_same_scorer=same_scorer,
            )
            for metric in metrics
        }

    compare("E1_oracle_vs_direct_clean_fcpg", clean["direct"], clean["oracle"])
    compare("E2_predicted_vs_direct_drift_fcpg", drift["direct"], drift["predicted"])
    compare(
        "E2_predicted_vs_cyclic_shuffle_drift_fcpg",
        shuffled,
        drift["predicted"],
        same_scorer=True,
    )
    compare(
        "E3_fcpg_vs_nominal_grounding_drift",
        drift_nominal,
        drift["predicted"],
        same_scorer=True,
    )
    compare("clean_predicted_vs_direct_fcpg", clean["direct"], clean["predicted"])
    compare(
        "clean_fcpg_vs_nominal_grounding",
        clean_nominal,
        clean["predicted"],
        same_scorer=True,
    )
    cells = {
        "clean_fcpg_direct": clean["direct"],
        "clean_fcpg_predicted": clean["predicted"],
        "clean_fcpg_oracle": clean["oracle"],
        "drift_fcpg_direct": drift["direct"],
        "drift_fcpg_predicted": drift["predicted"],
        "drift_fcpg_predicted_cyclic_shuffle": shuffled,
        "drift_nominal_predicted": drift_nominal,
    }
    return {
        "primary_target": "H5 actual_return",
        "candidate_subset": "nonrandom_without_random_0_random_1",
        "primary_candidate_names": [
            "expert",
            "policy",
            "expert_noise_0.05",
            "expert_noise_0.15",
            "expert_noise_0.3",
            "same_task_shuffled",
        ],
        "secondary_candidate_subset": "expert_plus_three_expert_noise",
        "secondary_candidate_names": [
            "expert",
            "expert_noise_0.05",
            "expert_noise_0.15",
            "expert_noise_0.3",
        ],
        "primary_metrics": [
            "task_macro_spearman",
            "task_macro_normalized_top1_regret",
        ],
        "secondary_metrics": [
            "task_macro_strict_near_spearman",
            "task_macro_strict_near_normalized_top1_regret",
        ],
        "architecture": "horizon_shared_cost_gru",
        "cells": {
            name: {
                metric: episode_task_macro_bootstrap(
                    cell,
                    metric,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                )
                for metric in metrics
            }
            for name, cell in cells.items()
        },
        "comparisons": comparisons,
    }


def validate_structured_experiment_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol") != STRUCTURED_SCORER_PROTOCOL:
        raise ValueError("unexpected structured CostAdapter protocol")
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or dict(scope) != {
        "classification": "external_development",
        "formal_claim_allowed": False,
        "allow_ineligible_external_development": True,
        "upstream_training_allowed": False,
        "online_cem_allowed": False,
        "hyperparameter_sweep_allowed": False,
    }:
        raise ValueError("structured scorer scope must remain external development")
    source = config.get("source")
    source_sha = str(source.get("result_sha256", "")) if isinstance(source, Mapping) else ""
    if (
        not isinstance(source, Mapping)
        or not source.get("directory")
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("structured source must bind a result SHA-256")
    data = config.get("data")
    if not isinstance(data, Mapping) or (
        data.get("target") != "actual_return"
        or data.get("horizon_index") != 4
        or data.get("model") != "fcpg"
        or data.get("probes") != ["direct", "predicted", "oracle"]
    ):
        raise ValueError("structured data contract changed")
    training = data.get("training")
    if not isinstance(training, Mapping) or dict(training) != {
        "view": "clean",
        "groundings": ["nominal", "fcpg"],
        "pooling": "equal_concat",
    }:
        raise ValueError("structured training split contract changed")
    model = config.get("model")
    expected_model = {
        "architecture": "horizon_shared_cost_gru",
        "latent_dim": 192,
        "state_projection_dim": 16,
        "history_hidden_dim": 16,
        "action_dim": 4,
        "action_hidden_dim": 16,
        "goal_dim": 48,
        "goal_hidden_dim": 8,
        "task_dim": 4,
        "task_hidden_dim": 4,
        "step_hidden_dim": 16,
        "horizon": 5,
        "action_prefix_stride": 5,
        "future_state_contract": {
            "direct": "repeat_z_history_last_for_all_horizons",
            "predicted": "z_pred_future",
            "oracle": "oracle_future_latent_offline_upper_bound",
        },
        "sharing": "identical_model_class_and_exact_parameter_count_all_probes",
        "expected_parameter_count": 7181,
    }
    if not isinstance(model, Mapping) or dict(model) != expected_model:
        raise ValueError("structured model contract changed")
    scaling = config.get("feature_scaling")
    if not isinstance(scaling, Mapping) or dict(scaling) != {
        "fit_scope": "pooled_clean_training_shared_across_probes",
        "method": "per_coordinate_block_standardization",
        "latent": "fit_z_history_only_applied_to_history_and_future_state",
        "action": "shared_across_history_and_candidate_actions",
        "task_onehot": "unchanged",
    }:
        raise ValueError("structured scaling contract changed")
    optimization = config.get("optimization")
    if not isinstance(optimization, Mapping) or dict(optimization) != {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "batching": "full_batch",
        "steps": 2000,
        "scheduler": None,
        "early_stopping": False,
        "seeds": [8201, 8202, 8203],
        "deterministic": True,
        "device": "cpu",
    }:
        raise ValueError("structured optimization contract changed")
    loss = config.get("loss")
    if not isinstance(loss, Mapping) or dict(loss) != {
        "name": "ranknet_plus_huber",
        "pairwise_weight": 1.0,
        "huber_weight": 0.1,
        "huber_beta": 1.0,
        "tie_tolerance": 1e-6,
    }:
        raise ValueError("structured loss contract changed")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping) or dict(evaluation) != {
        "bootstrap_samples": 10000,
        "bootstrap_seed": 8299,
        "drift_views": ["rotation5", "translation4"],
        "exclude_candidates": ["random_0", "random_1"],
        "resampling_unit": "physical_episode_within_task",
        "metrics": ["task_macro_spearman", "task_macro_normalized_top1_regret"],
        "gates": {
            "E1": "oracle_vs_direct_clean_fcpg",
            "E2": "predicted_vs_direct_drift_fcpg_with_cyclic_future_control",
            "E3": "same_predicted_scorer_fcpg_vs_nominal_grounding_drift",
        },
    }:
        raise ValueError("structured evaluation contract changed")
    output = config.get("output")
    if not isinstance(output, Mapping) or output.get("write_mode") != "exclusive_atomic_directory":
        raise ValueError("structured output must use exclusive atomic publication")
    code = config.get("code")
    if not isinstance(code, Mapping) or code.get("require_clean_git") is not True:
        raise ValueError("structured experiment must bind a clean git commit")


def _checkpoint_payload(
    trained: TrainedStructuredCostAdapter,
    *,
    config_sha256: str,
    source_result_sha256: str,
    code: Mapping[str, Any],
    scaler_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol": STRUCTURED_SCORER_PROTOCOL,
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


def run_structured_cost_experiment(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    confirm_external_development: bool = False,
) -> dict[str, Any]:
    """Train the one frozen structured comparison and seal its E1--E3 result."""

    validate_structured_experiment_config(config)
    if not confirm_external_development:
        raise PermissionError("structured experiment requires external-development confirmation")
    source_path = _resolve_path(repo_root, config["source"]["directory"])
    output_path = _resolve_path(repo_root, config["output"]["directory"])
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"exclusive structured output exists: {output_path}")
    dataset = load_counterfactual_v2_store(source_path)
    actual_source = validate_source_result_sha256(
        dataset, str(config["source"]["result_sha256"])
    )
    code = _git_provenance(repo_root)
    if code["dirty"]:
        raise RuntimeError("structured experiment requires a clean git worktree")
    config_sha = canonical_sha256(config)
    training_config = config["data"]["training"]
    pooled = {
        probe: assemble_structured_pooled_training_data(
            dataset,
            probe=probe,
            model=str(config["data"]["model"]),
            groundings=training_config["groundings"],
            view=str(training_config["view"]),
            target=str(config["data"]["target"]),
            horizon_index=int(config["data"]["horizon_index"]),
            allow_ineligible_external_development=True,
        )
        for probe in STRUCTURED_PROBES
    }
    pooled, scaler_fairness = bind_shared_structured_scaler(pooled)
    eligibility = pooled["direct"].source_eligibility
    if any(value.source_eligibility != eligibility for value in pooled.values()):
        raise RuntimeError("structured probes disagree on source eligibility")

    # The same class/config yields exact equality, not approximate budget matching.
    fairness_models = {
        probe: structured_model_from_config(config["model"])
        for probe in STRUCTURED_PROBES
    }
    parameter_counts = {
        probe: trainable_parameter_count(model)
        for probe, model in fairness_models.items()
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError("structured Direct/Predicted/Oracle parameter counts differ")
    expected_count = int(config["model"]["expected_parameter_count"])
    if set(parameter_counts.values()) != {expected_count}:
        raise RuntimeError("structured parameter count differs from frozen config")

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
        shared_scaler_path = staging / "scalers" / "shared.json"
        _write_json(shared_scaler_path, pooled["direct"].scaler.as_mapping())
        shared_scaler_file_sha = _sha256_file(shared_scaler_path)
        _write_json(
            staging / "fairness.json",
            {
                "passed": True,
                "model_class": "HorizonStructuredCostAdapter",
                "parameter_counts": parameter_counts,
                "exact_parameter_equality": True,
                "only_probe_specific_input": "future_state",
                "direct_future_state": "repeat_z_history_last_for_all_horizons",
                "shared_inputs": [
                    "z_history",
                    "history_action_raw",
                    "candidate_action_raw",
                    "goal_visual",
                    "task_onehot",
                ],
                "shared_scaler": {
                    **scaler_fairness,
                    "artifact": "scalers/shared.json",
                    "file_sha256": shared_scaler_file_sha,
                },
            },
        )
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
        scaler_hashes = {
            probe: shared_scaler_file_sha for probe in STRUCTURED_PROBES
        }

        cache: dict[tuple[str, str, str], GroupedScorerData] = {}

        def evaluation_data(
            probe: StructuredProbeName, grounding: str, view: str
        ) -> GroupedScorerData:
            key = (probe, grounding, view)
            if key not in cache:
                cache[key] = assemble_grouped_scorer_data(
                    dataset,
                    probe=probe,
                    model=str(config["data"]["model"]),
                    grounding=grounding,
                    view=view,
                    target=str(config["data"]["target"]),  # type: ignore[arg-type]
                    horizon_index=int(config["data"]["horizon_index"]),
                    allow_ineligible_external_development=True,
                )
            return cache[key]

        states_by_seed: dict[str, dict[str, str]] = {}
        optimization = config["optimization"]
        loss = config["loss"]
        for seed_value in optimization["seeds"]:
            seed = int(seed_value)
            trained_by_probe: dict[str, TrainedStructuredCostAdapter] = {}
            for probe in STRUCTURED_PROBES:
                trained = train_structured_cost_adapter(
                    pooled[probe],
                    model_config=config["model"],
                    seed=seed,
                    steps=int(optimization["steps"]),
                    learning_rate=float(optimization["learning_rate"]),
                    weight_decay=float(optimization["weight_decay"]),
                    gradient_clip_norm=float(optimization["gradient_clip_norm"]),
                    pairwise_weight=float(loss["pairwise_weight"]),
                    huber_weight=float(loss["huber_weight"]),
                    huber_beta=float(loss["huber_beta"]),
                    tie_tolerance=float(loss["tie_tolerance"]),
                    device=str(optimization["device"]),
                )
                if trained.parameter_count != expected_count:
                    raise RuntimeError("trained structured parameter count changed")
                trained_by_probe[probe] = trained
                training_rows.append(
                    {
                        "probe": probe,
                        "seed": seed,
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
            if len({value.parameter_count for value in trained_by_probe.values()}) != 1:
                raise RuntimeError("trained structured probe parameter counts differ")
            states_by_seed[str(seed)] = {
                probe: value.state_sha256 for probe, value in trained_by_probe.items()
            }
            for probe in ("direct", "predicted"):
                for view in ("clean", "rotation5", "translation4"):
                    evaluation_rows.extend(
                        score_structured_validation_split(
                            trained_by_probe[probe],
                            pooled[probe].scaler,
                            evaluation_data(probe, "fcpg", view),
                            view=view,
                            grounding="fcpg",
                        )
                    )
            evaluation_rows.extend(
                score_structured_validation_split(
                    trained_by_probe["oracle"],
                    pooled["oracle"].scaler,
                    evaluation_data("oracle", "fcpg", "clean"),
                    view="clean",
                    grounding="fcpg",
                )
            )
            for view in ("clean", "rotation5", "translation4"):
                evaluation_rows.extend(
                    score_structured_validation_split(
                        trained_by_probe["predicted"],
                        pooled["predicted"].scaler,
                        evaluation_data("predicted", "nominal", view),
                        view=view,
                        grounding="nominal",
                    )
                )
            for view in ("rotation5", "translation4"):
                evaluation_rows.extend(
                    score_structured_validation_split(
                        trained_by_probe["predicted"],
                        pooled["predicted"].scaler,
                        evaluation_data("predicted", "fcpg", view),
                        view=view,
                        grounding="fcpg",
                        variant="cyclic_candidate_future",
                    )
                )

        metrics = summarize_structured_metrics(
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
            "protocol": STRUCTURED_SCORER_PROTOCOL,
            "complete": True,
            "config_sha256": config_sha,
            "source_result_sha256": actual_source,
            "source_eligibility": {
                "g0_eligible": eligibility.g0_eligible,
                "external_development_override": eligibility.external_development_override,
                "ineligibility_reasons": list(eligibility.ineligibility_reasons),
            },
            "code": code,
            "parameter_fairness": {
                "passed": True,
                "counts": parameter_counts,
                "exact_equality": True,
                "shared_scaler_canonical_sha256": scaler_fairness[
                    "shared_scaler_canonical_sha256"
                ],
                "shared_scaler_file_sha256": shared_scaler_file_sha,
                "transformed_shared_inputs_byte_identical": True,
            },
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
            raise RuntimeError("published structured manifest changed after atomic rename")
        return published
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "HorizonStructuredCostAdapter",
    "STRUCTURED_SCORER_PROTOCOL",
    "StructuredCostInputs",
    "StructuredFeatureScaler",
    "StructuredPooledTrainingData",
    "TrainedStructuredCostAdapter",
    "assemble_structured_pooled_training_data",
    "bind_shared_structured_scaler",
    "cyclic_structured_future",
    "fit_structured_feature_scaler",
    "run_structured_cost_experiment",
    "score_structured_validation_split",
    "structured_inputs_from_probe",
    "structured_model_from_config",
    "summarize_structured_metrics",
    "train_structured_cost_adapter",
    "validate_structured_experiment_config",
]
