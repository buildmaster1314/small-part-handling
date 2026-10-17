"""End-to-end exposed P0 evaluation for policy-native anchor predictions.

The evaluator keeps three identities separate:

* deployment-side anchor predictions, which never receive a clean target;
* the frozen policy carrier, held fixed while anchor overrides are compared; and
* clean anchor tokens, loaded from a separate archive only for offline scoring.

Three frozen predictor seeds are reduced to one seed-macro row for each
``physical episode x branch x horizon`` cell.  This preserves the physical
episode as the statistical unit expected by :mod:`fcpg.control_sufficient_p0`
instead of pretending that model seeds create additional independent episodes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from fcpg.control_sufficient_p0 import (
    DEFAULT_CONTROLLABILITY_BETA,
    HORIZONS,
    OBSERVER_CONTRACT_PROTOCOL,
    ROW_PANEL_PROTOCOL,
    WRONG_ACTION_SHIFT_IDS,
)
from fcpg.lewm_analytic_feedback_observer import propagate_held_feedback_residual
from fcpg.lewm_p1_native_complementary import P1_NATIVE_COMPLEMENTARY_GAINS
from fcpg.policy_token_sensitivity import probe_policy_token_sensitivity


PROTOCOL: Final = ROW_PANEL_PROTOCOL
ANALYTIC_SCOPE: Final = "globally_frozen_missing_measurement_extension"
PREDICTOR_SEEDS: Final = (15801, 15802, 15803)
CONTROLLABILITY_BETA: Final = DEFAULT_CONTROLLABILITY_BETA
EXPECTED_PRE_DROP_PRIOR_CONTRACT: Final = (
    "predict visible H2 from causal H-1,H0,H1 plus executed "
    "H-1->H0,H0->H1,H1->H2 action blocks"
)

_SEED_INVARIANT_CARRIER_FIELDS: Final = (
    "schema_version",
    "protocol",
    "classification",
    "policy_row_id",
    "episode_uid",
    "physical_episode_id",
    "reset_physical_state_goal_digest",
    "observed_state_signature",
    "branch_index",
    "horizon",
    "policy_obs",
    "policy_q_nominal",
    "policy_keypoints_xyz",
    "policy_sampled_visual",
    "policy_global_visual",
    "policy_image_size",
    "policy_fcpg_q_corrected",
    "policy_fcpg_reliability",
    "pre_drop_history_horizons",
    "pre_drop_target_horizon",
    "pre_drop_history_token",
    "pre_drop_action_context",
    "pre_drop_visible_token",
    "pre_drop_visible_visual",
    "same_state_prefix",
    "policy_future_action",
    "policy_geometry_token",
    "recursive_inputs_are_from_identity_verified_same_replay",
)


@dataclass(frozen=True, slots=True)
class FrozenAnalyticObserverContract:
    """Globally frozen parameters for the missing-measurement extension.

    The source paper defines a feedback gain but does not define how its
    innovation should be held through a complete visual dropout.  Therefore
    both the scalar gain and this project's residual decay must be fixed before
    offline labels are scored.  Per-row or per-horizon target tuning is rejected.
    """

    feedback_gain: float
    residual_decay: float
    delta_t: float
    controllability_beta: float
    parameter_source: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SeedPanel:
    seed: int
    checkpoint_sha256: str
    raw_prediction: np.ndarray
    complementary_prediction: np.ndarray
    analytic_prediction: np.ndarray
    raw_action_error: np.ndarray
    complementary_action_error: np.ndarray
    analytic_action_error: np.ndarray
    matched_no_action_action_error: np.ndarray
    wrong_action_action_error: np.ndarray

    def __post_init__(self) -> None:
        if self.seed not in PREDICTOR_SEEDS:
            raise ValueError("seed panel is not one of the three frozen P1 seeds")
        if not _is_sha256(self.checkpoint_sha256):
            raise ValueError("seed panel checkpoint digest is invalid")
        predictions = (
            np.asarray(self.raw_prediction),
            np.asarray(self.complementary_prediction),
            np.asarray(self.analytic_prediction),
        )
        if any(value.ndim != 4 or value.shape[1] != 2 for value in predictions):
            raise ValueError(
                "seed predictions must have shape [rows,raw_adjacent_history,anchors,latent]"
            )
        if len({value.shape for value in predictions}) != 1:
            raise ValueError("all seed prediction variants must share one shape")
        row_count = int(predictions[0].shape[0])
        for name in (
            "raw_action_error",
            "complementary_action_error",
            "analytic_action_error",
            "matched_no_action_action_error",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (row_count,):
                raise ValueError(f"{name} must contain one value per prediction row")
            if not bool(np.isfinite(value).all() and np.all(value >= 0.0)):
                raise ValueError(f"{name} must be finite and non-negative")
        wrong = np.asarray(self.wrong_action_action_error)
        if wrong.shape != (row_count, 7):
            raise ValueError("wrong_action_action_error must have shape [rows,7]")
        if not bool(np.isfinite(wrong).all() and np.all(wrong >= 0.0)):
            raise ValueError("wrong_action_action_error must be finite and non-negative")
        if any(not bool(np.isfinite(value).all()) for value in predictions):
            raise ValueError("seed prediction contains non-finite values")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float64_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate_seed_invariant_carriers(
    deployments: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    """Require all model seeds to score the exact same physical carrier rows.

    Predictor outputs and their causal one-step priors are seed-specific.  All
    physical identities, frozen-policy inputs, pre-drop histories, executed
    actions and geometry measurements must otherwise be byte-identical before
    the offline-label archive is opened.
    """

    if tuple(sorted(int(seed) for seed in deployments)) != PREDICTOR_SEEDS:
        raise ValueError("carrier set must contain exactly the three frozen seeds")
    reference_seed = PREDICTOR_SEEDS[0]
    reference = deployments[reference_seed]
    for field in _SEED_INVARIANT_CARRIER_FIELDS:
        if field not in reference:
            raise KeyError(f"seed {reference_seed} carrier is missing {field}")
        expected = np.asarray(reference[field])
        for seed in PREDICTOR_SEEDS[1:]:
            if field not in deployments[seed]:
                raise KeyError(f"seed {seed} carrier is missing {field}")
            observed = np.asarray(deployments[seed][field])
            if observed.shape != expected.shape or not np.array_equal(observed, expected):
                raise PermissionError(
                    f"seed {seed} carrier differs from seed {reference_seed} for {field}"
                )
    for seed in PREDICTOR_SEEDS:
        receipt = np.asarray(deployments[seed].get("predictor_seed"))
        if receipt.shape != (1,) or receipt.dtype.kind not in "iu" or int(receipt[0]) != seed:
            raise PermissionError(f"carrier predictor-seed receipt does not bind seed {seed}")


def load_frozen_observer_contract(
    path: str | Path | None,
    *,
    expected_sha256: str | None,
) -> FrozenAnalyticObserverContract:
    """Load one globally frozen observer contract or fail closed.

    Requiring an externally supplied digest prevents a caller from silently
    editing the contract after inspecting the hidden-clean labels.
    """

    if path is None:
        raise PermissionError(
            "analytic observer is unscorable: no frozen observer contract was "
            "provided; feedback_gain and dropout residual_decay are not identified"
        )
    if not _is_sha256(expected_sha256):
        raise PermissionError(
            "analytic observer is unscorable: an expected observer-contract SHA-256 "
            "must be fixed before offline-label scoring"
        )
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_sha = _sha256_file(source)
    if actual_sha != expected_sha256:
        raise PermissionError("observer-contract SHA-256 differs from the frozen digest")
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol",
        "classification",
        "feedback_gain",
        "residual_decay",
        "delta_t",
        "controllability_beta",
        "parameter_source",
        "frozen_before_offline_label_scoring",
        "target_tuned_per_row",
        "target_tuned_per_horizon",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("observer contract schema drifted")
    if (
        payload["schema_version"] != 1
        or payload["protocol"] != OBSERVER_CONTRACT_PROTOCOL
        or payload["classification"] != "exposed_p0_global_fixed_parameters"
        or payload["frozen_before_offline_label_scoring"] is not True
        or payload["target_tuned_per_row"] is not False
        or payload["target_tuned_per_horizon"] is not False
    ):
        raise PermissionError("observer contract does not establish global pre-score freezing")
    values: dict[str, float] = {}
    for name in (
        "feedback_gain",
        "residual_decay",
        "delta_t",
        "controllability_beta",
    ):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"observer contract {name} must be numeric")
        values[name] = float(value)
        if not math.isfinite(values[name]):
            raise ValueError(f"observer contract {name} must be finite")
    if not 0.0 < values["delta_t"] * values["feedback_gain"] < 2.0:
        raise ValueError("observer contract violates 0 < delta_t * gain < 2")
    if not 0.0 <= values["residual_decay"] <= 1.0:
        raise ValueError("observer contract residual_decay must lie in [0,1]")
    if values["controllability_beta"] != CONTROLLABILITY_BETA:
        raise ValueError("P0 freezes Eq.15 controllability_beta to exactly 1")
    parameter_source = payload["parameter_source"]
    if not isinstance(parameter_source, str) or not parameter_source.strip():
        raise ValueError("observer contract parameter_source must be non-empty")
    return FrozenAnalyticObserverContract(
        feedback_gain=values["feedback_gain"],
        residual_decay=values["residual_decay"],
        delta_t=values["delta_t"],
        controllability_beta=values["controllability_beta"],
        parameter_source=parameter_source,
        sha256=actual_sha,
    )


def _validate_frozen_predictor(model: nn.Module) -> None:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("analytic rollout requires a frozen eval predictor")


@torch.inference_mode()
def recursive_analytic_feedback_complementary_rollout(
    model: nn.Module,
    history_token: torch.Tensor,
    history_action: torch.Tensor,
    future_action: torch.Tensor,
    geometry_token: torch.Tensor,
    final_visible_measurement: torch.Tensor,
    causal_final_visible_prior: torch.Tensor,
    *,
    observer: FrozenAnalyticObserverContract,
    complementary_gains: Sequence[float] = P1_NATIVE_COMPLEMENTARY_GAINS,
) -> torch.Tensor:
    """Roll out a causal held-residual Feedback-WM plus geometry observer.

    The clean H3-H8 target is deliberately absent from the signature.  The
    pre-drop innovation is formed once from a causal H2 prior and the actually
    observed H2 token, then decayed through the known missing interval.  This
    hold/decay operation is a project-specific extension, not an equation from
    the cited Feedback-WM paper.
    """

    _validate_frozen_predictor(model)
    if history_token.ndim != 4:
        raise ValueError("history_token must have shape [B,T,K,D]")
    batch, history, anchors, latent = (int(value) for value in history_token.shape)
    if history < 2:
        raise ValueError("analytic rollout requires at least two history states")
    expected_action = (batch, history - 1, int(getattr(model, "action_dim")))
    if tuple(history_action.shape) != expected_action:
        raise ValueError(f"history_action must have shape {expected_action}")
    if future_action.ndim != 4 or int(future_action.shape[0]) != batch:
        raise ValueError("future_action must have shape [B,branches,horizon,action]")
    branches, horizon = (int(value) for value in future_action.shape[1:3])
    if horizon != len(complementary_gains):
        raise ValueError("complementary gain count differs from rollout horizon")
    expected_geometry = (batch, branches, horizon, anchors, latent)
    if tuple(geometry_token.shape) != expected_geometry:
        raise ValueError(f"geometry_token must have shape {expected_geometry}")
    visible_shape = (batch, anchors, latent)
    if tuple(final_visible_measurement.shape) != visible_shape:
        raise ValueError("final_visible_measurement shape drifted")
    if tuple(causal_final_visible_prior.shape) != visible_shape:
        raise ValueError("causal_final_visible_prior shape drifted")
    tensors = (
        history_token,
        history_action,
        future_action,
        geometry_token,
        final_visible_measurement,
        causal_final_visible_prior,
    )
    if len({value.device for value in tensors}) != 1:
        raise ValueError("analytic rollout tensors must share one device")
    if any(not torch.is_floating_point(value) for value in tensors):
        raise TypeError("analytic rollout tensors must be floating point")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("analytic rollout input contains non-finite values")

    innovation = final_visible_measurement - causal_final_visible_prior
    embeddings = history_token[:, None].expand(-1, branches, -1, -1, -1).clone()
    actions = history_action[:, None].expand(-1, branches, -1, -1).clone()
    belief = causal_final_visible_prior[:, None].expand(-1, branches, -1, -1).clone()
    held_innovation = innovation[:, None].expand(-1, branches, -1, -1)
    outputs: list[torch.Tensor] = []
    for index, raw_gain in enumerate(complementary_gains):
        current_action = future_action[:, :, index : index + 1]
        action_context = torch.cat((actions, current_action), dim=2)
        flat_history = embeddings.reshape(batch * branches, history, anchors, latent)
        flat_action = action_context.reshape(
            batch * branches, history, int(getattr(model, "action_dim"))
        )
        predicted = model.predict(flat_history, flat_action)
        if tuple(predicted.shape) != tuple(flat_history.shape):
            raise ValueError("predictor output shape differs from recurrent context")
        raw_prior = predicted[:, -1].reshape(batch, branches, anchors, latent)
        feedback = propagate_held_feedback_residual(
            embeddings[:, :, -1],
            belief,
            raw_prior,
            held_innovation,
            feedback_gain=observer.feedback_gain,
            missing_age=index + 1,
            residual_decay=observer.residual_decay,
            delta_t=observer.delta_t,
        )
        gain = float(raw_gain)
        if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
            raise ValueError("complementary gains must lie in [0,1]")
        baseline = geometry_token[:, :, index]
        posterior = baseline + gain * (feedback.corrected_prediction - baseline)
        outputs.append(posterior)
        embeddings = torch.cat((embeddings[:, :, 1:], posterior[:, :, None]), dim=2)
        actions = action_context[:, :, 1:]
        belief = feedback.next_belief
    return torch.stack(outputs, dim=2)


def _slice_tensor_batch(
    arrays: Mapping[str, np.ndarray],
    start: int,
    stop: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    names = (
        "policy_obs",
        "policy_q_nominal",
        "policy_keypoints_xyz",
        "policy_sampled_visual",
        "policy_global_visual",
        "policy_image_size",
        "policy_fcpg_q_corrected",
    )
    result: dict[str, torch.Tensor] = {}
    for source, target in (
        ("policy_obs", "obs"),
        ("policy_q_nominal", "q_nominal"),
        ("policy_keypoints_xyz", "keypoints_xyz"),
        ("policy_sampled_visual", "sampled_visual"),
        ("policy_global_visual", "global_visual"),
        ("policy_image_size", "image_size"),
        ("policy_fcpg_q_corrected", "fcpg_q_corrected"),
    ):
        if source not in arrays:
            raise KeyError(f"policy carrier is missing {source}")
        value = np.ascontiguousarray(np.asarray(arrays[source])[start:stop])
        result[target] = torch.from_numpy(value).to(device)
    assert set(names).issubset(arrays)
    return result


def exact_carrier_held_fixed_action_error(
    policy: nn.Module,
    carrier: Mapping[str, np.ndarray],
    predicted_anchor: np.ndarray,
    target_anchor: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int,
    action_steps: int = 5,
) -> np.ndarray:
    """Return one exact policy action MSE per row with the carrier held fixed."""

    if batch_size <= 0:
        raise ValueError("policy batch_size must be positive")
    predicted = np.asarray(predicted_anchor, dtype=np.float32)
    target = np.asarray(target_anchor, dtype=np.float32)
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("policy anchor arrays must share shape [rows,history,K,D]")
    if not bool(np.isfinite(predicted).all() and np.isfinite(target).all()):
        raise ValueError("policy anchor array contains non-finite values")
    target_device = torch.device(device)
    values: list[np.ndarray] = []
    for start in range(0, int(predicted.shape[0]), batch_size):
        stop = min(start + batch_size, int(predicted.shape[0]))
        report = probe_policy_token_sensitivity(
            policy,
            _slice_tensor_batch(carrier, start, stop, device=target_device),
            torch.from_numpy(np.ascontiguousarray(predicted[start:stop])).to(target_device),
            torch.from_numpy(np.ascontiguousarray(target[start:stop])).to(target_device),
            action_steps=action_steps,
            jacobian_samples=0,
        )
        values.append(np.asarray(report["exact_action_mse"]["per_sample"], dtype=np.float64))
    result = np.concatenate(values)
    if result.shape != (predicted.shape[0],):
        raise RuntimeError("exact action evaluator returned an incomplete panel")
    return result


def repeat_endpoint_for_policy(prediction: np.ndarray, *, history: int = 2) -> np.ndarray:
    value = np.asarray(prediction, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError("endpoint prediction must have shape [rows,K,D]")
    if history <= 0:
        raise ValueError("policy history must be positive")
    return np.repeat(value[:, None], history, axis=1)


def fit_eq14_style_global_weights(
    fit_predictions_by_seed: Mapping[int, np.ndarray],
    *,
    beta: float = CONTROLLABILITY_BETA,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Fit one global Eq.14-style vector on the exposed strict cache.

    Each array must be ``[states, local_branches, horizons, anchors, latent]``.
    Variance is taken over the local action branches, then averaged globally
    over states, horizons, and the three frozen predictor seeds.  This is a
    project-specific local-branch estimator: the paper instead sampled actions
    from a state-independent demonstration action pool.
    """

    if tuple(sorted(int(seed) for seed in fit_predictions_by_seed)) != PREDICTOR_SEEDS:
        raise ValueError("weight fitting requires all three frozen predictor seeds")
    if float(beta) != CONTROLLABILITY_BETA:
        raise ValueError("P0 freezes Eq.15 beta to exactly 1")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("weight epsilon must be finite and positive")
    terms: list[np.ndarray] = []
    reference_tail: tuple[int, int] | None = None
    for seed in PREDICTOR_SEEDS:
        value = np.asarray(fit_predictions_by_seed[seed], dtype=np.float64)
        if value.ndim != 5 or value.shape[1] < 2 or value.shape[2] != len(HORIZONS):
            raise ValueError(
                "fit predictions must have shape [states,branches,H3-H8,anchors,latent]"
            )
        if not bool(np.isfinite(value).all()):
            raise ValueError("fit prediction contains non-finite values")
        tail = (int(value.shape[-2]), int(value.shape[-1]))
        if reference_tail is None:
            reference_tail = tail
        elif tail != reference_tail:
            raise ValueError("fit predictor token dimensions differ across seeds")
        # [state,H,K,D], one variance per same-state local branch bank.
        terms.append(np.var(value, axis=1, ddof=0))
    global_variance = np.mean(np.concatenate(terms, axis=0), axis=(0, 1))
    normalized = global_variance / (float(np.mean(global_variance)) + epsilon)
    weights = 1.0 + float(beta) * (normalized - 1.0)
    if not bool(np.isfinite(weights).all() and np.all(weights >= 0.0)):
        raise RuntimeError("global Eq.15 weight fit produced invalid values")
    return weights.astype(np.float64, copy=False)


def build_seed_macro_rows(
    identity: Mapping[str, np.ndarray],
    target_anchor: np.ndarray,
    seed_panels: Sequence[SeedPanel],
    *,
    frozen_controllability_weights: np.ndarray,
    weight_fit_sha256: str,
    weight_fit_source_id: str,
    weight_fit_ids_sha256: str,
    weight_fit_count: int,
    weight_fit_observed_state_signatures_sha256: str,
    scored_observed_state_signatures_sha256: str,
    weight_fit_scored_signature_overlap_count: int,
    weight_fit_source_artifact_sha256: str,
    controllability_beta: float = CONTROLLABILITY_BETA,
    weight_epsilon: float = 1.0e-8,
) -> list[dict[str, object]]:
    """Build one statistics-compatible row per physical episode/branch/H."""

    panels = sorted(seed_panels, key=lambda item: item.seed)
    if tuple(item.seed for item in panels) != PREDICTOR_SEEDS:
        raise ValueError("P0 row panel requires each frozen predictor seed exactly once")
    target = np.asarray(target_anchor, dtype=np.float32)
    if target.ndim != 4 or target.shape[1] != 2:
        raise ValueError("offline target must have shape [rows,2,anchors,latent]")
    prediction_shape = panels[0].raw_prediction.shape
    if target.shape != prediction_shape:
        raise ValueError("offline target shape differs from endpoint predictions")
    count = int(target.shape[0])
    required_identity = ("physical_episode_id", "branch_index", "horizon")
    for name in required_identity:
        if name not in identity or np.asarray(identity[name]).shape != (count,):
            raise ValueError(f"identity field {name} must contain one value per row")
    if tuple(sorted(set(np.asarray(identity["horizon"]).astype(int).tolist()))) != HORIZONS:
        raise ValueError("P0 identity horizons differ from H3-H8")
    if isinstance(controllability_beta, bool) or not isinstance(
        controllability_beta, (int, float)
    ):
        raise TypeError("controllability_beta must be numeric")
    beta = float(controllability_beta)
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("controllability_beta must lie in [0,1]")
    epsilon = float(weight_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("weight_epsilon must be finite and positive")

    raw_stack = np.stack([panel.raw_prediction for panel in panels], axis=1)
    error_stack = raw_stack - target[:, None]
    raw_action = np.stack([panel.raw_action_error for panel in panels], axis=1)
    complementary_action = np.stack(
        [panel.complementary_action_error for panel in panels], axis=1
    )
    analytic_action = np.stack([panel.analytic_action_error for panel in panels], axis=1)
    no_action = np.stack(
        [panel.matched_no_action_action_error for panel in panels], axis=1
    )
    wrong_action = np.stack(
        [panel.wrong_action_action_error for panel in panels], axis=1
    )
    weights = np.asarray(frozen_controllability_weights, dtype=np.float64)
    if weights.shape != target.shape[-2:]:
        raise ValueError("frozen controllability weight shape differs from anchor tokens")
    if not bool(np.isfinite(weights).all() and np.all(weights >= 0.0)):
        raise RuntimeError("Eq.15 controllability weights are invalid")
    if not _is_sha256(weight_fit_sha256):
        raise ValueError("weight fit requires a frozen source SHA-256")
    if not isinstance(weight_fit_source_id, str) or not weight_fit_source_id.strip():
        raise ValueError("weight_fit_source_id must be non-empty")
    if not _is_sha256(weight_fit_ids_sha256):
        raise ValueError("weight_fit_ids_sha256 must be a SHA-256")
    if isinstance(weight_fit_count, bool) or not isinstance(weight_fit_count, int):
        raise TypeError("weight_fit_count must be an integer")
    if weight_fit_count < 2:
        raise ValueError("weight_fit_count must be at least two")
    for label, digest in (
        (
            "weight_fit_observed_state_signatures_sha256",
            weight_fit_observed_state_signatures_sha256,
        ),
        ("scored_observed_state_signatures_sha256", scored_observed_state_signatures_sha256),
        ("weight_fit_source_artifact_sha256", weight_fit_source_artifact_sha256),
    ):
        if not _is_sha256(digest):
            raise ValueError(f"{label} must be a SHA-256")
    if weight_fit_scored_signature_overlap_count != 0:
        raise PermissionError("weight-fit and scored observed-state signatures overlap")
    weight_vector_sha256 = _float64_array_sha256(weights)
    rows: list[dict[str, object]] = []
    for index in range(count):
        per_seed = {
            str(panel.seed): {
                "checkpoint_sha256": panel.checkpoint_sha256,
                "raw_factual_action_error": float(raw_action[index, seed_index]),
                "complementary_observer_action_error": float(
                    complementary_action[index, seed_index]
                ),
                "analytic_observer_action_error": float(
                    analytic_action[index, seed_index]
                ),
                "matched_no_action_action_error": float(
                    no_action[index, seed_index]
                ),
                "wrong_action_action_errors": wrong_action[
                    index, seed_index
                ].tolist(),
                "wrong_action_action_error_by_shift": wrong_action[
                    index, seed_index
                ].tolist(),
                "wrong_action_shift_ids": list(WRONG_ACTION_SHIFT_IDS),
            }
            for seed_index, panel in enumerate(panels)
        }
        rows.append(
            {
                "physical_episode_id": str(np.asarray(identity["physical_episode_id"])[index]),
                "branch_index": int(np.asarray(identity["branch_index"])[index]),
                "horizon": int(np.asarray(identity["horizon"])[index]),
                "exact_action_error": float(np.mean(raw_action[index])),
                "complementary_observer_action_error": float(
                    np.mean(complementary_action[index])
                ),
                "analytic_observer_action_error": float(np.mean(analytic_action[index])),
                "matched_no_action_action_error": float(np.mean(no_action[index])),
                "wrong_action_action_errors": np.mean(
                    wrong_action[index], axis=0
                ).tolist(),
                "wrong_action_action_error_by_shift": np.mean(
                    wrong_action[index], axis=0
                ).tolist(),
                "wrong_action_shift_ids": list(WRONG_ACTION_SHIFT_IDS),
                "wrong_action_action_error_macro": float(np.mean(wrong_action[index])),
                "factual_minus_matched_no_action_action_error": float(
                    np.mean(raw_action[index]) - np.mean(no_action[index])
                ),
                "factual_minus_wrong_action_error_by_shift": (
                    np.mean(raw_action[index]) - np.mean(wrong_action[index], axis=0)
                ).tolist(),
                "uniform_token_error": float(np.mean(np.square(error_stack[index]))),
                "counterfactual_weighted_error": float(
                    np.mean(np.square(error_stack[index]) * weights[None])
                ),
                "token_error_scope": (
                    "same_two_slot_anchor_carrier_as_exact_action_intervention"
                ),
                "counterfactual_weight_source": (
                    "global_deployment_action_branch_prediction_variance_eq14"
                ),
                "counterfactual_weight_epsilon": epsilon,
                "counterfactual_weight_beta": beta,
                "counterfactual_weight_scope": (
                    "one_global_vector_fitted_on_distinct_deployment_only_source"
                ),
                "counterfactual_weight_fit_sha256": weight_fit_sha256,
                "counterfactual_weight_vector_sha256": weight_vector_sha256,
                "counterfactual_weight_fit_source_id": weight_fit_source_id,
                "counterfactual_weight_fit_ids_sha256": weight_fit_ids_sha256,
                "counterfactual_weight_fit_count": weight_fit_count,
                "counterfactual_weight_fit_observed_state_signatures_sha256": (
                    weight_fit_observed_state_signatures_sha256
                ),
                "scored_observed_state_signatures_sha256": (
                    scored_observed_state_signatures_sha256
                ),
                "counterfactual_weight_fit_scored_signature_overlap_count": 0,
                "counterfactual_weight_fit_source_artifact_sha256": (
                    weight_fit_source_artifact_sha256
                ),
                "weight_fit_observed_state_signatures_sha256": (
                    weight_fit_observed_state_signatures_sha256
                ),
                "scored_observed_state_signatures_sha256": (
                    scored_observed_state_signatures_sha256
                ),
                "weight_fit_scored_observed_state_overlap_count": 0,
                "observed_state_signature_contract": (
                    "sha256_shape_float32_causal_H0_H1_H2_anchor_history"
                ),
                "counterfactual_weight_estimator": (
                    "project_specific_local_branch_eq14_style_not_paper_action_pool"
                ),
                # Raw vectors remain in differently named audit fields so the
                # legacy summarizer consumes the precomputed global metric and
                # does not accidentally recompute per-cell weights.
                "raw_deployment_prediction_vector": raw_stack[index].reshape(-1).tolist(),
                "raw_prediction_error_vector": error_stack[index].reshape(-1).tolist(),
                "offline_labels_score_only": True,
                "target_leakage_detected": False,
                "deployment_prediction_uses_offline_target": False,
                "analytic_observer_scope": ANALYTIC_SCOPE,
                "action_error_scope": "carrier_held_fixed_exact_frozen_policy",
                "predictor_seed_reduction": "arithmetic_mean_over_three_fixed_model_seeds",
                "per_seed": per_seed,
            }
        )
    return rows


__all__ = [
    "ANALYTIC_SCOPE",
    "CONTROLLABILITY_BETA",
    "EXPECTED_PRE_DROP_PRIOR_CONTRACT",
    "HORIZONS",
    "OBSERVER_CONTRACT_PROTOCOL",
    "PREDICTOR_SEEDS",
    "PROTOCOL",
    "WRONG_ACTION_SHIFT_IDS",
    "FrozenAnalyticObserverContract",
    "SeedPanel",
    "build_seed_macro_rows",
    "exact_carrier_held_fixed_action_error",
    "fit_eq14_style_global_weights",
    "load_frozen_observer_contract",
    "recursive_analytic_feedback_complementary_rollout",
    "repeat_endpoint_for_policy",
    "validate_seed_invariant_carriers",
]
