"""Fail-closed statistics for the zero-training control-sufficiency P0 Gate.

The counterfactual coordinate weights follow Equations 14--15 of Feedback
World Model.  One global vector is fitted on a distinct, deployment-only
same-state action-branch artifact and is then frozen for the scored panel.
The scored rows are never used to estimate coordinate weights.

Rows are paired by ``physical_episode_id x horizon x branch_index``.  Rank
correlations are computed *within* each episode/horizon branch panel and then
macro-averaged.  Confidence intervals resample physical episodes, never
branches, horizons, or model seeds.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from fcpg.lewm_counterfactual import spearman_rank_correlation


PROTOCOL = "geometry_anchored_control_sufficient_p0_v2"
ROW_PANEL_PROTOCOL = "control_sufficient_p0_row_panel_v1"
WEIGHT_FIT_PROTOCOL = "control_sufficient_p0_weight_fit_v1"
OBSERVER_CONTRACT_PROTOCOL = "control_sufficient_p0_observer_contract_v1"
HORIZONS = (3, 4, 5, 6, 7, 8)
MINIMUM_PHYSICAL_EPISODES = 30
WEIGHT_SOURCE = "global_deployment_action_branch_prediction_variance_eq14"
WEIGHT_SCOPE = "one_global_vector_fitted_on_distinct_deployment_only_source"
DEFAULT_WEIGHT_EPSILON = 1.0e-8
DEFAULT_CONTROLLABILITY_BETA = 1.0
WRONG_ACTION_SHIFT_IDS = tuple(
    f"complementary_wrong_action_cyclic_shift_{shift}" for shift in range(1, 8)
)

MINIMUM_CORRELATION_DELTA = 0.10
MINIMUM_OBSERVER_IMPROVEMENT = 0.10
H3_MINIMUM_RELATIVE_IMPROVEMENT = -0.05
MINIMUM_POSITIVE_HORIZONS = 5

REQUIRED_FIELDS = (
    "physical_episode_id",
    "branch_index",
    "horizon",
    "exact_action_error",
    "complementary_observer_action_error",
    "analytic_observer_action_error",
    "matched_no_action_action_error",
    "wrong_action_action_error_by_shift",
    "wrong_action_shift_ids",
    "token_error_scope",
    "offline_labels_score_only",
    "target_leakage_detected",
    "deployment_prediction_uses_offline_target",
    "analytic_observer_scope",
)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _episode_id(value: object, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise TypeError(f"{label} must be a non-empty string or integer")
    result = str(value)
    if not result.strip():
        raise ValueError(f"{label} must be non-empty")
    return result


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _finite_vector(value: object, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 2:
        raise ValueError(f"{label} must be a one-dimensional vector of size >= 2")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result


def normalized_counterfactual_weights(
    fit_branch_predictions: np.ndarray,
    *,
    epsilon: float = DEFAULT_WEIGHT_EPSILON,
    beta: float = DEFAULT_CONTROLLABILITY_BETA,
) -> np.ndarray:
    """Compute one Eq.14/15 global coordinate-weight vector.

    ``fit_branch_predictions`` has shape ``[fit_states, actions, coordinates]``.
    Equation 14 first takes action variance per state and then averages across
    fit states.  Equation 15 is implemented exactly as

    ``w_bar = w / (mean(w) + epsilon)`` and
    ``w_beta = 1 + beta * (w_bar - 1)``.
    """

    predictions = np.asarray(fit_branch_predictions, dtype=np.float64)
    if predictions.ndim != 3:
        raise ValueError(
            "fit_branch_predictions must have shape [fit_states, actions, coordinates]"
        )
    if min(predictions.shape) < 2:
        raise ValueError(
            "weight fit requires at least two states, actions, and coordinates"
        )
    if not bool(np.isfinite(predictions).all()):
        raise ValueError("fit_branch_predictions must be finite")
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a real number")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError("beta must be a real number")
    beta = float(beta)
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")

    coordinate_variance = np.var(predictions, axis=1, ddof=0).mean(axis=0)
    denominator = float(np.mean(coordinate_variance)) + epsilon
    normalized = coordinate_variance / denominator
    weights = 1.0 + beta * (normalized - 1.0)
    if not bool(np.isfinite(weights).all() and np.all(weights >= 0.0)):
        raise RuntimeError("Eq.15 counterfactual weights are invalid")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("weight-fit actions contain no usable coordinate variance")
    return weights


def _validate_weight_fit(
    predictions: np.ndarray | None,
    metadata: Mapping[str, object] | None,
    *,
    scored_episode_ids: Sequence[str],
    scored_observed_state_signatures: Sequence[str],
    epsilon: float,
    beta: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if predictions is None or metadata is None:
        raise ValueError(
            "raw rows require a separate weight-fit artifact and weight-fit metadata"
        )
    if not isinstance(metadata, Mapping):
        raise TypeError("weight_fit_metadata must be a mapping")
    required = {
        "schema_version",
        "protocol",
        "classification",
        "fit_source_id",
        "fit_sample_ids",
        "fit_physical_episode_ids",
        "fit_observed_state_signatures",
        "deployment_only",
        "offline_labels_used",
        "scored_panel_used",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"weight-fit metadata is missing fields: {missing}")
    if (
        metadata["schema_version"] != 1
        or metadata["protocol"] != WEIGHT_FIT_PROTOCOL
        or metadata["classification"] != "deployment_only_distinct_weight_fit"
        or metadata["deployment_only"] is not True
        or metadata["offline_labels_used"] is not False
        or metadata["scored_panel_used"] is not False
    ):
        raise PermissionError("weight-fit metadata does not establish a distinct deployment-only fit")
    source_id = _string(metadata["fit_source_id"], label="fit_source_id")
    fit_array = np.asarray(predictions, dtype=np.float64)
    if fit_array.ndim != 3:
        raise ValueError("weight-fit artifact must have shape [states,actions,coordinates]")
    sample_ids_raw = metadata["fit_sample_ids"]
    episode_ids_raw = metadata["fit_physical_episode_ids"]
    if isinstance(sample_ids_raw, (str, bytes)) or not isinstance(sample_ids_raw, Sequence):
        raise TypeError("fit_sample_ids must be a sequence")
    if isinstance(episode_ids_raw, (str, bytes)) or not isinstance(episode_ids_raw, Sequence):
        raise TypeError("fit_physical_episode_ids must be a sequence")
    sample_ids = tuple(_episode_id(value, label="fit_sample_id") for value in sample_ids_raw)
    fit_episode_ids = tuple(
        _episode_id(value, label="fit_physical_episode_id") for value in episode_ids_raw
    )
    fit_state_signatures_raw = metadata["fit_observed_state_signatures"]
    if isinstance(fit_state_signatures_raw, (str, bytes)) or not isinstance(
        fit_state_signatures_raw, Sequence
    ):
        raise TypeError("fit_observed_state_signatures must be a sequence")
    fit_state_signatures = tuple(
        _string(value, label="fit_observed_state_signature")
        for value in fit_state_signatures_raw
    )
    if len(sample_ids) != fit_array.shape[0] or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("fit_sample_ids must uniquely identify every fit state")
    if not fit_episode_ids or len(set(fit_episode_ids)) != len(fit_episode_ids):
        raise ValueError("fit_physical_episode_ids must be non-empty and unique")
    if (
        len(fit_state_signatures) != fit_array.shape[0]
        or len(set(fit_state_signatures)) != len(fit_state_signatures)
    ):
        raise ValueError(
            "fit_observed_state_signatures must uniquely identify every fit state"
        )
    overlap = sorted(set(fit_episode_ids) & set(scored_episode_ids))
    if overlap:
        raise PermissionError(
            f"weight-fit and scored physical episodes must be disjoint; overlap={overlap[:3]}"
        )
    state_overlap = sorted(
        set(fit_state_signatures) & set(scored_observed_state_signatures)
    )
    if state_overlap:
        raise PermissionError(
            "weight-fit and scored observed-state signatures must be disjoint; "
            f"overlap={state_overlap[:3]}"
        )
    weights = normalized_counterfactual_weights(
        fit_array, epsilon=epsilon, beta=beta
    )
    provenance: dict[str, object] = {
        "source": WEIGHT_SOURCE,
        "scope": WEIGHT_SCOPE,
        "fit_source_id": source_id,
        "fit_state_count": int(fit_array.shape[0]),
        "fit_action_count": int(fit_array.shape[1]),
        "coordinate_count": int(fit_array.shape[2]),
        "fit_sample_ids_sha256": _canonical_digest(list(sample_ids)),
        "fit_physical_episode_ids_sha256": _canonical_digest(list(fit_episode_ids)),
        "fit_physical_episode_count": len(fit_episode_ids),
        "fit_observed_state_signatures_sha256": _canonical_digest(
            list(fit_state_signatures)
        ),
        "scored_observed_state_signatures_sha256": _canonical_digest(
            sorted(set(scored_observed_state_signatures))
        ),
        "fit_scored_observed_state_overlap_count": 0,
        "observed_state_nonoverlap_verified": True,
        "fit_array_sha256": _array_digest(fit_array),
        "weight_vector_sha256": _array_digest(weights),
        "epsilon": epsilon,
        "beta": beta,
    }
    artifact_sha = metadata.get("artifact_sha256")
    if artifact_sha is not None:
        if not _is_sha256(artifact_sha):
            raise ValueError("weight-fit artifact_sha256 is invalid")
        provenance["artifact_sha256"] = artifact_sha
    return weights, provenance


def _validate_observer_contract(metadata: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "analytic observer requires globally frozen non-target-cheating contract metadata"
        )
    required = {
        "schema_version",
        "protocol",
        "classification",
        "parameter_source",
        "frozen_before_offline_label_scoring",
        "target_tuned_per_row",
        "target_tuned_per_horizon",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"observer contract metadata is missing fields: {missing}")
    if (
        metadata["schema_version"] != 1
        or metadata["protocol"] != OBSERVER_CONTRACT_PROTOCOL
        or metadata["classification"] != "exposed_p0_global_fixed_parameters"
        or metadata["frozen_before_offline_label_scoring"] is not True
        or metadata["target_tuned_per_row"] is not False
        or metadata["target_tuned_per_horizon"] is not False
    ):
        raise PermissionError(
            "observer contract does not establish globally frozen non-target-cheating scope"
        )
    _string(metadata["parameter_source"], label="observer parameter_source")
    return {
        "protocol": OBSERVER_CONTRACT_PROTOCOL,
        "classification": "exposed_p0_global_fixed_parameters",
        "parameter_source": metadata["parameter_source"],
        "contract_sha256": _canonical_digest(dict(metadata)),
        "frozen_before_offline_label_scoring": True,
        "target_tuned_per_row": False,
        "target_tuned_per_horizon": False,
    }


def _precomputed_weight_provenance(
    records: Sequence[Mapping[str, Any]], *, epsilon: float, beta: float
) -> dict[str, object]:
    required = (
        "counterfactual_weight_fit_sha256",
        "counterfactual_weight_vector_sha256",
        "counterfactual_weight_fit_source_id",
        "counterfactual_weight_fit_ids_sha256",
        "counterfactual_weight_fit_count",
        "weight_fit_observed_state_signatures_sha256",
        "scored_observed_state_signatures_sha256",
        "weight_fit_scored_observed_state_overlap_count",
        "observed_state_signature_contract",
        "counterfactual_weight_fit_source_artifact_sha256",
    )
    signatures: set[tuple[object, ...]] = set()
    for row in records:
        if row.get("counterfactual_weight_source") != WEIGHT_SOURCE:
            raise ValueError("precomputed metric does not declare the global Eq.14 source")
        if row.get("counterfactual_weight_scope") != WEIGHT_SCOPE:
            raise ValueError("precomputed metric does not declare one distinct global vector")
        observed_epsilon = _finite_nonnegative(
            row.get("counterfactual_weight_epsilon"),
            label="counterfactual_weight_epsilon",
        )
        observed_beta = _finite_nonnegative(
            row.get("counterfactual_weight_beta"), label="counterfactual_weight_beta"
        )
        if observed_epsilon != epsilon or observed_beta != beta:
            raise ValueError("precomputed counterfactual weight contract mismatch")
        if any(field not in row for field in required):
            raise ValueError("precomputed metric is missing global weight-fit provenance")
        fit_count = _integer(
            row["counterfactual_weight_fit_count"],
            label="counterfactual_weight_fit_count",
        )
        if fit_count < 2:
            raise ValueError("counterfactual_weight_fit_count must be at least two")
        digests = (
            row["counterfactual_weight_fit_sha256"],
            row["counterfactual_weight_vector_sha256"],
            row["counterfactual_weight_fit_ids_sha256"],
            row["weight_fit_observed_state_signatures_sha256"],
            row["scored_observed_state_signatures_sha256"],
            row["counterfactual_weight_fit_source_artifact_sha256"],
        )
        if not all(_is_sha256(value) for value in digests):
            raise ValueError("precomputed global weight-fit digest is invalid")
        overlap_count = _integer(
            row["weight_fit_scored_observed_state_overlap_count"],
            label="weight_fit_scored_observed_state_overlap_count",
        )
        if overlap_count != 0:
            raise PermissionError("weight-fit and scored observed-state signatures overlap")
        signature_contract = _string(
            row["observed_state_signature_contract"],
            label="observed_state_signature_contract",
        )
        signatures.add(
            (
                *digests,
                _string(
                    row["counterfactual_weight_fit_source_id"],
                    label="counterfactual_weight_fit_source_id",
                ),
                fit_count,
                signature_contract,
            )
        )
    if len(signatures) != 1:
        raise ValueError("precomputed rows do not share one global weight vector")
    (
        fit_sha,
        vector_sha,
        ids_sha,
        fit_state_sha,
        scored_state_sha,
        source_artifact_sha,
        source_id,
        fit_count,
        signature_contract,
    ) = next(iter(signatures))
    return {
        "source": WEIGHT_SOURCE,
        "scope": WEIGHT_SCOPE,
        "fit_source_id": source_id,
        "fit_state_count": fit_count,
        "fit_ids_sha256": ids_sha,
        "fit_artifact_sha256": fit_sha,
        "fit_source_artifact_sha256": source_artifact_sha,
        "weight_vector_sha256": vector_sha,
        "fit_observed_state_signatures_sha256": fit_state_sha,
        "scored_observed_state_signatures_sha256": scored_state_sha,
        "fit_scored_observed_state_overlap_count": 0,
        "observed_state_signature_contract": signature_contract,
        "observed_state_nonoverlap_verified": True,
        "epsilon": epsilon,
        "beta": beta,
    }


def _normalize_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_physical_episodes: int,
    weight_epsilon: float,
    controllability_beta: float,
    weight_fit_predictions: np.ndarray | None,
    weight_fit_metadata: Mapping[str, object] | None,
) -> dict[str, Any]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise TypeError("rows must be a non-empty sequence of mappings")
    if minimum_physical_episodes < 2:
        raise ValueError("minimum_physical_episodes must be at least 2")

    records: list[dict[str, Any]] = []
    identities: set[tuple[str, int, int]] = set()
    raw_mode: bool | None = None
    raw_vector_dimensions: set[int] = set()
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping")
        missing = [field for field in REQUIRED_FIELDS if field not in source]
        if missing:
            raise ValueError(f"rows[{index}] is missing fields: {missing}")
        if source["offline_labels_score_only"] is not True:
            raise ValueError(f"rows[{index}] does not isolate offline labels to scoring")
        if source["target_leakage_detected"] is not False:
            raise ValueError(f"rows[{index}] reports target leakage")
        if source["deployment_prediction_uses_offline_target"] is not False:
            raise ValueError(f"rows[{index}] deployment prediction uses an offline target")
        if (
            source["analytic_observer_scope"]
            != "globally_frozen_missing_measurement_extension"
        ):
            raise ValueError(f"rows[{index}] analytic observer scope mismatch")
        if (
            source["token_error_scope"]
            != "same_two_slot_anchor_carrier_as_exact_action_intervention"
        ):
            raise ValueError(f"rows[{index}] token error scope mismatch")

        episode = _episode_id(
            source["physical_episode_id"], label=f"rows[{index}].physical_episode_id"
        )
        branch = _integer(source["branch_index"], label=f"rows[{index}].branch_index")
        horizon = _integer(source["horizon"], label=f"rows[{index}].horizon")
        if branch < 0:
            raise ValueError(f"rows[{index}].branch_index must be non-negative")
        if horizon not in HORIZONS:
            raise ValueError(f"rows[{index}] has unexpected horizon H{horizon}")
        identity = (episode, branch, horizon)
        if identity in identities:
            raise ValueError(f"duplicate P0 identity: {identity!r}")
        identities.add(identity)

        has_raw = "deployment_prediction_vector" in source or "prediction_error_vector" in source
        if has_raw != (
            "deployment_prediction_vector" in source and "prediction_error_vector" in source
        ):
            raise ValueError(f"rows[{index}] must provide both raw prediction vectors")
        if raw_mode is None:
            raw_mode = has_raw
        elif raw_mode != has_raw:
            raise ValueError("raw and precomputed metric rows cannot be mixed")

        record: dict[str, Any] = {
            "episode": episode,
            "branch": branch,
            "horizon": horizon,
            "exact": _finite_nonnegative(
                source["exact_action_error"], label=f"rows[{index}].exact_action_error"
            ),
            "complementary": _finite_nonnegative(
                source["complementary_observer_action_error"],
                label=f"rows[{index}].complementary_observer_action_error",
            ),
            "analytic": _finite_nonnegative(
                source["analytic_observer_action_error"],
                label=f"rows[{index}].analytic_observer_action_error",
            ),
            "matched_no_action": _finite_nonnegative(
                source["matched_no_action_action_error"],
                label=f"rows[{index}].matched_no_action_action_error",
            ),
        }
        if has_raw:
            record["observed_state_signature"] = _string(
                source.get("observed_state_signature"),
                label=f"rows[{index}].observed_state_signature",
            )
        wrong_action = np.asarray(
            source["wrong_action_action_error_by_shift"], dtype=np.float64
        )
        shift_ids_raw = source["wrong_action_shift_ids"]
        if isinstance(shift_ids_raw, (str, bytes)) or not isinstance(
            shift_ids_raw, Sequence
        ):
            raise TypeError(f"rows[{index}].wrong_action_shift_ids must be a sequence")
        shift_ids = tuple(str(value) for value in shift_ids_raw)
        if shift_ids != WRONG_ACTION_SHIFT_IDS:
            raise ValueError(f"rows[{index}].wrong_action_shift_ids contract drifted")
        if wrong_action.shape != (7,):
            raise ValueError(
                f"rows[{index}].wrong_action_action_error_by_shift must have shape [7]"
            )
        if not bool(np.isfinite(wrong_action).all() and np.all(wrong_action >= 0.0)):
            raise ValueError(
                f"rows[{index}].wrong_action_action_error_by_shift must be finite and non-negative"
            )
        record["wrong_action"] = wrong_action
        if has_raw:
            record["deployment_prediction_vector"] = _finite_vector(
                source["deployment_prediction_vector"],
                label=f"rows[{index}].deployment_prediction_vector",
            )
            record["prediction_error_vector"] = _finite_vector(
                source["prediction_error_vector"],
                label=f"rows[{index}].prediction_error_vector",
            )
            raw_vector_dimensions.update(
                (
                    record["deployment_prediction_vector"].size,
                    record["prediction_error_vector"].size,
                )
            )
        else:
            record.update(source)
        records.append(record)

    if raw_mode and len(raw_vector_dimensions) != 1:
        raise ValueError("all raw prediction and error vectors must share one dimension")

    episodes = tuple(sorted({row["episode"] for row in records}))
    if len(episodes) < minimum_physical_episodes:
        raise ValueError(f"P0 requires at least {minimum_physical_episodes} physical episodes")
    branch_sets: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in records:
        branch_sets[(row["episode"], row["horizon"])].add(row["branch"])
    expected_groups = {(episode, horizon) for episode in episodes for horizon in HORIZONS}
    if set(branch_sets) != expected_groups:
        missing = sorted(expected_groups - set(branch_sets))
        raise ValueError(f"incomplete episode/horizon panel; missing={missing[:3]}")
    branch_panels = {tuple(sorted(value)) for value in branch_sets.values()}
    if len(branch_panels) != 1:
        raise ValueError("every episode/horizon cell must contain the same branch indices")
    branch_order = next(iter(branch_panels))
    if len(branch_order) < 2:
        raise ValueError("P0 requires at least two same-state action branches")

    if raw_mode:
        scored_state_signatures = tuple(
            sorted({row["observed_state_signature"] for row in records})
        )
        weights, weight_provenance = _validate_weight_fit(
            weight_fit_predictions,
            weight_fit_metadata,
            scored_episode_ids=episodes,
            scored_observed_state_signatures=scored_state_signatures,
            epsilon=weight_epsilon,
            beta=controllability_beta,
        )
        dimension = next(iter(raw_vector_dimensions))
        if weights.shape != (dimension,):
            raise ValueError("global weight vector dimension differs from scored raw vectors")
        for row in records:
            squared = np.square(row["prediction_error_vector"])
            row["uniform"] = float(np.mean(squared))
            row["weighted"] = float(np.mean(squared * weights))
    else:
        if weight_fit_predictions is not None or weight_fit_metadata is not None:
            raise ValueError("precomputed rows must not silently replace their frozen weight fit")
        weight_provenance = _precomputed_weight_provenance(
            records, epsilon=weight_epsilon, beta=controllability_beta
        )
        for row in records:
            row["uniform"] = _finite_nonnegative(
                row.get("uniform_token_error"), label="uniform_token_error"
            )
            row["weighted"] = _finite_nonnegative(
                row.get("counterfactual_weighted_error"),
                label="counterfactual_weighted_error",
            )

    records.sort(key=lambda row: (row["episode"], row["horizon"], row["branch"]))
    arrays = {
        name: np.asarray([row[name] for row in records], dtype=np.float64)
        for name in (
            "uniform",
            "weighted",
            "exact",
            "complementary",
            "analytic",
            "matched_no_action",
            "wrong_action",
        )
    }
    return {
        "records": records,
        "arrays": arrays,
        "episodes": episodes,
        "branches": branch_order,
        "raw_mode": bool(raw_mode),
        "weight_provenance": weight_provenance,
    }


def _spearman(left: np.ndarray, right: np.ndarray, *, label: str) -> float:
    if np.ptp(left) <= 0.0 or np.ptp(right) <= 0.0:
        raise ValueError(f"{label} contains a constant metric")
    result = float(spearman_rank_correlation(left, right))
    if not math.isfinite(result):
        raise ValueError(f"{label} produced a non-finite rank correlation")
    return result


def _panel_control_statistics(
    records: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[(row["episode"], row["horizon"])].append(index)
    panels: list[dict[str, object]] = []
    uniform_values: list[float] = []
    weighted_values: list[float] = []
    delta_values: list[float] = []
    for (episode, horizon), raw_indices in sorted(grouped.items()):
        indices = np.asarray(raw_indices, dtype=np.int64)
        uniform = _spearman(
            arrays["uniform"][indices], arrays["exact"][indices],
            label=f"uniform panel {episode}/H{horizon}",
        )
        weighted = _spearman(
            arrays["weighted"][indices], arrays["exact"][indices],
            label=f"weighted panel {episode}/H{horizon}",
        )
        delta = weighted - uniform
        panels.append(
            {"episode": episode, "horizon": horizon, "uniform": uniform,
             "weighted": weighted, "delta": delta}
        )
        uniform_values.append(uniform)
        weighted_values.append(weighted)
        delta_values.append(delta)
    return (
        panels,
        np.asarray(uniform_values, dtype=np.float64),
        np.asarray(weighted_values, dtype=np.float64),
        np.asarray(delta_values, dtype=np.float64),
    )


def _relative_improvement(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.mean(reference))
    if denominator <= 0.0:
        raise ValueError("observer reference action error must have a positive mean")
    return float(1.0 - float(np.mean(candidate)) / denominator)


def _mean_error_reduction(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return positive values when ``candidate`` has lower error."""

    return float(np.mean(reference) - np.mean(candidate))


def summarize_control_sufficient_p0(
    rows: Sequence[Mapping[str, object]],
    *,
    weight_fit_predictions: np.ndarray | None = None,
    weight_fit_metadata: Mapping[str, object] | None = None,
    observer_contract_metadata: Mapping[str, object] | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 61_071,
    minimum_physical_episodes: int = MINIMUM_PHYSICAL_EPISODES,
    weight_epsilon: float = DEFAULT_WEIGHT_EPSILON,
    controllability_beta: float = DEFAULT_CONTROLLABILITY_BETA,
) -> dict[str, object]:
    """Validate, score, and decide one zero-training control-sufficiency Gate."""

    bootstrap_samples = _integer(bootstrap_samples, label="bootstrap_samples")
    bootstrap_seed = _integer(bootstrap_seed, label="bootstrap_seed")
    minimum_physical_episodes = _integer(
        minimum_physical_episodes, label="minimum_physical_episodes"
    )
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")
    weight_epsilon = float(weight_epsilon)
    controllability_beta = float(controllability_beta)
    observer_contract = _validate_observer_contract(observer_contract_metadata)
    normalized = _normalize_rows(
        rows,
        minimum_physical_episodes=minimum_physical_episodes,
        weight_epsilon=weight_epsilon,
        controllability_beta=controllability_beta,
        weight_fit_predictions=weight_fit_predictions,
        weight_fit_metadata=weight_fit_metadata,
    )
    records = normalized["records"]
    arrays = normalized["arrays"]
    episodes = normalized["episodes"]

    panels, panel_uniform, panel_weighted, panel_delta = _panel_control_statistics(
        records, arrays
    )
    uniform_rho = float(np.mean(panel_uniform))
    weighted_rho = float(np.mean(panel_weighted))
    rho_delta = float(np.mean(panel_delta))
    observer_improvement = _relative_improvement(
        arrays["analytic"], arrays["complementary"]
    )
    matched_no_action_reduction = _mean_error_reduction(
        arrays["exact"], arrays["matched_no_action"]
    )
    wrong_action_reductions = np.asarray(
        [
            _mean_error_reduction(arrays["exact"], arrays["wrong_action"][:, shift])
            for shift in range(7)
        ],
        dtype=np.float64,
    )

    episode_to_panel_delta: dict[str, np.ndarray] = {}
    for episode in episodes:
        episode_to_panel_delta[episode] = np.asarray(
            [panel["delta"] for panel in panels if panel["episode"] == episode],
            dtype=np.float64,
        )
        if episode_to_panel_delta[episode].shape != (len(HORIZONS),):
            raise RuntimeError("episode does not contain one control panel per horizon")
    by_episode_indices = {
        episode: np.asarray(
            [index for index, row in enumerate(records) if row["episode"] == episode],
            dtype=np.int64,
        )
        for episode in episodes
    }
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(
        0, len(episodes), size=(bootstrap_samples, len(episodes)), dtype=np.int64
    )
    rho_deltas = np.empty(bootstrap_samples, dtype=np.float64)
    observer_effects = np.empty(bootstrap_samples, dtype=np.float64)
    matched_no_action_effects = np.empty(bootstrap_samples, dtype=np.float64)
    wrong_action_effects = np.empty((bootstrap_samples, 7), dtype=np.float64)
    for draw_index, draw in enumerate(draws):
        selected = [episodes[index] for index in draw]
        rho_deltas[draw_index] = float(
            np.mean(np.concatenate([episode_to_panel_delta[item] for item in selected]))
        )
        row_indices = np.concatenate([by_episode_indices[item] for item in selected])
        observer_effects[draw_index] = _relative_improvement(
            arrays["analytic"][row_indices], arrays["complementary"][row_indices]
        )
        matched_no_action_effects[draw_index] = _mean_error_reduction(
            arrays["exact"][row_indices], arrays["matched_no_action"][row_indices]
        )
        for shift in range(7):
            wrong_action_effects[draw_index, shift] = _mean_error_reduction(
                arrays["exact"][row_indices],
                arrays["wrong_action"][row_indices, shift],
            )

    horizon_effects: dict[str, float] = {}
    positive_horizons = 0
    for horizon in HORIZONS:
        indices = np.asarray(
            [index for index, row in enumerate(records) if row["horizon"] == horizon],
            dtype=np.int64,
        )
        effect = _relative_improvement(
            arrays["analytic"][indices], arrays["complementary"][indices]
        )
        horizon_effects[f"H{horizon}"] = effect
        if effect > 0.0:
            positive_horizons += 1

    rho_ci = [float(np.quantile(rho_deltas, 0.025)), float(np.quantile(rho_deltas, 0.975))]
    observer_ci = [
        float(np.quantile(observer_effects, 0.025)),
        float(np.quantile(observer_effects, 0.975)),
    ]
    rho_pass = rho_delta >= MINIMUM_CORRELATION_DELTA and rho_ci[0] > 0.0
    observer_pass = (
        observer_improvement >= MINIMUM_OBSERVER_IMPROVEMENT
        and observer_ci[0] > 0.0
    )
    h3_pass = horizon_effects["H3"] >= H3_MINIMUM_RELATIVE_IMPROVEMENT
    horizon_pass = positive_horizons >= MINIMUM_POSITIVE_HORIZONS
    matched_no_action_direction_pass = matched_no_action_reduction > 0.0
    wrong_action_direction_passes = wrong_action_reductions > 0.0
    causal_controls_pass = matched_no_action_direction_pass and bool(
        np.all(wrong_action_direction_passes)
    )
    passed = (
        rho_pass
        and observer_pass
        and h3_pass
        and horizon_pass
        and causal_controls_pass
    )

    matched_no_action_ci = [
        float(np.quantile(matched_no_action_effects, 0.025)),
        float(np.quantile(matched_no_action_effects, 0.975)),
    ]
    wrong_action_reports = []
    for shift in range(7):
        wrong_action_reports.append(
            {
                "shift_id": WRONG_ACTION_SHIFT_IDS[shift],
                "factual_mean_action_error": float(np.mean(arrays["exact"])),
                "control_mean_action_error": float(
                    np.mean(arrays["wrong_action"][:, shift])
                ),
                "mean_action_error_reduction": float(wrong_action_reductions[shift]),
                "relative_action_error_improvement": _relative_improvement(
                    arrays["exact"], arrays["wrong_action"][:, shift]
                ),
                "episode_bootstrap_95_ci": [
                    float(np.quantile(wrong_action_effects[:, shift], 0.025)),
                    float(np.quantile(wrong_action_effects[:, shift], 0.975)),
                ],
                "point_direction_passed": bool(wrong_action_direction_passes[shift]),
            }
        )

    result: dict[str, object] = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "classification": "exposed_zero_training_diagnostic",
        "physical_episode_count": len(episodes),
        "branch_count": len(normalized["branches"]),
        "row_count": len(records),
        "horizons": list(HORIZONS),
        "physical_episode_ids_sha256": _canonical_digest(list(episodes)),
        "counterfactual_weights": {
            **normalized["weight_provenance"],
            "equation_14": "mean_over_fit_states(var_over_actions(prediction))",
            "equation_15": "w_bar=w/(mean(w)+epsilon); w_beta=1+beta*(w_bar-1)",
            "computed_from_separate_raw_fit_artifact": normalized["raw_mode"],
        },
        "analytic_observer_contract": observer_contract,
        "bootstrap": {
            "unit": "physical_episode",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "draw_matrix_sha256": hashlib.sha256(
                np.ascontiguousarray(draws).tobytes()
            ).hexdigest(),
            "two_sided_95_quantiles": [0.025, 0.975],
        },
        "control_metric": {
            "aggregation": "spearman_within_episode_horizon_then_panel_macro_average",
            "token_error_scope": (
                "same_two_slot_anchor_carrier_as_exact_action_intervention"
            ),
            "panel_count": len(panels),
            "uniform_spearman_macro_vs_exact_action_error": uniform_rho,
            "counterfactual_weighted_spearman_macro_vs_exact_action_error": weighted_rho,
            "spearman_delta": rho_delta,
            "bootstrap_95_ci": rho_ci,
            "panel_effects_sha256": _canonical_digest(panels),
        },
        "analytic_feedback_observer": {
            "scope": "globally_frozen_missing_measurement_extension",
            "relative_action_error_improvement_vs_complementary": observer_improvement,
            "bootstrap_95_ci": observer_ci,
            "relative_improvement_by_horizon": horizon_effects,
            "positive_horizon_count": positive_horizons,
        },
        "causal_action_controls": {
            "criterion": (
                "point estimate only: factual exact action error must be lower; "
                "episode-bootstrap intervals are diagnostic and are not Gate thresholds"
            ),
            "matched_no_action": {
                "factual_mean_action_error": float(np.mean(arrays["exact"])),
                "control_mean_action_error": float(
                    np.mean(arrays["matched_no_action"])
                ),
                "mean_action_error_reduction": matched_no_action_reduction,
                "relative_action_error_improvement": _relative_improvement(
                    arrays["exact"], arrays["matched_no_action"]
                ),
                "episode_bootstrap_95_ci": matched_no_action_ci,
                "point_direction_passed": matched_no_action_direction_pass,
            },
            "wrong_action_shifts": wrong_action_reports,
            "wrong_action_pass_count": int(np.sum(wrong_action_direction_passes)),
            "all_point_directions_passed": causal_controls_pass,
        },
        "gate": {
            "minimum_spearman_delta": MINIMUM_CORRELATION_DELTA,
            "spearman_delta_ci_lower_must_exceed": 0.0,
            "minimum_observer_relative_improvement": MINIMUM_OBSERVER_IMPROVEMENT,
            "observer_ci_lower_must_exceed": 0.0,
            "h3_minimum_relative_improvement": H3_MINIMUM_RELATIVE_IMPROVEMENT,
            "minimum_positive_horizons": MINIMUM_POSITIVE_HORIZONS,
            "causal_action_control_criterion": (
                "factual point estimate lower than matched no-action and each of 7 wrong shifts"
            ),
            "control_metric_passed": rho_pass,
            "observer_improvement_passed": observer_pass,
            "h3_noninferiority_passed": h3_pass,
            "positive_horizon_count_passed": horizon_pass,
            "matched_no_action_direction_passed": matched_no_action_direction_pass,
            "wrong_action_direction_pass_count": int(
                np.sum(wrong_action_direction_passes)
            ),
            "causal_action_controls_passed": causal_controls_pass,
            "passed": passed,
            "decision": "P0-Go" if passed else "P0-No-Go",
        },
        "claim_boundary": {
            "analytic_observer_prediction_uses_offline_target": False,
            "missing_measurement_routing_is_deployable": False,
            "missing_measurement_extension_is_from_feedback_wm_paper": False,
            "paper_always_observed_convergence_theorem_inherited": False,
            "eq14_fit_action_pool_matches_feedback_wm_paper": False,
            "offline_labels_are_scoring_only": True,
            "matched_2k_authorized_if_passed": True,
            "policy_rollout_authorized": False,
            "ten_thousand_step_training_authorized": False,
        },
    }
    result["result_sha256"] = _canonical_digest(result)
    return result


__all__ = [
    "DEFAULT_CONTROLLABILITY_BETA",
    "DEFAULT_WEIGHT_EPSILON",
    "HORIZONS",
    "MINIMUM_PHYSICAL_EPISODES",
    "OBSERVER_CONTRACT_PROTOCOL",
    "PROTOCOL",
    "ROW_PANEL_PROTOCOL",
    "WEIGHT_FIT_PROTOCOL",
    "WEIGHT_SCOPE",
    "WEIGHT_SOURCE",
    "WRONG_ACTION_SHIFT_IDS",
    "normalized_counterfactual_weights",
    "summarize_control_sufficient_p0",
]
