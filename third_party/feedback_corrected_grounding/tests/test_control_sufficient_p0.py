"""Tests for the zero-training control-sufficiency P0 statistics."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from fcpg.control_sufficient_p0 import (
    DEFAULT_CONTROLLABILITY_BETA,
    DEFAULT_WEIGHT_EPSILON,
    HORIZONS,
    OBSERVER_CONTRACT_PROTOCOL,
    WEIGHT_FIT_PROTOCOL,
    WEIGHT_SCOPE,
    WEIGHT_SOURCE,
    WRONG_ACTION_SHIFT_IDS,
    normalized_counterfactual_weights,
    summarize_control_sufficient_p0,
)
from fcpg.lewm_counterfactual import spearman_rank_correlation


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _observer_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": OBSERVER_CONTRACT_PROTOCOL,
        "classification": "exposed_p0_global_fixed_parameters",
        "parameter_source": "pre-registered exposed development contract",
        "frozen_before_offline_label_scoring": True,
        "target_tuned_per_row": False,
        "target_tuned_per_horizon": False,
        "feedback_gain": 1.0,
        "residual_decay": 0.9,
    }


def _weight_fit() -> tuple[np.ndarray, dict[str, object]]:
    states, actions, coordinates = 12, 4, 3
    predictions = np.zeros((states, actions, coordinates), dtype=np.float64)
    for state in range(states):
        predictions[state, :, 0] = np.arange(actions) * (1.0 + state / 20.0)
        predictions[state, :, 1] = 0.1 * state
        predictions[state, :, 2] = -0.2 * state
    metadata: dict[str, object] = {
        "schema_version": 1,
        "protocol": WEIGHT_FIT_PROTOCOL,
        "classification": "deployment_only_distinct_weight_fit",
        "fit_source_id": "strict-train-development-action-pool-v1",
        "fit_sample_ids": [f"fit-sample-{index:03d}" for index in range(states)],
        "fit_physical_episode_ids": [f"fit-episode-{index:03d}" for index in range(6)],
        "fit_observed_state_signatures": [
            f"fit-state-signature-{index:03d}" for index in range(states)
        ],
        "deployment_only": True,
        "offline_labels_used": False,
        "scored_panel_used": False,
    }
    return predictions, metadata


def _rows(*, episodes: int = 30, raw: bool = True) -> list[dict[str, object]]:
    fit, _ = _weight_fit()
    weights = normalized_counterfactual_weights(fit)
    rows: list[dict[str, object]] = []
    for episode in range(episodes):
        episode_scale = 1.0 + episode / 20.0
        for horizon in HORIZONS:
            for branch in range(4):
                exact = episode_scale * (0.05 + 0.04 * branch + 0.01 * (horizon - 3))
                prediction = np.asarray(
                    [0.5 * branch * episode_scale, 0.2 * episode_scale, 0.1 * horizon]
                )
                # Coordinate 0 follows exact policy error; coordinate 1 moves in
                # the opposite branch order and obscures it under a uniform norm.
                squared_error = np.asarray(
                    [exact, 1.0 - 0.2 * branch, 0.2 + 0.01 * horizon]
                )
                error = np.sqrt(squared_error)
                complementary = exact + 0.20
                analytic = 0.70 * complementary
                row: dict[str, object] = {
                    "physical_episode_id": f"score-episode-{episode:03d}",
                    "branch_index": branch,
                    "horizon": horizon,
                    "exact_action_error": exact,
                    "complementary_observer_action_error": complementary,
                    "analytic_observer_action_error": analytic,
                    "matched_no_action_action_error": exact + 0.15,
                    "wrong_action_action_error_by_shift": [
                        exact + 0.20 + 0.01 * shift for shift in range(7)
                    ],
                    "wrong_action_shift_ids": list(WRONG_ACTION_SHIFT_IDS),
                    "offline_labels_score_only": True,
                    "target_leakage_detected": False,
                    "deployment_prediction_uses_offline_target": False,
                    "analytic_observer_scope": (
                        "globally_frozen_missing_measurement_extension"
                    ),
                    "token_error_scope": (
                        "same_two_slot_anchor_carrier_as_exact_action_intervention"
                    ),
                }
                if raw:
                    row["observed_state_signature"] = (
                        f"score-state-signature-{episode:03d}-H{horizon}"
                    )
                    row["deployment_prediction_vector"] = prediction.tolist()
                    row["prediction_error_vector"] = error.tolist()
                else:
                    row.update(
                        {
                            "uniform_token_error": float(np.mean(squared_error)),
                            "counterfactual_weighted_error": float(
                                np.mean(squared_error * weights)
                            ),
                            "counterfactual_weight_source": WEIGHT_SOURCE,
                            "counterfactual_weight_scope": WEIGHT_SCOPE,
                            "counterfactual_weight_epsilon": DEFAULT_WEIGHT_EPSILON,
                            "counterfactual_weight_beta": DEFAULT_CONTROLLABILITY_BETA,
                            "counterfactual_weight_fit_sha256": _sha("fit-artifact"),
                            "counterfactual_weight_vector_sha256": _sha("weight-vector"),
                            "counterfactual_weight_fit_source_id": "fit-source-v1",
                            "counterfactual_weight_fit_ids_sha256": _sha("fit-ids"),
                            "counterfactual_weight_fit_count": 12,
                            "weight_fit_observed_state_signatures_sha256": _sha(
                                "fit-state-signatures"
                            ),
                            "scored_observed_state_signatures_sha256": _sha(
                                "scored-state-signatures"
                            ),
                            "weight_fit_scored_observed_state_overlap_count": 0,
                            "observed_state_signature_contract": (
                                "sha256_shape_float32_causal_H0_H1_H2_anchor_history"
                            ),
                            "counterfactual_weight_fit_source_artifact_sha256": _sha(
                                "fit-source-artifact"
                            ),
                        }
                    )
                rows.append(row)
    return rows


def _summarize(
    rows: list[dict[str, object]], *, raw: bool = True
) -> dict[str, object]:
    fit, metadata = _weight_fit()
    return summarize_control_sufficient_p0(
        rows,
        weight_fit_predictions=fit if raw else None,
        weight_fit_metadata=metadata if raw else None,
        observer_contract_metadata=_observer_contract(),
        bootstrap_samples=300,
        bootstrap_seed=19,
    )


def test_weights_implement_exact_equations_14_and_15() -> None:
    predictions = np.asarray(
        [
            [[0.0, 2.0], [2.0, 2.0], [4.0, 2.0]],
            [[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]],
        ]
    )
    variance = np.var(predictions, axis=1).mean(axis=0)
    normalized = variance / (variance.mean() + DEFAULT_WEIGHT_EPSILON)
    expected = 1.0 + DEFAULT_CONTROLLABILITY_BETA * (normalized - 1.0)
    weights = normalized_counterfactual_weights(predictions)
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=0.0)
    assert weights[0] > weights[1] >= 0.0


def test_raw_panel_passes_all_frozen_gates_with_global_fit() -> None:
    result = _summarize(_rows())

    assert result["physical_episode_count"] == 30
    assert result["row_count"] == 30 * 6 * 4
    provenance = result["counterfactual_weights"]
    assert provenance["computed_from_separate_raw_fit_artifact"] is True
    assert provenance["beta"] == 1.0
    assert len(provenance["weight_vector_sha256"]) == 64
    assert len(provenance["fit_sample_ids_sha256"]) == 64
    assert result["control_metric"]["aggregation"].startswith("spearman_within")
    assert result["control_metric"]["spearman_delta"] >= 0.10
    assert result["control_metric"]["bootstrap_95_ci"][0] > 0.0
    assert result["analytic_feedback_observer"][
        "relative_action_error_improvement_vs_complementary"
    ] == pytest.approx(0.30)
    assert result["analytic_feedback_observer"]["positive_horizon_count"] == 6
    assert result["claim_boundary"][
        "paper_always_observed_convergence_theorem_inherited"
    ] is False
    assert result["gate"]["decision"] == "P0-Go"
    assert result["gate"]["matched_no_action_direction_passed"] is True
    assert result["gate"]["wrong_action_direction_pass_count"] == 7
    assert result["causal_action_controls"]["all_point_directions_passed"] is True
    assert len(
        result["causal_action_controls"]["matched_no_action"][
            "episode_bootstrap_95_ci"
        ]
    ) == 2
    assert len(result["result_sha256"]) == 64


def test_raw_panel_fails_closed_without_distinct_weight_fit() -> None:
    with pytest.raises(ValueError, match="separate weight-fit artifact"):
        summarize_control_sufficient_p0(
            _rows(),
            observer_contract_metadata=_observer_contract(),
            bootstrap_samples=100,
        )


def test_weight_fit_must_be_deployment_only_and_episode_disjoint() -> None:
    fit, metadata = _weight_fit()
    metadata["offline_labels_used"] = True
    with pytest.raises(PermissionError, match="deployment-only"):
        summarize_control_sufficient_p0(
            _rows(),
            weight_fit_predictions=fit,
            weight_fit_metadata=metadata,
            observer_contract_metadata=_observer_contract(),
            bootstrap_samples=100,
        )

    fit, metadata = _weight_fit()
    metadata["fit_physical_episode_ids"] = ["score-episode-000"]
    with pytest.raises(PermissionError, match="must be disjoint"):
        summarize_control_sufficient_p0(
            _rows(),
            weight_fit_predictions=fit,
            weight_fit_metadata=metadata,
            observer_contract_metadata=_observer_contract(),
            bootstrap_samples=100,
        )

    fit, metadata = _weight_fit()
    metadata["fit_observed_state_signatures"][0] = (
        "score-state-signature-000-H3"
    )
    with pytest.raises(PermissionError, match="observed-state signatures must be disjoint"):
        summarize_control_sufficient_p0(
            _rows(),
            weight_fit_predictions=fit,
            weight_fit_metadata=metadata,
            observer_contract_metadata=_observer_contract(),
            bootstrap_samples=100,
        )


def test_precomputed_metrics_require_one_global_vector_provenance() -> None:
    rows = _rows(raw=False)
    result = _summarize(rows, raw=False)
    assert result["counterfactual_weights"][
        "computed_from_separate_raw_fit_artifact"
    ] is False

    rows[0]["counterfactual_weight_scope"] = "per_episode_horizon_fit"
    with pytest.raises(ValueError, match="distinct global vector"):
        _summarize(rows, raw=False)

    rows = _rows(raw=False)
    rows[0]["counterfactual_weight_vector_sha256"] = _sha("different-vector")
    with pytest.raises(ValueError, match="share one global weight vector"):
        _summarize(rows, raw=False)


def test_panel_macro_scoring_rejects_cross_panel_confound() -> None:
    rows = _rows()
    for row in rows:
        horizon = int(row["horizon"])
        branch = int(row["branch_index"])
        exact = 100.0 * horizon + branch
        weighted_squared = 100.0 * horizon - branch
        desired_uniform = 1000.0 + branch
        nuisance = (3.0 * desired_uniform - weighted_squared) / 2.0
        row["exact_action_error"] = exact
        row["prediction_error_vector"] = np.sqrt(
            [weighted_squared, nuisance, nuisance]
        ).tolist()

    fit, metadata = _weight_fit()
    weights = normalized_counterfactual_weights(fit)
    squared = np.square(
        np.asarray([row["prediction_error_vector"] for row in rows], dtype=np.float64)
    )
    exact = np.asarray([row["exact_action_error"] for row in rows])
    pooled_uniform = spearman_rank_correlation(squared.mean(axis=1), exact)
    pooled_weighted = spearman_rank_correlation((squared * weights).mean(axis=1), exact)
    assert pooled_weighted - pooled_uniform > 0.10

    result = summarize_control_sufficient_p0(
        rows,
        weight_fit_predictions=fit,
        weight_fit_metadata=metadata,
        observer_contract_metadata=_observer_contract(),
        bootstrap_samples=300,
        bootstrap_seed=7,
    )
    assert result["control_metric"]["spearman_delta"] < 0.0
    assert result["gate"]["control_metric_passed"] is False
    assert result["gate"]["passed"] is False


def test_gate_enforces_observer_h3_and_five_of_six_horizons() -> None:
    rows = _rows()
    for row in rows:
        if row["horizon"] == 3:
            row["analytic_observer_action_error"] = (
                1.06 * float(row["complementary_observer_action_error"])
            )
        elif row["horizon"] == 4:
            row["analytic_observer_action_error"] = (
                1.01 * float(row["complementary_observer_action_error"])
            )
    result = _summarize(rows)
    assert result["gate"]["h3_noninferiority_passed"] is False
    assert result["gate"]["positive_horizon_count_passed"] is False
    assert result["gate"]["passed"] is False


def test_gate_enforces_matched_no_action_and_all_seven_wrong_directions() -> None:
    rows = _rows()
    for row in rows:
        row["matched_no_action_action_error"] = max(
            0.0, float(row["exact_action_error"]) - 0.01
        )
    result = _summarize(rows)
    assert result["gate"]["matched_no_action_direction_passed"] is False
    assert result["gate"]["causal_action_controls_passed"] is False
    assert result["gate"]["passed"] is False

    rows = _rows()
    for row in rows:
        controls = list(row["wrong_action_action_error_by_shift"])
        controls[4] = max(0.0, float(row["exact_action_error"]) - 0.01)
        row["wrong_action_action_error_by_shift"] = controls
    result = _summarize(rows)
    assert result["gate"]["wrong_action_direction_pass_count"] == 6
    assert result["causal_action_controls"]["wrong_action_shifts"][4][
        "point_direction_passed"
    ] is False
    assert result["gate"]["passed"] is False


def test_observer_scope_requires_global_pre_score_contract() -> None:
    fit, metadata = _weight_fit()
    with pytest.raises(ValueError, match="globally frozen"):
        summarize_control_sufficient_p0(
            _rows(),
            weight_fit_predictions=fit,
            weight_fit_metadata=metadata,
            bootstrap_samples=100,
        )

    contract = _observer_contract()
    contract["target_tuned_per_horizon"] = True
    with pytest.raises(PermissionError, match="non-target-cheating"):
        summarize_control_sufficient_p0(
            _rows(),
            weight_fit_predictions=fit,
            weight_fit_metadata=metadata,
            observer_contract_metadata=contract,
            bootstrap_samples=100,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("offline_labels_score_only", False, "isolate offline labels"),
        ("target_leakage_detected", True, "target leakage"),
        ("deployment_prediction_uses_offline_target", True, "uses an offline target"),
    ],
)
def test_fails_closed_on_target_leakage_flags(
    field: str, value: object, match: str
) -> None:
    rows = _rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match=match):
        _summarize(rows)


def test_fails_closed_on_duplicate_missing_nonfinite_and_small_panels() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate P0 identity"):
        _summarize([*rows, copy.deepcopy(rows[0])])

    with pytest.raises(ValueError, match="incomplete episode/horizon panel"):
        _summarize(rows[:-4])

    rows = _rows()
    rows[0]["exact_action_error"] = np.nan
    with pytest.raises(ValueError, match="finite and non-negative"):
        _summarize(rows)

    with pytest.raises(ValueError, match="at least 30 physical episodes"):
        _summarize(_rows(episodes=29))


def test_fails_closed_on_constant_panel_and_branch_schema_drift() -> None:
    rows = _rows()
    for row in rows:
        row["exact_action_error"] = 1.0
    with pytest.raises(ValueError, match="constant metric"):
        _summarize(rows)

    rows = _rows()
    del rows[0]["prediction_error_vector"]
    with pytest.raises(ValueError, match="both raw prediction vectors"):
        _summarize(rows)

    rows = _rows()
    rows[0]["prediction_error_vector"] = [0.1, 0.2, 0.3, 0.4]
    with pytest.raises(ValueError, match="share one dimension"):
        _summarize(rows)

    rows = _rows()
    rows[0]["branch_index"] = 99
    with pytest.raises(ValueError, match="same branch indices"):
        _summarize(rows)

    rows = _rows()
    rows[0]["wrong_action_action_error_by_shift"] = [0.1] * 6
    with pytest.raises(ValueError, match=r"shape \[7\]"):
        _summarize(rows)

    rows = _rows()
    shift_ids = list(rows[0]["wrong_action_shift_ids"])
    shift_ids[0], shift_ids[1] = shift_ids[1], shift_ids[0]
    rows[0]["wrong_action_shift_ids"] = shift_ids
    with pytest.raises(ValueError, match="shift_ids contract drifted"):
        _summarize(rows)
