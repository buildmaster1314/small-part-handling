"""End-to-end contract tests for the control-sufficient P0 summarizer CLI."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.control_sufficient_p0 import WRONG_ACTION_SHIFT_IDS
from scripts import summarize_control_sufficient_p0 as cli


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _observer() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": "control_sufficient_p0_observer_contract_v1",
        "classification": "exposed_p0_global_fixed_parameters",
        "feedback_gain": 1.0,
        "residual_decay": 1.0,
        "delta_t": 1.0,
        "controllability_beta": 1.0,
        "parameter_source": "unit fixed before opening labels",
        "frozen_before_offline_label_scoring": True,
        "target_tuned_per_row": False,
        "target_tuned_per_horizon": False,
    }


def _rows(weights: np.ndarray) -> list[dict[str, object]]:
    weight_vector_sha = cli._float64_array_sha256(weights)
    rows: list[dict[str, object]] = []
    for episode in range(30):
        episode_scale = 1.0 + episode / 100.0
        for horizon in range(3, 9):
            for branch in range(4):
                exact = episode_scale * (0.1 + 0.1 * branch + 0.01 * horizon)
                weighted_error = exact
                uniform_error = 1.0 - 0.1 * branch + 0.001 * episode
                rows.append(
                    {
                        "physical_episode_id": f"episode-{episode:03d}",
                        "branch_index": branch,
                        "horizon": horizon,
                        "exact_action_error": exact,
                        "complementary_observer_action_error": exact + 0.5,
                        "analytic_observer_action_error": exact + 0.2,
                        "matched_no_action_action_error": exact + 0.3,
                        "wrong_action_action_error_by_shift": [
                            exact + 0.4 + 0.01 * shift for shift in range(7)
                        ],
                        "wrong_action_shift_ids": list(WRONG_ACTION_SHIFT_IDS),
                        "uniform_token_error": uniform_error,
                        "counterfactual_weighted_error": weighted_error,
                        "counterfactual_weight_source": (
                            "global_deployment_action_branch_prediction_variance_eq14"
                        ),
                        "counterfactual_weight_scope": (
                            "one_global_vector_fitted_on_distinct_deployment_only_source"
                        ),
                        "counterfactual_weight_epsilon": 1.0e-8,
                        "counterfactual_weight_beta": 1.0,
                        "counterfactual_weight_fit_sha256": _sha("fit"),
                        "counterfactual_weight_vector_sha256": weight_vector_sha,
                        "counterfactual_weight_fit_source_id": "fit-source",
                        "counterfactual_weight_fit_ids_sha256": _sha("fit-ids"),
                        "counterfactual_weight_fit_count": 12,
                        "weight_fit_observed_state_signatures_sha256": _sha(
                            "fit-signatures"
                        ),
                        "scored_observed_state_signatures_sha256": _sha(
                            "score-signatures"
                        ),
                        "weight_fit_scored_observed_state_overlap_count": 0,
                        "observed_state_signature_contract": (
                            "sha256_shape_float32_causal_H0_H1_H2_anchor_history"
                        ),
                        "counterfactual_weight_fit_source_artifact_sha256": _sha(
                            "fit-source-artifact"
                        ),
                        "counterfactual_weight_estimator": (
                            "project_specific_local_branch_eq14_style_not_paper_action_pool"
                        ),
                        "token_error_scope": (
                            "same_two_slot_anchor_carrier_as_exact_action_intervention"
                        ),
                        "offline_labels_score_only": True,
                        "target_leakage_detected": False,
                        "deployment_prediction_uses_offline_target": False,
                        "analytic_observer_scope": (
                            "globally_frozen_missing_measurement_extension"
                        ),
                    }
                )
    return rows


def _envelope() -> dict[str, object]:
    weights = np.asarray([[3.0, 0.0, 0.0]], dtype=np.float64)
    rows = _rows(weights)
    observer = _observer()
    contract_sha = _sha("observer-source-file")
    return {
        "schema_version": 1,
        "protocol": cli.EVALUATOR_PROTOCOL,
        "classification": cli.EVALUATOR_CLASSIFICATION,
        "completed": True,
        "row_count": len(rows),
        "physical_episode_count": 30,
        "causal_action_controls_scored_with_exact_frozen_policy": True,
        "observer_contract_metadata": observer,
        "observer_contract_metadata_sha256": cli._canonical_sha256(observer),
        "observer_contract_source_sha256": contract_sha,
        "analytic_observer": {
            "contract_sha256": contract_sha,
            "feedback_gain": 1.0,
            "residual_decay": 1.0,
            "delta_t": 1.0,
        },
        "controllability": {
            "beta": 1.0,
            "epsilon": 1.0e-8,
            "weight_fit_sha256": _sha("fit"),
            "weight_vector_sha256": cli._float64_array_sha256(weights),
            "weights": weights.tolist(),
            "fit_source": "fit-source",
            "fit_ids_sha256": _sha("fit-ids"),
            "fit_count": 12,
            "fit_source_artifact_sha256": _sha("fit-source-artifact"),
            "fit_observed_state_signatures_sha256": _sha("fit-signatures"),
            "scored_observed_state_signatures_sha256": _sha("score-signatures"),
            "fit_scored_observed_state_overlap_count": 0,
            "observed_state_signature_contract": (
                "sha256_shape_float32_causal_H0_H1_H2_anchor_history"
            ),
            "estimator": (
                "project_specific_local_branch_eq14_style_not_paper_action_pool"
            ),
            "scored_fresh_panel_used_to_fit_weights": False,
        },
        "rows_sha256": cli._canonical_sha256(rows),
        "rows": rows,
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_summarizer_directly_consumes_complete_evaluator_envelope(tmp_path: Path) -> None:
    source = tmp_path / "evaluation.json"
    output = tmp_path / "summary.json"
    _write(source, _envelope())

    assert cli.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--bootstrap-samples",
            "100",
        ]
    ) == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["input_evaluator"]["rows_sha256"] == _envelope()["rows_sha256"]
    assert result["input_evaluator"]["observer_contract_source_sha256"] == _sha(
        "observer-source-file"
    )
    assert result["gate"]["wrong_action_direction_pass_count"] == 7
    digest_payload = copy.deepcopy(result)
    observed_digest = digest_payload.pop("result_sha256")
    assert observed_digest == cli._canonical_sha256(digest_payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("rows", "rows SHA-256"),
        ("observer", "observer feedback_gain"),
        ("weight", "weight-vector SHA-256"),
        ("fit_signature", "differs from evaluator provenance"),
        ("wrong_action", "shift_ids contract drifted"),
    ],
)
def test_evaluator_contract_drift_fails_closed(
    tmp_path: Path, mutation: str, match: str
) -> None:
    payload = _envelope()
    rows = payload["rows"]
    assert isinstance(rows, list)
    if mutation == "rows":
        rows[0]["exact_action_error"] = 999.0
    elif mutation == "observer":
        analytic = payload["analytic_observer"]
        assert isinstance(analytic, dict)
        analytic["feedback_gain"] = 0.5
    elif mutation == "weight":
        control = payload["controllability"]
        assert isinstance(control, dict)
        control["weights"] = [[0.0, 3.0, 0.0]]
    elif mutation == "fit_signature":
        rows[0]["weight_fit_observed_state_signatures_sha256"] = _sha("tampered")
        payload["rows_sha256"] = cli._canonical_sha256(rows)
    elif mutation == "wrong_action":
        rows[0]["wrong_action_shift_ids"][0] = "wrong-order"
        payload["rows_sha256"] = cli._canonical_sha256(rows)
    else:  # pragma: no cover
        raise AssertionError(mutation)
    source = tmp_path / "evaluation.json"
    _write(source, payload)

    with pytest.raises((PermissionError, ValueError), match=match):
        cli.main(["--input", str(source), "--bootstrap-samples", "100"])


def test_external_observer_must_equal_embedded_contract(tmp_path: Path) -> None:
    payload = _envelope()
    source = tmp_path / "evaluation.json"
    external = tmp_path / "observer.json"
    _write(source, payload)
    changed = _observer()
    changed["residual_decay"] = 0.5
    _write(external, changed)

    with pytest.raises(PermissionError, match="external observer contract differs"):
        cli.main(
            [
                "--input",
                str(source),
                "--observer-contract-metadata",
                str(external),
                "--bootstrap-samples",
                "100",
            ]
        )


def test_bare_raw_panel_still_consumes_separate_weight_fit_artifacts(
    tmp_path: Path,
) -> None:
    rows = _rows(np.asarray([[3.0, 0.0, 0.0]], dtype=np.float64))
    precomputed_fields = {
        "uniform_token_error",
        "counterfactual_weighted_error",
        "counterfactual_weight_source",
        "counterfactual_weight_scope",
        "counterfactual_weight_epsilon",
        "counterfactual_weight_beta",
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
        "counterfactual_weight_estimator",
    }
    for row in rows:
        for field in precomputed_fields:
            row.pop(field)
        exact = float(row["exact_action_error"])
        branch = int(row["branch_index"])
        row["observed_state_signature"] = (
            f"score-{row['physical_episode_id']}-H{row['horizon']}"
        )
        row["deployment_prediction_vector"] = [float(branch), 0.0, 0.0]
        row["prediction_error_vector"] = [
            np.sqrt(exact),
            np.sqrt(max(0.0, 1.0 - 0.1 * branch)),
            0.5,
        ]

    fit = np.zeros((12, 4, 3), dtype=np.float64)
    fit[:, :, 0] = np.arange(4, dtype=np.float64)[None]
    fit_path = tmp_path / "fit.npz"
    np.savez_compressed(fit_path, branch_predictions=fit)
    metadata = {
        "schema_version": 1,
        "protocol": "control_sufficient_p0_weight_fit_v1",
        "classification": "deployment_only_distinct_weight_fit",
        "fit_source_id": "separate-fit",
        "fit_sample_ids": [f"fit-state-{index}" for index in range(12)],
        "fit_physical_episode_ids": [f"fit-episode-{index}" for index in range(6)],
        "fit_observed_state_signatures": [
            f"fit-signature-{index}" for index in range(12)
        ],
        "deployment_only": True,
        "offline_labels_used": False,
        "scored_panel_used": False,
        "artifact_sha256": cli._sha256_file(fit_path),
    }
    rows_path = tmp_path / "rows.json"
    metadata_path = tmp_path / "fit.metadata.json"
    observer_path = tmp_path / "observer.json"
    output_path = tmp_path / "summary.json"
    _write(rows_path, rows)
    _write(metadata_path, metadata)
    _write(observer_path, _observer())

    assert cli.main(
        [
            "--input",
            str(rows_path),
            "--weight-fit-artifact",
            str(fit_path),
            "--weight-fit-metadata",
            str(metadata_path),
            "--observer-contract-metadata",
            str(observer_path),
            "--bootstrap-samples",
            "100",
            "--output",
            str(output_path),
        ]
    ) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["counterfactual_weights"][
        "computed_from_separate_raw_fit_artifact"
    ] is True
    assert "input_evaluator" not in result
