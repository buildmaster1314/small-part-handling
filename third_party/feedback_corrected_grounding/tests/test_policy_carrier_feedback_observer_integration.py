from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_analytic_feedback_observer import audit_feedback_observer_artifact
from fcpg.policy_token_carrier_collection import (
    PREDICTION_ALIGNMENT,
    PRE_DROP_HISTORY_HORIZONS,
    PRE_DROP_TARGET_HORIZON,
    attach_aligned_predictions,
)
from scripts.evaluate_lewm_analytic_feedback_observer import run_audit


ROWS = 5
ANCHORS = 3
TOKEN_DIM = 256
ACTION_DIM = 20


def _sha(character: str) -> str:
    return character * 64


def _attached_policy_carrier() -> dict[str, np.ndarray]:
    """Build the causal artifact through the real collector attachment API."""

    row_id = np.asarray([f"row-{index}" for index in range(ROWS)])
    deployment = {
        "policy_row_id": row_id,
        "pre_drop_history_horizons": np.asarray(
            PRE_DROP_HISTORY_HORIZONS, dtype=np.int64
        ),
        "pre_drop_target_horizon": np.asarray(
            [PRE_DROP_TARGET_HORIZON], dtype=np.int64
        ),
        "pre_drop_history_token": np.zeros(
            (ROWS, 3, ANCHORS, TOKEN_DIM), dtype=np.float32
        ),
        "pre_drop_action_context": np.zeros(
            (ROWS, 3, ACTION_DIM), dtype=np.float32
        ),
        "pre_drop_visible_token": np.ones(
            (ROWS, ANCHORS, TOKEN_DIM), dtype=np.float32
        ),
    }
    prediction = np.zeros((ROWS, ANCHORS, TOKEN_DIM), dtype=np.float32)
    return attach_aligned_predictions(
        deployment,
        prediction,
        prediction_row_id=row_id,
        pre_drop_raw_prior_token=prediction,
        raw_factual_predictions=prediction,
        matched_no_action_predictions=prediction,
        wrong_action_predictions=np.zeros(
            (ROWS, 7, ANCHORS, TOKEN_DIM), dtype=np.float32
        ),
        predictor_seed=1,
        predictor_checkpoint_sha256=_sha("a"),
    )


def test_collector_attached_carrier_is_causally_scorable_through_cli(
    tmp_path: Path,
) -> None:
    artifact = _attached_policy_carrier()
    assert artifact["prediction_alignment"].tolist() == [PREDICTION_ALIGNMENT]

    direct = audit_feedback_observer_artifact(artifact)
    assert direct.scorable is True
    assert direct.source == "policy_carrier_causal_pre_drop_prior"
    assert direct.visible_state_count == 4
    assert direct.visible_action_count == 3

    archive = tmp_path / "policy_carrier_deployment.npz"
    np.savez_compressed(archive, **artifact)
    report = run_audit(archive)
    assert report["decision"] == "Scorable"
    assert report["source"] == "policy_carrier_causal_pre_drop_prior"
    assert report["uses_future_clean_target"] is False


def test_policy_carrier_audit_rejects_offline_label_leak() -> None:
    artifact = _attached_policy_carrier()
    artifact["target_token"] = np.zeros(
        (ROWS, 6, ANCHORS, TOKEN_DIM), dtype=np.float32
    )
    with pytest.raises(PermissionError, match="forbidden offline fields"):
        audit_feedback_observer_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        (
            "pre_drop_action_context",
            np.zeros((ROWS - 1, 3, ACTION_DIM), dtype=np.float32),
        ),
        (
            "pre_drop_visible_token",
            np.zeros((ROWS + 1, ANCHORS, TOKEN_DIM), dtype=np.float32),
        ),
        (
            "pre_drop_raw_prior_token",
            np.zeros((ROWS, ANCHORS - 1, TOKEN_DIM), dtype=np.float32),
        ),
        ("policy_row_id", np.asarray(["duplicate"] * ROWS)),
    ],
)
def test_policy_carrier_audit_rejects_row_or_state_misalignment(
    field: str,
    bad_value: np.ndarray,
) -> None:
    artifact = _attached_policy_carrier()
    artifact[field] = bad_value
    with pytest.raises(ValueError, match="align|identity|unique"):
        audit_feedback_observer_artifact(artifact)


def test_policy_carrier_audit_rejects_noncausal_horizon_receipt() -> None:
    artifact = _attached_policy_carrier()
    artifact["pre_drop_history_horizons"] = np.asarray([0, 1, 2])
    with pytest.raises(ValueError, match=r"\[-1,0,1\]"):
        audit_feedback_observer_artifact(artifact)


@pytest.mark.parametrize(
    "flag",
    [
        np.asarray([False], dtype=bool),
        np.asarray([1], dtype=np.int64),
        np.asarray([True, True], dtype=bool),
    ],
)
def test_policy_carrier_audit_requires_scalar_true_causal_provenance(
    flag: np.ndarray,
) -> None:
    artifact = _attached_policy_carrier()
    artifact["pre_drop_prior_is_causal"] = flag
    report = audit_feedback_observer_artifact(artifact)
    assert report.scorable is False
    assert report.required_fields == ("pre_drop_prior_is_causal",)


def test_policy_carrier_audit_fails_closed_on_incomplete_context() -> None:
    artifact = _attached_policy_carrier()
    del artifact["pre_drop_visible_token"]
    report = audit_feedback_observer_artifact(artifact)
    assert report.scorable is False
    assert report.required_fields == ("pre_drop_visible_token",)
