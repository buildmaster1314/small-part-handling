from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_analytic_feedback_observer import (
    audit_feedback_observer_artifact,
    feedback_world_model_step,
    hard_measurement_reanchor,
    partial_measurement_innovation_update,
    propagate_held_feedback_residual,
)


def test_feedback_world_model_step_matches_paper_equations() -> None:
    observed = torch.tensor([2.0])
    belief = torch.tensor([1.0])
    raw_next = torch.tensor([5.0])
    result = feedback_world_model_step(
        observed,
        belief,
        raw_next,
        feedback_gain=0.25,
        delta_t=2.0,
    )

    assert torch.equal(result.innovation, torch.tensor([1.0]))
    assert torch.equal(result.raw_velocity, torch.tensor([1.5]))
    assert torch.equal(result.corrected_velocity, torch.tensor([1.75]))
    assert torch.equal(result.corrected_prediction, torch.tensor([5.5]))
    assert torch.equal(result.next_belief, torch.tensor([4.5]))


def test_feedback_gain_enforces_discrete_stability() -> None:
    state = torch.zeros(1)
    with pytest.raises(ValueError, match="stability"):
        feedback_world_model_step(
            state, state, state, feedback_gain=2.0, delta_t=1.0
        )


@pytest.mark.parametrize("prior_weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_partial_measurement_update_is_existing_complementary_equation(
    prior_weight: float,
) -> None:
    prior = torch.randn(2, 3, 8)
    geometry = torch.randn(2, 3, 8)
    existing = geometry + prior_weight * (prior - geometry)
    innovation = partial_measurement_innovation_update(
        prior,
        geometry,
        measurement_gain=1.0 - prior_weight,
    )
    # The two algebraically identical expressions use a different floating-
    # point operation order, so equivalence is numerical rather than bitwise.
    assert torch.allclose(innovation, existing, rtol=0.0, atol=1.0e-6)


def test_recovery_measurement_is_bitwise_bypass() -> None:
    measurement = torch.randn(2, 3, 256)
    recovered = hard_measurement_reanchor(measurement)
    assert recovered is measurement
    assert torch.equal(recovered, measurement)


def test_missing_propagation_uses_only_last_observed_residual() -> None:
    result = propagate_held_feedback_residual(
        current_state=torch.tensor([2.0]),
        feedback_belief=torch.tensor([1.5]),
        raw_next_prediction=torch.tensor([3.0]),
        last_observed_innovation=torch.tensor([0.8]),
        feedback_gain=0.5,
        missing_age=2,
        residual_decay=0.5,
    )
    assert torch.equal(result.innovation, torch.tensor([0.4]))
    assert torch.equal(result.corrected_prediction, torch.tensor([3.2]))
    assert torch.equal(result.next_belief, torch.tensor([2.7]))


def test_missing_propagation_rejects_noncausal_age_and_unstable_gain() -> None:
    state = torch.zeros(1)
    with pytest.raises(ValueError, match="missing_age"):
        propagate_held_feedback_residual(
            state,
            state,
            state,
            state,
            feedback_gain=0.5,
            missing_age=0,
        )


def _current_three_state_artifact() -> dict[str, np.ndarray]:
    return {
        "history_token": np.zeros((2, 8, 3, 3, 256), dtype=np.float32),
        "history_action": np.zeros((2, 8, 2, 20), dtype=np.float32),
    }


def test_current_three_state_artifact_fails_closed_without_future_label() -> None:
    report = audit_feedback_observer_artifact(_current_three_state_artifact())
    assert report.scorable is False
    assert report.visible_state_count == 3
    assert report.visible_action_count == 2
    assert "cannot form a causal prediction error" in report.reason
    assert "observer_history_token" in report.required_fields


def test_audit_rejects_hidden_clean_target_in_deployment_archive() -> None:
    artifact = _current_three_state_artifact()
    artifact["target_token"] = np.zeros((2, 8, 6, 3, 256), dtype=np.float32)
    with pytest.raises(PermissionError, match="forbidden offline fields"):
        audit_feedback_observer_artifact(artifact)


def test_four_visible_states_make_temporal_feedback_scorable() -> None:
    artifact = _current_three_state_artifact()
    artifact["observer_history_token"] = np.zeros(
        (2, 8, 4, 3, 256), dtype=np.float32
    )
    artifact["observer_history_action"] = np.zeros(
        (2, 8, 3, 20), dtype=np.float32
    )
    report = audit_feedback_observer_artifact(artifact)
    assert report.scorable is True
    assert report.source == "four_visible_states"
    assert report.required_fields == ()


def test_stored_prior_requires_explicit_causal_provenance() -> None:
    artifact = _current_three_state_artifact()
    artifact["pre_drop_raw_prior_token"] = np.zeros(
        (2, 8, 3, 256), dtype=np.float32
    )
    artifact["pre_drop_prior_is_causal"] = np.asarray([False], dtype=bool)
    assert audit_feedback_observer_artifact(artifact).scorable is False

    artifact["pre_drop_prior_is_causal"] = np.asarray([True], dtype=bool)
    report = audit_feedback_observer_artifact(artifact)
    assert report.scorable is True
    assert report.source == "stored_causal_pre_drop_prior"
