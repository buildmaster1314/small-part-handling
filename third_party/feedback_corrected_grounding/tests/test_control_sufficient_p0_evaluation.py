"""Focused tests for the exposed control-sufficiency P0 evaluator."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.control_sufficient_p0 import summarize_control_sufficient_p0
from fcpg.control_sufficient_p0_evaluation import (
    OBSERVER_CONTRACT_PROTOCOL,
    FrozenAnalyticObserverContract,
    SeedPanel,
    build_seed_macro_rows,
    exact_carrier_held_fixed_action_error,
    fit_eq14_style_global_weights,
    load_frozen_observer_contract,
    recursive_analytic_feedback_complementary_rollout,
    validate_seed_invariant_carriers,
)
from scripts import evaluate_control_sufficient_p0 as p0_script


def _observer_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": OBSERVER_CONTRACT_PROTOCOL,
        "classification": "exposed_p0_global_fixed_parameters",
        "feedback_gain": 0.5,
        "residual_decay": 1.0,
        "delta_t": 1.0,
        "controllability_beta": 1.0,
        "parameter_source": "frozen before opening the P0 label archive",
        "frozen_before_offline_label_scoring": True,
        "target_tuned_per_row": False,
        "target_tuned_per_horizon": False,
    }


def test_observer_contract_requires_external_digest_and_rejects_target_tuning(tmp_path) -> None:
    with pytest.raises(PermissionError, match="no frozen observer contract"):
        load_frozen_observer_contract(None, expected_sha256=None)

    path = tmp_path / "observer.json"
    path.write_text(json.dumps(_observer_payload()), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    value = load_frozen_observer_contract(path, expected_sha256=digest)
    assert value.feedback_gain == 0.5
    assert value.controllability_beta == 1.0

    payload = _observer_payload()
    payload["target_tuned_per_horizon"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(PermissionError, match="global pre-score freezing"):
        load_frozen_observer_contract(path, expected_sha256=digest)


def test_weight_fit_rolls_all_actions_from_one_frozen_state(monkeypatch) -> None:
    history = np.zeros((2, 8, 3, 3, 256), dtype=np.float32)
    history_action = np.zeros((2, 8, 2, 20), dtype=np.float32)
    for branch in range(8):
        history[:, branch] = float(branch)
        history_action[:, branch] = float(branch)
    future = np.zeros((2, 8, 6, 20), dtype=np.float32)
    future[:, :, :, 0] = np.arange(8, dtype=np.float32)[None, :, None]
    cache = SimpleNamespace(
        history_token=history,
        history_action=history_action,
        future_action=future,
        branch_index=np.broadcast_to(np.arange(8)[None], (2, 8)).copy(),
    )
    seen_history: list[np.ndarray] = []

    def fake_rollout(
        factual: object,
        no_action: object,
        batch_history: torch.Tensor,
        batch_history_action: torch.Tensor,
        batch_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del factual, no_action
        seen_history.append(batch_history.cpu().numpy())
        assert torch.count_nonzero(batch_history) == 0
        assert torch.count_nonzero(batch_history_action) == 0
        prediction = batch_future[..., :1, None].expand(-1, -1, -1, 3, 256)
        return prediction, torch.zeros_like(prediction)

    monkeypatch.setattr(p0_script, "rollout_matched_p1_native", fake_rollout)
    result = p0_script._same_state_weight_fit_predict_cache(
        object(), object(), cache, device="cpu", batch_size=1
    )

    assert result.shape == (2, 8, 6, 3, 256)
    assert len(seen_history) == 2
    np.testing.assert_array_equal(result[0, :, 0, 0, 0], np.arange(8))


class _IncrementPredictor(nn.Module):
    action_dim = 1

    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        result = history.clone()
        result[:, -1] = history[:, -1] + action[:, -1, None]
        return result


def test_recursive_analytic_rollout_uses_only_causal_pre_drop_residual() -> None:
    model = _IncrementPredictor().eval()
    history = torch.tensor([[[[0.0]], [[1.0]], [[2.0]]]])
    history_action = torch.zeros(1, 2, 1)
    future_action = torch.ones(1, 1, 6, 1)
    geometry = torch.zeros(1, 1, 6, 1, 1)
    observer = FrozenAnalyticObserverContract(
        feedback_gain=0.5,
        residual_decay=1.0,
        delta_t=1.0,
        controllability_beta=1.0,
        parameter_source="test",
        sha256="a" * 64,
    )
    output = recursive_analytic_feedback_complementary_rollout(
        model,
        history,
        history_action,
        future_action,
        geometry,
        final_visible_measurement=torch.tensor([[[2.0]]]),
        causal_final_visible_prior=torch.tensor([[[1.0]]]),
        observer=observer,
        complementary_gains=(1.0,) * 6,
    )
    # Raw H3 is 3.0; the held causal innovation is +1 and L=0.5.
    assert output[0, 0, 0, 0, 0].item() == pytest.approx(3.5)
    assert output.shape == (1, 1, 6, 1, 1)


class _AnchorActionPolicy(nn.Module):
    obs_horizon = 2
    num_keypoints = 1
    model_dim = 2

    def forward(self, batch: dict[str, torch.Tensor]) -> SimpleNamespace:
        anchor = batch["grounded_token_override"]
        value = anchor.flatten(1).sum(1)
        return SimpleNamespace(action=value[:, None, None].expand(-1, 5, 1))


def test_exact_policy_error_holds_non_anchor_carrier_fixed() -> None:
    policy = _AnchorActionPolicy().eval()
    count = 3
    carrier = {
        "policy_obs": np.zeros((count, 2, 4), dtype=np.float32),
        "policy_q_nominal": np.zeros((count, 2, 1, 2), dtype=np.float32),
        "policy_keypoints_xyz": np.zeros((count, 2, 1, 3), dtype=np.float32),
        "policy_sampled_visual": np.zeros((count, 2, 1, 4), dtype=np.float32),
        "policy_global_visual": np.zeros((count, 2, 4), dtype=np.float32),
        "policy_image_size": np.full((count, 2, 2), 32, dtype=np.int64),
        "policy_fcpg_q_corrected": np.zeros((count, 2, 1, 2), dtype=np.float32),
    }
    predicted = np.ones((count, 2, 1, 2), dtype=np.float32)
    target = np.zeros_like(predicted)
    error = exact_carrier_held_fixed_action_error(
        policy,
        carrier,
        predicted,
        target,
        device="cpu",
        batch_size=2,
    )
    assert error.tolist() == pytest.approx([16.0, 16.0, 16.0])


def test_eq14_style_weight_is_global_and_fit_outside_scored_rows() -> None:
    fit: dict[int, np.ndarray] = {}
    for seed in (15801, 15802, 15803):
        value = np.zeros((2, 4, 6, 1, 2), dtype=np.float32)
        value[:, :, :, 0, 0] = np.arange(4)[None, :, None] + seed * 1.0e-6
        value[:, :, :, 0, 1] = 7.0
        fit[seed] = value
    weights = fit_eq14_style_global_weights(fit)
    assert weights.shape == (1, 2)
    assert weights[0, 0] == pytest.approx(2.0, rel=1.0e-6)
    assert weights[0, 1] == pytest.approx(0.0, abs=1.0e-12)


def test_seed_invariant_carrier_rejects_cross_seed_physical_drift() -> None:
    fields = {
        "schema_version": np.asarray([1]),
        "protocol": np.asarray(["carrier"]),
        "classification": np.asarray(["exposed"]),
        "policy_row_id": np.asarray(["row"]),
        "episode_uid": np.asarray(["episode"]),
        "physical_episode_id": np.asarray(["physical"]),
        "reset_physical_state_goal_digest": np.asarray(["a" * 64]),
        "observed_state_signature": np.asarray(["b" * 64]),
        "branch_index": np.asarray([0]),
        "horizon": np.asarray([3]),
        "policy_obs": np.zeros((1, 2, 1), np.float32),
        "policy_q_nominal": np.zeros((1, 2, 3, 2), np.float32),
        "policy_keypoints_xyz": np.zeros((1, 2, 3, 3), np.float32),
        "policy_sampled_visual": np.zeros((1, 2, 3, 1), np.float32),
        "policy_global_visual": np.zeros((1, 2, 1), np.float32),
        "policy_image_size": np.full((1, 2), 64),
        "policy_fcpg_q_corrected": np.zeros((1, 2, 3, 2), np.float32),
        "policy_fcpg_reliability": np.ones((1, 2, 3), np.float32),
        "pre_drop_history_horizons": np.asarray([-1, 0, 1]),
        "pre_drop_target_horizon": np.asarray([2]),
        "pre_drop_history_token": np.zeros((1, 3, 3, 256), np.float32),
        "pre_drop_action_context": np.zeros((1, 3, 20), np.float32),
        "pre_drop_visible_token": np.zeros((1, 3, 256), np.float32),
        "pre_drop_visible_visual": np.zeros((1, 3, 256), np.float32),
        "same_state_prefix": np.asarray([True]),
        "policy_future_action": np.zeros((1, 20), np.float32),
        "policy_geometry_token": np.zeros((1, 3, 256), np.float32),
        "recursive_inputs_are_from_identity_verified_same_replay": np.asarray(
            [True]
        ),
    }
    deployments = {
        seed: {**fields, "predictor_seed": np.asarray([seed])}
        for seed in (15801, 15802, 15803)
    }
    validate_seed_invariant_carriers(deployments)
    deployments[15803] = {
        **deployments[15803],
        "physical_episode_id": np.asarray(["different"]),
    }
    with pytest.raises(PermissionError, match="physical_episode_id"):
        validate_seed_invariant_carriers(deployments)


def test_seed_macro_rows_are_consumable_without_tripling_physical_episodes() -> None:
    episodes, branch_count, horizons = 30, 8, 6
    count = episodes * branch_count * horizons
    episode = np.repeat(
        np.asarray([f"episode-{index:02d}" for index in range(episodes)]),
        branch_count * horizons,
    )
    branch = np.tile(np.repeat(np.arange(branch_count), horizons), episodes)
    horizon = np.tile(np.arange(3, 9), episodes * branch_count)
    target = np.zeros((count, 2, 1, 2), dtype=np.float32)
    panels = []
    for seed_offset, seed in enumerate((15801, 15802, 15803)):
        prediction = np.zeros_like(target)
        prediction[:, :, 0, 0] = (
            0.1 * branch + 0.01 * seed_offset
        )[:, None]
        prediction[:, :, 0, 1] = (
            0.4 + 0.02 * np.arange(count) / count
        )[:, None]
        exact = np.square(prediction[:, -1, 0, 0]) + 0.001 * (horizon - 3)
        complementary = exact + 0.3
        analytic = 0.7 * complementary
        panels.append(
            SeedPanel(
                seed=seed,
                checkpoint_sha256=(f"{seed_offset + 1:x}" * 64)[:64],
                raw_prediction=prediction,
                complementary_prediction=0.8 * prediction,
                analytic_prediction=0.7 * prediction,
                raw_action_error=exact,
                complementary_action_error=complementary,
                analytic_action_error=analytic,
                matched_no_action_action_error=exact + 0.2,
                wrong_action_action_error=np.repeat(
                    (exact + 0.4)[:, None], 7, axis=1
                ),
            )
        )
    rows = build_seed_macro_rows(
        {
            "physical_episode_id": episode,
            "branch_index": branch,
            "horizon": horizon,
        },
        target,
        panels,
        frozen_controllability_weights=np.asarray([[2.0, 0.0]]),
        weight_fit_sha256="f" * 64,
        weight_fit_source_id="strict-train-development-v1",
        weight_fit_ids_sha256="e" * 64,
        weight_fit_count=180,
        weight_fit_observed_state_signatures_sha256="d" * 64,
        scored_observed_state_signatures_sha256="c" * 64,
        weight_fit_scored_signature_overlap_count=0,
        weight_fit_source_artifact_sha256="b" * 64,
    )
    assert len(rows) == count
    assert "deployment_prediction_vector" not in rows[0]
    assert len(rows[0]["per_seed"]) == 3
    result = summarize_control_sufficient_p0(
        rows,
        observer_contract_metadata=_observer_payload(),
        bootstrap_samples=100,
    )
    assert result["physical_episode_count"] == 30
    assert result["counterfactual_weights"][
        "computed_from_separate_raw_fit_artifact"
    ] is False
