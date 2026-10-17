from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.lewm_p1_native_dynamics import (
    P1NativePredictorConfig,
    build_matched_p1_native_predictors,
    build_native_examples,
    flatten_native_examples,
    p1_native_episode_mse,
    reconstruct_p1_native_baselines,
    set_p1_native_action_normalization,
    validate_matched_p1_native_pair,
)
from scripts.train_lewm_p1_native_dynamics import _build_report, _load_fixed_config
from scripts.train_lewm_p1_native_dynamics import (
    _matched_recursive_forecast,
    recursive_forecast,
)


ROWS = 4
BRANCHES = 8
HORIZONS = 8
ANCHORS = 3
ACTION_DIM = 20
TOKEN_DIM = 256
A3_DIM = 192


def _stacked_arrays() -> dict[str, np.ndarray]:
    """Make a strict stacked-cache fixture with traceable clock values."""

    rng = np.random.default_rng(183)
    hidden_token = np.empty(
        (ROWS, BRANCHES, HORIZONS, ANCHORS, TOKEN_DIM), dtype=np.float32
    )
    sampled_visual = np.empty_like(hidden_token)
    current_xyz = np.empty(
        (ROWS, BRANCHES, HORIZONS, ANCHORS, 3), dtype=np.float32
    )
    branch_action = np.empty(
        (ROWS, BRANCHES, HORIZONS, ACTION_DIM), dtype=np.float32
    )
    for row in range(ROWS):
        for branch in range(BRANCHES):
            for horizon_index in range(HORIZONS):
                horizon = horizon_index + 1
                marker = float(10_000 * row + 100 * branch + horizon)
                hidden_token[row, branch, horizon_index] = marker
                sampled_visual[row, branch, horizon_index] = marker + 0.25
                current_xyz[row, branch, horizon_index] = marker + 0.5
                branch_action[row, branch, horizon_index] = marker + 0.75

    seeds = np.asarray([29301, 29302, 29303], dtype=np.int64)
    arrays = {
        "source_row_index": np.arange(ROWS, dtype=np.int64),
        "task_index": np.zeros(ROWS, dtype=np.int64),
        "task_instance_index": np.arange(ROWS, dtype=np.int64),
        "source_episode_id": np.asarray(
            ["episode-train-a", "episode-train-a", "episode-train-b", "episode-dev-a"]
        ),
        "hazard_id": np.asarray([0, 1, 0, 0], dtype=np.int64),
        "split": np.asarray(["train", "train", "train", "development"]),
        "predictor_seed": seeds,
        "branch_index": np.arange(BRANCHES, dtype=np.int64),
        "history_embedding": rng.normal(
            size=(ROWS, 3, ANCHORS, A3_DIM)
        ).astype(np.float32),
        "history_action": rng.normal(size=(ROWS, 2, ACTION_DIM)).astype(np.float32),
        "branch_action": branch_action,
        "factual_prediction": rng.normal(
            size=(3, ROWS, BRANCHES, HORIZONS, ANCHORS, A3_DIM)
        ).astype(np.float32),
        "no_action_prediction": rng.normal(
            size=(3, ROWS, BRANCHES, HORIZONS, ANCHORS, A3_DIM)
        ).astype(np.float32),
        "hidden_clean_a3_target": rng.normal(
            size=(ROWS, BRANCHES, HORIZONS, ANCHORS, A3_DIM)
        ).astype(np.float32),
        "last_policy_token": np.stack(
            [np.full((ANCHORS, TOKEN_DIM), -100.0 - row, np.float32) for row in range(ROWS)]
        ),
        "last_sampled_visual": np.stack(
            [np.full((ANCHORS, TOKEN_DIM), -10.0 - row, np.float32) for row in range(ROWS)]
        ),
        "last_xyz": rng.normal(size=(ROWS, ANCHORS, 3)).astype(np.float32),
        "current_xyz": current_xyz,
        "hidden_clean_policy_token": hidden_token,
        "hidden_clean_sampled_visual": sampled_visual,
        "hidden_clean_q_corrected": rng.normal(
            size=(ROWS, BRANCHES, HORIZONS, ANCHORS, 2)
        ).astype(np.float32),
        "hidden_clean_reliability": np.ones(
            (ROWS, BRANCHES, HORIZONS, ANCHORS), dtype=np.float32
        ),
        "max_a3_history_reconstruction_error": np.zeros(ROWS, dtype=np.float64),
        "max_a3_target_reconstruction_error": np.zeros(ROWS, dtype=np.float64),
    }
    return arrays


def _attr(value: object, *names: str) -> object:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    raise AssertionError(f"missing expected field; tried {names}")


def test_native_examples_use_h0_h1_h2_then_predict_h3_through_h8() -> None:
    arrays = _stacked_arrays()
    cache = build_native_examples(arrays, split="train")

    history = np.asarray(_attr(cache, "history_token", "history_tokens"))
    history_action = np.asarray(_attr(cache, "history_action"))
    future_action = np.asarray(_attr(cache, "future_action"))
    target = np.asarray(_attr(cache, "target_token", "target_tokens"))

    assert history.shape == (3, BRANCHES, 3, ANCHORS, TOKEN_DIM)
    assert history_action.shape == (3, BRANCHES, 2, ACTION_DIM)
    assert future_action.shape == (3, BRANCHES, 6, ACTION_DIM)
    assert target.shape == (3, BRANCHES, 6, ANCHORS, TOKEN_DIM)

    # Row 0, branch 3: H0 is the last observed policy token; H1/H2 are the
    # two healthy branch endpoints immediately before the bounded dropout.
    np.testing.assert_array_equal(history[0, 3, 0], arrays["last_policy_token"][0])
    assert float(history[0, 3, 1, 0, 0]) == 301.0
    assert float(history[0, 3, 2, 0, 0]) == 302.0
    assert float(history_action[0, 3, 0, 0]) == 301.75
    assert float(history_action[0, 3, 1, 0]) == 302.75
    assert future_action[0, 3, :, 0].tolist() == pytest.approx(
        [303.75, 304.75, 305.75, 306.75, 307.75, 308.75]
    )
    assert target[0, 3, :, 0, 0].tolist() == pytest.approx(
        [303.0, 304.0, 305.0, 306.0, 307.0, 308.0]
    )


def test_native_builder_rejects_physical_episode_crossing_splits() -> None:
    arrays = _stacked_arrays()
    arrays["source_episode_id"] = arrays["source_episode_id"].copy()
    arrays["source_episode_id"][3] = "episode-train-a"

    with pytest.raises(ValueError, match="episode|split|leak"):
        build_native_examples(arrays)


def test_flatten_keeps_branch_rows_nested_under_physical_episode() -> None:
    cache = build_native_examples(_stacked_arrays(), split="train")
    flat = flatten_native_examples(cache)

    token = np.asarray(_attr(flat, "history_token", "history_tokens"))
    episode = np.asarray(
        _attr(flat, "physical_episode_id", "source_episode_id", "physical_episode_sha256")
    ).astype(str)
    branch = np.asarray(_attr(flat, "branch_index"), dtype=np.int64)
    assert token.shape[0] == 3 * BRANCHES
    assert episode.shape == (3 * BRANCHES,)
    assert branch.shape == (3 * BRANCHES,)
    assert len(set(episode.tolist())) == 2
    assert np.count_nonzero(episode == "episode-train-a") == 2 * BRANCHES
    np.testing.assert_array_equal(branch[:BRANCHES], np.arange(BRANCHES))


def test_wrong_action_bank_has_all_seven_cyclic_donors_without_fixed_points() -> None:
    cache = build_native_examples(_stacked_arrays(), split="development")
    wrong = cache.wrong_future_action
    assert wrong.shape == (1, 7, BRANCHES, 6, ACTION_DIM)
    slots = np.arange(BRANCHES)
    for shift_index, shift in enumerate(range(1, BRANCHES)):
        donor = (slots + shift) % BRANCHES
        np.testing.assert_array_equal(
            wrong[:, shift_index], cache.future_action[:, donor]
        )
        assert not np.any(donor == slots)


def test_episode_mse_nests_rows_hazards_branches_and_horizons() -> None:
    cache = build_native_examples(_stacked_arrays(), split="train")
    prediction = cache.target_token.copy()
    prediction[0] += 1.0
    prediction[1] += 3.0
    prediction[2] += 5.0

    result = p1_native_episode_mse(prediction, cache.target_token, cache)

    # Rows 0/1 are two hazards from the same episode.  Branches and horizons
    # are nested beneath it, so this is two observations, not 3*8*6.
    assert set(result) == {"episode-train-a", "episode-train-b"}
    assert result["episode-train-a"] == pytest.approx((1.0 + 9.0) / 2.0)
    assert result["episode-train-b"] == pytest.approx(25.0)


def test_matched_native_pair_is_same_capacity_and_exact_initialization() -> None:
    config = P1NativePredictorConfig(
        action_smoothing_dim=4,
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=32,
        predictor_dim_head=8,
        predictor_dropout=0.0,
        projector_hidden_dim=32,
    )
    factual, no_action = build_matched_p1_native_predictors(
        config, initialization_seed=717
    )
    validate_matched_p1_native_pair(factual, no_action, require_identical=True)

    assert factual.condition_on_action is True
    assert no_action.condition_on_action is False
    assert sum(p.numel() for p in factual.parameters()) == sum(
        p.numel() for p in no_action.parameters()
    )
    for name, parameter in factual.state_dict().items():
        torch.testing.assert_close(parameter, no_action.state_dict()[name], rtol=0, atol=0)

    with torch.no_grad():
        next(iter(no_action.parameters())).add_(1.0)
    with pytest.raises(ValueError, match="initial|matched|identical"):
        validate_matched_p1_native_pair(factual, no_action, require_identical=True)


def test_trainer_matched_rollout_shares_dropout_and_consumes_one_rng_stream() -> None:
    config = P1NativePredictorConfig(
        action_smoothing_dim=4,
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=32,
        predictor_dim_head=8,
        predictor_dropout=0.4,
        projector_hidden_dim=32,
    )
    factual, no_action = build_matched_p1_native_predictors(
        config, initialization_seed=727
    )
    factual.train()
    no_action.train()
    history = torch.randn(2, 3, ANCHORS, TOKEN_DIM)
    history_action = torch.zeros(2, 2, ACTION_DIM)
    future_action = torch.zeros(2, 6, ACTION_DIM)

    torch.manual_seed(991)
    factual_prediction, no_action_prediction = _matched_recursive_forecast(
        factual, no_action, history, history_action, future_action
    )
    matched_end_state = torch.random.get_rng_state().clone()
    torch.testing.assert_close(factual_prediction, no_action_prediction, rtol=0, atol=0)

    single_factual, _ = build_matched_p1_native_predictors(
        config, initialization_seed=727
    )
    single_factual.train()
    torch.manual_seed(991)
    recursive_forecast(single_factual, history, history_action, future_action)
    single_end_state = torch.random.get_rng_state().clone()
    torch.testing.assert_close(matched_end_state, single_end_state, rtol=0, atol=0)


def test_action_normalization_is_train_only_and_identical_between_heads() -> None:
    config = P1NativePredictorConfig(
        action_smoothing_dim=4,
        predictor_depth=1,
        predictor_heads=1,
        predictor_mlp_dim=32,
        predictor_dim_head=8,
        predictor_dropout=0.0,
        projector_hidden_dim=32,
    )
    factual, no_action = build_matched_p1_native_predictors(
        config, initialization_seed=719
    )
    train = build_native_examples(_stacked_arrays(), split="train")
    expected = np.concatenate(
        (train.history_action.reshape(-1, ACTION_DIM), train.future_action.reshape(-1, ACTION_DIM))
    )

    mean, std = set_p1_native_action_normalization(factual, no_action, train)

    np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(std, expected.std(axis=0), rtol=1e-6)
    torch.testing.assert_close(factual.action_mean, no_action.action_mean, rtol=0, atol=0)
    torch.testing.assert_close(factual.action_std, no_action.action_std, rtol=0, atol=0)

    development = build_native_examples(_stacked_arrays(), split="development")
    with pytest.raises(ValueError, match="train split only"):
        set_p1_native_action_normalization(factual, no_action, development)


class _AdditiveAdapter(nn.Module):
    def forward_sampled(
        self,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        *,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.equal(reliability, torch.ones_like(reliability))
        return visual + proprio


class _FakePolicy(nn.Module):
    num_keypoints = ANCHORS
    vision_dim = TOKEN_DIM
    model_dim = TOKEN_DIM

    def __init__(self) -> None:
        super().__init__()
        self.grounding_adapter = _AdditiveAdapter()

    def _proprio(self, xyz: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(xyz, (0, TOKEN_DIM - 3))


def test_visual_cv_and_geometry_hold_use_h1_h2_visual_with_exact_future_fk() -> None:
    arrays = _stacked_arrays()
    # Give the visual stream a simple velocity independent of token markers.
    arrays["hidden_clean_sampled_visual"][:, :, 0] = 1.0  # H1
    arrays["hidden_clean_sampled_visual"][:, :, 1] = 3.0  # H2
    arrays["current_xyz"][:] = 0.0
    for horizon_index in range(HORIZONS):
        arrays["current_xyz"][:, :, horizon_index, :, 0] = horizon_index + 1
    cache = build_native_examples(arrays, split="development")

    baselines = reconstruct_p1_native_baselines(
        cache, _FakePolicy().eval(), device="cpu", batch_size=3
    )
    hold = np.asarray(_attr(baselines, "hold_last", "hold_last_token"))
    geometry = np.asarray(_attr(baselines, "geometry_hold", "geometry_hold_token"))
    visual_cv = np.asarray(
        _attr(baselines, "visual_constant_velocity", "visual_cv", "visual_cv_token")
    )
    assert hold.shape == geometry.shape == visual_cv.shape == (
        1,
        BRANCHES,
        6,
        ANCHORS,
        TOKEN_DIM,
    )
    # H3 uses exact FK x=3. Geometry holds H2 visual=3; visual CV advances
    # H1=1,H2=3 by one step to 5.
    assert float(geometry[0, 0, 0, 0, 0]) == pytest.approx(6.0)
    assert float(visual_cv[0, 0, 0, 0, 0]) == pytest.approx(8.0)
    # At H8 the CV visual is 15 and exact FK x=8.
    assert float(geometry[0, 0, -1, 0, 0]) == pytest.approx(11.0)
    assert float(visual_cv[0, 0, -1, 0, 0]) == pytest.approx(23.0)
    # Hold-last is the complete H2 token, not geometry-updated.
    np.testing.assert_array_equal(hold[0, 0, 0], cache.history_token[0, 0, -1])


def test_single_seed_gate_requires_aggregate_hold_last_improvement() -> None:
    cache = build_native_examples(_stacked_arrays(), split="train")
    target = cache.target_token
    predictions = {
        "factual": target + 2.0,
        "matched_no_action": target + 3.0,
        "hold_last": target + 1.0,
        "geometry_hold_exact_fk": target + 3.0,
        "visual_cv_exact_fk": target + 3.0,
    }
    wrong = np.stack([target + 3.0] * 7, axis=0)
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config/lewm/eval/p1_native_dynamics_dev.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    report = _build_report(predictions, wrong, cache, config, seed=15801)

    assert report["screen"]["primary_pass"]["factual_vs_matched_no_action"]
    assert not report["screen"]["primary_pass"]["factual_vs_hold_last"]
    assert report["screen"]["passed"] is False


def test_trainer_accepts_matched_data_variant_but_keeps_training_contract(
    tmp_path: Path,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config/lewm/eval/p1_native_dynamics_dev.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data"]["strict_cache"] = "outputs/matched_nominal/cache.npz"
    config["data"]["strict_cache_sha256"] = "cache-digest"
    config["data"]["strict_metadata"] = "outputs/matched_nominal/cache.metadata.json"
    config["data"]["strict_metadata_sha256"] = "metadata-digest"
    config["output_dir_template"] = "outputs/matched_nominal/seed{seed}"
    matched_path = tmp_path / "matched.json"
    matched_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = _load_fixed_config(matched_path)
    assert loaded["data"]["strict_cache"] == "outputs/matched_nominal/cache.npz"

    config["training"]["optimizer_steps"] = 2001
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="training contract"):
        _load_fixed_config(invalid_path)
