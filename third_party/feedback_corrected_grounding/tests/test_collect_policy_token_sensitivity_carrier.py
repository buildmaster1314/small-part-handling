from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.collect_policy_token_sensitivity_carrier import (
    _deployment_only_native_cache,
    _same_replay_predictor_inputs,
    open_offline_labels_after_predictions,
)


def _deployment_arrays() -> dict[str, np.ndarray]:
    episodes, branches, horizons, anchors, latent = 30, 8, 6, 3, 256
    return {
        "task_name": np.full(episodes, "pick-place-v3"),
        "physical_episode_id": np.asarray(
            [f"pick-place-v3:{index:064x}" for index in range(episodes)]
        ),
        "hazard_id": np.arange(episodes, dtype=np.int64),
        "history_token": np.zeros(
            (episodes, branches, 3, anchors, latent), dtype=np.float32
        ),
        "history_action": np.zeros((episodes, branches, 2, 20), dtype=np.float32),
        "future_action": np.zeros(
            (episodes, branches, horizons, 20), dtype=np.float32
        ),
        "wrong_future_action": np.zeros(
            (episodes, 7, branches, horizons, 20), dtype=np.float32
        ),
        "history_sampled_visual": np.zeros(
            (episodes, branches, 3, anchors, latent), dtype=np.float32
        ),
        "history_xyz": np.zeros(
            (episodes, branches, 3, anchors, 3), dtype=np.float32
        ),
        "future_xyz": np.zeros(
            (episodes, branches, horizons, anchors, 3), dtype=np.float32
        ),
    }


def test_deployment_only_cache_contains_no_offline_target_value() -> None:
    arrays = _deployment_arrays()
    cache = _deployment_only_native_cache(
        arrays,
        source_path=Path("deployment-inputs.npz"),
        metadata={"completed": True},
    )

    assert np.isnan(cache.target_token).all()
    assert cache.metadata is not None
    assert cache.metadata["target_token_is_nonlabel_shape_sentinel"] is True
    np.testing.assert_array_equal(cache.future_action, arrays["future_action"])


def test_offline_label_loader_fails_before_prediction_completion() -> None:
    with pytest.raises(PermissionError, match="before predictions complete"):
        open_offline_labels_after_predictions(
            None,  # type: ignore[arg-type]
            Path("unused.json"),
            predictions_complete=False,
        )


def test_same_replay_predictor_inputs_preserve_common_prefix_and_branch_actions() -> None:
    episodes, branches, horizons, anchors, latent = 30, 8, 6, 3, 256
    count = episodes * branches * horizons
    history = np.zeros((episodes, branches, horizons, 3, anchors, latent), np.float32)
    visible = np.zeros((episodes, branches, horizons, anchors, latent), np.float32)
    action = np.zeros((episodes, branches, horizons, 3, 20), np.float32)
    for episode in range(episodes):
        history[episode] = episode
        visible[episode] = episode + 0.5
        action[episode] = episode + 1.0
    future = np.zeros((episodes, branches, horizons, 20), np.float32)
    for branch in range(branches):
        future[:, branch] = branch
    deployment = {
        "pre_drop_history_token": history.reshape(count, 3, anchors, latent),
        "pre_drop_action_context": action.reshape(count, 3, 20),
        "pre_drop_visible_token": visible.reshape(count, anchors, latent),
        "policy_future_action": future.reshape(count, 20),
    }
    geometry = np.zeros((count, anchors, latent), np.float32)

    result = _same_replay_predictor_inputs(deployment, geometry)

    assert result.history_token.shape == (episodes * branches, 3, anchors, latent)
    assert result.history_action.shape == (episodes * branches, 2, 20)
    assert result.future_action.shape == (episodes * branches, 1, horizons, 20)
    assert result.wrong_future_action.shape == (
        episodes * branches,
        branches - 1,
        1,
        horizons,
        20,
    )
    assert result.future_action[0, 0, 0, 0] == pytest.approx(0.0)
    assert result.future_action[horizons * 0 + 1, 0, 0, 0] == pytest.approx(1.0)
    assert result.wrong_future_action[0, 0, 0, 0, 0] == pytest.approx(1.0)
