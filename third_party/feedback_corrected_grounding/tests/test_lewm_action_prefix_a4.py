from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_a4_storage import ACTION_DIM, LATENT_DIM, STATE_SLOT_COUNT
from fcpg.lewm_action_branch_a4 import BRANCH_COUNT, HISTORY_STEPS, HORIZON, TASKS
from fcpg.lewm_action_prefix_a4 import (
    ACTION_PREFIX_MODEL_CONFIG,
    _slot_mse_by_horizon,
    action_normalization_from_cache,
    build_action_prefix_model,
    checkpoint_payload,
    decoder_aligned_geometry_loss,
    decoder_projected_anchor_error,
    flatten_action_prefix_batch,
    load_action_prefix_checkpoint,
)


def _cache() -> dict[str, np.ndarray]:
    count = len(TASKS)
    future = np.zeros(
        (count, BRANCH_COUNT, HORIZON, ACTION_DIM), dtype=np.float32
    )
    future[..., 0] = np.arange(BRANCH_COUNT, dtype=np.float32)[None, :, None]
    return {
        "proposal_id": np.asarray([f"p-{index}" for index in range(count)]),
        "task_class_index": np.arange(count, dtype=np.int64),
        "task_name": np.asarray(TASKS),
        "task_instance_index": np.zeros(count, dtype=np.int64),
        "source_episode_id": np.asarray([f"episode-{index}" for index in range(count)]),
        "split": np.asarray(["train"] * count),
        "history_state": np.zeros(
            (count, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM),
            dtype=np.float32,
        ),
        "history_action": np.zeros(
            (count, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float32
        ),
        "center_action": np.zeros((count, HORIZON, ACTION_DIM), dtype=np.float32),
        "future_action": future,
        "target_state": np.zeros(
            (
                count,
                BRANCH_COUNT,
                HORIZON,
                STATE_SLOT_COUNT,
                LATENT_DIM,
            ),
            dtype=np.float32,
        ),
        "branch_semantic_index": np.tile(
            np.arange(BRANCH_COUNT, dtype=np.int64), (count, 1)
        ),
    }


def test_action_normalization_uses_future_action_train_rows() -> None:
    mean, std = action_normalization_from_cache(_cache())

    assert mean.shape == (ACTION_DIM,)
    assert std.shape == (ACTION_DIM,)
    assert mean[0] == pytest.approx(3.5)
    assert std[0] == pytest.approx(np.std(np.arange(BRANCH_COUNT)))
    assert np.all(std > 0.0)


def test_flatten_action_prefix_batch_preserves_branch_alignment() -> None:
    batch_size = 2
    history = torch.zeros(batch_size, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM)
    history[:, -1, 0, 0] = torch.tensor([10.0, 20.0])
    actions = torch.zeros(batch_size, BRANCH_COUNT, HORIZON, ACTION_DIM)
    target = torch.zeros(
        batch_size,
        BRANCH_COUNT,
        HORIZON,
        STATE_SLOT_COUNT,
        LATENT_DIM,
    )
    for bundle in range(batch_size):
        for branch in range(BRANCH_COUNT):
            actions[bundle, branch, :, 0] = 100 * bundle + branch
            target[bundle, branch, :, 0, 0] = 1000 * bundle + branch
    flattened = flatten_action_prefix_batch(
        {
            "history_embedding": history,
            "future_action": actions,
            "target_embedding": target,
            "task_index": torch.tensor([1, 5]),
        }
    )

    assert flattened.current_state.shape == (
        batch_size * BRANCH_COUNT,
        STATE_SLOT_COUNT,
        LATENT_DIM,
    )
    assert flattened.future_action.shape == (
        batch_size * BRANCH_COUNT,
        HORIZON,
        ACTION_DIM,
    )
    assert flattened.target_state.shape == (
        batch_size * BRANCH_COUNT,
        HORIZON,
        STATE_SLOT_COUNT,
        LATENT_DIM,
    )
    assert flattened.current_state[BRANCH_COUNT, 0, 0] == 20.0
    assert flattened.future_action[BRANCH_COUNT + 3, 0, 0] == 103.0
    assert flattened.target_state[BRANCH_COUNT + 3, 0, 0, 0] == 1003.0
    assert flattened.task_index.tolist() == [1] * BRANCH_COUNT + [5] * BRANCH_COUNT


def test_slot_mse_keeps_bundle_branch_and_horizon_axes() -> None:
    prediction = torch.zeros(2, BRANCH_COUNT, HORIZON, 4, 3)
    target = torch.ones_like(prediction)
    target[..., 0, :] = 2.0

    value = _slot_mse_by_horizon(prediction, target)

    assert value.shape == (2, BRANCH_COUNT, HORIZON)
    torch.testing.assert_close(value, torch.full_like(value, 1.75))


def test_decoder_projected_anchor_error_uses_task_specific_anchor_directions() -> None:
    prediction = torch.zeros(2, 1, STATE_SLOT_COUNT, LATENT_DIM)
    target = torch.zeros_like(prediction)
    prediction[0, 0, 1, 0] = 2.0
    prediction[1, 0, 3, 5] = 3.0
    coefficient = torch.zeros(len(TASKS), 3 * LATENT_DIM, 2)
    coefficient[1, 0, 0] = 4.0
    coefficient[5, 2 * LATENT_DIM + 5, 1] = -2.0

    error = decoder_projected_anchor_error(
        prediction,
        target,
        torch.tensor([1, 5]),
        coefficient,
    )

    assert error.shape == (2, 1, 2)
    torch.testing.assert_close(error[:, 0], torch.tensor([[8.0, 0.0], [0.0, -6.0]]))


def test_decoder_aligned_geometry_loss_normalizes_pixel_axes() -> None:
    prediction = torch.zeros(1, 1, STATE_SLOT_COUNT, LATENT_DIM)
    target = torch.zeros_like(prediction)
    prediction[0, 0, 1, 0] = 1.0
    coefficient = torch.zeros(len(TASKS), 3 * LATENT_DIM, 2)
    coefficient[0, 0] = torch.tensor([9.0, 19.0])

    loss = decoder_aligned_geometry_loss(
        prediction,
        target,
        torch.tensor([0]),
        coefficient,
        image_size=(10, 20),
    )

    assert loss.item() == pytest.approx(1.0)


def test_checkpoint_round_trip_preserves_predictions() -> None:
    config = dict(ACTION_PREFIX_MODEL_CONFIG)
    config.update(
        {
            "state_dim": 12,
            "action_dim": 3,
            "num_slots": 4,
            "num_tasks": len(TASKS),
            "max_horizon": 4,
            "prefix_depth": 1,
            "prefix_heads": 3,
            "prefix_mlp_dim": 24,
            "predictor_depth": 1,
            "predictor_hidden_dim": 24,
            "fusion_dim": 24,
            "state_mlp_dim": 24,
            "dropout": 0.0,
        }
    )
    model = build_action_prefix_model(
        seed=9,
        action_mean=np.zeros(3, dtype=np.float32),
        action_std=np.ones(3, dtype=np.float32),
        model_config=config,
    ).eval()
    current = torch.randn(2, 4, 12)
    action = torch.randn(2, 4, 3)
    task = torch.tensor([0, 6])
    expected = model(current, action, task)
    payload = checkpoint_payload(model, seed=9, step=2000, input_npz="cache.npz")
    payload["model_config"] = config

    restored = load_action_prefix_checkpoint(payload).eval()
    observed = restored(current, action, task)

    torch.testing.assert_close(observed, expected)
