from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_a4c_coarse_proposer import (
    A4CCoarseActionProposer,
    checkpoint_payload,
    load_checkpoint,
    weighted_action_loss,
)
from scripts.train_lewm_a4c_coarse_proposer import (
    _apply_input_mode,
    _cuda0,
    _different_bundle_permutation,
)


def _inputs(batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.randn(batch_size, 4, 192)
    history = torch.empty(batch_size, 2, 20).uniform_(-1.0, 1.0)
    task = torch.arange(batch_size, dtype=torch.long) % 7
    return state, history, task


def test_zero_initialized_proposer_repeats_latest_action_chunk() -> None:
    model = A4CCoarseActionProposer()
    state, history, task = _inputs()
    prediction = model(state, history, task)
    expected = history[:, -1:, :].expand(-1, 8, -1)
    torch.testing.assert_close(prediction, expected)
    assert prediction.shape == (3, 8, 20)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_proposer_rejects_invalid_contracts() -> None:
    model = A4CCoarseActionProposer()
    state, history, task = _inputs()
    with pytest.raises(ValueError, match="current_state"):
        model(state[:, :3], history, task)
    with pytest.raises(ValueError, match="history_action"):
        model(state, history[:, :1], task)
    with pytest.raises(TypeError, match="torch.long"):
        model(state, history, task.float())


def test_camera_conditioned_proposer_requires_camera_features() -> None:
    model = A4CCoarseActionProposer(camera_feature_dim=12)
    state, history, task = _inputs()
    camera = torch.randn(state.shape[0], 12)
    prediction = model(state, history, task, camera_condition=camera)
    assert prediction.shape == (state.shape[0], 8, 20)
    with pytest.raises(ValueError, match="camera_condition"):
        model(state, history, task)


def test_weighted_action_loss_separates_xyz_and_gripper() -> None:
    prediction = torch.zeros(2, 8, 20)
    target = torch.zeros_like(prediction)
    target[0, :, 0] = 1.0
    target[1, :, 3] = 1.0
    loss, parts = weighted_action_loss(
        prediction,
        target,
        torch.tensor([2.0, 1.0]),
    )
    assert float(loss) > 0.0
    assert float(parts["xyz_loss"]) > 0.0
    assert float(parts["gripper_loss"]) > 0.0


def test_checkpoint_round_trip() -> None:
    model = A4CCoarseActionProposer(hidden_dim=64, slot_dim=32, history_dim=16)
    payload = checkpoint_payload(
        model,
        step=2_000,
        seed=40_591,
        parent_checkpoint="parent.pt",
        control_index="index.npz",
    )
    restored = load_checkpoint(payload)
    state, history, task = _inputs()
    torch.testing.assert_close(
        restored(state, history, task),
        model(state, history, task),
    )


def test_diagnostic_permutation_changes_physical_bundle_within_task() -> None:
    rows = np.arange(42, dtype=np.int64)
    task = np.repeat(np.arange(7, dtype=np.int64), 6)
    bundle = np.concatenate(
        [100 * index + np.repeat(np.arange(3, dtype=np.int64), 2) for index in range(7)]
    )
    permutation = _different_bundle_permutation(task, bundle, rows)
    assert np.array_equal(task[permutation], task[rows])
    assert np.all(bundle[permutation] != bundle[rows])


def test_cuda1_is_rejected_before_cuda_availability_check() -> None:
    with pytest.raises(ValueError, match="only CUDA0"):
        _cuda0("cuda:1")


def test_input_modes_remove_only_the_requested_signal() -> None:
    state, history, _task = _inputs()
    state_only = _apply_input_mode(state, history, "state_only")
    assert torch.equal(state_only[0], state)
    assert torch.count_nonzero(state_only[1]) == 0
    history_only = _apply_input_mode(state, history, "history_only")
    assert torch.count_nonzero(history_only[0]) == 0
    assert torch.equal(history_only[1], history)
