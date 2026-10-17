from __future__ import annotations

import pytest
import numpy as np
import torch

from fcpg.lewm_action_prefix import ActionPrefixDynamics
from fcpg.lewm_history_conditioned_metric3d_a4 import (
    CausalActionHistoryConditioner,
)
from scripts.train_lewm_history_conditioned_metric3d_a4 import (
    _mismatched_history_cache_indices,
)


def _base_dynamics() -> ActionPrefixDynamics:
    torch.manual_seed(7)
    model = ActionPrefixDynamics(
        action_dim=20,
        state_dim=192,
        num_slots=4,
        num_tasks=7,
        max_horizon=8,
        prefix_depth=1,
        prefix_heads=6,
        predictor_depth=1,
        predictor_hidden_dim=64,
        fusion_dim=64,
        state_mlp_dim=64,
        dropout=0.0,
    ).eval()
    # Production checkpoints have trained AdaLN modulation. The constructor
    # intentionally starts those layers at zero, which would also make the
    # synthetic parent's derivative with respect to its condition zero.
    with torch.no_grad():
        for layer in model.residual_predictor.layers:
            layer.modulation[-1].weight.normal_(std=0.01)
            layer.modulation[-1].bias.normal_(std=0.01)
    return model


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    return (
        torch.randn(2, 4, 192),
        torch.randn(2, 2, 20),
        torch.randn(2, 8, 20),
        torch.tensor([0, 6]),
    )


def test_history_conditioner_is_exact_parent_at_initialization() -> None:
    base = _base_dynamics()
    model = CausalActionHistoryConditioner(base, hidden_dim=32).eval()
    current, history, future, task = _inputs()

    expected = base(current, future, task_index=task)
    actual = model(current, history, future, task_index=task)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert sum(parameter.numel() for parameter in model.condition_adapter.parameters()) > 0


def test_history_conditioner_receives_gradient_without_unfreezing_parent() -> None:
    base = _base_dynamics()
    model = CausalActionHistoryConditioner(base, hidden_dim=32)
    current, history, future, task = _inputs()

    model(current, history, future, task_index=task).square().mean().backward()

    trainable_gradients = [
        parameter.grad
        for parameter in model.condition_adapter.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and bool(torch.any(gradient != 0)) for gradient in trainable_gradients)
    assert all(parameter.grad is None for parameter in base.parameters())


def test_history_conditioner_rejects_noncausal_shape() -> None:
    model = CausalActionHistoryConditioner(_base_dynamics(), hidden_dim=32)
    current, _history, future, task = _inputs()
    with pytest.raises(ValueError, match="history_action"):
        model(current, torch.zeros(2, 3, 20), future, task_index=task)


def test_mismatched_history_uses_same_task_different_physical_bundle() -> None:
    index = {
        "task_name": np.asarray(["pick", "pick", "pick", "pick", "push", "push"]),
        "cache_index": np.asarray([10, 10, 20, 20, 30, 40]),
        "split": np.asarray(["development"] * 6),
    }
    bundles = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)

    mismatch = _mismatched_history_cache_indices(index, bundles)

    np.testing.assert_array_equal(mismatch, [20, 20, 10, 10, 40, 30])
    assert np.all(mismatch != index["cache_index"][bundles])
