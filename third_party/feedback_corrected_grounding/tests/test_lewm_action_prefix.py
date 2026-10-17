from __future__ import annotations

import pytest
import torch
from torch import nn

from fcpg.lewm_action_prefix import ActionPrefixDynamics, dense_prefix_mse


STATE_DIM = 12
ACTION_DIM = 3
SLOTS = 4


def _small_model(*, num_tasks: int | None = None) -> ActionPrefixDynamics:
    return ActionPrefixDynamics(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        num_slots=SLOTS,
        num_tasks=num_tasks,
        max_horizon=6,
        prefix_depth=2,
        prefix_heads=3,
        predictor_depth=2,
        dropout=0.0,
        prefix_mlp_dim=24,
        predictor_hidden_dim=24,
    )


def test_shape_and_optional_task_conditioning() -> None:
    torch.manual_seed(11)
    model = _small_model(num_tasks=3)
    current = torch.randn(2, SLOTS, STATE_DIM)
    actions = torch.randn(2, 5, ACTION_DIM)

    prediction = model(current, actions, task_index=torch.tensor([0, 2]))

    assert prediction.shape == (2, 5, SLOTS, STATE_DIM)
    assert torch.isfinite(prediction).all()
    assert model.position_encoding.requires_grad is False
    torch.testing.assert_close(
        model.position_encoding[0, 0, 0::2],
        torch.zeros_like(model.position_encoding[0, 0, 0::2]),
    )
    torch.testing.assert_close(
        model.position_encoding[0, 0, 1::2],
        torch.ones_like(model.position_encoding[0, 0, 1::2]),
    )


def test_zero_initialized_residual_is_exact_identity() -> None:
    torch.manual_seed(12)
    model = _small_model(num_tasks=2)
    current = torch.randn(2, SLOTS, STATE_DIM)
    actions = torch.randn(2, 4, ACTION_DIM)

    prediction = model(current, actions, task_index=torch.tensor([0, 1]))
    expected = current[:, None].expand_as(prediction)

    for layer in model.residual_predictor.layers:
        assert torch.count_nonzero(layer.modulation[-1].weight) == 0
        assert torch.count_nonzero(layer.modulation[-1].bias) == 0
    assert torch.equal(prediction, expected)


def test_action_normalization_is_applied_before_embedding() -> None:
    torch.manual_seed(121)
    model = _small_model().eval()
    current = torch.randn(2, SLOTS, STATE_DIM)
    normalized = torch.randn(2, 4, ACTION_DIM)
    mean = torch.tensor([2.0, -1.0, 0.5])
    std = torch.tensor([0.5, 2.0, 4.0])
    model.set_action_normalization(mean, std)
    raw = normalized * std + mean

    with torch.no_grad():
        observed = model.encode_prefix(current, raw)
        model.set_action_normalization(
            torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM)
        )
        expected = model.encode_prefix(current, normalized)

    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)


def test_causal_prefix_is_invariant_to_later_actions() -> None:
    torch.manual_seed(13)
    model = _small_model().eval()
    current = torch.randn(2, SLOTS, STATE_DIM)
    actions = torch.randn(2, 6, ACTION_DIM)
    with torch.no_grad():
        for layer in model.residual_predictor.layers:
            nn.init.normal_(layer.modulation[-1].weight, std=0.05)
        baseline_context = model.encode_prefix(current, actions)
        baseline_prediction = model(current, actions)

        cutoff = 2
        changed_actions = actions.clone()
        changed_actions[:, cutoff + 1 :] = 50.0 * torch.randn_like(
            changed_actions[:, cutoff + 1 :]
        )
        changed_context = model.encode_prefix(current, changed_actions)
        changed_prediction = model(current, changed_actions)

    torch.testing.assert_close(
        changed_context[:, : cutoff + 1],
        baseline_context[:, : cutoff + 1],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        changed_prediction[:, : cutoff + 1],
        baseline_prediction[:, : cutoff + 1],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(
        changed_context[:, cutoff + 1 :],
        baseline_context[:, cutoff + 1 :],
    )


def test_each_horizon_predicts_from_the_same_current_state() -> None:
    torch.manual_seed(14)
    model = _small_model()
    current = torch.randn(2, SLOTS, STATE_DIM)
    actions = torch.randn(2, 5, ACTION_DIM)
    captured: list[torch.Tensor] = []

    def capture_predictor_input(
        _module: nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(args[0].detach().clone())

    handle = model.residual_predictor.register_forward_pre_hook(
        capture_predictor_input
    )
    try:
        model(current, actions)
    finally:
        handle.remove()

    assert len(captured) == 1
    expected = current[:, None].expand(2, 5, SLOTS, STATE_DIM)
    assert torch.equal(captured[0], expected)


def test_dense_prefix_mse_weights_context_and_anchor_slots() -> None:
    pred = torch.zeros(2, 3, SLOTS, 5)
    target = torch.ones_like(pred)
    target[:, :, 0] = 2.0

    loss = dense_prefix_mse(pred, target)
    custom = dense_prefix_mse(
        pred,
        target,
        context_weight=0.5,
        anchor_weight=2.0,
    )

    assert loss.item() == pytest.approx(0.25 * 4.0 + 0.75 * 1.0)
    assert custom.item() == pytest.approx(0.5 * 4.0 + 2.0 * 1.0)


def test_dense_prefix_mse_can_emphasize_late_horizons() -> None:
    pred = torch.zeros(1, 4, SLOTS, 2)
    target = torch.zeros_like(pred)
    target[:, -1, 1:] = 1.0

    uniform = dense_prefix_mse(pred, target)
    late = dense_prefix_mse(
        pred,
        target,
        horizon_weights=torch.tensor([1.0, 1.0, 1.0, 4.0]),
    )

    assert late > uniform
    assert late.item() == pytest.approx(uniform.item() * 4.0 / 1.75)


def test_dense_prefix_mse_rejects_bad_shape_and_nonfinite_values() -> None:
    valid = torch.zeros(2, 3, SLOTS, 5)

    with pytest.raises(ValueError, match="share shape"):
        dense_prefix_mse(valid, torch.zeros(2, 4, SLOTS, 5))
    with pytest.raises(ValueError, match=r"\[B, H, 4, D\]"):
        dense_prefix_mse(torch.zeros(2, 3, 3, 5), torch.zeros(2, 3, 3, 5))
    nonfinite = valid.clone()
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="target contains non-finite"):
        dense_prefix_mse(valid, nonfinite)
    with pytest.raises(ValueError, match="at least one"):
        dense_prefix_mse(valid, valid, context_weight=0.0, anchor_weight=0.0)


def test_dense_prediction_backpropagates_through_state_and_action_paths() -> None:
    torch.manual_seed(15)
    model = _small_model(num_tasks=2)
    with torch.no_grad():
        for layer in model.residual_predictor.layers:
            nn.init.normal_(layer.modulation[-1].weight, std=0.05)

    current = torch.randn(2, SLOTS, STATE_DIM, requires_grad=True)
    actions = torch.randn(2, 5, ACTION_DIM, requires_grad=True)
    target = torch.randn(2, 5, SLOTS, STATE_DIM)
    prediction = model(current, actions, task_index=torch.tensor([0, 1]))

    loss = dense_prefix_mse(prediction, target)
    loss.backward()

    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert current.grad.abs().sum() > 0
    assert actions.grad.abs().sum() > 0
    output_grad = model.residual_predictor.layers[-1].modulation[-1].weight.grad
    assert output_grad is not None and output_grad.abs().sum() > 0
    parameter_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert parameter_grads and all(grad is not None for grad in parameter_grads)
    assert all(torch.isfinite(grad).all() for grad in parameter_grads if grad is not None)
