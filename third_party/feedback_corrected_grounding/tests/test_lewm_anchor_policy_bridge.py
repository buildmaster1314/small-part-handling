from __future__ import annotations

import pytest
import torch

from fcpg.lewm_anchor_policy_bridge import (
    ResidualAnchorPolicyBridge,
    build_matched_bridge_pair,
)


def _inputs(*, batch: int = 2, history: int | None = None):
    leading = (batch,) if history is None else (batch, history)
    return (
        torch.randn(*leading, 3, 192),
        torch.randn(*leading, 3, 192),
        torch.randn(*leading, 3, 256),
        torch.randn(*leading, 3, 3),
        torch.randn(*leading, 3, 3),
    )


def test_default_bridge_has_frozen_small_parameter_budget() -> None:
    bridge = ResidualAnchorPolicyBridge()
    assert bridge.trainable_parameter_count == 41_664


def test_zero_initialized_bridge_is_exact_hold_last_for_sequence_axes() -> None:
    bridge = ResidualAnchorPolicyBridge()
    values = _inputs(history=4)
    result = bridge(*values)
    assert torch.equal(result, values[2])


def test_matched_pair_starts_byte_identically_and_is_independent() -> None:
    factual, no_action = build_matched_bridge_pair(initialization_seed=123)
    for factual_value, no_action_value in zip(
        factual.state_dict().values(), no_action.state_dict().values(), strict=True
    ):
        assert torch.equal(factual_value, no_action_value)
        assert factual_value.data_ptr() != no_action_value.data_ptr()


def test_bridge_learns_a_nonzero_residual_and_propagates_gradients() -> None:
    bridge = ResidualAnchorPolicyBridge()
    with torch.no_grad():
        bridge.output_projection.weight.fill_(0.01)
    values = list(_inputs())
    values[0].requires_grad_(True)
    result = bridge(*values)
    assert not torch.equal(result, values[2])
    result.square().mean().backward()
    assert values[0].grad is not None
    assert bool(torch.isfinite(values[0].grad).all())


def test_normalization_requires_positive_finite_scales() -> None:
    bridge = ResidualAnchorPolicyBridge()
    kwargs = {
        "context_mean": torch.zeros(192),
        "context_std": torch.ones(192),
        "latent_delta_mean": torch.zeros(192),
        "latent_delta_std": torch.ones(192),
        "xyz_delta_mean": torch.zeros(3),
        "xyz_delta_std": torch.ones(3),
    }
    kwargs["latent_delta_std"][0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        bridge.set_input_normalization(**kwargs)


def test_forward_rejects_misaligned_anchor_shape() -> None:
    bridge = ResidualAnchorPolicyBridge()
    values = list(_inputs())
    values[3] = torch.randn(2, 4, 3)
    with pytest.raises(ValueError, match="current_xyz"):
        bridge(*values)


def test_standardized_token_loss_detaches_teacher_target() -> None:
    bridge = ResidualAnchorPolicyBridge()
    prediction = torch.randn(2, 3, 256, requires_grad=True)
    target = torch.randn(2, 3, 256, requires_grad=True)
    loss = bridge.standardized_token_loss(prediction, target)
    loss.backward()
    assert prediction.grad is not None
    assert target.grad is None
