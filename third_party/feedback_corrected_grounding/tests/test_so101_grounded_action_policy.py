from __future__ import annotations

import pytest
import torch

from fcpg.so101_grounded_action_policy import SO101GroundedActionChunkPolicy


def model() -> SO101GroundedActionChunkPolicy:
    return SO101GroundedActionChunkPolicy(
        latent_dim=12,
        state_dim=6,
        num_views=2,
        num_anchors=3,
        obs_horizon=3,
        action_horizon=10,
        raw_steps_per_chunk=5,
        model_dim=12,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        ffn_dim=24,
    )


def inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(batch, 3, 2, 12, generator=generator),
        torch.randn(batch, 3, 2, 3, 12, generator=generator),
        torch.ones(batch, 3, 2, 3, dtype=torch.bool),
        0.1 * torch.randn(batch, 3, 6, generator=generator),
        0.1 * torch.randn(batch, 2, 30, generator=generator),
    )


def test_zero_initialized_policy_holds_current_state() -> None:
    policy = model().eval()
    context, anchors, valid, state, past = inputs()
    output = policy(context, anchors, valid, state, past)

    assert output.action.shape == (2, 10, 6)
    assert output.action_chunks.shape == (2, 2, 30)
    torch.testing.assert_close(output.action, state[:, -1:, :].expand(-1, 10, -1))
    torch.testing.assert_close(output.delta, torch.zeros_like(output.delta))


def test_per_step_motion_and_absolute_limits_are_enforced() -> None:
    policy = model().eval()
    policy.set_action_contract(
        step_cap=torch.full((6,), 0.03),
        lower=torch.full((6,), -0.2),
        upper=torch.full((6,), 0.2),
    )
    with torch.no_grad():
        policy.delta_head[-1].bias.fill_(20.0)
    context, anchors, valid, state, past = inputs(batch=1)
    state[:, -1] = 0.19
    output = policy(context, anchors, valid, state, past)
    transitions = torch.diff(torch.cat((state[:, -1:, :], output.action), dim=1), dim=1)

    assert float(transitions.abs().max()) <= 0.030001
    assert float(output.action.max()) <= 0.200001
    assert float(output.action.min()) >= -0.200001


def test_masked_anchor_values_cannot_change_actions() -> None:
    policy = model().eval()
    context, anchors, valid, state, past = inputs(batch=1)
    valid[:, 1, 0, 2] = False
    altered = anchors.clone()
    altered[:, 1, 0, 2] = torch.nan

    first = policy(context, anchors, valid, state, past).action
    second = policy(context, altered, valid, state, past).action
    torch.testing.assert_close(first, second)


def test_policy_rejects_wrong_history_timing() -> None:
    policy = model()
    context, anchors, valid, state, past = inputs(batch=1)
    with pytest.raises(ValueError, match="past_action_chunks"):
        policy(context, anchors, valid, state, past[:, :1])


def test_loss_masks_padded_tail() -> None:
    policy = model()
    context, anchors, valid, state, past = inputs(batch=1)
    target = state[:, -1:, :].expand(-1, 10, -1).clone()
    target[:, 8:] = 1000.0
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[:, 8:] = False

    loss, parts, _ = policy.compute_loss(
        context, anchors, valid, state, past, target, mask
    )
    assert float(loss) == pytest.approx(0.0, abs=1.0e-7)
    assert float(parts["action_loss"]) == pytest.approx(0.0, abs=1.0e-7)
    assert float(parts["velocity_loss"]) == pytest.approx(0.0, abs=1.0e-7)
