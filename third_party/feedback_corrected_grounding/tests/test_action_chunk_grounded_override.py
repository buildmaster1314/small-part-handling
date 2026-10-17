from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fcpg.action_chunk_policy as action_chunk_policy


class _FakeVisionEncoder(nn.Module):
    def __init__(self, output_dim: int, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.backbone_fully_frozen = True

    def forward(self, rgb: torch.Tensor) -> SimpleNamespace:
        pooled = rgb.mean(dim=1, keepdim=True)
        feature_map = pooled.expand(-1, self.output_dim, -1, -1)
        return SimpleNamespace(
            feature_map=feature_map,
            global_feature=feature_map.mean(dim=(-2, -1)),
        )


def _model(
    monkeypatch: pytest.MonkeyPatch,
) -> action_chunk_policy.GroundedActionChunkTransformer:
    monkeypatch.setattr(action_chunk_policy, "ViTDenseEncoder", _FakeVisionEncoder)
    return action_chunk_policy.GroundedActionChunkTransformer(
        mode="geoprop",
        obs_dim=4,
        action_dim=4,
        num_keypoints=3,
        obs_horizon=2,
        action_horizon=4,
        model_dim=32,
        vision_dim=16,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        dropout=0.0,
        pretrained_vision=False,
    ).eval()


def _batch() -> dict[str, torch.Tensor]:
    batch_size, history, anchors = 2, 2, 3
    return {
        "obs": torch.randn(batch_size, history, 4),
        "q_nominal": torch.rand(batch_size, history, anchors, 2) * 31.0,
        "keypoints_xyz": torch.randn(batch_size, history, anchors, 3),
        "sampled_visual": torch.randn(batch_size, history, anchors, 16),
        "global_visual": torch.randn(batch_size, history, 16),
        "image_size": torch.tensor([[32, 32], [32, 32]]),
    }


def _native_grounded(
    model: action_chunk_policy.GroundedActionChunkTransformer,
    batch: dict[str, torch.Tensor],
    *,
    prefix: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    memory, points, reliability = model._observation_tokens(batch, prefix=prefix)
    return memory[:, :, 2:], points, reliability


def test_partial_override_only_replaces_selected_history_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    model = _model(monkeypatch)
    batch = _batch()
    native_memory, native_points, native_reliability = model._observation_tokens(batch)
    native_grounded = native_memory[:, :, 2:]

    # Reuse tokens already sampled and adapted by the policy, but swap only the
    # explicitly selected history/anchor slots.
    replacement = native_grounded.roll(shifts=1, dims=2).clone()
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[0, 0, 1] = True
    mask[1, 1, 2] = True
    overridden_batch = dict(batch)
    overridden_batch["grounded_token_override"] = replacement
    overridden_batch["grounded_token_override_mask"] = mask

    memory, points, reliability = model._observation_tokens(overridden_batch)
    expected_grounded = native_grounded.clone()
    expected_grounded[mask] = replacement[mask]

    assert torch.equal(memory[:, :, :2], native_memory[:, :, :2])
    assert torch.equal(memory[:, :, 2:], expected_grounded)
    assert torch.equal(points, native_points)
    assert torch.equal(reliability, native_reliability)


def test_global_visual_override_only_replaces_selected_history_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(13)
    model = _model(monkeypatch)
    batch = _batch()
    native_memory, native_points, native_reliability = model._observation_tokens(batch)
    replacement = native_memory[:, :, 0] + 5.0
    mask = torch.zeros(2, 2, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 1] = True
    overridden_batch = dict(batch)
    overridden_batch["global_visual_token_override"] = replacement
    overridden_batch["global_visual_token_override_mask"] = mask

    memory, points, reliability = model._observation_tokens(overridden_batch)
    expected_global = native_memory[:, :, 0].clone()
    expected_global[mask] = replacement[mask]

    assert torch.equal(memory[:, :, 0], expected_global)
    assert torch.equal(memory[:, :, 1:], native_memory[:, :, 1:])
    assert torch.equal(points, native_points)
    assert torch.equal(reliability, native_reliability)


def test_empty_global_visual_override_is_bitwise_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(15)
    model = _model(monkeypatch)
    batch = _batch()
    native_memory, _, _ = model._observation_tokens(batch)
    native_output = model.predict_action_chunk(batch)
    batch["global_visual_token_override"] = torch.randn(2, 2, 32)
    batch["global_visual_token_override_mask"] = torch.zeros(
        2, 2, dtype=torch.bool
    )

    memory, _, _ = model._observation_tokens(batch)
    output = model.predict_action_chunk(batch)

    assert torch.equal(memory, native_memory)
    assert torch.equal(output.action, native_output.action)
    assert torch.equal(output.latent, native_output.latent)


def test_absent_and_empty_override_are_bitwise_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(17)
    model = _model(monkeypatch)
    batch = _batch()
    native_memory, native_points, native_reliability = model._observation_tokens(batch)
    native_output = model.predict_action_chunk(batch)

    empty_batch = dict(batch)
    empty_batch["grounded_token_override"] = torch.randn(2, 2, 3, 32)
    empty_batch["grounded_token_override_mask"] = torch.zeros(
        2, 2, 3, dtype=torch.bool
    )
    memory, points, reliability = model._observation_tokens(empty_batch)
    output = model.predict_action_chunk(empty_batch)

    assert torch.equal(memory, native_memory)
    assert torch.equal(points, native_points)
    assert torch.equal(reliability, native_reliability)
    assert torch.equal(output.action, native_output.action)
    assert torch.equal(output.latent, native_output.latent)


def test_prefixed_override_isolated_to_next_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    model = _model(monkeypatch)
    batch = _batch()
    batch.update(
        {
            "next_obs": torch.randn(2, 2, 4),
            "next_q_nominal": torch.rand(2, 2, 3, 2) * 31.0,
            "next_keypoints_xyz": torch.randn(2, 2, 3, 3),
            "next_sampled_visual": torch.randn(2, 2, 3, 16),
            "next_global_visual": torch.randn(2, 2, 16),
        }
    )
    current_memory, _, _ = model._observation_tokens(batch)
    next_grounded, _, _ = _native_grounded(model, batch, prefix="next_")
    replacement = next_grounded + 7.0
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[:, 1, 0] = True
    batch["next_grounded_token_override"] = replacement
    batch["next_grounded_token_override_mask"] = mask

    unchanged_current, _, _ = model._observation_tokens(batch)
    changed_next, _, _ = _native_grounded(model, batch, prefix="next_")
    expected_next = next_grounded.clone()
    expected_next[mask] = replacement[mask]

    assert torch.equal(unchanged_current, current_memory)
    assert torch.equal(changed_next, expected_next)


def test_override_gradient_is_limited_to_selected_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(29)
    model = _model(monkeypatch)
    batch = _batch()
    override = torch.randn(2, 2, 3, 32, requires_grad=True)
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[0, 1, 0] = True
    mask[1, 0, 2] = True
    batch["grounded_token_override"] = override
    batch["grounded_token_override_mask"] = mask

    model(batch).action.sum().backward()

    assert override.grad is not None
    assert bool(torch.any(override.grad[mask] != 0))
    assert torch.equal(
        override.grad[~mask],
        torch.zeros_like(override.grad[~mask]),
    )


@pytest.mark.parametrize(
    ("update", "error", "message"),
    [
        (
            {"grounded_token_override": torch.zeros(2, 2, 3, 32)},
            KeyError,
            "requires both",
        ),
        (
            {"grounded_token_override_mask": torch.zeros(2, 2, 3, dtype=torch.bool)},
            KeyError,
            "requires both",
        ),
        (
            {
                "grounded_token_override": torch.zeros(2, 2, 2, 32),
                "grounded_token_override_mask": torch.zeros(2, 2, 3, dtype=torch.bool),
            },
            ValueError,
            "must have shape",
        ),
        (
            {
                "grounded_token_override": torch.zeros(2, 2, 3, 32),
                "grounded_token_override_mask": torch.zeros(2, 3, 3, dtype=torch.bool),
            },
            ValueError,
            "must have shape",
        ),
        (
            {
                "grounded_token_override": torch.zeros(2, 2, 3, 32, dtype=torch.float64),
                "grounded_token_override_mask": torch.zeros(2, 2, 3, dtype=torch.bool),
            },
            TypeError,
            "dtype must match",
        ),
        (
            {
                "grounded_token_override": torch.zeros(2, 2, 3, 32),
                "grounded_token_override_mask": torch.zeros(2, 2, 3),
            },
            TypeError,
            "torch.bool",
        ),
    ],
)
def test_override_rejects_invalid_contract(
    monkeypatch: pytest.MonkeyPatch,
    update: dict[str, torch.Tensor],
    error: type[Exception],
    message: str,
) -> None:
    model = _model(monkeypatch)
    batch = _batch()
    batch.update(update)

    with pytest.raises(error, match=message):
        model._observation_tokens(batch)


def test_override_rejects_device_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch)
    batch = _batch()
    batch["grounded_token_override"] = torch.empty(2, 2, 3, 32, device="meta")
    batch["grounded_token_override_mask"] = torch.zeros(2, 2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="device must match"):
        model._observation_tokens(batch)


def test_override_valid_mask_rejects_only_selected_nonfinite_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch)
    batch = _batch()
    override = torch.full((2, 2, 3, 32), float("nan"))
    empty_mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    batch["grounded_token_override"] = override
    batch["grounded_token_override_mask"] = empty_mask

    # Invalid placeholder values are allowed outside the explicit valid mask.
    model._observation_tokens(batch)

    batch["grounded_token_override_mask"] = empty_mask.clone()
    batch["grounded_token_override_mask"][0, 1, 2] = True
    with pytest.raises(ValueError, match="non-finite selected tokens"):
        model._observation_tokens(batch)


@pytest.mark.parametrize(
    ("update", "error", "message"),
    [
        (
            {"global_visual_token_override": torch.zeros(2, 2, 32)},
            KeyError,
            "requires both",
        ),
        (
            {
                "global_visual_token_override": torch.zeros(2, 3, 32),
                "global_visual_token_override_mask": torch.zeros(
                    2, 2, dtype=torch.bool
                ),
            },
            ValueError,
            "must have shape",
        ),
        (
            {
                "global_visual_token_override": torch.zeros(2, 2, 32),
                "global_visual_token_override_mask": torch.zeros(2, 2),
            },
            TypeError,
            "torch.bool",
        ),
    ],
)
def test_global_visual_override_rejects_invalid_contract(
    monkeypatch: pytest.MonkeyPatch,
    update: dict[str, torch.Tensor],
    error: type[Exception],
    message: str,
) -> None:
    model = _model(monkeypatch)
    batch = _batch()
    batch.update(update)

    with pytest.raises(error, match=message):
        model._observation_tokens(batch)
