from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_expert_action_residual import (
    ACTION_STEPS,
    ExpertActionResidualHead,
    ExpertResidualArrays,
    METHODS,
    PROTOCOL,
    load_expert_action_residual_head,
    load_expert_residual_arrays,
    save_expert_residual_arrays,
    validate_expert_residual_arrays,
)


def _arrays(count: int = 6, model_dim: int = 12) -> ExpertResidualArrays:
    rng = np.random.default_rng(7)
    methods = len(METHODS)
    return ExpertResidualArrays(
        episode_id=np.repeat(np.arange(2, dtype=np.int64), count // 2),
        decision_index=np.tile(np.arange(count // 2, dtype=np.int64), 2),
        missing_age=np.tile(np.arange(1, count // 2 + 1, dtype=np.int64), 2),
        sensor_valid=np.zeros(count, dtype=bool),
        state_latent=rng.normal(size=(count, methods, model_dim)).astype(np.float32),
        action_latent=rng.normal(
            size=(count, methods, ACTION_STEPS, model_dim)
        ).astype(np.float32),
        base_action=rng.uniform(
            -1, 1, size=(count, methods, ACTION_STEPS, 4)
        ).astype(np.float32),
        posterior_token=rng.normal(
            size=(count, methods, 3, model_dim)
        ).astype(np.float32),
        expert_action=rng.uniform(-1, 1, size=(count, ACTION_STEPS, 4)).astype(
            np.float32
        ),
    )


def test_expert_residual_arrays_round_trip(tmp_path) -> None:
    arrays = _arrays()
    path = tmp_path / "cache.npz"
    save_expert_residual_arrays(path, arrays)
    loaded = load_expert_residual_arrays(path)
    assert loaded.sample_count == arrays.sample_count
    assert loaded.model_dim == arrays.model_dim
    np.testing.assert_array_equal(loaded.episode_id, arrays.episode_id)
    np.testing.assert_allclose(loaded.action_latent, arrays.action_latent)


def test_expert_residual_arrays_reject_visible_training_rows() -> None:
    arrays = _arrays()
    arrays.sensor_valid[0] = True
    with pytest.raises(PermissionError, match="invalid observations"):
        validate_expert_residual_arrays(arrays)


def test_zero_initialized_head_is_exact_identity() -> None:
    torch.manual_seed(3)
    model = ExpertActionResidualHead(model_dim=16, hidden_dim=24)
    base = torch.rand(4, ACTION_STEPS, 4) * 2 - 1
    result = model(
        base_action=base,
        state_latent=torch.randn(4, 16),
        action_latent=torch.randn(4, ACTION_STEPS, 16),
        missing_age=torch.tensor([1, 2, 5, 6], dtype=torch.long),
    )
    assert torch.equal(result, base)


def test_valid_rows_bypass_nonzero_residual_exactly() -> None:
    model = ExpertActionResidualHead(model_dim=8, hidden_dim=12)
    with torch.no_grad():
        model.output_projection.bias.fill_(0.75)
    base = torch.zeros(2, ACTION_STEPS, 4)
    result = model(
        base_action=base,
        state_latent=torch.randn(2, 8),
        action_latent=torch.randn(2, ACTION_STEPS, 8),
        missing_age=torch.tensor([0, 3], dtype=torch.long),
        sensor_valid=torch.tensor([True, False]),
    )
    assert torch.equal(result[0], base[0])
    assert not torch.equal(result[1], base[1])
    assert torch.equal(result[1, :, 3], base[1, :, 3])
    assert bool(torch.all(result <= 1.0) and torch.all(result >= -1.0))


def test_residual_output_respects_action_bounds_at_saturation() -> None:
    model = ExpertActionResidualHead(model_dim=8, hidden_dim=12)
    with torch.no_grad():
        model.output_projection.weight.fill_(10.0)
        model.output_projection.bias.fill_(10.0)
    base = torch.tensor([-1.0, -0.5, 0.5, 1.0]).view(1, 1, 4).expand(
        1, ACTION_STEPS, 4
    )
    result = model(
        base_action=base,
        state_latent=torch.ones(1, 8),
        action_latent=torch.ones(1, ACTION_STEPS, 8),
        missing_age=torch.tensor([6], dtype=torch.long),
    )
    assert bool(torch.all(result <= 1.0) and torch.all(result >= -1.0))
    assert torch.equal(result[..., 3], base[..., 3])


def test_checkpoint_loader_freezes_and_validates_identity(tmp_path) -> None:
    source = ExpertActionResidualHead(model_dim=8, hidden_dim=12)
    path = tmp_path / "head.pt"
    torch.save(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "method": "complementary_factual",
            "seed": 17,
            "steps": 2000,
            "model_dim": 8,
            "hidden_dim": 12,
            "model_state": source.state_dict(),
        },
        path,
    )
    loaded, metadata = load_expert_action_residual_head(
        path,
        device=torch.device("cpu"),
        expected_method="complementary_factual",
        expected_seed=17,
    )
    assert metadata["steps"] == 2000
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    with pytest.raises(ValueError, match="method mismatch"):
        load_expert_action_residual_head(
            path,
            device=torch.device("cpu"),
            expected_method="geometry_hold",
        )
