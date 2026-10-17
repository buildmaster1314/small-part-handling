from copy import deepcopy

import pytest
import torch

from fcpg.lewm_anchor_dynamics_pair import (
    MatchedAnchorDynamicsPredictor,
    build_matched_predictor_pair,
    load_matched_predictor_pair_checkpoint,
    matched_pair_forward,
    predictor_kwargs_from_base_checkpoint,
    trainable_parameter_count,
)


def _checkpoint() -> dict:
    return {
        "model_config": {
            "frame": {"latent_dim": 8, "num_keypoints": 2},
            "world_model": {
                "action_dim": 3,
                "history_size": 2,
                "action_smoothing_dim": 2,
                "predictor_depth": 1,
                "predictor_heads": 1,
                "predictor_mlp_dim": 16,
                "predictor_dim_head": 8,
                "predictor_dropout": 0.0,
                "projector_hidden_dim": 16,
            },
        }
    }


def test_matched_pair_starts_identically_and_has_equal_capacity() -> None:
    factual, no_action = build_matched_predictor_pair(
        _checkpoint(), initialization_seed=17
    )
    assert factual.condition_on_action is True
    assert no_action.condition_on_action is False
    assert trainable_parameter_count(factual) == trainable_parameter_count(no_action)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            factual.state_dict().values(), no_action.state_dict().values(), strict=True
        )
    )


def test_no_action_prediction_is_invariant_to_action() -> None:
    _, model = build_matched_predictor_pair(_checkpoint(), initialization_seed=23)
    model.eval()
    embedding = torch.randn(3, 2, 2, 8)
    first = model.predict(embedding, torch.randn(3, 2, 3))
    second = model.predict(embedding, torch.randn(3, 2, 3) * 100.0)
    torch.testing.assert_close(first, second)


def test_matched_forward_reuses_dropout_draws() -> None:
    factual, no_action = build_matched_predictor_pair(
        _checkpoint(), initialization_seed=41
    )
    factual.train()
    no_action.train()
    context = torch.randn(3, 2, 2, 8)
    mean_action = torch.zeros(3, 2, 3)
    target = torch.randn_like(context)
    factual_output, no_action_output = matched_pair_forward(
        factual, no_action, context, mean_action, target
    )
    torch.testing.assert_close(
        factual_output.predicted, no_action_output.predicted, rtol=0.0, atol=0.0
    )


def test_factual_prediction_uses_action_after_conditioning_weights_change() -> None:
    model, _ = build_matched_predictor_pair(_checkpoint(), initialization_seed=29)
    model.eval()
    with torch.no_grad():
        modulation = model.predictor.transformer.layers[0].adaLN_modulation[-1]
        modulation.weight.fill_(0.05)
        modulation.bias.fill_(0.01)
    embedding = torch.randn(3, 2, 2, 8)
    first = model.predict(embedding, torch.zeros(3, 2, 3))
    second = model.predict(embedding, torch.ones(3, 2, 3))
    assert not torch.allclose(first, second)


def test_forward_requires_matching_target_shape() -> None:
    model = MatchedAnchorDynamicsPredictor(
        **predictor_kwargs_from_base_checkpoint(_checkpoint()),
        condition_on_action=True,
    )
    context = torch.randn(2, 2, 2, 8)
    action = torch.randn(2, 2, 3)
    output = model(context, action, deepcopy(context))
    assert output.predicted.shape == context.shape
    assert output.prediction_loss.ndim == 0
    with pytest.raises(ValueError, match="target must have shape"):
        model(context, action, context[:, :, :1])


def test_action_normalization_rejects_bad_std() -> None:
    model, _ = build_matched_predictor_pair(_checkpoint(), initialization_seed=31)
    with pytest.raises(ValueError, match="strictly positive"):
        model.set_action_normalization(
            action_mean=[0.0, 0.0, 0.0],
            action_std=[1.0, 0.0, 1.0],
        )


def test_pair_checkpoint_round_trip(tmp_path) -> None:
    factual, no_action = build_matched_predictor_pair(
        _checkpoint(), initialization_seed=37
    )
    path = tmp_path / "pair.pt"
    torch.save(
        {
            "checkpoint_type": "fcpg_matched_anchor_dynamics_pair_v1",
            "predictor_config": predictor_kwargs_from_base_checkpoint(_checkpoint()),
            "parameter_count_per_head": trainable_parameter_count(factual),
            "factual_state": factual.state_dict(),
            "no_action_state": no_action.state_dict(),
        },
        path,
    )
    loaded_factual, loaded_no_action, _ = load_matched_predictor_pair_checkpoint(path)
    assert loaded_factual.condition_on_action
    assert not loaded_no_action.condition_on_action
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            factual.state_dict().values(),
            loaded_factual.state_dict().values(),
            strict=True,
        )
    )
