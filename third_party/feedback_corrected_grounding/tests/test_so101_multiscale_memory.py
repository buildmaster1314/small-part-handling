import pytest
import torch
from torch import nn

from fcpg.so101_multiscale_memory import (
    MultiScaleHistoryLeWM,
    rollout_multiscale_candidates,
)
from fcpg.so101_state_residual_lewm import TemporalStateResidualLeWM


class DummyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Linear(1, 4, bias=False)
        self.action_encoder = nn.Linear(12, 4, bias=False)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scalar = batch["pixels"].mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        return {
            "emb": self.image(scalar),
            "act_emb": self.action_encoder(batch["action"]),
        }

    def predict(self, embedding: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        return embedding + action_embedding


def _backbone() -> TemporalStateResidualLeWM:
    model = TemporalStateResidualLeWM(
        DummyBase(),
        state_dim=6,
        latent_dim=4,
        model_dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        max_steps=8,
        latent_residual_cap=0.2,
        state_residual_cap=0.5,
    )
    with torch.no_grad():
        nn.init.normal_(model.latent_head.weight, std=0.1)
        nn.init.normal_(model.state_head.weight, std=0.1)
    return model.eval()


def _model() -> MultiScaleHistoryLeWM:
    return MultiScaleHistoryLeWM(
        _backbone(),
        latent_dim=4,
        model_dim=16,
        heads=4,
        latent_residual_cap=0.1,
        state_residual_cap=0.2,
    )


def _reliability_model() -> MultiScaleHistoryLeWM:
    return MultiScaleHistoryLeWM(
        _backbone(),
        latent_dim=4,
        model_dim=16,
        heads=4,
        latent_residual_cap=0.1,
        state_residual_cap=0.2,
        recent_history=3,
        use_recent_reliability=True,
        use_fourier_age_encoding=True,
    )


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "pixels": torch.randn(2, 4, 3, 2, 2),
        "actions": torch.randn(2, 3, 12),
        "state": torch.randn(2, 4, 6),
        "memory_pixels": torch.randn(2, 4, 3, 2, 2),
        "memory_actions": torch.randn(2, 4, 12),
        "memory_state": torch.randn(2, 4, 6),
        "memory_age": torch.tensor([[1.0, 3.0, 6.0, 12.0]]).expand(2, -1),
    }


def test_zero_initialized_memory_adapter_exactly_preserves_backbone() -> None:
    model = _model().eval()
    values = _inputs()

    output = model(
        values["pixels"],
        values["actions"],
        values["state"],
        memory_pixels=values["memory_pixels"],
        memory_actions=values["memory_actions"],
        memory_state=values["memory_state"],
        memory_age=values["memory_age"],
    )

    assert torch.equal(output.prediction, output.backbone_prediction)
    assert torch.equal(output.state_prediction, output.backbone_state_prediction)
    assert output.adapter_embedding.shape == (2, 3, 16)
    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_trained_heads_make_predictions_depend_on_memory() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    nn.init.normal_(model.latent_head.weight, std=0.2)
    nn.init.normal_(model.state_head.weight, std=0.2)
    values = _inputs()
    context = model.backbone.encode(
        {"pixels": values["pixels"][:, :-1], "action": values["actions"]}
    )["emb"]
    memory_embedding = model.backbone.encode(
        {
            "pixels": values["memory_pixels"],
            "action": values["memory_actions"],
        }
    )["emb"]

    reference = model.predict_context(
        context,
        values["actions"],
        values["state"][:, :-1],
        memory_embedding=memory_embedding,
        memory_actions=values["memory_actions"],
        memory_state=values["memory_state"],
        memory_age=values["memory_age"],
    )
    changed = model.predict_context(
        context,
        values["actions"],
        values["state"][:, :-1],
        memory_embedding=memory_embedding.flip(1),
        memory_actions=values["memory_actions"].flip(1),
        memory_state=values["memory_state"].flip(1),
        memory_age=values["memory_age"],
    )

    assert not torch.allclose(reference[1], changed[1])
    assert not torch.allclose(reference[4], changed[4])


def test_memory_residuals_are_bounded() -> None:
    model = _model().eval()
    nn.init.constant_(model.latent_head.weight, 100.0)
    nn.init.constant_(model.state_head.weight, 100.0)
    values = _inputs()
    output = model(
        values["pixels"],
        values["actions"],
        values["state"],
        memory_pixels=values["memory_pixels"],
        memory_actions=values["memory_actions"],
        memory_state=values["memory_state"],
        memory_age=values["memory_age"],
    )

    assert (output.prediction - output.backbone_prediction).abs().max() <= 0.1 + 1e-6
    assert (
        output.state_prediction - output.backbone_state_prediction
    ).abs().max() <= 0.2 + 1e-6


def test_memory_mask_hides_padded_entries() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    nn.init.normal_(model.latent_head.weight, std=0.2)
    values = _inputs()
    valid = torch.tensor([[True, True, False, False]]).expand(2, -1)
    altered = values["memory_pixels"].clone()
    altered[:, 2:] += 1000.0

    first = model(
        values["pixels"],
        values["actions"],
        values["state"],
        memory_pixels=values["memory_pixels"],
        memory_actions=values["memory_actions"],
        memory_state=values["memory_state"],
        memory_age=values["memory_age"],
        memory_valid=valid,
    )
    second = model(
        values["pixels"],
        values["actions"],
        values["state"],
        memory_pixels=altered,
        memory_actions=values["memory_actions"],
        memory_state=values["memory_state"],
        memory_age=values["memory_age"],
        memory_valid=valid,
    )

    torch.testing.assert_close(first.prediction, second.prediction)


def test_memory_requires_positive_age_and_one_valid_token() -> None:
    model = _model().eval()
    values = _inputs()

    with pytest.raises(ValueError, match="positive"):
        model(
            values["pixels"],
            values["actions"],
            values["state"],
            memory_pixels=values["memory_pixels"],
            memory_actions=values["memory_actions"],
            memory_state=values["memory_state"],
            memory_age=torch.zeros_like(values["memory_age"]),
        )
    with pytest.raises(ValueError, match="at least one"):
        model(
            values["pixels"],
            values["actions"],
            values["state"],
            memory_pixels=values["memory_pixels"],
            memory_actions=values["memory_actions"],
            memory_state=values["memory_state"],
            memory_age=values["memory_age"],
            memory_valid=torch.zeros((2, 4), dtype=torch.bool),
        )


def test_fourier_age_encoding_preserves_token_shape_and_distinguishes_age() -> None:
    model = _reliability_model()
    encoded = model.encode_memory_age(torch.tensor([[1.0, 2.0, 4.0, 8.0]]))
    assert encoded.shape == (1, 4, 10)
    assert not torch.allclose(encoded[:, 0], encoded[:, 1])


def test_reliability_encoding_changes_trained_adapter_prediction() -> None:
    torch.manual_seed(29)
    model = _reliability_model().eval()
    nn.init.normal_(model.latent_head.weight, std=0.2)
    values = _inputs()
    clean = model(
        values["pixels"],
        values["actions"],
        values["state"],
        memory_pixels=values["memory_pixels"],
        memory_actions=values["memory_actions"],
        memory_state=values["memory_state"],
        memory_age=values["memory_age"],
        recent_visual_valid=torch.ones(2, 3),
    )
    dropped = model(
        values["pixels"],
        values["actions"],
        values["state"],
        memory_pixels=values["memory_pixels"],
        memory_actions=values["memory_actions"],
        memory_state=values["memory_state"],
        memory_age=values["memory_age"],
        recent_visual_valid=torch.zeros(2, 3),
    )
    assert not torch.allclose(clean.prediction, dropped.prediction)


def test_reliability_encoding_validates_shape_and_range() -> None:
    model = _reliability_model().eval()
    values = _inputs()
    kwargs = {
        "memory_pixels": values["memory_pixels"],
        "memory_actions": values["memory_actions"],
        "memory_state": values["memory_state"],
        "memory_age": values["memory_age"],
    }
    with pytest.raises(ValueError, match="shape"):
        model(
            values["pixels"],
            values["actions"],
            values["state"],
            recent_visual_valid=torch.ones(2, 2),
            **kwargs,
        )
    with pytest.raises(ValueError, match="values"):
        model(
            values["pixels"],
            values["actions"],
            values["state"],
            recent_visual_valid=torch.full((2, 3), 2.0),
            **kwargs,
        )


def test_multiscale_candidate_rollout_preserves_shapes_and_is_action_conditioned() -> None:
    torch.manual_seed(31)
    model = _model().eval()
    nn.init.normal_(model.latent_head.weight, std=0.2)
    nn.init.normal_(model.state_head.weight, std=0.2)
    values = _inputs()
    context = model.backbone.encode(
        {"pixels": values["pixels"][:1, :-1], "action": values["actions"][:1]}
    )["emb"][0]
    memory = model.backbone.encode(
        {
            "pixels": values["memory_pixels"][:1],
            "action": values["memory_actions"][:1],
        }
    )["emb"][0]
    candidate_actions = torch.stack(
        (
            torch.zeros(3, 12),
            torch.ones(3, 12),
        )
    )
    latent, state = rollout_multiscale_candidates(
        model,
        context,
        values["state"][0, :-1],
        values["actions"][0, :2],
        candidate_actions,
        memory_embedding=memory,
        memory_actions=values["memory_actions"][0],
        memory_state=values["memory_state"][0],
        memory_age=values["memory_age"][0],
    )
    assert latent.shape == (2, 3, 4)
    assert state.shape == (2, 3, 6)
    assert torch.isfinite(latent).all()
    assert torch.isfinite(state).all()
    assert not torch.allclose(latent[0], latent[1])
