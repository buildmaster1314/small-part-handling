import torch
from torch import nn
import pytest

from fcpg.so101_state_residual_lewm import (
    StateResidualLeWM,
    TemporalStateResidualLeWM,
)
from scripts.train_so101_state_residual_lewm import (
    load_pretrained_base,
    load_transfer_checkpoint,
    normalize_state,
    set_training_mode,
    state_action_normalization,
    validate_base_normalization,
)


class DummyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Linear(1, 4, bias=False)
        self.action_encoder = nn.Linear(12, 4, bias=False)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pixels = batch["pixels"]
        scalar = pixels.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        return {
            "emb": self.image(scalar),
            "act_emb": self.action_encoder(batch["action"]),
        }

    def predict(self, embedding: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        return embedding + action_embedding


def _model() -> StateResidualLeWM:
    return StateResidualLeWM(
        DummyBase(),
        state_dim=6,
        latent_dim=4,
        hidden_dim=16,
        latent_residual_cap=0.2,
        state_residual_cap=0.5,
    )


def _temporal_model(*, max_steps: int = 8) -> TemporalStateResidualLeWM:
    return TemporalStateResidualLeWM(
        DummyBase(),
        state_dim=6,
        latent_dim=4,
        model_dim=16,
        layers=2,
        heads=4,
        ffn_dim=32,
        max_steps=max_steps,
        latent_residual_cap=0.2,
        state_residual_cap=0.5,
    )


def test_zero_initialized_model_preserves_base_and_command_baselines() -> None:
    model = _model()
    pixels = torch.randn(2, 4, 3, 2, 2)
    actions = torch.randn(2, 3, 12)
    state = torch.randn(2, 4, 6)

    output = model(pixels, actions, state)

    assert output.prediction.shape == (2, 3, 4)
    assert output.state_prediction.shape == (2, 3, 6)
    assert torch.equal(output.prediction, output.base_prediction)
    assert torch.equal(output.state_prediction, actions.reshape(2, 3, 2, 6)[:, :, -1])


def test_predict_context_matches_forward_without_future_target() -> None:
    model = _model()
    pixels = torch.randn(2, 4, 3, 2, 2)
    actions = torch.randn(2, 3, 12)
    state = torch.randn(2, 4, 6)

    output = model(pixels, actions, state)
    context = output.embedding[:, :-1]
    direct = model.predict_context(context, actions, state[:, :-1])

    torch.testing.assert_close(direct[0], output.base_prediction)
    torch.testing.assert_close(direct[1], output.prediction)
    torch.testing.assert_close(direct[2], output.command_state)
    torch.testing.assert_close(direct[3], output.state_prediction)


def test_residual_outputs_are_bounded() -> None:
    model = _model()
    nn.init.constant_(model.latent_head.weight, 100.0)
    nn.init.constant_(model.state_head.weight, 100.0)
    pixels = torch.randn(2, 4, 3, 2, 2)
    actions = torch.randn(2, 3, 12)
    state = torch.randn(2, 4, 6)

    output = model(pixels, actions, state)

    assert (output.prediction - output.base_prediction).abs().max() <= 0.2 + 1e-6
    assert (output.state_prediction - output.command_state).abs().max() <= 0.5 + 1e-6


@pytest.mark.parametrize("factory", [_model, _temporal_model])
def test_residual_scale_interpolates_from_command_baseline(factory) -> None:
    torch.manual_seed(41)
    model = factory().eval()
    nn.init.normal_(model.latent_head.weight, std=0.1)
    nn.init.normal_(model.state_head.weight, std=0.1)
    context = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 12)
    state = torch.randn(2, 3, 6)

    zero = model.predict_context(context, actions, state, residual_scale=0.0)
    half = model.predict_context(context, actions, state, residual_scale=0.5)
    full = model.predict_context(context, actions, state, residual_scale=1.0)

    torch.testing.assert_close(zero[1], zero[0])
    torch.testing.assert_close(zero[3], zero[2])
    torch.testing.assert_close(half[1], torch.lerp(zero[1], full[1], 0.5))
    torch.testing.assert_close(half[3], torch.lerp(zero[3], full[3], 0.5))


@pytest.mark.parametrize("scale", [-0.01, 1.01, float("nan"), float("inf")])
def test_residual_scale_rejects_values_outside_unit_interval(scale: float) -> None:
    model = _model()

    with pytest.raises(ValueError, match="residual_scale"):
        model.predict_context(
            torch.randn(1, 3, 4),
            torch.randn(1, 3, 12),
            torch.randn(1, 3, 6),
            residual_scale=scale,
        )


def test_rejects_incomplete_state_or_action_dimensions() -> None:
    model = _model()
    pixels = torch.randn(2, 4, 3, 2, 2)
    state = torch.randn(2, 4, 6)

    try:
        model(pixels, torch.randn(2, 3, 11), state)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("invalid action width was accepted")

    try:
        model(pixels, torch.randn(2, 3, 12), torch.randn(2, 4, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("five-dimensional state was accepted")


def test_nested_training_checkpoint_initializes_base(tmp_path) -> None:
    source = DummyBase()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "step": 10000,
            "sigreg_weight": 0.02,
            "run_config": {
                "contract": {
                    "protocol": "so101-shared-lewm-v3",
                    "action_normalization": "physical-scale",
                    "action_center": [0.0] * 6,
                    "action_scale": [100.0] * 6,
                }
            },
        },
        path,
    )
    target = DummyBase()

    report = load_pretrained_base(target, path)

    assert report["step"] == 10000
    assert report["loaded_tensors"] == len(source.state_dict())
    assert report["action_normalization"] == "physical-scale"
    assert all(
        torch.equal(target.state_dict()[key], value)
        for key, value in source.state_dict().items()
    )


def test_complete_transfer_checkpoint_initializes_residual_model(tmp_path) -> None:
    source = _temporal_model()
    with torch.no_grad():
        source.state_head.bias.fill_(0.25)
    path = tmp_path / "transfer.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "step": 10000,
            "sigreg_weight": 0.02,
            "run_config": {
                "contract": {
                    "protocol": "so101-state-residual-lewm-v5-temporal",
                    "normalization_center": [0.0] * 6,
                    "normalization_scale": [100.0] * 6,
                }
            },
        },
        path,
    )
    target = _temporal_model()

    report = load_transfer_checkpoint(
        target,
        path,
        expected_protocol="so101-state-residual-lewm-v5-temporal",
        expected_center=torch.zeros(6).numpy(),
        expected_scale=torch.full((6,), 100.0).numpy(),
    )

    assert report["step"] == 10000
    assert report["loaded_tensors"] == len(source.state_dict())
    assert all(
        torch.equal(target.state_dict()[key], value)
        for key, value in source.state_dict().items()
    )


def test_complete_transfer_checkpoint_rejects_normalization_mismatch(tmp_path) -> None:
    source = _temporal_model()
    path = tmp_path / "transfer.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "step": 10000,
            "sigreg_weight": 0.02,
            "run_config": {
                "contract": {
                    "protocol": "so101-state-residual-lewm-v5-temporal",
                    "normalization_center": [0.0] * 6,
                    "normalization_scale": [100.0] * 6,
                }
            },
        },
        path,
    )

    with pytest.raises(ValueError, match="scale differs"):
        load_transfer_checkpoint(
            _temporal_model(),
            path,
            expected_protocol="so101-state-residual-lewm-v5-temporal",
            expected_center=torch.zeros(6).numpy(),
            expected_scale=torch.ones(6).numpy(),
        )


def test_frozen_base_stays_in_eval_while_residual_branch_trains() -> None:
    model = _temporal_model().eval()

    set_training_mode(model, train_base=False)

    assert model.training
    assert model.temporal.training
    assert not model.base.training

    set_training_mode(model, train_base=True)
    assert model.base.training


def test_normalize_state_uses_all_six_shared_coordinates() -> None:
    state = torch.tensor([[[1.0, 3.0, 5.0, 7.0, 9.0, 11.0]]])
    center = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    scale = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    normalized = normalize_state(state, center, scale)

    torch.testing.assert_close(normalized, torch.ones_like(normalized))


def test_physical_state_action_normalization_uses_joint_scale() -> None:
    center, scale, frames = state_action_normalization(
        "physical-scale", 100.0, [], {}, set()
    )

    assert frames == 0
    assert center.tolist() == [0.0] * 6
    assert scale.tolist() == [100.0] * 6


def test_base_normalization_must_match_residual_training() -> None:
    report = {
        "action_normalization": "physical-scale",
        "action_center": [0.0] * 6,
        "action_scale": [100.0] * 6,
    }
    validate_base_normalization(
        report,
        "physical-scale",
        torch.zeros(6).numpy(),
        torch.full((6,), 100.0).numpy(),
    )

    try:
        validate_base_normalization(
            report,
            "global-train-zscore",
            torch.zeros(6).numpy(),
            torch.ones(6).numpy(),
        )
    except ValueError as error:
        assert "normalization differs" in str(error)
    else:
        raise AssertionError("normalization mismatch was accepted")


def test_temporal_model_is_zero_initialized_and_bounded() -> None:
    model = _temporal_model()
    pixels = torch.randn(2, 4, 3, 2, 2)
    actions = torch.randn(2, 3, 12)
    state = torch.randn(2, 4, 6)

    initial = model(pixels, actions, state)
    assert torch.equal(initial.prediction, initial.base_prediction)
    assert torch.equal(
        initial.state_prediction,
        actions.reshape(2, 3, 2, 6)[:, :, -1],
    )

    nn.init.constant_(model.latent_head.weight, 100.0)
    nn.init.constant_(model.state_head.weight, 100.0)
    bounded = model(pixels, actions, state)
    assert (bounded.prediction - bounded.base_prediction).abs().max() <= 0.2 + 1e-6
    assert (bounded.state_prediction - bounded.command_state).abs().max() <= 0.5 + 1e-6


def test_temporal_model_does_not_leak_later_inputs_to_earlier_steps() -> None:
    torch.manual_seed(23)
    model = _temporal_model().eval()
    nn.init.normal_(model.latent_head.weight, std=0.1)
    nn.init.normal_(model.state_head.weight, std=0.1)
    context = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 12)
    state = torch.randn(2, 3, 6)

    reference = model.predict_context(context, actions, state)
    changed_context = context.clone()
    changed_actions = actions.clone()
    changed_state = state.clone()
    changed_context[:, -1] += 10.0
    changed_actions[:, -1] -= 8.0
    changed_state[:, -1] += 6.0
    changed = model.predict_context(changed_context, changed_actions, changed_state)

    torch.testing.assert_close(reference[1][:, :-1], changed[1][:, :-1])
    torch.testing.assert_close(reference[3][:, :-1], changed[3][:, :-1])
    assert not torch.allclose(reference[1][:, -1], changed[1][:, -1])
    assert not torch.allclose(reference[3][:, -1], changed[3][:, -1])


def test_temporal_model_enforces_sequence_capacity() -> None:
    model = _temporal_model(max_steps=2)
    with torch.no_grad():
        try:
            model.predict_context(
                torch.randn(1, 3, 4),
                torch.randn(1, 3, 12),
                torch.randn(1, 3, 6),
            )
        except ValueError as error:
            assert "max_steps" in str(error)
        else:
            raise AssertionError("temporal model accepted a sequence beyond max_steps")


def test_default_temporal_residual_is_about_five_million_parameters() -> None:
    class FullSizeBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_encoder = nn.Linear(30, 192)

    model = TemporalStateResidualLeWM(FullSizeBase())
    branch_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("base.")
    )

    assert 4_000_000 <= branch_parameters <= 6_000_000
