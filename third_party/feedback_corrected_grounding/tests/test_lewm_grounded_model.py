from __future__ import annotations

from dataclasses import dataclass
import sys
import types

import pytest
import torch
from torch import nn

from fcpg.lewm_grounded_model import GroundedFrameEncoder, GroundedLeWM, LeWMViTDenseEncoder
from third_party.lewm.module import ARPredictor, Embedder, MLP, SIGReg


@dataclass(frozen=True)
class _VisionOutput:
    feature_map: torch.Tensor
    global_feature: torch.Tensor


class _TinyVision(nn.Module):
    def __init__(self, visual_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, visual_dim, kernel_size=1)

    def forward(self, pixels: torch.Tensor) -> _VisionOutput:
        feature_map = self.projection(pixels)
        return _VisionOutput(
            feature_map=feature_map,
            global_feature=feature_map.mean(dim=(-1, -2)),
        )


def _model(
    *,
    target_mode: str = "online",
    alignment_weight: float = 0.0,
) -> GroundedLeWM:
    frame_encoder = GroundedFrameEncoder(
        vision_encoder=_TinyVision(visual_dim=8),
        visual_dim=8,
        obs_dim=4,
        num_keypoints=3,
        latent_dim=16,
        depth=1,
        heads=4,
        dropout=0.0,
    )
    return GroundedLeWM(
        frame_encoder=frame_encoder,
        action_dim=4,
        history_size=3,
        predictor_depth=1,
        predictor_heads=4,
        predictor_mlp_dim=32,
        predictor_dim_head=4,
        predictor_dropout=0.0,
        projector_hidden_dim=32,
        sigreg_knots=5,
        sigreg_num_proj=8,
        target_mode=target_mode,
        alignment_weight=alignment_weight,
    )


def _frame_encoder(
    anchor_visual_mode: str = "single_corrected",
) -> GroundedFrameEncoder:
    return GroundedFrameEncoder(
        vision_encoder=_TinyVision(visual_dim=8),
        visual_dim=8,
        obs_dim=4,
        num_keypoints=3,
        latent_dim=16,
        depth=1,
        heads=4,
        dropout=0.0,
        anchor_visual_mode=anchor_visual_mode,
    )


def _batch(batch_size: int = 4, frames: int = 4) -> dict[str, torch.Tensor]:
    q_nominal = torch.rand(batch_size, frames, 3, 2) * 15.0
    return {
        "pixels": torch.rand(batch_size, frames, 3, 16, 16),
        "proprio": torch.randn(batch_size, frames, 4),
        "q_nominal": q_nominal,
        "q_corrected": q_nominal + 0.5 * torch.randn_like(q_nominal),
        "keypoints_xyz": torch.randn(batch_size, frames, 3, 3),
        "valid": torch.ones(batch_size, frames, 3, dtype=torch.bool),
        "action": torch.randn(batch_size, frames - 1, 4),
        "source_index": torch.tensor([0, 0, 1, 1]),
    }


def _feature_inputs(sampled_visual: torch.Tensor) -> dict[str, object]:
    batch_size = sampled_visual.shape[0]
    return {
        "sampled_visual": sampled_visual,
        "global_visual": torch.randn(batch_size, 8),
        "robot_obs": torch.randn(batch_size, 4),
        "q_nominal_px": torch.randn(batch_size, 3, 2),
        "q_grounded_px": torch.randn(batch_size, 3, 2),
        "keypoints_xyz": torch.randn(batch_size, 3, 3),
        "valid": torch.ones(batch_size, 3, dtype=torch.bool),
        "image_size": (16, 16),
    }


def test_grounded_lewm_reuses_upstream_components() -> None:
    model = _model()
    assert isinstance(model.predictor, ARPredictor)
    assert isinstance(model.action_encoder, Embedder)
    assert isinstance(model.projector, MLP)
    assert isinstance(model.pred_proj, MLP)
    assert isinstance(model.sigreg, SIGReg)


def test_upstream_action_embedder_width_and_persisted_normalization() -> None:
    model = _model().eval()
    assert model.action_encoder.patch_embed.out_channels == 10
    model.set_input_normalization(
        proprio_mean=[1.0, 2.0, 3.0, 4.0],
        proprio_std=[2.0, 2.0, 2.0, 2.0],
        action_mean=[1.0, 1.0, 1.0, 1.0],
        action_std=[2.0, 2.0, 2.0, 2.0],
    )
    seen: dict[str, torch.Tensor] = {}

    def capture_action(_module, inputs) -> None:
        seen["action"] = inputs[0].detach().clone()

    handle = model.action_encoder.register_forward_pre_hook(capture_action)
    action = torch.tensor([3.0, 5.0, 7.0, 9.0]).view(1, 1, 4).expand(2, 3, 4)
    with torch.no_grad():
        model.predict(torch.zeros(2, 3, 16), action)
    handle.remove()

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4).expand_as(action)
    torch.testing.assert_close(seen["action"], expected)
    assert "action_mean" in model.state_dict()
    assert "proprio_std" in model.state_dict()


def test_faithful_teacher_forcing_objective_has_expected_shapes() -> None:
    torch.manual_seed(3)
    model = _model().train()
    output = model(_batch(), sigreg_weight=0.02, per_source_sigreg=True)
    assert output.frame_latent.shape == (4, 4, 16)
    assert output.embedding.shape == (4, 4, 16)
    assert output.predicted_embedding.shape == (4, 3, 16)
    assert output.target_embedding.shape == (4, 3, 16)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.copy_loss)
    output.loss.backward()
    assert model.frame_encoder.summary_token.grad is not None


def test_default_anchor_visual_mode_preserves_legacy_checkpoint_contract() -> None:
    default = _frame_encoder()
    restored = _frame_encoder("single_corrected")

    assert default.anchor_visual_mode == "single_corrected"
    assert default.anchor_visual_projection[0].normalized_shape == (8,)
    assert default.anchor_visual_projection[1].weight.shape == (16, 8)
    incompatible = restored.load_state_dict(default.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    corrected = torch.randn(2, 3, 8)
    seen: list[torch.Tensor] = []

    def capture_projection_input(_module, inputs) -> None:
        seen.append(inputs[0].detach().clone())

    handle = default.anchor_visual_projection.register_forward_pre_hook(
        capture_projection_input
    )
    default.forward_from_features(**_feature_inputs(corrected))
    handle.remove()
    torch.testing.assert_close(seen[0], corrected)

    with pytest.raises(ValueError, match="anchor_visual_mode must be one of"):
        _frame_encoder("unsupported")


def test_feature_token_sequence_exposes_slots_and_preserves_legacy_summary() -> None:
    torch.manual_seed(43)
    encoder = _frame_encoder().eval()
    inputs = _feature_inputs(torch.randn(2, 3, 8))
    state_keys_before = tuple(encoder.state_dict())

    with torch.no_grad():
        token_sequence = encoder.forward_token_sequence_from_features(**inputs)
        legacy_summary = encoder.forward_from_features(**inputs)

    assert token_sequence.shape == (2, 3 + encoder.num_keypoints, encoder.latent_dim)
    assert torch.equal(legacy_summary, token_sequence[:, 0])
    assert tuple(encoder.state_dict()) == state_keys_before

    restored = _frame_encoder().eval()
    incompatible = restored.load_state_dict(encoder.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_pixel_token_sequence_preserves_legacy_forward_exactly() -> None:
    torch.manual_seed(47)
    encoder = _frame_encoder().eval()
    batch = _batch(batch_size=2)
    inputs = {
        "pixels": batch["pixels"][:, 0],
        "robot_obs": batch["proprio"][:, 0],
        "q_nominal_px": batch["q_nominal"][:, 0],
        "q_grounded_px": batch["q_corrected"][:, 0],
        "keypoints_xyz": batch["keypoints_xyz"][:, 0],
        "valid": batch["valid"][:, 0],
    }

    with torch.no_grad():
        token_sequence = encoder.forward_token_sequence(**inputs)
        legacy_summary = encoder(**inputs)

    assert token_sequence.shape == (2, 3 + encoder.num_keypoints, encoder.latent_dim)
    assert torch.equal(legacy_summary, token_sequence[:, 0])


@pytest.mark.parametrize(
    "anchor_visual_mode",
    [
        "single_corrected",
        "triplet_nominal",
        "triplet_corrected",
        "paired_visual",
        "paired_innovation",
    ],
)
def test_anchor_visual_modes_produce_frame_latent_shape(
    anchor_visual_mode: str,
) -> None:
    encoder = _frame_encoder(anchor_visual_mode)
    batch = _batch(batch_size=2)
    frame_latent = encoder(
        batch["pixels"][:, 0],
        batch["proprio"][:, 0],
        batch["q_nominal"][:, 0],
        batch["q_corrected"][:, 0],
        batch["keypoints_xyz"][:, 0],
        batch["valid"][:, 0],
    )
    assert frame_latent.shape == (2, 16)


def test_triplet_anchor_visual_modes_have_identical_parameter_counts() -> None:
    triplet_modes = (
        "triplet_nominal",
        "triplet_corrected",
        "paired_visual",
        "paired_innovation",
    )
    parameter_counts = {
        mode: sum(parameter.numel() for parameter in _frame_encoder(mode).parameters())
        for mode in triplet_modes
    }

    assert len(set(parameter_counts.values())) == 1
    single_count = sum(
        parameter.numel() for parameter in _frame_encoder().parameters()
    )
    assert next(iter(parameter_counts.values())) > single_count


@pytest.mark.parametrize(
    ("anchor_visual_mode", "expected_kind"),
    [
        ("triplet_nominal", "nominal_triplet"),
        ("triplet_corrected", "corrected_triplet"),
        ("paired_visual", "paired_visual"),
        ("paired_innovation", "paired_innovation"),
    ],
)
def test_triplet_anchor_visual_combination_semantics(
    anchor_visual_mode: str,
    expected_kind: str,
) -> None:
    encoder = _frame_encoder(anchor_visual_mode)
    nominal = torch.randn(2, 3, 8)
    corrected = torch.randn(2, 3, 8)
    seen: list[torch.Tensor] = []

    def capture_projection_input(_module, inputs) -> None:
        seen.append(inputs[0].detach().clone())

    handle = encoder.anchor_visual_projection.register_forward_pre_hook(
        capture_projection_input
    )
    inputs = _feature_inputs(corrected)
    if anchor_visual_mode in {"triplet_nominal", "paired_visual", "paired_innovation"}:
        inputs["sampled_nominal_visual"] = nominal
    encoder.forward_from_features(**inputs)
    handle.remove()

    if expected_kind == "nominal_triplet":
        expected = torch.cat([nominal, nominal, nominal], dim=-1)
    elif expected_kind == "corrected_triplet":
        expected = torch.cat([corrected, corrected, corrected], dim=-1)
    elif expected_kind == "paired_visual":
        expected = torch.cat([nominal, corrected, corrected], dim=-1)
    else:
        expected = torch.cat([nominal, corrected, corrected - nominal], dim=-1)
    torch.testing.assert_close(seen[0], expected)


@pytest.mark.parametrize(
    "anchor_visual_mode",
    ["triplet_nominal", "paired_visual", "paired_innovation"],
)
def test_paired_anchor_visual_modes_require_nominal_features(
    anchor_visual_mode: str,
) -> None:
    encoder = _frame_encoder(anchor_visual_mode)
    with pytest.raises(ValueError, match="requires sampled_nominal_visual"):
        encoder.forward_from_features(**_feature_inputs(torch.randn(2, 3, 8)))


@pytest.mark.parametrize(
    "anchor_visual_mode",
    ["triplet_nominal", "triplet_corrected", "paired_visual", "paired_innovation"],
)
def test_triplet_anchor_visual_inputs_receive_gradients(
    anchor_visual_mode: str,
) -> None:
    torch.manual_seed(41)
    encoder = _frame_encoder(anchor_visual_mode).train()
    corrected = torch.randn(2, 3, 8, requires_grad=True)
    nominal = (
        torch.randn(2, 3, 8, requires_grad=True)
        if anchor_visual_mode in {"triplet_nominal", "paired_visual", "paired_innovation"}
        else None
    )
    inputs = _feature_inputs(corrected)
    if nominal is not None:
        inputs["sampled_nominal_visual"] = nominal
    output = encoder.forward_from_features(**inputs)
    weights = torch.linspace(0.5, 1.5, output.shape[-1])
    (output * weights).sum().backward()

    if anchor_visual_mode == "triplet_nominal":
        assert corrected.grad is None
    else:
        assert corrected.grad is not None
        assert corrected.grad.abs().sum().item() > 0.0
    if nominal is not None:
        assert nominal.grad is not None
        assert nominal.grad.abs().sum().item() > 0.0
    projection_weight = encoder.anchor_visual_projection[1].weight
    assert projection_weight.grad is not None
    assert projection_weight.grad.abs().sum().item() > 0.0


def test_frozen_oracle_target_is_shared_and_not_backpropagated() -> None:
    torch.manual_seed(31)
    model = _model(target_mode="frozen_oracle", alignment_weight=0.5).train()
    batch = _batch()
    oracle_target = torch.randn(4, 4, 16, requires_grad=True)
    batch["oracle_target"] = oracle_target

    output = model(batch, sigreg_weight=0.02)

    torch.testing.assert_close(output.state_target_embedding, oracle_target.detach())
    torch.testing.assert_close(output.target_embedding, oracle_target[:, 1:].detach())
    assert output.alignment_loss.item() > 0.0
    output.loss.backward()
    assert oracle_target.grad is None
    assert model.frame_encoder.summary_token.grad is not None

    without_target = _batch()
    with pytest.raises(KeyError, match="oracle_target"):
        model(without_target)


def test_recursive_rollout_and_goal_cost() -> None:
    torch.manual_seed(4)
    model = _model().eval()
    with torch.no_grad():
        _, embedding = model.encode_frames(_batch(batch_size=2))
        future = torch.randn(2, 5, 3, 4)
        prediction = model.rollout_embeddings(
            history_embedding=embedding[:, :3],
            history_action=torch.randn(2, 2, 4),
            future_action=future,
        )
        cost = model.goal_cost(prediction, embedding[:, -1])
    assert prediction.shape == (2, 5, 3, 16)
    assert cost.shape == (2, 5)
    assert bool(torch.isfinite(cost).all())


def test_faithful_objective_rejects_wrong_window_length() -> None:
    model = _model().eval()
    batch = _batch(frames=5)
    batch["source_index"] = torch.zeros(4, dtype=torch.long)
    try:
        model(batch)
    except ValueError as error:
        assert "H+1=4" in str(error)
    else:
        raise AssertionError("expected a strict H+1 window validation error")



def test_exact_upstream_vit_is_exposed_as_dense_patch_map(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: torch.Tensor | None = None
            self.tokens: torch.Tensor | None = None

        def forward(self, pixels: torch.Tensor, **kwargs: object) -> object:
            calls["forward_kwargs"] = kwargs
            self.seen = pixels.detach().clone()
            token_count = 1 + 16 * 16
            self.tokens = torch.arange(
                pixels.shape[0] * token_count * 192,
                dtype=pixels.dtype,
                device=pixels.device,
            ).reshape(pixels.shape[0], token_count, 192)
            return types.SimpleNamespace(last_hidden_state=self.tokens)

    backbone = FakeBackbone()

    def fake_vit_hf(**kwargs: object) -> nn.Module:
        calls["factory_kwargs"] = kwargs
        return backbone

    stable = types.ModuleType("stable_pretraining")
    backbone_module = types.ModuleType("stable_pretraining.backbone")
    utils_module = types.ModuleType("stable_pretraining.backbone.utils")
    utils_module.vit_hf = fake_vit_hf
    stable.backbone = backbone_module
    backbone_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "stable_pretraining", stable)
    monkeypatch.setitem(sys.modules, "stable_pretraining.backbone", backbone_module)
    monkeypatch.setitem(sys.modules, "stable_pretraining.backbone.utils", utils_module)

    encoder = LeWMViTDenseEncoder()
    output = encoder(torch.zeros(2, 3, 224, 224))

    assert calls["factory_kwargs"] == {
        "size": "tiny",
        "patch_size": 14,
        "image_size": 224,
        "pretrained": False,
        "use_mask_token": False,
    }
    assert calls["forward_kwargs"] == {"interpolate_pos_encoding": True}
    assert output.feature_map.shape == (2, 192, 16, 16)
    assert output.global_feature.shape == (2, 192)
    assert backbone.tokens is not None
    torch.testing.assert_close(output.global_feature, backbone.tokens[:, 0])
    torch.testing.assert_close(output.feature_map[:, :, 0, 0], backbone.tokens[:, 1])
    assert backbone.seen is not None
    expected = -encoder.image_mean / encoder.image_std
    torch.testing.assert_close(backbone.seen[:, :, :1, :1], expected.expand(2, -1, -1, -1))

    with pytest.raises(ValueError, match="unknown ViT size"):
        LeWMViTDenseEncoder(size="invalid")
