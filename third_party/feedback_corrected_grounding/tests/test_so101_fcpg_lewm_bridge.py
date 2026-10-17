from types import SimpleNamespace

import torch
from torch import nn

from fcpg.so101_fcpg_lewm_bridge import (
    SO101FCPGAnchorBridge,
    SO101FCPGConditionedLeWM,
    SO101DualCameraAnchorFusion,
    SO101DualCameraFCPGConditionedLeWM,
    build_so101_fcpg_corrector,
    infer_so101_fcpg_corrector_variant,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from fcpg.transformer_grounding import LeWMAnchorResidualCorrector


def bridge_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    batch, frames, anchors, latent = 2, 3, 3, 192
    nominal = torch.rand(batch, frames, anchors, 2) * 223.0
    return {
        "context_embedding": torch.randn(batch, frames, latent),
        "patch_tokens": torch.randn(batch, frames, 16 * 16, latent),
        "q_nominal_px": nominal,
        "q_corrected_px": nominal + torch.randn_like(nominal) * 4.0,
        "anchors_xyz": torch.randn(batch, frames, anchors, 3) * 0.1,
        "valid": torch.ones(batch, frames, anchors, dtype=torch.bool),
    }


def test_zero_gate_is_exactly_checkpoint_equivalent() -> None:
    bridge = SO101FCPGAnchorBridge(dropout=0.0).eval()
    inputs = bridge_inputs()

    output = bridge(**inputs, image_size=(224, 224))

    assert torch.equal(output.context_embedding, inputs["context_embedding"])
    assert output.anchor_tokens.shape == (2, 3, 3, 192)
    assert output.sampled_corrected.shape == (2, 3, 3, 192)


def test_open_gate_responds_to_fcpg_correction() -> None:
    bridge = SO101FCPGAnchorBridge(dropout=0.0).eval()
    bridge.gate.data.fill_(0.5)
    inputs = bridge_inputs()
    nominal = bridge(
        **{**inputs, "q_corrected_px": inputs["q_nominal_px"]},
        image_size=(224, 224),
    )
    corrected = bridge(**inputs, image_size=(224, 224))

    assert not torch.allclose(
        nominal.context_embedding, corrected.context_embedding
    )
    assert not torch.allclose(nominal.sampled_corrected, corrected.sampled_corrected)


def test_bridge_rejects_non_square_patch_grid() -> None:
    bridge = SO101FCPGAnchorBridge(dropout=0.0)
    inputs = bridge_inputs()
    inputs["patch_tokens"] = inputs["patch_tokens"][:, :, :-1]

    try:
        bridge(**inputs, image_size=(224, 224))
    except ValueError as error:
        assert "square grid" in str(error)
    else:
        raise AssertionError("bridge accepted a non-square patch grid")


def test_dual_camera_fusion_is_initially_primary_view_exact() -> None:
    torch.manual_seed(9)
    fusion = SO101DualCameraAnchorFusion(
        latent_dim=8,
        num_views=2,
        num_anchors=3,
        primary_view=0,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
    ).eval()
    context = torch.randn(2, 3, 2, 8)
    anchors = torch.randn(2, 3, 2, 3, 8)
    valid = torch.ones(2, 3, 2, 3, dtype=torch.bool)

    output = fusion(context, anchors, valid)

    assert torch.equal(output.context_embedding, context[:, :, 0])
    assert output.fused_tokens.shape == (2, 3, 8, 8)


def test_open_dual_camera_fusion_uses_secondary_view() -> None:
    torch.manual_seed(11)
    fusion = SO101DualCameraAnchorFusion(
        latent_dim=8,
        num_views=2,
        num_anchors=3,
        primary_view=0,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
    ).eval()
    fusion.gate.data.fill_(0.5)
    context = torch.randn(1, 2, 2, 8)
    anchors = torch.randn(1, 2, 2, 3, 8)
    valid = torch.ones(1, 2, 2, 3, dtype=torch.bool)
    baseline = fusion(context, anchors, valid)
    changed_context = context.clone()
    changed_context[:, :, 1, 0] += 2.0

    changed = fusion(changed_context, anchors, valid)

    assert not torch.allclose(
        baseline.context_embedding,
        changed.context_embedding,
    )


class TinyVisionEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, latent_dim, kernel_size=2, stride=2)
        self.calls = 0

    def forward(
        self,
        pixels: torch.Tensor,
        *,
        interpolate_pos_encoding: bool,
    ) -> SimpleNamespace:
        assert interpolate_pos_encoding
        self.calls += 1
        patches = self.projection(pixels).flatten(2).transpose(1, 2)
        cls = patches.mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patches), dim=1))


class TinyJEPA(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = TinyVisionEncoder(latent_dim)
        self.projector = nn.Identity()
        self.action_encoder = nn.Linear(action_dim, latent_dim, bias=False)

    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pixels = info["pixels"]
        batch, frames = pixels.shape[:2]
        output = self.encoder(
            pixels.flatten(0, 1), interpolate_pos_encoding=True
        ).last_hidden_state
        info["emb"] = self.projector(output[:, 0]).reshape(batch, frames, -1)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(
        self,
        embedding: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return embedding + action_embedding


def conditioned_fixture() -> tuple[
    StateResidualLeWM,
    SO101FCPGConditionedLeWM,
    dict[str, torch.Tensor],
]:
    torch.manual_seed(17)
    latent_dim, state_dim, action_dim = 8, 6, 30
    base = TinyJEPA(latent_dim, action_dim)
    model = StateResidualLeWM(
        base,
        state_dim=state_dim,
        latent_dim=latent_dim,
        hidden_dim=16,
    ).eval()
    corrector = LeWMAnchorResidualCorrector(
        visual_dim=latent_dim,
        obs_dim=state_dim,
        num_keypoints=3,
        model_dim=latent_dim,
        local_grid_size=3,
        local_radius_px=2.0,
        local_layers=1,
        global_layers=1,
        heads=2,
        dropout=0.0,
        max_residual_px=4.0,
    )
    bridge = SO101FCPGAnchorBridge(
        latent_dim=latent_dim,
        num_anchors=3,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
    )
    conditioned = SO101FCPGConditionedLeWM(
        model,
        corrector=corrector,
        bridge=bridge,
    ).eval()
    batch, frames, size = 2, 3, 8
    q_nominal = torch.rand(batch, frames, 3, 2) * float(size - 1)
    inputs = {
        "pixels": torch.randn(batch, frames, 3, size, size),
        "actions": torch.randn(batch, frames, action_dim),
        "normalized_state": torch.randn(batch, frames, state_dim),
        "q_nominal_px": q_nominal,
        "anchors_xyz": torch.randn(batch, frames, 3, 3) * 0.1,
        "valid": torch.ones(batch, frames, 3, dtype=torch.bool),
    }
    return model, conditioned, inputs


def test_corrector_variant_inference_preserves_old_checkpoints() -> None:
    assert infer_so101_fcpg_corrector_variant({"corrector": {"delta_head.0": 1}}) == (
        "independent3"
    )
    assert infer_so101_fcpg_corrector_variant(
        {"corrector": {"tip_corrector.delta_head.0": 1}}
    ) == "two_tip_structured"
    assert infer_so101_fcpg_corrector_variant(
        {"corrector_variant": "two_tip_structured", "corrector": {}}
    ) == "two_tip_structured"


def test_structured_corrector_factory_keeps_public_three_anchor_interface() -> None:
    model, _, _ = conditioned_fixture()
    corrector = build_so101_fcpg_corrector(
        model,
        camera_context_dim=5,
        variant="two_tip_structured",
    )

    assert corrector.num_keypoints == 3
    assert corrector.tip_corrector.num_keypoints == 2
    assert corrector.obs_dim == model.state_dim + 5


def test_corrector_payload_factory_strict_loads_both_variants() -> None:
    model, _, _ = conditioned_fixture()
    for variant in ("independent3", "two_tip_structured"):
        source = build_so101_fcpg_corrector(
            model,
            camera_context_dim=5,
            variant=variant,
        )
        payload = {"corrector": source.state_dict(), "corrector_variant": variant}
        restored = build_so101_fcpg_corrector(
            model,
            camera_context_dim=5,
            variant=infer_so101_fcpg_corrector_variant(payload),
        )

        restored.load_state_dict(payload["corrector"], strict=True)


def test_combined_fcpg_lewm_is_initially_checkpoint_equivalent() -> None:
    model, conditioned, inputs = conditioned_fixture()
    model.base.encoder.calls = 0
    encoded = model.base.encode(
        {"pixels": inputs["pixels"], "action": inputs["actions"]}
    )
    expected = model.predict_context(
        encoded["emb"],
        inputs["actions"],
        inputs["normalized_state"],
        action_embedding=encoded["act_emb"],
    )
    model.base.encoder.calls = 0

    actual = conditioned.predict_context(**inputs)

    assert model.base.encoder.calls == 1
    assert torch.equal(actual.fcpg.q_corrected_px, inputs["q_nominal_px"].flatten(0, 1))
    assert torch.equal(actual.bridge.context_embedding, encoded["emb"])
    for expected_tensor, actual_tensor in zip(
        expected,
        (
            actual.base_prediction,
            actual.prediction,
            actual.command_state,
            actual.state_prediction,
        ),
    ):
        assert torch.equal(expected_tensor, actual_tensor)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_combined_fcpg_lewm_uses_trained_correction_path() -> None:
    _, conditioned, inputs = conditioned_fixture()
    baseline = conditioned.predict_context(**inputs)
    conditioned.corrector.delta_head[-1].bias.data.fill_(0.5)
    conditioned.bridge.gate.data.fill_(0.5)

    corrected = conditioned.predict_context(**inputs)

    assert not torch.allclose(
        corrected.fcpg.q_corrected_px,
        inputs["q_nominal_px"].flatten(0, 1),
    )
    assert not torch.allclose(
        corrected.bridge.context_embedding,
        baseline.bridge.context_embedding,
    )
    assert not torch.allclose(
        corrected.base_prediction,
        baseline.base_prediction,
    )


def test_combined_fcpg_lewm_accepts_explicit_camera_context() -> None:
    model, _, inputs = conditioned_fixture()
    context_dim = 5
    corrector = LeWMAnchorResidualCorrector(
        visual_dim=model.latent_dim,
        obs_dim=model.state_dim + context_dim,
        num_keypoints=3,
        model_dim=model.latent_dim,
        local_grid_size=3,
        local_radius_px=2.0,
        local_layers=1,
        global_layers=1,
        heads=2,
        dropout=0.0,
        max_residual_px=4.0,
    )
    bridge = SO101FCPGAnchorBridge(
        latent_dim=model.latent_dim,
        num_anchors=3,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
    )
    conditioned = SO101FCPGConditionedLeWM(
        model,
        corrector=corrector,
        bridge=bridge,
        camera_context_dim=context_dim,
    ).eval()
    inputs["camera_context"] = torch.randn(
        inputs["pixels"].shape[0],
        inputs["pixels"].shape[1],
        context_dim,
    )

    output = conditioned.predict_context(**inputs)

    assert output.fcpg.q_corrected_px.shape == (6, 3, 2)


def test_dual_camera_fcpg_lewm_is_initially_primary_view_equivalent() -> None:
    model, _, single_inputs = conditioned_fixture()
    context_dim = 5
    corrector = LeWMAnchorResidualCorrector(
        visual_dim=model.latent_dim,
        obs_dim=model.state_dim + context_dim,
        num_keypoints=3,
        model_dim=model.latent_dim,
        local_grid_size=3,
        local_radius_px=2.0,
        local_layers=1,
        global_layers=1,
        heads=2,
        dropout=0.0,
        max_residual_px=4.0,
    )
    bridge = SO101FCPGAnchorBridge(
        latent_dim=model.latent_dim,
        num_anchors=3,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
    )
    single = SO101FCPGConditionedLeWM(
        model,
        corrector=corrector,
        bridge=bridge,
        camera_context_dim=context_dim,
    ).eval()
    fusion = SO101DualCameraAnchorFusion(
        latent_dim=model.latent_dim,
        num_views=2,
        num_anchors=3,
        primary_view=0,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
    )
    dual = SO101DualCameraFCPGConditionedLeWM(
        single,
        fusion=fusion,
        num_views=2,
    ).eval()
    pixels = torch.stack(
        (single_inputs["pixels"], single_inputs["pixels"] + 0.25), dim=2
    )
    q_nominal = torch.stack(
        (single_inputs["q_nominal_px"], single_inputs["q_nominal_px"] + 0.5),
        dim=2,
    )
    anchors_xyz = torch.stack(
        (single_inputs["anchors_xyz"], single_inputs["anchors_xyz"]), dim=2
    )
    valid = torch.stack((single_inputs["valid"], single_inputs["valid"]), dim=2)
    camera_context = torch.randn(
        pixels.shape[0], pixels.shape[1], pixels.shape[2], context_dim
    )
    encoded = model.base.encode(
        {
            "pixels": single_inputs["pixels"],
            "action": single_inputs["actions"],
        }
    )
    expected = model.predict_context(
        encoded["emb"],
        single_inputs["actions"],
        single_inputs["normalized_state"],
        action_embedding=encoded["act_emb"],
    )
    model.base.encoder.calls = 0

    output = dual.predict_context(
        pixels,
        single_inputs["actions"],
        single_inputs["normalized_state"],
        q_nominal,
        anchors_xyz,
        valid,
        camera_context,
    )

    assert model.base.encoder.calls == 1
    assert output.q_corrected_px.shape == (2, 3, 2, 3, 2)
    assert output.per_view_anchor_tokens.shape == (2, 3, 2, 3, 8)
    for expected_tensor, actual_tensor in zip(
        expected,
        (
            output.base_prediction,
            output.prediction,
            output.command_state,
            output.state_prediction,
        ),
    ):
        assert torch.equal(expected_tensor, actual_tensor)
