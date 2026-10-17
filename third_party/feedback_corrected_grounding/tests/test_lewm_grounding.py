"""Focused tests for the residual-only LeWM grounding path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import fcpg.transformer_grounding as grounding
from scripts.evaluate_transformer_residual import evaluate_mode, validate_vision_cache
from scripts.train_transformer_residual import compute_losses, resolve_training_weights


def _corrector_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_map = torch.randn(2, 8, 8, 8)
    q_nominal = torch.tensor(
        [
            [[16.0, 16.0], [20.0, 24.0], [28.0, 12.0]],
            [[14.0, 18.0], [22.0, 20.0], [26.0, 14.0]],
        ]
    )
    keypoints = torch.randn(2, 3, 3)
    obs = torch.randn(2, 4)
    return feature_map, q_nominal, keypoints, obs


def test_lewm_corrector_is_residual_only_and_identity_initialized() -> None:
    torch.manual_seed(0)
    module = grounding.LeWMAnchorResidualCorrector(
        visual_dim=8,
        obs_dim=4,
        num_keypoints=3,
        model_dim=16,
        local_grid_size=3,
        local_radius_px=4.0,
        heads=4,
        dropout=0.0,
        max_residual_px=12.0,
    ).eval()
    feature_map, q_nominal, keypoints, obs = _corrector_inputs()

    output = module(feature_map, q_nominal, keypoints, obs, image_size=(32, 32))

    assert len(module.local_transformer.layers) == 4
    assert len(module.keypoint_transformer.layers) == 2
    assert all(not layer.attention.causal for layer in module.local_transformer.layers)
    assert all(not layer.attention.causal for layer in module.keypoint_transformer.layers)
    assert output.delta_px.shape == (2, 3, 2)
    assert output.keypoint_latent.shape == (2, 3, 16)
    assert not hasattr(output, "reliability")
    torch.testing.assert_close(output.delta_px, torch.zeros_like(output.delta_px))
    torch.testing.assert_close(output.q_corrected_px, q_nominal, rtol=0.0, atol=0.0)


def test_lewm_corrector_zero_initialized_head_receives_gradient() -> None:
    torch.manual_seed(1)
    module = grounding.LeWMAnchorResidualCorrector(
        visual_dim=8,
        obs_dim=4,
        num_keypoints=3,
        model_dim=16,
        local_grid_size=3,
        local_radius_px=4.0,
        local_layers=1,
        global_layers=1,
        heads=4,
        dropout=0.0,
    )
    feature_map, q_nominal, keypoints, obs = _corrector_inputs()
    output = module(feature_map, q_nominal, keypoints, obs, image_size=(32, 32))

    loss = F.smooth_l1_loss(output.delta_px, torch.ones_like(output.delta_px))
    loss.backward()

    gradient = module.delta_head[-1].weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_two_tip_corrector_reconstructs_center_and_preserves_anchor_order() -> None:
    torch.manual_seed(2)
    module = grounding.TwoTipStructuredLeWMAnchorCorrector(
        visual_dim=8,
        obs_dim=4,
        model_dim=16,
        local_grid_size=3,
        local_radius_px=4.0,
        local_layers=1,
        global_layers=1,
        heads=4,
        dropout=0.0,
        max_residual_px=12.0,
    ).eval()
    feature_map, q_nominal, keypoints, obs = _corrector_inputs()
    q_nominal = q_nominal.clone()
    q_nominal[:, 0] = q_nominal[:, 1:].mean(dim=1)

    output = module(feature_map, q_nominal, keypoints, obs, image_size=(32, 32))

    assert output.delta_px.shape == (2, 3, 2)
    assert output.keypoint_latent.shape == (2, 3, 16)
    torch.testing.assert_close(
        output.q_corrected_px[:, 0],
        output.q_corrected_px[:, 1:].mean(dim=1),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(output.q_corrected_px, q_nominal, rtol=0.0, atol=0.0)


def test_two_tip_corrector_trains_tip_residuals_through_derived_center() -> None:
    torch.manual_seed(3)
    module = grounding.TwoTipStructuredLeWMAnchorCorrector(
        visual_dim=8,
        obs_dim=4,
        model_dim=16,
        local_grid_size=3,
        local_radius_px=4.0,
        local_layers=1,
        global_layers=1,
        heads=4,
        dropout=0.0,
    )
    feature_map, q_nominal, keypoints, obs = _corrector_inputs()
    q_nominal = q_nominal.clone()
    q_nominal[:, 0] = q_nominal[:, 1:].mean(dim=1)
    output = module(feature_map, q_nominal, keypoints, obs, image_size=(32, 32))

    loss = F.smooth_l1_loss(output.delta_px, torch.ones_like(output.delta_px))
    loss.backward()

    gradient = module.tip_corrector.delta_head[-1].weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_lewm_adapter_preserves_vision_outputs(monkeypatch) -> None:
    class FakeVisionEncoder(nn.Module):
        def __init__(self, output_dim: int, **_: object) -> None:
            super().__init__()
            self.output_dim = output_dim

        def forward(self, rgb: torch.Tensor) -> SimpleNamespace:
            batch = rgb.shape[0]
            feature_map = torch.randn(batch, self.output_dim, 8, 8, device=rgb.device)
            return SimpleNamespace(
                feature_map=feature_map,
                global_feature=feature_map.mean(dim=(-2, -1)),
            )

    monkeypatch.setattr(grounding, "ViTDenseEncoder", FakeVisionEncoder)
    adapter = grounding.LeWMFCPGAdapter(
        obs_dim=4,
        num_keypoints=3,
        vision_dim=8,
        model_dim=16,
        pretrained=False,
        local_grid_size=3,
        local_radius_px=4.0,
        local_layers=1,
        global_layers=1,
        heads=4,
        dropout=0.0,
    ).eval()
    rgb = torch.randn(2, 3, 32, 32)
    _, q_nominal, keypoints, obs = _corrector_inputs()

    output = adapter(rgb, q_nominal, keypoints, obs)

    assert output.feature_map is not None and output.feature_map.shape == (2, 8, 8, 8)
    assert output.global_feature is not None and output.global_feature.shape == (2, 8)
    torch.testing.assert_close(output.q_corrected_px, q_nominal, rtol=0.0, atol=0.0)


def test_loader_dispatches_lewm_and_defaults_missing_arch_to_legacy(
    monkeypatch,
    tmp_path,
) -> None:
    class DummyAdapter(nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = kwargs

    class DummyLegacyAdapter(DummyAdapter):
        pass

    class DummyLeWMAdapter(DummyAdapter):
        pass

    monkeypatch.setattr(grounding, "FCPGViTAdapter", DummyLegacyAdapter)
    monkeypatch.setattr(grounding, "LeWMFCPGAdapter", DummyLeWMAdapter)
    common = {
        "obs_dim": 4,
        "num_keypoints": 3,
        "model_state": {},
    }
    legacy_path = tmp_path / "legacy.pt"
    lewm_path = tmp_path / "lewm.pt"
    invalid_path = tmp_path / "invalid.pt"
    torch.save({**common, "args": {}}, legacy_path)
    torch.save({**common, "args": {"corrector_arch": "lewm"}}, lewm_path)
    torch.save({**common, "args": {"corrector_arch": "invalid"}}, invalid_path)

    legacy, legacy_args = grounding.load_fcpg_vit_adapter(legacy_path)
    lewm, lewm_args = grounding.load_fcpg_vit_adapter(lewm_path)

    assert isinstance(legacy, DummyLegacyAdapter)
    assert isinstance(lewm, DummyLeWMAdapter)
    assert legacy_args == {}
    assert lewm_args["corrector_arch"] == "lewm"
    assert legacy.kwargs["pretrained"] is False
    assert lewm.kwargs["pretrained"] is False
    assert not legacy.training and not lewm.training
    with pytest.raises(ValueError, match="unknown corrector architecture"):
        grounding.load_fcpg_vit_adapter(invalid_path)


def test_lewm_training_defaults_are_residual_only_and_strict() -> None:
    reliability, corruption, sigreg = resolve_training_weights("lewm", None, None, None)
    assert reliability == 0.0
    assert corruption == 0.0
    assert sigreg == pytest.approx(0.02)

    assert resolve_training_weights("lewm", 0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)
    assert resolve_training_weights("lewm", 0.0, 0.0, 0.02) == (0.0, 0.0, 0.02)
    with pytest.raises(ValueError, match="reliability-weight 0"):
        resolve_training_weights("lewm", 0.1, 0.0, 0.09)
    with pytest.raises(ValueError, match="SIGReg weights 0, 0.02, or 0.09"):
        resolve_training_weights("lewm", 0.0, 0.0, 0.01)


def test_residual_evaluator_rejects_mismatched_vision_cache(tmp_path) -> None:
    data_path = tmp_path / "data.npz"
    data_path.touch()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(data_path.resolve()),
                "model_name": "vit_base_patch16_224.dino",
                "selected_layers": [11],
            }
        ),
        encoding="utf-8",
    )

    validate_vision_cache(
        cache,
        data_path,
        {
            "model_name": "vit_base_patch16_224.dino",
            "vision_selected_layers": (11,),
        },
    )
    with pytest.raises(ValueError, match="cache model"):
        validate_vision_cache(
            cache,
            data_path,
            {"model_name": "different", "vision_selected_layers": (11,)},
        )
    with pytest.raises(ValueError, match="cache layers"):
        validate_vision_cache(
            cache,
            data_path,
            {
                "model_name": "vit_base_patch16_224.dino",
                "vision_selected_layers": (8, 11),
            },
        )


def test_lewm_training_loss_does_not_require_reliability_fields() -> None:
    q_nominal = torch.zeros(2, 3, 2)
    q_true = torch.ones(2, 3, 2)
    output = SimpleNamespace(
        delta_px=torch.zeros_like(q_nominal),
        q_corrected_px=q_nominal.clone(),
        keypoint_latent=torch.randn(2, 3, 8),
    )
    batch = {
        "q_nominal": q_nominal,
        "q_true": q_true,
        "valid": torch.ones(2, 3, dtype=torch.bool),
    }

    loss, parts = compute_losses(
        output,
        batch,
        reliability_sigma_px=3.0,
        reliability_weight=0.0,
        sigreg=None,
        sigreg_weight=0.0,
        corrector_arch="lewm",
    )

    expected = F.smooth_l1_loss(output.delta_px, q_true - q_nominal)
    torch.testing.assert_close(loss, expected)
    assert set(parts) == {"residual_loss", "sigreg_loss", "corrected_error"}
    with pytest.raises(ValueError, match="cannot use reliability corruptions"):
        compute_losses(
            output,
            batch,
            reliability_sigma_px=3.0,
            reliability_weight=0.0,
            sigreg=None,
            sigreg_weight=0.0,
            corrupted=torch.ones(2, dtype=torch.bool),
            corrector_arch="lewm",
        )


def test_schema_v2_lewm_loader_validates_upstream_provenance(
    monkeypatch,
    tmp_path,
) -> None:
    class DummyLeWMAdapter(nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()

    monkeypatch.setattr(grounding, "LeWMFCPGAdapter", DummyLeWMAdapter)
    common = {
        "schema_version": 2,
        "corrector_arch": "lewm",
        "has_reliability": False,
        "obs_dim": 4,
        "num_keypoints": 3,
        "model_state": {},
        "args": {"corrector_arch": "lewm"},
    }
    valid_path = tmp_path / "valid_v2.pt"
    invalid_commit_path = tmp_path / "invalid_commit.pt"
    invalid_reliability_path = tmp_path / "invalid_reliability.pt"
    torch.save(
        {**common, "lewm_upstream_commit": grounding.LEWM_UPSTREAM_COMMIT},
        valid_path,
    )
    torch.save({**common, "lewm_upstream_commit": "wrong"}, invalid_commit_path)
    torch.save(
        {
            **common,
            "lewm_upstream_commit": grounding.LEWM_UPSTREAM_COMMIT,
            "has_reliability": True,
        },
        invalid_reliability_path,
    )

    model, args = grounding.load_fcpg_vit_adapter(valid_path)
    assert isinstance(model, DummyLeWMAdapter)
    assert args["schema_version"] == 2
    assert args["lewm_upstream_commit"] == grounding.LEWM_UPSTREAM_COMMIT
    with pytest.raises(ValueError, match="upstream commit"):
        grounding.load_fcpg_vit_adapter(invalid_commit_path)
    with pytest.raises(ValueError, match="cannot declare reliability"):
        grounding.load_fcpg_vit_adapter(invalid_reliability_path)


def test_residual_evaluator_accepts_output_without_reliability() -> None:
    class ResidualOnlyModel(nn.Module):
        def forward(
            self,
            _rgb: torch.Tensor,
            q_nominal: torch.Tensor,
            _keypoints_xyz: torch.Tensor,
            _obs: torch.Tensor,
        ) -> SimpleNamespace:
            return SimpleNamespace(q_corrected_px=q_nominal + 1.0)

    q_nominal = torch.zeros(2, 3, 2)
    batch = {
        "rgb": torch.zeros(2, 3, 8, 8),
        "q_nominal": q_nominal,
        "q_true": torch.full_like(q_nominal, 2.0),
        "keypoints_xyz": torch.zeros(2, 3, 3),
        "obs": torch.zeros(2, 4),
        "valid": torch.ones(2, 3, dtype=torch.bool),
    }

    metrics = evaluate_mode(
        ResidualOnlyModel(),
        [batch],
        torch.device("cpu"),
        mode="normal",
        seed=0,
        cutout_radius_px=2.0,
    )

    assert metrics["has_reliability"] is False
    assert metrics["reliability_mean"] is None
    assert metrics["ratio"] == pytest.approx(0.5)



def test_residual_evaluator_accepts_cached_backbone_features() -> None:
    class CachedResidualOnlyModel(nn.Module):
        def forward_from_backbone_features(
            self,
            feature_maps: torch.Tensor,
            prefix: torch.Tensor,
            q_nominal: torch.Tensor,
            _keypoints_xyz: torch.Tensor,
            _obs: torch.Tensor,
            *,
            image_size: tuple[int, int],
        ) -> SimpleNamespace:
            assert feature_maps.shape == (2, 1, 8, 4, 4)
            assert prefix.shape == (2, 8)
            assert image_size == (32, 32)
            return SimpleNamespace(q_corrected_px=q_nominal + 1.0)

    q_nominal = torch.zeros(2, 3, 2)
    batch = {
        "vision_backbone_features": torch.zeros(2, 1, 8, 4, 4),
        "vision_prefix": torch.zeros(2, 8),
        "image_size": torch.tensor([[32, 32], [32, 32]]),
        "q_nominal": q_nominal,
        "q_true": torch.full_like(q_nominal, 2.0),
        "keypoints_xyz": torch.zeros(2, 3, 3),
        "obs": torch.zeros(2, 4),
        "valid": torch.ones(2, 3, dtype=torch.bool),
    }

    metrics = evaluate_mode(
        CachedResidualOnlyModel(),
        [batch],
        torch.device("cpu"),
        mode="normal",
        seed=0,
        cutout_radius_px=2.0,
    )

    assert metrics["has_reliability"] is False
    assert metrics["ratio"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="supports only normal mode"):
        evaluate_mode(
            CachedResidualOnlyModel(),
            [batch],
            torch.device("cpu"),
            mode="zero-image",
            seed=0,
            cutout_radius_px=2.0,
        )
