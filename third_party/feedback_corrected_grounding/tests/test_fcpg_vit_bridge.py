from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fcpg.lewm_grounded_model import FCPGViTDenseEncoder
from fcpg.lewm_runtime import _initialize_model_from_checkpoint


class _DummyFCPGVision(nn.Module):
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, self.output_dim, kernel_size=1)

    def forward(self, pixels: torch.Tensor) -> SimpleNamespace:
        feature_map = self.projection(pixels)
        return SimpleNamespace(
            feature_map=feature_map,
            global_feature=feature_map.mean(dim=(-2, -1)),
        )


def test_fcpg_bridge_keeps_source_frozen_and_trains_only_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _DummyFCPGVision()
    corrector = SimpleNamespace(vision=source)
    monkeypatch.setattr(
        "fcpg.transformer_grounding.load_fcpg_vit_adapter",
        lambda *_args, **_kwargs: (corrector, {}),
    )

    encoder = FCPGViTDenseEncoder("dummy.pt", output_dim=6)
    encoder.train()
    assert encoder.training
    assert not encoder.fcpg_vision.training
    assert all(not parameter.requires_grad for parameter in encoder.fcpg_vision.parameters())

    output = encoder(torch.randn(2, 3, 8, 8))
    assert output.feature_map.shape == (2, 6, 8, 8)
    assert output.global_feature.shape == (2, 6)
    (output.feature_map.sum() + output.global_feature.sum()).backward()

    assert all(parameter.grad is None for parameter in encoder.fcpg_vision.parameters())
    trainable = [
        parameter
        for name, parameter in encoder.named_parameters()
        if not name.startswith("fcpg_vision.")
    ]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)


class _WarmStartToy(nn.Module):
    checkpoint_type = "warm-start-toy-v1"

    def __init__(self) -> None:
        super().__init__()
        self.frame_encoder = nn.Module()
        self.frame_encoder.vision_encoder = nn.Linear(2, 3)
        self.dynamics = nn.Linear(3, 2)


def _write_warm_start_checkpoint(
    path: Path,
    *,
    omit_nonvision: bool = False,
) -> _WarmStartToy:
    source = _WarmStartToy()
    with torch.no_grad():
        source.dynamics.weight.fill_(0.25)
        source.dynamics.bias.fill_(-0.5)
    state = {
        name: value
        for name, value in source.state_dict().items()
        if name.startswith("dynamics.")
    }
    if omit_nonvision:
        state.pop("dynamics.weight")
    torch.save(
        {
            "checkpoint_type": source.checkpoint_type,
            "step": 2000,
            "model_state": state,
        },
        path,
    )
    return source


def test_warm_start_copies_nonvision_and_allows_new_visual_parameters(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    source = _write_warm_start_checkpoint(checkpoint)
    target = _WarmStartToy()
    original_visual = target.frame_encoder.vision_encoder.weight.detach().clone()

    receipt = _initialize_model_from_checkpoint(target, checkpoint)

    assert receipt["source_step"] == 2000
    assert receipt["missing_visual_parameters"] == 2
    assert torch.equal(target.dynamics.weight, source.dynamics.weight)
    assert torch.equal(target.dynamics.bias, source.dynamics.bias)
    assert torch.equal(target.frame_encoder.vision_encoder.weight, original_visual)


def test_warm_start_rejects_missing_nonvision_parameters(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    _write_warm_start_checkpoint(checkpoint, omit_nonvision=True)

    with pytest.raises(ValueError, match="missing non-vision parameters"):
        _initialize_model_from_checkpoint(_WarmStartToy(), checkpoint)
