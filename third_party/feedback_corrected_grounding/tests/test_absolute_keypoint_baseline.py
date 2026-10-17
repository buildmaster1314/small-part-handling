from __future__ import annotations

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import fcpg.absolute_keypoint_baseline as absolute
from fcpg.absolute_keypoint_baseline import (
    AbsoluteKeypointOutput,
    AnchorConditionedAbsoluteKeypointHead,
    AnchorConditionedAbsoluteKeypointViTAdapter,
    absolute_keypoint_contract,
    compute_absolute_keypoint_loss,
    load_absolute_keypoint_config,
    run_cpu_smoke,
    validate_absolute_keypoint_checkpoint_payload,
    validate_absolute_keypoint_config,
    validate_reference_training_protocol,
)
from fcpg.transformer_grounding import LeWMAnchorResidualCorrector
from scripts.train_absolute_keypoint_baseline import predict_from_batch


CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "config/absolute_keypoint/anchor_conditioned_shared_4task_seed0.json"
)


def test_anchor_conditioned_head_uses_direct_tanh_absolute_coordinates() -> None:
    head = AnchorConditionedAbsoluteKeypointHead(
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
    ).eval()
    logits = torch.tensor([torch.atanh(torch.tensor(0.5)), torch.atanh(torch.tensor(-0.5))])
    with torch.no_grad():
        head.absolute_head[-1].weight.zero_()
        head.absolute_head[-1].bias.copy_(logits)
    feature_map = torch.randn(2, 8, 8, 8)
    q_nominal = torch.tensor(
        [
            [[2.0, 3.0], [8.0, 9.0], [14.0, 15.0]],
            [[18.0, 4.0], [22.0, 11.0], [29.0, 20.0]],
        ]
    )
    output = head(
        feature_map,
        q_nominal,
        torch.randn(2, 3, 3),
        torch.randn(2, 4),
        image_size=(32, 24),
    )

    expected_norm = torch.tensor([0.75, 0.25]).expand(2, 3, 2)
    expected_px = expected_norm * torch.tensor([31.0, 23.0])
    torch.testing.assert_close(output.q_absolute_norm, expected_norm)
    torch.testing.assert_close(output.q_absolute_px, expected_px)
    torch.testing.assert_close(output.q_corrected_px, expected_px)
    assert not hasattr(output, "delta_px")
    assert not torch.equal(output.q_absolute_px, q_nominal)


def test_default_head_and_visual_projection_match_fcpg_trainable_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The FCPG checkpoint has 7,371,802 head parameters and 985,856 trainable
    # dense-vision projection parameters.  A tiny fake avoids constructing ViT.
    head = AnchorConditionedAbsoluteKeypointHead(
        visual_dim=256,
        obs_dim=4,
        num_keypoints=3,
        model_dim=256,
        local_grid_size=5,
        local_radius_px=24.0,
        local_layers=4,
        global_layers=2,
        heads=8,
        dropout=0.1,
    )
    assert sum(parameter.numel() for parameter in head.parameters()) == 7_371_802
    reference_head = LeWMAnchorResidualCorrector(
        visual_dim=256,
        obs_dim=4,
        num_keypoints=3,
        model_dim=256,
        local_grid_size=5,
        local_radius_px=24.0,
        local_layers=4,
        global_layers=2,
        heads=8,
        dropout=0.1,
        max_residual_px=20.0,
    )
    reference_shapes = {
        name.replace("delta_head", "coordinate_head"): tuple(value.shape)
        for name, value in reference_head.state_dict().items()
    }
    absolute_shapes = {
        name.replace("absolute_head", "coordinate_head"): tuple(value.shape)
        for name, value in head.state_dict().items()
    }
    assert absolute_shapes == reference_shapes

    class MatchedDenseVision(nn.Module):
        def __init__(self, *, selected_layers, **_: object) -> None:
            super().__init__()
            self.selected_layers = tuple(selected_layers)
            self.backbone = nn.Linear(1, 1, bias=False)
            self.backbone.requires_grad_(False)
            self.dense_projection_parameters = nn.Parameter(torch.zeros(985_856))

    monkeypatch.setattr(absolute, "ViTDenseEncoder", MatchedDenseVision)
    model = AnchorConditionedAbsoluteKeypointViTAdapter(
        obs_dim=4,
        num_keypoints=3,
        vision_dim=256,
        model_dim=256,
        vision_selected_layers=(11,),
        local_grid_size=5,
        local_radius_px=24.0,
        local_layers=4,
        global_layers=2,
        heads=8,
        dropout=0.1,
        target_trainable_parameters=8_357_658,
        allow_parameter_padding=False,
    )

    assert model.parameter_report() == {
        "total_parameters": 8_357_659,
        "trainable_parameters": 8_357_658,
        "frozen_parameters": 1,
        "functional_trainable_parameters": 8_357_658,
        "padding_trainable_parameters": 0,
        "target_trainable_parameters": 8_357_658,
        "exact_target_match": True,
    }
    assert model.sampling_report() == {
        "visual_feature_levels": 1,
        "sampled_visual_tokens_per_keypoint": 25,
        "sampled_visual_tokens_per_frame": 75,
        "local_transformer_sequence_length": 26,
        "keypoint_transformer_sequence_length": 3,
    }


def test_parameter_padding_is_opt_in_and_transparently_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyVision(nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()
            self.backbone = nn.Linear(1, 1, bias=False)
            self.backbone.requires_grad_(False)

    monkeypatch.setattr(absolute, "ViTDenseEncoder", TinyVision)
    kwargs = {
        "obs_dim": 4,
        "num_keypoints": 2,
        "vision_dim": 8,
        "model_dim": 16,
        "local_grid_size": 3,
        "local_radius_px": 4.0,
        "local_layers": 1,
        "global_layers": 1,
        "heads": 4,
        "dropout": 0.0,
    }
    unpadded = AnchorConditionedAbsoluteKeypointViTAdapter(**kwargs)
    functional = int(unpadded.parameter_report()["trainable_parameters"])
    with pytest.raises(ValueError, match="requires 7 padding"):
        AnchorConditionedAbsoluteKeypointViTAdapter(
            **kwargs,
            target_trainable_parameters=functional + 7,
            allow_parameter_padding=False,
        )
    padded = AnchorConditionedAbsoluteKeypointViTAdapter(
        **kwargs,
        target_trainable_parameters=functional + 7,
        allow_parameter_padding=True,
    )
    report = padded.parameter_report()
    assert report["padding_trainable_parameters"] == 7
    assert report["functional_trainable_parameters"] == functional
    assert report["exact_target_match"] is True


def test_loss_supervises_absolute_q_true_without_residual_target() -> None:
    prediction = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]])
    target = torch.tensor([[[2.0, 4.0], [7.0, 8.0]]])
    valid = torch.tensor([[True, False]])
    output = AbsoluteKeypointOutput(
        q_absolute_norm=prediction / 10.0,
        q_absolute_px=prediction,
        keypoint_latent=torch.zeros(1, 2, 4),
    )

    loss, metrics = compute_absolute_keypoint_loss(output, target, valid)

    expected = F.smooth_l1_loss(prediction[:, :1], target[:, :1], beta=1.0)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        metrics["absolute_error_px"], torch.sqrt(torch.tensor(5.0))
    )
    assert list(inspect.signature(compute_absolute_keypoint_loss).parameters) == [
        "output",
        "q_true_px",
        "valid",
    ]


def test_predict_batch_never_forwards_q_true() -> None:
    calls: list[tuple[torch.Tensor, ...]] = []

    class FakeModel:
        def forward_from_backbone_features(self, *args, image_size):
            calls.append((*args, torch.tensor(image_size)))
            return SimpleNamespace(q_absolute_px=torch.zeros(2, 3, 2))

    batch = {
        "vision_backbone_features": torch.zeros(2, 1, 4, 2, 2),
        "vision_prefix": torch.zeros(2, 4),
        "q_nominal": torch.ones(2, 3, 2),
        "keypoints_xyz_fk": torch.ones(2, 3, 3),
        "robot_obs": torch.ones(2, 4),
        "image_size": torch.tensor([[32, 24], [32, 24]]),
        "target_q_true": torch.full((2, 3, 2), 999.0),
        "target_valid": torch.ones(2, 3, dtype=torch.bool),
    }

    predict_from_batch(FakeModel(), batch)

    assert len(calls) == 1
    assert len(calls[0]) == 6
    assert all(not bool(torch.eq(value, 999.0).all()) for value in calls[0])


def test_config_and_checkpoint_schemas_fail_closed() -> None:
    config = load_absolute_keypoint_config(CONFIG_PATH)
    assert config["variant"] == "anchor_conditioned_absolute_keypoint"
    bad_config = copy.deepcopy(config)
    bad_config["unexpected"] = True
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        validate_absolute_keypoint_config(bad_config)

    model_config = copy.deepcopy(config["model"])
    model_config["target_trainable_parameters"] = 10
    report = {
        "total_parameters": 15,
        "trainable_parameters": 10,
        "frozen_parameters": 5,
        "functional_trainable_parameters": 10,
        "padding_trainable_parameters": 0,
        "target_trainable_parameters": 10,
        "exact_target_match": True,
    }
    grid_tokens = int(model_config["local_grid_size"]) ** 2
    payload = {
        "schema_version": 1,
        "checkpoint_type": "absolute_keypoint_baseline",
        "variant": "anchor_conditioned_absolute_keypoint",
        "contract": absolute_keypoint_contract(),
        "model_config": model_config,
        "model_state": {"weight": torch.zeros(1)},
        "parameter_report": report,
        "sampling_report": {
            "visual_feature_levels": 1,
            "sampled_visual_tokens_per_keypoint": grid_tokens,
            "sampled_visual_tokens_per_frame": grid_tokens * 3,
            "local_transformer_sequence_length": grid_tokens + 1,
            "keypoint_transformer_sequence_length": 3,
        },
        "training_provenance": {
            "config_path": "config.json",
            "config_sha256": "0" * 64,
            "reference_fcpg": {},
            "data_sources": [],
            "split": {},
            "budget": {},
            "q_true_usage": "training_supervision_and_offline_metrics_only",
        },
        "epoch": 1,
        "metrics": {},
    }
    validate_absolute_keypoint_checkpoint_payload(payload)
    bad_payload = copy.deepcopy(payload)
    bad_payload["contract"]["model_inputs"].append("q_true")
    with pytest.raises(ValueError, match="input contract"):
        validate_absolute_keypoint_checkpoint_payload(bad_payload)
    bad_totals = copy.deepcopy(payload)
    bad_totals["parameter_report"]["total_parameters"] = 16
    with pytest.raises(ValueError, match="totals are inconsistent"):
        validate_absolute_keypoint_checkpoint_payload(bad_totals)


def test_reference_training_protocol_requires_matched_data_split_and_budget() -> None:
    config = load_absolute_keypoint_config(CONFIG_PATH)
    sources = [
        CONFIG_PATH.parents[2] / source["path"]
        for source in config["data_sources"]
    ]
    report = {
        "data_sources": [str(path.resolve()) for path in sources],
        "training_protocol": {
            "obs_mode": "robot",
            "epochs": 50,
            "batch_size": 1024,
            "num_workers": 4,
            "balance_sources": True,
            "max_train_episodes_per_source": 0,
            "episode_selection_seed": 0,
            "val_fraction": 0.2,
            "split_unit": "task-episode",
            "split_seed": 0,
            "seed": 0,
            "lr": 0.0003,
            "weight_decay": 0.00001,
            "warmup_epochs": 3.0,
            "grad_clip": 1.0,
            "reliability_weight": 0.0,
            "reliability_corruption_prob": 0.0,
            "sigreg_weight": 0.0,
        },
    }
    validate_reference_training_protocol(report, config, sources)
    bad_report = copy.deepcopy(report)
    bad_report["training_protocol"]["epochs"] = 49
    with pytest.raises(ValueError, match="epochs"):
        validate_reference_training_protocol(bad_report, config, sources)


def test_action_chunk_loader_passes_absolute_adapter_as_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = SimpleNamespace(obs_dim=4, num_keypoints=3)
    policy = SimpleNamespace(obs_dim=4, num_keypoints=3)
    calls: list[object] = []
    monkeypatch.setattr(
        absolute,
        "load_absolute_keypoint_baseline",
        lambda *_args, **_kwargs: (baseline, {"variant": absolute.BASELINE_VARIANT}),
    )

    def fake_policy_loader(_path, *, device, fcpg_adapter_override):
        calls.append((device, fcpg_adapter_override))
        return policy, {"mode": "fcpg"}

    monkeypatch.setattr(
        "fcpg.action_chunk_policy.load_action_chunk_policy", fake_policy_loader
    )
    loaded, args, metadata = absolute.load_action_chunk_policy_with_absolute_keypoints(
        "policy.pt", "absolute.pt", device="cpu"
    )

    assert loaded is policy
    assert args["mode"] == "fcpg"
    assert metadata["variant"] == absolute.BASELINE_VARIANT
    assert calls == [("cpu", baseline)]


def test_cpu_smoke_is_finite_and_excludes_q_true_from_model_input() -> None:
    report = run_cpu_smoke(seed=3)
    assert report["device"] == "cpu"
    assert report["finite"] is True
    assert report["output_shape"] == [2, 3, 2]
    assert report["gradient_parameters"] > 0
    assert report["q_true_is_model_input"] is False
