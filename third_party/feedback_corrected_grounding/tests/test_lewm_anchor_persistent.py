from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import fcpg.lewm_runtime as runtime
from fcpg.lewm_anchor_persistent import (
    ANCHOR_PERSISTENT_CHECKPOINT_TYPE,
    AnchorInnovationScorer,
    AnchorPersistentLeWM,
    anchor_innovation_features,
)
from fcpg.lewm_anchor_evaluation import (
    evaluate_anchor_persistent,
    same_task_shuffled_actions,
)
from fcpg.lewm_grounded_model import GroundedFrameEncoder, GroundedLeWM
from third_party.lewm.module import ARPredictor, Embedder, MLP, SIGReg


@dataclass(frozen=True)
class _VisionOutput:
    feature_map: torch.Tensor
    global_feature: torch.Tensor


class _TinyVision(nn.Module):
    output_dim = 8

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, self.output_dim, kernel_size=1)

    def forward(self, pixels: torch.Tensor) -> _VisionOutput:
        feature_map = self.projection(pixels)
        return _VisionOutput(
            feature_map=feature_map,
            global_feature=feature_map.mean(dim=(-1, -2)),
        )


def _model() -> AnchorPersistentLeWM:
    frame_encoder = GroundedFrameEncoder(
        vision_encoder=_TinyVision(),
        visual_dim=8,
        obs_dim=4,
        num_keypoints=3,
        latent_dim=16,
        depth=1,
        heads=4,
        dropout=0.0,
    )
    return AnchorPersistentLeWM(
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
    )


def _batch(batch_size: int = 4) -> dict[str, Any]:
    frames = 4
    q_nominal = torch.rand(batch_size, frames, 3, 2) * 15.0
    batch = {
        "pixels": torch.rand(batch_size, frames, 3, 16, 16),
        "proprio": torch.randn(batch_size, frames, 4),
        "q_nominal": q_nominal,
        "q_corrected": q_nominal + 0.5 * torch.randn_like(q_nominal),
        "keypoints_xyz": torch.randn(batch_size, frames, 3, 3),
        "valid": torch.ones(batch_size, frames, 3, dtype=torch.bool),
        "action": torch.randn(batch_size, frames - 1, 4),
        "source_index": torch.tensor([0, 0, 1, 1])[:batch_size],
        "episode_id": torch.arange(batch_size),
        "task_name": ["pick", "pick", "button", "button"][:batch_size],
    }
    return batch


def test_anchor_persistent_model_reuses_official_lewm_components() -> None:
    model = _model()
    assert model.checkpoint_type == ANCHOR_PERSISTENT_CHECKPOINT_TYPE
    assert isinstance(model.predictor, ARPredictor)
    assert isinstance(model.action_encoder, Embedder)
    assert isinstance(model.projector, MLP)
    assert isinstance(model.pred_proj, MLP)
    assert isinstance(model.sigreg, SIGReg)


def test_anchor_axis_survives_teacher_forcing_and_backpropagation() -> None:
    torch.manual_seed(71)
    model = _model().train()
    output = model(_batch(), sigreg_weight=0.02, per_source_sigreg=True)

    assert output.frame_latent.shape == (4, 4, 3, 16)
    assert output.embedding.shape == (4, 4, 3, 16)
    assert output.predicted_embedding.shape == (4, 3, 3, 16)
    assert output.target_embedding.shape == (4, 3, 3, 16)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.frame_encoder.token_type.grad is not None
    assert model.predictor.pos_embedding.grad is not None


def test_predict_preserves_anchor_identity_and_repeats_actions_per_anchor() -> None:
    model = _model().eval()
    seen: dict[str, torch.Tensor] = {}

    def capture_action(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen["action"] = inputs[0].detach().clone()

    handle = model.action_encoder.register_forward_pre_hook(capture_action)
    embedding = torch.randn(2, 3, 3, 16)
    action = torch.randn(2, 3, 4)
    with torch.no_grad():
        predicted = model.predict(embedding, action)
    handle.remove()

    assert predicted.shape == embedding.shape
    assert seen["action"].shape == (6, 3, 4)
    expected = action[:, None].expand(-1, 3, -1, -1).reshape(6, 3, 4)
    torch.testing.assert_close(seen["action"], expected)


def test_anchor_rollout_keeps_candidate_time_and_anchor_axes() -> None:
    model = _model().eval()
    with torch.no_grad():
        rollout = model.rollout_anchor_embeddings(
            history_embedding=torch.randn(2, 3, 3, 16),
            history_action=torch.randn(2, 2, 4),
            future_action=torch.randn(2, 5, 4, 4),
        )
    assert rollout.shape == (2, 5, 4, 3, 16)


def test_innovation_features_and_scorer_are_candidate_specific() -> None:
    torch.manual_seed(73)
    candidate = torch.randn(2, 5, 3, 16, requires_grad=True)
    prior = torch.randn(2, 3, 16)
    features = anchor_innovation_features(candidate, prior)
    assert features.shape == (2, 5, 3, 82)
    squared_distance = features[..., -2]
    expected_distance = (candidate - prior[:, None]).square().mean(dim=-1)
    torch.testing.assert_close(squared_distance, expected_distance)

    scorer = AnchorInnovationScorer(latent_dim=16, num_keypoints=3, hidden_dim=24)
    scores = scorer(candidate, prior)
    assert scores.shape == (2, 5)
    assert not torch.equal(scores[:, 0], scores[:, 1])
    scores.sum().backward()
    assert candidate.grad is not None


def test_anchor_model_rejects_frozen_oracle_target_mode() -> None:
    frame = _model().frame_encoder
    with pytest.raises(ValueError, match="online targets only"):
        AnchorPersistentLeWM(
            frame_encoder=frame,
            action_dim=4,
            history_size=3,
            target_mode="frozen_oracle",
            alignment_weight=1.0,
        )


def test_runtime_builds_variant_without_changing_legacy_default(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "LeWMViTDenseEncoder", _TinyVision)
    base = {
        "vision": {},
        "frame": {
            "obs_dim": 4,
            "num_keypoints": 3,
            "latent_dim": 16,
            "depth": 1,
            "heads": 4,
            "mlp_ratio": 2,
            "dropout": 0.0,
        },
        "world_model": {
            "action_dim": 4,
            "history_size": 3,
            "predictor_depth": 1,
            "predictor_heads": 4,
            "action_smoothing_dim": 4,
            "predictor_mlp_dim": 32,
            "predictor_dim_head": 4,
            "predictor_dropout": 0.0,
            "projector_hidden_dim": 32,
            "sigreg_knots": 5,
            "sigreg_num_proj": 8,
            "target_mode": "online",
            "alignment_weight": 0.0,
        },
    }
    legacy = runtime._build_from_model_config(base)
    anchor = runtime._build_from_model_config(
        {**base, "variant": "anchor_persistent"}
    )
    assert isinstance(legacy, GroundedLeWM)
    assert isinstance(anchor, AnchorPersistentLeWM)


def test_anchor_evaluation_reports_dynamics_and_measurement_innovation() -> None:
    torch.manual_seed(79)
    model = _model().eval()
    metrics = evaluate_anchor_persistent(
        model,
        [_batch()],
        torch.device("cpu"),
        amp_enabled=False,
    )

    assert metrics["samples"] == 4
    assert metrics["num_keypoints"] == 3
    assert len(metrics["factual_per_anchor_mse"]) == 3
    assert len(metrics["fcpg_measurement_closer_rate_per_anchor"]) == 3
    assert metrics["shuffled_mse"] >= 0.0
    assert metrics["shuffle_contract"] == "same_task_different_episode_only"
    assert metrics["strict_shuffle_samples"] == 4
    assert metrics["nominal_intervention_mse"] >= 0.0
    assert len(metrics["representation"]["per_anchor"]) == 3
    assert metrics["representation"]["all_anchors"]["effective_rank"] > 0.0


def test_same_task_shuffle_never_uses_cross_task_or_same_episode_donor() -> None:
    action = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1)
    batch = {
        "task_name": ["pick", "pick", "pick", "button", "button", "button"],
        "episode_id": torch.tensor([0, 1, 2, 7, 8, 9]),
    }
    shuffled, valid = same_task_shuffled_actions(action, batch)
    assert shuffled is not None
    assert bool(valid.all())
    donor = shuffled[:, 0, 0].long().tolist()
    tasks = batch["task_name"]
    episodes = batch["episode_id"].tolist()
    for target, source in enumerate(donor):
        assert tasks[target] == tasks[source]
        assert episodes[target] != episodes[source]


def test_same_task_shuffle_masks_an_impossible_task_stratum() -> None:
    action = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1)
    batch = {
        "task_name": ["pick", "pick", "button", "button", "button"],
        "episode_id": torch.tensor([0, 1, 4, 4, 5]),
    }
    shuffled, valid = same_task_shuffled_actions(action, batch)
    assert shuffled is not None
    assert valid.tolist() == [True, True, False, False, False]
    torch.testing.assert_close(shuffled[2:], action[2:])


def test_anchor_evaluation_reports_optional_occlusion_segments() -> None:
    torch.manual_seed(83)
    model = _model().eval()
    batch = _batch()
    mask = torch.zeros(4, 4, 3, dtype=torch.bool)
    mask[:, 2, 0] = True
    batch["anchor_dropout_active"] = mask
    batch["anchor_recovery_onset"] = torch.roll(mask, shifts=1, dims=1)
    batch["anchor_recovery_segment"] = torch.roll(mask, shifts=1, dims=1)
    metrics = evaluate_anchor_persistent(
        model,
        [batch],
        torch.device("cpu"),
        amp_enabled=False,
    )
    assert set(metrics["segments"]) == {
        "anchor_dropout_active",
        "anchor_recovery_onset",
        "anchor_recovery_segment",
    }
    recovery = metrics["segments"]["anchor_recovery_onset"]
    assert recovery["factual_element_count"] == 4 * 16
    assert recovery["factual_per_anchor_mse"][0] >= 0.0
    assert torch.isnan(torch.tensor(recovery["factual_per_anchor_mse"][1]))
