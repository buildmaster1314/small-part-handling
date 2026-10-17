from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fcpg.action_chunk_policy as action_chunk_policy
from scripts.train_action_chunk_policy import (
    apply_projection_jitter,
    compute_sigreg_loss,
    resolve_policy_weights,
)


class _FakeVisionEncoder(nn.Module):
    def __init__(self, output_dim: int, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.backbone_fully_frozen = True
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, rgb: torch.Tensor) -> SimpleNamespace:
        pooled = rgb.mean(dim=1, keepdim=True)
        feature_map = pooled.expand(-1, self.output_dim, -1, -1) * self.scale
        return SimpleNamespace(
            feature_map=feature_map,
            global_feature=feature_map.mean(dim=(-2, -1)),
        )


class _UnusedResidualAdapter(nn.Module):
    def forward(self, *_: torch.Tensor) -> SimpleNamespace:
        raise AssertionError("cached residual corrections should bypass the adapter")


def _batch(
    batch_size: int = 2,
    obs_horizon: int = 2,
    action_horizon: int = 4,
    num_keypoints: int = 3,
) -> dict[str, torch.Tensor]:
    q_nominal = torch.rand(batch_size, obs_horizon, num_keypoints, 2) * 31.0
    next_q_nominal = torch.rand(batch_size, obs_horizon, num_keypoints, 2) * 31.0
    return {
        "rgb": torch.randn(batch_size, obs_horizon, 3, 32, 32),
        "obs": torch.randn(batch_size, obs_horizon, 4),
        "q_nominal": q_nominal,
        "q_true": q_nominal,
        "keypoints_xyz": torch.randn(batch_size, obs_horizon, num_keypoints, 3),
        "next_rgb": torch.randn(batch_size, obs_horizon, 3, 32, 32) + 1.0,
        "next_obs": torch.randn(batch_size, obs_horizon, 4) + 1.0,
        "next_q_nominal": next_q_nominal,
        "next_q_true": next_q_nominal,
        "next_keypoints_xyz": torch.randn(batch_size, obs_horizon, num_keypoints, 3),
        "history_action": torch.randn(batch_size, obs_horizon, 4),
        "next_mask": torch.ones(batch_size, dtype=torch.bool),
        "action": torch.cat(
            [
                torch.rand(batch_size, action_horizon, 3) * 2.0 - 1.0,
                torch.randint(0, 2, (batch_size, action_horizon, 1)).float(),
            ],
            dim=-1,
        ),
        "action_mask": torch.ones(batch_size, action_horizon, dtype=torch.bool),
    }


def _model(
    monkeypatch: pytest.MonkeyPatch,
    mode: str = "geoprop",
) -> action_chunk_policy.GroundedActionChunkTransformer:
    monkeypatch.setattr(action_chunk_policy, "ViTDenseEncoder", _FakeVisionEncoder)
    return action_chunk_policy.GroundedActionChunkTransformer(
        mode=mode,
        obs_dim=4,
        action_dim=4,
        num_keypoints=3,
        obs_horizon=2,
        action_horizon=4,
        model_dim=32,
        vision_dim=16,
        encoder_layers=1,
        decoder_layers=1,
        predictor_layers=1,
        heads=4,
        dropout=0.0,
        pretrained_vision=False,
        fcpg_adapter=_UnusedResidualAdapter() if mode == "fcpg" else None,
        policy_arch="lewm-predictive",
    )


def test_predictive_policy_weights_are_architecture_specific() -> None:
    assert resolve_policy_weights("legacy", None, None) == (0.0, 0.0)
    assert resolve_policy_weights("lewm-predictive", None, None) == (1.0, 0.02)
    assert resolve_policy_weights("lewm-predictive", 0.0, 0.0) == (0.0, 0.0)
    assert resolve_policy_weights("lewm-predictive", 1.0, 0.02) == (1.0, 0.02)
    with pytest.raises(ValueError, match="legacy policy"):
        resolve_policy_weights("legacy", 1.0, 0.0)
    with pytest.raises(ValueError, match="matched SIGReg"):
        resolve_policy_weights("lewm-predictive", 1.0, 0.1)


def test_projection_jitter_is_coherent_and_preserves_input() -> None:
    q_nominal = torch.zeros(2, 3, 4, 2)
    next_q_nominal = torch.ones_like(q_nominal)
    batch = {
        "q_nominal": q_nominal,
        "next_q_nominal": next_q_nominal,
    }
    torch.manual_seed(7)

    jittered = apply_projection_jitter(batch, std_px=8.0, max_px=10.0)

    delta = jittered["q_nominal"] - q_nominal
    next_delta = jittered["next_q_nominal"] - next_q_nominal
    torch.testing.assert_close(delta, delta[:, :1, :1].expand_as(delta))
    torch.testing.assert_close(next_delta, delta)
    assert bool((torch.linalg.vector_norm(delta[:, 0, 0], dim=-1) <= 10.0).all())
    torch.testing.assert_close(batch["q_nominal"], torch.zeros_like(q_nominal))
    torch.testing.assert_close(
        batch["next_q_nominal"], torch.ones_like(next_q_nominal)
    )


def test_zero_projection_jitter_is_noop() -> None:
    batch = {"q_nominal": torch.zeros(1, 2, 3, 2)}

    assert apply_projection_jitter(batch, std_px=0.0, max_px=0.0) is batch


def test_per_source_sigreg_weights_sources_equally() -> None:
    class MeanRegularizer(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.float().mean()

    representation = torch.tensor([[0.0], [0.0], [0.0], [10.0], [10.0]])
    source_index = torch.tensor([0, 0, 0, 1, 1])
    regularizer = MeanRegularizer()

    pooled = compute_sigreg_loss(
        regularizer, representation, source_index, "pooled"
    )
    per_source = compute_sigreg_loss(
        regularizer, representation, source_index, "per-source"
    )

    assert pooled == pytest.approx(4.0)
    assert per_source == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("checkpoint_update", "message"),
    [
        ({"checkpoint_type": "wrong"}, "not an FCPG action policy"),
        ({"policy_arch": "legacy"}, "architecture is inconsistent"),
        ({"lewm_upstream_commit": "wrong"}, "LeWM upstream commit"),
    ],
)
def test_schema_v2_loader_rejects_inconsistent_predictive_metadata(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_update: dict[str, object],
    message: str,
) -> None:
    checkpoint = {
        "schema_version": 2,
        "checkpoint_type": "fcpg_action_policy",
        "policy_arch": "lewm-predictive",
        "lewm_upstream_commit": action_chunk_policy.LEWM_UPSTREAM_COMMIT,
        "args": {"mode": "geoprop", "policy_arch": "lewm-predictive"},
    }
    checkpoint.update(checkpoint_update)
    monkeypatch.setattr(
        action_chunk_policy.torch,
        "load",
        lambda *_args, **_kwargs: checkpoint,
    )

    with pytest.raises(ValueError, match=message):
        action_chunk_policy.load_action_chunk_policy("unused.pt")


def test_residual_only_cache_uses_neutral_reliability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch, mode="fcpg").eval()
    batch = _batch()
    batch["fcpg_q_corrected"] = batch["q_nominal"] + 2.0
    batch["next_fcpg_q_corrected"] = batch["next_q_nominal"] + 3.0

    output = model.predict_action_chunk(batch)

    torch.testing.assert_close(output.grounding_points_px, batch["fcpg_q_corrected"])
    torch.testing.assert_close(output.reliability, torch.ones_like(output.reliability))
    assert output.next_latent_target is not None


def test_predictive_policy_has_shared_online_target_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    model = _model(monkeypatch)
    batch = _batch()

    loss, parts, output = model.compute_loss(batch, latent_pred_weight=1.0)

    assert output.action.shape == (2, 4, 4)
    assert output.state_latent is not None and output.state_latent.shape == (2, 32)
    assert output.next_latent is not None and output.next_latent.shape == (2, 32)
    assert output.next_latent_target is not None
    assert output.next_latent_target.shape == (2, 32)
    assert output.next_latent_target.requires_grad
    assert set(parts) == {"motion_loss", "gripper_loss", "next_latent_loss"}

    output.next_latent_target.retain_grad()
    loss.backward()

    assert output.next_latent_target.grad is not None
    assert bool(torch.any(output.next_latent_target.grad != 0))
    encoder_grads = [
        parameter.grad
        for parameter in model.memory_encoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    predictor_grads = [
        parameter.grad
        for parameter in model.next_latent_predictor.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert encoder_grads and any(bool(torch.any(grad != 0)) for grad in encoder_grads)
    assert predictor_grads and any(bool(torch.any(grad != 0)) for grad in predictor_grads)


def test_predictive_auxiliary_actions_do_not_change_policy_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    model = _model(monkeypatch).eval()
    batch = _batch()

    with torch.no_grad():
        baseline = model(batch).action.clone()
        batch["history_action"] = batch["history_action"] + 100.0
        changed = model(batch)

    torch.testing.assert_close(changed.action, baseline)
    assert changed.next_latent is not None


def test_terminal_mask_removes_next_latent_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch)
    batch = _batch()
    batch["next_mask"] = torch.zeros(2, dtype=torch.bool)

    _, parts, _ = model.compute_loss(batch, latent_pred_weight=1.0)

    torch.testing.assert_close(
        parts["next_latent_loss"],
        torch.zeros_like(parts["next_latent_loss"]),
    )


def test_predictive_policy_uses_compact_visual_cache_without_dense_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch, mode="fcpg")
    batch = _batch()
    batch.pop("rgb")
    batch.pop("next_rgb")
    batch["sampled_visual"] = torch.randn(2, 2, 3, 16)
    batch["global_visual"] = torch.randn(2, 2, 16)
    batch["next_sampled_visual"] = torch.randn(2, 2, 3, 16)
    batch["next_global_visual"] = torch.randn(2, 2, 16)
    batch["image_size"] = torch.tensor([[32, 32], [32, 32]])
    batch["fcpg_q_corrected"] = batch["q_nominal"] + 2.0
    batch["next_fcpg_q_corrected"] = batch["next_q_nominal"] + 3.0

    loss, parts, output = model.compute_loss(batch, latent_pred_weight=1.0)
    loss.backward()

    assert output.action.shape == (2, 4, 4)
    assert output.next_latent_target is not None
    assert "next_latent_loss" in parts
    torch.testing.assert_close(output.grounding_points_px, batch["fcpg_q_corrected"])
