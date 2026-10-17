from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import fcpg.action_chunk_policy as action_chunk_policy
import fcpg.lewm_known_drop_policy_evaluator as evaluator
from fcpg.lewm_known_drop_bridge_runtime import CausalKnownDropBridgeRuntime


class _FakeGroundingAdapter(nn.Module):
    def forward_sampled(
        self,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        return visual[..., :4] + proprio.mean(dim=-1, keepdim=True) + reliability[..., None]


class _FakeVisionEncoder(nn.Module):
    def __init__(self, output_dim: int, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.backbone_fully_frozen = True

    def forward(self, rgb: torch.Tensor) -> SimpleNamespace:
        feature = rgb.mean(dim=1, keepdim=True).expand(
            -1, self.output_dim, -1, -1
        )
        return SimpleNamespace(
            feature_map=feature,
            global_feature=feature.mean(dim=(-2, -1)),
        )


class _FakePolicy(nn.Module):
    num_keypoints = 2
    vision_dim = 6
    model_dim = 4
    obs_horizon = 2

    def __init__(self) -> None:
        super().__init__()
        self.grounding_adapter = _FakeGroundingAdapter()
        self.seen_batches: list[dict[str, torch.Tensor]] = []

    def _proprio(self, xyz: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(2, dtype=xyz.dtype, device=xyz.device)
        identity = identity.unsqueeze(0).expand(xyz.shape[0], -1, -1)
        return torch.cat((xyz, identity), dim=-1)

    def _observation_tokens(self, batch: dict[str, torch.Tensor]):
        sampled = batch["sampled_visual"]
        xyz = batch["keypoints_xyz"]
        batch_size, history, keypoints = sampled.shape[:3]
        reliability = torch.ones(batch_size, history, keypoints)
        flat = self.grounding_adapter.forward_sampled(
            sampled.reshape(batch_size * history, keypoints, 6),
            self._proprio(xyz.reshape(batch_size * history, keypoints, 3)),
            reliability.reshape(batch_size * history, keypoints),
        ).reshape(batch_size, history, keypoints, 4)
        if "grounded_token_override" in batch:
            flat = torch.where(
                batch["grounded_token_override_mask"].unsqueeze(-1),
                batch["grounded_token_override"],
                flat,
            )
        global_token = torch.zeros(batch_size, history, 1, 4)
        robot_token = torch.zeros(batch_size, history, 1, 4)
        memory = torch.cat((global_token, robot_token, flat), dim=2)
        return memory, batch["q_nominal"], reliability

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]):
        self.seen_batches.append(dict(batch))
        memory, _, _ = self._observation_tokens(batch)
        scalar = memory[:, -1, 2:].mean(dim=(1, 2))
        return SimpleNamespace(action=scalar[:, None, None].expand(-1, 8, 4).clone())


class _FakeAnchorEncoder:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def encode(self, records):
        record = dict(records[0])
        self.records.append(record)
        value = float(np.asarray(record["rgb"]).mean())
        return np.full((1, 2, 192), value, dtype=np.float32)


class _FakePredictor(nn.Module):
    history_size = 3
    num_keypoints = 2
    latent_dim = 192
    action_dim = 20

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        result = embedding.clone()
        result[:, -1] += action[:, -1].mean(dim=-1)[:, None, None]
        return result


class _FakeBridge(nn.Module):
    def forward(self, predicted, last_embedding, last_token, current_xyz, last_xyz):
        delta = (predicted - last_embedding).mean(dim=-1, keepdim=True)
        return last_token + delta


def _batch(value: float, *, privileged: bool = True) -> dict[str, torch.Tensor]:
    result = {
        "obs": torch.zeros(1, 2, 4),
        "q_nominal": torch.zeros(1, 2, 2, 2),
        "keypoints_xyz": torch.full((1, 2, 2, 3), value * 0.1),
        "sampled_visual": torch.full((1, 2, 2, 6), value),
        "global_visual": torch.zeros(1, 2, 6),
        "image_size": torch.tensor([[32, 32]]),
    }
    if privileged:
        result["q_true"] = torch.full((1, 2, 2, 2), 17.0)
        result["drop_active"] = torch.ones(1, dtype=torch.bool)
    return result


def _record(value: float) -> dict[str, np.ndarray]:
    return {
        "rgb": np.full((3, 8, 8), value, dtype=np.float32),
        "proprio": np.full((4,), value, dtype=np.float32),
        "q_nominal": np.zeros((2, 2), dtype=np.float32),
        "keypoints_xyz": np.full((2, 3), value * 0.1, dtype=np.float32),
        "valid_nominal": np.ones((2,), dtype=bool),
        "q_true": np.full((2, 2), 99.0, dtype=np.float32),
        "canonical_target": np.full((2, 2), 101.0, dtype=np.float32),
    }


def _controller(method: str):
    policy = _FakePolicy().requires_grad_(False).eval()
    encoder = _FakeAnchorEncoder()
    kwargs = {}
    if method == "bridge_factual":
        kwargs = {
            "factual_predictor": _FakePredictor(),
            "factual_bridge": _FakeBridge(),
        }
    elif method == "bridge_geometry":
        kwargs = {"geometry_policy": policy}
    runtime = CausalKnownDropBridgeRuntime(method=method, **kwargs)
    return evaluator.CausalKnownDropPolicyController(
        policy=policy,
        anchor_encoder=encoder,
        runtime=runtime,
    ), policy, encoder


def _warmup(controller: evaluator.CausalKnownDropPolicyController) -> None:
    for index in range(3):
        controller.select_valid(_batch(float(index)), _record(float(index)))
        controller.register_executed_action(torch.full((1, 5, 4), float(index + 1)))


def test_exact_policy_extraction_reconstructs_native_tokens_and_strips_privilege() -> None:
    policy = _FakePolicy().requires_grad_(False).eval()
    batch = _batch(2.0)
    result = evaluator.extract_policy_anchor_observation(policy, batch)

    assert result.sampled_visual.shape == (1, 2, 2, 6)
    assert result.native_token.shape == (1, 2, 2, 4)
    assert result.global_visual_token.shape == (1, 2, 4)
    assert "q_true" in batch


def test_exact_rgb_policy_extraction_matches_native_grounding_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(action_chunk_policy, "ViTDenseEncoder", _FakeVisionEncoder)
    policy = action_chunk_policy.GroundedActionChunkTransformer(
        mode="geoprop",
        obs_dim=4,
        action_dim=4,
        num_keypoints=2,
        obs_horizon=2,
        action_horizon=8,
        model_dim=8,
        vision_dim=6,
        encoder_layers=1,
        decoder_layers=1,
        heads=2,
        dropout=0.0,
        pretrained_vision=False,
    ).requires_grad_(False).eval()
    batch = {
        "obs": torch.zeros(1, 2, 4),
        "q_nominal": torch.tensor(
            [[[[2.0, 2.0], [5.0, 5.0]], [[3.0, 3.0], [4.0, 4.0]]]]
        ),
        "keypoints_xyz": torch.randn(1, 2, 2, 3),
        "rgb": torch.rand(1, 2, 3, 8, 8),
        "q_true": torch.full((1, 2, 2, 2), 77.0),
    }

    result = evaluator.extract_policy_anchor_observation(policy, batch)
    assert result.sampled_visual.shape == (1, 2, 2, 6)
    assert result.native_token.shape == (1, 2, 2, 8)
    assert result.global_visual_token.shape == (1, 2, 8)


def test_native_extraction_rejects_global_visual_override() -> None:
    policy = _FakePolicy().requires_grad_(False).eval()
    batch = _batch(2.0)
    batch["global_visual_token_override"] = torch.zeros(1, 2, 4)
    batch["global_visual_token_override_mask"] = torch.ones(
        1, 2, dtype=torch.bool
    )

    with pytest.raises(ValueError, match="pre-existing overrides"):
        evaluator.extract_policy_anchor_observation(policy, batch)


def test_valid_controller_caches_exact_sampled_visual_for_geometry_hold() -> None:
    controller, policy, _ = _controller("bridge_geometry")
    _warmup(controller)
    invalid = controller.select_invalid(_batch(-9.0), _record(-9.0))

    # The invalid corrupted sampled visual is -9, but the geometry token must
    # use the final valid descriptor 2 and current FK from the observed batch.
    assert torch.all(invalid.decision.policy_token > 1.0)
    assert torch.all(invalid.decision.policy_token < 4.0)
    assert "q_true" not in policy.seen_batches[-1]
    assert "drop_active" not in policy.seen_batches[-1]


def test_learned_invalid_path_never_encodes_hidden_or_corrupted_current_image() -> None:
    controller, policy, encoder = _controller("bridge_factual")
    _warmup(controller)
    calls_before = len(encoder.records)
    controller.select_invalid(_batch(-8.0), _record(-8.0))

    assert len(encoder.records) == calls_before
    assert all("q_true" not in record for record in encoder.records)
    assert all("canonical_target" not in record for record in encoder.records)
    assert "q_true" not in policy.seen_batches[-1]


def test_measurement_encodes_only_observed_corrupted_record() -> None:
    controller, _, encoder = _controller("measurement")
    _warmup(controller)
    controller.select_invalid(_batch(-4.0), _record(-4.0))

    assert np.asarray(encoder.records[-1]["rgb"]).mean() == pytest.approx(-4.0)
    assert "q_true" not in encoder.records[-1]


def test_oracle_hidden_clean_data_has_a_separate_guarded_entrypoint() -> None:
    controller, _, encoder = _controller("oracle")
    _warmup(controller)
    with pytest.raises(RuntimeError, match="select_invalid_oracle"):
        controller.select_invalid(_batch(-5.0), _record(-5.0))

    result = controller.select_invalid_oracle(
        _batch(-5.0),
        _batch(7.0),
        _record(7.0),
    )
    assert result.decision.source == "oracle"
    assert np.asarray(encoder.records[-1]["rgb"]).mean() == pytest.approx(7.0)


def test_recovery_uses_native_raw_adjacent_history_without_override() -> None:
    controller, policy, _ = _controller("bridge_factual")
    _warmup(controller)
    controller.select_invalid(_batch(-3.0), _record(-3.0))
    controller.register_executed_action(torch.ones(1, 5, 4))
    recovered = controller.select_valid(_batch(6.0), _record(6.0))

    assert not recovered.decision.override.any
    assert "grounded_token_override" not in policy.seen_batches[-1]
    assert "grounded_token_override_mask" not in policy.seen_batches[-1]


def test_paired_evaluator_reuses_byte_identical_seed_and_drop_contract() -> None:
    contract = evaluator.PairedEpisodeContract(
        env_seed=41,
        task_index=7,
        drop_start=3,
        drop_duration=4,
    )
    seen = []

    def run(method: str, received: evaluator.PairedEpisodeContract):
        seen.append((method, received))
        return {"seed": received.env_seed, "drop": received.drop_start}

    result = evaluator.evaluate_paired_methods(
        ("measurement", "hold_last", "oracle", "bridge_factual"),
        contract,
        run,
    )
    assert set(result) == {"measurement", "hold_last", "oracle", "bridge_factual"}
    assert all(received is contract for _, received in seen)
    assert [item[1].drop_is_active(3) for item in seen] == [True] * 4
