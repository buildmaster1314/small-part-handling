from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fcpg.lewm_known_drop_bridge_runtime as runtime_module
from fcpg.lewm_anchor_policy_bridge import (
    ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE,
    ResidualAnchorPolicyBridge,
)
from fcpg.lewm_known_drop_bridge_runtime import CausalKnownDropBridgeRuntime
from fcpg.lewm_known_drop_bridge_runtime import ANCHOR_POLICY_BRIDGE_SEMANTICS
from fcpg.lewm_policy_token_bridge import sha256_file


class _FakePredictor(nn.Module):
    history_size = 3
    num_keypoints = 2
    latent_dim = 4
    action_dim = 20

    def __init__(self, *, action_scale: float) -> None:
        super().__init__()
        self.action_scale = float(action_scale)
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.calls.append((embedding.detach().clone(), action.detach().clone()))
        result = embedding.clone()
        delta = action.mean(dim=-1, keepdim=True)[:, :, None]
        result[:, -1] = result[:, -1] + self.action_scale * delta[:, -1]
        return result


class _FakeBridge(nn.Module):
    num_keypoints = 2
    latent_dim = 4
    token_dim = 5

    def forward(
        self,
        predicted_embedding: torch.Tensor,
        last_valid_embedding: torch.Tensor,
        last_valid_token: torch.Tensor,
        current_xyz: torch.Tensor,
        last_valid_xyz: torch.Tensor,
    ) -> torch.Tensor:
        latent_delta = (predicted_embedding - last_valid_embedding).mean(
            dim=-1, keepdim=True
        )
        xyz_delta = (current_xyz - last_valid_xyz).mean(dim=-1, keepdim=True)
        return last_valid_token + latent_delta + xyz_delta


def _state(value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.full((1, 2, 4), value),
        torch.full((1, 2, 5), value * 10.0),
        torch.full((1, 2, 3), value * 0.1),
    )


def _action(value: float) -> torch.Tensor:
    return torch.full((1, 5, 4), value)


def _warmup(runtime: CausalKnownDropBridgeRuntime) -> None:
    for index in range(3):
        kwargs = {}
        if runtime.method == "bridge_geometry":
            kwargs["sampled_visual"] = torch.full((1, 2, 6), float(index))
        runtime.begin_valid(*_state(float(index)), **kwargs)
        runtime.register_executed_action(_action(float(index + 1)))


def test_invalid_prediction_requires_three_valid_decisions_and_actions() -> None:
    runtime = CausalKnownDropBridgeRuntime(
        method="hold_last",
        min_valid_warmup=3,
    )
    for index in range(2):
        runtime.begin_valid(*_state(float(index)))
        runtime.register_executed_action(_action(1.0))

    with pytest.raises(RuntimeError, match="three valid decisions"):
        runtime.begin_invalid_predicted(_state(3.0)[2])


def test_factual_bridge_recurses_on_prior_prediction_and_actual_actions() -> None:
    predictor = _FakePredictor(action_scale=1.0)
    runtime = CausalKnownDropBridgeRuntime(
        method="bridge_factual",
        factual_predictor=predictor,
        factual_bridge=_FakeBridge(),
    )
    _warmup(runtime)

    first = runtime.begin_invalid_predicted(_state(3.0)[2])
    assert torch.allclose(first.embedding, torch.full((1, 2, 4), 5.0))
    runtime.register_executed_action(_action(4.0))
    second = runtime.begin_invalid_predicted(_state(4.0)[2])

    assert len(predictor.calls) == 2
    assert torch.equal(predictor.calls[0][1][0, :, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(predictor.calls[1][0][:, -1], first.embedding)
    assert torch.equal(predictor.calls[1][1][0, :, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.allclose(second.embedding, torch.full((1, 2, 4), 9.0))


def test_invalid_repeats_current_prediction_and_valid_keeps_native_raw_adjacent() -> None:
    runtime = CausalKnownDropBridgeRuntime(
        method="hold_last",
    )
    _warmup(runtime)

    first = runtime.begin_invalid_predicted(_state(3.0)[2])
    assert first.override.mask[0, :, 0].tolist() == [True, True]
    torch.testing.assert_close(first.override.tokens[:, 0], first.policy_token)
    torch.testing.assert_close(first.override.tokens[:, 1], first.policy_token)
    runtime.register_executed_action(_action(4.0))

    second = runtime.begin_invalid_predicted(_state(4.0)[2])
    assert second.override.mask[0, :, 0].tolist() == [True, True]
    torch.testing.assert_close(second.override.tokens[:, 0], second.policy_token)
    torch.testing.assert_close(second.override.tokens[:, 1], second.policy_token)
    runtime.register_executed_action(_action(5.0))

    recovery_state = _state(5.0)
    recovery = runtime.begin_valid(*recovery_state)
    assert not recovery.override.any
    assert torch.equal(recovery.policy_token, recovery_state[1])
    runtime.register_executed_action(_action(6.0))

    settled = runtime.begin_valid(*_state(6.0))
    assert not settled.override.any


def test_learned_path_has_no_hidden_clean_input_and_oracle_is_separate() -> None:
    factual = CausalKnownDropBridgeRuntime(
        method="bridge_factual",
        factual_predictor=_FakePredictor(action_scale=1.0),
        factual_bridge=_FakeBridge(),
    )
    _warmup(factual)
    with pytest.raises(TypeError):
        factual.begin_invalid_predicted(
            _state(3.0)[2],
            hidden_clean_embedding=_state(3.0)[0],
        )
    with pytest.raises(RuntimeError, match="reserved for method='oracle'"):
        factual.begin_invalid_oracle(*_state(3.0))

    oracle = CausalKnownDropBridgeRuntime(method="oracle")
    _warmup(oracle)
    result = oracle.begin_invalid_oracle(*_state(9.0))
    assert result.source == "oracle"
    assert result.override.mask[0, :, 0].tolist() == [True, True]
    torch.testing.assert_close(result.override.tokens[:, 0], result.policy_token)
    torch.testing.assert_close(result.override.tokens[:, 1], result.policy_token)


def test_measurement_baseline_uses_corrupted_state_without_override() -> None:
    runtime = CausalKnownDropBridgeRuntime(method="measurement")
    _warmup(runtime)
    corrupted = _state(-3.0)
    result = runtime.begin_invalid_measurement(*corrupted)

    assert result.source == "corrupted_measurement"
    assert torch.equal(result.embedding, corrupted[0])
    assert torch.equal(result.policy_token, corrupted[1])
    assert not result.override.any


def test_no_action_uses_matched_predictor_and_bridge() -> None:
    factual = _FakePredictor(action_scale=100.0)
    no_action = _FakePredictor(action_scale=0.0)
    runtime = CausalKnownDropBridgeRuntime(
        method="bridge_noaction",
        factual_predictor=factual,
        no_action_predictor=no_action,
        no_action_bridge=_FakeBridge(),
    )
    _warmup(runtime)
    result = runtime.begin_invalid_predicted(_state(3.0)[2])

    assert not factual.calls
    assert len(no_action.calls) == 1
    assert torch.equal(result.embedding, _state(2.0)[0])


class _FakeGroundingAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("frozen_marker", torch.ones(()))
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward_sampled(
        self,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        *,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(
            (
                visual.detach().clone(),
                proprio.detach().clone(),
                reliability.detach().clone(),
            )
        )
        return visual[..., :5] + proprio.mean(dim=-1, keepdim=True)


class _FakeGeometryPolicy(nn.Module):
    num_keypoints = 2
    vision_dim = 6
    model_dim = 5

    def __init__(self) -> None:
        super().__init__()
        self.grounding_adapter = _FakeGroundingAdapter()

    def _proprio(self, xyz: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(2, device=xyz.device, dtype=xyz.dtype)
        identity = identity.unsqueeze(0).expand(xyz.shape[0], -1, -1)
        return torch.cat((xyz, identity), dim=-1)


def test_geometry_control_holds_visual_and_updates_only_current_fk() -> None:
    policy = _FakeGeometryPolicy().requires_grad_(False).eval()
    runtime = CausalKnownDropBridgeRuntime(
        method="bridge_geometry",
        geometry_policy=policy,
    )
    _warmup(runtime)
    result = runtime.begin_invalid_predicted(_state(4.0)[2])

    visual, proprio, reliability = policy.grounding_adapter.calls[-1]
    assert torch.equal(visual, torch.full((1, 2, 6), 2.0))
    assert torch.equal(proprio[..., :3], torch.full((1, 2, 3), 0.4))
    assert torch.equal(reliability, torch.ones(1, 2))
    expected = visual[..., :5] + proprio.mean(dim=-1, keepdim=True)
    assert torch.equal(result.policy_token, expected)


def test_action_registration_enforces_two_phase_cadence_and_20d_shape() -> None:
    runtime = CausalKnownDropBridgeRuntime(method="hold_last")
    with pytest.raises(RuntimeError, match="begin a decision"):
        runtime.register_executed_action(_action(1.0))

    runtime.begin_valid(*_state(0.0))
    with pytest.raises(RuntimeError, match="register the executed action"):
        runtime.begin_valid(*_state(1.0))
    with pytest.raises(ValueError, match=r"\[B,5,4\]"):
        runtime.register_executed_action(torch.zeros(1, 4, 4))
    with pytest.raises(ValueError, match="exactly 20"):
        runtime.register_executed_action(torch.zeros(1, 19))


def test_override_attach_does_not_mutate_batch_and_omits_empty_override() -> None:
    runtime = CausalKnownDropBridgeRuntime(method="hold_last")
    _warmup(runtime)
    valid = runtime.begin_valid(*_state(3.0))
    source = {"obs": torch.ones(1)}
    attached = valid.override.attach(source)
    assert attached == source
    assert attached is not source


def test_bridge_checkpoint_loader_supports_optional_controls(tmp_path: Path) -> None:
    factual = ResidualAnchorPolicyBridge(
        latent_dim=4,
        token_dim=5,
        num_keypoints=2,
        hidden_dim=7,
    )
    no_action = ResidualAnchorPolicyBridge(
        latent_dim=4,
        token_dim=5,
        num_keypoints=2,
        hidden_dim=7,
    )
    checkpoint = tmp_path / "bridge.pt"
    torch.save(
        {
            "checkpoint_type": ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE,
            "bridge_semantics": ANCHOR_POLICY_BRIDGE_SEMANTICS,
            "bridge_config": {
                "latent_dim": 4,
                "token_dim": 5,
                "num_keypoints": 2,
                "hidden_dim": 7,
            },
            "bridge_states": {
                "factual": factual.state_dict(),
                "no_action": no_action.state_dict(),
            },
        },
        checkpoint,
    )

    loaded, payload = runtime_module.load_anchor_policy_bridges(checkpoint)
    assert set(loaded) == {"factual", "no_action"}
    assert payload["checkpoint_type"] == ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE
    assert all(not parameter.requires_grad for model in loaded.values() for parameter in model.parameters())


def test_bridge_checkpoint_rejects_dead_geometry_state(tmp_path: Path) -> None:
    bridge = ResidualAnchorPolicyBridge(
        latent_dim=4,
        token_dim=5,
        num_keypoints=2,
        hidden_dim=7,
    )
    checkpoint = tmp_path / "bridge-with-dead-state.pt"
    torch.save(
        {
            "checkpoint_type": ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE,
            "bridge_semantics": ANCHOR_POLICY_BRIDGE_SEMANTICS,
            "bridge_config": {
                "latent_dim": 4,
                "token_dim": 5,
                "num_keypoints": 2,
                "hidden_dim": 7,
            },
            "bridge_states": {
                "factual": bridge.state_dict(),
                "geometry": bridge.state_dict(),
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="unsupported states.*geometry"):
        runtime_module.load_anchor_policy_bridges(checkpoint)


def test_full_loader_checks_dimensions_and_freezes_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakePolicy(nn.Module):
        num_keypoints = 2
        model_dim = 5

    policy = _FakePolicy()
    factual = _FakePredictor(action_scale=1.0)
    no_action = _FakePredictor(action_scale=0.0)
    bridge = ResidualAnchorPolicyBridge(
        latent_dim=4,
        token_dim=5,
        num_keypoints=2,
        hidden_dim=7,
    )

    policy_path = tmp_path / "policy.pt"
    corrector_path = tmp_path / "corrector.pt"
    dynamics_path = tmp_path / "dynamics.pt"
    for path, contents in (
        (policy_path, b"policy"),
        (corrector_path, b"corrector"),
        (dynamics_path, b"dynamics"),
    ):
        path.write_bytes(contents)
    policy_sha256 = sha256_file(policy_path)
    corrector_sha256 = sha256_file(corrector_path)
    dynamics_sha256 = sha256_file(dynamics_path)

    monkeypatch.setattr(
        "fcpg.action_chunk_policy.load_action_chunk_policy",
        lambda *args, **kwargs: (policy, {"obs_horizon": 2}),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_matched_predictor_pair_checkpoint",
        lambda *args, **kwargs: (
            factual,
            no_action,
            {"step": 10_000, "seed": 29301},
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_anchor_policy_bridges",
        lambda *args, **kwargs: (
            {"factual": bridge},
            {
                "step": 2_000,
                "compatibility": {
                    "single_seed_runtime": True,
                    "p1_policy_checkpoint_sha256": policy_sha256,
                    "p1_corrector_checkpoint_sha256": corrector_sha256,
                    "a3_replicates": [
                        {
                            "seed": 29301,
                            "checkpoint_sha256": dynamics_sha256,
                        }
                    ],
                },
            },
        ),
    )
    bundle = runtime_module.load_known_drop_bridge_models(
        policy_checkpoint=policy_path,
        corrector_checkpoint=corrector_path,
        dynamics_checkpoint=dynamics_path,
        bridge_checkpoint=tmp_path / "bridge.pt",
    )

    assert bundle.methods == ("measurement", "hold_last", "oracle", "bridge_factual")
    assert bundle.make_runtime("bridge_factual").method == "bridge_factual"
    assert not bundle.policy.training
    assert not bundle.factual_predictor.training
    assert all(not parameter.requires_grad for parameter in bundle.bridges["factual"].parameters())

    corrector_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="p1_corrector_checkpoint_sha256 mismatch"):
        runtime_module.load_known_drop_bridge_models(
            policy_checkpoint=policy_path,
            corrector_checkpoint=corrector_path,
            dynamics_checkpoint=dynamics_path,
            bridge_checkpoint=tmp_path / "bridge.pt",
        )
