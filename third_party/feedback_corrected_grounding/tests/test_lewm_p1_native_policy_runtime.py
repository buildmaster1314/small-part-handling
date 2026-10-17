from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fcpg.lewm_p1_native_policy_runtime import (
    P1NativeComplementaryPolicyRuntime,
)


class _GroundingAdapter(nn.Module):
    def forward_sampled(
        self,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        return visual[..., :4] + proprio.mean(dim=-1, keepdim=True)


class _Policy(nn.Module):
    num_keypoints = 2
    vision_dim = 6
    model_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.grounding_adapter = _GroundingAdapter()

    def _proprio(self, xyz: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(2, dtype=xyz.dtype, device=xyz.device)
        return torch.cat((xyz, identity.unsqueeze(0).expand(xyz.shape[0], -1, -1)), dim=-1)


class _Predictor(nn.Module):
    history_size = 3
    num_keypoints = 2
    latent_dim = 4
    action_dim = 20

    def __init__(self, action_scale: float = 1.0) -> None:
        super().__init__()
        self.action_scale = float(action_scale)

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        result = embedding.clone()
        delta = action[:, -1].mean(dim=-1) * self.action_scale
        result[:, -1] += delta[:, None, None]
        return result


def _runtime(
    method: str,
    *,
    freeze_global_visual_during_drop: bool = False,
) -> P1NativeComplementaryPolicyRuntime:
    policy = _Policy().requires_grad_(False).eval()
    factual = _Predictor(1.0).requires_grad_(False).eval()
    no_action = _Predictor(0.0).requires_grad_(False).eval()
    return P1NativeComplementaryPolicyRuntime(
        method=method,
        policy=policy,
        factual_predictor=factual,
        no_action_predictor=no_action,
        freeze_global_visual_during_drop=freeze_global_visual_during_drop,
    )


def _token(value: float) -> torch.Tensor:
    return torch.full((1, 2, 4), value)


def _visual(value: float) -> torch.Tensor:
    return torch.full((1, 2, 6), value)


def _xyz(value: float) -> torch.Tensor:
    return torch.full((1, 2, 3), value)


def _global_visual_token(value: float) -> torch.Tensor:
    return torch.full((1, 4), value)


def _action(value: float) -> torch.Tensor:
    return torch.full((1, 5, 4), value)


def _warmup(runtime: P1NativeComplementaryPolicyRuntime) -> None:
    for index in range(3):
        decision = runtime.begin_valid(
            _token(float(index)),
            _visual(float(index)),
            _global_visual_token(float(index)),
        )
        assert not decision.override_required
        runtime.register_executed_action(_action(float(index + 1)))


def test_factual_complementary_uses_executed_action_and_writes_back() -> None:
    runtime = _runtime("complementary_factual")
    _warmup(runtime)

    first = runtime.begin_invalid(_xyz(1.0))
    assert first.override_required
    assert first.missing_age == 1
    assert first.override_tokens.shape == (1, 2, 2, 4)
    assert torch.equal(first.override_tokens[:, 0], first.policy_token)
    assert torch.equal(first.override_tokens[:, 1], first.policy_token)
    attached = first.attach({"obs": torch.ones(1)})
    assert "global_visual_token_override" not in attached

    trusted_runtime = _runtime(
        "complementary_factual",
        freeze_global_visual_during_drop=True,
    )
    _warmup(trusted_runtime)
    trusted = trusted_runtime.begin_invalid(_xyz(1.0)).attach(
        {"obs": torch.ones(1)}
    )
    assert torch.equal(
        trusted["global_visual_token_override"],
        _global_visual_token(2.0).unsqueeze(1).expand(-1, 2, -1),
    )
    assert bool(trusted["global_visual_token_override_mask"].all())

    # The factual prior uses the already executed action block with value 3.
    # The frozen H3 gain is 0.25, so it must differ from geometry-only.
    geometry = _runtime("geometry_hold")
    _warmup(geometry)
    geometry_first = geometry.begin_invalid(_xyz(1.0))
    assert torch.all(first.policy_token > geometry_first.policy_token)

    runtime.register_executed_action(_action(4.0))
    second = runtime.begin_invalid(_xyz(2.0))
    assert second.missing_age == 2
    assert not torch.equal(second.policy_token, first.policy_token)


def test_matched_no_action_ignores_action_values() -> None:
    left = _runtime("complementary_no_action")
    right = _runtime("complementary_no_action")
    _warmup(left)
    _warmup(right)

    first_left = left.begin_invalid(_xyz(1.0))
    first_right = right.begin_invalid(_xyz(1.0))
    assert torch.equal(first_left.policy_token, first_right.policy_token)
    left.register_executed_action(_action(-7.0))
    right.register_executed_action(_action(9.0))
    second_left = left.begin_invalid(_xyz(2.0))
    second_right = right.begin_invalid(_xyz(2.0))
    assert torch.equal(second_left.policy_token, second_right.policy_token)


def test_measurement_is_native_and_recovery_hard_bypasses_prediction() -> None:
    runtime = _runtime("measurement")
    _warmup(runtime)
    corrupted = _token(-9.0)
    invalid = runtime.begin_invalid(_xyz(1.0), corrupted_token=corrupted)
    assert not invalid.override_required
    assert torch.equal(invalid.policy_token, corrupted)
    runtime.register_executed_action(_action(4.0))

    recovered_token = _token(12.0)
    recovered = runtime.begin_valid(
        recovered_token,
        _visual(12.0),
        _global_visual_token(12.0),
    )
    assert recovered.sensor_valid
    assert recovered.missing_age == 0
    assert runtime.missing_age == 0
    assert not recovered.override_required
    assert torch.equal(recovered.policy_token, recovered_token)
    attached = recovered.attach({"obs": torch.ones(1)})
    assert "grounded_token_override" not in attached
    assert "global_visual_token_override" not in attached


def test_measurement_baseline_keeps_current_global_visual() -> None:
    runtime = _runtime("measurement")
    _warmup(runtime)

    invalid = runtime.begin_invalid(_xyz(1.0), corrupted_token=_token(-4.0))
    attached = invalid.attach({"obs": torch.ones(1)})

    assert "global_visual_token_override" not in attached


def test_visual_cv_uses_two_visible_descriptors_and_current_fk() -> None:
    runtime = _runtime("visual_cv")
    _warmup(runtime)
    decision = runtime.begin_invalid(_xyz(3.0))
    assert decision.source == "visual_cv"
    assert decision.override_required


def test_runtime_rejects_early_drop_pending_action_and_horizon_overflow() -> None:
    runtime = _runtime("hold_last")
    with pytest.raises(RuntimeError, match="three prior decisions"):
        runtime.begin_invalid(_xyz(0.0))

    runtime.begin_valid(_token(0.0), _visual(0.0))
    with pytest.raises(RuntimeError, match="register the executed action"):
        runtime.begin_valid(_token(1.0), _visual(1.0))

    runtime = _runtime("hold_last")
    _warmup(runtime)
    for index in range(6):
        runtime.begin_invalid(_xyz(float(index)))
        runtime.register_executed_action(_action(float(index)))
    with pytest.raises(RuntimeError, match="exceeds the frozen H3-H8"):
        runtime.begin_invalid(_xyz(7.0))


def test_rule_runtime_can_fall_back_to_geometry_after_frozen_horizon() -> None:
    runtime = _runtime("complementary_factual")
    _warmup(runtime)
    for index in range(6):
        runtime.begin_invalid(_xyz(float(index)))
        runtime.register_executed_action(_action(float(index)))
    fallback = runtime.begin_invalid(
        _xyz(7.0), allow_geometry_fallback=True
    )
    assert fallback.source == "geometry_horizon_fallback"
    assert fallback.missing_age == 7
    assert fallback.override_required


def test_complementary_runtime_rejects_corrupted_measurement_input() -> None:
    runtime = _runtime("complementary_factual")
    _warmup(runtime)
    with pytest.raises(ValueError, match="must not consume"):
        runtime.begin_invalid(_xyz(1.0), corrupted_token=_token(-3.0))
