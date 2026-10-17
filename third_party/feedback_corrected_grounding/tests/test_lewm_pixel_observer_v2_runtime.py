from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fcpg.lewm_pixel_observer_v2 import (
    ORACLE_CEILING_TARGET_MODE,
    PAIRED_DEPLOYMENT_TARGET_MODE,
    SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
    PixelResidualVelocityInnovationAdapter,
)
from fcpg.lewm_pixel_observer_v2_runtime import (
    CausalLeWMPixelObserverV2Runtime,
    load_pixel_observer_v2_checkpoint,
    save_pixel_observer_v2_checkpoint,
)


class _DeterministicGroundedLeWM(nn.Module):
    history_size = 2
    latent_dim = 5
    action_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.frame_encoder = SimpleNamespace(num_keypoints=3)
        self.scale = nn.Parameter(torch.ones(()))

    def encode_frames(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grounded = batch.get("q_corrected", batch["q_nominal"])
        scalar = grounded.mean(dim=(2, 3)).unsqueeze(-1)
        embedding = self.scale * scalar.repeat(1, 1, self.latent_dim)
        return embedding, embedding

    def predict(
        self, embedding: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return embedding + action.mean(dim=-1, keepdim=True)


def _history() -> dict[str, torch.Tensor]:
    q_nominal = torch.zeros(2, 2, 3, 2)
    return {
        "pixels": torch.zeros(2, 2, 3, 8, 8),
        "proprio": torch.zeros(2, 2, 4),
        "q_nominal": q_nominal,
        "q_corrected": q_nominal + 0.5,
        "keypoints_xyz": torch.zeros(2, 2, 3, 3),
        "valid": torch.ones(2, 2, 3, dtype=torch.bool),
    }


def _current() -> dict[str, torch.Tensor]:
    return {
        "posterior_residual_px": torch.zeros(2, 3, 2),
        "residual_velocity_px": torch.full((2, 3, 2), 0.1),
        "previous_q_nominal_px": torch.zeros(2, 3, 2),
        "current_q_nominal_px": torch.ones(2, 3, 2),
        "previous_keypoints_xyz": torch.zeros(2, 3, 3),
        "current_keypoints_xyz": torch.ones(2, 3, 3),
        "previous_nominal_valid": torch.ones(2, 3, dtype=torch.bool),
        "current_nominal_valid": torch.ones(2, 3, dtype=torch.bool),
        "previous_measurement_valid": torch.ones(2, 3, dtype=torch.bool),
        "measurement_age_steps": torch.zeros(2, 3),
    }


def _runtime() -> CausalLeWMPixelObserverV2Runtime:
    torch.manual_seed(41)
    base = _DeterministicGroundedLeWM()
    adapter = PixelResidualVelocityInnovationAdapter(
        latent_dim=5,
        num_keypoints=3,
        image_size=(32, 24),
        hidden_dim=12,
        max_velocity_innovation_px=0.5,
    )
    with torch.no_grad():
        adapter.net[-1].weight.copy_(
            torch.linspace(
                -0.08, 0.09, adapter.net[-1].weight.numel()
            ).reshape_as(adapter.net[-1].weight)
        )
        adapter.net[-1].bias.copy_(torch.tensor([0.01, -0.02]))
    return CausalLeWMPixelObserverV2Runtime(base, adapter)


def _predict(
    runtime: CausalLeWMPixelObserverV2Runtime,
    *,
    history: dict[str, torch.Tensor] | None = None,
):
    return runtime.predict_prior(
        history_batch=_history() if history is None else history,
        transition_actions=torch.full((2, 2, 2), 0.25),
        **_current(),
    )


def _paired_provenance() -> dict[str, str]:
    return {
        "dataset": "unit-test",
        "target_contract": "independent_a_b_mutual_supervision",
        "measurement_a_sha256": "a" * 64,
        "measurement_b_sha256": "b" * 64,
    }


def test_runtime_owns_embedding_and_rejects_current_or_oracle_history() -> None:
    runtime = _runtime()
    reference = _predict(runtime)
    for forbidden in (
        "predicted_embedding",
        "current_rgb",
        "current_q_corrected",
        "q_true",
        "oracle_target",
    ):
        history = _history()
        history[forbidden] = torch.zeros(2, 2, 1)
        with pytest.raises(ValueError, match="forbidden"):
            _predict(runtime, history=history)
    with pytest.raises(TypeError, match="unexpected keyword"):
        runtime.predict_prior(
            history_batch=_history(),
            transition_actions=torch.zeros(2, 2, 2),
            predicted_embedding=torch.zeros(2, 5),
            **_current(),
        )

    # Target-frame data has no input channel, so mutating it is invariant.
    current_rgb = torch.randn(2, 3, 8, 8)
    current_q_corrected = torch.randn(2, 3, 2)
    q_true = torch.randn(2, 3, 2)
    current_rgb.add_(100.0)
    current_q_corrected.mul_(-5.0)
    q_true.zero_()
    repeated = _predict(runtime)
    torch.testing.assert_close(reference.residual_px, repeated.residual_px)


def test_runtime_uses_history_and_keeps_base_frozen() -> None:
    runtime = _runtime()
    first = _predict(runtime)
    history = _history()
    history["q_corrected"] = history["q_corrected"] + 4.0
    second = _predict(runtime, history=history)
    assert not torch.equal(first.velocity_innovation_px, second.velocity_innovation_px)
    assert runtime.base_model.training is False
    assert all(not p.requires_grad for p in runtime.base_model.parameters())
    second.residual_px.square().mean().backward()
    assert all(p.grad is None for p in runtime.base_model.parameters())
    assert runtime.adapter.net[-1].weight.grad is not None


def test_no_latent_ablation_skips_frozen_base_forward() -> None:
    runtime = _runtime()
    runtime.adapter.use_predicted_latent = False

    def forbidden_encode(_batch):
        raise AssertionError("no-latent ablation must not run the frozen base")

    runtime.base_model.encode_frames = forbidden_encode
    result = _predict(runtime)
    assert result.residual_px.shape == (2, 3, 2)


def test_runtime_rejects_mixed_dtype() -> None:
    runtime = _runtime()
    current = _current()
    current["current_q_nominal_px"] = current["current_q_nominal_px"].double()
    with pytest.raises(TypeError, match="dtype"):
        runtime.predict_prior(
            history_batch=_history(),
            transition_actions=torch.zeros(2, 2, 2),
            **current,
        )


def test_paired_checkpoint_round_trip_is_deployable(tmp_path) -> None:
    runtime = _runtime()
    path = tmp_path / "paired.pt"
    provenance = _paired_provenance()
    save_pixel_observer_v2_checkpoint(
        path,
        runtime,
        target_mode=PAIRED_DEPLOYMENT_TARGET_MODE,
        data_provenance=provenance,
    )
    loaded, payload = load_pixel_observer_v2_checkpoint(
        path,
        base_model=_DeterministicGroundedLeWM(),
        expected_data_provenance=provenance,
        for_deployment=True,
    )
    assert payload["target_mode"] == PAIRED_DEPLOYMENT_TARGET_MODE
    assert loaded.base_model.training is False
    torch.testing.assert_close(
        _predict(runtime).residual_px, _predict(loaded).residual_px
    )


@pytest.mark.parametrize(
    "target_mode",
    [ORACLE_CEILING_TARGET_MODE, SINGLE_MEASUREMENT_BASELINE_TARGET_MODE],
)
def test_diagnostic_checkpoint_is_rejected_for_deployment(
    tmp_path, target_mode: str
) -> None:
    runtime = _runtime()
    path = tmp_path / f"{target_mode}.pt"
    provenance = {"dataset": "unit-test", "target_mode": target_mode}
    save_pixel_observer_v2_checkpoint(
        path,
        runtime,
        target_mode=target_mode,
        data_provenance=provenance,
    )
    with pytest.raises(ValueError, match="only independent A/B"):
        load_pixel_observer_v2_checkpoint(
            path,
            base_model=_DeterministicGroundedLeWM(),
            expected_data_provenance=provenance,
            for_deployment=True,
        )


def test_checkpoint_mode_and_provenance_are_bound(tmp_path) -> None:
    runtime = _runtime()
    oracle_provenance = {"dataset": "oracle-unit-test"}
    oracle_path = tmp_path / "oracle.pt"
    save_pixel_observer_v2_checkpoint(
        oracle_path,
        runtime,
        target_mode=ORACLE_CEILING_TARGET_MODE,
        data_provenance=oracle_provenance,
    )
    payload = torch.load(oracle_path, weights_only=False)
    payload["target_mode"] = PAIRED_DEPLOYMENT_TARGET_MODE
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="metadata|binding"):
        load_pixel_observer_v2_checkpoint(
            tampered,
            base_model=_DeterministicGroundedLeWM(),
            expected_data_provenance=_paired_provenance(),
            for_deployment=True,
        )

    paired_path = tmp_path / "paired.pt"
    paired = _paired_provenance()
    save_pixel_observer_v2_checkpoint(
        paired_path,
        runtime,
        target_mode=PAIRED_DEPLOYMENT_TARGET_MODE,
        data_provenance=paired,
    )
    changed = dict(paired)
    changed["dataset"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        load_pixel_observer_v2_checkpoint(
            paired_path,
            base_model=_DeterministicGroundedLeWM(),
            expected_data_provenance=changed,
            for_deployment=True,
        )
