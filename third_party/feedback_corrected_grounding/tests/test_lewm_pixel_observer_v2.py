from __future__ import annotations

import copy

import pytest
import torch

from fcpg.lewm_pixel_observer_v2 import (
    DEPLOYMENT_TARGET_MODE,
    ORACLE_CEILING_TARGET_MODE,
    SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
    PixelResidualVelocityInnovationAdapter,
    build_pixel_observer_v2_paired_supervision,
    build_pixel_observer_v2_supervision,
    masked_huber_pixel_loss,
    paired_mutual_huber_pixel_loss,
    pixel_observer_v2_checkpoint_metadata,
)


def _adapter(*, use_predicted_latent: bool = True) -> PixelResidualVelocityInnovationAdapter:
    torch.manual_seed(29)
    return PixelResidualVelocityInnovationAdapter(
        latent_dim=5,
        num_keypoints=3,
        image_size=(32, 24),
        hidden_dim=12,
        max_velocity_innovation_px=0.5,
        use_predicted_latent=use_predicted_latent,
    )


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "predicted_embedding": torch.randn(2, 5, requires_grad=True),
        "posterior_residual_px": torch.randn(2, 3, 2),
        "residual_velocity_px": 0.1 * torch.randn(2, 3, 2),
        "previous_q_nominal_px": torch.rand(2, 3, 2) * 20.0,
        "current_q_nominal_px": torch.rand(2, 3, 2) * 20.0,
        "previous_keypoints_xyz": torch.randn(2, 3, 3),
        "current_keypoints_xyz": torch.randn(2, 3, 3),
        "previous_nominal_valid": torch.ones(2, 3, dtype=torch.bool),
        "current_nominal_valid": torch.ones(2, 3, dtype=torch.bool),
        "previous_measurement_valid": torch.ones(2, 3, dtype=torch.bool),
        "measurement_age_steps": torch.zeros(2, 3),
    }


def _activate_final_layer(adapter: PixelResidualVelocityInnovationAdapter) -> None:
    with torch.no_grad():
        adapter.net[-1].weight.fill_(0.03)
        adapter.net[-1].bias.copy_(torch.tensor([0.01, -0.02]))


def test_zero_initialized_adapter_is_exact_constant_velocity() -> None:
    adapter = _adapter()
    inputs = _inputs()
    output = adapter.predict_prior(inputs)
    expected_velocity = inputs["residual_velocity_px"]
    expected_residual = inputs["posterior_residual_px"] + expected_velocity

    torch.testing.assert_close(
        output.velocity_innovation_px,
        torch.zeros_like(output.velocity_innovation_px),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(output.velocity_px, expected_velocity, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output.residual_px, expected_residual, rtol=0.0, atol=0.0)

    output.residual_px.square().mean().backward()
    assert adapter.net[-1].weight.grad is not None
    assert float(adapter.net[-1].weight.grad.abs().sum()) > 0.0
    assert inputs["predicted_embedding"].grad is None


def test_learned_velocity_innovation_uses_semi_implicit_update() -> None:
    adapter = _adapter()
    _activate_final_layer(adapter)
    inputs = _inputs()
    output = adapter.predict_prior(inputs)
    expected_velocity = inputs["residual_velocity_px"] + output.velocity_innovation_px
    expected_residual = inputs["posterior_residual_px"] + expected_velocity
    torch.testing.assert_close(output.velocity_px, expected_velocity)
    torch.testing.assert_close(output.residual_px, expected_residual)


def test_velocity_innovation_can_damp_step_induced_pseudo_velocity() -> None:
    adapter = _adapter()
    inputs = _inputs()
    pseudo_velocity = 0.1
    inputs["residual_velocity_px"].fill_(pseudo_velocity)
    normalized = -pseudo_velocity / adapter.max_velocity_innovation_px
    with torch.no_grad():
        adapter.net[-1].weight.zero_()
        adapter.net[-1].bias.fill_(
            0.5 * torch.log(torch.tensor((1.0 + normalized) / (1.0 - normalized)))
        )
    output = adapter.predict_prior(inputs)
    torch.testing.assert_close(
        output.velocity_px, torch.zeros_like(output.velocity_px), atol=1e-7, rtol=0.0
    )
    torch.testing.assert_close(
        output.residual_px,
        inputs["posterior_residual_px"],
        atol=1e-7,
        rtol=0.0,
    )


def test_learned_prior_is_sensitive_to_current_known_geometry() -> None:
    adapter = _adapter()
    _activate_final_layer(adapter)
    reference = _inputs()
    changed = {name: value.detach().clone() for name, value in reference.items()}
    changed["current_q_nominal_px"][:, 1, 0] += 7.0
    changed["current_keypoints_xyz"][:, 2, 1] += 1.5

    first = adapter.predict_prior(reference)
    second = adapter.predict_prior(changed)
    assert not torch.equal(first.velocity_innovation_px, second.velocity_innovation_px)


@pytest.mark.parametrize(
    "forbidden",
    ["current_rgb", "current_q_corrected", "q_true", "true_extrinsic", "frame_id"],
)
def test_prior_api_rejects_current_observation_or_oracle_fields(forbidden: str) -> None:
    inputs = _inputs()
    inputs[forbidden] = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="forbidden"):
        _adapter().predict_prior(inputs)


def test_target_visual_and_correction_are_separate_from_prior() -> None:
    adapter = _adapter()
    _activate_final_layer(adapter)
    inputs = _inputs()
    before = adapter.predict_prior(inputs)

    q_nominal = inputs["current_q_nominal_px"]
    valid = inputs["current_nominal_valid"]
    first_target = build_pixel_observer_v2_supervision(
        q_nominal_px=q_nominal,
        nominal_valid=valid,
        q_corrected_px=q_nominal + 1.0,
        target_mode=SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
    )
    second_target = build_pixel_observer_v2_supervision(
        q_nominal_px=q_nominal,
        nominal_valid=valid,
        q_corrected_px=q_nominal - 9.0,
        target_mode=SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
    )
    after = adapter.predict_prior(inputs)

    torch.testing.assert_close(before.residual_px, after.residual_px, rtol=0.0, atol=0.0)
    assert not torch.equal(first_target.target_residual_px, second_target.target_residual_px)


def test_geometry_only_latent_ablation_is_parameter_matched_and_latent_invariant() -> None:
    with_latent = _adapter(use_predicted_latent=True)
    without_latent = _adapter(use_predicted_latent=False)
    without_latent.load_state_dict(copy.deepcopy(with_latent.state_dict()))
    _activate_final_layer(with_latent)
    without_latent.load_state_dict(copy.deepcopy(with_latent.state_dict()))
    assert sum(p.numel() for p in with_latent.parameters()) == sum(
        p.numel() for p in without_latent.parameters()
    )

    inputs = _inputs()
    changed = dict(inputs)
    changed["predicted_embedding"] = inputs["predicted_embedding"].detach() + 100.0
    no_latent_a = without_latent.predict_prior(inputs)
    no_latent_b = without_latent.predict_prior(changed)
    latent_a = with_latent.predict_prior(inputs)
    latent_b = with_latent.predict_prior(changed)
    torch.testing.assert_close(
        no_latent_a.velocity_innovation_px, no_latent_b.velocity_innovation_px, rtol=0.0, atol=0.0
    )
    assert not torch.equal(latent_a.velocity_innovation_px, latent_b.velocity_innovation_px)
    assert without_latent.export_config()["use_predicted_latent"] is False


def test_jump_mask_and_huber_loss_exclude_unpredictable_frame() -> None:
    q_nominal = torch.zeros(1, 3, 2, 2)
    q_corrected = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]], [[99.0, 0.0], [99.0, 0.0]], [[2.0, 0.0], [2.0, 0.0]]]]
    )
    valid = torch.ones(1, 3, 2, dtype=torch.bool)
    prior_valid = valid.clone()
    prior_valid[:, 1] = False
    supervision = build_pixel_observer_v2_supervision(
        q_nominal_px=q_nominal,
        nominal_valid=valid,
        q_corrected_px=q_corrected,
        prior_supervision_valid=prior_valid,
        target_mode=SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
    )
    prediction = torch.zeros_like(q_nominal, requires_grad=True)
    loss = masked_huber_pixel_loss(prediction, supervision, delta_px=0.5)

    # Four selected anchors: x errors [1, 1, 2, 2], y errors zero.
    expected = (0.375 + 0.375 + 0.875 + 0.875) / 8.0
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()
    assert bool((prediction.grad[:, 1] == 0.0).all())


def test_oracle_ceiling_is_diagnostic_and_checkpoint_is_not_deployable() -> None:
    q_nominal = torch.zeros(1, 3, 2)
    q_true = torch.ones_like(q_nominal)
    valid = torch.ones(1, 3, dtype=torch.bool)
    supervision = build_pixel_observer_v2_supervision(
        q_nominal_px=q_nominal,
        nominal_valid=valid,
        q_true_px=q_true,
        target_mode=ORACLE_CEILING_TARGET_MODE,
    )
    metadata = pixel_observer_v2_checkpoint_metadata(
        _adapter(), target_mode=ORACLE_CEILING_TARGET_MODE
    )

    assert supervision.diagnostic_only is True
    assert supervision.deployment_allowed is False
    contract = metadata["supervision_contract"]
    assert contract["diagnostic_only"] is True
    assert contract["deployment_allowed"] is False
    assert contract["oracle_target_used"] is True


def test_formal_deployment_uses_independent_a_b_mutual_supervision() -> None:
    q_nominal = torch.zeros(1, 3, 2)
    valid = torch.ones(1, 3, dtype=torch.bool)
    paired = build_pixel_observer_v2_paired_supervision(
        q_nominal_px=q_nominal,
        nominal_valid=valid,
        q_corrected_a_px=q_nominal + 1.0,
        q_corrected_b_px=q_nominal + 3.0,
    )
    torch.testing.assert_close(
        paired.target_for_a_history_px, torch.full_like(q_nominal, 3.0)
    )
    torch.testing.assert_close(
        paired.target_for_b_history_px, torch.full_like(q_nominal, 1.0)
    )
    torch.testing.assert_close(
        paired.consensus_target_px, torch.full_like(q_nominal, 2.0)
    )
    loss = paired_mutual_huber_pixel_loss(
        torch.zeros_like(q_nominal),
        torch.zeros_like(q_nominal),
        paired,
        delta_px=0.5,
    )
    # Huber(3)=1.375, Huber(1)=0.375, averaged across the two directions.
    torch.testing.assert_close(loss, torch.tensor(0.875))
    metadata = pixel_observer_v2_checkpoint_metadata(
        _adapter(), target_mode=DEPLOYMENT_TARGET_MODE
    )
    contract = metadata["supervision_contract"]
    assert contract["deployment_allowed"] is True
    assert contract["target_contract"] == "independent_a_b_mutual_supervision"


def test_single_measurement_is_non_deployable_noisy_baseline() -> None:
    q_nominal = torch.zeros(1, 3, 2)
    valid = torch.ones(1, 3, dtype=torch.bool)
    supervision = build_pixel_observer_v2_supervision(
        q_nominal_px=q_nominal,
        nominal_valid=valid,
        q_corrected_px=q_nominal + 1.0,
        target_mode=SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
    )
    metadata = pixel_observer_v2_checkpoint_metadata(
        _adapter(), target_mode=SINGLE_MEASUREMENT_BASELINE_TARGET_MODE
    )
    assert supervision.diagnostic_only is True
    assert supervision.deployment_allowed is False
    assert metadata["supervision_contract"]["deployment_allowed"] is False


def test_single_supervision_rejects_q_true_and_oracle_rejects_correction() -> None:
    q_nominal = torch.zeros(1, 3, 2)
    valid = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="must not receive q_true"):
        build_pixel_observer_v2_supervision(
            q_nominal_px=q_nominal,
            nominal_valid=valid,
            q_corrected_px=q_nominal,
            q_true_px=q_nominal,
            target_mode=SINGLE_MEASUREMENT_BASELINE_TARGET_MODE,
        )
    with pytest.raises(ValueError, match="isolate q_true"):
        build_pixel_observer_v2_supervision(
            q_nominal_px=q_nominal,
            nominal_valid=valid,
            q_corrected_px=q_nominal,
            q_true_px=q_nominal,
            target_mode=ORACLE_CEILING_TARGET_MODE,
        )


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        ("predicted_embedding", torch.zeros(2, 4), "predicted_embedding"),
        ("current_q_nominal_px", torch.zeros(2, 2, 2), "current_q_nominal_px"),
        ("current_keypoints_xyz", torch.full((2, 3, 3), float("nan")), "finite"),
        ("current_nominal_valid", torch.ones(2, 3), "boolean"),
    ],
)
def test_prior_rejects_bad_shapes_and_nonfinite_inputs(
    name: str, replacement: torch.Tensor, message: str
) -> None:
    inputs = _inputs()
    inputs[name] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        _adapter().predict_prior(inputs)


def test_prior_rejects_mixed_floating_dtype() -> None:
    inputs = _inputs()
    inputs["current_q_nominal_px"] = inputs["current_q_nominal_px"].double()
    with pytest.raises(TypeError, match="dtype"):
        _adapter().predict_prior(inputs)


def test_velocity_innovation_is_bounded_and_outputs_are_finite() -> None:
    adapter = _adapter()
    with torch.no_grad():
        adapter.net[-1].weight.fill_(1e4)
        adapter.net[-1].bias.fill_(1e4)
    output = adapter.predict_prior(_inputs())
    assert bool(torch.isfinite(output.residual_px).all())
    assert bool(torch.isfinite(output.velocity_px).all())
    assert bool(torch.isfinite(output.velocity_innovation_px).all())
    assert float(output.velocity_innovation_px.abs().max()) <= adapter.max_velocity_innovation_px
