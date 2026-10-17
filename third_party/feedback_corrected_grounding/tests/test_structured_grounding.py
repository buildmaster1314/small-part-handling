"""Tests for Stage-II structured kinematic projection geometry."""

from __future__ import annotations

import math

import pytest
import torch

from fcpg.structured_grounding import (
    StructuredKinematicProjection,
    chi_square_rms_threshold,
    projection_jacobian_left_se3,
    solve_observer_control_transfer,
    solve_whitened_observer_update,
    project_points_torch,
    se3_exp,
)


def test_se3_exp_identity_and_translation() -> None:
    identity = se3_exp(torch.zeros(2, 6))
    expected = torch.eye(4).expand(2, 4, 4)
    torch.testing.assert_close(identity, expected)

    twist = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0]])
    translated = se3_exp(twist)
    torch.testing.assert_close(translated[0, :3, 3], twist[0, :3])


def test_se3_exp_recovers_axis_angle_rotation() -> None:
    twist = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2]])
    transform = se3_exp(twist)
    expected = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(transform[0, :3, :3], expected, atol=1e-6, rtol=1e-6)


def test_structured_projection_separates_global_and_local_correction() -> None:
    intrinsic = torch.tensor(
        [
            [100.0, 0.0, 32.0],
            [0.0, 100.0, 32.0],
            [0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    nominal = torch.eye(4).unsqueeze(0)
    points = torch.tensor(
        [
            [
                [-0.2, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.2, 0.0, 2.0],
            ]
        ]
    )
    twist = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    local = torch.tensor([[[1.0, -2.0], [0.5, 0.0], [-1.0, 2.0]]])

    output = StructuredKinematicProjection()(intrinsic, nominal, points, twist, local)
    direct, depth = project_points_torch(intrinsic, se3_exp(twist) @ nominal, points)

    torch.testing.assert_close(output.global_points_px, direct)
    torch.testing.assert_close(output.corrected_points_px, direct + local)
    torch.testing.assert_close(output.depth, depth)
    assert output.global_transform.shape == (1, 4, 4)


def test_structured_projection_backpropagates_to_pose_and_local_residual() -> None:
    intrinsic = torch.eye(3).unsqueeze(0)
    nominal = torch.eye(4).unsqueeze(0)
    points = torch.tensor([[[0.2, 0.1, 2.0], [-0.1, 0.3, 2.5]]])
    twist = torch.zeros(1, 6, requires_grad=True)
    local = torch.zeros(1, 2, 2, requires_grad=True)

    output = StructuredKinematicProjection()(intrinsic, nominal, points, twist, local)
    loss = output.corrected_points_px.square().mean()
    loss.backward()

    assert twist.grad is not None
    assert bool(torch.isfinite(twist.grad).all())
    assert local.grad is not None
    assert bool(torch.isfinite(local.grad).all())


def test_left_se3_projection_jacobian_matches_central_difference() -> None:
    dtype = torch.float64
    intrinsic = torch.tensor(
        [
            [180.0, 2.0, 112.0],
            [0.0, 175.0, 108.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    ).unsqueeze(0)
    nominal = torch.eye(4, dtype=dtype).unsqueeze(0)
    points = torch.tensor(
        [[[-0.4, -0.2, 2.0], [0.3, -0.1, 2.4], [0.2, 0.5, 3.0]]],
        dtype=dtype,
    )
    pixels, depth, jacobian = projection_jacobian_left_se3(
        intrinsic,
        nominal,
        points,
    )
    assert bool((depth > 0).all())

    epsilon = 1e-7
    for dimension in range(6):
        delta = torch.zeros(1, 6, dtype=dtype)
        delta[0, dimension] = epsilon
        plus, _ = project_points_torch(
            intrinsic,
            se3_exp(delta) @ nominal,
            points,
        )
        minus, _ = project_points_torch(
            intrinsic,
            se3_exp(-delta) @ nominal,
            points,
        )
        numerical = (plus - minus) / (2.0 * epsilon)
        torch.testing.assert_close(
            jacobian[..., dimension],
            numerical,
            atol=2e-5,
            rtol=2e-5,
        )
    torch.testing.assert_close(
        pixels,
        project_points_torch(intrinsic, nominal, points)[0],
    )


def test_oracle_observers_transfer_to_held_out_control_anchors() -> None:
    dtype = torch.float64
    intrinsic = torch.tensor(
        [
            [220.0, 0.0, 112.0],
            [0.0, 220.0, 112.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    ).unsqueeze(0)
    nominal = torch.eye(4, dtype=dtype).unsqueeze(0)
    observers = torch.tensor(
        [
            [
                [-0.60, -0.45, 2.00],
                [0.55, -0.40, 2.20],
                [-0.50, 0.50, 2.45],
                [0.60, 0.45, 2.70],
                [-0.15, -0.65, 3.00],
                [0.25, 0.65, 3.20],
                [-0.35, 0.15, 1.80],
                [0.40, -0.10, 2.90],
            ]
        ],
        dtype=dtype,
    )
    controls = torch.tensor(
        [[[0.08, 0.02, 1.70], [0.20, 0.04, 1.75], [-0.04, 0.05, 1.72]]],
        dtype=dtype,
    )
    true_twist = torch.tensor(
        [
            [
                0.025,
                -0.015,
                0.010,
                math.radians(3.0),
                math.radians(-4.0),
                math.radians(8.0),
            ]
        ],
        dtype=dtype,
    )
    true_extrinsic = se3_exp(true_twist) @ nominal
    observer_pixels, _ = project_points_torch(
        intrinsic,
        true_extrinsic,
        observers,
    )
    control_true, _ = project_points_torch(
        intrinsic,
        true_extrinsic,
        controls,
    )

    output = solve_observer_control_transfer(
        intrinsic,
        nominal,
        observers,
        observer_pixels,
        controls,
        prior_std=torch.tensor(
            [
                0.10,
                0.10,
                0.10,
                math.radians(15.0),
                math.radians(15.0),
                math.radians(15.0),
            ],
            dtype=dtype,
        ),
        observation_std_px=0.25,
        min_whitened_singular_value=0.05,
        relative_singular_value=1e-5,
    )

    before = torch.linalg.vector_norm(
        output.control_nominal_points_px - control_true,
        dim=-1,
    ).mean()
    after = torch.linalg.vector_norm(
        output.control_corrected_points_px - control_true,
        dim=-1,
    ).mean()
    assert float(after / before) <= 0.25
    assert int(output.update.effective_rank.item()) >= 5
    assert float(output.update.posterior_control_variance.item()) < float(
        output.update.prior_control_variance.item()
    )


def test_no_valid_observers_produces_zero_shared_update() -> None:
    dtype = torch.float64
    intrinsic = torch.tensor(
        [[[150.0, 0.0, 64.0], [0.0, 150.0, 64.0], [0.0, 0.0, 1.0]]],
        dtype=dtype,
    )
    nominal = torch.eye(4, dtype=dtype).unsqueeze(0)
    observers = torch.tensor(
        [[[-0.2, 0.0, 2.0], [0.2, 0.0, 2.0]]],
        dtype=dtype,
    )
    controls = torch.tensor([[[0.0, 0.2, 2.0]]], dtype=dtype)
    observed, _ = project_points_torch(intrinsic, nominal, observers)

    output = solve_observer_control_transfer(
        intrinsic,
        nominal,
        observers,
        observed + 10.0,
        controls,
        prior_std=torch.ones(6, dtype=dtype),
        observation_std_px=1.0,
        observer_valid=torch.zeros(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(output.update.twist, torch.zeros_like(output.update.twist))
    torch.testing.assert_close(
        output.control_corrected_points_px,
        output.control_nominal_points_px,
    )
    assert int(output.update.effective_rank.item()) == 0


def test_control_utility_rejects_observable_but_irrelevant_mode() -> None:
    observer_jacobian = torch.zeros(1, 1, 2, 6, dtype=torch.float64)
    observer_jacobian[0, 0, 0, 0] = 10.0
    innovation = torch.tensor([[[2.0, 0.0]]], dtype=torch.float64)
    control_jacobian = torch.zeros(1, 1, 2, 6, dtype=torch.float64)
    control_jacobian[0, 0, 0, 1] = 1.0

    unfiltered = solve_whitened_observer_update(
        observer_jacobian,
        innovation,
        control_jacobian,
        prior_std=torch.ones(6, dtype=torch.float64),
        observation_std_px=1.0,
        min_whitened_singular_value=0.1,
        relative_singular_value=0.0,
        min_control_variance_reduction_fraction=0.0,
    )
    filtered = solve_whitened_observer_update(
        observer_jacobian,
        innovation,
        control_jacobian,
        prior_std=torch.ones(6, dtype=torch.float64),
        observation_std_px=1.0,
        min_whitened_singular_value=0.1,
        relative_singular_value=0.0,
        min_control_variance_reduction_fraction=1e-3,
    )

    assert int(unfiltered.effective_rank.item()) == 1
    assert int(filtered.effective_rank.item()) == 0
    torch.testing.assert_close(filtered.twist, torch.zeros_like(filtered.twist))


def test_innovation_gate_rejects_noise_sized_modal_update() -> None:
    observer_jacobian = torch.zeros(1, 1, 2, 6, dtype=torch.float64)
    observer_jacobian[0, 0, 0, 0] = 10.0
    control_jacobian = observer_jacobian.clone()

    rejected = solve_whitened_observer_update(
        observer_jacobian,
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        control_jacobian,
        prior_std=torch.ones(6, dtype=torch.float64),
        observation_std_px=1.0,
        min_whitened_singular_value=0.1,
        relative_singular_value=0.0,
        min_control_variance_reduction_fraction=0.0,
        min_innovation_z_score=2.5,
    )
    accepted = solve_whitened_observer_update(
        observer_jacobian,
        torch.tensor([[[4.0, 0.0]]], dtype=torch.float64),
        control_jacobian,
        prior_std=torch.ones(6, dtype=torch.float64),
        observation_std_px=1.0,
        min_whitened_singular_value=0.1,
        relative_singular_value=0.0,
        min_control_variance_reduction_fraction=0.0,
        min_innovation_z_score=2.5,
    )

    assert int(rejected.effective_rank.item()) == 0
    assert int(accepted.effective_rank.item()) == 1
    assert abs(float(rejected.modal_innovation_z_score[0, 0])) == pytest.approx(1.0)
    assert abs(float(accepted.modal_innovation_z_score[0, 0])) == pytest.approx(4.0)


def test_global_innovation_gate_preserves_nominal_under_noise() -> None:
    observer_jacobian = torch.zeros(1, 1, 2, 6, dtype=torch.float64)
    observer_jacobian[0, 0, 0, 0] = 10.0
    control_jacobian = observer_jacobian.clone()

    rejected = solve_whitened_observer_update(
        observer_jacobian,
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        control_jacobian,
        prior_std=torch.ones(6, dtype=torch.float64),
        observation_std_px=1.0,
        min_whitened_singular_value=0.1,
        relative_singular_value=0.0,
        min_control_variance_reduction_fraction=0.0,
        min_global_innovation_rms_z=2.5,
    )
    accepted = solve_whitened_observer_update(
        observer_jacobian,
        torch.tensor([[[4.0, 0.0]]], dtype=torch.float64),
        control_jacobian,
        prior_std=torch.ones(6, dtype=torch.float64),
        observation_std_px=1.0,
        min_whitened_singular_value=0.1,
        relative_singular_value=0.0,
        min_control_variance_reduction_fraction=0.0,
        min_global_innovation_rms_z=2.5,
    )

    assert int(rejected.effective_rank.item()) == 0
    assert int(accepted.effective_rank.item()) == 1
    assert float(rejected.global_innovation_rms_z[0]) == pytest.approx(1.0)
    assert float(accepted.global_innovation_rms_z[0]) == pytest.approx(4.0)


def test_chi_square_rms_threshold_is_rank_adaptive() -> None:
    rank_one = chi_square_rms_threshold(1, 0.01)
    rank_six = chi_square_rms_threshold(6, 0.01)

    assert rank_one > rank_six
    assert rank_one == pytest.approx(2.566, abs=0.02)
    assert rank_six == pytest.approx(1.674, abs=0.02)
