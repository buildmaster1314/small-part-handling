from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.lewm_action_prefix import ActionPrefixDynamics
from fcpg.lewm_mobile_base import (
    MOBILE_PREFIX_DIM,
    PlanarBaseTracker,
    expand_action_prefix_dynamics_for_mobile_base,
    pack_mobile_action_prefix,
    pack_mobile_action_prefix_tensor,
    replace_base_displacement_with_odometry,
    split_mobile_action_prefix,
)


def test_tracker_separates_true_slip_from_odometry_error() -> None:
    tracker = PlanarBaseTracker(initial_true_xy=(1.0, 2.0))
    first = tracker.step(
        (0.1, -0.2),
        slip_xy=(0.02, 0.01),
        odometry_error_xy=(-0.01, 0.03),
    )
    np.testing.assert_allclose(first.true_delta_xy, [0.12, -0.19])
    np.testing.assert_allclose(first.odometry_delta_xy, [0.09, -0.17])
    np.testing.assert_allclose(first.true_position_xy, [1.12, 1.81])
    np.testing.assert_allclose(first.odometry_position_xy, [0.09, -0.17])

    second = tracker.step((0.01, 0.02))
    np.testing.assert_allclose(second.true_position_xy, [1.13, 1.83])
    np.testing.assert_allclose(second.odometry_position_xy, [0.10, -0.15])


def test_mobile_prefix_round_trip_preserves_raw_step_alignment() -> None:
    arm = np.arange(2 * 8 * 20, dtype=np.float32).reshape(2, 8, 20)
    base = (1000 + np.arange(2 * 8 * 10, dtype=np.float32)).reshape(2, 8, 10)
    packed = pack_mobile_action_prefix(arm, base)
    assert packed.shape == (2, 8, MOBILE_PREFIX_DIM)
    np.testing.assert_array_equal(packed[0, 0, :6], [0, 1, 2, 3, 1000, 1001])
    restored_arm, restored_base = split_mobile_action_prefix(packed)
    np.testing.assert_array_equal(restored_arm, arm)
    np.testing.assert_array_equal(restored_base, base)


def test_odometry_replacement_keeps_arm_relative_geometry() -> None:
    true = np.asarray(
        [[1.2, 2.1, 0.3], [1.3, 1.9, 0.4], [1.1, 2.0, 0.5]],
        dtype=np.float64,
    )
    estimated = replace_base_displacement_with_odometry(
        true,
        true_base_displacement_xy=(0.2, 0.1),
        odometry_base_displacement_xy=(0.15, 0.12),
    )
    np.testing.assert_allclose(
        estimated,
        [[1.15, 2.12, 0.3], [1.25, 1.92, 0.4], [1.05, 2.02, 0.5]],
    )
    np.testing.assert_allclose(np.diff(estimated, axis=0), np.diff(true, axis=0))


@pytest.mark.parametrize(
    ("arm_shape", "base_shape", "message"),
    [
        ((2, 19), (2, 10), "arm_prefix"),
        ((2, 20), (2, 9), "base_prefix"),
    ],
)
def test_mobile_prefix_rejects_contract_drift(
    arm_shape: tuple[int, ...],
    base_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pack_mobile_action_prefix(
            np.zeros(arm_shape, dtype=np.float32),
            np.zeros(base_shape, dtype=np.float32),
        )


def test_mobile_prefix_rejects_non_finite_values() -> None:
    arm = np.zeros((1, 20), dtype=np.float32)
    base = np.zeros((1, 10), dtype=np.float32)
    base[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pack_mobile_action_prefix(arm, base)


def test_tensor_pack_matches_numpy_pack() -> None:
    arm = np.arange(3 * 20, dtype=np.float32).reshape(3, 20)
    base = (100 + np.arange(3 * 10, dtype=np.float32)).reshape(3, 10)
    actual = pack_mobile_action_prefix_tensor(
        torch.from_numpy(arm), torch.from_numpy(base)
    )
    np.testing.assert_array_equal(actual.numpy(), pack_mobile_action_prefix(arm, base))


def test_mobile_expansion_preserves_original_dynamics_before_training() -> None:
    torch.manual_seed(17)
    base = ActionPrefixDynamics(
        action_dim=20,
        state_dim=24,
        num_slots=4,
        num_tasks=2,
        max_horizon=3,
        prefix_depth=1,
        prefix_heads=4,
        predictor_depth=2,
        dropout=0.0,
        prefix_mlp_dim=48,
        predictor_hidden_dim=32,
        fusion_dim=40,
        state_mlp_dim=36,
    ).eval()
    mean = torch.linspace(-0.2, 0.2, 20)
    std = torch.linspace(0.5, 1.5, 20)
    base.set_action_normalization(mean, std)
    expanded = expand_action_prefix_dynamics_for_mobile_base(
        base,
        base_action_std=np.full(10, 0.001, dtype=np.float32),
    ).eval()
    state = torch.randn(5, 4, 24)
    arm = torch.randn(5, 3, 20)
    moving = torch.randn(5, 3, 10) * 0.001
    packed = pack_mobile_action_prefix_tensor(arm, moving)
    torch.testing.assert_close(
        expanded(state, packed, task_index=torch.tensor([0, 1, 0, 1, 0])),
        base(state, arm, task_index=torch.tensor([0, 1, 0, 1, 0])),
        rtol=0.0,
        atol=1.0e-6,
    )
    raw_weight = expanded.action_projection[0].weight
    base_columns = [index for step in range(5) for index in (step * 6 + 4, step * 6 + 5)]
    torch.testing.assert_close(raw_weight[:, base_columns], torch.zeros_like(raw_weight[:, base_columns]))
