from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.lewm_action_prefix_joint_a4 import (
    ANCHOR_COUNT,
    HORIZON,
    LATENT_DIM,
    STATE_SLOTS,
    AnchorMetricReadout,
)
from fcpg.lewm_metric3d_joint_a4 import (
    CameraAnchor3DReadout,
    Metric3DCameraActionLeWM,
    camera_to_world_points_torch,
    command_integrated_world_xyz,
    world_to_camera_points_torch,
)
from scripts.evaluate_lewm_metric3d_hybrid_a4 import optimal_blend_weight


def test_world_camera_point_transform_round_trip() -> None:
    points = torch.tensor(
        [[[[0.2, -0.1, 0.8], [0.4, 0.2, 1.1]]]], dtype=torch.float32
    )
    angle = torch.deg2rad(torch.tensor(35.0))
    cosine, sine = torch.cos(angle), torch.sin(angle)
    extrinsic = torch.eye(4)[None]
    extrinsic[0, :3, :3] = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    extrinsic[0, :3, 3] = torch.tensor([0.1, -0.2, 0.3])

    camera = world_to_camera_points_torch(points, extrinsic)
    restored = camera_to_world_points_torch(camera, extrinsic)

    torch.testing.assert_close(restored, points)


def test_command_integrated_world_xyz_uses_all_raw_actions() -> None:
    current = torch.zeros(1, ANCHOR_COUNT, 3)
    action = torch.zeros(1, HORIZON, 20)
    raw = action.reshape(1, HORIZON, 5, 4)
    raw[..., 0] = 0.2
    raw[..., 1] = -0.1

    result = command_integrated_world_xyz(current, action)

    expected_x = 0.01 * torch.arange(1, HORIZON + 1)
    expected_y = -0.005 * torch.arange(1, HORIZON + 1)
    np.testing.assert_allclose(result[0, :, 0, 0], expected_x)
    np.testing.assert_allclose(result[0, :, 2, 1], expected_y)


def test_camera_3d_readout_starts_at_command_reference() -> None:
    readout = CameraAnchor3DReadout(hidden_dim=32)
    readout.initialize_embeddings_from(AnchorMetricReadout())
    predicted = torch.randn(2, HORIZON, STATE_SLOTS, LATENT_DIM)
    current = torch.randn(2, STATE_SLOTS, LATENT_DIM)
    current_camera = torch.randn(2, ANCHOR_COUNT, 3)
    reference = torch.randn(2, HORIZON, ANCHOR_COUNT, 3)

    output, residual = readout(
        predicted_state=predicted,
        current_state=current,
        current_camera_xyz=current_camera,
        reference_camera_xyz=reference,
        task_index=torch.tensor([0, 6]),
    )

    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_world_3d_readout_starts_at_known_world_reference() -> None:
    readout = CameraAnchor3DReadout(hidden_dim=32)
    readout.initialize_embeddings_from(AnchorMetricReadout())
    predicted = torch.randn(2, HORIZON, STATE_SLOTS, LATENT_DIM)
    current = torch.randn(2, STATE_SLOTS, LATENT_DIM)
    current_world = torch.randn(2, ANCHOR_COUNT, 3)
    reference_world = torch.randn(2, HORIZON, ANCHOR_COUNT, 3)

    output, residual = readout(
        predicted_state=predicted,
        current_state=current,
        current_world_xyz=current_world,
        reference_world_xyz=reference_world,
        task_index=torch.tensor([0, 6]),
    )

    torch.testing.assert_close(output, reference_world)
    torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_camera_3d_readout_residual_is_bounded() -> None:
    readout = CameraAnchor3DReadout(hidden_dim=16, max_residual_m=0.12)
    with torch.no_grad():
        final = readout.head[-1]
        assert isinstance(final, nn.Linear)
        final.bias.fill_(100.0)
    predicted = torch.randn(1, HORIZON, STATE_SLOTS, LATENT_DIM)
    current = torch.randn(1, STATE_SLOTS, LATENT_DIM)
    reference = torch.zeros(1, HORIZON, ANCHOR_COUNT, 3)

    _, residual = readout(
        predicted_state=predicted,
        current_state=current,
        current_camera_xyz=torch.zeros(1, ANCHOR_COUNT, 3),
        reference_camera_xyz=reference,
        task_index=torch.tensor([1]),
    )

    assert float(residual.abs().max()) <= 0.120001


def test_point_transform_rejects_bad_extrinsic_shape() -> None:
    with pytest.raises(ValueError, match="extrinsic"):
        world_to_camera_points_torch(
            torch.zeros(2, 3, 3), torch.eye(4)[None]
        )


def test_metric3d_wrapper_forwards_observation_encoding() -> None:
    class StubCameraAction(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frame_encoder = nn.Identity()
            self.state_projector = nn.Identity()
            self.dynamics = nn.Identity()
            self.metric_readout = AnchorMetricReadout()

        def encode_observations(self, **kwargs: torch.Tensor) -> torch.Tensor:
            return kwargs["rgb"] + 3.0

    stub = StubCameraAction()
    wrapper = object.__new__(Metric3DCameraActionLeWM)
    nn.Module.__init__(wrapper)
    wrapper.camera_action_model = stub
    wrapper.metric3d_readout = CameraAnchor3DReadout(hidden_dim=16)

    rgb = torch.ones(2, 1, 4)
    result = wrapper.encode_observations(rgb=rgb)

    torch.testing.assert_close(result, rgb + 3.0)


def test_optimal_blend_weight_is_bounded_and_handles_degenerate_input() -> None:
    assert optimal_blend_weight(3.0, 4.0) == pytest.approx(0.75)
    assert optimal_blend_weight(-2.0, 4.0) == 0.0
    assert optimal_blend_weight(8.0, 4.0) == 1.0
    assert optimal_blend_weight(1.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="finite"):
        optimal_blend_weight(float("nan"), 1.0)
