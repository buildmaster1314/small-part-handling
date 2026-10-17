from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_action_prefix_joint_a4 import (
    ANCHOR_COUNT,
    HORIZON,
    LATENT_DIM,
    STATE_SLOTS,
    AnchorMetricReadout,
)
from fcpg.lewm_world_anchor_projection import (
    WORLD_ANCHOR_CHECKPOINT_TYPE,
    WorldAnchorDynamicsReadout,
    load_world_anchor_checkpoint,
    project_world_points_torch,
    world_anchor_checkpoint,
)
from scripts.train_lewm_world_anchor_readout import (
    command_integrated_xyz,
    multiview_training_groups,
)


def _inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    return {
        "predicted_state": torch.randn(
            batch_size,
            HORIZON,
            STATE_SLOTS,
            LATENT_DIM,
            generator=generator,
        ),
        "current_state": torch.randn(
            batch_size,
            STATE_SLOTS,
            LATENT_DIM,
            generator=generator,
        ),
        "current_keypoints_xyz": torch.randn(
            batch_size, ANCHOR_COUNT, 3, generator=generator
        ),
        "camera_rotation_world_to_camera": torch.eye(3).repeat(
            batch_size, 1, 1
        ),
        "task_index": torch.arange(batch_size) % 7,
    }


def test_projection_matches_known_pinhole_geometry_and_preserves_axes() -> None:
    points = torch.tensor(
        [[[[0.0, 0.0, 2.0], [1.0, 2.0, 4.0]]]], dtype=torch.float32
    )
    intrinsic = torch.tensor(
        [[[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]]
    )
    extrinsic = torch.eye(4)[None]
    output = project_world_points_torch(points, intrinsic, extrinsic)
    expected = torch.tensor([[[[50.0, 60.0], [75.0, 120.0]]]])
    torch.testing.assert_close(output.pixels, expected)
    torch.testing.assert_close(output.depth, torch.tensor([[[2.0, 4.0]]]))
    assert bool(output.valid_depth.all())


def test_projection_rejects_camera_shape_and_nonfinite_values() -> None:
    points = torch.zeros(2, 8, 3, 3)
    with pytest.raises(ValueError, match="intrinsic"):
        project_world_points_torch(points, torch.eye(3)[None], torch.eye(4)[None].repeat(2, 1, 1))
    intrinsic = torch.eye(3)[None].repeat(2, 1, 1)
    intrinsic[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        project_world_points_torch(points, intrinsic, torch.eye(4)[None].repeat(2, 1, 1))


def test_world_anchor_readout_starts_at_hold_current() -> None:
    model = WorldAnchorDynamicsReadout()
    model.initialize_embeddings_from(AnchorMetricReadout())
    values = _inputs()
    output = model(**values)
    expected = values["current_keypoints_xyz"][:, None].expand(
        -1, HORIZON, -1, -1
    )
    torch.testing.assert_close(output.predicted_xyz_world, expected)
    torch.testing.assert_close(output.delta_xyz_world, torch.zeros_like(output.delta_xyz_world))


def test_world_anchor_readout_starts_at_explicit_command_reference() -> None:
    model = WorldAnchorDynamicsReadout()
    values = _inputs()
    reference = values["current_keypoints_xyz"][:, None].expand(
        -1, HORIZON, -1, -1
    ).clone()
    reference[..., 0] += 0.1
    output = model(**values, reference_keypoints_xyz=reference)
    torch.testing.assert_close(output.predicted_xyz_world, reference)


def test_world_anchor_checkpoint_round_trip() -> None:
    model = WorldAnchorDynamicsReadout(hidden_dim=64, max_delta_m=0.4)
    checkpoint = world_anchor_checkpoint(
        model,
        step=2000,
        seed=40561,
        base_checkpoint="base.pt",
        raw_dataset="raw",
    )
    assert checkpoint["checkpoint_type"] == WORLD_ANCHOR_CHECKPOINT_TYPE
    loaded = load_world_anchor_checkpoint(checkpoint)
    values = _inputs()
    torch.testing.assert_close(
        loaded(**values).predicted_xyz_world,
        model(**values).predicted_xyz_world,
    )


def test_command_integration_uses_five_raw_actions_per_model_step() -> None:
    current = torch.zeros(1, ANCHOR_COUNT, 3)
    action = torch.zeros(1, HORIZON, 20)
    raw = action.reshape(1, HORIZON, 5, 4)
    raw[:, :, :, 0] = 1.0
    raw[:, :, :, 1] = -0.5
    raw[:, :, :, 3] = 1.0
    predicted = command_integrated_xyz(current, action)
    expected_x = 0.05 * torch.arange(1, HORIZON + 1, dtype=torch.float32)
    expected_y = -0.025 * torch.arange(1, HORIZON + 1, dtype=torch.float32)
    np.testing.assert_allclose(predicted[0, :, 0, 0].numpy(), expected_x.numpy())
    np.testing.assert_allclose(predicted[0, :, 2, 1].numpy(), expected_y.numpy())
    np.testing.assert_allclose(predicted[0, :, :, 2].numpy(), 0.0)


def test_multiview_training_groups_preserve_physical_camera_panels() -> None:
    index = {
        "split": np.asarray(["train"] * 12 + ["development"] * 6),
        "physical_bundle_index": np.asarray([0] * 6 + [1] * 6 + [2] * 6),
    }
    groups = multiview_training_groups(index)
    np.testing.assert_array_equal(groups, np.arange(12).reshape(2, 6))
    broken = dict(index)
    broken["split"] = np.asarray(["train"] * 11 + ["development"] * 7)
    with pytest.raises(ValueError, match="six views"):
        multiview_training_groups(broken)
