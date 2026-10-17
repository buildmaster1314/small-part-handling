from __future__ import annotations

import numpy as np

from scripts.probe_lewm_world_residual import (
    camera_to_world_numpy,
    command_integrated_xyz_numpy,
    project_world_numpy,
    ridge_solve,
    world_to_camera_numpy,
)


def test_command_integrated_xyz_numpy_accumulates_five_raw_actions() -> None:
    current = np.zeros((1, 3, 3), dtype=np.float32)
    action = np.zeros((1, 8, 8, 20), dtype=np.float32)
    raw = action.reshape(1, 8, 8, 5, 4)
    raw[..., 0] = 0.2
    prediction = command_integrated_xyz_numpy(current, action)
    np.testing.assert_allclose(prediction[0, 0, 0, :, 0], 0.01)
    np.testing.assert_allclose(prediction[0, 0, 7, :, 0], 0.08)


def test_project_world_numpy_uses_camera_translation() -> None:
    point = np.asarray([[[[[1.0, 2.0, 4.0]]]]], dtype=np.float32)
    intrinsic = np.asarray([[[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]]])
    extrinsic = np.eye(4, dtype=np.float32)[None]
    extrinsic[0, 0, 3] = 1.0
    pixel, valid = project_world_numpy(point, intrinsic, extrinsic)
    np.testing.assert_allclose(pixel.reshape(2), [11.0, 21.0])
    assert bool(valid.item())


def test_ridge_solve_recovers_linear_mapping() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(500, 4))
    design = np.concatenate([values, np.ones((values.shape[0], 1))], axis=1)
    expected = np.asarray(
        [
            [1.0, -0.5, 0.25],
            [0.2, 0.3, -0.1],
            [-0.4, 0.0, 0.7],
            [0.8, -0.2, 0.1],
            [0.5, 0.4, -0.3],
        ]
    )
    target = design @ expected
    fitted = ridge_solve(
        design.T @ design,
        design.T @ target,
        sample_count=design.shape[0],
        alpha=0.0,
    )
    np.testing.assert_allclose(fitted, expected, atol=1.0e-10)


def test_world_camera_transform_round_trip() -> None:
    points = np.asarray(
        [[[[[0.2, -0.1, 0.8], [0.3, 0.4, 0.9]]]]], dtype=np.float32
    )
    angle = np.deg2rad(35.0)
    cosine, sine = np.cos(angle), np.sin(angle)
    extrinsic = np.eye(4, dtype=np.float32)[None]
    extrinsic[0, :3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    extrinsic[0, :3, 3] = [0.1, -0.2, 0.3]
    camera = world_to_camera_numpy(points, extrinsic)
    restored = camera_to_world_numpy(camera, extrinsic)
    np.testing.assert_allclose(restored, points, atol=1.0e-6)
