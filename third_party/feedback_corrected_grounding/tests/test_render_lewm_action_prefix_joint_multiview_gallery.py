from __future__ import annotations

import numpy as np

from scripts.render_lewm_action_prefix_joint_multiview_gallery import (
    CAMERAS,
    _aperture,
    _camera_panel,
    _select_physical_bundle,
    _upright,
)


def test_aperture_uses_left_and_right_fingertips() -> None:
    points = np.asarray([[1.0, 1.0], [4.0, 5.0], [2.0, 3.0]], dtype=np.float32)
    assert float(_aperture(points)) == 5.0


def test_select_physical_bundle_prefers_larger_gripper_change() -> None:
    camera_names = np.asarray(CAMERAS * 2)
    index = {
        "task_name": np.asarray(["pick-place-v3"] * 10),
        "split": np.asarray(["development"] * 10),
        "physical_bundle_index": np.asarray([0] * 5 + [1] * 5),
        "camera_name": camera_names,
    }
    future = np.zeros((10, 8, 8, 3, 2), dtype=np.float32)
    future[:5, 7, :, 1, 0] = np.linspace(1.0, 2.0, 8)
    future[5:, 7, :, 1, 0] = np.linspace(1.0, 9.0, 8)
    physical, rows = _select_physical_bundle(
        index,
        {"future_q_true": future},
        task="pick-place-v3",
        branch=7,
    )
    assert physical == 1
    assert rows.tolist() == [5, 6, 7, 8, 9]


def test_upright_flips_metaworld_images_and_y_coordinates() -> None:
    rgb = np.zeros((3, 224, 224), dtype=np.uint8)
    rgb[:, 0, :] = 255
    points = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    image, transformed = _upright("pick-place-v3", rgb, points)
    assert image[-1].max() == 255
    np.testing.assert_allclose(transformed[:, 1], [221.0, 219.0, 217.0])


def test_upright_can_preserve_orbit_camera_orientation() -> None:
    rgb = np.zeros((3, 224, 224), dtype=np.uint8)
    rgb[:, 0, :] = 255
    points = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    image, transformed = _upright(
        "pick-place-v3",
        rgb,
        points,
        flip_vertical=False,
    )
    assert image[0].max() == 255
    np.testing.assert_allclose(transformed, points)


def test_camera_panel_orders_orbit_views_by_clock_hour() -> None:
    index = {
        "task_name": np.asarray(["pick-place-v3"] * 3),
        "split": np.asarray(["development"] * 3),
        "camera_name": np.asarray(["clock05_e35", "clock01_e35", "clock03_e35"]),
        "camera_clock_hour": np.asarray([5, 1, 3]),
    }
    assert _camera_panel(
        index,
        task="pick-place-v3",
        split="development",
    ) == ("clock01_e35", "clock03_e35", "clock05_e35")
