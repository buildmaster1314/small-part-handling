import numpy as np
import pytest

from fcpg.geometry import project_points
from fcpg.so101_camera_grounding import (
    free_camera_extrinsic_cv,
    free_camera_projection,
)


def test_free_camera_extrinsic_uses_opencv_axis_convention() -> None:
    extrinsic = free_camera_extrinsic_cv(
        np.zeros(3),
        np.asarray([0.0, 0.0, -1.0]),
        np.asarray([0.0, 1.0, 0.0]),
    )

    np.testing.assert_allclose(
        extrinsic,
        np.diag([1.0, -1.0, -1.0, 1.0]),
        atol=1.0e-8,
    )


def test_free_camera_projection_places_forward_point_at_image_center() -> None:
    camera = free_camera_projection(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0.0, 0.0, -1.0]),
        np.asarray([0.0, 1.0, 0.0]),
        width=640,
        height=480,
        fovy_degrees=45.0,
    )

    pixels, valid = project_points(np.asarray([[1.0, 2.0, 2.0]]), camera)

    np.testing.assert_allclose(pixels[0], [320.0, 240.0], atol=1.0e-7)
    assert valid.tolist() == [True]


def test_free_camera_extrinsic_rejects_collinear_axes() -> None:
    with pytest.raises(ValueError, match="collinear"):
        free_camera_extrinsic_cv(np.zeros(3), np.ones(3), np.ones(3))
