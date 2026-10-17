import numpy as np
import pytest

from fcpg.so101_real_image_drift import (
    SO101ImageAffineDrift,
    points_in_bounds,
    sample_bounded_drift,
    transform_points,
    warp_rgb,
)


def test_translation_transforms_pixels_and_rgb_consistently() -> None:
    drift = SO101ImageAffineDrift(translation_x_px=5.0, translation_y_px=-3.0)
    matrix = drift.matrix(32, 24)
    point = np.asarray([[10.0, 12.0]], dtype=np.float32)
    np.testing.assert_allclose(transform_points(point, matrix), [[15.0, 9.0]])
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[12, 10] = 255
    warped = warp_rgb(image, matrix)
    peak_y, peak_x = np.unravel_index(np.argmax(warped[..., 0]), warped.shape[:2])
    assert (peak_x, peak_y) == (15, 9)


def test_rotation_is_about_image_center() -> None:
    drift = SO101ImageAffineDrift(rotation_degrees=90.0)
    center = np.asarray([[(21 - 1) / 2, (21 - 1) / 2]], dtype=np.float32)
    np.testing.assert_allclose(
        transform_points(center, drift.matrix(21, 21)), center, atol=1.0e-6
    )


def test_sampled_drift_obeys_support_bounds() -> None:
    points = (
        np.asarray([[80, 90], [100, 90], [90, 110]], dtype=np.float32),
        np.asarray([[120, 80], [140, 80], [130, 100]], dtype=np.float32),
    )
    valid = (np.ones(3, dtype=bool), np.ones(3, dtype=bool))
    drift = sample_bounded_drift(
        np.random.default_rng(7),
        points,
        valid,
        width=224,
        height=224,
        max_rotation_degrees=4.0,
        max_translation_px=8.0,
        min_mean_residual_px=3.0,
        max_residual_px=18.0,
    )
    residuals = []
    for value in points:
        transformed = transform_points(value, drift.matrix(224, 224))
        assert np.all(points_in_bounds(transformed, 224, 224))
        residuals.extend(np.linalg.norm(transformed - value, axis=-1))
    assert np.mean(residuals) >= 3.0
    assert np.max(residuals) <= 18.0


def test_invalid_drift_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="max residual"):
        sample_bounded_drift(
            np.random.default_rng(1),
            (np.zeros((3, 2), dtype=np.float32),),
            (np.ones(3, dtype=bool),),
            width=32,
            height=32,
            max_rotation_degrees=1.0,
            max_translation_px=1.0,
            min_mean_residual_px=2.0,
            max_residual_px=1.0,
        )
