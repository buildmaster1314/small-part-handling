from __future__ import annotations

import numpy as np

from fcpg.so101_camera_calibration import (
    ProjectionFit,
    approximate_column_backprojection_on_plane,
    fit_projection,
    predict_projection,
    projection_features,
    projection_metrics,
    valid_projection_mask,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(10, 3, 3)).astype(np.float64)
    local = rng.normal(size=(10, 3, 3)).astype(np.float64)
    gripper = rng.uniform(0, 100, size=10)
    return base, local, gripper


def test_column_fit_recovers_affine_projection() -> None:
    base, local, gripper = _inputs()
    features, _ = projection_features("column", base, local, gripper)
    coefficient = (
        np.arange(features.shape[1] * 2, dtype=np.float64).reshape(-1, 2) * 0.1
    )
    pixels = (features @ coefficient).reshape(10, 3, 2)
    fit = fit_projection("column", base, local, gripper, pixels)
    predicted = predict_projection(fit, base, local, gripper)
    np.testing.assert_allclose(predicted, pixels, atol=1e-6)


def test_wrist_fit_uses_gripper_opening() -> None:
    base, local, gripper = _inputs()
    features, _ = projection_features("wrist", base, local, gripper)
    coefficient = (
        np.arange(features.shape[1] * 2, dtype=np.float64).reshape(-1, 2) * 0.03
    )
    pixels = (features @ coefficient).reshape(10, 3, 2)
    fit = fit_projection("wrist", base, local, gripper, pixels)
    predicted = predict_projection(fit, base, local, gripper)
    np.testing.assert_allclose(predicted, pixels, atol=1e-6)


def test_projection_metrics_and_valid_mask() -> None:
    predicted = np.array([[[1.0, 2.0], [4.0, 6.0]]])
    observed = np.array([[[1.0, 5.0], [4.0, 2.0]]])
    metrics = projection_metrics(predicted, observed, image_size=(8, 8))
    assert metrics["mean_epe_px"] == 3.5
    mask = valid_projection_mask(predicted, image_size=(8, 8))
    assert mask.tolist() == [[True, True]]
    assert not valid_projection_mask(
        np.array([[[8.0, 1.0]]]), image_size=(8, 8)
    )[0, 0]


def test_column_affine_fit_can_be_inverted_on_known_plane() -> None:
    coefficients = np.asarray(
        [
            [220.0, 18.0],
            [-12.0, 190.0],
            [30.0, -40.0],
            [112.0, 96.0],
        ]
    )
    fit = ProjectionFit(
        view="column",
        feature_names=("x_base", "y_base", "z_base", "bias"),
        coefficients=coefficients,
        ridge=0.0,
    )
    xyz = np.asarray([[0.20, -0.10, 0.03], [0.34, 0.08, 0.03]])
    features = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
    pixels = features @ coefficients

    reconstructed = approximate_column_backprojection_on_plane(
        fit, pixels, z_base=0.03
    )

    np.testing.assert_allclose(reconstructed, xyz, atol=1e-10)


def test_planar_backprojection_rejects_wrist_or_singular_fit() -> None:
    wrist = ProjectionFit(
        view="wrist",
        feature_names=("x_gripper", "y_gripper", "z_gripper", "opening", "bias"),
        coefficients=np.ones((5, 2)),
        ridge=0.0,
    )
    with np.testing.assert_raises_regex(ValueError, "column"):
        approximate_column_backprojection_on_plane(wrist, [10.0, 20.0], z_base=0.0)

    singular = ProjectionFit(
        view="column",
        feature_names=("x_base", "y_base", "z_base", "bias"),
        coefficients=np.asarray(
            [[1.0, 2.0], [2.0, 4.0], [0.0, 0.0], [0.0, 0.0]]
        ),
        ridge=0.0,
    )
    with np.testing.assert_raises_regex(ValueError, "not invertible"):
        approximate_column_backprojection_on_plane(
            singular, [10.0, 20.0], z_base=0.0
        )
