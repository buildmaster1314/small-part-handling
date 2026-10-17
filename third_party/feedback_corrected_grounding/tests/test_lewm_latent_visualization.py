from __future__ import annotations

import numpy as np
import pytest

from fcpg.lewm_latent_visualization import (
    fit_reference_pca,
    image_overlay_extent,
    image_xy_to_row_col,
    per_anchor_nearest_neighbors,
    per_anchor_normalized_errors,
    prepare_image_overlay,
    project_latents_from_reference,
)


def test_pca_is_fitted_only_on_reference_and_has_stable_component_signs() -> None:
    reference = np.array(
        [[-3.0, -0.5, 0.0], [-1.0, 0.5, 0.0], [1.0, -0.5, 0.0], [3.0, 0.5, 0.0]]
    )
    predictions = np.array([[1000.0, -2000.0, 500.0]])

    fitted, (projected_predictions,) = project_latents_from_reference(
        reference, predictions
    )
    repeated = fit_reference_pca(reference)

    np.testing.assert_allclose(fitted.mean, reference.mean(axis=0))
    np.testing.assert_array_equal(fitted.components, repeated.components)
    np.testing.assert_array_equal(
        projected_predictions, fitted.transform(predictions)
    )
    for component in fitted.components:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] >= 0.0


def test_reference_pca_flattens_only_sample_axes_and_retains_transform_shape() -> None:
    reference = np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)
    projection = fit_reference_pca(reference, n_components=2)

    transformed = projection.transform(reference[:2])

    assert projection.mean.shape == (5,)
    assert projection.components.shape == (2, 5)
    assert transformed.shape == (2, 3, 2)


def test_normalized_errors_use_an_independent_scale_for_each_anchor() -> None:
    reference = np.array(
        [
            [[-1.0, -1.0], [-2.0, -2.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ]
    )
    target = np.zeros((3, 2, 2), dtype=np.float64)
    predicted = np.broadcast_to([[1.0, 1.0], [2.0, 2.0]], target.shape)

    errors = per_anchor_normalized_errors(predicted, target, reference)

    assert errors.shape == (3, 2)
    np.testing.assert_allclose(errors, 1.0)


def test_normalized_errors_remain_finite_for_a_collapsed_reference_anchor() -> None:
    reference = np.zeros((4, 1, 3), dtype=np.float64)
    target = np.zeros((2, 1, 3), dtype=np.float64)

    exact = per_anchor_normalized_errors(target, target, reference)
    nonzero = per_anchor_normalized_errors(target + 1.0, target, reference, epsilon=0.5)

    np.testing.assert_array_equal(exact, np.zeros((2, 1)))
    np.testing.assert_allclose(nonzero, 2.0)


def test_nearest_neighbors_are_per_anchor_and_ties_use_reference_order() -> None:
    reference = np.array(
        [
            [[-1.0, 0.0], [10.0, 0.0]],
            [[1.0, 0.0], [20.0, 0.0]],
            [[5.0, 0.0], [30.0, 0.0]],
        ]
    )
    query = np.array([[[0.0, 0.0], [29.0, 0.0]]])

    result = per_anchor_nearest_neighbors(
        query, reference, reference_labels=np.array(["a", "b", "c"]), k=2
    )

    assert result.indices.shape == (1, 2, 2)
    np.testing.assert_array_equal(result.indices, [[[0, 1], [2, 1]]])
    np.testing.assert_allclose(result.distances, [[[1.0, 1.0], [1.0, 9.0]]])
    np.testing.assert_array_equal(result.labels, [[['a', 'b'], ['c', 'b']]])


def test_image_helpers_preserve_xy_but_index_arrays_as_row_column() -> None:
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    points_xy = np.array([[4.0, 0.0], [0.0, 2.0], [2.2, 1.4]])

    row_col = image_xy_to_row_col(points_xy, image)
    overlay = prepare_image_overlay(points_xy, image.shape)

    np.testing.assert_array_equal(row_col, [[0, 4], [2, 0], [1, 2]])
    np.testing.assert_array_equal(overlay.points_xy, points_xy)
    np.testing.assert_array_equal(overlay.row_col, row_col)
    assert overlay.origin == "upper"
    assert overlay.extent == (-0.5, 4.5, 2.5, -0.5)
    assert image_overlay_extent((3, 5)) == overlay.extent


def test_image_helpers_reject_or_explicitly_clip_out_of_bounds_points() -> None:
    points = np.array([[-2.0, 7.0]])

    with pytest.raises(ValueError, match="inside the image"):
        image_xy_to_row_col(points, (4, 6))

    overlay = prepare_image_overlay(points, (4, 6), clip=True)
    np.testing.assert_array_equal(overlay.points_xy, [[0.0, 3.0]])
    np.testing.assert_array_equal(overlay.row_col, [[3, 0]])


def test_latent_helpers_reject_misaligned_or_nonfinite_inputs() -> None:
    reference = np.zeros((3, 2, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="anchor and latent dimensions"):
        per_anchor_normalized_errors(
            np.zeros((1, 3, 4)), np.zeros((1, 3, 4)), reference
        )
    with pytest.raises(ValueError, match="finite"):
        fit_reference_pca(np.array([[0.0, np.nan], [1.0, 2.0]]))
    with pytest.raises(ValueError, match="reference_labels"):
        per_anchor_nearest_neighbors(
            np.zeros((1, 2, 4)), reference, reference_labels=["only-one"]
        )
