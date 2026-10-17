from types import SimpleNamespace

import numpy as np
import pytest

from scripts.render_so101_mujoco_prediction import (
    external_validation_assignments,
    fit_panel_image,
    selected_prediction_indices,
)


def test_selected_prediction_indices_keeps_complete_selected_cycles() -> None:
    display_inputs = np.repeat(np.asarray([10, 15, 20, 25]), 3)

    selected = selected_prediction_indices(display_inputs, origin_stride=2)

    np.testing.assert_array_equal(selected, [0, 1, 2, 6, 7, 8, 9, 10, 11])


def test_selected_prediction_indices_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        selected_prediction_indices(np.asarray([], dtype=np.int64), origin_stride=1)


def test_fit_panel_image_letterboxes_without_distortion() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)

    panel = fit_panel_image(image, (100, 100))

    assert panel.shape == (100, 100, 3)
    assert panel.dtype == np.uint8
    np.testing.assert_array_equal(panel[25:75], 255)
    assert np.all(panel[:24] < 255)
    assert np.all(panel[76:] < 255)


def test_fit_panel_image_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match=r"\[H, W, 3\]"):
        fit_panel_image(np.zeros((10, 10), dtype=np.uint8))


def test_external_cache_episodes_are_all_held_out() -> None:
    caches = [
        SimpleNamespace(
            dataset_id="external",
            data=SimpleNamespace(episodes=np.asarray([3, 3, 7, 7])),
        )
    ]

    assignments = external_validation_assignments(caches)

    assert assignments == {("external", 3): "val", ("external", 7): "val"}
