"""Tests for generation-only local visual-trust corruptions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fcpg.visual_trust_corruptions import (
    CORRUPTION_TYPES,
    apply_irregular_occlusion,
    apply_local_gaussian_blur,
    apply_square_occlusion,
    apply_texture_distractor,
    apply_visual_trust_corruption,
)


CORRUPTION_FUNCTIONS = (
    apply_square_occlusion,
    apply_irregular_occlusion,
    apply_texture_distractor,
    apply_local_gaussian_blur,
)


def _mask_bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


@pytest.mark.parametrize("corrupt", CORRUPTION_FUNCTIONS)
def test_corruptions_are_seeded_local_and_preserve_uint8_hwc(corrupt) -> None:
    image = np.random.default_rng(5).integers(
        0, 256, size=(21, 25, 3), dtype=np.uint8
    )
    original = image.copy()

    first, first_mask, first_metadata = corrupt(
        image, (12.25, 10.75), np.random.default_rng(73)
    )
    second, second_mask, second_metadata = corrupt(
        image, (12.25, 10.75), np.random.default_rng(73)
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert first_metadata == second_metadata
    assert first.shape == image.shape
    assert first.dtype == np.uint8
    assert first_mask.shape == image.shape[:2]
    assert first_mask.dtype == np.bool_
    assert bool(first_mask.any())
    np.testing.assert_array_equal(first[~first_mask], image[~first_mask])
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(first, image)
    assert first_metadata["affected_pixel_count"] == int(first_mask.sum())
    assert first_metadata["affected_bbox_xyxy"] == _mask_bbox(first_mask)
    assert first_metadata["mask_semantics"] == "intervention_support"
    json.dumps(first_metadata)


def test_square_occlusion_has_exact_clipped_support_and_color() -> None:
    image = np.full((6, 8, 3), 255, dtype=np.uint8)

    result, mask, metadata = apply_square_occlusion(
        image,
        (0.0, 0.0),
        np.random.default_rng(1),
        radius_px=2,
        fill_rgb=(3, 5, 7),
    )

    expected_mask = np.zeros((6, 8), dtype=np.bool_)
    expected_mask[:3, :3] = True
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(
        result[:3, :3], np.broadcast_to([3, 5, 7], (3, 3, 3))
    )
    assert metadata["radius_px"] == 2
    assert metadata["fill_rgb"] == [3, 5, 7]
    assert metadata["affected_bbox_xyxy"] == [0, 0, 3, 3]


def test_irregular_occlusion_is_nonrectangular_and_contains_anchor() -> None:
    image = np.full((25, 25, 3), 211, dtype=np.uint8)

    result, mask, metadata = apply_irregular_occlusion(
        image,
        (12.0, 12.0),
        np.random.default_rng(19),
        radius_px=8,
        vertex_count=9,
        fill_rgb=0,
    )

    assert bool(mask[12, 12])
    assert 20 < int(mask.sum()) < 17 * 17
    occupied_widths = mask.sum(axis=1)
    assert np.unique(occupied_widths[occupied_widths > 0]).size > 2
    np.testing.assert_array_equal(
        result[mask], np.zeros((mask.sum(), 3), dtype=np.uint8)
    )
    assert metadata["vertex_count"] == 9
    assert len(metadata["vertices_xy"]) == 9


def test_texture_distractor_uses_two_oriented_texture_colors() -> None:
    image = np.full((19, 19, 3), 127, dtype=np.uint8)

    result, mask, metadata = apply_texture_distractor(
        image,
        (9.0, 9.0),
        np.random.default_rng(29),
        radius_px=7,
        tile_size_px=2,
        opacity=1.0,
        colors_rgb=((0, 10, 20), (230, 240, 250)),
    )

    assert set(np.unique(result[..., 0][mask])) == {0, 230}
    np.testing.assert_array_equal(result[~mask], image[~mask])
    assert metadata["colors_rgb"] == [[0, 10, 20], [230, 240, 250]]
    assert metadata["tile_size_px"] == 2
    assert metadata["opacity"] == 1.0


def test_local_gaussian_blur_spreads_an_impulse_only_inside_support() -> None:
    image = np.zeros((15, 15, 3), dtype=np.uint8)
    image[7, 7] = 255

    result, mask, metadata = apply_local_gaussian_blur(
        image,
        (7.0, 7.0),
        np.random.default_rng(31),
        radius_px=4,
        sigma=1.0,
    )

    assert 0 < int(result[7, 7, 0]) < 255
    assert int(result[7, 8, 0]) > 0
    np.testing.assert_array_equal(result[~mask], image[~mask])
    assert metadata["sigma"] == 1.0
    assert metadata["kernel_radius_px"] == 3


@pytest.mark.parametrize("anchor", [(0.0, 0.0), (10.0, 8.0)])
@pytest.mark.parametrize("corrupt", CORRUPTION_FUNCTIONS)
def test_every_corruption_is_safe_at_image_boundaries(corrupt, anchor) -> None:
    image = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)

    result, mask, metadata = corrupt(
        image,
        anchor,
        np.random.default_rng(43),
        radius_px=20,
    )

    center_x, center_y = (int(value) for value in anchor)
    assert bool(mask[center_y, center_x])
    assert result.shape == image.shape
    assert metadata["affected_bbox_xyxy"][0] >= 0
    assert metadata["affected_bbox_xyxy"][1] >= 0
    assert metadata["affected_bbox_xyxy"][2] <= image.shape[1]
    assert metadata["affected_bbox_xyxy"][3] <= image.shape[0]


def test_dispatcher_samples_type_deterministically_and_accepts_aliases() -> None:
    image = np.zeros((13, 17, 3), dtype=np.uint8)

    first = apply_visual_trust_corruption(
        image, (8.0, 6.0), np.random.default_rng(101)
    )
    second = apply_visual_trust_corruption(
        image, (8.0, 6.0), np.random.default_rng(101)
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[2]["corruption_type"] in CORRUPTION_TYPES

    _, _, metadata = apply_visual_trust_corruption(
        image,
        (8.0, 6.0),
        np.random.default_rng(103),
        corruption_type="square",
        radius_px=1,
        fill_rgb=9,
    )
    assert metadata["corruption_type"] == "square_occlusion"

    with pytest.raises(ValueError, match="unknown corruption_type"):
        apply_visual_trust_corruption(
            image,
            (8.0, 6.0),
            np.random.default_rng(107),
            corruption_type="unknown",
        )


def test_corruptions_do_not_consume_numpy_global_random_state() -> None:
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    np.random.seed(211)
    expected = np.random.random()
    np.random.seed(211)

    apply_texture_distractor(image, (4.0, 4.0), np.random.default_rng(223))

    assert np.random.random() == expected


def test_invalid_images_anchors_generator_and_parameters_are_rejected() -> None:
    image = np.zeros((7, 9, 3), dtype=np.uint8)
    rng = np.random.default_rng(227)

    with pytest.raises(ValueError, match="shape"):
        apply_square_occlusion(np.zeros((3, 7, 9), dtype=np.uint8), (1, 1), rng)
    with pytest.raises(ValueError, match="dtype uint8"):
        apply_square_occlusion(image.astype(np.float32), (1, 1), rng)
    with pytest.raises(ValueError, match="inside the image"):
        apply_square_occlusion(image, (9, 1), rng)
    with pytest.raises(ValueError, match="finite"):
        apply_square_occlusion(image, (np.nan, 1), rng)
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        apply_square_occlusion(image, (1, 1), object())
    with pytest.raises(ValueError, match="radius_px"):
        apply_square_occlusion(image, (1, 1), rng, radius_px=-1)
    with pytest.raises(ValueError, match="fill_rgb"):
        apply_square_occlusion(image, (1, 1), rng, fill_rgb=(0, 1, 256))
    with pytest.raises(ValueError, match="vertex_count"):
        apply_irregular_occlusion(image, (1, 1), rng, vertex_count=4)
    with pytest.raises(ValueError, match="colors_rgb"):
        apply_texture_distractor(
            image, (1, 1), rng, colors_rgb=((1, 2, 3), (1, 2, 3))
        )
    with pytest.raises(ValueError, match="sigma"):
        apply_local_gaussian_blur(image, (1, 1), rng, sigma=0.0)
