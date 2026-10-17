from __future__ import annotations

import cv2
import numpy as np
import pytest

from fcpg.so101_active_gripper import (
    apply_live_center_translation,
    apply_live_similarity_transform,
    fit_point_track_similarity_transform,
    invert_live_similarity_transform,
    localize_gripper_from_pulse,
    next_gripper_position,
    select_live_moving_tip_similarity_transform,
)


def _pulse() -> tuple[np.ndarray, np.ndarray]:
    frames = np.full((7, 96, 96, 3), 190, dtype=np.uint8)
    states = np.zeros((7, 6), dtype=np.float32)
    states[:, 5] = [80, 60, 35, 5, 35, 60, 80]
    xs = [68, 62, 56, 50, 56, 62, 68]
    for frame, x in zip(frames, xs, strict=True):
        triangle = np.asarray([[30, 18], [x + 9, 58], [x, 66]], dtype=np.int32)
        cv2.fillConvexPoly(frame, triangle, (35, 35, 35))
    return frames, states


def test_active_gripper_localization_finds_closed_distal_tip() -> None:
    frames, states = _pulse()
    result = localize_gripper_from_pulse(frames, states, minimum_component_area=10)
    assert result.closed_frame_index == 3
    assert result.open_frame_index in (0, 6)
    assert result.opening_span == pytest.approx(75.0)
    assert result.max_arm_motion == 0.0
    assert 45.0 <= result.center_px[0] <= 62.0
    assert 55.0 <= result.center_px[1] <= 72.0
    assert np.all(np.isfinite(result.open_tip_px))
    tip_tracks = np.asarray(result.moving_tip_track_candidates_px)
    assert tip_tracks.shape == (2, 2, 2)
    assert np.all(np.isfinite(tip_tracks))
    # Each row is one material point observed first open, then closed. The
    # synthetic moving jaw closes toward the left.
    assert tip_tracks[0, 0, 0] > tip_tracks[0, 1, 0] + 5.0


def test_active_gripper_localization_rejects_arm_motion() -> None:
    frames, states = _pulse()
    states[-1, 2] = 4.0
    with pytest.raises(ValueError, match="arm moved"):
        localize_gripper_from_pulse(frames, states)


def test_active_gripper_localization_requires_aperture_change() -> None:
    frames, states = _pulse()
    states[:, 5] = 10.0
    with pytest.raises(ValueError, match="opening span"):
        localize_gripper_from_pulse(frames, states)


def test_live_center_translation_preserves_anchor_geometry() -> None:
    nominal = np.asarray(
        [
            [[100.0, 80.0], [94.0, 82.0], [106.0, 78.0]],
            [[102.0, 83.0], [96.0, 85.0], [108.0, 81.0]],
        ],
        dtype=np.float32,
    )
    shifted, offset = apply_live_center_translation(nominal, [70.0, 60.0])
    np.testing.assert_allclose(offset, [-32.0, -23.0])
    np.testing.assert_allclose(shifted[-1, 0], [70.0, 60.0])
    np.testing.assert_allclose(shifted[:, 1] - shifted[:, 2], nominal[:, 1] - nominal[:, 2])


def test_live_translation_can_align_the_observed_moving_tip() -> None:
    nominal = np.asarray(
        [[[100.0, 80.0], [94.0, 82.0], [106.0, 78.0]]],
        dtype=np.float32,
    )
    shifted, offset = apply_live_center_translation(
        nominal, [90.0, 70.0], anchor_index=2
    )
    np.testing.assert_allclose(offset, [-16.0, -8.0])
    np.testing.assert_allclose(shifted[-1, 2], [90.0, 70.0])


def test_tip_similarity_recovers_scale_rotation_and_translation() -> None:
    nominal_tips = np.asarray([[120.0, 80.0], [80.0, 80.0]], dtype=np.float32)
    angle = np.deg2rad(12.0)
    linear = 0.75 * np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    translation = np.asarray([14.0, -9.0], dtype=np.float32)
    observed_tips = nominal_tips @ linear.T + translation

    transform, scale, rotation = fit_point_track_similarity_transform(
        nominal_tips, observed_tips
    )

    assert scale == pytest.approx(0.75)
    assert rotation == pytest.approx(12.0)
    np.testing.assert_allclose(transform[:, :2], linear, atol=1.0e-6)
    np.testing.assert_allclose(transform[:, 2], translation, atol=1.0e-5)

    anchors = np.asarray(
        [[[100.0, 80.0], [120.0, 80.0], [80.0, 80.0]]],
        dtype=np.float32,
    )
    transformed = apply_live_similarity_transform(anchors, transform)
    np.testing.assert_allclose(transformed[0, 1:], observed_tips, atol=1.0e-5)
    np.testing.assert_allclose(
        transformed[0, 0], np.asarray([100.0, 80.0]) @ linear.T + translation
    )


def test_tip_similarity_rejects_degenerate_pairs() -> None:
    with pytest.raises(ValueError, match="one pixel"):
        fit_point_track_similarity_transform(
            np.asarray([[1.0, 2.0], [1.0, 2.0]]),
            np.asarray([[4.0, 5.0], [8.0, 9.0]]),
        )


def test_live_similarity_inverse_round_trip() -> None:
    transform = np.asarray(
        [[0.8, -0.2, 14.0], [0.2, 0.8, -9.0]], dtype=np.float32
    )
    nominal = np.asarray(
        [[[100.0, 80.0], [120.0, 82.0], [78.0, 76.0]]], dtype=np.float32
    )

    live = apply_live_similarity_transform(nominal, transform)
    recovered = invert_live_similarity_transform(live, transform)

    np.testing.assert_allclose(recovered, nominal, atol=1.0e-5)


def test_live_similarity_inverse_rejects_singular_transform() -> None:
    with pytest.raises(ValueError, match="not safely invertible"):
        invert_live_similarity_transform(
            np.asarray([10.0, 20.0]),
            np.asarray([[1.0, 0.0, 2.0], [2.0, 0.0, 3.0]]),
        )


def test_live_similarity_selector_uses_local_track_and_preserves_midpoint() -> None:
    nominal = np.asarray(
        [
            [[100.0, 80.0], [120.0, 80.0], [80.0, 80.0]],
            [[110.0, 80.0], [120.0, 80.0], [100.0, 80.0]],
        ],
        dtype=np.float32,
    )
    angle = np.deg2rad(10.0)
    linear = 0.75 * np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    translation = np.asarray([12.0, -7.0], dtype=np.float32)
    observed = nominal[:, 2] @ linear.T + translation

    result = select_live_moving_tip_similarity_transform(
        nominal,
        np.asarray(
            [[[40.0, 20.0], [50.0, 20.0]], observed], dtype=np.float32
        ),
    )

    assert result["candidate_index"] == 1
    np.testing.assert_allclose(result["transform_px"][:, :2], linear, atol=1.0e-6)
    transformed = result["transformed_anchors_px"]
    np.testing.assert_allclose(transformed[:, 2], observed, atol=1.0e-5)
    np.testing.assert_allclose(
        transformed[:, 0], transformed[:, 1:].mean(axis=1), atol=1.0e-5
    )


def test_live_similarity_selector_accepts_local_fifty_degree_camera_rotation() -> None:
    nominal = np.asarray(
        [
            [[102.295, 95.190], [122.289, 92.224], [82.301, 98.155]],
            [[119.143, 92.406], [122.289, 92.224], [115.997, 92.587]],
        ],
        dtype=np.float32,
    )
    observed = np.asarray(
        [[78.220, 61.806], [94.147, 75.422]], dtype=np.float32
    )
    candidates = np.asarray(
        [
            [[75.543, 30.070], [56.390, 3.774]],
            observed,
        ],
        dtype=np.float32,
    )

    result = select_live_moving_tip_similarity_transform(nominal, candidates)

    assert result["candidate_index"] == 1
    assert result["scale"] == pytest.approx(0.6135, abs=1.0e-3)
    assert result["rotation_deg"] == pytest.approx(49.91, abs=0.02)
    assert result["maximum_center_offset_norm_px"] < 35.0
    np.testing.assert_allclose(
        result["transformed_anchors_px"][:, 0],
        result["transformed_anchors_px"][:, 1:].mean(axis=1),
        atol=5.0e-4,
    )

    with pytest.raises(ValueError, match="no automatic moving-tip track fit"):
        select_live_moving_tip_similarity_transform(
            nominal,
            candidates,
            maximum_rotation_deg=45.0,
        )


def test_next_gripper_position_is_bounded() -> None:
    assert next_gripper_position(5.0, 60.0, 1.5) == pytest.approx(6.5)
    assert next_gripper_position(60.0, 5.0, 1.5) == pytest.approx(58.5)
    assert next_gripper_position(59.5, 60.0, 1.5) == pytest.approx(60.0)
    with pytest.raises(ValueError, match=r"\[0,100\]"):
        next_gripper_position(101.0, 60.0, 1.5)
