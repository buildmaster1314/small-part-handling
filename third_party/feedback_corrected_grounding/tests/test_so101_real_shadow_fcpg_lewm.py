import cv2
import numpy as np

from scripts.evaluate_so101_real_shadow_fcpg_lewm import (
    brown_box_descriptor,
    current_pixel_goal_cost,
    red_object_descriptor,
    red_object_phase_distances,
)


def _red_frame(center: tuple[int, int]) -> np.ndarray:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(frame, center, 3, (220, 20, 20), -1)
    return frame


def test_phase_distance_uses_fish_relative_to_wrist_gripper() -> None:
    query_frame = _red_frame((20, 10))
    bank_frame = _red_frame((12, 22))
    query = red_object_descriptor(
        query_frame, gripper_center_px=np.asarray([10.0, 12.0])
    )

    # The bank frame has different absolute fish pixels, but the same
    # fish-minus-gripper vector as the query.
    distances = red_object_phase_distances(
        query,
        np.stack([bank_frame]),
        gripper_centers_px=np.asarray([[2.0, 24.0]], dtype=np.float32),
    )

    assert distances[0] < 1e-6


def test_phase_distance_rejects_mismatched_gripper_count() -> None:
    query = red_object_descriptor(
        _red_frame((20, 10)), gripper_center_px=np.asarray([10.0, 12.0])
    )
    with np.testing.assert_raises(ValueError):
        red_object_phase_distances(
            query,
            np.stack([_red_frame((12, 22))]),
            gripper_centers_px=np.zeros((2, 2), dtype=np.float32),
        )


def test_brown_box_descriptor_uses_current_frame_component() -> None:
    frame = np.full((64, 64, 3), 235, dtype=np.uint8)
    frame[20:50, 24:56] = np.asarray([120, 78, 48], dtype=np.uint8)

    descriptor = brown_box_descriptor(frame)

    assert descriptor["visible"]
    np.testing.assert_allclose(descriptor["centroid_px"], [39.5, 34.5], atol=1.0)


def test_current_pixel_goal_cost_prefers_candidate_near_current_target() -> None:
    predicted = np.asarray(
        [
            [[10.0, 10.0], [12.0, 12.0]],
            [[10.0, 10.0], [28.0, 28.0]],
        ],
        dtype=np.float32,
    )
    cost = current_pixel_goal_cost(
        predicted,
        np.asarray([30.0, 30.0], dtype=np.float32),
        final_steps=1,
        image_size=(64, 64),
    )

    assert cost[1] < cost[0]
