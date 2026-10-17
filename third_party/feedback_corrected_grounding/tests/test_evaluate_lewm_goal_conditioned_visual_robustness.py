from __future__ import annotations

import numpy as np

from scripts.evaluate_lewm_goal_conditioned_visual_robustness import (
    aggregate_summary,
    axis_angle_rotation_matrix,
    global_gaussian_blur,
    rotate_extrinsic_about_world_axis,
)


def test_axis_angle_rotation_is_orthonormal() -> None:
    rotation = axis_angle_rotation_matrix((0.0, 1.0, 0.0), 5.0)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1.0e-12)


def test_rotated_extrinsic_preserves_camera_center() -> None:
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, 3] = np.asarray((0.4, -0.2, 1.3))
    before = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    rotated = rotate_extrinsic_about_world_axis(extrinsic, (1.0, 0.0, 0.0), 12.0)
    after = -rotated[:3, :3].T @ rotated[:3, 3]
    np.testing.assert_allclose(after, before, atol=1.0e-12)
    np.testing.assert_allclose(
        rotated[:3, :3].T @ rotated[:3, :3],
        np.eye(3),
        atol=1.0e-12,
    )


def test_global_gaussian_blur_is_deterministic_and_zero_is_identity() -> None:
    images = np.zeros((2, 17, 19, 3), dtype=np.uint8)
    images[:, 8, 9] = 255
    identity = global_gaussian_blur(images, 0.0)
    np.testing.assert_array_equal(identity, images)
    assert identity is not images
    first = global_gaussian_blur(images, 2.0)
    second = global_gaussian_blur(images, 2.0)
    np.testing.assert_array_equal(first, second)
    assert int(first[:, 8, 9].max()) < 255
    assert int(first[:, 8, 8].max()) > 0


def test_aggregate_summary_uses_only_drifted_cells() -> None:
    def condition(rotation: float, nominal: float, fcpg: float) -> dict[str, object]:
        future_nominal = {
            "robot_anchor_epe_mm": 10.0,
            "entity_epe_mm": 20.0,
            "spearman_mean": 0.4,
            "selected_actual_goal_distance_h8_mm": 30.0,
        }
        future_fcpg = {
            "robot_anchor_epe_mm": 8.0,
            "entity_epe_mm": 19.0,
            "spearman_mean": 0.5,
            "selected_actual_goal_distance_h8_mm": 27.0,
        }
        return {
            "rotation_degrees": rotation,
            "blur_sigma": 0.0,
            "grounding": {
                "nominal": {"epe_px": nominal},
                "fcpg": {"epe_px": fcpg},
            },
            "future": {
                "nominal": future_nominal,
                "fcpg": future_fcpg,
                "oracle": {"entity_epe_mm": 18.0},
            },
        }

    summary = aggregate_summary([condition(0.0, 0.0, 2.0), condition(5.0, 10.0, 4.0)])
    assert summary["drifted_condition_count"] == 1
    np.testing.assert_allclose(summary["mean_current_epe_reduction_percent"], 60.0)
    np.testing.assert_allclose(summary["mean_robot_anchor_epe_improvement_mm"], 2.0)
    np.testing.assert_allclose(summary["mean_entity_epe_improvement_mm"], 1.0)
    np.testing.assert_allclose(summary["mean_spearman_delta"], 0.1)
    np.testing.assert_allclose(summary["mean_selected_goal_distance_improvement_mm"], 3.0)
