from __future__ import annotations

import inspect

import numpy as np

from fcpg.geometry import Camera, project_points
from fcpg.lewm_goal_conditioned_deployable import (
    DeployableGoalConditionedMetric3DLeWM,
)
from scripts.evaluate_lewm_goal_conditioned_deployable_robustness import (
    aggregate_summary,
    nominal_camera_for_rotation,
)
from scripts.train_lewm_goal_conditioned_deployable import (
    _metric_block,
    infer_nominal_extrinsic,
)


def _camera_data() -> dict[str, np.ndarray]:
    intrinsic = np.asarray(
        [[180.0, 0.0, 112.0], [0.0, 180.0, 112.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    nominal = np.eye(4, dtype=np.float32)
    nominal[2, 3] = 1.0
    actual = nominal.copy()
    angle = np.deg2rad(3.0)
    actual[:3, :3] = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    xyz = np.asarray(
        [
            [[-0.05, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]],
            [[-0.04, 0.0, 1.0], [0.06, 0.0, 1.0], [0.01, 0.05, 1.0]],
        ],
        dtype=np.float32,
    )
    q_nominal = []
    q_true = []
    for index, points in enumerate(xyz):
        q_n, _ = project_points(points, Camera(intrinsic, nominal, 224, 224))
        ext = nominal if index == 0 else actual
        q_t, _ = project_points(points, Camera(intrinsic, ext, 224, 224))
        q_nominal.append(q_n)
        q_true.append(q_t)
    return {
        "current_rgb": np.zeros((2, 224, 224, 3), dtype=np.uint8),
        "current_xyz": xyz,
        "camera_intrinsic": np.repeat(intrinsic[None], 2, axis=0),
        "true_extrinsic": np.stack([nominal, actual]),
        "current_q_nominal": np.asarray(q_nominal, dtype=np.float32),
        "current_q_true": np.asarray(q_true, dtype=np.float32),
    }


def test_deployable_signature_has_no_current_entity_input() -> None:
    parameters = inspect.signature(
        DeployableGoalConditionedMetric3DLeWM.predict
    ).parameters
    assert "current_entity_xyz" not in parameters
    assert "current_goal_xyz" in parameters


def test_infer_nominal_extrinsic_reproduces_nominal_projection() -> None:
    data = _camera_data()
    inferred = infer_nominal_extrinsic(data)
    np.testing.assert_allclose(inferred, data["true_extrinsic"][0], atol=1.0e-6)


def test_nominal_camera_rotation_returns_matching_projection_and_extrinsic() -> None:
    data = _camera_data()
    q, valid, extrinsic = nominal_camera_for_rotation(
        data,
        axis_world=(0.0, 1.0, 0.0),
        degrees=0.0,
    )
    np.testing.assert_allclose(extrinsic, data["true_extrinsic"], atol=1.0e-6)
    np.testing.assert_allclose(q, data["current_q_true"], atol=1.0e-5)
    assert bool(valid.all())


def test_metric_block_reports_perfect_ranking_for_perfect_prediction() -> None:
    target_robot = np.zeros((2, 8, 3, 3, 3), dtype=np.float32)
    target_entity = np.zeros((2, 8, 3, 3), dtype=np.float32)
    target_goal = np.zeros((2, 8, 3, 3), dtype=np.float32)
    target_entity[:, :, -1, 0] = np.arange(8, dtype=np.float32)[None]
    report = _metric_block(
        target_robot,
        target_entity,
        target_robot,
        target_entity,
        target_goal,
    )
    assert report["robot_anchor_epe_mm"] == 0.0
    assert report["entity_epe_mm"] == 0.0
    assert report["top1_action_accuracy"] == 1.0
    assert report["spearman_mean"] == 1.0


def test_aggregate_summary_uses_fcpg_minus_nominal_for_spearman() -> None:
    condition = {
        "rotation_degrees": 5.0,
        "grounding": {
            "nominal": {"epe_px": 10.0},
            "fcpg": {"epe_px": 2.0},
        },
        "future": {
            "nominal": {
                "robot_anchor_epe_mm": 5.0,
                "entity_epe_mm": 7.0,
                "spearman_mean": 0.3,
                "selected_actual_goal_distance_h8_mm": 9.0,
                "by_task": {
                    task: {
                        "entity_epe_mm": 7.0,
                        "spearman_mean": 0.3,
                        "selected_actual_goal_distance_h8_mm": 9.0,
                    }
                    for task in ("reach-v3", "push-v3")
                },
            },
            "fcpg": {
                "robot_anchor_epe_mm": 4.0,
                "entity_epe_mm": 5.0,
                "spearman_mean": 0.5,
                "selected_actual_goal_distance_h8_mm": 6.0,
                "by_task": {
                    task: {
                        "entity_epe_mm": 5.0,
                        "spearman_mean": 0.5,
                        "selected_actual_goal_distance_h8_mm": 6.0,
                    }
                    for task in ("reach-v3", "push-v3")
                },
            },
        },
    }
    summary = aggregate_summary([condition])
    assert summary["mean_current_epe_reduction_percent"] == 80.0
    assert summary["mean_entity_epe_improvement_mm"] == 2.0
    assert summary["mean_spearman_delta"] == 0.2
    assert summary["mean_selected_goal_distance_improvement_mm"] == 3.0
