import argparse

import numpy as np
import pytest

from fcpg.geometry import Camera, project_points

from fcpg.metaworld_utils import (
    make_env,
    observer_visibility_from_segmentation,
)
from scripts.evaluate_kineparity_stress import (
    CONSTELLATIONS,
    aggregate_rows,
    append_observer_history,
    corrupt_observer_correspondences,
    parse_choices,
    solve_pnp,
    solve_pnp_ransac,
    transform_update_magnitude,
)
from scripts.evaluate_kineparity_visual import deployment_observer_mask


def test_parse_choices_deduplicates_and_rejects_unknown_values() -> None:
    assert parse_choices("full9,distal3,full9", set(CONSTELLATIONS), "set") == [
        "full9",
        "distal3",
    ]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_choices("missing", set(CONSTELLATIONS), "set")


def test_correspondence_corruption_is_seeded_and_supports_total_dropout() -> None:
    points = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    valid = np.asarray([True, True])
    first = corrupt_observer_correspondences(
        points,
        valid,
        noise_std_px=2.0,
        dropout_probability=1.0,
        rng=np.random.default_rng(7),
    )
    second = corrupt_observer_correspondences(
        points,
        valid,
        noise_std_px=2.0,
        dropout_probability=1.0,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_allclose(first[0], second[0])
    assert not bool(first[1].any())


def test_observer_history_concatenates_and_trims_old_frames() -> None:
    history = []
    first = append_observer_history(
        history,
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([[10.0, 20.0]]),
        np.asarray([True]),
        max_frames=2,
    )
    assert first[0].shape == (1, 3)

    append_observer_history(
        history,
        np.asarray([[2.0, 0.0, 0.0]]),
        np.asarray([[20.0, 30.0]]),
        np.asarray([False]),
        max_frames=2,
    )
    third = append_observer_history(
        history,
        np.asarray([[3.0, 0.0, 0.0]]),
        np.asarray([[30.0, 40.0]]),
        np.asarray([True]),
        max_frames=2,
    )

    np.testing.assert_allclose(third[0][:, 0], [2.0, 3.0])
    np.testing.assert_array_equal(third[2], [False, True])


def test_transform_update_magnitude_uses_left_camera_update() -> None:
    nominal = np.eye(4)
    corrected = np.eye(4)
    corrected[0, 3] = 0.04
    translation_cm, rotation_deg = transform_update_magnitude(corrected, nominal)
    assert translation_cm == pytest.approx(4.0)
    assert rotation_deg == pytest.approx(0.0)


def test_stress_aggregation_retains_tail_and_worsening_metrics() -> None:
    rows = []
    for after in ([2.0, 3.0], [70.0, 4.0]):
        rows.append(
            {
                "before": np.asarray([30.0, 30.0]),
                "after": np.asarray(after),
                "displacement": np.asarray([28.0, 27.0]),
                "valid_observer_count": 4,
                "effective_rank": 5,
                "singular_values": np.ones(6),
                "retained_modes": np.asarray(
                    [True, True, True, False, False, False]
                ),
                "prior_control_variance": 100.0,
                "posterior_control_variance": 25.0,
                "global_innovation_rms_z": 4.0,
                "global_innovation_threshold_rms_z": 1.7,
                "solver_success": True,
                "update_translation_cm": 4.0,
                "update_rotation_deg": 10.0,
            }
        )

    summary = aggregate_rows(rows, "rotation", 10.0, "uniform5", "kineparity")

    assert summary["after_error_max_px"] == pytest.approx(70.0)
    assert summary["catastrophic_anchor_fraction_64px"] == pytest.approx(0.25)
    assert summary["worsened_anchor_fraction"] == pytest.approx(0.25)
    assert summary["solver_success_fraction"] == pytest.approx(1.0)
    assert summary["retained_mode_fraction"] == pytest.approx(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    )
    assert summary["posterior_to_prior_variance_ratio"] == pytest.approx(0.25)
    assert summary["global_innovation_rms_z_mean"] == pytest.approx(4.0)
    assert summary["global_innovation_threshold_rms_z_mean"] == pytest.approx(1.7)


def test_observer_visibility_requires_the_matching_robot_link() -> None:
    import mujoco

    env = make_env("pick-place-v3", "corner", 32, 32, task_index=0)
    try:
        body_names = ["right_l0", "right_l1"]
        geom_ids = []
        for body_name in body_names:
            body_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            matching = np.flatnonzero(
                np.asarray(env.model.geom_bodyid) == body_id
            )
            geom_ids.append(int(matching[0]))

        segmentation = np.full((32, 32, 2), -1, dtype=np.int32)
        segmentation[10, 10] = [int(mujoco.mjtObj.mjOBJ_GEOM), geom_ids[0]]
        segmentation[20, 20] = [int(mujoco.mjtObj.mjOBJ_GEOM), geom_ids[0]]
        points = np.asarray([[10.0, 10.0], [20.0, 20.0]])

        visible = observer_visibility_from_segmentation(
            env.model, segmentation, points, body_names, radius_px=0
        )

        np.testing.assert_array_equal(visible, [True, False])
    finally:
        env.close()


def test_deployment_observer_mask_does_not_consult_true_visibility() -> None:
    predicted = np.asarray(
        [
            [[10.0, 10.0], [-1.0, 5.0], [5.0, 32.0], [np.nan, 4.0]],
        ]
    )
    nominal_valid = np.asarray([[True, True, True, True]])

    mask = deployment_observer_mask(predicted, nominal_valid, width=32, height=32)

    np.testing.assert_array_equal(mask, [[True, False, False, False]])


def test_pnp_ransac_rejects_a_large_correspondence_outlier() -> None:
    intrinsic = np.asarray(
        [[160.0, 0.0, 112.0], [0.0, 160.0, 112.0], [0.0, 0.0, 1.0]]
    )
    nominal_extrinsic = np.eye(4)
    angle = np.deg2rad(8.0)
    true_extrinsic = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle), 0.025],
            [0.0, 1.0, 0.0, -0.01],
            [-np.sin(angle), 0.0, np.cos(angle), 0.02],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    observers = np.asarray(
        [
            [-0.20, -0.10, 1.1],
            [-0.10, 0.15, 1.2],
            [0.00, -0.20, 1.3],
            [0.10, 0.10, 1.4],
            [0.20, -0.05, 1.5],
            [-0.15, 0.05, 1.6],
            [0.05, 0.20, 1.7],
            [0.15, -0.15, 1.8],
        ]
    )
    controls = np.asarray([[-0.05, 0.02, 1.25], [0.12, -0.03, 1.55]])
    measured, _ = project_points(
        observers,
        Camera(intrinsic, true_extrinsic, width=224, height=224),
    )
    measured[0] += np.asarray([90.0, -80.0])
    control_true, _ = project_points(
        controls,
        Camera(intrinsic, true_extrinsic, width=224, height=224),
    )
    valid = np.ones(len(observers), dtype=bool)

    iterative, _, _, _ = solve_pnp(
        intrinsic, nominal_extrinsic, observers, measured, valid, controls
    )
    robust, _, rank, solved = solve_pnp_ransac(
        intrinsic, nominal_extrinsic, observers, measured, valid, controls
    )

    iterative_error = np.linalg.norm(iterative - control_true, axis=1).mean()
    robust_error = np.linalg.norm(robust - control_true, axis=1).mean()
    assert solved
    assert rank == 6
    assert robust_error < 0.1
    assert robust_error < iterative_error
