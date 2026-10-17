from __future__ import annotations

import itertools
import math
from collections import deque

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from fcpg.geometric_baselines import (
    OnlinePnPCalibrator,
    generate_joint_prototype_hypotheses,
    refine_points_with_prototypes,
)
from fcpg.geometry import Camera, project_points
from fcpg.transformer_grounding import sample_local_feature_tokens
from scripts.build_vit_prototype_bank import load_keypoint_source, select_prototype_indices
from scripts.evaluate_action_chunk_rollout import history_batch, optional_mean


def test_prototype_indices_stay_inside_training_split_and_are_deterministic() -> None:
    train = np.array([0, 1, 2, 3, 4])
    valid = np.ones((6, 3), dtype=bool)
    valid[2, 1] = False
    first = select_prototype_indices(train, valid, count=3, seed=7)
    second = select_prototype_indices(train, valid, count=3, seed=7)

    np.testing.assert_array_equal(first, second)
    assert set(first).issubset({0, 1, 3, 4})
    assert 5 not in first


def test_prototype_refinement_selects_matching_local_descriptor() -> None:
    feature_map = torch.zeros(1, 2, 9, 9)
    feature_map[0, 0] = 1.0
    feature_map[0, :, 4, 6] = torch.tensor([0.0, 2.0])
    q_nominal = torch.tensor([[[4.0, 4.0]]])
    prototypes = torch.tensor([[[0.0, 1.0]]])

    result = refine_points_with_prototypes(
        feature_map,
        q_nominal,
        prototypes,
        image_size=(9, 9),
        radius_px=4.0,
        grid_size=9,
    )

    torch.testing.assert_close(result.points_px, torch.tensor([[[6.0, 4.0]]]))
    assert bool(result.valid.all())
    assert float(result.confidence.item()) == 1.0


def test_prototype_refinement_validates_bank_shape() -> None:
    feature_map = torch.zeros(1, 4, 5, 5)
    q_nominal = torch.zeros(1, 2, 2)
    prototypes = torch.zeros(3, 1, 4)
    try:
        refine_points_with_prototypes(feature_map, q_nominal, prototypes, (5, 5))
    except ValueError as error:
        assert "keypoint count" in str(error)
    else:
        raise AssertionError("mismatched prototype bank must fail")


def test_prototype_source_preserves_semantic_keypoint_order(tmp_path) -> None:
    path = tmp_path / "source.npz"
    q_true = np.zeros((2, 3, 2), dtype=np.float32)
    valid = np.ones((2, 3), dtype=bool)
    sites = np.asarray(["leftEndEffector", "rightEndEffector", "endEffector"])
    np.savez(path, q_true=q_true, valid_true=valid, keypoint_site=sites)

    with np.load(path, allow_pickle=False) as raw:
        loaded_q, loaded_valid, loaded_sites = load_keypoint_source(raw, "primary")

    np.testing.assert_array_equal(loaded_q, q_true)
    np.testing.assert_array_equal(loaded_valid, valid)
    assert loaded_sites == tuple(sites.tolist())


def test_joint_prototype_hypotheses_shapes_and_per_anchor_validity() -> None:
    feature_map = torch.ones(2, 1, 9, 9)
    q_nominal = torch.tensor(
        [
            [[4.0, 4.0], [4.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    prototypes = torch.ones(2, 1, 1)

    result = generate_joint_prototype_hypotheses(
        feature_map,
        q_nominal,
        prototypes,
        image_size=(9, 9),
        radius_px=4.0,
        grid_size=9,
        per_anchor_topk=2,
        max_hypotheses=3,
        min_confidence=0.5,
    )

    assert result.points_px.shape == (2, 3, 2, 2)
    assert result.joint_score.shape == (2, 3)
    assert result.valid.shape == (2, 3, 2)
    assert result.grid_indices.shape == (2, 3, 2)
    assert result.grid_indices.dtype == torch.long
    assert result.valid.dtype == torch.bool
    assert bool(result.valid[0].all())
    assert not bool(result.valid[1].any())


def test_joint_prototype_hypotheses_top1_exactly_matches_legacy_hard_match() -> None:
    generator = torch.Generator().manual_seed(17)
    feature_map = torch.randn(2, 5, 11, 11, generator=generator)
    q_nominal = torch.tensor(
        [
            [[5.0, 5.0], [4.0, 6.0], [6.0, 4.0]],
            [[4.5, 5.5], [5.5, 4.5], [5.0, 5.0]],
        ]
    )
    prototypes = torch.randn(3, 4, 5, generator=generator)

    legacy = refine_points_with_prototypes(
        feature_map,
        q_nominal,
        prototypes,
        image_size=(11, 11),
        radius_px=4.0,
        grid_size=9,
        temperature=0.0,
        min_confidence=0.1,
    )
    joint = generate_joint_prototype_hypotheses(
        feature_map,
        q_nominal,
        prototypes,
        image_size=(11, 11),
        radius_px=4.0,
        grid_size=9,
        per_anchor_topk=16,
        max_hypotheses=1,
        min_confidence=0.1,
    )

    torch.testing.assert_close(joint.points_px[:, 0], legacy.points_px, rtol=0, atol=0)
    torch.testing.assert_close(
        joint.joint_score[:, 0],
        legacy.confidence.sum(dim=-1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(joint.valid[:, 0], legacy.valid)


def test_joint_prototype_hypotheses_use_deterministic_lexicographic_ties() -> None:
    result = generate_joint_prototype_hypotheses(
        feature_map=torch.ones(1, 1, 5, 5),
        q_nominal_px=torch.tensor([[[2.0, 2.0], [2.0, 2.0]]]),
        prototype_bank=torch.ones(2, 1, 1),
        image_size=(5, 5),
        radius_px=2.0,
        grid_size=5,
        per_anchor_topk=2,
        max_hypotheses=4,
    )

    expected = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])
    torch.testing.assert_close(result.grid_indices, expected)
    torch.testing.assert_close(result.joint_score, torch.full((1, 4), 2.0))


def test_joint_prototype_hypotheses_match_cartesian_brute_force() -> None:
    generator = torch.Generator().manual_seed(23)
    feature_map = torch.randn(1, 4, 7, 7, generator=generator)
    q_nominal = torch.tensor([[[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]]])
    prototypes = torch.randn(3, 2, 4, generator=generator)
    per_anchor_topk = 3
    max_hypotheses = 7

    result = generate_joint_prototype_hypotheses(
        feature_map,
        q_nominal,
        prototypes,
        image_size=(7, 7),
        radius_px=3.0,
        grid_size=7,
        per_anchor_topk=per_anchor_topk,
        max_hypotheses=max_hypotheses,
    )

    local, _ = sample_local_feature_tokens(
        feature_map,
        q_nominal,
        image_size=(7, 7),
        radius_px=3.0,
        grid_size=7,
    )
    scores = torch.einsum(
        "bknc,kmc->bknm",
        F.normalize(local.float(), dim=-1),
        F.normalize(prototypes.float(), dim=-1),
    ).amax(dim=-1)[0]
    top_indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[
        :, :per_anchor_topk
    ]
    brute_force: list[tuple[float, tuple[int, ...]]] = []
    for indices in itertools.product(*[row.tolist() for row in top_indices]):
        score = sum(float(scores[keypoint, index]) for keypoint, index in enumerate(indices))
        brute_force.append((score, indices))
    brute_force.sort(key=lambda item: (-item[0], item[1]))
    expected_indices = torch.tensor(
        [[indices for _, indices in brute_force[:max_hypotheses]]]
    )
    expected_scores = torch.tensor(
        [[score for score, _ in brute_force[:max_hypotheses]]]
    )

    torch.testing.assert_close(result.grid_indices, expected_indices)
    torch.testing.assert_close(result.joint_score, expected_scores)


def test_joint_prototype_hypotheses_nms_is_deterministic_and_pads_tail() -> None:
    result = generate_joint_prototype_hypotheses(
        feature_map=torch.ones(1, 1, 7, 7),
        q_nominal_px=torch.tensor([[[3.0, 3.0], [3.0, 3.0]]]),
        prototype_bank=torch.ones(2, 1, 1),
        image_size=(7, 7),
        radius_px=3.0,
        grid_size=3,
        per_anchor_topk=2,
        max_hypotheses=4,
        nms_radius_px=3.1,
    )

    torch.testing.assert_close(result.grid_indices[0, 0], torch.tensor([0, 0]))
    assert bool(result.valid[0, 0].all())
    assert bool((result.grid_indices[0, 1:] == -1).all())
    assert bool((~result.valid[0, 1:]).all())
    assert bool(torch.isneginf(result.joint_score[0, 1:]).all())


def test_online_pnp_recovers_shared_extrinsic() -> None:
    intrinsic = np.array(
        [[180.0, 0.0, 112.0], [0.0, 180.0, 112.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.array(
        [
            [-0.25, -0.10, 1.8],
            [0.15, -0.15, 2.0],
            [0.30, 0.20, 2.2],
            [-0.20, 0.25, 2.4],
            [0.05, 0.05, 1.6],
            [0.10, -0.25, 2.5],
        ],
        dtype=np.float64,
    )
    angle = math.radians(5.0)
    true_extrinsic = np.eye(4, dtype=np.float64)
    true_extrinsic[:3, :3] = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    true_extrinsic[:3, 3] = np.array([0.03, -0.02, 0.04])
    image_points, valid = project_points(
        points,
        Camera(intrinsic, true_extrinsic, 224, 224),
    )
    assert bool(valid.all())

    calibrator = OnlinePnPCalibrator(
        intrinsic,
        np.eye(4),
        image_size=(224, 224),
        history_frames=3,
        max_reprojection_error_px=1.0,
    )
    result = calibrator.update(points, image_points)

    assert result.solved
    assert result.correspondence_count == len(points)
    assert result.reprojection_error_px < 1e-3
    np.testing.assert_allclose(result.points_px, image_points, atol=1e-3)
    estimated_rvec, _ = cv2.Rodrigues(result.extrinsic_cam_from_world[:3, :3])
    true_rvec, _ = cv2.Rodrigues(true_extrinsic[:3, :3])
    np.testing.assert_allclose(estimated_rvec, true_rvec, atol=1e-4)


def test_online_pnp_keeps_nominal_pose_with_too_few_correspondences() -> None:
    intrinsic = np.eye(3, dtype=np.float64)
    nominal = np.eye(4, dtype=np.float64)
    points = np.array([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [0.0, 0.2, 2.0]])
    image_points, _ = project_points(points, Camera(intrinsic, nominal, 8, 8))
    result = OnlinePnPCalibrator(intrinsic, nominal, (8, 8)).update(points, image_points)

    assert not result.solved
    assert result.correspondence_count == 3
    np.testing.assert_allclose(result.extrinsic_cam_from_world, nominal)



def test_online_pnp_pose_prior_compares_camera_centers() -> None:
    intrinsic = np.array(
        [[180.0, 0.0, 112.0], [0.0, 180.0, 112.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.array(
        [
            [-0.30, -0.20, 2.0],
            [0.30, -0.20, 2.2],
            [-0.20, 0.25, 2.4],
            [0.05, 0.05, 1.6],
            [0.10, -0.25, 2.5],
            [0.25, 0.15, 2.1],
        ],
        dtype=np.float64,
    )
    camera_center = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    nominal = np.eye(4, dtype=np.float64)
    nominal[:3, 3] = -camera_center
    angle = math.radians(5.0)
    rotation = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    true_extrinsic = np.eye(4, dtype=np.float64)
    true_extrinsic[:3, :3] = rotation
    true_extrinsic[:3, 3] = -rotation @ camera_center
    image_points, valid = project_points(
        points,
        Camera(intrinsic, true_extrinsic, 224, 224),
    )
    assert bool(valid.all())

    calibrator = OnlinePnPCalibrator(
        intrinsic,
        nominal,
        image_size=(224, 224),
        max_reprojection_error_px=1.0,
        max_pose_rotation_deg=10.0,
        max_pose_translation_cm=0.1,
    )
    result = calibrator.update(points, image_points)

    assert result.solved
    estimated_center = (
        -result.extrinsic_cam_from_world[:3, :3].T
        @ result.extrinsic_cam_from_world[:3, 3]
    )
    np.testing.assert_allclose(estimated_center, camera_center, atol=1e-4)


def test_online_pnp_rejects_pose_outside_nominal_prior() -> None:
    intrinsic = np.array(
        [[180.0, 0.0, 112.0], [0.0, 180.0, 112.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    nominal = np.eye(4, dtype=np.float64)
    points = np.array(
        [
            [-0.30, -0.20, 2.0],
            [0.30, -0.20, 2.2],
            [-0.20, 0.25, 2.4],
            [0.05, 0.05, 1.6],
            [0.10, -0.25, 2.5],
            [0.25, 0.15, 2.1],
        ],
        dtype=np.float64,
    )
    angle = math.radians(15.0)
    outside_prior = np.eye(4, dtype=np.float64)
    outside_prior[:3, :3] = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    image_points, valid = project_points(
        points,
        Camera(intrinsic, outside_prior, 224, 224),
    )
    assert bool(valid.all())

    calibrator = OnlinePnPCalibrator(
        intrinsic,
        nominal,
        image_size=(224, 224),
        max_reprojection_error_px=1.0,
        max_pose_rotation_deg=10.0,
    )
    result = calibrator.update(points, image_points)

    assert not result.solved
    np.testing.assert_allclose(result.extrinsic_cam_from_world, nominal)


def test_history_batch_carries_external_grounding() -> None:
    record = {
        "rgb": np.zeros((3, 8, 8), dtype=np.float32),
        "obs": np.zeros(4, dtype=np.float32),
        "keypoints_xyz": np.zeros((3, 3), dtype=np.float32),
        "q_nominal": np.zeros((3, 2), dtype=np.float32),
        "q_true": np.ones((3, 2), dtype=np.float32),
        "fcpg_q_corrected": np.full((3, 2), 2.0, dtype=np.float32),
        "fcpg_reliability": np.ones(3, dtype=np.float32),
    }
    batch = history_batch(deque([record]), obs_horizon=2, device=torch.device("cpu"))

    assert batch["fcpg_q_corrected"].shape == (1, 2, 3, 2)
    torch.testing.assert_close(
        batch["fcpg_q_corrected"],
        torch.full((1, 2, 3, 2), 2.0),
    )
    assert optional_mean([{"value": None}, {"value": 2.0}], "value") == 2.0
