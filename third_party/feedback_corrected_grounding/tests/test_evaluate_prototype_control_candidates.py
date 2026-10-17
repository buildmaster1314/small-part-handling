from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_prototype_control_candidates import (
    CandidatePool,
    ControlData,
    _json_ready,
    _mode_rows,
    action_distance_metrics,
    build_deployable_base_batch,
    crossed_paired_relative_reduction,
    crossed_policy_episode_summary,
    generate_random_joint_pool,
    history_grounding_points,
    infer_policy_actions,
    output_argument_is_explicit,
    paired_mode_comparison,
    select_evaluation_indices,
    select_candidate_indices,
    summarize_candidate_coverage,
    validate_artifacts,
    validate_policy_checkpoint_tasks,
    validate_task_output_contract,
)


class _PointSensitivePolicy:
    def __init__(self) -> None:
        self.mode = "fcpg"
        self.seen: list[dict[str, torch.Tensor]] = []

    def predict_action_chunk(self, batch):
        assert "q_true" not in batch
        assert not any(key.startswith("next_") for key in batch)
        self.seen.append(batch)
        points = batch["fcpg_q_corrected"]
        value = points[:, -1, 0, 0] / 10.0
        action = torch.zeros(len(points), 8, 4, device=points.device)
        action[..., 0] = value[:, None]
        return SimpleNamespace(action=action, grounding_points_px=points)


def _base_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.zeros(batch_size, 2, 4),
        "keypoints_xyz": torch.zeros(batch_size, 2, 3, 3),
        "q_nominal": torch.zeros(batch_size, 2, 3, 2),
        "valid": torch.ones(batch_size, 2, 3, dtype=torch.bool),
        "vision_backbone_features": torch.zeros(batch_size, 2, 1, 4, 2, 2),
        "vision_prefix": torch.zeros(batch_size, 2, 4),
        "image_size": torch.tensor([[32, 32]]).expand(batch_size, -1),
    }


def test_history_grounding_fixes_past_and_changes_only_current() -> None:
    past = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    current = torch.tensor(
        [
            [
                [[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]],
                [[20.0, 0.0], [21.0, 0.0], [22.0, 0.0]],
            ]
        ]
    )

    result = history_grounding_points(past, current)

    assert result.shape == (2, 2, 3, 2)
    torch.testing.assert_close(result[:, 0], past.expand(2, -1, -1))
    torch.testing.assert_close(result[:, 1], current.reshape(2, 3, 2))


def test_policy_intervention_never_exposes_q_true_and_preserves_candidate_order() -> None:
    model = _PointSensitivePolicy()
    past = torch.ones(1, 3, 2)
    current = torch.zeros(1, 3, 3, 2)
    current[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])

    action = infer_policy_actions(
        model,
        _base_batch(),
        past_wide=past,
        current_points=current,
        candidate_chunk_size=2,
        device_type="cpu",
    )

    assert action.shape == (1, 3, 8, 4)
    np.testing.assert_allclose(action[0, :, 0, 0], [0.1, 0.2, 0.3])
    assert len(model.seen) == 2
    for batch in model.seen:
        torch.testing.assert_close(
            batch["fcpg_q_corrected"][:, 0],
            past.expand(len(batch["fcpg_q_corrected"]), -1, -1),
        )


def test_policy_intervention_rejects_offline_label_keys() -> None:
    batch = _base_batch()
    batch["q_true"] = torch.zeros(1, 2, 3, 2)

    with pytest.raises(ValueError, match="offline label"):
        infer_policy_actions(
            _PointSensitivePolicy(),
            batch,
            past_wide=torch.zeros(1, 3, 2),
            current_points=torch.zeros(1, 1, 3, 2),
            candidate_chunk_size=1,
            device_type="cpu",
        )


def test_deployable_batch_uses_nominal_validity_not_true_validity() -> None:
    data = _control_data()
    data = ControlData(
        **{
            **data.__dict__,
            "valid_nominal": np.asarray([[True, False, True]]),
            "valid_true": np.asarray([[False, True, False]]),
        }
    )
    backbone = np.zeros((1, 1, 4, 2, 2), dtype=np.float16)
    prefix = np.zeros((1, 4), dtype=np.float16)

    batch = build_deployable_base_batch(
        data,
        indices=np.asarray([0]),
        obs_mode="robot",
        backbone=backbone,
        prefix=prefix,
        device=torch.device("cpu"),
    )

    assert "q_true" not in batch
    torch.testing.assert_close(
        batch["valid"],
        torch.tensor([[[True, False, True], [True, False, True]]]),
    )


def test_action_metrics_report_raw_and_checkpoint_standardized_distances() -> None:
    teacher = np.zeros((1, 8, 4), dtype=np.float32)
    actions = np.zeros((1, 2, 8, 4), dtype=np.float32)
    actions[:, 0, :, :3] = [1.0, 2.0, 2.0]
    actions[:, 0, :, 3] = 1.0
    stats = np.asarray([1.0, 2.0, 4.0, 0.5])

    metrics = action_distance_metrics(actions, teacher, stats, horizon=8)

    assert metrics["xyz_l2"][0, 0] == pytest.approx(3.0)
    assert metrics["xyz_normalized_mse"][0, 0] == pytest.approx(
        (1.0 + 1.0 + 0.25) / 3.0
    )
    assert metrics["gripper_l1"][0, 0] == pytest.approx(1.0)
    assert metrics["gripper_normalized_mse"][0, 0] == pytest.approx(4.0)
    assert metrics["gripper_disagreement"][0, 0] == pytest.approx(1.0)
    assert np.all(metrics["xyz_l2"][0, 1] == 0.0)


def test_candidate_oracles_exclude_invalid_and_enforce_capture_threshold() -> None:
    selected = select_candidate_indices(
        candidate_valid=np.asarray([[True, False, True], [True, True, True]]),
        epe=np.asarray([[2.0, 0.0, 3.0], [9.0, 5.0, 7.0]]),
        max_error=np.asarray([[3.0, 0.0, 4.0], [12.0, 6.0, 9.0]]),
        normalized_h8=np.asarray([[2.0, 0.0, 1.0], [0.1, 0.3, 0.2]]),
        capture_threshold_px=8.0,
    )

    np.testing.assert_array_equal(selected["geometry_oracle"], [0, 1])
    np.testing.assert_array_equal(selected["joint_candidate_oracle"], [2, 0])
    np.testing.assert_array_equal(
        selected["capture_constrained_control_oracle"], [2, 1]
    )


def _control_data() -> ControlData:
    q_true = np.asarray(
        [[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]], dtype=np.float32
    )
    return ControlData(
        obs=np.zeros((1, 39), dtype=np.float32),
        action=np.zeros((1, 4), dtype=np.float32),
        xyz=np.zeros((1, 3, 3), dtype=np.float32),
        q_nominal=q_true + 4.0,
        q_true=q_true,
        valid_nominal=np.ones((1, 3), dtype=bool),
        valid_true=np.ones((1, 3), dtype=bool),
        task=np.asarray(["pick-place-v3"]),
        episode=np.asarray([7]),
        frame=np.asarray([3]),
        previous_index=np.asarray([0]),
        rotation_deg=np.asarray([10.0], dtype=np.float32),
        translation_xyz=np.zeros((1, 3), dtype=np.float32),
        image_size=(64, 64),
    )


def test_task_selection_requires_explicit_choice_for_shared_dataset() -> None:
    single = _control_data()
    data = ControlData(
        **{
            **single.__dict__,
            "obs": np.repeat(single.obs, 4, axis=0),
            "action": np.repeat(single.action, 4, axis=0),
            "xyz": np.repeat(single.xyz, 4, axis=0),
            "q_nominal": np.repeat(single.q_nominal, 4, axis=0),
            "q_true": np.repeat(single.q_true, 4, axis=0),
            "valid_nominal": np.repeat(single.valid_nominal, 4, axis=0),
            "valid_true": np.repeat(single.valid_true, 4, axis=0),
            "task": np.asarray(
                ["pick-place-v3", "pick-place-v3", "button-press-v3", "button-press-v3"]
            ),
            "episode": np.asarray([0, 0, 0, 0]),
            "frame": np.asarray([0, 1, 0, 1]),
            "previous_index": np.asarray([0, 0, 2, 2]),
            "rotation_deg": np.full(4, 10.0, dtype=np.float32),
            "translation_xyz": np.zeros((4, 3), dtype=np.float32),
        }
    )

    with pytest.raises(ValueError, match="explicit --task"):
        select_evaluation_indices(data, task="", max_frames=0)
    np.testing.assert_array_equal(
        select_evaluation_indices(data, task="button-press-v3", max_frames=0),
        [2, 3],
    )
    np.testing.assert_array_equal(
        select_evaluation_indices(data, task="pick-place-v3", max_frames=1),
        [0],
    )
    with pytest.raises(ValueError, match="is absent"):
        select_evaluation_indices(data, task="push-v3", max_frames=0)


def test_policy_checkpoints_must_include_selected_training_task(tmp_path) -> None:
    pick = tmp_path / "pick.npz"
    shared = tmp_path / "shared.npz"
    np.savez(pick, task_name=np.asarray(["pick-place-v3"]))
    np.savez(
        shared,
        task_name=np.asarray(["pick-place-v3", "button-press-v3"]),
    )

    with pytest.raises(ValueError, match="was not trained on 'button-press-v3'"):
        validate_policy_checkpoint_tasks(
            [{"data": str(pick), "extra_data": []}],
            evaluation_task="button-press-v3",
        )

    validate_policy_checkpoint_tasks(
        [{"data": str(pick), "extra_data": [str(shared)]}],
        evaluation_task="button-press-v3",
    )


def test_non_pick_task_requires_explicit_output_path() -> None:
    assert not output_argument_is_explicit(["--task", "button-press-v3"])
    assert output_argument_is_explicit(["--output", "button.json"])
    assert output_argument_is_explicit(["--output=button.json"])
    validate_task_output_contract(
        selected_task="pick-place-v3",
        output_is_explicit=False,
    )
    validate_task_output_contract(
        selected_task="button-press-v3",
        output_is_explicit=True,
    )
    with pytest.raises(ValueError, match="requires explicit --output"):
        validate_task_output_contract(
            selected_task="button-press-v3",
            output_is_explicit=False,
        )


def test_primary_prototype_bank_requires_matching_semantic_control_sites(tmp_path) -> None:
    data_path = tmp_path / "evaluation.npz"
    sites = np.asarray(["leftEndEffector", "rightEndEffector", "endEffector"])
    np.savez(data_path, keypoint_site=sites, control_keypoint_site=sites)
    prototype_source = tmp_path / "independent_train.npz"
    np.savez(
        prototype_source,
        task_name=np.asarray(["button-press-v3"]),
        keypoint_site=sites,
    )
    cache = tmp_path / "cache"
    bank = tmp_path / "bank"
    cache.mkdir()
    bank.mkdir()
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(data_path),
                "model_name": "vit",
                "selected_layers": [11],
                "image_width": 224,
                "image_height": 224,
            }
        ),
        encoding="utf-8",
    )
    prototype_metadata = {
        "source_data": str(prototype_source),
        "keypoint_source": "primary",
        "keypoint_site": sites.tolist(),
        "num_keypoints": 3,
        "model_name": "vit",
        "selected_layers": [11],
        "image_width": 224,
        "image_height": 224,
    }
    (bank / "metadata.json").write_text(
        json.dumps(prototype_metadata), encoding="utf-8"
    )
    policy_args = [
        {
            "vision_model_name": "vit",
            "vision_selected_layers": [11],
            "obs_horizon": 2,
            "action_horizon": 8,
            "mode": "fcpg",
        }
    ]

    validate_artifacts(
        data_path=data_path,
        vision_cache=cache,
        prototype_bank=bank,
        policy_args=policy_args,
        evaluation_tasks=("button-press-v3",),
    )

    prototype_metadata["keypoint_site"] = sites[::-1].tolist()
    (bank / "metadata.json").write_text(
        json.dumps(prototype_metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metadata keypoint sites do not match"):
        validate_artifacts(
            data_path=data_path,
            vision_cache=cache,
            prototype_bank=bank,
            policy_args=policy_args,
            evaluation_tasks=("button-press-v3",),
        )


def test_prototype_provenance_backfills_sites_but_rejects_wrong_task_or_missing_sites(
    tmp_path,
) -> None:
    data_path = tmp_path / "evaluation.npz"
    source_path = tmp_path / "prototype_source.npz"
    sites = np.asarray(["leftEndEffector", "rightEndEffector", "endEffector"])
    np.savez(data_path, control_keypoint_site=sites)
    np.savez(
        source_path,
        task_name=np.asarray(["button-press-v3"]),
        control_keypoint_site=sites,
    )
    cache = tmp_path / "cache"
    bank = tmp_path / "bank"
    cache.mkdir()
    bank.mkdir()
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "source_data": str(data_path),
                "model_name": "vit",
                "selected_layers": [11],
                "image_width": 224,
                "image_height": 224,
            }
        ),
        encoding="utf-8",
    )
    prototype_metadata = {
        "source_data": str(source_path),
        "keypoint_source": "control",
        "num_keypoints": 3,
        "model_name": "vit",
        "selected_layers": [11],
        "image_width": 224,
        "image_height": 224,
    }
    (bank / "metadata.json").write_text(
        json.dumps(prototype_metadata), encoding="utf-8"
    )
    policy_args = [
        {
            "vision_model_name": "vit",
            "vision_selected_layers": [11],
            "obs_horizon": 2,
            "action_horizon": 8,
            "mode": "fcpg",
        }
    ]

    _, resolved_metadata = validate_artifacts(
        data_path=data_path,
        vision_cache=cache,
        prototype_bank=bank,
        policy_args=policy_args,
        evaluation_tasks=("button-press-v3",),
    )
    assert resolved_metadata["keypoint_site"] == sites.tolist()
    assert resolved_metadata["training_tasks"] == ["button-press-v3"]

    np.savez(
        source_path,
        task_name=np.asarray(["pick-place-v3"]),
        control_keypoint_site=sites,
    )
    with pytest.raises(ValueError, match="was not trained on the selected task"):
        validate_artifacts(
            data_path=data_path,
            vision_cache=cache,
            prototype_bank=bank,
            policy_args=policy_args,
            evaluation_tasks=("button-press-v3",),
        )

    np.savez(source_path, task_name=np.asarray(["button-press-v3"]))
    with pytest.raises(ValueError, match="lacks semantic keypoint sites"):
        validate_artifacts(
            data_path=data_path,
            vision_cache=cache,
            prototype_bank=bank,
            policy_args=policy_args,
            evaluation_tasks=("button-press-v3",),
        )


def test_mode_rows_select_control_and_geometry_oracles_and_exact_random_k() -> None:
    data = _control_data()
    candidates = CandidatePool(
        points=np.stack([data.q_true[0], data.q_true[0] + 6.0])[None],
        score=np.asarray([[2.0, 1.0]], dtype=np.float32),
        valid=np.ones((1, 2, 3), dtype=bool),
        grid_indices=np.zeros((1, 2, 3), dtype=np.int64),
    )
    random_candidates = CandidatePool(
        points=np.stack([data.q_true[0] + 2.0, data.q_true[0] + 4.0])[None],
        score=np.zeros((1, 2), dtype=np.float32),
        valid=np.ones((1, 2, 3), dtype=bool),
        grid_indices=np.ones((1, 2, 3), dtype=np.int64),
    )
    # [visual candidate0/1, random candidate0/1, nominal, native, teacher]
    actions = np.zeros((1, 7, 8, 4), dtype=np.float32)
    actions[:, 1, :, 0] = 1.0
    actions[:, 2, :, 0] = 0.2
    actions[:, 3, :, 0] = 0.4
    actions[:, 4, :, 0] = 2.0
    actions[:, 5, :, 0] = 0.5
    actions[:, :, :, 3] = 0.25

    rows, diagnostics = _mode_rows(
        data=data,
        indices=np.asarray([0]),
        policy_seed=2,
        candidates=candidates,
        random_candidates=random_candidates,
        native_points=data.q_true + 1.0,
        actions=actions,
        action_std=np.ones(4),
        capture_threshold_px=8.0,
    )

    by_mode = {row["mode"]: row for row in rows}
    assert by_mode["geometry_oracle"]["selected_candidate_index"] == 0
    assert by_mode["joint_candidate_oracle"]["selected_candidate_index"] == 0
    assert by_mode["capture_constrained_control_oracle"]["selected_candidate_index"] == 0
    assert by_mode["best_of_three_control_oracle"]["action_xyz_l2_h8"] == 0.0
    assert by_mode["uniform_visual_candidate_expectation"]["action_xyz_l2_h8"] == pytest.approx(0.5)
    assert by_mode["uniform_visual_candidate_expectation"]["epe_px"] == pytest.approx(
        3.0 * np.sqrt(2.0)
    )
    assert by_mode["random_k_uniform_expectation"]["action_xyz_l2_h8"] == pytest.approx(0.3)
    assert diagnostics[0]["coverage"] is True
    assert diagnostics[0]["valid_candidate_count"] == 2


def test_random_joint_bank_is_deterministic_unique_and_uses_same_grid() -> None:
    data = _control_data()

    first = generate_random_joint_pool(
        data,
        indices=np.asarray([0]),
        radius_px=8.0,
        grid_size=3,
        candidate_count=8,
        seed=19,
    )
    second = generate_random_joint_pool(
        data,
        indices=np.asarray([0]),
        radius_px=8.0,
        grid_size=3,
        candidate_count=8,
        seed=19,
    )

    np.testing.assert_array_equal(first.grid_indices, second.grid_indices)
    np.testing.assert_allclose(first.points, second.points)
    assert len(np.unique(first.grid_indices[0], axis=0)) == 8
    offsets = first.points[0] - data.q_nominal[0]
    assert set(np.unique(offsets).tolist()).issubset({-8.0, 0.0, 8.0})


def test_crossed_bootstrap_uses_shared_episode_support_and_is_reproducible() -> None:
    values = {
        0: {("pick", 1): 1.0, ("pick", 2): 3.0},
        1: {("pick", 1): 2.0, ("pick", 2): 4.0},
    }

    first = crossed_policy_episode_summary(
        values, bootstrap_samples=5_000, bootstrap_seed=13
    )
    second = crossed_policy_episode_summary(
        values, bootstrap_samples=5_000, bootstrap_seed=13
    )

    assert first == second
    assert first["estimate"] == pytest.approx(2.5)
    assert first["episodes"] == 2
    assert len(first["crossed_bootstrap_95_ci"]) == 2

    with pytest.raises(ValueError, match="identical episode keys"):
        crossed_policy_episode_summary(
            {0: {("pick", 1): 1.0}, 1: {("pick", 2): 2.0}},
            bootstrap_samples=10,
            bootstrap_seed=0,
        )


def test_paired_comparison_reports_delta_reduction_and_requires_exact_keys() -> None:
    rows = []
    for seed in (0, 1, 2):
        for episode in (4, 5):
            for mode, value in (("candidate", 1.0), ("baseline", 2.0)):
                rows.append(
                    {
                        "policy_seed": seed,
                        "task": "pick",
                        "episode": episode,
                        "frame": 3,
                        "mode": mode,
                        "metric": value,
                    }
                )

    result = paired_mode_comparison(
        rows,
        candidate_mode="candidate",
        baseline_mode="baseline",
        metric="metric",
        bootstrap_samples=2_000,
        bootstrap_seed=5,
    )

    assert result["candidate_minus_baseline"]["estimate"] == pytest.approx(-1.0)
    assert result["relative_reduction"]["estimate"] == pytest.approx(0.5)
    assert result["candidate_minus_baseline"]["improving_policy_seeds"] == 3
    assert result["relative_reduction"]["improving_policy_seeds"] == 3

    with pytest.raises(ValueError, match="paired mode keys differ"):
        paired_mode_comparison(
            rows[:-1],
            candidate_mode="candidate",
            baseline_mode="baseline",
            metric="metric",
            bootstrap_samples=0,
            bootstrap_seed=0,
        )


def test_relative_reduction_is_ratio_of_aggregate_means_not_mean_of_ratios() -> None:
    candidate = {0: {("pick", 1): 0.0, ("pick", 2): 99.0}}
    baseline = {0: {("pick", 1): 0.01, ("pick", 2): 100.0}}

    result = crossed_paired_relative_reduction(
        candidate,
        baseline,
        bootstrap_samples=0,
        bootstrap_seed=0,
    )

    expected = (100.01 - 99.0) / 100.01
    assert result["estimate"] == pytest.approx(expected)
    assert result["estimate"] != pytest.approx((1.0 + 0.01) / 2.0)
    assert result["candidate_aggregate_mean"] == pytest.approx(49.5)
    assert result["baseline_aggregate_mean"] == pytest.approx(50.005)


def test_candidate_coverage_clusters_frames_by_episode() -> None:
    rows = [
        {"task": "pick", "episode": 1, "coverage": True},
        {"task": "pick", "episode": 1, "coverage": False},
        {"task": "pick", "episode": 2, "coverage": True},
        {"task": "pick", "episode": 2, "coverage": True},
    ]

    summary = summarize_candidate_coverage(
        rows, bootstrap_samples=1_000, bootstrap_seed=2
    )

    assert summary["snapshot_coverage"] == pytest.approx(0.75)
    assert summary["episode_macro_coverage"] == pytest.approx(0.75)
    assert summary["episode_coverage_min"] == pytest.approx(0.5)
    assert summary["episode_coverage_median"] == pytest.approx(0.75)
    assert summary["episode_coverage_max"] == pytest.approx(1.0)
    assert len(summary["episode_cluster_bootstrap_95_ci"]) == 2


def test_strict_json_conversion_replaces_nonfinite_values() -> None:
    converted = _json_ready(
        {"finite": np.float32(1.5), "nan": float("nan"), "inf": np.inf}
    )

    assert converted == {"finite": 1.5, "nan": None, "inf": None}
