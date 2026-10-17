from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from torch import nn

from scripts.evaluate_p1_native_policy_rollout import (
    _apply_action_residual_head,
    _corrupt_record,
    _emit_rollout_trace,
    _exact_mcnemar,
    _maximum_success_streak,
    _trace_rgb_uint8,
    _validate_config,
    _validate_pairing,
    clustered_seed_macro_comparison,
    paired_comparison,
)


def test_rollout_trace_is_detached_and_display_ready() -> None:
    record = {
        "rgb": np.asarray(
            [
                [[0.0, 0.5], [1.0, 2.0]],
                [[1.0, 0.0], [0.5, -1.0]],
                [[0.25, 0.75], [0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        "q_nominal": np.zeros((3, 2), dtype=np.float32),
        "q_true": np.ones((3, 2), dtype=np.float32),
        "keypoints_xyz": np.ones((3, 3), dtype=np.float32),
    }
    display = _trace_rgb_uint8(record)
    assert display.shape == (2, 2, 3)
    assert display.dtype == np.uint8
    assert display[1, 1].tolist() == [255, 0, 255]

    emitted: list[Mapping[str, object]] = []
    selected = np.full((3, 2), 2.0, dtype=np.float32)
    _emit_rollout_trace(
        emitted.append,
        raw_step=4,
        decision=1,
        substep=3,
        corruption_active=True,
        prior_active=True,
        clean_record=record,
        observed_record=record,
        selected_points=selected,
        action=np.ones(4, dtype=np.float32),
        reward=2.0,
        cumulative_reward=5.0,
        success=1.0,
        moving_goal_position=np.ones(3, dtype=np.float32),
    )
    selected.fill(9.0)
    record["q_true"].fill(8.0)
    assert len(emitted) == 1
    assert emitted[0]["raw_step"] == 4
    assert bool(emitted[0]["prior_active"])
    assert np.all(np.asarray(emitted[0]["selected_points"]) == 2.0)
    assert np.all(np.asarray(emitted[0]["q_true"]) == 1.0)


class _LegacyResidualHead(nn.Module):
    def forward(
        self,
        *,
        base_action: torch.Tensor,
        state_latent: torch.Tensor,
        action_latent: torch.Tensor,
        missing_age: torch.Tensor,
        sensor_valid: torch.Tensor,
    ) -> torch.Tensor:
        del state_latent, action_latent, missing_age, sensor_valid
        return base_action


class _PosteriorResidualHead(_LegacyResidualHead):
    def forward(
        self,
        *,
        base_action: torch.Tensor,
        state_latent: torch.Tensor,
        action_latent: torch.Tensor,
        posterior_token: torch.Tensor,
        missing_age: torch.Tensor,
        sensor_valid: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.equal(state_latent, posterior_token.mean(dim=1))
        return super().forward(
            base_action=base_action,
            state_latent=state_latent,
            action_latent=action_latent,
            missing_age=missing_age,
            sensor_valid=sensor_valid,
        )


@pytest.mark.parametrize("head", [_LegacyResidualHead(), _PosteriorResidualHead()])
def test_action_residual_head_bridge_supports_legacy_and_posterior_inputs(
    head: nn.Module,
) -> None:
    base = torch.zeros(1, 5, 4)
    result = _apply_action_residual_head(
        head,
        base_action=base,
        action_latent=torch.zeros(1, 5, 8),
        posterior_token=torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
        missing_age=2,
    )
    assert torch.equal(result, base)


def test_maximum_success_streak_tracks_sustained_success() -> None:
    assert _maximum_success_streak([]) == 0
    assert _maximum_success_streak([0.0, 1.0, 1.0, 0.0, 1.0]) == 2
    assert _maximum_success_streak([1.0, 1.0, 1.0]) == 3


def test_corrupt_record_masks_all_q_true_anchors_without_mutating_source() -> None:
    clean = {
        "rgb": np.ones((3, 12, 14), dtype=np.float32),
        "q_true": np.asarray([[3.0, 4.0], [9.0, 7.0]], dtype=np.float32),
        "obs": np.ones(4, dtype=np.float32),
    }
    corrupted = _corrupt_record(clean, radius_px=1, value_uint8=0)
    assert np.all(clean["rgb"] == 1.0)
    assert np.all(corrupted["rgb"][:, 3:6, 2:5] == 0.0)
    assert np.all(corrupted["rgb"][:, 6:9, 8:11] == 0.0)
    assert np.array_equal(corrupted["obs"], clean["obs"])


def test_corrupt_record_can_use_deployable_nominal_centers() -> None:
    clean = {
        "rgb": np.ones((3, 12, 14), dtype=np.float32),
        "q_true": np.asarray([[2.0, 2.0]], dtype=np.float32),
        "q_nominal": np.asarray([[10.0, 8.0]], dtype=np.float32),
    }
    corrupted = _corrupt_record(
        clean,
        radius_px=1,
        value_uint8=0,
        center_key="q_nominal",
    )
    assert np.all(corrupted["rgb"][:, 7:10, 9:12] == 0.0)
    assert np.all(corrupted["rgb"][:, 1:4, 1:4] == 1.0)


def _row(episode: int, method: str, success: float, distance: float) -> dict[str, object]:
    return {
        "physical_episode": episode,
        "method": method,
        "success": success,
        "final_success": success,
        "success_fraction": 0.5 * success,
        "maximum_success_streak": int(3 * success),
        "reward_sum": 10.0 + success,
        "action_distance": distance,
        "reset_digest": "a" * 64,
        "camera_sha256": "b" * 64,
        "schedule_sha256": "c" * 64,
        "drop_onset_state_sha256": "0" * 64,
        "pre_drop_action_round6_sha256": "d" * 64,
        "pre_drop_native_tail_round6_sha256": "e" * 64,
        "pre_drop_visual_tail_round6_sha256": "f" * 64,
        "pre_drop_action_values": [0.1, 0.2],
        "recovery_exact_native_bypass": True,
    }


def test_paired_comparison_uses_physical_episode_pairs() -> None:
    rows = [
        _row(0, "candidate", 1.0, 0.2),
        _row(0, "baseline", 0.0, 0.4),
        _row(1, "candidate", 1.0, 0.1),
        _row(1, "baseline", 1.0, 0.2),
    ]
    result = paired_comparison(
        rows,
        candidate="candidate",
        baseline="baseline",
        bootstrap_samples=100,
        seed=7,
    )
    assert result["physical_episode_count"] == 2
    assert result["success_delta"] == 0.5
    assert result["final_success_delta"] == 0.5
    assert result["success_fraction_delta"] == 0.25
    assert result["maximum_success_streak_delta"] == 1.5
    assert result["action_distance_relative_improvement"] == pytest.approx(0.5)
    assert result["mcnemar"]["candidate_only_success"] == 1


def test_seed_macro_comparison_clusters_models_by_physical_episode() -> None:
    rows = []
    for episode in range(2):
        rows.append(_row(episode, "geometry_hold", 0.0, 0.4))
        for seed, success, distance in (
            (11, 1.0, 0.2),
            (12, float(episode), 0.1),
        ):
            rows.append(
                _row(episode, f"complementary_factual_s{seed}", success, distance)
            )
            rows.append(
                _row(episode, f"complementary_no_action_s{seed}", 0.0, 0.3)
            )
    result = clustered_seed_macro_comparison(
        rows,
        model_seeds=[11, 12],
        baseline="geometry_hold",
        bootstrap_samples=100,
        seed=7,
    )
    assert result["physical_episode_count"] == 2
    assert result["model_seed_count"] == 2
    assert result["success_delta"] == pytest.approx(0.75)
    assert result["action_distance_relative_improvement"] == pytest.approx(0.625)


def test_exact_mcnemar_is_two_sided_and_pairing_fails_closed() -> None:
    result = _exact_mcnemar(
        np.asarray([1.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0]),
    )
    assert result["discordant"] == 2
    assert result["two_sided_exact_p"] == 0.5

    rows = [_row(0, "a", 1.0, 0.1), _row(0, "b", 0.0, 0.2)]
    rows[1]["pre_drop_action_values"] = [0.1, 0.3]
    with pytest.raises(RuntimeError, match="pre-drop actions differ"):
        _validate_pairing(rows)


def test_config_rejects_horizon_without_complete_action_blocks() -> None:
    config = {
        "schema_version": 1,
        "protocol": "fcpg_p1_native_known_drop_policy_development_v1",
        "physical_episode_count": 1,
        "task_instance_indices": [0],
        "drop_start_decisions": [6],
        "drop_duration_decisions": 6,
        "horizon_raw_steps": 334,
        "action_steps_per_decision": 5,
        "camera_condition": "nominal",
        "known_sensor_validity_only": True,
    }
    with pytest.raises(ValueError, match="complete frozen five-action"):
        _validate_config(config)


def test_joint_drift_drop_config_accepts_fixed_camera_rotation() -> None:
    config = {
        "schema_version": 1,
        "protocol": "fcpg_p1_native_joint_drift_drop_screen_v1",
        "physical_episode_count": 1,
        "task_instance_indices": [0],
        "drop_start_decisions": [6],
        "drop_duration_decisions": 6,
        "horizon_raw_steps": 335,
        "action_steps_per_decision": 5,
        "camera_condition": "fixed_rotation",
        "camera_rotation_axis_world": [0.0, 0.0, 1.0],
        "camera_rotation_deg": 5.0,
        "known_sensor_validity_only": True,
    }
    _validate_config(config)


def test_joint_drift_drop_config_rejects_zero_camera_axis() -> None:
    config = {
        "schema_version": 1,
        "protocol": "fcpg_p1_native_joint_drift_drop_screen_v1",
        "physical_episode_count": 1,
        "task_instance_indices": [0],
        "drop_start_decisions": [6],
        "drop_duration_decisions": 6,
        "horizon_raw_steps": 335,
        "action_steps_per_decision": 5,
        "camera_condition": "fixed_rotation",
        "camera_rotation_axis_world": [0.0, 0.0, 0.0],
        "camera_rotation_deg": 5.0,
        "known_sensor_validity_only": True,
    }
    with pytest.raises(ValueError, match="non-zero 3-vector"):
        _validate_config(config)


def test_joint_confirmation_config_requires_frozen_panel() -> None:
    config = {
        "schema_version": 1,
        "protocol": "fcpg_p1_native_joint_drift_drop_confirmation_v1",
        "classification": "frozen_confirmation_not_policy_success",
        "physical_episode_count": 30,
        "task_instance_indices": list(range(30)),
        "drop_start_decisions": [6] * 30,
        "drop_duration_decisions": 6,
        "horizon_raw_steps": 335,
        "action_steps_per_decision": 5,
        "camera_condition": "fixed_rotation",
        "camera_rotation_axis_world": [0.0, 0.0, 1.0],
        "camera_rotation_deg": 5.0,
        "known_sensor_validity_only": True,
        "methods": [
            "clean_native",
            "measurement",
            "geometry_hold",
            "complementary_factual",
            "complementary_no_action",
        ],
        "full_model_seeds_after_development_go": [15801, 15802, 15803],
        "confirmation_gate": {
            "minimum_relative_improvement": 0.10,
            "primary_ci_lower_must_exceed": 0.0,
        },
    }
    _validate_config(config)

    config["task_instance_indices"] = list(range(29)) + [31]
    with pytest.raises(ValueError, match="30 fresh task-instance"):
        _validate_config(config)
