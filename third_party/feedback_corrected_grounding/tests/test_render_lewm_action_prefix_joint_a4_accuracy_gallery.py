from __future__ import annotations

import numpy as np
import torch

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS, assign_a4_splits
from scripts import render_lewm_action_branch_a4_accuracy_gallery as legacy_gallery
from scripts import render_lewm_action_prefix_joint_a4_accuracy_gallery as gallery


class _CameraActionProbe:
    def __init__(self) -> None:
        self.extrinsic: torch.Tensor | None = None

    def transform_future_action(
        self,
        future_action: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        self.extrinsic = batch["camera_extrinsic_cam_from_world"].clone()
        return future_action + 1.0


def test_reserved_test_rank_selects_distinct_task_instances() -> None:
    contract = legacy_gallery._safe_read_json(legacy_gallery.DEFAULT_CONFIG)
    assignments = assign_a4_splits(TASKS, contract)
    first = legacy_gallery._test_instances_by_rank(contract, rank=0)
    second = legacy_gallery._test_instances_by_rank(contract, rank=1)
    assert set(first) == set(TASKS)
    assert set(second) == set(TASKS)
    assert all(first[task] != second[task] for task in TASKS)
    assert all(assignments[(task, second[task])] == "test" for task in TASKS)


def test_reserved_test_rank_rejects_out_of_range() -> None:
    contract = legacy_gallery._safe_read_json(legacy_gallery.DEFAULT_CONFIG)
    with np.testing.assert_raises_regex(ValueError, "rank must be in"):
        legacy_gallery._test_instances_by_rank(contract, rank=12)


def test_full_episode_gallery_applies_model_camera_action_transform() -> None:
    model = _CameraActionProbe()
    actions = torch.zeros(3, HORIZON, 20)
    extrinsic = np.eye(4, dtype=np.float32)
    transformed = gallery._transform_future_actions_for_model(
        model,
        actions,
        {"camera_extrinsic_cam_from_world": extrinsic},
    )
    torch.testing.assert_close(transformed, torch.ones_like(actions))
    assert model.extrinsic is not None
    torch.testing.assert_close(
        model.extrinsic,
        torch.from_numpy(extrinsic)[None].expand(3, -1, -1),
    )


def test_full_episode_gallery_preserves_world_actions_for_legacy_model() -> None:
    actions = torch.randn(2, HORIZON, 20)
    transformed = gallery._transform_future_actions_for_model(
        object(), actions, {}
    )
    assert transformed is actions


def test_model_camera_extrinsic_prefers_explicit_deployment_calibration() -> None:
    actual = np.eye(4, dtype=np.float32)
    actual[0, 3] = 2.0
    nominal = np.eye(4, dtype=np.float32)
    nominal[0, 3] = 1.0
    selected = gallery._model_camera_extrinsic(
        {
            "camera_extrinsic_cam_from_world": actual,
            "model_camera_extrinsic_cam_from_world": nominal,
        }
    )
    np.testing.assert_array_equal(selected, nominal)


def test_constant_transport_adds_only_current_grounding_innovation() -> None:
    predicted = torch.zeros(2, HORIZON, 3, 2)
    nominal = torch.zeros(2, 1, 3, 2)
    grounded = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            [[[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]]],
        ]
    )
    transported = gallery._transport_current_grounding_residual(
        predicted, nominal, grounded, mode="constant"
    )
    torch.testing.assert_close(
        transported,
        grounded[:, 0, None].expand(-1, HORIZON, -1, -1),
    )
    assert (
        gallery._transport_current_grounding_residual(
            predicted, nominal, grounded, mode="none"
        )
        is predicted
    )


def test_magnitude_gated_transport_suppresses_clean_and_keeps_large_drift() -> None:
    predicted = torch.zeros(2, HORIZON, 3, 2)
    nominal = torch.zeros(2, 1, 3, 2)
    grounded = torch.zeros_like(nominal)
    grounded[0, 0, :, 0] = torch.tensor([2.0, 4.0, 6.0])
    grounded[1, 0, :, 0] = torch.tensor([8.0, 10.0, 12.0])
    transported = gallery._transport_current_grounding_residual(
        predicted,
        nominal,
        grounded,
        mode="magnitude-gated",
        gate_low_px=4.0,
        gate_high_px=8.0,
    )
    expected = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]],
            [[8.0, 0.0], [10.0, 0.0], [12.0, 0.0]],
        ]
    )
    torch.testing.assert_close(
        transported,
        expected[:, None].expand(-1, HORIZON, -1, -1),
    )


def test_magnitude_gate_rejects_invalid_range() -> None:
    with np.testing.assert_raises_regex(ValueError, "high > low"):
        gallery._transport_current_grounding_residual(
            torch.zeros(1, HORIZON, 3, 2),
            torch.zeros(1, 1, 3, 2),
            torch.zeros(1, 1, 3, 2),
            mode="magnitude-gated",
            gate_low_px=4.0,
            gate_high_px=4.0,
        )


def test_grounding_mode_requires_explicit_orbit_fcpg_opt_in() -> None:
    with np.testing.assert_raises_regex(ValueError, "allow-orbit-fcpg"):
        gallery._resolve_grounding_mode(
            "fcpg", orbit_clock_hour=3, allow_orbit_fcpg=False
        )
    assert (
        gallery._resolve_grounding_mode(
            "fcpg", orbit_clock_hour=3, allow_orbit_fcpg=True
        )
        == "fcpg"
    )


def test_grounding_mode_allows_oracle_diagnostic_on_orbit_view() -> None:
    assert (
        gallery._resolve_grounding_mode(
            "oracle", orbit_clock_hour=3, allow_orbit_fcpg=False
        )
        == "oracle"
    )
    assert (
        gallery._resolve_grounding_mode(
            "auto", orbit_clock_hour=3, allow_orbit_fcpg=False
        )
        == "nominal"
    )


def test_stitch_prefers_h1_then_consumes_final_horizons() -> None:
    starts = np.asarray([0, 5, 10], dtype=np.int64)
    predictions = np.zeros((3, HORIZON, 3, 2), dtype=np.float32)
    for decision in range(3):
        for horizon in range(HORIZON):
            predictions[decision, horizon, :, 0] = 100 * decision + horizon + 1
    current = np.zeros((3, 2), dtype=np.float32)
    dense, grid_steps, source_horizons = gallery._stitch_window_predictions(
        starts, predictions, current, raw_frame_count=51
    )
    assert grid_steps.tolist() == list(range(0, 51, 5))
    assert source_horizons.tolist() == [0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]
    assert dense[5, 0, 0] == 1
    assert dense[10, 0, 0] == 101
    assert dense[15, 0, 0] == 201
    assert dense[50, 0, 0] == 208


def test_chunk_history_actions_is_causal_and_zero_pads_episode_start() -> None:
    actions = np.arange(80, dtype=np.float32).reshape(20, 4)
    history = gallery._chunk_history_actions(
        actions, np.asarray([0, 5, 10, 15], dtype=np.int64)
    ).reshape(4, 10, 4)

    np.testing.assert_array_equal(history[0], np.zeros((10, 4), dtype=np.float32))
    np.testing.assert_array_equal(history[1, :5], 0.0)
    np.testing.assert_array_equal(history[1, 5:], actions[:5])
    np.testing.assert_array_equal(history[2], actions[:10])
    np.testing.assert_array_equal(history[3], actions[5:15])


def test_stitch_max_lookahead_uses_h8_for_full_trajectory() -> None:
    starts = np.asarray([0, 5, 10], dtype=np.int64)
    predictions = np.zeros((3, HORIZON, 3, 2), dtype=np.float32)
    for decision in range(3):
        for horizon in range(HORIZON):
            predictions[decision, horizon, :, 0] = 100 * decision + horizon + 1
    current = np.zeros((3, 2), dtype=np.float32)
    dense, grid_steps, source_horizons = (
        gallery._stitch_max_lookahead_predictions(
            starts, predictions, current, raw_frame_count=51
        )
    )
    assert grid_steps.tolist() == list(range(0, 51, 5))
    assert source_horizons.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]
    assert dense[5, 0, 0] == 1
    assert dense[10, 0, 0] == 2
    assert dense[40, 0, 0] == 8
    assert dense[45, 0, 0] == 108
    assert dense[50, 0, 0] == 208


def test_display_points_flips_only_standard_tasks() -> None:
    points = np.asarray([[[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]]])
    standard = gallery._display_points(points, task="pick-place-v3", height=128)
    custom = gallery._display_points(points, task="drawer-retrieve-v0", height=128)
    np.testing.assert_allclose(standard[0, :, 1], [107.0, 97.0, 87.0])
    np.testing.assert_allclose(custom, points)


def test_display_points_can_disable_flip_for_orbit_camera() -> None:
    points = np.asarray([[[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]]])
    displayed = gallery._display_points(
        points,
        task="pick-place-v3",
        height=128,
        flip_vertical=False,
    )
    np.testing.assert_allclose(displayed, points)


def test_sample_frame_indices_preserves_full_extent() -> None:
    indices = gallery._sample_frame_indices(501, 200)
    assert len(indices) == 200
    assert indices[0] == 0
    assert indices[-1] == 500
    assert np.all(np.diff(indices) > 0)


def test_render_html_states_joint_training_and_future_contract() -> None:
    metric = {
        f"H{horizon}": {
            "anchor_mean_epe_px": 1.0,
            "anchor_p95_epe_px": 2.0,
            "center_mean_epe_px": 0.5,
            "anchor_pck_at_4": 0.9,
            "anchor_pck_at_8": 1.0,
        }
        for horizon in range(1, HORIZON + 1)
    }
    document = gallery._render_html(
        [
            {
                "task_label": "Pick & Place",
                "task_instance": 1,
                "raw_frame_count": 100,
                "rendered_frame_count": 100,
                "stable_success_step": 90,
                "current_grounding_anchor_mean_epe_px": 0.25,
                "stitched_anchor_mean_epe_px": 1.25,
                "video": "pick.mp4",
                "horizon_metrics": metric,
            }
        ],
        {"seed": 40501, "step": 2000, "sigreg_weight": 0.02},
    )
    assert "One shared model" in document
    assert "future RGB or q_true" in document
    assert "SIGReg 0.02" in document
    assert "Repeated 40-step prediction" in document
    assert "advances five real frames and pauses" in document


def test_render_html_supports_metrics_only_entries() -> None:
    metric = {
        f"H{horizon}": {
            "anchor_mean_epe_px": 1.0,
            "anchor_p95_epe_px": 2.0,
            "center_mean_epe_px": 0.5,
            "anchor_pck_at_4": 0.9,
            "anchor_pck_at_8": 1.0,
        }
        for horizon in range(1, HORIZON + 1)
    }
    document = gallery._render_html(
        [
            {
                "task_label": "Towel Wipe",
                "task_instance": 1,
                "raw_frame_count": 100,
                "rendered_frame_count": 0,
                "stable_success_step": 90,
                "current_grounding_anchor_mean_epe_px": 0.25,
                "stitched_anchor_mean_epe_px": 1.25,
                "video": "",
                "gif": "",
                "gif_frame_count": 0,
                "horizon_metrics": metric,
            }
        ],
        {"seed": 40501, "step": 2000, "sigreg_weight": 0.02},
    )
    assert "Metrics-only run" in document
    assert "<video" not in document


def test_history_protocol_is_supported_by_gallery_loader_contract() -> None:
    assert gallery.HISTORY_CONDITIONED_PROTOCOL == (
        "a4_metric3d_world_action_history_v1"
    )


def test_interpolate_prediction_window_produces_40_smooth_steps() -> None:
    current = np.zeros((3, 2), dtype=np.float32)
    prediction = np.zeros((HORIZON, 3, 2), dtype=np.float32)
    for horizon in range(HORIZON):
        prediction[horizon, :, 0] = float((horizon + 1) * 5)
    dense = gallery._interpolate_prediction_window(current, prediction)
    assert dense.shape == (41, 3, 2)
    np.testing.assert_allclose(dense[:, 0, 0], np.arange(41, dtype=np.float32))


def test_prediction_cycle_advances_five_then_forecasts_forty() -> None:
    episode = {
        "frames": [gallery.Image.new("RGB", (16, 16)) for _ in range(46)],
        "successes": np.zeros(46, dtype=bool),
    }
    starts = np.asarray([0, 5], dtype=np.int64)
    corrected = np.zeros((2, 3, 2), dtype=np.float32)
    predictions = np.zeros((2, HORIZON, 3, 2), dtype=np.float32)
    truth = np.zeros((46, 3, 2), dtype=np.float32)
    frames = list(
        gallery._prediction_cycle_frames(
            episode,
            starts,
            corrected,
            predictions,
            truth,
            task_label="Pick",
        )
    )
    assert len(frames) == 40 + 5 + 40 + 40


def test_prediction_cycle_frame_count_includes_final_actual_catchup() -> None:
    starts = np.asarray([0, 5], dtype=np.int64)
    assert gallery._prediction_cycle_frame_count(starts, end_step=45) == 125
