from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.render_lewm_action_branch_a4_trajectory_comparison import (
    _chunk_actions,
    _dense_prediction,
    _draw_frame,
    _draw_split_frame,
    _decode_open_loop,
    _model_steps,
    _path_metrics,
    _render_html,
)
from fcpg.lewm_action_branch_a4 import TASKS


def test_model_steps_follow_full_episode_instead_of_stopping_at_40() -> None:
    steps = _model_steps(72)
    np.testing.assert_array_equal(steps, np.arange(0, 71, 5))
    assert int(steps[-1]) == 70
    assert int(steps[-1]) > 40


def test_chunk_actions_preserves_every_raw_action_in_each_model_step() -> None:
    actions = np.arange(80, dtype=np.float32).reshape(20, 4)
    chunks = _chunk_actions(actions, (0, 5, 10, 15))
    assert chunks.shape == (4, 20)
    np.testing.assert_array_equal(chunks.reshape(20, 4), actions)


def test_dense_prediction_interpolates_nodes_and_holds_only_incomplete_tail() -> None:
    steps = np.asarray([10, 15, 20], dtype=np.int64)
    points = np.asarray([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]], dtype=np.float32)
    dense = _dense_prediction(steps, points, raw_frame_count=24)
    assert dense.shape == (24, 2)
    assert np.isnan(dense[:10]).all()
    np.testing.assert_allclose(dense[steps], points)
    np.testing.assert_allclose(dense[23], points[-1])


def test_path_metrics_separates_h1_h8_from_long_horizon_extrapolation() -> None:
    steps = np.arange(10, 66, 5, dtype=np.int64)
    truth = np.zeros((66, 2), dtype=np.float32)
    prediction = np.stack(
        [np.arange(steps.size, dtype=np.float32), np.zeros(steps.size)], axis=-1
    )
    result = _path_metrics(truth, steps, prediction)
    assert result["model_prediction_nodes"] == steps.size - 1
    assert result["in_training_horizon_nodes"] == 8
    assert result["extrapolation_nodes"] == steps.size - 1 - 8


def test_overlay_uses_one_rgb_with_only_gt_and_lewm_paths() -> None:
    frame = Image.new("RGB", (224, 224), (128, 128, 128))
    gt = np.stack([np.linspace(20, 180, 73), np.linspace(50, 110, 73)], axis=-1)
    pred = gt.copy()
    rendered = _draw_frame(
        frame,
        ground_truth=gt,
        predicted=pred,
        raw_step=72,
        end_step=72,
        task_label="Push",
        success=True,
    )
    assert rendered.size == frame.size
    assert np.any(np.asarray(rendered) != np.asarray(frame))


def test_split_forecast_panel_does_not_reveal_future_rgb() -> None:
    context = Image.new("RGB", (32, 32), (90, 90, 90))
    future_a = Image.new("RGB", (32, 32), (220, 20, 20))
    future_b = Image.new("RGB", (32, 32), (20, 20, 220))
    gt = np.stack([np.linspace(5, 25, 12), np.linspace(8, 20, 12)], axis=-1)
    pred = gt.copy()

    kwargs = {
        "context_frame": context,
        "ground_truth": gt,
        "predicted": pred,
        "raw_step": 11,
        "end_step": 11,
        "context_step": 10,
        "task_label": "Drawer Retrieve",
        "success": False,
    }
    rendered_a = _draw_split_frame(future_a, **kwargs)
    rendered_b = _draw_split_frame(future_b, **kwargs)

    right_x = 12 + 32 * 2 + 12
    right_box = (right_x, 66, right_x + 32 * 2, 66 + 32 * 2)
    np.testing.assert_array_equal(
        np.asarray(rendered_a.crop(right_box)),
        np.asarray(rendered_b.crop(right_box)),
    )
    assert not np.array_equal(np.asarray(rendered_a), np.asarray(rendered_b))


def test_html_states_full_frame_contract_and_failed_gate_boundary() -> None:
    page = _render_html(
        [
            {
                "task": "push-v3",
                "task_instance": 0,
                "frame_count": 73,
                "first_success_step": 62,
                "stable_success_step": 64,
                "h1_h8_mean_epe_px": 3.0,
                "h9_plus_mean_epe_px": 8.0,
                "gif": "push.gif",
            }
        ],
        {
            "target_ceiling": {"macro_mean_epe": 7.19},
            "factual": {"macro_mean_epe": 12.64},
        },
        {
            "primary": {
                "comparisons": {
                    "no_action": {"relative_improvement": 0.6412},
                    "wrong_action_mean": {"relative_improvement": 0.8091},
                }
            }
        },
        checkpoint_step=10_000,
    )
    assert "73 continuous raw frames" in page
    assert "H9+" in page
    assert "open-loop extrapolation" in page
    assert "DIAGNOSTIC - GATE FAILED - NOT PAPER EVIDENCE" in page
    assert "10,000" in page
    assert "41" not in page


def test_open_loop_decoder_selects_task_conditioned_parameters() -> None:
    history = np.zeros((3, 4, 192), dtype=np.float32)
    prediction = np.zeros((2, 4, 192), dtype=np.float32)
    coefficient = np.zeros((len(TASKS), 3 * 192, 2), dtype=np.float64)
    intercept = np.stack(
        [np.asarray([index, -index], dtype=np.float64) for index in range(len(TASKS))]
    )
    result = _decode_open_loop(
        prediction,
        history,
        np.asarray([100.0, 80.0], dtype=np.float32),
        task="push-v3",
        height=224,
        decoder={
            "task_names": np.asarray(TASKS),
            "coefficient_raw": coefficient,
            "intercept_raw": intercept,
        },
    )
    task_index = TASKS.index("push-v3")
    np.testing.assert_allclose(result[0], [100.0, 80.0])
    np.testing.assert_allclose(
        result[1:],
        np.asarray([[100.0 + task_index, 80.0 + task_index]] * 2),
    )
