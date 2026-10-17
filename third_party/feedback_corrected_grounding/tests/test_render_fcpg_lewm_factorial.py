from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from scripts.render_fcpg_lewm_factorial import (
    CELL_ORDER,
    MATCH_METRICS,
    _render_frame,
    _validate_replay_contract,
    select_representative_episodes,
)


def _row(
    episode: int,
    cell: str,
    *,
    sustained: float,
    streak: float,
    reward: float,
) -> dict[str, object]:
    return {
        "physical_episode": episode,
        "method": cell,
        "success": sustained,
        "sustained_success": sustained,
        "final_success": sustained,
        "post_drop_success_fraction": sustained,
        "post_drop_maximum_success_streak": streak,
        "reward_sum": reward,
    }


def test_representative_selection_covers_gain_and_deficit() -> None:
    rows: list[dict[str, object]] = []
    values = {
        0: {
            "f0_l0": (0, 0, 0),
            "f1_l0": (0, 1, 1),
            "f0_l1": (0, 0, 0),
            "f1_l1": (1, 9, 9),
        },
        1: {
            "f0_l0": (0, 0, 0),
            "f1_l0": (1, 12, 12),
            "f0_l1": (0, 0, 0),
            "f1_l1": (1, 12, 12),
        },
        2: {
            "f0_l0": (0, 0, 0),
            "f1_l0": (1, 9, 9),
            "f0_l1": (0, 0, 0),
            "f1_l1": (0, 1, 1),
        },
    }
    for episode, panel in values.items():
        for cell, (sustained, streak, reward) in panel.items():
            rows.append(
                _row(
                    episode,
                    cell,
                    sustained=sustained,
                    streak=streak,
                    reward=reward,
                )
            )
    selected = select_representative_episodes(rows, 3)
    assert [episode for episode, _ in selected] == [0, 1, 2]
    assert len({episode for episode, _ in selected}) == 3


def test_rerun_metric_check_fails_closed() -> None:
    expected = {
        **{metric: 1.0 for metric in MATCH_METRICS},
        "task": "pick-place-v3",
        "physical_episode": 1,
        "method": "f0_l0",
        "visible_grounding_mode": "nominal",
        "environment_seed": 9,
        "reset_digest": "a",
        "camera_sha256": "b",
        "schedule_sha256": "c",
        "drop_onset_state_sha256": "d",
        "moving_goal_trajectory_sha256": "e",
        "prefix_action_sha256": "f",
        "prefix_native_token_sha256": "g",
        "prefix_sampled_visual_sha256": "h",
    }
    replay = dict(expected)
    replay["reward_sum"] = 1.01
    drift = _validate_replay_contract(replay, expected)
    assert drift["reward_sum"] == pytest.approx(0.01)
    replay["camera_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="camera_sha256"):
        _validate_replay_contract(replay, expected)


def _trace() -> Mapping[str, object]:
    return {
        "observed_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "q_true": np.asarray([[8, 8], [16, 16], [24, 24]], dtype=np.float32),
        "q_nominal": np.asarray([[9, 8], [17, 16], [25, 24]], dtype=np.float32),
        "selected_points": np.asarray(
            [[8, 9], [16, 17], [24, 25]], dtype=np.float32
        ),
        "prior_active": True,
        "success": 0.0,
        "cumulative_reward": 2.5,
    }


def test_render_frame_produces_nonblank_rgb_canvas() -> None:
    traces = {cell: [_trace()] for cell in CELL_ORDER}
    frame = _render_frame(
        traces,
        frame_index=0,
        episode=3,
        reason="test",
        drop_start=1,
        drop_duration=2,
        model_seed=15801,
    )
    assert frame.ndim == 3 and frame.shape[-1] == 3
    assert frame.dtype == np.uint8
    assert float(frame.std()) > 1.0
