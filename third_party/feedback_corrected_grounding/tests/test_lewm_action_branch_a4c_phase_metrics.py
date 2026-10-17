from __future__ import annotations

import numpy as np

from fcpg.lewm_action_branch_a4c_phase_metrics import (
    branch_centered_prediction_losses,
    summarize_phase_development,
)


def _panel():
    target = np.zeros((4, 2, 8, 1, 1), dtype=np.float32)
    target[:, 0, 5:, 0, 0] = -1.0
    target[:, 1, 5:, 0, 0] = 1.0
    factual = np.broadcast_to(target[None], (2, *target.shape)).copy()
    no_action = np.zeros_like(factual)
    return target, factual, no_action


def test_branch_centered_losses_reward_branch_specific_prediction() -> None:
    target, factual, no_action = _panel()

    assert np.allclose(branch_centered_prediction_losses(target, factual), 0.0)
    assert np.all(branch_centered_prediction_losses(target, no_action) > 0.0)


def test_phase_summary_keeps_tasks_phases_and_episodes_separate() -> None:
    target, factual, no_action = _panel()
    summary = summarize_phase_development(
        target=target,
        factual_prediction=factual,
        no_action_prediction=no_action,
        task_name=["pick", "pick", "towel", "towel"],
        episode_id=["p0", "p1", "t0", "t1"],
        phase_name=["approach", "grasp", "wipe_left", "wipe_right"],
        success_crossing=[False, True, False, True],
        seeds=[1, 2],
    )

    assert summary["tasks"]["pick"]["overall"]["relative_improvement"] == 1.0
    assert summary["tasks"]["towel"]["phases"]["wipe_right"][
        "positive_seed_count"
    ] == 2
    assert summary["tasks"]["pick"]["success_crossing"]["row_count"] == 1
