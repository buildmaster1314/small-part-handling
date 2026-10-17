from __future__ import annotations

import pytest

from scripts.evaluate_lewm_expert_action_residual_rollout import (
    _validate_rollout_contract,
    seed_macro_comparison,
)


def _row(episode: int, method: str, success: float, reward: float) -> dict[str, object]:
    return {
        "physical_episode": episode,
        "method": method,
        "success": success,
        "sustained_success": success,
        "final_success": success,
        "success_fraction": 0.5 * success,
        "maximum_success_streak": 3.0 * success,
        "post_drop_success_fraction": 0.5 * success,
        "post_drop_maximum_success_streak": 5.0 * success,
        "reward_sum": reward,
    }


def test_seed_macro_comparison_clusters_head_seeds_within_episode() -> None:
    rows = []
    for episode in range(3):
        rows.append(_row(episode, "base", 0.0, 10.0))
        rows.append(_row(episode, "candidate_s1", 1.0, 12.0))
        rows.append(_row(episode, "candidate_s2", float(episode > 0), 11.0))
    result = seed_macro_comparison(
        rows,
        candidate_methods=["candidate_s1", "candidate_s2"],
        baseline_methods=["base", "base"],
        bootstrap_samples=100,
        seed=7,
    )
    assert result["physical_episode_count"] == 3
    assert result["head_seed_count"] == 2
    assert result["sustained_success_delta"] == pytest.approx(5.0 / 6.0)
    assert result["reward_sum_delta"] == pytest.approx(1.5)


def test_seed_macro_comparison_rejects_incomplete_panel() -> None:
    rows = [_row(0, "candidate", 1.0, 12.0)]
    with pytest.raises(ValueError, match="incomplete rollout panel"):
        seed_macro_comparison(
            rows,
            candidate_methods=["candidate"],
            baseline_methods=["base"],
            bootstrap_samples=10,
            seed=1,
        )


def test_rollout_contract_requires_complete_blocks_and_recovery() -> None:
    config = {
        "action_steps_per_decision": 5,
        "drop_duration_decisions": 6,
        "drop_start_min_decision": 6,
        "drop_start_max_decision": 10,
    }
    rollout = {"episodes": 30, "horizon_raw_steps": 335}
    _validate_rollout_contract(config, rollout)
    rollout["horizon_raw_steps"] = 334
    with pytest.raises(ValueError, match="complete five-action"):
        _validate_rollout_contract(config, rollout)
