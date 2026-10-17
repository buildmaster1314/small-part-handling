from __future__ import annotations

import numpy as np
import pytest

from fcpg.lewm_counterfactual import (
    build_action_candidates,
    counterfactual_ranking_metrics,
    spearman_rank_correlation,
)


def test_action_candidates_are_deterministic_and_preserve_expert_gripper() -> None:
    expert = np.array(
        [[0.1, 0.2, 0.3, 1.0], [0.2, 0.1, 0.0, -1.0]], dtype=np.float32
    )
    policy = expert * 0.5
    shuffled = expert[::-1].copy()
    left = build_action_candidates(
        expert,
        policy_action=policy,
        shuffled_action=shuffled,
        rng=np.random.default_rng(7),
        noise_scales=(0.1, 0.2),
        random_count=1,
    )
    right = build_action_candidates(
        expert,
        policy_action=policy,
        shuffled_action=shuffled,
        rng=np.random.default_rng(7),
        noise_scales=(0.1, 0.2),
        random_count=1,
    )

    assert left.names == (
        "expert",
        "policy",
        "expert_noise_0.1",
        "expert_noise_0.2",
        "same_task_shuffled",
        "random_0",
    )
    np.testing.assert_array_equal(left.actions, right.actions)
    np.testing.assert_array_equal(left.actions[0], expert)
    np.testing.assert_array_equal(left.actions[1], policy)
    np.testing.assert_array_equal(left.actions[4], shuffled)
    np.testing.assert_array_equal(
        left.actions[2:4, :, -1],
        np.broadcast_to(expert[None, :, -1], (2, *expert[:, -1].shape)),
    )
    assert float(np.abs(left.actions).max()) <= 1.0


def test_action_candidates_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        build_action_candidates(
            np.zeros((5, 4), dtype=np.float32),
            policy_action=np.zeros((4, 4), dtype=np.float32),
            shuffled_action=np.zeros((5, 4), dtype=np.float32),
            rng=np.random.default_rng(0),
        )


def test_spearman_handles_order_and_ties() -> None:
    assert spearman_rank_correlation(
        np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])
    ) == pytest.approx(-1.0)
    assert spearman_rank_correlation(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 2.0])
    ) == pytest.approx(1.0)


def test_counterfactual_metrics_report_hits_and_regret() -> None:
    metrics = counterfactual_ranking_metrics(
        np.array([0.2, 0.1, 0.4, 0.8]),
        np.array([0.0, 0.3, 0.5, 1.0]),
        expert_index=0,
        top_k=2,
    )

    assert metrics["predicted_best_index"] == 1
    assert metrics["actual_best_index"] == 0
    assert metrics["top1_hit"] == 0
    assert metrics["topk_hit"] == 1
    assert metrics["expert_top1"] == 0
    assert metrics["expert_topk"] == 1
    assert metrics["expert_pairwise_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["regret"] == pytest.approx(0.3)
    assert metrics["normalized_regret"] == pytest.approx(0.3)
