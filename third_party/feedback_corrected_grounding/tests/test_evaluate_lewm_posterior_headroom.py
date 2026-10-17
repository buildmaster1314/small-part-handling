from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_lewm_posterior_headroom import (
    MASKED_WINDOW,
    _action_context,
    _episode_reduce,
    _fusion_candidates,
    _selective_writeback,
    _summarize_window,
)


def test_action_context_preserves_pre_action_transition_order() -> None:
    action = np.arange(12, dtype=np.float32).reshape(6, 2)
    sequence = np.arange(6, dtype=np.int64)
    chunks = _action_context(action, sequence, [0, 3], frame_skip=3)
    np.testing.assert_array_equal(chunks[0], action[:3].reshape(-1))
    np.testing.assert_array_equal(chunks[1], action[3:6].reshape(-1))


def test_fusion_candidates_keep_endpoints_and_oracle_dominates_them() -> None:
    prior = np.array([[[0.0, 0.0], [2.0, 0.0]]], dtype=np.float32)
    measurement = np.array([[[2.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    target = np.array([[[1.0, 1.0], [3.0, 0.0]]], dtype=np.float32)
    methods, alpha = _fusion_candidates(
        prior=prior,
        measurement=measurement,
        hold_last=prior,
        constant_velocity=prior,
        target=target,
        fixed_alphas=[0.0, 0.5, 1.0],
    )
    np.testing.assert_array_equal(methods["lewm_prior"], prior)
    np.testing.assert_array_equal(methods["measurement"], measurement)
    assert alpha.shape == (1, 2)
    oracle_error = np.mean((methods["oracle_convex"] - target) ** 2, axis=-1)
    prior_error = np.mean((prior - target) ** 2, axis=-1)
    measurement_error = np.mean((measurement - target) ** 2, axis=-1)
    assert bool(np.all(oracle_error <= prior_error + 1e-7))
    assert bool(np.all(oracle_error <= measurement_error + 1e-7))


def _frame_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in ("pick", "push"):
        for episode in ("0", "1"):
            for phase in (0, 1):
                for anchor, direct in ((0, False), (1, True)):
                    for method, mse in (("baseline", 2.0), ("posterior", 1.0)):
                        rows.append(
                            {
                                "task": task,
                                "episode_id": episode,
                                "frame_id": phase,
                                "phase": phase,
                                "anchor": anchor,
                                "directly_occluded": direct,
                                "history_crosses_camera_step": True,
                                "method": method,
                                "window": MASKED_WINDOW,
                                "mse": mse,
                            }
                        )
    return rows


def test_episode_reduction_prevents_phase_pseudoreplication_and_scopes_anchors() -> None:
    reduced = _episode_reduce(_frame_rows())
    assert len(reduced) == 2 * 2 * 2 * 2
    assert {row["phase_count"] for row in reduced} == {2}
    direct = _summarize_window(
        reduced,
        window=MASKED_WINDOW,
        baseline="baseline",
        repetitions=100,
        seed=3,
        anchor_scope="directly_occluded",
    )
    indirect = _summarize_window(
        reduced,
        window=MASKED_WINDOW,
        baseline="baseline",
        repetitions=100,
        seed=3,
        anchor_scope="indirectly_affected",
    )
    assert direct["task_anchor_count"] == 2
    assert indirect["task_anchor_count"] == 2
    assert direct["comparisons"]["posterior"]["improvement"] == pytest.approx(1.0)
    assert direct["comparisons"]["posterior"]["improvement_95_ci"] == [1.0, 1.0]


def test_episode_reduction_rejects_anchor_scope_changes() -> None:
    rows = _frame_rows()
    rows[0]["directly_occluded"] = True
    with pytest.raises(ValueError, match="anchor scope changed"):
        _episode_reduce(rows)


def test_selective_writeback_only_replaces_active_anchor() -> None:
    candidate = np.full((1, 3, 2), 9.0, dtype=np.float32)
    measurement = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    active = np.array([[False, True, False]], dtype=np.bool_)
    result = _selective_writeback(candidate, measurement, active)
    np.testing.assert_array_equal(result[:, 0], measurement[:, 0])
    np.testing.assert_array_equal(result[:, 1], candidate[:, 1])
    np.testing.assert_array_equal(result[:, 2], measurement[:, 2])


def test_selective_writeback_rejects_nonboolean_mask() -> None:
    values = np.zeros((1, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="active_mask"):
        _selective_writeback(values, values, np.ones((1, 2), dtype=np.int64))
