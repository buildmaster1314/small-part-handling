from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from fcpg.lewm_counterfactual_v3 import CANDIDATE_COUNT, HORIZON, V3Axes
from fcpg.lewm_counterfactual_v3_exploratory import (
    FutureStateResidualRanker,
    actual_return_ranking_loss,
    build_episode_crossfit_assignment,
    evaluate_actual_return_scores,
    train_actual_return_exploratory,
    train_actual_return_crossfit_exploratory,
)
from fcpg.lewm_counterfactual_v3_gate import V3GateBundle
from fcpg.lewm_counterfactual_v3_state import STATE_DIM


def _synthetic_bundle() -> V3GateBundle:
    rng = np.random.default_rng(73)
    samples = 8
    task = np.repeat(np.arange(4, dtype=np.int16), 2)
    split = np.tile(np.asarray([0, 1], dtype=np.int8), 4)
    action = rng.normal(size=(samples, CANDIDATE_COUNT, 25, 4)).astype(np.float32)
    current = rng.normal(size=(samples, STATE_DIM)).astype(np.float32)
    future = rng.normal(
        size=(samples, CANDIDATE_COUNT, HORIZON, STATE_DIM)
    ).astype(np.float32)
    quality = (
        np.linspace(-0.8, 0.8, CANDIDATE_COUNT, dtype=np.float32)[None]
        + 0.05 * task[:, None]
    )
    future[..., 0] = quality[:, :, None]
    actual_return = np.stack(
        [(step + 1) * quality for step in range(HORIZON)], axis=-1
    ).astype(np.float32)
    return V3GateBundle(
        deployment_inputs={
            "center_action_raw": np.zeros((samples, 25, 4), dtype=np.float32),
            "candidate_action_raw": action,
        },
        offline_labels={
            "current_state": current,
            "future_state": future,
            "goal_world": rng.normal(size=(samples, 3)).astype(np.float32),
            # The exploratory model must never read this formal-v3 target.
            "physical_task_score": np.full(
                (samples, CANDIDATE_COUNT), np.nan, dtype=np.float32
            ),
            "actual_return": actual_return,
            "progress": np.zeros_like(actual_return),
            "success": np.zeros_like(actual_return),
        },
        audit_arrays={"task_index": task, "split_index": split},
        index_rows=tuple({} for _ in range(samples)),
        axes=V3Axes(samples, CANDIDATE_COUNT, HORIZON, STATE_DIM),
    )


def _crossfit_bundle(*, phases_per_episode: int = 1) -> V3GateBundle:
    rng = np.random.default_rng(91)
    records: list[tuple[int, int, int]] = []
    for task in range(4):
        for episode in range(8):
            for phase in range(phases_per_episode):
                records.append((task, episode, phase))
        records.append((task, 100, 0))
    samples = len(records)
    task = np.asarray([value[0] for value in records], dtype=np.int16)
    split = np.asarray(
        [0 if value[1] < 8 else 1 for value in records], dtype=np.int8
    )
    action = rng.normal(size=(samples, CANDIDATE_COUNT, 25, 4)).astype(np.float32)
    current = rng.normal(size=(samples, STATE_DIM)).astype(np.float32)
    future = rng.normal(
        size=(samples, CANDIDATE_COUNT, HORIZON, STATE_DIM)
    ).astype(np.float32)
    quality = (
        np.linspace(-0.7, 0.9, CANDIDATE_COUNT, dtype=np.float32)[None]
        + 0.03 * task[:, None]
    )
    future[..., 0] = quality[:, :, None]
    actual_return = np.stack(
        [(step + 1) * quality for step in range(HORIZON)], axis=-1
    ).astype(np.float32)
    index_rows = tuple(
        {
            "task": (
                "pick-place-v3",
                "button-press-v3",
                "reach-v3",
                "push-v3",
            )[task_value],
            "physical_episode_sha256": f"task{task_value}-episode{episode:03d}",
            "physical_snapshot_sha256": (
                f"task{task_value}-episode{episode:03d}-phase{phase}"
            ),
        }
        for task_value, episode, phase in records
    )
    return V3GateBundle(
        deployment_inputs={
            "center_action_raw": np.zeros((samples, 25, 4), dtype=np.float32),
            "candidate_action_raw": action,
        },
        offline_labels={
            "current_state": current,
            "future_state": future,
            "goal_world": rng.normal(size=(samples, 3)).astype(np.float32),
            "physical_task_score": np.full(
                (samples, CANDIDATE_COUNT), np.nan, dtype=np.float32
            ),
            "actual_return": actual_return,
            "progress": np.zeros_like(actual_return),
            "success": np.zeros_like(actual_return),
        },
        audit_arrays={"task_index": task, "split_index": split},
        index_rows=index_rows,
        axes=V3Axes(samples, CANDIDATE_COUNT, HORIZON, STATE_DIM),
    )


def test_raw_return_mask_is_computed_before_label_normalization() -> None:
    scores = torch.zeros((1, 3))
    normalized = torch.tensor([[0.0, 10.0, 20.0]])
    raw = torch.tensor([[0.0, 0.005, 0.010]])
    loss = actual_return_ranking_loss(
        scores,
        normalized,
        raw,
        raw_tie_tolerance=0.01,
    )
    assert loss.informative_pair_count == 1
    assert loss.tie_pair_count == 2


def test_future_residual_is_exactly_nested_and_candidate_equivariant() -> None:
    torch.manual_seed(8)
    model = FutureStateResidualRanker(
        hidden_dim=16, trajectory_hidden_dim=8
    ).eval()
    base = torch.randn(2, CANDIDATE_COUNT, requires_grad=True)
    future = torch.randn(2, CANDIDATE_COUNT, HORIZON, STATE_DIM)
    goal = torch.randn(2, 3)
    task = torch.eye(4)[:2]
    permutation = torch.tensor([5, 1, 7, 0, 3, 6, 4, 2])
    score = model(base, future, goal, task)
    assert torch.equal(score, base)
    permuted = model(base[:, permutation], future[:, permutation], goal, task)
    assert torch.equal(permuted, score[:, permutation])
    score.sum().backward()
    assert base.grad is None


def test_cpu_synthetic_training_returns_seed_scores_and_parameter_counts() -> None:
    bundle = _synthetic_bundle()
    result = train_actual_return_exploratory(
        bundle,
        direct_seeds=(101,),
        residual_seeds=(201,),
        direct_steps=2,
        residual_steps=2,
        direct_hidden_dim=12,
        direct_action_hidden_dim=6,
        residual_hidden_dim=12,
        residual_trajectory_hidden_dim=6,
        device="cpu",
    )
    assert result.target_name == "actual_return[:, :, -1]"
    assert result.raw_tie_tolerance == pytest.approx(0.01)
    assert result.direct_training_scores.shape == (4, CANDIDATE_COUNT)
    assert result.direct_validation_scores.shape == (4, CANDIDATE_COUNT)
    assert result.nested_validation_scores.shape == (4, CANDIDATE_COUNT)
    assert result.direct_parameter_count > 0
    assert result.residual_parameter_count > 0
    assert result.nested_total_parameter_count == (
        result.direct_parameter_count + result.residual_parameter_count
    )
    assert len(result.direct_seed_results) == 1
    assert len(result.residual_seed_results) == 1
    assert result.residual_seed_results[0].initial_delta_max_abs == 0.0
    summary = evaluate_actual_return_scores(
        bundle, result.nested_validation_scores, split="validation"
    )
    assert summary["target"] == "actual_return[:, :, -1]"
    assert summary["snapshot_count"] == 4
    assert set(summary["per_task"]) == {
        "pick-place-v3",
        "button-press-v3",
        "reach-v3",
        "push-v3",
    }


def test_episode_crossfit_assignment_keeps_phases_together() -> None:
    bundle = _crossfit_bundle(phases_per_episode=2)
    assignment = build_episode_crossfit_assignment(bundle)
    assert np.all(assignment.fold_by_sample[bundle.train_mask] >= 0)
    assert np.all(assignment.fold_by_sample[bundle.validation_mask] == -1)
    for task_name, episode_mapping in assignment.episode_fold_by_task.items():
        assert task_name
        assert len(episode_mapping) == 8
        assert sorted(episode_mapping.values()) == [0, 0, 1, 1, 2, 2, 3, 3]
    observed: dict[str, set[int]] = {}
    for sample in np.flatnonzero(bundle.train_mask):
        episode = str(bundle.index_rows[int(sample)]["physical_episode_sha256"])
        observed.setdefault(episode, set()).add(
            int(assignment.fold_by_sample[sample])
        )
    assert all(len(folds) == 1 for folds in observed.values())


def test_crossfit_training_scores_every_training_sample_exactly_once() -> None:
    bundle = _crossfit_bundle()
    result = train_actual_return_crossfit_exploratory(
        bundle,
        direct_seeds=(301,),
        residual_seeds=(401,),
        direct_steps=1,
        residual_steps=1,
        direct_hidden_dim=10,
        direct_action_hidden_dim=5,
        residual_hidden_dim=10,
        residual_trajectory_hidden_dim=5,
        device="cpu",
    )
    assert np.array_equal(result.oof_coverage, np.ones(32, dtype=np.int8))
    assert result.oof_direct_training_scores.shape == (32, CANDIDATE_COUNT)
    assert result.full_direct_training_scores.shape == (32, CANDIDATE_COUNT)
    assert result.full_direct_validation_scores.shape == (4, CANDIDATE_COUNT)
    assert result.nested_training_scores.shape == (32, CANDIDATE_COUNT)
    assert result.nested_validation_scores.shape == (4, CANDIDATE_COUNT)
    assert result.oof_direct_training_scores.dtype == np.float32
    assert result.full_direct_training_scores.dtype == np.float32
    assert result.full_direct_validation_scores.dtype == np.float32
    assert result.nested_training_scores.dtype == np.float32
    assert result.nested_validation_scores.dtype == np.float32
    assert len(result.fold_results) == 4
    assert result.residual_seed_results[0].initial_delta_max_abs == 0.0
    for fold in result.fold_results:
        assert fold.training_sample_indices.size == 24
        assert fold.heldout_sample_indices.size == 8
        for task_name in (
            "pick-place-v3",
            "button-press-v3",
            "reach-v3",
            "push-v3",
        ):
            assert len(fold.training_episode_sha256_by_task[task_name]) == 6
            assert len(fold.heldout_episode_sha256_by_task[task_name]) == 2


@pytest.mark.parametrize("crossfit", [False, True])
def test_training_restores_python_numpy_and_torch_rng_state(
    crossfit: bool,
) -> None:
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_torch = torch.get_rng_state().clone()
    try:
        random.seed(811)
        np.random.seed(812)
        torch.manual_seed(813)
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_torch = torch.get_rng_state().clone()

        common = {
            "direct_seeds": (501,),
            "residual_seeds": (601,),
            "direct_steps": 1,
            "residual_steps": 1,
            "direct_hidden_dim": 10,
            "direct_action_hidden_dim": 5,
            "residual_hidden_dim": 10,
            "residual_trajectory_hidden_dim": 5,
            "device": "cpu",
        }
        if crossfit:
            train_actual_return_crossfit_exploratory(
                _crossfit_bundle(), **common
            )
        else:
            train_actual_return_exploratory(_synthetic_bundle(), **common)

        after_python = random.getstate()
        after_numpy = np.random.get_state()
        after_torch = torch.get_rng_state()
        assert after_python == before_python
        assert after_numpy[0] == before_numpy[0]
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        assert after_numpy[2:] == before_numpy[2:]
        assert torch.equal(after_torch, before_torch)
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.set_rng_state(original_torch)
