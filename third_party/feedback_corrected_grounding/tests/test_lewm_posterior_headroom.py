from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_posterior_headroom import (
    EpisodeEventRecord,
    METHOD_NAMES,
    build_episode_event_records,
    compute_posterior_headroom,
    event_relative_masks,
    task_anchor_macro,
)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_closed_form_fusion_endpoints_mask_and_hold_last(backend: str) -> None:
    prior_np = np.array([[[[0.0, 0.0]], [[10.0, 10.0]], [[20.0, 20.0]]]])
    measurement_np = np.array([[[[2.0, 0.0]], [[np.nan, np.nan]], [[6.0, 2.0]]]])
    target_np = np.array([[[[1.0, 1.0]], [[3.0, 1.0]], [[8.0, 2.0]]]])
    valid_np = np.array([[[True], [False], [True]]])
    if backend == "torch":
        prior, measurement, target = map(torch.from_numpy, (prior_np, measurement_np, target_np))
        valid = torch.from_numpy(valid_np)
    else:
        prior, measurement, target, valid = prior_np, measurement_np, target_np, valid_np

    result = compute_posterior_headroom(
        prior, measurement, target, fixed_alpha=0.25, measurement_valid=valid
    )
    expected_oracle = np.array([[[0.5], [0.0], [492.0 / 520.0]]])
    expected_hold = np.array([[[[2.0, 0.0]], [[2.0, 0.0]], [[6.0, 2.0]]]])
    expected_fixed = np.array([[[[0.5, 0.0]], [[10.0, 10.0]], [[16.5, 15.5]]]])
    if backend == "torch":
        torch.testing.assert_close(result.oracle_alpha, torch.from_numpy(expected_oracle))
        torch.testing.assert_close(result.hold_last.prediction, torch.from_numpy(expected_hold))
        torch.testing.assert_close(result.fixed_alpha_fusion.prediction, torch.from_numpy(expected_fixed))
        torch.testing.assert_close(result.measurement_only.prediction[:, 1], prior[:, 1])
    else:
        np.testing.assert_allclose(result.oracle_alpha, expected_oracle)
        np.testing.assert_allclose(result.hold_last.prediction, expected_hold)
        np.testing.assert_allclose(result.fixed_alpha_fusion.prediction, expected_fixed)
        np.testing.assert_allclose(result.measurement_only.prediction[:, 1], prior[:, 1])
    assert result.prior_only.sse.shape == (1, 3, 1)
    assert result.prior_only.mse.shape == (1, 3, 1)
    assert result.prior_only.episode_mse.shape == (1,)


def test_oracle_alpha_is_clipped_and_degenerate_direction_uses_zero() -> None:
    prior = torch.zeros(1, 1, 3, 2, dtype=torch.float64)
    measurement = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float64)
    target = torch.tensor([[[[-3.0, 0.0], [4.0, 0.0], [2.0, 2.0]]]], dtype=torch.float64)
    result = compute_posterior_headroom(prior, measurement, target, fixed_alpha=0.5)
    torch.testing.assert_close(result.oracle_alpha, torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64))
    assert bool((result.oracle_convex_alpha.sse <= result.prior_only.sse).all())
    assert bool((result.oracle_convex_alpha.sse <= result.measurement_only.sse).all())


def test_hold_last_falls_back_to_current_prior_before_first_valid_measurement() -> None:
    prior = torch.tensor([[[[3.0]], [[4.0]], [[5.0]]]])
    measurement = torch.tensor([[[[float("nan")]], [[7.0]], [[float("nan")]]]])
    valid = torch.tensor([[[False], [True], [False]]])
    result = compute_posterior_headroom(
        prior, measurement, torch.zeros_like(prior), fixed_alpha=0.5, measurement_valid=valid
    )
    torch.testing.assert_close(result.hold_last.prediction.flatten(), torch.tensor([3.0, 7.0, 7.0]))
    assert result.hold_last_available.flatten().tolist() == [False, True, True]


def test_event_masks_mark_contiguous_offsets() -> None:
    masked = np.array([[[False], [True], [True], [True], [False], [False]]])
    events = event_relative_masks(masked)
    assert events.o0.reshape(-1).tolist() == [False, True, False, False, False, False]
    assert events.o1.reshape(-1).tolist() == [False, False, True, False, False, False]
    assert events.o2.reshape(-1).tolist() == [False, False, False, True, False, False]
    assert events.r0.reshape(-1).tolist() == [False, False, False, False, True, False]
    assert events.r1.reshape(-1).tolist() == [False, False, False, False, False, True]


def test_event_records_are_episode_level_and_r0_is_never_primary() -> None:
    prior = torch.zeros(2, 6, 1, 2)
    target = torch.ones_like(prior)
    measurement = torch.full_like(prior, 2.0)
    masked = torch.tensor(
        [[[False], [True], [True], [True], [False], [False]]] * 2
    )
    valid = ~masked
    result = compute_posterior_headroom(
        prior, measurement, target, fixed_alpha=0.5, measurement_valid=valid
    )
    records = build_episode_event_records(
        result,
        masked,
        task_names=["pick", "push"],
        episode_ids=[10, 20],
    )
    assert len(records) == 2 * 5
    assert {(row.event, row.is_primary) for row in records} == {
        ("o0", False), ("o1", True), ("o2", True), ("r0", False), ("r1", True)
    }
    assert all(row.event_samples == 1 for row in records)
    assert all(set(row.mse) == set(METHOD_NAMES) for row in records)
    assert all(row.same_frame_measurement == (row.event == "r0") for row in records)
    with pytest.raises(ValueError, match="r0 same-frame"):
        build_episode_event_records(
            result,
            masked,
            task_names=["pick", "push"],
            episode_ids=[10, 20],
            primary_events=("r0",),
        )


def _record(task: str, episode: str, anchor: int, mse: float, samples: int) -> EpisodeEventRecord:
    elements = samples
    values = {name: mse for name in METHOD_NAMES}
    return EpisodeEventRecord(
        episode_index=0,
        episode_id=episode,
        task=task,
        anchor_index=anchor,
        event="o1",
        event_samples=samples,
        element_count=elements,
        sse={name: mse * elements for name in METHOD_NAMES},
        mse=values,
        is_primary=True,
        same_frame_measurement=False,
    )


def test_task_anchor_macro_uses_equal_episodes_not_pooled_elements() -> None:
    records = (
        _record("a", "a0", 0, 1.0, 100),
        _record("a", "a1", 0, 9.0, 1),
        _record("a", "a0", 1, 3.0, 1),
        _record("a", "a1", 1, 3.0, 1),
        _record("b", "b0", 0, 20.0, 1),
        _record("b", "b0", 1, 4.0, 1),
    )
    summary = task_anchor_macro(records, event="o1", method="fixed_alpha_fusion")
    assert summary.per_task_anchor_mse == {"a": {0: 5.0, 1: 3.0}, "b": {0: 20.0, 1: 4.0}}
    assert summary.per_task_mse == {"a": 4.0, "b": 12.0}
    assert summary.per_anchor_mse == {0: 12.5, 1: 3.5}
    assert summary.task_anchor_macro_mse == pytest.approx(8.0)
    assert summary.statistical_unit == "episode"


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("shape", "match prior shape"),
        ("dtype", "dtypes must match"),
        ("valid_nan", "valid measurements must be finite"),
        ("alpha", "lie in \\[0, 1\\]"),
    ],
)
def test_validation_fails_closed(change: str, error: str) -> None:
    prior = torch.zeros(1, 2, 1, 2)
    measurement = torch.ones_like(prior)
    target = torch.zeros_like(prior)
    alpha = 0.5
    if change == "shape":
        measurement = measurement[:, :1]
    elif change == "dtype":
        measurement = measurement.double()
    elif change == "valid_nan":
        measurement[0, 0, 0, 0] = float("nan")
    else:
        alpha = 1.5
    with pytest.raises((TypeError, ValueError), match=error):
        compute_posterior_headroom(prior, measurement, target, fixed_alpha=alpha)


def test_masked_windows_must_be_measurement_invalid() -> None:
    values = np.zeros((1, 3, 1, 1), dtype=np.float64)
    result = compute_posterior_headroom(values, values, values, fixed_alpha=0.5)
    masked = np.array([[[False], [True], [False]]])
    with pytest.raises(ValueError, match="measurement-invalid"):
        build_episode_event_records(
            result, masked, task_names=["pick"], episode_ids=[1]
        )
