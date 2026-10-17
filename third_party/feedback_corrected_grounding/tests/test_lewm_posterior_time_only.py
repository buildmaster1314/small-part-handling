from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect

import numpy as np
import pytest

from fcpg.lewm_posterior_time_only import (
    DEFAULT_N_TIME_BINS,
    TimeOnlyPolicy,
    fit_time_only_policy,
    predict_beta,
)


def _calibration() -> dict[str, np.ndarray]:
    return {
        "normalized_time": np.array([0.01, 0.02, 0.03, 0.21, 0.22, 1.0]),
        "frame_phase": np.array([0, 0, 0, 1, 1, 4]),
        "anchor_index": np.array([0, 0, 0, 1, 1, 1]),
        "measurement_error": np.array([3.0, 6.0, 9.0, 1.0, 1.0, 5.0]),
        "prior_error": np.array([1.0, 2.0, 3.0, 2.0, 2.0, 1.0]),
    }


def _fit(**overrides: object) -> TimeOnlyPolicy:
    values: dict[str, object] = _calibration()
    values.update({"n_time_bins": 5, "min_count": 3, "dead_zone": 0.1})
    values.update(overrides)
    return fit_time_only_policy(**values)


def test_fit_records_exact_cell_statistics_and_binary_switches() -> None:
    policy = _fit()

    assert policy.n_time_bins == 5
    assert policy.n_anchors == 2
    assert policy.count[0, 0, 0] == 3
    expected = np.mean(
        np.array(
            [
                (3.0 - 1.0) / (3.0 + 1.0 + policy.eps),
                (6.0 - 2.0) / (6.0 + 2.0 + policy.eps),
                (9.0 - 3.0) / (9.0 + 3.0 + policy.eps),
            ]
        )
    )
    assert policy.mean_advantage[0, 0, 0] == expected
    assert bool(policy.choose_prior[0, 0, 0])

    beta = predict_beta(
        policy,
        np.array([0.0, 0.199, 0.2, 1.0]),
        np.array([0, 0, 1, 4]),
        np.array([0, 0, 1, 1]),
    )
    np.testing.assert_array_equal(beta, np.array([1, 1, 0, 0], np.float32))
    assert beta.dtype == np.float32


def test_default_uses_ten_equal_width_time_bins() -> None:
    values = _calibration()
    policy = fit_time_only_policy(**values)
    assert policy.n_time_bins == DEFAULT_N_TIME_BINS == 10
    assert policy.count[0, 0, 0] == 3
    assert policy.count[2, 1, 1] == 2
    assert policy.count[9, 4, 1] == 1


def test_sparse_cells_fail_closed_to_measurement() -> None:
    policy = _fit(min_count=3)

    assert policy.count[1, 1, 1] == 2
    assert policy.mean_advantage[1, 1, 1] < 0.0
    assert not bool(policy.choose_prior[1, 1, 1])
    assert policy.count[4, 4, 1] == 1
    assert policy.mean_advantage[4, 4, 1] > policy.dead_zone
    assert not bool(policy.choose_prior[4, 4, 1])
    np.testing.assert_array_equal(
        predict_beta(policy, np.array([0.21, 1.0]), np.array([1, 4]), np.array([1, 1])),
        np.zeros(2, dtype=np.float32),
    )


def test_inverse_episode_row_weights_prevent_long_episode_domination() -> None:
    episode_index = np.array([0, 0, 0, 0, 1])
    episode_row_count = np.bincount(episode_index)
    sample_weight = 1.0 / episode_row_count[episode_index]
    values = {
        "normalized_time": np.full(5, 0.5),
        "frame_phase": np.zeros(5, dtype=np.int64),
        "anchor_index": np.zeros(5, dtype=np.int64),
        "measurement_error": np.array([3.0, 3.0, 3.0, 3.0, 1.0]),
        "prior_error": np.array([1.0, 1.0, 1.0, 1.0, 9.0]),
    }

    row_weighted = fit_time_only_policy(
        **values, n_time_bins=1, min_count=5
    )
    episode_weighted = fit_time_only_policy(
        **values,
        sample_weight=sample_weight,
        n_time_bins=1,
        min_count=5,
    )

    assert row_weighted.count[0, 0, 0] == 5
    assert episode_weighted.count[0, 0, 0] == 5
    assert row_weighted.mean_advantage[0, 0, 0] > 0.0
    assert episode_weighted.mean_advantage[0, 0, 0] < 0.0
    assert bool(row_weighted.choose_prior[0, 0, 0])
    assert not bool(episode_weighted.choose_prior[0, 0, 0])


def test_omitted_sample_weight_preserves_unweighted_fit() -> None:
    omitted = _fit()
    unit_weighted = _fit(sample_weight=np.ones(6, dtype=np.float64))

    assert omitted == unit_weighted


def test_sample_groups_count_independent_parents_not_paired_views() -> None:
    values = {
        "normalized_time": np.full(4, 0.5),
        "frame_phase": np.zeros(4, dtype=np.int64),
        "anchor_index": np.zeros(4, dtype=np.int64),
        "measurement_error": np.array([3.0, 5.0, 1.0, 1.0]),
        "prior_error": np.array([1.0, 1.0, 3.0, 3.0]),
        "sample_group": np.array(["parent-a", "parent-a", "parent-b", "parent-b"]),
    }

    policy = fit_time_only_policy(
        **values, n_time_bins=1, min_count=2
    )

    assert policy.count[0, 0, 0] == 2
    first = np.mean([(3.0 - 1.0) / (3.0 + 1.0 + policy.eps), (5.0 - 1.0) / (5.0 + 1.0 + policy.eps)])
    second = (1.0 - 3.0) / (1.0 + 3.0 + policy.eps)
    assert policy.mean_advantage[0, 0, 0] == pytest.approx((first + second) / 2.0)


def test_sample_subgroups_equalize_views_before_parent_aggregation() -> None:
    policy = fit_time_only_policy(
        normalized_time=np.full(6, 0.5),
        frame_phase=np.zeros(6, dtype=np.int64),
        anchor_index=np.zeros(6, dtype=np.int64),
        measurement_error=np.array([3.0, 3.0, 3.0, 1.0, 3.0, 3.0]),
        prior_error=np.array([1.0, 1.0, 1.0, 3.0, 5.0, 5.0]),
        sample_group=np.array(["parent-a"] * 4 + ["parent-b"] * 2),
        sample_subgroup=np.array(
            ["view-a-step"] * 3
            + ["view-a-linear", "view-b-step", "view-b-linear"]
        ),
        n_time_bins=1,
        min_count=2,
    )

    positive = (3.0 - 1.0) / (3.0 + 1.0 + policy.eps)
    negative = (1.0 - 3.0) / (1.0 + 3.0 + policy.eps)
    weak_negative = (3.0 - 5.0) / (3.0 + 5.0 + policy.eps)
    parent_a = (positive + negative) / 2.0
    expected = (parent_a + weak_negative) / 2.0
    assert policy.count[0, 0, 0] == 2
    assert policy.mean_advantage[0, 0, 0] == pytest.approx(expected)


def test_sample_subgroup_contract_fails_closed() -> None:
    values = _calibration()
    with pytest.raises(ValueError, match="requires sample_group"):
        fit_time_only_policy(
            **values,
            sample_subgroup=np.asarray(["view"] * 6),
        )
    with pytest.raises(ValueError, match="exactly one sample_group"):
        fit_time_only_policy(
            **values,
            sample_group=np.asarray(["parent-a"] * 3 + ["parent-b"] * 3),
            sample_subgroup=np.asarray(["shared-view"] * 6),
        )


def test_sample_groups_do_not_inflate_minimum_count() -> None:
    values = _calibration()
    policy = fit_time_only_policy(
        **values,
        sample_group=np.array(["same"] * 6),
        n_time_bins=5,
        min_count=2,
    )

    assert policy.count[0, 0, 0] == 1
    assert not bool(policy.choose_prior[0, 0, 0])


def test_sample_weight_and_group_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_time_only_policy(
            **_calibration(),
            sample_weight=np.ones(6),
            sample_group=np.arange(6),
        )


def test_explicit_anchor_capacity_keeps_unseen_anchors_fail_closed() -> None:
    policy = fit_time_only_policy(
        normalized_time=np.array([0.2, 0.2]),
        frame_phase=np.array([0, 0]),
        anchor_index=np.array([0, 0]),
        measurement_error=np.array([2.0, 2.0]),
        prior_error=np.array([0.0, 0.0]),
        n_anchors=3,
        min_count=2,
    )
    assert policy.n_anchors == 3
    np.testing.assert_array_equal(
        predict_beta(
            policy,
            np.array([0.2, 0.2]),
            np.array([0, 0]),
            np.array([0, 2]),
        ),
        np.array([1.0, 0.0], dtype=np.float32),
    )


def test_dead_zone_comparison_is_strict() -> None:
    values = {
        "normalized_time": np.array([0.5]),
        "frame_phase": np.array([2]),
        "anchor_index": np.array([0]),
        "measurement_error": np.array([3.0]),
        "prior_error": np.array([1.0]),
    }
    advantage = (3.0 - 1.0) / (3.0 + 1.0 + 1e-12)
    policy = fit_time_only_policy(
        **values, n_time_bins=1, min_count=1, dead_zone=advantage
    )
    assert not bool(policy.choose_prior[0, 2, 0])


def test_prediction_api_cannot_receive_task_or_event_labels() -> None:
    assert tuple(inspect.signature(predict_beta).parameters) == (
        "policy",
        "normalized_time",
        "frame_phase",
        "anchor_index",
    )
    policy = _fit()
    deployment = {
        "normalized_time": np.array([0.01]),
        "frame_phase": np.array([0]),
        "anchor_index": np.array([0]),
    }

    with pytest.raises(TypeError, match="unexpected keyword argument 'task'"):
        predict_beta(policy, **deployment, task=np.array(["task-a"]))
    with pytest.raises(TypeError, match="unexpected keyword argument 'event'"):
        predict_beta(policy, **deployment, event=np.array(["o1"]))


@pytest.mark.parametrize(
    "forbidden",
    (
        "onset",
        "duration",
        "canonical",
        "measurement_valid",
        "images",
        "q_true",
        "measurement_error",
        "prior_error",
    ),
)
def test_other_privileged_inputs_cannot_enter_prediction(forbidden: str) -> None:
    policy = _fit()
    deployment: dict[str, object] = {
        "normalized_time": np.array([0.01]),
        "frame_phase": np.array([0]),
        "anchor_index": np.array([0]),
        forbidden: np.array([0]),
    }
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        predict_beta(policy, **deployment)


def test_shuffled_calibration_rows_fit_identical_policy() -> None:
    rng = np.random.default_rng(20260716)
    count = 400
    values = {
        "normalized_time": rng.random(count),
        "frame_phase": rng.integers(0, 5, size=count),
        "anchor_index": rng.integers(0, 4, size=count),
        "measurement_error": rng.lognormal(size=count),
        "prior_error": rng.lognormal(size=count),
    }
    order = rng.permutation(count)
    first = fit_time_only_policy(
        **values, n_time_bins=7, min_count=4, dead_zone=0.05
    )
    shuffled = fit_time_only_policy(
        **{name: value[order] for name, value in values.items()},
        n_time_bins=7,
        min_count=4,
        dead_zone=0.05,
    )

    assert first == shuffled
    np.testing.assert_array_equal(first.count, shuffled.count)
    np.testing.assert_array_equal(first.mean_advantage, shuffled.mean_advantage)
    np.testing.assert_array_equal(first.choose_prior, shuffled.choose_prior)


def test_shuffled_weighted_rows_fit_identical_policy() -> None:
    rng = np.random.default_rng(20260716)
    count = 400
    values = {
        "normalized_time": rng.random(count),
        "frame_phase": rng.integers(0, 5, size=count),
        "anchor_index": rng.integers(0, 4, size=count),
        "measurement_error": rng.lognormal(size=count),
        "prior_error": rng.lognormal(size=count),
        "sample_weight": rng.lognormal(size=count),
    }
    order = rng.permutation(count)
    first = fit_time_only_policy(
        **values, n_time_bins=7, min_count=4, dead_zone=0.05
    )
    shuffled = fit_time_only_policy(
        **{name: value[order] for name, value in values.items()},
        n_time_bins=7,
        min_count=4,
        dead_zone=0.05,
    )

    assert first == shuffled
    np.testing.assert_array_equal(first.count, shuffled.count)
    np.testing.assert_array_equal(first.mean_advantage, shuffled.mean_advantage)
    np.testing.assert_array_equal(first.choose_prior, shuffled.choose_prior)


def test_policy_is_frozen_and_owns_read_only_tables() -> None:
    policy = _fit()
    assert {field.name for field in fields(policy)} == {
        "n_time_bins",
        "n_anchors",
        "min_count",
        "dead_zone",
        "eps",
        "mean_advantage",
        "count",
        "choose_prior",
    }
    with pytest.raises(FrozenInstanceError):
        policy.min_count = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        policy.count[0, 0, 0] = 0


@pytest.mark.parametrize(
    ("field", "bad_value", "error"),
    [
        ("normalized_time", np.array([np.nan] * 6), "must be finite"),
        ("normalized_time", np.array([1.1] * 6), r"must lie in \[0, 1\]"),
        ("frame_phase", np.array([5] * 6), r"must lie in \[0, 4\]"),
        ("anchor_index", np.array([-1] * 6), "must be non-negative"),
        ("measurement_error", np.array([-1.0] * 6), "must be non-negative"),
        ("prior_error", np.array([np.inf] * 6), "must be finite"),
        ("frame_phase", np.array([True] * 6), "must be an integer array"),
        ("anchor_index", np.arange(6, dtype=float), "must be an integer array"),
    ],
)
def test_fit_validates_values_and_strict_integer_arrays(
    field: str, bad_value: np.ndarray, error: str
) -> None:
    values: dict[str, object] = _calibration()
    values[field] = bad_value
    with pytest.raises((TypeError, ValueError), match=error):
        fit_time_only_policy(**values)


def test_fit_validates_shapes_and_bool_configuration() -> None:
    values = _calibration()
    values["prior_error"] = values["prior_error"][:-1]
    with pytest.raises(ValueError, match="identical shapes"):
        fit_time_only_policy(**values)
    with pytest.raises(ValueError, match="one-dimensional"):
        fit_time_only_policy(**{**_calibration(), "normalized_time": np.zeros((2, 3))})
    with pytest.raises(TypeError, match="n_time_bins must be an integer"):
        fit_time_only_policy(**_calibration(), n_time_bins=True)
    with pytest.raises(TypeError, match="dead_zone must be a real number"):
        fit_time_only_policy(**_calibration(), dead_zone=False)


@pytest.mark.parametrize(
    ("sample_weight", "error"),
    [
        (np.ones((2, 3)), "one-dimensional"),
        (np.ones(5), "identical shape"),
        (np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]), "positive"),
        (np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0]), "positive"),
        (np.array([1.0, 1.0, 1.0, 1.0, 1.0, np.nan]), "finite"),
        (np.array([1.0, 1.0, 1.0, 1.0, 1.0, np.inf]), "finite"),
    ],
)
def test_fit_validates_sample_weight(
    sample_weight: np.ndarray, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        fit_time_only_policy(
            **_calibration(), sample_weight=sample_weight
        )


def test_prediction_repeats_strict_validation_without_calibration_fields() -> None:
    policy = _fit()
    with pytest.raises(ValueError, match="identical shapes"):
        predict_beta(policy, np.array([0.0]), np.array([0, 1]), np.array([0]))
    with pytest.raises(TypeError, match="integer array"):
        predict_beta(policy, np.array([0.0]), np.array([False]), np.array([0]))
    with pytest.raises(ValueError, match="anchor_index must lie"):
        predict_beta(policy, np.array([0.0]), np.array([0]), np.array([2]))
