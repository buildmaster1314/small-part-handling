from __future__ import annotations

import math

import numpy as np
import pytest

from fcpg.visual_trust_metrics import (
    binary_auroc,
    clean_false_reject_rate,
    early_reopen_rate,
    recovery_detection_delay,
    recovery_detection_delays,
    summarize_recovery_detection,
    summarize_visual_trust,
    unsafe_recall,
)


def test_binary_auroc_uses_average_ranks_for_ties() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.0, 0.5, 0.5, 1.0])

    # The tied positive/negative pair contributes one half, so 3.5 of the
    # four positive-negative comparisons are correctly ordered.
    assert binary_auroc(labels, scores) == pytest.approx(0.875)
    assert binary_auroc(labels, np.ones(4)) == pytest.approx(0.5)


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1], []])
def test_binary_auroc_is_nan_when_a_class_is_missing(labels: list[int]) -> None:
    scores = np.arange(len(labels), dtype=np.float64)
    assert math.isnan(binary_auroc(labels, scores))


def test_threshold_rates_follow_reject_and_reopen_boundaries() -> None:
    trust = np.asarray([0.1, 0.5, 0.9])
    unsafe = np.asarray([True, True, False])
    clean = np.asarray([False, False, True])
    opportunities = np.asarray([True, True, False])

    assert unsafe_recall(trust, unsafe, threshold=0.5) == pytest.approx(0.5)
    assert clean_false_reject_rate(trust, clean, threshold=0.5) == 0.0
    assert early_reopen_rate(trust, opportunities, threshold=0.5) == pytest.approx(
        0.5
    )

    no_samples = np.zeros(3, dtype=np.bool_)
    assert math.isnan(unsafe_recall(trust, no_samples))
    assert math.isnan(clean_false_reject_rate(trust, no_samples))
    assert math.isnan(early_reopen_rate(trust, no_samples))


def test_recovery_delay_requires_a_full_consecutive_run() -> None:
    trust = np.asarray([0.9, 0.9, 0.2, 0.8, 0.8, 0.3, 0.8, 0.8, 0.8])

    # The run before recovery_start is ignored and the first post-start pair
    # is too short. The length-three run begins four frames after onset and is
    # causally confirmed two frames later.
    assert recovery_detection_delay(
        trust,
        threshold=0.8,
        min_consecutive=3,
        recovery_start=2,
    ) == pytest.approx(6.0)


def test_never_recovered_episode_has_nan_delay() -> None:
    trust = np.asarray(
        [
            [0.1, 0.8, 0.8, 0.8],
            [0.1, 0.8, 0.7, 0.8],
        ]
    )
    delays = recovery_detection_delays(
        trust, threshold=0.8, min_consecutive=3
    )

    np.testing.assert_allclose(delays[0], 3.0)
    assert math.isnan(delays[1])


def test_recovery_summary_uses_equal_family_and_task_macros() -> None:
    trust = np.asarray(
        [
            [0.8, 0.1, 0.1, 0.1, 0.1, 0.1],  # delay 0
            [0.1, 0.1, 0.8, 0.1, 0.1, 0.1],  # delay 2
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.8],  # delay 5
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # never detected
        ]
    )
    summary = summarize_recovery_detection(
        trust,
        episode_ids=["e0", "e1", "e2", "e3"],
        family_ids=["f1", "f1", "f2", "f2"],
        task_ids=["t1", "t2", "t2", "t2"],
        threshold=0.8,
    )

    assert summary["per_episode"]["e3"]["detected"] is False
    assert math.isnan(summary["per_episode"]["e3"]["delay"])
    assert summary["per_family"]["f1"]["mean_delay"] == pytest.approx(1.0)
    assert summary["per_family"]["f2"]["mean_delay"] == pytest.approx(5.0)
    assert summary["family_macro"]["mean_delay"] == pytest.approx(3.0)
    assert summary["family_macro"]["detection_rate"] == pytest.approx(0.75)

    assert summary["per_task"]["t2"]["mean_delay"] == pytest.approx(3.5)
    assert summary["task_macro"]["mean_delay"] == pytest.approx(1.75)
    assert summary["task_macro"]["detection_rate"] == pytest.approx(5.0 / 6.0)


def test_visual_trust_summary_uses_equal_group_macros() -> None:
    trust = np.asarray([0.1, 0.2, 0.3, 0.9, 0.9, 0.1])
    unsafe = np.asarray([1, 1, 1, 0, 1, 0], dtype=np.bool_)
    clean = ~unsafe
    summary = summarize_visual_trust(
        trust,
        unsafe_mask=unsafe,
        clean_mask=clean,
        early_reopen_mask=unsafe,
        task_ids=["large", "large", "large", "large", "small", "small"],
        family_ids=["a", "a", "a", "a", "b", "b"],
        threshold=0.5,
    )

    assert summary["overall"]["unsafe_recall"] == pytest.approx(0.75)
    assert summary["task_macro"]["unsafe_recall"] == pytest.approx(0.5)
    assert summary["family_macro"]["unsafe_recall"] == pytest.approx(0.5)
    assert summary["task_macro"]["auroc"] == pytest.approx(0.5)
    assert summary["task_macro"]["defined_group_count"]["auroc"] == 2


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: binary_auroc([0, 1], [0.0, np.nan]),
            ValueError,
            "finite",
        ),
        (
            lambda: binary_auroc([[0, 1]], [0.0, 1.0]),
            ValueError,
            "1-dimensional",
        ),
        (
            lambda: unsafe_recall([0.1, 0.2], [1], 0.5),
            ValueError,
            "identical shapes",
        ),
        (
            lambda: clean_false_reject_rate([0.1], [2], 0.5),
            ValueError,
            "only 0, 1",
        ),
        (
            lambda: recovery_detection_delays([0.1, 0.2]),
            ValueError,
            "2-dimensional",
        ),
        (
            lambda: recovery_detection_delay([0.1, np.inf]),
            ValueError,
            "finite",
        ),
        (
            lambda: recovery_detection_delay([0.1], min_consecutive=0),
            ValueError,
            "positive",
        ),
    ],
)
def test_input_validation_is_strict(call, error, message) -> None:
    with pytest.raises(error, match=message):
        call()


def test_summary_rejects_non_finite_ids_and_overlapping_classes() -> None:
    with pytest.raises(ValueError, match="finite identifiers"):
        summarize_recovery_detection(
            np.ones((2, 2)),
            episode_ids=["e0", "e1"],
            family_ids=[0.0, np.nan],
            task_ids=["t", "t"],
        )

    with pytest.raises(ValueError, match="must not overlap"):
        summarize_visual_trust(
            [0.2],
            unsafe_mask=[True],
            clean_mask=[True],
            early_reopen_mask=[False],
            task_ids=["t"],
            family_ids=["f"],
        )
