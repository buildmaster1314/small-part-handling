from __future__ import annotations

import numpy as np

from scripts.evaluate_frozen_feature_probe import (
    _apply_frame_mask,
    _cosine_gap,
    _normalized_l2,
    _relative_gain,
    _summary,
)


def test_frame_mask_broadcasts_over_multiple_anchors() -> None:
    valid = np.ones((2, 3), dtype=bool)
    selected = _apply_frame_mask(np.asarray([True, False]), valid)
    assert selected.shape == (2, 3)
    assert selected.tolist() == [[True, True, True], [False, False, False]]


def test_descriptor_metrics_are_zero_for_identical_vectors() -> None:
    values = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    assert np.allclose(_normalized_l2(values, values), 0.0)
    assert np.allclose(_cosine_gap(values, values), 0.0, atol=1e-6)


def test_descriptor_summary_filters_invalid_values() -> None:
    summary = _summary(np.asarray([1.0, np.nan, 3.0]), np.asarray([True, True, False]))
    assert summary["count"] == 1
    assert summary["mean"] == 1.0


def test_relative_gain_is_positive_when_candidate_is_better() -> None:
    assert _relative_gain({"mean": 2.0}, {"mean": 1.0}) == 0.5
    assert _relative_gain({"mean": 0.0}, {"mean": 1.0}) is None
