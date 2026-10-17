import argparse

import numpy as np
import pytest

from scripts.evaluate_kineparity_oracle import (
    aggregate_rows,
    fixed_camera_offset,
    parse_severity,
)


def test_parse_severity_sorts_deduplicates_and_rejects_negative() -> None:
    assert parse_severity("10,0,5,5") == [0.0, 5.0, 10.0]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_severity("0,-1")


def test_fixed_camera_offset_uses_exact_requested_magnitude() -> None:
    translation, _, rotation = fixed_camera_offset("rotation", 10.0, seed=7)
    assert np.linalg.norm(translation) == pytest.approx(0.0)
    assert abs(rotation) == pytest.approx(10.0)

    translation, _, rotation = fixed_camera_offset("translation", 4.0, seed=7)
    assert np.linalg.norm(translation) == pytest.approx(0.04)
    assert rotation == pytest.approx(0.0)


def test_aggregate_rows_applies_rotation_gate_to_held_out_errors() -> None:
    rows = []
    for scale in (1.0, 1.2):
        rows.append(
            {
                "before": np.asarray([30.0, 32.0]) * scale,
                "after": np.asarray([2.0, 2.5]) * scale,
                "displacement": np.asarray([28.0, 29.5]) * scale,
                "valid_observer_count": 7,
                "effective_rank": 6,
                "singular_values": np.asarray([20, 15, 10, 5, 2, 1]),
                "retained_modes": np.ones(6, dtype=bool),
                "prior_control_variance": 100.0,
                "posterior_control_variance": 10.0,
                "twist": np.zeros(6),
            }
        )

    summary = aggregate_rows(rows, "rotation", 10.0)

    assert summary["frames"] == 2
    assert summary["control_anchor_observations"] == 4
    assert summary["error_ratio"] < 0.25
    assert summary["gate_ratio_le_0_25"]
    assert summary["gate_p95_le_12px"]
    assert summary["effective_rank_min"] == 6
    assert summary["posterior_to_prior_variance_ratio"] == pytest.approx(0.1)

