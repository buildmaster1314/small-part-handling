from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import analyze_lewm_towel_motion_gain as analysis


def test_motion_gain_recovers_known_scale() -> None:
    current = torch.zeros(1, 3, 2)
    target = torch.ones(1, 8, 3, 2) * 4.0
    prediction = target * 0.5
    result = analysis.motion_gain_rows(
        prediction,
        target,
        current,
        np.asarray([0]),
    )["expert"]
    assert result["mean_signed_motion_gain"] == pytest.approx(0.5)
    assert result["median_motion_norm_ratio"] == pytest.approx(0.5)
    assert result["mean_signed_motion_gain_by_horizon"] == pytest.approx([0.5] * 8)


def test_motion_gain_rejects_cuda1() -> None:
    with pytest.raises(ValueError, match="CUDA device 0"):
        analysis._cuda0("cuda:1")
