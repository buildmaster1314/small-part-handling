from __future__ import annotations

import numpy as np
import pytest

from scripts.render_lewm_action_prefix_joint_a4_accuracy_gallery import (
    _dynamic_camera_rotation,
)


def test_dynamic_camera_rotation_combines_drift_and_jitter() -> None:
    assert _dynamic_camera_rotation(
        0,
        drift_degrees=6.0,
        jitter_degrees=1.5,
        ramp_steps=120,
        jitter_period_steps=24,
    ) == pytest.approx(0.0)
    assert _dynamic_camera_rotation(
        6,
        drift_degrees=6.0,
        jitter_degrees=1.5,
        ramp_steps=120,
        jitter_period_steps=24,
    ) == pytest.approx(1.8)
    assert _dynamic_camera_rotation(
        120,
        drift_degrees=6.0,
        jitter_degrees=1.5,
        ramp_steps=120,
        jitter_period_steps=24,
    ) == pytest.approx(6.0)


def test_dynamic_camera_rotation_stays_finite_over_episode() -> None:
    values = np.asarray(
        [
            _dynamic_camera_rotation(
                step,
                drift_degrees=5.0,
                jitter_degrees=1.0,
                ramp_steps=80,
                jitter_period_steps=20,
            )
            for step in range(500)
        ]
    )
    assert np.isfinite(values).all()
    assert float(values.min()) >= -1.0 - 1.0e-6
    assert float(values.max()) <= 6.0 + 1.0e-6


@pytest.mark.parametrize("name,value", [("ramp_steps", 0), ("jitter_period_steps", 0)])
def test_dynamic_camera_rotation_rejects_nonpositive_periods(
    name: str, value: int
) -> None:
    kwargs = {
        "step": 0,
        "drift_degrees": 5.0,
        "jitter_degrees": 1.0,
        "ramp_steps": 80,
        "jitter_period_steps": 20,
    }
    kwargs[name] = value
    with pytest.raises(ValueError, match="periods"):
        _dynamic_camera_rotation(**kwargs)
