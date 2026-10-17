from pathlib import Path

import numpy as np

from fcpg.so101_synthetic_grounding import (
    SO101CameraDrift,
    SO101SyntheticGroundingRenderer,
)


SCENE = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "so_arm100"
    / "Simulation"
    / "SO101"
    / "scene.xml"
)


def test_zero_drift_has_identical_anchor_projection() -> None:
    with SO101SyntheticGroundingRenderer(SCENE, width=96, height=96) as renderer:
        frame = renderer.render(np.full(6, 50.0), SO101CameraDrift())

    assert frame.rgb.shape == (96, 96, 3)
    assert frame.rgb.dtype == np.uint8
    assert np.array_equal(frame.valid_nominal, frame.valid_true)
    assert np.allclose(frame.q_nominal_px, frame.q_true_px, atol=1e-5)


def test_camera_drift_moves_supported_anchor_pixels() -> None:
    with SO101SyntheticGroundingRenderer(SCENE, width=128, height=128) as renderer:
        frame = renderer.render(
            np.asarray((40.0, 55.0, 45.0, 65.0, 50.0, 20.0)),
            SO101CameraDrift(
                azimuth_degrees=8.0,
                elevation_degrees=2.0,
                distance_meters=0.02,
            ),
        )

    valid = frame.valid_nominal & frame.valid_true
    assert np.any(valid)
    assert float(
        np.mean(np.linalg.norm(frame.q_nominal_px[valid] - frame.q_true_px[valid], axis=-1))
    ) > 1.0
