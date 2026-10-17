import numpy as np

from fcpg.so101_synthetic_dataset import (
    balanced_counts,
    clean_partner_indices,
    sample_camera_drift,
)
from scripts.build_so101_synthetic_fcpg_dataset import CAMERAS, camera_context


def test_balanced_counts_preserve_total() -> None:
    counts = balanced_counts(17, 5)

    assert counts.tolist() == [4, 4, 3, 3, 3]
    assert int(counts.sum()) == 17


def test_clean_partner_indices_preserve_view_and_physical_row() -> None:
    arrays = {
        "source_index": np.asarray([0, 0, 0, 0]),
        "source_row": np.asarray([7, 7, 7, 7]),
        "camera_index": np.asarray([0, 1, 0, 1]),
        "drift_variant": np.asarray([0, 0, 1, 1]),
        "episode": np.asarray([2, 2, 2, 2]),
        "task": np.asarray([3, 3, 3, 3]),
        "split": np.asarray([1, 1, 1, 1]),
    }

    np.testing.assert_array_equal(clean_partner_indices(arrays), [0, 1, 0, 1])


def test_camera_drift_sampler_supports_clean_and_bounded_drift() -> None:
    rng = np.random.default_rng(3)
    clean = sample_camera_drift(
        rng,
        clean_probability=1.0,
        max_azimuth_degrees=10.0,
        max_elevation_degrees=4.0,
        max_distance_meters=0.03,
    )
    assert clean.azimuth_degrees == 0.0

    rng = np.random.default_rng(7)
    for _ in range(50):
        drift = sample_camera_drift(
            rng,
            clean_probability=0.0,
            max_azimuth_degrees=10.0,
            max_elevation_degrees=4.0,
            max_distance_meters=0.03,
        )
        assert 1.0 <= abs(drift.azimuth_degrees) <= 10.0
        assert abs(drift.elevation_degrees) <= 4.0
        assert abs(drift.distance_meters) <= 0.03


def test_two_camera_contexts_are_finite_and_distinct() -> None:
    contexts = np.stack([camera_context(camera) for camera in CAMERAS])

    assert contexts.shape == (2, 5)
    assert np.all(np.isfinite(contexts))
    assert not np.allclose(contexts[0], contexts[1])
