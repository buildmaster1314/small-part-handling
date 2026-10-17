from __future__ import annotations

import numpy as np

from scripts.evaluate_action_chunk_offline import representation_statistics


def test_representation_statistics_recovers_two_dimensional_rank() -> None:
    latent = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )

    stats = representation_statistics(latent)

    assert stats["latent_samples"] == 4
    assert stats["latent_dim"] == 2
    assert np.isclose(stats["latent_mean_norm"], 0.0)
    assert np.isclose(stats["latent_effective_rank"], 2.0)
    assert np.isclose(stats["latent_participation_ratio"], 2.0)


def test_representation_statistics_reports_collapsed_latent() -> None:
    stats = representation_statistics(np.ones((5, 3), dtype=np.float32))

    assert np.isclose(stats["latent_std_mean"], 0.0)
    assert np.isclose(stats["latent_effective_rank"], 0.0)
    assert np.isclose(stats["latent_participation_ratio"], 0.0)
