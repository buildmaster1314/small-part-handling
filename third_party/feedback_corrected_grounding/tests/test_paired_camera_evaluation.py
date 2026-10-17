from __future__ import annotations

import numpy as np
import pytest

from fcpg.paired_camera_evaluation import (
    camera_phase_labels,
    summarize_paired_camera_arrays,
)


def _summary() -> dict[str, object]:
    q_nominal = np.array(
        [[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]], dtype=np.float32
    )
    q_true = q_nominal + np.array([[[2.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]])
    q_dynamic = q_nominal + np.array([[[1.0, 0.0]], [[0.5, 0.0]], [[0.0, 0.0]]])
    q_clean = q_nominal.copy()
    latent_clean = np.zeros((3, 1, 2), dtype=np.float32)
    latent_dynamic_nominal = np.array([[[2.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]])
    latent_dynamic_corrected = latent_dynamic_nominal * 0.5
    return summarize_paired_camera_arrays(
        q_nominal=q_nominal,
        q_true=q_true,
        q_dynamic_corrected=q_dynamic,
        q_clean_corrected=q_clean,
        valid_nominal=np.ones((3, 1), dtype=bool),
        valid_true=np.ones((3, 1), dtype=bool),
        latent_dynamic_nominal=latent_dynamic_nominal,
        latent_clean_nominal=latent_clean,
        latent_dynamic_corrected=latent_dynamic_corrected,
        latent_dynamic_oracle=latent_dynamic_corrected,
        latent_clean_corrected=latent_clean,
        task_names=["pick", "pick", "reach"],
        trajectory_phase=np.array([0.0, 0.5, 1.0]),
        intervention_onset=np.array([False, True, False]),
        image_size=(16, 16),
    )


def test_phase_labels_are_stable() -> None:
    assert camera_phase_labels(np.array([0.0, 0.25, 1.0])).tolist() == [
        "pre",
        "moving",
        "post",
    ]


def test_paired_summary_reports_pixel_and_latent_ratios() -> None:
    summary = _summary()
    overall = summary["all"]
    assert overall["fcpg_pixel_ratio"] == pytest.approx(0.5)
    assert overall["fcpg_latent_ratio"] == pytest.approx(0.5)
    assert overall["oracle_latent_ratio"] == pytest.approx(0.5)
    assert overall["fcpg_to_oracle_latent_ratio"] == pytest.approx(1.0)
    assert overall["fcpg_clean_epe_px"]["mean"] == 0.0
    assert set(summary["by_phase"]) == {"pre", "moving", "post"}
    assert summary["intervention_onset"]["frames"] == 1


def test_paired_summary_rejects_misaligned_latents() -> None:
    with pytest.raises(ValueError, match="latent arrays"):
        summarize_paired_camera_arrays(
            q_nominal=np.zeros((2, 1, 2)),
            q_true=np.zeros((2, 1, 2)),
            q_dynamic_corrected=np.zeros((2, 1, 2)),
            q_clean_corrected=np.zeros((2, 1, 2)),
            valid_nominal=np.ones((2, 1), dtype=bool),
            valid_true=np.ones((2, 1), dtype=bool),
            latent_dynamic_nominal=np.zeros((2, 2)),
            latent_clean_nominal=np.zeros((2, 1, 2)),
            latent_dynamic_corrected=np.zeros((2, 1, 2)),
            latent_dynamic_oracle=np.zeros((2, 1, 2)),
            latent_clean_corrected=np.zeros((2, 1, 2)),
            task_names=["a", "a"],
            trajectory_phase=np.zeros(2),
            intervention_onset=np.zeros(2, dtype=bool),
            image_size=(16, 16),
        )
