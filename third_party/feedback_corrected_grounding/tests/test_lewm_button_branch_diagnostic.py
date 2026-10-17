from __future__ import annotations

import numpy as np
import pytest

from fcpg.lewm_button_branch_diagnostic import (
    classify_button_phase,
    endpoint_pairwise_spread,
    flattened_branch_variance,
    spearman_rank_correlation,
    summarize_button_diagnostic,
)


def test_button_phase_classification_prefers_success() -> None:
    assert classify_button_phase(near_object=False, success=False) == "pre_contact"
    assert classify_button_phase(near_object=True, success=False) == "contact"
    assert (
        classify_button_phase(
            near_object=False,
            success=False,
            button_joint=-0.02,
        )
        == "contact"
    )
    assert classify_button_phase(near_object=True, success=True) == "post_success"
    assert classify_button_phase(near_object=False, success=True) == "post_success"


def test_branch_variance_and_endpoint_spread() -> None:
    values = np.zeros((2, 8, 3, 2), dtype=np.float32)
    values[0, :, :, 0] = np.arange(8, dtype=np.float32)[:, None]
    variance = flattened_branch_variance(values)
    assert variance.shape == (2,)
    assert variance[0] > 0.0
    assert variance[1] == 0.0

    spread = endpoint_pairwise_spread(values)
    assert spread.tolist() == pytest.approx([7.0, 0.0])
    assert spearman_rank_correlation(np.arange(4), np.arange(4)) == pytest.approx(1.0)
    assert spearman_rank_correlation(np.arange(4), np.arange(4)[::-1]) == pytest.approx(-1.0)


def test_summary_separates_button_phases() -> None:
    count, branches, horizon = 3, 8, 2
    tcp = np.zeros((count, branches, horizon, 3), dtype=np.float32)
    qpos = np.zeros((count, branches, horizon, 10), dtype=np.float32)
    button = np.zeros_like(tcp)
    button_joint = np.zeros((count, branches, horizon), dtype=np.float32)
    target = np.zeros((count, branches, horizon), dtype=np.float32)
    success = np.zeros((count, branches, horizon), dtype=bool)
    tcp[0, :, :, 0] = np.arange(branches)[:, None]
    button[1, :, :, 1] = np.arange(branches)[:, None]
    success[1, :4, -1] = True

    summary = summarize_button_diagnostic(
        phase=("pre_contact", "contact", "post_success"),
        latent_branch_variance=np.asarray((0.2, 0.03, 0.0001)),
        tcp=tcp,
        qpos=qpos,
        button_position=button,
        button_joint=button_joint,
        object_to_target=target,
        success=success,
        hazard_step=np.asarray((4, 11, 20)),
    )
    assert summary["bundle_count"] == 3
    assert summary["per_phase"]["pre_contact"]["latent_informative_fraction"] == 1.0
    assert summary["per_phase"]["post_success"]["latent_informative_fraction"] == 0.0
    assert summary["per_phase"]["contact"]["success_disagreement_fraction"] == 1.0
    assert summary["per_phase"]["contact"]["source_model_step"]["median"] == 11.0
    assert "tcp_branch_variance" in summary["latent_physical_spearman"]
