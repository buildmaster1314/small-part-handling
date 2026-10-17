from __future__ import annotations

import numpy as np
import pytest
import torch

from fcpg.lewm_fcpg_paired_invariance import (
    canonical_task_indices,
    clean_identity_loss,
    grounded_valid_mask,
    paired_alignment_loss,
    repeated_action_prefix,
    select_dynamic_grounding,
)


def test_canonical_task_indices_use_shared_seven_task_order() -> None:
    result = canonical_task_indices(["pick-place-v3", "towel-wipe-v0"])
    assert result.tolist() == [0, 6]
    with pytest.raises(ValueError, match="unknown task"):
        canonical_task_indices(["not-a-task"])


def test_grounded_valid_mask_rejects_oob_and_nonfinite_points() -> None:
    valid = np.ones((1, 3), dtype=bool)
    points = np.asarray([[[0.0, 0.0], [223.0, 223.0], [np.nan, 3.0]]])
    result = grounded_valid_mask(valid, points, image_size=(224, 224))
    assert result.tolist() == [[True, True, False]]
    points[0, 1, 0] = 224.0
    assert grounded_valid_mask(valid, points, image_size=(224, 224)).tolist() == [
        [True, False, False]
    ]


def test_dynamic_grounding_is_the_only_mode_switch() -> None:
    nominal = np.zeros((2, 3, 2), dtype=np.float32)
    corrected = np.ones_like(nominal)
    assert np.array_equal(
        select_dynamic_grounding(nominal, corrected, mode="nominal"), nominal
    )
    assert np.array_equal(
        select_dynamic_grounding(nominal, corrected, mode="fcpg"), corrected
    )


def test_repeated_action_prefix_matches_action_prefix_contract() -> None:
    action = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    prefix = repeated_action_prefix(action)
    assert prefix.shape == (2, 8, 20)
    assert torch.equal(prefix[:, 0, :4], action)
    assert torch.equal(prefix[:, 7, 16:20], action)


def test_alignment_and_identity_losses_are_zero_for_equal_states() -> None:
    state = torch.randn(3, 4, 192)
    assert paired_alignment_loss(state, state).item() == pytest.approx(0.0)
    assert clean_identity_loss(state, state).item() == pytest.approx(0.0)
    changed = state.clone()
    changed[:, 1:] += 1.0
    assert paired_alignment_loss(changed, state).item() == pytest.approx(1.0)
