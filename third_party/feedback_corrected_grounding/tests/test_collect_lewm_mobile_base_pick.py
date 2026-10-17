from __future__ import annotations

import numpy as np

from fcpg.lewm_mobile_base import BASE_PREFIX_DIM
from scripts.collect_lewm_mobile_base_pick import (
    BRANCH_COUNT,
    HORIZON,
    build_base_action_bank,
)


def test_base_action_bank_is_deterministic_diverse_and_bounded() -> None:
    first = build_base_action_bank(3, seed=41, speed_m=0.0012)
    second = build_base_action_bank(3, seed=41, speed_m=0.0012)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, BRANCH_COUNT, HORIZON, BASE_PREFIX_DIM)
    raw = first.reshape(3, BRANCH_COUNT, HORIZON * 5, 2)
    speed = np.linalg.norm(raw, axis=-1)
    assert float(speed.min()) >= 0.0012 * 0.69
    assert float(speed.max()) <= 0.0012 * 1.01
    endpoint = raw.sum(axis=2)
    assert np.unique(np.round(endpoint[0], decimals=6), axis=0).shape[0] == BRANCH_COUNT


def test_base_action_bank_changes_bundle_orientation() -> None:
    bank = build_base_action_bank(2, seed=77, speed_m=0.001)
    assert not np.array_equal(bank[0], bank[1])
