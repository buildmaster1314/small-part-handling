from __future__ import annotations

import numpy as np

from fcpg.lewm_action_branch_a4c_actions import (
    SEMANTIC_BRANCHES,
    build_a4c_action_branch_bank,
)


def test_a4c_bank_contains_pure_open_and_close_gripper_counterfactuals() -> None:
    center = np.zeros((40, 4), dtype=np.float32)
    center[:, 0] = 0.2
    center[:20, 3] = -1.0
    center[20:, 3] = 1.0
    permutation_seed = 41

    bank = build_a4c_action_branch_bank(
        center,
        direction_seed=17,
        permutation_seed=permutation_seed,
    )
    semantic_index = np.random.default_rng(permutation_seed).permutation(8)
    by_semantic = bank.actions[np.argsort(semantic_index)]

    assert SEMANTIC_BRANCHES[6:] == (
        "expert_xyz_gripper_open",
        "expert_xyz_gripper_closed",
    )
    np.testing.assert_array_equal(by_semantic[6, :, :3], center[:, :3])
    np.testing.assert_array_equal(by_semantic[7, :, :3], center[:, :3])
    np.testing.assert_array_equal(by_semantic[6, :, 3], -1.0)
    np.testing.assert_array_equal(by_semantic[7, :, 3], 1.0)
    np.testing.assert_array_equal(by_semantic[0], center)


def test_a4c_gripper_counterfactuals_preserve_eight_anonymous_slots() -> None:
    center = np.zeros((40, 4), dtype=np.float32)
    center[:, :3] = np.asarray([0.1, -0.2, 0.05], dtype=np.float32)
    center[:, 3] = 1.0

    bank = build_a4c_action_branch_bank(
        center,
        direction_seed=3,
        permutation_seed=5,
    )

    assert bank.actions.shape == (8, 40, 4)
    assert bank.names == tuple(f"candidate_{index}" for index in range(8))
    assert np.isfinite(bank.actions).all()
    same_xyz = np.all(bank.actions[:, :, :3] == center[None, :, :3], axis=(1, 2))
    assert int(same_xyz.sum()) == 3
