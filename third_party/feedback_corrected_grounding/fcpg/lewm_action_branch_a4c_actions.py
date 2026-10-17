"""A4c action branches with identifiable motion and gripper interventions."""

from __future__ import annotations

from typing import Any

import numpy as np

from fcpg.lewm_action_branch_a3 import build_diverse_action_branch_bank
from fcpg.lewm_action_branch_collection import ActionBranchBank


SEMANTIC_BRANCHES = (
    "expert",
    "reverse",
    "hold",
    "half_speed",
    "transverse_positive",
    "transverse_negative",
    "expert_xyz_gripper_open",
    "expert_xyz_gripper_closed",
)


def build_a4c_action_branch_bank(
    center_action: np.ndarray,
    *,
    direction_seed: int,
    permutation_seed: int,
    action_low: Any = -1.0,
    action_high: Any = 1.0,
    motion_delta_rms: float | None = None,
) -> ActionBranchBank:
    """Keep six XYZ interventions and add two matched-XYZ gripper branches."""

    return build_diverse_action_branch_bank(
        center_action,
        direction_seed=direction_seed,
        permutation_seed=permutation_seed,
        action_low=action_low,
        action_high=action_high,
        motion_delta_rms=motion_delta_rms,
        gripper_counterfactuals=True,
    )


__all__ = ["SEMANTIC_BRANCHES", "build_a4c_action_branch_bank"]
