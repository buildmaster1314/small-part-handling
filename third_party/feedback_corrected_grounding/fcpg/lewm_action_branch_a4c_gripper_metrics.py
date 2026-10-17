"""Semantic support checks for A4c matched-XYZ gripper interventions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


BRANCH_COUNT = 8
HORIZON = 8
RAW_STEPS_PER_HORIZON = 5
ACTION_DIM = 4
OPEN_SEMANTIC_INDEX = 6
CLOSED_SEMANTIC_INDEX = 7
EXPERT_SEMANTIC_INDEX = 0
MINIMUM_APERTURE_GAP = 0.05
MINIMUM_APERTURE_RESPONSE_FRACTION = 0.80
MINIMUM_CONTACT_DIVERGENCE_FRACTION = 0.50
REQUIRED_PHASES = {
    "drawer-retrieve-v0": ("grasp", "release"),
    "towel-wipe-v0": ("grasp_towel",),
}


def _semantic_slots(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 2 or values.shape[1] != BRANCH_COUNT:
        raise ValueError("branch_semantic_index must have shape [row,8]")
    expected = np.arange(BRANCH_COUNT, dtype=np.int64)
    if not all(np.array_equal(np.sort(row), expected) for row in values):
        raise ValueError("each branch_semantic_index row must be a permutation of 0..7")
    return np.argsort(values.astype(np.int64, copy=False), axis=1)


def summarize_gripper_support(
    *,
    future_action: np.ndarray,
    branch_semantic_index: np.ndarray,
    branch_gripper_aperture: np.ndarray,
    branch_contact_or_grasp: np.ndarray,
    task_name: Sequence[str],
    split: Sequence[str],
    phase_name: Sequence[str],
) -> dict[str, Any]:
    """Require identifiable open/close outcomes in grasp and release phases."""

    actions = np.asarray(future_action)
    if actions.ndim != 4 or actions.shape[1:] != (
        BRANCH_COUNT,
        HORIZON,
        RAW_STEPS_PER_HORIZON * ACTION_DIM,
    ):
        raise ValueError("future_action must have shape [row,8,8,20]")
    row_count = actions.shape[0]
    if not np.isfinite(actions).all():
        raise ValueError("future_action must be finite")
    aperture = np.asarray(branch_gripper_aperture)
    contact = np.asarray(branch_contact_or_grasp)
    if aperture.shape != (row_count, BRANCH_COUNT, HORIZON):
        raise ValueError("branch_gripper_aperture must have shape [row,8,8]")
    if contact.shape != (row_count, BRANCH_COUNT, HORIZON):
        raise ValueError("branch_contact_or_grasp must have shape [row,8,8]")
    if not np.isfinite(aperture).all() or contact.dtype.kind != "b":
        raise ValueError("gripper outcomes must be finite aperture and boolean contact")
    task = np.asarray(task_name).astype(str, copy=False)
    split_value = np.asarray(split).astype(str, copy=False)
    phase = np.asarray(phase_name).astype(str, copy=False)
    if any(value.shape != (row_count,) for value in (task, split_value, phase)):
        raise ValueError("gripper metadata must align with rows")

    slots = _semantic_slots(branch_semantic_index)
    raw_actions = actions.reshape(
        row_count, BRANCH_COUNT, HORIZON, RAW_STEPS_PER_HORIZON, ACTION_DIM
    )
    row_index = np.arange(row_count)
    expert = raw_actions[row_index, slots[:, EXPERT_SEMANTIC_INDEX]]
    opened = raw_actions[row_index, slots[:, OPEN_SEMANTIC_INDEX]]
    closed = raw_actions[row_index, slots[:, CLOSED_SEMANTIC_INDEX]]
    command_valid = (
        np.all(opened[..., :3] == expert[..., :3], axis=(1, 2, 3))
        & np.all(closed[..., :3] == expert[..., :3], axis=(1, 2, 3))
        & np.all(opened[..., 3] == -1.0, axis=(1, 2))
        & np.all(closed[..., 3] == 1.0, axis=(1, 2))
    )

    open_aperture = aperture[row_index, slots[:, OPEN_SEMANTIC_INDEX]]
    closed_aperture = aperture[row_index, slots[:, CLOSED_SEMANTIC_INDEX]]
    open_contact = contact[row_index, slots[:, OPEN_SEMANTIC_INDEX]]
    closed_contact = contact[row_index, slots[:, CLOSED_SEMANTIC_INDEX]]
    aperture_gap = np.max(np.abs(open_aperture - closed_aperture), axis=1)
    contact_diverges = np.any(open_contact != closed_contact, axis=1)

    groups: dict[str, Any] = {}
    failed_groups: list[str] = []
    for task_value, required_phases in REQUIRED_PHASES.items():
        for split_name in ("train", "development"):
            for phase_value in required_phases:
                selected = (
                    (task == task_value)
                    & (split_value == split_name)
                    & (phase == phase_value)
                )
                key = f"{task_value}/{split_name}/{phase_value}"
                count = int(np.count_nonzero(selected))
                aperture_fraction = (
                    float(np.mean(aperture_gap[selected] >= MINIMUM_APERTURE_GAP))
                    if count
                    else 0.0
                )
                contact_fraction = (
                    float(np.mean(contact_diverges[selected])) if count else 0.0
                )
                passed = bool(
                    count > 0
                    and aperture_fraction >= MINIMUM_APERTURE_RESPONSE_FRACTION
                    and contact_fraction >= MINIMUM_CONTACT_DIVERGENCE_FRACTION
                )
                groups[key] = {
                    "row_count": count,
                    "aperture_response_fraction": aperture_fraction,
                    "contact_divergence_fraction": contact_fraction,
                    "passed": passed,
                }
                if not passed:
                    failed_groups.append(key)

    command_passed = bool(np.all(command_valid))
    passed = command_passed and not failed_groups
    return {
        "schema_version": 1,
        "protocol": "fcpg_action_branch_a4c_gripper_support_v1",
        "passed": passed,
        "training_authorized": passed,
        "command_contract_passed": command_passed,
        "command_contract_valid_rows": int(np.count_nonzero(command_valid)),
        "row_count": row_count,
        "minimum_aperture_gap": MINIMUM_APERTURE_GAP,
        "minimum_aperture_response_fraction": MINIMUM_APERTURE_RESPONSE_FRACTION,
        "minimum_contact_divergence_fraction": MINIMUM_CONTACT_DIVERGENCE_FRACTION,
        "groups": groups,
        "failed_groups": failed_groups,
        "claim_boundary": (
            "This Gate checks that matched-XYZ open/close interventions create "
            "identifiable gripper and grasp outcomes. It does not measure model accuracy."
        ),
    }


__all__ = [
    "REQUIRED_PHASES",
    "summarize_gripper_support",
]
