from __future__ import annotations

import numpy as np

from fcpg.lewm_action_branch_a4c_gripper_metrics import summarize_gripper_support


def _panel() -> dict[str, np.ndarray | list[str]]:
    groups = (
        ("drawer-retrieve-v0", "train", "grasp"),
        ("drawer-retrieve-v0", "train", "release"),
        ("drawer-retrieve-v0", "development", "grasp"),
        ("drawer-retrieve-v0", "development", "release"),
        ("towel-wipe-v0", "train", "grasp_towel"),
        ("towel-wipe-v0", "development", "grasp_towel"),
    )
    count = len(groups)
    semantic = np.broadcast_to(np.arange(8), (count, 8)).copy()
    action = np.zeros((count, 8, 8, 20), dtype=np.float32)
    raw = action.reshape(count, 8, 8, 5, 4)
    raw[:, :, :, :, 0] = 0.2
    raw[:, :, :, :, 3] = 0.5
    raw[:, 6, :, :, 3] = -1.0
    raw[:, 7, :, :, 3] = 1.0
    aperture = np.full((count, 8, 8), 0.4, dtype=np.float32)
    aperture[:, 6] = 0.8
    aperture[:, 7] = 0.1
    contact = np.zeros((count, 8, 8), dtype=bool)
    contact[:, 7] = True
    return {
        "future_action": action,
        "branch_semantic_index": semantic,
        "branch_gripper_aperture": aperture,
        "branch_contact_or_grasp": contact,
        "task_name": [row[0] for row in groups],
        "split": [row[1] for row in groups],
        "phase_name": [row[2] for row in groups],
    }


def test_gripper_support_passes_identifiable_matched_xyz_panel() -> None:
    report = summarize_gripper_support(**_panel())

    assert report["passed"]
    assert report["command_contract_valid_rows"] == report["row_count"] == 6
    assert not report["failed_groups"]


def test_gripper_support_rejects_missing_physical_response() -> None:
    panel = _panel()
    panel["branch_gripper_aperture"] = np.full((6, 8, 8), 0.4, dtype=np.float32)

    report = summarize_gripper_support(**panel)

    assert not report["passed"]
    assert len(report["failed_groups"]) == 6
