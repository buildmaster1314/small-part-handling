from __future__ import annotations

import numpy as np
import pytest

from fcpg.lewm_a4c_goal_state import (
    RecoveredGoalState,
    build_a4c_goal_state_index,
    proposal_identity,
    selected_proposal_identities,
)
from fcpg.lewm_action_branch_a4 import TASKS


def _fixture() -> tuple[dict, dict, np.ndarray, np.ndarray, dict]:
    control_rows = []
    raw_rows = []
    proprio_rows = []
    anchor_rows = []
    recovered = {}
    raw_row = 0
    bundle = 0
    for task_index, task in enumerate(TASKS):
        for local_bundle in range(2):
            instance = task_index * 10 + local_bundle
            seed = 620000 + instance
            step = 10 + local_bundle * 5
            current_proprio = np.asarray(
                [0.1 * task_index, 0.6, 0.2, 0.8], dtype=np.float32
            )
            object_pose = np.asarray(
                [0.01 * task_index, 0.7, 0.05, 0, 0, 0, 1], dtype=np.float32
            )
            target_x = 0.2 if task != "towel-wipe-v0" else (-0.2 if local_bundle == 0 else 0.2)
            reference_pose = np.asarray(
                [target_x, 0.8, 0.1, 0, 0, 0, 1], dtype=np.float32
            )
            tcp = np.asarray([0.03 * task_index, 0.65, 0.15], dtype=np.float32)
            key = proposal_identity(task, instance, seed, step)
            recovered[key] = RecoveredGoalState(
                current_proprio=current_proprio,
                object_pose=object_pose,
                reference_pose=reference_pose,
                tcp_position=tcp,
                task_progress=0.25 + 0.1 * local_bundle,
                goal_relative_progress=0.2 + 0.1 * local_bundle,
                contact_or_grasp_state=bool(local_bundle),
                success=False,
                active_stage=True,
            )
            anchors = np.full((3, 3), 0.01 * bundle, dtype=np.float32)
            for clock in (1, 3):
                proposal = f"{task}:{instance}:{step}"
                raw_rows.append(
                    {
                        "task_name": task,
                        "task_instance": instance,
                        "environment_seed": seed,
                        "proposal_raw_step": step,
                        "physical_bundle_index": bundle,
                    }
                )
                control_rows.append(
                    {
                        "raw_row_index": raw_row,
                        "physical_bundle_index": bundle,
                        "proposal_id": proposal,
                        "task_index": task_index,
                        "task_name": task,
                        "task_instance": instance,
                        "proposal_raw_step": step,
                        "split": "development",
                        "camera_clock_hour": clock,
                        "goal_relative_progress": [0.2 + 0.1 * local_bundle],
                        "contact_or_grasp_state": [bool(local_bundle)],
                    }
                )
                proprio_rows.append(current_proprio)
                anchor_rows.append(anchors)
                raw_row += 1
            bundle += 1

    def arrays(rows: list[dict]) -> dict[str, np.ndarray]:
        return {name: np.asarray([row[name] for row in rows]) for name in rows[0]}

    return (
        arrays(control_rows),
        arrays(raw_rows),
        np.asarray(proprio_rows, dtype=np.float32),
        np.asarray(anchor_rows, dtype=np.float32),
        recovered,
    )


def test_selected_identities_deduplicate_camera_views() -> None:
    control, raw, _proprio, _anchors, recovered = _fixture()
    identities = selected_proposal_identities(control, raw)
    assert len(identities) == len(TASKS) * 2
    assert set(identities) == set(recovered)


def test_goal_state_index_uses_raw_tcp_for_goal_relative_geometry() -> None:
    control, raw, proprio, anchors, recovered = _fixture()
    payload, summary = build_a4c_goal_state_index(
        control, raw, proprio, anchors, recovered
    )
    np.testing.assert_allclose(
        payload["goal_delta_from_tcp_world"],
        payload["reference_pose_world"][:, :3] - payload["current_proprio"][:, :3],
    )
    assert summary["physical_bundles"] == len(TASKS) * 2
    assert summary["per_task"]["towel-wipe-v0"]["unique_reference_positions"] == 2


def test_goal_state_index_reports_separate_replay_proprio_mismatch() -> None:
    control, raw, proprio, anchors, recovered = _fixture()
    proprio[0, 0] += 0.01
    payload, summary = build_a4c_goal_state_index(
        control, raw, proprio, anchors, recovered
    )
    assert float(payload["replay_proprio_error"][0]) == pytest.approx(0.01)
    assert summary["maximum_separate_replay_proprio_error"] == pytest.approx(0.01)


def test_goal_state_index_rejects_cross_view_world_anchor_drift() -> None:
    control, raw, proprio, anchors, recovered = _fixture()
    anchors[1, 0, 0] += 0.01
    with pytest.raises(ValueError, match="world anchors vary across camera views"):
        build_a4c_goal_state_index(control, raw, proprio, anchors, recovered)


def test_goal_state_index_rejects_missing_recovered_state() -> None:
    control, raw, proprio, anchors, recovered = _fixture()
    recovered.pop(next(iter(recovered)))
    with pytest.raises(ValueError, match="coverage differs"):
        build_a4c_goal_state_index(control, raw, proprio, anchors, recovered)


def test_selected_identities_reject_control_raw_mismatch() -> None:
    control, raw, _proprio, _anchors, _recovered = _fixture()
    raw["proposal_raw_step"][0] += 1
    with pytest.raises(ValueError, match="disagree on proposal_raw_step"):
        selected_proposal_identities(control, raw)
