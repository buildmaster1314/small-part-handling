"""Explicit goal-state companion index for the seven-task A4c control rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_a4 import TASKS


PROTOCOL = "lewm_a4c_explicit_goal_state_v1"
POSE_DIM = 7
PROPRIO_DIM = 4
ANCHOR_COUNT = 3


@dataclass(frozen=True, slots=True)
class RecoveredGoalState:
    """World-frame task state recovered by replaying one proposal snapshot."""

    current_proprio: np.ndarray
    object_pose: np.ndarray
    reference_pose: np.ndarray
    tcp_position: np.ndarray
    task_progress: float
    goal_relative_progress: float
    contact_or_grasp_state: bool
    success: bool
    active_stage: bool


def proposal_identity(
    task_name: object,
    task_instance: object,
    environment_seed: object,
    proposal_raw_step: object,
) -> tuple[str, int, int, int]:
    return (
        str(task_name),
        int(task_instance),
        int(environment_seed),
        int(proposal_raw_step),
    )


def selected_proposal_identities(
    control_index: Mapping[str, np.ndarray],
    raw_index: Mapping[str, np.ndarray],
) -> tuple[tuple[str, int, int, int], ...]:
    """Return one deterministic replay request per selected physical state."""

    required_control = {
        "raw_row_index",
        "task_name",
        "task_instance",
        "proposal_raw_step",
        "physical_bundle_index",
    }
    required_raw = {
        "task_name",
        "task_instance",
        "environment_seed",
        "proposal_raw_step",
        "physical_bundle_index",
    }
    if required_control - set(control_index):
        raise ValueError("control index is missing replay identity fields")
    if required_raw - set(raw_index):
        raise ValueError("raw index is missing replay identity fields")
    count = int(np.asarray(control_index["raw_row_index"]).shape[0])
    rows = np.asarray(control_index["raw_row_index"], dtype=np.int64)
    if rows.shape != (count,) or count <= 0:
        raise ValueError("control raw_row_index must be a non-empty vector")
    raw_count = int(np.asarray(raw_index["task_name"]).shape[0])
    if int(rows.min()) < 0 or int(rows.max()) >= raw_count:
        raise ValueError("control raw_row_index is outside the raw replay")

    for name in (
        "task_name",
        "task_instance",
        "proposal_raw_step",
        "physical_bundle_index",
    ):
        control = np.asarray(control_index[name]).astype(str)
        raw = np.asarray(raw_index[name])[rows].astype(str)
        if control.shape != (count,) or not np.array_equal(control, raw):
            raise ValueError(f"control and raw replay disagree on {name}")

    bundle = np.asarray(control_index["physical_bundle_index"], dtype=np.int64)
    identities: list[tuple[str, int, int, int]] = []
    for bundle_id in np.unique(bundle):
        selected = np.flatnonzero(bundle == int(bundle_id))
        raw_selected = rows[selected]
        identity = proposal_identity(
            raw_index["task_name"][raw_selected[0]],
            raw_index["task_instance"][raw_selected[0]],
            raw_index["environment_seed"][raw_selected[0]],
            raw_index["proposal_raw_step"][raw_selected[0]],
        )
        observed = {
            proposal_identity(
                raw_index["task_name"][row],
                raw_index["task_instance"][row],
                raw_index["environment_seed"][row],
                raw_index["proposal_raw_step"][row],
            )
            for row in raw_selected
        }
        if observed != {identity}:
            raise ValueError("one physical bundle maps to multiple replay identities")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("multiple physical bundles map to the same proposal identity")
    return tuple(sorted(identities))


def _finite_vector(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _validate_recovered(value: RecoveredGoalState) -> RecoveredGoalState:
    proprio = _finite_vector(value.current_proprio, (PROPRIO_DIM,), "current_proprio")
    object_pose = _finite_vector(value.object_pose, (POSE_DIM,), "object_pose")
    reference_pose = _finite_vector(value.reference_pose, (POSE_DIM,), "reference_pose")
    tcp = _finite_vector(value.tcp_position, (3,), "tcp_position")
    for name, pose in (("object_pose", object_pose), ("reference_pose", reference_pose)):
        norm = float(np.linalg.norm(pose[3:]))
        if abs(norm - 1.0) > 1.0e-4:
            raise ValueError(f"{name} quaternion must have unit norm")
    task_progress = float(value.task_progress)
    goal_progress = float(value.goal_relative_progress)
    if not np.isfinite(task_progress) or not 0.0 <= task_progress <= 1.0:
        raise ValueError("task_progress must be finite in [0,1]")
    if not np.isfinite(goal_progress) or not 0.0 <= goal_progress <= 1.0:
        raise ValueError("goal_relative_progress must be finite in [0,1]")
    if bool(value.success) or not bool(value.active_stage):
        raise ValueError("A4c control rows must remain active and pre-success")
    return RecoveredGoalState(
        current_proprio=proprio,
        object_pose=object_pose,
        reference_pose=reference_pose,
        tcp_position=tcp,
        task_progress=task_progress,
        goal_relative_progress=goal_progress,
        contact_or_grasp_state=bool(value.contact_or_grasp_state),
        success=False,
        active_stage=True,
    )


def build_a4c_goal_state_index(
    control_index: Mapping[str, np.ndarray],
    raw_index: Mapping[str, np.ndarray],
    raw_current_proprio: np.ndarray,
    raw_current_keypoints_xyz: np.ndarray,
    recovered: Mapping[tuple[str, int, int, int], RecoveredGoalState],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Align replayed goals with every camera row and verify physical identity."""

    identities = selected_proposal_identities(control_index, raw_index)
    expected = set(identities)
    if set(recovered) != expected:
        missing = len(expected - set(recovered))
        extra = len(set(recovered) - expected)
        raise ValueError(f"recovered goal-state coverage differs: missing={missing}, extra={extra}")
    validated = {key: _validate_recovered(value) for key, value in recovered.items()}

    rows = np.asarray(control_index["raw_row_index"], dtype=np.int64)
    count = int(rows.size)
    proprio = np.asarray(raw_current_proprio, dtype=np.float32)[rows]
    anchors = np.asarray(raw_current_keypoints_xyz, dtype=np.float32)[rows]
    if proprio.shape != (count, PROPRIO_DIM) or not np.isfinite(proprio).all():
        raise ValueError("raw current proprio has an invalid shape or value")
    if anchors.shape != (count, ANCHOR_COUNT, 3) or not np.isfinite(anchors).all():
        raise ValueError("raw current world anchors have an invalid shape or value")

    task = np.asarray(control_index["task_name"]).astype(str)
    instance = np.asarray(control_index["task_instance"], dtype=np.int64)
    step = np.asarray(control_index["proposal_raw_step"], dtype=np.int64)
    seed = np.asarray(raw_index["environment_seed"], dtype=np.int64)[rows]
    recovered_rows = [
        validated[proposal_identity(task[i], instance[i], seed[i], step[i])]
        for i in range(count)
    ]
    replay_proprio = np.stack([value.current_proprio for value in recovered_rows])
    replay_proprio_error = np.max(np.abs(proprio - replay_proprio), axis=1).astype(
        np.float32
    )
    maximum_proprio_error = float(replay_proprio_error.max())

    bundle = np.asarray(control_index["physical_bundle_index"], dtype=np.int64)
    maximum_anchor_view_error = 0.0
    for bundle_id in np.unique(bundle):
        selected = np.flatnonzero(bundle == int(bundle_id))
        anchor_error = float(np.max(np.abs(anchors[selected] - anchors[selected[0]])))
        maximum_anchor_view_error = max(maximum_anchor_view_error, anchor_error)
    if maximum_anchor_view_error > 1.0e-6:
        raise ValueError(
            "world anchors vary across camera views: "
            f"max error {maximum_anchor_view_error:.9f}"
        )

    reference_pose = np.stack([value.reference_pose for value in recovered_rows]).astype(
        np.float32
    )
    # The raw visual replay is the actual input source for the current LeWM.
    # Its first three proprio fields are the exact TCP position for that row.
    # A separately replayed contact-rich trajectory can diverge, so recovered
    # object/progress/contact values remain diagnostics and never become labels.
    tcp = proprio[:, :3].copy()
    goal_delta = reference_pose[:, :3] - tcp
    goal_distance = np.linalg.norm(goal_delta, axis=1).astype(np.float32)
    recovered_goal_progress = np.asarray(
        [value.goal_relative_progress for value in recovered_rows], dtype=np.float32
    )
    indexed_goal_progress = np.asarray(
        control_index["goal_relative_progress"], dtype=np.float32
    ).reshape(count)
    maximum_progress_error = float(
        np.max(np.abs(recovered_goal_progress - indexed_goal_progress))
    )
    recovered_contact = np.asarray(
        [value.contact_or_grasp_state for value in recovered_rows], dtype=bool
    )
    indexed_contact = np.asarray(
        control_index["contact_or_grasp_state"], dtype=bool
    ).reshape(count)
    contact_mismatch = recovered_contact != indexed_contact

    payload = {
        "raw_row_index": rows.copy(),
        "physical_bundle_index": bundle.copy(),
        "proposal_id": np.asarray(control_index["proposal_id"]).copy(),
        "task_index": np.asarray(control_index["task_index"], dtype=np.int64).copy(),
        "task_name": task.copy(),
        "task_instance": instance.copy(),
        "environment_seed": seed.copy(),
        "proposal_raw_step": step.copy(),
        "split": np.asarray(control_index["split"]).copy(),
        "camera_clock_hour": np.asarray(
            control_index["camera_clock_hour"], dtype=np.int64
        ).copy(),
        "current_proprio": proprio.copy(),
        "robot_anchor_xyz_world": anchors.copy(),
        "tcp_position_world": tcp,
        "reference_pose_world": reference_pose,
        "goal_delta_from_tcp_world": goal_delta.astype(np.float32),
        "goal_distance_from_tcp_world": goal_distance,
        "replay_proprio_error": replay_proprio_error,
        "protocol": np.asarray(PROTOCOL),
    }
    per_task: dict[str, Any] = {}
    for task_name in TASKS:
        mask = task == task_name
        if not bool(mask.any()):
            raise ValueError(f"goal-state index is missing task {task_name}")
        unique_bundles = np.unique(bundle[mask])
        unique_targets = np.unique(
            np.round(reference_pose[mask, :3], decimals=6), axis=0
        )
        task_progress_error = np.abs(
            recovered_goal_progress[mask] - indexed_goal_progress[mask]
        )
        per_task[task_name] = {
            "rows": int(mask.sum()),
            "physical_bundles": int(unique_bundles.size),
            "unique_reference_positions": int(unique_targets.shape[0]),
            "tcp_to_goal_distance_median_m": float(np.median(goal_distance[mask])),
            "maximum_separate_replay_proprio_error": float(
                replay_proprio_error[mask].max()
            ),
            "separate_replay_proprio_mismatch_fraction": float(
                np.mean(replay_proprio_error[mask] > 1.0e-6)
            ),
            "maximum_source_cache_goal_progress_error": float(
                task_progress_error.max()
            ),
            "source_cache_contact_mismatch_fraction": float(
                contact_mismatch[mask].mean()
            ),
        }
    summary = {
        "protocol": PROTOCOL,
        "rows": count,
        "physical_bundles": len(identities),
        "maximum_separate_replay_proprio_error": maximum_proprio_error,
        "separate_replay_proprio_mismatch_fraction": float(
            np.mean(replay_proprio_error > 1.0e-6)
        ),
        "maximum_cross_view_world_anchor_error": maximum_anchor_view_error,
        "maximum_source_cache_goal_progress_error": maximum_progress_error,
        "source_cache_contact_mismatch_fraction": float(contact_mismatch.mean()),
        "per_task": per_task,
        "interpretation": (
            "reference_pose is deployment-available task geometry omitted from the "
            "original A4c labels; raw TCP/FK are authoritative, while separately "
            "replayed object/progress/contact values are intentionally not serialized"
        ),
    }
    return payload, summary


__all__ = [
    "PROTOCOL",
    "RecoveredGoalState",
    "build_a4c_goal_state_index",
    "proposal_identity",
    "selected_proposal_identities",
]
