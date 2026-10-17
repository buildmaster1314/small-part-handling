"""Audited seven-task control index derived from the A4c replay panel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from fcpg.lewm_action_branch_a4 import HORIZON, TASKS
from fcpg.lewm_action_prefix_joint_a4 import ACTION_DIM
from fcpg.lewm_action_prefix_joint_data import accepted_ledger_rows


PROTOCOL = "lewm_a4c_phase_aware_control_index_v1"
RAW_STEPS_PER_MODEL_STEP = 5
RAW_HORIZON = HORIZON * RAW_STEPS_PER_MODEL_STEP
EXPERT_SEMANTIC_INDEX = 0


def _vector(value: object, count: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}")
    return result


def _phase_success_lookup(
    schedule: Mapping[str, Any],
) -> dict[tuple[str, int], int | None]:
    episodes = schedule.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("phase schedule must contain episode rows")
    result: dict[tuple[str, int], int | None] = {}
    for row in episodes:
        if not isinstance(row, Mapping):
            raise TypeError("phase schedule episode must be a mapping")
        key = (str(row["task"]), int(row["task_instance"]))
        if key in result:
            raise ValueError("phase schedule contains duplicate task instances")
        value = row.get("first_success_step")
        if value is not None and (isinstance(value, bool) or int(value) < 0):
            raise ValueError("phase schedule contains an invalid success step")
        result[key] = None if value is None else int(value)
    return result


def _phase_weights(
    task_name: np.ndarray,
    split: np.ndarray,
    phase_name: np.ndarray,
) -> np.ndarray:
    weights = np.empty(len(task_name), dtype=np.float32)
    for task in TASKS:
        for split_name in ("train", "development"):
            group = (task_name == task) & (split == split_name)
            if not bool(group.any()):
                raise ValueError(f"missing {task} {split_name} rows")
            phases, counts = np.unique(phase_name[group], return_counts=True)
            count_by_phase = dict(zip(phases.tolist(), counts.tolist(), strict=True))
            for phase, count in count_by_phase.items():
                weights[group & (phase_name == phase)] = 1.0 / float(count)
            weights[group] /= float(weights[group].mean())
    return weights


def build_a4c_control_index(
    raw_index: Mapping[str, np.ndarray],
    state_cache: Mapping[str, np.ndarray],
    ledger: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, np.ndarray],
    phase_schedule: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract exact expert prefixes and reject phase-biased control data."""

    required_raw = {
        "cache_index",
        "proposal_id",
        "task_index",
        "task_name",
        "task_instance",
        "proposal_raw_step",
        "source_episode_id",
        "split",
        "future_action",
        "proposal_phase_name",
        "physical_bundle_index",
        "camera_name",
        "camera_clock_hour",
    }
    missing = sorted(required_raw - set(raw_index))
    if missing:
        raise ValueError(f"raw replay index is missing fields: {missing}")
    count = int(np.asarray(raw_index["cache_index"]).shape[0])
    if count <= 0:
        raise ValueError("raw replay index is empty")
    for name in required_raw - {"future_action"}:
        _vector(raw_index[name], count, name)
    future_action = np.asarray(raw_index["future_action"], dtype=np.float32)
    if future_action.shape != (count, 8, HORIZON, ACTION_DIM):
        raise ValueError("future_action must have shape [N,8,8,20]")
    if not np.isfinite(future_action).all():
        raise ValueError("future_action must be finite")

    cache_count = int(np.asarray(state_cache["proposal_id"]).shape[0])
    cache_index = np.asarray(raw_index["cache_index"], dtype=np.int64)
    if int(cache_index.min()) < 0 or int(cache_index.max()) >= cache_count:
        raise ValueError("raw replay cache_index is outside the state cache")
    for raw_name, cache_name in (
        ("proposal_id", "proposal_id"),
        ("task_index", "task_class_index"),
        ("task_name", "task_name"),
        ("task_instance", "task_instance_index"),
        ("source_episode_id", "source_episode_id"),
        ("split", "split"),
    ):
        raw_value = np.asarray(raw_index[raw_name]).astype(str)
        cache_value = np.asarray(state_cache[cache_name])[cache_index].astype(str)
        if not np.array_equal(raw_value, cache_value):
            raise ValueError(f"raw replay {raw_name} differs from the state cache")

    semantics = np.asarray(state_cache["branch_semantic_index"], dtype=np.int64)[
        cache_index
    ]
    if semantics.shape != (count, 8):
        raise ValueError("branch_semantic_index must have shape [N,8]")
    expert_mask = semantics == EXPERT_SEMANTIC_INDEX
    if not bool(np.all(expert_mask.sum(axis=1) == 1)):
        raise ValueError("each row must contain exactly one expert branch")
    expert_slot = np.argmax(expert_mask, axis=1).astype(np.int64)
    replay_expert_prefix = future_action[np.arange(count), expert_slot]
    center_action = np.asarray(state_cache["center_action"], dtype=np.float32)[
        cache_index
    ]
    history_action = np.asarray(state_cache["history_action"], dtype=np.float32)[
        cache_index
    ]
    if history_action.shape != (count, 2, ACTION_DIM) or not np.isfinite(
        history_action
    ).all():
        raise ValueError("history_action must have shape [N,2,20] and be finite")
    if center_action.shape != replay_expert_prefix.shape:
        raise ValueError("center_action shape differs from the expert prefix")
    xyz_columns = np.asarray(
        [index for index in range(ACTION_DIM) if index % 4 != 3], dtype=np.int64
    )
    gripper_columns = np.asarray(
        [index for index in range(ACTION_DIM) if index % 4 == 3], dtype=np.int64
    )
    maximum_replay_xyz_error = float(
        np.max(
            np.abs(
                replay_expert_prefix[:, :, xyz_columns]
                - center_action[:, :, xyz_columns]
            )
        )
    )
    if maximum_replay_xyz_error > 1.0e-6:
        raise ValueError(
            "expert replay XYZ differs from center_action: "
            f"max error {maximum_replay_xyz_error:.8f}"
        )
    replay_gripper_mismatch = np.abs(
        replay_expert_prefix[:, :, gripper_columns]
        - center_action[:, :, gripper_columns]
    ) > 1.0e-6
    gripper_mismatch_fraction = float(replay_gripper_mismatch.mean())
    expert_prefix = np.ascontiguousarray(center_action, dtype=np.float32)

    accepted = accepted_ledger_rows(state_cache, ledger)
    ledger_rows = np.asarray([accepted[int(index)] for index in cache_index])
    for name in ("success", "ever_success_before"):
        value = np.asarray(ledger[name], dtype=bool)[ledger_rows]
        if bool(value.any()):
            raise ValueError(f"control index contains {name} rows")
    if not bool(np.all(np.asarray(ledger["active_stage"], dtype=bool)[ledger_rows])):
        raise ValueError("control index contains non-active rows")
    if not np.array_equal(
        np.asarray(ledger["proposal_raw_step"], dtype=np.int64)[ledger_rows],
        np.asarray(raw_index["proposal_raw_step"], dtype=np.int64),
    ):
        raise ValueError("raw replay proposal steps differ from the ledger")

    diagnostic_ids = np.asarray(diagnostics["proposal_id"]).astype(str)
    cache_ids = np.asarray(state_cache["proposal_id"]).astype(str)
    if not np.array_equal(diagnostic_ids, cache_ids):
        raise ValueError("diagnostics and state cache proposal order differ")
    progress = np.asarray(diagnostics["goal_relative_progress"], dtype=np.float32)[
        cache_index
    ]
    contact = np.asarray(diagnostics["contact_or_grasp_state"], dtype=bool)[
        cache_index
    ]
    if progress.shape != (count, 1) or contact.shape != (count, 1):
        raise ValueError("diagnostic control labels have invalid shapes")

    task_name = np.asarray(raw_index["task_name"]).astype(str)
    split = np.asarray(raw_index["split"]).astype(str)
    phase_name = np.asarray(raw_index["proposal_phase_name"]).astype(str)
    task_set = set(task_name.tolist())
    if task_set != set(TASKS):
        raise ValueError("control index must contain the complete seven-task panel")
    if set(split.tolist()) != {"train", "development"}:
        raise ValueError("control index must contain only train and development rows")
    if bool(np.any(phase_name == "")):
        raise ValueError("control index contains empty phase labels")

    clock = np.asarray(raw_index["camera_clock_hour"], dtype=np.int64)
    if not bool(np.all(clock[split == "train"] % 2 == 0)):
        raise ValueError("training rows must use even clock views")
    if not bool(np.all(clock[split == "development"] % 2 == 1)):
        raise ValueError("development rows must use odd clock views")

    source_episode = np.asarray(raw_index["source_episode_id"]).astype(str)
    train_episodes = set(zip(task_name[split == "train"], source_episode[split == "train"]))
    dev_episodes = set(
        zip(task_name[split == "development"], source_episode[split == "development"])
    )
    if train_episodes & dev_episodes:
        raise ValueError("physical episodes overlap train and development")

    proposal_step = np.asarray(raw_index["proposal_raw_step"], dtype=np.int64)
    physical_bundle = np.asarray(raw_index["physical_bundle_index"], dtype=np.int64)
    per_task: dict[str, Any] = {}
    for task in TASKS:
        task_summary: dict[str, Any] = {}
        for split_name in ("train", "development"):
            mask = (task_name == task) & (split == split_name)
            if not bool(mask.any()):
                raise ValueError(f"missing {task} {split_name} rows")
            unique_steps = np.unique(proposal_step[mask])
            if unique_steps.size < 8 or int(unique_steps[-1] - unique_steps[0]) < 30:
                raise ValueError(
                    f"{task} {split_name} remains biased to a narrow time window"
                )
            phase_count = int(np.unique(phase_name[mask]).size)
            if phase_count < 2:
                raise ValueError(f"{task} {split_name} lacks phase diversity")
            task_summary[split_name] = {
                "rows": int(mask.sum()),
                "physical_bundles": int(np.unique(physical_bundle[mask]).size),
                "episodes": int(np.unique(source_episode[mask]).size),
                "phase_count": phase_count,
                "proposal_step_min": int(proposal_step[mask].min()),
                "proposal_step_median": float(np.median(proposal_step[mask])),
                "proposal_step_max": int(proposal_step[mask].max()),
            }
        per_task[task] = task_summary

    for split_name in ("train", "development"):
        task_counts = [int(np.sum((task_name == task) & (split == split_name))) for task in TASKS]
        if len(set(task_counts)) != 1:
            raise ValueError(f"{split_name} rows are not task balanced")

    success_lookup = _phase_success_lookup(phase_schedule)
    task_instance = np.asarray(raw_index["task_instance"], dtype=np.int64)
    crosses_success = np.zeros(count, dtype=bool)
    for index in range(count):
        first_success = success_lookup.get((task_name[index], int(task_instance[index])))
        if first_success is not None:
            start = int(proposal_step[index])
            crosses_success[index] = start < first_success <= start + RAW_HORIZON

    action_rms = np.sqrt(np.mean(expert_prefix * expert_prefix, axis=(1, 2)))
    if bool(np.any(action_rms < 1.0e-3)):
        raise ValueError("control index contains near-zero expert prefixes")

    sample_weight = _phase_weights(task_name, split, phase_name)
    payload = {
        "raw_row_index": np.arange(count, dtype=np.int64),
        "cache_index": cache_index,
        "proposal_id": np.asarray(raw_index["proposal_id"]),
        "task_index": np.asarray(raw_index["task_index"], dtype=np.int64),
        "task_name": np.asarray(raw_index["task_name"]),
        "task_instance": task_instance,
        "source_episode_id": np.asarray(raw_index["source_episode_id"]),
        "split": np.asarray(raw_index["split"]),
        "proposal_raw_step": proposal_step,
        "proposal_phase_name": np.asarray(raw_index["proposal_phase_name"]),
        "physical_bundle_index": physical_bundle,
        "camera_name": np.asarray(raw_index["camera_name"]),
        "camera_clock_hour": clock,
        "expert_branch_slot": expert_slot,
        "expert_action_prefix": np.ascontiguousarray(expert_prefix, dtype=np.float32),
        "history_action": np.ascontiguousarray(history_action, dtype=np.float32),
        "goal_relative_progress": np.ascontiguousarray(progress, dtype=np.float32),
        "contact_or_grasp_state": np.ascontiguousarray(contact, dtype=bool),
        "h8_crosses_success": crosses_success,
        "sample_weight": sample_weight,
        "protocol": np.asarray(PROTOCOL),
    }
    summary = {
        "protocol": PROTOCOL,
        "rows": count,
        "tasks": list(TASKS),
        "maximum_replay_expert_xyz_error": maximum_replay_xyz_error,
        "replay_expert_gripper_mismatch_fraction": gripper_mismatch_fraction,
        "train_rows": int(np.sum(split == "train")),
        "development_rows": int(np.sum(split == "development")),
        "train_physical_bundles": int(np.unique(physical_bundle[split == "train"]).size),
        "development_physical_bundles": int(
            np.unique(physical_bundle[split == "development"]).size
        ),
        "h8_crosses_success_fraction": float(crosses_success.mean()),
        "action_rms": {
            "minimum": float(action_rms.min()),
            "median": float(np.median(action_rms)),
            "maximum": float(action_rms.max()),
        },
        "per_task": per_task,
        "limitations": [
            "expert-centered on-manifold states only",
            "does not contain recovery actions from policy-induced deviations",
            "authorizes a coarse proposal pilot, not a closed-loop control claim",
        ],
    }
    return payload, summary


__all__ = [
    "PROTOCOL",
    "build_a4c_control_index",
]
