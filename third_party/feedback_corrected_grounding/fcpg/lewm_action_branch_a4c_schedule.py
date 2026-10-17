"""Phase-aware proposal schedules for the seven-task A4c data refresh.

The frozen A4 protocol samples proposal times uniformly from the first 160 raw
steps.  That is appropriate for an active-control smoke gate, but it does not
cover long tasks such as drawer retrieval and towel wiping.  A4c keeps the
same branch schema and H=8 horizon while selecting windows from a complete
expert replay before any matched branches are executed.

The replay is used only to design a deterministic data-collection schedule.
The schedule is materialized before branch collection and the branch outcome
is never used to select or discard a proposal.
"""

from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.lewm_a4_task_factory import make_a4_env, make_a4_expert
from fcpg.lewm_a4_task_signals import extract_a4_task_signals
from fcpg.lewm_action_branch_a4 import (
    FORMAL_TASK_INSTANCES,
    PROTOCOL as A4_PROTOCOL,
    RAW_STEPS_PER_MODEL_STEP,
    STATUS as A4_STATUS,
    TASKS,
    assign_a4_splits,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_collection import (
    a4_camera_name,
    recent_xyz_action_rms,
)
from fcpg.lewm_action_branch_a4c_actions import SEMANTIC_BRANCHES


PROTOCOL = "fcpg_action_branch_phase_schedule_a4c_v4"
STATUS = "collection_protocol_ready"
QUESTION = (
    "Can one shared seven-task LeWM learn action-conditioned transitions across "
    "the complete active task trajectory when data are sampled by task phase?"
)
RAW_HORIZON = 8 * RAW_STEPS_PER_MODEL_STEP
HISTORY_RAW_STEPS = 2 * RAW_STEPS_PER_MODEL_STEP
MINIMUM_PROPOSAL_RAW_STEP = HISTORY_RAW_STEPS
MINIMUM_SEPARATION_RAW_STEPS = 5
PROPOSALS_PER_EPISODE = 12
ACTIVE_ACTION_NORM = 0.05
MINIMUM_WORKSPACE_CONTROL_MARGIN = 0.05
REQUIRED_PHASE_IDS = {
    "drawer-retrieve-v0": (
        "pull_open",
        "lift_after_open",
        "move_above_block",
        "descend",
        "grasp",
        "lift_block",
        "transport_above",
        "lower",
        "release",
    ),
    "towel-wipe-v0": (
        "wipe_right_cycle_0",
        "wipe_left_cycle_1",
        "wipe_right_cycle_2",
        "wipe_left_cycle_3",
        "wipe_right_cycle_4",
        "wipe_left_cycle_5",
        "lift_after_wipe",
    ),
}

_A4_QUESTION = (
    "Can one shared seven-task LeWM propagate action-conditioned robot anchors "
    "and task context from genuinely active-control states?"
)
_A4_PROPOSAL_SCHEDULE = {
    "drawn_before_environment_reset": True,
    "independent_of_state_action_reward_task_phase_and_outcomes": True,
    "sampling": "uniform_without_replacement",
    "proposals_per_episode": 12,
    "minimum_raw_step": 3,
    "maximum_raw_step_inclusive": 160,
    "minimum_separation_raw_steps": 5,
}
_A4C_PROPOSAL_SCHEDULE = {
    "materialized_before_branch_execution": True,
    "uses_expert_replay_for_collection_design": True,
    "uses_branch_outcomes": False,
    "sampling": "phase_contact_progress_success_boundary_stratified",
    "proposals_per_episode": PROPOSALS_PER_EPISODE,
    "minimum_raw_step": MINIMUM_PROPOSAL_RAW_STEP,
    "maximum_raw_step": "full_expert_trajectory_minus_horizon",
    "minimum_separation_raw_steps": MINIMUM_SEPARATION_RAW_STEPS,
    "branch_horizon_raw_steps": RAW_HORIZON,
    "required_phase_ids": {
        task: list(phases) for task, phases in REQUIRED_PHASE_IDS.items()
    },
    "require_success_crossing_for_successful_source": True,
}
_A4_MATCHED_STATE_BRANCHES = {
    "branch_count": 8,
    "interruption_horizon_steps": 8,
    "raw_steps_per_model_step": RAW_STEPS_PER_MODEL_STEP,
    "interruption_horizon_raw_steps": RAW_HORIZON,
    "semantic_bank_before_permutation": [
        "expert",
        "reverse",
        "hold",
        "half_speed",
        "transverse_u_positive",
        "transverse_u_negative",
        "transverse_v_positive",
        "transverse_v_negative",
    ],
    "all_branches_executed_and_saved_after_acceptance": True,
    "slot_order_randomized_per_snapshot": True,
}
_A4C_MATCHED_STATE_BRANCHES = {
    **_A4_MATCHED_STATE_BRANCHES,
    "semantic_bank_before_permutation": list(SEMANTIC_BRANCHES),
}


def load_a4c_contract(path: str | Path) -> dict[str, Any]:
    """Load the compact A4c override and materialize its frozen A4 base."""

    source = Path(path).expanduser().resolve()
    override = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(override, Mapping):
        raise TypeError("A4c config must contain one JSON object")
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "base_contract",
        "question",
        "proposal_schedule",
        "matched_state_branches",
    }
    if set(override) != expected_keys:
        raise ValueError("A4c config keys drifted")
    base_name = override["base_contract"]
    if not isinstance(base_name, str) or not base_name:
        raise TypeError("A4c base_contract must be a non-empty path")
    base_path = (source.parent / base_name).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    validate_a4_contract(base)
    contract = deepcopy(base)
    contract["schema_version"] = override["schema_version"]
    contract["protocol"] = override["protocol"]
    contract["status"] = override["status"]
    contract["question"] = override["question"]
    contract["proposal_schedule"] = deepcopy(override["proposal_schedule"])
    contract["matched_state_branches"] = deepcopy(
        override["matched_state_branches"]
    )
    validate_a4c_contract(contract)
    return contract


def validate_a4c_contract(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate A4c while reusing every unchanged frozen A4 field."""

    if not isinstance(contract, Mapping):
        raise TypeError("A4c contract must be a mapping")
    if contract.get("schema_version") != 1:
        raise ValueError("A4c schema_version drifted")
    if contract.get("protocol") != PROTOCOL:
        raise ValueError("A4c protocol drifted")
    if contract.get("status") != STATUS:
        raise ValueError("A4c status drifted")
    if contract.get("question") != QUESTION:
        raise ValueError("A4c question drifted")
    if contract.get("proposal_schedule") != _A4C_PROPOSAL_SCHEDULE:
        raise ValueError("A4c proposal schedule drifted")
    if contract.get("matched_state_branches") != _A4C_MATCHED_STATE_BRANCHES:
        raise ValueError("A4c matched-state branches drifted")

    base_view = deepcopy(dict(contract))
    base_view["protocol"] = A4_PROTOCOL
    base_view["status"] = A4_STATUS
    base_view["question"] = _A4_QUESTION
    base_view["proposal_schedule"] = deepcopy(_A4_PROPOSAL_SCHEDULE)
    base_view["matched_state_branches"] = deepcopy(_A4_MATCHED_STATE_BRANCHES)
    return validate_a4_contract(base_view)


def a4c_base_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the unchanged A4 view used by shared split utilities."""

    validate_a4c_contract(contract)
    base_view = deepcopy(dict(contract))
    base_view["protocol"] = A4_PROTOCOL
    base_view["status"] = A4_STATUS
    base_view["question"] = _A4_QUESTION
    base_view["proposal_schedule"] = deepcopy(_A4_PROPOSAL_SCHEDULE)
    base_view["matched_state_branches"] = deepcopy(_A4_MATCHED_STATE_BRANCHES)
    validate_a4_contract(base_view)
    return base_view


def _success(info: Mapping[str, Any] | None) -> bool:
    if not isinstance(info, Mapping):
        return False
    value = info.get("success", False)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _phase_name(
    task: str,
    expert: Any,
    observation: np.ndarray,
    *,
    progress: float,
    contact: bool,
    success: bool,
) -> str:
    if success:
        return "success"
    phase_method = getattr(expert, "phase", None)
    if callable(phase_method):
        try:
            phase = phase_method(observation)
            value = getattr(phase, "value", phase)
            if isinstance(value, str) and value:
                return value
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    # Standard MetaWorld tasks do not expose a common phase enum.  Progress and
    # contact are still stable, deployment-independent labels for scheduling.
    del task
    progress_bin = min(9, max(0, int(np.floor(float(progress) * 10.0))))
    return f"{'contact' if contact else 'free'}_p{progress_bin}"


def _signals(
    task: str, env: Any, observation: np.ndarray, info: Mapping[str, Any]
) -> tuple[float, bool, bool, float, bool, str | None]:
    """Read planning signals, retaining an explicit fallback diagnostic."""

    try:
        signals = extract_a4_task_signals(task, env, observation, info)
        diagnostics = signals.diagnostics
        return (
            float(diagnostics.goal_relative_progress),
            bool(diagnostics.contact_or_grasp_state),
            bool(diagnostics.success),
            float(diagnostics.workspace_and_joint_margin),
            bool(diagnostics.unexpected_collision),
            None,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        # A schedule is only a collection design artifact.  Falling back to
        # task reward keeps the planner usable for a new task while recording
        # that its privileged progress signal was unavailable.
        value = info.get("in_place_reward", 0.0)
        try:
            progress = float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            progress = 0.0
        return progress, False, _success(info), 0.0, True, type(exc).__name__


def replay_expert_trajectory(
    task: str,
    task_instance: int,
    environment_seed: int,
    *,
    benchmark_seed: int,
    width: int = 32,
    height: int = 32,
) -> dict[str, Any]:
    """Replay one expert and return raw-step phase metadata."""

    env = make_a4_env(
        task,
        a4_camera_name(task),
        int(width),
        int(height),
        task_instance=int(task_instance),
        benchmark_seed=int(benchmark_seed),
    )
    expert = make_a4_expert(task)
    rows: list[dict[str, Any]] = []
    first_success: int | None = None
    terminated_at: int | None = None
    try:
        observation, reset_info = env.reset(seed=int(environment_seed))
        info: Mapping[str, Any] = reset_info if isinstance(reset_info, Mapping) else {}
        max_steps = int(getattr(env, "max_path_length", 500))
        for raw_step in range(max_steps):
            observation = np.asarray(observation, dtype=np.float32)
            success_before = _success(info)
            if success_before and first_success is None:
                first_success = raw_step
            (
                progress,
                contact,
                signal_success,
                workspace_control_margin,
                unexpected_collision,
                signal_error,
            ) = _signals(
                task, env, observation, info
            )
            success = bool(success_before or signal_success)
            if success and first_success is None:
                first_success = raw_step
            action = np.asarray(expert.get_action(observation), dtype=np.float32)
            if action.ndim != 1 or action.size < 3 or not np.isfinite(action).all():
                raise ValueError(f"expert action for {task} is invalid at raw step {raw_step}")
            phase = _phase_name(
                task,
                expert,
                observation,
                progress=progress,
                contact=contact,
                success=success,
            )
            if task == "towel-wipe-v0" and phase in {
                "wipe_left",
                "wipe_right",
            }:
                half_cycle = int(getattr(env, "wipe_half_cycles", -1))
                if not 0 <= half_cycle < 6:
                    raise ValueError(
                        "towel wipe phase requires a half-cycle index in 0..5"
                    )
                phase = f"{phase}_cycle_{half_cycle}"
            rows.append(
                {
                    "raw_step": raw_step,
                    "success": success,
                    "progress": progress,
                    "contact": contact,
                    "phase": phase,
                    "action_xyz_norm": float(np.linalg.norm(action[:3])),
                    "workspace_control_margin": workspace_control_margin,
                    "unexpected_collision": unexpected_collision,
                    "signal_error": signal_error,
                }
            )
            low = np.asarray(env.action_space.low, dtype=np.float32)
            high = np.asarray(env.action_space.high, dtype=np.float32)
            observation, _reward, terminated, truncated, next_info = env.step(
                np.clip(action, low, high)
            )
            info = next_info if isinstance(next_info, Mapping) else {}
            if terminated or truncated:
                terminated_at = raw_step + 1
                break
        if first_success is None:
            for row in rows:
                if row["success"]:
                    first_success = int(row["raw_step"])
                    break
    finally:
        env.close()
    return {
        "task": task,
        "task_instance": int(task_instance),
        "environment_seed": int(environment_seed),
        "rows": rows,
        "first_success_step": first_success,
        "trajectory_action_count": len(rows),
        "terminated_at": terminated_at,
    }


def _candidate_rows(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = list(trajectory["rows"])
    first_success = trajectory.get("first_success_step")
    action_count = int(trajectory["trajectory_action_count"])
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        step = int(row["raw_step"])
        if step < MINIMUM_PROPOSAL_RAW_STEP or step + RAW_HORIZON > action_count:
            continue
        if bool(row["success"]):
            continue
        if first_success is not None and step >= int(first_success):
            continue
        if float(row["action_xyz_norm"]) < ACTIVE_ACTION_NORM:
            continue
        if float(row["workspace_control_margin"]) < MINIMUM_WORKSPACE_CONTROL_MARGIN:
            continue
        if bool(row["unexpected_collision"]):
            continue
        recent = rows[max(0, index - 4) : index + 1]
        if len(recent) < 5:
            continue
        if recent_xyz_action_rms(
            [float(item["action_xyz_norm"]) for item in recent]
        ) < ACTIVE_ACTION_NORM:
            continue
        candidates.append(row)
    return candidates


def _nearest_available(
    candidates: Sequence[dict[str, Any]],
    target: float,
    selected: Sequence[int],
) -> dict[str, Any] | None:
    ranked = sorted(
        candidates,
        key=lambda row: (abs(int(row["raw_step"]) - target), int(row["raw_step"])),
    )
    for row in ranked:
        step = int(row["raw_step"])
        if step not in selected and all(
            abs(step - other) >= MINIMUM_SEPARATION_RAW_STEPS for other in selected
        ):
            return row
    return None


def choose_phase_schedule(
    trajectory: Mapping[str, Any],
    *,
    count: int = PROPOSALS_PER_EPISODE,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Choose a deterministic, phase-stratified schedule for one episode."""

    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    candidates = _candidate_rows(trajectory)
    if not candidates:
        task = str(trajectory.get("task", ""))
        required = list(REQUIRED_PHASE_IDS.get(task, ()))
        return (), {
            "candidate_count": 0,
            "selected_count": 0,
            "selected_phase_names": [],
            "selected_unique_phase_count": 0,
            "required_phase_ids": required,
            "covered_required_phase_ids": [],
            "missing_required_phase_ids": required,
            "source_success": trajectory.get("first_success_step") is not None,
            "h8_crosses_success": False,
            "h8_crosses_phase_transition": False,
            "status": "no_active_candidate",
        }

    candidate_steps = [int(row["raw_step"]) for row in candidates]
    selected: list[int] = []
    task = str(trajectory.get("task", ""))

    # Reserve a success-boundary window before phase transitions can consume
    # the fixed schedule budget. These targets are computed from the frozen
    # expert replay, before any matched branch is executed.
    first_success = trajectory.get("first_success_step")
    if first_success is not None:
        success_candidates = [
            row
            for row in candidates
            if int(row["raw_step"]) < int(first_success)
            <= int(row["raw_step"]) + RAW_HORIZON
        ]
        row = _nearest_available(
            success_candidates,
            float(int(first_success) - RAW_HORIZON // 4),
            selected,
        )
        if row is not None:
            selected.append(int(row["raw_step"]))

    # Long-horizon custom tasks have semantic phases that must not be replaced
    # by generic temporal spread. In particular, each towel half-cycle is a
    # different control state even though direction labels repeat.
    required_phase_ids = REQUIRED_PHASE_IDS.get(task, ())
    for phase_id in required_phase_ids:
        phase_candidates = [
            row for row in candidates if str(row["phase"]) == phase_id
        ]
        if phase_candidates:
            midpoint = np.median(
                [int(row["raw_step"]) for row in phase_candidates]
            )
            row = _nearest_available(phase_candidates, float(midpoint), selected)
            if row is not None:
                selected.append(int(row["raw_step"]))
        if len(selected) >= count:
            break

    # Add one target for each new phase. Contact-only transitions are retained
    # only when they do not duplicate a phase target; repeated toggles must not
    # crowd out late task stages.
    transition_targets: list[int] = []
    rows = list(trajectory["rows"])
    seen_phase_targets: set[str] = set()
    seen_contact_targets: set[tuple[str, bool]] = set()
    for previous, current in zip(rows, rows[1:]):
        phase = str(current["phase"])
        contact_key = (phase, bool(current["contact"]))
        if previous["phase"] != current["phase"] and phase not in seen_phase_targets:
            transition_targets.append(int(current["raw_step"]))
            seen_phase_targets.add(phase)
            seen_contact_targets.add(contact_key)
        elif (
            previous["contact"] != current["contact"]
            and contact_key not in seen_contact_targets
        ):
            transition_targets.append(int(current["raw_step"]))
            seen_contact_targets.add(contact_key)
    for target in transition_targets:
        row = _nearest_available(candidates, float(target), selected)
        if row is not None:
            selected.append(int(row["raw_step"]))
        if len(selected) >= count:
            break

    lo, hi = min(candidate_steps), max(candidate_steps)
    if len(selected) < count:
        for target in np.linspace(float(lo), float(hi), num=count):
            row = _nearest_available(candidates, float(target), selected)
            if row is not None:
                selected.append(int(row["raw_step"]))
            if len(selected) >= count:
                break

    # Fill any remaining slots by maximizing temporal spread.  This makes the
    # schedule robust when a task has long phases with no discrete transition.
    while len(selected) < min(count, len(candidates)):
        remaining = [row for row in candidates if int(row["raw_step"]) not in selected]
        if not remaining:
            break
        row = max(
            remaining,
            key=lambda item: min(
                abs(int(item["raw_step"]) - other) for other in selected
            )
            if selected
            else abs(int(item["raw_step"]) - lo),
        )
        step = int(row["raw_step"])
        if all(abs(step - other) >= MINIMUM_SEPARATION_RAW_STEPS for other in selected):
            selected.append(step)
        else:
            break

    schedule = tuple(sorted(selected))
    row_by_step = {int(row["raw_step"]): row for row in rows}
    phases = [str(row_by_step[step]["phase"]) for step in schedule]
    covered_required = [
        phase for phase in required_phase_ids if phase in set(phases)
    ]
    missing_required = [
        phase for phase in required_phase_ids if phase not in set(phases)
    ]
    crosses_success = bool(
        first_success is not None
        and any(step < int(first_success) <= step + RAW_HORIZON for step in schedule)
    )
    crosses_transition = any(
        len({str(row_by_step.get(step + offset, {}).get("phase", "")) for offset in range(RAW_HORIZON + 1)}) > 1
        for step in schedule
    )
    source_success = first_success is not None
    if not source_success:
        status = "source_failed"
    elif len(schedule) >= min(count, 8) and not missing_required:
        status = "ok"
    else:
        status = "sparse"
    return schedule, {
        "candidate_count": len(candidates),
        "selected_count": len(schedule),
        "selected_phase_names": phases,
        "selected_unique_phase_count": len(set(phases)),
        "required_phase_ids": list(required_phase_ids),
        "covered_required_phase_ids": covered_required,
        "missing_required_phase_ids": missing_required,
        "source_success": source_success,
        "h8_crosses_success": crosses_success,
        "h8_crosses_phase_transition": crosses_transition,
        "status": status,
    }


def build_phase_schedule_manifest(
    contract: Mapping[str, Any],
    *,
    width: int = 32,
    height: int = 32,
    count: int = PROPOSALS_PER_EPISODE,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Replay all train/development episodes and build the A4c manifest."""

    tasks = validate_a4c_contract(contract)
    if max_tasks is not None:
        if not 1 <= int(max_tasks) <= len(tasks):
            raise ValueError("max_tasks must be in the range 1..7")
        tasks = tasks[: int(max_tasks)]
    collection = contract["collection"]
    benchmark_seed = int(collection["benchmark_seed"])
    environment_seed_base = int(collection["environment_seed_base"])
    assignments = assign_a4_splits(
        tuple(contract["tasks"]), a4c_base_contract(contract)
    )
    episodes: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        for instance in FORMAL_TASK_INSTANCES:
            split = assignments[(task, int(instance))]
            if split not in {"train", "development"}:
                continue
            environment_seed = environment_seed_base + task_index * len(FORMAL_TASK_INSTANCES) + int(instance)
            trajectory = replay_expert_trajectory(
                task,
                int(instance),
                environment_seed,
                benchmark_seed=benchmark_seed,
                width=width,
                height=height,
            )
            schedule, coverage = choose_phase_schedule(trajectory, count=count)
            episodes.append(
                {
                    "task": task,
                    "task_class_index": task_index,
                    "task_instance": int(instance),
                    "split": split,
                    "environment_seed": environment_seed,
                    "proposal_steps": list(schedule),
                    "first_success_step": trajectory["first_success_step"],
                    "trajectory_action_count": trajectory["trajectory_action_count"],
                    "coverage": coverage,
                    "signal_fallback_count": int(
                        sum(row["signal_error"] is not None for row in trajectory["rows"])
                    ),
                }
            )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "planned",
        "tasks": list(tasks),
        "schedule_contract": {
            "schedule_materialized_before_branch_execution": True,
            "proposal_count_per_episode": int(count),
            "minimum_raw_step": MINIMUM_PROPOSAL_RAW_STEP,
            "minimum_separation_raw_steps": MINIMUM_SEPARATION_RAW_STEPS,
            "branch_horizon_raw_steps": RAW_HORIZON,
            "uses_branch_outcomes": False,
            "uses_expert_replay_for_collection_design": True,
            "required_phase_ids": {
                task: list(phases)
                for task, phases in REQUIRED_PHASE_IDS.items()
            },
            "require_success_crossing_for_successful_source": True,
        },
        "collection": {
            "benchmark_seed": benchmark_seed,
            "environment_seed_base": environment_seed_base,
            "width": int(width),
            "height": int(height),
            "horizon_steps": 8,
            "raw_steps_per_model_step": RAW_STEPS_PER_MODEL_STEP,
        },
        "episodes": episodes,
    }


def save_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_schedule_map(manifest: Mapping[str, Any]) -> dict[tuple[str, int], tuple[int, ...]]:
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("schedule manifest protocol mismatch")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("schedule manifest must contain episodes")
    result: dict[tuple[str, int], tuple[int, ...]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("schedule episode must be a mapping")
        key = (str(row["task"]), int(row["task_instance"]))
        steps = tuple(int(value) for value in row["proposal_steps"])
        if tuple(sorted(set(steps))) != steps:
            raise ValueError(f"schedule is not sorted/unique for {key}")
        if key in result:
            raise ValueError(f"duplicate schedule for {key}")
        result[key] = steps
    return result


def validate_schedule_manifest(
    manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    """Require one valid schedule for every A4c train/development episode."""

    tasks = validate_a4c_contract(contract)
    if manifest.get("schema_version") != 1:
        raise ValueError("schedule manifest schema_version mismatch")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("schedule manifest protocol mismatch")
    if manifest.get("status") != "planned":
        raise ValueError("schedule manifest status mismatch")
    if manifest.get("tasks") != list(tasks):
        raise ValueError("schedule manifest task order mismatch")
    expected_schedule_contract = {
        "schedule_materialized_before_branch_execution": True,
        "proposal_count_per_episode": PROPOSALS_PER_EPISODE,
        "minimum_raw_step": MINIMUM_PROPOSAL_RAW_STEP,
        "minimum_separation_raw_steps": MINIMUM_SEPARATION_RAW_STEPS,
        "branch_horizon_raw_steps": RAW_HORIZON,
        "uses_branch_outcomes": False,
        "uses_expert_replay_for_collection_design": True,
        "required_phase_ids": {
            task: list(phases) for task, phases in REQUIRED_PHASE_IDS.items()
        },
        "require_success_crossing_for_successful_source": True,
    }
    if manifest.get("schedule_contract") != expected_schedule_contract:
        raise ValueError("schedule manifest schedule_contract mismatch")
    collection = manifest.get("collection")
    if not isinstance(collection, Mapping):
        raise TypeError("schedule manifest collection must be a mapping")
    expected_collection = contract["collection"]
    for key in (
        "benchmark_seed",
        "environment_seed_base",
        "horizon_steps",
        "raw_steps_per_model_step",
    ):
        if collection.get(key) != expected_collection.get(key):
            raise ValueError(f"schedule manifest collection {key} mismatch")
    for key in ("width", "height"):
        if not isinstance(collection.get(key), int) or int(collection[key]) <= 0:
            raise ValueError(f"schedule manifest planning {key} must be positive")

    schedule_map = load_schedule_map(manifest)
    assignments = assign_a4_splits(tasks, a4c_base_contract(contract))
    expected_keys = {
        (task, instance)
        for task in tasks
        for instance in FORMAL_TASK_INSTANCES
        if assignments[(task, instance)] in {"train", "development"}
    }
    if set(schedule_map) != expected_keys:
        raise ValueError("schedule manifest episode set mismatch")

    rows = manifest.get("episodes")
    assert isinstance(rows, list)
    row_map = {(str(row["task"]), int(row["task_instance"])): row for row in rows}
    seed_base = int(contract["collection"]["environment_seed_base"])
    for task_index, task in enumerate(tasks):
        for instance in FORMAL_TASK_INSTANCES:
            key = (task, instance)
            if key not in expected_keys:
                continue
            row = row_map[key]
            steps = schedule_map[key]
            expected_split = assignments[key]
            expected_seed = seed_base + task_index * len(FORMAL_TASK_INSTANCES) + instance
            if (
                row.get("split") != expected_split
                or int(row.get("environment_seed", -1)) != expected_seed
                or int(row.get("task_class_index", -1)) != task_index
            ):
                raise ValueError(f"schedule identity mismatch for {key}")
            action_count = int(row.get("trajectory_action_count", -1))
            if not 1 <= len(steps) <= PROPOSALS_PER_EPISODE:
                raise ValueError(f"schedule count mismatch for {key}")
            if any(step < MINIMUM_PROPOSAL_RAW_STEP or step + RAW_HORIZON > action_count for step in steps):
                raise ValueError(f"schedule horizon mismatch for {key}")
            if any(right - left < MINIMUM_SEPARATION_RAW_STEPS for left, right in zip(steps, steps[1:])):
                raise ValueError(f"schedule separation mismatch for {key}")
            first_success = row.get("first_success_step")
            if first_success is not None and (
                isinstance(first_success, bool) or not isinstance(first_success, int)
            ):
                raise TypeError(f"schedule first_success_step is invalid for {key}")
            if first_success is not None and not 0 <= first_success < action_count:
                raise ValueError(f"schedule first_success_step is out of range for {key}")
            coverage = row.get("coverage")
            if not isinstance(coverage, Mapping):
                raise TypeError(f"schedule coverage must be a mapping for {key}")
            phases = coverage.get("selected_phase_names")
            if (
                not isinstance(phases, list)
                or len(phases) != len(steps)
                or any(not isinstance(value, str) or not value for value in phases)
            ):
                raise ValueError(f"schedule phase labels mismatch for {key}")
            expected_crosses_success = bool(
                first_success is not None
                and any(step < first_success <= step + RAW_HORIZON for step in steps)
            )
            required_phases = list(REQUIRED_PHASE_IDS.get(task, ()))
            covered_required = [
                phase for phase in required_phases if phase in set(phases)
            ]
            missing_required = [
                phase for phase in required_phases if phase not in set(phases)
            ]
            source_success = first_success is not None
            if not source_success:
                expected_status = "source_failed"
            elif len(steps) >= min(PROPOSALS_PER_EPISODE, 8) and not missing_required:
                expected_status = "ok"
            else:
                expected_status = "sparse"
            semantic_values = {
                "selected_count": len(steps),
                "selected_unique_phase_count": len(set(phases)),
                "required_phase_ids": required_phases,
                "covered_required_phase_ids": covered_required,
                "missing_required_phase_ids": missing_required,
                "source_success": source_success,
                "h8_crosses_success": expected_crosses_success,
                "status": expected_status,
            }
            for name, expected in semantic_values.items():
                if coverage.get(name) != expected:
                    raise ValueError(f"schedule coverage {name} mismatch for {key}")
            if source_success and not expected_crosses_success:
                raise ValueError(f"schedule lacks a success-boundary window for {key}")
            if missing_required:
                raise ValueError(
                    f"schedule lacks required phases {missing_required} for {key}"
                )
            candidate_count = coverage.get("candidate_count")
            if (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count < len(steps)
            ):
                raise ValueError(f"schedule candidate_count mismatch for {key}")
            if not isinstance(coverage.get("h8_crosses_phase_transition"), bool):
                raise TypeError(
                    f"schedule h8_crosses_phase_transition is invalid for {key}"
                )
            fallback_count = row.get("signal_fallback_count")
            if (
                isinstance(fallback_count, bool)
                or not isinstance(fallback_count, int)
                or not 0 <= fallback_count <= action_count
            ):
                raise ValueError(f"schedule signal_fallback_count mismatch for {key}")


def make_schedule_sampler(
    manifest: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> Callable[..., tuple[int, ...]]:
    if contract is not None:
        validate_schedule_manifest(manifest, contract)
    schedule_map = load_schedule_map(manifest)

    def sampler(task_name: str, task_instance: int, *, contract: Mapping[str, Any]) -> tuple[int, ...]:
        del contract
        try:
            return schedule_map[(str(task_name), int(task_instance))]
        except KeyError as exc:
            raise KeyError(
                f"phase schedule has no episode {(task_name, int(task_instance))}"
            ) from exc

    return sampler


__all__ = [
    "PROTOCOL",
    "PROPOSALS_PER_EPISODE",
    "RAW_HORIZON",
    "REQUIRED_PHASE_IDS",
    "a4c_base_contract",
    "build_phase_schedule_manifest",
    "choose_phase_schedule",
    "load_a4c_contract",
    "load_schedule_map",
    "make_schedule_sampler",
    "replay_expert_trajectory",
    "save_manifest",
    "validate_a4c_contract",
    "validate_schedule_manifest",
]
