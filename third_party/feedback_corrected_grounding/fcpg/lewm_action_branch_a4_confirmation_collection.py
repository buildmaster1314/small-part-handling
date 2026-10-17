"""Authorized reserved-test collection for Action-Branch A4.

The train/development collector deliberately cannot read reserved instances.
This module opens the fixed test panel only after an exact Development-Go and
formal-train-cache binding have been verified.  Its output is test-only; 10k
training therefore never receives reserved-test rows.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from fcpg import lewm_action_branch_a4_collection as _source
from fcpg.lewm_a4_storage import STATE_CACHE_KEYS, load_a4_state_cache, validate_a4_state_cache
from fcpg.lewm_action_branch_a4 import (
    DEVELOPMENT_TRAINING_SEEDS,
    FORMAL_TASK_INSTANCES,
    PROTOCOL,
    TASKS,
    assign_a4_splits,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_collection import (
    A4EpisodeRuntime,
    BundleRow,
    FrozenA4StateEncoder,
    ProposalRow,
)


DEVELOPMENT_STAGE = "a4_2k_formal_development"
DEVELOPMENT_DECISION = "Development-Go"
DEVELOPMENT_CHECKS = (
    "minimum_effect_vs_no_action",
    "minimum_effect_vs_every_wrong_action_shift",
    "paired_ci_lower_exceeds_zero",
    "minimum_positive_tasks",
    "minimum_positive_seeds",
    "raw_factual_not_worse_than_no_action",
    "raw_factual_better_than_hold_last",
    "raw_factual_better_than_constant_velocity",
)
PRIMARY_HORIZONS = (6, 7, 8)
EXPECTED_TEST_EPISODES_PER_TASK = 12


@dataclass(frozen=True)
class A4ConfirmationAuthorization:
    """Validated authorization and the fixed reserved instance panel."""

    formal_train_cache_path: Path
    formal_train_cache_sha256: str
    test_instances_by_task: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class A4ReservedTestCollection:
    """Test-only cache and attempted/accepted physical episode identities."""

    cache: Mapping[str, np.ndarray]
    attempted_instances_by_task: Mapping[str, tuple[int, ...]]
    accepted_instances_by_task: Mapping[str, tuple[int, ...]]


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_instance_panel(contract: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    tasks = validate_a4_contract(contract)
    assignments = assign_a4_splits(tasks, contract)
    expected_count = int(contract["split"]["test_episodes_per_task"])
    if expected_count != EXPECTED_TEST_EPISODES_PER_TASK:
        raise ValueError("A4 confirmation requires exactly 12 test episodes per task")
    panel = {
        task: tuple(
            sorted(
                instance
                for instance in FORMAL_TASK_INSTANCES
                if assignments[(task, instance)] == "test"
            )
        )
        for task in tasks
    }
    if tuple(panel) != TASKS or any(
        len(values) != EXPECTED_TEST_EPISODES_PER_TASK for values in panel.values()
    ):
        raise ValueError("A4 reserved-test assignment must be exactly seven by twelve")
    return panel


def _validate_formal_cache_assignments(
    cache: Mapping[str, np.ndarray], contract: Mapping[str, Any]
) -> None:
    validate_a4_state_cache(cache)
    assignments = assign_a4_splits(TASKS, contract)
    tasks = np.asarray(cache["task_name"]).astype(str)
    instances = np.asarray(cache["task_instance_index"], dtype=np.int64)
    observed = np.asarray(cache["split"]).astype(str)
    expected = np.asarray(
        [
            assignments[(task, int(instance))]
            for task, instance in zip(tasks, instances, strict=True)
        ]
    )
    if not np.array_equal(observed, expected):
        raise PermissionError(
            "formal train cache rows do not match the frozen A4 split assignment"
        )
    if set(observed.tolist()) != {"train", "development"}:
        raise PermissionError(
            "formal train cache must contain train/development rows and no reserved test"
        )


def validate_a4_development_go_summary(
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    formal_train_cache_path: str | os.PathLike[str],
) -> A4ConfirmationAuthorization:
    """Fail closed unless Development-Go binds the supplied formal cache."""

    validate_a4_contract(contract)
    root = _mapping(summary, label="A4 Development-Go summary")
    expected_identity = (
        root.get("protocol"),
        root.get("stage"),
        root.get("mode"),
        root.get("decision"),
        root.get("passed"),
        root.get("development_go"),
        root.get("reserved_test_collection_authorized"),
        root.get("input_contains_test_rows"),
        root.get("checkpoint_step"),
        tuple(root.get("primary_horizons", ())),
    )
    required_identity = (
        PROTOCOL,
        DEVELOPMENT_STAGE,
        "formal",
        DEVELOPMENT_DECISION,
        True,
        True,
        True,
        False,
        2_000,
        PRIMARY_HORIZONS,
    )
    if expected_identity != required_identity:
        raise PermissionError(
            "reserved-test collection requires an exact formal A4 Development-Go summary"
        )

    coverage = _mapping(root.get("coverage"), label="A4 development coverage")
    if tuple(coverage.get("tasks", ())) != TASKS or tuple(
        coverage.get("training_seeds", ())
    ) != DEVELOPMENT_TRAINING_SEEDS:
        raise PermissionError("A4 Development-Go task or seed panel is incomplete")

    development_gate = _mapping(
        contract.get("development_gate"), label="A4 development contract"
    )
    minimum_effect = float(development_gate["minimum_relative_improvement"])
    gate = _mapping(root.get("gate"), label="A4 nested development gate")
    checks = _mapping(gate.get("checks"), label="A4 development checks")
    if (
        set(checks) != set(DEVELOPMENT_CHECKS)
        or any(value is not True for value in checks.values())
        or gate.get("failed_checks") != []
        or gate.get("passed") is not True
        or gate.get("development_go") is not True
        or _finite(gate.get("minimum_relative_improvement"), label="A4 gate minimum effect")
        != minimum_effect
    ):
        raise PermissionError("A4 Development-Go nested checks are incomplete")

    positive_tasks = tuple(gate.get("jointly_positive_tasks", ()))
    positive_seeds = tuple(gate.get("jointly_positive_seeds", ()))
    if (
        len(positive_tasks) < int(development_gate["minimum_positive_tasks"])
        or not set(positive_tasks).issubset(TASKS)
        or len(positive_seeds) < int(development_gate["minimum_positive_seeds"])
        or not set(positive_seeds).issubset(DEVELOPMENT_TRAINING_SEEDS)
    ):
        raise PermissionError("A4 Development-Go replication evidence is incomplete")

    supplied_path = Path(formal_train_cache_path).expanduser().resolve()
    binding = _mapping(
        root.get("formal_train_cache"), label="A4 formal train cache binding"
    )
    bound_value = binding.get("path")
    if not isinstance(bound_value, str) or not Path(bound_value).expanduser().is_absolute():
        raise PermissionError("Development-Go must bind an absolute formal train cache path")
    bound_path = Path(bound_value).expanduser().resolve()
    if supplied_path != bound_path:
        raise PermissionError("supplied formal train cache differs from Development-Go")
    if not supplied_path.is_file():
        raise FileNotFoundError(supplied_path)
    expected_sha256 = binding.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise PermissionError("Development-Go formal train cache SHA-256 is invalid")
    observed_sha256 = _sha256_file(supplied_path)
    if observed_sha256 != expected_sha256:
        raise PermissionError("formal train cache content differs from Development-Go")
    formal_cache = load_a4_state_cache(supplied_path)
    _validate_formal_cache_assignments(formal_cache, contract)

    return A4ConfirmationAuthorization(
        formal_train_cache_path=supplied_path,
        formal_train_cache_sha256=observed_sha256,
        test_instances_by_task=_test_instance_panel(contract),
    )


def _collect_reserved_source_episode(
    *,
    task: str,
    task_class_index: int,
    task_instance_index: int,
    environment_seed: int,
    split: str,
    contract: Mapping[str, Any],
    runtime_factory: Callable[[str, int, int], A4EpisodeRuntime],
    encode_records: Callable[[Sequence[Any]], np.ndarray],
    max_proposals: int | None = None,
    schedule_sampler: Callable[..., tuple[int, ...]] = _source.sample_a4_proposal_schedule,
    snapshot_capture: Callable[..., Any] = _source.capture_branch_snapshot,
    snapshot_restore: Callable[[Any, Any], None] = _source.restore_branch_snapshot,
    branch_bank_builder: Callable[..., Any] = _source.build_diverse_action_branch_bank,
    signal_extractor: Callable[..., Any] = _source.extract_a4_task_signals,
) -> tuple[list[ProposalRow], list[BundleRow]]:
    """Collect one reserved episode with ``test`` set at row creation time."""

    configured_tasks = validate_a4_contract(contract)
    if task not in configured_tasks or configured_tasks[task_class_index] != task:
        raise ValueError("A4 task identity differs from the frozen seven-task panel")
    if split != "test":
        raise ValueError("reserved A4 collection requires split='test'")

    formal_schedule = tuple(
        int(value)
        for value in schedule_sampler(task, int(task_instance_index), contract=contract)
    )
    if tuple(sorted(set(formal_schedule))) != formal_schedule or not formal_schedule:
        raise ValueError("A4 proposal schedule must be non-empty, sorted, and unique")
    schedule = formal_schedule
    if max_proposals is not None:
        if isinstance(max_proposals, bool) or not 1 <= int(max_proposals) <= len(schedule):
            raise ValueError("max_proposals must be in the schedule range")
        complete = tuple(
            step
            for step in schedule
            if step >= 2 * _source.RAW_STEPS_PER_MODEL_STEP
        )
        schedule = complete[: int(max_proposals)] or formal_schedule[: int(max_proposals)]

    runtime = runtime_factory(task, int(task_instance_index), int(environment_seed))
    episode_id = f"{task}:task-instance-{int(task_instance_index):02d}"
    proposal_rows: list[ProposalRow] = []
    source_actions: list[np.ndarray] = []
    records: dict[int, Any] = {}
    accepted: dict[str, tuple[Any, Any, np.ndarray]] = {}
    needed_records = {
        step - offset
        for step in schedule
        for offset in (2 * _source.RAW_STEPS_PER_MODEL_STEP, _source.RAW_STEPS_PER_MODEL_STEP, 0)
        if step - offset >= 0
    }
    observation = np.asarray(runtime.observation, dtype=np.float32)
    info: Mapping[str, Any] = runtime.reset_info
    ever_success = False
    source_success = False
    last_raw_step = -1
    try:
        for raw_step in range(runtime.max_raw_steps):
            last_raw_step = raw_step
            if raw_step in needed_records:
                records[raw_step] = runtime.record_current(observation)

            expert_action = _source._bounded_expert_action(runtime, observation)
            current_success = _source._success(info, runtime.env)
            source_success |= current_success
            if raw_step in schedule:
                signals = signal_extractor(task, runtime.env, observation, info)
                signal_fields = signals.eligibility_fields()
                current_success = bool(signal_fields["success"])
                history_start = raw_step - 2 * _source.RAW_STEPS_PER_MODEL_STEP
                complete_history = (
                    history_start >= 0
                    and all(
                        index in records
                        for index in (
                            history_start,
                            raw_step - _source.RAW_STEPS_PER_MODEL_STEP,
                            raw_step,
                        )
                    )
                    and len(source_actions) >= raw_step
                )
                recent = np.asarray(
                    source_actions[-_source.RAW_STEPS_PER_MODEL_STEP :], dtype=np.float32
                )
                recent_rms = (
                    float(np.sqrt(np.mean(np.square(recent[:, :3]), dtype=np.float64)))
                    if recent.shape
                    == (_source.RAW_STEPS_PER_MODEL_STEP, _source.RAW_ACTION_DIM)
                    else 0.0
                )
                inputs = {
                    "proposal": True,
                    "complete_history": bool(complete_history),
                    "success": current_success,
                    "ever_success_before": bool(ever_success),
                    "active_stage": bool(signal_fields["active_stage"]),
                    "expert_xyz_norm": float(np.linalg.norm(expert_action[:3])),
                    "recent_action_rms": recent_rms,
                    "workspace_and_joint_margin": float(
                        signal_fields["workspace_and_joint_margin"]
                    ),
                    "unexpected_collision": bool(signal_fields["unexpected_collision"]),
                    "remaining_raw_steps": int(runtime.max_raw_steps - raw_step),
                }
                decision = _source.evaluate_active_control_eligibility(
                    inputs, contract=contract
                )
                proposal_id = f"{episode_id}:proposal-{raw_step:03d}"
                row = ProposalRow(
                    proposal_id=proposal_id,
                    task_class_index=int(task_class_index),
                    task_name=task,
                    task_instance_index=int(task_instance_index),
                    source_episode_id=episode_id,
                    environment_seed=int(environment_seed),
                    proposal_raw_step=int(raw_step),
                    split="test",
                    inputs=inputs,
                    eligible=decision.eligible,
                    rejection_reasons=decision.rejection_reasons,
                )
                proposal_rows.append(row)
                if decision.eligible:
                    accepted[proposal_id] = (
                        snapshot_capture(runtime.env, include_process_rng=True),
                        _source.read_physical_target(
                            runtime.env,
                            task,
                            observation,
                            info,
                            signal_extractor=signal_extractor,
                        ),
                        np.asarray(signals.deployment.proprio, dtype=np.float32).copy(),
                    )

            ever_success |= current_success
            next_observation, _reward, terminated, truncated, next_info = runtime.env.step(
                expert_action
            )
            if not isinstance(next_info, Mapping):
                raise TypeError("A4 source step info must be a mapping")
            source_actions.append(expert_action.copy())
            observation = np.asarray(next_observation, dtype=np.float32)
            info = next_info
            source_success |= _source._success(info, runtime.env)
            if terminated or truncated:
                break

        missing = sorted(set(schedule) - {row.proposal_raw_step for row in proposal_rows})
        if missing:
            raise RuntimeError(
                f"A4 source episode ended before scheduled proposals {missing} "
                f"(last raw step {last_raw_step})"
            )

        action_array = np.asarray(source_actions, dtype=np.float32)
        bundles: list[BundleRow] = []
        row_by_id = {row.proposal_id: row for row in proposal_rows}
        for proposal_id, (snapshot, current_target, current_proprio) in accepted.items():
            row = row_by_id[proposal_id]
            step = row.proposal_raw_step
            end = step + _source.BRANCH_HORIZON_RAW_STEPS
            if action_array.shape[0] < end:
                raise RuntimeError("accepted A4 proposal lacks a complete expert center")
            history_records = (
                records[step - 2 * _source.RAW_STEPS_PER_MODEL_STEP],
                records[step - _source.RAW_STEPS_PER_MODEL_STEP],
                records[step],
            )
            history_state = np.asarray(encode_records(history_records), dtype=np.float32)
            if history_state.shape != (
                _source.HISTORY_STEPS,
                _source.STATE_SLOT_COUNT,
                _source.LATENT_DIM,
            ):
                raise ValueError("A4 history encoder must emit [3,4,192]")
            history_action = np.stack(
                (
                    _source._model_actions(action_array[step - 10 : step - 5])[0],
                    _source._model_actions(action_array[step - 5 : step])[0],
                )
            ).astype(np.float32, copy=False)
            raw_center = np.ascontiguousarray(action_array[step:end], dtype=np.float32)
            direction_seed, permutation_seed = _source._branch_seeds(
                contract, task_class_index, task_instance_index, step
            )
            bank = branch_bank_builder(
                raw_center,
                direction_seed=direction_seed,
                permutation_seed=permutation_seed,
                action_low=runtime.env.action_space.low,
                action_high=runtime.env.action_space.high,
            )
            target_state, branch_targets, branch_reward, branch_success = (
                _source._execute_branches(
                    runtime,
                    snapshot,
                    bank.actions,
                    task=task,
                    encode_records=encode_records,
                    restore_snapshot=snapshot_restore,
                    signal_extractor=signal_extractor,
                )
            )
            semantic = np.random.default_rng(permutation_seed).permutation(
                _source.BRANCH_COUNT
            )
            bundles.append(
                BundleRow(
                    proposal=row,
                    history_state=history_state,
                    history_action=history_action,
                    center_action=_source._model_actions(bank.center_action),
                    future_action=np.stack(
                        [_source._model_actions(branch) for branch in bank.actions]
                    ).astype(np.float32, copy=False),
                    target_state=target_state,
                    branch_semantic_index=semantic.astype(np.int64, copy=False),
                    current_target=current_target,
                    current_proprio=current_proprio,
                    anchor_validity=_source._nominal_anchor_validity(history_records),
                    branch_targets=branch_targets,
                    branch_reward=branch_reward,
                    branch_success=branch_success,
                    source_expert_success=bool(source_success),
                )
            )
        return proposal_rows, bundles
    finally:
        runtime.env.close()


def _stack_test_cache(bundles: Sequence[BundleRow]) -> dict[str, np.ndarray]:
    rows = list(bundles)
    if not rows:
        raise RuntimeError("A4 reserved-test collection produced no eligible bundles")
    cache: dict[str, np.ndarray] = {
        "proposal_id": np.asarray([row.proposal.proposal_id for row in rows]),
        "task_class_index": np.asarray(
            [row.proposal.task_class_index for row in rows], dtype=np.int64
        ),
        "task_name": np.asarray([row.proposal.task_name for row in rows]),
        "task_instance_index": np.asarray(
            [row.proposal.task_instance_index for row in rows], dtype=np.int64
        ),
        "source_episode_id": np.asarray([row.proposal.source_episode_id for row in rows]),
        "split": np.full(len(rows), "test"),
        "history_state": np.stack([row.history_state for row in rows]).astype(
            np.float32, copy=False
        ),
        "history_action": np.stack([row.history_action for row in rows]).astype(
            np.float32, copy=False
        ),
        "center_action": np.stack([row.center_action for row in rows]).astype(
            np.float32, copy=False
        ),
        "future_action": np.stack([row.future_action for row in rows]).astype(
            np.float32, copy=False
        ),
        "target_state": np.stack([row.target_state for row in rows]).astype(
            np.float32, copy=False
        ),
        "branch_semantic_index": np.stack(
            [row.branch_semantic_index for row in rows]
        ).astype(np.int64, copy=False),
    }
    if tuple(cache) != STATE_CACHE_KEYS:
        raise RuntimeError("internal A4 reserved-test cache key order drifted")
    validate_a4_state_cache(cache)
    return cache


def _validate_reserved_rows(
    proposals: Sequence[ProposalRow],
    bundles: Sequence[BundleRow],
    *,
    task: str,
    task_index: int,
    instance: int,
    environment_seed: int,
    assignments: Mapping[tuple[str, int], str],
) -> None:
    if assignments[(task, instance)] != "test":
        raise PermissionError("attempted reserved instance is not assigned to test")
    for row in proposals:
        observed = (
            row.task_name,
            row.task_class_index,
            row.task_instance_index,
            row.split,
            row.environment_seed,
        )
        expected = (task, task_index, instance, "test", environment_seed)
        if observed != expected:
            raise PermissionError(
                "reserved proposal identity/split/environment seed differs from assignment"
            )
    proposal_ids = {row.proposal_id for row in proposals}
    if any(
        row.proposal.proposal_id not in proposal_ids or row.proposal.split != "test"
        for row in bundles
    ):
        raise PermissionError("reserved bundle is not backed by a direct test proposal")


def _validate_test_cache_assignments(
    cache: Mapping[str, np.ndarray], contract: Mapping[str, Any]
) -> dict[str, tuple[int, ...]]:
    validate_a4_state_cache(cache)
    split = np.asarray(cache["split"]).astype(str)
    if set(split.tolist()) != {"test"}:
        raise ValueError("reserved output must contain test rows only")
    assignments = assign_a4_splits(TASKS, contract)
    tasks = np.asarray(cache["task_name"]).astype(str)
    instances = np.asarray(cache["task_instance_index"], dtype=np.int64)
    if any(
        assignments[(task, int(instance))] != "test"
        for task, instance in zip(tasks, instances, strict=True)
    ):
        raise PermissionError("reserved output contains a non-test task instance")
    accepted = {
        task: tuple(sorted(set(instances[tasks == task].tolist()))) for task in TASKS
    }
    if any(len(values) != EXPECTED_TEST_EPISODES_PER_TASK for values in accepted.values()):
        raise RuntimeError(
            "reserved test requires exactly 12 accepted episodes per task"
        )
    return accepted


def collect_a4_reserved_test_cache(
    contract: Mapping[str, Any],
    authorization: A4ConfirmationAuthorization,
    *,
    encoder: FrozenA4StateEncoder,
    runtime_factory: Callable[[str, int, int], A4EpisodeRuntime],
    episode_collector: Callable[
        ..., tuple[list[ProposalRow], list[BundleRow]]
    ] = _collect_reserved_source_episode,
    cache_stacker: Callable[[Sequence[BundleRow]], Mapping[str, np.ndarray]] | None = None,
) -> A4ReservedTestCollection:
    """Collect the fixed 7x12 panel and return a test-only cache."""

    if not isinstance(authorization, A4ConfirmationAuthorization):
        raise TypeError("reserved collection requires validated A4 authorization")
    validate_a4_contract(contract)
    expected_panel = {
        task: tuple(values)
        for task, values in authorization.test_instances_by_task.items()
    }
    if expected_panel != _test_instance_panel(contract):
        raise PermissionError("authorization reserved panel differs from the A4 contract")
    assignments = assign_a4_splits(TASKS, contract)
    collection = _mapping(contract.get("collection"), label="A4 collection")
    seed_base = int(collection["environment_seed_base"])
    proposals: list[ProposalRow] = []
    bundles: list[BundleRow] = []
    for task_index, task in enumerate(TASKS):
        for instance in expected_panel[task]:
            environment_seed = (
                seed_base + task_index * len(FORMAL_TASK_INSTANCES) + instance
            )
            episode_proposals, episode_bundles = episode_collector(
                task=task,
                task_class_index=task_index,
                task_instance_index=instance,
                environment_seed=environment_seed,
                split="test",
                contract=contract,
                runtime_factory=runtime_factory,
                encode_records=encoder.encode,
                max_proposals=None,
            )
            _validate_reserved_rows(
                episode_proposals,
                episode_bundles,
                task=task,
                task_index=task_index,
                instance=instance,
                environment_seed=environment_seed,
                assignments=assignments,
            )
            proposals.extend(episode_proposals)
            bundles.extend(episode_bundles)

    test_cache = (
        _stack_test_cache(bundles)
        if cache_stacker is None
        else {name: np.asarray(value) for name, value in cache_stacker(bundles).items()}
    )
    accepted = _validate_test_cache_assignments(test_cache, contract)
    return A4ReservedTestCollection(
        cache=test_cache,
        attempted_instances_by_task=expected_panel,
        accepted_instances_by_task=accepted,
    )


def save_a4_reserved_test_cache(
    path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]
) -> Path:
    """Publish a validated test-only cache atomically without replacement."""

    validate_a4_state_cache(arrays)
    if set(np.asarray(arrays["split"]).astype(str).tolist()) != {"test"}:
        raise ValueError("reserved output cache must be test-only")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle, **{name: arrays[name] for name in STATE_CACHE_KEYS}
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(destination) from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "A4ConfirmationAuthorization",
    "A4ReservedTestCollection",
    "EXPECTED_TEST_EPISODES_PER_TASK",
    "MINIMUM_ACCEPTED_TEST_EPISODES_PER_TASK",
    "collect_a4_reserved_test_cache",
    "save_a4_reserved_test_cache",
    "validate_a4_development_go_summary",
]
