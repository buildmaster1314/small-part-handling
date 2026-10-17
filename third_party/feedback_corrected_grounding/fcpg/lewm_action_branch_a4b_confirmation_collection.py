"""Fresh, finite candidate-pool collection for A4b confirmation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fcpg import lewm_action_branch_a4_collection as _source
from fcpg.lewm_a4_storage import (
    PROPOSAL_LEDGER_KEYS,
    STATE_CACHE_KEYS,
    validate_a4_ledger_cache_pair,
    validate_a4_proposal_ledger,
    validate_a4_state_cache,
)
from fcpg.lewm_a4b_task_identity import (
    LEGACY_IDENTITY_SCOPES,
    ResetPhysicalIdentity,
    legacy_digest_set,
)
from fcpg.lewm_action_branch_a4 import (
    PROPOSALS_PER_EPISODE,
    evaluate_active_control_eligibility,
)
from fcpg.lewm_action_branch_a4_collection import (
    A4EpisodeRuntime,
    BundleRow,
    FrozenA4StateEncoder,
    ProposalRow,
)
from fcpg.lewm_action_branch_a4_confirmation_collection import (
    validate_a4_development_go_summary,
)
from fcpg.lewm_action_branch_a4b import (
    BENCHMARK_SEED,
    CONFIRMATION_TRAINING_SEEDS,
    CUSTOM_TASK_PARAMETER_SEED,
    FINAL_CANDIDATES_PER_TASK,
    FINAL_ENVIRONMENT_SEED_BASE,
    POOL_FINAL,
    POOL_PREFLIGHT,
    PREFLIGHT_CANDIDATES_PER_TASK,
    PREFLIGHT_ENVIRONMENT_SEED_BASE,
    PROPOSAL_SEED,
    PROTOCOL,
    RAW_TASK_INDICES,
    SOURCE_A4_PROTOCOL,
    SPLIT_MANIFEST_ID,
    SPLIT_NAMESPACE,
    SPLIT_SEED,
    TARGET_ELIGIBLE_CANDIDATES_PER_TASK,
    TASK_IDENTITY_NAMESPACE,
    TASKS,
    branch_seeds,
    candidate_environment_seed,
    candidate_episode_uid,
    candidate_order,
    sample_a4b_proposal_schedule,
    validate_a4b_contract,
)


RESULT_SCHEMA_VERSION = 1
PREFLIGHT_STAGE = "a4b_fresh_preflight_candidate_pool"
FINAL_STAGE = "a4b_fresh_final_candidate_pool"

PhysicalIdentityKey = tuple[str, str]


@dataclass(frozen=True)
class A4bCandidateEpisode:
    proposals: tuple[ProposalRow, ...]
    bundles: tuple[BundleRow, ...]
    reset_identity: ResetPhysicalIdentity


@dataclass(frozen=True)
class A4bCandidateAttempt:
    candidate_rank: int
    raw_task_index: int
    environment_seed: int
    episode_uid: str
    reset_physical_state_goal_digest: str
    proposal_count: int
    eligible_proposal_count: int
    rejection_reason_counts: Mapping[str, int]
    eligible: bool
    selected: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "candidate_rank": self.candidate_rank,
            "raw_task_index": self.raw_task_index,
            "environment_seed": self.environment_seed,
            "episode_uid": self.episode_uid,
            "reset_physical_state_goal_digest": (
                self.reset_physical_state_goal_digest
            ),
            "proposal_count": self.proposal_count,
            "eligible_proposal_count": self.eligible_proposal_count,
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "eligible": self.eligible,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class A4bCandidatePoolCollection:
    pool: str
    passed: bool
    failure_task: str | None
    attempts_by_task: Mapping[str, tuple[A4bCandidateAttempt, ...]]
    selected_episode_uids_by_task: Mapping[str, tuple[str, ...]]
    proposal_ledger: Mapping[str, np.ndarray] | None
    state_cache: Mapping[str, np.ndarray] | None


def _text(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(value) for value in values])


def _load_json_object(
    path: str | os.PathLike[str], *, label: str
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _identity_key(identity: ResetPhysicalIdentity) -> PhysicalIdentityKey:
    return identity.task_name, identity.digest


def _validate_reset_identity(
    identity: ResetPhysicalIdentity,
    *,
    task: str,
    raw_task_index: int,
    environment_seed: int,
) -> None:
    expected = (task, raw_task_index, BENCHMARK_SEED, environment_seed)
    observed = (
        identity.task_name,
        identity.raw_task_index,
        identity.benchmark_seed,
        identity.environment_seed,
    )
    if observed != expected:
        raise PermissionError("A4b reset identity differs from candidate identity")
    if not identity.camera_name:
        raise ValueError("A4b reset identity requires a camera name")
    if len(identity.digest) != 64 or any(
        character not in "0123456789abcdef" for character in identity.digest
    ):
        raise ValueError("A4b reset physical-state/goal digest is invalid")


def collect_a4b_candidate_episode(
    *,
    task: str,
    task_class_index: int,
    raw_task_index: int,
    candidate_rank: int,
    environment_seed: int,
    pool: str,
    contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    runtime_factory: Callable[[str, int, int], A4EpisodeRuntime],
    encode_records: Callable[[Sequence[Any]], np.ndarray] | None,
    materialize_branch_bundles: bool,
    schedule_sampler: Callable[..., tuple[int, ...]] = sample_a4b_proposal_schedule,
    snapshot_capture: Callable[..., Any] = _source.capture_branch_snapshot,
    snapshot_restore: Callable[[Any, Any], None] = _source.restore_branch_snapshot,
    branch_bank_builder: Callable[..., Any] = _source.build_diverse_action_branch_bank,
    signal_extractor: Callable[..., Any] = _source.extract_a4_task_signals,
) -> A4bCandidateEpisode:
    """Collect one candidate with v2 eligibility and A4b schedule/identity seeds."""

    tasks = validate_a4b_contract(
        contract, source_contract=source_contract, split_manifest=split_manifest
    )
    if task not in tasks or tasks[int(task_class_index)] != task:
        raise ValueError("A4b task identity differs from the frozen task panel")
    order = candidate_order(split_manifest, pool=pool, task=task)
    if not 0 <= int(candidate_rank) < len(order):
        raise ValueError("A4b candidate rank is outside its frozen pool")
    if int(raw_task_index) != order[int(candidate_rank)]:
        raise ValueError("A4b raw task index differs from the split manifest")
    expected_seed = candidate_environment_seed(
        pool=pool, task_index=int(task_class_index), rank=int(candidate_rank)
    )
    if int(environment_seed) != expected_seed:
        raise ValueError("A4b environment seed differs from the candidate identity")
    if pool == POOL_PREFLIGHT and materialize_branch_bundles:
        raise PermissionError("A4b preflight cannot materialize branch bundles")
    if pool == POOL_FINAL and not materialize_branch_bundles:
        raise PermissionError("A4b final collection requires complete branch bundles")
    if materialize_branch_bundles and encode_records is None:
        raise TypeError("A4b final collection requires a frozen state encoder")

    # The schedule is sampled before environment construction/reset.
    schedule = tuple(
        int(value)
        for value in schedule_sampler(
            task,
            int(raw_task_index),
            pool=pool,
            candidate_rank=int(candidate_rank),
            contract=contract,
            source_contract=source_contract,
            split_manifest=split_manifest,
        )
    )
    if tuple(sorted(set(schedule))) != schedule or not schedule:
        raise ValueError("A4b proposal schedule must be non-empty/sorted/unique")

    runtime = runtime_factory(task, int(raw_task_index), int(environment_seed))
    identity = getattr(runtime, "reset_identity", None)
    if not isinstance(identity, ResetPhysicalIdentity):
        runtime.env.close()
        raise TypeError("A4b runtime must expose ResetPhysicalIdentity")
    _validate_reset_identity(
        identity,
        task=task,
        raw_task_index=int(raw_task_index),
        environment_seed=int(environment_seed),
    )
    episode_uid = candidate_episode_uid(
        pool=pool,
        task=task,
        rank=int(candidate_rank),
        raw_task_index=int(raw_task_index),
        environment_seed=int(environment_seed),
    )
    proposals: list[ProposalRow] = []
    source_actions: list[np.ndarray] = []
    records: dict[int, Any] = {}
    accepted: dict[str, tuple[Any, Any, np.ndarray]] = {}
    needed_records = {
        step - offset
        for step in schedule
        for offset in (
            2 * _source.RAW_STEPS_PER_MODEL_STEP,
            _source.RAW_STEPS_PER_MODEL_STEP,
            0,
        )
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
            if materialize_branch_bundles and raw_step in needed_records:
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
                    and len(source_actions) >= raw_step
                    and (
                        not materialize_branch_bundles
                        or all(
                            index in records
                            for index in (
                                history_start,
                                raw_step - _source.RAW_STEPS_PER_MODEL_STEP,
                                raw_step,
                            )
                        )
                    )
                )
                recent = np.asarray(
                    source_actions[-_source.RAW_STEPS_PER_MODEL_STEP :],
                    dtype=np.float32,
                )
                recent_rms = (
                    float(
                        np.sqrt(
                            np.mean(np.square(recent[:, :3]), dtype=np.float64)
                        )
                    )
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
                    "unexpected_collision": bool(
                        signal_fields["unexpected_collision"]
                    ),
                    "remaining_raw_steps": int(runtime.max_raw_steps - raw_step),
                }
                decision = evaluate_active_control_eligibility(
                    inputs, contract=source_contract
                )
                proposal_id = f"{episode_uid}:proposal-{raw_step:03d}"
                row = ProposalRow(
                    proposal_id=proposal_id,
                    task_class_index=int(task_class_index),
                    task_name=task,
                    task_instance_index=int(raw_task_index),
                    source_episode_id=episode_uid,
                    environment_seed=int(environment_seed),
                    proposal_raw_step=int(raw_step),
                    split="test",
                    inputs=inputs,
                    eligible=decision.eligible,
                    rejection_reasons=decision.rejection_reasons,
                )
                proposals.append(row)
                if decision.eligible and materialize_branch_bundles:
                    accepted[proposal_id] = (
                        snapshot_capture(runtime.env, include_process_rng=True),
                        _source.read_physical_target(
                            runtime.env,
                            task,
                            observation,
                            info,
                            signal_extractor=signal_extractor,
                        ),
                        np.asarray(
                            signals.deployment.proprio, dtype=np.float32
                        ).copy(),
                    )
            ever_success |= current_success
            next_observation, _reward, terminated, truncated, next_info = (
                runtime.env.step(expert_action)
            )
            if not isinstance(next_info, Mapping):
                raise TypeError("A4b source step info must be a mapping")
            source_actions.append(expert_action.copy())
            observation = np.asarray(next_observation, dtype=np.float32)
            info = next_info
            source_success |= _source._success(info, runtime.env)
            if terminated or truncated:
                break

        missing = sorted(
            set(schedule) - {row.proposal_raw_step for row in proposals}
        )
        if missing:
            raise RuntimeError(
                f"A4b episode ended before proposals {missing} "
                f"(last raw step {last_raw_step})"
            )
        if not materialize_branch_bundles:
            return A4bCandidateEpisode(tuple(proposals), (), identity)

        action_array = np.asarray(source_actions, dtype=np.float32)
        bundles: list[BundleRow] = []
        row_by_id = {row.proposal_id: row for row in proposals}
        for proposal_id, (snapshot, current_target, current_proprio) in accepted.items():
            row = row_by_id[proposal_id]
            step = row.proposal_raw_step
            end = step + _source.BRANCH_HORIZON_RAW_STEPS
            if action_array.shape[0] < end:
                raise RuntimeError("accepted A4b proposal lacks a complete expert center")
            history_records = (
                records[step - 2 * _source.RAW_STEPS_PER_MODEL_STEP],
                records[step - _source.RAW_STEPS_PER_MODEL_STEP],
                records[step],
            )
            history_state = np.asarray(
                encode_records(history_records), dtype=np.float32
            )
            if history_state.shape != (
                _source.HISTORY_STEPS,
                _source.STATE_SLOT_COUNT,
                _source.LATENT_DIM,
            ):
                raise ValueError("A4b history encoder must emit [3,4,192]")
            history_action = np.stack(
                (
                    _source._model_actions(action_array[step - 10 : step - 5])[0],
                    _source._model_actions(action_array[step - 5 : step])[0],
                )
            ).astype(np.float32, copy=False)
            raw_center = np.ascontiguousarray(action_array[step:end], dtype=np.float32)
            direction_seed, permutation_seed = branch_seeds(
                pool=pool,
                task_index=int(task_class_index),
                raw_task_index=int(raw_task_index),
                candidate_rank=int(candidate_rank),
                proposal_step=step,
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
        return A4bCandidateEpisode(tuple(proposals), tuple(bundles), identity)
    finally:
        runtime.env.close()


def _validate_episode(
    episode: A4bCandidateEpisode,
    *,
    pool: str,
    task: str,
    task_index: int,
    rank: int,
    raw_task_index: int,
    environment_seed: int,
    materialize: bool,
) -> None:
    _validate_reset_identity(
        episode.reset_identity,
        task=task,
        raw_task_index=raw_task_index,
        environment_seed=environment_seed,
    )
    if len(episode.proposals) != PROPOSALS_PER_EPISODE:
        raise RuntimeError("A4b candidate must record all 12 proposals")
    uid = candidate_episode_uid(
        pool=pool,
        task=task,
        rank=rank,
        raw_task_index=raw_task_index,
        environment_seed=environment_seed,
    )
    expected = (task, task_index, raw_task_index, environment_seed, uid, "test")
    for row in episode.proposals:
        observed = (
            row.task_name,
            row.task_class_index,
            row.task_instance_index,
            row.environment_seed,
            row.source_episode_id,
            row.split,
        )
        if observed != expected:
            raise PermissionError("A4b proposal identity differs from episode_uid")
    proposal_ids = [row.proposal_id for row in episode.proposals]
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ValueError("A4b proposal IDs must be unique")
    eligible_ids = {row.proposal_id for row in episode.proposals if row.eligible}
    bundle_ids = {row.proposal.proposal_id for row in episode.bundles}
    if materialize and bundle_ids != eligible_ids:
        raise RuntimeError("A4b final candidate did not materialize all eligible rows")
    if not materialize and episode.bundles:
        raise RuntimeError("A4b preflight emitted forbidden branch bundles")


def _stack_final_artifacts(
    proposals: Sequence[ProposalRow], bundles: Sequence[BundleRow]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    proposal_rows = list(proposals)
    bundle_rows = list(bundles)
    if not proposal_rows or not bundle_rows:
        raise RuntimeError("complete A4b final collection requires bundles")
    accepted_index = {
        row.proposal.proposal_id: index for index, row in enumerate(bundle_rows)
    }
    ledger: dict[str, np.ndarray] = {
        "proposal_id": _text([row.proposal_id for row in proposal_rows]),
        "task_class_index": np.asarray(
            [row.task_class_index for row in proposal_rows], dtype=np.int64
        ),
        "task_name": _text([row.task_name for row in proposal_rows]),
        "task_instance_index": np.asarray(
            [row.task_instance_index for row in proposal_rows], dtype=np.int64
        ),
        "source_episode_id": _text([row.source_episode_id for row in proposal_rows]),
        "environment_seed": np.asarray(
            [row.environment_seed for row in proposal_rows], dtype=np.int64
        ),
        "proposal_raw_step": np.asarray(
            [row.proposal_raw_step for row in proposal_rows], dtype=np.int64
        ),
        "split": np.full(len(proposal_rows), "test"),
        "proposal": np.ones(len(proposal_rows), dtype=bool),
        "complete_history": np.asarray(
            [row.inputs["complete_history"] for row in proposal_rows], dtype=bool
        ),
        "success": np.asarray(
            [row.inputs["success"] for row in proposal_rows], dtype=bool
        ),
        "ever_success_before": np.asarray(
            [row.inputs["ever_success_before"] for row in proposal_rows], dtype=bool
        ),
        "active_stage": np.asarray(
            [row.inputs["active_stage"] for row in proposal_rows], dtype=bool
        ),
        "expert_xyz_norm": np.asarray(
            [row.inputs["expert_xyz_norm"] for row in proposal_rows], dtype=np.float32
        ),
        "recent_action_rms": np.asarray(
            [row.inputs["recent_action_rms"] for row in proposal_rows], dtype=np.float32
        ),
        "workspace_and_joint_margin": np.asarray(
            [row.inputs["workspace_and_joint_margin"] for row in proposal_rows],
            dtype=np.float32,
        ),
        "unexpected_collision": np.asarray(
            [row.inputs["unexpected_collision"] for row in proposal_rows], dtype=bool
        ),
        "remaining_raw_steps": np.asarray(
            [row.inputs["remaining_raw_steps"] for row in proposal_rows], dtype=np.int64
        ),
        "eligible": np.asarray([row.eligible for row in proposal_rows], dtype=bool),
        "rejection_reasons": _text(
            ["|".join(row.rejection_reasons) for row in proposal_rows]
        ),
        "accepted_bundle_index": np.asarray(
            [accepted_index.get(row.proposal_id, -1) for row in proposal_rows],
            dtype=np.int64,
        ),
    }
    cache: dict[str, np.ndarray] = {
        "proposal_id": _text([row.proposal.proposal_id for row in bundle_rows]),
        "task_class_index": np.asarray(
            [row.proposal.task_class_index for row in bundle_rows], dtype=np.int64
        ),
        "task_name": _text([row.proposal.task_name for row in bundle_rows]),
        "task_instance_index": np.asarray(
            [row.proposal.task_instance_index for row in bundle_rows], dtype=np.int64
        ),
        "source_episode_id": _text(
            [row.proposal.source_episode_id for row in bundle_rows]
        ),
        "split": np.full(len(bundle_rows), "test"),
        "history_state": np.stack([row.history_state for row in bundle_rows]).astype(
            np.float32, copy=False
        ),
        "history_action": np.stack([row.history_action for row in bundle_rows]).astype(
            np.float32, copy=False
        ),
        "center_action": np.stack([row.center_action for row in bundle_rows]).astype(
            np.float32, copy=False
        ),
        "future_action": np.stack([row.future_action for row in bundle_rows]).astype(
            np.float32, copy=False
        ),
        "target_state": np.stack([row.target_state for row in bundle_rows]).astype(
            np.float32, copy=False
        ),
        "branch_semantic_index": np.stack(
            [row.branch_semantic_index for row in bundle_rows]
        ).astype(np.int64, copy=False),
    }
    if set(ledger) != set(PROPOSAL_LEDGER_KEYS) or set(cache) != set(
        STATE_CACHE_KEYS
    ):
        raise RuntimeError("internal A4b artifact keys drifted")
    validate_a4_ledger_cache_pair(ledger, cache)
    return ledger, cache


def _run_pool(
    contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    *,
    pool: str,
    forbidden_physical_identities: Set[PhysicalIdentityKey],
    runtime_factory: Callable[[str, int, int], A4EpisodeRuntime],
    encoder: FrozenA4StateEncoder | None,
    episode_collector: Callable[..., A4bCandidateEpisode],
) -> A4bCandidatePoolCollection:
    validate_a4b_contract(
        contract, source_contract=source_contract, split_manifest=split_manifest
    )
    if pool == POOL_FINAL:
        if encoder is None:
            raise TypeError("A4b final requires a frozen state encoder")
    elif pool != POOL_PREFLIGHT:
        raise ValueError("unknown A4b pool")

    materialize = pool == POOL_FINAL
    seen = set(forbidden_physical_identities)
    attempts_by_task: dict[str, tuple[A4bCandidateAttempt, ...]] = {}
    selected_by_task: dict[str, tuple[str, ...]] = {}
    all_proposals: list[ProposalRow] = []
    all_bundles: list[BundleRow] = []
    failure_task: str | None = None
    for task_index, task in enumerate(TASKS):
        task_attempts: list[A4bCandidateAttempt] = []
        selected: list[str] = []
        if failure_task is None:
            for rank, raw_index in enumerate(
                candidate_order(split_manifest, pool=pool, task=task)
            ):
                environment_seed = candidate_environment_seed(
                    pool=pool, task_index=task_index, rank=rank
                )
                episode = episode_collector(
                    task=task,
                    task_class_index=task_index,
                    raw_task_index=raw_index,
                    candidate_rank=rank,
                    environment_seed=environment_seed,
                    pool=pool,
                    contract=contract,
                    source_contract=source_contract,
                    split_manifest=split_manifest,
                    runtime_factory=runtime_factory,
                    encode_records=None if encoder is None else encoder.encode,
                    materialize_branch_bundles=materialize,
                )
                _validate_episode(
                    episode,
                    pool=pool,
                    task=task,
                    task_index=task_index,
                    rank=rank,
                    raw_task_index=raw_index,
                    environment_seed=environment_seed,
                    materialize=materialize,
                )
                physical_key = _identity_key(episode.reset_identity)
                if physical_key in seen:
                    raise PermissionError(
                        "A4b reset physical-state/goal identity overlaps a legacy, "
                        "preflight, or earlier candidate identity"
                    )
                seen.add(physical_key)
                eligible_count = sum(row.eligible for row in episode.proposals)
                eligible = eligible_count > 0
                selected_candidate = materialize and eligible
                uid = candidate_episode_uid(
                    pool=pool,
                    task=task,
                    rank=rank,
                    raw_task_index=raw_index,
                    environment_seed=environment_seed,
                )
                if selected_candidate:
                    selected.append(uid)
                reasons = Counter(
                    reason
                    for row in episode.proposals
                    for reason in row.rejection_reasons
                )
                task_attempts.append(
                    A4bCandidateAttempt(
                        candidate_rank=rank,
                        raw_task_index=raw_index,
                        environment_seed=environment_seed,
                        episode_uid=uid,
                        reset_physical_state_goal_digest=episode.reset_identity.digest,
                        proposal_count=len(episode.proposals),
                        eligible_proposal_count=eligible_count,
                        rejection_reason_counts=dict(sorted(reasons.items())),
                        eligible=eligible,
                        selected=selected_candidate,
                    )
                )
                if materialize:
                    all_proposals.extend(episode.proposals)
                    all_bundles.extend(episode.bundles)
                    if len(selected) == TARGET_ELIGIBLE_CANDIDATES_PER_TASK:
                        break
            if materialize and len(selected) < TARGET_ELIGIBLE_CANDIDATES_PER_TASK:
                failure_task = task
        attempts_by_task[task] = tuple(task_attempts)
        selected_by_task[task] = tuple(selected)

    if pool == POOL_PREFLIGHT:
        return A4bCandidatePoolCollection(
            pool, True, None, attempts_by_task, selected_by_task, None, None
        )
    if failure_task is not None:
        return A4bCandidatePoolCollection(
            pool,
            False,
            failure_task,
            attempts_by_task,
            selected_by_task,
            None,
            None,
        )
    ledger, cache = _stack_final_artifacts(all_proposals, all_bundles)
    return A4bCandidatePoolCollection(
        pool, True, None, attempts_by_task, selected_by_task, ledger, cache
    )


def collect_a4b_preflight_pool(
    contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    *,
    legacy_physical_identities: Set[PhysicalIdentityKey],
    runtime_factory: Callable[[str, int, int], A4EpisodeRuntime],
    episode_collector: Callable[..., A4bCandidateEpisode] = collect_a4b_candidate_episode,
) -> A4bCandidatePoolCollection:
    return _run_pool(
        contract,
        source_contract,
        split_manifest,
        pool=POOL_PREFLIGHT,
        forbidden_physical_identities=legacy_physical_identities,
        runtime_factory=runtime_factory,
        encoder=None,
        episode_collector=episode_collector,
    )


def collect_a4b_final_pool(
    contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    *,
    preflight_receipt_path: str | os.PathLike[str],
    development_go_summary_path: str | os.PathLike[str],
    formal_train_cache_path: str | os.PathLike[str],
    encoder: FrozenA4StateEncoder,
    runtime_factory: Callable[[str, int, int], A4EpisodeRuntime],
    episode_collector: Callable[..., A4bCandidateEpisode] = collect_a4b_candidate_episode,
) -> A4bCandidatePoolCollection:
    preflight_receipt = load_a4b_preflight_receipt(preflight_receipt_path)
    forbidden_physical_identities = physical_identities_from_result(
        preflight_receipt, include_legacy=True
    )
    development_go_summary = _load_json_object(
        development_go_summary_path, label="A4 Development-Go summary"
    )
    validate_a4_development_go_summary(
        development_go_summary,
        source_contract,
        formal_train_cache_path=formal_train_cache_path,
    )
    return _run_pool(
        contract,
        source_contract,
        split_manifest,
        pool=POOL_FINAL,
        forbidden_physical_identities=forbidden_physical_identities,
        runtime_factory=runtime_factory,
        encoder=encoder,
        episode_collector=episode_collector,
    )


def _config_identity() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_A4_PROTOCOL,
        "split_manifest_id": SPLIT_MANIFEST_ID,
        "split_seed": SPLIT_SEED,
        "task_identity_namespace": TASK_IDENTITY_NAMESPACE,
        "benchmark_seed": BENCHMARK_SEED,
        "custom_task_parameter_seed": CUSTOM_TASK_PARAMETER_SEED,
        "proposal_seed": PROPOSAL_SEED,
        "preflight_environment_seed_base": PREFLIGHT_ENVIRONMENT_SEED_BASE,
        "final_environment_seed_base": FINAL_ENVIRONMENT_SEED_BASE,
        "camera_by_task": {
            task: _source.a4_camera_name(task) for task in TASKS
        },
        "image_size": {"width": 224, "height": 224},
        "endpoint_training_seeds": list(CONFIRMATION_TRAINING_SEEDS),
    }


def build_a4b_collection_result(
    collection: A4bCandidatePoolCollection,
    *,
    legacy_reset_identities: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the single non-overwrite success/failure result for one pool."""

    if collection.pool == POOL_PREFLIGHT:
        if legacy_reset_identities is None:
            raise ValueError("A4b preflight result requires legacy reset identities")
        status = "success"
        failure = None
    elif collection.passed:
        status = "success"
        failure = None
    else:
        status = "failure"
        available = len(
            collection.selected_episode_uids_by_task[str(collection.failure_task)]
        )
        failure = {
            "code": "insufficient_eligible_candidates",
            "task": collection.failure_task,
            "candidate_limit": FINAL_CANDIDATES_PER_TASK,
            "required": TARGET_ELIGIBLE_CANDIDATES_PER_TASK,
            "available": available,
        }
    if not collection.passed and output is not None:
        raise PermissionError("failed A4b collection cannot publish partial output")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "stage": (
            PREFLIGHT_STAGE
            if collection.pool == POOL_PREFLIGHT
            else FINAL_STAGE
        ),
        "pool": collection.pool,
        "status": status,
        "passed": bool(collection.passed),
        "config_identity": _config_identity(),
        "candidate_policy": {
            "candidate_limit_per_task": (
                PREFLIGHT_CANDIDATES_PER_TASK
                if collection.pool == POOL_PREFLIGHT
                else FINAL_CANDIDATES_PER_TASK
            ),
            "target_eligible_candidates_per_task": (
                None
                if collection.pool == POOL_PREFLIGHT
                else TARGET_ELIGIBLE_CANDIDATES_PER_TASK
            ),
            "selection": (
                "attempt_all_preregistered_candidates"
                if collection.pool == POOL_PREFLIGHT
                else "first_eligible_in_preregistered_order"
            ),
        },
        "attempts_by_task": {
            task: [row.as_record() for row in collection.attempts_by_task[task]]
            for task in TASKS
        },
        "selected_episode_uids_by_task": {
            task: list(collection.selected_episode_uids_by_task[task])
            for task in TASKS
        },
        "legacy_reset_identities": (
            None
            if legacy_reset_identities is None
            else dict(legacy_reset_identities)
        ),
        "output": None if output is None else dict(output),
        "failure": failure,
    }
    validate_a4b_collection_result(result)
    return result


def validate_a4b_collection_result(result: Mapping[str, Any]) -> None:
    """Validate candidate ordering, first-12 selection, and result state."""

    if not isinstance(result, Mapping):
        raise TypeError("A4b collection result must be a mapping")
    expected_keys = {
        "schema_version",
        "stage",
        "pool",
        "status",
        "passed",
        "config_identity",
        "candidate_policy",
        "attempts_by_task",
        "selected_episode_uids_by_task",
        "legacy_reset_identities",
        "output",
        "failure",
    }
    if set(result) != expected_keys:
        raise ValueError("A4b result keys drifted")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("A4b result schema version drifted")
    if result.get("config_identity") != _config_identity():
        raise ValueError("A4b result config identity drifted")
    pool = result.get("pool")
    if pool not in (POOL_PREFLIGHT, POOL_FINAL):
        raise ValueError("A4b result pool is invalid")
    expected_candidate_policy = {
        "candidate_limit_per_task": (
            PREFLIGHT_CANDIDATES_PER_TASK
            if pool == POOL_PREFLIGHT
            else FINAL_CANDIDATES_PER_TASK
        ),
        "target_eligible_candidates_per_task": (
            None if pool == POOL_PREFLIGHT else TARGET_ELIGIBLE_CANDIDATES_PER_TASK
        ),
        "selection": (
            "attempt_all_preregistered_candidates"
            if pool == POOL_PREFLIGHT
            else "first_eligible_in_preregistered_order"
        ),
    }
    if result.get("candidate_policy") != expected_candidate_policy:
        raise ValueError("A4b result candidate policy drifted")
    attempts = result.get("attempts_by_task")
    selected = result.get("selected_episode_uids_by_task")
    if not isinstance(attempts, Mapping) or set(attempts) != set(TASKS):
        raise ValueError("A4b result attempt task keys drifted")
    if not isinstance(selected, Mapping) or set(selected) != set(TASKS):
        raise ValueError("A4b result selection task keys drifted")

    failed_task: str | None = None
    fresh_identities: set[PhysicalIdentityKey] = set()
    for task_index, task in enumerate(TASKS):
        rows = attempts[task]
        selected_uids = selected[task]
        if not isinstance(rows, list) or not isinstance(selected_uids, list):
            raise TypeError("A4b attempts/selections must be lists")
        order = _frozen_order(pool=pool, task=task)
        if len(rows) > len(order):
            raise ValueError("A4b result exceeds candidate limit")
        eligible_uids: list[str] = []
        for rank, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError("A4b attempt row must be a mapping")
            if set(row) != {
                "candidate_rank",
                "raw_task_index",
                "environment_seed",
                "episode_uid",
                "reset_physical_state_goal_digest",
                "proposal_count",
                "eligible_proposal_count",
                "rejection_reason_counts",
                "eligible",
                "selected",
            }:
                raise ValueError("A4b attempt result keys drifted")
            raw_index = order[rank]
            environment_seed = candidate_environment_seed(
                pool=pool, task_index=task_index, rank=rank
            )
            uid = candidate_episode_uid(
                pool=pool,
                task=task,
                rank=rank,
                raw_task_index=raw_index,
                environment_seed=environment_seed,
            )
            if (
                row.get("candidate_rank") != rank
                or row.get("raw_task_index") != raw_index
                or row.get("environment_seed") != environment_seed
                or row.get("episode_uid") != uid
                or row.get("proposal_count") != 12
            ):
                raise ValueError("A4b attempt identity/order drifted")
            digest = row.get("reset_physical_state_goal_digest")
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("A4b attempt reset digest is invalid")
            physical_key = (task, digest)
            if physical_key in fresh_identities:
                raise PermissionError("A4b result contains repeated physical identity")
            fresh_identities.add(physical_key)
            reasons = row.get("rejection_reason_counts")
            if not isinstance(reasons, Mapping) or any(
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for name, value in reasons.items()
            ):
                raise ValueError("A4b rejection-reason counts are invalid")
            count = row.get("eligible_proposal_count")
            eligible = row.get("eligible")
            chosen = row.get("selected")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= 12
                or type(eligible) is not bool
                or type(chosen) is not bool
                or eligible != (count > 0)
            ):
                raise ValueError("A4b attempt eligibility is inconsistent")
            if pool == POOL_PREFLIGHT:
                if chosen:
                    raise ValueError("A4b preflight cannot select candidates")
            elif eligible:
                if not chosen:
                    raise ValueError("A4b final skipped an earlier eligible candidate")
                eligible_uids.append(uid)
        if pool == POOL_PREFLIGHT:
            if len(rows) != PREFLIGHT_CANDIDATES_PER_TASK or selected_uids:
                raise ValueError("A4b preflight must attempt its complete pool")
        else:
            expected_selected = eligible_uids[:TARGET_ELIGIBLE_CANDIDATES_PER_TASK]
            if selected_uids != expected_selected:
                raise ValueError("A4b final did not select the first 12 eligible")
            if len(selected_uids) == TARGET_ELIGIBLE_CANDIDATES_PER_TASK:
                if rows[-1]["episode_uid"] != selected_uids[-1]:
                    raise ValueError("A4b final did not stop at selected candidate 12")
            elif len(rows) == FINAL_CANDIDATES_PER_TASK and failed_task is None:
                failed_task = task
            elif rows:
                raise ValueError("A4b final stopped before target/pool exhaustion")

    if pool == POOL_PREFLIGHT:
        if (
            result.get("stage") != PREFLIGHT_STAGE
            or result.get("status") != "success"
            or result.get("passed") is not True
            or result.get("output") is not None
            or result.get("failure") is not None
            or not isinstance(result.get("legacy_reset_identities"), Mapping)
        ):
            raise ValueError("A4b preflight result state drifted")
        legacy = legacy_digest_set(result["legacy_reset_identities"])
        fresh = {
            (str(task), str(row["reset_physical_state_goal_digest"]))
            for task in TASKS
            for row in attempts[task]
        }
        if len(fresh) != len(TASKS) * PREFLIGHT_CANDIDATES_PER_TASK:
            raise ValueError("A4b preflight reset identities must be unique")
        if legacy & fresh:
            raise PermissionError("A4b preflight reset identities overlap A4 v2")
        return
    if result.get("stage") != FINAL_STAGE or result.get("legacy_reset_identities") is not None:
        raise ValueError("A4b final result state drifted")
    if result.get("passed") is True:
        output = result.get("output")
        if (
            result.get("status") != "success"
            or not isinstance(output, Mapping)
            or result.get("failure") is not None
            or failed_task is not None
        ):
            raise ValueError("A4b final success result is inconsistent")
        if set(output) != {
            "state_cache",
            "state_cache_sha256",
            "proposal_ledger",
            "proposal_ledger_sha256",
            "validated_endpoint_seeds",
        }:
            raise ValueError("A4b final output identity drifted")
        artifact_paths = [output["state_cache"], output["proposal_ledger"]]
        if any(
            not isinstance(path, str)
            or not path
            or not Path(path).is_absolute()
            for path in artifact_paths
        ) or artifact_paths[0] == artifact_paths[1]:
            raise ValueError("A4b final artifact paths must be distinct and absolute")
        for name in ("state_cache_sha256", "proposal_ledger_sha256"):
            digest = output[name]
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"A4b final {name} is invalid")
        if output["validated_endpoint_seeds"] != list(
            CONFIRMATION_TRAINING_SEEDS
        ):
            raise ValueError("A4b final endpoint seed identity drifted")
        if any(
            len(selected[task]) != TARGET_ELIGIBLE_CANDIDATES_PER_TASK
            for task in TASKS
        ):
            raise ValueError("A4b final success lacks 12 candidates per task")
    else:
        expected_failure = {
            "code": "insufficient_eligible_candidates",
            "task": failed_task,
            "candidate_limit": FINAL_CANDIDATES_PER_TASK,
            "required": TARGET_ELIGIBLE_CANDIDATES_PER_TASK,
            "available": -1 if failed_task is None else len(selected[failed_task]),
        }
        if (
            result.get("status") != "failure"
            or result.get("output") is not None
            or result.get("failure") != expected_failure
            or failed_task is None
        ):
            raise ValueError("A4b final failure result is inconsistent")


def _frozen_order(*, pool: str, task: str) -> tuple[int, ...]:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                SPLIT_SEED,
                zlib.crc32(SPLIT_NAMESPACE.encode("utf-8")),
                zlib.crc32(task.encode("utf-8")),
            ]
        )
    )
    order = tuple(int(value) for value in rng.permutation(RAW_TASK_INDICES))
    return (
        order[:PREFLIGHT_CANDIDATES_PER_TASK]
        if pool == POOL_PREFLIGHT
        else order[PREFLIGHT_CANDIDATES_PER_TASK:]
    )


def physical_identities_from_result(
    result: Mapping[str, Any], *, include_legacy: bool
) -> frozenset[PhysicalIdentityKey]:
    validate_a4b_collection_result(result)
    identities: set[PhysicalIdentityKey] = set()
    if include_legacy:
        inventory = result.get("legacy_reset_identities")
        if not isinstance(inventory, Mapping):
            raise ValueError("A4b result does not contain legacy reset identities")
        for scope in LEGACY_IDENTITY_SCOPES:
            for row in inventory[scope]:
                identities.add((str(row["task_name"]), str(row["digest"])))
    attempts = result["attempts_by_task"]
    for task in TASKS:
        identities.update(
            (str(task), str(row["reset_physical_state_goal_digest"]))
            for row in attempts[task]
        )
    return frozenset(identities)


def load_a4b_preflight_receipt(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load and validate the persisted receipt that authorizes final access."""

    receipt = _load_json_object(path, label="A4b preflight receipt")
    validate_a4b_collection_result(receipt)
    if receipt.get("pool") != POOL_PREFLIGHT or receipt.get("passed") is not True:
        raise PermissionError("A4b final requires a complete preflight receipt")
    return receipt


def _save_npz_exclusive(
    path: str | os.PathLike[str],
    arrays: Mapping[str, np.ndarray],
    keys: Sequence[str],
) -> Path:
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
            np.savez_compressed(handle, **{name: arrays[name] for name in keys})
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(destination) from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_a4b_final_cache_exclusive(
    path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]
) -> Path:
    validate_a4_state_cache(arrays)
    if set(np.asarray(arrays["split"]).astype(str).tolist()) != {"test"}:
        raise ValueError("A4b final cache must be test-only")
    return _save_npz_exclusive(path, arrays, STATE_CACHE_KEYS)


def save_a4b_final_ledger_exclusive(
    path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]
) -> Path:
    validate_a4_proposal_ledger(arrays)
    if set(np.asarray(arrays["split"]).astype(str).tolist()) != {"test"}:
        raise ValueError("A4b final ledger must be test-only")
    return _save_npz_exclusive(path, arrays, PROPOSAL_LEDGER_KEYS)


def save_a4b_result_exclusive(
    path: str | os.PathLike[str], result: Mapping[str, Any]
) -> Path:
    """Atomically publish one success/failure result without replacement."""

    validate_a4b_collection_result(result)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    payload = (
        json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(destination) from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def a4b_artifact_sha256(path: str | os.PathLike[str]) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_a4b_final_bundle_exclusive(
    *,
    result_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    ledger_path: str | os.PathLike[str],
    collection: A4bCandidatePoolCollection,
    validated_endpoint_seeds: Sequence[int],
) -> dict[str, Any]:
    """Write a complete final bundle, then publish its directory in one rename."""

    destinations = tuple(
        Path(path).expanduser().resolve()
        for path in (result_path, cache_path, ledger_path)
    )
    result_destination, cache_destination, ledger_destination = destinations
    if len(set(destinations)) != len(destinations):
        raise ValueError("A4b final result/cache/ledger paths must be distinct")
    destination_directory = result_destination.parent
    if any(path.parent != destination_directory for path in destinations):
        raise ValueError(
            "A4b final result/cache/ledger must share one publication directory"
        )
    if destination_directory.exists():
        raise FileExistsError(destination_directory)
    destination_directory.parent.mkdir(parents=True, exist_ok=True)

    temporary_directory = Path(
        tempfile.mkdtemp(
            dir=destination_directory.parent,
            prefix=f".{destination_directory.name}.",
            suffix=".tmp",
        )
    )
    try:
        staged_result = temporary_directory / result_destination.name
        staged_cache = temporary_directory / cache_destination.name
        staged_ledger = temporary_directory / ledger_destination.name
        if collection.passed:
            if collection.state_cache is None or collection.proposal_ledger is None:
                raise RuntimeError("successful A4b final collection has no artifacts")
            validate_a4_ledger_cache_pair(
                collection.proposal_ledger, collection.state_cache
            )
            save_a4b_final_cache_exclusive(staged_cache, collection.state_cache)
            save_a4b_final_ledger_exclusive(
                staged_ledger, collection.proposal_ledger
            )
            result = build_a4b_collection_result(
                collection,
                output={
                    "state_cache": str(cache_destination),
                    "state_cache_sha256": a4b_artifact_sha256(staged_cache),
                    "proposal_ledger": str(ledger_destination),
                    "proposal_ledger_sha256": a4b_artifact_sha256(staged_ledger),
                    "validated_endpoint_seeds": list(validated_endpoint_seeds),
                },
            )
        else:
            result = build_a4b_collection_result(collection)
        save_a4b_result_exclusive(staged_result, result)

        persisted = _load_json_object(staged_result, label="A4b final result")
        validate_a4b_collection_result(persisted)
        if persisted != result:
            raise RuntimeError("A4b final result changed during JSON roundtrip")
        if collection.passed:
            output = persisted["output"]
            if (
                a4b_artifact_sha256(staged_cache)
                != output["state_cache_sha256"]
                or a4b_artifact_sha256(staged_ledger)
                != output["proposal_ledger_sha256"]
            ):
                raise RuntimeError("A4b staged artifact SHA-256 changed")

        _fsync_directory(temporary_directory)
        if destination_directory.exists():
            raise FileExistsError(destination_directory)
        os.rename(temporary_directory, destination_directory)
        _fsync_directory(destination_directory.parent)
        return persisted
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)


__all__ = [
    "A4bCandidateAttempt",
    "A4bCandidateEpisode",
    "A4bCandidatePoolCollection",
    "a4b_artifact_sha256",
    "build_a4b_collection_result",
    "collect_a4b_candidate_episode",
    "collect_a4b_final_pool",
    "collect_a4b_preflight_pool",
    "load_a4b_preflight_receipt",
    "physical_identities_from_result",
    "publish_a4b_final_bundle_exclusive",
    "save_a4b_final_cache_exclusive",
    "save_a4b_final_ledger_exclusive",
    "save_a4b_result_exclusive",
    "validate_a4b_collection_result",
]
