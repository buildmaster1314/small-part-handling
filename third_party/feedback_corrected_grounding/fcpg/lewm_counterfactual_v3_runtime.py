"""Fresh physical collection for the Counterfactual v3 StateOracle gate.

Only the frozen FCPG action policy is loaded, and only to define the center of
each anonymous equal-norm candidate bank.  Every candidate is executed once
from an exact MetaWorld snapshot.  True semantic state and outcomes are stored
as offline diagnostics with no camera/view axis.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from fcpg.lewm_counterfactual_runtime import (
    _git_state,
    _input_file_record,
    _torch_payload,
    _validate_corrector_payload,
    _validate_policy_payload,
    _verify_loaded_corrector_binding,
    _versions,
    physical_snapshot_sha256,
    policy_action_chunk,
    restore_branch_snapshot,
    sha256_file,
    sha256_array,
    sha256_state_dict,
    validate_candidate_replay,
)
from fcpg.lewm_counterfactual_v2 import (
    assign_physical_episode_splits,
    physical_episode_sha256,
)
from fcpg.lewm_counterfactual_v2_runtime import (
    PhysicalEpisode,
    _close_episode_env_safely,
    collect_physical_episode,
)
from fcpg.lewm_counterfactual_v3 import (
    ACTION_DIM,
    CANDIDATE_COUNT,
    FRAME_SKIP,
    HORIZON,
    PHASE_NAMES,
    PHASE_STEPS,
    PROTOCOL,
    RAW_ACTION_STEPS,
    TASK_NAMES,
    V3Axes,
    canonical_sha256,
    validate_array_bundle,
    validate_action_hash_bindings,
    validate_config,
    validate_index_rows,
)
from fcpg.lewm_counterfactual_v3_candidates import (
    AnonymousCandidateBank,
    EqualNormCandidateAudit,
    build_equal_norm_candidate_bank,
)
from fcpg.lewm_counterfactual_v3_state import (
    STATE_DIM,
    deterministic_oracle_scores,
    extract_true_simulator_state,
)


@dataclass(frozen=True)
class FrozenPolicyInputs:
    policy: Any
    policy_args: Mapping[str, Any]
    corrector: Any
    device: torch.device
    file_records: Mapping[str, Mapping[str, Any]]
    state_sha256: Mapping[str, str]


@dataclass(frozen=True)
class V3BranchOutcome:
    future_state: np.ndarray
    cumulative_return: np.ndarray
    progress: np.ndarray
    success: np.ndarray
    action_sha256: str


@dataclass(frozen=True)
class V3CollectionPayload:
    deployment_inputs: Mapping[str, np.ndarray]
    offline_labels: Mapping[str, np.ndarray]
    audit_arrays: Mapping[str, np.ndarray]
    index_rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    axes: V3Axes


def _require_clean_git_state(repo_root: Path) -> Mapping[str, Any]:
    state = _git_state(repo_root)
    if bool(state.get("dirty", True)):
        raise RuntimeError("v3 scientific execution requires a clean git worktree")
    return state


def _verify_frozen_inputs_unchanged(frozen: FrozenPolicyInputs) -> None:
    modules = {
        "fcpg_corrector": frozen.corrector,
        "action_only_policy": frozen.policy,
    }
    for name, module in modules.items():
        actual_state = sha256_state_dict(module)
        if actual_state != frozen.state_sha256[name]:
            raise RuntimeError(f"loaded {name} state changed during v3 collection")
        record = frozen.file_records[name]
        actual_file = sha256_file(Path(str(record["path"])))
        if actual_file != record["sha256"]:
            raise RuntimeError(f"{name} checkpoint changed during v3 collection")


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _namespace_seed(
    config: Mapping[str, Any],
    namespace: str,
    *,
    task: str,
    episode: int,
    snapshot_step: int,
    attempt: int = 0,
) -> int:
    if task not in TASK_NAMES or not namespace.isascii() or not namespace:
        raise ValueError("invalid v3 seed namespace")
    payload = (
        f"{PROTOCOL}|{int(config['seed'])}|{namespace}|{TASK_NAMES.index(task)}|"
        f"{int(episode)}|{int(snapshot_step)}|{int(attempt)}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_frozen_policy_inputs(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> FrozenPolicyInputs:
    """Load only the corrector-bound action policy; never load a LeWM model."""

    validate_config(config)
    inputs = _object(config.get("inputs"), "inputs")
    corrector_path, corrector_record = _input_file_record(
        "fcpg_corrector",
        _object(inputs.get("fcpg_corrector"), "inputs.fcpg_corrector"),
        repo_root,
    )
    policy_path, policy_record = _input_file_record(
        "action_only_policy",
        _object(inputs.get("action_only_policy"), "inputs.action_only_policy"),
        repo_root,
    )
    corrector_payload = _torch_payload(corrector_path)
    policy_payload = _torch_payload(policy_path)
    _validate_corrector_payload(corrector_payload)
    _validate_policy_payload(policy_payload, corrector_path, repo_root)
    policy_metadata = _object(policy_payload.get("args"), "policy args")
    corrector_metadata = _object(corrector_payload.get("args"), "corrector args")
    if not bool(policy_metadata.get("reuse_fcpg_vision", False)):
        raise ValueError("v3 frozen policy must reuse the verified FCPG vision encoder")
    for policy_field, corrector_field in (
        ("vision_model_name", "model_name"),
        ("vision_selected_layers", "vision_selected_layers"),
    ):
        policy_value = policy_metadata.get(policy_field)
        corrector_value = corrector_metadata.get(corrector_field)
        if policy_field == "vision_selected_layers":
            policy_value = tuple(int(value) for value in policy_value)
            corrector_value = tuple(int(value) for value in corrector_value)
        if policy_value != corrector_value:
            raise ValueError(f"policy/corrector {policy_field} metadata differs")

    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    from fcpg.action_chunk_policy import load_action_chunk_policy
    from fcpg.transformer_grounding import load_fcpg_vit_adapter

    corrector, _ = load_fcpg_vit_adapter(corrector_path, device)
    corrector.requires_grad_(False).eval()
    state_before = sha256_state_dict(corrector)
    policy, policy_args = load_action_chunk_policy(
        policy_path,
        device,
        fcpg_adapter_override=corrector,
    )
    policy.requires_grad_(False).eval()
    binding = _verify_loaded_corrector_binding(
        policy,
        corrector,
        checkpoint_path=corrector_path,
        expected_file_sha256=corrector_record["sha256"],
        state_sha256_before_binding=state_before,
    )
    return FrozenPolicyInputs(
        policy=policy,
        policy_args=dict(policy_args),
        corrector=corrector,
        device=device,
        file_records={
            "fcpg_corrector": {**corrector_record, **binding},
            "action_only_policy": policy_record,
        },
        state_sha256={
            "fcpg_corrector": sha256_state_dict(corrector),
            "action_only_policy": sha256_state_dict(policy),
        },
    )


def build_v3_candidate_bank(
    episode: PhysicalEpisode,
    snapshot_step: int,
    frozen: FrozenPolicyInputs,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, AnonymousCandidateBank, EqualNormCandidateAudit]:
    plan = SimpleNamespace(
        snapshot_step=int(snapshot_step),
        records=episode.records_clean,
        actions=episode.actions,
    )
    center = np.asarray(policy_action_chunk(plan, frozen), dtype=np.float32)
    low = np.asarray(episode.env.action_space.low, dtype=np.float32)
    high = np.asarray(episode.env.action_space.high, dtype=np.float32)
    if not np.array_equal(np.clip(center, low, high).astype(np.float32), center):
        raise RuntimeError("frozen policy center changes under executor clipping")
    evaluation = _object(config.get("evaluation"), "evaluation")
    permutation_seed = _namespace_seed(
        config,
        "candidate_slot_permutation",
        task=episode.task,
        episode=episode.episode,
        snapshot_step=snapshot_step,
    )
    errors: list[str] = []
    for attempt in range(int(evaluation["max_basis_attempts"])):
        direction_seed = _namespace_seed(
            config,
            "candidate_direction_basis",
            task=episode.task,
            episode=episode.episode,
            snapshot_step=snapshot_step,
            attempt=attempt,
        )
        try:
            bank, audit = build_equal_norm_candidate_bank(
                center,
                direction_seed=direction_seed,
                permutation_seed=permutation_seed,
                action_low=low,
                action_high=high,
            )
            return center, bank, audit
        except ValueError as error:
            errors.append(str(error))
    raise RuntimeError(
        "all 64 pre-registered candidate bases failed: " + "; ".join(errors[-3:])
    )


def validate_loaded_candidate_artifacts(
    config: Mapping[str, Any],
    deployment_inputs: Mapping[str, np.ndarray],
    audit_arrays: Mapping[str, np.ndarray],
) -> None:
    """Regenerate every candidate bank and bind hidden audit metadata.

    This is intentionally stronger than checking stored RMS values.  The
    formal reader derives the frozen direction/permutation seeds again, tries
    the same first-valid-basis sequence, and requires byte-identical public
    actions plus exact canonical axis/sign/permutation metadata.
    """

    validate_config(config)
    centers = np.asarray(deployment_inputs["center_action_raw"])
    candidates = np.asarray(deployment_inputs["candidate_action_raw"])
    n = centers.shape[0]
    if centers.shape != (n, RAW_ACTION_STEPS, ACTION_DIM) or candidates.shape != (
        n,
        CANDIDATE_COUNT,
        RAW_ACTION_STEPS,
        ACTION_DIM,
    ):
        raise ValueError("loaded candidate arrays have changed shape")
    evaluation = _object(config.get("evaluation"), "evaluation")
    low = np.full(ACTION_DIM, -1.0, dtype=np.float32)
    high = np.full(ACTION_DIM, 1.0, dtype=np.float32)
    for sample in range(n):
        task = TASK_NAMES[int(audit_arrays["task_index"][sample])]
        episode = int(audit_arrays["episode_id"][sample])
        snapshot_step = int(audit_arrays["snapshot_step"][sample])
        permutation_seed = _namespace_seed(
            config,
            "candidate_slot_permutation",
            task=task,
            episode=episode,
            snapshot_step=snapshot_step,
        )
        matched_audit: EqualNormCandidateAudit | None = None
        for attempt in range(int(evaluation["max_basis_attempts"])):
            direction_seed = _namespace_seed(
                config,
                "candidate_direction_basis",
                task=task,
                episode=episode,
                snapshot_step=snapshot_step,
                attempt=attempt,
            )
            try:
                bank, candidate_audit = build_equal_norm_candidate_bank(
                    centers[sample],
                    direction_seed=direction_seed,
                    permutation_seed=permutation_seed,
                    action_low=low,
                    action_high=high,
                )
            except ValueError:
                continue
            if not np.array_equal(bank.actions, candidates[sample]):
                raise ValueError(
                    f"candidate bank at sample {sample} is not the first valid frozen bank"
                )
            matched_audit = candidate_audit
            break
        if matched_audit is None:
            raise ValueError(
                f"candidate bank at sample {sample} is not the first valid frozen bank"
            )
        canonical = np.asarray(
            matched_audit.slot_to_canonical_direction, dtype=np.int16
        )
        expected_axis = canonical // 2
        expected_sign = np.where(canonical % 2 == 0, 1, -1).astype(np.int8)
        expected_rms = np.asarray(
            matched_audit.realized_motion_rms, dtype=np.float32
        )
        if not np.array_equal(
            np.asarray(audit_arrays["candidate_permutation"][sample]), canonical
        ):
            raise ValueError("candidate permutation is not bound to regenerated actions")
        if not np.array_equal(
            np.asarray(audit_arrays["candidate_axis"][sample]), expected_axis
        ):
            raise ValueError("candidate axis is not bound to canonical permutation")
        if not np.array_equal(
            np.asarray(audit_arrays["candidate_sign"][sample]), expected_sign
        ):
            raise ValueError("candidate sign is not bound to canonical permutation")
        if not np.array_equal(
            np.asarray(audit_arrays["candidate_delta_rms"][sample]), expected_rms
        ):
            raise ValueError("candidate RMS audit does not match regenerated actions")


def execute_v3_branch(
    episode: PhysicalEpisode,
    snapshot_step: int,
    actions: np.ndarray,
    config: Mapping[str, Any],
) -> V3BranchOutcome:
    """Execute one already-clipped action chunk and extract five true states."""

    value = np.asarray(actions)
    if value.dtype != np.float32 or value.shape != (RAW_ACTION_STEPS, ACTION_DIM):
        raise ValueError("v3 candidate must be float32 [25,4]")
    low = np.asarray(episode.env.action_space.low, dtype=np.float32)
    high = np.asarray(episode.env.action_space.high, dtype=np.float32)
    if not np.array_equal(np.clip(value, low, high).astype(np.float32), value):
        raise RuntimeError("v3 candidate changed under executor clipping")
    restore_branch_snapshot(episode.env, episode.snapshots[int(snapshot_step)])
    observation = np.asarray(episode.observations[int(snapshot_step)], dtype=np.float32)
    evaluation = _object(config.get("evaluation"), "evaluation")
    progress_key = str(evaluation["progress_info_key"])
    states: list[np.ndarray] = []
    returns: list[float] = []
    progress: list[float] = []
    success: list[bool] = []
    cumulative = 0.0
    for raw_step, action in enumerate(value, start=1):
        observation, reward, terminated, truncated, info = episode.env.step(action)
        cumulative += float(reward)
        if raw_step % FRAME_SKIP == 0:
            if not isinstance(info, Mapping) or progress_key not in info:
                raise KeyError(f"MetaWorld branch info is missing {progress_key!r}")
            state = extract_true_simulator_state(
                episode.env,
                observation,
                episode.task,
            )
            if not np.array_equal(state.goal_world, episode.goal_world):
                raise RuntimeError("physical branch changed the task goal")
            states.append(state.state)
            returns.append(cumulative)
            progress.append(float(info[progress_key]))
            success.append(bool(float(info.get("success", 0.0)) > 0.0))
        if (terminated or truncated) and raw_step < RAW_ACTION_STEPS:
            raise RuntimeError("v3 branch ended before all 25 raw actions")
    future = np.stack(states).astype(np.float32, copy=False)
    if future.shape != (HORIZON, STATE_DIM):
        raise RuntimeError("v3 branch did not produce five semantic states")
    return V3BranchOutcome(
        future_state=future,
        cumulative_return=np.asarray(returns, dtype=np.float32),
        progress=np.asarray(progress, dtype=np.float32),
        success=np.asarray(success, dtype=bool),
        action_sha256=sha256_array(value),
    )


def _known_v2_overlap(
    config: Mapping[str, Any],
    repo_root: Path,
    identities: set[str],
) -> dict[str, Any]:
    known = _object(config.get("known_disjoint_dataset"), "known_disjoint_dataset")
    path = Path(str(known["path"]))
    path = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    if not path.exists():
        if bool(known.get("required", False)):
            raise FileNotFoundError(
                f"required known v2 dataset is absent; cannot prove non-overlap: {path}"
            )
        return {"status": "not_present", "path": str(path), "overlap": []}
    from fcpg.lewm_counterfactual_v2_storage import load_counterfactual_v2_store

    old = load_counterfactual_v2_store(path)
    old_ids = {
        str(row["physical_episode_sha256"])
        for row in old.index_rows
        if "physical_episode_sha256" in row
    }
    overlap = sorted(identities.intersection(old_ids))
    if overlap:
        raise RuntimeError("fresh v3 episodes overlap the known v2 dataset")
    return {
        "status": "passed",
        "path": str(path),
        "old_episode_count": len(old_ids),
        "new_episode_count": len(identities),
        "overlap": overlap,
    }


def _stack(values: list[np.ndarray], dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.stack(values)
    return result.astype(dtype, copy=False) if dtype is not None else result


def _branch_outcome_sha256(value: V3BranchOutcome) -> str:
    return canonical_sha256(
        {
            "future_state": sha256_array(value.future_state),
            "cumulative_return": sha256_array(value.cumulative_return),
            "progress": sha256_array(value.progress),
            "success": sha256_array(value.success),
            "action_sha256": value.action_sha256,
        }
    )


def run_v3_collection(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> V3CollectionPayload:
    """Collect the complete fresh P0/P1 array bundle in canonical task order."""

    validate_config(config)
    git_before = _require_clean_git_state(repo_root)
    _set_seed(int(config["seed"]))
    frozen = load_frozen_policy_inputs(config, repo_root=repo_root)
    evaluation = _object(config.get("evaluation"), "evaluation")
    split_config = _object(config.get("split"), "split")
    episodes_per_task = int(evaluation["episodes_per_task"])

    center_actions: list[np.ndarray] = []
    candidate_actions: list[np.ndarray] = []
    current_states: list[np.ndarray] = []
    future_states: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    physical_task_scores: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    progresses: list[np.ndarray] = []
    successes: list[np.ndarray] = []
    task_indices: list[int] = []
    episode_ids: list[int] = []
    env_seeds: list[int] = []
    snapshot_steps: list[int] = []
    phase_indices: list[int] = []
    axes: list[np.ndarray] = []
    signs: list[np.ndarray] = []
    permutations: list[np.ndarray] = []
    delta_norms: list[np.ndarray] = []
    row_drafts: list[dict[str, Any]] = []
    replay_receipts: list[dict[str, Any]] = []

    for task_index, task in enumerate(TASK_NAMES):
        for episode_id in range(episodes_per_task):
            episode = collect_physical_episode(task, episode_id, config)
            try:
                episode_identity = physical_episode_sha256(
                    task_index,
                    episode_id,
                    episode.env_seed,
                )
                for phase_index, snapshot_step in enumerate(PHASE_STEPS):
                    restore_branch_snapshot(episode.env, episode.snapshots[snapshot_step])
                    current = extract_true_simulator_state(
                        episode.env,
                        episode.observations[snapshot_step],
                        task,
                    )
                    if not np.array_equal(current.goal_world, episode.goal_world):
                        raise RuntimeError("snapshot goal disagrees with collected goal")
                    center, bank, bank_audit = build_v3_candidate_bank(
                        episode,
                        snapshot_step,
                        frozen,
                        config,
                    )
                    if episode_id == 0 and snapshot_step == PHASE_STEPS[0]:
                        names = tuple(bank.names)
                        action_by_name = {
                            name: np.asarray(bank.actions[index])
                            for index, name in enumerate(names)
                        }
                        original, replay = validate_candidate_replay(
                            names,
                            lambda name: execute_v3_branch(
                                episode,
                                snapshot_step,
                                action_by_name[name],
                                config,
                            ),
                            repeat_name=names[0],
                            outcome_sha256=_branch_outcome_sha256,
                        )
                        outcomes = [original[name] for name in names]
                        replay_receipts.append(
                            {
                                "task": task,
                                "episode_id": episode_id,
                                "snapshot_step": snapshot_step,
                                **replay,
                            }
                        )
                    else:
                        outcomes = [
                            execute_v3_branch(
                                episode,
                                snapshot_step,
                                np.asarray(bank.actions[index]),
                                config,
                            )
                            for index in range(CANDIDATE_COUNT)
                        ]
                    future_matrix = _stack(
                        [value.future_state for value in outcomes],
                        np.dtype(np.float32),
                    )
                    physical_task_score = deterministic_oracle_scores(
                        task,
                        current.state,
                        future_matrix,
                        current.goal_world,
                    ).score
                    canonical = np.asarray(
                        bank_audit.slot_to_canonical_direction,
                        dtype=np.int16,
                    )
                    candidate_axis = canonical // 2
                    candidate_sign = np.where(canonical % 2 == 0, 1, -1).astype(np.int8)
                    current_snapshot_hash = physical_snapshot_sha256(
                        episode.snapshots[snapshot_step]
                    )

                    center_actions.append(center)
                    candidate_actions.append(np.asarray(bank.actions))
                    current_states.append(current.state)
                    future_states.append(future_matrix)
                    goals.append(current.goal_world)
                    physical_task_scores.append(physical_task_score)
                    returns.append(_stack([value.cumulative_return for value in outcomes]))
                    progresses.append(_stack([value.progress for value in outcomes]))
                    successes.append(_stack([value.success for value in outcomes]))
                    task_indices.append(task_index)
                    episode_ids.append(episode_id)
                    env_seeds.append(episode.env_seed)
                    snapshot_steps.append(snapshot_step)
                    phase_indices.append(phase_index)
                    axes.append(candidate_axis)
                    signs.append(candidate_sign)
                    permutations.append(canonical)
                    delta_norms.append(
                        np.asarray(bank_audit.realized_motion_rms, dtype=np.float32)
                    )
                    row_drafts.append(
                        {
                            "task": task,
                            "episode_id": episode_id,
                            "env_seed": episode.env_seed,
                            "phase": PHASE_NAMES[phase_index],
                            "snapshot_step": snapshot_step,
                            "physical_episode_sha256": episode_identity,
                            "physical_snapshot_sha256": current_snapshot_hash,
                            "center_action_sha256": sha256_array(center),
                            "candidate_action_sha256": [
                                value.action_sha256 for value in outcomes
                            ],
                        }
                    )
            finally:
                _close_episode_env_safely(episode.env)

    task_array = np.asarray(task_indices, dtype=np.int16)
    episode_array = np.asarray(episode_ids, dtype=np.int32)
    env_seed_array = np.asarray(env_seeds, dtype=np.int64)
    assignments, split_report = assign_physical_episode_splits(
        task_array,
        episode_array,
        env_seed_array,
        train_count_per_task=int(split_config["train_count_per_task"]),
        validation_count_per_task=int(split_config["validation_count_per_task"]),
    )
    split_index = np.where(assignments == "train", 0, 1).astype(np.int8)
    index_rows = tuple(
        {
            "sample_index": index,
            **draft,
            "split": str(assignments[index]),
        }
        for index, draft in enumerate(row_drafts)
    )
    deployment = {
        "center_action_raw": _stack(center_actions, np.dtype(np.float32)),
        "candidate_action_raw": _stack(candidate_actions, np.dtype(np.float32)),
    }
    labels = {
        "current_state": _stack(current_states, np.dtype(np.float32)),
        "future_state": _stack(future_states, np.dtype(np.float32)),
        "goal_world": _stack(goals, np.dtype(np.float32)),
        "physical_task_score": _stack(
            physical_task_scores, np.dtype(np.float32)
        ),
        "actual_return": _stack(returns, np.dtype(np.float32)),
        "progress": _stack(progresses, np.dtype(np.float32)),
        "success": _stack(successes, np.dtype(bool)),
    }
    audit = {
        "task_index": task_array,
        "episode_id": episode_array,
        "env_seed": env_seed_array,
        "snapshot_step": np.asarray(snapshot_steps, dtype=np.int16),
        "phase_index": np.asarray(phase_indices, dtype=np.int8),
        "split_index": split_index,
        "candidate_axis": _stack(axes, np.dtype(np.int16)),
        "candidate_sign": _stack(signs, np.dtype(np.int8)),
        "candidate_permutation": _stack(permutations, np.dtype(np.int16)),
        "candidate_delta_rms": _stack(delta_norms, np.dtype(np.float32)),
    }
    resolved_axes = validate_array_bundle(deployment, labels, audit)
    validate_index_rows(index_rows, axes=resolved_axes)
    validate_action_hash_bindings(deployment, index_rows)
    identities = {str(row["physical_episode_sha256"]) for row in index_rows}
    v2_disjoint = _known_v2_overlap(config, repo_root, identities)
    _verify_frozen_inputs_unchanged(frozen)
    git_after = _require_clean_git_state(repo_root)
    if git_after != git_before:
        raise RuntimeError("git state changed during v3 collection")
    provenance = {
        "protocol": PROTOCOL,
        "config": json.loads(json.dumps(config)),
        "config_sha256": canonical_sha256(config),
        "input_files": dict(frozen.file_records),
        "loaded_state_sha256": dict(frozen.state_sha256),
        "lewm_loaded": False,
        "oracle_target_encoder_loaded": False,
        "physical_episode_split": split_report,
        "known_v2_non_overlap": v2_disjoint,
        "candidate_replay_audit": {
            "scope": "episode_0_early_snapshot_per_task",
            "tasks": len(replay_receipts),
            "extra_branch_executions": len(replay_receipts) * (CANDIDATE_COUNT + 1),
            "receipts": replay_receipts,
        },
        "upstream_checkpoint_episode_non_overlap": {
            "status": "unverifiable_missing_complete_training_episode_receipts",
            "formal_claim_allowed": False,
        },
        "git": dict(git_after),
        "versions": _versions(),
    }
    return V3CollectionPayload(
        deployment_inputs=deployment,
        offline_labels=labels,
        audit_arrays=audit,
        index_rows=index_rows,
        provenance=provenance,
        axes=resolved_axes,
    )


__all__ = [
    "FrozenPolicyInputs",
    "V3BranchOutcome",
    "V3CollectionPayload",
    "build_v3_candidate_bank",
    "execute_v3_branch",
    "load_frozen_policy_inputs",
    "run_v3_collection",
    "validate_loaded_candidate_artifacts",
    "_require_clean_git_state",
]
