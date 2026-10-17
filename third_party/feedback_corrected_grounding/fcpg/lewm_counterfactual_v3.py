"""Fail-closed contract for the Counterfactual v3 StateOracle gate.

V3 is intentionally independent from the v2 visual/latent experiment.  Its
first gate asks a narrower question: do anonymous, equal-norm action branches
expose enough future physical variation for a privileged state oracle to rank
them better than a direct current-state action ranker?

True simulator state and goal coordinates are offline diagnostic inputs.  They
are never represented as deployable inputs and may not be consumed by a later
LeWM or GoalToken experiment without a separate protocol.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


PROTOCOL = "fcpg_lewm_counterfactual_v3_state_oracle_gate_v1"
SCHEMA_VERSION = 3
EXPECTED_CONFIG_SHA256 = "8b8a63cd2dbb5e04bdc9362233c8d36414cd996338fb5e75bd4436d53aacdc6f"
TASK_NAMES = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)
PHASE_NAMES = ("early", "mid", "late")
PHASE_STEPS = (10, 25, 35)
HISTORY_SIZE = 3
FRAME_SKIP = 5
RAW_ACTION_STEPS = 25
ACTION_DIM = 4
HORIZON = 5
CANDIDATE_COUNT = 8

DEPLOYMENT_INPUT_KEYS = frozenset({"center_action_raw", "candidate_action_raw"})
OFFLINE_LABEL_KEYS = frozenset(
    {
        "current_state",
        "future_state",
        "goal_world",
        "physical_task_score",
        "actual_return",
        "progress",
        "success",
    }
)
AUDIT_ARRAY_KEYS = frozenset(
    {
        "task_index",
        "episode_id",
        "env_seed",
        "snapshot_step",
        "phase_index",
        "split_index",
        "candidate_axis",
        "candidate_sign",
        "candidate_permutation",
        "candidate_delta_rms",
    }
)
PRIVILEGED_KEYS = OFFLINE_LABEL_KEYS | frozenset(
    {"reward", "return", "info", "q_true", "future_success_frame"}
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"{label} keys changed: missing={missing}, extra={extra}")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the one permitted v3 StateOracle development configuration."""

    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("v3 schema_version must be 3")
    if config.get("protocol") != PROTOCOL:
        raise ValueError("v3 protocol identifier changed")
    scope = _mapping(config.get("scope"), "scope")
    if scope.get("classification") != "external_development":
        raise ValueError("v3 StateOracle is external development only")
    for forbidden in ("upstream_training_allowed", "online_cem_allowed", "formal_claim_allowed"):
        if bool(scope.get(forbidden, True)):
            raise ValueError(f"scope.{forbidden} must remain false")
    if str(config.get("device")) != "cuda:0":
        raise ValueError("the frozen-policy collector is pinned to cuda:0")

    inputs = _mapping(config.get("inputs"), "inputs")
    if set(inputs) != {"action_only_policy", "fcpg_corrector"}:
        raise ValueError("v3 P0/P1 may load only the frozen policy and FCPG corrector")
    for name, record in inputs.items():
        item = _mapping(record, f"inputs.{name}")
        digest = item.get("sha256")
        if not isinstance(item.get("path"), str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"inputs.{name} must bind path and SHA-256")

    evaluation = _mapping(config.get("evaluation"), "evaluation")
    if tuple(evaluation.get("tasks", ())) != TASK_NAMES:
        raise ValueError("v3 task family changed")
    if tuple(int(value) for value in evaluation.get("phase_steps", ())) != PHASE_STEPS:
        raise ValueError("v3 phase steps changed")
    if tuple(evaluation.get("phase_names", ())) != PHASE_NAMES:
        raise ValueError("v3 phase names changed")
    fixed = {
        "history_size": HISTORY_SIZE,
        "frame_skip": FRAME_SKIP,
        "raw_action_steps": RAW_ACTION_STEPS,
        "action_dim": ACTION_DIM,
        "horizon": HORIZON,
        "candidate_count": CANDIDATE_COUNT,
    }
    for name, expected in fixed.items():
        if int(evaluation.get(name, -1)) != expected:
            raise ValueError(f"evaluation.{name} must be {expected}")
    if float(evaluation.get("motion_delta_rms", -1.0)) != 0.15:
        raise ValueError("v3 motion_delta_rms is pre-registered at 0.15")
    if int(evaluation.get("max_basis_attempts", -1)) != 64:
        raise ValueError("v3 max_basis_attempts is pre-registered at 64")
    episodes = int(evaluation.get("episodes_per_task", 0))
    if episodes <= 1:
        raise ValueError("v3 needs independent train and validation episodes")

    split = _mapping(config.get("split"), "split")
    if split.get("unit") != "physical_episode" or split.get("assignment") != "sha256_sorted":
        raise ValueError("v3 split must be SHA-sorted physical episodes")
    train_count = int(split.get("train_count_per_task", 0))
    validation_count = int(split.get("validation_count_per_task", 0))
    if train_count <= 0 or validation_count <= 0 or train_count + validation_count != episodes:
        raise ValueError("v3 split counts must partition episodes_per_task")

    gate = _mapping(config.get("gate"), "gate")
    expected_gate = {
        "minimum_return_spread_fraction": 0.8,
        "shortcut_abs_spearman_max": 0.1,
        "shortcut_regret_improvement_max": 0.02,
        "bootstrap_samples": 10000,
    }
    for name, expected in expected_gate.items():
        value = gate.get(name)
        if float(value) != float(expected):
            raise ValueError(f"gate.{name} must remain {expected}")
    if gate.get("physical_return_alignment_required") is not True:
        raise ValueError("gate.physical_return_alignment_required must remain true")

    output = _mapping(config.get("output"), "output")
    if output.get("write_mode") != "exclusive_atomic_directory":
        raise ValueError("v3 output must be exclusive and immutable")
    for name in ("dataset_directory", "gate_directory"):
        if not isinstance(output.get(name), str) or not output[name]:
            raise ValueError(f"output.{name} must be a non-empty path")
    actual_sha256 = canonical_sha256(config)
    if actual_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError(
            "v3 pre-registered config changed: "
            f"expected {EXPECTED_CONFIG_SHA256}, got {actual_sha256}"
        )


@dataclass(frozen=True)
class V3Axes:
    sample_count: int
    candidate_count: int
    horizon: int
    state_dim: int


def _finite(value: np.ndarray, label: str) -> None:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{label} must be finite numeric data")


def validate_array_bundle(
    deployment_inputs: Mapping[str, np.ndarray],
    offline_labels: Mapping[str, np.ndarray],
    audit_arrays: Mapping[str, np.ndarray],
) -> V3Axes:
    """Prove the privileged/deployable split and all aligned v3 axes."""

    _exact_keys(deployment_inputs, DEPLOYMENT_INPUT_KEYS, "deployment_inputs")
    _exact_keys(offline_labels, OFFLINE_LABEL_KEYS, "offline_labels")
    _exact_keys(audit_arrays, AUDIT_ARRAY_KEYS, "audit_arrays")
    if set(deployment_inputs).intersection(PRIVILEGED_KEYS):
        raise ValueError("privileged state leaked into deployment_inputs")
    values = {**deployment_inputs, **offline_labels, **audit_arrays}
    if any(not isinstance(value, np.ndarray) for value in values.values()):
        raise TypeError("every v3 array must already be a numpy ndarray")

    candidate = deployment_inputs["candidate_action_raw"]
    if candidate.ndim != 4 or candidate.shape[1:] != (
        CANDIDATE_COUNT,
        RAW_ACTION_STEPS,
        ACTION_DIM,
    ):
        raise ValueError("candidate_action_raw must have shape [N,8,25,4]")
    n = int(candidate.shape[0])
    if n <= 0:
        raise ValueError("v3 dataset must not be empty")
    if deployment_inputs["center_action_raw"].shape != (n, RAW_ACTION_STEPS, ACTION_DIM):
        raise ValueError("center_action_raw must have shape [N,25,4]")

    current = offline_labels["current_state"]
    future = offline_labels["future_state"]
    if current.ndim != 2 or current.shape[0] != n or current.shape[1] <= 0:
        raise ValueError("current_state must have shape [N,S]")
    state_dim = int(current.shape[1])
    if future.shape != (n, CANDIDATE_COUNT, HORIZON, state_dim):
        raise ValueError("future_state must have shape [N,8,5,S]")
    if offline_labels["goal_world"].shape != (n, 3):
        raise ValueError("goal_world must have shape [N,3]")
    if offline_labels["physical_task_score"].shape != (n, CANDIDATE_COUNT):
        raise ValueError("physical_task_score must have shape [N,8]")
    for name in ("actual_return", "progress", "success"):
        if offline_labels[name].shape != (n, CANDIDATE_COUNT, HORIZON):
            raise ValueError(f"{name} must have shape [N,8,5]")
    if offline_labels["success"].dtype != np.bool_:
        raise TypeError("success must be boolean")

    vector_shapes = {
        "task_index": (n,),
        "episode_id": (n,),
        "env_seed": (n,),
        "snapshot_step": (n,),
        "phase_index": (n,),
        "split_index": (n,),
    }
    matrix_shapes = {
        "candidate_axis": (n, CANDIDATE_COUNT),
        "candidate_sign": (n, CANDIDATE_COUNT),
        "candidate_permutation": (n, CANDIDATE_COUNT),
        "candidate_delta_rms": (n, CANDIDATE_COUNT),
    }
    for name, shape in {**vector_shapes, **matrix_shapes}.items():
        if audit_arrays[name].shape != shape:
            raise ValueError(f"audit.{name} must have shape {shape}")
    for label, value in values.items():
        if value.shape[0] != n:
            raise ValueError(f"{label} leading sample axis changed")
        if value.dtype != np.bool_:
            _finite(value, label)
    if not np.isin(audit_arrays["task_index"], np.arange(len(TASK_NAMES))).all():
        raise ValueError("task_index is outside the fixed task family")
    if not np.isin(audit_arrays["phase_index"], np.arange(len(PHASE_NAMES))).all():
        raise ValueError("phase_index is outside the fixed phases")
    if not np.isin(audit_arrays["split_index"], (0, 1)).all():
        raise ValueError("split_index must encode train=0/validation=1")
    return V3Axes(n, CANDIDATE_COUNT, HORIZON, state_dim)


def validate_index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    axes: V3Axes,
) -> None:
    if len(rows) != axes.sample_count:
        raise ValueError("v3 index row count does not match arrays")
    required = {
        "sample_index",
        "task",
        "episode_id",
        "env_seed",
        "phase",
        "snapshot_step",
        "split",
        "physical_episode_sha256",
        "physical_snapshot_sha256",
        "center_action_sha256",
        "candidate_action_sha256",
    }
    for index, row in enumerate(rows):
        if set(row) != required:
            raise ValueError("v3 index row keys changed")
        if int(row["sample_index"]) != index:
            raise ValueError("v3 sample_index is not array aligned")
        if row["task"] not in TASK_NAMES or row["phase"] not in PHASE_NAMES:
            raise ValueError("v3 index row has unknown task or phase")
        if row["split"] not in {"train", "validation"}:
            raise ValueError("v3 index row has invalid split")
        hashes = (
            row["physical_episode_sha256"],
            row["physical_snapshot_sha256"],
            row["center_action_sha256"],
        )
        candidate_hashes = row["candidate_action_sha256"]
        if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
            raise ValueError("v3 index row has malformed SHA-256")
        if (
            not isinstance(candidate_hashes, list)
            or len(candidate_hashes) != CANDIDATE_COUNT
            or len(set(candidate_hashes)) != CANDIDATE_COUNT
            or any(not isinstance(value, str) or len(value) != 64 for value in candidate_hashes)
        ):
            raise ValueError("v3 candidate action hashes are malformed or non-unique")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_action_hash_bindings(
    deployment_inputs: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute every published action hash from the aligned array payload."""

    center = deployment_inputs["center_action_raw"]
    candidates = deployment_inputs["candidate_action_raw"]
    if len(rows) != center.shape[0] or candidates.shape[0] != center.shape[0]:
        raise ValueError("action arrays and index rows are not aligned")
    for index, row in enumerate(rows):
        if _array_sha256(center[index]) != row["center_action_sha256"]:
            raise ValueError(f"center action hash mismatch at sample {index}")
        actual = [_array_sha256(candidates[index, slot]) for slot in range(CANDIDATE_COUNT)]
        if actual != row["candidate_action_sha256"]:
            raise ValueError(f"candidate action hash mismatch at sample {index}")


def validate_episode_split_bindings(
    audit_arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    *,
    train_count_per_task: int,
    validation_count_per_task: int,
) -> None:
    """Reconstruct the physical-episode identity and split contract.

    Collection-time grouping is not sufficient evidence after serialization.
    The formal reader independently proves the 8/4 episode split, complete
    three-phase coverage, and identity hashes before fitting any scorer.
    """

    train_count = int(train_count_per_task)
    validation_count = int(validation_count_per_task)
    episode_count = train_count + validation_count
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("episode split counts must both be positive")
    n = len(rows)
    required_arrays = {
        "task_index",
        "episode_id",
        "env_seed",
        "snapshot_step",
        "phase_index",
        "split_index",
    }
    if not required_arrays.issubset(audit_arrays):
        raise ValueError("episode split audit arrays are incomplete")
    if any(np.asarray(audit_arrays[name]).shape != (n,) for name in required_arrays):
        raise ValueError("episode split audit arrays must be aligned vectors")

    groups: dict[tuple[int, int, int], list[int]] = {}
    digest_to_identity: dict[str, tuple[int, int, int]] = {}
    snapshot_hashes: set[str] = set()
    for index, row in enumerate(rows):
        task = int(audit_arrays["task_index"][index])
        episode = int(audit_arrays["episode_id"][index])
        env_seed = int(audit_arrays["env_seed"][index])
        phase = int(audit_arrays["phase_index"][index])
        snapshot_step = int(audit_arrays["snapshot_step"][index])
        split_index = int(audit_arrays["split_index"][index])
        if task not in range(len(TASK_NAMES)):
            raise ValueError("episode binding task index is invalid")
        if phase not in range(len(PHASE_NAMES)):
            raise ValueError("episode binding phase index is invalid")
        if snapshot_step != PHASE_STEPS[phase]:
            raise ValueError("phase index is not bound to its frozen snapshot step")
        if row["task"] != TASK_NAMES[task] or row["phase"] != PHASE_NAMES[phase]:
            raise ValueError("index task/phase is not bound to audit arrays")
        if int(row["episode_id"]) != episode or int(row["env_seed"]) != env_seed:
            raise ValueError("index episode/env identity is not bound to audit arrays")
        if int(row["snapshot_step"]) != snapshot_step:
            raise ValueError("index snapshot step is not bound to audit arrays")
        expected_split = "train" if split_index == 0 else "validation"
        if split_index not in (0, 1) or row["split"] != expected_split:
            raise ValueError("index split is not bound to split_index")
        expected_digest = canonical_sha256(
            {
                "task_index": task,
                "episode_id": episode,
                "env_seed": env_seed,
            }
        )
        digest = str(row["physical_episode_sha256"])
        if digest != expected_digest:
            raise ValueError("physical episode SHA-256 does not match its identity")
        identity = (task, episode, env_seed)
        previous = digest_to_identity.setdefault(digest, identity)
        if previous != identity:
            raise ValueError("physical episode SHA-256 is reused by another identity")
        snapshot_digest = str(row["physical_snapshot_sha256"])
        if snapshot_digest in snapshot_hashes:
            raise ValueError("physical snapshot SHA-256 is reused across rows")
        snapshot_hashes.add(snapshot_digest)
        groups.setdefault(identity, []).append(index)

    for task in range(len(TASK_NAMES)):
        identities = sorted(identity for identity in groups if identity[0] == task)
        if len(identities) != episode_count:
            raise ValueError(
                f"{TASK_NAMES[task]} must contain {episode_count} physical episodes"
            )
        if {identity[1] for identity in identities} != set(range(episode_count)):
            raise ValueError("physical episode IDs must be the frozen contiguous range")
        ranked = sorted(
            identities,
            key=lambda identity: canonical_sha256(
                {
                    "task_index": identity[0],
                    "episode_id": identity[1],
                    "env_seed": identity[2],
                }
            ),
        )
        expected_train = set(ranked[:train_count])
        for identity in identities:
            indices = groups[identity]
            if len(indices) != len(PHASE_NAMES):
                raise ValueError("each physical episode must contain exactly three phases")
            phases = [int(audit_arrays["phase_index"][index]) for index in indices]
            if sorted(phases) != list(range(len(PHASE_NAMES))):
                raise ValueError("each physical episode must contain each phase once")
            splits = {int(audit_arrays["split_index"][index]) for index in indices}
            if len(splits) != 1:
                raise ValueError("one physical episode crosses train/validation")
            actual_train = next(iter(splits)) == 0
            if actual_train != (identity in expected_train):
                raise ValueError("physical episode split differs from SHA-sorted assignment")
