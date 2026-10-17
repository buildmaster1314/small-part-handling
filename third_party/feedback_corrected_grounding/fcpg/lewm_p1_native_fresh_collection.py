"""Fresh Pick-only collection for the native P1 complementary-observer Gate.

This module intentionally bypasses the historical A3 bridge.  It samples one
hazard before each environment reset, branches eight actions from the same
physical state, and extracts native frozen-policy observations directly:

    visible history = H0, H1, H2
    offline labels  = H3, H4, H5, H6, H7, H8

Reset identities and exact observed-state signatures are checked against
caller-supplied legacy inventories.  Missing inventory scopes fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zlib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from fcpg.lewm_a4b_task_identity import reset_physical_state_goal_digest
from fcpg.lewm_action_branch_a3 import build_diverse_action_branch_bank
from fcpg.lewm_counterfactual_runtime import (
    FrameRecord,
    capture_branch_snapshot,
    restore_branch_snapshot,
)
from fcpg.lewm_counterfactual_v2_runtime import _close_episode_env_safely
from fcpg.lewm_known_drop_policy_evaluator import extract_policy_anchor_observation
from fcpg.lewm_p1_native_dynamics import (
    P1NativeDynamicsCache,
    P1_NATIVE_ACTION_DIM,
    P1_NATIVE_ANCHORS,
    P1_NATIVE_FUTURE_STEPS,
    P1_NATIVE_HISTORY_SIZE,
    P1_NATIVE_TOKEN_DIM,
    P1_NATIVE_VISUAL_DIM,
)


PROTOCOL: Final = "p1-native-fresh-v1"
SCHEMA_VERSION: Final = 1
TASK: Final = "pick-place-v3"
BENCHMARK_SEED: Final = 41_801
PROTOCOL_SEED: Final = 41_803
ENVIRONMENT_SEED_BASE: Final = 840_000
EPISODE_COUNT: Final = 30
RAW_TASK_INDEX_COUNT: Final = 48
BRANCH_COUNT: Final = 8
RAW_ACTION_DIM: Final = 4
METAWORLD_FULL_OBSERVATION_DIM: Final = 39
RAW_STEPS_PER_MODEL_STEP: Final = 5
TOTAL_MODEL_HORIZONS: Final = 8
TOTAL_RAW_BRANCH_STEPS: Final = TOTAL_MODEL_HORIZONS * RAW_STEPS_PER_MODEL_STEP
VISIBLE_HORIZONS: Final = (0, 1, 2)
LABEL_HORIZONS: Final = (3, 4, 5, 6, 7, 8)
HAZARD_FIRST_DECISION: Final = 3
HAZARD_LAST_DECISION: Final = 31
FRESH_SPLIT: Final = "fresh_confirmation"
LEGACY_TRAINING_RESET_STATUS: Final = "unverified_missing_legacy_reset_metadata"

# Legacy P1 training artifacts do not contain a complete reset-state receipt.
# Known reset inventories may still be supplied and collision checked, but none
# is required and their presence must not be interpreted as complete coverage.
REQUIRED_RESET_SCOPES: Final = ()
REQUIRED_OBSERVED_SIGNATURE_SCOPES: Final = ("p1_fcpg_training",)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _hash_array(digest: Any, label: str, value: object) -> None:
    array = np.ascontiguousarray(np.asarray(value), dtype="<f4")
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite and non-empty")
    digest.update(label.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def observed_state_signature(
    task_name: str,
    observation: object,
    keypoints_xyz: object,
) -> str:
    """Hash one exact full pre-action observation available in legacy P1 data.

    The 39-D observation is used only for the offline freshness receipt.  The
    deployed policy still receives robot-only proprioception.  Requiring the
    full shape prevents a robot-only 4-D state from being compared against the
    legacy archive's 39-D observations in a disjoint hash domain.
    """

    if str(task_name) != TASK:
        raise ValueError(f"fresh P1 collection is fixed to {TASK}")
    observed = np.asarray(observation)
    xyz = np.asarray(keypoints_xyz)
    if observed.shape != (METAWORLD_FULL_OBSERVATION_DIM,):
        raise ValueError(
            "freshness signature requires one full 39-D MetaWorld observation"
        )
    if xyz.shape != (P1_NATIVE_ANCHORS, 3):
        raise ValueError("freshness signature requires three 3-D keypoints")
    digest = hashlib.sha256()
    digest.update(PROTOCOL.encode("ascii"))
    digest.update(b"\0observed-state-v1\0")
    digest.update(str(task_name).encode("utf-8"))
    _hash_array(digest, "obs", observed)
    _hash_array(digest, "keypoints_xyz", xyz)
    return digest.hexdigest()


def legacy_observed_state_signatures(
    source: str | Path,
    *,
    task_name: str = TASK,
) -> frozenset[str]:
    """Build the exact-signature inventory missing from old training receipts."""

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"task_name", "obs", "keypoints_xyz"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"legacy observed-state source is missing {sorted(missing)}")
        tasks = np.asarray(archive["task_name"]).astype(str)
        observation = np.asarray(archive["obs"])
        xyz = np.asarray(archive["keypoints_xyz"])
    if observation.shape[0] != tasks.size or xyz.shape[0] != tasks.size:
        raise ValueError("legacy observed-state arrays have inconsistent row counts")
    selected = np.flatnonzero(tasks == str(task_name))
    if selected.size == 0:
        raise ValueError(f"legacy source contains no rows for {task_name}")
    return frozenset(
        observed_state_signature(task_name, observation[row], xyz[row])
        for row in selected.tolist()
    )


@dataclass(frozen=True, slots=True)
class FreshnessInventory:
    """Legacy identities checked before any fresh endpoint is accepted.

    Exact observed-state signatures are mandatory.  Reset digests are optional
    collision evidence because the legacy training set has no complete reset
    receipt.
    """

    reset_digest_scopes: Mapping[str, Collection[tuple[str, str]]]
    observed_signature_scopes: Mapping[str, Collection[str]]
    legacy_training_reset_digest_status: str = LEGACY_TRAINING_RESET_STATUS
    required_reset_scopes: tuple[str, ...] = REQUIRED_RESET_SCOPES
    required_observed_signature_scopes: tuple[str, ...] = (
        REQUIRED_OBSERVED_SIGNATURE_SCOPES
    )

    def __post_init__(self) -> None:
        if self.legacy_training_reset_digest_status != LEGACY_TRAINING_RESET_STATUS:
            raise ValueError(
                "legacy training reset non-overlap is unverified and must not be claimed"
            )
        reset = self._validate_reset_scopes()
        observed = self._validate_observed_scopes()
        object.__setattr__(self, "reset_digest_scopes", reset)
        object.__setattr__(self, "observed_signature_scopes", observed)

    def _validate_reset_scopes(self) -> dict[str, frozenset[tuple[str, str]]]:
        if not isinstance(self.reset_digest_scopes, Mapping):
            raise TypeError("legacy reset digest scopes must be a mapping")
        if self.required_reset_scopes:
            raise ValueError(
                "legacy reset receipt is incomplete; reset scopes cannot be required"
            )
        result: dict[str, frozenset[tuple[str, str]]] = {}
        for raw_scope, raw_rows in self.reset_digest_scopes.items():
            scope = str(raw_scope)
            if not scope:
                raise ValueError("legacy reset inventory scope must be non-empty")
            rows = frozenset(
                (str(task), str(digest))
                for task, digest in raw_rows
            )
            if any(task != TASK or not _is_sha256(digest) for task, digest in rows):
                raise ValueError(f"legacy reset inventory {scope!r} is invalid")
            result[scope] = rows
        return result

    def _validate_observed_scopes(self) -> dict[str, frozenset[str]]:
        if not isinstance(self.observed_signature_scopes, Mapping):
            raise TypeError("legacy observed signature scopes must be a mapping")
        missing = set(self.required_observed_signature_scopes) - set(
            self.observed_signature_scopes
        )
        if missing:
            raise PermissionError(
                "fresh collection lacks required observed-state inventories: "
                f"{sorted(missing)}"
            )
        result: dict[str, frozenset[str]] = {}
        for scope in self.required_observed_signature_scopes:
            rows = frozenset(str(value) for value in self.observed_signature_scopes[scope])
            if not rows:
                raise PermissionError(
                    f"legacy observed-state inventory {scope!r} is empty"
                )
            if any(not _is_sha256(value) for value in rows):
                raise ValueError(
                    f"legacy observed-state inventory {scope!r} is invalid"
                )
            result[scope] = rows
        return result

    def assert_fresh(
        self,
        *,
        task_name: str,
        reset_digest: str,
        observed_signature: str,
    ) -> None:
        if task_name != TASK or not _is_sha256(reset_digest):
            raise ValueError("fresh reset identity is invalid")
        if not _is_sha256(observed_signature):
            raise ValueError("fresh observed-state signature is invalid")
        reset_key = (task_name, reset_digest)
        collided_reset = [
            scope
            for scope, rows in self.reset_digest_scopes.items()
            if reset_key in rows
        ]
        collided_observed = [
            scope
            for scope, rows in self.observed_signature_scopes.items()
            if observed_signature in rows
        ]
        if collided_reset or collided_observed:
            raise PermissionError(
                "fresh physical identity overlaps legacy data: "
                f"reset={collided_reset}, observed={collided_observed}"
            )


@dataclass(frozen=True, slots=True)
class FreshEpisodePlan:
    candidate_rank: int
    raw_task_index: int
    environment_seed: int
    hazard_id: int
    episode_uid: str

    def __post_init__(self) -> None:
        if not 0 <= self.candidate_rank < EPISODE_COUNT:
            raise ValueError("fresh candidate rank must be in 0..29")
        if not 0 <= self.raw_task_index < RAW_TASK_INDEX_COUNT:
            raise ValueError("fresh raw task index must be in 0..47")
        if self.environment_seed != ENVIRONMENT_SEED_BASE + self.candidate_rank:
            raise ValueError("fresh environment seed differs from frozen rank identity")
        if not HAZARD_FIRST_DECISION <= self.hazard_id <= HAZARD_LAST_DECISION:
            raise ValueError("fresh hazard is outside the frozen decision range")
        expected_uid = fresh_episode_uid(
            candidate_rank=self.candidate_rank,
            raw_task_index=self.raw_task_index,
            environment_seed=self.environment_seed,
        )
        if self.episode_uid != expected_uid:
            raise ValueError("fresh episode UID differs from frozen identity")


def fresh_episode_uid(
    *, candidate_rank: int, raw_task_index: int, environment_seed: int
) -> str:
    return (
        f"{PROTOCOL}:benchmark-{BENCHMARK_SEED}:{TASK}:"
        f"rank-{int(candidate_rank):02d}:raw-{int(raw_task_index):02d}:"
        f"env-{int(environment_seed)}"
    )


def validate_fresh_collection_contract(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate and return the formal ``fresh_data`` config section."""

    if not isinstance(contract, Mapping):
        raise TypeError("fresh collection contract must be a mapping")
    data = contract.get("fresh_data", contract)
    if not isinstance(data, Mapping):
        raise TypeError("fresh_data must be a mapping")
    expected = {
        "protocol": PROTOCOL,
        "benchmark_seed": BENCHMARK_SEED,
        "protocol_seed": PROTOCOL_SEED,
        "environment_seed_base": ENVIRONMENT_SEED_BASE,
        "physical_episode_count": EPISODE_COUNT,
        "one_hazard_per_physical_episode": True,
        "hazard_drawn_before_environment_reset": True,
        "hazard_independent_of_state_action_reward_and_outcome": True,
        "visible_horizons": list(VISIBLE_HORIZONS),
        "offline_label_horizons": list(LABEL_HORIZONS),
        "raw_steps_per_model_step": RAW_STEPS_PER_MODEL_STEP,
        "branch_count": BRANCH_COUNT,
        "physical_identity": "task_name_plus_reset_physical_state_goal_sha256",
    }
    for name, value in expected.items():
        if data.get(name) != value:
            raise ValueError(f"fresh_data.{name} differs from the frozen contract")
    indices = tuple(int(value) for value in data.get("task_instance_indices", ()))
    hazards = tuple(int(value) for value in data.get("hazard_model_steps", ()))
    if len(indices) != EPISODE_COUNT or len(set(indices)) != EPISODE_COUNT:
        raise ValueError("fresh_data must contain exactly 30 unique task instances")
    if any(not 0 <= value < RAW_TASK_INDEX_COUNT for value in indices):
        raise ValueError("fresh task instances must lie in 0..47")
    if len(hazards) != EPISODE_COUNT or any(
        not HAZARD_FIRST_DECISION <= value <= HAZARD_LAST_DECISION
        for value in hazards
    ):
        raise ValueError("fresh_data must contain exactly 30 valid hazards")
    if "freshness" in contract:
        freshness = contract["freshness"]
        if not isinstance(freshness, Mapping):
            raise TypeError("freshness must be a mapping")
        if freshness.get("reset_state_nonoverlap_with_legacy_training_is_not_claimed") is not True:
            raise ValueError("legacy training reset non-overlap must remain unverified")
        if freshness.get("require_zero_exact_observed_state_signature_overlap") is not True:
            raise ValueError("legacy observed-state non-overlap must fail closed")
    return data


def build_fresh_episode_plans(
    contract: Mapping[str, Any],
) -> tuple[FreshEpisodePlan, ...]:
    """Read all formal identities/hazards from config before any reset."""

    data = validate_fresh_collection_contract(contract)
    raw_order = tuple(int(value) for value in data["task_instance_indices"])
    hazards = tuple(int(value) for value in data["hazard_model_steps"])
    plans = tuple(
        FreshEpisodePlan(
            candidate_rank=rank,
            raw_task_index=int(raw_index),
            environment_seed=ENVIRONMENT_SEED_BASE + rank,
            hazard_id=hazards[rank],
            episode_uid=fresh_episode_uid(
                candidate_rank=rank,
                raw_task_index=int(raw_index),
                environment_seed=ENVIRONMENT_SEED_BASE + rank,
            ),
        )
        for rank, raw_index in enumerate(raw_order)
    )
    if len(plans) != EPISODE_COUNT or len({plan.raw_task_index for plan in plans}) != EPISODE_COUNT:
        raise RuntimeError("fresh episode plan is not exactly 30 unique physical candidates")
    return plans


@dataclass(frozen=True, slots=True)
class FreshP1NativeEpisode:
    plan: FreshEpisodePlan
    reset_digest: str
    observed_signature: str
    history_token: np.ndarray
    history_action: np.ndarray
    future_action: np.ndarray
    target_token: np.ndarray
    history_sampled_visual: np.ndarray
    history_xyz: np.ndarray
    future_xyz: np.ndarray

    def __post_init__(self) -> None:
        if not _is_sha256(self.reset_digest) or not _is_sha256(self.observed_signature):
            raise ValueError("fresh row requires valid reset and observed-state digests")
        shapes = {
            "history_token": (BRANCH_COUNT, 3, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM),
            "history_action": (BRANCH_COUNT, 2, P1_NATIVE_ACTION_DIM),
            "future_action": (
                BRANCH_COUNT,
                P1_NATIVE_FUTURE_STEPS,
                P1_NATIVE_ACTION_DIM,
            ),
            "target_token": (
                BRANCH_COUNT,
                P1_NATIVE_FUTURE_STEPS,
                P1_NATIVE_ANCHORS,
                P1_NATIVE_TOKEN_DIM,
            ),
            "history_sampled_visual": (
                BRANCH_COUNT,
                3,
                P1_NATIVE_ANCHORS,
                P1_NATIVE_VISUAL_DIM,
            ),
            "history_xyz": (BRANCH_COUNT, 3, P1_NATIVE_ANCHORS, 3),
            "future_xyz": (
                BRANCH_COUNT,
                P1_NATIVE_FUTURE_STEPS,
                P1_NATIVE_ANCHORS,
                3,
            ),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError(f"fresh {name} must be finite floating point {shape}")


def _policy_pair_batch(
    pairs: Sequence[Sequence[FrameRecord]], *, device: torch.device
) -> dict[str, torch.Tensor]:
    rows = [tuple(pair) for pair in pairs]
    if not rows or any(len(pair) != 2 for pair in rows):
        raise ValueError("native P1 extraction requires raw-adjacent frame pairs")
    mapping = {
        "rgb": "rgb",
        "obs": "proprio",
        "q_nominal": "q_nominal",
        "keypoints_xyz": "keypoints_xyz",
    }
    result: dict[str, torch.Tensor] = {}
    for output, source in mapping.items():
        value = np.stack(
            [np.stack([np.asarray(getattr(record, source)) for record in pair]) for pair in rows]
        ).astype(np.float32, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"P1 pair field {source} contains non-finite values")
        result[output] = torch.from_numpy(np.ascontiguousarray(value)).to(device)
    return result


@torch.inference_mode()
def _extract_pair_endpoints(
    policy: nn.Module,
    pairs: Sequence[Sequence[FrameRecord]],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = extract_policy_anchor_observation(
        policy, _policy_pair_batch(pairs, device=device)
    )
    return tuple(
        tensor[:, -1].float().cpu().numpy().astype(np.float32, copy=False)
        for tensor in (observed.native_token, observed.sampled_visual, observed.xyz)
    )


def _branch_seeds(plan: FreshEpisodePlan) -> tuple[int, int]:
    values = np.random.SeedSequence(
        [
            PROTOCOL_SEED,
            zlib.crc32(TASK.encode("utf-8")),
            plan.candidate_rank,
            plan.raw_task_index,
            plan.hazard_id,
        ]
    ).generate_state(2, dtype=np.uint32)
    return int(values[0]), int(values[1])


def _bounded_expert_action(runtime: Any, observation: np.ndarray) -> np.ndarray:
    action = np.asarray(runtime.expert.get_action(observation), dtype=np.float32)
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if action.shape != (RAW_ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError("fresh Pick expert must return a finite 4D action")
    return np.clip(action, low, high).astype(np.float32, copy=False)


def _reshape_action_blocks(actions: np.ndarray) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float32)
    expected = (BRANCH_COUNT, TOTAL_RAW_BRANCH_STEPS, RAW_ACTION_DIM)
    if value.shape != expected:
        raise ValueError(f"fresh action branches must have shape {expected}")
    return np.ascontiguousarray(
        value.reshape(BRANCH_COUNT, TOTAL_MODEL_HORIZONS, P1_NATIVE_ACTION_DIM)
    )


@torch.inference_mode()
def collect_fresh_p1_native_episode(
    plan: FreshEpisodePlan,
    *,
    inventory: FreshnessInventory,
    policy: nn.Module,
    device: str | torch.device,
    runtime_factory: Callable[[str, int, int], Any],
    branch_bank_builder: Callable[..., Any] = build_diverse_action_branch_bank,
    snapshot_capture: Callable[..., Any] = capture_branch_snapshot,
    snapshot_restore: Callable[[Any, Any], None] = restore_branch_snapshot,
) -> FreshP1NativeEpisode:
    """Collect one native P1 bundle without loading any A3 model or cache."""

    target_device = torch.device(device)
    if int(getattr(policy, "obs_horizon", -1)) != 2:
        raise ValueError("fresh collector requires the frozen raw-adjacent P1 policy")
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("fresh collection requires a frozen eval policy")

    # The caller must provide a fully materialized plan, including its random
    # hazard, before this factory performs environment construction/reset.
    runtime = runtime_factory(TASK, plan.raw_task_index, plan.environment_seed)
    try:
        initial_record = runtime.record_current(
            np.asarray(runtime.observation, dtype=np.float32)
        )
        reset_digest = reset_physical_state_goal_digest(runtime.env)
        signature = observed_state_signature(
            TASK, runtime.observation, initial_record.keypoints_xyz
        )
        inventory.assert_fresh(
            task_name=TASK,
            reset_digest=reset_digest,
            observed_signature=signature,
        )

        hazard_raw = plan.hazard_id * RAW_STEPS_PER_MODEL_STEP
        final_raw = hazard_raw + TOTAL_RAW_BRANCH_STEPS
        needed = {hazard_raw - 1, hazard_raw}
        observation = np.asarray(runtime.observation, dtype=np.float32)
        source_actions: list[np.ndarray] = []
        records: dict[int, FrameRecord] = {}
        snapshot: Any | None = None
        branch_start_observation: np.ndarray | None = None
        for raw_step in range(final_raw + 1):
            if raw_step in needed:
                records[raw_step] = runtime.record_current(observation)
            if raw_step == hazard_raw:
                snapshot = snapshot_capture(runtime.env, include_process_rng=True)
                branch_start_observation = observation.copy()
            if raw_step == final_raw:
                break
            action = _bounded_expert_action(runtime, observation)
            source_actions.append(action.copy())
            observation, _, terminated, truncated, _ = runtime.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            if (terminated or truncated) and raw_step + 1 < final_raw:
                raise RuntimeError("fresh source episode ended before the H8 center")
        if snapshot is None or branch_start_observation is None:
            raise RuntimeError("fresh collector failed to capture the branch snapshot")

        source_action = np.asarray(source_actions, dtype=np.float32)
        center = source_action[hazard_raw:final_raw]
        if center.shape != (TOTAL_RAW_BRANCH_STEPS, RAW_ACTION_DIM):
            raise RuntimeError("fresh source did not provide the exact H1-H8 center")
        direction_seed, permutation_seed = _branch_seeds(plan)
        bank = branch_bank_builder(
            center,
            direction_seed=direction_seed,
            permutation_seed=permutation_seed,
            action_low=runtime.env.action_space.low,
            action_high=runtime.env.action_space.high,
            motion_delta_rms=None,
        )
        branch_actions_raw = np.asarray(bank.actions, dtype=np.float32)
        action_blocks = _reshape_action_blocks(branch_actions_raw)
        h0_token, h0_visual, h0_xyz = _extract_pair_endpoints(
            policy,
            ((records[hazard_raw - 1], records[hazard_raw]),),
            device=target_device,
        )

        endpoint_token: list[np.ndarray] = []
        endpoint_visual: list[np.ndarray] = []
        endpoint_xyz: list[np.ndarray] = []
        for branch in branch_actions_raw:
            snapshot_restore(runtime.env, snapshot)
            branch_observation = branch_start_observation.copy()
            raw_records: list[FrameRecord] = []
            endpoint_pairs: list[tuple[FrameRecord, FrameRecord]] = []
            for raw_index, action in enumerate(branch, start=1):
                branch_observation, _, terminated, truncated, _ = runtime.env.step(action)
                branch_observation = np.asarray(branch_observation, dtype=np.float32)
                raw_records.append(runtime.record_current(branch_observation))
                if raw_index % RAW_STEPS_PER_MODEL_STEP == 0:
                    endpoint_pairs.append((raw_records[-2], raw_records[-1]))
                if (terminated or truncated) and raw_index < TOTAL_RAW_BRANCH_STEPS:
                    raise RuntimeError("fresh branch ended before H8")
            if len(endpoint_pairs) != TOTAL_MODEL_HORIZONS:
                raise RuntimeError("fresh branch did not produce all H1-H8 endpoints")
            token, visual, xyz = _extract_pair_endpoints(
                policy, endpoint_pairs, device=target_device
            )
            endpoint_token.append(token)
            endpoint_visual.append(visual)
            endpoint_xyz.append(xyz)

        token = np.stack(endpoint_token)
        visual = np.stack(endpoint_visual)
        xyz = np.stack(endpoint_xyz)
        repeat_shape = (BRANCH_COUNT, *h0_token.shape[1:])
        h0_token_branch = np.broadcast_to(h0_token[0], repeat_shape)
        h0_visual_branch = np.broadcast_to(h0_visual[0], repeat_shape)
        h0_xyz_branch = np.broadcast_to(
            h0_xyz[0], (BRANCH_COUNT, P1_NATIVE_ANCHORS, 3)
        )
        return FreshP1NativeEpisode(
            plan=plan,
            reset_digest=reset_digest,
            observed_signature=signature,
            history_token=np.stack((h0_token_branch, token[:, 0], token[:, 1]), axis=1),
            history_action=action_blocks[:, :2],
            future_action=action_blocks[:, 2:],
            target_token=token[:, 2:],
            history_sampled_visual=np.stack(
                (h0_visual_branch, visual[:, 0], visual[:, 1]), axis=1
            ),
            history_xyz=np.stack((h0_xyz_branch, xyz[:, 0], xyz[:, 1]), axis=1),
            future_xyz=xyz[:, 2:],
        )
    finally:
        _close_episode_env_safely(runtime.env)


def _wrong_future_actions(future: np.ndarray) -> np.ndarray:
    slots = np.arange(BRANCH_COUNT)
    return np.ascontiguousarray(
        np.stack([future[:, (slots + shift) % BRANCH_COUNT] for shift in range(1, 8)], axis=1),
        dtype=np.float32,
    )


FRESH_CACHE_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol",
        "benchmark_seed",
        "protocol_seed",
        "environment_seed_base",
        "task_name",
        "episode_uid",
        "physical_episode_id",
        "reset_physical_state_goal_digest",
        "observed_state_signature",
        "raw_task_index",
        "environment_seed",
        "hazard_id",
        "branch_index",
        "visible_horizons",
        "label_horizons",
        "history_sensor_valid",
        "future_sensor_valid",
        "target_is_offline_label",
        "legacy_training_reset_digest_status",
        "history_token",
        "history_action",
        "future_action",
        "wrong_future_action",
        "target_token",
        "history_sampled_visual",
        "history_xyz",
        "future_xyz",
    }
)

IDENTITY_CACHE_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol",
        "benchmark_seed",
        "protocol_seed",
        "environment_seed_base",
        "task_name",
        "episode_uid",
        "physical_episode_id",
        "reset_physical_state_goal_digest",
        "observed_state_signature",
        "raw_task_index",
        "environment_seed",
        "hazard_id",
        "branch_index",
        "visible_horizons",
        "label_horizons",
        "legacy_training_reset_digest_status",
    }
)
DEPLOYMENT_CACHE_KEYS: Final = IDENTITY_CACHE_KEYS | frozenset(
    {
        "history_sensor_valid",
        "future_sensor_valid",
        "history_token",
        "history_action",
        "future_action",
        "wrong_future_action",
        "history_sampled_visual",
        "history_xyz",
        "future_xyz",
    }
)
OFFLINE_LABEL_CACHE_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol",
        "physical_episode_id",
        "branch_index",
        "label_horizons",
        "target_is_offline_label",
        "target_token",
    }
)


def stack_fresh_p1_native_episodes(
    rows: Sequence[FreshP1NativeEpisode],
    *,
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    values = list(rows)
    if len(values) != EPISODE_COUNT:
        raise ValueError("formal fresh cache requires exactly 30 physical episodes")
    expected_plans = build_fresh_episode_plans(contract)
    if tuple(row.plan for row in values) != expected_plans:
        raise PermissionError("fresh rows differ from the frozen plan or order")
    if len({row.reset_digest for row in values}) != EPISODE_COUNT:
        raise PermissionError("fresh rows contain repeated physical reset identities")
    if len({row.observed_signature for row in values}) != EPISODE_COUNT:
        raise PermissionError("fresh rows contain repeated observed-state signatures")

    history_token = np.stack([row.history_token for row in values]).astype(np.float32)
    history_action = np.stack([row.history_action for row in values]).astype(np.float32)
    future_action = np.stack([row.future_action for row in values]).astype(np.float32)
    arrays = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
        "protocol": np.asarray([PROTOCOL]),
        "benchmark_seed": np.asarray([BENCHMARK_SEED], dtype=np.int64),
        "protocol_seed": np.asarray([PROTOCOL_SEED], dtype=np.int64),
        "environment_seed_base": np.asarray([ENVIRONMENT_SEED_BASE], dtype=np.int64),
        "task_name": np.asarray([TASK] * EPISODE_COUNT),
        "episode_uid": np.asarray([row.plan.episode_uid for row in values]),
        "physical_episode_id": np.asarray(
            [f"{TASK}:{row.reset_digest}" for row in values]
        ),
        "reset_physical_state_goal_digest": np.asarray(
            [row.reset_digest for row in values]
        ),
        "observed_state_signature": np.asarray(
            [row.observed_signature for row in values]
        ),
        "raw_task_index": np.asarray(
            [row.plan.raw_task_index for row in values], dtype=np.int64
        ),
        "environment_seed": np.asarray(
            [row.plan.environment_seed for row in values], dtype=np.int64
        ),
        "hazard_id": np.asarray(
            [row.plan.hazard_id for row in values], dtype=np.int64
        ),
        "branch_index": np.arange(BRANCH_COUNT, dtype=np.int64),
        "visible_horizons": np.asarray(VISIBLE_HORIZONS, dtype=np.int64),
        "label_horizons": np.asarray(LABEL_HORIZONS, dtype=np.int64),
        "history_sensor_valid": np.ones(
            (EPISODE_COUNT, BRANCH_COUNT, len(VISIBLE_HORIZONS)), dtype=bool
        ),
        "future_sensor_valid": np.zeros(
            (EPISODE_COUNT, BRANCH_COUNT, len(LABEL_HORIZONS)), dtype=bool
        ),
        "target_is_offline_label": np.asarray([True], dtype=bool),
        "legacy_training_reset_digest_status": np.asarray(
            [LEGACY_TRAINING_RESET_STATUS]
        ),
        "history_token": history_token,
        "history_action": history_action,
        "future_action": future_action,
        "wrong_future_action": _wrong_future_actions(future_action),
        "target_token": np.stack([row.target_token for row in values]).astype(np.float32),
        "history_sampled_visual": np.stack(
            [row.history_sampled_visual for row in values]
        ).astype(np.float32),
        "history_xyz": np.stack([row.history_xyz for row in values]).astype(np.float32),
        "future_xyz": np.stack([row.future_xyz for row in values]).astype(np.float32),
    }
    validate_fresh_p1_native_arrays(arrays, contract=contract)
    return arrays


def validate_fresh_p1_native_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
) -> None:
    if set(arrays) != FRESH_CACHE_KEYS:
        raise ValueError(f"fresh P1 cache keys drifted: {sorted(set(arrays) ^ FRESH_CACHE_KEYS)}")
    scalar_values = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "benchmark_seed": BENCHMARK_SEED,
        "protocol_seed": PROTOCOL_SEED,
        "environment_seed_base": ENVIRONMENT_SEED_BASE,
        "target_is_offline_label": True,
        "legacy_training_reset_digest_status": LEGACY_TRAINING_RESET_STATUS,
    }
    for name, expected in scalar_values.items():
        value = np.asarray(arrays[name])
        if value.shape != (1,) or value[0] != expected:
            raise ValueError(f"fresh P1 cache {name} drifted")
    n = EPISODE_COUNT
    expected_shapes = {
        "task_name": (n,),
        "episode_uid": (n,),
        "physical_episode_id": (n,),
        "reset_physical_state_goal_digest": (n,),
        "observed_state_signature": (n,),
        "raw_task_index": (n,),
        "environment_seed": (n,),
        "hazard_id": (n,),
        "branch_index": (BRANCH_COUNT,),
        "visible_horizons": (3,),
        "label_horizons": (6,),
        "history_sensor_valid": (n, BRANCH_COUNT, 3),
        "future_sensor_valid": (n, BRANCH_COUNT, 6),
        "history_token": (n, BRANCH_COUNT, 3, 3, 256),
        "history_action": (n, BRANCH_COUNT, 2, 20),
        "future_action": (n, BRANCH_COUNT, 6, 20),
        "wrong_future_action": (n, 7, BRANCH_COUNT, 6, 20),
        "target_token": (n, BRANCH_COUNT, 6, 3, 256),
        "history_sampled_visual": (n, BRANCH_COUNT, 3, 3, 256),
        "history_xyz": (n, BRANCH_COUNT, 3, 3, 3),
        "future_xyz": (n, BRANCH_COUNT, 6, 3, 3),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"fresh P1 {name} must have shape {shape}, got {value.shape}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"fresh P1 {name} contains non-finite values")
    if not np.all(np.asarray(arrays["task_name"]).astype(str) == TASK):
        raise ValueError("fresh P1 cache must contain Pick only")
    if not np.array_equal(arrays["branch_index"], np.arange(BRANCH_COUNT)):
        raise ValueError("fresh P1 branch order must be exactly 0..7")
    if not np.array_equal(arrays["visible_horizons"], VISIBLE_HORIZONS):
        raise ValueError("fresh P1 visible horizons must be H0-H2")
    if not np.array_equal(arrays["label_horizons"], LABEL_HORIZONS):
        raise ValueError("fresh P1 labels must be H3-H8")
    if not bool(np.all(arrays["history_sensor_valid"])) or bool(
        np.any(arrays["future_sensor_valid"])
    ):
        raise ValueError("fresh P1 sensor routing must expose H0-H2 only")
    plans = build_fresh_episode_plans(contract)
    if np.asarray(arrays["episode_uid"]).astype(str).tolist() != [
        plan.episode_uid for plan in plans
    ]:
        raise ValueError("fresh P1 episode identities differ from the frozen panel")
    if np.asarray(arrays["hazard_id"], dtype=np.int64).tolist() != [
        plan.hazard_id for plan in plans
    ]:
        raise ValueError("fresh P1 hazards differ from the pre-reset plan")
    for name in ("reset_physical_state_goal_digest", "observed_state_signature"):
        values = np.asarray(arrays[name]).astype(str).tolist()
        if len(set(values)) != n or any(not _is_sha256(value) for value in values):
            raise PermissionError(f"fresh P1 {name} must contain 30 unique SHA-256 values")
    expected_physical = [
        f"{TASK}:{digest}"
        for digest in np.asarray(
            arrays["reset_physical_state_goal_digest"]
        ).astype(str)
    ]
    if np.asarray(arrays["physical_episode_id"]).astype(str).tolist() != expected_physical:
        raise ValueError("physical_episode_id must be task plus reset-state/goal digest")


def validate_fresh_deployment_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
) -> None:
    """Validate the independently loadable, deployable-input archive."""

    if set(arrays) != DEPLOYMENT_CACHE_KEYS:
        raise ValueError(
            "fresh deployment cache keys drifted: "
            f"{sorted(set(arrays) ^ DEPLOYMENT_CACHE_KEYS)}"
        )
    scalar_values = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "benchmark_seed": BENCHMARK_SEED,
        "protocol_seed": PROTOCOL_SEED,
        "environment_seed_base": ENVIRONMENT_SEED_BASE,
        "legacy_training_reset_digest_status": LEGACY_TRAINING_RESET_STATUS,
    }
    for name, expected in scalar_values.items():
        value = np.asarray(arrays[name])
        if value.shape != (1,) or value[0] != expected:
            raise ValueError(f"fresh deployment cache {name} drifted")

    n = EPISODE_COUNT
    expected_shapes = {
        "task_name": (n,),
        "episode_uid": (n,),
        "physical_episode_id": (n,),
        "reset_physical_state_goal_digest": (n,),
        "observed_state_signature": (n,),
        "raw_task_index": (n,),
        "environment_seed": (n,),
        "hazard_id": (n,),
        "branch_index": (BRANCH_COUNT,),
        "visible_horizons": (len(VISIBLE_HORIZONS),),
        "label_horizons": (len(LABEL_HORIZONS),),
        "history_sensor_valid": (n, BRANCH_COUNT, len(VISIBLE_HORIZONS)),
        "future_sensor_valid": (n, BRANCH_COUNT, len(LABEL_HORIZONS)),
        "history_token": (n, BRANCH_COUNT, 3, 3, 256),
        "history_action": (n, BRANCH_COUNT, 2, 20),
        "future_action": (n, BRANCH_COUNT, 6, 20),
        "wrong_future_action": (n, 7, BRANCH_COUNT, 6, 20),
        "history_sampled_visual": (n, BRANCH_COUNT, 3, 3, 256),
        "history_xyz": (n, BRANCH_COUNT, 3, 3, 3),
        "future_xyz": (n, BRANCH_COUNT, 6, 3, 3),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(
                f"fresh deployment {name} must have shape {shape}, got {value.shape}"
            )
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"fresh deployment {name} contains non-finite values")

    plans = build_fresh_episode_plans(contract)
    expected_plan_values = {
        "episode_uid": [plan.episode_uid for plan in plans],
        "raw_task_index": [plan.raw_task_index for plan in plans],
        "environment_seed": [plan.environment_seed for plan in plans],
        "hazard_id": [plan.hazard_id for plan in plans],
    }
    for name, expected in expected_plan_values.items():
        actual = np.asarray(arrays[name])
        if actual.dtype.kind in "SUO":
            actual_values = actual.astype(str).tolist()
        else:
            actual_values = actual.astype(np.int64).tolist()
        if actual_values != expected:
            raise ValueError(f"fresh deployment {name} differs from the frozen plan")
    if not np.all(np.asarray(arrays["task_name"]).astype(str) == TASK):
        raise ValueError("fresh deployment cache must contain Pick only")
    if not np.array_equal(arrays["branch_index"], np.arange(BRANCH_COUNT)):
        raise ValueError("fresh deployment branch order must be exactly 0..7")
    if not np.array_equal(arrays["visible_horizons"], VISIBLE_HORIZONS):
        raise ValueError("fresh deployment visible horizons must be H0-H2")
    if not np.array_equal(arrays["label_horizons"], LABEL_HORIZONS):
        raise ValueError("fresh deployment label horizons must be H3-H8")
    if not bool(np.all(arrays["history_sensor_valid"])) or bool(
        np.any(arrays["future_sensor_valid"])
    ):
        raise ValueError("fresh deployment sensor routing must expose H0-H2 only")
    for name in ("reset_physical_state_goal_digest", "observed_state_signature"):
        values = np.asarray(arrays[name]).astype(str).tolist()
        if len(set(values)) != n or any(not _is_sha256(value) for value in values):
            raise PermissionError(
                f"fresh deployment {name} must contain 30 unique SHA-256 values"
            )
    reset_digests = np.asarray(
        arrays["reset_physical_state_goal_digest"]
    ).astype(str)
    expected_physical = [f"{TASK}:{digest}" for digest in reset_digests]
    if np.asarray(arrays["physical_episode_id"]).astype(str).tolist() != expected_physical:
        raise ValueError("deployment physical_episode_id must bind the reset digest")


def validate_fresh_offline_label_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
) -> None:
    """Validate hidden labels without requiring any deployment input tensor."""

    validate_fresh_collection_contract(contract)
    if set(arrays) != OFFLINE_LABEL_CACHE_KEYS:
        raise ValueError(
            "fresh offline-label cache keys drifted: "
            f"{sorted(set(arrays) ^ OFFLINE_LABEL_CACHE_KEYS)}"
        )
    scalar_values = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "target_is_offline_label": True,
    }
    for name, expected in scalar_values.items():
        value = np.asarray(arrays[name])
        if value.shape != (1,) or value[0] != expected:
            raise ValueError(f"fresh offline-label cache {name} drifted")
    expected_shapes = {
        "physical_episode_id": (EPISODE_COUNT,),
        "branch_index": (BRANCH_COUNT,),
        "label_horizons": (len(LABEL_HORIZONS),),
        "target_token": (
            EPISODE_COUNT,
            BRANCH_COUNT,
            len(LABEL_HORIZONS),
            P1_NATIVE_ANCHORS,
            P1_NATIVE_TOKEN_DIM,
        ),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(
                f"fresh offline label {name} must have shape {shape}, got {value.shape}"
            )
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"fresh offline label {name} contains non-finite values")
    physical = np.asarray(arrays["physical_episode_id"]).astype(str).tolist()
    if len(set(physical)) != EPISODE_COUNT or any(
        not value.startswith(f"{TASK}:") or not _is_sha256(value.split(":", 1)[1])
        for value in physical
    ):
        raise PermissionError("fresh offline labels require 30 unique physical identities")
    if not np.array_equal(arrays["branch_index"], np.arange(BRANCH_COUNT)):
        raise ValueError("fresh offline-label branch order must be exactly 0..7")
    if not np.array_equal(arrays["label_horizons"], LABEL_HORIZONS):
        raise ValueError("fresh offline-label horizons must be H3-H8")


def combine_fresh_cache_arrays(
    deployment: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Join separately validated inputs and labels with strict identity binding."""

    validate_fresh_deployment_arrays(deployment, contract=contract)
    validate_fresh_offline_label_arrays(labels, contract=contract)
    shared = {
        "schema_version",
        "protocol",
        "physical_episode_id",
        "branch_index",
        "label_horizons",
    }
    for name in shared:
        if not np.array_equal(np.asarray(deployment[name]), np.asarray(labels[name])):
            raise PermissionError(
                f"fresh deployment/offline-label identity mismatch in {name}"
            )
    combined = {name: np.asarray(value).copy() for name, value in deployment.items()}
    combined["target_is_offline_label"] = np.asarray(
        labels["target_is_offline_label"]
    ).copy()
    combined["target_token"] = np.asarray(labels["target_token"]).copy()
    validate_fresh_p1_native_arrays(combined, contract=contract)
    return combined


def split_fresh_cache_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Separate deployment inputs from hidden labels before writing archives."""

    validate_fresh_p1_native_arrays(arrays, contract=contract)
    deployment = {
        name: np.asarray(arrays[name]).copy() for name in DEPLOYMENT_CACHE_KEYS
    }
    labels = {
        name: np.asarray(arrays[name]).copy() for name in OFFLINE_LABEL_CACHE_KEYS
    }
    if "target_token" in deployment or "history_token" in labels:
        raise RuntimeError("fresh deployment/label cache separation failed")
    validate_fresh_deployment_arrays(deployment, contract=contract)
    validate_fresh_offline_label_arrays(labels, contract=contract)
    return deployment, labels


def fresh_arrays_to_native_cache(
    arrays: Mapping[str, np.ndarray],
    *,
    source_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any],
) -> P1NativeDynamicsCache:
    """Convert the direct fresh schema without reconstructing an A3 bridge."""

    validate_fresh_p1_native_arrays(arrays, contract=contract)
    n = EPISODE_COUNT
    episode = np.broadcast_to(
        np.asarray(arrays["physical_episode_id"]).astype(str)[:, None],
        (n, BRANCH_COUNT),
    )
    task = np.full((n, BRANCH_COUNT), TASK, dtype="U32")
    split = np.full((n, BRANCH_COUNT), FRESH_SPLIT, dtype="U32")
    source_row = np.broadcast_to(
        np.arange(n, dtype=np.int64)[:, None], (n, BRANCH_COUNT)
    )
    branch = np.broadcast_to(
        np.arange(BRANCH_COUNT, dtype=np.int64)[None], (n, BRANCH_COUNT)
    )
    hazard = np.broadcast_to(
        np.asarray(arrays["hazard_id"], dtype=np.int64)[:, None], (n, BRANCH_COUNT)
    )
    return P1NativeDynamicsCache(
        source_path=(
            Path(source_path).expanduser().resolve() if source_path is not None else None
        ),
        metadata=dict(metadata) if metadata is not None else None,
        history_token=np.ascontiguousarray(arrays["history_token"], dtype=np.float32),
        history_action=np.ascontiguousarray(arrays["history_action"], dtype=np.float32),
        future_action=np.ascontiguousarray(arrays["future_action"], dtype=np.float32),
        wrong_future_action=np.ascontiguousarray(
            arrays["wrong_future_action"], dtype=np.float32
        ),
        target_token=np.ascontiguousarray(arrays["target_token"], dtype=np.float32),
        history_sampled_visual=np.ascontiguousarray(
            arrays["history_sampled_visual"], dtype=np.float32
        ),
        history_xyz=np.ascontiguousarray(arrays["history_xyz"], dtype=np.float32),
        future_xyz=np.ascontiguousarray(arrays["future_xyz"], dtype=np.float32),
        source_row_index=np.ascontiguousarray(source_row),
        branch_index=np.ascontiguousarray(branch),
        task_name=np.ascontiguousarray(task),
        physical_episode_id=np.ascontiguousarray(episode),
        hazard_id=np.ascontiguousarray(hazard),
        split=np.ascontiguousarray(split),
    )


def cache_contract_json(contract: Mapping[str, Any]) -> str:
    """Return a stable compact contract suitable for receipt hashing."""

    data = validate_fresh_collection_contract(contract)
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "task": TASK,
            "benchmark_seed": BENCHMARK_SEED,
            "protocol_seed": PROTOCOL_SEED,
            "environment_seed_base": ENVIRONMENT_SEED_BASE,
            "episode_count": EPISODE_COUNT,
            "branch_count": BRANCH_COUNT,
            "visible_horizons": list(VISIBLE_HORIZONS),
            "label_horizons": list(LABEL_HORIZONS),
            "task_instance_indices": [
                int(value) for value in data["task_instance_indices"]
            ],
            "hazard_model_steps": [
                int(value) for value in data["hazard_model_steps"]
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: str | Path) -> str:
    """Hash one provenance artifact without loading it into memory."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_repository_root(config_path: Path) -> Path:
    for parent in config_path.resolve().parents:
        if parent.name == "config":
            return parent.parent
    raise ValueError("fresh config path must be located below a config directory")


def _resolve_contract_artifact(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"fresh contract {name} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _write_npz_temp(
    arrays: Mapping[str, np.ndarray], destination: Path
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def save_fresh_cache_bundle(
    arrays: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
    config_path: str | Path,
    inventory: FreshnessInventory,
) -> dict[str, Any]:
    """Atomically publish split caches and a completion/provenance receipt.

    The deployment and offline-label archives remain physically separate.  A
    consumer must explicitly call :func:`combine_fresh_cache_arrays` before
    labels can enter an evaluator.
    """

    validate_fresh_p1_native_arrays(arrays, contract=contract)
    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    root = _contract_repository_root(config)
    fresh_data = validate_fresh_collection_contract(contract)
    freshness = contract.get("freshness")
    p1 = contract.get("p1")
    if not isinstance(freshness, Mapping) or not isinstance(p1, Mapping):
        raise TypeError("formal fresh contract requires freshness and p1 sections")

    deployment_path = _resolve_contract_artifact(
        root, fresh_data.get("deployment_cache"), name="fresh_data.deployment_cache"
    )
    label_path = _resolve_contract_artifact(
        root, fresh_data.get("label_cache"), name="fresh_data.label_cache"
    )
    metadata_path = _resolve_contract_artifact(
        root, fresh_data.get("metadata"), name="fresh_data.metadata"
    )
    inventory_path = _resolve_contract_artifact(
        root,
        freshness.get("legacy_observed_state_inventory"),
        name="freshness.legacy_observed_state_inventory",
    )
    policy_path = _resolve_contract_artifact(
        root, p1.get("policy_checkpoint"), name="p1.policy_checkpoint"
    )
    corrector_path = _resolve_contract_artifact(
        root, p1.get("corrector_checkpoint"), name="p1.corrector_checkpoint"
    )
    legacy_source_path = _resolve_contract_artifact(
        root,
        freshness.get("legacy_observation_source"),
        name="freshness.legacy_observation_source",
    )
    destinations = (deployment_path, label_path, metadata_path)
    if len(set(destinations)) != len(destinations):
        raise ValueError("fresh deployment, label, and metadata paths must be distinct")
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"fresh cache bundle already exists: {existing}")

    artifact_specs = (
        ("policy checkpoint", policy_path, p1.get("policy_checkpoint_sha256")),
        (
            "corrector checkpoint",
            corrector_path,
            p1.get("corrector_checkpoint_sha256"),
        ),
        (
            "legacy observation source",
            legacy_source_path,
            freshness.get("legacy_observation_source_sha256"),
        ),
    )
    artifact_hashes: dict[str, str] = {}
    for name, path, expected in artifact_specs:
        if not _is_sha256(expected):
            raise ValueError(f"fresh contract has invalid {name} SHA-256")
        actual = sha256_file(path)
        if actual != expected:
            raise PermissionError(f"fresh {name} SHA-256 mismatch")
        artifact_hashes[name] = actual
    inventory_sha = sha256_file(inventory_path)
    required_observed = inventory.observed_signature_scopes[
        REQUIRED_OBSERVED_SIGNATURE_SCOPES[0]
    ]
    derived_observed = legacy_observed_state_signatures(legacy_source_path)
    if required_observed != derived_observed:
        raise PermissionError(
            "fresh observed-state inventory does not match the verified legacy source"
        )
    with np.load(inventory_path, allow_pickle=False) as archive:
        if set(archive.files) != {"observed_state_signature"}:
            raise ValueError(
                "legacy observed-state inventory artifact must contain only "
                "observed_state_signature"
            )
        artifact_observed = frozenset(
            np.asarray(archive["observed_state_signature"]).astype(str).tolist()
        )
    if artifact_observed != required_observed:
        raise PermissionError(
            "legacy observed-state inventory artifact differs from the in-memory receipt"
        )

    reset_digests = np.asarray(
        arrays["reset_physical_state_goal_digest"]
    ).astype(str)
    signatures = np.asarray(arrays["observed_state_signature"]).astype(str)
    for reset_digest, signature in zip(reset_digests, signatures, strict=True):
        inventory.assert_fresh(
            task_name=TASK,
            reset_digest=str(reset_digest),
            observed_signature=str(signature),
        )

    deployment, labels = split_fresh_cache_arrays(arrays, contract=contract)
    deployment_temp: Path | None = None
    label_temp: Path | None = None
    metadata_temp: Path | None = None
    try:
        deployment_temp = _write_npz_temp(deployment, deployment_path)
        label_temp = _write_npz_temp(labels, label_path)
        with np.load(deployment_temp, allow_pickle=False) as archive:
            validate_fresh_deployment_arrays(
                {name: archive[name] for name in archive.files}, contract=contract
            )
        with np.load(label_temp, allow_pickle=False) as archive:
            validate_fresh_offline_label_arrays(
                {name: archive[name] for name in archive.files}, contract=contract
            )
        deployment_sha = sha256_file(deployment_temp)
        label_sha = sha256_file(label_temp)
        contract_json = cache_contract_json(contract)
        dataset_sha = hashlib.sha256(
            (
                contract_json
                + "\0deployment\0"
                + deployment_sha
                + "\0offline-labels\0"
                + label_sha
            ).encode("utf-8")
        ).hexdigest()
        observed_scope_counts = {
            scope: len(rows)
            for scope, rows in sorted(inventory.observed_signature_scopes.items())
        }
        known_reset_scope_counts = {
            scope: len(rows)
            for scope, rows in sorted(inventory.reset_digest_scopes.items())
        }
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "completed": True,
            "config_sha256": sha256_file(config),
            "dataset_sha256": dataset_sha,
            "cache_contract_json": contract_json,
            "deployment_cache": {
                "path": str(fresh_data["deployment_cache"]),
                "sha256": deployment_sha,
            },
            "label_cache": {
                "path": str(fresh_data["label_cache"]),
                "sha256": label_sha,
            },
            "policy_checkpoint": {
                "path": str(p1["policy_checkpoint"]),
                "sha256": artifact_hashes["policy checkpoint"],
            },
            "corrector_checkpoint": {
                "path": str(p1["corrector_checkpoint"]),
                "sha256": artifact_hashes["corrector checkpoint"],
            },
            "legacy_observation_source": {
                "path": str(freshness["legacy_observation_source"]),
                "sha256": artifact_hashes["legacy observation source"],
            },
            "freshness_receipt": {
                "legacy_training_reset_digest_status": (
                    inventory.legacy_training_reset_digest_status
                ),
                "reset_digest_scope_completeness_claimed": False,
                "known_optional_reset_scope_counts": known_reset_scope_counts,
                "observed_signature_scope_counts": observed_scope_counts,
                "legacy_observed_state_inventory": {
                    "path": str(freshness["legacy_observed_state_inventory"]),
                    "sha256": inventory_sha,
                    "signature_count": sum(observed_scope_counts.values()),
                },
                "fresh_observed_signature_count": int(signatures.size),
                "exact_observed_signature_overlap_count": 0,
            },
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, metadata_name = tempfile.mkstemp(
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            dir=metadata_path.parent,
        )
        metadata_temp = Path(metadata_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(deployment_temp, deployment_path)
        deployment_temp = None
        os.replace(label_temp, label_path)
        label_temp = None
        os.replace(metadata_temp, metadata_path)
        metadata_temp = None
        return metadata
    finally:
        for temporary in (deployment_temp, label_temp, metadata_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = [
    "BENCHMARK_SEED",
    "BRANCH_COUNT",
    "DEPLOYMENT_CACHE_KEYS",
    "ENVIRONMENT_SEED_BASE",
    "EPISODE_COUNT",
    "FRESH_CACHE_KEYS",
    "FreshEpisodePlan",
    "FreshP1NativeEpisode",
    "FreshnessInventory",
    "LABEL_HORIZONS",
    "LEGACY_TRAINING_RESET_STATUS",
    "OFFLINE_LABEL_CACHE_KEYS",
    "PROTOCOL",
    "PROTOCOL_SEED",
    "REQUIRED_OBSERVED_SIGNATURE_SCOPES",
    "REQUIRED_RESET_SCOPES",
    "TASK",
    "VISIBLE_HORIZONS",
    "build_fresh_episode_plans",
    "cache_contract_json",
    "combine_fresh_cache_arrays",
    "collect_fresh_p1_native_episode",
    "fresh_arrays_to_native_cache",
    "fresh_episode_uid",
    "legacy_observed_state_signatures",
    "observed_state_signature",
    "save_fresh_cache_bundle",
    "sha256_file",
    "stack_fresh_p1_native_episodes",
    "split_fresh_cache_arrays",
    "validate_fresh_collection_contract",
    "validate_fresh_deployment_arrays",
    "validate_fresh_offline_label_arrays",
    "validate_fresh_p1_native_arrays",
]
