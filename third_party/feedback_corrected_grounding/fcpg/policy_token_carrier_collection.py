"""Fresh compact policy carriers for exposed anchor-sensitivity diagnostics.

The historical P1 fresh bundle stores anchor dynamics but not the frozen
policy's complete two-frame carrier.  This module recollects the same frozen
physical episode/branch plan into *new* artifacts.  Deployment arrays contain
only quantities available to the frozen policy; clean native anchor tokens are
kept in a separate offline-label archive.

This is deliberately an exposed diagnostic.  Healthy-image global and local
features are retained while anchor tokens are overridden, so the artifact
measures the policy consequence of anchor error in isolation.  It is not a
claim that those image features remain available during a real sensor outage.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
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
from fcpg.lewm_known_drop_policy import grounded_tokens_from_memory
from fcpg.lewm_p1_native_dynamics import (
    P1_NATIVE_ACTION_DIM,
    P1_NATIVE_ANCHORS,
    P1_NATIVE_FUTURE_STEPS,
    P1_NATIVE_TOKEN_DIM,
    P1_NATIVE_VISUAL_DIM,
)
from fcpg.lewm_p1_native_fresh_collection import (
    BRANCH_COUNT,
    EPISODE_COUNT,
    LABEL_HORIZONS,
    RAW_ACTION_DIM,
    RAW_STEPS_PER_MODEL_STEP,
    TASK,
    TOTAL_MODEL_HORIZONS,
    TOTAL_RAW_BRANCH_STEPS,
    FreshEpisodePlan,
    FreshnessInventory,
    _bounded_expert_action,
    _branch_seeds,
    _policy_pair_batch,
    _reshape_action_blocks,
    build_fresh_episode_plans,
    observed_state_signature,
)
from fcpg.modules import sample_feature_map_at_pixels


PROTOCOL: Final = "policy-token-carrier-exposed-p0-v2-common-prefix"
SCHEMA_VERSION: Final = 1
CLASSIFICATION: Final = "exposed_development_diagnostic"
POLICY_HISTORY: Final = 2
PREDICTION_ALIGNMENT: Final = "repeat_model_step_endpoint_in_two_raw_adjacent_slots"
PRE_DROP_HISTORY_HORIZONS: Final = (-1, 0, 1)
PRE_DROP_TARGET_HORIZON: Final = 2
SAME_STATE_PREFIX_MODEL_STEPS: Final = 2

FORBIDDEN_DEPLOYMENT_KEYS: Final = frozenset(
    {
        "q_true",
        "valid_true",
        "target_token",
        "policy_target_anchor",
        "canonical_target",
        "reward",
        "return",
        "success",
    }
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _row_id(episode_uid: str, branch: int, horizon: int) -> str:
    payload = f"{PROTOCOL}\0{episode_uid}\0branch={branch}\0H={horizon}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CompactPolicyPairs:
    obs: np.ndarray
    q_nominal: np.ndarray
    keypoints_xyz: np.ndarray
    sampled_visual: np.ndarray
    global_visual: np.ndarray
    image_size: np.ndarray
    fcpg_q_corrected: np.ndarray
    fcpg_reliability: np.ndarray
    native_anchor: np.ndarray

    def __post_init__(self) -> None:
        count = int(np.asarray(self.obs).shape[0])
        if count <= 0:
            raise ValueError("compact policy pair batch must be non-empty")
        obs_dim = int(np.asarray(self.obs).shape[-1])
        shapes = {
            "obs": (count, POLICY_HISTORY, obs_dim),
            "q_nominal": (count, POLICY_HISTORY, P1_NATIVE_ANCHORS, 2),
            "keypoints_xyz": (count, POLICY_HISTORY, P1_NATIVE_ANCHORS, 3),
            "sampled_visual": (
                count,
                POLICY_HISTORY,
                P1_NATIVE_ANCHORS,
                P1_NATIVE_VISUAL_DIM,
            ),
            "global_visual": (count, POLICY_HISTORY, P1_NATIVE_VISUAL_DIM),
            "image_size": (count, 2),
            "fcpg_q_corrected": (count, POLICY_HISTORY, P1_NATIVE_ANCHORS, 2),
            "fcpg_reliability": (count, POLICY_HISTORY, P1_NATIVE_ANCHORS),
            "native_anchor": (
                count,
                POLICY_HISTORY,
                P1_NATIVE_ANCHORS,
                P1_NATIVE_TOKEN_DIM,
            ),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"compact {name} must have shape {shape}, got {value.shape}")
            if name == "image_size":
                if value.dtype.kind not in "iu" or np.any(value <= 0):
                    raise ValueError("compact image_size must contain positive integers")
            elif value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError(f"compact {name} must be finite floating point")


@torch.inference_mode()
def extract_compact_policy_pairs(
    policy: nn.Module,
    pairs: Sequence[Sequence[FrameRecord]],
    *,
    device: str | torch.device,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> CompactPolicyPairs:
    """Extract a compact carrier that exactly reconstructs native anchors."""

    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("compact policy extraction requires a frozen eval policy")
    target_device = torch.device(device)
    batch = _policy_pair_batch(pairs, device=target_device)
    memory, points, reliability = policy._observation_tokens(batch)
    native = grounded_tokens_from_memory(
        memory, num_keypoints=int(getattr(policy, "num_keypoints"))
    )

    rgb = batch["rgb"]
    count, frames = (int(value) for value in rgb.shape[:2])
    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    flat_rgb = rgb.reshape(count * frames, 3, height, width)
    vision = policy.vision_encoder(flat_rgb)
    sampled = sample_feature_map_at_pixels(
        vision.feature_map,
        points.reshape(count * frames, P1_NATIVE_ANCHORS, 2),
        image_size=(width, height),
    ).reshape(count, frames, P1_NATIVE_ANCHORS, P1_NATIVE_VISUAL_DIM)
    global_visual = vision.global_feature.reshape(
        count, frames, P1_NATIVE_VISUAL_DIM
    )
    xyz = batch["keypoints_xyz"]
    reconstructed = policy.grounding_adapter.forward_sampled(
        sampled.reshape(count * frames, P1_NATIVE_ANCHORS, P1_NATIVE_VISUAL_DIM),
        policy._proprio(xyz.reshape(count * frames, P1_NATIVE_ANCHORS, 3)),
        reliability=reliability.reshape(count * frames, P1_NATIVE_ANCHORS),
    ).reshape_as(native)
    if not torch.allclose(reconstructed, native, atol=atol, rtol=rtol):
        error = float(torch.max(torch.abs(reconstructed.float() - native.float())).cpu())
        raise RuntimeError(
            "compact visual carrier does not reconstruct native anchor tokens; "
            f"max error={error:.8f}"
        )
    size = np.broadcast_to(
        np.asarray([width, height], dtype=np.int64), (count, 2)
    ).copy()

    def numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().float().cpu().numpy().astype(np.float32, copy=False)

    return CompactPolicyPairs(
        obs=numpy(batch["obs"]),
        q_nominal=numpy(batch["q_nominal"]),
        keypoints_xyz=numpy(xyz),
        sampled_visual=numpy(sampled),
        global_visual=numpy(global_visual),
        image_size=size,
        fcpg_q_corrected=numpy(points),
        fcpg_reliability=numpy(reliability),
        native_anchor=numpy(native),
    )


@dataclass(frozen=True, slots=True)
class PolicyCarrierEpisode:
    plan: FreshEpisodePlan
    reset_digest: str
    observed_signature: str
    endpoint: CompactPolicyPairs
    pre_drop_history_token: np.ndarray
    pre_drop_action_context: np.ndarray
    pre_drop_visible_token: np.ndarray
    pre_drop_visible_visual: np.ndarray
    branch_future_action: np.ndarray
    same_state_prefix: bool

    def __post_init__(self) -> None:
        if not _is_sha256(self.reset_digest) or not _is_sha256(self.observed_signature):
            raise ValueError("policy carrier episode requires valid physical digests")
        count = BRANCH_COUNT * P1_NATIVE_FUTURE_STEPS
        if int(self.endpoint.obs.shape[0]) != count:
            raise ValueError("policy carrier endpoint count differs from 8 branches x H3-H8")
        shapes = {
            "pre_drop_history_token": (
                BRANCH_COUNT,
                len(PRE_DROP_HISTORY_HORIZONS),
                P1_NATIVE_ANCHORS,
                P1_NATIVE_TOKEN_DIM,
            ),
            "pre_drop_action_context": (
                BRANCH_COUNT,
                len(PRE_DROP_HISTORY_HORIZONS),
                P1_NATIVE_ACTION_DIM,
            ),
            "pre_drop_visible_token": (
                BRANCH_COUNT,
                P1_NATIVE_ANCHORS,
                P1_NATIVE_TOKEN_DIM,
            ),
            "pre_drop_visible_visual": (
                BRANCH_COUNT,
                P1_NATIVE_ANCHORS,
                P1_NATIVE_VISUAL_DIM,
            ),
            "branch_future_action": (
                BRANCH_COUNT,
                P1_NATIVE_FUTURE_STEPS,
                P1_NATIVE_ACTION_DIM,
            ),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite floating point {shape}")
        if not isinstance(self.same_state_prefix, bool):
            raise TypeError("same_state_prefix must be boolean")


def _reshape_branch_endpoint_carrier(
    rows: Sequence[CompactPolicyPairs],
    *,
    endpoint_start: int,
) -> CompactPolicyPairs:
    if len(rows) != BRANCH_COUNT:
        raise ValueError("one episode must contain exactly eight branch carriers")
    names = tuple(CompactPolicyPairs.__dataclass_fields__)
    selected: dict[str, np.ndarray] = {}
    for name in names:
        stacked = np.stack([np.asarray(getattr(row, name)) for row in rows])
        stop = endpoint_start + P1_NATIVE_FUTURE_STEPS
        if endpoint_start < 0 or stop > int(stacked.shape[1]):
            raise ValueError("branch endpoint selection cannot produce six horizons")
        selected[name] = np.ascontiguousarray(
            stacked[:, endpoint_start:stop].reshape(
                BRANCH_COUNT * P1_NATIVE_FUTURE_STEPS, *stacked.shape[2:]
            )
        )
    return CompactPolicyPairs(**selected)


@torch.inference_mode()
def collect_policy_carrier_episode(
    plan: FreshEpisodePlan,
    *,
    inventory: FreshnessInventory,
    policy: nn.Module,
    device: str | torch.device,
    runtime_factory: Callable[[str, int, int], Any],
    branch_bank_builder: Callable[..., Any] = build_diverse_action_branch_bank,
    snapshot_capture: Callable[..., Any] = capture_branch_snapshot,
    snapshot_restore: Callable[[Any, Any], None] = restore_branch_snapshot,
    pair_extractor: Callable[..., CompactPolicyPairs] = extract_compact_policy_pairs,
    same_state_prefix: bool = False,
) -> PolicyCarrierEpisode:
    """Re-run one frozen fresh episode and collect H3-H8 policy carriers.

    The legacy path branches at ``hazard_raw`` and uses its first two branch
    blocks as visible history.  ``same_state_prefix=True`` instead executes two
    common expert blocks, observes H1/H2, and branches only after H2.  The latter
    is the P0 intervention contract: every candidate starts from exactly the
    same causal state and differs only in future action.
    """

    target_device = torch.device(device)
    if int(getattr(policy, "obs_horizon", -1)) != POLICY_HISTORY:
        raise ValueError("policy carrier requires raw-adjacent history=2")
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("policy carrier collection requires a frozen eval policy")
    runtime = runtime_factory(TASK, plan.raw_task_index, plan.environment_seed)
    try:
        initial_record = runtime.record_current(np.asarray(runtime.observation, dtype=np.float32))
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
        branch_raw = (
            hazard_raw + SAME_STATE_PREFIX_MODEL_STEPS * RAW_STEPS_PER_MODEL_STEP
            if same_state_prefix
            else hazard_raw
        )
        final_raw = branch_raw + TOTAL_RAW_BRANCH_STEPS
        if same_state_prefix:
            endpoint_raw = tuple(
                branch_raw - offset * RAW_STEPS_PER_MODEL_STEP
                for offset in (3, 2, 1, 0)
            )
            needed = {
                raw_index
                for endpoint in endpoint_raw
                for raw_index in (endpoint - 1, endpoint)
            }
        else:
            # H-1 and H0 raw-adjacent pairs are both causal and visible.
            needed = {
                hazard_raw - RAW_STEPS_PER_MODEL_STEP - 1,
                hazard_raw - RAW_STEPS_PER_MODEL_STEP,
                hazard_raw - 1,
                hazard_raw,
            }
        observation = np.asarray(runtime.observation, dtype=np.float32)
        source_actions: list[np.ndarray] = []
        records: dict[int, FrameRecord] = {}
        snapshot: Any | None = None
        branch_start_observation: np.ndarray | None = None
        for raw_step in range(final_raw + 1):
            if raw_step in needed:
                records[raw_step] = runtime.record_current(observation)
            if raw_step == branch_raw:
                snapshot = snapshot_capture(runtime.env, include_process_rng=True)
                branch_start_observation = observation.copy()
            if raw_step == final_raw:
                break
            action = _bounded_expert_action(runtime, observation)
            source_actions.append(action.copy())
            observation, _, terminated, truncated, _ = runtime.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            if (terminated or truncated) and raw_step + 1 < final_raw:
                raise RuntimeError("policy-carrier source episode ended before H8")
        if snapshot is None or branch_start_observation is None:
            raise RuntimeError("policy-carrier collector failed to capture branch snapshot")

        source_action = np.asarray(source_actions, dtype=np.float32)
        center = source_action[branch_raw:final_raw]
        if center.shape != (TOTAL_RAW_BRANCH_STEPS, RAW_ACTION_DIM):
            raise RuntimeError("policy-carrier source lacks exact eight-block branch center")
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

        if same_state_prefix:
            common_pairs = tuple(
                (records[endpoint - 1], records[endpoint]) for endpoint in endpoint_raw
            )
        else:
            common_pairs = (
                (
                    records[hazard_raw - RAW_STEPS_PER_MODEL_STEP - 1],
                    records[hazard_raw - RAW_STEPS_PER_MODEL_STEP],
                ),
                (records[hazard_raw - 1], records[hazard_raw]),
            )
        common = pair_extractor(policy, common_pairs, device=target_device)
        branch_carriers: list[CompactPolicyPairs] = []
        h1_tokens: list[np.ndarray] = []
        h2_tokens: list[np.ndarray] = []
        scored_raw_steps = (
            P1_NATIVE_FUTURE_STEPS * RAW_STEPS_PER_MODEL_STEP
            if same_state_prefix
            else TOTAL_RAW_BRANCH_STEPS
        )
        expected_endpoint_count = (
            P1_NATIVE_FUTURE_STEPS if same_state_prefix else TOTAL_MODEL_HORIZONS
        )
        for branch in branch_actions_raw:
            snapshot_restore(runtime.env, snapshot)
            branch_observation = branch_start_observation.copy()
            raw_records: list[FrameRecord] = []
            endpoint_pairs: list[tuple[FrameRecord, FrameRecord]] = []
            for raw_index, action in enumerate(branch[:scored_raw_steps], start=1):
                branch_observation, _, terminated, truncated, _ = runtime.env.step(action)
                branch_observation = np.asarray(branch_observation, dtype=np.float32)
                raw_records.append(runtime.record_current(branch_observation))
                if raw_index % RAW_STEPS_PER_MODEL_STEP == 0:
                    endpoint_pairs.append((raw_records[-2], raw_records[-1]))
                if (terminated or truncated) and raw_index < scored_raw_steps:
                    raise RuntimeError("policy-carrier branch ended before H8")
            if len(endpoint_pairs) != expected_endpoint_count:
                raise RuntimeError("policy-carrier branch lacks all scored endpoints")
            carrier = pair_extractor(policy, endpoint_pairs, device=target_device)
            branch_carriers.append(carrier)
            h1_tokens.append(carrier.native_anchor[0, -1])
            h2_tokens.append(carrier.native_anchor[1, -1])

        if same_state_prefix:
            common_tokens = common.native_anchor[:, -1]
            common_actions = np.stack(
                [
                    source_action[
                        branch_raw - offset * RAW_STEPS_PER_MODEL_STEP :
                        branch_raw - (offset - 1) * RAW_STEPS_PER_MODEL_STEP
                    ].reshape(P1_NATIVE_ACTION_DIM)
                    for offset in (3, 2, 1)
                ],
                axis=0,
            ).astype(np.float32)
            if common_actions.shape != (
                len(PRE_DROP_HISTORY_HORIZONS),
                P1_NATIVE_ACTION_DIM,
            ):
                raise RuntimeError("common-prefix action context is incomplete")
            pre_drop_history = np.broadcast_to(
                common_tokens[:3][None],
                (BRANCH_COUNT, 3, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM),
            ).copy()
            pre_drop_action = np.broadcast_to(
                common_actions[None],
                (BRANCH_COUNT, 3, P1_NATIVE_ACTION_DIM),
            ).copy()
            pre_drop_visible = np.broadcast_to(
                common_tokens[3][None],
                (BRANCH_COUNT, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM),
            ).copy()
            pre_drop_visual = np.broadcast_to(
                common.sampled_visual[3, -1][None],
                (BRANCH_COUNT, P1_NATIVE_ANCHORS, P1_NATIVE_VISUAL_DIM),
            ).copy()
            endpoint_start = 0
            future_action = action_blocks[:, :P1_NATIVE_FUTURE_STEPS]
        else:
            prior_raw = source_action[
                hazard_raw - RAW_STEPS_PER_MODEL_STEP : hazard_raw
            ]
            if prior_raw.shape != (RAW_STEPS_PER_MODEL_STEP, RAW_ACTION_DIM):
                raise RuntimeError("policy-carrier source lacks causal H-1 to H0 action")
            prior_action = prior_raw.reshape(P1_NATIVE_ACTION_DIM).astype(np.float32)
            hminus1 = common.native_anchor[0, -1]
            h0 = common.native_anchor[1, -1]
            pre_drop_history = np.stack(
                [
                    np.stack((hminus1, h0, h1_tokens[branch]), axis=0)
                    for branch in range(BRANCH_COUNT)
                ]
            ).astype(np.float32)
            pre_drop_action = np.stack(
                [
                    np.stack(
                        (prior_action, action_blocks[branch, 0], action_blocks[branch, 1]),
                        axis=0,
                    )
                    for branch in range(BRANCH_COUNT)
                ]
            ).astype(np.float32)
            pre_drop_visible = np.stack(h2_tokens).astype(np.float32)
            pre_drop_visual = np.stack(
                [row.sampled_visual[1, -1] for row in branch_carriers]
            ).astype(np.float32)
            endpoint_start = 2
            future_action = action_blocks[:, 2:]
        return PolicyCarrierEpisode(
            plan=plan,
            reset_digest=reset_digest,
            observed_signature=signature,
            endpoint=_reshape_branch_endpoint_carrier(
                branch_carriers, endpoint_start=endpoint_start
            ),
            pre_drop_history_token=pre_drop_history,
            pre_drop_action_context=pre_drop_action,
            pre_drop_visible_token=pre_drop_visible,
            pre_drop_visible_visual=pre_drop_visual,
            branch_future_action=np.ascontiguousarray(future_action, dtype=np.float32),
            same_state_prefix=bool(same_state_prefix),
        )
    finally:
        _close_episode_env_safely(runtime.env)


def stack_policy_carrier_episodes(
    rows: Sequence[PolicyCarrierEpisode],
    *,
    contract: Mapping[str, Any],
    include_labels: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Flatten episode/branch/H3-H8 while preserving strict row identities.

    ``include_labels=False`` is the deployment-first path.  It deliberately
    does not materialize native target tokens; callers may request the labels
    from the same frozen replay only after all deployment predictions finish.
    """

    values = list(rows)
    if len(values) != EPISODE_COUNT:
        raise ValueError("formal policy carrier requires exactly 30 episodes")
    if tuple(row.plan for row in values) != build_fresh_episode_plans(contract):
        raise PermissionError("policy carrier rows differ from the frozen plan/order")
    if len({row.reset_digest for row in values}) != EPISODE_COUNT:
        raise PermissionError("policy carrier contains repeated physical resets")
    if len({row.observed_signature for row in values}) != EPISODE_COUNT:
        raise PermissionError("policy carrier contains repeated observed states")
    prefix_modes = {row.same_state_prefix for row in values}
    if len(prefix_modes) != 1:
        raise PermissionError("policy carrier mixes legacy and same-state prefix rows")
    if prefix_modes != {True}:
        raise PermissionError("formal policy carrier requires a common H2 branch prefix")

    per_episode = BRANCH_COUNT * P1_NATIVE_FUTURE_STEPS
    count = EPISODE_COUNT * per_episode
    episode_uid = np.repeat(
        np.asarray([row.plan.episode_uid for row in values]), per_episode
    )
    reset_digest = np.repeat(
        np.asarray([row.reset_digest for row in values]), per_episode
    )
    observed_signature = np.repeat(
        np.asarray([row.observed_signature for row in values]), per_episode
    )
    branch = np.tile(
        np.repeat(np.arange(BRANCH_COUNT, dtype=np.int64), P1_NATIVE_FUTURE_STEPS),
        EPISODE_COUNT,
    )
    horizon = np.tile(np.asarray(LABEL_HORIZONS, dtype=np.int64), EPISODE_COUNT * BRANCH_COUNT)
    row_id = np.asarray(
        [
            _row_id(str(uid), int(branch_value), int(horizon_value))
            for uid, branch_value, horizon_value in zip(
                episode_uid.tolist(), branch.tolist(), horizon.tolist(), strict=True
            )
        ]
    )
    if len(set(row_id.tolist())) != count:
        raise RuntimeError("policy carrier row identities are not unique")

    def endpoints(name: str) -> np.ndarray:
        return np.ascontiguousarray(
            np.concatenate([np.asarray(getattr(row.endpoint, name)) for row in values])
        )

    repeat_horizon = P1_NATIVE_FUTURE_STEPS
    pre_history = np.concatenate(
        [
            np.repeat(row.pre_drop_history_token, repeat_horizon, axis=0)
            for row in values
        ]
    ).astype(np.float32)
    pre_action = np.concatenate(
        [
            np.repeat(row.pre_drop_action_context, repeat_horizon, axis=0)
            for row in values
        ]
    ).astype(np.float32)
    pre_visible = np.concatenate(
        [
            np.repeat(row.pre_drop_visible_token, repeat_horizon, axis=0)
            for row in values
        ]
    ).astype(np.float32)
    pre_visible_visual = np.concatenate(
        [
            np.repeat(row.pre_drop_visible_visual, repeat_horizon, axis=0)
            for row in values
        ]
    ).astype(np.float32)
    future_action = np.concatenate(
        [row.branch_future_action.reshape(per_episode, P1_NATIVE_ACTION_DIM) for row in values]
    ).astype(np.float32)

    identity = {
        "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
        "protocol": np.asarray([PROTOCOL]),
        "classification": np.asarray([CLASSIFICATION]),
        "policy_row_id": row_id,
        "episode_uid": episode_uid,
        "physical_episode_id": np.asarray(
            [f"{TASK}:{value}" for value in reset_digest.tolist()]
        ),
        "reset_physical_state_goal_digest": reset_digest,
        "observed_state_signature": observed_signature,
        "branch_index": branch,
        "horizon": horizon,
    }
    deployment = {
        **identity,
        "policy_obs": endpoints("obs").astype(np.float32),
        "policy_q_nominal": endpoints("q_nominal").astype(np.float32),
        "policy_keypoints_xyz": endpoints("keypoints_xyz").astype(np.float32),
        "policy_sampled_visual": endpoints("sampled_visual").astype(np.float32),
        "policy_global_visual": endpoints("global_visual").astype(np.float32),
        "policy_image_size": endpoints("image_size").astype(np.int64),
        "policy_fcpg_q_corrected": endpoints("fcpg_q_corrected").astype(np.float32),
        "policy_fcpg_reliability": endpoints("fcpg_reliability").astype(np.float32),
        "pre_drop_history_horizons": np.asarray(PRE_DROP_HISTORY_HORIZONS, dtype=np.int64),
        "pre_drop_target_horizon": np.asarray([PRE_DROP_TARGET_HORIZON], dtype=np.int64),
        "pre_drop_history_token": pre_history,
        "pre_drop_action_context": pre_action,
        "pre_drop_visible_token": pre_visible,
        "pre_drop_visible_visual": pre_visible_visual,
        "policy_future_action": future_action,
        "same_state_prefix": np.asarray([bool(next(iter(prefix_modes)))], dtype=bool),
    }
    if not include_labels:
        return deployment, None
    labels = {
        **identity,
        "target_is_offline_label": np.asarray([True], dtype=bool),
        "policy_target_anchor": endpoints("native_anchor").astype(np.float32),
    }
    validate_policy_carrier_arrays(deployment, labels, require_predictions=False)
    return deployment, labels


def policy_row_ids_for_contract(contract: Mapping[str, Any]) -> np.ndarray:
    """Return the deterministic source row IDs for the frozen plan order."""

    plans = build_fresh_episode_plans(contract)
    values: list[str] = []
    for plan in plans:
        for branch in range(BRANCH_COUNT):
            for horizon in LABEL_HORIZONS:
                values.append(_row_id(plan.episode_uid, branch, horizon))
    result = np.asarray(values)
    expected = EPISODE_COUNT * BRANCH_COUNT * P1_NATIVE_FUTURE_STEPS
    if result.shape != (expected,) or len(set(result.tolist())) != expected:
        raise RuntimeError("deterministic policy row IDs are not unique/complete")
    return result


def attach_aligned_predictions(
    deployment: Mapping[str, np.ndarray],
    predictions: np.ndarray,
    *,
    prediction_row_id: np.ndarray,
    pre_drop_raw_prior_token: np.ndarray,
    raw_factual_predictions: np.ndarray,
    matched_no_action_predictions: np.ndarray,
    wrong_action_predictions: np.ndarray,
    predictor_seed: int,
    predictor_checkpoint_sha256: str,
) -> dict[str, np.ndarray]:
    """Attach endpoint predictions after an exact row-identity check.

    ``predictions`` may contain one model-step anchor per row ``[N,K,D]`` or
    an explicit raw-adjacent override ``[N,2,K,D]``.  The former is repeated in
    both slots to reproduce the existing P1 policy runtime contract.
    """

    if "policy_predicted_anchor" in deployment:
        raise ValueError("deployment already contains a predicted anchor")
    row_id = np.asarray(deployment["policy_row_id"]).astype(str)
    supplied_id = np.asarray(prediction_row_id).astype(str)
    if supplied_id.shape != row_id.shape or not np.array_equal(supplied_id, row_id):
        raise PermissionError("prediction identities/order differ from policy carrier rows")
    def as_pair(value: np.ndarray, *, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape == current_shape:
            array = np.repeat(array[:, None], POLICY_HISTORY, axis=1)
        elif array.shape != pair_shape:
            raise ValueError(
                f"{label} must have shape {current_shape} or {pair_shape}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{label} contains non-finite values")
        return np.ascontiguousarray(array, dtype=np.float32)

    value = np.asarray(predictions, dtype=np.float32)
    current_shape = (row_id.size, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM)
    pair_shape = (row_id.size, POLICY_HISTORY, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM)
    value = as_pair(value, label="complementary factual predictions")
    raw_pair = as_pair(raw_factual_predictions, label="raw factual predictions")
    no_action_pair = as_pair(
        matched_no_action_predictions, label="matched no-action predictions"
    )
    wrong = np.asarray(wrong_action_predictions, dtype=np.float32)
    wrong_current_shape = (
        row_id.size,
        7,
        P1_NATIVE_ANCHORS,
        P1_NATIVE_TOKEN_DIM,
    )
    wrong_pair_shape = (
        row_id.size,
        7,
        POLICY_HISTORY,
        P1_NATIVE_ANCHORS,
        P1_NATIVE_TOKEN_DIM,
    )
    if wrong.shape == wrong_current_shape:
        wrong = np.repeat(wrong[:, :, None], POLICY_HISTORY, axis=2)
    elif wrong.shape != wrong_pair_shape:
        raise ValueError(
            "wrong-action predictions must have shape "
            f"{wrong_current_shape} or {wrong_pair_shape}, got {wrong.shape}"
        )
    if not np.isfinite(wrong).all():
        raise ValueError("wrong-action predictions contain non-finite values")
    if predictor_seed < 0 or not _is_sha256(predictor_checkpoint_sha256):
        raise ValueError("prediction provenance is invalid")
    prior = np.asarray(pre_drop_raw_prior_token, dtype=np.float32)
    prior_shape = (row_id.size, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM)
    if prior.shape != prior_shape or not np.isfinite(prior).all():
        raise ValueError(f"pre_drop_raw_prior_token must be finite {prior_shape}")
    result = {str(name): np.asarray(array).copy() for name, array in deployment.items()}
    result["policy_predicted_anchor"] = value.copy()
    result["policy_predicted_anchor_primary_method"] = np.asarray(
        ["complementary_factual"]
    )
    result["policy_predicted_anchor_complementary_factual"] = value.copy()
    result["policy_predicted_anchor_raw_factual"] = raw_pair
    result["policy_predicted_anchor_matched_no_action"] = no_action_pair
    result["policy_predicted_anchor_wrong_action"] = np.ascontiguousarray(
        wrong, dtype=np.float32
    )
    result["pre_drop_raw_prior_token"] = np.ascontiguousarray(prior, dtype=np.float32)
    result["pre_drop_prior_is_causal"] = np.asarray([True], dtype=bool)
    result["predictor_seed"] = np.asarray([predictor_seed], dtype=np.int64)
    result["predictor_checkpoint_sha256"] = np.asarray([predictor_checkpoint_sha256])
    result["prediction_alignment"] = np.asarray([PREDICTION_ALIGNMENT])
    return result


def attach_frozen_recursive_inputs(
    deployment: Mapping[str, np.ndarray],
    *,
    source_row_id: np.ndarray,
    future_action: np.ndarray,
    geometry_token: np.ndarray,
) -> dict[str, np.ndarray]:
    """Attach H3-H8 action/geometry rows after an exact row-identity check.

    The v2 carrier supplies both arrays from the same physical replay as its
    native labels.  The legacy helper name is retained because historical unit
    tests also use it, but the receipt records the truthful same-replay source.
    """

    row_id = np.asarray(deployment["policy_row_id"]).astype(str)
    supplied = np.asarray(source_row_id).astype(str)
    if supplied.shape != row_id.shape or not np.array_equal(supplied, row_id):
        raise PermissionError("recursive-input identities/order differ from carrier")
    action = np.asarray(future_action, dtype=np.float32)
    geometry = np.asarray(geometry_token, dtype=np.float32)
    expected_action = (row_id.size, P1_NATIVE_ACTION_DIM)
    expected_geometry = (row_id.size, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM)
    if action.shape != expected_action or not np.isfinite(action).all():
        raise ValueError(f"future_action must be finite {expected_action}")
    collected_action = np.asarray(deployment.get("policy_future_action"), dtype=np.float32)
    if collected_action.shape != expected_action or not np.array_equal(
        action, collected_action
    ):
        raise PermissionError(
            "recursive future_action differs from the same-replay collected action"
        )
    if geometry.shape != expected_geometry or not np.isfinite(geometry).all():
        raise ValueError(f"geometry_token must be finite {expected_geometry}")
    result = {str(name): np.asarray(array).copy() for name, array in deployment.items()}
    result["policy_geometry_token"] = np.ascontiguousarray(geometry, dtype=np.float32)
    result["recursive_inputs_are_from_identity_verified_same_replay"] = np.asarray(
        [True], dtype=bool
    )
    return result


def verify_frozen_source_alignment(
    deployment: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    expected_physical_episode_id: np.ndarray,
    expected_current_target: np.ndarray | None = None,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> None:
    """Bind a recollected carrier to frozen physical row order.

    A target comparison is optional and is only valid when both artifacts were
    rendered in the same deterministic replay.  Fresh carriers normally use
    native labels from their own replay, because EGL rendering can introduce
    sub-milliscale RGB/token drift across otherwise identical processes.
    """

    if atol < 0.0 or rtol < 0.0:
        raise ValueError("alignment tolerances must be non-negative")
    observed_physical = np.asarray(deployment["physical_episode_id"]).astype(str)
    expected_physical = np.asarray(expected_physical_episode_id).astype(str)
    if (
        expected_physical.shape != observed_physical.shape
        or not np.array_equal(observed_physical, expected_physical)
    ):
        raise PermissionError(
            "new carrier physical identity/order does not match frozen P1 data"
        )
    if expected_current_target is not None:
        target_pair = np.asarray(labels["policy_target_anchor"], dtype=np.float32)
        if target_pair.ndim != 4 or target_pair.shape[1] != POLICY_HISTORY:
            raise ValueError("new carrier target must be raw-adjacent [N,2,K,D]")
        observed_current = target_pair[:, -1]
        expected_current = np.asarray(expected_current_target, dtype=np.float32)
        if expected_current.shape != observed_current.shape:
            raise ValueError("frozen current target shape differs from recollected carrier")
        if not np.allclose(observed_current, expected_current, atol=atol, rtol=rtol):
            maximum = float(np.max(np.abs(observed_current - expected_current)))
            raise PermissionError(
                "new carrier current anchors do not align to frozen P1 labels; "
                f"max error={maximum:.8f}"
            )


def validate_policy_carrier_arrays(
    deployment: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    require_predictions: bool,
) -> None:
    forbidden = FORBIDDEN_DEPLOYMENT_KEYS.intersection(deployment)
    if forbidden:
        raise ValueError(f"deployment carrier contains forbidden fields: {sorted(forbidden)}")
    count = EPISODE_COUNT * BRANCH_COUNT * P1_NATIVE_FUTURE_STEPS
    required_deployment = {
        "policy_row_id",
        "episode_uid",
        "physical_episode_id",
        "reset_physical_state_goal_digest",
        "observed_state_signature",
        "branch_index",
        "horizon",
        "policy_obs",
        "policy_q_nominal",
        "policy_keypoints_xyz",
        "policy_sampled_visual",
        "policy_global_visual",
        "policy_image_size",
        "policy_fcpg_q_corrected",
        "policy_fcpg_reliability",
        "pre_drop_history_token",
        "pre_drop_action_context",
        "pre_drop_visible_token",
        "pre_drop_visible_visual",
        "policy_future_action",
        "same_state_prefix",
    }
    missing = required_deployment - set(deployment)
    if missing:
        raise KeyError(f"deployment carrier is missing {sorted(missing)}")
    if require_predictions and "policy_predicted_anchor" not in deployment:
        raise KeyError("deployment carrier is missing policy_predicted_anchor")
    if require_predictions:
        prediction_fields = {
            "pre_drop_raw_prior_token",
            "pre_drop_prior_is_causal",
            "predictor_seed",
            "predictor_checkpoint_sha256",
            "prediction_alignment",
            "policy_predicted_anchor_complementary_factual",
            "policy_predicted_anchor_raw_factual",
            "policy_predicted_anchor_matched_no_action",
            "policy_predicted_anchor_wrong_action",
            "policy_predicted_anchor_primary_method",
            "policy_geometry_token",
            "recursive_inputs_are_from_identity_verified_same_replay",
        }
        missing_prediction = prediction_fields - set(deployment)
        if missing_prediction:
            raise KeyError(
                f"deployment carrier is missing prediction provenance {sorted(missing_prediction)}"
            )
    if "policy_target_anchor" not in labels:
        raise KeyError("offline labels are missing policy_target_anchor")
    identity_fields = (
        "policy_row_id",
        "episode_uid",
        "physical_episode_id",
        "reset_physical_state_goal_digest",
        "observed_state_signature",
        "branch_index",
        "horizon",
    )
    for name in identity_fields:
        left = np.asarray(deployment[name])
        right = np.asarray(labels[name])
        if left.shape != (count,) or not np.array_equal(left, right):
            raise PermissionError(f"deployment/label identity mismatch for {name}")
    row_ids = np.asarray(deployment["policy_row_id"]).astype(str)
    episode_ids = np.asarray(deployment["episode_uid"]).astype(str)
    branches = np.asarray(deployment["branch_index"], dtype=np.int64)
    horizons = np.asarray(deployment["horizon"], dtype=np.int64)
    if len(set(row_ids.tolist())) != count:
        raise PermissionError("policy carrier row IDs must be unique")
    expected_branches = np.tile(
        np.repeat(np.arange(BRANCH_COUNT, dtype=np.int64), P1_NATIVE_FUTURE_STEPS),
        EPISODE_COUNT,
    )
    expected_horizons = np.tile(
        np.asarray(LABEL_HORIZONS, dtype=np.int64), EPISODE_COUNT * BRANCH_COUNT
    )
    if not np.array_equal(branches, expected_branches) or not np.array_equal(
        horizons, expected_horizons
    ):
        raise PermissionError("policy carrier is not the fixed branch-major H3-H8 panel")
    episode_panel = episode_ids.reshape(EPISODE_COUNT, -1)
    if any(len(set(row.tolist())) != 1 for row in episode_panel):
        raise PermissionError("policy carrier episode identity changes inside a panel")
    if len(set(episode_panel[:, 0].tolist())) != EPISODE_COUNT:
        raise PermissionError("policy carrier must contain 30 distinct episode panels")
    expected_row_ids = np.asarray(
        [
            _row_id(uid, int(branch), int(horizon))
            for uid, branch, horizon in zip(
                episode_ids.tolist(), branches.tolist(), horizons.tolist(), strict=True
            )
        ]
    )
    if not np.array_equal(row_ids, expected_row_ids):
        raise PermissionError("policy carrier row IDs do not bind episode/branch/horizon")
    target_shape = (count, POLICY_HISTORY, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM)
    obs = np.asarray(deployment["policy_obs"])
    if (
        obs.ndim != 3
        or obs.shape[:2] != (count, POLICY_HISTORY)
        or obs.dtype.kind != "f"
        or not np.isfinite(obs).all()
    ):
        raise ValueError("policy_obs must be finite [N,2,obs_dim]")
    expected_deployment_shapes = {
        "policy_q_nominal": (count, POLICY_HISTORY, P1_NATIVE_ANCHORS, 2),
        "policy_keypoints_xyz": (count, POLICY_HISTORY, P1_NATIVE_ANCHORS, 3),
        "policy_sampled_visual": (
            count,
            POLICY_HISTORY,
            P1_NATIVE_ANCHORS,
            P1_NATIVE_VISUAL_DIM,
        ),
        "policy_global_visual": (count, POLICY_HISTORY, P1_NATIVE_VISUAL_DIM),
        "policy_fcpg_q_corrected": (
            count,
            POLICY_HISTORY,
            P1_NATIVE_ANCHORS,
            2,
        ),
        "policy_fcpg_reliability": (
            count,
            POLICY_HISTORY,
            P1_NATIVE_ANCHORS,
        ),
        "pre_drop_history_token": (
            count,
            len(PRE_DROP_HISTORY_HORIZONS),
            P1_NATIVE_ANCHORS,
            P1_NATIVE_TOKEN_DIM,
        ),
        "pre_drop_action_context": (
            count,
            len(PRE_DROP_HISTORY_HORIZONS),
            P1_NATIVE_ACTION_DIM,
        ),
        "pre_drop_visible_token": (
            count,
            P1_NATIVE_ANCHORS,
            P1_NATIVE_TOKEN_DIM,
        ),
        "pre_drop_visible_visual": (
            count,
            P1_NATIVE_ANCHORS,
            P1_NATIVE_VISUAL_DIM,
        ),
        "policy_future_action": (count, P1_NATIVE_ACTION_DIM),
    }
    for name, shape in expected_deployment_shapes.items():
        value = np.asarray(deployment[name])
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating point {shape}")
    image_size = np.asarray(deployment["policy_image_size"])
    if (
        image_size.shape != (count, 2)
        or image_size.dtype.kind not in "iu"
        or np.any(image_size <= 0)
    ):
        raise ValueError("policy_image_size must be positive integer [N,2]")
    same_state = np.asarray(deployment["same_state_prefix"])
    if (
        same_state.shape != (1,)
        or same_state.dtype.kind != "b"
        or not bool(same_state[0])
    ):
        raise ValueError("same_state_prefix receipt must be boolean true")
    target = np.asarray(labels["policy_target_anchor"])
    if target.shape != target_shape or target.dtype.kind != "f" or not np.isfinite(target).all():
        raise ValueError(f"policy_target_anchor must be finite {target_shape}")
    if require_predictions:
        predicted = np.asarray(deployment["policy_predicted_anchor"])
        if predicted.shape != target_shape or not np.isfinite(predicted).all():
            raise ValueError(f"policy_predicted_anchor must be finite {target_shape}")
        for name in (
            "policy_predicted_anchor_complementary_factual",
            "policy_predicted_anchor_raw_factual",
            "policy_predicted_anchor_matched_no_action",
        ):
            control = np.asarray(deployment[name])
            if control.shape != target_shape or not np.isfinite(control).all():
                raise ValueError(f"{name} must be finite {target_shape}")
        if not np.array_equal(
            predicted,
            np.asarray(deployment["policy_predicted_anchor_complementary_factual"]),
        ):
            raise ValueError("primary prediction alias must equal complementary factual")
        wrong_shape = (
            count,
            7,
            POLICY_HISTORY,
            P1_NATIVE_ANCHORS,
            P1_NATIVE_TOKEN_DIM,
        )
        wrong = np.asarray(deployment["policy_predicted_anchor_wrong_action"])
        if wrong.shape != wrong_shape or not np.isfinite(wrong).all():
            raise ValueError(f"wrong-action predictions must be finite {wrong_shape}")
        primary = np.asarray(deployment["policy_predicted_anchor_primary_method"])
        if primary.shape != (1,) or primary[0] != "complementary_factual":
            raise ValueError("primary predicted-anchor method receipt is invalid")
        future_action = np.asarray(deployment["policy_future_action"])
        geometry = np.asarray(deployment["policy_geometry_token"])
        if future_action.shape != (count, P1_NATIVE_ACTION_DIM) or not np.isfinite(
            future_action
        ).all():
            raise ValueError("policy_future_action has invalid shape/content")
        if geometry.shape != (
            count,
            P1_NATIVE_ANCHORS,
            P1_NATIVE_TOKEN_DIM,
        ) or not np.isfinite(geometry).all():
            raise ValueError("policy_geometry_token has invalid shape/content")
        recursive_receipt = np.asarray(
            deployment["recursive_inputs_are_from_identity_verified_same_replay"]
        )
        if (
            recursive_receipt.shape != (1,)
            or recursive_receipt.dtype.kind != "b"
            or not bool(recursive_receipt[0])
        ):
            raise ValueError("recursive same-replay identity receipt is invalid")
        prior = np.asarray(deployment["pre_drop_raw_prior_token"])
        prior_shape = (count, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM)
        if prior.shape != prior_shape or not np.isfinite(prior).all():
            raise ValueError(f"pre_drop_raw_prior_token must be finite {prior_shape}")
        causal = np.asarray(deployment["pre_drop_prior_is_causal"])
        if causal.shape != (1,) or causal.dtype.kind != "b" or not bool(causal[0]):
            raise ValueError("pre_drop_prior_is_causal must be boolean true")
        seed = np.asarray(deployment["predictor_seed"])
        checkpoint = np.asarray(deployment["predictor_checkpoint_sha256"])
        alignment = np.asarray(deployment["prediction_alignment"])
        if seed.shape != (1,) or seed.dtype.kind not in "iu" or int(seed[0]) < 0:
            raise ValueError("predictor_seed provenance is invalid")
        if checkpoint.shape != (1,) or not _is_sha256(str(checkpoint[0])):
            raise ValueError("predictor checkpoint provenance is invalid")
        if alignment.shape != (1,) or alignment[0] != PREDICTION_ALIGNMENT:
            raise ValueError("prediction alignment receipt is invalid")


def save_policy_carrier_bundle(
    output_dir: str | Path,
    deployment_by_seed: Mapping[int, Mapping[str, np.ndarray]],
    labels: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, object],
) -> Path:
    """Atomically create new carrier artifacts; existing paths are rejected."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite policy carrier: {destination}")
    if not deployment_by_seed:
        raise ValueError("at least one predictor seed is required")
    for deployment in deployment_by_seed.values():
        validate_policy_carrier_arrays(deployment, labels, require_predictions=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        label_path = temporary / "policy_offline_labels.npz"
        with label_path.open("xb") as handle:
            np.savez_compressed(handle, **labels)
            handle.flush()
            os.fsync(handle.fileno())
        records: dict[str, object] = {}
        for seed, deployment in sorted(deployment_by_seed.items()):
            path = temporary / f"policy_deployment_seed{int(seed)}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, **deployment)
                handle.flush()
                os.fsync(handle.fileno())
            records[str(seed)] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        report = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "classification": CLASSIFICATION,
            "completed": True,
            "deployment_contains_no_q_true_or_target": True,
            "healthy_visual_carrier_makes_this_exposed_not_deployable": True,
            "row_count": EPISODE_COUNT * BRANCH_COUNT * P1_NATIVE_FUTURE_STEPS,
            "offline_labels": {
                "path": label_path.name,
                "sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            },
            "deployments": records,
            **dict(metadata),
        }
        metadata_path = temporary / "carrier.metadata.json"
        with metadata_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "CLASSIFICATION",
    "CompactPolicyPairs",
    "FORBIDDEN_DEPLOYMENT_KEYS",
    "POLICY_HISTORY",
    "PRE_DROP_HISTORY_HORIZONS",
    "PRE_DROP_TARGET_HORIZON",
    "PROTOCOL",
    "PolicyCarrierEpisode",
    "attach_aligned_predictions",
    "attach_frozen_recursive_inputs",
    "collect_policy_carrier_episode",
    "extract_compact_policy_pairs",
    "save_policy_carrier_bundle",
    "stack_policy_carrier_episodes",
    "policy_row_ids_for_contract",
    "validate_policy_carrier_arrays",
    "verify_frozen_source_alignment",
]
