"""Strict A3-to-P1 bridge collection over replayed Pick action branches.

The collector joins three frozen interfaces without changing their clocks:

* A3 consumes decision-synchronous H=3 anchor embeddings and 5x4 action blocks;
* P1 consumes two raw-adjacent observations ``[t-1, t]``;
* hidden clean P1 tokens are labels only and never enter A3 prediction.

Only Pick train/development rows from the frozen A3 screen are eligible.  A
caller may bound rows/instances/branches for smoke collection, but cannot relax
the source-action, latent-reconstruction, or native-token checks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from fcpg.lewm_action_branch_a1 import validate_a1_arrays
from fcpg.lewm_action_branch_a3_collection import (
    TASKS,
    validate_a3_cache_conformance,
    validate_a3_contract,
)
from fcpg.lewm_action_branch_a3_physical import (
    validate_reconstructed_source_actions,
)
from fcpg.lewm_action_branch_collection import (
    ACTION_DIM,
    ANCHOR_COUNT,
    FRAME_SKIP,
    HISTORY_SIZE,
    LATENT_DIM,
    EpisodeRuntime,
    capture_branch_snapshot,
    restore_branch_snapshot,
)
from fcpg.lewm_action_branch_training import rollout_action_branches
from fcpg.lewm_counterfactual_runtime import FrameRecord
from fcpg.lewm_known_drop_policy_evaluator import (
    PolicyAnchorObservation,
    extract_policy_anchor_observation,
)


PICK_TASK = "pick-place-v3"
PICK_TASK_INDEX = 0
BRANCH_COUNT = 8
HORIZON = 8
P1_HISTORY = 2
P1_TOKEN_DIM = 256
P1_VISUAL_DIM = 256
# The source cache was encoded with CUDA autocast.  Replay is byte-exact in
# state/action/image space, while fused FP16 kernels vary by up to one 1e-3
# ULP-scale unit across batching/concurrent-kernel schedules.  This is a numeric
# reconstruction tolerance, not an episode/action tolerance.
A3_RECONSTRUCTION_ATOL = 1.0e-3

STRICT_P1_POLICY_SUFFIX = (
    "outputs/action_chunk/p1/pick_fcpg_pureres_seed0_retry1/"
    "fcpg_action_chunk_best.pt"
)
STRICT_P1_POLICY_SHA256 = (
    "f0650d8c1e44ef7dc88a70e0af7478572b0b1439078d3e6a50a43a4c573592b9"
)
STRICT_P1_CORRECTOR_SUFFIX = (
    "outputs/transformer_residual/lewm_stage1/pick_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
STRICT_P1_CORRECTOR_SHA256 = (
    "d1a617c91b4701d58c1814ca8de95c27128654cf62d359930ebd7f3a9198e3c0"
)
STRICT_A3_10K_SEEDS = (29301, 29302, 29303)
P1_VISIBLE_GROUNDING_MODES = ("fcpg", "nominal")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root_from_artifact(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "fcpg").is_dir() and (candidate / "outputs").is_dir():
            return candidate
    raise ValueError(f"cannot locate repository root from artifact {path}")


def validate_strict_p1_checkpoint(
    policy_checkpoint: str | Path,
    corrector_checkpoint: str | Path,
    *,
    expected_policy_sha256: str = STRICT_P1_POLICY_SHA256,
    expected_corrector_sha256: str = STRICT_P1_CORRECTOR_SHA256,
    expected_policy_suffix: str = STRICT_P1_POLICY_SUFFIX,
    expected_corrector_suffix: str = STRICT_P1_CORRECTOR_SUFFIX,
) -> dict[str, Any]:
    """Fail closed unless the policy embeds the exact authorized corrector."""

    policy_path = Path(policy_checkpoint).expanduser().resolve()
    corrector_path = Path(corrector_checkpoint).expanduser().resolve()
    if not policy_path.is_file() or not corrector_path.is_file():
        raise FileNotFoundError("strict P1 policy and corrector checkpoints must exist")
    if expected_policy_suffix and not policy_path.as_posix().endswith(
        expected_policy_suffix
    ):
        raise ValueError("strict collector received an unauthorized P1 policy path")
    if expected_corrector_suffix and not corrector_path.as_posix().endswith(
        expected_corrector_suffix
    ):
        raise ValueError("strict collector received an unauthorized P1 corrector path")
    policy_sha = sha256_file(policy_path)
    corrector_sha = sha256_file(corrector_path)
    if policy_sha != expected_policy_sha256:
        raise ValueError("strict P1 policy SHA-256 mismatch")
    if corrector_sha != expected_corrector_sha256:
        raise ValueError("strict P1 corrector SHA-256 mismatch")

    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("P1 checkpoint must contain a mapping")
    if checkpoint.get("checkpoint_type") != "fcpg_action_policy":
        raise ValueError("strict P1 checkpoint type mismatch")
    args = checkpoint.get("args")
    if not isinstance(args, Mapping) or str(args.get("mode")) != "fcpg":
        raise ValueError("strict P1 checkpoint must be an FCPG policy")
    if str(args.get("policy_arch", "legacy")) != "legacy":
        raise ValueError("strict P1 collector is fixed to the legacy P1 policy")
    if int(args.get("obs_horizon", -1)) != P1_HISTORY:
        raise ValueError("strict P1 policy must use raw-adjacent H=2 history")
    if int(checkpoint.get("num_keypoints", -1)) != ANCHOR_COUNT:
        raise ValueError("strict P1 policy must use exactly three anchors")
    if int(args.get("model_dim", -1)) != P1_TOKEN_DIM:
        raise ValueError("strict P1 policy token dimension must be 256")
    if int(args.get("vision_dim", -1)) != P1_VISUAL_DIM:
        raise ValueError("strict P1 sampled-visual dimension must be 256")

    repo_root = _repo_root_from_artifact(policy_path)
    embedded_value = str(args.get("fcpg_checkpoint", ""))
    embedded = Path(embedded_value).expanduser()
    if not embedded.is_absolute():
        embedded = repo_root / embedded
    embedded = embedded.resolve()
    if embedded != corrector_path:
        raise ValueError("P1 embedded corrector path differs from the authorized path")
    if sha256_file(embedded) != corrector_sha:
        raise ValueError("P1 embedded corrector bytes differ from the authorized bytes")
    return {
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha,
        "corrector_path": str(corrector_path),
        "corrector_sha256": corrector_sha,
        "obs_horizon": P1_HISTORY,
        "history_alignment": "raw-adjacent-pre-action-[t-1,t]",
    }


def validate_a3_10k_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    path: str | Path,
) -> dict[str, Any]:
    """Validate one frozen A3 fixed-latest 10k matched pair."""

    if checkpoint.get("checkpoint_type") != "fcpg_matched_anchor_dynamics_pair_v1":
        raise ValueError("A3 predictor checkpoint type mismatch")
    if checkpoint.get("training_protocol") != "fcpg_action_branch_physical_support_a3_v1":
        raise ValueError("predictor was not trained under the frozen A3 protocol")
    if int(checkpoint.get("step", -1)) != 10_000 or not bool(
        checkpoint.get("completed", False)
    ):
        raise ValueError("A3 predictor must be a completed step=10000 endpoint")
    if checkpoint.get("selection") != "fixed_latest":
        raise ValueError("A3 predictor must use fixed_latest selection")
    if not bool(checkpoint.get("encoder_frozen", False)):
        raise ValueError("A3 predictor checkpoint did not freeze the anchor encoder")
    seed = int(checkpoint.get("seed", -1))
    if seed not in STRICT_A3_10K_SEEDS:
        raise ValueError(f"unexpected A3 10k seed {seed}")
    config = checkpoint.get("predictor_config")
    if not isinstance(config, Mapping):
        raise ValueError("A3 predictor checkpoint is missing predictor_config")
    expected = {
        "history_size": HISTORY_SIZE,
        "num_keypoints": ANCHOR_COUNT,
        "latent_dim": LATENT_DIM,
        "action_dim": FRAME_SKIP * ACTION_DIM,
    }
    for name, value in expected.items():
        if int(config.get(name, -1)) != value:
            raise ValueError(f"A3 predictor {name} violates the bridge contract")
    resolved = Path(path).expanduser().resolve()
    return {"seed": seed, "path": str(resolved), "sha256": sha256_file(resolved)}


def select_pick_bridge_rows(
    arrays: Mapping[str, np.ndarray],
    *,
    max_instances: int | None = None,
    max_rows: int | None = None,
) -> np.ndarray:
    """Select deterministic Pick train/development rows without opening test."""

    validate_a1_arrays(arrays)
    task_index = np.asarray(arrays["task_index"], dtype=np.int64)
    split = np.asarray(arrays["split"]).astype(str)
    instance = np.asarray(arrays["task_instance_index"], dtype=np.int64)
    hazard = np.asarray(arrays["hazard_id"], dtype=np.int64)
    eligible = np.flatnonzero(
        (task_index == PICK_TASK_INDEX) & np.isin(split, ("train", "development"))
    )
    if eligible.size == 0:
        raise ValueError("A3 screen contains no eligible Pick train/development rows")
    order = np.lexsort((hazard[eligible], instance[eligible]))
    eligible = eligible[order]
    if max_instances is not None:
        if int(max_instances) <= 0:
            raise ValueError("max_instances must be positive")
        selected_instances = np.unique(instance[eligible])[: int(max_instances)]
        eligible = eligible[np.isin(instance[eligible], selected_instances)]
    if max_rows is not None:
        if int(max_rows) <= 0:
            raise ValueError("max_rows must be positive")
        eligible = eligible[: int(max_rows)]
    return eligible.astype(np.int64, copy=False)


def _record_array(record: FrameRecord, name: str) -> np.ndarray:
    value = np.asarray(getattr(record, name))
    if not np.isfinite(value).all():
        raise ValueError(f"P1 record field {name} contains non-finite values")
    return value


def build_raw_adjacent_policy_batch(
    record_pairs: Sequence[Sequence[FrameRecord]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build exact P1 H=2 batches from raw-adjacent frame pairs."""

    pairs = [tuple(pair) for pair in record_pairs]
    if not pairs or any(len(pair) != P1_HISTORY for pair in pairs):
        raise ValueError("each P1 example must contain raw-adjacent [t-1,t]")
    keys = {
        "rgb": "rgb",
        "obs": "proprio",
        "q_nominal": "q_nominal",
        "keypoints_xyz": "keypoints_xyz",
    }
    result: dict[str, torch.Tensor] = {}
    for output_name, record_name in keys.items():
        value = np.stack(
            [np.stack([_record_array(record, record_name) for record in pair]) for pair in pairs]
        ).astype(np.float32, copy=False)
        result[output_name] = torch.from_numpy(np.ascontiguousarray(value)).to(device)
    return result


def select_p1_visible_grounding(
    batch: Mapping[str, torch.Tensor],
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    """Select the visible anchor measurement while keeping one P1 policy.

    The frozen policy was trained in FCPG mode.  Its compact runtime interface
    accepts corrected points explicitly, so the matched nominal intervention
    replaces only those points and their reliability.  No oracle field is
    introduced and the original input mapping is left untouched.
    """

    if mode not in P1_VISIBLE_GROUNDING_MODES:
        raise ValueError(f"unsupported P1 visible grounding mode {mode!r}")
    result = dict(batch)
    if mode == "fcpg":
        return result
    nominal = result.get("q_nominal")
    if nominal is None:
        raise KeyError("nominal P1 grounding requires q_nominal")
    if nominal.ndim != 4 or nominal.shape[-1] != 2:
        raise ValueError("q_nominal must have shape [B,H,K,2]")
    result["fcpg_q_corrected"] = nominal
    result["fcpg_reliability"] = torch.ones_like(nominal[..., 0])
    return result


@torch.inference_mode()
def extract_raw_adjacent_p1(
    policy: nn.Module,
    record_pairs: Sequence[Sequence[FrameRecord]],
    *,
    device: torch.device,
    visible_grounding_mode: str = "fcpg",
) -> PolicyAnchorObservation:
    if int(getattr(policy, "obs_horizon", -1)) != P1_HISTORY:
        raise ValueError("P1 extraction requires obs_horizon=2")
    batch = build_raw_adjacent_policy_batch(record_pairs, device=device)
    return extract_policy_anchor_observation(
        policy,
        select_p1_visible_grounding(batch, mode=visible_grounding_mode),
    )


@dataclass(frozen=True)
class ReplayedBridgeRow:
    source_row_index: int
    branch_index: np.ndarray
    history_embedding: np.ndarray
    history_action: np.ndarray
    branch_action: np.ndarray
    factual_prediction: np.ndarray
    no_action_prediction: np.ndarray
    hidden_clean_a3_target: np.ndarray
    last_policy_token: np.ndarray
    last_sampled_visual: np.ndarray
    last_xyz: np.ndarray
    current_xyz: np.ndarray
    hidden_clean_policy_token: np.ndarray
    hidden_clean_sampled_visual: np.ndarray
    hidden_clean_q_corrected: np.ndarray
    hidden_clean_reliability: np.ndarray
    max_a3_history_reconstruction_error: float
    max_a3_target_reconstruction_error: float


def _require_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
    atol: float = A3_RECONSTRUCTION_ATOL,
) -> float:
    left = np.asarray(actual, dtype=np.float32)
    right = np.asarray(expected, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"{label} shape mismatch: {left.shape} vs {right.shape}")
    error = float(np.max(np.abs(left - right), initial=0.0))
    if not np.allclose(left, right, rtol=0.0, atol=float(atol)):
        raise RuntimeError(f"{label} reconstruction failed; max_abs_error={error:.8f}")
    return error


@torch.inference_mode()
def replay_one_a3_p1_row(
    arrays: Mapping[str, np.ndarray],
    row_index: int,
    *,
    runtime: EpisodeRuntime,
    anchor_encoder: Any,
    policy: nn.Module,
    predictors: Sequence[tuple[nn.Module, nn.Module]],
    branch_indices: Sequence[int],
    policy_device: torch.device,
    predictor_device: torch.device,
    p1_visible_grounding_mode: str = "fcpg",
    capture_snapshot: Callable[..., Any] = capture_branch_snapshot,
    restore_snapshot: Callable[[Any, Any], None] = restore_branch_snapshot,
) -> ReplayedBridgeRow:
    """Replay one exact A3 bundle and extract aligned A3/P1 bridge labels."""

    row = int(row_index)
    if int(np.asarray(arrays["task_index"])[row]) != PICK_TASK_INDEX:
        raise ValueError("strict A3-to-P1 bridge accepts Pick rows only")
    if str(np.asarray(arrays["split"])[row]) not in {"train", "development"}:
        raise ValueError("strict bridge refuses non-development A3 rows")
    chosen = np.asarray(tuple(int(value) for value in branch_indices), dtype=np.int64)
    if chosen.ndim != 1 or chosen.size == 0 or len(set(chosen.tolist())) != chosen.size:
        raise ValueError("branch_indices must be a non-empty unique sequence")
    if np.any((chosen < 0) | (chosen >= BRANCH_COUNT)):
        raise ValueError("branch_indices must lie in [0,7]")
    if not predictors:
        raise ValueError("at least one A3 predictor pair is required")

    hazard_step = int(np.asarray(arrays["hazard_id"])[row])
    hazard_raw = hazard_step * FRAME_SKIP
    history_action = np.asarray(arrays["history_action"])[row].astype(
        np.float32, copy=False
    )
    center_action = np.asarray(arrays["center_action"])[row].astype(
        np.float32, copy=False
    )
    cached_history = np.asarray(arrays["history_embedding"])[row].astype(
        np.float32, copy=False
    )
    cached_target = np.asarray(arrays["target_embedding"])[row, chosen].astype(
        np.float32, copy=False
    )
    branch_action = np.asarray(arrays["future_action"])[row, chosen].astype(
        np.float32, copy=False
    )
    raw_branch_action = branch_action.reshape(chosen.size, HORIZON * FRAME_SKIP, ACTION_DIM)
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if not np.array_equal(raw_branch_action, np.clip(raw_branch_action, low, high)):
        raise ValueError("cached A3 branch actions would be changed by clipping")

    needed = {hazard_raw - 2 * FRAME_SKIP, hazard_raw - FRAME_SKIP, hazard_raw - 1, hazard_raw}
    if min(needed) < 0:
        raise ValueError("A3 hazard lacks complete A3/P1 history")
    records: dict[int, FrameRecord] = {}
    source_actions: list[np.ndarray] = []
    observation = np.asarray(runtime.observation, dtype=np.float32)
    snapshot: Any | None = None
    source_observation: np.ndarray | None = None
    final_raw = hazard_raw + HORIZON * FRAME_SKIP
    for raw_step in range(final_raw + 1):
        if raw_step in needed:
            records[raw_step] = runtime.record_current(observation)
        if raw_step == hazard_raw:
            snapshot = capture_snapshot(runtime.env, include_process_rng=True)
            source_observation = observation.copy()
        if raw_step == final_raw:
            break
        action = np.asarray(runtime.expert.get_action(observation), dtype=np.float32)
        action = np.clip(action, low, high).astype(np.float32)
        source_actions.append(action.copy())
        observation, _, terminated, truncated, _ = runtime.env.step(action)
        observation = np.asarray(observation, dtype=np.float32)
        if (terminated or truncated) and raw_step + 1 < final_raw:
            raise RuntimeError("A3 source episode ended before cached center validation")
    if snapshot is None or source_observation is None:
        raise RuntimeError("failed to capture A3 source snapshot")
    validate_reconstructed_source_actions(
        history_action=history_action,
        center_action=center_action,
        hazard_step=hazard_step,
        source_actions=np.asarray(source_actions, dtype=np.float32),
    )

    history_records = [
        records[hazard_raw - 2 * FRAME_SKIP],
        records[hazard_raw - FRAME_SKIP],
        records[hazard_raw],
    ]
    rebuilt_history = np.asarray(anchor_encoder.encode(history_records), dtype=np.float32)
    history_error = _require_close(
        rebuilt_history, cached_history, label="A3 history_embedding"
    )
    source_pair = ((records[hazard_raw - 1], records[hazard_raw]),)
    source_p1 = extract_raw_adjacent_p1(
        policy,
        source_pair,
        device=policy_device,
        visible_grounding_mode=p1_visible_grounding_mode,
    )

    endpoint_records: list[FrameRecord] = []
    endpoint_pairs: list[tuple[FrameRecord, FrameRecord]] = []
    for actions in raw_branch_action:
        restore_snapshot(runtime.env, snapshot)
        branch_observation = source_observation.copy()
        raw_records: list[FrameRecord] = []
        for raw_index, action in enumerate(actions, start=1):
            branch_observation, _, terminated, truncated, _ = runtime.env.step(action)
            branch_observation = np.asarray(branch_observation, dtype=np.float32)
            raw_records.append(runtime.record_current(branch_observation))
            if raw_index % FRAME_SKIP == 0:
                endpoint_records.append(raw_records[-1])
                endpoint_pairs.append((raw_records[-2], raw_records[-1]))
            if (terminated or truncated) and raw_index < actions.shape[0]:
                raise RuntimeError("A3 cached branch ended before H8")
    expected_endpoints = chosen.size * HORIZON
    if len(endpoint_records) != expected_endpoints:
        raise RuntimeError("A3-to-P1 replay did not produce every H1-H8 endpoint")

    # Match the frozen A3 collector's original per-branch H8 batching.  A
    # single 64-frame call changes CUDA FP16 reduction scheduling needlessly.
    rebuilt_target = np.stack(
        [
            np.asarray(
                anchor_encoder.encode(
                    endpoint_records[
                        branch * HORIZON : (branch + 1) * HORIZON
                    ]
                ),
                dtype=np.float32,
            )
            for branch in range(chosen.size)
        ]
    )
    target_error = _require_close(
        rebuilt_target, cached_target, label="A3 target_embedding"
    )
    endpoint_p1 = extract_raw_adjacent_p1(
        policy,
        endpoint_pairs,
        device=policy_device,
        visible_grounding_mode=p1_visible_grounding_mode,
    )
    endpoint_shape = (chosen.size, HORIZON, ANCHOR_COUNT)

    history_tensor = torch.from_numpy(cached_history[None]).to(predictor_device)
    history_action_tensor = torch.from_numpy(history_action[None]).to(predictor_device)
    future_action_tensor = torch.from_numpy(branch_action[None]).to(predictor_device)
    factual_rows: list[np.ndarray] = []
    no_action_rows: list[np.ndarray] = []
    for factual, no_action in predictors:
        factual_rows.append(
            rollout_action_branches(
                factual, history_tensor, history_action_tensor, future_action_tensor
            )[0].float().cpu().numpy()
        )
        no_action_rows.append(
            rollout_action_branches(
                no_action, history_tensor, history_action_tensor, future_action_tensor
            )[0].float().cpu().numpy()
        )

    return ReplayedBridgeRow(
        source_row_index=row,
        branch_index=chosen.copy(),
        history_embedding=cached_history.copy(),
        history_action=history_action.copy(),
        branch_action=branch_action.copy(),
        factual_prediction=np.asarray(factual_rows, dtype=np.float32),
        no_action_prediction=np.asarray(no_action_rows, dtype=np.float32),
        hidden_clean_a3_target=rebuilt_target.copy(),
        last_policy_token=source_p1.native_token[0, -1].float().cpu().numpy(),
        last_sampled_visual=source_p1.sampled_visual[0, -1].float().cpu().numpy(),
        last_xyz=source_p1.xyz[0, -1].float().cpu().numpy(),
        current_xyz=endpoint_p1.xyz[:, -1]
        .reshape(*endpoint_shape, 3)
        .float()
        .cpu()
        .numpy(),
        hidden_clean_policy_token=endpoint_p1.native_token[:, -1]
        .reshape(*endpoint_shape, P1_TOKEN_DIM)
        .float()
        .cpu()
        .numpy(),
        hidden_clean_sampled_visual=endpoint_p1.sampled_visual[:, -1]
        .reshape(*endpoint_shape, P1_VISUAL_DIM)
        .float()
        .cpu()
        .numpy(),
        hidden_clean_q_corrected=endpoint_p1.points[:, -1]
        .reshape(*endpoint_shape, 2)
        .float()
        .cpu()
        .numpy(),
        hidden_clean_reliability=endpoint_p1.reliability[:, -1]
        .reshape(*endpoint_shape)
        .float()
        .cpu()
        .numpy(),
        max_a3_history_reconstruction_error=history_error,
        max_a3_target_reconstruction_error=target_error,
    )


def stack_replayed_bridge_rows(
    rows: Sequence[ReplayedBridgeRow],
    arrays: Mapping[str, np.ndarray],
    *,
    predictor_seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    values = list(rows)
    if not values:
        raise ValueError("bridge collection produced no rows")
    branch_index = values[0].branch_index
    if any(not np.array_equal(row.branch_index, branch_index) for row in values):
        raise ValueError("all bridge rows must use the same branch subset")
    source_rows = np.asarray([row.source_row_index for row in values], dtype=np.int64)
    result = {
        "source_row_index": source_rows,
        "task_index": np.asarray(arrays["task_index"])[source_rows].astype(np.int64),
        "task_instance_index": np.asarray(arrays["task_instance_index"])[source_rows].astype(np.int64),
        "source_episode_id": np.asarray(arrays["source_episode_id"])[source_rows].astype(str),
        "hazard_id": np.asarray(arrays["hazard_id"])[source_rows].astype(np.int64),
        "split": np.asarray(arrays["split"])[source_rows].astype(str),
        "predictor_seed": np.asarray(tuple(predictor_seeds), dtype=np.int64),
        "branch_index": branch_index.astype(np.int64),
    }
    fields = (
        "history_embedding",
        "history_action",
        "branch_action",
        "hidden_clean_a3_target",
        "last_policy_token",
        "last_sampled_visual",
        "last_xyz",
        "current_xyz",
        "hidden_clean_policy_token",
        "hidden_clean_sampled_visual",
        "hidden_clean_q_corrected",
        "hidden_clean_reliability",
    )
    for name in fields:
        result[name] = np.stack([np.asarray(getattr(row, name)) for row in values])
    # Prediction seed is the leading axis, ahead of source row.
    result["factual_prediction"] = np.stack(
        [row.factual_prediction for row in values], axis=1
    )
    result["no_action_prediction"] = np.stack(
        [row.no_action_prediction for row in values], axis=1
    )
    result["max_a3_history_reconstruction_error"] = np.asarray(
        [row.max_a3_history_reconstruction_error for row in values], dtype=np.float64
    )
    result["max_a3_target_reconstruction_error"] = np.asarray(
        [row.max_a3_target_reconstruction_error for row in values], dtype=np.float64
    )
    validate_bridge_arrays(result)
    return result


def validate_bridge_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate the strict structured A3-to-P1 NPZ schema."""

    required = {
        "source_row_index",
        "task_index",
        "task_instance_index",
        "source_episode_id",
        "hazard_id",
        "split",
        "predictor_seed",
        "branch_index",
        "history_embedding",
        "history_action",
        "branch_action",
        "factual_prediction",
        "no_action_prediction",
        "hidden_clean_a3_target",
        "last_policy_token",
        "last_sampled_visual",
        "last_xyz",
        "current_xyz",
        "hidden_clean_policy_token",
        "hidden_clean_sampled_visual",
        "hidden_clean_q_corrected",
        "hidden_clean_reliability",
        "max_a3_history_reconstruction_error",
        "max_a3_target_reconstruction_error",
    }
    if set(arrays) != required:
        raise ValueError(f"bridge arrays must have exact keys {sorted(required)}")
    n = int(np.asarray(arrays["source_row_index"]).size)
    s = int(np.asarray(arrays["predictor_seed"]).size)
    b = int(np.asarray(arrays["branch_index"]).size)
    if min(n, s, b) <= 0:
        raise ValueError("bridge row, predictor, and branch axes must be non-empty")
    expected = {
        "source_row_index": (n,),
        "task_index": (n,),
        "task_instance_index": (n,),
        "source_episode_id": (n,),
        "hazard_id": (n,),
        "split": (n,),
        "predictor_seed": (s,),
        "branch_index": (b,),
        "history_embedding": (n, HISTORY_SIZE, ANCHOR_COUNT, LATENT_DIM),
        "history_action": (n, HISTORY_SIZE - 1, FRAME_SKIP * ACTION_DIM),
        "branch_action": (n, b, HORIZON, FRAME_SKIP * ACTION_DIM),
        "factual_prediction": (s, n, b, HORIZON, ANCHOR_COUNT, LATENT_DIM),
        "no_action_prediction": (s, n, b, HORIZON, ANCHOR_COUNT, LATENT_DIM),
        "hidden_clean_a3_target": (n, b, HORIZON, ANCHOR_COUNT, LATENT_DIM),
        "last_policy_token": (n, ANCHOR_COUNT, P1_TOKEN_DIM),
        "last_sampled_visual": (n, ANCHOR_COUNT, P1_VISUAL_DIM),
        "last_xyz": (n, ANCHOR_COUNT, 3),
        "current_xyz": (n, b, HORIZON, ANCHOR_COUNT, 3),
        "hidden_clean_policy_token": (n, b, HORIZON, ANCHOR_COUNT, P1_TOKEN_DIM),
        "hidden_clean_sampled_visual": (n, b, HORIZON, ANCHOR_COUNT, P1_VISUAL_DIM),
        "hidden_clean_q_corrected": (n, b, HORIZON, ANCHOR_COUNT, 2),
        "hidden_clean_reliability": (n, b, HORIZON, ANCHOR_COUNT),
        "max_a3_history_reconstruction_error": (n,),
        "max_a3_target_reconstruction_error": (n,),
    }
    for name, shape in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"bridge {name} must have shape {shape}, got {value.shape}")
        if value.dtype.kind in "fciu" and not np.isfinite(value).all():
            raise ValueError(f"bridge {name} contains non-finite values")
    if not np.all(np.asarray(arrays["task_index"]) == PICK_TASK_INDEX):
        raise ValueError("bridge task_index must be Pick only")
    if not set(np.asarray(arrays["split"]).astype(str)).issubset(
        {"train", "development"}
    ):
        raise ValueError("bridge output may contain train/development only")
    if tuple(sorted(np.asarray(arrays["predictor_seed"]).tolist())) != tuple(
        sorted(STRICT_A3_10K_SEEDS)
    ):
        raise ValueError("bridge output must contain all three strict A3 10k seeds")
    branches = np.asarray(arrays["branch_index"], dtype=np.int64)
    if len(set(branches.tolist())) != b or np.any((branches < 0) | (branches >= 8)):
        raise ValueError("bridge branch_index is invalid")


__all__ = [
    "PICK_TASK",
    "PICK_TASK_INDEX",
    "P1_VISIBLE_GROUNDING_MODES",
    "STRICT_A3_10K_SEEDS",
    "STRICT_P1_CORRECTOR_SHA256",
    "STRICT_P1_POLICY_SHA256",
    "ReplayedBridgeRow",
    "build_raw_adjacent_policy_batch",
    "select_p1_visible_grounding",
    "extract_raw_adjacent_p1",
    "replay_one_a3_p1_row",
    "select_pick_bridge_rows",
    "sha256_file",
    "stack_replayed_bridge_rows",
    "validate_a3_10k_checkpoint",
    "validate_bridge_arrays",
    "validate_strict_p1_checkpoint",
]
