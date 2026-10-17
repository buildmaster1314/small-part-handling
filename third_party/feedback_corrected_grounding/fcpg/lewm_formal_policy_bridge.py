"""Fail-closed data, training, and checkpoint contracts for the formal bridge.

The formal experiment translates one *single-seed* A3 anchor prediction into
the frozen P1 policy-token space.  Three A3 checkpoints are correlated model
replicates: a shared bridge is applied to every replicate independently and
replicate losses are averaged.  Predicted embeddings are never ensembled
before the bridge, because deployment loads one A3 checkpoint at a time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_anchor_policy_bridge import (
    ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE,
    ResidualAnchorPolicyBridge,
    build_matched_bridge_pair,
)
from fcpg.lewm_policy_token_bridge import sha256_file


FORMAL_BRIDGE_CACHE_TYPE: Final = "fcpg_formal_anchor_policy_bridge_cache_v1"
FORMAL_BRIDGE_SCHEMA_VERSION: Final = 1
FORMAL_BRIDGE_SEMANTICS: Final = "last_valid_residual_v1"
FORMAL_BRIDGE_ARRAY_FILE: Final = "samples.npz"
FORMAL_BRIDGE_METADATA_FILE: Final = "metadata.json"
FORMAL_BRIDGE_HORIZONS: Final = tuple(range(1, 9))
FORMAL_BRIDGE_REPLICATES: Final = 3
FORMAL_A3_REPLICATE_SEEDS: Final = (29301, 29302, 29303)
FORMAL_BRIDGE_ANCHORS: Final = 3
FORMAL_BRIDGE_LATENT_DIM: Final = 192
FORMAL_BRIDGE_TOKEN_DIM: Final = 256
A3_FORMAL_CHECKPOINT_TYPE: Final = "fcpg_matched_anchor_dynamics_pair_v1"
A3_FORMAL_TRAINING_PROTOCOL: Final = "fcpg_action_branch_physical_support_a3_v1"
A3_FORMAL_CHECKPOINT_STEP: Final = 10_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_ARRAYS = (
    "factual_prediction",
    "matched_no_action_prediction",
    "target_oracle_embedding",
    "last_valid_embedding",
    "last_valid_token",
    "target_token",
    "geometry_hold_token",
    "last_valid_xyz",
    "current_xyz",
    "task_name",
    "physical_episode_sha256",
    "horizon",
)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _stable_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty stable string")
    return value


def _sha(value: object, *, label: str) -> str:
    result = _stable_text(value, label=label).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _artifact(
    value: object,
    *,
    label: str,
    verify_files: bool,
) -> dict[str, str]:
    item = _mapping(value, label=label)
    if set(item) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = Path(_stable_text(item["path"], label=f"{label}.path")).expanduser().resolve()
    digest = _sha(item["sha256"], label=f"{label}.sha256")
    if verify_files:
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        if sha256_file(path) != digest:
            raise ValueError(f"{label} SHA-256 mismatch")
    return {"path": str(path), "sha256": digest}


def validate_a3_formal_checkpoint_header(
    checkpoint: Mapping[str, Any],
    *,
    expected_seed: int,
) -> None:
    """Validate the real A3 10k paired-head checkpoint contract."""

    values = _mapping(checkpoint, label="A3 checkpoint")
    expected_scalars = {
        "checkpoint_type": A3_FORMAL_CHECKPOINT_TYPE,
        "training_protocol": A3_FORMAL_TRAINING_PROTOCOL,
        "schema_version": 1,
        "step": A3_FORMAL_CHECKPOINT_STEP,
        "completed": True,
        "selection": "fixed_latest",
        "seed": int(expected_seed),
    }
    for key, expected in expected_scalars.items():
        if values.get(key) != expected:
            raise ValueError(f"A3 checkpoint {key} must be exactly {expected!r}")
    for state_key in ("factual_state", "no_action_state"):
        state = values.get(state_key)
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"A3 checkpoint requires non-empty {state_key}")
    predictor = _mapping(values.get("predictor_config"), label="A3 predictor_config")
    expected_dimensions = {
        "latent_dim": FORMAL_BRIDGE_LATENT_DIM,
        "num_keypoints": FORMAL_BRIDGE_ANCHORS,
        "action_dim": 20,
        "history_size": 3,
    }
    for key, expected in expected_dimensions.items():
        if predictor.get(key) != expected:
            raise ValueError(f"A3 predictor_config.{key} must be {expected}")


def validate_formal_bridge_metadata(
    metadata: Mapping[str, Any],
    *,
    artifact_dir: str | Path,
    verify_upstream_files: bool = True,
) -> dict[str, Any]:
    """Validate one split manifest and return a canonical copy.

    P1 and all three A3 checkpoint digests are checked against files by
    default.  This prevents a collector cache from silently drifting away from
    the exact models whose predictions it contains.
    """

    values = _mapping(metadata, label="formal bridge metadata")
    required = {
        "schema_version",
        "cache_type",
        "bridge_semantics",
        "split",
        "arrays",
        "contract",
        "provenance",
    }
    if set(values) != required:
        raise ValueError(
            "formal bridge metadata keys must be exact; "
            f"missing={sorted(required.difference(values))}, "
            f"unexpected={sorted(set(values).difference(required))}"
        )
    if _integer(values["schema_version"], label="schema_version") != FORMAL_BRIDGE_SCHEMA_VERSION:
        raise ValueError("formal bridge schema_version mismatch")
    if values["cache_type"] != FORMAL_BRIDGE_CACHE_TYPE:
        raise ValueError("formal bridge cache_type mismatch")
    if values["bridge_semantics"] != FORMAL_BRIDGE_SEMANTICS:
        raise ValueError("formal bridge semantics mismatch")
    split = _stable_text(values["split"], label="split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")

    arrays = _mapping(values["arrays"], label="arrays")
    if set(arrays) != {"path", "sha256"}:
        raise ValueError("arrays must contain exactly path and sha256")
    relative = Path(_stable_text(arrays["path"], label="arrays.path"))
    if relative.is_absolute() or relative.name != FORMAL_BRIDGE_ARRAY_FILE:
        raise ValueError(f"arrays.path must be {FORMAL_BRIDGE_ARRAY_FILE!r}")
    array_path = Path(artifact_dir).expanduser().resolve() / relative
    array_digest = _sha(arrays["sha256"], label="arrays.sha256")
    if not array_path.is_file():
        raise FileNotFoundError(f"formal bridge array file is missing: {array_path}")
    if sha256_file(array_path) != array_digest:
        raise ValueError("formal bridge array SHA-256 mismatch")

    contract = _mapping(values["contract"], label="contract")
    expected_contract = {
        "num_keypoints": FORMAL_BRIDGE_ANCHORS,
        "latent_dim": FORMAL_BRIDGE_LATENT_DIM,
        "token_dim": FORMAL_BRIDGE_TOKEN_DIM,
        "replicate_count": FORMAL_BRIDGE_REPLICATES,
        "replicate_semantics": "correlated_single_seed_predictions",
        "horizons": list(FORMAL_BRIDGE_HORIZONS),
        "target_alignment": "future_pre_action",
    }
    if dict(contract) != expected_contract:
        raise ValueError("formal bridge contract does not match the frozen H1-H8 schema")

    provenance = _mapping(values["provenance"], label="provenance")
    if set(provenance) != {"p1_policy", "p1_corrector", "a3_replicates", "collector"}:
        raise ValueError("provenance must bind P1 policy/corrector, A3 replicates, and collector")
    p1_policy = _artifact(
        provenance["p1_policy"], label="p1_policy", verify_files=verify_upstream_files
    )
    p1_corrector = _artifact(
        provenance["p1_corrector"], label="p1_corrector", verify_files=verify_upstream_files
    )
    replicates = provenance["a3_replicates"]
    if isinstance(replicates, (str, bytes)) or not isinstance(replicates, Sequence):
        raise TypeError("a3_replicates must be an ordered sequence")
    if len(replicates) != FORMAL_BRIDGE_REPLICATES:
        raise ValueError("formal bridge requires exactly three A3 replicates")
    canonical_replicates: list[dict[str, Any]] = []
    seeds: list[int] = []
    digests: list[str] = []
    for index, raw in enumerate(replicates):
        item = _mapping(raw, label=f"a3_replicates[{index}]")
        if set(item) != {
            "seed",
            "checkpoint",
            "factual_state_key",
            "matched_no_action_state_key",
        }:
            raise ValueError(f"a3_replicates[{index}] has an invalid schema")
        seed = _integer(item["seed"], label=f"a3_replicates[{index}].seed")
        checkpoint = _artifact(
            item["checkpoint"],
            label=f"a3_replicates[{index}].checkpoint",
            verify_files=verify_upstream_files,
        )
        if item["factual_state_key"] != "factual_state":
            raise ValueError("A3 factual state key must be factual_state")
        if item["matched_no_action_state_key"] != "no_action_state":
            raise ValueError("A3 matched no-action state key must be no_action_state")
        if verify_upstream_files:
            payload = torch.load(
                checkpoint["path"], map_location="cpu", weights_only=False
            )
            validate_a3_formal_checkpoint_header(
                _mapping(payload, label=f"A3 checkpoint seed {seed}"),
                expected_seed=seed,
            )
        seeds.append(seed)
        digests.append(checkpoint["sha256"])
        canonical_replicates.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "factual_state_key": "factual_state",
                "matched_no_action_state_key": "no_action_state",
            }
        )
    if tuple(seeds) != FORMAL_A3_REPLICATE_SEEDS:
        raise ValueError(
            "A3 replicate seed axis must be exactly "
            f"{FORMAL_A3_REPLICATE_SEEDS}, got {tuple(seeds)}"
        )
    if len(set(digests)) != len(digests):
        raise ValueError("A3 replicate checkpoint digests must be unique")
    collector = _mapping(provenance["collector"], label="collector")
    if set(collector) != {"implementation_sha256", "source_data_sha256"}:
        raise ValueError("collector provenance keys must be exact")
    canonical_collector = {
        "implementation_sha256": _sha(
            collector["implementation_sha256"], label="collector.implementation_sha256"
        ),
        "source_data_sha256": _sha(
            collector["source_data_sha256"], label="collector.source_data_sha256"
        ),
    }
    return {
        "schema_version": FORMAL_BRIDGE_SCHEMA_VERSION,
        "cache_type": FORMAL_BRIDGE_CACHE_TYPE,
        "bridge_semantics": FORMAL_BRIDGE_SEMANTICS,
        "split": split,
        "arrays": {"path": str(array_path), "sha256": array_digest},
        "contract": expected_contract,
        "provenance": {
            "p1_policy": p1_policy,
            "p1_corrector": p1_corrector,
            "a3_replicates": canonical_replicates,
            "collector": canonical_collector,
        },
    }


@dataclass(frozen=True)
class FormalBridgeSplit:
    factual_prediction: np.ndarray
    matched_no_action_prediction: np.ndarray
    target_oracle_embedding: np.ndarray
    last_valid_embedding: np.ndarray
    last_valid_token: np.ndarray
    target_token: np.ndarray
    geometry_hold_token: np.ndarray
    last_valid_xyz: np.ndarray
    current_xyz: np.ndarray
    task_name: np.ndarray
    physical_episode_sha256: np.ndarray
    horizon: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.target_token.shape[0])

    @property
    def replicate_seeds(self) -> tuple[int, ...]:
        return tuple(
            int(item["seed"])
            for item in self.metadata["provenance"]["a3_replicates"]
        )


def _float_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if array.dtype.kind not in "fc":
        raise TypeError(f"{label} must be floating point")
    result = np.asarray(array, dtype=np.float32)
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{label} contains non-finite values")
    return result


def load_formal_bridge_split(
    artifact_dir: str | Path,
    *,
    verify_upstream_files: bool = True,
) -> FormalBridgeSplit:
    """Load one formal bridge split without permitting object arrays."""

    root = Path(artifact_dir).expanduser().resolve()
    metadata_path = root / FORMAL_BRIDGE_METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(f"formal bridge metadata is missing: {metadata_path}")
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = validate_formal_bridge_metadata(
        _mapping(raw_metadata, label="formal bridge metadata"),
        artifact_dir=root,
        verify_upstream_files=verify_upstream_files,
    )
    with np.load(metadata["arrays"]["path"], allow_pickle=False) as archive:
        missing = set(_REQUIRED_ARRAYS).difference(archive.files)
        if missing:
            raise ValueError(f"formal bridge arrays are missing {sorted(missing)}")
        count = int(np.asarray(archive["horizon"]).shape[0])
        if count <= 0:
            raise ValueError("formal bridge split must not be empty")
        factual = _float_array(
            archive["factual_prediction"],
            shape=(count, FORMAL_BRIDGE_REPLICATES, FORMAL_BRIDGE_ANCHORS, FORMAL_BRIDGE_LATENT_DIM),
            label="factual_prediction",
        )
        no_action = _float_array(
            archive["matched_no_action_prediction"],
            shape=factual.shape,
            label="matched_no_action_prediction",
        )
        oracle = _float_array(
            archive["target_oracle_embedding"],
            shape=(count, FORMAL_BRIDGE_ANCHORS, FORMAL_BRIDGE_LATENT_DIM),
            label="target_oracle_embedding",
        )
        last_embedding = _float_array(
            archive["last_valid_embedding"], shape=oracle.shape, label="last_valid_embedding"
        )
        last_token = _float_array(
            archive["last_valid_token"],
            shape=(count, FORMAL_BRIDGE_ANCHORS, FORMAL_BRIDGE_TOKEN_DIM),
            label="last_valid_token",
        )
        target_token = _float_array(
            archive["target_token"], shape=last_token.shape, label="target_token"
        )
        geometry = _float_array(
            archive["geometry_hold_token"], shape=last_token.shape, label="geometry_hold_token"
        )
        last_xyz = _float_array(
            archive["last_valid_xyz"],
            shape=(count, FORMAL_BRIDGE_ANCHORS, 3),
            label="last_valid_xyz",
        )
        current_xyz = _float_array(
            archive["current_xyz"], shape=last_xyz.shape, label="current_xyz"
        )
        tasks = np.asarray(archive["task_name"]).astype("U")
        episodes = np.asarray(archive["physical_episode_sha256"]).astype("U")
        horizons = np.asarray(archive["horizon"])

    if tasks.shape != (count,) or episodes.shape != (count,) or horizons.shape != (count,):
        raise ValueError("task_name, physical_episode_sha256, and horizon must be row vectors")
    if any(not task for task in tasks.tolist()):
        raise ValueError("task_name contains an empty value")
    if any(_SHA256.fullmatch(value) is None for value in episodes.tolist()):
        raise ValueError("physical_episode_sha256 contains an invalid digest")
    if horizons.dtype.kind not in "iu":
        raise TypeError("horizon must be integer")
    horizons = horizons.astype(np.int64, copy=False)
    if set(horizons.tolist()) != set(FORMAL_BRIDGE_HORIZONS):
        raise ValueError("formal bridge split must cover exactly H1-H8")
    for episode in np.unique(episodes):
        episode_horizons = set(horizons[episodes == episode].tolist())
        if episode_horizons != set(FORMAL_BRIDGE_HORIZONS):
            raise ValueError(f"physical episode {episode} does not cover H1-H8")
        episode_tasks = set(tasks[episodes == episode].tolist())
        if len(episode_tasks) != 1:
            raise ValueError("one physical episode cannot cross task identities")
    if np.allclose(factual, factual[:, :1], rtol=0.0, atol=0.0) or np.allclose(
        no_action, no_action[:, :1], rtol=0.0, atol=0.0
    ):
        raise ValueError("formal bridge replicate axis must contain three independently predicted arrays")

    return FormalBridgeSplit(
        factual_prediction=factual,
        matched_no_action_prediction=no_action,
        target_oracle_embedding=oracle,
        last_valid_embedding=last_embedding,
        last_valid_token=last_token,
        target_token=target_token,
        geometry_hold_token=geometry,
        last_valid_xyz=last_xyz,
        current_xyz=current_xyz,
        task_name=tasks,
        physical_episode_sha256=episodes,
        horizon=horizons,
        metadata=metadata,
    )


def assert_formal_split_pair(
    train: FormalBridgeSplit,
    validation: FormalBridgeSplit,
) -> None:
    """Require physical disjointness and byte-identical upstream provenance."""

    if train.metadata["split"] != "train" or validation.metadata["split"] != "validation":
        raise ValueError("formal split pair must be train then validation")
    overlap = set(train.physical_episode_sha256).intersection(validation.physical_episode_sha256)
    if overlap:
        raise ValueError("formal train and validation physical episodes overlap")
    for key in ("p1_policy", "p1_corrector", "a3_replicates", "collector"):
        if train.metadata["provenance"][key] != validation.metadata["provenance"][key]:
            raise ValueError(f"formal train/validation provenance mismatch for {key}")
    if train.replicate_seeds != validation.replicate_seeds:
        raise ValueError("formal train/validation A3 replicate order differs")


def episode_horizon_row_weights(split: FormalBridgeSplit) -> np.ndarray:
    """Return weights that equalize episodes and horizons within episodes."""

    episodes = np.asarray(split.physical_episode_sha256)
    horizons = np.asarray(split.horizon)
    unique_episodes = np.unique(episodes)
    result = np.zeros(split.sample_count, dtype=np.float64)
    episode_mass = 1.0 / float(unique_episodes.size)
    for episode in unique_episodes:
        episode_mask = episodes == episode
        episode_horizons = np.unique(horizons[episode_mask])
        horizon_mass = episode_mass / float(episode_horizons.size)
        for horizon in episode_horizons:
            indices = np.flatnonzero(episode_mask & (horizons == horizon))
            result[indices] = horizon_mass / float(indices.size)
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("episode/horizon row weights do not sum to one")
    return result


def forward_bridge_replicates(
    bridge: ResidualAnchorPolicyBridge,
    predicted_embedding: torch.Tensor,
    last_valid_embedding: torch.Tensor,
    last_valid_token: torch.Tensor,
    current_xyz: torch.Tensor,
    last_valid_xyz: torch.Tensor,
) -> torch.Tensor:
    """Apply a shared bridge to each A3 replicate without latent ensembling."""

    if predicted_embedding.ndim != 4:
        raise ValueError("predicted_embedding must have shape [B,R,K,D]")
    batch, replicates, keypoints, latent_dim = predicted_embedding.shape
    if replicates != FORMAL_BRIDGE_REPLICATES:
        raise ValueError("formal bridge requires exactly three prediction replicates")
    if (keypoints, latent_dim) != (bridge.num_keypoints, bridge.latent_dim):
        raise ValueError("predicted embedding dimensions mismatch bridge")

    def repeat(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(-1, replicates, *value.shape[1:]).reshape(
            batch * replicates, *value.shape[1:]
        )

    result = bridge(
        predicted_embedding.reshape(batch * replicates, keypoints, latent_dim),
        repeat(last_valid_embedding),
        repeat(last_valid_token),
        repeat(current_xyz),
        repeat(last_valid_xyz),
    )
    return result.reshape(batch, replicates, keypoints, bridge.token_dim)


def _balanced_reduce(
    row_values: np.ndarray,
    split: FormalBridgeSplit,
    *,
    row_mask: np.ndarray | None = None,
) -> float:
    values = np.asarray(row_values, dtype=np.float64)
    if values.shape != (split.sample_count,):
        raise ValueError("row_values must have one scalar per formal row")
    mask = np.ones(split.sample_count, dtype=bool) if row_mask is None else np.asarray(row_mask, dtype=bool)
    if mask.shape != (split.sample_count,) or not bool(mask.any()):
        raise ValueError("row_mask must select at least one row")
    episodes = split.physical_episode_sha256
    horizons = split.horizon
    episode_values: list[float] = []
    for episode in np.unique(episodes[mask]):
        episode_mask = mask & (episodes == episode)
        horizon_values = [
            float(values[episode_mask & (horizons == horizon)].mean())
            for horizon in np.unique(horizons[episode_mask])
        ]
        episode_values.append(float(np.mean(horizon_values)))
    return float(np.mean(episode_values))


def formal_token_metrics(
    prediction: np.ndarray,
    split: FormalBridgeSplit,
) -> dict[str, Any]:
    """Compute task/episode/horizon/anchor-equal MSE, preserving seed axis."""

    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim == 3:
        values = values[:, None]
    expected = (
        split.sample_count,
        values.shape[1],
        FORMAL_BRIDGE_ANCHORS,
        FORMAL_BRIDGE_TOKEN_DIM,
    )
    if values.shape != expected:
        raise ValueError(f"prediction has invalid shape {values.shape}")
    squared = np.square(values - split.target_token[:, None]).mean(axis=-1).mean(axis=-1)
    tasks = tuple(sorted(set(split.task_name.tolist())))
    per_replicate: list[dict[str, Any]] = []
    for replicate in range(values.shape[1]):
        per_task = {
            task: _balanced_reduce(
                squared[:, replicate], split, row_mask=split.task_name == task
            )
            for task in tasks
        }
        per_horizon_task_macro: dict[str, float] = {}
        per_horizon_per_task: dict[str, dict[str, float]] = {}
        for horizon in FORMAL_BRIDGE_HORIZONS:
            horizon_per_task = {
                task: _balanced_reduce(
                    squared[:, replicate],
                    split,
                    row_mask=(split.task_name == task) & (split.horizon == horizon),
                )
                for task in tasks
            }
            key = f"H{horizon}"
            per_horizon_per_task[key] = horizon_per_task
            per_horizon_task_macro[key] = float(
                np.mean(list(horizon_per_task.values()))
            )
        per_replicate.append(
            {
                "replicate_index": replicate,
                "task_macro_mse": float(np.mean(list(per_task.values()))),
                "per_task_mse": per_task,
                "per_horizon_task_macro_mse": per_horizon_task_macro,
                "per_horizon_per_task_mse": per_horizon_per_task,
            }
        )
    return {
        "replicate_count": values.shape[1],
        "replicates": per_replicate,
        "replicate_macro_mse": float(
            np.mean([item["task_macro_mse"] for item in per_replicate])
        ),
    }


def formal_episode_mse(
    prediction: np.ndarray,
    split: FormalBridgeSplit,
) -> dict[str, Any]:
    """Reduce tokens to one H1-H8-balanced MSE per physical episode/replicate."""

    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim == 3:
        values = values[:, None]
    if values.ndim != 4 or values.shape[0] != split.sample_count:
        raise ValueError("prediction must have shape [N,R,K,P] or [N,K,P]")
    if values.shape[2:] != (FORMAL_BRIDGE_ANCHORS, FORMAL_BRIDGE_TOKEN_DIM):
        raise ValueError("prediction anchor/token dimensions are invalid")
    row_mse = np.square(values - split.target_token[:, None]).mean(axis=-1).mean(axis=-1)
    episodes = tuple(sorted(set(split.physical_episode_sha256.tolist())))
    episode_horizon_mse = np.empty(
        (len(episodes), len(FORMAL_BRIDGE_HORIZONS), values.shape[1]),
        dtype=np.float64,
    )
    tasks: list[str] = []
    for episode_index, episode in enumerate(episodes):
        episode_mask = split.physical_episode_sha256 == episode
        episode_tasks = set(split.task_name[episode_mask].tolist())
        if len(episode_tasks) != 1:
            raise ValueError("physical episode crosses task identities")
        tasks.append(next(iter(episode_tasks)))
        horizon_values = [
            row_mse[episode_mask & (split.horizon == horizon)].mean(axis=0)
            for horizon in FORMAL_BRIDGE_HORIZONS
        ]
        episode_horizon_mse[episode_index] = np.stack(horizon_values, axis=0)
    return {
        "physical_episode_sha256": np.asarray(episodes, dtype="U64"),
        "task_name": np.asarray(tasks, dtype="U64"),
        "mse": episode_horizon_mse.mean(axis=1),
        "horizon_mse": episode_horizon_mse,
    }


def paired_episode_bootstrap(
    candidate_episode_mse: np.ndarray,
    reference_episode_mse: np.ndarray,
    task_name: np.ndarray,
    *,
    seed: int = 0,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Task-stratified paired bootstrap with physical episode as the unit.

    A3 replicate predictions must already be averaged *within* each episode
    before calling this function.  Consequently three model seeds never become
    three times as many statistical observations.
    """

    candidate = np.asarray(candidate_episode_mse, dtype=np.float64)
    reference = np.asarray(reference_episode_mse, dtype=np.float64)
    tasks = np.asarray(task_name).astype("U")
    if candidate.shape != reference.shape or candidate.shape != tasks.shape:
        raise ValueError("paired episode arrays must share shape [E]")
    if candidate.ndim != 1 or candidate.size == 0:
        raise ValueError("paired episode arrays must be non-empty vectors")
    if not bool(np.isfinite(candidate).all() and np.isfinite(reference).all()):
        raise ValueError("paired episode MSE contains non-finite values")
    if isinstance(replicates, bool) or replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    unique_tasks = tuple(sorted(set(tasks.tolist())))

    def macro(indices_by_task: list[np.ndarray]) -> tuple[float, float, float]:
        candidate_task = [float(candidate[index].mean()) for index in indices_by_task]
        reference_task = [float(reference[index].mean()) for index in indices_by_task]
        candidate_macro = float(np.mean(candidate_task))
        reference_macro = float(np.mean(reference_task))
        relative = (reference_macro - candidate_macro) / max(reference_macro, 1.0e-12)
        return candidate_macro, reference_macro, float(relative)

    task_indices = [np.flatnonzero(tasks == task) for task in unique_tasks]
    point_candidate, point_reference, point_relative = macro(task_indices)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for bootstrap_index in range(replicates):
        drawn = [
            rng.choice(indices, size=indices.size, replace=True)
            for indices in task_indices
        ]
        samples[bootstrap_index] = macro(drawn)[2]
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "scientific_unit": "physical_episode",
        "a3_replicates_averaged_within_episode": True,
        "episode_count": int(candidate.size),
        "task_count": len(unique_tasks),
        "candidate_task_macro_mse": point_candidate,
        "reference_task_macro_mse": point_reference,
        "relative_improvement": point_relative,
        "relative_improvement_ci95": [float(low), float(high)],
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def make_formal_bridge_triplet(
    *,
    initialization_seed: int,
    latent_dim: int = FORMAL_BRIDGE_LATENT_DIM,
    token_dim: int = FORMAL_BRIDGE_TOKEN_DIM,
    num_keypoints: int = FORMAL_BRIDGE_ANCHORS,
    hidden_dim: int = 64,
) -> tuple[ResidualAnchorPolicyBridge, ResidualAnchorPolicyBridge, ResidualAnchorPolicyBridge]:
    """Create factual, no-action, and oracle bridges with identical parameters."""

    factual, no_action = build_matched_bridge_pair(
        initialization_seed=initialization_seed,
        latent_dim=latent_dim,
        token_dim=token_dim,
        num_keypoints=num_keypoints,
        hidden_dim=hidden_dim,
    )
    return factual, no_action, deepcopy(factual)


def build_formal_bridge_checkpoint(
    *,
    factual: ResidualAnchorPolicyBridge,
    no_action: ResidualAnchorPolicyBridge,
    target_oracle: ResidualAnchorPolicyBridge,
    train: FormalBridgeSplit,
    validation: FormalBridgeSplit,
    training_config: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a runtime-compatible checkpoint plus formal diagnostic state."""

    assert_formal_split_pair(train, validation)
    dimensions = {
        (model.latent_dim, model.token_dim, model.num_keypoints, model.hidden_dim)
        for model in (factual, no_action, target_oracle)
    }
    if len(dimensions) != 1:
        raise ValueError("formal bridges do not share an identical architecture")
    latent_dim, token_dim, num_keypoints, hidden_dim = dimensions.pop()

    def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    upstream = train.metadata["provenance"]
    compatibility = {
        "p1_policy_checkpoint_sha256": upstream["p1_policy"]["sha256"],
        "p1_corrector_checkpoint_sha256": upstream["p1_corrector"]["sha256"],
        "a3_replicates": [
            {
                "seed": item["seed"],
                "checkpoint_sha256": item["checkpoint"]["sha256"],
            }
            for item in upstream["a3_replicates"]
        ],
        "single_seed_runtime": True,
        "training_replicate_reduction": "mean_loss_after_independent_forward",
    }
    return {
        "schema_version": FORMAL_BRIDGE_SCHEMA_VERSION,
        "checkpoint_type": ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE,
        "bridge_semantics": FORMAL_BRIDGE_SEMANTICS,
        "bridge_config": {
            "latent_dim": latent_dim,
            "token_dim": token_dim,
            "num_keypoints": num_keypoints,
            "hidden_dim": hidden_dim,
        },
        "bridge_states": {
            "factual": cpu_state(factual),
            "no_action": cpu_state(no_action),
        },
        "diagnostic_bridge_states": {"target_oracle": cpu_state(target_oracle)},
        "compatibility": compatibility,
        "training_config": dict(training_config),
        "source_caches": {
            "train_arrays_sha256": train.metadata["arrays"]["sha256"],
            "validation_arrays_sha256": validation.metadata["arrays"]["sha256"],
        },
        "report": dict(report),
    }


def load_formal_bridge_for_runtime(
    checkpoint_path: str | Path,
    *,
    bridge_name: str,
    policy_checkpoint: str | Path,
    corrector_checkpoint: str | Path,
    dynamics_checkpoint: str | Path,
    dynamics_seed: int,
    device: str | torch.device = "cpu",
) -> tuple[ResidualAnchorPolicyBridge, dict[str, Any]]:
    """Load one shared bridge only after exact P1/A3 compatibility checks."""

    if bridge_name not in {"factual", "no_action"}:
        raise ValueError("bridge_name must be factual or no_action")
    payload = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=torch.device(device),
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("formal bridge checkpoint must contain a mapping")
    if payload.get("checkpoint_type") != ANCHOR_POLICY_BRIDGE_CHECKPOINT_TYPE:
        raise ValueError("formal bridge checkpoint type mismatch")
    if payload.get("bridge_semantics") != FORMAL_BRIDGE_SEMANTICS:
        raise ValueError("formal bridge checkpoint semantics mismatch")
    compatibility = _mapping(payload.get("compatibility"), label="compatibility")
    if compatibility.get("single_seed_runtime") is not True:
        raise ValueError("formal checkpoint does not guarantee single-seed runtime")
    expected = {
        "p1_policy_checkpoint_sha256": sha256_file(
            Path(policy_checkpoint).expanduser().resolve()
        ),
        "p1_corrector_checkpoint_sha256": sha256_file(
            Path(corrector_checkpoint).expanduser().resolve()
        ),
    }
    for key, value in expected.items():
        if compatibility.get(key) != value:
            raise ValueError(f"formal bridge {key} mismatch")
    replicates = compatibility.get("a3_replicates")
    if not isinstance(replicates, Sequence):
        raise TypeError("formal checkpoint a3_replicates must be a sequence")
    dynamics_sha = sha256_file(Path(dynamics_checkpoint).expanduser().resolve())
    matches = [
        item
        for item in replicates
        if isinstance(item, Mapping) and int(item.get("seed", -1)) == int(dynamics_seed)
    ]
    if len(matches) != 1 or matches[0].get("checkpoint_sha256") != dynamics_sha:
        raise ValueError("runtime A3 seed/checkpoint is not bound by formal bridge")

    config = _mapping(payload.get("bridge_config"), label="bridge_config")
    states = _mapping(payload.get("bridge_states"), label="bridge_states")
    if set(states) != {"factual", "no_action"}:
        raise ValueError("formal bridge checkpoint requires factual and no_action states")
    state = _mapping(states[bridge_name], label=f"bridge_states[{bridge_name!r}]")
    model = ResidualAnchorPolicyBridge(**dict(config)).to(torch.device(device))
    model.load_state_dict(dict(state), strict=True)
    model.requires_grad_(False).eval()
    return model, dict(payload)


__all__ = [
    "A3_FORMAL_CHECKPOINT_STEP",
    "A3_FORMAL_CHECKPOINT_TYPE",
    "A3_FORMAL_TRAINING_PROTOCOL",
    "FORMAL_BRIDGE_ANCHORS",
    "FORMAL_BRIDGE_ARRAY_FILE",
    "FORMAL_BRIDGE_CACHE_TYPE",
    "FORMAL_BRIDGE_HORIZONS",
    "FORMAL_A3_REPLICATE_SEEDS",
    "FORMAL_BRIDGE_LATENT_DIM",
    "FORMAL_BRIDGE_METADATA_FILE",
    "FORMAL_BRIDGE_REPLICATES",
    "FORMAL_BRIDGE_SCHEMA_VERSION",
    "FORMAL_BRIDGE_SEMANTICS",
    "FORMAL_BRIDGE_TOKEN_DIM",
    "FormalBridgeSplit",
    "assert_formal_split_pair",
    "build_formal_bridge_checkpoint",
    "episode_horizon_row_weights",
    "formal_token_metrics",
    "formal_episode_mse",
    "forward_bridge_replicates",
    "load_formal_bridge_for_runtime",
    "load_formal_bridge_split",
    "make_formal_bridge_triplet",
    "paired_episode_bootstrap",
    "validate_formal_bridge_metadata",
    "validate_a3_formal_checkpoint_header",
]
