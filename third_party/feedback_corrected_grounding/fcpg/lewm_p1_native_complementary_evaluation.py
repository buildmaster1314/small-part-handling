"""Strict evaluation helpers for the fresh P1 complementary-observer Gate.

The fresh collector stores eight branch-specific H0--H2 histories for every
physical episode.  Evaluation therefore flattens ``episode x branch`` and
gives the dynamics runtime exactly one branch per history.  Treating the eight
histories as one shared history would silently evaluate a different, invalid
counterfactual problem.

Offline target tokens are accepted only by the scoring helpers.  They are not
part of :class:`FlattenedDeploymentInputs`, which is the only structure passed
to the frozen predictors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from fcpg.lewm_counterfactual_runtime import (
    _validate_corrector_payload,
    _verify_loaded_corrector_binding,
    sha256_file,
    sha256_state_dict,
)
from fcpg.lewm_p1_native_complementary import (
    ComplementaryMatchedRollout,
    FrozenP1NativeCheckpointSpec,
    LoadedP1NativePair,
    P1_NATIVE_COMPLEMENTARY_GAINS,
    load_frozen_p1_native_checkpoint,
    recursive_complementary_matched_rollout,
    recursive_complementary_rollout,
)
from fcpg.lewm_p1_native_complementary_statistics import (
    CANDIDATE,
    GEOMETRY_HOLD,
    HOLD_LAST,
    HORIZONS,
    MATCHED_NO_ACTION,
    PROTOCOL as ROW_PROTOCOL,
    TASK,
    VISUAL_CV,
    WRONG_ACTIONS,
)
from fcpg.lewm_p1_native_dynamics import (
    P1_NATIVE_ACTION_DIM,
    P1_NATIVE_ANCHORS,
    P1_NATIVE_FUTURE_STEPS,
    P1_NATIVE_HISTORY_SIZE,
    P1_NATIVE_TOKEN_DIM,
    P1NativeBaselines,
    P1NativeDynamicsCache,
    reconstruct_p1_native_baselines,
)
from fcpg.lewm_p1_native_fresh_collection import (
    DEPLOYMENT_CACHE_KEYS,
    FRESH_CACHE_KEYS,
    OFFLINE_LABEL_CACHE_KEYS,
    PROTOCOL as FRESH_DATA_PROTOCOL,
    cache_contract_json,
    combine_fresh_cache_arrays,
    fresh_arrays_to_native_cache,
)


FORMAL_CONFIG_PROTOCOL: Final = "fcpg_p1_native_complementary_fresh_v1"
FORMAL_CLASSIFICATION: Final = "formal"
RAW_LEWM_METHOD: Final = "raw_lewm_factual"
COLLECTION_METADATA_SCHEMA_VERSION: Final = 1
COLLECTION_METADATA_REQUIRED_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol",
        "completed",
        "config_sha256",
        "deployment_cache",
        "label_cache",
        "dataset_sha256",
        "policy_checkpoint",
        "corrector_checkpoint",
        "legacy_observation_source",
        "freshness_receipt",
        "cache_contract_json",
    }
)


@dataclass(frozen=True)
class FreshArtifactProvenance:
    config_path: Path
    deployment_path: Path
    label_path: Path
    metadata_path: Path
    training_config_path: Path
    config_sha256: str
    deployment_cache_sha256: str
    label_cache_sha256: str
    metadata_sha256: str
    training_config_sha256: str
    dataset_sha256: str


@dataclass(frozen=True)
class LoadedFreshEvaluation:
    config: Mapping[str, Any]
    training_config: Mapping[str, Any]
    collection_metadata: Mapping[str, Any]
    cache: P1NativeDynamicsCache
    target_token: np.ndarray
    provenance: FreshArtifactProvenance


@dataclass(frozen=True)
class FlattenedDeploymentInputs:
    """Predictor inputs only; deliberately contains no target/offline label."""

    episode_count: int
    branch_count: int
    history_token: np.ndarray
    history_action: np.ndarray
    future_action: np.ndarray
    wrong_future_action: np.ndarray
    geometry_token: np.ndarray


@dataclass(frozen=True)
class SeedPredictions:
    factual: np.ndarray
    no_action: np.ndarray
    wrong_action: np.ndarray
    raw_factual: np.ndarray


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _resolve(value: object, *, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_file_sha(path: Path, expected: object, *, label: str) -> str:
    expected_sha = _sha256(expected, label=f"{label} expected SHA-256")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_sha:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha}, got {observed}"
        )
    return observed


def validate_formal_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Path, str]:
    """Validate the evaluator-relevant checked-in contract and source config."""

    if config.get("schema_version") != 1 or config.get("protocol") != FORMAL_CONFIG_PROTOCOL:
        raise ValueError("fresh complementary formal config identity drifted")
    scope = _mapping(config.get("scope"), label="config.scope")
    fresh = _mapping(config.get("fresh_data"), label="config.fresh_data")
    evaluation = _mapping(config.get("evaluation"), label="config.evaluation")
    observer = _mapping(config.get("observer"), label="config.observer")
    predictors = config.get("frozen_predictors")
    if not isinstance(predictors, list) or len(predictors) != 3:
        raise ValueError("formal config must bind exactly three frozen predictors")
    if (
        scope.get("task") != TASK
        or fresh.get("protocol") != FRESH_DATA_PROTOCOL
        or int(fresh.get("physical_episode_count", -1)) != 30
        or int(fresh.get("branch_count", -1)) != 8
        or tuple(int(value) for value in fresh.get("offline_label_horizons", ()))
        != HORIZONS
        or evaluation.get("row_protocol") != ROW_PROTOCOL
        or evaluation.get("statistical_unit") != "physical_episode"
        or evaluation.get("raw_lewm_is_separate_diagnostic") is not True
        or observer.get("posterior_is_written_back_before_next_prediction") is not True
        or tuple(int(value) for value in observer.get("wrong_action_shifts", ()))
        != tuple(range(1, 8))
    ):
        raise ValueError("fresh complementary evaluation contract drifted")
    gains = _mapping(
        observer.get("gain_by_missing_horizon"),
        label="config.observer.gain_by_missing_horizon",
    )
    if tuple(float(gains.get(f"H{horizon}", math.nan)) for horizon in HORIZONS) != tuple(
        P1_NATIVE_COMPLEMENTARY_GAINS
    ):
        raise ValueError("formal complementary gain schedule drifted")

    training = _mapping(
        config.get("frozen_predictor_training_contract"),
        label="config.frozen_predictor_training_contract",
    )
    source_path = _resolve(training.get("config"), root=repo_root, label="training config")
    source_sha = _validate_file_sha(
        source_path,
        training.get("config_sha256"),
        label="frozen predictor training config",
    )
    expected_seeds = (15801, 15802, 15803)
    actual_seeds = tuple(int(_mapping(item, label="frozen predictor")["seed"]) for item in predictors)
    if actual_seeds != expected_seeds:
        raise ValueError("formal config predictor order/seeds drifted")
    return source_path, source_sha


def _load_npz_exact(path: Path, *, expected_keys: frozenset[str], label: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        observed = set(archive.files)
        if observed != expected_keys:
            raise ValueError(
                f"{label} keys drifted: missing={sorted(expected_keys - observed)}, "
                f"extra={sorted(observed - expected_keys)}"
            )
        return {name: np.asarray(archive[name]) for name in archive.files}


def _artifact_record(
    metadata: Mapping[str, Any],
    name: str,
    *,
    expected_path: Path,
    expected_sha: str,
    repo_root: Path,
) -> None:
    record = _mapping(metadata.get(name), label=f"collection metadata {name}")
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"collection metadata {name} must contain exactly path and sha256")
    recorded_path = _resolve(record["path"], root=repo_root, label=f"metadata {name}.path")
    if recorded_path != expected_path:
        raise ValueError(f"collection metadata {name} path mismatch")
    if _sha256(record["sha256"], label=f"metadata {name}.sha256") != expected_sha:
        raise ValueError(f"collection metadata {name} SHA-256 mismatch")


def _validate_bound_source_record(
    metadata: Mapping[str, Any],
    name: str,
    *,
    expected_path: Path,
    expected_sha: object,
    repo_root: Path,
) -> None:
    _artifact_record(
        metadata,
        name,
        expected_path=expected_path,
        expected_sha=_sha256(expected_sha, label=f"config {name}.sha256"),
        repo_root=repo_root,
    )


def _validate_freshness_receipt(
    metadata: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    repo_root: Path,
) -> None:
    receipt = _mapping(
        metadata.get("freshness_receipt"), label="metadata.freshness_receipt"
    )
    expected_keys = {
        "legacy_training_reset_digest_status",
        "reset_digest_scope_completeness_claimed",
        "known_optional_reset_scope_counts",
        "observed_signature_scope_counts",
        "legacy_observed_state_inventory",
        "fresh_observed_signature_count",
        "exact_observed_signature_overlap_count",
    }
    if set(receipt) != expected_keys:
        raise ValueError("freshness receipt schema drifted")
    if (
        receipt["legacy_training_reset_digest_status"]
        != "unverified_missing_legacy_reset_metadata"
        or receipt["reset_digest_scope_completeness_claimed"] is not False
        or int(receipt["fresh_observed_signature_count"]) != 30
        or int(receipt["exact_observed_signature_overlap_count"]) != 0
    ):
        raise PermissionError("freshness receipt makes an invalid freshness claim")
    reset_counts = _mapping(
        receipt["known_optional_reset_scope_counts"],
        label="freshness optional-reset counts",
    )
    observed_counts = _mapping(
        receipt["observed_signature_scope_counts"],
        label="freshness observed-signature counts",
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in reset_counts.values()
    ):
        raise ValueError("freshness optional-reset counts are invalid")
    if (
        set(observed_counts) != {"p1_fcpg_training"}
        or isinstance(observed_counts["p1_fcpg_training"], bool)
        or not isinstance(observed_counts["p1_fcpg_training"], int)
        or observed_counts["p1_fcpg_training"] <= 0
    ):
        raise ValueError("freshness observed-signature inventory is incomplete")
    inventory = _mapping(
        receipt["legacy_observed_state_inventory"],
        label="freshness legacy observed-state inventory",
    )
    if set(inventory) != {"path", "sha256", "signature_count"}:
        raise ValueError("legacy observed-state inventory receipt schema drifted")
    freshness = _mapping(config["freshness"], label="config.freshness")
    inventory_path = _resolve(
        freshness["legacy_observed_state_inventory"],
        root=repo_root,
        label="legacy observed-state inventory",
    )
    recorded_path = _resolve(
        inventory["path"], root=repo_root, label="recorded observed-state inventory"
    )
    if recorded_path != inventory_path:
        raise ValueError("legacy observed-state inventory path mismatch")
    inventory_sha = _validate_file_sha(
        inventory_path,
        inventory["sha256"],
        label="legacy observed-state inventory",
    )
    if inventory_sha != inventory["sha256"] or int(inventory["signature_count"]) != sum(
        int(value) for value in observed_counts.values()
    ):
        raise ValueError("legacy observed-state inventory receipt mismatch")


def load_fresh_evaluation_artifacts(
    config_path: str | Path,
    *,
    repo_root: str | Path,
) -> LoadedFreshEvaluation:
    """Load separate deployment/label archives with fail-closed provenance."""

    root = Path(repo_root).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    config = _read_json_mapping(config_file, label="fresh complementary config")
    training_path, training_sha = validate_formal_config(config, repo_root=root)
    training_config = _read_json_mapping(training_path, label="predictor training config")
    config_sha = sha256_file(config_file)

    fresh = _mapping(config["fresh_data"], label="config.fresh_data")
    deployment_path = _resolve(
        fresh["deployment_cache"], root=root, label="deployment cache"
    )
    label_path = _resolve(fresh["label_cache"], root=root, label="label cache")
    metadata_path = _resolve(fresh["metadata"], root=root, label="collection metadata")
    deployment_sha = sha256_file(deployment_path) if deployment_path.is_file() else ""
    label_sha = sha256_file(label_path) if label_path.is_file() else ""
    if not deployment_sha:
        raise FileNotFoundError(deployment_path)
    if not label_sha:
        raise FileNotFoundError(label_path)
    metadata = _read_json_mapping(metadata_path, label="fresh collection metadata")
    if set(metadata) != COLLECTION_METADATA_REQUIRED_KEYS:
        raise ValueError(
            "fresh collection metadata schema drifted: "
            f"missing={sorted(COLLECTION_METADATA_REQUIRED_KEYS - set(metadata))}, "
            f"extra={sorted(set(metadata) - COLLECTION_METADATA_REQUIRED_KEYS)}"
        )
    if (
        metadata["schema_version"] != COLLECTION_METADATA_SCHEMA_VERSION
        or metadata["protocol"] != FRESH_DATA_PROTOCOL
        or metadata["completed"] is not True
        or metadata["config_sha256"] != config_sha
    ):
        raise ValueError("fresh collection metadata identity/completion drifted")
    _artifact_record(
        metadata,
        "deployment_cache",
        expected_path=deployment_path,
        expected_sha=deployment_sha,
        repo_root=root,
    )
    _artifact_record(
        metadata,
        "label_cache",
        expected_path=label_path,
        expected_sha=label_sha,
        repo_root=root,
    )
    p1 = _mapping(config["p1"], label="config.p1")
    freshness = _mapping(config["freshness"], label="config.freshness")
    _validate_bound_source_record(
        metadata,
        "policy_checkpoint",
        expected_path=_resolve(
            p1["policy_checkpoint"], root=root, label="config P1 policy"
        ),
        expected_sha=p1["policy_checkpoint_sha256"],
        repo_root=root,
    )
    _validate_bound_source_record(
        metadata,
        "corrector_checkpoint",
        expected_path=_resolve(
            p1["corrector_checkpoint"], root=root, label="config P1 corrector"
        ),
        expected_sha=p1["corrector_checkpoint_sha256"],
        repo_root=root,
    )
    _validate_bound_source_record(
        metadata,
        "legacy_observation_source",
        expected_path=_resolve(
            freshness["legacy_observation_source"],
            root=root,
            label="config legacy observation source",
        ),
        expected_sha=freshness["legacy_observation_source_sha256"],
        repo_root=root,
    )
    _validate_freshness_receipt(metadata, config=config, repo_root=root)
    if not isinstance(metadata["cache_contract_json"], str):
        raise TypeError("fresh collection metadata cache_contract_json must be a string")
    if metadata["cache_contract_json"] != cache_contract_json(config):
        raise ValueError("fresh collection cache contract receipt mismatch")
    collection_dataset_sha = hashlib.sha256(
        (
            metadata["cache_contract_json"]
            + "\0deployment\0"
            + deployment_sha
            + "\0offline-labels\0"
            + label_sha
        ).encode("utf-8")
    ).hexdigest()
    if metadata["dataset_sha256"] != collection_dataset_sha:
        raise ValueError("fresh collection dataset SHA-256 receipt mismatch")

    deployment = _load_npz_exact(
        deployment_path,
        expected_keys=DEPLOYMENT_CACHE_KEYS,
        label="fresh deployment cache",
    )
    labels = _load_npz_exact(
        label_path,
        expected_keys=OFFLINE_LABEL_CACHE_KEYS,
        label="fresh offline-label cache",
    )
    if "target_token" in deployment or "history_token" in labels:
        raise RuntimeError("deployment/offline-label isolation was violated")
    arrays = combine_fresh_cache_arrays(deployment, labels, contract=config)
    if set(arrays) != FRESH_CACHE_KEYS:
        raise RuntimeError("recombined fresh cache does not match the public schema")
    cache = fresh_arrays_to_native_cache(
        arrays,
        source_path=deployment_path,
        metadata=metadata,
        contract=config,
    )
    target = np.ascontiguousarray(labels["target_token"], dtype=np.float32)
    metadata_sha = sha256_file(metadata_path)
    dataset_sha = canonical_sha256(
        {
            "config_sha256": config_sha,
            "deployment_cache_sha256": deployment_sha,
            "label_cache_sha256": label_sha,
            "metadata_sha256": metadata_sha,
        }
    )
    provenance = FreshArtifactProvenance(
        config_path=config_file,
        deployment_path=deployment_path,
        label_path=label_path,
        metadata_path=metadata_path,
        training_config_path=training_path,
        config_sha256=config_sha,
        deployment_cache_sha256=deployment_sha,
        label_cache_sha256=label_sha,
        metadata_sha256=metadata_sha,
        training_config_sha256=training_sha,
        dataset_sha256=dataset_sha,
    )
    return LoadedFreshEvaluation(
        config=config,
        training_config=training_config,
        collection_metadata=metadata,
        cache=cache,
        target_token=target,
        provenance=provenance,
    )


def validate_and_load_p1_policy(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    device: str | torch.device,
) -> nn.Module:
    """Load the exact policy/corrector pair used to reconstruct baselines."""

    p1 = _mapping(config.get("p1"), label="config.p1")
    policy_path = _resolve(p1["policy_checkpoint"], root=repo_root, label="P1 policy")
    corrector_path = _resolve(
        p1["corrector_checkpoint"], root=repo_root, label="P1 corrector"
    )
    policy_sha = _validate_file_sha(
        policy_path, p1["policy_checkpoint_sha256"], label="P1 policy"
    )
    corrector_sha = _validate_file_sha(
        corrector_path, p1["corrector_checkpoint_sha256"], label="P1 corrector"
    )
    policy_payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    corrector_payload = torch.load(corrector_path, map_location="cpu", weights_only=False)
    if not isinstance(policy_payload, Mapping) or not isinstance(corrector_payload, Mapping):
        raise TypeError("P1 policy/corrector checkpoints must contain mappings")
    _validate_corrector_payload(corrector_payload)
    args = _mapping(policy_payload.get("args"), label="P1 policy args")
    loss = _mapping(policy_payload.get("loss_config"), label="P1 policy loss_config")
    embedded = _resolve(args.get("fcpg_checkpoint"), root=repo_root, label="embedded corrector")
    if (
        policy_payload.get("checkpoint_type") != "fcpg_action_policy"
        or int(policy_payload.get("schema_version", 0)) < 2
        or args.get("mode") != "fcpg"
        or args.get("obs_mode") != "robot"
        or float(args.get("latent_pred_weight", -1.0)) != 0.0
        or float(loss.get("latent_pred_weight", -1.0)) != 0.0
        or (int(policy_payload.get("obs_dim", -1)), int(policy_payload.get("action_dim", -1)))
        != (4, 4)
        or int(policy_payload.get("num_keypoints", -1)) != P1_NATIVE_ANCHORS
        or embedded != corrector_path
    ):
        raise ValueError("P1 action-only policy contract drifted")

    from fcpg.action_chunk_policy import load_action_chunk_policy
    from fcpg.transformer_grounding import load_fcpg_vit_adapter

    target = torch.device(device)
    corrector, _ = load_fcpg_vit_adapter(corrector_path, target)
    corrector.requires_grad_(False).eval()
    state_before = sha256_state_dict(corrector)
    policy, _ = load_action_chunk_policy(
        policy_path, target, fcpg_adapter_override=corrector
    )
    policy.requires_grad_(False).eval()
    _verify_loaded_corrector_binding(
        policy,
        corrector,
        checkpoint_path=corrector_path,
        expected_file_sha256=corrector_sha,
        state_sha256_before_binding=state_before,
    )
    if sha256_file(policy_path) != policy_sha:
        raise RuntimeError("P1 policy checkpoint changed while loading")
    return policy


def flatten_branch_specific_deployment(
    cache: P1NativeDynamicsCache,
    geometry_token: np.ndarray,
) -> FlattenedDeploymentInputs:
    """Flatten ``N*S`` and retain a singleton predictor branch dimension."""

    history = np.asarray(cache.history_token)
    actions = np.asarray(cache.history_action)
    future = np.asarray(cache.future_action)
    wrong = np.asarray(cache.wrong_future_action)
    geometry = np.asarray(geometry_token)
    if history.ndim != 5:
        raise ValueError("history_token must have shape [N,S,T,K,D]")
    n, branches, history_steps, anchors, latent = history.shape
    expected_history = (P1_NATIVE_HISTORY_SIZE, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM)
    if (history_steps, anchors, latent) != expected_history:
        raise ValueError("history_token dimensions drifted")
    if actions.shape != (n, branches, history_steps - 1, P1_NATIVE_ACTION_DIM):
        raise ValueError("history_action shape drifted")
    if future.shape != (
        n,
        branches,
        P1_NATIVE_FUTURE_STEPS,
        P1_NATIVE_ACTION_DIM,
    ):
        raise ValueError("future_action shape drifted")
    if wrong.shape != (
        n,
        7,
        branches,
        P1_NATIVE_FUTURE_STEPS,
        P1_NATIVE_ACTION_DIM,
    ):
        raise ValueError("wrong_future_action shape drifted")
    if geometry.shape != (
        n,
        branches,
        P1_NATIVE_FUTURE_STEPS,
        anchors,
        latent,
    ):
        raise ValueError("geometry_token shape drifted")
    total = n * branches
    wrong_flat = np.transpose(wrong, (0, 2, 1, 3, 4)).reshape(
        total, 7, P1_NATIVE_FUTURE_STEPS, P1_NATIVE_ACTION_DIM
    )
    return FlattenedDeploymentInputs(
        episode_count=n,
        branch_count=branches,
        history_token=np.ascontiguousarray(
            history.reshape(total, history_steps, anchors, latent), dtype=np.float32
        ),
        history_action=np.ascontiguousarray(
            actions.reshape(total, history_steps - 1, P1_NATIVE_ACTION_DIM),
            dtype=np.float32,
        ),
        future_action=np.ascontiguousarray(
            future.reshape(total, 1, P1_NATIVE_FUTURE_STEPS, P1_NATIVE_ACTION_DIM),
            dtype=np.float32,
        ),
        wrong_future_action=np.ascontiguousarray(
            wrong_flat[:, :, None], dtype=np.float32
        ),
        geometry_token=np.ascontiguousarray(
            geometry.reshape(total, 1, P1_NATIVE_FUTURE_STEPS, anchors, latent),
            dtype=np.float32,
        ),
    )


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device=device, dtype=torch.float32)


def evaluate_loaded_seed(
    loaded: LoadedP1NativePair,
    deployment: FlattenedDeploymentInputs,
    *,
    device: str | torch.device,
    batch_size: int,
) -> SeedPredictions:
    """Evaluate one frozen seed sequentially; no offline label is accepted."""

    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    target_device = torch.device(device)
    total = int(deployment.history_token.shape[0])
    factual: list[np.ndarray] = []
    no_action: list[np.ndarray] = []
    wrong: list[np.ndarray] = []
    raw: list[np.ndarray] = []
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        history = _tensor(deployment.history_token[start:stop], target_device)
        history_action = _tensor(deployment.history_action[start:stop], target_device)
        future = _tensor(deployment.future_action[start:stop], target_device)
        wrong_future = _tensor(
            deployment.wrong_future_action[start:stop], target_device
        )
        geometry = _tensor(deployment.geometry_token[start:stop], target_device)
        matched: ComplementaryMatchedRollout = recursive_complementary_matched_rollout(
            loaded.factual,
            loaded.no_action,
            history,
            history_action,
            future,
            wrong_future,
            geometry,
            gains=P1_NATIVE_COMPLEMENTARY_GAINS,
        )
        raw_prediction = recursive_complementary_rollout(
            loaded.factual,
            history,
            history_action,
            future,
            geometry,
            gains=(1.0,) * P1_NATIVE_FUTURE_STEPS,
        )
        factual.append(matched.factual[:, 0].float().cpu().numpy())
        no_action.append(matched.no_action[:, 0].float().cpu().numpy())
        wrong.append(matched.wrong_action[:, :, 0].float().cpu().numpy())
        raw.append(raw_prediction[:, 0].float().cpu().numpy())
    n = deployment.episode_count
    branches = deployment.branch_count
    common = (n, branches, P1_NATIVE_FUTURE_STEPS, P1_NATIVE_ANCHORS, P1_NATIVE_TOKEN_DIM)
    return SeedPredictions(
        factual=np.concatenate(factual, axis=0).reshape(common),
        no_action=np.concatenate(no_action, axis=0).reshape(common),
        wrong_action=np.concatenate(wrong, axis=0).reshape(
            n,
            branches,
            7,
            P1_NATIVE_FUTURE_STEPS,
            P1_NATIVE_ANCHORS,
            P1_NATIVE_TOKEN_DIM,
        ).transpose(0, 2, 1, 3, 4, 5),
        raw_factual=np.concatenate(raw, axis=0).reshape(common),
    )


def episode_horizon_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reduce branch, anchor, and token dimensions within each episode/H."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction/target must share shape [N,S,H,K,D]")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction/target contains non-finite values")
    return np.mean(np.square(prediction - target, dtype=np.float32), axis=(1, 3, 4), dtype=np.float64)


def _base_row(
    *,
    seed: int,
    checkpoint_sha256: str,
    method: str,
    episode: str,
    horizon: int,
    mse: float,
    provenance: FreshArtifactProvenance,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "classification": FORMAL_CLASSIFICATION,
        "collector_protocol": FORMAL_CONFIG_PROTOCOL,
        "fresh_data_protocol": FRESH_DATA_PROTOCOL,
        "protocol": ROW_PROTOCOL,
        "task": TASK,
        "model_seed": int(seed),
        "method": method,
        "physical_episode_id": episode,
        "horizon": int(horizon),
        "episode_horizon_mse": float(mse),
        "statistical_unit": "physical_episode",
        "dataset_sha256": provenance.dataset_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": provenance.config_sha256,
        "deployment_cache_sha256": provenance.deployment_cache_sha256,
        "label_cache_sha256": provenance.label_cache_sha256,
        "metadata_sha256": provenance.metadata_sha256,
    }


def build_seed_rows(
    *,
    seed: int,
    checkpoint_sha256: str,
    predictions: SeedPredictions,
    baselines: P1NativeBaselines,
    target_token: np.ndarray,
    physical_episode_id: Sequence[object],
    provenance: FreshArtifactProvenance,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Create formal panel rows and separate raw-LeWM diagnostics."""

    target = np.asarray(target_token, dtype=np.float32)
    episodes = tuple(str(value) for value in physical_episode_id)
    if target.shape[0] != len(episodes) or len(set(episodes)) != len(episodes):
        raise ValueError("physical episode identities must be unique and target-aligned")
    predictions_by_method = {
        CANDIDATE: predictions.factual,
        MATCHED_NO_ACTION: predictions.no_action,
        HOLD_LAST: baselines.hold_last_token,
        GEOMETRY_HOLD: baselines.geometry_hold_exact_fk_token,
        VISUAL_CV: baselines.visual_cv_exact_fk_token,
        **{
            name: predictions.wrong_action[:, index]
            for index, name in enumerate(WRONG_ACTIONS)
        },
    }
    formal: list[dict[str, object]] = []
    for method, prediction in predictions_by_method.items():
        metric = episode_horizon_mse(prediction, target)
        for episode_index, episode in enumerate(episodes):
            for horizon_index, horizon in enumerate(HORIZONS):
                formal.append(
                    _base_row(
                        seed=seed,
                        checkpoint_sha256=checkpoint_sha256,
                        method=method,
                        episode=episode,
                        horizon=horizon,
                        mse=float(metric[episode_index, horizon_index]),
                        provenance=provenance,
                    )
                )
    raw_metric = episode_horizon_mse(predictions.raw_factual, target)
    diagnostic = [
        _base_row(
            seed=seed,
            checkpoint_sha256=checkpoint_sha256,
            method=RAW_LEWM_METHOD,
            episode=episode,
            horizon=horizon,
            mse=float(raw_metric[episode_index, horizon_index]),
            provenance=provenance,
        )
        for episode_index, episode in enumerate(episodes)
        for horizon_index, horizon in enumerate(HORIZONS)
    ]
    return formal, diagnostic


def checkpoint_specs_from_config(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[FrozenP1NativeCheckpointSpec, ...]:
    values = config.get("frozen_predictors")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("formal config must bind exactly three predictors")
    specs = []
    for item_value in values:
        item = _mapping(item_value, label="frozen predictor")
        path = _resolve(item["checkpoint"], root=repo_root, label="predictor checkpoint")
        expected = _sha256(item["sha256"], label="predictor checkpoint SHA-256")
        _validate_file_sha(path, expected, label=f"predictor seed {item['seed']}")
        specs.append(
            FrozenP1NativeCheckpointSpec(
                seed=int(item["seed"]), path=path, sha256=expected
            )
        )
    return tuple(specs)


def load_one_frozen_seed(
    spec: FrozenP1NativeCheckpointSpec,
    *,
    training_config: Mapping[str, Any],
    device: str | torch.device,
) -> LoadedP1NativePair:
    return load_frozen_p1_native_checkpoint(
        spec, expected_config=training_config, device=device
    )


def reconstruct_fresh_baselines(
    cache: P1NativeDynamicsCache,
    policy: nn.Module,
    *,
    device: str | torch.device,
    batch_size: int,
) -> P1NativeBaselines:
    return reconstruct_p1_native_baselines(
        cache, policy, device=device, batch_size=batch_size
    )


__all__ = [
    "COLLECTION_METADATA_REQUIRED_KEYS",
    "FORMAL_CLASSIFICATION",
    "FORMAL_CONFIG_PROTOCOL",
    "FlattenedDeploymentInputs",
    "FreshArtifactProvenance",
    "LoadedFreshEvaluation",
    "RAW_LEWM_METHOD",
    "SeedPredictions",
    "build_seed_rows",
    "canonical_sha256",
    "checkpoint_specs_from_config",
    "episode_horizon_mse",
    "evaluate_loaded_seed",
    "flatten_branch_specific_deployment",
    "load_fresh_evaluation_artifacts",
    "load_one_frozen_seed",
    "reconstruct_fresh_baselines",
    "validate_and_load_p1_policy",
    "validate_formal_config",
]
