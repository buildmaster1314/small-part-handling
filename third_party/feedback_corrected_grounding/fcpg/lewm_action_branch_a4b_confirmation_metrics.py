"""Fresh-identity validation and frozen A4 confirmation statistics for A4b."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.lewm_a4_storage import validate_a4_state_cache
from fcpg.lewm_action_branch_a4_confirmation_metrics import (
    _summarize_confirmation_panel,
    _validate_frozen_split_assignments,
    validate_a4_development_go_authorization,
)
from fcpg.lewm_action_branch_a4b import (
    BENCHMARK_SEED,
    CONFIRMATION_TRAINING_SEEDS,
    POOL_FINAL,
    POOL_PREFLIGHT,
    PROTOCOL,
    SOURCE_A4_PROTOCOL,
    SPLIT_MANIFEST_ID,
    TARGET_ELIGIBLE_CANDIDATES_PER_TASK,
    TASK_IDENTITY_NAMESPACE,
    TASKS,
    validate_a4b_contract,
)
from fcpg.lewm_action_branch_a4b_confirmation_collection import (
    a4b_artifact_sha256,
    physical_identities_from_result,
    validate_a4b_collection_result,
)


CONFIRMATION_STAGE = "a4b_fresh_10k_confirmation"


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def validate_a4b_fresh_results(
    preflight_result: Mapping[str, Any],
    final_result: Mapping[str, Any],
) -> None:
    """Require complete pools and zero legacy/preflight/final digest overlap."""

    validate_a4b_collection_result(preflight_result)
    validate_a4b_collection_result(final_result)
    if preflight_result.get("pool") != POOL_PREFLIGHT:
        raise PermissionError("A4b confirmation requires a preflight-pool result")
    if (
        final_result.get("pool") != POOL_FINAL
        or final_result.get("passed") is not True
    ):
        raise PermissionError("A4b confirmation requires a successful final-pool result")
    old_and_preflight = physical_identities_from_result(
        preflight_result, include_legacy=True
    )
    final = physical_identities_from_result(final_result, include_legacy=False)
    if old_and_preflight & final:
        raise PermissionError(
            "A4b final physical identities overlap legacy or preflight identities"
        )


def validate_a4b_result_cache_path(
    final_result: Mapping[str, Any], cache_path: str | Path
) -> Path:
    """Bind a supplied cache path to the successful non-overwrite result."""

    validate_a4b_collection_result(final_result)
    if final_result.get("pool") != POOL_FINAL or final_result.get("passed") is not True:
        raise PermissionError("A4b cache requires a successful final result")
    supplied = Path(cache_path).expanduser().resolve()
    output = _mapping(final_result.get("output"), label="A4b final output")
    expected = Path(str(output["state_cache"])).expanduser().resolve()
    if supplied != expected:
        raise PermissionError("A4b cache path differs from the final result")
    if a4b_artifact_sha256(supplied) != output["state_cache_sha256"]:
        raise PermissionError("A4b cache content differs from the final result")
    ledger = Path(str(output["proposal_ledger"])).expanduser().resolve()
    if a4b_artifact_sha256(ledger) != output["proposal_ledger_sha256"]:
        raise PermissionError("A4b ledger content differs from the final result")
    return supplied


def validate_a4b_final_cache(
    cache: Mapping[str, np.ndarray],
    final_result: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Validate that every test row belongs to one selected fresh episode UID."""

    validate_a4b_collection_result(final_result)
    if final_result.get("pool") != POOL_FINAL or final_result.get("passed") is not True:
        raise PermissionError("A4b final cache requires a successful final result")
    validate_a4_state_cache(cache)
    split = np.asarray(cache["split"]).astype(str, copy=False)
    if set(split.tolist()) != {"test"}:
        raise ValueError("A4b final cache must be test-only")
    task_names = np.asarray(cache["task_name"]).astype(str, copy=False)
    task_indices = np.asarray(cache["task_class_index"], dtype=np.int64)
    raw_indices = np.asarray(cache["task_instance_index"], dtype=np.int64)
    episode_uids = np.asarray(cache["source_episode_id"]).astype(str, copy=False)
    proposal_ids = np.asarray(cache["proposal_id"]).astype(str, copy=False)
    if len(set(proposal_ids.tolist())) != proposal_ids.size:
        raise ValueError("A4b final cache proposal_id values must be unique")

    attempts = _mapping(
        final_result.get("attempts_by_task"), label="A4b final attempts"
    )
    selected = _mapping(
        final_result.get("selected_episode_uids_by_task"),
        label="A4b final selections",
    )
    selected_identity: dict[str, tuple[str, int, int]] = {}
    for task_index, task in enumerate(TASKS):
        selected_uids = selected[task]
        if (
            not isinstance(selected_uids, list)
            or len(selected_uids) != TARGET_ELIGIBLE_CANDIDATES_PER_TASK
        ):
            raise ValueError(f"A4b final selection for {task} must contain 12 UIDs")
        selected_set = set(selected_uids)
        for row in attempts[task]:
            uid = str(row["episode_uid"])
            if uid in selected_set:
                selected_identity[uid] = (
                    task,
                    task_index,
                    int(row["raw_task_index"]),
                )
    if len(selected_identity) != len(TASKS) * TARGET_ELIGIBLE_CANDIDATES_PER_TASK:
        raise ValueError("A4b final result selected UID identities are incomplete")

    for task, task_index, raw_index, uid, proposal_id in zip(
        task_names,
        task_indices,
        raw_indices,
        episode_uids,
        proposal_ids,
        strict=True,
    ):
        expected = selected_identity.get(uid)
        if expected is None:
            raise PermissionError("A4b final cache contains an unselected episode UID")
        if (str(task), int(task_index), int(raw_index)) != expected:
            raise PermissionError("A4b final cache row identity differs from its UID")
        if not str(proposal_id).startswith(f"{uid}:proposal-"):
            raise ValueError("A4b proposal_id is not namespaced by episode_uid")

    coverage: dict[str, dict[str, int]] = {}
    for task_index, task in enumerate(TASKS):
        rows = task_indices == task_index
        observed_uids = set(episode_uids[rows].tolist())
        expected_uids = set(selected[task])
        if observed_uids != expected_uids:
            raise ValueError(f"A4b final cache coverage differs from result for {task}")
        if set(task_names[rows].tolist()) != {task}:
            raise ValueError(f"A4b task class/name alignment drifted for {task}")
        coverage[task] = {
            "test_physical_episode_count": len(observed_uids),
            "test_bundle_count": int(np.count_nonzero(rows)),
        }
    return coverage


def validate_a4b_confirmation_caches(
    train_cache: Mapping[str, np.ndarray],
    test_cache: Mapping[str, np.ndarray],
    *,
    authorized_train_episode_counts: Mapping[str, int],
    source_contract: Mapping[str, Any],
    final_result: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Validate frozen old train rows and independently namespaced A4b test rows."""

    train_split = _validate_frozen_split_assignments(
        train_cache,
        contract=source_contract,
        allowed_splits=frozenset({"train", "development"}),
        label="A4b confirmation source train cache",
    )
    if "train" not in set(train_split.tolist()):
        raise ValueError("A4b source train cache has no train rows")
    fresh_coverage = validate_a4b_final_cache(test_cache, final_result)
    if set(np.asarray(train_cache["proposal_id"]).astype(str).tolist()) & set(
        np.asarray(test_cache["proposal_id"]).astype(str).tolist()
    ):
        raise ValueError("A4b source-train/final proposal_id values overlap")

    train_tasks = np.asarray(train_cache["task_class_index"], dtype=np.int64)
    train_episodes = np.asarray(train_cache["source_episode_id"]).astype(
        str, copy=False
    )
    coverage: dict[str, dict[str, int]] = {}
    for task_index, task in enumerate(TASKS):
        rows = train_tasks == task_index
        train_ids = set(train_episodes[rows & (train_split == "train")].tolist())
        development_ids = set(
            train_episodes[rows & (train_split == "development")].tolist()
        )
        expected_train = int(authorized_train_episode_counts[task])
        if len(train_ids) != expected_train:
            raise ValueError(
                f"A4b train coverage for {task} must match Development-Go "
                f"({expected_train}), got {len(train_ids)}"
            )
        coverage[task] = {
            "train_physical_episode_count": len(train_ids),
            "development_physical_episode_count": len(development_ids),
            **fresh_coverage[task],
        }
    return coverage


def summarize_a4b_confirmation(
    train_cache: Mapping[str, np.ndarray],
    test_cache: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    development_go_summary: Mapping[str, Any],
    preflight_result: Mapping[str, Any],
    final_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the unchanged v2 10k statistics to the fresh A4b final panel."""

    validate_a4b_contract(
        contract,
        source_contract=source_contract,
        split_manifest=split_manifest,
    )
    validate_a4b_fresh_results(preflight_result, final_result)
    train_counts = validate_a4_development_go_authorization(
        development_go_summary, contract=source_contract
    )
    coverage = validate_a4b_confirmation_caches(
        train_cache,
        test_cache,
        authorized_train_episode_counts=train_counts,
        source_contract=source_contract,
        final_result=final_result,
    )
    summary = _summarize_confirmation_panel(
        train_cache,
        test_cache,
        predictions,
        gate=contract["confirmation_gate"],
        training_seeds=CONFIRMATION_TRAINING_SEEDS,
        coverage=coverage,
    )
    result = dict(summary)
    result.update(
        {
            "protocol": PROTOCOL,
            "source_endpoint_protocol": SOURCE_A4_PROTOCOL,
            "stage": CONFIRMATION_STAGE,
            "fresh_identity": {
                "split_manifest_id": SPLIT_MANIFEST_ID,
                "task_identity_namespace": TASK_IDENTITY_NAMESPACE,
                "benchmark_seed": BENCHMARK_SEED,
                "legacy_preflight_final_digest_intersection": 0,
                "selected_physical_episodes_per_task": (
                    TARGET_ELIGIBLE_CANDIDATES_PER_TASK
                ),
            },
        }
    )
    boundary = dict(result["claim_boundary"])
    if result["passed"]:
        boundary["authorized_claim"] = (
            "The three frozen A4 v2 fixed-latest step-10000 models improve "
            "A4b fresh-panel H6-H8 prediction over every preregistered baseline "
            "under the unchanged confirmation Gate."
        )
    else:
        boundary["negative_result_statement"] = (
            "The frozen A4b fresh-panel confirmation did not satisfy all "
            "preregistered Gates."
        )
    result["claim_boundary"] = boundary
    return result


__all__ = [
    "CONFIRMATION_STAGE",
    "summarize_a4b_confirmation",
    "validate_a4b_confirmation_caches",
    "validate_a4b_final_cache",
    "validate_a4b_fresh_results",
    "validate_a4b_result_cache_path",
]
