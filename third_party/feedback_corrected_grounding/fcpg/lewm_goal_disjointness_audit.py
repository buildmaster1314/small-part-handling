"""Post-hoc goal-signature audit for Counterfactual v2 development data.

This module deliberately proves less than complete upstream independence.  It
can bind the historical MetaWorld datasets used by Gate L1 and compare their
task-scoped goal signatures with a Counterfactual v2 store.  It cannot recover
missing training-episode receipts for the FCPG corrector or action policy, and
therefore must never upgrade the formal G0 eligibility of that store.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.lewm_counterfactual_v2_storage import (
    CounterfactualV2Dataset,
    load_counterfactual_v2_store,
)


AUDIT_PROTOCOL = "fcpg_lewm_upstream_goal_disjointness_forensic_v1"
PROOF_SCOPE = "data_and_gate_l1_lewm_only"
_SHA_CHUNK_SIZE = 1024 * 1024


class GoalDisjointnessAuditError(ValueError):
    """Raised when an input cannot satisfy the fail-closed audit contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_SHA_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoalDisjointnessAuditError(f"{label} must be an object")
    return value


def _require_sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise GoalDisjointnessAuditError(f"{label} must be an array")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GoalDisjointnessAuditError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    result = _require_string(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise GoalDisjointnessAuditError(f"{label} must be a lowercase SHA-256")
    return result


def _require_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GoalDisjointnessAuditError(f"{label} must be a non-negative integer")
    return value


def _resolve_path(repo_root: Path, value: object, label: str) -> Path:
    raw = Path(_require_string(value, label)).expanduser()
    return (repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()


def _verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GoalDisjointnessAuditError(f"{label} must be a regular non-symlink file: {path}")
    actual_size = int(path.stat().st_size)
    if actual_size != expected_size:
        raise GoalDisjointnessAuditError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise GoalDisjointnessAuditError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "path": str(path),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
    }


def _canonical_goal(goal: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(goal)
    if value.shape != (3,):
        raise GoalDisjointnessAuditError(f"{label} must have shape [3]")
    if value.dtype != np.dtype(np.float32):
        raise GoalDisjointnessAuditError(f"{label} must use float32, got {value.dtype}")
    result = np.asarray(value, dtype="<f4")
    if not np.isfinite(result).all():
        raise GoalDisjointnessAuditError(f"{label} contains a non-finite coordinate")
    return result


def _signature(goal: np.ndarray) -> dict[str, Any]:
    canonical = _canonical_goal(goal, "goal signature")
    payload = canonical.tobytes(order="C")
    return {
        "float32_le_hex": payload.hex(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "values": [float(item) for item in canonical],
    }


def episode_goal_signatures(
    goals: np.ndarray,
    episode_keys: Sequence[object],
    *,
    label: str,
) -> tuple[dict[bytes, dict[str, Any]], dict[str, Any]]:
    """Validate exact within-episode goal constancy and group signatures."""

    values = np.asarray(goals)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise GoalDisjointnessAuditError(f"{label} goals must have shape [N, 3]")
    if values.dtype != np.dtype(np.float32):
        raise GoalDisjointnessAuditError(f"{label} goals must use float32")
    if values.shape[0] != len(episode_keys):
        raise GoalDisjointnessAuditError(f"{label} episode key count mismatch")
    if values.shape[0] == 0:
        raise GoalDisjointnessAuditError(f"{label} must contain at least one row")

    rows_by_episode: dict[str, list[int]] = defaultdict(list)
    for row_index, episode_key in enumerate(episode_keys):
        key = str(episode_key)
        if not key:
            raise GoalDisjointnessAuditError(f"{label} has an empty episode key")
        rows_by_episode[key].append(row_index)

    grouped: dict[bytes, dict[str, Any]] = {}
    for episode_key in sorted(rows_by_episode):
        row_indices = rows_by_episode[episode_key]
        reference = _canonical_goal(values[row_indices[0]], f"{label}:{episode_key}")
        reference_bytes = reference.tobytes(order="C")
        for row_index in row_indices[1:]:
            candidate = _canonical_goal(values[row_index], f"{label}:{episode_key}")
            if candidate.tobytes(order="C") != reference_bytes:
                raise GoalDisjointnessAuditError(
                    f"{label} goal is not exactly constant within episode {episode_key!r}"
                )
        entry = grouped.setdefault(
            reference_bytes,
            {
                **_signature(reference),
                "episode_keys": [],
                "row_count": 0,
            },
        )
        entry["episode_keys"].append(episode_key)
        entry["row_count"] += len(row_indices)

    return grouped, {
        "row_count": int(values.shape[0]),
        "episode_count": len(rows_by_episode),
        "unique_signature_count": len(grouped),
        "within_episode_exact_constancy": True,
    }


def compare_task_goal_signatures(
    old_by_task: Mapping[str, Mapping[bytes, Mapping[str, Any]]],
    new_by_task: Mapping[str, Mapping[bytes, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compare exact float32 signatures within task only."""

    old_tasks = set(old_by_task)
    new_tasks = set(new_by_task)
    if old_tasks != new_tasks or not old_tasks:
        raise GoalDisjointnessAuditError(
            f"task sets differ: old={sorted(old_tasks)}, new={sorted(new_tasks)}"
        )

    task_reports: dict[str, Any] = {}
    total_intersections = 0
    global_minimum = math.inf
    for task_name in sorted(old_tasks):
        old_signatures = old_by_task[task_name]
        new_signatures = new_by_task[task_name]
        if not old_signatures or not new_signatures:
            raise GoalDisjointnessAuditError(f"task {task_name!r} has an empty signature set")
        intersections = sorted(set(old_signatures) & set(new_signatures))
        total_intersections += len(intersections)

        old_keys = sorted(old_signatures)
        new_keys = sorted(new_signatures)
        old_values = np.asarray(
            [[*np.frombuffer(key, dtype="<f4", count=3)] for key in old_keys],
            dtype=np.float64,
        )
        new_values = np.asarray(
            [[*np.frombuffer(key, dtype="<f4", count=3)] for key in new_keys],
            dtype=np.float64,
        )
        distances = np.linalg.norm(new_values[:, None, :] - old_values[None, :, :], axis=-1)
        flat_index = int(np.argmin(distances))
        new_index, old_index = np.unravel_index(flat_index, distances.shape)
        minimum = float(distances[new_index, old_index])
        global_minimum = min(global_minimum, minimum)

        task_reports[task_name] = {
            "old_signatures": [old_signatures[key] for key in old_keys],
            "new_signatures": [new_signatures[key] for key in new_keys],
            "exact_intersection_count": len(intersections),
            "exact_intersections": [old_signatures[key] for key in intersections],
            "minimum_l2_distance": minimum,
            "closest_pair": {
                "old": old_signatures[old_keys[old_index]],
                "new": new_signatures[new_keys[new_index]],
            },
        }

    return {
        "comparison_scope": "exact_float32_goal_signature_within_same_task_only",
        "task_reports": task_reports,
        "total_exact_intersection_count": total_intersections,
        "global_minimum_l2_distance": float(global_minimum),
        "task_scoped_exact_disjoint": total_intersections == 0,
    }


def _load_historical_task(
    path: Path,
    *,
    task_name: str,
    expected_row_count: int,
    expected_episode_count: int,
) -> tuple[dict[bytes, dict[str, Any]], dict[str, Any], set[int]]:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise GoalDisjointnessAuditError(f"cannot open historical NPZ {path}") from exc
    try:
        required = {"obs", "episode_id", "task_name"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise GoalDisjointnessAuditError(f"historical NPZ {path} is missing {missing}")
        observations = np.asarray(archive["obs"])
        episode_ids = np.asarray(archive["episode_id"])
        task_names = np.asarray(archive["task_name"])
    finally:
        archive.close()

    if observations.dtype != np.dtype(np.float32) or observations.ndim != 2 or observations.shape[1] < 3:
        raise GoalDisjointnessAuditError(f"historical {task_name} obs must be float32 [N, >=3]")
    if episode_ids.ndim != 1 or episode_ids.shape[0] != observations.shape[0]:
        raise GoalDisjointnessAuditError(f"historical {task_name} episode_id shape mismatch")
    if task_names.ndim != 1 or task_names.shape[0] != observations.shape[0]:
        raise GoalDisjointnessAuditError(f"historical {task_name} task_name shape mismatch")
    if int(observations.shape[0]) != expected_row_count:
        raise GoalDisjointnessAuditError(
            f"historical {task_name} row count mismatch: expected {expected_row_count}, got {observations.shape[0]}"
        )
    observed_task_names = {str(item) for item in np.unique(task_names)}
    if observed_task_names != {task_name}:
        raise GoalDisjointnessAuditError(
            f"historical task scope mismatch: expected {task_name!r}, got {sorted(observed_task_names)}"
        )
    integer_episode_ids = {int(item) for item in np.unique(episode_ids)}
    if len(integer_episode_ids) != expected_episode_count:
        raise GoalDisjointnessAuditError(
            f"historical {task_name} episode count mismatch: expected {expected_episode_count}, got {len(integer_episode_ids)}"
        )
    goals = np.asarray(observations[:, -3:], dtype=np.float32)
    signatures, summary = episode_goal_signatures(
        goals,
        [int(item) for item in episode_ids],
        label=f"historical:{task_name}",
    )
    return signatures, summary, integer_episode_ids


def _manifest_episode_ids(
    source: Mapping[str, Any],
    *,
    task_name: str,
) -> tuple[set[int], dict[str, int]]:
    groups = {
        "train": "train_episodes",
        "validation": "val_episodes",
        "excluded": "excluded_episodes",
    }
    all_ids: set[int] = set()
    counts: dict[str, int] = {}
    for group, key in groups.items():
        values = _require_sequence(source.get(key), f"split source {task_name}.{key}")
        group_ids: set[int] = set()
        for value in values:
            text = _require_string(value, f"split source {task_name}.{key} item")
            prefix = f"{task_name}:"
            if not text.startswith(prefix):
                raise GoalDisjointnessAuditError(
                    f"split episode {text!r} is outside task scope {task_name!r}"
                )
            try:
                episode_id = int(text[len(prefix) :])
            except ValueError as exc:
                raise GoalDisjointnessAuditError(f"invalid split episode {text!r}") from exc
            if episode_id in group_ids or episode_id in all_ids:
                raise GoalDisjointnessAuditError(f"duplicate split episode {text!r}")
            group_ids.add(episode_id)
        all_ids.update(group_ids)
        counts[group] = len(group_ids)
    return all_ids, counts


def _load_new_task_signatures(
    dataset: CounterfactualV2Dataset,
    expected_tasks: Sequence[str],
    *,
    expected_rows_per_task: int,
    expected_episodes_per_task: int,
) -> tuple[dict[str, dict[bytes, dict[str, Any]]], dict[str, Any]]:
    goals = np.asarray(dataset.offline_labels.get("goal_world"))
    if goals.dtype != np.dtype(np.float32) or goals.ndim != 2 or goals.shape[1:] != (3,):
        raise GoalDisjointnessAuditError("new v2 goal_world must be float32 [N, 3]")
    if len(dataset.index_rows) != goals.shape[0]:
        raise GoalDisjointnessAuditError("new v2 goal_world/index row count mismatch")

    rows_by_task: dict[str, list[int]] = defaultdict(list)
    episode_keys_by_task: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(dataset.index_rows):
        task_name = _require_string(row.get("task_name"), f"index_rows[{index}].task_name")
        physical = _require_sha256(
            row.get("physical_episode_sha256"),
            f"index_rows[{index}].physical_episode_sha256",
        )
        rows_by_task[task_name].append(index)
        episode_keys_by_task[task_name].append(physical)
    if set(rows_by_task) != set(expected_tasks):
        raise GoalDisjointnessAuditError(
            f"new v2 tasks differ: expected {sorted(expected_tasks)}, got {sorted(rows_by_task)}"
        )

    signatures_by_task: dict[str, dict[bytes, dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for task_name in sorted(expected_tasks):
        row_indices = rows_by_task[task_name]
        if len(row_indices) != expected_rows_per_task:
            raise GoalDisjointnessAuditError(
                f"new v2 {task_name} row count mismatch: expected {expected_rows_per_task}, got {len(row_indices)}"
            )
        signatures, summary = episode_goal_signatures(
            goals[row_indices],
            episode_keys_by_task[task_name],
            label=f"new_v2:{task_name}",
        )
        if summary["episode_count"] != expected_episodes_per_task:
            raise GoalDisjointnessAuditError(
                f"new v2 {task_name} episode count mismatch: expected {expected_episodes_per_task}, got {summary['episode_count']}"
            )
        signatures_by_task[task_name] = signatures
        summaries[task_name] = summary
    return signatures_by_task, summaries


def _discover_metaworld_source() -> tuple[str, Path]:
    try:
        version = importlib.metadata.version("metaworld")
        module = importlib.import_module("metaworld.sawyer_xyz_env")
        source = inspect.getsourcefile(module)
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise GoalDisjointnessAuditError("cannot discover installed MetaWorld") from exc
    if source is None:
        raise GoalDisjointnessAuditError("MetaWorld sawyer source path is unavailable")
    return version, Path(source).resolve()


def run_goal_disjointness_audit(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    metaworld_version: str | None = None,
    sawyer_source_path: Path | None = None,
) -> dict[str, Any]:
    """Run the bounded audit and return a content-addressed JSON report."""

    if config.get("protocol") != AUDIT_PROTOCOL:
        raise GoalDisjointnessAuditError("unexpected audit protocol")
    if config.get("proof_scope") != PROOF_SCOPE:
        raise GoalDisjointnessAuditError("proof_scope must remain data_and_gate_l1_lewm_only")
    if config.get("upgrade_g0_eligibility") is not False:
        raise GoalDisjointnessAuditError("audit must explicitly forbid G0 eligibility upgrades")
    repo_root = repo_root.expanduser().resolve()

    environment = _require_mapping(config.get("metaworld"), "metaworld")
    if metaworld_version is None or sawyer_source_path is None:
        discovered_version, discovered_source = _discover_metaworld_source()
        metaworld_version = metaworld_version or discovered_version
        sawyer_source_path = sawyer_source_path or discovered_source
    expected_version = _require_string(environment.get("version"), "metaworld.version")
    if metaworld_version != expected_version:
        raise GoalDisjointnessAuditError(
            f"MetaWorld version mismatch: expected {expected_version}, got {metaworld_version}"
        )
    source_record = _verify_file(
        sawyer_source_path.resolve(),
        expected_size=_require_integer(environment.get("sawyer_source_size_bytes"), "metaworld.sawyer_source_size_bytes"),
        expected_sha256=_require_sha256(environment.get("sawyer_source_sha256"), "metaworld.sawyer_source_sha256"),
        label="MetaWorld sawyer source",
    )
    source_record["package_version"] = metaworld_version
    source_record["module"] = "metaworld.sawyer_xyz_env"

    split_config = _require_mapping(config.get("gate_l1_split_manifest"), "gate_l1_split_manifest")
    split_path = _resolve_path(repo_root, split_config.get("path"), "gate_l1_split_manifest.path")
    split_record = _verify_file(
        split_path,
        expected_size=_require_integer(split_config.get("size_bytes"), "gate_l1_split_manifest.size_bytes"),
        expected_sha256=_require_sha256(split_config.get("sha256"), "gate_l1_split_manifest.sha256"),
        label="Gate L1 split manifest",
    )
    try:
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoalDisjointnessAuditError("Gate L1 split manifest is invalid JSON") from exc
    split_sources = _require_mapping(
        _require_mapping(split_manifest, "Gate L1 split manifest").get("sources"),
        "Gate L1 split manifest.sources",
    )

    historical_config = _require_mapping(config.get("historical_sources"), "historical_sources")
    historical_signatures: dict[str, dict[bytes, dict[str, Any]]] = {}
    historical_report: dict[str, Any] = {}
    expected_tasks: list[str] = []
    for source_name in sorted(historical_config):
        source = _require_mapping(historical_config[source_name], f"historical_sources.{source_name}")
        task_name = _require_string(source.get("task_name"), f"historical_sources.{source_name}.task_name")
        if task_name in historical_signatures:
            raise GoalDisjointnessAuditError(f"duplicate historical task {task_name!r}")
        expected_tasks.append(task_name)
        path = _resolve_path(repo_root, source.get("path"), f"historical_sources.{source_name}.path")
        file_record = _verify_file(
            path,
            expected_size=_require_integer(source.get("size_bytes"), f"historical_sources.{source_name}.size_bytes"),
            expected_sha256=_require_sha256(source.get("sha256"), f"historical_sources.{source_name}.sha256"),
            label=f"historical source {source_name}",
        )
        signatures, summary, episode_ids = _load_historical_task(
            path,
            task_name=task_name,
            expected_row_count=_require_integer(source.get("row_count"), f"historical_sources.{source_name}.row_count"),
            expected_episode_count=_require_integer(source.get("episode_count"), f"historical_sources.{source_name}.episode_count"),
        )
        split_source = _require_mapping(split_sources.get(source_name), f"split sources.{source_name}")
        split_sha = _require_sha256(split_source.get("data_sha256"), f"split sources.{source_name}.data_sha256")
        if split_sha != file_record["sha256"]:
            raise GoalDisjointnessAuditError(
                f"split manifest/source SHA mismatch for {source_name}: {split_sha} != {file_record['sha256']}"
            )
        manifest_episode_ids, split_counts = _manifest_episode_ids(split_source, task_name=task_name)
        if manifest_episode_ids != episode_ids:
            raise GoalDisjointnessAuditError(
                f"split manifest does not partition all historical episodes for {source_name}"
            )
        historical_signatures[task_name] = signatures
        historical_report[task_name] = {
            "source_name": source_name,
            "file": file_record,
            "goal_extraction": "obs[:, -3:]",
            "goal_dtype": "float32",
            "summary": summary,
            "split_counts": split_counts,
            "split_data_sha256_matches": True,
        }

    new_config = _require_mapping(config.get("counterfactual_v2_store"), "counterfactual_v2_store")
    store_path = _resolve_path(repo_root, new_config.get("path"), "counterfactual_v2_store.path")
    dataset = load_counterfactual_v2_store(store_path, mmap_mode="r")
    expected_result_sha256 = _require_sha256(
        new_config.get("result_sha256"), "counterfactual_v2_store.result_sha256"
    )
    actual_result_sha256 = str(dataset.manifest.get("result_sha256", ""))
    if actual_result_sha256 != expected_result_sha256:
        raise GoalDisjointnessAuditError(
            f"Counterfactual v2 result SHA mismatch: expected {expected_result_sha256}, got {actual_result_sha256}"
        )
    manifest_path = store_path / "manifest.json"
    manifest_record = _verify_file(
        manifest_path,
        expected_size=_require_integer(new_config.get("manifest_size_bytes"), "counterfactual_v2_store.manifest_size_bytes"),
        expected_sha256=_require_sha256(new_config.get("manifest_sha256"), "counterfactual_v2_store.manifest_sha256"),
        label="Counterfactual v2 manifest",
    )
    new_signatures, new_summaries = _load_new_task_signatures(
        dataset,
        expected_tasks,
        expected_rows_per_task=_require_integer(new_config.get("rows_per_task"), "counterfactual_v2_store.rows_per_task"),
        expected_episodes_per_task=_require_integer(new_config.get("episodes_per_task"), "counterfactual_v2_store.episodes_per_task"),
    )

    comparison = compare_task_goal_signatures(historical_signatures, new_signatures)
    full_bundle_formal_verified = False
    report: dict[str, Any] = {
        "protocol": AUDIT_PROTOCOL,
        "schema_version": 1,
        "proof_scope": PROOF_SCOPE,
        "full_bundle_formal_verified": full_bundle_formal_verified,
        "counterfactual_v2_g0_eligibility_upgraded": False,
        "audit_mode": "read_only_post_hoc",
        "goal_semantics": {
            "historical_extraction": "obs[:, -3:]",
            "new_extraction": "offline_labels.goal_world",
            "canonical_dtype": "little_endian_float32",
            "current_metaworld_source_support": "SawyerXYZEnv._get_obs appends pos_goal as the final three coordinates",
            "historical_runtime_receipt_available": False,
        },
        "bindings": {
            "metaworld": source_record,
            "gate_l1_split_manifest": split_record,
            "historical_sources": historical_report,
            "counterfactual_v2_store": {
                "path": str(store_path),
                "result_sha256": actual_result_sha256,
                "manifest": manifest_record,
                "task_summaries": new_summaries,
            },
        },
        "goal_signature_comparison": comparison,
        "bounded_conclusion": {
            "data_and_gate_l1_lewm_goal_signatures_disjoint": bool(
                comparison["task_scoped_exact_disjoint"]
            ),
            "full_bundle_formal_verified": full_bundle_formal_verified,
            "counterfactual_v2_g0_eligibility_upgraded": False,
            "formal_claim_allowed": False,
            "reason": (
                "The bound historical Gate L1 datasets and the new v2 store have no exact "
                "within-task float32 goal-signature intersection; missing corrector/action "
                "training episode receipts prevent a full upstream-independence claim."
                if comparison["task_scoped_exact_disjoint"]
                else "At least one exact within-task goal signature overlaps; even the bounded data-only disjointness check failed."
            ),
        },
        "missing_upstream_receipts": [
            {
                "component": "fcpg_corrector",
                "missing": "training_physical_episode_seed_or_goal_signature_ledger",
            },
            {
                "component": "action_only_policy",
                "missing": "training_physical_episode_seed_or_goal_signature_ledger",
            },
        ],
        "non_claims": [
            "This report does not verify complete upstream-training non-overlap.",
            "This report does not change the source store's upstream_non_overlap status.",
            "This report does not make Counterfactual v2 G0-eligible or formally claimable.",
            "A positive minimum L2 distance is descriptive and is not a probabilistic independence guarantee.",
        ],
    }
    report["result_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish canonical JSON atomically without replacing an existing report."""

    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.staging-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "AUDIT_PROTOCOL",
    "PROOF_SCOPE",
    "GoalDisjointnessAuditError",
    "canonical_json_bytes",
    "compare_task_goal_signatures",
    "episode_goal_signatures",
    "run_goal_disjointness_audit",
    "sha256_file",
    "write_atomic_json",
]
