#!/usr/bin/env python3
"""Run the independent PixelObserver-G0 fixed-gain filtering matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
FORMAL_BASE_CONFIG_PATH = (
    ROOT / "config/lewm/train/pixel_observer_base_online_sig002_seed3072.yaml"
).resolve()
FORMAL_ADAPTER_CONFIG_PATH = (
    ROOT / "config/lewm/train/pixel_observer_adapter_formal_seed6144.yaml"
).resolve()
FORMAL_DEVELOPMENT_CONFIG_PATH = (
    ROOT / "config/lewm/eval/pixel_observer_fixed_alpha_dev_seed6144.json"
).resolve()
FORMAL_CONFIRMATORY_DESIGN_PATH = (
    ROOT / "config/lewm/eval/pixel_observer_confirmatory_design_seed6144.json"
).resolve()
PROTOCOL_BUNDLE_RELATIVE_PATHS = (
    "scripts/evaluate_lewm_pixel_observer.py",
    "fcpg/lewm_pixel_observer_eval.py",
    "fcpg/lewm_camera_intervention.py",
    "fcpg/lewm_correction_derivation.py",
    "fcpg/lewm_pixel_observer.py",
    "fcpg/lewm_pixel_observer_runtime.py",
    "fcpg/lewm_sequence_data.py",
    "fcpg/transformer_grounding.py",
    "fcpg/observations.py",
    "fcpg/lewm_provenance.py",
)
MIN_CONFIRMATORY_EPISODES_PER_TASK = 30
REQUIRED_TASK_NAMES = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)
FORMAL_IMAGE_SIZE = (224, 224)
PRIMARY_CONFIRMATORY_SOURCES = ("rotation5", "translation4")
PRIMARY_EPISODES_PER_COHORT = 120
PRIMARY_WINDOWS_PER_COHORT = 7_800
CONFIRMATORY_PRIMARY_HORIZON = 5
CONFIRMATORY_MAX_TASK_RELATIVE_DEGRADATION = 0.10
CONFIRMATORY_MAX_RAW_FCPG_RELATIVE_DEGRADATION = 0.05
CONFIRMATORY_MAX_SOURCE_TASK_RELATIVE_DEGRADATION = 0.10
CONFIRMATORY_MIN_ELIGIBLE_ANCHOR_FRACTION = 0.95
CONFIRMATORY_MIN_CONTRIBUTING_WINDOW_FRACTION = 0.95
CONFIRMATORY_MIN_MEASUREMENT_COVERAGE = 0.95
CONFIRMATORY_RAW_FCPG_NONINFERIORITY_STATISTIC = (
    "bootstrap_relative_improvement_95_ci_lower"
)
CONFIRMATORY_SOURCE_TASK_NONINFERIORITY_STATISTIC = CONFIRMATORY_RAW_FCPG_NONINFERIORITY_STATISTIC
CONFIRMATORY_REQUIRE_EACH_PRIMARY_COHORT_CI_LOWER_POSITIVE = True
CONFIRMATORY_PRIMARY_COHORT_SOURCE_BINDING = {
    "rotation5": 0,
    "translation4": 1,
}
CONFIRMATORY_BOOTSTRAP_SAMPLES = 10_000
CONFIRMATORY_BOOTSTRAP_SEED = 6144
CONFIRMATORY_DESIGN_STATUS = "preregistered_before_alpha_selection"
CONFIRMATORY_NUMPY_VERSION = "1.26.4"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_camera_intervention import (  # noqa: E402
    validate_confirmatory_camera_interventions,
)
from fcpg.lewm_correction_derivation import (  # noqa: E402
    validate_fcpg_correction_derivations,
)
from fcpg.lewm_pixel_observer import LeWMPixelInnovationObserver  # noqa: E402
from fcpg.lewm_pixel_observer_eval import (  # noqa: E402
    ENDPOINT_HORIZONS,
    EvaluatorOracleDataset,
    evaluate_filter_batches,
    input_file_record,
    records_digest,
    reject_legacy_output_path,
    select_development_alpha,
    validate_protocol,
)
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.lewm_pixel_observer_runtime import (  # noqa: E402
    load_pixel_observer_checkpoint,
    validate_completed_training_budget,
)
from fcpg.lewm_sequence_data import LeWMSequenceDataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--observer-checkpoint", default=None)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument("--expected-observer-sha256", default=None)
    parser.add_argument("--phase", choices=("development", "confirmatory"), default=None)
    parser.add_argument("--alpha", action="append", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    return parser


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be an array")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _torch_payload(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return _mapping(value, str(path))


def _observer_bound_inputs(
    observer_path: Path,
    *,
    base_checkpoint_override: Path | None = None,
) -> tuple[Path, list[Path]]:
    payload = _torch_payload(observer_path)
    base = _mapping(payload.get("base_checkpoint"), "base_checkpoint")
    bound_base_path = _resolve(base.get("path"), base=observer_path.parent)
    actual_base_path = base_checkpoint_override or bound_base_path
    paths = [actual_base_path]
    artifacts = _sequence(payload.get("data_artifacts"), "data_artifacts")
    for index, record in enumerate(artifacts):
        artifact = _mapping(record, f"data_artifacts[{index}]")
        paths.append(_resolve(artifact.get("path"), base=observer_path.parent))
    binding = _mapping(payload.get("config_binding"), "config_binding")
    snapshot = binding.get("snapshot_path")
    if snapshot:
        paths.append(_resolve(snapshot, base=observer_path.parent))
    return actual_base_path, paths


def _resolve(value: object, *, base: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("required path is missing")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _expected_hash(path: Path, expected: object, label: str) -> str:
    actual = sha256_file(path)
    if expected is not None:
        normalized = str(expected).lower()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError(f"{label} expected SHA-256 is invalid")
        if actual != normalized:
            raise ValueError(f"{label} SHA-256 mismatch: expected {normalized}, got {actual}")
    return actual


def _protocol_bundle_paths() -> tuple[Path, ...]:
    """Files whose exact bytes define the cross-stage evaluation protocol."""

    return tuple((ROOT / value).resolve() for value in PROTOCOL_BUNDLE_RELATIVE_PATHS)


def _protocol_bundle_records() -> list[dict[str, object]]:
    return [input_file_record(path) for path in _protocol_bundle_paths()]


def _confirmatory_design_input_records(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, object]], list[Path]]:
    """Resolve the preregistered design and its frozen collection tooling."""

    binding = _mapping(
        config.get("confirmatory_design_binding"),
        "config.confirmatory_design_binding",
    )
    design_path = _resolve(binding.get("path"))
    if design_path != FORMAL_CONFIRMATORY_DESIGN_PATH:
        raise ValueError("confirmatory design must use the canonical path")
    design_sha = _expected_hash(
        design_path,
        binding.get("sha256"),
        "confirmatory design",
    )
    design = _load_json(design_path)
    if design.get("schema_version") != 1 or design.get("protocol") != (
        "lewm_pixel_observer_confirmatory_design_v1"
    ):
        raise ValueError("unexpected confirmatory design protocol")
    if design.get("status") != CONFIRMATORY_DESIGN_STATUS:
        raise ValueError(
            "confirmatory design is not preregistered before alpha selection"
        )

    statistics = _mapping(
        design.get("statistics"), "confirmatory design statistics"
    )
    expected_statistics = {
        "primary_horizon": CONFIRMATORY_PRIMARY_HORIZON,
        "bootstrap_samples": CONFIRMATORY_BOOTSTRAP_SAMPLES,
        "bootstrap_seed": CONFIRMATORY_BOOTSTRAP_SEED,
        "max_task_relative_degradation": (
            CONFIRMATORY_MAX_TASK_RELATIVE_DEGRADATION
        ),
        "max_raw_fcpg_relative_degradation": (
            CONFIRMATORY_MAX_RAW_FCPG_RELATIVE_DEGRADATION
        ),
        "max_source_task_relative_degradation": (
            CONFIRMATORY_MAX_SOURCE_TASK_RELATIVE_DEGRADATION
        ),
        "min_eligible_anchor_fraction": CONFIRMATORY_MIN_ELIGIBLE_ANCHOR_FRACTION,
        "min_contributing_window_fraction": (
            CONFIRMATORY_MIN_CONTRIBUTING_WINDOW_FRACTION
        ),
        "min_measurement_coverage": CONFIRMATORY_MIN_MEASUREMENT_COVERAGE,
        "raw_fcpg_noninferiority_statistic": (
            CONFIRMATORY_RAW_FCPG_NONINFERIORITY_STATISTIC
        ),
        "source_task_noninferiority_statistic": (
            CONFIRMATORY_SOURCE_TASK_NONINFERIORITY_STATISTIC
        ),
        "require_each_primary_cohort_ci_lower_positive": (
            CONFIRMATORY_REQUIRE_EACH_PRIMARY_COHORT_CI_LOWER_POSITIVE
        ),
        "primary_cohort_source_binding": (
            CONFIRMATORY_PRIMARY_COHORT_SOURCE_BINDING
        ),
    }
    if dict(statistics) != expected_statistics:
        raise ValueError(
            "confirmatory design statistics differ from frozen constants"
        )

    collection = _mapping(
        design.get("collection"), "confirmatory design collection"
    )
    declared_numpy = str(collection.get("numpy_version", ""))
    if (
        declared_numpy != CONFIRMATORY_NUMPY_VERSION
        or np.__version__ != declared_numpy
    ):
        raise ValueError(
            "confirmatory design NumPy version differs from the frozen runtime"
        )

    design_record = input_file_record(design_path)
    if design_record["sha256"] != design_sha:
        raise ValueError("confirmatory design record differs from its binding")
    records = [design_record]
    paths = [design_path]
    tooling = _mapping(design.get("tooling"), "confirmatory design tooling")
    if tuple(tooling) != ("collector", "camera_sampler"):
        raise ValueError("confirmatory design tooling set or order changed")
    for label in ("collector", "camera_sampler"):
        declared = _mapping(
            tooling.get(label), f"confirmatory tooling {label}"
        )
        tooling_path = _resolve(declared.get("path"))
        record = input_file_record(tooling_path)
        records.append(record)
        paths.append(tooling_path)
    return records, paths


def _episode_label(dataset: LeWMSequenceDataset, episode_index: int) -> str:
    raw = int(dataset.episodes[episode_index][0])
    return f"{dataset.task_name[raw]}:{dataset.episode_id[raw]}"


def _validate_manifest_partitions(
    source: Mapping[str, Any], source_name: str
) -> dict[str, set[str]]:
    aliases = {
        "train": "train_episodes",
        "validation": "val_episodes",
        "test": "test_episodes",
        "excluded": "excluded_episodes",
    }
    partitions: dict[str, set[str]] = {}
    for label, key in aliases.items():
        raw = source.get(key, [])
        values = {str(item) for item in _sequence(raw, f"{source_name}.{key}")}
        if len(values) != len(raw):
            raise ValueError(f"{source_name}.{key} contains duplicate episodes")
        partitions[label] = values
    for left, right in combinations(partitions, 2):
        overlap = partitions[left].intersection(partitions[right])
        if overlap:
            raise ValueError(
                f"episode manifest partitions overlap for {source_name}: "
                f"{left}/{right} {sorted(overlap)[:5]}"
            )
    return partitions


def select_manifest_episodes(
    dataset: LeWMSequenceDataset,
    *,
    source_name: str,
    episode_manifest: Mapping[str, Any],
    partition: str,
) -> Dataset:
    sources = _mapping(episode_manifest.get("sources"), "episode manifest sources")
    source = _mapping(sources.get(source_name), f"episode manifest source {source_name}")
    _validate_manifest_partitions(source, source_name)
    manifest_path = _resolve(source.get("path"))
    if manifest_path != dataset.path.expanduser().resolve():
        raise ValueError(
            f"episode manifest source path differs for {source_name}: "
            f"{manifest_path} != {dataset.path.resolve()}"
        )
    expected_hash = source.get("data_sha256")
    if not isinstance(expected_hash, str) or sha256_file(dataset.path) != expected_hash:
        raise ValueError(f"episode manifest source hash differs for {source_name}")
    key = {"train": "train_episodes", "validation": "val_episodes", "test": "test_episodes"}.get(
        partition
    )
    if key is None:
        raise ValueError("partition must be train, validation, or test")
    labels = {str(value) for value in _sequence(source.get(key), f"{source_name}.{key}")}
    if not labels:
        raise ValueError(f"{source_name}.{key} selects no episodes")
    selected = [
        index
        for index, (episode_index, _start) in enumerate(dataset.windows)
        if _episode_label(dataset, episode_index) in labels
    ]
    observed = {
        _episode_label(dataset, dataset.windows[index][0]) for index in selected
    }
    missing = sorted(labels.difference(observed))
    if missing:
        raise ValueError(f"source {source_name} is missing manifest episodes: {missing[:5]}")
    return Subset(dataset, selected)


def _correction_input_paths(cache: Path) -> list[Path]:
    if cache.is_file():
        return [cache]
    metadata = cache / "metadata.json"
    array = cache / "q_corrected.npy"
    if not metadata.is_file() or not array.is_file():
        raise FileNotFoundError(f"invalid correction cache: {cache}")
    paths = [metadata, array]
    payload = _load_json(metadata)
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, str) and checkpoint:
        paths.append(_resolve(checkpoint, base=metadata.parent))
    return paths


def _correction_checkpoint_record(cache: Path) -> dict[str, str]:
    if not cache.is_dir():
        raise ValueError(
            "PixelObserver-G0 requires directory correction caches with metadata"
        )
    metadata_path = cache / "metadata.json"
    metadata = _load_json(metadata_path)
    checkpoint_path = _resolve(
        metadata.get("checkpoint"), base=metadata_path.parent
    )
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
    }


def _source_file_record(path: Path, role: str) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} source is missing: {resolved}")
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def _checkpoint_source_paths(payload: Mapping[str, Any]) -> set[Path]:
    paths: set[Path] = set()

    def add_data_sources(value: object) -> None:
        if isinstance(value, str) and value:
            paths.add(_resolve(value))
            return
        if not isinstance(value, Mapping):
            return
        sources = value.get("sources", [])
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            return
        for source in sources:
            if isinstance(source, Mapping) and source.get("path"):
                paths.add(_resolve(source["path"]))

    resolved_value = payload.get("resolved_config", {})
    resolved_data = (
        resolved_value.get("data", {})
        if isinstance(resolved_value, Mapping)
        else {}
    )
    for container in (
        payload.get("data_config", {}),
        resolved_data,
    ):
        add_data_sources(container)
    values = payload.get("data_sources", [])
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for value in values:
            if isinstance(value, str) and value:
                paths.add(_resolve(value))
    source_manifest = payload.get("source_manifest", [])
    if isinstance(source_manifest, Sequence) and not isinstance(
        source_manifest, (str, bytes)
    ):
        for record in source_manifest:
            if isinstance(record, Mapping) and record.get("data"):
                paths.add(_resolve(record["data"]))
    args = payload.get("args", {})
    if isinstance(args, Mapping):
        add_data_sources(args.get("data"))
        extra = args.get("extra_data", [])
        if isinstance(extra, Sequence) and not isinstance(extra, (str, bytes)):
            for value in extra:
                if isinstance(value, str) and value:
                    paths.add(_resolve(value))
    return paths


def confirmatory_source_exposure_audit(
    *,
    observer_path: Path,
    base_path: Path,
    evaluation_sources: Sequence[Path],
    correction_caches: Sequence[Path],
) -> tuple[dict[str, object], list[Path]]:
    """Prove fresh evaluation at file level for every learned component.

    A confirmatory source must be a different file *and* SHA-256 from all data
    used by the adapter, its frozen LeWM, and the FCPG corrector. This is stricter
    than trusting a hand-written episode flag and intentionally rejects the old
    train/validation NPZs even when a new manifest selects different episodes.
    """

    observer_payload = _torch_payload(observer_path)
    if len(evaluation_sources) != len(correction_caches):
        raise ValueError("evaluation sources and correction caches must be paired")
    seen: list[dict[str, str]] = []
    observer_artifacts = _sequence(
        observer_payload.get("data_artifacts"), "observer data_artifacts"
    )
    for raw in observer_artifacts:
        record = _mapping(raw, "observer data artifact")
        if record.get("kind") == "source_data":
            path = _resolve(record.get("path"), base=observer_path.parent)
            current = _source_file_record(path, "adapter_train")
            if current["sha256"] != record.get("sha256"):
                raise ValueError("adapter source binding changed before exposure audit")
            seen.append(current)

    base_payload = _torch_payload(base_path)
    for path in sorted(_checkpoint_source_paths(base_payload)):
        seen.append(_source_file_record(path, "base_lewm_train"))

    corrector_checkpoints: set[Path] = set()
    for evaluation_source, cache in zip(evaluation_sources, correction_caches):
        metadata_path = (
            cache / "metadata.json"
            if cache.is_dir()
            else cache.parent / "metadata.json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"confirmatory correction cache lacks metadata: {metadata_path}"
            )
        metadata = _load_json(metadata_path)
        if int(metadata.get("schema_version", 0)) < 3:
            raise ValueError(
                "confirmatory evaluation requires a schema-3 correction cache"
            )
        declared_source = _resolve(
            metadata.get("source_data"), base=metadata_path.parent
        )
        if declared_source != evaluation_source.resolve():
            raise ValueError("correction cache source path differs from evaluation data")
        if metadata.get("source_data_sha256") != sha256_file(evaluation_source):
            raise ValueError("correction cache source hash differs from evaluation data")
        frame_identity = str(metadata.get("frame_identity_sha256", ""))
        if len(frame_identity) != 64:
            raise ValueError("correction cache lacks a bound frame identity hash")
        if metadata.get("q_true_usage") != "offline_metrics_only":
            raise ValueError("correction cache has an invalid q_true usage contract")
        correction_path = cache / "q_corrected.npy" if cache.is_dir() else cache
        if metadata.get("q_corrected_sha256") != sha256_file(correction_path):
            raise ValueError("correction cache array hash differs from metadata")
        checkpoint_value = metadata.get("checkpoint")
        if not isinstance(checkpoint_value, str) or not checkpoint_value:
            raise ValueError("confirmatory correction metadata lacks a corrector checkpoint")
        checkpoint_path = _resolve(checkpoint_value, base=metadata_path.parent)
        if metadata.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError("correction checkpoint hash differs from metadata")
        corrector_checkpoints.add(checkpoint_path)
    for checkpoint in sorted(corrector_checkpoints):
        corrector_payload = _torch_payload(checkpoint)
        paths = _checkpoint_source_paths(corrector_payload)
        if not paths:
            raise ValueError(
                f"cannot prove corrector training exposure from {checkpoint}"
            )
        for path in sorted(paths):
            seen.append(_source_file_record(path, "fcpg_corrector_train"))

    roles = {record["role"] for record in seen}
    required = {"adapter_train", "base_lewm_train", "fcpg_corrector_train"}
    if not required.issubset(roles):
        raise ValueError(
            "confirmatory exposure audit lacks learned-component source records: "
            f"{sorted(required.difference(roles))}"
        )
    evaluated = [_source_file_record(path, "confirmatory_test") for path in evaluation_sources]
    seen_paths = {record["path"] for record in seen}
    seen_hashes = {record["sha256"] for record in seen}
    overlap = [
        record
        for record in evaluated
        if record["path"] in seen_paths or record["sha256"] in seen_hashes
    ]
    if overlap:
        raise ValueError(
            "confirmatory source was exposed to a learned component: "
            f"{[record['path'] for record in overlap]}"
        )
    bound_paths = {
        Path(record["path"]) for record in (*seen, *evaluated)
    }.union(corrector_checkpoints)
    return {
        "passed": True,
        "criterion": "distinct_path_and_sha256_from_adapter_base_and_corrector_training_sources",
        "evaluation_sources": evaluated,
        "seen_sources": seen,
        "corrector_checkpoints": [str(path) for path in sorted(corrector_checkpoints)],
    }, sorted(bound_paths)


def validate_confirmatory_coverage(
    split_config: Mapping[str, Any],
    source_report: Sequence[Mapping[str, Any]],
    *,
    require_primary_cohorts: bool = False,
) -> dict[str, object]:
    expected_sources = [
        str(value)
        for value in _sequence(
            split_config.get("expected_sources"), "split.expected_sources"
        )
    ]
    expected_episodes = _mapping(
        split_config.get("expected_episode_counts"),
        "split.expected_episode_counts",
    )
    expected_windows = _mapping(
        split_config.get("expected_window_counts"),
        "split.expected_window_counts",
    )
    observed_names = [str(record["name"]) for record in source_report]
    if require_primary_cohorts and expected_sources != list(
        PRIMARY_CONFIRMATORY_SOURCES
    ):
        raise ValueError(
            "confirmatory G0 requires independent rotation5 and translation4 cohorts"
        )
    if observed_names != expected_sources:
        raise ValueError(
            "confirmatory source order/set differs from its frozen coverage ledger"
        )
    task_counts: dict[str, int] = {}
    for record in source_report:
        name = str(record["name"])
        if int(record["eligible_episodes"]) != int(expected_episodes.get(name, -1)):
            raise ValueError(f"confirmatory episode count differs for {name}")
        if int(record["eligible_windows"]) != int(expected_windows.get(name, -1)):
            raise ValueError(f"confirmatory window count differs for {name}")
        for task, count in _mapping(
            record.get("episodes_per_task"), f"{name}.episodes_per_task"
        ).items():
            task_counts[str(task)] = task_counts.get(str(task), 0) + int(count)
        if require_primary_cohorts:
            cohort_counts = {
                str(task): int(count)
                for task, count in _mapping(
                    record.get("episodes_per_task"),
                    f"{name}.episodes_per_task",
                ).items()
            }
            if (
                set(cohort_counts) != set(REQUIRED_TASK_NAMES)
                or any(count != 30 for count in cohort_counts.values())
                or int(record["eligible_episodes"])
                != PRIMARY_EPISODES_PER_COHORT
                or int(record["eligible_windows"])
                != PRIMARY_WINDOWS_PER_COHORT
            ):
                raise ValueError(
                    f"confirmatory cohort {name} must contain exactly 30 episodes "
                    "and 1950 windows per task"
                )
    insufficient = {
        task: count
        for task, count in task_counts.items()
        if count < MIN_CONFIRMATORY_EPISODES_PER_TASK
    }
    if not task_counts or insufficient:
        raise ValueError(
            "confirmatory coverage requires at least "
            f"{MIN_CONFIRMATORY_EPISODES_PER_TASK} episodes per task: {insufficient}"
        )
    expected_tasks = {
        str(value)
        for value in _sequence(
            split_config.get("expected_tasks"), "split.expected_tasks"
        )
    }
    if expected_tasks != set(REQUIRED_TASK_NAMES):
        raise ValueError(
            "PixelObserver-G0 requires the frozen four-task suite "
            f"{REQUIRED_TASK_NAMES}"
        )
    if set(task_counts) != expected_tasks:
        raise ValueError("confirmatory task set differs from its frozen coverage ledger")
    return {
        "passed": True,
        "expected_sources": expected_sources,
        "expected_episode_counts": {
            name: int(expected_episodes[name]) for name in expected_sources
        },
        "expected_window_counts": {
            name: int(expected_windows[name]) for name in expected_sources
        },
        "episodes_per_task": task_counts,
        "minimum_episodes_per_task": MIN_CONFIRMATORY_EPISODES_PER_TASK,
    }


def _physical_episode_content_hashes(path: Path) -> list[dict[str, object]]:
    required = (
        "task_name",
        "episode_id",
        "frame_id",
        "obs",
        "action",
        "keypoints_xyz",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(
                f"physical-identity audit is missing NPZ fields: {missing}"
            )
        task_name = np.asarray(archive["task_name"]).astype(str)
        episode_id = np.asarray(archive["episode_id"]).astype(np.int64)
        frame_id = np.asarray(archive["frame_id"]).astype(np.int64)
        groups = sorted(
            {
                (str(task), int(episode))
                for task, episode in zip(task_name.tolist(), episode_id.tolist())
            }
        )
        records = []
        arrays = {
            key: np.asarray(archive[key])
            for key in ("obs", "action", "keypoints_xyz")
        }
        for task, episode in groups:
            indices = np.flatnonzero(
                (task_name == task) & (episode_id == episode)
            )
            order = np.argsort(frame_id[indices], kind="stable")
            indices = indices[order]
            ordered_frames = frame_id[indices]
            if len(indices) != 100 or not np.array_equal(
                ordered_frames, np.arange(100, dtype=np.int64)
            ):
                raise ValueError(
                    f"physical episode {task}:{episode} must contain frames 0..99"
                )
            digest = hashlib.sha256()
            digest.update(b"pixel-observer-physical-content-v2\0")
            for key, array in arrays.items():
                selected = np.ascontiguousarray(
                    np.asarray(array[indices], dtype="<f4")
                )
                digest.update(key.encode("ascii"))
                digest.update(b"\0")
                digest.update(selected.dtype.str.encode("ascii"))
                digest.update(b"\0")
                digest.update(
                    json.dumps(list(selected.shape), separators=(",", ":")).encode(
                        "ascii"
                    )
                )
                digest.update(b"\0")
                digest.update(selected.tobytes(order="C"))
            records.append(
                {
                    "task": task,
                    "episode": episode,
                    "content_sha256": digest.hexdigest(),
                }
            )
    return records


def validate_physical_episode_independence(
    source_report: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    by_source = {}
    all_hashes: dict[str, str] = {}
    for source in source_report:
        name = str(source["name"])
        records = _physical_episode_content_hashes(_resolve(source.get("data")))
        hashes = {str(record["content_sha256"]) for record in records}
        if len(hashes) != len(records):
            raise ValueError(f"physical episodes repeat within source {name}")
        overlap = sorted(hashes.intersection(all_hashes))
        if overlap:
            raise ValueError(
                "confirmatory cohorts reuse physical episode content: "
                f"{name} overlaps {sorted({all_hashes[value] for value in overlap})}"
            )
        for value in hashes:
            all_hashes[value] = name
        ledger_digest = hashlib.sha256(
            "\n".join(sorted(hashes)).encode("ascii")
        ).hexdigest()
        by_source[name] = {
            "episodes": len(records),
            "content_set_sha256": ledger_digest,
        }
    return {
        "passed": True,
        "method": (
            "sha256(float32_le(obs,action,keypoints_xyz)) over ordered frames "
            "0..99; task_name, episode_id, and task_index are excluded"
        ),
        "cross_source_overlap_count": 0,
        "sources": by_source,
    }


def validate_development_confirmatory_independence(
    development_sources: Sequence[Mapping[str, Any]],
    confirmatory_sources: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Reject both byte-identical and repacked selector episodes."""

    development_records = []
    for source in development_sources:
        path = _resolve(source.get("path"))
        actual_hash = sha256_file(path)
        if actual_hash != source.get("sha256"):
            raise ValueError("development source changed after alpha selection")
        development_records.append(
            {"name": str(source.get("name")), "path": str(path), "sha256": actual_hash}
        )
    confirmatory_records = [
        {
            "name": str(source.get("name")),
            "path": str(_resolve(source.get("data"))),
            "sha256": str(source.get("data_sha256")),
        }
        for source in confirmatory_sources
    ]
    development_paths = {record["path"] for record in development_records}
    development_hashes = {record["sha256"] for record in development_records}
    overlap = [
        record
        for record in confirmatory_records
        if record["path"] in development_paths
        or record["sha256"] in development_hashes
    ]
    if overlap:
        raise ValueError(
            "confirmatory source overlaps development alpha-selection data: "
            f"{[record['path'] for record in overlap]}"
        )

    development_content: dict[str, str] = {}
    for record in development_records:
        for episode in _physical_episode_content_hashes(Path(record["path"])):
            digest = str(episode["content_sha256"])
            development_content.setdefault(digest, record["name"])
    confirmatory_content: dict[str, str] = {}
    for record in confirmatory_records:
        for episode in _physical_episode_content_hashes(Path(record["path"])):
            digest = str(episode["content_sha256"])
            owner = development_content.get(digest)
            if owner is not None:
                raise ValueError(
                    "confirmatory source reuses development physical content: "
                    f"{record['name']} overlaps {owner}"
                )
            confirmatory_content.setdefault(digest, record["name"])
    return {
        "passed": True,
        "criterion": "distinct_path_sha256_and_physical_episode_content_from_development",
        "development_sources": development_records,
        "confirmatory_sources": confirmatory_records,
        "development_episode_count": len(development_content),
        "confirmatory_episode_count": len(confirmatory_content),
        "cross_split_overlap_count": 0,
    }


def qualify_development_selection(
    selection: Mapping[str, Any],
    *,
    checkpoint_report: Mapping[str, object],
    coverage_audit: Mapping[str, object],
    correction_derivation_audit: Mapping[str, object],
    max_batches: int | None,
    git_state: Mapping[str, object],
) -> dict[str, object]:
    qualified = dict(selection)
    failures = []
    if max_batches is not None:
        failures.append("development evaluation was batch-truncated")
    if coverage_audit.get("passed") is not True:
        failures.append("development coverage ledger did not pass")
    if correction_derivation_audit.get("passed") is not True:
        failures.append("development correction derivation audit did not pass")
    if git_state.get("dirty") is not False:
        failures.append("development evaluation did not use a clean git worktree")
    try:
        _validate_confirmatory_checkpoint_report(checkpoint_report)
    except (TypeError, ValueError) as exc:
        failures.append(f"checkpoint is not confirmatory-grade: {exc}")
    if selection.get("status") != "candidate_selected":
        failures.append("no eligible interior fusion alpha")
    qualified["freeze_eligibility"] = {
        "passed": not failures,
        "failures": failures,
        "requires_untruncated_full_four_task_matrix": True,
        "requires_formal_completed_observer_and_base": True,
        "requires_raw_rgb_correction_derivation": True,
        "requires_clean_git_worktree": True,
    }
    qualified["status"] = "selected" if not failures else "ineligible"
    if failures:
        qualified["selected_alpha"] = None
    return qualified


def _canonical_correction_sink(
    datasets_by_source: Mapping[str, LeWMSequenceDataset],
    applications: dict[str, dict[str, object]],
):
    """Install raw-RGB replayed FCPG coordinates into the scored datasets."""

    def install(
        source_name: str,
        frame_indices: np.ndarray,
        corrected_px: np.ndarray,
    ) -> None:
        if source_name in applications:
            raise ValueError(
                f"canonical FCPG corrections were emitted twice for {source_name}"
            )
        dataset = datasets_by_source.get(source_name)
        if dataset is None:
            raise KeyError(
                f"canonical FCPG corrections reference unknown source {source_name}"
            )
        indices = np.asarray(frame_indices, dtype=np.int64)
        values = np.asarray(corrected_px, dtype=np.float32)
        cached = getattr(dataset, "q_corrected", None)
        if cached is None:
            raise ValueError(f"source {source_name} has no FCPG correction array")
        cached_array = np.asarray(cached)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("canonical FCPG frame indices must be nonempty 1-D")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("canonical FCPG frame indices must be unique")
        if indices.min() < 0 or indices.max() >= len(cached_array):
            raise IndexError("canonical FCPG frame indices are out of range")
        expected = (len(indices), *cached_array.shape[1:])
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(
                f"canonical FCPG output for {source_name} must have shape {expected} "
                "and contain only finite values"
            )
        canonical = np.array(cached_array, dtype=np.float32, copy=True)
        canonical[indices] = values
        dataset.q_corrected = canonical
        applications[source_name] = {
            "passed": True,
            "scored_frames_replaced": int(len(indices)),
            "source_array_frames": int(len(canonical)),
            "mode": "raw_rgb_replay_replaces_cached_coordinates_before_scoring",
        }

    return install


def _validate_development_canonical_replay(
    report: Mapping[str, Any],
    development_sources: Sequence[Mapping[str, Any]],
) -> None:
    """Recheck that alpha selection scored raw-RGB replayed corrections."""

    split = _mapping(report.get("split"), "development split")
    audit = _mapping(
        split.get("correction_derivation_audit"),
        "development correction derivation audit",
    )
    if (
        audit.get("passed") is not True
        or audit.get("protocol") != "fcpg_raw_rgb_correction_derivation_v1"
        or audit.get("all_scored_frames_replayed") is not True
        or audit.get("canonical_scoring_outputs_emitted") is not True
    ):
        raise ValueError(
            "development correction derivation audit is incomplete or failed"
        )
    source_audits = _mapping(
        audit.get("sources"), "development correction derivation sources"
    )
    canonical = _mapping(
        audit.get("canonical_scoring"),
        "development canonical scoring audit",
    )
    applications = _mapping(
        canonical.get("applications"),
        "development canonical scoring applications",
    )
    names = [str(source.get("name")) for source in development_sources]
    if (
        canonical.get("passed") is not True
        or canonical.get("required_sources") != sorted(names)
        or set(source_audits) != set(names)
        or set(applications) != set(names)
    ):
        raise ValueError(
            "development canonical scoring source coverage is incomplete"
        )
    for source in development_sources:
        name = str(source.get("name"))
        expected_frames = int(source.get("eligible_episodes", -1)) * 100
        source_audit = _mapping(
            source_audits.get(name),
            f"development correction derivation source {name}",
        )
        application = _mapping(
            applications.get(name),
            f"development canonical scoring application {name}",
        )
        if (
            expected_frames <= 0
            or source_audit.get("passed") is not True
            or source_audit.get("canonical_scoring_output_emitted") is not True
            or int(source_audit.get("scored_frames", -1)) != expected_frames
            or application.get("passed") is not True
            or int(application.get("scored_frames_replaced", -1))
            != expected_frames
        ):
            raise ValueError(
                f"development canonical scoring frame coverage differs for {name}"
            )


def validate_confirmatory_development_binding(
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[dict[str, object], Path]:
    binding = _mapping(
        config.get("development_binding"), "config.development_binding"
    )
    report_path = _resolve(binding.get("report"))
    report_sha = _expected_hash(
        report_path,
        binding.get("report_sha256"),
        "development report",
    )
    report = _load_json(report_path)
    development_provenance = _mapping(
        report.get("provenance"), "development provenance"
    )
    canonical_config_record = input_file_record(FORMAL_DEVELOPMENT_CONFIG_PATH)
    if development_provenance.get("config") != canonical_config_record:
        raise ValueError(
            "development report was not produced by the canonical config"
        )
    input_records = [
        _mapping(value, "development input record")
        for value in _sequence(
            development_provenance.get("input_records"),
            "development input records",
        )
    ]
    if records_digest(input_records) != development_provenance.get("input_digest"):
        raise ValueError("development input-record digest changed")
    recorded_inputs = {str(record.get("path")): record for record in input_records}
    if recorded_inputs.get(str(FORMAL_DEVELOPMENT_CONFIG_PATH)) != canonical_config_record:
        raise ValueError("canonical development config is absent from input records")
    canonical_config = _load_json(FORMAL_DEVELOPMENT_CONFIG_PATH)
    if (
        canonical_config.get("protocol") != "lewm_pixel_observer_g0_fixed_alpha"
        or canonical_config.get("phase") != "development"
    ):
        raise ValueError("canonical development config contract changed")
    current_design_records, _ = _confirmatory_design_input_records(config)
    canonical_design_records, _ = _confirmatory_design_input_records(
        canonical_config
    )
    if current_design_records != canonical_design_records:
        raise ValueError(
            "confirmatory design binding differs from development selection"
        )
    for record in canonical_design_records:
        if recorded_inputs.get(str(record["path"])) != record:
            raise ValueError(
                "development report lacks the frozen confirmatory design/tooling "
                f"record: {record['path']}"
            )
    protocol_bundle_records = _protocol_bundle_records()
    for record in protocol_bundle_records:
        if recorded_inputs.get(str(record["path"])) != record:
            raise ValueError(
                "development report lacks or changed a cross-stage protocol "
                f"bundle record: {record['path']}"
            )
    if report.get("protocol") != "lewm_pixel_observer_g0_report_v1":
        raise ValueError("confirmatory binding uses an unexpected development report")
    if report.get("phase") != "development":
        raise ValueError("confirmatory binding requires a development-phase report")
    development_evaluation = _mapping(
        report.get("evaluation"), "development report evaluation"
    )
    if development_evaluation.get("protocol") != "lewm_pixel_observer_g0_filter_v1":
        raise ValueError("development report uses an unexpected evaluator protocol")
    development_contract = _mapping(
        development_evaluation.get("contract"), "development evaluation contract"
    )
    development_alphas = tuple(
        float(value)
        for value in _sequence(
            development_contract.get("alphas"), "development alpha grid"
        )
    )
    development_horizons = tuple(
        int(value)
        for value in _sequence(
            development_contract.get("horizons"), "development horizons"
        )
    )
    validate_protocol(
        phase="development",
        alphas=development_alphas,
        horizons=development_horizons,
    )
    selection = _mapping(
        development_evaluation.get("development_selection"),
        "development selection",
    )
    eligibility = _mapping(
        selection.get("freeze_eligibility"),
        "development freeze eligibility",
    )
    if selection.get("status") != "selected" or eligibility.get("passed") is not True:
        raise ValueError("development report did not freeze an eligible alpha")
    recomputed = select_development_alpha(
        alphas=development_alphas,
        comparisons=_mapping(
            development_evaluation.get("paired_comparisons"),
            "development paired comparisons",
        ),
        macro=_mapping(development_evaluation.get("macro"), "development macro"),
        primary_horizon=CONFIRMATORY_PRIMARY_HORIZON,
        max_task_relative_degradation=(
            CONFIRMATORY_MAX_TASK_RELATIVE_DEGRADATION
        ),
    )
    for key in (
        "selected_alpha",
        "selection_uses",
        "objective",
        "eligibility",
        "tie_break",
        "oracle_consulted",
        "secondary_horizons_consulted",
        "diagnostic_endpoint_alphas",
        "candidates",
    ):
        if selection.get(key) != recomputed.get(key):
            raise ValueError(
                f"development selection differs from recomputation at {key}"
            )
    selected_alpha = float(selection.get("selected_alpha"))
    if tuple(settings["alphas"]) != (selected_alpha,):
        raise ValueError("confirmatory alpha differs from the development selection")
    development_checkpoint = _mapping(
        report.get("checkpoint"), "development checkpoint"
    )
    _validate_confirmatory_checkpoint_report(development_checkpoint)
    development_base = _mapping(
        development_checkpoint.get("base_checkpoint"),
        "development base checkpoint",
    )
    expected_observer = str(config.get("observer_checkpoint_sha256", "")).lower()
    expected_base = str(config.get("base_checkpoint_sha256", "")).lower()
    if development_checkpoint.get("sha256") != expected_observer:
        raise ValueError(
            "confirmatory observer hash differs from the development report"
        )
    if development_base.get("sha256") != expected_base:
        raise ValueError("confirmatory base hash differs from the development report")
    development_coverage = _mapping(
        _mapping(report.get("split"), "development split").get("coverage_audit"),
        "development coverage audit",
    )
    if development_coverage.get("passed") is not True:
        raise ValueError("development report lacks complete frozen coverage")
    development_sources = [
        _mapping(value, "development source")
        for value in _sequence(report.get("sources"), "development sources")
    ]
    _validate_development_canonical_replay(
        report, development_sources
    )
    canonical_sources = [
        _mapping(value, "canonical development source")
        for value in _sequence(
            canonical_config.get("sources"), "canonical development sources"
        )
    ]
    if len(canonical_sources) != len(development_sources):
        raise ValueError("development report source count differs from canonical config")
    for declared, observed in zip(
        canonical_sources, development_sources, strict=True
    ):
        name = str(declared.get("name"))
        if observed.get("name") != name:
            raise ValueError("development source order differs from canonical config")
        data_path = _resolve(declared.get("path"))
        cache_path = _resolve(declared.get("correction_cache"))
        data_record = input_file_record(data_path)
        if (
            observed.get("data") != str(data_path)
            or observed.get("data_sha256") != data_record["sha256"]
            or observed.get("correction_cache") != str(cache_path)
        ):
            raise ValueError(f"development source binding differs for {name}")
        if recorded_inputs.get(str(data_path)) != data_record:
            raise ValueError(f"development source {name} is absent from input records")
    canonical_split = _mapping(
        canonical_config.get("split"), "canonical development split"
    )
    canonical_manifest = input_file_record(
        _resolve(canonical_split.get("episode_manifest"))
    )
    if (
        development_provenance.get("episode_manifest") != canonical_manifest
        or recorded_inputs.get(str(canonical_manifest["path"]))
        != canonical_manifest
    ):
        raise ValueError("development episode manifest binding changed")
    if (
        canonical_config.get("observer_checkpoint_sha256") != expected_observer
        or canonical_config.get("base_checkpoint_sha256") != expected_base
    ):
        raise ValueError("canonical development checkpoint pins changed")
    recomputed_coverage = validate_confirmatory_coverage(
        {
            "expected_sources": development_coverage.get("expected_sources"),
            "expected_tasks": list(REQUIRED_TASK_NAMES),
            "expected_episode_counts": development_coverage.get(
                "expected_episode_counts"
            ),
            "expected_window_counts": development_coverage.get(
                "expected_window_counts"
            ),
        },
        development_sources,
    )
    if recomputed_coverage != development_coverage:
        raise ValueError("development coverage differs from recomputation")
    development_corrector_hashes = sorted(
        {
            str(
                _mapping(
                    source.get("corrector_checkpoint"),
                    "development corrector checkpoint",
                ).get("sha256")
            )
            for source in development_sources
        }
    )
    if len(development_corrector_hashes) != 1:
        raise ValueError(
            "development must use one shared frozen FCPG corrector checkpoint"
        )
    if development_provenance.get("inputs_unchanged") is not True:
        raise ValueError("development report did not preserve frozen inputs")
    development_git = _mapping(
        development_provenance.get("git"), "development git state"
    )
    if development_git.get("dirty") is not False:
        raise ValueError("development report did not use a clean git worktree")
    if _mapping(report.get("runtime"), "development runtime").get("max_batches") is not None:
        raise ValueError("development report was batch-truncated")
    return {
        "passed": True,
        "report": str(report_path),
        "report_sha256": report_sha,
        "selected_alpha": selected_alpha,
        "observer_checkpoint_sha256": expected_observer,
        "base_checkpoint_sha256": expected_base,
        "development_sources": [
            {
                "name": str(source["name"]),
                "path": str(source["data"]),
                "sha256": str(source["data_sha256"]),
            }
            for source in development_sources
        ],
        "corrector_checkpoint_sha256": development_corrector_hashes[0],
        "confirmatory_design_records": canonical_design_records,
        "protocol_bundle_records": protocol_bundle_records,
    }, report_path


def _git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status": status}


def _require_clean_confirmatory_git(
    phase: str, git_state: Mapping[str, object]
) -> None:
    if phase == "confirmatory" and git_state.get("dirty") is not False:
        raise ValueError("confirmatory evaluation requires a clean git worktree")


def _assert_evaluation_state_unchanged(
    before_records: Sequence[Mapping[str, object]],
    before_digest: str,
    git_before: Mapping[str, object],
) -> None:
    after_records = [input_file_record(record["path"]) for record in before_records]
    after_digest = records_digest(after_records)
    if after_records != list(before_records) or after_digest != before_digest:
        raise RuntimeError("PixelObserver-G0 input changed during evaluation")
    if _git_state() != git_before:
        raise RuntimeError("repository state changed during PixelObserver-G0 evaluation")


def _exclusive_atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite PixelObserver-G0 report: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite PixelObserver-G0 report: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _load_observer(
    *,
    observer_path: Path,
    base_checkpoint_override: Path | None,
    expected_observer_sha256: object,
    expected_base_sha256: object,
    device: torch.device,
    sample_dataset: LeWMSequenceDataset,
) -> tuple[LeWMPixelInnovationObserver, dict[str, object]]:
    observer_sha = _expected_hash(
        observer_path, expected_observer_sha256, "observer checkpoint"
    )
    observer, resolved_config, payload = load_pixel_observer_checkpoint(
        observer_path,
        device=device,
        base_checkpoint=base_checkpoint_override,
        dataset=sample_dataset,
        verify_artifacts=True,
    )
    supervision = _mapping(payload.get("supervision_contract"), "supervision_contract")
    if (
        supervision.get("q_true_forbidden") is not True
        or supervision.get("oracle_target_forbidden") is not True
        or supervision.get("base_grounded_lewm_frozen") is not True
    ):
        raise ValueError("observer checkpoint does not enforce oracle-free training")
    base_binding = _mapping(payload.get("base_checkpoint"), "base_checkpoint")
    base_path = (
        base_checkpoint_override.expanduser().resolve()
        if base_checkpoint_override is not None
        else _resolve(base_binding.get("path"), base=observer_path.parent)
    )
    base_sha = _expected_hash(
        base_path, base_binding.get("sha256"), "base checkpoint contract"
    )
    if expected_base_sha256 is not None:
        _expected_hash(base_path, expected_base_sha256, "base checkpoint CLI pin")
    return observer, {
        "path": str(observer_path),
        "sha256": observer_sha,
        "checkpoint_type": payload.get("checkpoint_type"),
        "step": int(payload.get("step", 0)),
        "epoch": int(payload.get("epoch", 0)),
        "run_kind": payload.get("run_kind"),
        "training_budget": payload.get("training_budget"),
        "base_checkpoint": {
            "path": str(base_path),
            "sha256": base_sha,
            "step": base_binding.get("step"),
            "run_kind": base_binding.get("run_kind"),
            "supervision_audit": base_binding.get("supervision_audit"),
            "training_budget": base_binding.get("training_budget"),
            "provenance": base_binding.get("provenance"),
        },
        "adapter_config": payload.get("adapter_config"),
        "supervision_contract": dict(supervision),
        "data_artifacts": payload.get("data_artifacts"),
        "config_binding": payload.get("config_binding"),
        "resolved_observer_config": resolved_config,
    }


def _resolve_settings(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    phase = str(args.phase or config.get("phase", "development"))
    if phase == "confirmatory":
        if str(config.get("phase", "")) != "confirmatory":
            raise ValueError("confirmatory phase must be frozen in the config")
        forbidden_overrides = {
            "--alpha": args.alpha,
            "--max-batches": args.max_batches,
            "--bootstrap-samples": args.bootstrap_samples,
            "--bootstrap-seed": args.bootstrap_seed,
            "--expected-observer-sha256": args.expected_observer_sha256,
            "--expected-checkpoint-sha256": args.expected_checkpoint_sha256,
            "--batch-size": args.batch_size,
            "--num-workers": args.num_workers,
            "--output": args.output,
        }
        used = [name for name, value in forbidden_overrides.items() if value is not None]
        if used:
            raise ValueError(
                "confirmatory evaluation forbids protocol CLI overrides: "
                f"{used}"
            )
    alphas = tuple(args.alpha) if args.alpha is not None else tuple(evaluation.get("alphas", ()))
    horizons = tuple(evaluation.get("horizons", ENDPOINT_HORIZONS))
    alphas, horizons = validate_protocol(phase=phase, alphas=alphas, horizons=horizons)
    values = {
        "phase": phase,
        "alphas": alphas,
        "horizons": horizons,
        "history_size": int(evaluation.get("history_size", 3)),
        "rollout_steps": int(evaluation.get("rollout_steps", 5)),
        "frame_skip": int(evaluation.get("frame_skip", 5)),
        "batch_size": int(args.batch_size or evaluation.get("batch_size", 16)),
        "num_workers": int(
            args.num_workers if args.num_workers is not None else evaluation.get("num_workers", 0)
        ),
        "bootstrap_samples": int(
            args.bootstrap_samples or evaluation.get("bootstrap_samples", 10_000)
        ),
        "bootstrap_seed": int(
            args.bootstrap_seed
            if args.bootstrap_seed is not None
            else evaluation.get("bootstrap_seed", 6144)
        ),
        "primary_horizon": int(evaluation.get("primary_horizon", 5)),
        "max_task_relative_degradation": float(
            evaluation.get("max_task_relative_degradation", 0.10)
        ),
        "max_raw_fcpg_relative_degradation": float(
            evaluation.get("max_raw_fcpg_relative_degradation", 0.05)
        ),
        "max_source_task_relative_degradation": float(
            evaluation.get("max_source_task_relative_degradation", 0.10)
        ),
        "min_eligible_anchor_fraction": float(
            evaluation.get("min_eligible_anchor_fraction", 0.95)
        ),
        "min_contributing_window_fraction": float(
            evaluation.get("min_contributing_window_fraction", 0.95)
        ),
        "min_measurement_coverage": float(
            evaluation.get("min_measurement_coverage", 0.95)
        ),
        "partition": str(evaluation.get("partition", "validation")),
    }
    if values["rollout_steps"] != max(horizons):
        raise ValueError("rollout_steps must equal the maximum endpoint horizon")
    if values["history_size"] <= 0 or values["frame_skip"] <= 0:
        raise ValueError("history_size and frame_skip must be positive")
    if values["batch_size"] <= 0 or values["num_workers"] < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if phase in {"development", "confirmatory"}:
        frozen = {
            "horizons": (tuple(values["horizons"]), ENDPOINT_HORIZONS),
            "history_size": (values["history_size"], 3),
            "rollout_steps": (values["rollout_steps"], 5),
            "frame_skip": (values["frame_skip"], 5),
            "bootstrap_samples": (
                values["bootstrap_samples"],
                CONFIRMATORY_BOOTSTRAP_SAMPLES,
            ),
            "bootstrap_seed": (
                values["bootstrap_seed"],
                CONFIRMATORY_BOOTSTRAP_SEED,
            ),
            "primary_horizon": (
                values["primary_horizon"],
                CONFIRMATORY_PRIMARY_HORIZON,
            ),
            "max_task_relative_degradation": (
                values["max_task_relative_degradation"],
                CONFIRMATORY_MAX_TASK_RELATIVE_DEGRADATION,
            ),
            "max_raw_fcpg_relative_degradation": (
                values["max_raw_fcpg_relative_degradation"],
                CONFIRMATORY_MAX_RAW_FCPG_RELATIVE_DEGRADATION,
            ),
            "max_source_task_relative_degradation": (
                values["max_source_task_relative_degradation"],
                CONFIRMATORY_MAX_SOURCE_TASK_RELATIVE_DEGRADATION,
            ),
            "min_eligible_anchor_fraction": (
                values["min_eligible_anchor_fraction"],
                CONFIRMATORY_MIN_ELIGIBLE_ANCHOR_FRACTION,
            ),
            "min_contributing_window_fraction": (
                values["min_contributing_window_fraction"],
                CONFIRMATORY_MIN_CONTRIBUTING_WINDOW_FRACTION,
            ),
            "min_measurement_coverage": (
                values["min_measurement_coverage"],
                CONFIRMATORY_MIN_MEASUREMENT_COVERAGE,
            ),
            "partition": (
                values["partition"],
                "test" if phase == "confirmatory" else "validation",
            ),
        }
        mismatched = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in frozen.items()
            if actual != expected
        }
        if mismatched:
            raise ValueError(
                f"{phase} protocol differs from frozen constants: {mismatched}"
            )
    return values


def _validate_confirmatory_checkpoint_report(
    checkpoint_report: Mapping[str, object],
) -> None:
    if checkpoint_report.get("run_kind") != "formal":
        raise ValueError("confirmatory evaluation requires a formal observer checkpoint")
    base_report = _mapping(
        checkpoint_report.get("base_checkpoint"), "base checkpoint report"
    )
    if base_report.get("run_kind") != "formal":
        raise ValueError("confirmatory evaluation requires a formal base GroundedLeWM")
    audit = _mapping(base_report.get("supervision_audit"), "base supervision audit")
    if audit.get("evidence") != "explicit":
        raise ValueError("confirmatory base requires explicit online target metadata")
    resolved_observer = _mapping(
        checkpoint_report.get("resolved_observer_config"),
        "resolved observer config",
    )
    contract = _mapping(
        resolved_observer.get("contract", {}), "resolved observer contract"
    )
    required_contract = {
        "development_only": False,
        "optimizer_step_budget": 2_000,
        "base_grounded_lewm_frozen": True,
        "require_formal_base": True,
        "require_explicit_online_base": True,
        "require_completed_base_budget": True,
        "expected_base_step": 10_000,
        "base_target": "online_self_target",
        "supervision": "q_corrected_minus_q_nominal_only",
        "q_true_forbidden": True,
        "oracle_target_forbidden": True,
        "old_gate_l1_test_reuse_forbidden": True,
    }
    mismatched_contract = {
        key: {"actual": contract.get(key), "expected": expected}
        for key, expected in required_contract.items()
        if contract.get(key) != expected
    }
    if mismatched_contract:
        raise ValueError(
            "confirmatory observer contract differs from the canonical formal "
            f"config: {mismatched_contract}"
        )
    config_binding = _mapping(
        checkpoint_report.get("config_binding"), "observer config binding"
    )
    origin_path = _resolve(config_binding.get("origin_path"))
    if origin_path != FORMAL_ADAPTER_CONFIG_PATH:
        raise ValueError("observer was not trained from the canonical formal config")
    if config_binding.get("origin_sha256_at_start") != sha256_file(
        FORMAL_ADAPTER_CONFIG_PATH
    ):
        raise ValueError("canonical formal observer config hash differs from binding")
    validate_completed_training_budget(
        base_report,
        label="confirmatory base",
        expected_target_steps=int(contract["expected_base_step"]),
        require_endpoint=True,
    )
    base_provenance = _mapping(
        base_report.get("provenance"), "base provenance"
    )
    base_manifest = _mapping(
        base_provenance.get("manifest"), "base provenance manifest"
    )
    base_config_record = _mapping(
        base_manifest.get("config"), "base provenance config"
    )
    if _resolve(base_config_record.get("absolute_path")) != FORMAL_BASE_CONFIG_PATH:
        raise ValueError("base was not trained from the canonical formal config")
    if base_config_record.get("sha256") != sha256_file(FORMAL_BASE_CONFIG_PATH):
        raise ValueError("canonical formal base config hash differs from provenance")
    validate_completed_training_budget(
        checkpoint_report,
        label="confirmatory observer",
        expected_target_steps=int(contract["optimizer_step_budget"]),
        require_endpoint=True,
    )


def _discover_preload_dependencies(
    args: argparse.Namespace,
    config_path: Path,
) -> dict[str, Any]:
    """Resolve every scored dependency without constructing a scored dataset/model."""

    config = _load_json(config_path)
    if config.get("protocol") != "lewm_pixel_observer_g0_fixed_alpha":
        raise ValueError("unexpected PixelObserver-G0 config protocol")
    settings = _resolve_settings(args, config)
    split_config = _mapping(config.get("split"), "config.split")
    split_manifest_path = _resolve(split_config.get("episode_manifest"))
    split_manifest = _load_json(split_manifest_path)
    sources = _sequence(config.get("sources"), "config.sources")

    development_binding_audit: dict[str, object] = {
        "passed": None,
        "status": "development_phase_has_no_upstream_selection_binding",
    }
    development_binding_path: Path | None = None
    if settings["phase"] == "confirmatory":
        if split_config.get("fresh_non_exposure_audited") is not True:
            raise ValueError(
                "confirmatory split must declare fresh_non_exposure_audited=true"
            )
        observer_pin = (
            args.expected_observer_sha256
            or config.get("observer_checkpoint_sha256")
        )
        base_pin = args.expected_checkpoint_sha256 or config.get(
            "base_checkpoint_sha256"
        )
        if not observer_pin or not base_pin:
            raise ValueError(
                "confirmatory evaluation requires explicit observer and base "
                "SHA-256 pins"
            )
        (
            development_binding_audit,
            development_binding_path,
        ) = validate_confirmatory_development_binding(config, settings)

    input_paths = [
        config_path,
        split_manifest_path,
        *_protocol_bundle_paths(),
        FORMAL_BASE_CONFIG_PATH,
        FORMAL_ADAPTER_CONFIG_PATH,
    ]
    design_records, design_paths = _confirmatory_design_input_records(config)
    input_paths.extend(design_paths)
    if development_binding_path is not None:
        input_paths.extend(
            [development_binding_path, FORMAL_DEVELOPMENT_CONFIG_PATH]
        )

    evaluation_source_paths: list[Path] = []
    correction_cache_paths: list[Path] = []
    source_graph = []
    for source_index, raw_source in enumerate(sources):
        source = _mapping(raw_source, f"sources[{source_index}]")
        name = str(source.get("name"))
        data_path = _resolve(source.get("path"))
        cache_path = _resolve(source.get("correction_cache"))
        evaluation_source_paths.append(data_path)
        correction_cache_paths.append(cache_path)
        correction_paths = _correction_input_paths(cache_path)
        input_paths.extend((data_path, *correction_paths))
        source_graph.append(
            {
                "name": name,
                "data": str(data_path),
                "correction_cache": str(cache_path),
                "correction_inputs": [str(path) for path in correction_paths],
            }
        )

    observer_value = args.observer_checkpoint or config.get("observer_checkpoint")
    observer_path = _resolve(observer_value)
    checkpoint_value = args.checkpoint or config.get("base_checkpoint")
    base_override = _resolve(checkpoint_value) if checkpoint_value else None
    actual_base_path, bound_paths = _observer_bound_inputs(
        observer_path,
        base_checkpoint_override=base_override,
    )
    exposure_audit: dict[str, object] = {
        "passed": None,
        "status": "development_not_a_freshness_claim",
    }
    if settings["phase"] == "confirmatory":
        if args.observer_checkpoint is not None or args.checkpoint is not None:
            raise ValueError(
                "confirmatory checkpoint paths must be frozen in the config"
            )
        exposure_audit, exposure_paths = confirmatory_source_exposure_audit(
            observer_path=observer_path,
            base_path=actual_base_path,
            evaluation_sources=evaluation_source_paths,
            correction_caches=correction_cache_paths,
        )
        bound_paths.extend(exposure_paths)
    input_paths.extend((observer_path, *bound_paths))
    unique_input_paths = tuple(sorted(set(input_paths)))
    graph = {
        "phase": settings["phase"],
        "split_manifest": str(split_manifest_path),
        "sources": source_graph,
        "observer": str(observer_path),
        "base": str(actual_base_path),
        "development_binding": development_binding_audit,
        "confirmatory_design_records": design_records,
        "input_paths": [str(path) for path in unique_input_paths],
    }
    return {
        "config": config,
        "settings": settings,
        "split_config": split_config,
        "split_manifest_path": split_manifest_path,
        "split_manifest": split_manifest,
        "sources": sources,
        "development_binding_audit": development_binding_audit,
        "observer_path": observer_path,
        "base_override": base_override,
        "exposure_audit": exposure_audit,
        "input_paths": unique_input_paths,
        "graph": graph,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve(args.config)
    discovery_before = _discover_preload_dependencies(args, config_path)
    input_paths = list(discovery_before["input_paths"])
    before_records = [input_file_record(path) for path in input_paths]
    before_digest = records_digest(before_records)
    git_before = _git_state()
    _require_clean_confirmatory_git(
        str(_mapping(discovery_before, "preload discovery")["settings"]["phase"]),
        git_before,
    )

    # Engineering double-read guard: this is not a private-data snapshot. It
    # proves only that dependency discovery is stable before scored objects load.
    discovery = _discover_preload_dependencies(args, config_path)
    if discovery["graph"] != discovery_before["graph"]:
        raise RuntimeError("PixelObserver-G0 dependency graph changed during preload")
    _assert_evaluation_state_unchanged(before_records, before_digest, git_before)

    config = _mapping(discovery["config"], "reloaded config")
    settings = _mapping(discovery["settings"], "reloaded settings")
    split_config = _mapping(discovery["split_config"], "reloaded split config")
    split_manifest_path = Path(discovery["split_manifest_path"])
    split_manifest = _mapping(discovery["split_manifest"], "reloaded manifest")
    sources = _sequence(discovery["sources"], "reloaded sources")
    development_binding_audit = _mapping(
        discovery["development_binding_audit"], "reloaded development binding"
    )
    exposure_audit = _mapping(discovery["exposure_audit"], "reloaded exposure audit")
    observer_path = Path(discovery["observer_path"])
    base_override_value = discovery["base_override"]
    base_override = (
        Path(base_override_value) if base_override_value is not None else None
    )

    datasets: list[EvaluatorOracleDataset] = []
    base_datasets: list[LeWMSequenceDataset] = []
    base_datasets_by_source: dict[str, LeWMSequenceDataset] = {}
    correction_derivation_jobs: list[dict[str, object]] = []
    source_report = []
    for source_index, raw_source in enumerate(sources):
        source = _mapping(raw_source, f"sources[{source_index}]")
        name = str(source.get("name"))
        if not name or name in base_datasets_by_source:
            raise ValueError("PixelObserver-G0 source names must be unique")
        data_path = _resolve(source.get("path"))
        cache_path = _resolve(source.get("correction_cache"))
        base = LeWMSequenceDataset(
            data_path,
            history_size=settings["history_size"],
            rollout_steps=settings["rollout_steps"],
            frame_skip=settings["frame_skip"],
            source_index=source_index,
            source_name=name,
            correction_cache=cache_path,
            require_pre_action_alignment=True,
        )
        if (
            settings["phase"] == "confirmatory"
            and tuple(base.image_size) != FORMAL_IMAGE_SIZE
        ):
            raise ValueError("confirmatory RGB frames must be exactly 224x224")
        selected = select_manifest_episodes(
            base,
            source_name=name,
            episode_manifest=split_manifest,
            partition=settings["partition"],
        )
        selected_labels = {
            _episode_label(base, base.windows[int(index)][0])
            for index in selected.indices
        }
        selected_episode_indices = {
            int(base.windows[int(index)][0]) for index in selected.indices
        }
        audited_frame_indices = np.unique(
            np.concatenate(
                [base.episodes[index] for index in sorted(selected_episode_indices)]
            )
        )
        correction_derivation_jobs.append(
            {
                "name": name,
                "data": str(data_path),
                "correction_cache": str(cache_path),
                "frame_indices": audited_frame_indices,
            }
        )
        episodes_per_task: dict[str, int] = {}
        for label in selected_labels:
            task = label.rsplit(":", 1)[0]
            episodes_per_task[task] = episodes_per_task.get(task, 0) + 1
        wrapped = EvaluatorOracleDataset(selected, data_path)
        base_datasets.append(base)
        base_datasets_by_source[name] = base
        datasets.append(wrapped)
        input_paths.append(data_path)
        input_paths.extend(_correction_input_paths(cache_path))
        corrector_record = _correction_checkpoint_record(cache_path)
        source_report.append(
            {
                "name": name,
                "data": str(data_path),
                "data_sha256": sha256_file(data_path),
                "correction_cache": str(cache_path),
                "corrector_checkpoint": corrector_record,
                "eligible_windows": len(wrapped),
                "eligible_episodes": len(selected_labels),
                "episodes_per_task": episodes_per_task,
            }
        )
    if not datasets:
        raise ValueError("PixelObserver-G0 config contains no sources")
    coverage_audit = validate_confirmatory_coverage(
        split_config,
        source_report,
        require_primary_cohorts=settings["phase"] == "confirmatory",
    )
    device = torch.device(args.device)
    canonical_scoring_applications: dict[str, dict[str, object]] = {}
    correction_derivation_audit, derivation_paths = (
        validate_fcpg_correction_derivations(
            correction_derivation_jobs,
            canonical_sink=_canonical_correction_sink(
                base_datasets_by_source,
                canonical_scoring_applications,
            ),
            device=device,
            batch_size=settings["batch_size"],
            require_schema3=settings["phase"] == "confirmatory",
        )
    )
    correction_derivation_audit = dict(correction_derivation_audit)
    derivation_sources = _mapping(
        correction_derivation_audit.get("sources"),
        "correction derivation sources",
    )
    expected_canonical_sources = set(base_datasets_by_source)
    canonical_scoring_passed = (
        correction_derivation_audit.get("canonical_scoring_outputs_emitted") is True
        and set(canonical_scoring_applications) == expected_canonical_sources
        and all(
            _mapping(
                derivation_sources.get(name),
                f"correction derivation source {name}",
            ).get("canonical_scoring_output_emitted")
            is True
            for name in expected_canonical_sources
        )
    )
    correction_derivation_audit["canonical_scoring"] = {
        "passed": canonical_scoring_passed,
        "required_sources": sorted(expected_canonical_sources),
        "applications": canonical_scoring_applications,
    }
    correction_derivation_audit["passed"] = (
        correction_derivation_audit.get("passed") is True
        and canonical_scoring_passed
    )
    undiscovered_derivation_paths = set(derivation_paths).difference(input_paths)
    if undiscovered_derivation_paths:
        raise RuntimeError(
            "correction replay discovered post-snapshot inputs: "
            f"{sorted(map(str, undiscovered_derivation_paths))}"
        )
    if settings["phase"] == "confirmatory":
        replay_counts = {
            name: int(
                _mapping(
                    derivation_sources.get(name),
                    f"correction derivation source {name}",
                ).get("scored_frames", -1)
            )
            for name in PRIMARY_CONFIRMATORY_SOURCES
        }
        if replay_counts != {name: 12_000 for name in PRIMARY_CONFIRMATORY_SOURCES}:
            raise ValueError(
                "confirmatory correction replay must cover all 12000 frames "
                f"per cohort: {replay_counts}"
            )
    physical_independence_audit: dict[str, object] = {
        "passed": None,
        "status": "development_does_not_claim_fresh_physical_independence",
    }
    development_independence_audit: dict[str, object] = {
        "passed": None,
        "status": "development_phase_has_no_confirmatory_sources",
    }
    camera_intervention_audit: dict[str, object] = {
        "passed": None,
        "status": "development_phase_has_no_primary_camera_intervention",
    }
    if settings["phase"] == "confirmatory":
        observed_corrector_hashes = {
            str(
                _mapping(
                    source.get("corrector_checkpoint"),
                    "confirmatory corrector checkpoint",
                ).get("sha256")
            )
            for source in source_report
        }
        expected_corrector = str(
            development_binding_audit.get("corrector_checkpoint_sha256")
        )
        if observed_corrector_hashes != {expected_corrector}:
            raise ValueError(
                "confirmatory FCPG corrector differs from development selection"
            )
        physical_independence_audit = validate_physical_episode_independence(
            source_report
        )
        development_independence_audit = (
            validate_development_confirmatory_independence(
                [
                    _mapping(value, "bound development source")
                    for value in _sequence(
                        development_binding_audit.get("development_sources"),
                        "bound development sources",
                    )
                ],
                source_report,
            )
        )
        camera_intervention_audit, intervention_paths = (
            validate_confirmatory_camera_interventions(
                FORMAL_CONFIRMATORY_DESIGN_PATH,
                source_report,
                repo_root=ROOT,
            )
        )
        undiscovered_intervention_paths = set(intervention_paths).difference(input_paths)
        if undiscovered_intervention_paths:
            raise RuntimeError(
                "camera audit discovered post-snapshot inputs: "
                f"{sorted(map(str, undiscovered_intervention_paths))}"
            )

    _assert_evaluation_state_unchanged(before_records, before_digest, git_before)

    observer, checkpoint_report = _load_observer(
        observer_path=observer_path,
        base_checkpoint_override=base_override,
        expected_observer_sha256=(
            args.expected_observer_sha256 or config.get("observer_checkpoint_sha256")
        ),
        expected_base_sha256=(
            args.expected_checkpoint_sha256 or config.get("base_checkpoint_sha256")
        ),
        device=device,
        sample_dataset=base_datasets[0],
    )
    if settings["phase"] == "confirmatory":
        _validate_confirmatory_checkpoint_report(checkpoint_report)
    _assert_evaluation_state_unchanged(before_records, before_digest, git_before)
    combined: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(
        combined,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    evaluation = evaluate_filter_batches(
        observer,
        loader,
        alphas=settings["alphas"],
        horizons=settings["horizons"],
        phase=settings["phase"],
        device=device,
        max_batches=args.max_batches,
        bootstrap_samples=settings["bootstrap_samples"],
        bootstrap_seed=settings["bootstrap_seed"],
        primary_horizon=settings["primary_horizon"],
        max_task_relative_degradation=settings["max_task_relative_degradation"],
        max_raw_fcpg_relative_degradation=settings[
            "max_raw_fcpg_relative_degradation"
        ],
        max_source_task_relative_degradation=settings[
            "max_source_task_relative_degradation"
        ],
        min_eligible_anchor_fraction=settings["min_eligible_anchor_fraction"],
        min_contributing_window_fraction=settings[
            "min_contributing_window_fraction"
        ],
        min_measurement_coverage=settings["min_measurement_coverage"],
    )
    if settings["phase"] == "development":
        evaluation["development_selection"] = qualify_development_selection(
            _mapping(
                evaluation.get("development_selection"),
                "evaluation.development_selection",
            ),
            checkpoint_report=checkpoint_report,
            coverage_audit=coverage_audit,
            correction_derivation_audit=correction_derivation_audit,
            max_batches=args.max_batches,
            git_state=git_before,
        )

    _assert_evaluation_state_unchanged(before_records, before_digest, git_before)

    split_disclosure = str(split_config.get("disclosure", ""))
    confirmatory_eligibility_conditions = {
        "confirmatory_phase": settings["phase"] == "confirmatory",
        "test_partition": settings["partition"] == "test",
        "fresh_non_exposure_declared": (
            split_config.get("fresh_non_exposure_audited") is True
        ),
        "source_exposure_audit": exposure_audit.get("passed") is True,
        "coverage_audit": coverage_audit.get("passed") is True,
        "correction_derivation_audit": (
            correction_derivation_audit.get("passed") is True
        ),
        "development_binding_audit": (
            development_binding_audit.get("passed") is True
        ),
        "physical_independence_audit": (
            physical_independence_audit.get("passed") is True
        ),
        "development_independence_audit": (
            development_independence_audit.get("passed") is True
        ),
        "camera_intervention_audit": (
            camera_intervention_audit.get("passed") is True
        ),
    }
    confirmatory_eligible = all(confirmatory_eligibility_conditions.values())
    if settings["phase"] == "confirmatory":
        scientific_gate = _mapping(evaluation.get("gate"), "evaluation.gate")
        gate = dict(scientific_gate)
        conditions = dict(
            _mapping(scientific_gate.get("conditions"), "evaluation.gate.conditions")
        )
        conditions["confirmatory_provenance_conjunction"] = {
            "passed": confirmatory_eligible,
            "operator": "AND",
            "conditions": confirmatory_eligibility_conditions,
        }
        scientific_passed = scientific_gate.get("passed") is True
        gate["conditions"] = conditions
        gate["scientific_effect_passed"] = scientific_passed
        gate["passed"] = scientific_passed and confirmatory_eligible
        gate["decision"] = "Go" if gate["passed"] else "No-Go"
        evaluation["gate"] = gate
    report = {
        "schema_version": 1,
        "protocol": "lewm_pixel_observer_g0_report_v1",
        "scope": "independent_from_legacy_grounded_lewm_gate_l1",
        "phase": settings["phase"],
        "provenance": {
            "input_records": before_records,
            "input_digest": before_digest,
            "inputs_unchanged": True,
            "git": git_before,
            "config": input_file_record(config_path),
            "episode_manifest": input_file_record(split_manifest_path),
        },
        "checkpoint": checkpoint_report,
        "split": {
            "partition": settings["partition"],
            "fresh_non_exposure_audited": bool(
                split_config.get("fresh_non_exposure_audited", False)
            ),
            "disclosure": split_disclosure,
            "confirmatory_eligible": confirmatory_eligible,
            "confirmatory_eligibility_conditions": (
                confirmatory_eligibility_conditions
            ),
            "source_exposure_audit": exposure_audit,
            "coverage_audit": coverage_audit,
            "correction_derivation_audit": correction_derivation_audit,
            "development_binding_audit": development_binding_audit,
            "physical_independence_audit": physical_independence_audit,
            "development_independence_audit": development_independence_audit,
            "camera_intervention_audit": camera_intervention_audit,
        },
        "sources": source_report,
        "runtime": {
            "device": str(device),
            "batch_size": settings["batch_size"],
            "num_workers": settings["num_workers"],
            "max_batches": args.max_batches,
        },
        "evaluation": evaluation,
        "gate": evaluation["gate"],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_value = args.output or _mapping(config.get("output"), "config.output").get(
        "report"
    )
    output_path = reject_legacy_output_path(_resolve(output_value), ROOT)
    _exclusive_atomic_write(output_path, rendered)
    print(rendered, end="")
    gate = evaluation["gate"]
    return (
        1
        if settings["phase"] == "confirmatory"
        and (not confirmatory_eligible or gate.get("passed") is not True)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
