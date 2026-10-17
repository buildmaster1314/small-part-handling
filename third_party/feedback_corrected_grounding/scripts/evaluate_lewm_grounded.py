#!/usr/bin/env python3
"""Offline diagnostics for a trained grounded LeWM checkpoint.

The evaluator intentionally separates four questions that a low training loss
does not answer on its own:

* does the model beat copying the current latent;
* does pairing each sequence with another sequence's actions hurt prediction;
* do predictions change when the action changes at all; and
* can the learned dynamics be rolled out recursively beyond one observed step.

Legacy global MSE fields remain element weighted. Gate metrics first aggregate
windows within an episode, then average episodes within each task and tasks in
the macro result with equal weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg import lewm_provenance as _lewm_provenance  # noqa: E402
from fcpg.lewm_provenance import validate_manifest  # noqa: E402
from fcpg.lewm_sequence_data import LeWMSequenceDataset  # noqa: E402
from fcpg.lewm_anchor_ablation import (  # noqa: E402
    EXPECTED_OPTIMIZER_STEP as ANCHOR_ABLATION_STEP,
    PARAMETER_COMPATIBLE_TRIPLET_MODES,
    validate_anchor_visual_override,
)
from fcpg.lewm_token_metrics import (  # noqa: E402
    merge_oracle_token_errors,
    oracle_token_squared_error,
    summarize_oracle_token_mse,
)


CHECKPOINT_TYPE = "fcpg_grounded_lewm_v1"
ENDPOINT_HORIZONS = (1, 3, 5)
REQUIRED_ROLLOUT_STEPS = 5
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_GATE_THRESHOLDS = {
    "R1_max": 0.90,
    "R3_max": 0.95,
    "R5_max": 0.95,
    "S1_min": 1.10,
    "ratio_boundary": 1.0,
}
ANCHOR_ABLATION_PAIR_ID = "anchor_visual_ablation_seed5120"
ANCHOR_ABLATION_SEED = 5120
ANCHOR_ABLATION_PARAMETER_COUNT = 19_152_816
ANCHOR_ABLATION_TARGET_SHA256 = (
    "7c8ce2120a592403c5908cb68733b7e8ee5ded79b9292d68ff78907c6ab91c31"
)
ANCHOR_ABLATION_SPLIT_SHA256 = (
    "f3448ef472bd93975126c203556094ad65229a714d58236a53b11b281768e299"
)
MODEL_EXCLUDED_BATCH_KEYS = {"q_true", "valid_true"}


@dataclass
class SquaredErrorAccumulator:
    """Element-weighted MSE accumulator used by all reported diagnostics."""

    squared_error: float = 0.0
    element_count: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {tuple(prediction.shape)} "
                f"vs {tuple(target.shape)}"
            )
        error = prediction.detach().to(torch.float64) - target.detach().to(
            torch.float64
        )
        self.squared_error += float(error.square().sum().cpu())
        self.element_count += int(error.numel())

    @property
    def mse(self) -> float | None:
        if self.element_count == 0:
            return None
        return self.squared_error / self.element_count


@dataclass
class RepresentationAccumulator:
    """Accumulate a covariance spectrum without retaining all embeddings."""

    count: int = 0
    vector_sum: torch.Tensor | None = None
    gram_sum: torch.Tensor | None = None

    def update(self, embedding: torch.Tensor) -> None:
        values = embedding.detach().reshape(-1, embedding.shape[-1]).double().cpu()
        if self.vector_sum is None:
            dim = int(values.shape[1])
            self.vector_sum = torch.zeros(dim, dtype=torch.float64)
            self.gram_sum = torch.zeros(dim, dim, dtype=torch.float64)
        if values.shape[1] != self.vector_sum.shape[0]:
            raise ValueError("representation dimension changed during evaluation")
        self.vector_sum += values.sum(dim=0)
        assert self.gram_sum is not None
        self.gram_sum += values.T @ values
        self.count += int(values.shape[0])

    def metrics(self) -> dict[str, object]:
        if self.count < 2 or self.vector_sum is None or self.gram_sum is None:
            return {
                "samples": self.count,
                "effective_rank": None,
                "participation_ratio": None,
                "numerical_rank": None,
            }
        mean = self.vector_sum / self.count
        covariance = self.gram_sum / self.count - torch.outer(mean, mean)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        total = float(eigenvalues.sum())
        if not math.isfinite(total) or total <= 0.0:
            effective_rank = 0.0
            participation_ratio = 0.0
            numerical_rank = 0
        else:
            probability = eigenvalues / total
            positive = probability > 0.0
            entropy = -(probability[positive] * probability[positive].log()).sum()
            effective_rank = float(entropy.exp())
            participation_ratio = float(
                eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-30)
            )
            threshold = float(eigenvalues.max()) * 1e-6
            numerical_rank = int((eigenvalues > threshold).sum())
        return {
            "samples": self.count,
            "effective_rank": _finite_or_none(effective_rank),
            "participation_ratio": _finite_or_none(participation_ratio),
            "numerical_rank": numerical_rank,
            "trace": _finite_or_none(total),
        }


@dataclass
class EpisodeAccumulator:
    """Window errors aggregated into one source/task/episode block."""

    source_index: int
    task_name: str
    episode_id: int
    sequence_count: int = 0
    shuffle_sequence_count: int = 0
    teacher_predicted_sse: float = 0.0
    teacher_identity_sse: float = 0.0
    teacher_shuffled_sse: float = 0.0
    teacher_action_sensitivity_sse: float = 0.0
    teacher_element_count: int = 0
    alignment_sse: float = 0.0
    alignment_element_count: int = 0
    predicted_sse: dict[int, float] = field(
        default_factory=lambda: {horizon: 0.0 for horizon in ENDPOINT_HORIZONS}
    )
    identity_sse: dict[int, float] = field(
        default_factory=lambda: {horizon: 0.0 for horizon in ENDPOINT_HORIZONS}
    )
    endpoint_element_count: dict[int, int] = field(
        default_factory=lambda: {horizon: 0 for horizon in ENDPOINT_HORIZONS}
    )
    shuffled_h1_sse: float = 0.0
    shuffled_h1_element_count: int = 0

    def teacher_mse(self, metric: str) -> float:
        if self.teacher_element_count <= 0:
            return math.nan
        values = {
            "predicted": self.teacher_predicted_sse,
            "identity": self.teacher_identity_sse,
            "shuffled": self.teacher_shuffled_sse,
            "action_sensitivity": self.teacher_action_sensitivity_sse,
        }
        if metric not in values:
            raise ValueError(f"unknown teacher-forced episode metric: {metric}")
        return values[metric] / self.teacher_element_count

    @property
    def alignment_mse(self) -> float:
        if self.alignment_element_count <= 0:
            return math.nan
        return self.alignment_sse / self.alignment_element_count

    def endpoint_mse(self, horizon: int, metric: str) -> float:
        count = self.endpoint_element_count[horizon]
        if count <= 0:
            return math.nan
        values = {
            "predicted": self.predicted_sse,
            "identity": self.identity_sse,
        }
        return values[metric][horizon] / count

    @property
    def shuffled_h1_mse(self) -> float:
        if self.shuffled_h1_element_count <= 0:
            return math.nan
        return self.shuffled_h1_sse / self.shuffled_h1_element_count


@dataclass(frozen=True)
class EvaluationWindow:
    dataset_index: int
    local_index: int
    window_index: int
    source_index: int
    task_name: str
    episode_id: int


def cyclic_action_shuffle(actions: torch.Tensor) -> torch.Tensor:
    """Return a deterministic cross-sequence derangement of action blocks.

    Shuffling only along the batch dimension preserves the temporal structure
    and marginal scale of each action sequence. A singleton batch has no valid
    cross-sequence permutation and is therefore rejected instead of fabricating
    an out-of-distribution action intervention.
    """

    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, T, A], got {tuple(actions.shape)}")
    if actions.shape[0] < 2:
        raise ValueError("action shuffle requires at least two sequences")
    return torch.roll(actions, shifts=1, dims=0)


def episode_aware_action_shuffle(
    actions: torch.Tensor,
    batch: Mapping[str, object],
) -> torch.Tensor:
    """Pair every row with same-source/task actions from another episode."""

    batch_size = int(actions.shape[0])
    sources = _integer_metadata(batch.get("source_index"), "source_index", batch_size)
    tasks = _string_metadata(batch.get("task_name"), "task_name", batch_size)
    episodes = _integer_metadata(batch.get("episode_id"), "episode_id", batch_size)
    permutation = same_task_action_donor_indices(sources, tasks, episodes)
    indices = torch.tensor(permutation, dtype=torch.long, device=actions.device)
    return actions.index_select(0, indices)


def _integer_metadata(value: object, name: str, size: int) -> list[int]:
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    elif value is not None and size == 1:
        values = [value]
    else:
        raise KeyError(f"action shuffle requires {name} metadata")
    if len(values) != size:
        raise ValueError(f"{name} metadata must contain {size} values")
    return [int(item) for item in values]


def _string_metadata(value: object, name: str, size: int) -> list[str]:
    if isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    elif isinstance(value, str) and size == 1:
        values = [value]
    else:
        raise KeyError(f"action shuffle requires {name} metadata")
    if len(values) != size:
        raise ValueError(f"{name} metadata must contain {size} values")
    return [str(item) for item in values]


def same_task_action_donor_indices(
    source_indices: Sequence[int],
    task_names: Sequence[str],
    episode_ids: Sequence[int],
) -> list[int]:
    """Return a total deterministic donor map with strict shuffle constraints."""

    count = len(source_indices)
    if count == 0:
        raise ValueError("action shuffle requires at least one evaluation window")
    if len(task_names) != count or len(episode_ids) != count:
        raise ValueError("action shuffle metadata lengths differ")
    grouped: dict[tuple[int, str], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (source, task, episode) in enumerate(
        zip(source_indices, task_names, episode_ids, strict=True)
    ):
        grouped[(int(source), str(task))][int(episode)].append(index)

    donors = [-1] * count
    for (source, task), episode_windows in sorted(grouped.items()):
        episodes = sorted(episode_windows)
        if len(episodes) < 2:
            raise ValueError(
                "100% action-shuffle coverage requires at least two episodes for "
                f"source={source}, task={task!r}"
            )
        for episode_index, episode in enumerate(episodes):
            donor_episode = episodes[(episode_index + 1) % len(episodes)]
            target_windows = episode_windows[episode]
            donor_windows = episode_windows[donor_episode]
            for offset, target in enumerate(target_windows):
                donors[target] = donor_windows[offset % len(donor_windows)]

    if any(index < 0 for index in donors):
        raise RuntimeError("failed to assign an action donor to every window")
    for target, donor in enumerate(donors):
        if int(source_indices[target]) != int(source_indices[donor]):
            raise RuntimeError("action donor changed source")
        if str(task_names[target]) != str(task_names[donor]):
            raise RuntimeError("action donor changed task")
        if int(episode_ids[target]) == int(episode_ids[donor]):
            raise RuntimeError("action donor reused the target episode")
    return donors


def _resolve_base_window(
    dataset: Dataset,
    local_index: int,
) -> tuple[LeWMSequenceDataset, int]:
    current = dataset
    resolved_index = int(local_index)
    while isinstance(current, Subset):
        resolved_index = int(current.indices[resolved_index])
        current = current.dataset
    if not isinstance(current, LeWMSequenceDataset):
        raise TypeError(
            "paired action evaluation requires LeWMSequenceDataset or Subset wrappers"
        )
    return current, resolved_index


def _evaluation_window(
    dataset: Dataset,
    dataset_index: int,
    local_index: int,
) -> EvaluationWindow:
    base, window_index = _resolve_base_window(dataset, local_index)
    episode_index, _ = base.windows[window_index]
    raw_index = int(base.episodes[episode_index][0])
    return EvaluationWindow(
        dataset_index=dataset_index,
        local_index=local_index,
        window_index=window_index,
        source_index=int(base.source_index),
        task_name=str(base.task_name[raw_index]),
        episode_id=int(base.episode_id[raw_index]),
    )


def _window_action(dataset: Dataset, local_index: int) -> torch.Tensor:
    base, window_index = _resolve_base_window(dataset, local_index)
    episode_index, start = base.windows[window_index]
    sequence = base.episodes[episode_index]
    state_positions = start + np.arange(base.num_state_frames) * base.frame_skip
    blocks = []
    for position in state_positions[:-1]:
        transition_indices = sequence[position : position + base.frame_skip]
        blocks.append(base.action[transition_indices].reshape(-1))
    return torch.from_numpy(np.stack(blocks).astype(np.float32, copy=False))


class SameTaskActionShuffleDataset(Dataset):
    """Attach a valid cross-episode donor action to every evaluation window."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        if not datasets:
            raise ValueError("at least one evaluation dataset is required")
        self.datasets = list(datasets)
        self.windows = [
            _evaluation_window(dataset, dataset_index, local_index)
            for dataset_index, dataset in enumerate(self.datasets)
            for local_index in range(len(dataset))
        ]
        self.donor_indices = same_task_action_donor_indices(
            [window.source_index for window in self.windows],
            [window.task_name for window in self.windows],
            [window.episode_id for window in self.windows],
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        target = self.windows[index]
        donor = self.windows[self.donor_indices[index]]
        sample = dict(self.datasets[target.dataset_index][target.local_index])
        for key in MODEL_EXCLUDED_BATCH_KEYS:
            sample.pop(key, None)
        sample["shuffled_action"] = _window_action(
            self.datasets[donor.dataset_index], donor.local_index
        )
        sample["shuffle_donor_source_index"] = torch.tensor(
            donor.source_index, dtype=torch.long
        )
        sample["shuffle_donor_task_name"] = donor.task_name
        sample["shuffle_donor_episode_id"] = donor.episode_id
        return sample


def _nested_mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _normalize_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return normalized


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


FileSnapshots = dict[str, dict[str, object]]


def _snapshot_file(
    path: str | Path,
    file_snapshots: FileSnapshots,
) -> dict[str, object]:
    requested = Path(path).expanduser()
    resolved = requested.resolve()
    key = str(resolved)
    cached = file_snapshots.get(key)
    if cached is not None:
        return cached

    before = requested.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"evaluation input is not a regular file: {requested}")
    digest = _file_sha256(requested)
    after = requested.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"evaluation input changed while hashing: {resolved}")
    snapshot: dict[str, object] = {
        "input": key,
        "sha256": digest,
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "identity": after_identity,
    }
    file_snapshots[key] = snapshot
    return snapshot


def _file_manifest(
    path: str | Path,
    file_snapshots: FileSnapshots,
) -> dict[str, str]:
    snapshot = _snapshot_file(path, file_snapshots)
    return {
        "input": str(snapshot["input"]),
        "sha256": str(snapshot["sha256"]),
    }


def dataset_file_manifests(
    paths: Sequence[str],
    *,
    file_snapshots: FileSnapshots | None = None,
) -> list[dict[str, str]]:
    snapshots = {} if file_snapshots is None else file_snapshots
    return [
        _file_manifest(path, snapshots)
        for path in paths
    ]


def _require_unchanged(label: str, before: object, after: object) -> None:
    if after != before:
        raise RuntimeError(f"{label} changed while diagnostics were running")


def execution_code_manifest(
    *,
    file_snapshots: FileSnapshots | None = None,
) -> dict[str, object]:
    """Hash every local Python source that can affect grounded-LeWM evaluation."""

    snapshots = {} if file_snapshots is None else file_snapshots
    files = {Path(__file__).resolve()}
    for directory in (ROOT / "fcpg", ROOT / "third_party/lewm"):
        files.update(path.resolve() for path in directory.rglob("*.py"))
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _snapshot_file(path, snapshots)["sha256"],
        }
        for path in sorted(files)
    ]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "algorithm": "sha256(path-and-content-manifest-v1)",
        "aggregate_sha256": hashlib.sha256(payload).hexdigest(),
        "files": entries,
    }


def _correction_array_path(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    array_path = requested / "q_corrected.npy" if requested.is_dir() else requested
    return array_path.resolve()


def correction_cache_manifest(
    path: str | Path,
    *,
    file_snapshots: FileSnapshots | None = None,
) -> dict[str, object]:
    """Return immutable provenance for a correction cache input."""

    snapshots = {} if file_snapshots is None else file_snapshots
    requested = Path(path).expanduser().resolve()
    if requested.is_dir():
        array_path = requested / "q_corrected.npy"
        metadata_path = requested / "metadata.json"
    else:
        array_path = requested
        metadata_path = None
    if not array_path.is_file():
        raise FileNotFoundError(f"missing corrected keypoints: {array_path}")
    manifest: dict[str, object] = {
        "input": str(requested),
        "array": str(array_path.resolve()),
        "array_sha256": _snapshot_file(array_path, snapshots)["sha256"],
    }
    if metadata_path is not None:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing correction metadata: {metadata_path}")
        manifest["metadata"] = str(metadata_path.resolve())
        manifest["metadata_sha256"] = _snapshot_file(
            metadata_path, snapshots
        )["sha256"]
    return manifest


def oracle_target_cache_manifest(
    path: str | Path,
    *,
    file_snapshots: FileSnapshots | None = None,
) -> dict[str, object]:
    """Return immutable provenance for a frozen oracle target cache."""

    snapshots = {} if file_snapshots is None else file_snapshots
    requested = Path(path).expanduser().resolve()
    if not requested.is_dir():
        raise ValueError("oracle target cache must be a directory")
    array_path = requested / "oracle_target.npy"
    metadata_path = requested / "metadata.json"
    if not array_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "oracle target cache requires oracle_target.npy and metadata.json"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "input": str(requested),
        "array": str(array_path.resolve()),
        "array_sha256": _snapshot_file(array_path, snapshots)["sha256"],
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": _snapshot_file(metadata_path, snapshots)["sha256"],
        "encoder_checkpoint_sha256": metadata.get("encoder_checkpoint_sha256"),
        "encoder_spec": metadata.get("encoder_spec"),
        "q_true_usage": metadata.get("q_true_usage"),
    }


def _manifest_snapshot_paths(manifest: Mapping[str, object]) -> list[Path]:
    paths: dict[str, Path] = {}

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            absolute_path = value.get("absolute_path")
            if (
                isinstance(absolute_path, str)
                and isinstance(value.get("sha256"), str)
                and isinstance(value.get("size"), int)
            ):
                path = Path(absolute_path).expanduser()
                paths[str(path.resolve())] = path
            for nested in value.values():
                visit(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                visit(nested)

    visit(manifest)
    return [paths[key] for key in sorted(paths)]


def _checkpoint_provenance_snapshot(
    checkpoint: Mapping[str, object],
    checkpoint_path: str | Path,
    file_snapshots: FileSnapshots,
    *,
    check_git: bool,
) -> dict[str, object] | None:
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        return None

    external_path = provenance.get("path")
    external_manifest = (
        _file_manifest(external_path, file_snapshots)
        if isinstance(external_path, str)
        else None
    )
    sidecar_path = (
        Path(checkpoint_path).expanduser().resolve().parent / "input_manifest.json"
    )
    run_sidecar = _file_manifest(sidecar_path, file_snapshots)
    manifest = provenance.get("manifest")
    frozen_inputs = (
        [
            _file_manifest(path, file_snapshots)
            for path in _manifest_snapshot_paths(manifest)
        ]
        if isinstance(manifest, Mapping)
        else []
    )

    git_state: Mapping[str, object] | None = None
    git_error: str | None = None
    if check_git and isinstance(manifest, Mapping):
        repository = manifest.get("repository")
        repository_path = (
            repository.get("absolute_path")
            if isinstance(repository, Mapping)
            else None
        )
        if isinstance(repository_path, str) and Path(repository_path).is_absolute():
            try:
                git_state = _lewm_provenance.capture_git_state(repository_path)
            except (OSError, ValueError) as error:
                git_error = str(error)

    return {
        "external_manifest": external_manifest,
        "run_sidecar": run_sidecar,
        "frozen_inputs": frozen_inputs,
        "git_state": git_state,
        "git_error": git_error,
    }


def _evaluation_input_snapshot(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, object],
    data_paths: Sequence[str],
    correction_caches: Sequence[str],
    oracle_target_caches: Sequence[str],
    *,
    check_git: bool,
    file_snapshots: FileSnapshots | None = None,
) -> dict[str, object]:
    snapshots = {} if file_snapshots is None else file_snapshots
    checkpoint_manifest = _file_manifest(checkpoint_path, snapshots)
    datasets = dataset_file_manifests(
        data_paths,
        file_snapshots=snapshots,
    )
    correction_manifests = [
        correction_cache_manifest(path, file_snapshots=snapshots)
        for path in correction_caches
    ]
    oracle_manifests = [
        oracle_target_cache_manifest(path, file_snapshots=snapshots)
        for path in oracle_target_caches
    ]
    provenance = _checkpoint_provenance_snapshot(
        checkpoint,
        checkpoint_path,
        snapshots,
        check_git=check_git,
    )
    execution_code = execution_code_manifest(file_snapshots=snapshots)
    return {
        "files": dict(snapshots),
        "checkpoint": checkpoint_manifest,
        "datasets": datasets,
        "manifest": provenance,
        "caches": {
            "correction": correction_manifests,
            "oracle_target": oracle_manifests,
        },
        "execution_code": execution_code,
    }


def _require_input_snapshot_unchanged(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> None:
    sections = (
        ("checkpoint", "checkpoint"),
        ("datasets", "evaluation dataset"),
        ("manifest", "checkpoint provenance manifest"),
        ("execution_code", "evaluation code"),
    )
    for key, label in sections:
        _require_unchanged(label, before.get(key), after.get(key))
    before_caches = before.get("caches")
    after_caches = after.get("caches")
    if not isinstance(before_caches, Mapping) or not isinstance(
        after_caches, Mapping
    ):
        raise RuntimeError("evaluation cache snapshot is invalid")
    _require_unchanged(
        "correction cache",
        before_caches.get("correction"),
        after_caches.get("correction"),
    )
    _require_unchanged(
        "oracle target cache",
        before_caches.get("oracle_target"),
        after_caches.get("oracle_target"),
    )


def resolve_oracle_token_spec(
    manifests: Sequence[Mapping[str, object]],
    *,
    required: bool,
) -> Mapping[str, object] | None:
    """Return one common token layout while preserving legacy cache support."""

    specs = [manifest.get("encoder_spec") for manifest in manifests]
    complete = bool(specs) and all(isinstance(spec, Mapping) for spec in specs)
    matched = complete and all(spec == specs[0] for spec in specs[1:])
    if matched:
        return specs[0]
    if required:
        raise ValueError(
            "anchor-visual ablation requires matching oracle encoder_spec metadata"
        )
    return None


def resolve_sequence_settings(
    checkpoint: Mapping[str, object],
    rollout_steps: int,
    history_size: int | None = None,
    frame_skip: int | None = None,
) -> tuple[int, int, int]:
    """Resolve strict sequence settings from checkpoint metadata and CLI overrides."""

    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    resolved = _nested_mapping(checkpoint, "resolved_config")
    data_config = _nested_mapping(checkpoint, "data_config")
    if not data_config:
        data_config = _nested_mapping(resolved, "data")
    model_config = _nested_mapping(checkpoint, "model_config")
    if not model_config:
        model_config = _nested_mapping(resolved, "model")

    checkpoint_history = model_config.get(
        "history_size", data_config.get("history_size", resolved.get("history_size", 3))
    )
    checkpoint_skip = data_config.get("frame_skip", resolved.get("frame_skip", 1))
    selected_history = int(checkpoint_history if history_size is None else history_size)
    selected_skip = int(checkpoint_skip if frame_skip is None else frame_skip)
    if selected_history <= 0:
        raise ValueError("history_size must be positive")
    if selected_skip <= 0:
        raise ValueError("frame_skip must be positive")
    return selected_history, int(rollout_steps), selected_skip


def _move_batch(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key not in MODEL_EXCLUDED_BATCH_KEYS
    }


def _teacher_forced_tensors(
    model: torch.nn.Module,
    batch: Mapping[str, object],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    model_batch = {
        key: value
        for key, value in batch.items()
        if key not in MODEL_EXCLUDED_BATCH_KEYS
    }
    _, embedding = model.encode_frames(model_batch)
    actions = batch.get("action")
    if not torch.is_tensor(actions):
        actions = batch.get("actions")
    if not torch.is_tensor(actions):
        raise KeyError("batch must contain tensor action/actions")
    history = int(model.history_size)
    if embedding.shape[1] < history + 1:
        raise ValueError(
            f"diagnostics require at least H+1={history + 1} frames, "
            f"got {embedding.shape[1]}"
        )
    if actions.shape[1] < history:
        raise ValueError(
            f"diagnostics require at least H={history} action blocks, "
            f"got {actions.shape[1]}"
        )
    context = embedding[:, :history]
    target_sequence = getattr(model, "target_sequence", None)
    state_target = (
        target_sequence(model_batch, embedding)
        if callable(target_sequence)
        else embedding
    )
    target = state_target[:, 1 : history + 1]
    teacher_actions = actions[:, :history].float()
    prediction = model.predict(context, teacher_actions)
    return embedding, state_target, teacher_actions, prediction, target


def _all_actions(batch: Mapping[str, object]) -> torch.Tensor:
    actions = batch.get("action")
    if not torch.is_tensor(actions):
        actions = batch.get("actions")
    if not torch.is_tensor(actions):
        raise KeyError("batch must contain tensor action/actions")
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, T, A], got {tuple(actions.shape)}")
    return actions.float()


def _validated_shuffled_actions(
    actions: torch.Tensor,
    batch: Mapping[str, object],
) -> torch.Tensor:
    shuffled = batch.get("shuffled_action")
    if not torch.is_tensor(shuffled):
        shuffled = batch.get("shuffled_actions")
    if torch.is_tensor(shuffled):
        shuffled = shuffled.float()
        if shuffled.shape != actions.shape:
            raise ValueError(
                "paired shuffled actions must match target action shape: "
                f"{tuple(shuffled.shape)} vs {tuple(actions.shape)}"
            )
        batch_size = int(actions.shape[0])
        sources = _integer_metadata(batch.get("source_index"), "source_index", batch_size)
        tasks = _string_metadata(batch.get("task_name"), "task_name", batch_size)
        episodes = _integer_metadata(batch.get("episode_id"), "episode_id", batch_size)
        donor_sources = _integer_metadata(
            batch.get("shuffle_donor_source_index"),
            "shuffle_donor_source_index",
            batch_size,
        )
        donor_tasks = _string_metadata(
            batch.get("shuffle_donor_task_name"),
            "shuffle_donor_task_name",
            batch_size,
        )
        donor_episodes = _integer_metadata(
            batch.get("shuffle_donor_episode_id"),
            "shuffle_donor_episode_id",
            batch_size,
        )
        for index in range(batch_size):
            if sources[index] != donor_sources[index]:
                raise ValueError("paired action shuffle changed source")
            if tasks[index] != donor_tasks[index]:
                raise ValueError("paired action shuffle changed task")
            if episodes[index] == donor_episodes[index]:
                raise ValueError("paired action shuffle reused the target episode")
        return shuffled
    return episode_aware_action_shuffle(actions, batch)


def _per_sample_sse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(prediction.shape)} "
            f"vs {tuple(target.shape)}"
        )
    error = prediction.detach().to(torch.float64) - target.detach().to(
        torch.float64
    )
    return error.square().reshape(error.shape[0], -1).sum(dim=1).cpu().numpy()


def _pooled_mse(values: Iterable[tuple[float, int]]) -> float | None:
    """Pool episode MSE/count pairs using the report's reconstruction order."""

    squared_error = 0.0
    element_count = 0
    for mse, count in values:
        if count <= 0 or not math.isfinite(mse):
            return None
        squared_error += mse * count
        element_count += count
    if element_count == 0:
        return None
    return squared_error / element_count


def _finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return int(value) if isinstance(value, int) else numeric


def _strict_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(float(value)) for value in values):
        return math.nan
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None:
        return math.nan
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator <= 0.0:
        return math.nan
    return numerator / denominator


def episode_block_bootstrap_ci(
    per_task_pairs: Mapping[str, Sequence[tuple[float, float]]],
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[float | None]:
    """Bootstrap an equal-task macro ratio by resampling whole episodes."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not per_task_pairs:
        raise ValueError("bootstrap requires at least one task")
    arrays: dict[str, np.ndarray] = {}
    for task, pairs in sorted(per_task_pairs.items()):
        values = np.asarray(list(pairs), dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
            raise ValueError(f"task {task!r} has no valid episode pairs")
        if not bool(np.all(np.isfinite(values))) or bool(np.any(values[:, 1] <= 0.0)):
            return [None, None]
        arrays[task] = values

    rng = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=np.float64)
    chunk_size = min(2_000, samples)
    task_count = len(arrays)
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        macro = np.zeros(count, dtype=np.float64)
        for values in arrays.values():
            episode_count = values.shape[0]
            indices = rng.integers(
                0,
                episode_count,
                size=(count, episode_count),
            )
            sampled = values[indices]
            numerator = sampled[:, :, 0].mean(axis=1)
            denominator = sampled[:, :, 1].mean(axis=1)
            if bool(np.any(denominator <= 0.0)):
                return [None, None]
            macro += numerator / denominator
        distribution[start : start + count] = macro / task_count
    if not bool(np.all(np.isfinite(distribution))):
        return [None, None]
    return [
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    ]


def _summarize_episodes(
    episodes: Mapping[tuple[int, str, int], EpisodeAccumulator],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    grouped: dict[str, list[EpisodeAccumulator]] = defaultdict(list)
    for episode in episodes.values():
        grouped[episode.task_name].append(episode)
    if not grouped:
        raise ValueError("episode aggregation received no samples")

    per_task: dict[str, object] = {}
    numeric: dict[str, dict[str, object]] = {}
    for task in sorted(grouped):
        task_episodes = sorted(
            grouped[task], key=lambda item: (item.source_index, item.episode_id)
        )
        endpoint_metrics: dict[str, object] = {}
        numeric_endpoints: dict[int, tuple[float, float, float]] = {}
        for horizon in ENDPOINT_HORIZONS:
            predicted = _strict_mean(
                [episode.endpoint_mse(horizon, "predicted") for episode in task_episodes]
            )
            identity = _strict_mean(
                [episode.endpoint_mse(horizon, "identity") for episode in task_episodes]
            )
            ratio = _safe_ratio(predicted, identity)
            numeric_endpoints[horizon] = (predicted, identity, ratio)
            endpoint_metrics[str(horizon)] = {
                "predicted_mse": _finite_or_none(predicted),
                "hold_last_identity_mse": _finite_or_none(identity),
                "ratio": _finite_or_none(ratio),
            }
        shuffled = _strict_mean(
            [episode.shuffled_h1_mse for episode in task_episodes]
        )
        original_h1 = numeric_endpoints[1][0]
        shuffle_ratio = _safe_ratio(shuffled, original_h1)
        sequence_count = sum(episode.sequence_count for episode in task_episodes)
        shuffle_count = sum(
            episode.shuffle_sequence_count for episode in task_episodes
        )
        coverage = shuffle_count / sequence_count if sequence_count else math.nan
        per_task[task] = {
            "num_sources": len({episode.source_index for episode in task_episodes}),
            "num_episodes": len(task_episodes),
            "num_sequences": sequence_count,
            "endpoints": endpoint_metrics,
            "action_shuffle": {
                "endpoint": 1,
                "original_predicted_mse": _finite_or_none(original_h1),
                "shuffled_action_mse": _finite_or_none(shuffled),
                "ratio": _finite_or_none(shuffle_ratio),
                "paired_sequences": shuffle_count,
                "coverage": _finite_or_none(coverage),
            },
        }
        numeric[task] = {
            "episodes": task_episodes,
            "endpoints": numeric_endpoints,
            "shuffle": (shuffled, original_h1, shuffle_ratio),
        }

    macro_endpoints: dict[str, object] = {}
    for horizon in ENDPOINT_HORIZONS:
        predicted = _strict_mean(
            [numeric[task]["endpoints"][horizon][0] for task in sorted(numeric)]
        )
        identity = _strict_mean(
            [numeric[task]["endpoints"][horizon][1] for task in sorted(numeric)]
        )
        ratios = [
            numeric[task]["endpoints"][horizon][2] for task in sorted(numeric)
        ]
        ratio = _strict_mean(ratios)
        pairs = {
            task: [
                (
                    episode.endpoint_mse(horizon, "predicted"),
                    episode.endpoint_mse(horizon, "identity"),
                )
                for episode in numeric[task]["episodes"]
            ]
            for task in sorted(numeric)
        }
        macro_endpoints[str(horizon)] = {
            "predicted_mse": _finite_or_none(predicted),
            "hold_last_identity_mse": _finite_or_none(identity),
            "ratio": _finite_or_none(ratio),
            "ratio_of_macro_mse": _finite_or_none(_safe_ratio(predicted, identity)),
            "bootstrap_95_ci": episode_block_bootstrap_ci(
                pairs, samples=bootstrap_samples, seed=bootstrap_seed
            ),
        }

    shuffled = _strict_mean(
        [numeric[task]["shuffle"][0] for task in sorted(numeric)]
    )
    original_h1 = _strict_mean(
        [numeric[task]["shuffle"][1] for task in sorted(numeric)]
    )
    shuffle_ratio = _strict_mean(
        [numeric[task]["shuffle"][2] for task in sorted(numeric)]
    )
    shuffle_pairs = {
        task: [
            (episode.shuffled_h1_mse, episode.endpoint_mse(1, "predicted"))
            for episode in numeric[task]["episodes"]
        ]
        for task in sorted(numeric)
    }
    sequence_count = sum(
        episode.sequence_count for task in numeric.values() for episode in task["episodes"]
    )
    shuffle_count = sum(
        episode.shuffle_sequence_count
        for task in numeric.values()
        for episode in task["episodes"]
    )
    coverage = shuffle_count / sequence_count if sequence_count else math.nan
    macro = {
        "num_tasks": len(numeric),
        "num_episodes": len(episodes),
        "num_sequences": sequence_count,
        "endpoints": macro_endpoints,
        "action_shuffle": {
            "endpoint": 1,
            "original_predicted_mse": _finite_or_none(original_h1),
            "shuffled_action_mse": _finite_or_none(shuffled),
            "ratio": _finite_or_none(shuffle_ratio),
            "ratio_of_macro_mse": _finite_or_none(
                _safe_ratio(shuffled, original_h1)
            ),
            "bootstrap_95_ci": episode_block_bootstrap_ci(
                shuffle_pairs, samples=bootstrap_samples, seed=bootstrap_seed
            ),
            "paired_sequences": shuffle_count,
            "coverage": _finite_or_none(coverage),
        },
    }
    return per_task, macro


def _number_is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def build_offline_gate(
    per_task: Mapping[str, object],
    macro: Mapping[str, object],
    *,
    evaluation_split: str = "validation",
    rollout_steps: int = REQUIRED_ROLLOUT_STEPS,
    thresholds: Mapping[str, float] | None = None,
    evaluated_sequences: int | None = None,
    expected_sequences: int | None = None,
    additional_finite_values: Sequence[object] = (),
) -> dict[str, object]:
    """Apply all point, CI, per-task, finite-value, and protocol conditions."""

    selected = dict(DEFAULT_GATE_THRESHOLDS)
    if thresholds is not None:
        selected.update({key: float(value) for key, value in thresholds.items()})

    endpoints = macro["endpoints"]
    shuffle = macro["action_shuffle"]
    finite_values: list[object] = [shuffle["ratio"], shuffle["coverage"]]
    finite_values.extend(additional_finite_values)
    finite_values.extend(shuffle["bootstrap_95_ci"])
    for horizon in ENDPOINT_HORIZONS:
        endpoint = endpoints[str(horizon)]
        finite_values.extend(
            [
                endpoint["predicted_mse"],
                endpoint["hold_last_identity_mse"],
                endpoint["ratio"],
                *endpoint["bootstrap_95_ci"],
            ]
        )
    for task_metrics in per_task.values():
        finite_values.extend(
            task_metrics["endpoints"][str(horizon)]["ratio"]
            for horizon in ENDPOINT_HORIZONS
        )
        finite_values.extend(
            [
                task_metrics["action_shuffle"]["ratio"],
                task_metrics["action_shuffle"]["coverage"],
            ]
        )

    conditions: dict[str, object] = {
        "validation_split": {
            "passed": evaluation_split == "validation",
            "value": evaluation_split,
            "required": "validation",
        },
        "rollout_steps_5": {
            "passed": rollout_steps == REQUIRED_ROLLOUT_STEPS,
            "value": rollout_steps,
            "required": REQUIRED_ROLLOUT_STEPS,
        },
        "complete_eligible_windows": {
            "passed": expected_sequences is None
            or evaluated_sequences == expected_sequences,
            "evaluated": evaluated_sequences,
            "expected": expected_sequences,
        },
        "all_required_values_finite": {
            "passed": bool(finite_values)
            and all(_number_is_finite(value) for value in finite_values),
        },
        "action_shuffle_coverage": {
            "passed": _number_is_finite(shuffle["coverage"])
            and float(shuffle["coverage"]) == 1.0,
            "value": shuffle["coverage"],
            "required": 1.0,
        },
    }
    boundary = selected["ratio_boundary"]
    for horizon in ENDPOINT_HORIZONS:
        label = f"R{horizon}"
        endpoint = endpoints[str(horizon)]
        point_limit = selected[f"{label}_max"]
        ratio = endpoint["ratio"]
        ci_upper = endpoint["bootstrap_95_ci"][1]
        task_ratios = {
            task: metrics["endpoints"][str(horizon)]["ratio"]
            for task, metrics in sorted(per_task.items())
        }
        conditions[f"{label}_point"] = {
            "passed": _number_is_finite(ratio) and float(ratio) <= point_limit,
            "value": ratio,
            "operator": "<=",
            "threshold": point_limit,
        }
        conditions[f"{label}_ci_upper"] = {
            "passed": _number_is_finite(ci_upper) and float(ci_upper) < boundary,
            "value": ci_upper,
            "operator": "<",
            "threshold": boundary,
        }
        conditions[f"{label}_every_task"] = {
            "passed": bool(task_ratios)
            and all(
                _number_is_finite(value) and float(value) < boundary
                for value in task_ratios.values()
            ),
            "values": task_ratios,
            "operator": "<",
            "threshold": boundary,
        }

    shuffle_ratio = shuffle["ratio"]
    shuffle_ci_lower = shuffle["bootstrap_95_ci"][0]
    shuffle_task_ratios = {
        task: metrics["action_shuffle"]["ratio"]
        for task, metrics in sorted(per_task.items())
    }
    conditions["S1_point"] = {
        "passed": _number_is_finite(shuffle_ratio)
        and float(shuffle_ratio) >= selected["S1_min"],
        "value": shuffle_ratio,
        "operator": ">=",
        "threshold": selected["S1_min"],
    }
    conditions["S1_ci_lower"] = {
        "passed": _number_is_finite(shuffle_ci_lower)
        and float(shuffle_ci_lower) > boundary,
        "value": shuffle_ci_lower,
        "operator": ">",
        "threshold": boundary,
    }
    conditions["S1_every_task"] = {
        "passed": bool(shuffle_task_ratios)
        and all(
            _number_is_finite(value) and float(value) > boundary
            for value in shuffle_task_ratios.values()
        ),
        "values": shuffle_task_ratios,
        "operator": ">",
        "threshold": boundary,
    }
    passed = all(bool(condition["passed"]) for condition in conditions.values())
    return {
        "passed": passed,
        "decision": "Go" if passed else "No-Go",
        "go_no_go": "Go" if passed else "No-Go",
        "thresholds": selected,
        "conditions": conditions,
    }


def gate_exit_code(gate: Mapping[str, object], enforce_gate: bool) -> int:
    return 1 if enforce_gate and not bool(gate.get("passed", False)) else 0


def evaluate_batches(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, object]],
    device: torch.device | str = "cpu",
    rollout_steps: int = REQUIRED_ROLLOUT_STEPS,
    max_batches: int | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    evaluation_split: str = "validation",
    gate_thresholds: Mapping[str, float] | None = None,
    expected_sequences: int | None = None,
    oracle_token_spec: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate paired diagnostics and build the mechanically decidable gate."""

    if rollout_steps not in {REQUIRED_ROLLOUT_STEPS, 10}:
        raise ValueError(
            "diagnostics require rollout_steps=5 for Gate L1 or 10 for stress testing"
        )
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    device = torch.device(device)
    recursive_error = SquaredErrorAccumulator()
    h10_prediction_error = SquaredErrorAccumulator()
    h10_identity_error = SquaredErrorAccumulator()
    representation = RepresentationAccumulator()
    task_representations: dict[str, RepresentationAccumulator] = {}
    prediction_token_errors = []
    alignment_token_errors = []
    batch_count = 0
    sample_count = 0
    shuffle_sample_count = 0
    episode_errors: dict[tuple[int, str, int], EpisodeAccumulator] = {}

    model.eval()
    with torch.inference_mode():
        for raw_batch in batches:
            if max_batches is not None and batch_count >= max_batches:
                break
            batch = _move_batch(raw_batch, device)
            embedding, state_target, actions, prediction, target = _teacher_forced_tensors(
                model, batch
            )
            if oracle_token_spec is not None:
                prediction_token_errors.append(
                    oracle_token_squared_error(
                        prediction,
                        target,
                        oracle_token_spec,
                    )
                )
                alignment_token_errors.append(
                    oracle_token_squared_error(
                        embedding,
                        state_target,
                        oracle_token_spec,
                    )
                )
            batch_size = int(embedding.shape[0])
            sources = _integer_metadata(batch.get("source_index"), "source_index", batch_size)
            tasks = _string_metadata(batch.get("task_name"), "task_name", batch_size)
            episode_ids = _integer_metadata(batch.get("episode_id"), "episode_id", batch_size)
            representation.update(embedding)
            for task in sorted(set(tasks)):
                indices = [index for index, value in enumerate(tasks) if value == task]
                task_representations.setdefault(task, RepresentationAccumulator()).update(
                    embedding[indices]
                )
            sample_count += batch_size
            batch_count += 1

            all_actions = _all_actions(batch)
            shuffled_all_actions = _validated_shuffled_actions(all_actions, batch)
            history = int(model.history_size)
            # Intervene only on the action that follows the observed history.
            # Earlier actions already produced the fixed history embeddings and
            # must remain factual; shuffling them would let stale-action models
            # pass S1 without responding to the action being evaluated.
            intervened_teacher_actions = all_actions[:, :history].clone()
            intervened_teacher_actions[:, history - 1] = shuffled_all_actions[
                :, history - 1
            ]
            shuffled_prediction = model.predict(
                embedding[:, :history], intervened_teacher_actions
            )
            shuffle_sample_count += batch_size
            teacher_predicted_sse = _per_sample_sse(prediction, target)
            teacher_identity_sse = _per_sample_sse(
                state_target[:, :history],
                target,
            )
            teacher_shuffled_sse = _per_sample_sse(
                shuffled_prediction,
                target,
            )
            teacher_action_sensitivity_sse = _per_sample_sse(
                shuffled_prediction,
                prediction,
            )
            alignment_sse = _per_sample_sse(embedding, state_target)
            teacher_elements = int(target[0].numel())
            alignment_elements = int(state_target[0].numel())

            if embedding.shape[1] < history + rollout_steps:
                raise ValueError(
                    "recursive evaluation requires "
                    f"{history + rollout_steps} frames, got {embedding.shape[1]}"
                )
            required_actions = history - 1 + rollout_steps
            if all_actions.shape[1] < required_actions:
                raise ValueError(
                    "recursive evaluation requires "
                    f"{required_actions} action blocks, got {all_actions.shape[1]}"
                )
            recursive_prediction = model.rollout_embeddings(
                embedding[:, :history],
                all_actions[:, : history - 1],
                all_actions[:, history - 1 : required_actions].unsqueeze(1),
            )[:, 0]
            recursive_target = state_target[:, history : history + rollout_steps]
            recursive_error.update(recursive_prediction, recursive_target)
            shuffled_h1_prediction = model.rollout_embeddings(
                embedding[:, :history],
                all_actions[:, : history - 1],
                shuffled_all_actions[:, history - 1 : history].unsqueeze(1),
            )[:, 0, 0]
            hold_last = state_target[:, history - 1]
            if rollout_steps >= 10:
                h10_prediction_error.update(
                    recursive_prediction[:, 9], recursive_target[:, 9]
                )
                h10_identity_error.update(hold_last, recursive_target[:, 9])
            predicted_sse = {
                horizon: _per_sample_sse(
                    recursive_prediction[:, horizon - 1],
                    recursive_target[:, horizon - 1],
                )
                for horizon in ENDPOINT_HORIZONS
            }
            identity_sse = {
                horizon: _per_sample_sse(
                    hold_last,
                    recursive_target[:, horizon - 1],
                )
                for horizon in ENDPOINT_HORIZONS
            }
            shuffled_h1_sse = _per_sample_sse(
                shuffled_h1_prediction, recursive_target[:, 0]
            )
            elements_per_endpoint = int(recursive_target[:, 0].numel() // batch_size)
            for index in range(batch_size):
                key = (sources[index], tasks[index], episode_ids[index])
                episode = episode_errors.setdefault(
                    key,
                    EpisodeAccumulator(
                        source_index=sources[index],
                        task_name=tasks[index],
                        episode_id=episode_ids[index],
                    ),
                )
                episode.sequence_count += 1
                episode.shuffle_sequence_count += 1
                episode.teacher_predicted_sse += float(
                    teacher_predicted_sse[index]
                )
                episode.teacher_identity_sse += float(
                    teacher_identity_sse[index]
                )
                episode.teacher_shuffled_sse += float(
                    teacher_shuffled_sse[index]
                )
                episode.teacher_action_sensitivity_sse += float(
                    teacher_action_sensitivity_sse[index]
                )
                episode.teacher_element_count += teacher_elements
                episode.alignment_sse += float(alignment_sse[index])
                episode.alignment_element_count += alignment_elements
                for horizon in ENDPOINT_HORIZONS:
                    episode.predicted_sse[horizon] += float(predicted_sse[horizon][index])
                    episode.identity_sse[horizon] += float(identity_sse[horizon][index])
                    episode.endpoint_element_count[horizon] += elements_per_endpoint
                episode.shuffled_h1_sse += float(shuffled_h1_sse[index])
                episode.shuffled_h1_element_count += elements_per_endpoint

    if batch_count == 0:
        raise ValueError("evaluation received no batches")
    ordered_episodes = [
        episode for _, episode in sorted(episode_errors.items())
    ]
    pred_mse = _pooled_mse(
        (episode.teacher_mse("predicted"), episode.teacher_element_count)
        for episode in ordered_episodes
    )
    copy_mse = _pooled_mse(
        (episode.teacher_mse("identity"), episode.teacher_element_count)
        for episode in ordered_episodes
    )
    shuffle_mse = _pooled_mse(
        (episode.teacher_mse("shuffled"), episode.teacher_element_count)
        for episode in ordered_episodes
    )
    sensitivity_mse = _pooled_mse(
        (
            episode.teacher_mse("action_sensitivity"),
            episode.teacher_element_count,
        )
        for episode in ordered_episodes
    )
    alignment_mse = _pooled_mse(
        (episode.alignment_mse, episode.alignment_element_count)
        for episode in ordered_episodes
    )
    per_task, macro = _summarize_episodes(
        episode_errors,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    episode_blocks = []
    for key, episode in sorted(episode_errors.items()):
        episode_blocks.append(
            {
                "source_index": key[0],
                "task_name": key[1],
                "episode_id": key[2],
                "num_sequences": episode.sequence_count,
                "num_action_shuffle_sequences": episode.shuffle_sequence_count,
                "teacher_forced": {
                    "element_count": episode.teacher_element_count,
                    "predicted_mse": _finite_or_none(
                        episode.teacher_mse("predicted")
                    ),
                    "identity_copy_mse": _finite_or_none(
                        episode.teacher_mse("identity")
                    ),
                    "prediction_vs_copy_ratio": _finite_or_none(
                        _safe_ratio(
                            episode.teacher_mse("predicted"),
                            episode.teacher_mse("identity"),
                        )
                    ),
                    "action_shuffle_mse": _finite_or_none(
                        episode.teacher_mse("shuffled")
                    ),
                    "action_shuffle_delta_mse": _finite_or_none(
                        episode.teacher_mse("shuffled")
                        - episode.teacher_mse("predicted")
                    ),
                    "action_sensitivity_mse": _finite_or_none(
                        episode.teacher_mse("action_sensitivity")
                    ),
                },
                "oracle_alignment": {
                    "element_count": episode.alignment_element_count,
                    "mse": _finite_or_none(episode.alignment_mse),
                },
                "recursive_h1_action_shuffle": {
                    "element_count": episode.shuffled_h1_element_count,
                    "mse": _finite_or_none(episode.shuffled_h1_mse),
                    "delta_mse": _finite_or_none(
                        episode.shuffled_h1_mse
                        - episode.endpoint_mse(1, "predicted")
                    ),
                },
                "endpoints": {
                    str(horizon): {
                        "element_count": episode.endpoint_element_count[horizon],
                        "predicted_mse": _finite_or_none(
                            episode.endpoint_mse(horizon, "predicted")
                        ),
                        "hold_last_identity_mse": _finite_or_none(
                            episode.endpoint_mse(horizon, "identity")
                        ),
                        "ratio": _finite_or_none(
                            _safe_ratio(
                                episode.endpoint_mse(horizon, "predicted"),
                                episode.endpoint_mse(horizon, "identity"),
                            )
                        ),
                    }
                    for horizon in ENDPOINT_HORIZONS
                },
            }
        )
    gate = build_offline_gate(
        per_task,
        macro,
        evaluation_split=evaluation_split,
        rollout_steps=rollout_steps,
        thresholds=gate_thresholds,
        evaluated_sequences=sample_count,
        expected_sequences=expected_sequences,
        additional_finite_values=(
            pred_mse,
            copy_mse,
            shuffle_mse,
            sensitivity_mse,
            recursive_error.mse,
            alignment_mse,
        ),
    )
    oracle_token_mse = None
    if oracle_token_spec is not None:
        oracle_token_mse = {
            "prediction": summarize_oracle_token_mse(
                merge_oracle_token_errors(prediction_token_errors)
            ),
            "alignment": summarize_oracle_token_mse(
                merge_oracle_token_errors(alignment_token_errors)
            ),
        }
    result: dict[str, object] = {
        "teacher_forced_pred_mse": _finite_or_none(pred_mse),
        "identity_copy_mse": _finite_or_none(copy_mse),
        "oracle_alignment_mse": _finite_or_none(alignment_mse),
        "oracle_token_mse": oracle_token_mse,
        "action_shuffle_mse": _finite_or_none(shuffle_mse),
        "action_shuffle_delta_mse": (
            _finite_or_none(shuffle_mse - pred_mse)
            if pred_mse is not None and shuffle_mse is not None
            else None
        ),
        "action_sensitivity_mse": _finite_or_none(sensitivity_mse),
        "recursive_rollout_mse": _finite_or_none(recursive_error.mse),
        "h10_stress": {
            "enabled": rollout_steps >= 10,
            "predicted_mse": _finite_or_none(h10_prediction_error.mse),
            "hold_last_identity_mse": _finite_or_none(h10_identity_error.mse),
            "ratio": _finite_or_none(
                _safe_ratio(h10_prediction_error.mse, h10_identity_error.mse)
            ),
            "part_of_primary_gate": False,
        },
        "representation": {
            "overall": representation.metrics(),
            "per_task": {
                task: accumulator.metrics()
                for task, accumulator in sorted(task_representations.items())
            },
            "source": "learned_observation_embedding",
        },
        "rollout_steps": int(rollout_steps),
        "num_batches": batch_count,
        "num_sequences": sample_count,
        "num_eligible_sequences": expected_sequences,
        "num_action_shuffle_sequences": shuffle_sample_count,
        "action_shuffle_coverage": _finite_or_none(
            shuffle_sample_count / sample_count if sample_count else math.nan
        ),
        "endpoint_horizons": list(ENDPOINT_HORIZONS),
        "endpoint_metrics": macro["endpoints"],
        "per_task": per_task,
        "episode_blocks": episode_blocks,
        "macro": macro,
        "bootstrap": {
            "method": "episode_block_within_task_equal_task_macro",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
        },
        "aggregation": {
            "error_accumulation": "float64_sse_then_element_count",
            "window_to_episode": "element_weighted_mse",
            "episode_to_task": "equal_episode_mean",
            "task_to_macro": "equal_task_mean",
            "ratio_macro": "equal_mean_of_task_ratios",
        },
        "gate": gate,
    }
    result["prediction_vs_copy_ratio"] = _finite_or_none(
        _safe_ratio(pred_mse, copy_mse)
    )
    return result



def _publish_text_exclusive_atomic(
    output: Path,
    payload: str,
    *,
    revalidate: Callable[[], None] | None = None,
) -> None:
    """Atomically publish one report without replacing an existing path."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if revalidate is not None:
            revalidate()
        try:
            os.link(temporary_path, output)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite evaluator report: {output}"
            ) from None
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=None,
        help=(
            "Optional trusted SHA-256 pin. It is verified before checkpoint "
            "deserialization; the unified input snapshot verifies it again after "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help="Strict pre-action NPZ source; repeat for multi-task evaluation.",
    )
    parser.add_argument(
        "--correction-cache",
        action="append",
        default=[],
        help="Optional q_corrected cache matching each --data source.",
    )
    parser.add_argument(
        "--oracle-target-cache",
        action="append",
        default=[],
        help=(
            "Frozen oracle target cache matching each --data source; required by "
            "frozen-oracle checkpoints for every train/eval grounding intervention."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("validation", "train", "all"),
        default="validation",
        help="Episode split persisted in the checkpoint; defaults to held-out validation.",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("checkpoint", "fcpg", "nominal"),
        default="checkpoint",
        help="Use the checkpoint grounding contract or explicitly evaluate FCPG/nominal.",
    )
    parser.add_argument(
        "--anchor-visual-mode-override",
        choices=("checkpoint", *PARAMETER_COMPATIBLE_TRIPLET_MODES),
        default="checkpoint",
        help=(
            "Development-only input intervention for the preregistered anchor-visual "
            "ablation. It is rejected for every other checkpoint and split."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        choices=(REQUIRED_ROLLOUT_STEPS, 10),
        default=REQUIRED_ROLLOUT_STEPS,
        help=(
            "Use 5 for the primary H=1/3/5 gate or 10 for a separately labeled "
            "recursive stress test."
        ),
    )
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--r1-max", type=float, default=DEFAULT_GATE_THRESHOLDS["R1_max"])
    parser.add_argument("--r3-max", type=float, default=DEFAULT_GATE_THRESHOLDS["R3_max"])
    parser.add_argument("--r5-max", type=float, default=DEFAULT_GATE_THRESHOLDS["R5_max"])
    parser.add_argument("--s1-min", type=float, default=DEFAULT_GATE_THRESHOLDS["S1_min"])
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help="Exit nonzero when the offline gate reports No-Go.",
    )
    parser.add_argument(
        "--allow-unfrozen-checkpoint",
        action="store_true",
        help="Explicitly allow legacy/smoke checkpoints without frozen provenance.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    return parser


def _load_checkpoint_metadata(path: str | Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("LeWM checkpoint must be a dictionary")
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type != CHECKPOINT_TYPE:
        raise ValueError(
            f"expected checkpoint_type={CHECKPOINT_TYPE!r}, got {checkpoint_type!r}"
        )
    return checkpoint


def _snapshot_digest(
    path: str | Path,
    file_snapshots: Mapping[str, Mapping[str, object]],
) -> str:
    requested = Path(path).expanduser()
    resolved = requested.resolve()
    snapshot = file_snapshots.get(str(resolved))
    if not isinstance(snapshot, Mapping):
        raise RuntimeError(f"missing evaluation input snapshot: {resolved}")
    current = requested.stat()
    identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if tuple(snapshot.get("identity", ())) != identity:
        raise RuntimeError(f"evaluation input changed after snapshot: {resolved}")
    digest = snapshot.get("sha256")
    if not isinstance(digest, str):
        raise RuntimeError(f"invalid evaluation input snapshot: {resolved}")
    return digest


def _validate_manifest_with_file_snapshots(
    manifest: Mapping[str, object],
    file_snapshots: Mapping[str, Mapping[str, object]],
    *,
    check_git: bool,
    git_state: Mapping[str, object] | None,
    git_error: str | None,
) -> None:
    if manifest.get("schema_version") != _lewm_provenance.SCHEMA_VERSION:
        raise _lewm_provenance.ManifestError(
            f"unsupported manifest schema_version: {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("created_at"), str):
        raise _lewm_provenance.ManifestError("created_at must be a string")
    repository = _lewm_provenance._require_mapping(
        manifest.get("repository"), "repository"
    )
    repo_path_value = repository.get("absolute_path")
    if not isinstance(repo_path_value, str) or not Path(repo_path_value).is_absolute():
        raise _lewm_provenance.ManifestError(
            "repository.absolute_path must be an absolute path"
        )
    repo_root = Path(repo_path_value)

    errors: list[str] = []
    expected_files: dict[str, tuple[int, int, str]] = {}
    for label, record in _lewm_provenance._iter_file_records(manifest):
        path, expected_size, expected_mtime_ns, expected_sha256 = (
            _lewm_provenance._validate_record_schema(label, record)
        )
        key = str(path)
        expected = (expected_size, expected_mtime_ns, expected_sha256)
        previous_expected = expected_files.get(key)
        if previous_expected is not None and previous_expected != expected:
            errors.append(f"conflicting manifest records for {path}")
            continue
        expected_files[key] = expected

        stored_relative = record["repo_relative_path"]
        actual_relative = _lewm_provenance._repo_relative_path(path, repo_root)
        if stored_relative != actual_relative:
            errors.append(
                f"repo-relative path changed for {path}: expected "
                f"{stored_relative!r}, found {actual_relative!r}"
            )

        try:
            snapshot = file_snapshots.get(str(path.expanduser().resolve()))
            if not isinstance(snapshot, Mapping):
                errors.append(f"missing file snapshot: {path}")
                continue
            current = path.stat()
            if not stat.S_ISREG(current.st_mode):
                errors.append(f"input is not a regular file: {path}")
                continue
            identity = (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            )
            if tuple(snapshot.get("identity", ())) != identity:
                errors.append(f"input changed after snapshot: {path}")
                continue
            current_size = snapshot.get("size")
            current_sha256 = snapshot.get("sha256")
        except FileNotFoundError:
            errors.append(f"missing file: {path}")
            continue
        except OSError as error:
            errors.append(f"cannot read {path}: {error}")
            continue

        if current_size != expected_size:
            errors.append(
                f"size changed for {path}: expected {expected_size}, "
                f"found {current_size}"
            )
        if current_sha256 != expected_sha256:
            errors.append(
                f"sha256 changed for {path}: expected {expected_sha256}, "
                f"found {current_sha256}"
            )

    if check_git:
        expected_head = repository.get("head")
        expected_diff_sha256 = repository.get("dirty_diff_sha256")
        expected_worktree_sha256 = repository.get("worktree_sha256")
        if not isinstance(expected_head, str):
            raise _lewm_provenance.ManifestError("repository.head must be a string")
        if not isinstance(expected_diff_sha256, str):
            raise _lewm_provenance.ManifestError(
                "repository.dirty_diff_sha256 must be a string"
            )
        if expected_worktree_sha256 is not None and not isinstance(
            expected_worktree_sha256, str
        ):
            raise _lewm_provenance.ManifestError(
                "repository.worktree_sha256 must be a string"
            )
        if git_error is not None:
            errors.append(f"cannot inspect repository state: {git_error}")
        elif git_state is None:
            errors.append("cannot inspect repository state: snapshot is missing")
        else:
            if git_state.get("head") != expected_head:
                errors.append(
                    f"git HEAD changed: expected {expected_head}, "
                    f"found {git_state.get('head')}"
                )
            current_worktree_sha256 = git_state.get("worktree_sha256")
            current_diff_sha256 = git_state.get("dirty_diff_sha256")
            if expected_worktree_sha256 is not None and (
                current_worktree_sha256 != expected_worktree_sha256
            ):
                errors.append(
                    "git worktree sha256 changed: expected "
                    f"{expected_worktree_sha256}, found {current_worktree_sha256}"
                )
            elif current_diff_sha256 != expected_diff_sha256:
                errors.append(
                    "git dirty diff sha256 changed: expected "
                    f"{expected_diff_sha256}, found {current_diff_sha256}"
                )

    if errors:
        raise _lewm_provenance.ManifestValidationError(errors)


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, object],
    checkpoint_path: str | Path,
    *,
    allow_unfrozen: bool,
    check_git: bool = True,
    file_snapshots: Mapping[str, Mapping[str, object]] | None = None,
    git_state: Mapping[str, object] | None = None,
    git_error: str | None = None,
) -> dict[str, object]:
    """Validate checkpoint-bound code/data provenance before evaluation."""

    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        if allow_unfrozen:
            return {
                "validated": False,
                "reason": "checkpoint_has_no_frozen_provenance",
            }
        raise ValueError(
            "checkpoint has no frozen provenance; pass "
            "--allow-unfrozen-checkpoint only for explicitly labeled smoke/legacy runs"
        )
    expected_sha256 = provenance.get("sha256")
    manifest = provenance.get("manifest")
    external_path_value = provenance.get("path")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("checkpoint provenance SHA-256 is invalid")
    if not isinstance(manifest, Mapping):
        raise ValueError("checkpoint provenance manifest payload is invalid")
    if not isinstance(external_path_value, str):
        raise ValueError("checkpoint provenance manifest path is invalid")

    external_path = Path(external_path_value).expanduser().resolve()
    if not external_path.is_file():
        raise FileNotFoundError(f"missing frozen input manifest: {external_path}")
    external_sha256 = (
        _file_sha256(external_path)
        if file_snapshots is None
        else _snapshot_digest(external_path, file_snapshots)
    )
    if external_sha256 != expected_sha256:
        raise ValueError("external frozen input manifest SHA-256 differs from checkpoint")
    external_manifest = json.loads(external_path.read_text(encoding="utf-8"))
    if external_manifest != dict(manifest):
        raise ValueError("external frozen input manifest payload differs from checkpoint")

    run_sidecar = Path(checkpoint_path).expanduser().resolve().parent / "input_manifest.json"
    if not run_sidecar.is_file():
        raise FileNotFoundError(f"missing run input manifest sidecar: {run_sidecar}")
    sidecar_sha256 = (
        _file_sha256(run_sidecar)
        if file_snapshots is None
        else _snapshot_digest(run_sidecar, file_snapshots)
    )
    if sidecar_sha256 != expected_sha256:
        raise ValueError("run input manifest sidecar SHA-256 differs from checkpoint")
    sidecar_manifest = json.loads(run_sidecar.read_text(encoding="utf-8"))
    if sidecar_manifest != dict(manifest):
        raise ValueError("run input manifest sidecar payload differs from checkpoint")

    if file_snapshots is None:
        validate_manifest(manifest, check_git=check_git)
    else:
        _validate_manifest_with_file_snapshots(
            manifest,
            file_snapshots,
            check_git=check_git,
            git_state=git_state,
            git_error=git_error,
        )
    return {
        "validated": True,
        "manifest": str(external_path),
        "manifest_sha256": expected_sha256,
        "run_sidecar": str(run_sidecar),
        "git_state_validated": check_git,
    }


def resolve_grounding_mode(
    checkpoint: Mapping[str, object], requested: str
) -> str:
    if requested != "checkpoint":
        return requested
    data_config = _nested_mapping(checkpoint, "data_config")
    mode = str(data_config.get("grounding_mode", "")).lower()
    if mode not in {"fcpg", "nominal"}:
        raise ValueError("checkpoint does not contain an explicit grounding_mode")
    return mode


def resolve_anchor_visual_ablation_contract(
    checkpoint: Mapping[str, object],
    checkpoint_path: str | Path,
    requested: str,
    *,
    split: str,
    grounding_mode_request: str,
) -> dict[str, object]:
    """Validate the metadata-only contract for a development input override."""

    if requested == "checkpoint":
        return {"active": False, "request": requested}
    eval_mode = validate_anchor_visual_override(requested)
    if split != "validation":
        raise ValueError(
            "anchor-visual mode overrides are development-only and require "
            "--split validation"
        )
    if grounding_mode_request != "checkpoint":
        raise ValueError(
            "anchor-visual mode overrides require --grounding-mode checkpoint"
        )
    if Path(checkpoint_path).name != "lewm_grounded_latest.pt":
        raise ValueError(
            "anchor-visual mode overrides require lewm_grounded_latest.pt; "
            "per-run best checkpoints are forbidden"
        )

    resolved = _nested_mapping(checkpoint, "resolved_config")
    contract = resolved.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("anchor-visual override checkpoint has no contract")
    required_contract = {
        "gate": "anchor_visual_input_ablation",
        "pair_id": ANCHOR_ABLATION_PAIR_ID,
        "development_only": True,
        "frozen_gate_l1_test_reuse_forbidden": True,
        "training_grounding": "fcpg",
        "optimizer_step_budget": ANCHOR_ABLATION_STEP,
    }
    for field, expected in required_contract.items():
        if contract.get(field) != expected:
            raise ValueError(
                "anchor-visual override contract mismatch for "
                f"{field}: expected {expected!r}, got {contract.get(field)!r}"
            )

    train_mode = validate_anchor_visual_override(
        contract.get("anchor_visual_mode"), field_name="contract.anchor_visual_mode"
    )
    configured_frame = _nested_mapping(checkpoint, "model_config").get("frame")
    resolved_frame = _nested_mapping(resolved, "model").get("frame")
    if not isinstance(configured_frame, Mapping) or not isinstance(
        resolved_frame, Mapping
    ):
        raise ValueError("anchor-visual override checkpoint has no frame config")
    configured_modes = (
        validate_anchor_visual_override(
            configured_frame.get("anchor_visual_mode"),
            field_name="model_config.frame.anchor_visual_mode",
        ),
        validate_anchor_visual_override(
            resolved_frame.get("anchor_visual_mode"),
            field_name="resolved_config.model.frame.anchor_visual_mode",
        ),
    )
    if any(mode != train_mode for mode in configured_modes):
        raise ValueError(
            "checkpoint frame configs disagree with the training anchor mode"
        )

    step = checkpoint.get("step")
    metrics = checkpoint.get("metrics")
    metric_step = metrics.get("step") if isinstance(metrics, Mapping) else None
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or isinstance(metric_step, bool)
        or not isinstance(metric_step, int)
    ):
        raise TypeError("anchor-visual checkpoint and metric steps must be integers")
    if step != ANCHOR_ABLATION_STEP or metric_step != ANCHOR_ABLATION_STEP:
        raise ValueError(
            "anchor-visual overrides require checkpoint.step == metrics.step == "
            f"{ANCHOR_ABLATION_STEP}; got {step} and {metric_step}"
        )
    if checkpoint.get("run_kind") != "formal":
        raise ValueError("anchor-visual overrides require a formal checkpoint")

    expected_parameter_count = contract.get("expected_parameter_count")
    if expected_parameter_count != ANCHOR_ABLATION_PARAMETER_COUNT:
        raise ValueError(
            "anchor-visual override requires expected_parameter_count="
            f"{ANCHOR_ABLATION_PARAMETER_COUNT}, got {expected_parameter_count!r}"
        )
    seed = resolved.get("seed")
    if seed != ANCHOR_ABLATION_SEED:
        raise ValueError(
            f"anchor-visual override requires seed={ANCHOR_ABLATION_SEED}, got {seed!r}"
        )
    loss = _nested_mapping(resolved, "loss")
    if loss.get("sigreg_weight") != 0.02:
        raise ValueError("anchor-visual override requires sigreg_weight=0.02")

    target_sha256 = contract.get("target_encoder_sha256")
    if target_sha256 != ANCHOR_ABLATION_TARGET_SHA256:
        raise ValueError("anchor-visual override target encoder hash is invalid")
    provenance = checkpoint.get("provenance")
    manifest = provenance.get("manifest") if isinstance(provenance, Mapping) else None
    episode_manifest = (
        manifest.get("episode_manifest") if isinstance(manifest, Mapping) else None
    )
    split_sha256 = (
        episode_manifest.get("sha256")
        if isinstance(episode_manifest, Mapping)
        else None
    )
    if split_sha256 != ANCHOR_ABLATION_SPLIT_SHA256:
        raise ValueError("anchor-visual override episode split hash is invalid")

    return {
        "active": True,
        "request": requested,
        "train_mode": train_mode,
        "eval_mode": eval_mode,
        "step": step,
        "seed": seed,
        "target_sha256": target_sha256,
        "data_split_sha256": split_sha256,
        "pair_id": contract.get("pair_id"),
        "development_only": True,
        "formal_gate_eligible": False,
        "frozen_gate_l1_test_reuse_forbidden": True,
    }


def apply_anchor_visual_mode_override(
    model: torch.nn.Module,
    checkpoint: Mapping[str, object],
    contract_report: Mapping[str, object],
) -> dict[str, object]:
    """Apply a previously validated, weight-compatible input intervention."""

    frame_encoder = getattr(model, "frame_encoder", None)
    actual_mode = getattr(frame_encoder, "anchor_visual_mode", None)
    if not contract_report.get("active"):
        return {
            **dict(contract_report),
            "train_mode": actual_mode,
            "eval_mode": actual_mode,
        }

    train_mode = contract_report["train_mode"]
    if actual_mode != train_mode:
        raise ValueError(
            "loaded model disagrees with the training anchor mode: "
            f"expected {train_mode!r}, got {actual_mode!r}"
        )
    resolved = _nested_mapping(checkpoint, "resolved_config")
    contract = resolved.get("contract")
    assert isinstance(contract, Mapping)
    expected_parameter_count = contract.get("expected_parameter_count")
    if (
        isinstance(expected_parameter_count, bool)
        or not isinstance(expected_parameter_count, int)
        or expected_parameter_count <= 0
    ):
        raise ValueError("anchor-visual expected_parameter_count is invalid")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != expected_parameter_count:
        raise ValueError(
            "anchor-visual parameter count mismatch: "
            f"expected {expected_parameter_count}, got {parameter_count}"
        )

    eval_mode = str(contract_report["eval_mode"])
    frame_encoder.anchor_visual_mode = eval_mode
    return {
        **dict(contract_report),
        "parameter_count": parameter_count,
        "override_applied": eval_mode != train_mode,
    }


def _checkpoint_source(
    checkpoint: Mapping[str, object],
    data_path: str | Path,
) -> Mapping[str, object]:
    data_config = _nested_mapping(checkpoint, "data_config")
    sources = data_config.get("sources", [])
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ValueError("checkpoint data_config.sources must be a sequence")
    requested = Path(data_path).expanduser().resolve()
    matches = []
    for source in sources:
        if not isinstance(source, Mapping) or not source.get("path"):
            continue
        if Path(str(source["path"])).expanduser().resolve() == requested:
            matches.append(source)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one checkpoint source for {requested}, found {len(matches)}"
        )
    return matches[0]


def validate_checkpoint_correction_caches(
    checkpoint: Mapping[str, object],
    data_paths: Sequence[str],
    correction_caches: Sequence[str],
) -> None:
    """Require checkpoint-mode evaluation to reuse the recorded FCPG caches."""

    if len(correction_caches) != len(data_paths):
        raise ValueError(
            "checkpoint FCPG evaluation requires one correction cache per data source"
        )
    for data_path, cache_path in zip(data_paths, correction_caches, strict=True):
        source = _checkpoint_source(checkpoint, data_path)
        expected = source.get("correction_cache")
        if not expected:
            raise ValueError(f"checkpoint source {data_path} has no correction cache")
        expected_path = Path(str(expected)).expanduser().resolve()
        supplied_path = _correction_array_path(cache_path)
        if expected_path != supplied_path:
            raise ValueError(
                f"checkpoint correction cache mismatch for {data_path}: "
                f"expected {expected_path}, got {supplied_path}"
            )


def validate_checkpoint_oracle_target_caches(
    checkpoint: Mapping[str, object],
    data_paths: Sequence[str],
    oracle_target_caches: Sequence[str],
    *,
    manifests: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """Bind every supplied frozen oracle target to its checkpoint contract."""

    if len(oracle_target_caches) != len(data_paths):
        raise ValueError(
            "frozen-oracle evaluation requires one recorded target cache per data source"
        )
    if manifests is not None and len(manifests) != len(data_paths):
        raise ValueError("oracle target manifest count differs from data source count")
    for index, (data_path, cache_path) in enumerate(
        zip(data_paths, oracle_target_caches, strict=True)
    ):
        source = _checkpoint_source(checkpoint, data_path)
        required = {
            "oracle_target_cache": source.get("oracle_target_cache"),
            "oracle_target_array": source.get("oracle_target_array"),
            "oracle_target_sha256": source.get("oracle_target_sha256"),
            "oracle_target_metadata_sha256": source.get(
                "oracle_target_metadata_sha256"
            ),
            "oracle_target_encoder_sha256": source.get(
                "oracle_target_encoder_sha256"
            ),
        }
        if any(not isinstance(value, str) or not value for value in required.values()):
            raise ValueError(
                f"checkpoint source {data_path} has an incomplete oracle target contract"
            )
        manifest = (
            oracle_target_cache_manifest(cache_path)
            if manifests is None
            else manifests[index]
        )
        expected_cache = Path(str(required["oracle_target_cache"])).expanduser().resolve()
        expected_array = Path(str(required["oracle_target_array"])).expanduser().resolve()
        if Path(str(manifest["input"])).resolve() != expected_cache:
            raise ValueError(
                f"checkpoint oracle target cache mismatch for {data_path}"
            )
        if Path(str(manifest["array"])).resolve() != expected_array:
            raise ValueError(
                f"checkpoint oracle target array mismatch for {data_path}"
            )
        comparisons = {
            "array_sha256": required["oracle_target_sha256"],
            "metadata_sha256": required["oracle_target_metadata_sha256"],
            "encoder_checkpoint_sha256": required[
                "oracle_target_encoder_sha256"
            ],
        }
        for field, expected in comparisons.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"checkpoint oracle target {field} mismatch for {data_path}"
                )


def validate_anchor_frozen_target_hash_contract(
    checkpoint: Mapping[str, object],
    data_paths: Sequence[str],
    oracle_target_manifests: Sequence[Mapping[str, object]],
    anchor_target_sha256: object,
) -> str:
    """Bind every frozen-target declaration and cache to the anchor hash."""

    anchor_hash = _normalize_sha256(
        anchor_target_sha256,
        "anchor-visual contract.target_encoder_sha256",
    )
    world_config = _nested_mapping(checkpoint, "model_config").get("world_model")
    if not isinstance(world_config, Mapping) or str(
        world_config.get("target_mode", "online")
    ) != "frozen_oracle":
        raise ValueError("anchor-visual ablation requires frozen_oracle targets")

    data_config = _nested_mapping(checkpoint, "data_config")
    target_contract = data_config.get("target_contract")
    if not isinstance(target_contract, Mapping):
        raise ValueError("anchor-visual ablation has no data_config.target_contract")
    if target_contract.get("mode") != "frozen_oracle":
        raise ValueError("anchor-visual target_contract mode must be frozen_oracle")

    sources = data_config.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ValueError("anchor-visual data_config.sources must be a sequence")
    if len(sources) != len(data_paths):
        raise ValueError(
            "anchor-visual evaluation must supply every checkpoint data source"
        )
    checkpoint_sources: dict[str, tuple[int, Mapping[str, object]]] = {}
    for source_index, source in enumerate(sources):
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
            raise ValueError("anchor-visual checkpoint source path is invalid")
        source_path = str(Path(str(source["path"])).expanduser().resolve())
        if source_path in checkpoint_sources:
            raise ValueError(
                "anchor-visual evaluation data sources differ from checkpoint sources"
            )
        checkpoint_sources[source_path] = (source_index, source)
    requested_paths = [
        str(Path(path).expanduser().resolve()) for path in data_paths
    ]
    if set(checkpoint_sources) != set(requested_paths):
        raise ValueError(
            "anchor-visual evaluation data sources differ from checkpoint sources"
        )
    if len(oracle_target_manifests) != len(data_paths):
        raise ValueError(
            "anchor-visual evaluation requires one oracle metadata manifest per source"
        )

    declared_hashes: list[tuple[str, object]] = [
        (
            "data_config.target_contract.encoder_checkpoint_sha256",
            target_contract.get("encoder_checkpoint_sha256"),
        )
    ]
    for cache_index, (data_path, manifest) in enumerate(
        zip(data_paths, oracle_target_manifests, strict=True)
    ):
        source_path = str(Path(data_path).expanduser().resolve())
        source_index, source = checkpoint_sources[source_path]
        declared_hashes.extend(
            [
                (
                    f"data_config.sources[{source_index}].oracle_target_encoder_sha256",
                    source.get("oracle_target_encoder_sha256"),
                ),
                (
                    f"oracle_target_cache[{cache_index}].metadata.encoder_checkpoint_sha256",
                    manifest.get("encoder_checkpoint_sha256"),
                ),
            ]
        )

    normalized_hashes = []
    for label, value in declared_hashes:
        try:
            normalized_hashes.append((label, _normalize_sha256(value, label)))
        except ValueError as error:
            raise ValueError(
                f"anchor-visual frozen target hash contract is incomplete: {label}"
            ) from error
    conflicts = [
        (label, value)
        for label, value in normalized_hashes
        if value != anchor_hash
    ]
    if conflicts:
        details = ", ".join(f"{label}={value}" for label, value in conflicts)
        raise ValueError(
            "anchor-visual frozen target hash conflict with "
            f"{anchor_hash}: {details}"
        )
    return anchor_hash


def _episode_label(dataset: LeWMSequenceDataset, episode_index: int) -> str:
    raw_index = int(dataset.episodes[episode_index][0])
    return f"{dataset.task_name[raw_index]}:{dataset.episode_id[raw_index]}"


def select_checkpoint_split(
    dataset: LeWMSequenceDataset,
    checkpoint: Mapping[str, object],
    data_path: str | Path,
    split: str,
) -> Dataset:
    """Select the exact episode split persisted in a training checkpoint."""

    if split == "all":
        return dataset
    key = {"validation": "val_episodes", "train": "train_episodes"}.get(split)
    if key is None:
        raise ValueError(f"unknown evaluation split: {split}")
    source = _checkpoint_source(checkpoint, data_path)
    labels = source.get(key)
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise ValueError(f"checkpoint source does not contain a valid {key} list")
    expected = {str(label) for label in labels}
    selected = [
        window_index
        for window_index, (episode_index, _) in enumerate(dataset.windows)
        if _episode_label(dataset, episode_index) in expected
    ]
    observed = {
        _episode_label(dataset, dataset.windows[window_index][0])
        for window_index in selected
    }
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(
            f"dataset is missing {len(missing)} checkpoint {split} episode(s): {missing[:5]}"
        )
    if not selected:
        raise ValueError(f"checkpoint {split} split selects no evaluation windows")
    return Subset(dataset, selected)


def _build_datasets(
    data_paths: Sequence[str],
    correction_caches: Sequence[str],
    oracle_target_caches: Sequence[str],
    history_size: int,
    rollout_steps: int,
    frame_skip: int,
    checkpoint: Mapping[str, object],
    split: str,
    grounding_mode: str,
) -> list[Dataset]:
    caches: Sequence[str | None]
    if grounding_mode == "fcpg":
        if len(correction_caches) != len(data_paths):
            raise ValueError(
                "FCPG evaluation requires --correction-cache once per --data source"
            )
        caches = correction_caches
    else:
        if correction_caches:
            raise ValueError("nominal evaluation must not receive --correction-cache")
        caches = [None] * len(data_paths)
    world_config = _nested_mapping(checkpoint, "model_config").get("world_model")
    if not isinstance(world_config, Mapping):
        raise ValueError("checkpoint is missing model_config.world_model")
    target_mode = str(world_config.get("target_mode", "online"))
    if target_mode == "frozen_oracle":
        if len(oracle_target_caches) != len(data_paths):
            raise ValueError(
                "frozen-oracle evaluation requires --oracle-target-cache once per "
                "--data source"
            )
        target_caches: Sequence[str | None] = oracle_target_caches
    else:
        if oracle_target_caches:
            raise ValueError("online-target checkpoints must not receive oracle caches")
        target_caches = [None] * len(data_paths)
    base_datasets = []
    datasets: list[Dataset] = []
    for source_index, (path, cache, target_cache) in enumerate(
        zip(data_paths, caches, target_caches, strict=True)
    ):
        dataset = LeWMSequenceDataset(
            path,
            history_size=history_size,
            rollout_steps=rollout_steps,
            frame_skip=frame_skip,
            source_index=source_index,
            correction_cache=cache,
            oracle_target_cache=target_cache,
            require_pre_action_alignment=True,
        )
        base_datasets.append(dataset)
        datasets.append(select_checkpoint_split(dataset, checkpoint, path, split))
    reference = base_datasets[0]
    dimensions = (reference.obs_dim, reference.num_keypoints, reference.transition_action_dim)
    for dataset in base_datasets[1:]:
        current = (dataset.obs_dim, dataset.num_keypoints, dataset.transition_action_dim)
        if current != dimensions:
            raise ValueError(
                f"all sources must share obs/keypoint/action dimensions; "
                f"expected {dimensions}, got {current} for {dataset.path}"
            )
    if target_mode == "frozen_oracle":
        data_config = _nested_mapping(checkpoint, "data_config")
        target_contract = data_config.get("target_contract")
        if not isinstance(target_contract, Mapping):
            raise ValueError("checkpoint is missing the frozen oracle target contract")
        expected_encoder_hash = target_contract.get("encoder_checkpoint_sha256")
        for dataset in base_datasets:
            actual_hash = dataset.oracle_target_metadata.get(
                "encoder_checkpoint_sha256"
            )
            if actual_hash != expected_encoder_hash:
                raise ValueError(
                    "evaluation oracle target encoder differs from checkpoint contract"
                )
    return datasets


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    anchor_override_requested = args.anchor_visual_mode_override != "checkpoint"
    if anchor_override_requested and args.expected_checkpoint_sha256 is None:
        raise ValueError(
            "anchor-visual overrides require --expected-checkpoint-sha256"
        )
    if anchor_override_requested and args.enforce_gate:
        raise ValueError(
            "anchor-visual development overrides cannot use --enforce-gate"
        )
    if anchor_override_requested and args.allow_unfrozen_checkpoint:
        raise ValueError(
            "anchor-visual development overrides require frozen provenance"
        )
    gate_thresholds = {
        "R1_max": args.r1_max,
        "R3_max": args.r3_max,
        "R5_max": args.r5_max,
        "S1_min": args.s1_min,
    }
    if not all(math.isfinite(value) for value in gate_thresholds.values()):
        raise ValueError("gate thresholds must be finite")
    expected_checkpoint_sha256 = (
        _normalize_sha256(
            args.expected_checkpoint_sha256,
            "--expected-checkpoint-sha256",
        )
        if args.expected_checkpoint_sha256 is not None
        else None
    )
    initial_file_snapshots: FileSnapshots = {}
    checkpoint_manifest = _file_manifest(args.checkpoint, initial_file_snapshots)
    checkpoint_sha256 = checkpoint_manifest["sha256"]
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError(
            "checkpoint SHA-256 mismatch before torch.load: "
            f"expected {expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )
    metadata = _load_checkpoint_metadata(args.checkpoint)
    anchor_visual_ablation = resolve_anchor_visual_ablation_contract(
        metadata,
        args.checkpoint,
        args.anchor_visual_mode_override,
        split=args.split,
        grounding_mode_request=args.grounding_mode,
    )
    grounding_mode = resolve_grounding_mode(metadata, args.grounding_mode)
    if grounding_mode == "fcpg":
        validate_checkpoint_correction_caches(
            metadata, args.data, args.correction_cache
        )
    world_config = _nested_mapping(metadata, "model_config").get("world_model")
    if not isinstance(world_config, Mapping):
        raise ValueError("checkpoint is missing model_config.world_model")
    target_mode = str(world_config.get("target_mode", "online"))
    if grounding_mode != "fcpg" and args.correction_cache:
        raise ValueError("nominal evaluation must not receive --correction-cache")
    if target_mode != "frozen_oracle" and args.oracle_target_cache:
        raise ValueError("online-target checkpoints must not receive oracle caches")

    check_git = not bool(anchor_visual_ablation["active"])
    input_snapshot = _evaluation_input_snapshot(
        args.checkpoint,
        metadata,
        args.data,
        args.correction_cache,
        args.oracle_target_cache,
        check_git=check_git,
        file_snapshots=initial_file_snapshots,
    )
    input_files = input_snapshot["files"]
    assert isinstance(input_files, Mapping)
    manifest_snapshot = input_snapshot.get("manifest")
    provenance_report = validate_checkpoint_provenance(
        metadata,
        args.checkpoint,
        allow_unfrozen=args.allow_unfrozen_checkpoint,
        check_git=check_git,
        file_snapshots=input_files,
        git_state=(
            manifest_snapshot.get("git_state")
            if isinstance(manifest_snapshot, Mapping)
            and isinstance(manifest_snapshot.get("git_state"), Mapping)
            else None
        ),
        git_error=(
            str(manifest_snapshot["git_error"])
            if isinstance(manifest_snapshot, Mapping)
            and manifest_snapshot.get("git_error") is not None
            else None
        ),
    )
    dataset_manifests = input_snapshot["datasets"]
    cache_snapshot = input_snapshot["caches"]
    assert isinstance(cache_snapshot, Mapping)
    correction_cache_manifests = cache_snapshot["correction"]
    oracle_target_manifests = cache_snapshot["oracle_target"]
    execution_code = input_snapshot["execution_code"]
    assert isinstance(oracle_target_manifests, Sequence)
    if target_mode == "frozen_oracle":
        validate_checkpoint_oracle_target_caches(
            metadata,
            args.data,
            args.oracle_target_cache,
            manifests=oracle_target_manifests,
        )
    if bool(anchor_visual_ablation["active"]):
        validate_anchor_frozen_target_hash_contract(
            metadata,
            args.data,
            oracle_target_manifests,
            anchor_visual_ablation.get("target_sha256"),
        )
    history_size, rollout_steps, frame_skip = resolve_sequence_settings(
        metadata,
        rollout_steps=args.rollout_steps,
        history_size=args.history_size,
        frame_skip=args.frame_skip,
    )
    datasets = _build_datasets(
        args.data,
        args.correction_cache,
        args.oracle_target_cache,
        history_size=history_size,
        rollout_steps=rollout_steps,
        frame_skip=frame_skip,
        checkpoint=metadata,
        split=args.split,
        grounding_mode=grounding_mode,
    )
    device = torch.device(args.device)
    # Kept local so metric/CLI unit tests do not construct the ViT runtime.
    from fcpg.lewm_runtime import load_lewm_grounded_checkpoint

    model, resolved_config, checkpoint = load_lewm_grounded_checkpoint(
        args.checkpoint,
        device=device,
        dataset=datasets[0],
    )
    anchor_visual_ablation = apply_anchor_visual_mode_override(
        model,
        checkpoint,
        anchor_visual_ablation,
    )
    oracle_token_spec = resolve_oracle_token_spec(
        oracle_target_manifests,
        required=bool(anchor_visual_ablation["active"]),
    )
    paired_dataset = SameTaskActionShuffleDataset(datasets)
    loader = DataLoader(
        paired_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    metrics = evaluate_batches(
        model,
        loader,
        device=device,
        rollout_steps=rollout_steps,
        max_batches=args.max_batches,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        evaluation_split=args.split,
        gate_thresholds=gate_thresholds,
        expected_sequences=len(paired_dataset),
        oracle_token_spec=oracle_token_spec,
    )
    final_input_snapshot = _evaluation_input_snapshot(
        args.checkpoint,
        metadata,
        args.data,
        args.correction_cache,
        args.oracle_target_cache,
        check_git=check_git,
    )
    final_input_files = final_input_snapshot["files"]
    assert isinstance(final_input_files, Mapping)
    final_manifest_snapshot = final_input_snapshot.get("manifest")
    final_provenance_report = validate_checkpoint_provenance(
        metadata,
        args.checkpoint,
        allow_unfrozen=args.allow_unfrozen_checkpoint,
        check_git=check_git,
        file_snapshots=final_input_files,
        git_state=(
            final_manifest_snapshot.get("git_state")
            if isinstance(final_manifest_snapshot, Mapping)
            and isinstance(final_manifest_snapshot.get("git_state"), Mapping)
            else None
        ),
        git_error=(
            str(final_manifest_snapshot["git_error"])
            if isinstance(final_manifest_snapshot, Mapping)
            and final_manifest_snapshot.get("git_error") is not None
            else None
        ),
    )
    _require_unchanged(
        "checkpoint provenance manifest",
        provenance_report,
        final_provenance_report,
    )
    final_cache_snapshot = final_input_snapshot["caches"]
    assert isinstance(final_cache_snapshot, Mapping)
    final_correction_cache_manifests = final_cache_snapshot["correction"]
    if grounding_mode == "fcpg":
        validate_checkpoint_correction_caches(
            metadata, args.data, args.correction_cache
        )
    final_oracle_target_manifests = final_cache_snapshot["oracle_target"]
    assert isinstance(final_oracle_target_manifests, Sequence)
    if target_mode == "frozen_oracle":
        validate_checkpoint_oracle_target_caches(
            metadata,
            args.data,
            args.oracle_target_cache,
            manifests=final_oracle_target_manifests,
        )
    if bool(anchor_visual_ablation["active"]):
        validate_anchor_frozen_target_hash_contract(
            metadata,
            args.data,
            final_oracle_target_manifests,
            anchor_visual_ablation.get("target_sha256"),
        )
    _require_input_snapshot_unchanged(input_snapshot, final_input_snapshot)
    report = {
        "schema_version": 1,
        "protocol": "grounded_lewm_offline_diagnostics",
        "grounding_mode": grounding_mode,
        "grounding_mode_request": args.grounding_mode,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "upstream_commit": checkpoint.get("upstream_commit"),
        "provenance": provenance_report,
        "execution_code": execution_code,
        "anchor_visual_ablation": anchor_visual_ablation,
        "data": [str(Path(path).resolve()) for path in args.data],
        "data_manifest": dataset_manifests,
        "correction_cache": correction_cache_manifests,
        "oracle_target_cache": oracle_target_manifests,
        "evaluation": {
            "split": args.split,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_batches": args.max_batches,
            "device": str(device),
            "evaluator_sha256": _snapshot_digest(__file__, input_files),
            "source_windows": [len(dataset) for dataset in datasets],
            "eligible_windows": len(paired_dataset),
            "action_shuffle_pairing": {
                "same_source": True,
                "same_task": True,
                "different_episode": True,
                "coverage": metrics["action_shuffle_coverage"],
            },
        },
        "sequence": {
            "history_size": history_size,
            "rollout_steps": rollout_steps,
            "frame_skip": frame_skip,
            "endpoint_horizons": list(ENDPOINT_HORIZONS),
            "shared_eligible_windows": True,
        },
        "resolved_config": resolved_config,
        "metrics": metrics,
        "per_task": metrics["per_task"],
        "macro": metrics["macro"],
        "bootstrap": metrics["bootstrap"],
        "gate": metrics["gate"],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        output = Path(args.output)

        def revalidate_publish_inputs() -> None:
            for input_path in sorted(input_files):
                _snapshot_digest(input_path, input_files)

        _publish_text_exclusive_atomic(
            output,
            rendered + "\n",
            revalidate=revalidate_publish_inputs,
        )
    print(rendered)
    return gate_exit_code(metrics["gate"], args.enforce_gate)


if __name__ == "__main__":
    raise SystemExit(main())
