"""Standalone training runtime for the grounded LeWM world model.

The runtime intentionally trains only the upstream-style predictive objective:
next-embedding MSE plus optional SIGReg.  It does not import or mix any behavior
cloning policy loss.
"""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from fcpg.lewm_grounded_model import (
    LEWM_UPSTREAM_COMMIT,
    FCPGViTDenseEncoder,
    GroundedFrameEncoder,
    GroundedLeWM,
    LeWMViTDenseEncoder,
)
from fcpg.lewm_anchor_persistent import (
    ANCHOR_PERSISTENT_CHECKPOINT_TYPE,
    ANCHOR_PERSISTENT_VARIANT,
    AnchorPersistentLeWM,
)
from fcpg.lewm_action_branch_a4 import (
    SIGREG_WEIGHT as A4_SIGREG_WEIGHT,
    TASKS as A4_TASK_NAMES,
)
from fcpg.lewm_provenance import load_yaml_config, sha256_file, validate_manifest
from fcpg.lewm_sequence_data import LeWMSequenceDataset
from fcpg.lewm_task_context_persistent import (
    TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE,
    TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT,
    TASK_CONTEXT_PERSISTENT_VARIANT,
    TaskContextPersistentLeWM,
)


CHECKPOINT_TYPE = "fcpg_grounded_lewm_v1"
TASK_CONTEXT_PERSISTENT_NUM_TASKS = 7
TASK_CONTEXT_PERSISTENT_TASK_NAMES = tuple(A4_TASK_NAMES)

if (
    len(TASK_CONTEXT_PERSISTENT_TASK_NAMES) != TASK_CONTEXT_PERSISTENT_NUM_TASKS
    or len(set(TASK_CONTEXT_PERSISTENT_TASK_NAMES))
    != TASK_CONTEXT_PERSISTENT_NUM_TASKS
):
    raise RuntimeError(
        "the frozen A4 task panel must contain exactly seven distinct tasks"
    )
if (
    A4_SIGREG_WEIGHT != TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT
    or TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT != 0.02
):
    raise RuntimeError("A4 task-context runtime requires SIGReg exactly 0.02")

DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "seed": 3072,
    "output_dir": "outputs/lewm_grounded",
    "provenance_manifest": None,
    "data": {
        "sources": [],
        "history_size": 3,
        "rollout_steps": 1,
        "frame_skip": 5,
        "grounding_mode": "fcpg",
        "target_mode": "online",
        "split_seed": 3072,
        "split_seed_stride": 1009,
        "episode_manifest": None,
        "require_corrector_split_match": False,
        "normalize_inputs": True,
        "val_fraction": 0.1,
        "balanced_sources": True,
        "require_pre_action_alignment": True,
    },
    "loader": {
        "batch_size": 128,
        "num_workers": 6,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 3,
    },
    "model": {
        "vision": {
            "size": "tiny",
            "patch_size": 14,
            "image_size": 224,
            "pretrained": False,
            "use_mask_token": False,
        },
        "frame": {
            "latent_dim": 192,
            "depth": 2,
            "heads": 6,
            "mlp_ratio": 4,
            "dropout": 0.1,
        },
        "predictor": {
            "depth": 6,
            "heads": 16,
            "mlp_dim": 2048,
            "dim_head": 64,
            "dropout": 0.1,
            "projector_hidden_dim": 2048,
            "action_smoothing_dim": 10,
        },
        "sigreg": {"knots": 17, "num_proj": 1024},
    },
    "loss": {
        "sigreg_weight": 0.09,
        "per_source_sigreg": False,
        "alignment_weight": 0.0,
    },
    "optimizer": {
        "lr": 5e-5,
        "weight_decay": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "warmup_steps": 300,
    },
    "trainer": {
        "epochs": 100,
        "max_steps": 0,
        "max_train_batches": 0,
        "max_val_batches": 0,
        "grad_clip": 1.0,
        "amp": True,
        "amp_dtype": "bfloat16",
        "checkpoint_every_epochs": 1,
        "keep_epoch_checkpoints": True,
        "require_input_manifest": False,
        "resume": None,
    },
}


@dataclass(frozen=True)
class EpisodeWindowSplit:
    train_windows: tuple[int, ...]
    val_windows: tuple[int, ...]
    train_episodes: tuple[str, ...]
    val_episodes: tuple[str, ...]


@dataclass
class LeWMDataBundle:
    base_datasets: list[LeWMSequenceDataset]
    train_sources: list[Subset]
    val_sources: list[Subset]
    train_loader: DataLoader
    val_loader: DataLoader
    data_config: dict[str, Any]
    loader_generator: torch.Generator
    val_generator: torch.Generator
    sampler_generator: torch.Generator | None

    @property
    def reference_dataset(self) -> LeWMSequenceDataset:
        return self.base_datasets[0]


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    metrics_path: Path
    global_step: int
    epoch: int
    best_val_pred_mse: float


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _validate_task_context_num_tasks(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be the integer 7")
    result = int(value)
    if result != TASK_CONTEXT_PERSISTENT_NUM_TASKS:
        raise ValueError(
            f"{label} must equal {TASK_CONTEXT_PERSISTENT_NUM_TASKS}, got {result}"
        )
    return result


def _validate_task_context_task_names(
    value: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be the ordered A4 task-name sequence")
    names = tuple(value)
    if any(not isinstance(name, str) for name in names):
        raise ValueError(f"{label} must contain only strings")
    if names != TASK_CONTEXT_PERSISTENT_TASK_NAMES:
        raise ValueError(
            f"{label} must equal the ordered A4 task panel "
            f"{TASK_CONTEXT_PERSISTENT_TASK_NAMES!r}, got {names!r}"
        )
    return names


def _validate_task_context_sigreg(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{label} must equal {TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT}"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must equal {TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT}"
        ) from exc
    if (
        not math.isfinite(result)
        or result != TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT
    ):
        raise ValueError(
            f"{label} must equal {TASK_CONTEXT_PERSISTENT_SIGREG_WEIGHT}, "
            f"got {result}"
        )
    return result


def _resolve_task_context_runtime_contract(resolved: dict[str, Any]) -> None:
    model = resolved["model"]
    if str(model.get("variant", "summary")) != TASK_CONTEXT_PERSISTENT_VARIANT:
        return
    model["num_tasks"] = _validate_task_context_num_tasks(
        model.get("num_tasks", TASK_CONTEXT_PERSISTENT_NUM_TASKS),
        label="model.num_tasks",
    )
    model["task_names"] = list(
        _validate_task_context_task_names(
            model.get("task_names", TASK_CONTEXT_PERSISTENT_TASK_NAMES),
            label="model.task_names",
        )
    )
    _validate_task_context_sigreg(
        resolved["loss"].get("sigreg_weight"),
        label="loss.sigreg_weight",
    )


def _validate_task_context_model_config(
    model_config: Mapping[str, Any],
    *,
    resolved_config: Mapping[str, Any] | None = None,
) -> None:
    if model_config.get("variant") != TASK_CONTEXT_PERSISTENT_VARIANT:
        raise ValueError(
            "task-context checkpoint model_config.variant must be "
            f"{TASK_CONTEXT_PERSISTENT_VARIANT!r}"
        )
    world_model = model_config.get("world_model")
    if not isinstance(world_model, Mapping):
        raise ValueError("task-context model_config.world_model must be a mapping")
    _validate_task_context_num_tasks(
        world_model.get("num_tasks"),
        label="model_config.world_model.num_tasks",
    )
    _validate_task_context_task_names(
        model_config.get("task_names"),
        label="model_config.task_names",
    )
    _validate_task_context_sigreg(
        model_config.get("sigreg_weight"),
        label="model_config.sigreg_weight",
    )
    if resolved_config is None:
        return
    resolved_model = resolved_config.get("model")
    if not isinstance(resolved_model, Mapping):
        raise ValueError("task-context resolved_config.model must be a mapping")
    if resolved_model.get("variant") != TASK_CONTEXT_PERSISTENT_VARIANT:
        raise ValueError(
            "task-context resolved_config.model.variant does not match the checkpoint"
        )
    _validate_task_context_num_tasks(
        resolved_model.get("num_tasks"),
        label="resolved_config.model.num_tasks",
    )
    _validate_task_context_task_names(
        resolved_model.get("task_names"),
        label="resolved_config.model.task_names",
    )
    resolved_loss = resolved_config.get("loss")
    if not isinstance(resolved_loss, Mapping):
        raise ValueError("task-context resolved_config.loss must be a mapping")
    _validate_task_context_sigreg(
        resolved_loss.get("sigreg_weight"),
        label="resolved_config.loss.sigreg_weight",
    )


def resolve_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge runtime defaults and validate the strict LeWM training contract."""

    resolved = _plain(_merge(DEFAULT_RUNTIME_CONFIG, _plain(config)))
    data = resolved["data"]
    if not data.get("sources"):
        raise ValueError("data.sources must contain at least one dataset")
    if int(data["history_size"]) <= 0:
        raise ValueError("data.history_size must be positive")
    if int(data["rollout_steps"]) != 1:
        raise ValueError(
            "faithful H+1 teacher forcing requires data.rollout_steps=1; "
            "recursive prediction belongs in rollout evaluation"
        )
    if int(data["frame_skip"]) <= 0:
        raise ValueError("data.frame_skip must be positive")
    val_fraction = float(data.get("val_fraction", 0.1))
    if "train_fraction" in data:
        val_fraction = 1.0 - float(data["train_fraction"])
        data["val_fraction"] = val_fraction
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("data.val_fraction must be strictly between zero and one")
    grounding_mode = str(data.get("grounding_mode", "fcpg")).lower()
    if grounding_mode not in {"fcpg", "nominal"}:
        raise ValueError("data.grounding_mode must be 'fcpg' or 'nominal'")
    data["grounding_mode"] = grounding_mode
    target_mode = str(data.get("target_mode", "online")).lower()
    if target_mode not in {"online", "frozen_oracle"}:
        raise ValueError("data.target_mode must be 'online' or 'frozen_oracle'")
    data["target_mode"] = target_mode
    if int(data.get("split_seed_stride", 0)) < 0:
        raise ValueError("data.split_seed_stride must be non-negative")
    if int(resolved["model"]["predictor"].get("action_smoothing_dim", 10)) <= 0:
        raise ValueError("model.predictor.action_smoothing_dim must be positive")
    if int(resolved["loader"]["batch_size"]) < 2:
        raise ValueError("loader.batch_size must be at least 2 for BatchNorm/SIGReg")
    if float(resolved["loss"]["sigreg_weight"]) < 0.0:
        raise ValueError("loss.sigreg_weight must be non-negative")
    alignment_weight = float(resolved["loss"].get("alignment_weight", 0.0))
    if alignment_weight < 0.0:
        raise ValueError("loss.alignment_weight must be non-negative")
    if target_mode == "frozen_oracle" and alignment_weight <= 0.0:
        raise ValueError(
            "data.target_mode='frozen_oracle' requires loss.alignment_weight > 0"
        )
    if target_mode == "online" and alignment_weight != 0.0:
        raise ValueError("loss.alignment_weight must be zero for online targets")
    if int(resolved["trainer"]["epochs"]) <= 0:
        raise ValueError("trainer.epochs must be positive")
    if int(resolved["trainer"].get("checkpoint_every_epochs", 1)) != 1:
        raise ValueError(
            "trainer.checkpoint_every_epochs must be 1 for exact epoch-boundary resume"
        )
    _resolve_task_context_runtime_contract(resolved)
    return resolved


def _load_training_provenance(
    resolved: Mapping[str, Any],
    *,
    run_kind: str,
) -> dict[str, Any] | None:
    trainer = resolved["trainer"]
    manifest_value = resolved.get("provenance_manifest")
    required = bool(trainer.get("require_input_manifest", False))
    if run_kind != "formal":
        return None
    if not manifest_value:
        if required:
            raise ValueError(
                "formal training requires provenance_manifest when "
                "trainer.require_input_manifest=true"
            )
        return None

    manifest_path = Path(str(manifest_value)).expanduser().resolve()
    validate_manifest(manifest_path, check_git=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("provenance manifest root must be a JSON object")
    config_record = manifest.get("config")
    if not isinstance(config_record, Mapping):
        raise ValueError("provenance manifest is missing its config record")
    config_path_value = config_record.get("absolute_path")
    if not isinstance(config_path_value, str):
        raise ValueError("provenance manifest config path is invalid")
    frozen_config = resolve_runtime_config(load_yaml_config(config_path_value))
    if _plain(frozen_config) != _plain(resolved):
        raise ValueError(
            "runtime config differs from the YAML frozen by provenance_manifest; "
            "formal CLI overrides are not allowed"
        )
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "manifest": manifest,
    }


def choose_device(value: str | torch.device = "auto") -> torch.device:
    if isinstance(value, torch.device):
        return value
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _source_entry(entry: Any, source_index: int) -> dict[str, Any]:
    if isinstance(entry, (str, Path)):
        result: dict[str, Any] = {"path": str(entry)}
    elif isinstance(entry, Mapping):
        result = _plain(entry)
    else:
        raise TypeError(f"data source {source_index} must be a path or mapping")
    path = result.get("path", result.get("data"))
    if not path:
        raise ValueError(f"data source {source_index} requires path or data")
    resolved_path = Path(str(path)).expanduser().resolve()
    result["path"] = str(resolved_path)
    result.pop("data", None)
    result["name"] = str(result.get("name") or resolved_path.stem)
    correction = result.get("correction_cache")
    result["correction_cache"] = (
        str(Path(str(correction)).expanduser().resolve()) if correction else None
    )
    oracle_target = result.get("oracle_target_cache")
    result["oracle_target_cache"] = (
        str(Path(str(oracle_target)).expanduser().resolve()) if oracle_target else None
    )
    return result


def split_episode_windows(
    dataset: LeWMSequenceDataset,
    val_fraction: float,
    seed: int,
) -> EpisodeWindowSplit:
    """Split complete eligible episodes, then map their windows to train/val."""

    eligible = sorted({episode_index for episode_index, _ in dataset.windows})
    by_task: dict[str, list[int]] = {}
    labels: dict[int, str] = {}
    for episode_index in eligible:
        raw_index = int(dataset.episodes[episode_index][0])
        task = str(dataset.task_name[raw_index])
        episode_id = str(dataset.episode_id[raw_index])
        label = f"{task}:{episode_id}"
        labels[episode_index] = label
        by_task.setdefault(task, []).append(episode_index)

    rng = np.random.default_rng(int(seed))
    val_episode_indices: set[int] = set()
    for task in sorted(by_task):
        episodes = sorted(by_task[task], key=lambda item: labels[item])
        if len(episodes) < 2:
            raise ValueError(
                f"task {task!r} has only {len(episodes)} eligible episode(s); "
                "an episode-level train/val split requires at least two"
            )
        permutation = rng.permutation(len(episodes))
        val_count = max(1, int(round(len(episodes) * float(val_fraction))))
        val_count = min(val_count, len(episodes) - 1)
        val_episode_indices.update(episodes[int(index)] for index in permutation[:val_count])

    train_windows: list[int] = []
    val_windows: list[int] = []
    for window_index, (episode_index, _) in enumerate(dataset.windows):
        target = val_windows if episode_index in val_episode_indices else train_windows
        target.append(window_index)
    if not train_windows or not val_windows:
        raise ValueError("episode split produced an empty train or validation window set")
    train_episode_indices = set(eligible).difference(val_episode_indices)
    return EpisodeWindowSplit(
        train_windows=tuple(train_windows),
        val_windows=tuple(val_windows),
        train_episodes=tuple(sorted(labels[index] for index in train_episode_indices)),
        val_episodes=tuple(sorted(labels[index] for index in val_episode_indices)),
    )


def _load_episode_manifest(value: object) -> tuple[Path, dict[str, Any]] | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("episode manifest root must be a JSON object")
    if payload.get("protocol") != "lewm_gate_l1_camera_aware_episode_split":
        raise ValueError("unexpected episode manifest protocol")
    if not isinstance(payload.get("sources"), Mapping):
        raise ValueError("episode manifest sources must be an object")
    return path, payload


def split_manifest_episode_windows(
    dataset: LeWMSequenceDataset,
    source_name: str,
    payload: Mapping[str, Any],
) -> EpisodeWindowSplit:
    """Select exact train/validation episodes from a frozen camera manifest."""

    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or source_name not in sources:
        raise ValueError(f"episode manifest has no source named {source_name!r}")
    source = sources[source_name]
    if not isinstance(source, Mapping):
        raise TypeError(f"episode manifest source {source_name!r} must be an object")
    manifest_path = Path(str(source.get("path", ""))).expanduser().resolve()
    if manifest_path != dataset.path.expanduser().resolve():
        raise ValueError(
            f"episode manifest source path {manifest_path} differs from {dataset.path}"
        )
    expected_data_sha256 = source.get("data_sha256")
    if not isinstance(expected_data_sha256, str):
        raise ValueError(
            f"episode manifest source {source_name!r} has no data_sha256"
        )
    actual_data_sha256 = sha256_file(dataset.path)
    if actual_data_sha256 != expected_data_sha256:
        raise ValueError(
            f"episode manifest source hash mismatch for {source_name!r}: "
            f"expected {expected_data_sha256}, got {actual_data_sha256}"
        )
    train_labels = source.get("train_episodes")
    val_labels = source.get("val_episodes")
    excluded_labels = source.get("excluded_episodes")
    if (
        not isinstance(train_labels, Sequence)
        or isinstance(train_labels, (str, bytes))
        or not isinstance(val_labels, Sequence)
        or isinstance(val_labels, (str, bytes))
        or not isinstance(excluded_labels, Sequence)
        or isinstance(excluded_labels, (str, bytes))
    ):
        raise ValueError(
            "episode manifest train/val/excluded episodes must be arrays"
        )
    train_expected = {str(label) for label in train_labels}
    val_expected = {str(label) for label in val_labels}
    excluded_expected = {str(label) for label in excluded_labels}
    if not train_expected or not val_expected:
        raise ValueError("episode manifest train/val sets must be non-empty")
    overlap = (
        (train_expected & val_expected)
        | (train_expected & excluded_expected)
        | (val_expected & excluded_expected)
    )
    if overlap:
        raise ValueError(
            "episode manifest train/val/excluded overlap: "
            f"{sorted(overlap)[:5]}"
        )

    labels = {
        episode_index: (
            f"{dataset.task_name[int(sequence[0])]}:"
            f"{dataset.episode_id[int(sequence[0])]}"
        )
        for episode_index, sequence in enumerate(dataset.episodes)
    }
    train_windows = tuple(
        index
        for index, (episode_index, _) in enumerate(dataset.windows)
        if labels[episode_index] in train_expected
    )
    val_windows = tuple(
        index
        for index, (episode_index, _) in enumerate(dataset.windows)
        if labels[episode_index] in val_expected
    )
    observed = set(labels.values())
    declared = train_expected | val_expected | excluded_expected
    missing = declared - observed
    if missing:
        raise ValueError(
            f"episode manifest references missing episodes: {sorted(missing)[:5]}"
        )
    undeclared = observed - declared
    if undeclared:
        raise ValueError(
            f"episode manifest omits observed episodes: {sorted(undeclared)[:5]}"
        )
    if not train_windows or not val_windows:
        raise ValueError("episode manifest selects no train or validation windows")
    return EpisodeWindowSplit(
        train_windows=train_windows,
        val_windows=val_windows,
        train_episodes=tuple(sorted(train_expected)),
        val_episodes=tuple(sorted(val_expected)),
    )


def _balanced_sampler(
    datasets: Sequence[Subset],
    generator: torch.Generator,
) -> WeightedRandomSampler:
    lengths = np.asarray([len(dataset) for dataset in datasets], dtype=np.int64)
    if bool((lengths <= 0).any()):
        raise ValueError("balanced source sampling requires non-empty sources")
    weights = np.concatenate(
        [np.full(length, 1.0 / float(length), dtype=np.float64) for length in lengths]
    )
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=int(lengths.sum()),
        replacement=True,
        generator=generator,
    )


def _loader_kwargs(loader_config: Mapping[str, Any]) -> dict[str, Any]:
    workers = int(loader_config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(loader_config["batch_size"]),
        "num_workers": workers,
        "pin_memory": bool(loader_config.get("pin_memory", False)),
        "persistent_workers": bool(loader_config.get("persistent_workers", False))
        and workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(loader_config.get("prefetch_factor", 2))
    return kwargs


def _validate_corrector_split(
    dataset: LeWMSequenceDataset,
    *,
    expected_val_fraction: float,
    expected_split_seed: int,
) -> dict[str, Any]:
    metadata = dataset.correction_metadata
    checkpoint_value = metadata.get("checkpoint")
    if not checkpoint_value:
        raise ValueError(
            f"FCPG cache for {dataset.path} does not identify its corrector checkpoint"
        )
    checkpoint_path = Path(str(checkpoint_value)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"corrector checkpoint is missing: {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    args = checkpoint.get("args", {})
    if not isinstance(args, Mapping):
        raise ValueError("corrector checkpoint args must be a mapping")
    split_unit = str(args.get("split_unit", ""))
    split_seed = int(args.get("split_seed", -1))
    val_fraction = float(args.get("val_fraction", -1.0))
    if split_unit != "task-episode":
        raise ValueError(f"corrector split_unit must be task-episode, got {split_unit!r}")
    if split_seed != int(expected_split_seed):
        raise ValueError(
            f"corrector split_seed={split_seed} differs from LeWM split_seed="
            f"{expected_split_seed} for {dataset.path}"
        )
    if not math.isclose(val_fraction, float(expected_val_fraction), abs_tol=1e-12):
        raise ValueError(
            f"corrector val_fraction={val_fraction} differs from LeWM val_fraction="
            f"{expected_val_fraction} for {dataset.path}"
        )
    source_manifest = checkpoint.get("source_manifest", [])
    if not any(
        isinstance(source, Mapping)
        and source.get("data")
        and Path(str(source["data"])).expanduser().resolve() == dataset.path.resolve()
        for source in source_manifest
    ):
        raise ValueError(f"corrector checkpoint does not contain source {dataset.path}")
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "corrector_arch": checkpoint.get("corrector_arch"),
        "obs_mode": checkpoint.get("obs_mode", args.get("obs_mode")),
        "split_unit": split_unit,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
    }


def _compute_training_normalization(
    train_sources: Sequence[Subset],
    *,
    enabled: bool,
) -> dict[str, Any]:
    reference = train_sources[0].dataset
    proprio_dim = int(reference.proprio_dim)
    action_dim = int(reference.transition_action_dim)
    if not enabled:
        return {
            "enabled": False,
            "proprio_mean": [0.0] * proprio_dim,
            "proprio_std": [1.0] * proprio_dim,
            "action_mean": [0.0] * action_dim,
            "action_std": [1.0] * action_dim,
        }
    proprio_sum = np.zeros(proprio_dim, dtype=np.float64)
    proprio_square = np.zeros(proprio_dim, dtype=np.float64)
    action_sum = np.zeros(action_dim, dtype=np.float64)
    action_square = np.zeros(action_dim, dtype=np.float64)
    proprio_count = 0
    action_count = 0
    for subset in train_sources:
        dataset = subset.dataset
        for window_index in subset.indices:
            episode_index, start = dataset.windows[int(window_index)]
            sequence = dataset.episodes[episode_index]
            state_positions = start + np.arange(dataset.num_state_frames) * dataset.frame_skip
            state_indices = sequence[state_positions]
            proprio = dataset.obs[state_indices, :proprio_dim].astype(np.float64, copy=False)
            blocks = []
            for position in state_positions[:-1]:
                raw_indices = sequence[position : position + dataset.frame_skip]
                blocks.append(dataset.action[raw_indices].reshape(-1))
            action = np.stack(blocks).astype(np.float64, copy=False)
            proprio_sum += proprio.sum(axis=0)
            proprio_square += np.square(proprio).sum(axis=0)
            action_sum += action.sum(axis=0)
            action_square += np.square(action).sum(axis=0)
            proprio_count += int(proprio.shape[0])
            action_count += int(action.shape[0])
    if proprio_count == 0 or action_count == 0:
        raise ValueError("cannot compute LeWM normalization from an empty training split")
    proprio_mean = proprio_sum / proprio_count
    action_mean = action_sum / action_count
    proprio_var = np.maximum(proprio_square / proprio_count - np.square(proprio_mean), 0.0)
    action_var = np.maximum(action_square / action_count - np.square(action_mean), 0.0)
    proprio_std = np.sqrt(proprio_var)
    action_std = np.sqrt(action_var)
    proprio_std[proprio_std < 1e-6] = 1.0
    action_std[action_std < 1e-6] = 1.0
    return {
        "enabled": True,
        "proprio_mean": proprio_mean.tolist(),
        "proprio_std": proprio_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
    }


def build_lewm_data(config: Mapping[str, Any]) -> LeWMDataBundle:
    resolved = resolve_runtime_config(config)
    data = resolved["data"]
    seed = int(resolved["seed"])
    grounding_mode = str(data["grounding_mode"])
    target_mode = str(data["target_mode"])
    split_seed = int(data.get("split_seed", seed))
    split_seed_stride = int(data.get("split_seed_stride", 0))
    require_corrector_split = bool(data.get("require_corrector_split_match", False))
    normalize_inputs = bool(data.get("normalize_inputs", True))
    source_entries = [
        _source_entry(entry, index) for index, entry in enumerate(data["sources"])
    ]
    if len({entry["name"] for entry in source_entries}) != len(source_entries):
        raise ValueError("data source names must be unique")
    episode_manifest = _load_episode_manifest(data.get("episode_manifest"))
    if episode_manifest is not None and require_corrector_split:
        raise ValueError(
            "camera-aware episode_manifest is incompatible with "
            "require_corrector_split_match"
        )

    bases: list[LeWMSequenceDataset] = []
    train_sources: list[Subset] = []
    val_sources: list[Subset] = []
    manifests: list[dict[str, Any]] = []
    for source_index, entry in enumerate(source_entries):
        configured_cache = entry.get("correction_cache")
        if grounding_mode == "fcpg" and not configured_cache:
            raise ValueError(
                f"FCPG grounding requires correction_cache for source {entry['name']}"
            )
        correction_cache = configured_cache if grounding_mode == "fcpg" else None
        oracle_target_cache = entry.get("oracle_target_cache")
        if target_mode == "frozen_oracle" and not oracle_target_cache:
            raise ValueError(
                f"frozen oracle targets require oracle_target_cache for source "
                f"{entry['name']}"
            )
        if target_mode == "online" and oracle_target_cache:
            raise ValueError(
                "oracle_target_cache is only allowed when data.target_mode="
                "'frozen_oracle'"
            )
        source_split_seed = split_seed + source_index * split_seed_stride
        dataset = LeWMSequenceDataset(
            data_path=entry["path"],
            history_size=int(data["history_size"]),
            rollout_steps=1,
            frame_skip=int(data["frame_skip"]),
            source_index=source_index,
            source_name=entry["name"],
            correction_cache=correction_cache,
            oracle_target_cache=oracle_target_cache,
            require_pre_action_alignment=bool(data["require_pre_action_alignment"]),
        )
        split = (
            split_manifest_episode_windows(
                dataset,
                entry["name"],
                episode_manifest[1],
            )
            if episode_manifest is not None
            else split_episode_windows(
                dataset,
                val_fraction=float(data["val_fraction"]),
                seed=source_split_seed,
            )
        )
        corrector_split = None
        if grounding_mode == "fcpg" and require_corrector_split:
            corrector_split = _validate_corrector_split(
                dataset,
                expected_val_fraction=float(data["val_fraction"]),
                expected_split_seed=source_split_seed,
            )
        train_subset = Subset(dataset, list(split.train_windows))
        val_subset = Subset(dataset, list(split.val_windows))
        bases.append(dataset)
        train_sources.append(train_subset)
        val_sources.append(val_subset)
        manifests.append(
            {
                **entry,
                "source_index": source_index,
                "correction_cache": (
                    str(dataset.correction_cache_path) if dataset.correction_cache_path else None
                ),
                "corrector_split": corrector_split,
                "oracle_target_cache": (
                    str(Path(str(oracle_target_cache)).resolve())
                    if oracle_target_cache
                    else None
                ),
                "oracle_target_array": (
                    str(dataset.oracle_target_cache_path)
                    if dataset.oracle_target_cache_path
                    else None
                ),
                "oracle_target_metadata_sha256": (
                    sha256_file(Path(str(oracle_target_cache)) / "metadata.json")
                    if oracle_target_cache
                    else None
                ),
                "oracle_target_sha256": dataset.oracle_target_metadata.get(
                    "oracle_target_sha256"
                ),
                "oracle_target_encoder_sha256": dataset.oracle_target_metadata.get(
                    "encoder_checkpoint_sha256"
                ),
                "train_windows": len(train_subset),
                "val_windows": len(val_subset),
                "train_episodes": list(split.train_episodes),
                "val_episodes": list(split.val_episodes),
            }
        )

    dimension_names = (
        "proprio_dim",
        "transition_action_dim",
        "num_keypoints",
        "image_size",
    )
    for name in dimension_names:
        values = [getattr(dataset, name) for dataset in bases]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"LeWM data sources disagree on {name}: {values}")
    target_contract: dict[str, Any] = {"mode": target_mode}
    if target_mode == "frozen_oracle":
        target_dims = [dataset.oracle_target_dim for dataset in bases]
        if any(value is None for value in target_dims):
            raise ValueError("frozen oracle mode requires targets for every source")
        if len(set(target_dims)) != 1:
            raise ValueError(f"oracle target dimensions differ across sources: {target_dims}")
        expected_dim = int(resolved["model"]["frame"]["latent_dim"])
        if int(target_dims[0]) != expected_dim:
            raise ValueError(
                f"oracle target dim {target_dims[0]} must equal latent_dim {expected_dim}"
            )
        encoder_hashes = {
            str(dataset.oracle_target_metadata.get("encoder_checkpoint_sha256"))
            for dataset in bases
        }
        encoder_specs = {
            json.dumps(
                dataset.oracle_target_metadata.get("encoder_spec"),
                sort_keys=True,
            )
            for dataset in bases
        }
        if len(encoder_hashes) != 1 or len(encoder_specs) != 1:
            raise ValueError("all sources must share one frozen oracle target encoder")
        reference_metadata = bases[0].oracle_target_metadata
        target_contract.update(
            {
                "target_dim": int(target_dims[0]),
                "encoder_checkpoint": reference_metadata["encoder_checkpoint"],
                "encoder_checkpoint_sha256": next(iter(encoder_hashes)),
                "encoder_spec": reference_metadata["encoder_spec"],
                "q_true_usage": "target_construction_only",
            }
        )
    normalization = _compute_training_normalization(train_sources, enabled=normalize_inputs)

    train_dataset = ConcatDataset(train_sources)
    val_dataset = ConcatDataset(val_sources)
    loader_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 104729)
    sampler_generator: torch.Generator | None = None
    sampler = None
    shuffle = True
    if len(train_sources) > 1 and bool(data.get("balanced_sources", True)):
        sampler_generator = torch.Generator().manual_seed(seed + 7919)
        sampler = _balanced_sampler(train_sources, sampler_generator)
        shuffle = False
    kwargs = _loader_kwargs(resolved["loader"])
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=True,
        generator=loader_generator,
        **kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        drop_last=False,
        generator=val_generator,
        **kwargs,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "training loader has no full batch; reduce loader.batch_size or add data"
        )
    reference = bases[0]
    data_config = {
        "sources": manifests,
        "grounding_mode": grounding_mode,
        "target_mode": target_mode,
        "target_contract": target_contract,
        "episode_manifest": (
            {
                "path": str(episode_manifest[0]),
                "sha256": sha256_file(episode_manifest[0]),
                "protocol": episode_manifest[1]["protocol"],
                "training_camera_limits": episode_manifest[1].get(
                    "training_camera_limits"
                ),
            }
            if episode_manifest is not None
            else None
        ),
        "val_fraction": float(data["val_fraction"]),
        "split_seed": split_seed,
        "split_seed_stride": split_seed_stride,
        "corrector_split_enforced": require_corrector_split,
        "history_size": reference.history_size,
        "rollout_steps": reference.rollout_steps,
        "frame_skip": reference.frame_skip,
        "num_state_frames": reference.num_state_frames,
        "proprio_dim": reference.proprio_dim,
        "action_dim": reference.transition_action_dim,
        "raw_action_dim": reference.action_dim,
        "num_keypoints": reference.num_keypoints,
        "image_size": list(reference.image_size),
        "balanced_sources": bool(sampler is not None),
        "normalization": {
            "pixels": "uint8_or_[0,1]_then_imagenet_in_LeWMViTDenseEncoder",
            "geometry": "normalized_inside_GroundedFrameEncoder",
            **normalization,
        },
    }
    return LeWMDataBundle(
        base_datasets=bases,
        train_sources=train_sources,
        val_sources=val_sources,
        train_loader=train_loader,
        val_loader=val_loader,
        data_config=data_config,
        loader_generator=loader_generator,
        val_generator=val_generator,
        sampler_generator=sampler_generator,
    )


def _dataset_dimensions(dataset: Any) -> dict[str, int]:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if isinstance(dataset, ConcatDataset):
        dataset = dataset.datasets[0]
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
    required = ("proprio_dim", "transition_action_dim", "num_keypoints", "history_size")
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise TypeError(f"dataset is missing LeWM dimensions: {missing}")
    return {
        "obs_dim": int(dataset.proprio_dim),
        "action_dim": int(dataset.transition_action_dim),
        "num_keypoints": int(dataset.num_keypoints),
        "history_size": int(dataset.history_size),
    }


def resolve_model_config(
    config: Mapping[str, Any],
    dataset: Any,
) -> dict[str, Any]:
    resolved = resolve_runtime_config(config)
    dimensions = _dataset_dimensions(dataset)
    if dimensions["history_size"] != int(resolved["data"]["history_size"]):
        raise ValueError("dataset history_size differs from resolved data.history_size")
    model = resolved["model"]
    vision = _plain(model["vision"])
    frame = _plain(model["frame"])
    predictor = _plain(model["predictor"])
    sigreg = _plain(model["sigreg"])
    target_mode = str(resolved["data"]["target_mode"])
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    if isinstance(base_dataset, ConcatDataset):
        base_dataset = base_dataset.datasets[0]
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
    oracle_target_dim = getattr(base_dataset, "oracle_target_dim", None)
    if target_mode == "frozen_oracle" and oracle_target_dim != int(frame["latent_dim"]):
        raise ValueError(
            "frozen oracle target dimension must match model.frame.latent_dim"
        )
    world_model_config = {
        "action_dim": dimensions["action_dim"],
        "history_size": dimensions["history_size"],
        "predictor_depth": int(predictor["depth"]),
        "predictor_heads": int(predictor["heads"]),
        "action_smoothing_dim": int(predictor.get("action_smoothing_dim", 10)),
        "predictor_mlp_dim": int(predictor["mlp_dim"]),
        "predictor_dim_head": int(predictor["dim_head"]),
        "predictor_dropout": float(predictor["dropout"]),
        "projector_hidden_dim": int(predictor["projector_hidden_dim"]),
        "sigreg_knots": int(sigreg["knots"]),
        "sigreg_num_proj": int(sigreg["num_proj"]),
        "target_mode": target_mode,
        "alignment_weight": float(resolved["loss"]["alignment_weight"]),
    }
    resolved_model_config = {
        "vision": vision,
        "frame": {
            **frame,
            "obs_dim": dimensions["obs_dim"],
            "num_keypoints": dimensions["num_keypoints"],
        },
        "world_model": world_model_config,
    }
    variant = model.get("variant")
    if variant is not None:
        resolved_model_config["variant"] = str(variant)
    if variant == TASK_CONTEXT_PERSISTENT_VARIANT:
        world_model_config["num_tasks"] = _validate_task_context_num_tasks(
            model.get("num_tasks"),
            label="model.num_tasks",
        )
        resolved_model_config["task_names"] = list(
            _validate_task_context_task_names(
                model.get("task_names"),
                label="model.task_names",
            )
        )
        resolved_model_config["sigreg_weight"] = _validate_task_context_sigreg(
            resolved["loss"].get("sigreg_weight"),
            label="loss.sigreg_weight",
        )
        _validate_task_context_model_config(resolved_model_config)
    return resolved_model_config


def _build_from_model_config(
    model_config: Mapping[str, Any],
) -> GroundedLeWM | AnchorPersistentLeWM | TaskContextPersistentLeWM:
    vision_config = dict(model_config["vision"])
    vision_source = str(vision_config.pop("source", "lewm_tiny"))
    if vision_source == "fcpg_vitb":
        vision = FCPGViTDenseEncoder(**vision_config)
    elif vision_source == "lewm_tiny":
        vision = LeWMViTDenseEncoder(**vision_config)
    else:
        raise ValueError(f"unsupported model.vision.source: {vision_source!r}")
    frame_config = dict(model_config["frame"])
    configured_visual_dim = frame_config.pop("visual_dim", None)
    visual_dim = int(getattr(vision, "output_dim"))
    if configured_visual_dim is not None and int(configured_visual_dim) != visual_dim:
        raise ValueError(
            f"configured visual_dim={configured_visual_dim} differs from encoder "
            f"output_dim={visual_dim}"
        )
    frame_encoder = GroundedFrameEncoder(
        vision_encoder=vision,
        visual_dim=visual_dim,
        **frame_config,
    )
    variant = str(model_config.get("variant", "summary"))
    world_model_config = dict(model_config["world_model"])
    if variant == "summary":
        model = GroundedLeWM(
            frame_encoder=frame_encoder,
            **world_model_config,
        )
    elif variant == ANCHOR_PERSISTENT_VARIANT:
        model = AnchorPersistentLeWM(
            frame_encoder=frame_encoder,
            **world_model_config,
        )
    elif variant == TASK_CONTEXT_PERSISTENT_VARIANT:
        _validate_task_context_model_config(model_config)
        model = TaskContextPersistentLeWM(
            frame_encoder=frame_encoder,
            **world_model_config,
        )
    else:
        raise ValueError(f"unsupported grounded LeWM model variant: {variant!r}")
    model.runtime_model_config = _plain(model_config)
    return model


def build_grounded_lewm(
    resolved_config: Mapping[str, Any],
    sample_dataset: Any,
) -> GroundedLeWM | AnchorPersistentLeWM | TaskContextPersistentLeWM:
    """Construct the exact ViT-backed grounded LeWM from data-derived dimensions."""

    model_config = resolve_model_config(resolved_config, sample_dataset)
    return _build_from_model_config(model_config)


def _validate_expected_parameter_count(
    model: nn.Module,
    resolved_config: Mapping[str, Any],
) -> int:
    actual = sum(parameter.numel() for parameter in model.parameters())
    contract = resolved_config.get("contract", {})
    expected = (
        contract.get("expected_parameter_count")
        if isinstance(contract, Mapping)
        else None
    )
    if expected is not None and actual != int(expected):
        raise ValueError(
            "model parameter count violates the preregistered contract: "
            f"expected={int(expected)} actual={actual}"
        )
    return int(actual)


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=device.type == "cuda")
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def _task_names_from_batch(value: object, batch_size: int) -> tuple[str, ...]:
    if isinstance(value, str):
        names = (value,)
    elif isinstance(value, np.ndarray):
        names = tuple(value.tolist())
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        names = tuple(value)
    else:
        raise TypeError("task_name must be a string sequence")
    if len(names) != batch_size or any(not isinstance(name, str) for name in names):
        raise ValueError(f"task_name must contain {batch_size} string labels")
    return names


def _task_index_from_batch(
    model: TaskContextPersistentLeWM,
    batch: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve explicit A4 task identity without selecting a default task."""

    if int(model.num_tasks) != TASK_CONTEXT_PERSISTENT_NUM_TASKS:
        raise ValueError("TaskContextPersistentLeWM runtime requires num_tasks=7")
    if batch_size <= 0:
        raise ValueError("task-context batch must not be empty")
    raw_task_index = batch.get("task_index")
    raw_task_names = batch.get("task_name")
    mapped_task_index: torch.Tensor | None = None
    if raw_task_names is not None:
        task_names = _task_names_from_batch(raw_task_names, batch_size)
        index_by_name = {
            name: index
            for index, name in enumerate(TASK_CONTEXT_PERSISTENT_TASK_NAMES)
        }
        unknown = sorted({name for name in task_names if name not in index_by_name})
        if unknown:
            raise ValueError(
                "task_name contains labels outside the ordered A4 task panel: "
                f"{unknown}"
            )
        mapped_task_index = torch.tensor(
            [index_by_name[name] for name in task_names],
            dtype=torch.long,
            device=device,
        )

    if raw_task_index is None:
        if mapped_task_index is None:
            raise KeyError(
                "TaskContextPersistentLeWM requires task_index or exact A4 task_name"
            )
        task_index = mapped_task_index
    else:
        if not torch.is_tensor(raw_task_index):
            raise TypeError("task_index must be a tensor")
        task_index = model._validated_task_index(
            raw_task_index,
            batch_size,
            device,
        )
        if mapped_task_index is not None and not torch.equal(
            task_index,
            mapped_task_index,
        ):
            raise ValueError(
                "task_index conflicts with the ordered A4 task_name labels"
            )
    return task_index


def _prepare_task_context_batch(
    model: GroundedLeWM | AnchorPersistentLeWM | TaskContextPersistentLeWM,
    batch: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind task identity for model.forward without an implicit default."""

    if not isinstance(model, TaskContextPersistentLeWM):
        return batch
    action = batch.get("action")
    if action is None:
        action = batch.get("actions")
    if not torch.is_tensor(action) or action.ndim < 1:
        raise KeyError("task-context batch requires tensor action/actions")
    task_index = _task_index_from_batch(
        model,
        batch,
        batch_size=int(action.shape[0]),
        device=action.device,
    )

    prepared = dict(batch)
    prepared["task_index"] = task_index
    return prepared


def _predict_with_batch_context(
    model: GroundedLeWM | AnchorPersistentLeWM | TaskContextPersistentLeWM,
    embedding: torch.Tensor,
    action: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    """Dispatch prediction while failing closed when task identity is absent."""

    if isinstance(model, TaskContextPersistentLeWM):
        if embedding.ndim < 1:
            raise ValueError("embedding must include a batch dimension")
        task_index = _task_index_from_batch(
            model,
            batch,
            batch_size=int(embedding.shape[0]),
            device=embedding.device,
        )
        return model.predict(embedding, action, task_index)
    return model.predict(embedding, action)


def cosine_warmup_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _amp_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"unsupported trainer.amp_dtype: {name}")


def _capture_rng(bundle: LeWMDataBundle | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if bundle is not None:
        state["loader_generator"] = bundle.loader_generator.get_state()
        state["val_generator"] = bundle.val_generator.get_state()
        if bundle.sampler_generator is not None:
            state["sampler_generator"] = bundle.sampler_generator.get_state()
    return state


def _restore_rng(state: Mapping[str, Any] | None, bundle: LeWMDataBundle | None = None) -> None:
    if not state:
        return

    def cpu_state(value: Any) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError("serialized RNG state must be a tensor")
        return value.detach().cpu()

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # A checkpoint loaded with map_location=cuda moves every tensor, including
    # CPU generator states. PyTorch generator APIs require CPU ByteTensors.
    torch.set_rng_state(cpu_state(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([cpu_state(value) for value in state["cuda"]])
    if bundle is not None and "loader_generator" in state:
        bundle.loader_generator.set_state(cpu_state(state["loader_generator"]))
    if bundle is not None and "val_generator" in state:
        bundle.val_generator.set_state(cpu_state(state["val_generator"]))
    if (
        bundle is not None
        and bundle.sampler_generator is not None
        and "sampler_generator" in state
    ):
        bundle.sampler_generator.set_state(cpu_state(state["sampler_generator"]))


def _torch_load(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dictionary: {path}")
    return checkpoint


def save_lewm_checkpoint(
    path: str | Path,
    *,
    model: GroundedLeWM | AnchorPersistentLeWM | TaskContextPersistentLeWM,
    resolved_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    step: int,
    epoch: int,
    run_kind: str,
    provenance: Mapping[str, Any] | None = None,
    refuse_existing: bool = False,
    metrics: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    model_config: Mapping[str, Any] | None = None,
    best_val_pred_mse: float | None = None,
    rng_state: Mapping[str, Any] | None = None,
    target_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    """Save a self-describing, resumable grounded LeWM checkpoint atomically."""

    if model_config is None:
        model_config = getattr(model, "runtime_model_config", None)
    if model_config is None:
        raise ValueError("model_config is required for reconstructable checkpoints")
    checkpoint_type = getattr(model, "checkpoint_type", CHECKPOINT_TYPE)
    model_variant = model_config.get("variant", "summary")
    if isinstance(model, TaskContextPersistentLeWM):
        if checkpoint_type != TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE:
            raise ValueError("task-context model has an unexpected checkpoint type")
        if int(model.num_tasks) != TASK_CONTEXT_PERSISTENT_NUM_TASKS:
            raise ValueError("task-context model must contain exactly seven tasks")
        _validate_task_context_model_config(
            model_config,
            resolved_config=resolved_config,
        )
    elif (
        checkpoint_type == TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE
        or model_variant == TASK_CONTEXT_PERSISTENT_VARIANT
    ):
        raise ValueError(
            "task-context checkpoint metadata requires TaskContextPersistentLeWM"
        )
    config_dict = _plain(resolved_config)
    data_dict = _plain(data_config)
    checkpoint = {
        "schema_version": 1,
        "checkpoint_type": checkpoint_type,
        "upstream_commit": LEWM_UPSTREAM_COMMIT,
        "lewm_upstream_commit": LEWM_UPSTREAM_COMMIT,
        "step": int(step),
        "epoch": int(epoch),
        "run_kind": str(run_kind),
        "provenance": _plain(provenance) if provenance is not None else None,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "resolved_config": config_dict,
        "args": config_dict,
        "data_config": data_dict,
        "model_config": _plain(model_config),
        "normalization": data_dict.get("normalization"),
        "metrics": _plain(metrics or {}),
        "best_val_pred_mse": best_val_pred_mse,
        "rng_state": dict(rng_state or _capture_rng()),
    }
    if target_optimizer_steps is not None:
        target_steps = int(target_optimizer_steps)
        if target_steps <= 0 or int(step) > target_steps:
            raise ValueError("invalid GroundedLeWM optimizer-step budget")
        completed = int(step) == target_steps
        checkpoint["training_budget"] = {
            "target_optimizer_steps": target_steps,
            "checkpoint_optimizer_step": int(step),
            "run_completed": completed,
            "final_optimizer_steps": int(step) if completed else None,
        }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(checkpoint, temporary)
        if refuse_existing:
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint


def _replace_with_hardlink(
    source: Path,
    destination: Path,
    *,
    refuse_existing: bool,
) -> None:
    """Publish an immutable link or atomically replace a mutable alias."""

    if refuse_existing:
        try:
            os.link(source, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                "refusing to overwrite immutable checkpoint: " + str(destination)
            ) from exc
        return
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp-link"
    )
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_checkpoint_dataset(
    checkpoint: Mapping[str, Any],
    dataset: Any,
) -> None:
    actual = _dataset_dimensions(dataset)
    expected_world = checkpoint["model_config"]["world_model"]
    expected_frame = checkpoint["model_config"]["frame"]
    expected = {
        "obs_dim": int(expected_frame["obs_dim"]),
        "action_dim": int(expected_world["action_dim"]),
        "num_keypoints": int(expected_frame["num_keypoints"]),
        "history_size": int(expected_world["history_size"]),
    }
    if actual != expected:
        raise ValueError(f"checkpoint/data dimension mismatch: expected {expected}, got {actual}")


def load_lewm_grounded_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    dataset: Any | None = None,
) -> tuple[
    GroundedLeWM | AnchorPersistentLeWM | TaskContextPersistentLeWM,
    dict[str, Any],
    dict[str, Any],
]:
    """Rebuild a model and return ``(model, resolved_config, checkpoint)``."""

    target_device = choose_device(device)
    checkpoint = _torch_load(path, map_location=target_device)
    checkpoint_type = checkpoint.get("checkpoint_type")
    allowed_checkpoint_types = {
        CHECKPOINT_TYPE,
        ANCHOR_PERSISTENT_CHECKPOINT_TYPE,
        TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE,
    }
    if checkpoint_type not in allowed_checkpoint_types:
        raise ValueError(
            f"expected checkpoint_type in {sorted(allowed_checkpoint_types)!r}, got "
            f"{checkpoint_type!r}"
        )
    commit = checkpoint.get("upstream_commit", checkpoint.get("lewm_upstream_commit"))
    if commit != LEWM_UPSTREAM_COMMIT:
        raise ValueError(
            f"checkpoint pins LeWM commit {commit!r}, expected {LEWM_UPSTREAM_COMMIT}"
        )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping")
    model_variant = model_config.get("variant", "summary")
    if checkpoint_type == TASK_CONTEXT_PERSISTENT_CHECKPOINT_TYPE:
        resolved_config = checkpoint.get("resolved_config")
        if not isinstance(resolved_config, Mapping):
            raise ValueError(
                "task-context checkpoint resolved_config must be a mapping"
            )
        _validate_task_context_model_config(
            model_config,
            resolved_config=resolved_config,
        )
    elif model_variant == TASK_CONTEXT_PERSISTENT_VARIANT:
        raise ValueError(
            "task-context model_config requires the task-context checkpoint type"
        )
    if dataset is not None:
        _validate_checkpoint_dataset(checkpoint, dataset)
    model = _build_from_model_config(model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(target_device)
    model.eval()
    return model, dict(checkpoint["resolved_config"]), checkpoint


def _shuffled_actions(
    action: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor | None:
    source_index = batch.get("source_index")
    episode_id = batch.get("episode_id")
    if not torch.is_tensor(source_index) or not torch.is_tensor(episode_id):
        raise KeyError("episode-aware action shuffle requires source_index and episode_id")
    sources = source_index.detach().cpu().reshape(-1).tolist()
    episodes = episode_id.detach().cpu().reshape(-1).tolist()
    if len(sources) != action.shape[0] or len(episodes) != action.shape[0]:
        raise ValueError("action shuffle group metadata must match the batch size")
    keys = [(int(source), int(episode)) for source, episode in zip(sources, episodes)]
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    largest_group = max(len(indices) for indices in groups.values())
    if largest_group * 2 > len(keys):
        return None
    ordered = [index for key in sorted(groups) for index in groups[key]]
    donors = ordered[largest_group:] + ordered[:largest_group]
    permutation = [0] * len(keys)
    for target, donor in zip(ordered, donors):
        if keys[target] == keys[donor]:
            raise RuntimeError("failed to construct a cross-episode action derangement")
        permutation[target] = donor
    indices = torch.tensor(permutation, dtype=torch.long, device=action.device)
    return action.index_select(0, indices)


@torch.no_grad()
def evaluate_lewm(
    model: GroundedLeWM | AnchorPersistentLeWM | TaskContextPersistentLeWM,
    loader: DataLoader,
    device: torch.device,
    *,
    sigreg_weight: float,
    per_source_sigreg: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 0,
) -> dict[str, float]:
    """Report prediction, copy, and deterministic action-shuffle MSE."""

    model.eval()
    totals = {
        "loss": 0.0,
        "pred_mse": 0.0,
        "copy_mse": 0.0,
        "alignment_mse": 0.0,
        "action_shuffle_mse": 0.0,
        "sigreg": 0.0,
    }
    count = 0
    shuffle_count = 0
    batches = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = _prepare_task_context_batch(
            model,
            move_batch_to_device(host_batch, device),
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(
                batch,
                sigreg_weight=sigreg_weight,
                per_source_sigreg=per_source_sigreg,
            )
            action = batch["action"][:, : model.history_size].float()
            shuffled_action = _shuffled_actions(action, batch)
            shuffled_mse = None
            if shuffled_action is not None:
                shuffled_prediction = _predict_with_batch_context(
                    model,
                    output.embedding[:, : model.history_size],
                    shuffled_action,
                    batch,
                )
                shuffled_mse = F.mse_loss(shuffled_prediction, output.target_embedding)
        batch_size = int(output.embedding.shape[0])
        totals["loss"] += float(output.loss.detach()) * batch_size
        totals["pred_mse"] += float(output.prediction_loss.detach()) * batch_size
        totals["copy_mse"] += float(output.copy_loss.detach()) * batch_size
        totals["alignment_mse"] += float(output.alignment_loss.detach()) * batch_size
        if shuffled_mse is not None:
            totals["action_shuffle_mse"] += float(shuffled_mse.detach()) * batch_size
            shuffle_count += batch_size
        totals["sigreg"] += float(output.sigreg_loss.detach()) * batch_size
        count += batch_size
        batches += 1
    if count == 0:
        raise ValueError("validation loader produced no batches")
    metrics = {
        key: value / count for key, value in totals.items() if key != "action_shuffle_mse"
    }
    metrics["action_shuffle_mse"] = (
        totals["action_shuffle_mse"] / shuffle_count if shuffle_count else math.nan
    )
    metrics["action_shuffle_delta"] = metrics["action_shuffle_mse"] - metrics["pred_mse"]
    metrics["batches"] = float(batches)
    metrics["samples"] = float(count)
    return metrics


@contextmanager
def _exclusive_run_lock(output_dir: Path):
    lock_path = output_dir / ".lewm.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another grounded LeWM process holds the run lock: {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _epoch_checkpoint_paths(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    prefix = "lewm_grounded_epoch_"
    for path in output_dir.glob(f"{prefix}*.pt"):
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    return sorted(checkpoints)


def _read_metrics_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid metrics JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise TypeError(f"metrics record {line_number} must be an object")
        records.append(record)
    return records


def _write_metrics_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(_plain(record), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_finite_metrics(value: Any, *, path: str = "metrics") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_metrics(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_metrics(item, path=f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        if ".action_shuffle" in path:
            return
        raise FloatingPointError(f"non-finite value at {path}: {value}")


def _reconcile_metrics_with_checkpoint(
    metrics_path: Path,
    checkpoint: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = _read_metrics_records(metrics_path)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    checkpoint_step = int(checkpoint.get("step", 0))
    epochs = [int(record.get("epoch", -1)) for record in records]
    if epochs and epochs != list(range(1, epochs[-1] + 1)):
        raise ValueError(f"metrics epochs are not contiguous: {epochs}")
    if epochs and epochs[-1] > checkpoint_epoch:
        raise ValueError(
            "metrics contain epochs newer than the resume checkpoint: "
            f"metrics={epochs[-1]} checkpoint={checkpoint_epoch}"
        )
    if epochs and epochs[-1] == checkpoint_epoch:
        if int(records[-1].get("step", -1)) != checkpoint_step:
            raise ValueError("metrics/checkpoint step mismatch")
        return records
    if checkpoint_epoch <= 0:
        return records
    expected_previous = checkpoint_epoch - 1
    if (epochs[-1] if epochs else 0) != expected_previous:
        raise ValueError(
            "cannot reconcile metrics with resume checkpoint: "
            f"last_metrics_epoch={epochs[-1] if epochs else 0} "
            f"checkpoint_epoch={checkpoint_epoch}"
        )
    checkpoint_record = checkpoint.get("metrics")
    if not isinstance(checkpoint_record, Mapping):
        raise ValueError("resume checkpoint does not contain an epoch metrics record")
    if int(checkpoint_record.get("epoch", -1)) != checkpoint_epoch:
        raise ValueError("checkpoint metrics epoch does not match checkpoint epoch")
    if int(checkpoint_record.get("step", -1)) != checkpoint_step:
        raise ValueError("checkpoint metrics step does not match checkpoint step")
    records.append(dict(checkpoint_record))
    _write_metrics_records(metrics_path, records)
    return records


def _repair_best_alias(
    output_dir: Path,
    best_path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    candidates: list[tuple[float, int]] = []
    for record in records:
        val = record.get("val")
        if not isinstance(val, Mapping):
            continue
        pred_mse = float(val.get("pred_mse", math.inf))
        epoch = int(record.get("epoch", 0))
        if math.isfinite(pred_mse) and epoch > 0:
            candidates.append((pred_mse, epoch))
    if not candidates:
        return
    _, best_epoch = min(candidates)
    source = output_dir / f"lewm_grounded_epoch_{best_epoch:03d}.pt"
    if source.is_file():
        _replace_with_hardlink(source, best_path, refuse_existing=False)


def _resume_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = _plain(copy.deepcopy(dict(config)))
    contract.pop("output_dir", None)
    trainer = contract.get("trainer")
    if isinstance(trainer, dict):
        trainer["resume"] = None
    return contract


def _initialize_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Warm-start matching LeWM weights while allowing a new visual source.

    This is intentionally separate from ``resume``: optimizer state, step
    count, and validation history are not reused.  Only same-name, same-shape
    parameters are copied; missing parameters are allowed only under the new
    frame vision encoder interface.
    """

    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state")
    if not isinstance(source_state, Mapping):
        raise ValueError("initialize_from checkpoint has no model_state mapping")
    expected_type = getattr(model, "checkpoint_type", CHECKPOINT_TYPE)
    if checkpoint.get("checkpoint_type") != expected_type:
        raise ValueError(
            "initialize_from checkpoint type differs from target model: "
            f"source={checkpoint.get('checkpoint_type')!r} target={expected_type!r}"
        )
    target_state = model.state_dict()
    compatible = {
        name: value
        for name, value in source_state.items()
        if name in target_state and tuple(value.shape) == tuple(target_state[name].shape)
    }
    missing = [name for name in target_state if name not in compatible]
    allowed_prefix = "frame_encoder.vision_encoder."
    unexpected_missing = [name for name in missing if not name.startswith(allowed_prefix)]
    if unexpected_missing:
        raise ValueError(
            "initialize_from is missing non-vision parameters: "
            f"{unexpected_missing[:8]}"
        )
    model.load_state_dict(compatible, strict=False)
    return {
        "path": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "source_step": int(checkpoint.get("step", 0)),
        "copied_parameters": int(len(compatible)),
        "missing_visual_parameters": int(len(missing)),
    }


def train_lewm_grounded(
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "auto",
    smoke_steps: int = 0,
    resume: str | Path | None = None,
) -> TrainingResult:
    """Train under an exclusive per-output-directory filesystem lock."""

    resolved = resolve_runtime_config(config)
    output_dir = Path(str(resolved["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_run_lock(output_dir):
        return _train_lewm_grounded_locked(
            config,
            device=device,
            smoke_steps=smoke_steps,
            resume=resume,
        )


def _train_lewm_grounded_locked(
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "auto",
    smoke_steps: int = 0,
    resume: str | Path | None = None,
) -> TrainingResult:
    """Train the independent grounded LeWM path and emit resumable checkpoints."""

    resolved = resolve_runtime_config(config)
    run_kind = "smoke" if smoke_steps > 0 else "formal"
    provenance = _load_training_provenance(resolved, run_kind=run_kind)
    output_dir = Path(str(resolved["output_dir"])).expanduser().resolve()
    latest_path = output_dir / "lewm_grounded_latest.pt"
    best_path = output_dir / "lewm_grounded_best.pt"
    metrics_path = output_dir / "metrics.jsonl"
    resume_value = resume or resolved["trainer"].get("resume")
    resume_path = Path(str(resume_value)).expanduser().resolve() if resume_value else None
    existing_names = {
        entry.name for entry in output_dir.iterdir() if entry.name != ".lewm.lock"
    }
    bootstrap_names = {"resolved_config.json", "input_manifest.json"}
    if resume_path is None:
        unexpected = existing_names.difference(bootstrap_names)
        if unexpected:
            raise FileExistsError(
                "refusing to start a new run in non-empty output directory "
                f"{output_dir}: {sorted(unexpected)}"
            )
    else:
        if resume_path.parent != output_dir:
            raise ValueError(
                "resume checkpoint must remain in its original output directory: "
                f"checkpoint={resume_path.parent} output={output_dir}"
            )
        if resume_path != latest_path:
            raise ValueError(
                "resume must use the mutable latest checkpoint alias, not an "
                f"older epoch checkpoint: expected={latest_path} got={resume_path}"
            )
        epoch_checkpoints = _epoch_checkpoint_paths(output_dir)
        if epoch_checkpoints:
            _, highest_path = epoch_checkpoints[-1]
            if (
                not latest_path.exists()
                or latest_path.stat().st_ino != highest_path.stat().st_ino
            ):
                _replace_with_hardlink(
                    highest_path,
                    latest_path,
                    refuse_existing=False,
                )
        if not latest_path.is_file():
            raise FileNotFoundError(latest_path)

    seed = int(resolved["seed"])
    seed_everything(seed)
    target_device = choose_device(device)
    bundle = build_lewm_data(resolved)
    model = build_grounded_lewm(resolved, bundle.reference_dataset)
    initialize_from = resolved["trainer"].get("initialize_from")
    initialization = None
    if initialize_from:
        initialization = _initialize_model_from_checkpoint(model, initialize_from)
    _validate_expected_parameter_count(model, resolved)
    model = model.to(target_device)
    normalization = bundle.data_config["normalization"]
    model.set_input_normalization(
        proprio_mean=normalization["proprio_mean"],
        proprio_std=normalization["proprio_std"],
        action_mean=normalization["action_mean"],
        action_std=normalization["action_std"],
    )
    model_config = getattr(model, "runtime_model_config")

    optimizer_config = resolved["optimizer"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
    )
    trainer = resolved["trainer"]
    max_train_batches = int(trainer.get("max_train_batches", 0))
    batches_per_epoch = len(bundle.train_loader)
    if max_train_batches > 0:
        batches_per_epoch = min(batches_per_epoch, max_train_batches)
    configured_max_steps = int(trainer.get("max_steps", 0))
    effective_max_steps = int(smoke_steps) if smoke_steps > 0 else configured_max_steps
    total_steps = (
        effective_max_steps
        if effective_max_steps > 0
        else max(1, int(trainer["epochs"]) * batches_per_epoch)
    )
    warmup_steps = min(int(optimizer_config.get("warmup_steps", 0)), total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(step, warmup_steps, total_steps),
    )
    amp_dtype = _amp_dtype(str(trainer.get("amp_dtype", "bfloat16")))
    amp_enabled = bool(trainer.get("amp", True)) and target_device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    start_epoch = 1
    global_step = 0
    best_val_pred_mse = math.inf
    metric_records: list[dict[str, Any]] = []
    if resume_path:
        checkpoint = _torch_load(resume_path, map_location=target_device)
        expected_checkpoint_type = getattr(model, "checkpoint_type", CHECKPOINT_TYPE)
        if checkpoint.get("checkpoint_type") != expected_checkpoint_type:
            raise ValueError(
                "resume checkpoint uses a different grounded LeWM variant: "
                f"checkpoint={checkpoint.get('checkpoint_type')!r} "
                f"expected={expected_checkpoint_type!r}"
            )
        if checkpoint.get("upstream_commit") != LEWM_UPSTREAM_COMMIT:
            raise ValueError("resume checkpoint uses a different upstream LeWM commit")
        checkpoint_run_kind = checkpoint.get("run_kind")
        if checkpoint_run_kind != run_kind:
            raise ValueError(
                "smoke and formal checkpoints are not interchangeable: "
                f"checkpoint={checkpoint_run_kind!r} requested={run_kind!r}"
            )
        if _plain(checkpoint.get("provenance")) != _plain(provenance):
            raise ValueError(
                "resume checkpoint provenance differs from the currently "
                "validated frozen-input manifest"
            )
        if _plain(checkpoint["model_config"]) != _plain(model_config):
            raise ValueError("resume checkpoint model_config differs from current config/data")
        if _resume_contract(checkpoint["resolved_config"]) != _resume_contract(resolved):
            raise ValueError("resume checkpoint training config differs from current config")
        if _plain(checkpoint["data_config"]) != _plain(bundle.data_config):
            raise ValueError(
                "resume checkpoint data/split/normalization manifest differs from current data"
            )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        global_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_value = checkpoint.get("best_val_pred_mse")
        if best_value is not None:
            best_val_pred_mse = float(best_value)
        _restore_rng(checkpoint.get("rng_state"), bundle)
        metric_records = _reconcile_metrics_with_checkpoint(metrics_path, checkpoint)
        _repair_best_alias(output_dir, best_path, metric_records)

    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if provenance is not None:
        (output_dir / "input_manifest.json").write_text(
            json.dumps(provenance["manifest"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    loss_config = resolved["loss"]
    sigreg_weight = float(loss_config["sigreg_weight"])
    per_source_sigreg = bool(loss_config.get("per_source_sigreg", False))
    grad_clip = float(trainer.get("grad_clip", 1.0))
    max_val_batches = int(trainer.get("max_val_batches", 0))
    if smoke_steps > 0:
        max_val_batches = 1 if max_val_batches <= 0 else min(max_val_batches, 1)
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, int(trainer["epochs"]) + 1):
        if effective_max_steps > 0 and global_step >= effective_max_steps:
            break
        model.train()
        train_totals = {
            "loss": 0.0,
            "pred_mse": 0.0,
            "copy_mse": 0.0,
            "alignment_mse": 0.0,
            "sigreg": 0.0,
        }
        train_samples = 0
        train_batches = 0
        train_started = time.perf_counter()
        for batch_index, host_batch in enumerate(bundle.train_loader):
            if max_train_batches > 0 and batch_index >= max_train_batches:
                break
            if effective_max_steps > 0 and global_step >= effective_max_steps:
                break
            batch = _prepare_task_context_batch(
                model,
                move_batch_to_device(host_batch, target_device),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=target_device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(
                    batch,
                    sigreg_weight=sigreg_weight,
                    per_source_sigreg=per_source_sigreg,
                )
            scaler.scale(output.loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            batch_size = int(output.embedding.shape[0])
            train_totals["loss"] += float(output.loss.detach()) * batch_size
            train_totals["pred_mse"] += float(output.prediction_loss.detach()) * batch_size
            train_totals["copy_mse"] += float(output.copy_loss.detach()) * batch_size
            train_totals["alignment_mse"] += (
                float(output.alignment_loss.detach()) * batch_size
            )
            train_totals["sigreg"] += float(output.sigreg_loss.detach()) * batch_size
            train_samples += batch_size
            train_batches += 1
        if train_samples == 0:
            break
        last_epoch = epoch
        train_metrics = {key: value / train_samples for key, value in train_totals.items()}
        train_metrics["batches"] = float(train_batches)
        train_metrics["samples"] = float(train_samples)
        train_metrics["seconds"] = time.perf_counter() - train_started
        val_metrics = evaluate_lewm(
            model,
            bundle.val_loader,
            target_device,
            sigreg_weight=sigreg_weight,
            per_source_sigreg=per_source_sigreg,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_batches=max_val_batches,
        )
        record = {
            "epoch": epoch,
            "step": global_step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "upstream_commit": LEWM_UPSTREAM_COMMIT,
            "objective": (
                "oracle_prediction_mse_plus_alignment_plus_lambda_sigreg"
                if model.target_mode == "frozen_oracle"
                else "prediction_mse_plus_lambda_sigreg"
            ),
            "sigreg_weight": sigreg_weight,
            "alignment_weight": float(model.alignment_weight),
            "train": train_metrics,
            "val": val_metrics,
        }
        _assert_finite_metrics(record)
        improved = float(val_metrics["pred_mse"]) < best_val_pred_mse
        if improved:
            best_val_pred_mse = float(val_metrics["pred_mse"])
        rng_state = _capture_rng(bundle)
        keep_epoch_checkpoints = bool(trainer.get("keep_epoch_checkpoints", True))
        if keep_epoch_checkpoints:
            committed_path = output_dir / f"lewm_grounded_epoch_{epoch:03d}.pt"
        else:
            committed_path = latest_path
        save_lewm_checkpoint(
            committed_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resolved_config=resolved,
            data_config=bundle.data_config,
            model_config=model_config,
            step=global_step,
            epoch=epoch,
            run_kind=run_kind,
            provenance=provenance,
            refuse_existing=keep_epoch_checkpoints,
            metrics=record,
            best_val_pred_mse=best_val_pred_mse,
            rng_state=rng_state,
            target_optimizer_steps=total_steps,
        )
        if keep_epoch_checkpoints:
            _replace_with_hardlink(
                committed_path,
                latest_path,
                refuse_existing=False,
            )
        metric_records.append(record)
        _write_metrics_records(metrics_path, metric_records)
        if improved:
            _replace_with_hardlink(
                committed_path,
                best_path,
                refuse_existing=False,
            )
        print(json.dumps(record, sort_keys=True), flush=True)

    if run_kind == "formal" and global_step != total_steps:
        raise RuntimeError(
            "GroundedLeWM training ended before its frozen optimizer-step budget: "
            f"{global_step}/{total_steps}"
        )
    if last_epoch < start_epoch:
        if not latest_path.exists() and resume_path:
            latest_path = Path(resume_path)
        if not best_path.exists() and latest_path.exists():
            best_path = latest_path
    return TrainingResult(
        output_dir=output_dir,
        latest_checkpoint=latest_path,
        best_checkpoint=best_path,
        metrics_path=metrics_path,
        global_step=global_step,
        epoch=last_epoch,
        best_val_pred_mse=float(best_val_pred_mse),
    )
