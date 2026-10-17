"""Strict causal windows for PixelObserver v2 training and evaluation.

The wrapper reuses :class:`LeWMSequenceDataset` for source/cache validation and
adds only observer-specific target fields.  Target-frame images, FCPG
measurements, and oracle pixels remain separate from the prior input mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_sequence_data import LeWMSequenceDataset, frame_identity_sha256


TARGET_MODES = frozenset(
    {"oracle_ceiling", "single_measurement_baseline", "paired_deployment"}
)


def _load_raw_extras(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        result = {
            name: np.asarray(raw[name])
            for name in (
                "q_true",
                "valid_true",
                "camera_step_event",
                "camera_step_frame",
                "camera_intervention_onset",
                "prior_supervision_valid",
                "measurement_valid",
                "actual_camera_trajectory_phase",
            )
            if name in raw.files
        }
    return result


def _cache_array_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    return path / "q_corrected.npy" if path.is_dir() else path


def _cache_metadata_path(value: str | Path) -> Path | None:
    path = Path(value).expanduser().resolve()
    candidate = path / "metadata.json" if path.is_dir() else path.parent / "metadata.json"
    return candidate if candidate.is_file() else None


def _measurement_sha(value: str | Path) -> str:
    return sha256_file(_cache_array_path(value))


def _array_sha256(value: np.ndarray) -> str:
    """Bind an array's dtype, shape, and contiguous payload deterministically."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _canonical_camera_step_event(
    extras: Mapping[str, np.ndarray], frame_id: np.ndarray
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Resolve supported camera-step annotations to one strict event mask."""

    frame_id = np.asarray(frame_id)
    if frame_id.ndim != 1:
        raise ValueError("frame_id must be one-dimensional")
    candidates: list[tuple[str, np.ndarray]] = []
    if "camera_step_frame" in extras:
        step_frame = np.asarray(extras["camera_step_frame"])
        if step_frame.shape != frame_id.shape:
            raise ValueError("camera_step_frame must match frame_id shape")
        if not np.issubdtype(step_frame.dtype, np.integer):
            raise TypeError("camera_step_frame must have integer dtype")
        candidates.append(
            (
                "camera_step_frame",
                np.asarray((step_frame >= 0) & (frame_id == step_frame), dtype=bool),
            )
        )
    for name in ("camera_step_event", "camera_intervention_onset"):
        if name not in extras:
            continue
        event = np.asarray(extras[name])
        if event.shape != frame_id.shape:
            raise ValueError(f"{name} must match frame_id shape")
        if event.dtype != np.bool_:
            raise TypeError(f"{name} must have boolean dtype")
        candidates.append((name, np.asarray(event, dtype=bool)))
    if not candidates:
        return None, ()
    reference_name, reference = candidates[0]
    for name, candidate in candidates[1:]:
        if not np.array_equal(reference, candidate):
            raise ValueError(
                "camera step annotations disagree: "
                f"{reference_name} versus {name}"
            )
    return np.ascontiguousarray(reference, dtype=bool), tuple(
        name for name, _ in candidates
    )


def _independent_measurement_lineage(
    value: str | Path, data_path: Path
) -> dict[str, Any]:
    metadata_path = _cache_metadata_path(value)
    if metadata_path is None:
        raise ValueError("paired deployment requires metadata-bound correction caches")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "independent_measurement_contract",
        "measurement_view_id",
        "measurement_generation_seed",
        "source_data_sha256",
        "q_corrected_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"paired measurement metadata is missing {missing}")
    if metadata["independent_measurement_contract"] != "independent_fcpg_measurement_v1":
        raise ValueError("paired measurement has an invalid independence contract")
    if metadata["source_data_sha256"] != sha256_file(data_path):
        raise ValueError("paired measurement source-data binding is invalid")
    if metadata["q_corrected_sha256"] != _measurement_sha(value):
        raise ValueError("paired measurement array binding is invalid")
    if not str(metadata["measurement_view_id"]):
        raise ValueError("paired measurement_view_id must be non-empty")
    seed = metadata["measurement_generation_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("measurement_generation_seed must be an integer")
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "measurement_view_id": str(metadata["measurement_view_id"]),
        "measurement_generation_seed": int(seed),
    }


def _history_tail_mask_config(
    value: Mapping[str, Any] | None, *, history_size: int
) -> dict[str, Any]:
    raw = dict(value or {})
    enabled = bool(raw.get("enabled", False))
    probability = float(raw.get("probability", 0.0))
    max_steps = int(raw.get("max_steps", 0))
    seed = int(raw.get("seed", 0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("history-tail mask probability must lie in [0,1]")
    if enabled and not 1 <= max_steps < history_size:
        raise ValueError("enabled history-tail max_steps must lie in [1,H-1]")
    if not enabled:
        probability = 0.0
        max_steps = 0
    return {
        "enabled": enabled,
        "probability": probability,
        "max_steps": max_steps,
        "seed": seed,
        "contract": "deterministic_task_episode_target_hash_v1",
    }


def _deterministic_tail_lengths(
    dataset: "PixelObserverV2WindowDataset", config: Mapping[str, Any]
) -> np.ndarray:
    result = np.zeros(len(dataset.windows), dtype=np.int16)
    if not bool(config["enabled"]):
        return result
    threshold = float(config["probability"])
    maximum = int(config["max_steps"])
    seed = int(config["seed"])
    for item, (episode_index, start) in enumerate(dataset.windows):
        sequence = dataset.episodes[episode_index]
        target = int(sequence[start + dataset.history_size * dataset.frame_skip])
        identity = (
            f"{dataset.task_name[target]}:{dataset.episode_id[target]}:"
            f"{dataset.frame_id[target]}:{seed}"
        ).encode("utf-8")
        digest = hashlib.sha256(identity).digest()
        draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if draw < threshold:
            result[item] = 1 + int.from_bytes(digest[8:16], "big") % maximum
    return result


class PixelObserverV2WindowDataset(Dataset):
    """LeWM-compatible windows with isolated observer supervision fields."""

    def __init__(
        self,
        *,
        data_path: str | Path,
        correction_cache_a: str | Path,
        correction_cache_b: str | Path | None,
        target_mode: str,
        history_size: int,
        frame_skip: int,
        source_index: int = 0,
        source_name: str | None = None,
        task_names: Sequence[str] | None = None,
        history_tail_mask: Mapping[str, Any] | None = None,
    ) -> None:
        if target_mode not in TARGET_MODES:
            raise ValueError(f"unsupported target_mode: {target_mode!r}")
        if history_size < 2:
            raise ValueError("PixelObserver v2 requires history_size >= 2")
        if target_mode == "paired_deployment" and correction_cache_b is None:
            raise ValueError("paired_deployment requires correction_cache_b")
        if target_mode != "paired_deployment" and correction_cache_b is not None:
            raise ValueError("correction_cache_b is reserved for paired_deployment")
        self.path = Path(data_path).expanduser().resolve()
        self.target_mode = str(target_mode)
        selected_indices = None
        self.task_names: tuple[str, ...] = ()
        if task_names is not None:
            requested = tuple(sorted({str(name) for name in task_names}))
            if not requested:
                raise ValueError("task_names must not be empty")
            with np.load(self.path, allow_pickle=False) as raw:
                observed_tasks = np.asarray(raw["task_name"]).astype(str)
            missing_tasks = sorted(set(requested) - set(observed_tasks.tolist()))
            if missing_tasks:
                raise ValueError(f"task_names are absent from source: {missing_tasks}")
            selected_indices = np.flatnonzero(np.isin(observed_tasks, requested))
            self.task_names = requested
        self.base_a = LeWMSequenceDataset(
            self.path,
            history_size=history_size,
            rollout_steps=1,
            frame_skip=frame_skip,
            source_index=source_index,
            source_name=source_name,
            indices=selected_indices,
            correction_cache=correction_cache_a,
            require_pre_action_alignment=True,
        )
        self.base_b = None
        self.paired_lineage: tuple[dict[str, Any], dict[str, Any]] | None = None
        if correction_cache_b is not None:
            self.base_b = LeWMSequenceDataset(
                self.path,
                history_size=history_size,
                rollout_steps=1,
                frame_skip=frame_skip,
                source_index=source_index,
                source_name=source_name,
                indices=selected_indices,
                correction_cache=correction_cache_b,
                require_pre_action_alignment=True,
            )
            if self.base_a.windows != self.base_b.windows:
                raise ValueError("paired correction caches produced different windows")
            if _measurement_sha(correction_cache_a) == _measurement_sha(correction_cache_b):
                raise ValueError("paired correction measurements must be independent")
            lineage_a = _independent_measurement_lineage(correction_cache_a, self.path)
            lineage_b = _independent_measurement_lineage(correction_cache_b, self.path)
            if (
                lineage_a["measurement_view_id"] == lineage_b["measurement_view_id"]
                or lineage_a["measurement_generation_seed"]
                == lineage_b["measurement_generation_seed"]
            ):
                raise ValueError("paired correction caches lack independent lineage")
            self.paired_lineage = (lineage_a, lineage_b)
        self.correction_cache_a = Path(correction_cache_a).expanduser().resolve()
        self.correction_cache_b = (
            Path(correction_cache_b).expanduser().resolve()
            if correction_cache_b is not None
            else None
        )
        self.extras = _load_raw_extras(self.path)
        count = int(self.base_a.q_nominal.shape[0])
        for name, value in self.extras.items():
            if value.ndim == 0 or int(value.shape[0]) != count:
                raise ValueError(f"raw field {name!r} must contain {count} frames")
        if target_mode == "oracle_ceiling":
            if "q_true" not in self.extras or "valid_true" not in self.extras:
                raise ValueError("oracle_ceiling requires q_true and valid_true")
        self.history_size = int(history_size)
        self.frame_skip = int(frame_skip)
        self.num_keypoints = self.base_a.num_keypoints
        self.proprio_dim = self.base_a.proprio_dim
        self.transition_action_dim = self.base_a.transition_action_dim
        self.image_size = self.base_a.image_size
        self.obs_dim = self.base_a.obs_dim
        self.source_name = self.base_a.source_name
        self.task_name = self.base_a.task_name
        self.episode_id = self.base_a.episode_id
        self.frame_id = self.base_a.frame_id
        self.episodes = self.base_a.episodes
        self.windows = self.base_a.windows
        (
            self.camera_step_event,
            self.camera_step_annotation_sources,
        ) = _canonical_camera_step_event(self.extras, self.frame_id)
        self.history_tail_mask_config = _history_tail_mask_config(
            history_tail_mask, history_size=self.history_size
        )
        self.history_tail_lengths = _deterministic_tail_lengths(
            self, self.history_tail_mask_config
        )
        self.history_tail_assignment_sha256 = _array_sha256(
            self.history_tail_lengths
        )
        explicit_measurement_valid = self.extras.get("measurement_valid")
        if explicit_measurement_valid is None:
            shared_measurement_valid = np.asarray(
                self.base_a.valid_nominal, dtype=bool
            ).copy()
            self.measurement_validity_origin = "source_valid_nominal"
        else:
            explicit_measurement_valid = np.asarray(explicit_measurement_valid)
            if explicit_measurement_valid.dtype != np.bool_:
                raise TypeError("measurement_valid must have boolean dtype")
            if explicit_measurement_valid.ndim == 1:
                explicit_measurement_valid = np.repeat(
                    explicit_measurement_valid[:, None], self.num_keypoints, axis=1
                )
            if explicit_measurement_valid.shape != (count, self.num_keypoints):
                raise ValueError("measurement_valid must be [N] or [N,K]")
            shared_measurement_valid = explicit_measurement_valid & np.asarray(
                self.base_a.valid_nominal, dtype=bool
            )
            self.measurement_validity_origin = (
                "source_measurement_valid_intersect_valid_nominal"
            )
        self.shared_measurement_valid = np.ascontiguousarray(
            shared_measurement_valid, dtype=bool
        )
        self.measurement_validity_sha256 = _array_sha256(
            self.shared_measurement_valid
        )

    def __len__(self) -> int:
        return len(self.base_a)

    def _crosses_discontinuous_step(
        self, item: int, raw_indices: np.ndarray
    ) -> bool:
        del raw_indices
        if self.camera_step_event is None:
            return False
        episode_index, start = self.windows[int(item)]
        sequence = self.episodes[episode_index]
        target_position = start + self.history_size * self.frame_skip
        begin = max(0, target_position - self.frame_skip + 1)
        interval = sequence[begin : target_position + 1]
        return bool(self.camera_step_event[interval].any())

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = dict(self.base_a[item])
        # The target frame image/observation is never emitted by this dataset.
        sample["pixels"] = sample["pixels"][: self.history_size]
        sample["proprio"] = sample["proprio"][: self.history_size]
        for name in ("rgb", "obs", "goal_pixels", "goal_rgb"):
            sample.pop(name, None)
        raw_indices = sample["raw_index"].numpy().astype(np.int64, copy=False)
        corrected_a = sample.pop("q_corrected")
        sample["q_corrected_a"] = corrected_a
        if self.base_b is not None:
            corrected_b = np.array(
                self.base_b.q_corrected[raw_indices], dtype=np.float32, copy=True
            )
            sample["q_corrected_b"] = torch.from_numpy(corrected_b)
        if self.target_mode == "oracle_ceiling":
            sample["q_true_history"] = torch.from_numpy(
                np.array(
                    self.extras["q_true"][raw_indices[:-1]],
                    dtype=np.float32,
                    copy=True,
                )
            )
            sample["valid_true_history"] = torch.from_numpy(
                np.array(
                    self.extras["valid_true"][raw_indices[:-1]], dtype=bool, copy=True
                )
            )
            sample["q_true_target"] = torch.from_numpy(
                np.array(self.extras["q_true"][raw_indices[-1]], dtype=np.float32, copy=True)
            )
            sample["valid_true_target"] = torch.from_numpy(
                np.array(self.extras["valid_true"][raw_indices[-1]], dtype=bool, copy=True)
            )
        target_valid = sample["valid_nominal"][-1].clone().bool()
        explicit = self.extras.get("prior_supervision_valid")
        if explicit is not None:
            episode_index, start = self.windows[int(item)]
            episode = self.episodes[episode_index]
            target_position = start + self.history_size * self.frame_skip
            begin = max(0, target_position - self.frame_skip + 1)
            interval = episode[begin : target_position + 1]
            value = np.asarray(explicit, dtype=bool)[interval]
            if value.ndim == 1:
                target_valid &= bool(value.all())
            elif value.shape[1:] == (self.num_keypoints,):
                target_valid &= torch.from_numpy(value.all(axis=0).copy())
            else:
                raise ValueError("prior_supervision_valid must be [N] or [N,K]")
        if self._crosses_discontinuous_step(int(item), raw_indices):
            target_valid[:] = False
        sample["prior_supervision_valid_target"] = target_valid
        selected_measurement_valid = torch.from_numpy(
            np.array(self.shared_measurement_valid[raw_indices], dtype=bool, copy=True)
        )
        tail = int(self.history_tail_lengths[int(item)])
        if tail > 0:
            selected_measurement_valid[self.history_size - tail : self.history_size] = False
        if self.target_mode == "oracle_ceiling":
            sample["valid_true_history"] &= selected_measurement_valid[: self.history_size]
        sample["measurement_valid"] = selected_measurement_valid
        sample["measurement_valid_a"] = selected_measurement_valid.clone()
        if self.base_b is not None:
            # Both correction views estimate the same source frames. A single
            # source-derived mask is part of the mutual-supervision contract.
            sample["measurement_valid_b"] = selected_measurement_valid.clone()
        sample["camera_step_crossed_target"] = torch.tensor(
            self._crosses_discontinuous_step(int(item), raw_indices), dtype=torch.bool
        )
        return sample

    def provenance(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source": str(self.source_name),
            "task_names": list(self.task_names),
            "data_path": str(self.path),
            "data_sha256": sha256_file(self.path),
            "frame_identity_sha256": frame_identity_sha256(
                self.episode_id, self.frame_id
            ),
            "measurement_a_path": str(_cache_array_path(self.correction_cache_a)),
            "measurement_a_sha256": _measurement_sha(self.correction_cache_a),
            "shared_measurement_validity_contract": (
                "identical_source_mask_for_independent_a_b_v1"
            ),
            "shared_measurement_validity_origin": self.measurement_validity_origin,
            "shared_measurement_validity_sha256": self.measurement_validity_sha256,
            "shared_measurement_validity_shape": list(
                self.shared_measurement_valid.shape
            ),
            "history_tail_measurement_mask": dict(self.history_tail_mask_config),
            "history_tail_assignment_sha256": (
                self.history_tail_assignment_sha256
            ),
        }
        metadata_a = _cache_metadata_path(self.correction_cache_a)
        if metadata_a is not None:
            result["measurement_a_metadata_sha256"] = sha256_file(metadata_a)
        if self.correction_cache_b is not None:
            result.update(
                {
                    "measurement_b_path": str(_cache_array_path(self.correction_cache_b)),
                    "measurement_b_sha256": _measurement_sha(self.correction_cache_b),
                }
            )
            metadata_b = _cache_metadata_path(self.correction_cache_b)
            if metadata_b is not None:
                result["measurement_b_metadata_sha256"] = sha256_file(metadata_b)
            result["independent_measurement_lineage"] = self.paired_lineage
        return result


def episode_window_split(
    dataset: PixelObserverV2WindowDataset,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, list[str]]]:
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must lie in (0,1)")
    labels = [
        f"{dataset.task_name[int(sequence[0])]}:{dataset.episode_id[int(sequence[0])]}"
        for sequence in dataset.episodes
    ]
    order = np.arange(len(labels), dtype=np.int64)
    np.random.default_rng(int(seed)).shuffle(order)
    count = min(max(1, int(round(len(order) * float(val_fraction)))), len(order) - 1)
    val_episodes = set(order[:count].tolist())
    train_windows = [
        index
        for index, (episode_index, _) in enumerate(dataset.windows)
        if episode_index not in val_episodes
    ]
    val_windows = [
        index
        for index, (episode_index, _) in enumerate(dataset.windows)
        if episode_index in val_episodes
    ]
    if not train_windows or not val_windows:
        raise ValueError("episode split produced an empty train or validation set")
    return train_windows, val_windows, {
        "train": sorted(labels[index] for index in range(len(labels)) if index not in val_episodes),
        "val": sorted(labels[index] for index in val_episodes),
    }


def global_episode_window_split(
    datasets: Sequence[PixelObserverV2WindowDataset],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[list[int], list[int]]], dict[str, dict[str, list[str]]]]:
    """Assign each task/episode group once across every configured source.

    The conservative global key groups an episode label even when it appears in
    multiple physical files. This prevents alternate camera/intervention views
    of one generated episode from straddling train and validation.
    """

    if not datasets:
        raise ValueError("global episode split requires at least one dataset")
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must lie in (0,1)")
    labels_by_source: list[list[str]] = []
    for dataset in datasets:
        labels = [
            f"{dataset.task_name[int(sequence[0])]}:"
            f"{dataset.episode_id[int(sequence[0])]}"
            for sequence in dataset.episodes
        ]
        if len(labels) != len(set(labels)):
            raise ValueError(
                f"source {dataset.source_name!r} contains duplicate episode labels"
            )
        labels_by_source.append(labels)
    global_labels = sorted({label for labels in labels_by_source for label in labels})
    if len(global_labels) < 2:
        raise ValueError("global episode split requires at least two episode groups")
    order = np.arange(len(global_labels), dtype=np.int64)
    np.random.default_rng(int(seed)).shuffle(order)
    count = min(
        max(1, int(round(len(order) * float(val_fraction)))), len(order) - 1
    )
    val_labels = {global_labels[int(index)] for index in order[:count]}

    assignments: list[tuple[list[int], list[int]]] = []
    splits: dict[str, dict[str, list[str]]] = {}
    for dataset, labels in zip(datasets, labels_by_source):
        episode_is_val = {
            episode_index: label in val_labels
            for episode_index, label in enumerate(labels)
        }
        train_windows = [
            index
            for index, (episode_index, _) in enumerate(dataset.windows)
            if not episode_is_val[episode_index]
        ]
        val_windows = [
            index
            for index, (episode_index, _) in enumerate(dataset.windows)
            if episode_is_val[episode_index]
        ]
        if not train_windows or not val_windows:
            raise ValueError(
                "global episode split produced an empty train or validation "
                f"partition for source {dataset.source_name!r}"
            )
        assignments.append((train_windows, val_windows))
        splits[str(dataset.source_name)] = {
            "train": sorted(label for label in labels if label not in val_labels),
            "val": sorted(label for label in labels if label in val_labels),
        }
    return assignments, splits


@dataclass(frozen=True)
class PixelObserverV2DataBundle:
    datasets: tuple[PixelObserverV2WindowDataset, ...]
    train_loader: DataLoader
    val_loader: DataLoader
    provenance: dict[str, Any]
    reference_dataset: PixelObserverV2WindowDataset
    geometry_mean: tuple[float, float, float]
    geometry_std: tuple[float, float, float]


def build_pixel_observer_v2_data(config: Mapping[str, Any]) -> PixelObserverV2DataBundle:
    data = config["data"]
    loader = config["loader"]
    sources = data.get("sources", [])
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("data.sources must be a non-empty array")
    datasets: list[PixelObserverV2WindowDataset] = []
    train_parts: list[Subset] = []
    val_parts: list[Subset] = []
    geometry_sum = np.zeros(3, dtype=np.float64)
    geometry_square = np.zeros(3, dtype=np.float64)
    geometry_count = 0
    source_names: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise TypeError("each data source must be a mapping")
        name = str(source.get("name", f"source{index}"))
        if not name:
            raise ValueError("data source name must be non-empty")
        if name in source_names:
            raise ValueError(f"duplicate data source name: {name!r}")
        source_names.add(name)
        dataset = PixelObserverV2WindowDataset(
            data_path=source["path"],
            correction_cache_a=source["correction_cache_a"],
            correction_cache_b=source.get("correction_cache_b"),
            target_mode=str(config["target_mode"]),
            history_size=int(data["history_size"]),
            frame_skip=int(data["frame_skip"]),
            source_index=index,
            source_name=name,
            task_names=source.get("task_names"),
            history_tail_mask=data.get("history_tail_mask"),
        )
        datasets.append(dataset)
    assignments, splits = global_episode_window_split(
        datasets,
        val_fraction=float(data["val_fraction"]),
        seed=int(data["split_seed"]),
    )
    for dataset, (train_indices, val_indices) in zip(datasets, assignments):
        train_parts.append(Subset(dataset, train_indices))
        val_parts.append(Subset(dataset, val_indices))
        train_episode_indices = {
            dataset.windows[window_index][0] for window_index in train_indices
        }
        raw_train_indices = np.concatenate(
            [dataset.episodes[episode_index] for episode_index in sorted(train_episode_indices)]
        )
        xyz = np.asarray(
            dataset.base_a.keypoints_xyz[raw_train_indices], dtype=np.float64
        ).reshape(-1, 3)
        geometry_sum += xyz.sum(axis=0)
        geometry_square += np.square(xyz).sum(axis=0)
        geometry_count += int(xyz.shape[0])
    kwargs = {
        "batch_size": int(loader["batch_size"]),
        "num_workers": int(loader.get("num_workers", 0)),
        "pin_memory": bool(loader.get("pin_memory", False)),
    }
    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)
    train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **kwargs)
    if geometry_count == 0:
        raise ValueError("training split contains no geometry")
    geometry_mean = geometry_sum / geometry_count
    geometry_variance = np.maximum(
        geometry_square / geometry_count - np.square(geometry_mean), 0.0
    )
    geometry_std = np.sqrt(geometry_variance)
    geometry_std[geometry_std < 1e-6] = 1.0
    provenance: dict[str, Any] = {
        "target_contract": (
            "independent_a_b_mutual_supervision"
            if config["target_mode"] == "paired_deployment"
            else (
                "oracle_true_history_and_target_ceiling_v1"
                if config["target_mode"] == "oracle_ceiling"
                else str(config["target_mode"])
            )
        ),
        "history_size": int(data["history_size"]),
        "frame_skip": int(data["frame_skip"]),
        "split_seed": int(data["split_seed"]),
        "split_contract": "global_task_episode_group_v1",
        "global_episode_assignment_sha256": hashlib.sha256(
            json.dumps(splits, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "splits": splits,
        "sources": [dataset.provenance() for dataset in datasets],
    }
    if len(datasets) == 1:
        provenance["measurement_a_sha256"] = datasets[0].provenance()[
            "measurement_a_sha256"
        ]
        if datasets[0].correction_cache_b is not None:
            provenance["measurement_b_sha256"] = datasets[0].provenance()[
                "measurement_b_sha256"
            ]
    elif config["target_mode"] == "paired_deployment":
        # Core checkpoint validation requires one aggregate A/B binding.
        provenance["measurement_a_sha256"] = hashlib.sha256(
            "".join(item.provenance()["measurement_a_sha256"] for item in datasets).encode()
        ).hexdigest()
        provenance["measurement_b_sha256"] = hashlib.sha256(
            "".join(item.provenance()["measurement_b_sha256"] for item in datasets).encode()
        ).hexdigest()
    return PixelObserverV2DataBundle(
        datasets=tuple(datasets),
        train_loader=train_loader,
        val_loader=val_loader,
        provenance=provenance,
        reference_dataset=datasets[0],
        geometry_mean=tuple(float(value) for value in geometry_mean),
        geometry_std=tuple(float(value) for value in geometry_std),
    )


__all__ = [
    "PixelObserverV2DataBundle",
    "PixelObserverV2WindowDataset",
    "TARGET_MODES",
    "build_pixel_observer_v2_data",
    "episode_window_split",
    "global_episode_window_split",
]
