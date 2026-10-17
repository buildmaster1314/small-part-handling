"""Episode-aware sequence datasets for deterministic action-chunk experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from fcpg.observations import select_policy_observation


@dataclass(frozen=True)
class NormalizationStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def as_torch(self) -> dict[str, torch.Tensor]:
        return {
            "obs_mean": torch.from_numpy(self.obs_mean.astype(np.float32)),
            "obs_std": torch.from_numpy(self.obs_std.astype(np.float32)),
            "action_mean": torch.from_numpy(self.action_mean.astype(np.float32)),
            "action_std": torch.from_numpy(self.action_std.astype(np.float32)),
        }


def compute_normalization_stats(
    data_path: str | Path,
    indices: np.ndarray,
    obs_mode: str,
    minimum_std: float = 1e-3,
) -> NormalizationStats:
    with np.load(data_path, allow_pickle=False) as data:
        selected = np.asarray(indices, dtype=np.int64)
        obs = select_policy_observation(data["obs"][selected], obs_mode)
        action = data["action"][selected].astype(np.float32)
    action_min = action.min(axis=0)
    action_max = action.max(axis=0)
    return NormalizationStats(
        obs_mean=obs.mean(axis=0).astype(np.float32),
        obs_std=np.maximum(obs.std(axis=0), minimum_std).astype(np.float32),
        action_mean=((action_min + action_max) * 0.5).astype(np.float32),
        action_std=np.maximum((action_max - action_min) * 0.5, minimum_std).astype(np.float32),
    )


def compute_multi_source_normalization_stats(
    data_paths: list[str | Path],
    indices_by_source: list[np.ndarray],
    obs_mode: str,
    minimum_std: float = 1e-3,
) -> NormalizationStats:
    """Compute one policy normalization over multiple task datasets."""

    if not data_paths or len(data_paths) != len(indices_by_source):
        raise ValueError("data_paths and indices_by_source must have equal non-zero length")
    count = 0
    obs_sum: np.ndarray | None = None
    obs_square_sum: np.ndarray | None = None
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    for data_path, indices in zip(data_paths, indices_by_source, strict=True):
        with np.load(data_path, allow_pickle=False) as data:
            selected = np.asarray(indices, dtype=np.int64)
            if selected.size == 0:
                raise ValueError(f"normalization source has no selected frames: {data_path}")
            obs = select_policy_observation(data["obs"][selected], obs_mode).astype(
                np.float64
            )
            action = data["action"][selected].astype(np.float64)
        if obs_sum is None:
            obs_sum = np.zeros(obs.shape[-1], dtype=np.float64)
            obs_square_sum = np.zeros(obs.shape[-1], dtype=np.float64)
            action_min = np.full(action.shape[-1], np.inf, dtype=np.float64)
            action_max = np.full(action.shape[-1], -np.inf, dtype=np.float64)
        assert action_min is not None
        if obs.shape[-1] != obs_sum.shape[0] or action.shape[-1] != action_min.shape[0]:
            raise ValueError(
                "all normalization sources must share observation/action dimensions"
            )
        count += int(obs.shape[0])
        obs_sum += obs.sum(axis=0)
        assert obs_square_sum is not None and action_max is not None
        obs_square_sum += np.square(obs).sum(axis=0)
        action_min = np.minimum(action_min, action.min(axis=0))
        action_max = np.maximum(action_max, action.max(axis=0))
    assert obs_sum is not None and obs_square_sum is not None
    assert action_min is not None and action_max is not None
    obs_mean = obs_sum / float(count)
    obs_variance = np.maximum(
        obs_square_sum / float(count) - np.square(obs_mean),
        0.0,
    )
    return NormalizationStats(
        obs_mean=obs_mean.astype(np.float32),
        obs_std=np.maximum(np.sqrt(obs_variance), minimum_std).astype(np.float32),
        action_mean=((action_min + action_max) * 0.5).astype(np.float32),
        action_std=np.maximum((action_max - action_min) * 0.5, minimum_std).astype(
            np.float32
        ),
    )


class HomogeneousConcatDataset(ConcatDataset):
    """Concatenate sources while enforcing the tensor-shape contract."""

    def __init__(self, datasets: list[Dataset], attributes: tuple[str, ...]) -> None:
        if not datasets:
            raise ValueError("at least one source dataset is required")
        super().__init__(datasets)
        self.source_lengths = tuple(len(dataset) for dataset in datasets)
        if any(length <= 0 for length in self.source_lengths):
            raise ValueError("source datasets must be non-empty")
        for attribute in attributes:
            values = [getattr(dataset, attribute) for dataset in datasets]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"source datasets disagree on {attribute}: {values}")
            setattr(self, attribute, values[0])


@dataclass(frozen=True)
class ProjectionSource:
    data: Path
    vision_cache: Path | None
    grounding_cache: Path | None
    policy_cache: Path | None


def resolve_projection_sources(
    data: str | Path,
    vision_cache: str | Path | None,
    extra_data: list[str] | None = None,
    extra_vision_cache: list[str] | None = None,
    grounding_cache: str | Path | None = None,
    extra_grounding_cache: list[str] | None = None,
    policy_cache: str | Path | None = None,
    extra_policy_cache: list[str] | None = None,
) -> list[ProjectionSource]:
    """Pair repeatable dataset/cache CLI arguments without implicit guessing."""

    extra_data = list(extra_data or [])
    extra_vision_cache = list(extra_vision_cache or [])
    extra_grounding_cache = list(extra_grounding_cache or [])
    extra_policy_cache = list(extra_policy_cache or [])
    if vision_cache:
        if len(extra_vision_cache) != len(extra_data):
            raise ValueError(
                "each --extra-data requires one --extra-vision-cache when the primary "
                "source uses --vision-cache"
            )
    elif extra_vision_cache:
        raise ValueError("--extra-vision-cache requires a primary --vision-cache")
    if grounding_cache:
        if len(extra_grounding_cache) != len(extra_data):
            raise ValueError(
                "each --extra-data requires one --extra-grounding-cache when the primary "
                "source uses --grounding-cache"
            )
    elif extra_grounding_cache:
        raise ValueError("--extra-grounding-cache requires a primary --grounding-cache")
    if policy_cache:
        if len(extra_policy_cache) != len(extra_data):
            raise ValueError(
                "each --extra-data requires one --extra-policy-cache when the primary "
                "source uses --policy-cache"
            )
    elif extra_policy_cache:
        raise ValueError("--extra-policy-cache requires a primary --policy-cache")
    data_paths = [Path(data), *(Path(path) for path in extra_data)]
    vision_paths: list[Path | None] = [
        Path(vision_cache) if vision_cache else None,
        *(Path(path) for path in extra_vision_cache),
    ]
    grounding_paths: list[Path | None] = [
        Path(grounding_cache) if grounding_cache else None,
        *(Path(path) for path in extra_grounding_cache),
    ]
    policy_paths: list[Path | None] = [
        Path(policy_cache) if policy_cache else None,
        *(Path(path) for path in extra_policy_cache),
    ]
    if len(vision_paths) == 1 and len(data_paths) > 1:
        vision_paths.extend([None] * (len(data_paths) - 1))
    if len(grounding_paths) == 1 and len(data_paths) > 1:
        grounding_paths.extend([None] * (len(data_paths) - 1))
    if len(policy_paths) == 1 and len(data_paths) > 1:
        policy_paths.extend([None] * (len(data_paths) - 1))
    return [
        ProjectionSource(source_data, source_vision, source_grounding, source_policy)
        for source_data, source_vision, source_grounding, source_policy in zip(
            data_paths,
            vision_paths,
            grounding_paths,
            policy_paths,
            strict=True,
        )
    ]


def make_balanced_source_sampler(
    datasets: list[Dataset],
    seed: int,
) -> WeightedRandomSampler:
    """Sample each source with equal probability, independent of source size."""

    if len(datasets) < 2:
        raise ValueError("balanced source sampling requires at least two datasets")
    lengths = np.asarray([len(dataset) for dataset in datasets], dtype=np.int64)
    if bool((lengths <= 0).any()):
        raise ValueError("balanced source sampling requires non-empty datasets")
    weights = np.concatenate(
        [np.full(length, 1.0 / float(length), dtype=np.float64) for length in lengths]
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=int(lengths.sum()),
        replacement=True,
        generator=generator,
    )


def select_episode_subset(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    max_episodes: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic, nested subset of complete task episodes."""

    selected = np.asarray(indices, dtype=np.int64)
    if max_episodes <= 0:
        return selected.copy()
    required = {"task_name", "episode_id"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"episode selection requires dataset fields: {sorted(missing)}")
    task_names = data["task_name"].astype(str)[selected]
    episode_ids = data["episode_id"].astype(str)[selected]
    group_keys = np.char.add(np.char.add(task_names, ":"), episode_ids)
    groups = np.unique(group_keys)
    if max_episodes > len(groups):
        raise ValueError(
            f"requested {max_episodes} episodes, but only {len(groups)} are available"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    keep = np.isin(group_keys, groups[:max_episodes])
    return selected[keep]


class ProjectionSequenceDataset(Dataset):
    """Return padded observation histories and future action chunks.

    Formal experiments split complete episodes before constructing this
    dataset. History is left-padded with the first frame and future actions are
    right-padded with the final action; ``action_mask`` identifies real steps.
    ``history_action`` is aligned one-to-one with the observation history. The
    ``next_*`` fields contain the history ending one step after the anchor, and
    ``next_mask`` excludes terminal anchors from one-step prediction losses.
    """

    def __init__(
        self,
        data_path: str | Path,
        indices: np.ndarray,
        obs_mode: str = "robot",
        obs_horizon: int = 2,
        action_horizon: int = 32,
        vision_cache: str | Path | None = None,
        grounding_cache: str | Path | None = None,
        policy_cache: str | Path | None = None,
        source_index: int | None = None,
    ) -> None:
        if obs_horizon <= 0 or action_horizon <= 0:
            raise ValueError("obs_horizon and action_horizon must be positive")
        self.path = Path(data_path)
        self.data = np.load(self.path, allow_pickle=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.obs_mode = str(obs_mode)
        if source_index is not None and source_index < 0:
            raise ValueError("source_index must be non-negative")
        self.source_index = source_index
        self.vision_cache_path = Path(vision_cache) if vision_cache else None
        self.policy_cache_path = Path(policy_cache) if policy_cache else None
        if self.vision_cache_path is not None and self.policy_cache_path is not None:
            raise ValueError("vision_cache and policy_cache are mutually exclusive")
        self.vision_backbone_features = None
        self.vision_prefix = None
        self.policy_sampled_visual = None
        self.policy_global_visual = None
        if self.policy_cache_path is not None:
            metadata_path = self.policy_cache_path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_data = metadata.get("source_data")
            if source_data and Path(source_data).resolve() != self.path.resolve():
                raise ValueError(
                    f"policy cache source {source_data} does not match dataset {self.path.resolve()}"
                )
            self.rgb = None
            self.image_size = (int(metadata["image_width"]), int(metadata["image_height"]))
            self.policy_sampled_visual = np.load(
                self.policy_cache_path / "sampled_visual.npy"
            )
            self.policy_global_visual = np.load(
                self.policy_cache_path / "global_visual.npy"
            )
            expected_samples = int(self.data["obs"].shape[0])
            if self.policy_sampled_visual.shape[0] != expected_samples:
                raise ValueError("policy cache sample count does not match dataset")
            if self.policy_global_visual.shape[0] != expected_samples:
                raise ValueError("policy cache global sample count does not match dataset")
        elif self.vision_cache_path is None:
            self.rgb = self.data["rgb"]
            self.image_size = (int(self.rgb.shape[2]), int(self.rgb.shape[1]))
        else:
            metadata_path = self.vision_cache_path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_data = metadata.get("source_data")
            if source_data and Path(source_data).resolve() != self.path.resolve():
                raise ValueError(
                    f"vision cache source {source_data} does not match dataset {self.path.resolve()}"
                )
            self.rgb = None
            self.image_size = (int(metadata["image_width"]), int(metadata["image_height"]))
            self.vision_backbone_features = np.load(
                self.vision_cache_path / "backbone_features.npy",
                mmap_mode="r",
            )
            self.vision_prefix = np.load(
                self.vision_cache_path / "prefix.npy",
                mmap_mode="r",
            )
            expected_samples = int(self.data["obs"].shape[0])
            if self.vision_backbone_features.shape[0] != expected_samples:
                raise ValueError(
                    "vision cache sample count does not match dataset: "
                    f"{self.vision_backbone_features.shape[0]} != {expected_samples}"
                )
            if self.vision_prefix.shape[0] != expected_samples:
                raise ValueError("vision cache prefix count does not match dataset")
        self.grounding_cache_path = Path(grounding_cache) if grounding_cache else None
        self.cached_q_corrected = None
        self.cached_reliability = None
        if self.grounding_cache_path is not None:
            grounding_metadata_path = self.grounding_cache_path / "metadata.json"
            grounding_metadata: dict[str, object] = {}
            if grounding_metadata_path.exists():
                grounding_metadata = json.loads(
                    grounding_metadata_path.read_text(encoding="utf-8")
                )
                source_data = grounding_metadata.get("source_data")
                if source_data and Path(source_data).resolve() != self.path.resolve():
                    raise ValueError(
                        f"grounding cache source {source_data} does not match dataset "
                        f"{self.path.resolve()}"
                    )
            self.cached_q_corrected = np.load(
                self.grounding_cache_path / "q_corrected.npy", mmap_mode="r"
            )
            reliability_path = self.grounding_cache_path / "reliability.npy"
            has_reliability = grounding_metadata.get("has_reliability")
            if has_reliability is None:
                # Version-1 caches did not declare this field and always wrote
                # reliability.npy. Infer the capability to preserve compatibility.
                has_reliability = reliability_path.exists()
            if bool(has_reliability):
                if not reliability_path.exists():
                    raise ValueError(
                        "grounding cache metadata declares reliability, but "
                        "reliability.npy is missing"
                    )
                self.cached_reliability = np.load(reliability_path, mmap_mode="r")
        self.obs = select_policy_observation(self.data["obs"], self.obs_mode)
        self.action = self.data["action"].astype(np.float32)
        self.keypoints_xyz = self.data["keypoints_xyz"].astype(np.float32)
        self.q_nominal = self.data["q_nominal"].astype(np.float32)
        self.q_true = self.data["q_true"].astype(np.float32)
        self.valid = (self.data["valid_true"] & self.data["valid_nominal"]).astype(bool)
        self.obs_dim = int(self.obs.shape[-1])
        self.action_dim = int(self.action.shape[-1])
        self.num_keypoints = int(self.q_nominal.shape[1])
        self.policy_visual_dim = None
        if self.policy_sampled_visual is not None:
            if (
                self.policy_sampled_visual.ndim != 3
                or self.policy_sampled_visual.shape[1] != self.num_keypoints
            ):
                raise ValueError(
                    "policy cache sampled_visual must have shape [N, K, C] "
                    "matching the dataset keypoints"
                )
            if (
                self.policy_global_visual is None
                or self.policy_global_visual.ndim != 2
                or self.policy_global_visual.shape[1]
                != self.policy_sampled_visual.shape[2]
            ):
                raise ValueError(
                    "policy cache global_visual must have shape [N, C] matching sampled_visual"
                )
            self.policy_visual_dim = int(self.policy_sampled_visual.shape[2])
        if self.cached_q_corrected is not None:
            expected_shape = self.q_nominal.shape
            if self.cached_q_corrected.shape != expected_shape:
                raise ValueError(
                    f"grounding cache q_corrected shape {self.cached_q_corrected.shape} "
                    f"does not match {expected_shape}"
                )
            if (
                self.cached_reliability is not None
                and self.cached_reliability.shape != expected_shape[:2]
            ):
                raise ValueError("grounding cache reliability shape does not match dataset")

        required = {"task_name", "episode_id", "frame_id"}
        missing = required.difference(self.data.files)
        if missing:
            raise KeyError(f"sequence dataset requires fields: {sorted(missing)}")
        task_names = self.data["task_name"].astype(str)
        episode_ids = self.data["episode_id"].astype(np.int64)
        frame_ids = self.data["frame_id"].astype(np.int64)
        grouped: dict[tuple[str, int], list[int]] = {}
        for index in self.indices.tolist():
            key = (str(task_names[index]), int(episode_ids[index]))
            grouped.setdefault(key, []).append(int(index))

        self.episodes: list[np.ndarray] = []
        self.anchors: list[tuple[int, int]] = []
        for key in sorted(grouped):
            sequence = np.asarray(
                sorted(grouped[key], key=lambda index: int(frame_ids[index])),
                dtype=np.int64,
            )
            sequence_frames = frame_ids[sequence]
            if len(np.unique(sequence_frames)) != len(sequence_frames):
                raise ValueError(f"duplicate frame_id values in episode {key}")
            if len(sequence_frames) > 1 and not np.all(np.diff(sequence_frames) == 1):
                raise ValueError(
                    f"non-contiguous frame selection in episode {key}; use complete episode splits"
                )
            episode_index = len(self.episodes)
            self.episodes.append(sequence)
            self.anchors.extend((episode_index, position) for position in range(len(sequence)))

    def __len__(self) -> int:
        return len(self.anchors)

    def _history_indices(self, sequence: np.ndarray, position: int) -> np.ndarray:
        start = position - self.obs_horizon + 1
        positions = np.arange(start, position + 1)
        return sequence[np.clip(positions, 0, len(sequence) - 1)]

    def _action_indices(self, sequence: np.ndarray, position: int) -> tuple[np.ndarray, np.ndarray]:
        positions = np.arange(position, position + self.action_horizon)
        mask = positions < len(sequence)
        return sequence[np.clip(positions, 0, len(sequence) - 1)], mask

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, position = self.anchors[item]
        sequence = self.episodes[episode_index]
        history = self._history_indices(sequence, position)
        has_next = position + 1 < len(sequence)
        next_position = min(position + 1, len(sequence) - 1)
        next_history = self._history_indices(sequence, next_position)
        future, action_mask = self._action_indices(sequence, position)
        result = {
            "obs": torch.from_numpy(self.obs[history]),
            "history_action": torch.from_numpy(self.action[history]),
            "keypoints_xyz": torch.from_numpy(self.keypoints_xyz[history]),
            "q_nominal": torch.from_numpy(self.q_nominal[history]),
            "q_true": torch.from_numpy(self.q_true[history]),
            "valid": torch.from_numpy(self.valid[history]),
            "next_obs": torch.from_numpy(self.obs[next_history]),
            "next_keypoints_xyz": torch.from_numpy(self.keypoints_xyz[next_history]),
            "next_q_nominal": torch.from_numpy(self.q_nominal[next_history]),
            "next_q_true": torch.from_numpy(self.q_true[next_history]),
            "next_valid": torch.from_numpy(self.valid[next_history]),
            "next_mask": torch.tensor(has_next, dtype=torch.bool),
            "action": torch.from_numpy(self.action[future]),
            "action_mask": torch.from_numpy(action_mask),
            "anchor_index": torch.tensor(int(sequence[position]), dtype=torch.long),
            "image_size": torch.tensor(self.image_size, dtype=torch.long),
        }
        if self.source_index is not None:
            result["source_index"] = torch.tensor(
                self.source_index, dtype=torch.long
            )
        if self.policy_sampled_visual is not None:
            assert self.policy_global_visual is not None
            result["sampled_visual"] = torch.from_numpy(
                np.array(self.policy_sampled_visual[history], dtype=np.float16, copy=True)
            )
            result["global_visual"] = torch.from_numpy(
                np.array(self.policy_global_visual[history], dtype=np.float16, copy=True)
            )
            result["next_sampled_visual"] = torch.from_numpy(
                np.array(
                    self.policy_sampled_visual[next_history],
                    dtype=np.float16,
                    copy=True,
                )
            )
            result["next_global_visual"] = torch.from_numpy(
                np.array(self.policy_global_visual[next_history], dtype=np.float16, copy=True)
            )
        elif self.rgb is not None:
            rgb = self.rgb[history].astype(np.float32) / 255.0
            rgb = np.transpose(rgb, (0, 3, 1, 2))
            result["rgb"] = torch.from_numpy(rgb)
            next_rgb = self.rgb[next_history].astype(np.float32) / 255.0
            next_rgb = np.transpose(next_rgb, (0, 3, 1, 2))
            result["next_rgb"] = torch.from_numpy(next_rgb)
        else:
            assert self.vision_backbone_features is not None
            assert self.vision_prefix is not None
            result["vision_backbone_features"] = torch.from_numpy(
                np.array(self.vision_backbone_features[history], dtype=np.float16, copy=True)
            )
            result["vision_prefix"] = torch.from_numpy(
                np.array(self.vision_prefix[history], dtype=np.float16, copy=True)
            )
            result["next_vision_backbone_features"] = torch.from_numpy(
                np.array(
                    self.vision_backbone_features[next_history],
                    dtype=np.float16,
                    copy=True,
                )
            )
            result["next_vision_prefix"] = torch.from_numpy(
                np.array(self.vision_prefix[next_history], dtype=np.float16, copy=True)
            )
        if self.cached_q_corrected is not None:
            result["fcpg_q_corrected"] = torch.from_numpy(
                np.array(self.cached_q_corrected[history], dtype=np.float32, copy=True)
            )
            result["next_fcpg_q_corrected"] = torch.from_numpy(
                np.array(self.cached_q_corrected[next_history], dtype=np.float32, copy=True)
            )
        if self.cached_reliability is not None:
            result["fcpg_reliability"] = torch.from_numpy(
                np.array(self.cached_reliability[history], dtype=np.float32, copy=True)
            )
            result["next_fcpg_reliability"] = torch.from_numpy(
                np.array(self.cached_reliability[next_history], dtype=np.float32, copy=True)
            )
        return result
