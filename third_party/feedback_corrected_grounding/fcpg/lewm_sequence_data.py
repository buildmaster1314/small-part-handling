"""Strict episode-local sequence windows for grounded LeWM training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fcpg.lewm_provenance import sha256_file


REQUIRED_FRAME_FIELDS = {
    "rgb",
    "obs",
    "action",
    "task_name",
    "episode_id",
    "frame_id",
    "keypoints_xyz",
    "q_nominal",
    "valid_nominal",
}

OPTIONAL_ANCHOR_MASK_FIELDS = (
    "anchor_dropout_active",
    "anchor_recovery_onset",
    "anchor_recovery_segment",
    "measurement_valid",
)


def _scalar_string(value: np.ndarray, field: str) -> str:
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    raise ValueError(f"{field} must be a scalar string, got shape {value.shape}")


def frame_identity_sha256(
    episode_id: np.ndarray,
    frame_id: np.ndarray,
) -> str:
    """Hash ordered frame identities without assuming numeric episode IDs."""

    episodes = np.asarray(episode_id).astype(str).reshape(-1)
    frames = np.asarray(frame_id, dtype=np.int64).reshape(-1)
    if episodes.shape != frames.shape:
        raise ValueError("episode_id and frame_id must have the same shape")
    digest = hashlib.sha256()
    for episode, frame in zip(episodes, frames, strict=True):
        encoded = episode.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
        digest.update(int(frame).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


class LeWMSequenceDataset(Dataset):
    """Return contiguous state sequences and the actions between them.

    A sample contains ``history_size + rollout_steps`` state frames. Consecutive
    state frames are ``frame_skip`` raw environment steps apart. Every transition
    action is the time-major concatenation of the intervening raw actions, so its
    final dimension is ``frame_skip * action_dim``.

    Unlike the action-chunk dataset, this dataset never pads history or terminal
    transitions. Episodes shorter than the required raw span contribute no
    samples, and no window may cross an episode or task boundary.
    """

    def __init__(
        self,
        data_path: str | Path,
        history_size: int,
        rollout_steps: int,
        frame_skip: int = 1,
        indices: np.ndarray | None = None,
        source_index: int = 0,
        source_name: str | None = None,
        correction_cache: str | Path | None = None,
        oracle_target_cache: str | Path | None = None,
        require_pre_action_alignment: bool = True,
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if source_index < 0:
            raise ValueError("source_index must be non-negative")

        self.path = Path(data_path)
        self.history_size = int(history_size)
        self.rollout_steps = int(rollout_steps)
        self.frame_skip = int(frame_skip)
        self.num_state_frames = self.history_size + self.rollout_steps
        self.raw_window_span = 1 + (self.num_state_frames - 1) * self.frame_skip
        self.source_index = int(source_index)
        self.source_name = str(source_name or self.path.stem)

        with np.load(self.path, allow_pickle=False) as raw:
            missing = REQUIRED_FRAME_FIELDS.difference(raw.files)
            if require_pre_action_alignment and "observation_action_alignment" not in raw.files:
                missing.add("observation_action_alignment")
            if missing:
                raise KeyError(
                    f"LeWM dataset {self.path} is missing required fields: "
                    f"{sorted(missing)}"
                )
            if require_pre_action_alignment:
                alignment = _scalar_string(
                    raw["observation_action_alignment"],
                    "observation_action_alignment",
                )
                if alignment != "pre_action":
                    raise ValueError(
                        f"LeWM dataset {self.path} requires "
                        "observation_action_alignment='pre_action' so action_t maps "
                        f"state_t to state_t+1; found {alignment!r}"
                    )

            self.rgb = np.asarray(raw["rgb"])
            self.obs = np.asarray(raw["obs"], dtype=np.float32)
            self.action = np.asarray(raw["action"], dtype=np.float32)
            self.task_name = raw["task_name"].astype(str)
            self.episode_id = np.asarray(raw["episode_id"])
            self.frame_id = np.asarray(raw["frame_id"], dtype=np.int64)
            self.keypoints_xyz = np.asarray(raw["keypoints_xyz"], dtype=np.float32)
            self.q_nominal = np.asarray(raw["q_nominal"], dtype=np.float32)
            self.valid_nominal = np.asarray(raw["valid_nominal"], dtype=bool)
            self.anchor_masks = {
                field: np.asarray(raw[field], dtype=bool)
                for field in OPTIONAL_ANCHOR_MASK_FIELDS
                if field in raw.files
            }

        self._validate_shapes()
        sample_count = int(self.rgb.shape[0])
        self.q_corrected = self._load_correction_cache(
            correction_cache,
            sample_count=sample_count,
            num_keypoints=int(self.q_nominal.shape[1]),
        )
        self.oracle_target = self._load_oracle_target_cache(
            oracle_target_cache,
            sample_count=sample_count,
        )
        if indices is None:
            selected = np.arange(sample_count, dtype=np.int64)
        else:
            selected = np.asarray(indices, dtype=np.int64)
            if selected.ndim != 1:
                raise ValueError(f"indices must be one-dimensional, got {selected.shape}")
            if len(np.unique(selected)) != len(selected):
                raise ValueError("indices must not contain duplicates")
            if selected.size and (selected.min() < 0 or selected.max() >= sample_count):
                raise IndexError(
                    f"indices must be in [0, {sample_count}), got range "
                    f"[{int(selected.min())}, {int(selected.max())}]"
                )
        if selected.size == 0:
            raise ValueError("indices select no frames")

        grouped: dict[tuple[str, str], list[int]] = {}
        for index in selected.tolist():
            key = (str(self.task_name[index]), str(self.episode_id[index]))
            grouped.setdefault(key, []).append(int(index))

        self.episodes: list[np.ndarray] = []
        self.windows: list[tuple[int, int]] = []
        for key in sorted(grouped):
            sequence = np.asarray(
                sorted(grouped[key], key=lambda index: int(self.frame_id[index])),
                dtype=np.int64,
            )
            frames = self.frame_id[sequence]
            if len(np.unique(frames)) != len(frames):
                raise ValueError(f"episode {key} contains duplicate frame_id values")
            if len(frames) > 1 and not bool(np.all(np.diff(frames) == 1)):
                raise ValueError(
                    f"episode {key} is not strictly contiguous in frame_id: "
                    f"{frames.tolist()}"
                )
            episode_index = len(self.episodes)
            self.episodes.append(sequence)
            window_count = max(0, len(sequence) - self.raw_window_span + 1)
            self.windows.extend((episode_index, start) for start in range(window_count))

        if not self.windows:
            raise ValueError(
                "no episode is long enough for a LeWM window requiring "
                f"{self.raw_window_span} contiguous raw frames "
                f"(history_size={self.history_size}, rollout_steps={self.rollout_steps}, "
                f"frame_skip={self.frame_skip})"
            )

        self.obs_dim = int(self.obs.shape[1])
        self.proprio_dim = 4
        self.action_dim = int(self.action.shape[1])
        self.transition_action_dim = self.frame_skip * self.action_dim
        self.num_keypoints = int(self.q_nominal.shape[1])
        self.oracle_target_dim = (
            int(self.oracle_target.shape[1]) if self.oracle_target is not None else None
        )
        self.image_size = (int(self.rgb.shape[2]), int(self.rgb.shape[1]))

    def _load_correction_cache(
        self,
        correction_cache: str | Path | None,
        sample_count: int,
        num_keypoints: int,
    ) -> np.ndarray | None:
        self.correction_metadata: dict[str, object] = {}
        self.correction_cache_path: Path | None = None
        if correction_cache is None:
            return None
        cache_path = Path(correction_cache)
        metadata: dict[str, object] = {}
        if cache_path.is_dir():
            metadata_path = cache_path / "metadata.json"
            correction_path = cache_path / "q_corrected.npy"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"missing correction metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_data = metadata.get("source_data")
            if source_data is not None:
                source_path = Path(str(source_data)).expanduser()
                if not source_path.is_absolute():
                    source_path = metadata_path.parent / source_path
                if not source_path.resolve() == self.path.resolve():
                    raise ValueError(
                        f"correction cache source {source_path.resolve()} does not "
                        f"match dataset {self.path.resolve()}"
                    )
        else:
            correction_path = cache_path
        if not correction_path.is_file():
            raise FileNotFoundError(f"missing corrected keypoints: {correction_path}")
        if not correction_path.suffix == ".npy":
            raise ValueError("correction_cache file must be a .npy array")
        if "sample_count" in metadata and int(metadata["sample_count"]) != sample_count:
            raise ValueError("correction cache sample_count does not match dataset")
        if "num_keypoints" in metadata and int(metadata["num_keypoints"]) != num_keypoints:
            raise ValueError("correction cache num_keypoints does not match dataset")
        if int(metadata.get("schema_version", 1)) >= 3:
            if metadata.get("source_data_sha256") != sha256_file(self.path):
                raise ValueError("correction cache source hash differs from dataset")
            identity_sha256 = frame_identity_sha256(self.episode_id, self.frame_id)
            if metadata.get("frame_identity_sha256") != identity_sha256:
                raise ValueError("correction cache frame identity differs from dataset")
            checkpoint_path = Path(str(metadata.get("checkpoint", ""))).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"missing correction checkpoint: {checkpoint_path}"
                )
            if metadata.get("checkpoint_sha256") != sha256_file(checkpoint_path):
                raise ValueError("correction checkpoint hash differs from metadata")
            if metadata.get("q_true_usage") != "offline_metrics_only":
                raise ValueError("correction cache has invalid q_true usage contract")
        corrected = np.load(correction_path, mmap_mode="r", allow_pickle=False)
        expected = (sample_count, num_keypoints, 2)
        if not corrected.shape == expected:
            raise ValueError(
                f"q_corrected must have shape {expected}, got {corrected.shape}"
            )
        if not np.issubdtype(corrected.dtype, np.floating):
            raise ValueError(f"q_corrected must use floating dtype, got {corrected.dtype}")
        if not bool(np.isfinite(corrected).all()):
            raise ValueError("q_corrected must contain only finite pixel coordinates")
        if int(metadata.get("schema_version", 1)) >= 3:
            if metadata.get("q_corrected_sha256") != sha256_file(correction_path):
                raise ValueError("correction array hash differs from metadata")
        self.correction_metadata = dict(metadata)
        self.correction_cache_path = correction_path.resolve()
        return corrected

    def _load_oracle_target_cache(
        self,
        oracle_target_cache: str | Path | None,
        *,
        sample_count: int,
    ) -> np.ndarray | None:
        self.oracle_target_metadata: dict[str, object] = {}
        self.oracle_target_cache_path: Path | None = None
        if oracle_target_cache is None:
            return None
        cache_path = Path(oracle_target_cache).expanduser().resolve()
        if not cache_path.is_dir():
            raise ValueError("oracle_target_cache must be a cache directory")
        metadata_path = cache_path / "metadata.json"
        target_path = cache_path / "oracle_target.npy"
        if not metadata_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(
                "oracle target cache requires metadata.json and oracle_target.npy"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_type") != "fcpg_oracle_target_v1":
            raise ValueError("unexpected oracle target cache type")
        source_data = metadata.get("source_data")
        if (
            not source_data
            or Path(str(source_data)).expanduser().resolve() != self.path.resolve()
        ):
            raise ValueError("oracle target cache source_data does not match dataset")
        expected_source_sha256 = metadata.get("source_data_sha256")
        if not isinstance(expected_source_sha256, str):
            raise ValueError("oracle target cache is missing source_data_sha256")
        if expected_source_sha256 != sha256_file(self.path):
            raise ValueError("oracle target cache source hash differs from dataset")
        if int(metadata.get("sample_count", -1)) != sample_count:
            raise ValueError("oracle target cache sample_count does not match dataset")
        identity_sha256 = frame_identity_sha256(self.episode_id, self.frame_id)
        if metadata.get("frame_identity_sha256") != identity_sha256:
            raise ValueError("oracle target cache frame identity differs from dataset")
        if metadata.get("q_true_usage") != "target_construction_only":
            raise ValueError("oracle target cache does not enforce target-only q_true usage")
        encoder_path = Path(str(metadata.get("encoder_checkpoint", ""))).expanduser()
        if not encoder_path.is_file():
            raise FileNotFoundError(f"missing frozen oracle target encoder: {encoder_path}")
        if metadata.get("encoder_checkpoint_sha256") != sha256_file(encoder_path):
            raise ValueError("oracle target encoder hash differs from metadata")
        target = np.load(target_path, mmap_mode="r", allow_pickle=False)
        if target.ndim != 2 or target.shape[0] != sample_count:
            raise ValueError(f"oracle_target must have shape [N, D], got {target.shape}")
        if int(metadata.get("target_dim", -1)) != int(target.shape[1]):
            raise ValueError("oracle target metadata dimension mismatch")
        if not np.issubdtype(target.dtype, np.floating):
            raise ValueError("oracle_target must use floating dtype")
        if metadata.get("oracle_target_sha256") != sha256_file(target_path):
            raise ValueError("oracle target array hash differs from metadata")
        self.oracle_target_metadata = dict(metadata)
        self.oracle_target_cache_path = target_path.resolve()
        return target

    def _validate_shapes(self) -> None:
        count = int(self.rgb.shape[0])
        frame_fields = {
            "obs": self.obs,
            "action": self.action,
            "task_name": self.task_name,
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "keypoints_xyz": self.keypoints_xyz,
            "q_nominal": self.q_nominal,
            "valid_nominal": self.valid_nominal,
        }
        mismatched = {
            name: tuple(value.shape)
            for name, value in frame_fields.items()
            if value.ndim == 0 or value.shape[0] != count
        }
        if mismatched:
            raise ValueError(
                f"all frame fields must contain {count} samples; mismatched: {mismatched}"
            )
        if self.rgb.ndim != 4 or self.rgb.shape[-1] != 3:
            raise ValueError(f"rgb must have shape [N, H, W, 3], got {self.rgb.shape}")
        if self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must use uint8 storage, got {self.rgb.dtype}")
        if self.obs.ndim != 2 or self.obs.shape[1] < 4:
            raise ValueError(f"obs must have shape [N, D] with D >= 4, got {self.obs.shape}")
        if self.action.ndim != 2:
            raise ValueError(f"action must have shape [N, A], got {self.action.shape}")
        if self.q_nominal.ndim != 3 or self.q_nominal.shape[-1] != 2:
            raise ValueError(
                f"q_nominal must have shape [N, K, 2], got {self.q_nominal.shape}"
            )
        expected_xyz = (*self.q_nominal.shape[:2], 3)
        if self.keypoints_xyz.shape != expected_xyz:
            raise ValueError(
                f"keypoints_xyz must have shape {expected_xyz}, got "
                f"{self.keypoints_xyz.shape}"
            )
        if self.valid_nominal.shape != self.q_nominal.shape[:2]:
            raise ValueError(
                f"valid_nominal must have shape {self.q_nominal.shape[:2]}, got "
                f"{self.valid_nominal.shape}"
            )
        for field, value in self.anchor_masks.items():
            if value.shape != self.q_nominal.shape[:2]:
                raise ValueError(
                    f"{field} must have shape {self.q_nominal.shape[:2]}, got "
                    f"{value.shape}"
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str | int]:
        episode_index, start = self.windows[item]
        sequence = self.episodes[episode_index]
        state_positions = start + np.arange(self.num_state_frames) * self.frame_skip
        state_indices = sequence[state_positions]

        action_blocks = []
        for position in state_positions[:-1]:
            transition_indices = sequence[position : position + self.frame_skip]
            action_blocks.append(self.action[transition_indices].reshape(-1))
        actions = torch.from_numpy(np.stack(action_blocks).astype(np.float32, copy=False))

        rgb_array = np.array(self.rgb[state_indices], dtype=np.uint8, copy=True)
        rgb = torch.from_numpy(rgb_array)
        pixels = torch.from_numpy(
            np.transpose(rgb_array.astype(np.float32) / 255.0, (0, 3, 1, 2))
        )
        obs = torch.from_numpy(
            np.array(self.obs[state_indices], dtype=np.float32, copy=True)
        )
        frame_ids = torch.from_numpy(
            np.array(self.frame_id[state_indices], dtype=np.int64, copy=True)
        )
        raw_indices = torch.from_numpy(np.array(state_indices, dtype=np.int64, copy=True))
        nominal_array = np.array(
            self.q_nominal[state_indices], dtype=np.float32, copy=True
        )
        grounded_source = self.q_corrected if self.q_corrected is not None else self.q_nominal
        grounded_array = np.array(
            grounded_source[state_indices], dtype=np.float32, copy=True
        )
        width, height = self.image_size
        inside = (
            (grounded_array[..., 0] >= 0.0)
            & (grounded_array[..., 0] <= float(width - 1))
            & (grounded_array[..., 1] >= 0.0)
            & (grounded_array[..., 1] <= float(height - 1))
        )
        valid_array = np.array(
            self.valid_nominal[state_indices], dtype=bool, copy=True
        ) & inside

        result: dict[str, torch.Tensor | str | int] = {
            "pixels": pixels,
            "rgb": rgb,
            "actions": actions,
            # Upstream LeWM names the same transition tensor ``action``.
            "action": actions,
            "obs": obs,
            "proprio": obs[:, : self.proprio_dim],
            "q_nominal": torch.from_numpy(nominal_array),
            "valid_nominal": torch.from_numpy(
                np.array(
                    self.valid_nominal[state_indices], dtype=bool, copy=True
                )
            ),
            "keypoints_xyz": torch.from_numpy(
                np.array(self.keypoints_xyz[state_indices], dtype=np.float32, copy=True)
            ),
            "valid": torch.from_numpy(valid_array),
            "task_name": str(self.task_name[state_indices[0]]),
            "episode_id": int(self.episode_id[state_indices[0]]),
            "frame_id": frame_ids,
            "raw_index": raw_indices,
            "source": self.source_name,
            "source_index": torch.tensor(self.source_index, dtype=torch.long),
            "goal_pixels": pixels[-1],
            "goal_rgb": rgb[-1],
            "goal_frame_id": frame_ids[-1],
        }
        if self.q_corrected is not None:
            result["q_corrected"] = torch.from_numpy(grounded_array)
        for field, value in self.anchor_masks.items():
            result[field] = torch.from_numpy(
                np.array(value[state_indices], dtype=bool, copy=True)
            )
        if self.oracle_target is not None:
            result["oracle_target"] = torch.from_numpy(
                np.array(
                    self.oracle_target[state_indices],
                    dtype=np.float32,
                    copy=True,
                )
            )
        return result
