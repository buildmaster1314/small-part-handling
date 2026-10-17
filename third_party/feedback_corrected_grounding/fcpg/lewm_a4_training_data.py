"""Strict seven-task dataset and loader construction for A4 training caches."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fcpg.lewm_a4_sampling import A4EpisodeBalancedBatchSampler
from fcpg.lewm_a4_storage import VALID_SPLITS, validate_a4_state_cache
from fcpg.lewm_action_branch_a4 import TASKS


_SUPPORTED_FLOAT_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


def _torch_float_dtype(value: object) -> torch.dtype:
    if not isinstance(value, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    if value not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError(
            "dtype must be a supported floating torch.dtype "
            "(float16, bfloat16, float32, or float64)"
        )
    return value


def _copy_cache(cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(cache, Mapping):
        raise TypeError("cache must be a mapping of numpy arrays")
    if any(not isinstance(name, str) for name in cache):
        raise TypeError("cache keys must be strings")
    if any(not isinstance(value, np.ndarray) for value in cache.values()):
        raise TypeError("every cache value must already be a numpy ndarray")
    copied = {name: np.array(value, copy=True) for name, value in cache.items()}
    validate_a4_state_cache(copied)
    return copied


def _split_name(value: object) -> str:
    if not isinstance(value, (str, np.str_)):
        raise TypeError("split must be a string")
    result = str(value)
    if result not in VALID_SPLITS:
        raise ValueError(
            f"split must be one of {list(VALID_SPLITS)}, got {result!r}"
        )
    return result


def _require_seven_tasks(
    task_class_index: np.ndarray,
    task_name: np.ndarray,
    *,
    label: str,
) -> None:
    expected_indices = set(range(len(TASKS)))
    observed_indices = {int(value) for value in task_class_index.tolist()}
    observed_names = set(task_name.astype(str, copy=False).tolist())
    expected_names = set(TASKS)
    if observed_indices != expected_indices or observed_names != expected_names:
        missing = [
            TASKS[index] for index in sorted(expected_indices - observed_indices)
        ]
        unexpected = sorted(observed_names - expected_names)
        raise ValueError(
            f"{label} must cover the frozen seven A4 tasks exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _selected_float_tensor(
    value: np.ndarray,
    indices: np.ndarray,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dtype == torch.float64:
        bridge_dtype = np.float64
    elif dtype == torch.float16:
        bridge_dtype = np.float16
    else:
        bridge_dtype = np.float32
    selected = np.array(
        value[indices], dtype=bridge_dtype, order="C", copy=True
    )
    tensor = torch.from_numpy(selected)
    return tensor if tensor.dtype == dtype else tensor.to(dtype=dtype)


def _episode_ids(value: np.ndarray, indices: np.ndarray) -> tuple[str | int, ...]:
    selected = value[indices]
    if selected.dtype.kind in {"i", "u"}:
        return tuple(int(item) for item in selected.tolist())
    return tuple(selected.astype(str, copy=False).tolist())


def _dataset_index(value: object, length: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("dataset index must be an integer")
    index = int(value)
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise IndexError("dataset index is out of range")
    return index


class A4StateCacheDataset(Dataset[dict[str, Any]]):
    """Copy-safe split view over one shared, frozen seven-task A4 cache."""

    def __init__(
        self,
        cache: Mapping[str, np.ndarray],
        *,
        split: str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dtype = _torch_float_dtype(dtype)
        self.split = _split_name(split)
        arrays = _copy_cache(cache)

        _require_seven_tasks(
            arrays["task_class_index"],
            arrays["task_name"],
            label="A4 state cache",
        )
        split_values = arrays["split"].astype(str, copy=False)
        indices = np.flatnonzero(split_values == self.split).astype(
            np.int64, copy=False
        )
        if indices.size == 0:
            available = sorted(set(split_values.tolist()))
            raise ValueError(
                f"A4 state cache has no rows for split {self.split!r}; "
                f"available={available}"
            )
        _require_seven_tasks(
            arrays["task_class_index"][indices],
            arrays["task_name"][indices],
            label=f"A4 state cache split {self.split!r}",
        )

        self.source_indices = tuple(int(index) for index in indices.tolist())
        self.task_class_index = tuple(
            int(value) for value in arrays["task_class_index"][indices].tolist()
        )
        self.task_name = tuple(
            arrays["task_name"][indices].astype(str, copy=False).tolist()
        )
        self.source_episode_id = _episode_ids(
            arrays["source_episode_id"], indices
        )

        self._history_embedding = _selected_float_tensor(
            arrays["history_state"], indices, dtype=self.dtype
        )
        self._history_action = _selected_float_tensor(
            arrays["history_action"], indices, dtype=self.dtype
        )
        self._future_action = _selected_float_tensor(
            arrays["future_action"], indices, dtype=self.dtype
        )
        self._target_embedding = _selected_float_tensor(
            arrays["target_state"], indices, dtype=self.dtype
        )
        self._task_index = torch.tensor(self.task_class_index, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.task_class_index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = _dataset_index(item, len(self))
        return {
            "history_embedding": self._history_embedding[index].clone(),
            "history_action": self._history_action[index].clone(),
            "future_action": self._future_action[index].clone(),
            "target_embedding": self._target_embedding[index].clone(),
            "task_index": self._task_index[index].clone(),
            "task_name": self.task_name[index],
            "source_episode_id": self.source_episode_id[index],
        }


def _nonnegative_worker_count(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("num_workers must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("num_workers must be non-negative")
    return result


def build_a4_state_cache_loader(
    cache: Mapping[str, np.ndarray],
    *,
    split: str,
    seed: int,
    dtype: torch.dtype = torch.float32,
    batches_per_epoch: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Any]]:
    """Build the only supported A4 loader: strict 7-by-4 episode batches."""

    workers = _nonnegative_worker_count(num_workers)
    if not isinstance(pin_memory, (bool, np.bool_)):
        raise TypeError("pin_memory must be boolean")
    dataset = A4StateCacheDataset(cache, split=split, dtype=dtype)
    batch_sampler = A4EpisodeBalancedBatchSampler(
        dataset.task_name,
        dataset.source_episode_id,
        seed=seed,
        batches_per_epoch=batches_per_epoch,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=bool(pin_memory),
    )


# Keep both common spellings while exposing one strict implementation.
build_a4_state_cache_dataloader = build_a4_state_cache_loader


__all__ = [
    "A4StateCacheDataset",
    "build_a4_state_cache_dataloader",
    "build_a4_state_cache_loader",
]
