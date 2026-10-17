"""Disk-backed SO-101 trajectory records with a bounded LFU memory cache."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


STORE_FORMAT = "so101-trajectory-memory-v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class SO101TrajectoryRecord:
    trajectory_id: str
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    @property
    def byte_size(self) -> int:
        return int(sum(value.nbytes for value in self.arrays.values()))


@dataclass
class _LFUEntry:
    record: SO101TrajectoryRecord
    frequency: int
    last_access: int


def _validate_trajectory_id(trajectory_id: str) -> None:
    if not _SAFE_ID.fullmatch(trajectory_id):
        raise ValueError(
            "trajectory_id must use only letters, digits, dot, underscore, or dash"
        )


def _freeze_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not arrays:
        raise ValueError("a trajectory record must contain at least one array")
    frozen: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not name or not _SAFE_ID.fullmatch(name):
            raise ValueError(f"invalid trajectory array name: {name!r}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"trajectory array {name!r} cannot use object dtype")
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"trajectory array {name!r} must be finite")
        array = np.ascontiguousarray(array).copy()
        array.flags.writeable = False
        frozen[name] = array
    return frozen


def _json_copy(metadata: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(metadata), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory metadata must be finite JSON data") from exc


class DiskBackedLFUTrajectoryStore:
    """Keep every record on disk while LFU-evicting decoded records in RAM."""

    def __init__(self, root: str | Path, *, memory_capacity: int = 32) -> None:
        if memory_capacity < 0:
            raise ValueError("memory_capacity must be non-negative")
        self.root = Path(root).expanduser().resolve()
        self.segment_root = self.root / "segments"
        self.index_path = self.root / "index.json"
        self.memory_capacity = int(memory_capacity)
        self._cache: dict[str, _LFUEntry] = {}
        self._clock = 0
        self.hits = 0
        self.misses = 0

        self.segment_root.mkdir(parents=True, exist_ok=True)
        if self.index_path.is_file():
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
            if index.get("format") != STORE_FORMAT:
                raise ValueError("unsupported SO-101 trajectory store format")
            trajectory_ids = index.get("trajectory_ids")
            if not isinstance(trajectory_ids, list) or len(set(trajectory_ids)) != len(
                trajectory_ids
            ):
                raise ValueError("trajectory store index is malformed")
            for trajectory_id in trajectory_ids:
                _validate_trajectory_id(trajectory_id)
            self._trajectory_ids = list(trajectory_ids)
        else:
            self._trajectory_ids: list[str] = []
            self._write_index()

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(self._trajectory_ids)

    @property
    def cached_ids(self) -> tuple[str, ...]:
        return tuple(self._cache)

    def _paths(self, trajectory_id: str) -> tuple[Path, Path]:
        _validate_trajectory_id(trajectory_id)
        return (
            self.segment_root / f"{trajectory_id}.npz",
            self.segment_root / f"{trajectory_id}.json",
        )

    def _write_index(self) -> None:
        payload = {
            "format": STORE_FORMAT,
            "trajectory_ids": self._trajectory_ids,
        }
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.index_path)

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _cache_record(self, record: SO101TrajectoryRecord) -> None:
        if self.memory_capacity == 0:
            return
        if record.trajectory_id in self._cache:
            entry = self._cache[record.trajectory_id]
            entry.record = record
            entry.frequency += 1
            entry.last_access = self._tick()
            return
        if len(self._cache) >= self.memory_capacity:
            victim = min(
                self._cache,
                key=lambda key: (
                    self._cache[key].frequency,
                    self._cache[key].last_access,
                ),
            )
            del self._cache[victim]
        self._cache[record.trajectory_id] = _LFUEntry(
            record=record,
            frequency=1,
            last_access=self._tick(),
        )

    def append(
        self,
        trajectory_id: str,
        *,
        arrays: Mapping[str, np.ndarray],
        metadata: Mapping[str, Any],
    ) -> SO101TrajectoryRecord:
        """Persist one immutable trajectory segment and add it to the LFU cache."""

        array_path, metadata_path = self._paths(trajectory_id)
        if trajectory_id in self._trajectory_ids or array_path.exists() or metadata_path.exists():
            raise FileExistsError(f"trajectory already exists: {trajectory_id}")
        frozen = _freeze_arrays(arrays)
        metadata_copy = _json_copy(metadata)

        array_temporary = array_path.with_suffix(".npz.tmp")
        with array_temporary.open("wb") as handle:
            np.savez_compressed(handle, **frozen)
        metadata_temporary = metadata_path.with_suffix(".json.tmp")
        metadata_temporary.write_text(
            json.dumps(metadata_copy, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        array_temporary.replace(array_path)
        metadata_temporary.replace(metadata_path)
        self._trajectory_ids.append(trajectory_id)
        self._write_index()

        record = SO101TrajectoryRecord(trajectory_id, frozen, metadata_copy)
        self._cache_record(record)
        return record

    def get(self, trajectory_id: str) -> SO101TrajectoryRecord:
        """Load one record, promoting it according to LFU access frequency."""

        _validate_trajectory_id(trajectory_id)
        cached = self._cache.get(trajectory_id)
        if cached is not None:
            self.hits += 1
            cached.frequency += 1
            cached.last_access = self._tick()
            return cached.record
        if trajectory_id not in self._trajectory_ids:
            raise KeyError(trajectory_id)
        self.misses += 1
        array_path, metadata_path = self._paths(trajectory_id)
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = _freeze_arrays({name: archive[name] for name in archive.files})
        metadata = _json_copy(
            json.loads(metadata_path.read_text(encoding="utf-8"))
        )
        record = SO101TrajectoryRecord(trajectory_id, arrays, metadata)
        self._cache_record(record)
        return record

    def cache_frequencies(self) -> dict[str, int]:
        return {
            trajectory_id: entry.frequency
            for trajectory_id, entry in self._cache.items()
        }


__all__ = [
    "DiskBackedLFUTrajectoryStore",
    "SO101TrajectoryRecord",
    "STORE_FORMAT",
]
