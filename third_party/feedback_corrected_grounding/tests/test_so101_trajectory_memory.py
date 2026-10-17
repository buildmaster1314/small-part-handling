from pathlib import Path

import numpy as np
import pytest

from fcpg.so101_trajectory_memory import DiskBackedLFUTrajectoryStore


def _append(
    store: DiskBackedLFUTrajectoryStore,
    trajectory_id: str,
    value: float,
) -> None:
    store.append(
        trajectory_id,
        arrays={
            "states": np.full((3, 6), value, dtype=np.float32),
            "actions": np.full((5, 6), value + 1.0, dtype=np.float32),
        },
        metadata={"target": "fish", "value": value},
    )


def test_store_persists_records_across_process_instances(tmp_path: Path) -> None:
    first = DiskBackedLFUTrajectoryStore(tmp_path / "memory", memory_capacity=2)
    _append(first, "round-0001", 1.0)

    second = DiskBackedLFUTrajectoryStore(tmp_path / "memory", memory_capacity=2)
    record = second.get("round-0001")

    assert second.trajectory_ids == ("round-0001",)
    np.testing.assert_array_equal(record.arrays["states"], np.ones((3, 6)))
    assert record.metadata["target"] == "fish"
    assert second.misses == 1
    assert not record.arrays["states"].flags.writeable


def test_memory_cache_uses_lfu_then_oldest_access_for_eviction(tmp_path: Path) -> None:
    store = DiskBackedLFUTrajectoryStore(tmp_path / "memory", memory_capacity=2)
    _append(store, "round-a", 1.0)
    _append(store, "round-b", 2.0)
    store.get("round-a")

    _append(store, "round-c", 3.0)

    assert set(store.cached_ids) == {"round-a", "round-c"}
    assert "round-b" in store.trajectory_ids
    assert store.cache_frequencies()["round-a"] == 2
    loaded = store.get("round-b")
    assert loaded.metadata["value"] == 2.0
    assert store.misses == 1


def test_cache_hit_returns_the_cached_record(tmp_path: Path) -> None:
    store = DiskBackedLFUTrajectoryStore(tmp_path / "memory", memory_capacity=1)
    _append(store, "round-a", 1.0)

    first = store.get("round-a")
    second = store.get("round-a")

    assert first is second
    assert store.hits == 2
    assert store.misses == 0


def test_zero_capacity_keeps_only_the_disk_copy(tmp_path: Path) -> None:
    store = DiskBackedLFUTrajectoryStore(tmp_path / "memory", memory_capacity=0)
    _append(store, "round-a", 1.0)

    assert store.cached_ids == ()
    store.get("round-a")
    store.get("round-a")
    assert store.cached_ids == ()
    assert store.misses == 2


def test_store_rejects_duplicate_or_unsafe_ids(tmp_path: Path) -> None:
    store = DiskBackedLFUTrajectoryStore(tmp_path / "memory")
    _append(store, "round-a", 1.0)

    with pytest.raises(FileExistsError):
        _append(store, "round-a", 2.0)
    with pytest.raises(ValueError, match="trajectory_id"):
        _append(store, "../escape", 2.0)
