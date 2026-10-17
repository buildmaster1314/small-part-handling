from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fcpg.camera_pairing import (
    CAMERA_OBSERVATION_KEYS,
    PHYSICAL_IDENTITY_KEYS,
    validate_paired_camera_datasets,
)


def _write_dataset(path: Path, *, action_offset: float = 0.0, camera_offset: float = 0.0) -> None:
    arrays = {}
    for key in PHYSICAL_IDENTITY_KEYS:
        if key in {"task_name", "translation_direction", "action_source", "observation_action_alignment"}:
            arrays[key] = np.asarray("fixed")
        elif key == "independent_rng_streams":
            arrays[key] = np.asarray(True)
        elif key == "action":
            arrays[key] = np.asarray([[1.0 + action_offset]], dtype=np.float32)
        elif key == "obs":
            arrays[key] = np.asarray([[2.0]], dtype=np.float32)
        else:
            arrays[key] = np.asarray([1], dtype=np.int64)
    for key in CAMERA_OBSERVATION_KEYS:
        arrays[key] = np.asarray([camera_offset], dtype=np.float32)
    np.savez(path, **arrays)


def test_paired_camera_validator_accepts_camera_only_change(tmp_path: Path) -> None:
    clean = tmp_path / "clean.npz"
    drift = tmp_path / "drift.npz"
    _write_dataset(clean, camera_offset=0.0)
    _write_dataset(drift, camera_offset=5.0)

    report = validate_paired_camera_datasets(clean, drift)

    assert report["validated"] is True
    assert report["sample_count"] == 1
    assert report["camera_differences"]["rgb"] is True


def test_paired_camera_validator_rejects_physical_change(tmp_path: Path) -> None:
    clean = tmp_path / "clean.npz"
    drift = tmp_path / "drift.npz"
    _write_dataset(clean, camera_offset=0.0)
    _write_dataset(drift, action_offset=1.0, camera_offset=5.0)

    with pytest.raises(ValueError, match="mismatched keys: action"):
        validate_paired_camera_datasets(clean, drift)
