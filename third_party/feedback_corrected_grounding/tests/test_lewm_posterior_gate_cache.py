from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_posterior_gate_cache import (
    DEPLOYMENT_FILENAME,
    DEPLOYMENT_KEYS,
    EVALUATION_FILENAME,
    LABEL_FILENAME,
    LABEL_KEYS,
    METADATA_FILENAME,
    concatenate_training_rows,
    load_posterior_gate_cache,
    training_rows_from_cache,
)
from fcpg.lewm_provenance import sha256_file


def _write_cache(path: Path, *, episode: int) -> Path:
    path.mkdir()
    n, h, k, d = 5, 3, 3, 2
    events = np.asarray(["o0", "o1", "o2", "r0", "r1"])
    measurement = np.arange(n * k * d, dtype=np.float32).reshape(n, k, d)
    history = np.repeat(measurement[:, None], h, axis=1)
    q = np.full((n, k, 2), 32.0, dtype=np.float32)
    deployment = {
        "measurement_history": history,
        "action_chunks": np.ones((n, h, 20), dtype=np.float32),
        "measurement_current": measurement,
        "q_nominal_history": np.repeat(q[:, None], h, axis=1),
        "q_corrected_history": np.repeat(q[:, None], h, axis=1),
        "valid_nominal_history": np.ones((n, h, k), dtype=np.bool_),
        "q_nominal_current": q,
        "q_corrected_current": q,
        "valid_nominal_current": np.ones((n, k), dtype=np.bool_),
        "teacher_forced_factual_prior": measurement + 1.0,
        "measurement_frame_embedding": np.zeros((25, k, d), dtype=np.float32),
    }
    assert set(deployment) == set(DEPLOYMENT_KEYS)
    labels = {
        "canonical_current": measurement + 0.5,
        "canonical_measurement_history": history + 0.5,
        "canonical_target": measurement + 0.5,
        "canonical_q_corrected_history": np.repeat(q[:, None], h, axis=1),
        "canonical_q_corrected": q,
        "canonical_valid_history": np.ones((n, h, k), dtype=np.bool_),
        "canonical_valid_current": np.ones((n, k), dtype=np.bool_),
    }
    assert set(labels) == set(LABEL_KEYS)
    direct = np.zeros((n, k), dtype=np.bool_)
    direct[:, 1] = True
    evaluation = {
        "event": events,
        "task_name": np.asarray(["pick-place-v3"] * n),
        "episode_id": np.full(n, episode, dtype=np.int64),
        "phase_index": np.zeros(n, dtype=np.int64),
        "dropout_active": direct & np.isin(events[:, None], ("o0", "o1", "o2")),
        "recovery_active": direct & np.isin(events[:, None], ("r0", "r1")),
    }
    np.savez_compressed(path / DEPLOYMENT_FILENAME, **deployment)
    np.savez_compressed(path / LABEL_FILENAME, **labels)
    np.savez_compressed(path / EVALUATION_FILENAME, **evaluation)
    metadata = {
        "schema_version": 1,
        "cache_type": "fcpg_lewm_posterior_gate_cache_v1",
        "sample_count": n,
        "image_width": 224,
        "image_height": 224,
        "contract": {
            "checkpoint_step": 2000,
            "history_size": 3,
            "frame_skip": 5,
            "action_dim": 20,
        },
        "artifacts": {
            name: {"sha256": sha256_file(path / name)}
            for name in (DEPLOYMENT_FILENAME, LABEL_FILENAME, EVALUATION_FILENAME)
        },
    }
    (path / METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_cache_maps_partitioned_arrays_to_training_rows(tmp_path: Path) -> None:
    cache = load_posterior_gate_cache(_write_cache(tmp_path / "train", episode=1))
    rows = training_rows_from_cache(cache)
    assert rows.z_measurement.shape == (5, 3, 2)
    assert rows.z_prior.shape == (5, 3, 2)
    assert rows.z_hold_last.shape == (5, 3, 2)
    assert rows.validity_history.shape == (5, 3, 3)
    assert rows.last_causal_action_chunk.shape == (5, 20)
    masks = rows.loss_masks
    assert masks.primary_direct_o0.sum().item() == 1
    assert masks.primary_direct_o1.sum().item() == 1
    assert masks.primary_direct_o2.sum().item() == 1
    assert masks.safety_recovery.sum().item() == 2
    assert masks.safety_indirect.sum().item() == 10


def test_concat_returns_disjoint_ranges_and_preserves_masks(tmp_path: Path) -> None:
    train = load_posterior_gate_cache(_write_cache(tmp_path / "train", episode=1))
    validation = load_posterior_gate_cache(
        _write_cache(tmp_path / "validation", episode=2)
    )
    rows, ranges = concatenate_training_rows((train, validation))
    assert ranges == ((0, 5), (5, 10))
    assert rows.sample_count == 10
    assert rows.loss_masks.primary_direct_o1.sum().item() == 2
    assert len(rows.source_hashes) == 8


def test_loader_rejects_partition_hash_mismatch(tmp_path: Path) -> None:
    path = _write_cache(tmp_path / "train", episode=1)
    with (path / DEPLOYMENT_FILENAME).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_posterior_gate_cache(path)


def test_loader_rejects_incomplete_event_set(tmp_path: Path) -> None:
    path = _write_cache(tmp_path / "train", episode=1)
    with np.load(path / EVALUATION_FILENAME, allow_pickle=False) as raw:
        values = {name: np.array(raw[name], copy=True) for name in raw.files}
    values["event"][-1] = "o0"
    np.savez_compressed(path / EVALUATION_FILENAME, **values)
    metadata = json.loads((path / METADATA_FILENAME).read_text(encoding="utf-8"))
    metadata["artifacts"][EVALUATION_FILENAME]["sha256"] = sha256_file(
        path / EVALUATION_FILENAME
    )
    (path / METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="event set"):
        load_posterior_gate_cache(path)
