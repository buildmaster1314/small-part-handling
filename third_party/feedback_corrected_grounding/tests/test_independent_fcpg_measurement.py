from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fcpg.independent_fcpg_measurement import (
    INDEPENDENT_MEASUREMENT_CONTRACT,
    seal_independent_fcpg_measurement,
    validate_independent_fcpg_measurement_view,
    validate_schema3_fcpg_cache,
)
from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_sequence_data import frame_identity_sha256


def _fixture(
    tmp_path: Path,
    *,
    seed: int = 7,
    has_reliability: bool = False,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sample_count = 6
    source = tmp_path / "source.npz"
    episode_id = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    frame_id = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    np.savez(
        source,
        episode_id=episode_id,
        frame_id=frame_id,
        q_nominal=np.zeros((sample_count, 2, 2), dtype=np.float32),
    )
    checkpoint = tmp_path / "corrector.pt"
    torch.save(
        {
            "schema_version": 2,
            "args": {"seed": seed, "obs_mode": "robot"},
            "model_state": {"weight": torch.ones(1)},
        },
        checkpoint,
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    q_path = cache / "q_corrected.npy"
    np.save(q_path, np.arange(24, dtype=np.float32).reshape(6, 2, 2))
    if has_reliability:
        np.save(cache / "reliability.npy", np.ones((6, 2), dtype=np.float32))
    metadata = {
        "version": 3,
        "schema_version": 3,
        "has_reliability": has_reliability,
        "source_data": str(source),
        "source_data_sha256": sha256_file(source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_count": sample_count,
        "num_keypoints": 2,
        "frame_identity_sha256": frame_identity_sha256(episode_id, frame_id),
        "q_corrected_sha256": sha256_file(q_path),
        "q_true_usage": "offline_metrics_only",
        "metrics": {"e_after": 0.5},
    }
    (cache / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return cache


def test_seal_copies_payload_and_preserves_parent_provenance(tmp_path: Path) -> None:
    parent = _fixture(tmp_path, seed=17, has_reliability=True)
    output = tmp_path / "view"

    metadata = seal_independent_fcpg_measurement(
        parent,
        output,
        measurement_view_id="corrector-seed17",
        measurement_generation_seed=17,
    )
    validated = validate_independent_fcpg_measurement_view(output)

    assert metadata["independent_measurement_contract"] == (
        INDEPENDENT_MEASUREMENT_CONTRACT
    )
    assert metadata["measurement_seed_evidence"] == "corrector_checkpoint.args.seed"
    assert metadata["metrics"] == {"e_after": 0.5}
    assert metadata["measurement_payload_transfer"] == {
        "q_corrected.npy": "copy",
        "reliability.npy": "copy",
    }
    assert validated.checkpoint_seed == 17
    assert np.array_equal(
        np.load(output / "q_corrected.npy"),
        np.load(parent / "q_corrected.npy"),
    )
    assert np.array_equal(
        np.load(output / "reliability.npy"),
        np.load(parent / "reliability.npy"),
    )
    assert (
        os.stat(output / "q_corrected.npy").st_ino
        != os.stat(parent / "q_corrected.npy").st_ino
    )


def test_hardlink_mode_is_recorded_and_hash_verified(tmp_path: Path) -> None:
    parent = _fixture(tmp_path, seed=3)
    output = tmp_path / "view"

    metadata = seal_independent_fcpg_measurement(
        parent,
        output,
        measurement_view_id="view3",
        measurement_generation_seed=3,
        transfer="hardlink",
    )

    assert metadata["measurement_payload_transfer"]["q_corrected.npy"] == "hardlink"
    assert (
        os.stat(output / "q_corrected.npy").st_ino
        == os.stat(parent / "q_corrected.npy").st_ino
    )


def test_seed_must_come_from_checkpoint_not_user_metadata(tmp_path: Path) -> None:
    parent = _fixture(tmp_path, seed=11)
    output = tmp_path / "view"
    metadata_path = parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["measurement_generation_seed"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint args.seed"):
        seal_independent_fcpg_measurement(
            parent,
            output,
            measurement_view_id="forged-seed",
            measurement_generation_seed=99,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("q", "q_corrected_sha256"),
        ("source", "source_data_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("qtrue", "q_true_usage"),
        ("frame", "frame_identity_sha256"),
    ],
)
def test_seal_rejects_tampered_parent_bindings(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    parent = _fixture(tmp_path)
    metadata_path = parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tamper == "q":
        q = np.load(parent / "q_corrected.npy")
        q[0] += 1
        np.save(parent / "q_corrected.npy", q)
    elif tamper == "source":
        Path(metadata["source_data"]).write_bytes(b"changed")
    elif tamper == "checkpoint":
        Path(metadata["checkpoint"]).write_bytes(b"changed")
    elif tamper == "qtrue":
        metadata["q_true_usage"] = "training_target"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif tamper == "frame":
        metadata["frame_identity_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError), match=message):
        seal_independent_fcpg_measurement(
            parent,
            tmp_path / "view",
            measurement_view_id="view7",
            measurement_generation_seed=7,
        )


def test_refuses_in_place_existing_output_and_resealing(tmp_path: Path) -> None:
    parent = _fixture(tmp_path)
    with pytest.raises(ValueError, match="in place"):
        seal_independent_fcpg_measurement(
            parent,
            parent,
            measurement_view_id="view7",
            measurement_generation_seed=7,
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        seal_independent_fcpg_measurement(
            parent,
            existing,
            measurement_view_id="view7",
            measurement_generation_seed=7,
        )

    sealed = tmp_path / "sealed"
    seal_independent_fcpg_measurement(
        parent,
        sealed,
        measurement_view_id="view7",
        measurement_generation_seed=7,
    )
    with pytest.raises(ValueError, match="re-sealed"):
        seal_independent_fcpg_measurement(
            sealed,
            tmp_path / "second-seal",
            measurement_view_id="another",
            measurement_generation_seed=7,
        )


def test_failed_transfer_never_publishes_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _fixture(tmp_path)
    output = tmp_path / "view"

    def fail_copy(_source: object, _destination: object) -> None:
        raise OSError("injected copy failure")

    monkeypatch.setattr("fcpg.independent_fcpg_measurement.shutil.copy2", fail_copy)
    with pytest.raises(OSError, match="injected"):
        seal_independent_fcpg_measurement(
            parent,
            output,
            measurement_view_id="view7",
            measurement_generation_seed=7,
        )
    assert not output.exists()


def test_validator_detects_parent_metadata_rewrite(tmp_path: Path) -> None:
    parent = _fixture(tmp_path)
    output = tmp_path / "view"
    seal_independent_fcpg_measurement(
        parent,
        output,
        measurement_view_id="view7",
        measurement_generation_seed=7,
    )
    parent_metadata = parent / "metadata.json"
    value = json.loads(parent_metadata.read_text(encoding="utf-8"))
    value["metrics"] = {"e_after": 123.0}
    parent_metadata.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="parent_metadata_sha256"):
        validate_independent_fcpg_measurement_view(output)


def test_schema3_validator_rejects_invalid_shape_and_view_id(tmp_path: Path) -> None:
    parent = _fixture(tmp_path)
    metadata_path = parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sample_count"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="shape"):
        validate_schema3_fcpg_cache(parent)

    parent = _fixture(tmp_path / "other")
    with pytest.raises(ValueError, match="measurement_view_id"):
        seal_independent_fcpg_measurement(
            parent,
            tmp_path / "view",
            measurement_view_id="bad/view",
            measurement_generation_seed=7,
        )
