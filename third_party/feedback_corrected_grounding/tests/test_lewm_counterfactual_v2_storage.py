from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

import fcpg.lewm_counterfactual_v2_storage as storage
from fcpg.lewm_counterfactual_v2_storage import (
    CounterfactualV2StorageError,
    audit_counterfactual_v2_store,
    load_counterfactual_v2_store,
    semantic_array_sha256,
    write_counterfactual_v2_store,
)


def _payload() -> dict[str, object]:
    return {
        "deployment_inputs": {
            "task_index": np.asarray([1, 0], dtype=np.uint8),
            "z_history": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        },
        "offline_labels": {
            "actual_return": np.asarray([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32),
        },
        "audit_arrays": {
            "valid_grounded": np.asarray([[True, False], [True, True]]),
        },
        "index_rows": [
            {
                "sample_index": 0,
                "snapshot_id": "z",
                "task": "reach-v3",
                "episode": 2,
            },
            {
                "sample_index": 1,
                "snapshot_id": "a",
                "task": "push-v3",
                "episode": 1,
            },
        ],
        "provenance": {
            "scientific_provenance": {
                "git_head": "a" * 40,
                "checkpoints": [{"name": "fcpg", "sha256": "b" * 64}],
                "config": {
                    "protocol": "fcpg_lewm_counterfactual_v2_development",
                    "seed": 8192,
                },
                "config_sha256": "c" * 64,
            },
            "g0_manifest": {"manifest_sha256": "d" * 64},
        },
    }


def _write(path: Path) -> dict[str, object]:
    return write_counterfactual_v2_store(path, **_payload())


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _rewrite_manifest(root: Path, mutate) -> dict[str, object]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["result_sha256"] = storage._manifest_result_sha256(manifest)
    path.write_bytes(_canonical(manifest) + b"\n")
    return manifest


def test_round_trip_writes_canonical_content_addressed_layout(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    manifest = _write(destination)

    assert set(path.name for path in destination.iterdir()) == {
        "deployment_inputs",
        "offline_labels",
        "audit",
        "index.jsonl",
        "manifest.json",
    }
    assert (destination / "index.jsonl").read_text(encoding="utf-8").splitlines() == [
        '{"episode":2,"sample_index":0,"snapshot_id":"z","task":"reach-v3"}',
        '{"episode":1,"sample_index":1,"snapshot_id":"a","task":"push-v3"}',
    ]
    record = manifest["arrays"]["deployment_inputs"]["z_history"]
    stored = np.load(destination / record["path"], allow_pickle=False)
    assert record["dtype"] == stored.dtype.str
    assert record["shape"] == [2, 3, 4]
    assert record["semantic_sha256"] == semantic_array_sha256(stored)
    assert record["file_sha256"] == hashlib.sha256(
        (destination / record["path"]).read_bytes()
    ).hexdigest()

    audited = audit_counterfactual_v2_store(destination)
    loaded = load_counterfactual_v2_store(destination)
    assert audited.result_sha256 == manifest["result_sha256"]
    assert audited.index_row_count == 2
    assert audited.array_count == 4
    assert loaded.index_rows[0]["snapshot_id"] == "z"
    scientific = loaded.provenance["scientific_provenance"]
    assert scientific["config"] == {
        "protocol": "fcpg_lewm_counterfactual_v2_development",
        "seed": 8192,
    }
    assert scientific["config_sha256"] == "c" * 64
    assert loaded.provenance["g0_manifest"]["manifest_sha256"] == "d" * 64
    assert np.array_equal(
        loaded.deployment_inputs["z_history"],
        _payload()["deployment_inputs"]["z_history"],
    )
    assert isinstance(loaded.deployment_inputs["z_history"], np.memmap)
    assert loaded.deployment_inputs["z_history"].flags.writeable is False


def test_result_hash_is_independent_of_mapping_input_order(
    tmp_path: Path,
) -> None:
    first = _payload()
    second = _payload()
    second["deployment_inputs"] = dict(
        reversed(list(second["deployment_inputs"].items()))
    )

    first_manifest = write_counterfactual_v2_store(tmp_path / "first", **first)
    second_manifest = write_counterfactual_v2_store(tmp_path / "second", **second)

    assert first_manifest == second_manifest


def test_result_hash_binds_array_aligned_index_row_order(tmp_path: Path) -> None:
    first = _payload()
    second = _payload()
    reversed_rows = list(reversed(second["index_rows"]))
    for sample_index, row in enumerate(reversed_rows):
        row["sample_index"] = sample_index
    second["index_rows"] = reversed_rows

    first_manifest = write_counterfactual_v2_store(tmp_path / "first", **first)
    second_manifest = write_counterfactual_v2_store(tmp_path / "second", **second)

    assert first_manifest["result_sha256"] != second_manifest["result_sha256"]
    assert (tmp_path / "first/index.jsonl").read_bytes() != (
        tmp_path / "second/index.jsonl"
    ).read_bytes()


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    destination.mkdir()
    sentinel = destination / "owner.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _write(destination)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (destination / "manifest.json").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_dangling_destination_symlink_is_never_replaced(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    destination.symlink_to(tmp_path / "missing-owner")

    with pytest.raises(FileExistsError):
        _write(destination)

    assert destination.is_symlink()
    assert os.readlink(destination) == str(tmp_path / "missing-owner")
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_write_failure_leaves_no_final_or_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "dataset"
    real_write = storage._write_npy_file
    calls = 0

    def fail_after_one(path: Path, value: np.ndarray):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        return real_write(path, value)

    monkeypatch.setattr(storage, "_write_npy_file", fail_after_one)
    with pytest.raises(OSError, match="injected"):
        _write(destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_publication_race_does_not_replace_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "dataset"

    def lose_race(source: Path, target: Path) -> None:
        target.mkdir()
        (target / "winner.txt").write_text("other writer", encoding="utf-8")
        raise FileExistsError(target)

    monkeypatch.setattr(storage, "_publish_directory_no_replace", lose_race)
    with pytest.raises(FileExistsError):
        _write(destination)

    assert (destination / "winner.txt").read_text(encoding="utf-8") == "other writer"
    assert not (destination / "manifest.json").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_unsupported_atomic_publication_fails_closed_without_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "dataset"
    monkeypatch.setattr(storage, "_linux_rename_noreplace", lambda *_args: False)

    with pytest.raises(
        CounterfactualV2StorageError,
        match="atomic no-replace directory publication is unsupported",
    ):
        _write(destination)

    assert not os.path.lexists(destination)
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_unsupported_atomic_publication_never_replaces_racing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "dataset"

    def unsupported_after_winner(_source: Path, target: Path) -> bool:
        target.mkdir()
        (target / "winner.txt").write_text("other writer", encoding="utf-8")
        return False

    monkeypatch.setattr(storage, "_linux_rename_noreplace", unsupported_after_winner)
    with pytest.raises(CounterfactualV2StorageError, match="unsupported"):
        _write(destination)

    assert (destination / "winner.txt").read_text(encoding="utf-8") == "other writer"
    assert not (destination / "manifest.json").exists()
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_files_and_directories_are_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = os.fsync
    modes: list[int] = []

    def record_fsync(descriptor: int) -> None:
        modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(storage.os, "fsync", record_fsync)
    _write(tmp_path / "dataset")

    assert any(stat.S_ISREG(mode) for mode in modes)
    assert any(stat.S_ISDIR(mode) for mode in modes)


def test_tampered_array_file_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    _write(destination)
    array = np.load(
        destination / "deployment_inputs" / "task_index.npy",
        mmap_mode="r+",
        allow_pickle=False,
    )
    array[0] = np.uint8(9)
    array.flush()
    del array

    with pytest.raises(CounterfactualV2StorageError, match="file_sha256"):
        audit_counterfactual_v2_store(destination)


def test_semantic_tamper_is_rejected_even_when_file_hash_is_rebound(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dataset"
    _write(destination)
    path = destination / "deployment_inputs" / "task_index.npy"
    np.save(path, np.asarray([8, 7], dtype=np.uint8), allow_pickle=False)

    def rebind_file_only(manifest: dict[str, object]) -> None:
        record = manifest["arrays"]["deployment_inputs"]["task_index"]
        payload = path.read_bytes()
        record["size_bytes"] = len(payload)
        record["file_sha256"] = hashlib.sha256(payload).hexdigest()

    _rewrite_manifest(destination, rebind_file_only)
    with pytest.raises(CounterfactualV2StorageError, match="semantic_sha256"):
        audit_counterfactual_v2_store(destination)


def test_result_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    _write(destination)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["result_sha256"] = "0" * 64
    path.write_bytes(_canonical(manifest) + b"\n")

    with pytest.raises(CounterfactualV2StorageError, match="result_sha256"):
        audit_counterfactual_v2_store(destination)


def test_noncanonical_index_order_is_rejected_after_hash_rebinding(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dataset"
    _write(destination)
    path = destination / "index.jsonl"
    lines = path.read_bytes().splitlines()
    payload = b"\n".join(reversed(lines)) + b"\n"
    path.write_bytes(payload)

    def rebind_index(manifest: dict[str, object]) -> None:
        manifest["index"]["size_bytes"] = len(payload)
        manifest["index"]["file_sha256"] = hashlib.sha256(payload).hexdigest()

    _rewrite_manifest(destination, rebind_index)
    with pytest.raises(CounterfactualV2StorageError, match="sample_index"):
        audit_counterfactual_v2_store(destination)


@pytest.mark.parametrize("relative", ["offline_labels/actual_return.npy", "index.jsonl"])
def test_missing_required_file_is_rejected(tmp_path: Path, relative: str) -> None:
    destination = tmp_path / "dataset"
    _write(destination)
    (destination / relative).unlink()

    with pytest.raises(CounterfactualV2StorageError, match="entry mismatch"):
        audit_counterfactual_v2_store(destination)


def test_unexpected_file_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    _write(destination)
    (destination / "partial.tmp").write_text("unexpected", encoding="utf-8")

    with pytest.raises(CounterfactualV2StorageError, match="entry mismatch"):
        audit_counterfactual_v2_store(destination)


def test_writer_rejects_unvalidated_values_before_creating_destination(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["deployment_inputs"] = {
        "unsafe": np.asarray([{"q_true": 1}], dtype=object)
    }
    with pytest.raises(CounterfactualV2StorageError, match="object dtype"):
        write_counterfactual_v2_store(tmp_path / "object", **payload)

    payload = _payload()
    payload["provenance"] = {"checkpoint": Path("not-json")}
    with pytest.raises(CounterfactualV2StorageError, match="non-JSON"):
        write_counterfactual_v2_store(tmp_path / "metadata", **payload)

    assert not (tmp_path / "object").exists()
    assert not (tmp_path / "metadata").exists()


def test_duplicate_index_rows_are_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["index_rows"] = [payload["index_rows"][0]] * 2

    with pytest.raises(CounterfactualV2StorageError, match="duplicate"):
        write_counterfactual_v2_store(tmp_path / "dataset", **payload)


def test_array_leading_axis_must_match_index_rows(tmp_path: Path) -> None:
    payload = _payload()
    payload["deployment_inputs"]["task_index"] = np.asarray([1], dtype=np.uint8)

    with pytest.raises(CounterfactualV2StorageError, match="leading sample axis"):
        write_counterfactual_v2_store(tmp_path / "dataset", **payload)


def test_load_rejects_writeable_mmap_mode(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    _write(destination)

    with pytest.raises(CounterfactualV2StorageError, match="read-only"):
        load_counterfactual_v2_store(destination, mmap_mode="r+")
