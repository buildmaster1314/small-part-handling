"""Atomic, content-addressed storage for Counterfactual v2 datasets.

The storage layer is deliberately schema-agnostic.  Callers must validate the
scientific contract before publication and pass three disjoint ndarray
mappings.  This module only guarantees immutable layout, canonical metadata,
and fail-closed integrity checks without importing model or simulator code.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


STORAGE_FORMAT = "fcpg_lewm_counterfactual_v2_array_store"
STORAGE_SCHEMA_VERSION = 2
SEMANTIC_ARRAY_HASH = "sha256(dtype.str+shape.json+c_order_bytes)-v1"
ARRAY_GROUPS = ("deployment_inputs", "offline_labels", "audit")
_ARRAY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_HASH_CHUNK_BYTES = 1024 * 1024
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class CounterfactualV2StorageError(ValueError):
    """Raised when a v2 store is malformed or fails integrity checks."""


@dataclass(frozen=True)
class CounterfactualV2Audit:
    """Summary returned after a complete on-disk integrity audit."""

    root: Path
    result_sha256: str
    index_row_count: int
    array_count: int


@dataclass(frozen=True)
class CounterfactualV2Dataset:
    """Audited arrays and metadata loaded from an immutable v2 store."""

    root: Path
    deployment_inputs: Mapping[str, np.ndarray]
    offline_labels: Mapping[str, np.ndarray]
    audit_arrays: Mapping[str, np.ndarray]
    index_rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and logical C-order array bytes without a large copy."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise CounterfactualV2StorageError("object arrays are forbidden")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    buffer_items = max(1, _HASH_CHUNK_BYTES // max(1, array.dtype.itemsize))
    iterator = np.nditer(
        array,
        flags=("external_loop", "buffered", "zerosize_ok"),
        op_flags=("readonly",),
        order="C",
        buffersize=buffer_items,
    )
    for chunk in iterator:
        digest.update(np.asarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def _normalize_json(value: object, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CounterfactualV2StorageError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CounterfactualV2StorageError(
                    f"{label} contains a non-string object key"
                )
            normalized[key] = _normalize_json(item, f"{label}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise CounterfactualV2StorageError(
        f"{label} contains a non-JSON value of type {type(value).__name__}"
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npy_file(path: Path, value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise CounterfactualV2StorageError(f"object array is forbidden: {path.name}")
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    stored = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        semantic = semantic_array_sha256(stored)
        dtype = stored.dtype.str
        shape = [int(size) for size in stored.shape]
    finally:
        del stored
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "dtype": dtype,
        "shape": shape,
        "size_bytes": int(stat.st_size),
        "file_sha256": _sha256_file(path),
        "semantic_sha256": semantic,
    }


def _validated_array_mapping(
    values: Mapping[str, np.ndarray], label: str
) -> tuple[tuple[str, np.ndarray], ...]:
    if not isinstance(values, Mapping) or not values:
        raise CounterfactualV2StorageError(f"{label} must be a non-empty mapping")
    rows = []
    for name, value in values.items():
        if not isinstance(name, str) or _ARRAY_NAME.fullmatch(name) is None:
            raise CounterfactualV2StorageError(
                f"invalid {label} array name: {name!r}"
            )
        if not isinstance(value, np.ndarray):
            raise CounterfactualV2StorageError(
                f"{label}.{name} must already be a numpy ndarray"
            )
        if value.dtype.hasobject:
            raise CounterfactualV2StorageError(
                f"{label}.{name} uses forbidden object dtype"
            )
        rows.append((name, value))
    return tuple(sorted(rows, key=lambda item: item[0]))


def _canonical_index_rows(
    index_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], bytes]:
    if not isinstance(index_rows, Sequence) or isinstance(
        index_rows, (str, bytes, bytearray)
    ):
        raise CounterfactualV2StorageError("index_rows must be a sequence")
    normalized = []
    for index, row in enumerate(index_rows):
        if not isinstance(row, Mapping):
            raise CounterfactualV2StorageError(
                f"index_rows[{index}] must be a JSON object"
            )
        normalized.append(dict(_normalize_json(row, f"index_rows[{index}]")))
    if not normalized:
        raise CounterfactualV2StorageError("index_rows must not be empty")
    encoded = [_canonical_json_bytes(row) for row in normalized]
    if len(encoded) != len(set(encoded)):
        raise CounterfactualV2StorageError("index_rows contains duplicate rows")
    for index, row in enumerate(normalized):
        sample_index = row.get("sample_index")
        if isinstance(sample_index, bool) or sample_index != index:
            raise CounterfactualV2StorageError(
                "index_rows sample_index must equal its array-aligned row position"
            )
    payload = b"\n".join(encoded) + b"\n"
    rows = tuple(json.loads(line) for line in encoded)
    return rows, payload


def _validate_sample_axis(
    groups: Mapping[str, Sequence[tuple[str, np.ndarray]]],
    row_count: int,
) -> None:
    for group, entries in groups.items():
        for name, value in entries:
            if value.ndim == 0 or int(value.shape[0]) != row_count:
                raise CounterfactualV2StorageError(
                    f"{group}.{name} leading sample axis must equal index row_count"
                )


def _manifest_result_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("result_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _path_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _linux_rename_noreplace(source: Path, destination: Path) -> bool:
    """Use renameat2 when available; return False when unsupported."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        return False
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        return False
    if error == errno.EEXIST:
        raise FileExistsError(destination)
    raise OSError(error, os.strerror(error), destination)


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    if _linux_rename_noreplace(source, destination):
        return
    # A check followed by plain rename(2) is a TOCTOU race: another collector
    # can publish after the check and POSIX rename may then replace its
    # directory.  Fail closed on platforms without atomic no-replace rename
    # support instead of weakening the immutable-publication contract.
    raise CounterfactualV2StorageError(
        "atomic no-replace directory publication is unsupported"
    )


def write_counterfactual_v2_store(
    destination: str | os.PathLike[str],
    *,
    deployment_inputs: Mapping[str, np.ndarray],
    offline_labels: Mapping[str, np.ndarray],
    audit_arrays: Mapping[str, np.ndarray],
    index_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one complete Counterfactual v2 array store.

    The caller owns semantic/schema validation.  This writer rejects generic
    Python values in array groups and never overwrites an existing destination.
    """

    destination_path = Path(destination).expanduser().absolute()
    if not destination_path.name:
        raise CounterfactualV2StorageError("destination must name a directory")
    groups = {
        "deployment_inputs": _validated_array_mapping(
            deployment_inputs, "deployment_inputs"
        ),
        "offline_labels": _validated_array_mapping(offline_labels, "offline_labels"),
        "audit": _validated_array_mapping(audit_arrays, "audit"),
    }
    normalized_provenance = _normalize_json(provenance, "provenance")
    if not isinstance(normalized_provenance, dict) or not normalized_provenance:
        raise CounterfactualV2StorageError("provenance must be a non-empty object")
    normalized_index, index_payload = _canonical_index_rows(index_rows)
    _validate_sample_axis(groups, len(normalized_index))

    parent = destination_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(destination_path):
        raise FileExistsError(destination_path)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.staging-", dir=parent)
    )
    published = False
    try:
        array_manifest: dict[str, dict[str, Any]] = {}
        for group_name in ARRAY_GROUPS:
            group_path = staging / group_name
            group_path.mkdir()
            entries: dict[str, Any] = {}
            for name, value in groups[group_name]:
                relative = Path(group_name) / f"{name}.npy"
                record = _write_npy_file(staging / relative, value)
                record["path"] = relative.as_posix()
                entries[name] = record
            _fsync_directory(group_path)
            array_manifest[group_name] = entries

        index_path = staging / "index.jsonl"
        _write_bytes(index_path, index_payload)
        index_record = {
            "path": "index.jsonl",
            "row_count": int(index_payload.count(b"\n")),
            "size_bytes": len(index_payload),
            "file_sha256": hashlib.sha256(index_payload).hexdigest(),
        }
        manifest: dict[str, Any] = {
            "storage_format": STORAGE_FORMAT,
            "schema_version": STORAGE_SCHEMA_VERSION,
            "complete": True,
            "semantic_array_hash": SEMANTIC_ARRAY_HASH,
            "arrays": array_manifest,
            "index": index_record,
            "provenance": normalized_provenance,
        }
        manifest["result_sha256"] = _manifest_result_sha256(manifest)
        manifest_path = staging / "manifest.json"
        _write_bytes(manifest_path, _canonical_json_bytes(manifest) + b"\n")
        _fsync_directory(staging)
        _fsync_directory(parent)
        _publish_directory_no_replace(staging, destination_path)
        published = True
        _fsync_directory(parent)
        return manifest
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _load_canonical_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise CounterfactualV2StorageError("missing regular manifest.json")
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CounterfactualV2StorageError("manifest.json is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise CounterfactualV2StorageError("manifest.json root must be an object")
    if raw != _canonical_json_bytes(manifest) + b"\n":
        raise CounterfactualV2StorageError("manifest.json is not canonical")
    if manifest.get("storage_format") != STORAGE_FORMAT:
        raise CounterfactualV2StorageError("unexpected storage_format")
    if manifest.get("schema_version") != STORAGE_SCHEMA_VERSION:
        raise CounterfactualV2StorageError("unexpected storage schema_version")
    if manifest.get("complete") is not True:
        raise CounterfactualV2StorageError("store is not marked complete")
    if manifest.get("semantic_array_hash") != SEMANTIC_ARRAY_HASH:
        raise CounterfactualV2StorageError("unexpected semantic array hash contract")
    expected_result = _manifest_result_sha256(manifest)
    if manifest.get("result_sha256") != expected_result:
        raise CounterfactualV2StorageError("manifest result_sha256 mismatch")
    return manifest


def _load_canonical_index(
    root: Path, record: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    if record.get("path") != "index.jsonl":
        raise CounterfactualV2StorageError("index path must be index.jsonl")
    path = root / "index.jsonl"
    if path.is_symlink() or not path.is_file():
        raise CounterfactualV2StorageError("missing regular index.jsonl")
    raw = path.read_bytes()
    if len(raw) != record.get("size_bytes"):
        raise CounterfactualV2StorageError("index size mismatch")
    if hashlib.sha256(raw).hexdigest() != record.get("file_sha256"):
        raise CounterfactualV2StorageError("index file_sha256 mismatch")
    if not raw or not raw.endswith(b"\n") or b"\n\n" in raw:
        raise CounterfactualV2StorageError("index.jsonl has invalid line framing")
    lines = raw[:-1].split(b"\n")
    if len(lines) != record.get("row_count"):
        raise CounterfactualV2StorageError("index row_count mismatch")
    if len(lines) != len(set(lines)):
        raise CounterfactualV2StorageError("index.jsonl contains duplicate rows")
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CounterfactualV2StorageError("index.jsonl contains invalid JSON") from exc
        if not isinstance(row, dict) or line != _canonical_json_bytes(row):
            raise CounterfactualV2StorageError(
                "index.jsonl contains a noncanonical object"
            )
        sample_index = row.get("sample_index")
        if isinstance(sample_index, bool) or sample_index != index:
            raise CounterfactualV2StorageError(
                "index.jsonl sample_index does not match its array-aligned row position"
            )
        rows.append(MappingProxyType(row))
    return tuple(rows)


def _validate_array_record(
    root: Path,
    group: str,
    name: str,
    record: Mapping[str, Any],
    *,
    mmap_mode: str | None,
) -> np.ndarray:
    if _ARRAY_NAME.fullmatch(name) is None:
        raise CounterfactualV2StorageError(f"invalid array name in manifest: {name!r}")
    expected_path = Path(group) / f"{name}.npy"
    if record.get("path") != expected_path.as_posix():
        raise CounterfactualV2StorageError(f"unexpected array path for {group}.{name}")
    path = root / expected_path
    if path.is_symlink() or not path.is_file():
        raise CounterfactualV2StorageError(f"missing regular array file: {expected_path}")
    stat = path.stat()
    if int(stat.st_size) != record.get("size_bytes"):
        raise CounterfactualV2StorageError(f"array size mismatch: {expected_path}")
    if _sha256_file(path) != record.get("file_sha256"):
        raise CounterfactualV2StorageError(f"array file_sha256 mismatch: {expected_path}")
    try:
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CounterfactualV2StorageError(
            f"array cannot be loaded safely: {expected_path}"
        ) from exc
    if array.dtype.hasobject:
        raise CounterfactualV2StorageError(f"object array is forbidden: {expected_path}")
    if array.dtype.str != record.get("dtype"):
        raise CounterfactualV2StorageError(f"array dtype mismatch: {expected_path}")
    if [int(size) for size in array.shape] != record.get("shape"):
        raise CounterfactualV2StorageError(f"array shape mismatch: {expected_path}")
    if semantic_array_sha256(array) != record.get("semantic_sha256"):
        raise CounterfactualV2StorageError(
            f"array semantic_sha256 mismatch: {expected_path}"
        )
    return array


def _expected_entries(manifest: Mapping[str, Any]) -> set[str]:
    expected = {"manifest.json", "index.jsonl", *ARRAY_GROUPS}
    groups = manifest.get("arrays")
    if not isinstance(groups, Mapping) or set(groups) != set(ARRAY_GROUPS):
        raise CounterfactualV2StorageError("manifest arrays must contain exact groups")
    for group in ARRAY_GROUPS:
        records = groups[group]
        if not isinstance(records, Mapping) or not records:
            raise CounterfactualV2StorageError(f"manifest group {group} is empty")
        expected.update(f"{group}/{name}.npy" for name in records)
    return expected


def _actual_entries(root: Path) -> set[str]:
    entries = set()
    for path in root.rglob("*"):
        entries.add(path.relative_to(root).as_posix())
    return entries


def _audit_and_load(
    root: Path, *, mmap_mode: str | None
) -> tuple[
    dict[str, Any],
    tuple[Mapping[str, Any], ...],
    dict[str, dict[str, np.ndarray]],
]:
    if root.is_symlink() or not root.is_dir():
        raise CounterfactualV2StorageError("store root must be a regular directory")
    manifest = _load_canonical_manifest(root)
    expected = _expected_entries(manifest)
    actual = _actual_entries(root)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise CounterfactualV2StorageError(
            f"store entry mismatch; missing={missing}, unexpected={unexpected}"
        )
    index_record = manifest.get("index")
    if not isinstance(index_record, Mapping):
        raise CounterfactualV2StorageError("manifest index must be an object")
    rows = _load_canonical_index(root, index_record)
    loaded: dict[str, dict[str, np.ndarray]] = {}
    groups = manifest["arrays"]
    for group in ARRAY_GROUPS:
        loaded[group] = {}
        for name, record in sorted(groups[group].items()):
            if not isinstance(record, Mapping):
                raise CounterfactualV2StorageError(
                    f"manifest record {group}.{name} must be an object"
                )
            loaded[group][name] = _validate_array_record(
                root,
                group,
                name,
                record,
                mmap_mode=mmap_mode,
            )
    _validate_sample_axis(
        {
            group: tuple(loaded[group].items())
            for group in ARRAY_GROUPS
        },
        len(rows),
    )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise CounterfactualV2StorageError("manifest provenance must be non-empty")
    return manifest, rows, loaded


def audit_counterfactual_v2_store(
    root: str | os.PathLike[str],
) -> CounterfactualV2Audit:
    """Verify the complete store, including every file and semantic array hash."""

    root_path = Path(root).expanduser().absolute()
    manifest, rows, arrays = _audit_and_load(root_path, mmap_mode="r")
    return CounterfactualV2Audit(
        root=root_path,
        result_sha256=str(manifest["result_sha256"]),
        index_row_count=len(rows),
        array_count=sum(len(group) for group in arrays.values()),
    )


def load_counterfactual_v2_store(
    root: str | os.PathLike[str],
    *,
    mmap_mode: str | None = "r",
) -> CounterfactualV2Dataset:
    """Audit and load a complete store; no unchecked loading path is exposed."""

    if mmap_mode not in {None, "r"}:
        raise CounterfactualV2StorageError("mmap_mode must be None or read-only 'r'")
    root_path = Path(root).expanduser().absolute()
    manifest, rows, arrays = _audit_and_load(root_path, mmap_mode=mmap_mode)
    return CounterfactualV2Dataset(
        root=root_path,
        deployment_inputs=MappingProxyType(arrays["deployment_inputs"]),
        offline_labels=MappingProxyType(arrays["offline_labels"]),
        audit_arrays=MappingProxyType(arrays["audit"]),
        index_rows=rows,
        provenance=MappingProxyType(dict(manifest["provenance"])),
        manifest=MappingProxyType(manifest),
    )


__all__ = [
    "ARRAY_GROUPS",
    "CounterfactualV2Audit",
    "CounterfactualV2Dataset",
    "CounterfactualV2StorageError",
    "SEMANTIC_ARRAY_HASH",
    "STORAGE_FORMAT",
    "STORAGE_SCHEMA_VERSION",
    "audit_counterfactual_v2_store",
    "load_counterfactual_v2_store",
    "semantic_array_sha256",
    "write_counterfactual_v2_store",
]
