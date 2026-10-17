"""Compact storage for episode/grounding policy-prefix bundles."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STORAGE_FORMAT = "fcpg_lewm_factorial_prefix_bundle"
SCHEMA_VERSION = 1
ARRAY_NAMES = ("actions", "native_tokens", "sampled_visual")
REQUIRED_PROVENANCE = frozenset(
    ("protocol", "git_commit", "config_sha256", "policy_checkpoint_sha256")
)
_GROUNDING = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_KEYS = frozenset(
    ("storage_format", "schema_version", "episode", "grounding", "arrays",
     "provenance", "bundle_sha256", "manifest_sha256")
)
_SEMANTIC_KEYS = tuple(_MANIFEST_KEYS - {"bundle_sha256", "manifest_sha256"})


class PrefixBundleError(ValueError):
    pass


class PrefixBundleIntegrityError(PrefixBundleError):
    pass


@dataclass(frozen=True, slots=True)
class PrefixBundle:
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, object]
    bundle_sha256: str
    manifest_sha256: str

    @property
    def actions(self) -> np.ndarray:
        return self.arrays["actions"]

    @property
    def native_tokens(self) -> np.ndarray:
        return self.arrays["native_tokens"]

    @property
    def sampled_visual(self) -> np.ndarray:
        return self.arrays["sampled_visual"]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash logical C-order bytes, independent of NPZ encoding."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise PrefixBundleError("object arrays are forbidden")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _provenance(value: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PrefixBundleError("provenance must be a string-keyed mapping")
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise PrefixBundleError("provenance must contain finite JSON values") from exc
    missing = REQUIRED_PROVENANCE - set(normalized)
    if missing:
        raise PrefixBundleError(f"provenance is missing fields: {sorted(missing)}")
    for name in ("protocol", "git_commit"):
        if not isinstance(normalized[name], str) or not normalized[name]:
            raise PrefixBundleError(f"provenance.{name} must be non-empty")
    for name in ("config_sha256", "policy_checkpoint_sha256"):
        digest = normalized[name]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise PrefixBundleError(f"provenance.{name} must be a SHA-256 digest")
    return normalized


def _identity(episode: int, grounding: str) -> tuple[int, str]:
    if isinstance(episode, bool) or not isinstance(episode, (int, np.integer)):
        raise PrefixBundleError("episode must be an integer")
    if int(episode) < 0:
        raise PrefixBundleError("episode must be non-negative")
    if not isinstance(grounding, str) or _GROUNDING.fullmatch(grounding) is None:
        raise PrefixBundleError("grounding is not a safe identifier")
    return int(episode), grounding


def _paths(root: Path, episode: int, grounding: str) -> tuple[Path, Path]:
    directory = Path(root).expanduser().absolute() / f"episode_{episode:03d}"
    return directory / f"{grounding}.npz", directory / f"{grounding}.json"


def _validated_arrays(
    actions: np.ndarray, native_tokens: np.ndarray, sampled_visual: np.ndarray
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in zip(
        ARRAY_NAMES, (actions, native_tokens, sampled_visual), strict=True
    ):
        if not isinstance(value, np.ndarray):
            raise PrefixBundleError(f"{name} must be a numpy ndarray")
        if value.dtype.hasobject or value.dtype.kind not in "biufc":
            raise PrefixBundleError(f"{name} must use a numeric dtype")
        result[name] = np.array(value, copy=True, order="C")
    return result


def _array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "shape": list(arrays[name].shape),
            "dtype": arrays[name].dtype.str,
            "sha256": array_sha256(arrays[name]),
        }
        for name in ARRAY_NAMES
    }


def _semantic(manifest: Mapping[str, object]) -> dict[str, object]:
    return {key: manifest[key] for key in _SEMANTIC_KEYS}


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return _json_sha256(payload)


def _build_manifest(
    episode: int,
    grounding: str,
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "storage_format": STORAGE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "episode": episode,
        "grounding": grounding,
        "arrays": _array_records(arrays),
        "provenance": dict(provenance),
    }
    manifest["bundle_sha256"] = _json_sha256(_semantic(manifest))
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


def _load(
    data_path: Path,
    manifest_path: Path,
    episode: int,
    grounding: str,
    expected_provenance: Mapping[str, object] | None,
) -> PrefixBundle:
    if not data_path.exists() and not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not data_path.is_file() or not manifest_path.is_file():
        raise PrefixBundleIntegrityError("prefix bundle is incomplete")
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        if raw != _canonical_json(manifest) + b"\n":
            raise PrefixBundleIntegrityError("manifest is not canonical JSON")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PrefixBundleIntegrityError("manifest is not valid canonical JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise PrefixBundleIntegrityError("manifest fields mismatch")
    if (
        manifest["storage_format"] != STORAGE_FORMAT
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise PrefixBundleIntegrityError("manifest schema mismatch")
    if type(manifest["episode"]) is not int or manifest["episode"] != episode:
        raise PrefixBundleIntegrityError("manifest episode mismatch")
    if manifest["grounding"] != grounding:
        raise PrefixBundleIntegrityError("manifest grounding mismatch")
    try:
        stored_provenance = _provenance(manifest["provenance"])
    except PrefixBundleError as exc:
        raise PrefixBundleIntegrityError(str(exc)) from exc
    if stored_provenance != manifest["provenance"]:
        raise PrefixBundleIntegrityError("manifest provenance mismatch")
    if manifest["bundle_sha256"] != _json_sha256(_semantic(manifest)):
        raise PrefixBundleIntegrityError("manifest bundle_sha256 mismatch")
    if manifest["manifest_sha256"] != _manifest_sha256(manifest):
        raise PrefixBundleIntegrityError("manifest manifest_sha256 mismatch")
    try:
        with np.load(io.BytesIO(data_path.read_bytes()), allow_pickle=False) as archive:
            if len(archive.files) != 3 or set(archive.files) != set(ARRAY_NAMES):
                raise PrefixBundleIntegrityError("prefix NPZ keys mismatch")
            arrays = _validated_arrays(*(archive[name] for name in ARRAY_NAMES))
    except PrefixBundleIntegrityError:
        raise
    except (OSError, ValueError, PrefixBundleError) as exc:
        raise PrefixBundleIntegrityError("prefix NPZ cannot be loaded safely") from exc
    if _array_records(arrays) != manifest["arrays"]:
        raise PrefixBundleIntegrityError("prefix NPZ shape/dtype/SHA-256 mismatch")
    if expected_provenance is not None and _provenance(expected_provenance) != stored_provenance:
        raise PrefixBundleIntegrityError("prefix bundle provenance mismatch")
    for array in arrays.values():
        array.setflags(write=False)
    return PrefixBundle(
        arrays,
        manifest,
        str(manifest["bundle_sha256"]),
        str(manifest["manifest_sha256"]),
    )


def write_prefix_bundle(
    root: Path,
    *,
    episode: int,
    grounding: str,
    actions: np.ndarray,
    native_tokens: np.ndarray,
    sampled_visual: np.ndarray,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Atomically write, or idempotently reuse, one prefix bundle."""

    episode, grounding = _identity(episode, grounding)
    arrays = _validated_arrays(actions, native_tokens, sampled_visual)
    provenance = _provenance(provenance)
    candidate = _build_manifest(episode, grounding, arrays, provenance)
    data_path, manifest_path = _paths(root, episode, grounding)
    if data_path.exists() or manifest_path.exists():
        existing = _load(data_path, manifest_path, episode, grounding, None)
        if existing.bundle_sha256 == candidate["bundle_sha256"]:
            return dict(existing.manifest)
        raise FileExistsError(
            "refusing to overwrite prefix bundle with different content"
        )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_fd, data_name = tempfile.mkstemp(dir=data_path.parent, suffix=".npz")
    json_fd, json_name = tempfile.mkstemp(dir=data_path.parent, suffix=".json")
    os.close(data_fd)
    os.close(json_fd)
    data_temp, json_temp = Path(data_name), Path(json_name)
    try:
        np.savez_compressed(data_temp, **arrays)
        json_temp.write_bytes(_canonical_json(candidate) + b"\n")
        os.replace(data_temp, data_path)
        os.replace(json_temp, manifest_path)
        return candidate
    finally:
        data_temp.unlink(missing_ok=True)
        json_temp.unlink(missing_ok=True)


def load_prefix_bundle(
    root: Path,
    *,
    episode: int,
    grounding: str,
    expected_provenance: Mapping[str, object] | None = None,
) -> PrefixBundle:
    episode, grounding = _identity(episode, grounding)
    paths = _paths(root, episode, grounding)
    return _load(*paths, episode, grounding, expected_provenance)
