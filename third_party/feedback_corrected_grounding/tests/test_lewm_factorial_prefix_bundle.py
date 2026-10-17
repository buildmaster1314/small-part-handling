from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_factorial_prefix_bundle import (
    PrefixBundleError,
    PrefixBundleIntegrityError,
    load_prefix_bundle,
    write_prefix_bundle,
)


def _payload() -> dict[str, object]:
    return {
        "actions": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "native_tokens": np.arange(40, dtype=np.float16).reshape(2, 5, 4),
        "sampled_visual": np.arange(56, dtype=np.float32).reshape(2, 7, 4),
        "provenance": {
            "protocol": "factorial-eval-v1",
            "git_commit": "b" * 40,
            "config_sha256": "c" * 64,
            "policy_checkpoint_sha256": "a" * 64,
            "seed": 17,
        },
    }


def _write(root: Path, **updates: object) -> dict[str, object]:
    payload = _payload()
    payload.update(updates)
    return write_prefix_bundle(
        root,
        episode=3,
        grounding="fcpg",
        actions=payload["actions"],
        native_tokens=payload["native_tokens"],
        sampled_visual=payload["sampled_visual"],
        provenance=payload["provenance"],
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def test_round_trip_records_semantic_hashes_and_provenance(tmp_path: Path) -> None:
    manifest = _write(tmp_path)
    episode_dir = tmp_path / "episode_003"
    assert {path.name for path in episode_dir.iterdir()} == {"fcpg.npz", "fcpg.json"}
    assert manifest["episode"] == 3
    assert manifest["grounding"] == "fcpg"
    assert manifest["provenance"] == _payload()["provenance"]

    for name in ("actions", "native_tokens", "sampled_visual"):
        value = _payload()[name]
        record = manifest["arrays"][name]
        assert record["shape"] == list(value.shape)
        assert record["dtype"] == value.dtype.str
        assert record["sha256"] == hashlib.sha256(
            np.ascontiguousarray(value).tobytes(order="C")
        ).hexdigest()

    without_manifest_hash = dict(manifest)
    without_manifest_hash.pop("manifest_sha256")
    assert manifest["manifest_sha256"] == hashlib.sha256(
        _canonical(without_manifest_hash)
    ).hexdigest()
    loaded = load_prefix_bundle(
        tmp_path,
        episode=3,
        grounding="fcpg",
        expected_provenance=_payload()["provenance"],
    )
    assert loaded.bundle_sha256 == manifest["bundle_sha256"]
    assert loaded.manifest_sha256 == manifest["manifest_sha256"]
    np.testing.assert_array_equal(loaded.actions, _payload()["actions"])
    np.testing.assert_array_equal(
        loaded.native_tokens, _payload()["native_tokens"]
    )
    np.testing.assert_array_equal(
        loaded.sampled_visual, _payload()["sampled_visual"]
    )
    assert all(not value.flags.writeable for value in loaded.arrays.values())


def test_same_content_is_idempotent_and_different_content_is_rejected(
    tmp_path: Path,
) -> None:
    first = _write(tmp_path)
    data_path = tmp_path / "episode_003/fcpg.npz"
    manifest_path = tmp_path / "episode_003/fcpg.json"
    before = (data_path.read_bytes(), manifest_path.read_bytes())
    assert _write(tmp_path) == first
    assert (data_path.read_bytes(), manifest_path.read_bytes()) == before

    changed = _payload()["actions"].copy()
    changed[0, 0, 0] = -1.0
    with pytest.raises(FileExistsError, match="different content"):
        _write(tmp_path, actions=changed)
    changed_provenance = _payload()["provenance"]
    changed_provenance["config_sha256"] = "d" * 64
    with pytest.raises(FileExistsError, match="different content"):
        _write(tmp_path, provenance=changed_provenance)
    assert (data_path.read_bytes(), manifest_path.read_bytes()) == before


def test_npz_encoding_is_not_a_semantic_digest(tmp_path: Path) -> None:
    manifest = _write(tmp_path)
    path = tmp_path / "episode_003/fcpg.npz"
    original_encoding = path.read_bytes()
    payload = _payload()
    np.savez(
        path,
        actions=payload["actions"],
        native_tokens=payload["native_tokens"],
        sampled_visual=payload["sampled_visual"],
    )
    assert path.read_bytes() != original_encoding
    loaded = load_prefix_bundle(tmp_path, episode=3, grounding="fcpg")
    assert loaded.bundle_sha256 == manifest["bundle_sha256"]


def test_array_or_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path)
    data_path = tmp_path / "episode_003/fcpg.npz"
    payload = _payload()
    changed = payload["sampled_visual"].copy()
    changed.flat[0] += 1.0
    np.savez_compressed(
        data_path,
        actions=payload["actions"],
        native_tokens=payload["native_tokens"],
        sampled_visual=changed,
    )
    with pytest.raises(PrefixBundleIntegrityError, match="shape/dtype/SHA-256 mismatch"):
        load_prefix_bundle(tmp_path, episode=3, grounding="fcpg")

    _write(tmp_path / "other")
    manifest_path = tmp_path / "other/episode_003/fcpg.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_sha256"] = "0" * 64
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    with pytest.raises(PrefixBundleIntegrityError, match="manifest_sha256 mismatch"):
        load_prefix_bundle(tmp_path / "other", episode=3, grounding="fcpg")


def test_expected_provenance_is_enforced(tmp_path: Path) -> None:
    _write(tmp_path)
    expected = _payload()["provenance"]
    expected["config_sha256"] = "d" * 64
    with pytest.raises(PrefixBundleIntegrityError, match="provenance mismatch"):
        load_prefix_bundle(
            tmp_path,
            episode=3,
            grounding="fcpg",
            expected_provenance=expected,
        )


def test_invalid_inputs_are_rejected_before_publication(tmp_path: Path) -> None:
    with pytest.raises(PrefixBundleError, match="grounding"):
        write_prefix_bundle(
            tmp_path,
            episode=0,
            grounding="../fcpg",
            **_payload(),
        )
    payload = _payload()
    payload["actions"] = np.asarray([{"unsafe": True}], dtype=object)
    with pytest.raises(PrefixBundleError, match="numeric dtype"):
        _write(tmp_path, actions=payload["actions"])
    provenance = dict(_payload()["provenance"])
    provenance.pop("git_commit")
    with pytest.raises(PrefixBundleError, match="missing fields"):
        _write(tmp_path, provenance=provenance)
    assert not list(tmp_path.glob("episode_*"))
