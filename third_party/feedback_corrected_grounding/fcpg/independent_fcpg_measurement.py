"""Seal schema-3 FCPG caches as independently generated measurement views.

The seal is deliberately a provenance-only operation: it never recomputes or
changes corrected coordinates.  Independence is bound to the seed stored in
the producing corrector checkpoint, rather than trusting a seed copied into a
JSON file or supplied only on the command line.
"""

from __future__ import annotations

import errno
import json
import numbers
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_sequence_data import frame_identity_sha256


INDEPENDENT_MEASUREMENT_CONTRACT = "independent_fcpg_measurement_v1"
_VIEW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_COPY_FALLBACK_ERRNOS = {
    errno.EXDEV,
    errno.EPERM,
    errno.EACCES,
    getattr(errno, "ENOTSUP", errno.EPERM),
}


@dataclass(frozen=True)
class ValidatedFCPGCache:
    cache_dir: Path
    metadata_path: Path
    metadata: dict[str, Any]
    metadata_sha256: str
    q_corrected_path: Path
    q_corrected_sha256: str
    source_data_path: Path
    source_data_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_seed: int
    reliability_path: Path | None
    reliability_sha256: str | None


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _resolve_metadata_path(value: object, metadata_path: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = metadata_path.parent / path
    return path.resolve()


def _validated_seed(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{label} must be an integer")
    seed = int(value)
    if seed < 0 or seed > (2**63 - 1):
        raise ValueError(f"{label} must be in [0, 2^63-1]")
    return seed


def _checkpoint_seed(checkpoint_path: Path) -> int:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(
            "corrector checkpoint must be readable with torch.load(weights_only=True)"
        ) from exc
    checkpoint_mapping = _mapping(checkpoint, "corrector checkpoint")
    args = _mapping(checkpoint_mapping.get("args"), "corrector checkpoint args")
    return _validated_seed(args.get("seed"), "corrector checkpoint args.seed")


def validate_schema3_fcpg_cache(cache_dir: str | Path) -> ValidatedFCPGCache:
    """Validate all immutable bindings required by a schema-3 correction cache."""

    cache = Path(cache_dir).expanduser().resolve()
    if not cache.is_dir():
        raise FileNotFoundError(f"FCPG correction cache is not a directory: {cache}")
    metadata_path = cache / "metadata.json"
    q_path = cache / "q_corrected.npy"
    if not metadata_path.is_file() or not q_path.is_file():
        raise FileNotFoundError(
            "FCPG correction cache requires metadata.json and q_corrected.npy"
        )
    metadata = dict(
        _mapping(
            json.loads(metadata_path.read_text(encoding="utf-8")),
            "FCPG correction metadata",
        )
    )
    if int(metadata.get("schema_version", 0)) != 3:
        raise ValueError("independent measurements require an exact schema-3 cache")
    if metadata.get("q_true_usage") != "offline_metrics_only":
        raise ValueError("q_true_usage must be exactly offline_metrics_only")

    source_path = _resolve_metadata_path(
        metadata.get("source_data"), metadata_path, "source_data"
    )
    checkpoint_path = _resolve_metadata_path(
        metadata.get("checkpoint"), metadata_path, "checkpoint"
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"source dataset is missing: {source_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"corrector checkpoint is missing: {checkpoint_path}")

    source_sha = sha256_file(source_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    q_sha = sha256_file(q_path)
    if metadata.get("source_data_sha256") != source_sha:
        raise ValueError("source_data_sha256 differs from the source dataset")
    if metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("checkpoint_sha256 differs from the corrector checkpoint")
    if metadata.get("q_corrected_sha256") != q_sha:
        raise ValueError("q_corrected_sha256 differs from q_corrected.npy")

    with np.load(source_path, allow_pickle=False) as source:
        if "episode_id" not in source.files or "frame_id" not in source.files:
            raise ValueError("source dataset lacks episode_id/frame_id identity fields")
        episode_id = np.asarray(source["episode_id"])
        frame_id = np.asarray(source["frame_id"])
        if episode_id.ndim != 1 or frame_id.shape != episode_id.shape:
            raise ValueError("source episode_id/frame_id must be aligned 1-D arrays")
    expected_frame_sha = frame_identity_sha256(episode_id, frame_id)
    if metadata.get("frame_identity_sha256") != expected_frame_sha:
        raise ValueError("frame_identity_sha256 differs from the source dataset")

    corrected = np.load(q_path, mmap_mode="r", allow_pickle=False)
    sample_count = int(metadata.get("sample_count", -1))
    num_keypoints = int(metadata.get("num_keypoints", -1))
    expected_shape = (sample_count, num_keypoints, 2)
    if sample_count <= 0 or num_keypoints <= 0 or corrected.shape != expected_shape:
        raise ValueError(
            f"q_corrected shape {corrected.shape} does not match metadata "
            f"{expected_shape}"
        )
    if sample_count != int(episode_id.shape[0]):
        raise ValueError("sample_count differs from the source frame count")
    if not np.issubdtype(corrected.dtype, np.floating):
        raise ValueError("q_corrected must use a floating dtype")
    if not bool(np.isfinite(corrected).all()):
        raise ValueError("q_corrected contains non-finite coordinates")

    reliability_path: Path | None = None
    reliability_sha: str | None = None
    if bool(metadata.get("has_reliability", False)):
        reliability_path = cache / "reliability.npy"
        if not reliability_path.is_file():
            raise FileNotFoundError(
                "metadata declares reliability but reliability.npy is missing"
            )
        reliability = np.load(reliability_path, mmap_mode="r", allow_pickle=False)
        if reliability.shape != (sample_count, num_keypoints):
            raise ValueError("reliability shape does not match correction metadata")
        if not np.issubdtype(reliability.dtype, np.floating):
            raise ValueError("reliability must use a floating dtype")
        if not bool(np.isfinite(reliability).all()):
            raise ValueError("reliability contains non-finite values")
        reliability_sha = sha256_file(reliability_path)
        declared_reliability_sha = metadata.get("reliability_sha256")
        if (
            declared_reliability_sha is not None
            and declared_reliability_sha != reliability_sha
        ):
            raise ValueError("reliability_sha256 differs from reliability.npy")

    return ValidatedFCPGCache(
        cache_dir=cache,
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_sha256=sha256_file(metadata_path),
        q_corrected_path=q_path,
        q_corrected_sha256=q_sha,
        source_data_path=source_path,
        source_data_sha256=source_sha,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_seed=_checkpoint_seed(checkpoint_path),
        reliability_path=reliability_path,
        reliability_sha256=reliability_sha,
    )


def _transfer_file(source: Path, destination: Path, mode: str) -> str:
    if mode not in {"copy", "hardlink", "auto"}:
        raise ValueError("transfer mode must be copy, hardlink, or auto")
    if mode in {"hardlink", "auto"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError as exc:
            if mode == "hardlink" or exc.errno not in _COPY_FALLBACK_ERRNOS:
                raise
    shutil.copy2(source, destination)
    return "copy"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def seal_independent_fcpg_measurement(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    measurement_view_id: str,
    measurement_generation_seed: int,
    transfer: str = "copy",
) -> dict[str, Any]:
    """Publish a non-overwriting, checkpoint-seed-bound measurement view."""

    if not _VIEW_ID.fullmatch(measurement_view_id):
        raise ValueError(
            "measurement_view_id must use 1-128 ASCII letters, digits, '.', '_', or '-'"
        )
    requested_seed = _validated_seed(
        measurement_generation_seed, "measurement_generation_seed"
    )
    parent = validate_schema3_fcpg_cache(cache_dir)
    if parent.metadata.get("independent_measurement_contract") is not None:
        raise ValueError("an independent measurement view cannot be re-sealed")
    if parent.checkpoint_seed != requested_seed:
        raise ValueError(
            "measurement_generation_seed does not match corrector checkpoint "
            "args.seed: "
            f"requested {requested_seed}, checkpoint {parent.checkpoint_seed}"
        )

    output = Path(output_dir).expanduser().resolve()
    if output == parent.cache_dir:
        raise ValueError("refusing to seal a correction cache in place")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite existing output: {output}"
        ) from exc

    try:
        transfers: dict[str, str] = {}
        output_q = output / "q_corrected.npy"
        transfers["q_corrected.npy"] = _transfer_file(
            parent.q_corrected_path, output_q, transfer
        )
        if sha256_file(output_q) != parent.q_corrected_sha256:
            raise RuntimeError("transferred q_corrected.npy failed hash verification")

        if bool(parent.metadata.get("has_reliability", False)):
            assert parent.reliability_path is not None
            assert parent.reliability_sha256 is not None
            output_reliability = output / "reliability.npy"
            transfers["reliability.npy"] = _transfer_file(
                parent.reliability_path, output_reliability, transfer
            )
            if sha256_file(output_reliability) != parent.reliability_sha256:
                raise RuntimeError(
                    "transferred reliability.npy failed hash verification"
                )

        sealed = dict(parent.metadata)
        sealed.update(
            {
                "independent_measurement_contract": INDEPENDENT_MEASUREMENT_CONTRACT,
                "measurement_view_id": measurement_view_id,
                "measurement_generation_seed": requested_seed,
                "measurement_seed_evidence": "corrector_checkpoint.args.seed",
                "measurement_payload_transfer": transfers,
                "parent_cache": str(parent.cache_dir),
                "parent_metadata": str(parent.metadata_path),
                "parent_metadata_sha256": parent.metadata_sha256,
                "parent_q_corrected_sha256": parent.q_corrected_sha256,
                "parent_checkpoint_sha256": parent.checkpoint_sha256,
                "parent_source_data_sha256": parent.source_data_sha256,
                "parent_reliability_sha256": parent.reliability_sha256,
            }
        )
        if "reliability.npy" in transfers:
            sealed["reliability_sha256"] = sha256_file(
                output / "reliability.npy"
            )
        _atomic_write_json(output / "metadata.json", sealed)
        return sealed
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def validate_independent_fcpg_measurement_view(
    cache_dir: str | Path,
) -> ValidatedFCPGCache:
    """Validate a sealed view and its complete parent-cache binding."""

    sealed = validate_schema3_fcpg_cache(cache_dir)
    metadata = sealed.metadata
    if metadata.get("independent_measurement_contract") != (
        INDEPENDENT_MEASUREMENT_CONTRACT
    ):
        raise ValueError("cache is not an independent FCPG measurement view")
    view_id = metadata.get("measurement_view_id")
    if not isinstance(view_id, str) or not _VIEW_ID.fullmatch(view_id):
        raise ValueError("sealed measurement_view_id is invalid")
    declared_seed = _validated_seed(
        metadata.get("measurement_generation_seed"),
        "sealed measurement_generation_seed",
    )
    if declared_seed != sealed.checkpoint_seed:
        raise ValueError("sealed generation seed differs from checkpoint args.seed")
    if metadata.get("measurement_seed_evidence") != (
        "corrector_checkpoint.args.seed"
    ):
        raise ValueError("sealed measurement seed evidence contract is invalid")

    parent_path = _resolve_metadata_path(
        metadata.get("parent_cache"), sealed.metadata_path, "parent_cache"
    )
    if parent_path == sealed.cache_dir:
        raise ValueError("sealed measurement parent cannot be itself")
    parent = validate_schema3_fcpg_cache(parent_path)
    parent_metadata_path = _resolve_metadata_path(
        metadata.get("parent_metadata"), sealed.metadata_path, "parent_metadata"
    )
    if parent_metadata_path != parent.metadata_path:
        raise ValueError("parent_metadata path differs from the bound parent cache")
    if parent.metadata.get("independent_measurement_contract") is not None:
        raise ValueError("sealed measurement parent must be an unsealed schema-3 cache")
    expected = {
        "parent_metadata_sha256": parent.metadata_sha256,
        "parent_q_corrected_sha256": parent.q_corrected_sha256,
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_source_data_sha256": parent.source_data_sha256,
        "parent_reliability_sha256": parent.reliability_sha256,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"{field} differs from the bound parent cache")
    if sealed.q_corrected_sha256 != parent.q_corrected_sha256:
        raise ValueError("sealed q_corrected payload differs from its parent")
    if sealed.checkpoint_seed != parent.checkpoint_seed:
        raise ValueError("sealed checkpoint seed differs from its parent")
    return sealed
