"""Create and validate frozen input manifests for grounded LeWM training."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024
GIT_DIFF_FORMAT = "git-diff-head-binary-full-index-v1"


class ManifestError(ValueError):
    """Base error for manifest creation and schema failures."""


class ManifestValidationError(ManifestError):
    """Raised when one or more frozen inputs no longer match a manifest."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        details = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"manifest validation failed:\n{details}")


def sha256_file(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Return a SHA-256 digest without loading the complete file into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_path(
    path: str | os.PathLike[str],
    *,
    base: str | os.PathLike[str] | None = None,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base) / candidate if base is not None else Path.cwd() / candidate
    # Preserve the final symlink so validation checks the configured path itself.
    return Path(os.path.abspath(os.fspath(candidate)))


def _run_git(repo_root: Path, *args: str) -> bytes:
    command = ["git", "-C", os.fspath(repo_root), *args]
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "LANG": "C", "LC_ALL": "C"})
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except FileNotFoundError as exc:
        raise ManifestError("git executable was not found") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        message = stderr or f"git command failed with exit code {exc.returncode}"
        raise ManifestError(message) from exc
    return result.stdout


def discover_repository_root(start: str | os.PathLike[str]) -> Path:
    """Find the Git worktree root containing ``start``."""

    candidate = _absolute_path(start)
    if candidate.is_file():
        candidate = candidate.parent
    output = _run_git(candidate, "rev-parse", "--show-toplevel")
    root = output.decode("utf-8", errors="strict").strip()
    if not root:
        raise ManifestError(f"git returned an empty repository root for {candidate}")
    return Path(root).resolve()


def capture_git_state(repo_root: str | os.PathLike[str]) -> dict[str, object]:
    """Capture HEAD and a content fingerprint of tracked and untracked changes."""

    root = discover_repository_root(repo_root)
    head = _run_git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    diff = _run_git(
        root,
        "-c",
        "color.ui=false",
        "-c",
        "core.quotePath=true",
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "HEAD",
        "--",
    )
    raw_untracked = _run_git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked_files: list[dict[str, object]] = []
    for raw_relative in sorted(part for part in raw_untracked.split(b"\0") if part):
        relative = os.fsdecode(raw_relative)
        path = root / relative
        if path.is_symlink():
            payload = os.fsencode(os.readlink(path))
            digest = hashlib.sha256(payload).hexdigest()
            size = len(payload)
            kind = "symlink"
        elif path.is_file():
            digest = sha256_file(path)
            size = path.stat().st_size
            kind = "file"
        else:
            raise ManifestError(f"unsupported untracked input: {path}")
        untracked_files.append(
            {
                "path": Path(relative).as_posix(),
                "kind": kind,
                "size": size,
                "sha256": digest,
            }
        )
    tracked_diff_sha256 = hashlib.sha256(diff).hexdigest()
    worktree_payload = json.dumps(
        {
            "tracked_diff_sha256": tracked_diff_sha256,
            "untracked_files": untracked_files,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "head": head,
        "dirty": bool(diff or untracked_files),
        "dirty_diff_format": GIT_DIFF_FORMAT,
        "dirty_diff_sha256": tracked_diff_sha256,
        "untracked_files": untracked_files,
        "worktree_sha256": hashlib.sha256(worktree_payload).hexdigest(),
    }


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, object]:
    """Load and resolve an OmegaConf YAML file into plain Python containers."""

    config_path = _absolute_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ManifestError("OmegaConf is required to read the LeWM YAML config") from exc

    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ManifestError("the YAML root must be a mapping")
    return resolved


def extract_sources(config: Mapping[str, object]) -> list[dict[str, str]]:
    """Extract normalized LeWM source entries from ``data.sources``."""

    data = config.get("data")
    raw_sources: object
    if isinstance(data, Mapping) and "sources" in data:
        raw_sources = data["sources"]
    else:
        raw_sources = config.get("sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise ManifestError("the YAML must define a data.sources sequence")
    if not raw_sources:
        raise ManifestError("data.sources must contain at least one source")

    sources: list[dict[str, str]] = []
    names: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, Mapping):
            raise ManifestError(f"data.sources[{index}] must be a mapping")
        data_path = raw_source.get("path")
        correction_cache = raw_source.get("correction_cache")
        if data_path is None:
            raise ManifestError(f"data.sources[{index}] is missing path")
        if correction_cache is None:
            raise ManifestError(f"data.sources[{index}] is missing correction_cache")
        name = str(raw_source.get("name") or Path(str(data_path)).stem).strip()
        if not name:
            raise ManifestError(f"data.sources[{index}] has an empty name")
        if name in names:
            raise ManifestError(f"duplicate source name: {name}")
        names.add(name)
        sources.append(
            {
                "name": name,
                "path": str(data_path),
                "correction_cache": str(correction_cache),
                **(
                    {"oracle_target_cache": str(raw_source["oracle_target_cache"])}
                    if raw_source.get("oracle_target_cache") is not None
                    else {}
                ),
            }
        )
    return sources


def _repo_relative_path(path: Path, repo_root: Path) -> str | None:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return None


def _path_reference(path: Path, repo_root: Path) -> dict[str, str | None]:
    return {
        "absolute_path": os.fspath(path),
        "repo_relative_path": _repo_relative_path(path, repo_root),
    }


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _snapshot_file(
    path: Path,
    repo_root: Path,
    cache: dict[str, tuple[dict[str, object], tuple[int, int, int, int]]],
) -> dict[str, object]:
    key = os.fspath(path)
    cached = cache.get(key)
    if cached is not None:
        return dict(cached[0])

    try:
        before = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing manifest input: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ManifestError(f"manifest input is not a regular file: {path}")
    digest = sha256_file(path)
    try:
        after = path.stat()
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest input disappeared while hashing: {path}") from exc
    if _stat_identity(before) != _stat_identity(after):
        raise ManifestError(f"manifest input changed while hashing: {path}")

    record: dict[str, object] = {
        **_path_reference(path, repo_root),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }
    cache[key] = (record, _stat_identity(after))
    return dict(record)


def _assert_snapshots_unchanged(
    cache: Mapping[str, tuple[dict[str, object], tuple[int, int, int, int]]],
) -> None:
    for raw_path, (_, expected_identity) in cache.items():
        path = Path(raw_path)
        try:
            current = path.stat()
        except FileNotFoundError as exc:
            raise ManifestError(f"manifest input disappeared during creation: {path}") from exc
        if _stat_identity(current) != expected_identity:
            raise ManifestError(f"manifest input changed during creation: {path}")


def _load_frozen_metadata(path: Path, expected_sha256: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid correction metadata JSON: {path}: {exc}") from exc
    if sha256_file(path) != expected_sha256:
        raise ManifestError(f"correction metadata changed while reading: {path}")
    if not isinstance(metadata, dict):
        raise ManifestError(f"correction metadata must be a JSON object: {path}")
    return metadata


def _metadata_target_path(value: object, metadata_path: Path) -> Path:
    target = Path(str(value)).expanduser()
    if target.is_absolute():
        return _absolute_path(target)
    return _absolute_path(target, base=metadata_path.parent)


def _normalize_source(source: Mapping[str, object], index: int) -> dict[str, str]:
    data_path = source.get("path")
    correction_cache = source.get("correction_cache")
    if data_path is None or correction_cache is None:
        raise ManifestError(
            f"source {index} must define both path and correction_cache"
        )
    name = str(source.get("name") or Path(str(data_path)).stem).strip()
    if not name:
        raise ManifestError(f"source {index} has an empty name")
    normalized = {
        "name": name,
        "path": str(data_path),
        "correction_cache": str(correction_cache),
    }
    oracle_target_cache = source.get("oracle_target_cache")
    if oracle_target_cache is not None:
        normalized["oracle_target_cache"] = str(oracle_target_cache)
    return normalized


def build_manifest(
    config_path: str | os.PathLike[str],
    sources: Sequence[Mapping[str, object]],
    *,
    repo_root: str | os.PathLike[str] | None = None,
    created_at: str | None = None,
) -> dict[str, object]:
    """Build a deterministic manifest for config, datasets, caches, and checkpoints."""

    config_absolute = _absolute_path(config_path)
    root = discover_repository_root(repo_root or config_absolute.parent)
    snapshots: dict[
        str,
        tuple[dict[str, object], tuple[int, int, int, int]],
    ] = {}
    config_record = _snapshot_file(config_absolute, root, snapshots)

    source_records: list[dict[str, object]] = []
    source_names: set[str] = set()
    for index, raw_source in enumerate(sources):
        if not isinstance(raw_source, Mapping):
            raise ManifestError(f"source {index} must be a mapping")
        source = _normalize_source(raw_source, index)
        name = source["name"]
        if name in source_names:
            raise ManifestError(f"duplicate source name: {name}")
        source_names.add(name)

        dataset_path = _absolute_path(source["path"], base=root)
        cache_path = _absolute_path(source["correction_cache"], base=root)
        if not cache_path.is_dir():
            raise FileNotFoundError(f"missing correction cache directory: {cache_path}")
        q_corrected_path = cache_path / "q_corrected.npy"
        metadata_path = cache_path / "metadata.json"

        dataset_record = _snapshot_file(dataset_path, root, snapshots)
        q_corrected_record = _snapshot_file(q_corrected_path, root, snapshots)
        metadata_record = _snapshot_file(metadata_path, root, snapshots)
        metadata = _load_frozen_metadata(
            metadata_path,
            str(metadata_record["sha256"]),
        )

        source_data = metadata.get("source_data")
        if source_data is not None:
            metadata_source = _metadata_target_path(source_data, metadata_path)
            if metadata_source.resolve() != dataset_path.resolve():
                raise ManifestError(
                    f"correction cache source {metadata_source} does not match "
                    f"dataset {dataset_path}"
                )
        checkpoint_value = metadata.get("checkpoint")
        if checkpoint_value is None or not str(checkpoint_value).strip():
            raise ManifestError(
                f"correction metadata is missing checkpoint: {metadata_path}"
            )
        checkpoint_path = _metadata_target_path(checkpoint_value, metadata_path)
        checkpoint_record = _snapshot_file(checkpoint_path, root, snapshots)

        oracle_record = None
        oracle_cache_value = source.get("oracle_target_cache")
        if oracle_cache_value is not None:
            oracle_cache_path = _absolute_path(oracle_cache_value, base=root)
            if not oracle_cache_path.is_dir():
                raise FileNotFoundError(
                    f"missing oracle target cache directory: {oracle_cache_path}"
                )
            oracle_array_path = oracle_cache_path / "oracle_target.npy"
            oracle_metadata_path = oracle_cache_path / "metadata.json"
            oracle_array_record = _snapshot_file(
                oracle_array_path, root, snapshots
            )
            oracle_metadata_record = _snapshot_file(
                oracle_metadata_path, root, snapshots
            )
            oracle_metadata = _load_frozen_metadata(
                oracle_metadata_path,
                str(oracle_metadata_record["sha256"]),
            )
            oracle_source_value = oracle_metadata.get("source_data")
            if oracle_source_value is None:
                raise ManifestError(
                    f"oracle target metadata is missing source_data: "
                    f"{oracle_metadata_path}"
                )
            oracle_source = _metadata_target_path(
                oracle_source_value, oracle_metadata_path
            )
            if oracle_source.resolve() != dataset_path.resolve():
                raise ManifestError(
                    f"oracle target cache source {oracle_source} does not match "
                    f"dataset {dataset_path}"
                )
            encoder_value = oracle_metadata.get("encoder_checkpoint")
            if encoder_value is None:
                raise ManifestError(
                    f"oracle target metadata is missing encoder_checkpoint: "
                    f"{oracle_metadata_path}"
                )
            encoder_path = _metadata_target_path(
                encoder_value, oracle_metadata_path
            )
            encoder_record = _snapshot_file(encoder_path, root, snapshots)
            if oracle_metadata.get("oracle_target_sha256") != oracle_array_record["sha256"]:
                raise ManifestError("oracle target array hash differs from metadata")
            if (
                oracle_metadata.get("encoder_checkpoint_sha256")
                != encoder_record["sha256"]
            ):
                raise ManifestError("oracle target encoder hash differs from metadata")
            oracle_record = {
                **_path_reference(oracle_cache_path, root),
                "oracle_target": oracle_array_record,
                "metadata": oracle_metadata_record,
                "encoder_checkpoint": encoder_record,
            }

        source_records.append(
            {
                "name": name,
                "dataset": dataset_record,
                "correction_cache": {
                    **_path_reference(cache_path, root),
                    "q_corrected": q_corrected_record,
                    "metadata": metadata_record,
                    "corrector_checkpoint": checkpoint_record,
                },
                **({"oracle_target_cache": oracle_record} if oracle_record else {}),
            }
        )

    source_records.sort(
        key=lambda source: (
            str(source["name"]),
            str(source["dataset"]["absolute_path"]),  # type: ignore[index]
        )
    )
    _assert_snapshots_unchanged(snapshots)
    git_state = capture_git_state(root)
    timestamp = created_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    if not isinstance(timestamp, str) or not timestamp:
        raise ManifestError("created_at must be a non-empty string")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": timestamp,
        "repository": {
            **_path_reference(root, root),
            **git_state,
        },
        "config": config_record,
        "sources": source_records,
    }


def create_manifest_from_yaml(
    config_path: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str] | None = None,
    created_at: str | None = None,
) -> dict[str, object]:
    """Load ``data.sources`` from YAML and build its frozen input manifest."""

    config = load_yaml_config(config_path)
    manifest = build_manifest(
        config_path,
        extract_sources(config),
        repo_root=repo_root,
        created_at=created_at,
    )
    data = config.get("data")
    episode_manifest = data.get("episode_manifest") if isinstance(data, Mapping) else None
    if episode_manifest:
        root = discover_repository_root(repo_root or config_path)
        episode_path = _absolute_path(episode_manifest, base=root)
        snapshots: dict[
            str,
            tuple[dict[str, object], tuple[int, int, int, int]],
        ] = {}
        manifest["episode_manifest"] = _snapshot_file(
            episode_path, root, snapshots
        )
        _assert_snapshots_unchanged(snapshots)
    return manifest


def manifest_json(manifest: Mapping[str, object]) -> str:
    """Serialize a manifest with stable key ordering and a trailing newline."""

    return json.dumps(
        manifest,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_manifest(
    output_path: str | os.PathLike[str],
    manifest: Mapping[str, object],
    *,
    force: bool = False,
) -> Path:
    """Write a manifest, refusing an existing destination unless forced."""

    output = _absolute_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_json(manifest)
    if not force:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        return output

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def _load_manifest(value: Mapping[str, object] | str | os.PathLike[str]) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    path = _absolute_path(value)
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid manifest JSON: {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root must be a JSON object")
    return manifest


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{label} must be an object")
    return value


def _iter_file_records(
    manifest: Mapping[str, object],
) -> list[tuple[str, Mapping[str, object]]]:
    records = [("config", _require_mapping(manifest.get("config"), "config"))]
    if manifest.get("episode_manifest") is not None:
        records.append(
            (
                "episode_manifest",
                _require_mapping(
                    manifest.get("episode_manifest"), "episode_manifest"
                ),
            )
        )
    sources = manifest.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ManifestError("sources must be an array")
    for index, raw_source in enumerate(sources):
        source = _require_mapping(raw_source, f"sources[{index}]")
        name = str(source.get("name", index))
        records.append(
            (
                f"source {name} dataset",
                _require_mapping(source.get("dataset"), f"source {name} dataset"),
            )
        )
        cache = _require_mapping(
            source.get("correction_cache"),
            f"source {name} correction_cache",
        )
        for field in ("q_corrected", "metadata", "corrector_checkpoint"):
            records.append(
                (
                    f"source {name} {field}",
                    _require_mapping(cache.get(field), f"source {name} {field}"),
                )
            )
        oracle_cache = source.get("oracle_target_cache")
        if oracle_cache is not None:
            oracle = _require_mapping(
                oracle_cache,
                f"source {name} oracle_target_cache",
            )
            for field in ("oracle_target", "metadata", "encoder_checkpoint"):
                records.append(
                    (
                        f"source {name} oracle {field}",
                        _require_mapping(
                            oracle.get(field),
                            f"source {name} oracle {field}",
                        ),
                    )
                )
    return records


def _validate_record_schema(
    label: str,
    record: Mapping[str, object],
) -> tuple[Path, int, int, str]:
    required = ("absolute_path", "repo_relative_path", "size", "mtime_ns", "sha256")
    missing = [field for field in required if field not in record]
    if missing:
        raise ManifestError(f"{label} is missing fields: {missing}")
    absolute_path = record["absolute_path"]
    if not isinstance(absolute_path, str) or not Path(absolute_path).is_absolute():
        raise ManifestError(f"{label}.absolute_path must be an absolute path")
    relative_path = record["repo_relative_path"]
    if relative_path is not None and not isinstance(relative_path, str):
        raise ManifestError(f"{label}.repo_relative_path must be a string or null")
    size = record["size"]
    mtime_ns = record["mtime_ns"]
    digest = record["sha256"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ManifestError(f"{label}.size must be a non-negative integer")
    if not isinstance(mtime_ns, int) or isinstance(mtime_ns, bool) or mtime_ns < 0:
        raise ManifestError(f"{label}.mtime_ns must be a non-negative integer")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ManifestError(f"{label}.sha256 must be a lowercase SHA-256 hex digest")
    return Path(absolute_path), size, mtime_ns, digest


def validate_manifest(
    manifest_or_path: Mapping[str, object] | str | os.PathLike[str],
    *,
    check_git: bool = True,
) -> None:
    """Raise ``ManifestValidationError`` if any frozen input or Git state changed."""

    manifest = _load_manifest(manifest_or_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version: {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("created_at"), str):
        raise ManifestError("created_at must be a string")
    repository = _require_mapping(manifest.get("repository"), "repository")
    repo_path_value = repository.get("absolute_path")
    if not isinstance(repo_path_value, str) or not Path(repo_path_value).is_absolute():
        raise ManifestError("repository.absolute_path must be an absolute path")
    repo_root = Path(repo_path_value)

    errors: list[str] = []
    current_files: dict[str, tuple[int, int, str] | None] = {}
    expected_files: dict[str, tuple[int, int, str]] = {}
    for label, record in _iter_file_records(manifest):
        path, expected_size, expected_mtime_ns, expected_sha256 = (
            _validate_record_schema(label, record)
        )
        key = os.fspath(path)
        expected = (expected_size, expected_mtime_ns, expected_sha256)
        previous_expected = expected_files.get(key)
        if previous_expected is not None and previous_expected != expected:
            errors.append(f"conflicting manifest records for {path}")
            continue
        expected_files[key] = expected

        stored_relative = record["repo_relative_path"]
        actual_relative = _repo_relative_path(path, repo_root)
        if stored_relative != actual_relative:
            errors.append(
                f"repo-relative path changed for {path}: expected "
                f"{stored_relative!r}, found {actual_relative!r}"
            )

        if key not in current_files:
            try:
                before = path.stat()
                if not stat.S_ISREG(before.st_mode):
                    errors.append(f"input is not a regular file: {path}")
                    current_files[key] = None
                    continue
                current_sha256 = sha256_file(path)
                after = path.stat()
                if _stat_identity(before) != _stat_identity(after):
                    errors.append(f"input changed while validating: {path}")
                    current_files[key] = None
                    continue
                current_files[key] = (
                    after.st_size,
                    after.st_mtime_ns,
                    current_sha256,
                )
            except FileNotFoundError:
                errors.append(f"missing file: {path}")
                current_files[key] = None
            except OSError as exc:
                errors.append(f"cannot read {path}: {exc}")
                current_files[key] = None

        current = current_files[key]
        if current is None:
            continue
        current_size, current_mtime_ns, current_sha256 = current
        if current_size != expected_size:
            errors.append(
                f"size changed for {path}: expected {expected_size}, found {current_size}"
            )
        if current_sha256 != expected_sha256:
            errors.append(
                f"sha256 changed for {path}: expected {expected_sha256}, "
                f"found {current_sha256}"
            )

    if check_git:
        expected_head = repository.get("head")
        expected_diff_sha256 = repository.get("dirty_diff_sha256")
        expected_worktree_sha256 = repository.get("worktree_sha256")
        if not isinstance(expected_head, str):
            raise ManifestError("repository.head must be a string")
        if not isinstance(expected_diff_sha256, str):
            raise ManifestError("repository.dirty_diff_sha256 must be a string")
        if expected_worktree_sha256 is not None and not isinstance(
            expected_worktree_sha256, str
        ):
            raise ManifestError("repository.worktree_sha256 must be a string")
        try:
            current_git = capture_git_state(repo_root)
        except ManifestError as exc:
            errors.append(f"cannot inspect repository state: {exc}")
        else:
            if current_git["head"] != expected_head:
                errors.append(
                    f"git HEAD changed: expected {expected_head}, "
                    f"found {current_git['head']}"
                )
            current_worktree_sha256 = current_git["worktree_sha256"]
            current_diff_sha256 = current_git["dirty_diff_sha256"]
            if expected_worktree_sha256 is not None and (
                current_worktree_sha256 != expected_worktree_sha256
            ):
                errors.append(
                    "git worktree sha256 changed: expected "
                    f"{expected_worktree_sha256}, "
                    f"found {current_worktree_sha256}"
                )
            elif current_diff_sha256 != expected_diff_sha256:
                errors.append(
                    "git dirty diff sha256 changed: expected "
                    f"{expected_diff_sha256}, "
                    f"found {current_diff_sha256}"
                )


    if errors:
        raise ManifestValidationError(errors)


__all__ = [
    "GIT_DIFF_FORMAT",
    "ManifestError",
    "ManifestValidationError",
    "SCHEMA_VERSION",
    "build_manifest",
    "capture_git_state",
    "create_manifest_from_yaml",
    "discover_repository_root",
    "extract_sources",
    "load_yaml_config",
    "manifest_json",
    "sha256_file",
    "validate_manifest",
    "write_manifest",
]
