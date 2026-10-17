from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from fcpg.lewm_provenance import (
    ManifestValidationError,
    build_manifest,
    capture_git_state,
    create_manifest_from_yaml,
    manifest_json,
    validate_manifest,
)
from scripts.create_lewm_input_manifest import build_parser, main


CREATED_AT = "2026-07-11T12:00:00Z"


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _initialize_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Test User")


def _write_source(repo: Path, name: str, checkpoint: Path) -> dict[str, str]:
    dataset = repo / "inputs" / f"{name}.npz"
    cache = repo / "corrections" / name
    dataset.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    dataset.write_bytes(f"dataset-{name}".encode("ascii"))
    (cache / "q_corrected.npy").write_bytes(f"corrected-{name}".encode("ascii"))
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "source_data": str(dataset),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "name": name,
        "path": dataset.relative_to(repo).as_posix(),
        "correction_cache": cache.relative_to(repo).as_posix(),
    }


def _write_config(config: Path, sources: list[dict[str, str]]) -> None:
    lines = ["data:", "  sources:"]
    for source in sources:
        lines.extend(
            [
                f"    - name: {source['name']}",
                f"      path: {source['path']}",
                f"      correction_cache: {source['correction_cache']}",
            ]
        )
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path, list[dict[str, str]]]:
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    checkpoint = repo / "checkpoints" / "corrector.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"shared-corrector-checkpoint")
    sources = [
        _write_source(repo, "zeta", checkpoint),
        _write_source(repo, "alpha", checkpoint),
    ]
    config = repo / "train.yaml"
    _write_config(config, sources)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo, config, sources


def _source_file_records(source: dict[str, object]) -> list[dict[str, object]]:
    cache = source["correction_cache"]
    assert isinstance(cache, dict)
    return [
        source["dataset"],
        cache["q_corrected"],
        cache["metadata"],
        cache["corrector_checkpoint"],
    ]


def test_manifest_source_sorting_and_serialization_are_stable(tmp_path: Path) -> None:
    repo, config, sources = _fixture_repo(tmp_path)

    first = build_manifest(
        config,
        sources,
        repo_root=repo,
        created_at=CREATED_AT,
    )
    second = build_manifest(
        config,
        list(reversed(sources)),
        repo_root=repo,
        created_at=CREATED_AT,
    )

    assert first == second
    assert manifest_json(first) == manifest_json(second)
    assert [source["name"] for source in first["sources"]] == ["alpha", "zeta"]
    records = [first["config"]]
    for source in first["sources"]:
        records.extend(_source_file_records(source))
    for record in records:
        assert Path(record["absolute_path"]).is_absolute()
        assert isinstance(record["repo_relative_path"], str)
        assert isinstance(record["size"], int)
        assert isinstance(record["mtime_ns"], int)
        assert len(record["sha256"]) == 64
    validate_manifest(first)


def test_validate_manifest_detects_same_path_content_replacement(tmp_path: Path) -> None:
    repo, config, sources = _fixture_repo(tmp_path)
    manifest = build_manifest(
        config,
        sources,
        repo_root=repo,
        created_at=CREATED_AT,
    )
    alpha = next(source for source in manifest["sources"] if source["name"] == "alpha")
    q_record = alpha["correction_cache"]["q_corrected"]
    q_path = Path(q_record["absolute_path"])
    original_stat = q_path.stat()
    q_path.write_bytes(b"x" * q_record["size"])
    os.utime(
        q_path,
        ns=(original_stat.st_atime_ns, q_record["mtime_ns"]),
    )

    with pytest.raises(ManifestValidationError, match="sha256 changed"):
        validate_manifest(manifest, check_git=False)


def test_validate_manifest_detects_missing_file(tmp_path: Path) -> None:
    repo, config, sources = _fixture_repo(tmp_path)
    manifest = build_manifest(
        config,
        sources,
        repo_root=repo,
        created_at=CREATED_AT,
    )
    dataset = Path(manifest["sources"][0]["dataset"]["absolute_path"])
    dataset.unlink()

    with pytest.raises(ManifestValidationError, match="missing file"):
        validate_manifest(manifest, check_git=False)


def test_capture_git_state_hashes_the_current_dirty_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "initial")

    clean = capture_git_state(repo)
    assert clean["dirty"] is False
    assert clean["dirty_diff_sha256"] == hashlib.sha256(b"").hexdigest()

    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = capture_git_state(repo)
    expected_diff = _git(
        repo,
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
    assert dirty["dirty"] is True
    assert dirty["dirty_diff_sha256"] == hashlib.sha256(expected_diff).hexdigest()
    assert dirty["dirty_diff_sha256"] != clean["dirty_diff_sha256"]
    assert dirty["worktree_sha256"] != clean["worktree_sha256"]


def test_capture_git_state_hashes_untracked_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _initialize_repo(repo)
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "initial")

    clean = capture_git_state(repo)
    untracked = repo / "new_module.py"
    untracked.write_text("VALUE = 1\n", encoding="utf-8")
    first = capture_git_state(repo)
    untracked.write_text("VALUE = 2\n", encoding="utf-8")
    second = capture_git_state(repo)

    assert clean["dirty"] is False
    assert first["dirty"] is True
    assert first["untracked_files"][0]["path"] == "new_module.py"
    assert first["worktree_sha256"] != clean["worktree_sha256"]
    assert second["worktree_sha256"] != first["worktree_sha256"]


def test_validation_uses_content_hash_not_mtime(tmp_path: Path) -> None:
    repo, config, sources = _fixture_repo(tmp_path)
    manifest = build_manifest(
        config,
        sources,
        repo_root=repo,
        created_at=CREATED_AT,
    )
    dataset = Path(manifest["sources"][0]["dataset"]["absolute_path"])
    original = dataset.stat()
    os.utime(
        dataset,
        ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000),
    )

    validate_manifest(manifest, check_git=False)


def test_cli_reads_yaml_and_requires_force_to_replace_output(tmp_path: Path) -> None:
    repo, config, _ = _fixture_repo(tmp_path)
    output = tmp_path / "manifest.json"

    assert "--force" in build_parser().format_help()
    assert main(["--config", str(config), "--output", str(output)]) == 0
    generated = json.loads(output.read_text(encoding="utf-8"))
    assert [source["name"] for source in generated["sources"]] == ["alpha", "zeta"]

    with pytest.raises(SystemExit) as exc_info:
        main(["--config", str(config), "--output", str(output)])
    assert exc_info.value.code == 2

    output.write_text("replace me", encoding="utf-8")
    assert (
        main(
            [
                "--config",
                str(config),
                "--output",
                str(output),
                "--force",
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


def test_yaml_manifest_freezes_oracle_targets_and_episode_manifest(
    tmp_path: Path,
) -> None:
    repo, config, sources = _fixture_repo(tmp_path)
    encoder = repo / "targets" / "oracle_encoder.pt"
    encoder.parent.mkdir()
    encoder.write_bytes(b"frozen-oracle-encoder")
    encoder_sha256 = hashlib.sha256(encoder.read_bytes()).hexdigest()

    for source in sources:
        cache = repo / "targets" / source["name"]
        cache.mkdir()
        target = cache / "oracle_target.npy"
        target.write_bytes(f"target-{source['name']}".encode("ascii"))
        (cache / "metadata.json").write_text(
            json.dumps(
                {
                    "source_data": str(repo / source["path"]),
                    "encoder_checkpoint": str(encoder),
                    "oracle_target_sha256": hashlib.sha256(
                        target.read_bytes()
                    ).hexdigest(),
                    "encoder_checkpoint_sha256": encoder_sha256,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        source["oracle_target_cache"] = cache.relative_to(repo).as_posix()

    episode_manifest = repo / "splits" / "episodes.json"
    episode_manifest.parent.mkdir()
    episode_manifest.write_text('{"train": [0], "validation": [1]}\n', encoding="utf-8")
    lines = [
        "data:",
        f"  episode_manifest: {episode_manifest.relative_to(repo).as_posix()}",
        "  sources:",
    ]
    for source in sources:
        lines.extend(
            [
                f"    - name: {source['name']}",
                f"      path: {source['path']}",
                f"      correction_cache: {source['correction_cache']}",
                f"      oracle_target_cache: {source['oracle_target_cache']}",
            ]
        )
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = create_manifest_from_yaml(config, repo_root=repo, created_at=CREATED_AT)

    assert manifest["episode_manifest"]["sha256"] == hashlib.sha256(
        episode_manifest.read_bytes()
    ).hexdigest()
    for source in manifest["sources"]:
        oracle = source["oracle_target_cache"]
        assert oracle["encoder_checkpoint"]["sha256"] == encoder_sha256
        assert len(oracle["oracle_target"]["sha256"]) == 64
    validate_manifest(manifest)

    target_path = Path(
        manifest["sources"][0]["oracle_target_cache"]["oracle_target"][
            "absolute_path"
        ]
    )
    target_path.write_bytes(b"tampered-oracle-target")
    with pytest.raises(ManifestValidationError, match="sha256 changed"):
        validate_manifest(manifest, check_git=False)
