from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import scripts.evaluate_lewm_counterfactual as cli


def test_exclusive_write_syncs_before_and_after_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    payload = '{"complete": true}\n'
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_link = os.link
    real_unlink = os.unlink

    def recording_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        kind = "directory" if stat.S_ISDIR(mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def recording_link(source: Path, destination: Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == tmp_path
        assert destination_path == output
        assert not output.exists()
        assert source_path.read_text(encoding="utf-8") == payload
        events.append(("link", "output"))
        real_link(source, destination)

    def recording_unlink(path: Path) -> None:
        assert Path(path) != output
        events.append(("unlink", "temporary"))
        real_unlink(path)

    monkeypatch.setattr(cli.os, "fsync", recording_fsync)
    monkeypatch.setattr(cli.os, "link", recording_link)
    monkeypatch.setattr(cli.os, "unlink", recording_unlink)

    cli._write_text_exclusive(output, payload)

    assert output.read_text(encoding="utf-8") == payload
    assert events == [
        ("fsync", "file"),
        ("link", "output"),
        ("fsync", "directory"),
        ("unlink", "temporary"),
        ("fsync", "directory"),
    ]
    assert list(tmp_path.iterdir()) == [output]


def test_temp_name_collision_does_not_modify_foreign_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    collision_token = "a" * 32
    foreign_temp = output.with_name(
        f".{output.name}.{os.getpid()}.{collision_token}.tmp"
    )
    foreign_payload = b"owned by another process\n"
    foreign_temp.write_bytes(foreign_payload)
    tokens = iter((collision_token, "b" * 32))

    monkeypatch.setattr(cli.secrets, "token_hex", lambda _size: next(tokens))

    cli._write_text_exclusive(output, "complete payload\n")

    assert output.read_text(encoding="utf-8") == "complete payload\n"
    assert foreign_temp.read_bytes() == foreign_payload


def test_exclusive_write_never_replaces_existing_output_or_unrelated_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    original = b'{"owner": "other-process"}\n'
    output.write_bytes(original)
    unrelated_temp = tmp_path / ".result.json.other-process.tmp"
    unrelated_temp.write_bytes(b"still owned by another process\n")
    removed: list[Path] = []
    real_unlink = os.unlink

    def recording_unlink(path: Path) -> None:
        removed.append(Path(path))
        real_unlink(path)

    monkeypatch.setattr(cli.os, "unlink", recording_unlink)

    with pytest.raises(FileExistsError, match="already exists"):
        cli._write_text_exclusive(output, '{"owner": "this-process"}\n')

    assert output.read_bytes() == original
    assert unrelated_temp.read_bytes() == b"still owned by another process\n"
    assert len(removed) == 1
    assert removed[0] not in {output, unrelated_temp}
    assert not removed[0].exists()


def test_interruption_before_publication_removes_only_owned_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    unrelated_temp = tmp_path / ".result.json.interrupted-process.tmp"
    unrelated_temp.write_bytes(b"unrelated\n")

    def interrupt_fsync(_descriptor: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.os, "fsync", interrupt_fsync)

    with pytest.raises(KeyboardInterrupt):
        cli._write_text_exclusive(output, "complete payload\n")

    assert not output.exists()
    assert list(tmp_path.iterdir()) == [unrelated_temp]
