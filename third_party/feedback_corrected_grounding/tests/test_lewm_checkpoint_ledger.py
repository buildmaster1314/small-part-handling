from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import torch

import fcpg.lewm_checkpoint_ledger as checkpoint_ledger
from fcpg.lewm_anchor_ablation import PARAMETER_COMPATIBLE_TRIPLET_MODES
from fcpg.lewm_checkpoint_ledger import (
    freeze_checkpoint_ledger,
    load_checkpoint_ledger,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolved_config(root: Path, mode: str) -> dict[str, object]:
    return {
        "seed": 5120,
        "output_dir": str(root / mode),
        "provenance_manifest": str(root / "manifests" / f"{mode}.json"),
        "data": {
            "target_mode": "frozen_oracle",
            "sources": [{"name": "source", "path": "source.npz"}],
            "history_size": 3,
            "rollout_steps": 1,
            "frame_skip": 5,
        },
        "loader": {"batch_size": 192, "num_workers": 0},
        "model": {"frame": {"anchor_visual_mode": mode, "latent_dim": 8}},
        "optimizer": {
            "lr": 5e-5,
            "weight_decay": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "warmup_steps": 127,
        },
        "loss": {
            "sigreg_weight": 0.02,
            "per_source_sigreg": True,
            "alignment_weight": 1.0,
        },
        "trainer": {
            "epochs": 100,
            "max_steps": 2000,
            "grad_clip": 1.0,
        },
        "contract": {
            "anchor_visual_mode": mode,
            "expected_parameter_count": 19_152_816,
            "optimizer_step_budget": 2000,
            "common_target": "frozen_oracle_anchor_v1",
            "target_encoder_sha256": "7" * 64,
        },
    }


def _data_config() -> dict[str, object]:
    return {
        "sources": [{"name": "source", "path": "source.npz"}],
        "target_mode": "frozen_oracle",
        "target_contract": {
            "mode": "frozen_oracle",
            "encoder_checkpoint_sha256": "7" * 64,
        },
        "history_size": 3,
        "rollout_steps": 1,
        "frame_skip": 5,
    }


def _write_run(root: Path, mode: str, *, optimizer_lr: float = 5e-5) -> None:
    output_dir = root / mode
    output_dir.mkdir(parents=True)
    (output_dir / ".lewm.lock").write_text("", encoding="utf-8")
    resolved = _resolved_config(root, mode)
    resolved["optimizer"]["lr"] = optimizer_lr  # type: ignore[index]
    _write_json(output_dir / "resolved_config.json", resolved)

    manifest = {
        "schema_version": 1,
        "config": {
            "sha256": hashlib.sha256(f"config-{mode}".encode()).hexdigest()
        },
        "sources": [],
    }
    external_manifest = root / "manifests" / f"{mode}.json"
    _write_json(external_manifest, manifest)
    _write_json(output_dir / "input_manifest.json", manifest)
    manifest_sha256 = sha256_file(external_manifest)
    assert sha256_file(output_dir / "input_manifest.json") == manifest_sha256

    metrics = {"step": 2000, "loss": 1.0}
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(metrics, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint = {
        "checkpoint_type": "fcpg_grounded_lewm_v1",
        "run_kind": "formal",
        "step": 2000,
        "metrics": metrics,
        "resolved_config": resolved,
        "args": deepcopy(resolved),
        "data_config": _data_config(),
        "model_config": {
            "frame": {"anchor_visual_mode": mode, "latent_dim": 8},
            "world_model": {
                "target_mode": "frozen_oracle",
                "alignment_weight": 1.0,
            },
        },
        "model_state": {"weight": torch.ones(1)},
        "optimizer_state": {},
        "scheduler_state": {},
        "scaler_state": {},
        "rng_state": {},
        "provenance": {
            "path": str(external_manifest),
            "sha256": manifest_sha256,
            "manifest": manifest,
        },
    }
    torch.save(checkpoint, output_dir / "lewm_grounded_latest.pt")


def _fixture_runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    for mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        _write_run(root, mode)
    return root


def test_freeze_and_load_bind_complete_training_contract(tmp_path: Path) -> None:
    root = _fixture_runs(tmp_path)
    ledger_path = tmp_path / "ledger.json"

    ledger = freeze_checkpoint_ledger(
        checkpoint_root=root,
        output=ledger_path,
        repo_root=tmp_path,
        require_git_trackable=False,
    )
    loaded = load_checkpoint_ledger(
        ledger_path,
        repo_root=tmp_path,
        require_git_tracked=False,
    )

    assert loaded["sha256"] == sha256_file(ledger_path)
    assert set(loaded["records"]) == set(PARAMETER_COMPATIBLE_TRIPLET_MODES)
    matched = ledger["matched_training_contract"]
    assert matched["step"] == 2000
    assert matched["seed"] == 5120
    assert matched["parameter_count"] == 19_152_816
    assert matched["optimizer"]["warmup_steps"] == 127
    assert matched["loss"]["alignment_weight"] == 1.0
    assert (
        matched["target"]["runtime_data"]["target_contract"]["mode"]
        == "frozen_oracle"
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_checkpoint_ledger(
            checkpoint_root=root,
            output=ledger_path,
            repo_root=tmp_path,
            require_git_trackable=False,
        )


def test_runner_trusts_only_the_head_committed_ledger(tmp_path: Path) -> None:
    root = _fixture_runs(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    freeze_checkpoint_ledger(
        checkpoint_root=root,
        output=ledger_path,
        repo_root=tmp_path,
        require_git_trackable=False,
    )
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "ledger.json"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="not committed in HEAD"):
        load_checkpoint_ledger(
            ledger_path,
            repo_root=tmp_path,
            require_git_tracked=True,
        )

    subprocess.run(
        [
            "git",
            "-c",
            "user.name=FCPG Test",
            "-c",
            "user.email=fcpg-test.invalid",
            "commit",
            "--quiet",
            "-m",
            "Freeze ledger",
        ],
        cwd=tmp_path,
        check=True,
    )
    loaded = load_checkpoint_ledger(
        ledger_path,
        repo_root=tmp_path,
        require_git_tracked=True,
    )
    assert loaded["git"]["head"]
    assert loaded["git"]["blob_oid"]
    expected_blob = subprocess.run(
        [
            "git",
            "rev-parse",
            f"{loaded['git']['head']}:ledger.json",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode("ascii").strip()
    assert loaded["git"]["blob_oid"] == expected_blob

    with ledger_path.open("a", encoding="utf-8") as stream:
        stream.write(chr(10))
    with pytest.raises(ValueError, match="committed HEAD blob"):
        load_checkpoint_ledger(
            ledger_path,
            repo_root=tmp_path,
            require_git_tracked=True,
        )


def test_staged_ledger_modification_is_rejected(tmp_path: Path) -> None:
    root = _fixture_runs(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    freeze_checkpoint_ledger(
        checkpoint_root=root,
        output=ledger_path,
        repo_root=tmp_path,
        require_git_trackable=False,
    )
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "ledger.json"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=FCPG Test",
            "-c",
            "user.email=fcpg-test.invalid",
            "commit",
            "--quiet",
            "-m",
            "Freeze ledger",
        ],
        cwd=tmp_path,
        check=True,
    )
    with ledger_path.open("a", encoding="utf-8") as stream:
        stream.write(chr(10))
    subprocess.run(["git", "add", "ledger.json"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="committed HEAD blob"):
        load_checkpoint_ledger(
            ledger_path,
            repo_root=tmp_path,
            require_git_tracked=True,
        )


def test_ledger_rejects_checkpoint_outside_declared_mode_root(
    tmp_path: Path,
) -> None:
    root = _fixture_runs(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    freeze_checkpoint_ledger(
        checkpoint_root=root,
        output=ledger_path,
        repo_root=tmp_path,
        require_git_trackable=False,
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["checkpoint_root"] = "other-runs"
    _write_json(ledger_path, ledger)

    with pytest.raises(ValueError, match="declared-root latest checkpoint"):
        load_checkpoint_ledger(
            ledger_path,
            repo_root=tmp_path,
            require_git_tracked=False,
        )


def test_ledger_hash_is_checked_before_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture_runs(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    freeze_checkpoint_ledger(
        checkpoint_root=root,
        output=ledger_path,
        repo_root=tmp_path,
        require_git_trackable=False,
    )
    first_mode = PARAMETER_COMPATIBLE_TRIPLET_MODES[0]
    checkpoint_path = root / first_mode / "lewm_grounded_latest.pt"
    checkpoint_path.write_bytes(b"tampered")

    def forbidden_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("torch.load ran before the ledger hash check")

    monkeypatch.setattr(checkpoint_ledger.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="before torch.load"):
        load_checkpoint_ledger(
            ledger_path,
            repo_root=tmp_path,
            require_git_tracked=False,
        )


def test_freeze_rejects_an_occupied_training_lock(tmp_path: Path) -> None:
    root = _fixture_runs(tmp_path)
    mode = PARAMETER_COMPATIBLE_TRIPLET_MODES[0]
    lock_path = root / mode / ".lewm.lock"
    ledger_path = tmp_path / "ledger.json"

    with lock_path.open("r+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="training lock is occupied"):
            freeze_checkpoint_ledger(
                checkpoint_root=root,
                output=ledger_path,
                repo_root=tmp_path,
                require_git_trackable=False,
            )
    assert not ledger_path.exists()


def test_freeze_requires_exact_step_2000(tmp_path: Path) -> None:
    root = _fixture_runs(tmp_path)
    mode = PARAMETER_COMPATIBLE_TRIPLET_MODES[0]
    checkpoint_path = root / mode / "lewm_grounded_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["step"] = 1999
    checkpoint["metrics"]["step"] = 1999
    torch.save(checkpoint, checkpoint_path)
    (root / mode / "metrics.jsonl").write_text(
        json.dumps(checkpoint["metrics"], sort_keys=True) + chr(10),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="step must be 2000"):
        freeze_checkpoint_ledger(
            checkpoint_root=root,
            output=tmp_path / "ledger.json",
            repo_root=tmp_path,
            require_git_trackable=False,
        )


def test_freeze_rejects_drift_in_any_complete_contract_field(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    for index, mode in enumerate(PARAMETER_COMPATIBLE_TRIPLET_MODES):
        _write_run(root, mode, optimizer_lr=1e-4 if index == 3 else 5e-5)

    with pytest.raises(ValueError, match="complete matched training contract"):
        freeze_checkpoint_ledger(
            checkpoint_root=root,
            output=tmp_path / "ledger.json",
            repo_root=tmp_path,
            require_git_trackable=False,
        )
