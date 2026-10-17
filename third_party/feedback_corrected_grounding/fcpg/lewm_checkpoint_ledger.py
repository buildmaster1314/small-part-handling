"""Freeze and validate the four LeWM anchor-visual latest checkpoints."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import torch

from fcpg.lewm_anchor_ablation import (
    PARAMETER_COMPATIBLE_TRIPLET_MODES,
    validate_anchor_visual_override,
)


LEDGER_SCHEMA_VERSION = 1
LEDGER_PROTOCOL = "fcpg_lewm_anchor_visual_checkpoint_ledger_v1"
LATEST_CHECKPOINT_NAME = "lewm_grounded_latest.pt"
EXPECTED_CHECKPOINT_TYPE = "fcpg_grounded_lewm_v1"
EXPECTED_STEP = 2000


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON value for content-addressed contract hashes."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _encoded_path(path: Path, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _resolved_path(value: object, repo_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _artifact(path: Path, repo_root: Path, digest: str | None = None) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _encoded_path(path, repo_root),
        "sha256": digest or sha256_file(path),
    }


def _artifact_path(
    record: object,
    *,
    repo_root: Path,
    label: str,
) -> tuple[Path, str]:
    value = _object(record, label)
    unexpected = set(value).difference({"path", "sha256"})
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {sorted(unexpected)}")
    path = _resolved_path(value.get("path"), repo_root, f"{label}.path")
    digest = _require_sha256(value.get("sha256"), f"{label}.sha256")
    return path, digest


def _load_json_hashed(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{label} changed while it was loaded: {path}")
    return value


def _load_metrics_tail_hashed(
    path: Path,
    expected_sha256: str,
    checkpoint_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"metrics SHA-256 mismatch: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            raise ValueError(f"blank metrics line at {path}:{line_number}")
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise TypeError(f"metrics line {line_number} must be an object")
        records.append(value)
    if not records:
        raise ValueError(f"metrics file is empty: {path}")
    tail = records[-1]
    if tail.get("step") != EXPECTED_STEP:
        raise ValueError(
            f"metrics tail step must be {EXPECTED_STEP}, got {tail.get('step')!r}"
        )
    if tail != dict(checkpoint_metrics):
        raise ValueError("metrics tail differs from checkpoint metrics")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"metrics changed while it was loaded: {path}")
    return tail


def canonical_resolved_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Mask only the four fields intentionally different between matched runs."""

    config = deepcopy(dict(value))
    if "output_dir" not in config or "provenance_manifest" not in config:
        raise ValueError("resolved config is missing output/provenance paths")
    config["output_dir"] = "<matched-output>"
    config["provenance_manifest"] = "<matched-manifest>"
    model = _object(config.get("model"), "resolved model")
    frame = _object(model.get("frame"), "resolved model.frame")
    if "anchor_visual_mode" not in frame:
        raise ValueError("resolved model.frame has no anchor_visual_mode")
    frame["anchor_visual_mode"] = "<matched-anchor-mode>"
    contract = _object(config.get("contract"), "resolved contract")
    if "anchor_visual_mode" not in contract:
        raise ValueError("resolved contract has no anchor_visual_mode")
    contract["anchor_visual_mode"] = "<matched-anchor-mode>"
    return config


def _canonical_model_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(value))
    frame = _object(config.get("frame"), "model_config.frame")
    if "anchor_visual_mode" not in frame:
        raise ValueError("model_config.frame has no anchor_visual_mode")
    frame["anchor_visual_mode"] = "<matched-anchor-mode>"
    return config


def _validate_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int]:
    if checkpoint.get("checkpoint_type") != EXPECTED_CHECKPOINT_TYPE:
        raise ValueError(f"{mode} checkpoint_type mismatch")
    if checkpoint.get("run_kind") != "formal":
        raise ValueError(f"{mode} checkpoint must be a formal run")
    if checkpoint.get("step") != EXPECTED_STEP:
        raise ValueError(
            f"{mode} checkpoint step must be {EXPECTED_STEP}, "
            f"got {checkpoint.get('step')!r}"
        )
    metrics = dict(_object(checkpoint.get("metrics"), f"{mode} metrics"))
    if metrics.get("step") != EXPECTED_STEP:
        raise ValueError(f"{mode} checkpoint metrics step mismatch")

    resolved = deepcopy(
        dict(_object(checkpoint.get("resolved_config"), f"{mode} resolved_config"))
    )
    if checkpoint.get("args") != resolved:
        raise ValueError(f"{mode} checkpoint args differ from resolved_config")
    data_config = deepcopy(
        dict(_object(checkpoint.get("data_config"), f"{mode} data_config"))
    )
    model_config = deepcopy(
        dict(_object(checkpoint.get("model_config"), f"{mode} model_config"))
    )
    state = _object(checkpoint.get("model_state"), f"{mode} model_state")
    if not state:
        raise ValueError(f"{mode} model_state is empty")
    for field in (
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
    ):
        _object(checkpoint.get(field), f"{mode} {field}")

    seed = _positive_integer(resolved.get("seed"), f"{mode} seed")
    contract = _object(resolved.get("contract"), f"{mode} contract")
    parameter_count = _positive_integer(
        contract.get("expected_parameter_count"),
        f"{mode} expected_parameter_count",
    )
    for field in ("optimizer", "loss", "data", "loader", "trainer"):
        _object(resolved.get(field), f"{mode} resolved_config.{field}")
    train_mode = validate_anchor_visual_override(
        contract.get("anchor_visual_mode"),
        field_name=f"{mode} contract.anchor_visual_mode",  # type: ignore[arg-type]
    )
    if train_mode != mode:
        raise ValueError(f"{mode} checkpoint records train mode {train_mode!r}")
    resolved_frame = _object(
        _object(resolved.get("model"), f"{mode} model").get("frame"),
        f"{mode} model.frame",
    )
    configured_frame = _object(model_config.get("frame"), f"{mode} model_config.frame")
    if resolved_frame.get("anchor_visual_mode") != mode or (
        configured_frame.get("anchor_visual_mode") != mode
    ):
        raise ValueError(f"{mode} checkpoint frame modes disagree")
    return resolved, data_config, model_config, seed, parameter_count


def _training_contract(
    *,
    resolved_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    step: int,
    seed: int,
    parameter_count: int,
) -> dict[str, Any]:
    canonical = canonical_resolved_config(resolved_config)
    canonical_model = _canonical_model_config(model_config)
    optimizer = deepcopy(dict(_object(canonical.get("optimizer"), "optimizer")))
    loss = deepcopy(dict(_object(canonical.get("loss"), "loss")))
    target = {
        "resolved_data": deepcopy(
            dict(_object(canonical.get("data"), "resolved data"))
        ),
        "runtime_data": deepcopy(dict(data_config)),
        "world_model": deepcopy(
            dict(_object(canonical_model.get("world_model"), "world_model"))
        ),
        "run_contract": deepcopy(
            dict(_object(canonical.get("contract"), "run contract"))
        ),
    }
    return {
        "step": step,
        "seed": seed,
        "parameter_count": parameter_count,
        "canonical_resolved_config": canonical,
        "canonical_resolved_config_sha256": object_sha256(canonical),
        "data_config": deepcopy(dict(data_config)),
        "data_config_sha256": object_sha256(data_config),
        "canonical_model_config": canonical_model,
        "canonical_model_config_sha256": object_sha256(canonical_model),
        "optimizer": optimizer,
        "optimizer_sha256": object_sha256(optimizer),
        "loss": loss,
        "loss_sha256": object_sha256(loss),
        "target": target,
        "target_sha256": object_sha256(target),
    }


def _checkpoint_artifact_path(
    expected_record: Mapping[str, Any] | None,
    *,
    checkpoint_root: Path | None,
    mode: str,
    repo_root: Path,
) -> tuple[Path, str | None]:
    if expected_record is None:
        if checkpoint_root is None:
            raise ValueError("checkpoint_root is required while creating a ledger")
        return checkpoint_root / mode / LATEST_CHECKPOINT_NAME, None
    path, digest = _artifact_path(
        expected_record.get("checkpoint"),
        repo_root=repo_root,
        label=f"ledger {mode} checkpoint",
    )
    return path, digest


def _inspect_checkpoint(
    mode: str,
    *,
    repo_root: Path,
    checkpoint_root: Path | None = None,
    expected_record: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path, expected_checkpoint_sha256 = _checkpoint_artifact_path(
        expected_record,
        checkpoint_root=checkpoint_root,
        mode=mode,
        repo_root=repo_root,
    )
    if checkpoint_path.name != LATEST_CHECKPOINT_NAME:
        raise ValueError(f"{mode} must use {LATEST_CHECKPOINT_NAME}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    # The ledger digest is checked before pickle-based checkpoint deserialization.
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if expected_checkpoint_sha256 is not None and (
        checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError(
            f"{mode} checkpoint SHA-256 mismatch before torch.load: "
            f"expected {expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{mode} checkpoint must be an object")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError(f"{mode} checkpoint changed while it was loaded")

    resolved, data_config, model_config, seed, parameter_count = (
        _validate_checkpoint_payload(checkpoint, mode=mode)
    )
    checkpoint_metrics = _object(checkpoint.get("metrics"), f"{mode} metrics")
    metrics_path = checkpoint_path.parent / "metrics.jsonl"
    run_manifest_path = checkpoint_path.parent / "input_manifest.json"
    resolved_config_path = checkpoint_path.parent / "resolved_config.json"
    provenance = _object(checkpoint.get("provenance"), f"{mode} provenance")
    manifest_sha256 = _require_sha256(
        provenance.get("sha256"), f"{mode} provenance.sha256"
    )
    external_manifest_path = _resolved_path(
        provenance.get("path"), repo_root, f"{mode} provenance.path"
    )

    if expected_record is None:
        metrics_sha256 = sha256_file(metrics_path)
        run_manifest_sha256 = sha256_file(run_manifest_path)
        external_manifest_sha256 = sha256_file(external_manifest_path)
        resolved_config_sha256 = sha256_file(resolved_config_path)
    else:
        expected_artifacts = {}
        for field in (
            "metrics",
            "run_manifest",
            "external_manifest",
            "resolved_config",
        ):
            expected_artifacts[field] = _artifact_path(
                expected_record.get(field),
                repo_root=repo_root,
                label=f"ledger {mode} {field}",
            )
        fixed_paths = {
            "metrics": metrics_path,
            "run_manifest": run_manifest_path,
            "external_manifest": external_manifest_path,
            "resolved_config": resolved_config_path,
        }
        for field, expected in expected_artifacts.items():
            if expected[0] != fixed_paths[field].resolve():
                raise ValueError(f"{mode} {field} path differs from checkpoint")
        metrics_sha256 = expected_artifacts["metrics"][1]
        run_manifest_sha256 = expected_artifacts["run_manifest"][1]
        external_manifest_sha256 = expected_artifacts["external_manifest"][1]
        resolved_config_sha256 = expected_artifacts["resolved_config"][1]

    metrics_tail = _load_metrics_tail_hashed(
        metrics_path,
        metrics_sha256,
        checkpoint_metrics,
    )
    run_manifest = _load_json_hashed(
        run_manifest_path, run_manifest_sha256, f"{mode} run manifest"
    )
    external_manifest = _load_json_hashed(
        external_manifest_path,
        external_manifest_sha256,
        f"{mode} external manifest",
    )
    embedded_manifest = dict(
        _object(provenance.get("manifest"), f"{mode} embedded manifest")
    )
    if (
        run_manifest_sha256 != manifest_sha256
        or external_manifest_sha256 != manifest_sha256
        or run_manifest != embedded_manifest
        or external_manifest != embedded_manifest
    ):
        raise ValueError(f"{mode} run/external manifests differ from checkpoint")
    resolved_sidecar = _load_json_hashed(
        resolved_config_path,
        resolved_config_sha256,
        f"{mode} resolved config",
    )
    if resolved_sidecar != resolved:
        raise ValueError(f"{mode} resolved-config sidecar differs from checkpoint")
    config_record = _object(embedded_manifest.get("config"), f"{mode} config record")
    config_sha256 = _require_sha256(
        config_record.get("sha256"), f"{mode} training config SHA-256"
    )

    record = {
        "mode": mode,
        "checkpoint": _artifact(checkpoint_path, repo_root, checkpoint_sha256),
        "metrics": _artifact(metrics_path, repo_root, metrics_sha256),
        "run_manifest": _artifact(
            run_manifest_path, repo_root, run_manifest_sha256
        ),
        "external_manifest": _artifact(
            external_manifest_path, repo_root, external_manifest_sha256
        ),
        "resolved_config": _artifact(
            resolved_config_path, repo_root, resolved_config_sha256
        ),
        "resolved_config_payload_sha256": object_sha256(resolved),
        "config_sha256": config_sha256,
        "step": EXPECTED_STEP,
        "seed": seed,
        "parameter_count": parameter_count,
        "metrics_tail": metrics_tail,
    }
    runtime = {
        "checkpoint": dict(checkpoint),
        "checkpoint_path": checkpoint_path,
        "resolved_config": resolved,
        "data_config": data_config,
        "model_config": model_config,
        "manifest": embedded_manifest,
        "record": record,
    }
    return record, runtime


def _matched_contract(
    runtimes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reference_mode = PARAMETER_COMPATIBLE_TRIPLET_MODES[0]
    reference = runtimes[reference_mode]
    record = _object(reference.get("record"), f"{reference_mode} record")
    expected = _training_contract(
        resolved_config=_object(reference.get("resolved_config"), "resolved_config"),
        data_config=_object(reference.get("data_config"), "data_config"),
        model_config=_object(reference.get("model_config"), "model_config"),
        step=_positive_integer(record.get("step"), "step"),
        seed=_positive_integer(record.get("seed"), "seed"),
        parameter_count=_positive_integer(
            record.get("parameter_count"), "parameter_count"
        ),
    )
    for mode in PARAMETER_COMPATIBLE_TRIPLET_MODES[1:]:
        current = runtimes[mode]
        current_record = _object(current.get("record"), f"{mode} record")
        observed = _training_contract(
            resolved_config=_object(current.get("resolved_config"), f"{mode} config"),
            data_config=_object(current.get("data_config"), f"{mode} data config"),
            model_config=_object(current.get("model_config"), f"{mode} model config"),
            step=_positive_integer(current_record.get("step"), f"{mode} step"),
            seed=_positive_integer(current_record.get("seed"), f"{mode} seed"),
            parameter_count=_positive_integer(
                current_record.get("parameter_count"), f"{mode} parameter_count"
            ),
        )
        if observed != expected:
            differing = sorted(
                key
                for key in set(expected).union(observed)
                if expected.get(key) != observed.get(key)
            )
            raise ValueError(
                f"{mode} violates the complete matched training contract: {differing}"
            )
    return expected


@contextmanager
def unlocked_training_runs(
    checkpoint_root: str | Path,
    modes: Sequence[str] = PARAMETER_COMPATIBLE_TRIPLET_MODES,
) -> Iterator[None]:
    """Hold non-blocking locks so no training process can publish while freezing."""

    root = Path(checkpoint_root).expanduser().resolve()
    handles = []
    try:
        for mode in modes:
            lock_path = root / mode / ".lewm.lock"
            if not lock_path.is_file():
                raise FileNotFoundError(
                    f"cannot prove {mode} training is unlocked; missing {lock_path}"
                )
            handle = lock_path.open("r+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError(f"training lock is occupied: {lock_path}") from error
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _ensure_git_trackable(path: Path, repo_root: Path) -> None:
    try:
        relative = path.resolve().relative_to(repo_root)
    except ValueError as error:
        raise ValueError("ledger must be created inside the Git worktree") from error
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=repo_root,
        check=False,
    )
    if ignored.returncode == 0:
        raise ValueError(f"ledger destination is Git-ignored: {path}")
    if ignored.returncode not in {1}:
        raise RuntimeError("git check-ignore failed for ledger destination")


def _require_git_tracked(
    path: Path, repo_root: Path, payload: bytes
) -> dict[str, str]:
    try:
        relative = path.resolve().relative_to(repo_root)
    except ValueError as error:
        raise ValueError("ledger must be inside the Git worktree") from error
    relative_text = relative.as_posix()
    head_result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if head_result.returncode != 0:
        raise ValueError(f"checkpoint ledger is not committed in HEAD: {path}")
    head = head_result.stdout.decode("ascii").strip()
    blob = subprocess.run(
        ["git", "rev-parse", f"{head}:{relative_text}"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if blob.returncode != 0:
        raise ValueError(f"checkpoint ledger is not committed in HEAD: {path}")
    blob_oid = blob.stdout.decode("ascii").strip()
    committed = subprocess.run(
        ["git", "cat-file", "blob", blob_oid],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    ).stdout
    if payload != committed:
        raise ValueError(
            f"checkpoint ledger differs from the committed HEAD blob: {path}"
        )
    for command, label in (
        (["git", "diff", "--quiet", "--", relative_text], "working tree"),
        (
            ["git", "diff", "--cached", "--quiet", "HEAD", "--", relative_text],
            "Git index",
        ),
    ):
        unchanged = subprocess.run(command, cwd=repo_root, check=False)
        if unchanged.returncode == 1:
            raise ValueError(f"checkpoint ledger differs from {label}: {path}")
        if unchanged.returncode != 0:
            raise RuntimeError(
                f"git diff failed while validating checkpoint ledger {label}"
            )
    return {"head": head, "blob_oid": blob_oid}


def freeze_checkpoint_ledger(
    *,
    checkpoint_root: str | Path,
    output: str | Path,
    repo_root: str | Path,
    require_git_trackable: bool = True,
) -> dict[str, Any]:
    """Create one immutable ledger after all four exact latest runs are complete."""

    root = Path(repo_root).expanduser().resolve()
    checkpoint_root_path = Path(checkpoint_root).expanduser()
    if not checkpoint_root_path.is_absolute():
        checkpoint_root_path = root / checkpoint_root_path
    checkpoint_root_path = checkpoint_root_path.resolve()
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint ledger: {output_path}")
    if require_git_trackable:
        _ensure_git_trackable(output_path, root)

    with unlocked_training_runs(checkpoint_root_path):
        records: dict[str, dict[str, Any]] = {}
        runtimes: dict[str, dict[str, Any]] = {}
        for mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
            record, runtime = _inspect_checkpoint(
                mode,
                repo_root=root,
                checkpoint_root=checkpoint_root_path,
            )
            records[mode] = record
            runtimes[mode] = runtime
        matched_contract = _matched_contract(runtimes)
        ledger = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "protocol": LEDGER_PROTOCOL,
            "development_only": True,
            "checkpoint_root": _encoded_path(checkpoint_root_path, root),
            "modes": list(PARAMETER_COMPATIBLE_TRIPLET_MODES),
            "matched_training_contract": matched_contract,
            "records": records,
        }

        # Recheck every source artifact while all four run locks are still held.
        for mode, record in records.items():
            for field in (
                "checkpoint",
                "metrics",
                "run_manifest",
                "external_manifest",
                "resolved_config",
            ):
                path, digest = _artifact_path(
                    record[field], repo_root=root, label=f"{mode} {field}"
                )
                if sha256_file(path) != digest:
                    raise RuntimeError(f"{mode} {field} changed before ledger creation")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            ledger,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        with output_path.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(payload)
    return ledger


def load_checkpoint_ledger(
    path: str | Path,
    *,
    repo_root: str | Path,
    require_git_tracked: bool = True,
) -> dict[str, Any]:
    """Load a ledger and validate all bound artifacts and checkpoint payloads."""

    root = Path(repo_root).expanduser().resolve()
    ledger_path = Path(path).expanduser()
    if not ledger_path.is_absolute():
        ledger_path = root / ledger_path
    ledger_path = ledger_path.resolve()
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    with ledger_path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        ledger_payload = stream.read()
        after = os.fstat(stream.fileno())
    identity = lambda stat_result: (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(ledger_payload) != before.st_size:
        raise RuntimeError("checkpoint ledger changed while it was loaded")
    ledger_git = (
        _require_git_tracked(ledger_path, root, ledger_payload)
        if require_git_tracked
        else None
    )
    ledger_sha256 = hashlib.sha256(ledger_payload).hexdigest()
    ledger = json.loads(ledger_payload.decode("utf-8"))
    if not isinstance(ledger, dict):
        raise TypeError("checkpoint ledger must contain a JSON object")
    if sha256_file(ledger_path) != ledger_sha256:
        raise RuntimeError("checkpoint ledger changed while it was loaded")
    expected_fields = {
        "schema_version",
        "protocol",
        "development_only",
        "checkpoint_root",
        "modes",
        "matched_training_contract",
        "records",
    }
    if set(ledger) != expected_fields:
        raise ValueError(
            "checkpoint ledger top-level fields differ: "
            f"{sorted(set(ledger).symmetric_difference(expected_fields))}"
        )
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION or (
        ledger.get("protocol") != LEDGER_PROTOCOL
    ):
        raise ValueError("unsupported checkpoint ledger protocol")
    if ledger.get("development_only") is not True:
        raise ValueError("checkpoint ledger must be development_only")
    if ledger.get("modes") != list(PARAMETER_COMPATIBLE_TRIPLET_MODES):
        raise ValueError("checkpoint ledger mode order/coverage mismatch")
    checkpoint_root = _resolved_path(
        ledger.get("checkpoint_root"), root, "ledger checkpoint_root"
    )
    raw_records = _object(ledger.get("records"), "ledger records")
    if set(raw_records) != set(PARAMETER_COMPATIBLE_TRIPLET_MODES):
        raise ValueError("checkpoint ledger must contain exactly four mode records")

    records: dict[str, dict[str, Any]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    for mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        expected_record = _object(raw_records.get(mode), f"ledger record {mode}")
        checkpoint_path, _ = _artifact_path(
            expected_record.get("checkpoint"),
            repo_root=root,
            label=f"ledger {mode} checkpoint",
        )
        expected_checkpoint_path = (
            checkpoint_root / mode / LATEST_CHECKPOINT_NAME
        ).resolve()
        if checkpoint_path != expected_checkpoint_path:
            raise ValueError(
                f"{mode} checkpoint is not the declared-root latest checkpoint"
            )
        record, runtime = _inspect_checkpoint(
            mode,
            repo_root=root,
            expected_record=expected_record,
        )
        if record != dict(expected_record):
            differing = sorted(
                key
                for key in set(record).union(expected_record)
                if record.get(key) != expected_record.get(key)
            )
            raise ValueError(f"{mode} checkpoint ledger record mismatch: {differing}")
        records[mode] = record
        runtimes[mode] = runtime
    matched_contract = _matched_contract(runtimes)
    if ledger.get("matched_training_contract") != matched_contract:
        raise ValueError("ledger matched training contract differs from checkpoints")
    if sha256_file(ledger_path) != ledger_sha256:
        raise RuntimeError("checkpoint ledger changed during checkpoint validation")
    return {
        "path": str(ledger_path),
        "sha256": ledger_sha256,
        "git": ledger_git,
        "ledger": ledger,
        "records": records,
        "runtimes": runtimes,
        "matched_training_contract": matched_contract,
    }


__all__ = [
    "EXPECTED_STEP",
    "LATEST_CHECKPOINT_NAME",
    "LEDGER_PROTOCOL",
    "LEDGER_SCHEMA_VERSION",
    "canonical_json_bytes",
    "canonical_resolved_config",
    "freeze_checkpoint_ledger",
    "load_checkpoint_ledger",
    "object_sha256",
    "sha256_file",
    "unlocked_training_runs",
]
