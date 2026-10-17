#!/usr/bin/env python3
"""Run the development-only 4x4 LeWM anchor-visual input matrix."""

from __future__ import annotations

import argparse
import fcntl
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_anchor_ablation import (  # noqa: E402
    PARAMETER_COMPATIBLE_TRIPLET_MODES,
    build_anchor_ablation_cells,
    summarize_anchor_ablation_reports,
)
from fcpg.lewm_checkpoint_ledger import (  # noqa: E402
    canonical_resolved_config as ledger_canonical_resolved_config,
    load_checkpoint_ledger,
)

from fcpg.lewm_dev_gate import (  # noqa: E402
    summarize_anchor_visual_dev_gate,
    validate_anchor_visual_dev_protocol,
)
from fcpg.lewm_episode_annotations import (  # noqa: E402
    annotate_episode_blocks,
    load_validation_episode_annotations,
)
from fcpg.lewm_matrix_analysis import summarize_paired_matrix  # noqa: E402
from fcpg.lewm_matrix_report import render_anchor_visual_markdown  # noqa: E402
from fcpg.lewm_provenance import capture_git_state, validate_manifest  # noqa: E402
from scripts.evaluate_lewm_grounded import (  # noqa: E402
    ANCHOR_ABLATION_PARAMETER_COUNT,
    ANCHOR_ABLATION_SEED,
    ANCHOR_ABLATION_SPLIT_SHA256,
    ANCHOR_ABLATION_TARGET_SHA256,
    execution_code_manifest,
    resolve_anchor_visual_ablation_contract,
)
from scripts.train_bc_policy import make_split  # noqa: E402


PROTOCOL = "fcpg_lewm_anchor_visual_matrix_v1"
CELL_CONTRACT_PROTOCOL = "fcpg_lewm_anchor_visual_cell_contract_v4"
CELL_COMPLETION_PROTOCOL = "fcpg_lewm_anchor_visual_cell_completion_v3"
SUMMARY_COMPLETION_PROTOCOL = "fcpg_lewm_anchor_visual_summary_completion_v2"
DEFAULT_CHECKPOINT_ROOT = Path(
    "outputs/lewm_grounded/anchor_visual_ablation_seed5120"
)
DEFAULT_RESULT_DIR = DEFAULT_CHECKPOINT_ROOT / "validation_matrix"
DEFAULT_DEV_PROTOCOL = (
    REPO_ROOT / "config/lewm/eval/anchor_visual_dev_protocol_seed5120.json"
)
EXPECTED_TRAINING_CONTRACT = {
    "seed": 5120,
    "batch_size": 192,
    "warmup_steps": 127,
    "max_steps": 2000,
    "parameter_count": 19_152_816,
    "sigreg_weight": 0.02,
}
EXPECTED_FULL_VALIDATION_COUNTS = {"clean": 56, "mild": 112, "total": 168}
EXPECTED_UNSEEN_COUNTS = {"clean": 12, "mild": 29, "total": 41}
TRAINING_SOURCE_FILES = (
    "scripts/train_lewm_grounded.py",
    "fcpg/__init__.py",
    "fcpg/lewm_runtime.py",
    "fcpg/lewm_grounded_model.py",
    "fcpg/lewm_sequence_data.py",
    "fcpg/lewm_provenance.py",
    "fcpg/modules.py",
)
TRAINING_SOURCE_PATHS = (*TRAINING_SOURCE_FILES, "third_party/lewm")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_repository_state() -> dict[str, Any]:
    state = capture_git_state(REPO_ROOT)
    if state.get("dirty") is not False:
        raise RuntimeError(
            "anchor-visual matrix requires a clean Git worktree committed at HEAD"
        )
    head = state.get("head")
    worktree_sha256 = state.get("worktree_sha256")
    if not isinstance(head, str) or not head:
        raise RuntimeError("cannot bind anchor-visual matrix to Git HEAD")
    if not isinstance(worktree_sha256, str) or len(worktree_sha256) != 64:
        raise RuntimeError("cannot bind anchor-visual matrix to a clean worktree hash")
    return dict(state)


def _require_repository_state_unchanged(expected: Mapping[str, Any]) -> None:
    current = _clean_repository_state()
    if current != dict(expected):
        raise RuntimeError("Git HEAD or worktree changed during matrix evaluation")


def _acquire_result_lock(result_dir: Path):
    lock_path = result_dir / ".matrix.lock"
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"another anchor-visual matrix runner holds {lock_path}"
        ) from error
    return handle


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _render_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json_idempotent(path: Path, value: Mapping[str, Any]) -> None:
    payload = _render_json(value)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different file: {path}") from None


def _write_text_idempotent(path: Path, value: str) -> None:
    payload = value.encode("ascii")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different file: {path}") from None


def _canonical_resolved_config(value: Mapping[str, Any]) -> dict[str, Any]:
    return ledger_canonical_resolved_config(value)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_json_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read()
        after = os.fstat(stream.fileno())
    identity = lambda stat_result: (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise RuntimeError(f"{label} changed while being inspected: {path}")
    initial_sha256 = hashlib.sha256(payload).hexdigest()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value, initial_sha256


def _validate_frozen_training_contract(
    checkpoint: Mapping[str, Any], mode: str
) -> None:
    """Enforce the preregistered optimizer and model-size contract."""

    expected_step = EXPECTED_TRAINING_CONTRACT["max_steps"]
    if checkpoint.get("step") != expected_step:
        raise ValueError(
            f"{mode} checkpoint step mismatch: expected {expected_step}, "
            f"got {checkpoint.get('step')!r}"
        )
    metrics = _object(checkpoint.get("metrics"), f"{mode} checkpoint metrics")
    if metrics.get("step") != expected_step:
        raise ValueError(
            f"{mode} checkpoint metrics step mismatch: expected {expected_step}, "
            f"got {metrics.get('step')!r}"
        )

    resolved = _object(
        checkpoint.get("resolved_config"), f"{mode} resolved_config"
    )
    sections = {
        "batch_size": (
            _object(resolved.get("loader"), f"{mode} loader"),
            "batch_size",
        ),
        "warmup_steps": (
            _object(resolved.get("optimizer"), f"{mode} optimizer"),
            "warmup_steps",
        ),
        "max_steps": (
            _object(resolved.get("trainer"), f"{mode} trainer"),
            "max_steps",
        ),
        "parameter_count": (
            _object(resolved.get("contract"), f"{mode} contract"),
            "expected_parameter_count",
        ),
        "sigreg_weight": (
            _object(resolved.get("loss"), f"{mode} loss"),
            "sigreg_weight",
        ),
    }
    observed = {"seed": resolved.get("seed")}
    observed.update(
        {
            name: section.get(field)
            for name, (section, field) in sections.items()
        }
    )
    for field, expected in EXPECTED_TRAINING_CONTRACT.items():
        if observed[field] != expected:
            raise ValueError(
                f"{mode} frozen training {field} mismatch: "
                f"expected {expected!r}, got {observed[field]!r}"
            )
    contract = sections["parameter_count"][0]
    if contract.get("optimizer_step_budget") != expected_step:
        raise ValueError(
            f"{mode} optimizer_step_budget mismatch: expected {expected_step}, "
            f"got {contract.get('optimizer_step_budget')!r}"
        )


def _load_metrics_tail(
    metrics_path: str | Path,
    checkpoint_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and bind the final JSONL metric to the latest checkpoint."""

    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    initial_sha256 = _sha256(path)
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
    expected_step = EXPECTED_TRAINING_CONTRACT["max_steps"]
    if tail.get("step") != expected_step:
        raise ValueError(
            f"metrics tail step mismatch: expected {expected_step}, "
            f"got {tail.get('step')!r}"
        )
    if tail != dict(checkpoint_metrics):
        raise ValueError("metrics tail differs from checkpoint metrics")
    if _sha256(path) != initial_sha256:
        raise RuntimeError(f"metrics changed while being inspected: {path}")
    return tail


def _training_blob_signature(head: str, mode: str) -> dict[str, str]:
    if (
        len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
    ):
        raise ValueError(f"{mode} manifest repository HEAD is invalid: {head!r}")
    type_result = subprocess.run(
        ["git", "cat-file", "-t", head],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if type_result.returncode != 0 or type_result.stdout.strip() != b"commit":
        raise ValueError(f"{mode} manifest HEAD is not a commit object: {head}")
    result = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            head,
            "--",
            *TRAINING_SOURCE_PATHS,
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"cannot inspect {mode} manifest HEAD {head}: "
            f"{message or 'git ls-tree failed'}"
        )

    signature: dict[str, str] = {}
    for raw_entry in (entry for entry in result.stdout.split(b"\0") if entry):
        try:
            raw_metadata, raw_path = raw_entry.split(b"\t", 1)
            file_mode, object_type, blob = raw_metadata.decode("ascii").split(" ")
            relative = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("git ls-tree returned an invalid training entry") from error
        if object_type != "blob" or file_mode not in {"100644", "100755"}:
            raise ValueError(f"training closure entry is not a file blob: {relative}")
        signature[relative] = f"{file_mode}:{blob}"

    missing = sorted(set(TRAINING_SOURCE_FILES).difference(signature))
    vendor = [path for path in signature if path.startswith("third_party/lewm/")]
    if missing or not vendor:
        raise ValueError(
            f"{mode} training closure is incomplete: "
            f"missing={missing} vendor_blob_count={len(vendor)}"
        )
    return dict(sorted(signature.items()))


def validate_training_blob_signatures(
    signatures: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    """Require identical training-source blobs across all four checkpoints."""

    expected_modes = set(PARAMETER_COMPATIBLE_TRIPLET_MODES)
    if set(signatures) != expected_modes:
        raise ValueError(
            "training source signatures must cover the complete mode set: "
            f"missing={sorted(expected_modes.difference(signatures))} "
            f"unexpected={sorted(set(signatures).difference(expected_modes))}"
        )
    normalized: dict[str, dict[str, str]] = {}
    for mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        raw_signature = _object(signatures[mode], f"{mode} training signature")
        signature: dict[str, str] = {}
        for raw_path, raw_blob in raw_signature.items():
            if not isinstance(raw_path, str) or not raw_path:
                raise TypeError("training signature paths must be non-empty strings")
            if (
                not isinstance(raw_blob, str)
                or len(raw_blob) != 47
                or raw_blob[:6] not in {"100644", "100755"}
                or raw_blob[6] != ":"
                or any(
                    character not in "0123456789abcdef"
                    for character in raw_blob[7:]
                )
            ):
                raise ValueError(f"invalid training source blob for {raw_path}")
            signature[raw_path] = raw_blob
        missing = sorted(set(TRAINING_SOURCE_FILES).difference(signature))
        vendor = [path for path in signature if path.startswith("third_party/lewm/")]
        if missing or not vendor:
            raise ValueError(
                f"{mode} training source signature is incomplete: "
                f"missing={missing} vendor_blob_count={len(vendor)}"
            )
        normalized[mode] = dict(sorted(signature.items()))

    reference_mode = PARAMETER_COMPATIBLE_TRIPLET_MODES[0]
    reference = normalized[reference_mode]
    for mode in PARAMETER_COMPATIBLE_TRIPLET_MODES[1:]:
        if normalized[mode] != reference:
            differing = sorted(
                path
                for path in set(reference).union(normalized[mode])
                if reference.get(path) != normalized[mode].get(path)
            )
            raise ValueError(
                f"training source blob mismatch between {reference_mode} and "
                f"{mode}: {differing[:8]}"
            )
    return dict(reference)


def _checkpoint_manifest_record(
    checkpoint: Mapping[str, Any], checkpoint_path: Path, mode: str
) -> dict[str, Any]:
    provenance = _object(checkpoint.get("provenance"), f"{mode} provenance")
    expected_sha256 = _require_sha256(
        provenance.get("sha256"), f"{mode} provenance SHA-256"
    )
    embedded = dict(
        _object(provenance.get("manifest"), f"{mode} provenance manifest")
    )
    input_manifest_path = checkpoint_path.parent / "input_manifest.json"
    sidecar, sidecar_sha256 = _load_json_snapshot(
        input_manifest_path, f"{mode} input manifest"
    )
    if sidecar_sha256 != expected_sha256 or sidecar != embedded:
        raise ValueError(f"{mode} input manifest differs from checkpoint")

    external_value = provenance.get("path")
    if not isinstance(external_value, str) or not external_value:
        raise TypeError(f"{mode} external provenance path must be a non-empty string")
    external_path = _resolve(external_value)
    external, external_sha256 = _load_json_snapshot(
        external_path, f"{mode} external input manifest"
    )
    if external_sha256 != expected_sha256 or external != embedded:
        raise ValueError(f"{mode} external input manifest differs from checkpoint")

    repository = _object(embedded.get("repository"), f"{mode} manifest repository")
    if repository.get("dirty") is not False:
        raise ValueError(f"{mode} training manifest must record a clean worktree")
    if repository.get("dirty_diff_sha256") != EMPTY_SHA256:
        raise ValueError(f"{mode} training manifest must record an empty Git diff")
    if repository.get("untracked_files") != []:
        raise ValueError(f"{mode} training manifest must not record untracked files")
    head = repository.get("head")
    if not isinstance(head, str):
        raise TypeError(f"{mode} manifest repository HEAD must be a string")

    validate_manifest(embedded, check_git=False)
    signature = _training_blob_signature(head, mode)
    if _sha256(input_manifest_path) != expected_sha256:
        raise RuntimeError(f"{mode} input manifest changed during preflight")
    if _sha256(external_path) != expected_sha256:
        raise RuntimeError(f"{mode} external input manifest changed during preflight")
    return {
        "input_manifest_path": str(input_manifest_path),
        "input_manifest_sha256": expected_sha256,
        "external_manifest_path": str(external_path),
        "external_manifest_sha256": expected_sha256,
        "manifest_head": head,
        "training_blob_signature": signature,
        "manifest": embedded,
    }


def _checkpoint_record(
    mode: str,
    path: Path,
    *,
    checkpoint: Mapping[str, Any] | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if path.name != "lewm_grounded_latest.pt":
        raise ValueError(f"{mode} must use lewm_grounded_latest.pt")
    if not path.is_file():
        raise FileNotFoundError(path)
    initial_sha256 = _sha256(path)
    if expected_sha256 is not None and initial_sha256 != expected_sha256:
        raise ValueError(f"{mode} checkpoint differs from ledger before torch.load")
    if checkpoint is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{mode} checkpoint must be an object")
    if _sha256(path) != initial_sha256:
        raise RuntimeError(f"{mode} checkpoint changed while being inspected")
    _validate_frozen_training_contract(checkpoint, mode)
    contract = resolve_anchor_visual_ablation_contract(
        checkpoint,
        path,
        mode,
        split="validation",
        grounding_mode_request="checkpoint",
    )
    if contract["train_mode"] != mode:
        recorded_mode = contract.get("train_mode")
        raise ValueError(
            f"checkpoint directory {mode!r} contains train mode "
            f"{recorded_mode!r}"
        )
    resolved = deepcopy(
        dict(_object(checkpoint.get("resolved_config"), "resolved_config"))
    )
    data_config = deepcopy(
        dict(_object(checkpoint.get("data_config"), "data_config"))
    )
    state = _object(checkpoint.get("model_state"), "model_state")
    state_signature = {
        str(key): (tuple(value.shape), str(value.dtype))
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    checkpoint_metrics = _object(
        checkpoint.get("metrics"), f"{mode} checkpoint metrics"
    )
    metrics_path = path.parent / "metrics.jsonl"
    metrics_tail = _load_metrics_tail(metrics_path, checkpoint_metrics)
    metrics_sha256 = _sha256(metrics_path)
    manifest_record = _checkpoint_manifest_record(checkpoint, path, mode)
    if _sha256(path) != initial_sha256:
        raise RuntimeError(f"{mode} checkpoint changed during preflight")
    if _sha256(metrics_path) != metrics_sha256:
        raise RuntimeError(f"{mode} metrics changed during preflight")
    return {
        "mode": mode,
        "path": str(path),
        "sha256": initial_sha256,
        "contract": contract,
        "resolved_config": resolved,
        "data_config": data_config,
        "state_signature": state_signature,
        "metrics_path": str(metrics_path),
        "metrics_sha256": metrics_sha256,
        "metrics_tail": metrics_tail,
        **manifest_record,
    }


def validate_checkpoint_set(ledger_path: str | Path) -> dict[str, Any]:
    """Load the ledger-pinned endpoints and enforce the matched-four contract."""

    ledger_set = load_checkpoint_ledger(
        ledger_path,
        repo_root=REPO_ROOT,
        require_git_tracked=True,
    )
    records = {}
    for mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        runtime = ledger_set["runtimes"][mode]
        ledger_record = ledger_set["records"][mode]
        checkpoint_artifact = ledger_record["checkpoint"]
        record = _checkpoint_record(
            mode,
            runtime["checkpoint_path"],
            checkpoint=runtime["checkpoint"],
            expected_sha256=checkpoint_artifact["sha256"],
        )
        record["ledger_record"] = ledger_record
        records[mode] = record

    reference = records[PARAMETER_COMPATIBLE_TRIPLET_MODES[0]]
    canonical = ledger_set["matched_training_contract"][
        "canonical_resolved_config"
    ]
    data_config = ledger_set["matched_training_contract"]["data_config"]
    signature = reference["state_signature"]
    for mode, record in records.items():
        if _canonical_resolved_config(record["resolved_config"]) != canonical:
            raise ValueError(f"{mode} checkpoint violates the ledger config contract")
        if record["data_config"] != data_config:
            raise ValueError(f"{mode} checkpoint uses a different data contract")
        if record["state_signature"] != signature:
            raise ValueError(f"{mode} checkpoint has incompatible state-dict shapes")
    training_blob_signature = validate_training_blob_signatures(
        {
            mode: record["training_blob_signature"]
            for mode, record in records.items()
        }
    )
    return {
        "ledger_path": ledger_set["path"],
        "ledger_sha256": ledger_set["sha256"],
        "ledger_git": ledger_set["git"],
        "records": records,
        "data_config": data_config,
        "training_blob_signature": training_blob_signature,
        "matched_training_contract": ledger_set["matched_training_contract"],
    }


def validate_result_dir_is_ignored(
    result_dir: str | Path,
    *,
    create: bool = True,
) -> Path:
    path = _resolve(result_dir)
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("result directory must be inside the repository") from error
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=REPO_ROOT,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(f"result directory is not Git-ignored: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _source_arguments(data_config: Mapping[str, Any]) -> list[str]:
    sources = data_config.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise TypeError("checkpoint data sources must be a sequence")
    if len(sources) != 4:
        raise ValueError(f"anchor-visual matrix requires four sources, got {len(sources)}")
    arguments: list[str] = []
    for source in sources:
        value = _object(source, "checkpoint source")
        for flag, field in (
            ("--data", "path"),
            ("--correction-cache", "correction_cache"),
            ("--oracle-target-cache", "oracle_target_cache"),
        ):
            path = value.get(field)
            if not isinstance(path, str) or not _resolve(path).exists():
                raise FileNotFoundError(f"checkpoint source {field} is missing: {path!r}")
            arguments.extend([flag, str(_resolve(path))])
    return arguments


def build_evaluator_command(
    *,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    eval_mode: str,
    data_config: Mapping[str, Any],
    output: str | Path,
    device: str,
    batch_size: int,
    num_workers: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/evaluate_lewm_grounded.py"),
        "--checkpoint",
        str(_resolve(checkpoint)),
        "--expected-checkpoint-sha256",
        _require_sha256(
            expected_checkpoint_sha256, "expected checkpoint SHA-256"
        ),
        "--split",
        "validation",
        "--grounding-mode",
        "checkpoint",
        "--anchor-visual-mode-override",
        eval_mode,
        "--rollout-steps",
        "5",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--bootstrap-seed",
        str(bootstrap_seed),
        "--device",
        device,
        "--output",
        str(_resolve(output)),
    ]
    command.extend(_source_arguments(data_config))
    return command


def _cell_contract_path(output: Path) -> Path:
    return output.with_suffix(".contract.json")


def _cell_completion_path(output: Path) -> Path:
    return output.with_suffix(".completion.json")


def _cell_contract(
    *,
    train_mode: str,
    eval_mode: str,
    checkpoint: Mapping[str, Any],
    ledger_path: Path,
    ledger_sha256: str,
    output: Path,
    dev_protocol_path: Path,
    dev_protocol_sha256: str,
    corrector_slice_path: Path,
    corrector_slice_sha256: str,
    evaluator_sha256: str,
    execution_code_sha256: str,
    matrix_runner_sha256: str,
    repository_state: Mapping[str, Any],
    frozen_input_artifacts_sha256: str,
    evaluator_parameters: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "protocol": CELL_CONTRACT_PROTOCOL,
        "repository_head": repository_state["head"],
        "repository_worktree_sha256": repository_state["worktree_sha256"],
        "frozen_input_artifacts_sha256": _require_sha256(
            frozen_input_artifacts_sha256, "frozen input artifacts SHA-256"
        ),
        "checkpoint_ledger_path": str(ledger_path),
        "checkpoint_ledger_sha256": _require_sha256(
            ledger_sha256, "checkpoint ledger SHA-256"
        ),
        "development_only": True,
        "train_mode": train_mode,
        "eval_mode": eval_mode,
        "checkpoint": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_step": 2000,
        "metrics_path": checkpoint["metrics_path"],
        "metrics_sha256": checkpoint["metrics_sha256"],
        "input_manifest_path": checkpoint["input_manifest_path"],
        "input_manifest_sha256": checkpoint["input_manifest_sha256"],
        "dev_protocol_path": str(dev_protocol_path),
        "dev_protocol_sha256": dev_protocol_sha256,
        "corrector_slice_path": str(corrector_slice_path),
        "corrector_slice_sha256": corrector_slice_sha256,
        "evaluator_sha256": evaluator_sha256,
        "execution_code_sha256": execution_code_sha256,
        "matrix_runner_sha256": matrix_runner_sha256,
        "evaluator_parameters": dict(evaluator_parameters),
        "output": str(output),
        "command": list(command),
        "target_sha256": ANCHOR_ABLATION_TARGET_SHA256,
        "data_split_sha256": ANCHOR_ABLATION_SPLIT_SHA256,
    }


def normalize_cell_report(
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    annotations: Mapping[tuple[int, str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one evaluator report and return the matrix-summary record."""

    def strict_int(value: object, label: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
        return value

    def finite(
        value: object,
        label: str,
        *,
        non_negative: bool = False,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{label} must be finite")
        if non_negative and result < 0.0:
            raise ValueError(f"{label} must be non-negative")
        return result

    def close(actual: object, expected: float, label: str) -> None:
        observed = finite(actual, label)
        if not math.isclose(
            observed,
            expected,
            rel_tol=1e-11,
            abs_tol=1e-14,
        ):
            raise ValueError(
                f"{label} disagrees with episode-block recomputation: "
                f"expected {expected!r}, got {observed!r}"
            )

    def ratio(numerator: float, denominator: float, label: str) -> float:
        if denominator <= 0.0:
            raise ValueError(f"{label} denominator must be positive")
        return numerator / denominator

    if contract.get("schema_version") != 4 or (
        contract.get("protocol") != CELL_CONTRACT_PROTOCOL
    ):
        raise ValueError("cell contract protocol mismatch")
    ledger_path = contract.get("checkpoint_ledger_path")
    if not isinstance(ledger_path, str) or not ledger_path:
        raise ValueError("cell contract has no checkpoint ledger path")
    _require_sha256(
        contract.get("checkpoint_ledger_sha256"),
        "cell contract checkpoint ledger SHA-256",
    )
    if report.get("schema_version") != 1:
        raise ValueError("cell report schema_version must be 1")
    if report.get("protocol") != "grounded_lewm_offline_diagnostics":
        raise ValueError("cell report protocol mismatch")
    ablation = _object(report.get("anchor_visual_ablation"), "ablation report")
    evaluation = _object(report.get("evaluation"), "evaluation report")
    metrics = _object(report.get("metrics"), "metrics report")
    provenance = _object(report.get("provenance"), "provenance report")
    execution_code = _object(report.get("execution_code"), "execution code report")
    expected = {
        "train_mode": contract["train_mode"],
        "eval_mode": contract["eval_mode"],
        "step": 2000,
        "parameter_count": ANCHOR_ABLATION_PARAMETER_COUNT,
        "seed": ANCHOR_ABLATION_SEED,
        "target_sha256": ANCHOR_ABLATION_TARGET_SHA256,
        "data_split_sha256": ANCHOR_ABLATION_SPLIT_SHA256,
        "development_only": True,
        "formal_gate_eligible": False,
    }
    for field, value in expected.items():
        if ablation.get(field) != value:
            raise ValueError(
                f"cell ablation field {field} mismatch: "
                f"expected {value!r}, got {ablation.get(field)!r}"
            )
    if ablation.get("active") is not True:
        raise ValueError("cell did not apply the development ablation contract")
    if report.get("checkpoint_sha256") != contract["checkpoint_sha256"]:
        raise ValueError("cell checkpoint SHA-256 differs from its contract")
    if report.get("checkpoint") != contract["checkpoint"]:
        raise ValueError("cell checkpoint path differs from its contract")
    if report.get("checkpoint_step") != contract["checkpoint_step"]:
        raise ValueError("cell checkpoint step differs from its contract")
    if evaluation.get("split") != "validation" or (
        evaluation.get("max_batches") is not None
    ):
        raise ValueError("cell must cover the full checkpoint validation split")
    if evaluation.get("evaluator_sha256") != contract["evaluator_sha256"]:
        raise ValueError("cell evaluator SHA-256 differs from its contract")
    if execution_code.get("aggregate_sha256") != contract["execution_code_sha256"]:
        raise ValueError("cell execution-code SHA-256 differs from its contract")

    pairing = _object(evaluation.get("action_shuffle_pairing"), "action pairing")
    expected_pairing = {
        "same_source": True,
        "same_task": True,
        "different_episode": True,
        "coverage": 1.0,
    }
    for field, expected_value in expected_pairing.items():
        if pairing.get(field) != expected_value:
            raise ValueError(f"cell action-shuffle pairing {field} mismatch")
    if provenance.get("validated") is not True:
        raise ValueError("cell input provenance was not validated")
    if provenance.get("git_state_validated") is not False:
        raise ValueError(
            "development cell must explicitly label skipped Git-state binding"
        )
    if provenance.get("manifest_sha256") != contract["input_manifest_sha256"]:
        raise ValueError("cell input manifest SHA-256 differs from its contract")
    if provenance.get("run_sidecar") != contract["input_manifest_path"]:
        raise ValueError("cell input manifest path differs from its contract")

    parameters = _object(
        contract.get("evaluator_parameters"), "cell evaluator parameters"
    )
    for field in ("split", "batch_size", "num_workers", "device", "max_batches"):
        if evaluation.get(field) != parameters.get(field):
            raise ValueError(f"cell evaluator parameter {field} mismatch")
    if report.get("grounding_mode_request") != parameters.get("grounding_mode"):
        raise ValueError("cell evaluator parameter grounding_mode mismatch")
    sequence = _object(report.get("sequence"), "cell sequence report")
    expected_sequence = {
        "rollout_steps": parameters.get("rollout_steps"),
        "endpoint_horizons": [1, 3, 5],
        "shared_eligible_windows": True,
    }
    for field, expected_value in expected_sequence.items():
        if sequence.get(field) != expected_value:
            raise ValueError(f"cell sequence field {field} mismatch")
    report_bootstrap = _object(report.get("bootstrap"), "cell bootstrap report")
    for field, parameter_field in (
        ("samples", "bootstrap_samples"),
        ("seed", "bootstrap_seed"),
    ):
        if report_bootstrap.get(field) != parameters.get(parameter_field):
            raise ValueError(f"cell bootstrap {field} mismatch")
    metrics_bootstrap = _object(metrics.get("bootstrap"), "metrics bootstrap report")
    if dict(metrics_bootstrap) != dict(report_bootstrap):
        raise ValueError("cell metrics/report bootstrap payloads differ")

    source_windows_value = evaluation.get("source_windows")
    if not isinstance(source_windows_value, list) or not source_windows_value:
        raise TypeError("evaluation.source_windows must be a non-empty integer list")
    source_windows = [
        strict_int(value, f"evaluation.source_windows[{index}]", minimum=1)
        for index, value in enumerate(source_windows_value)
    ]
    eligible_windows = strict_int(
        evaluation.get("eligible_windows"),
        "evaluation.eligible_windows",
        minimum=1,
    )
    if sum(source_windows) != eligible_windows:
        raise ValueError(
            "evaluation.eligible_windows must equal sum(evaluation.source_windows)"
        )
    for field in (
        "num_sequences",
        "num_eligible_sequences",
        "num_action_shuffle_sequences",
    ):
        if strict_int(metrics.get(field), f"metrics.{field}", minimum=1) != (
            eligible_windows
        ):
            raise ValueError(f"metrics.{field} must equal evaluation.eligible_windows")
    if finite(
        metrics.get("action_shuffle_coverage"),
        "metrics.action_shuffle_coverage",
    ) != 1.0:
        raise ValueError("metrics.action_shuffle_coverage must be 1.0")
    if metrics.get("endpoint_horizons") != [1, 3, 5]:
        raise ValueError("metrics.endpoint_horizons mismatch")
    if metrics.get("rollout_steps") != sequence.get("rollout_steps"):
        raise ValueError("metrics.rollout_steps differs from sequence contract")

    required_metrics = (
        "teacher_forced_pred_mse",
        "identity_copy_mse",
        "oracle_alignment_mse",
        "action_shuffle_mse",
        "action_shuffle_delta_mse",
        "action_sensitivity_mse",
    )
    for field in required_metrics:
        finite(
            metrics.get(field),
            f"cell metric {field}",
            non_negative=field not in {"action_shuffle_delta_mse"},
        )
    token_metrics = _object(metrics.get("oracle_token_mse"), "oracle token metrics")
    for name in ("prediction", "alignment"):
        summary = _object(token_metrics.get(name), f"oracle token {name}")
        anchors = summary.get("anchor_mse")
        if not isinstance(anchors, list) or len(anchors) != 3:
            raise ValueError(f"oracle token {name} must contain three anchor MSEs")
        for index, value in enumerate(anchors):
            finite(
                value,
                f"oracle token {name}.anchor_mse[{index}]",
                non_negative=True,
            )

    aggregation = _object(metrics.get("aggregation"), "metrics aggregation")
    expected_aggregation = {
        "error_accumulation": "float64_sse_then_element_count",
        "window_to_episode": "element_weighted_mse",
        "episode_to_task": "equal_episode_mean",
        "task_to_macro": "equal_task_mean",
        "ratio_macro": "equal_mean_of_task_ratios",
    }
    for field, expected_value in expected_aggregation.items():
        if aggregation.get(field) != expected_value:
            raise ValueError(f"metrics aggregation.{field} mismatch")

    episode_blocks = metrics.get("episode_blocks")
    if not isinstance(episode_blocks, list) or not episode_blocks:
        raise ValueError("cell metrics must contain non-empty episode_blocks")
    if any(not isinstance(block, Mapping) for block in episode_blocks):
        raise TypeError("every cell episode block must be an object")

    normalized_blocks: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str, int]] = set()
    source_sequence_counts = [0 for _ in source_windows]
    source_tasks: dict[int, set[str]] = {}
    task_sources: dict[str, set[int]] = {}
    task_blocks: dict[str, list[dict[str, Any]]] = {}
    denominator_factors: dict[str, int] = {}

    def bind_denominator(
        count: int,
        sequences: int,
        label: str,
    ) -> None:
        if count % sequences:
            raise ValueError(f"{label} must be divisible by block num_sequences")
        factor = count // sequences
        previous = denominator_factors.setdefault(label, factor)
        if previous != factor:
            raise ValueError(f"{label} per-sequence element count changed")

    for index, raw_block in enumerate(episode_blocks):
        block = dict(_object(raw_block, f"episode block {index}"))
        source_index = strict_int(
            block.get("source_index"),
            f"episode block {index} source_index",
        )
        if source_index >= len(source_windows):
            raise ValueError(f"episode block {index} source_index is out of range")
        task_name = block.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError(
                f"episode block {index} task_name must be a non-empty string"
            )
        if task_name != task_name.strip():
            raise ValueError(f"episode block {index} task_name is not canonical")
        episode_id = strict_int(
            block.get("episode_id"),
            f"episode block {index} episode_id",
        )
        key = (source_index, task_name, episode_id)
        if key in seen_keys:
            raise ValueError(f"duplicate episode block key: {key}")
        seen_keys.add(key)
        sequences = strict_int(
            block.get("num_sequences"),
            f"episode block {index} num_sequences",
            minimum=1,
        )
        shuffle_sequences = strict_int(
            block.get("num_action_shuffle_sequences"),
            f"episode block {index} num_action_shuffle_sequences",
            minimum=1,
        )
        if shuffle_sequences != sequences:
            raise ValueError(
                f"episode block {index} action-shuffle sequence count mismatch"
            )
        source_sequence_counts[source_index] += sequences
        source_tasks.setdefault(source_index, set()).add(task_name)
        task_sources.setdefault(task_name, set()).add(source_index)

        teacher = _object(
            block.get("teacher_forced"),
            f"episode block {index} teacher_forced",
        )
        teacher_count = strict_int(
            teacher.get("element_count"),
            f"episode block {index} teacher_forced.element_count",
            minimum=1,
        )
        bind_denominator(teacher_count, sequences, "teacher_forced.element_count")
        teacher_predicted = finite(
            teacher.get("predicted_mse"),
            f"episode block {index} teacher_forced.predicted_mse",
            non_negative=True,
        )
        teacher_identity = finite(
            teacher.get("identity_copy_mse"),
            f"episode block {index} teacher_forced.identity_copy_mse",
            non_negative=True,
        )
        teacher_shuffled = finite(
            teacher.get("action_shuffle_mse"),
            f"episode block {index} teacher_forced.action_shuffle_mse",
            non_negative=True,
        )
        teacher_sensitivity = finite(
            teacher.get("action_sensitivity_mse"),
            f"episode block {index} teacher_forced.action_sensitivity_mse",
            non_negative=True,
        )
        close(
            teacher.get("prediction_vs_copy_ratio"),
            ratio(
                teacher_predicted,
                teacher_identity,
                f"episode block {index} prediction_vs_copy_ratio",
            ),
            f"episode block {index} teacher_forced.prediction_vs_copy_ratio",
        )
        close(
            teacher.get("action_shuffle_delta_mse"),
            teacher_shuffled - teacher_predicted,
            f"episode block {index} teacher_forced.action_shuffle_delta_mse",
        )

        alignment = _object(
            block.get("oracle_alignment"),
            f"episode block {index} oracle_alignment",
        )
        alignment_count = strict_int(
            alignment.get("element_count"),
            f"episode block {index} oracle_alignment.element_count",
            minimum=1,
        )
        bind_denominator(
            alignment_count,
            sequences,
            "oracle_alignment.element_count",
        )
        finite(
            alignment.get("mse"),
            f"episode block {index} oracle_alignment.mse",
            non_negative=True,
        )

        endpoints = _object(
            block.get("endpoints"),
            f"episode block {index} endpoints",
        )
        if set(endpoints) != {"1", "3", "5"}:
            raise ValueError(
                f"episode block {index} endpoints must be exactly H=1/3/5"
            )
        endpoint_values: dict[str, tuple[int, float, float]] = {}
        for horizon in ("1", "3", "5"):
            endpoint = _object(
                endpoints.get(horizon),
                f"episode block {index} endpoint {horizon}",
            )
            endpoint_count = strict_int(
                endpoint.get("element_count"),
                f"episode block {index} endpoint {horizon}.element_count",
                minimum=1,
            )
            bind_denominator(
                endpoint_count,
                sequences,
                f"endpoint.{horizon}.element_count",
            )
            predicted = finite(
                endpoint.get("predicted_mse"),
                f"episode block {index} endpoint {horizon}.predicted_mse",
                non_negative=True,
            )
            identity_value = finite(
                endpoint.get("hold_last_identity_mse"),
                f"episode block {index} endpoint {horizon}.hold_last_identity_mse",
                non_negative=True,
            )
            close(
                endpoint.get("ratio"),
                ratio(
                    predicted,
                    identity_value,
                    f"episode block {index} endpoint {horizon}.ratio",
                ),
                f"episode block {index} endpoint {horizon}.ratio",
            )
            endpoint_values[horizon] = (
                endpoint_count,
                predicted,
                identity_value,
            )

        recursive_shuffle = _object(
            block.get("recursive_h1_action_shuffle"),
            f"episode block {index} recursive_h1_action_shuffle",
        )
        recursive_count = strict_int(
            recursive_shuffle.get("element_count"),
            (
                f"episode block {index} "
                "recursive_h1_action_shuffle.element_count"
            ),
            minimum=1,
        )
        bind_denominator(
            recursive_count,
            sequences,
            "recursive_h1_action_shuffle.element_count",
        )
        if recursive_count != endpoint_values["1"][0]:
            raise ValueError(
                f"episode block {index} recursive H1 and endpoint H1 "
                "element_count differ"
            )
        recursive_mse = finite(
            recursive_shuffle.get("mse"),
            f"episode block {index} recursive_h1_action_shuffle.mse",
            non_negative=True,
        )
        close(
            recursive_shuffle.get("delta_mse"),
            recursive_mse - endpoint_values["1"][1],
            f"episode block {index} recursive_h1_action_shuffle.delta_mse",
        )
        normalized_blocks.append(block)
        task_blocks.setdefault(task_name, []).append(block)

    endpoint_factors = {
        denominator_factors[f"endpoint.{horizon}.element_count"]
        for horizon in ("1", "3", "5")
    }
    if len(endpoint_factors) != 1:
        raise ValueError("H=1/3/5 endpoint per-sequence element counts differ")
    if denominator_factors["recursive_h1_action_shuffle.element_count"] not in (
        endpoint_factors
    ):
        raise ValueError("recursive H1 denominator differs from endpoint denominators")
    if source_sequence_counts != source_windows:
        raise ValueError(
            "episode-block per-source sequence counts differ from source_windows"
        )
    if sum(
        strict_int(block["num_sequences"], "episode block num_sequences", minimum=1)
        for block in normalized_blocks
    ) != eligible_windows:
        raise ValueError("episode-block num_sequences do not close to eligible_windows")
    if any(len(tasks) != 1 for tasks in source_tasks.values()) or (
        set(source_tasks) != set(range(len(source_windows)))
    ):
        raise ValueError("each source_index must map to exactly one task_name")
    if any(len(sources) != 1 for sources in task_sources.values()):
        raise ValueError("each task_name must map to exactly one source_index")

    per_task = _object(metrics.get("per_task"), "metrics.per_task")
    if set(per_task) != set(task_blocks):
        raise ValueError("metrics.per_task membership differs from episode blocks")
    for task_name, blocks in task_blocks.items():
        task_metric = _object(per_task.get(task_name), f"metrics.per_task.{task_name}")
        expected_task_counts = {
            "num_sources": len(task_sources[task_name]),
            "num_episodes": len(blocks),
            "num_sequences": sum(
                strict_int(
                    block["num_sequences"],
                    f"{task_name} block num_sequences",
                    minimum=1,
                )
                for block in blocks
            ),
        }
        for field, expected_value in expected_task_counts.items():
            if task_metric.get(field) != expected_value:
                raise ValueError(
                    f"metrics.per_task.{task_name}.{field} count mismatch"
                )

    macro = _object(metrics.get("macro"), "metrics.macro")
    expected_macro_counts = {
        "num_tasks": len(task_blocks),
        "num_episodes": len(normalized_blocks),
        "num_sequences": eligible_windows,
    }
    for field, expected_value in expected_macro_counts.items():
        if macro.get(field) != expected_value:
            raise ValueError(f"metrics.macro.{field} count mismatch")
    report_per_task = _object(report.get("per_task"), "report.per_task")
    report_macro = _object(report.get("macro"), "report.macro")
    if dict(report_per_task) != dict(per_task) or dict(report_macro) != dict(macro):
        raise ValueError("report and metrics per_task/macro payloads differ")
    if metrics.get("endpoint_metrics") != macro.get("endpoints"):
        raise ValueError("metrics.endpoint_metrics differs from metrics.macro.endpoints")

    if annotations is not None:
        normalized_blocks = annotate_episode_blocks(normalized_blocks, annotations)

    def pooled_mse(section: str, field: str) -> float:
        squared_error = 0.0
        element_count = 0
        for index, block in enumerate(normalized_blocks):
            payload = _object(
                block.get(section),
                f"episode block {index} {section}",
            )
            count = strict_int(
                payload.get("element_count"),
                f"episode block {index} {section}.element_count",
                minimum=1,
            )
            value = finite(
                payload.get(field),
                f"episode block {index} {section}.{field}",
            )
            squared_error += value * count
            element_count += count
        return squared_error / element_count

    teacher_predicted = pooled_mse("teacher_forced", "predicted_mse")
    teacher_identity = pooled_mse("teacher_forced", "identity_copy_mse")
    teacher_shuffled = pooled_mse("teacher_forced", "action_shuffle_mse")
    recomputed_metrics = {
        "teacher_forced_pred_mse": teacher_predicted,
        "identity_copy_mse": teacher_identity,
        "action_shuffle_mse": teacher_shuffled,
        "action_shuffle_delta_mse": teacher_shuffled - teacher_predicted,
        "action_sensitivity_mse": pooled_mse(
            "teacher_forced",
            "action_sensitivity_mse",
        ),
        "oracle_alignment_mse": pooled_mse("oracle_alignment", "mse"),
    }
    for field, expected_value in recomputed_metrics.items():
        close(metrics.get(field), expected_value, f"cell metric {field}")

    normalized_metrics = dict(metrics)
    normalized_metrics["episode_blocks"] = normalized_blocks
    return {
        **dict(ablation),
        "metrics": normalized_metrics,
        "episode_blocks": normalized_blocks,
        "eligible_windows": eligible_windows,
        "source_windows": source_windows,
        "report_sha256": _sha256(contract["output"]),
    }


def _completion_record(output: Path, contract_path: Path) -> dict[str, Any]:
    if not output.is_file():
        raise FileNotFoundError(output)
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract, _ = _load_json_snapshot(contract_path, "cell contract")
    return {
        "schema_version": 3,
        "protocol": CELL_COMPLETION_PROTOCOL,
        "repository_head": contract.get("repository_head"),
        "repository_worktree_sha256": contract.get(
            "repository_worktree_sha256"
        ),
        "frozen_input_artifacts_sha256": contract.get(
            "frozen_input_artifacts_sha256"
        ),
        "checkpoint_ledger_path": contract.get("checkpoint_ledger_path"),
        "checkpoint_ledger_sha256": contract.get("checkpoint_ledger_sha256"),
        "output_path": str(output),
        "output_sha256": _sha256(output),
        "contract_path": str(contract_path),
        "contract_sha256": _sha256(contract_path),
    }


def _load_reusable_cell(
    output: Path,
    contract: Mapping[str, Any],
    *,
    annotations: Mapping[tuple[int, str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    contract_path = _cell_contract_path(output)
    completion_path = _cell_completion_path(output)
    artifacts = (output, contract_path, completion_path)
    present = tuple(path.exists() for path in artifacts)
    if not any(present):
        return None
    if not all(present) or any(not path.is_file() for path in artifacts):
        raise RuntimeError(
            "partial cell artifacts found; expected report, contract, and completion: "
            f"{output}"
        )

    stored_contract, contract_sha256 = _load_json_snapshot(
        contract_path, "cell contract"
    )
    if stored_contract != dict(contract):
        raise ValueError(f"cell contract differs from this run: {contract_path}")
    completion, completion_sha256 = _load_json_snapshot(
        completion_path, "cell completion"
    )
    if completion.get("schema_version") != 3 or (
        completion.get("protocol") != CELL_COMPLETION_PROTOCOL
    ):
        raise ValueError("cell completion protocol mismatch")
    for field in (
        "repository_head",
        "repository_worktree_sha256",
        "frozen_input_artifacts_sha256",
    ):
        if completion.get(field) != contract.get(field):
            raise ValueError(f"cell completion {field} binding mismatch")
    if (
        completion.get("checkpoint_ledger_path")
        != contract.get("checkpoint_ledger_path")
        or completion.get("checkpoint_ledger_sha256")
        != contract.get("checkpoint_ledger_sha256")
    ):
        raise ValueError("cell completion checkpoint-ledger binding mismatch")
    if completion.get("output_path") != str(output):
        raise ValueError("cell completion output path mismatch")
    if completion.get("contract_path") != str(contract_path):
        raise ValueError("cell completion contract path mismatch")
    output_sha256 = _sha256(output)
    if completion.get("output_sha256") != output_sha256:
        raise ValueError("cell output SHA-256 differs from completion sidecar")
    if completion.get("contract_sha256") != contract_sha256:
        raise ValueError("cell contract SHA-256 differs from completion sidecar")

    report, report_sha256 = _load_json_snapshot(output, "cell output")
    if report_sha256 != output_sha256:
        raise RuntimeError(f"cell output changed while being loaded: {output}")
    normalized = normalize_cell_report(
        report,
        contract,
        annotations=annotations,
    )
    if _sha256(output) != output_sha256:
        raise RuntimeError(f"cell output changed while being validated: {output}")
    if _sha256(contract_path) != contract_sha256:
        raise RuntimeError(
            f"cell contract changed while being validated: {contract_path}"
        )
    if _sha256(completion_path) != completion_sha256:
        raise RuntimeError(
            f"cell completion changed while being validated: {completion_path}"
        )
    normalized["sealed_artifacts"] = {
        "output_sha256": output_sha256,
        "contract_sha256": contract_sha256,
        "completion_sha256": completion_sha256,
    }
    return normalized


def _seal_cell_output(
    output: Path,
    contract: Mapping[str, Any],
    *,
    annotations: Mapping[tuple[int, str, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    contract_path = _cell_contract_path(output)
    completion_path = _cell_completion_path(output)
    if completion_path.exists():
        raise FileExistsError(f"cell completion already exists: {completion_path}")
    stored_contract, contract_sha256 = _load_json_snapshot(
        contract_path, "cell contract"
    )
    if stored_contract != dict(contract):
        raise ValueError(f"cell contract differs from this run: {contract_path}")
    report, output_sha256 = _load_json_snapshot(output, "cell output")
    normalize_cell_report(report, contract, annotations=annotations)
    if _sha256(output) != output_sha256 or _sha256(contract_path) != contract_sha256:
        raise RuntimeError("cell artifacts changed before completion was written")
    _write_json_idempotent(
        completion_path,
        _completion_record(output, contract_path),
    )
    reusable = _load_reusable_cell(output, contract, annotations=annotations)
    if reusable is None:
        raise RuntimeError(f"failed to seal cell output: {output}")
    return reusable


def _validate_annotation_counts(
    annotations: Mapping[tuple[int, str, int], Mapping[str, Any]],
    expected_full_counts: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    full = {"clean": 0, "mild": 0, "total": 0}
    unseen = {"clean": 0, "mild": 0, "total": 0}
    for key, raw_annotation in annotations.items():
        annotation = _object(raw_annotation, f"episode annotation {key!r}")
        condition = annotation.get("camera_condition")
        if condition not in {"clean", "mild"}:
            raise ValueError(f"episode annotation {key!r} has invalid camera_condition")
        corrector_unseen = annotation.get("corrector_train_unseen")
        if not isinstance(corrector_unseen, bool):
            raise TypeError(
                f"episode annotation {key!r} corrector_train_unseen must be boolean"
            )
        full[str(condition)] += 1
        full["total"] += 1
        if corrector_unseen:
            unseen[str(condition)] += 1
            unseen["total"] += 1

    expected_full = {
        field: int(expected_full_counts[field])
        for field in ("clean", "mild", "total")
    }
    if full != expected_full:
        raise ValueError(
            f"full validation annotation counts differ: "
            f"expected {expected_full}, got {full}"
        )
    if unseen != EXPECTED_UNSEEN_COUNTS:
        raise ValueError(
            "corrector-train-unseen counts differ: "
            f"expected {EXPECTED_UNSEEN_COUNTS}, got {unseen}"
        )
    return {"full": full, "corrector_train_unseen": unseen}


def _validate_dev_protocol(protocol: Mapping[str, Any]) -> None:
    validate_anchor_visual_dev_protocol(protocol)
    if protocol.get("schema_version") != 1 or (
        protocol.get("protocol") != "fcpg_lewm_anchor_visual_dev_selection_v1"
    ):
        raise ValueError("unexpected anchor-visual development protocol")
    claim = _object(protocol.get("claim_boundary"), "claim_boundary")
    if claim.get("development_only") is not True:
        raise ValueError("development protocol must be development_only")

    data = _object(protocol.get("data"), "protocol data")
    full_counts = dict(
        _object(data.get("full_validation_counts"), "full_validation_counts")
    )
    if full_counts != EXPECTED_FULL_VALIDATION_COUNTS:
        raise ValueError(
            "full_validation_counts mismatch: "
            f"expected {EXPECTED_FULL_VALIDATION_COUNTS}, got {full_counts}"
        )
    unseen_counts = dict(
        _object(
            data.get("corrector_train_unseen_counts"),
            "corrector_train_unseen_counts",
        )
    )
    if unseen_counts != EXPECTED_UNSEEN_COUNTS:
        raise ValueError(
            "corrector_train_unseen_counts mismatch: "
            f"expected {EXPECTED_UNSEEN_COUNTS}, got {unseen_counts}"
        )
    if data.get("development_split") != "validation":
        raise ValueError("development_split must be validation")
    if data.get("episode_manifest_sha256") != ANCHOR_ABLATION_SPLIT_SHA256:
        raise ValueError("protocol episode manifest SHA-256 mismatch")
    for field in ("episode_manifest", "corrector_train_unseen_slice"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise TypeError(f"protocol data.{field} must be a non-empty path")

    training = _object(protocol.get("training_contract"), "training_contract")
    expected_training = {
        "seed": EXPECTED_TRAINING_CONTRACT["seed"],
        "batch_size": EXPECTED_TRAINING_CONTRACT["batch_size"],
        "warmup_steps": EXPECTED_TRAINING_CONTRACT["warmup_steps"],
        "optimizer_steps": EXPECTED_TRAINING_CONTRACT["max_steps"],
        "parameter_count": EXPECTED_TRAINING_CONTRACT["parameter_count"],
        "sigreg_weight": EXPECTED_TRAINING_CONTRACT["sigreg_weight"],
        "latest_checkpoint_only": True,
    }
    for field, expected in expected_training.items():
        if training.get(field) != expected:
            raise ValueError(
                f"training_contract.{field} mismatch: "
                f"expected {expected!r}, got {training.get(field)!r}"
            )

    bootstrap = _object(protocol.get("bootstrap"), "bootstrap protocol")
    expected_bootstrap = {
        "samples": 10_000,
        "seed": 5120,
        "confidence": 0.95,
        "unit": "episode_block_within_task",
    }
    for field, expected in expected_bootstrap.items():
        if bootstrap.get(field) != expected:
            raise ValueError(
                f"bootstrap.{field} mismatch: "
                f"expected {expected!r}, got {bootstrap.get(field)!r}"
            )


def _validate_corrector_slice(
    corrector_slice: Mapping[str, Any],
    protocol_data: Mapping[str, Any],
) -> None:
    if corrector_slice.get("schema_version") != 1 or (
        corrector_slice.get("protocol")
        != "fcpg_anchor_visual_corrector_train_unseen_slice_v1"
    ):
        raise ValueError("unexpected corrector-train-unseen slice protocol")
    if corrector_slice.get("development_only") is not True:
        raise ValueError("corrector-train-unseen slice must be development_only")
    if (
        corrector_slice.get("lewm_validation_manifest_sha256")
        != protocol_data.get("episode_manifest_sha256")
    ):
        raise ValueError("corrector slice validation-manifest hash mismatch")
    if (
        corrector_slice.get("episode_set_sha256")
        != protocol_data.get("episode_set_sha256")
    ):
        raise ValueError("corrector slice episode-set hash mismatch")
    expected_checkpoint_sha256 = (
        "3cfe621bbfb5b87962a3c569d6ebf9d4d702e0483adeebd45a4e00bb0a16ee79"
    )
    if corrector_slice.get("corrector_checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("corrector slice checkpoint SHA-256 mismatch")
    if corrector_slice.get("corrector_split") != {
        "split_seed": 0,
        "split_unit": "task-episode",
        "val_fraction": 0.2,
    }:
        raise ValueError("corrector slice split contract mismatch")
    if corrector_slice.get("overlap_set_sha256") != (
        "89737e09ab7bf33acdba30720eae8ff0a89622bf1e397f00c514ad68cba32a6e"
    ):
        raise ValueError("corrector slice overlap-set SHA-256 mismatch")
    disclosure = corrector_slice.get("corrector_selection_disclosure")
    if not isinstance(disclosure, str) or "not an independent test" not in disclosure:
        raise ValueError("corrector slice disclosure is incomplete")
    tasks = _object(corrector_slice.get("tasks"), "corrector slice tasks")
    counts = {"clean": 0, "mild": 0, "total": 0}
    for task, raw_conditions in tasks.items():
        conditions = _object(raw_conditions, f"corrector slice task {task}")
        for condition in ("clean", "mild"):
            episodes = conditions.get(condition)
            if not isinstance(episodes, list) or any(
                isinstance(episode, bool) or not isinstance(episode, int)
                for episode in episodes
            ):
                raise TypeError(
                    f"corrector slice {task}/{condition} must be an integer list"
                )
            if len(episodes) != len(set(episodes)):
                raise ValueError(f"duplicate corrector slice episode in {task}/{condition}")
            counts[condition] += len(episodes)
            counts["total"] += len(episodes)
    if counts != EXPECTED_UNSEEN_COUNTS or (
        corrector_slice.get("episode_count") != counts["total"]
    ):
        raise ValueError(
            f"corrector-train-unseen slice counts differ: "
            f"expected {EXPECTED_UNSEEN_COUNTS}, got {counts}"
        )


def _episode_membership_sha256(
    membership: set[tuple[str, int]],
) -> str:
    labels = sorted(f"{task}:{episode}" for task, episode in membership)
    if not labels:
        raise ValueError("episode membership must not be empty")
    return hashlib.sha256(
        (chr(10).join(labels) + chr(10)).encode("utf-8")
    ).hexdigest()


def _episode_labels(value: object, label: str) -> set[tuple[str, int]]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty list")
    result: set[tuple[str, int]] = set()
    for raw_item in value:
        if not isinstance(raw_item, str) or ":" not in raw_item:
            raise TypeError(f"{label} entries must be task:episode strings")
        task_name, raw_episode = raw_item.rsplit(":", 1)
        if not task_name:
            raise ValueError(f"{label} contains an empty task name")
        try:
            episode_id = int(raw_episode)
        except ValueError as error:
            raise ValueError(f"{label} contains an invalid episode id") from error
        if episode_id < 0 or str(episode_id) != raw_episode:
            raise ValueError(f"{label} contains a non-canonical episode id")
        key = (task_name, episode_id)
        if key in result:
            raise ValueError(f"{label} contains a duplicate episode: {raw_item}")
        result.add(key)
    return result


def _replay_npz_split(
    path: Path,
    expected_sha256: str,
    *,
    val_fraction: float,
    split_seed: int,
    split_unit: str,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"frozen corrector source SHA-256 mismatch: {path}"
            )
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as raw:
            split = make_split(
                raw,
                val_fraction,
                split_seed,
                split_unit,
            )
            if "task_name" not in raw.files or "episode_id" not in raw.files:
                raise KeyError("corrector split replay requires task_name/episode_id")
            task_names = np.asarray(raw["task_name"]).astype(str).reshape(-1)
            raw_episode_ids = np.asarray(raw["episode_id"]).reshape(-1)
            if raw_episode_ids.dtype.kind not in {"i", "u"}:
                raise TypeError("corrector episode_id must use an integer dtype")
            episode_ids = raw_episode_ids.astype(np.int64)
        after = os.fstat(stream.fileno())
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"corrector source changed during split replay: {path}")
    if task_names.size != episode_ids.size or task_names.size == 0:
        raise ValueError("corrector task/episode arrays must be non-empty and aligned")
    if any(not task_name for task_name in task_names):
        raise ValueError("corrector task_name values must be non-empty")
    if bool((episode_ids < 0).any()):
        raise ValueError("corrector episode_id values must be non-negative")

    train_indices = np.asarray(split.train, dtype=np.int64)
    validation_indices = np.asarray(split.test, dtype=np.int64)
    all_indices = np.concatenate((train_indices, validation_indices))
    if (
        np.unique(all_indices).size != task_names.size
        or not np.array_equal(np.sort(all_indices), np.arange(task_names.size))
    ):
        raise ValueError("make_split did not partition every source frame exactly once")

    def membership(indices: np.ndarray) -> set[tuple[str, int]]:
        return {
            (str(task_names[index]), int(episode_ids[index]))
            for index in indices.tolist()
        }

    train = membership(train_indices)
    validation = membership(validation_indices)
    if train & validation:
        raise ValueError("make_split divided one episode across train/validation")
    all_membership = {
        (str(task_name), int(episode_id))
        for task_name, episode_id in zip(task_names, episode_ids, strict=True)
    }
    if train | validation != all_membership:
        raise ValueError("make_split episode membership is incomplete")
    return train, validation


def replay_legacy_corrector_split(
    records: Mapping[str, Mapping[str, Any]],
    corrector_slice: Mapping[str, Any],
    episode_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay current make_split against four manifest-frozen NPZ sources."""

    implementation_path = Path(
        sys.modules[make_split.__module__].__file__ or ""
    ).resolve()
    expected_implementation_path = (
        REPO_ROOT / "scripts/train_bc_policy.py"
    ).resolve()
    if implementation_path != expected_implementation_path:
        raise RuntimeError("make_split was imported from an unexpected module")
    implementation_sha256 = _sha256(implementation_path)

    if episode_manifest.get("protocol") != "lewm_gate_l1_camera_aware_episode_split":
        raise ValueError("unexpected LeWM episode manifest protocol")
    manifest_sources = _object(
        episode_manifest.get("sources"), "LeWM episode manifest sources"
    )
    if len(manifest_sources) != 4:
        raise ValueError("legacy replay requires exactly four LeWM sources")

    frozen_sources: dict[str, Mapping[str, Any]] | None = None
    for mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        record = _object(records.get(mode), f"{mode} checkpoint record")
        manifest = _object(record.get("manifest"), f"{mode} input manifest")
        raw_sources = manifest.get("sources")
        if not isinstance(raw_sources, list) or len(raw_sources) != 4:
            raise ValueError(f"{mode} input manifest must freeze four sources")
        current: dict[str, Mapping[str, Any]] = {}
        for raw_source in raw_sources:
            source = _object(raw_source, f"{mode} manifest source")
            name = source.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{mode} manifest source name is invalid")
            if name in current:
                raise ValueError(f"{mode} manifest has duplicate source {name}")
            current[name] = _object(
                source.get("dataset"), f"{mode} source {name} dataset"
            )
        if frozen_sources is None:
            frozen_sources = current
        elif current != frozen_sources:
            raise ValueError("four checkpoint manifests freeze different NPZ sources")
    assert frozen_sources is not None
    if set(frozen_sources) != set(manifest_sources):
        raise ValueError("LeWM and checkpoint manifests name different sources")

    split = _object(corrector_slice.get("corrector_split"), "corrector split")
    val_fraction = split.get("val_fraction")
    split_seed = split.get("split_seed")
    split_unit = split.get("split_unit")
    if (
        isinstance(val_fraction, bool)
        or not isinstance(val_fraction, (int, float))
        or not math.isfinite(float(val_fraction))
        or isinstance(split_seed, bool)
        or not isinstance(split_seed, int)
        or not isinstance(split_unit, str)
    ):
        raise TypeError("corrector split replay parameters are invalid")

    corrector_train: set[tuple[str, int]] = set()
    corrector_validation: set[tuple[str, int]] = set()
    lewm_validation: set[tuple[str, int]] = set()
    dataset_sha256: dict[str, str] = {}
    for source_name in sorted(manifest_sources):
        declared = _object(
            manifest_sources[source_name],
            f"LeWM manifest source {source_name}",
        )
        frozen = frozen_sources[source_name]
        frozen_path = _resolve(
            _object(frozen, f"frozen source {source_name}").get("absolute_path")
        )
        declared_path = _resolve(declared.get("path"))
        if frozen_path != declared_path:
            raise ValueError(f"{source_name} NPZ path differs between manifests")
        frozen_sha256 = _require_sha256(
            frozen.get("sha256"), f"{source_name} frozen dataset SHA-256"
        )
        declared_sha256 = _require_sha256(
            declared.get("data_sha256"), f"{source_name} LeWM data SHA-256"
        )
        if frozen_sha256 != declared_sha256:
            raise ValueError(f"{source_name} NPZ SHA-256 differs between manifests")
        train_membership, validation_membership = _replay_npz_split(
            frozen_path,
            frozen_sha256,
            val_fraction=float(val_fraction),
            split_seed=split_seed,
            split_unit=split_unit,
        )
        if len(train_membership) != 240 or len(validation_membership) != 60:
            raise ValueError(
                f"{source_name} replay must select exactly 240/60 episodes"
            )
        if corrector_train & train_membership or (
            corrector_validation & validation_membership
        ):
            raise ValueError("corrector source episode keys are not globally unique")
        corrector_train.update(train_membership)
        corrector_validation.update(validation_membership)
        source_validation = _episode_labels(
            declared.get("val_episodes"),
            f"LeWM source {source_name} val_episodes",
        )
        if not source_validation <= train_membership | validation_membership:
            raise ValueError(f"{source_name} LeWM validation is absent from NPZ")
        if lewm_validation & source_validation:
            raise ValueError("LeWM validation episode keys are duplicated")
        lewm_validation.update(source_validation)
        dataset_sha256[source_name] = frozen_sha256

    if len(corrector_train) != 960 or len(corrector_validation) != 240:
        raise ValueError("corrector split replay must contain exactly 960/240 episodes")
    if corrector_train & corrector_validation:
        raise ValueError("corrector train/validation memberships overlap")
    if len(lewm_validation) != 168:
        raise ValueError("LeWM validation manifest must contain exactly 168 episodes")
    train_overlap = lewm_validation & corrector_train
    validation_overlap = lewm_validation & corrector_validation
    if (
        len(train_overlap) != 127
        or len(validation_overlap) != 41
        or train_overlap | validation_overlap != lewm_validation
    ):
        raise ValueError(
            "LeWM/corrector replay must partition validation exactly 127/41"
        )

    raw_slice_tasks = _object(corrector_slice.get("tasks"), "corrector slice tasks")
    expected_validation_by_task: dict[str, set[int]] = {}
    for task_name, raw_conditions in raw_slice_tasks.items():
        conditions = _object(raw_conditions, f"corrector slice task {task_name}")
        expected_validation_by_task[str(task_name)] = {
            int(episode)
            for condition in ("clean", "mild")
            for episode in conditions.get(condition, [])
        }
    task_names = sorted({task for task, _ in lewm_validation})
    if set(expected_validation_by_task) != set(task_names):
        raise ValueError("corrector slice task membership is incomplete")
    membership: dict[str, dict[str, list[int]]] = {}
    for task_name in task_names:
        task_validation = {
            episode for task, episode in validation_overlap if task == task_name
        }
        task_train = {
            episode for task, episode in train_overlap if task == task_name
        }
        if task_validation != expected_validation_by_task[task_name]:
            raise ValueError(
                f"{task_name} replayed validation membership differs from slice"
            )
        expected_train = {
            episode for task, episode in lewm_validation if task == task_name
        }.difference(expected_validation_by_task[task_name])
        if task_train != expected_train:
            raise ValueError(
                f"{task_name} replayed train-overlap membership differs"
            )
        membership[task_name] = {
            "corrector_train_overlap": sorted(task_train),
            "corrector_validation_overlap": sorted(task_validation),
        }

    episode_set_sha256 = _episode_membership_sha256(validation_overlap)
    overlap_set_sha256 = _episode_membership_sha256(train_overlap)
    if episode_set_sha256 != _require_sha256(
        corrector_slice.get("episode_set_sha256"),
        "corrector validation episode-set SHA-256",
    ):
        raise ValueError("replayed 41-episode membership SHA-256 mismatch")
    if overlap_set_sha256 != _require_sha256(
        corrector_slice.get("overlap_set_sha256"),
        "corrector train-overlap SHA-256",
    ):
        raise ValueError("replayed 127-episode overlap SHA-256 mismatch")
    if _sha256(implementation_path) != implementation_sha256:
        raise RuntimeError("make_split implementation changed during split replay")
    return {
        "evidence_level": "legacy_split_replay_not_historical_provenance",
        "implementation": "scripts.train_bc_policy.make_split",
        "implementation_path": str(implementation_path),
        "implementation_sha256": implementation_sha256,
        "split": dict(split),
        "dataset_sha256": dataset_sha256,
        "corrector_episode_counts": {"train": 960, "validation": 240},
        "lewm_validation_overlap_counts": {"train": 127, "validation": 41},
        "episode_set_sha256": episode_set_sha256,
        "overlap_set_sha256": overlap_set_sha256,
        "tasks": membership,
    }


def _validate_legacy_corrector_checkpoint_binding(
    records: Mapping[str, Mapping[str, Any]],
    corrector_slice: Mapping[str, Any],
) -> dict[str, str]:
    """Bind the replayed legacy split claim to the actual corrector artifact."""

    expected_sha256 = _require_sha256(
        corrector_slice.get("corrector_checkpoint_sha256"),
        "corrector slice checkpoint SHA-256",
    )
    checkpoint_records: dict[str, str] = {}
    for mode, record in records.items():
        manifest = _object(record.get("manifest"), f"{mode} input manifest")
        sources = manifest.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{mode} input manifest has no sources")
        for source in sources:
            source_record = _object(source, f"{mode} manifest source")
            correction = _object(
                source_record.get("correction_cache"),
                f"{mode} correction cache provenance",
            )
            corrector = _object(
                correction.get("corrector_checkpoint"),
                f"{mode} corrector checkpoint provenance",
            )
            raw_path = corrector.get("absolute_path")
            raw_sha256 = corrector.get("sha256")
            if not isinstance(raw_path, str) or not raw_path:
                raise TypeError("corrector checkpoint path must be non-empty")
            checkpoint_records[str(_resolve(raw_path))] = _require_sha256(
                raw_sha256, "manifest corrector checkpoint SHA-256"
            )
    if len(checkpoint_records) != 1:
        raise ValueError("all modes and sources must share one corrector checkpoint")
    checkpoint_path_text, manifest_sha256 = next(iter(checkpoint_records.items()))
    if manifest_sha256 != expected_sha256:
        raise ValueError("manifest corrector checkpoint differs from the slice")
    checkpoint_path = Path(checkpoint_path_text)
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != expected_sha256:
        raise ValueError("actual corrector checkpoint differs from the slice")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("legacy corrector checkpoint must be an object")
    arguments = _object(checkpoint.get("args"), "legacy corrector arguments")
    split = _object(corrector_slice.get("corrector_split"), "corrector split")
    observed_split = {
        "split_seed": arguments.get("split_seed"),
        "split_unit": arguments.get("split_unit"),
        "val_fraction": arguments.get("val_fraction"),
    }
    if observed_split != dict(split):
        raise ValueError("legacy corrector checkpoint split arguments differ")
    if _sha256(checkpoint_path) != expected_sha256:
        raise RuntimeError("corrector checkpoint changed while being inspected")
    return {
        "path": str(checkpoint_path),
        "sha256": expected_sha256,
        "evidence_level": "legacy_split_replay_not_historical_provenance",
    }


def _validate_hash_snapshot(
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    for label, raw_record in snapshot.items():
        record = _object(raw_record, f"hash snapshot {label}")
        raw_path = record.get("path")
        if not isinstance(raw_path, (str, Path)):
            raise TypeError(f"hash snapshot {label} path is invalid")
        path = Path(raw_path)
        expected = _require_sha256(
            record.get("sha256"), f"hash snapshot {label} SHA-256"
        )
        if not path.is_file():
            raise RuntimeError(f"{label} changed after preflight: missing {path}")
        if _sha256(path) != expected:
            raise RuntimeError(f"{label} changed after preflight: {path}")



def _manifest_artifact_hash_snapshot(
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Collect every file artifact frozen by the four input manifests."""

    artifacts: dict[str, dict[str, str]] = {}

    def visit(value: object, label: str) -> None:
        if isinstance(value, Mapping):
            raw_path = value.get("absolute_path")
            raw_sha256 = value.get("sha256")
            if raw_path is not None and raw_sha256 is not None:
                if not isinstance(raw_path, str) or not raw_path:
                    raise TypeError(f"{label}.absolute_path must be non-empty")
                path = Path(raw_path).expanduser().resolve()
                digest = _require_sha256(raw_sha256, f"{label}.sha256")
                if not path.is_file():
                    raise FileNotFoundError(path)
                key = str(path)
                previous = artifacts.get(key)
                record = {"path": key, "sha256": digest}
                if previous is not None and previous != record:
                    raise ValueError(
                        f"input manifests disagree on artifact hash: {path}"
                    )
                artifacts[key] = record
            for field, nested in value.items():
                visit(nested, f"{label}.{field}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, nested in enumerate(value):
                visit(nested, f"{label}[{index}]")

    for mode, manifest in sorted(manifests.items()):
        visit(manifest, f"{mode} manifest")
    if not artifacts:
        raise ValueError("checkpoint manifests contain no frozen file artifacts")
    return {
        f"manifest artifact {index:03d}": artifact
        for index, artifact in enumerate(
            (artifacts[path] for path in sorted(artifacts)), start=1
        )
    }


def _paired_report_sections(paired: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the full paired analysis into the compact report schema."""

    baseline = _object(paired.get("baseline"), "paired baseline")
    cells = paired.get("cells")
    if not isinstance(cells, list) or len(cells) != 16:
        raise ValueError("paired analysis must contain exactly 16 cells")
    bootstrap_cells = []
    for raw_cell in cells:
        cell = _object(raw_cell, "paired cell")
        bootstrap_cells.append(
            {
                "train_mode": cell["train_mode"],
                "eval_mode": cell["eval_mode"],
                "macro_relative_improvement": cell[
                    "macro_relative_improvement"
                ],
                "macro_absolute_delta": cell["absolute_delta"],
                "bootstrap_95_ci": cell["relative_improvement_95_ci"],
                "probability_improvement": cell[
                    "probability_improvement_gt_zero"
                ],
                "num_tasks": cell["task_count"],
                "num_episodes": cell["episode_count"],
            }
        )

    decomposition = _object(
        paired.get("nominal_corrected_2x2"), "paired 2x2 decomposition"
    )
    raw_cells = _object(decomposition.get("cells"), "paired 2x2 cells")
    nn = float(raw_cells["nominal_train_nominal_eval"])
    if not math.isfinite(nn) or nn <= 0.0:
        raise ValueError("NN MSE must be finite and positive for factor scaling")
    fixed_train = _object(
        decomposition.get("fixed_train_input_effect"), "fixed-train effects"
    )
    fixed_eval = _object(
        decomposition.get("fixed_eval_training_effect"), "fixed-eval effects"
    )
    raw_effects = {
        "eval_input_at_nominal_train": float(fixed_train["triplet_nominal"]),
        "eval_input_at_corrected_train": float(fixed_train["triplet_corrected"]),
        "training_at_nominal_eval": float(fixed_eval["triplet_nominal"]),
        "training_at_corrected_eval": float(fixed_eval["triplet_corrected"]),
        "matched_nominal_to_corrected": float(decomposition["matched_gain"]),
    }
    effects = {
        name: {
            "raw_mse_delta": value,
            "relative_improvement": value / nn,
        }
        for name, value in raw_effects.items()
    }
    effects["interaction"] = {
        "raw_mse_delta": float(decomposition["raw_interaction"])
    }
    return {
        "paired_episode_bootstrap": {
            "baseline": {
                "train_mode": baseline["train_mode"],
                "eval_mode": baseline["eval_mode"],
            },
            "cells": bootstrap_cells,
        },
        "factor_decomposition": {
            "cells": {
                "NN": nn,
                "NC": raw_cells["nominal_train_corrected_eval"],
                "CN": raw_cells["corrected_train_nominal_eval"],
                "CC": raw_cells["corrected_train_corrected_eval"],
            },
            "effects": effects,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON report must be an object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        required=True,
        help="Git-tracked checkpoint ledger created by the independent freeze CLI.",
    )
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--protocol", default=str(DEFAULT_DEV_PROTOCOL))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=5120)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.bootstrap_samples <= 0:
        raise ValueError("invalid evaluator batch/worker/bootstrap settings")
    repository_state = _clean_repository_state()

    protocol_path = _resolve(args.protocol)
    protocol, protocol_sha256 = _load_json_snapshot(
        protocol_path, "development protocol"
    )
    _validate_dev_protocol(protocol)
    protocol_data = _object(protocol.get("data"), "development protocol data")
    bootstrap = _object(protocol.get("bootstrap"), "development bootstrap")
    if (
        args.bootstrap_samples != bootstrap["samples"]
        or args.bootstrap_seed != bootstrap["seed"]
    ):
        raise ValueError(
            "evaluator bootstrap settings must match the frozen development protocol"
        )

    episode_manifest_path = _resolve(str(protocol_data["episode_manifest"]))
    episode_manifest, episode_manifest_sha256 = _load_json_snapshot(
        episode_manifest_path, "validation episode manifest"
    )
    if episode_manifest_sha256 != protocol_data["episode_manifest_sha256"]:
        raise ValueError("validation episode-manifest SHA-256 mismatch")
    corrector_slice_path = _resolve(
        str(protocol_data["corrector_train_unseen_slice"])
    )
    corrector_slice, corrector_slice_sha256 = _load_json_snapshot(
        corrector_slice_path, "corrector-train-unseen slice"
    )
    if (
        corrector_slice_sha256
        != protocol_data["corrector_train_unseen_slice_sha256"]
    ):
        raise ValueError("corrector-train-unseen slice SHA-256 mismatch")
    _validate_corrector_slice(corrector_slice, protocol_data)

    checkpoint_set = validate_checkpoint_set(args.ledger)
    split_replay = replay_legacy_corrector_split(
        checkpoint_set["records"],
        corrector_slice,
        episode_manifest,
    )
    corrector_checkpoint = _validate_legacy_corrector_checkpoint_binding(
        checkpoint_set["records"], corrector_slice
    )
    corrector_checkpoint["split_replay"] = split_replay
    annotations = load_validation_episode_annotations(
        checkpoint_set["data_config"],
        episode_manifest_path=episode_manifest_path,
        corrector_slice_path=corrector_slice_path,
        repo_root=REPO_ROOT,
        expected_counts=protocol_data["full_validation_counts"],
    )
    annotation_counts = _validate_annotation_counts(
        annotations,
        protocol_data["full_validation_counts"],
    )
    result_dir = validate_result_dir_is_ignored(
        args.result_dir,
        create=not args.dry_run,
    )
    result_lock = (
        _acquire_result_lock(result_dir) if not args.dry_run else None
    )
    matrix_runner = Path(__file__).resolve()
    matrix_runner_sha256 = _sha256(matrix_runner)
    evaluator = REPO_ROOT / "scripts/evaluate_lewm_grounded.py"
    evaluator_sha256 = _sha256(evaluator)
    code_manifest = execution_code_manifest()
    execution_code_sha256 = _require_sha256(
        code_manifest.get("aggregate_sha256"), "execution-code SHA-256"
    )
    requested_device = args.device
    if requested_device is None:
        requested_device = (
            "cpu" if args.dry_run else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    device = str(torch.device(requested_device))

    hash_snapshot: dict[str, dict[str, str]] = {
        "checkpoint ledger": {
            "path": checkpoint_set["ledger_path"],
            "sha256": checkpoint_set["ledger_sha256"],
        },
        "development protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        },
        "corrector-train-unseen slice": {
            "path": str(corrector_slice_path),
            "sha256": corrector_slice_sha256,
        },
        "validation episode manifest": {
            "path": str(episode_manifest_path),
            "sha256": episode_manifest_sha256,
        },
        "matrix runner": {
            "path": str(matrix_runner),
            "sha256": matrix_runner_sha256,
        },
        "evaluator": {
            "path": str(evaluator),
            "sha256": evaluator_sha256,
        },
        "legacy corrector checkpoint": {
            "path": corrector_checkpoint["path"],
            "sha256": corrector_checkpoint["sha256"],
        },
        "split replay implementation": {
            "path": split_replay["implementation_path"],
            "sha256": split_replay["implementation_sha256"],
        },
    }
    for mode, record in checkpoint_set["records"].items():
        for field, hash_field, label in (
            ("path", "sha256", "checkpoint"),
            ("metrics_path", "metrics_sha256", "metrics"),
            (
                "input_manifest_path",
                "input_manifest_sha256",
                "input manifest",
            ),
            (
                "external_manifest_path",
                "external_manifest_sha256",
                "external manifest",
            ),
        ):
            hash_snapshot[f"{mode} {label}"] = {
                "path": record[field],
                "sha256": record[hash_field],
            }
        resolved_config_artifact = _object(
            _object(record.get("ledger_record"), f"{mode} ledger record").get(
                "resolved_config"
            ),
            f"{mode} ledger resolved config",
        )
        hash_snapshot[f"{mode} resolved config"] = {
            "path": str(_resolve(resolved_config_artifact.get("path"))),
            "sha256": resolved_config_artifact.get("sha256"),
        }
    manifest_artifact_snapshot = _manifest_artifact_hash_snapshot(
        {
            mode: _object(record.get("manifest"), f"{mode} input manifest")
            for mode, record in checkpoint_set["records"].items()
        }
    )
    _validate_hash_snapshot(hash_snapshot)
    _validate_hash_snapshot(manifest_artifact_snapshot)
    frozen_input_artifacts_sha256 = hashlib.sha256(
        _render_json({"artifacts": manifest_artifact_snapshot})
    ).hexdigest()

    evaluator_parameters = {
        "split": "validation",
        "grounding_mode": "checkpoint",
        "rollout_steps": 5,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "device": device,
        "max_batches": None,
    }
    cell_artifacts: list[tuple[Path, dict[str, Any]]] = []

    for cell in build_anchor_ablation_cells():
        checkpoint = checkpoint_set["records"][cell.train_mode]
        output = result_dir / (
            f"train-{cell.train_mode}__eval-{cell.eval_mode}.json"
        )
        command = build_evaluator_command(
            checkpoint=checkpoint["path"],
            expected_checkpoint_sha256=checkpoint["sha256"],
            eval_mode=cell.eval_mode,
            data_config=checkpoint_set["data_config"],
            output=output,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        contract = _cell_contract(
            train_mode=cell.train_mode,
            eval_mode=cell.eval_mode,
            checkpoint=checkpoint,
            ledger_path=Path(checkpoint_set["ledger_path"]),
            ledger_sha256=checkpoint_set["ledger_sha256"],
            output=output,
            dev_protocol_path=protocol_path,
            dev_protocol_sha256=protocol_sha256,
            corrector_slice_path=corrector_slice_path,
            corrector_slice_sha256=corrector_slice_sha256,
            evaluator_sha256=evaluator_sha256,
            execution_code_sha256=execution_code_sha256,
            matrix_runner_sha256=matrix_runner_sha256,
            repository_state=repository_state,
            frozen_input_artifacts_sha256=frozen_input_artifacts_sha256,
            evaluator_parameters=evaluator_parameters,
            command=command,
        )
        reusable = _load_reusable_cell(
            output,
            contract,
            annotations=annotations,
        )
        if args.dry_run:
            print(shlex.join(command))
            continue
        if reusable is None:
            _write_json_idempotent(_cell_contract_path(output), contract)
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                check=True,
            )
            if not output.is_file():
                raise RuntimeError(f"evaluator did not create cell output: {output}")
            _require_repository_state_unchanged(repository_state)
            _validate_hash_snapshot(hash_snapshot)
            if execution_code_manifest() != code_manifest:
                raise RuntimeError("execution code changed before cell completion")
            reusable = _seal_cell_output(
                output,
                contract,
                annotations=annotations,
            )
        cell_artifacts.append((output, contract))

    if args.dry_run:
        _require_repository_state_unchanged(repository_state)
        _validate_hash_snapshot(hash_snapshot)
        _validate_hash_snapshot(manifest_artifact_snapshot)
        if execution_code_manifest() != code_manifest:
            raise RuntimeError("execution code changed during dry-run preflight")
        return 0

    _require_repository_state_unchanged(repository_state)
    _validate_hash_snapshot(hash_snapshot)
    _validate_hash_snapshot(manifest_artifact_snapshot)
    if execution_code_manifest() != code_manifest:
        raise RuntimeError("execution code changed while evaluating matrix cells")

    normalized_reports = []
    report_paths = []
    for output, contract in cell_artifacts:
        normalized = _load_reusable_cell(
            output,
            contract,
            annotations=annotations,
        )
        if normalized is None:
            raise RuntimeError(f"completed cell artifacts disappeared: {output}")
        normalized_reports.append(normalized)
        contract_path = _cell_contract_path(output)
        completion_path = _cell_completion_path(output)
        sealed_artifacts = _object(
            normalized.get("sealed_artifacts"), "sealed cell artifacts"
        )
        report_record = {
            "path": str(output),
            "sha256": sealed_artifacts["output_sha256"],
            "contract_path": str(contract_path),
            "contract_sha256": sealed_artifacts["contract_sha256"],
            "completion_path": str(completion_path),
            "completion_sha256": sealed_artifacts["completion_sha256"],
        }
        report_paths.append(report_record)
        cell_label = f"{normalized['train_mode']}/{normalized['eval_mode']}"
        for path_field, hash_field, artifact_label in (
            ("path", "sha256", "report"),
            ("contract_path", "contract_sha256", "contract"),
            ("completion_path", "completion_sha256", "completion"),
        ):
            hash_snapshot[f"cell {cell_label} {artifact_label}"] = {
                "path": report_record[path_field],
                "sha256": report_record[hash_field],
            }

    signatures = {
        json.dumps(
            {
                "eligible_windows": report["eligible_windows"],
                "source_windows": report["source_windows"],
            },
            sort_keys=True,
        )
        for report in normalized_reports
    }
    if len(signatures) != 1:
        raise ValueError("matrix cells evaluated different validation windows")
    summary = summarize_anchor_ablation_reports(
        normalized_reports,
        metric="metrics.teacher_forced_pred_mse",
        expected_parameter_count=ANCHOR_ABLATION_PARAMETER_COUNT,
        expected_seed=ANCHOR_ABLATION_SEED,
        expected_target_sha256=ANCHOR_ABLATION_TARGET_SHA256,
        expected_data_split_sha256=ANCHOR_ABLATION_SPLIT_SHA256,
    )
    summary["protocol"] = PROTOCOL
    summary["development_only"] = True
    summary["evaluator_sha256"] = evaluator_sha256
    summary["execution_code_sha256"] = execution_code_sha256
    summary["matrix_runner_sha256"] = matrix_runner_sha256
    summary["repository"] = {
        "head": repository_state["head"],
        "worktree_sha256": repository_state["worktree_sha256"],
    }
    summary["checkpoint_ledger"] = {
        "path": checkpoint_set["ledger_path"],
        "sha256": checkpoint_set["ledger_sha256"],
        "git": checkpoint_set["ledger_git"],
    }
    summary["reports"] = report_paths
    summary["development_protocol"] = {
        "path": str(protocol_path),
        "sha256": protocol_sha256,
    }
    summary["corrector_train_unseen_slice"] = {
        "path": str(corrector_slice_path),
        "sha256": corrector_slice_sha256,
        "checkpoint": corrector_checkpoint,
        "claim_boundary": (
            "The 41-episode slice matches a deterministic replay of the legacy "
            "split contract, but the legacy checkpoint did not persist exact "
            "episode membership or training-code provenance. It is not an "
            "independent test or historical non-exposure proof."
        ),
    }
    summary["annotation_counts"] = annotation_counts
    summary["training_contract"] = checkpoint_set[
        "matched_training_contract"
    ]
    summary["training_provenance"] = {
        "manifest_heads": {
            mode: record["manifest_head"]
            for mode, record in checkpoint_set["records"].items()
        },
        "source_blobs": checkpoint_set["training_blob_signature"],
        "frozen_input_artifact_count": len(manifest_artifact_snapshot),
        "frozen_input_artifacts_sha256": frozen_input_artifacts_sha256,
    }
    paired = summarize_paired_matrix(
        normalized_reports,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["paired_episode_statistics"] = paired
    summary.update(_paired_report_sections(paired))
    summary["development_gate"] = summarize_anchor_visual_dev_gate(
        normalized_reports,
        protocol,
    )
    _validate_hash_snapshot(hash_snapshot)
    if execution_code_manifest() != code_manifest:
        raise RuntimeError("execution code changed before summary write")
    markdown = render_anchor_visual_markdown(summary)
    summary_json_path = result_dir / "summary.json"
    summary_markdown_path = result_dir / "summary.md"
    _write_json_idempotent(summary_json_path, summary)
    _write_text_idempotent(summary_markdown_path, markdown)
    _validate_hash_snapshot(hash_snapshot)
    if execution_code_manifest() != code_manifest:
        raise RuntimeError("execution code changed before final hash recheck")
    summary_completion = {
        "schema_version": 2,
        "protocol": SUMMARY_COMPLETION_PROTOCOL,
        "development_only": True,
        "summary_json_path": str(summary_json_path),
        "summary_json_sha256": _sha256(summary_json_path),
        "summary_markdown_path": str(summary_markdown_path),
        "summary_markdown_sha256": _sha256(summary_markdown_path),
        "development_protocol_sha256": protocol_sha256,
        "checkpoint_ledger_path": checkpoint_set["ledger_path"],
        "checkpoint_ledger_sha256": checkpoint_set["ledger_sha256"],
        "checkpoint_ledger_git": checkpoint_set["ledger_git"],
        "matrix_runner_sha256": matrix_runner_sha256,
        "evaluator_sha256": evaluator_sha256,
        "execution_code_sha256": execution_code_sha256,
        "repository_head": repository_state["head"],
        "repository_worktree_sha256": repository_state["worktree_sha256"],
        "reports_manifest_sha256": hashlib.sha256(
            _render_json({"reports": report_paths})
        ).hexdigest(),
        "frozen_input_artifacts_sha256": summary["training_provenance"][
            "frozen_input_artifacts_sha256"
        ],
    }
    summary_completion_path = result_dir / "summary.completion.json"
    if (
        _sha256(summary_json_path) != summary_completion["summary_json_sha256"]
        or _sha256(summary_markdown_path)
        != summary_completion["summary_markdown_sha256"]
    ):
        raise RuntimeError("summary changed before completion publication")
    _require_repository_state_unchanged(repository_state)
    _validate_hash_snapshot(hash_snapshot)
    _validate_hash_snapshot(manifest_artifact_snapshot)
    if execution_code_manifest() != code_manifest:
        raise RuntimeError("execution code changed before completion publication")
    published_summary, final_summary_sha256 = _load_json_snapshot(
        summary_json_path, "summary JSON"
    )
    if published_summary != summary:
        raise RuntimeError(
            "summary JSON payload changed before completion publication"
        )
    with summary_markdown_path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        published_markdown = stream.read()
        after = os.fstat(stream.fileno())
    markdown_identity = lambda stat_result: (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )
    if markdown_identity(before) != markdown_identity(after):
        raise RuntimeError("summary Markdown changed before completion publication")
    if published_markdown != markdown.encode("ascii"):
        raise RuntimeError(
            "summary Markdown payload changed before completion publication"
        )
    summary_completion["summary_json_sha256"] = final_summary_sha256
    summary_completion["summary_markdown_sha256"] = hashlib.sha256(
        published_markdown
    ).hexdigest()
    _write_json_idempotent(summary_completion_path, summary_completion)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
