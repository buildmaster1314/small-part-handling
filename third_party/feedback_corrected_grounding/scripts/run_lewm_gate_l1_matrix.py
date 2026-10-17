#!/usr/bin/env python3
"""Run and summarize the preregistered grounded-LeWM Gate L1 matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, TextIO

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_gate_l1 import (  # noqa: E402
    EVAL_GROUNDINGS,
    TRAIN_GROUNDINGS,
    summarize_gate_l1,
)
from fcpg.lewm_provenance import sha256_file, validate_manifest  # noqa: E402


CELL_CONTRACT_PROTOCOL = "fcpg_grounded_lewm_gate_l1_cell_contract"
EXPECTED_TASKS = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)
EXPECTED_EPISODES_PER_TASK = 30
EXPECTED_PAIRING_SAMPLE_COUNT = 12_000
_CONDITION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _resolve(path: object) -> Path:
    value = Path(str(path)).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value, _, _ = _load_json_snapshot(path)
    return value


def _load_json_snapshot(path: Path) -> tuple[dict[str, Any], bytes, str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value, payload, hashlib.sha256(payload).hexdigest()


def _load_frozen_manifest(path: Path) -> tuple[dict[str, Any], bytes, str]:
    return _load_json_snapshot(path)


def _assert_manifest_unchanged(path: Path, initial_bytes: bytes) -> None:
    if path.read_bytes() != initial_bytes:
        raise RuntimeError(
            f"Gate L1 manifest changed while the runner was active: {path}"
        )


def _render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_render_json(value))


def _write_json_idempotent(path: Path, value: Mapping[str, Any]) -> bool:
    rendered = _render_json(value).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(rendered)
        return True
    except FileExistsError:
        if path.read_bytes() != rendered:
            raise FileExistsError(
                f"refusing to overwrite different Gate L1 summary: {path}"
            ) from None
        return False


def _validate_condition_name(name: object) -> str:
    if not isinstance(name, str) or _CONDITION_NAME.fullmatch(name) is None:
        raise ValueError(f"unsafe Gate L1 condition name: {name!r}")
    return name


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _metadata_path(value: object, metadata_path: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is missing")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else metadata_path.parent / path).resolve()


def validate_matrix_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != "fcpg_grounded_lewm_gate_l1_matrix":
        raise ValueError("unexpected Gate L1 matrix manifest protocol")
    conditions = _object(manifest.get("conditions"), "manifest conditions")
    evaluation = _object(manifest.get("evaluation"), "manifest evaluation")
    summary = _object(manifest.get("summary"), "manifest summary")
    primary = summary.get("primary_conditions")
    if not isinstance(primary, Sequence) or isinstance(primary, (str, bytes)):
        raise TypeError("summary primary_conditions must be a sequence")
    expected_conditions = [
        _validate_condition_name(summary.get("clean_condition")),
        *[_validate_condition_name(name) for name in primary],
        _validate_condition_name(summary.get("stress_condition")),
    ]
    if len(expected_conditions) != 4 or len(set(expected_conditions)) != 4:
        raise ValueError("Gate L1 requires four distinct matrix conditions")
    if any(
        _validate_condition_name(name) not in expected_conditions for name in conditions
    ):
        raise ValueError("manifest contains a non-preregistered condition")
    if set(conditions) != set(expected_conditions):
        raise ValueError("manifest conditions differ from the preregistered summary")
    if tuple(evaluation.get("expected_tasks", ())) != EXPECTED_TASKS:
        raise ValueError(f"Gate L1 expected_tasks must be exactly {EXPECTED_TASKS!r}")
    if evaluation.get("expected_episodes_per_task") != EXPECTED_EPISODES_PER_TASK:
        raise ValueError("Gate L1 requires exactly 30 episodes per task")
    if "max_batches" not in evaluation or evaluation["max_batches"] is not None:
        raise ValueError("Gate L1 evaluation requires max_batches=None")
    if int(evaluation.get("batch_size", 0)) <= 0:
        raise ValueError("Gate L1 batch_size must be positive")
    if int(evaluation.get("num_workers", -1)) < 0:
        raise ValueError("Gate L1 num_workers must be non-negative")
    if int(evaluation.get("cell_bootstrap_samples", 0)) <= 0:
        raise ValueError("Gate L1 cell bootstrap samples must be positive")
    for name, raw in conditions.items():
        condition = _object(raw, f"condition {name!r}")
        if int(condition.get("rollout_steps", 0)) not in {5, 10}:
            raise ValueError(f"condition {name} has invalid rollout_steps")
        if int(condition.get("expected_windows", 0)) <= 0:
            raise ValueError(f"condition {name} has no expected full window count")


def _matched_checkpoint_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["output_dir"] = "<paired-output>"
    value["provenance_manifest"] = "<paired-manifest>"
    value["data"]["grounding_mode"] = "<paired-grounding>"
    value["contract"]["training_grounding"] = "<paired-grounding>"
    return value


def _validate_checkpoint_provenance(
    checkpoint: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, str]:
    provenance = _object(checkpoint.get("provenance"), "checkpoint provenance")
    expected_sha256 = _require_sha256(
        provenance.get("sha256"), "checkpoint provenance SHA-256"
    )
    embedded_manifest = _object(
        provenance.get("manifest"), "checkpoint provenance manifest"
    )
    external_path = _resolve(provenance.get("path"))
    external_manifest, _, external_sha256 = _load_json_snapshot(external_path)
    if external_sha256 != expected_sha256:
        raise ValueError("external frozen input manifest SHA-256 differs from checkpoint")
    if external_manifest != dict(embedded_manifest):
        raise ValueError("external frozen input manifest differs from checkpoint")

    run_sidecar = checkpoint_path.parent / "input_manifest.json"
    sidecar_manifest, _, sidecar_sha256 = _load_json_snapshot(run_sidecar)
    if sidecar_sha256 != expected_sha256:
        raise ValueError("run input manifest sidecar SHA-256 differs from checkpoint")
    if sidecar_manifest != dict(embedded_manifest):
        raise ValueError("run input manifest sidecar differs from checkpoint")

    validate_manifest(embedded_manifest, check_git=True)
    if sha256_file(external_path) != expected_sha256:
        raise RuntimeError("external frozen input manifest changed during validation")
    if sha256_file(run_sidecar) != expected_sha256:
        raise RuntimeError("run input manifest sidecar changed during validation")
    return {
        "manifest_path": str(external_path),
        "manifest_sha256": expected_sha256,
        "run_sidecar_path": str(run_sidecar),
        "run_sidecar_sha256": expected_sha256,
    }


def validate_checkpoint_pair(paths: Mapping[str, Path]) -> dict[str, Any]:
    checkpoints: dict[str, Mapping[str, Any]] = {}
    checkpoint_hashes = {}
    provenance_records = {}
    target_encoder_hashes = {}
    for name in TRAIN_GROUNDINGS:
        path = paths[name].expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        initial_sha256 = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"{name} checkpoint payload must be an object")
        if sha256_file(path) != initial_sha256:
            raise RuntimeError(f"{name} checkpoint changed while it was being loaded")
        checkpoints[name] = checkpoint
        checkpoint_hashes[name] = initial_sha256
        provenance_records[name] = _validate_checkpoint_provenance(checkpoint, path)

    for name, checkpoint in checkpoints.items():
        if checkpoint.get("run_kind") != "formal":
            raise ValueError(f"{name} checkpoint is not a formal run")
        if int(checkpoint.get("step", -1)) != 10_000:
            raise ValueError(f"{name} checkpoint is not the fixed 10,000-step endpoint")
        config = _object(checkpoint.get("resolved_config"), f"{name} resolved_config")
        if float(config["loss"]["sigreg_weight"]) != 0.02:
            raise ValueError(f"{name} checkpoint does not use SIGReg=0.02")
        if config["data"]["target_mode"] != "frozen_oracle":
            raise ValueError(f"{name} checkpoint does not use frozen oracle targets")
        if _resolve(config.get("provenance_manifest")) != Path(
            provenance_records[name]["manifest_path"]
        ):
            raise ValueError(f"{name} checkpoint provenance path differs from config")
        config_contract = _object(config.get("contract"), f"{name} config contract")
        target_encoder_sha256 = _require_sha256(
            config_contract.get("target_encoder_sha256"),
            f"{name} target encoder SHA-256",
        )
        data_config = _object(checkpoint.get("data_config"), f"{name} data_config")
        target_contract = _object(
            data_config.get("target_contract"), f"{name} target contract"
        )
        if (
            target_contract.get("encoder_checkpoint_sha256")
            != target_encoder_sha256
        ):
            raise ValueError(
                f"{name} checkpoint target encoder differs from its training contract"
            )
        target_encoder_hashes[name] = target_encoder_sha256

    nominal_config = checkpoints["nominal"]["resolved_config"]
    fcpg_config = checkpoints["fcpg"]["resolved_config"]
    if _matched_checkpoint_contract(nominal_config) != _matched_checkpoint_contract(
        fcpg_config
    ):
        raise ValueError("nominal/FCPG checkpoints violate the matched config contract")
    if nominal_config["data"]["grounding_mode"] != "nominal":
        raise ValueError("nominal checkpoint has the wrong training grounding")
    if fcpg_config["data"]["grounding_mode"] != "fcpg":
        raise ValueError("FCPG checkpoint has the wrong training grounding")
    if checkpoints["nominal"]["data_config"]["normalization"] != checkpoints[
        "fcpg"
    ]["data_config"]["normalization"]:
        raise ValueError("nominal/FCPG checkpoints use different normalization")
    if len(set(target_encoder_hashes.values())) != 1:
        raise ValueError("nominal/FCPG checkpoints use different target encoders")
    return {
        name: {
            "path": str(paths[name].expanduser().resolve()),
            "sha256": checkpoint_hashes[name],
            "step": int(checkpoints[name]["step"]),
            "epoch": int(checkpoints[name]["epoch"]),
            "target_encoder_sha256": target_encoder_hashes[name],
            "provenance": provenance_records[name],
        }
        for name in TRAIN_GROUNDINGS
    }


def validate_condition_inputs(conditions: Mapping[str, Any]) -> dict[str, Any]:
    for name in conditions:
        _validate_condition_name(name)
    clean_raw = conditions.get("clean")
    if not isinstance(clean_raw, Mapping):
        raise ValueError("Gate L1 conditions must define one clean reference")
    clean_data = _resolve(clean_raw["data"])
    if not clean_data.is_file():
        raise FileNotFoundError(clean_data)
    clean_sha256 = sha256_file(clean_data)

    physical_hashes = set()
    records = {}
    for name, raw in conditions.items():
        condition = _object(raw, f"condition {name!r}")
        data = _resolve(condition["data"])
        correction = _resolve(condition["correction_cache"])
        target = _resolve(condition["oracle_target_cache"])
        correction_array = correction / "q_corrected.npy"
        correction_metadata_path = correction / "metadata.json"
        target_array = target / "oracle_target.npy"
        target_metadata_path = target / "metadata.json"
        for path in (
            data,
            correction_array,
            correction_metadata_path,
            target_array,
            target_metadata_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        data_sha256 = sha256_file(data)
        correction_sha256 = sha256_file(correction_array)
        target_sha256 = sha256_file(target_array)
        correction_metadata, _, correction_metadata_sha256 = _load_json_snapshot(
            correction_metadata_path
        )
        target_metadata, _, target_metadata_sha256 = _load_json_snapshot(
            target_metadata_path
        )
        if (
            _metadata_path(
                correction_metadata.get("source_data"),
                correction_metadata_path,
                f"condition {name} correction source",
            )
            != data
        ):
            raise ValueError(f"condition {name} correction source mismatch")
        if correction_metadata.get("source_data_sha256") != data_sha256:
            raise ValueError(f"condition {name} correction source hash mismatch")
        if correction_metadata.get("q_corrected_sha256") != correction_sha256:
            raise ValueError(f"condition {name} correction array hash mismatch")
        correction_checkpoint = _metadata_path(
            correction_metadata.get("checkpoint"),
            correction_metadata_path,
            f"condition {name} correction checkpoint",
        )
        if not correction_checkpoint.is_file():
            raise FileNotFoundError(correction_checkpoint)
        correction_checkpoint_sha256 = _require_sha256(
            correction_metadata.get("checkpoint_sha256"),
            f"condition {name} correction checkpoint SHA-256",
        )
        if sha256_file(correction_checkpoint) != correction_checkpoint_sha256:
            raise ValueError(f"condition {name} correction checkpoint hash mismatch")

        if (
            _metadata_path(
                target_metadata.get("source_data"),
                target_metadata_path,
                f"condition {name} oracle target source",
            )
            != data
        ):
            raise ValueError(f"condition {name} oracle target source mismatch")
        if target_metadata.get("source_data_sha256") != data_sha256:
            raise ValueError(f"condition {name} oracle target source hash mismatch")
        if target_metadata.get("oracle_target_sha256") != target_sha256:
            raise ValueError(f"condition {name} oracle target array hash mismatch")
        target_encoder = _metadata_path(
            target_metadata.get("encoder_checkpoint"),
            target_metadata_path,
            f"condition {name} target encoder",
        )
        if not target_encoder.is_file():
            raise FileNotFoundError(target_encoder)
        target_encoder_sha256 = _require_sha256(
            target_metadata.get("encoder_checkpoint_sha256"),
            f"condition {name} target encoder SHA-256",
        )
        if sha256_file(target_encoder) != target_encoder_sha256:
            raise ValueError(f"condition {name} target encoder hash mismatch")

        pairing_value = condition.get("pairing_report")
        pairing = None
        pairing_path = None
        pairing_sha256 = None
        if name == "clean":
            if pairing_value not in (None, ""):
                raise ValueError("clean condition must not define a pairing report")
        else:
            if not pairing_value:
                raise ValueError(f"condition {name} requires a pairing report")
            pairing_path = _resolve(pairing_value)
            pairing, _, pairing_sha256 = _load_json_snapshot(pairing_path)
            if pairing.get("validated") is not True:
                raise ValueError(f"condition {name} has no valid pairing report")
            if pairing.get("sample_count") != EXPECTED_PAIRING_SAMPLE_COUNT:
                raise ValueError(
                    f"condition {name} pairing report sample_count must be 12000"
                )
            if _resolve(pairing["intervention_path"]) != data:
                raise ValueError(
                    f"condition {name} pairing report points to other data"
                )
            if pairing.get("intervention_sha256") != data_sha256:
                raise ValueError(f"condition {name} data differs from pairing report")
            if _resolve(pairing["reference_path"]) != clean_data:
                raise ValueError(
                    f"condition {name} pairing report uses a different clean reference"
                )
            if pairing.get("reference_sha256") != clean_sha256:
                raise ValueError(
                    f"condition {name} clean reference differs from pairing report"
                )
            physical_hashes.add(
                _require_sha256(
                    pairing.get("physical_identity_sha256"),
                    f"condition {name} physical identity SHA-256",
                )
            )

        records[name] = {
            "data": str(data),
            "data_sha256": data_sha256,
            "correction_cache": str(correction),
            "correction_sha256": correction_sha256,
            "correction_metadata_path": str(correction_metadata_path),
            "correction_metadata_sha256": correction_metadata_sha256,
            "correction_checkpoint_path": str(correction_checkpoint),
            "correction_checkpoint_sha256": correction_checkpoint_sha256,
            "oracle_target_cache": str(target),
            "oracle_target_sha256": target_sha256,
            "oracle_target_metadata_path": str(target_metadata_path),
            "oracle_target_metadata_sha256": target_metadata_sha256,
            "target_encoder_checkpoint_path": str(target_encoder),
            "target_encoder_sha256": target_encoder_sha256,
            "pairing_report_path": str(pairing_path) if pairing_path else None,
            "pairing_report_sha256": pairing_sha256,
            "pairing": pairing,
        }
    if len(physical_hashes) != 1:
        raise ValueError("camera interventions do not share one physical identity hash")
    if len({record["target_encoder_sha256"] for record in records.values()}) != 1:
        raise ValueError("camera conditions do not share one oracle target encoder")
    return records


def validate_result_dir_is_ignored(result_dir: Path) -> None:
    try:
        relative = result_dir.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(
            "Gate L1 result_dir must be inside the Git worktree"
        ) from error
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if relative == Path(".") or ignored.returncode != 0:
        raise ValueError(f"Gate L1 result_dir is not Git-ignored: {result_dir}")
    tracked = subprocess.run(
        ["git", "ls-files", "--", relative.as_posix()],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise RuntimeError("could not verify Gate L1 result_dir tracked files")
    if tracked.stdout.strip():
        raise ValueError(f"Gate L1 result_dir contains tracked files: {result_dir}")


def _cell_output(result_dir: Path, condition: str, train: str, evaluation: str) -> Path:
    _validate_condition_name(condition)
    return result_dir / f"{condition}__train-{train}__eval-{evaluation}.json"


def _cell_contract_path(output: Path) -> Path:
    return output.with_suffix(".contract.json")


def _build_cell_contract(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    checkpoint: Mapping[str, Any],
    condition: Mapping[str, Any],
    condition_record: Mapping[str, Any],
    condition_name: str,
    train: str,
    evaluation: str,
    eval_config: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    _validate_condition_name(condition_name)
    if train not in TRAIN_GROUNDINGS or evaluation not in EVAL_GROUNDINGS:
        raise ValueError("unknown Gate L1 grounding")
    if checkpoint["target_encoder_sha256"] != condition_record["target_encoder_sha256"]:
        raise ValueError(
            f"condition {condition_name} target encoder differs from checkpoint"
        )
    provenance = _object(checkpoint.get("provenance"), "checkpoint provenance record")
    return {
        "schema_version": 1,
        "protocol": CELL_CONTRACT_PROTOCOL,
        "manifest_path": str(manifest_path),
        "manifest_initial_sha256": manifest_sha256,
        "condition": condition_name,
        "train_grounding": train,
        "evaluation_grounding": evaluation,
        "device": str(torch.device(device)),
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_provenance_manifest_path": provenance["manifest_path"],
        "checkpoint_provenance_manifest_sha256": provenance["manifest_sha256"],
        "checkpoint_provenance_run_sidecar_path": provenance["run_sidecar_path"],
        "checkpoint_provenance_run_sidecar_sha256": provenance[
            "run_sidecar_sha256"
        ],
        "data_path": condition_record["data"],
        "data_sha256": condition_record["data_sha256"],
        "correction_cache_path": condition_record["correction_cache"],
        "correction_cache_sha256": condition_record["correction_sha256"],
        "correction_metadata_path": condition_record["correction_metadata_path"],
        "correction_metadata_sha256": condition_record[
            "correction_metadata_sha256"
        ],
        "correction_checkpoint_path": condition_record[
            "correction_checkpoint_path"
        ],
        "correction_checkpoint_sha256": condition_record[
            "correction_checkpoint_sha256"
        ],
        "oracle_target_cache_path": condition_record["oracle_target_cache"],
        "oracle_target_cache_sha256": condition_record["oracle_target_sha256"],
        "oracle_target_metadata_path": condition_record[
            "oracle_target_metadata_path"
        ],
        "oracle_target_metadata_sha256": condition_record[
            "oracle_target_metadata_sha256"
        ],
        "target_encoder_checkpoint_path": condition_record[
            "target_encoder_checkpoint_path"
        ],
        "target_encoder_sha256": condition_record["target_encoder_sha256"],
        "pairing_report_path": condition_record["pairing_report_path"],
        "pairing_report_sha256": condition_record["pairing_report_sha256"],
        "split": eval_config["split"],
        "rollout_steps": int(condition["rollout_steps"]),
        "batch_size": int(eval_config["batch_size"]),
        "num_workers": int(eval_config["num_workers"]),
        "max_batches": eval_config["max_batches"],
        "bootstrap_samples": int(eval_config["cell_bootstrap_samples"]),
        "bootstrap_seed": int(eval_config["bootstrap_seed"]),
        "expected_windows": int(condition["expected_windows"]),
        "expected_tasks": list(eval_config["expected_tasks"]),
        "expected_episodes_per_task": int(eval_config["expected_episodes_per_task"]),
    }


def _validate_contract_inputs(contract: Mapping[str, Any]) -> None:
    files = [
        (contract["manifest_path"], contract["manifest_initial_sha256"], "manifest"),
        (contract["checkpoint_path"], contract["checkpoint_sha256"], "checkpoint"),
        (
            contract["checkpoint_provenance_manifest_path"],
            contract["checkpoint_provenance_manifest_sha256"],
            "checkpoint provenance manifest",
        ),
        (
            contract["checkpoint_provenance_run_sidecar_path"],
            contract["checkpoint_provenance_run_sidecar_sha256"],
            "checkpoint provenance run sidecar",
        ),
        (contract["data_path"], contract["data_sha256"], "data"),
        (
            Path(str(contract["correction_cache_path"])) / "q_corrected.npy",
            contract["correction_cache_sha256"],
            "correction cache",
        ),
        (
            contract["correction_metadata_path"],
            contract["correction_metadata_sha256"],
            "correction metadata",
        ),
        (
            contract["correction_checkpoint_path"],
            contract["correction_checkpoint_sha256"],
            "correction checkpoint",
        ),
        (
            Path(str(contract["oracle_target_cache_path"])) / "oracle_target.npy",
            contract["oracle_target_cache_sha256"],
            "oracle target cache",
        ),
        (
            contract["oracle_target_metadata_path"],
            contract["oracle_target_metadata_sha256"],
            "oracle target metadata",
        ),
        (
            contract["target_encoder_checkpoint_path"],
            contract["target_encoder_sha256"],
            "target encoder checkpoint",
        ),
    ]
    pairing_path = contract.get("pairing_report_path")
    pairing_sha256 = contract.get("pairing_report_sha256")
    if (pairing_path is None) != (pairing_sha256 is None):
        raise ValueError("pairing report path/hash contract is incomplete")
    if pairing_path is not None:
        files.append((pairing_path, pairing_sha256, "pairing report"))

    for path_value, expected_sha256, label in files:
        path = Path(str(path_value))
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{label} SHA-256 changed: {path}")


def _validate_cell_report(
    report: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    data = report.get("data")
    correction = report.get("correction_cache")
    target = report.get("oracle_target_cache")
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError("Gate L1 report requires exactly one data source")
    if not isinstance(correction, list) or not isinstance(target, list):
        raise TypeError("Gate L1 report cache manifests must be lists")
    if len(target) != 1 or not isinstance(target[0], Mapping):
        raise ValueError("Gate L1 report requires exactly one oracle target cache")

    grounding = contract["evaluation_grounding"]
    correction_path = None
    correction_sha256 = None
    correction_metadata_path = None
    correction_metadata_sha256 = None
    if grounding == "nominal":
        if correction:
            raise ValueError("nominal evaluation must not use a correction cache")
    else:
        if len(correction) != 1 or not isinstance(correction[0], Mapping):
            raise ValueError("FCPG evaluation requires exactly one correction cache")
        correction_path = correction[0].get("input")
        correction_sha256 = correction[0].get("array_sha256")
        correction_metadata_path = correction[0].get("metadata")
        correction_metadata_sha256 = correction[0].get("metadata_sha256")

    provenance = _object(report.get("provenance"), "report provenance")
    evaluation = _object(report.get("evaluation"), "report evaluation")
    sequence = _object(report.get("sequence"), "report sequence")
    bootstrap = _object(report.get("bootstrap"), "report bootstrap")
    metrics = _object(report.get("metrics"), "report metrics")
    actual = {
        "schema_version": report.get("schema_version"),
        "protocol": report.get("protocol"),
        "checkpoint_path": report.get("checkpoint"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "provenance_validated": provenance.get("validated"),
        "provenance_manifest_path": provenance.get("manifest"),
        "provenance_manifest_sha256": provenance.get("manifest_sha256"),
        "provenance_run_sidecar_path": provenance.get("run_sidecar"),
        "grounding": report.get("grounding_mode"),
        "grounding_request": report.get("grounding_mode_request"),
        "data_path": data[0],
        "correction_path": correction_path,
        "correction_sha256": correction_sha256,
        "correction_metadata_path": correction_metadata_path,
        "correction_metadata_sha256": correction_metadata_sha256,
        "target_path": target[0].get("input"),
        "target_sha256": target[0].get("array_sha256"),
        "target_metadata_path": target[0].get("metadata"),
        "target_metadata_sha256": target[0].get("metadata_sha256"),
        "target_encoder_sha256": target[0].get("encoder_checkpoint_sha256"),
        "split": evaluation.get("split"),
        "batch_size": evaluation.get("batch_size"),
        "num_workers": evaluation.get("num_workers"),
        "max_batches": evaluation.get("max_batches", "<missing>"),
        "device": evaluation.get("device"),
        "source_windows": evaluation.get("source_windows"),
        "eligible_windows": evaluation.get("eligible_windows"),
        "rollout_steps": sequence.get("rollout_steps"),
        "bootstrap_samples": bootstrap.get("samples"),
        "bootstrap_seed": bootstrap.get("seed"),
        "num_sequences": metrics.get("num_sequences"),
        "num_eligible_sequences": metrics.get("num_eligible_sequences"),
    }
    expected = {
        "schema_version": 1,
        "protocol": "grounded_lewm_offline_diagnostics",
        "checkpoint_path": contract["checkpoint_path"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "provenance_validated": True,
        "provenance_manifest_path": contract[
            "checkpoint_provenance_manifest_path"
        ],
        "provenance_manifest_sha256": contract[
            "checkpoint_provenance_manifest_sha256"
        ],
        "provenance_run_sidecar_path": contract[
            "checkpoint_provenance_run_sidecar_path"
        ],
        "grounding": grounding,
        "grounding_request": grounding,
        "data_path": contract["data_path"],
        "correction_path": (
            contract["correction_cache_path"] if grounding == "fcpg" else None
        ),
        "correction_sha256": (
            contract["correction_cache_sha256"] if grounding == "fcpg" else None
        ),
        "correction_metadata_path": (
            contract["correction_metadata_path"] if grounding == "fcpg" else None
        ),
        "correction_metadata_sha256": (
            contract["correction_metadata_sha256"] if grounding == "fcpg" else None
        ),
        "target_path": contract["oracle_target_cache_path"],
        "target_sha256": contract["oracle_target_cache_sha256"],
        "target_metadata_path": contract["oracle_target_metadata_path"],
        "target_metadata_sha256": contract["oracle_target_metadata_sha256"],
        "target_encoder_sha256": contract["target_encoder_sha256"],
        "split": contract["split"],
        "batch_size": contract["batch_size"],
        "num_workers": contract["num_workers"],
        "max_batches": None,
        "device": contract["device"],
        "source_windows": [contract["expected_windows"]],
        "eligible_windows": contract["expected_windows"],
        "rollout_steps": contract["rollout_steps"],
        "bootstrap_samples": contract["bootstrap_samples"],
        "bootstrap_seed": contract["bootstrap_seed"],
        "num_sequences": contract["expected_windows"],
        "num_eligible_sequences": contract["expected_windows"],
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise ValueError(
                f"Gate L1 report {key} mismatch: "
                f"expected {expected_value!r}, got {actual[key]!r}"
            )

    blocks = metrics.get("episode_blocks")
    if not isinstance(blocks, list):
        raise TypeError("Gate L1 report episode_blocks must be a list")
    expected_tasks = tuple(contract["expected_tasks"])
    episodes = {task: set() for task in expected_tasks}
    total_windows = 0
    for block in blocks:
        values = _object(block, "episode block")
        task = values.get("task_name")
        episode_id = values.get("episode_id")
        if (
            task not in episodes
            or not isinstance(episode_id, int)
            or isinstance(episode_id, bool)
        ):
            raise ValueError("Gate L1 report contains an unexpected task/episode")
        if episode_id in episodes[task]:
            raise ValueError(f"duplicate Gate L1 episode block: {(task, episode_id)!r}")
        episodes[task].add(episode_id)
        num_sequences = values.get("num_sequences")
        if (
            not isinstance(num_sequences, int)
            or isinstance(num_sequences, bool)
            or num_sequences <= 0
        ):
            raise ValueError("Gate L1 episode window count must be positive")
        total_windows += num_sequences

    expected_per_task = int(contract["expected_episodes_per_task"])
    expected_episode_ids = set(range(expected_per_task))
    for task, episode_ids in episodes.items():
        if episode_ids != expected_episode_ids:
            raise ValueError(
                f"Gate L1 task {task!r} episode IDs must be exactly "
                f"0..{expected_per_task - 1}"
            )
    if len(blocks) != len(expected_tasks) * expected_per_task:
        raise ValueError("Gate L1 report does not contain exactly 120 episode blocks")
    if total_windows != contract["expected_windows"]:
        raise ValueError("Gate L1 report does not contain the full window count")


def _cell_sidecar(
    contract: Mapping[str, Any], report_sha256: str
) -> dict[str, Any]:
    return {**dict(contract), "report_sha256": report_sha256}


def _validated_cell_artifacts(
    output: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str]:
    if not output.is_file():
        raise FileNotFoundError(output)
    contract_path = _cell_contract_path(output)
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"missing Gate L1 cell contract sidecar: {contract_path}"
        )

    sidecar, _, sidecar_sha256 = _load_json_snapshot(contract_path)
    expected_report_sha256 = _require_sha256(
        sidecar.pop("report_sha256", None), "cell report SHA-256"
    )
    if sidecar != dict(contract):
        raise ValueError(
            f"Gate L1 cell contract differs from this run: {contract_path}"
        )

    report, _, report_sha256 = _load_json_snapshot(output)
    if report_sha256 != expected_report_sha256:
        raise ValueError(f"Gate L1 cell report SHA-256 differs from sidecar: {output}")
    _validate_contract_inputs(contract)
    _validate_cell_report(report, contract)
    return report, report_sha256, sidecar_sha256


def _validate_cell_artifacts(
    output: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    report, _, _ = _validated_cell_artifacts(output, contract)
    return report


def _seal_cell_report(
    output: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    report, _, report_sha256 = _load_json_snapshot(output)
    _validate_cell_report(report, contract)
    _validate_contract_inputs(contract)
    _write_json_exclusive(
        _cell_contract_path(output),
        _cell_sidecar(contract, report_sha256),
    )
    return _validate_cell_artifacts(output, contract)


def _build_cell_command(
    contract: Mapping[str, Any], *, output: Path, device: str | None = None
) -> list[str]:
    if contract["max_batches"] is not None:
        raise ValueError("Gate L1 cells require max_batches=None")
    bound_device = str(torch.device(str(contract["device"])))
    if device is not None and str(torch.device(device)) != bound_device:
        raise ValueError(
            f"Gate L1 device differs from cell contract: "
            f"expected {bound_device!r}, got {device!r}"
        )
    grounding = str(contract["evaluation_grounding"])
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/evaluate_lewm_grounded.py"),
        "--checkpoint",
        str(contract["checkpoint_path"]),
        "--data",
        str(contract["data_path"]),
    ]
    if grounding == "fcpg":
        command.extend(
            ["--correction-cache", str(contract["correction_cache_path"])]
        )
    command.extend(
        [
            "--oracle-target-cache",
            str(contract["oracle_target_cache_path"]),
            "--split",
            str(contract["split"]),
            "--grounding-mode",
            grounding,
            "--batch-size",
            str(contract["batch_size"]),
            "--num-workers",
            str(contract["num_workers"]),
            "--rollout-steps",
            str(contract["rollout_steps"]),
            "--bootstrap-samples",
            str(contract["bootstrap_samples"]),
            "--bootstrap-seed",
            str(contract["bootstrap_seed"]),
            "--device",
            bound_device,
            "--output",
            str(output),
        ]
    )
    return command


def _open_attempt_log(output: Path) -> tuple[Path, TextIO]:
    attempt = 1
    while True:
        path = output.with_name(f"{output.stem}.attempt-{attempt:03d}.log")
        try:
            return path, path.open("x", encoding="utf-8")
        except FileExistsError:
            attempt += 1


def _run_cell(
    *,
    contract: Mapping[str, Any],
    condition: str,
    train: str,
    evaluation: str,
    result_dir: Path,
    device: str,
) -> dict[str, Any]:
    output = _cell_output(result_dir, condition, train, evaluation)
    contract_path = _cell_contract_path(output)
    if contract_path.exists():
        return _validate_cell_artifacts(output, contract)
    if output.exists():
        return _seal_cell_report(output, contract)

    _validate_contract_inputs(contract)
    _, log = _open_attempt_log(output)
    with log:
        subprocess.run(
            _build_cell_command(contract, output=output, device=device),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not output.is_file():
        raise RuntimeError(f"evaluator did not create output: {output}")
    return _seal_cell_report(output, contract)


def _gate_exit_code(passed: bool) -> int:
    return 0 if passed else 1


def _run_gate_l1(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="config/lewm/eval/gate_l1_matrix_seed3072.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    device = str(torch.device(args.device))

    manifest_path = _resolve(args.manifest)
    manifest, initial_bytes, initial_sha256 = _load_frozen_manifest(manifest_path)
    validate_matrix_manifest(manifest)
    checkpoint_paths = {
        name: _resolve(manifest["checkpoints"][name]) for name in TRAIN_GROUNDINGS
    }
    checkpoint_records = validate_checkpoint_pair(checkpoint_paths)
    condition_records = validate_condition_inputs(manifest["conditions"])
    _assert_manifest_unchanged(manifest_path, initial_bytes)
    eval_config = _object(manifest["evaluation"], "manifest evaluation")
    result_dir = _resolve(eval_config["result_dir"])
    validate_result_dir_is_ignored(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    matrix = {}
    cell_contracts = {}
    for condition_name, raw_condition in manifest["conditions"].items():
        condition = _object(raw_condition, f"condition {condition_name!r}")
        matrix[condition_name] = {}
        for train in TRAIN_GROUNDINGS:
            matrix[condition_name][train] = {}
            for evaluation in EVAL_GROUNDINGS:
                _assert_manifest_unchanged(manifest_path, initial_bytes)
                contract = _build_cell_contract(
                    manifest_path=manifest_path,
                    manifest_sha256=initial_sha256,
                    checkpoint=checkpoint_records[train],
                    condition=condition,
                    condition_record=condition_records[condition_name],
                    condition_name=condition_name,
                    train=train,
                    evaluation=evaluation,
                    eval_config=eval_config,
                    device=device,
                )
                output = _cell_output(result_dir, condition_name, train, evaluation)
                report = (
                    _validate_cell_artifacts(output, contract)
                    if args.summarize_only
                    else _run_cell(
                        contract=contract,
                        condition=condition_name,
                        train=train,
                        evaluation=evaluation,
                        result_dir=result_dir,
                        device=device,
                    )
                )
                matrix[condition_name][train][evaluation] = report
                cell_contracts[(condition_name, train, evaluation)] = (
                    output,
                    contract,
                )

    _assert_manifest_unchanged(manifest_path, initial_bytes)
    for name, record in checkpoint_records.items():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"{name} checkpoint changed during the Gate L1 matrix")

    cell_records = {}
    for (condition_name, train, evaluation), (
        output,
        contract,
    ) in cell_contracts.items():
        report, report_sha256, sidecar_sha256 = _validated_cell_artifacts(
            output, contract
        )
        matrix[condition_name][train][evaluation] = report
        sidecar = _cell_contract_path(output)
        cell_records[f"{condition_name}/{train}/{evaluation}"] = {
            "path": str(output),
            "sha256": report_sha256,
            "contract_path": str(sidecar),
            "contract_sha256": sidecar_sha256,
        }

    summary = _object(manifest["summary"], "manifest summary")
    decision = summarize_gate_l1(
        matrix,
        primary_conditions=summary["primary_conditions"],
        clean_condition=summary["clean_condition"],
        stress_condition=summary["stress_condition"],
        bootstrap_samples=int(summary["bootstrap_samples"]),
        bootstrap_seed=int(summary["bootstrap_seed"]),
        thresholds=summary["thresholds"],
    )
    decision["manifest"] = {
        "path": str(manifest_path),
        "sha256": initial_sha256,
        "initial_sha256": initial_sha256,
    }
    decision["checkpoints"] = checkpoint_records
    decision["conditions"] = condition_records
    decision["cells"] = cell_records
    output = result_dir / "gate_l1_summary.json"
    _assert_manifest_unchanged(manifest_path, initial_bytes)
    _write_json_idempotent(output, decision)
    _assert_manifest_unchanged(manifest_path, initial_bytes)
    print(_render_json(decision), end="")
    return _gate_exit_code(bool(decision["passed"]))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run_gate_l1(argv)
    except Exception as error:
        print(
            f"Gate L1 runtime failure: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
