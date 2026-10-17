#!/usr/bin/env python3
"""Run the pre-registered Counterfactual v3 P0/P1 StateOracle gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_counterfactual_runtime import _git_state, sha256_file  # noqa: E402
from fcpg.lewm_counterfactual_v2_scorer import (  # noqa: E402
    _write_json,
    _write_torch,
    publish_scorer_directory_exclusive,
    scorer_state_sha256,
)
from fcpg.lewm_counterfactual_v2_storage import (  # noqa: E402
    load_counterfactual_v2_store,
)
from fcpg.lewm_counterfactual_v3 import (  # noqa: E402
    PROTOCOL,
    canonical_sha256,
    validate_config,
    validate_episode_split_bindings,
)
from fcpg.lewm_counterfactual_v3_gate import (  # noqa: E402
    StopGate,
    PrivilegedDirectRanker,
    prepare_gate_bundle,
    run_state_oracle_gate,
)
from fcpg.lewm_counterfactual_v3_runtime import (  # noqa: E402
    load_frozen_policy_inputs,
    validate_loaded_candidate_artifacts,
)


DEFAULT_CONFIG = "config/lewm/eval/counterfactual_v3_state_oracle_seed9216.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("v3 config root must be an object")
    validate_config(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--confirm-state-oracle-gate",
        action="store_true",
        help="confirm the one frozen external-development P0/P1 run",
    )
    return parser


def _array_mapping(value: object) -> dict[str, list[Any]]:
    return {
        field.name: np.asarray(getattr(value, field.name)).tolist()
        for field in fields(value)
    }


def _dataset_contract(dataset: Any, config: dict[str, Any]) -> str:
    provenance = dataset.provenance
    if set(provenance) != {
        "scientific_provenance",
        "scientific_protocol",
        "scientific_config_sha256",
    }:
        raise ValueError("v3 dataset provenance keys changed")
    if provenance["scientific_protocol"] != PROTOCOL:
        raise ValueError("v3 dataset scientific protocol changed")
    expected_config_sha = canonical_sha256(config)
    if provenance["scientific_config_sha256"] != expected_config_sha:
        raise ValueError("v3 dataset was collected with a different config")
    scientific = provenance["scientific_provenance"]
    if not isinstance(scientific, dict):
        raise TypeError("v3 scientific provenance is malformed")
    if scientific.get("config_sha256") != expected_config_sha:
        raise ValueError("v3 embedded config hash changed")
    if canonical_sha256(scientific.get("config")) != expected_config_sha:
        raise ValueError("v3 embedded config content changed")
    if scientific.get("lewm_loaded") is not False:
        raise ValueError("P0/P1 source unexpectedly loaded LeWM")
    git = scientific.get("git")
    if not isinstance(git, dict) or git.get("dirty") is not False:
        raise ValueError("v3 source was not collected from a clean git worktree")
    if scientific.get("known_v2_non_overlap", {}).get("status") != "passed":
        raise ValueError("v3 source lacks a passed known-v2 non-overlap audit")
    upstream = scientific.get("upstream_checkpoint_episode_non_overlap", {})
    if upstream.get("formal_claim_allowed") is not False:
        raise ValueError("v3 source overclaims upstream episode non-overlap")
    input_records = scientific.get("input_files")
    state_records = scientific.get("loaded_state_sha256")
    if not isinstance(input_records, dict) or not isinstance(state_records, dict):
        raise TypeError("v3 source input provenance is malformed")
    frozen = load_frozen_policy_inputs(config, repo_root=ROOT)
    try:
        if input_records != dict(frozen.file_records):
            raise ValueError(
                "v3 source input path/file/loaded-corrector provenance changed"
            )
        if state_records != dict(frozen.state_sha256):
            raise ValueError("v3 source loaded model-state SHA-256 changed")
    finally:
        del frozen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    replay = scientific.get("candidate_replay_audit")
    if not isinstance(replay, dict) or set(replay) != {
        "scope",
        "tasks",
        "extra_branch_executions",
        "receipts",
    }:
        raise ValueError("v3 source candidate replay audit is missing or malformed")
    receipts = replay.get("receipts")
    if (
        replay.get("scope") != "episode_0_early_snapshot_per_task"
        or int(replay.get("tasks", -1)) != len(config["evaluation"]["tasks"])
        or int(replay.get("extra_branch_executions", -1))
        != len(config["evaluation"]["tasks"]) * 9
        or not isinstance(receipts, list)
        or len(receipts) != len(config["evaluation"]["tasks"])
    ):
        raise ValueError("v3 candidate replay audit summary changed")
    seen_tasks: set[str] = set()
    expected_names = {f"candidate_{index}" for index in range(8)}
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise TypeError("v3 candidate replay receipt is malformed")
        task = str(receipt.get("task"))
        original = receipt.get("original_outcome_sha256")
        reversed_hash = receipt.get("reversed_outcome_sha256")
        repeated = receipt.get("repeated_outcome_sha256")
        repeat_name = receipt.get("repeat_candidate")
        if (
            task not in config["evaluation"]["tasks"]
            or task in seen_tasks
            or int(receipt.get("episode_id", -1)) != 0
            or int(receipt.get("snapshot_step", -1)) != 10
            or receipt.get("validated") is not True
            or repeat_name != "candidate_0"
            or not isinstance(original, dict)
            or set(original) != expected_names
            or not isinstance(reversed_hash, dict)
            or original != reversed_hash
            or repeated != original[repeat_name]
        ):
            raise ValueError("v3 candidate replay receipt failed identity/hash binding")
        seen_tasks.add(task)
    if seen_tasks != set(config["evaluation"]["tasks"]):
        raise ValueError("v3 candidate replay receipts do not cover every task")
    result_sha = str(dataset.manifest.get("result_sha256", ""))
    if len(result_sha) != 64:
        raise ValueError("v3 source result SHA-256 is malformed")
    return result_sha


def _publish_result(
    destination: Path,
    *,
    config: dict[str, Any],
    source_result_sha256: str,
    report: dict[str, Any],
    ensemble: Any | None,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"exclusive v3 gate output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    published = False
    try:
        report_path = staging / "report.json"
        _write_json(report_path, report)
        checkpoint_records = []
        if ensemble is not None:
            scaler = _array_mapping(ensemble.input_scaler)
            normalizer = _array_mapping(ensemble.label_normalizer)
            for seed, state_dict, state_sha in zip(
                ensemble.training_seeds,
                ensemble.state_dicts,
                ensemble.state_sha256,
                strict=True,
            ):
                checkpoint_path = staging / f"privileged_direct_seed{seed}.pt"
                _write_torch(
                    checkpoint_path,
                    {
                        "checkpoint_type": "fcpg_counterfactual_v3_privileged_direct",
                        "scientific_protocol": PROTOCOL,
                        "source_result_sha256": source_result_sha256,
                        "config_sha256": canonical_sha256(config),
                        "seed": int(seed),
                        "model_config": dict(ensemble.model_config),
                        "model_state_dict": state_dict,
                        "model_state_sha256": str(state_sha),
                        "input_scaler": scaler,
                        "label_normalizer": normalizer,
                        "claim_scope": "privileged_external_development_only",
                    },
                )
                checkpoint_records.append(
                    {
                        "path": checkpoint_path.name,
                        "seed": int(seed),
                        "model_state_sha256": str(state_sha),
                        "file_sha256": sha256_file(checkpoint_path),
                        "size_bytes": checkpoint_path.stat().st_size,
                    }
                )
        manifest = {
            "protocol": PROTOCOL,
            "complete": True,
            "source_result_sha256": source_result_sha256,
            "config_sha256": canonical_sha256(config),
            "report_sha256": sha256_file(report_path),
            "report_passed": bool(report.get("passed", False)),
            "direct_training_started": ensemble is not None,
            "execution_git": report["execution_git"],
            "execution_source_files": report["execution_source_files"],
            "checkpoints": checkpoint_records,
        }
        manifest["result_sha256"] = canonical_sha256(manifest)
        _write_json(staging / "manifest.json", manifest)
        publish_scorer_directory_exclusive(staging, destination)
        published = True
        reloaded = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        expected_result = str(reloaded.pop("result_sha256"))
        if canonical_sha256(reloaded) != expected_result:
            raise RuntimeError("published v3 gate manifest hash mismatch")
        if sha256_file(destination / "report.json") != reloaded["report_sha256"]:
            raise RuntimeError("published v3 gate report hash mismatch")
        for record in reloaded["checkpoints"]:
            if sha256_file(destination / record["path"]) != record["file_sha256"]:
                raise RuntimeError("published v3 Direct checkpoint hash mismatch")
            payload = torch.load(
                destination / record["path"], map_location="cpu", weights_only=False
            )
            model = PrivilegedDirectRanker(**payload["model_config"])
            model.load_state_dict(payload["model_state_dict"], strict=True)
            if scorer_state_sha256(model) != payload["model_state_sha256"]:
                raise RuntimeError("published v3 Direct model-state hash mismatch")
        return {**reloaded, "result_sha256": expected_result}
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.confirm_state_oracle_gate:
        raise SystemExit("refusing gate without --confirm-state-oracle-gate")
    config = _load_config(_resolve(args.config))
    git_before = _git_state(ROOT)
    if bool(git_before.get("dirty", True)):
        raise RuntimeError("v3 gate requires a clean git worktree")
    dataset_path = _resolve(config["output"]["dataset_directory"])
    destination = _resolve(config["output"]["gate_directory"])
    dataset = load_counterfactual_v2_store(dataset_path)
    source_sha = _dataset_contract(dataset, config)
    source_git = dataset.provenance["scientific_provenance"]["git"]
    if source_git != git_before:
        raise RuntimeError(
            "v3 gate must run from the exact clean commit used for collection"
        )
    bundle = prepare_gate_bundle(
        dataset.deployment_inputs,
        dataset.offline_labels,
        dataset.audit_arrays,
        dataset.index_rows,
    )
    validate_loaded_candidate_artifacts(
        config,
        dataset.deployment_inputs,
        dataset.audit_arrays,
    )
    validate_episode_split_bindings(
        dataset.audit_arrays,
        dataset.index_rows,
        train_count_per_task=int(config["split"]["train_count_per_task"]),
        validation_count_per_task=int(config["split"]["validation_count_per_task"]),
    )
    try:
        result = run_state_oracle_gate(
            bundle,
            config["gate"],
            device=str(config["device"]),
        )
        report = dict(result.report)
        ensemble = result.direct_ensemble
    except StopGate as stopped:
        report = dict(stopped.report)
        ensemble = stopped.direct_ensemble
    report.update(
        {
            "source_result_sha256": source_sha,
            "config_sha256": canonical_sha256(config),
            "independent_physical_episodes": {
                task: len(
                    {
                        row["physical_episode_sha256"]
                        for row in dataset.index_rows
                        if row["task"] == task and row["split"] == "validation"
                    }
                )
                for task in config["evaluation"]["tasks"]
            },
            "execution_git": git_before,
            "execution_source_files": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in (
                    ROOT / "fcpg/lewm_counterfactual_v3.py",
                    ROOT / "fcpg/lewm_counterfactual_v3_gate.py",
                    ROOT / "fcpg/lewm_counterfactual_v3_state.py",
                    ROOT / "scripts/run_lewm_counterfactual_v3_state_oracle_gate.py",
                    _resolve(args.config),
                )
            },
            "authorization_boundary": (
                "P1 pass only permits a separately pre-registered VisualOracle gate; "
                "it does not establish a LeWM, GoalToken, FCPG-planning, or CEM claim"
            ),
        }
    )
    git_prepublication = _git_state(ROOT)
    if git_prepublication != git_before:
        raise RuntimeError("git state changed during v3 gate execution")
    manifest = _publish_result(
        destination,
        config=config,
        source_result_sha256=source_sha,
        report=report,
        ensemble=ensemble,
    )
    if _git_state(ROOT) != git_before:
        raise RuntimeError("git state changed while publishing the v3 gate")
    print(f"destination: {destination}")
    print(f"result_sha256: {manifest['result_sha256']}")
    print(f"passed: {str(bool(report.get('passed', False))).lower()}")
    print(f"stage: {report.get('stage', 'P1_state_oracle')}")
    print(f"direct_training_started: {str(ensemble is not None).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
