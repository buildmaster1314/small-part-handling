#!/usr/bin/env python3
"""Collect the fresh immutable Counterfactual v3 StateOracle dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_counterfactual_v2_storage import (  # noqa: E402
    audit_counterfactual_v2_store,
    load_counterfactual_v2_store,
    write_counterfactual_v2_store,
)
from fcpg.lewm_counterfactual_v3 import (  # noqa: E402
    canonical_sha256,
    validate_action_hash_bindings,
    validate_array_bundle,
    validate_config,
    validate_episode_split_bindings,
    validate_index_rows,
)
from fcpg.lewm_counterfactual_v3_runtime import (  # noqa: E402
    run_v3_collection,
    validate_loaded_candidate_artifacts,
)
from fcpg.lewm_counterfactual_runtime import _git_state  # noqa: E402


DEFAULT_CONFIG = "config/lewm/eval/counterfactual_v3_state_oracle_seed9216.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, object]:
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
        "--confirm-state-oracle-development",
        action="store_true",
        help="confirm the one pre-registered external-development collection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.confirm_state_oracle_development:
        raise SystemExit("refusing collection without --confirm-state-oracle-development")
    config = _load_config(_resolve(args.config))
    output = config["output"]
    assert isinstance(output, dict)
    destination = _resolve(str(output["dataset_directory"]))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"exclusive v3 dataset already exists: {destination}")

    payload = run_v3_collection(config, repo_root=ROOT)
    if _git_state(ROOT) != payload.provenance["git"]:
        raise RuntimeError("git state changed between v3 collection and publication")
    provenance = {
        "scientific_provenance": payload.provenance,
        "scientific_protocol": str(config["protocol"]),
        "scientific_config_sha256": canonical_sha256(config),
    }
    manifest = write_counterfactual_v2_store(
        destination,
        deployment_inputs=payload.deployment_inputs,
        offline_labels=payload.offline_labels,
        audit_arrays=payload.audit_arrays,
        index_rows=payload.index_rows,
        provenance=provenance,
    )
    storage_audit = audit_counterfactual_v2_store(destination)
    if storage_audit.result_sha256 != manifest["result_sha256"]:
        raise RuntimeError("v3 writer and storage audit hashes disagree")
    loaded = load_counterfactual_v2_store(destination)
    axes = validate_array_bundle(
        loaded.deployment_inputs,
        loaded.offline_labels,
        loaded.audit_arrays,
    )
    validate_index_rows(loaded.index_rows, axes=axes)
    validate_action_hash_bindings(loaded.deployment_inputs, loaded.index_rows)
    validate_loaded_candidate_artifacts(
        config,
        loaded.deployment_inputs,
        loaded.audit_arrays,
    )
    validate_episode_split_bindings(
        loaded.audit_arrays,
        loaded.index_rows,
        train_count_per_task=int(config["split"]["train_count_per_task"]),
        validation_count_per_task=int(config["split"]["validation_count_per_task"]),
    )
    if loaded.provenance != provenance:
        raise RuntimeError("published v3 provenance differs from in-memory provenance")
    if _git_state(ROOT) != payload.provenance["git"]:
        raise RuntimeError("git state changed while publishing the v3 dataset")

    print(f"destination: {destination}")
    print(f"result_sha256: {storage_audit.result_sha256}")
    print(f"samples: {axes.sample_count}")
    print(f"physical_branches: {axes.sample_count * axes.candidate_count}")
    print(
        "audit_replay_branches: "
        f"{payload.provenance['candidate_replay_audit']['extra_branch_executions']}"
    )
    print(f"state_dim: {axes.state_dim}")
    print("lewm_loaded: false")
    print("scope: external_development")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
