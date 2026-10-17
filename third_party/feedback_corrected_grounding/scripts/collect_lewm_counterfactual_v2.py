#!/usr/bin/env python3
"""Collect the leakage-safe Counterfactual v2 development dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_counterfactual_v2 import (  # noqa: E402
    build_g0_manifest,
    canonical_sha256,
    smoke_config,
    validate_array_bundle,
    validate_full_config,
    validate_index_and_provenance,
    validate_manifest_hash,
    validate_smoke_config,
)
from fcpg.lewm_counterfactual_v2_runtime import (  # noqa: E402
    run_counterfactual_v2_collection,
)
from fcpg.lewm_counterfactual_v2_storage import (  # noqa: E402
    audit_counterfactual_v2_store,
    load_counterfactual_v2_store,
    write_counterfactual_v2_store,
)


DEFAULT_CONFIG = "config/lewm/eval/counterfactual_v2_development_seed8192.json"


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("Counterfactual v2 config root must be a JSON object")
    return value


def _bind_resolved_config(
    provenance: Mapping[str, Any],
    *,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact validated config object, not merely its source path."""

    if not isinstance(provenance, Mapping):
        raise TypeError("collector provenance must be a mapping")
    # JSON round-tripping prevents a caller from mutating the nested config after
    # its digest is bound into the immutable store manifest.
    bound_config = json.loads(
        json.dumps(
            resolved_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    resolved_sha256 = canonical_sha256(bound_config)
    existing_config = provenance.get("config")
    if existing_config is not None and existing_config != bound_config:
        raise ValueError("collector config disagrees with CLI resolved config")
    existing_sha256 = provenance.get("config_sha256")
    if existing_sha256 is not None and existing_sha256 != resolved_sha256:
        raise ValueError("collector config_sha256 disagrees with CLI resolved config")
    result = dict(provenance)
    result.update(
        {
            "config": bound_config,
            "config_sha256": resolved_sha256,
        }
    )
    return result


def _upstream_non_overlap_status(provenance: Mapping[str, Any]) -> str:
    report = provenance.get("upstream_non_overlap")
    if isinstance(report, Mapping):
        status = report.get("status")
        if isinstance(status, str) and status:
            return status
    return "not_audited"


def _canonical_json(value: object) -> bytes:
    """Encode a scientific manifest for exact, order-independent comparison."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _audit_published_scientific_contract(
    destination: Path,
    *,
    config: Mapping[str, Any],
    expected_g0_manifest: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Reload and re-prove the scientific contract from published bytes.

    The storage audit proves immutable files and semantic ndarray hashes.  This
    second layer deliberately rebuilds the scientific schema/provenance/G0
    result from those audited arrays, so a writer or metadata wiring error
    cannot be hidden by a self-consistent storage manifest.
    """

    dataset = load_counterfactual_v2_store(destination)
    publication_provenance = dataset.provenance
    if set(publication_provenance) != {"scientific_provenance", "g0_manifest"}:
        raise RuntimeError(
            "published provenance must contain exactly scientific_provenance "
            "and g0_manifest"
        )
    scientific = publication_provenance["scientific_provenance"]
    embedded_g0 = publication_provenance["g0_manifest"]
    if not isinstance(scientific, Mapping) or not isinstance(embedded_g0, Mapping):
        raise RuntimeError("published scientific provenance or G0 manifest is malformed")

    published_config = scientific.get("config")
    if not isinstance(published_config, Mapping):
        raise RuntimeError("published scientific provenance has no valid config")
    if _canonical_json(published_config) != _canonical_json(config):
        raise RuntimeError("published scientific config disagrees with CLI config")

    candidate_count = len(published_config["evaluation"]["candidate_names"])
    axes = validate_array_bundle(
        dataset.deployment_inputs,
        dataset.offline_labels,
        dataset.audit_arrays,
        candidate_count=candidate_count,
    )
    smoke = published_config.get("evaluation", {}).get("episodes_per_task") == 1
    validate_index_and_provenance(
        published_config,
        dataset.deployment_inputs,
        dataset.offline_labels,
        dataset.audit_arrays,
        dataset.index_rows,
        scientific,
        smoke=smoke,
    )
    validate_manifest_hash(embedded_g0)
    rebuilt_g0 = build_g0_manifest(
        published_config,
        dataset.deployment_inputs,
        dataset.offline_labels,
        dataset.audit_arrays,
        index_rows=dataset.index_rows,
        provenance=scientific,
    )
    validate_manifest_hash(rebuilt_g0)

    expected_bytes = _canonical_json(expected_g0_manifest)
    embedded_bytes = _canonical_json(embedded_g0)
    rebuilt_bytes = _canonical_json(rebuilt_g0)
    if embedded_bytes != expected_bytes:
        raise RuntimeError("published G0 manifest disagrees with pre-publication G0")
    if rebuilt_bytes != embedded_bytes:
        raise RuntimeError("rebuilt published G0 manifest disagrees with embedded G0")
    if canonical_sha256(embedded_g0) != canonical_sha256(rebuilt_g0):
        raise RuntimeError("published G0 canonical semantic hash mismatch")
    return dataset, axes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect immutable Counterfactual v2 features and labels without "
            "training FCPG/LeWM or launching CEM."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="collect the bounded four-task smoke dataset",
    )
    mode.add_argument(
        "--confirm-development",
        action="store_true",
        help="explicitly authorize the complete external-development collection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = _resolve(args.config)
    raw_config = _load_json(config_path)
    validate_full_config(raw_config)
    if args.smoke:
        config = smoke_config(raw_config)
        validate_smoke_config(config)
    else:
        config = raw_config
        validate_full_config(config)

    destination = _resolve(config["output"]["directory"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"exclusive Counterfactual v2 destination already exists: {destination}"
        )

    payload = run_counterfactual_v2_collection(config, repo_root=ROOT)
    candidate_count = len(config["evaluation"]["candidate_names"])
    axes = validate_array_bundle(
        payload.deployment_inputs,
        payload.offline_labels,
        payload.audit_arrays,
        candidate_count=candidate_count,
    )
    if len(payload.index_rows) != axes.sample_count:
        raise ValueError(
            "Counterfactual v2 index row count does not match array sample count"
        )
    scientific_provenance = _bind_resolved_config(
        payload.provenance,
        resolved_config=config,
    )
    g0_manifest = build_g0_manifest(
        config,
        payload.deployment_inputs,
        payload.offline_labels,
        payload.audit_arrays,
        index_rows=payload.index_rows,
        provenance=scientific_provenance,
    )
    if g0_manifest["config_sha256"] != scientific_provenance["config_sha256"]:
        raise RuntimeError("G0 manifest and resolved provenance config hashes disagree")
    publication_provenance = {
        "scientific_provenance": scientific_provenance,
        "g0_manifest": g0_manifest,
    }

    manifest = write_counterfactual_v2_store(
        destination,
        deployment_inputs=payload.deployment_inputs,
        offline_labels=payload.offline_labels,
        audit_arrays=payload.audit_arrays,
        index_rows=payload.index_rows,
        provenance=publication_provenance,
    )
    audit = audit_counterfactual_v2_store(destination)
    if audit.result_sha256 != manifest["result_sha256"]:
        raise RuntimeError("post-publication store hash disagrees with writer manifest")
    _dataset, published_axes = _audit_published_scientific_contract(
        destination,
        config=config,
        expected_g0_manifest=g0_manifest,
    )
    if published_axes != axes:
        raise RuntimeError("published scientific axes disagree with collected arrays")

    physical_branches = axes.sample_count * axes.candidate_count
    g0_audit = g0_manifest["g0_audit"]
    contract_audit = g0_audit["index_and_provenance"]
    g0_eligible = bool(g0_audit["g0_eligible"])
    upstream_status = _upstream_non_overlap_status(scientific_provenance)
    print(f"destination: {destination}")
    print(f"result_sha256: {audit.result_sha256}")
    print(f"samples: {axes.sample_count}")
    print(f"physical_branches: {physical_branches}")
    print(
        f"candidate_families: {axes.candidate_count} "
        "(per-snapshot anonymous slot permutation)"
    )
    print(f"scope: {config['scope']['classification']}")
    print(f"g0_eligible: {str(g0_eligible).lower()}")
    print(f"upstream_non_overlap: {upstream_status}")
    reasons = contract_audit.get("ineligibility_reasons", [])
    if reasons:
        print(f"g0_ineligibility_reasons: {','.join(str(value) for value in reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
