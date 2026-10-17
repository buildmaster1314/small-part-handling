#!/usr/bin/env python3
"""Summarize one zero-training control-sufficiency P0 row panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.control_sufficient_p0 import (  # noqa: E402
    DEFAULT_CONTROLLABILITY_BETA,
    DEFAULT_WEIGHT_EPSILON,
    MINIMUM_PHYSICAL_EPISODES,
    ROW_PANEL_PROTOCOL,
    summarize_control_sufficient_p0,
)


EVALUATOR_PROTOCOL = ROW_PANEL_PROTOCOL
EVALUATOR_CLASSIFICATION = "exposed_zero_training_diagnostic"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float64_array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return dict(value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_evaluator_envelope(
    payload: Mapping[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind the evaluator envelope to the exact row panel it contains."""

    if (
        payload.get("schema_version") != 1
        or payload.get("protocol") != EVALUATOR_PROTOCOL
        or payload.get("classification") != EVALUATOR_CLASSIFICATION
        or payload.get("completed") is not True
    ):
        raise PermissionError("input evaluator envelope is incomplete or ineligible")
    if payload.get("row_count") != len(rows):
        raise PermissionError("input evaluator row_count differs from its row panel")
    physical_episodes = {
        str(row.get("physical_episode_id", "")) for row in rows
    }
    if "" in physical_episodes or payload.get("physical_episode_count") != len(
        physical_episodes
    ):
        raise PermissionError(
            "input evaluator physical_episode_count differs from its row panel"
        )
    rows_sha256 = _canonical_sha256(rows)
    if payload.get("rows_sha256") != rows_sha256:
        raise PermissionError("input evaluator rows SHA-256 mismatch")
    if payload.get("causal_action_controls_scored_with_exact_frozen_policy") is not True:
        raise PermissionError("input evaluator did not score causal action controls")

    observer = _mapping(
        payload.get("observer_contract_metadata"),
        label="embedded observer contract metadata",
    )
    observer_sha256 = _canonical_sha256(observer)
    if payload.get("observer_contract_metadata_sha256") != observer_sha256:
        raise PermissionError("embedded observer contract metadata SHA-256 mismatch")
    analytic = _mapping(payload.get("analytic_observer"), label="analytic observer")
    source_sha256 = analytic.get("contract_sha256")
    if (
        not _is_sha256(source_sha256)
        or payload.get("observer_contract_source_sha256") != source_sha256
    ):
        raise PermissionError("analytic observer source-contract binding drifted")
    for field in ("feedback_gain", "residual_decay", "delta_t"):
        if observer.get(field) != analytic.get(field):
            raise PermissionError(
                f"analytic observer {field} differs from embedded frozen contract"
            )

    controllability = _mapping(
        payload.get("controllability"), label="controllability provenance"
    )
    if (
        controllability.get("scored_fresh_panel_used_to_fit_weights") is not False
        or controllability.get("fit_scored_observed_state_overlap_count") != 0
    ):
        raise PermissionError("controllability fit is not distinct from scored rows")
    weights = np.asarray(controllability.get("weights"), dtype=np.float64)
    if weights.ndim != 2 or not bool(
        np.isfinite(weights).all() and np.all(weights >= 0.0)
    ):
        raise ValueError("embedded controllability weights must be finite [anchors,latent]")
    vector_sha256 = _float64_array_sha256(weights)
    if controllability.get("weight_vector_sha256") != vector_sha256:
        raise PermissionError("embedded controllability weight-vector SHA-256 mismatch")

    row_contract = {
        "counterfactual_weight_fit_sha256": controllability.get("weight_fit_sha256"),
        "counterfactual_weight_vector_sha256": vector_sha256,
        "counterfactual_weight_fit_source_id": controllability.get("fit_source"),
        "counterfactual_weight_fit_ids_sha256": controllability.get(
            "fit_ids_sha256"
        ),
        "counterfactual_weight_fit_count": controllability.get("fit_count"),
        "weight_fit_observed_state_signatures_sha256": controllability.get(
            "fit_observed_state_signatures_sha256"
        ),
        "scored_observed_state_signatures_sha256": controllability.get(
            "scored_observed_state_signatures_sha256"
        ),
        "weight_fit_scored_observed_state_overlap_count": 0,
        "observed_state_signature_contract": controllability.get(
            "observed_state_signature_contract"
        ),
        "counterfactual_weight_fit_source_artifact_sha256": controllability.get(
            "fit_source_artifact_sha256"
        ),
        "counterfactual_weight_beta": controllability.get("beta"),
        "counterfactual_weight_epsilon": controllability.get("epsilon"),
        "counterfactual_weight_estimator": controllability.get("estimator"),
    }
    for index, row in enumerate(rows):
        for field, expected in row_contract.items():
            if row.get(field) != expected:
                raise PermissionError(
                    f"rows[{index}].{field} differs from evaluator provenance"
                )
    return {
        "observer_contract_metadata": observer,
        "rows_sha256": rows_sha256,
        "observer_contract_metadata_sha256": observer_sha256,
        "observer_contract_source_sha256": source_sha256,
    }


def _load_input(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    envelope: dict[str, Any] | None = None
    if isinstance(payload, dict):
        envelope = payload
        payload = payload.get("rows")
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise TypeError("input must be a JSON row list or an object containing a row list")
    rows = list(payload)
    if envelope is not None and envelope.get("protocol") == EVALUATOR_PROTOCOL:
        binding = _validate_evaluator_envelope(envelope, rows)
        return rows, binding["observer_contract_metadata"], binding
    return rows, None, None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Backward-compatible row-only loader used by older callers/tests."""

    return _load_input(path)[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain one JSON object")
    return payload


def _load_weight_fit(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        payload = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"branch_predictions"}:
                raise ValueError(
                    "weight-fit NPZ must contain only the branch_predictions array"
                )
            payload = archive["branch_predictions"]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("weight-fit artifact must be .npy, .npz, or .json")
    result = np.asarray(payload, dtype=np.float64)
    if result.ndim != 3:
        raise ValueError(
            "weight-fit artifact must have shape [fit_states,actions,coordinates]"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--weight-fit-artifact",
        type=Path,
        help="separate deployment-only [states,actions,coordinates] fit artifact",
    )
    parser.add_argument(
        "--weight-fit-metadata",
        type=Path,
        help="JSON identity/provenance contract for --weight-fit-artifact",
    )
    parser.add_argument(
        "--observer-contract-metadata",
        type=Path,
        help=(
            "globally frozen analytic-observer contract JSON; optional when --input "
            "is a complete evaluator envelope containing the bound contract"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=61_071)
    parser.add_argument(
        "--minimum-physical-episodes", type=int, default=MINIMUM_PHYSICAL_EPISODES
    )
    parser.add_argument("--weight-epsilon", type=float, default=DEFAULT_WEIGHT_EPSILON)
    parser.add_argument(
        "--controllability-beta", type=float, default=DEFAULT_CONTROLLABILITY_BETA
    )
    args = parser.parse_args(argv)

    rows, embedded_observer, evaluator_binding = _load_input(args.input)

    if (args.weight_fit_artifact is None) != (args.weight_fit_metadata is None):
        parser.error(
            "--weight-fit-artifact and --weight-fit-metadata must be provided together"
        )
    if (
        args.weight_fit_artifact is not None
        and args.weight_fit_artifact.resolve() == args.input.resolve()
    ):
        parser.error("weight-fit artifact must be distinct from the scored row panel")
    weight_fit = None
    weight_metadata = None
    if args.weight_fit_artifact is not None:
        weight_fit = _load_weight_fit(args.weight_fit_artifact)
        weight_metadata = _load_mapping(
            args.weight_fit_metadata, label="weight-fit metadata"
        )
        actual_artifact_sha = _sha256_file(args.weight_fit_artifact)
        expected_artifact_sha = weight_metadata.get("artifact_sha256")
        if expected_artifact_sha is not None and expected_artifact_sha != actual_artifact_sha:
            raise PermissionError("weight-fit artifact SHA-256 differs from metadata")
        weight_metadata["artifact_sha256"] = actual_artifact_sha

    external_observer = None
    if args.observer_contract_metadata is not None:
        external_observer = _load_mapping(
            args.observer_contract_metadata, label="observer contract metadata"
        )
    if embedded_observer is not None and external_observer is not None:
        if embedded_observer != external_observer:
            raise PermissionError(
                "external observer contract differs from evaluator-embedded contract"
            )
        if (
            evaluator_binding is not None
            and _sha256_file(args.observer_contract_metadata)
            != evaluator_binding["observer_contract_source_sha256"]
        ):
            raise PermissionError(
                "external observer contract source SHA-256 differs from evaluator binding"
            )
    observer_metadata = embedded_observer or external_observer
    if observer_metadata is None:
        parser.error(
            "--observer-contract-metadata is required for a bare row panel"
        )

    result = summarize_control_sufficient_p0(
        rows,
        weight_fit_predictions=weight_fit,
        weight_fit_metadata=weight_metadata,
        observer_contract_metadata=observer_metadata,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        minimum_physical_episodes=args.minimum_physical_episodes,
        weight_epsilon=args.weight_epsilon,
        controllability_beta=args.controllability_beta,
    )
    if evaluator_binding is not None:
        result.pop("result_sha256", None)
        result["input_evaluator"] = {
            "path": str(args.input.expanduser().resolve()),
            "sha256": _sha256_file(args.input),
            **evaluator_binding,
        }
        result["result_sha256"] = _canonical_sha256(result)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
