#!/usr/bin/env python3
"""Run the frozen fresh P1 complementary-observer evaluation panel."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_counterfactual_runtime import sha256_file, sha256_state_dict  # noqa: E402
from fcpg.lewm_p1_native_complementary_evaluation import (  # noqa: E402
    FORMAL_CONFIG_PROTOCOL,
    build_seed_rows,
    checkpoint_specs_from_config,
    evaluate_loaded_seed,
    flatten_branch_specific_deployment,
    load_fresh_evaluation_artifacts,
    load_one_frozen_seed,
    reconstruct_fresh_baselines,
    validate_and_load_p1_policy,
)
from fcpg.lewm_p1_native_complementary_statistics import (  # noqa: E402
    HORIZONS,
    REQUIRED_METHODS,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/p1_native_complementary_fresh.json"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _json_line(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_fsync(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _output_dir(config: Mapping[str, object], requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    value = evaluation.get("output_dir")
    if not isinstance(value, str) or not value.strip():
        raise TypeError("config.evaluation.output_dir must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def run_evaluation(
    *,
    config_path: Path,
    device: str,
    output_dir: Path | None,
) -> Path:
    requested_config = config_path.expanduser().resolve()
    if sha256_file(requested_config) != sha256_file(DEFAULT_CONFIG.resolve()):
        raise ValueError(
            "fresh complementary evaluation requires the checked-in frozen config bytes"
        )
    loaded = load_fresh_evaluation_artifacts(config_path, repo_root=ROOT)
    evaluation = _mapping(loaded.config["evaluation"], "config.evaluation")
    batch_size = int(evaluation["batch_size"])
    destination = _output_dir(loaded.config, output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {destination}")

    policy = validate_and_load_p1_policy(
        loaded.config, repo_root=ROOT, device=device
    )
    policy_state_before = sha256_state_dict(policy)
    baselines = reconstruct_fresh_baselines(
        loaded.cache,
        policy,
        device=device,
        batch_size=batch_size,
    )
    if sha256_state_dict(policy) != policy_state_before:
        raise RuntimeError("frozen P1 policy changed during baseline reconstruction")
    deployment = flatten_branch_specific_deployment(
        loaded.cache, baselines.geometry_hold_exact_fk_token
    )
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    episodes = loaded.cache.physical_episode_id[:, 0].astype(str).tolist()
    specs = checkpoint_specs_from_config(loaded.config, repo_root=ROOT)

    formal_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    checkpoint_records: dict[str, dict[str, object]] = {}
    reference_action_mean: torch.Tensor | None = None
    reference_action_std: torch.Tensor | None = None
    for spec in specs:
        pair = load_one_frozen_seed(
            spec,
            training_config=loaded.training_config,
            device=device,
        )
        current_mean = pair.action_mean.detach().cpu()
        current_std = pair.action_std.detach().cpu()
        if reference_action_mean is None:
            reference_action_mean = current_mean.clone()
            reference_action_std = current_std.clone()
        elif not torch.equal(current_mean, reference_action_mean) or not torch.equal(
            current_std, reference_action_std
        ):
            raise ValueError("frozen predictor action normalization differs across seeds")
        factual_before = sha256_state_dict(pair.factual)
        no_action_before = sha256_state_dict(pair.no_action)
        predictions = evaluate_loaded_seed(
            pair,
            deployment,
            device=device,
            batch_size=batch_size,
        )
        if (
            sha256_state_dict(pair.factual) != factual_before
            or sha256_state_dict(pair.no_action) != no_action_before
        ):
            raise RuntimeError(f"frozen predictor seed {spec.seed} changed during evaluation")
        seed_rows, seed_diagnostics = build_seed_rows(
            seed=spec.seed,
            checkpoint_sha256=pair.sha256,
            predictions=predictions,
            baselines=baselines,
            target_token=loaded.target_token,
            physical_episode_id=episodes,
            provenance=loaded.provenance,
        )
        formal_rows.extend(seed_rows)
        diagnostic_rows.extend(seed_diagnostics)
        checkpoint_records[str(spec.seed)] = {
            "path": str(spec.path),
            "sha256": pair.sha256,
            "factual_state_sha256": factual_before,
            "no_action_state_sha256": no_action_before,
        }
        del predictions, pair
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected_formal = len(specs) * len(REQUIRED_METHODS) * len(episodes) * len(HORIZONS)
    expected_diagnostic = len(specs) * len(episodes) * len(HORIZONS)
    if len(formal_rows) != expected_formal or len(diagnostic_rows) != expected_diagnostic:
        raise RuntimeError("fresh complementary output panel is incomplete")
    provenance = loaded.provenance
    report = {
        "schema_version": 1,
        "protocol": FORMAL_CONFIG_PROTOCOL,
        "classification": "formal",
        "completed": True,
        "no_training": True,
        "known_visual_dropout_only": True,
        "offline_labels_used_only_after_prediction": True,
        "branch_specific_history_flattened_as_n_times_s_with_runtime_s_equal_one": True,
        "physical_episode_count": len(episodes),
        "branch_count": deployment.branch_count,
        "horizons": list(HORIZONS),
        "formal_row_count": len(formal_rows),
        "raw_diagnostic_row_count": len(diagnostic_rows),
        "device": str(device),
        "batch_size": batch_size,
        "config_path": str(provenance.config_path),
        "config_sha256": provenance.config_sha256,
        "training_config_path": str(provenance.training_config_path),
        "training_config_sha256": provenance.training_config_sha256,
        "deployment_cache": str(provenance.deployment_path),
        "deployment_cache_sha256": provenance.deployment_cache_sha256,
        "label_cache": str(provenance.label_path),
        "label_cache_sha256": provenance.label_cache_sha256,
        "collection_metadata": str(provenance.metadata_path),
        "metadata_sha256": provenance.metadata_sha256,
        "dataset_sha256": provenance.dataset_sha256,
        "policy_state_sha256": policy_state_before,
        "predictors": checkpoint_records,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        rows_path = temporary / "formal_rows.jsonl"
        diagnostic_path = temporary / "raw_lewm_diagnostics.jsonl"
        metadata_path = temporary / "evaluation.metadata.json"
        _write_fsync(rows_path, "".join(_json_line(row) + "\n" for row in formal_rows))
        _write_fsync(
            diagnostic_path,
            "".join(_json_line(row) + "\n" for row in diagnostic_rows),
        )
        report["formal_rows_sha256"] = sha256_file(rows_path)
        report["raw_diagnostics_sha256"] = sha256_file(diagnostic_path)
        _write_fsync(metadata_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = run_evaluation(
        config_path=args.config,
        device=str(args.device),
        output_dir=args.output_dir,
    )
    print(f"evaluation: {output}")
    print(f"formal rows: {output / 'formal_rows.jsonl'}")
    print(f"raw diagnostic: {output / 'raw_lewm_diagnostics.jsonl'}")
    print(f"metadata: {output / 'evaluation.metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
