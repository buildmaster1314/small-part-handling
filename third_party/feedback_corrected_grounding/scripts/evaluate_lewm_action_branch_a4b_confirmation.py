#!/usr/bin/env python3
"""Evaluate frozen A4 v2 10k endpoints on the fresh A4b final cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4 import BUNDLE_BATCH_SIZE  # noqa: E402
from fcpg.lewm_action_branch_a4_confirmation_metrics import (  # noqa: E402
    CONFIRMATION_CHECKPOINT_STEP,
    validate_a4_development_go_authorization,
)
from fcpg.lewm_action_branch_a4_training import (  # noqa: E402
    A4_AMP_DTYPE,
    atomic_save_a4_predictions,
    attach_a4_cache_metadata,
    build_a4_evaluation_loader,
    evaluate_a4_models,
)
from fcpg.lewm_action_branch_a4b import (  # noqa: E402
    BENCHMARK_SEED,
    CONFIRMATION_TRAINING_SEEDS,
    PROTOCOL,
    SOURCE_A4_PROTOCOL,
    SPLIT_MANIFEST_ID,
    TASK_IDENTITY_NAMESPACE,
    validate_a4b_contract,
)
from fcpg.lewm_action_branch_a4b_confirmation_metrics import (  # noqa: E402
    validate_a4b_final_cache,
    validate_a4b_fresh_results,
    validate_a4b_result_cache_path,
)
from scripts.evaluate_lewm_action_branch_a4_confirmation import (  # noqa: E402
    DEFAULT_BASE_CHECKPOINT,
    _attach_checkpoint_identity,
    _formal_train_cache_path,
    _load_json,
    _load_test_only_cache,
    prepare_confirmation_checkpoints,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_a4b_fresh_confirmation.json"
DEFAULT_SOURCE_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_SPLIT_MANIFEST = ROOT / (
    "config/lewm/splits/action_branch_a4b_fresh_confirmation_seed31401.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4b/confirmation_predictions"


def attach_a4b_prediction_identity(
    arrays: Mapping[str, np.ndarray], *, final_result_path: str | Path
) -> dict[str, np.ndarray]:
    """Replace the source protocol label with the actual A4b evaluation identity."""

    result = dict(arrays)
    result.update(
        {
            "protocol": np.asarray(PROTOCOL),
            "source_endpoint_protocol": np.asarray(SOURCE_A4_PROTOCOL),
            "split_manifest_id": np.asarray(SPLIT_MANIFEST_ID),
            "task_identity_namespace": np.asarray(TASK_IDENTITY_NAMESPACE),
            "benchmark_seed": np.asarray(BENCHMARK_SEED, dtype=np.int64),
            "evaluation_split": np.asarray("test"),
            "primary_horizons": np.asarray([6, 7, 8], dtype=np.int64),
            "final_collection_result": np.asarray(
                str(Path(final_result_path).expanduser().resolve())
            ),
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--preflight-result", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--development-go-summary", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    contract = _load_json(args.config, label="A4b config")
    source_contract = _load_json(args.source_config, label="A4 v2 source config")
    split_manifest = _load_json(args.split_manifest, label="A4b split manifest")
    preflight_result = _load_json(
        args.preflight_result, label="A4b preflight result"
    )
    final_result = _load_json(args.final_result, label="A4b final result")
    development_go = _load_json(
        args.development_go_summary, label="A4 Development-Go summary"
    )
    validate_a4b_contract(
        contract,
        source_contract=source_contract,
        split_manifest=split_manifest,
    )
    validate_a4b_fresh_results(preflight_result, final_result)
    validate_a4b_result_cache_path(final_result, args.test_cache)

    # Validate Development-Go and all endpoint states before reading final data.
    validate_a4_development_go_authorization(
        development_go, contract=source_contract
    )
    formal_train_cache = _formal_train_cache_path(development_go)
    prepared = prepare_confirmation_checkpoints(
        args.checkpoints,
        base_checkpoint_path=args.base_checkpoint,
        formal_train_cache_path=formal_train_cache,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A4b confirmation CUDA device is unavailable")
    test_cache = _load_test_only_cache(args.test_cache)
    validate_a4b_final_cache(test_cache, final_result)
    loader = build_a4_evaluation_loader(
        test_cache,
        split="test",
        batch_size=BUNDLE_BATCH_SIZE,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    for endpoint in prepared:
        endpoint.factual.to(device)
        endpoint.no_action.to(device)
        evaluation = evaluate_a4_models(
            endpoint.factual,
            endpoint.no_action,
            loader,
            device=device,
            step=endpoint.step,
            training_seed=endpoint.seed,
            amp_enabled=device.type == "cuda",
            amp_dtype=A4_AMP_DTYPE,
        )
        arrays = attach_a4_cache_metadata(evaluation.arrays, test_cache)
        arrays = _attach_checkpoint_identity(arrays, endpoint)
        arrays = attach_a4b_prediction_identity(
            arrays, final_result_path=args.final_result
        )
        destination = output_dir / f"seed{endpoint.seed}_test_predictions.npz"
        atomic_save_a4_predictions(arrays, destination)
        outputs.append(
            {
                "seed": endpoint.seed,
                "checkpoint": str(endpoint.path),
                "predictions": str(destination),
                "metrics": evaluation.metrics,
            }
        )
        endpoint.factual.cpu()
        endpoint.no_action.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "protocol": PROTOCOL,
        "source_endpoint_protocol": SOURCE_A4_PROTOCOL,
        "stage": "a4b_fresh_10k_test_prediction",
        "split_manifest_id": SPLIT_MANIFEST_ID,
        "task_identity_namespace": TASK_IDENTITY_NAMESPACE,
        "benchmark_seed": BENCHMARK_SEED,
        "selection": "fixed_latest",
        "checkpoint_step": CONFIRMATION_CHECKPOINT_STEP,
        "training_seeds": list(CONFIRMATION_TRAINING_SEEDS),
        "preflight_result": str(args.preflight_result.expanduser().resolve()),
        "final_result": str(args.final_result.expanduser().resolve()),
        "test_cache": str(args.test_cache.expanduser().resolve()),
        "outputs": outputs,
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "seeds": list(CONFIRMATION_TRAINING_SEEDS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
