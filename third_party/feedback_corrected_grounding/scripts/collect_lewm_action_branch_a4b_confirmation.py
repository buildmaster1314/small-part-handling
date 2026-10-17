#!/usr/bin/env python3
"""Run the independent A4b preflight or frozen final candidate pool."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
from typing import Any


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4b_task_identity import (  # noqa: E402
    collect_legacy_a4_reset_digests,
    legacy_digest_set,
    make_a4b_episode_runtime,
)
from fcpg.lewm_action_branch_a4_confirmation_collection import (  # noqa: E402
    validate_a4_development_go_summary,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    load_frozen_a4_state_encoder,
    make_a4_episode_runtime,
)
from fcpg.lewm_action_branch_a4b import (  # noqa: E402
    TASKS,
    validate_a4b_contract,
)
from fcpg.lewm_action_branch_a4b_confirmation_collection import (  # noqa: E402
    build_a4b_collection_result,
    collect_a4b_final_pool,
    collect_a4b_preflight_pool,
    publish_a4b_final_bundle_exclusive,
    save_a4b_result_exclusive,
)
from scripts.evaluate_lewm_action_branch_a4_confirmation import (  # noqa: E402
    prepare_confirmation_checkpoints,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_a4b_fresh_confirmation.json"
DEFAULT_SOURCE_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_SPLIT_MANIFEST = ROOT / (
    "config/lewm/splits/action_branch_a4b_fresh_confirmation_seed31401.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/lewm_action_branch_a4b"
DEFAULT_FINAL_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "final"
DEFAULT_BASE_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _common_contracts(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    contract = _load_json(args.config, label="A4b config")
    source_contract = _load_json(args.source_config, label="A4 v2 source config")
    split_manifest = _load_json(args.split_manifest, label="A4b split manifest")
    validate_a4b_contract(
        contract,
        source_contract=source_contract,
        split_manifest=split_manifest,
    )
    return contract, source_contract, split_manifest


def _require_absent(*paths: Path) -> tuple[Path, ...]:
    resolved = tuple(path.expanduser().resolve() for path in paths)
    for path in resolved:
        if path.exists():
            raise FileExistsError(path)
    return resolved


def _require_new_final_bundle(*paths: Path) -> tuple[Path, ...]:
    resolved = tuple(path.expanduser().resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("A4b final result/cache/ledger paths must be distinct")
    publication_directory = resolved[0].parent
    if any(path.parent != publication_directory for path in resolved):
        raise ValueError(
            "A4b final result/cache/ledger must share one publication directory"
        )
    if publication_directory.exists():
        raise FileExistsError(publication_directory)
    return resolved


def _preflight(args: argparse.Namespace) -> int:
    contract, source_contract, split_manifest = _common_contracts(args)
    (result_path,) = _require_absent(args.result)
    source_collection = source_contract["collection"]
    source_runtime = partial(
        make_a4_episode_runtime,
        width=int(source_collection["width"]),
        height=int(source_collection["height"]),
        benchmark_seed=int(source_collection["benchmark_seed"]),
    )
    legacy_inventory = collect_legacy_a4_reset_digests(
        source_contract,
        runtime_factory=source_runtime,
    )
    collection = contract["collection"]
    runtime = partial(
        make_a4b_episode_runtime,
        width=int(collection["width"]),
        height=int(collection["height"]),
    )
    preflight = collect_a4b_preflight_pool(
        contract,
        source_contract,
        split_manifest,
        legacy_physical_identities=legacy_digest_set(legacy_inventory),
        runtime_factory=runtime,
    )
    result = build_a4b_collection_result(
        preflight,
        legacy_reset_identities=legacy_inventory,
    )
    save_a4b_result_exclusive(result_path, result)
    eligible = {
        task: sum(
            row["eligible"] for row in result["attempts_by_task"][task]
        )
        for task in TASKS
    }
    print(json.dumps({"result": str(result_path), "eligible_by_task": eligible}))
    return 0


def _final(args: argparse.Namespace) -> int:
    contract, source_contract, split_manifest = _common_contracts(args)
    result_path, cache_path, ledger_path = _require_new_final_bundle(
        args.result, args.output_cache, args.output_ledger
    )

    development_go = _load_json(
        args.development_go_summary, label="A4 Development-Go summary"
    )
    authorization = validate_a4_development_go_summary(
        development_go,
        source_contract,
        formal_train_cache_path=args.formal_train_cache,
    )
    prepared = prepare_confirmation_checkpoints(
        args.checkpoints,
        base_checkpoint_path=args.base_checkpoint,
        formal_train_cache_path=str(authorization.formal_train_cache_path),
    )
    endpoint_seeds = [endpoint.seed for endpoint in prepared]
    del prepared

    encoder = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=args.device,
    )
    collection_config = contract["collection"]
    runtime = partial(
        make_a4b_episode_runtime,
        width=int(collection_config["width"]),
        height=int(collection_config["height"]),
    )
    final = collect_a4b_final_pool(
        contract,
        source_contract,
        split_manifest,
        preflight_receipt_path=args.preflight_result,
        development_go_summary_path=args.development_go_summary,
        formal_train_cache_path=args.formal_train_cache,
        encoder=encoder,
        runtime_factory=runtime,
    )
    publish_a4b_final_bundle_exclusive(
        result_path=result_path,
        cache_path=cache_path,
        ledger_path=ledger_path,
        collection=final,
        validated_endpoint_seeds=endpoint_seeds,
    )
    if not final.passed:
        print(json.dumps({"result": str(result_path), "status": "failure"}))
        return 2

    print(
        json.dumps(
            {
                "result": str(result_path),
                "status": "success",
                "state_cache": str(cache_path),
                "proposal_ledger": str(ledger_path),
            }
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", allow_abbrev=False)
    preflight.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "preflight_result.json",
    )
    preflight.set_defaults(handler=_preflight)

    final = subparsers.add_parser("final", allow_abbrev=False)
    final.add_argument("--preflight-result", type=Path, required=True)
    final.add_argument("--development-go-summary", type=Path, required=True)
    final.add_argument("--formal-train-cache", type=Path, required=True)
    final.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    final.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    final.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    final.add_argument("--device", default="cuda:0")
    final.add_argument(
        "--output-cache",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT_ROOT / "fresh_test_state_cache.npz",
    )
    final.add_argument(
        "--output-ledger",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT_ROOT / "fresh_test_proposal_ledger.npz",
    )
    final.add_argument(
        "--result", type=Path, default=DEFAULT_FINAL_OUTPUT_ROOT / "final_result.json"
    )
    final.set_defaults(handler=_final)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
