#!/usr/bin/env python3
"""Collect the strict physical A3-10k to Pick-P1 bridge cache."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_a3_p1_bridge_collection import (  # noqa: E402
    A3_RECONSTRUCTION_ATOL,
    P1_VISIBLE_GROUNDING_MODES,
    PICK_TASK,
    STRICT_A3_10K_SEEDS,
    replay_one_a3_p1_row,
    select_pick_bridge_rows,
    sha256_file,
    stack_replayed_bridge_rows,
    validate_a3_10k_checkpoint,
    validate_bridge_arrays,
    validate_strict_p1_checkpoint,
)
from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a3_collection import (  # noqa: E402
    validate_a3_cache_conformance,
    validate_a3_contract,
)
from fcpg.lewm_action_branch_collection import (  # noqa: E402
    load_frozen_fcpg_anchor_encoder,
    make_metaworld_episode_runtime,
)
from fcpg.lewm_anchor_dynamics_pair import (  # noqa: E402
    load_matched_predictor_pair_checkpoint,
)
from fcpg.lewm_counterfactual_v2_runtime import (  # noqa: E402
    _close_episode_env_safely,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
DEFAULT_A3_SCREEN = ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_screen.npz"
DEFAULT_ANCHOR = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_A3_CORRECTOR = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_a3_p1_bridge/strict_pick_train_development_smoke.npz"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--a3-screen", type=Path, default=DEFAULT_A3_SCREEN)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--a3-corrector-checkpoint", type=Path, default=DEFAULT_A3_CORRECTOR)
    parser.add_argument(
        "--predictor-checkpoint",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly three times for A3 seeds 29301/29302/29303",
    )
    # Deliberately no default: a historical shared-four-task policy is not P1.
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--p1-corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--p1-visible-grounding",
        choices=P1_VISIBLE_GROUNDING_MODES,
        default="fcpg",
        help="Encode P1 labels at native FCPG or matched nominal points.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--confirm-full-collection", action="store_true")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--branch-index", type=int, action="append", default=None)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A3 config root must be an object")
    return value


def _shape_report(arrays: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    return {name: list(np.asarray(value).shape) for name, value in sorted(arrays.items())}


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    with np.load(destination, allow_pickle=False) as stored:
        reloaded = {name: stored[name] for name in stored.files}
    validate_bridge_arrays(reloaded)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.smoke:
        max_instances = 1 if args.max_instances is None else int(args.max_instances)
        max_rows = 1 if args.max_rows is None else int(args.max_rows)
        branches = (0,) if args.branch_index is None else tuple(args.branch_index)
        if max_instances != 1 or max_rows != 1 or len(branches) != 1:
            raise ValueError("first strict smoke is fixed to 1 instance / 1 row / 1 branch")
    else:
        if args.max_instances is not None or args.max_rows is not None:
            raise ValueError("full collection forbids row/instance truncation")
        branches = tuple(range(8)) if args.branch_index is None else tuple(args.branch_index)
        if branches != tuple(range(8)):
            raise ValueError("full collection requires all eight A3 branches in order")
        max_instances = None
        max_rows = None

    target_device = torch.device(args.device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {target_device}")
    p1_provenance = validate_strict_p1_checkpoint(
        args.policy_checkpoint, args.p1_corrector_checkpoint
    )
    policy, policy_args = load_action_chunk_policy(
        args.policy_checkpoint.expanduser().resolve(), device=target_device
    )
    policy.requires_grad_(False).eval()
    if Path(str(policy_args["fcpg_checkpoint"])).as_posix() != Path(
        str(torch.load(
            args.policy_checkpoint,
            map_location="cpu",
            weights_only=False,
        )["args"]["fcpg_checkpoint"])
    ).as_posix():
        raise RuntimeError("loaded P1 policy args changed embedded corrector provenance")

    predictor_entries: list[tuple[int, Any, Any, dict[str, Any]]] = []
    for path in args.predictor_checkpoint:
        factual, no_action, checkpoint = load_matched_predictor_pair_checkpoint(
            path, device=target_device
        )
        provenance = validate_a3_10k_checkpoint(checkpoint, path=path)
        factual.requires_grad_(False).eval()
        no_action.requires_grad_(False).eval()
        predictor_entries.append((int(provenance["seed"]), factual, no_action, provenance))
    predictor_entries.sort(key=lambda item: item[0])
    seeds = tuple(item[0] for item in predictor_entries)
    if seeds != STRICT_A3_10K_SEEDS:
        raise ValueError(
            f"strict collector requires A3 seeds {STRICT_A3_10K_SEEDS}, got {seeds}"
        )

    contract = _read_json(args.config)
    tasks = validate_a3_contract(contract)
    if not tasks or tasks[0] != PICK_TASK:
        raise ValueError("A3 task index zero is not Pick")
    source_path = args.a3_screen.expanduser().resolve()
    source = load_a1_npz(source_path)
    validate_a3_cache_conformance(source, contract)
    rows = select_pick_bridge_rows(
        source, max_instances=max_instances, max_rows=max_rows
    )
    anchor = load_frozen_fcpg_anchor_encoder(
        args.anchor_checkpoint.expanduser().resolve(),
        args.a3_corrector_checkpoint.expanduser().resolve(),
        device=target_device,
    )
    collection = contract["collection"]
    runtime_factory = partial(
        make_metaworld_episode_runtime,
        camera=str(collection["camera"]),
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    replayed = []
    started_at = time.monotonic()
    total_rows = int(rows.size)
    for completed_before, row in enumerate(rows):
        task_index = int(source["task_index"][row])
        instance = int(source["task_instance_index"][row])
        environment_seed = (
            int(collection["environment_seed_base"])
            + task_index * 10_000
            + instance
        )
        runtime = runtime_factory(PICK_TASK, instance, environment_seed)
        try:
            replayed.append(
                replay_one_a3_p1_row(
                    source,
                    int(row),
                    runtime=runtime,
                    anchor_encoder=anchor,
                    policy=policy,
                    predictors=[(item[1], item[2]) for item in predictor_entries],
                    branch_indices=branches,
                    policy_device=target_device,
                    predictor_device=target_device,
                    p1_visible_grounding_mode=str(args.p1_visible_grounding),
                )
            )
        finally:
            _close_episode_env_safely(runtime.env)
        completed = completed_before + 1
        if completed % 10 == 0 or completed == total_rows:
            elapsed = time.monotonic() - started_at
            seconds_per_row = elapsed / completed
            print(
                json.dumps(
                    {
                        "event": "strict_bridge_collection_progress",
                        "completed_rows": completed,
                        "total_rows": total_rows,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": seconds_per_row * (total_rows - completed),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output_arrays = stack_replayed_bridge_rows(
        replayed, source, predictor_seeds=seeds
    )
    output = _save_npz(args.output, output_arrays)
    metadata = {
        "schema_version": 1,
        "protocol": "strict-a3-10k-to-pick-p1-physical-bridge-v1",
        "smoke": bool(args.smoke),
        "task": PICK_TASK,
        "p1_history_alignment": "raw-adjacent-pre-action-[t-1,t]",
        "p1_visible_grounding_mode": str(args.p1_visible_grounding),
        "a3_history_alignment": "decision-synchronous-frame-skip-5",
        "hidden_clean_targets_are_offline_labels_only": True,
        "source_action_byte_equality_checked": True,
        "p1_adapter_token_reconstruction_checked": True,
        "a3_history_and_target_reconstruction_checked": True,
        "a3_numeric_reconstruction_atol": A3_RECONSTRUCTION_ATOL,
        "a3_numeric_reconstruction_note": (
            "state/action/image replay remains exact; atol covers CUDA autocast "
            "FP16 batching/kernel scheduling only"
        ),
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "config": str(args.config.expanduser().resolve()),
            "config_sha256": sha256_file(args.config),
        },
        "anchor_encoder": {
            "checkpoint": str(args.anchor_checkpoint.expanduser().resolve()),
            "sha256": sha256_file(args.anchor_checkpoint),
            "corrector": str(args.a3_corrector_checkpoint.expanduser().resolve()),
            "corrector_sha256": sha256_file(args.a3_corrector_checkpoint),
        },
        "p1": p1_provenance,
        "predictors": [item[3] for item in predictor_entries],
        "selection": {
            "source_rows": rows.tolist(),
            "branch_indices": list(branches),
            "max_instances": max_instances,
            "max_rows": max_rows,
        },
        "checks": {
            "max_a3_history_reconstruction_error": float(
                np.max(output_arrays["max_a3_history_reconstruction_error"])
            ),
            "max_a3_target_reconstruction_error": float(
                np.max(output_arrays["max_a3_target_reconstruction_error"])
            ),
        },
        "arrays": _shape_report(output_arrays),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
