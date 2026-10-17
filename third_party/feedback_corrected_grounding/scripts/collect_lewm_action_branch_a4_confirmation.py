#!/usr/bin/env python3
"""Collect an authorized A4 reserved-test-only cache after Development-Go."""

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

from fcpg.lewm_action_branch_a4_confirmation_collection import (  # noqa: E402
    collect_a4_reserved_test_cache,
    save_a4_reserved_test_cache,
    validate_a4_development_go_summary,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    load_frozen_a4_state_encoder,
    make_a4_episode_runtime,
)
from scripts.evaluate_lewm_action_branch_a4_confirmation import (  # noqa: E402
    prepare_confirmation_checkpoints,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_OUTPUT_CACHE = (
    ROOT / "outputs/lewm_action_branch_a4/reserved_test_state_cache.npz"
)
DEFAULT_BASE_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--development-go-summary", type=Path, required=True)
    parser.add_argument(
        "--formal-train-cache",
        type=Path,
        required=True,
        help="exact formal train/development cache bound by Development-Go",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="the three fixed-latest A4 step-10000 checkpoints",
    )
    parser.add_argument("--output-cache", type=Path, default=DEFAULT_OUTPUT_CACHE)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = _load_json(args.config, label="A4 config")
    summary = _load_json(
        args.development_go_summary, label="A4 Development-Go summary"
    )

    # This verifies the exact Go and formal-cache content before loading any
    # encoder, creating an environment, or touching a reserved task instance.
    authorization = validate_a4_development_go_summary(
        summary,
        contract,
        formal_train_cache_path=args.formal_train_cache,
    )
    output_path = args.output_cache.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    # Validate all confirmation endpoints before creating an environment or
    # touching any reserved task instance. The returned models are not used by
    # collection; strict reconstruction here is the authorization gate.
    prepared = prepare_confirmation_checkpoints(
        args.checkpoints,
        base_checkpoint_path=args.base_checkpoint,
        formal_train_cache_path=str(authorization.formal_train_cache_path),
    )
    checkpoint_seeds = [endpoint.seed for endpoint in prepared]
    del prepared

    encoder = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=args.device,
    )
    collection = contract["collection"]
    runtime_factory = partial(
        make_a4_episode_runtime,
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    reserved = collect_a4_reserved_test_cache(
        contract,
        authorization,
        encoder=encoder,
        runtime_factory=runtime_factory,
    )
    destination = save_a4_reserved_test_cache(output_path, reserved.cache)
    print(
        json.dumps(
            {
                "cache": str(destination),
                "cache_scope": "reserved_test_only",
                "formal_train_cache": str(authorization.formal_train_cache_path),
                "tasks": list(reserved.attempted_instances_by_task),
                "model_scope": "one_shared_model_across_all_seven_tasks",
                "validated_checkpoint_seeds": checkpoint_seeds,
                "test_bundles": int(reserved.cache["split"].shape[0]),
                "attempted_test_episodes_per_task": dict(
                    (task, len(values))
                    for task, values in reserved.attempted_instances_by_task.items()
                ),
                "accepted_test_episodes_per_task": dict(
                    (task, len(values))
                    for task, values in reserved.accepted_instances_by_task.items()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
