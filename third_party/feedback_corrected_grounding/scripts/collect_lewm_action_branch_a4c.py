#!/usr/bin/env python3
"""Collect the phase-aware seven-task A4c train/development state cache."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_a4_storage import (  # noqa: E402
    save_a4_proposal_ledger,
    save_a4_state_cache,
)
from fcpg.lewm_action_branch_a4c_actions import (  # noqa: E402
    build_a4c_action_branch_bank,
)
from fcpg.lewm_action_branch_a4c_schedule import (  # noqa: E402
    PROTOCOL,
    a4c_base_contract,
    load_a4c_contract,
    make_schedule_sampler,
    validate_a4c_contract,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    load_frozen_a4_state_encoder,
    make_a4_episode_runtime,
    run_a4_collection,
    save_a4_diagnostics,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_phase_aware_a4c.json"
DEFAULT_SCHEDULE = ROOT / "outputs/lewm_action_branch_a4c_v4/phase_schedule.json"
DEFAULT_BASE_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4c_v4"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase-schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--max-proposals", type=int, default=None)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("A4c collection permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bounds = (args.max_tasks, args.episodes_per_task, args.max_proposals)
    if not args.smoke and any(value is not None for value in bounds):
        raise ValueError("A4c task/episode/proposal bounds are smoke-only")
    contract = load_a4c_contract(args.config)
    schedule_manifest = _json(args.phase_schedule)
    schedule_sampler = make_schedule_sampler(schedule_manifest, contract=contract)
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    collection = contract["collection"]
    encoder = load_frozen_a4_state_encoder(
        args.base_checkpoint.expanduser().resolve(),
        args.fcpg_checkpoint.expanduser().resolve(),
        device=device,
    )
    runtime_factory = partial(
        make_a4_episode_runtime,
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    base_contract = a4c_base_contract(contract)
    payload = run_a4_collection(
        contract,
        encoder=encoder,
        runtime_factory=runtime_factory,
        smoke=bool(args.smoke),
        max_tasks=args.max_tasks,
        max_episodes_per_task=args.episodes_per_task,
        max_proposals_per_episode=args.max_proposals,
        schedule_sampler=schedule_sampler,
        contract_validator=validate_a4c_contract,
        assignment_contract=base_contract,
        eligibility_contract=base_contract,
        branch_bank_builder=build_a4c_action_branch_bank,
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = save_a4_proposal_ledger(output / "proposal_ledger.npz", payload.proposal_ledger)
    cache_path = save_a4_state_cache(output / "state_cache.npz", payload.state_cache)
    diagnostics_path = save_a4_diagnostics(output / "preflight_labels.npz", payload.diagnostics)
    manifest = {
        "protocol": PROTOCOL,
        "workspace_control_margin": "mocap_signed_headroom_second_smallest_v1",
        "recent_action_window": "four_previous_plus_current_expert_action_v1",
        "status": "complete",
        "schedule_manifest": str(args.phase_schedule.expanduser().resolve()),
        "ledger": str(ledger_path),
        "cache": str(cache_path),
        "diagnostics": str(diagnostics_path),
        "smoke": bool(args.smoke),
        "tasks": payload.metadata["tasks"],
        "proposals": int(payload.proposal_ledger["proposal_id"].shape[0]),
        "bundles": int(payload.state_cache["proposal_id"].shape[0]),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
