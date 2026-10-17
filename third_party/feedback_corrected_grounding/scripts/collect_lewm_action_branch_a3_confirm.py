#!/usr/bin/env python3
"""Open A3 reserved test after Development-Go and build train+test canonical."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a1 import load_a1_npz, save_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a3_confirmation import (  # noqa: E402
    load_a3_development_go_summary,
    merge_a3_confirmation_canonical,
    run_a3_confirmation_collection,
    validate_a3_confirmation_contract,
)
from fcpg.lewm_action_branch_a3_collection import (  # noqa: E402
    validate_a3_cache_conformance,
)
from fcpg.lewm_action_branch_collection import (  # noqa: E402
    load_frozen_fcpg_anchor_encoder,
    make_metaworld_episode_runtime,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
DEFAULT_SCREEN = ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_screen.npz"
DEFAULT_DEVELOPMENT_SUMMARY = (
    ROOT / "outputs/lewm_action_branch_a3/a3d_screen_gate/summary.json"
)
DEFAULT_ANCHOR_CHECKPOINT = ROOT / (
    "outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/"
    "lewm_grounded_latest.pt"
)
DEFAULT_FCPG_CHECKPOINT = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_confirmation.npz"
)


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (ROOT / source).resolve()


def _load_contract(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("A3 config root must be a JSON object")
    validate_a3_confirmation_contract(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--development-summary",
        "--screen-summary",
        dest="development_summary",
        type=Path,
        default=DEFAULT_DEVELOPMENT_SUMMARY,
    )
    parser.add_argument(
        "--screen",
        "--screen-npz",
        "--input-npz",
        dest="screen",
        type=Path,
        default=DEFAULT_SCREEN,
    )
    parser.add_argument(
        "--output",
        "--output-npz",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_CHECKPOINT
    )
    parser.add_argument(
        "--fcpg-checkpoint", type=Path, default=DEFAULT_FCPG_CHECKPOINT
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = _load_contract(args.config)

    # These CPU-only checks happen before any encoder or environment is opened.
    development_summary = load_a3_development_go_summary(
        _resolve(args.development_summary),
        contract=contract,
    )
    screen_path = _resolve(args.screen)
    screen = load_a1_npz(screen_path)
    validate_a3_cache_conformance(screen, contract)

    output_path = _resolve(args.output)
    if output_path == screen_path:
        raise ValueError("A3 confirmation output must not overwrite the screen NPZ")

    collection = contract["collection"]
    encoder = load_frozen_fcpg_anchor_encoder(
        _resolve(args.anchor_checkpoint),
        _resolve(args.fcpg_checkpoint),
        device=args.device,
    )
    runtime_factory = partial(
        make_metaworld_episode_runtime,
        camera=str(collection["camera"]),
        width=int(collection["width"]),
        height=int(collection["height"]),
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    test_payload = run_a3_confirmation_collection(
        contract,
        development_summary=development_summary,
        encoder=encoder,
        runtime_factory=runtime_factory,
    )
    canonical = merge_a3_confirmation_canonical(
        screen,
        test_payload.arrays,
        contract,
    )
    destination = save_a1_npz(output_path, canonical)
    split = np.asarray(canonical["split"]).astype(str)
    print(
        json.dumps(
            {
                "output": str(destination),
                "train_bundles": int(np.count_nonzero(split == "train")),
                "test_bundles": int(np.count_nonzero(split == "test")),
                "development_bundles": int(
                    np.count_nonzero(split == "development")
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
