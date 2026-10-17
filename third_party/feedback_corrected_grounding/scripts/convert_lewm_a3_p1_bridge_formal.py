#!/usr/bin/env python3
"""Convert the full strict A3/P1 replay cache into formal bridge splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch


os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_a3_p1_bridge_collection import (  # noqa: E402
    validate_strict_p1_checkpoint,
)
from fcpg.lewm_formal_bridge_conversion import (  # noqa: E402
    write_formal_bridge_splits,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked-source", type=Path, required=True)
    parser.add_argument("--stacked-metadata", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--geometry-batch-size", type=int, default=1024)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.geometry_batch_size <= 0:
        raise ValueError("geometry-batch-size must be positive")
    source = args.stacked_source.expanduser().resolve()
    metadata = (
        source.with_suffix(".metadata.json")
        if args.stacked_metadata is None
        else args.stacked_metadata.expanduser().resolve()
    )
    receipt = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or not isinstance(receipt.get("p1"), dict):
        raise TypeError("stacked metadata is missing P1 provenance")
    p1 = receipt["p1"]
    policy_path = Path(str(p1.get("policy_path", ""))).expanduser().resolve()
    corrector_path = Path(str(p1.get("corrector_path", ""))).expanduser().resolve()
    # Re-run the strict collector policy/corrector binding before loading code.
    validate_strict_p1_checkpoint(policy_path, corrector_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    policy, _ = load_action_chunk_policy(policy_path, device=device)
    policy.requires_grad_(False).eval()
    report = write_formal_bridge_splits(
        stacked_source=source,
        stacked_metadata=metadata,
        output_root=args.output_root,
        policy=policy,
        policy_device=device,
        collector_implementation=ROOT / "fcpg/lewm_a3_p1_bridge_collection.py",
        geometry_batch_size=args.geometry_batch_size,
    )
    report_path = args.output_root.expanduser().resolve() / "conversion_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
