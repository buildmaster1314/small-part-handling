#!/usr/bin/env python3
"""Run the preregistered train/validation posterior-adapter Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_posterior_gate_orchestration import (  # noqa: E402
    load_posterior_gate_inputs,
    run_validation_posterior_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/lewm/train/anchor_persistent_posterior_adapter_gate_v1.yaml"
        ),
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--validation-clean-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = load_posterior_gate_inputs(
        args.config,
        args.train_cache,
        args.validation_cache,
        args.validation_clean_cache,
        repo_root=REPO_ROOT,
    )
    result = run_validation_posterior_gate(inputs, args.output_dir)
    decision = result["decision"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "decision": decision["decision"],
                "sealed_data_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
