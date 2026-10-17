#!/usr/bin/env python3
"""Run the one frozen Counterfactual v2 structured CostAdapter comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_counterfactual_v2_structured_cost import (
    run_structured_cost_experiment,
)


DEFAULT_CONFIG = (
    "config/lewm/eval/counterfactual_v2_structured_cost_e1_e3_seed8201.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the single horizon-structured Direct/Predicted/Oracle "
            "CostAdapter comparison. This does not retrain FCPG/LeWM and "
            "does not launch CEM."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--confirm-external-development",
        action="store_true",
        help=(
            "acknowledge that the source is external development and the "
            "result cannot support a formal paper claim"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = run_structured_cost_experiment(
        config,
        repo_root=ROOT,
        confirm_external_development=bool(args.confirm_external_development),
    )
    print(f"output: {config['output']['directory']}")
    print(f"source_result_sha256: {manifest['source_result_sha256']}")
    print(f"result_sha256: {manifest['result_sha256']}")
    print(f"parameter_count: {manifest['parameter_fairness']['counts']['direct']}")
    print("scope: external_development (formal_claim_allowed=false)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
