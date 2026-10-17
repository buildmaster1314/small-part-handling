#!/usr/bin/env python3
"""Score the fresh A2 confirmation training-data support guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a2_confirmation import (  # noqa: E402
    confirmation_support,
    validate_confirmation_contract,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_continuous_task_a2_confirm.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    tasks = validate_confirmation_contract(contract)
    arrays = load_a1_npz(args.input.expanduser().resolve())
    if set(arrays["split"].astype(str).tolist()) != {"train"}:
        raise ValueError("confirmation support guard accepts train rows only")
    gate = contract["support_gate"]
    summary, _ = confirmation_support(
        arrays,
        task_names=tasks,
        minimum_branch_variance=float(gate["minimum_branch_variance"]),
        minimum_episode_macro_fraction_per_task=float(
            gate["minimum_episode_macro_informative_fraction_per_task"]
        ),
        expected_episodes_per_task=int(gate["episodes_per_task"]),
    )
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
