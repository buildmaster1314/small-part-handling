#!/usr/bin/env python3
"""Score the collection-only intervention-support Gate for A2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a2 import (  # noqa: E402
    summarize_intervention_support,
    validate_a2_contract,
)


DEFAULT_CONFIG = "config/lewm/eval/action_branch_continuous_task_a2.json"
DEFAULT_INPUT = "outputs/lewm_action_branch_a2/action_branch_a2_screen.npz"
DEFAULT_OUTPUT = "outputs/lewm_action_branch_a2/support_gate/summary.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("config root must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# A2 continuous-task intervention support",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "| task | bundles | informative | fraction | median variance | decision |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for task, row in summary["per_task"].items():
        lines.append(
            f"| {task} | {row['bundle_count']} | {row['informative_bundle_count']} | "
            f"{row['episode_macro_informative_fraction']:.1%} | "
            f"{row['median_branch_variance']:.6f} | "
            f"{'Pass' if row['passed'] else 'Fail'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_json(_resolve(args.config))
    task_names = validate_a2_contract(config)
    gate = config["support_gate"]
    arrays = load_a1_npz(_resolve(args.input))
    full_horizon = summarize_intervention_support(
        arrays,
        task_names=task_names,
        minimum_branch_variance=float(gate["minimum_branch_variance"]),
        minimum_episode_macro_fraction_per_task=float(
            gate["minimum_episode_macro_informative_fraction_per_task"]
        ),
        minimum_development_episodes_per_task=int(
            gate["minimum_development_episodes_per_task"]
        ),
    )
    summary = summarize_intervention_support(
        arrays,
        task_names=task_names,
        primary_horizons=tuple(int(value) for value in gate["primary_horizons"]),
        minimum_branch_variance=float(gate["minimum_branch_variance"]),
        minimum_episode_macro_fraction_per_task=float(
            gate["minimum_episode_macro_informative_fraction_per_task"]
        ),
        minimum_development_episodes_per_task=int(
            gate["minimum_development_episodes_per_task"]
        ),
    )
    summary["full_horizon_diagnostic"] = {
        "decision": full_horizon["decision"],
        "primary_horizons": full_horizon["primary_horizons"],
        "per_task": full_horizon["per_task"],
    }
    summary["late_horizon_development_only"] = True
    summary["decision"] = (
        "Development-Go" if summary["decision"] == "Go" else "Development-No-Go"
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
