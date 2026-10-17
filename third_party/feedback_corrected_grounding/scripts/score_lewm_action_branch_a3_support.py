#!/usr/bin/env python3
"""Score the zero-training A3 physical and per-task latent support Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a1 import load_a1_npz  # noqa: E402
from fcpg.lewm_action_branch_a3 import summarize_a3_support  # noqa: E402
from fcpg.lewm_action_branch_a3_collection import (  # noqa: E402
    TASKS,
    validate_a3_cache_conformance,
    validate_a3_contract,
)
from fcpg.lewm_action_branch_a3_physical import (  # noqa: E402
    validate_a3_physical_arrays,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"
DEFAULT_LATENT = ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_screen.npz"
DEFAULT_PHYSICAL = ROOT / "outputs/lewm_action_branch_a3/action_branch_a3_physical.npz"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/lewm_action_branch_a3/support_gate"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--latent", type=Path, default=DEFAULT_LATENT)
    parser.add_argument("--physical", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A3 config root must be an object")
    return value


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# A3 zero-training physical support Gate",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "| task | physical support | latent support | joint support | physical median TCP | physical median object | latent median | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for task, row in summary["per_task"].items():
        lines.append(
            "| {task} | {physical:.1%} | {latent:.1%} | {joint:.1%} | {tcp:.5f} m | "
            "{object:.5f} m | {variance:.6f} | {decision} |".format(
                task=task,
                physical=row["physical_episode_macro_informative_fraction"],
                latent=row["latent_episode_macro_informative_fraction"],
                joint=row["joint_episode_macro_informative_fraction"],
                tcp=row["median_tcp_max_pairwise_m"],
                object=row["median_object_max_pairwise_m"],
                variance=row["median_latent_branch_variance"],
                decision="Go" if row["passed"] else "No-Go",
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = _load_json(args.config)
    validate_a3_contract(contract)
    latent = load_a1_npz(args.latent.expanduser().resolve())
    validate_a3_cache_conformance(latent, contract)
    with np.load(args.physical.expanduser().resolve(), allow_pickle=False) as store:
        physical = {name: store[name] for name in store.files}
    validate_a3_physical_arrays(physical, latent_arrays=latent)
    gate = contract["support_gate"]
    summary = summarize_a3_support(
        latent,
        physical,
        task_names=TASKS,
        primary_horizons=tuple(gate["primary_horizons"]),
        minimum_latent_variance=float(gate["minimum_latent_branch_variance"]),
        minimum_tcp_spread_m=float(gate["minimum_tcp_max_pairwise_m"]),
        minimum_object_spread_m=float(gate["minimum_object_max_pairwise_m"]),
        minimum_episode_macro_fraction=float(
            gate["minimum_episode_macro_informative_fraction_per_task"]
        ),
        expected_train_episodes=int(contract["split"]["train_episodes_per_task"]),
        expected_development_episodes=int(
            contract["split"]["development_episodes_per_task"]
        ),
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
