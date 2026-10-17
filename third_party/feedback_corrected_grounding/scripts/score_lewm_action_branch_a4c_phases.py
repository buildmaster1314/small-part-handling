#!/usr/bin/env python3
"""Report phase-resolved A4c factual versus no-action development errors."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4_metrics import (  # noqa: E402
    FACTUAL_METHOD,
    NO_ACTION_METHOD,
)
from fcpg.lewm_action_branch_a4c_phase_metrics import (  # noqa: E402
    summarize_phase_development,
)
from scripts.score_lewm_action_branch_a4 import (  # noqa: E402
    load_a4_prediction_panel,
)


DEFAULT_ROOT = ROOT / "outputs/lewm_action_branch_a4c_v4"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _ledger(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _phase_lookup(schedule: Mapping) -> dict[tuple[str, int, int], tuple[str, bool]]:
    result: dict[tuple[str, int, int], tuple[str, bool]] = {}
    for row in schedule.get("episodes", ()):  # validated separately by collection
        if not isinstance(row, Mapping):
            raise TypeError("schedule episodes must be objects")
        steps = row["proposal_steps"]
        phases = row["coverage"]["selected_phase_names"]
        first_success = row.get("first_success_step")
        for step, phase in zip(steps, phases, strict=True):
            key = (str(row["task"]), int(row["task_instance"]), int(step))
            crossing = bool(
                first_success is not None
                and int(step) < int(first_success) <= int(step) + 40
            )
            if key in result:
                raise ValueError(f"duplicate phase schedule key {key}")
            result[key] = (str(phase), crossing)
    return result


def _render(summary: Mapping) -> str:
    lines = [
        "# A4c phase-resolved development diagnostic",
        "",
        "| task | phase | rows | episodes | factual | no-action | improvement | seeds+ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, task_row in summary["tasks"].items():
        for phase, row in task_row["phases"].items():
            lines.append(
                f"| {task} | {phase} | {row['row_count']} | "
                f"{row['physical_episode_count']} | {row['factual_mse']:.6f} | "
                f"{row['no_action_mse']:.6f} | {row['relative_improvement']:.2%} | "
                f"{row['positive_seed_count']}/{len(summary['training_seeds'])} |"
            )
    lines.extend(["", str(summary["claim_boundary"])])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache", type=Path, default=DEFAULT_ROOT / "state_cache.npz")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_ROOT / "proposal_ledger.npz")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_ROOT / "phase_schedule.json")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "phase_development")
    args = parser.parse_args()

    cache, predictions, seeds = load_a4_prediction_panel(
        args.cache, args.predictions
    )
    ledger = _ledger(args.ledger)
    schedule = _json(args.schedule)
    phase_lookup = _phase_lookup(schedule)
    ledger_by_id = {
        str(proposal_id): (
            str(ledger["task_name"][index]),
            int(ledger["task_instance_index"][index]),
            int(ledger["proposal_raw_step"][index]),
        )
        for index, proposal_id in enumerate(ledger["proposal_id"])
    }
    development = np.asarray(cache["split"]).astype(str) == "development"
    phases: list[str] = []
    crossings: list[bool] = []
    for proposal_id in np.asarray(cache["proposal_id"])[development].astype(str):
        try:
            key = ledger_by_id[proposal_id]
            phase, crossing = phase_lookup[key]
        except KeyError as exc:
            raise ValueError(f"phase join failed for proposal {proposal_id}") from exc
        phases.append(phase)
        crossings.append(crossing)

    summary = summarize_phase_development(
        target=np.asarray(cache["target_state"])[development],
        factual_prediction=predictions[FACTUAL_METHOD],
        no_action_prediction=predictions[NO_ACTION_METHOD],
        task_name=np.asarray(cache["task_name"])[development].astype(str),
        episode_id=np.asarray(cache["source_episode_id"])[development].astype(str),
        phase_name=phases,
        success_crossing=crossings,
        seeds=seeds,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.md").write_text(_render(summary), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tasks": len(summary["tasks"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
