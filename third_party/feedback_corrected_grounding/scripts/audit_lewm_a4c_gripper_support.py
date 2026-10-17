#!/usr/bin/env python3
"""Audit A4c matched-XYZ gripper interventions before shared training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.lewm_action_branch_a4c_gripper_metrics import (  # noqa: E402
    summarize_gripper_support,
)


DEFAULT_ROOT = ROOT / "outputs/lewm_action_branch_a4c_v4"


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _phase_lookup(schedule: Mapping) -> dict[tuple[str, int, int], str]:
    result: dict[tuple[str, int, int], str] = {}
    for episode in schedule.get("episodes", ()):
        if not isinstance(episode, Mapping):
            raise TypeError("schedule episodes must be objects")
        for step, phase in zip(
            episode["proposal_steps"],
            episode["coverage"]["selected_phase_names"],
            strict=True,
        ):
            key = (
                str(episode["task"]),
                int(episode["task_instance"]),
                int(step),
            )
            if key in result:
                raise ValueError(f"duplicate schedule phase key {key}")
            result[key] = str(phase)
    return result


def _render(report: Mapping) -> str:
    lines = [
        "# A4c gripper intervention support",
        "",
        f"Decision: **{'Data-Go' if report['passed'] else 'Data-No-Go'}**",
        f"Command contract: **{report['command_contract_valid_rows']}/{report['row_count']}** rows",
        "",
        "| task / split / phase | rows | aperture response | contact divergence | pass |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, row in report["groups"].items():
        lines.append(
            f"| {key} | {row['row_count']} | "
            f"{row['aperture_response_fraction']:.2%} | "
            f"{row['contact_divergence_fraction']:.2%} | "
            f"{'yes' if row['passed'] else 'no'} |"
        )
    lines.extend(["", str(report["claim_boundary"])])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache", type=Path, default=DEFAULT_ROOT / "state_cache.npz")
    parser.add_argument("--labels", type=Path, default=DEFAULT_ROOT / "preflight_labels.npz")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_ROOT / "proposal_ledger.npz")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_ROOT / "phase_schedule.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "gripper_support")
    args = parser.parse_args()

    cache = _npz(args.cache)
    labels = _npz(args.labels)
    ledger = _npz(args.ledger)
    schedule = _json(args.schedule)
    if not np.array_equal(cache["proposal_id"], labels["proposal_id"]):
        raise ValueError("cache and labels proposal identities differ")
    ledger_rows = {
        str(proposal_id): (
            str(ledger["task_name"][index]),
            int(ledger["task_instance_index"][index]),
            int(ledger["proposal_raw_step"][index]),
        )
        for index, proposal_id in enumerate(ledger["proposal_id"])
    }
    lookup = _phase_lookup(schedule)
    phases = []
    for proposal_id in np.asarray(cache["proposal_id"]).astype(str):
        try:
            phases.append(lookup[ledger_rows[proposal_id]])
        except KeyError as exc:
            raise ValueError(f"phase join failed for proposal {proposal_id}") from exc

    report = summarize_gripper_support(
        future_action=cache["future_action"],
        branch_semantic_index=cache["branch_semantic_index"],
        branch_gripper_aperture=labels["branch_gripper_aperture"],
        branch_contact_or_grasp=labels["branch_contact_or_grasp"],
        task_name=cache["task_name"],
        split=cache["split"],
        phase_name=phases,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.md").write_text(_render(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": "Data-Go" if report["passed"] else "Data-No-Go",
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
