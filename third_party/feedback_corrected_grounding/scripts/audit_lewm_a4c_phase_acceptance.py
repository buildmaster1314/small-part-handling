#!/usr/bin/env python3
"""Audit A4c proposal acceptance and rejection reasons by task phase."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_PROTOCOL = "fcpg_action_branch_a4c_phase_acceptance_audit_v1"
SCHEDULE_PROTOCOLS = {
    "fcpg_action_branch_phase_schedule_a4c_v1",
    "fcpg_action_branch_phase_schedule_a4c_v2",
    "fcpg_action_branch_phase_schedule_a4c_v3",
    "fcpg_action_branch_phase_schedule_a4c_v4",
}
SCHEDULE_PROTOCOL = "fcpg_action_branch_phase_schedule_a4c_v4"
_REQUIRED_LEDGER_KEYS = (
    "task_name",
    "task_instance_index",
    "proposal_raw_step",
    "eligible",
    "rejection_reasons",
)

JoinKey = tuple[str, int, int]


def load_schedule(path: str | Path) -> dict[str, Any]:
    """Load one A4c phase schedule JSON object."""

    source = Path(path).expanduser()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A4c phase schedule must contain one JSON object")
    return value


def load_ledger(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ proposal ledger without permitting pickled arrays."""

    source = Path(path).expanduser()
    with np.load(source, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _schedule_lookup(schedule: Mapping[str, Any]) -> dict[JoinKey, dict[str, Any]]:
    if schedule.get("protocol") not in SCHEDULE_PROTOCOLS:
        raise ValueError("A4c phase schedule protocol mismatch")
    episodes = schedule.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("A4c phase schedule must contain a non-empty episodes list")

    lookup: dict[JoinKey, dict[str, Any]] = {}
    episode_identities: set[tuple[str, int]] = set()
    for episode_index, episode in enumerate(episodes):
        label = f"schedule episode {episode_index}"
        if not isinstance(episode, Mapping):
            raise TypeError(f"{label} must be an object")
        task = episode.get("task")
        if not isinstance(task, str) or not task.strip():
            raise TypeError(f"{label} task must be a non-empty string")
        instance = _nonnegative_integer(
            episode.get("task_instance"), f"{label} task_instance"
        )
        episode_identity = (task, instance)
        if episode_identity in episode_identities:
            raise ValueError(f"duplicate schedule episode {episode_identity}")
        episode_identities.add(episode_identity)

        steps = episode.get("proposal_steps")
        coverage = episode.get("coverage")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"{label} proposal_steps must be a non-empty list")
        if not isinstance(coverage, Mapping):
            raise TypeError(f"{label} coverage must be an object")
        phases = coverage.get("selected_phase_names")
        if not isinstance(phases, list):
            raise TypeError(f"{label} selected_phase_names must be a list")
        if len(steps) != len(phases):
            raise ValueError(
                f"{label} proposal_steps and selected_phase_names length mismatch"
            )

        environment_seed = episode.get("environment_seed")
        if environment_seed is not None:
            environment_seed = _nonnegative_integer(
                environment_seed, f"{label} environment_seed"
            )
        split = episode.get("split")
        if split is not None and (not isinstance(split, str) or not split.strip()):
            raise TypeError(f"{label} split must be a non-empty string")

        previous_step = -1
        for position, (step_value, phase_value) in enumerate(
            zip(steps, phases, strict=True)
        ):
            step = _nonnegative_integer(
                step_value, f"{label} proposal_steps[{position}]"
            )
            if step <= previous_step:
                raise ValueError(f"{label} proposal_steps must be sorted and unique")
            previous_step = step
            if not isinstance(phase_value, str) or not phase_value.strip():
                raise TypeError(
                    f"{label} selected_phase_names[{position}] must be non-empty"
                )
            key = (task, instance, step)
            if key in lookup:
                raise ValueError(f"duplicate scheduled proposal {key}")
            lookup[key] = {
                "phase": phase_value,
                "environment_seed": environment_seed,
                "split": split,
            }
    return lookup


def _vector(
    ledger: Mapping[str, np.ndarray], name: str, count: int | None = None
) -> np.ndarray:
    value = ledger[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"ledger {name} must be a numpy array")
    if value.ndim != 1:
        raise ValueError(f"ledger {name} must be one-dimensional")
    if count is not None and value.shape != (count,):
        raise ValueError(f"ledger {name} must have shape ({count},)")
    return value


def _text_vector(
    ledger: Mapping[str, np.ndarray], name: str, count: int
) -> np.ndarray:
    value = _vector(ledger, name, count)
    if value.dtype.kind not in {"S", "U"}:
        raise TypeError(f"ledger {name} must contain fixed-width strings")
    return value.astype(str, copy=False)


def _integer_vector(
    ledger: Mapping[str, np.ndarray], name: str, count: int
) -> np.ndarray:
    value = _vector(ledger, name, count)
    if value.dtype.kind not in {"i", "u"}:
        raise TypeError(f"ledger {name} must contain integers")
    result = value.astype(np.int64, copy=False)
    if bool(np.any(result < 0)):
        raise ValueError(f"ledger {name} must be non-negative")
    return result


def _reasons(value: str, *, accepted: bool, row_index: int) -> tuple[str, ...]:
    if accepted:
        if value != "":
            raise ValueError(f"accepted ledger row {row_index} has rejection reasons")
        return ()
    if not value:
        raise ValueError(f"rejected ledger row {row_index} has no rejection reason")
    reasons = tuple(value.split("|"))
    if any(not reason or reason.strip() != reason for reason in reasons):
        raise ValueError(f"ledger row {row_index} has malformed rejection reasons")
    if len(set(reasons)) != len(reasons):
        raise ValueError(f"ledger row {row_index} repeats a rejection reason")
    return reasons


def join_schedule_and_ledger(
    schedule: Mapping[str, Any], ledger: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Join each ledger decision to its precomputed A4c phase name."""

    missing = sorted(set(_REQUIRED_LEDGER_KEYS) - set(ledger))
    if missing:
        raise ValueError(f"proposal ledger is missing keys: {missing}")
    lookup = _schedule_lookup(schedule)

    task_names_raw = _vector(ledger, "task_name")
    count = int(task_names_raw.shape[0])
    if count <= 0:
        raise ValueError("proposal ledger must contain at least one row")
    task_names = _text_vector(ledger, "task_name", count)
    if any(not task.strip() for task in task_names):
        raise ValueError("ledger task_name values must be non-empty")
    instances = _integer_vector(ledger, "task_instance_index", count)
    steps = _integer_vector(ledger, "proposal_raw_step", count)
    eligible = _vector(ledger, "eligible", count)
    if eligible.dtype.kind != "b":
        raise TypeError("ledger eligible must contain booleans")
    reason_values = _text_vector(ledger, "rejection_reasons", count)

    accepted_indices: np.ndarray | None = None
    if "accepted_bundle_index" in ledger:
        accepted_indices = _vector(ledger, "accepted_bundle_index", count)
        if accepted_indices.dtype.kind not in {"i", "u"}:
            raise TypeError("ledger accepted_bundle_index must contain integers")
        accepted_indices = accepted_indices.astype(np.int64, copy=False)

    environment_seeds: np.ndarray | None = None
    if "environment_seed" in ledger:
        environment_seeds = _integer_vector(ledger, "environment_seed", count)
    splits: np.ndarray | None = None
    if "split" in ledger:
        splits = _text_vector(ledger, "split", count)

    joined: list[dict[str, Any]] = []
    seen: set[JoinKey] = set()
    for row_index in range(count):
        key = (
            str(task_names[row_index]),
            int(instances[row_index]),
            int(steps[row_index]),
        )
        if key in seen:
            raise ValueError(f"duplicate proposal ledger join key {key}")
        seen.add(key)
        scheduled = lookup.get(key)
        if scheduled is None:
            raise ValueError(f"ledger proposal has no scheduled phase: {key}")

        if environment_seeds is not None and scheduled["environment_seed"] is not None:
            if int(environment_seeds[row_index]) != scheduled["environment_seed"]:
                raise ValueError(f"ledger environment_seed differs from schedule for {key}")
        if splits is not None and scheduled["split"] is not None:
            if str(splits[row_index]) != scheduled["split"]:
                raise ValueError(f"ledger split differs from schedule for {key}")

        accepted = bool(eligible[row_index])
        if accepted_indices is not None:
            bundle_index = int(accepted_indices[row_index])
            if accepted and bundle_index < 0:
                raise ValueError(
                    f"accepted ledger row {row_index} has no accepted bundle index"
                )
            if not accepted and bundle_index != -1:
                raise ValueError(
                    f"rejected ledger row {row_index} has an accepted bundle index"
                )
        joined.append(
            {
                "task": key[0],
                "phase": scheduled["phase"],
                "accepted": accepted,
                "rejection_reasons": _reasons(
                    str(reason_values[row_index]),
                    accepted=accepted,
                    row_index=row_index,
                ),
            }
        )
    return joined


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposed = len(rows)
    accepted = sum(bool(row["accepted"]) for row in rows)
    reasons = Counter(
        reason
        for row in rows
        for reason in row["rejection_reasons"]
    )
    return {
        "proposed": proposed,
        "accepted": accepted,
        "rejected": proposed - accepted,
        "acceptance_rate": accepted / proposed,
        "rejection_reason_histogram": {
            reason: int(reasons[reason]) for reason in sorted(reasons)
        },
    }


def build_report(
    schedule: Mapping[str, Any], ledger: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Return an order-independent JSON-compatible acceptance report."""

    joined = join_schedule_and_ledger(schedule, ledger)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_task[str(row["task"])].append(row)

    tasks: dict[str, Any] = {}
    for task in sorted(by_task):
        task_rows = by_task[task]
        by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            by_phase[str(row["phase"])].append(row)
        tasks[task] = {
            **_counts(task_rows),
            "phases": {
                phase: _counts(by_phase[phase]) for phase in sorted(by_phase)
            },
        }
    return {
        "schema_version": 1,
        "protocol": AUDIT_PROTOCOL,
        "overall": _counts(joined),
        "tasks": tasks,
    }


def run(
    schedule_path: str | Path,
    ledger_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Load inputs, build the audit, and write deterministic JSON."""

    report = build_report(load_schedule(schedule_path), load_ledger(ledger_path))
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args.schedule, args.ledger, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
