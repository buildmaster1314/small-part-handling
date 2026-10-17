"""Deterministic full-cycle replay planning for the A4 towel task."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


TOWEL_TASK = "towel-wipe-v0"
REQUIRED_HALF_CYCLES = 6
BRANCH_HORIZON_RAW_STEPS = 40
MINIMUM_HISTORY_RAW_STEPS = 10
TRAIN_FRACTIONS = (0.25, 0.45, 0.65, 0.85)
DEVELOPMENT_ASSIGNMENTS = (
    (0, 0.55),
    (1, 0.55),
    (2, 0.55),
    (3, 0.55),
    (4, 0.55),
    (5, 0.55),
    (1, 0.85),
    (4, 0.85),
)


@dataclass(frozen=True, slots=True)
class TowelCycleInterval:
    """Inclusive raw-step interval for one physical wiping half-cycle."""

    cycle_index: int
    start_raw_step: int
    end_raw_step: int
    direction: str

    def __post_init__(self) -> None:
        if not 0 <= int(self.cycle_index) < REQUIRED_HALF_CYCLES:
            raise ValueError("towel cycle_index must be in [0,5]")
        if int(self.start_raw_step) < 0 or int(self.end_raw_step) < int(
            self.start_raw_step
        ):
            raise ValueError("towel cycle interval must be non-empty and ordered")
        expected = "right" if int(self.cycle_index) % 2 == 0 else "left"
        if str(self.direction) != expected:
            raise ValueError("towel cycle direction disagrees with cycle parity")


@dataclass(frozen=True, slots=True)
class TowelProposalSpec:
    """One active wiping proposal selected before the H=8 replay branch."""

    split: str
    bundle_ordinal: int
    cycle_index: int
    fraction: float
    interval_start_raw_step: int
    interval_end_raw_step: int
    proposal_raw_step: int
    direction: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "development"}:
            raise ValueError("towel proposal split must be train or development")
        if int(self.bundle_ordinal) < 0:
            raise ValueError("towel bundle ordinal must be non-negative")
        if not 0.0 <= float(self.fraction) <= 1.0:
            raise ValueError("towel proposal fraction must be in [0,1]")
        interval = TowelCycleInterval(
            cycle_index=int(self.cycle_index),
            start_raw_step=int(self.interval_start_raw_step),
            end_raw_step=int(self.interval_end_raw_step),
            direction=str(self.direction),
        )
        proposal = int(self.proposal_raw_step)
        if not interval.start_raw_step <= proposal <= interval.end_raw_step:
            raise ValueError("towel proposal must lie inside its wiping interval")
        if proposal < MINIMUM_HISTORY_RAW_STEPS:
            raise ValueError("towel proposal lacks the required causal history")
        if proposal + BRANCH_HORIZON_RAW_STEPS < interval.end_raw_step:
            raise ValueError("towel H=8 replay must include the endpoint transition")


def towel_cycle_intervals(
    phases: Sequence[str], half_cycles: Sequence[int]
) -> tuple[TowelCycleInterval, ...]:
    """Recover all six contiguous wipe intervals from one expert trajectory."""

    phase_values = tuple(str(value) for value in phases)
    cycle_values = np.asarray(half_cycles, dtype=np.int64)
    if cycle_values.shape != (len(phase_values),):
        raise ValueError("towel phase and half-cycle traces must be aligned")
    intervals = []
    for cycle_index in range(REQUIRED_HALF_CYCLES):
        phase = "wipe_right" if cycle_index % 2 == 0 else "wipe_left"
        steps = np.flatnonzero(
            (cycle_values == cycle_index)
            & (np.asarray(phase_values, dtype=str) == phase)
        )
        if steps.size == 0:
            raise ValueError(f"towel trace is missing half-cycle {cycle_index}")
        if steps.size > 1 and not bool(np.all(np.diff(steps) == 1)):
            raise ValueError(f"towel half-cycle {cycle_index} is not contiguous")
        intervals.append(
            TowelCycleInterval(
                cycle_index=cycle_index,
                start_raw_step=int(steps[0]),
                end_raw_step=int(steps[-1]),
                direction="right" if cycle_index % 2 == 0 else "left",
            )
        )
    for left, right in zip(intervals, intervals[1:], strict=False):
        if right.start_raw_step <= left.end_raw_step:
            raise ValueError("towel half-cycle intervals overlap or regress")
    return tuple(intervals)


def towel_split_assignments(split: str, count: int) -> tuple[tuple[int, float], ...]:
    """Return the frozen cycle/fraction assignments for one split."""

    if split == "train":
        expected = tuple(
            (cycle_index, fraction)
            for fraction in TRAIN_FRACTIONS
            for cycle_index in range(REQUIRED_HALF_CYCLES)
        )
    elif split == "development":
        expected = DEVELOPMENT_ASSIGNMENTS
    else:
        raise ValueError("towel split must be train or development")
    if int(count) != len(expected):
        raise ValueError(
            f"towel {split} replacement requires exactly {len(expected)} bundles"
        )
    return expected


def select_towel_proposal(
    interval: TowelCycleInterval,
    *,
    split: str,
    bundle_ordinal: int,
    fraction: float,
) -> TowelProposalSpec:
    """Choose an in-stroke proposal whose H=8 target crosses the endpoint."""

    span = int(interval.end_raw_step - interval.start_raw_step)
    proposed = int(round(interval.start_raw_step + float(fraction) * span))
    proposed = max(
        proposed,
        int(interval.end_raw_step - BRANCH_HORIZON_RAW_STEPS),
        MINIMUM_HISTORY_RAW_STEPS,
    )
    proposed = min(proposed, int(interval.end_raw_step))
    return TowelProposalSpec(
        split=str(split),
        bundle_ordinal=int(bundle_ordinal),
        cycle_index=int(interval.cycle_index),
        fraction=float(fraction),
        interval_start_raw_step=int(interval.start_raw_step),
        interval_end_raw_step=int(interval.end_raw_step),
        proposal_raw_step=proposed,
        direction=str(interval.direction),
    )


def validate_towel_full_cycle_specs(
    specs: Sequence[TowelProposalSpec],
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> None:
    """Fail when a replacement panel omits a cycle, direction, or split quota."""

    rows = tuple(specs)
    if not rows:
        raise ValueError("towel full-cycle proposal panel cannot be empty")
    counts = Counter(row.split for row in rows)
    expected = (
        {"train": 24, "development": 8}
        if expected_counts is None
        else {str(key): int(value) for key, value in expected_counts.items()}
    )
    if dict(counts) != expected:
        raise ValueError(
            f"towel full-cycle split counts differ: {dict(counts)} != {expected}"
        )
    for split in expected:
        split_rows = [row for row in rows if row.split == split]
        cycle_counts = Counter(row.cycle_index for row in split_rows)
        if set(cycle_counts) != set(range(REQUIRED_HALF_CYCLES)):
            raise ValueError(f"towel {split} panel does not cover all six half-cycles")
        if split == "train" and any(cycle_counts[index] != 4 for index in range(6)):
            raise ValueError("towel train panel requires four windows per half-cycle")
        if split == "development" and any(
            cycle_counts[index] < 1 for index in range(6)
        ):
            raise ValueError("towel development panel requires every half-cycle")
        if {row.direction for row in split_rows} != {"left", "right"}:
            raise ValueError(f"towel {split} panel must include both directions")
        ordinals = [row.bundle_ordinal for row in split_rows]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError(f"towel {split} bundle ordinals must be unique")


__all__ = [
    "BRANCH_HORIZON_RAW_STEPS",
    "DEVELOPMENT_ASSIGNMENTS",
    "MINIMUM_HISTORY_RAW_STEPS",
    "REQUIRED_HALF_CYCLES",
    "TOWEL_TASK",
    "TRAIN_FRACTIONS",
    "TowelCycleInterval",
    "TowelProposalSpec",
    "select_towel_proposal",
    "towel_cycle_intervals",
    "towel_split_assignments",
    "validate_towel_full_cycle_specs",
]
