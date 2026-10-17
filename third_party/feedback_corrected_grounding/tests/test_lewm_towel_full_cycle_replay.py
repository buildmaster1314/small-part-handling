from __future__ import annotations

import pytest

from fcpg.lewm_towel_full_cycle_replay import (
    TowelCycleInterval,
    select_towel_proposal,
    towel_cycle_intervals,
    towel_split_assignments,
    validate_towel_full_cycle_specs,
)


def _trace() -> tuple[list[str], list[int]]:
    phases = ["approach_towel"] * 20
    cycles = [0] * 20
    for cycle_index in range(6):
        phase = "wipe_right" if cycle_index % 2 == 0 else "wipe_left"
        phases.extend([phase] * 42)
        cycles.extend([cycle_index] * 42)
        phases.extend(["lower_to_start"] * 2)
        cycles.extend([min(cycle_index + 1, 6)] * 2)
    return phases, cycles


def test_cycle_intervals_recover_all_six_directions() -> None:
    phases, cycles = _trace()
    intervals = towel_cycle_intervals(phases, cycles)
    assert len(intervals) == 6
    assert [row.direction for row in intervals] == [
        "right",
        "left",
        "right",
        "left",
        "right",
        "left",
    ]
    assert all(row.end_raw_step - row.start_raw_step == 41 for row in intervals)


def test_full_panel_covers_each_cycle_and_endpoint() -> None:
    phases, cycles = _trace()
    intervals = towel_cycle_intervals(phases, cycles)
    specs = []
    for split, count in (("train", 24), ("development", 8)):
        for ordinal, (cycle_index, fraction) in enumerate(
            towel_split_assignments(split, count)
        ):
            specs.append(
                select_towel_proposal(
                    intervals[cycle_index],
                    split=split,
                    bundle_ordinal=ordinal,
                    fraction=fraction,
                )
            )
    validate_towel_full_cycle_specs(specs)
    assert all(
        row.proposal_raw_step + 40 >= row.interval_end_raw_step for row in specs
    )


def test_wrong_panel_size_fails_closed() -> None:
    with pytest.raises(ValueError, match="exactly 24"):
        towel_split_assignments("train", 23)


def test_missing_cycle_fails_coverage() -> None:
    interval = TowelCycleInterval(0, 60, 100, "right")
    specs = [
        select_towel_proposal(
            interval,
            split="train",
            bundle_ordinal=index,
            fraction=0.5,
        )
        for index in range(24)
    ]
    with pytest.raises(ValueError, match="all six"):
        validate_towel_full_cycle_specs(
            specs, expected_counts={"train": 24}
        )
