from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replace_lewm_towel_full_cycle_rows",
    ROOT / "scripts" / "replace_lewm_towel_full_cycle_rows.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_coverage_payload_records_full_cycle_span() -> None:
    from fcpg.lewm_towel_full_cycle_replay import (
        TowelCycleInterval,
        select_towel_proposal,
        towel_split_assignments,
    )

    intervals = tuple(
        TowelCycleInterval(
            cycle_index=index,
            start_raw_step=60 + index * 44,
            end_raw_step=101 + index * 44,
            direction="right" if index % 2 == 0 else "left",
        )
        for index in range(6)
    )
    specs = []
    for split, count in (("train", 24), ("development", 8)):
        for ordinal, (cycle, fraction) in enumerate(
            towel_split_assignments(split, count)
        ):
            specs.append(
                select_towel_proposal(
                    intervals[cycle],
                    split=split,
                    bundle_ordinal=ordinal,
                    fraction=fraction,
                )
            )
    payload = MODULE._coverage_payload(specs)
    assert payload["status"] == "coverage_go"
    assert payload["splits"]["train"]["cycles"] == {
        str(index): 4 for index in range(6)
    }
    assert payload["splits"]["development"]["proposal_raw_step_max"] > 250


def test_towel_metadata_defaults_are_representable() -> None:
    count = 5
    metadata = {
        "replacement": np.zeros(count, dtype=bool),
        "cycle": np.full(count, -1, dtype=np.int64),
        "fraction": np.full(count, np.nan, dtype=np.float32),
    }
    assert not bool(metadata["replacement"].any())
    assert np.all(metadata["cycle"] == -1)
    assert np.isnan(metadata["fraction"]).all()
