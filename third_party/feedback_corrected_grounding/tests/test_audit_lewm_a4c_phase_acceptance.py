from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_lewm_a4c_phase_acceptance",
    ROOT / "scripts" / "audit_lewm_a4c_phase_acceptance.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _schedule() -> dict[str, object]:
    return {
        "protocol": MODULE.SCHEDULE_PROTOCOL,
        "episodes": [
            {
                "task": "task-b",
                "task_instance": 0,
                "environment_seed": 100,
                "split": "train",
                "proposal_steps": [10, 20, 30],
                "coverage": {
                    "selected_phase_names": ["approach", "contact", "contact"]
                },
            },
            {
                "task": "task-a",
                "task_instance": 1,
                "environment_seed": 101,
                "split": "development",
                "proposal_steps": [5, 15, 25],
                "coverage": {
                    "selected_phase_names": ["approach", "lift", "lift"]
                },
            },
        ],
    }


def _ledger(order: tuple[int, ...] = (0, 1, 2, 3, 4, 5)) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        "task_name": np.asarray(
            ["task-b", "task-a", "task-b", "task-a", "task-b", "task-a"]
        ),
        "task_instance_index": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        "environment_seed": np.asarray([100, 101, 100, 101, 100, 101], dtype=np.int64),
        "proposal_raw_step": np.asarray([20, 25, 10, 5, 30, 15], dtype=np.int64),
        "split": np.asarray(
            ["train", "development", "train", "development", "train", "development"]
        ),
        "eligible": np.asarray([False, True, True, False, False, True]),
        "rejection_reasons": np.asarray(
            [
                "inactive_stage|low_action",
                "",
                "",
                "inactive_stage",
                "low_action",
                "",
            ]
        ),
        "accepted_bundle_index": np.asarray([-1, 0, 1, -1, -1, 2], dtype=np.int64),
    }
    indices = np.asarray(order, dtype=np.int64)
    return {name: array[indices] for name, array in values.items()}


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    schedule_path = tmp_path / "schedule.json"
    ledger_path = tmp_path / "ledger.npz"
    schedule_path.write_text(json.dumps(_schedule()), encoding="utf-8")
    np.savez(ledger_path, **_ledger())
    return schedule_path, ledger_path


def test_report_joins_phases_and_aggregates_all_rejection_reasons() -> None:
    report = MODULE.build_report(_schedule(), _ledger())

    assert report["overall"] == {
        "proposed": 6,
        "accepted": 3,
        "rejected": 3,
        "acceptance_rate": 0.5,
        "rejection_reason_histogram": {
            "inactive_stage": 2,
            "low_action": 2,
        },
    }
    assert list(report["tasks"]) == ["task-a", "task-b"]
    task_a = report["tasks"]["task-a"]
    assert task_a["proposed"] == 3
    assert task_a["accepted"] == 2
    assert task_a["acceptance_rate"] == pytest.approx(2.0 / 3.0)
    assert task_a["phases"]["approach"]["rejected"] == 1
    assert task_a["phases"]["lift"]["accepted"] == 2

    task_b_contact = report["tasks"]["task-b"]["phases"]["contact"]
    assert task_b_contact["proposed"] == 2
    assert task_b_contact["accepted"] == 0
    assert task_b_contact["acceptance_rate"] == 0.0
    assert task_b_contact["rejection_reason_histogram"] == {
        "inactive_stage": 1,
        "low_action": 2,
    }


def test_report_is_independent_of_ledger_row_order() -> None:
    forward = MODULE.build_report(_schedule(), _ledger())
    reverse = MODULE.build_report(_schedule(), _ledger((5, 4, 3, 2, 1, 0)))

    assert reverse == forward
    assert json.dumps(reverse, indent=2, sort_keys=True) == json.dumps(
        forward, indent=2, sort_keys=True
    )


def test_cli_writes_requested_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    schedule_path, ledger_path = _write_inputs(tmp_path)
    output_path = tmp_path / "nested" / "audit.json"

    assert MODULE.main(
        [
            "--schedule",
            str(schedule_path),
            "--ledger",
            str(ledger_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert printed == written
    assert written["overall"]["proposed"] == 6
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_fails_closed_when_a_ledger_proposal_has_no_scheduled_phase() -> None:
    ledger = _ledger()
    ledger["proposal_raw_step"][0] = 21

    with pytest.raises(ValueError, match="no scheduled phase"):
        MODULE.build_report(_schedule(), ledger)


def test_fails_closed_when_schedule_phase_names_are_not_aligned() -> None:
    schedule = _schedule()
    schedule["episodes"][0]["coverage"]["selected_phase_names"].pop()

    with pytest.raises(ValueError, match="length mismatch"):
        MODULE.build_report(schedule, _ledger())
