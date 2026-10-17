from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_lewm_a4c_phase_support",
    ROOT / "scripts" / "audit_lewm_a4c_phase_support.py",
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
                "task_class_index": 1,
                "task_instance": 0,
                "environment_seed": 100,
                "split": "train",
                "proposal_steps": [10, 20],
                "coverage": {
                    "selected_phase_names": ["approach", "contact"]
                },
            },
            {
                "task": "task-b",
                "task_class_index": 1,
                "task_instance": 1,
                "environment_seed": 101,
                "split": "development",
                "proposal_steps": [10, 20],
                "coverage": {
                    "selected_phase_names": ["approach", "contact"]
                },
            },
            {
                "task": "task-a",
                "task_class_index": 0,
                "task_instance": 0,
                "environment_seed": 200,
                "split": "train",
                "proposal_steps": [5, 15],
                "coverage": {
                    "selected_phase_names": ["approach", "lift"]
                },
            },
        ],
    }


def _ledger(order: tuple[int, ...] = (0, 1, 2, 3, 4, 5)) -> dict[str, np.ndarray]:
    arrays = {
        "proposal_id": np.asarray(
            ["b0-contact", "a-lift", "b1-approach", "a-approach", "b0-approach", "b1-contact"]
        ),
        "task_class_index": np.asarray([1, 0, 1, 0, 1, 1], dtype=np.int64),
        "task_name": np.asarray(
            ["task-b", "task-a", "task-b", "task-a", "task-b", "task-b"]
        ),
        "task_instance_index": np.asarray([0, 0, 1, 0, 0, 1], dtype=np.int64),
        "source_episode_id": np.asarray(["b0", "a0", "b1", "a0", "b0", "b1"]),
        "environment_seed": np.asarray([100, 200, 101, 200, 100, 101], dtype=np.int64),
        "proposal_raw_step": np.asarray([20, 15, 10, 5, 10, 20], dtype=np.int64),
        "split": np.asarray(
            ["train", "train", "development", "train", "train", "development"]
        ),
        "eligible": np.asarray([True, True, True, False, True, True]),
        "accepted_bundle_index": np.asarray([2, 0, 3, -1, 1, 4], dtype=np.int64),
    }
    indices = np.asarray(order, dtype=np.int64)
    return {name: value[indices].copy() for name, value in arrays.items()}


def _cache() -> dict[str, np.ndarray]:
    proposal_id = np.asarray(
        ["a-lift", "b0-approach", "b0-contact", "b1-approach", "b1-contact"]
    )
    task_class = np.asarray([0, 1, 1, 1, 1], dtype=np.int64)
    task_name = np.asarray(["task-a", "task-b", "task-b", "task-b", "task-b"])
    instance = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    episode = np.asarray(["a0", "b0", "b0", "b1", "b1"])
    split = np.asarray(["train", "train", "train", "development", "development"])

    scales = np.arange(5, dtype=np.float32)
    target = np.zeros((5, 2, MODULE.HORIZON, 1), dtype=np.float32)
    target[:, 0, :, 0] = -scales[:, None]
    target[:, 1, :, 0] = scales[:, None]
    return {
        "proposal_id": proposal_id,
        "task_class_index": task_class,
        "task_name": task_name,
        "task_instance_index": instance,
        "source_episode_id": episode,
        "split": split,
        "target_state": target,
    }


def _labels(*, branch_outcome: bool = True) -> dict[str, np.ndarray]:
    cache = _cache()
    labels = {
        name: cache[name].copy()
        for name in (
            "proposal_id",
            "task_class_index",
            "task_name",
            "task_instance_index",
            "source_episode_id",
            "split",
        )
    }
    labels["future_physical_spread"] = np.arange(5, dtype=np.float32)
    labels["branch_valid"] = np.asarray(
        [[True, True], [True, False], [True, True], [False, False], [True, True]]
    )
    if branch_outcome:
        outcome = np.zeros((5, 2, MODULE.HORIZON, 2), dtype=np.float32)
        outcome[1, :, -1, 1] = [0.0, 1.0]
        outcome[2, :, -1, 1] = [1.0, 0.0]
        outcome[4, :, -1, 1] = [1.0, 1.0]
        labels["branch_outcome"] = outcome
    return labels


def _copy(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in arrays.items()}


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    schedule_path = tmp_path / "phase_schedule.json"
    ledger_path = tmp_path / "proposal_ledger.npz"
    cache_path = tmp_path / "state_cache.npz"
    labels_path = tmp_path / "preflight_labels.npz"
    schedule_path.write_text(json.dumps(_schedule()), encoding="utf-8")
    np.savez(ledger_path, **_ledger())
    np.savez(cache_path, **_cache())
    np.savez(labels_path, **_labels())
    return schedule_path, ledger_path, cache_path, labels_path


def test_report_joins_accepted_bundles_and_scores_each_phase() -> None:
    report = MODULE.build_report(_schedule(), _ledger(), _cache(), _labels())

    assert report["overall"]["bundle_count"] == 5
    assert report["overall"]["episode_count"] == 3
    assert list(report["tasks"]) == ["task-a", "task-b"]

    contact = report["tasks"]["task-b"]["phases"]["contact"]
    assert contact["bundle_count"] == 2
    assert contact["episode_count"] == 2
    assert contact["future_physical_spread_quantiles"] == pytest.approx(
        [2.0, 2.5, 3.0, 3.5, 4.0]
    )
    assert contact["future_physical_spread_nonzero_fraction"] == 1.0
    assert contact["latent_branch_variance_quantiles"] == pytest.approx(
        [4.0, 7.0, 10.0, 13.0, 16.0]
    )
    assert contact["latent_branch_variance_nonzero_fraction"] == 1.0
    assert contact["valid_branch_fraction"] == 1.0
    assert contact["h8_success_fraction"] == 0.75

    task_a_approach = report["tasks"]["task-a"]["phases"]["approach"]
    assert task_a_approach["bundle_count"] == 0
    assert task_a_approach["episode_count"] == 0
    assert task_a_approach["future_physical_spread_quantiles"] is None
    assert task_a_approach["latent_branch_variance_nonzero_fraction"] is None
    assert task_a_approach["valid_branch_fraction"] is None
    assert task_a_approach["h8_success_fraction"] is None


def test_report_is_independent_of_ledger_row_order() -> None:
    forward = MODULE.build_report(_schedule(), _ledger(), _cache(), _labels())
    reverse = MODULE.build_report(
        _schedule(), _ledger((5, 4, 3, 2, 1, 0)), _cache(), _labels()
    )

    assert reverse == forward
    assert json.dumps(reverse, sort_keys=True) == json.dumps(forward, sort_keys=True)


def test_h8_success_is_omitted_when_branch_outcome_is_absent() -> None:
    report = MODULE.build_report(
        _schedule(), _ledger(), _cache(), _labels(branch_outcome=False)
    )

    assert "h8_success_fraction" not in report["overall"]
    assert "h8_success_fraction" not in report["tasks"]["task-b"]
    assert (
        "h8_success_fraction"
        not in report["tasks"]["task-b"]["phases"]["contact"]
    )


def test_cli_writes_requested_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    schedule_path, ledger_path, cache_path, labels_path = _write_inputs(tmp_path)
    output_path = tmp_path / "nested" / "phase_support.json"

    assert MODULE.main(
        [
            "--phase-schedule",
            str(schedule_path),
            "--proposal-ledger",
            str(ledger_path),
            "--state-cache",
            str(cache_path),
            "--preflight-labels",
            str(labels_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    written = json.loads(output_path.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert printed == written
    assert written["tasks"]["task-b"]["phases"]["contact"]["bundle_count"] == 2
    assert output_path.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize("source", ["cache", "labels"])
def test_fails_closed_when_accepted_rows_are_reordered(source: str) -> None:
    cache = _cache()
    labels = _labels()
    selected = cache if source == "cache" else labels
    for name in selected:
        selected[name] = selected[name][::-1].copy()

    with pytest.raises(ValueError, match="align|aligned"):
        MODULE.build_report(_schedule(), _ledger(), cache, labels)


def test_fails_closed_when_bundle_indices_do_not_cover_cache() -> None:
    ledger = _ledger()
    ledger["accepted_bundle_index"][ledger["accepted_bundle_index"] == 4] = 3

    with pytest.raises(ValueError, match="contiguous, unique"):
        MODULE.build_report(_schedule(), ledger, _cache(), _labels())


def test_fails_closed_when_an_accepted_proposal_has_no_scheduled_phase() -> None:
    ledger = _ledger()
    ledger["proposal_raw_step"][0] = 21

    with pytest.raises(ValueError, match="no scheduled phase"):
        MODULE.build_report(_schedule(), ledger, _cache(), _labels())


def test_fails_closed_when_cache_and_label_identity_differ() -> None:
    labels = _labels()
    labels["source_episode_id"][2] = "wrong-episode"

    with pytest.raises(ValueError, match="source_episode_id.*not row aligned"):
        MODULE.build_report(_schedule(), _ledger(), _cache(), labels)


def test_fails_closed_on_nonfinite_measurements() -> None:
    labels = _labels()
    labels["future_physical_spread"][0] = np.nan
    with pytest.raises(ValueError, match="finite floating point"):
        MODULE.build_report(_schedule(), _ledger(), _cache(), labels)

    cache = _copy(_cache())
    cache["target_state"][0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite floating point"):
        MODULE.build_report(_schedule(), _ledger(), cache, _labels())
