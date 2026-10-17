from __future__ import annotations

import numpy as np

from fcpg.lewm_a4c_control_index import build_a4c_control_index
from fcpg.lewm_action_branch_a4 import TASKS


def _fixture() -> tuple[dict, dict, dict, dict, dict]:
    raw_rows = []
    cache_rows = []
    ledger_rows = []
    diagnostic_rows = []
    schedule_rows = []
    cache_index = 0
    physical_index = 0
    for task_index, task in enumerate(TASKS):
        for split, clock in (("train", 2), ("development", 1)):
            for episode in range(2):
                task_instance = task_index * 100 + (0 if split == "train" else 10) + episode
                steps = [10, 15, 20, 25, 30, 35, 40, 45]
                phases = [
                    "phase_a",
                    "phase_a",
                    "phase_b",
                    "phase_b",
                    "phase_c",
                    "phase_c",
                    "phase_d",
                    "phase_d",
                ]
                schedule_rows.append(
                    {
                        "task": task,
                        "task_instance": task_instance,
                        "first_success_step": 50,
                        "proposal_steps": steps,
                        "coverage": {"selected_phase_names": phases},
                    }
                )
                source = f"{task}:{split}:{episode}"
                for step, phase in zip(steps, phases, strict=True):
                    proposal = f"{task}:{split}:{episode}:{step}"
                    semantics = np.asarray([3, 0, 2, 1, 4, 5, 6, 7], dtype=np.int64)
                    center = np.full((8, 20), 0.2 + 0.01 * task_index, dtype=np.float32)
                    branches = np.zeros((8, 8, 20), dtype=np.float32)
                    branches[1] = center
                    cache_rows.append(
                        {
                            "proposal_id": proposal,
                            "task_class_index": task_index,
                            "task_name": task,
                            "task_instance_index": task_instance,
                            "source_episode_id": source,
                            "split": split,
                            "center_action": center,
                            "history_action": np.full(
                                (2, 20), 0.1, dtype=np.float32
                            ),
                            "branch_semantic_index": semantics,
                        }
                    )
                    ledger_rows.append(
                        {
                            "proposal_id": proposal,
                            "accepted_bundle_index": cache_index,
                            "proposal_raw_step": step,
                            "success": False,
                            "ever_success_before": False,
                            "active_stage": True,
                        }
                    )
                    diagnostic_rows.append(
                        {
                            "proposal_id": proposal,
                            "goal_relative_progress": [step / 50.0],
                            "contact_or_grasp_state": [step >= 30],
                        }
                    )
                    raw_rows.append(
                        {
                            "cache_index": cache_index,
                            "proposal_id": proposal,
                            "task_index": task_index,
                            "task_name": task,
                            "task_instance": task_instance,
                            "proposal_raw_step": step,
                            "source_episode_id": source,
                            "split": split,
                            "future_action": branches,
                            "proposal_phase_name": phase,
                            "physical_bundle_index": physical_index,
                            "camera_name": f"clock{clock:02d}_e35",
                            "camera_clock_hour": clock,
                        }
                    )
                    cache_index += 1
                    physical_index += 1

    def arrays(rows: list[dict]) -> dict[str, np.ndarray]:
        return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}

    return (
        arrays(raw_rows),
        arrays(cache_rows),
        arrays(ledger_rows),
        arrays(diagnostic_rows),
        {"episodes": schedule_rows},
    )


def test_build_control_index_extracts_expert_branch() -> None:
    raw, cache, ledger, diagnostics, schedule = _fixture()
    payload, summary = build_a4c_control_index(
        raw, cache, ledger, diagnostics, schedule
    )

    assert payload["expert_action_prefix"].shape[1:] == (8, 20)
    assert payload["history_action"].shape[1:] == (2, 20)
    assert np.all(payload["expert_branch_slot"] == 1)
    assert set(payload["task_name"].tolist()) == set(TASKS)
    assert summary["maximum_replay_expert_xyz_error"] == 0.0
    assert summary["h8_crosses_success_fraction"] > 0.0


def test_control_index_rejects_early_only_steps() -> None:
    raw, cache, ledger, diagnostics, schedule = _fixture()
    raw["proposal_raw_step"] = raw["proposal_raw_step"] % 4
    ledger["proposal_raw_step"] = raw["proposal_raw_step"].copy()

    try:
        build_a4c_control_index(raw, cache, ledger, diagnostics, schedule)
    except ValueError as exc:
        assert "narrow time window" in str(exc)
    else:
        raise AssertionError("early-only control rows must be rejected")


def test_control_index_rejects_camera_split_leakage() -> None:
    raw, cache, ledger, diagnostics, schedule = _fixture()
    row = int(np.flatnonzero(raw["split"] == "development")[0])
    raw["camera_clock_hour"][row] = 2

    try:
        build_a4c_control_index(raw, cache, ledger, diagnostics, schedule)
    except ValueError as exc:
        assert "odd clock views" in str(exc)
    else:
        raise AssertionError("camera split leakage must be rejected")


def test_control_index_rejects_post_success_rows() -> None:
    raw, cache, ledger, diagnostics, schedule = _fixture()
    ledger["success"][0] = True

    try:
        build_a4c_control_index(raw, cache, ledger, diagnostics, schedule)
    except ValueError as exc:
        assert "success rows" in str(exc)
    else:
        raise AssertionError("post-success control rows must be rejected")


def test_control_index_rejects_expert_branch_mismatch() -> None:
    raw, cache, ledger, diagnostics, schedule = _fixture()
    raw["future_action"][0, 1, 0, 0] += 0.1

    try:
        build_a4c_control_index(raw, cache, ledger, diagnostics, schedule)
    except ValueError as exc:
        assert "expert replay XYZ differs from center_action" in str(exc)
    else:
        raise AssertionError("expert branch mismatch must be rejected")
