from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_a4_storage import ACTION_DIM, LATENT_DIM, STATE_SLOT_COUNT
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    DEVELOPMENT_TRAINING_SEEDS,
    HISTORY_STEPS,
    HORIZON,
    PROTOCOL,
    TASKS,
)
from fcpg.lewm_action_branch_a4_metrics import (
    FACTUAL_METHOD,
    NO_ACTION_METHOD,
    WRONG_ACTION_METHOD,
    summarize_a4_development,
    summarize_a4_preflight,
)
from scripts import score_lewm_action_branch_a4 as development_cli
from scripts import score_lewm_action_branch_a4_preflight as preflight_cli


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"


def _contract() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _panel() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    rows: list[tuple[int, int, str, str, float]] = []
    for task_index in range(len(TASKS)):
        for episode_index in range(2):
            rows.append(
                (
                    task_index,
                    episode_index,
                    f"task-{task_index}:train:{episode_index}",
                    "train",
                    float(task_index + episode_index + 1),
                )
            )
        rows.append(
            (
                task_index,
                2,
                f"task-{task_index}:development:0",
                "development",
                float(task_index + 3),
            )
        )
    count = len(rows)
    proposal_id = np.asarray([f"proposal-{index}" for index in range(count)])
    task_index = np.asarray([row[0] for row in rows], dtype=np.int64)
    task_name = np.asarray([TASKS[row[0]] for row in rows])
    task_instance = np.asarray([row[1] for row in rows], dtype=np.int64)
    episode = np.asarray([row[2] for row in rows])
    split = np.asarray([row[3] for row in rows])
    context = np.asarray([row[4] for row in rows], dtype=np.float32)

    history = np.zeros(
        (count, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float32
    )
    history[:, :, 0, 0] = context[:, None]
    target = np.zeros(
        (count, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM),
        dtype=np.float32,
    )
    branch = (np.arange(BRANCH_COUNT, dtype=np.float32) - 3.5).reshape(
        1, BRANCH_COUNT, 1, 1
    )
    horizon = np.arange(1, HORIZON + 1, dtype=np.float32).reshape(
        1, 1, HORIZON, 1
    )
    target[..., 0] = branch * horizon

    cache = {
        "proposal_id": proposal_id,
        "task_class_index": task_index,
        "task_name": task_name,
        "task_instance_index": task_instance,
        "source_episode_id": episode,
        "split": split,
        "history_state": history,
        "history_action": np.zeros(
            (count, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float32
        ),
        "center_action": np.zeros((count, HORIZON, ACTION_DIM), dtype=np.float32),
        "future_action": np.zeros(
            (count, BRANCH_COUNT, HORIZON, ACTION_DIM), dtype=np.float32
        ),
        "target_state": target,
        "branch_semantic_index": np.broadcast_to(
            np.arange(BRANCH_COUNT, dtype=np.int64), (count, BRANCH_COUNT)
        ).copy(),
    }
    ledger = {
        "proposal_id": proposal_id.copy(),
        "task_class_index": task_index.copy(),
        "task_name": task_name.copy(),
        "task_instance_index": task_instance.copy(),
        "source_episode_id": episode.copy(),
        "environment_seed": np.arange(count, dtype=np.int64) + 100,
        "proposal_raw_step": np.full(count, 10, dtype=np.int64),
        "split": split.copy(),
        "proposal": np.ones(count, dtype=bool),
        "complete_history": np.ones(count, dtype=bool),
        "success": np.zeros(count, dtype=bool),
        "ever_success_before": np.zeros(count, dtype=bool),
        "active_stage": np.ones(count, dtype=bool),
        "expert_xyz_norm": np.full(count, 0.2, dtype=np.float32),
        "recent_action_rms": np.full(count, 0.2, dtype=np.float32),
        "workspace_and_joint_margin": np.full(count, 0.2, dtype=np.float32),
        "unexpected_collision": np.zeros(count, dtype=bool),
        "remaining_raw_steps": np.full(count, 100, dtype=np.int64),
        "eligible": np.ones(count, dtype=bool),
        "rejection_reasons": np.full(count, "", dtype="<U1"),
        "accepted_bundle_index": np.arange(count, dtype=np.int64),
    }
    labels = {
        "proposal_id": proposal_id.copy(),
        "split": split.copy(),
        "task_class_index": task_index.copy(),
        "task_name": task_name.copy(),
        "source_episode_id": episode.copy(),
        "branch_valid": np.ones((count, BRANCH_COUNT), dtype=bool),
        "future_physical_spread": np.ones(count, dtype=np.float32),
        "source_expert_success": np.ones(count, dtype=bool),
        "proprio": np.zeros((count, 2), dtype=np.float32),
        "anchor_validity": np.ones(
            (count, HISTORY_STEPS, STATE_SLOT_COUNT - 1), dtype=bool
        ),
        **{name: context[:, None].copy() for name in (
            "object_pose",
            "goal_relative_progress",
            "contact_or_grasp_state",
            "branch_outcome",
        )},
    }
    return ledger, cache, labels


def _external_gates() -> dict[str, dict[str, bool]]:
    return {
        gate: {task: True for task in TASKS}
        for gate in (
            "environment_and_expert_rollout",
            "snapshot_restore",
            "zero_severe_penetration",
            "fcpg_anchor_smoke",
        )
    }


def _predictions(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    development = cache["split"].astype(str) == "development"
    target = cache["target_state"][development]
    factual = target[None].copy()
    no_action = np.zeros_like(factual)
    wrong = np.empty(
        (1, target.shape[0], BRANCH_COUNT - 1, *target.shape[1:]),
        dtype=np.float32,
    )
    slots = np.arange(BRANCH_COUNT)
    for shift in range(1, BRANCH_COUNT):
        wrong[:, :, shift - 1] = factual[:, :, (slots + shift) % BRANCH_COUNT]
    return {
        FACTUAL_METHOD: factual,
        NO_ACTION_METHOD: no_action,
        WRONG_ACTION_METHOD: wrong,
    }


def test_bounded_smoke_preflight_scores_all_required_gates() -> None:
    ledger, cache, labels = _panel()
    summary = summarize_a4_preflight(
        ledger,
        cache,
        labels,
        _external_gates(),
        contract=_contract(),
        mode="smoke",
    )

    assert summary["decision"] == "Smoke-Go"
    assert summary["passed"]
    assert not summary["training_authorized"]
    assert not summary["reserved_test_access_authorized"]
    assert summary["support"]["physical_support_passed"]
    assert summary["task_state_sufficiency"]["passed"]
    task_state = summary["task_state_sufficiency"]
    assert "shared_only" in task_state["candidates"]
    assert task_state["contextualized_state_gate"]["passed"]


def test_future_branch_outcome_is_diagnostic_not_current_state_gate_target() -> None:
    ledger, cache, labels = _panel()
    baseline = summarize_a4_preflight(
        ledger,
        cache,
        labels,
        _external_gates(),
        contract=_contract(),
        mode="smoke",
    )["task_state_sufficiency"]["contextualized_state_gate"]

    labels["branch_outcome"] = np.sin(
        np.arange(labels["branch_outcome"].shape[0], dtype=np.float32)
    )[:, None]
    changed = summarize_a4_preflight(
        ledger,
        cache,
        labels,
        _external_gates(),
        contract=_contract(),
        mode="smoke",
    )["task_state_sufficiency"]["contextualized_state_gate"]

    assert changed["gate_readouts"] == [
        "object_pose",
        "goal_relative_progress",
        "contact_or_grasp_state",
    ]
    assert changed["diagnostic_readouts"] == ["branch_outcome"]
    assert changed["comparison"]["relative_improvement"] == pytest.approx(
        baseline["comparison"]["relative_improvement"]
    )


@pytest.mark.parametrize("missing", ["future_physical_spread", "object_pose"])
def test_preflight_fails_closed_when_labels_are_absent(missing: str) -> None:
    ledger, cache, labels = _panel()
    labels.pop(missing)
    with pytest.raises(ValueError, match="missing mandatory"):
        summarize_a4_preflight(
            ledger,
            cache,
            labels,
            _external_gates(),
            contract=_contract(),
            mode="smoke",
        )


def test_development_scores_factual_against_all_controls() -> None:
    _, cache, _ = _panel()
    summary = summarize_a4_development(
        cache,
        _predictions(cache),
        contract=_contract(),
        training_seeds=(DEVELOPMENT_TRAINING_SEEDS[0],),
        mode="smoke",
    )

    assert summary["decision"] == "Smoke-Go"
    assert not summary["development_go"]
    assert summary["primary"]["wrong_action_shifts"]["count"] == 7
    assert summary["raw_h6_h8"]["methods"][FACTUAL_METHOD]["task_macro_mse"] == 0.0


def test_development_rejects_wrong_action_replay_drift_and_test_rows() -> None:
    _, cache, _ = _panel()
    predictions = _predictions(cache)
    predictions[WRONG_ACTION_METHOD][0, 0, 0, 0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="cyclic replay"):
        summarize_a4_development(
            cache,
            predictions,
            contract=_contract(),
            training_seeds=(DEVELOPMENT_TRAINING_SEEDS[0],),
            mode="smoke",
        )

    cache["split"] = cache["split"].copy()
    cache["split"][-1] = "test"
    with pytest.raises(PermissionError, match="reserved test"):
        summarize_a4_development(
            cache,
            _predictions(cache),
            contract=_contract(),
            training_seeds=(DEVELOPMENT_TRAINING_SEEDS[0],),
            mode="smoke",
        )


def test_smoke_cli_entrypoints_write_summaries(tmp_path: Path) -> None:
    ledger, cache, labels = _panel()
    ledger_path = tmp_path / "ledger.npz"
    cache_path = tmp_path / "cache.npz"
    labels_path = tmp_path / "labels.npz"
    gates_path = tmp_path / "gates.json"
    np.savez_compressed(ledger_path, **ledger)
    np.savez_compressed(cache_path, **cache)
    np.savez_compressed(labels_path, **labels)
    gates_path.write_text(json.dumps(_external_gates()), encoding="utf-8")

    preflight_output = tmp_path / "preflight"
    assert preflight_cli.main(
        [
            "--config",
            str(CONFIG),
            "--ledger",
            str(ledger_path),
            "--cache",
            str(cache_path),
            "--labels",
            str(labels_path),
            "--external-gates",
            str(gates_path),
            "--output",
            str(preflight_output),
            "--smoke",
        ]
    ) == 0
    assert json.loads((preflight_output / "summary.json").read_text())["decision"] == "Smoke-Go"

    development = cache["split"].astype(str) == "development"
    prediction = _predictions(cache)
    prediction_path = tmp_path / "prediction.npz"
    np.savez_compressed(
        prediction_path,
        protocol=np.asarray(PROTOCOL),
        checkpoint_step=np.asarray(0, dtype=np.int64),
        training_seed=np.asarray(DEVELOPMENT_TRAINING_SEEDS[0], dtype=np.int64),
        split=cache["split"][development],
        proposal_id=cache["proposal_id"][development],
        history_state=cache["history_state"][development],
        future_action=cache["future_action"][development],
        target_state=cache["target_state"][development],
        factual_prediction=prediction[FACTUAL_METHOD][0],
        no_action_prediction=prediction[NO_ACTION_METHOD][0],
        shuffled_action_prediction=prediction[WRONG_ACTION_METHOD][0],
    )
    development_output = tmp_path / "development"
    assert development_cli.main(
        [
            "--config",
            str(CONFIG),
            "--cache",
            str(cache_path),
            "--predictions",
            str(prediction_path),
            "--output",
            str(development_output),
            "--smoke",
        ]
    ) == 0
    assert json.loads((development_output / "summary.json").read_text())["decision"] == "Smoke-Go"


def test_split_first_cli_loader_refuses_reserved_test_archive(tmp_path: Path) -> None:
    path = tmp_path / "reserved.npz"
    np.savez_compressed(
        path,
        split=np.asarray(["test"]),
        payload=np.asarray([123.0]),
    )
    with pytest.raises(PermissionError, match="reserved test"):
        preflight_cli._load_npz_without_reserved_test(path, label="fixture")
