from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_action_branch_a1 import NPZ_KEYS
from fcpg.lewm_action_branch_a3 import build_diverse_action_branch_bank
from fcpg.lewm_action_branch_a3_collection import (
    PROTOCOL,
    TASKS,
    _a3_candidate_seeds,
    assign_a3_splits,
)
from fcpg.lewm_action_branch_a3_confirmation import (
    DEVELOPMENT_GO,
    DEVELOPMENT_STAGE,
    EXPECTED_GATE_CHECKS,
    _expected_hazards,
    load_a3_development_go_summary,
    merge_a3_confirmation_canonical,
    run_a3_confirmation_collection,
    validate_a3_confirmation_contract,
    validate_a3_confirmation_test_conformance,
    validate_a3_development_go_summary,
)
from fcpg.lewm_action_branch_a3_metrics import (
    CHECKPOINT_STEP,
    PRIMARY_HORIZONS,
    TRAINING_SEEDS,
)
from fcpg.lewm_action_branch_collection import (
    FORMAL_TASK_INSTANCES,
    ActionBranchRow,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/lewm/eval/action_branch_physical_support_a3.json"


def _contract() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _development_go_summary(contract: dict) -> dict:
    assignments = assign_a3_splits(TASKS, contract)
    per_task = {}
    for task_index, task in enumerate(TASKS):
        bundle_count = sum(
            len(
                _expected_hazards(
                    contract,
                    task_index=task_index,
                    instance=instance,
                )
            )
            for instance in FORMAL_TASK_INSTANCES
            if assignments[(task, instance)] == "development"
        )
        per_task[task] = {
            "train_physical_episode_count": 24,
            "development_physical_episode_count": 12,
            "development_hazard_bundle_count": bundle_count,
        }
    comparison = {
        "factual_method": "factual",
        "effect": 0.20,
        "positive_task_count": 3,
        "task_count": 4,
        "positive_seed_count": 2,
        "seed_count": 3,
    }
    return {
        "protocol": PROTOCOL,
        "stage": DEVELOPMENT_STAGE,
        "decision": DEVELOPMENT_GO,
        "passed": True,
        "input_contains_test_rows": False,
        "checkpoint_step": CHECKPOINT_STEP,
        "primary_horizons": list(PRIMARY_HORIZONS),
        "coverage": {
            "tasks": list(TASKS),
            "training_seeds": list(TRAINING_SEEDS),
            "development_hazard_bundle_count": sum(
                row["development_hazard_bundle_count"]
                for row in per_task.values()
            ),
            "per_task": per_task,
        },
        "whitening": {
            "source": "canonical_train_rows",
            "per_task": True,
            "episode_equal": True,
            "fit_horizons": list(PRIMARY_HORIZONS),
            "tasks": {
                task: {
                    "train_physical_episode_count": 24,
                    "floored_dimensions": 0,
                }
                for task in TASKS
            },
        },
        "comparisons": {
            "no_action": {**comparison, "baseline_method": "no_action"},
            "shuffled_action": {
                **comparison,
                "baseline_method": "shuffled_action",
            },
        },
        "shuffled_action_shifts": {
            "count": 7,
            "minimum_effect": 0.20,
            "per_shift": [
                {
                    "donor_shift": shift,
                    "effect": 0.20,
                    "positive_task_count": 3,
                    "positive_seed_count": 2,
                }
                for shift in range(1, 8)
            ],
        },
        "raw_h6_h8": {
            "methods": {
                "factual": {"task_macro_mse": 1.0},
                "no_action": {"task_macro_mse": 1.0},
                "hold_last": {"task_macro_mse": 2.0},
                "latent_constant_velocity": {"task_macro_mse": 3.0},
            }
        },
        "gate": {
            "decision": DEVELOPMENT_GO,
            "passed": True,
            "checks": {name: True for name in EXPECTED_GATE_CHECKS},
            "failed_checks": [],
            "minimum_relative_improvement": 0.05,
            "jointly_positive_tasks": list(TASKS),
            "jointly_positive_seeds": list(TRAINING_SEEDS),
            "cross_replicating_seeds": list(TRAINING_SEEDS),
            "cross_replicating_seed_task_counts": {
                str(seed): len(TASKS) for seed in TRAINING_SEEDS
            },
        },
    }


def _arrays_for_splits(contract: dict, splits: tuple[str, ...]) -> dict[str, np.ndarray]:
    assignments = assign_a3_splits(TASKS, contract)
    protocol_seed = int(contract["split"]["protocol_seed"])
    identities: list[tuple[int, int, str, int, str]] = []
    centers: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    for task_index, task in enumerate(TASKS):
        for instance in FORMAL_TASK_INSTANCES:
            split = assignments[(task, instance)]
            if split not in splits:
                continue
            for hazard in _expected_hazards(
                contract,
                task_index=task_index,
                instance=instance,
            ):
                center = np.zeros((40, 4), dtype=np.float32)
                direction_seed, permutation_seed = _a3_candidate_seeds(
                    protocol_seed,
                    task_index,
                    instance,
                    hazard,
                )
                bank = build_diverse_action_branch_bank(
                    center,
                    direction_seed=direction_seed,
                    permutation_seed=permutation_seed,
                    motion_delta_rms=None,
                )
                identities.append(
                    (
                        task_index,
                        instance,
                        f"{task}:task-instance-{instance:02d}",
                        hazard,
                        split,
                    )
                )
                centers.append(center.reshape(8, 20))
                futures.append(bank.actions.reshape(8, 8, 20))
    count = len(identities)
    return {
        "history_embedding": np.broadcast_to(
            np.zeros((1, 3, 3, 192), dtype=np.float32),
            (count, 3, 3, 192),
        ),
        "history_action": np.zeros((count, 2, 20), dtype=np.float32),
        "center_action": np.stack(centers),
        "future_action": np.stack(futures),
        "target_embedding": np.broadcast_to(
            np.zeros((1, 8, 8, 3, 192), dtype=np.float32),
            (count, 8, 8, 3, 192),
        ),
        "task_index": np.asarray([row[0] for row in identities], dtype=np.int64),
        "task_instance_index": np.asarray(
            [row[1] for row in identities], dtype=np.int64
        ),
        "source_episode_id": np.asarray([row[2] for row in identities]),
        "hazard_id": np.asarray([row[3] for row in identities], dtype=np.int64),
        "split": np.asarray([row[4] for row in identities]),
    }


@pytest.fixture(scope="module")
def contract() -> dict:
    return _contract()


@pytest.fixture(scope="module")
def test_arrays(contract: dict) -> dict[str, np.ndarray]:
    return _arrays_for_splits(contract, ("test",))


def test_confirmation_contract_locks_reserved_test_and_training(contract: dict) -> None:
    assert validate_a3_confirmation_contract(contract) == TASKS

    drifted = deepcopy(contract)
    drifted["confirmation_after_development_go"][
        "reserved_test_episodes_per_task"
    ] = 11
    with pytest.raises(ValueError, match="confirmation contract drifted"):
        validate_a3_confirmation_contract(drifted)

    wrong_type = deepcopy(contract)
    wrong_type["confirmation_after_development_go"]["optimizer_steps"] = 10_000.0
    with pytest.raises(ValueError, match="confirmation contract drifted"):
        validate_a3_confirmation_contract(wrong_type)

    missing_replication = deepcopy(contract)
    missing_replication["confirmation_after_development_go"].pop(
        "minimum_cross_replicating_seeds"
    )
    with pytest.raises(ValueError, match="confirmation contract drifted"):
        validate_a3_confirmation_contract(missing_replication)


def test_development_go_summary_is_read_and_jointly_validated(
    tmp_path: Path, contract: dict
) -> None:
    summary = _development_go_summary(contract)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    assert load_a3_development_go_summary(path, contract=contract) == summary
    assert "sha256" not in json.dumps(summary).lower()

    summary["gate"]["checks"]["late_effect_vs_every_shuffled_shift"] = False
    with pytest.raises(ValueError, match="does not authorize"):
        validate_a3_development_go_summary(summary, contract)

    inconsistent = _development_go_summary(contract)
    inconsistent["comparisons"]["no_action"]["effect"] = 0.01
    with pytest.raises(ValueError, match="no_action comparison failed"):
        validate_a3_development_go_summary(inconsistent, contract)

    insufficient_cross_seeds = _development_go_summary(contract)
    insufficient_cross_seeds["gate"]["cross_replicating_seeds"] = [
        TRAINING_SEEDS[0]
    ]
    with pytest.raises(ValueError, match="cross-replicating seeds"):
        validate_a3_development_go_summary(insufficient_cross_seeds, contract)

    insufficient_tasks = _development_go_summary(contract)
    insufficient_tasks["gate"]["cross_replicating_seed_task_counts"][
        str(TRAINING_SEEDS[0])
    ] = 2
    with pytest.raises(ValueError, match="evidence is inconsistent"):
        validate_a3_development_go_summary(insufficient_tasks, contract)


def test_no_go_is_rejected_before_any_episode_collection(
    monkeypatch: pytest.MonkeyPatch, contract: dict
) -> None:
    summary = _development_go_summary(contract)
    summary["decision"] = "Development-No-Go"
    calls = 0

    def forbidden_collection(**_kwargs: object) -> list[ActionBranchRow]:
        nonlocal calls
        calls += 1
        raise AssertionError("reserved test was opened")

    monkeypatch.setattr(
        "fcpg.lewm_action_branch_a3_confirmation.collect_source_episode",
        forbidden_collection,
    )
    with pytest.raises(ValueError, match="Development-Go"):
        run_a3_confirmation_collection(
            contract,
            development_summary=summary,
            encoder=object(),
            runtime_factory=lambda *_args: None,
        )
    assert calls == 0


def test_collection_uses_only_frozen_test_instances_env_seeds_and_a3_bank(
    monkeypatch: pytest.MonkeyPatch,
    contract: dict,
    test_arrays: dict[str, np.ndarray],
) -> None:
    calls: list[dict[str, object]] = []

    class Encoder:
        def encode(self, _records: object) -> np.ndarray:
            raise AssertionError("fake collector must not invoke the encoder")

    def fake_collection(**kwargs: object) -> list[ActionBranchRow]:
        calls.append(kwargs)
        task_index = int(kwargs["task_seed_index"])
        instance = int(kwargs["task_index_mod50"])
        selected = (test_arrays["task_index"] == task_index) & (
            test_arrays["task_instance_index"] == instance
        )
        indices = np.flatnonzero(selected)
        return [
            ActionBranchRow(
                task=str(kwargs["task"]),
                task_index=task_index,
                task_instance_index=instance,
                source_episode_id=str(test_arrays["source_episode_id"][row]),
                environment_seed=int(kwargs["environment_seed"]),
                hazard_step=int(test_arrays["hazard_id"][row]),
                split=str(kwargs["split"]),
                center_action=test_arrays["center_action"][row].reshape(40, 4),
                history_action=test_arrays["history_action"][row].reshape(2, 5, 4),
                branch_action=test_arrays["future_action"][row].reshape(8, 40, 4),
                history_embedding=test_arrays["history_embedding"][row],
                future_embedding=test_arrays["target_embedding"][row],
            )
            for row in indices
        ]

    monkeypatch.setattr(
        "fcpg.lewm_action_branch_a3_confirmation.collect_source_episode",
        fake_collection,
    )
    payload = run_a3_confirmation_collection(
        contract,
        development_summary=_development_go_summary(contract),
        encoder=Encoder(),
        runtime_factory=lambda *_args: None,
    )

    assert len(calls) == len(TASKS) * 12
    assert set(np.asarray(payload.arrays["split"]).astype(str)) == {"test"}
    for call in calls:
        task_index = int(call["task_seed_index"])
        instance = int(call["task_index_mod50"])
        assert int(call["environment_seed"]) == 520_000 + task_index * 10_000 + instance
        assert call["split"] == "test"
        assert call["branch_bank_builder"] is build_diverse_action_branch_bank


@pytest.mark.parametrize(
    "tamper",
    ("task", "episode", "split", "hazard", "action"),
)
def test_test_conformance_rejects_identity_hazard_and_action_drift(
    tamper: str,
    contract: dict,
    test_arrays: dict[str, np.ndarray],
) -> None:
    arrays = dict(test_arrays)
    if tamper == "task":
        arrays["task_index"] = test_arrays["task_index"].copy()
        arrays["task_index"][arrays["task_index"] == 0] = 1
        match = "task indices"
    elif tamper == "episode":
        arrays["source_episode_id"] = test_arrays["source_episode_id"].copy()
        arrays["source_episode_id"][0] = "wrong:episode"
        match = "episode identity"
    elif tamper == "split":
        arrays["split"] = test_arrays["split"].astype("<U11")
        first_task = arrays["task_index"][0]
        first_instance = arrays["task_instance_index"][0]
        selected = (arrays["task_index"] == first_task) & (
            arrays["task_instance_index"] == first_instance
        )
        arrays["split"][selected] = "development"
        match = "development is excluded"
    elif tamper == "hazard":
        arrays["hazard_id"] = test_arrays["hazard_id"].copy()
        arrays["hazard_id"][0] += 1
        match = "hazard schedule"
    else:
        arrays["future_action"] = test_arrays["future_action"].copy()
        arrays["future_action"][0, 0, 0, 0] += 0.25
        match = "action bank"

    with pytest.raises(ValueError, match=match):
        validate_a3_confirmation_test_conformance(arrays, contract)


def test_merge_keeps_screen_train_adds_fresh_test_and_excludes_development(
    contract: dict,
    test_arrays: dict[str, np.ndarray],
) -> None:
    screen = _arrays_for_splits(contract, ("train", "development"))
    canonical = merge_a3_confirmation_canonical(screen, test_arrays, contract)

    assert tuple(canonical) == NPZ_KEYS
    split = canonical["split"].astype(str)
    assert set(split.tolist()) == {"train", "test"}
    assert not np.any(split == "development")
    screen_train = screen["split"].astype(str) == "train"
    train_count = int(np.count_nonzero(screen_train))
    for name in NPZ_KEYS:
        np.testing.assert_array_equal(
            canonical[name][:train_count],
            np.asarray(screen[name])[screen_train],
        )
    for task_index in range(len(TASKS)):
        for label, expected in (("train", 24), ("test", 12)):
            selected = (canonical["task_index"] == task_index) & (split == label)
            assert np.unique(canonical["task_instance_index"][selected]).size == expected
