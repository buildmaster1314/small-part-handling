from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_a4_storage import ACTION_DIM, LATENT_DIM, STATE_SLOT_COUNT
from fcpg.lewm_a4_checkpoint import A4_CHECKPOINT_TYPE
from fcpg.lewm_action_branch_a4 import (
    BRANCH_COUNT,
    CONFIRMATION_TRAINING_SEEDS,
    DEVELOPMENT_TRAINING_SEEDS,
    HISTORY_STEPS,
    HORIZON,
    PROTOCOL,
    TASKS,
    assign_a4_splits,
)
from fcpg.lewm_action_branch_a4_confirmation_metrics import (
    _summarize_confirmation_panel,
    validate_a4_confirmation_caches,
    validate_a4_development_go_authorization,
)
from fcpg.lewm_action_branch_a4_metrics import (
    FACTUAL_METHOD,
    NO_ACTION_METHOD,
    WRONG_ACTION_METHOD,
    _secondary_predictions,
)
from scripts import score_lewm_action_branch_a4_confirmation as cli


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"


def _contract() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _development_go(*, train_episode_count: int = 18) -> dict:
    comparison = {
        "relative_improvement": 0.25,
        "paired_ci": {"lower": 0.1, "upper": 0.3},
    }
    shifts = [
        {"donor_shift": shift, **deepcopy(comparison)} for shift in range(1, 8)
    ]
    checks = {
        "minimum_effect_vs_no_action": True,
        "minimum_effect_vs_every_wrong_action_shift": True,
        "paired_ci_lower_exceeds_zero": True,
        "minimum_positive_tasks": True,
        "minimum_positive_seeds": True,
        "raw_factual_not_worse_than_no_action": True,
        "raw_factual_better_than_hold_last": True,
        "raw_factual_better_than_constant_velocity": True,
    }
    raw = {
        FACTUAL_METHOD: {"task_macro_mse": 0.2},
        NO_ACTION_METHOD: {"task_macro_mse": 0.4},
        "hold_last": {"task_macro_mse": 0.5},
        "latent_constant_velocity": {"task_macro_mse": 0.6},
    }
    return {
        "protocol": PROTOCOL,
        "stage": "a4_2k_formal_development",
        "mode": "formal",
        "decision": "Development-Go",
        "passed": True,
        "development_go": True,
        "reserved_test_collection_authorized": True,
        "input_contains_test_rows": False,
        "checkpoint_step": 2_000,
        "primary_horizons": [6, 7, 8],
        "coverage": {
            "tasks": list(TASKS),
            "training_seeds": list(DEVELOPMENT_TRAINING_SEEDS),
            "development_bundle_count": 84,
        },
        "whitening": {
            "source": "episode_equal_train_rows",
            "per_task": True,
            "fit_horizons": [6, 7, 8],
            "tasks": {
                task: {
                    "train_physical_episode_count": train_episode_count,
                    "floored_dimensions": 0,
                }
                for task in TASKS
            },
        },
        "primary": {
            "metric": "h6_h8_within_bundle_branch_centered_whitened_mse",
            "comparisons": {NO_ACTION_METHOD: deepcopy(comparison)},
            "wrong_action_shifts": {"count": 7, "per_shift": shifts},
        },
        "raw_h6_h8": {"methods": raw},
        "gate": {
            "checks": checks,
            "failed_checks": [],
            "jointly_positive_tasks": list(TASKS),
            "jointly_positive_seeds": list(DEVELOPMENT_TRAINING_SEEDS),
            "passed": True,
            "development_go": True,
        },
    }


def _cache(
    *,
    train_episodes: int = 0,
    development_episodes: int = 0,
    test_episodes: int = 0,
) -> dict[str, np.ndarray]:
    contract = _contract()
    assignments = assign_a4_splits(TASKS, contract)
    rows: list[tuple[int, int, str, str]] = []
    counts = {
        "train": int(train_episodes),
        "development": int(development_episodes),
        "test": int(test_episodes),
    }
    for task_index, task_name in enumerate(TASKS):
        for split_name, requested in counts.items():
            instances = sorted(
                instance
                for (task, instance), assigned in assignments.items()
                if task == task_name and assigned == split_name
            )
            for instance in instances[:requested]:
                rows.append(
                    (
                        task_index,
                        instance,
                        f"{task_name}:task-instance-{instance:02d}",
                        split_name,
                    )
                )
    if not rows:
        raise ValueError("test fixture cache must contain at least one row")
    count = len(rows)
    task = np.asarray([row[0] for row in rows], dtype=np.int64)
    split = np.asarray([row[3] for row in rows])
    target = np.zeros(
        (count, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM),
        dtype=np.float16,
    )
    branch = np.arange(BRANCH_COUNT, dtype=np.float16) - np.float16(3.5)
    horizon = np.arange(1, HORIZON + 1, dtype=np.float16)
    target[..., 0, 0] = branch[None, :, None] * horizon[None, None, :]
    return {
        "proposal_id": np.asarray(
            [
                f"proposal-{row[3]}-{row[0]}-{row[1]}-{index}"
                for index, row in enumerate(rows)
            ]
        ),
        "task_class_index": task,
        "task_name": np.asarray([TASKS[index] for index in task]),
        "task_instance_index": np.asarray([row[1] for row in rows], dtype=np.int64),
        "source_episode_id": np.asarray([row[2] for row in rows]),
        "split": split,
        "history_state": np.zeros(
            (count, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float16
        ),
        "history_action": np.zeros(
            (count, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float16
        ),
        "center_action": np.zeros((count, HORIZON, ACTION_DIM), dtype=np.float16),
        "future_action": np.zeros(
            (count, BRANCH_COUNT, HORIZON, ACTION_DIM), dtype=np.float16
        ),
        "target_state": target,
        "branch_semantic_index": np.broadcast_to(
            np.arange(BRANCH_COUNT, dtype=np.int64), (count, BRANCH_COUNT)
        ).copy(),
    }


def _predictions(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    assert set(cache["split"].astype(str).tolist()) == {"test"}
    target = cache["target_state"]
    factual = np.broadcast_to(
        target[None], (len(CONFIRMATION_TRAINING_SEEDS), *target.shape)
    ).copy()
    no_action = np.zeros_like(factual)
    wrong = np.empty(
        (
            len(CONFIRMATION_TRAINING_SEEDS),
            target.shape[0],
            BRANCH_COUNT - 1,
            *target.shape[1:],
        ),
        dtype=target.dtype,
    )
    slots = np.arange(BRANCH_COUNT)
    for shift in range(1, BRANCH_COUNT):
        wrong[:, :, shift - 1] = factual[:, :, (slots + shift) % BRANCH_COUNT]
    return {
        FACTUAL_METHOD: factual,
        NO_ACTION_METHOD: no_action,
        WRONG_ACTION_METHOD: wrong,
    }


def _prediction_archive(cache: dict[str, np.ndarray], *, seed: int) -> dict[str, np.ndarray]:
    assert set(cache["split"].astype(str).tolist()) == {"test"}
    test_rows = np.arange(len(cache["split"]), dtype=np.int64)
    target = cache["target_state"][test_rows]
    factual = target.copy()
    no_action = np.zeros_like(factual)
    slots = np.arange(BRANCH_COUNT)
    wrong = np.stack(
        [factual[:, (slots + shift) % BRANCH_COUNT] for shift in range(1, BRANCH_COUNT)],
        axis=1,
    )
    permutations = np.broadcast_to(
        np.stack([(slots + shift) % BRANCH_COUNT for shift in range(1, BRANCH_COUNT)]),
        (test_rows.size, BRANCH_COUNT - 1, BRANCH_COUNT),
    ).copy()
    baseline = _secondary_predictions(cache["history_state"][test_rows])
    count = int(test_rows.size)
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "protocol": np.asarray(PROTOCOL),
        "checkpoint_step": np.asarray(10_000, dtype=np.int64),
        "selection": np.asarray("fixed_latest"),
        "anchor_encoder_sigreg": np.asarray(0.02),
        "prediction_heads_add_sigreg": np.asarray(False),
        "evaluation_split": np.asarray("test"),
        "primary_horizons": np.asarray([6, 7, 8], dtype=np.int64),
        "checkpoint_type": np.asarray(A4_CHECKPOINT_TYPE),
        "checkpoint_seed": np.asarray(seed, dtype=np.int64),
        "checkpoint_path": np.asarray(
            f"/tmp/a4-seed-{seed}/lewm_action_branch_a4_latest.pt"
        ),
        "checkpoint_sha256": np.asarray(f"{seed % 16:x}" * 64),
        "training_seed": np.full(count, seed, dtype=np.int64),
        "seed": np.full(count, seed, dtype=np.int64),
        "sample_index": test_rows,
        "proposal_id": cache["proposal_id"][test_rows],
        "task_class_index": cache["task_class_index"][test_rows],
        "task_name": cache["task_name"][test_rows],
        "task_instance_index": cache["task_instance_index"][test_rows],
        "source_episode_id": cache["source_episode_id"][test_rows],
        "split": cache["split"][test_rows],
        "history_state": cache["history_state"][test_rows],
        "history_action": cache["history_action"][test_rows],
        "center_action": cache["center_action"][test_rows],
        "future_action": cache["future_action"][test_rows],
        "target_state": target,
        "branch_semantic_index": cache["branch_semantic_index"][test_rows],
        "factual_prediction": factual,
        "no_action_prediction": no_action,
        "shuffled_action_prediction": wrong,
        "hold_last_prediction": baseline["hold_last"].astype(np.float16),
        "constant_velocity_prediction": baseline[
            "latent_constant_velocity"
        ].astype(np.float16),
        "shuffle_permutation": permutations,
        "shuffled_action_donor_shifts": np.arange(1, 8, dtype=np.int64),
    }


def test_exact_development_go_authorization_passes_and_forgery_fails() -> None:
    summary = _development_go()
    counts = validate_a4_development_go_authorization(summary, contract=_contract())
    assert counts == {task: 18 for task in TASKS}

    forged = deepcopy(summary)
    forged["gate"]["checks"]["minimum_positive_tasks"] = False
    with pytest.raises(PermissionError, match="checks are incomplete"):
        validate_a4_development_go_authorization(forged, contract=_contract())

    exposed = deepcopy(summary)
    exposed["input_contains_test_rows"] = True
    with pytest.raises(PermissionError, match="exact formal"):
        validate_a4_development_go_authorization(exposed, contract=_contract())


def test_confirmation_cache_requires_authorized_train_and_twelve_test_episodes() -> None:
    train_cache = _cache(train_episodes=18, development_episodes=2)
    test_cache = _cache(test_episodes=12)
    coverage = validate_a4_confirmation_caches(
        train_cache,
        test_cache,
        authorized_train_episode_counts={task: 18 for task in TASKS},
        contract=_contract(),
    )
    assert all(row["test_physical_episode_count"] == 12 for row in coverage.values())
    assert all(
        row["development_physical_episode_count"] == 2
        for row in coverage.values()
    )

    too_small = _cache(test_episodes=1)
    with pytest.raises(ValueError, match="must be exactly 12"):
        validate_a4_confirmation_caches(
            train_cache,
            too_small,
            authorized_train_episode_counts={task: 18 for task in TASKS},
            contract=_contract(),
        )


def test_confirmation_caches_reject_relabelled_development_and_bad_identity() -> None:
    train_cache = _cache(train_episodes=18, development_episodes=1)
    test_cache = _cache(test_episodes=12)

    relabelled = deepcopy(test_cache)
    relabelled["task_instance_index"] = relabelled["task_instance_index"].copy()
    development_instance = next(
        instance
        for (task, instance), split in assign_a4_splits(TASKS, _contract()).items()
        if task == TASKS[0] and split == "development"
    )
    relabelled["task_instance_index"][0] = development_instance
    relabelled["source_episode_id"] = relabelled["source_episode_id"].copy()
    relabelled["source_episode_id"][0] = (
        f"{TASKS[0]}:task-instance-{development_instance:02d}"
    )
    with pytest.raises(PermissionError, match="frozen assignment"):
        validate_a4_confirmation_caches(
            train_cache,
            relabelled,
            authorized_train_episode_counts={task: 18 for task in TASKS},
            contract=_contract(),
        )

    wrong_episode = deepcopy(test_cache)
    wrong_episode["source_episode_id"] = wrong_episode["source_episode_id"].copy()
    wrong_episode["source_episode_id"][0] = "wrong-environment-identity"
    with pytest.raises(ValueError, match="task-instance identity"):
        validate_a4_confirmation_caches(
            train_cache,
            wrong_episode,
            authorized_train_episode_counts={task: 18 for task in TASKS},
            contract=_contract(),
        )


def test_confirmation_statistics_apply_frozen_seven_task_three_seed_gate() -> None:
    train_cache = _cache(train_episodes=1)
    test_cache = _cache(test_episodes=1)
    gate = _contract()["confirmation_after_development_go"]
    coverage = {
        task: {
            "train_physical_episode_count": 1,
            "test_physical_episode_count": 1,
            "test_bundle_count": 1,
        }
        for task in TASKS
    }
    summary = _summarize_confirmation_panel(
        train_cache,
        test_cache,
        _predictions(test_cache),
        gate=gate,
        training_seeds=CONFIRMATION_TRAINING_SEEDS,
        coverage=coverage,
    )

    assert summary["decision"] == "Confirmation-Go"
    assert summary["checkpoint_step"] == 10_000
    assert summary["selection"] == "fixed_latest"
    assert not summary["checkpoint_selection_performed"]
    assert summary["primary"]["wrong_action_shifts"]["count"] == 7
    assert summary["gate"]["minimum_positive_tasks"] == 6
    assert summary["gate"]["minimum_positive_seeds"] == 2
    assert summary["claim_boundary"]["positive_confirmation_claim_authorized"]
    assert "policy rollout" in " ".join(
        summary["claim_boundary"]["does_not_establish"]
    )


def test_confirmation_whitening_ignores_development_rows() -> None:
    train_cache = _cache(train_episodes=2, development_episodes=1)
    test_cache = _cache(test_episodes=1)
    gate = _contract()["confirmation_after_development_go"]
    coverage = {
        task: {
            "train_physical_episode_count": 2,
            "development_physical_episode_count": 1,
            "test_physical_episode_count": 1,
            "test_bundle_count": 1,
        }
        for task in TASKS
    }
    reference = _summarize_confirmation_panel(
        train_cache,
        test_cache,
        _predictions(test_cache),
        gate=gate,
        training_seeds=CONFIRMATION_TRAINING_SEEDS,
        coverage=coverage,
    )
    changed = deepcopy(train_cache)
    development = changed["split"].astype(str) == "development"
    changed["target_state"] = changed["target_state"].copy()
    changed["target_state"][development] = np.float16(2_000.0)
    observed = _summarize_confirmation_panel(
        changed,
        test_cache,
        _predictions(test_cache),
        gate=gate,
        training_seeds=CONFIRMATION_TRAINING_SEEDS,
        coverage=coverage,
    )
    assert observed["whitening"] == reference["whitening"]
    assert observed["primary"] == reference["primary"]


def test_archive_loader_enforces_fixed_10k_seed_test_and_wrong_action() -> None:
    cache = _cache(test_episodes=1)
    arrays = _prediction_archive(cache, seed=CONFIRMATION_TRAINING_SEEDS[0])
    loaded = cli._aligned_prediction_archive(
        arrays, cache, expected_seed=CONFIRMATION_TRAINING_SEEDS[0]
    )
    assert loaded[FACTUAL_METHOD].shape[0] == len(TASKS)

    wrong_step = deepcopy(arrays)
    wrong_step["checkpoint_step"] = np.asarray(9_999)
    with pytest.raises(ValueError, match="step 10000"):
        cli._aligned_prediction_archive(
            wrong_step, cache, expected_seed=CONFIRMATION_TRAINING_SEEDS[0]
        )

    selected = deepcopy(arrays)
    selected["selection"] = np.asarray("best")
    with pytest.raises(ValueError, match="forbids checkpoint selection"):
        cli._aligned_prediction_archive(
            selected, cache, expected_seed=CONFIRMATION_TRAINING_SEEDS[0]
        )

    wrong_horizons = deepcopy(arrays)
    wrong_horizons["primary_horizons"] = np.asarray([5, 6, 7])
    with pytest.raises(ValueError, match="H6-H8"):
        cli._aligned_prediction_archive(
            wrong_horizons, cache, expected_seed=CONFIRMATION_TRAINING_SEEDS[0]
        )

    malformed_checkpoint = deepcopy(arrays)
    malformed_checkpoint["checkpoint_sha256"] = np.asarray("NOT-A-SHA")
    with pytest.raises(ValueError, match="64 lowercase hex"):
        cli._aligned_prediction_archive(
            malformed_checkpoint,
            cache,
            expected_seed=CONFIRMATION_TRAINING_SEEDS[0],
        )

    short_seed = deepcopy(arrays)
    short_seed["seed"] = short_seed["seed"][:1]
    with pytest.raises(ValueError, match="one entry per test row"):
        cli._aligned_prediction_archive(
            short_seed, cache, expected_seed=CONFIRMATION_TRAINING_SEEDS[0]
        )

    development = deepcopy(arrays)
    development["split"] = np.full(development["split"].shape, "development")
    with pytest.raises(PermissionError, match="development rows"):
        cli._aligned_prediction_archive(
            development, cache, expected_seed=CONFIRMATION_TRAINING_SEEDS[0]
        )

    forged = deepcopy(arrays)
    forged["shuffled_action_prediction"] = np.zeros_like(
        forged["shuffled_action_prediction"]
    )
    with pytest.raises(ValueError, match="cyclic replay"):
        cli._aligned_prediction_archive(
            forged, cache, expected_seed=CONFIRMATION_TRAINING_SEEDS[0]
        )

    broadcast_baseline = deepcopy(arrays)
    broadcast_baseline["hold_last_prediction"] = broadcast_baseline[
        "hold_last_prediction"
    ][:, :1]
    with pytest.raises(ValueError, match="differs from causal baseline"):
        cli._aligned_prediction_archive(
            broadcast_baseline,
            cache,
            expected_seed=CONFIRMATION_TRAINING_SEEDS[0],
        )


def test_archive_loader_replays_float32_constant_velocity_receipt() -> None:
    cache = _cache(test_episodes=1)
    rng = np.random.default_rng(31_401)
    cache["history_state"] = rng.normal(
        size=cache["history_state"].shape
    ).astype(np.float32)
    arrays = _prediction_archive(cache, seed=CONFIRMATION_TRAINING_SEEDS[0])

    history = cache["history_state"].astype(np.float32, copy=False)
    last = history[:, -1]
    velocity = last - history[:, -2]
    factors = np.arange(1, HORIZON + 1, dtype=np.float32).reshape(
        1, 1, HORIZON, 1, 1
    )
    shape = (history.shape[0], BRANCH_COUNT, HORIZON, *last.shape[1:])
    arrays["hold_last_prediction"] = np.broadcast_to(
        last[:, None, None], shape
    ).copy()
    arrays["constant_velocity_prediction"] = np.broadcast_to(
        last[:, None, None] + factors * velocity[:, None, None], shape
    ).copy()

    loaded = cli._aligned_prediction_archive(
        arrays, cache, expected_seed=CONFIRMATION_TRAINING_SEEDS[0]
    )
    assert loaded[FACTUAL_METHOD].shape[0] == len(TASKS)

    forged = deepcopy(arrays)
    forged["constant_velocity_prediction"] = forged[
        "constant_velocity_prediction"
    ].copy()
    forged["constant_velocity_prediction"][0, 0, 7, 0, 0] += np.float32(1e-4)
    with pytest.raises(ValueError, match="differs from causal baseline"):
        cli._aligned_prediction_archive(
            forged, cache, expected_seed=CONFIRMATION_TRAINING_SEEDS[0]
        )


def test_authorization_is_checked_before_cache_read(tmp_path: Path) -> None:
    invalid = _development_go()
    invalid["decision"] = "Development-No-Go"
    with pytest.raises(PermissionError, match="exact formal"):
        cli.load_a4_confirmation_prediction_panel(
            tmp_path / "must-not-be-opened.npz",
            tmp_path / "must-not-be-opened-test.npz",
            [],
            contract=_contract(),
            development_go_summary=invalid,
        )


def test_prediction_panel_requires_all_three_fresh_confirmation_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_cache = _cache(train_episodes=1)
    test_cache = _cache(test_episodes=1)
    payloads = {
        f"seed-{seed}.npz": _prediction_archive(test_cache, seed=seed)
        for seed in CONFIRMATION_TRAINING_SEEDS
    }
    train_cache_name = "train-cache.npz"
    test_cache_name = "test-cache.npz"

    def fake_load(path: str | Path, *, label: str) -> dict[str, np.ndarray]:
        del label
        name = Path(path).name
        if name == train_cache_name:
            return train_cache
        if name == test_cache_name:
            return test_cache
        return payloads[name]

    monkeypatch.setattr(cli, "_load_npz", fake_load)
    monkeypatch.setattr(
        cli,
        "validate_a4_confirmation_caches",
        lambda *_args, **_kwargs: {},
    )
    paths = [tmp_path / name for name in payloads]
    loaded_train, loaded_test, predictions = cli.load_a4_confirmation_prediction_panel(
        tmp_path / train_cache_name,
        tmp_path / test_cache_name,
        paths,
        contract=_contract(),
        development_go_summary=_development_go(),
    )
    assert loaded_train is train_cache
    assert loaded_test is test_cache
    assert predictions[FACTUAL_METHOD].shape[0] == 3

    duplicate_checkpoint_payloads = deepcopy(payloads)
    first = duplicate_checkpoint_payloads[f"seed-{CONFIRMATION_TRAINING_SEEDS[0]}.npz"]
    second = duplicate_checkpoint_payloads[f"seed-{CONFIRMATION_TRAINING_SEEDS[1]}.npz"]
    second["checkpoint_path"] = first["checkpoint_path"].copy()
    second["checkpoint_sha256"] = first["checkpoint_sha256"].copy()
    payloads.update(duplicate_checkpoint_payloads)
    with pytest.raises(ValueError, match="distinct checkpoint path"):
        cli.load_a4_confirmation_prediction_panel(
            tmp_path / train_cache_name,
            tmp_path / test_cache_name,
            paths,
            contract=_contract(),
            development_go_summary=_development_go(),
        )

    payloads.clear()
    payloads.update(
        {
            f"seed-{seed}.npz": _prediction_archive(test_cache, seed=seed)
            for seed in CONFIRMATION_TRAINING_SEEDS
        }
    )

    with pytest.raises(ValueError, match="duplicate A4 confirmation seed"):
        cli.load_a4_confirmation_prediction_panel(
            tmp_path / train_cache_name,
            tmp_path / test_cache_name,
            [paths[0], paths[0], paths[1]],
            contract=_contract(),
            development_go_summary=_development_go(),
        )


def test_cli_writes_confirmation_boundary_without_running_real_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_path = tmp_path / "development.json"
    development_path.write_text(json.dumps(_development_go()), encoding="utf-8")
    train_cache = _cache(train_episodes=1)
    test_cache = _cache(test_episodes=1)
    predictions = _predictions(test_cache)
    summary = {
        "decision": "Confirmation-No-Go",
        "claim_boundary": {
            "positive_confirmation_claim_authorized": False,
            "authorized_claim": None,
            "negative_result_statement": "Frozen confirmation failed.",
            "does_not_establish": ["policy rollout"],
        },
        "primary": {
            "comparisons": {
                NO_ACTION_METHOD: {
                    "relative_improvement": 0.0,
                    "positive_task_count": 0,
                    "positive_seed_count": 0,
                    "paired_ci": {"lower": -1.0},
                }
            },
            "wrong_action_shifts": {"per_shift": []},
        },
        "raw_h6_h8": {
            "methods": {
                method: {"task_macro_mse": 1.0}
                for method in (
                    FACTUAL_METHOD,
                    NO_ACTION_METHOD,
                    "hold_last",
                    "latent_constant_velocity",
                )
            }
        },
        "gate": {
            "jointly_positive_tasks": [],
            "jointly_positive_seeds": [],
            "failed_checks": ["minimum_effect_vs_no_action"],
        },
    }
    monkeypatch.setattr(
        cli,
        "load_a4_confirmation_prediction_panel",
        lambda *_args, **_kwargs: (train_cache, test_cache, predictions),
    )
    monkeypatch.setattr(
        cli,
        "validate_a4_development_go_summary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli,
        "_prevalidate_prediction_sources",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(cli, "summarize_a4_confirmation", lambda *_args, **_kwargs: summary)
    output = tmp_path / "out"
    assert cli.main(
        [
            "--config",
            str(CONFIG),
            "--development-go-summary",
            str(development_path),
            "--train-cache",
            str(tmp_path / "unused-train.npz"),
            "--test-cache",
            str(tmp_path / "unused-test.npz"),
            "--predictions",
            str(tmp_path / "unused-prediction.npz"),
            "--output",
            str(output),
        ]
    ) == 0
    rendered = (output / "summary.md").read_text(encoding="utf-8")
    assert "No-Go" in rendered
    assert "does not establish" in rendered
    assert json.loads((output / "summary.json").read_text())["decision"] == "Confirmation-No-Go"


def test_prediction_sources_bind_three_real_checkpoint_files_before_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _cache(test_episodes=1)
    payloads: dict[str, dict[str, np.ndarray]] = {}
    paths: list[Path] = []
    for seed in CONFIRMATION_TRAINING_SEEDS:
        checkpoint = tmp_path / f"checkpoint-{seed}.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode("ascii"))
        arrays = _prediction_archive(cache, seed=seed)
        arrays["checkpoint_path"] = np.asarray(str(checkpoint.resolve()))
        arrays["checkpoint_sha256"] = np.asarray(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )
        prediction = tmp_path / f"seed-{seed}.npz"
        paths.append(prediction)
        payloads[prediction.name] = arrays

    monkeypatch.setattr(
        cli,
        "_load_npz",
        lambda path, *, label: payloads[Path(path).name],
    )
    loaded = cli._prevalidate_prediction_sources(paths[::-1])
    assert tuple(sorted(loaded)) == CONFIRMATION_TRAINING_SEEDS

    first_seed = CONFIRMATION_TRAINING_SEEDS[0]
    Path(loaded[first_seed]["checkpoint_path"].item()).write_bytes(b"changed")
    with pytest.raises(PermissionError, match="content mismatch"):
        cli._prevalidate_prediction_sources(paths)


def test_existing_score_output_fails_before_any_artifact_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        cli,
        "_load_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("output collision must fail first")
        ),
    )
    with pytest.raises(FileExistsError):
        cli.main(
            [
                "--development-go-summary",
                str(tmp_path / "go.json"),
                "--train-cache",
                str(tmp_path / "train.npz"),
                "--test-cache",
                str(tmp_path / "test.npz"),
                "--predictions",
                str(tmp_path / "prediction.npz"),
                "--output",
                str(output),
            ]
        )
