from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_action_branch_a3_metrics import (
    CONFIRMATION_CHECKPOINT_STEP,
    CONFIRMATION_TRAINING_SEEDS,
    PROTOCOL,
    TASKS,
    summarize_a3_confirmation,
    validate_a3_scoring_contract,
)
from fcpg.lewm_action_branch_metrics import (
    FACTUAL_METHOD,
    NO_ACTION_METHOD,
    SHUFFLED_ACTION_METHOD,
    make_all_cyclic_branch_shifts,
)
from scripts import score_lewm_action_branch_a3_confirm as module
from scripts import score_lewm_action_branch_a3 as development_module


def _canonical_panel() -> dict[str, np.ndarray]:
    rows: list[tuple[int, int, str, int, str]] = []
    for task in range(4):
        instance = 0
        for split, episode_count in (("train", 24), ("test", 12)):
            for episode in range(episode_count):
                rows.append(
                    (
                        task,
                        instance,
                        f"task-{task}:{split}:{episode}",
                        episode + 3,
                        split,
                    )
                )
                instance += 1
    count = len(rows)
    branch = (np.arange(8, dtype=np.float32) - 3.5).reshape(8, 1, 1, 1)
    horizon = np.arange(1, 9, dtype=np.float32).reshape(1, 8, 1, 1)
    target = np.broadcast_to(branch * horizon, (count, 8, 8, 1, 1)).copy()
    return {
        "history_embedding": np.zeros((count, 3, 1, 1), dtype=np.float32),
        "history_action": np.zeros((count, 2, 1), dtype=np.float32),
        "center_action": np.zeros((count, 8, 1), dtype=np.float32),
        "future_action": np.zeros((count, 8, 8, 1), dtype=np.float32),
        "target_embedding": target,
        "task_index": np.asarray([row[0] for row in rows], dtype=np.int64),
        "task_instance_index": np.asarray(
            [row[1] for row in rows], dtype=np.int64
        ),
        "source_episode_id": np.asarray([row[2] for row in rows]),
        "hazard_id": np.asarray([row[3] for row in rows], dtype=np.int64),
        "split": np.asarray([row[4] for row in rows]),
    }


def _predictions(
    canonical: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    test_rows = np.flatnonzero(canonical["split"].astype(str) == "test")
    target = canonical["target_embedding"][test_rows]
    factual = np.broadcast_to(
        target[None], (len(CONFIRMATION_TRAINING_SEEDS), *target.shape)
    ).copy()
    no_action = np.zeros_like(factual)
    permutation = make_all_cyclic_branch_shifts(target.shape[0])
    shuffled = np.empty(
        (
            len(CONFIRMATION_TRAINING_SEEDS),
            target.shape[0],
            7,
            *target.shape[1:],
        ),
        dtype=np.float32,
    )
    bundle_rows = np.arange(target.shape[0], dtype=np.int64)[:, None]
    for seed in range(len(CONFIRMATION_TRAINING_SEEDS)):
        for shift in range(7):
            shuffled[seed, :, shift] = factual[seed][
                bundle_rows, permutation[:, shift]
            ]
    return {
        FACTUAL_METHOD: factual,
        NO_ACTION_METHOD: no_action,
        SHUFFLED_ACTION_METHOD: shuffled,
    }


def _prediction_payload(
    canonical: dict[str, np.ndarray],
    *,
    seed: int,
    step: int = CONFIRMATION_CHECKPOINT_STEP,
    evaluation_split: str = "test",
) -> dict[str, np.ndarray]:
    evaluation_rows = np.flatnonzero(
        canonical["split"].astype(str) == evaluation_split
    )
    target = canonical["target_embedding"][evaluation_rows]
    count = int(evaluation_rows.size)
    permutation = make_all_cyclic_branch_shifts(count)
    shuffled = np.empty((count, 7, *target.shape[1:]), dtype=np.float32)
    bundle_rows = np.arange(count, dtype=np.int64)[:, None]
    for shift in range(7):
        shuffled[:, shift] = target[bundle_rows, permutation[:, shift]]
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "protocol": np.asarray(PROTOCOL),
        "checkpoint_step": np.asarray(step, dtype=np.int64),
        "selection": np.asarray("fixed_latest"),
        "prediction_heads_add_sigreg": np.asarray(False),
        "training_seed": np.full(count, seed, dtype=np.int64),
        "seed": np.full(count, seed, dtype=np.int64),
        "sample_index": evaluation_rows,
        "split": canonical["split"][evaluation_rows],
        "task_index": canonical["task_index"][evaluation_rows],
        "task_instance_index": canonical["task_instance_index"][evaluation_rows],
        "source_episode_id": canonical["source_episode_id"][evaluation_rows],
        "hazard_id": canonical["hazard_id"][evaluation_rows],
        "history": canonical["history_embedding"][evaluation_rows],
        "history_action": canonical["history_action"][evaluation_rows],
        "center_action": canonical["center_action"][evaluation_rows],
        "future_action": canonical["future_action"][evaluation_rows],
        "target": target,
        "factual_prediction": target.copy(),
        "no_action_prediction": np.zeros_like(target),
        "shuffled_action_prediction": shuffled,
        "shuffle_permutation": permutation,
        "shuffled_action_donor_shifts": np.arange(1, 8, dtype=np.int64),
    }


def _write_predictions(
    directory: Path,
    canonical: dict[str, np.ndarray],
) -> list[Path]:
    paths: list[Path] = []
    for seed in CONFIRMATION_TRAINING_SEEDS:
        path = directory / f"seed-{seed}.npz"
        np.savez_compressed(path, **_prediction_payload(canonical, seed=seed))
        paths.append(path)
    return paths


def test_confirmation_summary_is_exact_train_test_h6_h8_go() -> None:
    canonical = _canonical_panel()
    summary = summarize_a3_confirmation(canonical, _predictions(canonical))

    assert summary["stage"] == "a3_10k_confirmation"
    assert summary["decision"] == "Confirmation-Go"
    assert summary["checkpoint_step"] == 10_000
    assert summary["primary_horizons"] == [6, 7, 8]
    assert summary["input_contains_test_rows"] is True
    assert summary["coverage"]["training_seeds"] == [29301, 29302, 29303]
    assert summary["shuffled_action_shifts"]["count"] == 7
    assert len(summary["shuffled_action_shifts"]["per_shift"]) == 7
    assert summary["gate"]["cross_replicating_seeds"] == [29301, 29302, 29303]
    assert all(summary["gate"]["checks"].values())
    for task in TASKS:
        coverage = summary["coverage"]["per_task"][task]
        assert coverage["train_physical_episode_count"] == 24
        assert coverage["test_physical_episode_count"] == 12


def test_confirmation_rejects_development_and_wrong_episode_count() -> None:
    canonical = _canonical_panel()
    with_development = {name: value.copy() for name, value in canonical.items()}
    with_development["split"] = with_development["split"].astype("<U16")
    first_test = int(np.flatnonzero(with_development["split"] == "test")[0])
    with_development["split"][first_test] = "development"
    with pytest.raises(ValueError, match="refuses canonical development rows"):
        summarize_a3_confirmation(with_development, _predictions(canonical))

    wrong_count = {name: value.copy() for name, value in canonical.items()}
    task_zero_test = np.flatnonzero(
        (wrong_count["task_index"] == 0) & (wrong_count["split"] == "test")
    )
    wrong_count["source_episode_id"][task_zero_test[0]] = wrong_count[
        "source_episode_id"
    ][task_zero_test[1]]
    wrong_count.pop("task_instance_index")
    with pytest.raises(ValueError, match="exactly 12 test physical episodes"):
        summarize_a3_confirmation(wrong_count, _predictions(canonical))


def test_cross_seed_task_replication_blocks_marginal_only_success() -> None:
    canonical = _canonical_panel()
    predictions = _predictions(canonical)
    test_rows = canonical["split"].astype(str) == "test"
    test_task = canonical["task_index"][test_rows]
    target = canonical["target_embedding"][test_rows]
    alphas = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 2.01, 2.01],
            [2.01, 2.01, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    permutation = make_all_cyclic_branch_shifts(target.shape[0])
    bundle_rows = np.arange(target.shape[0], dtype=np.int64)[:, None]
    for seed_position in range(3):
        predictions[FACTUAL_METHOD][seed_position] = (
            target * alphas[seed_position, test_task, None, None, None, None]
        )
        for shift in range(7):
            predictions[SHUFFLED_ACTION_METHOD][seed_position, :, shift] = (
                predictions[FACTUAL_METHOD][seed_position][
                    bundle_rows, permutation[:, shift]
                ]
            )

    summary = summarize_a3_confirmation(canonical, predictions)

    assert summary["decision"] == "Confirmation-No-Go"
    checks = summary["gate"]["checks"]
    assert checks["jointly_positive_tasks"]
    assert checks["jointly_positive_seeds"]
    assert checks["late_effect_vs_no_action"]
    assert checks["late_effect_vs_shuffled_action"]
    assert checks["late_effect_vs_every_shuffled_shift"]
    assert not checks["cross_seed_task_replication"]
    assert summary["gate"]["cross_replicating_seeds"] == [29301]
    assert summary["gate"]["cross_replicating_seed_task_counts"] == {
        "29301": 4,
        "29302": 2,
        "29303": 2,
    }


def test_loader_rejects_forged_shuffled_predictions(tmp_path: Path) -> None:
    canonical = _canonical_panel()
    payload = _prediction_payload(canonical, seed=CONFIRMATION_TRAINING_SEEDS[0])
    payload["shuffled_action_prediction"] = np.full_like(
        payload["shuffled_action_prediction"], 1.0e6
    )
    path = tmp_path / "forged.npz"
    np.savez_compressed(path, **payload)
    test_rows = np.flatnonzero(canonical["split"].astype(str) == "test")

    with pytest.raises(ValueError, match="does not equal factual_prediction"):
        module._aligned_file_predictions(
            path,
            canonical,
            canonical_test_rows=test_rows,
            expected_seed=CONFIRMATION_TRAINING_SEEDS[0],
        )


def test_development_loader_also_rejects_forged_shuffled_predictions(
    tmp_path: Path,
) -> None:
    canonical = {name: value.copy() for name, value in _canonical_panel().items()}
    canonical["split"] = canonical["split"].astype("<U16")
    canonical["split"][canonical["split"] == "test"] = "development"
    payload = _prediction_payload(
        canonical,
        seed=29_201,
        step=2_000,
        evaluation_split="development",
    )
    payload["shuffled_action_prediction"] = np.full_like(
        payload["shuffled_action_prediction"], 1.0e6
    )
    path = tmp_path / "forged-development.npz"
    np.savez_compressed(path, **payload)
    development_rows = np.flatnonzero(canonical["split"] == "development")

    with pytest.raises(ValueError, match="does not equal factual_prediction"):
        development_module._aligned_file_predictions(
            path,
            canonical,
            canonical_development_rows=development_rows,
            expected_seed=29_201,
        )


def test_prediction_headers_are_fixed_10k_three_seed_test_only(
    tmp_path: Path,
) -> None:
    canonical = _canonical_panel()
    paths = _write_predictions(tmp_path, canonical)
    assert tuple(sorted(module._validate_prediction_headers(paths))) == (
        CONFIRMATION_TRAINING_SEEDS
    )

    payload = _prediction_payload(
        canonical,
        seed=CONFIRMATION_TRAINING_SEEDS[0],
        step=9_999,
    )
    np.savez_compressed(paths[0], **payload)
    with pytest.raises(ValueError, match="step 10000"):
        module._validate_prediction_headers(paths)

    payload["checkpoint_step"] = np.asarray(10_000, dtype=np.int64)
    payload["split"] = np.full(payload["split"].shape, "development")
    np.savez_compressed(paths[0], **payload)
    with pytest.raises(ValueError, match="refuses prediction development rows"):
        module._aligned_file_predictions(
            paths[0],
            canonical,
            canonical_test_rows=np.flatnonzero(canonical["split"] == "test"),
            expected_seed=CONFIRMATION_TRAINING_SEEDS[0],
        )


def test_loader_aligns_test_rows_by_canonical_sample_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = _canonical_panel()
    paths: list[Path] = []
    for seed in CONFIRMATION_TRAINING_SEEDS:
        payload = _prediction_payload(canonical, seed=seed)
        count = int(payload["sample_index"].size)
        order = np.arange(count - 1, -1, -1)
        for name, value in tuple(payload.items()):
            if value.shape[:1] == (count,):
                payload[name] = value[order]
        path = tmp_path / f"seed-{seed}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    monkeypatch.setattr(module, "load_a1_npz", lambda _: canonical)

    loaded, predictions = module.load_a3_confirmation_prediction_panel(
        tmp_path / "canonical.npz", paths
    )

    test_rows = loaded["split"].astype(str) == "test"
    np.testing.assert_array_equal(
        predictions[FACTUAL_METHOD][0], loaded["target_embedding"][test_rows]
    )


def test_checked_in_gate_contract_and_conformance_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = json.loads(module.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    validate_a3_scoring_contract(contract, phase="confirmation")
    drifted = deepcopy(contract)
    drifted["confirmation_after_development_go"][
        "minimum_cross_replicating_seeds"
    ] = 1
    with pytest.raises(ValueError, match="Gate contract drifted"):
        validate_a3_scoring_contract(drifted, phase="confirmation")

    observed: list[tuple[dict[str, np.ndarray], dict]] = []
    monkeypatch.setattr(
        module,
        "validate_a3_confirmation_canonical_conformance",
        lambda arrays, value: observed.append((arrays, value)),
    )
    module._validate_formal_episode_coverage(_canonical_panel())
    assert len(observed) == 1
    assert observed[0][1]["protocol"] == PROTOCOL


def test_development_gate_contract_and_conformance_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = json.loads(module.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    validate_a3_scoring_contract(contract, phase="development")
    canonical = {name: value.copy() for name, value in _canonical_panel().items()}
    canonical["split"] = canonical["split"].astype("<U16")
    canonical["split"][canonical["split"] == "test"] = "development"
    observed: list[tuple[dict[str, np.ndarray], dict]] = []
    monkeypatch.setattr(
        development_module,
        "validate_a3_cache_conformance",
        lambda arrays, value: observed.append((arrays, value)),
    )

    development_module._validate_formal_episode_coverage(canonical)

    assert len(observed) == 1
    assert observed[0][1]["development_gate"]["minimum_relative_improvement"] == 0.05


def test_cli_writes_confirmation_json_and_markdown_without_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = _canonical_panel()
    paths = _write_predictions(tmp_path, canonical)
    monkeypatch.setattr(module, "load_a1_npz", lambda _: canonical)
    monkeypatch.setattr(
        module,
        "validate_a3_confirmation_canonical_conformance",
        lambda _arrays, _contract: None,
    )
    output = tmp_path / "score"

    assert (
        module.main(
            [
                "--canonical",
                str(tmp_path / "canonical.npz"),
                "--predictions",
                *(str(path) for path in paths),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    summary_text = (output / "summary.json").read_text(encoding="utf-8")
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["decision"] == "Confirmation-Go"
    assert summary["coverage"]["training_seeds"] == [29301, 29302, 29303]
    assert "10k confirmation" in markdown
    assert "Wrong-action donor shift" in markdown
    assert "sha256" not in (summary_text + markdown).lower()
    assert "provenance" not in (summary_text + markdown).lower()
