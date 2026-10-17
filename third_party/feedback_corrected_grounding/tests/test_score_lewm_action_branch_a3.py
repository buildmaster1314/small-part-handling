from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_action_branch_a3_metrics import PROTOCOL, TRAINING_SEEDS
from fcpg.lewm_action_branch_metrics import make_all_cyclic_branch_shifts
from scripts import score_lewm_action_branch_a3 as module


def _canonical_arrays() -> dict[str, np.ndarray]:
    rows = [
        (task, instance, f"task-{task}:{split}", 10, split)
        for task in range(4)
        for instance, split in enumerate(("train", "development"))
    ]
    count = len(rows)
    branch = (np.arange(8, dtype=np.float32) - 3.5).reshape(8, 1, 1, 1)
    horizon = np.arange(1, 9, dtype=np.float32).reshape(1, 8, 1, 1)
    base = branch * horizon
    target = np.broadcast_to(base, (8, 8, 8, 3, 192)).copy()
    return {
        "history_embedding": np.zeros((count, 3, 3, 192), dtype=np.float32),
        "history_action": np.zeros((count, 2, 20), dtype=np.float32),
        "center_action": np.zeros((count, 8, 20), dtype=np.float32),
        "future_action": np.zeros((count, 8, 8, 20), dtype=np.float32),
        "target_embedding": target,
        "task_index": np.asarray([row[0] for row in rows], dtype=np.int64),
        "task_instance_index": np.asarray(
            [row[1] for row in rows], dtype=np.int64
        ),
        "source_episode_id": np.asarray([row[2] for row in rows]),
        "hazard_id": np.asarray([row[3] for row in rows], dtype=np.int64),
        "split": np.asarray([row[4] for row in rows]),
    }


def _write_canonical(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _prediction_payload(
    canonical: dict[str, np.ndarray],
    *,
    seed: int,
    step: int = 2000,
) -> dict[str, np.ndarray]:
    development_rows = np.flatnonzero(
        canonical["split"].astype(str) == "development"
    )
    target = canonical["target_embedding"][development_rows]
    count = development_rows.size
    permutations = make_all_cyclic_branch_shifts(count)
    shuffled = np.empty((count, 7, *target.shape[1:]), dtype=np.float32)
    for row in range(count):
        for shift in range(7):
            shuffled[row, shift] = target[row, permutations[row, shift]]
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "protocol": np.asarray(PROTOCOL),
        "checkpoint_step": np.asarray(step, dtype=np.int64),
        "selection": np.asarray("fixed_latest"),
        "prediction_heads_add_sigreg": np.asarray(False),
        "training_seed": np.full(count, seed, dtype=np.int64),
        "seed": np.full(count, seed, dtype=np.int64),
        "sample_index": development_rows,
        "split": canonical["split"][development_rows],
        "task_index": canonical["task_index"][development_rows],
        "task_instance_index": canonical["task_instance_index"][development_rows],
        "source_episode_id": canonical["source_episode_id"][development_rows],
        "hazard_id": canonical["hazard_id"][development_rows],
        "history": canonical["history_embedding"][development_rows],
        "history_action": canonical["history_action"][development_rows],
        "center_action": canonical["center_action"][development_rows],
        "future_action": canonical["future_action"][development_rows],
        "target": target,
        "factual_prediction": target.copy(),
        "no_action_prediction": np.zeros_like(target),
        "shuffled_action_prediction": shuffled,
        "shuffle_permutation": permutations,
        "shuffled_action_donor_shifts": np.arange(1, 8, dtype=np.int64),
        # Deliberately global and wrong; A3 must recompute per-task moments.
        "train_mean": np.full((3, 192), 999.0),
        "train_scale": np.ones((3, 192)),
        "train_moment_count": np.asarray(1, dtype=np.int64),
    }


def _write_prediction(
    path: Path,
    canonical: dict[str, np.ndarray],
    *,
    seed: int,
    step: int = 2000,
) -> None:
    np.savez_compressed(
        path, **_prediction_payload(canonical, seed=seed, step=step)
    )


def test_prediction_headers_require_three_fixed_a3_endpoints(tmp_path: Path) -> None:
    canonical = _canonical_arrays()
    paths: list[Path] = []
    for seed in TRAINING_SEEDS:
        path = tmp_path / f"seed-{seed}.npz"
        _write_prediction(path, canonical, seed=seed)
        paths.append(path)
    observed = module._validate_prediction_headers(paths)
    assert tuple(sorted(observed)) == TRAINING_SEEDS

    _write_prediction(paths[0], canonical, seed=TRAINING_SEEDS[0], step=1999)
    with pytest.raises(ValueError, match="step 2000"):
        module._validate_prediction_headers(paths)


def test_loader_aligns_by_canonical_sample_index(tmp_path: Path) -> None:
    canonical = _canonical_arrays()
    canonical_path = tmp_path / "canonical.npz"
    _write_canonical(canonical_path, canonical)
    paths: list[Path] = []
    for seed in TRAINING_SEEDS:
        payload = _prediction_payload(canonical, seed=seed)
        order = np.asarray([3, 1, 0, 2], dtype=np.int64)
        row_count = payload["sample_index"].size
        for name, value in tuple(payload.items()):
            if value.shape[:1] == (row_count,):
                payload[name] = value[order]
        path = tmp_path / f"seed-{seed}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)

    loaded, predictions = module.load_a3_prediction_panel(canonical_path, paths)
    development = loaded["split"].astype(str) == "development"
    np.testing.assert_array_equal(
        predictions["factual"][0], loaded["target_embedding"][development]
    )


def test_loader_rejects_prediction_test_rows(tmp_path: Path) -> None:
    canonical = _canonical_arrays()
    canonical_path = tmp_path / "canonical.npz"
    _write_canonical(canonical_path, canonical)
    paths: list[Path] = []
    for seed in TRAINING_SEEDS:
        payload = _prediction_payload(canonical, seed=seed)
        if seed == TRAINING_SEEDS[0]:
            payload["split"] = np.full(payload["split"].shape, "test")
        path = tmp_path / f"seed-{seed}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)

    with pytest.raises(ValueError, match="refuses prediction test rows"):
        module.load_a3_prediction_panel(canonical_path, paths)


def test_cli_writes_minimal_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = _canonical_arrays()
    canonical_path = tmp_path / "canonical.npz"
    _write_canonical(canonical_path, canonical)
    paths: list[Path] = []
    for seed in TRAINING_SEEDS:
        path = tmp_path / f"seed-{seed}.npz"
        _write_prediction(path, canonical, seed=seed)
        paths.append(path)
    output = tmp_path / "score"
    monkeypatch.setattr(module, "_validate_formal_episode_coverage", lambda _: None)

    result = module.main(
        [
            "--canonical",
            str(canonical_path),
            "--predictions",
            *(str(path) for path in paths),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert summary["protocol"] == PROTOCOL
    assert summary["decision"] == "Development-Go"
    assert summary["coverage"]["training_seeds"] == list(TRAINING_SEEDS)
    assert summary["whitening"]["source"] == "canonical_train_rows"
    assert summary["whitening"]["per_task"] is True
    assert "sha256" not in json.dumps(summary).lower()
    assert "A3 matched action-branch" in markdown
    assert "Wrong-action donor shift" in markdown


def test_formal_coverage_rejects_tiny_fixture() -> None:
    with pytest.raises(ValueError, match="episode coverage"):
        module._validate_formal_episode_coverage(_canonical_arrays())
