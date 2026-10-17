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
from scripts import fit_lewm_action_branch_a4_qcenter_geometry_gate as gate


def _sources(tmp_path: Path, *, include_test: bool = False) -> tuple[Path, Path, list[Path]]:
    rows = []
    for task_index, task_name in enumerate(TASKS):
        episodes = ((0, "train"), (1, "train"), (2, "development"))
        if include_test:
            episodes += ((3, "test"),)
        for episode_index, split in episodes:
            rows.append((task_index, task_name, episode_index, split))
    count = len(rows)
    proposal_id = np.asarray([f"proposal-{index}" for index in range(count)])
    task_index = np.asarray([row[0] for row in rows], dtype=np.int64)
    task_name = np.asarray([row[1] for row in rows])
    task_instance = np.asarray([row[2] for row in rows], dtype=np.int64)
    split = np.asarray([row[3] for row in rows])
    episode = np.asarray([f"{row[1]}:{row[3]}:{row[2]}" for row in rows])
    history = np.zeros((count, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float32)
    target = np.zeros(
        (count, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float32
    )
    branch_semantic = np.broadcast_to(
        np.arange(BRANCH_COUNT, dtype=np.int64), (count, BRANCH_COUNT)
    ).copy()
    q = np.empty((count, HORIZON + 1, 2), dtype=np.float32)
    future_action = np.zeros((count, BRANCH_COUNT, HORIZON, ACTION_DIM), dtype=np.float32)
    for row, (task_id, _task, episode_id, _split) in enumerate(rows):
        q[row, 0] = (60.0 + task_id, 80.0 + episode_id)
        for horizon in range(HORIZON):
            dx = 0.3 * (horizon + 1) + 0.05 * task_id
            dy = -0.2 * (horizon + 1) + 0.02 * episode_id
            q[row, horizon + 1] = q[row, 0] + (dx, dy)
            for anchor in range(1, 4):
                target[row, 0, horizon, anchor, 0] = dx * anchor
                target[row, 0, horizon, anchor, 1] = dy * anchor
            future_action[row, 0, horizon, :2] = (dx, dy)
    cache = {
        "proposal_id": proposal_id,
        "task_class_index": task_index,
        "task_name": task_name,
        "task_instance_index": task_instance,
        "source_episode_id": episode,
        "split": split,
        "history_state": history,
        "history_action": np.zeros((count, HISTORY_STEPS - 1, ACTION_DIM), dtype=np.float32),
        "center_action": np.zeros((count, HORIZON, ACTION_DIM), dtype=np.float32),
        "future_action": future_action,
        "target_state": target,
        "branch_semantic_index": branch_semantic,
    }
    cache_path = tmp_path / "cache.npz"
    np.savez_compressed(cache_path, **cache)
    companion_path = tmp_path / "companion.npz"
    np.savez_compressed(
        companion_path,
        proposal_id=proposal_id,
        split=split,
        task_name=task_name,
        source_episode_id=episode,
        q_center_model_raw=q,
    )
    development = split == "development"
    prediction_paths = []
    for seed in DEVELOPMENT_TRAINING_SEEDS:
        factual = target[development]
        no_action = np.repeat(factual[:, :1], BRANCH_COUNT, axis=1)
        wrong = np.stack(
            [
                factual[:, (np.arange(BRANCH_COUNT) + shift) % BRANCH_COUNT]
                for shift in range(1, BRANCH_COUNT)
            ],
            axis=1,
        )
        path = tmp_path / f"prediction-{seed}.npz"
        np.savez_compressed(
            path,
            protocol=np.asarray(PROTOCOL),
            checkpoint_step=np.asarray(2_000, dtype=np.int64),
            training_seed=np.full(np.count_nonzero(development), seed, dtype=np.int64),
            proposal_id=proposal_id[development],
            split=split[development],
            history_state=history[development],
            future_action=future_action[development],
            target_state=target[development],
            factual_prediction=factual,
            no_action_prediction=no_action,
            shuffled_action_prediction=wrong,
        )
        prediction_paths.append(path)
    return cache_path, companion_path, prediction_paths


def test_main_feature_contract_has_no_time_or_action_channel() -> None:
    latent = np.ones((2, HORIZON, 3 * LATENT_DIM), dtype=np.float32)
    feature = gate.build_main_features(
        latent, np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
    )
    assert feature.shape == (2, HORIZON, gate.MAIN_FEATURE_DIM)
    assert np.array_equal(feature[:, 0], feature[:, -1])
    assert gate.MAIN_FEATURE_DIM == 3 * LATENT_DIM + 2 + len(TASKS)
    assert gate.DIRECT_ACTION_FEATURE_DIM == (
        3 * 3 * LATENT_DIM + 2 * ACTION_DIM
        + 2 + len(TASKS) + HORIZON + HORIZON * ACTION_DIM
    )


def test_geometry_gate_is_episode_disjoint_and_reuses_one_head(tmp_path: Path) -> None:
    cache, companion, predictions = _sources(tmp_path)
    summary, artifact = gate.run_geometry_gate(
        cache, companion, predictions,
        device_name="cpu", steps=120, batch_size=128, hidden_dim=32,
        learning_rate=2.0e-3, head_seeds=(17, 18, 19),
        bootstrap_samples=32, bootstrap_seed=19,
    )
    assert summary["episode_disjoint"]
    assert summary["fit_split"] == "train_target_latent_only"
    assert summary["reserved_test_accessed"] is False
    assert summary["scope"] == "exploratory_development_only_fixed_view_privileged_q0_center"
    assert summary["main_forbidden_inputs"] == ["horizon", "time", "action"]
    assert set(("static_q0", "task_time_q0", "direct_action", "target", "factual", "no_action", "wrong_action")).issubset(summary["reports"])
    assert artifact["factual"].shape == (
        3, len(DEVELOPMENT_TRAINING_SEEDS), len(TASKS), HORIZON, 2
    )
    assert artifact["wrong_action"].shape == (
        3, len(DEVELOPMENT_TRAINING_SEEDS), BRANCH_COUNT - 1, len(TASKS), HORIZON, 2
    )
    assert summary["reports"]["target"]["primary_error_mean"]["macro_mean_epe"] < summary["reports"]["static_q0"]["macro_mean_epe"]
    parameter_counts = summary["parameter_count"]
    assert max(parameter_counts.values()) / min(parameter_counts.values()) < 1.02


def test_geometry_gate_refuses_reserved_test_cache(tmp_path: Path) -> None:
    cache, companion, predictions = _sources(tmp_path, include_test=True)
    with pytest.raises(PermissionError, match="reserved test"):
        gate.run_geometry_gate(
            cache, companion, predictions,
            device_name="cpu", steps=1, batch_size=16, hidden_dim=8,
            learning_rate=1.0e-3, head_seeds=(1, 2, 3),
            bootstrap_samples=0, bootstrap_seed=2,
            prediction_mode="smoke",
        )


def test_cli_writes_compact_diagnostic_artifacts(tmp_path: Path) -> None:
    cache, companion, predictions = _sources(tmp_path)
    output = tmp_path / "output"
    assert gate.main([
        "--cache", str(cache), "--companion", str(companion),
        "--predictions", *(str(path) for path in predictions),
        "--output", str(output), "--device", "cpu", "--steps", "4",
        "--batch-size", "32", "--hidden-dim", "8", "--bootstrap-samples", "4",
    ]) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert not summary["reserved_test_accessed"]
    with np.load(output / "development_predictions.npz", allow_pickle=False) as archive:
        assert set(archive["split"].astype(str).tolist()) == {"development"}
        assert "target_delta" in archive.files
