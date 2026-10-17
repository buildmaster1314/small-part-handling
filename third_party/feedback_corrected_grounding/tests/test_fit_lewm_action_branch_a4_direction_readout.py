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
from scripts import fit_lewm_action_branch_a4_direction_readout as direction_cli


def _synthetic_sources(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    rows = []
    for task_index, task_name in enumerate(TASKS):
        for episode_index, split in ((0, "train"), (1, "train"), (2, "development")):
            rows.append((task_index, task_name, episode_index, split))
    count = len(rows)
    proposal_id = np.asarray([f"proposal-{index}" for index in range(count)])
    task_index = np.asarray([row[0] for row in rows], dtype=np.int64)
    task_name = np.asarray([row[1] for row in rows])
    task_instance = np.asarray([row[2] for row in rows], dtype=np.int64)
    episode = np.asarray(
        [f"{row[1]}:{row[3]}:{row[2]}" for row in rows]
    )
    split = np.asarray([row[3] for row in rows])

    history = np.zeros(
        (count, HISTORY_STEPS, STATE_SLOT_COUNT, LATENT_DIM), dtype=np.float32
    )
    for row in range(count):
        history[row, :, 1:, 7] = 0.1 * (row + 1)
    target = np.repeat(
        history[:, -1:, None], BRANCH_COUNT * HORIZON, axis=1
    ).reshape(count, BRANCH_COUNT, HORIZON, STATE_SLOT_COUNT, LATENT_DIM)
    q_center = np.empty((count, HORIZON + 1, 2), dtype=np.float32)
    for row, (task_id, _task_name, episode_id, _split) in enumerate(rows):
        base = np.asarray([30.0 + task_id, 45.0 + episode_id], dtype=np.float32)
        q_center[row, 0] = base
        for horizon in range(1, HORIZON + 1):
            displacement = np.asarray(
                [0.35 * horizon + 0.02 * task_id, -0.2 * horizon + 0.01 * episode_id],
                dtype=np.float32,
            )
            q_center[row, horizon] = base + displacement
            for anchor in range(3):
                target[row, 0, horizon - 1, anchor + 1, 0] += displacement[0] * (
                    anchor + 1
                )
                target[row, 0, horizon - 1, anchor + 1, 1] += displacement[1] * (
                    anchor + 1
                )

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
        "target_state": target.astype(np.float32),
        "branch_semantic_index": np.broadcast_to(
            np.arange(BRANCH_COUNT, dtype=np.int64), (count, BRANCH_COUNT)
        ).copy(),
    }
    cache_path = tmp_path / "state_cache.npz"
    np.savez_compressed(cache_path, **cache)
    companion_path = tmp_path / "direction_companion.npz"
    np.savez_compressed(
        companion_path,
        proposal_id=proposal_id,
        split=split,
        task_name=task_name,
        source_episode_id=episode,
        q_center_model_raw=q_center,
    )

    development = split == "development"
    prediction_paths = []
    for seed in DEVELOPMENT_TRAINING_SEEDS:
        factual = target[development].copy()
        no_action = np.repeat(factual[:, :1], BRANCH_COUNT, axis=1)
        shuffled = np.stack(
            [
                factual[:, (np.arange(BRANCH_COUNT) + shift) % BRANCH_COUNT]
                for shift in range(1, BRANCH_COUNT)
            ],
            axis=1,
        )
        prediction_path = tmp_path / f"prediction-{seed}.npz"
        np.savez_compressed(
            prediction_path,
            protocol=np.asarray(PROTOCOL),
            checkpoint_step=np.asarray(2_000, dtype=np.int64),
            training_seed=np.full(np.count_nonzero(development), seed, dtype=np.int64),
            proposal_id=proposal_id[development],
            split=split[development],
            history_state=history[development],
            future_action=cache["future_action"][development],
            target_state=target[development],
            factual_prediction=factual,
            no_action_prediction=no_action,
            shuffled_action_prediction=shuffled,
        )
        prediction_paths.append(prediction_path)
    return cache_path, companion_path, prediction_paths


def test_direction_readout_writes_renderer_schema_and_beats_static(
    tmp_path: Path,
) -> None:
    cache, companion, predictions = _synthetic_sources(tmp_path)
    output = tmp_path / "output"

    assert (
        direction_cli.main(
            [
                "--cache",
                str(cache),
                "--companion",
                str(companion),
                "--predictions",
                *(str(path) for path in predictions),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["gate_pass"]
    assert metrics["fit_split"] == "train"
    assert metrics["evaluation_split"] == "development"
    assert not metrics["reserved_test_accessed"]
    assert set(metrics["per_task"]) == set(TASKS)
    assert metrics["target_ceiling"]["relative_improvement_vs_static_q0"] > 0.99
    assert metrics["factual"]["relative_improvement_vs_static_q0"] > 0.99

    with np.load(output / "development_predictions.npz", allow_pickle=False) as paths:
        assert set(
            (
                "proposal_id",
                "split",
                "task_name",
                "q_ground_truth_model",
                "q_decoded_target",
                "q_decoded_factual",
            )
        ).issubset(paths.files)
        assert paths["q_ground_truth_model"].shape == (len(TASKS), HORIZON + 1, 2)
        assert paths["q_decoded_target"].shape == (len(TASKS), HORIZON + 1, 2)
        assert paths["q_decoded_factual"].shape == (len(TASKS), HORIZON + 1, 2)
        assert paths["q_decoded_factual_by_seed"].shape == (
            len(DEVELOPMENT_TRAINING_SEEDS),
            len(TASKS),
            HORIZON + 1,
            2,
        )
        assert set(paths["split"].astype(str).tolist()) == {"development"}

    with np.load(output / "decoder.npz", allow_pickle=False) as decoder:
        assert decoder["coefficient_raw"].shape == (3 * LATENT_DIM, 2)
        assert decoder["intercept_raw"].shape == (2,)
        assert float(decoder["selected_alpha"]) in direction_cli.ALPHA_GRID
        assert str(decoder["fit_split"]) == "train"


def test_decoder_selection_and_weights_do_not_depend_on_development_labels(
    tmp_path: Path,
) -> None:
    cache, companion, predictions = _synthetic_sources(tmp_path)
    first_decoder, _, _ = direction_cli.fit_direction_readout(
        cache, companion, predictions
    )
    with np.load(companion, allow_pickle=False) as archive:
        changed = {name: np.array(archive[name], copy=True) for name in archive.files}
    development = changed["split"].astype(str) == "development"
    changed["q_center_model_raw"][development, 1:] += 1_000.0
    changed_path = tmp_path / "changed-development-companion.npz"
    np.savez_compressed(changed_path, **changed)

    second_decoder, _, _ = direction_cli.fit_direction_readout(
        cache, changed_path, predictions
    )

    assert float(first_decoder["selected_alpha"]) == float(
        second_decoder["selected_alpha"]
    )
    assert np.array_equal(
        first_decoder["cv_grouped_mse"], second_decoder["cv_grouped_mse"]
    )
    assert np.array_equal(
        first_decoder["coefficient_raw"], second_decoder["coefficient_raw"]
    )
    assert np.array_equal(
        first_decoder["intercept_raw"], second_decoder["intercept_raw"]
    )


def test_strict_gate_blocks_bad_factual_path_despite_good_target_ceiling(
    tmp_path: Path,
) -> None:
    cache, companion, predictions = _synthetic_sources(tmp_path)
    for path in predictions:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        arrays["factual_prediction"][..., 1:, :2] += 1_000.0
        np.savez_compressed(path, **arrays)

    _, _, metrics = direction_cli.fit_direction_readout(
        cache, companion, predictions
    )

    assert metrics["target_ceiling"]["macro_mean_epe"] <= 2.0
    assert not metrics["gate_pass"]
    assert not metrics["render_factual_path_authorized"]
    assert not metrics["gate"]["checks"]["factual"]["mean_epe_at_most_4px"]


def test_companion_with_reserved_test_row_is_rejected(tmp_path: Path) -> None:
    cache_path, companion_path, _ = _synthetic_sources(tmp_path)
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(companion_path, allow_pickle=False) as archive:
        companion = {name: np.array(archive[name], copy=True) for name in archive.files}
    companion["split"][0] = "test"
    forbidden = tmp_path / "forbidden-companion.npz"
    np.savez_compressed(forbidden, **companion)

    with pytest.raises(PermissionError, match="reserved test"):
        direction_cli._aligned_companion(forbidden, cache)


def test_legacy_q_center_alias_matches_raw_coordinate_contract(tmp_path: Path) -> None:
    cache_path, companion_path, _ = _synthetic_sources(tmp_path)
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(companion_path, allow_pickle=False) as archive:
        companion = {name: np.array(archive[name], copy=True) for name in archive.files}
    companion["q_center_model"] = companion.pop("q_center_model_raw")
    legacy = tmp_path / "legacy-companion.npz"
    np.savez_compressed(legacy, **companion)

    aligned = direction_cli._aligned_companion(legacy, cache)

    np.testing.assert_array_equal(
        aligned["q_center_model_raw"], companion["q_center_model"]
    )
    assert str(aligned["q_center_source_field"]) == "q_center_model"


def test_grouped_cv_never_splits_one_source_episode() -> None:
    task = np.asarray(["pick"] * 8 + ["pick"] * 8 + ["push"] * 8 + ["push"] * 8)
    episode = np.asarray(["p0"] * 8 + ["p1"] * 8 + ["u0"] * 8 + ["u1"] * 8)
    fold, count = direction_cli._grouped_folds(task, episode)

    assert count == 2
    for identity in np.unique(episode):
        assert np.unique(fold[episode == identity]).size == 1
