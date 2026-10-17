from __future__ import annotations

import numpy as np
import pytest

from fcpg.lewm_action_branch_a3_metrics import (
    TASKS,
    TRAINING_SEEDS,
    fit_task_whitening_moments,
    summarize_a3_metrics,
)
from fcpg.lewm_action_branch_metrics import (
    FACTUAL_METHOD,
    NO_ACTION_METHOD,
    SHUFFLED_ACTION_METHOD,
    make_all_cyclic_branch_shifts,
)


def test_task_whitening_is_late_horizon_and_episode_equal() -> None:
    target = np.empty((4, 8, 8, 1, 1), dtype=np.float64)
    target[:, :, :5] = np.asarray([100.0, 200.0, 300.0, 400.0])[
        :, None, None, None, None
    ]
    target[0, :, 5:] = 0.0
    target[1:, :, 5:] = 10.0
    moments = fit_task_whitening_moments(
        target,
        task_index=np.zeros(4, dtype=np.int64),
        source_episode_id=np.asarray(["one", "three", "three", "three"]),
        split=np.full(4, "train"),
        task_names=("task",),
    )["task"]

    assert moments.count == 2
    assert moments.mean.item() == pytest.approx(5.0)
    assert moments.scale.item() == pytest.approx(5.0)


def _canonical_panel() -> dict[str, np.ndarray]:
    rows: list[tuple[int, int, str, int, str]] = []
    for task in range(4):
        instance = 0
        for split, episode_count in (("train", 2), ("development", 2)):
            for episode in range(episode_count):
                hazard_count = 1 + episode
                identity = f"task-{task}:{split}:{episode}"
                for hazard in range(hazard_count):
                    rows.append((task, instance, identity, 10 + hazard, split))
                instance += 1

    count = len(rows)
    target = np.empty((count, 8, 8, 1, 1), dtype=np.float64)
    branch = np.arange(8, dtype=np.float64) - 3.5
    for row in range(count):
        for horizon in range(8):
            target[row, :, horizon, 0, 0] = branch * (horizon + 1)
    return {
        "target_embedding": target,
        "history_embedding": np.zeros((count, 3, 1, 1), dtype=np.float64),
        "task_index": np.asarray([row[0] for row in rows], dtype=np.int64),
        "task_instance_index": np.asarray(
            [row[1] for row in rows], dtype=np.int64
        ),
        "source_episode_id": np.asarray([row[2] for row in rows]),
        "hazard_id": np.asarray([row[3] for row in rows], dtype=np.int64),
        "split": np.asarray([row[4] for row in rows]),
    }


def _perfect_predictions(
    canonical: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    development = canonical["split"].astype(str) == "development"
    target = canonical["target_embedding"][development]
    seed_count = len(TRAINING_SEEDS)
    factual = np.broadcast_to(target[None], (seed_count, *target.shape)).copy()
    no_action = np.zeros_like(factual)
    permutations = make_all_cyclic_branch_shifts(target.shape[0])
    shuffled = np.empty(
        (seed_count, target.shape[0], 7, *target.shape[1:]), dtype=np.float64
    )
    for row in range(target.shape[0]):
        for shift in range(7):
            shuffled[:, row, shift] = target[row, permutations[row, shift]]
    return {
        FACTUAL_METHOD: factual,
        NO_ACTION_METHOD: no_action,
        SHUFFLED_ACTION_METHOD: shuffled,
    }


def test_a3_summary_applies_complete_late_horizon_gate() -> None:
    canonical = _canonical_panel()
    summary = summarize_a3_metrics(canonical, _perfect_predictions(canonical))

    assert summary["decision"] == "Development-Go"
    assert summary["primary_horizons"] == [6, 7, 8]
    assert summary["whitening"]["episode_equal"] is True
    assert summary["comparisons"][NO_ACTION_METHOD]["effect"] == pytest.approx(1.0)
    assert summary["comparisons"][SHUFFLED_ACTION_METHOD]["effect"] == pytest.approx(
        1.0
    )
    assert summary["shuffled_action_shifts"]["count"] == 7
    assert len(summary["shuffled_action_shifts"]["per_shift"]) == 7
    assert summary["gate"]["jointly_positive_tasks"] == list(TASKS)
    assert summary["gate"]["jointly_positive_seeds"] == list(TRAINING_SEEDS)
    assert all(summary["gate"]["checks"].values())


def test_development_hazards_are_reduced_to_episode_equal_task_values() -> None:
    canonical = _canonical_panel()
    development = canonical["split"].astype(str) == "development"
    second_episode = np.char.endswith(
        canonical["source_episode_id"].astype(str), ":1"
    )
    canonical["target_embedding"][development & second_episode] *= 3.0

    summary = summarize_a3_metrics(canonical, _perfect_predictions(canonical))

    for task in TASKS:
        # Train whitening makes the first episode's no-action loss 1 and the
        # second episode's loss 9. Equal episode weight is therefore 5, while
        # a hazard-row mean would incorrectly be 19/3.
        assert summary["methods"][NO_ACTION_METHOD]["per_task"][task][
            "mse"
        ] == pytest.approx(5.0)


def test_every_shuffled_shift_must_clear_five_percent() -> None:
    canonical = _canonical_panel()
    predictions = _perfect_predictions(canonical)
    predictions[SHUFFLED_ACTION_METHOD][:, :, 0] = predictions[FACTUAL_METHOD]

    summary = summarize_a3_metrics(canonical, predictions)

    assert summary["decision"] == "Development-No-Go"
    assert summary["shuffled_action_shifts"]["minimum_effect"] == 0.0
    assert not summary["gate"]["checks"][
        "late_effect_vs_every_shuffled_shift"
    ]


def test_two_of_three_positive_seeds_are_required_and_sufficient() -> None:
    canonical = _canonical_panel()
    predictions = _perfect_predictions(canonical)
    predictions[FACTUAL_METHOD][2] = predictions[NO_ACTION_METHOD][2]
    two_seed_summary = summarize_a3_metrics(canonical, predictions)
    assert two_seed_summary["gate"]["checks"]["jointly_positive_seeds"]

    predictions[FACTUAL_METHOD][1] = predictions[NO_ACTION_METHOD][1]
    one_seed_summary = summarize_a3_metrics(canonical, predictions)
    assert not one_seed_summary["gate"]["checks"]["jointly_positive_seeds"]
    assert one_seed_summary["decision"] == "Development-No-Go"


def test_raw_simple_baselines_block_centered_only_success() -> None:
    canonical = _canonical_panel()
    predictions = _perfect_predictions(canonical)
    predictions[FACTUAL_METHOD] += 100.0

    summary = summarize_a3_metrics(canonical, predictions)

    assert summary["comparisons"][NO_ACTION_METHOD]["effect"] == pytest.approx(1.0)
    assert summary["decision"] == "Development-No-Go"
    assert not summary["gate"]["checks"]["raw_not_worse_than_no_action"]
    assert not summary["gate"]["checks"]["raw_better_than_hold_last"]


def test_a3_scoring_refuses_any_canonical_test_row() -> None:
    canonical = _canonical_panel()
    canonical["split"] = canonical["split"].copy()
    canonical["split"][0] = "test"
    with pytest.raises(ValueError, match="refuses canonical test rows"):
        summarize_a3_metrics(canonical, _perfect_predictions(_canonical_panel()))
