from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_action_branch_metrics import (
    BRANCH_COUNT,
    CYCLIC_SHIFT_COUNT,
    FACTUAL_METHOD,
    HOLD_LAST_METHOD,
    LATENT_CONSTANT_VELOCITY_METHOD,
    NO_ACTION_METHOD,
    SHUFFLED_ACTION_METHOD,
    TASK_NAMES,
    TrainMoments,
    branch_centered_whitened_mse,
    fit_train_moments,
    make_all_cyclic_branch_shifts,
    make_within_bundle_derangement,
    summarize_action_branch_metrics,
    validate_all_cyclic_branch_shifts,
    validate_within_bundle_derangement,
)
from scripts import score_action_branch_random_hazard_a1 as cli


SEEDS = (18401, 18402, 18403)


def _panel(
    *,
    episodes_per_task: int = 2,
    hazards_per_episode: int = 2,
    split: str = "development",
    bad_seeds: set[int] | None = None,
    bad_tasks: set[int] | None = None,
) -> dict[str, np.ndarray | TrainMoments | dict[str, np.ndarray]]:
    tasks: list[int] = []
    task_instances: list[int] = []
    episodes: list[int] = []
    hazards: list[int] = []
    seeds: list[int] = []
    targets: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    bad_seeds = bad_seeds or set()
    bad_tasks = bad_tasks or set()
    branch = (np.arange(BRANCH_COUNT, dtype=np.float64) - 3.5) * 0.4
    for task_index in range(4):
        for episode_index in range(episodes_per_task):
            source_episode_id = task_index * 100 + episode_index
            for hazard_id in range(hazards_per_episode):
                target = np.empty((BRANCH_COUNT, 8, 1, 2), dtype=np.float64)
                common = task_index * 0.1 + episode_index * 0.05 + hazard_id * 0.02
                for horizon in range(8):
                    factor = float(horizon + 1)
                    target[:, horizon, 0, 0] = common + factor * branch
                    target[:, horizon, 0, 1] = -common - 0.5 * factor * branch
                history = np.empty((3, 1, 2), dtype=np.float64)
                history[0, 0] = (common - 0.2, -common + 0.1)
                history[1, 0] = (common - 0.1, -common + 0.05)
                history[2, 0] = (common, -common)
                for training_seed in SEEDS:
                    tasks.append(task_index)
                    task_instances.append(task_index * 10 + episode_index)
                    episodes.append(source_episode_id)
                    hazards.append(hazard_id)
                    seeds.append(training_seed)
                    targets.append(target.copy())
                    histories.append(history.copy())

    target_array = np.stack(targets)
    history_array = np.stack(histories)
    permutation = make_all_cyclic_branch_shifts(target_array.shape[0]).copy()
    no_action = np.broadcast_to(
        target_array.mean(axis=1, keepdims=True), target_array.shape
    ).copy()
    factual = target_array.copy()
    for index, (task_index, training_seed) in enumerate(zip(tasks, seeds, strict=True)):
        if training_seed in bad_seeds or task_index in bad_tasks:
            factual[index] = no_action[index]
    shuffled = np.empty(
        (target_array.shape[0], CYCLIC_SHIFT_COUNT, *target_array.shape[1:]),
        dtype=target_array.dtype,
    )
    for index in range(target_array.shape[0]):
        for shift in range(CYCLIC_SHIFT_COUNT):
            shuffled[index, shift] = target_array[
                index, permutation[index, shift]
            ]
    return {
        "target": target_array,
        "predictions": {
            FACTUAL_METHOD: factual,
            NO_ACTION_METHOD: no_action,
            SHUFFLED_ACTION_METHOD: shuffled,
        },
        "history": history_array,
        "task_index": np.asarray(tasks, dtype=np.int64),
        "task_instance_index": np.asarray(task_instances, dtype=np.int64),
        "source_episode_id": np.asarray(episodes, dtype=np.int64),
        "hazard_id": np.asarray(hazards, dtype=np.int64),
        "training_seed": np.asarray(seeds, dtype=np.int64),
        "split": np.full(target_array.shape[0], split),
        "shuffle_permutation": permutation,
        "moments": TrainMoments(
            mean=np.zeros((1, 2)),
            scale=np.ones((1, 2)),
            count=100,
        ),
    }


def _score(
    panel: dict[str, object],
    *,
    stage: str = "screen",
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 19,
) -> dict[str, object]:
    return summarize_action_branch_metrics(
        panel["target"],
        panel["predictions"],
        panel["history"],
        task_index=panel["task_index"],
        task_instance_index=panel["task_instance_index"],
        source_episode_id=panel["source_episode_id"],
        hazard_id=panel["hazard_id"],
        split=panel["split"],
        training_seed=panel["training_seed"],
        moments=panel["moments"],
        shuffle_permutation=panel["shuffle_permutation"],
        stage=stage,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def test_train_moments_and_perfect_factual_gate() -> None:
    train = np.zeros((2, BRANCH_COUNT, 2, 1, 2), dtype=np.float64)
    train[..., 0] = np.arange(BRANCH_COUNT)[None, :, None, None]
    train[..., 1] = 7.0
    moments = fit_train_moments(train)
    assert moments.mean[0, 0] == pytest.approx(3.5)
    assert moments.scale[0, 0] == pytest.approx(np.std(np.arange(8)))
    assert moments.mean[0, 1] == 7.0
    assert moments.scale[0, 1] == 1.0
    assert moments.floored_dimensions == 1

    summary = _score(_panel())
    assert summary["decision"] == "Go"
    assert summary["passed"] is True
    assert summary["coverage"]["task_names"] == list(TASK_NAMES)
    assert summary["coverage"]["split"] == "development"
    assert summary["primary"]["methods"][FACTUAL_METHOD]["task_macro_mse"] == 0.0
    assert summary["comparisons"][NO_ACTION_METHOD]["effect"] == 1.0
    assert summary["comparisons"][SHUFFLED_ACTION_METHOD]["effect"] == 1.0
    assert summary["gate"]["jointly_positive_task_count"] == 4
    assert summary["gate"]["jointly_positive_seed_count"] == 3
    assert summary["informative_bundles"]["all_tasks_passed"] is True
    assert HOLD_LAST_METHOD in summary["secondary"]["methods"]
    assert LATENT_CONSTANT_VELOCITY_METHOD in summary["secondary"]["methods"]


def test_task_instance_is_one_to_one_with_episode_within_task() -> None:
    panel = _panel(episodes_per_task=2, hazards_per_episode=1)
    task_instances = np.asarray(panel["task_instance_index"]).copy()
    task_zero = np.asarray(panel["task_index"]) == 0
    episodes = np.asarray(panel["source_episode_id"])
    second_episode = task_zero & (episodes == 1)
    task_instances[second_episode] = 0
    panel["task_instance_index"] = task_instances

    with pytest.raises(ValueError, match="maps to multiple source episodes"):
        _score(panel)


def test_action_insensitive_factual_is_no_go() -> None:
    panel = _panel(bad_seeds=set(SEEDS))
    summary = _score(panel)

    comparison = summary["comparisons"][NO_ACTION_METHOD]
    assert comparison["effect"] == 0.0
    assert comparison["positive_task_count"] == 0
    assert summary["decision"] == "No-Go"
    assert (
        "factual_over_no_action_at_least_five_percent"
        in summary["gate"]["failed_checks"]
    )


def test_common_bundle_horizon_offset_is_removed_from_both_sides() -> None:
    target = np.arange(2 * BRANCH_COUNT * 3 * 2 * 2, dtype=np.float64).reshape(
        2, BRANCH_COUNT, 3, 2, 2
    )
    offset = np.asarray(
        [
            [
                [
                    [[10.0, -20.0], [30.0, -40.0]],
                    [[20.0, 4.0], [8.0, 12.0]],
                    [[-4.0, 2.0], [6.0, -8.0]],
                ]
            ],
            [
                [
                    [[100.0, 50.0], [-10.0, 6.0]],
                    [[7.0, -9.0], [11.0, 13.0]],
                    [[2.0, 4.0], [8.0, 16.0]],
                ]
            ],
        ]
    )
    prediction = target + offset
    moments = TrainMoments(np.zeros((2, 2)), np.asarray([[2.0, 4.0], [8.0, 16.0]]), 10)

    loss = branch_centered_whitened_mse(target, prediction, moments)
    np.testing.assert_allclose(loss, 0.0, rtol=0.0, atol=1e-28)


def test_exhaustive_shuffle_averages_complete_bundle_loss_per_shift() -> None:
    branch = (np.arange(BRANCH_COUNT, dtype=np.float64) - 3.5).reshape(
        1, BRANCH_COUNT, 1, 1, 1
    )
    target = np.broadcast_to(branch, (2, BRANCH_COUNT, 8, 1, 1)).copy()
    shuffled = np.empty((2, CYCLIC_SHIFT_COUNT, *target.shape[1:]))
    for shift in range(CYCLIC_SHIFT_COUNT):
        shuffled[:, shift] = target + (shift + 1) * branch
    moments = TrainMoments(np.zeros((1, 1)), np.ones((1, 1)), 10)

    per_shift = np.stack(
        [
            branch_centered_whitened_mse(target, shuffled[:, shift], moments)
            for shift in range(CYCLIC_SHIFT_COUNT)
        ],
        axis=1,
    )
    aggregate = branch_centered_whitened_mse(target, shuffled, moments)
    np.testing.assert_allclose(aggregate, per_shift.mean(axis=1))


def test_late_window_h6_h8_is_reported_separately() -> None:
    panel = _panel()
    factual = np.asarray(panel["target"]).copy()
    no_action = panel["predictions"][NO_ACTION_METHOD]
    factual[:, :, :5] = no_action[:, :, :5]
    predictions = dict(panel["predictions"])
    predictions[FACTUAL_METHOD] = factual
    modified = dict(panel)
    modified["predictions"] = predictions

    summary = _score(modified)
    expected_full_effect = sum(value * value for value in range(6, 9)) / sum(
        value * value for value in range(1, 9)
    )
    assert summary["comparisons"][NO_ACTION_METHOD]["effect"] == pytest.approx(
        expected_full_effect
    )
    assert summary["late_window"]["horizons_1_based"] == [6, 7, 8]
    assert (
        summary["late_window"]["comparisons"][NO_ACTION_METHOD]["effect"]
        == 1.0
    )
    assert (
        summary["late_window"]["comparisons"][SHUFFLED_ACTION_METHOD]["effect"]
        == 1.0
    )
    assert summary["late_window"]["gating"] is True


def test_late_window_h6_h8_blocks_an_early_only_gain() -> None:
    panel = _panel()
    factual = np.asarray(panel["target"]).copy()
    no_action = panel["predictions"][NO_ACTION_METHOD]
    factual[:, :, 5:] = no_action[:, :, 5:]
    predictions = dict(panel["predictions"])
    predictions[FACTUAL_METHOD] = factual
    modified = dict(panel)
    modified["predictions"] = predictions

    summary = _score(modified)

    assert summary["comparisons"][NO_ACTION_METHOD]["effect"] > 0.05
    assert summary["late_window"]["comparisons"][NO_ACTION_METHOD]["effect"] == 0.0
    assert summary["decision"] == "No-Go"
    assert (
        "h6_h8_factual_over_no_action_positive"
        in summary["gate"]["failed_checks"]
    )


def test_raw_mse_sanity_blocks_centered_only_improvement() -> None:
    panel = _panel()
    predictions = dict(panel["predictions"])
    predictions[FACTUAL_METHOD] = np.asarray(panel["target"]) + 100.0
    modified = dict(panel)
    modified["predictions"] = predictions

    summary = _score(modified)
    sanity = summary["sanity"]["factual_raw_mse_not_worse_than_no_action"]
    assert summary["comparisons"][NO_ACTION_METHOD]["effect"] == pytest.approx(1.0)
    assert sanity["passed"] is False
    assert summary["decision"] == "No-Go"
    assert (
        "factual_raw_mse_not_worse_than_no_action"
        in summary["gate"]["failed_checks"]
    )


def test_episode_cluster_bootstrap_keeps_unequal_hazards_nested() -> None:
    episode = np.asarray([10, 20, 20, 20], dtype=np.int64)
    hazard = np.asarray([0, 0, 1, 2], dtype=np.int64)
    branch = (np.arange(BRANCH_COUNT, dtype=np.float64) - 3.5).reshape(
        1, BRANCH_COUNT, 1, 1, 1
    )
    target = np.broadcast_to(branch, (4, BRANCH_COUNT, 8, 1, 1)).copy()
    no_action = np.zeros_like(target)
    factual = no_action.copy()
    factual[0] = target[0]
    permutation = make_all_cyclic_branch_shifts(4).copy()
    shuffled = np.empty((4, CYCLIC_SHIFT_COUNT, *target.shape[1:]))
    for index in range(4):
        for shift in range(CYCLIC_SHIFT_COUNT):
            shuffled[index, shift] = target[index, permutation[index, shift]]
    samples = 500
    bootstrap_seed = 31
    summary = summarize_action_branch_metrics(
        target,
        {
            FACTUAL_METHOD: factual,
            NO_ACTION_METHOD: no_action,
            SHUFFLED_ACTION_METHOD: shuffled,
        },
        np.zeros((4, 2, 1, 1), dtype=np.float64),
        task_index=np.zeros(4, dtype=np.int64),
        task_instance_index=np.asarray([0, 1, 1, 1], dtype=np.int64),
        source_episode_id=episode,
        hazard_id=hazard,
        split=np.full(4, "test"),
        training_seed=np.ones(4, dtype=np.int64),
        moments=TrainMoments(np.zeros((1, 1)), np.ones((1, 1)), 10),
        shuffle_permutation=permutation,
        stage="final",
        bootstrap_samples=samples,
        bootstrap_seed=bootstrap_seed,
    )

    comparison = summary["comparisons"][NO_ACTION_METHOD]
    assert comparison["effect"] == pytest.approx(0.5)
    assert summary["coverage"]["hazard_bundle_count"] == 4
    assert summary["coverage"]["source_physical_episode_count"] == 2
    draws = np.random.default_rng(bootstrap_seed).integers(0, 2, size=(samples, 2))
    factual_episode = np.asarray([0.0, 1.0])
    expected = 1.0 - factual_episode[draws].mean(axis=1)
    assert comparison["effect_95_ci"] == pytest.approx(
        np.quantile(expected, [0.025, 0.975])
    )


def test_gate_requires_two_seeds_and_three_tasks_positive_over_both_controls() -> None:
    two_good_seeds = _score(_panel(bad_seeds={SEEDS[2]}))
    assert two_good_seeds["decision"] == "Go"
    assert two_good_seeds["gate"]["jointly_positive_seed_count"] == 2

    one_good_seed = _score(_panel(bad_seeds={SEEDS[1], SEEDS[2]}))
    assert one_good_seed["decision"] == "No-Go"
    assert "jointly_positive_seeds" in one_good_seed["gate"]["failed_checks"]

    three_good_tasks = _score(_panel(bad_tasks={3}))
    assert three_good_tasks["decision"] == "Go"
    assert three_good_tasks["gate"]["jointly_positive_task_count"] == 3

    two_good_tasks = _score(_panel(bad_tasks={2, 3}))
    assert two_good_tasks["decision"] == "No-Go"
    assert "jointly_positive_tasks" in two_good_tasks["gate"]["failed_checks"]


def test_derangement_generation_and_validation_fail_closed() -> None:
    first = make_within_bundle_derangement(20, seed=7)
    second = make_within_bundle_derangement(20, seed=7)
    np.testing.assert_array_equal(first, second)
    validated = validate_within_bundle_derangement(first, bundle_count=20)
    np.testing.assert_array_equal(validated, first)
    assert np.all(first != np.arange(BRANCH_COUNT))
    assert all(np.array_equal(np.sort(row), np.arange(BRANCH_COUNT)) for row in first)

    with pytest.raises(ValueError, match="derangement"):
        validate_within_bundle_derangement(np.arange(BRANCH_COUNT))
    duplicate = np.asarray([1, 0, 2, 3, 4, 5, 6, 6], dtype=np.int64)
    with pytest.raises(ValueError, match="not a branch permutation"):
        validate_within_bundle_derangement(duplicate)
    with pytest.raises(TypeError, match="integer dtype"):
        validate_within_bundle_derangement(np.arange(BRANCH_COUNT, dtype=np.float64))

    cyclic = make_all_cyclic_branch_shifts(3)
    np.testing.assert_array_equal(
        validate_all_cyclic_branch_shifts(cyclic, bundle_count=3), cyclic
    )
    assert cyclic.shape == (3, CYCLIC_SHIFT_COUNT, BRANCH_COUNT)
    duplicate_shift = cyclic.copy()
    duplicate_shift[:, 1] = duplicate_shift[:, 0]
    with pytest.raises(ValueError, match="exactly once"):
        validate_all_cyclic_branch_shifts(duplicate_shift, bundle_count=3)
    non_cyclic = cyclic.copy()
    non_cyclic[0, 0, [0, 1]] = non_cyclic[0, 0, [1, 0]]
    with pytest.raises(ValueError, match="only non-identity cyclic shifts"):
        validate_all_cyclic_branch_shifts(non_cyclic, bundle_count=3)


def test_nan_shape_and_no_action_branch_variation_are_rejected() -> None:
    panel = _panel()
    nan_target = np.asarray(panel["target"]).copy()
    nan_target[0, 0, 0, 0, 0] = np.nan
    poisoned = dict(panel)
    poisoned["target"] = nan_target
    with pytest.raises(ValueError, match="finite"):
        _score(poisoned)

    predictions = dict(panel["predictions"])
    predictions[FACTUAL_METHOD] = predictions[FACTUAL_METHOD][:, :7]
    bad_shape = dict(panel)
    bad_shape["predictions"] = predictions
    with pytest.raises(ValueError, match="must have shape"):
        _score(bad_shape)

    predictions = dict(panel["predictions"])
    predictions[SHUFFLED_ACTION_METHOD] = predictions[SHUFFLED_ACTION_METHOD][:, 0]
    single_shuffle = dict(panel)
    single_shuffle["predictions"] = predictions
    with pytest.raises(ValueError, match="must have shape"):
        _score(single_shuffle)

    predictions = dict(panel["predictions"])
    predictions[NO_ACTION_METHOD] = predictions[NO_ACTION_METHOD].copy()
    predictions[NO_ACTION_METHOD][0, 1] += 0.01
    varying = dict(panel)
    varying["predictions"] = predictions
    with pytest.raises(ValueError, match="branch-invariant"):
        _score(varying)

    with pytest.raises(ValueError, match="split='development'"):
        _score(_panel(split="test"))

    train = np.zeros((1, BRANCH_COUNT, 1, 1, 1))
    train[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_train_moments(train)


def test_cli_uses_canonical_metadata_and_train_split_moments(
    tmp_path: Path,
) -> None:
    train = _panel(episodes_per_task=12, split="train")
    development = _panel(episodes_per_task=12, split="development")
    train_rows = np.asarray(train["training_seed"]) == SEEDS[0]
    train_data = tmp_path / "core.npz"
    np.savez(
        train_data,
        target_embedding=np.asarray(train["target"])[train_rows],
        split=np.asarray(train["split"])[train_rows],
    )

    sources: list[Path] = []
    for training_seed in SEEDS:
        rows = np.asarray(development["training_seed"]) == training_seed
        payload = {
            "target_embedding": np.asarray(development["target"])[rows],
            "history_embedding": np.asarray(development["history"])[rows],
            "task_index": np.asarray(development["task_index"])[rows],
            "task_instance_index": np.asarray(
                development["task_instance_index"]
            )[rows],
            "source_episode_id": np.asarray(
                development["source_episode_id"]
            )[rows],
            "hazard_id": np.asarray(development["hazard_id"])[rows],
            "training_seed": np.asarray(development["training_seed"])[rows],
            "seed": np.asarray(development["training_seed"])[rows],
            "split": np.asarray(development["split"])[rows],
            "shuffle_permutation": np.asarray(
                development["shuffle_permutation"]
            )[rows],
            "factual_prediction": development["predictions"][FACTUAL_METHOD][
                rows
            ],
            "no_action_prediction": development["predictions"][NO_ACTION_METHOD][
                rows
            ],
            "shuffled_action_prediction": development["predictions"][
                SHUFFLED_ACTION_METHOD
            ][rows],
        }
        source = tmp_path / f"seed-{training_seed}.npz"
        np.savez(source, **payload)
        sources.append(source)
    output = tmp_path / "score"

    result = cli.main(
        [
            "--stage",
            "screen",
            "--predictions",
            *(str(source) for source in sources),
            "--train-data",
            str(train_data),
            "--output",
            str(output),
            "--bootstrap-samples",
            "0",
        ]
    )

    assert result == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert summary["decision"] == "Go"
    assert summary["coverage"]["task_names"] == list(TASK_NAMES)
    assert summary["coverage"]["split"] == "development"
    assert summary["coverage"]["shuffle_shift_count"] == CYCLIC_SHIFT_COUNT
    assert summary["whitening"]["source"] == "train_split_only"
    assert summary["sanity"]["factual_raw_mse_not_worse_than_no_action"][
        "passed"
    ]
    assert "Action-Branch Random-Hazard A1" in markdown
    assert "Matched no-action" in markdown
    assert "Late window H6-H8" in markdown


def test_confirmatory_support_rejects_wrong_horizon_seed_and_episode_count() -> None:
    contract = cli._load_frozen_contract(cli.DEFAULT_CONFIG)
    panel = _panel(episodes_per_task=12)
    arrays = {
        "target": np.asarray(panel["target"]),
        "task_index": np.asarray(panel["task_index"]),
        "task_instance_index": np.asarray(panel["task_instance_index"]),
        "source_episode_id": np.asarray(panel["source_episode_id"]),
        "seed": np.asarray(panel["training_seed"]),
    }
    cli._validate_confirmatory_support(
        arrays, stage="screen", contract=contract
    )

    wrong_horizon = dict(arrays)
    wrong_horizon["target"] = arrays["target"][:, :, :7]
    with pytest.raises(ValueError, match="exactly H=8"):
        cli._validate_confirmatory_support(
            wrong_horizon, stage="screen", contract=contract
        )

    wrong_seed = dict(arrays)
    wrong_seed["seed"] = np.where(arrays["seed"] == SEEDS[-1], 999, arrays["seed"])
    with pytest.raises(ValueError, match="seed set"):
        cli._validate_confirmatory_support(
            wrong_seed, stage="screen", contract=contract
        )

    fewer_episodes = _panel(episodes_per_task=11)
    short_arrays = {
        "target": np.asarray(fewer_episodes["target"]),
        "task_index": np.asarray(fewer_episodes["task_index"]),
        "task_instance_index": np.asarray(fewer_episodes["task_instance_index"]),
        "source_episode_id": np.asarray(fewer_episodes["source_episode_id"]),
        "seed": np.asarray(fewer_episodes["training_seed"]),
    }
    with pytest.raises(ValueError, match="exactly 12 source episodes"):
        cli._validate_confirmatory_support(
            short_arrays, stage="screen", contract=contract
        )


def test_smoke_support_requires_frozen_task_episode_and_branch_panel() -> None:
    contract = cli._load_frozen_contract(cli.DEFAULT_CONFIG)
    panel = _panel(episodes_per_task=2)
    arrays = {
        "target": np.asarray(panel["target"]),
        "task_index": np.asarray(panel["task_index"]),
        "task_instance_index": np.asarray(panel["task_instance_index"]),
        "source_episode_id": np.asarray(panel["source_episode_id"]),
        "seed": np.asarray(panel["training_seed"]),
    }
    cli._validate_confirmatory_support(arrays, stage="smoke", contract=contract)

    one_task = {
        name: value[np.asarray(arrays["task_index"]) == 0]
        for name, value in arrays.items()
    }
    with pytest.raises(ValueError, match="task set"):
        cli._validate_confirmatory_support(
            one_task, stage="smoke", contract=contract
        )

    wrong_branch_count = dict(arrays)
    wrong_branch_count["target"] = arrays["target"][:, :7]
    with pytest.raises(ValueError, match="exactly 8 branches"):
        cli._validate_confirmatory_support(
            wrong_branch_count, stage="smoke", contract=contract
        )

    short_panel = _panel(episodes_per_task=1)
    short_arrays = {
        "target": np.asarray(short_panel["target"]),
        "task_index": np.asarray(short_panel["task_index"]),
        "task_instance_index": np.asarray(short_panel["task_instance_index"]),
        "source_episode_id": np.asarray(short_panel["source_episode_id"]),
        "seed": np.asarray(short_panel["training_seed"]),
    }
    with pytest.raises(ValueError, match="exactly 2 source episodes"):
        cli._validate_confirmatory_support(
            short_arrays, stage="smoke", contract=contract
        )
