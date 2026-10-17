from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_posterior_dagger import (
    METHODS,
    PROTOCOL,
    PosteriorDaggerArrays,
    load_posterior_dagger_checkpoint,
    save_posterior_dagger_arrays,
)
from scripts.train_evaluate_lewm_posterior_dagger import (
    CacheReceipt,
    MetricRows,
    _load_cache_receipt,
    _validate_matched_collection,
    assert_disjoint_physical_episodes,
    balanced_dagger_batch_indices,
    episode_comparison_interval,
    main,
)


def _metric_rows(
    values: list[float],
    *,
    decisions: list[str] | None = None,
    episodes: list[str] | None = None,
    tasks: list[str] | None = None,
    iterations: list[int] | None = None,
) -> MetricRows:
    count = len(values)
    return MetricRows(
        value=np.asarray(values, dtype=np.float64),
        task_name=np.asarray(tasks or ["pick"] * count),
        physical_episode_id=np.asarray(
            episodes or [f"ep-{index // 2}" for index in range(count)]
        ),
        dagger_iteration=np.asarray(iterations or [0] * count, dtype=np.int64),
        decision_identity=np.asarray(
            decisions or [f"decision-{index}" for index in range(count)]
        ),
    )


def test_balanced_dagger_batches_do_not_drown_later_rounds() -> None:
    iterations = np.asarray([0] * 100 + [1] * 7 + [2] * 2, dtype=np.int64)
    first = balanced_dagger_batch_indices(
        iterations, steps=20, batch_size=11, seed=19
    )
    second = balanced_dagger_batch_indices(
        iterations, steps=20, batch_size=11, seed=19
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 11)
    for batch in first:
        counts = np.bincount(iterations[batch], minlength=3)
        assert counts.min() >= 3
        assert counts.max() - counts.min() <= 1


def test_balanced_dagger_batches_require_room_for_every_iteration() -> None:
    with pytest.raises(ValueError, match="at least the number"):
        balanced_dagger_batch_indices(
            np.asarray([0, 1, 2], dtype=np.int64),
            steps=2,
            batch_size=2,
            seed=1,
        )


def test_physical_episode_overlap_is_rejected_even_across_tasks() -> None:
    with pytest.raises(ValueError, match="physical episodes overlap"):
        assert_disjoint_physical_episodes(
            ["pick"],
            ["episode-1"],
            ["push"],
            ["episode-1"],
        )


def test_paired_interval_requires_and_uses_exact_decision_alignment() -> None:
    candidate = _metric_rows([1.0, 1.0, 2.0, 2.0])
    baseline = _metric_rows([2.0, 2.0, 4.0, 4.0])
    result = episode_comparison_interval(
        candidate, baseline, samples=2_000, seed=37
    )
    assert result["paired"] is True
    assert result["bootstrap_design"] == "paired_physical_episode_within_task"
    assert result["absolute_improvement"] == pytest.approx(1.5)
    assert result["relative_improvement"] == pytest.approx(0.5)
    assert result["absolute_improvement_95_ci"][0] > 0.0


def test_identity_mismatch_is_reported_as_unpaired_not_silently_paired() -> None:
    candidate = _metric_rows(
        [1.0, 1.2, 0.8, 1.1],
        decisions=["candidate-a", "candidate-b", "candidate-c", "candidate-d"],
        episodes=["candidate-0", "candidate-0", "candidate-1", "candidate-1"],
    )
    baseline = _metric_rows(
        [2.0, 2.2, 1.8, 2.1],
        decisions=["baseline-a", "baseline-b", "baseline-c", "baseline-d"],
        episodes=["baseline-0", "baseline-0", "baseline-1", "baseline-1"],
    )
    result = episode_comparison_interval(
        candidate, baseline, samples=1_000, seed=41
    )
    assert result["paired"] is False
    assert result["bootstrap_design"].startswith("unpaired_")
    assert "no paired claim" in result["pairing_note"]
    assert result["absolute_improvement"] == pytest.approx(1.0)


def test_divergent_decision_counts_on_same_episode_panel_are_unpaired() -> None:
    candidate = _metric_rows(
        [1.0, 1.2, 0.9],
        decisions=["ep0-r0-d0", "ep0-r0-d1", "ep0-r1-d0"],
        episodes=["ep0", "ep0", "ep0"],
        iterations=[0, 0, 1],
    )
    baseline = _metric_rows(
        [2.0, 2.2, 2.1, 1.9],
        decisions=["ep0-r0-d0", "ep0-r0-d1", "ep0-r1-d0", "ep0-r1-d1"],
        episodes=["ep0"] * 4,
        iterations=[0, 0, 1, 1],
    )
    result = episode_comparison_interval(
        candidate, baseline, samples=100, seed=43
    )
    assert result["paired"] is False
    assert result["bootstrap_design"].startswith("unpaired_")


def test_interval_is_task_macro_instead_of_row_pooled() -> None:
    candidate = _metric_rows(
        [0.0, 0.0, 9.0],
        tasks=["many", "many", "few"],
        episodes=["m0", "m1", "f0"],
    )
    baseline = _metric_rows(
        [2.0, 2.0, 11.0],
        tasks=["many", "many", "few"],
        episodes=["m0", "m1", "f0"],
    )
    result = episode_comparison_interval(
        candidate, baseline, samples=500, seed=5
    )
    assert result["candidate_xyz_mse"] == pytest.approx(4.5)
    assert result["baseline_xyz_mse"] == pytest.approx(6.5)
    assert result["absolute_improvement"] == pytest.approx(2.0)


def test_interval_weights_dagger_rounds_equally_in_point_and_bootstrap() -> None:
    decisions = [f"round0-{index}" for index in range(9)] + ["round1-0"]
    candidate = _metric_rows(
        [0.0] * 9 + [10.0],
        decisions=decisions,
        episodes=["same"] * 10,
        iterations=[0] * 9 + [1],
    )
    baseline = _metric_rows(
        [10.0] * 10,
        decisions=decisions,
        episodes=["same"] * 10,
        iterations=[0] * 9 + [1],
    )
    result = episode_comparison_interval(
        candidate, baseline, samples=200, seed=23
    )
    assert result["candidate_xyz_mse"] == pytest.approx(5.0)
    assert result["baseline_xyz_mse"] == pytest.approx(10.0)
    assert result["absolute_improvement"] == pytest.approx(5.0)
    assert result["absolute_improvement_95_ci"] == pytest.approx([5.0, 5.0])
    assert result["round_aggregation"] == "equal_weight_dagger_rounds"


def test_round_mean_precedes_episode_mean_when_an_episode_has_no_later_rows() -> None:
    decisions = ["ep0-r0", "ep0-r1", "ep1-r0"]
    candidate = _metric_rows(
        [0.0, 10.0, 0.0],
        decisions=decisions,
        episodes=["ep0", "ep0", "ep1"],
        iterations=[0, 1, 0],
    )
    baseline = _metric_rows(
        [10.0, 10.0, 10.0],
        decisions=decisions,
        episodes=["ep0", "ep0", "ep1"],
        iterations=[0, 1, 0],
    )
    result = episode_comparison_interval(
        candidate, baseline, samples=200, seed=29
    )
    assert result["candidate_xyz_mse"] == pytest.approx(2.5)
    assert result["baseline_xyz_mse"] == pytest.approx(10.0)


def test_duplicate_decision_identity_is_rejected() -> None:
    candidate = _metric_rows(
        [1.0, 2.0], decisions=["same", "same"], episodes=["a", "a"]
    )
    baseline = _metric_rows([2.0, 3.0])
    with pytest.raises(ValueError, match="decision identities must be unique"):
        episode_comparison_interval(candidate, baseline, samples=10, seed=3)


def _dagger_arrays(
    *,
    method_index: int,
    iteration: int,
    episodes: tuple[int, ...],
    model_dim: int = 8,
) -> PosteriorDaggerArrays:
    rows_per_episode = 2
    count = len(episodes) * rows_per_episode
    rng = np.random.default_rng(100 + 10 * method_index + iteration)
    episode_id = np.repeat(np.asarray(episodes, dtype=np.int64), rows_per_episode)
    decision = np.tile(np.arange(rows_per_episode, dtype=np.int64), len(episodes))
    base = rng.uniform(-0.5, 0.5, size=(count, 5, 4)).astype(np.float32)
    expert = np.array(base, copy=True)
    expert[..., :3] = np.clip(expert[..., :3] + 0.1, -1.0, 1.0)
    return PosteriorDaggerArrays(
        episode_id=episode_id,
        decision_index=decision,
        dagger_iteration=np.full(count, iteration, dtype=np.int64),
        method_index=np.full(count, method_index, dtype=np.int64),
        missing_age=np.full(count, 2, dtype=np.int64),
        sensor_valid=np.zeros(count, dtype=bool),
        state_latent=rng.normal(size=(count, model_dim)).astype(np.float32),
        action_latent=rng.normal(size=(count, 5, model_dim)).astype(np.float32),
        base_action=base,
        posterior_token=rng.normal(size=(count, 3, model_dim)).astype(np.float32),
        expert_action=expert,
        executed_action=np.array(base, copy=True),
    )


def _write_sidecar(
    path: Path,
    *,
    split: str,
    method: str,
    iteration: int,
    episodes: tuple[int, ...],
    rows: int,
) -> Path:
    metadata = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "split": split,
        "method": method,
        "dagger_iteration": iteration,
        "episodes": len(episodes),
        "rows": rows,
        "head_checkpoint": None,
        "head_metadata": None,
        "episode_metrics": [
            {
                "episode_id": episode,
                "task_index": 100 + index,
                "environment_seed": episode,
            }
            for index, episode in enumerate(episodes)
        ],
    }
    if iteration > 0:
        metadata["head_checkpoint"] = str(path.parent / f"{method}.round{iteration - 1}.pt")
        metadata["head_metadata"] = {
            "method": method,
            "dagger_iteration": iteration - 1,
        }
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def _write_cache(
    path: Path,
    *,
    split: str,
    method: str,
    method_index: int,
    iteration: int,
    episodes: tuple[int, ...],
) -> PosteriorDaggerArrays:
    arrays = _dagger_arrays(
        method_index=method_index,
        iteration=iteration,
        episodes=episodes,
    )
    save_posterior_dagger_arrays(path, arrays)
    _write_sidecar(
        path,
        split=split,
        method=method,
        iteration=iteration,
        episodes=episodes,
        rows=arrays.sample_count,
    )
    return arrays


def test_cache_receipt_requires_and_validates_sibling_metadata(tmp_path) -> None:
    path = tmp_path / "round0.npz"
    with pytest.raises(FileNotFoundError, match="requires sibling sidecar"):
        _load_cache_receipt(
            path,
            expected_split="train",
            expected_method=METHODS[0],
            expected_protocol=PROTOCOL,
            observed_rows=4,
            observed_iterations=np.zeros(4, dtype=np.int64),
        )

    metadata_path = _write_sidecar(
        path,
        split="train",
        method=METHODS[0],
        iteration=0,
        episodes=(1, 2),
        rows=4,
    )
    receipt = _load_cache_receipt(
        path,
        expected_split="train",
        expected_method=METHODS[0],
        expected_protocol=PROTOCOL,
        observed_rows=4,
        observed_iterations=np.zeros(4, dtype=np.int64),
    )
    assert receipt.dagger_iteration == 0
    assert len(receipt.episode_panel) == 2

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["rows"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="row count mismatch"):
        _load_cache_receipt(
            path,
            expected_split="train",
            expected_method=METHODS[0],
            expected_protocol=PROTOCOL,
            observed_rows=4,
            observed_iterations=np.zeros(4, dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("split", "development", "split mismatch"),
        ("method", METHODS[1], "method mismatch"),
        ("dagger_iteration", 1, "exactly sidecar DAgger round"),
    ],
)
def test_cache_receipt_rejects_identity_mismatch(
    tmp_path, field: str, replacement: object, message: str
) -> None:
    path = tmp_path / "round0.npz"
    metadata_path = _write_sidecar(
        path,
        split="train",
        method=METHODS[0],
        iteration=0,
        episodes=(1, 2),
        rows=4,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = replacement
    if field == "dagger_iteration":
        metadata["head_checkpoint"] = "round0.pt"
        metadata["head_metadata"] = {
            "method": METHODS[0],
            "dagger_iteration": 0,
        }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_cache_receipt(
            path,
            expected_split="train",
            expected_method=METHODS[0],
            expected_protocol=PROTOCOL,
            observed_rows=4,
            observed_iterations=np.zeros(4, dtype=np.int64),
        )


def test_cache_receipt_enforces_behavior_head_round_contract(tmp_path) -> None:
    path = tmp_path / "round1.npz"
    metadata_path = _write_sidecar(
        path,
        split="train",
        method=METHODS[0],
        iteration=1,
        episodes=(1, 2),
        rows=4,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["head_metadata"]["dagger_iteration"] = 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="must use a round 0 head"):
        _load_cache_receipt(
            path,
            expected_split="train",
            expected_method=METHODS[0],
            expected_protocol=PROTOCOL,
            observed_rows=4,
            observed_iterations=np.ones(4, dtype=np.int64),
        )

    metadata["head_metadata"]["dagger_iteration"] = 0
    metadata["head_metadata"]["method"] = METHODS[1]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="behavior head method mismatch"):
        _load_cache_receipt(
            path,
            expected_split="train",
            expected_method=METHODS[0],
            expected_protocol=PROTOCOL,
            observed_rows=4,
            observed_iterations=np.ones(4, dtype=np.int64),
        )


def test_round_zero_cache_forbids_behavior_head(tmp_path) -> None:
    path = tmp_path / "round0.npz"
    metadata_path = _write_sidecar(
        path,
        split="train",
        method=METHODS[0],
        iteration=0,
        episodes=(1, 2),
        rows=4,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["head_checkpoint"] = "unexpected.pt"
    metadata["head_metadata"] = {
        "method": METHODS[0],
        "dagger_iteration": 0,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="round 0 cache.*must not use"):
        _load_cache_receipt(
            path,
            expected_split="train",
            expected_method=METHODS[0],
            expected_protocol=PROTOCOL,
            observed_rows=4,
            observed_iterations=np.zeros(4, dtype=np.int64),
        )


def test_cache_receipt_rejects_multiple_rounds_in_one_npz(tmp_path) -> None:
    path = tmp_path / "mixed.npz"
    _write_sidecar(
        path,
        split="train",
        method=METHODS[0],
        iteration=0,
        episodes=(1, 2),
        rows=4,
    )
    with pytest.raises(ValueError, match="exactly sidecar DAgger round"):
        _load_cache_receipt(
            path,
            expected_split="train",
            expected_method=METHODS[0],
            expected_protocol=PROTOCOL,
            observed_rows=4,
            observed_iterations=np.asarray([0, 0, 1, 1], dtype=np.int64),
        )


def _receipt(
    *, method: str, iteration: int, panel: tuple[tuple[str, str, str], ...]
) -> CacheReceipt:
    return CacheReceipt(
        path=Path(f"{method}.{iteration}.npz"),
        split="train",
        method=method,
        dagger_iteration=iteration,
        rows=1,
        episode_panel=panel,
        head_checkpoint=None if iteration == 0 else "head.pt",
        head_iteration=None if iteration == 0 else iteration - 1,
    )


def test_matched_collection_requires_continuous_rounds_and_shared_panel() -> None:
    panel = (("1", "2", "3"),)
    receipts = {
        method: (
            _receipt(method=method, iteration=0, panel=panel),
            _receipt(method=method, iteration=1, panel=panel),
        )
        for method in METHODS
    }
    assert _validate_matched_collection(
        receipts, split="train", methods=METHODS
    ) == (0, 1)

    discontinuous = dict(receipts)
    discontinuous[METHODS[0]] = (
        receipts[METHODS[0]][0],
        _receipt(method=METHODS[0], iteration=2, panel=panel),
    )
    with pytest.raises(ValueError, match="continuous from zero"):
        _validate_matched_collection(
            discontinuous, split="train", methods=METHODS
        )

    mismatched_rounds = dict(receipts)
    mismatched_rounds[METHODS[1]] = (receipts[METHODS[1]][0],)
    with pytest.raises(ValueError, match="different DAgger round sets"):
        _validate_matched_collection(
            mismatched_rounds, split="train", methods=METHODS
        )

    mismatched_panel = dict(receipts)
    mismatched_panel[METHODS[2]] = (
        receipts[METHODS[2]][0],
        _receipt(method=METHODS[2], iteration=1, panel=(("9", "9", "9"),)),
    )
    with pytest.raises(ValueError, match="different initial physical episode panels"):
        _validate_matched_collection(
            mismatched_panel, split="train", methods=METHODS
        )

    different_row_counts = dict(receipts)
    different_row_counts[METHODS[2]] = (
        receipts[METHODS[2]][0],
        CacheReceipt(
            path=Path("different-row-count.npz"),
            split="train",
            method=METHODS[2],
            dagger_iteration=1,
            rows=99,
            episode_panel=panel,
            head_checkpoint="head.pt",
            head_iteration=0,
        ),
    )
    assert _validate_matched_collection(
        different_row_counts, split="train", methods=METHODS
    ) == (0, 1)


def test_main_aggregates_rounds_and_writes_loadable_latest_heads(tmp_path) -> None:
    caches: dict[str, dict[str, list[str]]] = {
        "train": {},
        "development": {},
    }
    for method_index, method in enumerate(METHODS):
        train_paths = []
        for iteration in (0, 1):
            path = tmp_path / f"{method}_train_round{iteration}.npz"
            _write_cache(
                path,
                split="train",
                method=method,
                method_index=method_index,
                iteration=iteration,
                episodes=(0, 1),
            )
            train_paths.append(str(path))
        development_paths = []
        for iteration in (0, 1):
            development_path = (
                tmp_path / f"{method}_development_round{iteration}.npz"
            )
            _write_cache(
                development_path,
                split="development",
                method=method,
                method_index=method_index,
                iteration=iteration,
                episodes=(10, 11),
            )
            development_paths.append(str(development_path))
        caches["train"][method] = train_paths
        caches["development"][method] = development_paths

    output = tmp_path / "output"
    config = {
        "protocol": PROTOCOL,
        "methods": list(METHODS),
        "caches": caches,
        "training": {
            "steps": 2,
            "batch_size": 6,
            "hidden_dim": 12,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "seeds": [17],
            "bootstrap_samples": 100,
            "bootstrap_seed": 31,
            "output_dir": str(output),
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["--config", str(config_path), "--device", "cpu"]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["optimizer_steps"] == 2
    assert report["training"][METHODS[0]]["dagger_iterations"] == [0, 1]
    assert report["gripper"]["changed_by_any_head"] is False
    assert all(
        comparison["paired"] is True
        for comparison in report["comparisons"].values()
    )
    for method in METHODS:
        checkpoint = output / f"{method}_seed17_latest.pt"
        _, metadata = load_posterior_dagger_checkpoint(
            checkpoint,
            expected_method=method,
            expected_seed=17,
            expected_steps=2,
            expected_dagger_iteration=1,
        )
        assert metadata["steps"] == 2
