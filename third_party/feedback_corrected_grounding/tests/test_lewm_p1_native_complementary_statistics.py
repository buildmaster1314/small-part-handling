"""Tests for the frozen fresh P1-native complementary-observer statistics."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pytest

from fcpg.lewm_p1_native_complementary_statistics import (
    CANDIDATE,
    GEOMETRY_HOLD,
    HOLD_LAST,
    HORIZONS,
    MATCHED_NO_ACTION,
    MODEL_SEEDS,
    PROTOCOL,
    REQUIRED_METHODS,
    TASK,
    VISUAL_CV,
    WRONG_ACTIONS,
    summarize_complementary_fresh_gate,
)


DATASET_SHA = "a" * 64
CHECKPOINT_SHA = {
    15801: "1" * 64,
    15802: "2" * 64,
    15803: "3" * 64,
}


def _default_mse(seed: int, method: str, episode: int, horizon: int) -> float:
    del episode, horizon
    if method == CANDIDATE:
        return 0.50 if seed in (15801, 15802) else 0.75
    if method == GEOMETRY_HOLD:
        return 0.80
    if method == HOLD_LAST:
        return 0.90
    if method == VISUAL_CV:
        return 1.10
    if method == MATCHED_NO_ACTION:
        return 1.00
    if method in WRONG_ACTIONS:
        return 1.05
    raise AssertionError(method)


def _rows(
    value: Callable[[int, str, int, int], float] = _default_mse,
    *,
    episodes: int = 30,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for seed in MODEL_SEEDS:
        for method in REQUIRED_METHODS:
            for episode in range(episodes):
                for horizon in HORIZONS:
                    result.append(
                        {
                            "protocol": PROTOCOL,
                            "task": TASK,
                            "model_seed": seed,
                            "method": method,
                            "physical_episode_id": f"fresh-{episode:03d}",
                            "horizon": horizon,
                            "episode_horizon_mse": value(
                                seed, method, episode, horizon
                            ),
                            "statistical_unit": "physical_episode",
                            "dataset_sha256": DATASET_SHA,
                            "checkpoint_sha256": CHECKPOINT_SHA[seed],
                        }
                    )
    return result


def _summarize(
    rows: list[dict[str, object]],
    *,
    bootstrap_samples: int = 400,
    bootstrap_seed: int = 71,
) -> dict[str, object]:
    return summarize_complementary_fresh_gate(
        rows,
        expected_dataset_sha256=DATASET_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def test_two_passing_seeds_and_positive_remaining_seed_pass() -> None:
    result = _summarize(_rows())

    assert result["physical_episode_count"] == 30
    assert result["bootstrap"]["unit"] == "physical_episode"
    assert len(result["bootstrap"]["shared_draw_matrix_sha256"]) == 64
    assert [row["screen"]["passed"] for row in result["per_seed"]] == [
        True,
        True,
        False,
    ]
    gate = result["cross_seed_gate"]
    assert gate == {
        "minimum_passing_seeds": 2,
        "passing_seed_count": 2,
        "passing_seeds": [15801, 15802],
        "remaining_seeds": [15803],
        "remaining_seed_positive_vs_geometry_and_no_action": {"15803": True},
        "at_least_two_of_three_pass": True,
        "remaining_positive_rule_passed": True,
        "passed": True,
        "decision": "Fresh-Go",
    }


def test_bootstrap_is_shared_deterministic_and_h3_uses_q05() -> None:
    def varying(seed: int, method: str, episode: int, horizon: int) -> float:
        base = _default_mse(seed, method, episode, horizon)
        if seed == 15801 and method == CANDIDATE and horizon == 3:
            return 0.45 + 0.04 * episode
        return base

    samples = 800
    bootstrap_seed = 917
    first = _summarize(
        _rows(varying),
        bootstrap_samples=samples,
        bootstrap_seed=bootstrap_seed,
    )
    second = _summarize(
        _rows(varying),
        bootstrap_samples=samples,
        bootstrap_seed=bootstrap_seed,
    )
    assert first == second

    draws = np.random.default_rng(bootstrap_seed).integers(
        0, 30, size=(samples, 30), dtype=np.int64
    )
    candidate = np.asarray(
        [varying(15801, CANDIDATE, episode, 3) for episode in range(30)]
    )
    bootstrap_effects = []
    for method in (HOLD_LAST, GEOMETRY_HOLD, VISUAL_CV):
        reference = np.asarray(
            [varying(15801, method, episode, 3) for episode in range(30)]
        )
        bootstrap_effects.append(
            1.0 - candidate[draws].mean(axis=1) / reference[draws].mean(axis=1)
        )
    simultaneous = np.min(np.stack(bootstrap_effects, axis=1), axis=1)
    observed = first["per_seed"][0]["strongest_nonlearning_per_horizon"]["H3"]
    assert observed["simultaneous_one_sided_95_lower"] == pytest.approx(
        np.quantile(simultaneous, 0.05)
    )
    assert observed["simultaneous_bootstrap_95_ci"][0] == pytest.approx(
        np.quantile(simultaneous, 0.025)
    )


def test_h3_uses_simultaneous_strongest_baseline_not_easiest_baseline() -> None:
    def value(seed: int, method: str, episode: int, horizon: int) -> float:
        del episode
        if method == CANDIDATE:
            return 0.78 if horizon == 3 else 0.50
        if method == GEOMETRY_HOLD:
            return 0.75 if horizon == 3 else 0.80
        if method == HOLD_LAST:
            return 0.95
        if method == VISUAL_CV:
            return 1.10
        if method == MATCHED_NO_ACTION:
            return 1.00
        if method in WRONG_ACTIONS:
            return 1.05
        raise AssertionError((seed, method))

    result = _summarize(_rows(value))
    h3 = result["per_seed"][0]["strongest_nonlearning_per_horizon"]["H3"]
    effects = h3["per_baseline_relative_improvement"]
    assert effects[HOLD_LAST] > 0.0
    assert effects[VISUAL_CV] > 0.0
    assert effects[GEOMETRY_HOLD] < 0.0
    assert h3["simultaneous_strongest_relative_improvement"] == pytest.approx(
        effects[GEOMETRY_HOLD]
    )
    assert result["per_seed"][0]["screen"][
        "h3_point_nonnegative_and_one_sided_lower_at_least_minus_5pct"
    ] is False


@pytest.mark.parametrize(("positive_count", "expected"), [(5, True), (4, False)])
def test_requires_five_of_six_positive_horizons(
    positive_count: int, expected: bool
) -> None:
    positive = set(HORIZONS[:positive_count])

    def value(seed: int, method: str, episode: int, horizon: int) -> float:
        del seed, episode
        if method == CANDIDATE:
            return 0.55 if horizon in positive else 0.85
        return _default_mse(15801, method, 0, horizon)

    result = _summarize(_rows(value))
    for row in result["per_seed"]:
        assert row["screen"]["positive_horizon_count"] == positive_count
        assert row["screen"]["horizon_pass"] is expected


def test_every_wrong_action_control_is_a_required_gate() -> None:
    failed_wrong = WRONG_ACTIONS[-1]

    def value(seed: int, method: str, episode: int, horizon: int) -> float:
        base = _default_mse(seed, method, episode, horizon)
        if method == failed_wrong:
            return 0.49
        return base

    result = _summarize(_rows(value))
    for row in result["per_seed"]:
        key = f"factual_vs_{failed_wrong}"
        assert row["screen"]["wrong_action_pass"][key] is False
        assert row["screen"]["passed"] is False


def test_negative_remaining_seed_blocks_cross_seed_gate() -> None:
    def value(seed: int, method: str, episode: int, horizon: int) -> float:
        if seed == 15803 and method == CANDIDATE:
            return 0.85
        return _default_mse(seed, method, episode, horizon)

    result = _summarize(_rows(value))
    gate = result["cross_seed_gate"]
    assert gate["passing_seed_count"] == 2
    assert gate["remaining_seed_positive_vs_geometry_and_no_action"] == {
        "15803": False
    }
    assert gate["remaining_positive_rule_passed"] is False
    assert gate["decision"] == "Fresh-No-Go"


def test_rejects_duplicate_and_incomplete_panel_cells() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate Gate panel cell"):
        _summarize([*rows, copy.deepcopy(rows[0])])

    with pytest.raises(ValueError, match="incomplete fresh Gate panel"):
        _summarize(rows[:-1])


def test_rejects_provenance_drift_and_invalid_expected_checkpoint_panel() -> None:
    rows = _rows()
    rows[0]["dataset_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="dataset provenance mismatch"):
        _summarize(rows)

    rows = _rows()
    rows[0]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint provenance mismatch"):
        _summarize(rows)

    with pytest.raises(ValueError, match="exactly seeds"):
        summarize_complementary_fresh_gate(
            _rows(),
            expected_dataset_sha256=DATASET_SHA,
            expected_checkpoint_sha256={15801: "1" * 64},
            bootstrap_samples=100,
        )


def test_rejects_underpowered_panel_unexpected_method_and_invalid_mse() -> None:
    with pytest.raises(ValueError, match="at least 30 physical episodes"):
        _summarize(_rows(episodes=29))

    rows = _rows()
    rows[0]["method"] = "raw_lewm"
    with pytest.raises(ValueError, match="unexpected method"):
        _summarize(rows)

    rows = _rows()
    rows[0]["episode_horizon_mse"] = np.nan
    with pytest.raises(ValueError, match="finite and non-negative"):
        _summarize(rows)


def test_rejects_non_episode_unit_protocol_and_bootstrap_drift() -> None:
    rows = _rows()
    rows[0]["statistical_unit"] = "branch"
    with pytest.raises(ValueError, match="statistical unit"):
        _summarize(rows)

    rows = _rows()
    rows[0]["protocol"] = "other"
    with pytest.raises(ValueError, match="protocol mismatch"):
        _summarize(rows)

    with pytest.raises(ValueError, match="at least 100"):
        _summarize(_rows(), bootstrap_samples=99)
