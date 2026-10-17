from __future__ import annotations

from scripts.summarize_fcpg_lewm_factorial import episode_level_lewm_effects


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    effects = {
        0: {11: 1.0, 12: 1.0},
        1: {11: -1.0, 12: -1.0},
        2: {11: 0.0, 12: 1.0},
        3: {11: 0.0, 12: 0.0},
    }
    for episode, by_seed in effects.items():
        for model_seed, delta in by_seed.items():
            baseline = {
                "model_seed": model_seed,
                "physical_episode": episode,
                "method": "f1_l0",
                "drop_start": 6 + episode,
                "sustained_success": 0.0,
                "final_success": 0.0,
                "post_drop_success_fraction": 0.2,
                "post_drop_maximum_success_streak": 3.0,
                "reward_sum": 10.0,
            }
            full = dict(baseline)
            full["method"] = "f1_l1"
            for metric in (
                "sustained_success",
                "final_success",
                "post_drop_success_fraction",
                "post_drop_maximum_success_streak",
                "reward_sum",
            ):
                full[metric] = float(baseline[metric]) + delta
            rows.extend((baseline, full))
    return rows


def test_episode_level_effects_preserve_seed_consistency() -> None:
    report = episode_level_lewm_effects(_rows())
    counts = report["direction_counts"]["sustained_success"]
    assert report["model_seeds"] == [11, 12]
    assert report["physical_episode_count"] == 4
    assert counts == {
        "positive_all_seeds": 1,
        "negative_all_seeds": 1,
        "zero_all_seeds": 1,
        "mixed_across_seeds": 1,
    }
    assert report["episodes"][0]["mean_effects"]["reward_sum"] == 1.0


def test_episode_level_effects_reject_missing_full_cell() -> None:
    rows = _rows()
    rows = [
        row
        for row in rows
        if not (
            row["model_seed"] == 12
            and row["physical_episode"] == 0
            and row["method"] == "f1_l1"
        )
    ]
    try:
        episode_level_lewm_effects(rows)
    except ValueError as error:
        assert "requires F1L0 and F1L1" in str(error)
    else:
        raise AssertionError("episode audit accepted a missing Full cell")
