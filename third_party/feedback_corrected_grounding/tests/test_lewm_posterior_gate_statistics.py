from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from fcpg.lewm_posterior_gate_statistics import (
    ESTIMANDS,
    EVENTS,
    PRIMARY_ESTIMAND,
    PRIMARY_EVENTS,
    RECOVERY_EVENTS,
    summarize_posterior_gate_statistics as _summarize_posterior_gate_statistics,
)


METHODS = (
    "adapter",
    "measurement_writeback",
    "fixed_beta",
    "no_lewm",
    "action_blind",
)
TASK_EPISODES = {
    "large": (("large-0", 10.0), ("large-1", 30.0)),
    "small": ((0, 1.0), (1, 3.0), (2, 5.0)),
}
PRIMARY_ADAPTER_RATIOS = {"large": 0.5, "small": 1.25}
EVENT_MULTIPLIERS = {"o0": 0.5, "o1": 1.0, "o2": 3.0, "r0": 2.0, "r1": 1.5}


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task, episodes in TASK_EPISODES.items():
        for episode_id, scale in episodes:
            for estimand in ESTIMANDS:
                estimand_multiplier = 0.75 if estimand == "post_update" else 1.0
                for event in EVENTS:
                    measurement = (
                        scale * estimand_multiplier * EVENT_MULTIPLIERS[event]
                    )
                    if estimand == PRIMARY_ESTIMAND and event in PRIMARY_EVENTS:
                        adapter = measurement * PRIMARY_ADAPTER_RATIOS[task]
                        fixed = adapter / 0.8
                        no_lewm = adapter / 0.75
                        action_blind = adapter / 0.70
                    elif estimand == "pre_prior" and event == "o0":
                        adapter = measurement * 1.01
                        fixed = measurement * 1.02
                        no_lewm = measurement * 1.03
                        action_blind = measurement * 1.04
                    else:
                        adapter = measurement * 0.9
                        fixed = measurement * 0.95
                        no_lewm = measurement * 1.1
                        action_blind = measurement * 1.15
                    values = {
                        "adapter": adapter,
                        "measurement_writeback": measurement,
                        "fixed_beta": fixed,
                        "no_lewm": no_lewm,
                        "action_blind": action_blind,
                    }
                    for anchor, is_direct in (("tip", True), ("wrist", False)):
                        for method in METHODS:
                            records.append(
                                {
                                    "task": task,
                                    "physical_episode_id": episode_id,
                                    "anchor": anchor,
                                    "estimand": estimand,
                                    "event": event,
                                    "method": method,
                                    "mse": values[method],
                                    "is_direct_anchor": is_direct,
                                }
                            )
    return records


def _clean_records() -> list[dict[str, object]]:
    records = deepcopy(_records())
    primary_scales = {"large": 30.0, "small": 4.5}
    for row in records:
        row["physical_episode_id"] = f"clean-{row['physical_episode_id']}"
        if row["estimand"] != "post_update":
            continue
        scale = primary_scales[str(row["task"])]
        if row["method"] == "measurement_writeback":
            row["mse"] = 0.0
        elif row["method"] == "adapter":
            row["mse"] = 0.01 * scale
        elif row["method"] == "fixed_beta":
            row["mse"] = 0.02 * scale
        elif row["method"] == "no_lewm":
            row["mse"] = 0.03 * scale
        elif row["method"] == "action_blind":
            row["mse"] = 0.04 * scale
    return records


def summarize_posterior_gate_statistics(
    records: list[dict[str, object]],
    **kwargs: object,
) -> dict[str, object]:
    clean_records = kwargs.pop("clean_records", _clean_records())
    return _summarize_posterior_gate_statistics(
        records,
        clean_records,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _constant_excess_normalized_bootstrap(
    *,
    excess_by_task: dict[str, float],
    samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reproduce primary-scale resampling for constant safety excess fixtures."""

    rng = np.random.default_rng(seed)
    macro = np.zeros(samples, dtype=np.float64)
    per_task: dict[str, np.ndarray] = {}
    for task, episodes in TASK_EPISODES.items():
        primary_measurement = np.asarray(
            [1.5 * scale for _, scale in episodes], dtype=np.float64
        )
        draws = rng.integers(
            0,
            len(primary_measurement),
            size=(samples, len(primary_measurement)),
        )
        task_samples = excess_by_task[task] / primary_measurement[draws].mean(
            axis=1
        )
        per_task[task] = task_samples
        macro += task_samples / len(TASK_EPISODES)
    return macro, per_task


def test_primary_is_o1_o2_episode_then_task_equal_ratio() -> None:
    summary = summarize_posterior_gate_statistics(
        _records(), bootstrap_samples=200, bootstrap_seed=7
    )
    primary = summary["primary"]

    # Per-task Adapter/measurement ratios are 0.5 and 1.25 despite their very
    # different MSE scales and episode counts.
    assert primary["theta"] == pytest.approx(0.125)
    assert primary["theta_95_ci"] == pytest.approx([0.125, 0.125])
    assert primary["adapter_to_measurement_ratio"] == pytest.approx(0.875)
    assert primary["measurement_mse"] == pytest.approx(17.25)
    assert primary["adapter_mse"] == pytest.approx(10.3125)
    assert primary["estimand"] == "post_update"
    assert primary["events"] == ["o1", "o2"]

    assert summary["comparators"]["adapter_vs_fixed"]["fixed_beta"][
        "relative_improvement"
    ] == pytest.approx(0.2)
    assert summary["comparators"]["adapter_vs_no_lewm"][
        "relative_improvement"
    ] == pytest.approx(0.25)
    assert set(summary["comparators"]) == {
        "adapter_vs_fixed",
        "adapter_vs_no_lewm",
    }
    assert summary["methods"]["no_lewm"] == "no_lewm"
    assert "action_blind" in summary["secondary_events"]["o0"]["post_update"][
        "method_task_macro_mse"
    ]
    assert summary["coverage"]["complete_paired_support"] is True
    assert summary["coverage"]["record_count"] == 5 * 2 * 2 * 5 * 5
    assert summary["coverage"]["physical_episode_count"] == 5


def test_bootstrap_recomputes_task_denominators_from_stratified_episode_draws() -> None:
    records = _records()
    episode_ratios = {
        ("large", "large-0"): 0.2,
        ("large", "large-1"): 0.8,
        ("small", 0): 0.5,
        ("small", 1): 1.25,
        ("small", 2): 2.0,
    }
    for row in records:
        if (
            row["estimand"] == PRIMARY_ESTIMAND
            and row["event"] in PRIMARY_EVENTS
            and row["method"] == "adapter"
        ):
            scale = dict(TASK_EPISODES[str(row["task"])])[row["physical_episode_id"]]
            measurement = 0.75 * scale * EVENT_MULTIPLIERS[str(row["event"])]
            row["mse"] = measurement * episode_ratios[
                (str(row["task"]), row["physical_episode_id"])
            ]

    samples = 500
    seed = 17
    summary = summarize_posterior_gate_statistics(
        records, bootstrap_samples=samples, bootstrap_seed=seed
    )

    large_measurement = np.asarray([15.0, 45.0])
    large_adapter = np.asarray([3.0, 36.0])
    small_measurement = np.asarray([1.5, 4.5, 7.5])
    small_adapter = np.asarray([0.75, 5.625, 15.0])
    rng = np.random.default_rng(seed)
    large_draws = rng.integers(0, 2, size=(samples, 2))
    small_draws = rng.integers(0, 3, size=(samples, 3))
    bootstrap_theta = 1.0 - 0.5 * (
        large_adapter[large_draws].mean(axis=1)
        / large_measurement[large_draws].mean(axis=1)
        + small_adapter[small_draws].mean(axis=1)
        / small_measurement[small_draws].mean(axis=1)
    )

    expected_point = 1.0 - 0.5 * (39.0 / 60.0 + 21.375 / 13.5)
    assert summary["primary"]["theta"] == pytest.approx(expected_point)
    assert summary["primary"]["theta_95_ci"] == pytest.approx(
        np.quantile(bootstrap_theta, [0.025, 0.975])
    )


def test_o1_o2_stay_bundled_inside_whole_episode_bootstrap() -> None:
    records = [row for row in _records() if row["task"] == "large"]
    for row in records:
        if row["estimand"] != PRIMARY_ESTIMAND or row["event"] not in PRIMARY_EVENTS:
            continue
        if row["method"] == "measurement_writeback":
            row["mse"] = 10.0
        elif row["method"] == "adapter":
            first_episode = row["physical_episode_id"] == "large-0"
            first_event = row["event"] == "o1"
            row["mse"] = 0.0 if first_episode == first_event else 20.0
        elif row["method"] == "fixed_beta":
            row["mse"] = 12.0
        elif row["method"] == "no_lewm":
            row["mse"] = 14.0

    summary = summarize_posterior_gate_statistics(
        records,
        clean_records=[row for row in _clean_records() if row["task"] == "large"],
        bootstrap_samples=1_000,
        bootstrap_seed=19,
    )
    assert summary["primary"]["theta"] == 0.0
    assert summary["primary"]["theta_95_ci"] == [0.0, 0.0]


def test_task_ratio_and_clean_excess_have_one_sided_upper_bounds() -> None:
    samples = 100
    seed = 23
    summary = summarize_posterior_gate_statistics(
        _records(), bootstrap_samples=samples, bootstrap_seed=seed
    )
    safety = summary["task_safety_ratios"]
    assert safety["large"][
        "adapter_to_measurement_ratio_one_sided_95_upper"
    ] == pytest.approx(0.5)
    assert safety["small"][
        "adapter_to_measurement_ratio_one_sided_95_upper"
    ] == pytest.approx(1.25)

    clean = summary["clean_safety"]
    assert clean["cohort"] == "explicit_independent_clean_records"
    assert clean["estimand"] == "post_update"
    assert clean["events"] == list(EVENTS)
    assert clean["control_macro_mse"] == 0.0
    assert clean["normalized_excess_mse"] == pytest.approx(0.01)
    clean_bootstrap, clean_by_task = _constant_excess_normalized_bootstrap(
        excess_by_task={"large": 0.3, "small": 0.045},
        samples=samples,
        seed=seed,
    )
    assert clean[
        "normalized_excess_mse_one_sided_95_upper"
    ] == pytest.approx(np.quantile(clean_bootstrap, 0.95))
    assert clean["per_task"]["large"][
        "normalized_excess_mse_one_sided_95_upper"
    ] == pytest.approx(np.quantile(clean_by_task["large"], 0.95))


def test_formal_recovery_safety_uses_r1_only_and_primary_scaled_excess() -> None:
    records = _records()
    primary_scales = {"large": 30.0, "small": 4.5}
    for row in records:
        if row["estimand"] != "post_update" or row["event"] not in ("r0", "r1"):
            continue
        if row["method"] == "measurement_writeback":
            row["mse"] = 0.0
        elif row["method"] == "adapter":
            multiplier = 4.0 if row["event"] == "r0" else 0.04
            row["mse"] = multiplier * primary_scales[str(row["task"])]

    samples = 100
    seed = 31
    summary = summarize_posterior_gate_statistics(
        records, bootstrap_samples=samples, bootstrap_seed=seed
    )
    recovery = summary["recovery_safety"]
    assert recovery["events"] == ["r1"]
    assert recovery["control_macro_mse"] == 0.0
    assert recovery["excess_mse"] > 0.0
    assert recovery["normalized_excess_mse"] == pytest.approx(0.04)
    recovery_bootstrap, _ = _constant_excess_normalized_bootstrap(
        excess_by_task={"large": 1.2, "small": 0.18},
        samples=samples,
        seed=seed,
    )
    assert recovery[
        "normalized_excess_mse_one_sided_95_upper"
    ] == pytest.approx(np.quantile(recovery_bootstrap, 0.95))
    assert summary["secondary_events"]["r0"]["post_update"][
        "adapter_vs_measurement"
    ]["control_macro_mse"] == 0.0
    assert "adapter_vs_measurement_normalized_safety" not in summary[
        "secondary_events"
    ]["r0"]["post_update"]
    assert summary["secondary_events"]["r1"]["post_update"][
        "adapter_vs_measurement_normalized_safety"
    ]["normalized_excess_mse"] == pytest.approx(0.04)


def test_indirect_safety_averages_anchors_and_episodes_then_weights_tasks() -> None:
    records = _records()
    elbow_rows = []
    for row in records:
        if row["anchor"] != "wrist":
            continue
        copy = deepcopy(row)
        copy["anchor"] = "elbow"
        elbow_rows.append(copy)
    records.extend(elbow_rows)

    indirect_excess = {
        "large": {"wrist": 0.0, "elbow": 2.0},
        "small": {"wrist": 2.0, "elbow": 6.0},
    }
    for row in records:
        if row["estimand"] != "post_update":
            continue
        anchor = str(row["anchor"])
        method = str(row["method"])
        if anchor == "tip":
            if method == "measurement_writeback":
                row["mse"] = 10.0
            elif method == "adapter":
                row["mse"] = 1_010.0
        elif method == "measurement_writeback":
            row["mse"] = 5.0
        elif method == "adapter":
            row["mse"] = 5.0 + indirect_excess[str(row["task"])][anchor]

    summary = summarize_posterior_gate_statistics(
        records, bootstrap_samples=40, bootstrap_seed=37
    )
    indirect = summary["indirect_safety"]

    # Large contributes (0 + 2) / 2 / 10 = 0.1 and small contributes
    # (2 + 6) / 2 / 10 = 0.4, regardless of their two versus three episodes.
    assert indirect["normalized_excess_mse"] == pytest.approx(0.25)
    assert indirect["excess_mse"] == pytest.approx(2.5)
    assert indirect["candidate_macro_mse"] == pytest.approx(7.5)
    assert indirect["control_macro_mse"] == pytest.approx(5.0)
    assert indirect["per_task"]["large"]["normalized_excess_mse"] == pytest.approx(
        0.1
    )
    assert indirect["per_task"]["small"]["normalized_excess_mse"] == pytest.approx(
        0.4
    )


def test_indirect_safety_uses_paired_physical_episode_bootstrap() -> None:
    records = _records()
    scales = {
        ("large", "large-0"): 10.0,
        ("large", "large-1"): 30.0,
        ("small", 0): 5.0,
        ("small", 1): 10.0,
        ("small", 2): 20.0,
    }
    excesses = {
        ("large", "large-0"): 1.0,
        ("large", "large-1"): 9.0,
        ("small", 0): 2.0,
        ("small", 1): 4.0,
        ("small", 2): 8.0,
    }
    for row in records:
        if row["estimand"] != "post_update":
            continue
        key = (str(row["task"]), row["physical_episode_id"])
        if row["anchor"] == "tip" and row["event"] in PRIMARY_EVENTS:
            if row["method"] == "measurement_writeback":
                row["mse"] = scales[key]
        elif row["anchor"] == "wrist":
            if row["method"] == "measurement_writeback":
                row["mse"] = 0.0
            elif row["method"] == "adapter":
                row["mse"] = excesses[key]

    samples = 500
    seed = 41
    summary = summarize_posterior_gate_statistics(
        records, bootstrap_samples=samples, bootstrap_seed=seed
    )

    rng = np.random.default_rng(seed)
    expected = np.zeros(samples, dtype=np.float64)
    for task, episodes in TASK_EPISODES.items():
        task_scales = np.asarray(
            [scales[(task, episode_id)] for episode_id, _ in episodes]
        )
        task_excesses = np.asarray(
            [excesses[(task, episode_id)] for episode_id, _ in episodes]
        )
        draws = rng.integers(0, len(episodes), size=(samples, len(episodes)))
        expected += (
            task_excesses[draws].mean(axis=1)
            / task_scales[draws].mean(axis=1)
            / len(TASK_EPISODES)
        )

    indirect = summary["indirect_safety"]
    assert indirect["normalized_excess_mse"] == pytest.approx(0.325)
    assert indirect[
        "normalized_excess_mse_one_sided_95_upper"
    ] == pytest.approx(np.quantile(expected, 0.95))


def test_indirect_safety_schema_and_one_sided_ci_include_boundary() -> None:
    records = _records()
    for row in records:
        if row["estimand"] != "post_update":
            continue
        if row["anchor"] == "tip" and row["event"] in PRIMARY_EVENTS:
            if row["method"] == "measurement_writeback":
                row["mse"] = 10.0
        elif row["anchor"] == "wrist":
            if row["method"] == "measurement_writeback":
                row["mse"] = 1.0
            elif row["method"] == "adapter":
                row["mse"] = 1.2

    indirect = summarize_posterior_gate_statistics(
        records, bootstrap_samples=50, bootstrap_seed=43
    )["indirect_safety"]
    assert indirect["estimand"] == "post_update"
    assert indirect["events"] == list(EVENTS)
    assert indirect["anchor_scope"] == "all_except_primary_direct"
    assert indirect["orientation"] == "candidate_mse_minus_control_mse"
    assert indirect["normalization"] == "each_task_primary_measurement_mse"
    assert indirect["normalized_excess_mse"] == pytest.approx(0.02)
    assert indirect[
        "normalized_excess_mse_one_sided_95_upper"
    ] == pytest.approx(0.02)


def test_r0_stays_diagnostic_while_r1_is_the_only_formal_recovery_event() -> None:
    summary = summarize_posterior_gate_statistics(
        _records(), bootstrap_samples=20
    )
    assert RECOVERY_EVENTS == ("r1",)
    assert list(summary["secondary_events"]) == ["o0", "r0", "r1"]
    assert summary["secondary_events"]["r0"]["pre_prior"][
        "primary_eligible"
    ] is False
    assert summary["secondary_events"]["r1"]["post_update"][
        "event"
    ] == "r1"

    with pytest.raises(ValueError, match="recovery events cannot be primary"):
        summarize_posterior_gate_statistics(
            _records(), primary_events=("o1", "r1"), bootstrap_samples=10
        )
    with pytest.raises(ValueError, match="primary_estimand must be exactly"):
        summarize_posterior_gate_statistics(
            _records(), primary_estimand="pre_prior", bootstrap_samples=10
        )


def test_clean_safety_cannot_be_inferred_from_o0_or_overlapping_records() -> None:
    records = _records()
    for row in records:
        if row["event"] == "o0" and row["method"] == "adapter":
            row["mse"] = 1.0e9
    summary = summarize_posterior_gate_statistics(records, bootstrap_samples=10)
    assert summary["clean_safety"]["normalized_excess_mse"] == pytest.approx(0.01)

    with pytest.raises(ValueError, match="physically disjoint cohort"):
        summarize_posterior_gate_statistics(
            _records(), clean_records=_records(), bootstrap_samples=10
        )
    with pytest.raises(TypeError, match="clean_records"):
        _summarize_posterior_gate_statistics(
            _records(), bootstrap_samples=10
        )


def test_fixed_seed_is_record_order_independent() -> None:
    records = _records()
    for row in records:
        if (
            row["task"] == "large"
            and row["physical_episode_id"] == "large-0"
            and row["estimand"] == PRIMARY_ESTIMAND
            and row["event"] == "o1"
            and row["method"] == "adapter"
        ):
            row["mse"] = 1.0
    first = summarize_posterior_gate_statistics(
        records, bootstrap_samples=300, bootstrap_seed=29
    )
    second = summarize_posterior_gate_statistics(
        list(reversed(records)), bootstrap_samples=300, bootstrap_seed=29
    )
    assert first == second


def test_requires_complete_estimand_event_method_panel() -> None:
    records = _records()
    records = [
        row
        for row in records
        if not (
            row["task"] == "large"
            and row["physical_episode_id"] == "large-0"
            and row["estimand"] == "post_update"
            and row["event"] == "r1"
            and row["method"] == "fixed_beta"
        )
    ]
    with pytest.raises(ValueError, match="incomplete estimand/event/method"):
        summarize_posterior_gate_statistics(records, bootstrap_samples=10)


def test_rejects_duplicate_records() -> None:
    records = _records()
    records.append(deepcopy(records[0]))
    with pytest.raises(ValueError, match="duplicate posterior Gate statistic"):
        summarize_posterior_gate_statistics(records, bootstrap_samples=10)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_mse(value: float) -> None:
    records = _records()
    records[0]["mse"] = value
    with pytest.raises(ValueError, match="mse must be finite"):
        summarize_posterior_gate_statistics(records, bootstrap_samples=10)


def test_rejects_negative_mse() -> None:
    records = _records()
    records[0]["mse"] = -1.0
    with pytest.raises(ValueError, match="mse must be non-negative"):
        summarize_posterior_gate_statistics(records, bootstrap_samples=10)


def test_direct_anchor_must_be_unambiguous_or_explicit() -> None:
    direct = [deepcopy(row) for row in _records() if row["anchor"] == "tip"]
    for row in direct:
        del row["is_direct_anchor"]
    indirect = []
    for row in direct:
        copy = deepcopy(row)
        copy["anchor"] = "wrist"
        copy["mse"] = float(copy["mse"]) * 100.0
        indirect.append(copy)
    records = direct + indirect

    with pytest.raises(ValueError, match="primary direct anchor is ambiguous"):
        summarize_posterior_gate_statistics(records, bootstrap_samples=10)

    summary = summarize_posterior_gate_statistics(
        records, direct_anchor="tip", bootstrap_samples=10
    )
    assert summary["primary"]["theta"] == pytest.approx(0.125)
    assert summary["coverage"]["direct_anchor_source"] == "argument"


def test_record_annotations_can_identify_one_direct_anchor_per_episode() -> None:
    summary = summarize_posterior_gate_statistics(
        _records(), bootstrap_samples=10
    )
    assert summary["primary"]["theta"] == pytest.approx(0.125)
    assert summary["coverage"]["direct_anchor_source"] == "record_annotation"


def test_ratio_denominators_must_be_positive_in_every_episode() -> None:
    records = _records()
    for row in records:
        if (
            row["task"] == "large"
            and row["physical_episode_id"] == "large-0"
            and row["estimand"] == PRIMARY_ESTIMAND
            and row["event"] in PRIMARY_EVENTS
            and row["method"] == "measurement_writeback"
        ):
            row["mse"] = 0.0
    with pytest.raises(
        ValueError, match="measurement_writeback episode MSE must be positive"
    ):
        summarize_posterior_gate_statistics(records, bootstrap_samples=10)


def test_custom_method_names_are_supported_but_all_roles_are_required() -> None:
    records = _records()
    clean_records = _clean_records()
    replacements = {
        "adapter": "bounded_adapter",
        "measurement_writeback": "raw_measurement",
        "fixed_beta": "fixed_075",
        "no_lewm": "hold_last_prior",
        "action_blind": "zero_action",
    }
    for row in records:
        row["method"] = replacements[str(row["method"])]
    for row in clean_records:
        row["method"] = replacements[str(row["method"])]

    summary = summarize_posterior_gate_statistics(
        records,
        clean_records=clean_records,
        adapter_method="bounded_adapter",
        measurement_method="raw_measurement",
        fixed_method="fixed_075",
        no_lewm_method="hold_last_prior",
        bootstrap_samples=10,
    )
    assert summary["primary"]["theta"] == pytest.approx(0.125)

    with pytest.raises(ValueError, match="required Gate methods are absent"):
        summarize_posterior_gate_statistics(records, bootstrap_samples=10)


def test_reports_every_preregistered_fixed_beta_without_selecting_one() -> None:
    records = _records()
    clean_records = _clean_records()
    for panel in (records, clean_records):
        extra = []
        for row in panel:
            if row["method"] != "fixed_beta":
                continue
            copy = deepcopy(row)
            copy["method"] = "fixed_beta_050"
            if (
                copy["estimand"] == PRIMARY_ESTIMAND
                and copy["event"] in PRIMARY_EVENTS
            ):
                task_ratio = PRIMARY_ADAPTER_RATIOS[str(copy["task"])]
                measurement = float(copy["mse"]) * 0.8 / task_ratio
                copy["mse"] = measurement * task_ratio / 0.5
            extra.append(copy)
        panel.extend(extra)

    summary = summarize_posterior_gate_statistics(
        records,
        clean_records=clean_records,
        fixed_methods=("fixed_beta", "fixed_beta_050"),
        bootstrap_samples=20,
    )
    fixed = summary["comparators"]["adapter_vs_fixed"]
    assert list(fixed) == ["fixed_beta", "fixed_beta_050"]
    assert fixed["fixed_beta"]["relative_improvement"] == pytest.approx(0.2)
    assert fixed["fixed_beta_050"]["relative_improvement"] == pytest.approx(0.5)
    assert "selected_fixed_method" not in summary["comparators"]
