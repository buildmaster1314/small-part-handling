from __future__ import annotations

from copy import deepcopy
import math

import pytest

from fcpg.lewm_anchor_ablation import (
    PARAMETER_COMPATIBLE_TRIPLET_MODES,
    TRIPLET_CORRECTED,
    TRIPLET_NOMINAL,
)
from fcpg.lewm_matrix_analysis import (
    nominal_corrected_2x2_decomposition,
    summarize_paired_matrix,
)


def _records() -> list[dict[str, object]]:
    records = []
    for train_index, train_mode in enumerate(PARAMETER_COMPATIBLE_TRIPLET_MODES):
        for eval_index, eval_mode in enumerate(PARAMETER_COMPATIBLE_TRIPLET_MODES):
            scale = 1.0 - 0.05 * train_index - 0.02 * eval_index
            blocks = []
            for source, task, episode, baseline in (
                (0, "small", 0, 2.0),
                (0, "small", 1, 2.0),
                (1, "large", 0, 20.0),
                (1, "large", 1, 20.0),
                (1, "large", 2, 20.0),
            ):
                blocks.append(
                    {
                        "source_index": source,
                        "task_name": task,
                        "episode_id": episode,
                        "teacher_forced": {"predicted_mse": baseline * scale},
                    }
                )
            records.append(
                {
                    "train_mode": train_mode,
                    "eval_mode": eval_mode,
                    "episode_blocks": blocks,
                }
            )
    return records


def _cell(
    summary: dict[str, object], train_mode: str, eval_mode: str
) -> dict[str, object]:
    return summary["train_by_eval"][train_mode][eval_mode]  # type: ignore[index,return-value]


def test_summary_is_paired_task_equal_and_reproducible() -> None:
    records = _records()
    first = summarize_paired_matrix(records, bootstrap_samples=200, bootstrap_seed=9)
    second = summarize_paired_matrix(
        list(reversed(records)), bootstrap_samples=200, bootstrap_seed=9
    )

    assert first == second
    assert first["episode_count"] == 5
    assert first["task_count"] == 2
    assert first["episodes_per_task"] == {"large": 3, "small": 2}
    assert first["baseline"]["task_macro_mse"] == pytest.approx(11.0)  # type: ignore[index]

    corrected = _cell(first, TRIPLET_CORRECTED, TRIPLET_CORRECTED)
    # Both task scales improve by 7%, so task-equal macro relative improvement
    # is 7%, independent of their MSE scale and episode counts.
    assert corrected["task_macro_mse"] == pytest.approx(10.23)
    assert corrected["macro_relative_improvement"] == pytest.approx(0.07)
    assert corrected["absolute_delta"] == pytest.approx(0.77)
    assert corrected["relative_improvement_95_ci"] == pytest.approx([0.07, 0.07])
    assert corrected["probability_improvement_gt_zero"] == 1.0

    baseline = _cell(first, TRIPLET_NOMINAL, TRIPLET_NOMINAL)
    assert baseline["macro_relative_improvement"] == 0.0
    assert baseline["relative_improvement_95_ci"] == [0.0, 0.0]
    assert baseline["probability_improvement_gt_zero"] == 0.0


def test_task_relative_macro_is_not_a_pooled_episode_ratio() -> None:
    records = _records()
    target = next(
        record
        for record in records
        if record["train_mode"] == TRIPLET_CORRECTED
        and record["eval_mode"] == TRIPLET_NOMINAL
    )
    for block in target["episode_blocks"]:  # type: ignore[index]
        teacher = block["teacher_forced"]  # type: ignore[index]
        if block["task_name"] == "small":  # type: ignore[index]
            teacher["predicted_mse"] = 1.0  # type: ignore[index]
        else:
            teacher["predicted_mse"] = 18.0  # type: ignore[index]

    summary = summarize_paired_matrix(records, bootstrap_samples=20, bootstrap_seed=1)
    cell = _cell(summary, TRIPLET_CORRECTED, TRIPLET_NOMINAL)
    assert cell["macro_relative_improvement"] == pytest.approx((0.5 + 0.1) / 2)
    assert cell["absolute_delta"] == pytest.approx((1.0 + 2.0) / 2)


def test_relative_bootstrap_resamples_paired_baseline_denominator() -> None:
    records = _records()
    baseline = records[0]
    target = next(
        record
        for record in records
        if record["train_mode"] == TRIPLET_CORRECTED
        and record["eval_mode"] == TRIPLET_CORRECTED
    )
    small_baseline = [1.0, 9.0]
    small_cell = [0.0, 10.0]
    small_index = 0
    for baseline_block, target_block in zip(
        baseline["episode_blocks"], target["episode_blocks"], strict=True  # type: ignore[arg-type]
    ):
        if baseline_block["task_name"] == "small":  # type: ignore[index]
            baseline_block["teacher_forced"]["predicted_mse"] = small_baseline[  # type: ignore[index]
                small_index
            ]
            target_block["teacher_forced"]["predicted_mse"] = small_cell[  # type: ignore[index]
                small_index
            ]
            small_index += 1
        else:
            target_block["teacher_forced"]["predicted_mse"] = baseline_block[  # type: ignore[index]
                "teacher_forced"
            ]["predicted_mse"]

    summary = summarize_paired_matrix(
        records, bootstrap_samples=4_000, bootstrap_seed=17
    )
    cell = _cell(summary, TRIPLET_CORRECTED, TRIPLET_CORRECTED)

    assert cell["macro_relative_improvement"] == pytest.approx(0.0)
    # Paired resamples containing the first small-task episode twice have a
    # 100% task improvement, while duplicates of the second have -1/9.  After
    # equal-task macro averaging these become +0.5 and -1/18 respectively.
    low, high = cell["relative_improvement_95_ci"]
    assert low == pytest.approx(-1.0 / 18.0)
    assert high == pytest.approx(0.5)


def test_paired_statistics_preserve_degradation_direction() -> None:
    records = _records()
    baseline = records[0]
    target = records[-1]
    for baseline_block, target_block in zip(
        baseline["episode_blocks"], target["episode_blocks"], strict=True  # type: ignore[arg-type]
    ):
        target_block["teacher_forced"]["predicted_mse"] = (  # type: ignore[index]
            1.2 * baseline_block["teacher_forced"]["predicted_mse"]  # type: ignore[index]
        )

    summary = summarize_paired_matrix(records, bootstrap_samples=100, bootstrap_seed=3)
    cell = _cell(summary, target["train_mode"], target["eval_mode"])  # type: ignore[arg-type]
    assert cell["macro_relative_improvement"] == pytest.approx(-0.2)
    assert cell["relative_improvement_95_ci"] == pytest.approx([-0.2, -0.2])
    assert cell["probability_improvement_gt_zero"] == 0.0


def test_nominal_corrected_2x2_point_decomposition() -> None:
    result = nominal_corrected_2x2_decomposition(
        {
            TRIPLET_NOMINAL: {
                TRIPLET_NOMINAL: 10.0,
                TRIPLET_CORRECTED: 8.0,
            },
            TRIPLET_CORRECTED: {
                TRIPLET_NOMINAL: 9.0,
                TRIPLET_CORRECTED: 6.0,
            },
        }
    )

    assert result["fixed_train_input_effect"] == pytest.approx(
        {TRIPLET_NOMINAL: 2.0, TRIPLET_CORRECTED: 3.0}
    )
    assert result["fixed_eval_training_effect"] == pytest.approx(
        {TRIPLET_NOMINAL: 1.0, TRIPLET_CORRECTED: 2.0}
    )
    assert result["matched_gain"] == pytest.approx(4.0)
    assert result["raw_interaction"] == pytest.approx(-1.0)


def test_rejects_missing_and_duplicate_matrix_cells() -> None:
    records = _records()
    with pytest.raises(ValueError, match="incomplete LeWM 4x4 matrix"):
        summarize_paired_matrix(records[:-1], bootstrap_samples=5)
    with pytest.raises(ValueError, match="duplicate matrix cell"):
        summarize_paired_matrix([*records, deepcopy(records[0])], bootstrap_samples=5)


def test_rejects_episode_key_mismatch_and_duplicate_episode() -> None:
    records = _records()
    records[-1]["episode_blocks"] = records[-1]["episode_blocks"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="episode keys differ"):
        summarize_paired_matrix(records, bootstrap_samples=5)

    records = _records()
    records[-1]["episode_blocks"].append(  # type: ignore[union-attr]
        deepcopy(records[-1]["episode_blocks"][0])  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="duplicate episode key"):
        summarize_paired_matrix(records, bootstrap_samples=5)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_episode_mse(value: float) -> None:
    records = _records()
    records[0]["episode_blocks"][0]["teacher_forced"]["predicted_mse"] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="must be finite"):
        summarize_paired_matrix(records, bootstrap_samples=5)


def test_rejects_nonpositive_baseline_task_mean() -> None:
    records = _records()
    baseline = records[0]
    for block in baseline["episode_blocks"]:  # type: ignore[index]
        if block["task_name"] == "small":  # type: ignore[index]
            block["teacher_forced"]["predicted_mse"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="baseline task mean must be positive"):
        summarize_paired_matrix(records, bootstrap_samples=5)


def test_rejects_zero_baseline_episode_for_relative_bootstrap() -> None:
    records = _records()
    records[0]["episode_blocks"][0]["teacher_forced"]["predicted_mse"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="every baseline episode MSE must be positive"):
        summarize_paired_matrix(records, bootstrap_samples=5)


def test_large_finite_mse_uses_stable_means_and_ratio_overflow_fails() -> None:
    records = _records()
    target = records[-1]
    for block in target["episode_blocks"]:  # type: ignore[index]
        block["teacher_forced"]["predicted_mse"] = 1e308  # type: ignore[index]
    summary = summarize_paired_matrix(records, bootstrap_samples=5)
    assert math.isfinite(
        _cell(summary, target["train_mode"], target["eval_mode"])[  # type: ignore[arg-type]
            "task_macro_mse"
        ]
    )

    records = _records()
    for block in records[0]["episode_blocks"]:  # type: ignore[index]
        block["teacher_forced"]["predicted_mse"] = 1e-308  # type: ignore[index]
    for block in records[-1]["episode_blocks"]:  # type: ignore[index]
        block["teacher_forced"]["predicted_mse"] = 1e308  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        summarize_paired_matrix(records, bootstrap_samples=5)


def test_2x2_rejects_nonfinite_derived_interaction() -> None:
    with pytest.raises(ValueError, match="raw 2x2 interaction must be finite"):
        nominal_corrected_2x2_decomposition(
            {
                TRIPLET_NOMINAL: {
                    TRIPLET_NOMINAL: 1e308,
                    TRIPLET_CORRECTED: 0.0,
                },
                TRIPLET_CORRECTED: {
                    TRIPLET_NOMINAL: 0.0,
                    TRIPLET_CORRECTED: 1e308,
                },
            }
        )


@pytest.mark.parametrize(
    ("samples", "seed", "message"),
    [(0, 0, "at least 1"), (True, 0, "must be an integer"), (5, -1, "at least 0")],
)
def test_rejects_invalid_bootstrap_contract(
    samples: object, seed: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        summarize_paired_matrix(
            _records(),
            bootstrap_samples=samples,  # type: ignore[arg-type]
            bootstrap_seed=seed,  # type: ignore[arg-type]
        )
