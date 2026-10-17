from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fcpg.lewm_anchor_ablation import (
    PAIRED_INNOVATION,
    PAIRED_VISUAL,
    PARAMETER_COMPATIBLE_TRIPLET_MODES,
    TRIPLET_CORRECTED,
    TRIPLET_NOMINAL,
)
from fcpg.lewm_dev_gate import (
    _paired_effect,
    _sign_flip_pvalue,
    summarize_anchor_visual_dev_gate,
    validate_anchor_visual_dev_protocol,
)


TASKS = ("reach", "push", "pick-place", "drawer-open")
HORIZONS = (1, 3, 5)
TN_CELL = (TRIPLET_NOMINAL, TRIPLET_NOMINAL)
TC_CELL = (TRIPLET_CORRECTED, TRIPLET_CORRECTED)
PV_CELL = (PAIRED_VISUAL, PAIRED_VISUAL)
PI_CELL = (PAIRED_INNOVATION, PAIRED_INNOVATION)


def _protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "fcpg_lewm_anchor_visual_dev_selection_v1",
        "claim_boundary": {
            "development_only": True,
            "posthoc_pilot_selection_protocol": True,
            "fresh_multiseed_confirmation_required": True,
            "fresh_outer_holdout_episodes_required": True,
            "old_gate_l1_test_reuse_forbidden": True,
            "single_seed_cannot_restore_lewm_as_a_paper_contribution": True,
        },
        "bootstrap": {
            "confidence": 0.95,
            "samples": 10_000,
            "seed": 5120,
            "unit": "episode_block_within_task",
        },
        "primary": {
            "aggregation": "normalized_endpoint_error_equal_horizon_equal_task",
            "candidate": "paired_innovation/paired_innovation",
            "condition": "mild",
            "control": "triplet_corrected/triplet_corrected",
            "estimand": (
                "mean_task(1 - mean_episode,H candidate_ratio / "
                "mean_episode,H control_ratio)"
            ),
            "horizons": [1, 3, 5],
            "minimum_relative_improvement": 0.03,
            "paired_bootstrap_ci_lower_must_exceed": 0.0,
        },
        "clean_noninferiority": {
            "candidate": "paired_innovation/paired_innovation",
            "control": "triplet_corrected/triplet_corrected",
            "one_sided_confidence": 0.95,
            "ratio_upper_margin": 1.05,
        },
        "task_direction": {
            "minimum_positive_tasks": 3,
            "maximum_single_task_relative_degradation": 0.05,
            "task_count": 4,
        },
        "representation_noncollapse": {
            "absolute_threshold_scope": "overall_and_each_task",
            "absolute_effective_rank_minimum": 8.0,
            "absolute_participation_ratio_minimum": 4.0,
            "overall_candidate_to_control_minimum": 0.9,
            "per_task_candidate_to_control_minimum": 0.8,
        },
        "secondary_family": {
            "alpha": 0.05,
            "alternative": "greater",
            "contrasts": [
                "paired_visual/paired_visual_vs_triplet_corrected/triplet_corrected",
                "paired_innovation/paired_innovation_vs_paired_visual/paired_visual",
                "triplet_corrected/triplet_corrected_vs_triplet_nominal/triplet_nominal",
            ],
            "correction": "holm",
            "cannot_rescue_primary_failure": True,
            "test": "paired_episode_sign_flip",
        },
        "data": {
            "corrector_train_unseen_counts": {
                "clean": 12,
                "mild": 29,
                "total": 41,
            },
            "corrector_train_unseen_slice": (
                "config/lewm/splits/"
                "anchor_visual_corrector_train_unseen_seed5120.json"
            ),
            "corrector_train_unseen_slice_sha256": (
                "5209ef4dd7ee12b59ee6f09ff5dfa2a1e950c11006ab4abfa8ec05d5f26cfef7"
            ),
            "development_split": "validation",
            "episode_manifest": (
                "config/lewm/splits/gate_l1_camera_train_val_seed3072.json"
            ),
            "episode_manifest_sha256": (
                "f3448ef472bd93975126c203556094ad65229a714d58236a53b11b281768e299"
            ),
            "episode_set_sha256": (
                "3f6202c24d6795398ff05abba46b9f0b7902bfd91ae44516dc3482eff30251c5"
            ),
            "full_validation_counts": {
                "clean": 56,
                "mild": 112,
                "total": 168,
            },
            "mixed_corrector_exposure_disclosure_required": True,
        },
        "decision": {
            "go_requires": [
                "primary_superiority",
                "clean_noninferiority",
                "task_direction",
                "representation_noncollapse",
                "corrector_train_unseen_directional_agreement",
            ],
            "no_go_action": "stop_the_correction_aware_input_route",
            "pass_action": (
                "promote_only_to_a_fresh_multiseed_outer_holdout_confirmatory_candidate"
            ),
        },
        "training_contract": {
            "batch_size": 192,
            "latest_checkpoint_only": True,
            "optimizer_steps": 2000,
            "parameter_count": 19_152_816,
            "seed": 5120,
            "sigreg_weight": 0.02,
            "warmup_steps": 127,
        },
    }


def _representation() -> dict[str, Any]:
    row = {
        "samples": 32,
        "effective_rank": 10.0,
        "participation_ratio": 6.0,
    }
    return {
        "overall": {**row, "samples": 128},
        "per_task": {task: dict(row) for task in TASKS},
    }


def _matched_cell_ratio(
    cell: tuple[str, str], *, condition: str
) -> float:
    if cell == TN_CELL:
        return 1.2
    if cell == TC_CELL:
        return 1.0
    if cell == PV_CELL:
        return 0.95
    if cell == PI_CELL:
        return 1.02 if condition == "clean" else 0.9
    return 1.1


def _records() -> list[dict[str, Any]]:
    mild_unseen_counts = (8, 8, 7, 6)
    episode_specs_by_task: dict[str, tuple[tuple[int, str, bool], ...]] = {}
    for task_index, task in enumerate(TASKS):
        specs = [
            (episode_id, "clean", episode_id >= 11)
            for episode_id in range(14)
        ]
        specs.extend(
            (
                14 + mild_index,
                "mild",
                mild_index < mild_unseen_counts[task_index],
            )
            for mild_index in range(28)
        )
        episode_specs_by_task[task] = tuple(specs)
    records = []
    for train_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        for eval_mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
            cell = (train_mode, eval_mode)
            blocks = []
            for source_index, task in enumerate(TASKS):
                for episode_id, condition, unseen in episode_specs_by_task[task]:
                    ratio = _matched_cell_ratio(cell, condition=condition)
                    blocks.append(
                        {
                            "source_index": source_index,
                            "task_name": task,
                            "episode_id": episode_id,
                            "camera_condition": condition,
                            "corrector_train_unseen": unseen,
                            "num_sequences": 2,
                            "endpoints": {
                                str(horizon): {
                                    "predicted_mse": 10.0 * ratio,
                                    "hold_last_identity_mse": 10.0,
                                    "element_count": 30,
                                    "ratio": ratio,
                                }
                                for horizon in HORIZONS
                            },
                        }
                    )
            records.append(
                {
                    "train_mode": train_mode,
                    "eval_mode": eval_mode,
                    "episode_blocks": blocks,
                    "metrics": {"representation": _representation()},
                }
            )
    return records


def _record(
    records: list[dict[str, Any]], cell: tuple[str, str]
) -> dict[str, Any]:
    return next(
        record
        for record in records
        if (record["train_mode"], record["eval_mode"]) == cell
    )


def _set_ratios(
    records: list[dict[str, Any]],
    cell: tuple[str, str],
    ratio: float,
    *,
    condition: str | None = None,
    corrector_train_unseen: bool | None = None,
    tasks: tuple[str, ...] = TASKS,
) -> None:
    for block in _record(records, cell)["episode_blocks"]:
        if block["task_name"] not in tasks:
            continue
        if condition is not None and block["camera_condition"] != condition:
            continue
        if (
            corrector_train_unseen is not None
            and block["corrector_train_unseen"] is not corrector_train_unseen
        ):
            continue
        for endpoint in block["endpoints"].values():
            endpoint["predicted_mse"] = endpoint["hold_last_identity_mse"] * ratio
            endpoint["ratio"] = ratio


def test_complete_4x4_development_gate_success_path() -> None:
    records = _records()

    summary = summarize_anchor_visual_dev_gate(records, _protocol())

    assert len(records) == 16
    assert len({(row["train_mode"], row["eval_mode"]) for row in records}) == 16
    assert summary["schema_version"] == 1
    assert summary["development_only"] is True
    assert summary["primary"]["candidate_macro"] == pytest.approx(0.9)
    assert summary["primary"]["control_macro"] == pytest.approx(1.0)
    assert summary["primary"]["relative_improvement"] == pytest.approx(0.1)
    assert summary["primary"]["relative_improvement_95_ci"] == pytest.approx(
        [0.1, 0.1]
    )
    assert summary["clean_noninferiority"][
        "candidate_to_control_ratio_one_sided_95_upper"
    ] == pytest.approx(1.02)
    assert summary["task_direction"] == {
        "positive_tasks": 4,
        "task_count": 4,
        "worst_relative_improvement": pytest.approx(0.1),
        "passed": True,
    }
    assert summary["corrector_train_unseen_sensitivity"]["passed"] is True
    assert summary["representation_noncollapse"]["passed"] is True
    assert len(summary["secondary_holm_family"]) == 3
    assert all(
        row["alternative"] == "greater"
        for row in summary["secondary_holm_family"]
    )
    assert all(summary["conditions"].values())
    assert summary["passed"] is True
    assert summary["decision"] == "fresh_multiseed_outer_holdout_confirmation_candidate"



def test_zero_prediction_error_is_a_valid_endpoint() -> None:
    records = _records()
    _set_ratios(records, PI_CELL, 0.0, condition="mild")

    summary = summarize_anchor_visual_dev_gate(records, _protocol())

    assert summary["primary"]["candidate_macro"] == pytest.approx(0.0)
    assert summary["primary"]["relative_improvement"] == pytest.approx(1.0)


def test_sign_flip_uses_task_equal_relative_ratio_not_raw_delta() -> None:
    grouped = {
        "large-scale": (
            np.full((8, 3), 99.0),
            np.full((8, 3), 100.0),
        ),
        "small-scale": (
            np.full((8, 3), 1.1),
            np.full((8, 3), 1.0),
        ),
    }

    # Raw deltas average positive, but the task-equal relative effect is -4.5%.
    p_value = _sign_flip_pvalue(grouped, samples=10_000, seed=5120)

    assert p_value > 0.5


def test_sign_flip_is_frozen_to_one_sided_greater() -> None:
    grouped = {
        "task": (
            np.full((16, 3), 0.8),
            np.full((16, 3), 1.0),
        )
    }

    p_value = _sign_flip_pvalue(
        grouped,
        samples=10_000,
        seed=5120,
        alternative="greater",
    )

    assert p_value < 0.05
    with pytest.raises(ValueError, match="alternative must be 'greater'"):
        _sign_flip_pvalue(
            grouped,
            samples=10_000,
            seed=5120,
            alternative="less",
        )



def test_gate_is_invariant_to_episode_serialization_order() -> None:
    records = _records()
    reversed_records = _records()
    for record in reversed_records:
        record["episode_blocks"].reverse()

    summary = summarize_anchor_visual_dev_gate(records, _protocol())
    reversed_summary = summarize_anchor_visual_dev_gate(
        reversed_records, _protocol()
    )

    assert reversed_summary["primary"]["relative_improvement_95_ci"] == pytest.approx(
        summary["primary"]["relative_improvement_95_ci"]
    )
    assert [row["p_value"] for row in reversed_summary["secondary_holm_family"]] == pytest.approx(
        [row["p_value"] for row in summary["secondary_holm_family"]]
    )


@pytest.mark.parametrize(
    ("section", "field", "tampered_value", "message"),
    [
        (
            "primary",
            "candidate",
            "paired_visual/paired_visual",
            "primary candidate",
        ),
        (
            "primary",
            "control",
            "triplet_nominal/triplet_nominal",
            "primary control",
        ),
        ("primary", "condition", "clean", "primary condition"),
        ("primary", "aggregation", "episode_micro", "primary aggregation"),
        ("primary", "estimand", "ratio_of_task_macros", "primary estimand"),
        ("primary", "horizons", [1, 5], "primary horizons"),
        ("bootstrap", "confidence", 0.9, "bootstrap confidence"),
        ("bootstrap", "unit", "task", "bootstrap unit"),
        (
            "clean_noninferiority",
            "candidate",
            "paired_visual/paired_visual",
            "clean candidate",
        ),
        (
            "clean_noninferiority",
            "control",
            "triplet_nominal/triplet_nominal",
            "clean control",
        ),
        (
            "clean_noninferiority",
            "one_sided_confidence",
            0.9,
            "clean one-sided confidence",
        ),
        ("task_direction", "task_count", 3, "task count"),
        (
            "representation_noncollapse",
            "absolute_threshold_scope",
            "overall_only",
            "representation absolute threshold scope",
        ),
        ("bootstrap", "samples", 9999, "bootstrap samples"),
        ("bootstrap", "seed", 1, "bootstrap seed"),
        (
            "primary",
            "minimum_relative_improvement",
            0.02,
            "minimum primary effect",
        ),
        (
            "primary",
            "paired_bootstrap_ci_lower_must_exceed",
            -0.01,
            "minimum primary CI lower bound",
        ),
        (
            "clean_noninferiority",
            "ratio_upper_margin",
            1.1,
            "clean ratio upper margin",
        ),
        ("task_direction", "minimum_positive_tasks", 2, "minimum positive tasks"),
        (
            "task_direction",
            "maximum_single_task_relative_degradation",
            0.1,
            "maximum task degradation",
        ),
        (
            "representation_noncollapse",
            "absolute_effective_rank_minimum",
            7.0,
            "representation absolute_effective_rank_minimum",
        ),
        ("secondary_family", "alpha", 0.0, "secondary alpha"),
        ("secondary_family", "alpha", 1.0, "secondary alpha"),
        (
            "secondary_family",
            "alternative",
            "two-sided",
            "secondary alternative",
        ),
        ("secondary_family", "correction", "none", "secondary correction"),
        (
            "secondary_family",
            "cannot_rescue_primary_failure",
            False,
            "secondary rescue policy",
        ),
        ("claim_boundary", "development_only", False, "claim boundary"),
        (
            "decision",
            "pass_action",
            "publish_as_final",
            "decision protocol",
        ),
        ("training_contract", "seed", 1, "training contract"),
        (
            "data",
            "full_validation_counts",
            {"clean": 56, "mild": 112, "total": 167},
            "full validation counts",
        ),
    ],
)
def test_frozen_protocol_fields_fail_closed_when_tampered(
    section: str,
    field: str,
    tampered_value: object,
    message: str,
) -> None:
    protocol = _protocol()
    protocol[section][field] = tampered_value

    with pytest.raises(ValueError, match=message):
        summarize_anchor_visual_dev_gate(_records(), protocol)


def test_protocol_schema_version_is_frozen() -> None:
    protocol = _protocol()
    protocol["schema_version"] = 2

    with pytest.raises(ValueError, match="protocol schema_version"):
        summarize_anchor_visual_dev_gate(_records(), protocol)


def test_checked_in_protocol_freezes_greater_alternative() -> None:
    protocol_path = (
        Path(__file__).parents[1]
        / "config/lewm/eval/anchor_visual_dev_protocol_seed5120.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    validate_anchor_visual_dev_protocol(protocol)

    assert protocol["secondary_family"]["alternative"] == "greater"


def test_secondary_alternative_is_required_by_the_frozen_protocol() -> None:
    protocol = _protocol()
    del protocol["secondary_family"]["alternative"]

    with pytest.raises(ValueError, match="secondary alternative"):
        summarize_anchor_visual_dev_gate(_records(), protocol)


def test_actual_episode_counts_are_frozen() -> None:
    records = _records()
    for record in records:
        record["episode_blocks"] = record["episode_blocks"][1:]

    with pytest.raises(ValueError, match="development episode counts differ"):
        summarize_anchor_visual_dev_gate(records, _protocol())


def test_primary_slice_requires_exactly_four_tasks() -> None:
    records = _records()
    for record in records:
        for block in record["episode_blocks"]:
            if block["task_name"] == TASKS[-1]:
                block["task_name"] = TASKS[0]

    with pytest.raises(
        ValueError, match="development episodes must contain exactly 4 tasks"
    ):
        summarize_anchor_visual_dev_gate(records, _protocol())


def test_corrector_train_unseen_slice_requires_exactly_four_tasks() -> None:
    records = _records()
    for record in records:
        for block in record["episode_blocks"]:
            if block["task_name"] == TASKS[-1]:
                block["corrector_train_unseen"] = False
        for condition, moved_count in (("clean", 3), ("mild", 6)):
            replacements = [
                block
                for block in record["episode_blocks"]
                if block["task_name"] == TASKS[0]
                and block["camera_condition"] == condition
                and not block["corrector_train_unseen"]
            ]
            for block in replacements[:moved_count]:
                block["corrector_train_unseen"] = True

    with pytest.raises(
        ValueError,
        match="corrector-train-unseen slice must contain exactly 4 tasks",
    ):
        summarize_anchor_visual_dev_gate(records, _protocol())


def test_representation_gate_requires_exactly_four_tasks() -> None:
    records = _records()
    for cell in (PI_CELL, TC_CELL):
        representation = _record(records, cell)["metrics"]["representation"]
        representation["per_task"].pop(TASKS[-1])

    with pytest.raises(
        ValueError,
        match="representation gate must contain exactly 4 tasks",
    ):
        summarize_anchor_visual_dev_gate(records, _protocol())


def test_primary_pi_vs_tc_fails_below_minimum_effect() -> None:
    records = _records()
    _set_ratios(records, PI_CELL, 0.98, condition="mild")

    summary = summarize_anchor_visual_dev_gate(records, _protocol())

    assert summary["primary"]["candidate_macro"] == pytest.approx(0.98)
    assert summary["primary"]["control_macro"] == pytest.approx(1.0)
    assert summary["primary"]["relative_improvement"] == pytest.approx(0.02)
    assert summary["primary"]["relative_improvement_95_ci"] == pytest.approx(
        [0.02, 0.02]
    )
    assert summary["primary"]["passed"] is False
    assert summary["conditions"] == {
        "primary_superiority": False,
        "clean_noninferiority": True,
        "task_direction": True,
        "representation_noncollapse": True,
        "corrector_train_unseen_directional_agreement": True,
    }
    assert summary["decision"] == "no_go"


def test_clean_slice_task_set_must_match_primary() -> None:
    records = _records()
    for record in records:
        for block in record["episode_blocks"]:
            if (
                block["camera_condition"] == "clean"
                and block["task_name"] == TASKS[-1]
            ):
                block["task_name"] = TASKS[0]

    with pytest.raises(ValueError, match="clean slice task set must match"):
        summarize_anchor_visual_dev_gate(records, _protocol())


def test_clean_noninferiority_rejects_ratio_above_margin() -> None:
    records = _records()
    _set_ratios(records, PI_CELL, 1.06, condition="clean")

    summary = summarize_anchor_visual_dev_gate(records, _protocol())

    clean = summary["clean_noninferiority"]
    assert clean["candidate_to_control_ratio_one_sided_95_upper"] == pytest.approx(
        1.06
    )
    assert clean["passed"] is False
    assert summary["primary"]["passed"] is True
    assert summary["conditions"]["clean_noninferiority"] is False
    assert summary["passed"] is False


def test_task_direction_requires_three_positive_tasks() -> None:
    records = _records()
    _set_ratios(records, PI_CELL, 0.8, condition="mild", tasks=TASKS[:2])
    _set_ratios(
        records,
        PI_CELL,
        1.1,
        condition="mild",
        corrector_train_unseen=False,
        tasks=TASKS[2:],
    )

    summary = summarize_anchor_visual_dev_gate(records, _protocol())

    assert summary["primary"]["passed"] is True
    assert summary["task_direction"]["positive_tasks"] == 2
    assert summary["task_direction"][
        "worst_relative_improvement"
    ] == pytest.approx(-2.0 / 35.0)
    assert summary["task_direction"]["passed"] is False
    assert summary["corrector_train_unseen_sensitivity"]["passed"] is True
    assert summary["conditions"]["task_direction"] is False


def test_corrector_train_unseen_slice_must_agree_directionally() -> None:
    records = _records()
    _set_ratios(records, PI_CELL, 0.7, condition="mild")
    _set_ratios(
        records,
        PI_CELL,
        0.8,
        condition="mild",
        corrector_train_unseen=True,
        tasks=TASKS[:2],
    )
    _set_ratios(
        records,
        PI_CELL,
        1.0,
        condition="mild",
        corrector_train_unseen=True,
        tasks=TASKS[2:],
    )

    summary = summarize_anchor_visual_dev_gate(records, _protocol())

    unseen = summary["corrector_train_unseen_sensitivity"]
    assert summary["primary"]["passed"] is True
    assert summary["task_direction"]["passed"] is True
    assert unseen["relative_improvement"] == pytest.approx(0.1)
    assert sum(
        row["relative_improvement"] > 0.0
        for row in unseen["per_task"].values()
    ) == 2
    assert unseen["passed"] is False
    assert "not an independent test" in unseen["selection_disclosure"]
    assert summary["conditions"][
        "corrector_train_unseen_directional_agreement"
    ] is False


def test_representation_rank_and_participation_ratio_collapse_blocks_gate() -> None:
    records = _records()
    representation = _record(records, PI_CELL)["metrics"]["representation"]
    representation["overall"]["effective_rank"] = 7.9
    representation["per_task"][TASKS[0]]["participation_ratio"] = 4.5

    summary = summarize_anchor_visual_dev_gate(records, _protocol())

    gate = summary["representation_noncollapse"]
    assert gate["overall"]["effective_rank"]["passed"] is False
    assert gate["per_task"][TASKS[0]]["participation_ratio"]["ratio"] == pytest.approx(
        0.75
    )
    assert gate["per_task"][TASKS[0]]["participation_ratio"]["passed"] is False
    assert gate["passed"] is False
    assert summary["conditions"]["representation_noncollapse"] is False
    assert summary["passed"] is False


def test_zero_candidate_rank_is_a_gate_failure_not_an_exception() -> None:
    records = _records()
    representation = _record(records, PI_CELL)["metrics"]["representation"]
    representation["overall"]["effective_rank"] = 0.0

    summary = summarize_anchor_visual_dev_gate(records, _protocol())

    row = summary["representation_noncollapse"]["overall"]["effective_rank"]
    assert row["ratio"] == pytest.approx(0.0)
    assert row["passed"] is False
    assert summary["conditions"]["representation_noncollapse"] is False


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("element_count", "element_count must be equal across H=1,3,5"),
        ("camera_condition", "contract field condition differs"),
    ],
)
def test_rejects_episode_contract_mismatch_across_cells(
    field: str, message: str
) -> None:
    records = _records()
    block = _record(records, PI_CELL)["episode_blocks"][0]
    if field == "element_count":
        block["endpoints"]["1"]["element_count"] += 1
    else:
        block["camera_condition"] = "mild"

    with pytest.raises(ValueError, match=message):
        summarize_anchor_visual_dev_gate(records, _protocol())


def test_pi_better_than_tn_but_worse_than_tc_still_fails_primary() -> None:
    records = _records()
    _set_ratios(records, PI_CELL, 1.1, condition="mild")

    pi_block = next(
        block
        for block in _record(records, PI_CELL)["episode_blocks"]
        if block["camera_condition"] == "mild"
    )
    tn_block = next(
        block
        for block in _record(records, TN_CELL)["episode_blocks"]
        if block["camera_condition"] == "mild"
    )
    assert pi_block["endpoints"]["1"]["ratio"] < tn_block["endpoints"]["1"][
        "ratio"
    ]

    summary = summarize_anchor_visual_dev_gate(records, _protocol())

    assert summary["primary"]["candidate_macro"] == pytest.approx(1.1)
    assert summary["primary"]["control_macro"] == pytest.approx(1.0)
    assert summary["primary"]["relative_improvement"] == pytest.approx(-0.1)
    assert summary["primary"]["passed"] is False
    assert summary["conditions"]["primary_superiority"] is False
    assert summary["decision"] == "no_go"


def test_paired_effect_fails_closed_on_finite_input_overflow() -> None:
    enormous = np.full((2, 3), 1.0e308, dtype=np.float64)
    grouped = {"task": (enormous.copy(), enormous.copy())}

    with pytest.raises(ValueError, match="remain finite"):
        _paired_effect(grouped, bootstrap_samples=10, bootstrap_seed=5120)


def test_sign_flip_fails_closed_on_finite_input_overflow() -> None:
    enormous = np.full((2, 3), 1.0e308, dtype=np.float64)
    grouped = {"task": (enormous.copy(), enormous.copy())}

    with pytest.raises(ValueError, match="remain finite"):
        _sign_flip_pvalue(grouped, samples=10, seed=5120)
