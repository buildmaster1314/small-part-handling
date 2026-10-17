from __future__ import annotations

from copy import deepcopy

import pytest

from fcpg.lewm_anchor_ablation import (
    EXPECTED_OPTIMIZER_STEP,
    PAIRED_INNOVATION,
    PAIRED_VISUAL,
    PARAMETER_COMPATIBLE_TRIPLET_MODES,
    TRIPLET_CORRECTED,
    TRIPLET_NOMINAL,
    build_anchor_ablation_cells,
    summarize_anchor_ablation_reports,
    validate_anchor_visual_override,
)


TARGET_HASH = "a" * 64
SPLIT_HASH = "b" * 64


def _reports() -> list[dict[str, object]]:
    reports = []
    for train_index, train_mode in enumerate(PARAMETER_COMPATIBLE_TRIPLET_MODES):
        for eval_index, eval_mode in enumerate(PARAMETER_COMPATIBLE_TRIPLET_MODES):
            reports.append(
                {
                    "train_mode": train_mode,
                    "eval_mode": eval_mode,
                    "step": EXPECTED_OPTIMIZER_STEP,
                    "parameter_count": 19_152_816,
                    "seed": 5120,
                    "target_sha256": TARGET_HASH,
                    "data_split_sha256": SPLIT_HASH,
                    "metrics": {
                        "prediction_mse": 1.0
                        - 0.1 * train_index
                        - 0.01 * eval_index
                    },
                }
            )
    return reports


def test_only_parameter_compatible_triplet_modes_are_legal_overrides() -> None:
    assert PARAMETER_COMPATIBLE_TRIPLET_MODES == (
        TRIPLET_NOMINAL,
        TRIPLET_CORRECTED,
        PAIRED_VISUAL,
        PAIRED_INNOVATION,
    )
    for mode in PARAMETER_COMPATIBLE_TRIPLET_MODES:
        assert validate_anchor_visual_override(mode) == mode
    with pytest.raises(ValueError, match="parameter-compatible"):
        validate_anchor_visual_override("single_corrected")
    with pytest.raises(TypeError, match="must be a string"):
        validate_anchor_visual_override(None)  # type: ignore[arg-type]


def test_cell_builder_covers_the_complete_deterministic_4x4() -> None:
    cells = build_anchor_ablation_cells()

    assert len(cells) == 16
    assert len({cell.key for cell in cells}) == 16
    assert cells[0].key == (TRIPLET_NOMINAL, TRIPLET_NOMINAL)
    assert cells[-1].key == (PAIRED_INNOVATION, PAIRED_INNOVATION)


def test_summary_validates_provenance_and_reports_relative_matrix() -> None:
    summary = summarize_anchor_ablation_reports(
        _reports(),
        expected_parameter_count=19_152_816,
        expected_seed=5120,
        expected_target_sha256=TARGET_HASH,
        expected_data_split_sha256=SPLIT_HASH,
    )

    assert summary["provenance"] == {
        "optimizer_step": 2000,
        "parameter_count": 19_152_816,
        "seed": 5120,
        "target_sha256": TARGET_HASH,
        "data_split_sha256": SPLIT_HASH,
        "complete_4x4": True,
    }
    assert summary["baseline"]["value"] == pytest.approx(1.0)
    assert summary["train_by_eval"][PAIRED_INNOVATION][PAIRED_INNOVATION] == pytest.approx(
        0.67
    )
    assert summary["relative_improvement_vs_triplet_nominal"][PAIRED_INNOVATION][
        PAIRED_INNOVATION
    ] == pytest.approx(0.33)
    assert len(summary["cells"]) == 16


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("step", 1999, "must use step=2000"),
        ("parameter_count", 19_152_817, "parameter counts differ"),
        ("seed", 5121, "training seeds differ"),
        ("target_sha256", "c" * 64, "target hashes differ"),
        ("data_split_sha256", "d" * 64, "data split hashes differ"),
    ],
)
def test_summary_fails_closed_on_mismatched_contract_fields(
    field: str,
    replacement: object,
    message: str,
) -> None:
    reports = _reports()
    reports[-1][field] = replacement

    with pytest.raises(ValueError, match=message):
        summarize_anchor_ablation_reports(reports)


def test_summary_fails_closed_on_missing_or_duplicate_cells() -> None:
    reports = _reports()
    with pytest.raises(ValueError, match="incomplete anchor ablation 4x4 matrix"):
        summarize_anchor_ablation_reports(reports[:-1])

    duplicate = deepcopy(reports[0])
    with pytest.raises(ValueError, match="duplicate anchor ablation matrix cell"):
        summarize_anchor_ablation_reports([*reports, duplicate])


def test_summary_rejects_parameter_incompatible_report_mode() -> None:
    reports = _reports()
    reports[0]["eval_mode"] = "single_corrected"

    with pytest.raises(ValueError, match="parameter-compatible"):
        summarize_anchor_ablation_reports(reports)

