"""Contract and inference tests for the seed0 FK-DeformAttn statistics."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from fcpg.fk_deform_attn_seed0_statistics import (
    DEFAULT_CONFIG,
    DIFFERENCE_ORDER,
    FROZEN_PROTOCOL_CONFIG,
    PROTOCOL,
    ROW_FIELDS,
    ROW_SCHEMA_VERSION,
    VARIANTS,
    EpisodeSpec,
    expected_episode_specs,
    load_protocol_config,
    protocol_config_sha256,
    summarize_rollout_rows,
    validate_protocol_config,
    validate_rollout_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

# Counts are ordered as pick, button, reach, push. Successes occupy the first
# ``count`` paired episodes, making every point estimate directly auditable.
SUCCESS_COUNTS = {
    "clean": {
        "policy_only_residual": (16, 14, 18, 12),
        "fk_deform_attn": (16, 14, 18, 12),
    },
    "rotation_5deg": {
        "policy_only_residual": (10, 8, 12, 6),
        "fk_deform_attn": (14, 10, 12, 8),
    },
    "rotation_10deg": {
        "policy_only_residual": (4, 2, 8, 6),
        "fk_deform_attn": (10, 6, 10, 4),
    },
    "translation_2cm": {
        "policy_only_residual": (12, 10, 14, 8),
        "fk_deform_attn": (16, 12, 16, 10),
    },
    "translation_4cm": {
        "policy_only_residual": (6, 4, 10, 2),
        "fk_deform_attn": (12, 8, 12, 6),
    },
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _camera_fields(spec: EpisodeSpec) -> dict[str, float]:
    severity = float(spec.condition.severity)
    translation = severity if spec.condition.perturbation == "translation" else 0.0
    rotation = severity if spec.condition.perturbation == "rotation" else 0.0
    return {
        "actual_camera_translation_cm": translation,
        "actual_camera_translation_x_cm": translation,
        "actual_camera_translation_y_cm": 0.0,
        "actual_camera_translation_z_cm": 0.0,
        "actual_camera_rotation_axis_x": 0.0,
        "actual_camera_rotation_axis_y": 0.0,
        "actual_camera_rotation_axis_z": 1.0,
        "actual_camera_rotation_deg": rotation,
    }


def _coordinate_fields(spec: EpisodeSpec, variant: str) -> dict[str, object]:
    initial_mean = 4.0 + spec.task.ordinal
    on_policy_mean = 6.0
    if variant == "fk_deform_attn":
        initial_mean -= 2.0
        on_policy_mean -= 3.0
    initial_count = 2
    on_policy_count = 3
    return {
        "coordinate_diagnostic_status": "available",
        "coordinate_diagnostic_name": "attention_barycenter_epe_px",
        "coordinate_diagnostic_reason": "",
        "coordinate_epe_claim_supported": True,
        "initial_attention_barycenter_epe_sum_px": initial_mean * initial_count,
        "initial_attention_barycenter_valid_count": initial_count,
        "initial_attention_barycenter_epe_px": initial_mean,
        "on_policy_attention_barycenter_epe_sum_px": (
            on_policy_mean * on_policy_count
        ),
        "on_policy_attention_barycenter_valid_count": on_policy_count,
        "on_policy_attention_barycenter_epe_px": on_policy_mean,
    }


def _synthetic_row(spec: EpisodeSpec, variant: str) -> dict[str, object]:
    success_count = SUCCESS_COUNTS[spec.condition.condition_id][variant][
        spec.task.ordinal
    ]
    success = float(spec.episode < success_count)
    row = {
        "row_schema_version": ROW_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "claim_scope": "seed0-development-controlled-comparison",
        "training_seed": 0,
        "task_key": spec.task.key,
        "task": spec.task.name,
        "task_ordinal": spec.task.ordinal,
        "condition_id": spec.condition.condition_id,
        "perturbation": spec.condition.perturbation,
        "severity": spec.condition.severity,
        "episode": spec.episode,
        "physical_episode_id": spec.physical_episode_id,
        "pair_id": spec.pair_id,
        "variant": variant,
        "task_index": spec.task_index,
        "effective_task_index": spec.effective_task_index,
        "env_seed": spec.env_seed,
        "camera_direction_seed": spec.camera_direction_seed,
        "physical_episode_sha256": _digest(
            f"physical:{spec.physical_episode_id}"
        ),
        "paired_initial_view_sha256": _digest(f"view:{spec.pair_id}"),
        "initial_rgb_sha256": _digest(f"rgb:{spec.pair_id}"),
        "initial_q_nominal_sha256": _digest(f"q-nominal:{spec.pair_id}"),
        "initial_q_true_sha256": _digest(f"q-true:{spec.pair_id}"),
        "initial_q_true_valid_sha256": _digest(f"q-valid:{spec.pair_id}"),
        "steps": 10,
        "inference_calls": 2,
        "success": success,
        "reward_sum": success + float(spec.task.ordinal),
        "action_abs_mean": 0.25,
        **_camera_fields(spec),
        **_coordinate_fields(spec, variant),
    }
    assert tuple(row) == ROW_FIELDS
    return row


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return load_protocol_config(REPO_ROOT / DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def complete_rows(config: dict[str, object]) -> list[dict[str, object]]:
    return [
        _synthetic_row(spec, variant)
        for spec in expected_episode_specs(config)
        for variant in VARIANTS
    ]


@pytest.fixture(scope="module")
def report(
    config: dict[str, object],
    complete_rows: list[dict[str, object]],
) -> dict[str, object]:
    return summarize_rollout_rows(config, complete_rows)


def _condition_report(report: dict[str, object], condition_id: str) -> dict[str, object]:
    return next(
        row
        for row in report["paired_success"]
        if row["condition"] == condition_id
    )


def _axis_report(report: dict[str, object], axis: str) -> dict[str, object]:
    return next(row for row in report["robustness_auc"] if row["axis"] == axis)


def _copy_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows]


def test_frozen_config_and_schedule(config: dict[str, object]) -> None:
    assert config == FROZEN_PROTOCOL_CONFIG
    assert protocol_config_sha256(config) == (
        "b029b94f03c033c10bac580362812799f71cc8010d664e9b1c1a3700bb5f9431"
    )
    assert validate_protocol_config(copy.deepcopy(config)) == config

    drifted = copy.deepcopy(config)
    drifted["evaluation"]["horizon"] = 101
    with pytest.raises(
        ValueError,
        match=r"frozen FK-DeformAttn protocol drift: config\.evaluation\.horizon",
    ):
        validate_protocol_config(drifted)

    specs = expected_episode_specs(config)
    assert len(specs) == 4 * 20 * 5
    assert len({spec.physical_episode_id for spec in specs}) == 4 * 20
    assert len({spec.pair_id for spec in specs}) == len(specs)
    assert [spec.condition.condition_id for spec in specs[:5]] == [
        "clean",
        "rotation_5deg",
        "rotation_10deg",
        "translation_2cm",
        "translation_4cm",
    ]

    clean, rotation_5, rotation_10, translation_2, translation_4 = specs[:5]
    assert clean.task_index == 1_726_071
    assert clean.effective_task_index == 21
    assert clean.env_seed == 1_726_071_800
    assert clean.camera_direction_seed == 0
    assert rotation_5.camera_direction_seed == 1_826_071_800
    assert rotation_10.camera_direction_seed == rotation_5.camera_direction_seed
    assert translation_2.camera_direction_seed == 1_926_071_800
    assert translation_4.camera_direction_seed == translation_2.camera_direction_seed
    assert {
        (spec.task_index, spec.effective_task_index, spec.env_seed)
        for spec in specs[:5]
    } == {(1_726_071, 21, 1_726_071_800)}
    assert specs[5].task_index == clean.task_index + 1
    assert specs[5].env_seed == clean.env_seed + 1


def test_complete_rows_pass_strict_pairing_audit(
    config: dict[str, object],
    complete_rows: list[dict[str, object]],
) -> None:
    assert len(complete_rows) == 800
    assert all(tuple(row) == ROW_FIELDS for row in complete_rows)
    assert validate_rollout_rows(config, complete_rows) == {
        "complete": True,
        "expected_pairs": 400,
        "observed_pairs": 400,
        "observed_rows": 800,
        "strict_variant_pairing": True,
        "physical_state_hash_shared_across_conditions": True,
        "camera_direction_shared_across_axis_severities": True,
    }

    mismatched = _copy_rows(complete_rows)
    candidate = next(
        row
        for row in mismatched
        if row["task_key"] == "pick"
        and row["episode"] == 0
        and row["condition_id"] == "clean"
        and row["variant"] == "fk_deform_attn"
    )
    candidate["paired_initial_view_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="strict pair field paired_initial_view_sha256"):
        validate_rollout_rows(config, mismatched)


@pytest.mark.parametrize(
    ("condition_id", "updates", "error"),
    [
        (
            "rotation_10deg",
            {
                "actual_camera_rotation_axis_x": 1.0,
                "actual_camera_rotation_axis_z": 0.0,
            },
            "rotation axis is not shared across severities",
        ),
        (
            "rotation_10deg",
            {"actual_camera_rotation_deg": -10.0},
            "rotation sign is not shared across severities",
        ),
        (
            "translation_4cm",
            {
                "actual_camera_translation_x_cm": 0.0,
                "actual_camera_translation_y_cm": 4.0,
            },
            "translation direction is not shared across severities",
        ),
        (
            "rotation_5deg",
            {"physical_episode_sha256": "0" * 64},
            "physical state changed across camera conditions",
        ),
    ],
)
def test_camera_direction_and_physical_identity_checks_fail_closed(
    config: dict[str, object],
    complete_rows: list[dict[str, object]],
    condition_id: str,
    updates: dict[str, object],
    error: str,
) -> None:
    corrupted = _copy_rows(complete_rows)
    for row in corrupted:
        if (
            row["task_key"] == "pick"
            and row["episode"] == 0
            and row["condition_id"] == condition_id
        ):
            row.update(updates)
    with pytest.raises(ValueError, match=error):
        validate_rollout_rows(config, corrupted)


def test_task_macro_paired_success(report: dict[str, object]) -> None:
    rotation_5 = _condition_report(report, "rotation_5deg")
    assert rotation_5["difference_order"] == DIFFERENCE_ORDER
    assert [row["task"] for row in rotation_5["tasks"]] == [
        "pick",
        "button",
        "reach",
        "push",
    ]
    assert [row["paired_difference"] for row in rotation_5["tasks"]] == pytest.approx(
        [0.2, 0.1, 0.0, 0.1]
    )
    assert rotation_5["tasks"][0]["wins"] == 4
    assert rotation_5["tasks"][0]["ties"] == 16
    assert rotation_5["tasks"][0]["losses"] == 0
    macro = rotation_5["task_macro"]
    assert macro["policy_only_residual"] == pytest.approx(0.45)
    assert macro["fk_deform_attn"] == pytest.approx(0.55)
    assert macro["paired_difference"] == pytest.approx(0.1)
    assert macro["paired_difference"] == pytest.approx(
        sum(row["paired_difference"] for row in rotation_5["tasks"]) / 4.0
    )
    assert macro["ci_lower"] > 0.0
    assert report["inference_contract"] == {
        "training_seed_count": 1,
        "seed0_only": True,
        "cross_seed_claim_supported": False,
        "bootstrap_unit": "paired physical episode within fixed task",
        "tasks_resampled": False,
        "seeds_resampled": False,
    }


def test_rotation_and_translation_auc(report: dict[str, object]) -> None:
    rotation = _axis_report(report, "rotation")
    translation = _axis_report(report, "translation")
    assert rotation["normalization"] == (
        "trapezoid-area-divided-by-maximum-severity"
    )
    assert translation["normalization"] == rotation["normalization"]

    rotation_pick = rotation["tasks"][0]
    assert rotation_pick["policy_only_residual"] == pytest.approx(0.5)
    assert rotation_pick["fk_deform_attn"] == pytest.approx(0.675)
    assert rotation["task_macro"]["policy_only_residual"] == pytest.approx(0.475)
    assert rotation["task_macro"]["fk_deform_attn"] == pytest.approx(0.55625)
    assert rotation["task_macro"]["paired_difference"] == pytest.approx(0.08125)

    translation_pick = translation["tasks"][0]
    assert translation_pick["policy_only_residual"] == pytest.approx(0.575)
    assert translation_pick["fk_deform_attn"] == pytest.approx(0.75)
    assert translation["task_macro"]["policy_only_residual"] == pytest.approx(
        0.53125
    )
    assert translation["task_macro"]["fk_deform_attn"] == pytest.approx(0.64375)
    assert translation["task_macro"]["paired_difference"] == pytest.approx(0.1125)


def test_clean_noninferiority_uses_paired_one_sided_bound(
    report: dict[str, object],
) -> None:
    clean = _condition_report(report, "clean")
    assert clean["task_macro"]["policy_only_residual"] == pytest.approx(0.75)
    assert clean["task_macro"]["fk_deform_attn"] == pytest.approx(0.75)
    assert clean["task_macro"]["paired_difference"] == pytest.approx(0.0)

    noninferiority = report["clean_noninferiority"]
    assert noninferiority["difference_order"] == DIFFERENCE_ORDER
    assert noninferiority["margin_success_probability"] == pytest.approx(-0.05)
    assert noninferiority["one_sided_confidence_level"] == pytest.approx(0.95)
    assert noninferiority["paired_task_macro_difference"] == pytest.approx(0.0)
    assert noninferiority["one_sided_lower_bound"] == pytest.approx(0.0)
    assert noninferiority["noninferior"] is True
    assert noninferiority["development_only"] is True


def test_attention_barycenter_diagnostic_and_global_fail_closed_behavior(
    config: dict[str, object],
    complete_rows: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    diagnostic = report["coordinate_diagnostic"]
    assert diagnostic["epe_claim_supported"] is True
    assert diagnostic["metric"] == "attention_barycenter_epe_px"
    assert diagnostic["unqualified_keypoint_epe_claim_supported"] is False
    clean = diagnostic["initial_same_state_paired"][0]
    assert clean["condition"] == "clean"
    assert clean["task_macro"]["policy_only_residual"] == pytest.approx(5.5)
    assert clean["task_macro"]["fk_deform_attn"] == pytest.approx(3.5)
    assert clean["task_macro"]["paired_difference"] == pytest.approx(-2.0)
    assert clean["task_macro"]["ci_lower"] == pytest.approx(-2.0)
    assert clean["task_macro"]["ci_upper"] == pytest.approx(-2.0)
    assert clean["higher_is_better"] is False
    assert clean["win_definition"] == (
        "fk_deform_attn_lower_than_policy_only_residual"
    )
    assert all(row["wins"] == 20 and row["losses"] == 0 for row in clean["tasks"])
    assert diagnostic["on_policy_descriptive_weighted_mean"] == pytest.approx(
        {"policy_only_residual": 6.0, "fk_deform_attn": 3.0}
    )
    assert diagnostic["on_policy_between_variant_inference_supported"] is False

    unavailable = _copy_rows(complete_rows)
    first = unavailable[0]
    first.update(
        {
            "coordinate_diagnostic_status": "unavailable",
            "coordinate_diagnostic_reason": "synthetic readout unavailable",
            "coordinate_epe_claim_supported": False,
            "initial_attention_barycenter_epe_sum_px": None,
            "initial_attention_barycenter_valid_count": 0,
            "initial_attention_barycenter_epe_px": None,
            "on_policy_attention_barycenter_epe_sum_px": None,
            "on_policy_attention_barycenter_valid_count": 0,
            "on_policy_attention_barycenter_epe_px": None,
        }
    )
    failed = summarize_rollout_rows(config, unavailable)["coordinate_diagnostic"]
    assert failed["epe_claim_supported"] is False
    assert failed["metric"] is None
    assert failed["reason"] == (
        "one or more rollout rows failed the coordinate semantic contract"
    )
    assert failed["failure_reasons"] == ["synthetic readout unavailable"]
    assert failed["statistics"] is None
    assert "initial_same_state_paired" not in failed

    malformed = _copy_rows(complete_rows)
    malformed[0].update(
        {
            "coordinate_diagnostic_status": "unavailable",
            "coordinate_diagnostic_reason": "synthetic readout unavailable",
            "coordinate_epe_claim_supported": False,
        }
    )
    with pytest.raises(
        ValueError,
        match="fail-closed coordinate diagnostic exposed an EPE sum",
    ):
        validate_rollout_rows(config, malformed)
