"""Focused tests for button P1 crossed-bootstrap inference."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fcpg.button_fresh_multiseed_statistics import (
    CAMERA_FIELDS,
    CrossedDesign,
    EpisodeRecord,
    canonical_conditions,
    crossed_bootstrap_summary,
    expected_episode_contracts,
    load_protocol_config,
    method_specs,
    summarize_records,
)


CONFIG_PATH = "config/experiments/button_p1_fresh_multiseed.json"


def _camera(condition) -> tuple[float, ...]:
    values = {
        "actual_camera_translation_cm": condition.translation_cm,
        "actual_camera_translation_x_cm": condition.translation_cm,
        "actual_camera_translation_y_cm": 0.0,
        "actual_camera_translation_z_cm": 0.0,
        "actual_camera_rotation_axis_x": 0.0,
        "actual_camera_rotation_axis_y": 0.0,
        "actual_camera_rotation_axis_z": 1.0,
        "actual_camera_rotation_deg": condition.rotation_deg,
    }
    return tuple(values[field] for field in CAMERA_FIELDS)


def _success(method: str, condition, episode: int) -> float:
    if method == "oracle":
        return 1.0
    if method in {"jitter", "prototype", "pnp"}:
        return 0.0
    if method == "fcpg_nominal":
        return 1.0 if condition.kind == "clean" else 0.0
    if method == "fcpg_native":
        if condition.kind == "clean":
            return 1.0
        if condition.severity in {5.0, 2.0}:
            return float(episode < condition.episodes // 2)
        return 0.0
    raise AssertionError(method)


def _records(config) -> list[EpisodeRecord]:
    result = []
    for seed in config["seeds"]["policy"]:
        for condition in canonical_conditions(config):
            camera = _camera(condition)
            for method in method_specs(config):
                for expected in expected_episode_contracts(config, condition):
                    episode = int(expected["episode"])
                    result.append(
                        EpisodeRecord(
                            task="button-press-v3",
                            policy_seed=seed,
                            method=method.name,
                            condition_id=condition.condition_id,
                            pair_key=str(expected["pair_key"]),
                            episode_identity=str(expected["episode_identity"]),
                            raw_task_index=int(expected["raw_task_index"]),
                            env_seed=int(expected["env_seed"]),
                            success=_success(method.name, condition, episode),
                            camera_values=camera,
                        )
                    )
    return result


def test_crossed_bootstrap_resamples_seed_and_shared_episode_axes() -> None:
    values = np.asarray([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    first = crossed_bootstrap_summary(values, samples=1000, seed=17)
    second = crossed_bootstrap_summary(values, samples=1000, seed=17)

    assert first == second
    assert first["estimate"] == pytest.approx(2.0 / 3.0)
    assert first["policy_seed_count"] == 2
    assert first["shared_episode_count"] == 3
    assert first["bootstrap_method"] == "policy_seed_x_shared_episode_two_way_crossed"


def test_summary_reports_paired_severity_auc_and_preregistered_clean_ni() -> None:
    config = load_protocol_config(CONFIG_PATH)
    summary = summarize_records(
        config,
        _records(config),
        bootstrap_samples=500,
        bootstrap_seed=31,
    )

    assert len(summary["severity_results"]) == 6
    rotation_five = next(
        row
        for row in summary["severity_results"]
        if row["axis"] == "rotation_deg" and row["severity"] == 5.0
    )
    delta = rotation_five["paired_comparisons"]["fcpg_native_minus_jitter"]
    assert delta["estimate"] == pytest.approx(0.5)
    assert delta["estimand"] == "paired_success_probability_difference"

    rotation_auc = summary["auc_results"]["rotation_deg"]
    assert rotation_auc["shared_episode_count"] == 20
    assert rotation_auc["clean_episodes_excluded_from_auc"] == 60
    assert rotation_auc["paired_comparisons"]["fcpg_native_minus_jitter"][
        "estimate"
    ] == pytest.approx(0.5)

    clean_ni = summary["clean_noninferiority"]
    assert clean_ni["margin_success_probability"] == -0.03
    assert clean_ni["joint_decision"] == "demonstrated"
    assert set(clean_ni["comparisons"]) == {
        "fcpg_native_minus_nominal",
        "fcpg_native_minus_jitter",
    }


def test_auc_uses_only_episode_keys_shared_by_all_severities() -> None:
    config = load_protocol_config(CONFIG_PATH)
    records = _records(config)
    clean_id = config["evaluation"]["zero_condition"]
    modified = []
    for record in records:
        if (
            record.method == "fcpg_native"
            and record.condition_id == clean_id
            and record.raw_task_index >= 2020
        ):
            modified.append(replace(record, success=0.0))
        else:
            modified.append(record)
    summary = summarize_records(
        config, modified, bootstrap_samples=300, bootstrap_seed=5
    )
    clean_rotation = next(
        row
        for row in summary["severity_results"]
        if row["axis"] == "rotation_deg" and row["severity"] == 0.0
    )
    assert clean_rotation["methods"]["fcpg_native"]["estimate"] == pytest.approx(0.25)
    assert summary["auc_results"]["rotation_deg"]["methods"]["fcpg_native"][
        "estimate"
    ] == pytest.approx(0.5)


def test_missing_or_duplicate_crossed_cell_fails_closed() -> None:
    config = load_protocol_config(CONFIG_PATH)
    records = _records(config)
    conditions = canonical_conditions(config)
    methods = [value.name for value in method_specs(config)]
    design_args = {
        "seeds": config["seeds"]["policy"],
        "methods": methods,
        "conditions": conditions,
        "fixed_task": "button-press-v3",
    }

    with pytest.raises(ValueError, match="duplicate crossed-design result"):
        CrossedDesign([*records, records[0]], **design_args)

    removed = records[:-1]
    with pytest.raises(ValueError, match="lacks shared episode keys"):
        CrossedDesign(removed, **design_args)


def test_noninferiority_is_not_claimed_from_absence_of_significance() -> None:
    config = load_protocol_config(CONFIG_PATH)
    clean_id = config["evaluation"]["zero_condition"]
    records = []
    for record in _records(config):
        if record.condition_id == clean_id and record.method == "fcpg_native":
            records.append(replace(record, success=0.0))
        elif record.condition_id == clean_id and record.method == "jitter":
            records.append(replace(record, success=1.0))
        elif record.condition_id == clean_id and record.method == "fcpg_nominal":
            records.append(replace(record, success=0.0))
        else:
            records.append(record)

    summary = summarize_records(
        config, records, bootstrap_samples=300, bootstrap_seed=23
    )
    clean_ni = summary["clean_noninferiority"]
    assert clean_ni["joint_decision"] == "not_demonstrated"
    assert clean_ni["comparisons"]["fcpg_native_minus_jitter"]["decision"] == (
        "not_demonstrated"
    )
    assert "absence of statistical significance is not used" in clean_ni[
        "interpretation"
    ]
