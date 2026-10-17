"""Focused tests for the paired seed0 FK-DeformAttn development runner."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_fk_deform_attn_seed0_development as runner
from fcpg.fk_deform_attn_seed0_statistics import (
    DEFAULT_CONFIG,
    ROW_FIELDS,
    VARIANTS,
    expected_episode_specs,
    load_protocol_config,
    pair_artifact_relative_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_result(**kwargs: object) -> dict[str, object]:
    translation, axis, rotation = kwargs["camera_offset"]
    translation = np.asarray(translation, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    physical = hashlib.sha256(
        (
            f"{kwargs['task_name']}:{kwargs['task_index']}:"
            f"{kwargs['env_seed']}"
        ).encode("ascii")
    ).hexdigest()
    view = hashlib.sha256(
        physical.encode("ascii")
        + translation.tobytes()
        + axis.tobytes()
        + str(rotation).encode("ascii")
    ).hexdigest()
    return {
        "task": kwargs["task_name"],
        "task_index": kwargs["task_index"],
        "env_seed": kwargs["env_seed"],
        "physical_episode_sha256": physical,
        "paired_initial_view_sha256": view,
        "initial_rgb_sha256": view,
        "initial_q_nominal_sha256": view,
        "initial_q_true_sha256": view,
        "initial_q_true_valid_sha256": view,
        "steps": 10,
        "inference_calls": 2,
        "success": 1.0,
        "reward_sum": 3.0,
        "action_abs_mean": 0.25,
        "actual_camera_translation_cm": float(np.linalg.norm(translation) * 100.0),
        "actual_camera_translation_x_cm": float(translation[0] * 100.0),
        "actual_camera_translation_y_cm": float(translation[1] * 100.0),
        "actual_camera_translation_z_cm": float(translation[2] * 100.0),
        "actual_camera_rotation_axis_x": float(axis[0]),
        "actual_camera_rotation_axis_y": float(axis[1]),
        "actual_camera_rotation_axis_z": float(axis[2]),
        "actual_camera_rotation_deg": float(rotation),
        "coordinate_diagnostic_status": "unavailable",
        "coordinate_diagnostic_name": "attention_barycenter_epe_px",
        "coordinate_diagnostic_reason": "fake rollout has no readout",
        "coordinate_epe_claim_supported": False,
        "initial_attention_barycenter_epe_sum_px": None,
        "initial_attention_barycenter_valid_count": 0,
        "initial_attention_barycenter_epe_px": None,
        "on_policy_attention_barycenter_epe_sum_px": None,
        "on_policy_attention_barycenter_valid_count": 0,
        "on_policy_attention_barycenter_epe_px": None,
    }


def _models() -> dict[str, tuple[object, dict[str, object]]]:
    return {variant: (variant, {}) for variant in VARIANTS}


def test_frozen_schedule_and_camera_directions() -> None:
    config = load_protocol_config(REPO_ROOT / DEFAULT_CONFIG)
    specs = expected_episode_specs(config)
    assert len(specs) == 4 * 5 * 20
    assert len({spec.physical_episode_id for spec in specs}) == 4 * 20
    first = specs[:5]
    assert [spec.condition.condition_id for spec in first] == [
        "clean",
        "rotation_5deg",
        "rotation_10deg",
        "translation_2cm",
        "translation_4cm",
    ]
    assert len({(spec.task_index, spec.env_seed) for spec in first}) == 1

    clean, rot5, rot10, trans2, trans4 = map(
        runner.camera_offset_for_spec,
        first,
    )
    np.testing.assert_array_equal(clean[0], np.zeros(3))
    assert clean[2] == 0.0
    np.testing.assert_allclose(rot5[1], rot10[1], atol=0.0, rtol=0.0)
    assert abs(rot5[2]) == 5.0
    assert abs(rot10[2]) == 10.0
    assert np.sign(rot5[2]) == np.sign(rot10[2])
    np.testing.assert_allclose(
        trans2[0] / np.linalg.norm(trans2[0]),
        trans4[0] / np.linalg.norm(trans4[0]),
        atol=0.0,
        rtol=0.0,
    )
    assert np.linalg.norm(trans2[0]) * 100.0 == pytest.approx(2.0)
    assert np.linalg.norm(trans4[0]) * 100.0 == pytest.approx(4.0)
    assert runner.variant_execution_order(0) == VARIANTS
    assert runner.variant_execution_order(1) == tuple(reversed(VARIANTS))


def test_atomic_pair_resume_skips_complete_pairs(tmp_path: Path) -> None:
    config = load_protocol_config(REPO_ROOT / DEFAULT_CONFIG)
    calls: list[str] = []

    def rollout(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["model"]))
        return _fake_result(**kwargs)

    rows, new_pairs = runner.run_pairs(
        config,
        tmp_path,
        _models(),
        "cpu",
        rollout_fn=rollout,
        max_new_pairs=2,
    )
    assert new_pairs == 2
    assert len(rows) == 4
    assert calls == [
        "policy_only_residual",
        "fk_deform_attn",
        "fk_deform_attn",
        "policy_only_residual",
    ]
    specs = expected_episode_specs(config)
    for pair_index in range(2):
        path = tmp_path / pair_artifact_relative_path(config, specs[pair_index])
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["variant_execution_order"] == list(
            runner.variant_execution_order(pair_index)
        )
        assert len(payload["rows"]) == 2

    calls.clear()
    rows, new_pairs = runner.run_pairs(
        config,
        tmp_path,
        _models(),
        "cpu",
        rollout_fn=rollout,
        max_new_pairs=1,
    )
    assert new_pairs == 1
    assert len(rows) == 6
    assert calls == ["policy_only_residual", "fk_deform_attn"]


def test_failed_half_pair_is_not_committed(tmp_path: Path) -> None:
    config = load_protocol_config(REPO_ROOT / DEFAULT_CONFIG)
    calls = 0

    def interrupted(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return _fake_result(**kwargs)

    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run_pairs(
            config,
            tmp_path,
            _models(),
            "cpu",
            rollout_fn=interrupted,
            max_new_pairs=1,
        )
    first = expected_episode_specs(config)[0]
    path = tmp_path / pair_artifact_relative_path(config, first)
    assert not path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_final_csv_and_summary_are_complete_only(tmp_path: Path) -> None:
    config = load_protocol_config(REPO_ROOT / DEFAULT_CONFIG)
    rows = []
    for spec in expected_episode_specs(config):
        offset = runner.camera_offset_for_spec(spec)
        kwargs = {
            "task_name": spec.task.name,
            "task_index": spec.task_index,
            "env_seed": spec.env_seed,
            "camera_offset": offset,
        }
        for variant in VARIANTS:
            rows.append(
                runner.build_rollout_row(spec, variant, _fake_result(**kwargs))
            )

    csv_path, summary_path = runner.write_final_artifacts(config, tmp_path, rows)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert tuple(written[0]) == ROW_FIELDS
    assert len(written) == 800
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["complete"] is True
    assert summary["report_type"] == "fk_deform_attn_seed0_development_statistics"
    assert summary["audit"]["expected_pairs"] == 400
    assert summary["audit"]["observed_rows"] == 800
