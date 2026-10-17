"""Tests for the frozen three-seed P1-native dynamics summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fcpg.lewm_a3_p1_bridge_collection import sha256_file
from fcpg.lewm_p1_native_dynamics import (
    P1_NATIVE_PROTOCOL,
    P1_NATIVE_TARGET_HORIZONS,
    P1_NATIVE_WRONG_ACTION_SHIFTS,
)
from scripts.summarize_lewm_p1_native_dynamics import (
    CHECKPOINT_TYPE,
    DEFAULT_CONFIG,
    PRIMARY_COMPARISONS,
    render_markdown,
    summarize_reports,
)


def _config() -> dict[str, object]:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _comparison(effect: float) -> dict[str, object]:
    return {
        "orientation": "positive_means_candidate_lower_MSE",
        "relative_improvement": effect,
        "confidence": 0.95,
        "confidence_interval": [min(effect / 2.0, 0.05), max(effect, 0.1)],
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 1,
        "statistical_unit": "physical_episode",
        "per_horizon_relative_improvement": {
            f"H{horizon}": effect for horizon in P1_NATIVE_TARGET_HORIZONS
        },
    }


def _write_run(
    root: Path,
    seed: int,
    *,
    primary_effect: float = 0.2,
    hold_effect: float | None = None,
    provenance_offset: float = 0.0,
    smoke: bool = False,
) -> Path:
    output = root / f"seed{seed}"
    output.mkdir()
    config = _config()
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    predictions = output / "development_predictions.npz"
    shape = (2, 2, 6, 3, 4)
    target = np.full(shape, provenance_offset, dtype=np.float32)
    static = np.zeros(shape, dtype=np.float32)
    np.savez_compressed(
        predictions,
        target_token=target,
        factual_prediction=np.ones(shape, dtype=np.float32),
        matched_no_action_prediction=np.ones(shape, dtype=np.float32) * 2,
        wrong_action_prediction=np.ones((7, *shape), dtype=np.float32) * 3,
        wrong_action_shift=np.asarray(P1_NATIVE_WRONG_ACTION_SHIFTS, dtype=np.int64),
        hold_last_prediction=static,
        geometry_hold_exact_fk_prediction=static,
        visual_cv_exact_fk_prediction=static,
        physical_episode_id=np.asarray(
            [["episode-0", "episode-0"], ["episode-1", "episode-1"]]
        ),
        source_row_index=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        branch_index=np.asarray([[0, 1], [0, 1]], dtype=np.int64),
        hazard_id=np.asarray([["h0", "h1"], ["h0", "h1"]]),
        horizon=np.asarray(P1_NATIVE_TARGET_HORIZONS, dtype=np.int64),
    )

    comparisons = {
        name: _comparison(primary_effect) for name in PRIMARY_COMPARISONS
    }
    if hold_effect is not None:
        comparisons["factual_vs_hold_last"] = _comparison(hold_effect)
    wrong_pass: dict[str, bool] = {}
    for shift in P1_NATIVE_WRONG_ACTION_SHIFTS:
        name = f"factual_vs_wrong_action_cyclic_shift_{shift}"
        comparisons[name] = _comparison(0.2)
        wrong_pass[name] = True
    primary_pass = {}
    for name in PRIMARY_COMPARISONS:
        effect = (
            hold_effect
            if name == "factual_vs_hold_last" and hold_effect is not None
            else primary_effect
        )
        primary_pass[name] = effect >= 0.1 and min(effect / 2.0, 0.05) > 0.0
    horizon_values = {
        f"H{horizon}": 0.2 for horizon in P1_NATIVE_TARGET_HORIZONS
    }
    passed = all(primary_pass.values())
    screen = {
        "primary_pass": primary_pass,
        "wrong_action_pass": wrong_pass,
        "strongest_nonlearning_per_horizon_relative_improvement": horizon_values,
        "positive_horizon_count": 6,
        "horizon_pass": True,
        "h3_noninferiority_pass": True,
        "passed": passed,
        "cross_seed_gate_not_evaluated_by_single_run": True,
        "passing_one_seed_does_not_authorize_a_claim": True,
    }
    checkpoint = output / "p1_native_dynamics_latest.pt"
    torch.save(
        {
            "checkpoint_type": CHECKPOINT_TYPE,
            "protocol": P1_NATIVE_PROTOCOL,
            "selection": "fixed_latest",
            "completed": not smoke,
            "step": 100 if smoke else 2000,
            "seed": seed,
            "predictor_config": {
                name: value
                for name, value in config["model"].items()
                if name != "sigreg_weight"
            },
            "parameter_count_per_head": 1234,
            "action_mean": np.zeros(20, dtype=np.float32),
            "action_std": np.ones(20, dtype=np.float32),
            "config": config,
            "strict_cache_sha256": config["data"]["strict_cache_sha256"],
            "p1_policy_checkpoint_sha256": config["p1"][
                "policy_checkpoint_sha256"
            ],
            "report_screen": screen,
        },
        checkpoint,
    )
    report = {
        "protocol": P1_NATIVE_PROTOCOL,
        "seed": seed,
        "scope": "exposed_train_development_only",
        "forecast_horizons": list(P1_NATIVE_TARGET_HORIZONS),
        "statistical_unit": "physical_episode",
        "metrics": {
            "factual": {
                "physical_episode_count": 2,
                "physical_episode_macro_mse": 0.01,
                "per_horizon_mse": {
                    f"H{horizon}": 0.01 for horizon in P1_NATIVE_TARGET_HORIZONS
                },
            }
        },
        "comparisons": comparisons,
        "screen": screen,
        "optimizer_steps": 100 if smoke else 2000,
        "smoke": smoke,
        "selection": "fixed_latest",
        "parameter_count_per_head": 1234,
        "train_physical_episode_count": 24,
        "development_physical_episode_count": 12,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "predictions": str(predictions),
            "predictions_sha256": sha256_file(predictions),
        },
    }
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_two_passing_seeds_and_positive_remaining_seed_pass(tmp_path: Path) -> None:
    reports = [
        _write_run(tmp_path, 15801),
        _write_run(tmp_path, 15802),
        _write_run(tmp_path, 15803, primary_effect=0.05),
    ]

    summary = summarize_reports(reports)

    assert summary["cross_seed_gate"] == {
        "minimum_passing_seeds": 2,
        "passing_seed_count": 2,
        "passing_seeds": [15801, 15802],
        "remaining_seeds": [15803],
        "at_least_two_of_three_pass": True,
        "remaining_seed_point_estimates_positive": True,
        "passed": True,
        "decision": "Development-Go",
    }
    markdown = render_markdown(summary)
    assert "**Decision: Development-Go** (2/3" in markdown
    assert "| 15803 | No-Go |" in markdown
    assert "does not authorize" in markdown


def test_negative_remaining_seed_blocks_cross_seed_gate(tmp_path: Path) -> None:
    reports = [
        _write_run(tmp_path, 15801),
        _write_run(tmp_path, 15802),
        _write_run(tmp_path, 15803, primary_effect=-0.01),
    ]

    summary = summarize_reports(reports)

    assert summary["cross_seed_gate"]["at_least_two_of_three_pass"] is True
    assert summary["cross_seed_gate"]["remaining_seed_point_estimates_positive"] is False
    assert summary["cross_seed_gate"]["decision"] == "Development-No-Go"


def test_hold_last_is_a_required_primary_control(tmp_path: Path) -> None:
    reports = [
        _write_run(tmp_path, 15801, hold_effect=-0.01),
        _write_run(tmp_path, 15802, hold_effect=-0.01),
        _write_run(tmp_path, 15803, hold_effect=-0.01),
    ]

    summary = summarize_reports(reports)

    assert summary["cross_seed_gate"]["passing_seed_count"] == 0
    assert summary["cross_seed_gate"]["decision"] == "Development-No-Go"
    for row in summary["per_seed"]:
        assert row["primary_relative_improvement"]["factual_vs_hold_last"] == -0.01


def test_recomputes_pre_hold_screen_for_seed_15801_only(tmp_path: Path) -> None:
    first = _write_run(
        tmp_path, 15801, primary_effect=0.05, hold_effect=-0.01
    )
    payload = json.loads(first.read_text(encoding="utf-8"))
    del payload["screen"]["primary_pass"]["factual_vs_hold_last"]
    checkpoint_path = Path(payload["artifacts"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["report_screen"] = payload["screen"]
    torch.save(checkpoint, checkpoint_path)
    payload["artifacts"]["checkpoint_sha256"] = sha256_file(checkpoint_path)
    first.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = summarize_reports(
        [
            first,
            _write_run(tmp_path, 15802),
            _write_run(tmp_path, 15803),
        ]
    )

    first_row = summary["per_seed"][0]
    assert first_row["passed"] is False
    assert (
        first_row["primary_relative_improvement"]["factual_vs_hold_last"]
        == -0.01
    )
    assert summary["cross_seed_gate"]["decision"] == "Development-No-Go"


def test_requires_exact_distinct_fixed_seed_set(tmp_path: Path) -> None:
    first = _write_run(tmp_path, 15801)
    second = _write_run(tmp_path, 15802)

    with pytest.raises(ValueError, match="each fixed seed exactly once"):
        summarize_reports([first, second, second])
    with pytest.raises(ValueError, match="exactly three reports"):
        summarize_reports([first, second])


def test_rejects_smoke_and_inconsistent_report_screen(tmp_path: Path) -> None:
    smoke = _write_run(tmp_path, 15801, smoke=True)
    with pytest.raises(ValueError, match="non-smoke 2,000-step"):
        summarize_reports(
            [
                smoke,
                _write_run(tmp_path, 15802),
                _write_run(tmp_path, 15803),
            ]
        )

    root = tmp_path / "tampered"
    root.mkdir()
    first = _write_run(root, 15801)
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["screen"]["passed"] = False
    first.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="screen field passed"):
        summarize_reports(
            [first, _write_run(root, 15802), _write_run(root, 15803)]
        )


def test_rejects_cross_seed_data_provenance_mismatch(tmp_path: Path) -> None:
    reports = [
        _write_run(tmp_path, 15801),
        _write_run(tmp_path, 15802),
        _write_run(tmp_path, 15803, provenance_offset=1.0),
    ]

    with pytest.raises(ValueError, match="development_provenance_sha256 mismatch"):
        summarize_reports(reports)


def test_rejects_resolved_config_and_checkpoint_provenance_drift(tmp_path: Path) -> None:
    first = _write_run(tmp_path, 15801)
    resolved_path = first.parent / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["training"]["learning_rate"] = 1.0
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved config differs"):
        summarize_reports(
            [first, _write_run(tmp_path, 15802), _write_run(tmp_path, 15803)]
        )
