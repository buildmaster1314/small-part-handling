from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import scripts.evaluate_fcpg_lewm_factorial as factorial
from scripts.evaluate_fcpg_lewm_factorial import (
    _moving_goal_contract,
    _predictor_specs_by_grounding,
    _reference_prefix_actions,
    _validate_physical_pairing,
    crossed_seed_factorial_effects,
    factorial_effects,
)
from scripts.evaluate_p1_native_policy_rollout import visible_grounding_batch


def test_nominal_visible_grounding_overrides_only_fcpg_points() -> None:
    nominal = torch.arange(24, dtype=torch.float32).reshape(1, 2, 6, 2)
    batch = {
        "q_nominal": nominal,
        "fcpg_q_corrected": nominal + 9.0,
        "fcpg_reliability": torch.zeros(1, 2, 6),
        "obs": torch.ones(1, 2, 4),
        "q_true": nominal + 3.0,
    }
    result = visible_grounding_batch(batch, mode="nominal")
    assert torch.equal(result["fcpg_q_corrected"], nominal)
    assert torch.equal(result["fcpg_reliability"], torch.ones(1, 2, 6))
    assert "q_true" not in result
    assert torch.equal(result["obs"], batch["obs"])


def test_fcpg_visible_grounding_keeps_deployable_correction() -> None:
    batch = {
        "q_nominal": torch.zeros(1, 2, 3, 2),
        "fcpg_q_corrected": torch.ones(1, 2, 3, 2),
        "q_true": torch.full((1, 2, 3, 2), 2.0),
    }
    result = visible_grounding_batch(batch, mode="fcpg")
    assert torch.equal(result["fcpg_q_corrected"], batch["fcpg_q_corrected"])
    assert "q_true" not in result


def _rows() -> list[dict[str, object]]:
    values = {"f0_l0": 0.0, "f1_l0": 1.0, "f0_l1": 2.0, "f1_l1": 4.0}
    rows = []
    for episode in range(3):
        for cell, value in values.items():
            grounding = "fcpg" if cell.startswith("f1") else "nominal"
            rows.append(
                {
                    "physical_episode": episode,
                    "method": cell,
                    "score": value,
                    "final_success": value > 0.0,
                    "reset_digest": "reset",
                    "camera_sha256": "camera",
                    "schedule_sha256": "schedule",
                    "drop_onset_state_sha256": "onset",
                    "pre_drop_action_values": [0.1, 0.2],
                    "prefix_action_sha256": "action-prefix",
                    "prefix_native_token_sha256": f"{grounding}-native",
                    "prefix_sampled_visual_sha256": f"{grounding}-visual",
                    "prefix_bundle_sha256": f"{grounding}-bundle",
                    "prefix_manifest_sha256": f"{grounding}-manifest",
                }
            )
    return rows


def test_factorial_effects_report_main_effects_and_interaction() -> None:
    report = factorial_effects(
        _rows(), metric="score", bootstrap_samples=100, seed=3
    )
    contrasts = report["contrasts"]
    assert contrasts["fcpg_main_effect"]["estimate"] == 1.5
    assert contrasts["lewm_main_effect"]["estimate"] == 2.5
    assert contrasts["interaction"]["estimate"] == 1.0
    assert contrasts["full_vs_fcpg_only"]["estimate"] == 3.0
    assert contrasts["full_vs_lewm_only"]["estimate"] == 2.0


def test_crossed_factorial_resamples_model_seed_and_episode_as_separate_factors() -> None:
    rows = []
    for model_seed, interaction in ((11, 1.0), (12, 3.0)):
        for row in _rows():
            copied = dict(row)
            copied["model_seed"] = model_seed
            if copied["method"] == "f1_l1":
                copied["score"] = float(copied["score"]) + interaction - 1.0
            rows.append(copied)

    report = crossed_seed_factorial_effects(
        rows, metric="score", bootstrap_samples=200, seed=5
    )
    interaction = report["contrasts"]["interaction"]
    assert report["model_seeds"] == [11, 12]
    assert report["physical_episode_count"] == 3
    assert interaction["estimate"] == 2.0
    assert interaction["per_model_seed"] == {"11": 1.0, "12": 3.0}


def test_factorial_pairing_allows_different_grounding_but_not_actions() -> None:
    rows = _rows()
    _validate_physical_pairing(rows)
    rows[-1]["pre_drop_action_values"] = np.asarray([0.1, 0.3])
    try:
        _validate_physical_pairing(rows)
    except RuntimeError as error:
        assert "unequal pre-drop actions" in str(error)
    else:
        raise AssertionError("pairing audit accepted unequal actions")


def test_factorial_pairing_rejects_moving_goal_trajectory_mismatch() -> None:
    rows = _rows()
    for row in rows:
        row.update(
            {
                "moving_goal_enabled": True,
                "moving_goal_seed": 91,
                "moving_goal_amplitude_xy": [0.01, 0.006],
                "moving_goal_period_raw_steps": 440.0,
                "moving_goal_trajectory_sha256": "a" * 64,
                "moving_goal_trajectory_points": 336,
                "moving_goal_start_xyz": [0.1, 0.7, 0.2],
                "moving_goal_final_xyz": [0.11, 0.704, 0.2],
                "moving_goal_max_displacement_m": 0.012,
            }
        )
    _validate_physical_pairing(rows)
    rows[-1]["moving_goal_trajectory_sha256"] = "b" * 64
    try:
        _validate_physical_pairing(rows)
    except RuntimeError as error:
        assert "moving_goal_trajectory_sha256" in str(error)
    else:
        raise AssertionError("pairing audit accepted unequal moving-goal trajectories")


def test_moving_goal_contract_is_canonical_and_fail_closed() -> None:
    assert _moving_goal_contract({}) == {"enabled": False}
    assert _moving_goal_contract(
        {
            "moving_goal": {
                "enabled": True,
                "seed_base": 11,
                "amplitude_xy": [0.01, 0.006],
                "period_raw_steps": 440,
            }
        }
    ) == {
        "enabled": True,
        "seed_base": 11,
        "amplitude_xy": [0.01, 0.006],
        "period_raw_steps": 440.0,
    }


def test_factorial_requires_matched_nominal_and_fcpg_predictors(tmp_path) -> None:
    training_config = tmp_path / "training.json"
    training_config.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "predictors_by_grounding": {
            grounding: {
                "seed": 15801,
                "training_config": str(training_config),
                "checkpoint": str(checkpoint),
                "sha256": grounding * 64,
            }
            for grounding in ("nominal", "fcpg")
        }
    }
    specs = _predictor_specs_by_grounding(config, {}, seed=15801)
    assert set(specs) == {"nominal", "fcpg"}
    assert specs["nominal"][1].path == checkpoint.resolve()

    del config["predictors_by_grounding"]["nominal"]
    try:
        _predictor_specs_by_grounding(config, {}, seed=15801)
    except ValueError as error:
        assert "nominal and fcpg" in str(error)
    else:
        raise AssertionError("factorial accepted an unmatched predictor mapping")


def test_factorial_selects_checkpoint_from_seed_indexed_mapping(tmp_path) -> None:
    training_config = tmp_path / "training.json"
    training_config.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "seed15802.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "predictors_by_grounding": {
            grounding: {
                "training_config": str(training_config),
                "checkpoints": {
                    "15802": {
                        "checkpoint": str(checkpoint),
                        "sha256": "a" * 64,
                    }
                },
            }
            for grounding in ("nominal", "fcpg")
        }
    }
    specs = _predictor_specs_by_grounding(config, {}, seed=15802)
    assert specs["nominal"][1].seed == 15802
    assert specs["fcpg"][1].path == checkpoint.resolve()


def test_factorial_reconstructs_reference_prefix_actions() -> None:
    report = {"rows": _rows()}
    for row in report["rows"]:
        row["pre_drop_action_values"] = list(range(40))
    actions = _reference_prefix_actions(report, episodes=3, drop_starts=[2, 2, 2])
    assert set(actions) == {0, 1, 2}
    assert actions[0].shape == (2, 5, 4)
    np.testing.assert_array_equal(actions[0].reshape(-1), np.arange(40))


class _FakePolicy:
    def set_reliability_override(self, value: str) -> None:
        assert value == "ones"

    def requires_grad_(self, value: bool) -> "_FakePolicy":
        assert value is False
        return self

    def eval(self) -> "_FakePolicy":
        return self


def test_v2_main_reuses_grounding_prefixes_and_canonical_l0(
    tmp_path: Path, monkeypatch,
) -> None:
    policy_path = tmp_path / "policy.pt"
    corrector_path = tmp_path / "corrector.pt"
    source_path = tmp_path / "source.json"
    policy_path.write_bytes(b"policy")
    corrector_path.write_bytes(b"corrector")
    source_path.write_text(
        json.dumps(
            {
                "p1": {
                    "policy_checkpoint": str(policy_path),
                    "policy_checkpoint_sha256": hashlib.sha256(b"policy").hexdigest(),
                    "corrector_checkpoint": str(corrector_path),
                    "corrector_checkpoint_sha256": hashlib.sha256(
                        b"corrector"
                    ).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "factorial.json"
    bundle_dir = tmp_path / "prefix"
    canonical_dir = tmp_path / "canonical"
    config = {
        "schema_version": 2,
        "protocol": factorial.PROTOCOL,
        "classification": "test_development",
        "task": "pick-place-v3",
        "camera": "corner",
        "width": 32,
        "height": 32,
        "benchmark_seed": 1,
        "environment_seed_base": 10,
        "physical_episode_count": 1,
        "task_instance_indices": [0],
        "drop_start_decisions": [2],
        "drop_duration_decisions": 1,
        "horizon_raw_steps": 10,
        "occlusion_radius_px": 3,
        "occlusion_value_uint8": 0,
        "camera_rotation_axis_world": [0.0, 0.0, 1.0],
        "camera_rotation_deg": 5.0,
        "moving_goal": {
            "enabled": True,
            "seed_base": 900,
            "amplitude_xy": [0.01, 0.006],
            "period_raw_steps": 440.0,
        },
        "model_seed": 1,
        "canonical_model_seed": 1,
        "model_seeds": [1, 2, 3],
        "source_fresh_gate_config": str(source_path),
        "bootstrap_samples": 10,
        "bootstrap_seed": 7,
        "prefix_bundle_dir": str(bundle_dir),
        "output_dir": str(canonical_dir),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    spec = SimpleNamespace(path=tmp_path / "predictor.pt", sha256="a" * 64)
    spec.path.write_bytes(b"predictor")
    monkeypatch.setattr(factorial, "choose_device", lambda value: torch.device("cpu"))
    monkeypatch.setattr(
        factorial,
        "load_action_chunk_policy",
        lambda path, device: (_FakePolicy(), {}),
    )
    monkeypatch.setattr(
        factorial,
        "_predictor_specs_by_grounding",
        lambda config, source, seed: {
            "nominal": ({}, spec),
            "fcpg": ({}, spec),
        },
    )
    monkeypatch.setattr(
        factorial,
        "load_frozen_p1_native_checkpoint",
        lambda spec, expected_config, device: object(),
    )
    monkeypatch.setattr(
        factorial,
        "_predictor_report_records",
        lambda config, source, pair_specs, seed: {
            grounding: {
                "seed": seed,
                "training_config": f"{grounding}.json",
                "training_config_sha256": ("b" if grounding == "nominal" else "c") * 64,
                "strict_cache": f"{grounding}.npz",
                "checkpoint": str(spec.path),
                "checkpoint_sha256": spec.sha256,
            }
            for grounding in ("nominal", "fcpg")
        },
    )

    calls: list[dict[str, object]] = []

    def fake_rollout(*args, **kwargs):
        grounding = str(kwargs["visible_grounding_mode"])
        forced_actions = kwargs["forced_prefix_actions"]
        forced_tokens = kwargs["forced_prefix_tokens"]
        forced_visual = kwargs["forced_prefix_visual"]
        actions = (
            np.arange(40, dtype=np.float32).reshape(2, 5, 4)
            if forced_actions is None
            else np.asarray(forced_actions, dtype=np.float32)
        )
        token_value = 1.0 if grounding == "fcpg" else 0.0
        tokens = (
            np.full((2, 1, 3, 4), token_value, dtype=np.float32)
            if forced_tokens is None
            else np.asarray(forced_tokens, dtype=np.float32)
        )
        visual = (
            np.full((2, 1, 3, 5), token_value + 2.0, dtype=np.float32)
            if forced_visual is None
            else np.asarray(forced_visual, dtype=np.float32)
        )
        calls.append(
            {
                "cell": kwargs["result_method"],
                "seed": kwargs["model_seed"],
                "forced_actions": forced_actions is not None,
                "forced_tokens": forced_tokens is not None,
                "forced_visual": forced_visual is not None,
                "token_value": float(tokens.flat[0]),
                "moving_goal_seed": kwargs["moving_goal_seed"],
            }
        )
        action_sha = hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
        return {
            "method": kwargs["result_method"],
            "model_seed": kwargs["model_seed"],
            "reset_digest": "reset",
            "camera_sha256": "d" * 64,
            "schedule_sha256": "e" * 64,
            "drop_onset_state_sha256": "f" * 64,
            "moving_goal_enabled": True,
            "moving_goal_seed": kwargs["moving_goal_seed"],
            "moving_goal_amplitude_xy": list(kwargs["moving_goal_amplitude_xy"]),
            "moving_goal_period_raw_steps": kwargs[
                "moving_goal_period_raw_steps"
            ],
            "moving_goal_trajectory_sha256": "9" * 64,
            "moving_goal_trajectory_points": 11,
            "moving_goal_path_length_m": 0.01,
            "moving_goal_max_displacement_m": 0.009,
            "moving_goal_start_xyz": [0.0, 0.0, 0.2],
            "moving_goal_final_xyz": [0.009, 0.0, 0.2],
            "pre_drop_action_sha256": action_sha,
            "pre_drop_action_values": actions.reshape(-1).tolist(),
            "all_action_sha256": action_sha,
            "drop_start": 2,
            "success": 0.0,
            "sustained_success": 0.0,
            "final_success": 0.0,
            "post_drop_success_fraction": 0.0,
            "post_drop_maximum_success_streak": 0.0,
            "reward_sum": 0.0,
            "_prefix_native_tokens": tokens,
            "_prefix_sampled_visual": visual,
        }

    monkeypatch.setattr(factorial, "rollout_once", fake_rollout)
    assert factorial.main(["--config", str(config_path), "--device", "cpu"]) == 0
    canonical_report = canonical_dir / "factorial_report.json"
    assert [call["cell"] for call in calls] == list(factorial.RUN_ORDER)
    assert calls[0]["forced_actions"] is False
    assert calls[1]["forced_actions"] is True
    assert calls[1]["forced_tokens"] is False
    assert calls[2]["forced_tokens"] is True and calls[2]["token_value"] == 0.0
    assert calls[3]["forced_tokens"] is True and calls[3]["token_value"] == 1.0
    assert all(call["moving_goal_seed"] == 900 for call in calls)

    calls.clear()
    second_dir = canonical_dir.parent / "seed2"
    assert factorial.main(
        [
            "--config",
            str(config_path),
            "--model-seed",
            "2",
            "--device",
            "cpu",
            "--canonical-baseline-report",
            str(canonical_report),
        ]
    ) == 0
    assert [call["cell"] for call in calls] == list(factorial.DYNAMICS_CELLS)
    assert all(call["forced_actions"] for call in calls)
    assert all(call["forced_tokens"] and call["forced_visual"] for call in calls)
    assert all(call["moving_goal_seed"] == 900 for call in calls)
    canonical = json.loads(canonical_report.read_text(encoding="utf-8"))
    reused = json.loads((second_dir / "factorial_report.json").read_text(encoding="utf-8"))
    for cell in factorial.CANONICAL_BASELINE_CELLS:
        left = next(row for row in canonical["rows"] if row["method"] == cell)
        right = next(row for row in reused["rows"] if row["method"] == cell)
        assert right["model_seed"] == 2
        assert {**right, "model_seed": 1} == left
