from __future__ import annotations

import json

import pytest
import torch

from fcpg.lewm_posterior_dagger import (
    PosteriorDaggerResidualHead,
    save_posterior_dagger_checkpoint,
)

from scripts.evaluate_lewm_posterior_dagger_rollout import (
    PROTOCOL,
    REPORT_METRICS,
    _checkpoint_paths,
    _call_head_loader,
    _dagger_api,
    _expected_dagger_iterations,
    _validate_config,
    condition_rollout_kwargs,
    summarize_condition,
    validate_checkpoint_metadata,
    validate_closed_loop_panel,
)


SEEDS = (11, 12)


def _method_rows(episode: int, *, factual: tuple[float, float]) -> list[dict[str, object]]:
    common = {
        "condition": "joint",
        "task": "pick-place-v3",
        "physical_episode": episode,
        "task_index": 10_000 + episode,
        "environment_seed": 20_000 + episode,
        "drop_start": 6,
        "drop_duration": 6,
        "camera_rotation_deg": 5.0,
        "camera_rotation_axis_world": [1.0, 0.0, 0.0],
        "pre_drop_action_values": [0.1, 0.2, 0.3, 0.4],
        "active_decisions_observed": 6,
        "recovery_exact_native_bypass": True,
    }

    def row(method: str, head_seed: int | None, value: float) -> dict[str, object]:
        metrics = {metric: value for metric in REPORT_METRICS}
        metrics["reward_sum"] = 10.0 * value
        metrics["post_drop_maximum_success_streak"] = 5.0 * value
        metrics["recovery_exact_native_bypass"] = True
        return {**common, "method": method, "head_seed": head_seed, **metrics}

    return [
        row("factual_base", None, 0.0),
        row("factual_dagger_s11", 11, factual[0]),
        row("factual_dagger_s12", 12, factual[1]),
        row("no_action_dagger_s11", 11, 0.25),
        row("no_action_dagger_s12", 12, 0.25),
        row("geometry_dagger_s11", 11, 0.0),
        row("geometry_dagger_s12", 12, 0.0),
    ]


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "classification": "development_only",
        "task": "pick-place-v3",
        "camera": "corner",
        "width": 224,
        "height": 224,
        "action_steps_per_decision": 5,
        "drop_duration_decisions": 6,
        "drop_start_min_decision": 6,
        "drop_start_max_decision": 10,
        "occlusion_radius_px": 24,
        "occlusion_value_uint8": 0,
        "occlusion_center_key": "q_nominal",
        "training": {
            "steps": 2000,
            "dagger_iteration": 1,
            "seeds": list(SEEDS),
            "output_dir": "unused",
        },
        "rollout": {
            "episodes": 3,
            "horizon_raw_steps": 335,
            "bootstrap_samples": 100,
            "conditions": [
                {
                    "name": "joint",
                    "primary": True,
                    "dropout_enabled": True,
                    "camera_rotation_deg": 5.0,
                    "camera_rotation_axis_world": [1.0, 0.0, 0.0],
                },
                {
                    "name": "clean",
                    "primary": False,
                    "dropout_enabled": False,
                    "camera_rotation_deg": 0.0,
                    "camera_rotation_axis_world": [0.0, 0.0, 1.0],
                },
            ],
        },
    }


def test_seed_macro_is_averaged_before_episode_bootstrap() -> None:
    rows = []
    rows.extend(_method_rows(0, factual=(1.0, 0.0)))
    rows.extend(_method_rows(1, factual=(1.0, 1.0)))
    result = summarize_condition(
        rows,
        condition="joint",
        head_seeds=SEEDS,
        dropout_enabled=True,
        bootstrap_samples=200,
        seed=3,
    )
    assert result["physical_episode_count"] == 2
    assert result["head_seed_count"] == 2
    assert result["method_summary"]["factual_dagger"]["sustained_success"] == pytest.approx(
        0.75
    )
    comparison = result["comparisons"]["factual_dagger_vs_factual_base"]
    assert comparison["metrics"]["sustained_success"]["delta"] == pytest.approx(0.75)
    assert comparison["metrics"]["reward_sum"]["delta"] == pytest.approx(7.5)
    assert result["primary_metric"] == "sustained_success"


def test_incomplete_seed_panel_is_rejected() -> None:
    rows = _method_rows(0, factual=(1.0, 1.0))
    rows = [row for row in rows if row["method"] != "geometry_dagger_s12"]
    with pytest.raises(ValueError, match="incomplete rollout panel"):
        validate_closed_loop_panel(
            rows,
            condition="joint",
            head_seeds=SEEDS,
            dropout_enabled=True,
        )


def test_duplicate_and_unpaired_rows_are_rejected() -> None:
    rows = _method_rows(0, factual=(1.0, 1.0))
    rows.append(dict(rows[-1]))
    with pytest.raises(ValueError, match="duplicate rollout row"):
        validate_closed_loop_panel(
            rows,
            condition="joint",
            head_seeds=SEEDS,
            dropout_enabled=True,
        )


def test_pairing_allows_post_drop_trajectory_divergence() -> None:
    rows = _method_rows(0, factual=(1.0, 1.0))
    for index, row in enumerate(rows):
        row["post_drop_action_trace"] = [float(index)]
    episodes = validate_closed_loop_panel(
        rows,
        condition="joint",
        head_seeds=SEEDS,
        dropout_enabled=True,
    )
    assert episodes == (("pick-place-v3", 0),)


def test_pairing_rejects_a_divergent_clean_prefix() -> None:
    rows = _method_rows(0, factual=(1.0, 1.0))
    rows[-1]["pre_drop_action_values"] = [0.1, 0.2, 0.3, 0.5]
    with pytest.raises(ValueError, match="divergent clean prefix"):
        validate_closed_loop_panel(
            rows,
            condition="joint",
            head_seeds=SEEDS,
            dropout_enabled=True,
        )


def test_primary_condition_requires_joint_five_degree_dropout() -> None:
    config = _config()
    _validate_config(config)
    primary = config["rollout"]["conditions"][0]
    kwargs = condition_rollout_kwargs(primary)
    assert kwargs == {
        "camera_rotation_deg": 5.0,
        "camera_rotation_axis_world": (1.0, 0.0, 0.0),
    }
    primary["dropout_enabled"] = False
    with pytest.raises(ValueError, match="primary condition"):
        _validate_config(config)


def test_optional_clean_no_drop_condition_is_allowed() -> None:
    config = _config()
    _validate_config(config)
    clean = config["rollout"]["conditions"][1]
    assert condition_rollout_kwargs(clean)["camera_rotation_deg"] == 0.0


def test_dropout_center_must_be_q_nominal() -> None:
    config = _config()
    config["occlusion_center_key"] = "q_corrected"
    with pytest.raises(ValueError, match="must be q_nominal"):
        _validate_config(config)


def test_negative_fixed_dagger_iteration_is_rejected() -> None:
    config = _config()
    config["training"]["dagger_iteration"] = -1
    with pytest.raises(ValueError, match="dagger_iteration"):
        _validate_config(config)


def test_latest_dagger_iteration_can_be_read_from_training_report(tmp_path) -> None:
    report = {
        "protocol": PROTOCOL,
        "optimizer_steps": 2000,
        "methods": [
            "complementary_factual",
            "complementary_no_action",
            "geometry_hold",
        ],
        "seeds": list(SEEDS),
        "training": {
            method: {"dagger_iterations": [0, 1, 2]}
            for method in (
                "complementary_factual",
                "complementary_no_action",
                "geometry_hold",
            )
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    iterations = _expected_dagger_iterations(
        {
            "steps": 2000,
            "seeds": list(SEEDS),
            "output_dir": str(tmp_path),
        },
        protocol=PROTOCOL,
    )
    assert iterations == {
        "complementary_factual": 2,
        "complementary_no_action": 2,
        "geometry_hold": 2,
    }


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"method": "geometry_hold"}, "method mismatch"),
        ({"seed": 99}, "seed mismatch"),
        ({"steps": 1999}, "not latest@2000"),
        ({"protocol": "wrong"}, "protocol mismatch"),
        ({"dagger_iteration": 0}, "iteration mismatch"),
    ],
)
def test_checkpoint_identity_is_exact(update: dict[str, object], message: str) -> None:
    metadata: dict[str, object] = {
        "protocol": "dagger-head-v1",
        "method": "complementary_factual",
        "seed": 11,
        "steps": 2000,
        "dagger_iteration": 1,
    }
    metadata.update(update)
    with pytest.raises(ValueError, match=message):
        validate_checkpoint_metadata(
            metadata,
            method="complementary_factual",
            seed=11,
            steps=2000,
            protocol="dagger-head-v1",
            dagger_iteration=1,
        )


def test_checkpoint_identity_accepts_exact_latest_endpoint() -> None:
    metadata = validate_checkpoint_metadata(
        {
            "protocol": "dagger-head-v1",
            "method": "complementary_factual",
            "seed": 11,
            "optimizer_step": 2000,
            "dagger_iteration": 1,
        },
        method="complementary_factual",
        seed=11,
        steps=2000,
        protocol="dagger-head-v1",
        dagger_iteration=1,
    )
    assert metadata["optimizer_step"] == 2000


def test_checkpoint_identity_accepts_loader_metadata_without_protocol() -> None:
    metadata = validate_checkpoint_metadata(
        {
            "method": "complementary_factual",
            "seed": 11,
            "steps": 2000,
            "dagger_iteration": 2,
        },
        method="complementary_factual",
        seed=11,
        steps=2000,
        protocol=PROTOCOL,
        dagger_iteration=2,
    )
    assert metadata["dagger_iteration"] == 2


def test_actual_dagger_api_exports_checkpoint_loader() -> None:
    protocol, methods, loader = _dagger_api()
    assert protocol == PROTOCOL
    assert methods == (
        "complementary_factual",
        "complementary_no_action",
        "geometry_hold",
    )
    assert loader.__name__ == "load_posterior_dagger_checkpoint"


def test_actual_checkpoint_loader_receives_exact_endpoint_identity(tmp_path) -> None:
    path = tmp_path / "complementary_factual_seed11_latest.pt"
    save_posterior_dagger_checkpoint(
        path,
        PosteriorDaggerResidualHead(model_dim=8, hidden_dim=12),
        method="complementary_factual",
        seed=11,
        steps=2000,
        dagger_iteration=2,
    )
    _protocol, _methods, loader = _dagger_api()
    head, metadata = _call_head_loader(
        loader,
        path,
        device=torch.device("cpu"),
        method="complementary_factual",
        seed=11,
        steps=2000,
        dagger_iteration=2,
    )
    assert head.training is False
    assert not any(parameter.requires_grad for parameter in head.parameters())
    assert metadata["dagger_iteration"] == 2


def test_explicit_checkpoint_panel_can_be_filtered_to_requested_seed() -> None:
    records = [
        {
            "method": method,
            "seed": seed,
            "path": f"outputs/{method}_seed{seed}_latest.pt",
        }
        for method in (
            "complementary_factual",
            "complementary_no_action",
            "geometry_hold",
        )
        for seed in SEEDS
    ]
    paths = _checkpoint_paths({"checkpoints": records}, seeds=(11,))
    assert set(paths) == {
        ("complementary_factual", 11),
        ("complementary_no_action", 11),
        ("geometry_hold", 11),
    }
