"""Focused tests for the strict button P1 multi-seed runner."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from fcpg.button_fresh_multiseed_statistics import (
    CAMERA_FIELDS,
    METHOD_SPECS,
    audit_rollout_artifact,
    canonical_conditions,
    expected_episode_contracts,
    load_protocol_config,
    policy_checkpoint_path,
    rollout_artifact_paths,
    rollout_output_dir,
    validate_protocol_config,
)
from scripts.run_button_p1_fresh_multiseed import (
    DEFAULT_CONFIG,
    InputContract,
    StageJob,
    artifact_presence,
    build_stage_jobs,
    execute_job,
    expected_corrector_args,
    expected_policy_args,
    parse_stages,
    validate_corrector_checkpoint_payload,
    validate_policy_checkpoint_payload,
    validate_training_metrics,
)


def _argument(command: tuple[str, ...], name: str) -> str:
    return command[command.index(name) + 1]


def test_frozen_config_deduplicates_zero_and_rejects_protocol_drift() -> None:
    config = load_protocol_config(DEFAULT_CONFIG)
    conditions = canonical_conditions(config)

    assert [value.condition_id for value in conditions] == [
        "shared_clean",
        "rotation_5deg",
        "rotation_10deg",
        "translation_2cm",
        "translation_4cm",
    ]
    assert conditions[0].episodes == 80
    assert all(value.episodes == 20 for value in conditions[1:])
    assert len(
        {
            row["episode_identity"]
            for condition in conditions
            for row in expected_episode_contracts(config, condition)
        }
    ) == 160

    drifted = copy.deepcopy(config)
    drifted["evaluation"]["task_index_offset"] = 0
    with pytest.raises(ValueError, match="task_index_offset"):
        validate_protocol_config(drifted)


def test_plan_freezes_same_seed_training_and_fresh_rollout_commands() -> None:
    config = load_protocol_config(DEFAULT_CONFIG)
    jobs = build_stage_jobs(config, "cuda:7")

    assert {stage: len(values) for stage, values in jobs.items()} == {
        "corrector": 3,
        "cache": 3,
        "prototype": 1,
        "policy": 6,
        "rollout": 90,
        "statistics": 1,
    }
    corrector_two = next(job for job in jobs["corrector"] if job.metadata["seed"] == 2)
    assert _argument(corrector_two.command, "--corrector-arch") == "lewm"
    assert _argument(corrector_two.command, "--local-grid-size") == "5"
    assert _argument(corrector_two.command, "--local-radius-px") == "24"
    assert _argument(corrector_two.command, "--reliability-weight") == "0"
    assert _argument(corrector_two.command, "--sigreg-weight") == "0"
    assert _argument(corrector_two.command, "--epochs") == "50"
    assert _argument(corrector_two.command, "--batch-size") == "1024"
    assert _argument(corrector_two.command, "--seed") == "2"
    assert _argument(corrector_two.command, "--split-seed") == "2"
    assert _argument(corrector_two.command, "--device") == "cuda:7"

    fcpg_two = next(
        job
        for job in jobs["policy"]
        if job.metadata == {"seed": 2, "family": "fcpg"}
    )
    assert _argument(fcpg_two.command, "--policy-arch") == "legacy"
    assert _argument(fcpg_two.command, "--obs-horizon") == "2"
    assert _argument(fcpg_two.command, "--action-horizon") == "32"
    assert _argument(fcpg_two.command, "--encoder-layers") == "2"
    assert _argument(fcpg_two.command, "--decoder-layers") == "6"
    assert _argument(fcpg_two.command, "--warmup-steps") == "100"
    assert _argument(fcpg_two.command, "--fcpg-checkpoint").endswith(
        "correctors/seed_2/fcpg_transformer_residual_best.pt"
    )
    assert _argument(fcpg_two.command, "--grounding-cache").endswith(
        "button_p1_fresh_multiseed/seed_2"
    )

    jitter_one = next(
        job
        for job in jobs["policy"]
        if job.metadata == {"seed": 1, "family": "jitter"}
    )
    assert _argument(jitter_one.command, "--mode") == "geoprop"
    assert _argument(jitter_one.command, "--projection-jitter-std-px") == "8"
    assert _argument(jitter_one.command, "--projection-jitter-max-px") == "24"
    assert "--fcpg-checkpoint" not in jitter_one.command

    clean_native = next(
        job
        for job in jobs["rollout"]
        if job.metadata["seed"] == 0
        and job.metadata["condition"].condition_id == "shared_clean"
        and job.metadata["method"].name == "fcpg_native"
    )
    assert _argument(clean_native.command, "--episodes") == "80"
    assert _argument(clean_native.command, "--task-index-offset") == "2000"
    assert _argument(clean_native.command, "--env-seed-base") == "80000"
    assert _argument(clean_native.command, "--seed") == "20260718"
    assert _argument(clean_native.command, "--rotation-deg-min") == "0"
    assert _argument(clean_native.command, "--translation-cm-min") == "0"

    pnp_rotation = next(
        job
        for job in jobs["rollout"]
        if job.metadata["seed"] == 2
        and job.metadata["condition"].condition_id == "rotation_5deg"
        and job.metadata["method"].name == "pnp"
    )
    assert _argument(pnp_rotation.command, "--episodes") == "20"
    assert _argument(pnp_rotation.command, "--external-grounding") == "pnp"
    assert _argument(pnp_rotation.command, "--pnp-history-frames") == "3"
    assert _argument(pnp_rotation.command, "--pnp-max-reprojection-error-px") == "8"
    assert _argument(pnp_rotation.command, "--pnp-max-pose-rotation-deg") == "30"
    assert _argument(pnp_rotation.command, "--pnp-max-pose-translation-cm") == "20"
    assert _argument(pnp_rotation.command, "--prototype-radius-px") == "32"
    assert _argument(pnp_rotation.command, "--prototype-grid-size") == "9"


def test_checkpoint_metadata_validation_rejects_old_or_cross_seed_contracts() -> None:
    config = load_protocol_config(DEFAULT_CONFIG)
    data_path = str(Path(config["inputs"]["data"]).resolve())
    vision_cache = str(Path(config["inputs"]["vision_cache"]).resolve())
    corrector_args = expected_corrector_args(config, 0)
    corrector_args.update(
        {
            "data": config["inputs"]["data"],
            "vision_cache": config["inputs"]["vision_cache"],
        }
    )
    corrector = {
        "schema_version": 2,
        "checkpoint_type": "fcpg_corrector",
        "corrector_arch": "lewm",
        "has_reliability": False,
        "data": data_path,
        "vision_cache": vision_cache,
        "args": corrector_args,
        "model_state": {"weight": object()},
    }
    validate_corrector_checkpoint_payload(corrector, config, 0)
    wrong_seed = copy.deepcopy(corrector)
    wrong_seed["args"]["seed"] = 1
    with pytest.raises(ValueError, match="args.seed"):
        validate_corrector_checkpoint_payload(wrong_seed, config, 0)

    policy_args = expected_policy_args(config, "fcpg", 0)
    policy_args.update(
        {
            "data": config["inputs"]["data"],
            "vision_cache": config["inputs"]["vision_cache"],
            "fcpg_checkpoint": config["reuse"]["corrector_checkpoints"]["0"],
            "grounding_cache": config["reuse"]["grounding_caches"]["0"],
        }
    )
    policy = {
        "schema_version": 2,
        "checkpoint_type": "fcpg_action_policy",
        "policy_arch": "legacy",
        "data": data_path,
        "vision_cache": vision_cache,
        "grounding_cache": str(
            Path(config["reuse"]["grounding_caches"]["0"]).resolve()
        ),
        "args": policy_args,
        "model_state": {"weight": object()},
    }
    validate_policy_checkpoint_payload(policy, config, "fcpg", 0)
    old_policy = copy.deepcopy(policy)
    old_policy["policy_arch"] = "lewm-predictive"
    old_policy["args"]["policy_arch"] = "lewm-predictive"
    old_policy["args"]["warmup_steps"] = 300
    with pytest.raises(ValueError, match="legacy architecture"):
        validate_policy_checkpoint_payload(old_policy, config, "fcpg", 0)


def test_partial_artifact_fails_and_dry_run_does_not_execute(tmp_path: Path) -> None:
    config = load_protocol_config(DEFAULT_CONFIG)
    output = tmp_path / "artifact"
    required = (output / "one", output / "two")
    job = StageJob("prototype", "test", ("must-not-run",), output, required, {})
    assert artifact_presence(job) == "missing"
    status, result = execute_job(
        config,
        job,
        InputContract(1, 1, 1),
        dry_run=True,
    )
    assert (status, result) == ("planned", None)
    assert not output.exists()

    output.mkdir()
    required[0].write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial artifact"):
        artifact_presence(job)

    assert parse_stages("rollout,statistics") == ("rollout", "statistics")
    with pytest.raises(Exception, match="duplicates"):
        parse_stages("policy,policy")


def test_training_metrics_require_complete_full_epoch_sequence(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    payload = {
        "schema_version": 2,
        "policy_arch": "legacy",
        "best": {"epoch": 2},
        "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
        "complete": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    validate_training_metrics(
        path,
        expected_epochs=3,
        architecture_key="policy_arch",
        expected_architecture="legacy",
        label="policy",
    )

    payload["complete"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not marked complete"):
        validate_training_metrics(
            path,
            expected_epochs=3,
            architecture_key="policy_arch",
            expected_architecture="legacy",
            label="policy",
        )

    payload["complete"] = True
    payload["history"] = [{"epoch": 1}, {"epoch": 3}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="epochs 1..3"):
        validate_training_metrics(
            path,
            expected_epochs=3,
            architecture_key="policy_arch",
            expected_architecture="legacy",
            label="policy",
        )


def _write_rollout(config, tmp_path: Path) -> tuple[dict, Path, object, object]:
    modified = copy.deepcopy(config)
    modified["artifacts"]["rollout_root"] = str(tmp_path / "rollouts")
    modified["artifacts"]["policy_root"] = str(tmp_path / "policies")
    modified["artifacts"]["prototype_bank"] = str(tmp_path / "prototypes")
    condition = canonical_conditions(modified)[0]
    method = METHOD_SPECS["jitter"]
    output_dir = rollout_output_dir(modified, tmp_path, 0, condition, method)
    output_dir.mkdir(parents=True)
    csv_path, summary_path = rollout_artifact_paths(output_dir, method)
    rows = []
    for expected in expected_episode_contracts(modified, condition):
        row = {
            "episode": expected["episode"],
            "model": method.model_name,
            "task": expected["task"],
            "task_index": expected["raw_task_index"],
            "env_seed": expected["env_seed"],
            "success": 0.0,
            "actual_camera_translation_cm": 0.0,
            "actual_camera_translation_x_cm": 0.0,
            "actual_camera_translation_y_cm": 0.0,
            "actual_camera_translation_z_cm": 0.0,
            "actual_camera_rotation_axis_x": 0.0,
            "actual_camera_rotation_axis_y": 0.0,
            "actual_camera_rotation_axis_z": 1.0,
            "actual_camera_rotation_deg": 0.0,
        }
        rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": method.model_name,
        "source_mode": method.source_mode,
        "grounding_override": method.grounding_override,
        "external_grounding": method.external_grounding,
        "checkpoint": str(
            policy_checkpoint_path(modified, tmp_path, method.policy_family, 0)
        ),
        "prototype_bank": None,
        "task": "button-press-v3",
        "episodes": 80,
        "horizon": 100,
        "action_steps": 8,
        "translation_cm_min": 0.0,
        "translation_cm_max": 0.0,
        "rotation_deg_min": 0.0,
        "rotation_deg_max": 0.0,
        "success_rate": 0.0,
        "prototype_radius_px": 32.0,
        "prototype_grid_size": 9,
        "pnp_history_frames": 3,
        "pnp_max_reprojection_error_px": 8.0,
        "pnp_max_pose_rotation_deg": 30.0,
        "pnp_max_pose_translation_cm": 20.0,
        "reliability_override_request": "native",
    }
    summary_path.write_text(
        json.dumps({"summary": summary, "episodes": rows}), encoding="utf-8"
    )
    return modified, summary_path, condition, method


def test_rollout_audit_rejects_duplicate_raw_identity(tmp_path: Path) -> None:
    config = load_protocol_config(DEFAULT_CONFIG)
    modified, summary_path, condition, method = _write_rollout(config, tmp_path)
    records = audit_rollout_artifact(
        modified, tmp_path, 0, condition, method
    )
    assert len(records) == 80
    assert records[0].raw_task_index == 2000
    assert records[0].env_seed == 80000
    assert records[0].condition_id == "shared_clean"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["episodes"][1]["task_index"] = 2000
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="raw identity differs"):
        audit_rollout_artifact(modified, tmp_path, 0, condition, method)
