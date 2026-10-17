import argparse
import json
from pathlib import Path

import pytest

from scripts.run_action_chunk_rollout_matrix import (
    PolicySpec,
    RolloutJob,
    build_jobs,
    completed_job_matches,
    normalized_auc,
    parse_named_value,
    rollout_command,
    severity_token,
)


def test_parse_named_value_and_severity_token() -> None:
    assert parse_named_value("sig0=checkpoint.pt") == ("sig0", "checkpoint.pt")
    assert severity_token(0.0) == "rot0"
    assert severity_token(2.5) == "rot2p5"
    assert severity_token(-1.0) == "rotm1"
    assert severity_token(4.0, "translation") == "trans4"
    with pytest.raises(argparse.ArgumentTypeError):
        parse_named_value("missing-separator")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_named_value("bad label=value")


def test_build_jobs_pairs_every_policy_task_severity(tmp_path: Path) -> None:
    jobs = build_jobs(
        [
            PolicySpec("sig0", "a.pt", "fcpg"),
            PolicySpec("sig002", "b.pt", "fcpg"),
        ],
        [("pick", "pick-place-v3")],
        [0.0, 10.0],
        tmp_path,
    )
    assert len(jobs) == 8
    assert {job.grounding_mode for job in jobs} == {"native", "nominal"}
    assert {job.policy_label for job in jobs} == {"sig0", "sig002"}
    assert jobs[0].output_dir == str(tmp_path / "sig0" / "pick" / "rot0" / "nominal")


def test_completed_job_requires_matching_provenance(tmp_path: Path) -> None:
    job = RolloutJob("sig0", "model.pt", "pick", "pick-place-v3", 5.0, "native", str(tmp_path))
    job.csv_path.write_text("episode\n0\n", encoding="utf-8")
    payload = {
        "summary": {
            "checkpoint": "model.pt",
            "source_mode": "fcpg",
            "grounding_override": "native",
            "reliability_override_request": "native",
            "task": "pick-place-v3",
            "episodes": 60,
            "horizon": 100,
            "action_steps": 8,
            "rotation_deg_min": 5.0,
            "rotation_deg_max": 5.0,
        }
    }
    job.summary_path.write_text(json.dumps(payload), encoding="utf-8")
    assert completed_job_matches(job, 60, 100, 8)
    assert not completed_job_matches(job, 20, 100, 8)


def test_rollout_command_and_auc(tmp_path: Path) -> None:
    job = RolloutJob("sig0", "model.pt", "pick", "pick-place-v3", 10.0, "nominal", str(tmp_path))
    args = argparse.Namespace(
        camera="corner",
        width=224,
        height=224,
        episodes=60,
        task_index_offset=0,
        horizon=100,
        action_steps=8,
        seed=0,
        env_seed_base=1000,
        device="cuda:0",
    )
    command = rollout_command(job, args)
    assert command[command.index("--rotation-deg-min") + 1] == "10.0"
    assert command[command.index("--grounding-override") + 1] == "nominal"

    translation_job = RolloutJob(
        "sig0",
        "model.pt",
        "pick",
        "pick-place-v3",
        4.0,
        "native",
        str(tmp_path / "translation"),
        "translation",
    )
    translation_command = rollout_command(translation_job, args)
    assert translation_command[translation_command.index("--translation-cm-min") + 1] == "4.0"
    assert translation_command[translation_command.index("--translation-cm-max") + 1] == "4.0"
    assert "--rotation-deg-min" not in translation_command
    assert normalized_auc([(0.0, 1.0), (10.0, 0.5), (20.0, 0.0)]) == pytest.approx(0.5)


def test_translation_job_requires_translation_provenance(tmp_path: Path) -> None:
    jobs = build_jobs(
        [PolicySpec("sig0", "model.pt", "fcpg")],
        [("pick", "pick-place-v3")],
        [4.0],
        tmp_path,
        perturbation="translation",
    )
    job = jobs[0]
    assert job.perturbation == "translation"
    assert Path(job.output_dir).parts[-2:] == ("trans4", "nominal")
    job.csv_path.parent.mkdir(parents=True, exist_ok=True)
    job.csv_path.write_text("episode\n0\n", encoding="utf-8")
    payload = {
        "summary": {
            "checkpoint": "model.pt",
            "source_mode": "fcpg",
            "grounding_override": "nominal",
            "task": "pick-place-v3",
            "episodes": 60,
            "horizon": 100,
            "action_steps": 8,
            "translation_cm_min": 4.0,
            "translation_cm_max": 4.0,
        }
    }
    job.summary_path.write_text(json.dumps(payload), encoding="utf-8")
    assert completed_job_matches(job, 60, 100, 8)
    payload["summary"]["translation_cm_max"] = 2.0
    job.summary_path.write_text(json.dumps(payload), encoding="utf-8")
    assert not completed_job_matches(job, 60, 100, 8)


def test_mixed_policy_modes_use_real_output_names(tmp_path: Path) -> None:
    jobs = build_jobs(
        [
            PolicySpec("jitter", "geoprop.pt", "geoprop"),
            PolicySpec("fcpg", "fcpg.pt", "fcpg"),
        ],
        [("pick", "pick-place-v3")],
        [5.0],
        tmp_path,
        reliability_override="ones",
    )
    assert len(jobs) == 3
    jitter = next(job for job in jobs if job.policy_label == "jitter")
    assert jitter.grounding_mode == "native"
    assert jitter.model_name == "geoprop"
    assert jitter.csv_path.name == "geoprop_rollout.csv"
    fcpg_nominal = next(
        job
        for job in jobs
        if job.policy_label == "fcpg" and job.grounding_mode == "nominal"
    )
    assert fcpg_nominal.model_name == "fcpg_nominal"
    fcpg_native = next(
        job
        for job in jobs
        if job.policy_label == "fcpg" and job.grounding_mode == "native"
    )
    assert fcpg_native.model_name == "fcpg_rho_ones"
