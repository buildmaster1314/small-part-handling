import argparse
import json
from pathlib import Path

import pytest

from scripts.run_action_chunk_rollout_matrix import PolicySpec
from scripts.run_external_grounding_matrix import (
    ExternalGroundingJob,
    build_jobs,
    completed_job_matches,
    rollout_command,
)


def rollout_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        camera="corner",
        width=224,
        height=224,
        episodes=30,
        task_index_offset=1000,
        horizon=100,
        action_steps=8,
        seed=7,
        env_seed_base=50000,
        device="cuda:0",
        prototype_bank=str(tmp_path / "bank"),
        prototype_radius_px=32.0,
        prototype_grid_size=9,
        pnp_history_frames=3,
        pnp_max_reprojection_error_px=8.0,
        pnp_max_pose_rotation_deg=30.0,
        pnp_max_pose_translation_cm=20.0,
    )


def test_build_jobs_covers_matched_methods_and_oracle(tmp_path: Path) -> None:
    jobs = build_jobs(
        [PolicySpec("seed0", "model.pt", "fcpg")],
        [("pick", "pick-place-v3")],
        [5.0],
        ["nominal", "native", "prototype", "pnp", "oracle"],
        tmp_path,
        "rotation",
    )
    assert len(jobs) == 5
    assert {job.method for job in jobs} == {"nominal", "native", "prototype", "pnp", "oracle"}
    assert {job.model_name for job in jobs} == {
        "fcpg_nominal",
        "fcpg_rho_ones",
        "fcpg_prototype",
        "fcpg_pnp",
        "fcpg_oracle",
    }
    with pytest.raises(ValueError):
        build_jobs(
            [PolicySpec("bad", "model.pt", "geoprop")],
            [("pick", "pick-place-v3")],
            [5.0],
            ["native"],
            tmp_path,
            "rotation",
        )


def test_rollout_command_keeps_residual_only_and_external_modes_separate(
    tmp_path: Path,
) -> None:
    args = rollout_args(tmp_path)
    native = ExternalGroundingJob(
        "seed0",
        "model.pt",
        "pick",
        "pick-place-v3",
        5.0,
        "native",
        str(tmp_path / "native"),
    )
    native_command = rollout_command(native, args)
    assert native_command[native_command.index("--reliability-override") + 1] == "ones"
    assert "--external-grounding" not in native_command

    pnp = ExternalGroundingJob(
        "seed0",
        "model.pt",
        "pick",
        "pick-place-v3",
        4.0,
        "pnp",
        str(tmp_path / "pnp"),
        "translation",
    )
    pnp_command = rollout_command(pnp, args)
    assert pnp_command[pnp_command.index("--external-grounding") + 1] == "pnp"
    assert pnp_command[pnp_command.index("--translation-cm-min") + 1] == "4.0"
    assert pnp_command[pnp_command.index("--pnp-max-pose-rotation-deg") + 1] == "30.0"
    assert "--reliability-override" not in pnp_command

    oracle = ExternalGroundingJob(
        "seed0",
        "model.pt",
        "pick",
        "pick-place-v3",
        5.0,
        "oracle",
        str(tmp_path / "oracle"),
    )
    oracle_command = rollout_command(oracle, args)
    assert oracle_command[oracle_command.index("--grounding-override") + 1] == "oracle"
    assert "--external-grounding" not in oracle_command


def test_completed_job_validates_external_provenance(tmp_path: Path) -> None:
    args = rollout_args(tmp_path)
    job = ExternalGroundingJob(
        "seed0",
        "model.pt",
        "pick",
        "pick-place-v3",
        5.0,
        "prototype",
        str(tmp_path / "prototype"),
    )
    job.csv_path.parent.mkdir(parents=True)
    job.csv_path.write_text("episode\n0\n", encoding="utf-8")
    summary = {
        "checkpoint": "model.pt",
        "model": "fcpg_prototype",
        "source_mode": "fcpg",
        "task": "pick-place-v3",
        "episodes": 30,
        "horizon": 100,
        "action_steps": 8,
        "grounding_override": "native",
        "external_grounding": "prototype",
        "reliability_override_request": "native",
        "prototype_bank": str((tmp_path / "bank").resolve()),
        "prototype_radius_px": 32.0,
        "prototype_grid_size": 9,
        "pnp_history_frames": 3,
        "pnp_max_reprojection_error_px": 8.0,
        "pnp_max_pose_rotation_deg": 30.0,
        "pnp_max_pose_translation_cm": 20.0,
        "pnp_pose_prior_mode": "camera_center_v1",
        "rotation_deg_min": 5.0,
        "rotation_deg_max": 5.0,
    }
    job.summary_path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    assert completed_job_matches(job, args)
    summary["external_grounding"] = "none"
    job.summary_path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    assert not completed_job_matches(job, args)
