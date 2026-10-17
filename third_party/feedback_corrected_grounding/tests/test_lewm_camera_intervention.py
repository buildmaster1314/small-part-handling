from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.geometry import Camera, project_points
from fcpg.lewm_camera_intervention import (
    _axis_angle_matrix,
    _expected_episode_ledger,
    validate_confirmatory_camera_interventions,
)
from fcpg.lewm_provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DESIGN = (
    ROOT / "config/lewm/eval/pixel_observer_confirmatory_design_seed6144.json"
)


def _write_cohort(
    path: Path,
    source: dict[str, object],
    camera_model: dict[str, object],
    *,
    clean_extrinsic: bool = False,
) -> None:
    episodes, _digest = _expected_episode_ledger(source)
    repeat = 100
    task_name = np.repeat(
        np.asarray([record["task"] for record in episodes]), repeat
    )
    episode_id = np.repeat(
        np.asarray([record["episode"] for record in episodes], dtype=np.int32),
        repeat,
    )
    task_index = np.repeat(
        np.asarray([record["task_index"] for record in episodes], dtype=np.int32),
        repeat,
    )
    env_seed = np.repeat(
        np.asarray([record["env_seed"] for record in episodes], dtype=np.int64),
        repeat,
    )
    frame_id = np.tile(np.arange(repeat, dtype=np.int32), len(episodes))
    translation = np.repeat(
        np.stack([record["translation"] for record in episodes]), repeat, axis=0
    ).astype(np.float32)
    axis = np.repeat(
        np.stack([record["axis"] for record in episodes]), repeat, axis=0
    ).astype(np.float32)
    degrees = np.repeat(
        np.asarray([record["degrees"] for record in episodes], dtype=np.float32),
        repeat,
    )
    base_one = np.asarray(camera_model["base_extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(camera_model["camera_intrinsic"], dtype=np.float32)
    base = np.tile(base_one, (len(task_name), 1, 1))
    true = base.copy()
    episode_rotations = np.stack(
        [
            _axis_angle_matrix(record["axis"], float(record["degrees"])).T
            for record in episodes
        ]
    )
    rotations = np.repeat(episode_rotations, repeat, axis=0)
    true_rotations = np.einsum(
        "ij,njk->nik",
        base_one[:3, :3].astype(np.float64),
        rotations,
    )
    base_center = -base_one[:3, :3].T @ base_one[:3, 3]
    true_centers = base_center[None, :] + translation.astype(np.float64)
    true[:, :3, :3] = true_rotations.astype(np.float32)
    true[:, :3, 3] = -np.einsum(
        "nij,nj->ni", true_rotations, true_centers
    ).astype(np.float32)
    if clean_extrinsic:
        true = base.copy()

    points = np.asarray(
        [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [-0.1, 0.05, 2.0]],
        dtype=np.float32,
    )
    q_nominal_one, valid_nominal_one = project_points(
        points,
        Camera(intrinsic, base[0], width=224, height=224),
    )
    q_true_episodes = []
    valid_true_episodes = []
    for index in range(len(episodes)):
        pixels, valid = project_points(
            points,
            Camera(intrinsic, true[index * repeat], width=224, height=224),
        )
        q_true_episodes.append(pixels)
        valid_true_episodes.append(valid)
    camera_intrinsic = np.tile(intrinsic, (len(task_name), 1, 1))
    keypoints_xyz = np.tile(points, (len(task_name), 1, 1))
    q_true = np.repeat(
        np.stack(q_true_episodes).astype(np.float32), repeat, axis=0
    )
    valid_true = np.repeat(
        np.stack(valid_true_episodes), repeat, axis=0
    )
    q_nominal = np.tile(
        q_nominal_one.astype(np.float32), (len(task_name), 1, 1)
    )
    valid_nominal = np.tile(valid_nominal_one, (len(task_name), 1))

    np.savez(
        path,
        task_name=task_name,
        task_index=task_index,
        episode_id=episode_id,
        env_seed=env_seed,
        frame_id=frame_id,
        action_source=np.asarray(["scripted"] * len(task_name)),
        observation_action_alignment=np.asarray("pre_action"),
        keypoint_layout=np.asarray("control"),
        actual_camera_translation_xyz=translation,
        actual_camera_rotation_axis=axis,
        actual_camera_rotation_deg=degrees,
        actual_camera_translation_cm_min=np.asarray(
            source["actual_translation_cm"], dtype=np.float32
        ),
        actual_camera_translation_cm_max=np.asarray(
            source["actual_translation_cm"], dtype=np.float32
        ),
        actual_camera_rotation_deg_min=np.asarray(
            source["actual_rotation_abs_deg"], dtype=np.float32
        ),
        actual_camera_rotation_deg_max=np.asarray(
            source["actual_rotation_abs_deg"], dtype=np.float32
        ),
        actual_camera_randomization=np.asarray("episode"),
        actual_camera_clean_probability=np.asarray(0.0, dtype=np.float32),
        independent_rng_streams=np.asarray(True),
        rng_seed=np.asarray(source["rng_seed"], dtype=np.int64),
        task_generation_seed=np.asarray(
            source["task_generation_seed"], dtype=np.int64
        ),
        translation_cm=np.zeros(len(task_name), dtype=np.float32),
        yaw_deg=np.zeros(len(task_name), dtype=np.float32),
        camera_intrinsic=camera_intrinsic,
        keypoints_xyz=keypoints_xyz,
        q_true=q_true,
        q_nominal=q_nominal,
        valid_true=valid_true,
        valid_nominal=valid_nominal,
        base_extrinsic=base,
        nominal_extrinsic=base,
        true_extrinsic=true,
    )


def _fixture(
    tmp_path: Path,
    *,
    clean_rotation_extrinsic: bool = False,
) -> tuple[Path, list[dict[str, object]]]:
    design = json.loads(CANONICAL_DESIGN.read_text(encoding="utf-8"))
    report = []
    for name in ("rotation5", "translation4"):
        path = tmp_path / f"{name}.npz"
        design["sources"][name]["path"] = str(path)
        _write_cohort(
            path,
            design["sources"][name],
            design["camera_model"],
            clean_extrinsic=(name == "rotation5" and clean_rotation_extrinsic),
        )
        report.append(
            {
                "name": name,
                "data": str(path),
                "data_sha256": sha256_file(path),
            }
        )
    design_path = tmp_path / "confirmatory-design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    return design_path, report


def _mutate_cohort(
    report: list[dict[str, object]],
    name: str,
    mutator,
) -> None:
    record = next(value for value in report if value["name"] == name)
    path = Path(str(record["data"]))
    with np.load(path, allow_pickle=False) as raw:
        payload = {key: np.asarray(raw[key]) for key in raw.files}
    mutator(payload)
    np.savez(path, **payload)
    record["data_sha256"] = sha256_file(path)


def test_realized_camera_audit_accepts_frozen_rotation_and_translation(
    tmp_path: Path,
) -> None:
    design, report = _fixture(tmp_path)

    audit, bound_paths = validate_confirmatory_camera_interventions(
        design,
        report,
        repo_root=ROOT,
    )

    assert audit["passed"] is True
    assert audit["cohorts"]["rotation5"]["actual_rotation_abs_deg"] == 5.0
    assert audit["cohorts"]["translation4"]["actual_translation_cm"] == 4.0
    assert design.resolve() in bound_paths


def test_realized_camera_audit_rejects_clean_extrinsic_labeled_rotation5(
    tmp_path: Path,
) -> None:
    design, report = _fixture(tmp_path, clean_rotation_extrinsic=True)

    with pytest.raises(ValueError, match="realized camera rotation is invalid"):
        validate_confirmatory_camera_interventions(
            design,
            report,
            repo_root=ROOT,
        )


def test_realized_camera_audit_rejects_unregistered_source_path(
    tmp_path: Path,
) -> None:
    design, report = _fixture(tmp_path)
    report[0] = dict(report[0], data=str(tmp_path / "renamed.npz"))

    with pytest.raises(ValueError, match="data path differs"):
        validate_confirmatory_camera_interventions(
            design,
            report,
            repo_root=ROOT,
        )


def test_realized_camera_audit_rejects_rank_expanded_translation(
    tmp_path: Path,
) -> None:
    design, report = _fixture(tmp_path)

    def expand_rank(payload):
        payload["actual_camera_translation_xyz"] = payload[
            "actual_camera_translation_xyz"
        ][:, None, :]

    _mutate_cohort(report, "rotation5", expand_rank)

    with pytest.raises(
        ValueError, match="actual_camera_translation_xyz must have shape"
    ):
        validate_confirmatory_camera_interventions(
            design,
            report,
            repo_root=ROOT,
        )


def test_realized_camera_audit_rejects_non_rigid_extrinsic(
    tmp_path: Path,
) -> None:
    design, report = _fixture(tmp_path)

    def scale_rotation(payload):
        payload["base_extrinsic"] = payload["base_extrinsic"].copy()
        payload["nominal_extrinsic"] = payload["nominal_extrinsic"].copy()
        payload["base_extrinsic"][:, :3, :3] *= 2.0
        payload["nominal_extrinsic"][:, :3, :3] *= 2.0

    _mutate_cohort(report, "rotation5", scale_rotation)

    with pytest.raises(ValueError, match="base_extrinsic rotation is not orthonormal"):
        validate_confirmatory_camera_interventions(
            design,
            report,
            repo_root=ROOT,
        )


def test_realized_camera_audit_rejects_intrinsic_model_drift(
    tmp_path: Path,
) -> None:
    design, report = _fixture(tmp_path)

    def drift_intrinsic(payload):
        payload["camera_intrinsic"] = payload["camera_intrinsic"].copy()
        payload["camera_intrinsic"][:, 0, 0] += 0.25

    _mutate_cohort(report, "rotation5", drift_intrinsic)

    with pytest.raises(
        ValueError,
        match="camera_intrinsic differs from preregistered camera model",
    ):
        validate_confirmatory_camera_interventions(
            design,
            report,
            repo_root=ROOT,
        )


def test_realized_camera_audit_rejects_base_camera_model_drift(
    tmp_path: Path,
) -> None:
    design, report = _fixture(tmp_path)

    def drift_base(payload):
        payload["base_extrinsic"] = payload["base_extrinsic"].copy()
        payload["nominal_extrinsic"] = payload["nominal_extrinsic"].copy()
        payload["base_extrinsic"][:, 0, 3] += 0.01
        payload["nominal_extrinsic"][:, 0, 3] += 0.01

    _mutate_cohort(report, "rotation5", drift_base)

    with pytest.raises(
        ValueError,
        match="base_extrinsic differs from preregistered camera model",
    ):
        validate_confirmatory_camera_interventions(
            design,
            report,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    ("field", "error"),
    (
        ("q_true", "q_true differs from the frozen geometric reprojection"),
        ("q_nominal", "q_nominal differs from the frozen geometric reprojection"),
        ("valid_true", "valid_true differs from the frozen projection validity"),
        (
            "valid_nominal",
            "valid_nominal differs from the frozen projection validity",
        ),
    ),
)
def test_realized_camera_audit_rejects_tampered_projection_fields(
    tmp_path: Path,
    field: str,
    error: str,
) -> None:
    design, report = _fixture(tmp_path)

    def tamper(payload):
        payload[field] = payload[field].copy()
        if field.startswith("valid_"):
            payload[field][0, 0] = not bool(payload[field][0, 0])
        else:
            payload[field][0, 0, 0] += 1.0

    _mutate_cohort(report, "rotation5", tamper)

    with pytest.raises(ValueError, match=error):
        validate_confirmatory_camera_interventions(
            design,
            report,
            repo_root=ROOT,
        )
