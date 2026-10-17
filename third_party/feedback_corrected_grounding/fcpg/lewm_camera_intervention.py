"""Frozen realized-camera audits for PixelObserver confirmatory cohorts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fcpg.lewm_provenance import sha256_file
from fcpg.mujoco_camera import sample_camera_pose_offset


PROTOCOL = "lewm_pixel_observer_confirmatory_design_v1"
REQUIRED_TASKS = (
    "pick-place-v3",
    "button-press-v3",
    "reach-v3",
    "push-v3",
)
REQUIRED_COHORTS = ("rotation5", "translation4")
LEDGER_DOMAIN = b"pixel-observer-transform-ledger-v1\0"
GEOMETRY_ATOL = 2e-6
SE3_ATOL = 2e-5
INTRINSIC_ATOL = 2e-5
PROJECTION_ATOL_PX = 5e-3
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CONTROL_KEYPOINTS = 3


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be an array")
    return value


def _resolve(value: object, *, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("required path is missing")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _scalar(raw: np.lib.npyio.NpzFile, key: str) -> object:
    value = np.asarray(raw[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    return value.reshape(-1)[0].item()


def _axis_angle_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    if abs(radians) <= 1e-15:
        return np.eye(3, dtype=np.float64)
    vector = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("nonzero rotation requires a nonzero axis")
    x, y, z = vector / norm
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(radians) * skew
        + (1.0 - math.cos(radians)) * (skew @ skew)
    )


def _update_ledger(
    digest: Any,
    *,
    task: str,
    episode: int,
    env_seed: int,
    task_index: int,
    translation: np.ndarray,
    axis: np.ndarray,
    degrees: float,
) -> None:
    digest.update(task.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        np.asarray([episode, env_seed, task_index], dtype="<i8").tobytes()
    )
    digest.update(np.asarray(translation, dtype="<f4").tobytes())
    digest.update(np.asarray(axis, dtype="<f4").tobytes())
    digest.update(np.asarray([degrees], dtype="<f4").tobytes())


def _expected_episode_ledger(
    source: Mapping[str, Any],
) -> tuple[list[dict[str, object]], str]:
    seed = int(source["rng_seed"])
    translation_cm = float(source["actual_translation_cm"])
    rotation_deg = float(source["actual_rotation_abs_deg"])
    env_seed_base = int(source["env_seed_base"])
    camera_seed = np.random.SeedSequence(seed).spawn(3)[1]
    rng = np.random.default_rng(camera_seed)
    digest = hashlib.sha256()
    digest.update(LEDGER_DOMAIN)
    records: list[dict[str, object]] = []
    for task_offset, task in enumerate(REQUIRED_TASKS):
        for episode in range(30):
            translation, axis, signed_degrees = sample_camera_pose_offset(
                rng,
                translation_cm,
                rotation_deg,
                0.0,
                translation_cm_min=translation_cm,
                rotation_deg_min=rotation_deg,
            )
            env_seed = env_seed_base + task_offset * 1_000_000 + episode
            task_index = episode
            record = {
                "task": task,
                "episode": episode,
                "env_seed": env_seed,
                "task_index": task_index,
                "translation": np.asarray(translation, dtype=np.float32),
                "axis": np.asarray(axis, dtype=np.float32),
                "degrees": float(np.float32(signed_degrees)),
            }
            records.append(record)
            _update_ledger(digest, **record)
    return records, digest.hexdigest()


def _camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return -np.einsum("nji,nj->ni", rotation, translation)


def _require_shape(
    value: np.ndarray,
    expected: tuple[int, ...],
    *,
    name: str,
) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def _validate_extrinsic_batch(
    value: np.ndarray,
    *,
    count: int,
    name: str,
) -> None:
    _require_shape(value, (count, 4, 4), name=name)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    expected_last_row = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(
        value[:, 3, :], expected_last_row, rtol=0.0, atol=SE3_ATOL
    ):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = value[:, :3, :3]
    gram = np.matmul(np.swapaxes(rotation, 1, 2), rotation)
    if not np.allclose(
        gram, np.eye(3, dtype=np.float64), rtol=0.0, atol=SE3_ATOL
    ):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = np.linalg.det(rotation)
    if not np.allclose(determinant, 1.0, rtol=0.0, atol=SE3_ATOL):
        raise ValueError(f"{name} rotation determinant is not +1")


def _validate_intrinsic_batch(
    intrinsic: np.ndarray,
    *,
    count: int,
) -> None:
    _require_shape(intrinsic, (count, 3, 3), name="camera_intrinsic")
    if not np.isfinite(intrinsic).all():
        raise ValueError("camera_intrinsic contains non-finite values")
    expected_last_row = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(
        intrinsic[:, 2, :],
        expected_last_row,
        rtol=0.0,
        atol=INTRINSIC_ATOL,
    ):
        raise ValueError("camera_intrinsic has an invalid homogeneous row")
    if np.any(intrinsic[:, 0, 0] <= 0.0) or np.any(intrinsic[:, 1, 1] <= 0.0):
        raise ValueError("camera_intrinsic focal lengths must be positive")


def _project_batch(
    points_xyz: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of ``geometry.project_points`` for [N, K]."""

    ones = np.ones((*points_xyz.shape[:2], 1), dtype=np.float64)
    points_h = np.concatenate((points_xyz, ones), axis=-1)
    camera_h = np.einsum("nij,nkj->nki", extrinsic, points_h)
    camera_xyz = camera_h[..., :3]
    uvw = np.einsum("nij,nkj->nki", intrinsic, camera_xyz)
    eps = 1e-8
    pixels = uvw[..., :2] / np.maximum(uvw[..., 2:3], eps)
    valid = (
        (camera_xyz[..., 2] > eps)
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] < IMAGE_WIDTH)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] < IMAGE_HEIGHT)
    )
    return pixels, valid


def _validate_projection(
    *,
    label: str,
    recorded_pixels: np.ndarray,
    recorded_valid: np.ndarray,
    points_xyz: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> None:
    expected_pixels, expected_valid = _project_batch(
        points_xyz,
        intrinsic,
        extrinsic,
    )
    if not np.allclose(
        recorded_pixels,
        expected_pixels,
        rtol=0.0,
        atol=PROJECTION_ATOL_PX,
    ):
        max_error = float(np.max(np.abs(recorded_pixels - expected_pixels)))
        raise ValueError(
            f"{label} differs from the frozen geometric reprojection "
            f"(max_abs_px={max_error:.6g})"
        )
    if not np.array_equal(recorded_valid, expected_valid):
        mismatches = int(np.count_nonzero(recorded_valid != expected_valid))
        valid_label = "valid_true" if label == "q_true" else "valid_nominal"
        raise ValueError(
            f"{valid_label} differs from the frozen projection validity "
            f"({mismatches} mismatches)"
        )


def _validate_common_design(design: Mapping[str, Any]) -> None:
    if design.get("schema_version") != 1 or design.get("protocol") != PROTOCOL:
        raise ValueError("unexpected confirmatory intervention design")
    if tuple(_sequence(design.get("tasks"), "design.tasks")) != REQUIRED_TASKS:
        raise ValueError("confirmatory intervention task order changed")
    collection = _mapping(design.get("collection"), "design.collection")
    required = {
        "camera": "corner",
        "width": 224,
        "height": 224,
        "episodes_per_task": 30,
        "steps_per_episode": 100,
        "warmup_steps": 2,
        "perturbations_per_frame": 1,
        "action_source": "scripted",
        "keypoint_layout": "control",
        "vary_task_index": True,
        "task_index_offset": 0,
        "actual_camera_randomization": "episode",
        "actual_camera_clean_probability": 0.0,
        "independent_rng_streams": True,
        "nominal_translation_cm": 0.0,
        "nominal_yaw_deg": 0.0,
        "numpy_version": "1.26.4",
    }
    mismatched = {
        key: {"actual": collection.get(key), "expected": value}
        for key, value in required.items()
        if collection.get(key) != value
    }
    if mismatched:
        raise ValueError(f"confirmatory collection contract changed: {mismatched}")



def _frozen_camera_model(
    design: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    camera_model = _mapping(design.get("camera_model"), "design.camera_model")
    intrinsic = np.asarray(
        camera_model.get("camera_intrinsic"), dtype=np.float64
    )
    base_extrinsic = np.asarray(
        camera_model.get("base_extrinsic"), dtype=np.float64
    )
    _validate_intrinsic_batch(intrinsic[None, ...], count=1)
    _validate_extrinsic_batch(
        base_extrinsic[None, ...],
        count=1,
        name="design.camera_model.base_extrinsic",
    )
    return intrinsic, base_extrinsic

def _validate_tooling(
    design: Mapping[str, Any], *, root: Path
) -> list[dict[str, object]]:
    tooling = _mapping(design.get("tooling"), "design.tooling")
    records = []
    for label in ("collector", "camera_sampler"):
        declared = _mapping(tooling.get(label), f"design.tooling.{label}")
        path = _resolve(declared.get("path"), root=root)
        record = _record(path)
        records.append(record)
    return records


def _validate_cohort(
    path: Path,
    *,
    name: str,
    source: Mapping[str, Any],
    frozen_intrinsic: np.ndarray,
    frozen_base_extrinsic: np.ndarray,
) -> dict[str, object]:
    expected, expected_digest = _expected_episode_ledger(source)
    if expected_digest != source.get("transform_ledger_sha256"):
        raise ValueError(f"frozen transform ledger changed for {name}")
    required_fields = {
        "task_name",
        "task_index",
        "episode_id",
        "env_seed",
        "frame_id",
        "action_source",
        "observation_action_alignment",
        "keypoint_layout",
        "actual_camera_translation_xyz",
        "actual_camera_rotation_axis",
        "actual_camera_rotation_deg",
        "actual_camera_translation_cm_min",
        "actual_camera_translation_cm_max",
        "actual_camera_rotation_deg_min",
        "actual_camera_rotation_deg_max",
        "actual_camera_randomization",
        "actual_camera_clean_probability",
        "independent_rng_streams",
        "rng_seed",
        "task_generation_seed",
        "translation_cm",
        "yaw_deg",
        "camera_intrinsic",
        "keypoints_xyz",
        "q_true",
        "q_nominal",
        "valid_true",
        "valid_nominal",
        "base_extrinsic",
        "nominal_extrinsic",
        "true_extrinsic",
    }
    with np.load(path, allow_pickle=False) as raw:
        missing = sorted(required_fields.difference(raw.files))
        if missing:
            raise ValueError(f"{name} lacks camera audit fields: {missing}")
        scalar_expected = {
            "observation_action_alignment": "pre_action",
            "keypoint_layout": "control",
            "actual_camera_translation_cm_min": float(
                source["actual_translation_cm"]
            ),
            "actual_camera_translation_cm_max": float(
                source["actual_translation_cm"]
            ),
            "actual_camera_rotation_deg_min": float(
                source["actual_rotation_abs_deg"]
            ),
            "actual_camera_rotation_deg_max": float(
                source["actual_rotation_abs_deg"]
            ),
            "actual_camera_randomization": "episode",
            "actual_camera_clean_probability": 0.0,
            "independent_rng_streams": True,
            "rng_seed": int(source["rng_seed"]),
            "task_generation_seed": int(source["task_generation_seed"]),
        }
        for key, expected_value in scalar_expected.items():
            actual = _scalar(raw, key)
            if isinstance(expected_value, float):
                matched = math.isclose(
                    float(actual), expected_value, rel_tol=0.0, abs_tol=1e-6
                )
            else:
                matched = actual == expected_value
            if not matched:
                raise ValueError(
                    f"{name} embedded collection metadata differs at {key}"
                )
        task_name = np.asarray(raw["task_name"]).astype(str)
        task_index = np.asarray(raw["task_index"], dtype=np.int64)
        episode_id = np.asarray(raw["episode_id"], dtype=np.int64)
        env_seed = np.asarray(raw["env_seed"], dtype=np.int64)
        frame_id = np.asarray(raw["frame_id"], dtype=np.int64)
        action_source = np.asarray(raw["action_source"]).astype(str)
        actual_translation = np.asarray(
            raw["actual_camera_translation_xyz"], dtype=np.float64
        )
        actual_axis = np.asarray(
            raw["actual_camera_rotation_axis"], dtype=np.float64
        )
        actual_degrees = np.asarray(
            raw["actual_camera_rotation_deg"], dtype=np.float64
        )
        base_extrinsic = np.asarray(raw["base_extrinsic"], dtype=np.float64)
        nominal_extrinsic = np.asarray(raw["nominal_extrinsic"], dtype=np.float64)
        true_extrinsic = np.asarray(raw["true_extrinsic"], dtype=np.float64)
        camera_intrinsic = np.asarray(raw["camera_intrinsic"], dtype=np.float64)
        keypoints_xyz = np.asarray(raw["keypoints_xyz"], dtype=np.float64)
        q_true = np.asarray(raw["q_true"], dtype=np.float64)
        q_nominal = np.asarray(raw["q_nominal"], dtype=np.float64)
        raw_valid_true = np.asarray(raw["valid_true"])
        raw_valid_nominal = np.asarray(raw["valid_nominal"])
        count = len(task_name)
        if count != 12_000:
            raise ValueError(f"{name} must contain exactly 12000 recorded frames")
        for field, value in (
            ("task_name", task_name),
            ("task_index", task_index),
            ("episode_id", episode_id),
            ("env_seed", env_seed),
            ("frame_id", frame_id),
            ("action_source", action_source),
            ("actual_camera_rotation_deg", actual_degrees),
        ):
            _require_shape(value, (count,), name=field)
        _require_shape(
            actual_translation,
            (count, 3),
            name="actual_camera_translation_xyz",
        )
        _require_shape(
            actual_axis,
            (count, 3),
            name="actual_camera_rotation_axis",
        )
        _require_shape(
            np.asarray(raw["translation_cm"]),
            (count,),
            name="translation_cm",
        )
        _require_shape(
            np.asarray(raw["yaw_deg"]),
            (count,),
            name="yaw_deg",
        )
        _validate_extrinsic_batch(
            base_extrinsic,
            count=count,
            name="base_extrinsic",
        )
        _validate_extrinsic_batch(
            nominal_extrinsic,
            count=count,
            name="nominal_extrinsic",
        )
        _validate_extrinsic_batch(
            true_extrinsic,
            count=count,
            name="true_extrinsic",
        )
        _validate_intrinsic_batch(camera_intrinsic, count=count)
        if not np.allclose(
            camera_intrinsic,
            frozen_intrinsic,
            rtol=0.0,
            atol=INTRINSIC_ATOL,
        ):
            raise ValueError(
                f"{name} camera_intrinsic differs from preregistered camera model"
            )
        if not np.allclose(
            base_extrinsic,
            frozen_base_extrinsic,
            rtol=0.0,
            atol=SE3_ATOL,
        ):
            raise ValueError(
                f"{name} base_extrinsic differs from preregistered camera model"
            )
        _require_shape(
            keypoints_xyz,
            (count, CONTROL_KEYPOINTS, 3),
            name="keypoints_xyz",
        )
        _require_shape(
            q_true,
            (count, CONTROL_KEYPOINTS, 2),
            name="q_true",
        )
        _require_shape(
            q_nominal,
            (count, CONTROL_KEYPOINTS, 2),
            name="q_nominal",
        )
        _require_shape(
            raw_valid_true,
            (count, CONTROL_KEYPOINTS),
            name="valid_true",
        )
        _require_shape(
            raw_valid_nominal,
            (count, CONTROL_KEYPOINTS),
            name="valid_nominal",
        )
        if raw_valid_true.dtype != np.bool_ or raw_valid_nominal.dtype != np.bool_:
            raise ValueError("valid_true and valid_nominal must use boolean dtype")
        valid_true = raw_valid_true.astype(bool, copy=False)
        valid_nominal = raw_valid_nominal.astype(bool, copy=False)
        if set(action_source.tolist()) != {"scripted"}:
            raise ValueError(f"{name} must contain only scripted actions")
        if not np.allclose(raw["translation_cm"], 0.0, rtol=0.0, atol=1e-7):
            raise ValueError(f"{name} nominal translation perturbation is nonzero")
        if not np.allclose(raw["yaw_deg"], 0.0, rtol=0.0, atol=1e-7):
            raise ValueError(f"{name} nominal yaw perturbation is nonzero")
        if not np.allclose(
            nominal_extrinsic, base_extrinsic, rtol=0.0, atol=GEOMETRY_ATOL
        ):
            raise ValueError(f"{name} nominal calibration differs from base camera")
        if not all(
            np.isfinite(value).all()
            for value in (
                actual_translation,
                actual_axis,
                actual_degrees,
                keypoints_xyz,
                q_true,
                q_nominal,
            )
        ):
            raise ValueError(f"{name} camera audit contains non-finite values")
        _validate_projection(
            label="q_true",
            recorded_pixels=q_true,
            recorded_valid=valid_true,
            points_xyz=keypoints_xyz,
            intrinsic=camera_intrinsic,
            extrinsic=true_extrinsic,
        )
        _validate_projection(
            label="q_nominal",
            recorded_pixels=q_nominal,
            recorded_valid=valid_nominal,
            points_xyz=keypoints_xyz,
            intrinsic=camera_intrinsic,
            extrinsic=nominal_extrinsic,
        )

        realized = hashlib.sha256()
        realized.update(LEDGER_DOMAIN)
        for record in expected:
            task = str(record["task"])
            episode = int(record["episode"])
            indices = np.flatnonzero(
                (task_name == task) & (episode_id == episode)
            )
            order = np.argsort(frame_id[indices], kind="stable")
            indices = indices[order]
            if len(indices) != 100 or not np.array_equal(
                frame_id[indices], np.arange(100, dtype=np.int64)
            ):
                raise ValueError(f"{name} episode {task}:{episode} is incomplete")
            if not np.all(task_index[indices] == int(record["task_index"])):
                raise ValueError(f"{name} task-index schedule changed")
            if not np.all(env_seed[indices] == int(record["env_seed"])):
                raise ValueError(f"{name} environment-seed schedule changed")
            expected_translation = np.asarray(record["translation"])
            expected_axis = np.asarray(record["axis"])
            expected_degrees = float(record["degrees"])
            if not np.allclose(
                actual_translation[indices],
                expected_translation,
                rtol=0.0,
                atol=GEOMETRY_ATOL,
            ):
                raise ValueError(f"{name} translation ledger differs")
            if not np.allclose(
                actual_axis[indices], expected_axis, rtol=0.0, atol=GEOMETRY_ATOL
            ):
                raise ValueError(f"{name} rotation-axis ledger differs")
            if not np.allclose(
                actual_degrees[indices],
                expected_degrees,
                rtol=0.0,
                atol=1e-5,
            ):
                raise ValueError(f"{name} rotation-angle ledger differs")

            base = base_extrinsic[indices]
            true = true_extrinsic[indices]
            center_delta = _camera_centers(true) - _camera_centers(base)
            if not np.allclose(
                center_delta,
                expected_translation,
                rtol=0.0,
                atol=GEOMETRY_ATOL,
            ):
                raise ValueError(f"{name} realized camera translation is invalid")
            delta_rotation = _axis_angle_matrix(
                expected_axis, expected_degrees
            )
            expected_true_rotation = np.matmul(
                base[:, :3, :3], delta_rotation.T
            )
            if not np.allclose(
                true[:, :3, :3],
                expected_true_rotation,
                rtol=0.0,
                atol=GEOMETRY_ATOL,
            ):
                raise ValueError(f"{name} realized camera rotation is invalid")
            _update_ledger(
                realized,
                task=task,
                episode=episode,
                env_seed=int(env_seed[indices[0]]),
                task_index=int(task_index[indices[0]]),
                translation=actual_translation[indices[0]],
                axis=actual_axis[indices[0]],
                degrees=float(actual_degrees[indices[0]]),
            )
        realized_digest = realized.hexdigest()
        if realized_digest != expected_digest:
            raise ValueError(f"{name} realized transform ledger hash differs")
    return {
        "passed": True,
        "frames": 12_000,
        "episodes": 120,
        "actual_translation_cm": float(source["actual_translation_cm"]),
        "actual_rotation_abs_deg": float(source["actual_rotation_abs_deg"]),
        "transform_ledger_sha256": expected_digest,
    }


def validate_confirmatory_camera_interventions(
    design_path: str | Path,
    source_report: Sequence[Mapping[str, Any]],
    *,
    repo_root: str | Path,
) -> tuple[dict[str, object], list[Path]]:
    """Bind the preregistered plan and verify realized camera SE(3) from NPZs."""

    root = Path(repo_root).expanduser().resolve()
    path = Path(design_path).expanduser().resolve()
    design = _mapping(
        json.loads(path.read_text(encoding="utf-8")), "confirmatory design"
    )
    _validate_common_design(design)
    tooling_records = _validate_tooling(design, root=root)
    frozen_intrinsic, frozen_base_extrinsic = _frozen_camera_model(design)
    declared_sources = _mapping(design.get("sources"), "design.sources")
    if tuple(declared_sources) != REQUIRED_COHORTS:
        raise ValueError("confirmatory intervention cohorts changed")
    observed = {str(record.get("name")): record for record in source_report}
    if tuple(observed) != REQUIRED_COHORTS:
        raise ValueError("confirmatory source order differs from intervention design")
    cohorts: dict[str, object] = {}
    bound_paths = [path, *(Path(record["path"]) for record in tooling_records)]
    for name in REQUIRED_COHORTS:
        source = _mapping(declared_sources.get(name), f"design.sources.{name}")
        report = _mapping(observed.get(name), f"source_report.{name}")
        data_path = _resolve(report.get("data"), root=root)
        if data_path != _resolve(source.get("path"), root=root):
            raise ValueError(f"{name} data path differs from preregistered design")
        if report.get("data_sha256") != sha256_file(data_path):
            raise ValueError(f"{name} source report hash changed")
        cohorts[name] = _validate_cohort(
            data_path,
            name=name,
            source=source,
            frozen_intrinsic=frozen_intrinsic,
            frozen_base_extrinsic=frozen_base_extrinsic,
        )
        bound_paths.append(data_path)
    return {
        "passed": True,
        "criterion": "realized_SE3_matches_frozen_episode_transform_ledger",
        "design": _record(path),
        "tooling": tooling_records,
        "cohorts": cohorts,
    }, bound_paths


__all__ = ["validate_confirmatory_camera_interventions"]
