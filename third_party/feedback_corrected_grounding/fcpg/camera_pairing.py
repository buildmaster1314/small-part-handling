"""Fail-closed validation for camera-intervention dataset pairs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.lewm_provenance import sha256_file


PHYSICAL_IDENTITY_KEYS = (
    "task_name",
    "task_index",
    "episode_id",
    "env_seed",
    "frame_id",
    "obs",
    "action",
    "reward",
    "success",
    "keypoints_xyz",
    "control_keypoints_xyz",
    "q_nominal",
    "control_q_nominal",
    "valid_nominal",
    "control_valid_nominal",
    "camera_intrinsic",
    "base_extrinsic",
    "nominal_extrinsic",
    "translation_cm",
    "translation_direction",
    "yaw_deg",
    "action_source",
    "observation_action_alignment",
    "task_generation_seed",
    "rng_seed",
    "independent_rng_streams",
)

CAMERA_OBSERVATION_KEYS = (
    "rgb",
    "q_true",
    "control_q_true",
    "true_extrinsic",
    "actual_camera_translation_xyz",
    "actual_camera_rotation_axis",
    "actual_camera_rotation_deg",
)


def _update_array_hash(digest: Any, key: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(key.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def physical_identity_sha256(data: Any) -> str:
    digest = hashlib.sha256()
    for key in PHYSICAL_IDENTITY_KEYS:
        if key not in data:
            raise ValueError(f"Paired dataset is missing required key: {key}")
        _update_array_hash(digest, key, np.asarray(data[key]))
    return digest.hexdigest()


def validate_paired_camera_datasets(
    reference_path: str | Path,
    intervention_path: str | Path,
    *,
    require_camera_difference: bool = True,
) -> dict[str, Any]:
    """Require two datasets to share an exact physical trajectory."""

    reference_path = Path(reference_path).resolve()
    intervention_path = Path(intervention_path).resolve()
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        intervention_path, allow_pickle=False
    ) as intervention:
        mismatches = []
        for key in PHYSICAL_IDENTITY_KEYS:
            if key not in reference or key not in intervention:
                raise ValueError(f"Paired dataset is missing required key: {key}")
            if not np.array_equal(reference[key], intervention[key]):
                mismatches.append(key)
        if mismatches:
            raise ValueError(
                "Camera datasets do not share an exact physical trajectory; "
                f"mismatched keys: {', '.join(mismatches)}"
            )

        if not bool(np.asarray(reference["independent_rng_streams"]).item()):
            raise ValueError("Paired data must use --independent-rng-streams")

        camera_differences = {
            key: not np.array_equal(reference[key], intervention[key])
            for key in CAMERA_OBSERVATION_KEYS
            if key in reference and key in intervention
        }
        if require_camera_difference and not any(camera_differences.values()):
            raise ValueError("Camera intervention produced no observable difference")

        identity_sha256 = physical_identity_sha256(reference)
        if identity_sha256 != physical_identity_sha256(intervention):
            raise AssertionError("Physical identity hash disagrees after exact checks")
        sample_count = int(reference["obs"].shape[0])

    return {
        "schema_version": 1,
        "validated": True,
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "intervention_path": str(intervention_path),
        "intervention_sha256": sha256_file(intervention_path),
        "sample_count": sample_count,
        "physical_identity_sha256": identity_sha256,
        "camera_differences": camera_differences,
    }


__all__ = [
    "CAMERA_OBSERVATION_KEYS",
    "PHYSICAL_IDENTITY_KEYS",
    "physical_identity_sha256",
    "validate_paired_camera_datasets",
]
