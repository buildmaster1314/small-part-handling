"""Fail-closed composition of paired camera cohorts into an episode step.

The compositor never estimates or interpolates camera observations.  It takes
two datasets that already share an exact physical trajectory and selects the
reference observation before a frame boundary and the intervention observation
at and after that boundary.  FCPG corrections are selected by the same mask.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fcpg.camera_pairing import (
    CAMERA_OBSERVATION_KEYS,
    PHYSICAL_IDENTITY_KEYS,
    validate_paired_camera_datasets,
)
from fcpg.lewm_provenance import sha256_file
from fcpg.lewm_sequence_data import frame_identity_sha256


PROTOCOL = "fcpg_paired_camera_step_composition_v1"

# These arrays depend on the rendered camera but were intentionally not part of
# the original minimal CAMERA_OBSERVATION_KEYS pairing contract.
DERIVED_CAMERA_ARRAY_KEYS = (
    "valid_true",
    "control_valid_true",
    "observer_visibility",
    "pixel_error",
    "q_true_inside_patch",
)
SWITCHED_FRAME_KEYS = tuple(
    dict.fromkeys((*CAMERA_OBSERVATION_KEYS, *DERIVED_CAMERA_ARRAY_KEYS))
)

SCALAR_PHYSICAL_KEYS = (
    "observation_action_alignment",
    "task_generation_seed",
    "rng_seed",
    "independent_rng_streams",
)

# Historical collection scalars describe a sampler.  For a deterministic
# composite they are replaced with the ranges actually present in the output.
RECOMPUTED_SCALAR_KEYS = (
    "actual_camera_translation_cm_min",
    "actual_camera_translation_cm_max",
    "actual_camera_rotation_deg_min",
    "actual_camera_rotation_deg_max",
    "actual_camera_randomization",
    "actual_camera_clean_probability",
)

REQUIRED_CACHE_METADATA_KEYS = (
    "schema_version",
    "source_data",
    "source_data_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "sample_count",
    "num_keypoints",
    "frame_identity_sha256",
    "q_corrected_sha256",
    "q_true_usage",
    "has_reliability",
    "corrector_arch",
    "checkpoint_schema_version",
    "obs_mode",
    "lewm_upstream_commit",
)

CACHE_SEMANTIC_KEYS = (
    "corrector_arch",
    "checkpoint_schema_version",
    "obs_mode",
    "lewm_upstream_commit",
    "num_keypoints",
    "has_reliability",
)


@dataclass(frozen=True)
class CameraStepCompositionResult:
    data_path: Path
    data_metadata_path: Path
    correction_cache: Path
    data_sha256: str
    q_corrected_sha256: str
    sample_count: int
    reference_frames: int
    intervention_frames: int


@dataclass(frozen=True)
class _ParentCacheSnapshot:
    metadata: dict[str, Any]
    corrected: np.ndarray
    checkpoint_path: Path
    metadata_sha256: str
    q_corrected_sha256: str


def _load_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _resolve_metadata_path(value: object, *, metadata_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid path in {metadata_path}: {value!r}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else metadata_path.parent / path).resolve()


def _require_output_absent(data_path: Path, metadata_path: Path, cache_path: Path) -> None:
    for path in (data_path, metadata_path, cache_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite compositor output: {path}")


def _validate_parent_cache(
    cache_dir: Path,
    *,
    data_path: Path,
    data_sha256: str,
    episode_id: np.ndarray,
    frame_id: np.ndarray,
    sample_count: int,
    num_keypoints: int,
) -> _ParentCacheSnapshot:
    cache_dir = cache_dir.expanduser().resolve()
    metadata_path = cache_dir / "metadata.json"
    q_path = cache_dir / "q_corrected.npy"
    if not metadata_path.is_file() or not q_path.is_file():
        raise FileNotFoundError(
            f"correction cache requires metadata.json and q_corrected.npy: {cache_dir}"
        )
    metadata, metadata_sha256 = _load_json_snapshot(metadata_path)
    missing = [key for key in REQUIRED_CACHE_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"correction cache metadata is missing fields: {missing}")
    if int(metadata["schema_version"]) < 3:
        raise ValueError("paired-step composition requires schema-3 correction caches")
    if _resolve_metadata_path(metadata["source_data"], metadata_path=metadata_path) != data_path:
        raise ValueError("correction cache source_data does not match its parent dataset")
    if metadata["source_data_sha256"] != data_sha256:
        raise ValueError("correction cache source_data_sha256 differs from parent dataset")
    if int(metadata["sample_count"]) != sample_count:
        raise ValueError("correction cache sample_count differs from parent dataset")
    if int(metadata["num_keypoints"]) != num_keypoints:
        raise ValueError("correction cache num_keypoints differs from parent dataset")
    expected_identity = frame_identity_sha256(episode_id, frame_id)
    if metadata["frame_identity_sha256"] != expected_identity:
        raise ValueError("correction cache frame identity differs from parent dataset")
    if metadata["q_true_usage"] != "offline_metrics_only":
        raise ValueError("correction cache has an invalid q_true usage contract")
    if metadata["has_reliability"] is not False:
        raise ValueError("paired-step composition currently requires residual-only caches")
    for key in ("corrector_arch", "obs_mode", "lewm_upstream_commit"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f"correction cache {key} must be a non-empty string")
    checkpoint_schema = metadata["checkpoint_schema_version"]
    if (
        not isinstance(checkpoint_schema, int)
        or isinstance(checkpoint_schema, bool)
        or checkpoint_schema < 1
    ):
        raise ValueError(
            "correction cache checkpoint_schema_version must be a positive integer"
        )

    checkpoint_path = _resolve_metadata_path(
        metadata["checkpoint"], metadata_path=metadata_path
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing correction checkpoint: {checkpoint_path}")
    if metadata["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError("correction checkpoint SHA-256 differs from cache metadata")
    q_payload = q_path.read_bytes()
    q_sha256 = hashlib.sha256(q_payload).hexdigest()
    if metadata["q_corrected_sha256"] != q_sha256:
        raise ValueError("q_corrected SHA-256 differs from cache metadata")

    corrected = np.load(io.BytesIO(q_payload), allow_pickle=False)
    expected_shape = (sample_count, num_keypoints, 2)
    if corrected.shape != expected_shape:
        raise ValueError(
            f"q_corrected must have shape {expected_shape}, got {corrected.shape}"
        )
    if not np.issubdtype(corrected.dtype, np.floating):
        raise ValueError("q_corrected must use floating dtype")
    if not bool(np.isfinite(corrected).all()):
        raise ValueError("q_corrected contains non-finite values")
    return _ParentCacheSnapshot(
        metadata=metadata,
        corrected=np.array(corrected, copy=True),
        checkpoint_path=checkpoint_path,
        metadata_sha256=metadata_sha256,
        q_corrected_sha256=q_sha256,
    )


def _stable_switch_frame(
    task: str,
    episode: str,
    *,
    switch_frame_min: int,
    switch_frame_max: int,
    switch_seed: int,
) -> int:
    payload = f"{PROTOCOL}\0{switch_seed}\0{task}\0{episode}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return switch_frame_min + value % (switch_frame_max - switch_frame_min + 1)


def _selection_schedule(
    task_name: np.ndarray,
    episode_id: np.ndarray,
    frame_id: np.ndarray,
    *,
    switch_frame: int | None,
    switch_frame_min: int | None,
    switch_frame_max: int | None,
    switch_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    frames = np.asarray(frame_id, dtype=np.int64).reshape(-1)
    tasks = np.asarray(task_name).astype(str).reshape(-1)
    episodes = np.asarray(episode_id).astype(str).reshape(-1)
    if not (frames.shape == tasks.shape == episodes.shape):
        raise ValueError("task_name, episode_id, and frame_id must have equal shapes")
    if len(frames) == 0:
        raise ValueError("paired camera datasets contain no frames")
    has_range = switch_frame_min is not None or switch_frame_max is not None
    if has_range and (switch_frame_min is None or switch_frame_max is None):
        raise ValueError("switch_frame_min and switch_frame_max must be provided together")
    if has_range and switch_frame is not None:
        raise ValueError("fixed and ranged switch schedules are mutually exclusive")
    if not has_range and switch_frame is None:
        switch_frame = 50
    if has_range:
        assert switch_frame_min is not None and switch_frame_max is not None
        if switch_frame_min < 0 or switch_frame_max < switch_frame_min:
            raise ValueError("switch-frame range must be non-negative and ordered")
    elif int(switch_frame) < 0:
        raise ValueError("switch_frame must be non-negative")

    int32_max = int(np.iinfo(np.int32).max)
    configured_max = (
        int(switch_frame_max)
        if has_range
        else int(switch_frame)
    )
    if configured_max > int32_max:
        raise ValueError(f"switch frame must not exceed int32 maximum {int32_max}")

    mask = np.zeros(len(frames), dtype=bool)
    event = np.zeros(len(frames), dtype=bool)
    frame_schedule = np.empty(len(frames), dtype=np.int32)
    ledger: list[dict[str, Any]] = []
    for task, episode in sorted(set(zip(tasks.tolist(), episodes.tolist()))):
        rows = (tasks == task) & (episodes == episode)
        episode_switch = (
            _stable_switch_frame(
                task,
                episode,
                switch_frame_min=int(switch_frame_min),
                switch_frame_max=int(switch_frame_max),
                switch_seed=int(switch_seed),
            )
            if has_range
            else int(switch_frame)
        )
        episode_frames = frames[rows]
        unique_frames = np.unique(episode_frames)
        if len(unique_frames) != len(episode_frames):
            raise ValueError(
                f"episode {task}:{episode} contains duplicate frame_id values"
            )
        if len(unique_frames) > 1 and not bool(
            np.all(np.diff(unique_frames) == 1)
        ):
            raise ValueError(
                f"episode {task}:{episode} is not contiguous in frame_id"
            )

        episode_mask = episode_frames >= episode_switch
        if not bool((~episode_mask).any() and episode_mask.any()):
            raise ValueError(
                "switch schedule must leave reference and intervention frames in every "
                f"episode; failed for {task}:{episode}"
            )
        episode_event = episode_frames == episode_switch
        if int(episode_event.sum()) != 1:
            raise ValueError(
                "each episode must contain exactly one frame at its switch boundary; "
                f"failed for {task}:{episode} at frame {episode_switch}"
            )
        mask[rows] = episode_mask
        event[rows] = episode_event
        frame_schedule[rows] = episode_switch
        ledger.append(
            {
                "task_name": task,
                "episode_id": episode,
                "switch_frame": episode_switch,
                "reference_frames": int((~episode_mask).sum()),
                "intervention_frames": int(episode_mask.sum()),
            }
        )
    return mask, event, frame_schedule, ledger


def _select_rows(
    reference: np.ndarray,
    intervention: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    if reference.shape != intervention.shape or reference.dtype != intervention.dtype:
        raise ValueError(f"paired field shape/dtype differs: {name}")
    if reference.ndim == 0 or reference.shape[0] != mask.shape[0]:
        raise ValueError(f"switched field must be frame-aligned: {name}")
    expanded = mask.reshape((len(mask),) + (1,) * (reference.ndim - 1))
    return np.where(expanded, intervention, reference)


def _realized_camera_scalars(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    translation = np.asarray(arrays["actual_camera_translation_xyz"], dtype=np.float64)
    rotation = np.abs(
        np.asarray(arrays["actual_camera_rotation_deg"], dtype=np.float64).reshape(-1)
    )
    if translation.ndim != 2 or translation.shape[1] != 3:
        raise ValueError("actual_camera_translation_xyz must have shape [N, 3]")
    if translation.shape[0] != rotation.shape[0]:
        raise ValueError("camera translation and rotation arrays differ in length")
    translation_cm = np.linalg.norm(translation, axis=1) * 100.0
    clean = (translation_cm <= 1e-8) & (rotation <= 1e-8)
    return {
        "actual_camera_translation_cm_min": np.asarray(
            float(translation_cm.min()), dtype=np.float32
        ),
        "actual_camera_translation_cm_max": np.asarray(
            float(translation_cm.max()), dtype=np.float32
        ),
        "actual_camera_rotation_deg_min": np.asarray(
            float(rotation.min()), dtype=np.float32
        ),
        "actual_camera_rotation_deg_max": np.asarray(
            float(rotation.max()), dtype=np.float32
        ),
        "actual_camera_randomization": np.asarray("paired_step"),
        "actual_camera_clean_probability": np.asarray(
            float(clean.mean()), dtype=np.float32
        ),
        "camera_step_clean_frame_fraction": np.asarray(
            float(clean.mean()), dtype=np.float32
        ),
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _require_unchanged(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} disappeared during composition: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} changed during composition: expected {expected_sha256}, "
            f"found {actual}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_file_no_replace(staged: Path, destination: Path) -> None:
    try:
        os.link(staged, destination)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite compositor output: {destination}"
        ) from None


def compose_paired_camera_step(
    *,
    reference_data: str | Path,
    intervention_data: str | Path,
    reference_correction_cache: str | Path,
    intervention_correction_cache: str | Path,
    switch_frame: int | None = None,
    switch_frame_min: int | None = None,
    switch_frame_max: int | None = None,
    switch_seed: int = 0,
    output_data: str | Path,
    output_correction_cache: str | Path,
    output_metadata: str | Path | None = None,
) -> CameraStepCompositionResult:
    """Compose a deterministic camera step and its matched correction cache."""

    reference_path = Path(reference_data).expanduser().resolve()
    intervention_path = Path(intervention_data).expanduser().resolve()
    output_path = Path(output_data).expanduser().resolve()
    cache_path = Path(output_correction_cache).expanduser().resolve()
    metadata_path = (
        Path(output_metadata).expanduser().resolve()
        if output_metadata is not None
        else output_path.with_suffix(output_path.suffix + ".metadata.json")
    )
    if output_path.suffix != ".npz":
        raise ValueError("output_data must use the .npz suffix")
    input_paths = {
        reference_path,
        intervention_path,
        Path(reference_correction_cache).expanduser().resolve(),
        Path(intervention_correction_cache).expanduser().resolve(),
    }
    if output_path in input_paths or cache_path in input_paths:
        raise ValueError("compositor outputs must differ from all input paths")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _require_output_absent(output_path, metadata_path, cache_path)

    pair_report = validate_paired_camera_datasets(
        reference_path, intervention_path, require_camera_difference=True
    )
    reference_sha = str(pair_report["reference_sha256"])
    intervention_sha = str(pair_report["intervention_sha256"])

    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        intervention_path, allow_pickle=False
    ) as intervention:
        if set(reference.files) != set(intervention.files):
            raise ValueError("paired dataset schemas differ")
        sample_count = int(np.asarray(reference["obs"]).shape[0])
        q_nominal = np.asarray(reference["q_nominal"])
        if q_nominal.ndim != 3 or q_nominal.shape[-1] != 2:
            raise ValueError("q_nominal must have shape [N, K, 2]")
        if q_nominal.shape[0] != sample_count:
            raise ValueError(
                "q_nominal and obs must contain the same number of frames"
            )
        num_keypoints = int(q_nominal.shape[1])
        task_name = np.asarray(reference["task_name"])
        episode_id = np.asarray(reference["episode_id"])
        frame_id = np.asarray(reference["frame_id"])
        mask, step_event, step_frame, schedule_ledger = _selection_schedule(
            task_name,
            episode_id,
            frame_id,
            switch_frame=switch_frame,
            switch_frame_min=switch_frame_min,
            switch_frame_max=switch_frame_max,
            switch_seed=switch_seed,
        )
        if mask.shape[0] != sample_count:
            raise ValueError("frame identity and obs must contain the same number of frames")

        missing_switched = [key for key in SWITCHED_FRAME_KEYS if key not in reference]
        if missing_switched:
            raise ValueError(
                f"paired datasets are missing switched camera fields: {missing_switched}"
            )
        for key in SCALAR_PHYSICAL_KEYS:
            if np.asarray(reference[key]).shape != ():
                raise ValueError(f"physical metadata field must be scalar: {key}")
        frame_aligned_keys = tuple(
            key for key in PHYSICAL_IDENTITY_KEYS if key not in SCALAR_PHYSICAL_KEYS
        ) + SWITCHED_FRAME_KEYS
        malformed_frame_fields = {
            key: tuple(np.asarray(reference[key]).shape)
            for key in frame_aligned_keys
            if np.asarray(reference[key]).ndim == 0
            or np.asarray(reference[key]).shape[0] != sample_count
        }
        if malformed_frame_fields:
            raise ValueError(
                "paired datasets contain non-frame-aligned fields: "
                f"{malformed_frame_fields}"
            )

        arrays: dict[str, np.ndarray] = {}
        for key in reference.files:
            if key in RECOMPUTED_SCALAR_KEYS:
                continue
            left = np.asarray(reference[key])
            right = np.asarray(intervention[key])
            if key in SWITCHED_FRAME_KEYS:
                arrays[key] = _select_rows(left, right, mask, name=key)
            else:
                if not np.array_equal(left, right):
                    raise ValueError(
                        "non-whitelisted dataset field differs across paired inputs: "
                        f"{key}"
                    )
                arrays[key] = np.array(left, copy=True)

    schedule_mode = (
        "range"
        if switch_frame_min is not None or switch_frame_max is not None
        else "fixed"
    )
    schedule_metadata = {
        "mode": schedule_mode,
        "fixed_frame": int(step_frame[0]) if schedule_mode == "fixed" else None,
        "frame_min": (
            int(switch_frame_min)
            if schedule_mode == "range"
            else int(step_frame[0])
        ),
        "frame_max": (
            int(switch_frame_max)
            if schedule_mode == "range"
            else int(step_frame[0])
        ),
        "seed": int(switch_seed),
        "per_episode": schedule_ledger,
    }
    arrays.update(_realized_camera_scalars(arrays))
    arrays.update(
        {
            "camera_step_schema_version": np.asarray(1, dtype=np.int32),
            "camera_step_schedule_mode": np.asarray(schedule_mode),
            "camera_step_switch_frame": np.asarray(
                int(step_frame[0]) if schedule_mode == "fixed" else -1,
                dtype=np.int32,
            ),
            "camera_step_frame": step_frame.astype(np.int32, copy=False),
            "camera_step_event": step_event.astype(bool, copy=False),
            "camera_step_reference_frames": np.asarray((~mask).sum(), dtype=np.int64),
            "camera_step_intervention_frames": np.asarray(mask.sum(), dtype=np.int64),
            "camera_step_reference_sha256": np.asarray(reference_sha),
            "camera_step_intervention_sha256": np.asarray(intervention_sha),
        }
    )

    reference_cache_path = Path(reference_correction_cache).expanduser().resolve()
    intervention_cache_path = Path(intervention_correction_cache).expanduser().resolve()
    reference_cache = _validate_parent_cache(
        reference_cache_path,
        data_path=reference_path,
        data_sha256=reference_sha,
        episode_id=episode_id,
        frame_id=frame_id,
        sample_count=sample_count,
        num_keypoints=num_keypoints,
    )
    intervention_cache = _validate_parent_cache(
            intervention_cache_path,
            data_path=intervention_path,
            data_sha256=intervention_sha,
            episode_id=episode_id,
            frame_id=frame_id,
            sample_count=sample_count,
            num_keypoints=num_keypoints,
    )
    reference_cache_metadata = reference_cache.metadata
    intervention_cache_metadata = intervention_cache.metadata
    reference_q = reference_cache.corrected
    intervention_q = intervention_cache.corrected
    reference_checkpoint = reference_cache.checkpoint_path
    intervention_checkpoint = intervention_cache.checkpoint_path
    if reference_checkpoint != intervention_checkpoint:
        raise ValueError("parent correction caches use different checkpoint paths")
    if (
        reference_cache_metadata["checkpoint_sha256"]
        != intervention_cache_metadata["checkpoint_sha256"]
    ):
        raise ValueError("parent correction caches use different checkpoint SHA-256")
    semantic_mismatch = [
        key
        for key in CACHE_SEMANTIC_KEYS
        if reference_cache_metadata.get(key) != intervention_cache_metadata.get(key)
    ]
    if semantic_mismatch:
        raise ValueError(
            f"parent correction cache semantics differ: {semantic_mismatch}"
        )
    q_corrected = _select_rows(
        np.asarray(reference_q), np.asarray(intervention_q), mask, name="q_corrected"
    ).astype(np.float32, copy=False)

    stage_data_fd, stage_data_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(stage_data_fd)
    stage_data = Path(stage_data_name)
    stage_data.unlink()
    stage_metadata_fd, stage_metadata_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.", suffix=".tmp", dir=metadata_path.parent
    )
    os.close(stage_metadata_fd)
    stage_metadata = Path(stage_metadata_name)
    stage_metadata.unlink()
    stage_cache = Path(
        tempfile.mkdtemp(prefix=f".{cache_path.name}.", dir=cache_path.parent)
    )
    data_published = False
    metadata_published = False
    cache_created = False
    try:
        _write_npz(stage_data, arrays)
        output_sha = sha256_file(stage_data)
        staged_q = stage_cache / "q_corrected.npy"
        with staged_q.open("xb") as stream:
            np.save(stream, q_corrected, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        q_sha = sha256_file(staged_q)
        output_identity = frame_identity_sha256(episode_id, frame_id)

        parent_cache_records = [
            {
                "path": str(reference_cache_path),
                "metadata_sha256": reference_cache.metadata_sha256,
                "q_corrected_sha256": reference_cache.q_corrected_sha256,
                "source_data_sha256": reference_sha,
            },
            {
                "path": str(intervention_cache_path),
                "metadata_sha256": intervention_cache.metadata_sha256,
                "q_corrected_sha256": intervention_cache.q_corrected_sha256,
                "source_data_sha256": intervention_sha,
            },
        ]
        cache_metadata = {
            "version": 3,
            "schema_version": 3,
            "cache_type": "fcpg_paired_camera_step_v1",
            "has_reliability": False,
            "cache_artifacts_complete": True,
            "corrector_arch": reference_cache_metadata["corrector_arch"],
            "checkpoint_schema_version": reference_cache_metadata[
                "checkpoint_schema_version"
            ],
            "obs_mode": reference_cache_metadata["obs_mode"],
            "lewm_upstream_commit": reference_cache_metadata[
                "lewm_upstream_commit"
            ],
            "source_data": str(output_path),
            "source_data_sha256": output_sha,
            "checkpoint": str(reference_checkpoint),
            "checkpoint_sha256": reference_cache_metadata["checkpoint_sha256"],
            "sample_count": sample_count,
            "num_keypoints": num_keypoints,
            "frame_identity_sha256": output_identity,
            "q_corrected_sha256": q_sha,
            "q_true_usage": "offline_metrics_only",
            "composition": {
                "protocol": PROTOCOL,
                "schedule": schedule_metadata,
                "selection": "reference_if_frame_id_lt_switch_else_intervention",
                "parent_caches": parent_cache_records,
            },
        }
        _write_json(stage_cache / "metadata.json", cache_metadata)

        composition_metadata = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "selection": {
                "field": "frame_id",
                "operator": ">=",
                "schedule": schedule_metadata,
                "reference_frames": int((~mask).sum()),
                "intervention_frames": int(mask.sum()),
                "every_episode_contains_both_branches": True,
            },
            "parents": {
                "reference": {
                    "path": str(reference_path),
                    "sha256": reference_sha,
                    "correction_cache": parent_cache_records[0],
                },
                "intervention": {
                    "path": str(intervention_path),
                    "sha256": intervention_sha,
                    "correction_cache": parent_cache_records[1],
                },
                "physical_identity_sha256": pair_report[
                    "physical_identity_sha256"
                ],
            },
            "field_contract": {
                "physical_identity_keys": list(PHYSICAL_IDENTITY_KEYS),
                "switched_frame_keys": list(SWITCHED_FRAME_KEYS),
                "recomputed_scalar_keys": list(RECOMPUTED_SCALAR_KEYS),
                "unknown_differing_fields_rejected": True,
            },
            "output": {
                "data": str(output_path),
                "data_sha256": output_sha,
                "metadata": str(metadata_path),
                "correction_cache": str(cache_path),
                "q_corrected_sha256": q_sha,
                "frame_identity_sha256": output_identity,
                "sample_count": sample_count,
                "num_keypoints": num_keypoints,
            },
            "scalar_metadata_semantics": {
                "camera_ranges": "realized_over_composite_frames",
                "actual_camera_clean_probability": (
                    "realized_clean_frame_fraction_for_deterministic_composite"
                ),
            },
            "publication": {
                "complete": True,
                "completion_marker": str(metadata_path),
                "completion_marker_published_last": True,
            },
        }
        _write_json(stage_metadata, composition_metadata)

        parent_snapshots = (
            (reference_path, reference_sha, "reference dataset"),
            (intervention_path, intervention_sha, "intervention dataset"),
            (
                reference_cache_path / "metadata.json",
                reference_cache.metadata_sha256,
                "reference correction metadata",
            ),
            (
                intervention_cache_path / "metadata.json",
                intervention_cache.metadata_sha256,
                "intervention correction metadata",
            ),
            (
                reference_cache_path / "q_corrected.npy",
                reference_cache.q_corrected_sha256,
                "reference correction array",
            ),
            (
                intervention_cache_path / "q_corrected.npy",
                intervention_cache.q_corrected_sha256,
                "intervention correction array",
            ),
            (
                reference_checkpoint,
                reference_cache_metadata["checkpoint_sha256"],
                "correction checkpoint",
            ),
        )
        for parent_path, expected_sha256, label in parent_snapshots:
            _require_unchanged(parent_path, expected_sha256, label=label)

        _require_output_absent(output_path, metadata_path, cache_path)
        _publish_file_no_replace(stage_data, output_path)
        data_published = True
        cache_path.mkdir(exist_ok=False)
        cache_created = True
        _publish_file_no_replace(staged_q, cache_path / "q_corrected.npy")
        _publish_file_no_replace(
            stage_cache / "metadata.json", cache_path / "metadata.json"
        )
        _fsync_directory(cache_path)
        _publish_file_no_replace(stage_metadata, metadata_path)
        metadata_published = True
        for parent in {
            output_path.parent,
            metadata_path.parent,
            cache_path.parent,
        }:
            _fsync_directory(parent)
    except Exception:
        if cache_created:
            shutil.rmtree(cache_path, ignore_errors=True)
        if metadata_published:
            metadata_path.unlink(missing_ok=True)
        if data_published:
            output_path.unlink(missing_ok=True)
        raise
    finally:
        stage_data.unlink(missing_ok=True)
        stage_metadata.unlink(missing_ok=True)
        shutil.rmtree(stage_cache, ignore_errors=True)

    return CameraStepCompositionResult(
        data_path=output_path,
        data_metadata_path=metadata_path,
        correction_cache=cache_path,
        data_sha256=output_sha,
        q_corrected_sha256=q_sha,
        sample_count=sample_count,
        reference_frames=int((~mask).sum()),
        intervention_frames=int(mask.sum()),
    )


__all__ = [
    "CameraStepCompositionResult",
    "DERIVED_CAMERA_ARRAY_KEYS",
    "PROTOCOL",
    "RECOMPUTED_SCALAR_KEYS",
    "SWITCHED_FRAME_KEYS",
    "compose_paired_camera_step",
]
