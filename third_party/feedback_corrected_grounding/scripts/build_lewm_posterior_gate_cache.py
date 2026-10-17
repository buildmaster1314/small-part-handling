#!/usr/bin/env python3
"""Build a frozen, leakage-isolated Anchor-Persistent posterior-gate cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_anchor_persistent import (  # noqa: E402
    ANCHOR_PERSISTENT_CHECKPOINT_TYPE,
    AnchorPersistentLeWM,
)
from fcpg.lewm_counterfactual_v2_storage import (  # noqa: E402
    _publish_directory_no_replace,
)
from fcpg.lewm_posterior_features import (  # noqa: E402
    POSTERIOR_FEATURE_ALLOWED_FIELDS,
    POSTERIOR_FEATURE_FORBIDDEN_FIELDS,
)
from fcpg.lewm_posterior_gate_data import (  # noqa: E402
    DEPLOYMENT_TENSOR_KEYS,
    EVALUATION_MASK_KEYS,
    FRESH_EVENT_NAMES,
    FRESH_FRAME_SKIP,
    FRESH_HISTORY_SIZE,
    LABEL_TENSOR_KEYS,
    build_fresh_posterior_gate_examples,
    physical_episode_sha256,
)
from fcpg.lewm_provenance import sha256_file  # noqa: E402
from fcpg.lewm_runtime import (  # noqa: E402
    choose_device,
    load_lewm_grounded_checkpoint,
)
from fcpg.lewm_sequence_data import frame_identity_sha256  # noqa: E402
from scripts.cache_vit_backbone_features import (  # noqa: E402
    IMAGE_VIEW_ROLES,
    _selected_image_sha256,
)
from scripts.evaluate_lewm_posterior_headroom import (  # noqa: E402
    _action_context as _reference_action_context,
    _encode_frames as _reference_encode_frames,
    _load_correction_cache as _reference_load_correction_cache,
    _predict as _reference_predict,
)


CACHE_TYPE = "fcpg_lewm_posterior_gate_cache_v1"
SCHEMA_VERSION = 1
EXPECTED_CHECKPOINT_STEP = 2000
EXPECTED_ACTION_DIM = 20
EVALUATION_VIEWS = ("perturbed_observation", "canonical_unoccluded_only")
DEPLOYMENT_FILENAME = "deployment_inputs.npz"
LABEL_FILENAME = "offline_labels.npz"
EVALUATION_FILENAME = "evaluation_metadata.npz"
METADATA_FILENAME = "metadata.json"
EVENT_NAMES = ("o0", "o1", "o2", "r0", "r1")
_EVENT_ORDER = {name: index for index, name in enumerate(EVENT_NAMES)}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# This tuple is the complete numeric boundary available to a deployment feature
# loader. Identity, intervention annotations, and canonical arrays are separate.
DEPLOYABLE_TENSOR_KEYS = (
    *DEPLOYMENT_TENSOR_KEYS,
    "teacher_forced_factual_prior",
    "measurement_frame_embedding",
)
OFFLINE_LABEL_KEYS = (
    "canonical_current",
    "canonical_measurement_history",
    "canonical_target",
    "canonical_q_corrected_history",
    "canonical_q_corrected",
    "canonical_valid_history",
    "canonical_valid_current",
)
IDENTITY_KEYS = (
    "sample_index",
    "source_frame_index",
    "frame_id",
    "task_name",
    "episode_id",
    "event_name",
    "event_index",
    "phase_index",
    "stream_offset",
    "physical_episode_sha256",
    "history_source_frame_index",
    "history_frame_id",
    "action_source_frame_index",
    "action_frame_id",
)
FORBIDDEN_DEPLOYMENT_FIELDS = (
    *POSTERIOR_FEATURE_FORBIDDEN_FIELDS,
    "anchor_dropout_active",
    "anchor_recovery_onset",
    "anchor_recovery_segment",
    "task_index",
    "dropout_schedule",
)

EXECUTION_SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPO_ROOT / "scripts/cache_vit_backbone_features.py",
    REPO_ROOT / "scripts/cache_fcpg_corrections.py",
    REPO_ROOT / "scripts/evaluate_lewm_posterior_headroom.py",
    REPO_ROOT / "fcpg/lewm_counterfactual_v2_storage.py",
    REPO_ROOT / "fcpg/lewm_posterior_features.py",
    REPO_ROOT / "fcpg/lewm_posterior_gate_data.py",
    REPO_ROOT / "fcpg/lewm_runtime.py",
    REPO_ROOT / "fcpg/lewm_anchor_persistent.py",
    REPO_ROOT / "fcpg/lewm_grounded_model.py",
    REPO_ROOT / "fcpg/lewm_sequence_data.py",
    REPO_ROOT / "fcpg/lewm_provenance.py",
)


@dataclass(frozen=True)
class DatasetInputs:
    path: Path
    sha256: str
    observed_images: np.ndarray
    canonical_images: np.ndarray
    obs: np.ndarray
    action: np.ndarray
    task_name: np.ndarray
    episode_id: np.ndarray
    frame_id: np.ndarray
    keypoints_xyz: np.ndarray
    q_nominal: np.ndarray
    valid_nominal: np.ndarray
    dropout_active: np.ndarray
    frame_identity_sha256: str
    task_frame_identity_sha256: str

    @property
    def sample_count(self) -> int:
        return int(self.frame_id.shape[0])

    @property
    def image_size(self) -> tuple[int, int]:
        return (
            int(self.observed_images.shape[2]),
            int(self.observed_images.shape[1]),
        )

    @property
    def num_keypoints(self) -> int:
        return int(self.q_nominal.shape[1])


@dataclass(frozen=True)
class FeatureCache:
    path: Path
    metadata: dict[str, Any]
    metadata_path: Path
    feature_path: Path
    prefix_path: Path
    metadata_sha256: str
    feature_sha256: str
    prefix_sha256: str
    feature_shape: tuple[int, ...]
    prefix_shape: tuple[int, ...]


@dataclass(frozen=True)
class CorrectionCache:
    path: Path
    metadata: dict[str, Any]
    metadata_path: Path
    corrected_path: Path
    corrected: np.ndarray
    metadata_sha256: str
    corrected_sha256: str
    corrector_checkpoint: Path
    corrector_checkpoint_sha256: str
    reliability_path: Path | None
    reliability_sha256: str | None


@dataclass(frozen=True)
class EventRow:
    source_index: int
    task_name: str
    episode_id: str
    frame_id: int
    event_name: str
    event_anchor_index: int
    phase: int
    history_indices: np.ndarray
    history_frame_ids: np.ndarray
    action_indices: np.ndarray
    action_chunks: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", "--data", dest="data", type=Path, required=True)
    parser.add_argument("--observed-correction-cache", type=Path, required=True)
    parser.add_argument(
        "--canonical-correction-cache",
        "--target-correction-cache",
        dest="canonical_correction_cache",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        "--lewm-checkpoint",
        dest="checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-checkpoint-step",
        "--expected-step",
        dest="expected_checkpoint_step",
        type=int,
        default=EXPECTED_CHECKPOINT_STEP,
    )
    parser.add_argument("--history-size", type=int, default=FRESH_HISTORY_SIZE)
    parser.add_argument("--frame-skip", type=int, default=FRESH_FRAME_SKIP)
    parser.add_argument(
        "--observed-image-key",
        choices=tuple(IMAGE_VIEW_ROLES),
        default="rgb",
    )
    parser.add_argument(
        "--canonical-image-key",
        choices=tuple(IMAGE_VIEW_ROLES),
        default="rgb_camera_only",
    )
    parser.add_argument(
        "--evaluation-view",
        choices=EVALUATION_VIEWS,
        default="perturbed_observation",
        help=(
            "Use the perturbed deployment observation, or treat the paired "
            "unoccluded camera-only view as the actual observation for an "
            "independent clean-safety cohort"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output",
        type=Path,
        help="atomic, immutable NPZ containing role-partitioned keys",
    )
    output.add_argument(
        "--output-dir",
        dest="output",
        type=Path,
        help="atomic, immutable directory with physically separated NPZ roles",
    )
    return parser


def _resolve_file(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} is not a file: {result}")
    return result


def _resolve_directory(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {result}")
    return result


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _same_path(value: object, expected: Path) -> bool:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        return False
    return Path(value).expanduser().resolve() == expected.resolve()


def _scalar_string(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    raise ValueError(f"{label} must be a scalar string, got {array.shape}")


def _require_floating_finite(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{label} must use floating storage, got {array.dtype}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _task_frame_identity_sha256(
    task_name: np.ndarray,
    episode_id: np.ndarray,
    frame_id: np.ndarray,
) -> str:
    tasks = np.asarray(task_name).astype(str).reshape(-1)
    episodes = np.asarray(episode_id).astype(str).reshape(-1)
    frames = np.asarray(frame_id, dtype=np.int64).reshape(-1)
    if tasks.shape != episodes.shape or episodes.shape != frames.shape:
        raise ValueError("task/frame identity arrays must have the same shape")
    digest = hashlib.sha256()
    for task, episode, frame in zip(tasks, episodes, frames, strict=True):
        for value in (task, episode):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
        digest.update(int(frame).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _load_dataset(
    data_path: Path,
    *,
    observed_image_key: str,
    canonical_image_key: str,
) -> DatasetInputs:
    source_sha256 = sha256_file(data_path)
    if observed_image_key == canonical_image_key:
        raise ValueError("observed and canonical image keys must be different")
    expected_roles = {
        observed_image_key: "deployment_observation",
        canonical_image_key: "target_only",
    }
    for image_key, expected_role in expected_roles.items():
        if IMAGE_VIEW_ROLES.get(image_key) != expected_role:
            raise ValueError(
                f"image key {image_key!r} must have role {expected_role!r}"
            )

    required = {
        observed_image_key,
        canonical_image_key,
        "obs",
        "action",
        "task_name",
        "episode_id",
        "frame_id",
        "keypoints_xyz",
        "q_nominal",
        "valid_nominal",
        "anchor_dropout_active",
        "observation_action_alignment",
    }
    with np.load(data_path, allow_pickle=False) as raw:
        missing = required.difference(raw.files)
        if missing:
            raise KeyError(f"dataset is missing fields: {sorted(missing)}")
        if _scalar_string(
            raw["observation_action_alignment"], "observation_action_alignment"
        ) != "pre_action":
            raise ValueError("dataset observation_action_alignment must be 'pre_action'")

        observed = np.asarray(raw[observed_image_key])
        canonical = np.asarray(raw[canonical_image_key])
        if (
            observed.ndim != 4
            or observed.shape[-1] != 3
            or any(int(size) <= 0 for size in observed.shape)
        ):
            raise ValueError(
                f"{observed_image_key} must have positive [N,H,W,3] shape"
            )
        if observed.dtype != np.uint8:
            raise TypeError(f"{observed_image_key} must use uint8 storage")
        if canonical.shape != observed.shape or canonical.dtype != np.uint8:
            raise ValueError(
                f"{canonical_image_key} must be uint8 and match "
                f"{observed_image_key} shape"
            )

        count = int(observed.shape[0])
        obs = np.asarray(raw["obs"])
        action = np.asarray(raw["action"])
        task_name = np.asarray(raw["task_name"]).astype(str).reshape(-1)
        episode_id = np.asarray(raw["episode_id"]).astype(str).reshape(-1)
        frame_raw = np.asarray(raw["frame_id"])
        keypoints_xyz = np.asarray(raw["keypoints_xyz"])
        q_nominal = np.asarray(raw["q_nominal"])
        valid_nominal = np.asarray(raw["valid_nominal"])
        dropout_active = np.asarray(raw["anchor_dropout_active"])

    if not np.issubdtype(frame_raw.dtype, np.integer):
        raise TypeError("frame_id must use integer storage")
    frame_id = frame_raw.astype(np.int64, copy=False).reshape(-1)
    obs = _require_floating_finite(obs, "obs")
    action = _require_floating_finite(action, "action")
    keypoints_xyz = _require_floating_finite(keypoints_xyz, "keypoints_xyz")
    q_nominal = _require_floating_finite(q_nominal, "q_nominal")
    if obs.ndim != 2 or obs.shape[0] != count or obs.shape[1] <= 0:
        raise ValueError(f"obs must have shape [N,O], got {obs.shape}")
    if action.ndim != 2 or action.shape[0] != count or action.shape[1] <= 0:
        raise ValueError(f"action must have shape [N,A], got {action.shape}")
    if q_nominal.ndim != 3 or q_nominal.shape[0] != count or q_nominal.shape[-1] != 2:
        raise ValueError(f"q_nominal must have shape [N,K,2], got {q_nominal.shape}")
    num_keypoints = int(q_nominal.shape[1])
    if num_keypoints <= 0 or keypoints_xyz.shape != (count, num_keypoints, 3):
        raise ValueError(
            "keypoints_xyz must match q_nominal with three coordinates per anchor"
        )
    expected_mask_shape = (count, num_keypoints)
    if valid_nominal.shape != expected_mask_shape or valid_nominal.dtype != np.bool_:
        raise ValueError(
            f"valid_nominal must be boolean {expected_mask_shape}, got "
            f"{valid_nominal.shape} {valid_nominal.dtype}"
        )
    if dropout_active.shape != expected_mask_shape or dropout_active.dtype != np.bool_:
        raise ValueError(
            f"anchor_dropout_active must be boolean {expected_mask_shape}"
        )
    for label, value in (
        ("task_name", task_name),
        ("episode_id", episode_id),
        ("frame_id", frame_id),
    ):
        if value.shape != (count,):
            raise ValueError(f"{label} must contain one value per frame")
    if any(not value for value in task_name.tolist()):
        raise ValueError("task_name values must be non-empty")
    if any(not value for value in episode_id.tolist()):
        raise ValueError("episode_id values must be non-empty")

    identities = list(
        zip(task_name.tolist(), episode_id.tolist(), frame_id.tolist(), strict=True)
    )
    if len(set(identities)) != count:
        raise ValueError("dataset contains duplicate task/episode/frame identities")
    if sha256_file(data_path) != source_sha256:
        raise RuntimeError("dataset changed while it was being loaded")
    identity_sha256 = frame_identity_sha256(episode_id, frame_id)
    return DatasetInputs(
        path=data_path,
        sha256=source_sha256,
        observed_images=observed,
        canonical_images=canonical,
        obs=obs.astype(np.float32, copy=False),
        action=action.astype(np.float32, copy=False),
        task_name=task_name,
        episode_id=episode_id,
        frame_id=frame_id,
        keypoints_xyz=keypoints_xyz.astype(np.float32, copy=False),
        q_nominal=q_nominal.astype(np.float32, copy=False),
        valid_nominal=valid_nominal,
        dropout_active=dropout_active,
        frame_identity_sha256=identity_sha256,
        task_frame_identity_sha256=_task_frame_identity_sha256(
            task_name, episode_id, frame_id
        ),
    )


def _finite_memmap(array: np.ndarray, label: str, chunk_size: int = 1024) -> None:
    for start in range(0, int(array.shape[0]), chunk_size):
        if not bool(np.isfinite(array[start : start + chunk_size]).all()):
            raise ValueError(f"{label} contains non-finite values")


def _load_feature_cache(
    cache_path: Path,
    *,
    dataset: DatasetInputs,
    selected_images: np.ndarray,
    image_key: str,
    required_view_role: str,
    deployment_cache_eligible: bool,
) -> FeatureCache:
    metadata_path = cache_path / "metadata.json"
    feature_path = cache_path / "backbone_features.npy"
    prefix_path = cache_path / "prefix.npy"
    for path in (metadata_path, feature_path, prefix_path):
        if not path.is_file():
            raise FileNotFoundError(f"feature cache is missing {path.name}: {cache_path}")
    metadata = _read_json_object(metadata_path, "feature cache metadata")
    if not _same_path(metadata.get("source_data"), dataset.path):
        raise ValueError(f"feature cache source_data differs: {cache_path}")
    if _required_sha256(
        metadata.get("source_data_sha256"), "feature source_data_sha256"
    ) != dataset.sha256:
        raise ValueError(f"feature cache source hash differs: {cache_path}")
    if metadata.get("image_key") != image_key:
        raise ValueError(f"feature cache image_key differs: {cache_path}")
    if metadata.get("view_role") != required_view_role:
        raise ValueError(f"feature cache view_role differs: {cache_path}")
    if metadata.get("deployment_cache_eligible") is not deployment_cache_eligible:
        raise ValueError(
            f"feature cache deployment_cache_eligible differs: {cache_path}"
        )
    selected_sha256 = _selected_image_sha256(selected_images)
    if _required_sha256(
        metadata.get("selected_image_sha256"), "feature selected_image_sha256"
    ) != selected_sha256:
        raise ValueError(f"feature cache selected image hash differs: {cache_path}")
    expected_count, height, width = selected_images.shape[:3]
    if int(metadata.get("sample_count", -1)) != expected_count:
        raise ValueError(f"feature cache sample count differs: {cache_path}")
    if (
        int(metadata.get("image_width", -1)) != width
        or int(metadata.get("image_height", -1)) != height
    ):
        raise ValueError(f"feature cache image shape differs: {cache_path}")
    if "source_size_bytes" in metadata and int(metadata["source_size_bytes"]) != int(
        dataset.path.stat().st_size
    ):
        raise ValueError(f"feature cache source size differs: {cache_path}")

    feature_sha256 = sha256_file(feature_path)
    prefix_sha256 = sha256_file(prefix_path)
    if _required_sha256(
        metadata.get("backbone_features_sha256"),
        "feature backbone_features_sha256",
    ) != feature_sha256:
        raise ValueError(f"feature cache backbone feature hash mismatch: {cache_path}")
    if _required_sha256(
        metadata.get("prefix_sha256"), "feature prefix_sha256"
    ) != prefix_sha256:
        raise ValueError(f"feature cache prefix hash mismatch: {cache_path}")
    feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    prefix = np.load(prefix_path, mmap_mode="r", allow_pickle=False)
    if (
        feature.ndim != 5
        or feature.shape[0] != expected_count
        or any(int(size) <= 0 for size in feature.shape[1:])
        or not np.issubdtype(feature.dtype, np.floating)
    ):
        raise ValueError(f"invalid backbone feature array: {feature.shape} {feature.dtype}")
    if (
        prefix.ndim != 2
        or prefix.shape[0] != expected_count
        or prefix.shape[1] != feature.shape[2]
        or not np.issubdtype(prefix.dtype, np.floating)
    ):
        raise ValueError(f"invalid prefix array: {prefix.shape} {prefix.dtype}")
    selected_layers = metadata.get("selected_layers")
    if (
        not isinstance(selected_layers, Sequence)
        or isinstance(selected_layers, (str, bytes))
        or len(selected_layers) != feature.shape[1]
    ):
        raise ValueError("feature cache selected_layers do not match feature array")
    if "feature_shape" in metadata and tuple(metadata["feature_shape"]) != tuple(
        feature.shape[1:]
    ):
        raise ValueError("feature cache feature_shape metadata differs")
    if "prefix_dim" in metadata and int(metadata["prefix_dim"]) != prefix.shape[1]:
        raise ValueError("feature cache prefix_dim metadata differs")
    _finite_memmap(feature, "backbone_features")
    _finite_memmap(prefix, "prefix")
    return FeatureCache(
        path=cache_path,
        metadata=metadata,
        metadata_path=metadata_path,
        feature_path=feature_path,
        prefix_path=prefix_path,
        metadata_sha256=sha256_file(metadata_path),
        feature_sha256=feature_sha256,
        prefix_sha256=prefix_sha256,
        feature_shape=tuple(int(size) for size in feature.shape),
        prefix_shape=tuple(int(size) for size in prefix.shape),
    )


def _load_correction_cache(
    cache_path: Path,
    *,
    feature_cache: FeatureCache,
    dataset: DatasetInputs,
    image_key: str,
    required_view_role: str,
    deployment_cache_eligible: bool,
) -> CorrectionCache:
    corrected, metadata = _reference_load_correction_cache(
        cache_path,
        data_path=dataset.path,
        sample_count=dataset.sample_count,
        num_keypoints=dataset.num_keypoints,
        required_image_key=image_key,
        required_view_role=required_view_role,
        deployment_cache_eligible=deployment_cache_eligible,
    )
    metadata_path = cache_path / "metadata.json"
    corrected_path = cache_path / "q_corrected.npy"
    if int(metadata.get("schema_version", -1)) < 3:
        raise ValueError("correction cache schema_version must be at least 3")
    if not bool(metadata.get("vision_image_lineage_verified", False)):
        raise ValueError("correction cache requires verified image lineage")
    if metadata.get("frame_identity_sha256") != dataset.frame_identity_sha256:
        raise ValueError("correction cache frame identity differs from dataset")
    if not _same_path(metadata.get("vision_cache"), feature_cache.path):
        raise ValueError("correction cache vision_cache differs from feature cache")
    if _required_sha256(
        metadata.get("vision_metadata_sha256"),
        "correction vision_metadata_sha256",
    ) != feature_cache.metadata_sha256:
        raise ValueError("correction cache vision metadata hash differs")
    if _required_sha256(
        metadata.get("backbone_features_sha256"),
        "correction backbone_features_sha256",
    ) != feature_cache.feature_sha256:
        raise ValueError("correction cache backbone feature hash differs")
    if _required_sha256(
        metadata.get("prefix_sha256"), "correction prefix_sha256"
    ) != feature_cache.prefix_sha256:
        raise ValueError("correction cache prefix hash differs")
    if metadata.get("selected_image_sha256") != feature_cache.metadata.get(
        "selected_image_sha256"
    ):
        raise ValueError("correction and feature selected image hashes differ")
    corrected_sha256 = sha256_file(corrected_path)
    if _required_sha256(
        metadata.get("q_corrected_sha256"), "correction q_corrected_sha256"
    ) != corrected_sha256:
        raise ValueError("correction cache q_corrected hash differs")

    checkpoint_value = metadata.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("correction cache is missing its frozen checkpoint")
    corrector_checkpoint = _resolve_file(
        Path(checkpoint_value), "correction checkpoint"
    )
    corrector_sha256 = sha256_file(corrector_checkpoint)
    if _required_sha256(
        metadata.get("checkpoint_sha256"), "correction checkpoint_sha256"
    ) != corrector_sha256:
        raise ValueError("correction checkpoint hash differs")

    reliability_path = cache_path / "reliability.npy"
    has_reliability = metadata.get("has_reliability") is True
    if has_reliability != reliability_path.is_file():
        raise ValueError("correction cache reliability presence differs from metadata")
    reliability_sha256: str | None = None
    if has_reliability:
        reliability = np.load(reliability_path, mmap_mode="r", allow_pickle=False)
        if reliability.shape != (dataset.sample_count, dataset.num_keypoints):
            raise ValueError("correction reliability shape differs from dataset")
        if not np.issubdtype(reliability.dtype, np.floating):
            raise TypeError("correction reliability must use floating storage")
        _finite_memmap(reliability, "correction reliability")
        reliability_sha256 = sha256_file(reliability_path)
    return CorrectionCache(
        path=cache_path,
        metadata=dict(metadata),
        metadata_path=metadata_path,
        corrected_path=corrected_path,
        corrected=corrected,
        metadata_sha256=sha256_file(metadata_path),
        corrected_sha256=corrected_sha256,
        corrector_checkpoint=corrector_checkpoint,
        corrector_checkpoint_sha256=corrector_sha256,
        reliability_path=reliability_path if has_reliability else None,
        reliability_sha256=reliability_sha256,
    )


def _validate_paired_caches(
    observed_feature: FeatureCache,
    canonical_feature: FeatureCache,
    observed_correction: CorrectionCache,
    canonical_correction: CorrectionCache,
) -> None:
    feature_contract_fields = ("model_name", "selected_layers", "dtype")
    for field in feature_contract_fields:
        if observed_feature.metadata.get(field) != canonical_feature.metadata.get(field):
            raise ValueError(f"observed/canonical feature cache {field} differs")
    if observed_feature.feature_shape[1:] != canonical_feature.feature_shape[1:]:
        raise ValueError("observed/canonical backbone feature shapes differ")
    if observed_feature.prefix_shape[1:] != canonical_feature.prefix_shape[1:]:
        raise ValueError("observed/canonical prefix shapes differ")
    if (
        observed_correction.corrector_checkpoint_sha256
        != canonical_correction.corrector_checkpoint_sha256
    ):
        raise ValueError("observed/canonical correction checkpoints differ")


def _checkpoint_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint {label} must be a mapping")
    return value


def _validate_checkpoint_contract(
    model: object,
    checkpoint: Mapping[str, Any],
    *,
    dataset: DatasetInputs,
    expected_step: int,
    history_size: int,
    frame_skip: int,
) -> None:
    if checkpoint.get("checkpoint_type") != ANCHOR_PERSISTENT_CHECKPOINT_TYPE:
        raise TypeError("checkpoint must contain AnchorPersistentLeWM")
    if not callable(getattr(model, "encode_frames", None)) or not callable(
        getattr(model, "predict", None)
    ):
        raise TypeError("loaded AnchorPersistentLeWM lacks encode/predict methods")
    if expected_step != EXPECTED_CHECKPOINT_STEP:
        raise ValueError(
            f"posterior Gate checkpoint step must be {EXPECTED_CHECKPOINT_STEP}"
        )
    if history_size != FRESH_HISTORY_SIZE:
        raise ValueError(
            f"posterior Gate history_size must be {FRESH_HISTORY_SIZE}"
        )
    if frame_skip != FRESH_FRAME_SKIP:
        raise ValueError(f"posterior Gate frame_skip must be {FRESH_FRAME_SKIP}")
    if int(checkpoint.get("step", -1)) != expected_step:
        raise ValueError("checkpoint step differs from --expected-checkpoint-step")
    data_config = _checkpoint_mapping(checkpoint.get("data_config"), "data_config")
    model_config = _checkpoint_mapping(checkpoint.get("model_config"), "model_config")
    world_config = _checkpoint_mapping(
        model_config.get("world_model"), "model_config.world_model"
    )
    frame_config = _checkpoint_mapping(
        model_config.get("frame"), "model_config.frame"
    )
    checkpoint_history_values = {
        int(data_config.get("history_size", -1)),
        int(world_config.get("history_size", -1)),
        int(getattr(model, "history_size", -1)),
    }
    if checkpoint_history_values != {history_size}:
        raise ValueError("history_size differs across CLI, checkpoint, and model")
    if int(data_config.get("frame_skip", -1)) != frame_skip:
        raise ValueError("frame_skip differs between CLI and checkpoint")
    if int(data_config.get("num_keypoints", -1)) != dataset.num_keypoints:
        raise ValueError("checkpoint keypoint count differs from dataset")
    if int(frame_config.get("num_keypoints", -1)) != dataset.num_keypoints:
        raise ValueError("checkpoint frame keypoint count differs from dataset")
    if int(getattr(model, "num_keypoints", -1)) != dataset.num_keypoints:
        raise ValueError("model keypoint count differs from dataset")
    obs_dim = int(frame_config.get("obs_dim", -1))
    model_obs_dim = int(
        getattr(getattr(model, "frame_encoder", None), "obs_dim", -1)
    )
    if obs_dim <= 0 or model_obs_dim != obs_dim:
        raise ValueError("checkpoint/model obs_dim differs")
    if int(dataset.obs.shape[1]) < obs_dim:
        raise ValueError("dataset does not contain the checkpoint proprio dimensions")
    raw_action_dim = int(dataset.action.shape[1])
    transition_action_dim = raw_action_dim * frame_skip
    if transition_action_dim != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"posterior Gate action_dim must be {EXPECTED_ACTION_DIM}, got "
            f"{transition_action_dim}"
        )
    if int(data_config.get("raw_action_dim", -1)) != raw_action_dim:
        raise ValueError("checkpoint raw_action_dim differs from dataset")
    if int(data_config.get("action_dim", -1)) != transition_action_dim:
        raise ValueError("checkpoint action_dim differs from causal action chunk")
    if int(world_config.get("action_dim", -1)) != transition_action_dim:
        raise ValueError("checkpoint world action_dim differs from causal action chunk")
    if int(getattr(model, "action_dim", -1)) != transition_action_dim:
        raise ValueError("model action_dim differs from causal action chunk")
    checkpoint_image_size = tuple(int(value) for value in data_config.get("image_size", ()))
    if checkpoint_image_size != dataset.image_size:
        raise ValueError("checkpoint image_size differs from dataset")
    if int(getattr(model, "latent_dim", 0)) <= 0:
        raise ValueError("model latent_dim must be positive")


def _episode_sequences(dataset: DatasetInputs) -> dict[tuple[str, str], np.ndarray]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (task, episode) in enumerate(
        zip(dataset.task_name, dataset.episode_id, strict=True)
    ):
        groups.setdefault((str(task), str(episode)), []).append(index)
    result: dict[tuple[str, str], np.ndarray] = {}
    for key in sorted(groups):
        sequence = np.asarray(groups[key], dtype=np.int64)
        order = np.argsort(dataset.frame_id[sequence], kind="stable")
        sequence = sequence[order]
        frames = dataset.frame_id[sequence]
        if frames.size < 2 or not np.all(np.diff(frames) == 1):
            raise ValueError(f"episode {key} frame IDs must be contiguous")
        result[key] = sequence
    return result


def _shift_mask(value: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros(1, dtype=np.bool_), value[:-1]))


def _event_masks(masked: np.ndarray) -> dict[str, np.ndarray]:
    if masked.ndim != 1 or masked.dtype != np.bool_ or masked.size == 0:
        raise ValueError("event mask stream must be a non-empty boolean vector")
    previous = _shift_mask(masked)
    o0 = masked & ~previous
    o1 = masked & _shift_mask(o0)
    o2 = masked & _shift_mask(o1)
    r0 = ~masked & previous
    r1 = ~masked & _shift_mask(r0)
    return {"o0": o0, "o1": o1, "o2": o2, "r0": r0, "r1": r1}


def _derive_event_rows(
    dataset: DatasetInputs,
    *,
    history_size: int,
    frame_skip: int,
) -> list[EventRow]:
    rows: list[EventRow] = []
    for (task, episode), sequence in _episode_sequences(dataset).items():
        frames = dataset.frame_id[sequence]
        episode_mask = dataset.dropout_active[sequence]
        active_anchors = np.flatnonzero(np.any(episode_mask, axis=0))
        for anchor in active_anchors.tolist():
            active_phases: set[int] = set()
            for phase in range(frame_skip):
                stream_positions = np.flatnonzero(frames % frame_skip == phase)
                if stream_positions.size == 0:
                    continue
                stream_mask = episode_mask[stream_positions, anchor]
                if not bool(np.any(stream_mask)):
                    continue
                active_phases.add(phase)
                masks = _event_masks(stream_mask)
                counts = {name: int(mask.sum()) for name, mask in masks.items()}
                if any(counts[name] != 1 for name in EVENT_NAMES):
                    raise ValueError(
                        f"episode {(task, episode)} anchor {anchor} phase {phase} "
                        f"must contain exactly one {EVENT_NAMES} event; got {counts}"
                    )
                for event_name in EVENT_NAMES:
                    stream_offset = int(np.flatnonzero(masks[event_name])[0])
                    position = int(stream_positions[stream_offset])
                    if position < history_size * frame_skip:
                        raise ValueError(
                            f"event {(task, episode, int(frames[position]))} lacks "
                            "the requested causal history"
                        )
                    starts = [
                        position - (history_size - offset) * frame_skip
                        for offset in range(history_size)
                    ]
                    history_positions = np.asarray(starts, dtype=np.int64)
                    history_indices = sequence[history_positions]
                    expected_history_frames = int(frames[position]) - frame_skip * np.arange(
                        history_size, 0, -1, dtype=np.int64
                    )
                    history_frame_ids = dataset.frame_id[history_indices]
                    if not np.array_equal(history_frame_ids, expected_history_frames):
                        raise ValueError("history frame IDs violate frame_skip")
                    action_indices = np.stack(
                        [
                            sequence[start : start + frame_skip]
                            for start in starts
                        ]
                    ).astype(np.int64, copy=False)
                    chunks = _reference_action_context(
                        dataset.action,
                        sequence,
                        starts,
                        frame_skip,
                    )
                    rows.append(
                        EventRow(
                            source_index=int(sequence[position]),
                            task_name=task,
                            episode_id=episode,
                            frame_id=int(frames[position]),
                            event_name=event_name,
                            event_anchor_index=int(anchor),
                            phase=int(phase),
                            history_indices=history_indices,
                            history_frame_ids=history_frame_ids.copy(),
                            action_indices=action_indices,
                            action_chunks=chunks,
                        )
                    )
            if active_phases != set(range(frame_skip)):
                raise ValueError(
                    f"episode {(task, episode)} anchor {anchor} dropout schedule "
                    "does not cover every frame_skip phase"
                )
    if not rows:
        raise ValueError("dataset contains no posterior-gate events")
    rows.sort(
        key=lambda row: (
            row.task_name,
            row.episode_id,
            row.frame_id,
            _EVENT_ORDER[row.event_name],
            row.event_anchor_index,
        )
    )
    identities = [
        (
            row.task_name,
            row.episode_id,
            row.frame_id,
            row.event_name,
            row.event_anchor_index,
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("derived event identities are not unique")
    return rows


def _inside_image(q: np.ndarray, *, width: int, height: int) -> np.ndarray:
    return (
        (q[..., 0] >= 0.0)
        & (q[..., 0] <= float(width - 1))
        & (q[..., 1] >= 0.0)
        & (q[..., 1] <= float(height - 1))
    )


def _encode_anchor_states(
    model: AnchorPersistentLeWM,
    *,
    images: np.ndarray,
    obs: np.ndarray,
    q_nominal: np.ndarray,
    q_corrected: np.ndarray,
    keypoints_xyz: np.ndarray,
    valid_nominal: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Boundary around the evaluator's exact frozen frame encoding path."""

    return _reference_encode_frames(
        model,
        images=images,
        obs=obs,
        q_nominal=q_nominal,
        q_corrected=q_corrected,
        keypoints_xyz=keypoints_xyz,
        valid_nominal=valid_nominal,
        batch_size=batch_size,
        device=device,
    )


def _predict_factual_prior(
    model: AnchorPersistentLeWM,
    histories: np.ndarray,
    actions: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Boundary around the evaluator's teacher-forced factual predictor."""

    return _reference_predict(
        model,
        histories,
        actions,
        device=device,
        batch_size=batch_size,
    )


def _validate_state_array(
    value: np.ndarray,
    *,
    label: str,
    count: int,
    num_keypoints: int,
    latent_dim: int,
) -> np.ndarray:
    array = np.asarray(value)
    expected = (count, num_keypoints, latent_dim)
    if array.shape != expected or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{label} must be floating {expected}, got {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} contains non-finite values")
    return array.astype(np.float32, copy=False)


def _file_record(path: Path, role: str, sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256 or sha256_file(resolved),
    }


def _artifact_source_manifest(
    *,
    dataset: DatasetInputs,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    observed_feature: FeatureCache,
    canonical_feature: FeatureCache,
    observed_correction: CorrectionCache,
    canonical_correction: CorrectionCache,
) -> list[dict[str, Any]]:
    records = [
        _file_record(dataset.path, "dataset", dataset.sha256),
        _file_record(checkpoint_path, "lewm_checkpoint", checkpoint_sha256),
    ]
    for prefix, cache in (
        ("observed_feature", observed_feature),
        ("canonical_feature", canonical_feature),
    ):
        records.extend(
            (
                _file_record(cache.metadata_path, f"{prefix}_metadata", cache.metadata_sha256),
                _file_record(cache.feature_path, f"{prefix}_backbone", cache.feature_sha256),
                _file_record(cache.prefix_path, f"{prefix}_prefix", cache.prefix_sha256),
            )
        )
    for prefix, cache in (
        ("observed_correction", observed_correction),
        ("canonical_correction", canonical_correction),
    ):
        records.extend(
            (
                _file_record(cache.metadata_path, f"{prefix}_metadata", cache.metadata_sha256),
                _file_record(cache.corrected_path, f"{prefix}_q_corrected", cache.corrected_sha256),
                _file_record(
                    cache.corrector_checkpoint,
                    f"{prefix}_checkpoint",
                    cache.corrector_checkpoint_sha256,
                ),
            )
        )
        if cache.reliability_path is not None:
            records.append(
                _file_record(
                    cache.reliability_path,
                    f"{prefix}_reliability",
                    cache.reliability_sha256,
                )
            )
    return records


def _execution_manifest() -> list[dict[str, Any]]:
    records = []
    for path in EXECUTION_SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"execution source is missing: {path}")
        records.append(_file_record(path, "execution_source"))
    return records


def _manifest_arrays(
    records: Sequence[Mapping[str, Any]], prefix: str
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_role": np.asarray([str(row["role"]) for row in records]),
        f"{prefix}_path": np.asarray([str(row["path"]) for row in records]),
        f"{prefix}_size_bytes": np.asarray(
            [int(row["size_bytes"]) for row in records], dtype=np.int64
        ),
        f"{prefix}_sha256": np.asarray([str(row["sha256"]) for row in records]),
    }


def _verify_manifest_unchanged(records: Sequence[Mapping[str, Any]]) -> None:
    for row in records:
        path = Path(str(row["path"]))
        if not path.is_file():
            raise RuntimeError(f"source file disappeared during cache build: {path}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"source file size changed during cache build: {path}")
        if sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"source file hash changed during cache build: {path}")


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    for name, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise TypeError(f"NPZ field {name!r} must not use object storage")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite output: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _assemble_cache(
    *,
    dataset: DatasetInputs,
    rows: Sequence[EventRow],
    observed_correction: CorrectionCache,
    canonical_correction: CorrectionCache,
    model: AnchorPersistentLeWM,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    observed_feature: FeatureCache,
    canonical_feature: FeatureCache,
    history_size: int,
    frame_skip: int,
    expected_step: int,
    batch_size: int,
    device: torch.device,
    artifact_manifest: list[dict[str, Any]],
    execution_manifest: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    event_indices = np.asarray([row.source_index for row in rows], dtype=np.int64)
    history_indices = np.stack([row.history_indices for row in rows]).astype(
        np.int64, copy=False
    )
    needed_observed_indices = np.unique(
        np.concatenate((event_indices, history_indices.reshape(-1)))
    )
    compact_lookup = np.full(dataset.sample_count, -1, dtype=np.int64)
    compact_lookup[needed_observed_indices] = np.arange(
        needed_observed_indices.size, dtype=np.int64
    )

    observed_states = _encode_anchor_states(
        model,
        images=dataset.observed_images[needed_observed_indices],
        obs=dataset.obs[needed_observed_indices],
        q_nominal=dataset.q_nominal[needed_observed_indices],
        q_corrected=np.asarray(observed_correction.corrected[needed_observed_indices]),
        keypoints_xyz=dataset.keypoints_xyz[needed_observed_indices],
        valid_nominal=dataset.valid_nominal[needed_observed_indices],
        batch_size=batch_size,
        device=device,
    )
    observed_states = _validate_state_array(
        observed_states,
        label="observed anchor states",
        count=int(needed_observed_indices.size),
        num_keypoints=dataset.num_keypoints,
        latent_dim=int(model.latent_dim),
    )
    canonical_target = _encode_anchor_states(
        model,
        images=dataset.canonical_images[event_indices],
        obs=dataset.obs[event_indices],
        q_nominal=dataset.q_nominal[event_indices],
        q_corrected=np.asarray(canonical_correction.corrected[event_indices]),
        keypoints_xyz=dataset.keypoints_xyz[event_indices],
        valid_nominal=dataset.valid_nominal[event_indices],
        batch_size=batch_size,
        device=device,
    )
    canonical_target = _validate_state_array(
        canonical_target,
        label="canonical target",
        count=len(rows),
        num_keypoints=dataset.num_keypoints,
        latent_dim=int(model.latent_dim),
    )
    z_measurement = observed_states[compact_lookup[event_indices]]
    factual_histories = observed_states[compact_lookup[history_indices]]
    action_chunks = np.stack([row.action_chunks for row in rows]).astype(
        np.float32, copy=False
    )
    expected_action_shape = (
        len(rows),
        history_size,
        frame_skip * int(dataset.action.shape[1]),
    )
    if action_chunks.shape != expected_action_shape or not np.isfinite(action_chunks).all():
        raise ValueError(
            f"causal action chunks must have shape {expected_action_shape}"
        )
    z_prior = _predict_factual_prior(
        model,
        factual_histories,
        action_chunks,
        device=device,
        batch_size=batch_size,
    )
    z_prior = _validate_state_array(
        z_prior,
        label="teacher-forced factual prior",
        count=len(rows),
        num_keypoints=dataset.num_keypoints,
        latent_dim=int(model.latent_dim),
    )

    width, height = dataset.image_size
    history_q = np.asarray(observed_correction.corrected[history_indices])
    online_validity_history = (
        dataset.valid_nominal[history_indices]
        & _inside_image(history_q, width=width, height=height)
    ).transpose(0, 2, 1)
    online_gate_history = np.zeros(
        (len(rows), dataset.num_keypoints, history_size), dtype=np.float32
    )
    causal_action_summary = action_chunks[:, -1].copy()

    arrays: dict[str, np.ndarray] = {
        "z_measurement": np.asarray(z_measurement, dtype=np.float32),
        "z_prior": np.asarray(z_prior, dtype=np.float32),
        "q_nominal": dataset.q_nominal[event_indices].astype(np.float32, copy=True),
        "q_corrected": np.asarray(
            observed_correction.corrected[event_indices], dtype=np.float32
        ),
        "valid_nominal": dataset.valid_nominal[event_indices].copy(),
        "online_validity_history": online_validity_history.astype(np.bool_, copy=False),
        "online_gate_history": online_gate_history,
        "causal_action_summary": causal_action_summary,
        "causal_action_chunks": action_chunks,
        "image_width": np.asarray(width, dtype=np.int64),
        "image_height": np.asarray(height, dtype=np.int64),
        "canonical_target": np.asarray(canonical_target, dtype=np.float32),
        "sample_index": np.arange(len(rows), dtype=np.int64),
        "source_frame_index": event_indices,
        "frame_id": np.asarray([row.frame_id for row in rows], dtype=np.int64),
        "task_name": np.asarray([row.task_name for row in rows]),
        "episode_id": np.asarray([row.episode_id for row in rows]),
        "event_name": np.asarray([row.event_name for row in rows]),
        "event_anchor_index": np.asarray(
            [row.event_anchor_index for row in rows], dtype=np.int64
        ),
        "phase": np.asarray([row.phase for row in rows], dtype=np.int64),
        "history_source_frame_index": history_indices,
        "history_frame_id": np.stack([row.history_frame_ids for row in rows]).astype(
            np.int64, copy=False
        ),
        "action_source_frame_index": np.stack(
            [row.action_indices for row in rows]
        ).astype(np.int64, copy=False),
    }
    arrays.update(_manifest_arrays(artifact_manifest, "source_file"))
    arrays.update(_manifest_arrays(execution_manifest, "execution_file"))

    if set(DEPLOYABLE_TENSOR_KEYS) & set(FORBIDDEN_DEPLOYMENT_FIELDS):
        raise RuntimeError("deployable tensor whitelist contains a forbidden field")
    if set(DEPLOYABLE_TENSOR_KEYS) & (set(OFFLINE_LABEL_KEYS) | set(IDENTITY_KEYS)):
        raise RuntimeError("deployable tensor whitelist crosses an output role boundary")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "cache_type": CACHE_TYPE,
        "sample_count": len(rows),
        "num_keypoints": dataset.num_keypoints,
        "latent_dim": int(model.latent_dim),
        "history_size": history_size,
        "frame_skip": frame_skip,
        "raw_action_dim": int(dataset.action.shape[1]),
        "causal_action_chunk_dim": int(action_chunks.shape[-1]),
        "checkpoint_step": expected_step,
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "observed_image_key": observed_feature.metadata["image_key"],
        "canonical_image_key": canonical_feature.metadata["image_key"],
        "observed_view_role": "deployment_observation",
        "canonical_view_role": "target_only",
        "measurement_state_role": "deployment_visible_observed_rgb",
        "prior_role": "teacher_forced_factual_observed_measurement_history",
        "canonical_target_role": "detached_offline_label_only",
        "training_history": "deployment_measurement_forced",
        "online_gate_history_contract": "all_zero_measurement_identity",
        "online_validity_contract": "valid_nominal_and_corrected_in_bounds_only",
        "action_alignment": "pre_action_causal_chunks_ending_at_event_frame",
        "feature_cache_usage": "strict_correction_and_image_lineage_validation",
        "lewm_visual_encoding": "frozen_checkpoint_encoder_from_dataset_image_array",
        "event_identity_source": "dropout_schedule_annotation_only",
        "deployable_tensor_keys": list(DEPLOYABLE_TENSOR_KEYS),
        "offline_label_keys": list(OFFLINE_LABEL_KEYS),
        "statistics_identity_keys": list(IDENTITY_KEYS),
        "forbidden_deployment_fields": list(FORBIDDEN_DEPLOYMENT_FIELDS),
        "frame_identity_sha256": dataset.frame_identity_sha256,
        "task_frame_identity_sha256": dataset.task_frame_identity_sha256,
        "event_counts": {
            name: sum(row.event_name == name for row in rows) for name in EVENT_NAMES
        },
        "source_files": artifact_manifest,
        "execution_files": execution_manifest,
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    return arrays


@dataclass(frozen=True)
class GateDataset:
    path: Path
    sha256: str
    observed_images: np.ndarray
    canonical_images: np.ndarray
    obs: np.ndarray
    action: np.ndarray
    task_name: np.ndarray
    task_index: np.ndarray
    episode_id: np.ndarray
    env_seed: np.ndarray
    frame_id: np.ndarray
    task_generation_seed: int
    keypoints_xyz: np.ndarray
    q_true: np.ndarray
    q_nominal: np.ndarray
    valid_true: np.ndarray
    valid_nominal: np.ndarray
    dropout_active: np.ndarray
    recovery_onset: np.ndarray
    recovery_segment: np.ndarray
    measurement_valid: np.ndarray
    camera_step_frame: np.ndarray
    frame_identity_sha256: str
    task_frame_identity_sha256: str

    @property
    def sample_count(self) -> int:
        return int(self.frame_id.shape[0])

    @property
    def num_keypoints(self) -> int:
        return int(self.q_nominal.shape[1])

    @property
    def image_size(self) -> tuple[int, int]:
        return (
            int(self.observed_images.shape[2]),
            int(self.observed_images.shape[1]),
        )


@dataclass(frozen=True)
class GateCorrectionCache:
    path: Path
    metadata: dict[str, Any]
    metadata_path: Path
    corrected_path: Path
    corrected: np.ndarray
    metadata_sha256: str
    corrected_sha256: str
    corrector_checkpoint_sha256: str


def _scalar_integer(value: np.ndarray, label: str) -> int:
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label} must be a scalar integer")
    return int(array.reshape(-1)[0])


def _integer_frame_array(
    value: np.ndarray, *, label: str, count: int
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label} must be an integer [{count}] array")
    return array.astype(np.int64, copy=False)


def _boolean_frame_array(
    value: np.ndarray, *, label: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"{label} must be boolean {shape}, got {array.shape}")
    return array


def _load_gate_dataset(
    data_path: Path,
    *,
    observed_image_key: str,
    canonical_image_key: str,
) -> GateDataset:
    source_sha256 = sha256_file(data_path)
    if observed_image_key == canonical_image_key:
        raise ValueError("observed and canonical image keys must be different")
    if IMAGE_VIEW_ROLES.get(observed_image_key) != "deployment_observation":
        raise ValueError("observed image_key must have deployment_observation role")
    if IMAGE_VIEW_ROLES.get(canonical_image_key) != "target_only":
        raise ValueError("canonical image_key must have target_only role")

    required = {
        observed_image_key,
        canonical_image_key,
        "obs",
        "action",
        "task_name",
        "task_index",
        "episode_id",
        "env_seed",
        "frame_id",
        "task_generation_seed",
        "keypoints_xyz",
        "q_true",
        "q_nominal",
        "valid_true",
        "valid_nominal",
        "anchor_dropout_active",
        "anchor_recovery_onset",
        "anchor_recovery_segment",
        "measurement_valid",
        "camera_step_frame",
        "observation_action_alignment",
    }
    with np.load(data_path, allow_pickle=False) as raw:
        missing = required.difference(raw.files)
        if missing:
            raise KeyError(f"dataset is missing fields: {sorted(missing)}")
        alignment = _scalar_string(
            raw["observation_action_alignment"], "observation_action_alignment"
        )
        if alignment != "pre_action":
            raise ValueError("dataset observation_action_alignment must be 'pre_action'")
        observed = np.asarray(raw[observed_image_key])
        canonical = np.asarray(raw[canonical_image_key])
        obs = np.asarray(raw["obs"])
        action = np.asarray(raw["action"])
        task_name = np.asarray(raw["task_name"]).astype(str).reshape(-1)
        task_index_raw = np.asarray(raw["task_index"])
        episode_id = np.asarray(raw["episode_id"]).astype(str).reshape(-1)
        env_seed_raw = np.asarray(raw["env_seed"])
        frame_raw = np.asarray(raw["frame_id"])
        task_generation_seed = _scalar_integer(
            raw["task_generation_seed"], "task_generation_seed"
        )
        keypoints_xyz = np.asarray(raw["keypoints_xyz"])
        q_true = np.asarray(raw["q_true"])
        q_nominal = np.asarray(raw["q_nominal"])
        valid_true = np.asarray(raw["valid_true"])
        valid_nominal = np.asarray(raw["valid_nominal"])
        dropout_active = np.asarray(raw["anchor_dropout_active"])
        recovery_onset = np.asarray(raw["anchor_recovery_onset"])
        recovery_segment = np.asarray(raw["anchor_recovery_segment"])
        measurement_valid = np.asarray(raw["measurement_valid"])
        camera_step_raw = np.asarray(raw["camera_step_frame"])

    if (
        observed.ndim != 4
        or observed.shape[-1] != 3
        or observed.dtype != np.uint8
        or any(int(size) <= 0 for size in observed.shape)
    ):
        raise ValueError(
            f"{observed_image_key} must be nonempty uint8 [N,H,W,3]"
        )
    if canonical.shape != observed.shape or canonical.dtype != np.uint8:
        raise ValueError(
            f"{canonical_image_key} must be uint8 and match "
            f"{observed_image_key} shape"
        )
    count = int(observed.shape[0])
    task_index = _integer_frame_array(
        task_index_raw, label="task_index", count=count
    )
    env_seed = _integer_frame_array(env_seed_raw, label="env_seed", count=count)
    frame_id = _integer_frame_array(frame_raw, label="frame_id", count=count)
    camera_step_frame = _integer_frame_array(
        camera_step_raw, label="camera_step_frame", count=count
    )
    for label, value in (("task_name", task_name), ("episode_id", episode_id)):
        if value.shape != (count,) or any(not item for item in value.tolist()):
            raise ValueError(f"{label} must contain {count} non-empty values")

    obs = _require_floating_finite(obs, "obs")
    action = _require_floating_finite(action, "action")
    keypoints_xyz = _require_floating_finite(keypoints_xyz, "keypoints_xyz")
    q_true = _require_floating_finite(q_true, "q_true")
    q_nominal = _require_floating_finite(q_nominal, "q_nominal")
    if obs.ndim != 2 or obs.shape[0] != count or obs.shape[1] <= 0:
        raise ValueError("obs must have nonempty shape [N,O]")
    if action.ndim != 2 or action.shape[0] != count or action.shape[1] <= 0:
        raise ValueError("action must have nonempty shape [N,A]")
    if q_nominal.ndim != 3 or q_nominal.shape[0] != count or q_nominal.shape[-1] != 2:
        raise ValueError("q_nominal must have shape [N,K,2]")
    anchors = int(q_nominal.shape[1])
    if anchors <= 0 or q_true.shape != q_nominal.shape:
        raise ValueError("q_true must match nonempty q_nominal [N,K,2]")
    if keypoints_xyz.shape != (count, anchors, 3):
        raise ValueError("keypoints_xyz must have shape [N,K,3]")
    mask_shape = (count, anchors)
    valid_true = _boolean_frame_array(
        valid_true, label="valid_true", shape=mask_shape
    )
    valid_nominal = _boolean_frame_array(
        valid_nominal, label="valid_nominal", shape=mask_shape
    )
    dropout_active = _boolean_frame_array(
        dropout_active, label="anchor_dropout_active", shape=mask_shape
    )
    recovery_onset = _boolean_frame_array(
        recovery_onset, label="anchor_recovery_onset", shape=mask_shape
    )
    recovery_segment = _boolean_frame_array(
        recovery_segment, label="anchor_recovery_segment", shape=mask_shape
    )
    measurement_valid = _boolean_frame_array(
        measurement_valid, label="measurement_valid", shape=mask_shape
    )
    if bool(np.any(dropout_active & recovery_segment)):
        raise ValueError("dropout and recovery schedules overlap")
    if bool(np.any(recovery_onset & ~recovery_segment)):
        raise ValueError("recovery onset must lie inside the recovery segment")
    if bool(np.any(dropout_active & measurement_valid)):
        raise ValueError("dropout-active anchors must not be measurement-valid")

    identities = list(
        zip(task_name.tolist(), episode_id.tolist(), frame_id.tolist(), strict=True)
    )
    if len(set(identities)) != count:
        raise ValueError("dataset contains duplicate task/episode/frame identities")
    episode_contract: dict[tuple[str, str], tuple[int, int]] = {}
    for task, episode, task_value, env_value in zip(
        task_name, episode_id, task_index, env_seed, strict=True
    ):
        key = (str(task), str(episode))
        identity = (int(task_value), int(env_value))
        previous = episode_contract.setdefault(key, identity)
        if previous != identity:
            raise ValueError(f"physical episode identity changes inside {key}")

    if sha256_file(data_path) != source_sha256:
        raise RuntimeError("dataset changed while it was being loaded")
    identity_sha256 = frame_identity_sha256(episode_id, frame_id)
    return GateDataset(
        path=data_path,
        sha256=source_sha256,
        observed_images=observed,
        canonical_images=canonical,
        obs=obs.astype(np.float32, copy=False),
        action=action.astype(np.float32, copy=False),
        task_name=task_name,
        task_index=task_index,
        episode_id=episode_id,
        env_seed=env_seed,
        frame_id=frame_id,
        task_generation_seed=task_generation_seed,
        keypoints_xyz=keypoints_xyz.astype(np.float32, copy=False),
        q_true=q_true.astype(np.float32, copy=False),
        q_nominal=q_nominal.astype(np.float32, copy=False),
        valid_true=valid_true,
        valid_nominal=valid_nominal,
        dropout_active=dropout_active,
        recovery_onset=recovery_onset,
        recovery_segment=recovery_segment,
        measurement_valid=measurement_valid,
        camera_step_frame=camera_step_frame,
        frame_identity_sha256=identity_sha256,
        task_frame_identity_sha256=_task_frame_identity_sha256(
            task_name, episode_id, frame_id
        ),
    )


def _load_gate_correction_cache(
    cache_path: Path,
    *,
    dataset: GateDataset,
    selected_images: np.ndarray,
    image_key: str,
    required_view_role: str,
    deployment_cache_eligible: bool,
) -> GateCorrectionCache:
    corrected, metadata = _reference_load_correction_cache(
        cache_path,
        data_path=dataset.path,
        sample_count=dataset.sample_count,
        num_keypoints=dataset.num_keypoints,
        required_image_key=image_key,
        required_view_role=required_view_role,
        deployment_cache_eligible=deployment_cache_eligible,
    )
    metadata_path = cache_path / "metadata.json"
    corrected_path = cache_path / "q_corrected.npy"
    if int(metadata.get("schema_version", -1)) < 3:
        raise ValueError("correction cache schema_version must be at least 3")
    if metadata.get("frame_identity_sha256") != dataset.frame_identity_sha256:
        raise ValueError("correction cache frame identity differs from dataset")
    if not bool(metadata.get("vision_image_lineage_verified", False)):
        raise ValueError("correction cache requires verified image lineage")
    selected_sha256 = _selected_image_sha256(selected_images)
    if metadata.get("selected_image_sha256") != selected_sha256:
        raise ValueError("correction cache selected image hash differs from dataset")
    corrected_sha256 = sha256_file(corrected_path)
    if _required_sha256(
        metadata.get("q_corrected_sha256"), "correction q_corrected_sha256"
    ) != corrected_sha256:
        raise ValueError("correction cache q_corrected hash differs")
    corrector_sha256 = _required_sha256(
        metadata.get("checkpoint_sha256"), "correction checkpoint_sha256"
    )
    return GateCorrectionCache(
        path=cache_path,
        metadata=dict(metadata),
        metadata_path=metadata_path,
        corrected_path=corrected_path,
        corrected=corrected,
        metadata_sha256=sha256_file(metadata_path),
        corrected_sha256=corrected_sha256,
        corrector_checkpoint_sha256=corrector_sha256,
    )


def _array_copy(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.array(value, copy=True)


def _metadata_episode_array(values: Sequence[object]) -> np.ndarray:
    if all(isinstance(value, (int, np.integer)) for value in values):
        return np.asarray(values, dtype=np.int64)
    return np.asarray([str(value) for value in values])


def _assert_deployment_partition(values: Mapping[str, np.ndarray]) -> None:
    forbidden_fragments = (
        "q_true",
        "valid_true",
        "measurement_valid",
        "dropout",
        "recovery",
        "schedule",
        "canonical",
        "target",
        "label",
        "oracle",
        "event",
        "phase",
        "task",
        "episode",
        "camera_step",
    )
    leaks = sorted(
        key
        for key in values
        if any(fragment in key.lower() for fragment in forbidden_fragments)
    )
    if leaks:
        raise ValueError(f"evaluation/label data leaked into deployment: {leaks}")
    for key, value in values.items():
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise ValueError(f"deployment tensor {key} contains non-finite values")


def _build_gate_partitions(
    *,
    dataset: GateDataset,
    measurement_embedding: np.ndarray,
    canonical_embedding: np.ndarray,
    observed_correction: GateCorrectionCache,
    canonical_correction: GateCorrectionCache,
    model: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    examples = build_fresh_posterior_gate_examples(
        task=dataset.task_name,
        episode=dataset.episode_id,
        frame=dataset.frame_id,
        action=dataset.action,
        measurement_embedding=measurement_embedding,
        canonical_label=canonical_embedding,
        q_nominal=dataset.q_nominal,
        q_corrected=np.asarray(observed_correction.corrected),
        valid_nominal=dataset.valid_nominal,
        anchor_dropout_active=dataset.dropout_active,
        anchor_recovery_segment=dataset.recovery_segment,
        observation_action_alignment="pre_action",
        frame_skip=FRESH_FRAME_SKIP,
        history_size=FRESH_HISTORY_SIZE,
    )

    deployment = {
        key: _array_copy(examples.deployment[key]) for key in DEPLOYMENT_TENSOR_KEYS
    }
    action_chunks = deployment["action_chunks"]
    expected_action_shape = (
        examples.sample_count,
        FRESH_HISTORY_SIZE,
        EXPECTED_ACTION_DIM,
    )
    if action_chunks.shape != expected_action_shape:
        raise ValueError(
            f"flat action_chunks must have shape {expected_action_shape}, got "
            f"{action_chunks.shape}"
        )
    factual_prior = _predict_factual_prior(
        model,
        deployment["measurement_history"],
        action_chunks,
        device=device,
        batch_size=batch_size,
    )
    factual_prior = _validate_state_array(
        factual_prior,
        label="teacher-forced factual prior",
        count=examples.sample_count,
        num_keypoints=dataset.num_keypoints,
        latent_dim=int(model.latent_dim),
    )
    deployment["teacher_forced_factual_prior"] = np.array(
        factual_prior, dtype=np.float32, copy=True
    )
    deployment["measurement_frame_embedding"] = np.array(
        measurement_embedding, dtype=np.float32, copy=True
    )
    _assert_deployment_partition(deployment)

    metadata_rows = examples.metadata
    target_indices = np.asarray(
        [row.target_raw_index for row in metadata_rows], dtype=np.int64
    )
    frame_lookup = {
        (str(task), str(episode), int(frame)): index
        for index, (task, episode, frame) in enumerate(
            zip(
                dataset.task_name,
                dataset.episode_id,
                dataset.frame_id,
                strict=True,
            )
        )
    }
    try:
        history_indices = np.asarray(
            [
                [
                    frame_lookup[(str(row.task), str(row.episode_id), int(frame))]
                    for frame in row.history_frames
                ]
                for row in metadata_rows
            ],
            dtype=np.int64,
        )
        action_indices = np.asarray(
            [
                [
                    [
                        frame_lookup[
                            (str(row.task), str(row.episode_id), int(frame))
                        ]
                        for frame in chunk
                    ]
                    for chunk in row.action_frames
                ]
                for row in metadata_rows
            ],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RuntimeError(f"builder metadata references an unknown frame: {exc}") from exc

    canonical_current = _array_copy(examples.labels["canonical_current"])
    width, height = dataset.image_size
    canonical_q_history = np.asarray(
        canonical_correction.corrected[history_indices], dtype=np.float32
    )
    canonical_q_current = np.asarray(
        canonical_correction.corrected[target_indices], dtype=np.float32
    )
    canonical_valid_history = dataset.valid_nominal[history_indices] & _inside_image(
        canonical_q_history, width=width, height=height
    )
    canonical_valid_current = dataset.valid_nominal[target_indices] & _inside_image(
        canonical_q_current, width=width, height=height
    )
    labels = {
        "canonical_current": np.array(
            canonical_current, dtype=np.float32, copy=True
        ),
        "canonical_measurement_history": np.array(
            canonical_embedding[history_indices], dtype=np.float32, copy=True
        ),
        "canonical_target": np.array(
            canonical_current, dtype=np.float32, copy=True
        ),
        "canonical_q_corrected_history": np.array(
            canonical_q_history, dtype=np.float32, copy=True
        ),
        "canonical_q_corrected": np.array(
            canonical_q_current, dtype=np.float32, copy=True
        ),
        "canonical_valid_history": canonical_valid_history.astype(
            np.bool_, copy=True
        ),
        "canonical_valid_current": canonical_valid_current.astype(
            np.bool_, copy=True
        ),
    }
    if set(labels) != set(OFFLINE_LABEL_KEYS):
        raise RuntimeError("offline canonical label schema changed")
    if set(labels) & set(DEPLOYABLE_TENSOR_KEYS):
        raise RuntimeError("target-only canonical arrays entered deployment whitelist")
    tasks = np.asarray([row.task for row in metadata_rows])
    episodes = _metadata_episode_array([row.episode_id for row in metadata_rows])
    task_indices = dataset.task_index[target_indices]
    env_seeds = dataset.env_seed[target_indices]
    physical_hashes = np.asarray(
        [
            physical_episode_sha256(
                str(task),
                int(task_index),
                int(env_seed),
                dataset.task_generation_seed,
            )
            for task, task_index, env_seed in zip(
                tasks, task_indices, env_seeds, strict=True
            )
        ]
    )
    if not np.array_equal(tasks, dataset.task_name[target_indices]):
        raise RuntimeError("builder task metadata differs from target frame identity")
    if not np.array_equal(
        np.asarray([str(value) for value in episodes]),
        dataset.episode_id[target_indices],
    ):
        raise RuntimeError("builder episode metadata differs from target frame identity")

    evaluation = {
        key: _array_copy(examples.evaluation[key]) for key in EVALUATION_MASK_KEYS
    }
    evaluation.update(
        {
            "sample_index": np.arange(examples.sample_count, dtype=np.int64),
            "task": tasks,
            "task_name": tasks.copy(),
            "task_index": task_indices.astype(np.int64, copy=True),
            "episode_id": episodes,
            "env_seed": env_seeds.astype(np.int64, copy=True),
            "task_generation_seed": np.full(
                examples.sample_count,
                dataset.task_generation_seed,
                dtype=np.int64,
            ),
            "physical_episode_sha256": physical_hashes,
            "phase_index": np.asarray(
                [row.phase_index for row in metadata_rows], dtype=np.int64
            ),
            "stream_offset": np.asarray(
                [row.stream_offset for row in metadata_rows], dtype=np.int64
            ),
            "frame_phase": np.asarray(
                [row.frame_phase for row in metadata_rows], dtype=np.int64
            ),
            "event": np.asarray([row.event for row in metadata_rows]),
            "event_index": np.asarray(
                [row.event_index for row in metadata_rows], dtype=np.int64
            ),
            "target_frame": np.asarray(
                [row.target_frame for row in metadata_rows], dtype=np.int64
            ),
            "target_raw_index": target_indices,
            "history_raw_indices": history_indices,
            "action_raw_indices": action_indices,
            "history_frames": np.asarray(
                [row.history_frames for row in metadata_rows], dtype=np.int64
            ),
            "action_frames": np.asarray(
                [row.action_frames for row in metadata_rows], dtype=np.int64
            ),
            "q_true_current": dataset.q_true[target_indices].copy(),
            "valid_true_current": dataset.valid_true[target_indices].copy(),
            "measurement_valid_current": dataset.measurement_valid[
                target_indices
            ].copy(),
            "anchor_dropout_active_current": dataset.dropout_active[
                target_indices
            ].copy(),
            "anchor_recovery_onset_current": dataset.recovery_onset[
                target_indices
            ].copy(),
            "anchor_recovery_segment_current": dataset.recovery_segment[
                target_indices
            ].copy(),
            "camera_step_frame_current": dataset.camera_step_frame[
                target_indices
            ].copy(),
            "source_task_name": dataset.task_name.copy(),
            "source_task_index": dataset.task_index.copy(),
            "source_episode_id": dataset.episode_id.copy(),
            "source_env_seed": dataset.env_seed.copy(),
            "source_frame_id": dataset.frame_id.copy(),
            "source_q_true": dataset.q_true.copy(),
            "source_valid_true": dataset.valid_true.copy(),
            "source_measurement_valid": dataset.measurement_valid.copy(),
            "source_anchor_dropout_active": dataset.dropout_active.copy(),
            "source_anchor_recovery_onset": dataset.recovery_onset.copy(),
            "source_anchor_recovery_segment": dataset.recovery_segment.copy(),
            "source_camera_step_frame": dataset.camera_step_frame.copy(),
        }
    )
    if tuple(sorted(set(evaluation["event"].tolist()))) != tuple(
        sorted(FRESH_EVENT_NAMES)
    ):
        raise RuntimeError("fresh builder did not emit the complete event set")
    return deployment, labels, evaluation


def _partition_record(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "keys": sorted(arrays),
        "shapes": {key: list(np.asarray(value).shape) for key, value in arrays.items()},
        "dtypes": {key: str(np.asarray(value).dtype) for key, value in arrays.items()},
    }


def _write_json_file(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish_cache_directory(
    output_dir: Path,
    *,
    deployment: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    evaluation: Mapping[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite cache directory: {output_dir}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)
        )
    )
    published = False
    try:
        partition_values = (
            (DEPLOYMENT_FILENAME, deployment),
            (LABEL_FILENAME, labels),
            (EVALUATION_FILENAME, evaluation),
        )
        artifact_records: dict[str, Any] = {}
        for filename, values in partition_values:
            path = temporary / filename
            _atomic_write_npz(path, values)
            with np.load(path, allow_pickle=False) as stored:
                if set(stored.files) != set(values):
                    raise RuntimeError(f"written NPZ key set differs for {filename}")
            artifact_records[filename] = _partition_record(path, values)
        metadata = dict(metadata)
        metadata["artifacts"] = artifact_records
        _write_json_file(temporary / METADATA_FILENAME, metadata)

        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite cache directory: {output_dir}"
            )
        os.rename(temporary, output_dir)
        published = True
        directory_fd = os.open(
            output_dir.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return metadata
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.expected_checkpoint_step != EXPECTED_CHECKPOINT_STEP:
        raise ValueError(
            f"posterior Gate requires checkpoint step {EXPECTED_CHECKPOINT_STEP}"
        )
    if args.history_size != FRESH_HISTORY_SIZE:
        raise ValueError(
            f"posterior Gate requires history_size {FRESH_HISTORY_SIZE}"
        )
    if args.frame_skip != FRESH_FRAME_SKIP:
        raise ValueError(f"posterior Gate requires frame_skip {FRESH_FRAME_SKIP}")

    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite cache directory: {output_path}")
    data_path = _resolve_file(args.data, "dataset")
    checkpoint_path = _resolve_file(args.checkpoint, "checkpoint")
    observed_correction_path = _resolve_directory(
        args.observed_correction_cache, "observed correction cache"
    )
    canonical_correction_path = _resolve_directory(
        args.canonical_correction_cache, "canonical correction cache"
    )

    dataset = _load_gate_dataset(
        data_path,
        observed_image_key=args.observed_image_key,
        canonical_image_key=args.canonical_image_key,
    )
    observed_correction = _load_gate_correction_cache(
        observed_correction_path,
        dataset=dataset,
        selected_images=dataset.observed_images,
        image_key=args.observed_image_key,
        required_view_role="deployment_observation",
        deployment_cache_eligible=True,
    )
    canonical_correction = _load_gate_correction_cache(
        canonical_correction_path,
        dataset=dataset,
        selected_images=dataset.canonical_images,
        image_key=args.canonical_image_key,
        required_view_role="target_only",
        deployment_cache_eligible=False,
    )
    if (
        observed_correction.corrector_checkpoint_sha256
        != canonical_correction.corrector_checkpoint_sha256
    ):
        raise ValueError("observed/canonical correction checkpoints differ")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    device = choose_device(args.device)
    model, _, checkpoint = load_lewm_grounded_checkpoint(
        checkpoint_path, device=device
    )
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed while it was being loaded")
    _validate_checkpoint_contract(
        model,
        checkpoint,
        dataset=dataset,
        expected_step=args.expected_checkpoint_step,
        history_size=args.history_size,
        frame_skip=args.frame_skip,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    observed_embedding = _encode_anchor_states(
        model,
        images=dataset.observed_images,
        obs=dataset.obs,
        q_nominal=dataset.q_nominal,
        q_corrected=np.asarray(observed_correction.corrected),
        keypoints_xyz=dataset.keypoints_xyz,
        valid_nominal=dataset.valid_nominal,
        batch_size=args.batch_size,
        device=device,
    )
    observed_embedding = _validate_state_array(
        observed_embedding,
        label="perturbed measurement frame embedding",
        count=dataset.sample_count,
        num_keypoints=dataset.num_keypoints,
        latent_dim=int(model.latent_dim),
    )
    canonical_embedding = _encode_anchor_states(
        model,
        images=dataset.canonical_images,
        obs=dataset.obs,
        q_nominal=dataset.q_nominal,
        q_corrected=np.asarray(canonical_correction.corrected),
        keypoints_xyz=dataset.keypoints_xyz,
        valid_nominal=dataset.valid_nominal,
        batch_size=args.batch_size,
        device=device,
    )
    canonical_embedding = _validate_state_array(
        canonical_embedding,
        label="target-only canonical frame embedding",
        count=dataset.sample_count,
        num_keypoints=dataset.num_keypoints,
        latent_dim=int(model.latent_dim),
    )

    clean_evaluation = args.evaluation_view == "canonical_unoccluded_only"
    if clean_evaluation:
        measurement_embedding = canonical_embedding
        measurement_correction = canonical_correction
        measurement_image_key = args.canonical_image_key
        measurement_view_role = "independent_clean_deployment_observation"
    else:
        measurement_embedding = observed_embedding
        measurement_correction = observed_correction
        measurement_image_key = args.observed_image_key
        measurement_view_role = "deployment_observation"

    deployment, labels, evaluation = _build_gate_partitions(
        dataset=dataset,
        measurement_embedding=measurement_embedding,
        canonical_embedding=canonical_embedding,
        observed_correction=measurement_correction,
        canonical_correction=canonical_correction,
        model=model,
        batch_size=args.batch_size,
        device=device,
    )
    event_values = evaluation["event"].astype(str)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "cache_type": CACHE_TYPE,
        "contract": {
            "checkpoint_frozen": True,
            "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
            "history_size": FRESH_HISTORY_SIZE,
            "frame_skip": FRESH_FRAME_SKIP,
            "action_dim": EXPECTED_ACTION_DIM,
            "observation_action_alignment": "pre_action",
            "deployment_label_partitioned": True,
            "measurement_frame_embedding_role": "deployment_observation",
            "teacher_forced_factual_prior_role": (
                "deployment_measurement_history_and_executed_actions_only"
            ),
            "canonical_arrays_role": (
                "target_only_clean_safety_diagnostic_and_offline_label"
            ),
            "canonical_encoded_frame_scope": (
                "all_source_frames_including_every_flat_history_and_target"
            ),
            "evaluation_view": args.evaluation_view,
            "clean_evaluation_only": clean_evaluation,
            "measurement_image_key": measurement_image_key,
            "measurement_view_role": measurement_view_role,
            "q_true_role": "evaluation_metadata_only",
            "measurement_valid_role": "evaluation_metadata_only",
            "dropout_schedule_role": "evaluation_metadata_only",
        },
        "frame_count": dataset.sample_count,
        "sample_count": int(evaluation["sample_index"].shape[0]),
        "num_keypoints": dataset.num_keypoints,
        "latent_dim": int(model.latent_dim),
        "image_width": dataset.image_size[0],
        "image_height": dataset.image_size[1],
        "frame_identity_sha256": dataset.frame_identity_sha256,
        "task_frame_identity_sha256": dataset.task_frame_identity_sha256,
        "event_counts": {
            name: int(np.count_nonzero(event_values == name))
            for name in FRESH_EVENT_NAMES
        },
        "deployable_tensor_keys": list(DEPLOYABLE_TENSOR_KEYS),
        "offline_label_keys": list(OFFLINE_LABEL_KEYS),
        "evaluation_mask_keys": list(EVALUATION_MASK_KEYS),
        "inputs": {
            "dataset": {"path": str(dataset.path), "sha256": dataset.sha256},
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "checkpoint_type": checkpoint.get("checkpoint_type"),
                "step": int(checkpoint["step"]),
            },
            "observed_correction_cache": {
                "path": str(observed_correction.path),
                "image_key": observed_correction.metadata["image_key"],
                "view_role": observed_correction.metadata["view_role"],
                "corrector_checkpoint_sha256": (
                    observed_correction.corrector_checkpoint_sha256
                ),
            },
            "measurement_correction_cache": {
                "path": str(measurement_correction.path),
                "image_key": measurement_correction.metadata["image_key"],
                "source_view_role": measurement_correction.metadata["view_role"],
                "evaluation_view_role": measurement_view_role,
                "corrector_checkpoint_sha256": (
                    measurement_correction.corrector_checkpoint_sha256
                ),
            },
            "canonical_correction_cache": {
                "path": str(canonical_correction.path),
                "image_key": canonical_correction.metadata["image_key"],
                "view_role": canonical_correction.metadata["view_role"],
                "corrector_checkpoint_sha256": (
                    canonical_correction.corrector_checkpoint_sha256
                ),
            },
        },
    }
    written_metadata = _publish_cache_directory(
        output_path,
        deployment=deployment,
        labels=labels,
        evaluation=evaluation,
        metadata=metadata,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_path),
                "sample_count": written_metadata["sample_count"],
                "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
