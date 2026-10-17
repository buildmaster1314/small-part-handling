"""End-to-end provenance audit for cached FCPG pixel corrections.

The evaluator must not trust a self-declared correction cache.  This module
replays the frozen corrector from raw RGB and non-oracle robot inputs, then
checks every scored cached coordinate against that replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from fcpg.lewm_provenance import sha256_file
from fcpg.observations import select_policy_observation
from fcpg.transformer_grounding import load_fcpg_vit_adapter


PROTOCOL = "fcpg_raw_rgb_correction_derivation_v1"
CORRECTION_ATOL_PX = 0.02
REQUIRED_INPUT_FIELDS = ("rgb", "q_nominal", "keypoints_xyz", "obs")
FORBIDDEN_ORACLE_FIELDS = ("q_true", "valid_true", "oracle_target")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _resolve(value: object, *, base: Path | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("required path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _indices_digest(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"fcpg-scored-frame-indices-v1\0")
    digest.update(values.tobytes())
    return digest.hexdigest()


def _extract_corrected(output: object, q_nominal: torch.Tensor) -> torch.Tensor:
    corrected = getattr(output, "q_corrected_px", None)
    if corrected is not None:
        return corrected
    delta = getattr(output, "delta_px", None)
    if delta is None:
        raise AttributeError(
            "frozen corrector must return q_corrected_px or delta_px"
        )
    return q_nominal + delta


def _validate_cache_contract(
    data_path: Path,
    cache_path: Path,
    *,
    require_schema3: bool,
) -> tuple[dict[str, Any], Path, Path, Path]:
    if not cache_path.is_dir():
        raise ValueError("correction derivation requires a cache directory")
    metadata_path = cache_path / "metadata.json"
    corrected_path = cache_path / "q_corrected.npy"
    if not metadata_path.is_file() or not corrected_path.is_file():
        raise FileNotFoundError(
            "correction cache requires metadata.json and q_corrected.npy"
        )
    metadata = dict(
        _mapping(
            json.loads(metadata_path.read_text(encoding="utf-8")),
            "correction metadata",
        )
    )
    schema = int(metadata.get("schema_version", metadata.get("version", 0)))
    if schema < (3 if require_schema3 else 2):
        raise ValueError("correction cache schema is too old for this phase")
    source_path = _resolve(metadata.get("source_data"), base=metadata_path.parent)
    if source_path != data_path:
        raise ValueError("correction cache source path differs from evaluation data")
    source_sha = sha256_file(data_path)
    if require_schema3 and metadata.get("source_data_sha256") != source_sha:
        raise ValueError("correction cache source hash differs from evaluation data")
    checkpoint_path = _resolve(
        metadata.get("checkpoint"), base=metadata_path.parent
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    if require_schema3 and metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("correction checkpoint hash differs from metadata")
    corrected_sha = sha256_file(corrected_path)
    if require_schema3 and metadata.get("q_corrected_sha256") != corrected_sha:
        raise ValueError("correction array hash differs from metadata")
    if require_schema3 and metadata.get("q_true_usage") != "offline_metrics_only":
        raise ValueError("correction cache has an invalid q_true usage contract")
    return metadata, metadata_path, corrected_path, checkpoint_path


def validate_fcpg_correction_derivations(
    jobs: Sequence[Mapping[str, object]],
    *,
    canonical_sink: Callable[
        [str, np.ndarray, np.ndarray], None
    ] | None = None,
    device: torch.device | str,
    batch_size: int = 64,
    require_schema3: bool,
    model_loader: Callable[..., tuple[torch.nn.Module, dict[str, object]]] = (
        load_fcpg_vit_adapter
    ),
) -> tuple[dict[str, object], list[Path]]:
    """Replay FCPG from raw RGB for every frame that contributes to a score."""

    if batch_size <= 0:
        raise ValueError("correction derivation batch_size must be positive")
    target_device = torch.device(device)
    model_cache: dict[
        tuple[str, str], tuple[torch.nn.Module, dict[str, object]]
    ] = {}
    reports: dict[str, object] = {}
    bound_paths: set[Path] = {Path(__file__).resolve()}
    for raw_job in jobs:
        job = _mapping(raw_job, "correction derivation job")
        name = str(job.get("name"))
        if not name or name in reports:
            raise ValueError("correction derivation job names must be unique")
        data_path = _resolve(job.get("data"))
        cache_path = _resolve(job.get("correction_cache"))
        metadata, metadata_path, corrected_path, checkpoint_path = (
            _validate_cache_contract(
                data_path,
                cache_path,
                require_schema3=require_schema3,
            )
        )
        indices = np.asarray(job.get("frame_indices"), dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"{name} scored frame indices must be nonempty 1-D")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} scored frame indices contain duplicates")
        indices = np.sort(indices)

        with np.load(data_path, allow_pickle=False) as raw:
            missing = [key for key in REQUIRED_INPUT_FIELDS if key not in raw.files]
            if missing:
                raise ValueError(f"{name} lacks non-oracle FCPG inputs: {missing}")
            sample_count = int(np.asarray(raw["rgb"]).shape[0])
            if indices[0] < 0 or indices[-1] >= sample_count:
                raise IndexError(f"{name} scored frame indices are out of range")
            rgb = np.asarray(raw["rgb"][indices], dtype=np.uint8)
            q_nominal = np.asarray(raw["q_nominal"][indices], dtype=np.float32)
            keypoints_xyz = np.asarray(
                raw["keypoints_xyz"][indices], dtype=np.float32
            )
            raw_obs = np.asarray(raw["obs"][indices], dtype=np.float32)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"{name} rgb must have shape [N,H,W,3]")
        expected_q = (len(indices), keypoints_xyz.shape[1], 2)
        if q_nominal.shape != expected_q:
            raise ValueError(f"{name} q_nominal has an invalid shape")
        if keypoints_xyz.shape != (len(indices), expected_q[1], 3):
            raise ValueError(f"{name} keypoints_xyz has an invalid shape")
        if not all(
            np.isfinite(value).all()
            for value in (q_nominal, keypoints_xyz, raw_obs)
        ):
            raise ValueError(f"{name} FCPG inputs contain non-finite values")

        corrected = np.load(corrected_path, mmap_mode="r", allow_pickle=False)
        if corrected.shape != (sample_count, expected_q[1], 2):
            raise ValueError(f"{name} correction array has an invalid shape")
        if not np.issubdtype(corrected.dtype, np.floating):
            raise ValueError(f"{name} correction array must be floating point")

        checkpoint_sha = sha256_file(checkpoint_path)
        model_key = (str(checkpoint_path), checkpoint_sha)
        if model_key not in model_cache:
            model, checkpoint_args = model_loader(
                checkpoint_path, target_device
            )
            model.to(target_device)
            model.eval()
            model_cache[model_key] = (model, dict(checkpoint_args))
        model, checkpoint_args = model_cache[model_key]
        if checkpoint_args.get("obs_mode") != "robot":
            raise ValueError(
                f"{name} corrector must explicitly use robot-only observations"
            )
        obs_mode = "robot"
        if metadata.get("obs_mode") != obs_mode:
            raise ValueError(f"{name} correction obs_mode differs from checkpoint")
        robot_obs = np.asarray(
            select_policy_observation(raw_obs, obs_mode), dtype=np.float32
        )

        max_abs = 0.0
        absolute_sum = 0.0
        element_count = 0
        canonical_corrected = np.empty(
            (len(indices), expected_q[1], 2), dtype=np.float32
        )
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                stop = min(start + batch_size, len(indices))
                rgb_tensor = (
                    torch.from_numpy(np.array(rgb[start:stop], copy=True))
                    .permute(0, 3, 1, 2)
                    .to(target_device, dtype=torch.float32)
                    / 255.0
                )
                q_tensor = torch.from_numpy(
                    np.array(q_nominal[start:stop], copy=True)
                ).to(target_device)
                xyz_tensor = torch.from_numpy(
                    np.array(keypoints_xyz[start:stop], copy=True)
                ).to(target_device)
                obs_tensor = torch.from_numpy(
                    np.array(robot_obs[start:stop], copy=True)
                ).to(target_device)
                with torch.autocast(
                    device_type=target_device.type,
                    dtype=torch.float16,
                    enabled=target_device.type == "cuda",
                ):
                    output = model(
                        rgb_tensor,
                        q_tensor,
                        xyz_tensor,
                        obs_tensor,
                    )
                replayed = (
                    _extract_corrected(output, q_tensor)
                    .float()
                    .cpu()
                    .numpy()
                )
                cached = np.asarray(
                    corrected[indices[start:stop]], dtype=np.float32
                )
                if (
                    replayed.shape != cached.shape
                    or not np.isfinite(replayed).all()
                    or not np.isfinite(cached).all()
                ):
                    raise ValueError(f"{name} replayed correction is invalid")
                canonical_corrected[start:stop] = replayed
                absolute = np.abs(replayed - cached)
                batch_max = float(absolute.max(initial=0.0))
                max_abs = max(max_abs, batch_max)
                absolute_sum += float(absolute.sum())
                element_count += int(absolute.size)
                if batch_max > CORRECTION_ATOL_PX:
                    raise ValueError(
                        f"{name} cached correction is not derived from raw RGB; "
                        f"max error {batch_max:.6f}px exceeds "
                        f"{CORRECTION_ATOL_PX:.6f}px"
                    )

        if canonical_sink is not None:
            canonical_sink(name, indices.copy(), canonical_corrected)

        reports[name] = {
            "passed": True,
            "scored_frames": int(len(indices)),
            "scored_frame_indices_sha256": _indices_digest(indices),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
            },
            "consumed_fields": list(REQUIRED_INPUT_FIELDS),
            "oracle_fields_consumed": [],
            "mode": "raw_rgb_end_to_end_frozen_vit_and_corrector_replay",
            "canonical_scoring_output_emitted": canonical_sink is not None,
            "absolute_tolerance_px": CORRECTION_ATOL_PX,
            "max_abs_error_px": max_abs,
            "mean_abs_error_px": absolute_sum / max(element_count, 1),
        }
        bound_paths.update(
            (data_path, metadata_path, corrected_path, checkpoint_path)
        )
    return {
        "passed": True,
        "canonical_scoring_outputs_emitted": canonical_sink is not None,
        "protocol": PROTOCOL,
        "all_scored_frames_replayed": True,
        "forbidden_oracle_fields": list(FORBIDDEN_ORACLE_FIELDS),
        "sources": reports,
    }, sorted(bound_paths)


__all__ = [
    "CORRECTION_ATOL_PX",
    "validate_fcpg_correction_derivations",
]
