#!/usr/bin/env python3
"""Measure zero-training posterior headroom for Anchor-Persistent LeWM.

This is an exposed-development diagnostic, not a deployable observer.  The
canonical view is target-only and may only be used to score frozen priors and
zero-training fusion upper bounds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.lewm_anchor_persistent import AnchorPersistentLeWM  # noqa: E402
from fcpg.lewm_posterior_headroom import oracle_convex_alpha  # noqa: E402
from fcpg.lewm_provenance import load_yaml_config, sha256_file  # noqa: E402
from fcpg.lewm_runtime import choose_device, load_lewm_grounded_checkpoint  # noqa: E402
from fcpg.lewm_sequence_data import frame_identity_sha256  # noqa: E402


MASKED_WINDOW = "phase_stream_masked"
NEXT_PRIOR_WINDOW = "phase_stream_next_pre_measurement"
_HASH_CHUNK_BYTES = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/lewm/eval/anchor_persistent_posterior_headroom_dev.yaml"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _resolve(path: object) -> Path:
    value = Path(str(path)).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve()


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _load_correction_cache(
    cache_dir: Path,
    *,
    data_path: Path,
    sample_count: int,
    num_keypoints: int,
    required_image_key: str,
    required_view_role: str,
    deployment_cache_eligible: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata_path = cache_dir / "metadata.json"
    corrected_path = cache_dir / "q_corrected.npy"
    if not metadata_path.is_file() or not corrected_path.is_file():
        raise FileNotFoundError(
            f"correction cache requires metadata.json and q_corrected.npy: {cache_dir}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if Path(str(metadata.get("source_data", ""))).resolve() != data_path:
        raise ValueError(f"correction cache source differs from dataset: {cache_dir}")
    if metadata.get("source_data_sha256") != sha256_file(data_path):
        raise ValueError(f"correction cache source hash differs: {cache_dir}")
    if int(metadata.get("sample_count", -1)) != sample_count:
        raise ValueError(f"correction cache sample count differs: {cache_dir}")
    if int(metadata.get("num_keypoints", -1)) != num_keypoints:
        raise ValueError(f"correction cache keypoint count differs: {cache_dir}")
    if metadata.get("view_role") != required_view_role:
        raise ValueError(
            f"correction cache {cache_dir} must have view_role="
            f"{required_view_role!r}"
        )
    if metadata.get("image_key") != required_image_key:
        raise ValueError(
            f"correction cache {cache_dir} must use image_key={required_image_key!r}"
        )
    if metadata.get("deployment_cache_eligible") is not deployment_cache_eligible:
        raise ValueError(
            f"correction cache deployment eligibility differs: {cache_dir}"
        )
    if not bool(metadata.get("vision_image_lineage_verified", False)):
        raise ValueError("correction cache requires verified image lineage")
    if metadata.get("q_true_usage") != "offline_metrics_only":
        raise ValueError("correction cache must restrict q_true to offline metrics")
    corrected = np.load(corrected_path, mmap_mode="r", allow_pickle=False)
    expected = (sample_count, num_keypoints, 2)
    if corrected.shape != expected or not np.issubdtype(corrected.dtype, np.floating):
        raise ValueError(
            f"q_corrected must be floating {expected}, got {corrected.shape} "
            f"{corrected.dtype}"
        )
    if not bool(np.isfinite(corrected).all()):
        raise ValueError("q_corrected contains non-finite values")
    if metadata.get("q_corrected_sha256") != sha256_file(corrected_path):
        raise ValueError(f"correction array hash differs: {cache_dir}")
    return corrected, metadata


def _image_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), _HASH_CHUNK_BYTES):
        digest.update(view[start : start + _HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _execution_manifest() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "fcpg/lewm_posterior_headroom.py",
        REPO_ROOT / "fcpg/lewm_anchor_persistent.py",
        REPO_ROOT / "fcpg/lewm_grounded_model.py",
    )
    files = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_head = None
        git_dirty = None
    return {
        "algorithm": "sha256(path-and-content-manifest-v1)",
        "aggregate_sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
        "git_head": git_head,
        "git_worktree_dirty": git_dirty,
    }


def _validate_images(value: np.ndarray, *, key: str, count: int) -> np.ndarray:
    images = np.asarray(value)
    if images.shape[:1] != (count,) or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"{key} must have shape [N,H,W,3], got {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"{key} must use uint8 storage, got {images.dtype}")
    return images


def _inside(q: np.ndarray, *, width: int, height: int) -> np.ndarray:
    return (
        (q[..., 0] >= 0.0)
        & (q[..., 0] <= float(width - 1))
        & (q[..., 1] >= 0.0)
        & (q[..., 1] <= float(height - 1))
    )


def _encode_frames(
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
    count, height, width = images.shape[:3]
    output = np.empty(
        (count, model.num_keypoints, model.latent_dim), dtype=np.float32
    )
    valid = valid_nominal & _inside(q_corrected, width=width, height=height)
    obs_dim = int(model.frame_encoder.obs_dim)
    if obs.shape[1] < obs_dim:
        raise ValueError(f"obs has {obs.shape[1]} dims but model requires {obs_dim}")
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            pixels = torch.from_numpy(
                np.ascontiguousarray(images[start:stop].transpose(0, 3, 1, 2))
            ).to(device=device, dtype=torch.float32, non_blocking=True)
            pixels = (pixels / 255.0).unsqueeze(1)
            batch = {
                "pixels": pixels,
                "proprio": torch.from_numpy(
                    np.ascontiguousarray(obs[start:stop, :obs_dim])
                ).to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1),
                "q_nominal": torch.from_numpy(
                    np.ascontiguousarray(q_nominal[start:stop])
                ).to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1),
                "q_corrected": torch.from_numpy(
                    np.ascontiguousarray(q_corrected[start:stop])
                ).to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1),
                "keypoints_xyz": torch.from_numpy(
                    np.ascontiguousarray(keypoints_xyz[start:stop])
                ).to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1),
                "valid": torch.from_numpy(np.ascontiguousarray(valid[start:stop])).to(
                    device=device, dtype=torch.bool, non_blocking=True
                ).unsqueeze(1),
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                _, embedding = model.encode_frames(batch)
            output[start:stop] = embedding[:, 0].float().cpu().numpy()
            if stop == count or stop % (batch_size * 10) == 0:
                print(f"encoded {stop}/{count}", flush=True)
    return output


def _action_context(
    action: np.ndarray,
    sequence: np.ndarray,
    starts: list[int],
    frame_skip: int,
) -> np.ndarray:
    chunks = [action[sequence[start : start + frame_skip]].reshape(-1) for start in starts]
    return np.stack(chunks).astype(np.float32, copy=False)


def _predict(
    model: AnchorPersistentLeWM,
    histories: np.ndarray,
    actions: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    count = int(histories.shape[0])
    output = np.empty(
        (count, model.num_keypoints, model.latent_dim), dtype=np.float32
    )
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            history = torch.from_numpy(
                np.ascontiguousarray(histories[start:stop])
            ).to(device=device, dtype=torch.float32, non_blocking=True)
            action = torch.from_numpy(np.ascontiguousarray(actions[start:stop])).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            predicted = model.predict(history, action)[:, -1]
            output[start:stop] = predicted.float().cpu().numpy()
    return output


def _fusion_candidates(
    *,
    prior: np.ndarray,
    measurement: np.ndarray,
    hold_last: np.ndarray,
    constant_velocity: np.ndarray,
    target: np.ndarray,
    fixed_alphas: list[float],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    methods: dict[str, np.ndarray] = {
        "lewm_prior": prior,
        "measurement": measurement,
        "hold_last": hold_last,
        "constant_velocity": constant_velocity,
        "canonical_oracle": target,
    }
    for alpha in fixed_alphas:
        if math.isclose(alpha, 0.0) or math.isclose(alpha, 1.0):
            continue
        methods[f"fixed_alpha_{alpha:.2f}"] = prior + alpha * (measurement - prior)
    # The occluded RGB is still an available observation; measurement_valid=False
    # labels it as corrupted rather than absent.  The privileged upper bound must
    # therefore compare both endpoints instead of replacing measurement by prior.
    oracle_alpha = oracle_convex_alpha(
        prior[:, None], measurement[:, None], target[:, None]
    )[:, 0, :, None]
    methods["oracle_convex"] = prior + oracle_alpha * (measurement - prior)
    prior_error = np.mean((prior - target) ** 2, axis=-1, keepdims=True)
    measurement_error = np.mean((measurement - target) ** 2, axis=-1, keepdims=True)
    methods["oracle_endpoint"] = np.where(
        prior_error <= measurement_error, prior, measurement
    )
    return methods, oracle_alpha[..., 0]


def _selective_writeback(
    candidate: np.ndarray,
    measurement: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    if candidate.shape != measurement.shape:
        raise ValueError("candidate and measurement writeback shapes must match")
    if active_mask.shape != candidate.shape[:-1] or active_mask.dtype != np.bool_:
        raise ValueError("active_mask must be boolean [event,anchor]")
    return np.where(active_mask[..., None], candidate, measurement)


def _frame_records(
    methods: Mapping[str, np.ndarray],
    target: np.ndarray,
    events: list[dict[str, Any]],
    *,
    window: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method, values in sorted(methods.items()):
        mse = np.mean((values - target) ** 2, axis=-1)
        for event_index, event in enumerate(events):
            for anchor in range(mse.shape[1]):
                records.append(
                    {
                        "task": event["task"],
                        "episode_id": event["episode_id"],
                        "frame_id": event["frame_id"],
                        "phase": event["phase"],
                        "anchor": anchor,
                        "directly_occluded": bool(event["active_mask"][anchor]),
                        "history_crosses_camera_step": bool(
                            event["history_crosses_camera_step"]
                        ),
                        "method": method,
                        "window": window,
                        "mse": float(mse[event_index, anchor]),
                    }
                )
    return records


def _episode_reduce(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(list)
    scopes: dict[tuple[str, str, int, str, str], str] = {}
    for row in records:
        key = (
            str(row["task"]),
            str(row["episode_id"]),
            int(row["anchor"]),
            str(row["method"]),
            str(row["window"]),
        )
        grouped[key].append(float(row["mse"]))
        scope = (
            "directly_occluded"
            if bool(row["directly_occluded"])
            else "indirectly_affected"
        )
        previous = scopes.setdefault(key, scope)
        if previous != scope:
            raise ValueError(f"anchor scope changed within episode record: {key}")
    return [
        {
            "task": key[0],
            "episode_id": key[1],
            "anchor": key[2],
            "method": key[3],
            "window": key[4],
            "mse": float(np.mean(values, dtype=np.float64)),
            "phase_count": len(values),
            "anchor_scope": scopes[key],
        }
        for key, values in sorted(grouped.items())
    ]


def _summarize_window(
    episode_records: list[dict[str, Any]],
    *,
    window: str,
    baseline: str,
    repetitions: int,
    seed: int,
    anchor_scope: str = "all",
) -> dict[str, Any]:
    if anchor_scope not in {"all", "directly_occluded", "indirectly_affected"}:
        raise ValueError(f"unknown anchor_scope {anchor_scope!r}")
    selected = [
        row
        for row in episode_records
        if row["window"] == window
        and (anchor_scope == "all" or row["anchor_scope"] == anchor_scope)
    ]
    methods = sorted({str(row["method"]) for row in selected})
    if baseline not in methods:
        raise ValueError(f"missing baseline {baseline!r} for window {window!r}")
    values = {
        (str(row["task"]), str(row["episode_id"]), int(row["anchor"]), str(row["method"])): float(row["mse"])
        for row in selected
    }
    tasks = sorted({key[0] for key in values})
    anchors_by_task = {
        task: sorted({key[2] for key in values if key[0] == task}) for task in tasks
    }
    episodes_by_task = {
        task: sorted({key[1] for key in values if key[0] == task}) for task in tasks
    }
    support = {
        (task, episode, anchor)
        for task in tasks
        for episode in episodes_by_task[task]
        for anchor in anchors_by_task[task]
    }
    for method in methods:
        actual = {key[:3] for key in values if key[3] == method}
        if actual != support:
            raise ValueError(f"method support mismatch for {method!r} in {window!r}")

    cells = [(task, anchor) for task in tasks for anchor in anchors_by_task[task]]
    rng = np.random.default_rng(seed)
    draws = {
        task: rng.integers(
            0,
            len(episodes_by_task[task]),
            size=(repetitions, len(episodes_by_task[task])),
        )
        for task in tasks
    }
    result: dict[str, Any] = {
        "window": window,
        "anchor_scope": anchor_scope,
        "baseline": baseline,
        "physical_episode_count": sum(len(value) for value in episodes_by_task.values()),
        "task_anchor_count": len(cells),
        "comparisons": {},
    }
    for method in methods:
        baseline_cells = []
        method_cells = []
        bootstrap = np.zeros(repetitions, dtype=np.float64)
        for task, anchor in cells:
            episodes = episodes_by_task[task]
            baseline_values = np.asarray(
                [values[(task, episode, anchor, baseline)] for episode in episodes],
                dtype=np.float64,
            )
            method_values = np.asarray(
                [values[(task, episode, anchor, method)] for episode in episodes],
                dtype=np.float64,
            )
            baseline_cells.append(float(baseline_values.mean()))
            method_cells.append(float(method_values.mean()))
            differences = baseline_values - method_values
            bootstrap += differences[draws[task]].mean(axis=1) / len(cells)
        result["comparisons"][method] = {
            "baseline_macro_mse": float(np.mean(baseline_cells)),
            "method_macro_mse": float(np.mean(method_cells)),
            "improvement": float(np.mean(np.asarray(baseline_cells) - method_cells)),
            "improvement_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        }
    return result


def _alpha_summary(
    alpha: np.ndarray,
    events: list[dict[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    active = np.stack([event["active_mask"] for event in events]).astype(bool)
    if scope == "all":
        selected = np.ones_like(active, dtype=bool)
    elif scope == "directly_occluded":
        selected = active
    elif scope == "indirectly_affected":
        selected = ~active
    else:
        raise ValueError(f"unknown alpha scope {scope!r}")
    values = np.asarray(alpha[selected], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"oracle alpha scope {scope!r} has invalid support")
    return {
        "scope": scope,
        "count": int(values.size),
        "mean": float(values.mean()),
        "quantiles": {
            str(quantile): float(np.quantile(values, quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "prior_endpoint_fraction": float(np.mean(values <= 0.05)),
        "measurement_endpoint_fraction": float(np.mean(values >= 0.95)),
    }


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    contract = _require_mapping(config.get("contract"), "contract")
    if contract.get("evaluation_scope") != "exposed_posthoc_development":
        raise ValueError("posterior headroom config must remain exposed development")
    if bool(contract.get("adapter_training_authorized", True)):
        raise ValueError("zero-training diagnostic cannot authorize adapter training")
    if contract.get("primary_anchor_scope") != "directly_occluded":
        raise ValueError("posterior headroom primary scope must be directly_occluded")
    if contract.get("primary_window") != NEXT_PRIOR_WINDOW:
        raise ValueError(f"posterior headroom primary window must be {NEXT_PRIOR_WINDOW}")
    if contract.get("writeback_scope") != "dropout_active_anchors_only":
        raise ValueError("posterior headroom writeback must only replace active anchors")

    data_path = _resolve(config["dataset"])
    checkpoint_path = _resolve(config["checkpoint"])
    observed_cache_path = _resolve(config["observed_correction_cache"])
    canonical_cache_path = _resolve(config["canonical_correction_cache"])
    device = choose_device(args.device)
    model, _, checkpoint = load_lewm_grounded_checkpoint(checkpoint_path, device=device)
    if not isinstance(model, AnchorPersistentLeWM):
        raise TypeError("checkpoint must contain AnchorPersistentLeWM")
    history_size = int(config["history_size"])
    frame_skip = int(config["frame_skip"])
    if history_size != model.history_size:
        raise ValueError("config history_size differs from checkpoint")
    if int(checkpoint["data_config"]["frame_skip"]) != frame_skip:
        raise ValueError("config frame_skip differs from checkpoint")

    required = {
        "rgb",
        "rgb_camera_only",
        "obs",
        "action",
        "task_name",
        "episode_id",
        "frame_id",
        "keypoints_xyz",
        "q_nominal",
        "valid_nominal",
        "anchor_dropout_active",
        "measurement_valid",
        "camera_step_frame",
    }
    with np.load(data_path, allow_pickle=False) as raw:
        missing = required.difference(raw.files)
        if missing:
            raise KeyError(f"dataset is missing fields: {sorted(missing)}")
        raw_count = int(raw["rgb"].shape[0])
        count = raw_count
        if args.max_frames > 0:
            count = min(count, int(args.max_frames))
        observed_images = _validate_images(raw["rgb"][:count], key="rgb", count=count)
        canonical_images = _validate_images(
            raw["rgb_camera_only"][:count], key="rgb_camera_only", count=count
        )
        obs = np.asarray(raw["obs"][:count], dtype=np.float32)
        action = np.asarray(raw["action"][:count], dtype=np.float32)
        task_name = raw["task_name"][:count].astype(str)
        episode_id = raw["episode_id"][:count].astype(str)
        frame_id = np.asarray(raw["frame_id"][:count], dtype=np.int64)
        keypoints_xyz = np.asarray(raw["keypoints_xyz"][:count], dtype=np.float32)
        q_nominal = np.asarray(raw["q_nominal"][:count], dtype=np.float32)
        valid_nominal = np.asarray(raw["valid_nominal"][:count], dtype=bool)
        dropout_active = np.asarray(raw["anchor_dropout_active"][:count], dtype=bool)
        measurement_valid = np.asarray(raw["measurement_valid"][:count], dtype=bool)
        camera_step_frame = np.asarray(raw["camera_step_frame"][:count], dtype=np.int64)
        if str(np.asarray(raw["observation_action_alignment"]).item()) != "pre_action":
            raise ValueError("dataset must use pre_action observation/action alignment")
    if q_nominal.shape[1] != model.num_keypoints:
        raise ValueError("dataset keypoint count differs from checkpoint")
    if np.any(dropout_active & measurement_valid):
        raise ValueError("dropout-active anchors must not be measurement-valid")

    observed_q, observed_metadata = _load_correction_cache(
        observed_cache_path,
        data_path=data_path,
        sample_count=raw_count,
        num_keypoints=model.num_keypoints,
        required_image_key=str(config["observed_image_key"]),
        required_view_role="deployment_observation",
        deployment_cache_eligible=True,
    )
    canonical_q, canonical_metadata = _load_correction_cache(
        canonical_cache_path,
        data_path=data_path,
        sample_count=raw_count,
        num_keypoints=model.num_keypoints,
        required_image_key=str(config["canonical_image_key"]),
        required_view_role="target_only",
        deployment_cache_eligible=False,
    )
    if observed_metadata.get("selected_image_sha256") != _image_sha256(observed_images):
        raise ValueError("observed correction cache image hash differs from rgb")
    if canonical_metadata.get("selected_image_sha256") != _image_sha256(canonical_images):
        raise ValueError("canonical correction cache image hash differs from rgb_camera_only")
    observed_q = np.array(observed_q[:count], dtype=np.float32, copy=True)
    canonical_q = np.array(canonical_q[:count], dtype=np.float32, copy=True)
    batch_size = int(config["batch_size"])
    print("encoding deployment observations", flush=True)
    measurement_embedding = _encode_frames(
        model,
        images=observed_images,
        obs=obs,
        q_nominal=q_nominal,
        q_corrected=observed_q,
        keypoints_xyz=keypoints_xyz,
        valid_nominal=valid_nominal,
        batch_size=batch_size,
        device=device,
    )
    print("encoding target-only canonical observations", flush=True)
    target_embedding = _encode_frames(
        model,
        images=canonical_images,
        obs=obs,
        q_nominal=q_nominal,
        q_corrected=canonical_q,
        keypoints_xyz=keypoints_xyz,
        valid_nominal=valid_nominal,
        batch_size=batch_size,
        device=device,
    )

    episodes: dict[tuple[str, str], np.ndarray] = {}
    for task in sorted(set(task_name.tolist())):
        for episode in sorted(set(episode_id[task_name == task].tolist())):
            indices = np.flatnonzero((task_name == task) & (episode_id == episode))
            indices = indices[np.argsort(frame_id[indices])]
            if indices.size > 1 and not np.all(np.diff(frame_id[indices]) == 1):
                raise ValueError(f"episode {(task, episode)} is not contiguous")
            episodes[(task, episode)] = indices

    events: list[dict[str, Any]] = []
    context_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for (task, episode), sequence in episodes.items():
        for position in range(history_size * frame_skip, len(sequence) - frame_skip):
            index = int(sequence[position])
            if not bool(np.any(dropout_active[index])):
                continue
            starts = [position - (history_size - offset) * frame_skip for offset in range(history_size)]
            context_indices = sequence[np.asarray(starts, dtype=np.int64)]
            event_frame = int(frame_id[index])
            step_frame = int(camera_step_frame[index])
            events.append(
                {
                    "task": task,
                    "episode_id": episode,
                    "frame_id": event_frame,
                    "phase": event_frame % frame_skip,
                    "index": index,
                    "next_index": int(sequence[position + frame_skip]),
                    "position": position,
                    "sequence": sequence,
                    "active_mask": dropout_active[index].copy(),
                    "history_crosses_camera_step": bool(
                        int(frame_id[context_indices[0]]) < step_frame <= event_frame
                    ),
                }
            )
            context_rows.append(measurement_embedding[context_indices])
            action_rows.append(_action_context(action, sequence, starts, frame_skip))
    if not events:
        raise ValueError("dataset contains no evaluable occlusion events")
    histories = np.stack(context_rows).astype(np.float32, copy=False)
    actions = np.stack(action_rows).astype(np.float32, copy=False)
    prior = _predict(
        model, histories, actions, device=device, batch_size=batch_size
    )
    event_indices = np.asarray([event["index"] for event in events], dtype=np.int64)
    next_indices = np.asarray([event["next_index"] for event in events], dtype=np.int64)
    measurement = measurement_embedding[event_indices]
    target = target_embedding[event_indices]
    hold_last = histories[:, -1]
    constant_velocity = histories[:, -1] + histories[:, -1] - histories[:, -2]
    fixed_alphas = [float(value) for value in config["fixed_alphas"]]
    methods, oracle_alpha = _fusion_candidates(
        prior=prior,
        measurement=measurement,
        hold_last=hold_last,
        constant_velocity=constant_velocity,
        target=target,
        fixed_alphas=fixed_alphas,
    )
    frame_records = _frame_records(methods, target, events, window=MASKED_WINDOW)

    next_target = target_embedding[next_indices]
    next_methods: dict[str, np.ndarray] = {}
    active_masks = np.stack([event["active_mask"] for event in events]).astype(
        np.bool_, copy=False
    )
    for method, posterior in methods.items():
        next_histories = np.empty_like(histories)
        next_actions = np.empty_like(actions)
        for event_index, event in enumerate(events):
            position = int(event["position"])
            sequence = np.asarray(event["sequence"], dtype=np.int64)
            starts = [
                position - (history_size - 1 - offset) * frame_skip
                for offset in range(history_size)
            ]
            history_indices = sequence[np.asarray(starts, dtype=np.int64)]
            next_histories[event_index] = measurement_embedding[history_indices]
            next_histories[event_index, -1] = _selective_writeback(
                posterior[event_index : event_index + 1],
                measurement[event_index : event_index + 1],
                active_masks[event_index : event_index + 1],
            )[0]
            next_actions[event_index] = _action_context(
                action, sequence, starts, frame_skip
            )
        next_methods[method] = _predict(
            model,
            next_histories,
            next_actions,
            device=device,
            batch_size=batch_size,
        )
    frame_records.extend(
        _frame_records(
            next_methods, next_target, events, window=NEXT_PRIOR_WINDOW
        )
    )
    episode_records = _episode_reduce(frame_records)
    repetitions = int(config["bootstrap_repetitions"])
    seed = int(config["seed"])
    summaries = {
        window: {
            scope: {
                "vs_lewm_prior": _summarize_window(
                    episode_records,
                    window=window,
                    baseline="lewm_prior",
                    repetitions=repetitions,
                    seed=seed,
                    anchor_scope=scope,
                ),
                "vs_measurement": _summarize_window(
                    episode_records,
                    window=window,
                    baseline="measurement",
                    repetitions=repetitions,
                    seed=seed,
                    anchor_scope=scope,
                ),
            }
            for scope in ("all", "directly_occluded", "indirectly_affected")
        }
        for window in (MASKED_WINDOW, NEXT_PRIOR_WINDOW)
    }
    alpha_summaries = {
        scope: _alpha_summary(oracle_alpha, events, scope=scope)
        for scope in ("all", "directly_occluded", "indirectly_affected")
    }

    result = {
        "schema_version": 2,
        "evaluation_type": "anchor_persistent_zero_training_posterior_headroom",
        "evaluation_scope": "exposed_posthoc_development",
        "formal_claim_allowed": False,
        "adapter_training_authorized": False,
        "measurement_valid_role": "annotation_only_not_model_input",
        "primary_estimand": {
            "anchor_scope": "directly_occluded",
            "window": NEXT_PRIOR_WINDOW,
            "statistical_unit": "physical_episode",
        },
        "writeback_scope": "dropout_active_anchors_only",
        "superseded_outputs": list(contract.get("superseded_outputs", [])),
        "interpretation": (
            "Target-only canonical latent upper-bound diagnostic. Existing five-frame "
            "occlusion supports five phase-stream masked events and one subsequent "
            "pre-measurement prior per phase, not "
            "multi-step posterior recovery."
        ),
        "execution_code": _execution_manifest(),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "dataset": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "observed_correction_cache": str(observed_cache_path),
        "observed_correction_metadata_sha256": sha256_file(
            observed_cache_path / "metadata.json"
        ),
        "observed_q_corrected_sha256": observed_metadata["q_corrected_sha256"],
        "canonical_correction_cache": str(canonical_cache_path),
        "canonical_correction_metadata_sha256": sha256_file(
            canonical_cache_path / "metadata.json"
        ),
        "canonical_q_corrected_sha256": canonical_metadata["q_corrected_sha256"],
        "frame_identity_sha256": frame_identity_sha256(episode_id, frame_id),
        "event_count": len(events),
        "physical_episode_count": len(episodes),
        "phase_counts": {
            str(phase): sum(event["phase"] == phase for event in events)
            for phase in range(frame_skip)
        },
        "history_crosses_camera_step_count": sum(
            event["history_crosses_camera_step"] for event in events
        ),
        "methods": sorted(methods),
        "oracle_alpha": alpha_summaries,
        "summaries": summaries,
        "episode_records": episode_records,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"refusing to overwrite different output: {args.output}")
    args.output.write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "event_count": len(events),
                "physical_episode_count": len(episodes),
                "execution_code_sha256": result["execution_code"]["aggregate_sha256"],
                "direct_oracle_alpha": alpha_summaries["directly_occluded"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
