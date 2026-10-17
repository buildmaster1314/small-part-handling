"""Convert strict stacked A3/P1 replay output into formal bridge splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from fcpg.lewm_a3_p1_bridge_collection import (
    BRANCH_COUNT,
    HORIZON,
    PICK_TASK,
    STRICT_A3_10K_SEEDS,
    validate_bridge_arrays,
)
from fcpg.lewm_formal_policy_bridge import (
    FORMAL_BRIDGE_ARRAY_FILE,
    FORMAL_BRIDGE_CACHE_TYPE,
    FORMAL_BRIDGE_HORIZONS,
    FORMAL_BRIDGE_METADATA_FILE,
    FORMAL_BRIDGE_SEMANTICS,
    load_formal_bridge_split,
)
from fcpg.lewm_known_drop_bridge_runtime import geometry_hold_policy_token
from fcpg.lewm_policy_token_bridge import sha256_file


STACKED_COLLECTION_PROTOCOL: Final = "strict-a3-10k-to-pick-p1-physical-bridge-v1"


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def physical_episode_digest(task_name: str, source_episode_id: str) -> str:
    """Build a stable task-qualified physical episode identity."""

    if not isinstance(task_name, str) or not task_name:
        raise ValueError("task_name must be non-empty")
    if not isinstance(source_episode_id, str) or not source_episode_id:
        raise ValueError("source_episode_id must be non-empty")
    payload = json.dumps(
        [task_name, source_episode_id], ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_stacked_collection_metadata(
    metadata: Mapping[str, Any],
    *,
    source_path: str | Path,
) -> dict[str, Any]:
    """Validate the full, non-smoke collector receipt and model provenance."""

    values = _mapping(metadata, label="stacked collection metadata")
    if values.get("schema_version") != 1:
        raise ValueError("stacked collection schema_version must be 1")
    if values.get("protocol") != STACKED_COLLECTION_PROTOCOL:
        raise ValueError("stacked collection protocol mismatch")
    if values.get("smoke") is not False:
        raise ValueError("formal conversion refuses smoke collection")
    if values.get("task") != PICK_TASK:
        raise ValueError("formal converter currently requires strict Pick collection")
    required_flags = (
        "hidden_clean_targets_are_offline_labels_only",
        "source_action_byte_equality_checked",
        "p1_adapter_token_reconstruction_checked",
        "a3_history_and_target_reconstruction_checked",
    )
    if any(values.get(flag) is not True for flag in required_flags):
        raise ValueError("stacked collection did not pass all strict replay checks")

    source = Path(source_path).expanduser().resolve()
    recorded_output = Path(str(values.get("output", ""))).expanduser().resolve()
    if source != recorded_output:
        raise ValueError("stacked source path differs from collector receipt")
    output_sha = str(values.get("output_sha256", ""))
    if sha256_file(source) != output_sha:
        raise ValueError("stacked source SHA-256 differs from collector receipt")

    p1 = _mapping(values.get("p1"), label="stacked P1 provenance")
    required_p1 = {
        "policy_path",
        "policy_sha256",
        "corrector_path",
        "corrector_sha256",
        "obs_horizon",
        "history_alignment",
    }
    if set(p1) != required_p1:
        raise ValueError("stacked P1 provenance keys are not exact")
    policy = Path(str(p1["policy_path"])).expanduser().resolve()
    corrector = Path(str(p1["corrector_path"])).expanduser().resolve()
    for path, digest, label in (
        (policy, str(p1["policy_sha256"]), "P1 policy"),
        (corrector, str(p1["corrector_sha256"]), "P1 corrector"),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"stacked {label} provenance mismatch")
    if p1["obs_horizon"] != 2 or p1["history_alignment"] != "raw-adjacent-pre-action-[t-1,t]":
        raise ValueError("stacked P1 clock contract mismatch")

    predictors = values.get("predictors")
    if isinstance(predictors, (str, bytes)) or not isinstance(predictors, Sequence):
        raise TypeError("stacked predictors must be an ordered sequence")
    if len(predictors) != 3:
        raise ValueError("stacked collection must bind three A3 predictors")
    canonical_predictors: list[dict[str, Any]] = []
    for expected_seed, raw in zip(STRICT_A3_10K_SEEDS, predictors, strict=True):
        item = _mapping(raw, label=f"A3 predictor seed {expected_seed}")
        if int(item.get("seed", -1)) != expected_seed:
            raise ValueError("stacked A3 predictor order differs from prediction axis")
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        digest = str(item.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"stacked A3 checkpoint provenance mismatch for seed {expected_seed}")
        canonical_predictors.append(
            {"seed": expected_seed, "path": str(path), "sha256": digest}
        )

    selection = _mapping(values.get("selection"), label="stacked selection")
    if selection.get("branch_indices") != list(range(BRANCH_COUNT)):
        raise ValueError("formal conversion requires all eight branches in order")
    if selection.get("max_instances") is not None or selection.get("max_rows") is not None:
        raise ValueError("formal conversion refuses truncated row/instance collection")
    source_record = _mapping(values.get("source"), label="stacked source provenance")
    source_data_sha = str(source_record.get("sha256", ""))
    if len(source_data_sha) != 64:
        raise ValueError("stacked source data digest is missing")
    return {
        "source_path": str(source),
        "source_sha256": output_sha,
        "source_data_sha256": source_data_sha,
        "p1_policy": {"path": str(policy), "sha256": str(p1["policy_sha256"])},
        "p1_corrector": {
            "path": str(corrector),
            "sha256": str(p1["corrector_sha256"]),
        },
        "predictors": canonical_predictors,
    }


def _repeat_row_state(value: np.ndarray, *, branches: int, horizons: int) -> np.ndarray:
    array = np.asarray(value)
    return np.broadcast_to(
        array[:, None, None],
        (array.shape[0], branches, horizons, *array.shape[1:]),
    ).reshape(array.shape[0] * branches * horizons, *array.shape[1:])


@torch.inference_mode()
def reconstruct_geometry_hold_tokens(
    policy: nn.Module,
    *,
    last_sampled_visual: np.ndarray,
    current_xyz: np.ndarray,
    device: str | torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run the frozen native P1 adapter on held visual and current FK."""

    visual = np.asarray(last_sampled_visual, dtype=np.float32)
    xyz = np.asarray(current_xyz, dtype=np.float32)
    if visual.ndim != 3 or visual.shape[1:] != (3, 256):
        raise ValueError("last_sampled_visual must have shape [N,3,256]")
    if xyz.shape != (visual.shape[0], 3, 3):
        raise ValueError("current_xyz must have shape [N,3,3]")
    if batch_size <= 0:
        raise ValueError("geometry batch_size must be positive")
    target = torch.device(device)
    policy.requires_grad_(False).eval()
    output: list[np.ndarray] = []
    for start in range(0, visual.shape[0], batch_size):
        stop = start + batch_size
        token = geometry_hold_policy_token(
            policy,
            torch.from_numpy(np.ascontiguousarray(visual[start:stop])).to(target),
            torch.from_numpy(np.ascontiguousarray(xyz[start:stop])).to(target),
        )
        output.append(token.float().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32, copy=False)


def flatten_stacked_bridge_split(
    arrays: Mapping[str, np.ndarray],
    *,
    source_rows: np.ndarray,
    policy: nn.Module,
    policy_device: str | torch.device,
    geometry_batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Flatten selected source rows while preserving independent seed inputs."""

    validate_bridge_arrays(arrays)
    selected = np.asarray(source_rows, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0 or len(set(selected.tolist())) != selected.size:
        raise ValueError("source_rows must be a non-empty unique vector")
    row_count = int(np.asarray(arrays["source_row_index"]).shape[0])
    if np.any((selected < 0) | (selected >= row_count)):
        raise IndexError("source_rows contains an out-of-range row")
    branches = int(np.asarray(arrays["branch_index"]).size)
    if branches != BRANCH_COUNT:
        raise ValueError("formal split requires all eight branches")
    horizons = HORIZON
    count = int(selected.size * branches * horizons)

    factual = np.asarray(arrays["factual_prediction"], dtype=np.float32)[:, selected]
    no_action = np.asarray(arrays["no_action_prediction"], dtype=np.float32)[:, selected]
    # [seed,row,branch,H,K,D] -> [row,branch,H,seed,K,D] -> [N,seed,K,D]
    factual = factual.transpose(1, 2, 3, 0, 4, 5).reshape(count, 3, 3, 192)
    no_action = no_action.transpose(1, 2, 3, 0, 4, 5).reshape(count, 3, 3, 192)
    oracle = np.asarray(arrays["hidden_clean_a3_target"], dtype=np.float32)[selected].reshape(
        count, 3, 192
    )
    target_token = np.asarray(arrays["hidden_clean_policy_token"], dtype=np.float32)[
        selected
    ].reshape(count, 3, 256)
    current_xyz = np.asarray(arrays["current_xyz"], dtype=np.float32)[selected].reshape(
        count, 3, 3
    )
    last_embedding = _repeat_row_state(
        np.asarray(arrays["history_embedding"], dtype=np.float32)[selected, -1],
        branches=branches,
        horizons=horizons,
    )
    last_token = _repeat_row_state(
        np.asarray(arrays["last_policy_token"], dtype=np.float32)[selected],
        branches=branches,
        horizons=horizons,
    )
    last_xyz = _repeat_row_state(
        np.asarray(arrays["last_xyz"], dtype=np.float32)[selected],
        branches=branches,
        horizons=horizons,
    )
    held_visual = _repeat_row_state(
        np.asarray(arrays["last_sampled_visual"], dtype=np.float32)[selected],
        branches=branches,
        horizons=horizons,
    )
    geometry = reconstruct_geometry_hold_tokens(
        policy,
        last_sampled_visual=held_visual,
        current_xyz=current_xyz,
        device=policy_device,
        batch_size=geometry_batch_size,
    )

    source_episode = np.asarray(arrays["source_episode_id"])[selected].astype("U")
    episode_hash_by_row = np.asarray(
        [physical_episode_digest(PICK_TASK, value) for value in source_episode],
        dtype="U64",
    )
    physical_episode = np.repeat(episode_hash_by_row, branches * horizons)
    task_name = np.full(count, PICK_TASK, dtype="U32")
    horizon = np.tile(np.arange(1, horizons + 1, dtype=np.int64), selected.size * branches)
    branch_index = np.tile(
        np.repeat(np.asarray(arrays["branch_index"], dtype=np.int64), horizons),
        selected.size,
    )
    expanded_source_row = np.repeat(
        np.asarray(arrays["source_row_index"], dtype=np.int64)[selected],
        branches * horizons,
    )
    expanded_source_episode = np.repeat(source_episode, branches * horizons)
    hazard_id = np.repeat(
        np.asarray(arrays["hazard_id"], dtype=np.int64)[selected],
        branches * horizons,
    )
    result = {
        "factual_prediction": np.ascontiguousarray(factual),
        "matched_no_action_prediction": np.ascontiguousarray(no_action),
        "target_oracle_embedding": np.ascontiguousarray(oracle),
        "last_valid_embedding": np.ascontiguousarray(last_embedding),
        "last_valid_token": np.ascontiguousarray(last_token),
        "target_token": np.ascontiguousarray(target_token),
        "geometry_hold_token": np.ascontiguousarray(geometry),
        "last_valid_xyz": np.ascontiguousarray(last_xyz),
        "current_xyz": np.ascontiguousarray(current_xyz),
        "task_name": task_name,
        "physical_episode_sha256": physical_episode,
        "horizon": horizon,
        # Audit-only arrays are covered by the samples.npz digest.
        "source_row_index": expanded_source_row,
        "source_episode_id": expanded_source_episode,
        "branch_index": branch_index,
        "hazard_id": hazard_id,
    }
    if set(np.unique(result["horizon"]).tolist()) != set(FORMAL_BRIDGE_HORIZONS):
        raise RuntimeError("flattened bridge split does not cover H1-H8")
    return result


def _split_source_rows(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    source_episode = np.asarray(arrays["source_episode_id"]).astype("U")
    source_split = np.asarray(arrays["split"]).astype("U")
    episode_to_split: dict[str, str] = {}
    for episode, split in zip(source_episode.tolist(), source_split.tolist(), strict=True):
        formal_split = "train" if split == "train" else "validation" if split == "development" else ""
        if not formal_split:
            raise ValueError(f"formal conversion refuses source split {split!r}")
        previous = episode_to_split.setdefault(episode, formal_split)
        if previous != formal_split:
            raise ValueError("one source_episode_id crosses train/development")
    result = {
        split: np.flatnonzero(
            np.asarray([episode_to_split[value] == split for value in source_episode])
        ).astype(np.int64)
        for split in ("train", "validation")
    }
    if any(indices.size == 0 for indices in result.values()):
        raise ValueError("formal conversion requires train and development episodes")
    train_episode = set(source_episode[result["train"]].tolist())
    validation_episode = set(source_episode[result["validation"]].tolist())
    if train_episode.intersection(validation_episode):
        raise RuntimeError("source_episode_id split leakage detected")
    return result


def write_formal_bridge_splits(
    *,
    stacked_source: str | Path,
    stacked_metadata: str | Path,
    output_root: str | Path,
    policy: nn.Module,
    policy_device: str | torch.device,
    collector_implementation: str | Path,
    geometry_batch_size: int = 1024,
) -> dict[str, Any]:
    """Write formal train/validation directories and reload both fail-closed."""

    source_path = Path(stacked_source).expanduser().resolve()
    metadata_path = Path(stacked_metadata).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"formal output root is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = validate_stacked_collection_metadata(
        _mapping(raw_metadata, label="stacked metadata"), source_path=source_path
    )
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_bridge_arrays(arrays)
    split_rows = _split_source_rows(arrays)
    collector_path = Path(collector_implementation).expanduser().resolve()
    if not collector_path.is_file():
        raise FileNotFoundError(f"collector implementation is missing: {collector_path}")

    formal_provenance = {
        "p1_policy": provenance["p1_policy"],
        "p1_corrector": provenance["p1_corrector"],
        "a3_replicates": [
            {
                "seed": item["seed"],
                "checkpoint": {"path": item["path"], "sha256": item["sha256"]},
                "factual_state_key": "factual_state",
                "matched_no_action_state_key": "no_action_state",
            }
            for item in provenance["predictors"]
        ],
        "collector": {
            "implementation_sha256": sha256_file(collector_path),
            "source_data_sha256": provenance["source_data_sha256"],
        },
    }
    report: dict[str, Any] = {
        "source": {"path": str(source_path), "sha256": provenance["source_sha256"]},
        "splits": {},
    }
    for split_name, rows in split_rows.items():
        split_dir = destination / split_name
        split_dir.mkdir()
        flattened = flatten_stacked_bridge_split(
            arrays,
            source_rows=rows,
            policy=policy,
            policy_device=policy_device,
            geometry_batch_size=geometry_batch_size,
        )
        array_path = split_dir / FORMAL_BRIDGE_ARRAY_FILE
        np.savez_compressed(array_path, **flattened)
        metadata = {
            "schema_version": 1,
            "cache_type": FORMAL_BRIDGE_CACHE_TYPE,
            "bridge_semantics": FORMAL_BRIDGE_SEMANTICS,
            "split": split_name,
            "arrays": {"path": FORMAL_BRIDGE_ARRAY_FILE, "sha256": sha256_file(array_path)},
            "contract": {
                "num_keypoints": 3,
                "latent_dim": 192,
                "token_dim": 256,
                "replicate_count": 3,
                "replicate_semantics": "correlated_single_seed_predictions",
                "horizons": list(FORMAL_BRIDGE_HORIZONS),
                "target_alignment": "future_pre_action",
            },
            "provenance": formal_provenance,
        }
        (split_dir / FORMAL_BRIDGE_METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        loaded = load_formal_bridge_split(split_dir)
        report["splits"][split_name] = {
            "source_row_count": int(rows.size),
            "flattened_row_count": loaded.sample_count,
            "physical_episode_count": len(set(loaded.physical_episode_sha256.tolist())),
            "arrays_sha256": loaded.metadata["arrays"]["sha256"],
        }
    train_episode = set(load_formal_bridge_split(destination / "train").physical_episode_sha256)
    validation_episode = set(
        load_formal_bridge_split(destination / "validation").physical_episode_sha256
    )
    if train_episode.intersection(validation_episode):
        raise RuntimeError("written formal splits share physical episodes")
    return report


__all__ = [
    "STACKED_COLLECTION_PROTOCOL",
    "flatten_stacked_bridge_split",
    "physical_episode_digest",
    "reconstruct_geometry_hold_tokens",
    "validate_stacked_collection_metadata",
    "write_formal_bridge_splits",
]
