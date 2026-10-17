"""Horizon-pair data for the oracle latent-to-policy-token ceiling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from fcpg.lewm_policy_token_bridge import (
    EXPECTED_ANCHORS,
    EXPECTED_POLICY_VISUAL_DIM,
    EXPECTED_STATE_DIM,
    reconstruct_native_policy_tokens,
    sha256_file,
)


PROTOCOL = "lewm_horizon_oracle_policy_token_bridge_ceiling_v1"


def build_horizon_pairs(
    task_name: np.ndarray,
    episode_id: np.ndarray,
    frame_id: np.ndarray,
    *,
    tasks: Sequence[str],
    horizons: Sequence[int],
    raw_steps_per_decision: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    task_name = np.asarray(task_name).astype(str)
    episode_id = np.asarray(episode_id)
    frame_id = np.asarray(frame_id, dtype=np.int64)
    if not (task_name.shape == episode_id.shape == frame_id.shape):
        raise ValueError("raw task, episode, and frame arrays must align")
    tasks = tuple(str(task) for task in tasks)
    horizons = tuple(int(horizon) for horizon in horizons)
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be non-empty and unique")
    if not horizons or len(horizons) != len(set(horizons)) or min(horizons) <= 0:
        raise ValueError("horizons must be unique positive integers")
    if raw_steps_per_decision <= 0:
        raise ValueError("raw_steps_per_decision must be positive")

    starts: list[int] = []
    targets: list[int] = []
    pair_horizons: list[int] = []
    pair_tasks: list[str] = []
    for task in tasks:
        task_rows = np.flatnonzero(task_name == task)
        if task_rows.size == 0:
            raise ValueError(f"raw source is missing task {task!r}")
        for episode in np.unique(episode_id[task_rows]):
            rows = task_rows[episode_id[task_rows] == episode]
            frames = frame_id[rows]
            if len(np.unique(frames)) != rows.size:
                raise ValueError("episode contains duplicate frame ids")
            frame_to_row = {int(frame): int(row) for frame, row in zip(frames, rows)}
            for horizon in horizons:
                offset = horizon * raw_steps_per_decision
                for frame in sorted(frame_to_row):
                    target_frame = frame + offset
                    if target_frame not in frame_to_row:
                        continue
                    starts.append(frame_to_row[frame])
                    targets.append(frame_to_row[target_frame])
                    pair_horizons.append(horizon)
                    pair_tasks.append(task)
    if not starts:
        raise ValueError("horizon construction produced no pairs")
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(pair_horizons, dtype=np.int64),
        np.asarray(pair_tasks),
    )


@dataclass(frozen=True)
class HorizonBridgeSourceSplit:
    raw_embedding: np.ndarray
    raw_sampled_visual: np.ndarray
    raw_xyz: np.ndarray
    raw_task_name: np.ndarray
    raw_episode_id: np.ndarray
    raw_frame_id: np.ndarray
    start_index: np.ndarray
    target_index: np.ndarray
    horizon: np.ndarray
    pair_task_name: np.ndarray
    pair_physical_episode_sha256: np.ndarray
    provenance: Mapping[str, Any]

    @property
    def pair_count(self) -> int:
        return int(self.start_index.shape[0])


@dataclass(frozen=True)
class HorizonBridgeAlignedSplit:
    raw_embedding: np.ndarray
    raw_token: np.ndarray
    raw_xyz: np.ndarray
    start_index: np.ndarray
    target_index: np.ndarray
    horizon: np.ndarray
    pair_task_name: np.ndarray
    pair_physical_episode_sha256: np.ndarray
    geometry_hold_exact_token: np.ndarray
    provenance: Mapping[str, Any]

    @property
    def pair_count(self) -> int:
        return int(self.start_index.shape[0])


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return value


def load_horizon_bridge_source_split(
    posterior_cache: str | Path,
    policy_visual_cache: str | Path,
    *,
    tasks: Sequence[str],
    horizons: Sequence[int] = tuple(range(1, 9)),
    raw_steps_per_decision: int = 5,
) -> HorizonBridgeSourceSplit:
    posterior_cache = Path(posterior_cache).expanduser().resolve()
    policy_visual_cache = Path(policy_visual_cache).expanduser().resolve()
    deployment_path = posterior_cache / "deployment_inputs.npz"
    evaluation_path = posterior_cache / "evaluation_metadata.npz"
    posterior_metadata_path = posterior_cache / "metadata.json"
    policy_metadata_path = policy_visual_cache / "metadata.json"
    sampled_path = policy_visual_cache / "sampled_visual.npy"
    for path in (
        deployment_path,
        evaluation_path,
        posterior_metadata_path,
        policy_metadata_path,
        sampled_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"horizon bridge artifact is missing: {path}")
    posterior_metadata = _read_json(posterior_metadata_path)
    policy_metadata = _read_json(policy_metadata_path)
    inputs = posterior_metadata.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("dataset"), Mapping):
        raise TypeError("posterior metadata is missing source dataset provenance")
    source_path = Path(str(inputs["dataset"]["path"])).expanduser().resolve()
    if Path(str(policy_metadata.get("source_data", ""))).expanduser().resolve() != source_path:
        raise ValueError("policy visual and posterior caches use different raw data")
    corrector_path = Path(str(policy_metadata.get("checkpoint", ""))).expanduser().resolve()
    corrector_sha = sha256_file(corrector_path)
    canonical = inputs.get("canonical_correction_cache")
    if not isinstance(canonical, Mapping) or corrector_sha != str(
        canonical.get("corrector_checkpoint_sha256", "")
    ):
        raise ValueError("policy visual cache corrector provenance is incompatible")

    with np.load(deployment_path, allow_pickle=False) as deployment, np.load(
        evaluation_path, allow_pickle=False
    ) as evaluation, np.load(source_path, allow_pickle=False) as source:
        if "measurement_frame_embedding" not in deployment:
            raise KeyError("posterior cache lacks measurement_frame_embedding")
        required_source = {"task_name", "episode_id", "frame_id", "env_seed", "keypoints_xyz"}
        absent = sorted(required_source - set(source.files))
        if absent:
            raise KeyError(f"raw source is missing {absent}")
        raw_embedding = np.asarray(
            deployment["measurement_frame_embedding"], dtype=np.float32
        )
        raw_task = np.asarray(source["task_name"]).astype(str)
        raw_episode = np.asarray(source["episode_id"])
        raw_frame = np.asarray(source["frame_id"], dtype=np.int64)
        raw_env_seed = np.asarray(source["env_seed"], dtype=np.int64)
        raw_xyz = np.asarray(source["keypoints_xyz"], dtype=np.float32)
        raw_count = raw_task.shape[0]
        if raw_embedding.shape != (raw_count, EXPECTED_ANCHORS, EXPECTED_STATE_DIM):
            raise ValueError("raw measurement embedding must have shape [M,3,192]")
        if raw_xyz.shape != (raw_count, EXPECTED_ANCHORS, 3):
            raise ValueError("raw XYZ must have shape [M,3,3]")
        sampled = np.load(sampled_path, mmap_mode="r")
        if sampled.shape != (
            raw_count,
            EXPECTED_ANCHORS,
            EXPECTED_POLICY_VISUAL_DIM,
        ):
            raise ValueError("raw sampled policy visual cache is misaligned")
        start, target, pair_horizon, pair_task = build_horizon_pairs(
            raw_task,
            raw_episode,
            raw_frame,
            tasks=tasks,
            horizons=horizons,
            raw_steps_per_decision=raw_steps_per_decision,
        )
        if not np.array_equal(raw_task[start], pair_task) or not np.array_equal(
            raw_task[target], pair_task
        ):
            raise ValueError("horizon pair crosses task identity")
        if not np.array_equal(raw_episode[start], raw_episode[target]):
            raise ValueError("horizon pair crosses episode identity")
        if not np.array_equal(
            raw_frame[target] - raw_frame[start],
            pair_horizon * raw_steps_per_decision,
        ):
            raise ValueError("horizon pair has an invalid frame offset")
        source_sha = str(inputs["dataset"].get("sha256", ""))
        episode_identity: dict[tuple[str, int, int], str] = {}
        for row in np.unique(start):
            key = (
                str(raw_task[row]),
                int(raw_episode[row]),
                int(raw_env_seed[row]),
            )
            if key not in episode_identity:
                episode_identity[key] = hashlib.sha256(
                    f"{source_sha}|{key[0]}|{key[1]}|{key[2]}".encode()
                ).hexdigest()
        pair_physical = np.asarray(
            [
                episode_identity[
                    (
                        str(raw_task[row]),
                        int(raw_episode[row]),
                        int(raw_env_seed[row]),
                    )
                ]
                for row in start
            ]
        )
        raw_sampled = np.array(sampled, dtype=np.float32, copy=True)

    for label, array in (
        ("raw embedding", raw_embedding),
        ("raw sampled visual", raw_sampled),
        ("raw xyz", raw_xyz),
    ):
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"{label} contains non-finite values")
    counts = {
        str(horizon): int(np.sum(pair_horizon == horizon))
        for horizon in sorted(set(pair_horizon.tolist()))
    }
    provenance = {
        "protocol": PROTOCOL,
        "role": "privileged_horizon_interface_ceiling",
        "posterior_cache": str(posterior_cache),
        "posterior_deployment_sha256": sha256_file(deployment_path),
        "posterior_evaluation_sha256": sha256_file(evaluation_path),
        "posterior_metadata_sha256": sha256_file(posterior_metadata_path),
        "source_data": str(source_path),
        "source_data_sha256": str(inputs["dataset"].get("sha256", "")),
        "policy_visual_cache": str(policy_visual_cache),
        "policy_visual_sampled_sha256": sha256_file(sampled_path),
        "policy_corrector_checkpoint": str(corrector_path),
        "policy_corrector_checkpoint_sha256": corrector_sha,
        "tasks": list(tasks),
        "horizons": list(horizons),
        "raw_steps_per_decision": raw_steps_per_decision,
        "raw_frame_count": int(raw_embedding.shape[0]),
        "pair_count": int(start.shape[0]),
        "pair_count_by_horizon": counts,
        "physical_episode_count": len(set(pair_physical.tolist())),
        "embedding_source": "measurement_frame_embedding_at_target_raw_frame",
    }
    return HorizonBridgeSourceSplit(
        raw_embedding=np.array(raw_embedding, copy=True),
        raw_sampled_visual=raw_sampled,
        raw_xyz=np.array(raw_xyz, copy=True),
        raw_task_name=np.array(raw_task, copy=True),
        raw_episode_id=np.array(raw_episode, copy=True),
        raw_frame_id=np.array(raw_frame, copy=True),
        start_index=start,
        target_index=target,
        horizon=pair_horizon,
        pair_task_name=pair_task,
        pair_physical_episode_sha256=pair_physical,
        provenance=provenance,
    )


def assert_disjoint_horizon_episodes(
    train: HorizonBridgeSourceSplit | HorizonBridgeAlignedSplit,
    validation: HorizonBridgeSourceSplit | HorizonBridgeAlignedSplit,
) -> None:
    overlap = set(train.pair_physical_episode_sha256.tolist()) & set(
        validation.pair_physical_episode_sha256.tolist()
    )
    if overlap:
        raise ValueError("horizon train and validation physical episodes overlap")


def reconstruct_horizon_policy_tokens(
    policy: nn.Module,
    split: HorizonBridgeSourceSplit,
    *,
    device: torch.device,
    batch_size: int,
) -> HorizonBridgeAlignedSplit:
    raw_token = reconstruct_native_policy_tokens(
        policy,
        split.raw_sampled_visual,
        split.raw_xyz,
        device=device,
        batch_size=batch_size,
    )
    geometry_hold = reconstruct_native_policy_tokens(
        policy,
        split.raw_sampled_visual[split.start_index],
        split.raw_xyz[split.target_index],
        device=device,
        batch_size=batch_size,
    )
    provenance = dict(split.provenance)
    provenance["token_contract"] = "frozen_native_policy_grounding_adapter_rho_one"
    provenance["geometry_hold_exact"] = (
        "start_visual_plus_target_xyz_through_frozen_policy_adapter"
    )
    return HorizonBridgeAlignedSplit(
        raw_embedding=np.array(split.raw_embedding, copy=True),
        raw_token=raw_token,
        raw_xyz=np.array(split.raw_xyz, copy=True),
        start_index=np.array(split.start_index, copy=True),
        target_index=np.array(split.target_index, copy=True),
        horizon=np.array(split.horizon, copy=True),
        pair_task_name=np.array(split.pair_task_name, copy=True),
        pair_physical_episode_sha256=np.array(
            split.pair_physical_episode_sha256, copy=True
        ),
        geometry_hold_exact_token=geometry_hold,
        provenance=provenance,
    )


def save_horizon_aligned_split(
    path: str | Path, split: HorizonBridgeAlignedSplit
) -> None:
    np.savez_compressed(
        Path(path),
        schema_version=np.asarray(1, dtype=np.int64),
        protocol=np.asarray(PROTOCOL),
        raw_embedding=np.asarray(split.raw_embedding, dtype=np.float32),
        raw_token=np.asarray(split.raw_token, dtype=np.float32),
        raw_xyz=np.asarray(split.raw_xyz, dtype=np.float32),
        start_index=np.asarray(split.start_index, dtype=np.int64),
        target_index=np.asarray(split.target_index, dtype=np.int64),
        horizon=np.asarray(split.horizon, dtype=np.int64),
        pair_task_name=np.asarray(split.pair_task_name),
        pair_physical_episode_sha256=np.asarray(
            split.pair_physical_episode_sha256
        ),
        geometry_hold_exact_token=np.asarray(
            split.geometry_hold_exact_token, dtype=np.float32
        ),
        provenance_json=np.asarray(json.dumps(split.provenance, sort_keys=True)),
    )


__all__ = [
    "HorizonBridgeAlignedSplit",
    "HorizonBridgeSourceSplit",
    "PROTOCOL",
    "assert_disjoint_horizon_episodes",
    "build_horizon_pairs",
    "load_horizon_bridge_source_split",
    "reconstruct_horizon_policy_tokens",
    "save_horizon_aligned_split",
]
