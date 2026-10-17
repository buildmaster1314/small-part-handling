"""Aligned data and geometry baseline for the residual policy-token pilot."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_policy_token_bridge import (
    EXPECTED_ANCHORS,
    EXPECTED_FRAME_SKIP,
    EXPECTED_POLICY_TOKEN_DIM,
    EXPECTED_POLICY_VISUAL_DIM,
    EXPECTED_STATE_DIM,
    reconstruct_native_policy_tokens,
    sha256_file,
)


PROTOCOL = "lewm_residual_policy_token_bridge_interface_pilot_v1"
EMBEDDING_SOURCES = ("teacher_forced_factual_prior", "oracle_current")


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _finite(array: np.ndarray, *, label: str) -> None:
    if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} contains non-finite values")


@dataclass(frozen=True)
class ResidualBridgeSourceSplit:
    predicted_embedding: np.ndarray
    last_valid_embedding: np.ndarray
    last_sampled_visual: np.ndarray
    target_sampled_visual: np.ndarray
    last_valid_xyz: np.ndarray
    current_xyz: np.ndarray
    task_name: np.ndarray
    physical_episode_sha256: np.ndarray
    source_episode_id: np.ndarray
    source_frame_id: np.ndarray
    history_raw_index: np.ndarray
    target_raw_index: np.ndarray
    provenance: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.predicted_embedding.shape[0])


@dataclass(frozen=True)
class ResidualBridgeAlignedSplit:
    predicted_embedding: np.ndarray
    last_valid_embedding: np.ndarray
    last_valid_token: np.ndarray
    geometry_hold_exact_token: np.ndarray
    target_token: np.ndarray
    last_valid_xyz: np.ndarray
    current_xyz: np.ndarray
    task_name: np.ndarray
    physical_episode_sha256: np.ndarray
    source_episode_id: np.ndarray
    source_frame_id: np.ndarray
    history_raw_index: np.ndarray
    target_raw_index: np.ndarray
    provenance: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.predicted_embedding.shape[0])


def load_residual_bridge_source_split(
    posterior_cache: str | Path,
    policy_visual_cache: str | Path,
    *,
    tasks: Sequence[str],
    embedding_source: str = "teacher_forced_factual_prior",
) -> ResidualBridgeSourceSplit:
    posterior_cache = Path(posterior_cache).expanduser().resolve()
    policy_visual_cache = Path(policy_visual_cache).expanduser().resolve()
    deployment_path = posterior_cache / "deployment_inputs.npz"
    offline_path = posterior_cache / "offline_labels.npz"
    evaluation_path = posterior_cache / "evaluation_metadata.npz"
    posterior_metadata_path = posterior_cache / "metadata.json"
    policy_metadata_path = policy_visual_cache / "metadata.json"
    sampled_path = policy_visual_cache / "sampled_visual.npy"
    required = (
        deployment_path,
        offline_path,
        evaluation_path,
        posterior_metadata_path,
        policy_metadata_path,
        sampled_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"residual bridge artifact is missing: {missing[0]}")
    tasks = tuple(str(task) for task in tasks)
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be non-empty and unique")
    if embedding_source not in EMBEDDING_SOURCES:
        raise ValueError(f"unknown embedding source {embedding_source!r}")

    posterior_metadata = _read_json(
        posterior_metadata_path, label="posterior metadata"
    )
    policy_metadata = _read_json(policy_metadata_path, label="policy metadata")
    contract = posterior_metadata.get("contract")
    if not isinstance(contract, Mapping):
        raise TypeError("posterior metadata is missing its contract")
    if int(contract.get("frame_skip", -1)) != EXPECTED_FRAME_SKIP:
        raise ValueError(f"residual pilot requires frame_skip={EXPECTED_FRAME_SKIP}")
    if contract.get("observation_action_alignment") != "pre_action":
        raise ValueError("residual pilot requires pre_action alignment")
    inputs = posterior_metadata.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("dataset"), Mapping):
        raise TypeError("posterior metadata is missing dataset provenance")
    source_path = Path(str(inputs["dataset"]["path"])).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source dataset is missing: {source_path}")
    policy_source = Path(str(policy_metadata.get("source_data", ""))).expanduser().resolve()
    if policy_source != source_path:
        raise ValueError("policy visual cache and posterior cache use different data")
    if (
        int(policy_metadata.get("sample_count", -1)) <= 0
        or int(policy_metadata.get("num_keypoints", -1)) != EXPECTED_ANCHORS
        or int(policy_metadata.get("visual_dim", -1)) != EXPECTED_POLICY_VISUAL_DIM
    ):
        raise ValueError("policy visual cache dimensions are incompatible")
    corrector_path = Path(str(policy_metadata.get("checkpoint", ""))).expanduser().resolve()
    if not corrector_path.is_file():
        raise FileNotFoundError(f"policy corrector is missing: {corrector_path}")
    canonical_cache = inputs.get("canonical_correction_cache")
    if not isinstance(canonical_cache, Mapping):
        raise TypeError("posterior metadata lacks canonical corrector provenance")
    corrector_sha256 = sha256_file(corrector_path)
    if corrector_sha256 != str(
        canonical_cache.get("corrector_checkpoint_sha256", "")
    ):
        raise ValueError("policy visual cache uses a different canonical corrector")

    with np.load(deployment_path, allow_pickle=False) as deployment, np.load(
        offline_path, allow_pickle=False
    ) as offline, np.load(evaluation_path, allow_pickle=False) as evaluation, np.load(
        source_path, allow_pickle=False
    ) as source:
        required_deployment = {"teacher_forced_factual_prior"}
        required_offline = {"canonical_measurement_history", "canonical_current"}
        required_evaluation = {
            "task_name",
            "physical_episode_sha256",
            "source_episode_id",
            "source_frame_id",
            "history_raw_indices",
            "target_raw_index",
        }
        if not required_deployment.issubset(deployment.files):
            raise KeyError("deployment cache lacks teacher-forced factual prior")
        if not required_offline.issubset(offline.files):
            raise KeyError("offline cache lacks canonical measurement history")
        absent = sorted(required_evaluation - set(evaluation.files))
        if absent:
            raise KeyError(f"evaluation metadata is missing {absent}")
        for key in ("task_name", "keypoints_xyz"):
            if key not in source:
                raise KeyError(f"source dataset is missing {key!r}")

        if embedding_source == "teacher_forced_factual_prior":
            predicted_all = np.asarray(
                deployment["teacher_forced_factual_prior"], dtype=np.float32
            )
        else:
            predicted_all = np.asarray(offline["canonical_current"], dtype=np.float32)
        canonical_history = np.asarray(
            offline["canonical_measurement_history"], dtype=np.float32
        )
        task_all = np.asarray(evaluation["task_name"]).astype(str)
        history_all = np.asarray(evaluation["history_raw_indices"], dtype=np.int64)
        target_all = np.asarray(evaluation["target_raw_index"], dtype=np.int64)
        count = predicted_all.shape[0]
        if predicted_all.shape != (count, EXPECTED_ANCHORS, EXPECTED_STATE_DIM):
            raise ValueError("predicted residual state must have shape [N,3,192]")
        if canonical_history.shape != (
            count,
            3,
            EXPECTED_ANCHORS,
            EXPECTED_STATE_DIM,
        ):
            raise ValueError("canonical measurement history must have shape [N,3,3,192]")
        if task_all.shape != (count,) or history_all.shape != (count, 3):
            raise ValueError("residual bridge row metadata is not aligned")
        if target_all.shape != (count,):
            raise ValueError("target_raw_index is not aligned")
        if len(np.unique(target_all)) != count:
            raise ValueError("target_raw_index contains duplicate rows")

        sampled_store = np.load(sampled_path, mmap_mode="r")
        source_count = int(policy_metadata["sample_count"])
        if sampled_store.shape != (
            source_count,
            EXPECTED_ANCHORS,
            EXPECTED_POLICY_VISUAL_DIM,
        ):
            raise ValueError("sampled visual shape disagrees with policy metadata")
        if source["keypoints_xyz"].shape[:2] != (source_count, EXPECTED_ANCHORS):
            raise ValueError("source keypoint geometry is not aligned")
        selected = np.flatnonzero(np.isin(task_all, np.asarray(tasks, dtype=str)))
        if set(task_all[selected].tolist()) != set(tasks):
            missing_tasks = sorted(set(tasks) - set(task_all[selected].tolist()))
            raise ValueError(f"residual split is missing tasks: {missing_tasks}")
        history_index = np.array(history_all[selected, -1], dtype=np.int64, copy=True)
        target_index = np.array(target_all[selected], dtype=np.int64, copy=True)
        if (
            np.any(history_index < 0)
            or np.any(target_index < 0)
            or np.any(history_index >= source_count)
            or np.any(target_index >= source_count)
        ):
            raise IndexError("residual bridge raw index is outside source data")
        selected_tasks = np.array(task_all[selected], dtype=str, copy=True)
        source_tasks = np.asarray(source["task_name"]).astype(str)
        if not np.array_equal(source_tasks[history_index], selected_tasks):
            raise ValueError("history raw index crosses task identity")
        if not np.array_equal(source_tasks[target_index], selected_tasks):
            raise ValueError("target raw index crosses task identity")
        source_episode_all = np.asarray(evaluation["source_episode_id"]).astype(str)
        if source_episode_all.shape != (source_count,):
            raise ValueError("source episode identity is not aligned to raw frames")
        if not np.array_equal(
            source_episode_all[history_index], source_episode_all[target_index]
        ):
            raise ValueError("history and target raw indices cross physical episodes")

        predicted = np.array(predicted_all[selected], dtype=np.float32, copy=True)
        last_embedding = np.array(
            canonical_history[selected, -1], dtype=np.float32, copy=True
        )
        last_visual = np.array(
            sampled_store[history_index], dtype=np.float32, copy=True
        )
        target_visual = np.array(
            sampled_store[target_index], dtype=np.float32, copy=True
        )
        xyz_all = np.asarray(source["keypoints_xyz"])
        last_xyz = np.array(xyz_all[history_index], dtype=np.float32, copy=True)
        current_xyz = np.array(xyz_all[target_index], dtype=np.float32, copy=True)
        physical = np.asarray(evaluation["physical_episode_sha256"])[selected].astype(str)
        source_episode = np.array(
            source_episode_all[target_index], dtype=str, copy=True
        )
        source_frame_all = np.asarray(evaluation["source_frame_id"], dtype=np.int64)
        if source_frame_all.shape != (source_count,):
            raise ValueError("source frame identity is not aligned to raw frames")
        source_frame = np.array(
            source_frame_all[target_index], dtype=np.int64, copy=True
        )

    for label, array in (
        ("predicted embedding", predicted),
        ("last valid embedding", last_embedding),
        ("last sampled visual", last_visual),
        ("target sampled visual", target_visual),
        ("last xyz", last_xyz),
        ("current xyz", current_xyz),
    ):
        _finite(array, label=label)
    if not all(len(value) == 64 for value in physical.tolist()):
        raise ValueError("physical episode identity must be a SHA-256 string")
    provenance = {
        "protocol": PROTOCOL,
        "role": "teacher_forced_residual_interface_source",
        "posterior_cache": str(posterior_cache),
        "posterior_deployment_sha256": sha256_file(deployment_path),
        "posterior_offline_sha256": sha256_file(offline_path),
        "posterior_evaluation_sha256": sha256_file(evaluation_path),
        "posterior_metadata_sha256": sha256_file(posterior_metadata_path),
        "source_data": str(source_path),
        "source_data_sha256": str(inputs["dataset"].get("sha256", "")),
        "policy_visual_cache": str(policy_visual_cache),
        "policy_visual_sampled_sha256": sha256_file(sampled_path),
        "policy_visual_metadata_sha256": sha256_file(policy_metadata_path),
        "policy_corrector_checkpoint": str(corrector_path),
        "policy_corrector_checkpoint_sha256": corrector_sha256,
        "sample_count": int(selected.shape[0]),
        "physical_episode_count": len(set(physical.tolist())),
        "tasks": list(tasks),
        "predicted_embedding": embedding_source,
        "privileged_oracle_embedding": embedding_source == "oracle_current",
        "last_embedding": "canonical_measurement_history_last",
        "last_token_raw_index": "history_raw_indices_last",
        "target_token_raw_index": "target_raw_index",
        "frame_skip": EXPECTED_FRAME_SKIP,
        "observation_action_alignment": "pre_action",
    }
    return ResidualBridgeSourceSplit(
        predicted_embedding=predicted,
        last_valid_embedding=last_embedding,
        last_sampled_visual=last_visual,
        target_sampled_visual=target_visual,
        last_valid_xyz=last_xyz,
        current_xyz=current_xyz,
        task_name=selected_tasks,
        physical_episode_sha256=physical,
        source_episode_id=source_episode,
        source_frame_id=source_frame,
        history_raw_index=history_index,
        target_raw_index=target_index,
        provenance=provenance,
    )


def assert_disjoint_residual_episodes(
    train: ResidualBridgeSourceSplit | ResidualBridgeAlignedSplit,
    validation: ResidualBridgeSourceSplit | ResidualBridgeAlignedSplit,
) -> None:
    overlap = set(train.physical_episode_sha256.tolist()) & set(
        validation.physical_episode_sha256.tolist()
    )
    if overlap:
        raise ValueError(f"residual train/validation episodes overlap: {sorted(overlap)[:3]}")


def reconstruct_residual_policy_tokens(
    policy: nn.Module,
    split: ResidualBridgeSourceSplit,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> ResidualBridgeAlignedSplit:
    last_token = reconstruct_native_policy_tokens(
        policy,
        split.last_sampled_visual,
        split.last_valid_xyz,
        device=device,
        batch_size=batch_size,
    )
    target_token = reconstruct_native_policy_tokens(
        policy,
        split.target_sampled_visual,
        split.current_xyz,
        device=device,
        batch_size=batch_size,
    )
    geometry_hold_exact_token = reconstruct_native_policy_tokens(
        policy,
        split.last_sampled_visual,
        split.current_xyz,
        device=device,
        batch_size=batch_size,
    )
    provenance = dict(split.provenance)
    provenance.update(
        {
            "token_contract": "frozen_native_policy_grounding_adapter_rho_one",
            "last_token_role": "exact_native_token_at_history_raw_indices_last",
            "geometry_hold_exact_role": (
                "last_sampled_visual_plus_current_xyz_through_frozen_adapter"
            ),
            "target_token_role": "exact_native_token_at_target_raw_index",
        }
    )
    return ResidualBridgeAlignedSplit(
        predicted_embedding=np.array(split.predicted_embedding, copy=True),
        last_valid_embedding=np.array(split.last_valid_embedding, copy=True),
        last_valid_token=last_token,
        geometry_hold_exact_token=geometry_hold_exact_token,
        target_token=target_token,
        last_valid_xyz=np.array(split.last_valid_xyz, copy=True),
        current_xyz=np.array(split.current_xyz, copy=True),
        task_name=np.array(split.task_name, copy=True),
        physical_episode_sha256=np.array(split.physical_episode_sha256, copy=True),
        source_episode_id=np.array(split.source_episode_id, copy=True),
        source_frame_id=np.array(split.source_frame_id, copy=True),
        history_raw_index=np.array(split.history_raw_index, copy=True),
        target_raw_index=np.array(split.target_raw_index, copy=True),
        provenance=provenance,
    )


def save_residual_aligned_split(
    path: str | Path, split: ResidualBridgeAlignedSplit
) -> None:
    np.savez_compressed(
        Path(path),
        schema_version=np.asarray(1, dtype=np.int64),
        protocol=np.asarray(PROTOCOL),
        predicted_embedding=np.asarray(split.predicted_embedding, dtype=np.float32),
        last_valid_embedding=np.asarray(split.last_valid_embedding, dtype=np.float32),
        last_valid_token=np.asarray(split.last_valid_token, dtype=np.float32),
        geometry_hold_exact_token=np.asarray(
            split.geometry_hold_exact_token, dtype=np.float32
        ),
        target_token=np.asarray(split.target_token, dtype=np.float32),
        last_valid_xyz=np.asarray(split.last_valid_xyz, dtype=np.float32),
        current_xyz=np.asarray(split.current_xyz, dtype=np.float32),
        task_name=np.asarray(split.task_name),
        physical_episode_sha256=np.asarray(split.physical_episode_sha256),
        source_episode_id=np.asarray(split.source_episode_id),
        source_frame_id=np.asarray(split.source_frame_id, dtype=np.int64),
        history_raw_index=np.asarray(split.history_raw_index, dtype=np.int64),
        target_raw_index=np.asarray(split.target_raw_index, dtype=np.int64),
        provenance_json=np.asarray(json.dumps(split.provenance, sort_keys=True)),
    )


def _vector_moments(array: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
    return (
        torch.from_numpy(values.mean(axis=0).astype(np.float32)),
        torch.from_numpy(np.maximum(values.std(axis=0), 1.0e-5).astype(np.float32)),
    )


class GeometryResidualPolicyBridge(nn.Module):
    """Parameter-matched geometry-only token-delta baseline."""

    def __init__(
        self,
        *,
        token_dim: int = EXPECTED_POLICY_TOKEN_DIM,
        num_keypoints: int = EXPECTED_ANCHORS,
        hidden_dim: int = 157,
    ) -> None:
        super().__init__()
        if min(token_dim, num_keypoints, hidden_dim) <= 0:
            raise ValueError("geometry bridge dimensions must be positive")
        self.token_dim = int(token_dim)
        self.num_keypoints = int(num_keypoints)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = 6 + self.num_keypoints
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.token_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer("last_xyz_mean", torch.zeros(3))
        self.register_buffer("last_xyz_std", torch.ones(3))
        self.register_buffer("xyz_delta_mean", torch.zeros(3))
        self.register_buffer("xyz_delta_std", torch.ones(3))
        self.register_buffer("token_error_std", torch.ones(self.token_dim))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_normalization(
        self,
        *,
        last_xyz: np.ndarray,
        current_xyz: np.ndarray,
        token_delta: np.ndarray,
    ) -> None:
        last_mean, last_std = _vector_moments(last_xyz)
        delta_mean, delta_std = _vector_moments(current_xyz - last_xyz)
        _, token_std = _vector_moments(token_delta)
        with torch.no_grad():
            self.last_xyz_mean.copy_(last_mean)
            self.last_xyz_std.copy_(last_std)
            self.xyz_delta_mean.copy_(delta_mean)
            self.xyz_delta_std.copy_(delta_std)
            self.token_error_std.copy_(token_std)

    def forward(
        self,
        last_token: torch.Tensor,
        current_xyz: torch.Tensor,
        last_xyz: torch.Tensor,
    ) -> torch.Tensor:
        expected_token = (*current_xyz.shape[:-1], self.token_dim)
        if current_xyz.shape != last_xyz.shape or current_xyz.shape[-2:] != (
            self.num_keypoints,
            3,
        ):
            raise ValueError("geometry bridge xyz input shape is incompatible")
        if last_token.shape != expected_token:
            raise ValueError("geometry bridge last token shape is incompatible")
        last_normalized = (last_xyz - self.last_xyz_mean) / self.last_xyz_std
        delta_normalized = (
            current_xyz - last_xyz - self.xyz_delta_mean
        ) / self.xyz_delta_std
        identity = torch.eye(
            self.num_keypoints, device=current_xyz.device, dtype=current_xyz.dtype
        )
        identity = identity.reshape(
            *(1 for _ in current_xyz.shape[:-2]),
            self.num_keypoints,
            self.num_keypoints,
        ).expand(*current_xyz.shape[:-2], -1, -1)
        features = torch.cat((last_normalized, delta_normalized, identity), dim=-1)
        delta = self.output_projection(F.gelu(self.input_projection(features)))
        return last_token + delta

    def standardized_token_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, *, beta: float = 0.1
    ) -> torch.Tensor:
        scale = self.token_error_std.to(
            device=prediction.device, dtype=prediction.dtype
        ).clamp_min(1.0e-6)
        return F.smooth_l1_loss(
            prediction / scale, target.detach() / scale, beta=float(beta)
        )


def matched_geometry_residual_hidden_dim(
    *, target_parameter_count: int, token_dim: int, num_keypoints: int
) -> int:
    input_dim = 6 + num_keypoints
    constant = token_dim
    coefficient = input_dim + 1 + token_dim
    estimate = (target_parameter_count - constant) / coefficient
    candidates = {max(1, int(np.floor(estimate))), max(1, int(np.ceil(estimate)))}
    return min(
        candidates,
        key=lambda hidden: (
            abs(constant + coefficient * hidden - target_parameter_count),
            hidden,
        ),
    )


__all__ = [
    "EMBEDDING_SOURCES",
    "GeometryResidualPolicyBridge",
    "PROTOCOL",
    "ResidualBridgeAlignedSplit",
    "ResidualBridgeSourceSplit",
    "assert_disjoint_residual_episodes",
    "load_residual_bridge_source_split",
    "matched_geometry_residual_hidden_dim",
    "reconstruct_residual_policy_tokens",
    "save_residual_aligned_split",
]
