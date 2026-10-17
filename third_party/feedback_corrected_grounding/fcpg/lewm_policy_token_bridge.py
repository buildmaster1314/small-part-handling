"""Offline interface pilot from Anchor-Persistent states to policy tokens.

This module deliberately addresses only representation compatibility.  It does
not train a LeWM transition, select measurements, or establish a control claim.
The target is the frozen action policy's native, pre-memory grounded token.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROTOCOL = "lewm_policy_token_bridge_interface_pilot_v1"
EXPECTED_FRAME_SKIP = 5
EXPECTED_ANCHORS = 3
EXPECTED_STATE_DIM = 192
EXPECTED_POLICY_VISUAL_DIM = 256
EXPECTED_POLICY_TOKEN_DIM = 256


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _require_finite(array: np.ndarray, *, label: str) -> None:
    if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} contains non-finite values")


@dataclass(frozen=True)
class BridgeSourceSplit:
    """Aligned LeWM inputs and frozen-policy target ingredients."""

    state: np.ndarray
    sampled_visual: np.ndarray
    keypoints_xyz: np.ndarray
    task_name: np.ndarray
    physical_episode_sha256: np.ndarray
    source_episode_id: np.ndarray
    source_frame_id: np.ndarray
    target_raw_index: np.ndarray
    provenance: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.state.shape[0])

    @property
    def num_keypoints(self) -> int:
        return int(self.state.shape[1])

    def geometry(self) -> np.ndarray:
        identity = np.broadcast_to(
            np.eye(self.num_keypoints, dtype=np.float32)[None],
            (self.sample_count, self.num_keypoints, self.num_keypoints),
        )
        return np.concatenate(
            [np.asarray(self.keypoints_xyz, dtype=np.float32), identity], axis=-1
        )


@dataclass(frozen=True)
class BridgeAlignedSplit:
    """Complete offline bridge examples after native target reconstruction."""

    state: np.ndarray
    geometry: np.ndarray
    target: np.ndarray
    task_name: np.ndarray
    physical_episode_sha256: np.ndarray
    source_episode_id: np.ndarray
    source_frame_id: np.ndarray
    target_raw_index: np.ndarray
    provenance: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.state.shape[0])


def load_bridge_source_split(
    posterior_cache: str | Path,
    policy_visual_cache: str | Path,
    *,
    tasks: Sequence[str],
    input_key: str = "teacher_forced_factual_prior",
) -> BridgeSourceSplit:
    """Load row-aligned prior states and clean policy-token ingredients.

    ``target_raw_index`` is the sole join key.  The join is checked against the
    source task identity and may not contain duplicate target rows.
    """

    posterior_cache = Path(posterior_cache).expanduser().resolve()
    policy_visual_cache = Path(policy_visual_cache).expanduser().resolve()
    deployment_path = posterior_cache / "deployment_inputs.npz"
    evaluation_path = posterior_cache / "evaluation_metadata.npz"
    posterior_metadata_path = posterior_cache / "metadata.json"
    policy_metadata_path = policy_visual_cache / "metadata.json"
    sampled_path = policy_visual_cache / "sampled_visual.npy"
    required = (
        deployment_path,
        evaluation_path,
        posterior_metadata_path,
        policy_metadata_path,
        sampled_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"bridge input artifact is missing: {missing[0]}")

    requested_tasks = tuple(str(task) for task in tasks)
    if not requested_tasks or len(set(requested_tasks)) != len(requested_tasks):
        raise ValueError("tasks must be a non-empty sequence without duplicates")

    posterior_metadata = _read_json(
        posterior_metadata_path, label="posterior cache metadata"
    )
    policy_metadata = _read_json(
        policy_metadata_path, label="policy visual cache metadata"
    )
    contract = posterior_metadata.get("contract")
    if not isinstance(contract, Mapping):
        raise TypeError("posterior cache metadata is missing its contract")
    if int(contract.get("frame_skip", -1)) != EXPECTED_FRAME_SKIP:
        raise ValueError(
            f"bridge pilot requires frame_skip={EXPECTED_FRAME_SKIP}"
        )
    if contract.get("observation_action_alignment") != "pre_action":
        raise ValueError("bridge pilot requires pre_action alignment")

    inputs = posterior_metadata.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("dataset"), Mapping):
        raise TypeError("posterior cache metadata is missing dataset provenance")
    source_data = Path(str(inputs["dataset"]["path"])).expanduser().resolve()
    if not source_data.is_file():
        raise FileNotFoundError(f"posterior source dataset is missing: {source_data}")
    if not _same_path(str(policy_metadata.get("source_data", "")), source_data):
        raise ValueError("policy visual cache and posterior cache use different data")
    if int(policy_metadata.get("sample_count", -1)) <= 0:
        raise ValueError("policy visual metadata has an invalid sample count")
    if int(policy_metadata.get("num_keypoints", -1)) != EXPECTED_ANCHORS:
        raise ValueError("policy visual cache must contain exactly three anchors")
    if int(policy_metadata.get("visual_dim", -1)) != EXPECTED_POLICY_VISUAL_DIM:
        raise ValueError("policy visual cache has an incompatible feature dimension")
    if (
        int(policy_metadata.get("image_width", -1)),
        int(policy_metadata.get("image_height", -1)),
    ) != (224, 224):
        raise ValueError("bridge pilot is fixed to 224x224 policy features")
    policy_corrector = Path(
        str(policy_metadata.get("checkpoint", ""))
    ).expanduser().resolve()
    if not policy_corrector.is_file():
        raise FileNotFoundError(
            f"policy visual corrector checkpoint is missing: {policy_corrector}"
        )
    canonical_cache = inputs.get("canonical_correction_cache")
    if not isinstance(canonical_cache, Mapping):
        raise TypeError(
            "posterior cache metadata is missing canonical correction provenance"
        )
    expected_corrector_sha256 = str(
        canonical_cache.get("corrector_checkpoint_sha256", "")
    )
    actual_corrector_sha256 = sha256_file(policy_corrector)
    if (
        len(expected_corrector_sha256) != 64
        or actual_corrector_sha256 != expected_corrector_sha256
    ):
        raise ValueError(
            "policy visual corrector does not match posterior canonical corrector"
        )

    with np.load(deployment_path, allow_pickle=False) as deployment, np.load(
        evaluation_path, allow_pickle=False
    ) as evaluation, np.load(source_data, allow_pickle=False) as source:
        if input_key not in deployment:
            raise KeyError(f"deployment cache has no input {input_key!r}")
        required_eval = {
            "task_name",
            "physical_episode_sha256",
            "source_episode_id",
            "source_frame_id",
            "target_raw_index",
        }
        absent_eval = sorted(required_eval - set(evaluation.files))
        if absent_eval:
            raise KeyError(f"evaluation metadata is missing {absent_eval}")
        for key in ("task_name", "keypoints_xyz"):
            if key not in source:
                raise KeyError(f"source dataset is missing {key!r}")

        state_all = np.asarray(deployment[input_key], dtype=np.float32)
        task_all = np.asarray(evaluation["task_name"]).astype(str)
        target_all = np.asarray(evaluation["target_raw_index"], dtype=np.int64)
        count = int(state_all.shape[0])
        if state_all.shape != (
            count,
            EXPECTED_ANCHORS,
            EXPECTED_STATE_DIM,
        ):
            raise ValueError(
                "bridge state must have shape [N,3,192], got "
                f"{state_all.shape}"
            )
        if task_all.shape != (count,) or target_all.shape != (count,):
            raise ValueError("posterior row metadata is not aligned with state")
        if len(np.unique(target_all)) != count:
            raise ValueError("posterior target_raw_index contains duplicate rows")

        sampled_store = np.load(sampled_path, mmap_mode="r")
        if sampled_store.shape != (
            int(policy_metadata["sample_count"]),
            EXPECTED_ANCHORS,
            EXPECTED_POLICY_VISUAL_DIM,
        ):
            raise ValueError("sampled policy visual cache shape disagrees with metadata")
        if source["keypoints_xyz"].shape[:2] != (
            sampled_store.shape[0],
            EXPECTED_ANCHORS,
        ):
            raise ValueError("source keypoint geometry is not aligned with policy cache")
        if np.any(target_all < 0) or np.any(target_all >= sampled_store.shape[0]):
            raise IndexError("posterior target_raw_index is outside the source cache")

        mask = np.isin(task_all, np.asarray(requested_tasks, dtype=str))
        observed_tasks = set(task_all[mask].tolist())
        if observed_tasks != set(requested_tasks):
            missing_tasks = sorted(set(requested_tasks) - observed_tasks)
            raise ValueError(f"bridge split is missing requested tasks: {missing_tasks}")
        selected_rows = np.flatnonzero(mask)
        target_index = np.array(target_all[selected_rows], dtype=np.int64, copy=True)
        source_tasks = np.asarray(source["task_name"])[target_index].astype(str)
        selected_tasks = np.array(task_all[selected_rows], dtype=str, copy=True)
        if not np.array_equal(source_tasks, selected_tasks):
            raise ValueError("target_raw_index joins a different source task")

        state = np.array(state_all[selected_rows], dtype=np.float32, copy=True)
        sampled = np.array(
            sampled_store[target_index], dtype=np.float32, copy=True
        )
        xyz = np.array(
            source["keypoints_xyz"][target_index], dtype=np.float32, copy=True
        )
        physical = np.array(
            evaluation["physical_episode_sha256"][selected_rows]
        ).astype(str)
        source_episode = np.array(
            evaluation["source_episode_id"][selected_rows]
        ).astype(str)
        source_frame = np.array(
            evaluation["source_frame_id"][selected_rows], dtype=np.int64, copy=True
        )

    for label, array in (
        ("bridge state", state),
        ("sampled policy visual", sampled),
        ("keypoint geometry", xyz),
    ):
        _require_finite(array, label=label)
    if not all(len(value) == 64 for value in physical.tolist()):
        raise ValueError("physical episode identity must be a SHA-256 string")

    provenance = {
        "protocol": PROTOCOL,
        "role": "interface_pilot_source_only",
        "posterior_cache": str(posterior_cache),
        "posterior_deployment_sha256": sha256_file(deployment_path),
        "posterior_evaluation_sha256": sha256_file(evaluation_path),
        "posterior_metadata_sha256": sha256_file(posterior_metadata_path),
        "source_data": str(source_data),
        "source_data_sha256": str(inputs["dataset"].get("sha256", "")),
        "policy_visual_cache": str(policy_visual_cache),
        "policy_visual_sampled_sha256": sha256_file(sampled_path),
        "policy_visual_metadata_sha256": sha256_file(policy_metadata_path),
        "policy_corrector_checkpoint": str(policy_corrector),
        "policy_corrector_checkpoint_sha256": actual_corrector_sha256,
        "input_key": input_key,
        "tasks": list(requested_tasks),
        "sample_count": int(state.shape[0]),
        "physical_episode_count": len(set(physical.tolist())),
        "frame_skip": EXPECTED_FRAME_SKIP,
        "observation_action_alignment": "pre_action",
    }
    return BridgeSourceSplit(
        state=state,
        sampled_visual=sampled,
        keypoints_xyz=xyz,
        task_name=selected_tasks,
        physical_episode_sha256=physical,
        source_episode_id=source_episode,
        source_frame_id=source_frame,
        target_raw_index=target_index,
        provenance=provenance,
    )


def assert_disjoint_physical_episodes(
    train: BridgeSourceSplit | BridgeAlignedSplit,
    validation: BridgeSourceSplit | BridgeAlignedSplit,
) -> None:
    overlap = set(train.physical_episode_sha256.tolist()) & set(
        validation.physical_episode_sha256.tolist()
    )
    if overlap:
        raise ValueError(
            "train and validation share physical episodes: "
            f"{sorted(overlap)[:3]}"
        )


@torch.inference_mode()
def reconstruct_native_policy_tokens(
    policy: nn.Module,
    sampled_visual: np.ndarray,
    keypoints_xyz: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run the actual frozen GroundedTokenAdapter on aligned clean features."""

    if batch_size <= 0:
        raise ValueError("target reconstruction batch_size must be positive")
    for name, expected in (
        ("num_keypoints", EXPECTED_ANCHORS),
        ("vision_dim", EXPECTED_POLICY_VISUAL_DIM),
        ("model_dim", EXPECTED_POLICY_TOKEN_DIM),
    ):
        if int(getattr(policy, name, -1)) != expected:
            raise ValueError(f"policy {name} is incompatible with bridge pilot")
    if not hasattr(policy, "grounding_adapter") or not hasattr(policy, "_proprio"):
        raise TypeError("policy does not expose its native grounding adapter")
    sampled_visual = np.asarray(sampled_visual, dtype=np.float32)
    keypoints_xyz = np.asarray(keypoints_xyz, dtype=np.float32)
    if sampled_visual.ndim != 3 or sampled_visual.shape[1:] != (
        EXPECTED_ANCHORS,
        EXPECTED_POLICY_VISUAL_DIM,
    ):
        raise ValueError("sampled policy visual input must have shape [N,3,256]")
    if keypoints_xyz.shape != (
        sampled_visual.shape[0],
        EXPECTED_ANCHORS,
        3,
    ):
        raise ValueError("keypoint geometry is not aligned with policy visual input")

    policy.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, sampled_visual.shape[0], batch_size):
        stop = min(start + batch_size, sampled_visual.shape[0])
        sampled = torch.from_numpy(sampled_visual[start:stop]).to(
            device=device, dtype=torch.float32
        )
        xyz = torch.from_numpy(keypoints_xyz[start:stop]).to(
            device=device, dtype=torch.float32
        )
        reliability = torch.ones(
            sampled.shape[:2], device=device, dtype=sampled.dtype
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            proprio = policy._proprio(xyz)
            target = policy.grounding_adapter.forward_sampled(
                sampled,
                proprio,
                reliability=reliability,
            )
        expected_shape = (
            stop - start,
            EXPECTED_ANCHORS,
            EXPECTED_POLICY_TOKEN_DIM,
        )
        if target.shape != expected_shape:
            raise ValueError(
                f"native policy target has shape {tuple(target.shape)}, "
                f"expected {expected_shape}"
            )
        if not bool(torch.isfinite(target).all()):
            raise ValueError("native policy target contains non-finite values")
        chunks.append(target.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def reconstruct_native_policy_targets(
    policy: nn.Module,
    split: BridgeSourceSplit,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> BridgeAlignedSplit:
    targets = reconstruct_native_policy_tokens(
        policy,
        split.sampled_visual,
        split.keypoints_xyz,
        device=device,
        batch_size=batch_size,
    )
    provenance = dict(split.provenance)
    provenance.update(
        {
            "target_contract": (
                "frozen_policy_grounding_adapter_forward_sampled_with_xyz_"
                "anchor_identity_and_reliability_one"
            ),
            "target_role": "native_pre_memory_grounded_policy_token",
            "target_dim": EXPECTED_POLICY_TOKEN_DIM,
        }
    )
    return BridgeAlignedSplit(
        state=np.array(split.state, copy=True),
        geometry=split.geometry(),
        target=targets,
        task_name=np.array(split.task_name, copy=True),
        physical_episode_sha256=np.array(
            split.physical_episode_sha256, copy=True
        ),
        source_episode_id=np.array(split.source_episode_id, copy=True),
        source_frame_id=np.array(split.source_frame_id, copy=True),
        target_raw_index=np.array(split.target_raw_index, copy=True),
        provenance=provenance,
    )


def save_aligned_split(path: str | Path, split: BridgeAlignedSplit) -> None:
    path = Path(path)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        protocol=np.asarray(PROTOCOL),
        state=np.asarray(split.state, dtype=np.float32),
        geometry=np.asarray(split.geometry, dtype=np.float32),
        target=np.asarray(split.target, dtype=np.float32),
        task_name=np.asarray(split.task_name),
        physical_episode_sha256=np.asarray(split.physical_episode_sha256),
        source_episode_id=np.asarray(split.source_episode_id),
        source_frame_id=np.asarray(split.source_frame_id, dtype=np.int64),
        target_raw_index=np.asarray(split.target_raw_index, dtype=np.int64),
        provenance_json=np.asarray(json.dumps(split.provenance, sort_keys=True)),
    )


def _moments(array: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.asarray(array, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.maximum(std, 1.0e-5)
    return (
        torch.from_numpy(mean.astype(np.float32)),
        torch.from_numpy(std.astype(np.float32)),
    )


class PolicyTokenBridge(nn.Module):
    """Anchor-shared MLP with train-only input/target normalization."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_keypoints: int = EXPECTED_ANCHORS,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, hidden_dim, num_keypoints) <= 0:
            raise ValueError("bridge dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_keypoints = int(num_keypoints)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.register_buffer(
            "input_mean", torch.zeros(self.num_keypoints, self.input_dim)
        )
        self.register_buffer(
            "input_std", torch.ones(self.num_keypoints, self.input_dim)
        )
        self.register_buffer(
            "target_mean", torch.zeros(self.num_keypoints, self.output_dim)
        )
        self.register_buffer(
            "target_std", torch.ones(self.num_keypoints, self.output_dim)
        )

    def set_normalization(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        if inputs.shape[1:] != (self.num_keypoints, self.input_dim):
            raise ValueError("bridge normalization input shape is incompatible")
        if targets.shape[1:] != (self.num_keypoints, self.output_dim):
            raise ValueError("bridge normalization target shape is incompatible")
        input_mean, input_std = _moments(inputs)
        target_mean, target_std = _moments(targets)
        self.input_mean.copy_(input_mean)
        self.input_std.copy_(input_std)
        self.target_mean.copy_(target_mean)
        self.target_std.copy_(target_std)

    def normalized_target(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.target_mean) / self.target_std

    def forward_normalized(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.num_keypoints, self.input_dim):
            raise ValueError("bridge input has an incompatible shape")
        normalized = (inputs - self.input_mean) / self.input_std
        return self.net(normalized)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        predicted = self.forward_normalized(inputs)
        return predicted * self.target_std + self.target_mean


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def matched_geometry_hidden_dim(
    *, target_parameter_count: int, output_dim: int, maximum: int = 4096
) -> int:
    if target_parameter_count <= 0 or output_dim <= 0:
        raise ValueError("parameter target and output dimension must be positive")
    # LayerNorm contributes 2*d parameters; the two affine layers contribute
    # h*(d + 1 + o) + o.  Check the two nearest integer solutions without
    # constructing thousands of temporary modules.
    input_dim = 6
    constant = 2 * input_dim + output_dim
    coefficient = input_dim + 1 + output_dim
    estimate = (target_parameter_count - constant) / coefficient
    candidates = {
        max(1, min(maximum, int(np.floor(estimate)))),
        max(1, min(maximum, int(np.ceil(estimate)))),
    }
    return min(
        candidates,
        key=lambda hidden: (
            abs(constant + coefficient * hidden - target_parameter_count),
            hidden,
        ),
    )


def token_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    task_name: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    tasks = np.asarray(task_name).astype(str)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("token metrics require matching [N,K,D] tensors")
    if tasks.shape != (prediction.shape[0],):
        raise ValueError("task labels are not aligned with token predictions")
    _require_finite(prediction, label="token prediction")
    _require_finite(target, label="token target")

    per_sample_mse = np.mean(np.square(prediction - target), axis=(1, 2))
    pred_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    cosine = np.sum(pred_flat * target_flat, axis=1) / np.maximum(
        np.linalg.norm(pred_flat, axis=1) * np.linalg.norm(target_flat, axis=1),
        1.0e-12,
    )

    def summary(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "samples": int(mask.sum()),
            "token_mse": float(per_sample_mse[mask].mean()),
            "cosine_similarity": float(cosine[mask].mean()),
            "cosine_distance": float((1.0 - cosine[mask]).mean()),
        }

    all_mask = np.ones(tasks.shape[0], dtype=bool)
    per_task = {task: summary(tasks == task) for task in sorted(set(tasks.tolist()))}
    result: dict[str, Any] = summary(all_mask)
    result["per_task"] = per_task
    result["task_macro_token_mse"] = float(
        np.mean([row["token_mse"] for row in per_task.values()])
    )
    result["task_macro_cosine_similarity"] = float(
        np.mean([row["cosine_similarity"] for row in per_task.values()])
    )
    return result


def bridge_training_loss(
    model: PolicyTokenBridge,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    cosine_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_normalized = model.forward_normalized(inputs)
    target_normalized = model.normalized_target(targets)
    normalized_mse = F.mse_loss(predicted_normalized, target_normalized)
    predicted = predicted_normalized * model.target_std + model.target_mean
    cosine_distance = 1.0 - F.cosine_similarity(
        predicted.flatten(1), targets.flatten(1), dim=-1
    ).mean()
    loss = normalized_mse + float(cosine_weight) * cosine_distance
    return loss, {
        "normalized_mse": normalized_mse.detach(),
        "cosine_distance": cosine_distance.detach(),
    }


__all__ = [
    "BridgeAlignedSplit",
    "BridgeSourceSplit",
    "EXPECTED_ANCHORS",
    "EXPECTED_POLICY_TOKEN_DIM",
    "EXPECTED_STATE_DIM",
    "PROTOCOL",
    "PolicyTokenBridge",
    "assert_disjoint_physical_episodes",
    "bridge_training_loss",
    "load_bridge_source_split",
    "matched_geometry_hidden_dim",
    "reconstruct_native_policy_tokens",
    "reconstruct_native_policy_targets",
    "save_aligned_split",
    "sha256_file",
    "token_metrics",
    "trainable_parameter_count",
]
