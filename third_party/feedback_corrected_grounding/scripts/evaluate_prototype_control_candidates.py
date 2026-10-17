#!/usr/bin/env python3
"""Post-hoc control-headroom diagnostic for joint prototype candidates.

The experiment is deliberately training-free.  A frozen policy is evaluated on
the same two-frame physical snapshot while only the current frame's three
grounding points are changed.  The previous frame always uses the deployable
wide-prototype top-1 match.  ``control_q_true`` is used only to construct an
offline teacher action and oracle labels; it is never inserted into a
deployable policy batch or the prototype candidate generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.geometric_baselines import (  # noqa: E402
    generate_joint_prototype_hypotheses,
)
from fcpg.observations import select_policy_observation  # noqa: E402


DEFAULT_DATA = (
    "data/kineparity/pick_observer_test_rot10_e30s50_seed20260732.npz"
)
DEFAULT_VISION_CACHE = (
    "data/kineparity/cache_pick_observer_test_rot10_vitb_l11"
)
DEFAULT_PROTOTYPE_BANK = (
    "data/kineparity/prototype_pick_control_train_rot5_vitb_l11_seed0"
)
DEFAULT_CHECKPOINTS = (
    "outputs/action_chunk/p1/pick_fcpg_pureres_seed0_retry1/"
    "fcpg_action_chunk_best.pt",
    "outputs/action_chunk/p1/pick_fcpg_pureres_seed1/"
    "fcpg_action_chunk_best.pt",
    "outputs/action_chunk/p1/pick_fcpg_pureres_seed2/"
    "fcpg_action_chunk_best.pt",
)
DEFAULT_OUTPUT = (
    "outputs/prototype_control_candidates/"
    "pick_rot10_seed012_r64g17_l16m16_development.json"
)
DEFAULT_TASK = "pick-place-v3"
MODE_ORDER = (
    "nominal",
    "native_fcpg",
    "wide_top1",
    "best_of_three_control_oracle",
    "geometry_oracle",
    "joint_candidate_oracle",
    "capture_constrained_control_oracle",
    "uniform_visual_candidate_expectation",
    "random_k_uniform_expectation",
)
ACTION_METRICS = (
    "action_xyz_l2_h1",
    "action_xyz_l2_h8",
    "action_xyz_normalized_mse_h1",
    "action_xyz_normalized_mse_h8",
    "gripper_l1_h1",
    "gripper_l1_h8",
    "gripper_normalized_mse_h1",
    "gripper_normalized_mse_h8",
    "gripper_disagreement_h1",
    "gripper_disagreement_h8",
    "epe_px",
    "max_anchor_error_px",
)


@dataclass(frozen=True)
class ControlData:
    obs: np.ndarray
    action: np.ndarray
    xyz: np.ndarray
    q_nominal: np.ndarray
    q_true: np.ndarray
    valid_nominal: np.ndarray
    valid_true: np.ndarray
    task: np.ndarray
    episode: np.ndarray
    frame: np.ndarray
    previous_index: np.ndarray
    rotation_deg: np.ndarray
    translation_xyz: np.ndarray
    image_size: tuple[int, int]


@dataclass(frozen=True)
class CandidatePool:
    points: np.ndarray
    score: np.ndarray
    valid: np.ndarray
    grid_indices: np.ndarray


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _previous_frame_indices(
    task: np.ndarray,
    episode: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """Return the previous frame in each episode, padding its first frame."""

    count = len(frame)
    if len(task) != count or len(episode) != count:
        raise ValueError("identity arrays must have equal lengths")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in range(count):
        groups[(str(task[index]), int(episode[index]))].append(index)
    previous = np.empty(count, dtype=np.int64)
    for key, indices in groups.items():
        ordered = sorted(indices, key=lambda index: int(frame[index]))
        frames = np.asarray([int(frame[index]) for index in ordered])
        if len(np.unique(frames)) != len(frames):
            raise ValueError(f"duplicate frame id in episode {key}")
        if len(frames) > 1 and not bool(np.all(np.diff(frames) == 1)):
            raise ValueError(f"non-contiguous frames in episode {key}")
        for position, index in enumerate(ordered):
            previous[index] = ordered[max(position - 1, 0)]
    return previous


def load_control_data(path: str | Path) -> ControlData:
    required = {
        "obs",
        "action",
        "task_name",
        "episode_id",
        "frame_id",
        "control_keypoints_xyz",
        "control_q_nominal",
        "control_q_true",
        "control_valid_nominal",
        "control_valid_true",
    }
    with np.load(path, allow_pickle=False) as raw:
        missing = sorted(required.difference(raw.files))
        if missing:
            raise ValueError(f"control dataset is missing fields: {missing}")
        q_nominal = np.asarray(raw["control_q_nominal"], dtype=np.float32)
        q_true = np.asarray(raw["control_q_true"], dtype=np.float32)
        xyz = np.asarray(raw["control_keypoints_xyz"], dtype=np.float32)
        if q_nominal.ndim != 3 or q_nominal.shape[1:] != (3, 2):
            raise ValueError("control_q_nominal must have shape [N, 3, 2]")
        if q_true.shape != q_nominal.shape or xyz.shape != (len(q_nominal), 3, 3):
            raise ValueError("control anchor arrays have inconsistent shapes")
        task = raw["task_name"].astype(str)
        episode = np.asarray(raw["episode_id"], dtype=np.int64)
        frame = np.asarray(raw["frame_id"], dtype=np.int64)
        valid_nominal = np.asarray(raw["control_valid_nominal"], dtype=bool)
        valid_true = np.asarray(raw["control_valid_true"], dtype=bool)
        if "rgb" in raw.files:
            height, width = (int(value) for value in raw["rgb"].shape[1:3])
        else:
            raise ValueError("control dataset requires rgb to establish image size")
        rotation = (
            np.asarray(raw["actual_camera_rotation_deg"], dtype=np.float32)
            if "actual_camera_rotation_deg" in raw.files
            else np.full(len(q_nominal), np.nan, dtype=np.float32)
        )
        translation = (
            np.asarray(raw["actual_camera_translation_xyz"], dtype=np.float32)
            if "actual_camera_translation_xyz" in raw.files
            else np.full((len(q_nominal), 3), np.nan, dtype=np.float32)
        )
        return ControlData(
            obs=np.asarray(raw["obs"], dtype=np.float32),
            action=np.asarray(raw["action"], dtype=np.float32),
            xyz=xyz,
            q_nominal=q_nominal,
            q_true=q_true,
            valid_nominal=valid_nominal,
            valid_true=valid_true,
            task=task,
            episode=episode,
            frame=frame,
            previous_index=_previous_frame_indices(task, episode, frame),
            rotation_deg=rotation,
            translation_xyz=translation,
            image_size=(width, height),
        )


def validate_artifacts(
    *,
    data_path: Path,
    vision_cache: Path,
    prototype_bank: Path,
    policy_args: Sequence[Mapping[str, object]],
    evaluation_tasks: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed if data, frozen features, prototypes, or policies disagree."""

    cache_metadata = json.loads(
        (vision_cache / "metadata.json").read_text(encoding="utf-8")
    )
    prototype_metadata = json.loads(
        (prototype_bank / "metadata.json").read_text(encoding="utf-8")
    )
    source = resolved_path(str(cache_metadata.get("source_data", "")))
    if source != data_path:
        raise ValueError(
            f"vision cache source {source} does not match dataset {data_path}"
        )
    prototype_source_value = prototype_metadata.get("source_data")
    if not prototype_source_value:
        raise ValueError("prototype bank metadata must record its training source")
    prototype_source = resolved_path(str(prototype_source_value))
    if prototype_source == data_path:
        raise ValueError(
            "prototype bank source must be distinct from the evaluation dataset"
        )
    if not prototype_source.is_file():
        raise ValueError(
            f"prototype bank training source is unavailable: {prototype_source}"
        )
    cached_model = str(cache_metadata["model_name"])
    cached_layers = tuple(int(value) for value in cache_metadata["selected_layers"])
    prototype_keypoint_source = str(prototype_metadata.get("keypoint_source"))
    if prototype_keypoint_source not in {"control", "primary"}:
        raise ValueError("prototype bank has an unknown keypoint source")
    if int(prototype_metadata.get("num_keypoints", -1)) != 3:
        raise ValueError("prototype bank must contain exactly three control anchors")
    requested_tasks = tuple(sorted({str(value).strip() for value in evaluation_tasks}))
    if not requested_tasks or any(not value for value in requested_tasks):
        raise ValueError("evaluation_tasks must contain at least one non-empty task")
    source_site_key = (
        "control_keypoint_site"
        if prototype_keypoint_source == "control"
        else "keypoint_site"
    )
    with np.load(prototype_source, allow_pickle=False) as source_raw:
        if "task_name" not in source_raw.files:
            raise ValueError("prototype training source must record task_name")
        prototype_training_tasks = tuple(
            sorted(
                {
                    str(value)
                    for value in np.asarray(source_raw["task_name"]).reshape(-1).tolist()
                }
            )
        )
        missing_tasks = sorted(set(requested_tasks).difference(prototype_training_tasks))
        if missing_tasks:
            raise ValueError(
                "prototype bank was not trained on the selected task(s): "
                f"{missing_tasks}; training tasks: {prototype_training_tasks}"
            )
        if source_site_key not in source_raw.files:
            raise ValueError(
                "prototype training source lacks semantic keypoint sites: "
                f"{source_site_key}"
            )
        source_sites = tuple(
            str(value)
            for value in np.asarray(source_raw[source_site_key]).reshape(-1).tolist()
        )
    if len(source_sites) != 3 or len(set(source_sites)) != 3:
        raise ValueError(
            "prototype training source must define three unique semantic keypoint sites"
        )

    metadata_tasks = tuple(
        sorted(
            {
                str(value)
                for value in prototype_metadata.get("training_tasks", ())
            }
        )
    )
    if metadata_tasks and metadata_tasks != prototype_training_tasks:
        raise ValueError(
            "prototype metadata training tasks do not match its training source"
        )

    metadata_sites = tuple(
        str(value) for value in prototype_metadata.get("keypoint_site", ())
    )
    if metadata_sites:
        if len(metadata_sites) != 3 or len(set(metadata_sites)) != 3:
            raise ValueError(
                "prototype metadata must define three unique semantic keypoint sites"
            )
        if metadata_sites != source_sites:
            raise ValueError(
                "prototype metadata keypoint sites do not match its training source"
            )
    else:
        # Legacy banks remain usable only when their immutable source deterministically
        # establishes the semantic order used by the prototype builder.
        prototype_metadata["keypoint_site"] = list(source_sites)
    prototype_metadata["training_tasks"] = list(prototype_training_tasks)

    with np.load(data_path, allow_pickle=False) as raw:
        if "control_keypoint_site" in raw.files:
            site_key = "control_keypoint_site"
        elif "keypoint_site" in raw.files:
            site_key = "keypoint_site"
        else:
            raise ValueError("evaluation dataset lacks semantic control keypoint sites")
        evaluation_sites = tuple(str(value) for value in raw[site_key].tolist())
    if len(evaluation_sites) != 3 or len(set(evaluation_sites)) != 3:
        raise ValueError(
            "evaluation dataset must define three unique semantic control keypoint sites"
        )
    if source_sites != evaluation_sites:
        raise ValueError(
            "prototype keypoint sites do not match evaluation control-anchor order"
        )
    if str(prototype_metadata["model_name"]) != cached_model:
        raise ValueError("prototype bank and vision cache use different ViT models")
    if tuple(int(value) for value in prototype_metadata["selected_layers"]) != cached_layers:
        raise ValueError("prototype bank and vision cache use different ViT layers")
    cache_size = (
        int(cache_metadata["image_width"]),
        int(cache_metadata["image_height"]),
    )
    prototype_size = (
        int(prototype_metadata["image_width"]),
        int(prototype_metadata["image_height"]),
    )
    if cache_size != prototype_size:
        raise ValueError("prototype bank and vision cache image sizes differ")
    for index, args in enumerate(policy_args):
        if str(args.get("vision_model_name")) != cached_model:
            raise ValueError(f"policy {index} ViT model does not match vision cache")
        layers = tuple(int(value) for value in args.get("vision_selected_layers", ()))
        if layers != cached_layers:
            raise ValueError(f"policy {index} ViT layers do not match vision cache")
        if int(args.get("obs_horizon", 0)) != 2:
            raise ValueError(f"policy {index} must use obs_horizon=2")
        if int(args.get("action_horizon", 0)) < 8:
            raise ValueError(f"policy {index} must predict at least eight actions")
        if str(args.get("mode")) != "fcpg":
            raise ValueError(f"policy {index} is not an FCPG checkpoint")
    return cache_metadata, prototype_metadata


def validate_policy_checkpoint_tasks(
    policy_args: Sequence[Mapping[str, object]],
    *,
    evaluation_task: str,
) -> None:
    """Require every policy to have been trained on the selected task."""

    requested = str(evaluation_task).strip()
    if not requested:
        raise ValueError("evaluation_task must be non-empty")
    for index, args in enumerate(policy_args):
        primary = args.get("data")
        extra = args.get("extra_data", ())
        if isinstance(extra, (str, Path)):
            extra_sources = (extra,)
        elif isinstance(extra, Sequence):
            extra_sources = tuple(extra)
        else:
            raise ValueError(f"policy {index} has invalid extra_data provenance")
        source_values = ((primary,) if primary else ()) + extra_sources
        if not source_values:
            raise ValueError(f"policy {index} does not record its training data")

        training_tasks: set[str] = set()
        for value in source_values:
            source = resolved_path(str(value))
            if not source.is_file():
                raise ValueError(
                    f"policy {index} training source is unavailable: {source}"
                )
            with np.load(source, allow_pickle=False) as raw:
                if "task_name" not in raw.files:
                    raise ValueError(
                        f"policy {index} training source lacks task_name: {source}"
                    )
                training_tasks.update(
                    str(task)
                    for task in np.asarray(raw["task_name"]).reshape(-1).tolist()
                )
        if requested not in training_tasks:
            raise ValueError(
                f"policy {index} was not trained on {requested!r}; "
                f"training tasks: {tuple(sorted(training_tasks))}"
            )


@torch.inference_mode()
def generate_candidate_pool(
    data: ControlData,
    *,
    indices: np.ndarray,
    backbone: np.ndarray,
    descriptors: torch.Tensor,
    feature_level: int,
    device: torch.device,
    batch_size: int,
    radius_px: float,
    grid_size: int,
    per_anchor_topk: int,
    max_hypotheses: int,
    nms_radius_px: float,
    min_confidence: float,
) -> tuple[np.ndarray, CandidatePool]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("indices must be a non-empty vector")
    if int(indices.min()) < 0 or int(indices.max()) >= len(data.frame):
        raise ValueError("candidate indices lie outside the dataset")
    count = len(indices)
    points: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    grid_indices: list[np.ndarray] = []
    for start in range(0, count, batch_size):
        selected = indices[start : start + batch_size]
        features = torch.from_numpy(
            np.array(backbone[selected, feature_level], dtype=np.float16, copy=True)
        ).to(device)
        if device.type == "cpu":
            features = features.float()
        q_nominal = torch.from_numpy(data.q_nominal[selected]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            result = generate_joint_prototype_hypotheses(
                features,
                q_nominal,
                descriptors,
                image_size=data.image_size,
                radius_px=radius_px,
                grid_size=grid_size,
                per_anchor_topk=per_anchor_topk,
                max_hypotheses=max_hypotheses,
                nms_radius_px=nms_radius_px,
                min_confidence=min_confidence,
            )
        points.append(result.points_px.detach().float().cpu().numpy())
        scores.append(result.joint_score.detach().float().cpu().numpy())
        valid.append(result.valid.detach().cpu().numpy().astype(bool))
        grid_indices.append(result.grid_indices.detach().cpu().numpy())
    return indices, CandidatePool(
        points=np.concatenate(points),
        score=np.concatenate(scores),
        valid=np.concatenate(valid),
        grid_indices=np.concatenate(grid_indices),
    )


def select_evaluation_indices(
    data: ControlData,
    *,
    task: str,
    max_frames: int,
) -> np.ndarray:
    """Select one task without copying a shared multi-task dataset/cache."""

    available = tuple(sorted(set(str(value) for value in data.task.tolist())))
    requested = str(task).strip()
    if requested:
        if requested not in available:
            raise ValueError(
                f"requested task {requested!r} is absent; available tasks: {available}"
            )
        indices = np.flatnonzero(data.task == requested).astype(np.int64)
    else:
        if len(available) != 1:
            raise ValueError(
                "multi-task datasets require an explicit --task; "
                f"available tasks: {available}"
            )
        indices = np.arange(len(data.frame), dtype=np.int64)
    if max_frames > 0:
        indices = indices[:max_frames]
    if len(indices) == 0:
        raise ValueError("task/frame selection produced no rows")

    selected = set(int(value) for value in indices)
    missing_previous = sorted(
        {
            int(data.previous_index[index])
            for index in indices
            if int(data.previous_index[index]) not in selected
        }
    )
    if missing_previous:
        raise ValueError(
            "selected task/frame rows are not prefix-closed within episodes: "
            f"{missing_previous[:5]}"
        )
    return indices


def generate_random_joint_pool(
    data: ControlData,
    *,
    indices: np.ndarray,
    radius_px: float,
    grid_size: int,
    candidate_count: int,
    seed: int,
) -> CandidatePool:
    """Generate a deterministic same-grid random-K spatial control bank."""

    if grid_size <= 0 or grid_size % 2 == 0:
        raise ValueError("grid_size must be a positive odd integer")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    offsets_1d = np.linspace(-radius_px, radius_px, grid_size, dtype=np.float32)
    offset_y, offset_x = np.meshgrid(offsets_1d, offsets_1d, indexing="ij")
    offsets = np.stack([offset_x, offset_y], axis=-1).reshape(-1, 2)
    grid_points = len(offsets)
    cartesian_size = grid_points**3
    if candidate_count > cartesian_size:
        raise ValueError("random candidate budget exceeds joint grid size")

    all_points = []
    all_grid_indices = []
    for frame_index in np.asarray(indices, dtype=np.int64):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(seed),
                    int(frame_index),
                    int(data.episode[frame_index]),
                    int(data.frame[frame_index]),
                ]
            )
        )
        flat = rng.choice(cartesian_size, size=candidate_count, replace=False)
        grid_indices = np.empty((candidate_count, 3), dtype=np.int64)
        remainder = flat.astype(np.int64, copy=True)
        for anchor in range(2, -1, -1):
            grid_indices[:, anchor] = remainder % grid_points
            remainder //= grid_points
        all_grid_indices.append(grid_indices)
        all_points.append(
            data.q_nominal[frame_index, None] + offsets[grid_indices]
        )
    points = np.asarray(all_points, dtype=np.float32)
    width, height = data.image_size
    valid = (
        np.isfinite(points).all(axis=-1)
        & (points[..., 0] >= 0.0)
        & (points[..., 0] < width)
        & (points[..., 1] >= 0.0)
        & (points[..., 1] < height)
    )
    return CandidatePool(
        points=points,
        score=np.zeros(points.shape[:2], dtype=np.float32),
        valid=valid,
        grid_indices=np.asarray(all_grid_indices, dtype=np.int64),
    )


def build_deployable_base_batch(
    data: ControlData,
    *,
    indices: np.ndarray,
    obs_mode: str,
    backbone: np.ndarray,
    prefix: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build H=2 policy inputs without any oracle/current-future label fields."""

    previous = data.previous_index[indices]
    history = np.stack([previous, indices], axis=1)
    host: dict[str, np.ndarray] = {
        "obs": select_policy_observation(data.obs[history], obs_mode),
        "keypoints_xyz": data.xyz[history],
        "q_nominal": data.q_nominal[history],
        # Only nominal, deployable geometry validity enters the policy.
        "valid": data.valid_nominal[history],
        "vision_backbone_features": np.array(
            backbone[history], dtype=np.float16, copy=True
        ),
        "vision_prefix": np.array(prefix[history], dtype=np.float16, copy=True),
        "image_size": np.tile(np.asarray(data.image_size, dtype=np.int64), (len(indices), 1)),
    }
    batch: dict[str, torch.Tensor] = {}
    for key, value in host.items():
        tensor = torch.from_numpy(np.asarray(value)).to(device)
        if device.type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        batch[key] = tensor
    forbidden = {"q_true", "control_q_true", "action", "next_obs", "reward", "success"}
    if forbidden.intersection(batch):
        raise AssertionError("offline labels leaked into deployable policy batch")
    return batch


def _repeat_snapshot_tensor(value: torch.Tensor, repeats: int) -> torch.Tensor:
    shape = value.shape
    return (
        value[:, None]
        .expand(shape[0], repeats, *shape[1:])
        .reshape(shape[0] * repeats, *shape[1:])
    )


def history_grounding_points(
    past_wide: torch.Tensor,
    current_points: torch.Tensor,
) -> torch.Tensor:
    """Combine a fixed wide past with one or more current interventions."""

    if past_wide.ndim != 3 or past_wide.shape[-1] != 2:
        raise ValueError("past_wide must have shape [B, K, 2]")
    if current_points.ndim != 4 or current_points.shape[-2:] != past_wide.shape[-2:]:
        raise ValueError("current_points must have shape [B, M, K, 2]")
    if current_points.shape[0] != past_wide.shape[0]:
        raise ValueError("past and current batch sizes differ")
    repeats = current_points.shape[1]
    past = past_wide[:, None].expand(-1, repeats, -1, -1)
    return torch.stack([past, current_points], dim=2).reshape(
        past_wide.shape[0] * repeats, 2, past_wide.shape[1], 2
    )


@torch.inference_mode()
def infer_policy_actions(
    model: Any,
    base_batch: Mapping[str, torch.Tensor],
    *,
    past_wide: torch.Tensor,
    current_points: torch.Tensor,
    candidate_chunk_size: int,
    device_type: str,
) -> np.ndarray:
    """Evaluate current grounding interventions without exposing oracle labels."""

    if str(model.mode) != "fcpg":
        raise ValueError("candidate intervention requires a frozen FCPG policy")
    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    if any("true" in key or key.startswith("next_") for key in base_batch):
        raise ValueError("deployable base batch contains an offline label")
    batch_size, candidate_count = current_points.shape[:2]
    action_parts: list[np.ndarray] = []
    for start in range(0, candidate_count, candidate_chunk_size):
        points = current_points[:, start : start + candidate_chunk_size]
        repeats = points.shape[1]
        expanded = {
            key: _repeat_snapshot_tensor(value, repeats)
            for key, value in base_batch.items()
        }
        expanded["fcpg_q_corrected"] = history_grounding_points(
            past_wide, points
        )
        expanded["fcpg_reliability"] = torch.ones(
            batch_size * repeats,
            2,
            past_wide.shape[1],
            device=past_wide.device,
            dtype=torch.float32,
        )
        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=device_type == "cuda",
        ):
            output = model.predict_action_chunk(expanded)
        expected_points = expanded["fcpg_q_corrected"]
        if not torch.allclose(
            output.grounding_points_px.float(), expected_points.float(), atol=1e-4, rtol=0.0
        ):
            raise RuntimeError("policy did not use the requested grounding intervention")
        action_parts.append(
            output.action.detach()
            .float()
            .reshape(batch_size, repeats, output.action.shape[1], 4)
            .cpu()
            .numpy()
        )
    return np.concatenate(action_parts, axis=1)


@torch.inference_mode()
def infer_native_current_points(
    model: Any,
    base_batch: Mapping[str, torch.Tensor],
    *,
    image_size: tuple[int, int],
    device_type: str,
) -> np.ndarray:
    """Run the policy checkpoint's own frozen corrector on the current frame."""

    adapter = getattr(model, "fcpg_adapter", None)
    if adapter is None:
        raise ValueError("native FCPG baseline requires the checkpoint corrector")
    with torch.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=device_type == "cuda",
    ):
        output = adapter.forward_from_backbone_features(
            base_batch["vision_backbone_features"][:, -1],
            base_batch["vision_prefix"][:, -1],
            base_batch["q_nominal"][:, -1],
            base_batch["keypoints_xyz"][:, -1],
            base_batch["obs"][:, -1],
            image_size=image_size,
        )
    return output.q_corrected_px.detach().float().cpu().numpy()


def action_distance_metrics(
    actions: np.ndarray,
    teacher: np.ndarray,
    action_std: np.ndarray,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Return raw and checkpoint-normalized distances for candidate actions."""

    values = np.asarray(actions, dtype=np.float64)
    target = np.asarray(teacher, dtype=np.float64)
    std = np.asarray(action_std, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 4:
        raise ValueError("actions must have shape [B, M, H, 4]")
    if target.shape != (values.shape[0], values.shape[2], 4):
        raise ValueError("teacher must have shape [B, H, 4]")
    if std.shape != (4,) or not bool(np.all(np.isfinite(std) & (std > 0))):
        raise ValueError("action_std must be a positive finite 4-vector")
    if horizon <= 0 or horizon > values.shape[2]:
        raise ValueError("invalid action horizon")
    delta = values[:, :, :horizon] - target[:, None, :horizon]
    xyz = delta[..., :3]
    gripper = delta[..., 3]
    return {
        "xyz_l2": np.linalg.norm(xyz, axis=-1).mean(axis=-1),
        "xyz_normalized_mse": np.square(xyz / std[None, None, None, :3]).mean(
            axis=(-1, -2)
        ),
        "gripper_l1": np.abs(gripper).mean(axis=-1),
        "gripper_normalized_mse": np.square(
            gripper / float(std[3])
        ).mean(axis=-1),
        "gripper_disagreement": (
            (values[:, :, :horizon, 3] >= 0.5)
            != (target[:, None, :horizon, 3] >= 0.5)
        ).mean(axis=-1),
    }


def candidate_point_metrics(
    points: np.ndarray,
    q_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.linalg.norm(
        np.asarray(points, dtype=np.float64)
        - np.asarray(q_true, dtype=np.float64)[:, None],
        axis=-1,
    )
    return errors.mean(axis=-1), errors.max(axis=-1)


def select_candidate_indices(
    *,
    candidate_valid: np.ndarray,
    epe: np.ndarray,
    max_error: np.ndarray,
    normalized_h8: np.ndarray,
    capture_threshold_px: float,
) -> dict[str, np.ndarray]:
    """Build geometry/control oracle labels without affecting candidate creation."""

    valid = np.asarray(candidate_valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("candidate_valid must have shape [B, M]")
    if any(array.shape != valid.shape for array in (epe, max_error, normalized_h8)):
        raise ValueError("candidate metric shapes differ")

    def argmin(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        eligible = np.where(mask, values, np.inf)
        result = eligible.argmin(axis=1).astype(np.int64)
        result[~mask.any(axis=1)] = -1
        return result

    return {
        "geometry_oracle": argmin(epe, valid),
        "joint_candidate_oracle": argmin(normalized_h8, valid),
        "capture_constrained_control_oracle": argmin(
            normalized_h8,
            valid & (max_error <= float(capture_threshold_px)),
        ),
    }


def _take_selected(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.full(len(indices), np.nan, dtype=np.float64)
    available = indices >= 0
    if bool(available.any()):
        rows = np.flatnonzero(available)
        result[available] = values[rows, indices[available]]
    return result


def _mode_rows(
    *,
    data: ControlData,
    indices: np.ndarray,
    policy_seed: int,
    candidates: CandidatePool,
    random_candidates: CandidatePool,
    native_points: np.ndarray,
    actions: np.ndarray,
    action_std: np.ndarray,
    capture_threshold_px: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create selected-mode rows and policy-independent candidate diagnostics."""

    candidate_count = candidates.points.shape[1]
    random_count = random_candidates.points.shape[1]
    candidate_actions = actions[:, :candidate_count]
    random_actions = actions[:, candidate_count : candidate_count + random_count]
    baseline_offset = candidate_count + random_count
    nominal_actions = actions[:, baseline_offset]
    native_actions = actions[:, baseline_offset + 1]
    teacher = actions[:, baseline_offset + 2]
    if actions.shape[1] != baseline_offset + 3:
        raise ValueError("combined action tensor has an unexpected candidate count")

    candidate_valid = candidates.valid.all(axis=-1) & np.isfinite(candidates.score)
    base_valid = (
        data.valid_nominal[indices] & data.valid_true[indices]
    ).all(axis=-1)
    candidate_valid &= base_valid[:, None]
    candidate_epe, candidate_max = candidate_point_metrics(
        candidates.points, data.q_true[indices]
    )
    random_valid = random_candidates.valid.all(axis=-1) & base_valid[:, None]
    random_epe, random_max = candidate_point_metrics(
        random_candidates.points, data.q_true[indices]
    )
    candidate_metrics: dict[str, dict[str, np.ndarray]] = {}
    for horizon in (1, 8):
        candidate_metrics[f"h{horizon}"] = action_distance_metrics(
            candidate_actions, teacher, action_std, horizon
        )
    random_metrics = {
        f"h{horizon}": action_distance_metrics(
            random_actions, teacher, action_std, horizon
        )
        for horizon in (1, 8)
    }
    selectors = select_candidate_indices(
        candidate_valid=candidate_valid,
        epe=candidate_epe,
        max_error=candidate_max,
        normalized_h8=candidate_metrics["h8"]["xyz_normalized_mse"],
        capture_threshold_px=capture_threshold_px,
    )

    baseline_points = np.stack(
        [data.q_nominal[indices], native_points, candidates.points[:, 0]], axis=1
    )
    baseline_actions = np.stack(
        [nominal_actions, native_actions, candidate_actions[:, 0]], axis=1
    )
    baseline_epe, baseline_max = candidate_point_metrics(
        baseline_points, data.q_true[indices]
    )
    baseline_metrics = {
        f"h{horizon}": action_distance_metrics(
            baseline_actions, teacher, action_std, horizon
        )
        for horizon in (1, 8)
    }
    best3 = baseline_metrics["h8"]["xyz_normalized_mse"].argmin(axis=1)

    selected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "nominal": (
            np.zeros(len(indices), dtype=np.int64),
            baseline_epe[:, 0],
            baseline_max[:, 0],
        ),
        "native_fcpg": (
            np.ones(len(indices), dtype=np.int64),
            baseline_epe[:, 1],
            baseline_max[:, 1],
        ),
        "wide_top1": (
            np.full(len(indices), 2, dtype=np.int64),
            baseline_epe[:, 2],
            baseline_max[:, 2],
        ),
        "best_of_three_control_oracle": (
            best3,
            _take_selected(baseline_epe, best3),
            _take_selected(baseline_max, best3),
        ),
    }

    rows: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        if mode in {
            "uniform_visual_candidate_expectation",
            "random_k_uniform_expectation",
        }:
            use_random = mode == "random_k_uniform_expectation"
            point_epe = random_epe if use_random else candidate_epe
            point_max = random_max if use_random else candidate_max
            point_valid = random_valid if use_random else candidate_valid
            mode_epe = np.asarray(
                [
                    point_epe[row, point_valid[row]].mean()
                    if point_valid[row].any()
                    else np.nan
                    for row in range(len(indices))
                ]
            )
            mode_max = np.asarray(
                [
                    point_max[row, point_valid[row]].mean()
                    if point_valid[row].any()
                    else np.nan
                    for row in range(len(indices))
                ]
            )
            source = "random" if use_random else "candidate"
            selected_indices = np.full(len(indices), -2, dtype=np.int64)
        elif mode in selectors:
            selected_indices = selectors[mode]
            mode_epe = _take_selected(candidate_epe, selected_indices)
            mode_max = _take_selected(candidate_max, selected_indices)
            source = "candidate"
        else:
            selected_indices, mode_epe, mode_max = selected[mode]
            source = "baseline"

        metrics_by_horizon: dict[int, dict[str, np.ndarray]] = {}
        for horizon in (1, 8):
            if source == "candidate":
                metric_source = candidate_metrics[f"h{horizon}"]
                point_valid = candidate_valid
            elif source == "random":
                metric_source = random_metrics[f"h{horizon}"]
                point_valid = random_valid
            else:
                metric_source = baseline_metrics[f"h{horizon}"]
                point_valid = candidate_valid
            if mode in {
                "uniform_visual_candidate_expectation",
                "random_k_uniform_expectation",
            }:
                metrics_by_horizon[horizon] = {
                    name: np.asarray(
                        [
                            values[row, point_valid[row]].mean()
                            if point_valid[row].any()
                            else np.nan
                            for row in range(len(indices))
                        ]
                    )
                    for name, values in metric_source.items()
                }
            else:
                metrics_by_horizon[horizon] = {
                    name: _take_selected(values, selected_indices)
                    for name, values in metric_source.items()
                }

        for row_index, frame_index in enumerate(indices):
            selected_candidate_index = None
            if source == "candidate" and selected_indices[row_index] >= 0:
                selected_candidate_index = int(selected_indices[row_index])
            elif mode == "wide_top1":
                selected_candidate_index = 0
            selected_baseline_option = None
            if source == "baseline":
                baseline_names = ("nominal", "native_fcpg", "wide_top1")
                baseline_index = int(selected_indices[row_index])
                if 0 <= baseline_index < len(baseline_names):
                    selected_baseline_option = baseline_names[baseline_index]
            row = {
                "task": str(data.task[frame_index]),
                "episode": int(data.episode[frame_index]),
                "frame": int(data.frame[frame_index]),
                "frame_index": int(frame_index),
                "policy_seed": int(policy_seed),
                "mode": mode,
                "selected_candidate_index": selected_candidate_index,
                "selected_baseline_option": selected_baseline_option,
                "epe_px": float(mode_epe[row_index]),
                "max_anchor_error_px": float(mode_max[row_index]),
                "actual_camera_rotation_deg": float(data.rotation_deg[frame_index]),
                "actual_camera_translation_xyz": data.translation_xyz[
                    frame_index
                ].astype(float).tolist(),
            }
            for horizon in (1, 8):
                suffix = f"h{horizon}"
                values = metrics_by_horizon[horizon]
                row[f"action_xyz_l2_{suffix}"] = float(
                    values["xyz_l2"][row_index]
                )
                row[f"action_xyz_normalized_mse_{suffix}"] = float(
                    values["xyz_normalized_mse"][row_index]
                )
                row[f"gripper_l1_{suffix}"] = float(
                    values["gripper_l1"][row_index]
                )
                row[f"gripper_normalized_mse_{suffix}"] = float(
                    values["gripper_normalized_mse"][row_index]
                )
                row[f"gripper_disagreement_{suffix}"] = float(
                    values["gripper_disagreement"][row_index]
                )
            rows.append(row)

    diagnostic_rows = []
    for row_index, frame_index in enumerate(indices):
        valid_errors = candidate_max[row_index, candidate_valid[row_index]]
        diagnostic_rows.append(
            {
                "task": str(data.task[frame_index]),
                "episode": int(data.episode[frame_index]),
                "frame": int(data.frame[frame_index]),
                "frame_index": int(frame_index),
                "valid_candidate_count": int(candidate_valid[row_index].sum()),
                "candidate_count": int(candidate_count),
                "random_valid_candidate_count": int(random_valid[row_index].sum()),
                "random_candidate_count": int(random_count),
                "capture_threshold_px": float(capture_threshold_px),
                "coverage": bool(
                    valid_errors.size
                    and bool((valid_errors <= capture_threshold_px).any())
                ),
                "best_candidate_epe_px": (
                    float(candidate_epe[row_index, candidate_valid[row_index]].min())
                    if candidate_valid[row_index].any()
                    else float("nan")
                ),
                "best_candidate_max_anchor_error_px": (
                    float(valid_errors.min()) if valid_errors.size else float("nan")
                ),
            }
        )
    return rows, diagnostic_rows


def _episode_arrays(
    rows: Sequence[Mapping[str, Any]],
    mode: str,
    metric: str,
) -> dict[int, dict[tuple[str, int], float]]:
    frame_values: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if str(row["mode"]) != mode:
            continue
        value = float(row[metric])
        if np.isfinite(value):
            frame_values[
                (int(row["policy_seed"]), str(row["task"]), int(row["episode"]))
            ].append(value)
    result: dict[int, dict[tuple[str, int], float]] = defaultdict(dict)
    for (seed, task, episode), values in frame_values.items():
        result[seed][(task, episode)] = float(np.mean(values))
    return dict(result)


def crossed_policy_episode_summary(
    episode_values: Mapping[int, Mapping[tuple[str, int], float]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Two-way crossed bootstrap over policy seeds and shared episodes."""

    if not episode_values:
        return {"available": False}
    seeds = tuple(sorted(episode_values))
    episode_keys = tuple(sorted(episode_values[seeds[0]]))
    if not episode_keys:
        return {"available": False}
    for seed in seeds:
        if tuple(sorted(episode_values[seed])) != episode_keys:
            raise ValueError("crossed bootstrap requires identical episode keys per seed")
    matrix = np.asarray(
        [[episode_values[seed][key] for key in episode_keys] for seed in seeds],
        dtype=np.float64,
    )
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("crossed bootstrap received a non-finite episode value")
    result: dict[str, Any] = {
        "available": True,
        "estimate": float(matrix.mean()),
        "policy_seeds": list(seeds),
        "episodes": int(len(episode_keys)),
        "per_policy_seed": {
            str(seed): float(matrix[index].mean())
            for index, seed in enumerate(seeds)
        },
    }
    if bootstrap_samples <= 0:
        return result
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    batch_size = min(5_000, bootstrap_samples)
    for start in range(0, bootstrap_samples, batch_size):
        count = min(batch_size, bootstrap_samples - start)
        seed_draw = rng.integers(0, len(seeds), size=(count, len(seeds)))
        episode_draw = rng.integers(
            0, len(episode_keys), size=(count, len(episode_keys))
        )
        values = np.empty((count, len(seeds)), dtype=np.float64)
        for position in range(len(seeds)):
            selected_seed_values = matrix[seed_draw[:, position]]
            values[:, position] = np.take_along_axis(
                selected_seed_values, episode_draw, axis=1
            ).mean(axis=1)
        draws[start : start + count] = values.mean(axis=1)
    result["crossed_bootstrap_95_ci"] = [
        float(value) for value in np.quantile(draws, [0.025, 0.975])
    ]
    return result


def crossed_paired_relative_reduction(
    candidate_values: Mapping[int, Mapping[tuple[str, int], float]],
    baseline_values: Mapping[int, Mapping[tuple[str, int], float]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Ratio of aggregate means under a crossed seed/episode bootstrap."""

    if set(candidate_values) != set(baseline_values) or not candidate_values:
        raise ValueError("paired relative reduction requires identical policy seeds")
    seeds = tuple(sorted(candidate_values))
    episode_keys = tuple(sorted(candidate_values[seeds[0]]))
    if not episode_keys:
        return {"available": False}
    for seed in seeds:
        if tuple(sorted(candidate_values[seed])) != episode_keys:
            raise ValueError("candidate relative-reduction episode keys differ by seed")
        if tuple(sorted(baseline_values[seed])) != episode_keys:
            raise ValueError("baseline relative-reduction episode keys differ by seed")
    candidate = np.asarray(
        [[candidate_values[seed][key] for key in episode_keys] for seed in seeds],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [[baseline_values[seed][key] for key in episode_keys] for seed in seeds],
        dtype=np.float64,
    )
    if not bool(np.isfinite(candidate).all() and np.isfinite(baseline).all()):
        raise ValueError("paired relative reduction received non-finite episode values")

    def reduction(candidate_mean: float, baseline_mean: float) -> float:
        return (baseline_mean - candidate_mean) / max(abs(baseline_mean), 1e-12)

    result: dict[str, Any] = {
        "available": True,
        "estimate": reduction(float(candidate.mean()), float(baseline.mean())),
        "candidate_aggregate_mean": float(candidate.mean()),
        "baseline_aggregate_mean": float(baseline.mean()),
        "policy_seeds": list(seeds),
        "episodes": int(len(episode_keys)),
        "per_policy_seed": {
            str(seed): reduction(
                float(candidate[index].mean()), float(baseline[index].mean())
            )
            for index, seed in enumerate(seeds)
        },
    }
    result["improving_policy_seeds"] = int(
        sum(float(value) > 0.0 for value in result["per_policy_seed"].values())
    )
    result["policy_seed_count"] = int(len(seeds))
    if bootstrap_samples <= 0:
        return result
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    near_zero = 0
    batch_size = min(5_000, bootstrap_samples)
    for start in range(0, bootstrap_samples, batch_size):
        count = min(batch_size, bootstrap_samples - start)
        seed_draw = rng.integers(0, len(seeds), size=(count, len(seeds)))
        episode_draw = rng.integers(
            0, len(episode_keys), size=(count, len(episode_keys))
        )
        candidate_means = np.empty((count, len(seeds)), dtype=np.float64)
        baseline_means = np.empty((count, len(seeds)), dtype=np.float64)
        for position in range(len(seeds)):
            candidate_selected = candidate[seed_draw[:, position]]
            baseline_selected = baseline[seed_draw[:, position]]
            candidate_means[:, position] = np.take_along_axis(
                candidate_selected, episode_draw, axis=1
            ).mean(axis=1)
            baseline_means[:, position] = np.take_along_axis(
                baseline_selected, episode_draw, axis=1
            ).mean(axis=1)
        candidate_draw = candidate_means.mean(axis=1)
        baseline_draw = baseline_means.mean(axis=1)
        near_zero += int((np.abs(baseline_draw) <= 1e-12).sum())
        draws[start : start + count] = (
            baseline_draw - candidate_draw
        ) / np.maximum(np.abs(baseline_draw), 1e-12)
    result["crossed_bootstrap_95_ci"] = [
        float(value) for value in np.quantile(draws, [0.025, 0.975])
    ]
    result["near_zero_baseline_draw_fraction"] = float(
        near_zero / bootstrap_samples
    )
    return result


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode_index, mode in enumerate(MODE_ORDER):
        mode_summary: dict[str, Any] = {}
        for metric_index, metric in enumerate(ACTION_METRICS):
            mode_summary[metric] = crossed_policy_episode_summary(
                _episode_arrays(rows, mode, metric),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=(
                    bootstrap_seed + mode_index * len(ACTION_METRICS) + metric_index
                ),
            )
        summary[mode] = mode_summary
    return summary


def paired_mode_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_mode: str,
    baseline_mode: str,
    metric: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Summarize a structurally paired lower-is-better mode comparison."""

    by_mode: dict[str, dict[tuple[int, str, int, int], float]] = {
        candidate_mode: {},
        baseline_mode: {},
    }
    for row in rows:
        mode = str(row["mode"])
        if mode not in by_mode:
            continue
        key = (
            int(row["policy_seed"]),
            str(row["task"]),
            int(row["episode"]),
            int(row["frame"]),
        )
        if key in by_mode[mode]:
            raise ValueError(f"duplicate paired row for mode={mode} key={key}")
        value = row[metric]
        by_mode[mode][key] = float(value) if value is not None else float("nan")
    candidate = by_mode[candidate_mode]
    baseline = by_mode[baseline_mode]
    if set(candidate) != set(baseline):
        raise ValueError(
            f"paired mode keys differ for {candidate_mode} vs {baseline_mode}"
        )

    episode_pairs: dict[tuple[int, str, int], list[tuple[float, float]]] = defaultdict(list)
    for key in sorted(candidate):
        candidate_value = candidate[key]
        baseline_value = baseline[key]
        if np.isfinite(candidate_value) and np.isfinite(baseline_value):
            episode_pairs[key[:3]].append((candidate_value, baseline_value))
    absolute: dict[int, dict[tuple[str, int], float]] = defaultdict(dict)
    candidate_episode: dict[int, dict[tuple[str, int], float]] = defaultdict(dict)
    baseline_episode: dict[int, dict[tuple[str, int], float]] = defaultdict(dict)
    frame_pairs = 0
    near_zero_baseline_episodes = 0
    for (seed, task, episode), pairs in episode_pairs.items():
        if not pairs:
            continue
        values = np.asarray(pairs, dtype=np.float64)
        candidate_mean = float(values[:, 0].mean())
        baseline_mean = float(values[:, 1].mean())
        absolute[seed][(task, episode)] = candidate_mean - baseline_mean
        candidate_episode[seed][(task, episode)] = candidate_mean
        baseline_episode[seed][(task, episode)] = baseline_mean
        if abs(baseline_mean) <= 1e-12:
            near_zero_baseline_episodes += 1
        frame_pairs += len(pairs)
    absolute_summary = crossed_policy_episode_summary(
        dict(absolute),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    relative_summary = crossed_paired_relative_reduction(
        dict(candidate_episode),
        dict(baseline_episode),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 1,
    )
    per_seed = absolute_summary.get("per_policy_seed", {})
    if per_seed:
        absolute_summary["improving_policy_seeds"] = int(
            sum(float(value) < 0.0 for value in per_seed.values())
        )
        absolute_summary["policy_seed_count"] = int(len(per_seed))
    return {
        "candidate_mode": candidate_mode,
        "baseline_mode": baseline_mode,
        "metric": metric,
        "metric_direction": "lower_is_better",
        "structural_pairs": int(len(candidate)),
        "finite_frame_pairs": int(frame_pairs),
        "near_zero_baseline_episodes": int(near_zero_baseline_episodes),
        "candidate_minus_baseline": absolute_summary,
        "relative_reduction": relative_summary,
        "relative_reduction_definition": (
            "ratio of aggregate crossed means: (baseline - candidate) / "
            "max(abs(baseline), 1e-12); positive favors candidate"
        ),
    }


def summarize_paired_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    comparisons = (
        ("joint_candidate_oracle", "best_of_three_control_oracle"),
        ("geometry_oracle", "best_of_three_control_oracle"),
        ("capture_constrained_control_oracle", "best_of_three_control_oracle"),
        ("wide_top1", "native_fcpg"),
    )
    return {
        f"{candidate}_vs_{baseline}": paired_mode_comparison(
            rows,
            candidate_mode=candidate,
            baseline_mode=baseline,
            metric="action_xyz_normalized_mse_h8",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index * 2,
        )
        for index, (candidate, baseline) in enumerate(comparisons)
    }


def summarize_candidate_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), int(row["episode"]))].append(
            float(bool(row["coverage"]))
        )
    episode_values = np.asarray(
        [np.mean(values) for _, values in sorted(grouped.items())], dtype=np.float64
    )
    result: dict[str, Any] = {
        "definition": "exists valid joint candidate with max-anchor error at or below threshold",
        "snapshot_coverage": float(
            np.mean([float(bool(row["coverage"])) for row in rows])
        ),
        "episode_macro_coverage": float(episode_values.mean()),
        "episodes": int(len(episode_values)),
        "episode_coverage_min": float(episode_values.min()),
        "episode_coverage_median": float(np.median(episode_values)),
        "episode_coverage_max": float(episode_values.max()),
    }
    if bootstrap_samples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        draws = np.empty(bootstrap_samples, dtype=np.float64)
        batch_size = min(5_000, bootstrap_samples)
        for start in range(0, bootstrap_samples, batch_size):
            count = min(batch_size, bootstrap_samples - start)
            selected = rng.integers(
                0,
                len(episode_values),
                size=(count, len(episode_values)),
            )
            draws[start : start + count] = episode_values[selected].mean(axis=1)
        result["episode_cluster_bootstrap_95_ci"] = [
            float(value) for value in np.quantile(draws, [0.025, 0.975])
        ]
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def output_argument_is_explicit(argv: Sequence[str]) -> bool:
    return any(value == "--output" or value.startswith("--output=") for value in argv)


def validate_task_output_contract(
    *,
    selected_task: str,
    output_is_explicit: bool,
) -> None:
    if selected_task != DEFAULT_TASK and not output_is_explicit:
        raise ValueError(
            f"task {selected_task!r} requires explicit --output; refusing to write "
            "the Pick default output path"
        )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=(
            "Training-free post-hoc control-headroom diagnostic for wide joint "
            "prototype candidates."
        )
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument(
        "--task",
        default="",
        help="required when --data contains more than one task",
    )
    parser.add_argument("--vision-cache", default=DEFAULT_VISION_CACHE)
    parser.add_argument("--prototype-bank", default=DEFAULT_PROTOTYPE_BANK)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-chunk-size", type=int, default=4)
    parser.add_argument("--radius-px", type=float, default=64.0)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--per-anchor-topk", type=int, default=16)
    parser.add_argument("--max-hypotheses", type=int, default=16)
    parser.add_argument("--nms-radius-px", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=-1.0)
    parser.add_argument("--capture-threshold-px", type=float, default=8.0)
    parser.add_argument("--random-bank-seed", type=int, default=20260715)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    args = parser.parse_args(raw_argv)

    if args.batch_size <= 0 or args.candidate_chunk_size <= 0:
        parser.error("batch sizes must be positive")
    if args.grid_size <= 0 or args.per_anchor_topk <= 0 or args.max_hypotheses <= 0:
        parser.error("candidate budgets must be positive")
    if args.max_frames < 0 or args.bootstrap_samples <= 0:
        parser.error("max-frames must be non-negative and bootstrap-samples positive")
    if args.capture_threshold_px <= 0:
        parser.error("capture-threshold-px must be positive")

    data_path = resolved_path(args.data)
    vision_cache = resolved_path(args.vision_cache)
    prototype_bank_path = resolved_path(args.prototype_bank)
    checkpoint_paths = tuple(
        resolved_path(value)
        for value in (args.checkpoint if args.checkpoint else DEFAULT_CHECKPOINTS)
    )
    output_path = resolved_path(args.output)
    device = choose_device(args.device)

    checkpoint_payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in checkpoint_paths
    ]
    checkpoint_args = [dict(payload["args"]) for payload in checkpoint_payloads]
    action_stds = [
        np.asarray(payload["normalization"]["action_std"], dtype=np.float64)
        for payload in checkpoint_payloads
    ]
    policy_seeds = [int(values.get("seed", -1)) for values in checkpoint_args]
    if len(set(policy_seeds)) != len(policy_seeds):
        parser.error("policy checkpoints must have unique seeds")
    if set(policy_seeds) != {0, 1, 2} and args.max_frames <= 0:
        parser.error(
            "full development evaluation requires policy seeds {0,1,2}; "
            "a custom checkpoint set is allowed only with --max-frames smoke mode"
        )
    del checkpoint_payloads

    data = load_control_data(data_path)
    evaluation_indices = select_evaluation_indices(
        data,
        task=args.task,
        max_frames=args.max_frames,
    )
    selected_task = str(data.task[evaluation_indices[0]])
    validate_task_output_contract(
        selected_task=selected_task,
        output_is_explicit=output_argument_is_explicit(raw_argv),
    )
    cache_metadata, prototype_metadata = validate_artifacts(
        data_path=data_path,
        vision_cache=vision_cache,
        prototype_bank=prototype_bank_path,
        policy_args=checkpoint_args,
        evaluation_tasks=(selected_task,),
    )
    validate_policy_checkpoint_tasks(
        checkpoint_args,
        evaluation_task=selected_task,
    )
    cache_size = (
        int(cache_metadata["image_width"]),
        int(cache_metadata["image_height"]),
    )
    if cache_size != data.image_size:
        raise ValueError("vision cache and dataset image sizes differ")
    backbone = np.load(vision_cache / "backbone_features.npy", mmap_mode="r")
    prefix = np.load(vision_cache / "prefix.npy", mmap_mode="r")
    if backbone.shape[0] != len(data.frame) or prefix.shape[0] != len(data.frame):
        raise ValueError("vision cache sample count does not match dataset")
    descriptors = torch.from_numpy(
        np.load(prototype_bank_path / "prototypes.npy")
    ).to(device)
    feature_level = int(prototype_metadata["feature_level"])
    if feature_level < 0 or feature_level >= backbone.shape[1]:
        raise ValueError("prototype feature level is absent from vision cache")

    selected_indices, candidate_pool = generate_candidate_pool(
        data,
        indices=evaluation_indices,
        backbone=backbone,
        descriptors=descriptors,
        feature_level=feature_level,
        device=device,
        batch_size=args.batch_size,
        radius_px=args.radius_px,
        grid_size=args.grid_size,
        per_anchor_topk=args.per_anchor_topk,
        max_hypotheses=args.max_hypotheses,
        nms_radius_px=args.nms_radius_px,
        min_confidence=args.min_confidence,
    )
    random_pool = generate_random_joint_pool(
        data,
        indices=selected_indices,
        radius_px=args.radius_px,
        grid_size=args.grid_size,
        candidate_count=args.max_hypotheses,
        seed=args.random_bank_seed,
    )
    candidate_lookup = {
        int(frame_index): candidate_pool.points[position, 0]
        for position, frame_index in enumerate(selected_indices)
    }
    # A selected frame's previous index may fall just outside a smoke prefix.
    missing_previous = sorted(
        {
            int(data.previous_index[index])
            for index in selected_indices
            if int(data.previous_index[index]) not in candidate_lookup
        }
    )
    if missing_previous:
        raise RuntimeError(
            "selected frames are not a prefix-closed history set: "
            f"{missing_previous[:5]}"
        )

    all_rows: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] | None = None
    for checkpoint_path, policy_seed, policy_args, action_std in zip(
        checkpoint_paths,
        policy_seeds,
        checkpoint_args,
        action_stds,
        strict=True,
    ):
        model, loaded_args = load_action_chunk_policy(checkpoint_path, device)
        if dict(loaded_args) != policy_args:
            raise RuntimeError("checkpoint metadata changed between audit and load")
        adapter = getattr(model, "fcpg_adapter", None)
        if adapter is None:
            raise RuntimeError("loaded FCPG policy has no frozen native corrector")
        if str(adapter.vision.model_name) != str(cache_metadata["model_name"]):
            raise ValueError("native FCPG corrector model does not match vision cache")
        if tuple(int(value) for value in adapter.vision.selected_layers) != tuple(
            int(value) for value in cache_metadata["selected_layers"]
        ):
            raise ValueError("native FCPG corrector layers do not match vision cache")
        obs_mode = str(policy_args.get("obs_mode", "robot"))
        policy_rows: list[dict[str, Any]] = []
        policy_diagnostics: list[dict[str, Any]] = []
        for start in range(0, len(selected_indices), args.batch_size):
            frame_indices = selected_indices[start : start + args.batch_size]
            positions = np.arange(start, start + len(frame_indices), dtype=np.int64)
            base = build_deployable_base_batch(
                data,
                indices=frame_indices,
                obs_mode=obs_mode,
                backbone=backbone,
                prefix=prefix,
                device=device,
            )
            native_points = infer_native_current_points(
                model,
                base,
                image_size=data.image_size,
                device_type=device.type,
            )
            current_candidates = candidate_pool.points[positions]
            current_random = random_pool.points[positions]
            combined_points = np.concatenate(
                [
                    current_candidates,
                    current_random,
                    data.q_nominal[frame_indices, None],
                    native_points[:, None],
                    data.q_true[frame_indices, None],
                ],
                axis=1,
            )
            past_wide = np.stack(
                [candidate_lookup[int(data.previous_index[index])] for index in frame_indices]
            )
            actions = infer_policy_actions(
                model,
                base,
                past_wide=torch.from_numpy(past_wide).to(device),
                current_points=torch.from_numpy(combined_points).to(device),
                candidate_chunk_size=args.candidate_chunk_size,
                device_type=device.type,
            )
            local_pool = CandidatePool(
                points=candidate_pool.points[positions],
                score=candidate_pool.score[positions],
                valid=candidate_pool.valid[positions],
                grid_indices=candidate_pool.grid_indices[positions],
            )
            local_random_pool = CandidatePool(
                points=random_pool.points[positions],
                score=random_pool.score[positions],
                valid=random_pool.valid[positions],
                grid_indices=random_pool.grid_indices[positions],
            )
            rows, diagnostics = _mode_rows(
                data=data,
                indices=frame_indices,
                policy_seed=policy_seed,
                candidates=local_pool,
                random_candidates=local_random_pool,
                native_points=native_points,
                actions=actions,
                action_std=action_std,
                capture_threshold_px=args.capture_threshold_px,
            )
            policy_rows.extend(rows)
            policy_diagnostics.extend(diagnostics)
        all_rows.extend(policy_rows)
        if candidate_diagnostics is None:
            candidate_diagnostics = policy_diagnostics
        elif policy_diagnostics != candidate_diagnostics:
            raise RuntimeError("policy-independent candidate diagnostics changed by seed")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert candidate_diagnostics is not None
    smoke_only = args.max_frames > 0
    inferential_samples = 0 if smoke_only else args.bootstrap_samples
    payload = {
        "contract": {
            "status": "posthoc_development_not_confirmatory",
            "training_performed": False,
            "same_physical_snapshot": True,
            "observation_horizon": 2,
            "past_grounding": "wide_hard_top1_for_every_mode",
            "current_grounding_only_intervention": True,
            "q_true_usage": (
                "offline_teacher_and_oracle_label_only; absent_from_candidate_generation_"
                "and_deployable_policy_batch"
            ),
            "candidate_generator_inputs": (
                "current_frozen_visual_feature_map, control_q_nominal, frozen_control_prototypes"
            ),
            "policy_parameters_updated": False,
            "smoke_only": smoke_only,
            "crossed_confidence_intervals_reported": not smoke_only,
            "claim_boundary": (
                "policy-internal control-equivalence headroom only; not rollout success, "
                "a deployable selector, or evidence for LeWM"
            ),
        },
        "inputs": {
            "data": str(data_path),
            "vision_cache": str(vision_cache),
            "prototype_bank": str(prototype_bank_path),
            "prototype_training_source": str(
                resolved_path(str(prototype_metadata["source_data"]))
            ),
            "prototype_sha256": sha256_file(prototype_bank_path / "prototypes.npy"),
            "checkpoints": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "policy_seed": seed,
                    "obs_mode": str(values.get("obs_mode", "robot")),
                    "native_fcpg_checkpoint": str(
                        resolved_path(str(values["fcpg_checkpoint"]))
                    ),
                    "action_std_from_checkpoint": std.tolist(),
                }
                for path, seed, values, std in zip(
                    checkpoint_paths,
                    policy_seeds,
                    checkpoint_args,
                    action_stds,
                    strict=True,
                )
            ],
            "candidate_settings": {
                "radius_px": float(args.radius_px),
                "grid_size": int(args.grid_size),
                "per_anchor_topk": int(args.per_anchor_topk),
                "max_hypotheses": int(args.max_hypotheses),
                "nms_radius_px": float(args.nms_radius_px),
                "min_confidence": float(args.min_confidence),
                "capture_threshold_px": float(args.capture_threshold_px),
                "random_bank_seed": int(args.random_bank_seed),
                "random_bank_definition": (
                    "unique joint grid tuples sampled without replacement on the same "
                    "radius/grid as visual candidates"
                ),
            },
            "selected_frames": int(len(selected_indices)),
            "selected_task": selected_task,
            "max_frames": int(args.max_frames),
            "bootstrap_samples": int(inferential_samples),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "candidate_coverage": summarize_candidate_coverage(
            candidate_diagnostics,
            bootstrap_samples=inferential_samples,
            bootstrap_seed=args.bootstrap_seed + 10_000,
        ),
        "summary": summarize_rows(
            all_rows,
            bootstrap_samples=inferential_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        "paired_comparisons": summarize_paired_comparisons(
            all_rows,
            bootstrap_samples=inferential_samples,
            bootstrap_seed=args.bootstrap_seed + 20_000,
        ),
        "candidate_rows": candidate_diagnostics,
        "rows": all_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ready = _json_ready(payload)
    output_path.write_text(
        json.dumps(ready, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(ready["candidate_coverage"], indent=2))
    for mode in MODE_ORDER:
        metric = ready["summary"][mode]["action_xyz_normalized_mse_h8"]
        print(f"{mode}: {metric}")
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
