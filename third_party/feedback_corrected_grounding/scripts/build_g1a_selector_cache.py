#!/usr/bin/env python3
"""Build the post-hoc G1A deployment/label cache for candidate selection.

The builder freezes the r64/g17/L16/M16 prototype candidate generator, three
action policies, and the oracle-free 10k GroundedLeWM.  Candidate inputs and
offline control-equivalence labels are written to separate NPZ files.  Ground
truth keypoints and oracle actions are used transiently to construct labels and
are never written to the deployment partition.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fcpg.action_chunk_policy import load_action_chunk_policy  # noqa: E402
from fcpg.lewm_runtime import load_lewm_grounded_checkpoint  # noqa: E402
from fcpg.modules import sample_feature_map_at_pixels  # noqa: E402
from fcpg.observations import select_policy_observation  # noqa: E402
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402
from scripts.evaluate_prototype_control_candidates import (  # noqa: E402
    DEFAULT_CHECKPOINTS,
    DEFAULT_DATA,
    DEFAULT_PROTOTYPE_BANK,
    DEFAULT_VISION_CACHE,
    CandidatePool,
    ControlData,
    action_distance_metrics,
    build_deployable_base_batch,
    choose_device,
    infer_policy_actions,
    load_control_data,
    resolved_path,
    sha256_file,
    validate_artifacts,
    validate_policy_checkpoint_tasks,
)


DEFAULT_LEWM_CHECKPOINT = (
    "outputs/lewm_grounded/pixel_observer_base_online_sig002_seed3072/"
    "lewm_grounded_latest.pt"
)
CANONICAL_LEWM_SHA256 = (
    "49f252eb526a61aad0bf29b9b3400b1b2bbfa2b7bb83e35164d716f9315435a9"
)
DEFAULT_HISTORY_CORRECTOR_CHECKPOINT = (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
CANONICAL_HISTORY_CORRECTOR_SHA256 = (
    "3cfe621bbfb5b87962a3c569d6ebf9d4d702e0483adeebd45a4e00bb0a16ee79"
)
CANONICAL_CONTROL_KEYPOINT_SITES = (
    "leftEndEffector",
    "rightEndEffector",
    "endEffector",
)
PRIMARY_HISTORY_SOURCE = "native_fcpg"
DEFAULT_OUTPUT_DIR = (
    "outputs/g1a_selector_cache/"
    "pick_rot10_r64g17_l16m16_online10k_matched_history_native_labels_"
    "splitlocal_v3_development"
)

RADIUS_PX = 64.0
GRID_SIZE = 17
PER_ANCHOR_TOPK = 16
MAX_HYPOTHESES = 16
NMS_RADIUS_PX = 0.0
MIN_CONFIDENCE = -1.0
FRAME_SKIP = 5
HISTORY_SIZE = 3
ACTION_DIM = 4
TRANSITION_ACTION_DIM = FRAME_SKIP * ACTION_DIM
POLICY_HORIZON = 8
SPLIT_SEED = 20260715
SHUFFLE_SEED = 20260716
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SPLIT_NAMES = ("train", "validation", "test")
DEPLOYMENT_FORBIDDEN_SUBSTRINGS = (
    "q_true",
    "qtrue",
    "valid_true",
    "oracle",
    "teacher",
    "privileged",
    "ground_truth",
    "reward",
    "success",
    "label",
    "epe",
    "future",
    "target",
    "shuffled",
)


@dataclass(frozen=True)
class StrictHistory:
    current_indices: np.ndarray
    state_indices: np.ndarray
    action_indices: np.ndarray
    action_blocks: np.ndarray


@dataclass(frozen=True)
class LeWMVisualCache:
    raw_indices: np.ndarray
    feature_maps: np.ndarray
    global_features: np.ndarray


@dataclass(frozen=True)
class CandidateTokens:
    summary_token: np.ndarray
    summary_embedding: np.ndarray
    anchor_tokens: np.ndarray


def _scalar_string(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.shape == () or array.size == 1:
        return str(array.reshape(-1)[0])
    raise ValueError(f"{field} must be a scalar string")


def load_rgb_and_alignment(path: Path) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as raw:
        if "rgb" not in raw.files or "observation_action_alignment" not in raw.files:
            raise KeyError("G1A data requires rgb and observation_action_alignment")
        rgb = np.asarray(raw["rgb"], dtype=np.uint8)
        alignment = _scalar_string(
            raw["observation_action_alignment"], "observation_action_alignment"
        )
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape [N,H,W,3]")
    if alignment != "pre_action":
        raise ValueError("G1A requires observation_action_alignment='pre_action'")
    return rgb, alignment


def load_control_keypoint_sites(path: Path) -> tuple[str, ...]:
    """Load only the deployable semantic control-anchor order."""

    with np.load(path, allow_pickle=False) as raw:
        if "control_keypoint_site" not in raw.files:
            raise KeyError("G1A data requires control_keypoint_site")
        sites = tuple(str(value) for value in raw["control_keypoint_site"].tolist())
    if sites != CANONICAL_CONTROL_KEYPOINT_SITES:
        raise ValueError(
            "G1A control-anchor order differs from the canonical FCPG corrector: "
            f"{sites}"
        )
    return sites


def strict_history_from_data(
    data: ControlData,
    *,
    frame_skip: int = FRAME_SKIP,
    history_size: int = HISTORY_SIZE,
) -> StrictHistory:
    """Build strictly pre-current LeWM states and executed action blocks."""

    if frame_skip <= 0 or history_size <= 0:
        raise ValueError("frame_skip and history_size must be positive")
    lookup = {
        (str(task), int(episode), int(frame)): index
        for index, (task, episode, frame) in enumerate(
            zip(data.task, data.episode, data.frame, strict=True)
        )
    }
    current: list[int] = []
    states: list[list[int]] = []
    action_indices: list[list[list[int]]] = []
    action_blocks: list[np.ndarray] = []
    for index, (task, episode, frame) in enumerate(
        zip(data.task, data.episode, data.frame, strict=True)
    ):
        frame_id = int(frame)
        first_state = frame_id - history_size * frame_skip
        if first_state < 0:
            continue
        state_frames = [first_state + offset * frame_skip for offset in range(history_size)]
        try:
            state_row = [lookup[(str(task), int(episode), value)] for value in state_frames]
            block_rows = [
                [
                    lookup[(str(task), int(episode), value)]
                    for value in range(start, start + frame_skip)
                ]
                for start in state_frames
            ]
        except KeyError:
            continue
        if block_rows[-1][-1] != lookup[(str(task), int(episode), frame_id - 1)]:
            raise RuntimeError("strict action history does not end at current-1")
        current.append(index)
        states.append(state_row)
        action_indices.append(block_rows)
        action_blocks.append(data.action[np.asarray(block_rows)].reshape(history_size, -1))
    if not current:
        raise ValueError("dataset has no frame with a complete strict LeWM history")
    action_array = np.asarray(action_blocks, dtype=np.float32)
    expected_action = (len(current), history_size, frame_skip * data.action.shape[1])
    if action_array.shape != expected_action:
        raise ValueError(
            f"strict action blocks must have shape {expected_action}, got {action_array.shape}"
        )
    return StrictHistory(
        current_indices=np.asarray(current, dtype=np.int64),
        state_indices=np.asarray(states, dtype=np.int64),
        action_indices=np.asarray(action_indices, dtype=np.int64),
        action_blocks=action_array,
    )


def select_smoke_snapshots(
    history: StrictHistory,
    data: ControlData,
    max_snapshots: int,
) -> np.ndarray:
    if max_snapshots <= 0 or max_snapshots >= len(history.current_indices):
        return np.arange(len(history.current_indices), dtype=np.int64)
    ordering = sorted(
        range(len(history.current_indices)),
        key=lambda position: (
            int(data.frame[history.current_indices[position]]),
            str(data.task[history.current_indices[position]]),
            int(data.episode[history.current_indices[position]]),
        ),
    )
    return np.asarray(ordering[:max_snapshots], dtype=np.int64)


def subset_history(history: StrictHistory, positions: np.ndarray) -> StrictHistory:
    selected = np.asarray(positions, dtype=np.int64)
    return StrictHistory(
        current_indices=history.current_indices[selected],
        state_indices=history.state_indices[selected],
        action_indices=history.action_indices[selected],
        action_blocks=history.action_blocks[selected],
    )


def validate_selected_policy_checkpoint_tasks(
    policy_args: Sequence[Mapping[str, object]],
    selected_tasks: Sequence[object] | np.ndarray,
) -> None:
    """Require every policy training provenance to cover every selected task."""

    tasks = tuple(
        sorted({str(task) for task in np.asarray(selected_tasks).reshape(-1)})
    )
    if not tasks:
        raise ValueError("selected G1A cache must contain at least one task")
    for task in tasks:
        validate_policy_checkpoint_tasks(policy_args, evaluation_task=task)


def episode_splits(
    task: np.ndarray,
    episode: np.ndarray,
    *,
    seed: int = SPLIT_SEED,
    smoke_only: bool = False,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    keys = sorted({f"{str(t)}:{int(e)}" for t, e in zip(task, episode, strict=True)})
    if smoke_only:
        return np.full(len(task), "smoke", dtype="<U10"), {"smoke": keys}
    if len(keys) < 5:
        raise ValueError("full G1A cache requires at least five physical episodes")
    rng = np.random.default_rng(int(seed))
    shuffled = [keys[index] for index in rng.permutation(len(keys))]
    train_count = max(1, int(np.floor(SPLIT_FRACTIONS[0] * len(keys))))
    validation_count = max(1, int(np.floor(SPLIT_FRACTIONS[1] * len(keys))))
    if train_count + validation_count >= len(keys):
        validation_count = 1
        train_count = len(keys) - 2
    groups = {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }
    mapping = {key: name for name, values in groups.items() for key in values}
    split = np.asarray(
        [mapping[f"{str(t)}:{int(e)}"] for t, e in zip(task, episode, strict=True)],
        dtype="<U10",
    )
    return split, groups


def executed_action_derangement(
    task: np.ndarray,
    episode: np.ndarray,
    frame: np.ndarray,
    split: np.ndarray,
    action_blocks: np.ndarray,
    *,
    seed: int = SHUFFLE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Swap only the current transition within each data split.

    The first ``H-1`` action blocks remain factual because they connect the
    fixed history embeddings.  Only the final ``t-5..t-1`` block is
    intervened when probing the current-state prior.  Sources match the
    target split, task, and frame but come from another physical episode.
    """

    count = len(task)
    if (
        episode.shape != (count,)
        or frame.shape != (count,)
        or split.shape != (count,)
        or action_blocks.shape[0] != count
    ):
        raise ValueError("derangement inputs have inconsistent leading dimensions")
    groups: dict[tuple[str, str, int], list[int]] = {}
    for index, (split_name, task_name, frame_id) in enumerate(
        zip(split, task, frame, strict=True)
    ):
        groups.setdefault(
            (str(split_name), str(task_name), int(frame_id)), []
        ).append(index)
    permutation = np.empty(count, dtype=np.int64)
    for group_number, group_key in enumerate(sorted(groups)):
        indices = np.asarray(groups[group_key], dtype=np.int64)
        if len(indices) < 2:
            raise ValueError(
                "cannot shuffle executed actions for singleton "
                f"split/task/frame group {group_key}"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), int(group_key[2]), len(indices), group_number]
            )
        )
        ordered = indices[rng.permutation(len(indices))]
        source = np.roll(ordered, 1)
        permutation[ordered] = source
    if bool(np.any(permutation == np.arange(count))):
        raise RuntimeError("executed-action derangement contains a fixed point")
    if bool(np.any(episode[permutation] == episode)):
        raise RuntimeError("executed actions were not moved to another physical episode")
    if (
        not np.array_equal(split[permutation], split)
        or not np.array_equal(task[permutation], task)
        or not np.array_equal(frame[permutation], frame)
    ):
        raise RuntimeError(
            "executed-action derangement changed split/task/frame support"
        )
    shuffled = np.array(action_blocks, dtype=np.float32, copy=True)
    shuffled[:, -1] = action_blocks[permutation, -1]
    return shuffled, permutation


@torch.inference_mode()
def generate_candidate_pool_at_indices(
    data: ControlData,
    *,
    indices: np.ndarray,
    backbone: np.ndarray,
    descriptors: torch.Tensor,
    feature_level: int,
    device: torch.device,
    batch_size: int,
) -> CandidatePool:
    from fcpg.geometric_baselines import generate_joint_prototype_hypotheses

    selected = np.asarray(indices, dtype=np.int64)
    points: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    grid_indices: list[np.ndarray] = []
    for start in range(0, len(selected), batch_size):
        rows = selected[start : start + batch_size]
        feature = torch.from_numpy(
            np.array(backbone[rows, feature_level], dtype=np.float16, copy=True)
        ).to(device)
        if device.type == "cpu":
            feature = feature.float()
        nominal = torch.from_numpy(data.q_nominal[rows]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = generate_joint_prototype_hypotheses(
                feature,
                nominal,
                descriptors,
                image_size=data.image_size,
                radius_px=RADIUS_PX,
                grid_size=GRID_SIZE,
                per_anchor_topk=PER_ANCHOR_TOPK,
                max_hypotheses=MAX_HYPOTHESES,
                nms_radius_px=NMS_RADIUS_PX,
                min_confidence=MIN_CONFIDENCE,
            )
        points.append(output.points_px.float().cpu().numpy())
        scores.append(output.joint_score.float().cpu().numpy())
        valid.append(output.valid.cpu().numpy().astype(bool))
        grid_indices.append(output.grid_indices.cpu().numpy().astype(np.int64))
    return CandidatePool(
        points=np.concatenate(points).astype(np.float32, copy=False),
        score=np.concatenate(scores).astype(np.float32, copy=False),
        valid=np.concatenate(valid),
        grid_indices=np.concatenate(grid_indices),
    )


def candidate_lookup(
    indices: np.ndarray,
    pool: CandidatePool,
) -> dict[int, int]:
    if len(indices) != len(pool.points):
        raise ValueError("candidate index/pool lengths differ")
    result = {int(raw): position for position, raw in enumerate(indices)}
    if len(result) != len(indices):
        raise ValueError("candidate raw indices contain duplicates")
    return result


def select_pool_rows(pool: CandidatePool, positions: np.ndarray) -> CandidatePool:
    return CandidatePool(
        points=pool.points[positions],
        score=pool.score[positions],
        valid=pool.valid[positions],
        grid_indices=pool.grid_indices[positions],
    )


@torch.inference_mode()
def sample_candidate_visual_descriptors(
    backbone: np.ndarray,
    *,
    feature_level: int,
    raw_indices: np.ndarray,
    points: np.ndarray,
    image_size: tuple[int, int],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows = np.asarray(raw_indices, dtype=np.int64)
    result: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        feature = torch.from_numpy(
            np.array(backbone[rows[start:stop], feature_level], dtype=np.float16, copy=True)
        ).to(device)
        locations = torch.from_numpy(points[start:stop]).to(device)
        if device.type == "cpu":
            feature = feature.float()
        batch, hypotheses, anchors, _ = locations.shape
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            sampled = sample_feature_map_at_pixels(
                feature,
                locations.reshape(batch, hypotheses * anchors, 2),
                image_size=image_size,
                padding_mode="border",
            ).reshape(batch, hypotheses, anchors, -1)
        result.append(sampled.float().cpu().numpy().astype(np.float16))
    return np.concatenate(result)


def candidate_geometry(
    data: ControlData,
    *,
    raw_indices: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(raw_indices, dtype=np.int64)
    if points.shape[0] != len(rows) or points.shape[-2:] != (3, 2):
        raise ValueError("candidate points must have shape [N,M,3,2]")
    width, height = data.image_size
    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    nominal = data.q_nominal[rows, None] / scale
    nominal = np.broadcast_to(nominal, points.shape)
    residual = (points - data.q_nominal[rows, None]) / scale
    xyz = np.broadcast_to(
        data.xyz[rows, None], (*points.shape[:-1], 3)
    )
    identity = np.broadcast_to(
        np.eye(3, dtype=np.float32)[None, None], (*points.shape[:-1], 3)
    )
    combined_valid = valid & data.valid_nominal[rows, None]
    geometry = np.concatenate(
        [
            nominal,
            residual,
            xyz,
            identity,
            combined_valid[..., None].astype(np.float32),
        ],
        axis=-1,
    )
    if geometry.shape != (*points.shape[:-1], 11):
        raise RuntimeError("candidate geometry contract has the wrong shape")
    return geometry.astype(np.float32, copy=False)


@torch.inference_mode()
def extract_lewm_visual_cache(
    model: Any,
    rgb: np.ndarray,
    *,
    raw_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype,
) -> LeWMVisualCache:
    rows = np.asarray(sorted(set(np.asarray(raw_indices, dtype=np.int64).tolist())))
    maps: list[np.ndarray] = []
    globals_: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        pixels = torch.from_numpy(
            np.transpose(rgb[selected].astype(np.float32) / 255.0, (0, 3, 1, 2))
        ).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            output = model.frame_encoder.vision_encoder(pixels)
        maps.append(output.feature_map.float().cpu().numpy())
        globals_.append(output.global_feature.float().cpu().numpy())
    return LeWMVisualCache(
        raw_indices=rows,
        feature_maps=np.concatenate(maps),
        global_features=np.concatenate(globals_),
    )


def _cache_positions(cache: LeWMVisualCache, raw_indices: np.ndarray) -> np.ndarray:
    lookup = {int(raw): index for index, raw in enumerate(cache.raw_indices)}
    try:
        return np.asarray([lookup[int(raw)] for raw in raw_indices], dtype=np.int64)
    except KeyError as error:
        raise KeyError(f"LeWM visual cache lacks raw frame {error.args[0]}") from error


@torch.inference_mode()
def encode_candidate_tokens(
    model: Any,
    cache: LeWMVisualCache,
    data: ControlData,
    *,
    raw_indices: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
    candidate_chunk_size: int,
    amp_dtype: torch.dtype,
) -> CandidateTokens:
    rows = np.asarray(raw_indices, dtype=np.int64)
    positions = _cache_positions(cache, rows)
    summaries: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    for start in range(0, points.shape[1], candidate_chunk_size):
        local = points[:, start : start + candidate_chunk_size]
        local_valid = valid[:, start : start + candidate_chunk_size]
        hypotheses = local.shape[1]
        feature = torch.from_numpy(
            np.array(cache.feature_maps[positions], dtype=np.float32, copy=True)
        ).to(device)
        global_visual = torch.from_numpy(
            np.array(cache.global_features[positions], dtype=np.float32, copy=True)
        ).to(device)
        if device.type == "cpu":
            feature = feature.float()
            global_visual = global_visual.float()
        points_tensor = torch.from_numpy(local).to(device)
        batch, _, anchors_per_candidate, _ = points_tensor.shape
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            sampled = sample_feature_map_at_pixels(
                feature,
                points_tensor.reshape(batch, hypotheses * anchors_per_candidate, 2),
                image_size=data.image_size,
                padding_mode="border",
            ).reshape(batch * hypotheses, anchors_per_candidate, -1)
            normalized_obs = (
                torch.from_numpy(data.obs[rows, :4]).to(device)
                - model.proprio_mean
            ) / model.proprio_std
            tokens = model.frame_encoder.forward_token_sequence_from_features(
                sampled_visual=sampled,
                global_visual=global_visual[:, None]
                .expand(-1, hypotheses, -1)
                .reshape(batch * hypotheses, -1),
                robot_obs=normalized_obs[:, None]
                .expand(-1, hypotheses, -1)
                .reshape(batch * hypotheses, -1),
                q_nominal_px=torch.from_numpy(data.q_nominal[rows]).to(device)[:, None]
                .expand(-1, hypotheses, -1, -1)
                .reshape(batch * hypotheses, 3, 2),
                q_grounded_px=points_tensor.reshape(batch * hypotheses, 3, 2),
                keypoints_xyz=torch.from_numpy(data.xyz[rows]).to(device)[:, None]
                .expand(-1, hypotheses, -1, -1)
                .reshape(batch * hypotheses, 3, 3),
                valid=torch.from_numpy(local_valid).to(device).reshape(
                    batch * hypotheses, 3
                ),
                image_size=data.image_size,
            )
            summary = tokens[:, 0]
            embedding = model.projector(summary)
        token_shape = (batch, hypotheses, -1)
        summaries.append(
            summary.reshape(*token_shape).float().cpu().numpy()
        )
        embeddings.append(
            embedding.reshape(*token_shape).float().cpu().numpy()
        )
        anchors.append(
            tokens[:, 3:6]
            .reshape(batch, hypotheses, 3, -1)
            .float()
            .cpu()
            .numpy()
        )
    return CandidateTokens(
        summary_token=np.concatenate(summaries, axis=1),
        summary_embedding=np.concatenate(embeddings, axis=1),
        anchor_tokens=np.concatenate(anchors, axis=1),
    )


def grounding_valid_mask(
    points: np.ndarray,
    valid_nominal: np.ndarray,
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    nominal = np.asarray(valid_nominal, dtype=bool)
    if values.ndim != 3 or values.shape[-2:] != (3, 2):
        raise ValueError("history grounding points must have shape [N,3,2]")
    if nominal.shape != values.shape[:-1]:
        raise ValueError("nominal validity must match history grounding points")
    width, height = image_size
    finite = np.isfinite(values).all(axis=-1)
    inside = (
        (values[..., 0] >= 0.0)
        & (values[..., 0] <= float(width - 1))
        & (values[..., 1] >= 0.0)
        & (values[..., 1] <= float(height - 1))
    )
    return nominal & finite & inside


@torch.inference_mode()
def infer_canonical_fcpg_history_points(
    adapter: Any,
    *,
    obs_mode: str,
    backbone: np.ndarray,
    prefix: np.ndarray,
    data: ControlData,
    raw_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Infer history grounding without accepting any privileged array."""

    rows = np.asarray(raw_indices, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("history corrector indices must be a non-empty vector")
    corrected: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        feature = torch.from_numpy(
            np.array(backbone[selected], dtype=np.float16, copy=True)
        ).to(device)
        visual_prefix = torch.from_numpy(
            np.array(prefix[selected], dtype=np.float16, copy=True)
        ).to(device)
        if device.type == "cpu":
            feature = feature.float()
            visual_prefix = visual_prefix.float()
        q_nominal = torch.from_numpy(data.q_nominal[selected]).to(device)
        xyz = torch.from_numpy(data.xyz[selected]).to(device)
        robot_obs = torch.from_numpy(
            select_policy_observation(data.obs[selected], obs_mode)
        ).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = adapter.forward_from_backbone_features(
                feature,
                visual_prefix,
                q_nominal,
                xyz,
                robot_obs,
                image_size=data.image_size,
            )
        values = output.q_corrected_px.detach().float().cpu().numpy()
        if values.shape != (len(selected), 3, 2):
            raise ValueError("canonical FCPG corrector returned an invalid history shape")
        corrected.append(values)
    result = np.concatenate(corrected).astype(np.float32, copy=False)
    if not bool(np.isfinite(result).all()):
        raise ValueError("canonical FCPG corrector returned non-finite history points")
    return result


@torch.inference_mode()
def encode_history_tokens(
    model: Any,
    cache: LeWMVisualCache,
    data: ControlData,
    *,
    raw_indices: np.ndarray,
    grounding_points: np.ndarray,
    grounding_valid: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(raw_indices, dtype=np.int64)
    if grounding_points.shape != (len(rows), 3, 2):
        raise ValueError("history grounding points must have shape [N,3,2]")
    if grounding_valid.shape != (len(rows), 3):
        raise ValueError("history grounding validity must have shape [N,3]")
    positions = _cache_positions(cache, rows)
    summaries: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        selected = rows[start:stop]
        selected_pos = positions[start:stop]
        feature = torch.from_numpy(
            np.array(cache.feature_maps[selected_pos], dtype=np.float32, copy=True)
        ).to(device)
        global_visual = torch.from_numpy(
            np.array(cache.global_features[selected_pos], dtype=np.float32, copy=True)
        ).to(device)
        if device.type == "cpu":
            feature = feature.float()
            global_visual = global_visual.float()
        point_tensor = torch.from_numpy(grounding_points[start:stop]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            sampled = sample_feature_map_at_pixels(
                feature,
                point_tensor,
                image_size=data.image_size,
                padding_mode="border",
            )
            normalized_obs = (
                torch.from_numpy(data.obs[selected, :4]).to(device)
                - model.proprio_mean
            ) / model.proprio_std
            tokens = model.frame_encoder.forward_token_sequence_from_features(
                sampled_visual=sampled,
                global_visual=global_visual,
                robot_obs=normalized_obs,
                q_nominal_px=torch.from_numpy(data.q_nominal[selected]).to(device),
                q_grounded_px=point_tensor,
                keypoints_xyz=torch.from_numpy(data.xyz[selected]).to(device),
                valid=torch.from_numpy(grounding_valid[start:stop]).to(device),
                image_size=data.image_size,
            )
            summary = tokens[:, 0]
            embedding = model.projector(summary)
        summaries.append(summary.float().cpu().numpy())
        embeddings.append(embedding.float().cpu().numpy())
        anchors.append(tokens[:, 3:6].float().cpu().numpy())
    return np.concatenate(summaries), np.concatenate(embeddings), np.concatenate(anchors)


@torch.inference_mode()
def predict_current_summary_prior(
    model: Any,
    history_embedding: np.ndarray,
    action_blocks: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    if history_embedding.ndim != 3 or history_embedding.shape[1:] != (3, 192):
        raise ValueError("history embedding must have shape [N,3,192]")
    if action_blocks.shape != (len(history_embedding), 3, 20):
        raise ValueError("executed action blocks must have shape [N,3,20]")
    output: list[np.ndarray] = []
    for start in range(0, len(history_embedding), batch_size):
        stop = min(start + batch_size, len(history_embedding))
        embedding = torch.from_numpy(
            history_embedding[start:stop].astype(np.float32)
        ).to(device)
        actions = torch.from_numpy(action_blocks[start:stop]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            predicted = model.rollout_embeddings(
                history_embedding=embedding,
                history_action=actions[:, :2],
                future_action=actions[:, 2:3, None],
            )[:, 0, 0]
        output.append(predicted.float().cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def build_policy_labels(
    *,
    data: ControlData,
    current_indices: np.ndarray,
    current_pool: CandidatePool,
    past_grounding_points: np.ndarray,
    backbone: np.ndarray,
    prefix: np.ndarray,
    checkpoint_paths: Sequence[Path],
    checkpoint_args: Sequence[Mapping[str, object]],
    action_stds: Sequence[np.ndarray],
    policy_seeds: Sequence[int],
    device: torch.device,
    batch_size: int,
    candidate_chunk_size: int,
) -> np.ndarray:
    expected_past_shape = (len(current_indices), 3, 2)
    if past_grounding_points.shape != expected_past_shape:
        raise ValueError(
            "offline policy-label history must have shape "
            f"{expected_past_shape}, got {past_grounding_points.shape}"
        )
    if not bool(np.isfinite(past_grounding_points).all()):
        raise ValueError("offline policy-label history contains non-finite points")
    label_by_seed: list[np.ndarray] = []
    for checkpoint_path, expected_args, action_std, _seed in zip(
        checkpoint_paths,
        checkpoint_args,
        action_stds,
        policy_seeds,
        strict=True,
    ):
        model, loaded_args = load_action_chunk_policy(checkpoint_path, device)
        if dict(loaded_args) != dict(expected_args):
            raise RuntimeError("policy checkpoint metadata changed during G1A build")
        seed_labels: list[np.ndarray] = []
        for start in range(0, len(current_indices), batch_size):
            rows = current_indices[start : start + batch_size]
            local = slice(start, start + len(rows))
            base = build_deployable_base_batch(
                data,
                indices=rows,
                obs_mode=str(expected_args.get("obs_mode", "robot")),
                backbone=backbone,
                prefix=prefix,
                device=device,
            )
            past_native_fcpg = past_grounding_points[start : start + len(rows)]
            combined = np.concatenate(
                [current_pool.points[local], data.q_true[rows, None]], axis=1
            )
            actions = infer_policy_actions(
                model,
                base,
                past_wide=torch.from_numpy(past_native_fcpg).to(device),
                current_points=torch.from_numpy(combined).to(device),
                candidate_chunk_size=candidate_chunk_size,
                device_type=device.type,
            )
            candidate_actions = actions[:, :-1]
            teacher = actions[:, -1]
            metrics = action_distance_metrics(
                candidate_actions,
                teacher,
                np.asarray(action_std, dtype=np.float64),
                POLICY_HORIZON,
            )
            seed_labels.append(
                metrics["xyz_normalized_mse"].astype(np.float32, copy=False)
            )
        label_by_seed.append(np.concatenate(seed_labels))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.stack(label_by_seed, axis=1)


def assert_label_history_contract(contract: Mapping[str, object]) -> None:
    """Require offline policy labels to use the deployment-primary history."""

    primary = str(contract.get("history_grounding_primary", ""))
    label_source = str(contract.get("label_history_source", ""))
    if not primary or not label_source:
        raise ValueError("history contract must name primary and label history sources")
    if label_source != primary:
        raise ValueError(
            "offline label history source differs from deployment-primary history: "
            f"{label_source!r} != {primary!r}"
        )
    if contract.get("label_history_matches_primary") is not True:
        raise ValueError(
            "label_history_matches_primary must explicitly certify the source match"
        )


def assert_deployment_partition(values: Mapping[str, np.ndarray]) -> None:
    lowered = [str(key).lower() for key in values]
    leaks = sorted(
        key
        for key in lowered
        if any(forbidden in key for forbidden in DEPLOYMENT_FORBIDDEN_SUBSTRINGS)
    )
    if leaks:
        raise ValueError(f"offline information leaked into deployment partition: {leaks}")
    leading = {value.shape[0] for value in values.values() if value.ndim > 0}
    if len(leading) != 1:
        raise ValueError(f"deployment arrays disagree on sample count: {leading}")
    for key, value in values.items():
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            raise ValueError(f"deployment array {key} contains non-finite values")


def _write_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    np.savez(path, **values)


def _resolved_checkpoint_inputs(
    checkpoint_paths: Sequence[Path],
) -> tuple[list[dict[str, object]], list[np.ndarray], list[int]]:
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in checkpoint_paths
    ]
    args = [dict(payload["args"]) for payload in payloads]
    stds = [
        np.asarray(payload["normalization"]["action_std"], dtype=np.float64)
        for payload in payloads
    ]
    seeds = [int(value.get("seed", -1)) for value in args]
    return args, stds, seeds


def _load_canonical_history_corrector(
    path: Path,
    *,
    device: torch.device,
    cache_metadata: Mapping[str, Any],
    control_sites: tuple[str, ...],
) -> tuple[Any, Mapping[str, object]]:
    if sha256_file(path) != CANONICAL_HISTORY_CORRECTOR_SHA256:
        raise ValueError("history corrector does not match the canonical SHA256")
    if control_sites != CANONICAL_CONTROL_KEYPOINT_SITES:
        raise ValueError("history corrector control-anchor semantics do not match")
    adapter, args = load_fcpg_vit_adapter(path, device)
    if str(args.get("corrector_arch")) != "lewm":
        raise ValueError("canonical history corrector must use the LeWM architecture")
    if str(args.get("obs_mode")) != "robot":
        raise ValueError("canonical history corrector must use robot proprioception")
    if str(args.get("model_name")) != str(cache_metadata.get("model_name")):
        raise ValueError("history corrector ViT model differs from the visual cache")
    expected_layers = tuple(int(value) for value in cache_metadata["selected_layers"])
    actual_layers = tuple(int(value) for value in args.get("vision_selected_layers", ()))
    if actual_layers != expected_layers:
        raise ValueError("history corrector ViT layers differ from the visual cache")
    if int(adapter.grounding.num_keypoints) != 3:
        raise ValueError("canonical history corrector must use three control anchors")
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter, args


def _validate_lewm_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    model, resolved, checkpoint = load_lewm_grounded_checkpoint(path, device=device)
    world = checkpoint.get("model_config", {}).get("world_model", {})
    data = resolved.get("data", {})
    loss = resolved.get("loss", {})
    trainer = resolved.get("trainer", {})
    if int(checkpoint.get("step", -1)) != 10_000:
        raise ValueError("G1A requires the frozen 10k GroundedLeWM checkpoint")
    if data.get("target_mode") != "online" or world.get("target_mode") != "online":
        raise ValueError("G1A deployment base must use an online, oracle-free target")
    if float(loss.get("alignment_weight", -1.0)) != 0.0:
        raise ValueError("G1A deployment base must not use oracle alignment")
    if data.get("grounding_mode") != "fcpg":
        raise ValueError("G1A deployment base must be trained with FCPG grounding")
    if int(resolved.get("seed", -1)) != 3072:
        raise ValueError("G1A requires canonical GroundedLeWM seed 3072")
    if not bool(trainer.get("amp")) or trainer.get("amp_dtype") != "bfloat16":
        raise ValueError("G1A requires the canonical bfloat16 training contract")
    if int(world.get("history_size", -1)) != HISTORY_SIZE:
        raise ValueError("G1A LeWM history size mismatch")
    if int(world.get("action_dim", -1)) != TRANSITION_ACTION_DIM:
        raise ValueError("G1A LeWM transition action dimension mismatch")
    if int(model.frame_encoder.num_keypoints) != 3 or int(model.latent_dim) != 192:
        raise ValueError("G1A requires three 192D GroundedLeWM anchors")
    if model.frame_encoder.anchor_visual_mode != "single_corrected":
        raise ValueError("G1A requires single-corrected anchor visual tokens")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, resolved, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--vision-cache", default=DEFAULT_VISION_CACHE)
    parser.add_argument("--prototype-bank", default=DEFAULT_PROTOTYPE_BANK)
    parser.add_argument("--policy-checkpoint", action="append", default=[])
    parser.add_argument("--lewm-checkpoint", default=DEFAULT_LEWM_CHECKPOINT)
    parser.add_argument(
        "--history-corrector-checkpoint",
        default=DEFAULT_HISTORY_CORRECTOR_CHECKPOINT,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-chunk-size", type=int, default=4)
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="Development smoke only; selects same-frame snapshots across episodes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.candidate_chunk_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.max_snapshots < 0:
        raise ValueError("max-snapshots must be non-negative")

    data_path = resolved_path(args.data)
    vision_cache_path = resolved_path(args.vision_cache)
    prototype_path = resolved_path(args.prototype_bank)
    policy_paths = tuple(
        resolved_path(value)
        for value in (
            args.policy_checkpoint if args.policy_checkpoint else DEFAULT_CHECKPOINTS
        )
    )
    lewm_path = resolved_path(args.lewm_checkpoint)
    if sha256_file(lewm_path) != CANONICAL_LEWM_SHA256:
        raise ValueError("G1A LeWM checkpoint does not match the canonical SHA256")
    history_corrector_path = resolved_path(args.history_corrector_checkpoint)
    if sha256_file(history_corrector_path) != CANONICAL_HISTORY_CORRECTOR_SHA256:
        raise ValueError("G1A history corrector does not match the canonical SHA256")
    output_dir = resolved_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty G1A cache: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    checkpoint_args, action_stds, policy_seeds = _resolved_checkpoint_inputs(policy_paths)
    smoke_only = args.max_snapshots > 0
    if len(set(policy_seeds)) != len(policy_seeds):
        raise ValueError("policy checkpoints must have unique seeds")
    if not smoke_only and set(policy_seeds) != {0, 1, 2}:
        raise ValueError("full G1A cache requires policy seeds {0,1,2}")
    data = load_control_data(data_path)
    cache_metadata, prototype_metadata = validate_artifacts(
        data_path=data_path,
        vision_cache=vision_cache_path,
        prototype_bank=prototype_path,
        policy_args=checkpoint_args,
        evaluation_tasks=tuple(sorted(set(str(value) for value in data.task.tolist()))),
    )
    control_sites = load_control_keypoint_sites(data_path)
    rgb, alignment = load_rgb_and_alignment(data_path)
    if len(rgb) != len(data.frame):
        raise ValueError("RGB and control data sample counts differ")
    history = strict_history_from_data(data)
    selected_positions = select_smoke_snapshots(history, data, args.max_snapshots)
    history = subset_history(history, selected_positions)

    selected_task = data.task[history.current_indices]
    validate_selected_policy_checkpoint_tasks(checkpoint_args, selected_task)
    selected_episode = data.episode[history.current_indices]
    selected_frame = data.frame[history.current_indices]
    split, split_episodes = episode_splits(
        selected_task,
        selected_episode,
        smoke_only=smoke_only,
    )
    shuffled_actions, shuffle_permutation = executed_action_derangement(
        selected_task,
        selected_episode,
        selected_frame,
        split,
        history.action_blocks,
    )

    backbone = np.load(
        vision_cache_path / "backbone_features.npy", mmap_mode="r", allow_pickle=False
    )
    prefix = np.load(
        vision_cache_path / "prefix.npy", mmap_mode="r", allow_pickle=False
    )
    descriptors = torch.from_numpy(
        np.load(prototype_path / "prototypes.npy", allow_pickle=False)
    ).to(device)
    feature_level = int(prototype_metadata["feature_level"])
    if backbone.shape[0] != len(data.frame) or prefix.shape[0] != len(data.frame):
        raise ValueError("vision cache arrays do not match the control dataset length")
    if backbone.ndim != 5 or not 0 <= feature_level < backbone.shape[1]:
        raise ValueError("prototype feature level lies outside the backbone cache")

    required_candidate_indices = np.unique(
        np.concatenate(
            [
                history.current_indices,
                history.state_indices.reshape(-1),
            ]
        )
    )
    all_pool = generate_candidate_pool_at_indices(
        data,
        indices=required_candidate_indices,
        backbone=backbone,
        descriptors=descriptors,
        feature_level=feature_level,
        device=device,
        batch_size=args.batch_size,
    )
    pool_lookup = candidate_lookup(required_candidate_indices, all_pool)
    current_positions = np.asarray(
        [pool_lookup[int(index)] for index in history.current_indices], dtype=np.int64
    )
    current_pool = select_pool_rows(all_pool, current_positions)
    candidate_valid = current_pool.valid & data.valid_nominal[
        history.current_indices, None
    ]
    candidate_score = current_pool.score.astype(np.float32, copy=False)
    visual_descriptor = sample_candidate_visual_descriptors(
        backbone,
        feature_level=feature_level,
        raw_indices=history.current_indices,
        points=current_pool.points,
        image_size=data.image_size,
        device=device,
        batch_size=args.batch_size,
    )
    geometry = candidate_geometry(
        data,
        raw_indices=history.current_indices,
        points=current_pool.points,
        valid=candidate_valid,
    )

    unique_history_indices = np.unique(history.state_indices.reshape(-1))
    label_history_indices = data.previous_index[history.current_indices]
    native_inference_indices = np.unique(
        np.concatenate([unique_history_indices, label_history_indices])
    )
    history_pool_positions = np.asarray(
        [pool_lookup[int(index)] for index in unique_history_indices], dtype=np.int64
    )
    wide_points = all_pool.points[history_pool_positions, 0].astype(
        np.float32, copy=False
    )
    wide_valid = grounding_valid_mask(
        wide_points,
        data.valid_nominal[unique_history_indices],
        image_size=data.image_size,
    ) & all_pool.valid[history_pool_positions, 0]
    nominal_points = data.q_nominal[unique_history_indices].astype(
        np.float32, copy=False
    )
    nominal_valid = grounding_valid_mask(
        nominal_points,
        data.valid_nominal[unique_history_indices],
        image_size=data.image_size,
    )
    history_corrector, history_corrector_args = _load_canonical_history_corrector(
        history_corrector_path,
        device=device,
        cache_metadata=cache_metadata,
        control_sites=control_sites,
    )
    native_inference_points = infer_canonical_fcpg_history_points(
        history_corrector,
        obs_mode=str(history_corrector_args["obs_mode"]),
        backbone=backbone,
        prefix=prefix,
        data=data,
        raw_indices=native_inference_indices,
        device=device,
        batch_size=args.batch_size,
    )
    native_inference_valid = grounding_valid_mask(
        native_inference_points,
        data.valid_nominal[native_inference_indices],
        image_size=data.image_size,
    )
    native_inference_lookup = {
        int(raw): position for position, raw in enumerate(native_inference_indices)
    }
    native_history_positions = np.asarray(
        [native_inference_lookup[int(raw)] for raw in unique_history_indices],
        dtype=np.int64,
    )
    native_label_positions = np.asarray(
        [native_inference_lookup[int(raw)] for raw in label_history_indices],
        dtype=np.int64,
    )
    native_points = native_inference_points[native_history_positions]
    native_valid = native_inference_valid[native_history_positions]
    native_label_past = native_inference_points[native_label_positions]
    del history_corrector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, lewm_resolved, lewm_payload = _validate_lewm_checkpoint(lewm_path, device)
    lewm_amp_dtype = torch.bfloat16
    lewm_required_indices = np.unique(
        np.concatenate([history.current_indices, history.state_indices.reshape(-1)])
    )
    lewm_visual = extract_lewm_visual_cache(
        model,
        rgb,
        raw_indices=lewm_required_indices,
        device=device,
        batch_size=args.batch_size,
        amp_dtype=lewm_amp_dtype,
    )
    candidate_tokens = encode_candidate_tokens(
        model,
        lewm_visual,
        data,
        raw_indices=history.current_indices,
        points=current_pool.points,
        valid=candidate_valid,
        device=device,
        candidate_chunk_size=args.candidate_chunk_size,
        amp_dtype=lewm_amp_dtype,
    )

    history_lookup = {
        int(raw): index for index, raw in enumerate(unique_history_indices)
    }
    history_positions = np.asarray(
        [
            [history_lookup[int(raw)] for raw in row]
            for row in history.state_indices
        ],
        dtype=np.int64,
    )
    history_variants: dict[str, dict[str, np.ndarray]] = {}
    for name, unique_points, unique_valid in (
        ("native_fcpg", native_points, native_valid),
        ("wide_top1", wide_points, wide_valid),
        ("nominal", nominal_points, nominal_valid),
    ):
        summary_unique, embedding_unique, anchor_unique = encode_history_tokens(
            model,
            lewm_visual,
            data,
            raw_indices=unique_history_indices,
            grounding_points=unique_points,
            grounding_valid=unique_valid,
            device=device,
            batch_size=args.batch_size,
            amp_dtype=lewm_amp_dtype,
        )
        embedding = embedding_unique[history_positions]
        history_variants[name] = {
            "grounding_points_px": unique_points[history_positions],
            "grounding_valid": unique_valid[history_positions],
            "summary_token": summary_unique[history_positions],
            "summary_embedding": embedding,
            "anchor_tokens": anchor_unique[history_positions],
            "predicted_current_summary_prior": predict_current_summary_prior(
                model,
                embedding,
                history.action_blocks,
                device=device,
                batch_size=args.batch_size,
                amp_dtype=lewm_amp_dtype,
            ),
            "shuffled_predicted_current_summary_prior": predict_current_summary_prior(
                model,
                embedding,
                shuffled_actions,
                device=device,
                batch_size=args.batch_size,
                amp_dtype=lewm_amp_dtype,
            ),
        }
    primary_history_name = PRIMARY_HISTORY_SOURCE
    label_history_source = PRIMARY_HISTORY_SOURCE
    primary_history = history_variants[primary_history_name]

    labels = build_policy_labels(
        data=data,
        current_indices=history.current_indices,
        current_pool=current_pool,
        past_grounding_points=native_label_past,
        backbone=backbone,
        prefix=prefix,
        checkpoint_paths=policy_paths,
        checkpoint_args=checkpoint_args,
        action_stds=action_stds,
        policy_seeds=policy_seeds,
        device=device,
        batch_size=args.batch_size,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    expected_label_shape = (
        len(history.current_indices),
        len(policy_seeds),
        MAX_HYPOTHESES,
    )
    if labels.shape != expected_label_shape:
        raise ValueError(
            f"offline labels must have axes [sample, policy, candidate] and shape "
            f"{expected_label_shape}, got {labels.shape}"
        )

    deployment = {
        "task_name": selected_task.astype("<U32"),
        "episode_id": selected_episode.astype(np.int64),
        "frame_id": selected_frame.astype(np.int64),
        "raw_index": history.current_indices.astype(np.int64),
        "split": split,
        "candidate_points_px": current_pool.points.astype(np.float32),
        "candidate_valid": candidate_valid.astype(bool),
        "candidate_grid_indices": current_pool.grid_indices.astype(np.int64),
        "candidate_joint_visual_score": candidate_score,
        "candidate_visual_descriptor": visual_descriptor,
        "candidate_geometry": geometry,
        "candidate_summary_token": candidate_tokens.summary_token,
        "candidate_summary_embedding": candidate_tokens.summary_embedding,
        "candidate_anchor_tokens": candidate_tokens.anchor_tokens,
        "history_raw_indices": history.state_indices.astype(np.int64),
        "history_frame_ids": data.frame[history.state_indices].astype(np.int64),
        "history_grounding_points_px": primary_history["grounding_points_px"].astype(
            np.float32
        ),
        "history_grounding_valid": primary_history["grounding_valid"].astype(bool),
        "history_action_blocks": history.action_blocks.astype(np.float32),
        "history_summary_token": primary_history["summary_token"],
        "history_summary_embedding": primary_history["summary_embedding"],
        "history_anchor_tokens": primary_history["anchor_tokens"],
        "predicted_current_summary_prior": primary_history[
            "predicted_current_summary_prior"
        ],
    }
    for variant in ("wide_top1", "nominal"):
        values = history_variants[variant]
        deployment.update(
            {
                f"history_grounding_points_px_{variant}": values[
                    "grounding_points_px"
                ].astype(np.float32),
                f"history_grounding_valid_{variant}": values[
                    "grounding_valid"
                ].astype(bool),
                f"history_summary_token_{variant}": values["summary_token"],
                f"history_summary_embedding_{variant}": values[
                    "summary_embedding"
                ],
                f"history_anchor_tokens_{variant}": values["anchor_tokens"],
                f"predicted_current_summary_prior_{variant}": values[
                    "predicted_current_summary_prior"
                ],
            }
        )
    assert_deployment_partition(deployment)
    offline = {
        "task_name": selected_task.astype("<U32"),
        "episode_id": selected_episode.astype(np.int64),
        "frame_id": selected_frame.astype(np.int64),
        "raw_index": history.current_indices.astype(np.int64),
        "policy_seed": np.asarray(policy_seeds, dtype=np.int64),
        "action_xyz_normalized_mse_h8": labels.astype(np.float32),
        "shuffled_current_action_blocks": shuffled_actions[:, -1].astype(np.float32),
        "shuffled_action_source_index": shuffle_permutation.astype(np.int64),
        "shuffled_predicted_current_summary_prior": primary_history[
            "shuffled_predicted_current_summary_prior"
        ],
        "shuffled_predicted_current_summary_prior_wide_top1": history_variants[
            "wide_top1"
        ]["shuffled_predicted_current_summary_prior"],
        "shuffled_predicted_current_summary_prior_nominal": history_variants[
            "nominal"
        ]["shuffled_predicted_current_summary_prior"],
    }
    if not np.isfinite(labels).all():
        raise ValueError("offline action-divergence labels contain non-finite values")

    deployment_path = output_dir / "deployment_inputs.npz"
    labels_path = output_dir / "offline_labels.npz"
    try:
        _write_npz(deployment_path, deployment)
        _write_npz(labels_path, offline)
        contract = {
            "status": "posthoc_development_not_confirmatory",
            "smoke_only": smoke_only,
            "training_performed": False,
            "deployment_offline_partitioned": True,
            "q_true_usage": "transient_offline_teacher_only_not_persisted",
            "oracle_action_usage": "transient_offline_teacher_only_not_persisted",
            "executed_action_alignment": alignment,
            "current_prior_uses_frames": "t-15,t-10,t-5",
            "current_prior_uses_actions": "t-15_through_t-1_only",
            "history_grounding_source": (
                "canonical_native_fcpg_corrector_matched_to_lewm_training"
            ),
            "history_grounding_primary": primary_history_name,
            "label_history_source": label_history_source,
            "label_history_frame": "t-1",
            "label_history_matches_primary": label_history_source
            == primary_history_name,
            "history_grounding_diagnostics": ["wide_top1", "nominal"],
            "history_grounding_variants": {
                "native_fcpg": (
                    "primary deployment history, generic prior fields, and offline "
                    "policy-label history; canonical frozen FCPG corrector"
                ),
                "wide_top1": (
                    "deployment diagnostic history; frozen wide-search visual top-1"
                ),
                "nominal": (
                    "deployment diagnostic history; uncorrected kinematic projection"
                ),
            },
            "history_inference_q_true_usage": "none",
            "history_shuffled_prior_partition": "offline_labels_only",
            "history_shuffled_source_scope": (
                "same_split_task_frame_different_physical_episode"
            ),
            "offline_label_axes": "sample,policy,candidate",
            "structured_temporal_prediction_claim": False,
            "claim_boundary": (
                "G1A selector development cache only; no trained selector, rollout, "
                "CEM, LeWM control benefit, or confirmatory result"
            ),
        }
        assert_label_history_contract(contract)
        metadata = {
            "schema_version": 3,
            "cache_type": "fcpg_g1a_selector_development_v3",
            "contract": contract,
            "sample_count": len(history.current_indices),
            "candidate_count": MAX_HYPOTHESES,
            "policy_seeds": policy_seeds,
            "split_episodes": split_episodes,
            "candidate_settings": {
                "radius_px": RADIUS_PX,
                "grid_size": GRID_SIZE,
                "per_anchor_topk": PER_ANCHOR_TOPK,
                "max_hypotheses": MAX_HYPOTHESES,
                "nms_radius_px": NMS_RADIUS_PX,
                "min_confidence": MIN_CONFIDENCE,
            },
            "inputs": {
                "data": {"path": str(data_path), "sha256": sha256_file(data_path)},
                "vision_cache": {
                    "path": str(vision_cache_path),
                    "metadata_sha256": sha256_file(vision_cache_path / "metadata.json"),
                    "backbone_features_sha256": sha256_file(
                        vision_cache_path / "backbone_features.npy"
                    ),
                    "prefix_sha256": sha256_file(vision_cache_path / "prefix.npy"),
                },
                "prototype_bank": {
                    "path": str(prototype_path),
                    "metadata_sha256": sha256_file(prototype_path / "metadata.json"),
                    "prototypes_sha256": sha256_file(prototype_path / "prototypes.npy"),
                },
                "lewm_checkpoint": {
                    "path": str(lewm_path),
                    "sha256": sha256_file(lewm_path),
                    "step": int(lewm_payload["step"]),
                    "seed": int(lewm_resolved["seed"]),
                    "target_mode": "online",
                },
                "history_corrector_checkpoint": {
                    "path": str(history_corrector_path),
                    "sha256": sha256_file(history_corrector_path),
                    "corrector_arch": str(history_corrector_args["corrector_arch"]),
                    "obs_mode": str(history_corrector_args["obs_mode"]),
                    "control_keypoint_sites": list(control_sites),
                    "primary_history_variant": "native_fcpg",
                },
                "policy_checkpoints": [
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "seed": int(seed),
                        "action_std": np.asarray(std).tolist(),
                    }
                    for path, seed, std in zip(
                        policy_paths, policy_seeds, action_stds, strict=True
                    )
                ],
            },
            "artifacts": {
                "deployment_inputs": {
                    "path": str(deployment_path),
                    "sha256": sha256_file(deployment_path),
                    "keys": sorted(deployment),
                    "shapes": {key: list(value.shape) for key, value in deployment.items()},
                    "dtypes": {key: str(value.dtype) for key, value in deployment.items()},
                },
                "offline_labels": {
                    "path": str(labels_path),
                    "sha256": sha256_file(labels_path),
                    "keys": sorted(offline),
                    "shapes": {key: list(value.shape) for key, value in offline.items()},
                    "dtypes": {key: str(value.dtype) for key, value in offline.items()},
                },
            },
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        for path in (deployment_path, labels_path, output_dir / "metadata.json"):
            path.unlink(missing_ok=True)
        if output_dir.exists() and not any(output_dir.iterdir()):
            shutil.rmtree(output_dir)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_count": len(history.current_indices),
                "candidate_count": MAX_HYPOTHESES,
                "policy_seeds": policy_seeds,
                "smoke_only": smoke_only,
                "deployment_sha256": sha256_file(deployment_path),
                "offline_labels_sha256": sha256_file(labels_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
