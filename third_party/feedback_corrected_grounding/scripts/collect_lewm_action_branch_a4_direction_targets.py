#!/usr/bin/env python3
"""Replay A4 Expert branches and collect exact 2D anchor-center targets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.geometry import Camera, project_points  # noqa: E402
from fcpg.lewm_a4_task_factory import CUSTOM_TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    BRANCH_COUNT,
    BRANCH_HORIZON_RAW_STEPS,
    HORIZON,
    RAW_STEPS_PER_MODEL_STEP,
    TASKS,
    assign_a4_splits,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    RAW_ACTION_DIM,
    _bounded_expert_action,
    _physical_support_pose,
    a4_camera_name,
    make_a4_episode_runtime,
    read_physical_target,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    capture_branch_snapshot,
    restore_branch_snapshot,
)
from fcpg.lewm_counterfactual_v2_runtime import (  # noqa: E402
    _close_episode_env_safely,
)
from fcpg.metaworld_utils import (  # noqa: E402
    DEFAULT_KEYPOINT_SITES,
    get_site_keypoints,
)
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_LEDGER = ROOT / "outputs/lewm_action_branch_a4/formal_v2/proposal_ledger.npz"
DEFAULT_CACHE = ROOT / "outputs/lewm_action_branch_a4/formal_v2/state_cache.npz"
DEFAULT_LABELS = ROOT / "outputs/lewm_action_branch_a4/formal_v2/preflight_labels.npz"
DEFAULT_OUTPUT = (
    ROOT / "outputs/lewm_action_branch_a4/formal_v2/expert_direction_targets.npz"
)

SCHEMA_VERSION = "fcpg_a4_expert_direction_targets_v1"
MODEL_STEP_OFFSETS = np.arange(
    0,
    BRANCH_HORIZON_RAW_STEPS + 1,
    RAW_STEPS_PER_MODEL_STEP,
    dtype=np.int64,
)
RAW_STEP_OFFSETS = np.arange(BRANCH_HORIZON_RAW_STEPS + 1, dtype=np.int64)

OUTPUT_KEYS = (
    "schema_version",
    "proposal_id",
    "cache_row_index",
    "split",
    "task_name",
    "task_instance_index",
    "source_episode_id",
    "environment_seed",
    "proposal_raw_step",
    "expert_slot",
    "image_width",
    "image_height",
    "display_flip_y",
    "raw_step_offsets",
    "model_step_raw_offsets",
    "q_center_raw_coords_raw_steps",
    "q_center_display_coords_raw_steps",
    "q_center_model_raw",
    "q_center_display",
    "all_anchors_valid_raw_steps",
    "all_anchors_valid_model_steps",
    "physical_validation_max_abs",
    "reserved_test_opened",
)


@dataclass(frozen=True, slots=True)
class ReplayRow:
    cache_row_index: int
    proposal_id: str
    split: str
    task_name: str
    task_instance_index: int
    source_episode_id: str
    environment_seed: int
    proposal_raw_step: int
    expert_slot: int
    expert_actions: np.ndarray
    expected_physical: np.ndarray


@dataclass(frozen=True, slots=True)
class ReplayResult:
    q_raw: np.ndarray
    q_display: np.ndarray
    all_anchors_valid: np.ndarray
    physical_validation_max_abs: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON sidecar; defaults to OUTPUT with a .json suffix",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-source-episodes", type=int, default=None)
    parser.add_argument("--max-proposals", type=int, default=None)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_npz(path: Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    source = path.expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(set(fields) - set(archive.files))
        if missing:
            raise ValueError(f"{source} is missing {missing}")
        return {name: np.array(archive[name], copy=True) for name in fields}


def _require_unique_ids(values: np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(values).astype(str)
    if result.ndim != 1 or len(set(result.tolist())) != result.size:
        raise ValueError(f"{label} proposal_id values must be a unique vector")
    return result


def _require_development_archives_only(
    split: np.ndarray, *, label: str
) -> np.ndarray:
    result = np.asarray(split).astype(str)
    observed = set(result.tolist())
    if not observed or not observed <= {"train", "development"}:
        raise PermissionError(
            f"{label} must contain train/development only; observed {sorted(observed)}"
        )
    return result


def _expert_raw_actions(
    future_action: np.ndarray,
    center_action: np.ndarray,
    semantic_index: np.ndarray,
    *,
    expert_semantic_id: int,
) -> tuple[int, np.ndarray]:
    future = np.asarray(future_action, dtype=np.float32)
    center = np.asarray(center_action, dtype=np.float32)
    semantic = np.asarray(semantic_index, dtype=np.int64)
    if future.shape != (BRANCH_COUNT, HORIZON, RAW_STEPS_PER_MODEL_STEP * RAW_ACTION_DIM):
        raise ValueError("A4 future_action row must have shape [8,8,20]")
    if center.shape != (HORIZON, RAW_STEPS_PER_MODEL_STEP * RAW_ACTION_DIM):
        raise ValueError("A4 center_action row must have shape [8,20]")
    if semantic.shape != (BRANCH_COUNT,):
        raise ValueError("A4 branch_semantic_index row must have shape [8]")
    slots = np.flatnonzero(semantic == int(expert_semantic_id))
    if slots.size != 1:
        raise ValueError("A4 row must contain exactly one semantic Expert branch")
    slot = int(slots[0])
    if not np.array_equal(future[slot], center):
        raise RuntimeError("semantic Expert branch differs from center_action")
    actions = np.ascontiguousarray(
        future[slot].reshape(HORIZON, RAW_STEPS_PER_MODEL_STEP, RAW_ACTION_DIM).reshape(
            BRANCH_HORIZON_RAW_STEPS, RAW_ACTION_DIM
        ),
        dtype=np.float32,
    )
    if not np.isfinite(actions).all():
        raise ValueError("semantic Expert actions must be finite")
    return slot, actions


def load_replay_rows(
    contract: Mapping[str, Any],
    *,
    ledger_path: Path,
    cache_path: Path,
    labels_path: Path,
) -> list[ReplayRow]:
    tasks = validate_a4_contract(contract)
    semantics = tuple(
        str(value)
        for value in contract["matched_state_branches"][
            "semantic_bank_before_permutation"
        ]
    )
    if semantics.count("expert") != 1:
        raise ValueError("A4 semantic bank must contain exactly one Expert branch")
    expert_semantic_id = semantics.index("expert")

    ledger = _load_npz(
        ledger_path,
        (
            "proposal_id",
            "task_name",
            "task_instance_index",
            "source_episode_id",
            "environment_seed",
            "proposal_raw_step",
            "split",
            "eligible",
            "accepted_bundle_index",
        ),
    )
    cache = _load_npz(
        cache_path,
        (
            "proposal_id",
            "task_name",
            "task_instance_index",
            "source_episode_id",
            "split",
            "future_action",
            "center_action",
            "branch_semantic_index",
        ),
    )
    labels = _load_npz(
        labels_path,
        ("proposal_id", "split", "physical_state"),
    )

    ledger_ids = _require_unique_ids(ledger["proposal_id"], label="ledger")
    cache_ids = _require_unique_ids(cache["proposal_id"], label="cache")
    label_ids = _require_unique_ids(labels["proposal_id"], label="labels")
    ledger_split = _require_development_archives_only(ledger["split"], label="ledger")
    cache_split = _require_development_archives_only(cache["split"], label="cache")
    label_split = _require_development_archives_only(labels["split"], label="labels")

    eligible = np.asarray(ledger["eligible"])
    accepted = np.asarray(ledger["accepted_bundle_index"])
    if eligible.shape != ledger_ids.shape or eligible.dtype.kind != "b":
        raise ValueError("ledger eligible must be a boolean proposal vector")
    if accepted.shape != ledger_ids.shape or accepted.dtype.kind not in "iu":
        raise ValueError("ledger accepted_bundle_index must be an integer vector")
    eligible_rows = np.flatnonzero(eligible)
    ordered_ledger_rows = eligible_rows[np.argsort(accepted[eligible_rows])]
    if not np.array_equal(accepted[ordered_ledger_rows], np.arange(cache_ids.size)):
        raise ValueError("eligible ledger bundle indices must be contiguous")
    if not np.array_equal(ledger_ids[ordered_ledger_rows], cache_ids):
        raise ValueError("cache order differs from accepted ledger order")
    if not np.array_equal(label_ids, cache_ids):
        raise ValueError("physical labels must be row-aligned with the cache")
    if not np.array_equal(label_split, cache_split):
        raise ValueError("physical-label split differs from the cache")

    count = cache_ids.size
    expected_shapes = {
        "future_action": (
            count,
            BRANCH_COUNT,
            HORIZON,
            RAW_STEPS_PER_MODEL_STEP * RAW_ACTION_DIM,
        ),
        "center_action": (
            count,
            HORIZON,
            RAW_STEPS_PER_MODEL_STEP * RAW_ACTION_DIM,
        ),
        "branch_semantic_index": (count, BRANCH_COUNT),
        "physical_state": (count, BRANCH_COUNT, HORIZON, 7),
    }
    for name, shape in expected_shapes.items():
        source = labels if name == "physical_state" else cache
        if np.asarray(source[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    assignment = assign_a4_splits(tasks, contract)
    rows: list[ReplayRow] = []
    for cache_row, ledger_row in enumerate(ordered_ledger_rows.tolist()):
        task = str(cache["task_name"][cache_row])
        instance = int(cache["task_instance_index"][cache_row])
        split = str(cache_split[cache_row])
        metadata_pairs = (
            (task, str(ledger["task_name"][ledger_row]), "task_name"),
            (
                instance,
                int(ledger["task_instance_index"][ledger_row]),
                "task_instance_index",
            ),
            (
                str(cache["source_episode_id"][cache_row]),
                str(ledger["source_episode_id"][ledger_row]),
                "source_episode_id",
            ),
            (split, str(ledger_split[ledger_row]), "split"),
        )
        for cache_value, ledger_value, name in metadata_pairs:
            if cache_value != ledger_value:
                raise ValueError(f"cache/ledger {name} mismatch for {cache_ids[cache_row]}")
        if task not in tasks or assignment[(task, instance)] != split:
            raise PermissionError("cache row violates the frozen train/development split")
        slot, actions = _expert_raw_actions(
            cache["future_action"][cache_row],
            cache["center_action"][cache_row],
            cache["branch_semantic_index"][cache_row],
            expert_semantic_id=expert_semantic_id,
        )
        rows.append(
            ReplayRow(
                cache_row_index=cache_row,
                proposal_id=str(cache_ids[cache_row]),
                split=split,
                task_name=task,
                task_instance_index=instance,
                source_episode_id=str(cache["source_episode_id"][cache_row]),
                environment_seed=int(ledger["environment_seed"][ledger_row]),
                proposal_raw_step=int(ledger["proposal_raw_step"][ledger_row]),
                expert_slot=slot,
                expert_actions=actions,
                expected_physical=np.asarray(
                    labels["physical_state"][cache_row, slot], dtype=np.float32
                ).copy(),
            )
        )
    return rows


def _display_coordinates(points: np.ndarray, *, task: str, height: int) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).copy()
    if result.shape[-1] != 2 or not np.isfinite(result).all():
        raise ValueError("q_true points must be finite [...,2]")
    if task not in CUSTOM_TASKS:
        result[..., 1] = float(height - 1) - result[..., 1]
    return result


def _make_center_reader(
    env: Any, *, task: str, width: int, height: int
) -> Callable[[], tuple[np.ndarray, np.ndarray, bool]]:
    camera_id = get_camera_id(env.model, a4_camera_name(task))
    intrinsic = camera_intrinsic_from_fovy(
        int(width), int(height), float(env.model.cam_fovy[camera_id])
    )

    def read() -> tuple[np.ndarray, np.ndarray, bool]:
        keypoints = get_site_keypoints(
            env.model, env.data, DEFAULT_KEYPOINT_SITES
        ).xyz
        if keypoints.shape != (3, 3) or not np.isfinite(keypoints).all():
            raise ValueError("A4 q_true reader requires three finite robot anchors")
        extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
        q_true, valid = project_points(
            keypoints,
            Camera(intrinsic, extrinsic, int(width), int(height)),
        )
        q_raw = np.asarray(q_true, dtype=np.float32)
        valid_array = np.asarray(valid, dtype=bool)
        if q_raw.shape != (3, 2) or valid_array.shape != (3,):
            raise ValueError("A4 q_true projection must emit [3,2] and [3]")
        q_raw_center = q_raw.mean(axis=0, dtype=np.float32)
        q_display_center = _display_coordinates(
            q_raw_center, task=task, height=height
        )
        return (
            q_raw_center,
            q_display_center,
            bool(valid_array.all()),
        )

    return read


def _execute_expert_branch(
    runtime: Any,
    row: ReplayRow,
    *,
    observation: np.ndarray,
    info: Mapping[str, Any],
    width: int,
    height: int,
    center_reader_factory: Callable[..., Callable[[], tuple[np.ndarray, np.ndarray, bool]]] = _make_center_reader,
) -> ReplayResult:
    actions = np.asarray(row.expert_actions, dtype=np.float32)
    if actions.shape != (BRANCH_HORIZON_RAW_STEPS, RAW_ACTION_DIM):
        raise ValueError("Expert branch must contain exactly 40 raw actions")
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if low.shape != (RAW_ACTION_DIM,) or high.shape != (RAW_ACTION_DIM,):
        raise ValueError("A4 action bounds must have shape [4]")
    if not np.array_equal(actions, np.clip(actions, low, high)):
        raise ValueError("cached Expert actions must not require clipping")

    read_center = center_reader_factory(
        runtime.env, task=row.task_name, width=width, height=height
    )
    raw_rows: list[np.ndarray] = []
    display_rows: list[np.ndarray] = []
    valid_rows: list[bool] = []
    raw, display, valid = read_center()
    raw_rows.append(raw)
    display_rows.append(display)
    valid_rows.append(valid)

    current = np.asarray(observation, dtype=np.float32)
    current_info: Mapping[str, Any] = info
    physical_rows = []
    for raw_index, action in enumerate(actions, start=1):
        current, _reward, terminated, truncated, current_info = runtime.env.step(action)
        if not isinstance(current_info, Mapping):
            raise TypeError("A4 branch info must be a mapping")
        if terminated or truncated:
            raise RuntimeError("A4 Expert replay ended before 40 raw steps")
        current = np.asarray(current, dtype=np.float32)
        raw, display, valid = read_center()
        raw_rows.append(raw)
        display_rows.append(display)
        valid_rows.append(valid)
        if raw_index % RAW_STEPS_PER_MODEL_STEP == 0:
            target = read_physical_target(
                runtime.env, row.task_name, current, current_info
            )
            physical_rows.append(_physical_support_pose(row.task_name, target))

    q_raw = np.asarray(raw_rows, dtype=np.float32)
    q_display = np.asarray(display_rows, dtype=np.float32)
    all_valid = np.asarray(valid_rows, dtype=bool)
    physical = np.asarray(physical_rows, dtype=np.float32)
    if q_raw.shape != (BRANCH_HORIZON_RAW_STEPS + 1, 2):
        raise RuntimeError("raw q_true center replay must have shape [41,2]")
    if q_display.shape != q_raw.shape or all_valid.shape != (q_raw.shape[0],):
        raise RuntimeError("display q_true/valid replay shape mismatch")
    if physical.shape != (HORIZON, 7):
        raise RuntimeError("physical replay must have shape [8,7]")
    max_abs = float(np.max(np.abs(physical - row.expected_physical)))
    if max_abs != 0.0:
        raise RuntimeError(
            f"exact replay physical validation failed for {row.proposal_id}: "
            f"max_abs={max_abs}"
        )
    return ReplayResult(q_raw, q_display, all_valid, max_abs)


def _episode_key(row: ReplayRow) -> tuple[str, int, int, str, str]:
    return (
        row.task_name,
        row.task_instance_index,
        row.environment_seed,
        row.source_episode_id,
        row.split,
    )


def _smoke_subset(
    rows: Sequence[ReplayRow],
    *,
    smoke: bool,
    max_source_episodes: int | None,
    max_proposals: int | None,
) -> list[ReplayRow]:
    if not smoke and (max_source_episodes is not None or max_proposals is not None):
        raise ValueError("source-episode/proposal limits are smoke-only")
    values = list(rows)
    if not smoke:
        return values
    episode_limit = 1 if max_source_episodes is None else int(max_source_episodes)
    proposal_limit = 1 if max_proposals is None else int(max_proposals)
    if episode_limit <= 0 or proposal_limit <= 0:
        raise ValueError("smoke limits must be positive")
    selected_keys: list[tuple[str, int, int, str, str]] = []
    for row in values:
        key = _episode_key(row)
        if key not in selected_keys:
            selected_keys.append(key)
        if len(selected_keys) == episode_limit:
            break
    allowed = set(selected_keys)
    return [row for row in values if _episode_key(row) in allowed][:proposal_limit]


def collect_direction_targets(
    rows: Sequence[ReplayRow],
    *,
    width: int,
    height: int,
    benchmark_seed: int,
    runtime_factory: Callable[..., Any] = make_a4_episode_runtime,
    center_reader_factory: Callable[..., Callable[[], tuple[np.ndarray, np.ndarray, bool]]] = _make_center_reader,
) -> dict[int, ReplayResult]:
    selected = list(rows)
    if not selected:
        raise ValueError("direction-target replay requires at least one proposal")
    grouped: dict[tuple[str, int, int, str, str], list[ReplayRow]] = defaultdict(list)
    for row in selected:
        grouped[_episode_key(row)].append(row)

    results: dict[int, ReplayResult] = {}
    for key, episode_rows in grouped.items():
        task, instance, environment_seed, _episode_id, _split = key
        episode_rows.sort(key=lambda row: (row.proposal_raw_step, row.cache_row_index))
        steps: dict[int, list[ReplayRow]] = defaultdict(list)
        for row in episode_rows:
            steps[row.proposal_raw_step].append(row)
        runtime = runtime_factory(
            task,
            instance,
            environment_seed,
            width=int(width),
            height=int(height),
            benchmark_seed=int(benchmark_seed),
        )
        try:
            observation = np.asarray(runtime.observation, dtype=np.float32)
            info: Mapping[str, Any] = runtime.reset_info
            last_step = max(steps)
            for source_step in range(last_step + 1):
                source_action = _bounded_expert_action(runtime, observation)
                for row in steps.get(source_step, ()):
                    snapshot = capture_branch_snapshot(
                        runtime.env, include_process_rng=True
                    )
                    try:
                        result = _execute_expert_branch(
                            runtime,
                            row,
                            observation=observation,
                            info=info,
                            width=width,
                            height=height,
                            center_reader_factory=center_reader_factory,
                        )
                    finally:
                        restore_branch_snapshot(runtime.env, snapshot)
                    if row.cache_row_index in results:
                        raise RuntimeError("duplicate A4 cache row replay")
                    results[row.cache_row_index] = result
                if source_step == last_step:
                    break
                observation, _reward, terminated, truncated, info = runtime.env.step(
                    source_action
                )
                if not isinstance(info, Mapping):
                    raise TypeError("A4 source info must be a mapping")
                if terminated or truncated:
                    raise RuntimeError("A4 source episode ended before a proposal")
                observation = np.asarray(observation, dtype=np.float32)
        finally:
            _close_episode_env_safely(runtime.env)
        print(
            json.dumps(
                {
                    "source_episode_id": episode_rows[0].source_episode_id,
                    "task": task,
                    "proposals": len(episode_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if set(results) != {row.cache_row_index for row in selected}:
        raise RuntimeError("direction-target replay did not cover every selected row")
    return results


def stack_direction_targets(
    rows: Sequence[ReplayRow],
    results: Mapping[int, ReplayResult],
    *,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    ordered = sorted(rows, key=lambda row: row.cache_row_index)
    q_raw = np.stack([results[row.cache_row_index].q_raw for row in ordered]).astype(
        np.float32, copy=False
    )
    q_display = np.stack(
        [results[row.cache_row_index].q_display for row in ordered]
    ).astype(np.float32, copy=False)
    all_valid = np.stack(
        [results[row.cache_row_index].all_anchors_valid for row in ordered]
    ).astype(bool, copy=False)
    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "proposal_id": np.asarray([row.proposal_id for row in ordered]),
        "cache_row_index": np.asarray(
            [row.cache_row_index for row in ordered], dtype=np.int64
        ),
        "split": np.asarray([row.split for row in ordered]),
        "task_name": np.asarray([row.task_name for row in ordered]),
        "task_instance_index": np.asarray(
            [row.task_instance_index for row in ordered], dtype=np.int64
        ),
        "source_episode_id": np.asarray(
            [row.source_episode_id for row in ordered]
        ),
        "environment_seed": np.asarray(
            [row.environment_seed for row in ordered], dtype=np.int64
        ),
        "proposal_raw_step": np.asarray(
            [row.proposal_raw_step for row in ordered], dtype=np.int64
        ),
        "expert_slot": np.asarray([row.expert_slot for row in ordered], dtype=np.int64),
        "image_width": np.asarray(int(width), dtype=np.int64),
        "image_height": np.asarray(int(height), dtype=np.int64),
        "display_flip_y": np.asarray(
            [row.task_name not in CUSTOM_TASKS for row in ordered], dtype=bool
        ),
        "raw_step_offsets": RAW_STEP_OFFSETS.copy(),
        "model_step_raw_offsets": MODEL_STEP_OFFSETS.copy(),
        "q_center_raw_coords_raw_steps": q_raw,
        "q_center_display_coords_raw_steps": q_display,
        "q_center_model_raw": q_raw[:, MODEL_STEP_OFFSETS],
        "q_center_display": q_display[:, MODEL_STEP_OFFSETS],
        "all_anchors_valid_raw_steps": all_valid,
        "all_anchors_valid_model_steps": all_valid[:, MODEL_STEP_OFFSETS],
        "physical_validation_max_abs": np.asarray(
            [
                results[row.cache_row_index].physical_validation_max_abs
                for row in ordered
            ],
            dtype=np.float64,
        ),
        "reserved_test_opened": np.asarray(False, dtype=bool),
    }
    validate_direction_targets(arrays)
    return arrays


def validate_direction_targets(arrays: Mapping[str, np.ndarray]) -> None:
    if tuple(arrays) != OUTPUT_KEYS:
        raise ValueError("direction-target output keys or key order drifted")
    proposal_id = _require_unique_ids(arrays["proposal_id"], label="output")
    count = proposal_id.size
    if count <= 0:
        raise ValueError("direction-target output cannot be empty")
    split = _require_development_archives_only(arrays["split"], label="output")
    if split.shape != (count,):
        raise ValueError("output split must align with proposals")
    vector_fields = (
        "cache_row_index",
        "task_name",
        "task_instance_index",
        "source_episode_id",
        "environment_seed",
        "proposal_raw_step",
        "expert_slot",
        "display_flip_y",
        "physical_validation_max_abs",
    )
    for name in vector_fields:
        if np.asarray(arrays[name]).shape != (count,):
            raise ValueError(f"output {name} must have shape [{count}]")
    if str(np.asarray(arrays["schema_version"]).item()) != SCHEMA_VERSION:
        raise ValueError("direction-target schema version mismatch")
    if bool(np.asarray(arrays["reserved_test_opened"]).item()):
        raise PermissionError("direction-target output may not open reserved test")
    width = int(np.asarray(arrays["image_width"]).item())
    height = int(np.asarray(arrays["image_height"]).item())
    if width <= 0 or height <= 0:
        raise ValueError("direction-target image size must be positive")
    if not np.array_equal(arrays["raw_step_offsets"], RAW_STEP_OFFSETS):
        raise ValueError("raw-step offsets drifted")
    if not np.array_equal(arrays["model_step_raw_offsets"], MODEL_STEP_OFFSETS):
        raise ValueError("model-step offsets drifted")

    q_raw = np.asarray(arrays["q_center_raw_coords_raw_steps"])
    q_display = np.asarray(arrays["q_center_display_coords_raw_steps"])
    expected_raw_shape = (count, BRANCH_HORIZON_RAW_STEPS + 1, 2)
    if q_raw.shape != expected_raw_shape or q_display.shape != expected_raw_shape:
        raise ValueError("raw/display q-center trajectories must have shape [N,41,2]")
    if q_raw.dtype.kind != "f" or q_display.dtype.kind != "f":
        raise TypeError("q-center trajectories must be floating point")
    if not np.isfinite(q_raw).all() or not np.isfinite(q_display).all():
        raise ValueError("q-center trajectories must be finite")
    if not np.array_equal(arrays["q_center_model_raw"], q_raw[:, MODEL_STEP_OFFSETS]):
        raise ValueError("model-step raw centers must be the exact raw-step subset")
    if not np.array_equal(arrays["q_center_display"], q_display[:, MODEL_STEP_OFFSETS]):
        raise ValueError("model-step display centers must be the exact raw-step subset")
    expected_model_shape = (count, HORIZON + 1, 2)
    if np.asarray(arrays["q_center_model_raw"]).shape != expected_model_shape:
        raise ValueError("q_center_model_raw must have shape [N,9,2]")
    if np.asarray(arrays["q_center_display"]).shape != expected_model_shape:
        raise ValueError("q_center_display must have shape [N,9,2]")
    valid_raw = np.asarray(arrays["all_anchors_valid_raw_steps"])
    valid_model = np.asarray(arrays["all_anchors_valid_model_steps"])
    if valid_raw.shape != (count, BRANCH_HORIZON_RAW_STEPS + 1):
        raise ValueError("raw-step validity must have shape [N,41]")
    if valid_raw.dtype.kind != "b" or valid_model.dtype.kind != "b":
        raise TypeError("anchor validity must be boolean")
    if not np.array_equal(valid_model, valid_raw[:, MODEL_STEP_OFFSETS]):
        raise ValueError("model-step validity must be the exact raw-step subset")
    max_abs = np.asarray(arrays["physical_validation_max_abs"], dtype=np.float64)
    if not np.isfinite(max_abs).all() or not np.array_equal(max_abs, np.zeros(count)):
        raise RuntimeError("all exact physical replay errors must equal zero")

    flips = np.asarray(arrays["display_flip_y"], dtype=bool)
    expected_display = q_raw.copy()
    expected_display[flips, :, 1] = float(height - 1) - expected_display[flips, :, 1]
    if not np.array_equal(q_display, expected_display):
        raise ValueError("display q-center coordinates violate the upright transform")


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_save_json(path: Path, value: Mapping[str, Any]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = _load_json(args.config)
    rows = load_replay_rows(
        contract,
        ledger_path=args.ledger,
        cache_path=args.cache,
        labels_path=args.labels,
    )
    rows = _smoke_subset(
        rows,
        smoke=bool(args.smoke),
        max_source_episodes=args.max_source_episodes,
        max_proposals=args.max_proposals,
    )
    collection = contract["collection"]
    width = int(collection["width"])
    height = int(collection["height"])
    results = collect_direction_targets(
        rows,
        width=width,
        height=height,
        benchmark_seed=int(collection["benchmark_seed"]),
    )
    arrays = stack_direction_targets(rows, results, width=width, height=height)
    output = _atomic_save_npz(args.output, arrays)
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output.with_suffix(".json")
    )
    task_counts = {
        task: int(np.count_nonzero(arrays["task_name"].astype(str) == task))
        for task in TASKS
        if bool(np.any(arrays["task_name"].astype(str) == task))
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "output": str(output),
        "rows": int(arrays["proposal_id"].shape[0]),
        "splits": sorted(set(arrays["split"].astype(str).tolist())),
        "task_counts": task_counts,
        "reserved_test_opened": False,
        "physical_validation_max_abs": float(
            np.max(arrays["physical_validation_max_abs"])
        ),
        "coordinates": {
            "q_center_raw_coords_raw_steps": (
                "runtime q_true coordinates aligned with the frozen model input; "
                "41 samples at offsets 0..40"
            ),
            "q_center_model_raw": (
                "runtime q_true coordinates subsampled at model offsets "
                "0,5,...,40; use this field for train-only readouts"
            ),
            "q_center_display_coords_raw_steps": (
                "upright display coordinates; standard MetaWorld tasks apply "
                "y=height-1-y and custom tasks are unchanged"
            ),
            "q_center_display": (
                "upright display coordinates subsampled at model offsets 0,5,...,40"
            ),
        },
    }
    _atomic_save_json(manifest_path, manifest)
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
