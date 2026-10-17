#!/usr/bin/env python3
"""Replay A4 expert branches and collect all three 2-D anchor trajectories.

The historical direction-target collector reduced the three robot anchors to a
single pixel center.  That is useful for a coarse direction diagnostic, but it
cannot show gripper aperture or anchor identity.  This collector keeps the
three projected points separate and is used only for a train-only coordinate
readout.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.geometry import Camera, project_points  # noqa: E402
from fcpg.lewm_a4_task_factory import CUSTOM_TASKS  # noqa: E402
from fcpg.lewm_action_branch_a4 import (  # noqa: E402
    BRANCH_HORIZON_RAW_STEPS,
    HORIZON,
    RAW_STEPS_PER_MODEL_STEP,
    TASKS,
    validate_a4_contract,
)
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    RAW_ACTION_DIM,
    _bounded_expert_action,
    a4_camera_name,
    make_a4_episode_runtime,
    read_physical_target,
    _physical_support_pose,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    capture_branch_snapshot,
    restore_branch_snapshot,
)
from fcpg.lewm_counterfactual_v2_runtime import _close_episode_env_safely  # noqa: E402
from fcpg.metaworld_utils import DEFAULT_KEYPOINT_SITES, get_site_keypoints  # noqa: E402
from fcpg.mujoco_camera import (  # noqa: E402
    camera_extrinsic_cv_from_mujoco,
    camera_intrinsic_from_fovy,
    get_camera_id,
)
from scripts.collect_lewm_action_branch_a4_direction_targets import (  # noqa: E402
    ReplayRow,
    _episode_key,
    load_replay_rows,
    _load_json,
    _smoke_subset,
)


DEFAULT_CONFIG = ROOT / "config/lewm/eval/action_branch_active_control_a4.json"
DEFAULT_LEDGER = ROOT / "outputs/lewm_action_branch_a4/formal_v2/proposal_ledger.npz"
DEFAULT_CACHE = ROOT / "outputs/lewm_action_branch_a4/formal_v2/state_cache.npz"
DEFAULT_LABELS = ROOT / "outputs/lewm_action_branch_a4/formal_v2/preflight_labels.npz"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_action_branch_a4/formal_v2/anchor_targets.npz"

SCHEMA_VERSION = "fcpg_a4_expert_anchor_targets_v1"
RAW_STEP_OFFSETS = np.arange(BRANCH_HORIZON_RAW_STEPS + 1, dtype=np.int64)
MODEL_STEP_OFFSETS = np.arange(
    0, BRANCH_HORIZON_RAW_STEPS + 1, RAW_STEPS_PER_MODEL_STEP, dtype=np.int64
)


def _display_coordinates(points: np.ndarray, *, task: str, height: int) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).copy()
    if result.shape != (3, 2) or not np.isfinite(result).all():
        raise ValueError("anchor projection must be finite with shape [3,2]")
    if task not in CUSTOM_TASKS:
        result[:, 1] = float(height - 1) - result[:, 1]
    return result


def _make_anchor_reader(
    env: object, *, task: str, width: int, height: int
):
    camera_id = get_camera_id(env.model, a4_camera_name(task))
    intrinsic = camera_intrinsic_from_fovy(
        int(width), int(height), float(env.model.cam_fovy[camera_id])
    )

    def read() -> tuple[np.ndarray, np.ndarray, bool]:
        keypoints = get_site_keypoints(env.model, env.data, DEFAULT_KEYPOINT_SITES).xyz
        if keypoints.shape != (3, 3) or not np.isfinite(keypoints).all():
            raise ValueError("A4 anchor reader requires three finite 3-D points")
        extrinsic = camera_extrinsic_cv_from_mujoco(env.data, camera_id)
        q_true, valid = project_points(
            keypoints,
            Camera(intrinsic, extrinsic, int(width), int(height)),
        )
        raw = np.asarray(q_true, dtype=np.float32)
        valid_array = np.asarray(valid, dtype=bool)
        if raw.shape != (3, 2) or valid_array.shape != (3,):
            raise ValueError("A4 anchor projection must emit [3,2] and [3]")
        return raw, _display_coordinates(raw, task=task, height=height), bool(valid_array.all())

    return read


def _execute_anchor_branch(
    runtime: object,
    row: ReplayRow,
    *,
    observation: np.ndarray,
    info: Mapping[str, object],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    actions = np.asarray(row.expert_actions, dtype=np.float32)
    if actions.shape != (BRANCH_HORIZON_RAW_STEPS, RAW_ACTION_DIM):
        raise ValueError("expert branch must contain exactly 40 raw actions")
    low = np.asarray(runtime.env.action_space.low, dtype=np.float32)
    high = np.asarray(runtime.env.action_space.high, dtype=np.float32)
    if low.shape != (RAW_ACTION_DIM,) or high.shape != (RAW_ACTION_DIM,):
        raise ValueError("A4 action bounds must have shape [4]")
    if not np.array_equal(actions, np.clip(actions, low, high)):
        raise ValueError("cached expert actions must not require clipping")

    read = _make_anchor_reader(runtime.env, task=row.task_name, width=width, height=height)
    raw_rows: list[np.ndarray] = []
    display_rows: list[np.ndarray] = []
    valid_rows: list[bool] = []
    raw, display, valid = read()
    raw_rows.append(raw)
    display_rows.append(display)
    valid_rows.append(valid)

    current = np.asarray(observation, dtype=np.float32)
    current_info: Mapping[str, object] = info
    physical_rows: list[np.ndarray] = []
    for raw_index, action in enumerate(actions, start=1):
        current, _reward, terminated, truncated, current_info = runtime.env.step(action)
        if not isinstance(current_info, Mapping):
            raise TypeError("A4 branch info must be a mapping")
        if terminated or truncated:
            raise RuntimeError("A4 expert replay ended before 40 raw steps")
        current = np.asarray(current, dtype=np.float32)
        raw, display, valid = read()
        raw_rows.append(raw)
        display_rows.append(display)
        valid_rows.append(valid)
        if raw_index % RAW_STEPS_PER_MODEL_STEP == 0:
            target = read_physical_target(runtime.env, row.task_name, current, current_info)
            physical_rows.append(_physical_support_pose(row.task_name, target))

    q_raw = np.asarray(raw_rows, dtype=np.float32)
    q_display = np.asarray(display_rows, dtype=np.float32)
    valid = np.asarray(valid_rows, dtype=bool)
    physical = np.asarray(physical_rows, dtype=np.float32)
    if q_raw.shape != (BRANCH_HORIZON_RAW_STEPS + 1, 3, 2):
        raise RuntimeError(f"raw anchor replay shape drifted: {q_raw.shape}")
    if q_display.shape != q_raw.shape or valid.shape != (q_raw.shape[0],):
        raise RuntimeError("display anchor/valid replay shape mismatch")
    if physical.shape != (HORIZON, 7):
        raise RuntimeError("physical replay must have shape [8,7]")
    max_abs = float(np.max(np.abs(physical - row.expected_physical)))
    if max_abs != 0.0:
        raise RuntimeError(
            f"exact replay physical validation failed for {row.proposal_id}: max_abs={max_abs}"
        )
    return q_raw, q_display, valid, max_abs


def collect_anchor_targets(
    rows: Sequence[ReplayRow],
    *,
    width: int,
    height: int,
    benchmark_seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if not rows:
        raise ValueError("anchor-target replay requires at least one proposal")
    grouped: dict[tuple[str, int, int, str, str], list[ReplayRow]] = defaultdict(list)
    for row in rows:
        grouped[_episode_key(row)].append(row)
    results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for key, episode_rows in grouped.items():
        task, instance, environment_seed, episode_id, _split = key
        episode_rows.sort(key=lambda item: (item.proposal_raw_step, item.cache_row_index))
        by_step: dict[int, list[ReplayRow]] = defaultdict(list)
        for row in episode_rows:
            by_step[row.proposal_raw_step].append(row)
        runtime = make_a4_episode_runtime(
            task,
            instance,
            environment_seed,
            width=int(width),
            height=int(height),
            benchmark_seed=int(benchmark_seed),
        )
        try:
            observation = np.asarray(runtime.observation, dtype=np.float32)
            info: Mapping[str, object] = runtime.reset_info
            for source_step in range(max(by_step) + 1):
                source_action = _bounded_expert_action(runtime, observation)
                for row in by_step.get(source_step, ()):
                    snapshot = capture_branch_snapshot(runtime.env, include_process_rng=True)
                    try:
                        result = _execute_anchor_branch(
                            runtime,
                            row,
                            observation=observation,
                            info=info,
                            width=width,
                            height=height,
                        )
                    finally:
                        restore_branch_snapshot(runtime.env, snapshot)
                    if row.cache_row_index in results:
                        raise RuntimeError("duplicate anchor-target replay row")
                    results[row.cache_row_index] = result
                if source_step == max(by_step):
                    break
                observation, _reward, terminated, truncated, info = runtime.env.step(source_action)
                if terminated or truncated:
                    raise RuntimeError("source episode ended before an accepted proposal")
                observation = np.asarray(observation, dtype=np.float32)
        finally:
            _close_episode_env_safely(runtime.env)
        print(json.dumps({"task": task, "source_episode_id": episode_id, "proposals": len(episode_rows)}, sort_keys=True), flush=True)
    expected = {row.cache_row_index for row in rows}
    if set(results) != expected:
        raise RuntimeError("anchor-target replay did not cover every selected row")
    return results


def stack_anchor_targets(
    rows: Sequence[ReplayRow],
    results: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    *,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    ordered = sorted(rows, key=lambda row: row.cache_row_index)
    raw = np.stack([results[row.cache_row_index][0] for row in ordered]).astype(np.float32)
    display = np.stack([results[row.cache_row_index][1] for row in ordered]).astype(np.float32)
    valid = np.stack([results[row.cache_row_index][2] for row in ordered]).astype(bool)
    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "proposal_id": np.asarray([row.proposal_id for row in ordered]),
        "cache_row_index": np.asarray([row.cache_row_index for row in ordered], dtype=np.int64),
        "split": np.asarray([row.split for row in ordered]),
        "task_name": np.asarray([row.task_name for row in ordered]),
        "task_instance_index": np.asarray([row.task_instance_index for row in ordered], dtype=np.int64),
        "source_episode_id": np.asarray([row.source_episode_id for row in ordered]),
        "environment_seed": np.asarray([row.environment_seed for row in ordered], dtype=np.int64),
        "proposal_raw_step": np.asarray([row.proposal_raw_step for row in ordered], dtype=np.int64),
        "expert_slot": np.asarray([row.expert_slot for row in ordered], dtype=np.int64),
        "image_width": np.asarray(int(width), dtype=np.int64),
        "image_height": np.asarray(int(height), dtype=np.int64),
        "display_flip_y": np.asarray([row.task_name not in CUSTOM_TASKS for row in ordered], dtype=bool),
        "raw_step_offsets": RAW_STEP_OFFSETS.copy(),
        "model_step_raw_offsets": MODEL_STEP_OFFSETS.copy(),
        "q_anchor_raw_coords_raw_steps": raw,
        "q_anchor_display_coords_raw_steps": display,
        "q_anchor_model_raw": raw[:, MODEL_STEP_OFFSETS],
        "q_anchor_display": display[:, MODEL_STEP_OFFSETS],
        "all_anchors_valid_raw_steps": valid,
        "all_anchors_valid_model_steps": valid[:, MODEL_STEP_OFFSETS],
        "physical_validation_max_abs": np.asarray([results[row.cache_row_index][3] for row in ordered], dtype=np.float64),
        "reserved_test_opened": np.asarray(False, dtype=bool),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-source-episodes", type=int)
    parser.add_argument("--max-proposals", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = _load_json(args.config)
    validate_a4_contract(contract)
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
    arrays = stack_anchor_targets(
        rows,
        collect_anchor_targets(
            rows,
            width=int(collection["width"]),
            height=int(collection["height"]),
            benchmark_seed=int(collection["benchmark_seed"]),
        ),
        width=int(collection["width"]),
        height=int(collection["height"]),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "output": str(output),
        "rows": int(len(rows)),
        "splits": sorted(set(arrays["split"].astype(str).tolist())),
        "anchor_count": 3,
        "reserved_test_opened": False,
        "coordinate_contract": "raw_projection_and_upright_display_projection",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
