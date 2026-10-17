#!/usr/bin/env python3
"""Collect Pick action branches with commanded planar mobile-base motion."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.geometry import Camera, project_points  # noqa: E402
from fcpg.lewm_action_branch_a4 import BRANCH_COUNT, HORIZON  # noqa: E402
from fcpg.lewm_action_branch_a4_collection import (  # noqa: E402
    _bounded_expert_action,
    make_a4_episode_runtime,
)
from fcpg.lewm_counterfactual_runtime import (  # noqa: E402
    capture_branch_snapshot,
    restore_branch_snapshot,
)
from fcpg.lewm_mobile_base import (  # noqa: E402
    BASE_PREFIX_DIM,
    PlanarBaseTracker,
    apply_planar_base_translation,
    replace_base_displacement_with_odometry,
)
from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402
from scripts.collect_lewm_action_prefix_joint_a4 import (  # noqa: E402
    DEFAULT_CORRECTOR,
    RAW_STEPS_PER_MODEL_STEP,
    _allocate,
    _npz,
    _record_arrays,
)


DEFAULT_SOURCE = ROOT / "outputs/lewm_action_prefix_a4/joint_raw_24train_8dev/index.npz"
DEFAULT_OUTPUT = ROOT / "outputs/lewm_mobile_base/pick_exact_24train_8dev"
SLIP_MATRIX = np.asarray([[0.26, -0.08], [0.05, 0.20]], dtype=np.float64)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corrector", type=Path, default=DEFAULT_CORRECTOR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--benchmark-seed", type=int, default=30103)
    parser.add_argument("--base-speed-m", type=float, default=0.0012)
    parser.add_argument(
        "--geometry-mode",
        choices=("exact", "odometry"),
        default="exact",
        help="Use known base pose for the main pilot or odometry only as a diagnostic.",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("nominal", "fcpg"),
        default="nominal",
        help="Keep clean training exact; FCPG is evaluated under camera drift later.",
    )
    return parser.parse_args()


def build_base_action_bank(
    count: int,
    *,
    seed: int,
    speed_m: float,
) -> np.ndarray:
    """Build smooth eight-direction base prefixes with per-bundle rotation."""

    if count <= 0 or not np.isfinite(speed_m) or speed_m <= 0.0:
        raise ValueError("base action bank count and speed must be positive")
    rng = np.random.default_rng(int(seed))
    result = np.empty(
        (count, BRANCH_COUNT, HORIZON, BASE_PREFIX_DIM), dtype=np.float32
    )
    raw_count = HORIZON * RAW_STEPS_PER_MODEL_STEP
    for bundle in range(count):
        rotation = float(rng.uniform(-np.pi, np.pi))
        branch_order = rng.permutation(BRANCH_COUNT)
        for public_branch, direction_index in enumerate(branch_order.tolist()):
            angle = rotation + 2.0 * np.pi * float(direction_index) / BRANCH_COUNT
            rows = []
            for raw_step in range(raw_count):
                curve = 0.20 * np.sin(2.0 * np.pi * raw_step / max(raw_count - 1, 1))
                local_angle = angle + curve
                magnitude = speed_m * (0.85 + 0.15 * np.cos(np.pi * raw_step / raw_count))
                rows.append(
                    [magnitude * np.cos(local_angle), magnitude * np.sin(local_angle)]
                )
            result[bundle, public_branch] = np.asarray(rows, dtype=np.float32).reshape(
                HORIZON, BASE_PREFIX_DIM
            )
    return result


def _odometry_record(
    record: Any,
    *,
    transition: Any,
    camera: Camera,
) -> Any:
    xyz = replace_base_displacement_with_odometry(
        record.keypoints_xyz,
        true_base_displacement_xy=transition.true_position_xy,
        odometry_base_displacement_xy=transition.odometry_position_xy,
    )
    q_nominal, valid_nominal = project_points(xyz, camera)
    return replace(
        record,
        keypoints_xyz=np.asarray(xyz, dtype=np.float32),
        q_nominal=np.asarray(q_nominal, dtype=np.float32),
        valid_nominal=np.asarray(valid_nominal, dtype=bool),
    )


def _collect_bundle(
    row: Mapping[str, Any],
    *,
    base_action: np.ndarray,
    width: int,
    height: int,
    benchmark_seed: int,
    geometry_mode: str,
) -> tuple[Any, list[list[Any]], np.ndarray, np.ndarray]:
    runtime = make_a4_episode_runtime(
        "pick-place-v3",
        int(row["task_instance"]),
        int(row["environment_seed"]),
        width=int(width),
        height=int(height),
        benchmark_seed=int(benchmark_seed),
        camera_name="corner",
    )
    observation = np.asarray(runtime.observation, dtype=np.float32)
    try:
        proposal_raw_step = int(row["proposal_raw_step"])
        for raw_step in range(proposal_raw_step + 1):
            if raw_step == proposal_raw_step:
                current = runtime.record_current(observation)
                snapshot = capture_branch_snapshot(runtime.env, include_process_rng=True)
                break
            action = _bounded_expert_action(runtime, observation)
            observation, _reward, terminated, truncated, info = runtime.env.step(action)
            if terminated or truncated or not isinstance(info, Mapping):
                raise RuntimeError("source replay ended before the Pick proposal")
            observation = np.asarray(observation, dtype=np.float32)
        else:  # pragma: no cover - guarded by loop bounds
            raise RuntimeError("Pick proposal was not reached")

        arm_chunks = np.asarray(row["future_action"], dtype=np.float32)
        base_chunks = np.asarray(base_action, dtype=np.float32)
        if arm_chunks.shape != (BRANCH_COUNT, HORIZON, 20):
            raise ValueError("arm future_action must have shape [8,8,20]")
        if base_chunks.shape != (BRANCH_COUNT, HORIZON, BASE_PREFIX_DIM):
            raise ValueError("base future_action must have shape [8,8,10]")
        arm_raw = arm_chunks.reshape(BRANCH_COUNT, HORIZON * 5, 4)
        base_raw = base_chunks.reshape(BRANCH_COUNT, HORIZON * 5, 2)
        camera = Camera(
            np.asarray(runtime.camera_intrinsic, dtype=np.float64),
            np.asarray(runtime.camera_extrinsic_cam_from_world, dtype=np.float64),
            int(width),
            int(height),
        )
        branch_records: list[list[Any]] = []
        true_positions = np.empty((BRANCH_COUNT, HORIZON, 2), dtype=np.float32)
        odometry_positions = np.empty_like(true_positions)
        for branch_index in range(BRANCH_COUNT):
            restore_branch_snapshot(runtime.env, snapshot)
            tracker = PlanarBaseTracker()
            records = []
            for raw_index, (arm_action, base_command) in enumerate(
                zip(arm_raw[branch_index], base_raw[branch_index], strict=True),
                start=1,
            ):
                slip = SLIP_MATRIX @ np.asarray(base_command, dtype=np.float64)
                transition = tracker.step(base_command, slip_xy=slip)
                apply_planar_base_translation(runtime.env, transition.true_delta_xy)
                next_observation, _reward, terminated, truncated, info = runtime.env.step(
                    arm_action
                )
                if terminated or truncated or not isinstance(info, Mapping):
                    raise RuntimeError("mobile-base Pick branch ended before H8")
                if raw_index % RAW_STEPS_PER_MODEL_STEP == 0:
                    horizon = raw_index // RAW_STEPS_PER_MODEL_STEP - 1
                    record = runtime.record_current(
                        np.asarray(next_observation, dtype=np.float32)
                    )
                    if geometry_mode == "exact":
                        records.append(record)
                    elif geometry_mode == "odometry":
                        records.append(
                            _odometry_record(record, transition=transition, camera=camera)
                        )
                    else:
                        raise ValueError(f"unknown geometry mode {geometry_mode!r}")
                    true_positions[branch_index, horizon] = transition.true_position_xy
                    odometry_positions[branch_index, horizon] = (
                        transition.odometry_position_xy
                    )
            if len(records) != HORIZON:
                raise RuntimeError("mobile-base Pick branch did not produce H1-H8")
            branch_records.append(records)
        return current, branch_records, true_positions, odometry_positions
    finally:
        runtime.env.close()


def _source_rows(path: Path, limit: int | None) -> dict[str, np.ndarray]:
    source = _npz(path)
    task = np.asarray(source["task_name"]).astype(str)
    selected = np.flatnonzero(task == "pick-place-v3")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[: int(limit)]
    if selected.size == 0:
        raise ValueError("source index has no Pick rows")
    return {
        name: np.ascontiguousarray(np.asarray(value)[selected])
        for name, value in source.items()
    }


def main() -> int:
    args = _parser()
    if args.device not in {"cpu", "cuda:0"}:
        raise ValueError("mobile-base collection may use CPU or cuda:0, never cuda:1")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")
    index = _source_rows(args.source_index.expanduser().resolve(), args.limit)
    count = int(index["task_name"].shape[0])
    base_action = build_base_action_bank(
        count,
        seed=int(args.benchmark_seed) + 77,
        speed_m=float(args.base_speed_m),
    )
    index["future_base_action"] = base_action
    index["mobile_action_dim"] = np.full(count, 30, dtype=np.int64)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.npz"
    if index_path.exists():
        existing = _npz(index_path)
        if existing.keys() != index.keys() or any(
            not np.array_equal(existing[name], value) for name, value in index.items()
        ):
            raise ValueError("existing mobile-base index differs from requested data")
    else:
        np.savez(index_path, **index)

    shape_future = (count, BRANCH_COUNT, HORIZON)
    arrays = {
        "current_rgb": _allocate(output / "current_rgb.npy", (count, 3, args.height, args.width), np.uint8),
        "current_proprio": _allocate(output / "current_proprio.npy", (count, 4), np.float32),
        "current_q_nominal": _allocate(output / "current_q_nominal.npy", (count, 3, 2), np.float32),
        "current_q_corrected": _allocate(output / "current_q_corrected.npy", (count, 3, 2), np.float32),
        "current_keypoints_xyz": _allocate(output / "current_keypoints_xyz.npy", (count, 3, 3), np.float32),
        "current_valid": _allocate(output / "current_valid.npy", (count, 3), np.bool_),
        "current_q_true": _allocate(output / "current_q_true.npy", (count, 3, 2), np.float32),
        "future_rgb": _allocate(output / "future_rgb.npy", (*shape_future, 3, args.height, args.width), np.uint8),
        "future_proprio": _allocate(output / "future_proprio.npy", (*shape_future, 4), np.float32),
        "future_q_nominal": _allocate(output / "future_q_nominal.npy", (*shape_future, 3, 2), np.float32),
        "future_q_corrected": _allocate(output / "future_q_corrected.npy", (*shape_future, 3, 2), np.float32),
        "future_keypoints_xyz": _allocate(output / "future_keypoints_xyz.npy", (*shape_future, 3, 3), np.float32),
        "future_valid": _allocate(output / "future_valid.npy", (*shape_future, 3), np.bool_),
        "future_q_true": _allocate(output / "future_q_true.npy", (*shape_future, 3, 2), np.float32),
        "future_base_true_xy": _allocate(output / "future_base_true_xy.npy", (*shape_future, 2), np.float32),
        "future_base_odometry_xy": _allocate(output / "future_base_odometry_xy.npy", (*shape_future, 2), np.float32),
        "completed": _allocate(output / "completed.npy", (count,), np.bool_),
    }
    device = torch.device(str(args.device))
    corrector = None
    if args.grounding_mode == "fcpg":
        corrector, _ = load_fcpg_vit_adapter(
            args.corrector.expanduser().resolve(), device
        )
        corrector.requires_grad_(False).eval()
    started = time.perf_counter()
    for output_index in range(count):
        if bool(arrays["completed"][output_index]):
            continue
        row = {name: value[output_index] for name, value in index.items()}
        current, branches, true_xy, odometry_xy = _collect_bundle(
            row,
            base_action=base_action[output_index],
            width=int(args.width),
            height=int(args.height),
            benchmark_seed=int(args.benchmark_seed),
            geometry_mode=str(args.geometry_mode),
        )
        values = _record_arrays(
            current,
            branches,
            corrector=corrector,
            device=device,
            grounding_mode=str(args.grounding_mode),
        )
        for name, value in values.items():
            arrays[name][output_index] = value
        arrays["future_base_true_xy"][output_index] = true_xy
        arrays["future_base_odometry_xy"][output_index] = odometry_xy
        arrays["completed"][output_index] = True
        for array in arrays.values():
            array.flush()
        print(
            json.dumps(
                {
                    "completed": int(np.count_nonzero(arrays["completed"])),
                    "total": count,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    manifest = {
        "protocol": "mobile_base_pick_action_prefix_v1",
        "status": "complete",
        "count": count,
        "task": "pick-place-v3",
        "width": int(args.width),
        "height": int(args.height),
        "arm_prefix_dim": 20,
        "base_prefix_dim": BASE_PREFIX_DIM,
        "mobile_prefix_dim": 30,
        "base_speed_m_per_raw_step": float(args.base_speed_m),
        "slip_matrix": SLIP_MATRIX.tolist(),
        "geometry_mode": str(args.geometry_mode),
        "grounding_mode": str(args.grounding_mode),
        "corrector": (
            str(args.corrector.expanduser().resolve())
            if args.grounding_mode == "fcpg"
            else None
        ),
        "source_index": str(args.source_index.expanduser().resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
